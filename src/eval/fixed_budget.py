"""Fixed-budget 4-method baseline harness (`direct_sql`, `exec_feedback`, `evodw_seed`, `evodw_gepa_transfer`)
with unified Executable vs Caliber-Correct telemetry and cross-schema zero-leakage verification."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Literal

from src.agent.generator import clean_sql_text
from src.agent.pipeline import MultiModuleSQLAgent
from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory, ModuleInvocation
from src.benchmarks.types import BenchmarkTask
from src.config import PROJECT_ROOT
from src.env.sandbox import DialectType, WarehouseSandbox
from src.env.schema import WarehouseCatalog
from src.eval.manifest_audit import wilson_confidence_interval
from src.llm.client import BaseLLMClient

BaselineMethod = Literal["direct_sql", "exec_feedback", "evodw_seed", "evodw_gepa_transfer"]
ALL_BASELINE_METHODS: tuple[BaselineMethod, ...] = (
    "direct_sql",
    "exec_feedback",
    "evodw_seed",
    "evodw_gepa_transfer",
)

DIRECT_SQL_SYSTEM_PROMPT = (
    "You are an enterprise data warehouse Text-to-SQL generator.\n"
    "Given the target SQL dialect, full warehouse DDL schemas, business question, and hint, "
    "produce a single read-only SELECT query.\n"
    'Output strictly valid JSON with exact keys: {"thought": "step-by-step plan", "sql": "SELECT ..."}.'
)

EXEC_FEEDBACK_REPAIR_SYSTEM_PROMPT = (
    "You are GatewayRefiner, an execution-feedback SQL repair module.\n"
    "Given the warehouse DDL schemas, business question, previous SQL, and sandbox execution error feedback, "
    "diagnose the error and output a corrected single read-only SELECT query.\n"
    'Output strictly valid JSON with exact keys: {"thought": "...", "diagnosis": "...", "sql": "SELECT ..."}.'
)

_EXTERNAL_OPAQUE_ENUMS: tuple[str, ...] = (
    "POPLATEK MESICNE",
    "POPLATEK TYDNE",
    "POPLATEK PO OBRATU",
    "PRIJEM",
    "VYDAJ",
    "PAT_SPECKLED",
    "PAT_HOMOG",
    "REF_ER",
    "REF_TRANSFER",
    "FUND_DIRECT",
    "FUND_LOCAL",
)


def classify_trajectory_verdict(traj: AgentTrajectory) -> str:
    """Classify trajectory into mutually exclusive Executable vs Result-Correct (EX) outcomes."""
    executable = bool(traj.final_gateway_passed and traj.final_error_stage == "none")
    if not executable:
        if not traj.final_gateway_passed or traj.final_error_stage == "gateway_ast":
            return "gateway_ast_block"
        return "runtime_error"
    if traj.ex_match:
        return "executable_and_correct"
    row_cnt = int(traj.final_exec.get("row_count", 0) or 0)
    if row_cnt == 0:
        return "empty_result"
    return "silent_caliber_error"


def resolve_transfer_gepa_policy(
    dialect: DialectType = "odps",
    project_root: Path = PROJECT_ROOT,
) -> Policy:
    """Load the canonical e-commerce-evolved GEPA policy matching the target SQL dialect."""
    rel = (
        "outputs/policies/hard_v2_hive/gepa_best_policy.json"
        if dialect == "hive"
        else "outputs/policies/hard_v2_odps/gepa_best_policy.json"
    )
    path = project_root / rel
    if not path.exists():
        raise FileNotFoundError(f"Canonical GEPA policy not found at {path}")
    return Policy.load_json(path)


def verify_zero_cross_schema_leakage(
    policy: Policy,
    external_catalog: WarehouseCatalog,
) -> dict[str, Any]:
    """Verify transferred policy prompts contain zero external or e-commerce table/enum/task-ID leakage."""
    from src.benchmarks.catalog_builder import build_warehouse_catalog

    combined_low = "\n".join(
        [policy.linker_prompt, policy.generator_prompt, policy.refiner_prompt]
    ).lower()

    ext_tables = [t.name for t in external_catalog.list_tables()]
    ecom_tables = [t.name for t in build_warehouse_catalog().list_tables()]

    ext_hits = [t for t in ext_tables if t.lower() in combined_low]
    ecom_hits = [t for t in ecom_tables if t.lower() in combined_low]
    enum_hits = [e for e in _EXTERNAL_OPAQUE_ENUMS if e.lower() in combined_low]
    tid_hits = [p for p in ("ext_fin_", "ext_clin_", "ext_edu_") if p in combined_low]

    passed = not (ext_hits or ecom_hits or enum_hits or tid_hits)
    return {
        "passed": passed,
        "policy_id": policy.policy_id,
        "external_table_hits": ext_hits,
        "ecommerce_table_hits": ecom_hits,
        "external_enum_hits": enum_hits,
        "external_task_id_hits": tid_hits,
    }


async def run_direct_or_exec_feedback_task(
    task: BenchmarkTask,
    *,
    llm: BaseLLMClient,
    sandbox: WarehouseSandbox,
    dialect: DialectType,
    allow_exec_feedback: bool,
    max_llm_calls: int = 4,
) -> AgentTrajectory:
    """Execute 1-shot `direct_sql` or multi-turn `exec_feedback` under an explicit call cap."""
    all_table_names = [t.name for t in sandbox.catalog.list_tables()]
    full_ddl = sandbox.catalog.format_table_ddl(all_table_names)

    gen_user_msg = (
        f"Target Dialect: {dialect.upper()}\n"
        f"Business Question: {task.question}\n"
        f"Business Hint: {task.hint or 'None'}\n\n"
        f"### Full Warehouse DDL Schemas\n{full_ddl}\n\n"
        'Generate the single read-only SQL query and return strict JSON: {"thought": "...", "sql": "..."}.'
    )
    gen_messages = [
        {"role": "system", "content": DIRECT_SQL_SYSTEM_PROMPT},
        {"role": "user", "content": gen_user_msg},
    ]
    resp0 = await llm.chat_json(gen_messages, role_type="actor", temperature=0.0)
    parsed0 = resp0.parsed_json or {}
    initial_sql = clean_sql_text(str(parsed0.get("sql", "")))

    first_exec = sandbox.run_sql(initial_sql, dialect=dialect)
    first_pass_gw = bool(first_exec["gateway_passed"])

    module_calls: list[ModuleInvocation] = [
        ModuleInvocation(
            module_name="generator",
            step_index=0,
            messages=gen_messages,
            raw_output=resp0.content,
            parsed_json=parsed0,
            prompt_tokens=resp0.prompt_tokens,
            completion_tokens=resp0.completion_tokens,
            total_tokens=resp0.total_tokens,
            latency_ms=resp0.latency_ms,
        )
    ]

    current_sql = initial_sql
    current_exec = first_exec
    refine_steps: list[dict[str, Any]] = []

    if allow_exec_feedback:
        max_repairs = max(0, max_llm_calls - 1)
        for r_turn in range(1, max_repairs + 1):
            suspicious, reason = sandbox.is_suspicious_result(current_exec)
            if not suspicious:
                break
            feedback_payload = {
                "gateway_passed": current_exec.get("gateway_passed", False),
                "error_stage": current_exec.get("error_stage", "none"),
                "error_msg": current_exec.get("error_msg", ""),
                "suspicion_reason": reason,
                "row_count": current_exec.get("row_count", 0),
                "sample_rows": [str(r) for r in (current_exec.get("rows") or [])[:5]],
            }
            ref_user_msg = (
                f"Target Dialect: {dialect.upper()}\n"
                f"Business Question: {task.question}\n"
                f"Business Hint: {task.hint or 'None'}\n\n"
                f"### Full Warehouse DDL Schemas\n{full_ddl}\n\n"
                f"Previous SQL:\n{current_sql}\n\n"
                f"Sandbox Execution Feedback:\n{json.dumps(feedback_payload, ensure_ascii=False, indent=2)}\n\n"
                'Diagnose the root cause and output strict JSON: {"thought": "...", "diagnosis": "...", "sql": "..."}.'
            )
            ref_messages = [
                {"role": "system", "content": EXEC_FEEDBACK_REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": ref_user_msg},
            ]
            ref_resp = await llm.chat_json(ref_messages, role_type="actor", temperature=0.0)
            ref_parsed = ref_resp.parsed_json or {}
            repaired_sql = clean_sql_text(str(ref_parsed.get("sql", ""))) or current_sql
            new_exec = sandbox.run_sql(repaired_sql, dialect=dialect)

            module_calls.append(
                ModuleInvocation(
                    module_name="refiner",
                    step_index=len(module_calls),
                    messages=ref_messages,
                    raw_output=ref_resp.content,
                    parsed_json=ref_parsed,
                    prompt_tokens=ref_resp.prompt_tokens,
                    completion_tokens=ref_resp.completion_tokens,
                    total_tokens=ref_resp.total_tokens,
                    latency_ms=ref_resp.latency_ms,
                )
            )
            refine_steps.append(
                {
                    "turn": r_turn,
                    "trigger_reason": reason,
                    "thought": str(ref_parsed.get("thought", "")).strip(),
                    "diagnosis": str(ref_parsed.get("diagnosis", "")).strip(),
                    "sql": repaired_sql,
                    "exec_res": new_exec,
                }
            )
            current_sql = repaired_sql
            current_exec = new_exec

    eval_outcome = sandbox.evaluate_sql(current_sql, task, dialect=dialect)
    reward_info = sandbox.compute_reward(
        current_sql,
        task,
        refine_turns=len(refine_steps),
        first_pass_gateway=first_pass_gw,
        dialect=dialect,
    )
    return AgentTrajectory(
        task_id=task.task_id,
        category=task.category,
        split=task.split,
        question=task.question,
        hint=task.hint,
        dialect=dialect,
        policy_id="exec_feedback_v0" if allow_exec_feedback else "direct_sql_v0",
        linked_tables=all_table_names,
        probe_requests=[],
        probe_results=[],
        linked_schema_context=full_ddl,
        linker_notes="",
        initial_sql=initial_sql,
        first_pass_gateway=first_pass_gw,
        first_pass_exec=first_exec,
        refine_steps=refine_steps,
        final_sql=current_sql,
        final_exec=current_exec,
        eval_outcome=eval_outcome,
        reward_info=reward_info,
        module_calls=module_calls,
        conversation_turns=[],
    )


def summarize_method_telemetry(
    method_name: str,
    trajectories: list[AgentTrajectory],
    tasks: list[BenchmarkTask],
    max_llm_calls: int,
) -> dict[str, Any]:
    """Compute unified fixed-budget telemetry separating Executable SQL Rate from Silent Caliber Errors."""
    n = len(trajectories)
    if n == 0:
        return {"method": method_name, "total": 0}

    task_domain_map = {
        t.task_id: str((t.metadata or {}).get("domain") or t.category)
        for t in tasks
    }

    verdicts = [classify_trajectory_verdict(t) for t in trajectories]
    verdict_counts = Counter(verdicts)

    ex_hits = sum(1 for t in trajectories if t.ex_match)
    exec_hits = sum(
        1 for t in trajectories if t.final_gateway_passed and t.final_error_stage == "none"
    )
    gw1_hits = sum(1 for t in trajectories if t.first_pass_gateway)
    silent_hits = verdict_counts.get("silent_caliber_error", 0)
    max_observed_calls = max(t.num_llm_steps for t in trajectories)
    probe_cnt = sum(1 for t in trajectories if len(t.probe_requests) > 0)

    ci_lo, ci_hi = wilson_confidence_interval(ex_hits, n)

    by_domain: dict[str, dict[str, Any]] = {}
    grouped_dom: dict[str, list[AgentTrajectory]] = defaultdict(list)
    for t in trajectories:
        grouped_dom[task_domain_map.get(t.task_id, "default")].append(t)

    for dom in sorted(grouped_dom):
        items = grouped_dom[dom]
        dn = len(items)
        d_ex = sum(1 for x in items if x.ex_match)
        d_exec = sum(1 for x in items if x.final_gateway_passed and x.final_error_stage == "none")
        d_sil = sum(1 for x in items if classify_trajectory_verdict(x) == "silent_caliber_error")
        by_domain[dom] = {
            "count": dn,
            "ex_hits": d_ex,
            "ex": round(d_ex / dn, 4),
            "executable_sql_rate": round(d_exec / dn, 4),
            "silent_caliber_error_rate": round(d_sil / dn, 4),
        }

    return {
        "method": method_name,
        "total": n,
        "max_llm_calls_budget": max_llm_calls,
        "max_observed_llm_calls": max_observed_calls,
        "budget_compliant": max_observed_calls <= max_llm_calls,
        "ex_hits": ex_hits,
        "ex": round(ex_hits / n, 4),
        "ex_wilson_95_ci": [ci_lo, ci_hi],
        "executable_sql_rate": round(exec_hits / n, 4),
        "invalid_sql_rate": round(1.0 - (exec_hits / n), 4),
        "first_pass_gateway_rate": round(gw1_hits / n, 4),
        "silent_caliber_error_count": silent_hits,
        "silent_caliber_error_rate": round(silent_hits / n, 4),
        "probed_task_rate": round(probe_cnt / n, 4),
        "avg_llm_calls": round(sum(t.num_llm_steps for t in trajectories) / n, 2),
        "avg_tokens": round(sum(t.total_tokens for t in trajectories) / n, 1),
        "avg_latency_ms": round(sum(t.total_latency_ms for t in trajectories) / n, 2),
        "failure_taxonomy": {
            "gateway_ast_block": verdict_counts.get("gateway_ast_block", 0),
            "runtime_error": verdict_counts.get("runtime_error", 0),
            "empty_result": verdict_counts.get("empty_result", 0),
            "silent_caliber_error": silent_hits,
        },
        "by_domain": by_domain,
    }


async def compare_methods_fixed_budget(
    *,
    tasks: list[BenchmarkTask],
    sandbox: WarehouseSandbox,
    llm: BaseLLMClient,
    dialect: DialectType = "odps",
    methods: tuple[BaselineMethod, ...] = ALL_BASELINE_METHODS,
    max_llm_calls_per_task: int = 4,
    seed_prompts_dir: Path | None = None,
    transfer_policy: Policy | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, list[AgentTrajectory]]]:
    """Evaluate all selected methods under an identical per-task LLM call cap and unified telemetry."""
    prompts_dir = seed_prompts_dir or (project_root / "prompts/seed")
    seed_policy = Policy.load_seed(prompts_dir)
    gepa_policy = transfer_policy or resolve_transfer_gepa_policy(
        dialect=dialect, project_root=project_root
    )

    leakage_audit = verify_zero_cross_schema_leakage(gepa_policy, sandbox.catalog)
    agent = MultiModuleSQLAgent(
        llm=llm,
        sandbox=sandbox,
        max_refine_turns=max(0, max_llm_calls_per_task - 2),
        max_total_llm_calls=max_llm_calls_per_task,
    )

    trajectories_by_method: dict[str, list[AgentTrajectory]] = {}
    summaries_by_method: dict[str, dict[str, Any]] = {}

    for method in methods:
        trajs: list[AgentTrajectory] = []
        for task in tasks:
            if method == "direct_sql":
                traj = await run_direct_or_exec_feedback_task(
                    task,
                    llm=llm,
                    sandbox=sandbox,
                    dialect=dialect,
                    allow_exec_feedback=False,
                    max_llm_calls=1,
                )
            elif method == "exec_feedback":
                traj = await run_direct_or_exec_feedback_task(
                    task,
                    llm=llm,
                    sandbox=sandbox,
                    dialect=dialect,
                    allow_exec_feedback=True,
                    max_llm_calls=max_llm_calls_per_task,
                )
            elif method == "evodw_seed":
                traj = await agent.run_task(task, policy=seed_policy, dialect=dialect, evaluate=True)
            elif method == "evodw_gepa_transfer":
                traj = await agent.run_task(task, policy=gepa_policy, dialect=dialect, evaluate=True)
            else:
                raise ValueError(f"Unsupported baseline method: {method!r}")
            trajs.append(traj)

        trajectories_by_method[method] = trajs
        summaries_by_method[method] = summarize_method_telemetry(
            method_name=method,
            trajectories=trajs,
            tasks=tasks,
            max_llm_calls=max_llm_calls_per_task,
        )

    payload = {
        "dialect": dialect,
        "task_count": len(tasks),
        "max_llm_calls_per_task": max_llm_calls_per_task,
        "transferred_policy_id": gepa_policy.policy_id,
        "zero_cross_schema_leakage_audit": leakage_audit,
        "methods": summaries_by_method,
    }
    return payload, trajectories_by_method
