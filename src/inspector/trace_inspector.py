"""Query & Trajectory Inspector (`evodw inspect`) separating Gateway-Executable SQL from Caliber-Correct (EX) results."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import difflib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.generator import clean_sql_text
from src.agent.pipeline import MultiModuleSQLAgent
from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory
from src.benchmarks import load_benchmark
from src.benchmarks.types import BenchmarkTask
from src.config import PROJECT_ROOT, Settings
from src.env.sandbox import DialectType, WarehouseSandbox
from src.eval.fixed_budget import resolve_transfer_gepa_policy
from src.llm import DualRoleLLMClient, MissingAPIKeyError, MockWarehouseLLMClient
from src.llm.client import JSONParseError, extract_json_object

_ERROR_CODE_RE = re.compile(r"\b(ODPS-\d{7}|HIVE-\d{4,6})\b")


@dataclass
class InspectedTrace:
    """Normalized multi-module SQL trace with explicit Dual Verdict (Executable vs Caliber-Correct)."""

    trace_id: str
    suite: str
    task_id: str
    category: str
    split: str
    dialect: str
    policy_id: str
    question: str
    hint: str
    linker: dict[str, Any]
    generator_and_gateway: dict[str, Any]
    refinement: dict[str, Any]
    dual_verdict: dict[str, Any]
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_error_code(error_msg: str) -> str:
    m = _ERROR_CODE_RE.search(error_msg or "")
    return m.group(1) if m else ""


def _sql_unified_diff(sql_before: str, sql_after: str) -> str:
    lines = list(
        difflib.unified_diff(
            (sql_before or "").strip().splitlines(),
            (sql_after or "").strip().splitlines(),
            fromfile="sql_before",
            tofile="sql_after",
            lineterm="",
        )
    )
    return "\n".join(lines) if lines else "(unchanged)"


def _summarize_probed_facts(probe_results: list[dict[str, Any]]) -> list[str]:
    facts: list[str] = []
    for item in probe_results or []:
        if not isinstance(item, dict) or not item.get("found"):
            continue
        tname = item.get("table_name", "unknown")
        pinfo = item.get("partition_info") or {}
        pcol = pinfo.get("partition_col")
        if pcol:
            facts.append(
                f"{tname}: partition_col={pcol}, max_partition={pinfo.get('max_partition')} "
                f"(count={pinfo.get('partition_count')})"
            )
        else:
            facts.append(f"{tname}: unpartitioned dimension table")

        for cname, prof in (item.get("column_profiles") or {}).items():
            if not isinstance(prof, dict):
                continue
            vals = prof.get("sample_non_empty_values")
            num_r = prof.get("numeric_range")
            if vals and any(k in cname.lower() for k in ("code", "status", "channel", "type", "pattern", "freq", "flag")):
                facts.append(f"{tname}.{cname} enums={vals[:5]}")
            elif isinstance(num_r, dict) and any(
                k in cname.lower() for k in ("cent", "bps", "amt", "yuan", "grant", "charge", "plt")
            ):
                facts.append(
                    f"{tname}.{cname} range=[{num_r.get('min')}, {num_r.get('max')}] (avg={num_r.get('avg')})"
                )
    return facts


def _build_dual_verdict(
    *,
    executable: bool,
    result_correct: bool,
    refine_turns: int,
    final_error_stage: str,
    final_error_msg: str,
    final_row_count: int,
    gold_row_count: int,
    pred_sample_rows: list[str],
    gold_sample_rows: list[str],
    gold_sql: str,
    diff_reason: str,
) -> dict[str, Any]:
    if not executable:
        verdict_code = "gateway_or_syntax_block"
        verdict_title = "网关/语法拦截 (Gateway/Syntax Block: Executable=FAIL)"
    elif result_correct:
        if refine_turns > 0:
            verdict_code = "self_repaired_pass"
            verdict_title = "多轮自修复通过 (Self-Repaired Correct: Executable=PASS & EX=PASS)"
        else:
            verdict_code = "first_pass_pass"
            verdict_title = "首轮口径全对 (First-Pass Correct: Executable=PASS & EX=PASS)"
    elif final_row_count == 0:
        verdict_code = "empty_result_mismatch"
        verdict_title = "空结果失配 (Empty Result Mismatch: Executable=PASS & 0 Rows)"
    else:
        verdict_code = "silent_caliber_trap"
        verdict_title = "静默口径陷阱 (Silent Caliber Trap: Executable=PASS & EX=FAIL)"

    return {
        "executable": executable,
        "executable_badge": f"语法与网关可执行 (Executable): {'PASS' if executable else 'FAIL'}",
        "result_correct": result_correct,
        "result_correct_badge": f"业务口径结果正确 (Result Correct / EX): {'PASS' if result_correct else 'FAIL'}",
        "is_silent_caliber_trap": verdict_code == "silent_caliber_trap",
        "verdict_code": verdict_code,
        "verdict_title": verdict_title,
        "final_error_stage": final_error_stage,
        "final_error_code": _extract_error_code(final_error_msg),
        "final_error_msg": final_error_msg,
        "final_row_count": final_row_count,
        "gold_row_count": gold_row_count,
        "pred_sample_rows": pred_sample_rows,
        "gold_sample_rows": gold_sample_rows,
        "gold_sql": gold_sql,
        "diff_reason": diff_reason,
    }


def inspect_agent_trajectory(
    traj: AgentTrajectory,
    *,
    task: BenchmarkTask,
    sandbox: WarehouseSandbox,
    suite: str,
) -> InspectedTrace:
    """Convert an executed `AgentTrajectory` into a structured `InspectedTrace`."""
    dialect: DialectType = "hive" if str(traj.dialect).lower() == "hive" else "odps"
    candidate_tables = [t.name for t in sandbox.catalog.list_tables()]

    ev = traj.eval_outcome or sandbox.evaluate_sql(traj.final_sql, task, dialect=dialect)
    final_exec = traj.final_exec or sandbox.run_sql(traj.final_sql, dialect=dialect)
    first_exec = traj.first_pass_exec or sandbox.run_sql(traj.initial_sql, dialect=dialect)

    rounds: list[dict[str, Any]] = []
    prev_sql = traj.initial_sql
    for step in traj.refine_steps:
        trig = str(step.get("trigger_reason", ""))
        if "POST_EXECUTION_SANITY_AUDIT" in trig:
            trig_type = "post_execution_sanity_audit"
        elif "empty result" in trig.lower() or "0 rows" in trig.lower():
            trig_type = "empty_result"
        else:
            trig_type = "gateway_or_runtime"
        new_sql = str(step.get("sql", ""))
        ex_r = step.get("exec_res") or {}
        rounds.append(
            {
                "turn": int(step.get("turn", len(rounds) + 1)),
                "trigger_type": trig_type,
                "trigger_reason": trig,
                "diagnosis": str(step.get("diagnosis", "")),
                "sql_before": prev_sql,
                "sql_after": new_sql,
                "unified_diff": _sql_unified_diff(prev_sql, new_sql),
                "post_gateway_passed": bool(ex_r.get("gateway_passed", False)),
                "post_row_count": int(ex_r.get("row_count", 0) or 0),
            }
        )
        prev_sql = new_sql

    executable = bool(final_exec.get("gateway_passed", False) and final_exec.get("error_stage", "none") == "none")
    dual_verdict = _build_dual_verdict(
        executable=executable,
        result_correct=bool(ev.ex_match),
        refine_turns=len(rounds),
        final_error_stage=str(final_exec.get("error_stage", "none")),
        final_error_msg=str(final_exec.get("error_msg", "")),
        final_row_count=int(final_exec.get("row_count", len(ev.pred_rows)) or 0),
        gold_row_count=len(ev.gold_rows),
        pred_sample_rows=[str(r) for r in ev.pred_rows[:5]],
        gold_sample_rows=[str(r) for r in ev.gold_rows[:5]],
        gold_sql=task.gold_sql,
        diff_reason=ev.diff_reason,
    )

    return InspectedTrace(
        trace_id=f"{suite}:{traj.policy_id}:{traj.task_id}",
        suite=suite,
        task_id=traj.task_id,
        category=traj.category,
        split=traj.split,
        dialect=dialect,
        policy_id=traj.policy_id,
        question=traj.question,
        hint=traj.hint,
        linker={
            "candidate_table_count": len(candidate_tables),
            "candidate_tables": candidate_tables,
            "selected_tables": list(traj.linked_tables),
            "probe_requests": list(traj.probe_requests),
            "discovered_facts": _summarize_probed_facts(traj.probe_results),
            "schema_notes": traj.linker_notes,
        },
        generator_and_gateway={
            "initial_sql": traj.initial_sql,
            "first_pass_gateway_passed": bool(traj.first_pass_gateway),
            "first_pass_error_stage": str(first_exec.get("error_stage", "none")),
            "first_pass_error_code": _extract_error_code(str(first_exec.get("error_msg", ""))),
            "first_pass_error_msg": str(first_exec.get("error_msg", "")),
            "first_pass_row_count": int(first_exec.get("row_count", 0) or 0),
            "final_sql": traj.final_sql,
            "transpiled_duckdb_sql": str(final_exec.get("transpiled_sql") or ev.transpiled_sql or ""),
        },
        refinement={
            "refine_turns": len(rounds),
            "rounds": rounds,
        },
        dual_verdict=dual_verdict,
        telemetry={
            "num_llm_steps": traj.num_llm_steps,
            "total_tokens": traj.total_tokens,
            "latency_ms": round(traj.total_latency_ms, 2),
        },
    )


def inspect_sft_jsonl_record(
    record: dict[str, Any],
    *,
    task: BenchmarkTask,
    sandbox: WarehouseSandbox,
    suite: str,
) -> InspectedTrace:
    """Reconstruct and live-verify a saved multi-turn trajectory from `sft_openai_tool_use.jsonl`."""
    dialect: DialectType = "hive" if str(record.get("dialect", "odps")).lower() == "hive" else "odps"
    msgs = record.get("messages") or []
    candidate_tables = [t.name for t in sandbox.catalog.list_tables()]

    selected_tables: list[str] = []
    probe_requests: list[dict[str, Any]] = []
    probe_results: list[dict[str, Any]] = []
    schema_notes = ""
    initial_sql = ""
    rounds: list[dict[str, Any]] = []
    pending_feedback: dict[str, Any] | None = None

    for m in msgs:
        role = m.get("role")
        mod = m.get("module")
        content = str(m.get("content") or "")
        if role == "assistant" and mod == "linker":
            try:
                obj = extract_json_object(content)
                if str(obj.get("action", "")).lower() == "probe":
                    probe_requests = list(obj.get("probe_requests") or [])
                elif str(obj.get("action", "")).lower() == "finalize":
                    selected_tables = [str(x) for x in (obj.get("selected_tables") or [])]
                    schema_notes = str(obj.get("schema_notes") or "")
            except JSONParseError:
                pass
        elif role == "tool" and m.get("name") == "inspect_table_sample":
            try:
                loaded = json.loads(content)
                if isinstance(loaded, list):
                    probe_results = loaded
            except Exception:
                pass
        elif role == "assistant" and mod == "generator":
            try:
                obj = extract_json_object(content)
                initial_sql = clean_sql_text(str(obj.get("sql") or ""))
            except JSONParseError:
                pass
        elif role == "tool" and mod == "refiner":
            try:
                pending_feedback = json.loads(content)
            except Exception:
                pending_feedback = {"suspicion_reason": content}
        elif role == "assistant" and mod == "refiner":
            try:
                obj = extract_json_object(content)
                rep_sql = clean_sql_text(str(obj.get("sql") or ""))
                diag = str(obj.get("diagnosis") or "")
            except JSONParseError:
                rep_sql = initial_sql
                diag = ""
            prev_sql = rounds[-1]["sql_after"] if rounds else initial_sql
            trig = str((pending_feedback or {}).get("suspicion_reason") or (pending_feedback or {}).get("error_msg") or "")
            trig_type = (
                "post_execution_sanity_audit"
                if "POST_EXECUTION_SANITY_AUDIT" in trig
                else ("empty_result" if "empty" in trig.lower() else "gateway_or_runtime")
            )
            rep_exec = sandbox.run_sql(rep_sql, dialect=dialect)
            rounds.append(
                {
                    "turn": len(rounds) + 1,
                    "trigger_type": trig_type,
                    "trigger_reason": trig,
                    "diagnosis": diag,
                    "sql_before": prev_sql,
                    "sql_after": rep_sql,
                    "unified_diff": _sql_unified_diff(prev_sql, rep_sql),
                    "post_gateway_passed": bool(rep_exec.get("gateway_passed", False)),
                    "post_row_count": int(rep_exec.get("row_count", 0) or 0),
                }
            )
            pending_feedback = None

    final_sql = rounds[-1]["sql_after"] if rounds else initial_sql
    first_exec = sandbox.run_sql(initial_sql, dialect=dialect) if initial_sql else {"gateway_passed": False, "error_stage": "gateway_ast", "error_msg": "Empty SQL", "row_count": 0}
    final_exec = sandbox.run_sql(final_sql, dialect=dialect) if final_sql else first_exec
    ev = sandbox.evaluate_sql(final_sql, task, dialect=dialect)

    executable = bool(final_exec.get("gateway_passed", False) and final_exec.get("error_stage", "none") == "none")
    dual_verdict = _build_dual_verdict(
        executable=executable,
        result_correct=bool(ev.ex_match),
        refine_turns=len(rounds),
        final_error_stage=str(final_exec.get("error_stage", "none")),
        final_error_msg=str(final_exec.get("error_msg", "")),
        final_row_count=int(final_exec.get("row_count", len(ev.pred_rows)) or 0),
        gold_row_count=len(ev.gold_rows),
        pred_sample_rows=[str(r) for r in ev.pred_rows[:5]],
        gold_sample_rows=[str(r) for r in ev.gold_rows[:5]],
        gold_sql=task.gold_sql,
        diff_reason=ev.diff_reason,
    )

    rec_id = str(record.get("id") or f"{suite}:{record.get('policy_id')}:{task.task_id}")
    return InspectedTrace(
        trace_id=rec_id,
        suite=suite,
        task_id=task.task_id,
        category=task.category,
        split=task.split,
        dialect=dialect,
        policy_id=str(record.get("policy_id", "exported")),
        question=task.question,
        hint=task.hint,
        linker={
            "candidate_table_count": len(candidate_tables),
            "candidate_tables": candidate_tables,
            "selected_tables": selected_tables,
            "probe_requests": probe_requests,
            "discovered_facts": _summarize_probed_facts(probe_results),
            "schema_notes": schema_notes,
        },
        generator_and_gateway={
            "initial_sql": initial_sql,
            "first_pass_gateway_passed": bool(first_exec.get("gateway_passed", False)),
            "first_pass_error_stage": str(first_exec.get("error_stage", "none")),
            "first_pass_error_code": _extract_error_code(str(first_exec.get("error_msg", ""))),
            "first_pass_error_msg": str(first_exec.get("error_msg", "")),
            "first_pass_row_count": int(first_exec.get("row_count", 0) or 0),
            "final_sql": final_sql,
            "transpiled_duckdb_sql": str(final_exec.get("transpiled_sql") or ev.transpiled_sql or ""),
        },
        refinement={
            "refine_turns": len(rounds),
            "rounds": rounds,
        },
        dual_verdict=dual_verdict,
        telemetry={
            "num_llm_steps": int(record.get("num_llm_steps", 0)),
            "total_tokens": int(record.get("total_tokens", 0)),
            "latency_ms": round(float(record.get("latency_ms", 0.0)), 2),
        },
    )


def summarize_inspected_traces(traces: list[InspectedTrace]) -> dict[str, Any]:
    """Aggregate Executable vs Caliber-Correct gap and verdict breakdown across inspected traces."""
    n = len(traces)
    if n == 0:
        return {
            "total_traces": 0,
            "executable_count": 0,
            "executable_rate": 0.0,
            "result_correct_count": 0,
            "result_correct_rate": 0.0,
            "silent_caliber_trap_count": 0,
            "silent_caliber_trap_rate": 0.0,
            "gateway_or_syntax_block_count": 0,
            "by_verdict_code": {},
        }

    exec_cnt = sum(1 for t in traces if t.dual_verdict["executable"])
    ex_cnt = sum(1 for t in traces if t.dual_verdict["result_correct"])
    silent_cnt = sum(1 for t in traces if t.dual_verdict["is_silent_caliber_trap"])
    gw_cnt = sum(1 for t in traces if t.dual_verdict["verdict_code"] == "gateway_or_syntax_block")
    by_code = Counter(t.dual_verdict["verdict_code"] for t in traces)

    return {
        "total_traces": n,
        "executable_count": exec_cnt,
        "executable_rate": round(exec_cnt / n, 4),
        "result_correct_count": ex_cnt,
        "result_correct_rate": round(ex_cnt / n, 4),
        "executable_minus_correct_gap": round((exec_cnt - ex_cnt) / n, 4),
        "silent_caliber_trap_count": silent_cnt,
        "silent_caliber_trap_rate": round(silent_cnt / n, 4),
        "gateway_or_syntax_block_count": gw_cnt,
        "by_verdict_code": dict(sorted(by_code.items())),
    }


async def build_showcase_traces(
    *,
    suite: str | None = None,
    task_ids: list[str] | None = None,
    policy_mode: str = "both",
    dialect: DialectType = "odps",
    use_mock: bool = True,
    from_jsonl: Path | None = None,
    limit: int = 12,
    project_root: Path = PROJECT_ROOT,
) -> list[InspectedTrace]:
    """Build inspected traces either from saved JSONL trajectories or side-by-side Seed vs GEPA runs."""
    settings = Settings.from_env(dialect=dialect)

    if from_jsonl is not None:
        bm_core = load_benchmark(seed=2026, dialect=dialect, precompute_gold=True, suite="all")
        try:
            v1_ids = {t.task_id for t in load_benchmark(seed=2026, dialect=dialect, precompute_gold=False, suite="v1").all_tasks}
            tmap = {t.task_id: t for t in bm_core.all_tasks}
            out: list[InspectedTrace] = []
            for raw_line in from_jsonl.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                rec = json.loads(raw_line)
                tid = str(rec.get("task_id", ""))
                if task_ids and tid not in task_ids:
                    continue
                if tid not in tmap:
                    continue
                s_name = "v1" if tid in v1_ids else "hard_v2"
                if suite and suite != "all" and s_name != suite:
                    continue
                out.append(inspect_sft_jsonl_record(rec, task=tmap[tid], sandbox=bm_core.sandbox, suite=s_name))
                if len(out) >= limit:
                    break
            return out
        finally:
            bm_core.sandbox.close()

    seed_policy = Policy.load_seed(project_root / "prompts/seed")
    gepa_policy = resolve_transfer_gepa_policy(dialect=dialect, project_root=project_root)

    if use_mock:
        llm: Any = MockWarehouseLLMClient(dialect=dialect)
    else:
        llm = DualRoleLLMClient(settings=settings)

    policies_to_run: list[Policy] = (
        [seed_policy]
        if policy_mode == "seed"
        else ([gepa_policy] if policy_mode == "gepa" else [seed_policy, gepa_policy])
    )

    try:
        if suite in ("v1", "hard_v2", "external_bird_mini"):
            suite_targets: list[tuple[str, tuple[str, ...]]] = [
                (
                    suite,
                    tuple(task_ids)
                    if task_ids
                    else (
                        ("ja_03", "pp_01")
                        if suite == "v1"
                        else (("sc_01", "scd_01") if suite == "hard_v2" else ("ext_fin_03", "ext_clin_03", "ext_edu_05"))
                    ),
                )
            ]
        else:
            # Default multi-suite diagnostic showcase covering Gateway Blocks, Silent Caliber Traps,
            # Self-Repairs, and Probed First-Pass Correctness across v1, Hard-v2, and external_bird_mini!
            suite_targets = [
                ("v1", tuple(task_ids) if task_ids else ("ja_03", "pp_01")),
                ("hard_v2", tuple(task_ids) if task_ids else ("sc_01", "scd_01")),
                ("external_bird_mini", tuple(task_ids) if task_ids else ("ext_fin_03", "ext_clin_03")),
            ]

        collected: list[InspectedTrace] = []

        # Also include 2 real live-model trajectories (1 probe + 1 self-repair) from sft_openai_tool_use.jsonl when available!
        sft_disk = project_root / "outputs/rl_export/sft_openai_tool_use.jsonl"
        if suite is None and not task_ids and sft_disk.exists():
            bm_real = load_benchmark(seed=2026, dialect=dialect, precompute_gold=True, suite="all")
            try:
                tmap_real = {t.task_id: t for t in bm_real.all_tasks}
                found_probe = False
                found_repair = False
                for line in sft_disk.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    tid = str(rec.get("task_id", ""))
                    if tid not in tmap_real:
                        continue
                    has_p = any(m.get("name") == "inspect_table_sample" for m in (rec.get("messages") or []))
                    is_rep = rec.get("trajectory_type") == "self_repaired_after_error"
                    s_lbl = "hard_v2" if tid.startswith(("sl_", "scd_", "sc_")) else "v1"
                    if has_p and not found_probe:
                        collected.append(inspect_sft_jsonl_record(rec, task=tmap_real[tid], sandbox=bm_real.sandbox, suite=s_lbl))
                        found_probe = True
                    elif is_rep and not found_repair:
                        collected.append(inspect_sft_jsonl_record(rec, task=tmap_real[tid], sandbox=bm_real.sandbox, suite=s_lbl))
                        found_repair = True
                    if found_probe and found_repair:
                        break
            finally:
                bm_real.sandbox.close()

        for s_name, target_tids in suite_targets:
            bm = load_benchmark(seed=2026, dialect=dialect, precompute_gold=True, suite=s_name)  # type: ignore[arg-type]
            try:
                agent = MultiModuleSQLAgent(llm=llm, sandbox=bm.sandbox, max_refine_turns=2)
                selected = [t for t in bm.all_tasks if t.task_id in target_tids]
                for task in selected:
                    for pol in policies_to_run:
                        traj = await agent.run_task(task, policy=pol, dialect=dialect, evaluate=True)
                        collected.append(
                            inspect_agent_trajectory(traj, task=task, sandbox=bm.sandbox, suite=s_name)
                        )
                        if len(collected) >= limit:
                            return collected
            finally:
                bm.sandbox.close()

        return collected[:limit]
    finally:
        if isinstance(llm, DualRoleLLMClient):
            await llm.close()


def render_inspector_markdown(traces: list[InspectedTrace]) -> str:
    """Render concise Markdown summary separating Gateway-Executable from Caliber-Correct SQL."""
    summary = summarize_inspected_traces(traces)
    lines: list[str] = [
        "# EvoDW-RSI SQL Trace Inspector Summary (Executable vs Caliber-Correct)",
        "",
        f"- Total Traces: `{summary['total_traces']}`",
        f"- **语法与网关可执行 (Executable)**: `{summary['executable_count']}/{summary['total_traces']}` (`{summary['executable_rate'] * 100:.1f}%`)",
        f"- **业务口径结果正确 (Result Correct / EX)**: `{summary['result_correct_count']}/{summary['total_traces']}` (`{summary['result_correct_rate'] * 100:.1f}%`)",
        f"- **静默口径陷阱 (Silent Caliber Traps: Executable=PASS & EX=FAIL)**: `{summary['silent_caliber_trap_count']}` (`{summary['silent_caliber_trap_rate'] * 100:.1f}%`)",
        f"- **网关/语法拦截 (Gateway/Syntax Blocks: Executable=FAIL)**: `{summary['gateway_or_syntax_block_count']}`",
        "",
        "| Trace ID | Suite | Policy | Executable | Result Correct (EX) | Verdict Classification | Probes | Refines |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for t in traces:
        dv = t.dual_verdict
        lines.append(
            f"| `{t.trace_id}` | `{t.suite}` | `{t.policy_id}` | "
            f"`{'PASS' if dv['executable'] else 'FAIL'}` | "
            f"`{'PASS' if dv['result_correct'] else 'FAIL'}` | "
            f"{dv['verdict_title']} | {len(t.linker['probe_requests'])} | {t.refinement['refine_turns']} |"
        )
    lines.append("")

    for t in traces:
        dv = t.dual_verdict
        gw = t.generator_and_gateway
        lines.append(f"## `{t.trace_id}` — {dv['verdict_title']}")
        lines.append(f"- **业务问题**: {t.question}")
        lines.append(f"- **双维判定**: `{dv['executable_badge']}` | `{dv['result_correct_badge']}`")
        lines.append(
            f"- **1. Schema Linking & Probing**: 候选 `{t.linker['candidate_table_count']}` 表 -> 选中 `{t.linker['selected_tables']}` | "
            f"探查事实: `{t.linker['discovered_facts'] or '无探查'}`"
        )
        lines.append(
            f"- **2. 首轮网关校验**: `{'PASS' if gw['first_pass_gateway_passed'] else 'BLOCKED'}` "
            f"{gw['first_pass_error_msg'] or ''}"
        )
        lines.append("```sql\n" + (gw["final_sql"] or "(empty)") + "\n```")
        if t.refinement["rounds"]:
            for r in t.refinement["rounds"]:
                lines.append(f"- **3. Refiner Turn {r['turn']} (`{r['trigger_type']}`)**: {r['diagnosis']}")
                lines.append("```diff\n" + r["unified_diff"] + "\n```")
        lines.append(
            f"- **4. 结果集核对**: Pred Rows=`{dv['final_row_count']}` `{dv['pred_sample_rows'][:2]}` vs "
            f"Gold Rows=`{dv['gold_row_count']}` `{dv['gold_sample_rows'][:2]}` (`{dv['diff_reason']}`)"
        )
        lines.append("")
    return "\n".join(lines)


def render_inspector_html(traces: list[InspectedTrace]) -> str:
    """Render a self-contained, zero-CDN interactive HTML workbench separating Executable from EX."""
    summary = summarize_inspected_traces(traces)
    payload = {
        "summary": summary,
        "traces": [t.to_dict() for t in traces],
    }
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoDW-RSI SQL Trace Inspector — Executable vs Caliber-Correct</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --pass-bg: #064e3b;
    --pass-fg: #6ee7b7;
    --fail-bg: #7f1d1d;
    --fail-fg: #fca5a5;
    --trap-bg: #78350f;
    --trap-fg: #fcd34d;
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  header {{ max-width: 1320px; margin: 0 auto 20px auto; }}
  h1 {{ margin: 0 0 6px 0; font-size: 22px; color: #f8fafc; }}
  .subtitle {{ color: var(--muted); font-size: 14px; }}
  .kpi-grid {{
    max-width: 1320px; margin: 0 auto 20px auto;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px;
  }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  }}
  .kpi .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .kpi.trap {{ border-color: #d97706; background: rgba(120, 53, 15, 0.28); }}
  .controls {{
    max-width: 1320px; margin: 0 auto 18px auto;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px;
  }}
  select, input {{
    background: #0f172a; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 11px; font-size: 13px;
  }}
  input {{ flex: 1; min-width: 240px; }}
  .trace-list {{ max-width: 1320px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px;
  }}
  .card.silent_caliber_trap {{ border-left: 5px solid #f59e0b; }}
  .card.gateway_or_syntax_block {{ border-left: 5px solid #ef4444; }}
  .card.self_repaired_pass, .card.first_pass_pass {{ border-left: 5px solid #10b981; }}
  .card-header {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 10px; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  }}
  .badge.pass {{ background: var(--pass-bg); color: var(--pass-fg); border: 1px solid #059669; }}
  .badge.fail {{ background: var(--fail-bg); color: var(--fail-fg); border: 1px solid #dc2626; }}
  .badge.trap {{ background: var(--trap-bg); color: var(--trap-fg); border: 1px solid #d97706; }}
  .badge.meta {{ background: #0f172a; color: var(--accent); border: 1px solid var(--border); }}
  .question {{ font-size: 14px; margin: 8px 0 12px 0; color: #e2e8f0; }}
  .stage-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }}
  @media (max-width: 900px) {{ .stage-grid {{ grid-template-columns: 1fr; }} }}
  .box {{ background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 13px; }}
  .box h4 {{ margin: 0 0 8px 0; font-size: 13px; color: var(--accent); }}
  pre {{
    margin: 6px 0 0 0; padding: 10px; background: #020617; border: 1px solid #1e293b;
    border-radius: 6px; overflow-x: auto; font-size: 12px; color: #e2e8f0; white-space: pre-wrap;
  }}
  ul.facts {{ margin: 4px 0 0 18px; padding: 0; color: #cbd5e1; }}
</style>
</head>
<body>
<header>
  <h1>EvoDW-RSI SQL &amp; Multi-Module Trajectory Inspector</h1>
  <div class="subtitle">Explicitly decoupling <strong>语法与网关可执行 (Gateway-Executable)</strong> from <strong>业务口径结果正确 (Caliber-Correct / EX)</strong> to expose silent warehouse caliber traps.</div>
</header>

<section class="kpi-grid">
  <div class="kpi">
    <div class="label">Inspected Traces</div>
    <div class="value">{summary['total_traces']}</div>
  </div>
  <div class="kpi">
    <div class="label">语法与网关可执行 (Executable)</div>
    <div class="value">{summary['executable_count']}/{summary['total_traces']} ({summary['executable_rate'] * 100:.1f}%)</div>
  </div>
  <div class="kpi">
    <div class="label">业务口径结果正确 (EX)</div>
    <div class="value">{summary['result_correct_count']}/{summary['total_traces']} ({summary['result_correct_rate'] * 100:.1f}%)</div>
  </div>
  <div class="kpi trap">
    <div class="label">静默口径陷阱 (Exec=PASS, EX=FAIL)</div>
    <div class="value">{summary['silent_caliber_trap_count']} ({summary['silent_caliber_trap_rate'] * 100:.1f}%)</div>
  </div>
  <div class="kpi">
    <div class="label">网关/语法拦截 (Exec=FAIL)</div>
    <div class="value">{summary['gateway_or_syntax_block_count']}</div>
  </div>
</section>

<section class="controls">
  <select id="verdictFilter">
    <option value="all">All Dual-Verdict Categories</option>
    <option value="silent_caliber_trap">Silent Caliber Traps (Executable=PASS &amp; EX=FAIL)</option>
    <option value="gateway_or_syntax_block">Gateway / Syntax Blocks (Executable=FAIL)</option>
    <option value="self_repaired_pass">Self-Repaired Correct</option>
    <option value="first_pass_pass">First-Pass Correct</option>
  </select>
  <select id="suiteFilter">
    <option value="all">All Suites (v1 / hard_v2 / external_bird_mini)</option>
    <option value="v1">v1</option>
    <option value="hard_v2">hard_v2</option>
    <option value="external_bird_mini">external_bird_mini</option>
  </select>
  <input id="searchBox" type="search" placeholder="Filter by task_id, table name, ODPS error code, SQL keyword...">
</section>

<main id="traceContainer" class="trace-list"></main>

<script type="application/json" id="trace-payload">{safe_json}</script>
<script>
(function() {{
  const data = JSON.parse(document.getElementById('trace-payload').textContent);
  const container = document.getElementById('traceContainer');
  const verdictSel = document.getElementById('verdictFilter');
  const suiteSel = document.getElementById('suiteFilter');
  const searchBox = document.getElementById('searchBox');

  function esc(s) {{
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }}

  function render() {{
    const vFilter = verdictSel.value;
    const sFilter = suiteSel.value;
    const q = searchBox.value.trim().toLowerCase();
    const filtered = data.traces.filter(t => {{
      if (vFilter !== 'all' && t.dual_verdict.verdict_code !== vFilter) return false;
      if (sFilter !== 'all' && t.suite !== sFilter) return false;
      if (q && !JSON.stringify(t).toLowerCase().includes(q)) return false;
      return true;
    }});

    container.innerHTML = filtered.map(t => {{
      const dv = t.dual_verdict;
      const gw = t.generator_and_gateway;
      const lk = t.linker;
      const execCls = dv.executable ? 'pass' : 'fail';
      const exCls = dv.result_correct ? 'pass' : 'fail';
      const catCls = dv.is_silent_caliber_trap ? 'trap' : (dv.result_correct ? 'pass' : 'fail');
      const factsHtml = (lk.discovered_facts && lk.discovered_facts.length)
        ? '<ul class="facts">' + lk.discovered_facts.map(f => '<li>' + esc(f) + '</li>').join('') + '</ul>'
        : '<div style="color:#94a3b8">No inspect_table_sample probe issued.</div>';
      const roundsHtml = (t.refinement.rounds && t.refinement.rounds.length)
        ? t.refinement.rounds.map(r => (
            '<div style="margin-top:8px"><strong>Turn ' + r.turn + ' [' + esc(r.trigger_type) + ']:</strong> '
            + esc(r.diagnosis) + '<pre>' + esc(r.unified_diff) + '</pre></div>'
          )).join('')
        : '<div style="color:#94a3b8">0 refinement rounds executed.</div>';

      return '<article class="card ' + esc(dv.verdict_code) + '">'
        + '<div class="card-header">'
        +   '<div class="badges">'
        +     '<span class="badge meta">' + esc(t.suite) + ' · ' + esc(t.task_id) + ' · policy=' + esc(t.policy_id) + ' · ' + esc(t.dialect.toUpperCase()) + '</span>'
        +     '<span class="badge ' + catCls + '">' + esc(dv.verdict_title) + '</span>'
        +   '</div>'
        +   '<div class="badges">'
        +     '<span class="badge ' + execCls + '">' + esc(dv.executable_badge) + '</span>'
        +     '<span class="badge ' + exCls + '">' + esc(dv.result_correct_badge) + '</span>'
        +   '</div>'
        + '</div>'
        + '<div class="question"><strong>Question:</strong> ' + esc(t.question) + (t.hint ? '<br><span style="color:#94a3b8">Hint: ' + esc(t.hint) + '</span>' : '') + '</div>'
        + '<div class="stage-grid">'
        +   '<div class="box"><h4>1. Schema Linking &amp; Sample Probing (WarehouseLinker)</h4>'
        +     '<div>Catalog (' + lk.candidate_table_count + ' tables) &rarr; Selected: <code>' + esc((lk.selected_tables || []).join(', ')) + '</code></div>'
        +     '<div style="margin-top:6px"><strong>Probed Facts:</strong>' + factsHtml + '</div>'
        +     (lk.schema_notes ? '<div style="margin-top:6px;color:#cbd5e1"><strong>Notes:</strong> ' + esc(lk.schema_notes) + '</div>' : '')
        +   '</div>'
        +   '<div class="box"><h4>2. Dialect SQL &amp; AST Gateway Check (Generator + GatewayLinter)</h4>'
        +     '<div>1st-Pass Gateway: <strong>' + (gw.first_pass_gateway_passed ? 'PASS' : 'BLOCKED (' + esc(gw.first_pass_error_code || gw.first_pass_error_stage) + ')') + '</strong></div>'
        +     (gw.first_pass_error_msg ? '<div style="color:#fca5a5;margin-top:4px">' + esc(gw.first_pass_error_msg) + '</div>' : '')
        +     '<pre>' + esc(gw.final_sql) + '</pre>'
        +   '</div>'
        +   '<div class="box"><h4>3. Refinement &amp; Post-Execution Sanity Audit (GatewayRefiner)</h4>' + roundsHtml + '</div>'
        +   '<div class="box"><h4>4. Dual-Verdict Result Comparison vs Gold</h4>'
        +     '<div><strong>Diff Verdict:</strong> ' + esc(dv.diff_reason) + '</div>'
        +     '<div style="margin-top:4px">Pred (' + dv.final_row_count + ' rows): <code>' + esc((dv.pred_sample_rows || []).slice(0, 3).join(' | ')) + '</code></div>'
        +     '<div style="margin-top:4px">Gold (' + dv.gold_row_count + ' rows): <code>' + esc((dv.gold_sample_rows || []).slice(0, 3).join(' | ')) + '</code></div>'
        +     '<pre>' + esc(dv.gold_sql) + '</pre>'
        +   '</div>'
        + '</div>'
        + '</article>';
    }}).join('');
  }}

  verdictSel.addEventListener('change', render);
  suiteSel.addEventListener('change', render);
  searchBox.addEventListener('input', render);
  render();
}})();
</script>
</body>
</html>
"""


def create_inspector_http_server(
    traces: list[InspectedTrace],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a local HTTP server serving `/` (HTML workbench) and `/api/traces.json`."""
    html_bytes = render_inspector_html(traces).encode("utf-8")
    json_bytes = json.dumps(
        {
            "summary": summarize_inspected_traces(traces),
            "traces": [t.to_dict() for t in traces],
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    class _InspectorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path_clean = self.path.split("?", 1)[0]
            if path_clean in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif path_clean == "/api/traces.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(json_bytes)))
                self.end_headers()
                self.wfile.write(json_bytes)
            else:
                msg = html.escape(f"Not Found: {path_clean}").encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    return ThreadingHTTPServer((host, port), _InspectorHandler)


def serve_inspector_http(
    traces: list[InspectedTrace],
    host: str = "127.0.0.1",
    port: int = 8765,
    console: Console | None = None,
) -> None:
    """Serve the interactive SQL Trace Inspector locally until Ctrl+C."""
    server = create_inspector_http_server(traces, host=host, port=port)
    actual_port = server.server_address[1]
    con = console or Console()
    con.print(
        Panel.fit(
            f"[bold green]EvoDW-RSI SQL Trace Inspector Live Workbench[/bold green]\n"
            f"URL: http://{host}:{actual_port}/\n"
            f"JSON API: http://{host}:{actual_port}/api/traces.json\n"
            "Press Ctrl+C to stop."
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _render_console_summary(console: Console, traces: list[InspectedTrace]) -> None:
    summary = summarize_inspected_traces(traces)
    console.print(
        Panel(
            f"[bold cyan]EvoDW-RSI SQL Trace Inspector (Executable vs Caliber-Correct)[/bold cyan]\n"
            f"Traces: `{summary['total_traces']}` | "
            f"语法与网关可执行 (Executable): [green]{summary['executable_count']}/{summary['total_traces']} ({summary['executable_rate'] * 100:.1f}%)[/green] | "
            f"业务口径结果正确 (EX): [bold green]{summary['result_correct_count']}/{summary['total_traces']} ({summary['result_correct_rate'] * 100:.1f}%)[/bold green]\n"
            f"[yellow]静默口径陷阱 (Executable=PASS & EX=FAIL): {summary['silent_caliber_trap_count']} ({summary['silent_caliber_trap_rate'] * 100:.1f}%)[/yellow] | "
            f"[red]网关/语法拦截 (Executable=FAIL): {summary['gateway_or_syntax_block_count']}[/red]"
        )
    )

    tbl = Table(title="Inspected Query Dual-Verdict Matrix", header_style="bold cyan")
    tbl.add_column("Suite / Task")
    tbl.add_column("Policy")
    tbl.add_column("Probes", justify="right")
    tbl.add_column("Refines", justify="right")
    tbl.add_column("语法与网关可执行")
    tbl.add_column("业务口径正确 (EX)")
    tbl.add_column("Diagnostic Classification")

    for t in traces:
        dv = t.dual_verdict
        exec_cell = "[green]PASS[/green]" if dv["executable"] else f"[red]FAIL ({dv['final_error_code'] or dv['final_error_stage']})[/red]"
        ex_cell = "[green]PASS[/green]" if dv["result_correct"] else "[red]FAIL[/red]"
        v_style = "yellow" if dv["is_silent_caliber_trap"] else ("green" if dv["result_correct"] else "red")
        tbl.add_row(
            f"{t.suite}:{t.task_id}",
            t.policy_id,
            str(len(t.linker["probe_requests"])),
            str(t.refinement["refine_turns"]),
            exec_cell,
            ex_cell,
            f"[{v_style}]{dv['verdict_title']}[/{v_style}]",
        )
    console.print(tbl)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evodw inspect",
        description="Inspect multi-module SQL traces separating Gateway-Executable from Caliber-Correct (EX).",
    )
    p.add_argument(
        "--suite",
        choices=["v1", "hard_v2", "external_bird_mini"],
        default=None,
        help="Optional single suite filter (default showcases v1 + hard_v2 + external_bird_mini).",
    )
    p.add_argument("--tasks", nargs="+", default=None, help="Optional specific task_id(s) to inspect.")
    p.add_argument(
        "--policy",
        choices=["both", "seed", "gepa"],
        default="both",
        help="Run seed_v0, evolved gepa, or both side-by-side (default: both).",
    )
    p.add_argument("--dialect", choices=["odps", "hive"], default="odps")
    p.add_argument("--from-jsonl", type=Path, default=None, help="Inspect trajectories from an SFT JSONL file.")
    p.add_argument("--limit", type=int, default=14, help="Maximum traces to inspect.")
    p.add_argument("--html-out", type=Path, default=None, help="Write self-contained zero-CDN HTML workbench to path.")
    p.add_argument("--md-out", type=Path, default=None, help="Write Markdown trace report to path.")
    p.add_argument("--json-out", type=Path, default=None, help="Write structured JSON trace report to path.")
    p.add_argument("--serve", action="store_true", help="Launch local HTTP server for interactive inspection.")
    p.add_argument("--host", default="127.0.0.1", help="Bind host for --serve (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8765, help="Bind port for --serve (default: 8765).")
    p.add_argument("--mock", action="store_true", help="Force deterministic offline sandbox execution.")
    p.add_argument("--live", action="store_true", help="Require live OpenAI-compatible API endpoint.")
    return p


async def _run_cli_async(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.from_env(dialect=args.dialect)
    use_mock = bool(args.mock) or (not getattr(args, "live", False) and not settings.has_usable_api_key)

    try:
        traces = await build_showcase_traces(
            suite=args.suite,
            task_ids=list(args.tasks) if args.tasks else None,
            policy_mode=args.policy,
            dialect=args.dialect,
            use_mock=use_mock,
            from_jsonl=args.from_jsonl,
            limit=args.limit,
        )
    except MissingAPIKeyError as exc:
        console.print(f"[bold yellow]Configuration required:[/bold yellow] {exc}")
        return 2

    _render_console_summary(console, traces)

    if args.html_out is not None:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(render_inspector_html(traces), encoding="utf-8")
        console.print(f"[green]Wrote zero-CDN HTML workbench -> {args.html_out}[/green]")

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_inspector_markdown(traces), encoding="utf-8")
        console.print(f"[green]Wrote Markdown trace summary -> {args.md_out}[/green]")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summarize_inspected_traces(traces),
            "traces": [t.to_dict() for t in traces],
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Wrote JSON trace report -> {args.json_out}[/green]")

    if args.serve:
        serve_inspector_http(traces, host=args.host, port=args.port, console=console)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_run_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
