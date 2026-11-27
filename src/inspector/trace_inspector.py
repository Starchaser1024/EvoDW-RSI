"""Query & Trajectory Inspector (`evodw inspect`) separating Gateway-Executable SQL from Caliber-Correct (EX) results."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import difflib
import json
from pathlib import Path
import re
from typing import Any

from src.agent.generator import clean_sql_text
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.types import BenchmarkTask
from src.env.sandbox import DialectType, WarehouseSandbox
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


def _assemble_trace(
    *,
    trace_id: str,
    suite: str,
    task: BenchmarkTask,
    dialect: str,
    policy_id: str,
    candidate_tables: list[str],
    selected_tables: list[str],
    probe_requests: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    schema_notes: str,
    initial_sql: str,
    final_sql: str,
    first_exec: dict[str, Any],
    final_exec: dict[str, Any],
    ev: Any,
    rounds: list[dict[str, Any]],
    telemetry: dict[str, Any],
) -> InspectedTrace:
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
        trace_id=trace_id,
        suite=suite,
        task_id=task.task_id,
        category=task.category,
        split=task.split,
        dialect=dialect,
        policy_id=policy_id,
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
        refinement={"refine_turns": len(rounds), "rounds": rounds},
        dual_verdict=dual_verdict,
        telemetry=telemetry,
    )


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
    first_exec_view = {**first_exec, "gateway_passed": bool(traj.first_pass_gateway)}

    rounds: list[dict[str, Any]] = []
    prev_sql = traj.initial_sql
    for step in traj.refine_steps:
        trig = str(step.get("trigger_reason", ""))
        trig_type = (
            "post_execution_sanity_audit"
            if "POST_EXECUTION_SANITY_AUDIT" in trig
            else ("empty_result" if ("empty result" in trig.lower() or "0 rows" in trig.lower()) else "gateway_or_runtime")
        )
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

    return _assemble_trace(
        trace_id=f"{suite}:{traj.policy_id}:{traj.task_id}",
        suite=suite,
        task=task,
        dialect=dialect,
        policy_id=traj.policy_id,
        candidate_tables=candidate_tables,
        selected_tables=list(traj.linked_tables),
        probe_requests=list(traj.probe_requests),
        probe_results=traj.probe_results,
        schema_notes=traj.linker_notes,
        initial_sql=traj.initial_sql,
        final_sql=traj.final_sql,
        first_exec=first_exec_view,
        final_exec=final_exec,
        ev=ev,
        rounds=rounds,
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
    first_exec = (
        sandbox.run_sql(initial_sql, dialect=dialect)
        if initial_sql
        else {"gateway_passed": False, "error_stage": "gateway_ast", "error_msg": "Empty SQL", "row_count": 0}
    )
    final_exec = sandbox.run_sql(final_sql, dialect=dialect) if final_sql else first_exec
    ev = sandbox.evaluate_sql(final_sql, task, dialect=dialect)

    rec_id = str(record.get("id") or f"{suite}:{record.get('policy_id')}:{task.task_id}")
    return _assemble_trace(
        trace_id=rec_id,
        suite=suite,
        task=task,
        dialect=dialect,
        policy_id=str(record.get("policy_id", "exported")),
        candidate_tables=candidate_tables,
        selected_tables=selected_tables,
        probe_requests=probe_requests,
        probe_results=probe_results,
        schema_notes=schema_notes,
        initial_sql=initial_sql,
        final_sql=final_sql,
        first_exec=first_exec,
        final_exec=final_exec,
        ev=ev,
        rounds=rounds,
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


def render_inspector_html(traces: list[InspectedTrace]) -> str:
    """Render a self-contained, zero-CDN interactive HTML workbench separating Executable from EX."""
    summary = summarize_inspected_traces(traces)
    payload = {
        "summary": summary,
        "traces": [t.to_dict() for t in traces],
    }
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = Path(__file__).with_name("template.html").read_text(encoding="utf-8")
    return (
        template.replace("__TOTAL_TRACES__", str(summary["total_traces"]))
        .replace("__EXECUTABLE_COUNT__", str(summary["executable_count"]))
        .replace("__EXECUTABLE_PCT__", f"{summary['executable_rate'] * 100:.1f}")
        .replace("__RESULT_CORRECT_COUNT__", str(summary["result_correct_count"]))
        .replace("__RESULT_CORRECT_PCT__", f"{summary['result_correct_rate'] * 100:.1f}")
        .replace("__SILENT_TRAP_COUNT__", str(summary["silent_caliber_trap_count"]))
        .replace("__SILENT_TRAP_PCT__", f"{summary['silent_caliber_trap_rate'] * 100:.1f}")
        .replace("__GATEWAY_BLOCK_COUNT__", str(summary["gateway_or_syntax_block_count"]))
        .replace("__SAFE_JSON_PAYLOAD__", safe_json)
    )


from src.inspector.server_cli import (  # noqa: E402
    build_parser,
    build_showcase_traces,
    create_inspector_http_server,
    main,
    render_inspector_markdown,
    serve_inspector_http,
)

__all__ = [
    "InspectedTrace",
    "build_parser",
    "build_showcase_traces",
    "create_inspector_http_server",
    "inspect_agent_trajectory",
    "inspect_sft_jsonl_record",
    "main",
    "render_inspector_html",
    "render_inspector_markdown",
    "serve_inspector_http",
    "summarize_inspected_traces",
]

if __name__ == "__main__":
    raise SystemExit(main())
