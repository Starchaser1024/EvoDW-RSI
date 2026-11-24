"""Agentic RL Bridge: Exports Loss-Masked Multi-Turn SFT (OpenAI Tool-Use / ShareGPT) + GRPO datasets and standalone reward_fn."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from src.agent.generator import clean_sql_text
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.dw_generator import create_populated_sandbox
from src.benchmarks.tasks import BenchmarkTask, get_all_tasks
from src.env.sandbox import DialectType, WarehouseSandbox
from src.llm.client import JSONParseError, extract_json_object
from src.rsi.attribution import repair_prompt_dialect_violations

_DEFAULT_SANDBOX_CACHE: dict[DialectType, WarehouseSandbox] = {}
_DEFAULT_TASK_MAP: dict[str, BenchmarkTask] | None = None


def _get_default_sandbox(dialect: DialectType = "odps") -> WarehouseSandbox:
    if dialect not in _DEFAULT_SANDBOX_CACHE:
        _DEFAULT_SANDBOX_CACHE[dialect] = create_populated_sandbox(seed=2026, dialect=dialect)
    return _DEFAULT_SANDBOX_CACHE[dialect]


def _get_task_map() -> dict[str, BenchmarkTask]:
    global _DEFAULT_TASK_MAP
    if _DEFAULT_TASK_MAP is None:
        sb_odps = _get_default_sandbox("odps")
        all_t = get_all_tasks(suite="all")
        for t in all_t:
            res = sb_odps.run_sql(t.gold_sql, dialect="odps")
            if res["gateway_passed"] and res["error_stage"] == "none":
                t.gold_rows = res["rows"]
                t.gold_columns = res["columns"]
        _DEFAULT_TASK_MAP = {t.task_id: t for t in all_t}
    return _DEFAULT_TASK_MAP


def _normalize_sql_key(sql: str) -> str:
    return re.sub(r"\s+", " ", clean_sql_text(sql)).strip()


def _is_mock_record(messages: list[dict[str, Any]], meta: dict[str, Any]) -> bool:
    if meta.get("latency_ms") is not None and 0 < float(meta["latency_ms"]) < 200.0:
        return True
    blob = json.dumps(messages, ensure_ascii=False)
    mock_markers = (
        "Selected tables matching the business entities.",
        "Attempted minimal fix based on feedback.",
        "Applying partition pruning and MAX_PT rule.",
    )
    return any(m in blob for m in mock_markers)


def _load_valid_existing_records(
    openai_path: Path,
    *,
    task_map: dict[str, BenchmarkTask],
    sandbox: WarehouseSandbox,
    dialect: DialectType,
) -> list[dict[str, Any]]:
    """Load prior live verified SFT records across suites/dialects that pass strict gateway + EX verification."""
    if not openai_path.exists():
        return []
    full_task_map = {**_get_task_map(), **task_map}
    kept: list[dict[str, Any]] = []
    for raw_line in openai_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except Exception:
            continue
        tid = str(obj.get("task_id", ""))
        split = str(obj.get("split", ""))
        if split not in ("train", "val") or tid not in full_task_map:
            continue
        target_task = full_task_map[tid]
        if target_task.split not in ("train", "val"):
            continue
        msgs = obj.get("messages") or []
        if not isinstance(msgs, list) or _is_mock_record(msgs, obj):
            continue
        sql = _extract_sql_from_trajectory(msgs)
        if not sql:
            continue
        rec_dialect: DialectType = (
            "hive" if str(obj.get("dialect", dialect)).lower() == "hive" else "odps"
        )
        sb_rec = sandbox if rec_dialect == dialect else _get_default_sandbox(rec_dialect)
        ev = sb_rec.evaluate_sql(sql, target_task, dialect=rec_dialect)
        if not ev.gateway_passed or not ev.ex_match:
            continue

        cleaned_msgs: list[dict[str, Any]] = []
        for m in msgs:
            mc = dict(m)
            if mc.get("role") == "system" and isinstance(mc.get("content"), str):
                mc["content"], _ = repair_prompt_dialect_violations(mc["content"], dialect=rec_dialect)
            cleaned_msgs.append(mc)
        obj["messages"] = cleaned_msgs
        obj["dialect"] = rec_dialect
        obj["category"] = obj.get("category") or target_task.category
        obj["_norm_sql"] = _normalize_sql_key(sql)
        kept.append(obj)
    return kept


def export_sft_and_grpo_datasets(
    trajectories: list[AgentTrajectory],
    tasks: list[BenchmarkTask],
    output_dir: Path,
    *,
    merge_existing: bool = False,
    sandbox: WarehouseSandbox | None = None,
    dialect: DialectType = "odps",
    evolution_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export deduplicated verified multi-turn trajectories (`first_pass_correct` vs `self_repaired_after_error`) and GRPO prompts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    openai_path = output_dir / "sft_openai_tool_use.jsonl"
    sharegpt_path = output_dir / "sft_sharegpt.jsonl"
    grpo_path = output_dir / "grpo_prompts.jsonl"
    summary_path = output_dir / "export_summary.json"

    task_map = {t.task_id: t for t in tasks}
    prior_records: list[dict[str, Any]] = []
    if merge_existing and openai_path.exists():
        sb = sandbox or _get_default_sandbox(dialect=dialect)
        prior_records = _load_valid_existing_records(
            openai_path,
            task_map=task_map,
            sandbox=sb,
            dialect=dialect,
        )

    verified = [t for t in trajectories if t.ex_match and t.split in ("train", "val")]

    # Build unified deduplicated entry list (preferring fresh trajectories with full token/latency telemetry).
    unified_entries: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for traj in verified:
        norm_sql = _normalize_sql_key(traj.final_sql)
        key = (str(traj.dialect), traj.task_id, traj.outcome_label, norm_sql)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        messages: list[dict[str, Any]] = []
        for turn in traj.conversation_turns:
            role = turn["role"]
            loss_mask = 1 if role == "assistant" else 0
            content = turn["content"]
            if role == "system" and isinstance(content, str):
                content, _ = repair_prompt_dialect_violations(content, dialect=traj.dialect)
            msg_entry: dict[str, Any] = {
                "role": role,
                "content": content,
                "loss_mask": loss_mask,
            }
            if "name" in turn:
                msg_entry["name"] = turn["name"]
            if "module" in turn:
                msg_entry["module"] = turn["module"]
            messages.append(msg_entry)

        unified_entries.append(
            {
                "task_id": traj.task_id,
                "category": traj.category,
                "split": traj.split,
                "dialect": traj.dialect,
                "policy_id": traj.policy_id,
                "trajectory_type": traj.outcome_label,
                "refine_turns": traj.refine_turns,
                "num_llm_steps": traj.num_llm_steps,
                "total_tokens": traj.total_tokens,
                "latency_ms": round(traj.total_latency_ms, 2),
                "reward": traj.reward_info.get("total_reward", 1.0),
                "messages": messages,
            }
        )

    reused_prior_count = 0
    for rec in prior_records:
        ttype = str(rec.get("trajectory_type", "first_pass_correct"))
        key = (str(rec.get("dialect", dialect)), str(rec["task_id"]), ttype, str(rec["_norm_sql"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        reused_prior_count += 1
        msgs = rec["messages"]
        ass_steps = sum(1 for m in msgs if m.get("role") == "assistant")
        unified_entries.append(
            {
                "task_id": rec["task_id"],
                "category": rec.get("category") or _get_task_map()[rec["task_id"]].category,
                "split": rec["split"],
                "dialect": rec.get("dialect", dialect),
                "policy_id": rec.get("policy_id", "prior_live"),
                "trajectory_type": ttype,
                "refine_turns": int(rec.get("refine_turns", 0)),
                "num_llm_steps": int(rec.get("num_llm_steps", ass_steps)),
                "total_tokens": int(rec.get("total_tokens", 0)),
                "latency_ms": float(rec.get("latency_ms", 0.0)),
                "reward": float(rec.get("reward", 1.0)),
                "messages": msgs,
            }
        )

    openai_records: list[str] = []
    sharegpt_records: list[str] = []
    first_pass_count = 0
    self_repaired_count = 0
    split_counts: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    telemetry_buckets: dict[str, list[dict[str, float]]] = {
        "first_pass_correct": [],
        "self_repaired_after_error": [],
    }

    for idx, item in enumerate(unified_entries):
        ttype = item["trajectory_type"]
        if ttype == "first_pass_correct":
            first_pass_count += 1
        else:
            self_repaired_count += 1

        split_counts[item["split"]] += 1
        cat_counts[item["category"]] += 1
        policy_counts[item["policy_id"]] += 1

        if item["total_tokens"] > 0:
            telemetry_buckets.setdefault(ttype, []).append(
                {
                    "steps": float(item["num_llm_steps"]),
                    "tokens": float(item["total_tokens"]),
                    "latency_ms": float(item["latency_ms"]),
                }
            )

        conversations = []
        for m in item["messages"]:
            role = m["role"]
            sg_role = {
                "system": "system",
                "user": "human",
                "assistant": "gpt",
                "tool": "observation",
            }.get(role, "observation")
            conversations.append(
                {
                    "from": sg_role,
                    "value": m["content"],
                    "loss_mask": m["loss_mask"],
                }
            )

        base_meta = {
            "id": f"evodw_traj_{idx:05d}",
            "task_id": item["task_id"],
            "category": item["category"],
            "split": item["split"],
            "dialect": item["dialect"],
            "policy_id": item["policy_id"],
            "trajectory_type": ttype,
            "refine_turns": item["refine_turns"],
            "num_llm_steps": item["num_llm_steps"],
            "total_tokens": item["total_tokens"],
            "latency_ms": item["latency_ms"],
            "reward": item["reward"],
        }
        openai_records.append(json.dumps({**base_meta, "messages": item["messages"]}, ensure_ascii=False))
        sharegpt_records.append(json.dumps({**base_meta, "conversations": conversations}, ensure_ascii=False))

    openai_path.write_text("\n".join(openai_records) + ("\n" if openai_records else ""), encoding="utf-8")
    sharegpt_path.write_text("\n".join(sharegpt_records) + ("\n" if sharegpt_records else ""), encoding="utf-8")

    # Export train + val tasks for GRPO rollout training (strictly excluding test set!).
    grpo_task_source = get_all_tasks(suite="all") if merge_existing else tasks
    grpo_lines: list[str] = []
    for task in grpo_task_source:
        if task.split not in ("train", "val"):
            continue
        grpo_lines.append(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "split": task.split,
                    "category": task.category,
                    "dialect": dialect,
                    "prompt": f"Business Question: {task.question}\nBusiness Hint: {task.hint}",
                    "question": task.question,
                    "hint": task.hint,
                    "gold_sql": task.gold_sql,
                    "ordered": task.ordered,
                },
                ensure_ascii=False,
            )
        )
    grpo_path.write_text("\n".join(grpo_lines) + ("\n" if grpo_lines else ""), encoding="utf-8")

    total_exp = len(unified_entries)
    type_telemetry_summary: dict[str, Any] = {}
    for k, rows in telemetry_buckets.items():
        if rows:
            n_r = len(rows)
            type_telemetry_summary[k] = {
                "telemetry_sample_count": n_r,
                "avg_steps": round(sum(r["steps"] for r in rows) / n_r, 2),
                "avg_tokens": round(sum(r["tokens"] for r in rows) / n_r, 1),
                "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n_r, 1),
            }
        else:
            type_telemetry_summary[k] = {
                "telemetry_sample_count": 0,
                "avg_steps": 0.0,
                "avg_tokens": 0.0,
                "avg_latency_ms": 0.0,
            }

    summary_payload: dict[str, Any] = {
        "total_exported_trajectories": total_exp,
        "first_pass_correct_count": first_pass_count,
        "first_pass_correct_ratio": round(first_pass_count / max(1, total_exp), 4),
        "self_repaired_after_error_count": self_repaired_count,
        "self_repaired_after_error_ratio": round(self_repaired_count / max(1, total_exp), 4),
        "reused_prior_live_count": reused_prior_count,
        "grpo_prompt_count": len(grpo_lines),
        "by_split": dict(sorted(split_counts.items())),
        "by_category": dict(sorted(cat_counts.items())),
        "by_policy_id": dict(sorted(policy_counts.items())),
        "trajectory_type_telemetry": type_telemetry_summary,
        "evolution_comparison": evolution_comparison or {},
        "openai_jsonl": _rel_or_posix(openai_path),
        "sharegpt_jsonl": _rel_or_posix(sharegpt_path),
        "grpo_jsonl": _rel_or_posix(grpo_path),
        "summary_json": _rel_or_posix(summary_path),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


def _rel_or_posix(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_sql_from_trajectory(trajectory: str | list[dict[str, Any]] | AgentTrajectory) -> str:
    if isinstance(trajectory, AgentTrajectory):
        return clean_sql_text(trajectory.final_sql)

    if isinstance(trajectory, list):
        for msg in reversed(trajectory):
            if msg.get("role") in ("assistant", "gpt"):
                text = str(msg.get("content") or msg.get("value") or "")
                sql = _extract_sql_from_text(text)
                if sql:
                    return sql
        return ""

    return _extract_sql_from_text(str(trajectory))


def _extract_sql_from_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    try:
        obj = extract_json_object(text)
        if isinstance(obj.get("sql"), str) and obj["sql"].strip():
            return clean_sql_text(obj["sql"])
    except JSONParseError:
        pass
    m = re.search(r"```(?:sql)?\s*(SELECT.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return clean_sql_text(m.group(1))
    m_sel = re.search(r"\b((?:WITH|SELECT)\b.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m_sel:
        return clean_sql_text(m_sel.group(1))
    return ""


def reward_fn(
    prompt: str | dict[str, Any],
    trajectory: str | list[dict[str, Any]] | AgentTrajectory,
    *,
    sandbox: WarehouseSandbox | None = None,
    dialect: DialectType = "odps",
) -> float:
    """Standalone GRPO/TRL/verl-compatible scalar reward function."""
    sb = sandbox or _get_default_sandbox(dialect=dialect)
    pred_sql = _extract_sql_from_trajectory(trajectory)
    if not pred_sql:
        return -0.5

    task_map = _get_task_map()
    target: Any = None
    if isinstance(prompt, dict):
        if "gold_sql" in prompt and prompt["gold_sql"]:
            target = prompt["gold_sql"]
            if prompt.get("task_id") in task_map:
                target = task_map[prompt["task_id"]]
        elif prompt.get("task_id") in task_map:
            target = task_map[prompt["task_id"]]
    else:
        p_str = str(prompt)
        for tid, task in task_map.items():
            if tid in p_str or task.question in p_str:
                target = task
                break

    if target is None:
        exec_res = sb.run_sql(pred_sql, dialect=dialect)
        if not exec_res["gateway_passed"]:
            return 0.0
        return 0.3 if exec_res["error_stage"] == "none" else 0.1

    refine_turns = trajectory.refine_turns if isinstance(trajectory, AgentTrajectory) else 0
    first_gw = trajectory.first_pass_gateway if isinstance(trajectory, AgentTrajectory) else None
    breakdown = sb.compute_reward(
        pred_sql,
        target,
        refine_turns=refine_turns,
        first_pass_gateway=first_gw,
        dialect=dialect,
    )
    return float(breakdown["total_reward"])
