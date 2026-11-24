"""Four-stage progressive dense + self-repair + efficiency reward suite for EvoDW-RL v2."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
import re
from typing import Any

from src.agent.generator import clean_sql_text
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.tasks import BenchmarkTask
from src.env.sandbox import DialectType, WarehouseSandbox
from src.eval.comparator import compare_result_sets
from src.export.rl_bridge import (
    _extract_sql_from_trajectory,
    _get_default_sandbox,
    _get_task_map,
    reward_fn as legacy_bridge_reward_fn,
)
from src.llm.client import JSONParseError, extract_json_object
from src.rl.config import RewardAblationMode, RewardStageWeights


@dataclass
class HierarchicalRewardBreakdown:
    """Detailed 4-stage + multi-turn self-repair & length/step efficiency reward breakdown."""

    total_reward: float
    format_reward: float
    gateway_ast_reward: float
    exec_validity_reward: float
    ex_outcome_reward: float

    strict_json_ok: bool
    format_collapsed: bool
    extracted_sql: str
    gateway_passed: bool
    gateway_error_code: str
    error_stage: str
    error_msg: str
    is_suspicious_output: bool
    ex_match: bool
    first_pass_bonus: float
    self_repair_bonus: float
    step_penalty: float
    length_penalty: float
    shape_bonus: float
    refine_turns: int
    probe_turns: int
    had_initial_error: bool
    diff_reason: str
    legacy_bridge_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_multiturn_actions_and_meta(completion: Any) -> dict[str, Any]:
    """Parse final assistant text and multi-turn metadata from str, multi-turn ChatML, dict, or AgentTrajectory."""
    if isinstance(completion, AgentTrajectory):
        action_texts = [mc.raw_output for mc in completion.module_calls] if completion.module_calls else [completion.final_sql]
        final_text = action_texts[-1] if action_texts else completion.final_sql
        had_err = (not completion.first_pass_gateway) or (completion.refine_turns > 0)
        return {
            "final_text": final_text,
            "action_texts": action_texts,
            "total_action_chars": sum(len(x) for x in action_texts),
            "refine_turns": completion.refine_turns,
            "probe_turns": 1 if completion.probe_requests else 0,
            "first_pass_gateway": completion.first_pass_gateway,
            "had_initial_error": had_err,
        }

    if isinstance(completion, dict) and "final_action_text" in completion:
        action_texts = list(completion.get("action_texts") or [completion["final_action_text"]])
        return {
            "final_text": str(completion["final_action_text"]),
            "action_texts": action_texts,
            "total_action_chars": int(completion.get("total_action_chars") or sum(len(x) for x in action_texts)),
            "refine_turns": int(completion.get("refine_turns", 0)),
            "probe_turns": int(completion.get("probe_turns", 0)),
            "first_pass_gateway": completion.get("first_pass_gateway"),
            "had_initial_error": bool(completion.get("had_initial_error", False)),
        }

    if isinstance(completion, str):
        # Detect if decoded completion contains interleaved multi-turn <tool_response> blocks
        if "<tool_response" in completion and "<|im_start|>assistant" in completion:
            parts = re.split(r"<\|im_start\|>assistant\s*", completion)
            action_texts: list[str] = []
            for idx, chunk in enumerate(parts):
                if idx == 0 and "<|im_start|>user" in chunk:
                    first_act = chunk.split("<|im_start|>user", 1)[0].replace("<|im_end|>", "").strip()
                    if first_act:
                        action_texts.append(first_act)
                    continue
                act = chunk.split("<|im_start|>", 1)[0].replace("<|im_end|>", "").strip()
                if act:
                    action_texts.append(act)
            probe_turns = completion.count('name="inspect_table_sample"')
            refine_turns = completion.count('name="sandbox_execute"')
            final_text = action_texts[-1] if action_texts else completion
            return {
                "final_text": final_text,
                "action_texts": action_texts or [final_text],
                "total_action_chars": sum(len(x) for x in action_texts) if action_texts else len(final_text),
                "refine_turns": refine_turns,
                "probe_turns": probe_turns,
                "first_pass_gateway": False if refine_turns > 0 else None,
                "had_initial_error": refine_turns > 0,
            }
        return {
            "final_text": completion,
            "action_texts": [completion],
            "total_action_chars": len(completion),
            "refine_turns": None,
            "probe_turns": 0,
            "first_pass_gateway": None,
            "had_initial_error": None,
        }

    if isinstance(completion, dict):
        txt = str(completion.get("content") or completion.get("value") or completion.get("sql") or "")
        return {
            "final_text": txt,
            "action_texts": [txt],
            "total_action_chars": len(txt),
            "refine_turns": None,
            "probe_turns": 0,
            "first_pass_gateway": None,
            "had_initial_error": None,
        }

    if isinstance(completion, list):
        ass_turns = [
            str(msg.get("content") or msg.get("value") or "")
            for msg in completion
            if isinstance(msg, dict) and msg.get("role") in ("assistant", "gpt", None)
        ]
        ass_turns = [t for t in ass_turns if t.strip()]
        tool_turns = [
            msg for msg in completion if isinstance(msg, dict) and msg.get("role") in ("tool", "observation")
        ]
        probe_cnt = sum(1 for m in tool_turns if m.get("name") == "inspect_table_sample")
        refine_cnt = sum(1 for m in tool_turns if m.get("name") == "sandbox_execute")
        final_txt = ass_turns[-1] if ass_turns else ""
        return {
            "final_text": final_txt,
            "action_texts": ass_turns,
            "total_action_chars": sum(len(x) for x in ass_turns),
            "refine_turns": refine_cnt if tool_turns else None,
            "probe_turns": probe_cnt,
            "first_pass_gateway": False if refine_cnt > 0 else None,
            "had_initial_error": refine_cnt > 0 if tool_turns else None,
        }

    txt = str(completion or "")
    return {
        "final_text": txt,
        "action_texts": [txt],
        "total_action_chars": len(txt),
        "refine_turns": None,
        "probe_turns": 0,
        "first_pass_gateway": None,
        "had_initial_error": None,
    }


def unwrap_completion_text(completion: Any) -> str:
    """Extract raw final assistant text from TRL str, multi-turn payload, chat-message list, or AgentTrajectory."""
    return str(extract_multiturn_actions_and_meta(completion)["final_text"])


def _is_select_or_with(sql: str) -> bool:
    s = (sql or "").lstrip()
    if not s:
        return False
    first_tok = s.split(None, 1)[0].upper().lstrip("(")
    return first_tok in ("SELECT", "WITH")


def compute_action_length_penalty(
    action_texts: list[str],
    *,
    length_free_chars: int = 1400,
    max_length_penalty: float = 0.15,
) -> float:
    """Compute smooth length & anti-repetition penalty strictly over model-generated Action characters."""
    joined = "\n".join(t for t in action_texts if t)
    total_chars = len(joined)
    penalty = 0.0

    if total_chars > length_free_chars:
        excess = total_chars - length_free_chars
        penalty += min(max_length_penalty, 0.05 * (excess / 600.0))

    # Detect degenerate n-gram babbling / repetition loops
    if total_chars >= 240:
        window = 32
        chunks = [joined[i : i + window] for i in range(0, len(joined) - window + 1, window)]
        if chunks:
            unique_ratio = len(set(chunks)) / len(chunks)
            if unique_ratio < 0.45:
                penalty += 0.08

    return round(min(max_length_penalty, penalty), 4)


def score_format_stage(completion: Any) -> tuple[float, str, bool]:
    """Stage 1 — Format Reward: strict JSON schema + read-only SQL extractability.

    Returns:
        (raw_format_score, extracted_sql, strict_json_ok)
    """
    if isinstance(completion, AgentTrajectory):
        sql = clean_sql_text(completion.final_sql)
        if _is_select_or_with(sql):
            return 0.20, sql, True
        return -0.50, "", False

    text = unwrap_completion_text(completion)
    if not text or not text.strip():
        return -0.50, "", False

    try:
        parsed = extract_json_object(text)
        raw_sql = parsed.get("sql")
        if isinstance(raw_sql, str) and raw_sql.strip():
            cleaned = clean_sql_text(raw_sql)
            if _is_select_or_with(cleaned):
                has_thought = isinstance(parsed.get("thought"), str) and bool(parsed["thought"].strip())
                return (0.20 if has_thought else 0.15), cleaned, has_thought
    except JSONParseError:
        pass

    fallback_sql = _extract_sql_from_trajectory(text)
    if _is_select_or_with(fallback_sql):
        return 0.05, fallback_sql, False

    return -0.50, "", False


def score_gateway_ast_stage(
    extracted_sql: str,
    exec_res: dict[str, Any] | None,
) -> tuple[float, bool, str]:
    """Stage 2 — Gateway AST Reward: partition pruning (`ds`/`dt`/`MAX_PT`), ODPS dialect & join safety."""
    if not extracted_sql or exec_res is None:
        return -0.20, False, "NO_SQL"

    if exec_res.get("gateway_passed", False):
        return 0.30, True, ""

    err_code = str(exec_res.get("error_code") or "")
    if err_code == "ODPS-0123031":
        # Valid ODPS SELECT grammar, but missing partition pruning predicate.
        return -0.10, False, err_code
    if err_code in ("ODPS-0130071", "ODPS-0123055", "ODPS-0130131") or err_code.startswith("HIVE-"):
        # Dialect function hallucination, wrong DATEDIFF/DATEADD arity, or Cartesian join.
        return -0.15, False, err_code
    # Tokenization / AST syntax parse exception (e.g., ODPS-0130161).
    return -0.20, False, err_code or "ODPS-0130161"


def score_execution_validity_stage(
    exec_res: dict[str, Any] | None,
    *,
    gold_is_empty: bool = False,
) -> tuple[float, bool, str]:
    """Stage 3 — Execution Validity Reward: zero DuckDB runtime errors & non-empty/non-all-NULL output."""
    if exec_res is None or not exec_res.get("gateway_passed", False):
        return 0.0, True, "Gateway check did not pass."

    if exec_res.get("error_stage", "none") != "none":
        return -0.10, True, str(exec_res.get("error_msg", "Runtime execution error."))

    suspicious, reason = WarehouseSandbox.is_suspicious_result(exec_res)
    if suspicious:
        if gold_is_empty and len(exec_res.get("rows") or []) == 0:
            return 0.30, False, ""
        return 0.10, True, reason

    return 0.30, False, ""


class HierarchicalSQLReward:
    """Stateful, cached, ablation-ready 4-stage + multi-turn self-repair reward engine wrapping WarehouseSandbox."""

    def __init__(
        self,
        sandbox: WarehouseSandbox | None = None,
        *,
        dialect: DialectType = "odps",
        weights: RewardStageWeights | None = None,
        cache_size: int = 2048,
    ) -> None:
        self.sandbox = sandbox or _get_default_sandbox(dialect=dialect)
        self.dialect: DialectType = dialect
        self.weights = weights or RewardStageWeights()
        self._task_map: dict[str, BenchmarkTask] = _get_task_map()
        self._sql_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._gold_cache: OrderedDict[tuple[str, str], tuple[list[tuple[Any, ...]], list[str]]] = OrderedDict()
        self._cache_size = max(64, cache_size)

    def _run_sql_cached(self, sql: str, dialect: DialectType) -> dict[str, Any]:
        key = (dialect, sql)
        if key in self._sql_cache:
            self._sql_cache.move_to_end(key)
            return self._sql_cache[key]
        res = self.sandbox.run_sql(sql, dialect=dialect)
        self._sql_cache[key] = res
        if len(self._sql_cache) > self._cache_size:
            self._sql_cache.popitem(last=False)
        return res

    def _resolve_gold(
        self,
        prompt_or_meta: Any,
        dialect: DialectType,
    ) -> tuple[BenchmarkTask | None, list[tuple[Any, ...]], list[str], bool] | None:
        task: BenchmarkTask | None = None
        gold_sql = ""
        ordered = False

        if isinstance(prompt_or_meta, BenchmarkTask):
            task = prompt_or_meta
        elif isinstance(prompt_or_meta, dict):
            tid = str(prompt_or_meta.get("task_id") or "")
            if tid and tid in self._task_map:
                task = self._task_map[tid]
            if prompt_or_meta.get("gold_sql"):
                gold_sql = str(prompt_or_meta["gold_sql"])
            if "ordered" in prompt_or_meta and prompt_or_meta["ordered"] is not None:
                ordered = bool(prompt_or_meta["ordered"])
        elif prompt_or_meta is not None:
            p_text = str(prompt_or_meta)
            for tid, cand in self._task_map.items():
                if tid in p_text or cand.question in p_text:
                    task = cand
                    break

        if task is not None:
            ordered = bool(task.ordered)
            if task.gold_rows is not None and task.gold_columns is not None:
                return task, task.gold_rows, task.gold_columns, ordered
            gold_sql = gold_sql or task.gold_sql

        if not gold_sql:
            return None

        g_key = (dialect, gold_sql)
        if g_key in self._gold_cache:
            self._gold_cache.move_to_end(g_key)
            g_rows, g_cols = self._gold_cache[g_key]
            return task, g_rows, g_cols, ordered

        g_res = self._run_sql_cached(gold_sql, dialect=dialect)
        g_rows = g_res["rows"]
        g_cols = g_res["columns"]
        self._gold_cache[g_key] = (g_rows, g_cols)
        if task is not None:
            task.gold_rows = g_rows
            task.gold_columns = g_cols
        return task, g_rows, g_cols, ordered

    def evaluate_rollout(
        self,
        prompt_or_meta: Any,
        completion: Any,
        *,
        dialect: DialectType | None = None,
        refine_turns: int | None = None,
        probe_turns: int | None = None,
        first_pass_gateway: bool | None = None,
        had_initial_error: bool | None = None,
    ) -> HierarchicalRewardBreakdown:
        """Compute full 4-stage progressive + self-repair + step/length penalty reward breakdown."""
        active_dialect: DialectType = dialect or self.dialect
        w = self.weights

        parsed_meta = extract_multiturn_actions_and_meta(completion)
        meta_dict = prompt_or_meta if isinstance(prompt_or_meta, dict) else {}

        eff_refine = int(
            refine_turns
            if refine_turns is not None
            else (
                meta_dict.get("refine_turns")
                if meta_dict.get("refine_turns") is not None
                else (parsed_meta["refine_turns"] or 0)
            )
        )
        eff_probe = int(
            probe_turns
            if probe_turns is not None
            else (
                meta_dict.get("probe_turns")
                if meta_dict.get("probe_turns") is not None
                else (parsed_meta["probe_turns"] or 0)
            )
        )

        # Stage 1: Format & SQL Extraction
        raw_fmt, extracted_sql, strict_json_ok = score_format_stage(completion)
        format_collapsed = not bool(extracted_sql)
        fmt_score = (raw_fmt * w.format_scale) if w.enable_format else 0.0

        exec_res: dict[str, Any] | None = None
        if extracted_sql:
            exec_res = self._run_sql_cached(extracted_sql, dialect=active_dialect)

        # Stage 2: Gateway AST Linter
        raw_gw, gw_passed, gw_code = score_gateway_ast_stage(extracted_sql, exec_res)
        gw_score = (raw_gw * w.gateway_scale) if w.enable_gateway_ast else 0.0

        # Resolve Gold answer set if available
        gold_info = self._resolve_gold(prompt_or_meta, dialect=active_dialect)
        gold_is_empty = bool(gold_info is not None and len(gold_info[1]) == 0)

        # Stage 3: DuckDB Execution Validity & Non-Suspicious Check
        raw_exec, is_suspicious, susp_reason = score_execution_validity_stage(
            exec_res,
            gold_is_empty=gold_is_empty,
        )
        exec_score = (raw_exec * w.exec_scale) if w.enable_exec_validity else 0.0

        # Resolve first_pass_gateway & had_initial_error for multi-turn self-repair credit assignment
        eff_first_gw_raw = (
            first_pass_gateway
            if first_pass_gateway is not None
            else (
                meta_dict.get("first_pass_gateway")
                if meta_dict.get("first_pass_gateway") is not None
                else parsed_meta["first_pass_gateway"]
            )
        )
        eff_first_gw = bool(eff_first_gw_raw) if eff_first_gw_raw is not None else (gw_passed and eff_refine == 0)

        eff_had_err_raw = (
            had_initial_error
            if had_initial_error is not None
            else (
                meta_dict.get("had_initial_error")
                if meta_dict.get("had_initial_error") is not None
                else parsed_meta["had_initial_error"]
            )
        )
        eff_had_err = bool(eff_had_err_raw) if eff_had_err_raw is not None else (eff_refine > 0 or not eff_first_gw)

        ex_match = False
        first_pass_bonus = 0.0
        self_repair_bonus = 0.0
        step_penalty = 0.0
        length_penalty = 0.0
        shape_bonus = 0.0
        diff_reason = ""
        raw_outcome = 0.0

        action_texts = list(meta_dict.get("action_texts") or parsed_meta["action_texts"])
        if w.enable_step_length_penalty and not w.sparse_outcome_only:
            step_penalty = round(
                w.step_penalty_per_turn * max(0, eff_refine) + w.probe_step_penalty * max(0, eff_probe),
                4,
            )
            length_penalty = compute_action_length_penalty(
                action_texts,
                length_free_chars=w.length_free_chars,
                max_length_penalty=w.max_length_penalty,
            )

        if gold_info is not None and exec_res is not None:
            _, gold_rows, gold_cols, ordered = gold_info
            if gw_passed and exec_res.get("error_stage") == "none":
                pred_rows = exec_res.get("rows") or []
                pred_cols = exec_res.get("columns") or []
                ex_match, diff_reason = compare_result_sets(
                    pred_rows=pred_rows,
                    gold_rows=gold_rows,
                    ordered=ordered,
                )
                if w.sparse_outcome_only:
                    # Pure 0/1 sparse outcome reward for clean sparse-vs-dense RL ablation
                    raw_outcome = 1.0 if ex_match else 0.0
                elif ex_match:
                    if eff_first_gw and eff_refine == 0 and not eff_had_err:
                        first_pass_bonus = w.first_pass_bonus
                    elif w.enable_self_repair_bonus and eff_had_err and eff_refine >= 1:
                        # Explicit Self-Repair Bonus: Turn 1 erred in sandbox, Turn 2/3 diagnosed & fixed to Gold!
                        self_repair_bonus = w.self_repair_bonus
                    raw_outcome = 1.00 + first_pass_bonus + self_repair_bonus - step_penalty - length_penalty
                else:
                    if gold_cols and len(pred_cols) == len(gold_cols):
                        shape_bonus += 0.05
                    if len(pred_rows) == len(gold_rows) and len(gold_rows) > 0:
                        shape_bonus += 0.05
                    if w.enable_self_repair_bonus and eff_had_err and eff_refine >= 1:
                        # Partial self-repair credit: repaired ODPS gateway/runtime error into clean execution
                        self_repair_bonus = w.gateway_repair_bonus
                    raw_outcome = shape_bonus + self_repair_bonus - step_penalty - length_penalty
            else:
                diff_reason = str(exec_res.get("error_msg") or susp_reason)
                raw_outcome = 0.0 if w.sparse_outcome_only else (-step_penalty - length_penalty)
        elif not extracted_sql:
            diff_reason = "No valid SELECT/WITH SQL could be extracted from model output."
            raw_outcome = 0.0 if w.sparse_outcome_only else (-step_penalty - length_penalty)

        outcome_score = (raw_outcome * w.outcome_scale) if w.enable_ex_outcome else 0.0
        total_reward = round(fmt_score + gw_score + exec_score + outcome_score, 4)

        # Compute lightweight sandbox scalar score for cross-stage telemetry alignment.
        if not extracted_sql:
            legacy_score = -0.5
        elif gold_info is not None:
            legacy_ex = 1.0 if ex_match else 0.0
            legacy_gw = 0.2 if gw_passed else 0.0
            legacy_ex_valid = 0.1 if (exec_res and exec_res.get("error_stage") == "none") else 0.0
            legacy_fp = 0.1 if eff_first_gw else 0.0
            legacy_score = round(
                (legacy_ex + legacy_fp - step_penalty)
                if ex_match
                else (legacy_gw + legacy_ex_valid - step_penalty),
                4,
            )
        else:
            legacy_score = legacy_bridge_reward_fn(
                prompt_or_meta or "",
                extracted_sql,
                sandbox=self.sandbox,
                dialect=active_dialect,
            )

        return HierarchicalRewardBreakdown(
            total_reward=total_reward,
            format_reward=round(fmt_score, 4),
            gateway_ast_reward=round(gw_score, 4),
            exec_validity_reward=round(exec_score, 4),
            ex_outcome_reward=round(outcome_score, 4),
            strict_json_ok=strict_json_ok,
            format_collapsed=format_collapsed,
            extracted_sql=extracted_sql,
            gateway_passed=gw_passed,
            gateway_error_code=gw_code,
            error_stage=str(exec_res.get("error_stage", "gateway_ast")) if exec_res else "format",
            error_msg=str(exec_res.get("error_msg", "")) if exec_res else diff_reason,
            is_suspicious_output=is_suspicious,
            ex_match=ex_match,
            first_pass_bonus=round(first_pass_bonus, 4),
            self_repair_bonus=round(self_repair_bonus, 4),
            step_penalty=step_penalty,
            length_penalty=length_penalty,
            shape_bonus=round(shape_bonus, 4),
            refine_turns=eff_refine,
            probe_turns=eff_probe,
            had_initial_error=eff_had_err,
            diff_reason=diff_reason,
            legacy_bridge_reward=legacy_score,
        )

    @staticmethod
    def _build_meta_list(prompts: list[Any], kwargs: dict[str, Any], n: int) -> list[dict[str, Any]]:
        task_ids = kwargs.get("task_id") or [None] * n
        gold_sqls = kwargs.get("gold_sql") or [None] * n
        ordereds = kwargs.get("ordered") or [None] * n
        dialects = kwargs.get("dialect") or [None] * n
        rollout_metas = kwargs.get("rollout_meta") or [None] * n

        metas: list[dict[str, Any]] = []
        for idx in range(n):
            p = prompts[idx] if idx < len(prompts) else ""
            if isinstance(p, list):
                p_str = "\n".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
            else:
                p_str = str(p)
            rm = rollout_metas[idx] if idx < len(rollout_metas) and isinstance(rollout_metas[idx], dict) else {}
            entry: dict[str, Any] = {
                "task_id": task_ids[idx] if idx < len(task_ids) else None,
                "gold_sql": gold_sqls[idx] if idx < len(gold_sqls) else None,
                "ordered": ordereds[idx] if idx < len(ordereds) else None,
                "dialect": dialects[idx] if idx < len(dialects) else None,
                "prompt_text": p_str,
            }
            entry.update(rm)
            metas.append(entry)
        return metas

    @staticmethod
    def _pair_completion_with_meta(comp: Any, meta: dict[str, Any]) -> Any:
        if "final_action_text" in meta:
            return meta
        return comp

    def format_reward_func(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        """TRL GRPOTrainer Stage-1 reward function: strict JSON & SQL extractability."""
        metas = self._build_meta_list(prompts, kwargs, len(completions))
        return [
            self.evaluate_rollout(meta, self._pair_completion_with_meta(comp, meta)).format_reward
            for meta, comp in zip(metas, completions)
        ]

    def gateway_ast_reward_func(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        """TRL GRPOTrainer Stage-2 reward function: ODPS partition pruning & dialect AST compliance."""
        metas = self._build_meta_list(prompts, kwargs, len(completions))
        return [
            self.evaluate_rollout(meta, self._pair_completion_with_meta(comp, meta)).gateway_ast_reward
            for meta, comp in zip(metas, completions)
        ]

    def execution_validity_reward_func(
        self, prompts: list[Any], completions: list[Any], **kwargs: Any
    ) -> list[float]:
        """TRL GRPOTrainer Stage-3 reward function: DuckDB deterministic execution & non-null validity."""
        metas = self._build_meta_list(prompts, kwargs, len(completions))
        return [
            self.evaluate_rollout(meta, self._pair_completion_with_meta(comp, meta)).exec_validity_reward
            for meta, comp in zip(metas, completions)
        ]

    def ex_outcome_efficiency_reward_func(
        self, prompts: list[Any], completions: list[Any], **kwargs: Any
    ) -> list[float]:
        """TRL GRPOTrainer Stage-4 reward function: EX match + 1st-pass / self-repair bonus - step/length penalty."""
        metas = self._build_meta_list(prompts, kwargs, len(completions))
        return [
            self.evaluate_rollout(meta, self._pair_completion_with_meta(comp, meta)).ex_outcome_reward
            for meta, comp in zip(metas, completions)
        ]

    def combined_reward_func(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        """Single scalar TRL reward function summing all enabled reward stages."""
        metas = self._build_meta_list(prompts, kwargs, len(completions))
        return [
            self.evaluate_rollout(meta, self._pair_completion_with_meta(comp, meta)).total_reward
            for meta, comp in zip(metas, completions)
        ]

    def build_trl_reward_funcs(self) -> list[Callable[..., list[float]]]:
        """Return list of named TRL reward functions respecting active ablation toggles."""
        funcs: list[Callable[..., list[float]]] = []
        w = self.weights

        if w.enable_format:
            def format_reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
                return self.format_reward_func(prompts, completions, **kwargs)
            funcs.append(format_reward)

        if w.enable_gateway_ast:
            def gateway_ast_reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
                return self.gateway_ast_reward_func(prompts, completions, **kwargs)
            funcs.append(gateway_ast_reward)

        if w.enable_exec_validity:
            def execution_validity_reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
                return self.execution_validity_reward_func(prompts, completions, **kwargs)
            funcs.append(execution_validity_reward)

        if w.enable_ex_outcome:
            def ex_outcome_efficiency_reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
                return self.ex_outcome_efficiency_reward_func(prompts, completions, **kwargs)
            funcs.append(ex_outcome_efficiency_reward)

        if not funcs:
            raise ValueError("At least one reward stage must be enabled in RewardStageWeights.")
        return funcs


def compute_hierarchical_reward(
    prompt: str | dict[str, Any] | BenchmarkTask,
    trajectory_or_completion: Any,
    *,
    sandbox: WarehouseSandbox | None = None,
    dialect: DialectType = "odps",
    ablation_mode: RewardAblationMode = "full",
    refine_turns: int | None = None,
    first_pass_gateway: bool | None = None,
    had_initial_error: bool | None = None,
) -> HierarchicalRewardBreakdown:
    """Convenience functional API extending `src.export.rl_bridge.reward_fn` with v2 4-stage + self-repair diagnostics."""
    weights = RewardStageWeights.from_ablation(ablation_mode)
    engine = HierarchicalSQLReward(sandbox=sandbox, dialect=dialect, weights=weights)
    return engine.evaluate_rollout(
        prompt,
        trajectory_or_completion,
        dialect=dialect,
        refine_turns=refine_turns,
        first_pass_gateway=first_pass_gateway,
        had_initial_error=had_initial_error,
    )
