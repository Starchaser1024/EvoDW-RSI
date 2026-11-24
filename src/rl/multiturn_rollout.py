"""Multi-Turn Agentic Sandbox Rollout & Exact Action/Observation Mask Engine (`src/rl/multiturn_rollout.py`).

Implements 2~3 turn interactive tool/sandbox rollouts for Stage-2 GRPO:
- Turn 1 may emit `action="probe"` (`inspect_table_sample`) or initial SQL.
- When SQL triggers `WarehouseSandbox` gateway AST interception (e.g., `ODPS-0123031`) or DuckDB
  runtime/suspicious errors, the real sandbox error payload is formatted as a ChatML `tool` + `user`
  observation and appended to context so the policy generates a Turn-2/Turn-3 repair SQL.
- Enforces two distinct masks:
  1. `action_loss_mask` (`completion_mask` in GRPO): `1` STRICTLY on model-generated Action tokens,
     `0` STRICTLY on all injected Sandbox Observation tokens, ChatML role wrappers, and right-padding.
  2. `completion_attn_mask`: `1` on ALL real Action + Observation tokens so later turns causally attend
     to earlier sandbox error stacks during policy & KL log-prob evaluation, `0` ONLY on right-padding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from src.agent.generator import clean_sql_text
from src.env.sandbox import DialectType, WarehouseSandbox
from src.export.rl_bridge import _extract_sql_from_trajectory
from src.llm.client import JSONParseError, extract_json_object

# Pluggable batch generator signature:
#   (prompt_token_ids_batch: list[list[int]], max_new_tokens: int) -> list[list[int]]
BatchTokenGeneratorFn = Callable[[list[list[int]], int], list[list[int]]]


@dataclass
class MultiTurnEpisodeRollout:
    """Token-exact trace of a single 1~3 turn interactive warehouse rollout."""

    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    action_loss_mask: list[int] = field(default_factory=list)
    completion_attn_mask: list[int] = field(default_factory=list)

    action_texts: list[str] = field(default_factory=list)
    observation_texts: list[str] = field(default_factory=list)
    action_spans: list[tuple[int, int]] = field(default_factory=list)
    observation_spans: list[tuple[int, int]] = field(default_factory=list)

    num_turns: int = 0
    probe_turns: int = 0
    refine_turns: int = 0
    first_pass_gateway: bool = False
    had_initial_error: bool = False
    final_action_text: str = ""
    terminated_with_eos: bool = True

    @property
    def action_token_count(self) -> int:
        return sum(self.action_loss_mask)

    @property
    def observation_token_count(self) -> int:
        return sum(1 for a, m in zip(self.completion_attn_mask, self.action_loss_mask) if a == 1 and m == 0)

    def to_reward_meta(self) -> dict[str, Any]:
        return {
            "final_action_text": self.final_action_text,
            "action_texts": list(self.action_texts),
            "total_action_chars": sum(len(x) for x in self.action_texts),
            "num_turns": self.num_turns,
            "probe_turns": self.probe_turns,
            "refine_turns": self.refine_turns,
            "first_pass_gateway": self.first_pass_gateway,
            "had_initial_error": self.had_initial_error,
        }


def get_chatml_stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Collect EOS and `<|im_end|>` token IDs for clean ChatML turn stopping."""
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in stop_ids:
        stop_ids.append(im_end)
    return stop_ids


def trim_at_first_stop_token(token_ids: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    """Keep tokens up to and including the first stop token; return `(trimmed_ids, hit_eos)`."""
    for idx, tid in enumerate(token_ids):
        if tid in stop_ids:
            return list(token_ids[: idx + 1]), True
    return list(token_ids), False


def format_probe_observation_chatml(
    probe_results: list[dict[str, Any]],
    *,
    dialect: str = "ODPS",
    prepend_im_end: bool = False,
) -> str:
    """Format `inspect_table_sample` tool output into a masked (`mask=0`) ChatML observation block."""
    prefix = "<|im_end|>\n" if prepend_im_end else "\n"
    obs_json = json.dumps(probe_results, ensure_ascii=False)
    return (
        f"{prefix}<|im_start|>user\n"
        f'<tool_response name="inspect_table_sample">\n{obs_json}\n</tool_response><|im_end|>\n'
        f"<|im_start|>user\n"
        f"Based on the sampled partition and column observations above, generate the single read-only "
        f'{dialect.upper()} SQL query and return strict JSON: {{"thought": "...", "sql": "..."}}.<|im_end|>\n'
        f"<|im_start|>assistant\n"
    )


def format_sandbox_error_observation_chatml(
    previous_sql: str,
    exec_res: dict[str, Any],
    *,
    suspicion_reason: str = "",
    dialect: str = "ODPS",
    prepend_im_end: bool = False,
) -> str:
    """Format `WarehouseSandbox` gateway/runtime error feedback into a masked (`mask=0`) ChatML observation block."""
    prefix = "<|im_end|>\n" if prepend_im_end else "\n"
    feedback_payload = {
        "gateway_passed": bool(exec_res.get("gateway_passed", False)),
        "error_code": str(exec_res.get("error_code") or ""),
        "error_stage": str(exec_res.get("error_stage", "gateway_ast")),
        "error_msg": str(exec_res.get("error_msg") or suspicion_reason or "")[:420],
        "suspicion_reason": str(suspicion_reason or "")[:200],
        "row_count": int(exec_res.get("row_count", 0) or 0),
    }
    compact_feedback = json.dumps(feedback_payload, ensure_ascii=False)
    pretty_feedback = json.dumps(feedback_payload, ensure_ascii=False, indent=2)
    shown_sql = (previous_sql or "(no valid SQL extracted)")[:500]
    return (
        f"{prefix}<|im_start|>user\n"
        f'<tool_response name="sandbox_execute">\n{compact_feedback}\n</tool_response><|im_end|>\n'
        f"<|im_start|>user\n"
        f"Target Dialect: {dialect.upper()}\n"
        f"Previous SQL:\n{shown_sql}\n\n"
        f"Sandbox Execution Feedback:\n{pretty_feedback}\n\n"
        f"Diagnose the root cause (e.g., missing partition filter / MAX_PT, ODPS function arity, "
        f'join key, or column name) and output repaired strict JSON: {{"thought": "...", "diagnosis": "...", "sql": "..."}}.<|im_end|>\n'
        f"<|im_start|>assistant\n"
    )


def _is_select_or_with_sql(sql: str) -> bool:
    s = (sql or "").lstrip()
    if not s:
        return False
    first_tok = s.split(None, 1)[0].upper().lstrip("(")
    return first_tok in ("SELECT", "WITH")


def parse_agent_turn_output(raw_text: str) -> tuple[str, dict[str, Any], str]:
    """Classify an assistant turn into `('probe' | 'sql' | 'invalid', parsed_dict, cleaned_sql)`."""
    text = (raw_text or "").strip()
    if not text:
        return "invalid", {}, ""

    parsed: dict[str, Any] = {}
    try:
        parsed = extract_json_object(text)
    except JSONParseError:
        parsed = {}

    action_tag = str(parsed.get("action") or "").strip().lower()
    if (action_tag == "probe" or "probe_requests" in parsed) and not parsed.get("sql"):
        reqs = parsed.get("probe_requests")
        if isinstance(reqs, list) and len(reqs) > 0:
            return "probe", parsed, ""

    raw_sql = parsed.get("sql")
    if isinstance(raw_sql, str) and raw_sql.strip():
        cleaned = clean_sql_text(raw_sql)
        if _is_select_or_with_sql(cleaned):
            return "sql", parsed, cleaned

    fallback_sql = _extract_sql_from_trajectory(text)
    if _is_select_or_with_sql(fallback_sql):
        return "sql", parsed, fallback_sql

    return "invalid", parsed, ""


class MultiTurnSandboxRolloutEngine:
    """Executes batched 2~3 turn interactive rollouts against `WarehouseSandbox` with exact Observation Masking."""

    def __init__(
        self,
        *,
        sandbox: WarehouseSandbox,
        tokenizer: PreTrainedTokenizerBase,
        dialect: DialectType = "odps",
        max_rollout_turns: int = 3,
        max_prompt_length: int = 4096,
        max_completion_length: int = 2048,
        max_per_turn_new_tokens: int = 768,
    ) -> None:
        self.sandbox = sandbox
        self.tokenizer = tokenizer
        self.dialect: DialectType = dialect
        self.max_rollout_turns = max(1, int(max_rollout_turns))
        self.max_prompt_length = max(128, int(max_prompt_length))
        self.max_completion_length = max(64, int(max_completion_length))
        self.max_per_turn_new_tokens = max(32, int(max_per_turn_new_tokens))
        self.stop_ids_list = get_chatml_stop_token_ids(tokenizer)
        self.stop_ids_set = set(self.stop_ids_list)

    def _encode_prompt(self, prompt_text: str) -> list[int]:
        ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        if len(ids) > self.max_prompt_length:
            # Left-truncate only if over max_prompt_length so the trailing `<|im_start|>assistant\n` stays intact
            ids = ids[-self.max_prompt_length :]
        return ids

    def run_episodes(
        self,
        prompts: list[str],
        generate_batch_fn: BatchTokenGeneratorFn,
        *,
        dialects: list[str] | None = None,
    ) -> list[MultiTurnEpisodeRollout]:
        """Run up to `max_rollout_turns` interactive turns across a batch of prompts."""
        n = len(prompts)
        episodes = [MultiTurnEpisodeRollout(prompt_ids=self._encode_prompt(p)) for p in prompts]
        first_sql_recorded = [False] * n
        active_indices = list(range(n))

        for turn_idx in range(1, self.max_rollout_turns + 1):
            if not active_indices:
                break

            # Build full context (prompt + accumulated actions/observations) for all still-active trajectories
            contexts: list[list[int]] = []
            min_remaining_budget = self.max_per_turn_new_tokens
            still_schedulable: list[int] = []

            for i in active_indices:
                ep = episodes[i]
                rem = self.max_completion_length - len(ep.completion_ids)
                if rem < 16:
                    continue
                still_schedulable.append(i)
                contexts.append(ep.prompt_ids + ep.completion_ids)
                min_remaining_budget = min(min_remaining_budget, rem)

            if not still_schedulable:
                break

            turn_budget = min(self.max_per_turn_new_tokens, max(16, min_remaining_budget))
            raw_gen_lists = generate_batch_fn(contexts, turn_budget)

            next_active: list[int] = []
            for i, gen_ids_raw in zip(still_schedulable, raw_gen_lists):
                ep = episodes[i]
                rem = self.max_completion_length - len(ep.completion_ids)
                trimmed_ids, hit_eos = trim_at_first_stop_token(gen_ids_raw[:rem], self.stop_ids_set)
                if not trimmed_ids and self.stop_ids_list:
                    trimmed_ids = [self.stop_ids_list[0]]
                    hit_eos = True

                # Append Action tokens -> action_loss_mask = 1, completion_attn_mask = 1
                act_start = len(ep.completion_ids)
                ep.completion_ids.extend(trimmed_ids)
                ep.action_loss_mask.extend([1] * len(trimmed_ids))
                ep.completion_attn_mask.extend([1] * len(trimmed_ids))
                act_end = len(ep.completion_ids)
                ep.action_spans.append((act_start, act_end))

                act_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
                ep.action_texts.append(act_text)
                ep.final_action_text = act_text
                ep.num_turns += 1
                ep.terminated_with_eos = hit_eos

                row_dialect: DialectType = (
                    str(dialects[i]).lower()  # type: ignore[assignment]
                    if (dialects and i < len(dialects) and dialects[i])
                    else self.dialect
                )
                kind, parsed_obj, cleaned_sql = parse_agent_turn_output(act_text)

                # Case 1: Model requested live table sample inspection (`action="probe"`)
                if kind == "probe" and ep.probe_turns == 0 and turn_idx < self.max_rollout_turns:
                    probe_results: list[dict[str, Any]] = []
                    for req in list(parsed_obj.get("probe_requests") or [])[:3]:
                        if not isinstance(req, dict):
                            continue
                        tname = str(req.get("table_name") or "").strip()
                        cols = req.get("columns")
                        col_list = [str(c) for c in cols] if isinstance(cols, list) else None
                        if tname:
                            probe_results.append(
                                self.sandbox.inspect_table_sample(tname, columns=col_list, sample_rows=2)
                            )
                    if probe_results:
                        obs_str = format_probe_observation_chatml(
                            probe_results,
                            dialect=row_dialect,
                            prepend_im_end=not hit_eos,
                        )
                        obs_ids = list(self.tokenizer.encode(obs_str, add_special_tokens=False))
                        if len(ep.completion_ids) + len(obs_ids) + 32 <= self.max_completion_length:
                            obs_start = len(ep.completion_ids)
                            ep.completion_ids.extend(obs_ids)
                            # STRICT MASK INVARIANT: Observation tokens get loss_mask = 0, attn_mask = 1!
                            ep.action_loss_mask.extend([0] * len(obs_ids))
                            ep.completion_attn_mask.extend([1] * len(obs_ids))
                            ep.observation_spans.append((obs_start, len(ep.completion_ids)))
                            ep.observation_texts.append(obs_str)
                            ep.probe_turns += 1
                            next_active.append(i)
                        continue

                # Case 2: Evaluate SQL against WarehouseSandbox (GatewayLinter + DuckDB)
                if kind == "sql" and cleaned_sql:
                    exec_res = self.sandbox.run_sql(cleaned_sql, dialect=row_dialect)
                    suspicious, susp_reason = self.sandbox.is_suspicious_result(exec_res)
                    gw_ok = bool(exec_res.get("gateway_passed", False))
                    exec_ok = gw_ok and exec_res.get("error_stage", "none") == "none" and not suspicious
                else:
                    exec_res = {
                        "gateway_passed": False,
                        "error_code": "INVALID_FORMAT",
                        "error_stage": "format",
                        "error_msg": "Model output did not contain a valid JSON object with read-only SELECT/WITH SQL.",
                        "row_count": 0,
                    }
                    susp_reason = "Invalid JSON/SQL format"
                    gw_ok = False
                    exec_ok = False

                if not first_sql_recorded[i]:
                    ep.first_pass_gateway = gw_ok
                    ep.had_initial_error = not exec_ok
                    first_sql_recorded[i] = True

                # Clean execution on current turn -> terminate rollout immediately!
                if exec_ok or turn_idx >= self.max_rollout_turns:
                    continue

                # Sandbox intercepted error (e.g., ODPS-0123031) -> inject masked observation for self-repair turn!
                obs_str = format_sandbox_error_observation_chatml(
                    cleaned_sql or act_text,
                    exec_res,
                    suspicion_reason=susp_reason,
                    dialect=row_dialect,
                    prepend_im_end=not hit_eos,
                )
                obs_ids = list(self.tokenizer.encode(obs_str, add_special_tokens=False))
                if len(ep.completion_ids) + len(obs_ids) + 32 <= self.max_completion_length:
                    obs_start = len(ep.completion_ids)
                    ep.completion_ids.extend(obs_ids)
                    # STRICT MASK INVARIANT: Observation tokens get loss_mask = 0, attn_mask = 1!
                    ep.action_loss_mask.extend([0] * len(obs_ids))
                    ep.completion_attn_mask.extend([1] * len(obs_ids))
                    ep.observation_spans.append((obs_start, len(ep.completion_ids)))
                    ep.observation_texts.append(obs_str)
                    ep.refine_turns += 1
                    next_active.append(i)

            active_indices = next_active

        return episodes


def collate_multiturn_rollouts(
    episodes: list[MultiTurnEpisodeRollout],
    *,
    pad_token_id: int,
    device: torch.device | str = "cpu",
    mask_truncated_completions: bool = False,
) -> dict[str, torch.Tensor]:
    """Collate variable-length multi-turn episodes into padded GRPO tensors with separated Loss vs Attention masks.

    Returns:
        - `prompt_ids`: `(B, P)` left-padded with `pad_token_id`
        - `prompt_mask`: `(B, P)` `1` on prompt tokens, `0` on left-pad
        - `completion_ids`: `(B, C)` right-padded with `pad_token_id`
        - `completion_mask`: `(B, C)` **Action Loss Mask** (`1` strictly on Action tokens, `0` on Observations & pads)
        - `completion_attn_mask`: `(B, C)` **Causal Attention Mask** (`1` on Action + Observation tokens, `0` on pads)
    """
    if not episodes:
        raise ValueError("Cannot collate an empty list of MultiTurnEpisodeRollout.")

    max_p = max(len(ep.prompt_ids) for ep in episodes)
    max_c = max(max(1, len(ep.completion_ids)) for ep in episodes)

    p_ids_rows: list[list[int]] = []
    p_mask_rows: list[list[int]] = []
    c_ids_rows: list[list[int]] = []
    c_loss_mask_rows: list[list[int]] = []
    c_attn_mask_rows: list[list[int]] = []

    for ep in episodes:
        assert len(ep.completion_ids) == len(ep.action_loss_mask) == len(ep.completion_attn_mask), (
            "Multi-turn token/mask length mismatch inside episode!"
        )
        p_pad = max_p - len(ep.prompt_ids)
        p_ids_rows.append([pad_token_id] * p_pad + ep.prompt_ids)
        p_mask_rows.append([0] * p_pad + [1] * len(ep.prompt_ids))

        c_ids = ep.completion_ids if ep.completion_ids else [pad_token_id]
        loss_m = list(ep.action_loss_mask) if ep.completion_ids else [0]
        attn_m = list(ep.completion_attn_mask) if ep.completion_ids else [0]

        if mask_truncated_completions and not ep.terminated_with_eos and ep.action_spans:
            # Zero out only the final truncated action span while preserving completed earlier turns!
            last_s, last_e = ep.action_spans[-1]
            for pos in range(last_s, min(last_e, len(loss_m))):
                loss_m[pos] = 0

        c_pad = max_c - len(c_ids)
        c_ids_rows.append(c_ids + [pad_token_id] * c_pad)
        c_loss_mask_rows.append(loss_m + [0] * c_pad)
        c_attn_mask_rows.append(attn_m + [0] * c_pad)

    prompt_ids = torch.tensor(p_ids_rows, dtype=torch.long, device=device)
    prompt_mask = torch.tensor(p_mask_rows, dtype=torch.long, device=device)
    completion_ids = torch.tensor(c_ids_rows, dtype=torch.long, device=device)
    completion_mask = torch.tensor(c_loss_mask_rows, dtype=torch.long, device=device)
    completion_attn_mask = torch.tensor(c_attn_mask_rows, dtype=torch.long, device=device)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completion_attn_mask": completion_attn_mask,
    }
