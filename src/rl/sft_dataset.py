"""Loss-masked multi-turn ChatML trajectory dataset, collator, and adaptive scarcity reweighting."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.benchmarks.tasks import get_all_tasks

IGNORE_INDEX = -100


@dataclass
class SFTCorpusStats:
    """Summary statistics of the loaded multi-turn SFT trajectory dataset."""

    total_records: int
    effective_train_episodes: int
    first_pass_correct_count: int
    self_repaired_count: int
    probe_trajectory_count: int
    gateway_error_trajectory_count: int
    effective_self_repaired_episodes: int
    effective_probe_episodes: int
    truncated_trajectory_count: int
    max_observed_seq_len: int
    total_tokens: int
    supervised_assistant_tokens: int
    masked_context_tokens: int

    @property
    def supervised_token_ratio(self) -> float:
        return self.supervised_assistant_tokens / self.total_tokens if self.total_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "effective_train_episodes": self.effective_train_episodes,
            "first_pass_correct_count": self.first_pass_correct_count,
            "self_repaired_count": self.self_repaired_count,
            "probe_trajectory_count": self.probe_trajectory_count,
            "gateway_error_trajectory_count": self.gateway_error_trajectory_count,
            "effective_self_repaired_episodes": self.effective_self_repaired_episodes,
            "effective_probe_episodes": self.effective_probe_episodes,
            "truncated_trajectory_count": self.truncated_trajectory_count,
            "max_observed_seq_len": self.max_observed_seq_len,
            "total_tokens": self.total_tokens,
            "supervised_assistant_tokens": self.supervised_assistant_tokens,
            "masked_context_tokens": self.masked_context_tokens,
            "supervised_token_ratio": round(self.supervised_token_ratio, 4),
        }


def get_isolated_test_task_ids(suite: str = "all") -> set[str]:
    """Return strictly held-out test task IDs (30 in v1; 48 across v1 + hard_v2) that must NEVER enter SFT or GRPO."""
    try:
        tasks = get_all_tasks(suite=suite)  # type: ignore[call-arg]
    except TypeError:
        tasks = get_all_tasks()
    return {t.task_id for t in tasks if t.split == "test"}


def _is_self_repaired_record(rec: dict[str, Any]) -> bool:
    return (
        str(rec.get("trajectory_type", "")) == "self_repaired_after_error"
        or int(rec.get("refine_turns", 0) or 0) > 0
    )


def _has_probe_record(rec: dict[str, Any]) -> bool:
    return any(m.get("name") == "inspect_table_sample" for m in rec.get("messages", []))


def reweight_sft_records(
    records: list[dict[str, Any]],
    *,
    adaptive_reweight: bool = False,
    upsample_self_repaired: int = 1,
    upsample_probe: int = 1,
    target_self_repair_ratio: float = 0.22,
    target_probe_ratio: float = 0.18,
    max_per_record_copies: int = 4,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Expand scarce `self_repaired_after_error` and `inspect_table_sample` trajectories deterministically."""
    if not records:
        return []

    copy_counts = [1 for _ in records]
    repaired_indices = [i for i, r in enumerate(records) if _is_self_repaired_record(r)]
    probe_indices = [i for i, r in enumerate(records) if _has_probe_record(r)]

    # Base explicit integer multipliers
    if upsample_self_repaired > 1:
        for i in repaired_indices:
            copy_counts[i] = max(copy_counts[i], min(max_per_record_copies, upsample_self_repaired))
    if upsample_probe > 1:
        for i in probe_indices:
            copy_counts[i] = max(copy_counts[i], min(max_per_record_copies, upsample_probe))

    if adaptive_reweight:
        rng = random.Random(seed)

        # 1. Adaptively top up self-repaired trajectories toward target_self_repair_ratio
        if repaired_indices and 0.0 < target_self_repair_ratio < 0.85:
            total_now = sum(copy_counts)
            rep_now = sum(copy_counts[i] for i in repaired_indices)
            if rep_now / max(1, total_now) < target_self_repair_ratio:
                # Solve (rep_now + delta) / (total_now + delta) >= target_ratio
                needed = math.ceil(
                    (target_self_repair_ratio * total_now - rep_now) / (1.0 - target_self_repair_ratio)
                )
                order = list(repaired_indices)
                rng.shuffle(order)
                added = 0
                for _ in range(max_per_record_copies):
                    for idx in order:
                        if added >= needed:
                            break
                        if copy_counts[idx] < max_per_record_copies:
                            copy_counts[idx] += 1
                            added += 1
                    if added >= needed:
                        break

        # 2. Adaptively top up inspect_table_sample probe trajectories toward target_probe_ratio
        if probe_indices and 0.0 < target_probe_ratio < 0.85:
            total_now = sum(copy_counts)
            prb_now = sum(copy_counts[i] for i in probe_indices)
            if prb_now / max(1, total_now) < target_probe_ratio:
                needed = math.ceil(
                    (target_probe_ratio * total_now - prb_now) / (1.0 - target_probe_ratio)
                )
                order = list(probe_indices)
                rng.shuffle(order)
                added = 0
                for _ in range(max_per_record_copies):
                    for idx in order:
                        if added >= needed:
                            break
                        if copy_counts[idx] < max_per_record_copies:
                            copy_counts[idx] += 1
                            added += 1
                    if added >= needed:
                        break

    expanded: list[dict[str, Any]] = []
    for rec, n_copies in zip(records, copy_counts):
        for _ in range(n_copies):
            expanded.append(rec)
    return expanded


def load_sft_trajectories(
    jsonl_path: Path | str,
    *,
    strict_no_test_leakage: bool = True,
) -> list[dict[str, Any]]:
    """Load `sft_openai_tool_use.jsonl` and enforce strict train/val split isolation."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT trajectory file not found: {path}")

    forbidden_test_ids = get_isolated_test_task_ids() if strict_no_test_leakage else set()
    records: list[dict[str, Any]] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        split = str(obj.get("split", ""))
        task_id = str(obj.get("task_id", ""))

        if strict_no_test_leakage:
            if split == "test" or task_id in forbidden_test_ids:
                raise ValueError(
                    f"Data leakage blocked at {path}:{line_no}! Found held-out test task "
                    f"task_id={task_id!r} (split={split!r}) in SFT training data."
                )
            if split not in ("train", "val"):
                raise ValueError(
                    f"Unexpected split={split!r} at {path}:{line_no}; only 'train' and 'val' allowed."
                )

        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Trajectory at {path}:{line_no} has empty or invalid 'messages'.")
        records.append(obj)

    if not records:
        raise ValueError(f"No valid SFT trajectories found in {path}.")
    return records


def format_non_supervised_turn_chatml(turn: dict[str, Any]) -> str:
    """Format a masked (`loss_mask == 0`) system/user/tool turn into Qwen ChatML text."""
    role = str(turn.get("role", "user")).strip().lower()
    content = str(turn.get("content", ""))
    if role == "tool":
        tool_name = str(turn.get("name") or turn.get("module") or "tool")
        return (
            f"<|im_start|>user\n<tool_response name=\"{tool_name}\">\n"
            f"{content}\n</tool_response><|im_end|>\n"
        )
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _get_tokenizer_turn_cache(tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    cache = getattr(tokenizer, "_evodw_turn_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(tokenizer, "_evodw_turn_cache", cache)
        except Exception:
            return {}
    elif len(cache) > 4096:
        cache.clear()
    return cache


def encode_multiturn_with_loss_mask(
    messages: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 9216,
) -> dict[str, Any]:
    """Tokenize a multi-turn trajectory with exact token-level `loss_mask` supervision.

    Invariants:
    - Every token from any `loss_mask == 0` turn (`system`, `user`, `tool` including
      `inspect_table_sample` payloads and `ODPS-0123031` gateway errors) gets `labels = -100`.
    - For `loss_mask == 1` (`assistant` turns), the `<|im_start|>assistant\\n` role header and
      trailing newline get `labels = -100`, while `content + <|im_end|>` gets `labels = input_ids`.
    """
    turn_cache = _get_tokenizer_turn_cache(tokenizer)
    header_key = "<|im_start|>assistant\n"
    nl_key = "\n"
    header_ids = turn_cache.get(header_key)
    if header_ids is None:
        header_ids = list(tokenizer.encode(header_key, add_special_tokens=False))
        turn_cache[header_key] = header_ids
    nl_ids = turn_cache.get(nl_key)
    if nl_ids is None:
        nl_ids = list(tokenizer.encode(nl_key, add_special_tokens=False))
        turn_cache[nl_key] = nl_ids

    header_mask = [IGNORE_INDEX] * len(header_ids)
    nl_mask = [IGNORE_INDEX] * len(nl_ids)

    turn_specs: list[tuple[list[int], int]] = []
    untruncated_sup_tokens = 0
    raw_untruncated_len = 0

    for turn in messages:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", ""))
        raw_mask = turn.get("loss_mask")
        loss_mask = int(raw_mask) if raw_mask is not None else (1 if role == "assistant" else 0)

        if loss_mask == 1 and role == "assistant":
            body_text = f"{content}<|im_end|>"
            body_ids = turn_cache.get(body_text)
            if body_ids is None:
                body_ids = list(tokenizer.encode(body_text, add_special_tokens=False))
                if len(body_text) <= 8192:
                    turn_cache[body_text] = body_ids

            untruncated_sup_tokens += len(body_ids)
            raw_untruncated_len += len(header_ids) + len(body_ids) + len(nl_ids)
            turn_specs.append((body_ids, 1))
        else:
            full_text = format_non_supervised_turn_chatml(turn)
            t_input_cached = turn_cache.get(full_text)
            if t_input_cached is None:
                t_input_cached = list(tokenizer.encode(full_text, add_special_tokens=False))
                if len(full_text) <= 32768:
                    turn_cache[full_text] = t_input_cached
            raw_untruncated_len += len(t_input_cached)
            turn_specs.append((t_input_cached, 0))

    was_truncated = False
    sup_token_count = untruncated_sup_tokens

    # Fast single-pass construction when within max_seq_length (zero intermediate label lists).
    if raw_untruncated_len <= max_seq_length:
        input_ids: list[int] = []
        labels: list[int] = []
        for span_ids, is_sup in turn_specs:
            if is_sup == 1:
                input_ids.extend(header_ids)
                input_ids.extend(span_ids)
                input_ids.extend(nl_ids)
                labels.extend(header_mask)
                labels.extend(span_ids)
                labels.extend(nl_mask)
            else:
                input_ids.extend(span_ids)
                labels.extend([IGNORE_INDEX] * len(span_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "was_truncated": False,
            "raw_token_length": raw_untruncated_len,
            "supervised_token_count": sup_token_count,
        }

    # Materialize per-turn chunks only when truncation is required.
    turn_chunks: list[tuple[list[int], list[int], int]] = []
    for span_ids, is_sup in turn_specs:
        if is_sup == 1:
            turn_chunks.append(
                (header_ids + span_ids + nl_ids, header_mask + span_ids + nl_mask, 1)
            )
        else:
            turn_chunks.append((span_ids, [IGNORE_INDEX] * len(span_ids), 0))

    trimmed_chunks = list(turn_chunks)
    while len(trimmed_chunks) > 1 and trimmed_chunks[-1][2] == 0:
        trimmed_chunks.pop()
        cur_len = sum(len(c[0]) for c in trimmed_chunks)
        if cur_len <= max_seq_length:
            break

    # 2. If still over max_seq_length, preserve system turn + rightmost tail turns
    #    so self-repair (`refiner`) and SQL (`generator`) assistant turns are never dropped!
    cur_len = sum(len(c[0]) for c in trimmed_chunks)
    if cur_len > max_seq_length and len(trimmed_chunks) > 2:
        was_truncated = True
        head_chunk = trimmed_chunks[0]
        tail_chunks: list[tuple[list[int], list[int], int]] = []
        budget = max_seq_length - len(head_chunk[0])
        for ch in reversed(trimmed_chunks[1:]):
            if sum(len(x[0]) for x in tail_chunks) + len(ch[0]) <= budget:
                tail_chunks.append(ch)
            else:
                break
        if tail_chunks and any(c[2] == 1 for c in tail_chunks):
            trimmed_chunks = [head_chunk] + list(reversed(tail_chunks))

    input_ids = []
    labels = []
    for t_in, t_lb, _ in trimmed_chunks:
        input_ids.extend(t_in)
        labels.extend(t_lb)

    # Final hard cap if a single huge turn still exceeds max_seq_length.
    if len(input_ids) > max_seq_length:
        was_truncated = True
        left_labels = labels[:max_seq_length]
        if any(x != IGNORE_INDEX for x in left_labels):
            input_ids = input_ids[:max_seq_length]
            labels = left_labels
        else:
            input_ids = input_ids[-max_seq_length:]
            labels = labels[-max_seq_length:]

    if was_truncated:
        sup_token_count = sum(1 for y in labels if y != IGNORE_INDEX)

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "was_truncated": was_truncated,
        "raw_token_length": raw_untruncated_len,
        "supervised_token_count": sup_token_count,
    }


class MultiTurnSFTDataset(Dataset):
    """PyTorch Dataset wrapping pre-tokenized loss-masked multi-turn warehouse trajectories."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_length: int = 9216,
        upsample_self_repaired: int = 1,
        upsample_probe: int = 1,
        adaptive_reweight: bool = False,
        target_self_repair_ratio: float = 0.22,
        target_probe_ratio: float = 0.18,
        seed: int = 2026,
    ) -> None:
        self.raw_records = records
        self.examples: list[dict[str, Any]] = []

        expanded = reweight_sft_records(
            records,
            adaptive_reweight=adaptive_reweight,
            upsample_self_repaired=upsample_self_repaired,
            upsample_probe=upsample_probe,
            target_self_repair_ratio=target_self_repair_ratio,
            target_probe_ratio=target_probe_ratio,
            seed=seed,
        )

        total_toks = 0
        sup_toks = 0
        truncated_n = 0
        max_obs_len = 0
        encoded_by_rec_id: dict[int, tuple[dict[str, Any], int, bool, int]] = {}
        for rec in expanded:
            rec_key = id(rec)
            cached_rec = encoded_by_rec_id.get(rec_key)
            if cached_rec is None:
                encoded = encode_multiturn_with_loss_mask(
                    rec["messages"],
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                )
                sup_cnt = int(encoded["supervised_token_count"])
                if sup_cnt == 0:
                    raise ValueError(
                        f"Trajectory {rec.get('id')} produced 0 supervised tokens after encoding!"
                    )
                example = {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["labels"],
                }
                cached_rec = (
                    example,
                    sup_cnt,
                    bool(encoded.get("was_truncated")),
                    len(encoded["input_ids"]),
                )
                encoded_by_rec_id[rec_key] = cached_rec
            else:
                example, sup_cnt, _, _ = cached_rec

            if cached_rec[2]:
                truncated_n += 1
            seq_len = cached_rec[3]
            if seq_len > max_obs_len:
                max_obs_len = seq_len
            total_toks += seq_len
            sup_toks += sup_cnt
            self.examples.append(example)

        first_pass_n = sum(1 for r in records if not _is_self_repaired_record(r))
        repaired_n = sum(1 for r in records if _is_self_repaired_record(r))
        probe_n = sum(1 for r in records if _has_probe_record(r))
        gw_err_n = sum(
            1
            for r in records
            if any("ODPS-" in str(m.get("content", "")) and m.get("role") == "tool" for m in r.get("messages", []))
        )
        eff_repaired_n = sum(1 for r in expanded if _is_self_repaired_record(r))
        eff_probe_n = sum(1 for r in expanded if _has_probe_record(r))

        self.stats = SFTCorpusStats(
            total_records=len(records),
            effective_train_episodes=len(self.examples),
            first_pass_correct_count=first_pass_n,
            self_repaired_count=repaired_n,
            probe_trajectory_count=probe_n,
            gateway_error_trajectory_count=gw_err_n,
            effective_self_repaired_episodes=eff_repaired_n,
            effective_probe_episodes=eff_probe_n,
            truncated_trajectory_count=truncated_n,
            max_observed_seq_len=max_obs_len,
            total_tokens=total_toks,
            supervised_assistant_tokens=sup_toks,
            masked_context_tokens=total_toks - sup_toks,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class LossMaskedMultiTurnCollator:
    """Data collator enforcing `labels = -100` on all `loss_mask == 0` turns and pad positions."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 9216,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_to_multiple_of = pad_to_multiple_of
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_batch: list[dict[str, list[int]]] = []
        for feat in features:
            if "input_ids" in feat and "labels" in feat:
                encoded_batch.append(
                    {
                        "input_ids": list(feat["input_ids"]),
                        "attention_mask": list(feat.get("attention_mask") or [1] * len(feat["input_ids"])),
                        "labels": list(feat["labels"]),
                    }
                )
            elif "messages" in feat:
                enc = encode_multiturn_with_loss_mask(
                    feat["messages"],
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                )
                encoded_batch.append(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": enc["labels"],
                    }
                )
            else:
                raise KeyError("Feature must contain either ('input_ids', 'labels') or 'messages'.")

        max_len = max(len(x["input_ids"]) for x in encoded_batch)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = int(self.tokenizer.pad_token_id)
        batch_input_ids: list[list[int]] = []
        batch_attn: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for item in encoded_batch:
            pad_n = max_len - len(item["input_ids"])
            if self.tokenizer.padding_side == "left":
                batch_input_ids.append([pad_id] * pad_n + item["input_ids"])
                batch_attn.append([0] * pad_n + item["attention_mask"])
                batch_labels.append([IGNORE_INDEX] * pad_n + item["labels"])
            else:
                batch_input_ids.append(item["input_ids"] + [pad_id] * pad_n)
                batch_attn.append(item["attention_mask"] + [0] * pad_n)
                batch_labels.append(item["labels"] + [IGNORE_INDEX] * pad_n)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
