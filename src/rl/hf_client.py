"""Local HuggingFace CausalLM client with dynamic GPU micro-batching."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch

from src.export.rl_bridge import _extract_sql_from_trajectory
from src.llm.client import BaseLLMClient, JSONParseError, LLMResponse, RoleType, extract_json_object
from src.rl.sft_dataset import format_non_supervised_turn_chatml


class LocalHFLLMClient(BaseLLMClient):
    """Adapts a local HuggingFace CausalLM (Base / LoRA / Merged) to `BaseLLMClient` with dynamic GPU micro-batching."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_label: str = "qwen2.5-coder-local",
        max_new_tokens_cap: int | None = None,
        max_batch_size: int = 10,
    ) -> None:
        super().__init__(actor_model=model_label, reflector_model=model_label)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_new_tokens_cap = max_new_tokens_cap
        self.max_batch_size = max(1, max_batch_size)
        self._pending_queue: list[tuple[str, int, float, asyncio.Future[tuple[str, int, int, float]]]] = []
        self._flushing: bool = False

    def _get_eos_ids(self) -> list[int]:
        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        return eos_ids

    def generate_batch_from_chatml_prompts(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 896,
        temperature: float = 0.0,
    ) -> list[tuple[str, int, int, float]]:
        """Run batched left-padded GPU generation across independent prompts without tail truncation."""
        if not prompts:
            return []
        if self.max_new_tokens_cap is not None:
            max_new_tokens = min(max_new_tokens, self.max_new_tokens_cap)

        t0 = time.perf_counter()
        prev_pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = prev_pad_side

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        padded_len = int(input_ids.shape[1])
        prompt_lens = [int(attention_mask[i].sum().item()) for i in range(len(prompts))]

        eos_ids = self._get_eos_ids()
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
            "use_cache": True,
        }
        if temperature > 1e-5:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_item_latency_ms = batch_latency_ms / len(prompts)

        eos_set = set(eos_ids)
        results: list[tuple[str, int, int, float]] = []
        for i in range(len(prompts)):
            row_ids = out_ids[i, padded_len:].tolist()
            stop_idx = len(row_ids)
            for pos, tok_id in enumerate(row_ids):
                if tok_id in eos_set:
                    stop_idx = pos + 1
                    break
            trimmed_ids = row_ids[:stop_idx]
            text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
            results.append((text, prompt_lens[i], len(trimmed_ids), per_item_latency_ms))
        return results

    def generate_from_chatml_prompt(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 896,
        temperature: float = 0.0,
    ) -> tuple[str, int, int, float]:
        """Run single-prompt generation (delegates to batched implementation)."""
        return self.generate_batch_from_chatml_prompts(
            [prompt_text],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]

    async def async_generate_from_chatml_prompt(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 896,
        temperature: float = 0.0,
    ) -> tuple[str, int, int, float]:
        """Dynamically coalesce concurrent coroutine generation calls into single GPU batches."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[str, int, int, float]] = loop.create_future()
        self._pending_queue.append((prompt_text, max_new_tokens, temperature, fut))

        if not self._flushing:
            self._flushing = True
            loop.create_task(self._drain_queue())

        return await fut

    async def _drain_queue(self) -> None:
        try:
            await asyncio.sleep(0.015)
            while self._pending_queue:
                batch = self._pending_queue[: self.max_batch_size]
                del self._pending_queue[: self.max_batch_size]
                prompts = [item[0] for item in batch]
                max_toks = max(item[1] for item in batch)
                temp = max(item[2] for item in batch)
                try:
                    outs = self.generate_batch_from_chatml_prompts(
                        prompts,
                        max_new_tokens=max_toks,
                        temperature=temp,
                    )
                    for (_, _, _, f), res in zip(batch, outs):
                        if not f.done():
                            f.set_result(res)
                except Exception as exc:
                    for _, _, _, f in batch:
                        if not f.done():
                            f.set_exception(exc)
                await asyncio.sleep(0)
        finally:
            self._flushing = False

    def _format_messages_chatml(self, messages: list[dict[str, str]]) -> str:
        chunks = [format_non_supervised_turn_chatml(m) for m in messages]
        chunks.append("<|im_start|>assistant\n")
        return "".join(chunks)

    async def chat_text(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session_id: str | None = None,
    ) -> LLMResponse:
        _ = session_id
        prompt_text = self._format_messages_chatml(messages)
        content, p_toks, c_toks, lat_ms = await self.async_generate_from_chatml_prompt(
            prompt_text,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        resp = LLMResponse(
            content=content,
            parsed_json=None,
            model=self.resolve_model(role_type, model),
            role_type=role_type,
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            total_tokens=p_toks + c_toks,
            latency_ms=lat_ms,
        )
        self.ledger.record(resp)
        return resp

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session_id: str | None = None,
    ) -> LLMResponse:
        _ = session_id
        prompt_text = self._format_messages_chatml(messages)
        content, p_toks, c_toks, lat_ms = await self.async_generate_from_chatml_prompt(
            prompt_text,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        parsed: dict[str, Any] = {}
        try:
            parsed = extract_json_object(content)
        except JSONParseError:
            salvaged_sql = _extract_sql_from_trajectory(content)
            if salvaged_sql:
                parsed = {"thought": "salvaged_from_raw_text", "sql": salvaged_sql}

        resp = LLMResponse(
            content=content,
            parsed_json=parsed,
            model=self.resolve_model(role_type, model),
            role_type=role_type,
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            total_tokens=p_toks + c_toks,
            latency_ms=lat_ms,
        )
        self.ledger.record(resp)
        return resp
