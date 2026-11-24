"""Dual-role async OpenAI-compatible LLM client with concurrency pool, truncation recovery, and token telemetry."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import openai

from src.config import Settings

RoleType = Literal["actor", "reflector"]

_PLACEHOLDER_KEYS = frozenset(
    {
        "",
        "your_api_key_here",
        "your_llm_api_key_here",
        "your_openai_api_key_here",
        "your_opencode_api_key_here",
    }
)


class MissingAPIKeyError(RuntimeError):
    """Raised when attempting a live LLM API call without a valid API key (`LLM_API_KEY` / `OPENAI_API_KEY`)."""


class JSONParseError(ValueError):
    """Raised when an expected JSON response cannot be parsed or was truncated."""


@dataclass
class LLMResponse:
    """Structured response and telemetry from a single logical LLM invocation."""

    content: str
    parsed_json: dict[str, Any] | None
    model: str
    role_type: RoleType
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    retries: int = 0
    session_id: str = ""


@dataclass
class LLMCallStats:
    """Aggregated call metrics for a role."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    retries: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0


@dataclass
class UsageLedger:
    """Tracks token and latency budgets separately for actor and reflector models."""

    actor: LLMCallStats = field(default_factory=LLMCallStats)
    reflector: LLMCallStats = field(default_factory=LLMCallStats)

    def record(self, resp: LLMResponse) -> None:
        bucket = self.actor if resp.role_type == "actor" else self.reflector
        bucket.calls += 1
        bucket.prompt_tokens += resp.prompt_tokens
        bucket.completion_tokens += resp.completion_tokens
        bucket.total_tokens += resp.total_tokens
        bucket.total_latency_ms += resp.latency_ms
        bucket.retries += resp.retries

    @property
    def total_tokens(self) -> int:
        return self.actor.total_tokens + self.reflector.total_tokens

    @property
    def total_calls(self) -> int:
        return self.actor.calls + self.reflector.calls

    def summary(self) -> dict[str, Any]:
        return {
            "actor": {
                "calls": self.actor.calls,
                "prompt_tokens": self.actor.prompt_tokens,
                "completion_tokens": self.actor.completion_tokens,
                "total_tokens": self.actor.total_tokens,
                "avg_latency_ms": round(self.actor.avg_latency_ms, 2),
                "retries": self.actor.retries,
            },
            "reflector": {
                "calls": self.reflector.calls,
                "prompt_tokens": self.reflector.prompt_tokens,
                "completion_tokens": self.reflector.completion_tokens,
                "total_tokens": self.reflector.total_tokens,
                "avg_latency_ms": round(self.reflector.avg_latency_ms, 2),
                "retries": self.reflector.retries,
            },
            "grand_total_tokens": self.total_tokens,
            "grand_total_calls": self.total_calls,
        }


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON dictionary from LLM output, tolerating reasoning blocks and code fences."""
    if not raw_text or not raw_text.strip():
        raise JSONParseError("Empty response content when expecting JSON.")

    cleaned = _THINK_TAG_RE.sub("", raw_text).strip()

    # Try fenced code block contents first if present.
    fence_matches = _CODE_FENCE_RE.findall(cleaned)
    candidates = [m.strip() for m in fence_matches if m.strip()] + [cleaned]

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Fall back to scanning balanced `{ ... }` objects from first `{`.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = cleaned[start : idx + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)

    raise JSONParseError(f"Could not parse valid JSON object from response: {cleaned[:240]!r}")


class BaseLLMClient(ABC):
    """Abstract async LLM interface supporting live OpenAI-compatible gateways and deterministic offline mocks."""

    def __init__(self, actor_model: str, reflector_model: str) -> None:
        self.actor_model = actor_model
        self.reflector_model = reflector_model
        self.ledger = UsageLedger()

    def resolve_model(self, role_type: RoleType, model_override: str | None = None) -> str:
        if model_override:
            return model_override
        return self.actor_model if role_type == "actor" else self.reflector_model

    @abstractmethod
    async def chat_json(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        session_id: str | None = None,
    ) -> LLMResponse:
        """Invoke chat completion expecting a strict JSON object response."""

    @abstractmethod
    async def chat_text(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        session_id: str | None = None,
    ) -> LLMResponse:
        """Invoke chat completion returning raw text."""


class DualRoleLLMClient(BaseLLMClient):
    """Dual-role async OpenAI-compatible client with concurrency pool, adaptive truncation recovery, and telemetry."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        actor_model: str | None = None,
        reflector_model: str | None = None,
        max_concurrency: int | None = None,
        timeout_sec: float | None = None,
        max_retries: int = 4,
        base_backoff_sec: float = 1.0,
        session_id: str | None = None,
        response_cache_path: Path | None = None,
    ) -> None:
        cfg = settings or Settings.from_env()
        super().__init__(
            actor_model=actor_model or cfg.actor_model,
            reflector_model=reflector_model or cfg.reflector_model,
        )
        self.api_key = (api_key if api_key is not None else cfg.api_key).strip()
        self.base_url = (base_url or cfg.base_url).strip()
        self.max_concurrency = max_concurrency if max_concurrency is not None else cfg.max_concurrency
        self.timeout_sec = timeout_sec if timeout_sec is not None else max(cfg.timeout_sec, 150.0)
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self._explicit_session_id = session_id is not None
        self.session_id = session_id or str(uuid.uuid4())
        self._semaphore = asyncio.Semaphore(max(1, self.max_concurrency))
        self.response_cache_path = response_cache_path
        self._reflector_cache: dict[str, dict[str, Any]] = {}
        if self.response_cache_path and self.response_cache_path.exists():
            for line in self.response_cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                        self._reflector_cache[str(row["key"])] = dict(row["resp"])
                    except Exception:
                        pass
        self._client: openai.AsyncOpenAI | None = None

        if self._has_usable_key():
            self._client = self._build_async_client()

    def _has_usable_key(self) -> bool:
        return bool(self.api_key and self.api_key not in _PLACEHOLDER_KEYS)

    def _use_session_header(self, session_override: str | None = None) -> bool:
        return (
            "opencode.ai" in self.base_url.lower()
            or self._explicit_session_id
            or session_override is not None
        )

    def _build_async_client(self) -> openai.AsyncOpenAI:
        default_headers = (
            {"x-opencode-session": self.session_id} if self._use_session_header() else None
        )
        return openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers,
            timeout=self.timeout_sec,
            max_retries=0,
        )

    def _ensure_client(self) -> openai.AsyncOpenAI:
        if self._client is not None:
            return self._client
        if not self._has_usable_key():
            raise MissingAPIKeyError(
                "LLM_API_KEY / OPENAI_API_KEY is not configured. Copy .env.example to .env and set LLM_API_KEY "
                "before running live API calls (or pass --mock for deterministic offline execution)."
            )
        self._client = self._build_async_client()
        return self._client

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        session_id: str | None = None,
    ) -> LLMResponse:
        return await self._chat_impl(
            messages=messages,
            role_type=role_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            expect_json=True,
            session_id=session_id,
        )

    async def chat_text(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        session_id: str | None = None,
    ) -> LLMResponse:
        return await self._chat_impl(
            messages=messages,
            role_type=role_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            expect_json=False,
            session_id=session_id,
        )

    async def _chat_impl(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType,
        model: str | None,
        temperature: float,
        max_tokens: int,
        expect_json: bool,
        session_id: str | None,
    ) -> LLMResponse:
        resolved_model = self.resolve_model(role_type, model)
        cache_key = ""
        if role_type == "reflector" and self.response_cache_path is not None:
            raw_k = json.dumps(
                {
                    "model": resolved_model,
                    "temperature": temperature,
                    "expect_json": expect_json,
                    "messages": messages,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            cache_key = hashlib.sha256(raw_k.encode("utf-8")).hexdigest()[:32]
            if cache_key in self._reflector_cache:
                cached = self._reflector_cache[cache_key]
                resp_cached = LLMResponse(
                    content=str(cached["content"]),
                    parsed_json=cached.get("parsed_json"),
                    model=str(cached["model"]),
                    role_type=role_type,
                    prompt_tokens=int(cached["prompt_tokens"]),
                    completion_tokens=int(cached["completion_tokens"]),
                    total_tokens=int(cached["total_tokens"]),
                    latency_ms=float(cached["latency_ms"]),
                    finish_reason=str(cached.get("finish_reason", "stop")),
                    retries=int(cached.get("retries", 0)),
                    session_id=str(cached.get("session_id", self.session_id)),
                )
                self.ledger.record(resp_cached)
                return resp_cached

        client = self._ensure_client()
        active_session_id = session_id or self.session_id
        inject_session = self._use_session_header(session_id)
        current_max_tokens = max_tokens
        current_messages = [dict(m) for m in messages]
        use_response_format = expect_json

        total_prompt_tokens = 0
        total_completion_tokens = 0
        t_start = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    kwargs: dict[str, Any] = {
                        "model": resolved_model,
                        "messages": current_messages,
                        "temperature": temperature,
                        "max_tokens": current_max_tokens,
                    }
                    if inject_session:
                        kwargs["extra_headers"] = {"x-opencode-session": active_session_id}
                    if use_response_format:
                        kwargs["response_format"] = {"type": "json_object"}
                    raw = await client.chat.completions.create(**kwargs)

                choice = raw.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason or "stop"
                usage = getattr(raw, "usage", None)
                if usage is not None:
                    total_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    total_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

                parsed: dict[str, Any] | None = None
                if expect_json:
                    if finish_reason == "length":
                        try:
                            parsed = extract_json_object(content)
                        except JSONParseError as exc:
                            raise JSONParseError(
                                f"JSON output truncated (finish_reason='length') at max_tokens={current_max_tokens}"
                            ) from exc
                    else:
                        parsed = extract_json_object(content)

                latency_ms = (time.perf_counter() - t_start) * 1000.0
                resp = LLMResponse(
                    content=content,
                    parsed_json=parsed,
                    model=resolved_model,
                    role_type=role_type,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    latency_ms=latency_ms,
                    finish_reason=finish_reason,
                    retries=attempt,
                    session_id=active_session_id,
                )
                self.ledger.record(resp)
                if cache_key and self.response_cache_path is not None:
                    entry = {
                        "content": resp.content,
                        "parsed_json": resp.parsed_json,
                        "model": resp.model,
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "total_tokens": resp.total_tokens,
                        "latency_ms": resp.latency_ms,
                        "finish_reason": resp.finish_reason,
                        "retries": resp.retries,
                        "session_id": resp.session_id,
                    }
                    self._reflector_cache[cache_key] = entry
                    self.response_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.response_cache_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": cache_key, "resp": entry}, ensure_ascii=False) + "\n")
                return resp

            except JSONParseError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                current_max_tokens = min(int(current_max_tokens * 1.5), 16384)
                current_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was truncated or not valid JSON. "
                            "Respond ONLY with a single complete, valid JSON object without extra prose."
                        ),
                    }
                ]
                await self._backoff_sleep(attempt)

            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await self._backoff_sleep(attempt)

            except openai.APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", 0)
                if status == 400 and use_response_format and "response_format" in str(exc).lower():
                    use_response_format = False
                    continue
                if status == 429 or status >= 500:
                    if attempt >= self.max_retries:
                        break
                    await self._backoff_sleep(attempt)
                else:
                    raise

        raise RuntimeError(
            f"DualRoleLLMClient failed after {self.max_retries + 1} attempts "
            f"(model={resolved_model}, session={active_session_id}): {last_error}"
        ) from last_error

    async def _backoff_sleep(self, attempt: int) -> None:
        delay = self.base_backoff_sec * (2**attempt) + random.uniform(0.0, 0.5)
        await asyncio.sleep(delay)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# Backwards-compatible alias for existing callers and OpenCode Go gateway setups.
OpenCodeGoClient = DualRoleLLMClient
