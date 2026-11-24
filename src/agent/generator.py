"""DialectSQLGenerator module: complex business logic reasoning and dialect SQL synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agent.trajectory import ModuleInvocation
from src.llm.client import BaseLLMClient

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


def clean_sql_text(raw_sql: str) -> str:
    """Normalize SQL extracted from JSON output."""
    text = (raw_sql or "").strip()
    m = _SQL_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    return text.rstrip(";").strip()


@dataclass
class GeneratorResult:
    """Output produced by DialectSQLGenerator."""

    thought: str
    sql: str
    invocation: ModuleInvocation
    turns: list[dict[str, Any]]


class DialectSQLGenerator:
    """Generates ODPS/Hive SQL from linked schema, probes, and business requirements."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    async def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        hint: str,
        dialect: str,
        linked_schema_text: str,
        step_index: int,
    ) -> GeneratorResult:
        user_msg = (
            f"Target Dialect: {dialect.upper()}\n"
            f"Business Question: {question}\n"
            f"Business Hint: {hint or 'None'}\n\n"
            f"{linked_schema_text}\n\n"
            "Generate the single read-only SQL query and return strict JSON: "
            '{"thought": "...", "sql": "..."}.'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        resp = await self.llm.chat_json(messages, role_type="actor", temperature=0.0)
        parsed = resp.parsed_json or {}
        thought = str(parsed.get("thought", "")).strip()
        sql = clean_sql_text(str(parsed.get("sql", "")))

        inv = ModuleInvocation(
            module_name="generator",
            step_index=step_index,
            messages=messages,
            raw_output=resp.content,
            parsed_json=parsed,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            latency_ms=resp.latency_ms,
        )
        turns = [
            {"role": "user", "module": "generator", "content": user_msg, "loss_mask": 0},
            {"role": "assistant", "module": "generator", "content": resp.content, "loss_mask": 1},
        ]
        return GeneratorResult(thought=thought, sql=sql, invocation=inv, turns=turns)
