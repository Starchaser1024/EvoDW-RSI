"""GatewayRefiner module: diagnoses gateway AST blocks, runtime errors, and suspicious results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent.generator import clean_sql_text
from src.agent.trajectory import ModuleInvocation
from src.llm.client import BaseLLMClient


@dataclass
class RefinerResult:
    """Output produced by GatewayRefiner."""

    thought: str
    diagnosis: str
    sql: str
    invocation: ModuleInvocation
    turns: list[dict[str, Any]]


class GatewayRefiner:
    """Iteratively repairs failed or suspicious SQL based on sandbox diagnostics."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    async def refine(
        self,
        *,
        system_prompt: str,
        question: str,
        hint: str,
        dialect: str,
        linked_schema_text: str,
        previous_sql: str,
        exec_res: dict[str, Any],
        suspicion_reason: str,
        prior_refine_steps: list[dict[str, Any]],
        step_index: int,
    ) -> RefinerResult:
        rows_preview = [str(r) for r in (exec_res.get("rows") or [])[:5]]
        feedback_payload = {
            "gateway_passed": exec_res.get("gateway_passed", False),
            "error_stage": exec_res.get("error_stage", "none"),
            "error_msg": exec_res.get("error_msg", ""),
            "suspicion_reason": suspicion_reason,
            "columns": list(exec_res.get("columns") or []),
            "row_count": exec_res.get("row_count", 0),
            "sample_rows": rows_preview,
        }
        history_block = ""
        if prior_refine_steps:
            history_block = (
                "\nPrior Repair Attempts:\n"
                + json.dumps(
                    [
                        {
                            "turn": s["turn"],
                            "diagnosis": s["diagnosis"],
                            "sql": s["sql"],
                            "error": s["exec_res"].get("error_msg") or s["trigger_reason"],
                        }
                        for s in prior_refine_steps
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )

        user_msg = (
            f"Target Dialect: {dialect.upper()}\n"
            f"Business Question: {question}\n"
            f"Business Hint: {hint or 'None'}\n\n"
            f"{linked_schema_text}\n\n"
            f"Previous SQL:\n{previous_sql}\n\n"
            f"Sandbox Execution Feedback:\n{json.dumps(feedback_payload, ensure_ascii=False, indent=2)}"
            f"{history_block}\n\n"
            "Diagnose the root cause and output strict JSON: "
            '{"thought": "...", "diagnosis": "...", "sql": "..."}.'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        resp = await self.llm.chat_json(messages, role_type="actor", temperature=0.0)
        parsed = resp.parsed_json or {}
        thought = str(parsed.get("thought", "")).strip()
        diagnosis = str(parsed.get("diagnosis", "")).strip()
        sql = clean_sql_text(str(parsed.get("sql", ""))) or previous_sql

        inv = ModuleInvocation(
            module_name="refiner",
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
            {
                "role": "tool",
                "name": "sandbox_execute",
                "module": "refiner",
                "content": json.dumps(feedback_payload, ensure_ascii=False),
                "loss_mask": 0,
            },
            {"role": "user", "module": "refiner", "content": user_msg, "loss_mask": 0},
            {"role": "assistant", "module": "refiner", "content": resp.content, "loss_mask": 1},
        ]
        return RefinerResult(
            thought=thought,
            diagnosis=diagnosis,
            sql=sql,
            invocation=inv,
            turns=turns,
        )
