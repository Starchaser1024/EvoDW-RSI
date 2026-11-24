"""WarehouseLinker module: layered table selection and sandbox sample probing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.agent.trajectory import ModuleInvocation
from src.env.sandbox import WarehouseSandbox
from src.llm.client import BaseLLMClient


@dataclass
class LinkerResult:
    """Output produced by WarehouseLinker."""

    selected_tables: list[str]
    probe_requests: list[dict[str, Any]]
    probe_results: list[dict[str, Any]]
    schema_notes: str
    linked_schema_text: str
    invocations: list[ModuleInvocation]
    turns: list[dict[str, Any]]


def _fallback_select_tables(question: str, hint: str, catalog_table_names: list[str]) -> list[str]:
    text = f"{question} {hint}".lower()
    explicit = [t for t in catalog_table_names if t.lower() in text]
    if explicit:
        return explicit
    return list(catalog_table_names)


class WarehouseLinker:
    """Selects relevant warehouse tables and optionally probes live partitions & sample values."""

    def __init__(self, llm: BaseLLMClient, sandbox: WarehouseSandbox) -> None:
        self.llm = llm
        self.sandbox = sandbox

    async def link(
        self,
        *,
        system_prompt: str,
        question: str,
        hint: str,
        dialect: str,
        step_offset: int = 0,
    ) -> LinkerResult:
        catalog_summary = self.sandbox.catalog.format_catalog_summary()
        valid_tables = [t.name for t in self.sandbox.catalog.list_tables()]
        valid_lower_map = {t.lower(): t for t in valid_tables}

        user_msg = (
            f"Target Dialect: {dialect.upper()}\n"
            f"Business Question: {question}\n"
            f"Business Hint: {hint or 'None'}\n\n"
            f"Warehouse Catalog Summary:\n{catalog_summary}\n\n"
            "Return JSON to either probe table samples (action='probe') or finalize selected tables (action='finalize')."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        invocations: list[ModuleInvocation] = []
        turns: list[dict[str, Any]] = [
            {"role": "user", "module": "linker", "content": user_msg, "loss_mask": 0}
        ]
        probe_requests: list[dict[str, Any]] = []
        probe_results: list[dict[str, Any]] = []

        resp1 = await self.llm.chat_json(messages, role_type="actor", temperature=0.0)
        parsed1 = resp1.parsed_json or {}
        invocations.append(
            ModuleInvocation(
                module_name="linker",
                step_index=step_offset,
                messages=list(messages),
                raw_output=resp1.content,
                parsed_json=parsed1,
                prompt_tokens=resp1.prompt_tokens,
                completion_tokens=resp1.completion_tokens,
                total_tokens=resp1.total_tokens,
                latency_ms=resp1.latency_ms,
            )
        )
        turns.append(
            {
                "role": "assistant",
                "module": "linker",
                "content": resp1.content,
                "loss_mask": 1,
            }
        )

        final_parsed = parsed1
        if str(parsed1.get("action", "")).lower() == "probe" and parsed1.get("probe_requests"):
            raw_reqs = list(parsed1.get("probe_requests") or [])[:4]
            for req in raw_reqs:
                if not isinstance(req, dict):
                    continue
                tname = str(req.get("table_name", "")).strip()
                cols = req.get("columns")
                col_list = [str(c) for c in cols] if isinstance(cols, list) else None
                probe_requests.append({"table_name": tname, "columns": col_list})
                probe_res = self.sandbox.inspect_table_sample(tname, columns=col_list, sample_rows=3)
                probe_results.append(probe_res)

            tool_obs_text = json.dumps(probe_results, ensure_ascii=False, indent=2)
            turns.append(
                {
                    "role": "tool",
                    "name": "inspect_table_sample",
                    "module": "linker",
                    "content": tool_obs_text,
                    "loss_mask": 0,
                }
            )
            messages.append({"role": "assistant", "content": resp1.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation from inspect_table_sample:\n{tool_obs_text}\n\n"
                        "Now return action='finalize' with your selected_tables and concise schema_notes in strict JSON."
                    ),
                }
            )
            resp2 = await self.llm.chat_json(messages, role_type="actor", temperature=0.0)
            final_parsed = resp2.parsed_json or {}
            invocations.append(
                ModuleInvocation(
                    module_name="linker",
                    step_index=step_offset + 1,
                    messages=list(messages),
                    raw_output=resp2.content,
                    parsed_json=final_parsed,
                    prompt_tokens=resp2.prompt_tokens,
                    completion_tokens=resp2.completion_tokens,
                    total_tokens=resp2.total_tokens,
                    latency_ms=resp2.latency_ms,
                )
            )
            turns.append(
                {
                    "role": "assistant",
                    "module": "linker",
                    "content": resp2.content,
                    "loss_mask": 1,
                }
            )

        raw_selected = final_parsed.get("selected_tables") or []
        selected: list[str] = []
        seen: set[str] = set()
        for t in raw_selected:
            t_norm = re.sub(r"[`\"'\s]", "", str(t)).split(".")[-1].lower()
            if t_norm in valid_lower_map and t_norm not in seen:
                seen.add(t_norm)
                selected.append(valid_lower_map[t_norm])

        if not selected:
            selected = _fallback_select_tables(question, hint, valid_tables)

        # Ensure any probed valid tables that were selected (or needed) have detailed DDL.
        ddl_text = self.sandbox.catalog.format_table_ddl(selected)
        schema_notes = str(final_parsed.get("schema_notes", "")).strip()

        parts = [f"### Selected Table Schemas\n{ddl_text}"]
        if probe_results:
            parts.append(
                "### Sample Inspection Observations\n"
                + json.dumps(probe_results, ensure_ascii=False, indent=2)
            )
        if schema_notes:
            parts.append(f"### Linker Schema Notes\n{schema_notes}")

        return LinkerResult(
            selected_tables=selected,
            probe_requests=probe_requests,
            probe_results=probe_results,
            schema_notes=schema_notes,
            linked_schema_text="\n\n".join(parts),
            invocations=invocations,
            turns=turns,
        )
