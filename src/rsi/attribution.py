"""Reflector: implicit module-level fault attribution, prompt mutation, and anti-cheat guards."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.policy import ModuleName, Policy
from src.agent.trajectory import AgentTrajectory
from src.llm.client import BaseLLMClient
from src.rsi.dialect_guards import (
    HIVE_FORBIDDEN_FUNCTION_SPECS,
    ODPS_FORBIDDEN_FUNCTION_SPECS,
    detect_prompt_dialect_violations,
    repair_prompt_dialect_violations,
)

FORBIDDEN_TABLE_NAMES = (
    "dim_user",
    "dim_shop",
    "dim_item",
    "dim_content_creator",
    "ods_app_event_log_di",
    "dwd_trade_order_di",
    "dwd_refund_di",
    "dwd_content_post_di",
    "dws_shop_metrics_1d",
    "dwd_trade_order_legacy_di",
    "dwd_order_settle_di",
    "dwd_aftersale_refund_di",
    "dim_user_vip_scd2",
    "dim_shop_commission_scd2",
    "dws_shop_operating_1d",
)

_FORBIDDEN_TASK_ID_RE = re.compile(
    r"\b(?:pp|ja|rd|fd|rn|sl|scd|sc)_(?:0[1-9]|1[0-5])\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_ENUM_CODE_LITERAL_RE = re.compile(
    r"\b(?:10_WAIT_CLEAR|20_SETTLED|90_REVERSED|P01_WECHAT|P02_ALIPAY|P03_CREDIT|"
    r"01_APPLY|02_REFUND_OK|03_REJECTED|RT01_ONLY_REFUND|RT02_RETURN_GOODS|RT03_PRICE_PROTECT|"
    r"OP1_OPEN|OP2_PAUSED)\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_DATE_LITERAL_RE = re.compile(
    r"(?:202609(?:2[5-9]|30)|202610(?:0[1-9]|1[0-5])|2026-09-(?:2[5-9]|30)|2026-10-(?:0[1-9]|1[0-5]))"
)

_REQUIRED_JSON_CONTRACTS: dict[ModuleName, tuple[str, ...]] = {
    "linker": ("selected_tables", "action"),
    "generator": ("sql", "thought"),
    "refiner": ("diagnosis", "sql"),
}


def validate_and_sanitize_prompt(
    prompt_text: str,
    target_module: ModuleName,
    *,
    dialect: str = "odps",
    auto_repair_dialect: bool = True,
) -> tuple[bool, str, str]:
    """Validate mutated prompt against benchmark leakage, JSON contract breaks, and dialect hallucinations."""
    cleaned = (prompt_text or "").strip()
    if len(cleaned) < 40:
        return False, cleaned, "Mutated prompt is suspiciously short (<40 chars)."

    lower = cleaned.lower()
    leaked_tables = [tbl for tbl in FORBIDDEN_TABLE_NAMES if re.search(rf"\b{re.escape(tbl)}\b", lower)]
    if leaked_tables:
        return False, cleaned, f"Prompt leaked benchmark table name(s): {leaked_tables}."

    if _FORBIDDEN_TASK_ID_RE.search(cleaned):
        return False, cleaned, "Prompt leaked benchmark task ID literal."

    if _FORBIDDEN_DATE_LITERAL_RE.search(cleaned):
        return False, cleaned, "Prompt leaked benchmark partition date literal."

    m_enum = _FORBIDDEN_ENUM_CODE_LITERAL_RE.search(cleaned)
    if m_enum:
        return (
            False,
            cleaned,
            f"Prompt leaked hardcoded warehouse enum code literal '{m_enum.group(0)}'; "
            "instruct Linker to discover codes via inspect_table_sample.",
        )

    required_keys = _REQUIRED_JSON_CONTRACTS[target_module]
    missing = [k for k in required_keys if k not in lower]
    if missing:
        return False, cleaned, f"Prompt dropped required JSON contract keys {missing} for {target_module}."

    dialect_violations = detect_prompt_dialect_violations(cleaned, dialect=dialect)
    if dialect_violations:
        if not auto_repair_dialect:
            return (
                False,
                cleaned,
                f"Prompt contains forbidden {dialect.upper()} function(s): {dialect_violations}.",
            )
        repaired_text, repaired_list = repair_prompt_dialect_violations(cleaned, dialect=dialect)
        remaining = detect_prompt_dialect_violations(repaired_text, dialect=dialect)
        if remaining:
            return (
                False,
                repaired_text,
                f"Prompt retained unresolved {dialect.upper()} dialect violation(s): {remaining}.",
            )
        return (
            True,
            repaired_text,
            f"Auto-repaired {dialect.upper()} dialect violation(s): {repaired_list}.",
        )

    return True, cleaned, ""


def sanitize_rule_text(text: str, dialect: str = "odps") -> str:
    """Replace accidentally leaked table/date literals and repair dialect-invalid functions in distilled rules."""
    sanitized = text
    for tbl in FORBIDDEN_TABLE_NAMES:
        sanitized = re.sub(rf"\b{re.escape(tbl)}\b", "<warehouse_table>", sanitized, flags=re.IGNORECASE)
    sanitized = _FORBIDDEN_TASK_ID_RE.sub("<task_id>", sanitized)
    sanitized = _FORBIDDEN_DATE_LITERAL_RE.sub("<YYYYMMDD>", sanitized)
    sanitized = _FORBIDDEN_ENUM_CODE_LITERAL_RE.sub("<probed_code>", sanitized)
    sanitized, _ = repair_prompt_dialect_violations(sanitized, dialect=dialect)
    return sanitized.strip()


def build_dialect_constraint_block(dialect: str) -> str:
    """Return strict dialect syntax constraints for Reflector and SystemAwareMerger."""
    if dialect.lower() == "hive":
        return (
            "STRICT DIALECT CONSISTENCY (Target Dialect = HIVE):\n"
            "- NEVER recommend or exemplify ODPS-only or DuckDB/MySQL functions such as `MAX_PT`, `DATEADD`, `TO_CHAR`, "
            "3-argument `DATEDIFF(a, b, 'dd')`, `IFNULL`, `JSON_EXTRACT`, `STRFTIME`, or `UNNEST`.\n"
            "- In Hive SQL: `MAX_PT` DOES NOT EXIST! To query the latest partition of any partitioned table (`ds`/`dt`), "
            "`WarehouseLinker` MUST probe the table via `action='probe'` (`inspect_table_sample`) to inspect `partition_info.max_partition`, "
            "record that exact partition literal in `schema_notes`, and `DialectSQLGenerator` must filter with that explicit partition literal.\n"
            "- Standardize Hive functions on 2-argument `DATEDIFF(end_date, start_date)`, `DATE_ADD(start_date, days)`, "
            "`DATE_SUB(start_date, days)`, `DATE_FORMAT(expr, 'yyyy-MM-dd')` / `FROM_UNIXTIME(UNIX_TIMESTAMP(str, 'yyyyMMdd'), 'yyyy-MM-dd')`, "
            "`NVL`/`COALESCE`, `GET_JSON_OBJECT(col, '$.key')`, and `LATERAL VIEW [OUTER] explode(arr) t AS col`."
        )
    return (
        "STRICT DIALECT CONSISTENCY (Target Dialect = ODPS / MaxCompute):\n"
        "- NEVER recommend or exemplify unsupported Hive/MySQL/DuckDB functions such as `DATE_FORMAT`, `DATE_ADD`, "
        "`DATE_SUB`, `UNIX_TIMESTAMP`, `FROM_UNIXTIME`, `IFNULL`, `JSON_EXTRACT`, `STRFTIME`, `UNNEST`, or 2-argument `DATEDIFF`.\n"
        "- In ODPS SQL, standardize on `MAX_PT('<table_name>')` for latest partitions, `TO_CHAR(datetime, 'yyyymmdd')` / "
        "`TO_CHAR(TO_DATE(str, 'yyyymmdd'), 'yyyy-mm-dd')`, `TO_DATE(date_str, 'yyyymmdd')`, `DATEADD(datetime, delta, 'dd')`, "
        "3-argument `DATEDIFF(end_date, start_date, 'dd')`, `NVL`/`COALESCE`, `GET_JSON_OBJECT(col, '$.key')`, and "
        "`LATERAL VIEW [OUTER] explode(arr) t AS col`."
    )


class Reflector:
    """Offline reflective prompt optimizer powered by REFLECTOR_MODEL with 3-module fault attribution."""

    @staticmethod
    def build_system_prompt(dialect: str = "odps", preferred_module: ModuleName | None = None) -> str:
        focus_block = ""
        if preferred_module is not None:
            focus_block = (
                f"\nSPECIALIST EVOLUTION DIRECTIVE: For this generation, you MUST set "
                f"`target_module` to \"{preferred_module}\" and produce a comprehensive, high-leverage "
                f"upgraded System Prompt specifically for `{preferred_module}` addressing its responsibilities below.\n"
            )

        return (
            "You are an expert data warehouse SQL optimization architect acting as the offline Reflector "
            "for a 3-module Text-to-SQL agent (`linker`, `generator`, `refiner`).\n"
            "Your goal is to diagnose gateway interceptions, execution errors, and silent result-set mismatches "
            "on training episodes, attribute root causes across the 3 modules, distill generalizable enterprise "
            "warehouse rules, and emit an improved System Prompt for the target module.\n\n"
            "MULTI-MODULE ATTRIBUTION RESPONSIBILITIES:\n"
            "- `linker` (`WarehouseLinker`): Responsible for (a) distinguishing canonical business tables from legacy/deprecated/incomplete tables "
            "by comparing table/column comments, (b) proactively calling `action='probe'` (`inspect_table_sample`, up to 4 tables) WHENEVER "
            "candidate tables contain internal/opaque status, channel, or type code columns (`*_code`), unit-sensitive integer columns "
            "(cents/bps vs yuan), heterogeneous partition keys (`ds` vs `dt`), JSON/array columns, OR when Target Dialect is HIVE and the "
            "question asks for the latest partition (since Hive lacks `MAX_PT`), and (c) documenting discovered code mappings, unit conversion "
            "factors (`/ 100.0`, `/ 10000.0`), SCD2 effective-date columns, and probed `max_partition` values in `schema_notes`.\n"
            "- `generator` (`DialectSQLGenerator`): Responsible for dialect-compliant SQL synthesis, partition pruning, SCD Type-2 historical "
            "point-in-time interval slicing (`biz_date BETWEEN eff_start_dt AND eff_end_dt` with `YYYY-MM-DD` date format alignment — NEVER "
            "comparing `YYYYMMDD` strings directly against `YYYY-MM-DD` bounds and NEVER substituting `is_current=1` for historical event-date "
            "attribution), CTE pre-aggregation by join grain before joining multi-row fact tables to prevent Cartesian fanout, `LEFT JOIN` "
            "right-table filtering inside `ON` or pre-aggregated CTEs rather than `WHERE`, and explicit window frames (`ROWS BETWEEN ...`).\n"
            "- `refiner` (`GatewayRefiner`): Responsible for BOTH (a) repairing gateway AST/dialect/partition errors and empty/null outputs, AND "
            "(b) performing a systematic **Post-Execution Sanity & Caliber Audit** (`POST_EXECUTION_SANITY_AUDIT`) on queries that passed the "
            "gateway and returned non-empty rows! The evolved `refiner` prompt MUST explicitly instruct `GatewayRefiner` to audit non-empty SQL "
            "for silent caliber bugs: (1) `LEFT JOIN` right-table predicates placed in `WHERE` that silently drop unmatched entities, "
            "(2) `LAST_VALUE`/`FIRST_VALUE` or tied `ORDER BY` cumulative/rolling windows missing explicit `ROWS BETWEEN ...` frames, "
            "(3) unexcluded test/drill/abnormal/brush traffic or gross-vs-clean metric mixups, (4) SCD2 date-format mismatches (`YYYYMMDD` vs "
            "`YYYY-MM-DD`) or multi-fact join fanout, and (5) cent/bps vs yuan scaling or unapplied probed code values (returning the exact same "
            "SQL unchanged if the audit finds zero defects).\n"
            f"{focus_block}\n"
            "CRITICAL ANTI-CHEAT & CONTRACT CONSTRAINTS:\n"
            "1. NEVER hardcode or mention specific benchmark table names, task IDs, specific dates, or specific warehouse enum code literals "
            "(describe patterns generically, e.g., 'probed status/channel codes from inspect_table_sample', 'SCD2 eff_start_dt/eff_end_dt').\n"
            "2. ANTI-OVER-FILTERING GUARD: Distinguish test/abnormal exclusions from lifecycle-status filters: map explicit "
            "external/formal/non-test wording to the probed test-account exclusion flag (= 0), and map risk-normal wording to the "
            "probed abnormal flag, ONLY emit lifecycle status/channel/operating-status predicates when the question explicitly asks "
            "for that status/channel/operating state (e.g., never add a settled-status filter unless asked for settled, never add "
            "an operating-status filter unless asked for open/operating), and never add `col IS NOT NULL` or `col <> ''` on "
            "grouping dimensions unless requested.\n"
            "3. Preserve the exact JSON output schema keys required by the target module (`selected_tables` & `action` for linker; "
            "`thought` & `sql` for generator; `thought`, `diagnosis`, & `sql` for refiner).\n"
            f"4. {build_dialect_constraint_block(dialect)}\n\n"
            "Output strictly valid JSON:\n"
            "{\n"
            '  "target_module": "linker" | "generator" | "refiner",\n'
            '  "fault_diagnosis": "concise root-cause diagnosis",\n'
            '  "distilled_rules": ["rule 1", "rule 2"],\n'
            '  "updated_system_prompt": "complete updated system prompt for target_module"\n'
            "}"
        )

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    async def reflect_and_mutate(
        self,
        parent_policy: Policy,
        train_trajectories: list[AgentTrajectory],
        *,
        generation: int,
        new_policy_id: str,
        dialect: str = "odps",
        preferred_module: ModuleName | None = None,
    ) -> tuple[Policy, dict[str, Any]]:
        """Analyze suboptimal/failed train traces and produce one mutated child Policy."""
        suboptimal = [
            t
            for t in train_trajectories
            if not t.ex_match or not t.first_pass_gateway
        ]
        all_passed = not suboptimal
        if not suboptimal:
            suboptimal = list(train_trajectories)

        traces = [t.compact_summary_for_reflector() for t in suboptimal[:6]]
        status_note = (
            "NOTE: All sampled trajectories below already passed EX and first-pass gateway. "
            "Strengthen generalizable warehouse guards conservatively without restricting valid behavior.\n\n"
            if all_passed
            else ""
        )
        pref_note = (
            f"IMPORTANT: Focus this mutation strictly on `target_module = \"{preferred_module}\"` "
            f"and rewrite `[{preferred_module}]`'s system prompt to eliminate its class of failures.\n\n"
            if preferred_module is not None
            else ""
        )
        user_msg = (
            f"Target Dialect: {dialect.upper()}\n"
            f"Parent Policy ID: {parent_policy.policy_id}\n\n"
            "Current System Prompts:\n"
            f"[linker]\n{parent_policy.linker_prompt}\n[/linker]\n\n"
            f"[generator]\n{parent_policy.generator_prompt}\n[/generator]\n\n"
            f"[refiner]\n{parent_policy.refiner_prompt}\n[/refiner]\n\n"
            f"{status_note}{pref_note}"
            "Suboptimal / Failed Training Trajectories (with gateway errors, probe usage, and gold row samples):\n"
            f"{json.dumps(traces, ensure_ascii=False, indent=2)}\n\n"
            "Diagnose the root causes and output strict JSON rewriting the target module's system prompt "
            f"with generalizable {dialect.upper()}-compliant enterprise data warehouse rules."
        )

        sys_prompt = self.build_system_prompt(dialect=dialect, preferred_module=preferred_module)
        resp = await self.llm.chat_json(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            role_type="reflector",
            temperature=0.2,
            max_tokens=8192,
        )
        parsed = resp.parsed_json or {}
        raw_target = str(parsed.get("target_module", preferred_module or "generator")).strip().lower()
        if preferred_module is not None:
            target_module: ModuleName = preferred_module
        else:
            target_module = (
                raw_target if raw_target in ("linker", "generator", "refiner") else "generator"
            )  # type: ignore[assignment]

        diagnosis = str(parsed.get("fault_diagnosis", "")).strip()
        rules = [
            sanitize_rule_text(str(r), dialect=dialect)
            for r in (parsed.get("distilled_rules") or [])
            if str(r).strip()
        ]
        candidate_prompt = str(parsed.get("updated_system_prompt", "")).strip()
        # Sanitize accidental table/date/enum leaks first so rich LLM prompt rewrites aren't unnecessarily discarded!
        candidate_prompt = sanitize_rule_text(candidate_prompt, dialect=dialect)

        valid, validated_prompt, guard_msg = validate_and_sanitize_prompt(
            candidate_prompt,
            target_module,
            dialect=dialect,
            auto_repair_dialect=True,
        )
        dialect_repaired = bool(valid and guard_msg)
        if dialect_repaired:
            diagnosis = f"{diagnosis} [{guard_msg}]".strip()
        elif not valid:
            base_prompt = parent_policy.get_prompt(target_module)
            if rules:
                addition = "\n\nLearned Warehouse Guidelines:\n" + "\n".join(f"- {r}" for r in rules)
                _, validated_prompt, _ = validate_and_sanitize_prompt(
                    base_prompt + addition,
                    target_module,
                    dialect=dialect,
                    auto_repair_dialect=True,
                )
            else:
                validated_prompt = base_prompt
            diagnosis = f"{diagnosis} [Guard fallback applied: {guard_msg}]".strip()

        child = parent_policy.clone_with_mutation(
            new_policy_id=new_policy_id,
            target_module=target_module,
            new_prompt=validated_prompt,
            generation=generation,
            rationale=diagnosis,
        )
        meta = {
            "target_module": target_module,
            "fault_diagnosis": diagnosis,
            "distilled_rules": rules,
            "guard_triggered": (not valid) or dialect_repaired,
            "dialect_repaired": dialect_repaired,
            "guard_reason": guard_msg,
        }
        return child, meta
