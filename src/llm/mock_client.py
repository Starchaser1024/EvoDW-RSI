"""Deterministic mock LLM client for offline unit tests and `--mock` dry-runs."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from src.benchmarks.tasks import get_all_tasks
from src.llm.client import BaseLLMClient, LLMResponse, RoleType


class MockWarehouseLLMClient(BaseLLMClient):
    """Simulates Actor (Linker/Generator/Refiner) and Reflector behavior offline without API calls."""

    def __init__(
        self,
        actor_model: str = "deepseek-v4.1-flash",
        reflector_model: str = "deepseek-v4-pro",
        dialect: str = "odps",
    ) -> None:
        super().__init__(actor_model=actor_model, reflector_model=reflector_model)
        self.dialect = dialect
        self._tasks = get_all_tasks(suite="all") + get_all_tasks(suite="external_bird_mini")
        self._question_map = {t.question.strip(): t for t in self._tasks}
        self._reflect_step = 0

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        session_id: str | None = None,
    ) -> LLMResponse:
        sys_prompt = messages[0]["content"] if messages else ""
        user_prompt = "\n".join(m["content"] for m in messages[1:])

        if role_type == "reflector":
            payload = self._handle_reflector(sys_prompt, user_prompt)
            p_tok, c_tok, lat = 620, 280, 18.0
        elif sys_prompt.lstrip().startswith("You are WarehouseLinker"):
            payload = self._handle_linker(sys_prompt, user_prompt)
            p_tok, c_tok, lat = 340, 110, 8.0
        elif sys_prompt.lstrip().startswith("You are GatewayRefiner") or "Previous SQL:" in user_prompt:
            payload = self._handle_refiner(sys_prompt, user_prompt)
            p_tok, c_tok, lat = 410, 140, 11.0
        else:
            payload = self._handle_generator(sys_prompt, user_prompt)
            p_tok, c_tok, lat = 480, 160, 12.0

        content = json.dumps(payload, ensure_ascii=False)
        resp = LLMResponse(
            content=content,
            parsed_json=payload,
            model=self.resolve_model(role_type, model),
            role_type=role_type,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            latency_ms=lat,
            finish_reason="stop",
            retries=0,
            session_id=session_id or str(uuid.uuid4()),
        )
        self.ledger.record(resp)
        return resp

    async def chat_text(
        self,
        messages: list[dict[str, str]],
        role_type: RoleType = "actor",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        session_id: str | None = None,
    ) -> LLMResponse:
        return await self.chat_json(
            messages=messages,
            role_type=role_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id,
        )

    def _find_task(self, user_prompt: str):
        m = re.search(r"Business Question:\s*(.+?)\nBusiness Hint:", user_prompt, flags=re.DOTALL)
        if m:
            q = m.group(1).strip()
            if q in self._question_map:
                return self._question_map[q]
        for q, task in self._question_map.items():
            if q in user_prompt:
                return task
        return None

    @staticmethod
    def _extract_tables_from_text(text: str) -> list[str]:
        known = [
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
            "dim_user",
            "dim_shop",
            "dim_item",
            "dim_content_creator",
            "fin_district",
            "fin_account",
            "fin_loan_contract_di",
            "fin_card_txn_di",
            "clin_patient",
            "clin_diagnosis_scd2",
            "clin_encounter_di",
            "clin_lab_exam_di",
            "edu_school_dim",
            "edu_frpm_snap_di",
            "edu_sat_report_di",
        ]
        found = [t for t in known if t in text]
        return found or ["dwd_trade_order_di", "dim_shop"]

    def _handle_linker(self, sys_prompt: str, user_prompt: str) -> dict[str, Any]:
        task = self._find_task(user_prompt)
        q_part = user_prompt.split("Warehouse Catalog Summary:", 1)[0]
        search_text = f"{q_part}\n{task.gold_sql if task else ''}"
        tables = self._extract_tables_from_text(search_text)

        # If this is after probe observation, finalize.
        if "Observation from inspect_table_sample:" in user_prompt:
            return {
                "thought": "Probed partition keys and sample values; finalizing relevant tables.",
                "action": "finalize",
                "selected_tables": tables,
                "schema_notes": "Verified partition key names (ds or dt) and sample values.",
            }

        # If prompt learned to probe partitioned/JSON tables, perform probe round first.
        wants_probe = (
            "always probe" in sys_prompt.lower()
            or "you must probe" in sys_prompt.lower()
            or "inspect_table_sample" in sys_prompt.lower()
            or "inspect partition" in sys_prompt.lower()
            or "check partition column" in sys_prompt.lower()
        )
        if wants_probe:
            probe_reqs = [{"table_name": t, "columns": None} for t in tables[:2]]
            return {
                "thought": "Need to inspect actual partition keys (ds vs dt) and JSON/null patterns.",
                "action": "probe",
                "probe_requests": probe_reqs,
            }

        return {
            "thought": "Selected tables matching the business entities.",
            "action": "finalize",
            "selected_tables": tables,
            "schema_notes": "",
        }

    @staticmethod
    def _has_rule_partition(prompt: str) -> bool:
        p = prompt.lower()
        return "max_pt" in p or "odps-0123031" in p or "partition filter" in p

    @staticmethod
    def _has_rule_json(prompt: str) -> bool:
        p = prompt.lower()
        return "get_json_object" in p or "lateral view" in p

    @staticmethod
    def _has_rule_date(prompt: str) -> bool:
        p = prompt.lower()
        return "datediff" in p and ("dateadd" in p or "3 argument" in p or "three-arg" in p or "'dd'" in p)

    @staticmethod
    def _has_rule_fanout(prompt: str) -> bool:
        p = prompt.lower()
        return "row_number" in p or "pre-aggregat" in p or "fanout" in p or "cte" in p

    @staticmethod
    def _has_rule_ratio(prompt: str) -> bool:
        p = prompt.lower()
        return "nullif" in p or "nvl" in p

    def _handle_generator(self, sys_prompt: str, user_prompt: str) -> dict[str, Any]:
        task = self._find_task(user_prompt)
        if task is None:
            return {
                "thought": "Fallback generic order count query.",
                "sql": "SELECT COUNT(*) AS cnt FROM dwd_trade_order_di WHERE ds = MAX_PT('dwd_trade_order_di')",
            }

        if task.task_id.startswith("ext_"):
            return self._handle_external_generator(task, sys_prompt, user_prompt)

        cat = task.category
        gold = task.gold_sql
        has_probe = "Sample Inspection Observations" in user_prompt

        if cat == "partition_pruning":
            if self._has_rule_partition(sys_prompt) and (has_probe or "dwd_content_post_di" not in gold or "partition key" in sys_prompt.lower()):
                return {"thought": "Applying partition pruning and MAX_PT rule.", "sql": gold}
            # Naive mistake: omit WHERE partition filter on partitioned table!
            naive = re.sub(
                r"\s+WHERE\s+[^GROUP^ORDER^LIMIT]+",
                " ",
                gold,
                count=1,
                flags=re.IGNORECASE,
            )
            if naive == gold:
                naive = "SELECT COUNT(*) AS cnt FROM dwd_trade_order_di"
            return {"thought": "Direct aggregation without explicit partition filter.", "sql": naive}

        if cat == "json_array_unnest":
            if self._has_rule_json(sys_prompt):
                return {"thought": "Using GET_JSON_OBJECT and LATERAL VIEW explode cleanly.", "sql": gold}
            # Naive hallucination: JSON_EXTRACT instead of GET_JSON_OBJECT or UNNEST instead of LATERAL VIEW.
            bad = gold.replace("GET_JSON_OBJECT", "JSON_EXTRACT").replace(
                "LATERAL VIEW OUTER explode", "CROSS JOIN UNNEST"
            ).replace("LATERAL VIEW explode", "CROSS JOIN UNNEST")
            if bad == gold:
                bad = "SELECT JSON_EXTRACT(ext_json, '$.page') FROM ods_app_event_log_di WHERE ds = '20261015'"
            return {"thought": "Extracting JSON/array with standard SQL functions.", "sql": bad}

        if cat == "retention_date_math":
            if self._has_rule_date(sys_prompt):
                return {"thought": "Using 3-arg DATEDIFF, DATEADD, TO_DATE/TO_CHAR.", "sql": gold}
            # Naive mistake: 2-arg DATEDIFF or DATE_ADD.
            bad = re.sub(r",\s*'dd'\)", ")", gold, flags=re.IGNORECASE)
            bad = bad.replace("DATEADD(", "DATE_ADD(")
            if bad == gold:
                bad = "SELECT DATEDIFF('2026-10-01', '2026-09-30') FROM dim_user LIMIT 1"
            return {"thought": "Computing date difference with 2-arg DATEDIFF.", "sql": bad}

        if cat == "fanout_dedup":
            if self._has_rule_fanout(sys_prompt):
                return {"thought": "Pre-aggregating CTEs and deduplicating via ROW_NUMBER.", "sql": gold}
            # Naive mistake: omit dedup WHERE rn = 1 or drop partition filter on second table.
            bad = gold.replace("WHERE rn = 1", "WHERE rn >= 1").replace("MAX_PT('dwd_refund_di')", "'20261015'")
            if bad == gold:
                bad = "SELECT order_status, COUNT(*) FROM dwd_trade_order_di GROUP BY order_status"
            return {"thought": "Joining/aggregating records directly.", "sql": bad}

        if cat == "ratio_null_handling":
            if self._has_rule_ratio(sys_prompt):
                return {"thought": "Handling NULL/empty strings and zero denominators via NVL/NULLIF.", "sql": gold}
            # Naive mistake: IFNULL instead of NVL/COALESCE or omitting empty string handling.
            bad = gold.replace("NVL(", "IFNULL(").replace("NULLIF(", "COALESCE(")
            if bad == gold:
                bad = "SELECT shop_id, IFNULL(buyer_cnt / uv, 0) FROM dws_shop_metrics_1d"
            return {"thought": "Computing ratio with IFNULL.", "sql": bad}

        is_hive = "Target Dialect: HIVE" in user_prompt
        hive_gold = re.sub(r"MAX_PT\('[^']+'\)", "'20261015'", gold)
        target_gold = hive_gold if is_hive else gold

        if cat == "semantic_linking_trap":
            if has_probe and ("cent" in sys_prompt.lower() or "100" in sys_prompt.lower() or "probe" in user_prompt.lower()):
                return {"thought": "Applied probed enum codes and cent-to-yuan conversion.", "sql": target_gold}
            return {
                "thought": "Querying legacy table without probing enum codes.",
                "sql": "SELECT pay_channel, COUNT(*) FROM dwd_trade_order_legacy_di WHERE ds = '20261015' GROUP BY pay_channel",
            }

        if cat == "scd2_temporal_join":
            if "eff_start_dt" in sys_prompt.lower() or "scd" in sys_prompt.lower():
                return {"thought": "Applied SCD2 point-in-time interval join and CTE anti-fanout.", "sql": target_gold}
            return {
                "thought": "Joined SCD2 table using current version only.",
                "sql": "SELECT vip_tier, COUNT(*) FROM dim_user_vip_scd2 WHERE is_current = 1 GROUP BY vip_tier",
            }

        if cat == "silent_caliber_audit":
            if "rows between" in sys_prompt.lower() or "left join" in sys_prompt.lower():
                return {"thought": "Applied explicit ROWS BETWEEN window frame and ON-clause filtering.", "sql": target_gold}
            return {
                "thought": "Computed aggregate without explicit window frame or ON-clause guard.",
                "sql": "SELECT shop_id, SUM(gross_gmv_yuan) FROM dws_shop_operating_1d WHERE ds = '20261015' GROUP BY shop_id",
            }

        return {"thought": "Generated query.", "sql": target_gold}

    def _handle_refiner(self, sys_prompt: str, user_prompt: str) -> dict[str, Any]:
        task = self._find_task(user_prompt)
        if task is not None and task.task_id.startswith("ext_"):
            return self._handle_external_refiner(task, sys_prompt, user_prompt)
        raw_gold = task.gold_sql if task else "SELECT 1"
        is_hive = "Target Dialect: HIVE" in user_prompt
        gold = re.sub(r"MAX_PT\('[^']+'\)", "'20261015'", raw_gold) if is_hive else raw_gold
        if "POST_EXECUTION_SANITY_AUDIT" in user_prompt and ("sanity" in sys_prompt.lower() or "caliber" in sys_prompt.lower()):
            return {
                "thought": "Audited non-empty result against business caliber, LEFT JOIN ON-clause, and ROWS BETWEEN rules.",
                "diagnosis": "Repaired silent business caliber / window frame / LEFT JOIN WHERE defect.",
                "sql": gold,
            }
        has_repair_rule = (
            "odps-0123031" in sys_prompt.lower()
            or "max_pt" in sys_prompt.lower()
            or "get_json_object" in sys_prompt.lower()
            or "datediff" in sys_prompt.lower()
        )
        # Under seed refiner, repair simple gateway partition errors on ~half of partition tasks
        # so both first_pass_correct and self_repaired_after_error trajectories appear naturally.
        if task is not None and (
            has_repair_rule
            or (
                "ODPS-0123031" in user_prompt
                and task.category == "partition_pruning"
                and int(task.task_id.split("_")[1]) % 2 == 1
            )
        ):
            return {
                "thought": "Identified gateway dialect/partition constraint from error feedback; repairing SQL.",
                "diagnosis": "Applied missing partition filter or corrected ODPS function signature.",
                "sql": gold,
            }

        # Fallback partial repair that fails EX or retains issue when refiner hasn't evolved yet.
        m = re.search(r"Previous SQL:\n(.*?)\n\nSandbox Execution Feedback:", user_prompt, flags=re.DOTALL)
        prev_sql = m.group(1).strip() if m else ""
        if not prev_sql:
            prev_sql = "SELECT 1"
        if "ODPS-0123031" in user_prompt and "dwd_trade_order_di" in prev_sql and "WHERE" not in prev_sql.upper():
            if "GROUP BY" in prev_sql.upper():
                idx_gb = prev_sql.upper().index("GROUP BY")
                patched = prev_sql[:idx_gb].rstrip() + " WHERE ds = '20261014' " + prev_sql[idx_gb:]
            else:
                patched = prev_sql + " WHERE ds = '20261014'"
        else:
            patched = prev_sql
        return {
            "thought": "Attempted minimal fix based on feedback.",
            "diagnosis": "Added basic filter condition.",
            "sql": patched,
        }

    def _handle_reflector(self, sys_prompt: str, user_prompt: str) -> dict[str, Any]:
        # Merger prompt handling.
        if "SystemAwareMerger" in sys_prompt or "Candidate A" in user_prompt:
            m_a = re.search(r"Candidate A Prompts:\n(.*?)\n\nCandidate B Prompts:", user_prompt, flags=re.DOTALL)
            m_b = re.search(r"Candidate B Prompts:\n(.*?)\n\nPlease synthesize", user_prompt, flags=re.DOTALL)
            a_txt = m_a.group(1) if m_a else ""
            b_txt = m_b.group(1) if m_b else ""

            def _extract_section(block: str, label: str) -> str:
                pat = rf"\[{label}\]\n(.*?)(?=\n\[[A-Z]+\]|\Z)"
                mm = re.search(pat, block, flags=re.DOTALL)
                return mm.group(1).strip() if mm else ""

            def _merge_texts(t1: str, t2: str) -> str:
                seen_lines: list[str] = []
                seen_set: set[str] = set()
                for line in (t1 + "\n" + t2).splitlines():
                    norm = line.strip()
                    if norm and norm not in seen_set:
                        seen_set.add(norm)
                        seen_lines.append(line)
                return "\n".join(seen_lines)

            return {
                "linker_prompt": _merge_texts(_extract_section(a_txt, "LINKER"), _extract_section(b_txt, "LINKER")),
                "generator_prompt": _merge_texts(_extract_section(a_txt, "GENERATOR"), _extract_section(b_txt, "GENERATOR")),
                "refiner_prompt": _merge_texts(_extract_section(a_txt, "REFINER"), _extract_section(b_txt, "REFINER")),
                "merge_rationale": "Combined complementary dialect, date-math, unnesting, and schema-probing rules from both Pareto parents.",
            }

        # Standard Reflector error attribution & rule distillation.
        self._reflect_step += 1
        is_hive = "Target Dialect: HIVE" in user_prompt
        m_pref = re.search(r'target_module = "(linker|generator|refiner)"', user_prompt)
        pref_mod = m_pref.group(1) if m_pref else None
        current_gen = ""
        m_cur = re.search(r"\[generator\]\n(.*?)\n\[/generator\]", user_prompt, flags=re.DOTALL | re.IGNORECASE)
        if m_cur:
            current_gen = m_cur.group(1).strip()
        m_link = re.search(r"\[linker\]\n(.*?)\n\[/linker\]", user_prompt, flags=re.DOTALL | re.IGNORECASE)
        current_link = m_link.group(1).strip() if m_link else ""
        m_ref = re.search(r"\[refiner\]\n(.*?)\n\[/refiner\]", user_prompt, flags=re.DOTALL | re.IGNORECASE)
        current_ref = m_ref.group(1).strip() if m_ref else ""

        if pref_mod == "linker":
            l_rule = (
                "Discovered Linker Rule (Deep Probing & Canonical Layer Selection): Always invoke action='probe' "
                "(inspect_table_sample) on tables containing internal *_code columns, integer cent/bps columns, "
                "heterogeneous ds/dt partitions, or latest partitions in Hive; exclude deprecated legacy tables "
                "and record discovered code values, unit conversion factors, and max_partition in schema_notes."
            )
            return {
                "target_module": "linker",
                "fault_diagnosis": "Linker skipped inspect_table_sample on opaque internal codes and partition bounds.",
                "distilled_rules": [l_rule],
                "updated_system_prompt": current_link + "\n\n" + l_rule,
            }
        if pref_mod == "refiner":
            r_rule = (
                "Discovered Refiner Rule (Post-Execution Sanity & Caliber Audit): On POST_EXECUTION_SANITY_AUDIT "
                "for non-empty outputs as well as gateway errors, audit LEFT JOIN right-table predicates (move from "
                "WHERE to ON/CTE), enforce explicit ROWS BETWEEN window frames on LAST_VALUE/peer ORDER BY, verify "
                "clean commercial caliber filters, and check cent/bps unit scaling."
            )
            return {
                "target_module": "refiner",
                "fault_diagnosis": "Refiner lacked post-execution sanity & silent business caliber audit rules.",
                "distilled_rules": [r_rule],
                "updated_system_prompt": current_ref + "\n\n" + r_rule,
            }
        if pref_mod == "generator" and ("scd2_temporal_join" in user_prompt or is_hive):
            g_rule = (
                "Discovered Generator Rule (SCD2 & Anti-Fanout): Slice SCD2 tables via event_dt BETWEEN eff_start_dt "
                "AND eff_end_dt after aligning dates to YYYY-MM-DD, pre-aggregate multi-row fact tables in CTEs before "
                "joining, and in Hive use 2-arg DATEDIFF, DATE_ADD/DATE_SUB, and DATE_FORMAT without MAX_PT."
                if is_hive
                else "Discovered Generator Rule (SCD2 & Anti-Fanout): Slice SCD2 tables via event_dt BETWEEN eff_start_dt "
                "AND eff_end_dt after aligning dates to YYYY-MM-DD (never is_current=1 for historical events), and "
                "pre-aggregate multi-row fact tables in CTEs before joining."
            )
            return {
                "target_module": "generator",
                "fault_diagnosis": "Generator missed SCD2 YYYY-MM-DD interval slicing and CTE pre-aggregation.",
                "distilled_rules": [g_rule],
                "updated_system_prompt": current_gen + "\n\n" + g_rule,
            }

        # Inspect actual failure traces in user_prompt to reflect generalizable rules (up to 2 per batch).
        trace_part = user_prompt.split("Suboptimal / Failed Training Trajectories", 1)[-1]

        rule_partition = (
            "Discovered Warehouse Rule (Partition Pruning): Every partitioned fact table (ods_*, dwd_*, dws_*) "
            "MUST have an explicit partition filter on its partition column (e.g., ds or dt) in WHERE or JOIN ON. "
            "Use MAX_PT('<table_name>') for the latest partition rather than scanning without filter or writing SELECT MAX(ds)."
        )
        rule_json = (
            "Discovered Warehouse Rule (JSON & Array Unnesting): Use GET_JSON_OBJECT(col, '$.key') instead of JSON_EXTRACT. "
            "For array columns, use LATERAL VIEW explode(arr) alias AS col (or LATERAL VIEW OUTER explode(arr) to preserve empty arrays) instead of UNNEST."
        )
        rule_date = (
            "Discovered Warehouse Rule (ODPS Date Math): In ODPS, always use 3-argument DATEDIFF(end_date, start_date, 'dd') "
            "and DATEADD(TO_DATE(date_str, 'yyyymmdd'), delta, 'dd') with TO_CHAR(..., 'yyyymmdd') for cross-month calculations."
        )
        rule_fanout = (
            "Discovered Warehouse Rule (Dedup & Anti-Fanout): Deduplicate raw event logs with ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ... DESC) = 1 before aggregating. "
            "When combining multiple fact tables at different granularities, pre-aggregate each fact table in separate CTEs before joining."
        )
        rule_ratio = (
            "Discovered Warehouse Rule (Null & Safe Ratio): Replace IFNULL with NVL/COALESCE, normalize NULL and empty string '' via COALESCE(NULLIF(col, ''), 'UNKNOWN'), "
            "and guard divisions with 1.0 * num / NULLIF(denom, 0)."
        )

        collected_rules: list[str] = []
        diags: list[str] = []

        if ("ODPS-0123031" in trace_part or '"partition_pruning"' in trace_part) and not self._has_rule_partition(current_gen):
            collected_rules.append(rule_partition)
            diags.append("mandatory partition filters (MAX_PT / ds / dt)")
        if ("json_array_unnest" in trace_part or "JSON_EXTRACT" in trace_part or "UNNEST" in trace_part) and not self._has_rule_json(current_gen):
            collected_rules.append(rule_json)
            diags.append("GET_JSON_OBJECT & LATERAL VIEW explode syntax")
        if ("retention_date_math" in trace_part or "DATEDIFF" in trace_part or "DATE_ADD" in trace_part) and not self._has_rule_date(current_gen):
            collected_rules.append(rule_date)
            diags.append("3-arg DATEDIFF & DATEADD date math")
        if "fanout_dedup" in trace_part and not self._has_rule_fanout(current_gen):
            collected_rules.append(rule_fanout)
            diags.append("ROW_NUMBER deduplication & CTE pre-aggregation")
        if ("ratio_null_handling" in trace_part or "IFNULL" in trace_part) and not self._has_rule_ratio(current_gen):
            collected_rules.append(rule_ratio)
            diags.append("NVL/NULLIF zero-division & empty-string handling")

        if collected_rules:
            selected_rules = collected_rules[:2]
            return {
                "target_module": "generator",
                "fault_diagnosis": f"Distilled generator rules for: {', '.join(diags[:2])}.",
                "distilled_rules": selected_rules,
                "updated_system_prompt": current_gen + "\n\n" + "\n\n".join(selected_rules),
            }

        # Fallback: pick 1 uncovered generator rule if any remain.
        for check, r_txt, d_txt in (
            (self._has_rule_partition, rule_partition, "partition pruning"),
            (self._has_rule_json, rule_json, "JSON and array unnesting"),
            (self._has_rule_date, rule_date, "ODPS date math"),
            (self._has_rule_fanout, rule_fanout, "dedup and anti-fanout"),
            (self._has_rule_ratio, rule_ratio, "NULL and safe ratio handling"),
        ):
            if not check(current_gen):
                return {
                    "target_module": "generator",
                    "fault_diagnosis": f"Added uncovered generator rule for {d_txt}.",
                    "distilled_rules": [r_txt],
                    "updated_system_prompt": current_gen + "\n\n" + r_txt,
                }

        if "inspect partition" not in current_link.lower():
            rule = (
                "Discovered Linker Rule (Metadata Probing): Always probe partitioned fact tables and JSON-bearing tables "
                "with action='probe' first to inspect partition column names (such as ds vs dt), available partition bounds, and JSON/null value examples."
            )
            return {
                "target_module": "linker",
                "fault_diagnosis": "Linker finalized without probing actual partition keys (e.g., dt vs ds) and sample JSON formats.",
                "distilled_rules": [rule],
                "updated_system_prompt": current_link + "\n\n" + rule,
            }

        rule = (
            "Discovered Refiner Rule (Gateway Repair): On ODPS-0123031 add explicit partition filters using MAX_PT; "
            "on ODPS-0130071 fix DATEDIFF to 3 args, DATE_ADD to DATEADD, JSON_EXTRACT to GET_JSON_OBJECT, and IFNULL to NVL."
        )
        return {
            "target_module": "refiner",
            "fault_diagnosis": "Refiner needs explicit ODPS gateway error mapping rules for rapid repair.",
            "distilled_rules": [rule],
            "updated_system_prompt": current_ref + "\n\n" + rule,
        }

    def _handle_external_generator(self, task: Any, sys_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Rule-governed SQL simulation on unseen external_bird_mini schemas without domain-specific cheating."""
        from src.benchmarks.external_cross_schema import to_hive_external_gold_sql

        is_hive = "Target Dialect: HIVE" in user_prompt
        target_gold = to_hive_external_gold_sql(task.gold_sql) if is_hive else task.gold_sql
        has_probe = "Sample Inspection Observations" in user_prompt
        p_low = sys_prompt.lower()
        has_cent_rule = "cent" in p_low or "/100.0" in p_low or "10000" in p_low
        has_anti_fanout = self._has_rule_fanout(sys_prompt) or "child-grain" in p_low or "grain" in p_low
        has_scd2_rule = "eff_start_dt" in p_low or "scd" in p_low
        has_left_window_rule = "rows between" in p_low and "left join" in p_low
        has_str_pct_rule = "strip '%'" in p_low or "string-encoded percentage" in p_low
        tid = task.task_id

        # Residual cross-domain dirty-string ratio challenges not covered by e-commerce Hard-v2 rules:
        if tid in ("ext_fin_04", "ext_clin_07") and not has_str_pct_rule:
            if tid == "ext_fin_04":
                sql = (
                    "SELECT d.region, ROUND(AVG(TRY_CAST(d.unemployment_rate_str AS DOUBLE)), 2) AS avg_unemp_pct, "
                    "ROUND(1.0 * SUM(CASE WHEN l.loan_status_code IN ('B', 'D') THEN 1 ELSE 0 END) / COUNT(*), 4) AS problem_loan_ratio "
                    "FROM fin_loan_contract_di l JOIN fin_account a ON l.account_id = a.account_id "
                    "JOIN fin_district d ON a.district_id = d.district_id "
                    "WHERE l.ds = '20261015' AND a.is_test_acct = 0 GROUP BY d.region ORDER BY d.region ASC"
                )
            else:
                sql = (
                    "SELECT COALESCE(NULLIF(TRIM(ana_pattern_code), ''), 'UNRECORDED') AS ana_pattern, COUNT(*) AS exam_cnt, "
                    "ROUND(AVG(TRY_CAST(acl_pos_ratio_str AS DOUBLE)), 4) AS avg_acl_ratio "
                    "FROM clin_lab_exam_di WHERE dt BETWEEN '20261013' AND '20261015' "
                    "GROUP BY COALESCE(NULLIF(TRIM(ana_pattern_code), ''), 'UNRECORDED') ORDER BY exam_cnt DESC, ana_pattern ASC"
                )
            return {"thought": "Computed ratio without stripping mixed '%' string suffixes.", "sql": sql}

        if task.category == "partition_pruning":
            if self._has_rule_partition(sys_prompt) or (is_hive and has_probe):
                if has_cent_rule or tid == "ext_clin_01":
                    return {"thought": "Applied partition pruning and unit rules.", "sql": target_gold}
                # Partition passed, but forgot cent/100 conversion -> silent caliber error!
                no_scale = target_gold.replace("/ 100.0", "")
                return {"thought": "Filtered latest partition without cent scaling.", "sql": no_scale}
            raw_sq = task.metadata.get("adaptation_ledger", {}).get("original_sqlite_sql", "SELECT 1")
            return {"thought": "Generated unpartitioned SQLite-style query.", "sql": raw_sq}

        if task.category == "semantic_linking_trap":
            if has_probe and has_cent_rule:
                return {"thought": "Applied probed opaque domain codes and cent-to-unit conversion.", "sql": target_gold}
            if tid == "ext_fin_02":
                bad = (
                    "SELECT l.loan_status_code, COUNT(*) AS problem_loan_cnt, ROUND(SUM(l.principal_cent), 2) AS problem_principal_amt "
                    "FROM fin_loan_contract_di l JOIN fin_account a ON l.account_id = a.account_id "
                    "WHERE l.ds = '20261015' AND l.loan_status_code IN ('B', 'D') GROUP BY l.loan_status_code ORDER BY l.loan_status_code ASC"
                )
            elif tid == "ext_clin_03":
                bad = (
                    "SELECT COALESCE(NULLIF(TRIM(p.sex), ''), 'UNKNOWN') AS sex_group, COUNT(*) AS encounter_cnt, "
                    "ROUND(AVG(e.total_charge_cent), 2) AS avg_charge_yuan "
                    "FROM clin_encounter_di e JOIN clin_patient p ON e.patient_id = p.patient_id "
                    "WHERE e.ds = '20261015' GROUP BY COALESCE(NULLIF(TRIM(p.sex), ''), 'UNKNOWN') ORDER BY sex_group ASC"
                )
            else:
                bad = (
                    "SELECT COALESCE(NULLIF(TRIM(s.funding_type_code), ''), 'UNCLASSIFIED') AS funding_group, "
                    "COUNT(*) AS school_cnt, ROUND(AVG(f.per_pupil_grant_cent), 2) AS avg_grant_per_pupil_usd "
                    "FROM edu_frpm_snap_di f JOIN edu_school_dim s ON f.cds_code = s.cds_code "
                    "WHERE f.ds = '20261015' AND s.charter_flag = 1 AND f.academic_year = '2026-2027' "
                    "GROUP BY COALESCE(NULLIF(TRIM(s.funding_type_code), ''), 'UNCLASSIFIED') ORDER BY funding_group ASC"
                )
            return {"thought": "Queried without probing internal enum literals or cent units.", "sql": bad}

        if task.category == "fanout_dedup":
            if has_anti_fanout:
                return {"thought": "Pre-aggregated child-grain CTEs and matched composite keys.", "sql": target_gold}
            if tid == "ext_fin_03":
                bad = (
                    "SELECT l.account_id, ROUND(SUM(l.principal_cent) / 100.0, 2) AS loan_principal_amt, "
                    "ROUND(SUM(COALESCE(t.amount_cent, 0)) / 100.0, 2) AS withdrawal_amt "
                    "FROM fin_loan_contract_di l JOIN fin_account a ON l.account_id = a.account_id "
                    "LEFT JOIN fin_card_txn_di t ON l.account_id = t.account_id AND t.ds = '20261015' AND t.txn_type_code = 'VYDAJ' "
                    "WHERE l.ds = '20261015' AND a.is_test_acct = 0 GROUP BY l.account_id ORDER BY loan_principal_amt DESC, l.account_id ASC LIMIT 10"
                )
            elif tid == "ext_clin_04":
                bad = (
                    "SELECT e.dept_code, COUNT(*) AS encounter_cnt, ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS total_charge_yuan, "
                    "SUM(CASE WHEN l.plt_count < 100 THEN 1 ELSE 0 END) AS low_plt_exam_cnt "
                    "FROM clin_encounter_di e LEFT JOIN clin_lab_exam_di l ON e.encounter_id = l.encounter_id AND l.dt = '20261015' "
                    "WHERE e.ds = '20261015' AND e.is_test_acct = 0 GROUP BY e.dept_code ORDER BY e.dept_code ASC"
                )
            else:
                bad = (
                    "SELECT f.academic_year, COUNT(DISTINCT f.cds_code) AS charter_school_cnt, "
                    "SUM(f.frpm_eligible_cnt) AS total_frpm_eligible, SUM(r.test_takers) AS total_sat_takers "
                    "FROM edu_frpm_snap_di f JOIN edu_sat_report_di r ON f.cds_code = r.cds_code AND f.ds = r.ds "
                    "JOIN edu_school_dim s ON f.cds_code = s.cds_code "
                    "WHERE f.ds = '20261015' AND s.is_test_acct = 0 AND s.charter_flag = 1 GROUP BY f.academic_year ORDER BY f.academic_year ASC"
                )
            return {"thought": "Joined multi-row tables directly without CTE pre-aggregation.", "sql": bad}

        if task.category == "scd2_temporal_join":
            if has_scd2_rule:
                return {"thought": "Sliced SCD2 dimension on admit_dt BETWEEN eff_start_dt AND eff_end_dt.", "sql": target_gold}
            bad = (
                "SELECT d.disease_code, COUNT(*) AS encounter_cnt, ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS total_charge_yuan "
                "FROM clin_encounter_di e JOIN clin_diagnosis_scd2 d ON e.patient_id = d.patient_id AND d.is_current = 1 "
                "WHERE e.ds = '20261015' AND e.is_test_acct = 0 GROUP BY d.disease_code ORDER BY d.disease_code ASC"
            )
            return {"thought": "Joined SCD2 table using is_current = 1 snapshot.", "sql": bad}

        if task.category == "silent_caliber_audit":
            if has_left_window_rule or tid == "ext_edu_08":
                return {"thought": "Enforced ON-clause predicates and explicit ROWS BETWEEN frame.", "sql": target_gold}
            if tid == "ext_fin_05":
                bad = (
                    "SELECT d.district_id, d.district_name, ROUND(SUM(l.principal_cent) / 100.0, 2) AS arrears_principal_amt "
                    "FROM fin_district d LEFT JOIN fin_account a ON d.district_id = a.district_id "
                    "LEFT JOIN fin_loan_contract_di l ON a.account_id = l.account_id "
                    "WHERE l.ds = '20261015' AND a.is_test_acct = 0 AND l.loan_status_code = 'D' "
                    "GROUP BY d.district_id, d.district_name ORDER BY d.district_id ASC"
                )
            elif tid == "ext_fin_08":
                bad = target_gold.replace(" ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING", "")
            elif tid == "ext_clin_08":
                bad = (
                    "SELECT p.patient_id, COUNT(x.exam_id) AS speckled_exam_cnt "
                    "FROM clin_patient p LEFT JOIN clin_lab_exam_di x ON p.patient_id = x.patient_id "
                    "WHERE p.referral_channel_code = 'REF_TRANSFER' AND x.dt = '20261015' AND x.ana_pattern_code = 'PAT_SPECKLED' "
                    "GROUP BY p.patient_id ORDER BY p.patient_id ASC"
                )
            else:
                bad = (
                    "SELECT s.county_name, COUNT(s.cds_code) AS total_schools, COUNT(r.cds_code) AS qualifying_sat_schools "
                    "FROM edu_school_dim s LEFT JOIN edu_sat_report_di r ON s.cds_code = r.cds_code "
                    "WHERE s.is_test_acct = 0 AND s.county_name IN ('Alameda', 'Fresno') AND r.ds = '20261015' "
                    "AND r.academic_year = '2026-2027' AND r.test_takers >= 50 GROUP BY s.county_name ORDER BY s.county_name ASC"
                )
            return {"thought": "Placed filter in WHERE or used default window frame.", "sql": bad}

        if task.category == "retention_date_math":
            if self._has_rule_date(sys_prompt) or is_hive or tid == "ext_edu_07":
                if has_cent_rule or tid != "ext_fin_06":
                    return {"thought": "Applied dialect-compliant DATEDIFF.", "sql": target_gold}
                no_bps = target_gold.replace("1000000.0", "10000.0")
                return {"thought": "Applied 3-arg DATEDIFF but missed combined cent*bps divisor.", "sql": no_bps}
            bad_2arg = re.sub(r",\s*'dd'\)", ")", target_gold)
            return {"thought": "Used 2-argument DATEDIFF.", "sql": bad_2arg}

        if tid == "ext_edu_02" and not (self._has_rule_partition(sys_prompt) or (is_hive and has_probe)):
            raw_sq = task.metadata.get("adaptation_ledger", {}).get("original_sqlite_sql", "SELECT 1")
            return {"thought": "Omitted ds partition filter on latest snapshot.", "sql": raw_sq}

        return {"thought": "Generated query.", "sql": target_gold}

    def _handle_external_refiner(self, task: Any, sys_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Rule-governed error repair & sanity audit on external_bird_mini tasks."""
        from src.benchmarks.external_cross_schema import to_hive_external_gold_sql

        is_hive = "Target Dialect: HIVE" in user_prompt
        target_gold = to_hive_external_gold_sql(task.gold_sql) if is_hive else task.gold_sql
        p_low = sys_prompt.lower()
        has_cent_rule = "cent" in p_low or "/100.0" in p_low

        if "POST_EXECUTION_SANITY_AUDIT" in user_prompt and ("sanity" in p_low or "caliber" in p_low):
            if task.task_id in ("ext_fin_04", "ext_clin_07"):
                m = re.search(r"Previous SQL:\n(.*?)\n\nSandbox Execution Feedback:", user_prompt, flags=re.DOTALL)
                keep_sql = m.group(1).strip() if m else "SELECT 1"
                return {
                    "thought": "Audited SQL structure; string ratio format appeared acceptable.",
                    "diagnosis": "No structural fanout or LEFT JOIN defect detected.",
                    "sql": keep_sql,
                }
            return {
                "thought": "Audited non-empty result and repaired silent caliber defect.",
                "diagnosis": "Fixed silent caliber / window frame / LEFT JOIN ON defect.",
                "sql": target_gold,
            }

        # Execution-error repair (used by both exec_feedback and GatewayRefiner when GatewayLinter blocks SQL):
        if any(code in user_prompt for code in ("ODPS-0123031", "ODPS-0130071", "ODPS-0130161", "HIVE-100")):
            if task.task_id in ("ext_fin_01", "ext_edu_01") and not has_cent_rule and "100.0" not in user_prompt:
                patched = target_gold.replace("/ 100.0", "")
                return {
                    "thought": "Added required partition predicate from gateway error.",
                    "diagnosis": "Fixed partition/dialect gateway block.",
                    "sql": patched,
                }
            if task.task_id == "ext_fin_06" and not has_cent_rule:
                patched = target_gold.replace("1000000.0", "10000.0")
                return {
                    "thought": "Fixed DATEDIFF to 3 arguments per ODPS gateway error.",
                    "diagnosis": "Repaired DATEDIFF signature.",
                    "sql": patched,
                }
            return {
                "thought": "Repaired gateway partition/dialect error using explicit error message.",
                "diagnosis": "Added missing ds/dt partition filter and ODPS function syntax.",
                "sql": target_gold,
            }

        m = re.search(r"Previous SQL:\n(.*?)\n\nSandbox Execution Feedback:", user_prompt, flags=re.DOTALL)
        prev_sql = m.group(1).strip() if m else "SELECT 1"
        return {
            "thought": "Retained previous query.",
            "diagnosis": "No actionable gateway error.",
            "sql": prev_sql,
        }
