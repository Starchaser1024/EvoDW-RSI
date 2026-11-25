"""Unit tests for external cross-schema warehouse suite (`external_bird_mini`),
SQLite-to-ODPS AST adaptation ledgers, zero schema leakage, and fixed-budget 4-method comparison."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from scripts.convert_bird_to_odps import transpile_sqlite_to_odps_with_ledger
from src.agent.policy import Policy
from src.benchmarks import (
    EXTERNAL_DOMAINS,
    load_benchmark,
    to_hive_external_gold_sql,
)
from src.cli import main as cli_main
from src.config import PROJECT_ROOT
from src.eval.fixed_budget import (
    compare_methods_fixed_budget,
    resolve_transfer_gepa_policy,
    verify_zero_cross_schema_leakage,
)
from src.llm.mock_client import MockWarehouseLLMClient


@pytest.fixture(scope="module")
def ext_benchmark():
    bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True, suite="external_bird_mini")
    yield bm
    bm.sandbox.close()


class TestExternalCrossSchemaSuiteAndAdaptationLedger:
    def test_24_tasks_across_3_domains_11_tables_odps_and_hive_gold_execution(self, ext_benchmark):
        assert len(ext_benchmark.all_tasks) == 24
        assert len(ext_benchmark.train) == 6
        assert len(ext_benchmark.val) == 6
        assert len(ext_benchmark.test) == 12

        tables = ext_benchmark.sandbox.catalog.list_tables()
        assert len(tables) == 11
        # Verify heterogeneous partition columns (`ds`, `dt`, and unpartitioned `dim`)
        part_cols = Counter(t.partition_col for t in tables)
        assert part_cols[None] == 5
        assert part_cols["ds"] == 5
        assert part_cols["dt"] == 1

        domain_counts = Counter(t.metadata["domain"] for t in ext_benchmark.all_tasks)
        assert set(domain_counts.keys()) == set(EXTERNAL_DOMAINS)
        assert all(cnt == 8 for cnt in domain_counts.values())

        for task in ext_benchmark.all_tasks:
            ledger = task.metadata.get("adaptation_ledger")
            assert isinstance(ledger, dict)
            assert ledger["domain"] in EXTERNAL_DOMAINS
            assert "SELECT" in ledger["original_sqlite_sql"].upper()
            assert len(ledger["ast_transformations"]) >= 1
            assert len(ledger["injected_partition_predicates"]) >= 1
            assert "MUST NOT be conflated" in ledger["disclaimer"]

            # Verify ODPS execution
            res_odps = ext_benchmark.sandbox.run_sql(task.gold_sql, dialect="odps")
            assert res_odps["gateway_passed"], f"{task.task_id} ODPS gateway failed: {res_odps['error_msg']}"
            assert res_odps["error_stage"] == "none", f"{task.task_id} ODPS runtime error: {res_odps['error_msg']}"
            assert res_odps["row_count"] > 0

            # Verify Hive execution matches cached Gold rows
            hive_sql = to_hive_external_gold_sql(task.gold_sql)
            ev_hive = ext_benchmark.sandbox.evaluate_sql(hive_sql, task, dialect="hive")
            assert ev_hive.gateway_passed and ev_hive.ex_match, (
                f"{task.task_id} Hive Gold mismatch: {ev_hive.error_msg} | {ev_hive.diff_reason}"
            )

    def test_sqlite_to_odps_ast_transpiler_preserves_left_join_and_records_ledger(self, ext_benchmark):
        sqlite_q = (
            "SELECT d.district_id, STRFTIME('%Y-%m', l.grant_dt) AS ym, IFNULL(SUM(l.principal_cent), 0) AS tot "
            "FROM fin_district d LEFT JOIN fin_loan_contract_di l ON d.district_id = l.duration_months "
            "GROUP BY d.district_id, STRFTIME('%Y-%m', l.grant_dt) LIMIT 0, 5"
        )
        odps_sql, ledger = transpile_sqlite_to_odps_with_ledger(
            sqlite_q,
            partitioned_tables={"fin_loan_contract_di": "ds"},
            default_ds="20261015",
        )
        assert "IFNULL -> COALESCE/NVL" in ledger["ast_transformations"]
        assert "STRFTIME -> TO_CHAR(TO_DATE(...))" in ledger["ast_transformations"]
        assert "LIMIT offset, count -> LIMIT count OFFSET offset" in ledger["ast_transformations"]
        assert any(p.startswith("ON l.ds =") for p in ledger["injected_partition_predicates"])
        assert "fin_district" in ledger["unpartitioned_dim_tables"]

        res = ext_benchmark.sandbox.run_sql(odps_sql, dialect="odps")
        assert res["gateway_passed"] and res["error_stage"] == "none"
        assert res["row_count"] == 5


class TestZeroLeakageAndFixedBudgetComparison:
    def test_transferred_gepa_policies_have_zero_ecommerce_or_external_leakage(self, ext_benchmark):
        for dial in ("odps", "hive"):
            pol = resolve_transfer_gepa_policy(dialect=dial, project_root=PROJECT_ROOT)
            audit = verify_zero_cross_schema_leakage(pol, ext_benchmark.sandbox.catalog)
            assert audit["passed"] is True, f"Leakage detected in {dial} policy: {audit}"

        seed = Policy.load_seed(PROJECT_ROOT / "prompts/seed")
        poisoned = seed.clone_with_mutation(
            new_policy_id="poisoned_v1",
            target_module="generator",
            new_prompt=seed.generator_prompt + "\nFilter fin_loan_contract_di on POPLATEK MESICNE.",
            generation=1,
        )
        bad_audit = verify_zero_cross_schema_leakage(poisoned, ext_benchmark.sandbox.catalog)
        assert bad_audit["passed"] is False
        assert "fin_loan_contract_di" in bad_audit["external_table_hits"]
        assert "POPLATEK MESICNE" in bad_audit["external_enum_hits"]

    @pytest.mark.asyncio
    async def test_fixed_budget_4_method_harness_and_cli(self, ext_benchmark, tmp_path: Path):
        llm = MockWarehouseLLMClient(dialect="odps")
        summary, trajs = await compare_methods_fixed_budget(
            tasks=ext_benchmark.all_tasks,
            sandbox=ext_benchmark.sandbox,
            llm=llm,
            dialect="odps",
            max_llm_calls_per_task=4,
        )
        assert summary["zero_cross_schema_leakage_audit"]["passed"] is True
        methods = summary["methods"]
        for m_name, m_stat in methods.items():
            assert m_stat["budget_compliant"] is True, f"{m_name} exceeded call cap!"
            assert m_stat["max_observed_llm_calls"] <= 4
            assert len(trajs[m_name]) == 24

        # Execution feedback fixes loud gateway errors (`executable_sql_rate` rises),
        # yet leaves silent business caliber errors high without Linker probing or sanity audit.
        assert methods["exec_feedback"]["executable_sql_rate"] > methods["direct_sql"]["executable_sql_rate"]
        assert methods["exec_feedback"]["silent_caliber_error_rate"] >= 0.45

        # Zero-shot GEPA policy transfer generalizes structural warehouse rules across unseen schemas
        # while remaining < 1.0 on domain-specific dirty-string ratio formats.
        assert methods["evodw_gepa_transfer"]["ex"] > methods["evodw_seed"]["ex"]
        assert methods["evodw_gepa_transfer"]["ex"] > methods["exec_feedback"]["ex"]
        assert methods["evodw_gepa_transfer"]["ex"] > methods["direct_sql"]["ex"]
        assert methods["evodw_gepa_transfer"]["ex"] < 1.0
        assert (
            methods["evodw_gepa_transfer"]["silent_caliber_error_rate"]
            < methods["exec_feedback"]["silent_caliber_error_rate"]
        )
        assert methods["evodw_gepa_transfer"]["probed_task_rate"] > 0.50

    def test_cli_compare_methods_subcommand(self, tmp_path: Path):
        out_json = tmp_path / "ext_compare.json"
        rc = cli_main(
            [
                "compare-methods",
                "--suite",
                "external_bird_mini",
                "--split",
                "test",
                "--max-calls",
                "4",
                "--mock",
                "--json-out",
                str(out_json),
            ]
        )
        assert rc == 0
        saved = json.loads(out_json.read_text(encoding="utf-8"))
        assert saved["task_count"] == 12
        assert saved["execution_mode"] == "mock"
        assert "MUST NOT be conflated" in saved["adaptation_disclaimer"]
