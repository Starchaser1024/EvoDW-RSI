"""Unit tests for EvoDW-RSI v2: Hard-v2 deep-water benchmark, Hive dialect evolution, and 3-module GEPA."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.agent.policy import Policy
from src.benchmarks.loader import load_benchmark
from src.benchmarks.tasks import get_all_tasks
from src.config import PROJECT_ROOT
from src.llm.mock_client import MockWarehouseLLMClient
from src.rsi.attribution import (
    detect_prompt_dialect_violations,
    repair_prompt_dialect_violations,
    validate_and_sanitize_prompt,
)
from src.rsi.engine import EvolutionEngine


def test_multi_suite_counts_and_gold_execution_odps_and_hive():
    v1_tasks = get_all_tasks(suite="v1")
    v2_tasks = get_all_tasks(suite="hard_v2")
    all_tasks = get_all_tasks(suite="all")
    assert len(v1_tasks) == 75
    assert len(v2_tasks) == 45
    assert len(all_tasks) == 120

    bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True, suite="hard_v2")
    try:
        assert len(bm.train) == 15
        assert len(bm.val) == 12
        assert len(bm.test) == 18

        cats = {t.category for t in bm.all_tasks}
        assert cats == {"semantic_linking_trap", "scd2_temporal_join", "silent_caliber_audit"}

        # Every Hard-v2 task must execute cleanly with non-empty gold rows in BOTH ODPS and Hive!
        for t in bm.all_tasks:
            assert t.gold_rows is not None and len(t.gold_rows) >= 1, f"{t.task_id} empty in ODPS"
            hive_sql = re.sub(r"MAX_PT\('[^']+'\)", "'20261015'", t.gold_sql)
            ev_hive = bm.sandbox.evaluate_sql(hive_sql, t, dialect="hive")
            assert ev_hive.gateway_passed and ev_hive.ex_match, f"{t.task_id} failed Hive gold check: {ev_hive.error_msg}"

        # Verify silent trap discrimination: naive SQL passes Gateway & returns rows yet fails ex_match!
        task_map = {t.task_id: t for t in bm.all_tasks}
        naive_sc01 = """
        SELECT s.shop_id, s.shop_name, ROUND(NVL(SUM(o.clean_gmv_yuan), 0.0), 2) AS normal_clean_gmv, COUNT(o.ds) AS normal_days
        FROM dim_shop s
        LEFT JOIN dws_shop_operating_1d o ON s.shop_id = o.shop_id
        WHERE s.shop_type = 'flagship' AND o.ds BETWEEN '20261012' AND '20261015' AND o.abnormal_flag = 'NORMAL'
        GROUP BY s.shop_id, s.shop_name ORDER BY s.shop_id ASC
        """
        ev_naive = bm.sandbox.evaluate_sql(naive_sc01, task_map["sc_01"], dialect="odps")
        assert ev_naive.gateway_passed is True
        assert bm.sandbox.run_sql(naive_sc01, dialect="odps")["row_count"] > 0
        assert ev_naive.ex_match is False

        # Verify enriched inspect_table_sample exposes distinct enum codes & numeric_range
        sample = bm.sandbox.inspect_table_sample("dwd_order_settle_di")
        profs = sample["column_profiles"]
        assert "20_SETTLED" in profs["settle_status_code"]["sample_non_empty_values"]
        assert "P01_WECHAT" in profs["pay_channel_code"]["sample_non_empty_values"]
        assert "numeric_range" in profs["settle_amt_cent"]
        assert profs["settle_amt_cent"]["numeric_range"]["max"] > 1000
    finally:
        bm.sandbox.close()


def test_hive_dialect_prompt_guards_and_hard_v2_anticheat():
    seed = Policy.load_seed(PROJECT_ROOT / "prompts" / "seed")

    # Anti-cheat must block Hard-v2 table names, task IDs, and warehouse enum code literals!
    ok_tbl, _, msg_tbl = validate_and_sanitize_prompt(
        seed.linker_prompt + "\nAlways query dwd_order_settle_di.",
        "linker",
        dialect="hive",
    )
    assert not ok_tbl and "dwd_order_settle_di" in msg_tbl

    ok_tid, _, msg_tid = validate_and_sanitize_prompt(
        seed.generator_prompt + "\nSpecial rule for scd_04.",
        "generator",
        dialect="hive",
    )
    assert not ok_tid and "task ID" in msg_tid

    ok_enum, _, msg_enum = validate_and_sanitize_prompt(
        seed.generator_prompt + "\nFilter status = '20_SETTLED'.",
        "generator",
        dialect="hive",
    )
    assert not ok_enum and "20_SETTLED" in msg_enum

    # Hive prompt dialect violation detection & negation-safe auto-repair
    bad_hive_prompt = (
        seed.generator_prompt
        + "\nNever use MAX_PT in Hive; instead use `DATEADD(dt, -7, 'dd')`, `TO_CHAR(ts, 'yyyymmdd')`, "
        "and `DATEDIFF(end_dt, start_dt, 'dd')`."
    )
    viols = detect_prompt_dialect_violations(bad_hive_prompt, dialect="hive")
    assert "DATEADD" in viols
    assert "TO_CHAR" in viols
    assert "DATEDIFF_3ARG" in viols
    assert "MAX_PT" not in viols  # Negated warning must NOT be flagged!

    repaired, rep_list = repair_prompt_dialect_violations(bad_hive_prompt, dialect="hive")
    assert len(rep_list) >= 3
    assert detect_prompt_dialect_violations(repaired, dialect="hive") == []
    assert "Never use MAX_PT in Hive" in repaired  # Negated warning preserved intact!
    assert "DATE_ADD(" in repaired
    assert "DATE_FORMAT(" in repaired


@pytest.mark.asyncio
async def test_mock_three_module_evolution_hard_v2_odps_and_hive():
    seed_policy = Policy.load_seed(PROJECT_ROOT / "prompts" / "seed")

    # 1. ODPS Hard-v2 ablation: seed vs greedy vs gepa
    bm_odps = load_benchmark(seed=2026, dialect="odps", precompute_gold=True, suite="hard_v2")
    try:
        res_seed = await EvolutionEngine(
            llm=MockWarehouseLLMClient(dialect="odps"), benchmark=bm_odps, seed_policy=seed_policy, dialect="odps"
        ).run(method="seed", budget=85)
        res_greedy = await EvolutionEngine(
            llm=MockWarehouseLLMClient(dialect="odps"), benchmark=bm_odps, seed_policy=seed_policy, dialect="odps"
        ).run(method="greedy", budget=85)
        res_gepa = await EvolutionEngine(
            llm=MockWarehouseLLMClient(dialect="odps"), benchmark=bm_odps, seed_policy=seed_policy, dialect="odps"
        ).run(method="gepa", budget=85)

        assert res_gepa.best_test_metrics.ex > res_greedy.best_test_metrics.ex >= res_seed.best_test_metrics.ex
        assert any(h.operation == "merge" for h in res_gepa.history)
        # Verify ALL THREE modules evolved in GEPA best policy!
        seed_p = res_gepa.seed_policy
        best_p = res_gepa.best_policy
        assert best_p.linker_prompt.strip() != seed_p.linker_prompt.strip()
        assert best_p.generator_prompt.strip() != seed_p.generator_prompt.strip()
        assert best_p.refiner_prompt.strip() != seed_p.refiner_prompt.strip()
    finally:
        bm_odps.sandbox.close()

    # 2. Hive Hard-v2 evolution: seed vs gepa
    bm_hive = load_benchmark(seed=2026, dialect="hive", precompute_gold=True, suite="hard_v2")
    try:
        h_seed = await EvolutionEngine(
            llm=MockWarehouseLLMClient(dialect="hive"), benchmark=bm_hive, seed_policy=seed_policy, dialect="hive"
        ).run(method="seed", budget=80)
        h_gepa = await EvolutionEngine(
            llm=MockWarehouseLLMClient(dialect="hive"), benchmark=bm_hive, seed_policy=seed_policy, dialect="hive"
        ).run(method="gepa", budget=80)

        assert h_gepa.best_test_metrics.ex > h_seed.best_test_metrics.ex
        assert detect_prompt_dialect_violations(h_gepa.best_policy.generator_prompt, dialect="hive") == []
        assert h_gepa.best_policy.linker_prompt.strip() != h_gepa.seed_policy.linker_prompt.strip()
        assert h_gepa.best_policy.generator_prompt.strip() != h_gepa.seed_policy.generator_prompt.strip()
        assert h_gepa.best_policy.refiner_prompt.strip() != h_gepa.seed_policy.refiner_prompt.strip()
    finally:
        bm_hive.sandbox.close()
