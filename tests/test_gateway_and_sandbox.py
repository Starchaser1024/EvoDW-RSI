"""Offline unit tests for Layer 1 Gateway Linter, Layer 2 DuckDB Executor, and all 75 Gold SQLs."""

from __future__ import annotations

from collections import Counter

import pytest

from src.benchmarks import load_benchmark


@pytest.fixture(scope="module")
def benchmark():
    bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True)
    yield bm
    bm.sandbox.close()


class TestGatewayPartitionPruning:
    def test_missing_ds_partition_filter_blocked_with_exact_error(self, benchmark):
        sb = benchmark.sandbox
        for tbl in ("dwd_trade_order_di", "ods_app_event_log_di", "dws_shop_metrics_1d"):
            res = sb.run_sql(f"SELECT COUNT(*) FROM {tbl}", dialect="odps")
            assert not res["gateway_passed"]
            assert res["error_stage"] == "gateway_ast"
            assert (
                res["error_msg"]
                == f"FAILED: ODPS-0123031: Partition exception - full table scan is prohibited without partition filter on column 'ds' for table '{tbl}'"
            )

    def test_missing_dt_partition_filter_on_content_post(self, benchmark):
        res = benchmark.sandbox.run_sql("SELECT COUNT(*) FROM dwd_content_post_di", dialect="odps")
        assert not res["gateway_passed"]
        assert res["error_stage"] == "gateway_ast"
        assert "column 'dt' for table 'dwd_content_post_di'" in res["error_msg"]

    def test_fake_partition_filters_rejected(self, benchmark):
        sb = benchmark.sandbox
        bad_queries = [
            "SELECT COUNT(*) FROM dwd_trade_order_di WHERE ds IS NOT NULL",
            "SELECT COUNT(*) FROM dwd_trade_order_di WHERE ds != ''",
            "SELECT COUNT(*) FROM dwd_trade_order_di WHERE ds LIKE '%'",
            "SELECT * FROM dwd_trade_order_di WHERE ds = (SELECT MAX(ds) FROM dwd_trade_order_di)",
            (
                "SELECT o.order_id FROM dwd_trade_order_di o "
                "JOIN dwd_refund_di r ON o.order_id = r.order_id "
                "WHERE o.ds = '20261015'"
            ),
        ]
        for sql in bad_queries:
            res = sb.run_sql(sql, dialect="odps")
            assert not res["gateway_passed"], f"Expected block for: {sql}"
            assert res["error_stage"] == "gateway_ast"
            assert "ODPS-0123031" in res["error_msg"]

    def test_valid_partition_filters_and_unpartitioned_dim_pass(self, benchmark):
        sb = benchmark.sandbox
        ok_queries = [
            "SELECT COUNT(*) FROM dim_user",
            "SELECT COUNT(*) FROM dwd_trade_order_di WHERE ds = MAX_PT('dwd_trade_order_di')",
            "SELECT COUNT(*) FROM dwd_trade_order_di WHERE ds BETWEEN '20261001' AND '20261003'",
            (
                "SELECT COUNT(*) FROM dwd_trade_order_di o "
                "JOIN dwd_refund_di r ON o.order_id = r.order_id AND o.ds = r.ds "
                "WHERE o.ds = '20261015'"
            ),
        ]
        for sql in ok_queries:
            res = sb.run_sql(sql, dialect="odps")
            assert res["gateway_passed"], f"Expected pass for: {sql}; err={res['error_msg']}"
            assert res["error_stage"] == "none"


class TestGatewayDialectAndJoinHygiene:
    @pytest.mark.parametrize(
        "bad_sql",
        [
            "SELECT STRFTIME('%Y-%m-%d', reg_dt) FROM dim_user",
            "SELECT JSON_EXTRACT(ext_json, '$.page') FROM ods_app_event_log_di WHERE ds = '20261015'",
            "SELECT IFNULL(city_tier, 'UNKNOWN') FROM dim_user",
            "SELECT user_id FROM dim_user LIMIT 5, 10",
            "SELECT DATEDIFF('2026-10-10', '2026-10-01') FROM dim_user LIMIT 1",
            "SELECT DATE_ADD(TO_DATE(reg_dt, 'yyyy-mm-dd'), 1) FROM dim_user LIMIT 1",
            "SELECT DATE_FORMAT(DATEADD(TO_DATE(reg_dt, 'yyyy-mm-dd'), 1, 'dd'), 'yyyyMMdd') FROM dim_user LIMIT 1",
            "SELECT ext_json ->> '$.page' FROM ods_app_event_log_di WHERE ds = '20261015'",
            "SELECT list_price::INT FROM dim_item LIMIT 1",
            "SELECT * FROM dim_user, dim_shop",
            "SELECT * FROM dim_user CROSS JOIN dim_shop",
            "SELECT * FROM dim_user JOIN dim_shop ON 1 = 1",
        ],
    )
    def test_odps_rejects_hallucinations_and_cartesian_joins(self, benchmark, bad_sql: str):
        res = benchmark.sandbox.run_sql(bad_sql, dialect="odps")
        assert not res["gateway_passed"]
        assert res["error_stage"] == "gateway_ast"

    def test_hive_mode_dialect_rules(self, benchmark):
        sb = benchmark.sandbox
        # Hive allows 2-arg DATEDIFF, rejects MAX_PT and 3-arg DATEDIFF.
        ok_hive = sb.run_sql(
            "SELECT DATEDIFF('2026-10-05', '2026-10-01') FROM dim_user LIMIT 1",
            dialect="hive",
        )
        assert ok_hive["gateway_passed"] and ok_hive["error_stage"] == "none"
        assert ok_hive["rows"] == [(4,)]

        bad_max_pt = sb.run_sql(
            "SELECT COUNT(*) FROM dwd_trade_order_di WHERE ds = MAX_PT('dwd_trade_order_di')",
            dialect="hive",
        )
        assert not bad_max_pt["gateway_passed"]
        assert bad_max_pt["error_stage"] == "gateway_ast"

        bad_3arg = sb.run_sql(
            "SELECT DATEDIFF('2026-10-05', '2026-10-01', 'dd') FROM dim_user LIMIT 1",
            dialect="hive",
        )
        assert not bad_3arg["gateway_passed"]


class TestLocalDuckDBTranspilation:
    def test_odps_date_math_across_month_boundary(self, benchmark):
        sb = benchmark.sandbox
        res = sb.run_sql(
            "SELECT "
            "DATEDIFF(TO_DATE('20261001', 'yyyymmdd'), TO_DATE('2026-09-30', 'yyyy-mm-dd'), 'dd'), "
            "TO_CHAR(DATEADD(TO_DATE('20261001', 'yyyymmdd'), -1, 'dd'), 'yyyymmdd') "
            "FROM dim_user LIMIT 1",
            dialect="odps",
        )
        assert res["gateway_passed"] and res["error_stage"] == "none"
        assert res["rows"] == [(1, "20260930")]

    def test_max_pt_and_get_json_object_resilience(self, benchmark):
        sb = benchmark.sandbox
        res = sb.run_sql(
            "SELECT MAX_PT('dwd_trade_order_di'), "
            "COUNT(*), "
            "COUNT(GET_JSON_OBJECT(ext_json, '$.device.os')) "
            "FROM ods_app_event_log_di WHERE ds = MAX_PT('ods_app_event_log_di')",
            dialect="odps",
        )
        assert res["gateway_passed"] and res["error_stage"] == "none"
        max_pt_val, total_cnt, valid_os_cnt = res["rows"][0]
        assert max_pt_val == "20261015"
        assert total_cnt > valid_os_cnt > 0

    def test_lateral_view_explode_and_outer_explode(self, benchmark):
        sb = benchmark.sandbox
        # Inner explode before JOIN dim_item.
        inner_res = sb.run_sql(
            "SELECT i.cate_l1, COUNT(*) AS cnt "
            "FROM ods_app_event_log_di e "
            "LATERAL VIEW explode(e.expo_item_ids) t AS item_id "
            "JOIN dim_item i ON t.item_id = i.item_id "
            "WHERE e.ds = '20261015' "
            "GROUP BY i.cate_l1",
            dialect="odps",
        )
        assert inner_res["gateway_passed"] and inner_res["error_stage"] == "none"
        assert inner_res["row_count"] > 0

        # Outer explode preserves empty array rows (`COUNT(*)` > `COUNT(t.item_id)`).
        outer_res = sb.run_sql(
            "SELECT COUNT(*), COUNT(t.item_id) "
            "FROM ods_app_event_log_di e "
            "LATERAL VIEW OUTER explode(e.expo_item_ids) t AS item_id "
            "WHERE e.ds = '20261015'",
            dialect="odps",
        )
        assert outer_res["gateway_passed"] and outer_res["error_stage"] == "none"
        tot_rows, non_null_items = outer_res["rows"][0]
        assert tot_rows > non_null_items > 0

    def test_window_and_nvl(self, benchmark):
        sb = benchmark.sandbox
        res = sb.run_sql(
            "WITH d AS ("
            "  SELECT event_id, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts DESC) AS rn "
            "  FROM ods_app_event_log_di WHERE ds = '20261015'"
            ") SELECT COUNT(*), NVL(MAX(rn), 0) FROM d WHERE rn = 1",
            dialect="odps",
        )
        assert res["gateway_passed"] and res["error_stage"] == "none"
        assert res["rows"][0][0] > 0 and res["rows"][0][1] == 1


class TestAll75BenchmarkGoldSQLs:
    def test_benchmark_split_and_category_distribution(self, benchmark):
        assert len(benchmark.all_tasks) == 75
        assert len(benchmark.train) == 25
        assert len(benchmark.val) == 20
        assert len(benchmark.test) == 30
        cat_counts = Counter(t.category for t in benchmark.all_tasks)
        assert len(cat_counts) == 5
        assert all(v == 15 for v in cat_counts.values())
        for cat in cat_counts:
            assert sum(1 for t in benchmark.train if t.category == cat) == 5
            assert sum(1 for t in benchmark.val if t.category == cat) == 4
            assert sum(1 for t in benchmark.test if t.category == cat) == 6

    def test_all_75_gold_queries_execute_with_non_empty_results(self, benchmark):
        sb = benchmark.sandbox
        for task in benchmark.all_tasks:
            res = sb.run_sql(task.gold_sql, dialect="odps")
            assert res["gateway_passed"], f"{task.task_id} failed gateway: {res['error_msg']}"
            assert res["error_stage"] == "none", f"{task.task_id} runtime error: {res['error_msg']}"
            assert res["row_count"] > 0, f"{task.task_id} returned empty rows"
            all_null = all(v is None for row in res["rows"] for v in row)
            assert not all_null, f"{task.task_id} returned all-NULL rows"
