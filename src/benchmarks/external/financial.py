"""Domain 1 (`financial`): Retail Banking & Loans schema, deterministic populator, and 8 benchmark tasks."""

from __future__ import annotations

import datetime as dt
from typing import Any

from src.benchmarks.external.adaptation import make_external_task
from src.benchmarks.types import BenchmarkTask
from src.env.local_executor import bulk_insert_rows
from src.env.schema import ColumnMeta, TableMeta, WarehouseCatalog


def register_financial_tables(catalog: WarehouseCatalog) -> None:
    """Register the 4 Retail Banking & Loans (`financial`) tables into `catalog`."""
    catalog.register(
        TableMeta(
            name="fin_district",
            layer="dim",
            comment="Demographic & macroeconomic district dimension (BIRD financial domain)",
            columns=(
                ColumnMeta("district_id", "INTEGER", "Primary district key"),
                ColumnMeta("district_name", "VARCHAR", "Municipality / district name"),
                ColumnMeta("region", "VARCHAR", "Macro region (Prague / Central Bohemia / South Moravia / North Moravia)"),
                ColumnMeta("inhabitants", "INTEGER", "Total resident population"),
                ColumnMeta("unemployment_rate_str", "VARCHAR", "String-encoded unemployment percentage (e.g. '4.25%', '', 'N/A', NULL)"),
                ColumnMeta("avg_salary", "INTEGER", "Average monthly gross salary"),
            ),
            partition_col=None,
        )
    )
    catalog.register(
        TableMeta(
            name="fin_account",
            layer="dim",
            comment="Retail bank account master dimension with statement frequency codes",
            columns=(
                ColumnMeta("account_id", "VARCHAR", "Account primary key"),
                ColumnMeta("district_id", "INTEGER", "Branch district key"),
                ColumnMeta("frequency_code", "VARCHAR", "Opaque statement frequency code (probe for exact literals)"),
                ColumnMeta("open_dt", "VARCHAR", "Account opening date YYYY-MM-DD"),
                ColumnMeta("is_test_acct", "INTEGER", "1 = internal bank QA sandbox account, 0 = formal customer account"),
            ),
            partition_col=None,
        )
    )
    catalog.register(
        TableMeta(
            name="fin_loan_contract_di",
            layer="dwd",
            comment="Daily partitioned loan contract snapshot fact (amounts in cents, rates in bps)",
            columns=(
                ColumnMeta("loan_id", "VARCHAR", "Loan contract ID"),
                ColumnMeta("account_id", "VARCHAR", "Borrower account ID"),
                ColumnMeta("grant_dt", "VARCHAR", "Loan grant date YYYY-MM-DD"),
                ColumnMeta("principal_cent", "BIGINT", "Granted loan principal in integer cents (/100.0 for currency unit)"),
                ColumnMeta("interest_rate_bps", "INTEGER", "Annual interest rate in basis points (/10000.0)"),
                ColumnMeta("duration_months", "INTEGER", "Contract duration in months"),
                ColumnMeta("loan_status_code", "VARCHAR", "Loan status code ('A'=closed_ok, 'B'=closed_default, 'C'=running_ok, 'D'=running_arrears)"),
                ColumnMeta("ds", "VARCHAR", "Snapshot partition key YYYYMMDD", is_partition=True),
            ),
            partition_col="ds",
        )
    )
    catalog.register(
        TableMeta(
            name="fin_card_txn_di",
            layer="dwd",
            comment="Daily partitioned account & card transaction ledger (multiple rows per account; composite FK account_id + district_id)",
            columns=(
                ColumnMeta("txn_id", "VARCHAR", "Transaction primary ID"),
                ColumnMeta("account_id", "VARCHAR", "Account ID"),
                ColumnMeta("district_id", "INTEGER", "Branch district ID (composite FK with account_id)"),
                ColumnMeta("txn_dt", "VARCHAR", "Transaction date YYYY-MM-DD"),
                ColumnMeta("txn_type_code", "VARCHAR", "Direction code ('PRIJEM'=credit inflow, 'VYDAJ'=withdrawal outflow)"),
                ColumnMeta("amount_cent", "BIGINT", "Transaction amount in integer cents (/100.0)"),
                ColumnMeta("k_symbol", "VARCHAR", "Payment characterization symbol ('UVER', 'SIPO', 'SLUZBY', '', NULL)"),
                ColumnMeta("ds", "VARCHAR", "Partition date YYYYMMDD", is_partition=True),
            ),
            partition_col="ds",
        )
    )


def populate_financial_tables(con: Any) -> None:
    """Create and deterministically populate the 4 `financial` tables in DuckDB."""
    con.execute(
        """
        CREATE TABLE fin_district (
            district_id INTEGER,
            district_name VARCHAR,
            region VARCHAR,
            inhabitants INTEGER,
            unemployment_rate_str VARCHAR,
            avg_salary INTEGER
        );
        CREATE TABLE fin_account (
            account_id VARCHAR,
            district_id INTEGER,
            frequency_code VARCHAR,
            open_dt VARCHAR,
            is_test_acct INTEGER
        );
        CREATE TABLE fin_loan_contract_di (
            loan_id VARCHAR,
            account_id VARCHAR,
            grant_dt VARCHAR,
            principal_cent BIGINT,
            interest_rate_bps INTEGER,
            duration_months INTEGER,
            loan_status_code VARCHAR,
            ds VARCHAR
        );
        CREATE TABLE fin_card_txn_di (
            txn_id VARCHAR,
            account_id VARCHAR,
            district_id INTEGER,
            txn_dt VARCHAR,
            txn_type_code VARCHAR,
            amount_cent BIGINT,
            k_symbol VARCHAR,
            ds VARCHAR
        );
        """
    )

    regions = ["Prague", "Central Bohemia", "South Moravia", "North Moravia"]
    unemp_samples = ["2.15%", "3.40%", "5.10%", "6.85%", "", "N/A", None, "4.20%", "3.95%", "7.10%", "", "2.80%"]
    dist_rows = []
    for did in range(1, 13):
        reg = regions[(did - 1) % len(regions)]
        dist_rows.append(
            (
                did,
                f"District_{did:02d}",
                reg,
                55_000 + did * 18_500,
                unemp_samples[did - 1],
                28_000 + (4 - ((did - 1) % 4)) * 3_400,
            )
        )
    bulk_insert_rows(con, "fin_district", dist_rows)

    freqs = ["POPLATEK MESICNE", "POPLATEK TYDNE", "POPLATEK PO OBRATU"]
    acct_rows = []
    acct_dist_map: dict[str, int] = {}
    acct_open_map: dict[str, dt.date] = {}
    for i in range(1, 61):
        aid = f"ACC{i:04d}"
        did = ((i - 1) % 12) + 1
        open_d = dt.date(2024, 6, 1) + dt.timedelta(days=(i * 11) % 620)
        is_test = 1 if i % 15 == 0 else 0
        acct_dist_map[aid] = did
        acct_open_map[aid] = open_d
        acct_rows.append((aid, did, freqs[i % 3], open_d.strftime("%Y-%m-%d"), is_test))
    bulk_insert_rows(con, "fin_account", acct_rows)

    loan_statuses = ["A", "B", "C", "D"]
    loan_rows = []
    txn_rows = []
    txn_seq = 1
    for ds in ("20261013", "20261014", "20261015"):
        for i in range(1, 46):
            aid = f"ACC{i:04d}"
            did = acct_dist_map[aid]
            # Leave districts 11 and 12 without status 'D' loans so LEFT JOIN test catches dropped districts!
            if did in (11, 12):
                l_stat = loan_statuses[i % 3]
            else:
                l_stat = loan_statuses[i % 4]
            lag_days = 90 + (i * 23) % 360
            grant_d = acct_open_map[aid] + dt.timedelta(days=lag_days)
            principal_c = (80_000 + (i * 17_500)) * 100 + (0 if ds == "20261015" else -50_000)
            rate_bps = 380 + (i % 6) * 65
            dur = [12, 24, 36, 48, 60][i % 5]
            loan_rows.append(
                (f"LN_{ds}_{i:03d}", aid, grant_d.strftime("%Y-%m-%d"), principal_c, rate_bps, dur, l_stat, ds)
            )

        ksym_choices = ["UVER", "SIPO", "SLUZBY", "", None]
        for i in range(1, 61):
            aid = f"ACC{i:04d}"
            did = acct_dist_map[aid]
            n_txns = 3 if i <= 30 else 2
            for t_idx in range(n_txns):
                tx_d = dt.date(2026, 10, 10 + ((i + t_idx) % 6))
                ttype = "VYDAJ" if (t_idx % 2 == 0) else "PRIJEM"
                amt_c = (1_250 + i * 140 + t_idx * 330) * 100
                ksym = ksym_choices[(i + t_idx) % len(ksym_choices)]
                txn_rows.append(
                    (f"TXN_{ds}_{txn_seq:05d}", aid, did, tx_d.strftime("%Y-%m-%d"), ttype, amt_c, ksym, ds)
                )
                txn_seq += 1

    bulk_insert_rows(con, "fin_loan_contract_di", loan_rows)
    bulk_insert_rows(con, "fin_card_txn_di", txn_rows)


def get_financial_tasks() -> list[BenchmarkTask]:
    """Return the 8 cross-schema Retail Banking & Loans (`financial`) tasks."""
    return [
        make_external_task(
            task_id="ext_fin_01",
            domain="financial",
            category="partition_pruning",
            split="train",
            question=(
                "查询 fin_loan_contract_di 最新分区中正式客户账户（排除银行内部压测账户）的贷款合同数（loan_cnt）"
                "及贷款本金总额（total_principal_amt，由分换算为标准货币单位并保留2位小数），按所属大区（region）升序排列。"
            ),
            hint="关联 fin_account 与 fin_district；仅统计正式客户账户；输出列 region, loan_cnt, total_principal_amt。",
            original_sqlite_sql="""
SELECT d.region, COUNT(*) AS loan_cnt, ROUND(SUM(IFNULL(l.principal_cent, 0)) / 100.0, 2) AS total_principal_amt
FROM fin_loan_contract_di l
JOIN fin_account a ON l.account_id = a.account_id
JOIN fin_district d ON a.district_id = d.district_id
WHERE a.is_test_acct = 0
GROUP BY d.region
ORDER BY d.region ASC
LIMIT 0, 10
""",
            gold_sql="""
SELECT d.region, COUNT(*) AS loan_cnt, ROUND(SUM(NVL(l.principal_cent, 0)) / 100.0, 2) AS total_principal_amt
FROM fin_loan_contract_di l
JOIN fin_account a ON l.account_id = a.account_id
JOIN fin_district d ON a.district_id = d.district_id
WHERE l.ds = MAX_PT('fin_loan_contract_di')
  AND a.is_test_acct = 0
GROUP BY d.region
ORDER BY d.region ASC
LIMIT 10
""",
            ast_transformations=[
                "IFNULL -> NVL",
                "LIMIT offset, count -> LIMIT count",
                "FACT_PARTITION_PREDICATE_INJECTION (MAX_PT)",
            ],
            injected_partition_predicates=["l.ds = MAX_PT('fin_loan_contract_di')"],
            unpartitioned_dim_tables=["fin_account", "fin_district"],
            caliber_trap="cent_to_unit_scaling_and_test_acct_exclusion",
        ),
        make_external_task(
            task_id="ext_fin_02",
            domain="financial",
            category="semantic_linking_trap",
            split="train",
            question=(
                "在 fin_loan_contract_di 最新分区中，筛选正式客户账户中账单周期为“按月出账（monthly）”且处于违约或逾期状态"
                "（loan_status_code 为 B 或 D）的风险贷款，按 loan_status_code 分组统计合同笔数（problem_loan_cnt）"
                "与风险本金总额（problem_principal_amt，换算为标准单位保留2位小数），按 loan_status_code 升序排列。"
            ),
            hint="先探查 fin_account.frequency_code 的真实内部编码；输出列 loan_status_code, problem_loan_cnt, problem_principal_amt。",
            original_sqlite_sql="""
SELECT l.loan_status_code, COUNT(*) AS problem_loan_cnt, ROUND(SUM(l.principal_cent) / 100.0, 2) AS problem_principal_amt
FROM fin_loan_contract_di l
JOIN fin_account a ON l.account_id = a.account_id
WHERE a.is_test_acct = 0
  AND a.frequency_code = 'POPLATEK MESICNE'
  AND l.loan_status_code IN ('B', 'D')
GROUP BY l.loan_status_code
ORDER BY l.loan_status_code ASC
""",
            gold_sql="""
SELECT l.loan_status_code, COUNT(*) AS problem_loan_cnt, ROUND(SUM(l.principal_cent) / 100.0, 2) AS problem_principal_amt
FROM fin_loan_contract_di l
JOIN fin_account a ON l.account_id = a.account_id
WHERE l.ds = MAX_PT('fin_loan_contract_di')
  AND a.is_test_acct = 0
  AND a.frequency_code = 'POPLATEK MESICNE'
  AND l.loan_status_code IN ('B', 'D')
GROUP BY l.loan_status_code
ORDER BY l.loan_status_code ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (MAX_PT)"],
            injected_partition_predicates=["l.ds = MAX_PT('fin_loan_contract_di')"],
            unpartitioned_dim_tables=["fin_account"],
            caliber_trap="opaque_czech_frequency_enum_and_cent_scaling",
        ),
        make_external_task(
            task_id="ext_fin_03",
            domain="financial",
            category="fanout_dedup",
            split="val",
            question=(
                "针对 '20261015' 快照中持有贷款合同的正式客户账户，分别汇总每个账户的贷款本金总额（loan_principal_amt）"
                "以及同日 fin_card_txn_di 中支取类交易（VYDAJ）的交易总额（withdrawal_amt，无支取记 0.0，均换算为标准单位保留2位小数），"
                "按 loan_principal_amt 降序、account_id 升序返回前 10 个账户。"
            ),
            hint="同一账户存在多笔卡交易流水，严禁未预聚合直接关联导致贷款本金扇出膨胀；输出列 account_id, loan_principal_amt, withdrawal_amt。",
            original_sqlite_sql="""
WITH loan_agg AS (
  SELECT account_id, SUM(principal_cent) AS principal_cent
  FROM fin_loan_contract_di
  GROUP BY account_id
),
txn_agg AS (
  SELECT account_id, SUM(amount_cent) AS withdrawal_cent
  FROM fin_card_txn_di
  WHERE txn_type_code = 'VYDAJ'
  GROUP BY account_id
)
SELECT l.account_id,
       ROUND(l.principal_cent / 100.0, 2) AS loan_principal_amt,
       ROUND(IFNULL(t.withdrawal_cent, 0) / 100.0, 2) AS withdrawal_amt
FROM loan_agg l
JOIN fin_account a ON l.account_id = a.account_id
LEFT JOIN txn_agg t ON l.account_id = t.account_id
WHERE a.is_test_acct = 0
ORDER BY loan_principal_amt DESC, l.account_id ASC
LIMIT 0, 10
""",
            gold_sql="""
WITH loan_agg AS (
  SELECT account_id, SUM(principal_cent) AS principal_cent
  FROM fin_loan_contract_di
  WHERE ds = '20261015'
  GROUP BY account_id
),
txn_agg AS (
  SELECT account_id, SUM(amount_cent) AS withdrawal_cent
  FROM fin_card_txn_di
  WHERE ds = '20261015'
    AND txn_type_code = 'VYDAJ'
  GROUP BY account_id
)
SELECT l.account_id,
       ROUND(l.principal_cent / 100.0, 2) AS loan_principal_amt,
       ROUND(NVL(t.withdrawal_cent, 0) / 100.0, 2) AS withdrawal_amt
FROM loan_agg l
JOIN fin_account a ON l.account_id = a.account_id
LEFT JOIN txn_agg t ON l.account_id = t.account_id
WHERE a.is_test_acct = 0
ORDER BY loan_principal_amt DESC, l.account_id ASC
LIMIT 10
""",
            ast_transformations=[
                "IFNULL -> NVL",
                "LIMIT offset, count -> LIMIT count",
                "FACT_PARTITION_PREDICATE_INJECTION",
            ],
            injected_partition_predicates=["fin_loan_contract_di.ds = '20261015'", "fin_card_txn_di.ds = '20261015'"],
            unpartitioned_dim_tables=["fin_account"],
            caliber_trap="one_to_many_loan_card_txn_fanout",
        ),
        make_external_task(
            task_id="ext_fin_04",
            domain="financial",
            category="ratio_null_handling",
            split="val",
            question=(
                "结合 fin_district 与最新分区的 fin_loan_contract_di（仅限正式客户账户），按大区（region）统计："
                "（1）大区内有效解析的地区平均失业率数值（avg_unemp_pct，去除 '%' 号并忽略空串与 'N/A'，保留2位小数）；"
                "（2）该大区问题贷款（状态 B 或 D）笔数占总贷款笔数的比例（problem_loan_ratio，保留4位小数），按 region 升序排列。"
            ),
            hint="地区平均失业率按 fin_district 去重地区口径计算后再关联大区贷款统计，避免按贷款行数重复加权；输出列 region, avg_unemp_pct, problem_loan_ratio。",
            original_sqlite_sql="""
WITH dist_clean AS (
  SELECT region,
         AVG(CASE
               WHEN unemployment_rate_str IS NULL OR unemployment_rate_str IN ('', 'N/A') THEN NULL
               ELSE CAST(REPLACE(unemployment_rate_str, '%', '') AS REAL)
             END) AS avg_unemp
  FROM fin_district
  GROUP BY region
),
loan_reg AS (
  SELECT d.region,
         COUNT(*) AS total_loans,
         SUM(CASE WHEN l.loan_status_code IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans
  FROM fin_loan_contract_di l
  JOIN fin_account a ON l.account_id = a.account_id
  JOIN fin_district d ON a.district_id = d.district_id
  WHERE a.is_test_acct = 0
  GROUP BY d.region
)
SELECT dc.region,
       ROUND(dc.avg_unemp, 2) AS avg_unemp_pct,
       ROUND(1.0 * lr.bad_loans / NULLIF(lr.total_loans, 0), 4) AS problem_loan_ratio
FROM dist_clean dc
JOIN loan_reg lr ON dc.region = lr.region
ORDER BY dc.region ASC
""",
            gold_sql="""
WITH dist_clean AS (
  SELECT region,
         AVG(CASE
               WHEN unemployment_rate_str IS NULL OR TRIM(unemployment_rate_str) IN ('', 'N/A') THEN NULL
               ELSE CAST(REPLACE(TRIM(unemployment_rate_str), '%', '') AS DOUBLE)
             END) AS avg_unemp
  FROM fin_district
  GROUP BY region
),
loan_reg AS (
  SELECT d.region,
         COUNT(*) AS total_loans,
         SUM(CASE WHEN l.loan_status_code IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans
  FROM fin_loan_contract_di l
  JOIN fin_account a ON l.account_id = a.account_id
  JOIN fin_district d ON a.district_id = d.district_id
  WHERE l.ds = MAX_PT('fin_loan_contract_di')
    AND a.is_test_acct = 0
  GROUP BY d.region
)
SELECT dc.region,
       ROUND(dc.avg_unemp, 2) AS avg_unemp_pct,
       ROUND(1.0 * lr.bad_loans / NULLIF(lr.total_loans, 0), 4) AS problem_loan_ratio
FROM dist_clean dc
JOIN loan_reg lr ON dc.region = lr.region
ORDER BY dc.region ASC
""",
            ast_transformations=["REAL -> DOUBLE", "FACT_PARTITION_PREDICATE_INJECTION (MAX_PT)"],
            injected_partition_predicates=["l.ds = MAX_PT('fin_loan_contract_di')"],
            unpartitioned_dim_tables=["fin_district", "fin_account"],
            caliber_trap="dirty_string_percentage_and_district_grain_weighting",
        ),
        make_external_task(
            task_id="ext_fin_05",
            domain="financial",
            category="silent_caliber_audit",
            split="test",
            question=(
                "面向全部 12 个行政地区（fin_district），统计每个地区在 '20261015' 快照下正式客户账户处于运行中逾期状态"
                "（loan_status_code = 'D'）的贷款本金总额（arrears_principal_amt，换算为标准单位保留2位小数，无逾期贷款的地区必须保留并记 0.0），"
                "输出 district_id, district_name, arrears_principal_amt，按 district_id 升序排列。"
            ),
            hint="注意保留零逾期贷款的行政地区（勿在 WHERE 子句过滤右表条件导致地区丢失）。",
            original_sqlite_sql="""
WITH arrears AS (
  SELECT a.district_id, SUM(l.principal_cent) AS arrears_cent
  FROM fin_loan_contract_di l
  JOIN fin_account a ON l.account_id = a.account_id
  WHERE a.is_test_acct = 0 AND l.loan_status_code = 'D'
  GROUP BY a.district_id
)
SELECT d.district_id,
       d.district_name,
       ROUND(IFNULL(ar.arrears_cent, 0) / 100.0, 2) AS arrears_principal_amt
FROM fin_district d
LEFT JOIN arrears ar ON d.district_id = ar.district_id
ORDER BY d.district_id ASC
""",
            gold_sql="""
WITH arrears AS (
  SELECT a.district_id, SUM(l.principal_cent) AS arrears_cent
  FROM fin_loan_contract_di l
  JOIN fin_account a ON l.account_id = a.account_id
  WHERE l.ds = '20261015'
    AND a.is_test_acct = 0
    AND l.loan_status_code = 'D'
  GROUP BY a.district_id
)
SELECT d.district_id,
       d.district_name,
       ROUND(NVL(ar.arrears_cent, 0) / 100.0, 2) AS arrears_principal_amt
FROM fin_district d
LEFT JOIN arrears ar ON d.district_id = ar.district_id
ORDER BY d.district_id ASC
""",
            ast_transformations=["IFNULL -> NVL", "FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["l.ds = '20261015'"],
            unpartitioned_dim_tables=["fin_district", "fin_account"],
            caliber_trap="left_join_right_table_where_drop",
        ),
        make_external_task(
            task_id="ext_fin_06",
            domain="financial",
            category="retention_date_math",
            split="test",
            question=(
                "在 '20261015' 分区的 fin_loan_contract_di 中关联正式客户账户（fin_account），计算放款日（grant_dt）距开户日（open_dt）的天数："
                "小于 180 天归为 'fast_lt_180d'，大于等于 180 天归为 'seasoned_ge_180d'；统计各分组的合同笔数（loan_cnt）"
                "及单笔平均年化利息金额（avg_annual_interest_amt，按 principal_cent * interest_rate_bps 换算为标准货币单位后取均值并保留2位小数），"
                "按 tenure_bucket 升序排列。"
            ),
            hint="principal_cent 为分（/100），interest_rate_bps 为基点（/10000），合并除数为 1000000.0；输出列 tenure_bucket, loan_cnt, avg_annual_interest_amt。",
            original_sqlite_sql="""
SELECT CASE
         WHEN JULIANDAY(l.grant_dt) - JULIANDAY(a.open_dt) < 180 THEN 'fast_lt_180d'
         ELSE 'seasoned_ge_180d'
       END AS tenure_bucket,
       COUNT(*) AS loan_cnt,
       ROUND(AVG(l.principal_cent * l.interest_rate_bps / 1000000.0), 2) AS avg_annual_interest_amt
FROM fin_loan_contract_di l
JOIN fin_account a ON l.account_id = a.account_id
WHERE a.is_test_acct = 0
GROUP BY 1
ORDER BY tenure_bucket ASC
""",
            gold_sql="""
WITH tagged AS (
  SELECT CASE
           WHEN DATEDIFF(TO_DATE(l.grant_dt, 'yyyy-mm-dd'), TO_DATE(a.open_dt, 'yyyy-mm-dd'), 'dd') < 180 THEN 'fast_lt_180d'
           ELSE 'seasoned_ge_180d'
         END AS tenure_bucket,
         l.principal_cent * l.interest_rate_bps / 1000000.0 AS annual_interest_amt
  FROM fin_loan_contract_di l
  JOIN fin_account a ON l.account_id = a.account_id
  WHERE l.ds = '20261015'
    AND a.is_test_acct = 0
)
SELECT tenure_bucket,
       COUNT(*) AS loan_cnt,
       ROUND(AVG(annual_interest_amt), 2) AS avg_annual_interest_amt
FROM tagged
GROUP BY tenure_bucket
ORDER BY tenure_bucket ASC
""",
            ast_transformations=[
                "JULIANDAY_DIFF -> DATEDIFF(end, start, 'dd')",
                "FACT_PARTITION_PREDICATE_INJECTION",
            ],
            injected_partition_predicates=["l.ds = '20261015'"],
            unpartitioned_dim_tables=["fin_account"],
            caliber_trap="odps_3arg_datediff_and_cent_bps_combined_divisor",
        ),
        make_external_task(
            task_id="ext_fin_07",
            domain="financial",
            category="ratio_null_handling",
            split="test",
            question=(
                "统计 fin_card_txn_di 在 '20261013' 至 '20261015' 分区中正式客户账户的交易分布：使用复合键（account_id 与 district_id）"
                "关联 fin_account，将 k_symbol 中的 NULL 与空串统一归并为 'UNSPECIFIED'，计算各 k_symbol_clean 的交易笔数（txn_cnt）"
                "及净资金流入额（net_flow_amt，PRIJEM 记正、VYDAJ 记负，换算为标准单位保留2位小数），按 txn_cnt 降序、k_symbol_clean 升序排列。"
            ),
            hint="输出列 k_symbol_clean, txn_cnt, net_flow_amt。",
            original_sqlite_sql="""
SELECT COALESCE(NULLIF(TRIM(t.k_symbol), ''), 'UNSPECIFIED') AS k_symbol_clean,
       COUNT(*) AS txn_cnt,
       ROUND(SUM(CASE WHEN t.txn_type_code = 'PRIJEM' THEN t.amount_cent ELSE -t.amount_cent END) / 100.0, 2) AS net_flow_amt
FROM fin_card_txn_di t
JOIN fin_account a ON t.account_id = a.account_id AND t.district_id = a.district_id
WHERE a.is_test_acct = 0
GROUP BY COALESCE(NULLIF(TRIM(t.k_symbol), ''), 'UNSPECIFIED')
ORDER BY txn_cnt DESC, k_symbol_clean ASC
""",
            gold_sql="""
SELECT COALESCE(NULLIF(TRIM(t.k_symbol), ''), 'UNSPECIFIED') AS k_symbol_clean,
       COUNT(*) AS txn_cnt,
       ROUND(SUM(CASE WHEN t.txn_type_code = 'PRIJEM' THEN t.amount_cent ELSE -t.amount_cent END) / 100.0, 2) AS net_flow_amt
FROM fin_card_txn_di t
JOIN fin_account a ON t.account_id = a.account_id AND t.district_id = a.district_id
WHERE t.ds BETWEEN '20261013' AND '20261015'
  AND a.is_test_acct = 0
GROUP BY COALESCE(NULLIF(TRIM(t.k_symbol), ''), 'UNSPECIFIED')
ORDER BY txn_cnt DESC, k_symbol_clean ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (BETWEEN)"],
            injected_partition_predicates=["t.ds BETWEEN '20261013' AND '20261015'"],
            unpartitioned_dim_tables=["fin_account"],
            caliber_trap="composite_fk_null_empty_coalesce_and_signed_flow",
        ),
        make_external_task(
            task_id="ext_fin_08",
            domain="financial",
            category="silent_caliber_audit",
            split="test",
            question=(
                "在 fin_card_txn_di 的 '20261015' 分区中，筛选所属大区为 'Prague' 的正式客户账户支取类交易（VYDAJ），"
                "按交易日期（txn_dt）汇总当日支取金额（daily_withdrawal_amt），并计算截至当日的累计支取金额（cum_withdrawal_amt）"
                "以及窗口内最后一个交易日的当日支取金额（last_day_withdrawal_amt，均换算为标准单位保留2位小数），按 txn_dt 升序排列。"
            ),
            hint="LAST_VALUE 与累计 SUM 窗口必须显式声明 ROWS BETWEEN 窗口帧；输出列 txn_dt, daily_withdrawal_amt, cum_withdrawal_amt, last_day_withdrawal_amt。",
            original_sqlite_sql="""
WITH daily AS (
  SELECT t.txn_dt,
         ROUND(SUM(t.amount_cent) / 100.0, 2) AS daily_withdrawal_amt
  FROM fin_card_txn_di t
  JOIN fin_account a ON t.account_id = a.account_id AND t.district_id = a.district_id
  JOIN fin_district d ON a.district_id = d.district_id
  WHERE a.is_test_acct = 0 AND d.region = 'Prague' AND t.txn_type_code = 'VYDAJ'
  GROUP BY t.txn_dt
)
SELECT txn_dt,
       daily_withdrawal_amt,
       ROUND(SUM(daily_withdrawal_amt) OVER (ORDER BY txn_dt ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cum_withdrawal_amt,
       LAST_VALUE(daily_withdrawal_amt) OVER (ORDER BY txn_dt ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_day_withdrawal_amt
FROM daily
ORDER BY txn_dt ASC
""",
            gold_sql="""
WITH daily AS (
  SELECT t.txn_dt,
         ROUND(SUM(t.amount_cent) / 100.0, 2) AS daily_withdrawal_amt
  FROM fin_card_txn_di t
  JOIN fin_account a ON t.account_id = a.account_id AND t.district_id = a.district_id
  JOIN fin_district d ON a.district_id = d.district_id
  WHERE t.ds = '20261015'
    AND a.is_test_acct = 0
    AND d.region = 'Prague'
    AND t.txn_type_code = 'VYDAJ'
  GROUP BY t.txn_dt
)
SELECT txn_dt,
       daily_withdrawal_amt,
       ROUND(SUM(daily_withdrawal_amt) OVER (ORDER BY txn_dt ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cum_withdrawal_amt,
       LAST_VALUE(daily_withdrawal_amt) OVER (ORDER BY txn_dt ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_day_withdrawal_amt
FROM daily
ORDER BY txn_dt ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["t.ds = '20261015'"],
            unpartitioned_dim_tables=["fin_account", "fin_district"],
            caliber_trap="explicit_rows_between_frame_on_last_value",
        ),
    ]
