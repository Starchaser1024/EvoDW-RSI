"""Domain 3 (`california_schools`): California Public Schools schema, populator, and 8 benchmark tasks."""

from __future__ import annotations

import datetime as dt
from typing import Any

from src.benchmarks.external.adaptation import make_external_task
from src.benchmarks.types import BenchmarkTask
from src.env.local_executor import bulk_insert_rows
from src.env.schema import ColumnMeta, TableMeta, WarehouseCatalog


def register_schools_tables(catalog: WarehouseCatalog) -> None:
    """Register the 3 California Public Schools (`california_schools`) tables into `catalog`."""
    catalog.register(
        TableMeta(
            name="edu_school_dim",
            layer="dim",
            comment="Public K-12 school directory dimension (BIRD california_schools domain)",
            columns=(
                ColumnMeta("cds_code", "VARCHAR", "County-District-School unique identifier"),
                ColumnMeta("county_name", "VARCHAR", "County name"),
                ColumnMeta("district_name", "VARCHAR", "School district name"),
                ColumnMeta("school_name", "VARCHAR", "Public school name"),
                ColumnMeta("charter_flag", "INTEGER", "1 = charter school, 0 = traditional non-charter school"),
                ColumnMeta("funding_type_code", "VARCHAR", "Opaque charter funding code ('FUND_DIRECT', 'FUND_LOCAL', 'FUND_NONCHARTER', '', NULL)"),
                ColumnMeta("open_dt", "VARCHAR", "School opening date YYYY-MM-DD"),
                ColumnMeta("is_test_acct", "INTEGER", "1 = CDE placeholder test record, 0 = accredited active school"),
            ),
            partition_col=None,
        )
    )
    catalog.register(
        TableMeta(
            name="edu_frpm_snap_di",
            layer="dwd",
            comment="Partitioned Free or Reduced-Price Meal (FRPM) snapshot (composite grain cds_code + academic_year)",
            columns=(
                ColumnMeta("cds_code", "VARCHAR", "School CDS identifier"),
                ColumnMeta("academic_year", "VARCHAR", "Academic year ('2025-2026' or '2026-2027')"),
                ColumnMeta("enrollment_k12", "INTEGER", "Total K-12 student enrollment"),
                ColumnMeta("free_meal_cnt", "INTEGER", "Free-meal eligible count (nullable)"),
                ColumnMeta("frpm_eligible_cnt", "INTEGER", "Total FRPM eligible student count"),
                ColumnMeta("elig_rate_str", "VARCHAR", "String-formatted eligibility percentage (e.g. '68.4%', '', NULL)"),
                ColumnMeta("per_pupil_grant_cent", "BIGINT", "Supplemental grant per pupil in integer cents (/100.0 for USD)"),
                ColumnMeta("ds", "VARCHAR", "Snapshot partition date YYYYMMDD", is_partition=True),
            ),
            partition_col="ds",
        )
    )
    catalog.register(
        TableMeta(
            name="edu_sat_report_di",
            layer="dws",
            comment="Partitioned SAT college-readiness summary table (composite grain cds_code + academic_year; zero test_takers possible)",
            columns=(
                ColumnMeta("cds_code", "VARCHAR", "School CDS identifier"),
                ColumnMeta("academic_year", "VARCHAR", "Academic year ('2025-2026' or '2026-2027')"),
                ColumnMeta("test_takers", "INTEGER", "Number of SAT test takers (0 for cold-start/non-participating cohorts)"),
                ColumnMeta("avg_math", "INTEGER", "Mean SAT Math score (NULL when test_takers = 0)"),
                ColumnMeta("avg_reading", "INTEGER", "Mean SAT Reading score (NULL when test_takers = 0)"),
                ColumnMeta("avg_writing", "INTEGER", "Mean SAT Writing score (NULL when test_takers = 0)"),
                ColumnMeta("num_ge_1500", "INTEGER", "Students scoring >= 1500 total"),
                ColumnMeta("ds", "VARCHAR", "Snapshot partition date YYYYMMDD", is_partition=True),
            ),
            partition_col="ds",
        )
    )


def populate_schools_tables(con: Any) -> None:
    """Create and deterministically populate the 3 `california_schools` tables in DuckDB."""
    con.execute(
        """
        CREATE TABLE edu_school_dim (
            cds_code VARCHAR,
            county_name VARCHAR,
            district_name VARCHAR,
            school_name VARCHAR,
            charter_flag INTEGER,
            funding_type_code VARCHAR,
            open_dt VARCHAR,
            is_test_acct INTEGER
        );
        CREATE TABLE edu_frpm_snap_di (
            cds_code VARCHAR,
            academic_year VARCHAR,
            enrollment_k12 INTEGER,
            free_meal_cnt INTEGER,
            frpm_eligible_cnt INTEGER,
            elig_rate_str VARCHAR,
            per_pupil_grant_cent BIGINT,
            ds VARCHAR
        );
        CREATE TABLE edu_sat_report_di (
            cds_code VARCHAR,
            academic_year VARCHAR,
            test_takers INTEGER,
            avg_math INTEGER,
            avg_reading INTEGER,
            avg_writing INTEGER,
            num_ge_1500 INTEGER,
            ds VARCHAR
        );
        """
    )

    counties = ["Los Angeles", "Alameda", "Santa Clara", "Fresno"]
    fund_codes = ["FUND_DIRECT", "FUND_LOCAL", "", None]
    sch_rows = []
    for i in range(1, 37):
        cds = f"CDS{i:05d}"
        cnty = counties[(i - 1) % 4]
        dist = f"{cnty.replace(' ', '')}_USD_{(i % 3) + 1}"
        is_charter = 1 if i % 2 == 1 else 0
        f_code = fund_codes[(i // 2) % len(fund_codes)] if is_charter else "FUND_NONCHARTER"
        open_yr = 1988 + (i % 32)
        open_d = dt.date(open_yr, 9, 1)
        is_test = 1 if i % 18 == 0 else 0
        sch_rows.append((cds, cnty, dist, f"School_{i:02d}", is_charter, f_code, open_d.strftime("%Y-%m-%d"), is_test))
    bulk_insert_rows(con, "edu_school_dim", sch_rows)

    frpm_rows = []
    sat_rows = []
    for ds in ("20261014", "20261015"):
        for ay in ("2025-2026", "2026-2027"):
            ay_bonus = 12 if ay == "2026-2027" else 0
            for i in range(1, 37):
                cds = f"CDS{i:05d}"
                enr = 380 + i * 35 + ay_bonus
                frpm_elig = int(enr * (0.32 + (i % 6) * 0.09))
                free_m = None if i % 8 == 0 else int(frpm_elig * 0.78)
                rate_str = "" if i % 9 == 0 else (None if i % 11 == 0 else f"{round(100.0 * frpm_elig / enr, 1)}%")
                grant_c = (950 + (i % 5) * 125) * 100
                frpm_rows.append((cds, ay, enr, free_m, frpm_elig, rate_str, grant_c, ds))

                # Include zero-test-taker cohorts on every 9th school to test zero-division safety!
                if i % 9 == 0:
                    sat_rows.append((cds, ay, 0, None, None, None, 0, ds))
                else:
                    takers = 25 + (i * 7) % 140 + (5 if ay == "2026-2027" else 0)
                    m_sc = 470 + (i * 9) % 150
                    r_sc = 465 + (i * 7) % 140
                    w_sc = 460 + (i * 11) % 135
                    ge1500 = int(takers * (0.18 + (i % 5) * 0.08))
                    sat_rows.append((cds, ay, takers, m_sc, r_sc, w_sc, ge1500, ds))

    bulk_insert_rows(con, "edu_frpm_snap_di", frpm_rows)
    bulk_insert_rows(con, "edu_sat_report_di", sat_rows)


def get_schools_tasks() -> list[BenchmarkTask]:
    """Return the 8 cross-schema California Public Schools (`california_schools`) tasks."""
    return [
        make_external_task(
            task_id="ext_edu_01",
            domain="california_schools",
            category="partition_pruning",
            split="train",
            question=(
                "查询 edu_frpm_snap_di 最新分区中 '2026-2027' 学年正式认证公立学校（排除 CDE 占位测试记录）的各县（county_name）"
                "学校数量（school_cnt）、K-12 在校总人数（total_enrollment）及生均补助折合总经费（total_grant_usd，"
                "按 enrollment_k12 * per_pupil_grant_cent 换算为美元保留2位小数），按 county_name 升序排列。"
            ),
            hint="per_pupil_grant_cent 单位为美分（/100.0）；输出列 county_name, school_cnt, total_enrollment, total_grant_usd。",
            original_sqlite_sql="""
SELECT s.county_name,
       COUNT(*) AS school_cnt,
       SUM(f.enrollment_k12) AS total_enrollment,
       ROUND(SUM(f.enrollment_k12 * f.per_pupil_grant_cent) / 100.0, 2) AS total_grant_usd
FROM edu_frpm_snap_di f
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE s.is_test_acct = 0
  AND f.academic_year = '2026-2027'
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            gold_sql="""
SELECT s.county_name,
       COUNT(*) AS school_cnt,
       SUM(f.enrollment_k12) AS total_enrollment,
       ROUND(SUM(f.enrollment_k12 * f.per_pupil_grant_cent) / 100.0, 2) AS total_grant_usd
FROM edu_frpm_snap_di f
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE f.ds = MAX_PT('edu_frpm_snap_di')
  AND s.is_test_acct = 0
  AND f.academic_year = '2026-2027'
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (MAX_PT)"],
            injected_partition_predicates=["f.ds = MAX_PT('edu_frpm_snap_di')"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="per_pupil_cent_scaling_and_academic_year_filter",
        ),
        make_external_task(
            task_id="ext_edu_02",
            domain="california_schools",
            category="ratio_null_handling",
            split="train",
            question=(
                "关联最新分区的 edu_sat_report_di（'2026-2027' 学年）与正式认证公立学校（edu_school_dim），按县（county_name）统计"
                "参考总人数（total_takers）、总分 >= 1500 的高分总人数（excel_cnt）及加权高分率（excel_rate，保留4位小数，须安全处理 0 参考人数），"
                "按 county_name 升序排列。"
            ),
            hint="使用 SUM(num_ge_1500) / NULLIF(SUM(test_takers), 0) 计算加权高分率；输出列 county_name, total_takers, excel_cnt, excel_rate。",
            original_sqlite_sql="""
SELECT s.county_name,
       SUM(r.test_takers) AS total_takers,
       SUM(r.num_ge_1500) AS excel_cnt,
       ROUND(1.0 * SUM(r.num_ge_1500) / NULLIF(SUM(r.test_takers), 0), 4) AS excel_rate
FROM edu_sat_report_di r
JOIN edu_school_dim s ON r.cds_code = s.cds_code
WHERE s.is_test_acct = 0
  AND r.academic_year = '2026-2027'
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            gold_sql="""
SELECT s.county_name,
       SUM(r.test_takers) AS total_takers,
       SUM(r.num_ge_1500) AS excel_cnt,
       ROUND(1.0 * SUM(r.num_ge_1500) / NULLIF(SUM(r.test_takers), 0), 4) AS excel_rate
FROM edu_sat_report_di r
JOIN edu_school_dim s ON r.cds_code = s.cds_code
WHERE r.ds = MAX_PT('edu_sat_report_di')
  AND s.is_test_acct = 0
  AND r.academic_year = '2026-2027'
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (MAX_PT)"],
            injected_partition_predicates=["r.ds = MAX_PT('edu_sat_report_di')"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="zero_test_takers_safe_weighted_ratio",
        ),
        make_external_task(
            task_id="ext_edu_03",
            domain="california_schools",
            category="fanout_dedup",
            split="val",
            question=(
                "在 '20261015' 快照下，对比 '2025-2026' 与 '2026-2027' 两个学年正式认证特许学校（charter_flag = 1）的资助与参考规模："
                "关联 edu_frpm_snap_di、edu_sat_report_di 与 edu_school_dim，按学年（academic_year）统计特许学校数（charter_school_cnt）、"
                "FRPM 资助学生总数（total_frpm_eligible）及 SAT 参考总人数（total_sat_takers），按 academic_year 升序排列。"
            ),
            hint="edu_frpm_snap_di 与 edu_sat_report_di 均为 (cds_code, academic_year) 复合粒度，关联时必须同时对齐 cds_code 与 academic_year，否则产生 2x2 跨学年扇出膨胀；输出列 academic_year, charter_school_cnt, total_frpm_eligible, total_sat_takers。",
            original_sqlite_sql="""
SELECT f.academic_year,
       COUNT(DISTINCT f.cds_code) AS charter_school_cnt,
       SUM(f.frpm_eligible_cnt) AS total_frpm_eligible,
       SUM(r.test_takers) AS total_sat_takers
FROM edu_frpm_snap_di f
JOIN edu_sat_report_di r ON f.cds_code = r.cds_code AND f.academic_year = r.academic_year
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE s.is_test_acct = 0
  AND s.charter_flag = 1
GROUP BY f.academic_year
ORDER BY f.academic_year ASC
""",
            gold_sql="""
SELECT f.academic_year,
       COUNT(DISTINCT f.cds_code) AS charter_school_cnt,
       SUM(f.frpm_eligible_cnt) AS total_frpm_eligible,
       SUM(r.test_takers) AS total_sat_takers
FROM edu_frpm_snap_di f
JOIN edu_sat_report_di r
  ON f.cds_code = r.cds_code
 AND f.academic_year = r.academic_year
 AND f.ds = r.ds
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE f.ds = '20261015'
  AND s.is_test_acct = 0
  AND s.charter_flag = 1
GROUP BY f.academic_year
ORDER BY f.academic_year ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (join-equivalence ds propagation)"],
            injected_partition_predicates=["f.ds = '20261015'", "f.ds = r.ds"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="composite_cds_academic_year_join_anti_fanout",
        ),
        make_external_task(
            task_id="ext_edu_04",
            domain="california_schools",
            category="semantic_linking_trap",
            split="val",
            question=(
                "针对 '20261015' 分区 '2026-2027' 学年的正式认证特许学校（charter_flag = 1），将 edu_school_dim.funding_type_code"
                " 中的 NULL 与空串统一归为 'UNCLASSIFIED'，按拨款类型编码分组统计学校数（school_cnt）及平均生均补助美元金额"
                "（avg_grant_per_pupil_usd，保留2位小数），按 funding_group 升序排列。"
            ),
            hint="探查 funding_type_code 编码与 per_pupil_grant_cent 单位；输出列 funding_group, school_cnt, avg_grant_per_pupil_usd。",
            original_sqlite_sql="""
SELECT COALESCE(NULLIF(TRIM(s.funding_type_code), ''), 'UNCLASSIFIED') AS funding_group,
       COUNT(*) AS school_cnt,
       ROUND(AVG(f.per_pupil_grant_cent / 100.0), 2) AS avg_grant_per_pupil_usd
FROM edu_frpm_snap_di f
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE s.is_test_acct = 0
  AND s.charter_flag = 1
  AND f.academic_year = '2026-2027'
GROUP BY COALESCE(NULLIF(TRIM(s.funding_type_code), ''), 'UNCLASSIFIED')
ORDER BY funding_group ASC
""",
            gold_sql="""
SELECT COALESCE(NULLIF(TRIM(s.funding_type_code), ''), 'UNCLASSIFIED') AS funding_group,
       COUNT(*) AS school_cnt,
       ROUND(AVG(f.per_pupil_grant_cent / 100.0), 2) AS avg_grant_per_pupil_usd
FROM edu_frpm_snap_di f
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE f.ds = '20261015'
  AND s.is_test_acct = 0
  AND s.charter_flag = 1
  AND f.academic_year = '2026-2027'
GROUP BY COALESCE(NULLIF(TRIM(s.funding_type_code), ''), 'UNCLASSIFIED')
ORDER BY funding_group ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["f.ds = '20261015'"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="opaque_funding_code_null_coalesce_and_cent_scaling",
        ),
        make_external_task(
            task_id="ext_edu_05",
            domain="california_schools",
            category="silent_caliber_audit",
            split="test",
            question=(
                "针对 'Alameda' 与 'Fresno' 两县的全部正式认证公立学校（edu_school_dim），统计每县的学校总数（total_schools）"
                "以及在 '20261015' 分区 '2026-2027' 学年 edu_sat_report_di 中参考人数达到 50 人及以上的达标学校数"
                "（qualifying_sat_schools），按 county_name 升序排列。"
            ),
            hint="右表 edu_sat_report_di 的分区、学年与 test_takers >= 50 条件须放在 LEFT JOIN ON 子句，确保未达标与 0 参考学校计入 total_schools；输出列 county_name, total_schools, qualifying_sat_schools。",
            original_sqlite_sql="""
SELECT s.county_name,
       COUNT(s.cds_code) AS total_schools,
       COUNT(r.cds_code) AS qualifying_sat_schools
FROM edu_school_dim s
LEFT JOIN edu_sat_report_di r
  ON s.cds_code = r.cds_code
 AND r.academic_year = '2026-2027'
 AND r.test_takers >= 50
WHERE s.is_test_acct = 0
  AND s.county_name IN ('Alameda', 'Fresno')
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            gold_sql="""
SELECT s.county_name,
       COUNT(s.cds_code) AS total_schools,
       COUNT(r.cds_code) AS qualifying_sat_schools
FROM edu_school_dim s
LEFT JOIN edu_sat_report_di r
  ON s.cds_code = r.cds_code
 AND r.ds = '20261015'
 AND r.academic_year = '2026-2027'
 AND r.test_takers >= 50
WHERE s.is_test_acct = 0
  AND s.county_name IN ('Alameda', 'Fresno')
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (LEFT JOIN ON r.ds)"],
            injected_partition_predicates=["ON r.ds = '20261015'"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="left_join_on_clause_threshold_preservation",
        ),
        make_external_task(
            task_id="ext_edu_06",
            domain="california_schools",
            category="ratio_null_handling",
            split="test",
            question=(
                "在 '20261015' 分区 '2026-2027' 学年的 edu_frpm_snap_di 中关联正式认证公立学校（edu_school_dim），按县（county_name）"
                "计算按在校生人数加权的整体 FRPM 资助比例（weighted_frpm_rate，SUM(frpm_eligible_cnt) / SUM(enrollment_k12)，保留4位小数）"
                "以及非空 elig_rate_str 上报学校数（valid_rate_str_cnt），按 county_name 升序排列。"
            ),
            hint="勿对字符串百分比直接取算术平均代替人数加权比例；输出列 county_name, weighted_frpm_rate, valid_rate_str_cnt。",
            original_sqlite_sql="""
SELECT s.county_name,
       ROUND(1.0 * SUM(f.frpm_eligible_cnt) / NULLIF(SUM(f.enrollment_k12), 0), 4) AS weighted_frpm_rate,
       SUM(CASE WHEN f.elig_rate_str IS NOT NULL AND TRIM(f.elig_rate_str) <> '' THEN 1 ELSE 0 END) AS valid_rate_str_cnt
FROM edu_frpm_snap_di f
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE s.is_test_acct = 0
  AND f.academic_year = '2026-2027'
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            gold_sql="""
SELECT s.county_name,
       ROUND(1.0 * SUM(f.frpm_eligible_cnt) / NULLIF(SUM(f.enrollment_k12), 0), 4) AS weighted_frpm_rate,
       SUM(CASE WHEN f.elig_rate_str IS NOT NULL AND TRIM(f.elig_rate_str) <> '' THEN 1 ELSE 0 END) AS valid_rate_str_cnt
FROM edu_frpm_snap_di f
JOIN edu_school_dim s ON f.cds_code = s.cds_code
WHERE f.ds = '20261015'
  AND s.is_test_acct = 0
  AND f.academic_year = '2026-2027'
GROUP BY s.county_name
ORDER BY s.county_name ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["f.ds = '20261015'"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="enrollment_weighted_ratio_vs_unweighted_rate_string",
        ),
        make_external_task(
            task_id="ext_edu_07",
            domain="california_schools",
            category="retention_date_math",
            split="test",
            question=(
                "针对 '20261015' 分区 '2026-2027' 学年有实际 SAT 参考人数（test_takers > 0）的正式认证公立学校，"
                "以 '2026-10-15' 为基准按建校年份差（DATEDIFF 年差）将学校划分为 'established_ge_20y'（>= 20 年）与 'modern_lt_20y'（< 20 年），"
                "统计两组学校数量（school_cnt）及三科平均总分均值（avg_total_sat，avg_math + avg_reading + avg_writing 的均值，保留1位小数），"
                "按 age_cohort 升序排列。"
            ),
            hint="ODPS 使用 DATEDIFF(TO_DATE('2026-10-15', 'yyyy-mm-dd'), TO_DATE(open_dt, 'yyyy-mm-dd'), 'yyyy')；输出列 age_cohort, school_cnt, avg_total_sat。",
            original_sqlite_sql="""
SELECT CASE
         WHEN CAST(STRFTIME('%Y', '2026-10-15') AS INTEGER) - CAST(STRFTIME('%Y', s.open_dt) AS INTEGER) >= 20
         THEN 'established_ge_20y'
         ELSE 'modern_lt_20y'
       END AS age_cohort,
       COUNT(*) AS school_cnt,
       ROUND(AVG(r.avg_math + r.avg_reading + r.avg_writing), 1) AS avg_total_sat
FROM edu_sat_report_di r
JOIN edu_school_dim s ON r.cds_code = s.cds_code
WHERE s.is_test_acct = 0
  AND r.academic_year = '2026-2027'
  AND r.test_takers > 0
GROUP BY 1
ORDER BY age_cohort ASC
""",
            gold_sql="""
WITH aged AS (
  SELECT CASE
           WHEN DATEDIFF(TO_DATE('2026-10-15', 'yyyy-mm-dd'), TO_DATE(s.open_dt, 'yyyy-mm-dd'), 'yyyy') >= 20
           THEN 'established_ge_20y'
           ELSE 'modern_lt_20y'
         END AS age_cohort,
         (r.avg_math + r.avg_reading + r.avg_writing) AS total_sat
  FROM edu_sat_report_di r
  JOIN edu_school_dim s ON r.cds_code = s.cds_code
  WHERE r.ds = '20261015'
    AND s.is_test_acct = 0
    AND r.academic_year = '2026-2027'
    AND r.test_takers > 0
)
SELECT age_cohort,
       COUNT(*) AS school_cnt,
       ROUND(AVG(CAST(total_sat AS DOUBLE)), 1) AS avg_total_sat
FROM aged
GROUP BY age_cohort
ORDER BY age_cohort ASC
""",
            ast_transformations=[
                "STRFTIME('%Y', ...) -> DATEDIFF(..., 'yyyy')",
                "FACT_PARTITION_PREDICATE_INJECTION",
            ],
            injected_partition_predicates=["r.ds = '20261015'"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="strftime_year_to_odps_datediff_yyyy",
        ),
        make_external_task(
            task_id="ext_edu_08",
            domain="california_schools",
            category="silent_caliber_audit",
            split="test",
            question=(
                "在 '20261015' 分区 '2026-2027' 学年的 edu_frpm_snap_di 中，筛选 'Los Angeles' 县的正式认证公立学校，"
                "按 frpm_eligible_cnt 降序、cds_code 升序排列，使用显式逐行窗口帧计算逐校累计资助人数（cum_frpm_eligible），"
                "返回排名前 5 的学校 cds_code, frpm_eligible_cnt, cum_frpm_eligible。"
            ),
            hint="窗口函数须显式使用 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW 防止同分并列帧膨胀；输出列 cds_code, frpm_eligible_cnt, cum_frpm_eligible。",
            original_sqlite_sql="""
WITH ranked AS (
  SELECT f.cds_code,
         f.frpm_eligible_cnt,
         SUM(f.frpm_eligible_cnt) OVER (
           ORDER BY f.frpm_eligible_cnt DESC, f.cds_code ASC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS cum_frpm_eligible,
         ROW_NUMBER() OVER (ORDER BY f.frpm_eligible_cnt DESC, f.cds_code ASC) AS rn
  FROM edu_frpm_snap_di f
  JOIN edu_school_dim s ON f.cds_code = s.cds_code
  WHERE s.is_test_acct = 0
    AND s.county_name = 'Los Angeles'
    AND f.academic_year = '2026-2027'
)
SELECT cds_code, frpm_eligible_cnt, cum_frpm_eligible
FROM ranked
WHERE rn <= 5
ORDER BY frpm_eligible_cnt DESC, cds_code ASC
""",
            gold_sql="""
WITH ranked AS (
  SELECT f.cds_code,
         f.frpm_eligible_cnt,
         SUM(f.frpm_eligible_cnt) OVER (
           ORDER BY f.frpm_eligible_cnt DESC, f.cds_code ASC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS cum_frpm_eligible,
         ROW_NUMBER() OVER (ORDER BY f.frpm_eligible_cnt DESC, f.cds_code ASC) AS rn
  FROM edu_frpm_snap_di f
  JOIN edu_school_dim s ON f.cds_code = s.cds_code
  WHERE f.ds = '20261015'
    AND s.is_test_acct = 0
    AND s.county_name = 'Los Angeles'
    AND f.academic_year = '2026-2027'
)
SELECT cds_code, frpm_eligible_cnt, cum_frpm_eligible
FROM ranked
WHERE rn <= 5
ORDER BY frpm_eligible_cnt DESC, cds_code ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["f.ds = '20261015'"],
            unpartitioned_dim_tables=["edu_school_dim"],
            caliber_trap="explicit_rows_between_cumulative_window",
        ),
    ]
