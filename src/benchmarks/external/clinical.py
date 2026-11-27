"""Domain 2 (`thrombosis_clinical`): Clinical Hospital Admissions schema, populator, and 8 benchmark tasks."""

from __future__ import annotations

import datetime as dt
from typing import Any

from src.benchmarks.external.adaptation import make_external_task
from src.benchmarks.types import BenchmarkTask
from src.env.local_executor import bulk_insert_rows
from src.env.schema import ColumnMeta, TableMeta, WarehouseCatalog


def register_clinical_tables(catalog: WarehouseCatalog) -> None:
    """Register the 4 Thrombosis Clinical (`thrombosis_clinical`) tables into `catalog`."""
    catalog.register(
        TableMeta(
            name="clin_patient",
            layer="dim",
            comment="Clinical patient demographic & referral dimension (BIRD thrombosis domain)",
            columns=(
                ColumnMeta("patient_id", "VARCHAR", "Patient primary identifier"),
                ColumnMeta("sex", "VARCHAR", "Biological sex ('F', 'M', '', NULL)"),
                ColumnMeta("birth_dt", "VARCHAR", "Birth date YYYY-MM-DD"),
                ColumnMeta("first_visit_dt", "VARCHAR", "Initial hospital visit date YYYY-MM-DD"),
                ColumnMeta("referral_channel_code", "VARCHAR", "Opaque admission referral code ('REF_OUTPATIENT', 'REF_ER', 'REF_TRANSFER')"),
            ),
            partition_col=None,
        )
    )
    catalog.register(
        TableMeta(
            name="clin_diagnosis_scd2",
            layer="dim",
            comment="SCD Type-2 historical clinical diagnosis & thrombosis severity dimension",
            columns=(
                ColumnMeta("patient_id", "VARCHAR", "Patient identifier"),
                ColumnMeta("diag_line_no", "INTEGER", "Diagnosis rank line (1 = primary diagnosis)"),
                ColumnMeta("disease_code", "VARCHAR", "Autoimmune disease code ('SLE', 'APS', 'RA', 'SSc')"),
                ColumnMeta("thrombosis_grade", "INTEGER", "Thrombosis severity grade (0=none, 1=mild, 2=severe)"),
                ColumnMeta("eff_start_dt", "VARCHAR", "Version effective start date YYYY-MM-DD"),
                ColumnMeta("eff_end_dt", "VARCHAR", "Version effective end date YYYY-MM-DD ('9999-12-31' for open-ended)"),
                ColumnMeta("is_current", "INTEGER", "1 = latest active version, 0 = historical superseded version"),
            ),
            partition_col=None,
        )
    )
    catalog.register(
        TableMeta(
            name="clin_encounter_di",
            layer="dwd",
            comment="Daily partitioned inpatient clinical encounter fact (charges in cents)",
            columns=(
                ColumnMeta("encounter_id", "VARCHAR", "Inpatient encounter ID"),
                ColumnMeta("patient_id", "VARCHAR", "Patient ID"),
                ColumnMeta("admit_dt", "VARCHAR", "Admission date YYYY-MM-DD"),
                ColumnMeta("discharge_dt", "VARCHAR", "Discharge date YYYY-MM-DD"),
                ColumnMeta("dept_code", "VARCHAR", "Clinical department code ('RHEUM', 'HEMAT', 'IMMUNO')"),
                ColumnMeta("total_charge_cent", "BIGINT", "Total inpatient charge in integer cents (/100.0 for yuan)"),
                ColumnMeta("is_test_acct", "INTEGER", "1 = HIS QA synthetic drill record, 0 = formal clinical encounter"),
                ColumnMeta("ds", "VARCHAR", "Snapshot partition date YYYYMMDD", is_partition=True),
            ),
            partition_col="ds",
        )
    )
    catalog.register(
        TableMeta(
            name="clin_lab_exam_di",
            layer="ods",
            comment="Partitioned laboratory examination log (heterogeneous partition key `dt`!)",
            columns=(
                ColumnMeta("exam_id", "VARCHAR", "Lab examination record ID"),
                ColumnMeta("patient_id", "VARCHAR", "Patient ID"),
                ColumnMeta("encounter_id", "VARCHAR", "Associated inpatient encounter ID"),
                ColumnMeta("exam_dt", "VARCHAR", "Lab specimen date YYYY-MM-DD"),
                ColumnMeta("plt_count", "INTEGER", "Platelet count (10^9/L, nullable)"),
                ColumnMeta("ana_pattern_code", "VARCHAR", "ANA immunofluorescence pattern ('PAT_SPECKLED', 'PAT_HOMOG', 'PAT_NEG', '', NULL)"),
                ColumnMeta("acl_pos_ratio_str", "VARCHAR", "String-encoded anticardiolipin positivity ratio (e.g. '0.72', '35.0%', '', NULL)"),
                ColumnMeta("dt", "VARCHAR", "Lab ingestion partition key YYYYMMDD (note: `dt`, not `ds`)", is_partition=True),
            ),
            partition_col="dt",
        )
    )


def populate_clinical_tables(con: Any) -> None:
    """Create and deterministically populate the 4 `thrombosis_clinical` tables in DuckDB."""
    con.execute(
        """
        CREATE TABLE clin_patient (
            patient_id VARCHAR,
            sex VARCHAR,
            birth_dt VARCHAR,
            first_visit_dt VARCHAR,
            referral_channel_code VARCHAR
        );
        CREATE TABLE clin_diagnosis_scd2 (
            patient_id VARCHAR,
            diag_line_no INTEGER,
            disease_code VARCHAR,
            thrombosis_grade INTEGER,
            eff_start_dt VARCHAR,
            eff_end_dt VARCHAR,
            is_current INTEGER
        );
        CREATE TABLE clin_encounter_di (
            encounter_id VARCHAR,
            patient_id VARCHAR,
            admit_dt VARCHAR,
            discharge_dt VARCHAR,
            dept_code VARCHAR,
            total_charge_cent BIGINT,
            is_test_acct INTEGER,
            ds VARCHAR
        );
        CREATE TABLE clin_lab_exam_di (
            exam_id VARCHAR,
            patient_id VARCHAR,
            encounter_id VARCHAR,
            exam_dt VARCHAR,
            plt_count INTEGER,
            ana_pattern_code VARCHAR,
            acl_pos_ratio_str VARCHAR,
            dt VARCHAR
        );
        """
    )

    sex_choices = ["F", "F", "M", "", None]
    ref_codes = ["REF_OUTPATIENT", "REF_ER", "REF_TRANSFER"]
    pat_rows = []
    for i in range(1, 51):
        pid = f"PAT{i:04d}"
        b_dt = dt.date(1965, 1, 1) + dt.timedelta(days=(i * 173) % 12_000)
        fv_dt = dt.date(2025, 1, 10) + dt.timedelta(days=(i * 9) % 240)
        pat_rows.append(
            (pid, sex_choices[i % len(sex_choices)], b_dt.strftime("%Y-%m-%d"), fv_dt.strftime("%Y-%m-%d"), ref_codes[i % 3])
        )
    bulk_insert_rows(con, "clin_patient", pat_rows)

    diseases = ["SLE", "APS", "RA", "SSc"]
    scd_rows = []
    for i in range(1, 51):
        pid = f"PAT{i:04d}"
        base_dis = diseases[i % 4]
        if i <= 28:
            # Two SCD2 versions: historical grade 0/1 before 2026-10-08, upgraded to grade 2 after 2026-10-08!
            g_old = i % 2
            g_new = 2
            scd_rows.append((pid, 1, base_dis, g_old, "2025-01-01", "2026-10-07", 0))
            scd_rows.append((pid, 1, base_dis, g_new, "2026-10-08", "9999-12-31", 1))
        else:
            scd_rows.append((pid, 1, base_dis, i % 3, "2025-01-01", "9999-12-31", 1))
    bulk_insert_rows(con, "clin_diagnosis_scd2", scd_rows)

    depts = ["RHEUM", "HEMAT", "IMMUNO"]
    enc_rows = []
    lab_rows = []
    exam_seq = 1
    ana_choices = ["PAT_SPECKLED", "PAT_HOMOG", "PAT_NEG", "", None]
    acl_choices = ["0.24", "0.86", "42.0%", "18.5%", "", None]
    for ds in ("20261013", "20261014", "20261015"):
        for i in range(1, 45):
            pid = f"PAT{i:04d}"
            eid = f"ENC_{ds}_{i:03d}"
            # Half admitted before 2026-10-08 (so historical SCD2 differs from is_current=1!)
            admit_d = dt.date(2026, 10, 2 + (i % 10))
            los = 3 + (i % 9)
            disch_d = admit_d + dt.timedelta(days=los)
            dept = depts[i % 3]
            charge_c = (4_200 + i * 315) * 100
            is_qa = 1 if i % 14 == 0 else 0
            enc_rows.append(
                (eid, pid, admit_d.strftime("%Y-%m-%d"), disch_d.strftime("%Y-%m-%d"), dept, charge_c, is_qa, ds)
            )
            # Multiple lab exams per encounter (except patients 42..50 who have 0 speckled exams for LEFT JOIN test!)
            n_labs = 3 if i <= 20 else 2
            for l_idx in range(n_labs):
                ex_d = admit_d + dt.timedelta(days=min(los, l_idx))
                plt = None if (i + l_idx) % 7 == 0 else (65 + ((i * 19 + l_idx * 37) % 210))
                ana = "PAT_NEG" if i >= 38 else ana_choices[(i + l_idx) % len(ana_choices)]
                acl = acl_choices[(i + l_idx) % len(acl_choices)]
                lab_rows.append(
                    (f"EXM_{ds}_{exam_seq:05d}", pid, eid, ex_d.strftime("%Y-%m-%d"), plt, ana, acl, ds)
                )
                exam_seq += 1

    bulk_insert_rows(con, "clin_encounter_di", enc_rows)
    bulk_insert_rows(con, "clin_lab_exam_di", lab_rows)


def get_clinical_tasks() -> list[BenchmarkTask]:
    """Return the 8 cross-schema Thrombosis Clinical (`thrombosis_clinical`) tasks."""
    return [
        make_external_task(
            task_id="ext_clin_01",
            domain="thrombosis_clinical",
            category="partition_pruning",
            split="train",
            question=(
                "关联最新分区的临床住院记录表（clin_encounter_di，排除 HIS 压测记录）与最新分区的实验室化验表（clin_lab_exam_di），"
                "按临床科室（dept_code）统计去重住院患者数（patient_cnt）、化验单总次数（exam_cnt）及血小板计数均值（avg_plt，保留2位小数），"
                "按 dept_code 升序排列。"
            ),
            hint="注意两张分区表的分区键名称不同（ds vs dt），关联键为 encounter_id 与 patient_id；输出列 dept_code, patient_cnt, exam_cnt, avg_plt。",
            original_sqlite_sql="""
SELECT e.dept_code,
       COUNT(DISTINCT e.patient_id) AS patient_cnt,
       COUNT(x.exam_id) AS exam_cnt,
       ROUND(AVG(x.plt_count), 2) AS avg_plt
FROM clin_encounter_di e
JOIN clin_lab_exam_di x ON e.encounter_id = x.encounter_id AND e.patient_id = x.patient_id
WHERE e.is_test_acct = 0
GROUP BY e.dept_code
ORDER BY e.dept_code ASC
""",
            gold_sql="""
SELECT e.dept_code,
       COUNT(DISTINCT e.patient_id) AS patient_cnt,
       COUNT(x.exam_id) AS exam_cnt,
       ROUND(AVG(CAST(x.plt_count AS DOUBLE)), 2) AS avg_plt
FROM clin_encounter_di e
JOIN clin_lab_exam_di x ON e.encounter_id = x.encounter_id AND e.patient_id = x.patient_id
WHERE e.ds = MAX_PT('clin_encounter_di')
  AND x.dt = MAX_PT('clin_lab_exam_di')
  AND e.is_test_acct = 0
GROUP BY e.dept_code
ORDER BY e.dept_code ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (heterogeneous ds + dt MAX_PT)"],
            injected_partition_predicates=[
                "e.ds = MAX_PT('clin_encounter_di')",
                "x.dt = MAX_PT('clin_lab_exam_di')",
            ],
            unpartitioned_dim_tables=[],
            caliber_trap="heterogeneous_ds_and_dt_partition_pruning",
        ),
        make_external_task(
            task_id="ext_clin_02",
            domain="thrombosis_clinical",
            category="scd2_temporal_join",
            split="train",
            question=(
                "针对 '20261015' 分区的正式住院记录（clin_encounter_di），按患者入院当日（admit_dt）在 clin_diagnosis_scd2 中生效的主诊断"
                "（diag_line_no = 1）历史拉链版本，分组统计各疾病编码（disease_code）的住院人次（encounter_cnt）与住院总费用（total_charge_yuan，"
                "由分换算为元保留2位小数），按 disease_code 升序排列。"
            ),
            hint="必须按 admit_dt 落在 [eff_start_dt, eff_end_dt] 闭区间关联历史主诊断，严禁误用 is_current = 1；输出列 disease_code, encounter_cnt, total_charge_yuan。",
            original_sqlite_sql="""
SELECT d.disease_code,
       COUNT(*) AS encounter_cnt,
       ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS total_charge_yuan
FROM clin_encounter_di e
JOIN clin_diagnosis_scd2 d
  ON e.patient_id = d.patient_id
 AND d.diag_line_no = 1
 AND e.admit_dt BETWEEN d.eff_start_dt AND IFNULL(NULLIF(d.eff_end_dt, ''), '9999-12-31')
WHERE e.is_test_acct = 0
GROUP BY d.disease_code
ORDER BY d.disease_code ASC
""",
            gold_sql="""
SELECT d.disease_code,
       COUNT(*) AS encounter_cnt,
       ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS total_charge_yuan
FROM clin_encounter_di e
JOIN clin_diagnosis_scd2 d
  ON e.patient_id = d.patient_id
 AND d.diag_line_no = 1
 AND e.admit_dt BETWEEN d.eff_start_dt AND COALESCE(NULLIF(d.eff_end_dt, ''), '9999-12-31')
WHERE e.ds = '20261015'
  AND e.is_test_acct = 0
GROUP BY d.disease_code
ORDER BY d.disease_code ASC
""",
            ast_transformations=["IFNULL -> COALESCE", "FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["e.ds = '20261015'"],
            unpartitioned_dim_tables=["clin_diagnosis_scd2"],
            caliber_trap="scd2_as_of_admit_dt_slice_and_cent_scaling",
        ),
        make_external_task(
            task_id="ext_clin_03",
            domain="thrombosis_clinical",
            category="semantic_linking_trap",
            split="val",
            question=(
                "在 '20261015' 分区的正式临床住院记录中，关联 clin_patient 筛选通过“急诊转入（emergency）”渠道入院的患者，"
                "将患者性别（sex）中的 NULL 与空串归并为 'UNKNOWN'，统计各性别分组（sex_group）的住院人次（encounter_cnt）"
                "及次均住院费用（avg_charge_yuan，换算为元保留2位小数），按 sex_group 升序排列。"
            ),
            hint="先探查 clin_patient.referral_channel_code 的急诊枚举编码；输出列 sex_group, encounter_cnt, avg_charge_yuan。",
            original_sqlite_sql="""
SELECT COALESCE(NULLIF(TRIM(p.sex), ''), 'UNKNOWN') AS sex_group,
       COUNT(*) AS encounter_cnt,
       ROUND(AVG(e.total_charge_cent / 100.0), 2) AS avg_charge_yuan
FROM clin_encounter_di e
JOIN clin_patient p ON e.patient_id = p.patient_id
WHERE e.is_test_acct = 0
  AND p.referral_channel_code = 'REF_ER'
GROUP BY COALESCE(NULLIF(TRIM(p.sex), ''), 'UNKNOWN')
ORDER BY sex_group ASC
""",
            gold_sql="""
SELECT COALESCE(NULLIF(TRIM(p.sex), ''), 'UNKNOWN') AS sex_group,
       COUNT(*) AS encounter_cnt,
       ROUND(AVG(e.total_charge_cent / 100.0), 2) AS avg_charge_yuan
FROM clin_encounter_di e
JOIN clin_patient p ON e.patient_id = p.patient_id
WHERE e.ds = '20261015'
  AND e.is_test_acct = 0
  AND p.referral_channel_code = 'REF_ER'
GROUP BY COALESCE(NULLIF(TRIM(p.sex), ''), 'UNKNOWN')
ORDER BY sex_group ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["e.ds = '20261015'"],
            unpartitioned_dim_tables=["clin_patient"],
            caliber_trap="opaque_referral_code_dirty_sex_and_cent_scaling",
        ),
        make_external_task(
            task_id="ext_clin_04",
            domain="thrombosis_clinical",
            category="fanout_dedup",
            split="val",
            question=(
                "针对 '20261015' 分区的正式住院记录（clin_encounter_di），结合同日（dt = '20261015'）clin_lab_exam_di 的化验明细，"
                "按科室（dept_code）统计住院总人次（encounter_cnt）、科室住院总费用（total_charge_yuan，换算为元保留2位小数）"
                "以及血小板计数低于 100 的异常化验单总数（low_plt_exam_cnt），按 dept_code 升序排列。"
            ),
            hint="一次住院对应多张化验单，必须在关联前按住院粒度预聚合化验表，防止住院总费用与人次膨胀；输出列 dept_code, encounter_cnt, total_charge_yuan, low_plt_exam_cnt。",
            original_sqlite_sql="""
WITH lab_agg AS (
  SELECT encounter_id, patient_id,
         SUM(CASE WHEN plt_count IS NOT NULL AND plt_count < 100 THEN 1 ELSE 0 END) AS low_plt_cnt
  FROM clin_lab_exam_di
  GROUP BY encounter_id, patient_id
)
SELECT e.dept_code,
       COUNT(*) AS encounter_cnt,
       ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS total_charge_yuan,
       SUM(IFNULL(l.low_plt_cnt, 0)) AS low_plt_exam_cnt
FROM clin_encounter_di e
LEFT JOIN lab_agg l ON e.encounter_id = l.encounter_id AND e.patient_id = l.patient_id
WHERE e.is_test_acct = 0
GROUP BY e.dept_code
ORDER BY e.dept_code ASC
""",
            gold_sql="""
WITH lab_agg AS (
  SELECT encounter_id, patient_id,
         SUM(CASE WHEN plt_count IS NOT NULL AND plt_count < 100 THEN 1 ELSE 0 END) AS low_plt_cnt
  FROM clin_lab_exam_di
  WHERE dt = '20261015'
  GROUP BY encounter_id, patient_id
)
SELECT e.dept_code,
       COUNT(*) AS encounter_cnt,
       ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS total_charge_yuan,
       SUM(NVL(l.low_plt_cnt, 0)) AS low_plt_exam_cnt
FROM clin_encounter_di e
LEFT JOIN lab_agg l ON e.encounter_id = l.encounter_id AND e.patient_id = l.patient_id
WHERE e.ds = '20261015'
  AND e.is_test_acct = 0
GROUP BY e.dept_code
ORDER BY e.dept_code ASC
""",
            ast_transformations=["IFNULL -> NVL", "FACT_PARTITION_PREDICATE_INJECTION (ds + dt)"],
            injected_partition_predicates=["clin_lab_exam_di.dt = '20261015'", "e.ds = '20261015'"],
            unpartitioned_dim_tables=[],
            caliber_trap="encounter_lab_fanout_and_heterogeneous_dt_key",
        ),
        make_external_task(
            task_id="ext_clin_05",
            domain="thrombosis_clinical",
            category="scd2_temporal_join",
            split="test",
            question=(
                "在 '20261015' 分区的正式住院记录（clin_encounter_di）中，找出入院当日（admit_dt）对应的历史主诊断（diag_line_no = 1）"
                "血栓分级（thrombosis_grade）严格低于该患者当前最新有效主诊断（is_current = 1）血栓分级的病情恶化住院记录，"
                "按入院时主诊断疾病编码（disease_code）统计恶化住院人次（progressed_encounter_cnt）及对应住院总费用（progressed_charge_yuan，保留2位小数），"
                "按 disease_code 升序排列。"
            ),
            hint="同时关联入院时点拉链切片与当前快照（is_current = 1）进行分级对比；输出列 disease_code, progressed_encounter_cnt, progressed_charge_yuan。",
            original_sqlite_sql="""
SELECT d_hist.disease_code,
       COUNT(*) AS progressed_encounter_cnt,
       ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS progressed_charge_yuan
FROM clin_encounter_di e
JOIN clin_diagnosis_scd2 d_hist
  ON e.patient_id = d_hist.patient_id
 AND d_hist.diag_line_no = 1
 AND e.admit_dt BETWEEN d_hist.eff_start_dt AND IFNULL(NULLIF(d.eff_end_dt, ''), '9999-12-31')
JOIN clin_diagnosis_scd2 d_cur
  ON e.patient_id = d_cur.patient_id
 AND d_cur.diag_line_no = 1
 AND d_cur.is_current = 1
WHERE e.is_test_acct = 0
  AND d_hist.thrombosis_grade < d_cur.thrombosis_grade
GROUP BY d_hist.disease_code
ORDER BY d_hist.disease_code ASC
""",
            gold_sql="""
SELECT d_hist.disease_code,
       COUNT(*) AS progressed_encounter_cnt,
       ROUND(SUM(e.total_charge_cent) / 100.0, 2) AS progressed_charge_yuan
FROM clin_encounter_di e
JOIN clin_diagnosis_scd2 d_hist
  ON e.patient_id = d_hist.patient_id
 AND d_hist.diag_line_no = 1
 AND e.admit_dt BETWEEN d_hist.eff_start_dt AND COALESCE(NULLIF(d_hist.eff_end_dt, ''), '9999-12-31')
JOIN clin_diagnosis_scd2 d_cur
  ON e.patient_id = d_cur.patient_id
 AND d_cur.diag_line_no = 1
 AND d_cur.is_current = 1
WHERE e.ds = '20261015'
  AND e.is_test_acct = 0
  AND d_hist.thrombosis_grade < d_cur.thrombosis_grade
GROUP BY d_hist.disease_code
ORDER BY d_hist.disease_code ASC
""",
            ast_transformations=["IFNULL -> COALESCE", "FACT_PARTITION_PREDICATE_INJECTION"],
            injected_partition_predicates=["e.ds = '20261015'"],
            unpartitioned_dim_tables=["clin_diagnosis_scd2"],
            caliber_trap="dual_scd2_historical_vs_current_severity_comparison",
        ),
        make_external_task(
            task_id="ext_clin_06",
            domain="thrombosis_clinical",
            category="retention_date_math",
            split="test",
            question=(
                "统计 '20261015' 分区正式住院记录（clin_encounter_di）中各临床科室（dept_code）的住院人次（encounter_cnt）、"
                "平均住院天数（avg_los_days，discharge_dt 与 admit_dt 之差，保留2位小数）以及长住院占比（long_stay_ratio，"
                "住院天数 >= 7 天的比例，保留4位小数），按 dept_code 升序排列。"
            ),
            hint="ODPS 中 DATEDIFF 须使用三参数形式；输出列 dept_code, encounter_cnt, avg_los_days, long_stay_ratio。",
            original_sqlite_sql="""
SELECT dept_code,
       COUNT(*) AS encounter_cnt,
       ROUND(AVG(JULIANDAY(discharge_dt) - JULIANDAY(admit_dt)), 2) AS avg_los_days,
       ROUND(1.0 * SUM(CASE WHEN JULIANDAY(discharge_dt) - JULIANDAY(admit_dt) >= 7 THEN 1 ELSE 0 END) / COUNT(*), 4) AS long_stay_ratio
FROM clin_encounter_di
WHERE is_test_acct = 0
GROUP BY dept_code
ORDER BY dept_code ASC
""",
            gold_sql="""
WITH los_calc AS (
  SELECT dept_code,
         DATEDIFF(TO_DATE(discharge_dt, 'yyyy-mm-dd'), TO_DATE(admit_dt, 'yyyy-mm-dd'), 'dd') AS los_days
  FROM clin_encounter_di
  WHERE ds = '20261015'
    AND is_test_acct = 0
)
SELECT dept_code,
       COUNT(*) AS encounter_cnt,
       ROUND(AVG(CAST(los_days AS DOUBLE)), 2) AS avg_los_days,
       ROUND(1.0 * SUM(CASE WHEN los_days >= 7 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 4) AS long_stay_ratio
FROM los_calc
GROUP BY dept_code
ORDER BY dept_code ASC
""",
            ast_transformations=[
                "JULIANDAY_DIFF -> DATEDIFF(end, start, 'dd')",
                "FACT_PARTITION_PREDICATE_INJECTION",
            ],
            injected_partition_predicates=["clin_encounter_di.ds = '20261015'"],
            unpartitioned_dim_tables=[],
            caliber_trap="odps_3arg_datediff_length_of_stay",
        ),
        make_external_task(
            task_id="ext_clin_07",
            domain="thrombosis_clinical",
            category="ratio_null_handling",
            split="test",
            question=(
                "分析 clin_lab_exam_di 在 '20261013' 至 '20261015' 分区（dt）的免疫学化验记录：将 ana_pattern_code 中的 NULL 与空串"
                "统一归为 'UNRECORDED'，并将混合格式的 acl_pos_ratio_str 归一化为 [0, 1] 小数（若含 '%' 则去掉百分号除以 100.0，"
                "否则直接转为浮点数，忽略空串与 NULL），统计各 ana_pattern 的化验总单数（exam_cnt）及归一化抗心磷脂阳性比均值"
                "（avg_acl_ratio，保留4位小数），按 exam_cnt 降序、ana_pattern 升序排列。"
            ),
            hint="注意分区列名为 dt；输出列 ana_pattern, exam_cnt, avg_acl_ratio。",
            original_sqlite_sql="""
SELECT COALESCE(NULLIF(TRIM(ana_pattern_code), ''), 'UNRECORDED') AS ana_pattern,
       COUNT(*) AS exam_cnt,
       ROUND(AVG(
         CASE
           WHEN acl_pos_ratio_str IS NULL OR TRIM(acl_pos_ratio_str) = '' THEN NULL
           WHEN acl_pos_ratio_str LIKE '%\\%' ESCAPE '\\' THEN CAST(REPLACE(acl_pos_ratio_str, '%', '') AS REAL) / 100.0
           ELSE CAST(acl_pos_ratio_str AS REAL)
         END
       ), 4) AS avg_acl_ratio
FROM clin_lab_exam_di
GROUP BY COALESCE(NULLIF(TRIM(ana_pattern_code), ''), 'UNRECORDED')
ORDER BY exam_cnt DESC, ana_pattern ASC
""",
            gold_sql="""
SELECT COALESCE(NULLIF(TRIM(ana_pattern_code), ''), 'UNRECORDED') AS ana_pattern,
       COUNT(*) AS exam_cnt,
       ROUND(AVG(
         CASE
           WHEN acl_pos_ratio_str IS NULL OR TRIM(acl_pos_ratio_str) = '' THEN NULL
           WHEN TRIM(acl_pos_ratio_str) LIKE '%\\%' THEN CAST(REPLACE(TRIM(acl_pos_ratio_str), '%', '') AS DOUBLE) / 100.0
           ELSE CAST(TRIM(acl_pos_ratio_str) AS DOUBLE)
         END
       ), 4) AS avg_acl_ratio
FROM clin_lab_exam_di
WHERE dt BETWEEN '20261013' AND '20261015'
GROUP BY COALESCE(NULLIF(TRIM(ana_pattern_code), ''), 'UNRECORDED')
ORDER BY exam_cnt DESC, ana_pattern ASC
""",
            ast_transformations=["REAL -> DOUBLE", "FACT_PARTITION_PREDICATE_INJECTION (dt BETWEEN)"],
            injected_partition_predicates=["clin_lab_exam_di.dt BETWEEN '20261013' AND '20261015'"],
            unpartitioned_dim_tables=[],
            caliber_trap="mixed_percent_decimal_string_ratio_and_dt_partition",
        ),
        make_external_task(
            task_id="ext_clin_08",
            domain="thrombosis_clinical",
            category="silent_caliber_audit",
            split="test",
            question=(
                "面向 clin_patient 中全部院际转诊患者（referral_channel_code = 'REF_TRANSFER'），统计每位患者在 clin_lab_exam_di"
                " 的 '20261015' 分区（dt）中检出斑点型抗核抗体（ana_pattern_code = 'PAT_SPECKLED'）的化验单数（speckled_exam_cnt，"
                "未检出或无化验记录的转诊患者必须保留并记 0），按 patient_id 升序排列。"
            ),
            hint="右表 clin_lab_exam_di 的 dt 与 ana_pattern_code 过滤条件必须置于 LEFT JOIN ON 或预聚合 CTE 中；输出列 patient_id, speckled_exam_cnt。",
            original_sqlite_sql="""
SELECT p.patient_id,
       COUNT(x.exam_id) AS speckled_exam_cnt
FROM clin_patient p
LEFT JOIN clin_lab_exam_di x
  ON p.patient_id = x.patient_id
 AND x.ana_pattern_code = 'PAT_SPECKLED'
WHERE p.referral_channel_code = 'REF_TRANSFER'
GROUP BY p.patient_id
ORDER BY p.patient_id ASC
""",
            gold_sql="""
SELECT p.patient_id,
       COUNT(x.exam_id) AS speckled_exam_cnt
FROM clin_patient p
LEFT JOIN clin_lab_exam_di x
  ON p.patient_id = x.patient_id
 AND x.dt = '20261015'
 AND x.ana_pattern_code = 'PAT_SPECKLED'
WHERE p.referral_channel_code = 'REF_TRANSFER'
GROUP BY p.patient_id
ORDER BY p.patient_id ASC
""",
            ast_transformations=["FACT_PARTITION_PREDICATE_INJECTION (LEFT JOIN ON x.dt)"],
            injected_partition_predicates=["ON x.dt = '20261015'"],
            unpartitioned_dim_tables=["clin_patient"],
            caliber_trap="left_join_on_clause_dt_and_enum_preservation",
        ),
    ]
