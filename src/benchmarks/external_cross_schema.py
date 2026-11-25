"""External cross-schema warehouse benchmark (`external_bird_mini`):
24 tasks across 3 unseen non-e-commerce BIRD Mini-Dev domains (11 normalized tables):
  1. Retail Banking & Loans (`financial`)
  2. Clinical Hospital Admissions (`thrombosis_clinical`)
  3. Public School Performance (`california_schools`)

Note: Scores on warehouse-adapted BIRD schemas measure cross-schema & partition-constraint
generalization under EvoDW's ODPS/Hive gateway, and MUST NOT be conflated with raw SQLite
leaderboard numbers on original BIRD-bench.
"""

from __future__ import annotations

import datetime as dt
import random
import re
from typing import Any

from src.benchmarks.types import BenchmarkTask, CategoryType, SplitType
from src.env.local_executor import LocalDuckDBExecutor
from src.env.sandbox import DialectType, WarehouseSandbox
from src.env.schema import ColumnMeta, TableMeta, WarehouseCatalog

EXTERNAL_ADAPTATION_DISCLAIMER = (
    "Warehouse-adapted BIRD Mini-Dev schemas inject ds/dt partition contracts and transpile SQLite idioms "
    "to ODPS/Hive ASTs; scores measure cross-schema & partition-constraint generalization under EvoDW's "
    "gateway and MUST NOT be conflated with raw SQLite leaderboard numbers on original BIRD-bench."
)

EXTERNAL_DOMAINS: tuple[str, ...] = (
    "financial",
    "thrombosis_clinical",
    "california_schools",
)


def build_external_cross_schema_catalog() -> WarehouseCatalog:
    """Build the 11-table normalized multi-domain catalog across banking, clinical, and schools."""
    catalog = WarehouseCatalog()

    # --- Domain 1: Retail Banking & Loans (`financial`, 4 tables) ---
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

    # --- Domain 2: Clinical Hospital Admissions (`thrombosis_clinical`, 4 tables) ---
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

    # --- Domain 3: Public School Performance (`california_schools`, 3 tables) ---
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

    return catalog


def create_external_cross_schema_sandbox(
    seed: int = 2026,
    dialect: DialectType = "odps",
) -> WarehouseSandbox:
    """Create and deterministically populate the 11-table external multi-domain warehouse sandbox."""
    catalog = build_external_cross_schema_catalog()
    executor = LocalDuckDBExecutor(catalog=catalog, db_path=":memory:")
    rng = random.Random(seed + 101)
    con = executor.conn

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

    # 1. Populate Banking (`financial`)
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
    con.executemany("INSERT INTO fin_district VALUES (?, ?, ?, ?, ?, ?)", dist_rows)

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
    con.executemany("INSERT INTO fin_account VALUES (?, ?, ?, ?, ?)", acct_rows)

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
                l_stat = loan_statuses[i % 3]  # only A, B, C
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

    con.executemany("INSERT INTO fin_loan_contract_di VALUES (?, ?, ?, ?, ?, ?, ?, ?)", loan_rows)
    con.executemany("INSERT INTO fin_card_txn_di VALUES (?, ?, ?, ?, ?, ?, ?, ?)", txn_rows)

    # 2. Populate Clinical (`thrombosis_clinical`)
    sex_choices = ["F", "F", "M", "", None]
    ref_codes = ["REF_OUTPATIENT", "REF_ER", "REF_TRANSFER"]
    pat_rows = []
    for i in range(1, 51):
        pid = f"PAT{i:04d}"
        b_dt = dt.date(1965, 1, 1) + dt.timedelta(days=(i * 173) % 12_000)
        fv_dt = dt.date(2025, 1, 10) + dt.timedelta(days=(i * 9) % 240)
        pat_rows.append((pid, sex_choices[i % len(sex_choices)], b_dt.strftime("%Y-%m-%d"), fv_dt.strftime("%Y-%m-%d"), ref_codes[i % 3]))
    con.executemany("INSERT INTO clin_patient VALUES (?, ?, ?, ?, ?)", pat_rows)

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
    con.executemany("INSERT INTO clin_diagnosis_scd2 VALUES (?, ?, ?, ?, ?, ?, ?)", scd_rows)

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

    con.executemany("INSERT INTO clin_encounter_di VALUES (?, ?, ?, ?, ?, ?, ?, ?)", enc_rows)
    con.executemany("INSERT INTO clin_lab_exam_di VALUES (?, ?, ?, ?, ?, ?, ?, ?)", lab_rows)

    # 3. Populate Schools (`california_schools`)
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
    con.executemany("INSERT INTO edu_school_dim VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sch_rows)

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

    con.executemany("INSERT INTO edu_frpm_snap_di VALUES (?, ?, ?, ?, ?, ?, ?, ?)", frpm_rows)
    con.executemany("INSERT INTO edu_sat_report_di VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sat_rows)

    _ = rng
    return WarehouseSandbox(catalog=catalog, executor=executor, default_dialect=dialect)


def _mk_ext_task(
    *,
    task_id: str,
    domain: str,
    category: CategoryType,
    split: SplitType,
    question: str,
    hint: str,
    original_sqlite_sql: str,
    gold_sql: str,
    ast_transformations: list[str],
    injected_partition_predicates: list[str],
    unpartitioned_dim_tables: list[str],
    caliber_trap: str,
    ordered: bool = True,
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=category,
        split=split,
        question=question,
        hint=hint,
        gold_sql=gold_sql.strip(),
        ordered=ordered,
        metadata={
            "domain": domain,
            "adaptation_ledger": {
                "domain": domain,
                "original_sqlite_sql": original_sqlite_sql.strip(),
                "ast_transformations": ast_transformations,
                "injected_partition_predicates": injected_partition_predicates,
                "unpartitioned_dim_tables": unpartitioned_dim_tables,
                "caliber_trap": caliber_trap,
                "disclaimer": EXTERNAL_ADAPTATION_DISCLAIMER,
            },
        },
    )


def get_external_bird_mini_tasks() -> list[BenchmarkTask]:
    """Return all 24 cross-schema tasks across Banking, Clinical, and Schools with adaptation ledgers."""
    return [
        # ==================== Domain 1: Retail Banking & Loans (`financial`) ====================
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        # ==================== Domain 2: Clinical Hospital Admissions (`thrombosis_clinical`) ====================
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
 AND e.admit_dt BETWEEN d_hist.eff_start_dt AND IFNULL(NULLIF(d_hist.eff_end_dt, ''), '9999-12-31')
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        # ==================== Domain 3: Public School Performance (`california_schools`) ====================
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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
        _mk_ext_task(
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


def to_hive_external_gold_sql(odps_sql: str, latest_ds: str = "20261015") -> str:
    """Deterministically adapt an external_bird_mini ODPS Gold SQL into Hive-compliant Gold SQL."""
    hive_sql = re.sub(r"MAX_PT\('[^']+'\)", f"'{latest_ds}'", odps_sql)
    hive_sql = re.sub(
        r"DATEDIFF\(\s*TO_DATE\(([^,]+),\s*'yyyy-mm-dd'\)\s*,\s*TO_DATE\(([^,]+),\s*'yyyy-mm-dd'\)\s*,\s*'dd'\s*\)",
        r"DATEDIFF(\1, \2)",
        hive_sql,
    )
    hive_sql = re.sub(
        r"DATEDIFF\(\s*TO_DATE\('2026-10-15',\s*'yyyy-mm-dd'\)\s*,\s*TO_DATE\(([^,]+),\s*'yyyy-mm-dd'\)\s*,\s*'yyyy'\s*\)",
        r"(CAST(SUBSTR('2026-10-15', 1, 4) AS INT) - CAST(SUBSTR(\1, 1, 4) AS INT))",
        hive_sql,
    )
    return hive_sql
