"""Deterministic data populator for the 6 Hard-v2 enterprise deep-water tables."""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

from src.env.local_executor import bulk_insert_rows


def populate_hard_v2_tables(
    *,
    con: Any,
    seed: int,
    start_date: dt.date,
    num_days: int,
    all_user_ids: list[str],
    all_shop_ids: list[str],
    paid_orders_pool: list[tuple[str, str, str, float, dt.date]],
) -> None:
    """Deterministically populate the 6 Hard-v2 tables using an isolated RNG stream."""
    rng2 = random.Random(seed + 2000)

    con.execute(
        """
        CREATE TABLE dwd_trade_order_legacy_di (
            order_id VARCHAR,
            user_id VARCHAR,
            shop_id VARCHAR,
            pay_channel VARCHAR,
            settle_status VARCHAR,
            order_amt DOUBLE,
            ds VARCHAR
        );
        CREATE TABLE dwd_order_settle_di (
            settle_id VARCHAR,
            order_id VARCHAR,
            user_id VARCHAR,
            shop_id VARCHAR,
            order_dt VARCHAR,
            pay_channel_code VARCHAR,
            settle_status_code VARCHAR,
            gross_order_amt_cent BIGINT,
            settle_amt_cent BIGINT,
            platform_subsidy_cent BIGINT,
            merchant_net_amt BIGINT,
            is_test_acct INTEGER,
            ds VARCHAR
        );
        CREATE TABLE dwd_aftersale_refund_di (
            aftersale_id VARCHAR,
            order_id VARCHAR,
            user_id VARCHAR,
            shop_id VARCHAR,
            refund_type_code VARCHAR,
            refund_status_code VARCHAR,
            refund_amt_cent BIGINT,
            shipping_comp_yuan DOUBLE,
            dt VARCHAR
        );
        CREATE TABLE dim_user_vip_scd2 (
            scd_id VARCHAR,
            user_id VARCHAR,
            vip_tier VARCHAR,
            growth_score INTEGER,
            eff_start_dt VARCHAR,
            eff_end_dt VARCHAR,
            is_current INTEGER
        );
        CREATE TABLE dim_shop_commission_scd2 (
            comm_scd_id VARCHAR,
            shop_id VARCHAR,
            contract_tier VARCHAR,
            comm_rate_bps INTEGER,
            eff_start_dt VARCHAR,
            eff_end_dt VARCHAR,
            is_current INTEGER
        );
        CREATE TABLE dws_shop_operating_1d (
            shop_id VARCHAR,
            biz_date VARCHAR,
            op_status_code VARCHAR,
            abnormal_flag VARCHAR,
            gross_gmv_yuan DOUBLE,
            clean_gmv_yuan DOUBLE,
            test_gmv_yuan DOUBLE,
            promo_cost_yuan DOUBLE,
            valid_order_cnt INTEGER,
            ds VARCHAR
        );
        """
    )

    # 1. dim_user_vip_scd2 (600 users, full timeline coverage 2025-01-01 .. 9999-12-31)
    tier_ladder = ["normal", "silver", "gold", "platinum", "black_diamond"]
    user_scd_rows: list[tuple[Any, ...]] = []
    scd_seq = 1
    for idx_u, uid in enumerate(all_user_ids):
        base_idx = rng2.choices([0, 1, 2, 3], weights=[0.42, 0.28, 0.20, 0.10], k=1)[0]
        if idx_u % 20 < 10:
            # 50% static single segment
            user_scd_rows.append(
                (
                    f"USCD{scd_seq:05d}",
                    uid,
                    tier_ladder[base_idx],
                    (base_idx + 1) * 300 + rng2.randint(0, 180),
                    "2025-01-01",
                    "9999-12-31",
                    1,
                )
            )
            scd_seq += 1
        elif idx_u % 20 < 17:
            # 35% two segments (split inside 2026-10-02 .. 2026-10-08)
            split_day = dt.date(2026, 10, 2) + dt.timedelta(days=(idx_u % 7))
            prev_end = (split_day - dt.timedelta(days=1)).isoformat()
            next_idx = min(len(tier_ladder) - 1, base_idx + 1)
            user_scd_rows.append(
                (
                    f"USCD{scd_seq:05d}",
                    uid,
                    tier_ladder[base_idx],
                    (base_idx + 1) * 300 + rng2.randint(0, 120),
                    "2025-01-01",
                    prev_end,
                    0,
                )
            )
            scd_seq += 1
            user_scd_rows.append(
                (
                    f"USCD{scd_seq:05d}",
                    uid,
                    tier_ladder[next_idx],
                    (next_idx + 1) * 350 + rng2.randint(20, 200),
                    split_day.isoformat(),
                    "9999-12-31",
                    1,
                )
            )
            scd_seq += 1
        else:
            # 15% three segments (upgrade then upgrade/downgrade)
            d1 = dt.date(2026, 9, 29) + dt.timedelta(days=(idx_u % 3))
            d2 = dt.date(2026, 10, 9) + dt.timedelta(days=(idx_u % 4))
            mid_idx = min(len(tier_ladder) - 1, base_idx + 1)
            last_idx = max(0, mid_idx - 1) if idx_u % 2 == 0 else min(len(tier_ladder) - 1, mid_idx + 1)
            user_scd_rows.append(
                (
                    f"USCD{scd_seq:05d}",
                    uid,
                    tier_ladder[base_idx],
                    (base_idx + 1) * 280,
                    "2025-01-01",
                    (d1 - dt.timedelta(days=1)).isoformat(),
                    0,
                )
            )
            scd_seq += 1
            user_scd_rows.append(
                (
                    f"USCD{scd_seq:05d}",
                    uid,
                    tier_ladder[mid_idx],
                    (mid_idx + 1) * 340,
                    d1.isoformat(),
                    (d2 - dt.timedelta(days=1)).isoformat(),
                    0,
                )
            )
            scd_seq += 1
            user_scd_rows.append(
                (
                    f"USCD{scd_seq:05d}",
                    uid,
                    tier_ladder[last_idx],
                    (last_idx + 1) * 390,
                    d2.isoformat(),
                    "9999-12-31",
                    1,
                )
            )
            scd_seq += 1

    # 2. dim_shop_commission_scd2 (40 shops, continuous timeline 2025-01-01 .. 9999-12-31)
    shop_scd_rows: list[tuple[Any, ...]] = []
    comm_seq = 1
    contract_tiers = ["KA", "CORE", "STD"]
    base_bps_choices = [300, 450, 600, 800, 1000]
    for idx_s, sid in enumerate(all_shop_ids):
        ctier = contract_tiers[idx_s % 3]
        b_bps = base_bps_choices[idx_s % len(base_bps_choices)]
        if idx_s % 5 in (0, 1):
            shop_scd_rows.append(
                (f"CSCD{comm_seq:04d}", sid, ctier, b_bps, "2025-01-01", "9999-12-31", 1)
            )
            comm_seq += 1
        elif idx_s % 5 in (2, 3):
            split_day = dt.date(2026, 10, 3) + dt.timedelta(days=(idx_s % 5))
            new_bps = b_bps + 150 if idx_s % 2 == 0 else max(200, b_bps - 100)
            shop_scd_rows.append(
                (
                    f"CSCD{comm_seq:04d}",
                    sid,
                    ctier,
                    b_bps,
                    "2025-01-01",
                    (split_day - dt.timedelta(days=1)).isoformat(),
                    0,
                )
            )
            comm_seq += 1
            shop_scd_rows.append(
                (
                    f"CSCD{comm_seq:04d}",
                    sid,
                    "KA" if new_bps <= 400 else ctier,
                    new_bps,
                    split_day.isoformat(),
                    "9999-12-31",
                    1,
                )
            )
            comm_seq += 1
        else:
            d1 = dt.date(2026, 9, 29)
            d2 = dt.date(2026, 10, 9)
            shop_scd_rows.append(
                (f"CSCD{comm_seq:04d}", sid, ctier, b_bps, "2025-01-01", "2026-09-28", 0)
            )
            comm_seq += 1
            shop_scd_rows.append(
                (f"CSCD{comm_seq:04d}", sid, ctier, b_bps + 100, d1.isoformat(), "2026-10-08", 0)
            )
            comm_seq += 1
            shop_scd_rows.append(
                (f"CSCD{comm_seq:04d}", sid, "KA", b_bps + 250, d2.isoformat(), "9999-12-31", 1)
            )
            comm_seq += 1

    # Organize v1 paid_orders_pool by date for realistic cross-table order_id linkage.
    orders_by_day: dict[dt.date, list[tuple[str, str, str, float, dt.date]]] = {}
    for entry in paid_orders_pool:
        orders_by_day.setdefault(entry[4], []).append(entry)

    legacy_rows: list[tuple[Any, ...]] = []
    settle_rows: list[tuple[Any, ...]] = []
    aftersale_rows: list[tuple[Any, ...]] = []
    operating_rows: list[tuple[Any, ...]] = []

    settle_seq = 1
    aftersale_seq = 1
    settled_pool: list[tuple[str, str, str, int, dt.date]] = []

    # Keep shops S035..S038 refund-free in recent window so LEFT JOIN null-preservation traps discriminate sharply.
    zero_recent_refund_shops = {"S035", "S036", "S037", "S038"}
    # Keep shops S037..S040 sparse in dws_shop_operating_1d during 20261012..20261015 for LEFT JOIN traps.
    sparse_operating_shops = {"S037", "S038", "S039", "S040"}

    for day_offset in range(num_days):
        day = start_date + dt.timedelta(days=day_offset)
        ds = day.strftime("%Y%m%d")
        day_str = day.strftime("%Y-%m-%d")

        # Pull ~36 real orders from paid_orders_pool (including lagged 1-3 day settlement!) + ~24 standalone orders
        candidate_real: list[tuple[str, str, str, float, dt.date]] = []
        for lag in (0, 1, 2):
            src_day = day - dt.timedelta(days=lag)
            day_pool = orders_by_day.get(src_day, [])
            take_n = 18 if lag == 0 else (10 if lag == 1 else 6)
            if day_pool:
                # Deterministic slice based on lag so we never duplicate an order_id across settle partitions!
                start_i = 0 if lag == 0 else (18 if lag == 1 else 28)
                candidate_real.extend(day_pool[start_i : start_i + take_n])

        for oid, uid, sid, opay_yuan, o_date in candidate_real:
            gross_cent = max(500, int(round(opay_yuan * 100)))
            subsidy_cent = int(round(gross_cent * rng2.choice([0.0, 0.05, 0.08, 0.12])))
            settle_cent = max(100, gross_cent - subsidy_cent)
            fee_cent = max(10, int(round(settle_cent * 0.04)))
            net_cent = settle_cent - fee_cent
            p_code = rng2.choices(
                ["P01_WECHAT", "P02_ALIPAY", "P03_CREDIT"],
                weights=[0.48, 0.37, 0.15],
                k=1,
            )[0]
            s_code = rng2.choices(
                ["20_SETTLED", "10_WAIT_CLEAR", "90_REVERSED"],
                weights=[0.74, 0.16, 0.10],
                k=1,
            )[0]
            is_test = 1 if rng2.random() < 0.11 else 0
            if is_test == 1:
                # Inflate test orders so failing to filter is_test_acct=0 skews totals noticeably!
                gross_cent *= 3
                settle_cent *= 3
                net_cent *= 3

            settle_rows.append(
                (
                    f"STL{ds}_{settle_seq:05d}",
                    oid,
                    uid,
                    sid,
                    o_date.isoformat(),
                    p_code,
                    s_code,
                    gross_cent,
                    settle_cent,
                    subsidy_cent,
                    net_cent,
                    is_test,
                    ds,
                )
            )
            settle_seq += 1
            if s_code == "20_SETTLED" and is_test == 0:
                settled_pool.append((oid, uid, sid, settle_cent, day))

            # Populate partial/incomplete legacy decoy table (only ~35% of rows and stale/degraded after 20261010)
            if rng2.random() < (0.40 if ds <= "20261010" else 0.15):
                legacy_rows.append(
                    (
                        oid,
                        uid,
                        sid,
                        "wechat" if p_code == "P01_WECHAT" else "alipay",
                        "settled" if s_code == "20_SETTLED" else "unsettled",
                        round(settle_cent / 100.0 * 0.85, 2),
                        ds,
                    )
                )

        # Add 20 synthetic settlement orders (including dedicated high-value test accounts and lagged order_dt)
        for synth_i in range(20):
            oid = f"SO{ds}_{settle_seq:05d}"
            uid = rng2.choice(all_user_ids)
            sid = all_shop_ids[(day_offset + synth_i) % 36]
            lag_days = rng2.choices([0, 1, 2, 3], weights=[0.50, 0.25, 0.15, 0.10], k=1)[0]
            o_dt_str = max(dt.date(2026, 9, 22), day - dt.timedelta(days=lag_days)).isoformat()
            is_test = 1 if synth_i < 4 else 0
            gross_cent = rng2.randint(8000, 95000) * (4 if is_test else 1)
            subsidy_cent = 0 if rng2.random() < 0.45 else int(gross_cent * 0.06)
            settle_cent = gross_cent - subsidy_cent
            net_cent = int(round(settle_cent * 0.95))
            p_code = ["P01_WECHAT", "P02_ALIPAY", "P03_CREDIT"][synth_i % 3]
            s_code = "20_SETTLED" if synth_i % 5 != 4 else ("10_WAIT_CLEAR" if synth_i % 2 == 0 else "90_REVERSED")
            settle_rows.append(
                (
                    f"STL{ds}_{settle_seq:05d}",
                    oid,
                    uid,
                    sid,
                    o_dt_str,
                    p_code,
                    s_code,
                    gross_cent,
                    settle_cent,
                    subsidy_cent,
                    net_cent,
                    is_test,
                    ds,
                )
            )
            settle_seq += 1
            if s_code == "20_SETTLED" and is_test == 0:
                settled_pool.append((oid, uid, sid, settle_cent, day))

        # 3. dwd_aftersale_refund_di (partitioned by dt!)
        elig_settled = [
            x
            for x in settled_pool
            if 0 <= (day - x[4]).days <= 3
            and not (ds >= "20261010" and x[2] in zero_recent_refund_shops)
        ]
        if elig_settled:
            chosen_as = rng2.sample(elig_settled, k=min(len(elig_settled), 18))
            for idx_a, (oid, uid, sid, s_cent, _) in enumerate(chosen_as):
                # 25% of refunded orders get 2 partial aftersale tickets to trigger join fanout!
                parts = 2 if idx_a % 4 == 0 else 1
                for p_i in range(parts):
                    as_id = f"AS{ds}_{aftersale_seq:05d}"
                    aftersale_seq += 1
                    r_type = ["RT01_ONLY_REFUND", "RT02_RETURN_GOODS", "RT03_PRICE_PROTECT"][(idx_a + p_i) % 3]
                    r_stat = rng2.choices(
                        ["02_REFUND_OK", "01_APPLY", "03_REJECTED"],
                        weights=[0.64, 0.20, 0.16],
                        k=1,
                    )[0]
                    frac = 0.28 if parts == 2 else rng2.uniform(0.35, 0.90)
                    r_cent = max(100, int(round(s_cent * frac)))
                    ship_yuan = (
                        None
                        if (r_type != "RT02_RETURN_GOODS" or rng2.random() < 0.35)
                        else round(rng2.choice([8.0, 10.0, 12.5, 15.0]), 2)
                    )
                    aftersale_rows.append(
                        (as_id, oid, uid, sid, r_type, r_stat, r_cent, ship_yuan, ds)
                    )

        # 4. dws_shop_operating_1d (partitioned by ds)
        for s_idx, sid in enumerate(all_shop_ids):
            if ds >= "20261012" and sid in sparse_operating_shops:
                continue
            op_code = "OP2_PAUSED" if (s_idx + day_offset) % 9 == 0 else "OP1_OPEN"
            if (s_idx + day_offset) % 11 == 0:
                ab_flag = "BRUSH_SUSPECT"
            elif (s_idx + day_offset) % 17 == 0:
                ab_flag = "SYS_DRILL"
            else:
                ab_flag = "NORMAL"

            if op_code == "OP2_PAUSED":
                valid_cnt = 0
                clean_gmv = 0.0
                test_gmv = 0.0 if rng2.random() < 0.5 else None
                gross_gmv = round(rng2.uniform(0.0, 300.0), 2) if ab_flag != "NORMAL" else 0.0
                promo_cost = None if rng2.random() < 0.7 else 0.0
            else:
                # Intentional discrete valid_order_cnt bucket values so ties occur within a shop across days!
                valid_cnt = rng2.choice([6, 8, 8, 12, 12, 15, 20, 25, 30])
                clean_gmv = round(valid_cnt * rng2.uniform(85.0, 210.0), 2)
                test_gmv = (
                    round(rng2.uniform(600.0, 2400.0), 2)
                    if (ab_flag == "SYS_DRILL" or s_idx % 6 == 0)
                    else (None if s_idx % 3 == 0 else 0.0)
                )
                brush_gmv = round(rng2.uniform(1800.0, 5500.0), 2) if ab_flag == "BRUSH_SUSPECT" else 0.0
                gross_gmv = round(clean_gmv + brush_gmv + (test_gmv or 0.0), 2)
                p_roll = rng2.random()
                if p_roll < 0.20:
                    promo_cost = None
                elif p_roll < 0.38:
                    promo_cost = 0.0
                else:
                    promo_cost = round(rng2.uniform(80.0, 950.0), 2)

            operating_rows.append(
                (
                    sid,
                    day_str,
                    op_code,
                    ab_flag,
                    gross_gmv,
                    clean_gmv,
                    test_gmv,
                    promo_cost,
                    valid_cnt,
                    ds,
                )
            )

    bulk_insert_rows(con, "dim_user_vip_scd2", user_scd_rows)
    bulk_insert_rows(con, "dim_shop_commission_scd2", shop_scd_rows)
    bulk_insert_rows(con, "dwd_trade_order_legacy_di", legacy_rows)
    bulk_insert_rows(con, "dwd_order_settle_di", settle_rows)
    bulk_insert_rows(con, "dwd_aftersale_refund_di", aftersale_rows)
    bulk_insert_rows(con, "dws_shop_operating_1d", operating_rows)
