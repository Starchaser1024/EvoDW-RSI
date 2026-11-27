"""Deterministic enterprise e-commerce & content data warehouse dataset generator."""

from __future__ import annotations

import datetime as dt
import json
import random
from typing import Any

from src.benchmarks.catalog_builder import build_warehouse_catalog
from src.benchmarks.hard_v2_populator import populate_hard_v2_tables
from src.env.local_executor import LocalDuckDBExecutor, bulk_insert_rows
from src.env.sandbox import DialectType, WarehouseSandbox

_populate_hard_v2_tables = populate_hard_v2_tables
_SEED_SNAPSHOT_CACHE: dict[int, list[tuple[str, str, list[list[Any]]]]] = {}


def create_populated_sandbox(
    seed: int = 2026,
    dialect: DialectType = "odps",
) -> WarehouseSandbox:
    """Create an in-memory DuckDB sandbox populated with realistic enterprise warehouse data."""
    catalog = build_warehouse_catalog()
    executor = LocalDuckDBExecutor(catalog=catalog, db_path=":memory:")
    con = executor.conn

    cached_snapshot = _SEED_SNAPSHOT_CACHE.get(seed)
    if cached_snapshot is not None:
        for ddl_sql, tbl_name, cols in cached_snapshot:
            con.execute(ddl_sql)
            if cols:
                unnest_exprs = ", ".join(f"unnest(${i + 1})" for i in range(len(cols)))
                con.execute(f"INSERT INTO {tbl_name} SELECT {unnest_exprs}", cols)
        return WarehouseSandbox(
            catalog=catalog,
            executor=executor,
            default_dialect=dialect,
            immutable_cache_tag=f"core_seed_{seed}",
        )

    rng = random.Random(seed)

    # 1. Create tables.
    con.execute(
        """
        CREATE TABLE dim_user (
            user_id VARCHAR,
            reg_dt VARCHAR,
            vip_level VARCHAR,
            city_tier VARCHAR,
            gender VARCHAR,
            channel VARCHAR,
            is_blacklist INTEGER
        );
        CREATE TABLE dim_shop (
            shop_id VARCHAR,
            shop_name VARCHAR,
            shop_type VARCHAR,
            main_cate VARCHAR,
            open_dt VARCHAR,
            rating_score DOUBLE
        );
        CREATE TABLE dim_item (
            item_id VARCHAR,
            shop_id VARCHAR,
            cate_l1 VARCHAR,
            cate_l2 VARCHAR,
            brand VARCHAR,
            list_price DOUBLE,
            status VARCHAR
        );
        CREATE TABLE dim_content_creator (
            creator_id VARCHAR,
            creator_name VARCHAR,
            creator_tier VARCHAR,
            mcn_org VARCHAR,
            fans_cnt INTEGER
        );
        CREATE TABLE ods_app_event_log_di (
            event_id VARCHAR,
            user_id VARCHAR,
            event_type VARCHAR,
            page_name VARCHAR,
            ingest_ts VARCHAR,
            ext_json VARCHAR,
            expo_item_ids VARCHAR[],
            ds VARCHAR
        );
        CREATE TABLE dwd_trade_order_di (
            order_id VARCHAR,
            order_item_id VARCHAR,
            user_id VARCHAR,
            shop_id VARCHAR,
            item_id VARCHAR,
            creator_id VARCHAR,
            order_status VARCHAR,
            qty INTEGER,
            gross_amt DOUBLE,
            discount_amt DOUBLE,
            pay_amt DOUBLE,
            pay_ts VARCHAR,
            ds VARCHAR
        );
        CREATE TABLE dwd_refund_di (
            refund_id VARCHAR,
            order_id VARCHAR,
            user_id VARCHAR,
            shop_id VARCHAR,
            refund_amt DOUBLE,
            refund_reason VARCHAR,
            refund_status VARCHAR,
            ds VARCHAR
        );
        CREATE TABLE dwd_content_post_di (
            post_id VARCHAR,
            creator_id VARCHAR,
            content_type VARCHAR,
            topic_tags VARCHAR[],
            meta_json VARCHAR,
            view_cnt INTEGER,
            like_cnt INTEGER,
            dt VARCHAR
        );
        CREATE TABLE dws_shop_metrics_1d (
            shop_id VARCHAR,
            uv INTEGER,
            pv INTEGER,
            order_cnt INTEGER,
            buyer_cnt INTEGER,
            gmv DOUBLE,
            return_amt DOUBLE,
            ad_spend DOUBLE,
            ds VARCHAR
        );
        """
    )

    # 2. Populate dimension tables.
    # dim_user (600 rows)
    vip_levels = ["normal", "silver", "gold", "platinum"]
    vip_weights = [0.50, 0.25, 0.17, 0.08]
    channels = ["organic", "paid_ads", "referral", "live_stream"]
    city_choices = ["T1", "T2", "T3", "", None]
    city_weights = [0.28, 0.32, 0.22, 0.10, 0.08]
    gender_choices = ["M", "F", "", None]
    gender_weights = [0.43, 0.45, 0.06, 0.06]

    reg_start = dt.date(2026, 9, 15)
    user_rows = []
    user_reg_map: dict[str, dt.date] = {}
    for i in range(1, 601):
        uid = f"U{i:04d}"
        # Concentrate ~35% registrations inside the Sep 25 - Oct 10 cohort window.
        if i <= 210:
            reg_d = dt.date(2026, 9, 25) + dt.timedelta(days=rng.randint(0, 15))
        else:
            reg_d = reg_start + dt.timedelta(days=rng.randint(0, 27))
        user_reg_map[uid] = reg_d
        user_rows.append(
            (
                uid,
                reg_d.strftime("%Y-%m-%d"),
                rng.choices(vip_levels, weights=vip_weights, k=1)[0],
                rng.choices(city_choices, weights=city_weights, k=1)[0],
                rng.choices(gender_choices, weights=gender_weights, k=1)[0],
                rng.choice(channels),
                1 if rng.random() < 0.05 else 0,
            )
        )
    bulk_insert_rows(con, "dim_user", user_rows)

    # dim_shop (40 rows)
    cates = ["electronics", "beauty", "apparel", "home", "food"]
    cate_l2_map = {
        "electronics": ["phone", "audio"],
        "beauty": ["skincare", "makeup"],
        "apparel": ["dress", "shoes"],
        "home": ["kitchen", "bedding"],
        "food": ["snacks", "beverage"],
    }
    shop_types = ["flagship", "pop_store", "cross_border", "brand_direct"]
    shop_rows = []
    shop_cate_map: dict[str, str] = {}
    for i in range(1, 41):
        sid = f"S{i:03d}"
        mc = cates[(i - 1) % len(cates)]
        shop_cate_map[sid] = mc
        open_d = dt.date(2022, 1, 1) + dt.timedelta(days=rng.randint(0, 1400))
        shop_rows.append(
            (
                sid,
                f"Shop_{mc[:4].title()}_{i:02d}",
                shop_types[(i - 1) % len(shop_types)],
                mc,
                open_d.strftime("%Y-%m-%d"),
                round(rng.uniform(4.15, 4.98), 2),
            )
        )
    bulk_insert_rows(con, "dim_shop", shop_rows)

    # dim_item (200 rows)
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "BrandF", "", None]
    brand_weights = [0.16, 0.15, 0.15, 0.14, 0.13, 0.12, 0.08, 0.07]
    item_rows = []
    item_shop_map: dict[str, str] = {}
    item_price_map: dict[str, float] = {}
    for i in range(1, 201):
        iid = f"I{i:04d}"
        sid = f"S{((i - 1) % 40) + 1:03d}"
        c1 = shop_cate_map[sid]
        c2 = rng.choice(cate_l2_map[c1])
        price = round(rng.uniform(29.0, 1899.0), 2)
        item_shop_map[iid] = sid
        item_price_map[iid] = price
        item_rows.append(
            (
                iid,
                sid,
                c1,
                c2,
                rng.choices(brands, weights=brand_weights, k=1)[0],
                price,
                "off_shelf" if rng.random() < 0.08 else "on_sale",
            )
        )
    bulk_insert_rows(con, "dim_item", item_rows)

    # dim_content_creator (80 rows)
    creator_tiers = ["head", "waist", "tail"]
    creator_tier_weights = [0.15, 0.35, 0.50]
    mcn_choices = ["StarMCN", "NovaMedia", "PulseStudio", "PeakMCN", "", None]
    mcn_weights = [0.22, 0.22, 0.20, 0.16, 0.10, 0.10]
    creator_rows = []
    for i in range(1, 81):
        cid = f"C{i:03d}"
        tier = rng.choices(creator_tiers, weights=creator_tier_weights, k=1)[0]
        fans = (
            rng.randint(500_000, 3_000_000)
            if tier == "head"
            else rng.randint(50_000, 499_999)
            if tier == "waist"
            else rng.randint(2_000, 49_999)
        )
        creator_rows.append(
            (
                cid,
                f"Creator_{i:03d}",
                tier,
                rng.choices(mcn_choices, weights=mcn_weights, k=1)[0],
                fans,
            )
        )
    bulk_insert_rows(con, "dim_content_creator", creator_rows)

    # 3. Populate 21 days of partitioned tables: 2026-09-25 through 2026-10-15.
    days = [dt.date(2026, 9, 25) + dt.timedelta(days=d) for d in range(21)]
    all_item_ids = [f"I{i:04d}" for i in range(1, 201)]
    all_creator_ids = [f"C{i:03d}" for i in range(1, 81)]

    # Keep cohort active users deterministic across days so retention tasks always have rich overlap.
    event_rows = []
    order_rows = []
    refund_rows = []
    post_rows = []
    shop_metric_rows = []

    event_seq = 1
    order_seq = 1
    order_item_seq = 1
    refund_seq = 1
    post_seq = 1

    # Track paid orders to create realistic refunds (and partial multi-refunds).
    paid_orders_pool: list[tuple[str, str, str, float, dt.date]] = []

    for day in days:
        ds = day.strftime("%Y%m%d")
        day_str = day.strftime("%Y-%m-%d")

        # Eligible users registered on or before `day`.
        eligible_users = [uid for uid, rd in user_reg_map.items() if rd <= day]

        # Guarantee strong cohort activity on D0, D1, D3, D7 for retention queries.
        active_today = set(rng.sample(eligible_users, k=min(len(eligible_users), 90)))
        for uid, rd in user_reg_map.items():
            lag = (day - rd).days
            if lag in (0, 1, 3, 7) and (int(uid[1:]) % 3 != 0):
                active_today.add(uid)
            elif lag in (1, 3, 7) and rng.random() < 0.45:
                active_today.add(uid)

        # ods_app_event_log_di (~115 events/day + ~9 duplicate event_id rows/day)
        day_events_list = sorted(active_today)
        for idx_u, uid in enumerate(day_events_list):
            ev_id = f"E{ds}_{event_seq:05d}"
            event_seq += 1
            ev_type = rng.choices(
                ["exposure", "click", "add_cart", "purchase"],
                weights=[0.48, 0.28, 0.14, 0.10],
                k=1,
            )[0]
            page = rng.choice(["home_feed", "search_result", "live_room", "item_detail"])
            hh = rng.randint(8, 22)
            mm = rng.randint(0, 59)
            ss = rng.randint(0, 59)
            ingest_dt = dt.datetime(day.year, day.month, day.day, hh, mm, ss)

            r_json = rng.random()
            if r_json < 0.05:
                ext_json = None
            elif r_json < 0.09:
                ext_json = ""
            elif r_json < 0.11:
                ext_json = "{broken_json"
            else:
                ext_json = json.dumps(
                    {
                        "page": page,
                        "stay_ms": rng.randint(400, 18000),
                        "ab_group": rng.choice(["control", "exp_a", "exp_b"]),
                        "device": {
                            "os": rng.choice(["ios", "android", "harmony"]),
                            "app_ver": rng.choice(["8.1.0", "8.2.0", "8.3.0"]),
                        },
                    },
                    ensure_ascii=False,
                )

            if ev_type == "purchase" or rng.random() < 0.15:
                expo_items: list[str] = []
            else:
                expo_items = rng.sample(all_item_ids, k=rng.randint(1, 3))

            event_rows.append(
                (
                    ev_id,
                    uid,
                    ev_type,
                    page,
                    ingest_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    ext_json,
                    expo_items,
                    ds,
                )
            )
            # Intentional duplicate event_id with later ingest_ts (~9% of events).
            if idx_u % 11 == 0:
                dup_ts = ingest_dt + dt.timedelta(seconds=rng.randint(15, 180))
                event_rows.append(
                    (
                        ev_id,
                        uid,
                        ev_type,
                        page,
                        dup_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        ext_json,
                        expo_items,
                        ds,
                    )
                )

        # dwd_trade_order_di (~60 orders/day -> ~76 order-item rows/day)
        shop_daily_stats: dict[str, dict[str, object]] = {
            f"S{s:03d}": {"uv": rng.randint(40, 320), "pv": 0, "order_ids": set(), "buyers": set(), "gmv": 0.0, "ret": 0.0}
            for s in range(1, 41)
        }
        # Give a couple shops cold-start zero UV days.
        for s_cold in (39, 40):
            if (day - dt.date(2026, 9, 25)).days % 3 == 0:
                shop_daily_stats[f"S{s_cold:03d}"]["uv"] = 0

        for _ in range(58):
            oid = f"O{ds}_{order_seq:05d}"
            order_seq += 1
            uid = rng.choice(day_events_list)
            status = rng.choices(
                ["paid", "shipped", "completed", "cancelled"],
                weights=[0.28, 0.22, 0.38, 0.12],
                k=1,
            )[0]
            # 28% multi-item orders to trigger fanout if joined directly with refunds.
            n_items = 2 if rng.random() < 0.28 else 1
            chosen_items = rng.sample(all_item_ids, k=n_items)
            # All items in an order belong to the same shop for realistic checkout.
            primary_shop = item_shop_map[chosen_items[0]]
            same_shop_items = [iid for iid in all_item_ids if item_shop_map[iid] == primary_shop]
            if len(chosen_items) == 2:
                chosen_items = rng.sample(same_shop_items, k=min(2, len(same_shop_items)))

            creator_roll = rng.random()
            if creator_roll < 0.45:
                cid = rng.choice(all_creator_ids)
            elif creator_roll < 0.72:
                cid = None
            else:
                cid = ""

            order_total_pay = 0.0
            for iid in chosen_items:
                oiid = f"OI{ds}_{order_item_seq:06d}"
                order_item_seq += 1
                qty = rng.randint(1, 3)
                gross = round(item_price_map[iid] * qty, 2)
                if rng.random() < 0.22:
                    disc = None
                    disc_val = 0.0
                else:
                    disc_val = round(gross * rng.uniform(0.05, 0.20), 2)
                    disc = disc_val

                if status == "cancelled":
                    pay = 0.0
                    pay_ts = None
                else:
                    pay = round(max(1.0, gross - disc_val), 2)
                    hh = rng.randint(0, 23)
                    mm = rng.randint(0, 59)
                    ss = rng.randint(0, 59)
                    pay_ts = f"{day_str} {hh:02d}:{mm:02d}:{ss:02d}"
                    order_total_pay += pay

                order_rows.append(
                    (
                        oid,
                        oiid,
                        uid,
                        primary_shop,
                        iid,
                        cid,
                        status,
                        qty,
                        gross,
                        disc,
                        pay,
                        pay_ts,
                        ds,
                    )
                )

            if status != "cancelled":
                paid_orders_pool.append((oid, uid, primary_shop, round(order_total_pay, 2), day))
                st = shop_daily_stats[primary_shop]
                st["order_ids"].add(oid)  # type: ignore[attr-defined]
                st["buyers"].add(uid)  # type: ignore[attr-defined]
                st["gmv"] = float(st["gmv"]) + order_total_pay  # type: ignore[arg-type]

        # dwd_refund_di (~20 refund records/day, including multiple partial refunds on ~20% refunded orders)
        eligible_for_refund = [
            o for o in paid_orders_pool if 0 <= (day - o[4]).days <= 3
        ]
        if eligible_for_refund:
            sample_refunds = rng.sample(
                eligible_for_refund, k=min(len(eligible_for_refund), 16)
            )
            for idx_r, (oid, uid, sid, opay, _) in enumerate(sample_refunds):
                n_parts = 2 if idx_r % 5 == 0 else 1
                for _ in range(n_parts):
                    rid = f"R{ds}_{refund_seq:05d}"
                    refund_seq += 1
                    frac = 0.30 if n_parts == 2 else rng.uniform(0.4, 1.0)
                    ramt = round(max(1.0, opay * frac), 2)
                    r_status = rng.choices(
                        ["approved", "pending", "rejected"],
                        weights=[0.68, 0.17, 0.15],
                        k=1,
                    )[0]
                    r_reason = rng.choice(
                        ["quality", "size_mismatch", "late_delivery", "buyer_remorse"]
                    )
                    refund_rows.append((rid, oid, uid, sid, ramt, r_reason, r_status, ds))
                    if r_status == "approved":
                        shop_daily_stats[sid]["ret"] = float(shop_daily_stats[sid]["ret"]) + ramt  # type: ignore[arg-type]

        # dwd_content_post_di (partitioned by dt! ~28 posts/day)
        all_tags = ["tech", "beauty_tips", "ootd", "home_hack", "foodie", "unboxing"]
        for idx_p in range(28):
            pid = f"P{ds}_{post_seq:05d}"
            post_seq += 1
            cid = rng.choice(all_creator_ids)
            ctype = rng.choice(["short_video", "live_replay", "article"])
            tags = [] if idx_p % 9 == 0 else rng.sample(all_tags, k=rng.randint(1, 3))
            if idx_p % 15 == 0:
                mjson = None
            else:
                mjson = json.dumps(
                    {
                        "duration_sec": rng.randint(15, 3600),
                        "is_promoted": rng.random() < 0.35,
                        "score": round(rng.uniform(55.0, 99.5), 1),
                    }
                )
            vcnt = rng.randint(300, 120_000)
            lcnt = int(vcnt * rng.uniform(0.01, 0.12))
            post_rows.append((pid, cid, ctype, tags, mjson, vcnt, lcnt, ds))

        # dws_shop_metrics_1d (40 shops/day)
        for s_idx in range(1, 41):
            sid = f"S{s_idx:03d}"
            st = shop_daily_stats[sid]
            uv = int(st["uv"])  # type: ignore[arg-type]
            pv = 0 if uv == 0 else uv * rng.randint(2, 6)
            ocnt = len(st["order_ids"])  # type: ignore[arg-type]
            bcnt = len(st["buyers"])  # type: ignore[arg-type]
            gmv = round(float(st["gmv"]), 2)  # type: ignore[arg-type]
            ret_val = float(st["ret"])  # type: ignore[arg-type]
            ret = round(ret_val, 2) if ret_val > 0 else (None if rng.random() < 0.6 else 0.0)
            ad_roll = rng.random()
            if ad_roll < 0.18:
                ad_spend = None
            elif ad_roll < 0.38:
                ad_spend = 0.0
            else:
                ad_spend = round(rng.uniform(50.0, 1800.0), 2)
            shop_metric_rows.append((sid, uv, pv, ocnt, bcnt, gmv, ret, ad_spend, ds))

    bulk_insert_rows(con, "ods_app_event_log_di", event_rows)
    bulk_insert_rows(con, "dwd_trade_order_di", order_rows)
    bulk_insert_rows(con, "dwd_refund_di", refund_rows)
    bulk_insert_rows(con, "dwd_content_post_di", post_rows)
    bulk_insert_rows(con, "dws_shop_metrics_1d", shop_metric_rows)

    populate_hard_v2_tables(
        con=con,
        seed=seed,
        start_date=days[0],
        num_days=len(days),
        all_user_ids=[f"U{i:04d}" for i in range(1, 601)],
        all_shop_ids=[f"S{i:03d}" for i in range(1, 41)],
        paid_orders_pool=paid_orders_pool,
    )

    if len(_SEED_SNAPSHOT_CACHE) < 4:
        snapshot: list[tuple[str, str, list[list[Any]]]] = []
        ddl_map = dict(con.execute("SELECT table_name, sql FROM duckdb_tables() WHERE internal = false").fetchall())
        for tmeta in catalog.list_tables():
            ddl_sql = ddl_map.get(tmeta.name)
            if not ddl_sql:
                continue
            rows = con.execute(f"SELECT * FROM {tmeta.name}").fetchall()
            cols = [list(c) for c in zip(*rows)] if rows else []
            snapshot.append((ddl_sql, tmeta.name, cols))
        _SEED_SNAPSHOT_CACHE[seed] = snapshot

    return WarehouseSandbox(
        catalog=catalog,
        executor=executor,
        default_dialect=dialect,
        immutable_cache_tag=f"core_seed_{seed}",
    )
