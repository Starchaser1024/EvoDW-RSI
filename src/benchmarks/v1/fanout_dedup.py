"""Multi-fact fanout & window deduplication tasks (15 tasks: 5 train, 4 val, 6 test)."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask


def get_fanout_dedup_tasks() -> list[BenchmarkTask]:
    return [
        # =====================================================================
        # 4. fanout_dedup (5 train, 4 val, 6 test)
        # =====================================================================
        BenchmarkTask(
            task_id="fd_01",
            category="fanout_dedup",
            split="train",
            question=(
                "对 ods_app_event_log_di 最新分区按 event_id 去重（重复上报时保留服务端落库时间 ingest_ts 最新的一条），"
                "统计去重后各 event_type 的事件数（dedup_event_cnt），按 event_type 升序排列。"
            ),
            hint="同一 event_id 可能因客户端重试重复上报，应按 event_id 分区排序去重后再聚合。",
            gold_sql=(
                "WITH dedup AS ("
                "  SELECT event_id, event_type, "
                "         ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts DESC) AS rn "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                ") "
                "SELECT event_type, COUNT(*) AS dedup_event_cnt "
                "FROM dedup WHERE rn = 1 "
                "GROUP BY event_type "
                "ORDER BY event_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_02",
            category="fanout_dedup",
            split="train",
            question=(
                "统计 '20261010' 至 '20261015' 期间各店铺的有效订单实付总额、已审批通过退款总额及净实付额（pay_amt - refund_amt），"
                "仅输出期间有有效下单的店铺（无退款记 0.0），返回 shop_id, pay_amt, refund_amt, net_amt（均保留2位小数），按 shop_id 升序排列。"
            ),
            hint="一笔主订单可能包含多个子订单且可能存在多次部分退款，直接按 order_id 关联明细表会产生多对多膨胀。",
            gold_sql=(
                "WITH ord_shop AS ("
                "  SELECT shop_id, ROUND(SUM(pay_amt), 2) AS pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015' AND order_status != 'cancelled' "
                "  GROUP BY shop_id"
                "), ref_shop AS ("
                "  SELECT shop_id, ROUND(SUM(refund_amt), 2) AS refund_amt "
                "  FROM dwd_refund_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015' AND refund_status = 'approved' "
                "  GROUP BY shop_id"
                ") "
                "SELECT o.shop_id, o.pay_amt, NVL(r.refund_amt, 0.0) AS refund_amt, "
                "ROUND(o.pay_amt - NVL(r.refund_amt, 0.0), 2) AS net_amt "
                "FROM ord_shop o "
                "LEFT JOIN ref_shop r ON o.shop_id = r.shop_id "
                "ORDER BY o.shop_id ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_03",
            category="fanout_dedup",
            split="train",
            question=(
                "统计 '20261010' 至 '20261015' 期间各达人层级（dim_content_creator.creator_tier）的"
                "内容帖子总浏览量（total_views，来自 dwd_content_post_di）和带货有效实付总额（total_pay_amt，来自 dwd_trade_order_di，保留2位小数），"
                "按 creator_tier 升序排列。"
            ),
            hint="达人的多篇帖子和多笔带货订单不可直接两表相连，需先各自按达人聚合后再关联维表。",
            gold_sql=(
                "WITH post_c AS ("
                "  SELECT creator_id, SUM(view_cnt) AS views "
                "  FROM dwd_content_post_di "
                "  WHERE dt BETWEEN '20261010' AND '20261015' "
                "  GROUP BY creator_id"
                "), ord_c AS ("
                "  SELECT creator_id, SUM(pay_amt) AS pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015' "
                "    AND order_status != 'cancelled' "
                "    AND creator_id IS NOT NULL AND creator_id != '' "
                "  GROUP BY creator_id"
                ") "
                "SELECT c.creator_tier, SUM(NVL(p.views, 0)) AS total_views, "
                "ROUND(SUM(NVL(o.pay_amt, 0.0)), 2) AS total_pay_amt "
                "FROM dim_content_creator c "
                "LEFT JOIN post_c p ON c.creator_id = p.creator_id "
                "LEFT JOIN ord_c o ON c.creator_id = o.creator_id "
                "GROUP BY c.creator_tier "
                "ORDER BY c.creator_tier ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_04",
            category="fanout_dedup",
            split="train",
            question=(
                "对 ods_app_event_log_di 最新分区按 event_id 取最新 ingest_ts 去重后，"
                "统计各页面（page_name）的去重事件数（event_cnt）与平均停留毫秒数（avg_stay_ms，取 ext_json 中 $.stay_ms，保留2位小数），"
                "按 page_name 升序排列。"
            ),
            hint="先对埋点日志按 event_id 去重，再对去重后记录统计事件数和停留均值。",
            gold_sql=(
                "WITH dedup AS ("
                "  SELECT event_id, page_name, ext_json, "
                "         ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts DESC) AS rn "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                ") "
                "SELECT page_name, COUNT(*) AS event_cnt, "
                "ROUND(AVG(CAST(GET_JSON_OBJECT(ext_json, '$.stay_ms') AS DOUBLE)), 2) AS avg_stay_ms "
                "FROM dedup WHERE rn = 1 "
                "GROUP BY page_name "
                "ORDER BY page_name ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_05",
            category="fanout_dedup",
            split="train",
            question=(
                "查询 dwd_trade_order_di 最新分区中的有效订单，统计各一级类目（dim_item.cate_l1）内实付金额最高的 Top-2 商品："
                "输出 cate_l1, item_id, item_pay_amt（保留2位小数）, rn，按 cate_l1 升序、rn 升序排列。"
            ),
            hint="先按类目与商品聚合实付金额，再在类目内排序取前两名。",
            gold_sql=(
                "WITH item_agg AS ("
                "  SELECT i.cate_l1, o.item_id, ROUND(SUM(o.pay_amt), 2) AS item_pay_amt "
                "  FROM dwd_trade_order_di o "
                "  JOIN dim_item i ON o.item_id = i.item_id "
                "  WHERE o.ds = MAX_PT('dwd_trade_order_di') AND o.order_status != 'cancelled' "
                "  GROUP BY i.cate_l1, o.item_id"
                "), ranked AS ("
                "  SELECT cate_l1, item_id, item_pay_amt, "
                "         ROW_NUMBER() OVER (PARTITION BY cate_l1 ORDER BY item_pay_amt DESC, item_id ASC) AS rn "
                "  FROM item_agg"
                ") "
                "SELECT cate_l1, item_id, item_pay_amt, rn "
                "FROM ranked WHERE rn <= 2 "
                "ORDER BY cate_l1 ASC, rn ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_06",
            category="fanout_dedup",
            split="val",
            question=(
                "对 ods_app_event_log_di 在 '20261013' 至 '20261015' 分区先按 (ds, event_id) 取最新 ingest_ts 去重，"
                "再将曝光商品数组 expo_item_ids 展开，统计曝光次数排名前 5 的商品（item_id, expo_cnt），"
                "按 expo_cnt 降序、item_id 升序排列。"
            ),
            hint="必须先去重再行转列展开，避免重复上报事件被放大多次。",
            gold_sql=(
                "WITH dedup AS ("
                "  SELECT ds, event_id, expo_item_ids, "
                "         ROW_NUMBER() OVER (PARTITION BY ds, event_id ORDER BY ingest_ts DESC) AS rn "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20261013' AND '20261015'"
                ") "
                "SELECT t.item_id, COUNT(*) AS expo_cnt "
                "FROM dedup d "
                "LATERAL VIEW explode(d.expo_item_ids) t AS item_id "
                "WHERE d.rn = 1 "
                "GROUP BY t.item_id "
                "ORDER BY expo_cnt DESC, t.item_id ASC "
                "LIMIT 5"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_07",
            category="fanout_dedup",
            split="val",
            question=(
                "查询 '20261001' 至 '20261015' 期间发生过审核通过退款（refund_status = 'approved'）的订单，"
                "按店铺类型（dim_shop.shop_type）统计这些被退款订单的原始有效实付总额（ordered_amt）与对应审核通过退款总额（refunded_amt），"
                "均保留2位小数，按 shop_type 升序排列。"
            ),
            hint="同一主订单在订单明细表和退款表都可能有多行，必须先各自按 order_id 聚合后再关联。",
            gold_sql=(
                "WITH ord_o AS ("
                "  SELECT order_id, MAX(shop_id) AS shop_id, SUM(pay_amt) AS order_pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261001' AND '20261015' AND order_status != 'cancelled' "
                "  GROUP BY order_id"
                "), ref_o AS ("
                "  SELECT order_id, SUM(refund_amt) AS order_refund_amt "
                "  FROM dwd_refund_di "
                "  WHERE ds BETWEEN '20261001' AND '20261015' AND refund_status = 'approved' "
                "  GROUP BY order_id"
                ") "
                "SELECT s.shop_type, ROUND(SUM(o.order_pay_amt), 2) AS ordered_amt, "
                "ROUND(SUM(r.order_refund_amt), 2) AS refunded_amt "
                "FROM ref_o r "
                "JOIN ord_o o ON r.order_id = o.order_id "
                "JOIN dim_shop s ON o.shop_id = s.shop_id "
                "GROUP BY s.shop_type "
                "ORDER BY s.shop_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_08",
            category="fanout_dedup",
            split="val",
            question=(
                "统计 '20261010' 至 '20261015' 期间各会员等级（dim_user.vip_level）用户的"
                "有效主订单数（order_cnt）与退款单数（refund_cnt，不限退款状态），按 vip_level 升序排列。"
            ),
            hint="为防止用户多笔订单与多笔退款交叉产生笛卡尔积，请按 user_id 分别预聚合后再关联用户维表。",
            gold_sql=(
                "WITH u_ord AS ("
                "  SELECT user_id, COUNT(DISTINCT order_id) AS order_cnt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015' AND order_status != 'cancelled' "
                "  GROUP BY user_id"
                "), u_ref AS ("
                "  SELECT user_id, COUNT(DISTINCT refund_id) AS refund_cnt "
                "  FROM dwd_refund_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015' "
                "  GROUP BY user_id"
                ") "
                "SELECT u.vip_level, SUM(NVL(o.order_cnt, 0)) AS order_cnt, "
                "SUM(NVL(r.refund_cnt, 0)) AS refund_cnt "
                "FROM dim_user u "
                "LEFT JOIN u_ord o ON u.user_id = o.user_id "
                "LEFT JOIN u_ref r ON u.user_id = r.user_id "
                "GROUP BY u.vip_level "
                "ORDER BY u.vip_level ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_09",
            category="fanout_dedup",
            split="val",
            question=(
                "查询 dws_shop_metrics_1d 最新分区中，每个主营类目（dim_shop.main_cate）内 GMV 排名第 1 的店铺："
                "输出 main_cate, shop_id, shop_name, gmv（保留2位小数），按 main_cate 升序排列。"
            ),
            hint="使用窗口排序函数在类目内取排名第一的记录（同分按 shop_id 升序）。",
            gold_sql=(
                "WITH ranked AS ("
                "  SELECT s.main_cate, m.shop_id, s.shop_name, ROUND(m.gmv, 2) AS gmv, "
                "         ROW_NUMBER() OVER (PARTITION BY s.main_cate ORDER BY m.gmv DESC, m.shop_id ASC) AS rn "
                "  FROM dws_shop_metrics_1d m "
                "  JOIN dim_shop s ON m.shop_id = s.shop_id "
                "  WHERE m.ds = MAX_PT('dws_shop_metrics_1d')"
                ") "
                "SELECT main_cate, shop_id, shop_name, gmv "
                "FROM ranked WHERE rn = 1 "
                "ORDER BY main_cate ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_10",
            category="fanout_dedup",
            split="test",
            question=(
                "对 ods_app_event_log_di 在 '20261010' 至 '20261015' 分区先按 (ds, event_id) 取最新 ingest_ts 去重，"
                "再关联 dim_user 过滤黑名单用户（is_blacklist = 0），统计各拉新渠道（channel）的有效事件数（event_cnt）与去重活跃用户数（uv），"
                "按 channel 升序排列。"
            ),
            hint="先对日志按天和事件 ID 去重，再关联用户维表过滤。",
            gold_sql=(
                "WITH dedup AS ("
                "  SELECT ds, event_id, user_id, "
                "         ROW_NUMBER() OVER (PARTITION BY ds, event_id ORDER BY ingest_ts DESC) AS rn "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015'"
                ") "
                "SELECT u.channel, COUNT(*) AS event_cnt, COUNT(DISTINCT d.user_id) AS uv "
                "FROM dedup d "
                "JOIN dim_user u ON d.user_id = u.user_id "
                "WHERE d.rn = 1 AND u.is_blacklist = 0 "
                "GROUP BY u.channel "
                "ORDER BY u.channel ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_11",
            category="fanout_dedup",
            split="test",
            question=(
                "统计 '20261005' 至 '20261015' 期间每个店铺在汇总表 dws_shop_metrics_1d 的总访客数（total_uv）"
                "和在明细表 dwd_trade_order_di 的有效主订单数（order_cnt）及实付总额（pay_amt，保留2位小数），"
                "返回 pay_amt 排名前 10 的店铺（shop_id, total_uv, order_cnt, pay_amt），按 pay_amt 降序、shop_id 升序排列。"
            ),
            hint="日汇总表与子订单明细表粒度不同，必须分别按 shop_id 聚合后再关联。",
            gold_sql=(
                "WITH m_s AS ("
                "  SELECT shop_id, SUM(uv) AS total_uv "
                "  FROM dws_shop_metrics_1d "
                "  WHERE ds BETWEEN '20261005' AND '20261015' "
                "  GROUP BY shop_id"
                "), o_s AS ("
                "  SELECT shop_id, COUNT(DISTINCT order_id) AS order_cnt, ROUND(SUM(pay_amt), 2) AS pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261005' AND '20261015' AND order_status != 'cancelled' "
                "  GROUP BY shop_id"
                ") "
                "SELECT s.shop_id, NVL(m.total_uv, 0) AS total_uv, "
                "NVL(o.order_cnt, 0) AS order_cnt, NVL(o.pay_amt, 0.0) AS pay_amt "
                "FROM dim_shop s "
                "LEFT JOIN m_s m ON s.shop_id = m.shop_id "
                "LEFT JOIN o_s o ON s.shop_id = o.shop_id "
                "ORDER BY pay_amt DESC, s.shop_id ASC "
                "LIMIT 10"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_12",
            category="fanout_dedup",
            split="test",
            question=(
                "查询 '20261001' 至 '20261015' 期间既在 dwd_content_post_di 发过帖、又在 dwd_trade_order_di 产生过有效带货订单的达人，"
                "统计每位达人的平均单帖浏览量（avg_post_views，保留2位小数）、带货有效主订单数（order_cnt）和带货实付总额（total_pay_amt，保留2位小数），"
                "取 total_pay_amt 排名前 10 名，按 total_pay_amt 降序、creator_id 升序排列。"
            ),
            hint="先按 creator_id 分别聚合帖子浏览与订单成交，再做内连接避免多对多膨胀。",
            gold_sql=(
                "WITH p_c AS ("
                "  SELECT creator_id, ROUND(AVG(view_cnt), 2) AS avg_post_views "
                "  FROM dwd_content_post_di "
                "  WHERE dt BETWEEN '20261001' AND '20261015' "
                "  GROUP BY creator_id"
                "), o_c AS ("
                "  SELECT creator_id, COUNT(DISTINCT order_id) AS order_cnt, ROUND(SUM(pay_amt), 2) AS total_pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261001' AND '20261015' "
                "    AND order_status != 'cancelled' "
                "    AND creator_id IS NOT NULL AND creator_id != '' "
                "  GROUP BY creator_id"
                ") "
                "SELECT p.creator_id, p.avg_post_views, o.order_cnt, o.total_pay_amt "
                "FROM p_c p "
                "JOIN o_c o ON p.creator_id = o.creator_id "
                "ORDER BY o.total_pay_amt DESC, p.creator_id ASC "
                "LIMIT 10"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_13",
            category="fanout_dedup",
            split="test",
            question=(
                "针对 dwd_trade_order_di 在 '20261010' 至 '20261015' 分区中的所有有效订单（order_status != 'cancelled'），"
                "取每位用户时间最近的一条子订单记录（按 pay_ts 降序、order_item_id 降序取第 1 条），"
                "统计各 order_status 的用户数（user_cnt），按 order_status 升序排列。"
            ),
            hint="使用窗口函数按用户分区取最新支付记录后再统计状态分布。",
            gold_sql=(
                "WITH latest_u AS ("
                "  SELECT user_id, order_status, "
                "         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY pay_ts DESC, order_item_id DESC) AS rn "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015' AND order_status != 'cancelled'"
                ") "
                "SELECT order_status, COUNT(*) AS user_cnt "
                "FROM latest_u WHERE rn = 1 "
                "GROUP BY order_status "
                "ORDER BY order_status ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_14",
            category="fanout_dedup",
            split="test",
            question=(
                "统计 '20261008' 至 '20261015' 期间各主营类目（dim_shop.main_cate）的："
                "有效订单总实付额（total_pay）、已审批通过退款总额（total_refund）及总广告花费（total_ad_spend，空值按0），"
                "均保留2位小数，按 main_cate 升序排列。"
            ),
            hint="三张分区表（订单、退款、店铺汇总）先各自按 shop_id 聚合再关联店铺维表。",
            gold_sql=(
                "WITH o_s AS ("
                "  SELECT shop_id, SUM(pay_amt) AS pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261008' AND '20261015' AND order_status != 'cancelled' "
                "  GROUP BY shop_id"
                "), r_s AS ("
                "  SELECT shop_id, SUM(refund_amt) AS refund_amt "
                "  FROM dwd_refund_di "
                "  WHERE ds BETWEEN '20261008' AND '20261015' AND refund_status = 'approved' "
                "  GROUP BY shop_id"
                "), m_s AS ("
                "  SELECT shop_id, SUM(NVL(ad_spend, 0.0)) AS ad_spend "
                "  FROM dws_shop_metrics_1d "
                "  WHERE ds BETWEEN '20261008' AND '20261015' "
                "  GROUP BY shop_id"
                ") "
                "SELECT s.main_cate, "
                "ROUND(SUM(NVL(o.pay_amt, 0.0)), 2) AS total_pay, "
                "ROUND(SUM(NVL(r.refund_amt, 0.0)), 2) AS total_refund, "
                "ROUND(SUM(NVL(m.ad_spend, 0.0)), 2) AS total_ad_spend "
                "FROM dim_shop s "
                "LEFT JOIN o_s o ON s.shop_id = o.shop_id "
                "LEFT JOIN r_s r ON s.shop_id = r.shop_id "
                "LEFT JOIN m_s m ON s.shop_id = m.shop_id "
                "GROUP BY s.main_cate "
                "ORDER BY s.main_cate ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="fd_15",
            category="fanout_dedup",
            split="test",
            question=(
                "对 ods_app_event_log_di 最新分区先按 event_id 去重（取最新 ingest_ts），"
                "再找出每位用户当日时间最早的一条事件（按 ingest_ts 升序、event_id 升序取第 1 条），"
                "统计各页面（page_name）作为用户当日首访页面的用户数（first_visit_users），按 first_visit_users 降序、page_name 升序排列。"
            ),
            hint="需经历两层窗口处理：先按 event_id 去重，再按 user_id 取首访页面。",
            gold_sql=(
                "WITH dedup AS ("
                "  SELECT event_id, user_id, page_name, ingest_ts, "
                "         ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts DESC) AS rn "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                "), user_first AS ("
                "  SELECT user_id, page_name, "
                "         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ingest_ts ASC, event_id ASC) AS urn "
                "  FROM dedup WHERE rn = 1"
                ") "
                "SELECT page_name, COUNT(*) AS first_visit_users "
                "FROM user_first WHERE urn = 1 "
                "GROUP BY page_name "
                "ORDER BY first_visit_users DESC, page_name ASC"
            ),
            ordered=True,
        ),
    ]
