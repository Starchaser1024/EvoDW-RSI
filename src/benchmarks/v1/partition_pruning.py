"""Partition pruning benchmark tasks (15 tasks: 5 train, 4 val, 6 test)."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask


def get_partition_pruning_tasks() -> list[BenchmarkTask]:
    return [
        # =====================================================================
        # 1. partition_pruning (5 train, 4 val, 6 test)
        # =====================================================================
        BenchmarkTask(
            task_id="pp_01",
            category="partition_pruning",
            split="train",
            question=(
                "查询 dwd_trade_order_di 最新分区中各订单状态（order_status）的子订单明细行数（order_item_cnt）"
                "和总实付金额（total_pay_amt，保留2位小数），按 order_status 升序排列。"
            ),
            hint="取明细表最新可用分区统计各状态明细行数及实付金额之和。",
            gold_sql=(
                "SELECT order_status, COUNT(*) AS order_item_cnt, "
                "ROUND(SUM(pay_amt), 2) AS total_pay_amt "
                "FROM dwd_trade_order_di "
                "WHERE ds = MAX_PT('dwd_trade_order_di') "
                "GROUP BY order_status "
                "ORDER BY order_status ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_02",
            category="partition_pruning",
            split="train",
            question=(
                "查询 dws_shop_metrics_1d 最新分区中，结合 dim_shop 统计各店铺类型（shop_type）"
                "的总访客数（total_uv）和总 GMV（total_gmv，保留2位小数），按 shop_type 升序排列。"
            ),
            hint="店铺日汇总表按最新分区过滤后关联店铺维表。",
            gold_sql=(
                "SELECT s.shop_type, SUM(m.uv) AS total_uv, ROUND(SUM(m.gmv), 2) AS total_gmv "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds = MAX_PT('dws_shop_metrics_1d') "
                "GROUP BY s.shop_type "
                "ORDER BY s.shop_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_03",
            category="partition_pruning",
            split="train",
            question=(
                "查询内容帖子明细表 dwd_content_post_di 最新分区中各内容形式（content_type）的帖子数（post_cnt）"
                "与平均浏览量（avg_views，保留2位小数），按 content_type 升序排列。"
            ),
            hint="注意检查内容帖子明细表的分区字段名称与最新分区取值。",
            gold_sql=(
                "SELECT content_type, COUNT(DISTINCT post_id) AS post_cnt, "
                "ROUND(AVG(view_cnt), 2) AS avg_views "
                "FROM dwd_content_post_di "
                "WHERE dt = MAX_PT('dwd_content_post_di') "
                "GROUP BY content_type "
                "ORDER BY content_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_04",
            category="partition_pruning",
            split="train",
            question=(
                "查询 2026年9月28日至 2026年10月3日（含首尾）期间 dwd_trade_order_di 每天的"
                "有效主订单数（order_cnt）与实付总额（total_pay_amt，保留2位小数），按分区日期 ds 升序排列。"
            ),
            hint="有效订单指 order_status != 'cancelled'；同一 order_id 可能有多条明细，按主订单去重计数。",
            gold_sql=(
                "SELECT ds, COUNT(DISTINCT order_id) AS order_cnt, "
                "ROUND(SUM(pay_amt), 2) AS total_pay_amt "
                "FROM dwd_trade_order_di "
                "WHERE ds BETWEEN '20260928' AND '20261003' "
                "AND order_status != 'cancelled' "
                "GROUP BY ds "
                "ORDER BY ds ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_05",
            category="partition_pruning",
            split="train",
            question=(
                "统计 dwd_trade_order_di 最新分区各店铺的有效实付总额，并对齐关联 dwd_refund_di 最新分区"
                "审核通过（refund_status = 'approved'）的退款总额（无退款记 0.0）。"
                "仅输出有有效成交的店铺，返回 shop_id, valid_pay_amt, approved_refund_amt（均保留2位小数），按 shop_id 升序。"
            ),
            hint="两张分区表均需各自限定最新分区并先按 shop_id 聚合再关联。",
            gold_sql=(
                "WITH ord AS ("
                "  SELECT shop_id, ROUND(SUM(pay_amt), 2) AS valid_pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds = MAX_PT('dwd_trade_order_di') AND order_status != 'cancelled' "
                "  GROUP BY shop_id"
                "), ref AS ("
                "  SELECT shop_id, ROUND(SUM(refund_amt), 2) AS approved_refund_amt "
                "  FROM dwd_refund_di "
                "  WHERE ds = MAX_PT('dwd_refund_di') AND refund_status = 'approved' "
                "  GROUP BY shop_id"
                ") "
                "SELECT o.shop_id, o.valid_pay_amt, NVL(r.approved_refund_amt, 0.0) AS approved_refund_amt "
                "FROM ord o "
                "LEFT JOIN ref r ON o.shop_id = r.shop_id "
                "ORDER BY o.shop_id ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_06",
            category="partition_pruning",
            split="val",
            question=(
                "查询 dws_shop_metrics_1d 最新分区中 GMV 排名前 5 的店铺："
                "输出 shop_id, shop_name, main_cate, gmv（保留2位小数），按 gmv 降序、shop_id 升序排列。"
            ),
            hint="结合店铺维表输出名称和主营类目，使用最新分区数据。",
            gold_sql=(
                "SELECT m.shop_id, s.shop_name, s.main_cate, ROUND(m.gmv, 2) AS gmv "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds = MAX_PT('dws_shop_metrics_1d') "
                "ORDER BY gmv DESC, m.shop_id ASC "
                "LIMIT 5"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_07",
            category="partition_pruning",
            split="val",
            question=(
                "查询 dwd_refund_di 最新分区中各退款原因（refund_reason）的退款单数（refund_cnt）"
                "和退款总金额（total_refund_amt，保留2位小数），按 total_refund_amt 降序、refund_reason 升序排列。"
            ),
            hint="统计最新分区中所有退款单记录。",
            gold_sql=(
                "SELECT refund_reason, COUNT(DISTINCT refund_id) AS refund_cnt, "
                "ROUND(SUM(refund_amt), 2) AS total_refund_amt "
                "FROM dwd_refund_di "
                "WHERE ds = MAX_PT('dwd_refund_di') "
                "GROUP BY refund_reason "
                "ORDER BY total_refund_amt DESC, refund_reason ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_08",
            category="partition_pruning",
            split="val",
            question=(
                "对比 dws_shop_metrics_1d 最新分区与前一天分区所有店铺的总 GMV，"
                "返回一行两列：latest_gmv, prev_gmv（均保留2位小数）。"
            ),
            hint="基于最新分区往前推 1 天得到前日分区日期。",
            gold_sql=(
                "SELECT "
                "ROUND(SUM(CASE WHEN ds = MAX_PT('dws_shop_metrics_1d') THEN gmv ELSE 0.0 END), 2) AS latest_gmv, "
                "ROUND(SUM(CASE WHEN ds = TO_CHAR(DATEADD(TO_DATE(MAX_PT('dws_shop_metrics_1d'), 'yyyymmdd'), -1, 'dd'), 'yyyymmdd') THEN gmv ELSE 0.0 END), 2) AS prev_gmv "
                "FROM dws_shop_metrics_1d "
                "WHERE ds IN ("
                "  MAX_PT('dws_shop_metrics_1d'), "
                "  TO_CHAR(DATEADD(TO_DATE(MAX_PT('dws_shop_metrics_1d'), 'yyyymmdd'), -1, 'dd'), 'yyyymmdd')"
                ")"
            ),
            ordered=False,
        ),
        BenchmarkTask(
            task_id="pp_09",
            category="partition_pruning",
            split="val",
            question=(
                "查询 ods_app_event_log_di 最新分区中（直接统计原始日志记录，无需去重）各事件类型（event_type）的"
                "原始日志条数（raw_log_cnt）与去重活跃用户数（uv），按 event_type 升序排列。"
            ),
            hint="对埋点表限定最新分区后按 event_type 分组统计。",
            gold_sql=(
                "SELECT event_type, COUNT(*) AS raw_log_cnt, COUNT(DISTINCT user_id) AS uv "
                "FROM ods_app_event_log_di "
                "WHERE ds = MAX_PT('ods_app_event_log_di') "
                "GROUP BY event_type "
                "ORDER BY event_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_10",
            category="partition_pruning",
            split="test",
            question=(
                "查询 dwd_trade_order_di 最新分区中非黑名单用户（dim_user.is_blacklist = 0）的有效订单，"
                "按会员等级（vip_level）统计实付总额（total_pay_amt，保留2位小数）和去重下单用户数（buyer_cnt），"
                "按 vip_level 升序排列。"
            ),
            hint="过滤掉已取消订单（order_status != 'cancelled'）及黑名单用户。",
            gold_sql=(
                "SELECT u.vip_level, ROUND(SUM(o.pay_amt), 2) AS total_pay_amt, "
                "COUNT(DISTINCT o.user_id) AS buyer_cnt "
                "FROM dwd_trade_order_di o "
                "JOIN dim_user u ON o.user_id = u.user_id "
                "WHERE o.ds = MAX_PT('dwd_trade_order_di') "
                "AND o.order_status != 'cancelled' "
                "AND u.is_blacklist = 0 "
                "GROUP BY u.vip_level "
                "ORDER BY u.vip_level ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_11",
            category="partition_pruning",
            split="test",
            question=(
                "查询 dwd_content_post_di 最新分区关联 dim_content_creator，"
                "按达人层级（creator_tier）统计帖子数（post_cnt）和总点赞数（total_likes），按 creator_tier 升序排列。"
            ),
            hint="内容帖子明细表按其自身分区列取最新分区后关联达人维表。",
            gold_sql=(
                "SELECT c.creator_tier, COUNT(DISTINCT p.post_id) AS post_cnt, SUM(p.like_cnt) AS total_likes "
                "FROM dwd_content_post_di p "
                "JOIN dim_content_creator c ON p.creator_id = c.creator_id "
                "WHERE p.dt = MAX_PT('dwd_content_post_di') "
                "GROUP BY c.creator_tier "
                "ORDER BY c.creator_tier ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_12",
            category="partition_pruning",
            split="test",
            question=(
                "统计 2026年9月29日至 2026年10月5日（含首尾）期间 dws_shop_metrics_1d 中"
                "各主营类目（dim_shop.main_cate）的总有效订单数（total_orders）和总 GMV（total_gmv，保留2位小数），"
                "按 total_gmv 降序、main_cate 升序排列。"
            ),
            hint="注意跨越9月末与10月初的分区区间写法。",
            gold_sql=(
                "SELECT s.main_cate, SUM(m.order_cnt) AS total_orders, ROUND(SUM(m.gmv), 2) AS total_gmv "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds BETWEEN '20260929' AND '20261005' "
                "GROUP BY s.main_cate "
                "ORDER BY total_gmv DESC, s.main_cate ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_13",
            category="partition_pruning",
            split="test",
            question=(
                "结合 dws_shop_metrics_1d 最新分区和 dwd_trade_order_di 最新分区，"
                "按 dim_shop.main_cate 统计汇总表总访客数（total_uv）和订单明细表有效实付总额（total_pay_amt，保留2位小数），"
                "按 main_cate 升序排列。"
            ),
            hint="先各自按 shop_id 对最新分区汇总再与 dim_shop 关联，避免直接明细相乘膨胀。",
            gold_sql=(
                "WITH m_agg AS ("
                "  SELECT shop_id, SUM(uv) AS uv "
                "  FROM dws_shop_metrics_1d "
                "  WHERE ds = MAX_PT('dws_shop_metrics_1d') "
                "  GROUP BY shop_id"
                "), o_agg AS ("
                "  SELECT shop_id, SUM(pay_amt) AS pay_amt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds = MAX_PT('dwd_trade_order_di') AND order_status != 'cancelled' "
                "  GROUP BY shop_id"
                ") "
                "SELECT s.main_cate, SUM(NVL(m.uv, 0)) AS total_uv, "
                "ROUND(SUM(NVL(o.pay_amt, 0.0)), 2) AS total_pay_amt "
                "FROM dim_shop s "
                "LEFT JOIN m_agg m ON s.shop_id = m.shop_id "
                "LEFT JOIN o_agg o ON s.shop_id = o.shop_id "
                "GROUP BY s.main_cate "
                "ORDER BY s.main_cate ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_14",
            category="partition_pruning",
            split="test",
            question=(
                "查询 dwd_refund_di 最新 3 天分区（最大分区前2天至最大分区，含首尾3天）内"
                "每天各退款状态（refund_status）的退款总金额，输出 ds, refund_status, refund_amt（保留2位小数），"
                "按 ds 升序、refund_status 升序排列。"
            ),
            hint="使用最新分区往前推 2 天作为下界进行分区范围裁剪。",
            gold_sql=(
                "SELECT ds, refund_status, ROUND(SUM(refund_amt), 2) AS refund_amt "
                "FROM dwd_refund_di "
                "WHERE ds BETWEEN TO_CHAR(DATEADD(TO_DATE(MAX_PT('dwd_refund_di'), 'yyyymmdd'), -2, 'dd'), 'yyyymmdd') "
                "AND MAX_PT('dwd_refund_di') "
                "GROUP BY ds, refund_status "
                "ORDER BY ds ASC, refund_status ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="pp_15",
            category="partition_pruning",
            split="test",
            question=(
                "在最新一天分区中，分别统计 dwd_trade_order_di 有有效带货成交（creator_id 非 NULL 且非空串，order_status != 'cancelled'）"
                "的去重达人数（order_creator_cnt）以及 dwd_content_post_di 有发帖的去重达人数（post_creator_cnt），返回一行两列。"
            ),
            hint="两张表均按各自的最新分区过滤并独立计数后合并输出。",
            gold_sql=(
                "WITH ord_c AS ("
                "  SELECT COUNT(DISTINCT creator_id) AS order_creator_cnt "
                "  FROM dwd_trade_order_di "
                "  WHERE ds = MAX_PT('dwd_trade_order_di') "
                "    AND order_status != 'cancelled' "
                "    AND creator_id IS NOT NULL AND creator_id != ''"
                "), post_c AS ("
                "  SELECT COUNT(DISTINCT creator_id) AS post_creator_cnt "
                "  FROM dwd_content_post_di "
                "  WHERE dt = MAX_PT('dwd_content_post_di')"
                ") "
                "SELECT o.order_creator_cnt, p.post_creator_cnt "
                "FROM ord_c o JOIN post_c p ON o.order_creator_cnt >= 0 AND p.post_creator_cnt >= 0"
            ),
            ordered=False,
        ),
    ]
