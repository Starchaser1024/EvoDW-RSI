"""Ratio division-by-zero & NULL/empty-string tasks (15 tasks: 5 train, 4 val, 6 test)."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask


def get_ratio_null_handling_tasks() -> list[BenchmarkTask]:
    return [
        # =====================================================================
        # 5. ratio_null_handling (5 train, 4 val, 6 test)
        # =====================================================================
        BenchmarkTask(
            task_id="rn_01",
            category="ratio_null_handling",
            split="train",
            question=(
                "查询 dws_shop_metrics_1d 最新分区，计算每个店铺的访客成交转化率 uv_cvr（buyer_cnt / uv，uv为0时记 0.0，保留4位小数）"
                "以及净 GMV net_gmv（gmv - return_amt，退货额为空时按0计，保留2位小数），输出 shop_id, uv_cvr, net_gmv，按 shop_id 升序排列。"
            ),
            hint="注意冷启动店铺 uv=0 及无退货记录 return_amt 为 NULL 的边界处理。",
            gold_sql=(
                "SELECT shop_id, "
                "ROUND(NVL(1.0 * buyer_cnt / NULLIF(uv, 0), 0.0), 4) AS uv_cvr, "
                "ROUND(gmv - NVL(return_amt, 0.0), 2) AS net_gmv "
                "FROM dws_shop_metrics_1d "
                "WHERE ds = MAX_PT('dws_shop_metrics_1d') "
                "ORDER BY shop_id ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_02",
            category="ratio_null_handling",
            split="train",
            question=(
                "查询 dws_shop_metrics_1d 在 '20261001' 至 '20261015' 分区，按店铺类型（dim_shop.shop_type）统计"
                "总 GMV（total_gmv，保留2位小数）、总广告投放（total_ad_spend，空值按0，保留2位小数）"
                "及整体广告 ROI（ad_roi = total_gmv / total_ad_spend，分母为0记 0.0，保留4位小数），按 shop_type 升序排列。"
            ),
            hint="部分天数 ad_spend 为 NULL 或 0，需使用浮点除法并做好除零保护。",
            gold_sql=(
                "SELECT s.shop_type, "
                "ROUND(SUM(m.gmv), 2) AS total_gmv, "
                "ROUND(SUM(NVL(m.ad_spend, 0.0)), 2) AS total_ad_spend, "
                "ROUND(NVL(1.0 * SUM(m.gmv) / NULLIF(SUM(NVL(m.ad_spend, 0.0)), 0), 0.0), 4) AS ad_roi "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds BETWEEN '20261001' AND '20261015' "
                "GROUP BY s.shop_type "
                "ORDER BY s.shop_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_03",
            category="ratio_null_handling",
            split="train",
            question=(
                "查询 dwd_trade_order_di 最新分区有效订单关联 dim_user，"
                "按城市线级（city_tier，将 NULL 和空字符串 '' 统一归为 'UNKNOWN'）统计实付总额（pay_amt，保留2位小数），"
                "输出 city_tier_norm, pay_amt，按 city_tier_norm 升序排列。"
            ),
            hint="注意 NULL 与空字符串混杂的清洗。",
            gold_sql=(
                "SELECT COALESCE(NULLIF(u.city_tier, ''), 'UNKNOWN') AS city_tier_norm, "
                "ROUND(SUM(o.pay_amt), 2) AS pay_amt "
                "FROM dwd_trade_order_di o "
                "JOIN dim_user u ON o.user_id = u.user_id "
                "WHERE o.ds = MAX_PT('dwd_trade_order_di') "
                "AND o.order_status != 'cancelled' "
                "GROUP BY COALESCE(NULLIF(u.city_tier, ''), 'UNKNOWN') "
                "ORDER BY city_tier_norm ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_04",
            category="ratio_null_handling",
            split="train",
            question=(
                "查询 dwd_trade_order_di 在 '20261010' 至 '20261015' 分区，计算总主订单数（total_orders）、"
                "已取消主订单数（cancelled_orders）、取消率（cancel_rate，保留4位小数）以及"
                "整体优惠占比（discount_rate = 总优惠金额 / 总应付原价金额，优惠空值按0，保留4位小数），返回一行四列。"
            ),
            hint="折扣字段 discount_amt 为 NULL 时表示无优惠；取消单计数按主订单去重。",
            gold_sql=(
                "SELECT COUNT(DISTINCT order_id) AS total_orders, "
                "COUNT(DISTINCT CASE WHEN order_status = 'cancelled' THEN order_id END) AS cancelled_orders, "
                "ROUND(NVL(1.0 * COUNT(DISTINCT CASE WHEN order_status = 'cancelled' THEN order_id END) / NULLIF(COUNT(DISTINCT order_id), 0), 0.0), 4) AS cancel_rate, "
                "ROUND(NVL(1.0 * SUM(NVL(discount_amt, 0.0)) / NULLIF(SUM(gross_amt), 0), 0.0), 4) AS discount_rate "
                "FROM dwd_trade_order_di "
                "WHERE ds BETWEEN '20261010' AND '20261015'"
            ),
            ordered=False,
        ),
        BenchmarkTask(
            task_id="rn_05",
            category="ratio_null_handling",
            split="train",
            question=(
                "查询 dwd_trade_order_di 最新分区有效订单关联 dim_item，按商品品牌（brand，将 NULL 和空字符串 '' 统一规范为 'NO_BRAND'）"
                "统计有效实付总额（total_pay_amt）和总优惠金额（total_discount_amt，空值按0计，均保留2位小数），"
                "输出 brand_norm, total_pay_amt, total_discount_amt，按 brand_norm 升序排列。"
            ),
            hint="兼顾维度字段 NULL/空串清洗与指标字段空值兜底。",
            gold_sql=(
                "SELECT COALESCE(NULLIF(i.brand, ''), 'NO_BRAND') AS brand_norm, "
                "ROUND(SUM(o.pay_amt), 2) AS total_pay_amt, "
                "ROUND(SUM(NVL(o.discount_amt, 0.0)), 2) AS total_discount_amt "
                "FROM dwd_trade_order_di o "
                "JOIN dim_item i ON o.item_id = i.item_id "
                "WHERE o.ds = MAX_PT('dwd_trade_order_di') "
                "AND o.order_status != 'cancelled' "
                "GROUP BY COALESCE(NULLIF(i.brand, ''), 'NO_BRAND') "
                "ORDER BY brand_norm ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_06",
            category="ratio_null_handling",
            split="val",
            question=(
                "查询 dws_shop_metrics_1d 最新分区关联 dim_shop，按主营类目（main_cate）统计"
                "退货率（return_rate = 总退货额 / 总GMV，退货额为空按0，保留4位小数）和"
                "笔单价（aov = 总GMV / 总订单数，保留2位小数），输出 main_cate, return_rate, aov，按 main_cate 升序排列。"
            ),
            hint="分子分母均需考虑空值和除零保护，使用浮点运算。",
            gold_sql=(
                "SELECT s.main_cate, "
                "ROUND(NVL(1.0 * SUM(NVL(m.return_amt, 0.0)) / NULLIF(SUM(m.gmv), 0), 0.0), 4) AS return_rate, "
                "ROUND(NVL(1.0 * SUM(m.gmv) / NULLIF(SUM(m.order_cnt), 0), 0.0), 2) AS aov "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds = MAX_PT('dws_shop_metrics_1d') "
                "GROUP BY s.main_cate "
                "ORDER BY s.main_cate ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_07",
            category="ratio_null_handling",
            split="val",
            question=(
                "查询 dwd_trade_order_di 在 '20261001' 至 '20261015' 分区的有效订单，统计达人带货订单（creator_id 非 NULL 且非空串 ''）"
                "占全部有效订单的主订单数占比（creator_order_share）与实付金额占比（creator_gmv_share），"
                "均保留4位小数，返回一行两列。"
            ),
            hint="注意空字符串 creator_id = '' 不算达人带货订单。",
            gold_sql=(
                "SELECT "
                "ROUND(NVL(1.0 * COUNT(DISTINCT CASE WHEN creator_id IS NOT NULL AND creator_id != '' THEN order_id END) "
                " / NULLIF(COUNT(DISTINCT order_id), 0), 0.0), 4) AS creator_order_share, "
                "ROUND(NVL(1.0 * SUM(CASE WHEN creator_id IS NOT NULL AND creator_id != '' THEN pay_amt ELSE 0.0 END) "
                " / NULLIF(SUM(pay_amt), 0), 0.0), 4) AS creator_gmv_share "
                "FROM dwd_trade_order_di "
                "WHERE ds BETWEEN '20261001' AND '20261015' "
                "AND order_status != 'cancelled'"
            ),
            ordered=False,
        ),
        BenchmarkTask(
            task_id="rn_08",
            category="ratio_null_handling",
            split="val",
            question=(
                "对 ods_app_event_log_di 最新分区按 event_id 取最新 ingest_ts 去重后，"
                "按页面（page_name）统计点击事件数（click_cnt）、曝光事件数（exposure_cnt）以及"
                "点击曝光比（click_to_expo_rate = click_cnt / exposure_cnt，分母为0记 0.0，保留4位小数），按 page_name 升序排列。"
            ),
            hint="先去重再使用条件聚合求点击与曝光比。",
            gold_sql=(
                "WITH dedup AS ("
                "  SELECT event_id, page_name, event_type, "
                "         ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts DESC) AS rn "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                ") "
                "SELECT page_name, "
                "SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS click_cnt, "
                "SUM(CASE WHEN event_type = 'exposure' THEN 1 ELSE 0 END) AS exposure_cnt, "
                "ROUND(NVL(1.0 * SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) "
                " / NULLIF(SUM(CASE WHEN event_type = 'exposure' THEN 1 ELSE 0 END), 0), 0.0), 4) AS click_to_expo_rate "
                "FROM dedup WHERE rn = 1 "
                "GROUP BY page_name "
                "ORDER BY page_name ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_09",
            category="ratio_null_handling",
            split="val",
            question=(
                "查询 dwd_content_post_di 最新分区关联 dim_content_creator，按签约 MCN 机构（mcn_org，将 NULL 和空串 '' 归类为 'INDEPENDENT'）"
                "统计平均赞播比（like_view_rate = SUM(like_cnt) / SUM(view_cnt)，保留4位小数），"
                "输出 mcn_norm, like_view_rate，按 mcn_norm 升序排列。"
            ),
            hint="兼顾 mcn_org 空值/空串清洗与浮点比值除零保护。",
            gold_sql=(
                "SELECT COALESCE(NULLIF(c.mcn_org, ''), 'INDEPENDENT') AS mcn_norm, "
                "ROUND(NVL(1.0 * SUM(p.like_cnt) / NULLIF(SUM(p.view_cnt), 0), 0.0), 4) AS like_view_rate "
                "FROM dwd_content_post_di p "
                "JOIN dim_content_creator c ON p.creator_id = c.creator_id "
                "WHERE p.dt = MAX_PT('dwd_content_post_di') "
                "GROUP BY COALESCE(NULLIF(c.mcn_org, ''), 'INDEPENDENT') "
                "ORDER BY mcn_norm ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_10",
            category="ratio_null_handling",
            split="test",
            question=(
                "查询 dws_shop_metrics_1d 在 '20261010' 至 '20261015' 分区，统计每个店铺有广告投放的天数（ad_days，即 ad_spend > 0 的天数）、"
                "总广告花费（total_ad_spend，空值按0，保留2位小数）和整体浏览访客比（pv_per_uv = SUM(pv) / SUM(uv)，uv和为0记 0.0，保留4位小数），"
                "输出 shop_id, ad_days, total_ad_spend, pv_per_uv，按 shop_id 升序排列。"
            ),
            hint="条件统计有投放天数并做好空值和除零兜底。",
            gold_sql=(
                "SELECT shop_id, "
                "SUM(CASE WHEN NVL(ad_spend, 0.0) > 0 THEN 1 ELSE 0 END) AS ad_days, "
                "ROUND(SUM(NVL(ad_spend, 0.0)), 2) AS total_ad_spend, "
                "ROUND(NVL(1.0 * SUM(pv) / NULLIF(SUM(uv), 0), 0.0), 4) AS pv_per_uv "
                "FROM dws_shop_metrics_1d "
                "WHERE ds BETWEEN '20261010' AND '20261015' "
                "GROUP BY shop_id "
                "ORDER BY shop_id ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_11",
            category="ratio_null_handling",
            split="test",
            question=(
                "查询 dwd_trade_order_di 最新分区有效订单关联 dim_user，"
                "按性别（gender，将 NULL 和空串 '' 统一归为 'UNKNOWN'）统计下单去重用户数（buyer_cnt）、"
                "人均实付额（arpu，保留2位小数）与客单价（aov = 总实付 / 主订单数，保留2位小数），按 gender_norm 升序排列。"
            ),
            hint="主订单数需按 order_id 去重计数，分子使用 pay_amt 求和。",
            gold_sql=(
                "SELECT COALESCE(NULLIF(u.gender, ''), 'UNKNOWN') AS gender_norm, "
                "COUNT(DISTINCT o.user_id) AS buyer_cnt, "
                "ROUND(NVL(1.0 * SUM(o.pay_amt) / NULLIF(COUNT(DISTINCT o.user_id), 0), 0.0), 2) AS arpu, "
                "ROUND(NVL(1.0 * SUM(o.pay_amt) / NULLIF(COUNT(DISTINCT o.order_id), 0), 0.0), 2) AS aov "
                "FROM dwd_trade_order_di o "
                "JOIN dim_user u ON o.user_id = u.user_id "
                "WHERE o.ds = MAX_PT('dwd_trade_order_di') "
                "AND o.order_status != 'cancelled' "
                "GROUP BY COALESCE(NULLIF(u.gender, ''), 'UNKNOWN') "
                "ORDER BY gender_norm ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_12",
            category="ratio_null_handling",
            split="test",
            question=(
                "查询 dwd_refund_di 在 '20261001' 至 '20261015' 分区，"
                "按退款原因（refund_reason）统计审核通过率（approved_rate = 通过退款单数 / 总退款单数，保留4位小数）"
                "及审核通过的退款总金额（approved_amt，空值按0，保留2位小数），按 refund_reason 升序排列。"
            ),
            hint="同一退款单按 refund_id 计数。",
            gold_sql=(
                "SELECT refund_reason, "
                "ROUND(NVL(1.0 * COUNT(DISTINCT CASE WHEN refund_status = 'approved' THEN refund_id END) "
                " / NULLIF(COUNT(DISTINCT refund_id), 0), 0.0), 4) AS approved_rate, "
                "ROUND(SUM(CASE WHEN refund_status = 'approved' THEN refund_amt ELSE 0.0 END), 2) AS approved_amt "
                "FROM dwd_refund_di "
                "WHERE ds BETWEEN '20261001' AND '20261015' "
                "GROUP BY refund_reason "
                "ORDER BY refund_reason ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_13",
            category="ratio_null_handling",
            split="test",
            question=(
                "查询 dwd_trade_order_di 在 '20261012' 至 '20261015' 分区中的有效订单（order_status != 'cancelled'），"
                "计算各店铺的实付折后比 pay_ratio（SUM(pay_amt) / SUM(gross_amt)，分母为0记 0.0，保留4位小数），"
                "返回实付折后比排名前 10 的店铺（shop_id, pay_ratio），按 pay_ratio 降序、shop_id 升序排列。"
            ),
            hint="过滤取消单后再按店铺汇总 pay_amt 与 gross_amt。",
            gold_sql=(
                "SELECT shop_id, "
                "ROUND(NVL(1.0 * SUM(pay_amt) / NULLIF(SUM(gross_amt), 0), 0.0), 4) AS pay_ratio "
                "FROM dwd_trade_order_di "
                "WHERE ds BETWEEN '20261012' AND '20261015' "
                "AND order_status != 'cancelled' "
                "GROUP BY shop_id "
                "ORDER BY pay_ratio DESC, shop_id ASC "
                "LIMIT 10"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_14",
            category="ratio_null_handling",
            split="test",
            question=(
                "查询 dws_shop_metrics_1d 最新分区，将店铺划分为 'NO_AD'（ad_spend IS NULL OR ad_spend = 0）"
                "和 'HAS_AD'（ad_spend > 0）两组，统计每组的店铺数（shop_cnt）、总 GMV（total_gmv，保留2位小数）"
                "和单店平均 GMV（avg_gmv，保留2位小数），输出 ad_flag, shop_cnt, total_gmv, avg_gmv，按 ad_flag 升序排列。"
            ),
            hint="注意 NULL 和 0 均归类为无广告投放组。",
            gold_sql=(
                "SELECT CASE WHEN ad_spend IS NULL OR ad_spend = 0 THEN 'NO_AD' ELSE 'HAS_AD' END AS ad_flag, "
                "COUNT(*) AS shop_cnt, "
                "ROUND(SUM(gmv), 2) AS total_gmv, "
                "ROUND(AVG(gmv), 2) AS avg_gmv "
                "FROM dws_shop_metrics_1d "
                "WHERE ds = MAX_PT('dws_shop_metrics_1d') "
                "GROUP BY CASE WHEN ad_spend IS NULL OR ad_spend = 0 THEN 'NO_AD' ELSE 'HAS_AD' END "
                "ORDER BY ad_flag ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rn_15",
            category="ratio_null_handling",
            split="test",
            question=(
                "查询 dwd_trade_order_di 在 '20261001' 至 '20261015' 分区中的有效订单（order_status != 'cancelled'），"
                "按 ds 统计每日有效实付总额（total_pay_amt，保留2位小数）和无优惠订单（discount_amt IS NULL OR discount_amt = 0）"
                "实付额占比（no_disc_share，保留4位小数），按 ds 升序排列。"
            ),
            hint="无优惠订单指 discount_amt 为 NULL 或 0 的有效订单明细。",
            gold_sql=(
                "SELECT ds, "
                "ROUND(SUM(pay_amt), 2) AS total_pay_amt, "
                "ROUND(NVL(1.0 * SUM(CASE WHEN discount_amt IS NULL OR discount_amt = 0 THEN pay_amt ELSE 0.0 END) "
                " / NULLIF(SUM(pay_amt), 0), 0.0), 4) AS no_disc_share "
                "FROM dwd_trade_order_di "
                "WHERE ds BETWEEN '20261001' AND '20261015' "
                "AND order_status != 'cancelled' "
                "GROUP BY ds "
                "ORDER BY ds ASC"
            ),
            ordered=True,
        ),
    ]
