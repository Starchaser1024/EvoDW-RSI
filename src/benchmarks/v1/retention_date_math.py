"""Cohort retention & cross-month date math tasks (15 tasks: 5 train, 4 val, 6 test)."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask


def get_retention_date_math_tasks() -> list[BenchmarkTask]:
    return [
        # =====================================================================
        # 3. retention_date_math (5 train, 4 val, 6 test)
        # =====================================================================
        BenchmarkTask(
            task_id="rd_01",
            category="retention_date_math",
            split="train",
            question=(
                "计算 2026-09-29 至 2026-10-02 每天注册的新用户数（cohort_users）、"
                "次日（D1）在 ods_app_event_log_di 有活跃记录的留存用户数（d1_users）以及次日留存率（d1_rate，保留4位小数），"
                "按 reg_dt 升序排列。"
            ),
            hint="注意跨越9月30日到10月1日的月份边界，且 dim_user.reg_dt 为 YYYY-MM-DD 格式而埋点 ds 为 YYYYMMDD。",
            gold_sql=(
                "WITH active_days AS ("
                "  SELECT DISTINCT user_id, ds "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20260929' AND '20261003'"
                ") "
                "SELECT u.reg_dt, COUNT(DISTINCT u.user_id) AS cohort_users, "
                "COUNT(DISTINCT a.user_id) AS d1_users, "
                "ROUND(1.0 * COUNT(DISTINCT a.user_id) / NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS d1_rate "
                "FROM dim_user u "
                "LEFT JOIN active_days a "
                "  ON u.user_id = a.user_id "
                "  AND DATEDIFF(TO_DATE(a.ds, 'yyyymmdd'), TO_DATE(u.reg_dt, 'yyyy-mm-dd'), 'dd') = 1 "
                "WHERE u.reg_dt BETWEEN '2026-09-29' AND '2026-10-02' "
                "GROUP BY u.reg_dt "
                "ORDER BY u.reg_dt ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_02",
            category="retention_date_math",
            split="train",
            question=(
                "计算 2026-09-27 至 2026-09-30 每天注册用户的第 7 日（D7）留存情况（第7天在 ods_app_event_log_di 活跃）："
                "输出 reg_dt, cohort_users, d7_users, d7_rate（保留4位小数），按 reg_dt 升序排列。"
            ),
            hint="第 7 日即活跃日期减去注册日期等于 7 天（注意跨月对应 10 月份分区）。",
            gold_sql=(
                "WITH active_days AS ("
                "  SELECT DISTINCT user_id, ds "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20261004' AND '20261007'"
                ") "
                "SELECT u.reg_dt, COUNT(DISTINCT u.user_id) AS cohort_users, "
                "COUNT(DISTINCT a.user_id) AS d7_users, "
                "ROUND(1.0 * COUNT(DISTINCT a.user_id) / NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS d7_rate "
                "FROM dim_user u "
                "LEFT JOIN active_days a "
                "  ON u.user_id = a.user_id "
                "  AND DATEDIFF(TO_DATE(a.ds, 'yyyymmdd'), TO_DATE(u.reg_dt, 'yyyy-mm-dd'), 'dd') = 7 "
                "WHERE u.reg_dt BETWEEN '2026-09-27' AND '2026-09-30' "
                "GROUP BY u.reg_dt "
                "ORDER BY u.reg_dt ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_03",
            category="retention_date_math",
            split="train",
            question=(
                "统计 dwd_trade_order_di 最近 7 天（以最新分区为终点往前推6天至最新分区，含7天）"
                "各拉新渠道（dim_user.channel）的有效订单实付总额（total_pay_amt，保留2位小数），按 channel 升序排列。"
            ),
            hint="基于最新可用分区向前回溯 6 天确定 7 日窗口区间。",
            gold_sql=(
                "SELECT u.channel, ROUND(SUM(o.pay_amt), 2) AS total_pay_amt "
                "FROM dwd_trade_order_di o "
                "JOIN dim_user u ON o.user_id = u.user_id "
                "WHERE o.ds BETWEEN TO_CHAR(DATEADD(TO_DATE(MAX_PT('dwd_trade_order_di'), 'yyyymmdd'), -6, 'dd'), 'yyyymmdd') "
                "AND MAX_PT('dwd_trade_order_di') "
                "AND o.order_status != 'cancelled' "
                "GROUP BY u.channel "
                "ORDER BY u.channel ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_04",
            category="retention_date_math",
            split="train",
            question=(
                "基于 dwd_trade_order_di 在 '20260925' 至 '20261015' 分区中的有效订单，计算每位已成交用户"
                "首次下单分区日期（MIN(ds)）与注册日期（reg_dt）相隔的天数，按会员等级（vip_level）统计"
                "平均首单转化天数（avg_convert_days，保留2位小数），按 vip_level 升序排列。"
            ),
            hint="先按用户求最早有效下单分区日期，再与用户表注册日期做天数差。",
            gold_sql=(
                "WITH first_ord AS ("
                "  SELECT user_id, MIN(ds) AS first_ds "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20260925' AND '20261015' "
                "  AND order_status != 'cancelled' "
                "  GROUP BY user_id"
                ") "
                "SELECT u.vip_level, "
                "ROUND(AVG(DATEDIFF(TO_DATE(f.first_ds, 'yyyymmdd'), TO_DATE(u.reg_dt, 'yyyy-mm-dd'), 'dd')), 2) AS avg_convert_days "
                "FROM first_ord f "
                "JOIN dim_user u ON f.user_id = u.user_id "
                "GROUP BY u.vip_level "
                "ORDER BY u.vip_level ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_05",
            category="retention_date_math",
            split="train",
            question=(
                "基于 dws_shop_metrics_1d 在 '20261009' 至 '20261015' 分区的数据，"
                "统计 '20261010' 至 '20261015' 每天的总 GMV（total_gmv，保留2位小数）、"
                "日环比增长额（dod_diff，保留2位小数）和日环比增长率（dod_rate，保留4位小数），按 ds 升序排列。"
            ),
            hint="先按天汇总每日 GMV，再借助前一天数据或窗口函数求环比。",
            gold_sql=(
                "WITH daily AS ("
                "  SELECT ds, SUM(gmv) AS total_gmv "
                "  FROM dws_shop_metrics_1d "
                "  WHERE ds BETWEEN '20261009' AND '20261015' "
                "  GROUP BY ds"
                "), lagged AS ("
                "  SELECT ds, total_gmv, LAG(total_gmv) OVER (ORDER BY ds) AS prev_gmv "
                "  FROM daily"
                ") "
                "SELECT ds, ROUND(total_gmv, 2) AS total_gmv, "
                "ROUND(total_gmv - prev_gmv, 2) AS dod_diff, "
                "ROUND((total_gmv - prev_gmv) / NULLIF(prev_gmv, 0), 4) AS dod_rate "
                "FROM lagged "
                "WHERE ds BETWEEN '20261010' AND '20261015' "
                "ORDER BY ds ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_06",
            category="retention_date_math",
            split="val",
            question=(
                "计算 2026-09-28 至 2026-10-03 每天注册用户的第 3 日（D3）留存情况（注册后第3天在 ods_app_event_log_di 活跃）："
                "输出 reg_dt, cohort_users, d3_users, d3_rate（保留4位小数），按 reg_dt 升序排列。"
            ),
            hint="注意跨越月末的日期偏移计算。",
            gold_sql=(
                "WITH active_days AS ("
                "  SELECT DISTINCT user_id, ds "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20261001' AND '20261006'"
                ") "
                "SELECT u.reg_dt, COUNT(DISTINCT u.user_id) AS cohort_users, "
                "COUNT(DISTINCT a.user_id) AS d3_users, "
                "ROUND(1.0 * COUNT(DISTINCT a.user_id) / NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS d3_rate "
                "FROM dim_user u "
                "LEFT JOIN active_days a "
                "  ON u.user_id = a.user_id "
                "  AND DATEDIFF(TO_DATE(a.ds, 'yyyymmdd'), TO_DATE(u.reg_dt, 'yyyy-mm-dd'), 'dd') = 3 "
                "WHERE u.reg_dt BETWEEN '2026-09-28' AND '2026-10-03' "
                "GROUP BY u.reg_dt "
                "ORDER BY u.reg_dt ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_07",
            category="retention_date_math",
            split="val",
            question=(
                "对比 dws_shop_metrics_1d 最近一周（'20261009' 至 '20261015'）与上一周（'20261002' 至 '20261008'），"
                "按 dim_shop.main_cate 统计本期 GMV（curr_gmv，保留2位小数）、上期 GMV（prev_gmv，保留2位小数）"
                "和周环比增长率（wow_rate，保留4位小数），按 main_cate 升序排列。"
            ),
            hint="使用单次扫描条件聚合计算前后两周 GMV 及环比。",
            gold_sql=(
                "SELECT s.main_cate, "
                "ROUND(SUM(CASE WHEN m.ds BETWEEN '20261009' AND '20261015' THEN m.gmv ELSE 0.0 END), 2) AS curr_gmv, "
                "ROUND(SUM(CASE WHEN m.ds BETWEEN '20261002' AND '20261008' THEN m.gmv ELSE 0.0 END), 2) AS prev_gmv, "
                "ROUND("
                "  (SUM(CASE WHEN m.ds BETWEEN '20261009' AND '20261015' THEN m.gmv ELSE 0.0 END) "
                "   - SUM(CASE WHEN m.ds BETWEEN '20261002' AND '20261008' THEN m.gmv ELSE 0.0 END)) "
                "  / NULLIF(SUM(CASE WHEN m.ds BETWEEN '20261002' AND '20261008' THEN m.gmv ELSE 0.0 END), 0), 4"
                ") AS wow_rate "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds BETWEEN '20261002' AND '20261015' "
                "GROUP BY s.main_cate "
                "ORDER BY s.main_cate ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_08",
            category="retention_date_math",
            split="val",
            question=(
                "统计在 dwd_trade_order_di 2026年10月1日（'20261001'）产生过有效订单的用户中，"
                "在其后 7 天（'20261002' 至 '20261008'）再次产生有效订单的复购用户数与复购率，"
                "返回一行三列：base_buyers, repeat_buyers, repeat_rate（保留4位小数）。"
            ),
            hint="先分别提取基准日有效买家集合与后续7天有效买家集合再关联。",
            gold_sql=(
                "WITH base_u AS ("
                "  SELECT DISTINCT user_id FROM dwd_trade_order_di "
                "  WHERE ds = '20261001' AND order_status != 'cancelled'"
                "), rep_u AS ("
                "  SELECT DISTINCT user_id FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261002' AND '20261008' AND order_status != 'cancelled'"
                ") "
                "SELECT COUNT(DISTINCT b.user_id) AS base_buyers, "
                "COUNT(DISTINCT r.user_id) AS repeat_buyers, "
                "ROUND(1.0 * COUNT(DISTINCT r.user_id) / NULLIF(COUNT(DISTINCT b.user_id), 0), 4) AS repeat_rate "
                "FROM base_u b "
                "LEFT JOIN rep_u r ON b.user_id = r.user_id"
            ),
            ordered=False,
        ),
        BenchmarkTask(
            task_id="rd_09",
            category="retention_date_math",
            split="val",
            question=(
                "查询 dwd_trade_order_di 在 '20261001' 至 '20261015' 分区中，用户下单日期（ds）距离注册日期（dim_user.reg_dt）"
                "在 0 至 7 天内（含首尾）的新客有效订单：按 ds 统计每日新客主订单数（new_order_cnt）与实付总额（new_pay_amt，保留2位小数），"
                "按 ds 升序排列。"
            ),
            hint="将交易日期 ds 与用户 reg_dt 转为日期后计算天数差在 0 到 7 之间。",
            gold_sql=(
                "SELECT o.ds, COUNT(DISTINCT o.order_id) AS new_order_cnt, "
                "ROUND(SUM(o.pay_amt), 2) AS new_pay_amt "
                "FROM dwd_trade_order_di o "
                "JOIN dim_user u ON o.user_id = u.user_id "
                "WHERE o.ds BETWEEN '20261001' AND '20261015' "
                "AND o.order_status != 'cancelled' "
                "AND DATEDIFF(TO_DATE(o.ds, 'yyyymmdd'), TO_DATE(u.reg_dt, 'yyyy-mm-dd'), 'dd') BETWEEN 0 AND 7 "
                "GROUP BY o.ds "
                "ORDER BY o.ds ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_10",
            category="retention_date_math",
            split="test",
            question=(
                "汇总统计 2026-09-26 至 2026-10-01 期间注册的所有用户，按拉新渠道（channel）统计"
                "总体注册人数（cohort_users）、次日（D1）在 ods_app_event_log_di 活跃的留存人数（d1_users）"
                "及次日留存率（d1_rate，保留4位小数），按 channel 升序排列。"
            ),
            hint="注意 2026-09-30 的次日为 20261001。",
            gold_sql=(
                "WITH active_days AS ("
                "  SELECT DISTINCT user_id, ds "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20260927' AND '20261002'"
                ") "
                "SELECT u.channel, COUNT(DISTINCT u.user_id) AS cohort_users, "
                "COUNT(DISTINCT a.user_id) AS d1_users, "
                "ROUND(1.0 * COUNT(DISTINCT a.user_id) / NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS d1_rate "
                "FROM dim_user u "
                "LEFT JOIN active_days a "
                "  ON u.user_id = a.user_id "
                "  AND DATEDIFF(TO_DATE(a.ds, 'yyyymmdd'), TO_DATE(u.reg_dt, 'yyyy-mm-dd'), 'dd') = 1 "
                "WHERE u.reg_dt BETWEEN '2026-09-26' AND '2026-10-01' "
                "GROUP BY u.channel "
                "ORDER BY u.channel ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_11",
            category="retention_date_math",
            split="test",
            question=(
                "统计开店时间（dim_shop.open_dt）距离 dwd_trade_order_di 最新分区超过 365 天的老店，"
                "在最新分区的有效成交表现：输出一行两列 old_shop_cnt（有有效成交的老店数）和 total_pay_amt（保留2位小数）。"
            ),
            hint="使用最新交易分区日期减去店铺 open_dt 判断天数大于 365。",
            gold_sql=(
                "SELECT COUNT(DISTINCT o.shop_id) AS old_shop_cnt, "
                "ROUND(SUM(o.pay_amt), 2) AS total_pay_amt "
                "FROM dwd_trade_order_di o "
                "JOIN dim_shop s ON o.shop_id = s.shop_id "
                "WHERE o.ds = MAX_PT('dwd_trade_order_di') "
                "AND o.order_status != 'cancelled' "
                "AND DATEDIFF(TO_DATE(o.ds, 'yyyymmdd'), TO_DATE(s.open_dt, 'yyyy-mm-dd'), 'dd') > 365"
            ),
            ordered=False,
        ),
        BenchmarkTask(
            task_id="rd_12",
            category="retention_date_math",
            split="test",
            question=(
                "查询 '20261001' 至 '20261015' 期间的订单与退款数据，统计退款发生在下单后 0 至 3 天内的情况："
                "按退款分区日期减去下单分区日期的天数差 lag_days（0, 1, 2, 3）统计去重退款订单数（refund_order_cnt），"
                "按 lag_days 升序排列。"
            ),
            hint="为避免多行子订单造成重复计算，先提取去重后的 (order_id, order_ds) 与 (order_id, refund_ds)。",
            gold_sql=(
                "WITH ord AS ("
                "  SELECT DISTINCT order_id, ds AS order_ds "
                "  FROM dwd_trade_order_di "
                "  WHERE ds BETWEEN '20261001' AND '20261015' AND order_status != 'cancelled'"
                "), ref AS ("
                "  SELECT DISTINCT order_id, ds AS refund_ds "
                "  FROM dwd_refund_di "
                "  WHERE ds BETWEEN '20261001' AND '20261015'"
                ") "
                "SELECT DATEDIFF(TO_DATE(r.refund_ds, 'yyyymmdd'), TO_DATE(o.order_ds, 'yyyymmdd'), 'dd') AS lag_days, "
                "COUNT(DISTINCT r.order_id) AS refund_order_cnt "
                "FROM ord o "
                "JOIN ref r ON o.order_id = r.order_id "
                "WHERE DATEDIFF(TO_DATE(r.refund_ds, 'yyyymmdd'), TO_DATE(o.order_ds, 'yyyymmdd'), 'dd') BETWEEN 0 AND 3 "
                "GROUP BY DATEDIFF(TO_DATE(r.refund_ds, 'yyyymmdd'), TO_DATE(o.order_ds, 'yyyymmdd'), 'dd') "
                "ORDER BY lag_days ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_13",
            category="retention_date_math",
            split="test",
            question=(
                "对比 dws_shop_metrics_1d 最新分区与前第 7 天分区（即最新分区减 7 天）各店铺类型（shop_type）的总 GMV，"
                "输出 shop_type, latest_gmv, prev7_gmv（保留2位小数）和周同比变化率 wow_rate（保留4位小数），按 shop_type 升序排列。"
            ),
            hint="基于最新分区往前减 7 天得到对比分区。",
            gold_sql=(
                "SELECT s.shop_type, "
                "ROUND(SUM(CASE WHEN m.ds = MAX_PT('dws_shop_metrics_1d') THEN m.gmv ELSE 0.0 END), 2) AS latest_gmv, "
                "ROUND(SUM(CASE WHEN m.ds = TO_CHAR(DATEADD(TO_DATE(MAX_PT('dws_shop_metrics_1d'), 'yyyymmdd'), -7, 'dd'), 'yyyymmdd') THEN m.gmv ELSE 0.0 END), 2) AS prev7_gmv, "
                "ROUND("
                "  (SUM(CASE WHEN m.ds = MAX_PT('dws_shop_metrics_1d') THEN m.gmv ELSE 0.0 END) "
                "   - SUM(CASE WHEN m.ds = TO_CHAR(DATEADD(TO_DATE(MAX_PT('dws_shop_metrics_1d'), 'yyyymmdd'), -7, 'dd'), 'yyyymmdd') THEN m.gmv ELSE 0.0 END)) "
                "  / NULLIF(SUM(CASE WHEN m.ds = TO_CHAR(DATEADD(TO_DATE(MAX_PT('dws_shop_metrics_1d'), 'yyyymmdd'), -7, 'dd'), 'yyyymmdd') THEN m.gmv ELSE 0.0 END), 0), 4"
                ") AS wow_rate "
                "FROM dws_shop_metrics_1d m "
                "JOIN dim_shop s ON m.shop_id = s.shop_id "
                "WHERE m.ds IN ("
                "  MAX_PT('dws_shop_metrics_1d'), "
                "  TO_CHAR(DATEADD(TO_DATE(MAX_PT('dws_shop_metrics_1d'), 'yyyymmdd'), -7, 'dd'), 'yyyymmdd')"
                ") "
                "GROUP BY s.shop_type "
                "ORDER BY s.shop_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_14",
            category="retention_date_math",
            split="test",
            question=(
                "统计 2026-09-30 注册且在当天或次日（'20260930' 或 '20261001'）在 ods_app_event_log_di 活跃过的用户中，"
                "注册第 7 天（'20261007'）仍在 ods_app_event_log_di 活跃的用户数与占比，"
                "按 vip_level 输出 vip_level, base_users, d7_users, d7_rate（保留4位小数），按 vip_level 升序排列。"
            ),
            hint="分母是 9月30日注册且在 0930/1001 活跃的种子用户，分子是其中在 1007 活跃的人数。",
            gold_sql=(
                "WITH seed_active AS ("
                "  SELECT DISTINCT user_id FROM ods_app_event_log_di WHERE ds IN ('20260930', '20261001')"
                "), d7_active AS ("
                "  SELECT DISTINCT user_id FROM ods_app_event_log_di WHERE ds = '20261007'"
                ") "
                "SELECT u.vip_level, COUNT(DISTINCT u.user_id) AS base_users, "
                "COUNT(DISTINCT d.user_id) AS d7_users, "
                "ROUND(1.0 * COUNT(DISTINCT d.user_id) / NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS d7_rate "
                "FROM dim_user u "
                "JOIN seed_active s ON u.user_id = s.user_id "
                "LEFT JOIN d7_active d ON u.user_id = d.user_id "
                "WHERE u.reg_dt = '2026-09-30' "
                "GROUP BY u.vip_level "
                "ORDER BY u.vip_level ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="rd_15",
            category="retention_date_math",
            split="test",
            question=(
                "查询 dwd_content_post_di 最近 7 天分区（最大分区往前推6天至最大分区）每天的总浏览量（daily_views），"
                "以及当天浏览量与这 7 天日均浏览量的差值（diff_from_7d_avg，保留2位小数），按 dt 升序排列。"
            ),
            hint="先按天聚合浏览量，再用窗口函数计算 7 天均值差。",
            gold_sql=(
                "WITH daily AS ("
                "  SELECT dt, SUM(view_cnt) AS daily_views "
                "  FROM dwd_content_post_di "
                "  WHERE dt BETWEEN TO_CHAR(DATEADD(TO_DATE(MAX_PT('dwd_content_post_di'), 'yyyymmdd'), -6, 'dd'), 'yyyymmdd') "
                "  AND MAX_PT('dwd_content_post_di') "
                "  GROUP BY dt"
                ") "
                "SELECT dt, daily_views, "
                "ROUND(daily_views - AVG(daily_views) OVER (), 2) AS diff_from_7d_avg "
                "FROM daily "
                "ORDER BY dt ASC"
            ),
            ordered=True,
        ),
    ]
