"""JSON & LATERAL VIEW explode benchmark tasks (15 tasks: 5 train, 4 val, 6 test)."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask


def get_json_array_unnest_tasks() -> list[BenchmarkTask]:
    return [
        # =====================================================================
        # 2. json_array_unnest (5 train, 4 val, 6 test)
        # =====================================================================
        BenchmarkTask(
            task_id="ja_01",
            category="json_array_unnest",
            split="train",
            question=(
                "查询 ods_app_event_log_di 最新分区中解析 ext_json 的实验分组（$.ab_group，过滤 NULL 和空串），"
                "统计各分组的事件行数（event_cnt），输出 ab_group, event_cnt，按 ab_group 升序排列。"
            ),
            hint="注意 ext_json 中可能混有 NULL、空串或异常格式文本，需确保解析结果有效非空。",
            gold_sql=(
                "SELECT ab_group, COUNT(*) AS event_cnt "
                "FROM ("
                "  SELECT GET_JSON_OBJECT(ext_json, '$.ab_group') AS ab_group "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                ") t "
                "WHERE ab_group IS NOT NULL AND ab_group != '' "
                "GROUP BY ab_group "
                "ORDER BY ab_group ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_02",
            category="json_array_unnest",
            split="train",
            question=(
                "查询 ods_app_event_log_di 最新分区中解析 ext_json 里的嵌套设备系统字段（$.device.os，过滤 NULL 和空串）"
                "和停留时长（$.stay_ms），统计各操作系统的事件数（event_cnt）与平均停留毫秒数（avg_stay_ms，保留2位小数），"
                "按 event_cnt 降序、device_os 升序排列。"
            ),
            hint="stay_ms 需转为数值类型后再求平均值。",
            gold_sql=(
                "SELECT device_os, COUNT(*) AS event_cnt, "
                "ROUND(AVG(CAST(stay_ms AS DOUBLE)), 2) AS avg_stay_ms "
                "FROM ("
                "  SELECT GET_JSON_OBJECT(ext_json, '$.device.os') AS device_os, "
                "         GET_JSON_OBJECT(ext_json, '$.stay_ms') AS stay_ms "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                ") t "
                "WHERE device_os IS NOT NULL AND device_os != '' "
                "GROUP BY device_os "
                "ORDER BY event_cnt DESC, device_os ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_03",
            category="json_array_unnest",
            split="train",
            question=(
                "将 ods_app_event_log_di 最新分区的曝光商品数组 expo_item_ids 行转列展开，"
                "统计各被曝光商品（item_id）的曝光次数（expo_cnt），返回前 10 名，按 expo_cnt 降序、item_id 升序排列。"
            ),
            hint="使用数组行转列语法展开 expo_item_ids 后聚合计数。",
            gold_sql=(
                "SELECT t.item_id, COUNT(*) AS expo_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW explode(e.expo_item_ids) t AS item_id "
                "WHERE e.ds = MAX_PT('ods_app_event_log_di') "
                "GROUP BY t.item_id "
                "ORDER BY expo_cnt DESC, t.item_id ASC "
                "LIMIT 10"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_04",
            category="json_array_unnest",
            split="train",
            question=(
                "将 dwd_content_post_di 最新分区的话题标签数组 topic_tags 展开，"
                "统计每个话题标签（tag）关联的帖子数（post_cnt）和总浏览量（total_views），"
                "按 post_cnt 降序、tag 升序排列。"
            ),
            hint="注意 dwd_content_post_di 的分区键，展开 topic_tags 后按标签聚合。",
            gold_sql=(
                "SELECT t.tag, COUNT(DISTINCT p.post_id) AS post_cnt, SUM(p.view_cnt) AS total_views "
                "FROM dwd_content_post_di p "
                "LATERAL VIEW explode(p.topic_tags) t AS tag "
                "WHERE p.dt = MAX_PT('dwd_content_post_di') "
                "GROUP BY t.tag "
                "ORDER BY post_cnt DESC, t.tag ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_05",
            category="json_array_unnest",
            split="train",
            question=(
                "查询 ods_app_event_log_di 最新分区，对曝光商品数组 expo_item_ids 做外展开（保留空数组事件行），"
                "按 event_type 统计展开后的总明细行数（total_rows）以及非空曝光商品计数（non_null_item_cnt），"
                "按 event_type 升序排列。"
            ),
            hint="若使用普通内展开会丢失空数组行，需保留空数组事件后分别统计总行数与非空商品数。",
            gold_sql=(
                "SELECT e.event_type, COUNT(*) AS total_rows, COUNT(t.item_id) AS non_null_item_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW OUTER explode(e.expo_item_ids) t AS item_id "
                "WHERE e.ds = MAX_PT('ods_app_event_log_di') "
                "GROUP BY e.event_type "
                "ORDER BY e.event_type ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_06",
            category="json_array_unnest",
            split="val",
            question=(
                "将 ods_app_event_log_di 最新分区的 expo_item_ids 展开并关联 dim_item，"
                "统计各商品一级类目（cate_l1）的曝光总次数（expo_cnt），按 expo_cnt 降序、cate_l1 升序排列。"
            ),
            hint="展开出 item_id 后关联商品维表。",
            gold_sql=(
                "SELECT i.cate_l1, COUNT(*) AS expo_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW explode(e.expo_item_ids) t AS item_id "
                "JOIN dim_item i ON t.item_id = i.item_id "
                "WHERE e.ds = MAX_PT('ods_app_event_log_di') "
                "GROUP BY i.cate_l1 "
                "ORDER BY expo_cnt DESC, i.cate_l1 ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_07",
            category="json_array_unnest",
            split="val",
            question=(
                "查询 dwd_content_post_di 最新分区中从 meta_json 提取 $.is_promoted 和 $.duration_sec，"
                "过滤 is_promoted 为 NULL 的记录，统计各 is_promoted 值的帖子数（post_cnt）和平均内容时长（avg_duration_sec，保留2位小数），"
                "按 is_promoted 升序排列。"
            ),
            hint="注意 meta_json 可能为 NULL，需先提取并过滤掉空值。",
            gold_sql=(
                "SELECT is_promoted, COUNT(*) AS post_cnt, "
                "ROUND(AVG(CAST(duration_sec AS DOUBLE)), 2) AS avg_duration_sec "
                "FROM ("
                "  SELECT GET_JSON_OBJECT(meta_json, '$.is_promoted') AS is_promoted, "
                "         GET_JSON_OBJECT(meta_json, '$.duration_sec') AS duration_sec "
                "  FROM dwd_content_post_di "
                "  WHERE dt = MAX_PT('dwd_content_post_di')"
                ") t "
                "WHERE is_promoted IS NOT NULL AND is_promoted != '' "
                "GROUP BY is_promoted "
                "ORDER BY is_promoted ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_08",
            category="json_array_unnest",
            split="val",
            question=(
                "查询 ods_app_event_log_di 在 2026年10月10日至 2026年10月15日分区中，"
                "解析 ext_json 里的页面（$.page，过滤 NULL 和空串）和停留时长（$.stay_ms），"
                "统计各页面的去重用户数（uv）和平均停留时长（avg_stay_ms，保留2位小数），按 page 升序排列。"
            ),
            hint="限定分区范围后解析 JSON 字符串。",
            gold_sql=(
                "SELECT page, COUNT(DISTINCT user_id) AS uv, "
                "ROUND(AVG(CAST(stay_ms AS DOUBLE)), 2) AS avg_stay_ms "
                "FROM ("
                "  SELECT user_id, "
                "         GET_JSON_OBJECT(ext_json, '$.page') AS page, "
                "         GET_JSON_OBJECT(ext_json, '$.stay_ms') AS stay_ms "
                "  FROM ods_app_event_log_di "
                "  WHERE ds BETWEEN '20261010' AND '20261015'"
                ") t "
                "WHERE page IS NOT NULL AND page != '' "
                "GROUP BY page "
                "ORDER BY page ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_09",
            category="json_array_unnest",
            split="val",
            question=(
                "将 dwd_content_post_di 最新分区的 topic_tags 展开并关联 dim_content_creator，"
                "统计头部达人（creator_tier = 'head'）在各话题标签（tag）下的发帖数（post_cnt），"
                "按 post_cnt 降序、tag 升序排列。"
            ),
            hint="展开 topic_tags 后与达人维表关联并筛选 creator_tier。",
            gold_sql=(
                "SELECT t.tag, COUNT(DISTINCT p.post_id) AS post_cnt "
                "FROM dwd_content_post_di p "
                "LATERAL VIEW explode(p.topic_tags) t AS tag "
                "JOIN dim_content_creator c ON p.creator_id = c.creator_id "
                "WHERE p.dt = MAX_PT('dwd_content_post_di') "
                "AND c.creator_tier = 'head' "
                "GROUP BY t.tag "
                "ORDER BY post_cnt DESC, t.tag ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_10",
            category="json_array_unnest",
            split="test",
            question=(
                "将 ods_app_event_log_di 最新分区的 expo_item_ids 展开并关联 dim_item，"
                "仅统计在售商品（status = 'on_sale'），输出曝光次数排名前 5 的二级类目（cate_l2, expo_cnt），"
                "按 expo_cnt 降序、cate_l2 升序排列。"
            ),
            hint="展开商品 ID 后关联 dim_item 过滤在售状态。",
            gold_sql=(
                "SELECT i.cate_l2, COUNT(*) AS expo_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW explode(e.expo_item_ids) t AS item_id "
                "JOIN dim_item i ON t.item_id = i.item_id "
                "WHERE e.ds = MAX_PT('ods_app_event_log_di') "
                "AND i.status = 'on_sale' "
                "GROUP BY i.cate_l2 "
                "ORDER BY expo_cnt DESC, i.cate_l2 ASC "
                "LIMIT 5"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_11",
            category="json_array_unnest",
            split="test",
            question=(
                "查询 ods_app_event_log_di 最新分区中事件类型为 'click' 且 ext_json 内 $.ab_group = 'exp_a' 的记录，"
                "将 expo_item_ids 展开关联 dim_item，统计各一级类目（cate_l1）的点击曝光商品数（expo_cnt），"
                "按 cate_l1 升序排列。"
            ),
            hint="同时组合 JSON 条件过滤与数组行转列关联。",
            gold_sql=(
                "SELECT i.cate_l1, COUNT(*) AS expo_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW explode(e.expo_item_ids) t AS item_id "
                "JOIN dim_item i ON t.item_id = i.item_id "
                "WHERE e.ds = MAX_PT('ods_app_event_log_di') "
                "AND e.event_type = 'click' "
                "AND GET_JSON_OBJECT(e.ext_json, '$.ab_group') = 'exp_a' "
                "GROUP BY i.cate_l1 "
                "ORDER BY i.cate_l1 ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_12",
            category="json_array_unnest",
            split="test",
            question=(
                "查询 dwd_content_post_di 在 '20261010' 至 '20261015' 分区中 meta_json 评分（$.score）大于等于 80 分的帖子，"
                "展开 topic_tags 统计每个标签（tag）的出现次数（post_cnt）及总点赞量（total_likes），"
                "按 post_cnt 降序、tag 升序排列。"
            ),
            hint="先提取 $.score 转浮点型过滤，再展开数组标签。",
            gold_sql=(
                "SELECT t.tag, COUNT(*) AS post_cnt, SUM(p.like_cnt) AS total_likes "
                "FROM dwd_content_post_di p "
                "LATERAL VIEW explode(p.topic_tags) t AS tag "
                "WHERE p.dt BETWEEN '20261010' AND '20261015' "
                "AND CAST(GET_JSON_OBJECT(p.meta_json, '$.score') AS DOUBLE) >= 80.0 "
                "GROUP BY t.tag "
                "ORDER BY post_cnt DESC, t.tag ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_13",
            category="json_array_unnest",
            split="test",
            question=(
                "查询 ods_app_event_log_di 最新分区，提取 $.device.os（过滤 NULL 和空串）并外展开 expo_item_ids（保留空数组行），"
                "统计各设备系统的去重事件数（distinct_events）和展开后的非空商品曝光数（expo_item_cnt），"
                "按 device_os 升序排列。"
            ),
            hint="使用外展开防止没有曝光数组的事件被直接丢弃，事件数按 event_id 去重。",
            gold_sql=(
                "SELECT GET_JSON_OBJECT(e.ext_json, '$.device.os') AS device_os, "
                "COUNT(DISTINCT e.event_id) AS distinct_events, "
                "COUNT(t.item_id) AS expo_item_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW OUTER explode(e.expo_item_ids) t AS item_id "
                "WHERE e.ds = MAX_PT('ods_app_event_log_di') "
                "AND GET_JSON_OBJECT(e.ext_json, '$.device.os') IS NOT NULL "
                "AND GET_JSON_OBJECT(e.ext_json, '$.device.os') != '' "
                "GROUP BY GET_JSON_OBJECT(e.ext_json, '$.device.os') "
                "ORDER BY device_os ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_14",
            category="json_array_unnest",
            split="test",
            question=(
                "查询 ods_app_event_log_di 最新分区中 ext_json 的停留时长 $.stay_ms 超过 5000 毫秒的记录，"
                "按 $.ab_group 与 $.page（两者均过滤 NULL 和空串）分组统计事件数（event_cnt），"
                "按 ab_group 升序、page 升序排列。"
            ),
            hint="从 ext_json 中分别提取 ab_group, page, stay_ms 过滤后分组。",
            gold_sql=(
                "SELECT ab_group, page, COUNT(*) AS event_cnt "
                "FROM ("
                "  SELECT GET_JSON_OBJECT(ext_json, '$.ab_group') AS ab_group, "
                "         GET_JSON_OBJECT(ext_json, '$.page') AS page, "
                "         CAST(GET_JSON_OBJECT(ext_json, '$.stay_ms') AS DOUBLE) AS stay_ms "
                "  FROM ods_app_event_log_di "
                "  WHERE ds = MAX_PT('ods_app_event_log_di')"
                ") t "
                "WHERE ab_group IS NOT NULL AND ab_group != '' "
                "AND page IS NOT NULL AND page != '' "
                "AND stay_ms > 5000 "
                "GROUP BY ab_group, page "
                "ORDER BY ab_group ASC, page ASC"
            ),
            ordered=True,
        ),
        BenchmarkTask(
            task_id="ja_15",
            category="json_array_unnest",
            split="test",
            question=(
                "将 ods_app_event_log_di 在 '20261012' 至 '20261015' 分区的 expo_item_ids 展开，"
                "依次关联 dim_item 和 dim_shop，统计各店铺类型（shop_type）的商品曝光总次数（expo_cnt），"
                "按 shop_type 升序排列。"
            ),
            hint="展开商品 ID 后先关联商品表再关联店铺维表。",
            gold_sql=(
                "SELECT s.shop_type, COUNT(*) AS expo_cnt "
                "FROM ods_app_event_log_di e "
                "LATERAL VIEW explode(e.expo_item_ids) t AS item_id "
                "JOIN dim_item i ON t.item_id = i.item_id "
                "JOIN dim_shop s ON i.shop_id = s.shop_id "
                "WHERE e.ds BETWEEN '20261012' AND '20261015' "
                "GROUP BY s.shop_type "
                "ORDER BY s.shop_type ASC"
            ),
            ordered=True,
        ),
    ]
