# EvoDW-RL 实验评测报告（数仓 SQL 小模型 SFT + GRPO 后训练对比）

- 生成时间: `2026-11-16 22:21:58`
- 小模型基座: `Qwen2.5-Coder-1.5B-Instruct`
- 隔离评测集: `Test Split = 30 道企业数仓题`（与 SFT 85 条轨迹 / GRPO 45 道训练题严格零交集）
- 本地沙盒环境: `WarehouseSandbox`（`GatewayLinter` 静态分区/方言拦截 + `LocalDuckDBExecutor` 确定性转译执行）

## 1. 五组模型核心指标全链路对比（Isolated Test Set）

| 模型方案 | EX (执行准确率) | First-Pass Gateway Rate | Invalid SQL Rate | 1st-Pass 对 / 自修复对 | Avg Steps | 单题推理延迟 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **Base 小模型 (Qwen2.5-Coder-1.5B-Instruct Zero-shot)** | **3.3%** | 13.3% | 76.7% | 0 / 1 | 3.80 | 9289.7 |
| **Base + GEPA Evolved Prompt (Qwen2.5-Coder-1.5B-Instruct)** | **3.3%** | 13.3% | 80.0% | 0 / 1 | 3.97 | 9670.5 |
| **SFT 冷启动模型 (Loss-Masked Multi-Turn)** | **20.0%** | 60.0% | 66.7% | 6 / 0 | 3.33 | 12084.7 |
| **SFT + GRPO 模型 (4-Stage Dense Reward)** | **20.0%** | 56.7% | 70.0% | 6 / 0 | 3.47 | 19677.5 |
| **在线大模型 deepseek-v4.1-flash (GEPA)** | **90.0%** | 90.0% | 0.0% | 24 / 3 | 2.17 | 106377.0 |

## 2. 五大数仓高频错题分项准确率（EX）对比

| 题型类别 | 题数 | Base 小模型 (Qwen2.5-Coder-1.5B-Instruct Zero-shot) | Base + GEPA Evolved Prompt (Qwen2.5-Coder-1.5B-Instruct) | SFT 冷启动模型 (Loss-Masked Multi-Turn) | SFT + GRPO 模型 (4-Stage Dense Reward) | 在线大模型 deepseek-v4.1-flash (GEPA) |
|---|---:| ---: | ---: | ---: | ---: | ---: |
| `partition_pruning` | 6 | 0.0% _(GW 16.7%)_ | 0.0% _(GW 16.7%)_ | 16.7% _(GW 83.3%)_ | 0.0% _(GW 66.7%)_ | 83.3% _(GW 83.3%)_ |
| `json_array_unnest` | 6 | 0.0% _(GW 0.0%)_ | 0.0% _(GW 0.0%)_ | 33.3% _(GW 100.0%)_ | 50.0% _(GW 83.3%)_ | 83.3% _(GW 100.0%)_ |
| `retention_date_math` | 6 | 0.0% _(GW 0.0%)_ | 0.0% _(GW 0.0%)_ | 0.0% _(GW 16.7%)_ | 0.0% _(GW 33.3%)_ | 83.3% _(GW 66.7%)_ |
| `fanout_dedup` | 6 | 0.0% _(GW 16.7%)_ | 0.0% _(GW 16.7%)_ | 33.3% _(GW 50.0%)_ | 33.3% _(GW 50.0%)_ | 100.0% _(GW 100.0%)_ |
| `ratio_null_handling` | 6 | 16.7% _(GW 33.3%)_ | 16.7% _(GW 33.3%)_ | 16.7% _(GW 50.0%)_ | 16.7% _(GW 50.0%)_ | 100.0% _(GW 100.0%)_ |

## 3. 两阶段后训练遥测摘要（Stage-1 SFT & Stage-2 GRPO）

### 3.1 第一阶段：Loss-Masked 多轮轨迹 SFT
- 轨迹构成: 原始验证通过轨迹 `85` 条 （首轮即对 `70` 条 + 网关报错自修复 `15` 条，含 `inspect_table_sample` 探查 `8` 条）
- Loss Mask 统计: 总 Token `320430`，仅参与 Cross-Entropy 的 `assistant` Token `58294` （占比 `18.2%`），其余 `system`/`user`/`tool` 全部严格屏蔽为 `-100`
- 训练步数 / 最终 Loss / 耗时: `36` steps / `0.8485` / `530.7s`

### 3.2 第二阶段：四级递进稠密奖励 GRPO
- 训练题集: `grpo_prompts.jsonl` 共 `45` 道 `train + val` 题（严禁且零 `test` 泄露）
- 激活奖励组件 (`full`): `format_reward, gateway_ast_reward, execution_validity_reward, ex_outcome_efficiency_reward`
- 训练步数 / 耗时: `8` steps / `467.9s`

## 4. 典型测试题演进案例对比（Base -> SFT -> SFT+GRPO）

### 案例 `fd_15`（类别: `fanout_dedup`）
- **业务问题**: 对 ods_app_event_log_di 最新分区先按 event_id 去重（取最新 ingest_ts），再找出每位用户当日时间最早的一条事件（按 ingest_ts 升序、event_id 升序取第 1 条），统计各页面（page_name）作为用户当日首访页面的用户数（first_visit_users），按 first_visit_users 降序、page_name 升序排列。
- **Base Zero-shot**: EX=`False`, 首轮网关=`False` | SQL: `SELECT
  page_name,
  COUNT(DISTINCT user_id) AS first_visit_users
FROM
  ods_app_event_log_di
WHERE
  ds BETWEEN '20261015' AND '20261015'
  AND event_id IN (
    SELECT
      event_id
    FROM
      ods_app_event_log_d`
  - 报错诊断: `FAILED: ODPS-0123031: Partition exception - full table scan is prohibited without partition filter on column 'ds' for table 'ods_app_event_log_di'`
- **SFT 冷启动**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH deduplicated AS (
    SELECT
        event_id,
        user_id,
        page_name,
        ingest_ts,
        ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts ASC, event_id ASC) AS rn
    FROM ods_app_eve`
- **SFT + GRPO**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH deduplicated AS (
    SELECT
        event_id,
        user_id,
        page_name,
        ingest_ts,
        ROW_NUMBER() OVER (
            PARTITION BY event_id
            ORDER BY ingest_ts ASC, event_id ASC
  `

### 案例 `ja_10`（类别: `json_array_unnest`）
- **业务问题**: 将 ods_app_event_log_di 最新分区的 expo_item_ids 展开并关联 dim_item，仅统计在售商品（status = 'on_sale'），输出曝光次数排名前 5 的二级类目（cate_l2, expo_cnt），按 expo_cnt 降序、cate_l2 升序排列。
- **Base Zero-shot**: EX=`False`, 首轮网关=`False` | SQL: `SELECT T2.cate_l2 AS cate_l2, COUNT(T1.expo_item_ids) AS expo_cnt FROM ods_app_event_log_di AS T1 INNER JOIN dim_item AS T2 ON T1.expo_item_ids = T2.item_id WHERE T2.status = 'on_sale' AND T1.ds = 'latest_partition_value`
  - 报错诊断: `FAILED: ODPS-0123031: Partition exception - full table scan is prohibited without partition filter on column 'ds' for table 'ods_app_event_log_di'`
- **SFT 冷启动**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH exploded AS (
    SELECT exploded_item_id
    FROM ods_app_event_log_di
    LATERAL VIEW explode(expo_item_ids) t AS exploded_item_id
    WHERE ds = MAX_PT('ods_app_event_log_di')
)
SELECT d.cate_l2,
       COUNT(*)`
- **SFT + GRPO**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH exploded_item_id AS (
    SELECT item_id
    FROM ods_app_event_log_di
    LATERAL VIEW explode(expo_item_ids) t AS item_id
    WHERE ds = MAX_PT('ods_app_event_log_di')
)
SELECT d.cate_l2,
       COUNT(*) AS expo_c`

### 案例 `ja_14`（类别: `json_array_unnest`）
- **业务问题**: 查询 ods_app_event_log_di 最新分区中 ext_json 的停留时长 $.stay_ms 超过 5000 毫秒的记录，按 $.ab_group 与 $.page（两者均过滤 NULL 和空串）分组统计事件数（event_cnt），按 ab_group 升序、page 升序排列。
- **Base Zero-shot**: EX=`False`, 首轮网关=`False` | SQL: `with the `JSON_TABLE` function to parse the JSON string and extract the required fields.",
  "sql": "SELECT COUNT(event_id) AS event_cnt, ab_group, page FROM (\n" +
         "    SELECT\n" +
         "        t1.ab_group`
  - 报错诊断: `FAILED: ODPS-0130161: Parse exception - tokenization error: Error tokenizing '
         "ORDER BY ab_group ASC, page ASC;"
}
``'`
- **SFT 冷启动**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH e AS (
    SELECT
        user_id,
        event_type,
        page_name,
        ext_json,
        get_json_object(ext_json, '$.ab_group') AS ab_group,
        get_json_object(ext_json, '$.page') AS page,
        g`
- **SFT + GRPO**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH e AS (
    SELECT
        user_id,
        event_type,
        page_name,
        ext_json,
        get_json_object(ext_json, '$.ab_group') AS ab_group,
        get_json_object(ext_json, '$.page') AS page,
        g`
