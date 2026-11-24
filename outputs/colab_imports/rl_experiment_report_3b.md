# EvoDW-RL v2 实验评测报告（多轮工具交互 SFT + Observation-Masked Agentic GRPO）

- 生成时间: `2026-11-17 12:07:11`
- 主实验模型基座: `Qwen2.5-Coder-3B-Instruct`
- 隔离评测集: `Test Split = 30 道企业数仓题`（与 SFT `299` 条验证轨迹 / GRPO `45` 道训练题严格零交集）
- 本地双层仿真沙盒: `WarehouseSandbox`（`GatewayLinter` 静态分区/方言 AST 拦截 + `LocalDuckDBExecutor` 确定性转译执行）

## 1. 五组/全链路模型核心指标对比（Isolated Test Set）

| 模型方案 | EX (执行准确率) | First-Pass Gateway Rate | Invalid SQL Rate | 1st-Pass 对 / 自修复对 | Avg Steps | 单题推理延迟 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **Base 小模型 (Qwen2.5-Coder-3B-Instruct Zero-shot)** | **10.0%** | 30.0% | 56.7% | 2 / 1 | 3.73 | 6550.9 |
| **Base + GEPA Evolved Prompt (Qwen2.5-Coder-3B-Instruct)** | **16.7%** | 60.0% | 66.7% | 4 / 1 | 3.60 | 9212.4 |
| **垂直 SQL 模型 (XiYanSQL-QwenCoder-3B-2504 Zero-shot)** | **16.7%** | 46.7% | 76.7% | 5 / 0 | 1.00 | 857.2 |
| **SFT 冷启动模型 (Loss-Masked Multi-Turn)** | **53.3%** | 93.3% | 26.7% | 16 / 0 | 2.67 | 8815.6 |
| **Multi-Turn SFT + GRPO (Outcome-Only Sparse Reward)** | **40.0%** | 90.0% | 30.0% | 11 / 1 | 2.93 | 9553.8 |
| **Multi-Turn SFT + GRPO (4-Stage Dense + Self-Repair)** | **46.7%** | 90.0% | 26.7% | 13 / 1 | 2.80 | 9557.1 |
| **在线大模型 deepseek-v4.1-flash (GEPA)** | **100.0%** | 96.7% | 0.0% | 29 / 1 | 2.17 | 27836.2 |

## 2. 五大数仓高频错题分项准确率（EX）对比

| 题型类别 | 题数 | Base 小模型 (Qwen2.5-Coder-3B-Instruct Zero-shot) | Base + GEPA Evolved Prompt (Qwen2.5-Coder-3B-Instruct) | 垂直 SQL 模型 (XiYanSQL-QwenCoder-3B-2504 Zero-shot) | SFT 冷启动模型 (Loss-Masked Multi-Turn) | Multi-Turn SFT + GRPO (Outcome-Only Sparse Reward) | Multi-Turn SFT + GRPO (4-Stage Dense + Self-Repair) | 在线大模型 deepseek-v4.1-flash (GEPA) |
|---|---:| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `partition_pruning` | 6 | 16.7% _(GW 33.3%)_ | 0.0% _(GW 66.7%)_ | 16.7% _(GW 16.7%)_ | 66.7% _(GW 83.3%)_ | 50.0% _(GW 83.3%)_ | 50.0% _(GW 100.0%)_ | 100.0% _(GW 83.3%)_ |
| `json_array_unnest` | 6 | 0.0% _(GW 0.0%)_ | 16.7% _(GW 50.0%)_ | 0.0% _(GW 16.7%)_ | 66.7% _(GW 100.0%)_ | 66.7% _(GW 100.0%)_ | 66.7% _(GW 100.0%)_ | 100.0% _(GW 100.0%)_ |
| `retention_date_math` | 6 | 0.0% _(GW 33.3%)_ | 16.7% _(GW 50.0%)_ | 0.0% _(GW 66.7%)_ | 16.7% _(GW 83.3%)_ | 0.0% _(GW 66.7%)_ | 16.7% _(GW 50.0%)_ | 100.0% _(GW 100.0%)_ |
| `fanout_dedup` | 6 | 0.0% _(GW 50.0%)_ | 16.7% _(GW 66.7%)_ | 16.7% _(GW 66.7%)_ | 66.7% _(GW 100.0%)_ | 33.3% _(GW 100.0%)_ | 50.0% _(GW 100.0%)_ | 100.0% _(GW 100.0%)_ |
| `ratio_null_handling` | 6 | 33.3% _(GW 33.3%)_ | 33.3% _(GW 66.7%)_ | 50.0% _(GW 66.7%)_ | 50.0% _(GW 100.0%)_ | 50.0% _(GW 100.0%)_ | 50.0% _(GW 100.0%)_ | 100.0% _(GW 100.0%)_ |

## 4. 垂直开源 SQL 模型水土不服对照实验（XiYanSQL vs EvoDW-RL）

| 对比方案 | Test EX | First-Pass Gateway Rate | Invalid SQL Rate | 分区裁剪题 EX (`partition_pruning`) | JSON/数组展开题 EX (`json_array_unnest`) | 单题延迟 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **垂直 SQL 模型 (XiYanSQL-QwenCoder-3B-2504 Zero-shot)** | **16.7%** | 46.7% | 76.7% | 16.7% | 0.0% | 857.2 |
| **Multi-Turn SFT + GRPO (4-Stage Dense + Self-Repair)** | **46.7%** | 90.0% | 26.7% | 50.0% | 66.7% | 9557.1 |

- **垂直模型典型网关拦截归因**: `{"ODPS-0123031": 11, "ODPS-0130071": 3, "ODPS-0130161": 2}`

## 5. GRPO 奖励函数消融实验（Outcome-Only Sparse vs 4-Stage Dense + Self-Repair）

| 奖励配置 | 收敛步数 (达峰值80%) | 训练期格式坍塌率 | 平均 KL 散度 | 在线自修复命中率 | 隔离测试集 Invalid SQL | 隔离测试集最终 EX |
|---|---:|---:|---:|---:|---:|---:|
| **Outcome-Only Sparse (仅最终 EX 0/1)** | `4` | 3.5% | `0.0026` | 5.1% | 30.0% | **40.0%** |
| **4-Stage Dense + Self-Repair (本文完整奖励)** | `4` | 3.1% | `0.0047` | 4.4% | 26.7% | **46.7%** |

## 6. 两阶段后训练遥测摘要（Stage-1 SFT & Stage-2 Multi-Turn GRPO）

### 6.1 第一阶段：Loss-Masked 多轮轨迹 SFT（自适应稀缺轨迹重加权）
- 原始去重验证轨迹: `299` 条 （首轮即对 `269` 条 + 网关报错自修复 `30` 条，含 `inspect_table_sample` 探查 `27` 条）
- 自适应重加权后训练轮数: `391` 条 （自修复 `84` 条 / 探查 `71` 条，尾部截断轨迹数 `0`）
- Loss Mask 统计: 总 Token `1456844`，仅参与 Cross-Entropy 的 `assistant` Token `238999` （占比 `16.4%`），其余 `system`/`user`/`tool` 严格屏蔽为 `-100`
- 训练步数 / 最终 Loss / 耗时: `147` steps / `0.5095` / `1992.6s`

### 6.2 第二阶段：多轮沙盒反馈 Observation-Masked Agentic GRPO
- 训练题集: `grpo_prompts.jsonl` 共 `45` 道 `train + val` 题（严格隔离 30 道 `test` 题）
- 多轮交互配置: `max_rollout_turns=3`, `G=8`, `KL beta=0.04` （Action Token `completion_mask=1`，沙盒报错 Observation Token 严格 `completion_mask=0`）
- 激活奖励组件 (`full`): `format_reward, gateway_ast_reward, execution_validity_reward, ex_outcome_efficiency_reward`
- 训练步数 / 峰值奖励 / 耗时: `100` steps / `1.8226` / `5108.0s`

## 7. 典型测试题演进案例对比（Base -> SFT -> Multi-Turn SFT+GRPO）

### 案例 `ja_11`（类别: `json_array_unnest`）
- **业务问题**: 查询 ods_app_event_log_di 最新分区中事件类型为 'click' 且 ext_json 内 $.ab_group = 'exp_a' 的记录，将 expo_item_ids 展开关联 dim_item，统计各一级类目（cate_l1）的点击曝光商品数（expo_cnt），按 cate_l1 升序排列。
- **Base Zero-shot**: EX=`False`, 首轮网关=`False` | SQL: `SELECT T2.cate_l1, COUNT(DISTINCT T1.expo_item_ids[1]) AS expo_cnt
FROM ods_app_event_log_di AS T1
JOIN dim_item AS T2 ON T1.expo_item_ids[1] = T2.item_id
WHERE T1.event_type = 'click' AND T1.ext_json LIKE '%"ab_group": "exp_a"%' AND T1.ds `
  - 报错诊断: `FAILED: ODPS-0123031: Partition exception - full table scan is prohibited without partition filter on column 'ds' for table 'ods_app_event_log_di'`
- **SFT 冷启动**: EX=`False`, 首轮网关=`True`, 修复轮次=`2` | SQL: `SELECT
    b.cate_l1 AS cate_l1,
    COUNT(1) AS expo_cnt
FROM (
    SELECT
        item_id,
        cate_l1
    FROM ods_app_event_log_di
    LATERAL VIEW EXPLODE(expo_item_ids) t AS item_id
    WHERE ds = MAX_PT('ods_app_event_log_di')
  `
- **Multi-Turn SFT + GRPO**: EX=`True`, 首轮网关=`True`, 修复轮次=`1` | SQL: `WITH base AS (
  SELECT item_id
  FROM ods_app_event_log_di
  LATERAL VIEW explode(expo_item_ids) t AS item_id
  WHERE ds = MAX_PT('ods_app_event_log_di')
    AND event_type = 'click'
    AND GET_JSON_OBJECT(ext_json, '$.ab_group') = 'exp_a`

### 案例 `rd_10`（类别: `retention_date_math`）
- **业务问题**: 汇总统计 2026-09-26 至 2026-10-01 期间注册的所有用户，按拉新渠道（channel）统计总体注册人数（cohort_users）、次日（D1）在 ods_app_event_log_di 活跃的留存人数（d1_users）及次日留存率（d1_rate，保留4位小数），按 channel 升序排列。
- **Base Zero-shot**: EX=`False`, 首轮网关=`False` | SQL: `SELECT ur.channel, COUNT(DISTINCT ur.user_id) AS cohort_users, COUNT(DISTINCT CASE WHEN TO_DATE(ur.register_date) = DATEADD(TO_DATE(o.ds), INTERVAL 1 DAY, 'dd') THEN ur.user_id END) AS d1_users, ROUND(COUNT(DISTINCT CASE WHEN TO_DATE(ur.reg`
  - 报错诊断: `FAILED: ODPS-0130071: Semantic analysis exception - function 'DATE_ADD' is not supported in ODPS; use DATEADD(datetime, delta, datepart).`
- **SFT 冷启动**: EX=`False`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH cohort AS (
  SELECT
    user_id,
    channel,
    reg_dt,
    TO_CHAR(DATEADD(TO_DATE(reg_dt, 'yyyy-mm-dd'), 1, 'dd'), 'yyyymmdd') AS next_day_ds
  FROM dim_user
  WHERE reg_dt BETWEEN '2026-09-26' AND '2026-10-01'
),
cohort_stats AS `
- **Multi-Turn SFT + GRPO**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH cohort AS (
  SELECT
    u.channel,
    COUNT(DISTINCT u.user_id) AS cohort_users
  FROM dim_user u
  WHERE u.reg_dt BETWEEN '2026-09-26' AND '2026-10-01'
  GROUP BY u.channel
),
d1 AS (
  SELECT
    u.channel,
    COUNT(DISTINCT e.use`

### 案例 `ja_12`（类别: `json_array_unnest`）
- **业务问题**: 查询 dwd_content_post_di 在 '20261010' 至 '20261015' 分区中 meta_json 评分（$.score）大于等于 80 分的帖子，展开 topic_tags 统计每个标签（tag）的出现次数（post_cnt）及总点赞量（total_likes），按 post_cnt 降序、tag 升序排列。
- **Base Zero-shot**: EX=`False`, 首轮网关=`False` | SQL: `SELECT
  t.tag,
  COUNT(t.post_id) AS post_cnt,
  SUM(t.like_cnt) AS total_likes
FROM (
  SELECT
    t1.post_id,
    t1.creator_id,
    t1.content_type,
    t1.meta_json,
    CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(t1.meta_json, '"score":', -1`
  - 报错诊断: `FAILED: ODPS-0130071: Semantic analysis exception - function 'UNNEST' is not supported in ODPS SQL.`
- **SFT 冷启动**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `WITH base AS (
    SELECT
        tag,
        COUNT(*) AS post_cnt,
        SUM(like_cnt) AS total_likes
    FROM dwd_content_post_di
    LATERAL VIEW EXPLODE(topic_tags) t AS tag
    WHERE dt BETWEEN '20261010' AND '20261015'
      AND CA`
- **Multi-Turn SFT + GRPO**: EX=`True`, 首轮网关=`True`, 修复轮次=`0` | SQL: `SELECT
  tag,
  COUNT(*) AS post_cnt,
  SUM(like_cnt) AS total_likes
FROM dwd_content_post_di
LATERAL VIEW EXPLODE(topic_tags) t AS tag
WHERE dt BETWEEN '20261010' AND '20261015'
  AND CAST(GET_JSON_OBJECT(meta_json, '$.score') AS DOUBLE) >`
