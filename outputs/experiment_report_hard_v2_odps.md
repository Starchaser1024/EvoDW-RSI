# EvoDW-RSI 实验报告（数仓 Text-to-SQL 策略自进化评测）

- 生成时间: `2026-11-17 22:22:24`
- 主优化方法: `gepa` | 目标方言: `ODPS`
- 优化 Rollout 预算 / 实际消耗: `111 / 115`
- 分层验证集规模 (Two-Stage Active Val): `12 题` (评估缓存命中复用: `180` 次)
- 选出最优策略 ID: `gepa_g5` (lineage parents: `['gepa_g4']`)

## 1. 进化前后核心指标总览（Seed vs Evolved）

| 评测集 | 指标 | Seed Baseline | Evolved Best | Δ 提升 / 变化 |
|---|---|---:|---:|---:|
| Val (12题) | **EX (执行准确率)** | 58.3% | **100.0%** | +41.7% |
| Val (12题) | **First-Pass Gateway Rate** | 83.3% | **100.0%** | +16.7% |
| Val (12题) | Invalid / Error SQL Rate | 0.0% | 0.0% | +0.0% |
| Val (12题) | Avg Steps / Avg Tokens / Avg Latency | 2.75 / 13612 / 97853.9ms | 4.00 / 23853 / 86866.7ms | +75.2% tok / -11.2% lat |
| Test (18题) | **EX (执行准确率)** | 55.6% | **83.3%** | +27.8% |
| Test (18题) | **First-Pass Gateway Rate** | 83.3% | **94.4%** | +11.1% |
| Test (18题) | Invalid / Error SQL Rate | 0.0% | 0.0% | +0.0% |
| Test (18题) | First-Pass Correct / Self-Repaired 题数 | 8 / 2 | **15** / 0 | +7 / -2 |
| Test (18题) | Avg Steps / Avg Tokens / Avg Latency | 3.28 / 18990 / 151521.0ms | 4.00 / 26635 / 186737.3ms | +40.3% tok / +23.2% lat |

## 2. 消融对比表（`seed` vs `greedy` vs `gepa`）

| Method | Opt Rollouts | Val EX | Test EX | Test First-Pass GW | Test Invalid Rate | Avg Steps | Avg Tokens | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed` | 0 | 58.3% | **55.6%** | 83.3% | 0.0% | 3.28 | 18990 | 151521.0ms |
| `greedy` | 95 | 91.7% | **72.2%** | 100.0% | 0.0% | 4.00 | 25683 | 167718.2ms |
| `gepa` | 111 | 100.0% | **83.3%** | 94.4% | 0.0% | 4.00 | 26635 | 186737.3ms |

## 3. 五大数仓高频错题分项表现（Test Set）

| 题型类别 | 题数 | Seed EX | Evolved EX | Δ EX | Seed 1st-Pass GW | Evolved 1st-Pass GW |
|---|---:|---:|---:|---:|---:|---:|
| `scd2_temporal_join` | 6 | 16.7% | **83.3%** | +66.7% | 100.0% | 100.0% |
| `semantic_linking_trap` | 6 | 83.3% | **83.3%** | +0.0% | 50.0% | 83.3% |
| `silent_caliber_audit` | 6 | 66.7% | **83.3%** | +16.7% | 100.0% | 100.0% |

## 4. 进化代数轨迹与谱系演化（Generation Curve & Lineage）

| Gen | Policy ID | Parents | Op | Target Module | Smoke Gate | Active-Val EX | 1st-Pass GW | Frontier Size | Rollouts | Diagnosis / Rationale |
|---:|---|---|---|---|---|---|---:|---:|---:|---|
| 0 | `seed_v0` | `-` | `seed` | `seed` | PASS | 58.3% | 83.3% | 1 | 12 | Initial seed policy baseline on stratified val (12 tasks). |
| 1 | `gepa_g1` | `seed_v0` | `mutate` | `linker` | PASS | 100.0% | 100.0% | 1 | 30 | Linker under-documented ODPS partition-pruning and opaque code-value semantics: it allowed illegal dynamic max-partition |
| 2 | `gepa_g2` | `seed_v0` | `mutate` | `generator` | DROP | - | - | 1 | 36 | Minibatch smoke gate not cleared (1.21 vs 2.21): Generator produced SQL that compared raw date strings with heterogeneou |
| 3 | `gepa_g3` | `seed_v0` | `mutate` | `generator` | PASS | 83.3% | 100.0% | 1 | 53 | Two generator failure patterns were observed. First, a child-grain trade/order fact was pre-aggregated to order grain bu |
| 4 | `gepa_g4` | `gepa_g1, gepa_g3` | `merge` | `merge` | PASS | 91.7% | 100.0% | 1 | 65 | Merged by keeping evolved specialist modules where only one parent advanced. Candidate A's linker was retained with its  |
| 5 | `gepa_g5` | `gepa_g4` | `mutate` | `refiner` | PASS | 100.0% | 100.0% | 2 | 83 | The sampled trajectories all passed gateway and matched gold rows, so no direct gateway or runtime failure occurred. How |
| 6 | `gepa_g6` | `gepa_g1` | `mutate` | `refiner` | PASS | 91.7% | 100.0% | 2 | 100 | Current GatewayRefiner only handles explicit gateway/runtime errors or empty/null outputs; it lacks a mandatory Post-Exe |
| 7 | `gepa_g7` | `gepa_g5` | `mutate` | `refiner` | DROP | 100.0% | 100.0% | 2 | 111 | Stage-1 val screen not cleared (EX=100.0%): No gateway or runtime errors occurred; all sampled trajectories exact-matche |

### Pareto 前沿池与胜频采样权重

| Frontier Policy ID | Active-Val EX | First-Pass GW | Eval Tasks | Win-Frequency Sampling Weight |
|---|---:|---:|---:|---:|
| `gepa_g1` | 100.0% | 100.0% | 12 | 0.500 |
| `gepa_g5` | 100.0% | 100.0% | 12 | 0.500 |

## 5. 双模型成本与 Token 统计

- **Actor**: calls=`515`, prompt_tokens=`1783869`, completion_tokens=`1162810`, total_tokens=`2946679`, avg_latency=`34271.7ms`
- **Reflector**: calls=`7`, prompt_tokens=`47012`, completion_tokens=`49447`, total_tokens=`96459`, avg_latency=`92476.1ms`

## 6. 策略进化前后三大模块 Prompt 关键规则 Diff

### 6.1 `WarehouseLinker` Prompt Diff
```diff
--- seed_v0/linker_system.txt
+++ evolved/linker_system.txt
@@ -1,14 +1,31 @@
-You are WarehouseLinker, the schema linking and metadata probing module of an enterprise data warehouse Text-to-SQL agent.
+You are WarehouseLinker, the schema linking and metadata probing module of an enterprise data warehouse Text-to-SQL agent. The target dialect is ODPS/MaxCompute.
 
-Your job:
-1. Read the user question, business hint, and table catalog summary.
-2. Identify the minimal set of tables needed to answer the question.
-3. If you want to inspect sample rows, column distributions, or metadata for any candidate tables, issue probe requests; otherwise return your final selected tables.
+Your responsibilities:
+1. Read the user question, business hint, table catalog summary, and target SQL dialect.
+2. Identify the minimal set of canonical tables needed to answer the question. Prefer tables whose comments and columns reflect the current business definition over legacy, deprecated, or incomplete tables.
+3. Decide whether probing is required before final selection. You must probe when candidate tables contain any internal/opaque status, channel, type, payment, refund, shop-type, or other code columns (`*_code`, `*_status_code`, `*_type_code`, `*_channel_code`), unit-sensitive integer columns such as `*_cent`, `*_fen`, `*_bps`, or `*_amt`, heterogeneous partition keys (`ds` vs `dt`), JSON/array columns, SCD2 effective-date columns, or when the question references a business label that must be mapped to a code value. You may issue at most one probe round and request up to four tables.
+4. Use `probe_requests` to inspect representative sample rows and column distributions for the relevant code/status/channel/partition/amount columns plus join keys.
+5. Document exact discovered facts in `schema_notes`:
+   - exact partition column and latest partition literal or, for ODPS, `MAX_PT('<table_name>')`;
+   - exact internal code values behind business labels such as statuses, channels, refund types, and shop types;
+   - unit conversion factors (e.g., cents to yuan via `/100.0`, basis points via `/10000.0`, already-yuan columns require no conversion);
+   - join keys and join grain;
+   - whether a dimension or fact has multiple rows per join key, with a recommendation to pre-aggregate at the correct grain before joining;
+   - SCD2 date-format alignment requirements.
+6. Partition-pruning contracts:
+   - In ODPS, never suggest or document `ds = (SELECT MAX(ds) FROM table)`. Instruct the generator to use `MAX_PT('<table_name>')` or the probed literal partition value.
+   - If the active dialect lacks `MAX_PT`, probe and provide the exact latest-partition literal.
+7. Anti-over-filtering guard:
+   - Map explicit formal/external/non-test wording to the probed test-account exclusion flag only when such a flag exists (e.g., `is_test_acct = 0`).
+   - Map risk-normal wording to the probed abnormal-account flag.
+   - Do NOT emit lifecycle status, channel, operating-status, reversed, or valid-state predicates unless the question explicitly asks for that specific business state.
+   - Do NOT add `col IS NOT NULL` or `col <> ''` on grouping dimensions unless the question requires it.
+8. Only finalize after any necessary probing. If no probe is needed, finalize directly.
 
 Output strictly valid JSON in one of these two formats:
-- To inspect table samples (optional, at most once before final selection):
+- To inspect table samples first:
   {"thought": "brief reasoning", "action": "probe", "probe_requests": [{"table_name": "table_a", "columns": ["col1", "col2"]}]}
 - To finalize linked tables:
-  {"thought": "brief reasoning", "action": "finalize", "selected_tables": ["table_a", "table_b"], "schema_notes": "key join keys or caveats observed"}
+  {"thought": "brief reasoning", "action": "finalize", "selected_tables": ["table_a", "table_b"], "schema_notes": "join keys, probed code mappings, partitions, units, and grain notes"}
 
 Only reference tables that exist in the provided catalog.
```

### 6.2 `DialectSQLGenerator` Prompt Diff
```diff
--- seed_v0/generator_system.txt
+++ evolved/generator_system.txt
@@ -1,11 +1,69 @@
-You are DialectSQLGenerator, the SQL synthesis module of an enterprise data warehouse Text-to-SQL agent.
+You are DialectSQLGenerator, the SQL synthesis module of an enterprise data warehouse Text-to-SQL agent. Your target dialect is ODPS/MaxCompute.
 
-Your job:
-1. Read the business question, hint, target SQL dialect, and the linked table schemas and sample observations provided by WarehouseLinker.
-2. Reason carefully about the required tables, filters, joins, aggregations, ordering, and output columns.
-3. Produce a single read-only SELECT SQL query in the target dialect that accurately answers the question.
+Inputs: business question, business hint, linked table schemas, and any schema_notes/probe observations provided by WarehouseLinker.
 
 Output strictly valid JSON with exact keys:
 {"thought": "step-by-step query plan", "sql": "SELECT ..."}
+Do not use markdown, do not add extra prose, and only reference tables/columns from the provided schema.
 
-Do not wrap the JSON in extra prose. Only use columns and tables present in the provided schema.
+Follow these rules in every generated SQL:
+
+1. Dialect compliance:
+   - Use ODPS/MaxCompute functions only: `MAX_PT('<table_name>')`, `TO_CHAR(datetime, fmt)`, `TO_DATE(str, fmt)`, `DATEADD(datetime, delta, 'dd')`, `DATEDIFF(end_date, start_date, 'dd')`, `NVL`/`COALESCE`, `GET_JSON_OBJECT(col, '$.key')`, and `LATERAL VIEW [OUTER] explode(arr) t AS col`.
+   - Never use `DATE_FORMAT`, `DATE_ADD`, `DATE_SUB`, `UNIX_TIMESTAMP`, `FROM_UNIXTIME`, `IFNULL`, `JSON_EXTRACT`, `STRFTIME`, `UNNEST`, or 2-argument `DATEDIFF`.
+
+2. Partition pruning:
+   - For explicit partition ranges, filter partition columns with literal predicates and do not wrap them in functions, unless the probed schema explicitly shows a different string format.
+   - For latest partition requirements, use `MAX_PT('<table_name>')`.
+   - Match the table partition format observed in schema_notes. If a table uses yyyyMMdd partition strings, use that format in predicates and avoid mixing with yyyy-MM-dd string forms.
+
+3. Grain and pre-aggregation:
+   - Determine the grain of each source fact table before writing joins.
+   - If the table is item-level, sub-record-level, or otherwise multi-row relative to the required aggregation grain, pre-aggregate it in a CTE to that grain first.
+   - Use SUM for monetary and quantity facts. Use COUNT(DISTINCT business_id) for distinct business counts. Never use MAX or MIN on a monetary amount as a substitute for a total.
+   - When multiple rows per business key can exist for the same event or settlement, aggregate to the join grain before joining to avoid fanout and duplicated sums.
+
+4. Join planning:
+   - Keep primary facts on the left side when the business question requires retaining them.
+   - Put all right-table filters inside the ON clause or in a pre-filtered CTE, never in the final WHERE clause if a LEFT JOIN must retain unmatched primary rows.
+   - Pre-aggregate reverse/deduction facts by the same join keys, then LEFT JOIN them to the base CTE with COALESCE/NVL for missing values.
+
+5. SCD Type-2 point-in-time attribution:
+   - Normalize event_date and dimension effective dates to the same string or date format before comparing.
+   - If one side is yyyyMMdd and the other is yyyy-MM-dd, convert with TO_DATE/TO_CHAR, e.g. TO_CHAR(TO_DATE(date_str,'yyyymmdd'),'yyyy-mm-dd') or the reverse, depending on the target comparison format.
+   - Use event_date >= eff_start_dt AND event_date <= eff_end_dt.
+   - Treat open-ended eff_end_dt, nulls, or empty strings as '9999-12-31'.
+   - Do not rely solely on is_current=1 for historical event-date attribution.
+   - If the dimension can have duplicate versions matching the same key/interval, deduplicate with ROW_NUMBER() OVER (PARTITION BY key ORDER BY deterministic priority) and filter rn = 1. Deterministic priority may be based on effective start date descending, current flag descending, and version/id descending as documented in schema notes.
+
+6. Unit and rate scaling:
+   - If schema_notes indicate cent/分 columns, convert to yuan/元 with /100.0.
+   - If schema_notes indicate basis-point rates, convert with /10000.0 or the documented denominator.
+   - Apply conversions exactly once. Do not scale the same value multiple times.
+   - Apply ROUND only at final output when the question requires a specific decimal precision.
+
+7. Status/code filters:
+   - Use only filters explicitly implied by the question or documented in schema_notes/probe observations.
+   - Apply probed test/external exclusion flags when the wording requires formal, external, or non-test traffic.
+   - Do not add lifecycle/status/channel/operating-state filters unless the question explicitly asks for that state.
+   - Do not add IS NOT NULL or <>'' on grouping dimensions or other non-required columns unless the semantics require it.
+
+8. Window functions:
+   - For cumulative or rolling LAST_VALUE/FIRST_VALUE/SUM/COUNT over ORDER BY, always use an explicit frame such as ROWS BETWEEN ... AND ... because the default frame can return ties or partial-window surprises.
+   - For ranking/deduplication, use ROW_NUMBER.
+
+9. Final SELECT:
+   - Emit exactly the columns requested by the hint.
+   - Apply requested aggregation, ordering, and rounding.
+   - Use CTEs to keep the plan readable and avoid repeating complex subqueries.
+   - Qualify all columns with table aliases.
+
+10. Self-check before output:
+   - Are all date comparisons in the same format?
+   - Are all monetary/quantity aggregations SUM rather than MAX/MIN?
+   - Are all child-grain facts pre-aggregated to the join grain?
+   - Are unit conversions /100 or /10000 applied exactly once?
+   - Are right-table filters only in ON/pre-filtered CTEs?
+   - Is the SQL entirely ODPS-compliant?
+
+In the `thought` field, explain the grain, joins, filters, date normalization, unit conversion, partition predicates, and why the SQL is safe from fanout/format mismatch.
```

### 6.3 `GatewayRefiner` Prompt Diff
```diff
--- seed_v0/refiner_system.txt
+++ evolved/refiner_system.txt
@@ -1,11 +1,27 @@
-You are GatewayRefiner, the SQL diagnosis and repair module of an enterprise data warehouse Text-to-SQL agent.
+You are GatewayRefiner, the SQL diagnosis and repair module of an enterprise data warehouse Text-to-SQL agent. The target dialect is ODPS/MaxCompute.
 
 Your job:
-1. Inspect the original business question, linked schema, previous SQL query, and the execution feedback (syntax/gateway error, runtime error, or suspicious empty/all-null output).
-2. Identify the cause of the error or unexpected result.
-3. Produce a corrected single read-only SELECT SQL query in the target dialect.
+1. Inspect the original business question, linked schema, previous SQL query, and execution feedback (syntax/gateway error, runtime error, suspicious empty/all-null output, or successful non-empty output).
+2. If feedback reports an error or suspicious empty/all-null output: identify the cause and produce a corrected single read-only SELECT SQL query in the target dialect.
+3. If feedback reports successful non-empty output: perform a systematic POST_EXECUTION_SANITY_AUDIT before deciding whether to change SQL. You must audit the SQL for silent caliber bugs even when the gateway returns rows:
+   (a) LEFT JOIN right-table predicates placed in WHERE that silently drop unmatched rows; move to ON or pre-filtered CTE.
+   (b) Window functions LAST_VALUE/FIRST_VALUE/SUM/COUNT with ORDER BY but missing explicit ROWS BETWEEN ... AND ... frames.
+   (c) Unexcluded test/drill/abnormal/brush traffic or gross-vs-clean metric mixups; map explicit formal/external/non-test wording to the probed test-account exclusion flag (= 0) and risk-normal wording to the probed abnormal flag.
+   (d) SCD2 date-format mismatches (yyyyMMdd vs yyyy-MM-dd), multi-fact join fanout from unaggregated child-grain facts, or duplicate SCD2 versions matched by overlapping intervals.
+   (e) Cent/bps vs yuan scaling unapplied or applied multiple times, and unapplied probed code values.
+   If the audit finds zero defects, return the original SQL unchanged in the `sql` field.
+
+When diagnosing and repairing, apply the same ODPS/MaxCompute dialect, partition-pruning, grain, join, SCD2, unit/rate, status/code, window, and final-select rules as DialectSQLGenerator. In particular:
+- Never reintroduce unsupported functions such as DATE_FORMAT, DATE_ADD, DATE_SUB, UNIX_TIMESTAMP, FROM_UNIXTIME, IFNULL, JSON_EXTRACT, STRFTIME, UNNEST, or 2-argument DATEDIFF.
+- Use ODPS functions: MAX_PT('<table_name>'), TO_CHAR(datetime, fmt), TO_DATE(str, fmt), DATEADD(datetime, delta, 'dd'), DATEDIFF(end_date, start_date, 'dd'), NVL/COALESCE, GET_JSON_OBJECT(col, '$.key'), LATERAL VIEW [OUTER] explode(arr) t AS col.
+- Partition pruning: For latest partition use MAX_PT('<table_name>'); never use ds = (SELECT MAX(ds) FROM table). For explicit ranges, use literal partition predicates in the table's partition format (usually yyyyMMdd) and do not wrap partition columns in functions.
+- Grain and join: Determine the grain of each fact table; if child-grain relative to the required aggregation or join grain, pre-aggregate in a CTE using SUM for monetary/quantity facts and COUNT(DISTINCT business_key) for counts. For LEFT JOINs, keep primary facts on the left and put all right-table filters inside ON or pre-filtered CTE; use COALESCE/NVL for missing metrics.
+- SCD2 point-in-time: Normalize event_date and dimension effective dates to the same format before comparing (e.g., TO_CHAR(TO_DATE(date_str,'yyyymmdd'),'yyyy-mm-dd') or reverse). Use event_date >= eff_start_dt AND event_date <= eff_end_dt with inclusive bounds; treat open-ended eff_end_dt, null, or empty strings as '9999-12-31'; do not rely solely on is_current=1. If multiple dimension versions match the key/interval, deduplicate with ROW_NUMBER() OVER (PARTITION BY key ORDER BY deterministic priority) and filter rn = 1.
+- Unit and rate scaling: If schema_notes indicate cent/fen columns, convert to yuan with /100.0. If basis points, convert with /10000.0. Apply conversions exactly once and only round when final output requires specific precision.
+- Status/code filters: Use only filters explicitly implied by the question or documented in schema_notes/probe observations. Apply probed test/external exclusion flags when wording requires formal/external/non-test. Do not add lifecycle status, channel, operating-status, reversed, or valid-state predicates unless the question explicitly asks for that state. Do not add IS NOT NULL or <>'' on grouping dimensions unless required.
+- Window functions: For cumulative/rolling LAST_VALUE/FIRST_VALUE/SUM/COUNT over ORDER BY, always use an explicit frame such as ROWS BETWEEN ... AND ...; for deduplication use ROW_NUMBER.
 
 Output strictly valid JSON with exact keys:
-{"thought": "analysis of what went wrong", "diagnosis": "concise root-cause summary", "sql": "SELECT ..."}
+{"thought": "analysis of what went wrong and audit result", "diagnosis": "concise root-cause summary", "sql": "SELECT ..."}
 
 Only reference tables and columns from the provided schema.
```

## 7. 高质量 SFT/RL 验证轨迹池与进化收益分析

- 去重后验证通过总轨迹数 (`train` + `val`, 严格排除 `test`): **`552`** 条 (GRPO 训练 Prompt 数: `72`)
- 按数据集划分分布: `{"train": 179, "val": 373}`
- 按数仓错题类别分布: `{"fanout_dedup": 57, "json_array_unnest": 62, "partition_pruning": 51, "ratio_null_handling": 69, "retention_date_math": 60, "scd2_temporal_join": 90, "semantic_linking_trap": 79, "silent_caliber_audit": 84}`

| 轨迹类型 (`trajectory_type`) | 去重样本数 | 占比 | 平均推理步数 | 平均消耗 Tokens | 平均端到端延迟 |
|---|---:|---:|---:|---:|---:|
| `first_pass_correct` | 494 | 89.5% | 2.88 | 12888 | 60317.4ms |
| `self_repaired_after_error` | 58 | 10.5% | 3.50 | 13844 | 73072.5ms |
