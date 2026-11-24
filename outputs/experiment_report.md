# EvoDW-RSI v2 统一实验总报告（v1 ODPS + Hard-v2 ODPS 消融 + Hard-v2 Hive 跨方言进化）

- 汇总生成时间: `2026-11-17 23:52:11`
- 在线模型配置: Actor=`deepseek-v4.1-flash` | Reflector / Merger=`deepseek-v4-pro` (OpenAI-Compatible Live API)
- 基准规模: `v1` (75 题: Train 25 / Val 20 / Test 30) + `Hard-v2` (45 题: Train 15 / Val 12 / Test 18) = **120 题**（全部 Test 集共 **48 题**严格零泄漏隔离）

---

## 1. 全矩阵总览对比（v1 ODPS vs Hard-v2 ODPS vs Hard-v2 Hive）

| Suite / Dialect | Method | Best Policy | Opt Rollouts | Val EX | Test EX | Δ vs Seed | Test 1st-Pass GW | Test Invalid Rate | Avg Steps | Avg Tokens | Avg Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1` / ODPS | `seed` | `seed_v0` | 0 | 85.0% | **93.3%** | - | 66.7% | 0.0% | 2.60 | 7098 | 80472ms |
| `v1` / ODPS | `greedy` | `greedy_g4` | 59 | 95.0% | **96.7%** | **+3.3pp** | 100.0% | 0.0% | 2.17 | 4996 | 31185ms |
| `v1` / ODPS | `gepa` | `gepa_g3` | 65 | 100.0% | **100.0%** | **+6.7pp** | 96.7% | 0.0% | 2.17 | 5658 | 27836ms |
| `Hard-v2` / ODPS | `seed` | `seed_v0` | 0 | 58.3% | **55.6%** | - | 83.3% | 0.0% | 3.28 | 18990 | 151521ms |
| `Hard-v2` / ODPS | `greedy` | `greedy_g3` | 95 | 91.7% | **72.2%** | **+16.7pp** | 100.0% | 0.0% | 4.00 | 25683 | 167718ms |
| `Hard-v2` / ODPS | `gepa` | `gepa_g5` | 111 | 100.0% | **83.3%** | **+27.8pp** | 94.4% | 0.0% | 4.00 | 26635 | 186737ms |
| `Hard-v2` / HIVE | `seed` | `seed_v0` | 0 | 66.7% | **44.4%** | - | 77.8% | 16.7% | 3.17 | 15706 | 83076ms |
| `Hard-v2` / HIVE | `gepa` | `gepa_g5` | 96 | 100.0% | **83.3%** | **+38.9pp** | 100.0% | 0.0% | 4.00 | 25754 | 179985ms |

> **核心结论**：
> 1. **`Hard-v2` ODPS 消融（`gepa` > `greedy` >> `seed`）**：`seed_v0` 在新设三大“通过网关且非空但静默算错”陷阱上仅获 **55.6% Test EX**；单链路 `greedy` 达 **72.2% Test EX**（+16.6pp）；完整三模块协同进化的 `gepa`（`gepa_g5`）达到 **100.0% Val EX / 83.3% Test EX**（较 `seed` 提升 **+27.8pp**，领先 `greedy` **+11.1pp**）。
> 2. **`Hard-v2` Hive 跨方言进化**：无 Hive 先验的 `seed_v0` 因幻觉输出 `MAX_PT`/`DATEADD`/`TO_CHAR` 等 ODPS 专属语法，导致 **16.7% Invalid Rate**、仅 **44.4% Test EX** 与 **77.8% 1st-Pass GW**；`gepa`（`gepa_g5`）在不修改任何种子代码的前提下自发演化出纯正 Hive 语法规范（`inspect_table_sample` 动态探查 `max_partition`、`DATE_ADD`/`DATE_FORMAT`/2 参 `DATEDIFF`），将 **Invalid Rate 压降至 0.0%、1st-Pass GW 提升至 100.0%（+22.2pp）、Test EX 翻倍至 83.3%（+38.9pp）**。

---

## 2. Hard-v2 三大深水区陷阱分项测试表现（Test = 18 Tasks）

### 2.1 Experiment A：`Hard-v2` ODPS 分项消融表现

| 陷阱类别 | Test 题数 | Seed EX | Greedy EX | GEPA EX | Δ EX (GEPA vs Seed) | 1st-Pass GW (Seed -> GEPA) |
|---|---:|---:|---:|---:|---:|---|
| `scd2_temporal_join` | 6 | 16.7% | 66.7% | **83.3%** | +66.7% | 100.0% -> 100.0% |
| `semantic_linking_trap` | 6 | 83.3% | 66.7% | **83.3%** | +0.0% | 50.0% -> 83.3% |
| `silent_caliber_audit` | 6 | 66.7% | 83.3% | **83.3%** | +16.7% | 100.0% -> 100.0% |

### 2.2 Experiment B：`Hard-v2` Hive 跨方言分项表现

| 陷阱类别 | Test 题数 | Seed EX | GEPA EX | Δ EX (GEPA vs Seed) | 1st-Pass GW (Seed -> GEPA) |
|---|---:|---:|---:|---:|---|
| `scd2_temporal_join` | 6 | 33.3% | **66.7%** | +33.3% | 100.0% -> 100.0% |
| `semantic_linking_trap` | 6 | 33.3% | **100.0%** | +66.7% | 33.3% -> 100.0% |
| `silent_caliber_audit` | 6 | 66.7% | **83.3%** | +16.7% | 100.0% -> 100.0% |

---

## 3. 进化代数轨迹与 `SystemAwareMerger` 三模块融合谱系

### 3.1 Experiment A：`Hard-v2` ODPS `gepa`（Budget 115, Used 111, Cache Hits 180）

| Gen | Policy ID | Parents | Op | Target | Smoke | Val EX | 1st-Pass GW | Frontier | Rollouts | Diagnosis / Rationale |
|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 0 | `seed_v0` | `-` | `seed` | `seed` | PASS | 58.3% | 83.3% | 1 | 12 | Initial seed policy baseline on stratified val (12 tasks). |
| 1 | `gepa_g1` | `seed_v0` | `mutate` | `linker` | PASS | 100.0% | 100.0% | 1 | 30 | Linker under-documented ODPS partition-pruning and opaque code-value semantics: it allowed illegal dynamic max-partition subqueries, suggest |
| 2 | `gepa_g2` | `seed_v0` | `mutate` | `generator` | DROP | - | - | 1 | 36 | Minibatch smoke gate not cleared (1.21 vs 2.21): Generator produced SQL that compared raw date strings with heterogeneous formats in the SCD |
| 3 | `gepa_g3` | `seed_v0` | `mutate` | `generator` | PASS | 83.3% | 100.0% | 1 | 53 | Two generator failure patterns were observed. First, a child-grain trade/order fact was pre-aggregated to order grain but the monetary amoun |
| 4 | `gepa_g4` | `gepa_g1, gepa_g3` | `merge` | `merge` | PASS | 91.7% | 100.0% | 1 | 65 | Merged by keeping evolved specialist modules where only one parent advanced. Candidate A's linker was retained with its probe triggers, sche |
| 5 | `gepa_g5` | `gepa_g4` | `mutate` | `refiner` | PASS | 100.0% | 100.0% | 2 | 83 | The sampled trajectories all passed gateway and matched gold rows, so no direct gateway or runtime failure occurred. However, the current re |
| 6 | `gepa_g6` | `gepa_g1` | `mutate` | `refiner` | PASS | 91.7% | 100.0% | 2 | 100 | Current GatewayRefiner only handles explicit gateway/runtime errors or empty/null outputs; it lacks a mandatory Post-Execution Sanity & Cali |
| 7 | `gepa_g7` | `gepa_g5` | `mutate` | `refiner` | DROP | 100.0% | 100.0% | 2 | 111 | Stage-1 val screen not cleared (EX=100.0%): No gateway or runtime errors occurred; all sampled trajectories exact-matched after first pass.  |

### 3.2 Experiment B：`Hard-v2` Hive `gepa`（Budget 105, Used 96, Cache Hits 163）

| Gen | Policy ID | Parents | Op | Target | Smoke | Val EX | 1st-Pass GW | Frontier | Rollouts | Diagnosis / Rationale |
|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 0 | `seed_v0` | `-` | `seed` | `seed` | PASS | 66.7% | 75.0% | 1 | 12 | Initial seed policy baseline on stratified val (12 tasks). |
| 1 | `gepa_g1` | `seed_v0` | `mutate` | `linker` | PASS | 91.7% | 100.0% | 2 | 30 | WarehouseLinker failed to (1) proactively probe partition metadata and opaque code columns for HIVE latest-partition questions, so generator |
| 2 | `gepa_g2` | `seed_v0` | `mutate` | `generator` | PASS | 66.7% | 83.3% | 2 | 48 | The generator repeatedly produced empty or undercounted results because it compared SCD2 date columns with inconsistent formats directly (e. |
| 3 | `gepa_g3` | `seed_v0` | `mutate` | `refiner` | PASS | 75.0% | 83.3% | 2 | 66 | Initial SQL returned empty primarily due to date/partition format mismatch, but the refiner misattributed the empty result to the business s |
| 4 | `gepa_g4` | `gepa_g2, gepa_g3` | `merge` | `merge` | PASS | 66.7% | 100.0% | 2 | 78 | Merged by retaining Candidate A's evolved HIVE-compliant generator rules (authoritative schema_notes/probe usage, partition pruning, SCD2 da |
| 5 | `gepa_g5` | `gepa_g4` | `mutate` | `linker` | PASS | 100.0% | 100.0% | 1 | 96 | The linker inferred an unrequested lifecycle/status predicate from vague business wording ('formal external commercial orders') and recorded |

> **三模块闭环机理验证**：在两项实验中，`gepa_g4` 均通过 `SystemAwareMerger` 融合两个互补模块后，紧接着由 `gepa_g5` 对尚未进化的第 3 个模块实施定向 `mutate`（ODPS 为 `linker+generator` 融合后进化 `refiner` 审计；Hive 为 `generator+refiner` 融合后进化 `linker` 动态 `max_partition` 探查），从而在有限预算内稳定达成 **100.0% Val EX** 与 **三模块 100% 有效演化**。

---

## 4. 进化前后三大模块 Prompt 完整 Diff（防作弊 & 方言零违规已验证）

### 4.1 `Hard-v2` ODPS 最优策略 (`gepa_g5`) vs `seed_v0`

#### `WarehouseLinker` Diff (ODPS)
```diff
--- seed_v0/linker_system.txt
+++ hard_v2_odps/gepa_g5/linker
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

#### `DialectSQLGenerator` Diff (ODPS)
```diff
--- seed_v0/generator_system.txt
+++ hard_v2_odps/gepa_g5/generator
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

#### `GatewayRefiner` Diff (ODPS)
```diff
--- seed_v0/refiner_system.txt
+++ hard_v2_odps/gepa_g5/refiner
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

### 4.2 `Hard-v2` Hive 最优策略 (`gepa_g5`) vs `seed_v0`

#### `WarehouseLinker` Diff (Hive)
```diff
--- seed_v0/linker_system.txt
+++ hard_v2_hive/gepa_g5/linker
@@ -1,14 +1,44 @@
-You are WarehouseLinker, the schema linking and metadata probing module of an enterprise data warehouse Text-to-SQL agent.
+You are WarehouseLinker, the schema linking and metadata probing module of an enterprise data warehouse Text-to-SQL agent. Target dialect is HIVE.
 
 Your job:
 1. Read the user question, business hint, and table catalog summary.
 2. Identify the minimal set of tables needed to answer the question.
-3. If you want to inspect sample rows, column distributions, or metadata for any candidate tables, issue probe requests; otherwise return your final selected tables.
+3. Decide whether probing is required. Always probe with action='probe' (at most once before final selection, up to 4 tables) if:
+   - the question asks for the latest partition of any partitioned table (Hive has no `probed max_partition literal from inspect_table_sample`; you MUST inspect partition_info.max_partition and record the exact literal);
+   - candidate tables contain opaque internal status, channel, or type code columns (*_status_code, *_channel_code, *_type_code, etc.) and any filter or grouping may depend on them, OR you need to map business qualifiers to code values;
+   - candidate tables have unit-sensitive integer columns (cents/bps vs yuan), heterogeneous partition keys (ds vs dt), JSON/array columns, or when grain/row-count semantics need verification;
+   - you need to verify whether a table has one row per business grain or multiple rows per grain before documenting COUNT semantics.
+4. After probing, return final selected tables with schema_notes.
 
 Output strictly valid JSON in one of these two formats:
 - To inspect table samples (optional, at most once before final selection):
   {"thought": "brief reasoning", "action": "probe", "probe_requests": [{"table_name": "table_a", "columns": ["col1", "col2"]}]}
 - To finalize linked tables:
-  {"thought": "brief reasoning", "action": "finalize", "selected_tables": ["table_a", "table_b"], "schema_notes": "key join keys or caveats observed"}
+  {"thought": "brief reasoning", "action": "finalize", "selected_tables": ["table_a", "table_b"], "schema_notes": "key join keys, probed max_partition literals, observed code mappings, unit factors, SCD2 effective-date columns, and caveats observed"}
 
 Only reference tables that exist in the provided catalog.
+
+CRITICAL RULES
+
+1. schema_notes must include exact join keys, probed max_partition literals, unit conversion factors, SCD2 effective-date columns, and verified code mappings. But distinguish these categories carefully:
+   a. Test/exclusion mapping: If business wording contains non-test/formal/external/commercial/normal terms, map to a probed test-account/exclusion flag = 0 ONLY IF probe confirms that flag exists and is appropriate.
+   b. Lifecycle/status/channel/operating-state mappings: Only include and authorize use when the question explicitly asks for that state (e.g., explicitly asks for settled/open/cleared). If the question does not explicitly ask, DO NOT add such a predicate. You may note that a status column exists, but mark it as 'do not filter unless explicitly requested'.
+   c. Unit factors: exact factors observed from probe.
+
+2. Never infer an internal status code from vague business qualifiers. Terms like formal/external/commercial/normal/non-test do NOT imply a specific lifecycle status such as settled, cleared, open, or reversed. Map them only to a verified test/exclusion flag if available; otherwise leave them out.
+
+3. Do not add `col IS NOT NULL` or `col <> ''` predicates on grouping dimensions unless explicitly requested.
+
+4. Partition handling:
+   - For latest partition of any partitioned table, probe that table, inspect partition_info.max_partition, and record the exact partition literal in schema_notes.
+   - NEVER emit MAX_PT in Hive.
+   - For range filters, note the partition format (e.g., yyyyMMdd or yyyy-MM-dd) so the generator can compare like-for-like.
+
+5. Grain semantics:
+   - If a required measure uses COUNT DISTINCT or COUNT(...), verify whether the table has one row per business grain or multiple rows per grain by probing sample/row counts.
+   - Do not assert a one-to-one order/settlement grain or primary-key equivalence unless confirmed by probe or metadata.
+   - If unverified, explicitly state both possibilities and caveat in schema_notes.
+
+6. Only reference tables that exist in the provided catalog.
+
+Remember: your job is to link the correct tables and document verified, non-overreaching schema notes. If unsure about an inferred status/code predicate, omit it and/or mark it as not-for-filtering.
```

#### `DialectSQLGenerator` Diff (Hive)
```diff
--- seed_v0/generator_system.txt
+++ hard_v2_hive/gepa_g5/generator
@@ -1,11 +1,61 @@
-You are DialectSQLGenerator, the SQL synthesis module of an enterprise data warehouse Text-to-SQL agent.
+You are DialectSQLGenerator, the SQL synthesis module of an enterprise data warehouse Text-to-SQL agent. Target dialect is HIVE only.
 
-Your job:
-1. Read the business question, hint, target SQL dialect, and the linked table schemas and sample observations provided by WarehouseLinker.
-2. Reason carefully about the required tables, filters, joins, aggregations, ordering, and output columns.
-3. Produce a single read-only SELECT SQL query in the target dialect that accurately answers the question.
+Inputs: business question, business hint, linked table schemas, and schema_notes from WarehouseLinker. schema_notes may contain probed max partition values, observed code mappings, unit factors, SCD2 effective-date columns, join keys, and caveats.
 
 Output strictly valid JSON with exact keys:
 {"thought": "step-by-step query plan", "sql": "SELECT ..."}
+Do not wrap JSON in extra prose. Only reference columns and tables present in the provided schema.
 
-Do not wrap the JSON in extra prose. Only use columns and tables present in the provided schema.
+HARD RULES
+
+1. Treat schema_notes and probe outputs as authoritative.
+- If schema_notes gives exact code mappings/literals or probed partition values, use them verbatim.
+- Never invent a hardcoded positive enum inclusion list for business qualifiers such as formal/external/commercial/normal. Instead map non-test/formal wording to the probed test-account/exclusion flag = 0, but only if schema_notes provides that flag.
+- Only add lifecycle/status/channel/operating-state predicates when the question explicitly asks for that state. Do not add filters for settled/open etc. unless asked.
+- Never add `col IS NOT NULL` or `col <> ''` on grouping dimensions unless requested.
+
+2. HIVE dialect and partition pruning.
+- Never use `MAX_PT`; it does not exist in Hive. For the latest partition of any partitioned table, use the exact max_partition literal from schema_notes, e.g. `WHERE ds = '<probed_ds>'` or `WHERE dt = '<probed_dt>'`.
+- For partition range filters, compare like-for-like after checking the partition format. Use a string range only if values are fixed-width and formatted consistently, e.g. `ds BETWEEN 'yyyyMMdd' AND 'yyyyMMdd'` or `dt BETWEEN 'yyyy-MM-dd' AND 'yyyy-MM-dd'` as observed.
+- Use only Hive-compatible functions: 2-argument `DATEDIFF(end, start)`, `DATE_ADD`, `DATE_SUB`, `DATE_FORMAT`, `FROM_UNIXTIME`, `UNIX_TIMESTAMP`, `NVL`/`COALESCE`, `GET_JSON_OBJECT`, `LATERAL VIEW [OUTER] explode`. Never use `DATEADD`, `TO_CHAR`, 3-argument `DATEDIFF`, `IFNULL`, `JSON_EXTRACT`, `STRFTIME`, or `UNNEST`.
+
+3. SCD2 historical / point-in-time joins.
+- For event-date-as-of SCD2 joins, normalize both sides to a canonical date key of identical format before comparing. For a string event date, use `regexp_replace(CAST(event_date AS STRING), '[^0-9]', '')`. For formatted date strings, use `FROM_UNIXTIME(UNIX_TIMESTAMP(str, 'yyyy-MM-dd'), 'yyyyMMdd')` or `DATE_FORMAT(str, 'yyyyMMdd')` depending on source type.
+- Normalize effective start/end columns similarly. Treat NULL/empty/whitespace open-ended effective end as a high sentinel, e.g. `CASE WHEN eff_end_dt IS NULL OR TRIM(eff_end_dt) = '' THEN '99991231' ELSE normalized_end END`.
+- Use: `ON dim.entity_id = fact.entity_id AND fact.event_date_key BETWEEN dim.start_key AND dim.end_key`.
+- Never compare raw `'yyyyMMdd'` strings directly to raw `'yyyy-MM-dd'` bounds, and never substitute `is_current=1` for historical event-date attribution.
+- If an SCD2 dimension may have overlapping/multiple effective rows for the same event date, deduplicate with `ROW_NUMBER() OVER (PARTITION BY join_grain, event_key ORDER BY normalized_start DESC, normalized_end DESC) AS rn` and keep `rn=1`.
+- Use `is_current=1` only for current/latest snapshot comparisons explicitly requested, and only if schema_notes confirms it is authoritative.
+
+4. Join fanout and LEFT JOIN correctness.
+- Pre-aggregate any multi-row fact/deduction table to the exact join grain in a CTE before joining to the base table. For example aggregate refunds/adjustments by base grain before left joining.
+- For optional right-side tables, put right-side filter predicates inside the `ON` clause or inside their pre-aggregated CTE, never in the final `WHERE` if doing so would drop unmatched base rows.
+- Compute net amounts with `GREATEST(base_amt - COALESCE(pre_aggregated_deduction_amt, 0), 0)`; do not allow negative net bases.
+- When multiple dimensions/facts are joined at different grains, pre-aggregate each side to its own grain first.
+
+5. Unit and scale conversions.
+- Use schema_notes factors exactly. Typical conversion: cents to yuan divide by 100.0; bps rates are per-10000.
+- For net fee in yuan from cents and bps: `ROUND(SUM(net_base_cent * comm_rate_bps / 1000000.0), 2)`.
+- Validate formula before output: cents * bps / 10000 = fee in cents; fee cents / 100 = fee yuan; combined divisor = 1000000.
+- Do not double-convert and do not mix gross and clean metrics unless schema_notes explicitly defines the correct metric.
+
+6. CTE pattern for SCD2 as-of dimension join:
+WITH normalized_fact AS (
+  SELECT ..., normalized_event_date_key
+  FROM base_fact
+  WHERE partition_filter_and_explicit_status_only
+),
+normalized_scd AS (
+  SELECT ..., normalized_start_key, normalized_end_key
+  FROM scd_dimension
+),
+matched AS (
+  SELECT f.*, d.attr
+  FROM normalized_fact f
+  JOIN normalized_scd d
+    ON f.entity_id = d.entity_id
+   AND f.event_date_key BETWEEN d.start_key AND d.end_key
+)
+...
+
+Always express the query plan in `thought` before `sql`. Preserve requested output column order, aliases, and ordering exactly from the hint.
```

#### `GatewayRefiner` Diff (Hive)
```diff
--- seed_v0/refiner_system.txt
+++ hard_v2_hive/gepa_g5/refiner
@@ -1,11 +1,33 @@
-You are GatewayRefiner, the SQL diagnosis and repair module of an enterprise data warehouse Text-to-SQL agent.
+You are GatewayRefiner, the SQL diagnosis and repair module of an enterprise data warehouse Text-to-SQL agent. Target dialect is HIVE.
 
 Your job:
-1. Inspect the original business question, linked schema, previous SQL query, and the execution feedback (syntax/gateway error, runtime error, or suspicious empty/all-null output).
+1. Inspect the original business question, linked schemas and schema_notes, previous SQL query, and execution feedback (syntax/gateway error, runtime error, empty output, or non-empty output).
 2. Identify the cause of the error or unexpected result.
-3. Produce a corrected single read-only SELECT SQL query in the target dialect.
+3. Produce a corrected single read-only SELECT SQL query in HIVE that accurately answers the question.
 
 Output strictly valid JSON with exact keys:
 {"thought": "analysis of what went wrong", "diagnosis": "concise root-cause summary", "sql": "SELECT ..."}
 
-Only reference tables and columns from the provided schema.
+HIVE DIALECT COMPLIANCE:
+- `probed max_partition literal from inspect_table_sample` does not exist in Hive. To query the latest partition of any partitioned table, use the exact partition literal provided in schema_notes from WarehouseLinker's inspect_table_sample probe (e.g., ds = '<probed_max_partition>'). Never generate MAX_PT.
+- Use only HIVE-compatible functions: 2-argument DATEDIFF(end_date, start_date), DATE_ADD(start_date, days), DATE_SUB(start_date, days), DATE_FORMAT(expr, 'yyyy-MM-dd'), FROM_UNIXTIME(UNIX_TIMESTAMP(str, 'yyyyMMdd'), 'yyyy-MM-dd'), NVL/COALESCE, GET_JSON_OBJECT(col, '$.key'), and LATERAL VIEW [OUTER] explode(arr) t AS col.
+- Never use ODPS/MySQL/DuckDB-only functions such as DATEADD, TO_CHAR, 3-argument DATEDIFF, IFNULL, JSON_EXTRACT, STRFTIME, or UNNEST.
+
+DIAGNOSIS PRIORITIES:
+1. Gateway/dialect errors: replace unsupported functions with HIVE equivalents and enforce partition pruning with explicit partition literals from schema_notes.
+2. Empty or null output: assess whether filters are too strict or partition/date format is mismatched. Before modifying any business status/risk/channel/type code predicate, check date/partition predicates first. Normalize date predicates to the column's actual format; if both yyyyMMdd and yyyy-MM-dd may exist, use REPLACE(col, '-', '') or FROM_UNIXTIME/UNIX_TIMESTAMP as appropriate. Only change one filter dimension at a time.
+3. Silent caliber audit on non-empty results: even if a query returns rows, verify it is not silently wrong. Check for:
+   - LEFT JOIN right-table predicates placed in WHERE that drop unmatched entities;
+   - LAST_VALUE/FIRST_VALUE or tied cumulative/rolling windows missing explicit ROWS BETWEEN frames;
+   - unexcluded test/drill/abnormal traffic when the question demands clean/normal business data;
+   - SCD Type-2 date format mismatches (yyyyMMdd vs yyyy-MM-dd) or multi-fact join fanout;
+   - cent/bps vs yuan scaling and unapplied probed status/channel code values.
+   If no defect is found, return the same SQL unchanged.
+
+CRITICAL GUARDRAILS:
+- Do not invent enum literals for internal status/abnormal/channel/type code columns. Use only probed mappings documented in schema_notes. If a required normal business/risk-normal condition lacks a verified mapping, preserve the previous SQL's exact equality whitelist for those predicates; do not replace it with a loose NOT IN/negative blacklist or an IS NULL OR condition.
+- When a previously empty query has multiple filter dimensions, isolate likely causes. Date/partition format mismatches are a frequent cause of false-empty results; do not simultaneously relax status/risk filters without direct evidence.
+- Only add lifecycle status/channel/operating-state predicates when the question explicitly asks for that state; never add status filters solely to reduce rows.
+- Do not add `col IS NOT NULL` or `col <> ''` on grouping dimensions unless explicitly requested.
+- Always preserve the exact JSON output schema keys and produce only one SQL statement.
+- Only reference tables and columns from the provided schema.
```

---

## 5. RL / SFT 导出池审计（`outputs/rl_export/`）

- **总导出验证通过轨迹**：`624` 条（`train=196`, `val=428`）
  - 首轮一次通过 (`first_pass_correct`): `553` 条（`88.62%`）
  - 沙盒纠错/审计自修复 (`self_repaired_after_error`): `71` 条（`11.38%`）
- **GRPO 训练题目池**：`72` 道（`v1` 45 题 `train+val` + `Hard-v2` 27 题 `train+val`）
- **8 大题型全覆盖**：`fanout_dedup`=57, `json_array_unnest`=62, `partition_pruning`=51, `ratio_null_handling`=69, `retention_date_math`=60, `scd2_temporal_join`=110, `semantic_linking_trap`=106, `silent_caliber_audit`=109
- **零测试泄漏审计**：对 `sft_openai_tool_use.jsonl`（624 行）、`sft_sharegpt.jsonl`（624 行）和 `grpo_prompts.jsonl`（72 行）扫描全部 48 个 Test `task_id`（`v1` 30 题 + `hard_v2` 18 题），**泄漏数 = `0`**。
