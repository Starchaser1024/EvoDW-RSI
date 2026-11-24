# EvoDW-RSI 实验报告（数仓 Text-to-SQL 策略自进化评测）

- 生成时间: `2026-11-17 23:50:53`
- 主优化方法: `gepa` | 目标方言: `HIVE`
- 优化 Rollout 预算 / 实际消耗: `96 / 105`
- 分层验证集规模 (Two-Stage Active Val): `12 题` (评估缓存命中复用: `163` 次)
- 选出最优策略 ID: `gepa_g5` (lineage parents: `['gepa_g4']`)

## 1. 进化前后核心指标总览（Seed vs Evolved）

| 评测集 | 指标 | Seed Baseline | Evolved Best | Δ 提升 / 变化 |
|---|---|---:|---:|---:|
| Val (12题) | **EX (执行准确率)** | 66.7% | **100.0%** | +33.3% |
| Val (12题) | **First-Pass Gateway Rate** | 75.0% | **100.0%** | +25.0% |
| Val (12题) | Invalid / Error SQL Rate | 16.7% | 0.0% | -16.7% |
| Val (12题) | Avg Steps / Avg Tokens / Avg Latency | 3.00 / 11454 / 51864.8ms | 4.00 / 22525 / 49691.2ms | +96.7% tok / -4.2% lat |
| Test (18题) | **EX (执行准确率)** | 44.4% | **83.3%** | +38.9% |
| Test (18题) | **First-Pass Gateway Rate** | 77.8% | **100.0%** | +22.2% |
| Test (18题) | Invalid / Error SQL Rate | 16.7% | 0.0% | -16.7% |
| Test (18题) | First-Pass Correct / Self-Repaired 题数 | 8 / 0 | **13** / 2 | +5 / +2 |
| Test (18题) | Avg Steps / Avg Tokens / Avg Latency | 3.17 / 15706 / 83076.1ms | 4.00 / 25754 / 179985.3ms | +64.0% tok / +116.7% lat |

## 2. 消融对比表（`seed` vs `greedy` vs `gepa`）

| Method | Opt Rollouts | Val EX | Test EX | Test First-Pass GW | Test Invalid Rate | Avg Steps | Avg Tokens | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed` | 0 | 66.7% | **44.4%** | 77.8% | 16.7% | 3.17 | 15706 | 83076.1ms |
| `gepa` | 96 | 100.0% | **83.3%** | 100.0% | 0.0% | 4.00 | 25754 | 179985.3ms |

## 3. 五大数仓高频错题分项表现（Test Set）

| 题型类别 | 题数 | Seed EX | Evolved EX | Δ EX | Seed 1st-Pass GW | Evolved 1st-Pass GW |
|---|---:|---:|---:|---:|---:|---:|
| `scd2_temporal_join` | 6 | 33.3% | **66.7%** | +33.3% | 100.0% | 100.0% |
| `semantic_linking_trap` | 6 | 33.3% | **100.0%** | +66.7% | 33.3% | 100.0% |
| `silent_caliber_audit` | 6 | 66.7% | **83.3%** | +16.7% | 100.0% | 100.0% |

## 4. 进化代数轨迹与谱系演化（Generation Curve & Lineage）

| Gen | Policy ID | Parents | Op | Target Module | Smoke Gate | Active-Val EX | 1st-Pass GW | Frontier Size | Rollouts | Diagnosis / Rationale |
|---:|---|---|---|---|---|---|---:|---:|---:|---|
| 0 | `seed_v0` | `-` | `seed` | `seed` | PASS | 66.7% | 75.0% | 1 | 12 | Initial seed policy baseline on stratified val (12 tasks). |
| 1 | `gepa_g1` | `seed_v0` | `mutate` | `linker` | PASS | 91.7% | 100.0% | 2 | 30 | WarehouseLinker failed to (1) proactively probe partition metadata and opaque code columns for HIVE latest-partition que |
| 2 | `gepa_g2` | `seed_v0` | `mutate` | `generator` | PASS | 66.7% | 83.3% | 2 | 48 | The generator repeatedly produced empty or undercounted results because it compared SCD2 date columns with inconsistent  |
| 3 | `gepa_g3` | `seed_v0` | `mutate` | `refiner` | PASS | 75.0% | 83.3% | 2 | 66 | Initial SQL returned empty primarily due to date/partition format mismatch, but the refiner misattributed the empty resu |
| 4 | `gepa_g4` | `gepa_g2, gepa_g3` | `merge` | `merge` | PASS | 66.7% | 100.0% | 2 | 78 | Merged by retaining Candidate A's evolved HIVE-compliant generator rules (authoritative schema_notes/probe usage, partit |
| 5 | `gepa_g5` | `gepa_g4` | `mutate` | `linker` | PASS | 100.0% | 100.0% | 1 | 96 | The linker inferred an unrequested lifecycle/status predicate from vague business wording ('formal external commercial o |

### Pareto 前沿池与胜频采样权重

| Frontier Policy ID | Active-Val EX | First-Pass GW | Eval Tasks | Win-Frequency Sampling Weight |
|---|---:|---:|---:|---:|
| `gepa_g5` | 100.0% | 100.0% | 12 | 1.000 |

## 5. 双模型成本与 Token 统计

- **Actor**: calls=`455`, prompt_tokens=`1345906`, completion_tokens=`883065`, total_tokens=`2228971`, avg_latency=`24346.1ms`
- **Reflector**: calls=`5`, prompt_tokens=`26235`, completion_tokens=`34147`, total_tokens=`60382`, avg_latency=`91409.0ms`

## 6. 策略进化前后三大模块 Prompt 关键规则 Diff

### 6.1 `WarehouseLinker` Prompt Diff
```diff
--- seed_v0/linker_system.txt
+++ evolved/linker_system.txt
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

### 6.2 `DialectSQLGenerator` Prompt Diff
```diff
--- seed_v0/generator_system.txt
+++ evolved/generator_system.txt
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

### 6.3 `GatewayRefiner` Prompt Diff
```diff
--- seed_v0/refiner_system.txt
+++ evolved/refiner_system.txt
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

## 7. 高质量 SFT/RL 验证轨迹池与进化收益分析

- 去重后验证通过总轨迹数 (`train` + `val`, 严格排除 `test`): **`624`** 条 (GRPO 训练 Prompt 数: `72`)
- 按数据集划分分布: `{"train": 196, "val": 428}`
- 按数仓错题类别分布: `{"fanout_dedup": 57, "json_array_unnest": 62, "partition_pruning": 51, "ratio_null_handling": 69, "retention_date_math": 60, "scd2_temporal_join": 110, "semantic_linking_trap": 106, "silent_caliber_audit": 109}`

| 轨迹类型 (`trajectory_type`) | 去重样本数 | 占比 | 平均推理步数 | 平均消耗 Tokens | 平均端到端延迟 |
|---|---:|---:|---:|---:|---:|
| `first_pass_correct` | 553 | 88.6% | 2.93 | 13072 | 59409.5ms |
| `self_repaired_after_error` | 71 | 11.4% | 3.54 | 15051 | 76459.4ms |
