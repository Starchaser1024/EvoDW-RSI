"""Unified dual-layer Warehouse Sandbox exposing run_sql, inspect_table_sample, evaluate_sql, and compute_reward."""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
import math
from typing import Any, Literal

from src.env.gateway_linter import GatewayLinter
from src.env.local_executor import LocalDuckDBExecutor
from src.env.schema import WarehouseCatalog
from src.eval.comparator import compare_result_sets

DialectType = Literal["odps", "hive"]

_SHARED_IMMUTABLE_SQL_CACHE: dict[str, OrderedDict[tuple[str, str], dict[str, Any]]] = {}
_SHARED_IMMUTABLE_SAMPLE_CACHE: dict[str, dict[tuple[str, tuple[str, ...] | None, int], dict[str, Any]]] = {}


@dataclass
class EvalOutcome:
    """Standardized evaluation result returned by WarehouseSandbox.evaluate_sql."""

    ex_match: bool
    gateway_passed: bool
    error_stage: str  # none | gateway_ast | runtime
    error_msg: str
    pred_rows: list[tuple[Any, ...]]
    pred_columns: list[str]
    gold_rows: list[tuple[Any, ...]]
    gold_columns: list[str]
    latency_ms: float
    transpiled_sql: str = ""
    diff_reason: str = ""


class WarehouseSandbox:
    """Two-layer local warehouse sandbox (Gateway AST Linter + DuckDB Execution Engine)."""

    def __init__(
        self,
        catalog: WarehouseCatalog,
        executor: LocalDuckDBExecutor,
        default_dialect: DialectType = "odps",
        *,
        immutable_cache_tag: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.executor = executor
        self.default_dialect: DialectType = default_dialect
        self._linters: dict[DialectType, GatewayLinter] = {
            "odps": GatewayLinter(catalog=catalog, dialect="odps"),
            "hive": GatewayLinter(catalog=catalog, dialect="hive"),
        }
        if immutable_cache_tag:
            self._sql_cache = _SHARED_IMMUTABLE_SQL_CACHE.setdefault(immutable_cache_tag, OrderedDict())
            self._sample_cache = _SHARED_IMMUTABLE_SAMPLE_CACHE.setdefault(immutable_cache_tag, {})
        else:
            self._sql_cache = OrderedDict()
            self._sample_cache = {}
        self._sql_cache_max: int = 4096

    def run_sql(
        self,
        sql: str,
        dialect: DialectType | None = None,
        timeout_sec: float = 3.0,
    ) -> dict[str, Any]:
        """Run a query through Layer 1 (Gateway AST Linter) and Layer 2 (DuckDB Executor)."""
        active_dialect: DialectType = dialect or self.default_dialect
        cache_key = (active_dialect, sql)
        if timeout_sec >= 1.0 and cache_key in self._sql_cache:
            self._sql_cache.move_to_end(cache_key)
            cached = self._sql_cache[cache_key]
            return {
                **cached,
                "columns": list(cached["columns"]),
                "rows": list(cached["rows"]),
            }

        linter = self._linters[active_dialect]
        lint_res = linter.lint(sql)
        if not lint_res.passed:
            out_fail = {
                "gateway_passed": False,
                "error_stage": "gateway_ast",
                "error_msg": lint_res.error_msg,
                "error_code": lint_res.error_code,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "latency_ms": 0.0,
                "transpiled_sql": "",
            }
            if timeout_sec >= 1.0:
                self._sql_cache[cache_key] = out_fail
                if len(self._sql_cache) > self._sql_cache_max:
                    self._sql_cache.popitem(last=False)
            return {**out_fail, "columns": [], "rows": []}

        exec_res = self.executor.execute_warehouse_sql(
            sql=sql,
            dialect=active_dialect,
            timeout_sec=timeout_sec,
        )
        out_ok = {
            "gateway_passed": True,
            "error_stage": exec_res["error_stage"],
            "error_msg": exec_res["error_msg"],
            "error_code": "",
            "columns": exec_res["columns"],
            "rows": exec_res["rows"],
            "row_count": exec_res["row_count"],
            "latency_ms": exec_res["latency_ms"],
            "transpiled_sql": exec_res["transpiled_sql"],
        }
        if timeout_sec >= 1.0:
            self._sql_cache[cache_key] = out_ok
            if len(self._sql_cache) > self._sql_cache_max:
                self._sql_cache.popitem(last=False)
        return {
            **out_ok,
            "columns": list(out_ok["columns"]),
            "rows": list(out_ok["rows"]),
        }

    def inspect_table_sample(
        self,
        table_name: str,
        columns: list[str] | None = None,
        sample_rows: int = 3,
    ) -> dict[str, Any]:
        """Probe table partition availability, column value distributions, and sample rows."""
        meta = self.catalog.get(table_name)
        if meta is None:
            return {"found": False, "error": f"Table '{table_name}' does not exist in catalog."}

        norm_cols_key = tuple(sorted({col.lower() for col in columns})) if columns else None
        sample_key = (meta.name, norm_cols_key, max(1, int(sample_rows)))
        cached_sample = self._sample_cache.get(sample_key)
        if cached_sample is not None:
            return copy.deepcopy(cached_sample)

        pcol = meta.partition_col
        selected_cols = [
            c.name
            for c in meta.columns
            if not columns or c.name.lower() in {col.lower() for col in columns}
        ]
        if not selected_cols:
            selected_cols = meta.column_names()

        partition_summary: dict[str, Any] = {"partition_col": pcol}
        latest_pt = None
        if pcol:
            pt_res = self.executor.execute_raw_duckdb(
                f"SELECT MIN({pcol}), MAX({pcol}), COUNT(DISTINCT {pcol}) FROM {meta.name}"
            )
            if pt_res["ok"] and pt_res["rows"]:
                min_pt, max_pt, cnt_pt = pt_res["rows"][0]
                latest_pt = max_pt
                partition_summary.update(
                    {
                        "min_partition": min_pt,
                        "max_partition": max_pt,
                        "partition_count": cnt_pt,
                    }
                )
            recent_res = self.executor.execute_raw_duckdb(
                f"SELECT DISTINCT {pcol} FROM {meta.name} ORDER BY {pcol} DESC LIMIT 5"
            )
            if recent_res["ok"]:
                partition_summary["recent_partitions"] = [r[0] for r in recent_res["rows"]]

        where_pt = f"WHERE {pcol} = '{latest_pt}'" if (pcol and latest_pt is not None) else ""
        col_str = ", ".join(selected_cols)
        sample_res = self.executor.execute_raw_duckdb(
            f"SELECT {col_str} FROM {meta.name} {where_pt} LIMIT {max(1, int(sample_rows))}"
        )

        # Probe compact value summaries / enum examples / JSON examples per column.
        col_profiles: dict[str, Any] = {}
        for col_meta in meta.columns:
            if col_meta.name not in selected_cols:
                continue
            cname = col_meta.name
            prof: dict[str, Any] = {"type": col_meta.data_type, "comment": col_meta.comment}
            dtype_upper = col_meta.data_type.upper()
            if dtype_upper == "VARCHAR" and not col_meta.is_partition:
                enum_q = (
                    f"SELECT DISTINCT {cname} FROM {meta.name} "
                    f"{where_pt + ' AND ' if where_pt else 'WHERE '} {cname} IS NOT NULL AND {cname} != '' "
                    f"ORDER BY {cname} ASC LIMIT 6"
                )
                enum_r = self.executor.execute_raw_duckdb(enum_q)
                if enum_r["ok"]:
                    vals = [str(r[0])[:160] for r in enum_r["rows"]]
                    prof["sample_non_empty_values"] = vals
                null_q = (
                    f"SELECT SUM(CASE WHEN {cname} IS NULL THEN 1 ELSE 0 END), "
                    f"SUM(CASE WHEN {cname} = '' THEN 1 ELSE 0 END), COUNT(*) "
                    f"FROM {meta.name} {where_pt}"
                )
                null_r = self.executor.execute_raw_duckdb(null_q)
                if null_r["ok"] and null_r["rows"]:
                    n_null, n_empty, n_tot = null_r["rows"][0]
                    prof["latest_pt_nulls"] = int(n_null or 0)
                    prof["latest_pt_empty_strings"] = int(n_empty or 0)
                    prof["latest_pt_rows"] = int(n_tot or 0)
            elif any(num_t in dtype_upper for num_t in ("INT", "DOUBLE", "DECIMAL", "FLOAT")) and not col_meta.is_partition:
                num_q = (
                    f"SELECT MIN({cname}), MAX({cname}), ROUND(AVG(CAST({cname} AS DOUBLE)), 2), "
                    f"SUM(CASE WHEN {cname} IS NULL THEN 1 ELSE 0 END), COUNT(*) "
                    f"FROM {meta.name} {where_pt}"
                )
                num_r = self.executor.execute_raw_duckdb(num_q)
                if num_r["ok"] and num_r["rows"]:
                    vmin, vmax, vavg, n_null, n_tot = num_r["rows"][0]
                    prof["numeric_range"] = {
                        "min": _json_safe(vmin),
                        "max": _json_safe(vmax),
                        "avg": _json_safe(vavg),
                    }
                    prof["latest_pt_nulls"] = int(n_null or 0)
                    prof["latest_pt_rows"] = int(n_tot or 0)
            col_profiles[cname] = prof

        serializable_rows = [
            {col: _json_safe(val) for col, val in zip(sample_res["columns"], row)}
            for row in sample_res["rows"]
        ]

        result = {
            "found": True,
            "table_name": meta.name,
            "layer": meta.layer,
            "partition_info": partition_summary,
            "column_profiles": col_profiles,
            "sample_rows": serializable_rows,
        }
        self._sample_cache[sample_key] = result
        return copy.deepcopy(result)

    def evaluate_sql(
        self,
        pred_sql: str,
        task_or_gold: Any,
        dialect: DialectType | None = None,
    ) -> EvalOutcome:
        """Execute predicted SQL and compare against Gold SQL or a BenchmarkTask."""
        active_dialect: DialectType = dialect or self.default_dialect
        pred_res = self.run_sql(pred_sql, dialect=active_dialect)

        ordered = False
        gold_rows: list[tuple[Any, ...]] = []
        gold_cols: list[str] = []

        if isinstance(task_or_gold, str):
            gold_res = self.run_sql(task_or_gold, dialect=active_dialect)
            gold_rows = gold_res["rows"]
            gold_cols = gold_res["columns"]
        else:
            ordered = bool(getattr(task_or_gold, "ordered", False))
            cached_rows = getattr(task_or_gold, "gold_rows", None)
            cached_cols = getattr(task_or_gold, "gold_columns", None)
            if cached_rows is not None and cached_cols is not None:
                gold_rows = cached_rows
                gold_cols = cached_cols
            else:
                gold_sql = getattr(task_or_gold, "gold_sql", "")
                gold_res = self.run_sql(gold_sql, dialect=active_dialect)
                gold_rows = gold_res["rows"]
                gold_cols = gold_res["columns"]

        if not pred_res["gateway_passed"] or pred_res["error_stage"] != "none":
            return EvalOutcome(
                ex_match=False,
                gateway_passed=pred_res["gateway_passed"],
                error_stage=pred_res["error_stage"],
                error_msg=pred_res["error_msg"],
                pred_rows=[],
                pred_columns=[],
                gold_rows=gold_rows,
                gold_columns=gold_cols,
                latency_ms=pred_res["latency_ms"],
                transpiled_sql=pred_res.get("transpiled_sql", ""),
                diff_reason=f"Execution failed at {pred_res['error_stage']}: {pred_res['error_msg']}",
            )

        matched, reason = compare_result_sets(
            pred_rows=pred_res["rows"],
            gold_rows=gold_rows,
            ordered=ordered,
        )
        return EvalOutcome(
            ex_match=matched,
            gateway_passed=True,
            error_stage="none",
            error_msg="",
            pred_rows=pred_res["rows"],
            pred_columns=pred_res["columns"],
            gold_rows=gold_rows,
            gold_columns=gold_cols,
            latency_ms=pred_res["latency_ms"],
            transpiled_sql=pred_res.get("transpiled_sql", ""),
            diff_reason=reason,
        )

    def compute_reward(
        self,
        pred_sql: str,
        task_or_gold: Any,
        refine_turns: int = 0,
        first_pass_gateway: bool | None = None,
        dialect: DialectType | None = None,
    ) -> dict[str, float]:
        """Standardized scalar & shaped RL reward breakdown reusable for SFT/GRPO."""
        outcome = self.evaluate_sql(pred_sql, task_or_gold, dialect=dialect)
        ex_score = 1.0 if outcome.ex_match else 0.0
        gw_score = 0.2 if outcome.gateway_passed else 0.0
        exec_score = 0.1 if outcome.error_stage == "none" else 0.0
        first_pass_bonus = 0.1 if (first_pass_gateway if first_pass_gateway is not None else (outcome.gateway_passed and refine_turns == 0)) else 0.0
        step_penalty = 0.03 * max(0, refine_turns)

        if outcome.ex_match:
            total_reward = 1.0 + first_pass_bonus - step_penalty
        else:
            total_reward = gw_score + exec_score - step_penalty

        return {
            "total_reward": round(total_reward, 4),
            "ex_reward": ex_score,
            "gateway_reward": gw_score,
            "exec_reward": exec_score,
            "first_pass_bonus": first_pass_bonus,
            "step_penalty": round(step_penalty, 4),
        }

    @staticmethod
    def is_suspicious_result(exec_res: dict[str, Any]) -> tuple[bool, str]:
        """Detect empty results or all-NULL metric rows that warrant GatewayRefiner check."""
        if not exec_res.get("gateway_passed", False):
            return True, exec_res.get("error_msg", "Gateway check failed.")
        if exec_res.get("error_stage", "none") != "none":
            return True, exec_res.get("error_msg", "Runtime execution error.")
        rows = exec_res.get("rows") or []
        if len(rows) == 0:
            return True, "Query executed cleanly but returned an empty result set (0 rows)."
        # Check if every value in every row is None or NaN.
        all_null = all(
            (val is None or (isinstance(val, float) and math.isnan(val)))
            for row in rows
            for val in row
        )
        if all_null:
            return True, "Query returned rows where every projected column is NULL/NaN."
        return False, ""

    def close(self) -> None:
        self.executor.close()


def _json_safe(val: Any) -> Any:
    if val is None or isinstance(val, (str, int, bool)):
        return val
    if isinstance(val, float):
        return None if (math.isnan(val) or math.isinf(val)) else round(val, 6)
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    return str(val)
