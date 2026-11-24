"""Layer 2: Deterministic ODPS/Hive-to-DuckDB AST transpiler and safe local executor."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import duckdb
import sqlglot
from sqlglot import TokenType, exp

from src.env.schema import WarehouseCatalog

DialectType = Literal["odps", "hive"]

ODPS_MACRO_DDL = r"""
CREATE OR REPLACE MACRO odps_parse_ts(val) AS (
  CASE
    WHEN val IS NULL OR TRIM(CAST(val AS VARCHAR)) = '' THEN NULL
    ELSE COALESCE(
      try_cast(val AS TIMESTAMP),
      try_strptime(TRIM(CAST(val AS VARCHAR)), '%Y%m%d'),
      try_strptime(TRIM(CAST(val AS VARCHAR)), '%Y-%m-%d'),
      try_strptime(substr(TRIM(CAST(val AS VARCHAR)), 1, 19), '%Y-%m-%d %H:%M:%S')
    )
  END
);

CREATE OR REPLACE MACRO odps_fmt_to_strftime(fmt) AS (
  replace(
    replace(
      replace(
        replace(
          replace(
            replace(
              replace(lower(CAST(fmt AS VARCHAR)), 'yyyy', '%Y'),
            'hh24', '%H'),
          'hh', '%H'),
        'mi', '%M'),
      'ss', '%S'),
    'mm', '%m'),
  'dd', '%d')
);

CREATE OR REPLACE MACRO odps_to_date_1(val) AS (
  odps_parse_ts(val)
);

CREATE OR REPLACE MACRO odps_to_date(val, fmt) AS (
  CASE
    WHEN val IS NULL OR TRIM(CAST(val AS VARCHAR)) = '' THEN NULL
    WHEN fmt IS NULL THEN odps_parse_ts(val)
    ELSE COALESCE(
      try_strptime(TRIM(CAST(val AS VARCHAR)), odps_fmt_to_strftime(fmt)),
      odps_parse_ts(val)
    )
  END
);

CREATE OR REPLACE MACRO odps_to_char(val, fmt) AS (
  CASE
    WHEN val IS NULL THEN NULL
    ELSE strftime(odps_parse_ts(val), odps_fmt_to_strftime(fmt))
  END
);

CREATE OR REPLACE MACRO odps_datediff(d1, d2, unit) AS (
  CASE lower(CAST(unit AS VARCHAR))
    WHEN 'yyyy' THEN date_diff('year', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'yy'   THEN date_diff('year', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'year' THEN date_diff('year', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'mm'   THEN date_diff('month', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'mon'  THEN date_diff('month', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'month' THEN date_diff('month', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'dd'   THEN date_diff('day', CAST(odps_parse_ts(d2) AS DATE), CAST(odps_parse_ts(d1) AS DATE))
    WHEN 'day'  THEN date_diff('day', CAST(odps_parse_ts(d2) AS DATE), CAST(odps_parse_ts(d1) AS DATE))
    WHEN 'hh'   THEN date_diff('hour', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'hour' THEN date_diff('hour', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'mi'   THEN date_diff('minute', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'minute' THEN date_diff('minute', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'ss'   THEN date_diff('second', odps_parse_ts(d2), odps_parse_ts(d1))
    WHEN 'second' THEN date_diff('second', odps_parse_ts(d2), odps_parse_ts(d1))
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO hive_datediff(d1, d2) AS (
  date_diff('day', CAST(odps_parse_ts(d2) AS DATE), CAST(odps_parse_ts(d1) AS DATE))
);

CREATE OR REPLACE MACRO hive_date_add(d, delta) AS (
  CASE
    WHEN d IS NULL OR delta IS NULL THEN NULL
    ELSE strftime(CAST(odps_parse_ts(d) AS DATE) + to_days(CAST(delta AS INTEGER)), '%Y-%m-%d')
  END
);

CREATE OR REPLACE MACRO hive_date_sub(d, delta) AS (
  CASE
    WHEN d IS NULL OR delta IS NULL THEN NULL
    ELSE strftime(CAST(odps_parse_ts(d) AS DATE) - to_days(CAST(delta AS INTEGER)), '%Y-%m-%d')
  END
);

CREATE OR REPLACE MACRO hive_to_date_1(val) AS (
  CASE
    WHEN val IS NULL OR TRIM(CAST(val AS VARCHAR)) = '' THEN NULL
    ELSE strftime(odps_parse_ts(val), '%Y-%m-%d')
  END
);

CREATE OR REPLACE MACRO hive_unix_timestamp_1(val) AS (
  CASE
    WHEN val IS NULL OR TRIM(CAST(val AS VARCHAR)) = '' THEN NULL
    ELSE CAST(epoch(odps_parse_ts(val)) AS BIGINT)
  END
);

CREATE OR REPLACE MACRO hive_unix_timestamp_2(val, fmt) AS (
  CASE
    WHEN val IS NULL OR TRIM(CAST(val AS VARCHAR)) = '' THEN NULL
    ELSE CAST(epoch(odps_to_date(val, fmt)) AS BIGINT)
  END
);

CREATE OR REPLACE MACRO hive_from_unixtime_1(ts) AS (
  CASE
    WHEN ts IS NULL THEN NULL
    ELSE strftime(to_timestamp(CAST(ts AS DOUBLE)), '%Y-%m-%d %H:%M:%S')
  END
);

CREATE OR REPLACE MACRO hive_from_unixtime_2(ts, fmt) AS (
  CASE
    WHEN ts IS NULL THEN NULL
    ELSE strftime(to_timestamp(CAST(ts AS DOUBLE)), odps_fmt_to_strftime(fmt))
  END
);

CREATE OR REPLACE MACRO odps_dateadd(d, delta, unit) AS (
  odps_parse_ts(d) + (
    CASE lower(CAST(unit AS VARCHAR))
      WHEN 'yyyy' THEN to_years(CAST(delta AS INTEGER))
      WHEN 'yy'   THEN to_years(CAST(delta AS INTEGER))
      WHEN 'year' THEN to_years(CAST(delta AS INTEGER))
      WHEN 'mm'   THEN to_months(CAST(delta AS INTEGER))
      WHEN 'mon'  THEN to_months(CAST(delta AS INTEGER))
      WHEN 'month' THEN to_months(CAST(delta AS INTEGER))
      WHEN 'dd'   THEN to_days(CAST(delta AS INTEGER))
      WHEN 'day'  THEN to_days(CAST(delta AS INTEGER))
      WHEN 'hh'   THEN to_hours(CAST(delta AS BIGINT))
      WHEN 'hour' THEN to_hours(CAST(delta AS BIGINT))
      WHEN 'mi'   THEN to_minutes(CAST(delta AS BIGINT))
      WHEN 'minute' THEN to_minutes(CAST(delta AS BIGINT))
      WHEN 'ss'   THEN to_seconds(CAST(delta AS BIGINT))
      WHEN 'second' THEN to_seconds(CAST(delta AS BIGINT))
      ELSE NULL
    END
  )
);

CREATE OR REPLACE MACRO odps_get_json_object(json_str, path) AS (
  CASE
    WHEN json_str IS NULL OR TRIM(CAST(json_str AS VARCHAR)) = '' THEN NULL
    ELSE json_extract_string(TRY_CAST(json_str AS JSON), CAST(path AS VARCHAR))
  END
);
"""

_TOKEN_REWRITES_ODPS = {
    "DATEDIFF": "__ODPS_DATEDIFF",
    "DATEADD": "__ODPS_DATEADD",
    "TO_DATE": "__ODPS_TO_DATE",
    "TO_CHAR": "__ODPS_TO_CHAR",
    "GET_JSON_OBJECT": "__ODPS_GET_JSON_OBJECT",
    "MAX_PT": "__ODPS_MAX_PT",
}

_TOKEN_REWRITES_HIVE = {
    "DATEDIFF": "__HIVE_DATEDIFF",
    "DATE_ADD": "__HIVE_DATE_ADD",
    "DATE_SUB": "__HIVE_DATE_SUB",
    "DATE_FORMAT": "__HIVE_DATE_FORMAT",
    "GET_JSON_OBJECT": "__ODPS_GET_JSON_OBJECT",
    "TO_DATE": "__HIVE_TO_DATE",
    "UNIX_TIMESTAMP": "__HIVE_UNIX_TIMESTAMP",
    "FROM_UNIXTIME": "__HIVE_FROM_UNIXTIME",
}


def _rewrite_function_tokens(sql: str, dialect: DialectType) -> str:
    """Rename dialect-sensitive functions before sqlglot Hive parse so arg signatures stay intact."""
    mapping = _TOKEN_REWRITES_ODPS if dialect == "odps" else _TOKEN_REWRITES_HIVE
    tokens = sqlglot.tokenize(sql, read="hive")
    replacements: list[tuple[int, int, str]] = []
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok.token_type == TokenType.COMMENT:
            continue
        upper_name = tok.text.upper()
        if (
            upper_name in mapping
            and i + 1 < n
            and tokens[i + 1].token_type == TokenType.L_PAREN
            and not (i > 0 and tokens[i - 1].token_type == TokenType.DOT)
        ):
            replacements.append((tok.start, tok.end, mapping[upper_name]))

    if not replacements:
        return sql

    parts: list[str] = []
    cursor = 0
    for start, end, repl in replacements:
        parts.append(sql[cursor:start])
        parts.append(repl)
        cursor = end + 1
    parts.append(sql[cursor:])
    return "".join(parts)


def transpile_to_duckdb(
    sql: str,
    catalog: WarehouseCatalog,
    dialect: DialectType = "odps",
    max_pt_resolver: Callable[[str], str | None] | None = None,
) -> str:
    """Deterministically transpile ODPS/Hive SQL AST into executable DuckDB SQL."""
    cleaned = sql.strip().rstrip(";").strip()
    preserved_sql = _rewrite_function_tokens(cleaned, dialect=dialect)
    tree = sqlglot.parse_one(preserved_sql, read="hive")

    def _transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Anonymous):
            fn = str(node.this).upper()
            args = [arg.transform(_transform) for arg in (node.expressions or [])]

            if fn == "__ODPS_MAX_PT" and len(args) == 1 and isinstance(args[0], exp.Literal):
                tbl_name = args[0].this
                if max_pt_resolver is not None:
                    resolved = max_pt_resolver(tbl_name)
                    if resolved is not None:
                        return exp.Literal.string(resolved)
                pcol = catalog.get_partition_col(tbl_name) or "ds"
                subq = exp.Select(expressions=[exp.Max(this=exp.column(pcol))]).from_(
                    exp.to_table(tbl_name)
                )
                return exp.Subquery(this=subq)

            if fn == "__ODPS_DATEDIFF" and len(args) == 3:
                return exp.Anonymous(this="odps_datediff", expressions=args)

            if fn == "__HIVE_DATEDIFF" and len(args) == 2:
                return exp.Anonymous(this="hive_datediff", expressions=args)

            if fn == "__ODPS_DATEADD" and len(args) == 3:
                return exp.Anonymous(this="odps_dateadd", expressions=args)

            if fn == "__HIVE_DATE_ADD" and len(args) == 2:
                return exp.Anonymous(this="hive_date_add", expressions=args)

            if fn == "__HIVE_DATE_SUB" and len(args) == 2:
                return exp.Anonymous(this="hive_date_sub", expressions=args)

            if fn == "__HIVE_DATE_FORMAT" and len(args) == 2:
                return exp.Anonymous(this="odps_to_char", expressions=args)

            if fn == "__HIVE_TO_DATE":
                if len(args) == 1:
                    return exp.Anonymous(this="hive_to_date_1", expressions=args)
                if len(args) == 2:
                    return exp.Anonymous(this="odps_to_date", expressions=args)

            if fn == "__HIVE_UNIX_TIMESTAMP":
                if len(args) == 1:
                    return exp.Anonymous(this="hive_unix_timestamp_1", expressions=args)
                if len(args) == 2:
                    return exp.Anonymous(this="hive_unix_timestamp_2", expressions=args)

            if fn == "__HIVE_FROM_UNIXTIME":
                if len(args) == 1:
                    return exp.Anonymous(this="hive_from_unixtime_1", expressions=args)
                if len(args) == 2:
                    return exp.Anonymous(this="hive_from_unixtime_2", expressions=args)

            if fn == "__ODPS_TO_DATE":
                if len(args) == 1:
                    return exp.Anonymous(this="odps_to_date_1", expressions=args)
                if len(args) == 2:
                    return exp.Anonymous(this="odps_to_date", expressions=args)

            if fn == "__ODPS_TO_CHAR" and len(args) == 2:
                return exp.Anonymous(this="odps_to_char", expressions=args)

            if fn == "__ODPS_GET_JSON_OBJECT" and len(args) == 2:
                return exp.Anonymous(this="odps_get_json_object", expressions=args)

        if isinstance(node, exp.Select) and node.args.get("laterals"):
            _convert_laterals_in_select(node)

        return node

    transformed = tree.transform(_transform)
    return transformed.sql(dialect="duckdb")


def _convert_laterals_in_select(select: exp.Select) -> None:
    """Convert Hive/ODPS LATERAL VIEW [OUTER] explode/posexplode into positional DuckDB UNNEST joins."""
    laterals: list[exp.Lateral] = list(select.args.get("laterals") or [])
    if not laterals:
        return

    joins: list[exp.Join] = list(select.args.get("joins") or [])

    # Build alias -> join index map for existing joins (0-based; -1 means FROM table).
    from_alias = ""
    from_clause = select.args.get("from_")
    if from_clause is not None and isinstance(from_clause.this, exp.Expression):
        from_alias = (from_clause.this.alias_or_name or "").lower()

    for lat in laterals:
        lat_expr = lat.this
        is_outer = bool(lat.args.get("outer"))
        alias_node = lat.args.get("alias")

        # Determine insert position right after the referenced source table/join.
        referenced_tables = {
            c.table.lower()
            for c in lat.find_all(exp.Column)
            if c.table
        }
        insert_idx = 0
        if referenced_tables:
            for idx, j in enumerate(joins):
                j_alias = (j.this.alias_or_name if isinstance(j.this, exp.Expression) else "").lower()
                if j_alias and j_alias in referenced_tables:
                    insert_idx = max(insert_idx, idx + 1)
                elif from_alias and from_alias in referenced_tables:
                    insert_idx = max(insert_idx, 0)

        if isinstance(lat_expr, exp.Explode):
            arr_expr = lat_expr.this
            unnest = exp.Unnest(
                expressions=[arr_expr],
                alias=alias_node.copy() if isinstance(alias_node, exp.TableAlias) else None,
                offset=False,
            )
            if is_outer:
                new_join = exp.Join(this=unnest, side="LEFT", on=exp.Boolean(this=True))
            else:
                new_join = exp.Join(this=unnest, kind="CROSS")
            joins.insert(insert_idx, new_join)

        elif isinstance(lat_expr, exp.Posexplode):
            # Fallback for posexplode: keep as CROSS/LEFT JOIN LATERAL UNNEST WITH ORDINALITY.
            arr_expr = lat_expr.this
            cols = list(alias_node.columns) if isinstance(alias_node, exp.TableAlias) else []
            pos_id = cols[0] if len(cols) >= 1 else exp.to_identifier("pos")
            val_id = cols[1] if len(cols) >= 2 else exp.to_identifier("val")
            inner_unnest = exp.Unnest(
                expressions=[arr_expr],
                alias=exp.TableAlias(
                    this=exp.to_identifier("_u"),
                    columns=[val_id.copy(), exp.to_identifier("_ord")],
                ),
                offset=True,
            )
            sub_sel = exp.Select(
                expressions=[
                    exp.Alias(
                        this=exp.Sub(this=exp.column("_ord"), expression=exp.Literal.number(1)),
                        alias=pos_id.copy(),
                    ),
                    exp.Column(this=val_id.copy()),
                ]
            ).from_(inner_unnest)
            lat_sub = exp.Lateral(
                this=exp.Subquery(this=sub_sel),
                alias=exp.TableAlias(this=alias_node.this.copy()) if isinstance(alias_node, exp.TableAlias) and alias_node.this else None,
            )
            if is_outer:
                new_join = exp.Join(this=lat_sub, side="LEFT", on=exp.Boolean(this=True))
            else:
                new_join = exp.Join(this=lat_sub, kind="CROSS")
            joins.insert(insert_idx, new_join)

    select.set("joins", joins)
    select.set("laterals", [])


class LocalDuckDBExecutor:
    """Thread-safe DuckDB executor with deterministic ODPS/Hive macros and query timeout."""

    def __init__(self, catalog: WarehouseCatalog, db_path: str = ":memory:") -> None:
        self.catalog = catalog
        self.conn = duckdb.connect(database=db_path)
        self._lock = threading.Lock()
        self._install_macros()

    def _install_macros(self) -> None:
        with self._lock:
            self.conn.execute(ODPS_MACRO_DDL)

    def execute_raw_duckdb(self, sql: str, timeout_sec: float = 3.0) -> dict[str, Any]:
        """Run already-valid DuckDB SQL with interrupt timer protection."""
        t0 = time.perf_counter()
        with self._lock:
            timer = threading.Timer(timeout_sec, self.conn.interrupt)
            timer.start()
            try:
                cur = self.conn.execute(sql)
                columns = [desc[0] for desc in (cur.description or [])]
                rows = cur.fetchall()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "ok": True,
                    "error_msg": "",
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "latency_ms": round(latency_ms, 3),
                }
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "ok": False,
                    "error_msg": str(exc),
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "latency_ms": round(latency_ms, 3),
                }
            finally:
                timer.cancel()

    def _resolve_max_pt(self, table_name: str) -> str | None:
        pcol = self.catalog.get_partition_col(table_name)
        if not pcol:
            return None
        with self._lock:
            row = self.conn.execute(f"SELECT MAX({pcol}) FROM {table_name}").fetchone()
        return str(row[0]) if row and row[0] is not None else None

    def execute_warehouse_sql(
        self,
        sql: str,
        dialect: DialectType = "odps",
        timeout_sec: float = 3.0,
    ) -> dict[str, Any]:
        """Transpile ODPS/Hive SQL to DuckDB and run safely."""
        t0 = time.perf_counter()
        try:
            duck_sql = transpile_to_duckdb(
                sql,
                catalog=self.catalog,
                dialect=dialect,
                max_pt_resolver=self._resolve_max_pt,
            )
        except Exception as exc:
            return {
                "error_stage": "runtime",
                "error_msg": f"Transpilation error: {exc}",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "transpiled_sql": "",
            }

        raw_res = self.execute_raw_duckdb(duck_sql, timeout_sec=timeout_sec)
        if not raw_res["ok"]:
            return {
                "error_stage": "runtime",
                "error_msg": raw_res["error_msg"],
                "columns": [],
                "rows": [],
                "row_count": 0,
                "latency_ms": raw_res["latency_ms"],
                "transpiled_sql": duck_sql,
            }

        return {
            "error_stage": "none",
            "error_msg": "",
            "columns": raw_res["columns"],
            "rows": raw_res["rows"],
            "row_count": raw_res["row_count"],
            "latency_ms": raw_res["latency_ms"],
            "transpiled_sql": duck_sql,
        }

    def close(self) -> None:
        with self._lock:
            self.conn.close()
