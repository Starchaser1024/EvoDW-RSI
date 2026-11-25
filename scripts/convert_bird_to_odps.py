"""Convert BIRD Mini-Dev SQLite tasks into ds/dt-partitioned ODPS warehouse tasks with AST adaptation ledgers."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import ColumnMeta, LocalDuckDBExecutor, TableMeta, WarehouseCatalog, WarehouseSandbox

EXTERNAL_ADAPTATION_DISCLAIMER = (
    "Warehouse-adapted BIRD Mini-Dev schemas inject ds/dt partition contracts and transpile SQLite idioms "
    "to ODPS/Hive ASTs; scores measure cross-schema & partition-constraint generalization under EvoDW's "
    "gateway and MUST NOT be conflated with raw SQLite leaderboard numbers on original BIRD-bench."
)

_SQLITE_STRFTIME_MAP = (
    ("%Y-%m-%d", "yyyy-mm-dd"),
    ("%Y%m%d", "yyyymmdd"),
    ("%Y-%m", "yyyy-mm"),
    ("%Y", "yyyy"),
    ("%m", "mm"),
    ("%d", "dd"),
)

_COMMA_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\s*,\s*(\d+)\b", flags=re.IGNORECASE)


def _is_dimension_table(table_name: str) -> bool:
    low = table_name.lower()
    return any(
        k in low
        for k in (
            "dim_",
            "_dim",
            "_scd2",
            "user",
            "customer",
            "product",
            "item",
            "category",
            "shop",
            "lookup",
            "district",
            "account",
            "patient",
            "school",
        )
    )


def _sqlite_fmt_to_odps(fmt: str) -> str:
    out = fmt
    for sq_pat, odps_pat in _SQLITE_STRFTIME_MAP:
        out = out.replace(sq_pat, odps_pat)
    return out


def transpile_sqlite_to_odps_with_ledger(
    sqlite_sql: str,
    partitioned_tables: set[str] | dict[str, str],
    default_ds: str = "20261015",
    *,
    use_max_pt: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Transpile a SQLite query into ODPS SQL and return a deterministic AST adaptation ledger."""
    part_map: dict[str, str] = (
        {k.lower(): v for k, v in partitioned_tables.items()}
        if isinstance(partitioned_tables, dict)
        else {t.lower(): "ds" for t in partitioned_tables}
    )

    ast_rewrites: list[str] = []
    injected_predicates: list[str] = []
    unpartitioned_tables: list[str] = []

    if re.search(r"\bIFNULL\s*\(", sqlite_sql, flags=re.IGNORECASE):
        ast_rewrites.append("IFNULL -> COALESCE/NVL")
    if re.search(r"\bSTRFTIME\s*\(", sqlite_sql, flags=re.IGNORECASE):
        ast_rewrites.append("STRFTIME -> TO_CHAR(TO_DATE(...))")
    if _COMMA_LIMIT_RE.search(sqlite_sql):
        ast_rewrites.append("LIMIT offset, count -> LIMIT count OFFSET offset")

    tree = sqlglot.parse_one(sqlite_sql, read="sqlite")

    def _ast_xform(node: exp.Expression) -> exp.Expression:
        # Rewrite SQLite STRFTIME(fmt, col) -> TO_CHAR(TO_DATE(col, 'yyyy-mm-dd'), odps_fmt)
        if isinstance(node, exp.TimeToStr):
            fmt_node = node.args.get("format")
            fmt_raw = fmt_node.this if isinstance(fmt_node, exp.Literal) else "%Y-%m-%d"
            odps_fmt = _sqlite_fmt_to_odps(str(fmt_raw))
            inner = node.this.transform(_ast_xform)
            to_date_expr = exp.Anonymous(
                this="TO_DATE",
                expressions=[inner, exp.Literal.string("yyyy-mm-dd")],
            )
            return exp.Anonymous(
                this="TO_CHAR",
                expressions=[to_date_expr, exp.Literal.string(odps_fmt)],
            )

        if isinstance(node, exp.Anonymous) and str(node.this).upper() == "STRFTIME":
            args = list(node.expressions or [])
            if len(args) >= 2:
                fmt_raw = args[0].this if isinstance(args[0], exp.Literal) else "%Y-%m-%d"
                odps_fmt = _sqlite_fmt_to_odps(str(fmt_raw))
                inner = args[1].transform(_ast_xform)
                to_date_expr = exp.Anonymous(
                    this="TO_DATE",
                    expressions=[inner, exp.Literal.string("yyyy-mm-dd")],
                )
                return exp.Anonymous(
                    this="TO_CHAR",
                    expressions=[to_date_expr, exp.Literal.string(odps_fmt)],
                )

        if isinstance(node, exp.Select):
            cte_names = {
                cte.alias_or_name.lower()
                for cte in (tree.args.get("with_") or [])
                if getattr(cte, "alias_or_name", None)
            }
            part_filters_where: list[exp.Expression] = []
            from_clause = node.args.get("from_")
            if from_clause is not None and isinstance(from_clause.this, exp.Table):
                src = from_clause.this
                t_low = src.name.lower()
                if t_low not in cte_names:
                    if t_low in part_map:
                        pcol = part_map[t_low]
                        alias = src.alias_or_name or src.name
                        rhs: exp.Expression = (
                            exp.Anonymous(this="MAX_PT", expressions=[exp.Literal.string(src.name)])
                            if use_max_pt
                            else exp.Literal.string(default_ds)
                        )
                        eq = exp.EQ(
                            this=exp.Column(this=exp.to_identifier(pcol), table=exp.to_identifier(alias)),
                            expression=rhs,
                        )
                        part_filters_where.append(eq)
                        injected_predicates.append(f"{alias}.{pcol} = {rhs.sql(dialect='hive')}")
                    elif src.name not in unpartitioned_tables:
                        unpartitioned_tables.append(src.name)

            for j in node.args.get("joins") or []:
                if isinstance(j.this, exp.Table):
                    src = j.this
                    t_low = src.name.lower()
                    if t_low in cte_names:
                        continue
                    if t_low in part_map:
                        pcol = part_map[t_low]
                        alias = src.alias_or_name or src.name
                        rhs = (
                            exp.Anonymous(this="MAX_PT", expressions=[exp.Literal.string(src.name)])
                            if use_max_pt
                            else exp.Literal.string(default_ds)
                        )
                        eq = exp.EQ(
                            this=exp.Column(this=exp.to_identifier(pcol), table=exp.to_identifier(alias)),
                            expression=rhs,
                        )
                        # Preserve LEFT JOIN semantics by injecting right-table partition predicate into ON!
                        join_side = str(j.args.get("side") or "").upper()
                        if join_side in ("LEFT", "FULL") and j.args.get("on") is not None:
                            j.set("on", exp.And(this=j.args["on"], expression=eq))
                            injected_predicates.append(f"ON {alias}.{pcol} = {rhs.sql(dialect='hive')}")
                        else:
                            part_filters_where.append(eq)
                            injected_predicates.append(f"{alias}.{pcol} = {rhs.sql(dialect='hive')}")
                    elif src.name not in unpartitioned_tables:
                        unpartitioned_tables.append(src.name)

            if part_filters_where:
                combined = part_filters_where[0]
                for f in part_filters_where[1:]:
                    combined = exp.And(this=combined, expression=f)
                existing_where = node.args.get("where")
                if existing_where is not None and existing_where.this is not None:
                    node.set("where", exp.Where(this=exp.And(this=existing_where.this, expression=combined)))
                else:
                    node.set("where", exp.Where(this=combined))
        return node

    transformed = tree.transform(_ast_xform)
    odps_sql = transformed.sql(dialect="hive")
    if injected_predicates:
        ast_rewrites.append("FACT_PARTITION_PREDICATE_INJECTION")

    ledger = {
        "source_dialect": "sqlite",
        "target_dialect": "odps",
        "ast_transformations": ast_rewrites,
        "injected_partition_predicates": injected_predicates,
        "unpartitioned_dim_tables": unpartitioned_tables,
        "disclaimer": EXTERNAL_ADAPTATION_DISCLAIMER,
    }
    return odps_sql, ledger


def convert_sqlite_sql_to_odps(
    sqlite_sql: str,
    partitioned_tables: set[str] | dict[str, str],
    default_ds: str = "20261015",
) -> str:
    """Transpile SQLite query to Hive/ODPS dialect and inject ds/dt partition predicates."""
    odps_sql, _ = transpile_sqlite_to_odps_with_ledger(
        sqlite_sql=sqlite_sql,
        partitioned_tables=partitioned_tables,
        default_ds=default_ds,
    )
    return odps_sql


def convert_bird_database_and_tasks(
    db_path: Path,
    entries: list[dict[str, Any]],
    *,
    default_ds: str = "20261015",
) -> tuple[WarehouseCatalog, WarehouseSandbox, list[dict[str, Any]]]:
    """Load SQLite tables into DuckDB sandbox injecting `ds` partition column and convert BIRD SQLs."""
    sq_conn = sqlite3.connect(str(db_path))
    sq_cur = sq_conn.cursor()

    tables_raw = [
        r[0]
        for r in sq_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    ]

    catalog = WarehouseCatalog()
    executor = LocalDuckDBExecutor(catalog=catalog, db_path=":memory:")
    partitioned_tables: dict[str, str] = {}

    for tname in tables_raw:
        pragma_rows = sq_cur.execute(f'PRAGMA table_info("{tname}")').fetchall()
        is_part = not _is_dimension_table(tname)
        if is_part:
            partitioned_tables[tname.lower()] = "ds"

        col_metas: list[ColumnMeta] = []
        col_defs: list[str] = []
        for _, cname, ctype, *_ in pragma_rows:
            dtype = "DOUBLE" if "REAL" in ctype.upper() or "FLOAT" in ctype.upper() else (
                "INTEGER" if "INT" in ctype.upper() else "VARCHAR"
            )
            col_metas.append(ColumnMeta(name=cname, data_type=dtype, comment="Imported from BIRD SQLite"))
            col_defs.append(f'"{cname}" {dtype}')

        if is_part:
            col_metas.append(ColumnMeta(name="ds", data_type="VARCHAR", comment="Injected partition date", is_partition=True))
            col_defs.append('"ds" VARCHAR')

        catalog.register(
            TableMeta(
                name=tname,
                layer="dwd" if is_part else "dim",
                comment=f"Converted BIRD table ({tname})",
                columns=tuple(col_metas),
                partition_col="ds" if is_part else None,
            )
        )

        executor.conn.execute(f'CREATE TABLE "{tname}" ({", ".join(col_defs)})')
        src_rows = sq_cur.execute(f'SELECT * FROM "{tname}"').fetchall()
        if src_rows:
            if is_part:
                padded = [tuple(r) + (default_ds,) for r in src_rows]
            else:
                padded = [tuple(r) for r in src_rows]
            placeholders = ", ".join(["?"] * len(col_defs))
            executor.conn.executemany(f'INSERT INTO "{tname}" VALUES ({placeholders})', padded)

    sq_conn.close()
    sandbox = WarehouseSandbox(catalog=catalog, executor=executor, default_dialect="odps")

    converted_tasks: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        raw_sql = str(entry.get("SQL") or entry.get("sql") or "")
        odps_sql, ledger = transpile_sqlite_to_odps_with_ledger(
            raw_sql, partitioned_tables, default_ds=default_ds
        )
        exec_res = sandbox.run_sql(odps_sql, dialect="odps")
        converted_tasks.append(
            {
                "task_id": entry.get("question_id", f"bird_{idx:04d}"),
                "question": entry.get("question", ""),
                "hint": entry.get("evidence", ""),
                "original_sqlite_sql": raw_sql,
                "odps_sql": odps_sql,
                "adaptation_ledger": ledger,
                "gateway_passed": exec_res["gateway_passed"],
                "error_stage": exec_res["error_stage"],
                "error_msg": exec_res["error_msg"],
                "row_count": exec_res["row_count"],
            }
        )

    return catalog, sandbox, converted_tasks


def run_self_test() -> int:
    """Create a temporary SQLite DB + BIRD-style queries and verify conversion to ODPS sandbox."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "mini_bird.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE customers (customer_id INTEGER, segment TEXT)")
        con.execute("CREATE TABLE fact_orders (order_id INTEGER, customer_id INTEGER, amount REAL, order_dt TEXT)")
        con.executemany(
            "INSERT INTO customers VALUES (?, ?)",
            [(1, "VIP"), (2, "Standard"), (3, None)],
        )
        con.executemany(
            "INSERT INTO fact_orders VALUES (?, ?, ?, ?)",
            [(101, 1, 250.5, "2026-10-15"), (102, 1, 100.0, "2026-10-15"), (103, 2, None, "2026-10-15")],
        )
        con.commit()
        con.close()

        sample_entries = [
            {
                "question_id": "bird_demo_1",
                "question": "List customer segments and total non-null order amounts",
                "evidence": "Treat null amount as 0",
                "SQL": (
                    "SELECT IFNULL(c.segment, 'UNKNOWN') AS seg, SUM(IFNULL(o.amount, 0)) AS tot "
                    "FROM fact_orders o JOIN customers c ON o.customer_id = c.customer_id "
                    "GROUP BY IFNULL(c.segment, 'UNKNOWN') ORDER BY tot DESC LIMIT 0, 5"
                ),
            },
            {
                "question_id": "bird_demo_2",
                "question": "Count orders by order month",
                "evidence": "",
                "SQL": (
                    "SELECT STRFTIME('%Y-%m', order_dt) AS ym, COUNT(*) AS cnt "
                    "FROM fact_orders GROUP BY STRFTIME('%Y-%m', order_dt) ORDER BY cnt DESC"
                ),
            },
        ]

        _, sandbox, converted = convert_bird_database_and_tasks(db_path, sample_entries, default_ds="20261015")
        try:
            for item in converted:
                assert item["gateway_passed"], f"Gateway failed for {item['task_id']}: {item['error_msg']}"
                assert item["error_stage"] == "none", f"Exec failed for {item['task_id']}: {item['error_msg']}"
                assert item["row_count"] > 0, f"Empty result for {item['task_id']}"
                assert item["adaptation_ledger"]["ast_transformations"], "Expected recorded AST rewrites"
                print(f"[PASS] {item['task_id']} -> ODPS SQL: {item['odps_sql']} (rows={item['row_count']})")
        finally:
            sandbox.close()
    print("BIRD-to-ODPS self-test passed!")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert BIRD Mini-Dev SQLite benchmarks to ds-partitioned ODPS SQL.")
    parser.add_argument("--bird-json", type=Path, help="Path to BIRD dev/mini-dev JSON file.")
    parser.add_argument("--db-path", type=Path, help="Path to target SQLite database file.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/bird_odps_converted.json"))
    parser.add_argument("--default-ds", default="20261015")
    parser.add_argument("--self-test", action="store_true", help="Run built-in temporary SQLite conversion self-test.")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.bird_json or not args.db_path:
        parser.error("Provide both --bird-json and --db-path, or pass --self-test.")

    entries = json.loads(args.bird_json.read_text(encoding="utf-8"))
    if args.db_path.is_file():
        _, sandbox, converted = convert_bird_database_and_tasks(
            args.db_path, entries, default_ds=args.default_ds
        )
        sandbox.close()
    else:
        raise FileNotFoundError(f"SQLite DB not found: {args.db_path}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Converted {len(converted)} tasks -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
