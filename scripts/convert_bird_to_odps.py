"""Convert BIRD Mini-Dev SQLite tasks into ds-partitioned ODPS warehouse tasks."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import ColumnMeta, LocalDuckDBExecutor, TableMeta, WarehouseCatalog, WarehouseSandbox


def _is_dimension_table(table_name: str) -> bool:
    low = table_name.lower()
    return any(k in low for k in ("dim_", "user", "customer", "product", "item", "category", "shop", "lookup"))


def convert_sqlite_sql_to_odps(
    sqlite_sql: str,
    partitioned_tables: set[str],
    default_ds: str = "20261015",
) -> str:
    """Transpile SQLite query to Hive/ODPS dialect and inject ds partition predicates."""
    tree = sqlglot.parse_one(sqlite_sql, read="sqlite")

    # Replace IFNULL -> NVL and STRFTIME -> TO_CHAR where possible via AST transform.
    def _ast_xform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Coalesce) and node.args.get("is_nvl"):
            return node
        if isinstance(node, exp.Select):
            # Inject ds = '<default_ds>' for every partitioned physical table in this Select.
            part_filters: list[exp.Expression] = []
            from_clause = node.args.get("from_")
            sources: list[exp.Expression] = []
            if from_clause is not None and from_clause.this is not None:
                sources.append(from_clause.this)
            for j in node.args.get("joins") or []:
                if j.this is not None:
                    sources.append(j.this)

            for src in sources:
                if isinstance(src, exp.Table) and src.name.lower() in partitioned_tables:
                    alias = src.alias_or_name or src.name
                    col_expr = exp.Column(
                        this=exp.to_identifier("ds"),
                        table=exp.to_identifier(alias),
                    )
                    part_filters.append(exp.EQ(this=col_expr, expression=exp.Literal.string(default_ds)))

            if part_filters:
                combined = part_filters[0]
                for f in part_filters[1:]:
                    combined = exp.And(this=combined, expression=f)
                existing_where = node.args.get("where")
                if existing_where is not None and existing_where.this is not None:
                    node.set("where", exp.Where(this=exp.And(this=existing_where.this, expression=combined)))
                else:
                    node.set("where", exp.Where(this=combined))
        return node

    transformed = tree.transform(_ast_xform)
    # Emit Hive/ODPS-compatible SQL (note: sqlglot hive dialect turns IFNULL into COALESCE and LIMIT a,b into LIMIT b OFFSET a).
    odps_sql = transformed.sql(dialect="hive")
    # Convert STRFTIME('%Y-%m', col) -> TO_CHAR(TO_DATE(col, 'yyyy-mm-dd'), 'yyyy-mm') if any remained.
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
    partitioned_tables: set[str] = set()

    for tname in tables_raw:
        pragma_rows = sq_cur.execute(f'PRAGMA table_info("{tname}")').fetchall()
        is_part = not _is_dimension_table(tname)
        if is_part:
            partitioned_tables.add(tname.lower())

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
        odps_sql = convert_sqlite_sql_to_odps(raw_sql, partitioned_tables, default_ds=default_ds)
        exec_res = sandbox.run_sql(odps_sql, dialect="odps")
        converted_tasks.append(
            {
                "task_id": entry.get("question_id", f"bird_{idx:04d}"),
                "question": entry.get("question", ""),
                "hint": entry.get("evidence", ""),
                "original_sqlite_sql": raw_sql,
                "odps_sql": odps_sql,
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
                "question": "Count orders by customer",
                "evidence": "",
                "SQL": "SELECT customer_id, COUNT(*) AS cnt FROM fact_orders GROUP BY customer_id ORDER BY cnt DESC",
            },
        ]

        _, sandbox, converted = convert_bird_database_and_tasks(db_path, sample_entries, default_ds="20261015")
        try:
            for item in converted:
                assert item["gateway_passed"], f"Gateway failed for {item['task_id']}: {item['error_msg']}"
                assert item["error_stage"] == "none", f"Exec failed for {item['task_id']}: {item['error_msg']}"
                assert item["row_count"] > 0, f"Empty result for {item['task_id']}"
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
