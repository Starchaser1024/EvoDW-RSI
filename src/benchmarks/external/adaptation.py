"""SQLite-to-ODPS/Hive adaptation ledger helpers and dialect Gold SQL converters for external domains."""

from __future__ import annotations

import re

from src.benchmarks.types import BenchmarkTask, CategoryType, SplitType

EXTERNAL_ADAPTATION_DISCLAIMER = (
    "Warehouse-adapted BIRD Mini-Dev schemas inject ds/dt partition contracts and transpile SQLite idioms "
    "to ODPS/Hive ASTs; scores measure cross-schema & partition-constraint generalization under EvoDW's "
    "gateway and MUST NOT be conflated with raw SQLite leaderboard numbers on original BIRD-bench."
)

EXTERNAL_DOMAINS: tuple[str, ...] = (
    "financial",
    "thrombosis_clinical",
    "california_schools",
)


def make_external_task(
    *,
    task_id: str,
    domain: str,
    category: CategoryType,
    split: SplitType,
    question: str,
    hint: str,
    original_sqlite_sql: str,
    gold_sql: str,
    ast_transformations: list[str],
    injected_partition_predicates: list[str],
    unpartitioned_dim_tables: list[str],
    caliber_trap: str,
    ordered: bool = True,
) -> BenchmarkTask:
    """Build a `BenchmarkTask` annotated with its full SQLite->ODPS AST adaptation ledger."""
    return BenchmarkTask(
        task_id=task_id,
        category=category,
        split=split,
        question=question,
        hint=hint,
        gold_sql=gold_sql.strip(),
        ordered=ordered,
        metadata={
            "domain": domain,
            "adaptation_ledger": {
                "domain": domain,
                "original_sqlite_sql": original_sqlite_sql.strip(),
                "ast_transformations": ast_transformations,
                "injected_partition_predicates": injected_partition_predicates,
                "unpartitioned_dim_tables": unpartitioned_dim_tables,
                "caliber_trap": caliber_trap,
                "disclaimer": EXTERNAL_ADAPTATION_DISCLAIMER,
            },
        },
    )


def to_hive_external_gold_sql(odps_sql: str, latest_ds: str = "20261015") -> str:
    """Deterministically adapt an external_bird_mini ODPS Gold SQL into Hive-compliant Gold SQL."""
    hive_sql = re.sub(r"MAX_PT\('[^']+'\)", f"'{latest_ds}'", odps_sql)
    hive_sql = re.sub(
        r"DATEDIFF\(\s*TO_DATE\(([^,]+),\s*'yyyy-mm-dd'\)\s*,\s*TO_DATE\(([^,]+),\s*'yyyy-mm-dd'\)\s*,\s*'dd'\s*\)",
        r"DATEDIFF(\1, \2)",
        hive_sql,
    )
    hive_sql = re.sub(
        r"DATEDIFF\(\s*TO_DATE\('2026-10-15',\s*'yyyy-mm-dd'\)\s*,\s*TO_DATE\(([^,]+),\s*'yyyy-mm-dd'\)\s*,\s*'yyyy'\s*\)",
        r"(CAST(SUBSTR('2026-10-15', 1, 4) AS INT) - CAST(SUBSTR(\1, 1, 4) AS INT))",
        hive_sql,
    )
    return hive_sql


__all__ = [
    "EXTERNAL_ADAPTATION_DISCLAIMER",
    "EXTERNAL_DOMAINS",
    "make_external_task",
    "to_hive_external_gold_sql",
]
