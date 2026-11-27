"""External cross-schema warehouse benchmark (`external_bird_mini`):
24 tasks across 3 unseen non-e-commerce BIRD Mini-Dev domains (11 normalized tables):
  1. Retail Banking & Loans (`financial`)
  2. Clinical Hospital Admissions (`thrombosis_clinical`)
  3. Public School Performance (`california_schools`)
"""

from __future__ import annotations

from src.benchmarks.external.adaptation import (
    EXTERNAL_ADAPTATION_DISCLAIMER,
    EXTERNAL_DOMAINS,
    make_external_task,
    to_hive_external_gold_sql,
)
from src.benchmarks.external.clinical import (
    get_clinical_tasks,
    populate_clinical_tables,
    register_clinical_tables,
)
from src.benchmarks.external.financial import (
    get_financial_tasks,
    populate_financial_tables,
    register_financial_tables,
)
from src.benchmarks.external.schools import (
    get_schools_tasks,
    populate_schools_tables,
    register_schools_tables,
)
from src.benchmarks.types import BenchmarkTask
from src.env.local_executor import LocalDuckDBExecutor
from src.env.sandbox import DialectType, WarehouseSandbox
from src.env.schema import WarehouseCatalog


def build_external_cross_schema_catalog() -> WarehouseCatalog:
    """Build the 11-table normalized multi-domain catalog across banking, clinical, and schools."""
    catalog = WarehouseCatalog()
    register_financial_tables(catalog)
    register_clinical_tables(catalog)
    register_schools_tables(catalog)
    return catalog


def create_external_cross_schema_sandbox(
    seed: int = 2026,
    dialect: DialectType = "odps",
) -> WarehouseSandbox:
    """Create and deterministically populate the 11-table external multi-domain warehouse sandbox."""
    catalog = build_external_cross_schema_catalog()
    executor = LocalDuckDBExecutor(catalog=catalog, db_path=":memory:")
    con = executor.conn
    populate_financial_tables(con)
    populate_clinical_tables(con)
    populate_schools_tables(con)
    return WarehouseSandbox(
        catalog=catalog,
        executor=executor,
        default_dialect=dialect,
        immutable_cache_tag=f"ext_seed_{seed}",
    )


def get_external_bird_mini_tasks() -> list[BenchmarkTask]:
    """Return all 24 cross-schema tasks across Banking, Clinical, and Schools with adaptation ledgers."""
    return [
        *get_financial_tasks(),
        *get_clinical_tasks(),
        *get_schools_tasks(),
    ]


__all__ = [
    "EXTERNAL_ADAPTATION_DISCLAIMER",
    "EXTERNAL_DOMAINS",
    "build_external_cross_schema_catalog",
    "create_external_cross_schema_sandbox",
    "get_clinical_tasks",
    "get_external_bird_mini_tasks",
    "get_financial_tasks",
    "get_schools_tasks",
    "make_external_task",
    "populate_clinical_tables",
    "populate_financial_tables",
    "populate_schools_tables",
    "register_clinical_tables",
    "register_financial_tables",
    "register_schools_tables",
    "to_hive_external_gold_sql",
]
