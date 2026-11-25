"""Core types and task data model for EvoDW-RSI warehouse Text-to-SQL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CategoryType = Literal[
    "partition_pruning",
    "json_array_unnest",
    "retention_date_math",
    "fanout_dedup",
    "ratio_null_handling",
    "semantic_linking_trap",
    "scd2_temporal_join",
    "silent_caliber_audit",
]
SplitType = Literal["train", "val", "test"]
SuiteType = Literal["v1", "hard_v2", "all", "external_bird_mini"]


@dataclass
class BenchmarkTask:
    """Single warehouse Text-to-SQL evaluation task."""

    task_id: str
    category: CategoryType
    split: SplitType
    question: str
    hint: str
    gold_sql: str
    ordered: bool = False
    gold_rows: list[tuple[Any, ...]] | None = field(default=None, repr=False)
    gold_columns: list[str] | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
