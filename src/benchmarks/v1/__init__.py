"""Stratified v1 enterprise data warehouse benchmark suite (75 tasks across 5 categories)."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask
from src.benchmarks.v1.fanout_dedup import get_fanout_dedup_tasks
from src.benchmarks.v1.json_array_unnest import get_json_array_unnest_tasks
from src.benchmarks.v1.partition_pruning import get_partition_pruning_tasks
from src.benchmarks.v1.ratio_null_handling import get_ratio_null_handling_tasks
from src.benchmarks.v1.retention_date_math import get_retention_date_math_tasks


def get_v1_tasks() -> list[BenchmarkTask]:
    """Return all 75 stratified v1 benchmark tasks (train=25, val=20, test=30)."""
    return (
        get_partition_pruning_tasks()
        + get_json_array_unnest_tasks()
        + get_retention_date_math_tasks()
        + get_fanout_dedup_tasks()
        + get_ratio_null_handling_tasks()
    )


__all__ = [
    "get_fanout_dedup_tasks",
    "get_json_array_unnest_tasks",
    "get_partition_pruning_tasks",
    "get_ratio_null_handling_tasks",
    "get_retention_date_math_tasks",
    "get_v1_tasks",
]
