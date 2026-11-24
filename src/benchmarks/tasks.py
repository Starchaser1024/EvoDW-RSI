"""Unified benchmark task registry across v1 (75 tasks) and Hard-v2 (45 tasks) warehouse suites."""

from __future__ import annotations

from src.benchmarks.types import BenchmarkTask, CategoryType, SplitType, SuiteType
from src.benchmarks.v1 import get_v1_tasks


def get_all_tasks(suite: SuiteType = "v1") -> list[BenchmarkTask]:
    """Return benchmark tasks for the selected suite ('v1'=75, 'hard_v2'=45, 'all'=120)."""
    if suite == "v1":
        return get_v1_tasks()
    from src.benchmarks.hard_v2_tasks import get_hard_v2_tasks

    if suite == "hard_v2":
        return get_hard_v2_tasks()
    if suite == "all":
        return get_v1_tasks() + get_hard_v2_tasks()
    raise ValueError(f"Unknown benchmark suite: {suite!r}; expected 'v1', 'hard_v2', or 'all'.")


__all__ = [
    "BenchmarkTask",
    "CategoryType",
    "SplitType",
    "SuiteType",
    "get_all_tasks",
    "get_v1_tasks",
]
