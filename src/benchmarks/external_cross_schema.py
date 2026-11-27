"""Compatibility re-export shim for `src.benchmarks.external` (`external_bird_mini` 3-domain suite).

Note: Scores on warehouse-adapted BIRD schemas measure cross-schema & partition-constraint
generalization under EvoDW's ODPS/Hive gateway, and MUST NOT be conflated with raw SQLite
leaderboard numbers on original BIRD-bench.
"""

from __future__ import annotations

from src.benchmarks.external import (
    EXTERNAL_ADAPTATION_DISCLAIMER,
    EXTERNAL_DOMAINS,
    build_external_cross_schema_catalog,
    create_external_cross_schema_sandbox,
    get_external_bird_mini_tasks,
    make_external_task,
    to_hive_external_gold_sql,
)

_mk_ext_task = make_external_task

__all__ = [
    "EXTERNAL_ADAPTATION_DISCLAIMER",
    "EXTERNAL_DOMAINS",
    "build_external_cross_schema_catalog",
    "create_external_cross_schema_sandbox",
    "get_external_bird_mini_tasks",
    "make_external_task",
    "to_hive_external_gold_sql",
]
