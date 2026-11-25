"""Benchmark loader binding tasks with precomputed Gold result sets on the sandbox."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.benchmarks.dw_generator import create_populated_sandbox
from src.benchmarks.tasks import BenchmarkTask, SuiteType, get_all_tasks
from src.env.sandbox import DialectType, WarehouseSandbox


@dataclass
class LoadedBenchmark:
    """Container holding the populated sandbox and stratified train/val/test task splits."""

    sandbox: WarehouseSandbox
    all_tasks: list[BenchmarkTask]
    train: list[BenchmarkTask]
    val: list[BenchmarkTask]
    test: list[BenchmarkTask]
    suite: SuiteType = "v1"

    def get_active_val(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Return full val or a stratified active subset (round-robin across categories)."""
        if limit is None or limit >= len(self.val):
            return list(self.val)
        by_cat: dict[str, list[BenchmarkTask]] = defaultdict(list)
        for t in self.val:
            by_cat[t.category].append(t)
        selected: list[BenchmarkTask] = []
        idx = 0
        cats = sorted(by_cat.keys())
        while len(selected) < limit:
            progressed = False
            for cat in cats:
                if idx < len(by_cat[cat]) and len(selected) < limit:
                    selected.append(by_cat[cat][idx])
                    progressed = True
            if not progressed:
                break
            idx += 1
        return selected


def load_benchmark(
    seed: int = 2026,
    dialect: DialectType = "odps",
    precompute_gold: bool = True,
    suite: SuiteType = "v1",
) -> LoadedBenchmark:
    """Create the populated warehouse sandbox and verify/cache dialect-invariant Gold answers."""
    if suite == "external_bird_mini":
        from src.benchmarks.external_cross_schema import create_external_cross_schema_sandbox

        sandbox = create_external_cross_schema_sandbox(seed=seed, dialect=dialect)
    else:
        sandbox = create_populated_sandbox(seed=seed, dialect=dialect)
    tasks = get_all_tasks(suite=suite)

    if precompute_gold:
        for task in tasks:
            # Ground-truth result rows over the deterministic DuckDB warehouse are dialect-invariant;
            # execute canonical Gold SQL via ODPS mode so Hive runs also evaluate against exact cached gold_rows.
            res = sandbox.run_sql(task.gold_sql, dialect="odps")
            if not res["gateway_passed"] or res["error_stage"] != "none":
                raise RuntimeError(
                    f"Gold SQL failed for task {task.task_id}: {res['error_stage']} -> {res['error_msg']}"
                )
            task.gold_rows = res["rows"]
            task.gold_columns = res["columns"]

    train = [t for t in tasks if t.split == "train"]
    val = [t for t in tasks if t.split == "val"]
    test = [t for t in tasks if t.split == "test"]
    return LoadedBenchmark(
        sandbox=sandbox,
        all_tasks=tasks,
        train=train,
        val=val,
        test=test,
        suite=suite,
    )
