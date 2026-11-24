"""Data structures, stratified batching, validation gates, and parent curriculum for GEPA evolution."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.agent.policy import ModuleName, Policy
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.tasks import BenchmarkTask
from src.env.sandbox import DialectType
from src.eval.metrics import EvalMetrics
from src.rsi.pareto import CandidateRecord, ParetoArchive, score_trajectory_for_pareto

EvolutionMethod = Literal["seed", "greedy", "gepa"]


@dataclass
class GenerationLog:
    """Single candidate lifecycle record inside an evolution run."""

    generation: int
    policy_id: str
    parent_ids: list[str]
    operation: str  # seed | mutate | merge
    target_module: str
    smoke_passed: bool
    active_val_ex: float | None
    active_val_first_pass_gw: float | None
    frontier_size: int
    rollouts_used: int
    rationale: str = ""


@dataclass
class EvolutionRunResult:
    """Complete artifacts and metrics produced by an EvolutionEngine run."""

    method: EvolutionMethod
    dialect: DialectType
    budget: int
    rollouts_used: int
    seed_policy: Policy
    best_policy: Policy
    policies_by_id: dict[str, Policy]
    history: list[GenerationLog]
    archive: ParetoArchive
    seed_val_trajectories: list[AgentTrajectory]
    seed_test_trajectories: list[AgentTrajectory]
    best_val_trajectories: list[AgentTrajectory]
    best_test_trajectories: list[AgentTrajectory]
    seed_val_metrics: EvalMetrics
    seed_test_metrics: EvalMetrics
    best_val_metrics: EvalMetrics
    best_test_metrics: EvalMetrics
    verified_train_val_trajectories: list[AgentTrajectory]
    llm_usage_summary: dict[str, Any] = field(default_factory=dict)
    cache_hits: int = 0
    effective_val_size: int = 0

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "dialect": self.dialect,
            "budget": self.budget,
            "rollouts_used": self.rollouts_used,
            "effective_val_size": self.effective_val_size,
            "cache_hits": self.cache_hits,
            "best_policy_id": self.best_policy.policy_id,
            "seed_val": self.seed_val_metrics.to_dict(),
            "seed_test": self.seed_test_metrics.to_dict(),
            "best_val": self.best_val_metrics.to_dict(),
            "best_test": self.best_test_metrics.to_dict(),
            "history": [asdict(h) for h in self.history],
            "llm_usage": self.llm_usage_summary,
        }


def compute_stage_limits(
    *,
    full_val_len: int,
    budget: int,
    val_active_size: int | None,
    minibatch_size: int | None,
) -> tuple[int, int, int]:
    """Compute `(stage1_limit, stage2_limit, mb_size)` for two-stage validation."""
    stage1_limit = (
        min(5, val_active_size)
        if val_active_size is not None
        else (5 if budget < 110 else min(10, full_val_len))
    )
    if val_active_size is not None:
        stage2_limit = min(val_active_size, full_val_len)
    elif budget <= 65:
        stage2_limit = min(8, full_val_len)
    elif budget <= 90:
        stage2_limit = min(10, full_val_len)
    elif budget <= 125:
        stage2_limit = min(15, full_val_len)
    else:
        stage2_limit = full_val_len

    mb_size = (
        minibatch_size
        if minibatch_size is not None
        else (3 if (budget < 120 or stage2_limit <= 12) else 5)
    )
    return stage1_limit, stage2_limit, mb_size


def stage1_passes_gate(
    s1_trajs: list[AgentTrajectory],
    archive: ParetoArchive,
    method: EvolutionMethod,
) -> tuple[bool, float, float]:
    """Evaluate Stage-1 stratified validation gate against existing Pareto archive."""
    s1_ex = sum(1.0 for t in s1_trajs if t.ex_match) / max(1, len(s1_trajs))
    s1_gw = sum(1.0 for t in s1_trajs if t.first_pass_gateway) / max(1, len(s1_trajs))
    if not archive.candidates:
        return True, s1_ex, s1_gw

    s1_ids = [t.task_id for t in s1_trajs]
    cand_scores = {t.task_id: score_trajectory_for_pareto(t) for t in s1_trajs}

    best_existing_s1_ex = 0.0
    wins_or_ties_any_task = False
    strictly_wins_any_task = False
    for rec in archive.candidates.values():
        rec_s1_hits = [
            1.0 if rec.scores_by_task.get(tid, -1.0) >= 1.0 else 0.0
            for tid in s1_ids
            if tid in rec.scores_by_task
        ]
        if rec_s1_hits:
            best_existing_s1_ex = max(best_existing_s1_ex, sum(rec_s1_hits) / len(rec_s1_hits))

    for tid in s1_ids:
        existing_max = max(
            (rec.scores_by_task.get(tid, -1.0) for rec in archive.candidates.values()),
            default=-1.0,
        )
        if cand_scores[tid] >= existing_max - 1e-9:
            wins_or_ties_any_task = True
        if cand_scores[tid] > existing_max + 1e-9:
            strictly_wins_any_task = True

    if s1_ex < best_existing_s1_ex - 0.2001 and not strictly_wins_any_task:
        return False, s1_ex, s1_gw
    if method == "greedy" and s1_ex < best_existing_s1_ex - 0.0001 and not strictly_wins_any_task:
        return False, s1_ex, s1_gw
    return wins_or_ties_any_task or (s1_ex >= best_existing_s1_ex - 0.2001), s1_ex, s1_gw


def evolved_modules_in_archive(archive: ParetoArchive, seed_policy: Policy) -> set[ModuleName]:
    """Return the set of modules whose prompt differs from `seed_policy` in any archived candidate."""
    mods: set[ModuleName] = set()
    for rec in archive.candidates.values():
        for m_name in ("linker", "generator", "refiner"):
            if rec.policy.get_prompt(m_name).strip() != seed_policy.get_prompt(m_name).strip():  # type: ignore[arg-type]
                mods.add(m_name)  # type: ignore[arg-type]
    return mods


def select_parent_and_preferred_module(
    *,
    method: EvolutionMethod,
    is_deep_suite: bool,
    last_op: str,
    cand_idx: int,
    evolved_mods_now: set[ModuleName],
    seed_policy: Policy,
    archive: ParetoArchive,
    current_greedy_best: CandidateRecord,
    rng: random.Random,
) -> tuple[Policy, ModuleName | None]:
    """Select parent policy and optional specialist target module for a mutation generation."""
    preferred_mod: ModuleName | None = None
    if method == "greedy":
        return current_greedy_best.policy, None

    if is_deep_suite:
        uncovered_mods: list[ModuleName] = [
            m_cand
            for m_cand in ("linker", "generator", "refiner")  # type: ignore[misc]
            if m_cand not in evolved_mods_now
        ]
        if last_op == "merge":
            merged_cands = [
                r for r in archive.candidates.values() if r.policy.mutation_target == "merge"
            ]
            if merged_cands:
                parent_policy = max(
                    merged_cands,
                    key=lambda r: (r.val_ex, r.mean_score, r.first_pass_gw_rate, r.policy.generation),
                ).policy
            else:
                parent_policy = archive.best_scalar_candidate().policy
            parent_uncovered: list[ModuleName] = [
                m_cand
                for m_cand in ("linker", "generator", "refiner")  # type: ignore[misc]
                if parent_policy.get_prompt(m_cand).strip() == seed_policy.get_prompt(m_cand).strip()
            ]
            if parent_uncovered:
                preferred_mod = parent_uncovered[0]
        elif uncovered_mods and cand_idx <= 4:
            preferred_mod = uncovered_mods[0]
            parent_policy = seed_policy if cand_idx <= 3 else archive.sample_parent(rng)
        else:
            parent_policy = archive.sample_parent(rng)
        return parent_policy, preferred_mod

    frontier = archive.get_pareto_frontier()
    if cand_idx == 2 and len(frontier) == 1 and seed_policy.policy_id not in frontier:
        parent_policy = seed_policy
    else:
        parent_policy = archive.sample_parent(rng)
    return parent_policy, None


def select_generation_train_batch(
    *,
    preferred_mod: ModuleName | None,
    is_deep_suite: bool,
    method: EvolutionMethod,
    hard_train_ids: list[str],
    train_task_map: dict[str, BenchmarkTask],
    benchmark_train: list[BenchmarkTask],
    train_batches: list[list[BenchmarkTask]],
    batch_cursor: int,
    cand_idx: int,
    mb_size: int,
) -> tuple[list[BenchmarkTask], int]:
    """Select training minibatch for reflective diagnosis and smoke gating."""
    if preferred_mod is not None and is_deep_suite:
        mod_cat_map = {
            "linker": ("semantic_linking_trap", "partition_pruning", "json_array_unnest"),
            "generator": ("scd2_temporal_join", "retention_date_math", "fanout_dedup"),
            "refiner": ("silent_caliber_audit", "ratio_null_handling"),
        }
        pref_cats = mod_cat_map[preferred_mod]
        cat_pool = [t for t in benchmark_train if t.category in pref_cats]
        if len(cat_pool) >= mb_size:
            offset = ((cand_idx - 1) * mb_size) % max(1, len(cat_pool))
            doubled = cat_pool + cat_pool
            return doubled[offset : offset + mb_size], batch_cursor
        batch = train_batches[batch_cursor % len(train_batches)]
        return batch, batch_cursor + 1

    if is_deep_suite and method == "gepa" and hard_train_ids:
        train_batch = [train_task_map[tid] for tid in hard_train_ids[:mb_size]]
        if len(train_batch) < mb_size:
            fallback_b = train_batches[batch_cursor % len(train_batches)]
            batch_cursor += 1
            for item in fallback_b:
                if item not in train_batch and len(train_batch) < mb_size:
                    train_batch.append(item)
        return train_batch, batch_cursor

    batch = train_batches[batch_cursor % len(train_batches)]
    return batch, batch_cursor + 1


def build_stratified_minibatches(
    train_tasks: list[BenchmarkTask],
    batch_size: int,
) -> list[list[BenchmarkTask]]:
    """Group train tasks into balanced minibatches spanning categories cleanly (v1, hard_v2, or all)."""
    ordered = sorted(train_tasks, key=lambda t: t.task_id)
    batches: list[list[BenchmarkTask]] = []
    v1_cats = [
        "partition_pruning",
        "json_array_unnest",
        "retention_date_math",
        "fanout_dedup",
        "ratio_null_handling",
    ]
    present_cats = {t.category for t in ordered}
    if set(v1_cats).issubset(present_cats) and len(present_cats) == 5:
        cats_order = v1_cats
        by_cat = {c: [t for t in ordered if t.category == c] for c in cats_order}
        b1 = [by_cat["partition_pruning"][0], by_cat["json_array_unnest"][0], by_cat["partition_pruning"][1]][:batch_size]
        b2 = [by_cat["retention_date_math"][0], by_cat["fanout_dedup"][0], by_cat["retention_date_math"][1]][:batch_size]
        b3 = [by_cat["ratio_null_handling"][0], by_cat["json_array_unnest"][1], by_cat["fanout_dedup"][1]][:batch_size]
        batches.extend([b1, b2, b3])
    else:
        preferred_order = [
            "semantic_linking_trap",
            "scd2_temporal_join",
            "silent_caliber_audit",
        ] + v1_cats
        cats_order = [c for c in preferred_order if c in present_cats] + sorted(
            c for c in present_cats if c not in preferred_order
        )
        by_cat = {c: [t for t in ordered if t.category == c] for c in cats_order}

    used_ids = {t.task_id for b in batches for t in b}
    flat: list[BenchmarkTask] = []
    max_len = max((len(v) for v in by_cat.values()), default=0)
    for i in range(max_len):
        for c in cats_order:
            if i < len(by_cat[c]) and by_cat[c][i].task_id not in used_ids:
                flat.append(by_cat[c][i])
                used_ids.add(by_cat[c][i].task_id)
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        if chunk:
            batches.append(chunk)
    return batches
