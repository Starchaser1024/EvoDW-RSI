"""GEPA Reflective Strategy Self-Evolution Engine with Cache-Reusing Two-Stage Validation."""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from src.agent.pipeline import MultiModuleSQLAgent, WallClockCheckpoint
from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.loader import LoadedBenchmark
from src.benchmarks.tasks import BenchmarkTask
from src.env.sandbox import DialectType
from src.eval.metrics import compute_eval_metrics
from src.llm.client import BaseLLMClient
from src.rsi.attribution import Reflector
from src.rsi.eval_cache import (
    RolloutEvalSession,
    build_ordered_finalists,
    load_disk_trajectory_cache,
)
from src.rsi.merger import SystemAwareMerger, find_lowest_common_ancestor
from src.rsi.pareto import ParetoArchive, score_trajectory_for_pareto
from src.rsi.scheduling import (
    EvolutionMethod,
    EvolutionRunResult,
    GenerationLog,
    build_stratified_minibatches,
    compute_stage_limits,
    evolved_modules_in_archive,
    select_generation_train_batch,
    select_parent_and_preferred_module,
    stage1_passes_gate,
)


class EvolutionEngine:
    """Orchestrates GEPA (Reflector + ParetoArchive + SystemAwareMerger) and Greedy/Seed baselines."""

    def __init__(
        self,
        *,
        llm: BaseLLMClient,
        benchmark: LoadedBenchmark,
        seed_policy: Policy,
        dialect: DialectType = "odps",
        seed: int = 2026,
        max_refine_turns: int = 2,
        eval_cache: dict[tuple[str, str], AgentTrajectory] | None = None,
        cache_dir: Path | None = None,
        wall_limit_sec: float | None = None,
    ) -> None:
        self.llm = llm
        self.benchmark = benchmark
        self.seed_policy = seed_policy
        self.dialect: DialectType = dialect
        self.rng = random.Random(seed)
        self.agent = MultiModuleSQLAgent(
            llm=llm,
            sandbox=benchmark.sandbox,
            max_refine_turns=max_refine_turns,
        )
        self.reflector = Reflector(llm=llm)
        self.merger = SystemAwareMerger(llm=llm)
        self.eval_cache: dict[tuple[str, str], AgentTrajectory] = (
            eval_cache if eval_cache is not None else {}
        )
        self.cache_dir = cache_dir
        self.wall_limit_sec = wall_limit_sec
        self._traj_cache_file: Path | None = (
            (cache_dir / "trajectories.jsonl") if cache_dir is not None else None
        )
        self._disk_loaded_keys = load_disk_trajectory_cache(self._traj_cache_file, self.eval_cache)
        self.cache_hits: int = 0

    @staticmethod
    def _build_stratified_minibatches(
        train_tasks: list[BenchmarkTask],
        batch_size: int,
    ) -> list[list[BenchmarkTask]]:
        return build_stratified_minibatches(train_tasks, batch_size)

    async def run(
        self,
        *,
        method: EvolutionMethod = "gepa",
        budget: int = 60,
        val_active_size: int | None = None,
        minibatch_size: int | None = None,
    ) -> EvolutionRunResult:
        """Run strategy optimization with cache-reusing two-stage stratified validation and full val/test eval."""
        stage1_limit, stage2_limit, mb_size = compute_stage_limits(
            full_val_len=len(self.benchmark.val),
            budget=budget,
            val_active_size=val_active_size,
            minibatch_size=minibatch_size,
        )

        stage1_val_tasks = self.benchmark.get_active_val(limit=stage1_limit)
        stage2_val_tasks = self.benchmark.get_active_val(limit=stage2_limit)
        eager_two_stage = (budget >= 80) or (
            val_active_size is not None and len(stage2_val_tasks) > len(stage1_val_tasks)
        )

        archive = ParetoArchive()
        policies_by_id: dict[str, Policy] = {self.seed_policy.policy_id: self.seed_policy}
        history: list[GenerationLog] = []
        history_by_pid: dict[str, GenerationLog] = {}
        t_run_start = time.monotonic()

        session = RolloutEvalSession(
            agent=self.agent,
            llm=self.llm,
            dialect=self.dialect,
            eval_cache=self.eval_cache,
            traj_cache_file=self._traj_cache_file,
            wall_limit_sec=self.wall_limit_sec,
            t_run_start=t_run_start,
        )

        async def _eval_cached(
            tasks: list[BenchmarkTask],
            policy: Policy,
            *,
            count_budget: bool = True,
            record_verified: bool = True,
        ) -> list[AgentTrajectory]:
            prev_hits = session.run_cache_hits
            res = await session.eval_cached(
                tasks,
                policy,
                count_budget=count_budget,
                record_verified=record_verified,
            )
            self.cache_hits += session.run_cache_hits - prev_hits
            return res

        if method == "seed":
            seed_val_trajs = await _eval_cached(self.benchmark.val, self.seed_policy, count_budget=False)
            seed_test_trajs = await _eval_cached(
                self.benchmark.test,
                self.seed_policy,
                count_budget=False,
                record_verified=False,
            )
            archive.add_candidate(self.seed_policy, seed_val_trajs)
            seed_val_m = compute_eval_metrics(seed_val_trajs)
            seed_test_m = compute_eval_metrics(seed_test_trajs)
            history.append(
                GenerationLog(
                    generation=0,
                    policy_id=self.seed_policy.policy_id,
                    parent_ids=[],
                    operation="seed",
                    target_module="seed",
                    smoke_passed=True,
                    active_val_ex=seed_val_m.ex,
                    active_val_first_pass_gw=seed_val_m.first_pass_gateway_rate,
                    frontier_size=1,
                    rollouts_used=0,
                    rationale="Initial generic seed policy evaluated on full validation set.",
                )
            )
            return EvolutionRunResult(
                method=method,
                dialect=self.dialect,
                budget=budget,
                rollouts_used=0,
                seed_policy=self.seed_policy,
                best_policy=self.seed_policy,
                policies_by_id=policies_by_id,
                history=history,
                archive=archive,
                seed_val_trajectories=seed_val_trajs,
                seed_test_trajectories=seed_test_trajs,
                best_val_trajectories=seed_val_trajs,
                best_test_trajectories=seed_test_trajs,
                seed_val_metrics=seed_val_m,
                seed_test_metrics=seed_test_m,
                best_val_metrics=seed_val_m,
                best_test_metrics=seed_test_m,
                verified_train_val_trajectories=session.verified_pool,
                llm_usage_summary=self.llm.ledger.summary(),
                cache_hits=session.run_cache_hits,
                effective_val_size=len(seed_val_trajs),
            )

        online_val_tasks = stage2_val_tasks if eager_two_stage else stage1_val_tasks
        seed_active_trajs = await _eval_cached(online_val_tasks, self.seed_policy, count_budget=True)
        seed_rec = archive.add_candidate(self.seed_policy, seed_active_trajs)
        seed_log = GenerationLog(
            generation=0,
            policy_id=self.seed_policy.policy_id,
            parent_ids=[],
            operation="seed",
            target_module="seed",
            smoke_passed=True,
            active_val_ex=seed_rec.val_ex,
            active_val_first_pass_gw=seed_rec.first_pass_gw_rate,
            frontier_size=len(archive.get_pareto_frontier()),
            rollouts_used=session.rollouts_used,
            rationale=f"Initial seed policy baseline on stratified val ({len(online_val_tasks)} tasks).",
        )
        history.append(seed_log)
        history_by_pid[self.seed_policy.policy_id] = seed_log

        train_batches = self._build_stratified_minibatches(self.benchmark.train, mb_size)
        train_task_map = {t.task_id: t for t in self.benchmark.train}
        hard_train_ids: list[str] = []
        suboptimal_train_by_pid: dict[str, list[AgentTrajectory]] = {}

        def _record_train_feedback(pid: str, trajs: list[AgentTrajectory]) -> list[AgentTrajectory]:
            sub = [t for t in trajs if not t.ex_match or not t.first_pass_gateway]
            if sub:
                existing = suboptimal_train_by_pid.setdefault(pid, [])
                seen_tids = {x.task_id for x in existing}
                for item in sub:
                    if item.task_id not in seen_tids:
                        existing.append(item)
                        seen_tids.add(item.task_id)
                    if item.task_id not in hard_train_ids:
                        hard_train_ids.append(item.task_id)
            return sub

        batch_cursor = 0
        cand_idx = 0
        last_op = "seed"
        merged_pairs_seen: set[tuple[str, str]] = set()
        current_greedy_best = seed_rec
        stage2_extra_per_cand = max(0, len(stage2_val_tasks) - len(stage1_val_tasks))

        while session.rollouts_used < budget:
            remaining = budget - session.rollouts_used
            need_stage2_reserve = (
                not eager_two_stage
                and stage2_extra_per_cand > 0
                and (
                    (method == "greedy" and cand_idx >= 4)
                    or (method == "gepa" and cand_idx >= (4 if budget <= 70 else 5))
                )
            )
            reserve_rollouts = (2 * stage2_extra_per_cand) if need_stage2_reserve else 0

            evolved_mods_now = evolved_modules_in_archive(archive, self.seed_policy)
            is_deep_suite = getattr(self.benchmark, "suite", "v1") in ("hard_v2", "all") or self.dialect == "hive"
            ready_to_merge = (
                (len(evolved_mods_now) >= 3 or cand_idx >= 3)
                if is_deep_suite
                else (cand_idx >= 2 or len(evolved_mods_now) >= 2)
            )
            min_step_cost_peek = 2 * mb_size + len(stage1_val_tasks)
            allow_merge_now = (last_op != "merge") or ((remaining - reserve_rollouts) < min_step_cost_peek)
            merge_pair = (
                self.merger.select_complementary_pair(archive, merged_pairs_seen, policies_by_id=policies_by_id)
                if method == "gepa" and ready_to_merge and allow_merge_now
                else None
            )

            if merge_pair is not None and (remaining - reserve_rollouts) >= len(stage1_val_tasks):
                cand_idx += 1
                generation = cand_idx
                new_pid = f"{method}_g{generation}"
                rec_a, rec_b = merge_pair
                merged_pairs_seen.add(
                    (
                        min(rec_a.policy.policy_id, rec_b.policy.policy_id),
                        max(rec_a.policy.policy_id, rec_b.policy.policy_id),
                    )
                )
                ancestor = find_lowest_common_ancestor(
                    rec_a.policy.policy_id, rec_b.policy.policy_id, policies_by_id
                )
                if self.wall_limit_sec is not None and (time.monotonic() - t_run_start) >= self.wall_limit_sec - 25.0:
                    raise WallClockCheckpoint(f"Wall-clock checkpoint before merge g{generation}.")
                _merge_coro = self.merger.merge(
                    rec_a,
                    rec_b,
                    ancestor,
                    generation=generation,
                    new_policy_id=new_pid,
                    dialect=self.dialect,
                )
                if self.wall_limit_sec is not None:
                    _rem_m = max(10.0, self.wall_limit_sec - (time.monotonic() - t_run_start))
                    try:
                        child_policy, merge_meta = await asyncio.wait_for(_merge_coro, timeout=_rem_m)
                    except asyncio.TimeoutError:
                        raise WallClockCheckpoint(f"Wall-clock checkpoint during merge g{generation}.")
                else:
                    child_policy, merge_meta = await _merge_coro
                policies_by_id[new_pid] = child_policy

                s1_trajs = await _eval_cached(stage1_val_tasks, child_policy, count_budget=True)
                s1_ok, s1_ex, s1_gw = stage1_passes_gate(s1_trajs, archive, method)

                if eager_two_stage:
                    inc_s2 = session.incremental_cost(stage2_val_tasks, child_policy)
                    if s1_ok and (session.rollouts_used + inc_s2 <= budget):
                        val_trajs = await _eval_cached(stage2_val_tasks, child_policy, count_budget=True)
                        rec = archive.add_candidate(child_policy, val_trajs)
                        glog = GenerationLog(
                            generation=generation,
                            policy_id=new_pid,
                            parent_ids=child_policy.parent_ids,
                            operation="merge",
                            target_module="merge",
                            smoke_passed=True,
                            active_val_ex=rec.val_ex,
                            active_val_first_pass_gw=rec.first_pass_gw_rate,
                            frontier_size=len(archive.get_pareto_frontier()),
                            rollouts_used=session.rollouts_used,
                            rationale=merge_meta["merge_rationale"],
                        )
                    else:
                        glog = GenerationLog(
                            generation=generation,
                            policy_id=new_pid,
                            parent_ids=child_policy.parent_ids,
                            operation="merge",
                            target_module="merge",
                            smoke_passed=False,
                            active_val_ex=s1_ex,
                            active_val_first_pass_gw=s1_gw,
                            frontier_size=len(archive.get_pareto_frontier()),
                            rollouts_used=session.rollouts_used,
                            rationale=f"Stage-1 val screen dropped ({s1_ex*100:.1f}%): {merge_meta['merge_rationale']}",
                        )
                else:
                    rec = archive.add_candidate(child_policy, s1_trajs)
                    glog = GenerationLog(
                        generation=generation,
                        policy_id=new_pid,
                        parent_ids=child_policy.parent_ids,
                        operation="merge",
                        target_module="merge",
                        smoke_passed=True,
                        active_val_ex=rec.val_ex,
                        active_val_first_pass_gw=rec.first_pass_gw_rate,
                        frontier_size=len(archive.get_pareto_frontier()),
                        rollouts_used=session.rollouts_used,
                        rationale=merge_meta["merge_rationale"],
                    )
                history.append(glog)
                history_by_pid[new_pid] = glog
                last_op = "merge"
                print(
                    f"    [{method} g{generation}] {new_pid} ({glog.operation}:{glog.target_module}) "
                    f"smoke={glog.smoke_passed} val_ex={glog.active_val_ex} rollouts={session.rollouts_used}/{budget}",
                    flush=True,
                )
                continue

            if (remaining - reserve_rollouts) < (2 * mb_size + len(stage1_val_tasks)):
                break

            cand_idx += 1
            generation = cand_idx
            new_pid = f"{method}_g{generation}"

            parent_policy, preferred_mod = select_parent_and_preferred_module(
                method=method,
                is_deep_suite=is_deep_suite,
                last_op=last_op,
                cand_idx=cand_idx,
                evolved_mods_now=evolved_mods_now,
                seed_policy=self.seed_policy,
                archive=archive,
                current_greedy_best=current_greedy_best,
                rng=self.rng,
            )
            train_batch, batch_cursor = select_generation_train_batch(
                preferred_mod=preferred_mod,
                is_deep_suite=is_deep_suite,
                method=method,
                hard_train_ids=hard_train_ids,
                train_task_map=train_task_map,
                benchmark_train=self.benchmark.train,
                train_batches=train_batches,
                batch_cursor=batch_cursor,
                cand_idx=cand_idx,
                mb_size=mb_size,
            )

            if session.rollouts_used + session.incremental_cost(train_batch, parent_policy) > budget:
                break

            parent_train_trajs = await _eval_cached(train_batch, parent_policy, count_budget=True)
            sub_traces = _record_train_feedback(parent_policy.policy_id, parent_train_trajs)

            reflect_trajs = parent_train_trajs
            if not sub_traces:
                cached_subs = suboptimal_train_by_pid.get(parent_policy.policy_id, [])
                if cached_subs:
                    reflect_trajs = cached_subs
                else:
                    probe_candidates = [
                        train_task_map[tid]
                        for tid in hard_train_ids
                        if session.pkey(parent_policy, tid) not in self.eval_cache
                    ]
                    if len(probe_candidates) < mb_size:
                        next_b = train_batches[batch_cursor % len(train_batches)]
                        for t_item in next_b:
                            if (
                                session.pkey(parent_policy, t_item.task_id) not in self.eval_cache
                                and t_item not in probe_candidates
                            ):
                                probe_candidates.append(t_item)
                    probe_batch = probe_candidates[:mb_size]
                    if (
                        probe_batch
                        and (
                            session.rollouts_used
                            + session.incremental_cost(probe_batch, parent_policy)
                            + len(probe_batch)
                            + len(stage1_val_tasks)
                            + reserve_rollouts
                        )
                        <= budget
                    ):
                        batch_cursor += 1
                        probe_trajs = await _eval_cached(probe_batch, parent_policy, count_budget=True)
                        if _record_train_feedback(parent_policy.policy_id, probe_trajs):
                            train_batch = probe_batch
                            parent_train_trajs = probe_trajs
                            reflect_trajs = probe_trajs

            if self.wall_limit_sec is not None and (time.monotonic() - t_run_start) >= self.wall_limit_sec - 25.0:
                raise WallClockCheckpoint(f"Wall-clock checkpoint before reflect g{generation}.")
            _ref_coro = self.reflector.reflect_and_mutate(
                parent_policy,
                reflect_trajs,
                generation=generation,
                new_policy_id=new_pid,
                dialect=self.dialect,
                preferred_module=preferred_mod,
            )
            if self.wall_limit_sec is not None:
                _rem_r = max(10.0, self.wall_limit_sec - (time.monotonic() - t_run_start))
                try:
                    child_policy, attr_meta = await asyncio.wait_for(_ref_coro, timeout=_rem_r)
                except asyncio.TimeoutError:
                    raise WallClockCheckpoint(f"Wall-clock checkpoint during reflect g{generation}.")
            else:
                child_policy, attr_meta = await _ref_coro
            policies_by_id[new_pid] = child_policy

            if session.rollouts_used + session.incremental_cost(train_batch, child_policy) > budget:
                break
            child_smoke_trajs = await _eval_cached(train_batch, child_policy, count_budget=True)
            _record_train_feedback(child_policy.policy_id, child_smoke_trajs)

            parent_mb_score = sum(score_trajectory_for_pareto(t) for t in parent_train_trajs)
            child_mb_score = sum(score_trajectory_for_pareto(t) for t in child_smoke_trajs)
            smoke_passed = child_mb_score >= parent_mb_score - 1e-9 and (
                child_mb_score > 0.0 or parent_mb_score == 0.0
            )

            if smoke_passed and (session.rollouts_used + session.incremental_cost(stage1_val_tasks, child_policy) <= budget):
                s1_trajs = await _eval_cached(stage1_val_tasks, child_policy, count_budget=True)
                s1_ok, s1_ex, s1_gw = stage1_passes_gate(s1_trajs, archive, method)

                val_for_cand: list[AgentTrajectory] | None = None
                if eager_two_stage:
                    inc_s2 = session.incremental_cost(stage2_val_tasks, child_policy)
                    if s1_ok and (session.rollouts_used + inc_s2 <= budget):
                        val_for_cand = await _eval_cached(stage2_val_tasks, child_policy, count_budget=True)
                elif s1_ok:
                    val_for_cand = s1_trajs

                if val_for_cand is not None:
                    rec = archive.add_candidate(child_policy, val_for_cand)
                    if method == "greedy" and (
                        rec.val_ex,
                        rec.mean_score,
                        rec.first_pass_gw_rate,
                    ) > (
                        current_greedy_best.val_ex,
                        current_greedy_best.mean_score,
                        current_greedy_best.first_pass_gw_rate,
                    ):
                        current_greedy_best = rec
                    glog = GenerationLog(
                        generation=generation,
                        policy_id=new_pid,
                        parent_ids=child_policy.parent_ids,
                        operation="mutate",
                        target_module=attr_meta["target_module"],
                        smoke_passed=True,
                        active_val_ex=rec.val_ex,
                        active_val_first_pass_gw=rec.first_pass_gw_rate,
                        frontier_size=len(archive.get_pareto_frontier()),
                        rollouts_used=session.rollouts_used,
                        rationale=attr_meta["fault_diagnosis"],
                    )
                else:
                    glog = GenerationLog(
                        generation=generation,
                        policy_id=new_pid,
                        parent_ids=child_policy.parent_ids,
                        operation="mutate",
                        target_module=attr_meta["target_module"],
                        smoke_passed=False,
                        active_val_ex=s1_ex,
                        active_val_first_pass_gw=s1_gw,
                        frontier_size=len(archive.get_pareto_frontier()),
                        rollouts_used=session.rollouts_used,
                        rationale=f"Stage-1 val screen not cleared (EX={s1_ex*100:.1f}%): {attr_meta['fault_diagnosis']}",
                    )
                history.append(glog)
                history_by_pid[new_pid] = glog
            else:
                glog = GenerationLog(
                    generation=generation,
                    policy_id=new_pid,
                    parent_ids=child_policy.parent_ids,
                    operation="mutate",
                    target_module=attr_meta["target_module"],
                    smoke_passed=False,
                    active_val_ex=None,
                    active_val_first_pass_gw=None,
                    frontier_size=len(archive.get_pareto_frontier()),
                    rollouts_used=session.rollouts_used,
                    rationale=f"Minibatch smoke gate not cleared ({child_mb_score:.2f} vs {parent_mb_score:.2f}): {attr_meta['fault_diagnosis']}",
                )
                history.append(glog)
                history_by_pid[new_pid] = glog
            last_op = "mutate"
            print(
                f"    [{method} g{generation}] {new_pid} ({glog.operation}:{glog.target_module}) "
                f"smoke={glog.smoke_passed} val_ex={glog.active_val_ex} rollouts={session.rollouts_used}/{budget}",
                flush=True,
            )

        effective_val_tasks = stage2_val_tasks if eager_two_stage else stage1_val_tasks
        if not eager_two_stage and len(stage2_val_tasks) > len(stage1_val_tasks):
            ordered_finalists = build_ordered_finalists(
                archive,
                method=method,
                current_greedy_best=current_greedy_best,
                seed_policy_id=self.seed_policy.policy_id,
            )
            min_pair_cost = sum(
                session.incremental_cost(stage2_val_tasks, policies_by_id[c])
                for c in ordered_finalists[: min(2, len(ordered_finalists))]
            )
            if session.rollouts_used + min_pair_cost <= budget:
                promoted_any = False
                for cid in ordered_finalists:
                    pol = policies_by_id[cid]
                    inc = session.incremental_cost(stage2_val_tasks, pol)
                    if session.rollouts_used + inc <= budget:
                        s2_trajs = await _eval_cached(stage2_val_tasks, pol, count_budget=True)
                        rec = archive.add_candidate(pol, s2_trajs)
                        promoted_any = True
                        if cid in history_by_pid:
                            h_entry = history_by_pid[cid]
                            h_entry.active_val_ex = rec.val_ex
                            h_entry.active_val_first_pass_gw = rec.first_pass_gw_rate
                            h_entry.rollouts_used = max(h_entry.rollouts_used, session.rollouts_used)
                    else:
                        break

                if promoted_any:
                    effective_val_tasks = stage2_val_tasks
                    max_d = max(r.num_eval_tasks for r in archive.candidates.values())
                    s2_recs = [r for r in archive.candidates.values() if r.num_eval_tasks == max_d]
                    for h_entry in history:
                        if (
                            h_entry.policy_id in archive.candidates
                            and archive.candidates[h_entry.policy_id].num_eval_tasks == max_d
                        ):
                            h_entry.frontier_size = len(archive.get_pareto_frontier())
                    if method == "greedy" and s2_recs:
                        current_greedy_best = max(
                            s2_recs,
                            key=lambda r: (r.val_ex, r.mean_score, r.first_pass_gw_rate),
                        )

        best_rec = current_greedy_best if method == "greedy" else archive.best_scalar_candidate()
        best_policy = best_rec.policy

        seed_val_trajs = await _eval_cached(self.benchmark.val, self.seed_policy, count_budget=False)
        best_val_trajs = await _eval_cached(self.benchmark.val, best_policy, count_budget=False)
        seed_test_trajs = await _eval_cached(
            self.benchmark.test,
            self.seed_policy,
            count_budget=False,
            record_verified=False,
        )
        best_test_trajs = await _eval_cached(
            self.benchmark.test,
            best_policy,
            count_budget=False,
            record_verified=False,
        )

        return EvolutionRunResult(
            method=method,
            dialect=self.dialect,
            budget=budget,
            rollouts_used=session.rollouts_used,
            seed_policy=self.seed_policy,
            best_policy=best_policy,
            policies_by_id=policies_by_id,
            history=history,
            archive=archive,
            seed_val_trajectories=seed_val_trajs,
            seed_test_trajectories=seed_test_trajs,
            best_val_trajectories=best_val_trajs,
            best_test_trajectories=best_test_trajs,
            seed_val_metrics=compute_eval_metrics(seed_val_trajs),
            seed_test_metrics=compute_eval_metrics(seed_test_trajs),
            best_val_metrics=compute_eval_metrics(best_val_trajs),
            best_test_metrics=compute_eval_metrics(best_test_trajs),
            verified_train_val_trajectories=session.verified_pool,
            llm_usage_summary=self.llm.ledger.summary(),
            cache_hits=session.run_cache_hits,
            effective_val_size=len(effective_val_tasks),
        )


__all__ = [
    "EvolutionEngine",
    "EvolutionMethod",
    "EvolutionRunResult",
    "GenerationLog",
]
