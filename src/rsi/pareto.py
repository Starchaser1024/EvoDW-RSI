"""Instance-level Pareto archive and win-frequency parent sampling for GEPA."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory


@dataclass
class CandidateRecord:
    """Evaluated policy entry stored inside the Pareto archive."""

    policy: Policy
    scores_by_task: dict[str, float]
    val_ex: float
    first_pass_gw_rate: float
    final_gw_rate: float
    by_category_ex: dict[str, float]
    mean_score: float
    trajectories: list[AgentTrajectory] = field(default_factory=list, repr=False)

    @property
    def num_eval_tasks(self) -> int:
        return len(self.scores_by_task)


def score_trajectory_for_pareto(traj: AgentTrajectory) -> float:
    """Primary EX accuracy + tie-breaker bonus for first-pass gateway compliance."""
    ex = 1.0 if traj.ex_match else 0.0
    gw1 = 1.0 if traj.first_pass_gateway else 0.0
    gw_final = 1.0 if traj.final_gateway_passed else 0.0
    return round(ex + 0.05 * gw1 + 0.02 * gw_final, 4)


class ParetoArchive:
    """Maintains per-task evaluation matrix Score[candidate][task] and instance-level Pareto frontier."""

    def __init__(self) -> None:
        self.candidates: dict[str, CandidateRecord] = {}
        self.insertion_order: list[str] = []

    def add_candidate(
        self,
        policy: Policy,
        val_trajectories: list[AgentTrajectory],
    ) -> CandidateRecord:
        scores_by_task: dict[str, float] = {}
        cat_hits: dict[str, list[float]] = defaultdict(list)

        ex_sum = 0.0
        gw1_sum = 0.0
        gwf_sum = 0.0
        for traj in val_trajectories:
            sc = score_trajectory_for_pareto(traj)
            scores_by_task[traj.task_id] = sc
            hit = 1.0 if traj.ex_match else 0.0
            cat_hits[traj.category].append(hit)
            ex_sum += hit
            gw1_sum += 1.0 if traj.first_pass_gateway else 0.0
            gwf_sum += 1.0 if traj.final_gateway_passed else 0.0

        n = max(1, len(val_trajectories))
        by_cat = {cat: sum(vals) / len(vals) for cat, vals in cat_hits.items()}
        mean_sc = sum(scores_by_task.values()) / n

        rec = CandidateRecord(
            policy=policy,
            scores_by_task=scores_by_task,
            val_ex=ex_sum / n,
            first_pass_gw_rate=gw1_sum / n,
            final_gw_rate=gwf_sum / n,
            by_category_ex=by_cat,
            mean_score=mean_sc,
            trajectories=list(val_trajectories),
        )
        self.candidates[policy.policy_id] = rec
        if policy.policy_id not in self.insertion_order:
            self.insertion_order.append(policy.policy_id)
        return rec

    def _active_pool_and_tasks(self) -> tuple[list[str], list[str]]:
        """Select candidates at the deepest validation stage (if two-stage promotion occurred) and shared tasks."""
        if not self.insertion_order:
            return [], []
        max_depth = max(self.candidates[cid].num_eval_tasks for cid in self.insertion_order)
        pool = [cid for cid in self.insertion_order if self.candidates[cid].num_eval_tasks == max_depth]
        task_ids = sorted(self.candidates[pool[0]].scores_by_task.keys()) if pool else []
        return pool, task_ids

    @staticmethod
    def _dominates(scores_b: dict[str, float], scores_a: dict[str, float], task_ids: list[str]) -> bool:
        """Return True if B is >= A on all tasks and strictly > A on at least one task."""
        strictly_better = False
        for tid in task_ids:
            sb = scores_b.get(tid, -1.0)
            sa = scores_a.get(tid, -1.0)
            if sb < sa - 1e-9:
                return False
            if sb > sa + 1e-9:
                strictly_better = True
        return strictly_better

    def get_pareto_frontier(self) -> list[str]:
        """Compute non-dominated instance-level Pareto frontier candidate IDs."""
        pool_ids, task_ids = self._active_pool_and_tasks()
        if not pool_ids:
            return []

        # Deduplicate exact identical score vectors unless a candidate brings a distinct evolved module footprint.
        seed_pol = self.candidates[self.insertion_order[0]].policy if self.insertion_order else None
        unique_ids: list[str] = []
        seen_vectors: set[tuple[tuple[float, ...], tuple[bool, bool, bool]]] = set()
        for cid in pool_ids:
            vec = tuple(self.candidates[cid].scores_by_task.get(tid, -1.0) for tid in task_ids)
            pol = self.candidates[cid].policy
            mod_sig = (
                (
                    pol.linker_prompt.strip() != seed_pol.linker_prompt.strip(),
                    pol.generator_prompt.strip() != seed_pol.generator_prompt.strip(),
                    pol.refiner_prompt.strip() != seed_pol.refiner_prompt.strip(),
                )
                if seed_pol is not None
                else (False, False, False)
            )
            sig_key = (vec, mod_sig)
            if sig_key not in seen_vectors:
                seen_vectors.add(sig_key)
                unique_ids.append(cid)

        # Retain candidates not strictly Pareto-dominated by any other candidate.
        non_dominated: list[str] = []
        for cid in unique_ids:
            sc_a = self.candidates[cid].scores_by_task
            dominated = any(
                other != cid and self._dominates(self.candidates[other].scores_by_task, sc_a, task_ids)
                for other in unique_ids
            )
            if not dominated:
                non_dominated.append(cid)

        # Keep candidates that tie or win on at least one task among the non-dominated pool.
        if not task_ids:
            return non_dominated

        instance_winners: set[str] = set()
        for tid in task_ids:
            max_sc = max(self.candidates[cid].scores_by_task.get(tid, -1.0) for cid in non_dominated)
            for cid in non_dominated:
                if abs(self.candidates[cid].scores_by_task.get(tid, -1.0) - max_sc) <= 1e-9:
                    instance_winners.add(cid)

        frontier = [cid for cid in non_dominated if cid in instance_winners]
        return frontier or non_dominated

    def get_win_frequency_weights(self) -> dict[str, float]:
        """Compute win-frequency weights across validation tasks for frontier candidates."""
        frontier = self.get_pareto_frontier()
        if not frontier:
            return {}
        _, task_ids = self._active_pool_and_tasks()
        if not task_ids:
            w = 1.0 / len(frontier)
            return {cid: w for cid in frontier}

        credits: dict[str, float] = {cid: 0.0 for cid in frontier}
        for tid in task_ids:
            max_sc = max(self.candidates[cid].scores_by_task.get(tid, -1.0) for cid in frontier)
            winners = [
                cid
                for cid in frontier
                if abs(self.candidates[cid].scores_by_task.get(tid, -1.0) - max_sc) <= 1e-9
            ]
            share = 1.0 / len(winners)
            for w_cid in winners:
                credits[w_cid] += share

        total_credit = sum(credits.values())
        if total_credit <= 0:
            uniform = 1.0 / len(frontier)
            return {cid: uniform for cid in frontier}
        return {cid: c / total_credit for cid, c in credits.items()}

    def sample_parent(self, rng: random.Random) -> Policy:
        """Sample a parent policy from the Pareto frontier proportional to win-frequency weights."""
        weights = self.get_win_frequency_weights()
        if not weights:
            raise RuntimeError("ParetoArchive is empty; cannot sample parent.")
        cids = list(weights.keys())
        probs = [weights[cid] for cid in cids]
        chosen = rng.choices(cids, weights=probs, k=1)[0]
        return self.candidates[chosen].policy

    def best_scalar_candidate(self) -> CandidateRecord:
        """Return the highest-mean-score candidate among deepest-verified entries (breaking ties by multi-module completeness)."""
        if not self.candidates:
            raise RuntimeError("ParetoArchive is empty.")
        seed_pol = self.candidates[self.insertion_order[0]].policy if self.insertion_order else None

        def _evolved_mod_count(pol: Policy) -> int:
            if seed_pol is None:
                return 0
            return int(pol.linker_prompt.strip() != seed_pol.linker_prompt.strip()) + int(
                pol.generator_prompt.strip() != seed_pol.generator_prompt.strip()
            ) + int(pol.refiner_prompt.strip() != seed_pol.refiner_prompt.strip())

        max_depth = max(r.num_eval_tasks for r in self.candidates.values())
        deepest = [r for r in self.candidates.values() if r.num_eval_tasks == max_depth]
        return max(
            deepest,
            key=lambda r: (
                r.val_ex,
                r.mean_score,
                r.first_pass_gw_rate,
                _evolved_mod_count(r.policy),
                r.policy.generation,
            ),
        )
