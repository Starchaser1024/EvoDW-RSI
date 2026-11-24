"""Content-addressed trajectory evaluation cache and rollout budget tracker for GEPA evolution."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.agent.pipeline import MultiModuleSQLAgent
from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.tasks import BenchmarkTask
from src.env.sandbox import DialectType
from src.llm.client import BaseLLMClient
from src.rsi.pareto import CandidateRecord, ParetoArchive


def load_disk_trajectory_cache(
    traj_cache_file: Path | None,
    eval_cache: dict[tuple[str, str], AgentTrajectory],
) -> set[tuple[str, str]]:
    """Load persisted JSONL trajectories into `eval_cache` if `traj_cache_file` exists."""
    loaded_keys: set[tuple[str, str]] = set()
    if traj_cache_file is not None and traj_cache_file.exists():
        for line in traj_cache_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                k = (str(row["key"][0]), str(row["key"][1]))
                eval_cache[k] = AgentTrajectory.from_dict(row["traj"])
                loaded_keys.add(k)
            except Exception:
                pass
    return loaded_keys


class RolloutEvalSession:
    """Per-run session managing content-hash caching, disk persistence, and rollout budget accounting."""

    def __init__(
        self,
        *,
        agent: MultiModuleSQLAgent,
        llm: BaseLLMClient,
        dialect: DialectType,
        eval_cache: dict[tuple[str, str], AgentTrajectory],
        traj_cache_file: Path | None,
        wall_limit_sec: float | None,
        t_run_start: float,
    ) -> None:
        self.agent = agent
        self.llm = llm
        self.dialect = dialect
        self.eval_cache = eval_cache
        self.traj_cache_file = traj_cache_file
        self.wall_limit_sec = wall_limit_sec
        self.t_run_start = t_run_start

        self.verified_pool: list[AgentTrajectory] = []
        self.counted_pairs: set[tuple[str, str]] = set()
        self.ledger_credited: set[tuple[str, str]] = set()
        self.rollouts_used: int = 0
        self.run_cache_hits: int = 0

    def pkey(self, policy: Policy, task_id: str) -> tuple[str, str]:
        return (policy.content_hash(self.dialect), task_id)

    def incremental_cost(self, tasks: list[BenchmarkTask], policy: Policy) -> int:
        return sum(1 for t in tasks if (policy.policy_id, t.task_id) not in self.counted_pairs)

    async def eval_cached(
        self,
        tasks: list[BenchmarkTask],
        policy: Policy,
        *,
        count_budget: bool = True,
        record_verified: bool = True,
    ) -> list[AgentTrajectory]:
        phash = policy.content_hash(self.dialect)
        uncached = [t for t in tasks if (phash, t.task_id) not in self.eval_cache]
        hits = len(tasks) - len(uncached)
        self.run_cache_hits += hits

        fresh_tids: set[str] = set()
        if uncached:
            rem_wall: float | None = None
            if self.wall_limit_sec is not None:
                rem_wall = self.wall_limit_sec - (time.monotonic() - self.t_run_start)

            def _persist_one(task_item: BenchmarkTask, traj_item: AgentTrajectory) -> None:
                ck = (phash, task_item.task_id)
                self.eval_cache[ck] = traj_item
                fresh_tids.add(task_item.task_id)
                self.ledger_credited.add(ck)
                if self.traj_cache_file is not None:
                    self.traj_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with self.traj_cache_file.open("a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {"key": [phash, task_item.task_id], "traj": traj_item.to_dict()},
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n"
                        )

            await self.agent.evaluate_tasks(
                uncached,
                policy,
                dialect=self.dialect,
                on_complete=_persist_one,
                wall_timeout_sec=rem_wall,
            )

        out: list[AgentTrajectory] = []
        for task_obj in tasks:
            ck = (phash, task_obj.task_id)
            raw_traj = self.eval_cache[ck]
            if raw_traj.policy_id != policy.policy_id:
                traj = AgentTrajectory.from_dict(
                    raw_traj.to_dict(), override_policy_id=policy.policy_id
                )
            else:
                traj = raw_traj

            if ck not in self.ledger_credited and task_obj.task_id not in fresh_tids:
                self.ledger_credited.add(ck)
                for mcall in traj.module_calls:
                    self.llm.ledger.actor.calls += 1
                    self.llm.ledger.actor.prompt_tokens += mcall.prompt_tokens
                    self.llm.ledger.actor.completion_tokens += mcall.completion_tokens
                    self.llm.ledger.actor.total_tokens += mcall.total_tokens
                    self.llm.ledger.actor.total_latency_ms += mcall.latency_ms

            out.append(traj)
            bkey = (policy.policy_id, task_obj.task_id)
            if count_budget and bkey not in self.counted_pairs:
                self.counted_pairs.add(bkey)
                self.rollouts_used += 1
            if record_verified and traj.ex_match and traj.split in ("train", "val"):
                self.verified_pool.append(traj)
        return out


def build_ordered_finalists(
    archive: ParetoArchive,
    *,
    method: str,
    current_greedy_best: CandidateRecord,
    seed_policy_id: str,
) -> list[str]:
    """Order candidate policy IDs for Stage-2 stratified finalist verification."""
    ranked_cids = [
        r.policy.policy_id
        for r in sorted(
            archive.candidates.values(),
            key=lambda r: (r.val_ex, r.mean_score, r.first_pass_gw_rate, r.policy.generation),
            reverse=True,
        )
    ]
    top_cid = current_greedy_best.policy.policy_id if method == "greedy" else ranked_cids[0]
    frontier_now = archive.get_pareto_frontier()
    ordered_finalists: list[str] = []
    for cid in [top_cid, seed_policy_id] + frontier_now + ranked_cids:
        if cid in archive.candidates and cid not in ordered_finalists:
            ordered_finalists.append(cid)
    return ordered_finalists
