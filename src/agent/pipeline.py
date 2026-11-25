"""Multi-module Text-to-SQL warehouse agent pipeline (Linker -> Generator -> Sandbox -> Refiner)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import re
from typing import Any

from src.agent.generator import DialectSQLGenerator, clean_sql_text
from src.agent.linker import WarehouseLinker
from src.agent.policy import Policy
from src.agent.refiner import GatewayRefiner
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.tasks import BenchmarkTask
from src.env.sandbox import DialectType, WarehouseSandbox
from src.llm.client import BaseLLMClient


class WallClockCheckpoint(Exception):
    """Raised when a batch stops gracefully at a wall-clock checkpoint after flushing completed tasks."""


_SANITY_AUDIT_POLICY_CUES = (
    "sanity",
    "caliber",
    "non-empty",
    "silent",
    "post-execution",
    "left join",
    "rows between",
    "last_value",
    "window frame",
    "audit",
    "口径",
    "静默",
    "非空",
    "窗口帧",
    "复核",
    "审计",
)

_AUDIT_SURFACE_SQL_OR_SCHEMA_CUES = (
    " join ",
    " over(",
    " over (",
    "eff_start_dt",
    "eff_end_dt",
    "is_test_acct",
    "abnormal_flag",
    "gross_gmv_yuan",
    "_cent",
    "_bps",
    "_code",
    "merchant_net_amt",
    "shipping_comp_yuan",
)


def _norm_sql_compare(sql: str) -> str:
    return re.sub(r"\s+", " ", clean_sql_text(sql)).strip().lower()


class MultiModuleSQLAgent:
    """Executes the 3-module warehouse SQL agent under a given Policy triplet."""

    def __init__(
        self,
        llm: BaseLLMClient,
        sandbox: WarehouseSandbox,
        max_refine_turns: int = 2,
        max_total_llm_calls: int | None = None,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.max_refine_turns = max_refine_turns
        self.max_total_llm_calls = max_total_llm_calls
        self.linker = WarehouseLinker(llm=llm, sandbox=sandbox)
        self.generator = DialectSQLGenerator(llm=llm)
        self.refiner = GatewayRefiner(llm=llm)

    @staticmethod
    def _policy_enables_sanity_audit(refiner_prompt: str) -> bool:
        p_low = refiner_prompt.lower()
        return any(cue in p_low for cue in _SANITY_AUDIT_POLICY_CUES)

    @staticmethod
    def _sql_warrants_sanity_audit(sql: str, linked_schema_text: str) -> bool:
        blob = f" {sql} \n {linked_schema_text} ".lower()
        return any(cue in blob for cue in _AUDIT_SURFACE_SQL_OR_SCHEMA_CUES)

    async def run_task(
        self,
        task: BenchmarkTask | None = None,
        *,
        policy: Policy,
        dialect: DialectType | None = None,
        question: str | None = None,
        hint: str = "",
        task_id: str = "adhoc",
        category: str = "adhoc",
        split: str = "adhoc",
        evaluate: bool = True,
    ) -> AgentTrajectory:
        active_dialect: DialectType = dialect or self.sandbox.default_dialect
        q_text = task.question if task is not None else (question or "")
        q_hint = task.hint if task is not None else hint
        q_id = task.task_id if task is not None else task_id
        q_cat = task.category if task is not None else category
        q_split = task.split if task is not None else split

        # 1. WarehouseLinker: select tables and optionally probe partitions/samples.
        link_res = await self.linker.link(
            system_prompt=policy.linker_prompt,
            question=q_text,
            hint=q_hint,
            dialect=active_dialect,
            step_offset=0,
        )

        # 2. DialectSQLGenerator: synthesize dialect-compliant SQL.
        gen_step = len(link_res.invocations)
        gen_res = await self.generator.generate(
            system_prompt=policy.generator_prompt,
            question=q_text,
            hint=q_hint,
            dialect=active_dialect,
            linked_schema_text=link_res.linked_schema_text,
            step_index=gen_step,
        )

        # 3. Execute generated SQL against the two-layer sandbox.
        first_exec = self.sandbox.run_sql(gen_res.sql, dialect=active_dialect)
        first_pass_gw = bool(first_exec["gateway_passed"])

        module_calls = list(link_res.invocations) + [gen_res.invocation]
        turns: list[dict[str, Any]] = [
            {
                "role": "system",
                "module": "policy",
                "content": (
                    f"[Policy={policy.policy_id} | Dialect={active_dialect.upper()}]\n"
                    f"## Linker Policy\n{policy.linker_prompt}\n\n"
                    f"## Generator Policy\n{policy.generator_prompt}\n\n"
                    f"## Refiner Policy\n{policy.refiner_prompt}"
                ),
                "loss_mask": 0,
            }
        ]
        turns.extend(link_res.turns)
        turns.extend(gen_res.turns)

        # 4. Iterative GatewayRefiner on gateway blocks, runtime errors, empty outputs,
        #    OR a 1-pass Post-Execution Sanity & Caliber Audit when evolved Refiner policy enables it.
        current_sql = gen_res.sql
        current_exec = first_exec
        refine_steps: list[dict[str, Any]] = []
        sanity_audit_done = False

        for r_turn in range(1, self.max_refine_turns + 1):
            if self.max_total_llm_calls is not None and len(module_calls) >= self.max_total_llm_calls:
                break
            suspicious, reason = self.sandbox.is_suspicious_result(current_exec)
            is_sanity_audit = False
            if not suspicious:
                if (
                    r_turn == 1
                    and not sanity_audit_done
                    and self._policy_enables_sanity_audit(policy.refiner_prompt)
                    and self._sql_warrants_sanity_audit(current_sql, link_res.linked_schema_text)
                ):
                    is_sanity_audit = True
                    sanity_audit_done = True
                    reason = (
                        "POST_EXECUTION_SANITY_AUDIT: Query passed gateway and returned non-empty rows. "
                        "Audit the SQL and sample rows against the business question, hint, and sample observations "
                        "for silent caliber traps (LEFT JOIN right-table WHERE filtering, missing ROWS BETWEEN window "
                        "frames on LAST_VALUE/FIRST_VALUE/peer ORDER BY, unexcluded test/abnormal records, SCD2 date "
                        "format mismatch or fanout, cent/bps vs yuan scaling, or unprobed enum codes). "
                        "If the SQL is already 100% accurate, output the exact same SQL unchanged."
                    )
                else:
                    break

            ref_res = await self.refiner.refine(
                system_prompt=policy.refiner_prompt,
                question=q_text,
                hint=q_hint,
                dialect=active_dialect,
                linked_schema_text=link_res.linked_schema_text,
                previous_sql=current_sql,
                exec_res=current_exec,
                suspicion_reason=reason,
                prior_refine_steps=refine_steps,
                step_index=len(module_calls),
            )
            module_calls.append(ref_res.invocation)
            turns.extend(ref_res.turns)

            if is_sanity_audit:
                if _norm_sql_compare(ref_res.sql) == _norm_sql_compare(current_sql):
                    break
                audit_exec = self.sandbox.run_sql(ref_res.sql, dialect=active_dialect)
                audit_susp, _ = self.sandbox.is_suspicious_result(audit_exec)
                if not audit_susp:
                    refine_steps.append(
                        {
                            "turn": r_turn,
                            "trigger_reason": reason,
                            "thought": ref_res.thought,
                            "diagnosis": ref_res.diagnosis,
                            "sql": ref_res.sql,
                            "exec_res": audit_exec,
                        }
                    )
                    current_sql = ref_res.sql
                    current_exec = audit_exec
                # Only 1 sanity-audit pass on already non-empty valid results.
                break

            new_exec = self.sandbox.run_sql(ref_res.sql, dialect=active_dialect)
            refine_steps.append(
                {
                    "turn": r_turn,
                    "trigger_reason": reason,
                    "thought": ref_res.thought,
                    "diagnosis": ref_res.diagnosis,
                    "sql": ref_res.sql,
                    "exec_res": new_exec,
                }
            )
            current_sql = ref_res.sql
            current_exec = new_exec

        # Record final sandbox observation in trajectory turns.
        final_obs = {
            "gateway_passed": current_exec.get("gateway_passed", False),
            "error_stage": current_exec.get("error_stage", "none"),
            "error_msg": current_exec.get("error_msg", ""),
            "row_count": current_exec.get("row_count", 0),
            "sample_rows": [str(r) for r in (current_exec.get("rows") or [])[:5]],
        }
        turns.append(
            {
                "role": "tool",
                "name": "sandbox_execute",
                "module": "sandbox",
                "content": json.dumps(final_obs, ensure_ascii=False),
                "loss_mask": 0,
            }
        )

        eval_outcome = None
        reward_info: dict[str, float] = {}
        if evaluate and task is not None:
            eval_outcome = self.sandbox.evaluate_sql(current_sql, task, dialect=active_dialect)
            reward_info = self.sandbox.compute_reward(
                current_sql,
                task,
                refine_turns=len(refine_steps),
                first_pass_gateway=first_pass_gw,
                dialect=active_dialect,
            )

        return AgentTrajectory(
            task_id=q_id,
            category=q_cat,
            split=q_split,
            question=q_text,
            hint=q_hint,
            dialect=active_dialect,
            policy_id=policy.policy_id,
            linked_tables=link_res.selected_tables,
            probe_requests=link_res.probe_requests,
            probe_results=link_res.probe_results,
            linked_schema_context=link_res.linked_schema_text,
            linker_notes=link_res.schema_notes,
            initial_sql=gen_res.sql,
            first_pass_gateway=first_pass_gw,
            first_pass_exec=first_exec,
            refine_steps=refine_steps,
            final_sql=current_sql,
            final_exec=current_exec,
            eval_outcome=eval_outcome,
            reward_info=reward_info,
            module_calls=module_calls,
            conversation_turns=turns,
        )

    async def evaluate_tasks(
        self,
        tasks: list[BenchmarkTask],
        policy: Policy,
        dialect: DialectType | None = None,
        *,
        on_complete: Callable[[BenchmarkTask, AgentTrajectory], None] | None = None,
        wall_timeout_sec: float | None = None,
    ) -> list[AgentTrajectory]:
        """Run the agent across a batch of benchmark tasks concurrently with incremental callbacks."""
        if not tasks:
            return []
        if wall_timeout_sec is not None and wall_timeout_sec <= 1.0:
            raise WallClockCheckpoint(f"Wall-clock checkpoint reached before batch ({len(tasks)} tasks remaining).")

        async def _indexed_run(idx: int, task_item: BenchmarkTask) -> tuple[int, AgentTrajectory]:
            traj = await self.run_task(task_item, policy=policy, dialect=dialect, evaluate=True)
            if on_complete is not None:
                on_complete(task_item, traj)
            return idx, traj

        if wall_timeout_sec is None:
            pairs = await asyncio.gather(*(_indexed_run(i, t) for i, t in enumerate(tasks)))
            return [traj for _, traj in sorted(pairs, key=lambda x: x[0])]

        fut_list = [asyncio.create_task(_indexed_run(i, t)) for i, t in enumerate(tasks)]
        done, pending = await asyncio.wait(fut_list, timeout=wall_timeout_sec)
        if pending:
            for p_fut in pending:
                p_fut.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for d_fut in done:
                if d_fut.exception() is not None:
                    raise d_fut.exception()  # type: ignore[misc]
            raise WallClockCheckpoint(
                f"Graceful wall-clock checkpoint: flushed {len(done)}/{len(tasks)} tasks in batch ({len(pending)} remaining)."
            )
        pairs = [f.result() for f in done]
        return [traj for _, traj in sorted(pairs, key=lambda x: x[0])]
