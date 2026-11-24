"""Quick 3-task smoke test verifying sandbox execution, dual-role LLM client, and multi-module trajectories."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import MultiModuleSQLAgent, Policy
from src.benchmarks import load_benchmark
from src.config import Settings
from src.llm import DualRoleLLMClient, MissingAPIKeyError, MockWarehouseLLMClient

console = Console()


async def run_smoke(args: argparse.Namespace) -> int:
    settings = Settings.from_env(
        actor_model=args.actor_model,
        dialect=args.dialect,
    )
    benchmark = load_benchmark(seed=2026, dialect=settings.dialect, precompute_gold=True)
    seed_policy = Policy.load_seed(settings.prompts_dir)

    use_mock = bool(args.mock) or (not getattr(args, "live", False) and not settings.has_usable_api_key)
    if use_mock:
        llm = MockWarehouseLLMClient(
            actor_model=settings.actor_model,
            reflector_model=settings.reflector_model,
            dialect=settings.dialect,
        )
        mode_label = "OFFLINE MOCK" if args.mock else "OFFLINE MOCK (auto-fallback: no live API key configured)"
    else:
        llm = DualRoleLLMClient(settings=settings)
        mode_label = f"LIVE ({settings.base_url})"

    console.print(
        Panel.fit(
            f"[bold cyan]EvoDW-RSI Smoke Test[/bold cyan]\n"
            f"Mode: {mode_label}\n"
            f"Actor: {settings.actor_model} | Dialect: {settings.dialect.upper()}"
        )
    )

    agent = MultiModuleSQLAgent(llm=llm, sandbox=benchmark.sandbox, max_refine_turns=settings.max_refine_turns)
    target_ids = ("pp_01", "ja_03", "fd_02")
    selected_tasks = [t for t in benchmark.all_tasks if t.task_id in target_ids]

    try:
        trajectories = []
        for task in selected_tasks:
            traj = await agent.run_task(task, policy=seed_policy, dialect=settings.dialect, evaluate=True)
            trajectories.append(traj)
    except MissingAPIKeyError as exc:
        console.print(f"[bold yellow]Configuration required:[/bold yellow] {exc}")
        console.print("Tip: Copy `.env.example` to `.env` with `LLM_API_KEY`, or pass `--mock`.")
        return 2
    finally:
        if isinstance(llm, DualRoleLLMClient):
            await llm.close()
        benchmark.sandbox.close()

    summary_tbl = Table(title="3-Task Smoke Test Summary")
    summary_tbl.add_column("Task ID", style="cyan")
    summary_tbl.add_column("Category")
    summary_tbl.add_column("1st-Pass GW")
    summary_tbl.add_column("Refine Turns")
    summary_tbl.add_column("EX Match")
    summary_tbl.add_column("Tokens")
    summary_tbl.add_column("Latency (ms)")

    for traj in trajectories:
        summary_tbl.add_row(
            traj.task_id,
            traj.category,
            "PASS" if traj.first_pass_gateway else "BLOCKED",
            str(traj.refine_turns),
            "[green]YES[/green]" if traj.ex_match else "[red]NO[/red]",
            str(traj.total_tokens),
            f"{traj.total_latency_ms:.1f}",
        )
    console.print(summary_tbl)

    for traj in trajectories:
        refine_txt = ""
        for step in traj.refine_steps:
            refine_txt += (
                f"\n  [Refine Turn {step['turn']}] Trigger: {step['trigger_reason']}\n"
                f"  Diagnosis: {step['diagnosis']}\n"
                f"  Repaired SQL: {step['sql']}\n"
            )
        console.print(
            Panel(
                f"[bold]Question:[/bold] {traj.question}\n"
                f"[bold]Linked Tables:[/bold] {traj.linked_tables} (probes: {len(traj.probe_requests)})\n"
                f"[bold]Initial SQL:[/bold] {traj.initial_sql}\n"
                f"[bold]1st-Pass Exec:[/bold] gw={traj.first_pass_gateway}, stage={traj.first_pass_exec.get('error_stage')}, err={traj.first_pass_exec.get('error_msg')}\n"
                f"{refine_txt}"
                f"[bold]Final SQL:[/bold] {traj.final_sql}\n"
                f"[bold]Evaluation:[/bold] EX={traj.ex_match} | Reward={traj.reward_info.get('total_reward')} | Reason={traj.eval_outcome.diff_reason if traj.eval_outcome else ''}",
                title=f"Trajectory: {traj.task_id} ({traj.outcome_label})",
                border_style="green" if traj.ex_match else "yellow",
            )
        )
    return 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(description="EvoDW-RSI 3-task smoke test.")
    p.add_argument("--dialect", choices=["odps", "hive"], default=None)
    p.add_argument("--actor-model", default=None)
    p.add_argument("--mock", action="store_true", help="Force deterministic offline MockWarehouseLLMClient.")
    p.add_argument("--live", action="store_true", help="Require live OpenAI-compatible API endpoint.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_smoke(args))


if __name__ == "__main__":
    raise SystemExit(main())
