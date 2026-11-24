"""CLI entry point to run GEPA / Greedy / Seed strategy evolution, reporting, and RL/SFT export."""

from __future__ import annotations
from pathlib import Path as _P
PROJECT_ROOT = _P(__file__).resolve().parents[1]

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import AgentTrajectory, Policy, WallClockCheckpoint
from src.benchmarks import load_benchmark
from src.config import Settings
from src.eval import generate_experiment_report
from src.export import export_sft_and_grpo_datasets
from src.llm import DualRoleLLMClient, MissingAPIKeyError, MockWarehouseLLMClient, UsageLedger
from src.rsi import EvolutionEngine, EvolutionRunResult

console = Console()


async def _execute(args: argparse.Namespace) -> int:
    settings = Settings.from_env(
        actor_model=args.actor_model,
        reflector_model=args.reflector_model,
        max_concurrency=args.concurrency,
        dialect=args.dialect,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir: Path | None = None
    if not args.mock and not args.no_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / ".live_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(seed=2026, dialect=settings.dialect, precompute_gold=True, suite=args.suite)
    seed_policy = Policy.load_seed(settings.prompts_dir)

    if args.mock:
        llm = MockWarehouseLLMClient(
            actor_model=settings.actor_model,
            reflector_model=settings.reflector_model,
            dialect=settings.dialect,
        )
        mode_desc = "OFFLINE MOCK"
    else:
        llm = DualRoleLLMClient(
            settings=settings,
            response_cache_path=(cache_dir / "reflector_responses.jsonl") if cache_dir else None,
        )
        mode_desc = f"LIVE ({settings.base_url}, concurrency={settings.max_concurrency})"

    console.print(
        Panel.fit(
            f"[bold cyan]EvoDW-RSI Evolution Runner[/bold cyan]\n"
            f"Mode: {mode_desc}\n"
            f"Method: {args.method} | Ablation: {args.run_ablation} | Budget: {args.budget} | Dialect: {settings.dialect.upper()}\n"
            f"Actor: {settings.actor_model} | Reflector: {settings.reflector_model}"
        )
    )

    methods_to_run = ["seed", "greedy", "gepa"] if args.run_ablation else [args.method]
    results: list[EvolutionRunResult] = []
    verified_all: list[AgentTrajectory] = []
    shared_eval_cache: dict[tuple[str, str], AgentTrajectory] = {}
    t_global = time.monotonic()

    try:
        for m in methods_to_run:
            rem_wall: float | None = None
            if args.wall_limit_sec is not None:
                rem_wall = args.wall_limit_sec - (time.monotonic() - t_global)
                if rem_wall <= 5.0:
                    raise WallClockCheckpoint(f"Wall-clock budget reached before starting '{m}'.")

            llm.ledger = UsageLedger()
            console.print(f"[bold]>>> Running method='{m}' (budget={args.budget})...[/bold]")
            engine = EvolutionEngine(
                llm=llm,
                benchmark=benchmark,
                seed_policy=seed_policy,
                dialect=settings.dialect,
                max_refine_turns=settings.max_refine_turns,
                eval_cache=shared_eval_cache,
                cache_dir=cache_dir,
                wall_limit_sec=rem_wall,
            )
            res = await engine.run(
                method=m,
                budget=args.budget,
                val_active_size=args.val_active_size,
            )
            results.append(res)
            verified_all.extend(res.verified_train_val_trajectories)
            ns_tag = f"{args.suite}_{settings.dialect}"
            res.best_policy.save_json(out_dir / "policies" / f"{m}_best_policy.json")
            res.best_policy.save_json(out_dir / "policies" / ns_tag / f"{m}_best_policy.json")
            # Immediately persist method summary (both default and suite+dialect namespaced).
            runs_dir = out_dir / "runs"
            ns_runs_dir = runs_dir / ns_tag
            runs_dir.mkdir(parents=True, exist_ok=True)
            ns_runs_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            summary_json_str = _json.dumps(res.to_summary_dict(), ensure_ascii=False, indent=2)
            (runs_dir / f"{m}_summary.json").write_text(summary_json_str, encoding="utf-8")
            (ns_runs_dir / f"{m}_summary.json").write_text(summary_json_str, encoding="utf-8")
            console.print(
                f"    [green]Done {m}[/green]: Val EX={res.best_val_metrics.ex*100:.1f}%, "
                f"Test EX={res.best_test_metrics.ex*100:.1f}%, "
                f"Test 1st-Pass GW={res.best_test_metrics.first_pass_gateway_rate*100:.1f}% "
                f"(rollouts={res.rollouts_used}, active_val={res.effective_val_size}, cache_hits={res.cache_hits})"
            )
    except WallClockCheckpoint as ckpt:
        console.print(
            f"[bold yellow]CHECKPOINT SAVED:[/bold yellow] {ckpt} "
            f"(total cached trajectories={len(shared_eval_cache)}). Re-run same command to resume seamlessly."
        )
        benchmark.sandbox.close()
        return 0
    except MissingAPIKeyError as exc:
        console.print(f"[bold yellow]Configuration required:[/bold yellow] {exc}")
        console.print("Tip: Copy `.env.example` to `.env` with `LLM_API_KEY`, or pass `--mock`.")
        benchmark.sandbox.close()
        return 2
    finally:
        if isinstance(llm, DualRoleLLMClient):
            await llm.close()

    # Pool any verified train/val trajectories present in the shared/disk cache across ablation methods.
    for cached_traj in shared_eval_cache.values():
        if cached_traj.ex_match and cached_traj.split in ("train", "val"):
            verified_all.append(cached_traj)

    primary_result = next((r for r in results if r.method == args.method), results[-1])

    evolution_comparison = {
        "primary_method": primary_result.method,
        "best_policy_id": primary_result.best_policy.policy_id,
        "val": {
            "seed_avg_steps": primary_result.seed_val_metrics.avg_steps,
            "evolved_avg_steps": primary_result.best_val_metrics.avg_steps,
            "seed_avg_tokens": primary_result.seed_val_metrics.avg_tokens,
            "evolved_avg_tokens": primary_result.best_val_metrics.avg_tokens,
            "seed_avg_latency_ms": primary_result.seed_val_metrics.avg_latency_ms,
            "evolved_avg_latency_ms": primary_result.best_val_metrics.avg_latency_ms,
        },
        "test": {
            "seed_avg_steps": primary_result.seed_test_metrics.avg_steps,
            "evolved_avg_steps": primary_result.best_test_metrics.avg_steps,
            "seed_avg_tokens": primary_result.seed_test_metrics.avg_tokens,
            "evolved_avg_tokens": primary_result.best_test_metrics.avg_tokens,
            "seed_avg_latency_ms": primary_result.seed_test_metrics.avg_latency_ms,
            "evolved_avg_latency_ms": primary_result.best_test_metrics.avg_latency_ms,
            "seed_first_pass_correct": primary_result.seed_test_metrics.first_pass_correct_count,
            "evolved_first_pass_correct": primary_result.best_test_metrics.first_pass_correct_count,
            "seed_self_repaired": primary_result.seed_test_metrics.self_repaired_count,
            "evolved_self_repaired": primary_result.best_test_metrics.self_repaired_count,
        },
    }

    rl_summary = export_sft_and_grpo_datasets(
        trajectories=verified_all,
        tasks=benchmark.all_tasks,
        output_dir=out_dir / "rl_export",
        merge_existing=(not args.mock),
        sandbox=benchmark.sandbox,
        dialect=settings.dialect,
        evolution_comparison=evolution_comparison,
    )

    ns_tag = f"{args.suite}_{settings.dialect}"
    ns_runs_dir = out_dir / "runs" / ns_tag
    report_path = out_dir / "experiment_report.md"
    generate_experiment_report(
        primary_result,
        output_path=report_path,
        runs_dir=ns_runs_dir if ns_runs_dir.exists() else (out_dir / "runs"),
        ablation_results=results,
        rl_export_summary=rl_summary,
    )
    generate_experiment_report(
        primary_result,
        output_path=out_dir / f"experiment_report_{ns_tag}.md",
        runs_dir=ns_runs_dir if ns_runs_dir.exists() else (out_dir / "runs"),
        ablation_results=results,
        rl_export_summary=rl_summary,
    )

    benchmark.sandbox.close()

    tbl = Table(title=f"Evolution Results Summary ({args.suite} | {settings.dialect.upper()} | Test={len(benchmark.test)} Tasks)")
    tbl.add_column("Method", style="cyan")
    tbl.add_column("Seed EX")
    tbl.add_column("Evolved EX", style="bold green")
    tbl.add_column("1st-Pass GW")
    tbl.add_column("Invalid Rate")
    tbl.add_column("Avg Tokens")
    tbl.add_column("Avg Latency")
    for res in results:
        tbl.add_row(
            res.method,
            f"{res.seed_test_metrics.ex*100:.1f}%",
            f"{res.best_test_metrics.ex*100:.1f}%",
            f"{res.best_test_metrics.first_pass_gateway_rate*100:.1f}%",
            f"{res.best_test_metrics.invalid_sql_rate*100:.1f}%",
            f"{res.best_test_metrics.avg_tokens:.0f}",
            f"{res.best_test_metrics.avg_latency_ms:.0f}ms",
        )
    console.print(tbl)
    console.print(f"[bold]Experiment Report:[/bold] {report_path}")
    console.print(f"[bold]RL/SFT Export:[/bold] {rl_summary}")
    return 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(description="Run EvoDW-RSI GEPA strategy evolution.")
    p.add_argument("--budget", type=int, default=80, help="Optimization rollout budget.")
    p.add_argument("--val-active-size", type=int, default=None, help="Explicit stratified active val size.")
    p.add_argument("--method", choices=["seed", "greedy", "gepa"], default="gepa")
    p.add_argument("--run-ablation", action="store_true", help="Run seed, greedy, and gepa sequentially.")
    p.add_argument("--suite", choices=["v1", "hard_v2", "all"], default="v1", help="Benchmark suite to run.")
    p.add_argument("--dialect", choices=["odps", "hive"], default=None)
    p.add_argument("--actor-model", default=None)
    p.add_argument("--reflector-model", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    p.add_argument("--cache-dir", default=None, help="Directory for persistent trajectory/reflector cache.")
    p.add_argument("--no-cache", action="store_true", help="Disable persistent disk cache.")
    p.add_argument("--wall-limit-sec", type=float, default=None, help="Graceful wall-clock checkpoint limit in seconds.")
    p.add_argument("--mock", action="store_true", help="Use deterministic offline mock LLM.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
