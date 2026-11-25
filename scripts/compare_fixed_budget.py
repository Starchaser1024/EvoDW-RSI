"""Fixed-budget 4-method comparison runner (`evodw compare-methods` / `python scripts/compare_fixed_budget.py`).

Evaluates under an identical per-task LLM call cap:
  1. `direct_sql`          — 1-shot single-prompt SQL generation (no probing, no execution retry)
  2. `exec_feedback`       — Single generator + sandbox error retry (no Linker probing or sanity audit)
  3. `evodw_seed`          — 3-module pipeline with zero-prior seed prompts
  4. `evodw_gepa_transfer` — Zero-shot cross-schema transfer of frozen e-commerce-evolved GEPA policy
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.benchmarks import load_benchmark
from src.benchmarks.external_cross_schema import EXTERNAL_ADAPTATION_DISCLAIMER
from src.config import Settings
from src.eval.fixed_budget import (
    ALL_BASELINE_METHODS,
    BaselineMethod,
    compare_methods_fixed_budget,
)
from src.llm import DualRoleLLMClient, MissingAPIKeyError, MockWarehouseLLMClient


async def run_comparison(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.from_env(actor_model=args.actor_model, dialect=args.dialect)
    bm = load_benchmark(
        seed=2026,
        dialect=settings.dialect,
        precompute_gold=True,
        suite=args.suite,
    )

    if args.split == "all":
        tasks = list(bm.all_tasks)
    elif args.split == "test":
        tasks = list(bm.test)
    elif args.split == "val":
        tasks = list(bm.val)
    else:
        tasks = list(bm.train)

    if args.limit is not None and args.limit > 0:
        tasks = tasks[: args.limit]

    use_mock = bool(args.mock) or (not getattr(args, "live", False) and not settings.has_usable_api_key)
    if use_mock:
        llm = MockWarehouseLLMClient(
            actor_model=settings.actor_model,
            reflector_model=settings.reflector_model,
            dialect=settings.dialect,
        )
        mode_str = "OFFLINE DETERMINISTIC RULE SIMULATION (--mock)"
    else:
        llm = DualRoleLLMClient(settings=settings)
        mode_str = f"LIVE API ({settings.base_url} | model={settings.actor_model})"

    methods: tuple[BaselineMethod, ...] = tuple(args.methods)  # type: ignore[assignment]

    try:
        summary, _ = await compare_methods_fixed_budget(
            tasks=tasks,
            sandbox=bm.sandbox,
            llm=llm,
            dialect=settings.dialect,
            methods=methods,
            max_llm_calls_per_task=args.max_calls,
        )
    except MissingAPIKeyError as exc:
        console.print(f"[bold yellow]Configuration required:[/bold yellow] {exc}")
        return 2
    finally:
        if isinstance(llm, DualRoleLLMClient):
            await llm.close()
        bm.sandbox.close()

    summary["suite"] = args.suite
    summary["split"] = args.split
    summary["execution_mode"] = "mock" if use_mock else "live"
    if args.suite == "external_bird_mini":
        summary["adaptation_disclaimer"] = EXTERNAL_ADAPTATION_DISCLAIMER

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    leak = summary["zero_cross_schema_leakage_audit"]
    leak_txt = "PASS (0 external/e-commerce table or enum leaks)" if leak["passed"] else f"FAIL ({leak})"
    disclaimer_line = (
        f"\n[yellow]Disclaimer:[/yellow] {EXTERNAL_ADAPTATION_DISCLAIMER}"
        if args.suite == "external_bird_mini"
        else ""
    )
    console.print(
        Panel(
            f"[bold cyan]EvoDW-RSI Fixed-Budget 4-Method Comparison[/bold cyan]\n"
            f"Suite: `{args.suite}` (`{args.split}`, N={len(tasks)}) | Dialect: `{settings.dialect.upper()}` | "
            f"Call Cap: `<={args.max_calls}` LLM calls/task\n"
            f"Mode: `{mode_str}`\n"
            f"Zero-Leakage Policy Audit (`{summary['transferred_policy_id']}`): [green]{leak_txt}[/green]"
            f"{disclaimer_line}"
        )
    )

    tbl = Table(title="Unified Fixed-Budget Telemetry (Executable vs Caliber-Correct)", header_style="bold cyan")
    tbl.add_column("Method")
    tbl.add_column("EX (Hits/N)", justify="right")
    tbl.add_column("95% Wilson CI", justify="right")
    tbl.add_column("Executable SQL", justify="right")
    tbl.add_column("1st-Pass GW", justify="right")
    tbl.add_column("Silent Caliber Err", justify="right")
    tbl.add_column("Avg Calls", justify="right")
    tbl.add_column("Avg Tokens", justify="right")
    tbl.add_column("Failure Taxon (GW/RT/Emp/Sil)", justify="right")

    for m_name in methods:
        m = summary["methods"][m_name]
        lo, hi = m["ex_wilson_95_ci"]
        ft = m["failure_taxonomy"]
        tbl.add_row(
            m_name,
            f"{m['ex'] * 100:.1f}% ({m['ex_hits']}/{m['total']})",
            f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
            f"{m['executable_sql_rate'] * 100:.1f}%",
            f"{m['first_pass_gateway_rate'] * 100:.1f}%",
            f"{m['silent_caliber_error_rate'] * 100:.1f}%",
            f"{m['avg_llm_calls']:.2f}/{m['max_llm_calls_budget']}",
            f"{m['avg_tokens']:.0f}",
            f"{ft['gateway_ast_block']}/{ft['runtime_error']}/{ft['empty_result']}/{ft['silent_caliber_error']}",
        )
    console.print(tbl)

    if args.suite == "external_bird_mini" and methods:
        dom_tbl = Table(title="Cross-Schema Domain Breakdown (Banking / Clinical / Schools)", header_style="bold magenta")
        dom_tbl.add_column("Domain")
        for m_name in methods:
            dom_tbl.add_column(f"{m_name} EX / SilentErr", justify="right")
        domains = sorted(summary["methods"][methods[0]]["by_domain"].keys())
        for dom in domains:
            cells = []
            for m_name in methods:
                dm = summary["methods"][m_name]["by_domain"][dom]
                cells.append(
                    f"{dm['ex'] * 100:.1f}% / {dm['silent_caliber_error_rate'] * 100:.1f}%"
                )
            dom_tbl.add_row(dom, *cells)
        console.print(dom_tbl)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evodw compare-methods",
        description="Compare direct_sql, exec_feedback, evodw_seed, and evodw_gepa_transfer under fixed call budget.",
    )
    p.add_argument(
        "--suite",
        choices=["external_bird_mini", "v1", "hard_v2", "all"],
        default="external_bird_mini",
        help="Target benchmark suite (default: external_bird_mini).",
    )
    p.add_argument(
        "--split",
        choices=["all", "test", "val", "train"],
        default="all",
        help="Task split to evaluate (for external_bird_mini, all 24 tasks are unseen during e-commerce evolution).",
    )
    p.add_argument("--dialect", choices=["odps", "hive"], default=None)
    p.add_argument("--actor-model", default=None)
    p.add_argument(
        "--max-calls",
        type=int,
        default=4,
        help="Maximum LLM calls permitted per task across every method (default: 4).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=list(ALL_BASELINE_METHODS),
        default=list(ALL_BASELINE_METHODS),
        help="Subset of baseline methods to run.",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional max tasks to evaluate.")
    p.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary output path.")
    p.add_argument("--mock", action="store_true", help="Force deterministic offline rule simulation.")
    p.add_argument("--live", action="store_true", help="Require live OpenAI-compatible API endpoint.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return asyncio.run(run_comparison(args))


if __name__ == "__main__":
    raise SystemExit(main())
