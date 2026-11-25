"""Unified CLI entrypoint for EvoDW-RSI (`evodw` / `python -m src`)."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import __version__
from src.config import PROJECT_ROOT, Settings


def _pkg_version(dist_name: str, import_name: str | None = None) -> tuple[bool, str]:
    mod_name = import_name or dist_name.replace("-", "_")
    if importlib.util.find_spec(mod_name) is None:
        return False, "missing"
    try:
        return True, importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return True, "installed"


def run_doctor(args: argparse.Namespace) -> int:
    """Validate runtime dependencies, LLM gateway configuration, and 120-task warehouse sandbox health."""
    console = Console()
    env_file = PROJECT_ROOT / ".env"
    example_file = PROJECT_ROOT / ".env.example"

    if getattr(args, "init_env", False) and not env_file.exists() and example_file.exists():
        shutil.copy2(example_file, env_file)
        try:
            os.chmod(env_file, 0o600)
        except OSError:
            pass
        console.print(f"[green]Created {env_file} from .env.example (edit LLM_API_KEY for live runs).[/green]")

    settings = Settings.from_env()

    tbl = Table(title=f"EvoDW-RSI Environment & Configuration Diagnostics (v{__version__})", header_style="bold cyan")
    tbl.add_column("Check")
    tbl.add_column("Status")
    tbl.add_column("Details")

    py_ok = sys.version_info >= (3, 11)
    tbl.add_row(
        "Python Runtime",
        "[green]OK[/green]" if py_ok else "[red]FAIL[/red]",
        f"{platform.python_version()} ({platform.system()} {platform.machine()})",
    )

    core_failures = 0
    for dist, mod in (
        ("duckdb", "duckdb"),
        ("sqlglot", "sqlglot"),
        ("openai", "openai"),
        ("pydantic", "pydantic"),
        ("python-dotenv", "dotenv"),
        ("rich", "rich"),
    ):
        ok, ver = _pkg_version(dist, mod)
        if not ok:
            core_failures += 1
        tbl.add_row(f"Core Package `{dist}`", "[green]OK[/green]" if ok else "[red]MISSING[/red]", ver)

    for dist, mod in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("trl", "trl"),
    ):
        ok, ver = _pkg_version(dist, mod)
        tbl.add_row(
            f"RL Package `{dist}` (optional)",
            "[green]OK[/green]" if ok else "[yellow]OPTIONAL[/yellow]",
            ver,
        )

    key_status = (
        "[green]CONFIGURED[/green] (redacted)"
        if settings.has_usable_api_key
        else "[yellow]MOCK ONLY[/yellow] (set LLM_API_KEY in .env for live API)"
    )
    tbl.add_row("LLM Gateway Endpoint", "[green]OK[/green]", settings.base_url)
    tbl.add_row("Live API Key", key_status, f".env present={env_file.exists()}")
    tbl.add_row(
        "Dual-Role Models",
        "[green]OK[/green]",
        f"Actor={settings.actor_model} | Reflector={settings.reflector_model} | Dialect={settings.dialect.upper()}",
    )

    sandbox_ok = False
    try:
        from src.agent import Policy
        from src.benchmarks.catalog_builder import build_warehouse_catalog
        from src.benchmarks.tasks import get_all_tasks

        cat = build_warehouse_catalog()
        v1_tasks = get_all_tasks("v1")
        v2_tasks = get_all_tasks("hard_v2")
        Policy.load_seed(settings.prompts_dir)
        sandbox_ok = len(cat.list_tables()) == 15 and len(v1_tasks) == 75 and len(v2_tasks) == 45
        sb_detail = f"15 tables | {len(v1_tasks)} v1 tasks + {len(v2_tasks)} Hard-v2 tasks | seed prompts verified"
    except Exception as exc:
        sb_detail = f"Failed: {exc}"

    tbl.add_row(
        "Warehouse Catalog & Benchmark",
        "[green]OK[/green]" if sandbox_ok else "[red]FAIL[/red]",
        sb_detail,
    )

    console.print(tbl)
    if not py_ok or core_failures > 0 or not sandbox_ok:
        return 1

    console.print(
        Panel.fit(
            "[bold green]Ready![/bold green] Run `evodw smoke-test --mock` or `pytest -q` for offline verification."
        )
    )
    return 0


SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "doctor": ("src.cli", "Validate environment, dependencies, LLM endpoint config, and sandbox integrity."),
    "config-check": ("src.cli", "Alias for `doctor`: check `.env` & dual-role OpenAI-compatible configuration."),
    "audit-experiments": (
        "src.eval.manifest_audit",
        "Audit frozen experiment lineage manifest, split isolation, and SFT-vs-GRPO CIs.",
    ),
    "smoke-test": ("scripts.smoke_test", "Run 3-task end-to-end warehouse trajectory smoke test (`--mock` or live)."),
    "evolve": ("scripts.run_evolution", "Run GEPA / Greedy / Seed strategy self-evolution & RL trajectory export."),
    "train-sft": ("scripts.train_sft", "Run Stage-1 Loss-Masked Multi-Turn Trajectory SFT post-training."),
    "train-grpo": ("scripts.train_grpo", "Run Stage-2 Multi-Turn Observation-Masked Agentic GRPO post-training."),
    "eval-rl": ("scripts.eval_rl_models", "Evaluate Base, GEPA, Vertical SQL, SFT, and Multi-Turn GRPO on isolated Test split."),
    "pack-colab": ("scripts.pack_for_colab", "Package secret-scrubbed archive & Colab A100/H100 notebook."),
    "merge-colab": ("scripts.merge_colab_results", "Merge Colab A100 evaluation archives into unified RL Markdown/JSON reports."),
    "convert-bird": ("scripts.convert_bird_to_odps", "Transpile BIRD Mini-Dev SQLite tasks into ds-partitioned ODPS SQL."),
}


def build_root_parser() -> argparse.ArgumentParser:
    epilog_lines = ["Available subcommands:"]
    for cmd, (_, desc) in SUBCOMMANDS.items():
        epilog_lines.append(f"  {cmd:<18} {desc}")
    epilog_lines.append("\nRun `evodw <subcommand> --help` for subcommand-specific options.")

    parser = argparse.ArgumentParser(
        prog="evodw",
        description=(
            "EvoDW-RSI: Reflective Strategy Self-Evolution & Multi-Turn Agentic RL Framework "
            "for Enterprise Data Warehouse (ODPS / Hive) Text-to-SQL."
        ),
        epilog="\n".join(epilog_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"evodw-rsi {__version__}")
    parser.add_argument("subcommand", nargs="?", choices=list(SUBCOMMANDS.keys()), help="Command to execute.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    root_parser = build_root_parser()

    if not raw_args or raw_args[0] in ("-h", "--help"):
        root_parser.print_help()
        return 0
    if raw_args[0] == "--version":
        print(f"evodw-rsi {__version__}")
        return 0

    subcmd = raw_args[0]
    sub_argv = raw_args[1:]
    if subcmd not in SUBCOMMANDS:
        root_parser.error(f"Unknown subcommand {subcmd!r}. Choose from: {', '.join(SUBCOMMANDS)}.")

    if subcmd in ("doctor", "config-check"):
        doc_parser = argparse.ArgumentParser(
            prog=f"evodw {subcmd}",
            description=SUBCOMMANDS[subcmd][1],
        )
        doc_parser.add_argument(
            "--init-env",
            action="store_true",
            help="Create `.env` from `.env.example` (chmod 600) if missing.",
        )
        doc_args = doc_parser.parse_args(sub_argv)
        return run_doctor(doc_args)

    module_name, _ = SUBCOMMANDS[subcmd]
    mod = importlib.import_module(module_name)
    res = mod.main(sub_argv)
    return int(res) if isinstance(res, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
