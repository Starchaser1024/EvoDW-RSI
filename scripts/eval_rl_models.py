#!/usr/bin/env python3
"""Unified multi-model evaluation & Markdown report generator on the isolated Test split (`scripts/eval_rl_models.py`).

Supports evaluating:
1. Base 小模型 (Zero-shot)
2. Base + GEPA Evolved Prompt
3. 垂直 SQL 开源模型 Zero-shot (`XiYanSQL-QwenCoder-3B/7B-2504`, optional `--vertical-model`)
4. Stage-1 SFT 冷启动模型 (Loss-Masked Multi-Turn Trajectory SFT on 299+ verified trajectories)
5. Stage-2 SFT + GRPO 稀疏消融模型 (`Outcome-Only Sparse Reward`, optional `--grpo-sparse-dir`)
6. Stage-2 SFT + GRPO 主模型 (`Multi-Turn 4-Stage Dense + Self-Repair Reward`)
7. 在线大模型 `deepseek-v4.1-flash (GEPA)`
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gc
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
import torch

from src.agent.pipeline import MultiModuleSQLAgent
from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory
from src.benchmarks.loader import load_benchmark
from src.benchmarks.tasks import BenchmarkTask
from src.config import Settings
from src.env.sandbox import DialectType, WarehouseSandbox
from src.eval.metrics import compute_eval_metrics
from src.llm.client import DualRoleLLMClient
from src.rl.config import DEFAULT_BASE_MODEL, resolve_model_name_or_path, resolve_vertical_sql_model
from src.rl.eval_report import (
    CATEGORIES_ORDER,
    build_test_case_studies,
    render_rl_markdown_report,
    summarize_grpo_curve,
)
from src.rl.inference import (
    LocalHFLLMClient,
    StatefulTrajectorySQLAgent,
    VerticalSQLZeroShotAgent,
)
from src.rl.model_utils import load_model_for_inference


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _rel_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _release_gpu() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _load_cached_online_metrics(gepa_summary_path: Path, variant: str = "best_test") -> dict[str, Any] | None:
    if not gepa_summary_path.exists():
        return None
    data = json.loads(gepa_summary_path.read_text(encoding="utf-8"))
    metrics_dict = data.get(variant) or data.get("seed_test")
    if not isinstance(metrics_dict, dict):
        return None
    return metrics_dict


def _serialize_traj_digest(t: AgentTrajectory) -> dict[str, Any]:
    return {
        "task_id": t.task_id,
        "category": t.category,
        "question": t.question,
        "ex_match": t.ex_match,
        "first_pass_gateway": t.first_pass_gateway,
        "refine_turns": t.refine_turns,
        "final_sql": t.final_sql,
        "first_pass_error_msg": t.first_pass_exec.get("error_msg", ""),
        "first_pass_error_code": t.first_pass_exec.get("error_code", ""),
    }


async def evaluate_local_variant(
    *,
    variant_key: str,
    label: str,
    base_model_path: str,
    adapter_path: Path | None,
    mode: str,  # "modular" | "stateful_trajectory" | "vertical_zeroshot"
    policy: Policy,
    tasks: list[BenchmarkTask],
    sandbox: WarehouseSandbox,
    dialect: DialectType = "odps",
    use_4bit: bool = False,
    batch_size: int = 6,
    dry_run: bool = False,
    cache_dir: Path | None = None,
    resume_cache: bool = False,
    console: Console | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one local model variant, run deterministic greedy evaluation on Test tasks, and free VRAM."""
    cache_file = (cache_dir / f"{variant_key}_{len(tasks)}.json") if (cache_dir and not dry_run) else None
    if resume_cache and cache_file and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if console:
            console.print(f"[green]✔ Loaded cached evaluation for [bold]{label}[/bold] from {cache_file}[/green]")
        return cached["metrics"], cached["traj_digests"]

    if console:
        console.print(f"[cyan]▶ Evaluating [bold]{label}[/bold] ({len(tasks)} test tasks)...[/cyan]")

    partial_pkl = (cache_dir / f"{variant_key}_{len(tasks)}_partial.pkl") if (cache_dir and not dry_run) else None
    trajs: list[AgentTrajectory] = []
    if resume_cache and partial_pkl and partial_pkl.exists():
        trajs = pickle.loads(partial_pkl.read_bytes())
        if console and trajs:
            console.print(f"  [yellow]↻ Resuming {label} from {len(trajs)}/{len(tasks)} completed tasks...[/yellow]")

    model, tokenizer = load_model_for_inference(
        base_model_path,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
        dry_run=dry_run,
    )
    try:
        hf_client = LocalHFLLMClient(
            model=model,
            tokenizer=tokenizer,
            model_label=label,
            max_new_tokens_cap=8 if dry_run else 1024,
            max_prompt_tokens=96 if dry_run else None,
            max_batch_size=max(1, batch_size),
        )
        start_idx = len(trajs)
        chunk_size = max(1, batch_size)
        eff_refine_turns = 1 if dry_run else 2

        if mode == "modular":
            runner: Any = MultiModuleSQLAgent(llm=hf_client, sandbox=sandbox, max_refine_turns=eff_refine_turns)
        elif mode == "vertical_zeroshot":
            runner = VerticalSQLZeroShotAgent(hf_client=hf_client, sandbox=sandbox)
        else:
            runner = StatefulTrajectorySQLAgent(
                hf_client=hf_client,
                sandbox=sandbox,
                max_refine_turns=eff_refine_turns,
            )

        for idx in range(start_idx, len(tasks), chunk_size):
            batch_tasks = tasks[idx : idx + chunk_size]
            if mode == "modular":
                batch_res = await asyncio.gather(
                    *(runner.run_task(t, policy=policy, dialect=dialect, evaluate=True) for t in batch_tasks)
                )
            else:
                batch_res = await asyncio.gather(
                    *(runner.run_task(t, policy=policy, dialect=dialect) for t in batch_tasks)
                )
            trajs.extend(batch_res)
            if partial_pkl is not None:
                partial_pkl.parent.mkdir(parents=True, exist_ok=True)
                partial_pkl.write_bytes(pickle.dumps(trajs))
            _release_gpu()
            done = len(trajs)
            if console:
                ex_now = sum(1 for x in trajs if x.ex_match)
                gw_now = sum(1 for x in trajs if x.first_pass_gateway)
                console.print(f"  [{done}/{len(tasks)}] EX={ex_now}/{done} | 1st-Pass GW={gw_now}/{done}")

        metrics = compute_eval_metrics(trajs).to_dict()
        digests = [_serialize_traj_digest(t) for t in trajs]
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"label": label, "metrics": metrics, "traj_digests": digests}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if partial_pkl and partial_pkl.exists():
                partial_pkl.unlink()
        return metrics, digests
    finally:
        del model
        del tokenizer
        _release_gpu()


async def run_full_evaluation(
    *,
    model_spec: str = DEFAULT_BASE_MODEL,
    vertical_model_spec: str | None = None,
    sft_checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "rl_checkpoints" / "sft",
    grpo_checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "rl_checkpoints" / "grpo",
    grpo_sparse_checkpoint_dir: Path | None = None,
    gepa_policy_path: Path = PROJECT_ROOT / "outputs" / "policies" / "gepa_best_policy.json",
    gepa_summary_path: Path = PROJECT_ROOT / "outputs" / "runs" / "gepa_summary.json",
    output_report_path: Path = PROJECT_ROOT / "outputs" / "rl_experiment_report.md",
    output_json_path: Path = PROJECT_ROOT / "outputs" / "rl_eval" / "rl_eval_summary.json",
    use_4bit: bool = False,
    batch_size: int = 6,
    live_online: bool = False,
    limit_test_tasks: int | None = None,
    resume_cache: bool = False,
    variants_filter: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run all configured model variants on the isolated Test split and persist report + JSON."""
    console = Console()
    bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True)
    try:
        test_tasks = list(bm.test)
        assert all(t.split == "test" for t in test_tasks), "Non-test task detected in bm.test!"
        if limit_test_tasks is not None and limit_test_tasks > 0:
            test_tasks = test_tasks[:limit_test_tasks]

        seed_policy = Policy.load_seed(PROJECT_ROOT / "prompts" / "seed")
        evolved_policy = Policy.load_json(gepa_policy_path) if gepa_policy_path.exists() else seed_policy

        resolved_base = "dry-run-tiny-qwen2" if dry_run else resolve_model_name_or_path(model_spec)
        short_model_name = "Tiny-Qwen2-DryRun" if dry_run else Path(resolved_base).name

        sft_merged = sft_checkpoint_dir / "sft_merged"
        sft_adapter = sft_checkpoint_dir / "sft_adapter"
        dry_base = sft_checkpoint_dir / "dry_run_base"
        if dry_run and dry_base.exists():
            resolved_base = str(dry_base)

        if sft_merged.exists() and (sft_merged / "config.json").exists():
            sft_base_path = str(sft_merged)
            sft_lora_path: Path | None = None
        else:
            sft_base_path = resolved_base
            sft_lora_path = sft_adapter if sft_adapter.exists() else None

        grpo_adapter = grpo_checkpoint_dir / "grpo_adapter"
        cache_dir = output_json_path.parent / "cache"

        local_specs: list[tuple[str, str, str, str, Path | None, str, Policy]] = [
            (
                "base",
                "base_zeroshot",
                f"Base 小模型 ({short_model_name} Zero-shot)",
                resolved_base,
                None,
                "modular",
                seed_policy,
            ),
            (
                "base_gepa",
                "base_gepa",
                f"Base + GEPA Evolved Prompt ({short_model_name})",
                resolved_base,
                None,
                "modular",
                evolved_policy,
            ),
        ]

        vertical_label: str | None = None
        if vertical_model_spec:
            resolved_vert = resolved_base if dry_run else resolve_vertical_sql_model(vertical_model_spec)
            vert_short = "Tiny-XiYanSQL-DryRun" if dry_run else Path(resolved_vert).name
            vertical_label = f"垂直 SQL 模型 ({vert_short} Zero-shot)"
            local_specs.append(
                (
                    "vertical_sql",
                    "vertical_sql_zeroshot",
                    vertical_label,
                    resolved_vert,
                    None,
                    "vertical_zeroshot",
                    seed_policy,
                )
            )

        local_specs.append(
            (
                "sft",
                "sft_coldstart",
                "SFT 冷启动模型 (Loss-Masked Multi-Turn)",
                sft_base_path,
                sft_lora_path,
                "stateful_trajectory",
                evolved_policy,
            )
        )

        if grpo_sparse_checkpoint_dir is not None:
            sparse_adapter = grpo_sparse_checkpoint_dir / "grpo_adapter"
            local_specs.append(
                (
                    "sft_grpo_sparse",
                    "sft_grpo_outcome_only",
                    "Multi-Turn SFT + GRPO (Outcome-Only Sparse Reward)",
                    sft_base_path,
                    sparse_adapter if sparse_adapter.exists() else sft_lora_path,
                    "stateful_trajectory",
                    evolved_policy,
                )
            )

        dense_grpo_label = "Multi-Turn SFT + GRPO (4-Stage Dense + Self-Repair)"
        local_specs.append(
            (
                "sft_grpo",
                "sft_grpo",
                dense_grpo_label,
                sft_base_path,
                grpo_adapter if grpo_adapter.exists() else sft_lora_path,
                "stateful_trajectory",
                evolved_policy,
            )
        )

        results_ordered: list[tuple[str, dict[str, Any]]] = []
        metrics_by_short_key: dict[str, dict[str, Any]] = {}
        traj_store: dict[str, list[dict[str, Any]]] = {}

        for short_key, v_key, v_label, v_base, v_lora, v_mode, v_policy in local_specs:
            if variants_filter is not None and v_key not in variants_filter and short_key not in variants_filter:
                cfile = cache_dir / f"{v_key}_{len(test_tasks)}.json"
                if not dry_run and cfile.exists():
                    cached = json.loads(cfile.read_text(encoding="utf-8"))
                    results_ordered.append((v_label, cached["metrics"]))
                    metrics_by_short_key[short_key] = cached["metrics"]
                    traj_store[short_key] = cached["traj_digests"]
                continue
            m_var, t_var = await evaluate_local_variant(
                variant_key=v_key,
                label=v_label,
                base_model_path=v_base,
                adapter_path=v_lora,
                mode=v_mode,
                policy=v_policy,
                tasks=test_tasks,
                sandbox=bm.sandbox,
                use_4bit=use_4bit,
                batch_size=batch_size,
                dry_run=dry_run,
                cache_dir=cache_dir,
                resume_cache=resume_cache,
                console=console,
            )
            results_ordered.append((v_label, m_var))
            metrics_by_short_key[short_key] = m_var
            traj_store[short_key] = t_var

        # Online Large Model `deepseek-v4.1-flash`
        online_metrics: dict[str, Any] | None = None
        if live_online and not dry_run:
            settings = Settings.from_env()
            console.print(f"[cyan]▶ Evaluating live online model [bold]{settings.actor_model}[/bold]...[/cyan]")
            online_client = DualRoleLLMClient(settings=settings)
            try:
                online_agent = MultiModuleSQLAgent(llm=online_client, sandbox=bm.sandbox, max_refine_turns=2)
                online_trajs = await online_agent.evaluate_tasks(test_tasks, policy=evolved_policy, dialect="odps")
                online_metrics = compute_eval_metrics(online_trajs).to_dict()
            finally:
                await online_client.close()
        else:
            online_metrics = _load_cached_online_metrics(gepa_summary_path, variant="best_test")

        if online_metrics is not None:
            results_ordered.append(("在线大模型 deepseek-v4.1-flash (GEPA)", online_metrics))
            metrics_by_short_key["online"] = online_metrics

        vertical_comp: dict[str, Any] | None = None
        if vertical_label and "vertical_sql" in metrics_by_short_key and "sft_grpo" in metrics_by_short_key:
            err_codes: dict[str, int] = {}
            for d in traj_store.get("vertical_sql", []):
                code = str(d.get("first_pass_error_code") or "").strip()
                if code:
                    err_codes[code] = err_codes.get(code, 0) + 1
            vertical_comp = {
                "vertical_label": vertical_label,
                "vertical_metrics": metrics_by_short_key["vertical_sql"],
                "ours_label": dense_grpo_label,
                "ours_metrics": metrics_by_short_key["sft_grpo"],
                "gateway_intercept_breakdown": err_codes,
            }

        ablation_test_metrics: dict[str, dict[str, Any]] | None = None
        if "sft_grpo_sparse" in metrics_by_short_key and "sft_grpo" in metrics_by_short_key:
            ablation_test_metrics = {
                "outcome_only": metrics_by_short_key["sft_grpo_sparse"],
                "full": metrics_by_short_key["sft_grpo"],
            }

        case_studies = build_test_case_studies(traj_store)

        sft_sum_file = sft_checkpoint_dir / "sft_training_summary.json"
        grpo_sum_file = grpo_checkpoint_dir / "grpo_training_summary.json"
        sparse_sum_file = (
            (grpo_sparse_checkpoint_dir / "grpo_training_summary.json")
            if grpo_sparse_checkpoint_dir is not None
            else None
        )
        sft_sum = json.loads(sft_sum_file.read_text(encoding="utf-8")) if sft_sum_file.exists() else None
        grpo_sum = json.loads(grpo_sum_file.read_text(encoding="utf-8")) if grpo_sum_file.exists() else None
        sparse_sum = (
            json.loads(sparse_sum_file.read_text(encoding="utf-8"))
            if (sparse_sum_file is not None and sparse_sum_file.exists())
            else None
        )

        report_md = render_rl_markdown_report(
            model_results=results_ordered,
            base_model_spec=short_model_name,
            test_task_count=len(test_tasks),
            sft_summary=sft_sum,
            grpo_summary=grpo_sum,
            grpo_sparse_summary=sparse_sum,
            vertical_comparison=vertical_comp,
            ablation_test_metrics=ablation_test_metrics,
            case_studies=case_studies,
        )
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(report_md, encoding="utf-8")

        summary_payload = {
            "version": "v2",
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "base_model": resolved_base,
            "base_model_short": short_model_name,
            "test_task_count": len(test_tasks),
            "models": {name: m for name, m in results_ordered},
            "variants_by_key": metrics_by_short_key,
            "vertical_comparison": vertical_comp,
            "ablation_comparison": {
                "dense_curve": summarize_grpo_curve(grpo_sum),
                "sparse_curve": summarize_grpo_curve(sparse_sum),
                "test_metrics": ablation_test_metrics,
            },
            "sft_training_summary": sft_sum,
            "grpo_training_summary": grpo_sum,
            "grpo_sparse_training_summary": sparse_sum,
            "case_studies": case_studies[:5],
            "report_path": _rel_or_posix(output_report_path),
        }
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        rich_tbl = Table(
            title=f"EvoDW-RL v2 Isolated Test Set ({len(test_tasks)} Tasks) Comparison",
            header_style="bold cyan",
        )
        rich_tbl.add_column("Model Variant")
        rich_tbl.add_column("EX", justify="right")
        rich_tbl.add_column("1st-Pass GW", justify="right")
        rich_tbl.add_column("Invalid SQL", justify="right")
        rich_tbl.add_column("Avg Latency (ms)", justify="right")
        for name, m in results_ordered:
            rich_tbl.add_row(
                name,
                _pct(m["ex"]),
                _pct(m["first_pass_gateway_rate"]),
                _pct(m["invalid_sql_rate"]),
                f"{m.get('avg_latency_ms', 0.0):.1f}",
            )
        console.print(rich_tbl)
        console.print(f"[bold green]✔ Saved Markdown report -> {output_report_path}[/bold green]")
        console.print(f"[bold green]✔ Saved JSON telemetry  -> {output_json_path}[/bold green]")
        return summary_payload
    finally:
        bm.sandbox.close()


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(
        description="Evaluate Base / GEPA / Vertical SQL / SFT / Sparse GRPO / Multi-Turn GRPO / Online models on Test set."
    )
    p.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="Base small model ('1.5b', '3b', '7b', or path).")
    p.add_argument(
        "--vertical-model",
        type=str,
        default=None,
        help="Optional vertical SQL baseline ('xiyansql-3b', 'xiyansql-7b', or HF path).",
    )
    p.add_argument("--sft-dir", type=Path, default=PROJECT_ROOT / "outputs" / "rl_checkpoints" / "sft")
    p.add_argument("--grpo-dir", type=Path, default=PROJECT_ROOT / "outputs" / "rl_checkpoints" / "grpo")
    p.add_argument(
        "--grpo-sparse-dir",
        type=Path,
        default=None,
        help="Optional Outcome-Only Sparse GRPO checkpoint directory for reward ablation comparison.",
    )
    p.add_argument("--report-out", type=Path, default=PROJECT_ROOT / "outputs" / "rl_experiment_report.md")
    p.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "outputs" / "rl_eval" / "rl_eval_summary.json")
    p.add_argument("--use-4bit", action="store_true", help="Load local inference models in 4-bit NF4.")
    p.add_argument("--batch-size", type=int, default=6, help="Concurrent evaluation batch size on GPU.")
    p.add_argument("--live-online", action="store_true", help="Re-run live online API for deepseek-v4.1-flash.")
    p.add_argument("--limit", type=int, default=None, help="Optional subset limit for quick smoke checks.")
    p.add_argument("--variants", type=str, default=None, help="Comma-separated variant keys to run (default: all).")
    p.add_argument("--resume", action="store_true", help="Reuse per-variant cached results if already evaluated.")
    p.add_argument("--dry-run", action="store_true", help="Run fast offline smoke evaluation with tiny Qwen2.")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    v_filter = {v.strip() for v in args.variants.split(",") if v.strip()} if args.variants else None
    asyncio.run(
        run_full_evaluation(
            model_spec=args.model,
            vertical_model_spec=args.vertical_model,
            sft_checkpoint_dir=args.sft_dir,
            grpo_checkpoint_dir=args.grpo_dir,
            grpo_sparse_checkpoint_dir=args.grpo_sparse_dir,
            output_report_path=args.report_out,
            output_json_path=args.json_out,
            use_4bit=args.use_4bit,
            batch_size=args.batch_size,
            live_online=args.live_online,
            limit_test_tasks=args.limit,
            resume_cache=args.resume,
            variants_filter=v_filter,
            dry_run=args.dry_run,
        )
    )
    return 0


__all__ = [
    "CATEGORIES_ORDER",
    "build_parser",
    "evaluate_local_variant",
    "main",
    "parse_args",
    "render_rl_markdown_report",
    "run_full_evaluation",
    "summarize_grpo_curve",
]


if __name__ == "__main__":
    raise SystemExit(main())
