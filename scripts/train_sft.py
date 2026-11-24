#!/usr/bin/env python3
"""CLI entrypoint for Stage-1 Loss-Masked Multi-Turn Trajectory SFT (`scripts/train_sft.py`)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.rl.config import DEFAULT_BASE_MODEL, RLModelConfig, SFTTrainingConfig
from src.rl.sft_trainer import run_sft_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoDW-RL v2 Stage 1: Loss-Masked Multi-Turn Trajectory SFT (299+ Verified Trajectories)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base small model preset ('1.5b', '3b', '7b') or HuggingFace / local path.",
    )
    parser.add_argument(
        "--sft-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_export" / "sft_openai_tool_use.jsonl",
        help="Path to verified multi-turn SFT trajectory JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_checkpoints" / "sft",
        help="Directory to save SFT LoRA adapter and merged checkpoint.",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of SFT training epochs.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional max training steps override.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=9216,
        help="Max multi-turn sequence length (9216 eliminates all truncation across 299 trajectories).",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument(
        "--upsample-repaired",
        type=int,
        default=2,
        help="Base upsample factor for scarce self-repaired trajectories.",
    )
    parser.add_argument(
        "--upsample-probe",
        type=int,
        default=2,
        help="Base upsample factor for scarce inspect_table_sample probe trajectories.",
    )
    parser.add_argument(
        "--no-adaptive-reweight",
        action="store_true",
        help="Disable adaptive scarcity target-ratio reweighting.",
    )
    parser.add_argument("--attn-impl", type=str, default="auto", help="Attention kernel: auto | flash_attention_2 | sdpa | eager.")
    parser.add_argument("--save-steps", type=int, default=0, help="Periodic checkpoint steps (e.g. 25 on Google Drive).")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint or skip if completed.")
    parser.add_argument("--live-curve-jsonl", type=Path, default=None, help="Optional live loss curve JSONL path.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA quantization (use bf16/fp16 LoRA).")
    parser.add_argument("--no-merge", action="store_true", help="Skip saving merged SFT base model.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast offline CPU smoke test with tiny Qwen2.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    model_cfg = RLModelConfig(
        model_name_or_path=args.model,
        use_4bit=not args.no_4bit,
        attn_implementation=args.attn_impl,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    sft_cfg = SFTTrainingConfig(
        sft_jsonl_path=args.sft_jsonl,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
        live_curve_jsonl_path=args.live_curve_jsonl,
        save_merged_model=not args.no_merge,
        adaptive_reweight=not args.no_adaptive_reweight,
        upsample_self_repaired=args.upsample_repaired,
        upsample_probe=args.upsample_probe,
    )

    console.rule("[bold cyan]EvoDW-RL v2 Stage 1: Loss-Masked Multi-Turn Trajectory SFT")
    summary = run_sft_training(model_cfg=model_cfg, sft_cfg=sft_cfg)

    stats = summary["corpus_stats"]
    tbl = Table(title="Stage-1 SFT Summary", show_header=True, header_style="bold green")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Model", str(model_cfg.resolved_model_path()))
    tbl.add_row(
        "Raw Trajectories (First-Pass / Repaired)",
        f"{stats['total_records']} ({stats['first_pass_correct_count']} / {stats['self_repaired_count']})",
    )
    tbl.add_row("Inspect-Table-Sample Trajectories", str(stats["probe_trajectory_count"]))
    tbl.add_row(
        "Effective Episodes (Repaired / Probe)",
        f"{stats['effective_train_episodes']} ({stats.get('effective_self_repaired_episodes', 0)} / {stats.get('effective_probe_episodes', 0)})",
    )
    tbl.add_row(
        "Truncated Trajectories / Max Seq Len",
        f"{stats.get('truncated_trajectory_count', 0)} / {stats.get('max_observed_seq_len', 0)}",
    )
    tbl.add_row(
        "Supervised Assistant Token Ratio",
        f"{stats['supervised_token_ratio'] * 100:.1f}% ({stats['supervised_assistant_tokens']} / {stats['total_tokens']})",
    )
    tbl.add_row("Global Steps", str(summary["global_steps"]))
    tbl.add_row("Final Train Loss", f"{summary['train_loss']:.4f}")
    tbl.add_row("Runtime (s)", f"{summary['train_runtime_sec']:.1f}s")
    tbl.add_row("Adapter Saved", str(summary["adapter_dir"]))
    if summary.get("merged_dir"):
        tbl.add_row("Merged SFT Saved", str(summary["merged_dir"]))
    console.print(tbl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
