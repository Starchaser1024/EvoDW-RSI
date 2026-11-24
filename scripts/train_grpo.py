#!/usr/bin/env python3
"""CLI entrypoint for Stage-2 Multi-Turn Observation-Masked GRPO (`scripts/train_grpo.py`)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.rl.config import DEFAULT_BASE_MODEL, GRPOTrainingConfig, RLModelConfig
from src.rl.grpo_trainer import run_grpo_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoDW-RL v2 Stage 2: Multi-Turn Agentic Sandbox-Feedback & Observation-Masked GRPO"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Fallback base model preset ('1.5b', '3b', '7b') or path if no merged SFT checkpoint is given.",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_checkpoints" / "sft",
        help="Directory containing Stage-1 SFT output (`sft_merged` or `sft_adapter`).",
    )
    parser.add_argument(
        "--grpo-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_export" / "grpo_prompts.jsonl",
        help="Path to 45 train+val GRPO prompts JSONL (test split strictly excluded).",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "policies" / "gepa_best_policy.json",
        help="Policy prompt JSON used to format warehouse context.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_checkpoints" / "grpo",
        help="Directory to save Stage-2 GRPO LoRA adapter & telemetry.",
    )
    parser.add_argument(
        "--ablate-reward",
        type=str,
        choices=["full", "no_format", "no_gateway", "no_exec", "no_self_repair", "outcome_only"],
        default="full",
        help="Ablation mode for the 4-stage + self-repair progressive reward suite.",
    )
    parser.add_argument("--single-turn", action="store_true", help="Disable multi-turn sandbox rollout (ablation).")
    parser.add_argument("--max-rollout-turns", type=int, default=3, help="Max interactive sandbox turns per episode (2~3).")
    parser.add_argument("--max-turn-tokens", type=int, default=768, help="Max new tokens per single assistant turn.")
    parser.add_argument("--num-generations", type=int, default=8, help="Group size G per prompt (default G=8 on A100).")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--max-steps", type=int, default=100, help="Total GRPO optimization steps.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Epochs if max-steps <= 0.")
    parser.add_argument("--lr", type=float, default=2e-5, help="GRPO learning rate.")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty anchored to SFT reference (>0 prevents collapse).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Rollout sampling temperature.")
    parser.add_argument("--max-prompt-length", type=int, default=4096, help="Max prompt token length.")
    parser.add_argument("--max-completion-length", type=int, default=2048, help="Max multi-turn trajectory completion length.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--attn-impl", type=str, default="auto", help="Attention kernel: auto | flash_attention_2 | sdpa | eager.")
    parser.add_argument("--use-vllm", action="store_true", help="Enable colocate vLLM rollout engine if compatible.")
    parser.add_argument("--save-steps", type=int, default=0, help="Periodic checkpoint steps on Google Drive.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint or skip if completed.")
    parser.add_argument("--live-curve-jsonl", type=Path, default=None, help="Optional live reward/KL curve JSONL path.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA quantization (recommended on A100 40/80GB).")
    parser.add_argument("--dry-run", action="store_true", help="Run fast offline CPU smoke test.")
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
    grpo_cfg = GRPOTrainingConfig(
        grpo_jsonl_path=args.grpo_jsonl,
        sft_checkpoint_dir=args.sft_checkpoint if args.sft_checkpoint.exists() else None,
        policy_json_path=args.policy_json,
        output_dir=args.output_dir,
        ablation_mode=args.ablate_reward,  # type: ignore[arg-type]
        enable_multiturn_rollout=not args.single_turn,
        max_rollout_turns=args.max_rollout_turns,
        max_per_turn_new_tokens=args.max_turn_tokens,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
        live_curve_jsonl_path=args.live_curve_jsonl,
    )

    console.rule("[bold magenta]EvoDW-RL v2 Stage 2: Multi-Turn Observation-Masked Agentic GRPO")
    summary = run_grpo_training(model_cfg=model_cfg, grpo_cfg=grpo_cfg)

    tbl = Table(title="Stage-2 Multi-Turn GRPO Summary", show_header=True, header_style="bold magenta")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Initialized From", str(summary["base_model_used"]))
    tbl.add_row("Train+Val Prompts (Test Excluded)", str(summary["prompt_count"]))
    tbl.add_row("Multi-Turn Rollout / Max Turns", f"{summary.get('enable_multiturn_rollout', True)} / {summary.get('max_rollout_turns', 3)}")
    tbl.add_row("Group Size (G) / KL Beta", f"{summary.get('num_generations', 8)} / {summary.get('beta_kl', 0.04)}")
    tbl.add_row("Reward Ablation Mode", str(summary["ablation_mode"]))
    tbl.add_row("Active Reward Stages", ", ".join(summary["active_reward_funcs"]))
    tbl.add_row("Global Steps", str(summary["global_steps"]))
    tbl.add_row("Runtime (s)", f"{summary['train_runtime_sec']:.1f}s")
    tbl.add_row("GRPO Adapter Saved", str(summary["adapter_dir"]))
    console.print(tbl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
