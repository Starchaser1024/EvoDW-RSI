"""Stage 2: Multi-Turn Agentic Observation-Masked GRPO Runner & Entrypoint."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch
from trl import GRPOConfig, GRPOTrainer

from src.agent.policy import Policy
from src.benchmarks.dw_generator import create_populated_sandbox
from src.config import PROJECT_ROOT, to_portable_posix_path
from src.env.sandbox import WarehouseSandbox
from src.rl.config import GRPOTrainingConfig, RLModelConfig, RewardStageWeights
from src.rl.grpo_data import (
    build_grpo_chatml_prompt,
    build_grpo_hf_dataset,
    load_grpo_prompts,
    select_candidate_tables_leak_free,
)
from src.rl.grpo_engine import MultiTurnAgenticGRPOTrainer
from src.rl.model_utils import (
    build_dry_run_model_and_tokenizer,
    build_lora_config,
    is_vllm_available,
    load_base_model_for_training,
)
from src.rl.multiturn_rollout import MultiTurnSandboxRolloutEngine
from src.rl.rewards import HierarchicalSQLReward
from src.rl.sft_trainer import LiveJSONLTelemetryCallback, _find_latest_trainer_checkpoint


def _load_policy_safe(policy_json_path: Path) -> Policy:
    if policy_json_path.exists():
        return Policy.load_json(policy_json_path)
    return Policy.load_seed(PROJECT_ROOT / "prompts" / "seed")


def _filter_supported_kwargs(cls_or_fn: Any, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls_or_fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return raw_kwargs
    return {k: v for k, v in raw_kwargs.items() if k in sig.parameters}


def run_grpo_training(
    model_cfg: RLModelConfig,
    grpo_cfg: GRPOTrainingConfig,
    *,
    sandbox: WarehouseSandbox | None = None,
) -> dict[str, Any]:
    """Execute Stage-2 Multi-Turn Observation-Masked GRPO with `MultiTurnAgenticGRPOTrainer`."""
    t0 = time.perf_counter()
    output_dir = Path(grpo_cfg.output_dir)
    adapter_dir = output_dir / "grpo_adapter"
    summary_path = output_dir / "grpo_training_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disconnect-proof fast-skip if this GRPO stage already finished cleanly
    if (
        grpo_cfg.resume_from_checkpoint
        and summary_path.exists()
        and (adapter_dir / "adapter_config.json").exists()
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    owns_sandbox = sandbox is None
    sb = sandbox or create_populated_sandbox(seed=model_cfg.seed, dialect=grpo_cfg.dialect)

    try:
        items = load_grpo_prompts(grpo_cfg.grpo_jsonl_path, strict_no_test_leakage=True)
        if model_cfg.dry_run:
            items = items[:2]

        policy = _load_policy_safe(Path(grpo_cfg.policy_json_path))
        hf_dataset = build_grpo_hf_dataset(
            items,
            sandbox=sb,
            policy=policy,
            dialect=grpo_cfg.dialect,
        )

        weights = (
            RewardStageWeights.from_ablation(grpo_cfg.ablation_mode)
            if grpo_cfg.ablation_mode != "full"
            else grpo_cfg.reward_weights
        )
        reward_engine = HierarchicalSQLReward(
            sandbox=sb,
            dialect=grpo_cfg.dialect,
            weights=weights,
        )
        reward_funcs = reward_engine.build_trl_reward_funcs()

        # Prefer Stage-1 merged SFT checkpoint so reference policy pi_ref == pi_SFT!
        base_init_path: Path | None = None
        if grpo_cfg.sft_checkpoint_dir is not None:
            cand = Path(grpo_cfg.sft_checkpoint_dir)
            if (cand / "sft_merged" / "config.json").exists():
                base_init_path = cand / "sft_merged"
            elif (cand / "config.json").exists():
                base_init_path = cand

        if model_cfg.dry_run and base_init_path is None:
            dry_base = output_dir / "dry_run_base"
            build_dry_run_model_and_tokenizer(seed=model_cfg.seed, save_dir=dry_base)
            base_init_path = dry_base

        model, tokenizer = load_base_model_for_training(
            model_cfg,
            base_override_path=base_init_path,
            attach_lora=False,
        )
        if model_cfg.gradient_checkpointing and not model_cfg.dry_run and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        tokenizer.padding_side = "left"
        lora_cfg = build_lora_config(model_cfg)

        use_cuda = torch.cuda.is_available() and not model_cfg.dry_run
        use_bf16 = (
            use_cuda
            and torch.cuda.is_bf16_supported()
            and model_cfg.compute_dtype.lower() in ("bfloat16", "bf16", "auto")
        )
        use_fp16 = (
            use_cuda
            and not use_bf16
            and model_cfg.compute_dtype.lower() in ("float16", "fp16", "auto")
        )

        num_gen = 2 if model_cfg.dry_run else max(2, grpo_cfg.num_generations)
        per_dev_bs = 2 if model_cfg.dry_run else grpo_cfg.per_device_train_batch_size
        grad_accum = 1 if model_cfg.dry_run else grpo_cfg.gradient_accumulation_steps

        if (per_dev_bs * grad_accum) % num_gen != 0:
            grad_accum = max(1, (num_gen + per_dev_bs - 1) // per_dev_bs)
            if (per_dev_bs * grad_accum) % num_gen != 0:
                grad_accum = num_gen

        max_prompt_len = min(96, grpo_cfg.max_prompt_length) if model_cfg.dry_run else grpo_cfg.max_prompt_length
        max_comp_len = 48 if model_cfg.dry_run else grpo_cfg.max_completion_length
        max_turn_new = 8 if model_cfg.dry_run else grpo_cfg.max_per_turn_new_tokens
        max_turns = 2 if model_cfg.dry_run else grpo_cfg.max_rollout_turns

        active_vllm = bool(grpo_cfg.use_vllm and use_cuda and is_vllm_available() and not model_cfg.use_4bit)
        runs_dir = output_dir / "trainer_runs"
        save_strategy = "steps" if (grpo_cfg.save_steps > 0 and not model_cfg.dry_run) else "no"

        raw_grpo_kwargs: dict[str, Any] = {
            "output_dir": str(runs_dir),
            "learning_rate": grpo_cfg.learning_rate,
            "per_device_train_batch_size": per_dev_bs,
            "gradient_accumulation_steps": grad_accum,
            "num_generations": num_gen,
            "max_prompt_length": max_prompt_len,
            "max_completion_length": max_comp_len,
            "num_train_epochs": grpo_cfg.num_train_epochs,
            "max_steps": 2 if model_cfg.dry_run else grpo_cfg.max_steps,
            "beta": grpo_cfg.beta,
            "temperature": grpo_cfg.temperature,
            "logging_steps": 1 if model_cfg.dry_run else grpo_cfg.logging_steps,
            "save_strategy": save_strategy,
            "save_steps": max(1, grpo_cfg.save_steps) if save_strategy == "steps" else 500,
            "save_total_limit": grpo_cfg.save_total_limit,
            "report_to": "none",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "use_cpu": model_cfg.dry_run or (not torch.cuda.is_available()),
            "gradient_checkpointing": model_cfg.gradient_checkpointing and not model_cfg.dry_run,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "use_vllm": active_vllm,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": grpo_cfg.vllm_gpu_memory_utilization,
            "seed": model_cfg.seed,
        }
        grpo_args = GRPOConfig(**_filter_supported_kwargs(GRPOConfig.__init__, raw_grpo_kwargs))

        rollout_engine = (
            MultiTurnSandboxRolloutEngine(
                sandbox=sb,
                tokenizer=tokenizer,
                dialect=grpo_cfg.dialect,
                max_rollout_turns=max_turns,
                max_prompt_length=max_prompt_len,
                max_completion_length=max_comp_len,
                max_per_turn_new_tokens=max_turn_new,
            )
            if grpo_cfg.enable_multiturn_rollout
            else None
        )

        curve_path = grpo_cfg.live_curve_jsonl_path or (output_dir / f"grpo_live_curve_{grpo_cfg.ablation_mode}.jsonl")
        callbacks = [LiveJSONLTelemetryCallback(curve_path, stage=f"grpo_{grpo_cfg.ablation_mode}")]

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "reward_funcs": reward_funcs,
            "args": grpo_args,
            "train_dataset": hf_dataset,
            "peft_config": lora_cfg,
            "processing_class": tokenizer,
            "callbacks": callbacks,
        }
        init_sig = inspect.signature(GRPOTrainer.__init__)
        if "processing_class" not in init_sig.parameters and "tokenizer" in init_sig.parameters:
            trainer_kwargs.pop("processing_class", None)
            trainer_kwargs["tokenizer"] = tokenizer

        filtered_kwargs = _filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs)
        trainer = MultiTurnAgenticGRPOTrainer(
            **filtered_kwargs,
            rollout_engine=rollout_engine,
            reward_engine=reward_engine,
            enable_multiturn_rollout=grpo_cfg.enable_multiturn_rollout,
        )
        if model_cfg.gradient_checkpointing and not model_cfg.dry_run and hasattr(trainer.model, "enable_input_require_grads"):
            trainer.model.enable_input_require_grads()

        resume_ckpt = _find_latest_trainer_checkpoint(runs_dir) if grpo_cfg.resume_from_checkpoint else None
        train_res = trainer.train(resume_from_checkpoint=resume_ckpt)
        raw_logs = list(trainer.state.log_history)

        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        base_used = str(base_init_path) if base_init_path is not None else model_cfg.resolved_model_path()

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed_sec = round(time.perf_counter() - t0, 2)
        summary = {
            "stage": "grpo",
            "ablation_mode": grpo_cfg.ablation_mode,
            "enable_multiturn_rollout": grpo_cfg.enable_multiturn_rollout,
            "max_rollout_turns": max_turns,
            "beta_kl": grpo_cfg.beta,
            "num_generations": num_gen,
            "use_vllm": active_vllm,
            "reward_weights": weights.to_dict(),
            "active_reward_funcs": [fn.__name__ for fn in reward_funcs],
            "prompt_count": len(hf_dataset),
            "base_model_used": to_portable_posix_path(base_used),
            "adapter_dir": to_portable_posix_path(adapter_dir),
            "resumed_from_checkpoint": to_portable_posix_path(resume_ckpt) if resume_ckpt else None,
            "live_curve_jsonl": to_portable_posix_path(curve_path),
            "train_runtime_sec": elapsed_sec,
            "global_steps": int(train_res.global_step),
            "train_loss": round(float(train_res.training_loss), 5),
            "log_history": raw_logs,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        if owns_sandbox:
            sb.close()
