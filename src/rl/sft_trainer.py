"""Stage 1: Loss-Masked Multi-Turn Trajectory SFT Trainer (src/rl/sft_trainer.py)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.config import to_portable_posix_path
from src.rl.config import RLModelConfig, SFTTrainingConfig
from src.rl.model_utils import (
    build_dry_run_model_and_tokenizer,
    load_base_model_for_training,
    merge_and_save_lora_checkpoint,
)
from src.rl.sft_dataset import (
    IGNORE_INDEX,
    LossMaskedMultiTurnCollator,
    MultiTurnSFTDataset,
    SFTCorpusStats,
    encode_multiturn_with_loss_mask,
    format_non_supervised_turn_chatml,
    get_isolated_test_task_ids,
    load_sft_trajectories,
    reweight_sft_records,
)


class LiveJSONLTelemetryCallback(TrainerCallback):
    """Flushes step-by-step training metrics (loss, reward, KL, collapse rate) to disk/Google Drive in real time."""

    def __init__(self, jsonl_path: Path | None, stage: str = "sft") -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.stage = stage
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        del args, control, kwargs
        if self.jsonl_path is None or not logs:
            return
        payload = {
            "stage": self.stage,
            "step": int(getattr(state, "global_step", 0)),
            "epoch": round(float(getattr(state, "epoch", 0.0) or 0.0), 4),
            "timestamp": round(time.time(), 3),
        }
        for k, v in logs.items():
            if isinstance(v, (int, float, bool, str)):
                payload[k] = round(v, 6) if isinstance(v, float) else v
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()


class LossMaskedSFTTrainer(Trainer):
    """Memory-efficient CausalLM Trainer for Loss-Masked Multi-Turn Trajectories.

    Projects `lm_head` ONLY on supervised positions (`shift_labels != -100`), producing
    bit-for-bit identical Cross-Entropy loss while avoiding multi-GB `[B, L, V]` FP32
    logit allocations over masked `system`/`user`/`tool` context tokens.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        unwrapped = model.get_base_model() if hasattr(model, "get_base_model") else model
        labels = inputs.get("labels")
        if (
            labels is not None
            and not return_outputs
            and hasattr(unwrapped, "model")
            and hasattr(unwrapped, "lm_head")
        ):
            backbone_out = unwrapped.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )
            hidden_states = backbone_out[0]
            shift_hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.size(-1))
            shift_labels = labels[..., 1:].reshape(-1)
            valid_mask = shift_labels != IGNORE_INDEX
            if bool(valid_mask.any()):
                sup_logits = unwrapped.lm_head(shift_hidden[valid_mask]).float()
                loss = torch.nn.functional.cross_entropy(sup_logits, shift_labels[valid_mask])
            else:
                loss = shift_hidden.sum() * 0.0
            return loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def _find_latest_trainer_checkpoint(runs_dir: Path) -> str | None:
    if not runs_dir.exists():
        return None
    ckpts = sorted(
        [p for p in runs_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def run_sft_training(
    model_cfg: RLModelConfig,
    sft_cfg: SFTTrainingConfig,
) -> dict[str, Any]:
    """Execute Stage-1 Loss-Masked Multi-Turn SFT and save LoRA + optional merged checkpoint."""
    t0 = time.perf_counter()
    output_dir = Path(sft_cfg.output_dir)
    adapter_dir = output_dir / "sft_adapter"
    merged_dir = output_dir / "sft_merged"
    summary_path = output_dir / "sft_training_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disconnect-proof fast-skip if Stage-1 SFT already completed cleanly
    if (
        sft_cfg.resume_from_checkpoint
        and summary_path.exists()
        and (adapter_dir / "adapter_config.json").exists()
        and (not sft_cfg.save_merged_model or (merged_dir / "config.json").exists())
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    records = load_sft_trajectories(sft_cfg.sft_jsonl_path, strict_no_test_leakage=True)
    if model_cfg.dry_run:
        records = records[:4]

    dry_base_dir: Path | None = None
    if model_cfg.dry_run:
        dry_base_dir = output_dir / "dry_run_base"
        build_dry_run_model_and_tokenizer(seed=model_cfg.seed, save_dir=dry_base_dir)
        model, tokenizer = load_base_model_for_training(
            model_cfg,
            base_override_path=dry_base_dir,
            attach_lora=True,
        )
    else:
        model, tokenizer = load_base_model_for_training(model_cfg, attach_lora=True)

    tokenizer.padding_side = "right"
    eff_seq_len = min(256, sft_cfg.max_seq_length) if model_cfg.dry_run else sft_cfg.max_seq_length
    dataset = MultiTurnSFTDataset(
        records,
        tokenizer=tokenizer,
        max_seq_length=eff_seq_len,
        upsample_self_repaired=1 if model_cfg.dry_run else sft_cfg.upsample_self_repaired,
        upsample_probe=1 if model_cfg.dry_run else sft_cfg.upsample_probe,
        adaptive_reweight=False if model_cfg.dry_run else sft_cfg.adaptive_reweight,
        target_self_repair_ratio=sft_cfg.target_self_repair_ratio,
        target_probe_ratio=sft_cfg.target_probe_ratio,
        seed=model_cfg.seed,
    )
    collator = LossMaskedMultiTurnCollator(
        tokenizer=tokenizer,
        max_seq_length=eff_seq_len,
        pad_to_multiple_of=8,
    )

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

    effective_max_steps = sft_cfg.max_steps
    if model_cfg.dry_run and effective_max_steps <= 0:
        effective_max_steps = 3

    runs_dir = output_dir / "trainer_runs"
    save_strategy = "steps" if (sft_cfg.save_steps > 0 and not model_cfg.dry_run) else "no"

    train_args = TrainingArguments(
        output_dir=str(runs_dir),
        num_train_epochs=sft_cfg.num_train_epochs,
        max_steps=effective_max_steps,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=1 if model_cfg.dry_run else sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        warmup_ratio=sft_cfg.warmup_ratio,
        weight_decay=sft_cfg.weight_decay,
        max_grad_norm=sft_cfg.max_grad_norm,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        logging_steps=1 if model_cfg.dry_run else sft_cfg.logging_steps,
        save_strategy=save_strategy,
        save_steps=max(1, sft_cfg.save_steps) if save_strategy == "steps" else 500,
        save_total_limit=sft_cfg.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=model_cfg.dry_run or (not torch.cuda.is_available()),
        dataloader_pin_memory=use_cuda,
        seed=model_cfg.seed,
    )

    curve_path = sft_cfg.live_curve_jsonl_path or (output_dir / "sft_live_curve.jsonl")
    callbacks = [LiveJSONLTelemetryCallback(curve_path, stage="sft")]

    trainer = LossMaskedSFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    resume_ckpt = _find_latest_trainer_checkpoint(runs_dir) if sft_cfg.resume_from_checkpoint else None
    train_res = trainer.train(resume_from_checkpoint=resume_ckpt)
    raw_logs = list(trainer.state.log_history)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    saved_merged_path: str | None = None
    if sft_cfg.save_merged_model:
        base_for_merge = str(dry_base_dir) if dry_base_dir is not None else model_cfg.resolved_model_path()
        merge_and_save_lora_checkpoint(
            base_model_path=base_for_merge,
            adapter_dir=adapter_dir,
            merged_output_dir=merged_dir,
            compute_dtype="float32" if model_cfg.dry_run else model_cfg.compute_dtype,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        saved_merged_path = str(merged_dir)

    elapsed_sec = round(time.perf_counter() - t0, 2)
    loss_history = [
        {"step": int(entry["step"]), "loss": round(float(entry["loss"]), 5)}
        for entry in raw_logs
        if "loss" in entry and "step" in entry
    ]

    summary = {
        "stage": "sft",
        "model_config": model_cfg.to_dict(),
        "sft_config": sft_cfg.to_dict(),
        "corpus_stats": dataset.stats.to_dict(),
        "train_runtime_sec": elapsed_sec,
        "train_loss": round(float(train_res.training_loss), 5),
        "global_steps": int(train_res.global_step),
        "resumed_from_checkpoint": to_portable_posix_path(resume_ckpt) if resume_ckpt else None,
        "loss_history": loss_history,
        "live_curve_jsonl": to_portable_posix_path(curve_path),
        "adapter_dir": to_portable_posix_path(adapter_dir),
        "merged_dir": to_portable_posix_path(saved_merged_path) if saved_merged_path else None,
        "dry_run_base_dir": to_portable_posix_path(dry_base_dir) if dry_base_dir else None,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
