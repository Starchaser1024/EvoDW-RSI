"""Configuration dataclasses for EvoDW-RL v2 (Loss-Masked SFT & Multi-Turn Agentic GRPO)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.config import PROJECT_ROOT, DialectType, to_portable_posix_path

MODEL_PRESETS: dict[str, str] = {
    "1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "xiyansql-3b": "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
    "xiyansql-7b": "XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
}

VERTICAL_SQL_MODEL_PRESETS: dict[str, str] = {
    "1.5b": "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
    "3b": "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
    "7b": "XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
}

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

RewardAblationMode = Literal[
    "full",
    "no_format",
    "no_gateway",
    "no_exec",
    "no_self_repair",
    "outcome_only",
]


def resolve_model_name_or_path(
    model_spec: str = DEFAULT_BASE_MODEL,
    local_cache_roots: tuple[Path, ...] = (
        Path("/data/llm-models"),
        Path("/content/drive/MyDrive/EvoDW_RL_v2_Workspace/models"),
    ),
) -> str:
    """Resolve shorthand presets ('1.5b', '3b', '7b', 'xiyansql-3b/7b') and prefer local directories."""
    spec = (model_spec or DEFAULT_BASE_MODEL).strip()
    canonical = MODEL_PRESETS.get(spec.lower(), spec)

    direct_path = Path(canonical)
    if direct_path.exists():
        return str(direct_path.resolve())

    repo_leaf = canonical.split("/")[-1]
    for root in local_cache_roots:
        candidate = root / repo_leaf
        if (candidate / "config.json").exists():
            return str(candidate.resolve())

    return canonical


def resolve_vertical_sql_model(scale_or_spec: str = "3b") -> str:
    """Resolve corresponding open-source vertical Text-to-SQL model (`XiYanSQL-QwenCoder-3B/7B-2504`)."""
    key = (scale_or_spec or "3b").strip().lower()
    if key in VERTICAL_SQL_MODEL_PRESETS:
        return resolve_model_name_or_path(VERTICAL_SQL_MODEL_PRESETS[key])
    if "7b" in key:
        return resolve_model_name_or_path(VERTICAL_SQL_MODEL_PRESETS["7b"])
    if "xiyan" in key:
        return resolve_model_name_or_path(scale_or_spec)
    return resolve_model_name_or_path(VERTICAL_SQL_MODEL_PRESETS["3b"])


@dataclass
class RLModelConfig:
    """Base small-LM and LoRA / 4-bit QLoRA / FlashAttention-2 configuration."""

    model_name_or_path: str = DEFAULT_BASE_MODEL
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"  # bfloat16 | float16 | float32 | auto
    attn_implementation: str = "auto"  # auto | flash_attention_2 | sdpa | eager
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    dry_run: bool = False
    seed: int = 2026

    def resolved_model_path(self) -> str:
        if self.dry_run:
            return "dry-run-tiny-qwen2"
        return resolve_model_name_or_path(self.model_name_or_path)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        data["resolved_model_path"] = to_portable_posix_path(self.resolved_model_path())
        return data


@dataclass
class SFTTrainingConfig:
    """Hyperparameters for Stage-1 Loss-Masked Multi-Turn Trajectory SFT (299+ verified trajectories)."""

    sft_jsonl_path: Path = PROJECT_ROOT / "outputs" / "rl_export" / "sft_openai_tool_use.jsonl"
    output_dir: Path = PROJECT_ROOT / "outputs" / "rl_checkpoints" / "sft"
    max_seq_length: int = 9216  # Eliminates tail/middle truncation across all 299 multi-turn trajectories (max=8666)
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    save_steps: int = 0  # Set >0 (e.g., 25 on Colab Drive) for disconnect-proof periodic checkpoints
    save_total_limit: int = 2
    resume_from_checkpoint: bool = False
    live_curve_jsonl_path: Path | None = None
    save_merged_model: bool = True

    # Adaptive scarcity reweighting for self-repair & inspect_table_sample probe trajectories
    adaptive_reweight: bool = True
    upsample_self_repaired: int = 2
    upsample_probe: int = 2
    target_self_repair_ratio: float = 0.22
    target_probe_ratio: float = 0.18

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sft_jsonl_path"] = to_portable_posix_path(self.sft_jsonl_path)
        d["output_dir"] = to_portable_posix_path(self.output_dir)
        d["live_curve_jsonl_path"] = (
            to_portable_posix_path(self.live_curve_jsonl_path) if self.live_curve_jsonl_path else None
        )
        return d


@dataclass
class RewardStageWeights:
    """Configurable weights and ablation switches for the v2 4-stage progressive + self-repair SQL RL reward."""

    enable_format: bool = True
    enable_gateway_ast: bool = True
    enable_exec_validity: bool = True
    enable_ex_outcome: bool = True
    enable_self_repair_bonus: bool = True
    enable_step_length_penalty: bool = True
    sparse_outcome_only: bool = False

    format_scale: float = 1.0
    gateway_scale: float = 1.0
    exec_scale: float = 1.0
    outcome_scale: float = 1.0

    first_pass_bonus: float = 0.20
    self_repair_bonus: float = 0.25
    gateway_repair_bonus: float = 0.08
    step_penalty_per_turn: float = 0.04
    probe_step_penalty: float = 0.01
    length_free_chars: int = 1400
    max_length_penalty: float = 0.15

    @classmethod
    def from_ablation(cls, mode: RewardAblationMode = "full") -> "RewardStageWeights":
        if mode == "full":
            return cls()
        if mode == "no_format":
            return cls(enable_format=False)
        if mode == "no_gateway":
            return cls(enable_gateway_ast=False)
        if mode == "no_exec":
            return cls(enable_exec_validity=False)
        if mode == "no_self_repair":
            return cls(enable_self_repair_bonus=False)
        if mode == "outcome_only":
            # Pure 0/1 sparse EX outcome reward for clean sparse-vs-dense RL ablation
            return cls(
                enable_format=False,
                enable_gateway_ast=False,
                enable_exec_validity=False,
                enable_ex_outcome=True,
                enable_self_repair_bonus=False,
                enable_step_length_penalty=False,
                sparse_outcome_only=True,
            )
        raise ValueError(f"Unsupported reward ablation mode: {mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GRPOTrainingConfig:
    """Hyperparameters for Stage-2 Multi-Turn Agentic Observation-Masked GRPO."""

    grpo_jsonl_path: Path = PROJECT_ROOT / "outputs" / "rl_export" / "grpo_prompts.jsonl"
    sft_checkpoint_dir: Path | None = None
    policy_json_path: Path = PROJECT_ROOT / "outputs" / "policies" / "gepa_best_policy.json"
    output_dir: Path = PROJECT_ROOT / "outputs" / "rl_checkpoints" / "grpo"
    dialect: DialectType = "odps"
    ablation_mode: RewardAblationMode = "full"
    reward_weights: RewardStageWeights = field(default_factory=RewardStageWeights)

    # Multi-turn Agentic Rollout settings (Action mask=1, Sandbox Observation mask=0)
    enable_multiturn_rollout: bool = True
    max_rollout_turns: int = 3
    max_per_turn_new_tokens: int = 768

    num_generations: int = 8
    max_prompt_length: int = 4096
    max_completion_length: int = 2048
    num_train_epochs: float = 1.0
    max_steps: int = 100
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    beta: float = 0.04  # Strictly >0 KL anchor to SFT reference policy to prevent format collapse
    temperature: float = 0.8
    logging_steps: int = 2

    # Optional vLLM rollout acceleration & Drive checkpoint resume
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.45
    save_steps: int = 0
    save_total_limit: int = 2
    resume_from_checkpoint: bool = False
    live_curve_jsonl_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["grpo_jsonl_path"] = to_portable_posix_path(self.grpo_jsonl_path)
        d["sft_checkpoint_dir"] = (
            to_portable_posix_path(self.sft_checkpoint_dir) if self.sft_checkpoint_dir else None
        )
        d["policy_json_path"] = to_portable_posix_path(self.policy_json_path)
        d["output_dir"] = to_portable_posix_path(self.output_dir)
        d["live_curve_jsonl_path"] = (
            to_portable_posix_path(self.live_curve_jsonl_path) if self.live_curve_jsonl_path else None
        )
        return d
