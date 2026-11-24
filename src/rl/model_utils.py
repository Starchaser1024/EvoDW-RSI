"""Model & tokenizer loader supporting Qwen2.5-Coder / XiYanSQL (1.5B/3B/7B), QLoRA, FlashAttention-2/SDPA, LoRA merge, and CPU Dry-Run."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from src.rl.config import RLModelConfig

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "{% if message['role'] == 'tool' %}"
    "<|im_start|>user\n<tool_response>\n{{ message['content'] }}\n</tool_response><|im_end|>\n"
    "{% else %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def is_flash_attn_2_supported() -> bool:
    """Return True when `flash_attn` is importable and GPU is Ampere (sm_80) or newer."""
    if not torch.cuda.is_available():
        return False
    if importlib.util.find_spec("flash_attn") is None:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def is_vllm_available() -> bool:
    """Return True when `vllm` is installed and CUDA is available."""
    return torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None


def resolve_attn_implementation(requested: str = "auto", *, compute_dtype: torch.dtype | None = None) -> str:
    """Resolve optimal attention kernel (`flash_attention_2` -> `sdpa` -> `eager`)."""
    req = (requested or "auto").strip().lower()
    if not torch.cuda.is_available():
        return "eager"
    if req in ("eager", "sdpa", "flash_attention_2"):
        if req == "flash_attention_2" and not is_flash_attn_2_supported():
            return "sdpa"
        return req
    if compute_dtype in (torch.bfloat16, torch.float16, None) and is_flash_attn_2_supported():
        return "flash_attention_2"
    return "sdpa"


def resolve_torch_dtype(dtype_name: str = "bfloat16") -> torch.dtype:
    """Select optimal compute dtype given hardware availability."""
    name = (dtype_name or "auto").lower()
    if not torch.cuda.is_available():
        return torch.float32
    if name in ("bfloat16", "bf16", "auto"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if name in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def build_dry_run_tokenizer() -> PreTrainedTokenizerFast:
    """Create a deterministic offline ByteLevel ChatML tokenizer (lossless on UTF-8 Chinese + SQL)."""
    byte_alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {tok: idx for idx, tok in enumerate(byte_alphabet)}
    tok_backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok_backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_backend.decoder = decoders.ByteLevel()

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="<|endoftext|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        model_input_names=["input_ids", "attention_mask"],
    )
    fast_tok.chat_template = QWEN_CHAT_TEMPLATE
    fast_tok.padding_side = "right"
    return fast_tok


def build_dry_run_model_and_tokenizer(
    seed: int = 2026,
    save_dir: Path | None = None,
) -> tuple[Qwen2ForCausalLM, PreTrainedTokenizerFast]:
    """Create an ultra-lightweight (~0.4M params) Qwen2 causal LM for offline CPU/GPU smoke tests."""
    torch.manual_seed(seed)
    tokenizer = build_dry_run_tokenizer()
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16384,
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Qwen2ForCausalLM(cfg)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    return model, tokenizer


def build_lora_config(cfg: RLModelConfig) -> LoraConfig:
    """Build PEFT LoraConfig targeting attention & SwiGLU MLP projections."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )


def build_bnb_4bit_config(cfg: RLModelConfig) -> BitsAndBytesConfig | None:
    """Build 4-bit NF4 QLoRA quantization config when CUDA is available."""
    if not cfg.use_4bit or cfg.dry_run or not torch.cuda.is_available():
        return None
    compute_dtype = resolve_torch_dtype(cfg.compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(
    cfg: RLModelConfig,
    override_path: str | Path | None = None,
) -> PreTrainedTokenizerFast:
    """Load and standardize ChatML tokenizer (pad_token, chat_template)."""
    if cfg.dry_run and override_path is None:
        return build_dry_run_tokenizer()

    target = str(override_path) if override_path is not None else cfg.resolved_model_path()
    tok = AutoTokenizer.from_pretrained(target, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        tok.chat_template = QWEN_CHAT_TEMPLATE
    return tok


def load_base_model_for_training(
    cfg: RLModelConfig,
    *,
    base_override_path: str | Path | None = None,
    attach_lora: bool = True,
) -> tuple[Any, PreTrainedTokenizerFast]:
    """Load CausalLM (4-bit QLoRA / bf16 FlashAttention-2 LoRA / CPU Dry-Run) ready for SFT or GRPO."""
    if cfg.dry_run and base_override_path is None:
        model, tokenizer = build_dry_run_model_and_tokenizer(seed=cfg.seed)
        if attach_lora:
            model = get_peft_model(model, build_lora_config(cfg))
        return model, tokenizer

    model_path = str(base_override_path) if base_override_path is not None else cfg.resolved_model_path()
    tokenizer = load_tokenizer(cfg, override_path=model_path)
    compute_dtype = resolve_torch_dtype(cfg.compute_dtype)
    bnb_cfg = build_bnb_4bit_config(cfg)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
    }
    if not cfg.dry_run and torch.cuda.is_available():
        load_kwargs["attn_implementation"] = resolve_attn_implementation(
            cfg.attn_implementation,
            compute_dtype=compute_dtype,
        )

    if bnb_cfg is not None:
        load_kwargs["quantization_config"] = bnb_cfg
        load_kwargs["torch_dtype"] = compute_dtype
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif torch.cuda.is_available() and not cfg.dry_run:
        load_kwargs["torch_dtype"] = compute_dtype
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False

    if bnb_cfg is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.gradient_checkpointing and not cfg.dry_run and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if attach_lora:
        model = get_peft_model(model, build_lora_config(cfg))
        if cfg.gradient_checkpointing and not cfg.dry_run and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, tokenizer


def merge_and_save_lora_checkpoint(
    base_model_path: str | Path,
    adapter_dir: Path,
    merged_output_dir: Path,
    *,
    compute_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> Path:
    """Merge a trained LoRA adapter into non-quantized base weights for zero-overhead GRPO reference/eval."""
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_torch_dtype(compute_dtype)

    # On A100/H100 (>=24GB free VRAM), merge directly on GPU for 10x faster merging; on 12GB GPUs, merge safely on CPU.
    merge_device: Any = "cpu"
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes >= 22 * (1024**3):
            merge_device = {"": torch.cuda.current_device()}

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        device_map=merge_device,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_output_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=trust_remote_code)
    tok.save_pretrained(merged_output_dir)
    del merged, peft_model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged_output_dir


def load_model_for_inference(
    model_name_or_path: str | Path,
    *,
    adapter_path: str | Path | None = None,
    use_4bit: bool = False,
    compute_dtype: str = "bfloat16",
    attn_implementation: str = "auto",
    trust_remote_code: bool = True,
    dry_run: bool = False,
) -> tuple[Any, PreTrainedTokenizerFast]:
    """Load Base, Base+LoRA, or Merged model in eval mode for deterministic SQL agent evaluation."""
    if dry_run and not Path(str(model_name_or_path)).exists():
        model, tok = build_dry_run_model_and_tokenizer()
        if adapter_path and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        return model, tok

    resolved_base = str(model_name_or_path)
    tok_source = (
        str(adapter_path)
        if (adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists())
        else resolved_base
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    dtype = resolve_torch_dtype(compute_dtype)
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if not dry_run and torch.cuda.is_available():
        load_kwargs["attn_implementation"] = resolve_attn_implementation(
            attn_implementation,
            compute_dtype=dtype,
        )

    if use_4bit and torch.cuda.is_available() and not dry_run:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif torch.cuda.is_available() and not dry_run:
        load_kwargs["torch_dtype"] = dtype
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(resolved_base, **load_kwargs)
    if adapter_path is not None and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if not use_4bit and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer
