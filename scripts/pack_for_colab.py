#!/usr/bin/env python3
"""Zero-friction Colab A100 bundle packer & standalone Notebook generator (`scripts/pack_for_colab.py`).

Creates:
1. `notebooks/EvoDW_RL_Colab_A100.ipynb`: Self-contained Colab A100/H100 notebook with Google Drive
   real-time checkpointing, vLLM / BF16 FlashAttention-2 auto-detection, and 1-click execution of
   all 3 core EvoDW-RL v2 experiments (Main Ladder, XiYanSQL Vertical Control, Sparse-vs-Dense GRPO Ablation).
2. `outputs/colab_bundle/evodw_rl_v2_colab_bundle.tar.gz`: Lightweight (<2MB compressed), secret-scrubbed
   source + benchmark + 299-trajectory `rl_export` archive (excluding `.venv`, `.git`, `.env`, and model weights).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.config import to_portable_posix_path

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:LLM|OPENAI|OPENCODE)_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{16,}"),
)

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "evodw_rsi.egg-info",
    "rl_checkpoints",
    ".live_cache",
    "colab_bundle",
    "cache",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".pkl",
    ".tar.gz",
    ".zip",
}

INCLUDED_TOP_PATHS = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-rl.txt",
    ".env.example",
    "README.md",
    "docs",
    "prompts",
    "src",
    "scripts",
    "tests",
    "notebooks/EvoDW_RL_Colab_A100.ipynb",
    "outputs/rl_export/export_summary.json",
    "outputs/rl_export/grpo_prompts.jsonl",
    "outputs/rl_export/sft_openai_tool_use.jsonl",
    "outputs/policies/gepa_best_policy.json",
    "outputs/runs/gepa_summary.json",
    "outputs/rl_eval/rl_eval_summary.json",
)


def sanitize_text_secrets(text: str) -> tuple[str, int]:
    """Redact any accidental token-like substrings from exported text files."""
    redactions = 0
    out = text
    for pat in SECRET_PATTERNS:
        out, n = pat.subn("[REDACTED_SECRET]", out)
        redactions += n
    return out, redactions


def _md_cell(cell_id: str, lines: list[str]) -> dict[str, Any]:
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines],
    }


def _code_cell(cell_id: str, code: str) -> dict[str, Any]:
    raw_lines = code.strip("\n").splitlines()
    return {
        "id": cell_id,
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"{line}\n" for line in raw_lines],
    }


def build_colab_notebook_dict() -> dict[str, Any]:
    """Construct the standalone `notebooks/EvoDW_RL_Colab_A100.ipynb` notebook structure."""
    cells: list[dict[str, Any]] = [
        _md_cell(
            "intro-header",
            [
                "# EvoDW-RL v2：数仓多轮工具交互 SFT + Observation-Masked Agentic GRPO（Colab A100/H100 一键实训）",
                "",
                "本 Notebook 原生适配 **Google Colab A100 (40GB / 80GB) & H100**，内置 **Google Drive 实时断连保护**，一键完成以下 **3 组核心实验** 并导出可直接合入本地仓库的评测包：",
                "",
                "1. **主实验（`Qwen2.5-Coder-3B-Instruct` 或 `7B-Instruct`）**：`Zero-shot` vs `+GEPA Prompt` vs `Loss-Masked Multi-Turn SFT (3 epochs, 299+ 轨迹)` vs `Multi-turn Agentic SFT+GRPO (100+ steps, G=8, beta=0.04)`；",
                "2. **垂直开源 SQL 模型水土不服对照实验**：在企业私有 ODPS 分区网关（`ODPS-0123031` 分区裁剪强制拦截、`MAX_PT`、`LATERAL VIEW explode`）下，评测 `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`（或 `7B-2504`）Zero-shot 表现 vs 本文后训练模型表现；",
                "3. **GRPO 奖励消融实验（Ablation）**：严格固定起点 `pi_SFT` 与超参，对比 `Outcome-Only Sparse Reward（仅最终 EX 0/1）` vs `4-Stage Dense + Self-Repair Reward（四级稠密 + 自修复奖励 + 步数/长度惩罚）` 的收敛速度、格式坍塌率与最终隔离测试集 `EX`。",
            ],
        ),
        _code_cell(
            "cfg-and-drive",
            """
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
import torch

# ===================== 实验全局配置（可按 A100 40GB / 80GB 自由切换） =====================
MODEL_SCALE = os.environ.get("EVODW_MODEL_SCALE", "3b")          # 可选: "3b" 或 "7b"
SFT_EPOCHS = float(os.environ.get("EVODW_SFT_EPOCHS", "3.0"))
GRPO_STEPS = int(os.environ.get("EVODW_GRPO_STEPS", "100"))      # >= 100 steps
GRPO_GROUP_SIZE = int(os.environ.get("EVODW_GRPO_G", "8"))       # G = 8
GRPO_KL_BETA = float(os.environ.get("EVODW_GRPO_BETA", "0.04"))  # beta > 0 防格式坍塌
ENABLE_VLLM = os.environ.get("EVODW_USE_VLLM", "auto")           # "auto" | "1" | "0"

# 自动检测 GPU 规格与显存容量
assert torch.cuda.is_available(), "请在 Colab Runtime -> Change runtime type 中选择 A100 / H100 GPU！"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
USE_BF16_NO_4BIT = vram_gb >= 36.0  # A100 40GB/80GB & H100 直接启用高吞吐 BF16 LoRA（免 QLoRA 反量化开销）
print(f"[GPU] {gpu_name} | VRAM={vram_gb:.1f} GB | SM={cc_major}.{cc_minor} | BF16_Full_LoRA={USE_BF16_NO_4BIT}")

# 自动挂载 Google Drive 实时保存 LoRA Checkpoints、曲线 JSONL 与断点评测缓存
IN_COLAB = False
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    IN_COLAB = True
    DRIVE_ROOT = Path("/content/drive/MyDrive/EvoDW_RL_v2_Workspace")
except Exception:
    DRIVE_ROOT = Path("./colab_workspace").resolve()

DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT = DRIVE_ROOT / f"checkpoints_{MODEL_SCALE}"
CURVE_ROOT = DRIVE_ROOT / f"curves_{MODEL_SCALE}"
EVAL_ROOT = DRIVE_ROOT / f"eval_{MODEL_SCALE}"
EXPORT_ROOT = DRIVE_ROOT / "colab_exports"
for d in (CKPT_ROOT, CURVE_ROOT, EVAL_ROOT, EXPORT_ROOT):
    d.mkdir(parents=True, exist_ok=True)
print(f"[Workspace] Persistent checkpoint & curve root -> {DRIVE_ROOT}")
""",
        ),
        _code_cell(
            "unpack-bundle",
            """
# ===================== 解压 EvoDW-RL v2 实训代码与 299 条脱敏轨迹包 =====================
WORK_REPO = Path("/content/EvoDW-RSI") if Path("/content").exists() else Path.cwd()
bundle_candidates = [
    DRIVE_ROOT / "evodw_rl_v2_colab_bundle.tar.gz",
    Path("/content/evodw_rl_v2_colab_bundle.tar.gz"),
    Path("evodw_rl_v2_colab_bundle.tar.gz"),
]
bundle_found = next((p for p in bundle_candidates if p.exists()), None)

if bundle_found is not None:
    print(f"[Bundle] Extracting {bundle_found} -> {WORK_REPO}")
    WORK_REPO.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_found, "r:gz") as tar:
        tar.extractall(path=WORK_REPO)
elif not (WORK_REPO / "src" / "rl" / "multiturn_rollout.py").exists():
    if IN_COLAB:
        from google.colab import files  # type: ignore
        print("请上传本地生成的 outputs/colab_bundle/evodw_rl_v2_colab_bundle.tar.gz：")
        uploaded = files.upload()
        up_name = next(iter(uploaded.keys()))
        shutil.copy2(up_name, DRIVE_ROOT / "evodw_rl_v2_colab_bundle.tar.gz")
        WORK_REPO.mkdir(parents=True, exist_ok=True)
        with tarfile.open(up_name, "r:gz") as tar:
            tar.extractall(path=WORK_REPO)

os.chdir(WORK_REPO)
print(f"[Repo] Active working directory: {Path.cwd()}")
assert (WORK_REPO / "outputs" / "rl_export" / "sft_openai_tool_use.jsonl").exists(), "Missing sft_openai_tool_use.jsonl!"
assert (WORK_REPO / "outputs" / "rl_export" / "grpo_prompts.jsonl").exists(), "Missing grpo_prompts.jsonl!"
""",
        ),
        _code_cell(
            "install-deps",
            """
# ===================== 安装 RL / 沙盒依赖并自动探测 FlashAttention-2 & vLLM =====================
subprocess.run(["pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run(
    ["pip", "install", "-q", "-U", "duckdb>=1.1.0", "sqlglot>=25.0.0", "rich>=13.7.0",
     "pydantic>=2.7.0", "python-dotenv>=1.0.1", "openai>=1.50.0",
     "transformers>=4.51.0,<5.0.0", "peft>=0.14.0", "trl>=0.16.0,<1.0.0",
     "bitsandbytes>=0.45.0", "datasets>=3.0.0,<5.0.0", "accelerate>=1.2.0", "pytest>=8.0.0", "pytest-asyncio>=0.23.0"],
    check=True,
)
subprocess.run(["pip", "install", "-q", "-e", ".", "--no-deps"], check=True)

# 尝试启用 FlashAttention-2（若 Colab 预编译 wheel 匹配则加速长序列多轮训练，否则自动回退 PyTorch CuDNN SDPA）
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except Exception:
    ATTN_IMPL = "auto"

USE_VLLM_FLAG = False
if ENABLE_VLLM in ("1", "true", "auto") and USE_BF16_NO_4BIT:
    try:
        import vllm  # noqa: F401
        USE_VLLM_FLAG = True
    except Exception:
        USE_VLLM_FLAG = False

print(f"[Runtime] ATTN_IMPL={ATTN_IMPL} | USE_VLLM_FLAG={USE_VLLM_FLAG}")
""",
        ),
        _code_cell(
            "sanity-pytest",
            """
# ===================== 训前自检：多轮 Observation Mask 与零测试集泄露校验 =====================
subprocess.run(["pytest", "-q", "tests/test_rl_sft_and_grpo.py", "-k", "not TestDryRunEndToEndPipeline"], check=True)
""",
        ),
        _code_cell(
            "stage1-sft",
            """
# ===================== Stage 1: Loss-Masked 多轮工具轨迹 SFT（299+ 轨迹 + 稀缺重加权） =====================
sft_dir = CKPT_ROOT / "sft"
sft_curve = CURVE_ROOT / "sft_live_curve.jsonl"
sft_batch = "2" if vram_gb >= 70 else "1"
sft_accum = "4" if sft_batch == "2" else "8"

sft_cmd = [
    "python", "scripts/train_sft.py",
    "--model", MODEL_SCALE,
    "--epochs", str(SFT_EPOCHS),
    "--batch-size", sft_batch,
    "--grad-accum", sft_accum,
    "--max-seq-length", "9216",
    "--upsample-repaired", "2",
    "--upsample-probe", "2",
    "--attn-impl", ATTN_IMPL,
    "--save-steps", "25",
    "--resume",
    "--live-curve-jsonl", str(sft_curve),
    "--output-dir", str(sft_dir),
]
if USE_BF16_NO_4BIT:
    sft_cmd.append("--no-4bit")

print("Running Stage-1 SFT:", " ".join(sft_cmd))
subprocess.run(sft_cmd, check=True)
""",
        ),
        _code_cell(
            "exp3-grpo-sparse",
            """
# ===================== 实验 3A（消融对照）：Outcome-Only 稀疏奖励多轮 GRPO =====================
grpo_sparse_dir = CKPT_ROOT / "grpo_outcome_only"
grpo_sparse_curve = CURVE_ROOT / "grpo_outcome_only_curve.jsonl"
grpo_bs = "8" if vram_gb >= 70 else "4"
grpo_accum = "1" if grpo_bs == "8" else "2"

sparse_cmd = [
    "python", "scripts/train_grpo.py",
    "--model", MODEL_SCALE,
    "--sft-checkpoint", str(sft_dir),
    "--output-dir", str(grpo_sparse_dir),
    "--ablate-reward", "outcome_only",
    "--max-rollout-turns", "3",
    "--num-generations", str(GRPO_GROUP_SIZE),
    "--batch-size", grpo_bs,
    "--grad-accum", grpo_accum,
    "--max-steps", str(GRPO_STEPS),
    "--beta", str(GRPO_KL_BETA),
    "--attn-impl", ATTN_IMPL,
    "--save-steps", "20",
    "--resume",
    "--live-curve-jsonl", str(grpo_sparse_curve),
]
if USE_BF16_NO_4BIT:
    sparse_cmd.append("--no-4bit")
if USE_VLLM_FLAG:
    sparse_cmd.append("--use-vllm")

print("Running Outcome-Only Sparse GRPO Ablation:", " ".join(sparse_cmd))
subprocess.run(sparse_cmd, check=True)
""",
        ),
        _code_cell(
            "exp1-grpo-dense",
            """
# ===================== 实验 1 & 3B（核心主模型）：4-Stage 稠密 + 自修复奖励多轮 GRPO =====================
grpo_dense_dir = CKPT_ROOT / "grpo_dense_full"
grpo_dense_curve = CURVE_ROOT / "grpo_dense_full_curve.jsonl"

dense_cmd = [
    "python", "scripts/train_grpo.py",
    "--model", MODEL_SCALE,
    "--sft-checkpoint", str(sft_dir),
    "--output-dir", str(grpo_dense_dir),
    "--ablate-reward", "full",
    "--max-rollout-turns", "3",
    "--num-generations", str(GRPO_GROUP_SIZE),
    "--batch-size", grpo_bs,
    "--grad-accum", grpo_accum,
    "--max-steps", str(GRPO_STEPS),
    "--beta", str(GRPO_KL_BETA),
    "--attn-impl", ATTN_IMPL,
    "--save-steps", "20",
    "--resume",
    "--live-curve-jsonl", str(grpo_dense_curve),
]
if USE_BF16_NO_4BIT:
    dense_cmd.append("--no-4bit")
if USE_VLLM_FLAG:
    dense_cmd.append("--use-vllm")

print("Running 4-Stage Dense + Self-Repair Multi-Turn GRPO:", " ".join(dense_cmd))
subprocess.run(dense_cmd, check=True)
""",
        ),
        _code_cell(
            "eval-all-three-experiments",
            """
# ===================== 一键评测三大实验（含 XiYanSQL 水土不服对照 + 断连续测缓存） =====================
vertical_preset = "xiyansql-7b" if "7b" in MODEL_SCALE.lower() else "xiyansql-3b"
report_md_path = EVAL_ROOT / f"rl_experiment_report_{MODEL_SCALE}.md"
summary_json_path = EVAL_ROOT / f"rl_eval_summary_{MODEL_SCALE}.json"

eval_cmd = [
    "python", "scripts/eval_rl_models.py",
    "--model", MODEL_SCALE,
    "--vertical-model", vertical_preset,
    "--sft-dir", str(sft_dir),
    "--grpo-dir", str(grpo_dense_dir),
    "--grpo-sparse-dir", str(grpo_sparse_dir),
    "--batch-size", "10" if vram_gb >= 70 else "6",
    "--resume",
    "--report-out", str(report_md_path),
    "--json-out", str(summary_json_path),
]
if not USE_BF16_NO_4BIT:
    eval_cmd.append("--use-4bit")

print("Running Unified 3-Experiment Evaluation:", " ".join(eval_cmd))
subprocess.run(eval_cmd, check=True)
""",
        ),
        _code_cell(
            "plot-and-export",
            """
# ===================== 绘制消融收敛曲线 & 打包回传本地 outputs/colab_imports/ =====================
import json
import matplotlib.pyplot as plt

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

sparse_logs = [x for x in _read_jsonl(grpo_sparse_curve) if "reward" in x]
dense_logs = [x for x in _read_jsonl(grpo_dense_curve) if "reward" in x]

if sparse_logs or dense_logs:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, rows, color in [
        ("Outcome-Only Sparse", sparse_logs, "#d95f02"),
        ("4-Stage Dense + Self-Repair", dense_logs, "#1b9e77"),
    ]:
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r.get("multiturn/online_ex_rate", 0.0) * 100 for r in rows], label=label, color=color, lw=2)
        axes[1].plot(steps, [r.get("multiturn/format_collapse_rate", 0.0) * 100 for r in rows], label=label, color=color, lw=2)
        axes[2].plot(steps, [r.get("kl", 0.0) for r in rows], label=label, color=color, lw=2)

    axes[0].set_title("Online Rollout EX Rate (%)")
    axes[1].set_title("Format Collapse Rate (%)")
    axes[2].set_title("Action-Masked KL Divergence")
    for ax in axes:
        ax.set_xlabel("GRPO Step")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plot_path = EVAL_ROOT / f"grpo_ablation_curves_{MODEL_SCALE}.png"
    plt.savefig(plot_path, dpi=180)
    plt.show()

export_tar = EXPORT_ROOT / f"evodw_colab_results_{MODEL_SCALE}.tar.gz"
with tarfile.open(export_tar, "w:gz") as tar:
    for fpath in [
        summary_json_path,
        report_md_path,
        sft_dir / "sft_training_summary.json",
        grpo_dense_dir / "grpo_training_summary.json",
        grpo_sparse_dir / "grpo_training_summary.json",
        sft_curve,
        grpo_dense_curve,
        grpo_sparse_curve,
    ]:
        if fpath.exists():
            tar.add(fpath, arcname=fpath.name)

shutil.copy2(summary_json_path, EXPORT_ROOT / summary_json_path.name)
shutil.copy2(report_md_path, EXPORT_ROOT / report_md_path.name)
print(f"[Export] Saved result archive to Google Drive -> {export_tar}")
print("下载后在本机运行: python scripts/merge_colab_results.py --input", export_tar.name)

if IN_COLAB:
    from google.colab import files  # type: ignore
    files.download(str(export_tar))
""",
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "A100",
                "machine_shape": "hm",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "cells": cells,
    }


def _should_skip_file(rel_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
        return True
    if rel_path.name.startswith(".env") and rel_path.name != ".env.example":
        return True
    for suf in EXCLUDED_SUFFIXES:
        if str(rel_path).endswith(suf):
            return True
    return False


def pack_colab_bundle(
    *,
    project_root: Path = PROJECT_ROOT,
    bundle_out_path: Path = PROJECT_ROOT / "outputs" / "colab_bundle" / "evodw_rl_v2_colab_bundle.tar.gz",
    notebook_out_path: Path = PROJECT_ROOT / "notebooks" / "EvoDW_RL_Colab_A100.ipynb",
) -> dict[str, Any]:
    """Generate the Colab A100 Notebook and build the lightweight secret-scrubbed `.tar.gz` archive."""
    notebook_out_path.parent.mkdir(parents=True, exist_ok=True)
    nb_dict = build_colab_notebook_dict()
    notebook_out_path.write_text(
        json.dumps(nb_dict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    bundle_out_path.parent.mkdir(parents=True, exist_ok=True)
    included_files: list[str] = []
    total_raw_bytes = 0
    total_redactions = 0

    with tarfile.open(bundle_out_path, "w:gz", compresslevel=6) as tar:
        for entry in INCLUDED_TOP_PATHS:
            target = project_root / entry
            if not target.exists():
                continue
            candidates = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
            for file_path in candidates:
                rel = file_path.relative_to(project_root)
                if _should_skip_file(rel):
                    continue
                raw_bytes = file_path.read_bytes()
                try:
                    txt = raw_bytes.decode("utf-8")
                    clean_txt, n_red = sanitize_text_secrets(txt)
                    total_redactions += n_red
                    payload_bytes = clean_txt.encode("utf-8")
                except UnicodeDecodeError:
                    payload_bytes = raw_bytes

                info = tarfile.TarInfo(name=rel.as_posix())
                info.size = len(payload_bytes)
                info.mtime = int(file_path.stat().st_mtime)
                info.mode = 0o755 if file_path.suffix == ".py" and rel.parts[0] == "scripts" else 0o644
                tar.addfile(info, io.BytesIO(payload_bytes))
                included_files.append(rel.as_posix())
                total_raw_bytes += len(payload_bytes)

    bundle_bytes = bundle_out_path.stat().st_size
    sha256_hex = hashlib.sha256(bundle_out_path.read_bytes()).hexdigest()

    def _rel_or_posix(p: Path) -> str:
        return to_portable_posix_path(p, project_root=project_root)

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "bundle_path": _rel_or_posix(bundle_out_path),
        "notebook_path": _rel_or_posix(notebook_out_path),
        "file_count": len(included_files),
        "uncompressed_bytes": total_raw_bytes,
        "compressed_bytes": bundle_bytes,
        "compressed_mb": round(bundle_bytes / (1024 * 1024), 3),
        "sha256": sha256_hex,
        "secret_redactions": total_redactions,
        "files": included_files,
    }
    manifest_path = bundle_out_path.parent / "colab_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package EvoDW-RL v2 lightweight archive & generate standalone Colab A100 Notebook."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "colab_bundle" / "evodw_rl_v2_colab_bundle.tar.gz",
        help="Output path for the lightweight .tar.gz Colab training bundle.",
    )
    parser.add_argument(
        "--notebook-out",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "EvoDW_RL_Colab_A100.ipynb",
        help="Output path for the generated Colab A100 Jupyter Notebook.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    console.rule("[bold green]EvoDW-RL v2: Packing Zero-Friction Colab A100 Bundle")
    manifest = pack_colab_bundle(bundle_out_path=args.output, notebook_out_path=args.notebook_out)

    tbl = Table(title="Colab A100 Bundle Summary", header_style="bold green")
    tbl.add_column("Field")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Notebook Generated", str(manifest["notebook_path"]))
    tbl.add_row("Archive Path", str(manifest["bundle_path"]))
    tbl.add_row("Packaged Files", str(manifest["file_count"]))
    tbl.add_row("Uncompressed Size", f"{manifest['uncompressed_bytes'] / (1024 * 1024):.2f} MB")
    tbl.add_row("Compressed Size", f"{manifest['compressed_mb']:.3f} MB")
    tbl.add_row("Secret Redactions", str(manifest["secret_redactions"]))
    tbl.add_row("SHA-256", manifest["sha256"][:24] + "...")
    console.print(tbl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
