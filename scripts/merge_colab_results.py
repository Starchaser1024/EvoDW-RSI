#!/usr/bin/env python3
"""Merge Colab A100 evaluation archives/JSONs into local `outputs/rl_experiment_report.md` (`scripts/merge_colab_results.py`).

Accepts either:
- A `.tar.gz` archive exported by `notebooks/EvoDW_RL_Colab_A100.ipynb` (`evodw_colab_results_3b.tar.gz`),
- A directory (`outputs/colab_imports/`), or
- Multiple `rl_eval_summary_*.json` files (e.g. local `1.5B` v1 + Colab `3B` / `7B` v2),

and automatically regenerates a publication-grade `outputs/rl_experiment_report.md` and
`outputs/rl_eval/rl_eval_summary.json` containing:
- Cross-scale progression table (`v1 1.5B` -> `v2 3B` / `7B`),
- Main 5+ model isolated Test Split comparison & 5-category accuracy table,
- `XiYanSQL-QwenCoder` vertical SQL model domain-shift comparison, and
- `Outcome-Only Sparse` vs `4-Stage Dense + Self-Repair` GRPO ablation table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import shutil
import sys
import tarfile
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.rl.eval_report import render_rl_markdown_report


def _extract_scale_entry(payload: dict[str, Any], fallback_label: str) -> dict[str, Any] | None:
    models_dict = payload.get("models")
    if not isinstance(models_dict, dict) or not models_dict:
        return None

    v_by_key = payload.get("variants_by_key") or {}

    def _find_metric(short_key: str, substrs: tuple[str, ...]) -> dict[str, Any]:
        if short_key in v_by_key and isinstance(v_by_key[short_key], dict):
            return v_by_key[short_key]
        for k, m in models_dict.items():
            kl = k.lower()
            if all(s.lower() in kl for s in substrs) and isinstance(m, dict):
                return m
        return {}

    m_base = _find_metric("base", ("base", "zero-shot"))
    m_gepa = _find_metric("base_gepa", ("base", "gepa"))
    m_sft = _find_metric("sft", ("sft", "冷启动"))
    # Prefer dense multi-turn GRPO over outcome_only sparse GRPO
    m_grpo = v_by_key.get("sft_grpo") or {}
    if not m_grpo:
        for k, m in models_dict.items():
            if "grpo" in k.lower() and "outcome-only" not in k.lower() and "sparse" not in k.lower():
                m_grpo = m
                break

    if not m_grpo:
        return None

    sft_sum = payload.get("sft_training_summary") or {}
    grpo_sum = payload.get("grpo_training_summary") or {}
    sft_n = (sft_sum.get("corpus_stats") or {}).get("total_records")
    grpo_steps = grpo_sum.get("global_steps")
    if sft_n and grpo_steps:
        budget_str = f"SFT {sft_n} trajs / GRPO {grpo_steps} steps"
    elif payload.get("version") == "v2":
        budget_str = "SFT 299 trajs / Multi-Turn GRPO"
    else:
        budget_str = "v1 Baseline (85 trajs / 8 steps)"

    short_name = payload.get("base_model_short") or Path(str(payload.get("base_model", fallback_label))).name
    return {
        "scale_label": f"{short_name} ({'v2 Colab' if payload.get('version') == 'v2' else 'v1 Local'})",
        "train_budget": budget_str,
        "base_ex": float(m_base.get("ex", 0.0)),
        "gepa_ex": float(m_gepa.get("ex", 0.0)),
        "sft_ex": float(m_sft.get("ex", 0.0)),
        "grpo_ex": float(m_grpo.get("ex", 0.0)),
        "grpo_invalid_rate": float(m_grpo.get("invalid_sql_rate", 0.0)),
        "grpo_latency_ms": float(m_grpo.get("avg_latency_ms", 0.0)),
    }


def _collect_json_payloads_from_path(
    src: Path,
    import_dest_dir: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Load summary JSON(s) from `.tar.gz`, directory, or `.json` file and persist imported copies."""
    import_dest_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[str, dict[str, Any]]] = []

    if src.is_file() and src.name.endswith((".tar.gz", ".tgz")):
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_path = Path(tmpd)
            with tarfile.open(src, "r:gz") as tar:
                tar.extractall(path=tmp_path)
            for f in sorted(tmp_path.rglob("*")):
                if f.is_file():
                    shutil.copy2(f, import_dest_dir / f.name)
            for jf in sorted(tmp_path.rglob("rl_eval_summary*.json")):
                loaded.append((jf.stem, json.loads(jf.read_text(encoding="utf-8"))))
        return loaded

    if src.is_dir():
        for jf in sorted(src.glob("rl_eval_summary*.json")):
            if jf.parent.resolve() != import_dest_dir.resolve():
                shutil.copy2(jf, import_dest_dir / jf.name)
            loaded.append((jf.stem, json.loads(jf.read_text(encoding="utf-8"))))
        return loaded

    if src.is_file() and src.suffix == ".json":
        if src.parent.resolve() != import_dest_dir.resolve():
            shutil.copy2(src, import_dest_dir / src.name)
        loaded.append((src.stem, json.loads(src.read_text(encoding="utf-8"))))
        return loaded

    raise FileNotFoundError(f"Unsupported or missing Colab result input: {src}")


def merge_colab_results(
    inputs: list[Path],
    *,
    baseline_v1_json: Path | None = PROJECT_ROOT / "outputs" / "rl_eval" / "rl_eval_summary.json",
    import_dir: Path = PROJECT_ROOT / "outputs" / "colab_imports",
    report_out: Path = PROJECT_ROOT / "outputs" / "rl_experiment_report.md",
    json_out: Path = PROJECT_ROOT / "outputs" / "rl_eval" / "rl_eval_summary.json",
) -> dict[str, Any]:
    """Merge one or more Colab evaluation summaries with optional v1 1.5B baseline and render Markdown report."""
    scale_rows: list[dict[str, Any]] = []

    if baseline_v1_json is not None and baseline_v1_json.exists():
        v1_data = json.loads(baseline_v1_json.read_text(encoding="utf-8"))
        # Preserve backup of v1 summary before overwriting json_out
        v1_backup = import_dir / "rl_eval_summary_v1_local_1.5b.json"
        import_dir.mkdir(parents=True, exist_ok=True)
        if not v1_backup.exists() and v1_data.get("version") != "v2":
            v1_backup.write_text(json.dumps(v1_data, ensure_ascii=False, indent=2), encoding="utf-8")
        sc_v1 = _extract_scale_entry(v1_data, "Qwen2.5-Coder-1.5B-Instruct")
        if sc_v1 is not None:
            scale_rows.append(sc_v1)

    imported_payloads: list[tuple[str, dict[str, Any]]] = []
    for inp in inputs:
        imported_payloads.extend(_collect_json_payloads_from_path(inp, import_dir))

    if not imported_payloads:
        raise ValueError("No valid `rl_eval_summary*.json` payloads found in provided inputs.")

    # Select primary v2 payload (prefer highest SFT+GRPO EX if multiple scales like 3B and 7B are imported)
    best_idx = 0
    best_ex = -1.0
    for idx, (stem, p) in enumerate(imported_payloads):
        sc = _extract_scale_entry(p, stem)
        if sc is not None:
            # Avoid duplicate scale rows
            if not any(r["scale_label"] == sc["scale_label"] for r in scale_rows):
                scale_rows.append(sc)
            if sc["grpo_ex"] >= best_ex:
                best_ex = sc["grpo_ex"]
                best_idx = idx

    _, primary = imported_payloads[best_idx]
    models_ordered = list((primary.get("models") or {}).items())
    base_short = primary.get("base_model_short") or Path(str(primary.get("base_model", "Qwen2.5-Coder-3B-Instruct"))).name
    test_count = int(primary.get("test_task_count", 30))

    ablation_block = primary.get("ablation_comparison") or {}
    report_md = render_rl_markdown_report(
        model_results=models_ordered,
        base_model_spec=base_short,
        test_task_count=test_count,
        sft_summary=primary.get("sft_training_summary"),
        grpo_summary=primary.get("grpo_training_summary"),
        grpo_sparse_summary=primary.get("grpo_sparse_training_summary"),
        vertical_comparison=primary.get("vertical_comparison"),
        ablation_test_metrics=ablation_block.get("test_metrics"),
        scale_comparison=scale_rows if len(scale_rows) >= 2 else None,
        case_studies=primary.get("case_studies"),
    )

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report_md, encoding="utf-8")

    merged_json = dict(primary)
    merged_json["scale_comparison"] = scale_rows
    merged_json["report_path"] = str(report_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(merged_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Colab A100 result archives/JSONs into local EvoDW-RL v2 Markdown & JSON reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[PROJECT_ROOT / "outputs" / "colab_imports"],
        help="Path(s) to Colab .tar.gz export, directory, or rl_eval_summary_*.json file(s).",
    )
    parser.add_argument(
        "--v1-baseline",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_eval" / "rl_eval_summary.json",
        help="Optional local v1 1.5B summary JSON for cross-scale progression table.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_experiment_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rl_eval" / "rl_eval_summary.json",
        help="Output merged JSON summary path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    merged = merge_colab_results(
        inputs=list(args.input),
        baseline_v1_json=args.v1_baseline if args.v1_baseline.exists() else None,
        report_out=args.report_out,
        json_out=args.json_out,
    )

    tbl = Table(title="Merged Colab + Local EvoDW-RL v2 Summary", header_style="bold cyan")
    tbl.add_column("Model / Variant")
    tbl.add_column("Test EX", justify="right")
    tbl.add_column("1st-Pass GW", justify="right")
    tbl.add_column("Invalid SQL", justify="right")
    for name, m in (merged.get("models") or {}).items():
        tbl.add_row(
            name,
            f"{float(m['ex']) * 100:.1f}%",
            f"{float(m['first_pass_gateway_rate']) * 100:.1f}%",
            f"{float(m['invalid_sql_rate']) * 100:.1f}%",
        )
    console.print(tbl)
    console.print(f"[bold green]✔ Updated Markdown report -> {args.report_out}[/bold green]")
    console.print(f"[bold green]✔ Updated JSON summary    -> {args.json_out}[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
