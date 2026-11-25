"""Evaluation comparators, metrics aggregation, and experiment report generation."""

from src.eval.comparator import compare_result_sets
from src.eval.manifest_audit import (
    analyze_sft_vs_grpo_dynamics,
    audit_experiment_manifest,
    build_experiment_manifest,
    exact_mcnemar_two_sided_p,
    wilson_confidence_interval,
)
from src.eval.metrics import CategoryMetrics, EvalMetrics, compute_eval_metrics
from src.eval.reporter import generate_experiment_report

__all__ = [
    "CategoryMetrics",
    "EvalMetrics",
    "analyze_sft_vs_grpo_dynamics",
    "audit_experiment_manifest",
    "build_experiment_manifest",
    "compare_result_sets",
    "compute_eval_metrics",
    "exact_mcnemar_two_sided_p",
    "generate_experiment_report",
    "wilson_confidence_interval",
]
