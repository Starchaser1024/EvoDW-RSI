"""Evaluation comparators, metrics aggregation, manifest audit, and fixed-budget baseline harness."""

from __future__ import annotations

from typing import Any

from src.eval.comparator import compare_result_sets
from src.eval.metrics import CategoryMetrics, EvalMetrics, compute_eval_metrics
from src.eval.reporter import generate_experiment_report

__all__ = [
    "ALL_BASELINE_METHODS",
    "CategoryMetrics",
    "EvalMetrics",
    "analyze_sft_vs_grpo_dynamics",
    "audit_experiment_manifest",
    "build_experiment_manifest",
    "classify_trajectory_verdict",
    "compare_methods_fixed_budget",
    "compare_result_sets",
    "compute_eval_metrics",
    "exact_mcnemar_two_sided_p",
    "generate_experiment_report",
    "verify_zero_cross_schema_leakage",
    "wilson_confidence_interval",
]


def __getattr__(name: str) -> Any:
    if name in {
        "analyze_sft_vs_grpo_dynamics",
        "audit_experiment_manifest",
        "build_experiment_manifest",
        "exact_mcnemar_two_sided_p",
        "wilson_confidence_interval",
    }:
        from src.eval import manifest_audit

        return getattr(manifest_audit, name)
    if name in {
        "ALL_BASELINE_METHODS",
        "classify_trajectory_verdict",
        "compare_methods_fixed_budget",
        "verify_zero_cross_schema_leakage",
    }:
        from src.eval import fixed_budget

        return getattr(fixed_budget, name)
    raise AttributeError(f"module 'src.eval' has no attribute {name!r}")
