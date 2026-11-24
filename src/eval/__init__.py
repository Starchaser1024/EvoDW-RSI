"""Evaluation comparators, metrics aggregation, and experiment report generation."""

from src.eval.comparator import compare_result_sets
from src.eval.metrics import CategoryMetrics, EvalMetrics, compute_eval_metrics
from src.eval.reporter import generate_experiment_report

__all__ = [
    "CategoryMetrics",
    "EvalMetrics",
    "compare_result_sets",
    "compute_eval_metrics",
    "generate_experiment_report",
]
