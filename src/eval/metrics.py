"""Comprehensive evaluation metrics across accuracy, gateway compliance, categories, and cost."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.trajectory import AgentTrajectory


@dataclass
class CategoryMetrics:
    """Per-category evaluation breakdown."""

    category: str
    count: int
    ex: float
    first_pass_gateway_rate: float
    invalid_sql_rate: float


@dataclass
class EvalMetrics:
    """Full evaluation metrics across a split of trajectories."""

    total: int
    ex: float
    first_pass_gateway_rate: float
    invalid_sql_rate: float
    avg_steps: float
    avg_refine_turns: float
    avg_tokens: float
    avg_latency_ms: float
    first_pass_correct_count: int
    self_repaired_count: int
    by_category: dict[str, CategoryMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ex": round(self.ex, 4),
            "first_pass_gateway_rate": round(self.first_pass_gateway_rate, 4),
            "invalid_sql_rate": round(self.invalid_sql_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_refine_turns": round(self.avg_refine_turns, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "first_pass_correct_count": self.first_pass_correct_count,
            "self_repaired_count": self.self_repaired_count,
            "by_category": {
                cat: {
                    "count": cm.count,
                    "ex": round(cm.ex, 4),
                    "first_pass_gateway_rate": round(cm.first_pass_gateway_rate, 4),
                    "invalid_sql_rate": round(cm.invalid_sql_rate, 4),
                }
                for cat, cm in self.by_category.items()
            },
        }


def compute_eval_metrics(trajectories: list[AgentTrajectory]) -> EvalMetrics:
    """Compute aggregate and per-category evaluation metrics from trajectories."""
    n = len(trajectories)
    if n == 0:
        return EvalMetrics(
            total=0,
            ex=0.0,
            first_pass_gateway_rate=0.0,
            invalid_sql_rate=0.0,
            avg_steps=0.0,
            avg_refine_turns=0.0,
            avg_tokens=0.0,
            avg_latency_ms=0.0,
            first_pass_correct_count=0,
            self_repaired_count=0,
            by_category={},
        )

    ex_hits = sum(1 for t in trajectories if t.ex_match)
    gw1_hits = sum(1 for t in trajectories if t.first_pass_gateway)
    invalid_hits = sum(
        1
        for t in trajectories
        if (not t.final_gateway_passed or t.final_error_stage != "none")
    )
    total_steps = sum(t.num_llm_steps for t in trajectories)
    total_refine = sum(t.refine_turns for t in trajectories)
    total_tokens = sum(t.total_tokens for t in trajectories)
    total_latency = sum(t.total_latency_ms for t in trajectories)
    first_pass_correct = sum(1 for t in trajectories if t.outcome_label == "first_pass_correct")
    self_repaired = sum(1 for t in trajectories if t.outcome_label == "self_repaired_after_error")

    grouped: dict[str, list[AgentTrajectory]] = defaultdict(list)
    for t in trajectories:
        grouped[t.category].append(t)

    by_cat: dict[str, CategoryMetrics] = {}
    for cat in sorted(grouped.keys()):
        items = grouped[cat]
        cn = len(items)
        by_cat[cat] = CategoryMetrics(
            category=cat,
            count=cn,
            ex=sum(1 for t in items if t.ex_match) / cn,
            first_pass_gateway_rate=sum(1 for t in items if t.first_pass_gateway) / cn,
            invalid_sql_rate=sum(
                1
                for t in items
                if (not t.final_gateway_passed or t.final_error_stage != "none")
            )
            / cn,
        )

    return EvalMetrics(
        total=n,
        ex=ex_hits / n,
        first_pass_gateway_rate=gw1_hits / n,
        invalid_sql_rate=invalid_hits / n,
        avg_steps=total_steps / n,
        avg_refine_turns=total_refine / n,
        avg_tokens=total_tokens / n,
        avg_latency_ms=total_latency / n,
        first_pass_correct_count=first_pass_correct,
        self_repaired_count=self_repaired,
        by_category=by_cat,
    )
