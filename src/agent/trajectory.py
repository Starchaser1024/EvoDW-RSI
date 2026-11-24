"""Structured execution trajectory capturing multi-module calls, tool probes, and gateway repairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.env.sandbox import EvalOutcome


@dataclass
class ModuleInvocation:
    """Single module call record inside an agent episode."""

    module_name: str  # linker | generator | refiner
    step_index: int
    messages: list[dict[str, str]]
    raw_output: str
    parsed_json: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


@dataclass
class AgentTrajectory:
    """Complete episode trace across Linker, Generator, Sandbox, and Refiner."""

    task_id: str
    category: str
    split: str
    question: str
    hint: str
    dialect: str
    policy_id: str
    linked_tables: list[str] = field(default_factory=list)
    probe_requests: list[dict[str, Any]] = field(default_factory=list)
    probe_results: list[dict[str, Any]] = field(default_factory=list)
    linked_schema_context: str = ""
    linker_notes: str = ""
    initial_sql: str = ""
    first_pass_gateway: bool = False
    first_pass_exec: dict[str, Any] = field(default_factory=dict)
    refine_steps: list[dict[str, Any]] = field(default_factory=list)
    final_sql: str = ""
    final_exec: dict[str, Any] = field(default_factory=dict)
    eval_outcome: EvalOutcome | None = None
    reward_info: dict[str, float] = field(default_factory=dict)
    module_calls: list[ModuleInvocation] = field(default_factory=list)
    conversation_turns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_llm_steps(self) -> int:
        return len(self.module_calls)

    @property
    def refine_turns(self) -> int:
        return len(self.refine_steps)

    @property
    def total_tokens(self) -> int:
        return sum(m.total_tokens for m in self.module_calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(m.latency_ms for m in self.module_calls) + float(
            self.final_exec.get("latency_ms", 0.0)
        )

    @property
    def ex_match(self) -> bool:
        return bool(self.eval_outcome and self.eval_outcome.ex_match)

    @property
    def final_gateway_passed(self) -> bool:
        return bool(self.final_exec.get("gateway_passed", False))

    @property
    def final_error_stage(self) -> str:
        return str(self.final_exec.get("error_stage", "none"))

    @property
    def outcome_label(self) -> str:
        if not self.ex_match:
            return "failed"
        if self.refine_turns == 0 and self.first_pass_gateway:
            return "first_pass_correct"
        return "self_repaired_after_error"

    def compact_summary_for_reflector(self) -> dict[str, Any]:
        """Compact JSON-serializable trace for offline Reflector error attribution."""
        eval_info = {}
        if self.eval_outcome is not None:
            eval_info = {
                "ex_match": self.eval_outcome.ex_match,
                "diff_reason": self.eval_outcome.diff_reason,
                "pred_sample_rows": [str(r) for r in self.eval_outcome.pred_rows[:5]],
                "gold_sample_rows": [str(r) for r in self.eval_outcome.gold_rows[:5]],
            }
        return {
            "task_id": self.task_id,
            "category": self.category,
            "dialect": self.dialect,
            "question": self.question,
            "hint": self.hint,
            "linked_tables": self.linked_tables,
            "probe_tables": [p.get("table_name") for p in self.probe_requests],
            "linker_notes": self.linker_notes,
            "initial_sql": self.initial_sql,
            "first_pass_gateway": self.first_pass_gateway,
            "first_pass_error_stage": self.first_pass_exec.get("error_stage", "none"),
            "first_pass_error_msg": self.first_pass_exec.get("error_msg", ""),
            "refine_steps": [
                {
                    "turn": step["turn"],
                    "trigger_reason": step["trigger_reason"],
                    "diagnosis": step["diagnosis"],
                    "sql": step["sql"],
                    "error_stage": step["exec_res"].get("error_stage", "none"),
                    "error_msg": step["exec_res"].get("error_msg", ""),
                    "row_count": step["exec_res"].get("row_count", 0),
                }
                for step in self.refine_steps
            ],
            "final_sql": self.final_sql,
            "final_error_stage": self.final_error_stage,
            "final_error_msg": self.final_exec.get("error_msg", ""),
            "final_row_count": self.final_exec.get("row_count", 0),
            "evaluation": eval_info,
        }

    def to_dict(self) -> dict[str, Any]:
        eval_dict = None
        if self.eval_outcome is not None:
            eval_dict = {
                "ex_match": self.eval_outcome.ex_match,
                "gateway_passed": self.eval_outcome.gateway_passed,
                "error_stage": self.eval_outcome.error_stage,
                "error_msg": self.eval_outcome.error_msg,
                "pred_rows": [list(r) for r in self.eval_outcome.pred_rows[:50]],
                "pred_columns": list(self.eval_outcome.pred_columns),
                "gold_rows": [list(r) for r in self.eval_outcome.gold_rows[:50]],
                "gold_columns": list(self.eval_outcome.gold_columns),
                "latency_ms": self.eval_outcome.latency_ms,
                "transpiled_sql": self.eval_outcome.transpiled_sql,
                "diff_reason": self.eval_outcome.diff_reason,
            }
        return {
            "task_id": self.task_id,
            "category": self.category,
            "split": self.split,
            "question": self.question,
            "hint": self.hint,
            "dialect": self.dialect,
            "policy_id": self.policy_id,
            "linked_tables": self.linked_tables,
            "probe_requests": self.probe_requests,
            "probe_results": self.probe_results,
            "linked_schema_context": self.linked_schema_context,
            "linker_notes": self.linker_notes,
            "initial_sql": self.initial_sql,
            "first_pass_gateway": self.first_pass_gateway,
            "first_pass_exec": self.first_pass_exec,
            "refine_steps": self.refine_steps,
            "final_sql": self.final_sql,
            "final_exec": self.final_exec,
            "eval_outcome": eval_dict,
            "reward_info": self.reward_info,
            "module_calls": [asdict(m) for m in self.module_calls],
            "conversation_turns": self.conversation_turns,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        override_policy_id: str | None = None,
    ) -> "AgentTrajectory":
        from src.env.sandbox import EvalOutcome

        eo_raw = data.get("eval_outcome")
        eo = None
        if isinstance(eo_raw, dict):
            eo = EvalOutcome(
                ex_match=bool(eo_raw.get("ex_match", False)),
                gateway_passed=bool(eo_raw.get("gateway_passed", False)),
                error_stage=str(eo_raw.get("error_stage", "none")),
                error_msg=str(eo_raw.get("error_msg", "")),
                pred_rows=[tuple(r) for r in eo_raw.get("pred_rows", [])],
                pred_columns=list(eo_raw.get("pred_columns", [])),
                gold_rows=[tuple(r) for r in eo_raw.get("gold_rows", [])],
                gold_columns=list(eo_raw.get("gold_columns", [])),
                latency_ms=float(eo_raw.get("latency_ms", 0.0)),
                transpiled_sql=str(eo_raw.get("transpiled_sql", "")),
                diff_reason=str(eo_raw.get("diff_reason", "")),
            )
        pid = override_policy_id or str(data["policy_id"])
        turns = [dict(t) for t in data.get("conversation_turns", [])]
        if turns and turns[0].get("role") == "system" and override_policy_id:
            turns[0]["content"] = re.sub(
                r"^\[Policy=[^|\]]+",
                f"[Policy={override_policy_id} ",
                str(turns[0].get("content", "")),
            )
        calls = [
            ModuleInvocation(
                module_name=str(m["module_name"]),
                step_index=int(m["step_index"]),
                messages=[dict(msg) for msg in m.get("messages", [])],
                raw_output=str(m.get("raw_output", "")),
                parsed_json=dict(m.get("parsed_json") or {}),
                prompt_tokens=int(m.get("prompt_tokens", 0)),
                completion_tokens=int(m.get("completion_tokens", 0)),
                total_tokens=int(m.get("total_tokens", 0)),
                latency_ms=float(m.get("latency_ms", 0.0)),
            )
            for m in data.get("module_calls", [])
        ]
        return cls(
            task_id=str(data["task_id"]),
            category=str(data["category"]),
            split=str(data["split"]),
            question=str(data["question"]),
            hint=str(data.get("hint", "")),
            dialect=str(data["dialect"]),
            policy_id=pid,
            linked_tables=list(data.get("linked_tables", [])),
            probe_requests=list(data.get("probe_requests", [])),
            probe_results=list(data.get("probe_results", [])),
            linked_schema_context=str(data.get("linked_schema_context", "")),
            linker_notes=str(data.get("linker_notes", "")),
            initial_sql=str(data.get("initial_sql", "")),
            first_pass_gateway=bool(data.get("first_pass_gateway", False)),
            first_pass_exec=dict(data.get("first_pass_exec", {})),
            refine_steps=list(data.get("refine_steps", [])),
            final_sql=str(data.get("final_sql", "")),
            final_exec=dict(data.get("final_exec", {})),
            eval_outcome=eo,
            reward_info={k: float(v) for k, v in dict(data.get("reward_info", {})).items()},
            module_calls=calls,
            conversation_turns=turns,
        )

