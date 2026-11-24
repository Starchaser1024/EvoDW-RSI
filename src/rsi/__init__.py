"""GEPA reflective strategy self-evolution engine."""

from src.rsi.attribution import (
    FORBIDDEN_TABLE_NAMES,
    ODPS_FORBIDDEN_FUNCTION_SPECS,
    Reflector,
    detect_prompt_dialect_violations,
    repair_prompt_dialect_violations,
    sanitize_rule_text,
    validate_and_sanitize_prompt,
)
from src.rsi.engine import EvolutionEngine, EvolutionMethod, EvolutionRunResult, GenerationLog
from src.rsi.merger import SystemAwareMerger, find_lowest_common_ancestor
from src.rsi.pareto import CandidateRecord, ParetoArchive, score_trajectory_for_pareto

__all__ = [
    "CandidateRecord",
    "EvolutionEngine",
    "EvolutionMethod",
    "EvolutionRunResult",
    "FORBIDDEN_TABLE_NAMES",
    "GenerationLog",
    "ODPS_FORBIDDEN_FUNCTION_SPECS",
    "ParetoArchive",
    "Reflector",
    "SystemAwareMerger",
    "detect_prompt_dialect_violations",
    "find_lowest_common_ancestor",
    "repair_prompt_dialect_violations",
    "sanitize_rule_text",
    "score_trajectory_for_pareto",
    "validate_and_sanitize_prompt",
]
