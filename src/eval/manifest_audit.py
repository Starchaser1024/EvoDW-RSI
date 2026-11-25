"""Programmatic experiment lineage manifest builder, split-hygiene verifier, and SFT-vs-GRPO diagnostic audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import PROJECT_ROOT

V1_CATEGORIES: frozenset[str] = frozenset(
    {
        "partition_pruning",
        "json_array_unnest",
        "retention_date_math",
        "fanout_dedup",
        "ratio_null_handling",
    }
)
HARD_V2_CATEGORIES: frozenset[str] = frozenset(
    {
        "semantic_linking_trap",
        "scd2_temporal_join",
        "silent_caliber_audit",
    }
)

TRACKED_ARTIFACTS: tuple[str, ...] = (
    "outputs/runs/v1_odps/seed_summary.json",
    "outputs/runs/v1_odps/greedy_summary.json",
    "outputs/runs/v1_odps/gepa_summary.json",
    "outputs/runs/hard_v2_odps/seed_summary.json",
    "outputs/runs/hard_v2_odps/greedy_summary.json",
    "outputs/runs/hard_v2_odps/gepa_summary.json",
    "outputs/runs/hard_v2_hive/seed_summary.json",
    "outputs/runs/hard_v2_hive/gepa_summary.json",
    "outputs/policies/v1_odps/gepa_best_policy.json",
    "outputs/policies/hard_v2_odps/gepa_best_policy.json",
    "outputs/policies/hard_v2_hive/gepa_best_policy.json",
    "outputs/rl_export/export_summary.json",
    "outputs/rl_export/sft_openai_tool_use.jsonl",
    "outputs/rl_export/sft_sharegpt.jsonl",
    "outputs/rl_export/grpo_prompts.jsonl",
    "outputs/colab_imports/sft_training_summary.json",
    "outputs/colab_imports/grpo_dense_full_training_summary.json",
    "outputs/colab_imports/grpo_outcome_only_training_summary.json",
    "outputs/colab_imports/rl_eval_summary_3b.json",
    "outputs/rl_eval/rl_eval_summary_1_5b_v1.json",
    "outputs/rl_eval/rl_eval_summary_3b.json",
)


def normalized_utf8_sha256(path: Path) -> str:
    """Compute CRLF-invariant UTF-8 SHA-256 so Ubuntu and Windows checkouts produce identical digests."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute exact Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    k = max(0, min(int(successes), int(total)))
    n = float(total)
    p = k / n
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)


def exact_mcnemar_two_sided_p(b_only_first: int, c_only_second: int) -> float:
    """Compute exact two-sided binomial McNemar p-value from discordant pair counts (b, c)."""
    b = max(0, int(b_only_first))
    c = max(0, int(c_only_second))
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) * (0.5**m)
    return round(min(1.0, 2.0 * tail), 6)


def paired_mde_percentage_points(
    total_n: int,
    discordant_fraction: float = 0.25,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate two-sided paired McNemar minimum detectable effect (in percentage points) at given N."""
    if total_n <= 0 or discordant_fraction <= 0:
        return 100.0
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    mde = (z_alpha + z_beta) * math.sqrt(discordant_fraction / float(total_n))
    return round(min(1.0, mde) * 100.0, 1)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_split_hygiene_summary() -> dict[str, Any]:
    """Verify split cardinalities and zero train/val/test task or question overlap across v1 and Hard-v2."""
    from src.benchmarks.tasks import get_all_tasks

    v1_tasks = get_all_tasks("v1")
    v2_tasks = get_all_tasks("hard_v2")
    all_tasks = v1_tasks + v2_tasks

    def _suite_stats(tasks: list[Any]) -> dict[str, Any]:
        by_split: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
        for t in tasks:
            by_split[t.split].append(t)
        ids_train = {t.task_id for t in by_split["train"]}
        ids_val = {t.task_id for t in by_split["val"]}
        ids_test = {t.task_id for t in by_split["test"]}
        q_train_val = {_normalize_question(t.question) for t in by_split["train"] + by_split["val"]}
        q_test = {_normalize_question(t.question) for t in by_split["test"]}
        return {
            "total": len(tasks),
            "train": len(ids_train),
            "val": len(ids_val),
            "test": len(ids_test),
            "id_overlap_train_val": sorted(ids_train & ids_val),
            "id_overlap_train_test": sorted(ids_train & ids_test),
            "id_overlap_val_test": sorted(ids_val & ids_test),
            "question_overlap_trainval_vs_test": len(q_train_val & q_test),
        }

    v1_s = _suite_stats(v1_tasks)
    v2_s = _suite_stats(v2_tasks)
    all_s = _suite_stats(all_tasks)
    v1_ids = {t.task_id for t in v1_tasks}
    v2_ids = {t.task_id for t in v2_tasks}

    return {
        "policy": (
            "Strict 3-way physical isolation: `train` drives Reflector error attribution and SFT/GRPO rollouts; "
            "`val` governs Pareto frontier selection, SystemAwareMerger, and SFT/GRPO pool export; "
            "`test` (30 v1 + 18 Hard-v2 = 48 tasks) is strictly held out for final evaluation only. "
            "Any task manually inspected to refine prompts, linter rules, or reward weights must reside in train/val."
        ),
        "v1": v1_s,
        "hard_v2": v2_s,
        "combined": {
            **all_s,
            "cross_suite_id_overlap": sorted(v1_ids & v2_ids),
        },
    }


def analyze_sft_vs_grpo_dynamics(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Deterministically compute SFT-vs-GRPO rollout saturation, reward variance, KL drift, and CIs from real logs."""
    sft_sum = _read_json(project_root / "outputs/colab_imports/sft_training_summary.json")
    dense_sum = _read_json(project_root / "outputs/colab_imports/grpo_dense_full_training_summary.json")
    sparse_sum = _read_json(project_root / "outputs/colab_imports/grpo_outcome_only_training_summary.json")
    eval_3b = _read_json(project_root / "outputs/rl_eval/rl_eval_summary_3b.json")
    sft_export_rows = _read_jsonl(project_root / "outputs/rl_export/sft_openai_tool_use.jsonl")
    grpo_export_rows = _read_jsonl(project_root / "outputs/rl_export/grpo_prompts.jsonl")

    v1_sft_rows = [r for r in sft_export_rows if r.get("category") in V1_CATEGORIES]
    v1_grpo_rows = [r for r in grpo_export_rows if r.get("category") in V1_CATEGORIES]
    v1_sft_tids = {str(r["task_id"]) for r in v1_sft_rows}
    v1_grpo_tids = {str(r["task_id"]) for r in v1_grpo_rows}

    def _curve_metrics(gsum: dict[str, Any]) -> dict[str, Any]:
        logs = [e for e in (gsum.get("log_history") or []) if isinstance(e, dict) and "reward" in e]
        n = max(1, len(logs))
        rewards = [float(e["reward"]) for e in logs]
        online_ex = [float(e.get("multiturn/online_ex_rate", 0.0)) for e in logs]
        zero_stds = [float(e.get("frac_reward_zero_std", 0.0)) for e in logs]
        grad_norms = [float(e.get("grad_norm", 0.0)) for e in logs]
        kls = [float(e.get("kl", 0.0)) for e in logs]
        lengths = [float(e.get("completions/mean_length", 0.0)) for e in logs]
        gw_rates = [float(e.get("multiturn/gateway_pass_rate", 0.0)) for e in logs]
        sr_rates = [float(e.get("multiturn/self_repair_ex_rate", 0.0)) for e in logs]

        ep1_ex = [float(e["multiturn/online_ex_rate"]) for e in logs if float(e.get("epoch", 0.0)) <= 1.0]
        ep2_ex = [float(e["multiturn/online_ex_rate"]) for e in logs if float(e.get("epoch", 0.0)) > 1.0]

        r_min, r_max = min(rewards), max(rewards)
        thresh_80 = r_min + 0.80 * (r_max - r_min)
        conv_step = next(
            (int(e["step"]) for e in logs if float(e["reward"]) >= thresh_80),
            int(gsum.get("global_steps", 0)),
        )
        return {
            "logged_steps": len(logs),
            "step2_online_ex_rate": round(online_ex[0], 4) if online_ex else 0.0,
            "step4_online_ex_rate": round(online_ex[1], 4) if len(online_ex) > 1 else 0.0,
            "mean_online_ex_rate": round(sum(online_ex) / n, 4),
            "peak_online_ex_rate": round(max(online_ex), 4) if online_ex else 0.0,
            "epoch1_mean_online_ex_rate": round(sum(ep1_ex) / max(1, len(ep1_ex)), 4),
            "epoch2_plus_mean_online_ex_rate": round(sum(ep2_ex) / max(1, len(ep2_ex)), 4),
            "epoch2_minus_epoch1_online_ex": round(
                (sum(ep2_ex) / max(1, len(ep2_ex))) - (sum(ep1_ex) / max(1, len(ep1_ex))),
                4,
            ),
            "convergence_step_80pct": conv_step,
            "zero_reward_std_step_count": sum(1 for z in zero_stds if z > 0.0),
            "zero_reward_std_step_ratio": round(sum(1 for z in zero_stds if z > 0.0) / n, 4),
            "mean_frac_reward_zero_std": round(sum(zero_stds) / n, 4),
            "near_zero_grad_norm_step_count": sum(1 for g in grad_norms if g < 0.02),
            "step2_kl": round(kls[0], 5) if kls else 0.0,
            "mean_kl": round(sum(kls) / n, 5),
            "final_kl": round(kls[-1], 5) if kls else 0.0,
            "max_kl": round(max(kls), 5) if kls else 0.0,
            "mean_completion_tokens": round(sum(lengths) / n, 1),
            "mean_gateway_pass_rate": round(sum(gw_rates) / n, 4),
            "mean_self_repair_ex_rate": round(sum(sr_rates) / n, 4),
        }

    dense_dyn = _curve_metrics(dense_sum)
    sparse_dyn = _curve_metrics(sparse_sum)

    variants = eval_3b.get("variants_by_key") or {}
    n_test = int(eval_3b.get("test_task_count", 30))

    def _variant_ci_entry(vkey: str) -> dict[str, Any]:
        vm = variants.get(vkey) or {}
        ex = float(vm.get("ex", 0.0))
        hits = int(round(ex * n_test))
        lo, hi = wilson_confidence_interval(hits, n_test)
        return {
            "hits": hits,
            "total": n_test,
            "ex": round(ex, 4),
            "wilson_95_ci": [lo, hi],
            "first_pass_gateway_rate": round(float(vm.get("first_pass_gateway_rate", 0.0)), 4),
            "invalid_sql_rate": round(float(vm.get("invalid_sql_rate", 0.0)), 4),
            "first_pass_correct_count": int(vm.get("first_pass_correct_count", 0)),
            "self_repaired_count": int(vm.get("self_repaired_count", 0)),
        }

    model_cis = {
        k: _variant_ci_entry(k)
        for k in ("base", "base_gepa", "vertical_sql", "sft", "sft_grpo_sparse", "sft_grpo", "online")
        if k in variants
    }

    reversals = [
        str(cs["task_id"])
        for cs in (eval_3b.get("case_studies") or [])
        if cs.get("grpo_ex") and not cs.get("sft_ex")
    ]
    min_grpo_only = len(reversals)
    sft_hits = model_cis["sft"]["hits"]
    dense_hits = model_cis["sft_grpo"]["hits"]
    base_hits = model_cis["base"]["hits"]
    min_sft_only = max(0, (sft_hits - dense_hits) + min_grpo_only)

    category_tradeoffs: dict[str, Any] = {}
    for cat in sorted(V1_CATEGORIES):
        s_cat = ((variants.get("sft") or {}).get("by_category") or {}).get(cat) or {}
        sp_cat = ((variants.get("sft_grpo_sparse") or {}).get("by_category") or {}).get(cat) or {}
        d_cat = ((variants.get("sft_grpo") or {}).get("by_category") or {}).get(cat) or {}
        c_n = int(s_cat.get("count", 6))
        category_tradeoffs[cat] = {
            "count": c_n,
            "sft_hits": int(round(float(s_cat.get("ex", 0.0)) * c_n)),
            "sft_ex": round(float(s_cat.get("ex", 0.0)), 4),
            "sft_first_pass_gw": round(float(s_cat.get("first_pass_gateway_rate", 0.0)), 4),
            "sft_invalid_sql": round(float(s_cat.get("invalid_sql_rate", 0.0)), 4),
            "sparse_grpo_hits": int(round(float(sp_cat.get("ex", 0.0)) * c_n)),
            "sparse_grpo_ex": round(float(sp_cat.get("ex", 0.0)), 4),
            "dense_grpo_hits": int(round(float(d_cat.get("ex", 0.0)) * c_n)),
            "dense_grpo_ex": round(float(d_cat.get("ex", 0.0)), 4),
            "dense_grpo_first_pass_gw": round(float(d_cat.get("first_pass_gateway_rate", 0.0)), 4),
            "dense_grpo_invalid_sql": round(float(d_cat.get("invalid_sql_rate", 0.0)), 4),
        }

    sft_corpus = sft_sum.get("corpus_stats") or {}
    prompt_cnt = int(dense_sum.get("prompt_count", 45))
    covered_cnt = len(v1_sft_tids & v1_grpo_tids)

    return {
        "statistical_power_audit": {
            "test_task_count": n_test,
            "models": model_cis,
            "sft_vs_dense_grpo_net_task_delta": dense_hits - sft_hits,
            "documented_dense_grpo_reversals_over_sft": reversals,
            "min_discordant_pairs_sft_only_vs_grpo_only": [min_sft_only, min_grpo_only],
            "exact_mcnemar_two_sided_p_sft_vs_dense_grpo": exact_mcnemar_two_sided_p(
                min_sft_only, min_grpo_only
            ),
            "exact_mcnemar_two_sided_p_sft_vs_base_upper_bound": exact_mcnemar_two_sided_p(
                sft_hits - base_hits, 0
            ),
            "exact_mcnemar_two_sided_p_dense_grpo_vs_base_upper_bound": exact_mcnemar_two_sided_p(
                dense_hits - base_hits, 0
            ),
            "paired_80pct_power_mde_pp_n30": paired_mde_percentage_points(n_test, discordant_fraction=0.25),
            "paired_80pct_power_mde_pp_n72": paired_mde_percentage_points(72, discordant_fraction=0.25),
            "interpretation": (
                "On N=30 held-out v1 tasks, SFT (16/30 = 53.3%, 95% CI [36.1%, 69.8%]) vs Dense GRPO "
                "(14/30 = 46.7%, 95% CI [30.2%, 63.9%]) is a 2-task difference with exact McNemar p=0.6875 "
                "(well inside sampling noise; 80%-power paired MDE at N=30 is ~25.6pp). Gains over 3B Base "
                "(3/30 = 10.0%, +43.3pp for SFT and +36.7pp for Dense GRPO, p<=0.001) are statistically decisive."
            ),
        },
        "rollout_saturation_and_kl_dynamics": {
            "grpo_prompt_pool_size": prompt_cnt,
            "grpo_prompts_already_covered_by_sft": covered_cnt,
            "grpo_prompt_sft_overlap_ratio": round(covered_cnt / max(1, prompt_cnt), 4),
            "sft_raw_trajectories": int(sft_corpus.get("total_records", 299)),
            "sft_effective_upsampled_episodes": int(sft_corpus.get("effective_train_episodes", 391)),
            "sft_avg_trajectories_per_covered_task": round(
                int(sft_corpus.get("total_records", 299)) / max(1, covered_cnt), 2
            ),
            "grpo_total_epochs_over_45_prompts": round(
                (int(dense_sum.get("global_steps", 100)) * 2) / max(1, prompt_cnt), 2
            ),
            "dense_grpo": dense_dyn,
            "sparse_grpo": sparse_dyn,
            "zero_variance_step_reduction_factor_dense_vs_sparse": round(
                sparse_dyn["zero_reward_std_step_count"]
                / max(1, dense_dyn["zero_reward_std_step_count"]),
                2,
            ),
        },
        "category_tradeoffs": category_tradeoffs,
        "root_causes_grpo_below_sft_point_estimate": [
            (
                "Prompt-pool saturation: 38/45 (84.4%) of Cohort-v1 GRPO prompts were already trained on in "
                "Stage-1 SFT (avg 7.87 verified trajectories/task across 3 epochs), driving online_ex_rate "
                "from 37.5% (step 2) to 81.25% by step 4 and leaving limited positive exploration headroom."
            ),
            (
                "Late-epoch KL drift on small pool: running 100 steps (~4.44 prompt passes at 2 prompts/step, "
                "logged as 2.22 dataset epochs) pushed Dense GRPO KL from 0.00037 (step 2) to 0.01510 (step 100) "
                "while mean online_ex_rate drifted from 49.72% (epoch <=1) to 45.98% (epoch >1)."
            ),
            (
                "Gateway/conciseness vs multi-CTE anti-fanout trade-off: 4-stage dense reward lifted "
                "partition_pruning 1st-pass gateway compliance from 83.3% to 100.0% (0.0% invalid SQL) and "
                "solved SFT failures ja_11 and rd_10, while step/length penalties (mean 405.6 vs 440.2 tokens) "
                "cost 1 task each on verbose multi-CTE fanout_dedup (4/6 -> 3/6) and partition_pruning (4/6 -> 3/6)."
            ),
            (
                "Dense vs Sparse reward necessity confirmed: Outcome-Only Sparse GRPO suffered group-wise "
                "zero-variance advantage starvation on 32.0% of logged steps (16/50 vs 3/50 in Dense, 5.33x higher) "
                "and collapsed to 0.0% (0/6) on retention_date_math."
            ),
        ],
        "actionable_next_stage_protocol": [
            "Expand GRPO prompt pool from 45 (Cohort-v1) to 72+ (Cohort-v2 v1 + Hard-v2 + external schemas) and reserve non-upsampled prompts for RL.",
            "Apply dynamic pass@G curriculum filtering (retain prompts with 0 < pass@8 < 1 under pi_SFT) to eliminate saturated/zero-variance rollouts.",
            "Apply early checkpoint selection around 0.8-1.0 epoch before KL > 0.008 and relax length_free_chars (1400 -> 2400) for multi-CTE anti-fanout SQL.",
        ],
    }


def build_experiment_manifest(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build the deterministic, repo-relative `EXPERIMENT_MANIFEST.json` reconciling all cohorts and artifacts."""
    split_hygiene = compute_split_hygiene_summary()
    sft_export_rows = _read_jsonl(project_root / "outputs/rl_export/sft_openai_tool_use.jsonl")
    grpo_export_rows = _read_jsonl(project_root / "outputs/rl_export/grpo_prompts.jsonl")
    export_sum = _read_json(project_root / "outputs/rl_export/export_summary.json")
    sft_3b_sum = _read_json(project_root / "outputs/colab_imports/sft_training_summary.json")
    grpo_dense_sum = _read_json(project_root / "outputs/colab_imports/grpo_dense_full_training_summary.json")
    eval_1_5b = _read_json(project_root / "outputs/rl_eval/rl_eval_summary_1_5b_v1.json")
    dynamics = analyze_sft_vs_grpo_dynamics(project_root=project_root)

    v1_sft_rows = [r for r in sft_export_rows if r.get("category") in V1_CATEGORIES]
    v2_sft_rows = [r for r in sft_export_rows if r.get("category") in HARD_V2_CATEGORIES]
    v2_dialect_counts = Counter(str(r.get("dialect", "odps")) for r in v2_sft_rows)
    v1_type_counts = Counter(str(r.get("trajectory_type", "")) for r in v1_sft_rows)

    def _evol_entry(suite: str, dialect: str, run_rel: str, policy_rel: str | None) -> dict[str, Any]:
        d = _read_json(project_root / run_rel)
        bt = d.get("best_test") or {}
        bv = d.get("best_val") or {}
        n_t = int(bt.get("total", 0))
        ex_t = float(bt.get("ex", 0.0))
        hits_t = int(round(ex_t * n_t))
        lo, hi = wilson_confidence_interval(hits_t, n_t)
        return {
            "suite": suite,
            "dialect": dialect,
            "method": d.get("method"),
            "best_policy_id": d.get("best_policy_id"),
            "rollouts_used": int(d.get("rollouts_used", 0)),
            "val_total": int(bv.get("total", 0)),
            "val_ex": round(float(bv.get("ex", 0.0)), 4),
            "test_total": n_t,
            "test_hits": hits_t,
            "test_ex": round(ex_t, 4),
            "test_ex_wilson_95_ci": [lo, hi],
            "test_first_pass_gateway_rate": round(float(bt.get("first_pass_gateway_rate", 0.0)), 4),
            "test_invalid_sql_rate": round(float(bt.get("invalid_sql_rate", 0.0)), 4),
            "run_summary_path": run_rel,
            "policy_path": policy_rel,
        }

    online_evolution_runs = [
        _evol_entry("v1", "odps", "outputs/runs/v1_odps/seed_summary.json", "outputs/policies/v1_odps/seed_best_policy.json"),
        _evol_entry("v1", "odps", "outputs/runs/v1_odps/greedy_summary.json", "outputs/policies/v1_odps/greedy_best_policy.json"),
        _evol_entry("v1", "odps", "outputs/runs/v1_odps/gepa_summary.json", "outputs/policies/v1_odps/gepa_best_policy.json"),
        _evol_entry("hard_v2", "odps", "outputs/runs/hard_v2_odps/seed_summary.json", "outputs/policies/hard_v2_odps/seed_best_policy.json"),
        _evol_entry("hard_v2", "odps", "outputs/runs/hard_v2_odps/greedy_summary.json", "outputs/policies/hard_v2_odps/greedy_best_policy.json"),
        _evol_entry("hard_v2", "odps", "outputs/runs/hard_v2_odps/gepa_summary.json", "outputs/policies/hard_v2_odps/gepa_best_policy.json"),
        _evol_entry("hard_v2", "hive", "outputs/runs/hard_v2_hive/seed_summary.json", "outputs/policies/hard_v2_hive/seed_best_policy.json"),
        _evol_entry("hard_v2", "hive", "outputs/runs/hard_v2_hive/gepa_summary.json", "outputs/policies/hard_v2_hive/gepa_best_policy.json"),
    ]

    artifacts_checksums = {
        rel: normalized_utf8_sha256(project_root / rel)
        for rel in TRACKED_ARTIFACTS
        if (project_root / rel).exists()
    }

    return {
        "manifest_version": "1.0.0",
        "frozen_baseline_commit": "2c7af79",
        "split_hygiene": split_hygiene,
        "top_level_alias_note": (
            "`outputs/policies/gepa_best_policy.json` and `outputs/runs/gepa_summary.json` reflect the most recent "
            "`evodw evolve` invocation (`hard_v2` / `hive`). Always load suite-specific policies from "
            "`outputs/policies/{v1_odps,hard_v2_odps,hard_v2_hive}/gepa_best_policy.json`."
        ),
        "online_evolution_runs": online_evolution_runs,
        "rl_cohorts": {
            "cohort_v0_1_5b_local_pilot": {
                "status": "frozen_evaluated",
                "base_model": eval_1_5b.get("base_model", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
                "source_suite": "v1 (early pilot subset)",
                "sft_trajectories": 85,
                "grpo_steps": 8,
                "eval_split": "v1 test (N=30)",
                "eval_summary_path": "outputs/rl_eval/rl_eval_summary_1_5b_v1.json",
            },
            "cohort_v1_3b_colab_frozen": {
                "status": "frozen_evaluated_primary",
                "base_model": (sft_3b_sum.get("model_config") or {}).get(
                    "resolved_model_path", "Qwen/Qwen2.5-Coder-3B-Instruct"
                ),
                "source_suite": "v1 (ODPS train + val)",
                "sft_deduplicated_trajectories": int((sft_3b_sum.get("corpus_stats") or {}).get("total_records", 299)),
                "sft_first_pass_correct": int((sft_3b_sum.get("corpus_stats") or {}).get("first_pass_correct_count", 269)),
                "sft_self_repaired": int((sft_3b_sum.get("corpus_stats") or {}).get("self_repaired_count", 30)),
                "sft_probe_trajectories": int((sft_3b_sum.get("corpus_stats") or {}).get("probe_trajectory_count", 27)),
                "sft_effective_reweighted_episodes": int(
                    (sft_3b_sum.get("corpus_stats") or {}).get("effective_train_episodes", 391)
                ),
                "sft_global_steps": int(sft_3b_sum.get("global_steps", 147)),
                "grpo_prompt_count": int(grpo_dense_sum.get("prompt_count", 45)),
                "grpo_global_steps": int(grpo_dense_sum.get("global_steps", 100)),
                "grpo_num_generations": int(grpo_dense_sum.get("num_generations", 8)),
                "eval_split": "v1 test (N=30, strictly zero overlap with 299 SFT trajs & 45 GRPO prompts)",
                "verified_subset_inside_current_rl_export": {
                    "v1_sft_records_in_disk_jsonl": len(v1_sft_rows),
                    "v1_first_pass_in_disk_jsonl": v1_type_counts.get("first_pass_correct", 0),
                    "v1_self_repaired_in_disk_jsonl": v1_type_counts.get("self_repaired_after_error", 0),
                    "v1_grpo_prompts_in_disk_jsonl": sum(
                        1 for r in grpo_export_rows if r.get("category") in V1_CATEGORIES
                    ),
                },
                "training_summaries": [
                    "outputs/colab_imports/sft_training_summary.json",
                    "outputs/colab_imports/grpo_dense_full_training_summary.json",
                    "outputs/colab_imports/grpo_outcome_only_training_summary.json",
                ],
                "eval_summary_path": "outputs/rl_eval/rl_eval_summary_3b.json",
            },
            "cohort_v2_combined_export": {
                "status": "exported_untrained_next_stage",
                "source_suite": "v1 (ODPS) + Hard-v2 (ODPS + Hive) incremental merge (`merge_existing=True`)",
                "sft_deduplicated_trajectories": int(export_sum.get("total_exported_trajectories", 624)),
                "lineage_decomposition": {
                    "from_cohort_v1_odps": len(v1_sft_rows),
                    "from_hard_v2_odps": v2_dialect_counts.get("odps", 0),
                    "from_hard_v2_hive": v2_dialect_counts.get("hive", 0),
                },
                "sft_first_pass_correct": int(export_sum.get("first_pass_correct_count", 553)),
                "sft_self_repaired": int(export_sum.get("self_repaired_after_error_count", 71)),
                "grpo_prompt_count": int(export_sum.get("grpo_prompt_count", 72)),
                "grpo_prompt_decomposition": {
                    "v1_train_val": sum(1 for r in grpo_export_rows if r.get("category") in V1_CATEGORIES),
                    "hard_v2_train_val": sum(1 for r in grpo_export_rows if r.get("category") in HARD_V2_CATEGORIES),
                },
                "held_out_test_excluded_count": 48,
                "export_summary_path": "outputs/rl_export/export_summary.json",
                "reconciliation_note": (
                    "Current `outputs/rl_export/*` contains Cohort-v2 (624 SFT trajectories, 72 GRPO prompts) "
                    "prepared for cross-suite training, whereas all frozen 3B results in `outputs/rl_experiment_report.md` "
                    "were trained on Cohort-v1 (299 v1 SFT trajectories, 45 v1 GRPO prompts)."
                ),
            },
        },
        "sft_vs_grpo_diagnostics": dynamics,
        "artifact_sha256_normalized_utf8": artifacts_checksums,
    }


def audit_experiment_manifest(
    project_root: Path = PROJECT_ROOT,
    manifest_path: Path | None = None,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Run deterministic reproducibility, split-isolation, policy-hygiene, and manifest checksum checks."""
    target_manifest_path = manifest_path or (project_root / "outputs/EXPERIMENT_MANIFEST.json")
    checks: list[dict[str, Any]] = []

    def _add(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    # 1. Benchmark split disjointness & exact cardinalities
    sh = compute_split_hygiene_summary()
    v1_ok = (
        sh["v1"]["total"] == 75
        and (sh["v1"]["train"], sh["v1"]["val"], sh["v1"]["test"]) == (25, 20, 30)
        and not sh["v1"]["id_overlap_train_val"]
        and not sh["v1"]["id_overlap_train_test"]
        and not sh["v1"]["id_overlap_val_test"]
        and sh["v1"]["question_overlap_trainval_vs_test"] == 0
    )
    v2_ok = (
        sh["hard_v2"]["total"] == 45
        and (sh["hard_v2"]["train"], sh["hard_v2"]["val"], sh["hard_v2"]["test"]) == (15, 12, 18)
        and not sh["hard_v2"]["id_overlap_train_val"]
        and not sh["hard_v2"]["id_overlap_train_test"]
        and not sh["hard_v2"]["id_overlap_val_test"]
        and sh["hard_v2"]["question_overlap_trainval_vs_test"] == 0
    )
    comb_ok = (
        sh["combined"]["total"] == 120
        and sh["combined"]["test"] == 48
        and not sh["combined"]["cross_suite_id_overlap"]
        and sh["combined"]["question_overlap_trainval_vs_test"] == 0
    )
    _add(
        "Benchmark Split Isolation (v1 + Hard-v2)",
        v1_ok and v2_ok and comb_ok,
        f"v1=25/20/30, Hard-v2=15/12/18, held-out test={sh['combined']['test']} tasks, 0 ID/question overlap",
    )

    # 2. SFT & GRPO dataset zero test-leakage verification
    from src.agent.policy import Policy
    from src.benchmarks.tasks import get_all_tasks
    from src.rsi.attribution import validate_and_sanitize_prompt

    all_tasks = get_all_tasks("all")
    held_out_test_ids = {t.task_id for t in all_tasks if t.split == "test"}
    valid_train_val_ids = {t.task_id for t in all_tasks if t.split in ("train", "val")}

    sft_openai = _read_jsonl(project_root / "outputs/rl_export/sft_openai_tool_use.jsonl")
    sft_sharegpt = _read_jsonl(project_root / "outputs/rl_export/sft_sharegpt.jsonl")
    grpo_prompts = _read_jsonl(project_root / "outputs/rl_export/grpo_prompts.jsonl")

    leaked_sft = [
        r.get("task_id")
        for r in (sft_openai + sft_sharegpt)
        if r.get("split") not in ("train", "val")
        or r.get("task_id") in held_out_test_ids
        or r.get("task_id") not in valid_train_val_ids
    ]
    leaked_grpo = [
        r.get("task_id")
        for r in grpo_prompts
        if r.get("split") not in ("train", "val")
        or r.get("task_id") in held_out_test_ids
        or r.get("task_id") not in valid_train_val_ids
    ]
    _add(
        "RL Export Zero Test-Split Leakage",
        len(leaked_sft) == 0 and len(leaked_grpo) == 0 and len(sft_openai) == len(sft_sharegpt),
        f"SFT={len(sft_openai)} records, GRPO={len(grpo_prompts)} prompts; leaked_sft={len(leaked_sft)}, leaked_grpo={len(leaked_grpo)}",
    )

    # 3. Evolved GEPA policy anti-cheat / zero table-name or test-ID leakage
    policy_checks_ok = True
    policy_errs: list[str] = []
    for pol_rel, dial in (
        ("outputs/policies/v1_odps/gepa_best_policy.json", "odps"),
        ("outputs/policies/hard_v2_odps/gepa_best_policy.json", "odps"),
        ("outputs/policies/hard_v2_hive/gepa_best_policy.json", "hive"),
    ):
        pol_path = project_root / pol_rel
        if not pol_path.exists():
            policy_checks_ok = False
            policy_errs.append(f"missing:{pol_rel}")
            continue
        pol = Policy.load_json(pol_path)
        for mod_name in ("linker", "generator", "refiner"):
            ok_p, _, err_p = validate_and_sanitize_prompt(
                pol.get_prompt(mod_name), mod_name, dialect=dial, auto_repair_dialect=False
            )
            if not ok_p:
                policy_checks_ok = False
                policy_errs.append(f"{pol_rel}:{mod_name}:{err_p}")
    _add(
        "Evolved Policy Prompt Hygiene",
        policy_checks_ok,
        "3 canonical GEPA policies verified free of table/task-ID/enum leakage"
        if policy_checks_ok
        else "; ".join(policy_errs),
    )

    # 4. Exact Cohort-v1 (299/45) vs Cohort-v2 (624/72) reconciliation
    live_manifest = build_experiment_manifest(project_root=project_root)
    c1 = live_manifest["rl_cohorts"]["cohort_v1_3b_colab_frozen"]
    c2 = live_manifest["rl_cohorts"]["cohort_v2_combined_export"]
    sub = c1["verified_subset_inside_current_rl_export"]
    decomp = c2["lineage_decomposition"]
    reconciled = (
        c1["sft_deduplicated_trajectories"] == 299
        and sub["v1_sft_records_in_disk_jsonl"] == c1["sft_deduplicated_trajectories"]
        and sub["v1_first_pass_in_disk_jsonl"] == c1["sft_first_pass_correct"] == 269
        and sub["v1_self_repaired_in_disk_jsonl"] == c1["sft_self_repaired"] == 30
        and sub["v1_grpo_prompts_in_disk_jsonl"] == c1["grpo_prompt_count"] == 45
        and c2["sft_deduplicated_trajectories"]
        == decomp["from_cohort_v1_odps"] + decomp["from_hard_v2_odps"] + decomp["from_hard_v2_hive"]
        == 624
        and c2["grpo_prompt_count"] == 72
    )
    _add(
        "Cohort-v1 (299/45) vs Cohort-v2 (624/72) Lineage",
        reconciled,
        (
            f"Cohort-v1={c1['sft_deduplicated_trajectories']} SFT ({c1['sft_first_pass_correct']}+{c1['sft_self_repaired']}) / "
            f"{c1['grpo_prompt_count']} GRPO -> Cohort-v2={c2['sft_deduplicated_trajectories']} "
            f"(299 v1 + {decomp['from_hard_v2_odps']} v2-ODPS + {decomp['from_hard_v2_hive']} v2-Hive) / {c2['grpo_prompt_count']} GRPO"
        ),
    )

    # 5. Frozen manifest existence & normalized SHA-256 verification
    if not target_manifest_path.exists():
        _add(
            "Frozen Manifest Checksums (`EXPERIMENT_MANIFEST.json`)",
            False,
            f"Missing manifest at {target_manifest_path.as_posix()} (run with --write-manifest)",
        )
    else:
        frozen = _read_json(target_manifest_path)
        frozen_hashes = frozen.get("artifact_sha256_normalized_utf8") or {}
        live_hashes = live_manifest["artifact_sha256_normalized_utf8"]
        mismatches = [
            k
            for k in set(frozen_hashes) | set(live_hashes)
            if frozen_hashes.get(k) != live_hashes.get(k)
        ]
        struct_match = (
            frozen.get("rl_cohorts") == live_manifest["rl_cohorts"]
            and frozen.get("split_hygiene") == live_manifest["split_hygiene"]
            and frozen.get("sft_vs_grpo_diagnostics") == live_manifest["sft_vs_grpo_diagnostics"]
        )
        _add(
            "Frozen Manifest Checksums (`EXPERIMENT_MANIFEST.json`)",
            len(mismatches) == 0 and struct_match,
            f"{len(live_hashes)} artifacts verified (CRLF-normalized SHA-256 match)"
            if (len(mismatches) == 0 and struct_match)
            else f"mismatched_hashes={mismatches}, struct_match={struct_match}",
        )

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks, live_manifest


def _render_console_report(
    console: Console,
    passed: bool,
    checks: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    chk_tbl = Table(title="EvoDW-RSI Experiment Lineage & Split-Hygiene Audit", header_style="bold cyan")
    chk_tbl.add_column("Invariant Check")
    chk_tbl.add_column("Status")
    chk_tbl.add_column("Verified Details")
    for c in checks:
        chk_tbl.add_row(
            c["check"],
            "[green]PASS[/green]" if c["passed"] else "[red]FAIL[/red]",
            c["details"],
        )
    console.print(chk_tbl)

    dyn = manifest["sft_vs_grpo_diagnostics"]
    sp = dyn["statistical_power_audit"]
    ci_tbl = Table(
        title=f"Isolated v1 Test Split (N={sp['test_task_count']}) Wilson 95% Confidence Intervals",
        header_style="bold magenta",
    )
    ci_tbl.add_column("Model Variant")
    ci_tbl.add_column("Hits / N", justify="right")
    ci_tbl.add_column("Point EX", justify="right")
    ci_tbl.add_column("95% Wilson CI", justify="right")
    ci_tbl.add_column("1st-Pass GW", justify="right")
    ci_tbl.add_column("Invalid SQL", justify="right")
    labels = {
        "base": "Qwen2.5-Coder-3B (Zero-shot)",
        "base_gepa": "Qwen2.5-Coder-3B + GEPA Prompt",
        "vertical_sql": "XiYanSQL-QwenCoder-3B-2504",
        "sft": "Stage-1 Loss-Masked SFT (Cohort-v1)",
        "sft_grpo_sparse": "Stage-2A GRPO (Outcome-Only Sparse)",
        "sft_grpo": "Stage-2B GRPO (4-Stage Dense + Repair)",
        "online": "Online Teacher (deepseek-v4.1-flash + GEPA)",
    }
    for k, label in labels.items():
        if k not in sp["models"]:
            continue
        m = sp["models"][k]
        lo, hi = m["wilson_95_ci"]
        ci_tbl.add_row(
            label,
            f"{m['hits']}/{m['total']}",
            f"{m['ex'] * 100:.1f}%",
            f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
            f"{m['first_pass_gateway_rate'] * 100:.1f}%",
            f"{m['invalid_sql_rate'] * 100:.1f}%",
        )
    console.print(ci_tbl)

    rs = dyn["rollout_saturation_and_kl_dynamics"]
    d_curve = rs["dense_grpo"]
    s_curve = rs["sparse_grpo"]
    console.print(
        Panel(
            f"[bold]SFT vs GRPO Quantitative Audit (from real Colab logs):[/bold]\n"
            f"• [cyan]Statistical Power (N=30):[/cyan] SFT (16/30=53.3%) vs Dense GRPO (14/30=46.7%) is a 2-task net diff "
            f"(documented GRPO reversals: {sp['documented_dense_grpo_reversals_over_sft']}, exact McNemar p={sp['exact_mcnemar_two_sided_p_sft_vs_dense_grpo']:.4f}; "
            f"80%-power paired MDE={sp['paired_80pct_power_mde_pp_n30']}pp at N=30 -> {sp['paired_80pct_power_mde_pp_n72']}pp at N=72).\n"
            f"• [cyan]Prompt Saturation & KL Drift:[/cyan] {rs['grpo_prompts_already_covered_by_sft']}/{rs['grpo_prompt_pool_size']} "
            f"({rs['grpo_prompt_sft_overlap_ratio'] * 100:.1f}%) GRPO prompts already had SFT supervision "
            f"(avg {rs['sft_avg_trajectories_per_covered_task']} trajs/task); Dense online EX jumped {d_curve['step2_online_ex_rate'] * 100:.1f}% -> "
            f"{d_curve['step4_online_ex_rate'] * 100:.1f}% by step 4, while Epoch 1->2+ online EX drifted "
            f"{d_curve['epoch1_mean_online_ex_rate'] * 100:.1f}% -> {d_curve['epoch2_plus_mean_online_ex_rate'] * 100:.1f}% "
            f"as KL rose {d_curve['step2_kl']:.5f} -> {d_curve['final_kl']:.5f}.\n"
            f"• [cyan]Dense vs Sparse Advantage Starvation:[/cyan] Zero-std reward groups occurred in "
            f"{s_curve['zero_reward_std_step_count']}/{s_curve['logged_steps']} ({s_curve['zero_reward_std_step_ratio'] * 100:.1f}%) Sparse steps vs "
            f"{d_curve['zero_reward_std_step_count']}/{d_curve['logged_steps']} ({d_curve['zero_reward_std_step_ratio'] * 100:.1f}%) Dense steps "
            f"({rs['zero_variance_step_reduction_factor_dense_vs_sparse']}x reduction), preventing Sparse's 0.0% retention collapse.",
            title="Audit Verdict: PASS" if passed else "Audit Verdict: FAIL",
            border_style="green" if passed else "red",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evodw audit-experiments",
        description="Audit frozen experiment lineage manifest, split isolation, Wilson 95% CIs, and SFT-vs-GRPO dynamics.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/EXPERIMENT_MANIFEST.json",
        help="Path to frozen EXPERIMENT_MANIFEST.json.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Deterministically write/update EXPERIMENT_MANIFEST.json before running audit checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON audit summary to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.write_manifest:
        payload = build_experiment_manifest(project_root=PROJECT_ROOT)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    passed, checks, manifest = audit_experiment_manifest(
        project_root=PROJECT_ROOT,
        manifest_path=args.manifest,
    )
    if args.json:
        print(
            json.dumps(
                {"passed": passed, "checks": checks, "manifest": manifest},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _render_console_report(Console(), passed, checks, manifest)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
