"""Unit tests for frozen experiment manifest audit, statistical CIs, and SFT-vs-GRPO diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import main as cli_main
from src.config import PROJECT_ROOT
from src.eval.manifest_audit import (
    analyze_sft_vs_grpo_dynamics,
    audit_experiment_manifest,
    build_experiment_manifest,
    exact_mcnemar_two_sided_p,
    normalized_utf8_sha256,
    wilson_confidence_interval,
)


class TestExperimentManifestAndAudit:
    def test_wilson_ci_and_mcnemar_exact_math(self):
        # 16/30 = 53.33% -> [0.3614, 0.6977]; 14/30 = 46.67% -> [0.3023, 0.6386]
        lo_sft, hi_sft = wilson_confidence_interval(16, 30)
        lo_grpo, hi_grpo = wilson_confidence_interval(14, 30)
        assert (lo_sft, hi_sft) == pytest.approx((0.3614, 0.6977), abs=1e-3)
        assert (lo_grpo, hi_grpo) == pytest.approx((0.3023, 0.6386), abs=1e-3)
        assert lo_sft < hi_grpo and lo_grpo < hi_sft  # CIs overlap heavily at N=30

        # Boundary cases: 0/30 and 30/30
        assert wilson_confidence_interval(0, 30)[0] == 0.0
        assert wilson_confidence_interval(30, 30)[1] == 1.0

        # Exact two-sided McNemar test
        assert exact_mcnemar_two_sided_p(4, 2) == pytest.approx(0.6875, abs=1e-4)
        assert exact_mcnemar_two_sided_p(13, 0) < 0.001

    def test_frozen_manifest_passes_all_invariants_and_crlf_hash_invariance(self, tmp_path: Path):
        passed, checks, manifest = audit_experiment_manifest(project_root=PROJECT_ROOT)
        assert passed is True, f"Audit failed: {checks}"
        assert len(checks) == 5
        assert all(c["passed"] for c in checks)

        # Verify Cohort-v1 (299/45) vs Cohort-v2 (624/72) exact reconciliation
        c1 = manifest["rl_cohorts"]["cohort_v1_3b_colab_frozen"]
        c2 = manifest["rl_cohorts"]["cohort_v2_combined_export"]
        assert c1["sft_deduplicated_trajectories"] == 299
        assert c1["grpo_prompt_count"] == 45
        assert c2["sft_deduplicated_trajectories"] == 624
        assert c2["grpo_prompt_count"] == 72
        assert c2["held_out_test_excluded_count"] == 48

        # Verify CRLF vs LF normalization produces identical SHA-256 across Linux & Windows
        lf_file = tmp_path / "lf.json"
        crlf_file = tmp_path / "crlf.json"
        lf_file.write_bytes(b'{"a": 1,\n"b": 2}\n')
        crlf_file.write_bytes(b'{"a": 1,\r\n"b": 2}\r\n')
        assert normalized_utf8_sha256(lf_file) == normalized_utf8_sha256(crlf_file)

    def test_tampered_manifest_detected_and_cli_audit_subcommand(self, tmp_path: Path):
        tampered = build_experiment_manifest(project_root=PROJECT_ROOT)
        tampered["artifact_sha256_normalized_utf8"]["outputs/rl_export/export_summary.json"] = "0" * 64
        bad_manifest_path = tmp_path / "TAMPERED_MANIFEST.json"
        bad_manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

        passed, checks, _ = audit_experiment_manifest(
            project_root=PROJECT_ROOT, manifest_path=bad_manifest_path
        )
        assert passed is False
        assert any(not c["passed"] and "Checksums" in c["check"] for c in checks)

        assert cli_main(["audit-experiments"]) == 0

    def test_sft_vs_grpo_quantitative_log_diagnostics(self):
        dyn = analyze_sft_vs_grpo_dynamics(project_root=PROJECT_ROOT)
        rs = dyn["rollout_saturation_and_kl_dynamics"]
        assert rs["grpo_prompt_pool_size"] == 45
        assert rs["grpo_prompts_already_covered_by_sft"] == 38
        assert rs["dense_grpo"]["step4_online_ex_rate"] == pytest.approx(0.8125, abs=1e-4)
        assert rs["sparse_grpo"]["zero_reward_std_step_count"] == 16
        assert rs["dense_grpo"]["zero_reward_std_step_count"] == 3
        assert rs["zero_variance_step_reduction_factor_dense_vs_sparse"] > 5.0
        assert dyn["statistical_power_audit"]["documented_dense_grpo_reversals_over_sft"] == ["ja_11", "rd_10"]
