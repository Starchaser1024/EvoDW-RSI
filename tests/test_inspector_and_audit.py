"""Unit tests for frozen experiment manifest audit, statistical CIs, SFT-vs-GRPO diagnostics,
and SQL Trace Inspector (`Executable` vs `Result Correct / EX` dual-verdict workbench)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import urllib.error
import urllib.request

import pytest

from src.cli import main as cli_main
from src.config import PROJECT_ROOT
from src.eval.manifest_audit import (
    analyze_sft_vs_grpo_dynamics,
    audit_experiment_manifest,
    build_experiment_manifest,
    exact_mcnemar_two_sided_p,
    normalized_utf8_sha256,
    scan_outputs_for_host_path_leaks,
    wilson_confidence_interval,
)
from src.inspector.trace_inspector import (
    build_showcase_traces,
    create_inspector_http_server,
    render_inspector_html,
    render_inspector_markdown,
    summarize_inspected_traces,
)


class TestExperimentManifestAndAudit:
    def test_wilson_ci_and_mcnemar_exact_math(self):
        lo_sft, hi_sft = wilson_confidence_interval(16, 30)
        lo_grpo, hi_grpo = wilson_confidence_interval(14, 30)
        assert (lo_sft, hi_sft) == pytest.approx((0.3614, 0.6977), abs=1e-3)
        assert (lo_grpo, hi_grpo) == pytest.approx((0.3023, 0.6386), abs=1e-3)
        assert lo_sft < hi_grpo and lo_grpo < hi_sft

        assert wilson_confidence_interval(0, 30)[0] == 0.0
        assert wilson_confidence_interval(30, 30)[1] == 1.0

        assert exact_mcnemar_two_sided_p(4, 2) == pytest.approx(0.6875, abs=1e-4)
        assert exact_mcnemar_two_sided_p(13, 0) < 0.001

    def test_frozen_manifest_passes_all_invariants_and_crlf_hash_invariance(self, tmp_path: Path):
        passed, checks, manifest = audit_experiment_manifest(project_root=PROJECT_ROOT)
        assert passed is True, f"Audit failed: {checks}"
        assert len(checks) == 6
        assert all(c["passed"] for c in checks)

        c1 = manifest["rl_cohorts"]["cohort_v1_3b_colab_frozen"]
        c2 = manifest["rl_cohorts"]["cohort_v2_combined_export"]
        assert c1["sft_deduplicated_trajectories"] == 299
        assert c1["grpo_prompt_count"] == 45
        assert c2["sft_deduplicated_trajectories"] == 624
        assert c2["grpo_prompt_count"] == 72
        assert c2["held_out_test_excluded_count"] == 48

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

    def test_zero_host_or_colab_absolute_paths_in_tracked_outputs(self, tmp_path: Path):
        scanned_n, leaks = scan_outputs_for_host_path_leaks(project_root=PROJECT_ROOT)
        assert scanned_n >= 20
        assert leaks == []

        for jf in sorted((PROJECT_ROOT / "outputs").rglob("*.json")):
            txt = jf.read_text(encoding="utf-8")
            for forbidden in ("/home/", "/Users/", "/content/", "C:\\Users", "C:\\\\Users", "C:/Users"):
                assert forbidden not in txt, f"Leaked host path {forbidden!r} in {jf}"

        # Verify detector catches leaked host/Colab/Windows paths in synthetic outputs tree
        fake_out = tmp_path / "outputs" / "colab_imports"
        fake_out.mkdir(parents=True, exist_ok=True)
        (fake_out / "leaked.json").write_text(
            json.dumps({"adapter_dir": "/content/colab_workspace/ckpt", "win": "C:\\Users\\dev\\out"}),
            encoding="utf-8",
        )
        _, detected = scan_outputs_for_host_path_leaks(project_root=tmp_path)
        assert len(detected) == 1
        assert "outputs/colab_imports/leaked.json" in detected[0]

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


class TestSQLTraceInspectorDualVerdict:
    @pytest.mark.asyncio
    async def test_multi_suite_dual_verdict_silent_traps_gateway_blocks_and_zero_cdn_html(self):
        traces = await build_showcase_traces(
            dialect="odps",
            use_mock=True,
            limit=14,
            project_root=PROJECT_ROOT,
        )
        assert len(traces) >= 10
        suites_seen = {t.suite for t in traces}
        assert {"v1", "hard_v2", "external_bird_mini"}.issubset(suites_seen)

        summary = summarize_inspected_traces(traces)
        # Must expose all 4 core verdict classes cleanly:
        # 1) silent_caliber_trap (Executable=PASS, EX=FAIL),
        # 2) gateway_or_syntax_block (Executable=FAIL),
        # 3) self_repaired_pass, and
        # 4) first_pass_pass
        by_code = summary["by_verdict_code"]
        assert by_code.get("silent_caliber_trap", 0) >= 3
        assert by_code.get("gateway_or_syntax_block", 0) >= 1
        assert by_code.get("self_repaired_pass", 0) >= 1
        assert by_code.get("first_pass_pass", 0) >= 3
        assert summary["executable_rate"] > summary["result_correct_rate"]

        # Verify silent caliber trap invariant: Executable is True while Result Correct is False!
        silent_traces = [t for t in traces if t.dual_verdict["is_silent_caliber_trap"]]
        for st in silent_traces:
            assert st.dual_verdict["executable"] is True
            assert st.dual_verdict["result_correct"] is False
            assert st.dual_verdict["final_row_count"] > 0

        # Verify probed facts & selected tables exist on GEPA / SFT probed traces
        assert any(len(t.linker["discovered_facts"]) > 0 for t in traces)

        html_doc = render_inspector_html(traces)
        assert "语法与网关可执行 (Executable)" in html_doc
        assert "业务口径结果正确 (Result Correct / EX)" in html_doc
        assert "静默口径陷阱" in html_doc
        # Strictly zero external CDN dependencies
        assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', html_doc, flags=re.IGNORECASE)

        md_doc = render_inspector_markdown(traces)
        assert "Silent Caliber Traps" in md_doc
        assert "```sql" in md_doc

    @pytest.mark.asyncio
    async def test_sft_jsonl_trace_reconstruction_and_http_server(self):
        sft_path = PROJECT_ROOT / "outputs/rl_export/sft_openai_tool_use.jsonl"
        traces = await build_showcase_traces(
            from_jsonl=sft_path,
            limit=4,
            project_root=PROJECT_ROOT,
        )
        assert len(traces) == 4
        assert all(t.dual_verdict["executable"] is True for t in traces)
        assert all(t.dual_verdict["result_correct"] is True for t in traces)
        assert any(len(t.linker["probe_requests"]) > 0 for t in traces)
        assert any(t.refinement["refine_turns"] > 0 and "--- sql_before" in t.refinement["rounds"][0]["unified_diff"] for t in traces)

        server = create_inspector_http_server(traces, host="127.0.0.1", port=0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp_html:
                assert resp_html.status == 200
                body_html = resp_html.read().decode("utf-8")
                assert "EvoDW-RSI SQL &amp; Multi-Module Trajectory Inspector" in body_html

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/traces.json", timeout=5) as resp_json:
                assert resp_json.status == 200
                payload = json.loads(resp_json.read().decode("utf-8"))
                assert payload["summary"]["total_traces"] == 4
                assert len(payload["traces"]) == 4

            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nonexistent", timeout=5)
            assert exc_info.value.code == 404
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_cli_inspect_outputs_to_temp_files_without_polluting_repo(self, tmp_path: Path):
        html_out = tmp_path / "workbench.html"
        md_out = tmp_path / "traces.md"
        json_out = tmp_path / "traces.json"

        rc = cli_main(
            [
                "inspect",
                "--suite",
                "external_bird_mini",
                "--tasks",
                "ext_fin_03",
                "ext_edu_05",
                "--policy",
                "both",
                "--mock",
                "--html-out",
                str(html_out),
                "--md-out",
                str(md_out),
                "--json-out",
                str(json_out),
            ]
        )
        assert rc == 0
        assert html_out.exists() and md_out.exists() and json_out.exists()
        saved = json.loads(json_out.read_text(encoding="utf-8"))
        assert saved["summary"]["total_traces"] == 4
        # Under seed_v0 both tasks hit silent caliber traps; under gepa_g5 both pass EX!
        assert saved["summary"]["silent_caliber_trap_count"] == 2
        assert saved["summary"]["result_correct_count"] == 2
