"""Offline tests for DualRoleLLMClient, GEPA Reflector/Pareto/Merger, Report, and RL/SFT Bridge."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.agent import Policy
from src.benchmarks import load_benchmark
from src.config import PROJECT_ROOT, Settings
from src.eval import generate_experiment_report
from src.export import export_sft_and_grpo_datasets, reward_fn
from src.llm import DualRoleLLMClient, MissingAPIKeyError, MockWarehouseLLMClient, OpenCodeGoClient
from src.rsi import (
    EvolutionEngine,
    ParetoArchive,
    Reflector,
    SystemAwareMerger,
    detect_prompt_dialect_violations,
    repair_prompt_dialect_violations,
    sanitize_rule_text,
    validate_and_sanitize_prompt,
)


class TestDualRoleLLMClient:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_on_live_call(self):
        assert OpenCodeGoClient is DualRoleLLMClient
        client = DualRoleLLMClient(api_key="", base_url="https://api.openai.com/v1")
        with pytest.raises(MissingAPIKeyError):
            await client.chat_json([{"role": "user", "content": "hi"}], role_type="actor")

    @pytest.mark.asyncio
    async def test_standard_openai_compatible_client_and_settings(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-standard-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        cfg = Settings.from_env()
        assert cfg.api_key == "test-standard-key"
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.has_usable_api_key is True

        client = DualRoleLLMClient(settings=cfg)
        assert client._client is not None
        assert "x-opencode-session" not in client._client.default_headers
        await client.close()

    @pytest.mark.asyncio
    async def test_session_header_backoff_and_json_truncation_recovery(self):
        client = OpenCodeGoClient(
            api_key="dummy-test-key",
            base_url="https://opencode.ai/zen/go/v1",
            actor_model="deepseek-v4.1-flash",
            reflector_model="deepseek-v4-pro",
            base_backoff_sec=0.01,
            max_retries=3,
            session_id="test-session-uuid-1234",
        )
        assert client._client is not None
        assert client._client.default_headers.get("x-opencode-session") == "test-session-uuid-1234"

        seen_calls = []
        fake_req = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions")
        fake_resp_429 = httpx.Response(429, request=fake_req)

        async def fake_create(**kwargs):
            seen_calls.append(kwargs)
            idx = len(seen_calls)
            if idx == 1:
                raise openai.RateLimitError("rate limited", response=fake_resp_429, body=None)
            if idx == 2:
                # Truncated JSON at length limit.
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"thought": "halfway cut'),
                            finish_reason="length",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='```json\n{"thought": "ok", "sql": "SELECT 1"}\n```'),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
            )

        client._client.chat.completions.create = fake_create  # type: ignore[method-assign]
        resp = await client.chat_json(
            [{"role": "user", "content": "test"}],
            role_type="actor",
            max_tokens=100,
        )

        assert len(seen_calls) == 3
        for call in seen_calls:
            assert call["extra_headers"]["x-opencode-session"] == "test-session-uuid-1234"
        assert seen_calls[2]["max_tokens"] == 150
        assert resp.parsed_json == {"thought": "ok", "sql": "SELECT 1"}
        assert resp.prompt_tokens == 220
        assert resp.completion_tokens == 80
        assert resp.retries == 2
        assert client.ledger.actor.calls == 1
        await client.close()


class TestReflectorAntiCheatingAndPareto:
    def test_anti_cheating_prompt_validator(self):
        good = (
            "You are DialectSQLGenerator. Always filter partitioned tables on ds or dt using MAX_PT. "
            'Output JSON {"thought": "...", "sql": "..."}.'
        )
        ok, _, err = validate_and_sanitize_prompt(good, "generator")
        assert ok and err == ""

        leaked_table = good + " Remember to query dwd_trade_order_di."
        ok_t, _, err_t = validate_and_sanitize_prompt(leaked_table, "generator")
        assert not ok_t and "dwd_trade_order_di" in err_t

        leaked_date = good + " Use ds = '20261015'."
        ok_d, _, _ = validate_and_sanitize_prompt(leaked_date, "generator")
        assert not ok_d

        broken_contract = "You are generator. Always filter ds using MAX_PT and emit plain text only."
        ok_c, _, _ = validate_and_sanitize_prompt(broken_contract, "generator")
        assert not ok_c

        sanitized = sanitize_rule_text("Always filter dwd_trade_order_di on ds = '20261015' for pp_01.")
        assert "dwd_trade_order_di" not in sanitized
        assert "20261015" not in sanitized
        assert "pp_01" not in sanitized


    def test_dialect_consistency_guard_and_auto_repair(self):
        # Negative warnings must NOT be flagged as violations.
        clean_with_warnings = (
            "You are DialectSQLGenerator. "
            "`DATE_ADD` is unsupported and will cause an error; use `DATEADD(dt, 1, 'dd')`. "
            "Use `GET_JSON_OBJECT(col, '$.k')` instead of `JSON_EXTRACT`. "
            "Replace `IFNULL` with `NVL`, and never use `UNNEST`. "
            'Output JSON {"thought": "...", "sql": "..."}.'
        )
        assert detect_prompt_dialect_violations(clean_with_warnings, dialect="odps") == []
        ok_clean, out_clean, msg_clean = validate_and_sanitize_prompt(clean_with_warnings, "generator", dialect="odps")
        assert ok_clean and msg_clean == "" and out_clean == clean_with_warnings

        # Exact gepa_g3 Hive DATE_FORMAT + other forbidden ODPS functions must be caught and auto-repaired!
        poisoned = (
            "You are DialectSQLGenerator. "
            "To compare a shifted date to a partition key formatted as yyyyMMdd, "
            "use `DATE_FORMAT(DATEADD(TO_DATE(date_string, 'yyyy-mm-dd'), N, 'dd'), 'yyyyMMdd')`. "
            "Also compute `DATE_ADD(d, 7)`, `IFNULL(x, 0)`, `JSON_EXTRACT(ext, '$.p')`, "
            "`STRFTIME('%Y%m%d', ts)`, and `CROSS JOIN UNNEST(arr)`. "
            'Output strictly valid JSON: {"thought": "...", "sql": "..."}.'
        )
        viols = detect_prompt_dialect_violations(poisoned, dialect="odps")
        for expected_fn in ("DATE_FORMAT", "DATE_ADD", "IFNULL", "JSON_EXTRACT", "STRFTIME", "UNNEST"):
            assert expected_fn in viols

        ok_strict, _, err_strict = validate_and_sanitize_prompt(
            poisoned, "generator", dialect="odps", auto_repair_dialect=False
        )
        assert not ok_strict and "DATE_FORMAT" in err_strict

        ok_rep, repaired, msg_rep = validate_and_sanitize_prompt(
            poisoned, "generator", dialect="odps", auto_repair_dialect=True
        )
        assert ok_rep
        assert "Auto-repaired ODPS dialect violation" in msg_rep
        assert "TO_CHAR(DATEADD(TO_DATE(date_string, 'yyyy-mm-dd'), N, 'dd'), 'yyyymmdd')" in repaired
        assert "DATEADD(d, 7, 'dd')" in repaired
        assert "NVL(x, 0)" in repaired
        assert "GET_JSON_OBJECT(ext, '$.p')" in repaired
        assert "LATERAL VIEW explode(arr)" in repaired
        assert detect_prompt_dialect_violations(repaired, dialect="odps") == []

        long_neg = (
            "- NULL handling: use `NVL` or `COALESCE`, not `IFNULL`.\n"
            "- Do NOT use Hive/MySQL/DuckDB functions such as `DATE_FORMAT`, `DATE_ADD`, `DATE_SUB`, "
            "`IFNULL`, `JSON_EXTRACT`, `STRFTIME`, `UNNEST`, or 2-argument `DATEDIFF`."
        )
        assert detect_prompt_dialect_violations(long_neg, dialect="odps") == []
        rep_neg, rep_neg_fns = repair_prompt_dialect_violations(long_neg, dialect="odps")
        assert rep_neg == long_neg and rep_neg_fns == []

    def test_pareto_frontier_and_win_frequency_weights(self):
        archive = ParetoArchive()
        seed = Policy.load_seed(PROJECT_ROOT / "prompts" / "seed")
        p_a = seed.clone_with_mutation(new_policy_id="cand_a", target_module="generator", new_prompt=seed.generator_prompt, generation=1)
        p_b = seed.clone_with_mutation(new_policy_id="cand_b", target_module="linker", new_prompt=seed.linker_prompt, generation=1)
        p_dom = seed.clone_with_mutation(new_policy_id="cand_dom", target_module="refiner", new_prompt=seed.refiner_prompt, generation=2)

        def _mk_traj(tid: str, cat: str, ex: bool, gw: bool = True):
            return SimpleNamespace(
                task_id=tid,
                category=cat,
                ex_match=ex,
                first_pass_gateway=gw,
                final_gateway_passed=gw,
            )

        archive.add_candidate(
            p_a,
            [_mk_traj("t1", "partition_pruning", True), _mk_traj("t2", "json_array_unnest", False)],
        )
        archive.add_candidate(
            p_b,
            [_mk_traj("t1", "partition_pruning", False), _mk_traj("t2", "json_array_unnest", True)],
        )
        archive.add_candidate(
            p_dom,
            [_mk_traj("t1", "partition_pruning", False), _mk_traj("t2", "json_array_unnest", False)],
        )

        frontier = archive.get_pareto_frontier()
        assert set(frontier) == {"cand_a", "cand_b"}
        weights = archive.get_win_frequency_weights()
        assert pytest.approx(weights["cand_a"], abs=1e-6) == 0.5
        assert pytest.approx(weights["cand_b"], abs=1e-6) == 0.5

        rng = random.Random(42)
        sampled = archive.sample_parent(rng)
        assert sampled.policy_id in ("cand_a", "cand_b")


class TestEndToEndMockEvolutionAndRLBridge:
    @pytest.mark.asyncio
    async def test_gepa_evolution_ablation_report_and_rl_export(self, tmp_path):
        bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True)
        try:
            seed_policy = Policy.load_seed(Settings.from_env().prompts_dir)
            llm = MockWarehouseLLMClient()

            greedy_engine = EvolutionEngine(llm=llm, benchmark=bm, seed_policy=seed_policy, dialect="odps")
            greedy_res = await greedy_engine.run(method="greedy", budget=60)

            gepa_engine = EvolutionEngine(llm=llm, benchmark=bm, seed_policy=seed_policy, dialect="odps")
            gepa_res = await gepa_engine.run(method="gepa", budget=60)

            # Verify GEPA improved significantly over seed on test EX and 1st-pass gateway rate.
            assert gepa_res.best_test_metrics.ex > gepa_res.seed_test_metrics.ex
            assert gepa_res.best_test_metrics.first_pass_gateway_rate > gepa_res.seed_test_metrics.first_pass_gateway_rate
            assert gepa_res.best_test_metrics.ex >= greedy_res.best_test_metrics.ex
            assert gepa_res.rollouts_used <= 60
            assert gepa_res.effective_val_size >= 8
            assert gepa_res.cache_hits > 0
            assert any(h.operation == "merge" for h in gepa_res.history)

            # Verify experiment report creation.
            report_path = tmp_path / "experiment_report.md"
            generate_experiment_report(
                gepa_res,
                output_path=report_path,
                runs_dir=tmp_path / "runs",
                ablation_results=[greedy_res, gepa_res],
            )
            report_text = report_path.read_text(encoding="utf-8")
            assert "## 2. 消融对比表" in report_text
            assert "## 3. 五大数仓高频错题分项表现" in report_text
            assert "```diff" in report_text

            # Verify RL/SFT dataset export & loss masks.
            all_verified = greedy_res.verified_train_val_trajectories + gepa_res.verified_train_val_trajectories
            rl_stats = export_sft_and_grpo_datasets(
                trajectories=all_verified,
                tasks=bm.all_tasks,
                output_dir=tmp_path / "rl_export",
            )
            assert rl_stats["total_exported_trajectories"] > 0
            assert rl_stats["first_pass_correct_count"] > 0
            assert rl_stats["self_repaired_after_error_count"] > 0
            assert rl_stats["grpo_prompt_count"] == 45  # 25 train + 20 val (test excluded!)

            openai_lines = (tmp_path / "rl_export" / "sft_openai_tool_use.jsonl").read_text(encoding="utf-8").splitlines()
            first_obj = json.loads(openai_lines[0])
            for msg in first_obj["messages"]:
                if msg["role"] == "assistant":
                    assert msg["loss_mask"] == 1
                else:
                    assert msg["loss_mask"] == 0

            # Verify reward_fn standalone scoring.
            t_sample = bm.train[0]
            r_gold = reward_fn(
                {"task_id": t_sample.task_id, "gold_sql": t_sample.gold_sql},
                json.dumps({"thought": "ok", "sql": t_sample.gold_sql}),
                sandbox=bm.sandbox,
                dialect="odps",
            )
            r_bad_gw = reward_fn(
                {"task_id": t_sample.task_id},
                json.dumps({"thought": "no partition", "sql": "SELECT COUNT(*) FROM dwd_trade_order_di"}),
                sandbox=bm.sandbox,
                dialect="odps",
            )
            assert r_gold == pytest.approx(1.1, abs=1e-4)
            assert r_bad_gw == pytest.approx(0.0, abs=1e-4)
        finally:
            bm.sandbox.close()


class TestUnifiedCLI:
    def test_cli_doctor_and_help(self):
        from src.cli import main as cli_main

        assert cli_main(["--help"]) == 0
        assert cli_main(["doctor"]) == 0
        assert cli_main(["config-check"]) == 0
        assert cli_main(["smoke-test", "--mock"]) == 0
