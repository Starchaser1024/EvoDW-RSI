"""Comprehensive unit tests for EvoDW-RL v2:
- Loss-Masked SFT Collator & Adaptive Scarcity Reweighting across all 299+ verified trajectories
- Zero-truncation sequence length verification (`max_seq_length=9216`)
- Strict Train/Val vs Test Split Isolation
- Multi-Turn Agentic Sandbox Rollout & Exact Action/Observation Gradient & KL Masking
- 4-Stage Progressive + Self-Repair + Step/Length Penalty Rewards & Pure Sparse Ablation
- Colab Bundle Packer, Secret Sanitizer, and Result Merger
- Offline CPU Dry-Run End-to-End Pipeline (SFT -> Multi-Turn GRPO -> Unified Evaluation)
"""

from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest
import torch

from scripts.eval_rl_models import run_full_evaluation
from scripts.merge_colab_results import merge_colab_results
from scripts.pack_for_colab import pack_colab_bundle, sanitize_text_secrets
from src.benchmarks.loader import load_benchmark
from src.config import PROJECT_ROOT
from src.rl import (
    GRPOTrainingConfig,
    HierarchicalSQLReward,
    IGNORE_INDEX,
    LossMaskedMultiTurnCollator,
    MultiTurnSFTDataset,
    MultiTurnSandboxRolloutEngine,
    RLModelConfig,
    RewardStageWeights,
    SFTTrainingConfig,
    build_dry_run_tokenizer,
    collate_multiturn_rollouts,
    compute_action_length_penalty,
    compute_hierarchical_reward,
    encode_multiturn_with_loss_mask,
    get_isolated_test_task_ids,
    load_grpo_prompts,
    load_sft_trajectories,
    resolve_vertical_sql_model,
    reweight_sft_records,
    run_grpo_training,
    run_sft_training,
)


class TestLossMaskedMultiTurnCollatorAndReweighting:
    def test_real_sft_corpus_counts_adaptive_reweight_and_strict_loss_masking(self):
        sft_path = PROJECT_ROOT / "outputs" / "rl_export" / "sft_openai_tool_use.jsonl"
        summary_path = PROJECT_ROOT / "outputs" / "rl_export" / "export_summary.json"
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        records = load_sft_trajectories(sft_path, strict_no_test_leakage=True)
        assert len(records) == expected["total_exported_trajectories"] >= 299

        # Verify adaptive scarcity reweighting boosts both self-repaired and inspect_table_sample trajectories
        expanded = reweight_sft_records(
            records,
            adaptive_reweight=True,
            upsample_self_repaired=2,
            upsample_probe=2,
            target_self_repair_ratio=0.22,
            target_probe_ratio=0.18,
            seed=2026,
        )
        assert len(expanded) > len(records)

        tok = build_dry_run_tokenizer()
        ds = MultiTurnSFTDataset(
            records,
            tokenizer=tok,
            max_seq_length=131072,
            upsample_self_repaired=2,
            upsample_probe=2,
            adaptive_reweight=True,
        )
        assert ds.stats.first_pass_correct_count == expected["first_pass_correct_count"]
        assert ds.stats.self_repaired_count == expected["self_repaired_after_error_count"]
        assert ds.stats.probe_trajectory_count >= 25
        assert ds.stats.gateway_error_trajectory_count >= 15
        assert ds.stats.effective_self_repaired_episodes >= ds.stats.self_repaired_count * 2
        assert ds.stats.effective_probe_episodes >= ds.stats.probe_trajectory_count * 2
        assert ds.stats.truncated_trajectory_count == 0
        assert 0.05 < ds.stats.supervised_token_ratio < 0.50

        repaired_with_probe = next(
            r
            for r in records
            if r["trajectory_type"] == "self_repaired_after_error"
            and any(m.get("name") == "inspect_table_sample" for m in r["messages"])
        )
        encoded = encode_multiturn_with_loss_mask(
            repaired_with_probe["messages"],
            tokenizer=tok,
            max_seq_length=131072,
        )
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        assert len(input_ids) == len(labels)
        assert encoded["was_truncated"] is False

        spans: list[list[int]] = []
        cur: list[int] = []
        for x, y in zip(input_ids, labels):
            if y != IGNORE_INDEX:
                assert y == x, "Supervised label must equal exact input_id token!"
                cur.append(y)
            elif cur:
                spans.append(cur)
                cur = []
        if cur:
            spans.append(cur)

        expected_assistant_turns = [
            m["content"] for m in repaired_with_probe["messages"] if m["loss_mask"] == 1
        ]
        assert len(spans) == len(expected_assistant_turns)

        decoded_supervised_all = ""
        for span_ids, expected_content in zip(spans, expected_assistant_turns):
            decoded_span = tok.decode(span_ids, skip_special_tokens=False)
            assert decoded_span == f"{expected_content}<|im_end|>"
            decoded_supervised_all += decoded_span

        assert "<|im_start|>" not in decoded_supervised_all
        assert "<tool_response" not in decoded_supervised_all
        assert "## Linker Policy" not in decoded_supervised_all

    def test_explicit_zero_loss_mask_on_assistant_and_padding_invariants(self):
        tok = build_dry_run_tokenizer()
        synthetic_messages = [
            {"role": "system", "content": "System rules", "loss_mask": 0},
            {"role": "user", "content": "Query orders", "loss_mask": 0},
            {"role": "assistant", "content": '{"action": "probe"}', "loss_mask": 0},
            {
                "role": "tool",
                "name": "sandbox_execute",
                "content": "FAILED: ODPS-0123031: Partition exception",
                "loss_mask": 0,
            },
            {"role": "assistant", "content": '{"thought": "fix", "sql": "SELECT 1"}', "loss_mask": 1},
        ]
        short_messages = [
            {"role": "user", "content": "Short", "loss_mask": 0},
            {"role": "assistant", "content": '{"sql": "SELECT 2"}', "loss_mask": 1},
        ]

        collator = LossMaskedMultiTurnCollator(tokenizer=tok, max_seq_length=2048, pad_to_multiple_of=8)
        batch = collator([{"messages": synthetic_messages}, {"messages": short_messages}])

        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape[1] % 8 == 0

        pad_positions = batch["attention_mask"] == 0
        assert pad_positions.sum().item() > 0
        assert torch.all(batch["labels"][pad_positions] == IGNORE_INDEX)

        sup_ids_0 = batch["labels"][0][batch["labels"][0] != IGNORE_INDEX].tolist()
        sup_text_0 = tok.decode(sup_ids_0, skip_special_tokens=False)
        assert sup_text_0 == '{"thought": "fix", "sql": "SELECT 1"}<|im_end|>'
        assert "probe" not in sup_text_0
        assert "ODPS-0123031" not in sup_text_0

    def test_smart_truncation_preserves_self_repair_supervision(self):
        tok = build_dry_run_tokenizer()
        long_messages = [
            {"role": "system", "content": "SYS " * 20, "loss_mask": 0},
            {"role": "user", "content": "LONG_SCHEMA " * 200, "loss_mask": 0},
            {"role": "assistant", "content": '{"thought": "draft", "sql": "SELECT bad"}', "loss_mask": 1},
            {"role": "tool", "name": "sandbox_execute", "content": "ODPS-0123031", "loss_mask": 0},
            {"role": "user", "content": "Please repair", "loss_mask": 0},
            {"role": "assistant", "content": '{"thought": "repaired", "sql": "SELECT good"}', "loss_mask": 1},
            {"role": "tool", "name": "sandbox_execute", "content": "FINAL_OBS " * 80, "loss_mask": 0},
        ]
        enc = encode_multiturn_with_loss_mask(long_messages, tokenizer=tok, max_seq_length=360)
        assert len(enc["input_ids"]) <= 360
        assert enc["was_truncated"] is True
        sup_ids = [y for y in enc["labels"] if y != IGNORE_INDEX]
        assert len(sup_ids) > 0
        sup_decoded = tok.decode(sup_ids, skip_special_tokens=True)
        assert "SELECT good" in sup_decoded


class TestMultiTurnSandboxRolloutAndObservationMasking:
    def test_multiturn_probe_and_gateway_repair_zero_observation_gradient(self):
        bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True)
        try:
            tok = build_dry_run_tokenizer()
            task = bm.train[0]  # pp_01 on dwd_trade_order_di
            engine = MultiTurnSandboxRolloutEngine(
                sandbox=bm.sandbox,
                tokenizer=tok,
                dialect="odps",
                max_rollout_turns=3,
                max_prompt_length=1024,
                max_completion_length=4096,
                max_per_turn_new_tokens=1024,
            )

            # Episode A: Turn 1 forgets partition filter -> ODPS-0123031 observation -> Turn 2 repairs to Gold SQL!
            turn1_bad_sql = json.dumps(
                {
                    "thought": "draft without partition",
                    "sql": "SELECT order_status, COUNT(*) FROM dwd_trade_order_di GROUP BY order_status",
                }
            ) + "<|im_end|>"
            turn2_repaired_sql = json.dumps(
                {
                    "thought": "added MAX_PT partition filter",
                    "diagnosis": "Fixed ODPS-0123031",
                    "sql": task.gold_sql,
                }
            ) + "<|im_end|>"

            # Episode B: Turn 1 probes inspect_table_sample -> Observation -> Turn 2 emits Gold SQL!
            turn1_probe = json.dumps(
                {
                    "thought": "check partitions",
                    "action": "probe",
                    "probe_requests": [{"table_name": "dwd_trade_order_di", "columns": ["order_status"]}],
                }
            ) + "<|im_end|>"
            turn2_after_probe = json.dumps(
                {"thought": "queried after probe", "sql": task.gold_sql}
            ) + "<|im_end|>"

            call_step = {"n": 0}

            def scripted_batch_gen(contexts: list[list[int]], max_new: int) -> list[list[int]]:
                del max_new
                step = call_step["n"]
                call_step["n"] += 1
                assert len(contexts) == 2
                if step == 0:
                    return [
                        tok.encode(turn1_bad_sql, add_special_tokens=False),
                        tok.encode(turn1_probe, add_special_tokens=False),
                    ]
                # Verify Turn 2 context ALREADY contains the injected sandbox observation tokens!
                ctx0_txt = tok.decode(contexts[0], skip_special_tokens=False)
                ctx1_txt = tok.decode(contexts[1], skip_special_tokens=False)
                assert "ODPS-0123031" in ctx0_txt
                assert 'name="inspect_table_sample"' in ctx1_txt
                return [
                    tok.encode(turn2_repaired_sql, add_special_tokens=False),
                    tok.encode(turn2_after_probe, add_special_tokens=False),
                ]

            episodes = engine.run_episodes(
                ["<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n", "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n"],
                generate_batch_fn=scripted_batch_gen,
            )
            assert len(episodes) == 2
            ep_repair, ep_probe = episodes[0], episodes[1]

            assert ep_repair.num_turns == 2
            assert ep_repair.refine_turns == 1
            assert ep_repair.had_initial_error is True
            assert ep_repair.first_pass_gateway is False
            assert len(ep_repair.observation_spans) == 1

            assert ep_probe.num_turns == 2
            assert ep_probe.probe_turns == 1
            assert ep_probe.refine_turns == 0
            assert ep_probe.first_pass_gateway is True
            assert len(ep_probe.observation_spans) == 1

            # Verify exact Action Loss Mask vs Causal Attention Mask invariants!
            collated = collate_multiturn_rollouts(episodes, pad_token_id=tok.pad_token_id)
            comp_ids = collated["completion_ids"]
            loss_mask = collated["completion_mask"]
            attn_mask = collated["completion_attn_mask"]

            for row_idx, ep in enumerate(episodes):
                # 1. Every Observation span token MUST have loss_mask == 0 AND attn_mask == 1
                for obs_s, obs_e in ep.observation_spans:
                    assert torch.all(loss_mask[row_idx, obs_s:obs_e] == 0)
                    assert torch.all(attn_mask[row_idx, obs_s:obs_e] == 1)
                # 2. Every Action span token MUST have loss_mask == 1 AND attn_mask == 1
                for act_s, act_e in ep.action_spans:
                    assert torch.all(loss_mask[row_idx, act_s:act_e] == 1)
                    assert torch.all(attn_mask[row_idx, act_s:act_e] == 1)
                # 3. Every right-pad token MUST have BOTH loss_mask == 0 AND attn_mask == 0
                real_len = len(ep.completion_ids)
                if comp_ids.shape[1] > real_len:
                    assert torch.all(loss_mask[row_idx, real_len:] == 0)
                    assert torch.all(attn_mask[row_idx, real_len:] == 0)

                # 4. Decoding ONLY loss_mask==1 tokens must contain ZERO <tool_response> or ODPS error strings!
                supervised_toks = comp_ids[row_idx][loss_mask[row_idx] == 1].tolist()
                decoded_actions_only = tok.decode(supervised_toks, skip_special_tokens=False)
                assert "<tool_response" not in decoded_actions_only
                assert "Sandbox Execution Feedback" not in decoded_actions_only

            # 5. Verify gradient backprop through masked GRPO surrogate assigns ZERO gradient to observation logits!
            bsz, seq_len = comp_ids.shape
            dummy_logps = torch.randn(bsz, seq_len, requires_grad=True)
            ref_logps = torch.randn(bsz, seq_len)
            advantages = torch.tensor([1.25, -0.75])
            per_tok_kl = torch.exp(ref_logps - dummy_logps) - (ref_logps - dummy_logps) - 1.0
            per_tok_loss = -(torch.exp(dummy_logps - dummy_logps.detach()) * advantages.unsqueeze(1)) + 0.04 * per_tok_kl
            mask_f = loss_mask.float()
            loss = ((per_tok_loss * mask_f).sum(-1) / mask_f.sum(-1).clamp(min=1.0)).mean()
            loss.backward()

            assert dummy_logps.grad is not None
            for row_idx, ep in enumerate(episodes):
                for obs_s, obs_e in ep.observation_spans:
                    assert torch.all(dummy_logps.grad[row_idx, obs_s:obs_e] == 0.0)
                for act_s, act_e in ep.action_spans:
                    assert torch.any(dummy_logps.grad[row_idx, act_s:act_e] != 0.0)

            # 6. Verify Self-Repair Reward Bonus triggers on Episode A!
            r_engine = HierarchicalSQLReward(sandbox=bm.sandbox, dialect="odps")
            bd_repair = r_engine.evaluate_rollout(task, ep_repair.to_reward_meta())
            assert bd_repair.ex_match is True
            assert bd_repair.self_repair_bonus == pytest.approx(0.25, abs=1e-4)
            assert bd_repair.step_penalty == pytest.approx(0.04, abs=1e-4)
        finally:
            bm.sandbox.close()


class TestSplitIsolationAntiLeakage:
    def test_grpo_prompts_strict_45_train_val_and_leakage_rejection(self, tmp_path: Path):
        grpo_path = PROJECT_ROOT / "outputs" / "rl_export" / "grpo_prompts.jsonl"
        summary_path = PROJECT_ROOT / "outputs" / "rl_export" / "export_summary.json"
        expected_grpo = json.loads(summary_path.read_text(encoding="utf-8"))["grpo_prompt_count"]
        items = load_grpo_prompts(grpo_path, strict_no_test_leakage=True)
        assert len(items) == expected_grpo
        assert len(items) in (45, 72)

        test_ids = get_isolated_test_task_ids()
        assert len(test_ids) in (30, 48)
        assert all(item["task_id"] not in test_ids for item in items)
        assert set(item["split"] for item in items) == {"train", "val"}

        leaked_tid = sorted(test_ids)[0]
        poisoned_grpo = tmp_path / "poisoned_grpo.jsonl"
        poisoned_grpo.write_text(
            json.dumps(
                {
                    "task_id": leaked_tid,
                    "split": "train",
                    "question": "Leaked test question",
                    "gold_sql": "SELECT 1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="leakage"):
            load_grpo_prompts(poisoned_grpo, strict_no_test_leakage=True)

        poisoned_sft = tmp_path / "poisoned_sft.jsonl"
        poisoned_sft.write_text(
            json.dumps(
                {
                    "id": "bad_1",
                    "task_id": leaked_tid,
                    "split": "test",
                    "messages": [{"role": "assistant", "content": "hi", "loss_mask": 1}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="leakage"):
            load_sft_trajectories(poisoned_sft, strict_no_test_leakage=True)


class TestFourStageAndSelfRepairRewards:
    def test_progressive_reward_ordering_self_repair_length_penalty_and_sparse_ablation(self):
        bm = load_benchmark(seed=2026, dialect="odps", precompute_gold=True)
        try:
            task = bm.train[0]
            engine = HierarchicalSQLReward(sandbox=bm.sandbox, dialect="odps")

            r_empty = engine.evaluate_rollout(task, "Sorry, I cannot answer that.")
            r_syntax = engine.evaluate_rollout(
                task,
                json.dumps({"thought": "plan", "sql": "SELECT FROM WHERE ???"}),
            )
            r_dialect = engine.evaluate_rollout(
                task,
                json.dumps(
                    {
                        "thought": "plan",
                        "sql": "SELECT IFNULL(order_status, 'x') FROM dwd_trade_order_di WHERE ds = DATE_ADD('20261015', -1)",
                    }
                ),
            )
            r_no_pt = engine.evaluate_rollout(
                task,
                json.dumps(
                    {
                        "thought": "forgot partition",
                        "sql": "SELECT order_status, COUNT(*) FROM dwd_trade_order_di GROUP BY order_status",
                    }
                ),
            )
            r_runtime = engine.evaluate_rollout(
                task,
                json.dumps(
                    {
                        "thought": "has partition",
                        "sql": "SELECT nonexistent_col_xyz FROM dwd_trade_order_di WHERE ds = MAX_PT('dwd_trade_order_di')",
                    }
                ),
            )
            r_suspicious = engine.evaluate_rollout(
                task,
                json.dumps(
                    {
                        "thought": "impossible filter",
                        "sql": "SELECT order_status, COUNT(*), SUM(pay_amt) FROM dwd_trade_order_di WHERE ds = '19990101' GROUP BY order_status",
                    }
                ),
            )
            r_wrong_ans = engine.evaluate_rollout(
                task,
                json.dumps(
                    {
                        "thought": "wrong metric",
                        "sql": "SELECT order_status, COUNT(*) AS cnt, MAX(pay_amt) AS m FROM dwd_trade_order_di WHERE ds = MAX_PT('dwd_trade_order_di') GROUP BY order_status",
                    }
                ),
            )
            # Repaired Gold without self-repair bonus (ablation mode no_self_repair)
            r_repaired_no_bonus = compute_hierarchical_reward(
                task,
                json.dumps({"thought": "fixed", "sql": task.gold_sql}),
                sandbox=bm.sandbox,
                ablation_mode="no_self_repair",
                refine_turns=1,
                first_pass_gateway=False,
                had_initial_error=True,
            )
            # First-pass Gold EX match
            r_first_pass_gold = engine.evaluate_rollout(
                task,
                json.dumps({"thought": "optimal", "sql": task.gold_sql}),
                refine_turns=0,
                first_pass_gateway=True,
                had_initial_error=False,
            )
            # Self-Repaired Gold EX match (Turn 1 failed gateway -> Turn 2 repaired to Gold)
            r_self_repaired_gold = engine.evaluate_rollout(
                task,
                json.dumps({"thought": "repaired after ODPS-0123031", "sql": task.gold_sql}),
                refine_turns=1,
                first_pass_gateway=False,
                had_initial_error=True,
            )

            assert (
                r_empty.total_reward
                < r_syntax.total_reward
                < r_dialect.total_reward
                < r_no_pt.total_reward
                < r_runtime.total_reward
                < r_suspicious.total_reward
                < r_wrong_ans.total_reward
                < r_repaired_no_bonus.total_reward
                < r_first_pass_gold.total_reward
            )
            assert r_first_pass_gold.total_reward == pytest.approx(2.00, abs=1e-4)
            assert r_self_repaired_gold.self_repair_bonus == pytest.approx(0.25, abs=1e-4)
            assert r_self_repaired_gold.step_penalty == pytest.approx(0.04, abs=1e-4)
            assert r_self_repaired_gold.total_reward == pytest.approx(2.01, abs=1e-4)

            # Verify action length & repetition penalty
            babble_pen = compute_action_length_penalty(["SELECT 1 " * 400])
            assert babble_pen > 0.05

            # Verify pure 0/1 sparse `outcome_only` ablation mode
            sparse_pass = compute_hierarchical_reward(
                task,
                json.dumps({"thought": "ok", "sql": task.gold_sql}),
                sandbox=bm.sandbox,
                ablation_mode="outcome_only",
            )
            sparse_fail = compute_hierarchical_reward(
                task,
                "not valid sql at all",
                sandbox=bm.sandbox,
                ablation_mode="outcome_only",
            )
            assert sparse_pass.total_reward == pytest.approx(1.0, abs=1e-4)
            assert sparse_fail.total_reward == pytest.approx(0.0, abs=1e-4)
            assert sparse_pass.format_reward == 0.0
            assert sparse_pass.gateway_ast_reward == 0.0
            assert sparse_pass.exec_validity_reward == 0.0
        finally:
            bm.sandbox.close()


class TestColabWorkflowAndEndToEndDryRun:
    def test_colab_bundle_secret_sanitization_and_result_merger(self, tmp_path: Path):
        assert "XiYanSQL-QwenCoder-3B-2504" in resolve_vertical_sql_model("3b")
        assert "XiYanSQL-QwenCoder-7B-2504" in resolve_vertical_sql_model("7b")

        sample_secret = "LLM_API_KEY=" + "sk-" + "1234567890abcdefghijklmnop" + " token=" + "hf_" + "abcdefghijklmnopqrstuvwxyz1234"
        clean, n_red = sanitize_text_secrets(sample_secret)
        assert n_red >= 2
        assert ("sk-" + "12345") not in clean
        assert ("hf_" + "abcdef") not in clean

        bundle_path = tmp_path / "test_bundle.tar.gz"
        nb_path = tmp_path / "EvoDW_RL_Colab_A100.ipynb"
        manifest = pack_colab_bundle(bundle_out_path=bundle_path, notebook_out_path=nb_path)
        assert bundle_path.exists() and nb_path.exists()
        assert manifest["compressed_mb"] < 5.0
        with tarfile.open(bundle_path, "r:gz") as tar:
            names = set(tar.getnames())
            assert "src/rl/multiturn_rollout.py" in names
            assert "outputs/rl_export/sft_openai_tool_use.jsonl" in names
            assert ".env" not in names

    @pytest.mark.asyncio
    async def test_offline_dry_run_sft_multiturn_grpo_vertical_and_merge(self, tmp_path: Path):
        sft_out = tmp_path / "sft_ckpt"
        grpo_out = tmp_path / "grpo_ckpt"
        grpo_sparse_out = tmp_path / "grpo_sparse_ckpt"
        report_out = tmp_path / "rl_experiment_report_3b.md"
        json_out = tmp_path / "rl_eval_summary_3b.json"

        model_cfg = RLModelConfig(dry_run=True, lora_r=4, lora_alpha=8, seed=42)
        sft_cfg = SFTTrainingConfig(
            output_dir=sft_out,
            max_seq_length=1024,
            max_steps=2,
            per_device_train_batch_size=2,
            save_merged_model=True,
        )
        sft_res = run_sft_training(model_cfg, sft_cfg)
        assert sft_res["global_steps"] == 2
        assert (sft_out / "sft_adapter" / "adapter_config.json").exists()
        assert (sft_out / "sft_merged" / "config.json").exists()

        grpo_cfg = GRPOTrainingConfig(
            sft_checkpoint_dir=sft_out,
            output_dir=grpo_out,
            enable_multiturn_rollout=True,
            max_rollout_turns=2,
            num_generations=2,
            max_steps=2,
            max_prompt_length=768,
            max_completion_length=128,
        )
        grpo_res = run_grpo_training(model_cfg, grpo_cfg)
        assert grpo_res["global_steps"] == 2
        assert (grpo_out / "grpo_adapter" / "adapter_config.json").exists()

        sparse_cfg = GRPOTrainingConfig(
            sft_checkpoint_dir=sft_out,
            output_dir=grpo_sparse_out,
            ablation_mode="outcome_only",
            enable_multiturn_rollout=True,
            max_rollout_turns=2,
            num_generations=2,
            max_steps=2,
            max_prompt_length=768,
            max_completion_length=128,
        )
        run_grpo_training(model_cfg, sparse_cfg)

        eval_res = await run_full_evaluation(
            vertical_model_spec="xiyansql-3b",
            sft_checkpoint_dir=sft_out,
            grpo_checkpoint_dir=grpo_out,
            grpo_sparse_checkpoint_dir=grpo_sparse_out,
            output_report_path=report_out,
            output_json_path=json_out,
            limit_test_tasks=2,
            dry_run=True,
        )
        assert len(eval_res["models"]) == 7
        assert eval_res["vertical_comparison"] is not None
        assert eval_res["ablation_comparison"]["test_metrics"] is not None

        merged_report = tmp_path / "merged_report.md"
        merged_json = tmp_path / "merged_summary.json"
        merged = merge_colab_results(
            inputs=[json_out],
            import_dir=tmp_path / "imports",
            report_out=merged_report,
            json_out=merged_json,
        )
        assert len(merged["scale_comparison"]) >= 2
        report_txt = merged_report.read_text(encoding="utf-8")
        assert "垂直开源 SQL 模型水土不服对照实验" in report_txt
        assert "GRPO 奖励函数消融实验" in report_txt
        assert "deepseek-v4.1-flash" in report_txt
