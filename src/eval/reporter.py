"""Markdown experiment report generator with evolution curve, ablation table, RL pool stats, and prompt diffs."""

from __future__ import annotations

import datetime as dt
import difflib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.rsi.engine import EvolutionRunResult


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _delta_pct(before: float, after: float) -> str:
    diff = (after - before) * 100.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}%"


def _rel_change_pct(before: float, after: float) -> str:
    if before <= 0:
        return "0.0%"
    rel = ((after - before) / before) * 100.0
    sign = "+" if rel >= 0 else ""
    return f"{sign}{rel:.1f}%"


def _unified_prompt_diff(before: str, after: str, module_name: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"seed_v0/{module_name}.txt",
            tofile=f"evolved/{module_name}.txt",
            lineterm="",
        )
    )
    if not lines:
        return "_No changes relative to seed prompt._"
    return "```diff\n" + "\n".join(lines) + "\n```"


def generate_experiment_report(
    run_result: "EvolutionRunResult",
    output_path: Path,
    *,
    runs_dir: Path | None = None,
    ablation_results: list["EvolutionRunResult"] | None = None,
    rl_export_summary: dict[str, Any] | None = None,
) -> Path:
    """Persist run summary JSON and write comprehensive Markdown report to `output_path`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_runs_dir = runs_dir or (output_path.parent / "runs")
    effective_runs_dir.mkdir(parents=True, exist_ok=True)

    all_to_save = [run_result] + [r for r in (ablation_results or []) if r is not run_result]
    for res in all_to_save:
        summary_file = effective_runs_dir / f"{res.method}_summary.json"
        summary_file.write_text(
            json.dumps(res.to_summary_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    ablation_rows: dict[str, dict[str, Any]] = {}
    for method_name in ("seed", "greedy", "gepa"):
        p = effective_runs_dir / f"{method_name}_summary.json"
        if p.exists():
            ablation_rows[method_name] = json.loads(p.read_text(encoding="utf-8"))

    if "seed" not in ablation_rows:
        ablation_rows["seed"] = {
            "method": "seed",
            "rollouts_used": 0,
            "best_val": run_result.seed_val_metrics.to_dict(),
            "best_test": run_result.seed_test_metrics.to_dict(),
        }

    seed_test = run_result.seed_test_metrics
    best_test = run_result.best_test_metrics
    seed_val = run_result.seed_val_metrics
    best_val = run_result.best_val_metrics

    lines: list[str] = []
    lines.append("# EvoDW-RSI 实验报告（数仓 Text-to-SQL 策略自进化评测）")
    lines.append("")
    lines.append(f"- 生成时间: `{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- 主优化方法: `{run_result.method}` | 目标方言: `{run_result.dialect.upper()}`")
    lines.append(f"- 优化 Rollout 预算 / 实际消耗: `{run_result.rollouts_used} / {run_result.budget}`")
    if getattr(run_result, "effective_val_size", 0) > 0:
        lines.append(
            f"- 分层验证集规模 (Two-Stage Active Val): `{run_result.effective_val_size} 题` "
            f"(评估缓存命中复用: `{run_result.cache_hits}` 次)"
        )
    lines.append(
        f"- 选出最优策略 ID: `{run_result.best_policy.policy_id}` "
        f"(lineage parents: `{run_result.best_policy.parent_ids}`)"
    )
    lines.append("")

    # 1. Overall Before/After summary
    lines.append("## 1. 进化前后核心指标总览（Seed vs Evolved）")
    lines.append("")
    lines.append("| 评测集 | 指标 | Seed Baseline | Evolved Best | Δ 提升 / 变化 |")
    lines.append("|---|---|---:|---:|---:|")
    lines.append(f"| Val ({seed_val.total}题) | **EX (执行准确率)** | {_pct(seed_val.ex)} | **{_pct(best_val.ex)}** | {_delta_pct(seed_val.ex, best_val.ex)} |")
    lines.append(f"| Val ({seed_val.total}题) | **First-Pass Gateway Rate** | {_pct(seed_val.first_pass_gateway_rate)} | **{_pct(best_val.first_pass_gateway_rate)}** | {_delta_pct(seed_val.first_pass_gateway_rate, best_val.first_pass_gateway_rate)} |")
    lines.append(f"| Val ({seed_val.total}题) | Invalid / Error SQL Rate | {_pct(seed_val.invalid_sql_rate)} | {_pct(best_val.invalid_sql_rate)} | {_delta_pct(seed_val.invalid_sql_rate, best_val.invalid_sql_rate)} |")
    lines.append(f"| Val ({seed_val.total}题) | Avg Steps / Avg Tokens / Avg Latency | {seed_val.avg_steps:.2f} / {seed_val.avg_tokens:.0f} / {seed_val.avg_latency_ms:.1f}ms | {best_val.avg_steps:.2f} / {best_val.avg_tokens:.0f} / {best_val.avg_latency_ms:.1f}ms | {_rel_change_pct(seed_val.avg_tokens, best_val.avg_tokens)} tok / {_rel_change_pct(seed_val.avg_latency_ms, best_val.avg_latency_ms)} lat |")
    lines.append(f"| Test ({seed_test.total}题) | **EX (执行准确率)** | {_pct(seed_test.ex)} | **{_pct(best_test.ex)}** | {_delta_pct(seed_test.ex, best_test.ex)} |")
    lines.append(f"| Test ({seed_test.total}题) | **First-Pass Gateway Rate** | {_pct(seed_test.first_pass_gateway_rate)} | **{_pct(best_test.first_pass_gateway_rate)}** | {_delta_pct(seed_test.first_pass_gateway_rate, best_test.first_pass_gateway_rate)} |")
    lines.append(f"| Test ({seed_test.total}题) | Invalid / Error SQL Rate | {_pct(seed_test.invalid_sql_rate)} | {_pct(best_test.invalid_sql_rate)} | {_delta_pct(seed_test.invalid_sql_rate, best_test.invalid_sql_rate)} |")
    lines.append(f"| Test ({seed_test.total}题) | First-Pass Correct / Self-Repaired 题数 | {seed_test.first_pass_correct_count} / {seed_test.self_repaired_count} | **{best_test.first_pass_correct_count}** / {best_test.self_repaired_count} | {best_test.first_pass_correct_count - seed_test.first_pass_correct_count:+d} / {best_test.self_repaired_count - seed_test.self_repaired_count:+d} |")
    lines.append(f"| Test ({seed_test.total}题) | Avg Steps / Avg Tokens / Avg Latency | {seed_test.avg_steps:.2f} / {seed_test.avg_tokens:.0f} / {seed_test.avg_latency_ms:.1f}ms | {best_test.avg_steps:.2f} / {best_test.avg_tokens:.0f} / {best_test.avg_latency_ms:.1f}ms | {_rel_change_pct(seed_test.avg_tokens, best_test.avg_tokens)} tok / {_rel_change_pct(seed_test.avg_latency_ms, best_test.avg_latency_ms)} lat |")
    lines.append("")

    # 2. Ablation table
    lines.append("## 2. 消融对比表（`seed` vs `greedy` vs `gepa`）")
    lines.append("")
    lines.append("| Method | Opt Rollouts | Val EX | Test EX | Test First-Pass GW | Test Invalid Rate | Avg Steps | Avg Tokens | Avg Latency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m_name in ("seed", "greedy", "gepa"):
        if m_name not in ablation_rows:
            continue
        row = ablation_rows[m_name]
        bv = row["best_val"]
        bt = row["best_test"]
        lines.append(
            f"| `{m_name}` | {row.get('rollouts_used', 0)} | {_pct(bv['ex'])} | **{_pct(bt['ex'])}** | "
            f"{_pct(bt['first_pass_gateway_rate'])} | {_pct(bt['invalid_sql_rate'])} | "
            f"{bt['avg_steps']:.2f} | {bt['avg_tokens']:.0f} | {bt.get('avg_latency_ms', 0.0):.1f}ms |"
        )
    lines.append("")

    # 3. 5-Category breakdown
    lines.append("## 3. 五大数仓高频错题分项表现（Test Set）")
    lines.append("")
    lines.append("| 题型类别 | 题数 | Seed EX | Evolved EX | Δ EX | Seed 1st-Pass GW | Evolved 1st-Pass GW |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    cats = sorted(set(seed_test.by_category.keys()) | set(best_test.by_category.keys()))
    for cat in cats:
        sc = seed_test.by_category.get(cat)
        bc = best_test.by_category.get(cat)
        cnt = bc.count if bc else (sc.count if sc else 0)
        s_ex = sc.ex if sc else 0.0
        b_ex = bc.ex if bc else 0.0
        s_gw = sc.first_pass_gateway_rate if sc else 0.0
        b_gw = bc.first_pass_gateway_rate if bc else 0.0
        lines.append(
            f"| `{cat}` | {cnt} | {_pct(s_ex)} | **{_pct(b_ex)}** | {_delta_pct(s_ex, b_ex)} | "
            f"{_pct(s_gw)} | {_pct(b_gw)} |"
        )
    lines.append("")

    # 4. Evolution generation progression
    lines.append("## 4. 进化代数轨迹与谱系演化（Generation Curve & Lineage）")
    lines.append("")
    lines.append("| Gen | Policy ID | Parents | Op | Target Module | Smoke Gate | Active-Val EX | 1st-Pass GW | Frontier Size | Rollouts | Diagnosis / Rationale |")
    lines.append("|---:|---|---|---|---|---|---|---:|---:|---:|---|")
    for h in run_result.history:
        ex_str = _pct(h.active_val_ex) if h.active_val_ex is not None else "-"
        gw_str = _pct(h.active_val_first_pass_gw) if h.active_val_first_pass_gw is not None else "-"
        parents_str = ", ".join(h.parent_ids) if h.parent_ids else "-"
        gate_str = "PASS" if h.smoke_passed else "DROP"
        short_rat = h.rationale.replace("\n", " ")[:120]
        lines.append(
            f"| {h.generation} | `{h.policy_id}` | `{parents_str}` | `{h.operation}` | `{h.target_module}` | "
            f"{gate_str} | {ex_str} | {gw_str} | {h.frontier_size} | {h.rollouts_used} | {short_rat} |"
        )
    lines.append("")

    # 5. Pareto frontier weights snapshot
    weights = run_result.archive.get_win_frequency_weights()
    if weights:
        lines.append("### Pareto 前沿池与胜频采样权重")
        lines.append("")
        lines.append("| Frontier Policy ID | Active-Val EX | First-Pass GW | Eval Tasks | Win-Frequency Sampling Weight |")
        lines.append("|---|---:|---:|---:|---:|")
        for cid, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            rec = run_result.archive.candidates[cid]
            lines.append(
                f"| `{cid}` | {_pct(rec.val_ex)} | {_pct(rec.first_pass_gw_rate)} | "
                f"{rec.num_eval_tasks} | {w:.3f} |"
            )
        lines.append("")

    # 6. LLM cost & token telemetry
    if run_result.llm_usage_summary:
        usage = run_result.llm_usage_summary
        lines.append("## 5. 双模型成本与 Token 统计")
        lines.append("")
        for role in ("actor", "reflector"):
            ru = usage.get(role, {})
            lines.append(
                f"- **{role.capitalize()}**: calls=`{ru.get('calls', 0)}`, "
                f"prompt_tokens=`{ru.get('prompt_tokens', 0)}`, "
                f"completion_tokens=`{ru.get('completion_tokens', 0)}`, "
                f"total_tokens=`{ru.get('total_tokens', 0)}`, "
                f"avg_latency=`{ru.get('avg_latency_ms', 0.0):.1f}ms`"
            )
        lines.append("")

    # 7. Unified Prompt Diffs across all 3 modules
    lines.append("## 6. 策略进化前后三大模块 Prompt 关键规则 Diff")
    lines.append("")
    lines.append("### 6.1 `WarehouseLinker` Prompt Diff")
    lines.append(_unified_prompt_diff(run_result.seed_policy.linker_prompt, run_result.best_policy.linker_prompt, "linker_system"))
    lines.append("")
    lines.append("### 6.2 `DialectSQLGenerator` Prompt Diff")
    lines.append(_unified_prompt_diff(run_result.seed_policy.generator_prompt, run_result.best_policy.generator_prompt, "generator_system"))
    lines.append("")
    lines.append("### 6.3 `GatewayRefiner` Prompt Diff")
    lines.append(_unified_prompt_diff(run_result.seed_policy.refiner_prompt, run_result.best_policy.refiner_prompt, "refiner_system"))
    lines.append("")

    # 8. Optional SFT / RL Trajectory Pool & Efficiency Gains Summary
    if rl_export_summary:
        tot_t = rl_export_summary.get("total_exported_trajectories", 0)
        fpc = rl_export_summary.get("first_pass_correct_count", 0)
        sre = rl_export_summary.get("self_repaired_after_error_count", 0)
        fpc_r = rl_export_summary.get("first_pass_correct_ratio", 0.0)
        sre_r = rl_export_summary.get("self_repaired_after_error_ratio", 0.0)
        tt_tel = rl_export_summary.get("trajectory_type_telemetry", {})
        fpc_tel = tt_tel.get("first_pass_correct", {})
        sre_tel = tt_tel.get("self_repaired_after_error", {})

        lines.append("## 7. 高质量 SFT/RL 验证轨迹池与进化收益分析")
        lines.append("")
        lines.append(
            f"- 去重后验证通过总轨迹数 (`train` + `val`, 严格排除 `test`): **`{tot_t}`** 条 "
            f"(GRPO 训练 Prompt 数: `{rl_export_summary.get('grpo_prompt_count', 0)}`)"
        )
        lines.append(f"- 按数据集划分分布: `{json.dumps(rl_export_summary.get('by_split', {}), ensure_ascii=False)}`")
        lines.append(f"- 按数仓错题类别分布: `{json.dumps(rl_export_summary.get('by_category', {}), ensure_ascii=False)}`")
        lines.append("")
        lines.append("| 轨迹类型 (`trajectory_type`) | 去重样本数 | 占比 | 平均推理步数 | 平均消耗 Tokens | 平均端到端延迟 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(
            f"| `first_pass_correct` | {fpc} | {_pct(fpc_r)} | "
            f"{fpc_tel.get('avg_steps', 0.0):.2f} | {fpc_tel.get('avg_tokens', 0.0):.0f} | "
            f"{fpc_tel.get('avg_latency_ms', 0.0):.1f}ms |"
        )
        lines.append(
            f"| `self_repaired_after_error` | {sre} | {_pct(sre_r)} | "
            f"{sre_tel.get('avg_steps', 0.0):.2f} | {sre_tel.get('avg_tokens', 0.0):.0f} | "
            f"{sre_tel.get('avg_latency_ms', 0.0):.1f}ms |"
        )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
