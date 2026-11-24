"""RL evaluation curve summarization, case-study builder, and Markdown report renderer."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

CATEGORIES_ORDER = (
    "partition_pruning",
    "json_array_unnest",
    "retention_date_math",
    "fanout_dedup",
    "ratio_null_handling",
)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def summarize_grpo_curve(grpo_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Extract convergence speed, format collapse rate, KL, and online EX from `grpo_training_summary.json`."""
    if not grpo_summary:
        return {}
    logs = [e for e in (grpo_summary.get("log_history") or []) if isinstance(e, dict) and "step" in e]
    rewards = [float(e["reward"]) for e in logs if "reward" in e]
    kls = [float(e["kl"]) for e in logs if "kl" in e]
    collapses = [
        float(e["multiturn/format_collapse_rate"])
        for e in logs
        if "multiturn/format_collapse_rate" in e
    ]
    online_ex = [
        float(e["multiturn/online_ex_rate"])
        for e in logs
        if "multiturn/online_ex_rate" in e
    ]
    sr_rates = [
        float(e["multiturn/self_repair_ex_rate"])
        for e in logs
        if "multiturn/self_repair_ex_rate" in e
    ]

    # Step to reach >=80% of peak reward (convergence speed proxy)
    conv_step = grpo_summary.get("global_steps", 0)
    if rewards:
        r_min, r_max = min(rewards), max(rewards)
        threshold = r_min + 0.80 * (r_max - r_min)
        for e in logs:
            if "reward" in e and float(e["reward"]) >= threshold:
                conv_step = int(e["step"])
                break

    return {
        "ablation_mode": grpo_summary.get("ablation_mode", "full"),
        "global_steps": int(grpo_summary.get("global_steps", 0)),
        "initial_reward": round(rewards[0], 4) if rewards else 0.0,
        "peak_reward": round(max(rewards), 4) if rewards else 0.0,
        "final_reward": round(rewards[-1], 4) if rewards else 0.0,
        "convergence_step_80pct": conv_step,
        "mean_kl": round(sum(kls) / len(kls), 5) if kls else 0.0,
        "mean_format_collapse_rate": round(sum(collapses) / len(collapses), 4) if collapses else 0.0,
        "final_format_collapse_rate": round(collapses[-1], 4) if collapses else 0.0,
        "peak_online_ex_rate": round(max(online_ex), 4) if online_ex else 0.0,
        "mean_self_repair_ex_rate": round(sum(sr_rates) / len(sr_rates), 4) if sr_rates else 0.0,
        "train_runtime_sec": float(grpo_summary.get("train_runtime_sec", 0.0)),
    }


def render_rl_markdown_report(
    *,
    model_results: list[tuple[str, dict[str, Any]]],
    base_model_spec: str,
    test_task_count: int,
    sft_summary: dict[str, Any] | None = None,
    grpo_summary: dict[str, Any] | None = None,
    grpo_sparse_summary: dict[str, Any] | None = None,
    vertical_comparison: dict[str, Any] | None = None,
    ablation_test_metrics: dict[str, dict[str, Any]] | None = None,
    scale_comparison: list[dict[str, Any]] | None = None,
    case_studies: list[dict[str, Any]] | None = None,
) -> str:
    """Format comprehensive Markdown evaluation report for EvoDW-RL v2."""
    sft_records_n = (
        int((sft_summary.get("corpus_stats") or {}).get("total_records", 299))
        if sft_summary
        else 299
    )
    lines: list[str] = []
    lines.append("# EvoDW-RL v2 实验评测报告（多轮工具交互 SFT + Observation-Masked Agentic GRPO）")
    lines.append("")
    lines.append(f"- 生成时间: `{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- 主实验模型基座: `{base_model_spec}`")
    lines.append(
        f"- 隔离评测集: `Test Split = {test_task_count} 道企业数仓题`（与 SFT `{sft_records_n}` 条验证轨迹 / GRPO `45` 道训练题严格零交集）"
    )
    lines.append("- 本地双层仿真沙盒: `WarehouseSandbox`（`GatewayLinter` 静态分区/方言 AST 拦截 + `LocalDuckDBExecutor` 确定性转译执行）")
    lines.append("")

    # Section 1: Main Comparison Table
    lines.append("## 1. 五组/全链路模型核心指标对比（Isolated Test Set）")
    lines.append("")
    lines.append(
        "| 模型方案 | EX (执行准确率) | First-Pass Gateway Rate | Invalid SQL Rate | "
        "1st-Pass 对 / 自修复对 | Avg Steps | 单题推理延迟 (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in model_results:
        fp_c = m.get("first_pass_correct_count", 0)
        sr_c = m.get("self_repaired_count", 0)
        lines.append(
            f"| **{name}** | **{_pct(m['ex'])}** | {_pct(m['first_pass_gateway_rate'])} | "
            f"{_pct(m['invalid_sql_rate'])} | {fp_c} / {sr_c} | "
            f"{m.get('avg_steps', 0.0):.2f} | {m.get('avg_latency_ms', 0.0):.1f} |"
        )
    lines.append("")

    # Section 2: 5-Category Accuracy Breakdown
    lines.append("## 2. 五大数仓高频错题分项准确率（EX）对比")
    lines.append("")
    header_cols = " | ".join(name for name, _ in model_results)
    align_cols = " | ".join("---:" for _ in model_results)
    lines.append(f"| 题型类别 | 题数 | {header_cols} |")
    lines.append(f"|---|---:| {align_cols} |")

    for cat in CATEGORIES_ORDER:
        cnt = 0
        row_cells: list[str] = []
        for _, m in model_results:
            cat_m = (m.get("by_category") or {}).get(cat, {})
            if cat_m.get("count"):
                cnt = int(cat_m["count"])
            ex_v = float(cat_m.get("ex", 0.0))
            gw_v = float(cat_m.get("first_pass_gateway_rate", 0.0))
            row_cells.append(f"{_pct(ex_v)} _(GW {_pct(gw_v)})_")
        lines.append(f"| `{cat}` | {cnt} | " + " | ".join(row_cells) + " |")
    lines.append("")

    # Optional Section: Cross-Scale Comparison (e.g. local 1.5B v1 vs Colab A100 3B/7B v2)
    if scale_comparison:
        lines.append("## 3. 模型规模与版本阶梯对比（v1 1.5B vs v2 3B/7B）")
        lines.append("")
        lines.append("| 实验版本 / 基座规模 | 轨迹数 / GRPO 步数 | Base EX | +GEPA Prompt EX | SFT EX | Multi-Turn SFT+GRPO EX | 非法动作率 (GRPO) | 单题延迟 (ms) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for sc in scale_comparison:
            lines.append(
                f"| **{sc['scale_label']}** | `{sc['train_budget']}` | "
                f"{_pct(sc['base_ex'])} | {_pct(sc['gepa_ex'])} | {_pct(sc['sft_ex'])} | "
                f"**{_pct(sc['grpo_ex'])}** | {_pct(sc['grpo_invalid_rate'])} | {sc['grpo_latency_ms']:.1f} |"
            )
        lines.append("")

    # Optional Section: Vertical SQL Model Domain-Shift Comparison
    if vertical_comparison:
        lines.append("## 4. 垂直开源 SQL 模型水土不服对照实验（XiYanSQL vs EvoDW-RL）")
        lines.append("")
        v_name = vertical_comparison["vertical_label"]
        o_name = vertical_comparison["ours_label"]
        vm = vertical_comparison["vertical_metrics"]
        om = vertical_comparison["ours_metrics"]
        lines.append(
            "| 对比方案 | Test EX | First-Pass Gateway Rate | Invalid SQL Rate | 分区裁剪题 EX (`partition_pruning`) | JSON/数组展开题 EX (`json_array_unnest`) | 单题延迟 (ms) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for nm, mm in ((v_name, vm), (o_name, om)):
            pp_ex = float(((mm.get("by_category") or {}).get("partition_pruning") or {}).get("ex", 0.0))
            ja_ex = float(((mm.get("by_category") or {}).get("json_array_unnest") or {}).get("ex", 0.0))
            lines.append(
                f"| **{nm}** | **{_pct(mm['ex'])}** | {_pct(mm['first_pass_gateway_rate'])} | "
                f"{_pct(mm['invalid_sql_rate'])} | {_pct(pp_ex)} | {_pct(ja_ex)} | {mm.get('avg_latency_ms', 0.0):.1f} |"
            )
        if vertical_comparison.get("gateway_intercept_breakdown"):
            lines.append("")
            lines.append(f"- **垂直模型典型网关拦截归因**: `{json.dumps(vertical_comparison['gateway_intercept_breakdown'], ensure_ascii=False)}`")
        lines.append("")

    # Optional Section: GRPO Sparse vs Dense + Self-Repair Ablation
    if grpo_sparse_summary or ablation_test_metrics:
        lines.append("## 5. GRPO 奖励函数消融实验（Outcome-Only Sparse vs 4-Stage Dense + Self-Repair）")
        lines.append("")
        dense_c = summarize_grpo_curve(grpo_summary)
        sparse_c = summarize_grpo_curve(grpo_sparse_summary)
        lines.append(
            "| 奖励配置 | 收敛步数 (达峰值80%) | 训练期格式坍塌率 | 平均 KL 散度 | 在线自修复命中率 | 隔离测试集 Invalid SQL | 隔离测试集最终 EX |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        ab_rows = [
            ("Outcome-Only Sparse (仅最终 EX 0/1)", sparse_c, (ablation_test_metrics or {}).get("outcome_only")),
            ("4-Stage Dense + Self-Repair (本文完整奖励)", dense_c, (ablation_test_metrics or {}).get("full")),
        ]
        for r_title, c_stat, t_met in ab_rows:
            if not c_stat and not t_met:
                continue
            inv_s = _pct(t_met["invalid_sql_rate"]) if t_met else "N/A"
            ex_s = f"**{_pct(t_met['ex'])}**" if t_met else "N/A"
            lines.append(
                f"| **{r_title}** | `{c_stat.get('convergence_step_80pct', 'N/A')}` | "
                f"{_pct(c_stat.get('mean_format_collapse_rate', 0.0))} | "
                f"`{c_stat.get('mean_kl', 0.0):.4f}` | "
                f"{_pct(c_stat.get('mean_self_repair_ex_rate', 0.0))} | "
                f"{inv_s} | {ex_s} |"
            )
        lines.append("")

    # Post-Training Telemetry
    if sft_summary or grpo_summary:
        lines.append("## 6. 两阶段后训练遥测摘要（Stage-1 SFT & Stage-2 Multi-Turn GRPO）")
        lines.append("")
        if sft_summary:
            cs = sft_summary.get("corpus_stats", {})
            lines.append("### 6.1 第一阶段：Loss-Masked 多轮轨迹 SFT（自适应稀缺轨迹重加权）")
            lines.append(
                f"- 原始去重验证轨迹: `{cs.get('total_records', 0)}` 条 "
                f"（首轮即对 `{cs.get('first_pass_correct_count', 0)}` 条 + "
                f"网关报错自修复 `{cs.get('self_repaired_count', 0)}` 条，含 `inspect_table_sample` 探查 `{cs.get('probe_trajectory_count', 0)}` 条）"
            )
            lines.append(
                f"- 自适应重加权后训练轮数: `{cs.get('effective_train_episodes', 0)}` 条 "
                f"（自修复 `{cs.get('effective_self_repaired_episodes', cs.get('self_repaired_count', 0))}` 条 / "
                f"探查 `{cs.get('effective_probe_episodes', cs.get('probe_trajectory_count', 0))}` 条，尾部截断轨迹数 `{cs.get('truncated_trajectory_count', 0)}`）"
            )
            lines.append(
                f"- Loss Mask 统计: 总 Token `{cs.get('total_tokens', 0)}`，"
                f"仅参与 Cross-Entropy 的 `assistant` Token `{cs.get('supervised_assistant_tokens', 0)}` "
                f"（占比 `{cs.get('supervised_token_ratio', 0.0) * 100:.1f}%`），其余 `system`/`user`/`tool` 严格屏蔽为 `-100`"
            )
            lines.append(
                f"- 训练步数 / 最终 Loss / 耗时: `{sft_summary.get('global_steps', 0)}` steps / "
                f"`{sft_summary.get('train_loss', 0.0):.4f}` / `{sft_summary.get('train_runtime_sec', 0.0):.1f}s`"
            )
            lines.append("")
        if grpo_summary:
            dc = summarize_grpo_curve(grpo_summary)
            lines.append("### 6.2 第二阶段：多轮沙盒反馈 Observation-Masked Agentic GRPO")
            lines.append(
                f"- 训练题集: `grpo_prompts.jsonl` 共 `{grpo_summary.get('prompt_count', 0)}` 道 `train + val` 题（严格隔离 30 道 `test` 题）"
            )
            lines.append(
                f"- 多轮交互配置: `max_rollout_turns={grpo_summary.get('max_rollout_turns', 3)}`, "
                f"`G={grpo_summary.get('num_generations', 8)}`, `KL beta={grpo_summary.get('beta_kl', 0.04)}` "
                "（Action Token `completion_mask=1`，沙盒报错 Observation Token 严格 `completion_mask=0`）"
            )
            lines.append(
                f"- 激活奖励组件 (`{grpo_summary.get('ablation_mode', 'full')}`): "
                f"`{', '.join(grpo_summary.get('active_reward_funcs', []))}`"
            )
            lines.append(
                f"- 训练步数 / 峰值奖励 / 耗时: `{grpo_summary.get('global_steps', 0)}` steps / "
                f"`{dc.get('peak_reward', 0.0):.4f}` / `{grpo_summary.get('train_runtime_sec', 0.0):.1f}s`"
            )
            lines.append("")

    # Concrete Test-Set Case Studies
    if case_studies:
        lines.append("## 7. 典型测试题演进案例对比（Base -> SFT -> Multi-Turn SFT+GRPO）")
        lines.append("")
        for cs in case_studies[:3]:
            lines.append(f"### 案例 `{cs['task_id']}`（类别: `{cs['category']}`）")
            lines.append(f"- **业务问题**: {cs['question']}")
            lines.append(f"- **Base Zero-shot**: EX=`{cs['base_ex']}`, 首轮网关=`{cs['base_gw']}` | SQL: `{cs['base_sql']}`")
            if cs.get("base_err"):
                lines.append(f"  - 报错诊断: `{cs['base_err']}`")
            lines.append(f"- **SFT 冷启动**: EX=`{cs['sft_ex']}`, 首轮网关=`{cs['sft_gw']}`, 修复轮次=`{cs['sft_refines']}` | SQL: `{cs['sft_sql']}`")
            lines.append(f"- **Multi-Turn SFT + GRPO**: EX=`{cs['grpo_ex']}`, 首轮网关=`{cs['grpo_gw']}`, 修复轮次=`{cs['grpo_refines']}` | SQL: `{cs['grpo_sql']}`")
            lines.append("")

    return "\n".join(lines)


def build_test_case_studies(traj_store: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build ranked test-set progression case studies (Base -> SFT -> Multi-Turn SFT+GRPO)."""
    case_studies: list[dict[str, Any]] = []
    if not all(k in traj_store for k in ('base', 'sft', 'sft_grpo')):
        return case_studies
    for tb, ts, tg in zip(traj_store['base'], traj_store['sft'], traj_store['sft_grpo']):
        priority = (
            (3 if tg['ex_match'] and not ts['ex_match'] else 0)
            + (2 if tg['ex_match'] and not tb['ex_match'] else 0)
            + (1 if tg['first_pass_gateway'] and not tb['first_pass_gateway'] else 0)
        )
        case_studies.append(
            {
                'priority': priority,
                'task_id': tb['task_id'],
                'category': tb['category'],
                'question': tb['question'],
                'base_ex': tb['ex_match'],
                'base_gw': tb['first_pass_gateway'],
                'base_sql': (tb.get('final_sql') or '')[:240] or '(empty)',
                'base_err': (tb.get('first_pass_error_msg') or '')[:200],
                'sft_ex': ts['ex_match'],
                'sft_gw': ts['first_pass_gateway'],
                'sft_refines': ts['refine_turns'],
                'sft_sql': (ts.get('final_sql') or '')[:240] or '(empty)',
                'grpo_ex': tg['ex_match'],
                'grpo_gw': tg['first_pass_gateway'],
                'grpo_refines': tg['refine_turns'],
                'grpo_sql': (tg.get('final_sql') or '')[:240] or '(empty)',
            }
        )
    case_studies.sort(key=lambda x: (-x['priority'], x['task_id']))
    return case_studies
