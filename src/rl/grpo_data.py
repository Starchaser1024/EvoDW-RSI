"""GRPO dataset loader, leak-free table linker, and ChatML prompt formatter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.agent.policy import Policy
from src.env.sandbox import DialectType, WarehouseSandbox
from src.rl.sft_dataset import format_non_supervised_turn_chatml, get_isolated_test_task_ids


def load_grpo_prompts(
    jsonl_path: Path | str,
    *,
    strict_no_test_leakage: bool = True,
) -> list[dict[str, Any]]:
    """Load `grpo_prompts.jsonl` (45 train+val tasks) and block any held-out test task leakage."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"GRPO prompt file not found: {path}")

    forbidden_test_ids = get_isolated_test_task_ids() if strict_no_test_leakage else set()
    items: list[dict[str, Any]] = []

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        split = str(obj.get("split", ""))
        task_id = str(obj.get("task_id", ""))

        if strict_no_test_leakage:
            if split == "test" or task_id in forbidden_test_ids:
                raise ValueError(
                    f"Test-set leakage blocked at {path}:{line_no}! Found held-out test task "
                    f"task_id={task_id!r} (split={split!r}) in GRPO training prompts."
                )
            if split not in ("train", "val"):
                raise ValueError(
                    f"Invalid split={split!r} at {path}:{line_no}; GRPO strictly allows 'train' and 'val' only."
                )

        if not obj.get("question") or not obj.get("gold_sql"):
            raise ValueError(f"Malformed GRPO prompt entry at {path}:{line_no}: missing question/gold_sql.")
        items.append(obj)

    if not items:
        raise ValueError(f"No valid GRPO prompts loaded from {path}.")
    return items


def select_candidate_tables_leak_free(
    question: str,
    hint: str,
    sandbox: WarehouseSandbox,
) -> list[str]:
    """Select relevant catalog tables using ONLY question, hint, and catalog metadata (never gold_sql)."""
    query_text = f"{question} {hint}".lower()
    all_tables = sandbox.catalog.list_tables()
    selected: list[str] = []

    # 1. Explicit table mentions in question or business hint.
    for meta in all_tables:
        if meta.name.lower() in query_text:
            selected.append(meta.name)

    # 2. Domain/column keyword overlap scoring across catalog metadata.
    scored: list[tuple[int, str]] = []
    for meta in all_tables:
        if meta.name in selected:
            continue
        score = 0
        for col in meta.columns:
            if len(col.name) >= 4 and col.name.lower() in query_text:
                score += 2
        domain_keywords = {
            "dim_user": ("用户", "注册", "会员", "等级", "城市", "reg_dt", "vip"),
            "dim_shop": ("店铺", "商家", "类目", "shop", "category"),
            "dim_item": ("商品", "品牌", "类目", "item", "brand"),
            "dim_content_creator": ("达人", "创作者", "粉丝", "creator", "mcn"),
            "ods_app_event_log_di": ("埋点", "曝光", "点击", "留存", "活跃", "event", "json", "properties", "页面"),
            "dwd_trade_order_di": ("订单", "支付", "实付", "下单", "购买", "复购", "gmv", "order", "pay_amt", "coupon"),
            "dwd_refund_di": ("退款", "售后", "退货", "refund"),
            "dwd_content_post_di": ("内容", "帖子", "图文", "视频", "话题", "标签", "post", "tag", "explode", "点赞"),
            "dws_shop_metrics_1d": ("汇总", "日报", "uv", "转化率", "客单价", "曝光", "visitor", "metrics"),
        }
        for kw in domain_keywords.get(meta.name, ()):
            if kw in query_text:
                score += 2
        if score > 0:
            scored.append((score, meta.name))

    scored.sort(key=lambda x: (-x[0], x[1]))
    for _, tname in scored[:4]:
        if tname not in selected:
            selected.append(tname)

    if not selected:
        return [m.name for m in all_tables]
    return selected


def build_grpo_chatml_prompt(
    item: dict[str, Any],
    *,
    sandbox: WarehouseSandbox,
    policy: Policy,
    dialect: DialectType = "odps",
) -> str:
    """Format a leak-free multi-turn ChatML prompt aligned with Stage-1 SFT trajectory prefixes."""
    question = str(item["question"])
    hint = str(item.get("hint") or "None")
    active_dialect = str(item.get("dialect") or dialect).upper()

    catalog_summary = sandbox.catalog.format_catalog_summary()
    selected_tables = select_candidate_tables_leak_free(question, hint, sandbox)
    ddl_text = sandbox.catalog.format_table_ddl(selected_tables)

    pt_notes: list[str] = []
    for tname in selected_tables:
        if sandbox.catalog.is_partitioned(tname):
            sample_info = sandbox.inspect_table_sample(tname, columns=[], sample_rows=1)
            pinfo = sample_info.get("partition_info", {})
            if pinfo.get("max_partition"):
                pt_notes.append(
                    f"{tname}: partition_col={pinfo.get('partition_col')}, "
                    f"range=[{pinfo.get('min_partition')}..{pinfo.get('max_partition')}], "
                    f"recent={pinfo.get('recent_partitions', [])[:3]}"
                )

    schema_notes = "; ".join(pt_notes) if pt_notes else "Use explicit partition pruning on partitioned tables."
    linker_json = json.dumps(
        {
            "thought": "Selected relevant warehouse tables and verified partition metadata.",
            "action": "finalize",
            "selected_tables": selected_tables,
            "schema_notes": schema_notes,
        },
        ensure_ascii=False,
    )

    sys_turn = {
        "role": "system",
        "content": (
            f"[Policy={policy.policy_id} | Dialect={active_dialect}]\n"
            f"## Linker Policy\n{policy.linker_prompt}\n\n"
            f"## Generator Policy\n{policy.generator_prompt}\n\n"
            f"## Refiner Policy\n{policy.refiner_prompt}"
        ),
    }
    linker_user = {
        "role": "user",
        "content": (
            f"Target Dialect: {active_dialect}\n"
            f"Business Question: {question}\n"
            f"Business Hint: {hint}\n\n"
            f"Warehouse Catalog Summary:\n{catalog_summary}\n\n"
            "Return JSON to either probe table samples (action='probe') or finalize selected tables (action='finalize')."
        ),
    }
    gen_user = {
        "role": "user",
        "content": (
            f"Target Dialect: {active_dialect}\n"
            f"Business Question: {question}\n"
            f"Business Hint: {hint}\n\n"
            f"### Selected Table Schemas\n{ddl_text}\n\n"
            f"### Linker Schema Notes\n{schema_notes}\n\n"
            "You may either probe column samples first (`{\"thought\": \"...\", \"action\": \"probe\", "
            "\"probe_requests\": [{\"table_name\": \"...\", \"columns\": [...]}]}`) or directly generate the single "
            'read-only SQL query in strict JSON: {"thought": "...", "sql": "..."}.'
        ),
    }

    prompt_str = (
        format_non_supervised_turn_chatml(sys_turn)
        + format_non_supervised_turn_chatml(linker_user)
        + f"<|im_start|>assistant\n{linker_json}<|im_end|>\n"
        + format_non_supervised_turn_chatml(gen_user)
        + "<|im_start|>assistant\n"
    )
    return prompt_str


def build_grpo_hf_dataset(
    items: list[dict[str, Any]],
    *,
    sandbox: WarehouseSandbox,
    policy: Policy,
    dialect: DialectType = "odps",
) -> Dataset:
    """Build HuggingFace `Dataset` with formatted `prompt` column and task ground-truth metadata."""
    rows: list[dict[str, Any]] = []
    for item in items:
        prompt_text = build_grpo_chatml_prompt(
            item,
            sandbox=sandbox,
            policy=policy,
            dialect=dialect,
        )
        rows.append(
            {
                "prompt": prompt_text,
                "task_id": str(item["task_id"]),
                "split": str(item["split"]),
                "category": str(item.get("category", "")),
                "dialect": str(item.get("dialect", dialect)),
                "question": str(item["question"]),
                "hint": str(item.get("hint", "")),
                "gold_sql": str(item["gold_sql"]),
                "ordered": bool(item.get("ordered", False)),
            }
        )
    return Dataset.from_list(rows)
