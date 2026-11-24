"""Stateful Multi-Turn Trajectory SQL Agent & Vertical SQL Baseline Agent."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.generator import clean_sql_text
from src.agent.policy import Policy
from src.agent.trajectory import AgentTrajectory, ModuleInvocation
from src.benchmarks.tasks import BenchmarkTask
from src.env.sandbox import DialectType, WarehouseSandbox
from src.export.rl_bridge import _extract_sql_from_trajectory
from src.llm.client import JSONParseError, extract_json_object
from src.rl.grpo_data import select_candidate_tables_leak_free
from src.rl.hf_client import LocalHFLLMClient
from src.rl.sft_dataset import format_non_supervised_turn_chatml


class VerticalSQLZeroShotAgent:
    """Evaluates vertical open-source Text-to-SQL models (e.g. `XiYanSQL-QwenCoder-3B/7B-2504`) in their native prompt format.

    Uses XGenerationLab's official Chinese DDL prompt template with leak-free candidate table DDLs so
    vertical SQL models are tested without any artificial JSON wrapper handicap, isolating genuine
    enterprise ODPS partition-gateway & dialect adaptation gaps (`ODPS-0123031`, `MAX_PT`, `LATERAL VIEW`).
    """

    XIYAN_CN_TEMPLATE = (
        "你是一名{dialect_label}专家，现在需要阅读并理解下面的【数据库schema】描述，"
        "以及可能用到的【参考信息】，并运用{dialect_label}知识生成sql语句回答【用户问题】。\n"
        "【用户问题】\n{question}\n\n"
        "【数据库schema】\n{db_schema}\n\n"
        "【参考信息】\n{evidence}\n\n"
        "【用户问题】\n{question}\n\n"
        "```sql\n"
    )

    def __init__(
        self,
        hf_client: LocalHFLLMClient,
        sandbox: WarehouseSandbox,
    ) -> None:
        self.hf_client = hf_client
        self.sandbox = sandbox

    @staticmethod
    def _extract_vertical_sql(raw_output: str) -> str:
        txt = (raw_output or "").strip()
        if not txt:
            return ""
        # Handle continuation from trailing ```sql\n in official template
        if "```" in txt:
            before_close = txt.split("```", 1)[0].strip()
            if before_close.upper().startswith(("SELECT", "WITH", "(")):
                return clean_sql_text(before_close)
        extracted = _extract_sql_from_trajectory(txt)
        if extracted:
            return clean_sql_text(extracted)
        return clean_sql_text(txt)

    async def run_task(
        self,
        task: BenchmarkTask,
        policy: Policy | None = None,
        *,
        dialect: DialectType | None = None,
    ) -> AgentTrajectory:
        active_dialect: DialectType = dialect or self.sandbox.default_dialect
        dialect_label = "MaxCompute ODPS SQL" if active_dialect == "odps" else "Hive SQL"
        selected_tables = select_candidate_tables_leak_free(task.question, task.hint, self.sandbox)
        ddl_text = self.sandbox.catalog.format_table_ddl(selected_tables)

        user_prompt = self.XIYAN_CN_TEMPLATE.format(
            dialect_label=dialect_label,
            question=task.question,
            db_schema=ddl_text,
            evidence=task.hint or "无额外参考信息",
        )
        chatml_prompt = (
            format_non_supervised_turn_chatml({"role": "user", "content": user_prompt})
            + "<|im_start|>assistant\n"
        )
        raw_out, p_toks, c_toks, lat_ms = await self.hf_client.async_generate_from_chatml_prompt(
            chatml_prompt,
            max_new_tokens=896,
            temperature=0.0,
        )
        sql = self._extract_vertical_sql(raw_out)
        exec_res = self.sandbox.run_sql(sql, dialect=active_dialect)
        first_pass_gw = bool(exec_res.get("gateway_passed", False))
        eval_outcome = self.sandbox.evaluate_sql(sql, task, dialect=active_dialect)
        reward_info = self.sandbox.compute_reward(
            sql,
            task,
            refine_turns=0,
            first_pass_gateway=first_pass_gw,
            dialect=active_dialect,
        )

        call = ModuleInvocation(
            module_name="generator",
            step_index=0,
            messages=[{"role": "user", "content": user_prompt}],
            raw_output=raw_out,
            parsed_json={"sql": sql},
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            total_tokens=p_toks + c_toks,
            latency_ms=lat_ms,
        )
        return AgentTrajectory(
            task_id=task.task_id,
            category=task.category,
            split=task.split,
            question=task.question,
            hint=task.hint,
            dialect=active_dialect,
            policy_id=policy.policy_id if policy else "vertical_zeroshot",
            linked_tables=selected_tables,
            probe_requests=[],
            probe_results=[],
            linked_schema_context=ddl_text,
            linker_notes="",
            initial_sql=sql,
            first_pass_gateway=first_pass_gw,
            first_pass_exec=exec_res,
            refine_steps=[],
            final_sql=sql,
            final_exec=exec_res,
            eval_outcome=eval_outcome,
            reward_info=reward_info,
            module_calls=[call],
            conversation_turns=[
                {"role": "user", "module": "generator", "content": user_prompt, "loss_mask": 0},
                {"role": "assistant", "module": "generator", "content": raw_out, "loss_mask": 1},
            ],
        )


class StatefulTrajectorySQLAgent:
    """Multi-turn ChatML SQL Agent aligned with Stage-1 SFT & Stage-2 GRPO trajectory distributions."""

    def __init__(
        self,
        hf_client: LocalHFLLMClient,
        sandbox: WarehouseSandbox,
        max_refine_turns: int = 2,
    ) -> None:
        self.hf_client = hf_client
        self.sandbox = sandbox
        self.max_refine_turns = max_refine_turns

    @staticmethod
    def _parse_json_or_sql(raw_text: str) -> dict[str, Any]:
        try:
            return extract_json_object(raw_text)
        except JSONParseError:
            sql = _extract_sql_from_trajectory(raw_text)
            return {"sql": sql} if sql else {}

    async def run_task(
        self,
        task: BenchmarkTask,
        policy: Policy,
        *,
        dialect: DialectType | None = None,
    ) -> AgentTrajectory:
        active_dialect: DialectType = dialect or self.sandbox.default_dialect
        dialect_upper = active_dialect.upper()
        valid_tables = [t.name for t in self.sandbox.catalog.list_tables()]
        valid_lower_map = {t.lower(): t for t in valid_tables}
        catalog_summary = self.sandbox.catalog.format_catalog_summary()

        turns: list[dict[str, Any]] = []
        module_calls: list[ModuleInvocation] = []

        sys_turn = {
            "role": "system",
            "module": "policy",
            "content": (
                f"[Policy={policy.policy_id} | Dialect={dialect_upper}]\n"
                f"## Linker Policy\n{policy.linker_prompt}\n\n"
                f"## Generator Policy\n{policy.generator_prompt}\n\n"
                f"## Refiner Policy\n{policy.refiner_prompt}"
            ),
            "loss_mask": 0,
        }
        linker_user = {
            "role": "user",
            "module": "linker",
            "content": (
                f"Target Dialect: {dialect_upper}\n"
                f"Business Question: {task.question}\n"
                f"Business Hint: {task.hint or 'None'}\n\n"
                f"Warehouse Catalog Summary:\n{catalog_summary}\n\n"
                "Return JSON to either probe table samples (action='probe') or finalize selected tables (action='finalize')."
            ),
            "loss_mask": 0,
        }
        turns.extend([sys_turn, linker_user])
        chat_history_str = (
            format_non_supervised_turn_chatml(sys_turn)
            + format_non_supervised_turn_chatml(linker_user)
        )

        # 1. Linker Turn 1
        link_out1, p1, c1, lat1 = await self.hf_client.async_generate_from_chatml_prompt(
            chat_history_str + "<|im_start|>assistant\n",
            max_new_tokens=512,
            temperature=0.0,
        )
        parsed_link = self._parse_json_or_sql(link_out1)
        module_calls.append(
            ModuleInvocation(
                module_name="linker",
                step_index=0,
                messages=[{"role": "user", "content": linker_user["content"]}],
                raw_output=link_out1,
                parsed_json=parsed_link,
                prompt_tokens=p1,
                completion_tokens=c1,
                total_tokens=p1 + c1,
                latency_ms=lat1,
            )
        )
        turns.append({"role": "assistant", "module": "linker", "content": link_out1, "loss_mask": 1})
        chat_history_str += f"<|im_start|>assistant\n{link_out1}<|im_end|>\n"

        probe_requests: list[dict[str, Any]] = []
        probe_results: list[dict[str, Any]] = []

        if str(parsed_link.get("action", "")).lower() == "probe" and parsed_link.get("probe_requests"):
            for req in list(parsed_link.get("probe_requests") or [])[:4]:
                if not isinstance(req, dict):
                    continue
                tname = str(req.get("table_name", "")).strip()
                cols = req.get("columns")
                col_list = [str(c) for c in cols] if isinstance(cols, list) else None
                probe_requests.append({"table_name": tname, "columns": col_list})
                probe_results.append(
                    self.sandbox.inspect_table_sample(tname, columns=col_list, sample_rows=3)
                )

            if probe_results:
                tool_obs_text = json.dumps(probe_results, ensure_ascii=False, indent=2)
                tool_turn = {
                    "role": "tool",
                    "name": "inspect_table_sample",
                    "module": "linker",
                    "content": tool_obs_text,
                    "loss_mask": 0,
                }
                turns.append(tool_turn)
                chat_history_str += format_non_supervised_turn_chatml(tool_turn)

                link_out2, p2, c2, lat2 = await self.hf_client.async_generate_from_chatml_prompt(
                    chat_history_str + "<|im_start|>assistant\n",
                    max_new_tokens=512,
                    temperature=0.0,
                )
                parsed_link = self._parse_json_or_sql(link_out2)
                module_calls.append(
                    ModuleInvocation(
                        module_name="linker",
                        step_index=len(module_calls),
                        messages=[{"role": "tool", "content": tool_obs_text}],
                        raw_output=link_out2,
                        parsed_json=parsed_link,
                        prompt_tokens=p2,
                        completion_tokens=c2,
                        total_tokens=p2 + c2,
                        latency_ms=lat2,
                    )
                )
                turns.append(
                    {"role": "assistant", "module": "linker", "content": link_out2, "loss_mask": 1}
                )
                chat_history_str += f"<|im_start|>assistant\n{link_out2}<|im_end|>\n"

        selected_tables: list[str] = []
        seen: set[str] = set()
        for raw_t in parsed_link.get("selected_tables") or []:
            t_norm = re.sub(r"[`\"'\s]", "", str(raw_t)).split(".")[-1].lower()
            if t_norm in valid_lower_map and t_norm not in seen:
                seen.add(t_norm)
                selected_tables.append(valid_lower_map[t_norm])

        if not selected_tables:
            selected_tables = select_candidate_tables_leak_free(
                task.question, task.hint, self.sandbox
            )

        ddl_text = self.sandbox.catalog.format_table_ddl(selected_tables)
        schema_notes = str(parsed_link.get("schema_notes", "")).strip()
        schema_parts = [f"### Selected Table Schemas\n{ddl_text}"]
        if probe_results:
            schema_parts.append(
                "### Sample Inspection Observations\n"
                + json.dumps(probe_results, ensure_ascii=False, indent=2)
            )
        if schema_notes:
            schema_parts.append(f"### Linker Schema Notes\n{schema_notes}")
        linked_schema_text = "\n\n".join(schema_parts)

        # 2. Generator Turn (896 max_new_tokens eliminates tail truncation on complex CTE/window SQL)
        gen_user = {
            "role": "user",
            "module": "generator",
            "content": (
                f"Target Dialect: {dialect_upper}\n"
                f"Business Question: {task.question}\n"
                f"Business Hint: {task.hint or 'None'}\n\n"
                f"{linked_schema_text}\n\n"
                'Generate the single read-only SQL query and return strict JSON: {"thought": "...", "sql": "..."}.'
            ),
            "loss_mask": 0,
        }
        turns.append(gen_user)
        chat_history_str += format_non_supervised_turn_chatml(gen_user)

        gen_out, pg, cg, latg = await self.hf_client.async_generate_from_chatml_prompt(
            chat_history_str + "<|im_start|>assistant\n",
            max_new_tokens=896,
            temperature=0.0,
        )
        parsed_gen = self._parse_json_or_sql(gen_out)
        initial_sql = clean_sql_text(str(parsed_gen.get("sql", "")))
        module_calls.append(
            ModuleInvocation(
                module_name="generator",
                step_index=len(module_calls),
                messages=[{"role": "user", "content": gen_user["content"]}],
                raw_output=gen_out,
                parsed_json=parsed_gen,
                prompt_tokens=pg,
                completion_tokens=cg,
                total_tokens=pg + cg,
                latency_ms=latg,
            )
        )
        turns.append({"role": "assistant", "module": "generator", "content": gen_out, "loss_mask": 1})
        chat_history_str += f"<|im_start|>assistant\n{gen_out}<|im_end|>\n"

        # 3. Execute against Sandbox & Iterative GatewayRefiner
        first_exec = self.sandbox.run_sql(initial_sql, dialect=active_dialect)
        first_pass_gw = bool(first_exec["gateway_passed"])
        current_sql = initial_sql
        current_exec = first_exec
        refine_steps: list[dict[str, Any]] = []

        for r_turn in range(1, self.max_refine_turns + 1):
            suspicious, reason = self.sandbox.is_suspicious_result(current_exec)
            if not suspicious:
                break

            feedback_payload = {
                "gateway_passed": current_exec.get("gateway_passed", False),
                "error_stage": current_exec.get("error_stage", "none"),
                "error_msg": current_exec.get("error_msg", ""),
                "suspicion_reason": reason,
                "row_count": current_exec.get("row_count", 0),
                "sample_rows": [str(r) for r in (current_exec.get("rows") or [])[:5]],
            }
            history_block = ""
            if refine_steps:
                history_block = "\nPrior Repair Attempts:\n" + json.dumps(
                    [
                        {
                            "turn": s["turn"],
                            "diagnosis": s["diagnosis"],
                            "sql": s["sql"],
                            "error": s["exec_res"].get("error_msg") or s["trigger_reason"],
                        }
                        for s in refine_steps
                    ],
                    ensure_ascii=False,
                    indent=2,
                )

            ref_tool_turn = {
                "role": "tool",
                "name": "sandbox_execute",
                "module": "refiner",
                "content": json.dumps(feedback_payload, ensure_ascii=False),
                "loss_mask": 0,
            }
            ref_user_turn = {
                "role": "user",
                "module": "refiner",
                "content": (
                    f"Target Dialect: {dialect_upper}\n"
                    f"Business Question: {task.question}\n"
                    f"Business Hint: {task.hint or 'None'}\n\n"
                    f"{linked_schema_text}\n\n"
                    f"Previous SQL:\n{current_sql}\n\n"
                    f"Sandbox Execution Feedback:\n{json.dumps(feedback_payload, ensure_ascii=False, indent=2)}"
                    f"{history_block}\n\n"
                    'Diagnose the root cause and output strict JSON: {"thought": "...", "diagnosis": "...", "sql": "..."}.'
                ),
                "loss_mask": 0,
            }
            turns.extend([ref_tool_turn, ref_user_turn])
            chat_history_str += (
                format_non_supervised_turn_chatml(ref_tool_turn)
                + format_non_supervised_turn_chatml(ref_user_turn)
            )

            ref_out, pr, cr, latr = await self.hf_client.async_generate_from_chatml_prompt(
                chat_history_str + "<|im_start|>assistant\n",
                max_new_tokens=1024,
                temperature=0.0,
            )
            parsed_ref = self._parse_json_or_sql(ref_out)
            repaired_sql = clean_sql_text(str(parsed_ref.get("sql", ""))) or current_sql
            module_calls.append(
                ModuleInvocation(
                    module_name="refiner",
                    step_index=len(module_calls),
                    messages=[{"role": "user", "content": ref_user_turn["content"]}],
                    raw_output=ref_out,
                    parsed_json=parsed_ref,
                    prompt_tokens=pr,
                    completion_tokens=cr,
                    total_tokens=pr + cr,
                    latency_ms=latr,
                )
            )
            turns.append({"role": "assistant", "module": "refiner", "content": ref_out, "loss_mask": 1})
            chat_history_str += f"<|im_start|>assistant\n{ref_out}<|im_end|>\n"

            new_exec = self.sandbox.run_sql(repaired_sql, dialect=active_dialect)
            refine_steps.append(
                {
                    "turn": r_turn,
                    "trigger_reason": reason,
                    "thought": str(parsed_ref.get("thought", "")),
                    "diagnosis": str(parsed_ref.get("diagnosis", "")),
                    "sql": repaired_sql,
                    "exec_res": new_exec,
                }
            )
            current_sql = repaired_sql
            current_exec = new_exec

        eval_outcome = self.sandbox.evaluate_sql(current_sql, task, dialect=active_dialect)
        reward_info = self.sandbox.compute_reward(
            current_sql,
            task,
            refine_turns=len(refine_steps),
            first_pass_gateway=first_pass_gw,
            dialect=active_dialect,
        )

        return AgentTrajectory(
            task_id=task.task_id,
            category=task.category,
            split=task.split,
            question=task.question,
            hint=task.hint,
            dialect=active_dialect,
            policy_id=policy.policy_id,
            linked_tables=selected_tables,
            probe_requests=probe_requests,
            probe_results=probe_results,
            linked_schema_context=linked_schema_text,
            linker_notes=schema_notes,
            initial_sql=initial_sql,
            first_pass_gateway=first_pass_gw,
            first_pass_exec=first_exec,
            refine_steps=refine_steps,
            final_sql=current_sql,
            final_exec=current_exec,
            eval_outcome=eval_outcome,
            reward_info=reward_info,
            module_calls=module_calls,
            conversation_turns=turns,
        )

    async def evaluate_tasks(
        self,
        tasks: list[BenchmarkTask],
        policy: Policy,
        dialect: DialectType | None = None,
    ) -> list[AgentTrajectory]:
        """Sequentially evaluate tasks on GPU/CPU with deterministic greedy decoding."""
        results: list[AgentTrajectory] = []
        for task in tasks:
            traj = await self.run_task(task, policy=policy, dialect=dialect)
            results.append(traj)
        return results
