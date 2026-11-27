"""Markdown formatter, zero-CDN HTTP server, and CLI entrypoint for `evodw inspect`."""

from __future__ import annotations

import argparse
import asyncio
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.pipeline import MultiModuleSQLAgent
from src.agent.policy import Policy
from src.benchmarks import load_benchmark
from src.config import PROJECT_ROOT, Settings
from src.env.sandbox import DialectType
from src.eval.fixed_budget import resolve_transfer_gepa_policy
from src.inspector.trace_inspector import (
    InspectedTrace,
    inspect_agent_trajectory,
    inspect_sft_jsonl_record,
    render_inspector_html,
    summarize_inspected_traces,
)
from src.llm import DualRoleLLMClient, MissingAPIKeyError, MockWarehouseLLMClient


async def build_showcase_traces(
    *,
    suite: str | None = None,
    task_ids: list[str] | None = None,
    policy_mode: str = "both",
    dialect: DialectType = "odps",
    use_mock: bool = True,
    from_jsonl: Path | None = None,
    limit: int = 12,
    project_root: Path = PROJECT_ROOT,
) -> list[InspectedTrace]:
    """Build inspected traces either from saved JSONL trajectories or side-by-side Seed vs GEPA runs."""
    settings = Settings.from_env(dialect=dialect)

    if from_jsonl is not None:
        bm_core = load_benchmark(seed=2026, dialect=dialect, precompute_gold=True, suite="all")
        try:
            tmap = {t.task_id: t for t in bm_core.all_tasks}
            out: list[InspectedTrace] = []
            with from_jsonl.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    if not raw_line.strip():
                        continue
                    rec = json.loads(raw_line)
                    tid = str(rec.get("task_id", ""))
                    if task_ids and tid not in task_ids:
                        continue
                    if tid not in tmap:
                        continue
                    s_name = "hard_v2" if tid.startswith(("sl_", "scd_", "sc_")) else "v1"
                    if suite and suite != "all" and s_name != suite:
                        continue
                    out.append(inspect_sft_jsonl_record(rec, task=tmap[tid], sandbox=bm_core.sandbox, suite=s_name))
                    if len(out) >= limit:
                        break
            return out
        finally:
            bm_core.sandbox.close()

    seed_policy = Policy.load_seed(project_root / "prompts/seed")
    gepa_policy = resolve_transfer_gepa_policy(dialect=dialect, project_root=project_root)
    llm: object = MockWarehouseLLMClient(dialect=dialect) if use_mock else DualRoleLLMClient(settings=settings)
    policies_to_run = (
        [seed_policy]
        if policy_mode == "seed"
        else ([gepa_policy] if policy_mode == "gepa" else [seed_policy, gepa_policy])
    )

    try:
        if suite in ("v1", "hard_v2", "external_bird_mini"):
            default_tids = (
                ("ja_03", "pp_01")
                if suite == "v1"
                else (("sc_01", "scd_01") if suite == "hard_v2" else ("ext_fin_03", "ext_clin_03", "ext_edu_05"))
            )
            suite_targets: list[tuple[str, tuple[str, ...]]] = [(suite, tuple(task_ids) if task_ids else default_tids)]
        else:
            suite_targets = [
                ("v1", tuple(task_ids) if task_ids else ("ja_03", "pp_01")),
                ("hard_v2", tuple(task_ids) if task_ids else ("sc_01", "scd_01")),
                ("external_bird_mini", tuple(task_ids) if task_ids else ("ext_fin_03", "ext_clin_03")),
            ]

        collected: list[InspectedTrace] = []
        sft_disk = project_root / "outputs/rl_export/sft_openai_tool_use.jsonl"
        if suite is None and not task_ids and sft_disk.exists():
            bm_real = load_benchmark(seed=2026, dialect=dialect, precompute_gold=True, suite="all")
            try:
                tmap_real = {t.task_id: t for t in bm_real.all_tasks}
                found_probe = False
                found_repair = False
                with sft_disk.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        tid = str(rec.get("task_id", ""))
                        if tid not in tmap_real:
                            continue
                        has_p = any(m.get("name") == "inspect_table_sample" for m in (rec.get("messages") or []))
                        is_rep = rec.get("trajectory_type") == "self_repaired_after_error"
                        s_lbl = "hard_v2" if tid.startswith(("sl_", "scd_", "sc_")) else "v1"
                        if has_p and not found_probe:
                            collected.append(inspect_sft_jsonl_record(rec, task=tmap_real[tid], sandbox=bm_real.sandbox, suite=s_lbl))
                            found_probe = True
                        elif is_rep and not found_repair:
                            collected.append(inspect_sft_jsonl_record(rec, task=tmap_real[tid], sandbox=bm_real.sandbox, suite=s_lbl))
                            found_repair = True
                        if found_probe and found_repair:
                            break
            finally:
                bm_real.sandbox.close()

        for s_name, target_tids in suite_targets:
            bm = load_benchmark(seed=2026, dialect=dialect, precompute_gold=True, suite=s_name)  # type: ignore[arg-type]
            try:
                agent = MultiModuleSQLAgent(llm=llm, sandbox=bm.sandbox, max_refine_turns=2)  # type: ignore[arg-type]
                selected = [t for t in bm.all_tasks if t.task_id in target_tids]
                for task in selected:
                    for pol in policies_to_run:
                        traj = await agent.run_task(task, policy=pol, dialect=dialect, evaluate=True)
                        collected.append(inspect_agent_trajectory(traj, task=task, sandbox=bm.sandbox, suite=s_name))
                        if len(collected) >= limit:
                            return collected
            finally:
                bm.sandbox.close()

        return collected[:limit]
    finally:
        if isinstance(llm, DualRoleLLMClient):
            await llm.close()


def render_inspector_markdown(traces: list[InspectedTrace]) -> str:
    """Render concise Markdown summary separating Gateway-Executable from Caliber-Correct SQL."""
    summary = summarize_inspected_traces(traces)
    lines: list[str] = [
        "# EvoDW-RSI SQL Trace Inspector Summary (Executable vs Caliber-Correct)",
        "",
        f"- Total Traces: `{summary['total_traces']}`",
        f"- **语法与网关可执行 (Executable)**: `{summary['executable_count']}/{summary['total_traces']}` (`{summary['executable_rate'] * 100:.1f}%`)",
        f"- **业务口径结果正确 (Result Correct / EX)**: `{summary['result_correct_count']}/{summary['total_traces']}` (`{summary['result_correct_rate'] * 100:.1f}%`)",
        f"- **静默口径陷阱 (Silent Caliber Traps: Executable=PASS & EX=FAIL)**: `{summary['silent_caliber_trap_count']}` (`{summary['silent_caliber_trap_rate'] * 100:.1f}%`)",
        f"- **网关/语法拦截 (Gateway/Syntax Blocks: Executable=FAIL)**: `{summary['gateway_or_syntax_block_count']}`",
        "",
        "| Trace ID | Suite | Policy | Executable | Result Correct (EX) | Verdict Classification | Probes | Refines |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for t in traces:
        dv = t.dual_verdict
        lines.append(
            f"| `{t.trace_id}` | `{t.suite}` | `{t.policy_id}` | "
            f"`{'PASS' if dv['executable'] else 'FAIL'}` | "
            f"`{'PASS' if dv['result_correct'] else 'FAIL'}` | "
            f"{dv['verdict_title']} | {len(t.linker['probe_requests'])} | {t.refinement['refine_turns']} |"
        )
    lines.append("")

    for t in traces:
        dv = t.dual_verdict
        gw = t.generator_and_gateway
        lines.append(f"## `{t.trace_id}` — {dv['verdict_title']}")
        lines.append(f"- **业务问题**: {t.question}")
        lines.append(f"- **双维判定**: `{dv['executable_badge']}` | `{dv['result_correct_badge']}`")
        lines.append(
            f"- **1. Schema Linking & Probing**: 候选 `{t.linker['candidate_table_count']}` 表 -> 选中 `{t.linker['selected_tables']}` | "
            f"探查事实: `{t.linker['discovered_facts'] or '无探查'}`"
        )
        lines.append(
            f"- **2. 首轮网关校验**: `{'PASS' if gw['first_pass_gateway_passed'] else 'BLOCKED'}` "
            f"{gw['first_pass_error_msg'] or ''}"
        )
        lines.append("```sql\n" + (gw["final_sql"] or "(empty)") + "\n```")
        if t.refinement["rounds"]:
            for r in t.refinement["rounds"]:
                lines.append(f"- **3. Refiner Turn {r['turn']} (`{r['trigger_type']}`)**: {r['diagnosis']}")
                lines.append("```diff\n" + r["unified_diff"] + "\n```")
        lines.append(
            f"- **4. 结果集核对**: Pred Rows=`{dv['final_row_count']}` `{dv['pred_sample_rows'][:2]}` vs "
            f"Gold Rows=`{dv['gold_row_count']}` `{dv['gold_sample_rows'][:2]}` (`{dv['diff_reason']}`)"
        )
        lines.append("")
    return "\n".join(lines)


def create_inspector_http_server(
    traces: list[InspectedTrace],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a local HTTP server serving `/` (HTML workbench) and `/api/traces.json`."""
    html_bytes = render_inspector_html(traces).encode("utf-8")
    json_bytes = json.dumps(
        {
            "summary": summarize_inspected_traces(traces),
            "traces": [t.to_dict() for t in traces],
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    class _InspectorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path_clean = self.path.split("?", 1)[0]
            if path_clean in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif path_clean == "/api/traces.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(json_bytes)))
                self.end_headers()
                self.wfile.write(json_bytes)
            else:
                msg = html.escape(f"Not Found: {path_clean}").encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    return ThreadingHTTPServer((host, port), _InspectorHandler)


def serve_inspector_http(
    traces: list[InspectedTrace],
    host: str = "127.0.0.1",
    port: int = 8765,
    console: Console | None = None,
) -> None:
    """Serve the interactive SQL Trace Inspector locally until Ctrl+C."""
    server = create_inspector_http_server(traces, host=host, port=port)
    actual_port = server.server_address[1]
    con = console or Console()
    con.print(
        Panel.fit(
            f"[bold green]EvoDW-RSI SQL Trace Inspector Live Workbench[/bold green]\n"
            f"URL: http://{host}:{actual_port}/\n"
            f"JSON API: http://{host}:{actual_port}/api/traces.json\n"
            "Press Ctrl+C to stop."
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _render_console_summary(console: Console, traces: list[InspectedTrace]) -> None:
    summary = summarize_inspected_traces(traces)
    console.print(
        Panel(
            f"[bold cyan]EvoDW-RSI SQL Trace Inspector (Executable vs Caliber-Correct)[/bold cyan]\n"
            f"Traces: `{summary['total_traces']}` | "
            f"语法与网关可执行 (Executable): [green]{summary['executable_count']}/{summary['total_traces']} ({summary['executable_rate'] * 100:.1f}%)[/green] | "
            f"业务口径结果正确 (EX): [bold green]{summary['result_correct_count']}/{summary['total_traces']} ({summary['result_correct_rate'] * 100:.1f}%)[/bold green]\n"
            f"[yellow]静默口径陷阱 (Executable=PASS & EX=FAIL): {summary['silent_caliber_trap_count']} ({summary['silent_caliber_trap_rate'] * 100:.1f}%)[/yellow] | "
            f"[red]网关/语法拦截 (Executable=FAIL): {summary['gateway_or_syntax_block_count']}[/red]"
        )
    )

    tbl = Table(title="Inspected Query Dual-Verdict Matrix", header_style="bold cyan")
    tbl.add_column("Suite / Task")
    tbl.add_column("Policy")
    tbl.add_column("Probes", justify="right")
    tbl.add_column("Refines", justify="right")
    tbl.add_column("语法与网关可执行")
    tbl.add_column("业务口径正确 (EX)")
    tbl.add_column("Diagnostic Classification")

    for t in traces:
        dv = t.dual_verdict
        exec_cell = (
            "[green]PASS[/green]"
            if dv["executable"]
            else f"[red]FAIL ({dv['final_error_code'] or dv['final_error_stage']})[/red]"
        )
        ex_cell = "[green]PASS[/green]" if dv["result_correct"] else "[red]FAIL[/red]"
        v_style = "yellow" if dv["is_silent_caliber_trap"] else ("green" if dv["result_correct"] else "red")
        tbl.add_row(
            f"{t.suite}:{t.task_id}",
            t.policy_id,
            str(len(t.linker["probe_requests"])),
            str(t.refinement["refine_turns"]),
            exec_cell,
            ex_cell,
            f"[{v_style}]{dv['verdict_title']}[/{v_style}]",
        )
    console.print(tbl)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evodw inspect",
        description="Inspect multi-module SQL traces separating Gateway-Executable from Caliber-Correct (EX).",
    )
    p.add_argument(
        "--suite",
        choices=["v1", "hard_v2", "external_bird_mini"],
        default=None,
        help="Optional single suite filter (default showcases v1 + hard_v2 + external_bird_mini).",
    )
    p.add_argument("--tasks", nargs="+", default=None, help="Optional specific task_id(s) to inspect.")
    p.add_argument(
        "--policy",
        choices=["both", "seed", "gepa"],
        default="both",
        help="Run seed_v0, evolved gepa, or both side-by-side (default: both).",
    )
    p.add_argument("--dialect", choices=["odps", "hive"], default="odps")
    p.add_argument("--from-jsonl", type=Path, default=None, help="Inspect trajectories from an SFT JSONL file.")
    p.add_argument("--limit", type=int, default=14, help="Maximum traces to inspect.")
    p.add_argument("--html-out", type=Path, default=None, help="Write self-contained zero-CDN HTML workbench to path.")
    p.add_argument("--md-out", type=Path, default=None, help="Write Markdown trace report to path.")
    p.add_argument("--json-out", type=Path, default=None, help="Write structured JSON trace report to path.")
    p.add_argument("--serve", action="store_true", help="Launch local HTTP server for interactive inspection.")
    p.add_argument("--host", default="127.0.0.1", help="Bind host for --serve (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8765, help="Bind port for --serve (default: 8765).")
    p.add_argument("--mock", action="store_true", help="Force deterministic offline sandbox execution.")
    p.add_argument("--live", action="store_true", help="Require live OpenAI-compatible API endpoint.")
    return p


async def _run_cli_async(args: argparse.Namespace) -> int:
    console = Console()
    settings = Settings.from_env(dialect=args.dialect)
    use_mock = bool(args.mock) or (not getattr(args, "live", False) and not settings.has_usable_api_key)

    try:
        traces = await build_showcase_traces(
            suite=args.suite,
            task_ids=list(args.tasks) if args.tasks else None,
            policy_mode=args.policy,
            dialect=args.dialect,
            use_mock=use_mock,
            from_jsonl=args.from_jsonl,
            limit=args.limit,
        )
    except MissingAPIKeyError as exc:
        console.print(f"[bold yellow]Configuration required:[/bold yellow] {exc}")
        return 2

    _render_console_summary(console, traces)

    if args.html_out is not None:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(render_inspector_html(traces), encoding="utf-8")
        console.print(f"[green]Wrote zero-CDN HTML workbench -> {args.html_out}[/green]")

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_inspector_markdown(traces), encoding="utf-8")
        console.print(f"[green]Wrote Markdown trace summary -> {args.md_out}[/green]")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summarize_inspected_traces(traces),
            "traces": [t.to_dict() for t in traces],
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Wrote JSON trace report -> {args.json_out}[/green]")

    if args.serve:
        serve_inspector_http(traces, host=args.host, port=args.port, console=console)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_run_cli_async(args))
