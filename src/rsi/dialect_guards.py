"""Negation-aware ODPS & Hive prompt dialect violation detection and auto-repair."""

from __future__ import annotations

import re

from src.env.gateway_linter import FORBIDDEN_FUNCTIONS_COMMON, FORBIDDEN_FUNCTIONS_ODPS

# Canonical ODPS replacements for functions blocked by GatewayLinter (FORBIDDEN_FUNCTIONS_COMMON | FORBIDDEN_FUNCTIONS_ODPS).
ODPS_FORBIDDEN_FUNCTION_SPECS: dict[str, str] = {
    "DATE_FORMAT": "TO_CHAR(datetime, 'yyyymmdd')",
    "DATE_ADD": "DATEADD(datetime, delta, 'dd')",
    "DATE_SUB": "DATEADD(datetime, -delta, 'dd')",
    "DATE_DIFF": "DATEDIFF(end_date, start_date, 'dd')",
    "IFNULL": "NVL(expr, default)",
    "JSON_EXTRACT": "GET_JSON_OBJECT(col, '$.key')",
    "JSON_EXTRACT_SCALAR": "GET_JSON_OBJECT(col, '$.key')",
    "JSON_VALUE": "GET_JSON_OBJECT(col, '$.key')",
    "STRFTIME": "TO_CHAR(datetime, 'yyyymmdd')",
    "TRY_STRPTIME": "TO_DATE(date_str, 'yyyymmdd')",
    "UNNEST": "LATERAL VIEW explode(arr_col) t AS item",
    "GROUP_CONCAT": "WM_CONCAT(',', expr)",
    "ILIKE": "LOWER(col) LIKE LOWER(pattern)",
    "STR_SPLIT": "SPLIT(str, pattern)",
    "STRING_SPLIT": "SPLIT(str, pattern)",
    "STR_SPLIT_REGEX": "SPLIT(str, pattern)",
    "READ_CSV": "<catalog_table>",
    "READ_CSV_AUTO": "<catalog_table>",
    "READ_PARQUET": "<catalog_table>",
    "READ_JSON": "<catalog_table>",
    "READ_JSON_AUTO": "<catalog_table>",
}

HIVE_FORBIDDEN_FUNCTION_SPECS: dict[str, str] = {
    "DATEADD": "DATE_ADD(start_date, num_days)",
    "TO_CHAR": "DATE_FORMAT(ts, 'yyyy-MM-dd')",
    "MAX_PT": "probed max_partition literal from inspect_table_sample",
    "IFNULL": "NVL(expr, default)",
    "JSON_EXTRACT": "GET_JSON_OBJECT(col, '$.key')",
    "JSON_EXTRACT_SCALAR": "GET_JSON_OBJECT(col, '$.key')",
    "JSON_VALUE": "GET_JSON_OBJECT(col, '$.key')",
    "STRFTIME": "DATE_FORMAT(ts, 'yyyyMMdd')",
    "TRY_STRPTIME": "TO_DATE(date_str)",
    "UNNEST": "LATERAL VIEW explode(arr_col) t AS item",
    "DATE_DIFF": "DATEDIFF(end_date, start_date)",
}

# Ensure parity with GatewayLinter sets.
for _fn in FORBIDDEN_FUNCTIONS_COMMON | FORBIDDEN_FUNCTIONS_ODPS:
    ODPS_FORBIDDEN_FUNCTION_SPECS.setdefault(_fn, "ODPS built-in equivalent")

_NEGATION_PREFIX_CUES = (
    "never ",
    "do not ",
    "don't ",
    ", not ",
    "or not ",
    "and not ",
    "not use",
    "not call",
    "not emit",
    "not write",
    "not supported",
    "unsupported",
    "avoid ",
    "instead of ",
    "rather than ",
    "replace ",
    "replacing ",
    "prohibit",
    "forbid",
    "disallow",
    "reject",
    "fix ",
    "ban ",
    "不要",
    "禁止",
    "避免",
    "而非",
    "替代",
    "不支持",
    "禁用",
)

_NEGATION_SUFFIX_CUES = (
    "is unsupported",
    "are unsupported",
    "is not supported",
    "are not supported",
    "is forbidden",
    "are forbidden",
    "is prohibited",
    "are prohibited",
    "is invalid",
    "is disallowed",
    "will fail",
    "will cause",
    "causes a",
    "triggers a",
    "to dateadd",
    "to to_char",
    "to nvl",
    "to coalesce",
    "to get_json_object",
    "to lateral view",
    "with nvl",
    "with coalesce",
    "with dateadd",
    "with to_char",
    "with get_json_object",
    "不支持",
    "被禁用",
    "会报错",
)

_SUBCLAUSE_RESET_RE = re.compile(
    r"(?:;|—|–|\binstead\s+use\b|\binstead\s*,|\bhowever\b|\bbut\s+use\b)",
    flags=re.IGNORECASE,
)


def _split_clauses_preserving_code(line: str) -> list[str]:
    """Split a single line on `. ` or `; ` outside backticks and quotes."""
    clauses: list[str] = []
    cur: list[str] = []
    in_tick = False
    in_squote = False
    n = len(line)
    i = 0
    while i < n:
        ch = line[i]
        if ch == "`" and not in_squote:
            in_tick = not in_tick
            cur.append(ch)
        elif ch == "'" and not in_tick:
            in_squote = not in_squote
            cur.append(ch)
        elif not in_tick and not in_squote and ch in ".;。；" and (i + 1 == n or line[i + 1].isspace()):
            cur.append(ch)
            clauses.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        clauses.append("".join(cur))
    return clauses


def _is_negated_occurrence(clause: str, start: int, end: int) -> bool:
    """Return True if the function mention at `clause[start:end]` is an explicit negative/prohibitive warning."""
    raw_prefix = clause[:start]
    # Truncate prefix at any sub-clause reset boundary (e.g. `; instead use ...`).
    last_reset = 0
    for m in _SUBCLAUSE_RESET_RE.finditer(raw_prefix):
        last_reset = m.end()
    prefix = raw_prefix[last_reset:].lower()
    prefix_clean = prefix.replace("`", "").rstrip()
    suffix = clause[end : min(len(clause), end + 90)].lower()

    if prefix_clean.endswith(" not") or prefix_clean.endswith(",not"):
        return True
    if any(cue in prefix for cue in _NEGATION_PREFIX_CUES):
        return True
    if any(cue in suffix for cue in _NEGATION_SUFFIX_CUES):
        return True
    return False


def _extract_balanced_call(text: str, lparen_idx: int) -> tuple[list[str], int] | None:
    """Parse balanced `(arg1, arg2, ...)` starting at `lparen_idx`, returning `(args, rparen_idx)`."""
    if lparen_idx >= len(text) or text[lparen_idx] != "(":
        return None
    depth = 0
    in_sq = False
    in_dq = False
    args: list[str] = []
    cur: list[str] = []
    for idx in range(lparen_idx, len(text)):
        ch = text[idx]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            cur.append(ch)
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            cur.append(ch)
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
                if depth > 1:
                    cur.append(ch)
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    arg_str = "".join(cur).strip()
                    if arg_str or args:
                        args.append(arg_str)
                    return args, idx
                cur.append(ch)
            elif ch == "," and depth == 1:
                args.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        else:
            cur.append(ch)
    return None


def _normalize_odps_date_fmt(fmt_expr: str) -> str:
    """Convert Hive/strftime format literals (e.g. `'yyyyMMdd'`, `'%Y%m%d'`) to ODPS `'yyyymmdd'`."""
    s = fmt_expr.strip()
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        q = s[0]
        inner = s[1:-1]
        inner = (
            inner.replace("%Y", "yyyy")
            .replace("%m", "mm")
            .replace("%d", "dd")
            .replace("%H", "hh")
            .replace("%M", "mi")
            .replace("%S", "ss")
        )
        inner = inner.replace("YYYY", "yyyy").replace("MM", "mm").replace("DD", "dd")
        return f"{q}{inner}{q}"
    return s


def detect_prompt_dialect_violations(prompt_text: str, dialect: str = "odps") -> list[str]:
    """Scan backtick SQL snippets, fenced blocks, and prose for non-negated forbidden dialect functions."""
    if not prompt_text or not prompt_text.strip():
        return []

    target_dialect = dialect.strip().lower()
    spec_map = ODPS_FORBIDDEN_FUNCTION_SPECS if target_dialect == "odps" else HIVE_FORBIDDEN_FUNCTION_SPECS
    func_names = sorted(spec_map.keys(), key=len, reverse=True)
    fn_pattern = re.compile(r"\b(" + "|".join(re.escape(f) for f in func_names) + r")\b", flags=re.IGNORECASE)
    datediff_call_re = re.compile(r"\bDATEDIFF\s*\(", flags=re.IGNORECASE)

    violations: list[str] = []
    seen: set[str] = set()

    for line in prompt_text.splitlines():
        for clause in _split_clauses_preserving_code(line):
            for m in fn_pattern.finditer(clause):
                fn_upper = m.group(1).upper()
                # Ignore lowercase English prose word `unnest` unless uppercase, in backticks, or followed by `(`.
                if fn_upper == "UNNEST":
                    after = clause[m.end() :].lstrip()
                    in_backticks = clause[: m.start()].count("`") % 2 == 1
                    if not (m.group(1).isupper() or in_backticks or after.startswith("(")):
                        continue
                if not _is_negated_occurrence(clause, m.start(), m.end()):
                    if fn_upper not in seen:
                        seen.add(fn_upper)
                        violations.append(fn_upper)

            if target_dialect == "odps":
                for dm in datediff_call_re.finditer(clause):
                    lparen = clause.find("(", dm.start())
                    parsed = _extract_balanced_call(clause, lparen) if lparen != -1 else None
                    if parsed is not None:
                        args, rparen = parsed
                        if len(args) == 2 and not _is_negated_occurrence(clause, dm.start(), rparen + 1):
                            if "DATEDIFF_2ARG" not in seen:
                                seen.add("DATEDIFF_2ARG")
                                violations.append("DATEDIFF_2ARG")
            else:
                for dm in datediff_call_re.finditer(clause):
                    lparen = clause.find("(", dm.start())
                    parsed = _extract_balanced_call(clause, lparen) if lparen != -1 else None
                    if parsed is not None:
                        args, rparen = parsed
                        if len(args) == 3 and not _is_negated_occurrence(clause, dm.start(), rparen + 1):
                            if "DATEDIFF_3ARG" not in seen:
                                seen.add("DATEDIFF_3ARG")
                                violations.append("DATEDIFF_3ARG")

    return violations


def _rewrite_single_odps_call(fn_upper: str, args: list[str]) -> str | None:
    """Rewrite a single forbidden ODPS function call into canonical ODPS SQL."""
    if fn_upper == "DATE_FORMAT":
        if len(args) >= 2:
            return f"TO_CHAR({args[0]}, {_normalize_odps_date_fmt(args[1])})"
        if len(args) == 1:
            return f"TO_CHAR({args[0]}, 'yyyymmdd')"
    elif fn_upper == "STRFTIME":
        if len(args) >= 2:
            if args[0].strip().startswith(("'%", '"%')):
                return f"TO_CHAR({args[1]}, {_normalize_odps_date_fmt(args[0])})"
            return f"TO_CHAR({args[0]}, {_normalize_odps_date_fmt(args[1])})"
    elif fn_upper == "TRY_STRPTIME":
        if len(args) >= 2:
            return f"TO_DATE({args[0]}, {_normalize_odps_date_fmt(args[1])})"
        if len(args) == 1:
            return f"TO_DATE({args[0]}, 'yyyymmdd')"
    elif fn_upper == "DATE_ADD":
        if len(args) == 2:
            return f"DATEADD({args[0]}, {args[1]}, 'dd')"
        if len(args) == 3:
            return f"DATEADD({args[0]}, {args[1]}, {args[2]})"
    elif fn_upper == "DATE_SUB":
        if len(args) == 2:
            delta = args[1].strip()
            neg_delta = f"-{delta}" if re.fullmatch(r"\d+|\w+", delta) else f"-({delta})"
            return f"DATEADD({args[0]}, {neg_delta}, 'dd')"
    elif fn_upper == "DATE_DIFF":
        if len(args) == 3 and args[0].strip().startswith(("'", '"')):
            return f"DATEDIFF({args[2]}, {args[1]}, 'dd')"
        if len(args) == 2:
            return f"DATEDIFF({args[0]}, {args[1]}, 'dd')"
    elif fn_upper == "DATEDIFF":
        if len(args) == 2:
            return f"DATEDIFF({args[0]}, {args[1]}, 'dd')"
    elif fn_upper == "IFNULL":
        if len(args) == 2:
            return f"NVL({args[0]}, {args[1]})"
    elif fn_upper in ("JSON_EXTRACT", "JSON_EXTRACT_SCALAR", "JSON_VALUE"):
        if len(args) >= 2:
            return f"GET_JSON_OBJECT({args[0]}, {args[1]})"
    elif fn_upper == "UNNEST":
        if len(args) >= 1:
            return f"LATERAL VIEW explode({args[0]}) t AS item"
    elif fn_upper in ("STR_SPLIT", "STRING_SPLIT", "STR_SPLIT_REGEX"):
        if len(args) >= 2:
            return f"SPLIT({args[0]}, {args[1]})"
    elif fn_upper == "GROUP_CONCAT":
        if len(args) == 1:
            return f"WM_CONCAT(',', {args[0]})"
        if len(args) >= 2:
            return f"WM_CONCAT({args[1]}, {args[0]})"
    return None


def _repair_clause_odps(clause: str) -> tuple[str, list[str]]:
    """Repair non-negated forbidden ODPS function calls and standalone mentions inside a single clause."""
    repaired_fns: list[str] = []
    cur = clause

    # Handle `CROSS JOIN UNNEST(arr)` idiom first if non-negated.
    cj_unnest_re = re.compile(
        r"\bCROSS\s+JOIN\s+UNNEST\s*\(([^()]+)\)(?:\s+(?:AS\s+)?\w+\s*(?:\([^()]+\))?)?",
        flags=re.IGNORECASE,
    )
    m_cju = cj_unnest_re.search(cur)
    if m_cju and not _is_negated_occurrence(cur, m_cju.start(), m_cju.end()):
        arr_expr = m_cju.group(1).strip()
        cur = cur[: m_cju.start()] + f"LATERAL VIEW explode({arr_expr}) t AS item" + cur[m_cju.end() :]
        repaired_fns.append("UNNEST")

    call_names = sorted(set(ODPS_FORBIDDEN_FUNCTION_SPECS.keys()) | {"DATEDIFF"}, key=len, reverse=True)
    call_re = re.compile(r"\b(" + "|".join(re.escape(f) for f in call_names) + r")\s*\(", flags=re.IGNORECASE)

    # Iteratively rewrite calls (handling nested calls cleanly).
    for _ in range(8):
        changed = False
        # Find the rightmost (innermost) rewritable call first so inner args are already ODPS-clean.
        matches = list(call_re.finditer(cur))
        for m in reversed(matches):
            fn_upper = m.group(1).upper()
            lparen = cur.find("(", m.start())
            parsed = _extract_balanced_call(cur, lparen) if lparen != -1 else None
            if parsed is None:
                continue
            args, rparen = parsed
            if fn_upper == "DATEDIFF" and len(args) != 2:
                continue
            if _is_negated_occurrence(cur, m.start(), rparen + 1):
                continue
            replacement = _rewrite_single_odps_call(fn_upper, args)
            if replacement is not None and replacement != cur[m.start() : rparen + 1]:
                cur = cur[: m.start()] + replacement + cur[rparen + 1 :]
                tag = "DATEDIFF_2ARG" if fn_upper == "DATEDIFF" else fn_upper
                if tag not in repaired_fns:
                    repaired_fns.append(tag)
                changed = True
                break
        if not changed:
            break

    # Repair any remaining non-negated standalone forbidden function names (e.g. ``use `DATE_FORMAT` ``).
    standalone_names = sorted(ODPS_FORBIDDEN_FUNCTION_SPECS.keys(), key=len, reverse=True)
    standalone_re = re.compile(
        r"(`?)\b(" + "|".join(re.escape(f) for f in standalone_names) + r")\b(`?)",
        flags=re.IGNORECASE,
    )

    for _ in range(8):
        changed = False
        for m in standalone_re.finditer(cur):
            fn_upper = m.group(2).upper()
            if fn_upper == "UNNEST":
                in_ticks = bool(m.group(1) and m.group(3)) or (cur[: m.start()].count("`") % 2 == 1)
                if not (m.group(2).isupper() or in_ticks):
                    continue
            if _is_negated_occurrence(cur, m.start(), m.end()):
                continue
            spec = ODPS_FORBIDDEN_FUNCTION_SPECS[fn_upper]
            in_existing_ticks = (cur[: m.start()].count("`") % 2 == 1) and not (m.group(1) and m.group(3))
            repl = spec if in_existing_ticks else f"`{spec}`"
            cur = cur[: m.start()] + repl + cur[m.end() :]
            if fn_upper not in repaired_fns:
                repaired_fns.append(fn_upper)
            changed = True
            break
        if not changed:
            break

    if "DATE_FORMAT" in repaired_fns or "STRFTIME" in repaired_fns:
        cur = cur.replace("yyyyMMdd", "yyyymmdd").replace("yyyy-MM-dd", "yyyy-mm-dd")

    return cur, repaired_fns


def repair_prompt_dialect_violations(
    prompt_text: str,
    dialect: str = "odps",
) -> tuple[str, list[str]]:
    """Detect and auto-repair forbidden dialect SQL examples/function mentions into target dialect conventions."""
    initial_violations = detect_prompt_dialect_violations(prompt_text, dialect=dialect)
    if not initial_violations:
        return prompt_text, []

    target_dialect = dialect.strip().lower()
    if target_dialect != "odps":
        def _rewrite_single_hive_call(fn_upper: str, args: list[str]) -> str | None:
            if fn_upper == "DATEADD" and len(args) >= 2:
                return f"DATE_ADD({args[0]}, {args[1]})"
            if fn_upper == "TO_CHAR":
                if len(args) >= 2:
                    fmt = args[1].replace("yyyymmdd", "yyyyMMdd").replace("yyyy-mm-dd", "yyyy-MM-dd")
                    return f"DATE_FORMAT({args[0]}, {fmt})"
                if len(args) == 1:
                    return f"DATE_FORMAT({args[0]}, 'yyyy-MM-dd')"
            if fn_upper == "MAX_PT":
                return "'<probed_max_partition>'"
            if fn_upper == "IFNULL" and len(args) >= 2:
                return f"NVL({args[0]}, {args[1]})"
            if fn_upper in ("JSON_EXTRACT", "JSON_EXTRACT_SCALAR", "JSON_VALUE") and len(args) >= 2:
                return f"GET_JSON_OBJECT({args[0]}, {args[1]})"
            if fn_upper == "STRFTIME" and len(args) >= 2:
                return f"DATE_FORMAT({args[-1]}, 'yyyyMMdd')"
            if fn_upper == "TRY_STRPTIME" and len(args) >= 1:
                return f"TO_DATE({args[0]})"
            if fn_upper == "DATE_DIFF" and len(args) >= 2:
                return f"DATEDIFF({args[-1]}, {args[-2]})"
            if fn_upper == "UNNEST" and len(args) >= 1:
                return f"LATERAL VIEW explode({args[0]}) t AS item"
            if fn_upper == "DATEDIFF" and len(args) == 3:
                return f"DATEDIFF({args[0]}, {args[1]})"
            return None

        hive_call_names = sorted(list(HIVE_FORBIDDEN_FUNCTION_SPECS.keys()) + ["DATEDIFF"], key=len, reverse=True)
        hive_call_re = re.compile(
            r"\b(" + "|".join(re.escape(f) for f in hive_call_names) + r")\s*\(",
            flags=re.IGNORECASE,
        )
        hive_standalone_names = sorted(HIVE_FORBIDDEN_FUNCTION_SPECS.keys(), key=len, reverse=True)
        hive_standalone_re = re.compile(
            r"(`?)\b(" + "|".join(re.escape(f) for f in hive_standalone_names) + r")\b(`?)",
            flags=re.IGNORECASE,
        )

        def _repair_clause_hive(clause: str) -> tuple[str, list[str]]:
            cur = clause
            rep_fns: list[str] = []
            for _ in range(12):
                changed = False
                for m in hive_call_re.finditer(cur):
                    fn_up = m.group(1).upper()
                    lp = cur.find("(", m.start())
                    parsed = _extract_balanced_call(cur, lp) if lp != -1 else None
                    if parsed is None:
                        continue
                    args, rp = parsed
                    if fn_up == "DATEDIFF" and len(args) != 3:
                        continue
                    if _is_negated_occurrence(cur, m.start(), rp + 1):
                        continue
                    repl = _rewrite_single_hive_call(fn_up, args)
                    if repl is not None and repl != cur[m.start() : rp + 1]:
                        cur = cur[: m.start()] + repl + cur[rp + 1 :]
                        tag = "DATEDIFF_3ARG" if fn_up == "DATEDIFF" else fn_up
                        if tag not in rep_fns:
                            rep_fns.append(tag)
                        changed = True
                        break
                if not changed:
                    break
            for _ in range(8):
                changed = False
                for m in hive_standalone_re.finditer(cur):
                    fn_up = m.group(2).upper()
                    if fn_up == "UNNEST":
                        in_ticks = bool(m.group(1) and m.group(3)) or (cur[: m.start()].count("`") % 2 == 1)
                        if not (m.group(2).isupper() or in_ticks):
                            continue
                    if _is_negated_occurrence(cur, m.start(), m.end()):
                        continue
                    spec = HIVE_FORBIDDEN_FUNCTION_SPECS[fn_up]
                    in_existing_ticks = (cur[: m.start()].count("`") % 2 == 1) and not (m.group(1) and m.group(3))
                    repl = spec if in_existing_ticks else f"`{spec}`"
                    cur = cur[: m.start()] + repl + cur[m.end() :]
                    if fn_up not in rep_fns:
                        rep_fns.append(fn_up)
                    changed = True
                    break
                if not changed:
                    break
            return cur, rep_fns

        h_lines: list[str] = []
        for line in prompt_text.splitlines():
            h_clauses = [_repair_clause_hive(cl)[0] for cl in _split_clauses_preserving_code(line)]
            h_lines.append("".join(h_clauses))
        return "\n".join(h_lines), initial_violations

    out_lines: list[str] = []
    all_repaired: list[str] = []
    for line in prompt_text.splitlines():
        clauses = _split_clauses_preserving_code(line)
        new_clauses: list[str] = []
        for cl in clauses:
            fixed_cl, rep = _repair_clause_odps(cl)
            new_clauses.append(fixed_cl)
            for r in rep:
                if r not in all_repaired:
                    all_repaired.append(r)
        out_lines.append("".join(new_clauses))

    repaired_text = "\n".join(out_lines)

    # Append explicit ODPS compliance note for the repaired functions if not already warned against.
    warn_items: list[str] = []
    for fn in initial_violations:
        if fn == "DATEDIFF_2ARG":
            warn_items.append("3-arg `DATEDIFF(end_date, start_date, 'dd')` instead of 2-arg `DATEDIFF`")
        elif fn in ODPS_FORBIDDEN_FUNCTION_SPECS:
            warn_items.append(f"`{ODPS_FORBIDDEN_FUNCTION_SPECS[fn]}` instead of `{fn}`")
    if warn_items:
        guard_line = (
            "- ODPS dialect compliance: Do not use unsupported functions; always use "
            + "; ".join(warn_items)
            + "."
        )
        # Insert right before JSON output contract footer if present, else append.
        anchor = "Output strictly valid JSON"
        if anchor in repaired_text and guard_line not in repaired_text:
            idx = repaired_text.index(anchor)
            repaired_text = repaired_text[:idx].rstrip() + "\n" + guard_line + "\n\n" + repaired_text[idx:]
        elif guard_line not in repaired_text:
            repaired_text = repaired_text.rstrip() + "\n" + guard_line

    return repaired_text, initial_violations
