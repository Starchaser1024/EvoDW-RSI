"""Layer 1: Production-grade static AST & dialect gateway linter for ODPS and Hive SQL."""

from __future__ import annotations

from typing import Literal

import sqlglot
from sqlglot import Token, TokenType, exp

from src.env.partition_pruning import LintResult, PartitionPruningMixin
from src.env.schema import WarehouseCatalog

DialectType = Literal["odps", "hive"]

FORBIDDEN_FUNCTIONS_COMMON = {
    "STRFTIME",
    "JSON_EXTRACT",
    "JSON_EXTRACT_SCALAR",
    "JSON_VALUE",
    "IFNULL",
    "GROUP_CONCAT",
    "UNNEST",
    "ILIKE",
    "STR_SPLIT",
    "STRING_SPLIT",
    "STR_SPLIT_REGEX",
    "DATE_DIFF",
    "TRY_STRPTIME",
    "READ_CSV",
    "READ_CSV_AUTO",
    "READ_PARQUET",
    "READ_JSON",
    "READ_JSON_AUTO",
}

FORBIDDEN_FUNCTIONS_ODPS = {
    "DATE_ADD",
    "DATE_SUB",
    "DATE_FORMAT",
    "UNIX_TIMESTAMP",
    "FROM_UNIXTIME",
}

FORBIDDEN_FUNCTIONS_HIVE = {
    "TO_CHAR",
}

VALID_ODPS_DATEPARTS = {"yyyy", "yy", "year", "mm", "mon", "month", "dd", "day", "hh", "hour", "mi", "minute", "ss", "second"}

_FUNC_NAME_TOKEN_TYPES = {
    tt
    for tt_name in (
        "VAR",
        "LEFT",
        "RIGHT",
        "REPLACE",
        "YEAR",
        "MONTH",
        "DAY",
        "DATE",
        "TIMESTAMP",
        "STRUCT",
        "ARRAY",
        "MAP",
        "UNNEST",
        "ILIKE",
        "CURRENT_DATE",
        "CURRENT_TIMESTAMP",
    )
    if (tt := getattr(TokenType, tt_name, None)) is not None
}

# SQL keywords that immediately precede `(` without being function calls.
_NON_FUNCTION_KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "EXISTS",
    "ON",
    "USING",
    "AS",
    "OVER",
    "PARTITION",
    "ORDER",
    "GROUP",
    "HAVING",
    "VALUES",
    "WHEN",
    "THEN",
    "ELSE",
    "CASE",
    "UNION",
    "ALL",
    "DISTINCT",
    "LATERAL",
    "VIEW",
    "OUTER",
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "CROSS",
    "WITH",
    "BETWEEN",
    "ANY",
    "SOME",
    "ROW",
    "ROWS",
    "RANGE",
    "FILTER",
    "WITHIN",
    "ARRAY",
    "STRUCT",
    "MAP",
    "DECIMAL",
    "NUMERIC",
    "VARCHAR",
    "CHAR",
}


class GatewayLinter(PartitionPruningMixin):
    """Static ODPS/Hive gateway linter enforcing partition pruning, dialect safety, and join hygiene."""

    def __init__(self, catalog: WarehouseCatalog, dialect: DialectType = "odps") -> None:
        self.catalog = catalog
        self.dialect: DialectType = dialect

    def lint(self, sql: str) -> LintResult:
        if not sql or not sql.strip():
            return LintResult(
                passed=False,
                error_code="ODPS-0130161",
                error_msg="FAILED: ODPS-0130161: Parse exception - empty SQL statement.",
            )

        cleaned_sql = sql.strip().rstrip(";").strip()
        if not cleaned_sql:
            return LintResult(
                passed=False,
                error_code="ODPS-0130161",
                error_msg="FAILED: ODPS-0130161: Parse exception - empty SQL statement.",
            )

        # 1. Tokenize (comments stripped automatically by sqlglot tokenizer).
        try:
            tokens = [
                t
                for t in sqlglot.tokenize(cleaned_sql, read="hive")
                if t.token_type != TokenType.COMMENT
            ]
        except Exception as exc:
            return LintResult(
                passed=False,
                error_code="ODPS-0130161",
                error_msg=f"FAILED: ODPS-0130161: Parse exception - tokenization error: {exc}",
            )

        # 2. Token-stream checks (catches constructs normalized away by sqlglot AST parser).
        tok_res = self._check_tokens(tokens)
        if not tok_res.passed:
            return tok_res

        # 3. Parse AST using Hive dialect (super-set of ODPS/Hive grammar).
        try:
            trees = sqlglot.parse(cleaned_sql, read="hive")
        except Exception as exc:
            return LintResult(
                passed=False,
                error_code="ODPS-0130161",
                error_msg=f"FAILED: ODPS-0130161: Parse exception - syntax error: {exc}",
            )

        if len(trees) != 1 or trees[0] is None:
            return LintResult(
                passed=False,
                error_code="ODPS-0130161",
                error_msg="FAILED: ODPS-0130161: Parse exception - only a single SELECT query is allowed.",
            )

        root = trees[0]
        if not isinstance(root, (exp.Select, exp.Union, exp.Subquery)):
            return LintResult(
                passed=False,
                error_code="ODPS-0130071",
                error_msg=(
                    "FAILED: ODPS-0130071: Semantic analysis exception - "
                    f"only read-only SELECT queries are permitted (got {type(root).__name__})."
                ),
            )

        # 4. AST join hygiene (no implicit Cartesian products / missing ON).
        join_res = self._check_joins(root)
        if not join_res.passed:
            return join_res

        # 5. AST partition pruning validation across all CTEs, subqueries, and selects.
        part_res = self._check_partition_pruning(root)
        if not part_res.passed:
            return part_res

        return LintResult(passed=True)

    def _check_tokens(self, tokens: list[Token]) -> LintResult:
        n = len(tokens)
        depth = 0
        for i, tok in enumerate(tokens):
            tt = tok.token_type
            txt_upper = tok.text.upper()

            if tt == TokenType.L_PAREN:
                depth += 1
            elif tt == TokenType.R_PAREN:
                depth = max(0, depth - 1)

            # Reject Postgres `::` cast operator.
            if tt == TokenType.DCOLON:
                return LintResult(
                    passed=False,
                    error_code="ODPS-0130161",
                    error_msg=(
                        "FAILED: ODPS-0130161: Parse exception - PostgreSQL '::' cast operator "
                        f"is not supported in {self.dialect.upper()}; use CAST(expr AS type)."
                    ),
                )

            # Reject DuckDB/Postgres JSON arrow operators (`->` / `->>`) when extracting JSON keys.
            if tt == TokenType.DARROW or (
                tt == TokenType.ARROW
                and i + 1 < n
                and tokens[i + 1].token_type in (TokenType.STRING, TokenType.NUMBER)
            ):
                return LintResult(
                    passed=False,
                    error_code="ODPS-0130161",
                    error_msg=(
                        "FAILED: ODPS-0130161: Parse exception - JSON operator "
                        f"'{tok.text}' is not supported in {self.dialect.upper()}; "
                        "use GET_JSON_OBJECT(col, '$.key')."
                    ),
                )

            # Reject MySQL comma LIMIT: `LIMIT offset, count`.
            if tt == TokenType.LIMIT:
                limit_depth = depth
                j = i + 1
                sub_depth = 0
                while j < n:
                    nt = tokens[j].token_type
                    if nt == TokenType.L_PAREN:
                        sub_depth += 1
                    elif nt == TokenType.R_PAREN:
                        if sub_depth == 0:
                            break
                        sub_depth -= 1
                    elif sub_depth == 0:
                        if nt in (
                            TokenType.OFFSET,
                            TokenType.UNION,
                            TokenType.EXCEPT,
                            TokenType.INTERSECT,
                            TokenType.SEMICOLON,
                        ):
                            break
                        if nt == TokenType.COMMA:
                            return LintResult(
                                passed=False,
                                error_code="ODPS-0130161",
                                error_msg=(
                                    "FAILED: ODPS-0130161: Parse exception - MySQL style "
                                    "'LIMIT offset, count' is not supported in ODPS/Hive; "
                                    "use 'LIMIT count' or 'LIMIT count OFFSET offset'."
                                ),
                            )
                    j += 1
                _ = limit_depth

            # Inspect function calls `FUNC(...)`.
            if (
                i + 1 < n
                and tokens[i + 1].token_type == TokenType.L_PAREN
                and (tt in _FUNC_NAME_TOKEN_TYPES or txt_upper.isidentifier())
                and txt_upper not in _NON_FUNCTION_KEYWORDS
            ):
                if i > 0 and tokens[i - 1].token_type == TokenType.DOT:
                    continue

                args = self._extract_call_args(tokens, i + 1)
                if args is None:
                    return LintResult(
                        passed=False,
                        error_code="ODPS-0130161",
                        error_msg=f"FAILED: ODPS-0130161: Parse exception - unmatched parenthesis in {txt_upper}(...).",
                    )

                func_err = self._validate_function_call(txt_upper, args)
                if func_err is not None:
                    return func_err

        return LintResult(passed=True)

    @staticmethod
    def _extract_call_args(tokens: list[Token], lparen_idx: int) -> list[list[Token]] | None:
        """Extract top-level argument token lists for the call starting at `lparen_idx`."""
        depth = 0
        args: list[list[Token]] = []
        current: list[Token] = []
        n = len(tokens)
        for idx in range(lparen_idx, n):
            tok = tokens[idx]
            if tok.token_type == TokenType.L_PAREN:
                depth += 1
                if depth > 1:
                    current.append(tok)
            elif tok.token_type == TokenType.R_PAREN:
                depth -= 1
                if depth == 0:
                    if current or args:
                        args.append(current)
                    return args
                current.append(tok)
            elif tok.token_type == TokenType.COMMA and depth == 1:
                args.append(current)
                current = []
            else:
                current.append(tok)
        return None

    def _validate_function_call(self, func_name: str, args: list[list[Token]]) -> LintResult | None:
        if func_name in FORBIDDEN_FUNCTIONS_COMMON:
            return LintResult(
                passed=False,
                error_code="ODPS-0130071",
                error_msg=(
                    f"FAILED: ODPS-0130071: Semantic analysis exception - function '{func_name}' "
                    f"is not supported in {self.dialect.upper()} SQL."
                ),
            )

        if self.dialect == "odps" and func_name in FORBIDDEN_FUNCTIONS_ODPS:
            odps_hint = (
                "use TO_CHAR(datetime, 'yyyymmdd') or TO_DATE(str, 'yyyymmdd')."
                if func_name in ("DATE_FORMAT", "UNIX_TIMESTAMP", "FROM_UNIXTIME")
                else "use DATEADD(datetime, delta, datepart)."
            )
            return LintResult(
                passed=False,
                error_code="ODPS-0130071",
                error_msg=(
                    f"FAILED: ODPS-0130071: Semantic analysis exception - function '{func_name}' "
                    f"is not supported in ODPS; {odps_hint}"
                ),
            )

        if self.dialect == "hive" and func_name in FORBIDDEN_FUNCTIONS_HIVE:
            return LintResult(
                passed=False,
                error_code="HIVE-10011",
                error_msg=(
                    f"FAILED: SemanticException [Error 10011]: function '{func_name}' is ODPS syntax "
                    "and not supported in Hive; use DATE_FORMAT(expr, 'yyyyMMdd') or SUBSTR/CONCAT_WS."
                ),
            )

        if func_name == "DATEDIFF":
            if self.dialect == "odps":
                if len(args) != 3:
                    return LintResult(
                        passed=False,
                        error_code="ODPS-0130071",
                        error_msg=(
                            "FAILED: ODPS-0130071: Semantic analysis exception - "
                            f"function DATEDIFF in ODPS requires exactly 3 arguments "
                            f"DATEDIFF(end_date, start_date, 'dd'), got {len(args)}."
                        ),
                    )
                unit_toks = args[2]
                if len(unit_toks) == 1 and unit_toks[0].token_type == TokenType.STRING:
                    if unit_toks[0].text.strip().lower() not in VALID_ODPS_DATEPARTS:
                        return LintResult(
                            passed=False,
                            error_code="ODPS-0130071",
                            error_msg=(
                                "FAILED: ODPS-0130071: Semantic analysis exception - "
                                f"invalid DATEDIFF datepart '{unit_toks[0].text}'; expected 'dd', 'mm', 'yyyy', etc."
                            ),
                        )
            else:
                if len(args) != 2:
                    return LintResult(
                        passed=False,
                        error_code="HIVE-10015",
                        error_msg=(
                            "FAILED: SemanticException [Error 10015]: "
                            f"Hive DATEDIFF expects exactly 2 arguments DATEDIFF(end_date, start_date), got {len(args)}."
                        ),
                    )

        elif func_name == "DATEADD":
            if self.dialect == "odps":
                if len(args) != 3:
                    return LintResult(
                        passed=False,
                        error_code="ODPS-0130071",
                        error_msg=(
                            "FAILED: ODPS-0130071: Semantic analysis exception - "
                            f"function DATEADD in ODPS requires exactly 3 arguments "
                            f"DATEADD(datetime, delta, 'dd'), got {len(args)}."
                        ),
                    )
                unit_toks = args[2]
                if len(unit_toks) == 1 and unit_toks[0].token_type == TokenType.STRING:
                    if unit_toks[0].text.strip().lower() not in VALID_ODPS_DATEPARTS:
                        return LintResult(
                            passed=False,
                            error_code="ODPS-0130071",
                            error_msg=(
                                "FAILED: ODPS-0130071: Semantic analysis exception - "
                                f"invalid DATEADD datepart '{unit_toks[0].text}'."
                            ),
                        )
            else:
                return LintResult(
                    passed=False,
                    error_code="HIVE-10011",
                    error_msg=(
                        "FAILED: SemanticException [Error 10011]: "
                        "DATEADD(d, delta, unit) is ODPS syntax; use DATE_ADD(d, delta) in Hive."
                    ),
                )

        elif func_name == "MAX_PT":
            if self.dialect != "odps":
                return LintResult(
                    passed=False,
                    error_code="HIVE-10011",
                    error_msg="FAILED: SemanticException: MAX_PT is an ODPS-specific function, not supported in Hive.",
                )
            if len(args) != 1 or len(args[0]) != 1 or args[0][0].token_type != TokenType.STRING:
                return LintResult(
                    passed=False,
                    error_code="ODPS-0130071",
                    error_msg=(
                        "FAILED: ODPS-0130071: Semantic analysis exception - "
                        "MAX_PT requires a single string literal table name, e.g., MAX_PT('table_name')."
                    ),
                )
            target_tbl = args[0][0].text.strip()
            meta = self.catalog.get(target_tbl)
            if meta is None:
                return LintResult(
                    passed=False,
                    error_code="ODPS-0130131",
                    error_msg=f"FAILED: ODPS-0130131: Table not found - table '{target_tbl}' in MAX_PT.",
                )
            if not meta.is_partitioned:
                return LintResult(
                    passed=False,
                    error_code="ODPS-0123031",
                    error_msg=(
                        f"FAILED: ODPS-0123031: Partition exception - "
                        f"MAX_PT cannot be called on unpartitioned table '{target_tbl}'."
                    ),
                )

        elif func_name == "GET_JSON_OBJECT":
            if len(args) != 2:
                return LintResult(
                    passed=False,
                    error_code="ODPS-0130071",
                    error_msg=(
                        "FAILED: ODPS-0130071: Semantic analysis exception - "
                        f"GET_JSON_OBJECT expects 2 arguments (json_str, '$.path'), got {len(args)}."
                    ),
                )
            path_toks = args[1]
            if len(path_toks) == 1 and path_toks[0].token_type == TokenType.STRING:
                path_val = path_toks[0].text.strip()
                if not path_val.startswith("$"):
                    return LintResult(
                        passed=False,
                        error_code="ODPS-0130071",
                        error_msg=(
                            "FAILED: ODPS-0130071: Semantic analysis exception - "
                            f"GET_JSON_OBJECT path '{path_val}' must start with '$'."
                        ),
                    )

        return None

    def _check_joins(self, root: exp.Expression) -> LintResult:
        """Ban implicit comma joins, CROSS JOIN without ON, and tautological ON clauses."""
        for select in root.find_all(exp.Select):
            for join in select.args.get("joins") or []:
                has_on = join.args.get("on") is not None
                has_using = bool(join.args.get("using"))
                if not has_on and not has_using:
                    return LintResult(
                        passed=False,
                        error_code="ODPS-0123055",
                        error_msg=(
                            "FAILED: ODPS-0123055: Cartesian product exception - "
                            "implicit comma join or JOIN without explicit ON condition is prohibited."
                        ),
                    )
                if has_on and self._is_trivial_tautology(join.args["on"]):
                    return LintResult(
                        passed=False,
                        error_code="ODPS-0123055",
                        error_msg=(
                            "FAILED: ODPS-0123055: Cartesian product exception - "
                            "JOIN with constant tautology ON condition (e.g., ON 1=1 or ON TRUE) is prohibited."
                        ),
                    )
        return LintResult(passed=True)

    @staticmethod
    def _is_trivial_tautology(node: exp.Expression) -> bool:
        if isinstance(node, exp.Boolean) and node.this is True:
            return True
        if isinstance(node, exp.EQ):
            left, right = node.this, node.expression
            if isinstance(left, exp.Literal) and isinstance(right, exp.Literal):
                return left.this == right.this
        return False
