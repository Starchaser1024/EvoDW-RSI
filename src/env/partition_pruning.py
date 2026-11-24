"""Recursive AST partition-pruning & join-equivalence analyzer for ODPS and Hive SQL."""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from src.env.schema import WarehouseCatalog


@dataclass(frozen=True)
class LintResult:
    """Result from the static gateway AST & dialect linter."""

    passed: bool
    error_code: str = ""
    error_msg: str = ""


class PartitionPruningMixin:
    """Mixin providing recursive CTE/subquery/join partition-pruning verification."""

    catalog: WarehouseCatalog

    def _check_partition_pruning(
        self,
        node: exp.Expression,
        cte_names: set[str] | None = None,
    ) -> LintResult:
        """Recursively verify every partitioned table scan has an explicit partition filter."""
        local_ctes = set(cte_names or set())
        with_clause = node.args.get("with_")
        if isinstance(with_clause, exp.With):
            for cte in with_clause.expressions:
                if isinstance(cte, exp.CTE):
                    cte_body = cte.this
                    if cte_body is not None:
                        res = self._check_partition_pruning(cte_body, local_ctes)
                        if not res.passed:
                            return res
                    alias_name = cte.alias_or_name
                    if alias_name:
                        local_ctes.add(alias_name.lower())

        if isinstance(node, exp.Union):
            for branch in (node.left, node.right):
                if branch is not None:
                    res = self._check_partition_pruning(branch, local_ctes)
                    if not res.passed:
                        return res
            return LintResult(passed=True)

        if isinstance(node, exp.Subquery):
            if node.this is not None:
                return self._check_partition_pruning(node.this, local_ctes)
            return LintResult(passed=True)

        if not isinstance(node, exp.Select):
            return LintResult(passed=True)

        # First recurse into any nested Select/Subquery in FROM, JOINs, WHERE, SELECT projections, HAVING.
        for child_sub in self._direct_child_queries(node):
            res = self._check_partition_pruning(child_sub, local_ctes)
            if not res.passed:
                return res

        # Gather all physical partitioned table aliases in this Select's FROM & JOINs.
        # Map alias_lower -> (table_name, partition_col)
        part_aliases: dict[str, tuple[str, str]] = {}
        from_clause = node.args.get("from_")
        sources: list[exp.Expression] = []
        if from_clause is not None and from_clause.this is not None:
            sources.append(from_clause.this)
        for join in node.args.get("joins") or []:
            if join.this is not None:
                sources.append(join.this)

        for src in sources:
            if isinstance(src, exp.Table):
                tbl_name = src.name
                tbl_lower = tbl_name.lower()
                # CTE reference is not a physical table scan.
                if not src.db and tbl_lower in local_ctes:
                    continue
                if self.catalog.is_partitioned(tbl_lower):
                    pcol = self.catalog.get_partition_col(tbl_lower) or "ds"
                    alias = (src.alias_or_name or tbl_name).lower()
                    part_aliases[alias] = (tbl_name, pcol)

        if not part_aliases:
            return LintResult(passed=True)

        # Collect top-level conjunct predicates from WHERE and JOIN ... ON in this Select.
        predicates: list[exp.Expression] = []
        where_clause = node.args.get("where")
        if where_clause is not None and where_clause.this is not None:
            self._split_conjuncts(where_clause.this, predicates)
        for join in node.args.get("joins") or []:
            if join.args.get("on") is not None:
                self._split_conjuncts(join.args["on"], predicates)

        pruned_aliases: set[str] = set()
        join_edges: list[tuple[str, str]] = []

        for pred in predicates:
            direct, edges = self._analyze_predicate(pred, part_aliases)
            pruned_aliases.update(direct)
            join_edges.extend(edges)

        # Propagate pruning across join partition equivalences (e.g. a.ds = b.ds or b.ds = DATEADD(a.ds, ...)).
        changed = True
        while changed:
            changed = False
            for src_alias, dst_alias in join_edges:
                if src_alias in pruned_aliases and dst_alias not in pruned_aliases:
                    pruned_aliases.add(dst_alias)
                    changed = True
                elif dst_alias in pruned_aliases and src_alias not in pruned_aliases:
                    pruned_aliases.add(src_alias)
                    changed = True

        for alias, (tbl_name, pcol) in part_aliases.items():
            if alias not in pruned_aliases:
                return LintResult(
                    passed=False,
                    error_code="ODPS-0123031",
                    error_msg=(
                        f"FAILED: ODPS-0123031: Partition exception - full table scan is prohibited "
                        f"without partition filter on column '{pcol}' for table '{tbl_name}'"
                    ),
                )

        return LintResult(passed=True)

    @staticmethod
    def _direct_child_queries(select: exp.Select) -> list[exp.Expression]:
        """Collect immediate subqueries/unions inside a Select without double-scanning deep children."""
        children: list[exp.Expression] = []

        def _walk(expr: exp.Expression) -> None:
            for child in expr.iter_expressions():
                if isinstance(child, (exp.Select, exp.Union)):
                    children.append(child)
                elif isinstance(child, exp.Subquery) and child.this is not None:
                    children.append(child.this)
                else:
                    _walk(child)

        for key in ("expressions", "from_", "joins", "where", "group", "having", "qualify"):
            val = select.args.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, exp.Expression):
                        _walk(item)
            elif isinstance(val, exp.Expression):
                _walk(val)
        return children

    @classmethod
    def _split_conjuncts(cls, expr: exp.Expression, out: list[exp.Expression]) -> None:
        if isinstance(expr, exp.Paren):
            cls._split_conjuncts(expr.this, out)
        elif isinstance(expr, exp.And):
            cls._split_conjuncts(expr.left, out)
            cls._split_conjuncts(expr.right, out)
        else:
            out.append(expr)

    def _analyze_predicate(
        self,
        pred: exp.Expression,
        part_aliases: dict[str, tuple[str, str]],
    ) -> tuple[set[str], list[tuple[str, str]]]:
        """Determine which partition aliases are directly filtered or linked by a predicate."""
        pruned: set[str] = set()
        edges: list[tuple[str, str]] = []

        # Handle OR disjuncts: a disjunct prunes alias `a` only if ALL branches prune `a`!
        if isinstance(pred, exp.Paren):
            return self._analyze_predicate(pred.this, part_aliases)

        if isinstance(pred, exp.Or):
            l_pruned, _ = self._analyze_predicate(pred.left, part_aliases)
            r_pruned, _ = self._analyze_predicate(pred.right, part_aliases)
            return (l_pruned & r_pruned), []

        # Tautological / non-pruning checks: `ds IS NOT NULL`, `ds != ''`, `ds LIKE '%'`.
        if isinstance(pred, exp.Not) and isinstance(pred.this, exp.Is):
            return set(), []
        if isinstance(pred, exp.NEQ):
            return set(), []
        if isinstance(pred, exp.Like):
            rhs = pred.expression
            if isinstance(rhs, exp.Literal) and rhs.is_string and "%" in rhs.this:
                return set(), []

        # Comparison predicates: =, >=, >, <=, <
        if isinstance(pred, (exp.EQ, exp.GTE, exp.GT, exp.LTE, exp.LT)):
            left_aliases = self._find_partition_cols_in_expr(pred.this, part_aliases)
            right_aliases = self._find_partition_cols_in_expr(pred.expression, part_aliases)

            if left_aliases and right_aliases:
                for la in left_aliases:
                    for ra in right_aliases:
                        if la != ra:
                            edges.append((la, ra))
                return pruned, edges

            if left_aliases and not right_aliases:
                if not self._is_empty_string_literal(pred.expression):
                    pruned.update(left_aliases)
            elif right_aliases and not left_aliases:
                if not self._is_empty_string_literal(pred.this):
                    pruned.update(right_aliases)
            return pruned, edges

        # BETWEEN low AND high
        if isinstance(pred, exp.Between):
            target_aliases = self._find_partition_cols_in_expr(pred.this, part_aliases)
            low_aliases = self._find_partition_cols_in_expr(pred.args.get("low"), part_aliases)
            high_aliases = self._find_partition_cols_in_expr(pred.args.get("high"), part_aliases)
            if target_aliases and not low_aliases and not high_aliases:
                pruned.update(target_aliases)
            elif target_aliases and (low_aliases or high_aliases):
                for ta in target_aliases:
                    for oa in (low_aliases | high_aliases):
                        if ta != oa:
                            edges.append((ta, oa))
            return pruned, edges

        # IN (...)
        if isinstance(pred, exp.In):
            target_aliases = self._find_partition_cols_in_expr(pred.this, part_aliases)
            if target_aliases:
                vals = pred.expressions or []
                query = pred.args.get("query")
                if query is not None or any(not self._is_empty_string_literal(v) for v in vals):
                    pruned.update(target_aliases)
            return pruned, edges

        return pruned, edges

    @staticmethod
    def _is_empty_string_literal(node: exp.Expression | None) -> bool:
        return isinstance(node, exp.Literal) and node.is_string and node.this == ""

    def _find_partition_cols_in_expr(
        self,
        expr: exp.Expression | None,
        part_aliases: dict[str, tuple[str, str]],
    ) -> set[str]:
        """Find which partitioned table aliases' partition columns appear in `expr` (excluding nested queries)."""
        if expr is None:
            return set()
        found: set[str] = set()

        def _scan(node: exp.Expression) -> None:
            if isinstance(node, (exp.Select, exp.Union, exp.Subquery)):
                if isinstance(node, exp.Subquery) and not isinstance(node.this, (exp.Select, exp.Union)):
                    pass
                else:
                    return
            if isinstance(node, exp.Column):
                col_name = node.name.lower()
                tbl_qual = node.table.lower() if node.table else ""
                if tbl_qual:
                    info = part_aliases.get(tbl_qual)
                    if info is not None and col_name == info[1].lower():
                        found.add(tbl_qual)
                else:
                    matching = [a for a, (_, pcol) in part_aliases.items() if pcol.lower() == col_name]
                    if len(matching) == 1:
                        found.add(matching[0])
                return
            for child in node.iter_expressions():
                _scan(child)

        _scan(expr)
        return found
