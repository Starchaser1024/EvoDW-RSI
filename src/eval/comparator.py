"""Result set comparator with ordered/unordered multiset and float-tolerance matching."""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import Any


def _normalize_cell(val: Any) -> tuple[str, Any]:
    if val is None:
        return ("NULL", None)
    if isinstance(val, bool):
        return ("BOOL", val)
    if isinstance(val, (int, float, Decimal)):
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return ("NULL", None)
        return ("NUM", fval)
    if isinstance(val, dt.datetime):
        if val.hour == 0 and val.minute == 0 and val.second == 0 and val.microsecond == 0:
            return ("STR", val.strftime("%Y-%m-%d"))
        return ("STR", val.strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(val, dt.date):
        return ("STR", val.strftime("%Y-%m-%d"))
    if isinstance(val, (list, tuple)):
        return ("LIST", tuple(_normalize_cell(x) for x in val))
    return ("STR", str(val).strip())


def _cells_equal(
    a: tuple[str, Any],
    b: tuple[str, Any],
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
) -> bool:
    if a[0] != b[0]:
        return False
    tag = a[0]
    if tag == "NULL":
        return True
    if tag == "NUM":
        return math.isclose(a[1], b[1], rel_tol=rel_tol, abs_tol=abs_tol)
    if tag == "LIST":
        if len(a[1]) != len(b[1]):
            return False
        return all(_cells_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a[1], b[1]))
    return a[1] == b[1]


def _rows_equal(
    r1: tuple[tuple[str, Any], ...],
    r2: tuple[tuple[str, Any], ...],
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
) -> bool:
    if len(r1) != len(r2):
        return False
    return all(_cells_equal(c1, c2, rel_tol=rel_tol, abs_tol=abs_tol) for c1, c2 in zip(r1, r2))


def _row_sort_key(norm_row: tuple[tuple[str, Any], ...]) -> tuple[Any, ...]:
    keys: list[Any] = []
    for tag, val in norm_row:
        if tag == "NULL":
            keys.append((0, ""))
        elif tag == "BOOL":
            keys.append((1, int(val)))
        elif tag == "NUM":
            keys.append((2, round(val, 2)))
        elif tag == "LIST":
            keys.append((3, str(val)))
        else:
            keys.append((4, val))
    return tuple(keys)


def compare_result_sets(
    pred_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    ordered: bool = False,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
) -> tuple[bool, str]:
    """Compare predicted SQL result rows against Gold rows."""
    if len(pred_rows) != len(gold_rows):
        return False, f"Row count mismatch: predicted {len(pred_rows)} vs gold {len(gold_rows)}."

    if not gold_rows:
        return True, "Both result sets are empty."

    if len(pred_rows[0]) != len(gold_rows[0]):
        return (
            False,
            f"Column count mismatch: predicted {len(pred_rows[0])} vs gold {len(gold_rows[0])}.",
        )

    norm_pred = [tuple(_normalize_cell(c) for c in row) for row in pred_rows]
    norm_gold = [tuple(_normalize_cell(c) for c in row) for row in gold_rows]

    if ordered:
        for idx, (pr, gr) in enumerate(zip(norm_pred, norm_gold)):
            if not _rows_equal(pr, gr, rel_tol=rel_tol, abs_tol=abs_tol):
                return False, f"Ordered row mismatch at index {idx}: predicted {pred_rows[idx]} vs gold {gold_rows[idx]}."
        return True, "Exact ordered match."

    # Unordered multiset comparison: try sorted pairing first.
    sorted_pred = sorted(norm_pred, key=_row_sort_key)
    sorted_gold = sorted(norm_gold, key=_row_sort_key)
    if all(_rows_equal(pr, gr, rel_tol=rel_tol, abs_tol=abs_tol) for pr, gr in zip(sorted_pred, sorted_gold)):
        return True, "Exact unordered multiset match."

    # Fallback bipartite / greedy match for float boundary rounding edge cases.
    if len(norm_pred) <= 300:
        used = [False] * len(norm_gold)
        all_matched = True
        for pr in norm_pred:
            found_idx = -1
            for j, gr in enumerate(norm_gold):
                if not used[j] and _rows_equal(pr, gr, rel_tol=rel_tol, abs_tol=abs_tol):
                    found_idx = j
                    break
            if found_idx == -1:
                all_matched = False
                break
            used[found_idx] = True
        if all_matched:
            return True, "Unordered match within float tolerance."

    return False, f"Row content mismatch (sample pred={pred_rows[0]} vs gold={gold_rows[0]})."
