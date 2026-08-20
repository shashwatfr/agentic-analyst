"""The whitelist of computations an agent may request.

Agents emit a structured plan naming an op and some columns. This module is the only
thing that touches the frame. Nothing here evals, execs, or takes a code string, so
there is no path from model output to arbitrary execution - the worst a bad plan can
do is name a column that doesn't exist and get rejected.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..data.summaries import BudgetExceeded, frame_to_records

# Guards against a "grouping" that is really a per-row listing.
MAX_GROUPS = 50
OUTLIER_IQR_MULTIPLIER = 1.5


class OpError(ValueError):
    """A plan referenced something that doesn't exist or isn't allowed."""


def _require_columns(frame: pd.DataFrame, columns: list[str], op: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise OpError(f"{op}: unknown column(s) {missing}. Available: {list(frame.columns)}")


def _numeric_columns(frame: pd.DataFrame, columns: list[str], op: str) -> list[str]:
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        raise OpError(f"{op}: none of {columns} are numeric")
    return numeric


def apply_filters(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the plan's row filters. Returns the frame plus a human-readable trace."""
    out = frame
    described: list[str] = []

    for spec in filters or []:
        column, op, raw_value = spec.get("column"), spec.get("op"), spec.get("value")
        _require_columns(out, [column], "filter")
        series = out[column]

        if op in {"in", "not_in"}:
            wanted = [v.strip() for v in str(raw_value).split(",") if v.strip()]
            mask = series.astype(str).isin(wanted)
            mask = ~mask if op == "not_in" else mask
        else:
            value: Any = raw_value
            if pd.api.types.is_numeric_dtype(series):
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise OpError(f"filter: {raw_value!r} is not numeric for {column}") from exc
                comparisons: dict[str, Callable[[], pd.Series]] = {
                    "==": lambda: series == value, "!=": lambda: series != value,
                    ">": lambda: series > value, ">=": lambda: series >= value,
                    "<": lambda: series < value, "<=": lambda: series <= value,
                }
            else:
                text = series.astype(str)
                comparisons = {
                    "==": lambda: text == str(value), "!=": lambda: text != str(value),
                    ">": lambda: text > str(value), ">=": lambda: text >= str(value),
                    "<": lambda: text < str(value), "<=": lambda: text <= str(value),
                }
            if op not in comparisons:
                raise OpError(f"filter: unsupported operator {op!r}")
            mask = comparisons[op]()

        out = out[mask]
        described.append(f"{column} {op} {raw_value}")

        if out.empty:
            raise OpError(f"Filters produced an empty result after '{described[-1]}'")

    return out, described


def op_value_counts(frame: pd.DataFrame, columns: list[str], **_: Any) -> dict[str, Any]:
    _require_columns(frame, columns, "value_counts")
    results = {}
    for column in columns:
        counts = frame[column].value_counts(dropna=False)
        if len(counts) > MAX_GROUPS:
            raise OpError(
                f"value_counts: {column} has {len(counts)} distinct values (limit {MAX_GROUPS})"
            )
        table = counts.rename("count").to_frame()
        table["share"] = (table["count"] / len(frame)).round(4)
        table.index.name = column
        results[column] = frame_to_records(table)
    return results


def op_group_agg(
    frame: pd.DataFrame, columns: list[str], group_by: list[str], agg: str = "mean", **_: Any
) -> dict[str, Any]:
    if not group_by:
        raise OpError("group_agg: group_by is required")
    _require_columns(frame, group_by + columns, "group_agg")
    numeric = _numeric_columns(frame, columns, "group_agg")

    grouped = frame.groupby(group_by, dropna=False)[numeric].agg(agg)
    if len(grouped) > MAX_GROUPS:
        raise OpError(f"group_agg: {len(grouped)} groups (limit {MAX_GROUPS})")
    grouped = grouped.round(4)
    grouped["n"] = frame.groupby(group_by, dropna=False).size()
    return frame_to_records(grouped)


def op_describe(frame: pd.DataFrame, columns: list[str], **_: Any) -> dict[str, Any]:
    targets = columns or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    _require_columns(frame, targets, "describe")
    numeric = _numeric_columns(frame, targets, "describe")
    table = frame[numeric].describe().round(4)
    table.index.name = "stat"
    return frame_to_records(table)


def op_correlate(frame: pd.DataFrame, columns: list[str], **_: Any) -> dict[str, Any]:
    targets = columns or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    _require_columns(frame, targets, "correlate")
    numeric = _numeric_columns(frame, targets, "correlate")
    if len(numeric) < 2:
        raise OpError("correlate: needs at least two numeric columns")
    matrix = frame[numeric].corr(numeric_only=True).round(4)
    matrix.index.name = "column"

    # The matrix is for charting; the ranked pairs are what the narrator can actually
    # use in a sentence.
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            pairs.append({"pair": f"{a} ~ {b}", "r": round(float(matrix.loc[a, b]), 4)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)

    return {"matrix": frame_to_records(matrix), "ranked_pairs": pairs[:15]}


def op_detect_outliers(frame: pd.DataFrame, columns: list[str], **_: Any) -> dict[str, Any]:
    """IQR fences. Counts and bounds only - never the outlying rows themselves."""
    targets = columns or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    _require_columns(frame, targets, "detect_outliers")
    numeric = _numeric_columns(frame, targets, "detect_outliers")

    findings = []
    for column in numeric:
        series = frame[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        mask = (series < low) | (series > high)
        findings.append(
            {
                "column": column,
                "lower_fence": round(float(low), 4),
                "upper_fence": round(float(high), 4),
                "outlier_count": int(mask.sum()),
                "outlier_share": round(float(mask.mean()), 4),
            }
        )
    return {"method": f"IQR x{OUTLIER_IQR_MULTIPLIER}", "findings": findings}


def op_crosstab_rate(
    frame: pd.DataFrame, columns: list[str], group_by: list[str], **_: Any
) -> dict[str, Any]:
    """Rate of each outcome level within each group - the churn-by-X workhorse."""
    if not group_by or not columns:
        raise OpError("crosstab_rate: needs group_by and a target column")
    target = columns[0]
    _require_columns(frame, group_by + [target], "crosstab_rate")

    table = pd.crosstab(
        [frame[g] for g in group_by], frame[target], normalize="index"
    ).round(4)
    if len(table) > MAX_GROUPS:
        raise OpError(f"crosstab_rate: {len(table)} groups (limit {MAX_GROUPS})")
    table.columns = [f"{target}={c}" for c in table.columns]
    table["n"] = frame.groupby(group_by, dropna=False).size()
    return frame_to_records(table)


OPS: dict[str, Callable[..., dict[str, Any]]] = {
    "value_counts": op_value_counts,
    "group_agg": op_group_agg,
    "describe": op_describe,
    "correlate": op_correlate,
    "detect_outliers": op_detect_outliers,
    "crosstab_rate": op_crosstab_rate,
}


def run_operation(frame: pd.DataFrame, operation: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one planned operation. Errors are returned, not raised.

    A single bad operation shouldn't sink the run - the graph collects the error,
    keeps the operations that worked, and lets the narrator note the gap.
    """
    name = operation.get("op")
    handler = OPS.get(name)
    if handler is None:
        return {"op": name, "ok": False, "error": f"Unknown op {name!r}. Allowed: {sorted(OPS)}"}

    try:
        result = handler(
            frame,
            columns=list(operation.get("columns") or []),
            group_by=list(operation.get("group_by") or []),
            agg=operation.get("agg", "mean"),
        )
    except (OpError, BudgetExceeded) as exc:
        return {"op": name, "ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface the failure, don't kill the graph
        return {"op": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "op": name,
        "ok": True,
        "columns": operation.get("columns") or [],
        "group_by": operation.get("group_by") or [],
        "agg": operation.get("agg", "mean"),
        "rationale": operation.get("rationale", ""),
        "result": result,
    }
