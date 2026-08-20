"""Schema cards and the payload budget.

This is where the project's central rule is actually enforced: an LLM sees column
metadata and aggregate results, never rows. enforce_budget() raises rather than
truncating, because a guardrail that quietly trims is a guardrail you find out about
in the LangSmith bill.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

# A payload this size is a summary. Anything larger means an aggregation was missed
# somewhere and raw-ish data is about to be sent.
MAX_PAYLOAD_CHARS = 12_000
MAX_RESULT_ROWS = 60

# Columns with more distinct values than this are treated as identifiers: they get a
# cardinality count and nothing else. No customer ID ever reaches a prompt.
MAX_CATEGORY_LABELS = 12


class BudgetExceeded(RuntimeError):
    """Raised when a model-bound payload is too large to be a summary."""


def build_schema_card(frame: pd.DataFrame, origin: str = "") -> dict[str, Any]:
    """Column metadata the agents need in order to plan a query.

    Distinct *labels* are included for low-cardinality categoricals, because an agent
    cannot write `Contract == "Monthly"` without knowing the label is "Monthly" and
    not "Month-to-month". Those are schema, not data: no row is reconstructable from
    them, and high-cardinality columns are excluded entirely.
    """
    columns = []
    for name in frame.columns:
        series = frame[name]
        nunique = int(series.nunique(dropna=True))
        info: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "distinct": nunique,
        }

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            info["kind"] = "numeric"
            desc = series.describe()
            info["range"] = {
                "min": round(float(desc["min"]), 4),
                "max": round(float(desc["max"]), 4),
                "mean": round(float(desc["mean"]), 4),
            }
        elif pd.api.types.is_bool_dtype(series):
            info["kind"] = "boolean"
            info["true_count"] = int(series.sum())
        else:
            if nunique <= MAX_CATEGORY_LABELS:
                info["kind"] = "categorical"
                info["labels"] = sorted(str(v) for v in series.dropna().unique())
            else:
                # Identifier-like. Deliberately no sample values.
                info["kind"] = "identifier"

        columns.append(info)

    return {
        "origin": origin,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "columns": columns,
    }


def frame_to_records(frame: pd.DataFrame, max_rows: int = MAX_RESULT_ROWS) -> dict[str, Any]:
    """Turn a small aggregate frame into JSON-safe records.

    Guards on row count: an aggregate result with hundreds of rows is a sign the
    grouping keys were wrong and the frame is closer to raw than summarised.
    """
    if len(frame) > max_rows:
        raise BudgetExceeded(
            f"Result has {len(frame)} rows (limit {max_rows}). "
            "This looks like raw data rather than a summary - tighten the grouping."
        )
    prepared = frame.reset_index() if frame.index.name or isinstance(frame.index, pd.MultiIndex) else frame
    records = json.loads(prepared.to_json(orient="records", date_format="iso"))
    return {"columns": list(prepared.columns.astype(str)), "rows": records, "row_count": len(records)}


def enforce_budget(payload: Any, label: str = "payload") -> str:
    """Serialise a model-bound payload, refusing anything too big to be a summary."""
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    if len(text) > MAX_PAYLOAD_CHARS:
        raise BudgetExceeded(
            f"{label} is {len(text)} chars (limit {MAX_PAYLOAD_CHARS}). "
            "Aggregate further before sending this to a model."
        )
    return text


def _is_table(value: Any) -> bool:
    return isinstance(value, dict) and "columns" in value and "rows" in value


def collect_tables(*blocks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten every result table out of the executed operations into one catalog.

    The viz agent picks charts by naming a key from this catalog, so the keys have to
    be stable and self-describing - hence op name plus the columns involved.
    """
    tables: dict[str, dict[str, Any]] = {}

    def walk(node: Any, prefix: str) -> None:
        if _is_table(node):
            key = prefix
            suffix = 2
            while key in tables:
                key = f"{prefix}_{suffix}"
                suffix += 1
            tables[key] = node
            return
        if isinstance(node, dict):
            for name, child in node.items():
                walk(child, f"{prefix}.{name}" if prefix else str(name))

    for block in blocks:
        for entry in (block or {}).get("operations", []):
            if not entry.get("ok"):
                continue
            parts = entry.get("group_by") or entry.get("columns") or []
            label = "_".join([entry["op"], *[str(p) for p in parts]])
            walk(entry.get("result"), label)

    return tables


def table_preview(table: dict[str, Any], max_rows: int = 12) -> dict[str, Any]:
    """Trim a table for prompt use. The full table stays in state for charting."""
    rows = table.get("rows", [])
    return {
        "columns": table.get("columns", []),
        "rows": rows[:max_rows],
        "row_count": table.get("row_count", len(rows)),
        "truncated": len(rows) > max_rows,
    }


def cleaning_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Compact view of the cleaning report for prompts.

    Drops affected_ids: the agents don't need identifiers to interpret the cleaning,
    and shipping them would violate the rule for no benefit. The full list stays in
    state and lands in the written report.
    """
    return {
        "rows": report.get("rows_out"),
        "coercions": [
            {k: v for k, v in c.items() if k != "affected_ids"}
            for c in report.get("coercions", [])
        ],
        "recodes": report.get("recodes", []),
        "warnings": report.get("warnings", []),
        "remaining_nulls": report.get("remaining_nulls", {}),
    }
