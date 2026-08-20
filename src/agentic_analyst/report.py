"""Assembles the markdown report from computed state.

Deterministic - no model involvement. The narrative is dropped in as written; every
other section is rendered from the numbers already in state.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .data.summaries import collect_tables


def _md_table(table: dict[str, Any], max_rows: int = 25) -> str:
    columns = [str(c) for c in table.get("columns", [])]
    rows = table.get("rows", [])[:max_rows]
    if not columns or not rows:
        return "_(no rows)_"

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            cells.append(f"{value:,.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")

    if len(table.get("rows", [])) > max_rows:
        lines.append(f"\n_{len(table['rows']) - max_rows} more rows omitted._")
    return "\n".join(lines)


def _cleaning_section(report: dict[str, Any]) -> str:
    if not report:
        return ""

    lines = ["## Data preparation", ""]
    lines.append(
        f"Loaded **{report.get('rows_in', 0):,} rows** from `{report.get('source', 'source')}`; "
        f"**{report.get('rows_out', 0):,} rows** after cleaning. No rows were dropped."
    )
    lines.append("")

    for coercion in report.get("coercions", []):
        flag = coercion.get("flag_column")
        lines.append(
            f"- **`{coercion['column']}`** was stored as text. "
            f"{coercion['affected_rows']} rows could not be converted "
            f"({coercion['blank']} blank, {coercion['unparseable_text']} unparseable) "
            f"and were set to zero under the `{coercion['strategy']}` strategy"
            + (f", flagged as `{flag}`." if flag else ".")
        )
        check = coercion.get("driver_check")
        if check:
            verdict = "confirmed" if check["holds"] else "**does not hold**"
            lines.append(
                f"  - Every affected row has `{check['column']} == {check['expected']}` "
                f"({verdict}), so zero is the accurate value rather than an imputation."
            )
        ids = coercion.get("affected_ids")
        if ids:
            lines.append(f"  - Affected IDs: {', '.join(f'`{i}`' for i in ids)}")

    for recode in report.get("recodes", []):
        lines.append(
            f"- **`{recode['column']}`** recoded from {recode['before']} to {recode['after']} "
            "for consistency with the other categorical columns."
        )

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "**Warnings raised during cleaning:**", ""])
        lines.extend(f"- {w}" for w in warnings)

    nulls = report.get("remaining_nulls", {})
    lines.extend(["", f"Remaining nulls after cleaning: {nulls or 'none'}."])
    return "\n".join(lines)


def _findings_section(state: dict[str, Any]) -> str:
    tables = collect_tables(state.get("query_result", {}), state.get("analysis", {}))
    if not tables:
        return ""

    lines = ["## Computed results", ""]
    for key, table in tables.items():
        lines.extend([f"### `{key}`", "", _md_table(table), ""])
    return "\n".join(lines)


def _charts_section(specs: list[dict[str, Any]], relative_to: Path | None = None) -> str:
    if not specs:
        return ""
    lines = ["## Charts", ""]
    for spec in specs:
        path = Path(spec.get("path", ""))
        # Relative links so the markdown renders next to the images wherever it lands.
        href = path.name if relative_to is None else path.name
        lines.append(f"**{spec.get('title', path.stem)}**")
        lines.append("")
        lines.append(f"![{spec.get('title', '')}]({href})")
        if spec.get("caption"):
            lines.append("")
            lines.append(f"_{spec['caption']}_")
        lines.append("")
    return "\n".join(lines)


def build_report(state: dict[str, Any], generated_at: datetime | None = None) -> str:
    generated_at = generated_at or datetime.now()
    query_result = state.get("query_result", {})

    parts = [
        f"# {state.get('question', 'Analysis')}",
        "",
        f"_Generated {generated_at:%Y-%m-%d %H:%M}_",
        "",
    ]

    if state.get("revision_count"):
        parts.extend([f"_Revision {state['revision_count']} after review._", ""])

    interpretation = query_result.get("interpretation")
    if interpretation:
        parts.extend(["## How the question was read", "", interpretation, ""])

    filters = query_result.get("filters", [])
    considered, total = query_result.get("rows_considered"), query_result.get("rows_total")
    if considered is not None:
        scope = f"Analysed **{considered:,}** of {total:,} rows"
        scope += f" (filters: {', '.join(f'`{f}`' for f in filters)})." if filters else " (no filters applied)."
        parts.extend([scope, ""])

    parts.extend(["## Findings", "", state.get("narrative", "_No narrative generated._"), ""])

    charts = _charts_section(state.get("chart_specs", []))
    if charts:
        parts.extend([charts, ""])

    findings = _findings_section(state)
    if findings:
        parts.extend([findings, ""])

    cleaning = _cleaning_section(state.get("cleaning_report", {}))
    if cleaning:
        parts.extend([cleaning, ""])

    errors = state.get("errors", [])
    if errors:
        parts.extend(["## Issues encountered", ""])
        parts.extend(f"- {e}" for e in errors)
        parts.append("")

    return "\n".join(parts)
