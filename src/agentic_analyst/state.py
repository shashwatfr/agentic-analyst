"""Graph state and the structured payloads the agents are allowed to emit.

The single most important thing in this file: AnalysisState has no DataFrame in it.
It carries `dataset_id`, a handle into the in-process registry. Two reasons - the
models physically cannot see data that never enters state, and the SQLite
checkpointer would otherwise serialise 7043 rows on every superstep.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

# Ops the agents may request. Anything outside this list is rejected before it
# reaches pandas - see tools/pandas_ops.py.
OpName = Literal[
    "value_counts",
    "group_agg",
    "describe",
    "correlate",
    "detect_outliers",
    "crosstab_rate",
]

ChartKind = Literal["bar", "barh", "line", "hist", "box", "scatter", "heatmap"]


class Filter(BaseModel):
    """A single row filter. Values are matched against the schema card's labels."""

    column: str = Field(description="Column to filter on. Must exist in the schema card.")
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in"] = Field(
        description="Comparison operator."
    )
    value: str = Field(
        description="Comparison value as a string; numeric columns are cast. "
        "For 'in'/'not_in', a comma-separated list."
    )


class Operation(BaseModel):
    """One computation to run against the (already filtered) frame."""

    op: OpName = Field(description="Which whitelisted operation to run.")
    columns: list[str] = Field(
        default_factory=list, description="Columns the operation applies to."
    )
    group_by: list[str] = Field(
        default_factory=list, description="Grouping keys, for group_agg / crosstab_rate."
    )
    agg: Literal["mean", "median", "sum", "count", "min", "max", "std"] = Field(
        default="mean", description="Aggregation for group_agg."
    )
    rationale: str = Field(default="", description="One line on why this answers the question.")


class QueryPlan(BaseModel):
    """What the query agent returns. It plans; it does not execute."""

    interpretation: str = Field(description="Restate the question in terms of actual columns.")
    filters: list[Filter] = Field(default_factory=list)
    operations: list[Operation] = Field(
        description="1-4 operations that together answer the question."
    )
    target_column: str = Field(
        default="", description="Outcome column of interest, if the question implies one."
    )


class AnalysisPlan(BaseModel):
    """Follow-up computations the analysis agent wants, given the query results."""

    operations: list[Operation] = Field(
        default_factory=list, description="0-4 additional operations for depth."
    )
    focus: str = Field(default="", description="What this pass is trying to establish.")


class ChartSpec(BaseModel):
    """A declarative chart request.

    The model describes the chart; renderer code draws it. No generated code is ever
    executed, which keeps the whole surface non-exploitable.
    """

    kind: ChartKind
    title: str
    x: str = Field(default="", description="Column for the x axis / categories.")
    y: str = Field(default="", description="Column for the y axis / values.")
    hue: str = Field(default="", description="Optional grouping column.")
    source_key: str = Field(
        default="",
        description="Key of the computed result table this chart draws from.",
    )
    caption: str = Field(default="", description="One sentence on what the chart shows.")


class VizPlan(BaseModel):
    charts: list[ChartSpec] = Field(description="1-3 charts that support the findings.")


class AnalysisState(TypedDict, total=False):
    """State threaded through the graph. JSON-serialisable throughout."""

    question: str
    dataset_id: str          # handle into data.registry, never the frame itself
    schema_card: dict[str, Any]
    cleaning_report: dict[str, Any]

    query_plan: dict[str, Any]
    query_result: dict[str, Any]
    analysis: dict[str, Any]

    chart_specs: list[dict[str, Any]]
    chart_paths: list[str]

    narrative: str
    report_md: str

    review: dict[str, Any] | None
    revision_count: int
    commit_result: dict[str, Any]

    errors: Annotated[list[str], operator.add]
