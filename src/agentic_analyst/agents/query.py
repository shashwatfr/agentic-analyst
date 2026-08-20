"""Query agent: turns a plain-English question into a structured, executable plan.

The split that matters: the model decides *what* to compute, this module runs it. The
model never receives a row and never emits code.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..data import registry
from ..data.summaries import cleaning_summary, enforce_budget
from ..models import get_structured_model
from ..state import AnalysisState, QueryPlan
from ..tools.pandas_ops import OPS, apply_filters, run_operation

SYSTEM = """You plan data analysis. You never see the data itself - only its schema.

You are given a column schema and a question. Return a plan of 1-4 operations that
together answer it. Rules:

- Only use column names exactly as they appear in the schema. Do not invent columns.
- Use the exact category labels from the schema. They may differ from what you expect
  for a well-known dataset; trust the schema, not your memory of the dataset.
- Available operations: {ops}
- crosstab_rate is the right choice for "rate of <outcome> by <group>" questions:
  put the outcome column in `columns` and the grouping column(s) in `group_by`.
- group_agg needs numeric columns in `columns` and the grouping key in `group_by`.
- Prefer few, well-chosen operations over many overlapping ones.
- Only add filters if the question genuinely restricts the population. Most questions
  about "customers" mean all of them - no filter.
"""


def build_plan(state: AnalysisState, settings: Settings) -> QueryPlan:
    model = get_structured_model("query", settings, QueryPlan)
    payload = enforce_budget(
        {
            "schema": state["schema_card"],
            "cleaning": cleaning_summary(state.get("cleaning_report", {})),
        },
        "query prompt",
    )
    messages = [
        SystemMessage(content=SYSTEM.format(ops=", ".join(sorted(OPS)))),
        HumanMessage(content=f"Question: {state['question']}\n\nSchema and cleaning notes:\n{payload}"),
    ]
    return model.invoke(messages)


def execute_plan(state: AnalysisState, plan: QueryPlan) -> dict[str, Any]:
    """Run the plan against the real frame, collecting per-operation outcomes."""
    frame = registry.get(state["dataset_id"])
    errors: list[str] = []

    try:
        filtered, filter_trace = apply_filters(frame, [f.model_dump() for f in plan.filters])
    except Exception as exc:  # noqa: BLE001 - a bad filter shouldn't end the run
        errors.append(f"filter failed ({exc}); continuing on the full dataset")
        filtered, filter_trace = frame, []

    operations = [run_operation(filtered, op.model_dump()) for op in plan.operations]
    errors.extend(f"query op '{o['op']}': {o['error']}" for o in operations if not o["ok"])

    return {
        "interpretation": plan.interpretation,
        "target_column": plan.target_column,
        "filters": filter_trace,
        "rows_considered": int(len(filtered)),
        "rows_total": int(len(frame)),
        "operations": operations,
        "_errors": errors,
    }


def make_query_node(settings: Settings):
    def query_node(state: AnalysisState) -> dict[str, Any]:
        plan = build_plan(state, settings)
        result = execute_plan(state, plan)
        errors = result.pop("_errors")

        # If nothing at all computed there is no point continuing to the analysis
        # agent - record it and let the downstream nodes report an honest failure.
        if not any(o["ok"] for o in result["operations"]):
            errors.append("query agent produced no usable results")

        return {
            "query_plan": plan.model_dump(),
            "query_result": result,
            "errors": errors,
        }

    return query_node
