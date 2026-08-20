"""Analysis agent: adds statistical depth on top of the query results.

Same contract as the query agent - it picks which further computations are worth
running, and pandas does them. It sees the numbers that came back, not the rows they
came from.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..data import registry
from ..data.summaries import collect_tables, enforce_budget, table_preview
from ..models import get_structured_model
from ..state import AnalysisPlan, AnalysisState
from ..tools.pandas_ops import OPS, apply_filters, run_operation

SYSTEM = """You deepen a data analysis that has already run once.

You are shown the question, the operations already executed, and their results. Pick
0-4 *additional* operations that add something the first pass missed - a correlation,
a distribution, an outlier check, or a breakdown by a second dimension.

Rules:
- Only use columns from the schema, spelled exactly as given.
- Available operations: {ops}
- Do not repeat an operation that already ran with the same columns.
- correlate and detect_outliers only accept numeric columns.
- If the first pass already answers the question well, return an empty list. That is
  a valid and often correct answer.
"""


def make_analysis_node(settings: Settings):
    def analysis_node(state: AnalysisState) -> dict[str, Any]:
        query_result = state.get("query_result", {})
        tables = collect_tables(query_result)

        model = get_structured_model("analysis", settings, AnalysisPlan)
        payload = enforce_budget(
            {
                "schema": state["schema_card"],
                "already_run": [
                    {"op": o["op"], "columns": o["columns"], "group_by": o["group_by"]}
                    for o in query_result.get("operations", [])
                    if o.get("ok")
                ],
                "results_so_far": {k: table_preview(v) for k, v in tables.items()},
            },
            "analysis prompt",
        )
        messages = [
            SystemMessage(content=SYSTEM.format(ops=", ".join(sorted(OPS)))),
            HumanMessage(content=f"Question: {state['question']}\n\n{payload}"),
        ]
        plan: AnalysisPlan = model.invoke(messages)

        frame = registry.get(state["dataset_id"])
        errors: list[str] = []

        # Re-apply the same filters so the second pass describes the same population
        # the first pass did. Skipping this is a subtle way to produce two sets of
        # numbers that quietly disagree.
        raw_filters = state.get("query_plan", {}).get("filters", [])
        try:
            filtered, _ = apply_filters(frame, raw_filters)
        except Exception:  # noqa: BLE001
            filtered = frame

        operations = [run_operation(filtered, op.model_dump()) for op in plan.operations]
        errors.extend(f"analysis op '{o['op']}': {o['error']}" for o in operations if not o["ok"])

        return {
            "analysis": {
                "focus": plan.focus,
                "operations": operations,
            },
            "errors": errors,
        }

    return analysis_node
