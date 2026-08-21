"""Narrator agent: writes the findings in plain English.

Prose only - no tool calls, no structured output. That is deliberate: the default
narrator runs on Groq's gpt-oss, which is quick and cheap but has a habit of leaking
reasoning fragments into tool-call slots. With nothing to parse, that failure mode
has nowhere to land.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..data.summaries import cleaning_summary, collect_tables, enforce_budget, table_preview
from ..models import get_agent_model
from ..state import AnalysisState

# gpt-oss likes narrow no-break spaces (U+202F) around units and comparisons. They render
# fine in a browser and then blow up the moment anything prints them to a cp1252 console,
# so the prose gets flattened to ASCII whitespace on the way into state rather than at
# each of the three places that read it.
_SPACE_LIKE = dict.fromkeys([0x00A0, 0x2007, 0x2009, 0x202F, 0x205F, 0x3000], " ")
_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)
# U+2011 and U+2212 look like an ordinary hyphen and are just as fatal to cp1252.
_DASH_LIKE = dict.fromkeys([0x2010, 0x2011, 0x2212], "-")


def normalise_prose(text: str) -> str:
    """Strip invisible codepoints and collapse exotic spaces and dashes to plain ones."""
    return text.translate({**_SPACE_LIKE, **_ZERO_WIDTH, **_DASH_LIKE})


SYSTEM = """You are a data analyst writing up results for a colleague.

You are given a question and the computed numbers that answer it. Write the findings
in clear prose.

Rules:
- Every number you cite must appear in the results given to you. Never estimate,
  round differently, or infer a figure that isn't there.
- Lead with the direct answer to the question, then the supporting detail.
- Say what the numbers show, not what you did to get them. No "I ran a groupby".
- If the data-cleaning notes affect how a result should be read, say so briefly.
- Note genuine limitations, but don't hedge every sentence.
- No headings, no bullet lists, no preamble. Three to five short paragraphs.
"""

REVISION_NOTE = """
This is a revision. The reviewer asked for the following changes - apply them
directly and rewrite the whole narrative, don't append a note about the edit:

{feedback}
"""


def make_narrator_node(settings: Settings):
    def narrator_node(state: AnalysisState) -> dict[str, Any]:
        tables = collect_tables(state.get("query_result", {}), state.get("analysis", {}))
        query_result = state.get("query_result", {})

        # Failed ops are shown to the narrator on purpose: a report that quietly omits
        # what didn't compute is worse than one that mentions the gap.
        failed = [
            {"op": o["op"], "error": o["error"]}
            for block in (query_result, state.get("analysis", {}))
            for o in (block or {}).get("operations", [])
            if not o.get("ok")
        ]

        payload = enforce_budget(
            {
                "interpretation": query_result.get("interpretation", ""),
                "population": {
                    "rows_considered": query_result.get("rows_considered"),
                    "rows_total": query_result.get("rows_total"),
                    "filters": query_result.get("filters", []),
                },
                "results": {k: table_preview(v, max_rows=20) for k, v in tables.items()},
                "correlations": [
                    o["result"].get("ranked_pairs", [])
                    for block in (query_result, state.get("analysis", {}))
                    for o in (block or {}).get("operations", [])
                    if o.get("ok") and o["op"] == "correlate"
                ],
                "cleaning_notes": cleaning_summary(state.get("cleaning_report", {})),
                "charts": [
                    {"title": c.get("title"), "caption": c.get("caption")}
                    for c in state.get("chart_specs", [])
                ],
                "operations_that_failed": failed,
            },
            "narrator prompt",
        )

        system = SYSTEM
        feedback = (state.get("review") or {}).get("feedback", "")
        if feedback:
            system += REVISION_NOTE.format(feedback=feedback)

        model = get_agent_model("narrator", settings)
        response = model.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=f"Question: {state['question']}\n\nComputed results:\n{payload}"),
            ]
        )

        text = response.content
        if isinstance(text, list):  # some providers return content blocks
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

        return {"narrative": normalise_prose(text).strip()}

    return narrator_node
