"""LangGraph wiring.

Flow:

    load_and_clean -> query -> analyze -> visualize -> narrate -> assemble_report
                                                 ^                      |
                                                 |                      v
                                                 +---- edit ----- human_review
                                                                        |
                                                        approve -> commit -> END
                                                        reject  -----------> END

human_review calls interrupt(), which halts the graph and persists state to SQLite.
Resuming means invoking with Command(resume=...) - including from a completely
different process, which is what makes it a real interrupt rather than a paused
function call.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents.analysis import make_analysis_node
from .agents.narrator import make_narrator_node
from .agents.query import make_query_node
from .agents.viz import make_viz_node
from .config import CHECKPOINT_DIR, Settings
from .data.source import DataSource
from .data.summaries import build_schema_card
from .report import build_report
from .sinks.base import ReportSink
from .state import AnalysisState

# Two revisions is enough to act on feedback; beyond that the loop is the problem.
MAX_REVISIONS = 2


def _slug(text: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "analysis")[:40]


def make_load_node(source: DataSource):
    def load_and_clean(state: AnalysisState) -> dict[str, Any]:
        dataset = source.load()
        return {
            "dataset_id": dataset.dataset_id,
            "schema_card": build_schema_card(dataset.frame, dataset.origin),
            "cleaning_report": dataset.cleaning_report,
            "revision_count": state.get("revision_count", 0),
        }

    return load_and_clean


def assemble_report(state: AnalysisState) -> dict[str, Any]:
    return {"report_md": build_report(dict(state))}


def human_review(state: AnalysisState) -> dict[str, Any]:
    """The HITL gate. Everything above this line is reversible; below it isn't."""
    decision = interrupt(
        {
            "type": "report_review",
            "question": state.get("question", ""),
            "report_md": state.get("report_md", ""),
            "chart_paths": state.get("chart_paths", []),
            "revision_count": state.get("revision_count", 0),
            "revisions_remaining": MAX_REVISIONS - state.get("revision_count", 0),
            "errors": state.get("errors", []),
            "instructions": "Respond with {'decision': 'approve'|'edit'|'reject', 'feedback': str}",
        }
    )

    if isinstance(decision, str):
        decision = {"decision": decision, "feedback": ""}
    decision = dict(decision or {})
    decision.setdefault("decision", "reject")
    decision.setdefault("feedback", "")

    update: dict[str, Any] = {"review": decision}
    if decision["decision"] == "edit":
        update["revision_count"] = state.get("revision_count", 0) + 1
    return update


def route_review(state: AnalysisState) -> str:
    decision = (state.get("review") or {}).get("decision", "reject")
    if decision == "approve":
        return "commit"
    if decision == "edit" and state.get("revision_count", 0) <= MAX_REVISIONS:
        return "narrate"
    return END


def make_commit_node(sink: ReportSink):
    def commit(state: AnalysisState) -> dict[str, Any]:
        meta = {
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "slug": _slug(state.get("question", "analysis")),
            "question": state.get("question", ""),
        }
        result = sink.commit(state.get("report_md", ""), state.get("chart_paths", []), meta)
        return {
            "commit_result": result.to_dict(),
            "errors": [] if result.ok else [f"commit failed: {result.detail}"],
        }

    return commit


def build_checkpointer(path: Path | None = None) -> SqliteSaver:
    """SQLite rather than in-memory, so a paused graph survives the process exiting."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = path or (CHECKPOINT_DIR / "analyst.sqlite")
    # check_same_thread=False because LangGraph may touch the connection from its own
    # worker threads.
    connection = sqlite3.connect(target, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def build_graph(settings: Settings, source: DataSource, sink: ReportSink, checkpointer=None):
    graph = StateGraph(AnalysisState)

    graph.add_node("load_and_clean", make_load_node(source))
    graph.add_node("query", make_query_node(settings))
    graph.add_node("analyze", make_analysis_node(settings))
    graph.add_node("visualize", make_viz_node(settings))
    graph.add_node("narrate", make_narrator_node(settings))
    graph.add_node("assemble_report", assemble_report)
    graph.add_node("human_review", human_review)
    graph.add_node("commit", make_commit_node(sink))

    graph.add_edge(START, "load_and_clean")
    graph.add_edge("load_and_clean", "query")
    graph.add_edge("query", "analyze")
    graph.add_edge("analyze", "visualize")
    graph.add_edge("visualize", "narrate")
    graph.add_edge("narrate", "assemble_report")
    graph.add_edge("assemble_report", "human_review")
    graph.add_conditional_edges(
        "human_review", route_review, {"commit": "commit", "narrate": "narrate", END: END}
    )
    graph.add_edge("commit", END)

    return graph.compile(checkpointer=checkpointer or build_checkpointer())
