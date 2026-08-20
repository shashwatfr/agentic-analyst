#!/usr/bin/env python
"""Streamlit front end.

    uv run streamlit run app.py

Streamlit re-executes this script top to bottom on every interaction, which is
normally a fight with state. It isn't here: the graph pauses at interrupt() and its
state lives in SQLite keyed by thread_id, so a rerun just reloads the checkpoint.
The rerun model and the interrupt model happen to want the same thing.

The graph itself is untouched by this file - hitl.py renders the same interrupt
payload for the console. The graph doesn't know which one is driving it.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_analyst.config import (  # noqa: E402
    DEFAULT_DATASET,
    OUTPUT_DIR,
    configure_tracing,
    ensure_directories,
    load_settings,
)
from agentic_analyst.data.csv_source import CsvSource  # noqa: E402
from agentic_analyst.sinks import build_sink  # noqa: E402

st.set_page_config(page_title="agentic-analyst", page_icon="*", layout="wide")

EXAMPLES = [
    "Which contract type has the highest churn rate?",
    "What drives churn among fiber optic customers?",
    "How does tenure differ between customers who churned and those who stayed?",
    "Do customers on paperless billing churn more?",
]


@st.cache_resource(show_spinner=False)
def get_pipeline(dataset_path: str):
    """Built once and reused across reruns.

    cache_resource is doing real work here - it keeps the SQLite connection and the
    loaded DataFrame alive between reruns, so the in-process registry still resolves
    dataset_id after Streamlit re-executes the script.
    """
    settings = load_settings()
    tracing = configure_tracing(settings)
    ensure_directories()

    from agentic_analyst.graph import build_graph

    sink = build_sink(settings, OUTPUT_DIR)

    source = CsvSource(dataset_path)
    return build_graph(settings, source, sink), source, settings, tracing


def resolve_interrupt(result):
    pending = result.get("__interrupt__") if isinstance(result, dict) else None
    if not pending:
        return None
    first = pending[0]
    return getattr(first, "value", first)


def reset() -> None:
    for key in ("phase", "result", "thread_id", "question"):
        st.session_state.pop(key, None)


st.session_state.setdefault("phase", "idle")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("agentic-analyst")
    st.caption("Four agents plan, compute, chart, and write. You approve before anything ships.")

    dataset_path = st.text_input("Dataset", value=str(DEFAULT_DATASET))

    try:
        graph, source, settings, tracing = get_pipeline(dataset_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the dataset:\n\n{exc}")
        st.stop()

    st.divider()
    st.subheader("Models")
    for agent in ("query", "analysis", "viz", "narrator"):
        st.caption(f"**{agent}** · `{settings.model_for(agent)}`")

    st.divider()
    st.subheader("Environment")
    st.caption(f"Tracing · {'on → ' + settings.tracing_project if tracing else 'off'}")
    st.caption(f"Sink · {'Google Drive (MCP)' if settings.drive.enabled else 'local outputs/'}")

    if st.session_state.get("thread_id"):
        st.divider()
        st.caption("Thread")
        st.code(st.session_state["thread_id"], language=None)
        # The same paused graph can be resumed from the terminal. Same checkpoint,
        # different front end - which is the point of using a real interrupt.
        st.caption("Resumable from the CLI:")
        st.code(
            f"uv run python run.py --resume {st.session_state['thread_id']} "
            f"--decision approve",
            language="bash",
        )

# ---------------------------------------------------------------- ask
if st.session_state["phase"] == "idle":
    st.header("Ask a question about the data")
    st.caption(
        "Every number is computed in pandas. The models only ever see the summaries - "
        "no raw rows are sent to an LLM."
    )

    question = st.text_input("Question", placeholder=EXAMPLES[0], key="question_input")

    st.caption("Or start from one of these:")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if columns[i % 2].button(example, width="stretch", key=f"ex{i}"):
            question = example

    if question:
        thread_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.update(thread_id=thread_id, question=question)
        config = {"configurable": {"thread_id": thread_id}}
        with st.spinner("Cleaning data, planning the query, computing, charting, writing…"):
            st.session_state["result"] = graph.invoke({"question": question, "errors": []}, config)
        st.session_state["phase"] = "review"
        st.rerun()

# ---------------------------------------------------------------- review gate
elif st.session_state["phase"] == "review":
    payload = resolve_interrupt(st.session_state["result"])

    if payload is None:
        st.session_state["phase"] = "done"
        st.rerun()

    st.header("Review required")
    st.caption(
        "The graph is paused at a real LangGraph interrupt. Its state is checkpointed "
        "to SQLite - nothing is committed until you approve."
    )

    if payload.get("errors"):
        with st.expander(f"{len(payload['errors'])} issue(s) logged during the run"):
            for error in payload["errors"]:
                st.warning(error)

    charts = [Path(p) for p in payload.get("chart_paths", [])]
    if charts:
        st.subheader("Charts")
        for column, chart in zip(st.columns(min(len(charts), 3)), charts):
            if chart.exists():
                column.image(str(chart), width="stretch")

    st.subheader("Draft report")
    with st.container(height=460, border=True):
        st.markdown(payload.get("report_md", ""))

    st.divider()

    remaining = payload.get("revisions_remaining", 0)
    approve_col, edit_col, reject_col = st.columns(3)

    with approve_col:
        if st.button("Approve & commit", type="primary", width="stretch"):
            st.session_state["decision"] = {"decision": "approve", "feedback": ""}

    with edit_col:
        if st.button(
            f"Request changes ({remaining} left)",
            width="stretch",
            disabled=remaining <= 0,
        ):
            st.session_state["show_feedback"] = True

    with reject_col:
        if st.button("Reject", width="stretch"):
            st.session_state["decision"] = {"decision": "reject", "feedback": ""}

    if st.session_state.get("show_feedback"):
        feedback = st.text_area(
            "What should change?",
            placeholder="Cut it to two paragraphs and lead with the dollar difference.",
        )
        if st.button("Send back to the narrator", disabled=not feedback):
            st.session_state["decision"] = {"decision": "edit", "feedback": feedback}
            st.session_state["show_feedback"] = False

    if st.session_state.get("decision"):
        from langgraph.types import Command

        decision = st.session_state.pop("decision")
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        with st.spinner(f"Resuming the graph with '{decision['decision']}'…"):
            st.session_state["result"] = graph.invoke(Command(resume=decision), config)
        # An 'edit' loops back to narrate and pauses again, so stay on this screen and
        # let the interrupt check at the top decide.
        st.rerun()

# ---------------------------------------------------------------- outcome
else:
    result = st.session_state.get("result") or {}
    review = result.get("review") or {}

    if review.get("decision") == "reject":
        st.warning("Rejected. Nothing was committed.")
    else:
        commit = result.get("commit_result") or {}
        if commit.get("ok"):
            st.success(f"Committed to {commit.get('destination')}")
            st.caption(commit.get("detail", ""))
            if commit.get("link"):
                st.markdown(f"[Open the output folder]({commit['link']})")
            with st.expander("Files written"):
                for path in commit.get("files", []):
                    st.code(path, language=None)
        else:
            st.error(f"Commit failed: {commit.get('detail', 'unknown error')}")

    if result.get("report_md"):
        st.divider()
        st.subheader("Final report")
        with st.container(height=460, border=True):
            st.markdown(result["report_md"])

    st.divider()
    if st.button("Ask another question", type="primary"):
        reset()
        st.rerun()
