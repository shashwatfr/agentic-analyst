#!/usr/bin/env python
"""CLI entrypoint.

Two ways to drive the human-in-the-loop gate:

    uv run python run.py "Which contract type has the highest churn rate?"
        Runs to the review gate, shows the draft, prompts, resumes in-process.

    uv run python run.py --resume <thread_id> --decision approve
        Picks up a graph that is already paused, in a brand new process. State comes
        back from the SQLite checkpoint - nothing is held in memory between runs.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# The Windows console defaults to cp1252, and models cheerfully emit things like
# narrow no-break spaces inside numbers. Without this the whole run dies at the
# point where it tries to *print* a perfectly good report.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

# src layout without an install step, so a fresh clone runs straight away.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_analyst.config import (  # noqa: E402
    DEFAULT_DATASET,
    OUTPUT_DIR,
    configure_tracing,
    ensure_directories,
    load_settings,
)
from agentic_analyst.data.csv_source import CsvSource  # noqa: E402
from agentic_analyst.hitl import console, prompt_decision, render_interrupt, report_commit  # noqa: E402
from agentic_analyst.sinks import build_sink  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question about a CSV dataset.")
    parser.add_argument("question", nargs="?", help="Plain-English question about the data.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to the CSV.")
    parser.add_argument("--thread", default="", help="Thread id (defaults to a new one).")
    parser.add_argument("--resume", default="", help="Resume a paused thread by id.")
    parser.add_argument(
        "--decision",
        choices=["approve", "edit", "reject"],
        default="",
        help="Decision to resume with, for non-interactive use.",
    )
    parser.add_argument("--feedback", default="", help="Feedback text when --decision edit.")
    parser.add_argument(
        "--no-interaction",
        action="store_true",
        help="Stop at the review gate and print the thread id instead of prompting.",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace):
    settings = load_settings()
    tracing = configure_tracing(settings)
    ensure_directories()

    # Imported after configure_tracing so the LangChain clients pick up the env.
    from agentic_analyst.graph import build_graph

    sink = build_sink(settings, OUTPUT_DIR)

    console.print(
        f"[dim]models: query={settings.query_model} analysis={settings.analysis_model} "
        f"viz={settings.viz_model} narrator={settings.narrator_model}[/dim]"
    )
    console.print(
        f"[dim]tracing: {'on -> ' + settings.tracing_project if tracing else 'off'} | "
        f"sink: {sink.name}[/dim]\n"
    )

    source = CsvSource(args.dataset)
    return build_graph(settings, source, sink), source


def resolve_interrupt(result: dict) -> dict | None:
    """Pull the interrupt payload out of a paused invoke result."""
    pending = result.get("__interrupt__") if isinstance(result, dict) else None
    if not pending:
        return None
    first = pending[0]
    return getattr(first, "value", first)


def main() -> int:
    args = parse_args()
    if not args.question and not args.resume:
        console.print("[red]Give a question, or --resume a paused thread.[/red]")
        return 2

    graph, source = build_pipeline(args)

    thread_id = args.resume or args.thread or f"run-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    from langgraph.types import Command

    if args.resume:
        # Resuming in a fresh process: the checkpoint has the state but the frame
        # itself was never in it, so reload the source to repopulate the registry.
        source.load()
        if not args.decision:
            console.print("[red]--resume needs --decision.[/red]")
            return 2
        console.print(f"[dim]resuming thread {thread_id} with '{args.decision}'[/dim]\n")
        result = graph.invoke(
            Command(resume={"decision": args.decision, "feedback": args.feedback}), config
        )
    else:
        console.print(f"[dim]thread {thread_id}[/dim]\n")
        result = graph.invoke({"question": args.question, "errors": []}, config)

    # Loop because an 'edit' decision runs the graph back round to another review.
    while True:
        payload = resolve_interrupt(result)
        if payload is None:
            break

        render_interrupt(payload)

        if args.no_interaction:
            console.print(
                f"\n[yellow]Paused at review.[/yellow] Resume with:\n"
                f"  uv run python run.py --resume {thread_id} --decision approve"
            )
            return 0

        decision = prompt_decision(payload)
        result = graph.invoke(Command(resume=decision), config)

    review = (result or {}).get("review") or {}
    if review.get("decision") == "reject":
        console.print("\n[yellow]Rejected. Nothing was committed.[/yellow]")
        return 0

    report_commit((result or {}).get("commit_result", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
