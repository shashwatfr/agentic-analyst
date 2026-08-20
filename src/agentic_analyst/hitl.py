"""Console review UI for the interrupt payload.

Rendering only - the decision it returns is handed straight back to the graph as a
Command(resume=...). Keeping this separate from graph.py means a web or Slack review
UI later is a swap of this module, not a change to the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

VALID = {"a": "approve", "approve": "approve", "e": "edit", "edit": "edit",
         "r": "reject", "reject": "reject"}


def render_interrupt(payload: dict[str, Any]) -> None:
    console.rule("[bold]Review required[/bold]")

    report = payload.get("report_md", "")
    console.print(Panel(Markdown(report), title="Draft report", border_style="cyan"))

    charts = payload.get("chart_paths", [])
    if charts:
        console.print("\n[bold]Charts written:[/bold]")
        for chart in charts:
            path = Path(chart)
            marker = "[green]OK[/green]" if path.exists() else "[red]missing[/red]"
            console.print(f"  {marker}  {path}")

    errors = payload.get("errors", [])
    if errors:
        console.print("\n[bold yellow]Issues logged during the run:[/bold yellow]")
        for error in errors:
            console.print(f"  - {error}")

    remaining = payload.get("revisions_remaining", 0)
    console.print(f"\n[dim]Revisions remaining: {remaining}[/dim]")


def prompt_decision(payload: dict[str, Any]) -> dict[str, str]:
    """Ask for approve / edit / reject. Only offers edit if revisions remain."""
    options = ["approve", "reject"] if payload.get("revisions_remaining", 0) <= 0 else [
        "approve", "edit", "reject"
    ]
    console.print(f"\n[bold]Decision[/bold] ({' / '.join(options)})")

    while True:
        raw = Prompt.ask("  >", default="approve").strip().lower()
        decision = VALID.get(raw)
        if decision is None:
            console.print("  [red]Enter approve, edit, or reject.[/red]")
            continue
        if decision == "edit" and "edit" not in options:
            console.print("  [red]No revisions remaining - approve or reject.[/red]")
            continue

        feedback = ""
        if decision == "edit":
            feedback = Prompt.ask("  What should change?").strip()
            if not feedback:
                console.print("  [red]An edit needs feedback to act on.[/red]")
                continue
        return {"decision": decision, "feedback": feedback}


def report_commit(result: dict[str, Any]) -> None:
    if not result:
        console.print("\n[yellow]Nothing was committed.[/yellow]")
        return

    if result.get("ok"):
        console.print(f"\n[bold green]Committed to {result.get('destination')}[/bold green]")
        console.print(f"  {result.get('detail', '')}")
        if result.get("link"):
            console.print(f"  [link]{result['link']}[/link]")
    else:
        console.print(f"\n[bold red]Commit failed:[/bold red] {result.get('detail')}")
