"""Local filesystem sink - the always-works fallback.

Also what runs when no Drive MCP server is configured, so the pipeline is fully
demoable on a machine with no Google credentials at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..config import OUTPUT_DIR
from .base import CommitResult, ReportSink, folder_name


class LocalReportSink(ReportSink):
    def __init__(self, root: Path | None = None):
        self.root = root or OUTPUT_DIR

    @property
    def name(self) -> str:
        return "local filesystem"

    def commit(
        self, report_md: str, chart_paths: list[str], meta: dict[str, Any]
    ) -> CommitResult:
        try:
            destination = self.root / folder_name(meta)
            destination.mkdir(parents=True, exist_ok=True)

            report_path = destination / "report.md"
            report_path.write_text(report_md, encoding="utf-8")
            written = [str(report_path)]

            for chart in chart_paths:
                source = Path(chart)
                if not source.exists():
                    continue
                # Copy rather than move: charts stay where the viz node put them so a
                # re-run or a second sink can still find them.
                target = destination / source.name
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                written.append(str(target))

            return CommitResult(
                ok=True,
                destination=self.name,
                files=written,
                link=destination.as_uri(),
                detail=f"Wrote {len(written)} files to {destination}",
            )
        except Exception as exc:  # noqa: BLE001 - a sink must never kill the graph
            return CommitResult(
                ok=False, destination=self.name, detail=f"{type(exc).__name__}: {exc}"
            )
