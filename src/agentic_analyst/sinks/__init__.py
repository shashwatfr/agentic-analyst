"""Report sinks, and the factory that picks one.

Local storage is the default and always works. Drive over MCP is opt-in: it only
engages when DRIVE_MCP_COMMAND is set, and it falls back to local if the server can't
actually write. Both entrypoints (run.py, app.py) go through build_sink so there is
one place that decides.
"""

from __future__ import annotations

from pathlib import Path

from ..config import OUTPUT_DIR, Settings
from .base import CommitResult, ReportSink
from .drive_mcp import DriveMCPSink
from .local import LocalReportSink

__all__ = ["CommitResult", "DriveMCPSink", "LocalReportSink", "ReportSink", "build_sink"]


def build_sink(settings: Settings, output_dir: Path | None = None) -> ReportSink:
    """Local by default; Drive only when it has been deliberately configured.

    Defaulting to local is the whole reason a fresh clone runs with nothing but an
    OpenAI key. Drive stays implemented behind the same ABC, so switching to a
    write-capable MCP server is a .env change rather than a code change.
    """
    local = LocalReportSink(output_dir or OUTPUT_DIR)
    if not settings.drive.enabled:
        return local
    return DriveMCPSink(settings.drive, fallback=local)
