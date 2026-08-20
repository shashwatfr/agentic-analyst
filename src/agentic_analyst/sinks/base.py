"""Where an approved report goes.

Behind an interface so the Drive dependency is one swappable class rather than
something the graph knows about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CommitResult:
    ok: bool
    destination: str
    files: list[str] = field(default_factory=list)
    link: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "destination": self.destination,
            "files": self.files,
            "link": self.link,
            "detail": self.detail,
        }


class ReportSink(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def commit(
        self, report_md: str, chart_paths: list[str], meta: dict[str, Any]
    ) -> CommitResult:
        """Publish the report and its charts. Must not raise - return ok=False."""


def folder_name(meta: dict[str, Any]) -> str:
    """Timestamped folder name shared by every sink so output layout is consistent."""
    stamp = meta.get("timestamp", "")
    slug = meta.get("slug", "analysis")
    return f"{slug}-{stamp}" if stamp else slug


def readable_size(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1024:.0f} KB"
    except OSError:
        return "?"
