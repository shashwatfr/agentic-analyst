"""The data source interface.

This exists so the second pass - swapping the CSV for a SQL/database MCP source -
is a new subclass rather than a rewrite. Everything downstream of load() only ever
sees a DataFrame plus a Dataset handle, so nothing else in the package knows or
cares where the rows came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Dataset:
    """A loaded, cleaned frame plus the provenance needed to describe it."""

    dataset_id: str
    frame: pd.DataFrame
    origin: str                      # human-readable: file path, table name, query
    cleaning_report: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Loads rows and hands back a cleaned Dataset."""

    @property
    @abstractmethod
    def origin(self) -> str:
        """Human-readable description of where the data comes from."""

    @abstractmethod
    def load(self) -> Dataset:
        """Read, clean, and register the data. Must return a ready-to-analyse frame."""
