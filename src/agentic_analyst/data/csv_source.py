"""CSV implementation of DataSource.

A SQL/MCP source added later subclasses DataSource the same way; the only contract
is "return a cleaned Dataset".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import registry
from .cleaning import CleaningRules, TELCO_RULES, clean_dataframe
from .source import Dataset, DataSource


class CsvSource(DataSource):
    def __init__(self, path: str | Path, rules: CleaningRules = TELCO_RULES):
        self.path = Path(path)
        self.rules = rules

    @property
    def origin(self) -> str:
        return str(self.path.name)

    def load(self) -> Dataset:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        # keep_default_na=False so the whitespace cells arrive as-is and the cleaning
        # step gets to account for them, rather than pandas quietly nulling some of
        # them on the way in.
        raw = pd.read_csv(self.path, keep_default_na=False, na_values=[])
        frame, report = clean_dataframe(raw, self.rules)
        report["source"] = self.origin

        dataset_id = registry.register(frame)
        return Dataset(
            dataset_id=dataset_id,
            frame=frame,
            origin=self.origin,
            cleaning_report=report,
        )
