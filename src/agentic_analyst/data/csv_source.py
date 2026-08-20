"""CSV implementation of DataSource.

A SQL/MCP source added later subclasses DataSource the same way; the only contract
is "return a cleaned Dataset".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import registry
from .cleaning import CleaningRules, clean_dataframe, rules_for
from .source import Dataset, DataSource


class CsvSource(DataSource):
    def __init__(self, path: str | Path, rules: CleaningRules | None = None):
        self.path = Path(path)
        # None means "decide from the data": the Telco file gets its hand-written
        # rules, anything else gets inferred ones. Passing rules explicitly still
        # overrides both, which is what the tests rely on.
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

        rules, known = (self.rules, None) if self.rules else rules_for(raw)
        frame, report = clean_dataframe(raw, rules)
        report["source"] = self.origin
        if known is not None:
            report["ruleset"] = "telco (hand-written)" if known else "inferred"
            if not known:
                report["warnings"].insert(
                    0,
                    "Unrecognised dataset: cleaning rules were inferred from the file. "
                    "Numeric columns that failed conversion were left as nulls rather "
                    "than filled, since there is no basis for choosing a value.",
                )

        dataset_id = registry.register(frame)
        return Dataset(
            dataset_id=dataset_id,
            frame=frame,
            origin=self.origin,
            cleaning_report=report,
        )
