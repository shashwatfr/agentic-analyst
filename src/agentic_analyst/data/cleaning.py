"""Explicit cleaning pass, run before any agent sees anything.

Two principles here:

1. Declared rules transform the data; anything the auto-detector finds is *reported*
   but never silently changed. Silent transformation is how you end up explaining a
   number you can't reproduce.
2. Nothing is dropped. Row count in must equal row count out - the report asserts it.

The rules are declarative rather than hardcoded because the file in data/ is a
modified Telco export: columns are TitleCase, Contract uses "Monthly" instead of
"Month-to-month", PaymentMethod collapses the two check types into "Manual", and the
"No internet service" third levels are flattened to Yes/No. Code copied from a
standard Telco tutorial breaks on it, so the schema is asserted at load instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

FillStrategy = Literal["zero", "leave_nan"]

# Below this share of parseable values, a text column is probably genuinely
# categorical and shouldn't be treated as a broken numeric column.
NUMERIC_DETECTION_THRESHOLD = 0.9


@dataclass(frozen=True)
class NumericCoercion:
    """Coerce a text column to numeric and decide what happens to the failures."""

    column: str
    strategy: FillStrategy = "leave_nan"
    flag_column: str | None = None
    # If given, every failing row is expected to hold this value in driver_column.
    # A mismatch means the assumption behind the fill no longer holds, and it gets
    # recorded as a warning rather than quietly applied.
    driver_column: str | None = None
    driver_value: Any = 0


@dataclass(frozen=True)
class CategoricalRecode:
    """Remap a column's values, e.g. an int 0/1 flag into readable labels."""

    column: str
    mapping: dict[Any, Any]


@dataclass(frozen=True)
class CleaningRules:
    required_columns: tuple[str, ...] = ()
    id_column: str | None = None
    numeric_coercions: tuple[NumericCoercion, ...] = ()
    categorical_recodes: tuple[CategoricalRecode, ...] = ()
    # Columns that must not be negative, checked and reported.
    non_negative: tuple[str, ...] = ()


# Rules for the churn file in data/. Written against the header that is actually
# there, verified by profiling the file rather than assumed from the Kaggle version.
TELCO_RULES = CleaningRules(
    required_columns=(
        "customerID", "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn",
    ),
    id_column="customerID",
    numeric_coercions=(
        # 11 rows hold a single space rather than a number. All 11 are Tenure=0 -
        # customers who signed up but haven't been billed a cycle yet - so 0.0 is the
        # honest value, not an imputation. driver_column re-checks that on every run
        # instead of trusting the profile I did once.
        NumericCoercion(
            column="TotalCharges",
            strategy="zero",
            flag_column="is_new_customer",
            driver_column="Tenure",
            driver_value=0,
        ),
    ),
    categorical_recodes=(
        # Stored as int 0/1 while every other binary column is Yes/No. Recoding makes
        # it group and chart consistently with its neighbours.
        CategoricalRecode(column="SeniorCitizen", mapping={0: "No", 1: "Yes", "0": "No", "1": "Yes"}),
    ),
    non_negative=("Tenure", "MonthlyCharges", "TotalCharges"),
)


class SchemaError(RuntimeError):
    """Raised when the file doesn't have the columns the rules expect."""


def _blank_to_na(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Trim text columns and turn empty/whitespace cells into real NA.

    Worth doing first: the dirty cells in this file are single spaces, so isna()
    reports zero nulls and a naive missing-value check finds nothing at all.
    """
    blanks = 0
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            stripped = out[col].astype(str).str.strip()
            mask = stripped.eq("") | stripped.str.lower().isin({"na", "n/a", "null", "none"})
            blanks += int(mask.sum())
            out[col] = stripped.mask(mask, pd.NA)
    return out, blanks


def _detect_numeric_like(frame: pd.DataFrame, already_handled: set[str]) -> list[dict[str, Any]]:
    """Find text columns that are mostly numbers - reported, never auto-converted."""
    findings = []
    for col in frame.columns:
        if col in already_handled or frame[col].dtype != object:
            continue
        values = frame[col].dropna()
        if values.empty:
            continue
        parsed = pd.to_numeric(values, errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio >= NUMERIC_DETECTION_THRESHOLD:
            findings.append(
                {
                    "column": col,
                    "parseable_ratio": round(ratio, 4),
                    "unparseable": int(parsed.isna().sum()),
                }
            )
    return findings


def clean_dataframe(
    raw: pd.DataFrame, rules: CleaningRules = TELCO_RULES
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the rules and return (clean_frame, report). Never drops rows."""
    report: dict[str, Any] = {
        "rows_in": int(len(raw)),
        "columns_in": int(raw.shape[1]),
        "coercions": [],
        "recodes": [],
        "warnings": [],
        "detected_numeric_like": [],
    }

    missing = [c for c in rules.required_columns if c not in raw.columns]
    if missing:
        raise SchemaError(
            f"Dataset is missing expected columns: {missing}. "
            f"Found: {list(raw.columns)}"
        )

    df, blank_count = _blank_to_na(raw)
    report["blank_cells_normalised"] = blank_count

    for rule in rules.numeric_coercions:
        if rule.column not in df.columns:
            report["warnings"].append(f"Coercion rule skipped: no column {rule.column!r}")
            continue

        numeric = pd.to_numeric(df[rule.column], errors="coerce")
        failed = numeric.isna() & df[rule.column].notna()
        blank = df[rule.column].isna()
        affected = failed | blank

        entry: dict[str, Any] = {
            "column": rule.column,
            "strategy": rule.strategy,
            "affected_rows": int(affected.sum()),
            "unparseable_text": int(failed.sum()),
            "blank": int(blank.sum()),
        }

        if rule.driver_column and rule.driver_column in df.columns and affected.any():
            driver_values = df.loc[affected, rule.driver_column]
            consistent = bool((driver_values == rule.driver_value).all())
            entry["driver_check"] = {
                "column": rule.driver_column,
                "expected": rule.driver_value,
                "holds": consistent,
                "observed": sorted({str(v) for v in driver_values.unique()}),
            }
            if not consistent:
                # The fill assumption is dataset-specific; if it stops holding, say so
                # loudly in the report rather than pretending the number is grounded.
                report["warnings"].append(
                    f"{rule.column}: fill assumes {rule.driver_column}=={rule.driver_value} "
                    f"but saw {entry['driver_check']['observed']}"
                )

        if rules.id_column and rules.id_column in df.columns and affected.any():
            entry["affected_ids"] = df.loc[affected, rules.id_column].tolist()

        if rule.strategy == "zero":
            numeric = numeric.fillna(0.0)
        entry["nulls_remaining"] = int(numeric.isna().sum())

        df[rule.column] = numeric.astype(float)
        if rule.flag_column:
            df[rule.flag_column] = affected
            entry["flag_column"] = rule.flag_column

        report["coercions"].append(entry)

    for recode in rules.categorical_recodes:
        if recode.column not in df.columns:
            continue
        before = sorted({str(v) for v in df[recode.column].dropna().unique()})
        df[recode.column] = df[recode.column].map(
            lambda v, m=recode.mapping: m.get(v, m.get(str(v), v))
        )
        after = sorted({str(v) for v in df[recode.column].dropna().unique()})
        report["recodes"].append({"column": recode.column, "before": before, "after": after})

    handled = {r.column for r in rules.numeric_coercions}
    report["detected_numeric_like"] = _detect_numeric_like(df, handled)
    for finding in report["detected_numeric_like"]:
        report["warnings"].append(
            f"{finding['column']} looks numeric "
            f"({finding['parseable_ratio']:.0%} parseable) but was left as text"
        )

    if rules.id_column and rules.id_column in df.columns:
        dupes = int(df[rules.id_column].duplicated().sum())
        report["duplicate_ids"] = dupes
        if dupes:
            report["warnings"].append(f"{dupes} duplicate values in {rules.id_column}")

    for col in rules.non_negative:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            negatives = int((df[col] < 0).sum())
            if negatives:
                report["warnings"].append(f"{col} has {negatives} negative values")

    remaining_nulls = {c: int(n) for c, n in df.isna().sum().items() if n}
    report["remaining_nulls"] = remaining_nulls
    report["rows_out"] = int(len(df))
    report["columns_out"] = int(df.shape[1])

    # The no-silent-drop guarantee, enforced rather than promised.
    if report["rows_out"] != report["rows_in"]:
        raise RuntimeError(
            f"Cleaning changed the row count ({report['rows_in']} -> {report['rows_out']}). "
            "That should never happen."
        )

    return df, report
