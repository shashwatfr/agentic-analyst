#!/usr/bin/env python
"""Checks the two claims this project actually rests on.

Run with: uv run python verify.py

Neither check calls a model, so this is free and fast. The point is that "no raw data
reaches the LLM" and "the dirty rows are handled deliberately" are things you can run,
not things the README asserts.
"""

from __future__ import annotations


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from agentic_analyst.agents.narrator import normalise_prose  # noqa: E402
from agentic_analyst.config import DEFAULT_DATASET  # noqa: E402
from agentic_analyst.data.cleaning import TELCO_RULES, clean_dataframe  # noqa: E402
from agentic_analyst.data.csv_source import CsvSource  # noqa: E402
from agentic_analyst.data.summaries import (  # noqa: E402
    BudgetExceeded,
    build_schema_card,
    enforce_budget,
)

PASS, FAIL = "  [pass]", "  [FAIL]"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL} {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    print("\n== Cleaning ==")
    dataset = CsvSource(DEFAULT_DATASET).load()
    frame, report = dataset.frame, dataset.cleaning_report

    check("no rows dropped", report["rows_in"] == report["rows_out"] == 7043,
          f"{report['rows_in']} -> {report['rows_out']}")

    coercion = next((c for c in report["coercions"] if c["column"] == "TotalCharges"), None)
    check("TotalCharges coercion recorded", coercion is not None)
    if coercion:
        check("11 unusable rows found", coercion["affected_rows"] == 11,
              f"{coercion['affected_rows']} rows")
        check("all 11 ids logged", len(coercion.get("affected_ids", [])) == 11)
        check("no nulls left behind", coercion["nulls_remaining"] == 0)
        driver = coercion.get("driver_check", {})
        check("fill assumption holds (Tenure == 0 on every affected row)",
              bool(driver.get("holds")), f"observed {driver.get('observed')}")

    check("TotalCharges is numeric", pd.api.types.is_numeric_dtype(frame["TotalCharges"]),
          str(frame["TotalCharges"].dtype))
    check("is_new_customer flags exactly 11", int(frame["is_new_customer"].sum()) == 11)
    check("SeniorCitizen recoded to labels",
          set(frame["SeniorCitizen"].unique()) == {"No", "Yes"})

    print("\n== Schema drift guard ==")
    renamed = pd.read_csv(DEFAULT_DATASET, keep_default_na=False).rename(columns={"Tenure": "tenure"})
    try:
        clean_dataframe(renamed, TELCO_RULES)
        check("missing column raises", False, "it silently accepted a renamed column")
    except Exception as exc:  # noqa: BLE001
        check("missing column raises loudly", "tenure" in str(exc).lower() or "Tenure" in str(exc))

    print("\n== Arbitrary CSVs ==")
    check("Telco file still picks the hand-written rules",
          report.get("ruleset") == "telco (hand-written)", str(report.get("ruleset")))

    import tempfile

    unknown = pd.DataFrame({
        "order_id": [f"O-{i}" for i in range(20)],
        "region": ["N", "S"] * 10,
        "amount": ["10.5"] * 18 + ["  ", "3.25"],
    })
    tmp = Path(tempfile.gettempdir()) / "agentic_analyst_unknown.csv"
    unknown.to_csv(tmp, index=False)
    other = CsvSource(tmp).load()

    check("unknown CSV loads instead of raising",
          len(other.frame) == 20, f"{len(other.frame)} rows")
    check("rules were inferred, not assumed",
          other.cleaning_report.get("ruleset") == "inferred")
    check("numeric-looking text column was coerced",
          pd.api.types.is_numeric_dtype(other.frame["amount"]), str(other.frame["amount"].dtype))
    check("inference is disclosed in the warnings",
          any("inferred" in w for w in other.cleaning_report.get("warnings", [])))
    check("no rows dropped from the unknown CSV",
          other.cleaning_report["rows_in"] == other.cleaning_report["rows_out"] == 20)
    inferred_coercion = other.cleaning_report["coercions"][0]
    check("unknown data is not filled with invented values",
          inferred_coercion["strategy"] == "leave_nan", inferred_coercion["strategy"])
    tmp.unlink(missing_ok=True)

    print("\n== No raw data reaches the LLM ==")
    card = build_schema_card(frame, "verify")
    serialised = enforce_budget(card, "schema_card")

    ids = set(frame["customerID"].astype(str))
    leaked = [i for i in ids if i in serialised]
    check("no customer id appears in the schema card", not leaked,
          f"{len(leaked)} leaked" if leaked else "0 of 7043")

    identifier = next(c for c in card["columns"] if c["name"] == "customerID")
    check("customerID classified as identifier, no labels emitted",
          identifier["kind"] == "identifier" and "labels" not in identifier)

    contract = next(c for c in card["columns"] if c["name"] == "Contract")
    check("low-cardinality labels still exposed (agents need them)",
          contract.get("labels") == ["Monthly", "One year", "Two year"],
          str(contract.get("labels")))

    print("\n== Payload size is flat in dataset size ==")
    doubled = pd.concat([frame, frame], ignore_index=True)
    # Both sides go through enforce_budget so they're serialised identically -
    # comparing an indented dump against a compact one measures the formatting, not
    # the payload.
    small = len(serialised)
    large = len(enforce_budget(build_schema_card(doubled, "verify"), "schema_card"))
    growth = abs(large - small) / small
    check("doubling the rows barely changes the prompt", growth < 0.02,
          f"{len(frame):,} rows -> {small:,} chars | {len(doubled):,} rows -> {large:,} chars "
          f"({growth:.2%} change)")

    print("\n== Narrator prose is console-safe ==")
    # Groq's gpt-oss sprinkles narrow no-break spaces through its output. They look
    # identical to a normal space, as does the non-breaking hyphen it reaches for in
    # compound adjectives, and both take down a cp1252 console the first time anything
    # prints them. The narrator flattens them before they reach state.
    messy = "Churn\u202f=\u202fYes\u200b for 3,875\u00a0month\u2011to\u2011month customers"
    cleaned = normalise_prose(messy)
    check("invisible codepoints stripped from the narrative",
          all(ord(c) < 128 for c in cleaned), repr(cleaned))
    check("the wording survives intact",
          cleaned == "Churn = Yes for 3,875 month-to-month customers")

    print("\n== Budget guard fails loudly ==")
    try:
        enforce_budget({"rows": frame.head(500).to_dict("records")}, "raw rows")
        check("oversized payload raises", False, "it was allowed through")
    except BudgetExceeded:
        check("oversized payload raises BudgetExceeded", True)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
