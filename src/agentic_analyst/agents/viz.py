"""Viz agent: chooses charts declaratively, then renders them here.

The model returns ChartSpec objects naming a result table and a chart kind. This
module does the drawing. That means no model-authored code is ever executed, and a
nonsense spec degrades to a skipped chart instead of a traceback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

# Agg before pyplot: the pipeline runs headless and must not try to open a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from ..config import OUTPUT_DIR, Settings  # noqa: E402
from ..data.summaries import collect_tables, enforce_budget, table_preview  # noqa: E402
from ..models import get_structured_model  # noqa: E402
from ..state import AnalysisState, VizPlan  # noqa: E402

SYSTEM = """You choose charts for an analysis report.

You are given the question and a catalog of computed result tables. Choose 1-3 charts
that best support the findings.

Rules:
- `source_key` must be exactly one of the catalog keys given to you.
- `x` and `y` must be column names present in that table.
- Use `bar`/`barh` to compare categories, `line` for an ordered trend, `hist` for a
  distribution, `box` to compare distributions, `scatter` for two numerics, and
  `heatmap` only for a correlation matrix.
- Prefer few clear charts to many redundant ones.
- Titles should state the finding, not the mechanic: "Month-to-month customers churn
  most", not "Bar chart of churn by contract".
"""

PALETTE = "colorblind"
FIGSIZE = (9, 5.5)
DPI = 150


def _slug(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (cleaned or fallback)[:60]


def _to_frame(table: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(table.get("rows", []))


def render_chart(spec: dict[str, Any], table: dict[str, Any], index: int, outdir: Path) -> Path:
    """Draw one spec. Raises on a bad spec so the caller can skip just that chart."""
    frame = _to_frame(table)
    if frame.empty:
        raise ValueError("source table is empty")

    kind = spec.get("kind", "bar")
    x, y, hue = spec.get("x", ""), spec.get("y", ""), spec.get("hue", "")
    for name, value in (("x", x), ("y", y), ("hue", hue)):
        if value and value not in frame.columns:
            raise ValueError(f"{name}={value!r} is not a column in the source table")

    sns.set_theme(style="whitegrid", palette=PALETTE)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if kind == "heatmap":
        numeric = frame.select_dtypes("number")
        labels = frame[x] if x and x in frame.columns else numeric.index
        sns.heatmap(numeric, annot=True, fmt=".2f", cmap="vlag", center=0,
                    yticklabels=list(labels), ax=ax)
    elif kind == "hist":
        sns.histplot(data=frame, x=x or y, hue=hue or None, bins=30, ax=ax)
    elif kind == "box":
        sns.boxplot(data=frame, x=x or None, y=y or None, hue=hue or None, ax=ax)
    elif kind == "scatter":
        sns.scatterplot(data=frame, x=x, y=y, hue=hue or None, alpha=0.7, ax=ax)
    elif kind == "line":
        sns.lineplot(data=frame, x=x, y=y, hue=hue or None, marker="o", ax=ax)
    elif kind == "barh":
        sns.barplot(data=frame, y=x, x=y, hue=hue or None, ax=ax)
    else:
        sns.barplot(data=frame, x=x, y=y, hue=hue or None, ax=ax)
        # Long category labels overlap badly at this figure width.
        if frame[x].astype(str).str.len().max() > 10:
            ax.tick_params(axis="x", rotation=20)

    ax.set_title(spec.get("title", ""), fontsize=13, weight="bold", pad=12)
    fig.tight_layout()

    path = outdir / f"chart_{index:02d}_{_slug(spec.get('title', ''), f'chart{index}')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def make_viz_node(settings: Settings, outdir: Path | None = None):
    outdir = outdir or OUTPUT_DIR

    def viz_node(state: AnalysisState) -> dict[str, Any]:
        tables = collect_tables(state.get("query_result", {}), state.get("analysis", {}))
        if not tables:
            return {"chart_specs": [], "chart_paths": [], "errors": ["no result tables to chart"]}

        model = get_structured_model("viz", settings, VizPlan)
        payload = enforce_budget(
            {"catalog": {k: table_preview(v, max_rows=8) for k, v in tables.items()}},
            "viz prompt",
        )
        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Question: {state['question']}\n\nResult catalog:\n{payload}"),
        ]
        plan: VizPlan = model.invoke(messages)

        outdir.mkdir(parents=True, exist_ok=True)
        paths, specs, errors = [], [], []

        for i, chart in enumerate(plan.charts, start=1):
            spec = chart.model_dump()
            table = tables.get(spec.get("source_key", ""))
            if table is None:
                errors.append(
                    f"chart '{spec.get('title')}': unknown source_key "
                    f"{spec.get('source_key')!r}"
                )
                continue
            try:
                path = render_chart(spec, table, i, outdir)
            except Exception as exc:  # noqa: BLE001 - one bad chart, not a dead run
                errors.append(f"chart '{spec.get('title')}' failed: {exc}")
                continue
            spec["path"] = str(path)
            specs.append(spec)
            paths.append(str(path))

        return {"chart_specs": specs, "chart_paths": paths, "errors": errors}

    return viz_node
