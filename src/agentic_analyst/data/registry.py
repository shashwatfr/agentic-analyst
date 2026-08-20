"""In-process store for loaded DataFrames.

The graph state carries a dataset_id string; nodes exchange it for the real frame
here. This is what keeps the frame out of state (and therefore out of both the
checkpointer and every prompt) while still letting any node do real pandas work.
"""

from __future__ import annotations

import uuid

import pandas as pd

_FRAMES: dict[str, pd.DataFrame] = {}


def register(frame: pd.DataFrame, dataset_id: str | None = None) -> str:
    dataset_id = dataset_id or f"ds_{uuid.uuid4().hex[:12]}"
    _FRAMES[dataset_id] = frame
    return dataset_id


def get(dataset_id: str) -> pd.DataFrame:
    try:
        return _FRAMES[dataset_id]
    except KeyError as exc:
        # Most likely cause: resuming a checkpointed thread in a fresh process, where
        # state survived but the in-memory frame did not. The resume path reloads.
        raise KeyError(
            f"Dataset {dataset_id!r} is not loaded in this process. "
            "Reload the source before resuming."
        ) from exc


def has(dataset_id: str) -> bool:
    return dataset_id in _FRAMES


def clear() -> None:
    _FRAMES.clear()
