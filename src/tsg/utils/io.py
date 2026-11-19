"""Persist generated series to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from tsg.generators.base import GeneratedSeries

logger = logging.getLogger(__name__)


def save_series_csv(series: GeneratedSeries, path: Path | str) -> Path:
    """Write a single series as a two-column CSV (index, value)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"t": range(series.length), "value": series.values})
    frame.to_csv(out, index=False)
    logger.info("Wrote %s (%d points)", out, series.length)
    return out


def save_suite(
    series_list: list[GeneratedSeries],
    output_dir: Path | str,
) -> Path:
    """Save each series as CSV plus a JSON metadata index."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for series in series_list:
        csv_name = f"{series.dataset_id}.csv"
        save_series_csv(series, root / csv_name)
        meta = series.to_metadata()
        meta["csv"] = csv_name
        index.append(meta)
    meta_path = root / "metadata.json"
    meta_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Wrote suite metadata to %s", meta_path)
    return root
