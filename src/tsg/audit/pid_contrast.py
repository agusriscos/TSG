"""Side-by-side MMI vs BROJA PID contrast for the RQ1 report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from tsg.audit.information import FAMILY_SHORT
from tsg.audit.pid_advanced import AdvancedPIDAtom

logger = logging.getLogger(__name__)

_ATOM_COLS = ("redundancy", "unique_i", "unique_j", "synergy")


def _pair_key(family_i: str, family_j: str) -> str:
    return f"{FAMILY_SHORT[family_i]}|{FAMILY_SHORT[family_j]}"


def mmi_atoms_from_metrics_json(path: Path | str) -> pd.DataFrame:
    """Load global MMI pairwise atoms from ``information_metrics.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("pid_pairs") or []
    if not rows:
        raise KeyError(f"No pid_pairs in {path}")
    frame = pd.DataFrame(rows)
    frame["estimator"] = "mmi"
    frame["pair"] = [
        _pair_key(i, j) for i, j in zip(frame["family_i"], frame["family_j"])
    ]
    return frame


def broja_global_frame(atoms: Sequence[AdvancedPIDAtom]) -> pd.DataFrame:
    """Select global-scope BROJA/CCS atoms."""
    rows = [
        atom.as_dict()
        for atom in atoms
        if atom.scope == "global"
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["pair"] = [
        _pair_key(i, j) for i, j in zip(frame["family_i"], frame["family_j"])
    ]
    return frame


def contrast_mmi_vs_broja(
    mmi: pd.DataFrame,
    broja: pd.DataFrame,
) -> pd.DataFrame:
    """Wide table: one row per family pair with MMI and BROJA atoms + deltas."""
    if mmi.empty or broja.empty:
        raise ValueError("Both MMI and BROJA frames must be non-empty")
    left = mmi.set_index("pair")
    right = broja.set_index("pair")
    pairs = [p for p in left.index if p in right.index]
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        m_row = left.loc[pair]
        b_row = right.loc[pair]
        row: dict[str, Any] = {
            "pair": pair,
            "family_i": m_row["family_i"],
            "family_j": m_row["family_j"],
            "broja_estimator": b_row.get("estimator", "broja"),
            "broja_n_bins": b_row.get("n_bins", float("nan")),
            "broja_n_samples": b_row.get("n_samples", float("nan")),
        }
        for col in _ATOM_COLS:
            row[f"mmi_{col}"] = float(m_row[col])
            row[f"broja_{col}"] = float(b_row[col])
            row[f"delta_{col}"] = float(b_row[col]) - float(m_row[col])
        # Highlight MMI's artificial unique zeros.
        row["mmi_forced_unique_zero"] = bool(
            float(m_row["unique_i"]) == 0.0 or float(m_row["unique_j"]) == 0.0
        )
        row["broja_both_unique_positive"] = bool(
            float(b_row["unique_i"]) > 1e-6 and float(b_row["unique_j"]) > 1e-6
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    logger.info(
        "MMI vs BROJA contrast: %d pairs; BROJA both-unique=%d / MMI forced-zero=%d",
        len(frame),
        int(frame["broja_both_unique_positive"].sum()),
        int(frame["mmi_forced_unique_zero"].sum()),
    )
    return frame


def contrast_to_markdown(frame: pd.DataFrame, *, float_fmt: str = ".4f") -> str:
    """Compact Markdown table for the report (global pairs only)."""
    if frame.empty:
        return "_No MMI/BROJA contrast rows._\n"
    cols = [
        "pair",
        "mmi_redundancy",
        "broja_redundancy",
        "mmi_unique_i",
        "broja_unique_i",
        "mmi_unique_j",
        "broja_unique_j",
        "mmi_synergy",
        "broja_synergy",
        "mmi_forced_unique_zero",
        "broja_both_unique_positive",
    ]
    present = [c for c in cols if c in frame.columns]
    show = frame[present].copy()
    for col in present:
        if show[col].dtype.kind == "f":
            show[col] = show[col].map(lambda v, fmt=float_fmt: format(float(v), fmt))
    header = "| " + " | ".join(present) + " |"
    sep = "| " + " | ".join("---" for _ in present) + " |"
    body = "\n".join(
        "| " + " | ".join(str(show.iloc[i][c]) for c in present) + " |"
        for i in range(len(show))
    )
    note = (
        "\n\nNotes: `mmi_forced_unique_zero` marks pairs where MMI set at least "
        "one unique atom to exactly 0. `broja_both_unique_positive` marks pairs "
        "where BROJA assigns strictly positive unique information to both families. "
        "BROJA atoms come from quantile-binned family PCA scores; MMI atoms come "
        "from the predictive-MI pipeline — absolute bit scales are not directly "
        "comparable, but the unique-atom pattern is.\n"
    )
    return "\n".join(
        [
            "## PID contrast: MMI vs BROJA (global pairs)",
            "",
            header,
            sep,
            body,
            note,
        ]
    )


def write_mmi_broja_contrast(
    *,
    mmi_metrics_json: Path | str,
    broja_atoms: Sequence[AdvancedPIDAtom],
    tables_dir: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    """Write CSV + Markdown contrast tables under ``tables_dir``."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    mmi = mmi_atoms_from_metrics_json(mmi_metrics_json)
    broja = broja_global_frame(broja_atoms)
    contrast = contrast_mmi_vs_broja(mmi, broja)
    csv_path = tables_dir / "pid_mmi_vs_broja.csv"
    md_path = tables_dir / "pid_mmi_vs_broja.md"
    contrast.to_csv(csv_path, index=False, float_format="%.10g")
    md_path.write_text(contrast_to_markdown(contrast), encoding="utf-8")
    logger.info("Wrote MMI vs BROJA contrast → %s , %s", csv_path, md_path)
    return contrast, csv_path, md_path
