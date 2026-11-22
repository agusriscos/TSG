"""Paper-suite infrastructure: determinism, persistence, plot-only."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import build_ensemble_records, realization_seed
from tsg.audit.pipeline import AuditPipeline
from tsg.descriptors.base import DESCRIPTOR_IDS


def _tiny_config() -> AuditConfig:
    return AuditConfig(
        n_steps=64,
        n_realizations=2,
        seed=7,
        n_folds=2,
        profile="dev",
        brownian_mode="levels",
        regimes=("white_noise", "logistic"),
        methods=("HVG", "STN"),
        families={"visibility": ("HVG",), "transition": ("STN",)},
        critical_pairs=(("white_noise", "logistic"),),
        figure_dpi=80,
    )


def test_workers_determinism_tiny_ensemble() -> None:
    """``--workers 1`` and ``--workers 2`` must yield identical descriptor matrices."""
    cfg = _tiny_config()
    seq = build_ensemble_records(cfg, n_workers=1)
    par = build_ensemble_records(cfg, n_workers=2)
    assert len(seq) == len(par) == 4
    for a, b in zip(seq, par):
        assert a["dataset_id"] == b["dataset_id"]
        assert a["realization"] == b["realization"]
        assert a["seed"] == b["seed"]
        pd.testing.assert_frame_equal(
            a["matrix"].sort_index(axis=0).sort_index(axis=1),
            b["matrix"].sort_index(axis=0).sort_index(axis=1),
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )


def test_realization_seed_stable() -> None:
    assert realization_seed(42, "white_noise", 0) == realization_seed(42, "white_noise", 0)
    assert realization_seed(42, "white_noise", 0) != realization_seed(42, "logistic", 0)


def test_persist_artifacts_and_plot_only(tmp_path: Path) -> None:
    cfg = _tiny_config()
    out = tmp_path / "audit"
    pipeline = AuditPipeline(cfg, lang="en", n_workers=2)
    result = pipeline.run(output_dir=out)
    assert result.verdict is not None

    data = out / "data"
    assert (data / "features_raw.npz").exists()
    assert (data / "features_clean.npz").exists()
    assert (data / "oof_predictions.npz").exists()
    assert (out / "config_snapshot.json").exists()
    assert (out / "metrics.json").exists()
    assert (data / "tables" / "ablation.csv").exists()

    # PDF + SVG + PNG for at least one figure.
    pngs = list((out / "figures").glob("*.png"))
    assert pngs
    for png in pngs:
        assert png.with_suffix(".pdf").exists()
        assert png.with_suffix(".svg").exists()

    # Mutate a figure timestamp by removing PDFs, then plot-only restores them.
    for pdf in (out / "figures").glob("*.pdf"):
        pdf.unlink()
    pipeline.plot_only(out, lang="en")
    for png in (out / "figures").glob("*.png"):
        assert png.with_suffix(".pdf").exists()

    # Features NPZ round-trip.
    with np.load(data / "features_clean.npz", allow_pickle=True) as z:
        assert "X" in z.files
        assert "feature_names" in z.files
        assert "y" in z.files
        assert z["X"].shape[0] == len(cfg.regimes) * cfg.n_realizations


def test_builtin_paper_profile_lengths() -> None:
    from tsg.audit.config import BUILTIN_PROFILES, apply_profile
    from tsg.audit.pareto import timing_lengths_for_profile

    assert BUILTIN_PROFILES["paper"]["n_steps"] == 1500
    paper = apply_profile(AuditConfig(), "paper")
    assert paper.n_steps == 1500
    assert timing_lengths_for_profile("paper") == (500, 1000, 1500)
