"""Tests for HTML report generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.registry import generate_suite
from tsg.reporting.html_report import HTMLReportBuilder


def test_html_report_writes_file(tmp_path: Path) -> None:
    cfg = GeneratorSuiteConfig()
    cfg = replace(
        cfg,
        global_params=replace(cfg.global_params, n_steps=128, seed=3),
        lorenz=replace(cfg.lorenz, burn_in=20),
        logistic=replace(cfg.logistic, burn_in=10),
    )
    series_list = generate_suite(cfg)
    out = tmp_path / "summary.html"
    path = HTMLReportBuilder().write(
        series_list,
        out,
        seed=3,
        n_steps=128,
        also_latest=True,
    )
    html = path.read_text(encoding="utf-8")
    assert "white_noise" in html
    assert "lorenz" in html
    assert (tmp_path / "latest.html").exists()
