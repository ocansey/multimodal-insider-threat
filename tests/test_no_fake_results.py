"""The most important tests in the repository.

This project ships a data simulator so that the pipeline can be tested without
a 1.5 GB download. That is a convenience with a sharp edge: it makes it very
easy to publish a number that came from the simulator and looks exactly like a
number that came from the real thing. Nobody would do it on purpose. Everybody
could do it by accident, at eleven at night, three minutes before pushing.

So the guard is mechanical and it is tested here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mint.artifacts import SyntheticDataError, load, save, stamp_results

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_bundles_refuse_to_load_by_default(bundle, tmp_path):
    save(bundle, tmp_path / "b")
    with pytest.raises(SyntheticDataError):
        load(tmp_path / "b")


def test_synthetic_bundles_load_when_asked_explicitly(bundle, tmp_path):
    save(bundle, tmp_path / "b")
    reloaded = load(tmp_path / "b", allow_synthetic=True)
    assert reloaded.is_synthetic
    assert len(reloaded) == len(bundle)


def test_the_manifest_says_so_in_words(bundle, tmp_path):
    save(bundle, tmp_path / "b")
    with open(tmp_path / "b" / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["synthetic"] is True
    assert "not a result" in manifest["warning"].lower()


def test_results_carry_their_provenance(bundle):
    table = pd.DataFrame({"metric": ["auroc"], "value": [0.9]})
    stamped = stamp_results(table, bundle, "test-config")
    assert stamped["data_source"].iloc[0] == "SYNTHETIC-FIXTURE"


def test_no_committed_result_came_from_the_simulator():
    """Scan everything under reports/ for the synthetic marker.

    If this fails, a results file in the repository was produced from fixture
    data and must be deleted before anyone reads it as a finding.
    """
    reports = ROOT / "reports"
    if not reports.exists():
        pytest.skip("no reports directory yet")

    offenders = []
    for path in reports.rglob("*.csv"):
        try:
            head = pd.read_csv(path, nrows=50)
        except Exception:  # noqa: BLE001 - an unreadable file is not a result
            continue
        if "data_source" in head.columns and \
                (head["data_source"] == "SYNTHETIC-FIXTURE").any():
            offenders.append(path.relative_to(ROOT))

    for path in reports.rglob("*.json"):
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(blob, dict) and blob.get("synthetic"):
            offenders.append(path.relative_to(ROOT))

    assert not offenders, (
        "these files under reports/ were produced from synthetic fixture data "
        f"and must not be published as results: {offenders}"
    )


def test_readme_does_not_quote_numbers_without_a_results_file():
    """A crude but effective check that the README's results table is backed.

    If the README claims a headline table, the corresponding CSV has to exist.
    It is possible to defeat this by writing prose instead of a table, but it
    catches the specific failure mode of pasting a promising number from a
    console into the README and forgetting where it came from.
    """
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("no README yet")
    text = readme.read_text(encoding="utf-8")
    marker = "<!-- results-pending -->"
    has_results = (ROOT / "reports" / "tables" / "headline.csv").exists()
    if marker in text:
        assert not has_results, (
            "the README still carries the results-pending marker but "
            "reports/tables/headline.csv exists — regenerate the README"
        )
