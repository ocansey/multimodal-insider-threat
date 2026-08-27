"""Fixtures. The synthetic organisation is built once per session and reused."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mint.config import load_config  # noqa: E402
from mint.prepare import prepare  # noqa: E402
from mint.simulate import build_fixture  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    """A small synthetic organisation written to a temporary directory.

    Deliberately tiny — twenty people over sixty days. The tests here check
    invariants, not performance, and every one of them should hold on twenty
    people as firmly as on a thousand.
    """
    out = tmp_path_factory.mktemp("cert_fixture")
    build_fixture(out, n_users=20, n_days=60, n_insiders=3,
                  campaign_length_days=(6, 12))
    return out


@pytest.fixture(scope="session")
def bundle(fixture_dir, cfg):
    return prepare(fixture_dir, cfg, text_encoder_kind="hashing", synthetic=True)
