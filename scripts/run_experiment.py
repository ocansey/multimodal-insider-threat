#!/usr/bin/env python3
"""Train, score and evaluate. Writes every table the README quotes.

    # smoke test: build a synthetic organisation and run the whole pipeline
    python scripts/run_experiment.py --fixture

    # the real thing, after scripts/prepare_local.py has produced artefacts
    python scripts/run_experiment.py --artifacts data/artifacts/cert

    # just the normalisation experiment, no modality ablations (much faster)
    python scripts/run_experiment.py --artifacts data/artifacts/cert --arms full

Results land in ``reports/tables``. With ``--fixture`` they land in
``reports/fixture`` instead and are stamped SYNTHETIC-FIXTURE, because a
pipeline check is not a finding and must never be filed as one.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mint.artifacts import load, save  # noqa: E402
from mint.config import load_config  # noqa: E402
from mint.pipeline import ABLATIONS, NORMALISATIONS, run_study  # noqa: E402
from mint.prepare import prepare  # noqa: E402
from mint.simulate import build_fixture  # noqa: E402


def build_fixture_bundle(cfg, users: int, days: int):
    raw = cfg.path("fixtures")
    logging.info("generating a synthetic organisation of %d people over %d days",
                 users, days)
    build_fixture(raw, n_users=users, n_days=days,
                  n_insiders=max(3, users // 20))
    bundle = prepare(raw, cfg, text_encoder_kind="hashing", synthetic=True)
    save(bundle, cfg.path("artifacts") / "fixture")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=None,
                    help="directory written by scripts/prepare_local.py")
    ap.add_argument("--fixture", action="store_true",
                    help="run on synthetic data instead (pipeline check only)")
    ap.add_argument("--fixture-users", type=int, default=60)
    ap.add_argument("--fixture-days", type=int, default=120)
    ap.add_argument("--arms", nargs="*", default=None, choices=ABLATIONS,
                    help=f"modality ablations to run (default: {' '.join(ABLATIONS)})")
    ap.add_argument("--normalisations", nargs="*", default=None,
                    choices=NORMALISATIONS)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the configured epoch count")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    pd.set_option("display.width", 200)

    cfg = load_config()
    cfg.ensure_dirs()
    if args.epochs:
        cfg.as_dict()["training"]["epochs"] = args.epochs

    if args.fixture:
        bundle = build_fixture_bundle(cfg, args.fixture_users, args.fixture_days)
        default_out = cfg.path("reports") / "fixture"
    elif args.artifacts:
        bundle = load(args.artifacts)
        default_out = cfg.path("tables")
    else:
        raise SystemExit(
            "pass --artifacts <dir> for a real run, or --fixture for a "
            "pipeline check. There is no default, on purpose."
        )

    print(bundle.summary().to_string(index=False), "\n")
    if bundle.is_synthetic:
        print("!" * 70)
        print("SYNTHETIC DATA. This run validates the pipeline. Nothing it")
        print("produces is a result and nothing it produces belongs in a README.")
        print("!" * 70, "\n")

    started = time.time()
    results = run_study(bundle, cfg, ablations=args.arms,
                        normalisations=args.normalisations)
    logging.info("study finished in %.1f minutes", (time.time() - started) / 60)

    out = args.out or default_out
    results.save(out)

    head = results.tables["headline"]
    cols = [c for c in ["arm", "normalisation", "auroc", "average_precision",
                        "campaigns", "campaigns_detected", "campaign_recall",
                        "median_days_into_campaign"] if c in head.columns]
    print("\nHeld-out window\n")
    print(head[cols].round(3).to_string(index=False))

    sweep = results.tables["capacity_sweep"]
    scols = [c for c in ["arm", "normalisation", "capacity_per_day", "precision",
                         "recall_user_days", "distinct_users_alerted",
                         "alert_concentration"] if c in sweep.columns]
    print("\nWhat the analyst queue looks like\n")
    print(sweep[scols].round(3).to_string(index=False))

    if "alert_burden" in results.tables:
        burden = results.tables["alert_burden"]
        if "over_alerting_ratio" in burden.columns:
            print("\nWhich roles absorb the alert budget (top 5)\n")
            top = burden.sort_values("over_alerting_ratio", ascending=False).head(5)
            print(top[[c for c in ["arm", "normalisation", "role",
                                   "share_of_alerts", "over_alerting_ratio",
                                   "precision"] if c in top.columns]]
                  .round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
