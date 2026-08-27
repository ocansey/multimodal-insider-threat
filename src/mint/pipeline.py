"""One command, every number.

``run_study`` takes a prepared bundle and produces every table quoted in the
README. Nothing is transcribed by hand, and every output carries a provenance
stamp saying which data it came from — including a loud one if it came from
the synthetic fixture.

The order matters in one place. References for the peer and self baselines are
fitted on the **training window only**, then applied to the test window. It
would be easy, and wrong, to fit a role's centroid using the same days you are
about to score against it: every day would be pulled towards its own group
mean and the most anomalous days would be the ones dragging the mean towards
themselves. That mistake makes results better and is invisible in the output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import evaluate as ev
from . import scoring as sc
from .artifacts import Bundle, stamp_results
from .config import Config
from .training import embed_and_score, train

log = logging.getLogger(__name__)

NORMALISATIONS = ["global", "self", "peer"]
ABLATIONS = ["full", "no_content", "no_context", "behaviour_only"]


@dataclass
class StudyResults:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artefacts: dict[str, Any] = field(default_factory=dict)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables.items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
        with open(out_dir / "run.json", "w", encoding="utf-8") as fh:
            json.dump(self.artefacts, fh, indent=2, default=str)
        log.info("wrote %d tables to %s", len(self.tables), out_dir)


# --------------------------------------------------------------------------
def _ablate(bundle: Bundle, arm: str) -> Bundle:
    """Return a view of the bundle with one modality removed.

    Zeroing rather than deleting, so the architecture is identical across arms
    and the comparison measures the information a modality carries rather than
    the effect of changing model capacity.
    """
    if arm == "full":
        return bundle
    out = Bundle(tokens=bundle.tokens, hours=bundle.hours, flags=bundle.flags,
                 content=bundle.content, context=bundle.context.copy(),
                 index=bundle.index, manifest=bundle.manifest)
    if arm in ("no_content", "behaviour_only"):
        out.content = np.zeros_like(bundle.content)
    if arm in ("no_context", "behaviour_only"):
        for col in out.context.columns:
            if col.endswith("_code"):
                out.context[col] = 0
            elif out.context[col].dtype.kind in "fi":
                out.context[col] = 0.0
    return out


def _rows(bundle: Bundle, split: str) -> np.ndarray:
    return np.nonzero((bundle.index["split"] == split).to_numpy())[0]


def compute_signals(
    model, bundle: Bundle, cfg: Config, normalisation: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Model outputs plus the two distance components, for one normalisation."""
    # Masked-modelling scores are only needed where they will be used, which
    # is the calibration and test windows. Embeddings are needed everywhere,
    # because the peer references are fitted on the training window.
    scored = np.concatenate([_rows(bundle, "calibrate"), _rows(bundle, "test")])
    raw = embed_and_score(
        model, bundle,
        mask_p=float(cfg.model["mask_probability"]),
        seed=cfg.seed,
        score_rows=scored,
    )
    embeddings = raw["day"]
    train_rows = _rows(bundle, "train")

    signals = {
        "masked_behaviour": raw["masked_behaviour"],
        "cross_modal_disagreement": raw["cross_modal_disagreement"],
    }
    meta: dict[str, Any] = {"normalisation": normalisation}

    if normalisation == "peer":
        refs, fallback = sc.fit_peer_references(
            embeddings, bundle.context, train_rows,
            int(cfg.scoring["min_peer_group"]))
        distance, sizes = sc.peer_distance(embeddings, bundle.context,
                                           refs, fallback)
        signals["peer_distance"] = distance
        meta["n_peer_groups"] = len(refs)
        meta["median_peer_group_size"] = float(np.median(sizes))
    elif normalisation == "global":
        ref = sc.fit_global_reference(embeddings, train_rows)
        signals["peer_distance"] = ref.distance(embeddings)
        meta["n_peer_groups"] = 1
    elif normalisation == "self":
        signals["peer_distance"] = sc.self_distance(
            embeddings, bundle.index, int(cfg.scoring["self_history_days"]))
        meta["n_peer_groups"] = 0
    else:
        raise ValueError(f"unknown normalisation '{normalisation}'")

    signals["self_distance"] = sc.self_distance(
        embeddings, bundle.index, int(cfg.scoring["self_history_days"]))
    return signals, meta


def score_and_evaluate(
    signals: dict[str, np.ndarray], bundle: Bundle, cfg: Config,
    label: dict[str, Any],
) -> dict[str, Any]:
    calib = _rows(bundle, "calibrate")
    test = _rows(bundle, "test")
    labels = bundle.index["label"].to_numpy()

    weights = sc.fit_weights(
        signals, labels, calib,
        capacity_fraction=(int(cfg.operations["default_capacity"])
                           / max(1, bundle.index.loc[calib, "user"].nunique())),
        seed=cfg.seed,
    )
    card = sc.combine(signals, calib, weights, label.get("normalisation", "peer"))
    scores = card.score

    test_index = bundle.index.iloc[test].reset_index(drop=True)
    test_scores = scores[test]
    test_labels = labels[test]

    capacities = [int(c) for c in cfg.operations["daily_review_capacity"]]
    default_capacity = int(cfg.operations["default_capacity"])

    sweep = ev.capacity_sweep(test_index, test_scores, capacities)
    campaigns = ev.campaign_detection(test_index, test_scores, default_capacity)
    lo, hi = ev.bootstrap_campaign_recall(test_index, test_scores,
                                          default_capacity, seed=cfg.seed)
    burden = ev.burden_by_group(
        test_index, bundle.context.iloc[test].reset_index(drop=True),
        test_scores, default_capacity, attribute="role")

    headline = {**label, **ev.ranking_metrics(test_labels, test_scores),
                **ev.campaign_summary(campaigns),
                "campaign_recall_lo": lo, "campaign_recall_hi": hi,
                **{f"weight_{k}": round(v, 3) for k, v in weights.items()}}

    return {
        "headline": headline,
        "sweep": sweep.assign(**label),
        "campaigns": campaigns.assign(**label) if not campaigns.empty else campaigns,
        "per_scenario": (ev.per_scenario(campaigns).assign(**label)
                         if not campaigns.empty else pd.DataFrame()),
        "burden": burden.assign(**label) if not burden.empty else pd.DataFrame(),
        "scores": scores,
    }


def run_study(
    bundle: Bundle,
    cfg: Config,
    ablations: list[str] | None = None,
    normalisations: list[str] | None = None,
) -> StudyResults:
    res = StudyResults()
    res.tables["data_summary"] = bundle.summary()
    ablations = ablations or ABLATIONS
    normalisations = normalisations or NORMALISATIONS

    headlines, sweeps, campaigns, scenarios, burdens, histories = [], [], [], [], [], []
    trained: dict[str, Any] = {}

    for arm in ablations:
        view = _ablate(bundle, arm)
        log.info("=== training arm '%s' ===", arm)
        model, report = train(view, cfg.model, cfg.training, cfg.seed)
        trained[arm] = report
        histories.append(report.frame().assign(arm=arm))

        # Every normalisation is scored from the same trained model, so the
        # comparison isolates the choice of reference population rather than
        # confounding it with a different fit.
        for norm in (normalisations if arm == "full" else
                     [cfg.scoring["normalisation"]]):
            signals, meta = compute_signals(model, view, cfg, norm)
            label = {"arm": arm, "normalisation": norm}
            out = score_and_evaluate(signals, view, cfg, label)
            headlines.append({**out["headline"], **meta})
            sweeps.append(out["sweep"])
            if not out["campaigns"].empty:
                campaigns.append(out["campaigns"])
            if not out["per_scenario"].empty:
                scenarios.append(out["per_scenario"])
            if not out["burden"].empty:
                burdens.append(out["burden"])
            if arm == "full" and norm == cfg.scoring["normalisation"]:
                res.artefacts["primary_scores_checksum"] = float(
                    np.round(out["scores"].sum(), 6))

    desc = cfg.describe()
    res.tables["headline"] = stamp_results(pd.DataFrame(headlines), bundle, desc)
    res.tables["capacity_sweep"] = stamp_results(
        pd.concat(sweeps, ignore_index=True), bundle, desc)
    res.tables["training_history"] = pd.concat(histories, ignore_index=True)
    if campaigns:
        res.tables["campaigns"] = stamp_results(
            pd.concat(campaigns, ignore_index=True), bundle, desc)
    if scenarios:
        res.tables["per_scenario"] = stamp_results(
            pd.concat(scenarios, ignore_index=True), bundle, desc)
    if burdens:
        res.tables["alert_burden"] = stamp_results(
            pd.concat(burdens, ignore_index=True), bundle, desc)

    res.artefacts.update({
        "synthetic": bundle.is_synthetic,
        "config": desc,
        "manifest": bundle.manifest,
        "parameters": {a: r.n_parameters for a, r in trained.items()},
        "training_minutes": {a: round(r.seconds / 60, 2) for a, r in trained.items()},
        "torch": torch.__version__,
    })
    return res
