"""Invariants that decide whether the numbers mean anything.

Each of these guards a specific mistake that would make results better and
would not look like an error in any output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mint.evaluate import (
    campaign_detection,
    daily_queue,
    queue_metrics,
    ranking_metrics,
)
from mint.model import apply_masking
from mint.schema import MASK_ID, PAD_ID
from mint.scoring import robust_z, self_distance
from mint.sessionise import token_for, typed_events
from mint.training import context_cardinalities


# -- sessionisation ---------------------------------------------------------
def test_the_day_boundary_is_four_am(fixture_dir):
    """An event at 02:00 belongs to the previous working day, not its own."""
    events = typed_events(fixture_dir, day_start_hour=4)
    small_hours = events[events["hour"] < 4]
    if small_hours.empty:
        pytest.skip("the fixture produced no events before 4am")
    row = small_hours.iloc[0]
    assert row["day"].date() == (row["ts"] - pd.Timedelta(hours=4)).date()
    assert row["day"] < row["ts"].normalize()


def test_empty_cc_fields_do_not_make_every_email_external():
    """The bug that made the internal class disappear entirely."""
    row = pd.Series({"from": "AAA0001@company.invalid", "user": "AAA0001",
                     "to": "AAA0002@company.invalid", "cc": np.nan,
                     "bcc": np.nan, "attachments": 0})
    assert token_for("email", row) == "email_send_internal"


def test_a_genuinely_external_recipient_is_detected():
    row = pd.Series({"from": "AAA0001@company.invalid", "user": "AAA0001",
                     "to": "someone@elsewhere.invalid", "cc": np.nan,
                     "bcc": np.nan, "attachments": 0})
    assert token_for("email", row) == "email_send_external"


def test_mail_from_someone_else_is_a_receipt():
    row = pd.Series({"from": "AAA0009@company.invalid", "user": "AAA0001",
                     "to": "AAA0001@company.invalid", "cc": "", "bcc": "",
                     "attachments": 0})
    assert token_for("email", row) == "email_receive"


def test_removable_media_writes_are_typed_separately():
    assert token_for("file", pd.Series({"filename": r"R:\\dump.zip"})) \
        == "file_copy_to_removable"
    assert token_for("file", pd.Series({"filename": r"C:\\work\\notes.docx"})) \
        == "file_open"


# -- splits and labels ------------------------------------------------------
def test_splits_are_chronological_and_disjoint(bundle):
    days = bundle.index.groupby("split")["day"].agg(["min", "max"])
    assert days.loc["train", "max"] < days.loc["calibrate", "min"]
    assert days.loc["calibrate", "max"] < days.loc["test", "min"]


def test_no_known_malicious_day_is_in_the_training_window(bundle):
    train = bundle.index[bundle.index["split"] == "train"]
    assert int(train["label"].sum()) == 0, (
        "a detector fitted on the campaign it is meant to find is not a detector"
    )


def test_labels_exist_at_all(bundle):
    assert bundle.n_labelled_malicious > 0


def test_campaign_day_is_never_negative(bundle):
    cd = bundle.index["campaign_day"].dropna()
    assert (cd >= 0).all()


# -- masking ----------------------------------------------------------------
def test_padding_is_never_masked():
    tokens = torch.tensor([[3, 4, 5, PAD_ID, PAD_ID], [3, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])
    masked, selected = apply_masking(tokens, probability=0.9)
    assert not selected[tokens == PAD_ID].any()
    assert (masked[tokens == PAD_ID] == PAD_ID).all()


def test_every_non_empty_row_gets_at_least_one_target():
    tokens = torch.tensor([[3, 4, PAD_ID], [5, PAD_ID, PAD_ID]])
    _, selected = apply_masking(tokens, probability=0.0)
    assert selected.any(dim=1).all()


def test_masking_replaces_with_the_mask_token():
    tokens = torch.tensor([[3, 4, 5, 6]])
    masked, selected = apply_masking(tokens, probability=1.0)
    assert (masked[selected] == MASK_ID).all()


# -- scoring ----------------------------------------------------------------
def test_self_distance_only_looks_backwards():
    """A person's future must not influence how their present is scored.

    Constructed so the answer is unambiguous: one person, flat history, then a
    single enormous jump on the last day. If the baseline were computed over
    the whole record, the jump would drag the centre towards itself and score
    lower than it does here.
    """
    n = 12
    emb = np.zeros((n, 4), dtype=np.float32)
    emb[:, 0] = np.arange(n) * 0.01
    emb[-1, 0] = 50.0
    index = pd.DataFrame({
        "user": ["u"] * n,
        "day": pd.date_range("2010-01-01", periods=n, freq="D"),
    })
    d = self_distance(emb, index, history_days=30)
    assert d[0] == 0.0, "the first day has no history and must score zero"
    assert (d[:5] == 0).all(), "days before the minimum history must score zero"
    assert d[-1] == d.max()
    assert d[-1] > 10 * np.median(d[5:-1])


def test_self_distance_is_zero_before_any_history_exists():
    emb = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    index = pd.DataFrame({"user": ["a", "b", "a", "b"],
                          "day": pd.to_datetime(["2010-01-01", "2010-01-01",
                                                 "2010-01-02", "2010-01-02"])})
    d = self_distance(emb, index, history_days=30)
    assert (d == 0).all()


def test_a_single_day_of_history_cannot_produce_an_infinite_score():
    """The bug the minimum-history floor exists to prevent.

    One person, two identical days, then a small deviation. Without the floor
    the within-person spread is zero and the third day scores in the
    thousands, which would push a genuine anomaly off the top of the queue.
    """
    emb = np.array([[0.0, 0.0], [0.0, 0.0], [0.01, 0.0],
                    [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.02, 0.0]],
                   dtype=np.float32)
    index = pd.DataFrame({"user": ["u"] * 7,
                          "day": pd.date_range("2010-01-01", periods=7)})
    d = self_distance(emb, index, history_days=30)
    assert np.isfinite(d).all()
    assert d.max() < 100


def test_robust_z_is_not_moved_by_a_few_extreme_values():
    base = np.concatenate([np.zeros(1000), np.full(5, 1e6)])
    z = robust_z(base)
    assert abs(np.median(z)) < 1e-9
    assert abs(z[:1000]).max() < 1.0


# -- evaluation -------------------------------------------------------------
def _toy_queue():
    return pd.DataFrame({
        "user": ["a", "b", "c", "a", "b", "c"],
        "day": pd.to_datetime(["2010-01-01"] * 3 + ["2010-01-02"] * 3),
        "label": [1, 0, 0, 0, 1, 0],
        "scenario": [1, 0, 0, 0, 1, 0],
        "campaign_day": [0, np.nan, np.nan, np.nan, 1, np.nan],
    })


def test_daily_queue_ranks_within_each_day():
    index = _toy_queue()
    scores = np.array([0.9, 0.5, 0.1, 0.2, 0.8, 0.3])
    q = daily_queue(index, scores, capacity=1)
    first_day = q[q["day"] == pd.Timestamp("2010-01-01")]
    assert first_day.iloc[0]["user"] == "a"
    assert int(q["alerted"].sum()) == 2      # one per day


def test_queue_metrics_by_hand():
    index = _toy_queue()
    scores = np.array([0.9, 0.5, 0.1, 0.2, 0.8, 0.3])
    q = daily_queue(index, scores, capacity=1)
    m = queue_metrics(q, capacity=1)
    # Two alerts raised, both of them the genuine insider day.
    assert m["alerts_raised"] == 2
    assert m["alerts_true"] == 2
    assert m["precision"] == 1.0
    assert m["recall_user_days"] == 1.0
    assert m["wasted_reviews_per_catch"] == 0.0


def test_capacity_of_one_never_alerts_twice_in_a_day():
    index = _toy_queue()
    scores = np.arange(6, dtype=float)
    q = daily_queue(index, scores, capacity=1)
    assert q.groupby("day")["alerted"].sum().max() == 1


def test_campaign_detection_reports_days_into_the_campaign():
    n = 5
    index = pd.DataFrame({
        "user": ["x"] * n,
        "day": pd.date_range("2010-02-01", periods=n, freq="D"),
        "label": [1] * n,
        "scenario": [2] * n,
        "campaign_day": list(range(n)),
    })
    # A decoy colleague on every day, so that ranking actually decides who is
    # alerted. Without them a capacity of one always alerts the only user
    # present, and the test would pass for the wrong reason.
    decoy = index.copy()
    decoy["user"] = "decoy"
    decoy[["label", "scenario"]] = 0
    decoy["campaign_day"] = np.nan
    index = pd.concat([index, decoy], ignore_index=True)

    # Only the third campaign day outscores the decoy.
    scores = np.array([0.1, 0.1, 9.0, 0.1, 0.1] + [1.0] * n)
    camp = campaign_detection(index, scores, capacity=1)
    assert len(camp) == 1
    assert bool(camp.iloc[0]["detected"])
    assert camp.iloc[0]["days_into_campaign"] == 2


def test_ranking_metrics_handle_a_window_with_no_positives():
    m = ranking_metrics(np.zeros(10), np.random.default_rng(0).normal(size=10))
    assert np.isnan(m["auroc"])


# -- model plumbing ---------------------------------------------------------
def test_context_cardinalities_cover_every_observed_level(bundle):
    cards = context_cardinalities(bundle)
    assert len(cards) == 5
    assert all(c >= 1 for c in cards)


def test_ablation_zeroes_the_intended_modality(bundle):
    from mint.pipeline import _ablate
    no_content = _ablate(bundle, "no_content")
    assert not no_content.content.any()
    assert no_content.tokens.any(), "behaviour must survive a content ablation"
