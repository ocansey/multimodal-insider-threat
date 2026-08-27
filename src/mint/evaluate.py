"""Measuring the thing a security operations centre would actually feel.

A ranked list of half a million user-days is not a product. What a SOC has is
a fixed number of analysts, each able to work a handful of cases a day, and
what they need is a morning queue. So every metric here is defined against a
daily budget rather than a threshold.

**Precision at daily capacity.** Each day, take the top *k* users by score
among those who were active. What fraction of those *k* were genuinely
malicious? This is the number that decides whether the queue gets worked or
ignored after a fortnight.

**Campaign recall and time to detection.** Per-user-day recall is the wrong
headline. An insider campaign runs for weeks; catching it on any of those days
is a catch, and catching it on day two rather than day nineteen is the entire
value of the system. So campaigns are counted once, and the measure attached
to them is how many days into the campaign the first alert fired.

**Alert concentration.** How many *distinct people* the budget lands on over
the test period. A model that flags the same six administrators every single
day has a defensible AUROC and is completely useless, and no conventional
metric will tell you that. This one will.

AUROC and average precision are reported too, because comparison with the
literature would be impossible without them, and because the gap between a
respectable AUROC and a poor precision-at-capacity is itself the argument for
why the other metrics exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------
# conventional
# --------------------------------------------------------------------------
def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    out = {
        "n_user_days": int(len(labels)),
        "n_malicious": int(labels.sum()),
        "prevalence": float(labels.mean()),
    }
    if labels.sum() == 0 or labels.sum() == len(labels):
        out.update({"auroc": float("nan"), "average_precision": float("nan")})
        return out
    out["auroc"] = float(roc_auc_score(labels, scores))
    out["average_precision"] = float(average_precision_score(labels, scores))
    out["lift_at_1pct"] = _lift(labels, scores, 0.01)
    return out


def _lift(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    k = max(1, int(round(fraction * len(labels))))
    top = np.argsort(-scores, kind="stable")[:k]
    base = labels.mean()
    return float(labels[top].mean() / base) if base > 0 else float("nan")


# --------------------------------------------------------------------------
# the daily queue
# --------------------------------------------------------------------------
def daily_queue(
    index: pd.DataFrame, scores: np.ndarray, capacity: int
) -> pd.DataFrame:
    """Build the alert queue a SOC would actually receive, one day at a time.

    Ties are broken by user id rather than by array order, so the queue does
    not depend on how the data happened to be sorted — a detail that quietly
    changes results when the same experiment is re-run after a reshuffle.
    """
    required = ["user", "day", "label", "scenario", "campaign_day"]
    missing = [c for c in required if c not in index.columns]
    if missing:
        raise KeyError(f"the index is missing {missing}")
    # Carry any extra columns the caller attached — the burden analysis joins
    # role onto the index before calling this, and silently dropping it here
    # cost an hour of debugging the first time.
    extras = [c for c in index.columns if c not in required and c != "score"]
    frame = index[required + extras].copy()
    frame["score"] = scores
    frame = frame.sort_values(["day", "score", "user"],
                              ascending=[True, False, True], kind="stable")
    frame["rank_in_day"] = frame.groupby("day").cumcount() + 1
    frame["alerted"] = frame["rank_in_day"] <= capacity
    return frame


def queue_metrics(queue: pd.DataFrame, capacity: int) -> dict[str, float]:
    alerts = queue[queue["alerted"]]
    n_alerts = len(alerts)
    n_malicious = int(queue["label"].sum())
    caught = int(alerts["label"].sum())
    days = queue["day"].nunique()
    return {
        "capacity_per_day": capacity,
        "days": int(days),
        "alerts_raised": n_alerts,
        "alerts_true": caught,
        "precision": caught / n_alerts if n_alerts else float("nan"),
        "recall_user_days": caught / n_malicious if n_malicious else float("nan"),
        "distinct_users_alerted": int(alerts["user"].nunique()),
        "alert_concentration": (
            float(alerts["user"].value_counts().head(10).sum() / n_alerts)
            if n_alerts else float("nan")),
        "wasted_reviews_per_catch": (
            (n_alerts - caught) / caught if caught else float("nan")),
    }


def capacity_sweep(
    index: pd.DataFrame, scores: np.ndarray, capacities: list[int]
) -> pd.DataFrame:
    rows = []
    for capacity in capacities:
        q = daily_queue(index, scores, capacity)
        rows.append(queue_metrics(q, capacity))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# campaigns
# --------------------------------------------------------------------------
def campaign_detection(
    index: pd.DataFrame, scores: np.ndarray, capacity: int
) -> pd.DataFrame:
    """One row per insider campaign present in the evaluated window.

    ``days_into_campaign`` is the headline: zero means the first malicious day
    was caught, five means five days of the campaign ran unnoticed, and a
    missing value means it was never caught at all.
    """
    queue = daily_queue(index, scores, capacity)
    malicious = queue[queue["label"] == 1]
    if malicious.empty:
        return pd.DataFrame(columns=["user", "scenario", "campaign_days_observed",
                                     "detected", "days_into_campaign"])

    rows = []
    for (user, scenario), g in malicious.groupby(["user", "scenario"]):
        g = g.sort_values("day")
        hits = g[g["alerted"]]
        detected = not hits.empty
        first = hits.iloc[0] if detected else None
        rows.append({
            "user": user,
            "scenario": int(scenario),
            "campaign_days_observed": int(len(g)),
            "detected": bool(detected),
            "days_into_campaign": (
                int((first["day"] - g.iloc[0]["day"]).days) if detected else np.nan),
            "malicious_days_caught": int(g["alerted"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["scenario", "user"])


def campaign_summary(campaigns: pd.DataFrame) -> dict[str, float]:
    if campaigns.empty:
        return {"campaigns": 0}
    detected = campaigns[campaigns["detected"]]
    return {
        "campaigns": int(len(campaigns)),
        "campaigns_detected": int(len(detected)),
        "campaign_recall": float(len(detected) / len(campaigns)),
        "median_days_into_campaign": (
            float(detected["days_into_campaign"].median()) if len(detected)
            else float("nan")),
        "caught_on_first_day": (
            float((detected["days_into_campaign"] == 0).mean()) if len(detected)
            else float("nan")),
        "caught_within_3_days": (
            float((detected["days_into_campaign"] <= 3).mean()) if len(detected)
            else float("nan")),
    }


def per_scenario(campaigns: pd.DataFrame) -> pd.DataFrame:
    if campaigns.empty:
        return campaigns
    return (campaigns.groupby("scenario")
            .agg(campaigns=("user", "size"),
                 detected=("detected", "sum"),
                 median_days_into_campaign=("days_into_campaign", "median"),
                 median_days_caught=("malicious_days_caught", "median"))
            .assign(recall=lambda d: d["detected"] / d["campaigns"])
            .reset_index())


# --------------------------------------------------------------------------
# who absorbs the budget
# --------------------------------------------------------------------------
def burden_by_group(
    index: pd.DataFrame, context: pd.DataFrame, scores: np.ndarray,
    capacity: int, attribute: str = "role",
) -> pd.DataFrame:
    """Which roles the alert budget lands on, against their share of the staff.

    The failure this exposes is specific and common: a globally-normalised
    model spends most of its budget on whichever role has the most irregular
    legitimate working pattern. The ratio column makes it visible in one
    number — 1.0 means a role receives alerts in proportion to its size,
    5.0 means it is being audited five times more heavily than everyone else
    for doing its job.
    """
    if attribute not in context.columns:
        return pd.DataFrame()
    queue = daily_queue(index.assign(**{attribute: context[attribute].to_numpy()}),
                        scores, capacity)
    alerts = queue[queue["alerted"]]
    if alerts.empty:
        return pd.DataFrame()

    share_of_days = (queue.groupby(attribute).size() / len(queue)).rename("share_of_days")
    share_of_alerts = (alerts.groupby(attribute).size() / len(alerts)
                       ).rename("share_of_alerts")
    true_rate = alerts.groupby(attribute)["label"].mean().rename("precision")
    out = pd.concat([share_of_days, share_of_alerts, true_rate], axis=1).fillna(0.0)
    out["over_alerting_ratio"] = out["share_of_alerts"] / out["share_of_days"].replace(0, np.nan)
    return out.reset_index().sort_values("over_alerting_ratio", ascending=False)


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
def bootstrap_campaign_recall(
    index: pd.DataFrame, scores: np.ndarray, capacity: int,
    n_boot: int = 300, seed: int = 0,
) -> tuple[float, float]:
    """Interval on campaign recall, resampling campaigns rather than days.

    With a handful of campaigns this interval is wide, and it should be: the
    honest summary of "we caught five of six" is not a point estimate. Papers
    on this dataset routinely quote three decimal places on a quantity
    estimated from fewer than ten independent events.
    """
    campaigns = campaign_detection(index, scores, capacity)
    if campaigns.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    detected = campaigns["detected"].to_numpy().astype(float)
    draws = [detected[rng.integers(0, len(detected), len(detected))].mean()
             for _ in range(n_boot)]
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
