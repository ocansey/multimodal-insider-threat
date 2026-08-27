"""Unusual compared to whom?

This is the module the project exists for. Everything upstream produces a
representation of a person's day; everything downstream measures how well the
resulting alerts serve an analyst. In between sits one question that is almost
always answered by default rather than on purpose: when we say a day is
anomalous, anomalous *relative to what population*?

Three answers are implemented, and the difference between them is the
experiment.

**Global.** Compare the day against the whole organisation. This is what an
unmodified autoencoder or density model does, and it is why these systems have
the reputation they do. The people who look strangest against a company-wide
baseline are systems administrators, field sales staff and anyone whose job is
irregular by nature. They are not threats. They are the night shift, and they
will absorb the entire alert budget every single day until somebody switches
the tool off.

**Self.** Compare the day against the same person's own recent history. This
fixes the night-shift problem completely and introduces a worse one: an
insider who has been slowly exfiltrating for three weeks has established that
behaviour as their own normal. Self-relative scoring habituates to exactly the
campaigns it is supposed to find, and the habituation gets stronger the longer
the campaign runs — the opposite of what anyone wants.

**Peer.** Compare the day against other people who hold the same role. An
administrator's odd hours are ordinary among administrators, so the night shift
stops burning the budget; and a single administrator drifting away from the
other administrators still stands out, because the reference class does not
move with them. The organisation has already told us who is comparable, in the
directory, and we are simply reading it.

The rolling history used for the self-relative baseline is strictly backward
looking: day *t* is scored against days before *t* only. Using a person's
whole record — including the future — is a leak that flatters self-relative
scoring specifically, which would corrupt the very comparison being made here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import PEER_KEY

log = logging.getLogger(__name__)

#: Peer groupings, coarsest last. A group that is too small to estimate a
#: covariance from falls back to the next entry rather than producing a
#: confident number from six observations.
PEER_FALLBACK = [PEER_KEY, "functional_unit", "business_unit"]


@dataclass
class Reference:
    """A fitted comparison class: a centre, a spread, and how it was chosen."""

    centre: np.ndarray
    precision: np.ndarray      # inverse covariance, shrunk
    n: int
    level: str

    def distance(self, x: np.ndarray) -> np.ndarray:
        d = x - self.centre
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))


def _shrunk_precision(x: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    """Inverse covariance with Ledoit-Wolf-style shrinkage towards a diagonal.

    The embedding is 128-dimensional and a role may have forty people in it.
    An unshrunk covariance there is singular, and inverting it produces
    spectacular distances driven entirely by directions the data never
    populated. Shrinking towards the scaled identity keeps the useful
    structure and discards the fantasy.
    """
    d = x.shape[1]
    if len(x) <= d:
        shrinkage = max(shrinkage, 0.5)
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    target = np.trace(cov) / d * np.eye(d)
    blended = (1 - shrinkage) * cov + shrinkage * target
    return np.linalg.pinv(blended + 1e-6 * np.eye(d))


def fit_global_reference(embeddings: np.ndarray, rows: np.ndarray) -> Reference:
    x = embeddings[rows]
    return Reference(x.mean(0), _shrunk_precision(x), len(x), "global")


def fit_peer_references(
    embeddings: np.ndarray,
    context: pd.DataFrame,
    rows: np.ndarray,
    min_group: int,
) -> tuple[dict[str, Reference], Reference]:
    """One reference per peer group, with fallback for the thin ones."""
    fallback = fit_global_reference(embeddings, rows)
    refs: dict[str, Reference] = {}

    for level in PEER_FALLBACK:
        if level not in context.columns:
            continue
        groups = context.iloc[rows][level].astype(str)
        for value, idx in groups.groupby(groups, observed=True).groups.items():
            key = f"{level}={value}"
            if key in refs:
                continue
            member_rows = rows[np.isin(context.index[rows], idx)]
            if len(member_rows) < min_group:
                continue
            refs[key] = Reference(
                embeddings[member_rows].mean(0),
                _shrunk_precision(embeddings[member_rows]),
                len(member_rows), level,
            )
    log.info("fitted %d peer references across levels %s (fallback n=%d)",
             len(refs), PEER_FALLBACK, fallback.n)
    return refs, fallback


def peer_distance(
    embeddings: np.ndarray,
    context: pd.DataFrame,
    refs: dict[str, Reference],
    fallback: Reference,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each user-day from its own peer group's centre.

    Returns the distances and the group size used, so an analyst can tell when
    a score rests on a thin reference.
    """
    out = np.zeros(len(embeddings))
    sizes = np.zeros(len(embeddings), dtype=int)
    assigned = np.zeros(len(embeddings), dtype=bool)

    for level in PEER_FALLBACK:
        if level not in context.columns:
            continue
        keys = level + "=" + context[level].astype(str)
        for key, ref in refs.items():
            if not key.startswith(level + "="):
                continue
            mask = (keys == key).to_numpy() & ~assigned
            if not mask.any():
                continue
            out[mask] = ref.distance(embeddings[mask])
            sizes[mask] = ref.n
            assigned |= mask

    if (~assigned).any():
        out[~assigned] = fallback.distance(embeddings[~assigned])
        sizes[~assigned] = fallback.n
        log.info("%d user-days fell back to the organisation-wide reference",
                 int((~assigned).sum()))
    return out, sizes


#: A person needs at least this many previous active days before a
#: self-relative score means anything.
MIN_SELF_HISTORY = 5


def self_distance(
    embeddings: np.ndarray, index: pd.DataFrame, history_days: int,
    min_history: int = MIN_SELF_HISTORY,
) -> np.ndarray:
    """Distance from the same person's own recent past, strictly backward.

    A rolling comparison against the person's previous ``history_days`` active
    days. Their first few days score zero, which is the honest answer: there
    is nothing yet to be unusual against.

    The ``min_history`` floor is not fussiness. With a single previous day the
    within-person spread is exactly zero, the division blows up, and that one
    person's second day outranks every genuine anomaly in the organisation.
    The first version of this function did precisely that, and it took a
    unit test rather than a metric to notice — the AUROC barely moved, because
    one absurd score among thousands is invisible in an average and fatal in a
    top-ten queue.

    The denominator is also floored at a fraction of the organisation-wide
    spread, so that somebody with a genuinely rigid routine cannot generate an
    unbounded score from a small deviation.
    """
    order = np.lexsort((index["day"].to_numpy(), index["user"].to_numpy()))
    out = np.zeros(len(embeddings))
    users = index["user"].to_numpy()[order]
    emb = embeddings[order]
    floor = float(embeddings.std()) * 0.05 + 1e-9

    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or users[i] != users[start]:
            block = emb[start:i]
            for j in range(min_history, len(block)):
                lo = max(0, j - history_days)
                past = block[lo:j]
                centre = past.mean(0)
                spread = max(float(past.std(0).mean()), floor)
                out[order[start + j]] = np.linalg.norm(block[j] - centre) / spread
            start = i
    return out


# --------------------------------------------------------------------------
# combining the components
# --------------------------------------------------------------------------
def robust_z(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Median/MAD standardisation, fitted on a reference slice if given.

    Median and MAD rather than mean and standard deviation because every one
    of these components is heavy-tailed by construction — they are anomaly
    scores — and a handful of extreme days would otherwise set the scale for
    everything else.
    """
    ref = values if reference is None else reference
    med = np.median(ref)
    mad = np.median(np.abs(ref - med))
    scale = 1.4826 * mad if mad > 0 else (ref.std() + 1e-9)
    return (values - med) / scale


@dataclass
class ScoreCard:
    """Per-user-day component scores and the combined score."""

    components: pd.DataFrame
    weights: dict[str, float]
    normalisation: str

    @property
    def score(self) -> np.ndarray:
        return self.components["score"].to_numpy()


def combine(
    signals: dict[str, np.ndarray],
    calibration_rows: np.ndarray,
    weights: dict[str, float] | None = None,
    normalisation: str = "peer",
) -> ScoreCard:
    """Standardise each component on the calibration window and add them up.

    Standardising on the calibration slice rather than on everything means the
    scale of a score does not shift when the test period arrives — which is
    what would happen in a live deployment, where you cannot standardise
    against data you have not seen yet.
    """
    names = list(signals)
    frame = pd.DataFrame({n: signals[n] for n in names})
    for n in names:
        frame[n + "_z"] = robust_z(frame[n].to_numpy(),
                                   frame[n].to_numpy()[calibration_rows])

    weights = weights or {n: 1.0 for n in names}
    total = sum(abs(w) for w in weights.values()) or 1.0
    frame["score"] = sum(
        (weights.get(n, 0.0) / total) * frame[n + "_z"] for n in names)
    return ScoreCard(frame, weights, normalisation)


def fit_weights(
    signals: dict[str, np.ndarray],
    labels: np.ndarray,
    calibration_rows: np.ndarray,
    capacity_fraction: float,
    candidates: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Choose component weights on the calibration window, by random search.

    A grid over four weights is coarse and a gradient method is overkill for
    four numbers; a couple of hundred random simplex draws scored on the thing
    we actually care about — precision at the operating capacity — finds a
    good combination and is trivial to explain. The test window is never
    touched.

    If the calibration window contains no malicious days at all, this returns
    equal weights and says so, because fitting to zero positives would be
    fitting to noise.
    """
    names = list(signals)
    y = labels[calibration_rows]
    if y.sum() == 0:
        log.warning("no malicious days in the calibration window — using equal "
                    "weights for all %d score components", len(names))
        return {n: 1.0 for n in names}

    rng = np.random.default_rng(seed)
    z = {n: robust_z(signals[n], signals[n][calibration_rows])[calibration_rows]
         for n in names}
    k = max(1, int(round(capacity_fraction * len(y))))

    best, best_score = {n: 1.0 for n in names}, -1.0
    for _ in range(candidates):
        w = rng.dirichlet(np.ones(len(names)))
        combined = sum(wi * z[n] for wi, n in zip(w, names))
        flagged = np.argsort(-combined)[:k]
        precision = y[flagged].mean()
        if precision > best_score:
            best_score, best = precision, {n: float(wi) for n, wi in zip(names, w)}
    log.info("fitted score weights on the calibration window "
             "(precision@%d = %.3f): %s", k, best_score,
             {k2: round(v, 3) for k2, v in best.items()})
    return best
