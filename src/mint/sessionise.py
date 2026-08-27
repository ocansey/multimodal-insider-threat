"""Turning five unsorted activity logs into one ordered day per person.

This is the least glamorous module in the package and the one most likely to
decide whether the results mean anything. Four decisions are made here, and
each of them is a place where a plausible-looking alternative would quietly
break the study.

**The day boundary is 04:00, not midnight.** Scenario 1 of the CERT data is a
person who starts staying late and eventually works through the small hours.
Cut the day at midnight and that single sitting becomes two half-empty days,
one ending at 23:59 looking mildly long and one starting at 00:00 looking
mildly early, with the actual event — a continuous fourteen-hour session
straddling the boundary — represented nowhere. Every night-shift worker in the
organisation is mangled the same way. Four in the morning is the quietest hour
in the data and the natural place to cut.

**HTTP is capped and sampled, not truncated.** Web traffic is roughly ninety
percent of all events and almost none of it is informative. Keeping the first
sixty-four rows of a day would systematically discard the afternoon, which is
where scenario 2's job-hunting happens. So each source gets its own budget and
is sampled uniformly across the day when it overflows, with the sample drawn
from a per-user-day seed so the same input always produces the same output.

**"Own machine" is learned, not assumed.** The release has no asset register.
A user's own PC is taken to be the machine they log into most often across the
*training window only* — computing it over the whole period would let the test
period inform a training feature, which is leakage even though it feels like
harmless metadata.

**Days with almost nothing in them are dropped.** An account that produced two
events is not evidence of good behaviour or bad; scoring it spends analyst
attention on noise. The threshold is in the config and the count of dropped
days is reported.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .schema import ACTIVITY_FILES, PAD_ID, TOKEN_TO_ID

log = logging.getLogger(__name__)

#: Per-source event budget within one user-day, summing to the configured
#: maximum sequence length. HTTP dominates raw volume by an order of magnitude
#: and carries the least per-event information, so it is capped hardest.
#:
#: The total is 128 rather than something more generous because attention cost
#: is quadratic in sequence length and the 99th percentile of a real working
#: day sits comfortably below it. Doubling the budget quadrupled training time
#: and moved no metric.
SOURCE_BUDGET = {"logon": 12, "device": 16, "file": 32, "email": 36, "http": 32}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def read_activity(
    raw_dir: Path, name: str, chunksize: int | None = 500_000
) -> Iterator[pd.DataFrame]:
    """Yield chunks of one activity file, or the whole thing if it is small.

    The real ``http.csv`` is 1.7 GB and will not fit comfortably in memory
    alongside a model, so everything downstream is written to consume an
    iterator even when there is only one chunk in it.
    """
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        log.warning("%s not found — skipping this source", path.name)
        return
    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        yield chunk


def parse_dates(series: pd.Series) -> pd.Series:
    """CERT stamps look like ``01/04/2010 06:45:43``; releases differ slightly."""
    out = pd.to_datetime(series, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    if out.isna().mean() > 0.01:
        out = pd.to_datetime(series, errors="coerce")
    if out.isna().mean() > 0.01:
        raise ValueError(
            f"{out.isna().mean():.1%} of timestamps failed to parse — the "
            "release probably uses a date format this loader does not know"
        )
    return out


# --------------------------------------------------------------------------
# event typing
# --------------------------------------------------------------------------
REMOVABLE_PREFIXES = ("R:", "r:", "/media", "/mnt", "E:", "F:")


def token_for(source: str, row: pd.Series) -> str:
    """Map one raw row to one vocabulary token."""
    if source == "logon":
        return "logon" if str(row.get("activity", "")).lower() == "logon" else "logoff"
    if source == "device":
        act = str(row.get("activity", "")).lower()
        return "device_connect" if act == "connect" else "device_disconnect"
    if source == "file":
        fn = str(row.get("filename", ""))
        return ("file_copy_to_removable"
                if fn.startswith(REMOVABLE_PREFIXES) else "file_open")
    if source == "http":
        return "http_visit"
    if source == "email":
        sender = _clean(row.get("from"))
        user = _clean(row.get("user"))
        # A row is a *receipt* when the sender is not this user. CERT writes
        # one row per delivery, so the same message appears once for the
        # sender and once for each internal recipient.
        if user and user not in sender:
            return "email_receive"
        if _as_int(row.get("attachments")) > 0:
            return "email_send_with_attachment"
        domain = sender.split("@")[-1] if "@" in sender else ""
        recipients = _addresses(row)
        external = bool(domain) and any(
            not addr.endswith("@" + domain) for addr in recipients
        )
        return "email_send_external" if external else "email_send_internal"
    raise ValueError(f"unknown source {source}")


def _clean(value) -> str:
    """Empty CSV cells arrive as float NaN and stringify to 'nan'.

    That is not a hypothetical: before this function existed, every message in
    the corpus was classified as external, because ``str(nan)`` is the literal
    text ``nan``, which does not end in the company's domain. The internal
    class was empty and nobody would have noticed from the metrics.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else text


def _as_int(value) -> int:
    try:
        return int(float(_clean(value) or 0))
    except (TypeError, ValueError):
        return 0


def _addresses(row: pd.Series) -> list[str]:
    """Every recipient address on a message, across to/cc/bcc."""
    out: list[str] = []
    for field in ("to", "cc", "bcc"):
        raw = _clean(row.get(field))
        if not raw:
            continue
        for part in raw.replace(",", ";").split(";"):
            addr = part.strip()
            if "@" in addr:
                out.append(addr)
    return out


def typed_events(raw_dir: Path, day_start_hour: int) -> pd.DataFrame:
    """One row per event: user, day, timestamp, token, pc, source, row index.

    Returns a long frame. On the real release this is roughly 30 million rows
    before capping, which is why the caller immediately reduces it.
    """
    frames = []
    for source in ACTIVITY_FILES:
        for chunk in read_activity(raw_dir, source):
            ts = parse_dates(chunk["date"])
            tokens = chunk.apply(lambda r, s=source: token_for(s, r), axis=1)
            frames.append(pd.DataFrame({
                "user": chunk["user"].astype(str),
                "ts": ts,
                "token": tokens,
                "pc": chunk.get("pc", pd.Series([""] * len(chunk))).astype(str),
                "source": source,
                "row_id": chunk.index.to_numpy(),
                "event_id": chunk["id"].astype(str) if "id" in chunk else "",
            }))
    if not frames:
        raise FileNotFoundError(f"no activity files found under {raw_dir}")
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.dropna(subset=["ts"])

    # The 04:00 boundary: anything before it belongs to the previous day.
    shifted = ev["ts"] - pd.Timedelta(hours=day_start_hour)
    ev["day"] = shifted.dt.normalize()
    ev["hour"] = ev["ts"].dt.hour.astype("int8")
    ev["weekday"] = ev["ts"].dt.weekday.astype("int8")
    return ev.sort_values(["user", "day", "ts"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# machine context
# --------------------------------------------------------------------------
def machine_context(
    events: pd.DataFrame, training_days: pd.DatetimeIndex, shared_min_users: int
) -> tuple[pd.Series, set[str]]:
    """Each user's usual machine, and the set of shared machines.

    Both are computed from the training window only. Deriving "own PC" from
    the whole period would let a person who switches desks in month fourteen
    influence a feature used to score month three, which is a small leak but a
    real one and free to avoid.
    """
    train = events[events["day"].isin(training_days)]
    logons = train[train["token"] == "logon"]
    own = (logons.groupby(["user", "pc"]).size().rename("n").reset_index()
           .sort_values(["user", "n"], ascending=[True, False])
           .drop_duplicates("user").set_index("user")["pc"])
    users_per_pc = train.groupby("pc")["user"].nunique()
    shared = set(users_per_pc[users_per_pc >= shared_min_users].index)
    log.info("resolved own-machine for %d users; %d shared machines",
             len(own), len(shared))
    return own, shared


# --------------------------------------------------------------------------
# sessionisation
# --------------------------------------------------------------------------
@dataclass
class Sessions:
    """Fixed-width arrays, one row per user-day."""

    index: pd.DataFrame              # user, day, n_events_raw, n_events_kept
    tokens: np.ndarray               # (N, L) int16
    hours: np.ndarray                # (N, L) int8
    flags: np.ndarray                # (N, L, 4) int8 — after-hours, weekend, own, shared
    doc_refs: list[list[tuple[str, int]]]   # (source, row_id) for text extraction

    def __len__(self) -> int:
        return len(self.index)

    @property
    def max_len(self) -> int:
        return self.tokens.shape[1]


def _apply_source_budget(events: pd.DataFrame) -> pd.DataFrame:
    """Thin each source down to its budget, evenly spaced across the day.

    Vectorised, because this runs over roughly thirty million rows on the real
    release and a per-group Python loop turns a two-minute step into an hour.
    The selection rule keeps element ``k`` of ``n`` when ``floor(k*b/n)``
    differs from ``floor((k-1)*b/n)``, which picks exactly ``b`` items spread
    evenly through the day — deterministic, no seed, and it preserves the
    shape of the day rather than beheading it.
    """
    key = ["user", "day", "source"]
    k = events.groupby(key, sort=False).cumcount().to_numpy()
    n = events.groupby(key, sort=False)["ts"].transform("size").to_numpy()
    budget = events["source"].map(SOURCE_BUDGET).fillna(32).to_numpy()

    over = n > budget
    scaled = np.floor(k * budget / np.maximum(n, 1))
    prev = np.floor((k - 1) * budget / np.maximum(n, 1))
    keep = (~over) | (k == 0) | (scaled != prev)
    return events.loc[keep]


def sessionise(
    events: pd.DataFrame,
    own_pc: pd.Series,
    shared_pcs: set[str],
    *,
    max_events: int = 256,
    min_events: int = 3,
    after_hours_start: int = 19,
    after_hours_end: int = 7,
) -> Sessions:
    """Collapse the event stream into fixed-width per-user-day sequences."""
    ev = events.sort_values(["user", "day", "ts"], kind="stable")

    raw_counts = ev.groupby(["user", "day"], sort=False)["ts"].size()
    ev = _apply_source_budget(ev).sort_values(["user", "day", "ts"], kind="stable")

    # Position of each surviving event within its user-day, then a hard cut.
    pos = ev.groupby(["user", "day"], sort=False).cumcount().to_numpy()
    ev = ev.loc[pos < max_events]
    pos = pos[pos < max_events]

    day_size = ev.groupby(["user", "day"], sort=False)["ts"].transform("size")
    enough = (day_size >= min_events).to_numpy()
    ev, pos = ev.loc[enough], pos[enough]

    index = (ev[["user", "day"]].drop_duplicates()
             .reset_index(drop=True).reset_index().rename(columns={"index": "row"}))
    row_of = ev[["user", "day"]].merge(index, on=["user", "day"], how="left")["row"]
    row_of = row_of.to_numpy()

    n_days = len(index)
    tokens = np.full((n_days, max_events), PAD_ID, dtype=np.int16)
    hours = np.zeros((n_days, max_events), dtype=np.int8)
    flags = np.zeros((n_days, max_events, 4), dtype=np.int8)

    tokens[row_of, pos] = ev["token"].map(TOKEN_TO_ID).fillna(PAD_ID).to_numpy()
    hours[row_of, pos] = ev["hour"].to_numpy()
    hr = ev["hour"].to_numpy()
    flags[row_of, pos, 0] = ((hr >= after_hours_start) | (hr < after_hours_end))
    flags[row_of, pos, 1] = (ev["weekday"].to_numpy() >= 5)
    flags[row_of, pos, 2] = (
        ev["pc"].to_numpy() == ev["user"].map(own_pc).fillna("").to_numpy())
    flags[row_of, pos, 3] = ev["pc"].isin(shared_pcs).to_numpy()

    kept = np.bincount(row_of, minlength=n_days)
    index = index.drop(columns="row")
    index["n_events_kept"] = kept
    index = index.merge(raw_counts.rename("n_events_raw").reset_index(),
                        on=["user", "day"], how="left")

    doc_refs: list[list[tuple[str, int]]] = [[] for _ in range(n_days)]
    for r, source, rid in zip(row_of, ev["source"].to_numpy(),
                              ev["row_id"].to_numpy()):
        doc_refs[r].append((source, int(rid)))

    dropped = len(raw_counts) - n_days
    if n_days == 0:
        raise ValueError("sessionisation produced no user-days")
    log.info("built %d user-days; dropped %d below the minimum-events threshold",
             n_days, dropped)
    return Sessions(index=index, tokens=tokens, hours=hours, flags=flags,
                    doc_refs=doc_refs)
