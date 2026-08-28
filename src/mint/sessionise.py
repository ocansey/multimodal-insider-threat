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
from .sources import Release

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
def _as_release(source: "Release | Path | str") -> Release:
    """Accept a Release, a directory, or an archive path interchangeably."""
    return source if isinstance(source, Release) else Release(Path(source))


def read_activity(
    release: "Release | Path", name: str, chunksize: int | None = 500_000
) -> Iterator[pd.DataFrame]:
    """Yield chunks of one activity table, wherever it physically lives.

    The real ``http.csv`` is over 14 GB and will not fit in memory
    alongside a model, so everything downstream consumes an iterator even when
    there is only one chunk in it. The source may be a plain CSV, a compressed
    CSV, or a member inside the downloaded tarball — see :mod:`mint.sources`.
    """
    release = _as_release(release)
    filename = f"{name}.csv"
    with release.open(filename) as stream:
        if stream is None:
            log.warning("%s not found in %s — skipping this source",
                        filename, release.root)
            return
        for chunk in pd.read_csv(stream, chunksize=chunksize, low_memory=False):
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
    return _addresses_from_values(row.get("to"), row.get("cc"), row.get("bcc"))


def tokens_for_frame(chunk: pd.DataFrame, source: str) -> np.ndarray:
    """Vectorised token assignment for a whole chunk.

    :func:`token_for` is the readable, row-at-a-time definition and it is what
    the unit tests check. It is also unusable at scale: ``DataFrame.apply``
    with ``axis=1`` builds a Series object per row, and at twenty-eight
    million rows of web traffic that is roughly half an hour of pure overhead
    per pass. This produces identical output with array operations, and
    ``test_vectorised_tokens_match_the_row_definition`` asserts they agree.
    """
    n = len(chunk)
    if source == "http":
        return np.full(n, "http_visit", dtype=object)

    if source == "logon":
        act = chunk["activity"].astype(str).str.lower()
        return np.where(act == "logon", "logon", "logoff")

    if source == "device":
        act = chunk["activity"].astype(str).str.lower()
        return np.where(act == "connect", "device_connect", "device_disconnect")

    if source == "file":
        fn = chunk["filename"].astype(str)
        removable = fn.str.startswith(REMOVABLE_PREFIXES)
        return np.where(removable, "file_copy_to_removable", "file_open")

    if source == "email":
        senders = chunk.get("from", pd.Series([""] * n, index=chunk.index)) \
            .map(_clean).to_numpy()
        users = chunk.get("user", pd.Series([""] * n, index=chunk.index)) \
            .map(_clean).to_numpy()
        attach = chunk.get("attachments", pd.Series([0] * n, index=chunk.index)) \
            .map(_as_int).to_numpy()

        received = np.array([bool(u) and (u not in s)
                             for u, s in zip(users, senders)])
        domains = np.array([s.split("@")[-1] if "@" in s else "" for s in senders])
        recipients = [
            _addresses_from_values(to, cc, bcc)
            for to, cc, bcc in zip(
                chunk.get("to", pd.Series([""] * n, index=chunk.index)),
                chunk.get("cc", pd.Series([""] * n, index=chunk.index)),
                chunk.get("bcc", pd.Series([""] * n, index=chunk.index)))
        ]
        external = np.array([
            bool(d) and any(not a.endswith("@" + d) for a in addrs)
            for d, addrs in zip(domains, recipients)
        ])

        out = np.where(external, "email_send_external", "email_send_internal")
        out = np.where(attach > 0, "email_send_with_attachment", out)
        return np.where(received, "email_receive", out)

    raise ValueError(f"unknown source {source}")


def _addresses_from_values(to, cc, bcc) -> list[str]:
    out: list[str] = []
    for raw in (to, cc, bcc):
        text = _clean(raw)
        if not text:
            continue
        for part in text.replace(",", ";").split(";"):
            addr = part.strip()
            if "@" in addr:
                out.append(addr)
    return out


def _reduce_chunk(chunk: pd.DataFrame, source: str, day_start_hour: int
                  ) -> pd.DataFrame:
    """One chunk of raw CSV -> the eight small columns the pipeline needs.

    Every column here is chosen for size as much as for content. The obvious
    implementation keeps the raw frame and adds columns to it, which on the
    real release means roughly three hundred bytes per row across thirty-two
    million rows — about ten gigabytes, on a machine with eight. Categoricals
    and 32-bit indices bring that under a gigabyte.

    The event id is dropped outright. It is the natural thing to carry
    "just in case", it is a 38-character string, and nothing downstream ever
    reads it.
    """
    ts = parse_dates(chunk["date"])
    keep = ts.notna()
    chunk, ts = chunk.loc[keep], ts.loc[keep]

    out = pd.DataFrame({
        "user": chunk["user"].astype(str).astype("category"),
        "ts": ts.to_numpy(),
        "token": pd.Categorical(tokens_for_frame(chunk, source),
                                categories=list(TOKEN_TO_ID)),
        "pc": chunk.get("pc", pd.Series([""] * len(chunk), index=chunk.index))
        .astype(str).astype("category"),
        "source": pd.Categorical([source] * len(chunk),
                                 categories=ACTIVITY_FILES),
        "row_id": chunk.index.to_numpy().astype(np.int32),
    })
    shifted = out["ts"] - pd.Timedelta(hours=day_start_hour)
    out["day"] = shifted.dt.normalize()
    out["hour"] = out["ts"].dt.hour.astype("int8")
    out["weekday"] = out["ts"].dt.weekday.astype("int8")
    return out


def typed_events(release: "Release | Path", day_start_hour: int,
                 cache_dir: "Path | None" = None) -> pd.DataFrame:
    """One row per event: user, day, timestamp, token, pc, source, row index.

    Each source is read, reduced and *budgeted* before the next one is
    touched. Budgeting early is the difference between holding twenty-eight
    million web-visit rows and holding the ten million that survive the
    per-user-day cap, and it means peak memory is set by the largest single
    table rather than by their sum.
    """
    release = _as_release(release)
    frames = []
    for source in ACTIVITY_FILES:
        # Each source is checkpointed on its own, because the run may well be
        # interrupted between two of them and re-decompressing a 4.8 GB
        # archive to recover work already done is an hour nobody has.
        ckpt = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            ckpt = Path(cache_dir) / f"typed_{source}.pkl"
            if ckpt.exists():
                try:
                    cached = pd.read_pickle(ckpt)
                    log.info("  %-6s resumed from checkpoint (%s rows)",
                             source, f"{len(cached):,}")
                    frames.append(cached)
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.warning("  %-6s checkpoint unreadable (%s); re-reading",
                                source, exc)

        parts = []
        rows_read = 0
        for chunk in read_activity(release, source):
            reduced = _reduce_chunk(chunk, source, day_start_hour)
            rows_read += len(reduced)
            parts.append(reduced)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        del parts
        df = df.sort_values(["user", "day", "ts"], kind="stable")
        df = _apply_source_budget(df)
        log.info("  %-6s %10s rows read, %9s kept after the per-day budget "
                 "(%.0f MB)", source, f"{rows_read:,}", f"{len(df):,}",
                 df.memory_usage(deep=True).sum() / 1e6)
        if ckpt is not None:
            tmp = ckpt.with_suffix(".partial")
            pd.to_pickle(df, tmp)
            tmp.replace(ckpt)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"no activity files found under {release.root}")
    ev = pd.concat(frames, ignore_index=True)
    del frames
    log.info("typed %s events across %d sources (%.0f MB in memory)",
             f"{len(ev):,}", len(ACTIVITY_FILES),
             ev.memory_usage(deep=True).sum() / 1e6)
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
