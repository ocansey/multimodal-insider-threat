"""Raw release in, model-ready bundle out.

This is the only module that touches the CERT files directly, and it is
deliberately the same code path for the real release and the synthetic
fixture — a preparation step that behaves differently on test data is a
preparation step nobody has actually tested.

Four jobs, in order:

1. Type and sessionise the activity logs (delegated to :mod:`mint.sessionise`).
2. Resolve each person's organisational context *as of the day being scored*,
   not as of some fixed snapshot. The release ships one LDAP file per month
   precisely because people move; pinning everyone to January would make every
   promotion look like an anomaly for the rest of the year.
3. Sample and encode the text each person produced or consumed that day.
4. Attach labels and the chronological split.

On labels: they are attached, and then almost entirely ignored. Training is
self-supervised and never sees them. They exist to evaluate with, and to
*exclude* known-malicious days from the training window — a detector fitted on
the attack it is supposed to find is not a detector.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import Bundle, config_fingerprint
from .schema import CONTEXT_CATEGORICAL, CONTEXT_NUMERIC, TEXT_COLUMNS
from .sessionise import (
    machine_context,
    parse_dates,
    read_activity,
    sessionise,
    typed_events,
)
from .sources import Release, describe
from .text import build_encoder, pool_documents

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
"""Why this exists.

Preparing the full release takes two to three hours, almost all of it bzip2
decompression, and it runs on machines that stop when you stop looking at
them — a Codespace idles out after thirty minutes, a laptop sleeps, an SSH
session drops. Losing two hours to a closed lid is not a modelling problem but
it is the difference between a project that finishes and one that does not.

So every expensive stage writes a checkpoint and every rerun skips what is
already done. The pipeline is deterministic, so resuming produces exactly the
same artefacts as an uninterrupted run; there is a test that asserts it.

Checkpoints live under ``data/cache`` and are keyed by the parameters that
would change their contents. Delete the directory to force a clean run.
"""


def _cache_path(cache_dir: Path | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / name


def _load_checkpoint(path: Path | None, label: str):
    if path is None or not path.exists():
        return None
    try:
        obj = pd.read_pickle(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint is not fatal
        log.warning("ignoring unreadable checkpoint %s (%s)", path.name, exc)
        return None
    log.info("resuming from checkpoint: %s (%s)", path.name, label)
    return obj


def _save_checkpoint(obj, path: Path | None, label: str) -> None:
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    pd.to_pickle(obj, tmp)
    tmp.replace(path)          # atomic, so a kill mid-write cannot corrupt it
    log.info("checkpointed %s (%s, %.0f MB)", path.name, label,
             path.stat().st_size / 1e6)


# --------------------------------------------------------------------------
# organisational context
# --------------------------------------------------------------------------
def load_ldap(release: Release) -> pd.DataFrame:
    """All monthly LDAP snapshots, stacked with the month they describe."""
    references = release.list_dir("LDAP")
    if not references:
        raise FileNotFoundError(
            f"no LDAP snapshots found in {describe(release)}. The "
            "organisational context modality cannot be built without them, "
            "and the peer-relative method rests on it."
        )
    frames = []
    for ref in sorted(references):
        stem = Path(ref.split("::")[-1]).stem
        try:
            snapshot = pd.Timestamp(stem + "-01")
        except ValueError:
            continue
        with release.open_path(ref) as stream:
            snap = pd.read_csv(stream)
        snap["snapshot"] = snapshot
        frames.append(snap)
    if not frames:
        raise FileNotFoundError("LDAP files were found but none parsed")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"user_id": "user"})
    log.info("loaded %d LDAP snapshots covering %d people",
             out["snapshot"].nunique(), out["user"].nunique())
    return out


def load_psychometric(release: Release) -> pd.DataFrame:
    with release.open("psychometric.csv") as stream:
        if stream is None:
            log.warning("no psychometric.csv — the Big Five columns will be zero")
            return pd.DataFrame(columns=["user", "O", "C", "E", "A", "N"])
        df = pd.read_csv(stream).rename(columns={"user_id": "user"})
    return df[["user", "O", "C", "E", "A", "N"]]


def resolve_context(
    index: pd.DataFrame, ldap: pd.DataFrame, psych: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One context row per user-day, using the snapshot in force that month.

    ``merge_asof`` does the temporal join: for each user-day, take the most
    recent LDAP snapshot at or before it. Sorted by date on both sides, which
    ``merge_asof`` requires and which is easy to get wrong silently.
    """
    left = index[["user", "day"]].copy().sort_values("day")
    right = ldap.sort_values("snapshot")

    merged = pd.merge_asof(
        left, right, left_on="day", right_on="snapshot", by="user",
        direction="backward",
    )
    # Anyone with no snapshot at or before their first active day takes the
    # earliest one available rather than being dropped.
    missing = merged["role"].isna()
    if missing.any():
        earliest = right.drop_duplicates("user", keep="first").set_index("user")
        for col in CONTEXT_CATEGORICAL + ["supervisor"]:
            if col in earliest:
                merged.loc[missing, col] = (
                    merged.loc[missing, "user"].map(earliest[col]))
        log.info("%d user-days had no prior LDAP snapshot; used the earliest",
                 int(missing.sum()))

    merged = merged.merge(psych, on="user", how="left")
    merged["is_supervisor"] = (
        merged["user"].isin(ldap["supervisor"].dropna().unique()).astype(int))
    team_size = ldap.drop_duplicates(["user", "team"]).groupby("team")["user"].nunique()
    merged["team_size"] = merged["team"].map(team_size).fillna(1).astype(float)

    # Categorical codes. The vocabulary is fixed here and stored, so the same
    # role maps to the same integer in training and scoring.
    vocab: dict[str, list[str]] = {}
    for col in CONTEXT_CATEGORICAL:
        values = merged[col].astype("string").fillna("(unknown)")
        levels = sorted(values.unique())
        vocab[col] = levels
        merged[col + "_code"] = values.map({v: i for i, v in enumerate(levels)})

    for col in CONTEXT_NUMERIC:
        if col not in merged:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # Standardise the numeric block once, globally. These are static per
    # person, so there is no temporal leakage in using all of them.
    num = merged[CONTEXT_NUMERIC].to_numpy(dtype=np.float32)
    mu, sd = num.mean(0), num.std(0)
    merged[CONTEXT_NUMERIC] = (num - mu) / np.where(sd > 0, sd, 1.0)

    keep = ["user", "day", *[c + "_code" for c in CONTEXT_CATEGORICAL],
            *CONTEXT_NUMERIC, *CONTEXT_CATEGORICAL]
    out = merged[keep].reset_index(drop=True)
    return out, vocab


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def collect_documents(
    release: Release, doc_refs: list[list[tuple[str, int]]], max_docs: int,
    sources: list[str],
) -> list[list[str]]:
    """Pull the text for each user-day's sampled events.

    The document references are ``(source, row_id)`` pairs recorded during
    sessionisation, so this reads each activity file once and pays no cost for
    the ninety-odd percent of rows that were never selected.
    """
    wanted: dict[str, set[int]] = {s: set() for s in sources}
    for refs in doc_refs:
        for source, row_id in refs:
            if source in wanted:
                wanted[source].add(row_id)

    lookup: dict[tuple[str, int], str] = {}
    for source in sources:
        column = TEXT_COLUMNS.get(source)
        if column is None or not wanted[source]:
            continue
        offset = 0
        for chunk in read_activity(release, source):
            ids = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            hits = np.isin(ids, list(wanted[source]))
            if not hits.any():
                continue
            sub = chunk.loc[hits]
            texts = sub.get(column, pd.Series([""] * len(sub))).astype(str)
            for row_id, text in zip(ids[hits], texts):
                lookup[(source, int(row_id))] = text

    out = []
    for refs in doc_refs:
        docs = [lookup.get((s, r), "") for s, r in refs if s in wanted]
        docs = [d for d in docs if d and d != "nan"]
        out.append(docs[:max_docs])
    n_docs = sum(len(d) for d in out)
    log.info("collected %d documents across %d user-days (%.1f per day)",
             n_docs, len(out), n_docs / max(1, len(out)))
    return out


def reduce_content(
    content: np.ndarray, train_mask: np.ndarray, out_dim: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Project document embeddings down, fitting only on the training window.

    Two constraints meet here. The array has to be small enough to move
    between machines and hold in memory, and the projection must not be
    informed by anything outside the training period. A PCA fitted on the
    training rows satisfies both; a PCA fitted on everything would satisfy
    only the first, and would be leakage of the quiet kind that never shows up
    as an obviously wrong number.
    """
    n, d_docs, d_emb = content.shape
    if out_dim >= d_emb:
        return content, {"reduction": "none", "dim": d_emb}

    flat = content.reshape(-1, d_emb)
    row_of = np.repeat(np.arange(n), d_docs)
    real = flat.any(axis=1)
    fit_rows = real & train_mask[row_of]
    if fit_rows.sum() < out_dim * 4:
        log.warning("too few training documents (%d) to fit a %d-component PCA; "
                    "falling back to a fixed random projection",
                    int(fit_rows.sum()), out_dim)
        rng = np.random.default_rng(seed)
        proj = rng.normal(0, 1 / np.sqrt(out_dim), (d_emb, out_dim)).astype(np.float32)
        mean = np.zeros(d_emb, dtype=np.float32)
        explained = float("nan")
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=out_dim, random_state=seed)
        pca.fit(flat[fit_rows])
        proj = pca.components_.T.astype(np.float32)
        mean = pca.mean_.astype(np.float32)
        explained = float(pca.explained_variance_ratio_.sum())
        log.info("PCA %d -> %d retains %.1f%% of document variance "
                 "(fitted on %d training documents)",
                 d_emb, out_dim, 100 * explained, int(fit_rows.sum()))

    reduced = np.zeros((n * d_docs, out_dim), dtype=np.float32)
    reduced[real] = (flat[real] - mean) @ proj
    return (reduced.reshape(n, d_docs, out_dim),
            {"reduction": "pca", "dim": out_dim,
             "explained_variance": explained,
             "fitted_on_documents": int(fit_rows.sum())})


def encode_documents(docs: list[list[str]], encoder, max_docs: int, dim: int
                     ) -> np.ndarray:
    """Encode every document once, then scatter back into per-day blocks."""
    flat, owners = [], []
    for i, day_docs in enumerate(docs):
        for text in day_docs[:max_docs]:
            flat.append(text)
            owners.append(i)
    out = np.zeros((len(docs), max_docs, dim), dtype=np.float32)
    if not flat:
        log.warning("no documents to encode — the content modality will be empty")
        return out

    embeddings = encoder.encode(flat)
    if embeddings.shape[1] != dim:
        raise ValueError(
            f"encoder returned dimension {embeddings.shape[1]}, config says {dim}"
        )
    cursor = np.zeros(len(docs), dtype=int)
    for emb, owner in zip(embeddings, owners):
        slot = cursor[owner]
        if slot < max_docs:
            out[owner, slot] = emb
            cursor[owner] = slot + 1
    return out


# --------------------------------------------------------------------------
# labels and splits
# --------------------------------------------------------------------------
def load_answers(release: Release, day_start_hour: int) -> pd.DataFrame:
    """Malicious events, reduced to the user-days they fall on.

    The real release ships ``answers/`` as a directory of per-scenario CSVs
    with an inconsistent header; the fixture writes a single ``answers.csv``.
    Both are handled, because a loader that only reads the shape you tested
    against is a loader that fails on the real thing.
    """
    frames = []
    with release.open("answers.csv") as stream:
        if stream is not None:
            frames.append(pd.read_csv(stream))

    # The answers tarball unpacks beside the activity files as often as inside
    # them, and may not have been unpacked at all. All three are handled.
    references = release.list_dir("answers")
    if references:
        log.info("reading %d answer files", len(references))
    for ref in sorted(references):
        try:
            with release.open_path(ref) as stream:
                df = pd.read_csv(stream, header=None, on_bad_lines="skip")
        except Exception:  # noqa: BLE001 - malformed answer files are common
            continue
        if df.shape[1] < 4:
            continue
        df = df.rename(columns={0: "kind", 1: "id", 2: "date", 3: "user"})
        df["scenario"] = _scenario_from_path(Path(ref.split("::")[-1]))
        frames.append(df[["id", "date", "user", "scenario"]])

    if not frames:
        log.warning("no answer files found — evaluation will be impossible")
        return pd.DataFrame(columns=["user", "day", "scenario"])

    ans = pd.concat(frames, ignore_index=True)
    ans["ts"] = parse_dates(ans["date"])
    ans = ans.dropna(subset=["ts"])
    ans["day"] = (ans["ts"] - pd.Timedelta(hours=day_start_hour)).dt.normalize()
    ans["user"] = ans["user"].astype(str)
    out = (ans.groupby(["user", "day"])["scenario"]
           .agg(lambda s: int(pd.Series(s).mode().iloc[0])).reset_index())
    found = sorted(out["scenario"].unique())
    log.info("%d malicious user-days across %d users, scenarios %s",
             len(out), out["user"].nunique(), found)
    if found == [0]:
        log.warning("every malicious day was labelled scenario 0 — the answer "
                    "file naming was not recognised, so the per-scenario "
                    "breakdown will be empty. Detection metrics are unaffected.")
    return out


#: Answer files are named for the release *and* the scenario, e.g.
#: ``r4.2-1.csv`` for scenario 1 of release 4.2. Splitting that on every
#: separator and taking the first digit in range returns 2 — the release's
#: minor version — and silently relabels every scenario in the study. The
#: release prefix has to be matched and consumed before the scenario is read.
_SCENARIO_RE = re.compile(r"r\d+(?:\.\d+)?-(\d)\b")


def _scenario_from_path(path: Path) -> int:
    """Scenario number from an answer file's name or its parent directory."""
    for part in [path.name, *[p for p in path.parts[::-1]]]:
        m = _SCENARIO_RE.search(part)
        if m:
            return int(m.group(1))
    # Some layouts use a directory per scenario instead: "3/" or "scenario_3/".
    for part in path.parts[::-1]:
        m = re.fullmatch(r"(?:scenario[_-]?)?(\d)", part, flags=re.IGNORECASE)
        if m and 1 <= int(m.group(1)) <= 5:
            return int(m.group(1))
    return 0


def attach_labels_and_splits(
    index: pd.DataFrame, answers: pd.DataFrame, split_cfg: dict
) -> pd.DataFrame:
    out = index.copy()
    out = out.merge(answers, on=["user", "day"], how="left")
    out["label"] = out["scenario"].notna().astype(int)
    out["scenario"] = out["scenario"].fillna(0).astype(int)

    # Day zero of a person's campaign, so time-to-detection has an origin.
    first = (out[out["label"] == 1].groupby("user")["day"].min()
             .rename("campaign_start"))
    out = out.merge(first, on="user", how="left")
    out["campaign_day"] = (out["day"] - out["campaign_start"]).dt.days
    out.loc[out["campaign_day"] < 0, "campaign_day"] = np.nan

    days = np.sort(out["day"].unique())
    n = len(days)
    n_train = int(n * float(split_cfg["train_fraction"]))
    n_calib = int(n * float(split_cfg["calibrate_fraction"]))
    boundaries = {"train": days[:n_train],
                  "calibrate": days[n_train:n_train + n_calib],
                  "test": days[n_train + n_calib:]}
    out["split"] = "test"
    for name, block in boundaries.items():
        out.loc[out["day"].isin(block), "split"] = name

    if split_cfg.get("exclude_known_malicious_from_training", True):
        poisoned = (out["split"] == "train") & (out["label"] == 1)
        out.loc[poisoned, "split"] = "excluded"
        if poisoned.any():
            log.info("removed %d known-malicious days from the training window",
                     int(poisoned.sum()))

    log.info("split: %s", out["split"].value_counts().to_dict())
    return out


# --------------------------------------------------------------------------
# the whole preparation
# --------------------------------------------------------------------------
def prepare(
    raw_dir: Path | Release,
    cfg,
    text_encoder_kind: str = "hashing",
    synthetic: bool = False,
    answers_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Bundle:
    release = raw_dir if isinstance(raw_dir, Release) else Release(
        Path(raw_dir), extra=[answers_dir] if answers_dir else None)
    s_cfg, t_cfg = cfg.sessionisation, cfg.text

    sessions_ckpt = _cache_path(cache_dir, "sessions.pkl")
    sessions = _load_checkpoint(sessions_ckpt, "event sequences per user-day")

    if sessions is None:
        events = typed_events(release, int(s_cfg["day_start_hour"]),
                              cache_dir=cache_dir)
        days = pd.DatetimeIndex(np.sort(events["day"].unique()))
        n_train = int(len(days) * float(cfg.split["train_fraction"]))
        own_pc, shared = machine_context(events, days[:n_train],
                                         int(s_cfg["shared_pc_min_users"]))

        sessions = sessionise(
            events, own_pc, shared,
            max_events=int(s_cfg["max_events_per_day"]),
            min_events=int(s_cfg["min_events_per_day"]),
            after_hours_start=int(s_cfg["after_hours_start"]),
            after_hours_end=int(s_cfg["after_hours_end"]),
        )
        del events
        _save_checkpoint(sessions, sessions_ckpt, "event sequences")

    answers = load_answers(release, int(s_cfg["day_start_hour"]))
    index = attach_labels_and_splits(sessions.index, answers, cfg.split)

    ldap = load_ldap(release)
    psych = load_psychometric(release)
    context, vocab = resolve_context(index, ldap, psych)

    docs_ckpt = _cache_path(cache_dir, "documents.pkl")
    docs = _load_checkpoint(docs_ckpt, "sampled document text")
    if docs is None:
        docs = collect_documents(release, sessions.doc_refs,
                                 int(t_cfg["max_docs_per_day"]),
                                 list(t_cfg["sources"]))
        _save_checkpoint(docs, docs_ckpt, "document text")
    encoder = build_encoder(text_encoder_kind, t_cfg["encoder"],
                            int(t_cfg["dim"]), seed=cfg.seed)
    content = encode_documents(docs, encoder, int(t_cfg["max_docs_per_day"]),
                               int(t_cfg["dim"]))
    train_mask = (index["split"] == "train").to_numpy()
    content, reduction = reduce_content(
        content, train_mask, int(t_cfg.get("reduce_dim", t_cfg["dim"])), cfg.seed)

    manifest = {
        "synthetic": bool(synthetic),
        "release": cfg.data["release"] if not synthetic else "SYNTHETIC-FIXTURE",
        "text_encoder": encoder.name,
        "config_fingerprint": config_fingerprint(cfg.as_dict()),
        "context_vocabulary": {k: len(v) for k, v in vocab.items()},
        "n_documents": int(sum(len(d) for d in docs)),
        "content_reduction": reduction,
        "source": describe(release),
    }
    if synthetic:
        manifest["warning"] = (
            "Built from mint.simulate output. Not a result under any circumstances."
        )
        fixture_manifest = Path(release.directory or ".") / "manifest.json"
        if fixture_manifest.exists():
            with open(fixture_manifest, "r", encoding="utf-8") as fh:
                manifest["fixture"] = json.load(fh)

    return Bundle(
        tokens=sessions.tokens, hours=sessions.hours, flags=sessions.flags,
        content=content, context=context, index=index, manifest=manifest,
    )
