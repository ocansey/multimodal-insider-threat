"""The handoff format between the machine holding the data and the one modelling it.

The real CERT release is over 17 GB of raw text and there is no reason to move it
anywhere. The preparation step reduces it, on whatever machine downloaded it,
to five compact arrays that contain no raw message content — token ids,
embeddings, context, an index and a manifest — and those are what the model
consumes. On the real release the reduction is roughly forty to one.

The manifest is the part that matters most. It records the release, the text
encoder used, the row counts, the configuration hash and, critically, whether
the data was synthetic. :func:`load` refuses to hand back a synthetic bundle
unless the caller says ``allow_synthetic=True``, and every result file written
by the pipeline inherits the flag. It is very easy, in a project that ships a
data simulator so the tests can run, to publish a number that came from the
simulator. This is the machinery that stops that happening by accident.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FORMAT_VERSION = 1


class SyntheticDataError(RuntimeError):
    """Raised when fixture data reaches something that reports results."""


@dataclass
class Bundle:
    """Everything the model needs, and nothing it does not."""

    tokens: np.ndarray        # (N, L) int16   event vocabulary ids
    hours: np.ndarray         # (N, L) int8
    flags: np.ndarray         # (N, L, 4) int8
    content: np.ndarray       # (N, D_docs, E) float16 text embeddings
    context: pd.DataFrame     # one row per user-day: categorical + numeric context
    index: pd.DataFrame       # user, day, split, label, scenario, campaign_day
    manifest: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.manifest.get("synthetic", False))

    @property
    def n_labelled_malicious(self) -> int:
        return int(self.index["label"].sum()) if "label" in self.index else 0

    def summary(self) -> pd.DataFrame:
        rows = [
            {"item": "user-days", "value": len(self.index)},
            {"item": "users", "value": self.index["user"].nunique()},
            {"item": "days", "value": self.index["day"].nunique()},
            {"item": "sequence length", "value": self.tokens.shape[1]},
            {"item": "documents per day", "value": self.content.shape[1]},
            {"item": "embedding dim", "value": self.content.shape[2]},
            {"item": "malicious user-days", "value": self.n_labelled_malicious},
            {"item": "text encoder", "value": self.manifest.get("text_encoder", "?")},
            {"item": "synthetic", "value": self.is_synthetic},
        ]
        return pd.DataFrame(rows)

    def slice(self, mask: np.ndarray) -> "Bundle":
        """Subset every array consistently — used by the split logic."""
        return Bundle(
            tokens=self.tokens[mask],
            hours=self.hours[mask],
            flags=self.flags[mask],
            content=self.content[mask],
            context=self.context.loc[mask].reset_index(drop=True),
            index=self.index.loc[mask].reset_index(drop=True),
            manifest=self.manifest,
        )


def config_fingerprint(cfg_dict: dict[str, Any]) -> str:
    """Short hash of the configuration, so artefacts and results can be paired."""
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save(bundle: Bundle, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "events.npz",
        tokens=bundle.tokens, hours=bundle.hours, flags=bundle.flags,
    )
    _save_content(bundle.content.astype(np.float16), out_dir)
    # Gzipped CSV rather than parquet, for two reasons. It removes a
    # compiled dependency that a reviewer may not have, and it means the two
    # tables a human might actually want to inspect can be opened with `zcat`
    # instead of a Python session. They are small; the arrays are not, and
    # those stay binary.
    bundle.context.to_csv(out_dir / "context.csv.gz", index=False)
    bundle.index.to_csv(out_dir / "index.csv.gz", index=False)

    manifest = dict(bundle.manifest)
    manifest.update({
        "format_version": FORMAT_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_user_days": len(bundle),
        "n_users": int(bundle.index["user"].nunique()),
        "shapes": {
            "tokens": list(bundle.tokens.shape),
            "content": list(bundle.content.shape),
        },
    })
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    log.info("wrote artefacts to %s (%.1f MB)%s", out_dir, total / 1e6,
             "  [SYNTHETIC]" if manifest.get("synthetic") else "")
    return out_dir


#: Largest single file the artefact writer will emit. The transfer path
#: between a laptop and a compute environment caps individual files at
#: 400 MB, and the full release's content array lands just over that, so it is
#: written in shards rather than leaving the user to discover the limit.
MAX_SHARD_BYTES = 350_000_000


def _save_content(content: np.ndarray, out_dir: Path) -> None:
    nbytes = content.nbytes
    if nbytes <= MAX_SHARD_BYTES:
        np.save(out_dir / "content.npy", content)
        return
    n_shards = int(np.ceil(nbytes / MAX_SHARD_BYTES))
    for i, part in enumerate(np.array_split(content, n_shards)):
        np.save(out_dir / f"content.part{i:02d}.npy", part)
    log.info("content array is %.0f MB — written as %d shards",
             nbytes / 1e6, n_shards)


def _load_content(in_dir: Path) -> np.ndarray:
    single = in_dir / "content.npy"
    if single.exists():
        return np.load(single)
    shards = sorted(in_dir.glob("content.part*.npy"))
    if not shards:
        raise FileNotFoundError(f"no content array under {in_dir}")
    return np.concatenate([np.load(s) for s in shards], axis=0)


def load(in_dir: Path, allow_synthetic: bool = False) -> Bundle:
    in_dir = Path(in_dir)
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}. Run scripts/prepare_local.py "
            "(real data) or `make fixture` (synthetic smoke test) first."
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("synthetic") and not allow_synthetic:
        raise SyntheticDataError(
            f"{in_dir} holds SYNTHETIC fixture data generated by "
            "mint.simulate, not the CERT release. Nothing computed from it is "
            "a result. Pass allow_synthetic=True if you are running the smoke "
            "test on purpose."
        )
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"artefact format v{manifest.get('format_version')} does not match "
            f"the v{FORMAT_VERSION} this package reads; re-run the preparation step"
        )

    with np.load(in_dir / "events.npz") as z:
        tokens, hours, flags = z["tokens"], z["hours"], z["flags"]
    content = _load_content(in_dir)
    context = pd.read_csv(in_dir / "context.csv.gz")
    index = pd.read_csv(in_dir / "index.csv.gz", parse_dates=["day"])

    n = len(index)
    for name, arr in (("tokens", tokens), ("content", content)):
        if len(arr) != n:
            raise ValueError(
                f"{name} has {len(arr)} rows but the index has {n}; the "
                "artefacts were written by different runs"
            )
    return Bundle(tokens=tokens, hours=hours, flags=flags, content=content,
                  context=context, index=index, manifest=manifest)


def stamp_results(frame: pd.DataFrame, bundle: Bundle, cfg_desc: str) -> pd.DataFrame:
    """Attach provenance to a results table before it is written to disk.

    Every CSV under ``reports/`` carries these columns. A reader who finds one
    on its own, with no surrounding context, can still tell whether it came
    from real data.
    """
    out = frame.copy()
    out["data_source"] = "SYNTHETIC-FIXTURE" if bundle.is_synthetic else \
        bundle.manifest.get("release", "cert")
    out["text_encoder"] = bundle.manifest.get("text_encoder", "?")
    out["config"] = cfg_desc
    return out
