#!/usr/bin/env python3
"""Reduce the raw CERT release to model-ready artefacts. Run this where the data is.

The release is roughly 1.5 GB of CSV, most of it web-page text. There is no
reason to move that anywhere. This script reads it in place, keeps the twelve
most informative documents per user-day, encodes them once with a sentence
transformer, projects the embeddings down, and writes about 300 MB of arrays
that contain no raw message content at all.

    # 1. get the data
    #    https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
    tar xjf r4.2.tar.bz2
    tar xjf answers.tar.bz2

    # 2. install what the encoder needs (skip if using --text-encoder hashing)
    pip install -r requirements.txt sentence-transformers

    # 3. reduce
    python scripts/prepare_local.py --raw ~/Downloads/r4.2 --out data/artifacts/cert

Expect twenty minutes to an hour, dominated by reading http.csv and by the
encoder. Everything is streamed, so peak memory stays around 4 GB.

Two flags worth knowing:

``--text-encoder hashing`` skips the sentence transformer entirely. No
download, no torch needed for this step, several times faster. The results are
worse but it is a legitimate baseline — see the note in ``mint/text.py`` about
why a hashed bag of n-grams is the right floor to compare a pretrained encoder
against.

``--sample-users N`` prepares only the first N users. Useful for a first pass
to check everything works before committing an hour to the full run.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mint.artifacts import save  # noqa: E402
from mint.config import load_config  # noqa: E402
from mint.prepare import prepare  # noqa: E402

REQUIRED = ["logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"]


def check_raw(raw: Path) -> None:
    missing = [f for f in REQUIRED if not (raw / f).exists()]
    if missing:
        # A very common shape: the tarball extracts into a nested directory.
        nested = [d for d in raw.iterdir() if d.is_dir()
                  and (d / "logon.csv").exists()]
        if nested:
            raise SystemExit(
                f"{raw} has no activity files, but {nested[0]} does.\n"
                f"Point --raw at {nested[0]} instead."
            )
        raise SystemExit(
            f"missing from {raw}: {', '.join(missing)}\n"
            "Extract r4.2.tar.bz2 and point --raw at the directory that "
            "contains logon.csv."
        )
    if not (raw / "LDAP").is_dir():
        raise SystemExit(
            f"{raw}/LDAP is missing. The organisational context modality "
            "cannot be built without it, and it is what the whole peer-relative "
            "method rests on."
        )
    answers = (raw / "answers").is_dir() or (raw / "answers.csv").exists()
    if not answers:
        print(
            "WARNING: no answers/ directory or answers.csv found. The pipeline "
            "will run, but with no labels there is nothing to evaluate against. "
            "Extract answers.tar.bz2 into the same directory.",
            file=sys.stderr,
        )


def maybe_subset(raw: Path, n_users: int, workdir: Path) -> Path:
    """Write a smaller copy covering the first N users, for a trial run."""
    print(f"building a {n_users}-user subset under {workdir} …")
    workdir.mkdir(parents=True, exist_ok=True)
    users = None
    for name in REQUIRED:
        frames = []
        for chunk in pd.read_csv(raw / name, chunksize=500_000, low_memory=False):
            if users is None:
                users = set(sorted(chunk["user"].astype(str).unique())[:n_users])
            frames.append(chunk[chunk["user"].astype(str).isin(users)])
        pd.concat(frames, ignore_index=True).to_csv(workdir / name, index=False)
        print(f"  {name}")
    for extra in ("psychometric.csv", "answers.csv"):
        if (raw / extra).exists():
            shutil.copy(raw / extra, workdir / extra)
    if (raw / "LDAP").is_dir():
        shutil.copytree(raw / "LDAP", workdir / "LDAP", dirs_exist_ok=True)
    if (raw / "answers").is_dir():
        shutil.copytree(raw / "answers", workdir / "answers", dirs_exist_ok=True)
    return workdir


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path,
                    help="directory holding logon.csv, http.csv, LDAP/ …")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the artefacts (default data/artifacts/cert)")
    ap.add_argument("--text-encoder", default="sentence-transformers",
                    choices=["sentence-transformers", "hashing"])
    ap.add_argument("--sample-users", type=int, default=0,
                    help="prepare only the first N users, for a trial run")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config()
    cfg.ensure_dirs()
    raw = args.raw.expanduser().resolve()
    check_raw(raw)

    if args.sample_users:
        raw = maybe_subset(raw, args.sample_users,
                           cfg.path("raw") / f"subset_{args.sample_users}")

    out = (args.out or cfg.path("artifacts") / "cert").expanduser()
    started = time.time()
    bundle = prepare(raw, cfg, text_encoder_kind=args.text_encoder,
                     synthetic=False)
    save(bundle, out)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print("\n" + "=" * 68)
    print(bundle.summary().to_string(index=False))
    print("=" * 68)
    print(f"\nwrote {total / 1e6:.0f} MB to {out} in "
          f"{(time.time() - started) / 60:.1f} minutes\n")
    print("Files to hand to the modelling step:")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size / 1e6:8.1f} MB")

    labelled = bundle.n_labelled_malicious
    if labelled == 0:
        print("\nNo malicious user-days were labelled. Check that answers.tar.bz2 "
              "was extracted alongside the activity files — without labels the "
              "model will still train, but nothing can be evaluated.")
    else:
        print(f"\n{labelled} malicious user-days across "
              f"{bundle.index.loc[bundle.index['label'] == 1, 'user'].nunique()} "
              "users are available for evaluation.")
    with open(out / "manifest.json", "r", encoding="utf-8") as fh:
        print("text encoder:", json.load(fh).get("text_encoder"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
