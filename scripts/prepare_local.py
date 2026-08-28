#!/usr/bin/env python3
"""Reduce the raw CERT release to model-ready artefacts. Run this where the data is.

The release is over 17 GB of CSV, most of it web-page text. There is no
reason to move that anywhere. This script reads it in place, keeps the twelve
most informative documents per user-day, encodes them once with a sentence
transformer, projects the embeddings down, and writes about 300 MB of arrays
that contain no raw message content at all.

    # extraction is optional — point --raw at the tarball or its folder
    python scripts/prepare_local.py --raw ~/Downloads --out data/artifacts/cert

    # or, if you did extract
    python scripts/prepare_local.py --raw ~/Downloads/r4.2 --out data/artifacts/cert

A warning about size, learned the hard way. The published file listing implies
a few gigabytes. It is wrong for r4.2: http.csv alone exceeds 14 GB extracted
and the whole release is over 17 GB, which does not fit alongside the 4.6 GB
archive on a 32 GB machine. Two ways round it, and they trade different costs:

  Streaming from the tarball needs no disk, but bzip2 cannot seek and
  http.csv sits second in the archive, so every later table pays a full
  decompression of it — twice, because each table is read twice. Measured on
  one core, that is most of a day.

  ``scripts/slim_release.py`` makes a single pass and writes a working copy
  with the free-text columns truncated, about 6 GB at the default setting.
  That is a real reduction to the content modality and it is recorded in the
  manifest rather than hidden. Read its docstring before using it.

Expect twenty minutes to an hour from a directory, dominated by reading
http.csv and by the encoder. Everything is streamed, so peak memory stays
around 4 GB.

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
from mint.sessionise import read_activity  # noqa: E402
from mint.sources import Release, describe  # noqa: E402

REQUIRED = ["logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"]


def check_release(release: Release) -> None:
    """Fail early and usefully rather than three minutes into a read."""
    missing = [f for f in REQUIRED if not release.exists(f)]
    if missing:
        # The commonest shape: --raw points at the folder holding the release
        # rather than the release itself.
        if release.directory:
            nested = [d for d in release.directory.iterdir()
                      if d.is_dir() and (d / "logon.csv").exists()]
            if nested:
                raise SystemExit(
                    f"{release.root} has no activity files, but {nested[0]} "
                    f"does.\nPoint --raw at {nested[0]} instead."
                )
        raise SystemExit(
            f"cannot find {', '.join(missing)} in {describe(release)}.\n\n"
            "Point --raw at one of:\n"
            "  the directory containing logon.csv, or\n"
            "  the r4.2.tar.bz2 file itself (no extraction needed), or\n"
            "  the folder holding the tarballs."
        )
    if not release.list_dir("LDAP"):
        raise SystemExit(
            "no LDAP snapshots found. The organisational context modality "
            "cannot be built without them, and the peer-relative method rests "
            "on it."
        )
    if not (release.list_dir("answers") or release.exists("answers.csv")):
        print(
            "WARNING: no answer files found. The pipeline will run, but with "
            "no labels there is nothing to evaluate against. Point --answers "
            "at answers.tar.bz2 or the extracted directory.",
            file=sys.stderr,
        )


def maybe_subset(release: Release, n_users: int, workdir: Path) -> Path:
    """Write a smaller copy covering the first N users, for a trial run.

    Reads through the same source layer as everything else, so it works
    against an unextracted tarball too — which is the case where a trial run
    matters most, because a full streaming pass is the slow one.
    """
    print(f"building a {n_users}-user subset under {workdir} …")
    workdir.mkdir(parents=True, exist_ok=True)
    users: set[str] | None = None
    for name in REQUIRED:
        stem = name.replace(".csv", "")
        frames = []
        for chunk in read_activity(release, stem):
            if users is None:
                users = set(sorted(chunk["user"].astype(str).unique())[:n_users])
            frames.append(chunk[chunk["user"].astype(str).isin(users)])
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(workdir / name, index=False)
            print(f"  {name}")

    with release.open("psychometric.csv") as stream:
        if stream is not None:
            pd.read_csv(stream).to_csv(workdir / "psychometric.csv", index=False)
    for dirname in ("LDAP", "answers"):
        refs = release.list_dir(dirname)
        if not refs:
            continue
        (workdir / dirname).mkdir(exist_ok=True)
        for ref in refs:
            target = workdir / dirname / Path(ref.split("::")[-1]).name
            with release.open_path(ref) as stream:
                target.write_bytes(stream.read())
    with release.open("answers.csv") as stream:
        if stream is not None:
            (workdir / "answers.csv").write_bytes(stream.read())
    return workdir


def record_source_reduction(release: Release, out: Path) -> None:
    """Carry a slimmed release's truncation setting into the artefacts.

    If the working copy was built by ``slim_release.py`` then its free text was
    cut to a fixed prefix, and results computed from it are not comparable
    with results from the full release. That fact has to travel with the
    artefacts rather than living in the shell history of whoever ran the
    reduction, so it is written into the manifest that every downstream table
    is stamped from.
    """
    if not release.directory:
        return
    slim = release.directory / "slim_manifest.json"
    if not slim.exists():
        return
    reduction = json.loads(slim.read_text(encoding="utf-8"))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_reduction"] = {
        "content_chars": reduction.get("content_chars"),
        "tables": reduction.get("content_truncated_tables"),
        "source_archive": reduction.get("source_archive"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nnote: this release was slimmed — free text truncated to "
          f"{reduction.get('content_chars')} characters. Recorded in the "
          "manifest; cite it with any result.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path,
                    help="the extracted release directory, the r4.2.tar.bz2 "
                         "file itself, or the folder holding the tarballs")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the artefacts (default data/artifacts/cert)")
    ap.add_argument("--text-encoder", default="sentence-transformers",
                    choices=["sentence-transformers", "hashing"])
    ap.add_argument("--answers", type=Path, default=None,
                    help="answers.tar.bz2 or the extracted answers directory, "
                         "if it is not beside the activity files")
    ap.add_argument("--cache", type=Path, default=Path("data/cache"),
                    help="checkpoint directory; a rerun resumes from here "
                         "rather than re-reading the archive. Pass 'none' to "
                         "disable.")
    ap.add_argument("--sample-users", type=int, default=0,
                    help="prepare only the first N users, for a trial run")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config()
    cfg.ensure_dirs()
    extra = [args.answers.expanduser().resolve()] if args.answers else None
    release = Release(args.raw.expanduser().resolve(), extra=extra)
    print(f"reading from: {describe(release)}\n")
    check_release(release)

    if args.sample_users:
        release = Release(maybe_subset(
            release, args.sample_users,
            cfg.path("raw") / f"subset_{args.sample_users}"))

    out = (args.out or cfg.path("artifacts") / "cert").expanduser()
    started = time.time()
    cache = None if str(args.cache).lower() == "none" else args.cache
    bundle = prepare(release, cfg, text_encoder_kind=args.text_encoder,
                     synthetic=False, cache_dir=cache)
    save(bundle, out)
    record_source_reduction(release, out)

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
