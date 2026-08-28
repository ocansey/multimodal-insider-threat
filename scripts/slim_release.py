#!/usr/bin/env python3
"""Turn r4.2.tar.bz2 into a working copy that fits on a normal disk.

Why this exists
---------------
The published size figures for this release are misleading, and I believed
them until a disk filled up twice. Extracted, r4.2 is not the ~3 GB the file
listing suggests. ``http.csv`` on its own is around 14 GB: twenty-eight
million rows, each carrying a few hundred characters of web page text. The
whole release lands somewhere near 17-18 GB, which does not fit alongside the
4.6 GB archive on a 32 GB machine, and does not fit on most laptops either.

Streaming members straight out of the tarball avoids the disk problem and
creates a worse one. bzip2 cannot seek, so reaching any member means
decompressing everything before it, and ``http.csv`` sits second in the
archive. Every other table therefore costs a full pass through the largest
file in the release, twice over, because the pipeline reads each table once to
build the vocabulary and once to encode. Measured on a Codespace core, that is
most of a day.

This script makes one pass. It reads each member sequentially as bzip2 hands
it over, and writes a directory that the rest of the pipeline treats exactly
like an extracted release. The saving comes from one deliberate reduction:
the free-text ``content`` column of ``file.csv``, ``http.csv`` and
``email.csv`` is truncated to a fixed prefix.

What that reduction costs, honestly
-----------------------------------
It is a real change to the content modality and it is recorded, not hidden.
``slim_manifest.json`` carries the character limit, the row counts and the
resulting sizes, ``prepare_local.py`` copies it into the artefact manifest,
and the data card describes it.

The argument that it is defensible rather than merely convenient: preparation
already discards almost all of this text. It keeps the twelve most informative
documents per user-day, so of twenty-eight million web page bodies, a few
hundred thousand survive to be encoded at all. What the encoder needs from a
page is its topic, and for web text the topic is front-loaded — title, lede,
first sentence. A trailing paragraph of boilerplate is not what separates a
job-listing browse from a documentation browse.

What it does cost is the tail: a page whose only distinguishing content
appears late will be indistinguishable from its neighbours. Set
``--content-chars`` higher if the disk allows it. At 0 the column is dropped
entirely, which is a legitimate ablation rather than a mistake — it answers
"how much is the text worth?" directly.

    # about 6 GB out, from a 4.6 GB archive, single pass
    python scripts/slim_release.py --archive ~/Downloads/r4.2.tar.bz2 \
        --answers ~/Downloads/answers.tar.bz2 --out data/raw/r4.2-slim

    # then, unchanged
    python scripts/prepare_local.py --raw data/raw/r4.2-slim \
        --answers data/raw/r4.2-slim/answers --out data/artifacts/cert

Everything except the three text columns is copied through byte for byte.
Timestamps, ids, users, machines, filenames, URLs, mail recipients, the LDAP
snapshots, the psychometric scores and the answer keys are all untouched.
"""

from __future__ import annotations

import argparse
import bz2
import contextlib
import csv
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

log = logging.getLogger("slim")

#: content is the final column of each of these, per mint.schema.COLUMNS
TEXT_TABLES = {"file.csv", "http.csv", "email.csv"}
ACTIVITY = {"logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"}
COPY_WHOLE = {"psychometric.csv", "license.txt", "answers.csv"}

#: a single content field can be large; the stdlib default rejects it
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


class _Forward(io.RawIOBase):
    """Present a stream-mode tar member as something TextIOWrapper accepts.

    ``tarfile`` in stream mode hands back a file object whose underlying
    ``_Stream`` has no ``seekable``, and ``TextIOWrapper`` asks. Answering
    "no, and you may only go forwards" is both true and all the CSV reader
    needs.
    """

    def __init__(self, inner):
        self._inner = inner

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buf) -> int:
        chunk = self._inner.read(len(buf))
        buf[:len(chunk)] = chunk
        return len(chunk)


def trim_table(stream, target: Path, limit: int) -> tuple[int, int]:
    """Copy a CSV through, truncating its final column. Returns (rows, cut).

    Parsed with the csv module rather than cut down by line length, because
    content fields are quoted and contain commas and newlines. Truncating
    bytes would silently corrupt the column count on exactly the rows whose
    text is most unusual, which is the worst possible place to introduce a
    bias.
    """
    text = io.TextIOWrapper(io.BufferedReader(_Forward(stream), 4 << 20),
                            encoding="utf-8", errors="replace", newline="")
    rows = cut = 0
    first = True
    started = time.time()
    with open(target, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        reader = csv.reader(text)
        for row in reader:
            if not row:
                continue
            # A header row names its columns; never truncate it, and never
            # count it — a row count that silently includes the header is the
            # kind of off-by-one that ends up in a paper.
            if first:
                first = False
                if row[-1].strip().lower() == "content":
                    writer.writerow(row)
                    continue
            rows += 1
            if limit == 0:
                row[-1] = ""
            elif len(row[-1]) > limit:
                row[-1] = row[-1][:limit]
                cut += 1
            writer.writerow(row)
            if rows % 2_000_000 == 0:
                log.info("      %s rows, %s written, %.0fs elapsed",
                         f"{rows:,}", human(target.stat().st_size),
                         time.time() - started)
    return rows, cut


def copy_table(stream, target: Path) -> int:
    """Byte-for-byte copy, for the tables with no free text in them."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        shutil.copyfileobj(stream, out, length=4 << 20)
    return target.stat().st_size


def destination(out: Path, member_name: str) -> Path | None:
    """Where a member of the archive belongs in the slim release.

    The archive nests everything under ``r4.2/``. That prefix is dropped so
    the result looks like a release directory rather than a directory holding
    one, which is the shape ``Release`` and ``check_release`` expect.
    """
    parts = Path(member_name).parts
    if not parts:
        return None
    base = parts[-1]
    parent = parts[-2] if len(parts) > 1 else ""
    if parent in {"LDAP", "answers"}:
        return out / parent / base
    if base in ACTIVITY or base in COPY_WHOLE:
        return out / base
    return None


@contextlib.contextmanager
def open_archive(archive: Path):
    """Open a .tar.bz2 for a single forward pass, multi-stream included.

    ``tarfile.open(mode='r|bz2')`` looks like the obvious call and fails on
    this release with ``EOFError: End of stream already reached``. The reason
    is that r4.2 was compressed by a parallel bzip2, which writes several
    concatenated bzip2 streams into one file. GNU tar decompresses all of
    them; tarfile's own reader stops at the first boundary and reports the
    end of the archive, having silently delivered only part of it.

    Silently is the dangerous word. Had the boundary fallen after the tables
    this script cares about, it would have produced a working copy missing
    some of the release with no error at all. ``bz2.open`` handles
    concatenated streams properly, so the decompression is done there and
    tarfile is handed a plain byte stream with ``mode='r|'``.
    """
    with open(archive, "rb") as raw:
        if raw.read(3) != b"BZh":
            raise SystemExit(
                f"{archive} is not a bzip2 file. If it came from a download, "
                "it is probably a truncated transfer or an error page — check "
                "its size against the 4.6 GB the release should be."
            )
    decompressed = bz2.open(archive, "rb")
    try:
        with tarfile.open(fileobj=decompressed, mode="r|") as tf:
            yield tf
    finally:
        decompressed.close()


def walk(archive: Path, out: Path, limit: int, counts: dict) -> None:
    """One sequential pass over a bzip2 stream. No seeking, no re-reads."""
    log.info("reading %s (%s)", archive.name, human(archive.stat().st_size))
    # Members arrive in archive order and each can be read exactly once.
    # A seeking reader would be worse than it sounds: seeking a bzip2 file
    # means decompressing from the start again for every member.
    with open_archive(archive) as tf:
        for member in tf:
            if not member.isfile():
                continue
            target = destination(out, member.name)
            if target is None:
                continue
            stream = tf.extractfile(member)
            if stream is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            if target.name in TEXT_TABLES:
                log.info("  %-16s truncating content to %d chars",
                         target.name, limit)
                rows, cut = trim_table(stream, target, limit)
                size = target.stat().st_size
                counts[target.name] = {
                    "rows": rows, "truncated_rows": cut, "bytes": size}
                log.info("  %-16s %s rows, %s truncated, %s, %.1f min",
                         target.name, f"{rows:,}", f"{cut:,}", human(size),
                         (time.time() - started) / 60)
            else:
                size = copy_table(stream, target)
                counts[str(target.relative_to(out))] = {"bytes": size}
                log.info("  %-16s %s (copied unchanged)",
                         str(target.relative_to(out)), human(size))
            remaining = free_bytes(out)
            if remaining < 1 << 30:
                raise SystemExit(
                    f"\nstopping: only {human(remaining)} of disk left.\n"
                    f"Rerun with a smaller --content-chars (currently {limit}) "
                    "or a larger disk. Nothing written so far is usable on "
                    "its own; delete " + str(out) + " to reclaim the space."
                )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, type=Path,
                    help="r4.2.tar.bz2 (not extracted)")
    ap.add_argument("--answers", type=Path, default=None,
                    help="answers.tar.bz2, or a directory already extracted")
    ap.add_argument("--out", type=Path, default=Path("data/raw/r4.2-slim"))
    ap.add_argument("--content-chars", type=int, default=160,
                    help="characters of free text kept per row (0 drops the "
                         "column entirely). 160 keeps roughly a third of the "
                         "release size; raise it if the disk allows.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    archive = args.archive.expanduser().resolve()
    if not archive.exists():
        raise SystemExit(f"no archive at {archive}")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # A rough forecast, so a doomed run fails in ten seconds rather than
    # forty minutes. The release is ~4x its compressed size once the text is
    # trimmed to a couple of hundred characters; the constant is empirical.
    projected = archive.stat().st_size * (0.8 + args.content_chars / 320)
    available = free_bytes(out)
    log.info("projecting about %s of output; %s free at %s",
             human(projected), human(available), out)
    if projected > available:
        raise SystemExit(
            f"\nthat will not fit. Projected {human(projected)}, "
            f"{human(available)} free.\n"
            f"Try --content-chars {max(0, args.content_chars // 2)}, or free "
            "space, or point --out at a bigger disk."
        )

    started = time.time()
    counts: dict = {}
    walk(archive, out, args.content_chars, counts)

    if args.answers:
        answers = args.answers.expanduser().resolve()
        if answers.is_dir():
            shutil.copytree(answers, out / "answers", dirs_exist_ok=True)
            log.info("  answers          copied from %s", answers)
        else:
            walk(answers, out, args.content_chars, counts)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    manifest = {
        "source_archive": archive.name,
        "source_bytes": archive.stat().st_size,
        "content_chars": args.content_chars,
        "content_truncated_tables": sorted(TEXT_TABLES),
        "output_bytes": total,
        "files": counts,
        "built_in_minutes": round((time.time() - started) / 60, 1),
        "note": ("Free-text content columns are truncated to content_chars. "
                 "Every other column is byte-identical to the release. "
                 "Results produced from this copy must cite this setting."),
    }
    (out / "slim_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    for name in sorted(counts):
        entry = counts[name]
        rows = f"{entry['rows']:>12,} rows  " if "rows" in entry else " " * 19
        print(f"  {name:<22} {rows}{human(entry['bytes']):>10}")
    print("=" * 68)
    print(f"  {'total':<22}{' ' * 19}{human(total):>10}")
    print(f"\n{human(archive.stat().st_size)} archive -> {human(total)} "
          f"working copy in {(time.time() - started) / 60:.1f} minutes")
    missing = [f for f in ACTIVITY if not (out / f).exists()]
    if missing:
        print(f"\nWARNING: {', '.join(missing)} not found in the archive.")
    if not (out / "LDAP").exists():
        print("\nWARNING: no LDAP directory. Preparation will refuse to run "
              "without it — the peer-relative method needs the org chart.")
    print(f"\nNext:\n  python scripts/prepare_local.py --raw {out} "
          f"--answers {out / 'answers'} --text-encoder hashing "
          "--out data/artifacts/cert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
