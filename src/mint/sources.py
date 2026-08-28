"""Reading the release without unpacking it.

Extracted, r4.2 is over 17 GB, and ``http.csv`` is more than 14 GB of that
on its own — see ``scripts/slim_release.py`` for what to do about it.
Added to the 1.2 GB of tarballs you already downloaded, that is over four
gigabytes of free disk needed to run a study whose output is three hundred
megabytes. On a laptop that is often simply not there, and the failure arrives
halfway through a `tar` command as a truncated CSV that still looks like a
file.

So nothing has to be extracted. This module resolves each table to a byte
stream, whether it lives in a plain directory, as a standalone compressed
file, or as a member inside the ``.tar.bz2`` exactly as it was downloaded.
Everything downstream already consumed an iterator of chunks, so the change is
invisible to the rest of the package.

The cost is honest: bzip2 cannot seek, so reading a member means decompressing
the archive up to it, and the pipeline reads each table twice — once to type
the events, once to collect the text of the events that survived sampling.
That roughly doubles the read time for ``http.csv``. In exchange the whole
study runs with no free disk beyond the download itself, which on most laptops
is the difference between running and not.
"""

from __future__ import annotations

import bz2
import gzip
import logging
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

log = logging.getLogger(__name__)

ARCHIVE_SUFFIXES = (".tar.bz2", ".tar.gz", ".tbz2", ".tgz", ".tar")


def is_archive(path: Path) -> bool:
    name = str(path).lower()
    return path.is_file() and any(name.endswith(s) for s in ARCHIVE_SUFFIXES)


class Release:
    """Where the tables actually are, whichever way the release was unpacked.

    Accepts a directory of CSVs, a directory of compressed CSVs, or the
    downloaded tarball itself. ``extra`` archives are searched too, which is
    how a separately-downloaded ``answers.tar.bz2`` is found without asking
    anyone to move files around.
    """

    def __init__(self, root: Path, extra: list[Path] | None = None):
        self.root = Path(root).expanduser()
        self.archives: list[Path] = []
        self.directory: Path | None = None

        if is_archive(self.root):
            self.archives.append(self.root)
        elif self.root.is_dir():
            self.directory = self.root
            # A directory holding the tarballs rather than their contents is a
            # very common state — it is what you have the moment the download
            # finishes — so treat it as a source rather than an error.
            self.archives.extend(sorted(
                p for p in self.root.iterdir() if is_archive(p)))
        else:
            raise FileNotFoundError(f"{self.root} is neither a directory nor an archive")

        for path in extra or []:
            path = Path(path).expanduser()
            if is_archive(path):
                self.archives.append(path)
            elif path.is_dir() and self.directory is None:
                self.directory = path

        self._members: dict[Path, list[str]] | None = None
        log.info("release rooted at %s (%d archive%s, directory=%s)",
                 self.root, len(self.archives),
                 "" if len(self.archives) == 1 else "s", self.directory)

    # -- membership ---------------------------------------------------------
    def _member_index(self) -> dict[Path, list[str]]:
        if self._members is None:
            self._members = {}
            for archive in self.archives:
                try:
                    with tarfile.open(archive, "r:*") as tf:
                        self._members[archive] = tf.getnames()
                except tarfile.TarError as exc:
                    log.warning("could not read %s: %s", archive.name, exc)
                    self._members[archive] = []
        return self._members

    def _find_member(self, filename: str) -> tuple[Path, str] | None:
        """Locate ``filename`` inside any archive, preferring a shallow match."""
        candidates: list[tuple[int, Path, str]] = []
        for archive, names in self._member_index().items():
            for name in names:
                if Path(name).name == filename:
                    candidates.append((name.count("/"), archive, name))
        if not candidates:
            return None
        _, archive, name = min(candidates)
        return archive, name

    # -- opening ------------------------------------------------------------
    @contextmanager
    def open(self, filename: str) -> Iterator[IO[bytes] | None]:
        """Yield a binary stream for one table, or ``None`` if it is absent."""
        if self.directory is not None:
            plain = self.directory / filename
            if plain.exists() and plain.stat().st_size > 0:
                with open(plain, "rb") as fh:
                    yield fh
                return
            for suffix, opener in ((".gz", gzip.open), (".bz2", bz2.open)):
                packed = self.directory / (filename + suffix)
                if packed.exists():
                    with opener(packed, "rb") as fh:
                        yield fh
                    return

        found = self._find_member(filename)
        if found is None:
            yield None
            return
        archive, member = found
        log.info("streaming %s from %s", member, archive.name)
        with tarfile.open(archive, "r:*") as tf:
            stream = tf.extractfile(member)
            if stream is None:
                yield None
            else:
                yield stream

    def exists(self, filename: str) -> bool:
        if self.directory is not None:
            if (self.directory / filename).exists():
                return True
            if any((self.directory / (filename + s)).exists()
                   for s in (".gz", ".bz2")):
                return True
        return self._find_member(filename) is not None

    # -- directories --------------------------------------------------------
    def list_dir(self, dirname: str) -> list[str]:
        """Names of the CSV files inside a directory such as ``LDAP`` or ``answers``."""
        out: list[str] = []
        if self.directory is not None:
            for candidate in (self.directory / dirname,
                              self.directory.parent / dirname):
                if candidate.is_dir():
                    out.extend(sorted(str(p) for p in candidate.rglob("*.csv")))
        if out:
            return out
        for archive, names in self._member_index().items():
            for name in names:
                parts = Path(name).parts
                if dirname in parts and name.lower().endswith(".csv"):
                    out.append(f"{archive}::{name}")
        return sorted(out)

    @contextmanager
    def open_path(self, reference: str) -> Iterator[IO[bytes]]:
        """Open something returned by :meth:`list_dir`, archive member or file."""
        if "::" in reference:
            archive, member = reference.split("::", 1)
            with tarfile.open(archive, "r:*") as tf:
                stream = tf.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(reference)
                yield stream
        else:
            with open(reference, "rb") as fh:
                yield fh


def describe(release: Release) -> str:
    bits = []
    if release.directory:
        bits.append(f"directory {release.directory}")
    for archive in release.archives:
        bits.append(f"archive {archive.name}")
    return "; ".join(bits) or "nothing found"
