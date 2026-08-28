"""The slim working copy must change the text column and nothing else.

This script exists because r4.2 does not fit on an ordinary disk, and the way
it makes the release fit is by throwing away most of the free text. That is a
reduction with consequences, so the thing worth testing is not that it runs —
it is that the reduction is confined to exactly the column it claims and that
the awkward CSV rows survive it.

The awkward rows are the point. Web page content contains commas, quotation
marks and newlines. Truncating this file by byte length, which is the obvious
shell one-liner, corrupts the column count on precisely those rows — the ones
whose text is least ordinary. A corrupted row does not raise; pandas either
drops it or shifts every field left, and the rows lost are a biased sample.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "slim_release.py"

NASTY = 'Careers, jobs and "opportunities"\nline two of the page ' + "z" * 500


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture(scope="module")
def archive(tmp_path_factory) -> tuple[Path, Path]:
    """A miniature release with the real column layout and the real nasties."""
    work = tmp_path_factory.mktemp("release")
    src = work / "r4.2"
    _write(src / "logon.csv", ["id", "date", "user", "pc", "activity"],
           [[f"{{L{i}}}", "01/02/2010 07:15:00", "AAM0658", "PC-1", "Logon"]
            for i in range(4)])
    _write(src / "device.csv", ["id", "date", "user", "pc", "activity"],
           [[f"{{D{i}}}", "01/02/2010 07:20:00", "AAM0658", "PC-1", "Connect"]
            for i in range(3)])
    _write(src / "http.csv",
           ["id", "date", "user", "pc", "url", "content"],
           [[f"{{H{i}}}", "01/02/2010 08:00:00", "AAM0658", "PC-1",
             "http://jobs.example.com/a,b", NASTY] for i in range(5)])
    _write(src / "file.csv",
           ["id", "date", "user", "pc", "filename", "content"],
           [["{F0}", "01/02/2010 09:00:00", "AAM0658", "PC-1",
             "R:\\dump.doc", "short"]])
    _write(src / "email.csv",
           ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size",
            "attachments", "content"],
           [["{E0}", "01/02/2010 10:00:00", "AAM0658", "PC-1", "a@x.com",
             "", "", "b@dtaa.com", "2000", "0", "body " + "y" * 400]])
    _write(src / "psychometric.csv",
           ["employee_name", "user_id", "O", "C", "E", "A", "N"],
           [["Ann M", "AAM0658", 30, 40, 50, 20, 10]])
    _write(src / "LDAP" / "2010-01.csv",
           ["employee_name", "user_id", "email", "role", "business_unit",
            "functional_unit", "department", "team", "supervisor"],
           [["Ann M", "AAM0658", "a@dtaa.com", "Salesman", "1 - BU",
             "2 - FU", "3 - Dept", "4 - Team", "Boss B"]])
    (src / "license.txt").write_text("terms\n", encoding="utf-8")

    ans = work / "ansdir" / "answers"
    ans.mkdir(parents=True)
    (ans / "r4.2-1.csv").write_text(
        "logon,{L1},01/02/2010 07:15:00,AAM0658,PC-1,Logon\n", encoding="utf-8")

    tarball = work / "r4.2.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as tf:
        tf.add(src, arcname="r4.2")
    answers = work / "answers.tar.bz2"
    with tarfile.open(answers, "w:bz2") as tf:
        tf.add(ans, arcname="answers")
    return tarball, answers


def _run(archive: Path, answers: Path, out: Path, chars: int) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--archive", str(archive),
         "--answers", str(answers), "--out", str(out),
         "--content-chars", str(chars)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def slim(archive, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("slim") / "r4.2-slim"
    _run(archive[0], archive[1], out, 160)
    return out


def test_the_release_keeps_the_shape_the_pipeline_expects(slim):
    for name in ("logon.csv", "device.csv", "file.csv", "http.csv",
                 "email.csv", "psychometric.csv"):
        assert (slim / name).exists(), f"{name} missing from the slim copy"
    assert (slim / "LDAP" / "2010-01.csv").exists()
    assert (slim / "answers" / "r4.2-1.csv").exists()


def test_only_the_content_column_is_shortened(slim):
    http = pd.read_csv(slim / "http.csv")
    assert list(http.columns) == ["id", "date", "user", "pc", "url", "content"]
    assert len(http) == 5, "a row was lost or split"
    # the fields that are not free text arrive untouched, commas and all
    assert (http["url"] == "http://jobs.example.com/a,b").all()
    assert (http["user"] == "AAM0658").all()
    assert http["id"].tolist() == [f"{{H{i}}}" for i in range(5)]


def test_the_prefix_is_exactly_what_was_asked_for(slim):
    http = pd.read_csv(slim / "http.csv")
    assert (http["content"].str.len() == 160).all()
    # and it is a prefix of the original, not a re-encoding of it
    assert http["content"].iloc[0] == NASTY[:160]
    # which means the quote and the newline inside it came through intact
    assert '"' in http["content"].iloc[0]
    assert "\n" in http["content"].iloc[0]


def test_tables_with_no_free_text_are_untouched(slim):
    logon = pd.read_csv(slim / "logon.csv")
    assert len(logon) == 4
    assert logon["activity"].tolist() == ["Logon"] * 4


def test_short_content_is_left_alone(slim):
    assert pd.read_csv(slim / "file.csv")["content"].iloc[0] == "short"


def test_zero_chars_empties_the_column_without_breaking_the_row(
        archive, tmp_path):
    out = tmp_path / "dropped"
    _run(archive[0], archive[1], out, 0)
    http = pd.read_csv(out / "http.csv", keep_default_na=False)
    assert len(http) == 5
    assert (http["content"] == "").all()
    assert (http["url"] == "http://jobs.example.com/a,b").all()


def test_the_manifest_records_the_reduction(slim):
    import json
    manifest = json.loads((slim / "slim_manifest.json").read_text())
    assert manifest["content_chars"] == 160
    assert set(manifest["content_truncated_tables"]) == {
        "file.csv", "http.csv", "email.csv"}
    # row counts exclude the header, or every count in the paper is off by one
    assert manifest["files"]["http.csv"]["rows"] == 5
    assert manifest["files"]["http.csv"]["truncated_rows"] == 5
    assert manifest["files"]["file.csv"]["truncated_rows"] == 0


def test_a_multi_stream_bzip2_archive_is_read_to_the_end(archive, tmp_path):
    """r4.2 is compressed as several concatenated bzip2 streams.

    Parallel bzip2 implementations write one stream per block, and the result
    is a valid .bz2 that GNU tar reads without complaint. Python's
    ``tarfile.open(mode='r|bz2')`` stops at the first stream boundary — it
    does not raise where the boundary falls between members, it simply reports
    the archive as finished. A working copy silently missing half the release
    is the worst failure this script could have, so the multi-stream case is
    built here explicitly rather than trusted.
    """
    import bz2 as _bz2

    plain = tmp_path / "release.tar"
    with tarfile.open(plain, "w") as tf, tarfile.open(archive[0], "r:bz2") as src:
        for member in src:
            extracted = src.extractfile(member)
            tf.addfile(member, extracted)

    raw = plain.read_bytes()
    half = len(raw) // 2
    multi = tmp_path / "multi.tar.bz2"
    # two independent bzip2 streams, concatenated — exactly what pbzip2 emits
    multi.write_bytes(_bz2.compress(raw[:half]) + _bz2.compress(raw[half:]))
    assert raw == _bz2.BZ2File(multi).read(), "fixture is not multi-stream"

    out = tmp_path / "slim"
    _run(multi, archive[1], out, 160)

    # every table survived the stream boundary
    for name in ("logon.csv", "device.csv", "file.csv", "http.csv",
                 "email.csv", "psychometric.csv"):
        assert (out / name).exists(), f"{name} lost at a bzip2 stream boundary"
    assert len(pd.read_csv(out / "http.csv")) == 5


def test_a_file_that_is_not_bzip2_is_rejected_clearly(tmp_path):
    """A truncated download or an HTML error page should say so."""
    fake = tmp_path / "r4.2.tar.bz2"
    fake.write_bytes(b"<html>404 Not Found</html>")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--archive", str(fake),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "not a bzip2 file" in result.stdout + result.stderr
