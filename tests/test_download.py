"""Testy jednostkowe auto-pobrania datasetu (file:// URL — bez sieci)."""
import hashlib
import tarfile

import pytest

from data.download import ensure_dataset, has_pairs


def _make_pair(root, name, text="ab"):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.png").write_bytes(b"\x89PNG-fake")
    (root / f"{name}.txt").write_text(text, encoding="utf-8")


def _make_archive(tar_path, workdir):
    """Archiwum z train/ i val/ w korzeniu — jak ocr-pl-lines-v1.tar.gz."""
    sub = workdir / "content"
    for split, n in (("train", 2), ("val", 1)):
        for i in range(n):
            _make_pair(sub / split, f"{i:05d}")
    with tarfile.open(tar_path, "w:gz") as tar:
        for member in sorted(sub.iterdir()):
            tar.add(member, arcname=member.name)
    return hashlib.sha256(tar_path.read_bytes()).hexdigest()


def test_has_pairs(tmp_path):
    assert not has_pairs(tmp_path / "nie-ma")
    d = tmp_path / "train"
    _make_pair(d, "00000")
    assert has_pairs(d)


def test_ensure_skips_when_data_present(tmp_path):
    train, val = tmp_path / "train", tmp_path / "val"
    _make_pair(train, "00000")
    _make_pair(val, "00000")
    assert ensure_dataset(train, val, url=None, sha256=None) is False


def test_ensure_no_data_and_no_url_exits(tmp_path):
    with pytest.raises(SystemExit):
        ensure_dataset(tmp_path / "train", tmp_path / "val",
                       url=None, sha256=None)


def test_ensure_downloads_and_extracts(tmp_path):
    raw = tmp_path / "raw"
    train, val = raw / "train", raw / "val"
    sha = _make_archive(tmp_path / "ds.tar.gz", tmp_path)
    url = (tmp_path / "ds.tar.gz").as_uri()
    assert ensure_dataset(train, val, url=url, sha256=sha) is True
    assert has_pairs(train) and has_pairs(val)
    assert not (raw / "ocr-pl-lines.tar.gz.part").exists()   # .part sprzątany


def test_ensure_sha_mismatch_aborts_and_cleans(tmp_path):
    raw = tmp_path / "raw"
    train, val = raw / "train", raw / "val"
    _make_archive(tmp_path / "ds.tar.gz", tmp_path)          # sha będzie zła
    with pytest.raises(SystemExit):
        ensure_dataset(train, val, url=(tmp_path / "ds.tar.gz").as_uri(),
                       sha256="0" * 64)
    assert not train.exists()               # nic nie rozpakowane
    assert not (raw / "ocr-pl-lines.tar.gz.part").exists()
