"""Testy jednostkowe LineDataset: filtry charset i CTC, tmp_path bez data/raw."""
import torch
from PIL import Image

from data.charset import Charset
from data.dataset import TARGET_H, LineDataset


def _write_pair(d, name, text, w_px, h_px):
    Image.new("L", (w_px, h_px), 255).save(d / f"{name}.png")
    (d / f"{name}.txt").write_text(text, encoding="utf-8")


def test_skips_samples_breaking_ctc_contract(tmp_path):
    cs = Charset("abcdefghi")
    # w=32, h=32 -> po resize w=32 -> T=8; labelka 9 znaków > 8 -> skip
    _write_pair(tmp_path, "toolong", "abcdefghi", 32, 32)
    _write_pair(tmp_path, "ok", "ab", 32, 32)

    ds = LineDataset(str(tmp_path), cs)
    assert ds.ids == ["ok"]                      # sortowanie + selekcja
    assert ds.skipped_ctc == 1


def test_filter_uses_resized_width(tmp_path):
    cs = Charset("abcdefghi")
    # h=8 < 32: po resize w = round(16 * 32/8) = 64 -> T=16; 9 znaków się mieści
    _write_pair(tmp_path, "upscaled", "abcdefghi", 16, 8)

    ds = LineDataset(str(tmp_path), cs)
    assert ds.ids == ["upscaled"]
    assert ds.skipped_ctc == 0


def test_skips_samples_with_chars_outside_alphabet(tmp_path):
    cs = Charset("abc")
    _write_pair(tmp_path, "foreign", "aXd", 64, 64)     # X poza alfabetem
    _write_pair(tmp_path, "ok", "abc", 64, 64)

    ds = LineDataset(str(tmp_path), cs)
    assert ds.ids == ["ok"]
    assert ds.skipped_charset == 1
    assert ds.skipped_names == ["foreign"]      # nazwy bez rozszerzeń
    assert ds.skipped_ctc == 0


def test_charset_filter_runs_before_ctc(tmp_path):
    # próbka łamie OBA warunki — liczy się raz, jako charset-skip
    cs = Charset("ab")
    _write_pair(tmp_path, "both", "aX" * 20, 32, 32)    # obcy znak + za długa labelka

    ds = LineDataset(str(tmp_path), cs)
    assert ds.skipped_charset == 1
    assert ds.skipped_ctc == 0
    assert len(ds) == 0


def test_png_without_txt_ignored(tmp_path):
    Image.new("L", (32, 32), 255).save(tmp_path / "orphan.png")
    ds = LineDataset(str(tmp_path), Charset("ab"))
    assert len(ds) == 0


def test_getitem_shape_and_encoding(tmp_path):
    cs = Charset("ab")
    _write_pair(tmp_path, "p", "ab", 64, 64)     # kwadrat 64x64 -> po resize 32x32

    ds = LineDataset(str(tmp_path), cs)
    x, y = ds[0]
    assert x.shape == (1, TARGET_H, 32)
    assert y.dtype == torch.long
    assert cs.decode(y.tolist()) == "ab"
