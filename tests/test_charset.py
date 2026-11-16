"""Testy jednostkowe Charset (alfabet CTC): kodowanie, blank=0, persistencja."""
import pytest

from data.charset import Charset, build_charset


def test_blank_at_index_zero():
    cs = Charset("abc")
    assert cs.itos[0] == "<blank>"
    assert cs.stoi["<blank>"] == 0


def test_len_includes_blank():
    assert len(Charset("abc")) == 4


def test_sorted_unique_without_blank():
    cs = Charset("cabc")                     # duplikaty mają zniknąć
    assert cs.itos[1:] == ["a", "b", "c"]


def test_encode_decode_roundtrip():
    cs = Charset("aąbc ń")
    text = "bą cń"
    assert cs.decode(cs.encode(text)) == text


def test_encode_unknown_char_raises():
    cs = Charset("ab")
    # jawny wyjątek z znakiem i hexem (celowo brak <unk> w treningu)
    with pytest.raises(KeyError, match=r"U\+0058"):
        cs.encode("aXb")


def test_save_load_roundtrip(tmp_path):
    cs = Charset("aąbc")
    p = tmp_path / "charset.json"
    cs.save(p)
    cs2 = Charset.load(p)
    assert cs2.itos == cs.itos
    assert cs2.stoi == cs.stoi


def test_build_charset_sorted(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (d / "1.txt").write_text("cab", encoding="utf-8")
    cs = build_charset(str(d), min_count=1)
    assert cs.itos[1:] == ["a", "b", "c"]


def test_min_count_filters_rare_chars(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (d / "1.txt").write_text("aaab", encoding="utf-8")
    (d / "2.txt").write_text("aa", encoding="utf-8")    # a=5, b=1
    cs = build_charset(str(d), min_count=3)
    assert cs.itos[1:] == ["a"]
    assert cs.dropped_chars == {"b": 1}


def test_build_charset_reports_val_chars_outside_alphabet(tmp_path, capsys):
    train, val = tmp_path / "train", tmp_path / "val"
    train.mkdir()
    val.mkdir()
    (train / "1.txt").write_text("cab", encoding="utf-8")
    (val / "1.txt").write_text("ab", encoding="utf-8")
    (val / "2.txt").write_text("ZZ", encoding="utf-8")   # Z tylko w val

    cs = build_charset(str(train), str(val), min_count=1)
    assert cs.itos[1:] == ["a", "b", "c"]
    out = capsys.readouterr().out
    assert "'Z'" in out        # znak jawnie w raporcie
    assert "2.txt" in out      # plik jawnie w raporcie


def test_nfd_input_normalized_to_nfc(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (d / "1.txt").write_text("z\u0301z\u0301", encoding="utf-8")   # ź w NFD

    cs = build_charset(str(d), min_count=1)
    assert cs.itos[1:] == ["ź"]     # jeden znak NFC, nie 'z' + U+0301 osobno
