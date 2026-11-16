"""Testy jednostkowe metryk: CER, WER, exact match, batch (bez danych, GPU)."""
import pytest

from metrics import batch_metrics, cer, exact_match, wer


def test_cer_identical():
    assert cer("kotek", "kotek") == 0.0


def test_cer_substitution():
    assert cer("kotek", "kotel") == pytest.approx(1 / 5)


def test_cer_insertion():
    assert cer("abc", "abcd") == pytest.approx(1 / 3)


def test_cer_empty_ref():
    assert cer("", "") == 0.0
    assert cer("", "abc") == 1.0


def test_wer_identical_sentence():
    assert wer("ala ma kota", "ala ma kota") == 0.0


def test_wer_one_word_wrong():
    # ala ma kota -> ala ma psa: 1 ze 3 słów == 1/3
    assert wer("ala ma kota", "ala ma psa") == pytest.approx(1 / 3)


def test_wer_char_errors_inside_words():
    # literówki wewnątrz słów = całe słowa błędne na poziomie Levenshteina
    assert wer("żubr pączek", "zubre pączekk") == pytest.approx(1.0)


def test_exact_match():
    assert exact_match("abc", "abc") == 1.0
    assert exact_match("abc", "abd") == 0.0


def test_batch_metrics_means():
    refs = ["ala ma kota", "żubr"]
    hyps = ["ala ma kota", "zubr"]
    m = batch_metrics(refs, hyps)
    assert m["n"] == 2
    assert m["exact_match"] == 0.5
    assert m["cer"] == pytest.approx((0.0 + 0.25) / 2)
    assert m["wer"] == pytest.approx((0.0 + 1.0) / 2)


def test_batch_metrics_empty():
    assert batch_metrics([], []) == {"cer": 0.0, "wer": 0.0, "exact_match": 0.0, "n": 0}


def test_batch_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="różne długości"):
        batch_metrics(["a"], [])
