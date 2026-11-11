"""Testy jednostkowe greedy CTC decode (blank=0, kolaps powtórzeń, batch)."""
import torch

from data.charset import Charset
from decoding.greedy import greedy_decode


def _logits(sequences, n_classes):
    """Z sekwencji id (lista list) buduje (B, T, C), którego argmax da te id."""
    T = max(len(s) for s in sequences)
    x = torch.zeros(len(sequences), T, n_classes)
    for i, seq in enumerate(sequences):
        for t, c in enumerate(seq):
            x[i, t, c] = 1.0
    return x


def test_repeats_and_blanks_removed():
    cs = Charset("abc")                                   # a=1, b=2, c=3
    ids = [1, 1, 0, 2, 2, 2, 0, 3]
    assert greedy_decode(_logits([ids], len(cs)), cs) == ["abc"]


def test_blank_between_repeats_keeps_char():
    cs = Charset("a")
    ids = [1, 0, 1]                                      # a _ a -> "aa"
    assert greedy_decode(_logits([ids], len(cs)), cs) == ["aa"]


def test_all_blanks_empty_prediction():
    cs = Charset("ab")
    assert greedy_decode(_logits([[0, 0, 0, 0]], len(cs)), cs) == [""]


def test_batch_decodes_independently():
    cs = Charset("ab")
    seqs = [[1, 1, 2, 2], [2, 0, 0, 1]]                  # -> ["ab", "ba"]
    assert greedy_decode(_logits(seqs, len(cs)), cs) == ["ab", "ba"]
