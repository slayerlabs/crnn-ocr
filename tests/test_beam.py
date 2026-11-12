"""Testy beam search CTC — kontrakt jak greedy + krzyżowo z brute-force."""
import itertools

import torch

from data.charset import Charset
from decoding.beam import beam_decode
from decoding.greedy import greedy_decode


def _logits_from_rows(rows):
    """rows: prawdopodobieństwa per krok (sumują się do 1) -> (1, T, C)."""
    return torch.log(torch.tensor(rows, dtype=torch.float64)).unsqueeze(0)


def test_clear_cases_match_greedy():
    cs = Charset("ab")
    # a _ b (blank rozdziela) -> "ab"
    rows = [[0.10, 0.80, 0.10],
            [0.80, 0.10, 0.10],
            [0.10, 0.10, 0.80]]
    assert beam_decode(_logits_from_rows(rows), cs) == ["ab"]

    # a _ a -> "aa" (powtórka po blanku NIE zwija się)
    rows = [[0.10, 0.80, 0.10],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10]]
    assert beam_decode(_logits_from_rows(rows), cs) == ["aa"]

    # a a a bez blanków -> "a" (kolaps powtórek)
    rows = [[0.10, 0.80, 0.10]] * 3
    assert beam_decode(_logits_from_rows(rows), cs) == ["a"]


def test_all_blanks_empty_prediction():
    cs = Charset("ab")
    rows = [[0.95, 0.03, 0.02]] * 4
    assert beam_decode(_logits_from_rows(rows), cs) == [""]


def test_batch_decodes_independently():
    cs = Charset("ab")
    r1 = [[0.10, 0.80, 0.10], [0.80, 0.10, 0.10], [0.10, 0.10, 0.80]]  # -> "ab"
    r2 = [[0.10, 0.10, 0.80], [0.85, 0.10, 0.05], [0.10, 0.10, 0.80]]  # -> "bb"
    logits = torch.log(torch.stack([
        torch.tensor(r1, dtype=torch.float64),
        torch.tensor(r2, dtype=torch.float64),
    ]))
    assert beam_decode(logits, cs, beam_size=20) == ["ab", "bb"]


def _collapse(path):
    out, prev = [], None
    for c in path:
        if c != 0 and c != prev:
            out.append(c)
        prev = c
    return tuple(out)


def _brute_best_label(rows):
    """Suma mas wszystkich ścieżek per etykieta -> najlepsza etykieta."""
    mass = {}
    for path in itertools.product(range(len(rows[0])), repeat=len(rows)):
        p = 1.0
        for t, c in enumerate(path):
            p *= rows[t][c]
        lab = _collapse(path)
        mass[lab] = mass.get(lab, 0.0) + p
    return max(mass, key=mass.get)


def test_beam_equals_bruteforce():
    """Dla małego przypadku beam (szeroki) = globalnie najlepsza etykieta."""
    cs = Charset("ab")
    torch.manual_seed(0)
    for _ in range(5):
        rows = torch.softmax(torch.randn(4, 3), dim=-1).tolist()
        expected = "".join(cs.itos[i] for i in _brute_best_label(rows))
        got = beam_decode(_logits_from_rows(rows), cs, beam_size=64)
        assert got == [expected]


def test_beam_agrees_with_greedy_when_unambiguous():
    """Pikowane prawdopodobieństwa -> obie metody dają to samo."""
    cs = Charset("abc")
    seq = [1, 1, 0, 2, 0, 3, 3]                      # -> "abc"
    n = len(cs.itos)
    rows = []
    for c in seq:
        r = [0.001] * n
        r[c] = 0.99
        rows.append(r)
    logits = _logits_from_rows(rows)
    assert greedy_decode(logits, cs) == beam_decode(logits, cs, beam_size=10)
