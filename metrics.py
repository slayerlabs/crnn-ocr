"""Metryki jakości OCR: CER, WER, exact match (rapidfuzz).

CER = odległość Levenshteina (znaki) / długość referencji,
WER = odległość Levenshteina (słowa) / liczba słów referencji.
Wartości to ułamki: 0.0 = idealnie, 1.0 = 100% błędu.

Wersja batchowa (batch_metrics) służy walidacji w pętli treningu (train.py):
    m = batch_metrics(refs_batch, hyps_batch)   # -> dict z cer/wer/exact_match
"""
from rapidfuzz.distance import Levenshtein


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate dla jednej pary referencja → hipoteza."""
    if not ref:
        return 0.0 if hyp == ref else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate dla jednej pary (split po białych znakach)."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if h == r else 1.0
    return Levenshtein.distance(r, h) / len(r)


def exact_match(ref: str, hyp: str) -> float:
    return float(ref == hyp)


def batch_metrics(refs, hyps) -> dict:
    """Średnie metryki dla list par (batch z walidacji w train.py)."""
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) i hyps ({len(hyps)}) — różne długości")
    if not refs:
        return {"cer": 0.0, "wer": 0.0, "exact_match": 0.0, "n": 0}
    n = len(refs)
    pairs = list(zip(refs, hyps, strict=True))
    return {
        "cer": sum(cer(r, h) for r, h in pairs) / n,
        "wer": sum(wer(r, h) for r, h in pairs) / n,
        "exact_match": sum(exact_match(r, h) for r, h in pairs) / n,
        "n": n,
    }
