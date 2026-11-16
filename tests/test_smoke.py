"""Smoke test: diagnostyka alfabetu + sprawdzenie loadera na realnych danych.

Uruchomienie:  python tests/test_smoke.py
"""
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, ".")                     # widoczność pakietu `data`
sys.stdout.reconfigure(encoding="utf-8")    # UTF-8 przy redirectach na Windows

from data.charset import build_charset  # noqa: E402  (wymaga sys.path wyżej)
from data.dataset import LineDataset, collate  # noqa: E402

TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/val"
MIN_COUNT = 25   # próg build_charset — znaki o częstości poniżej wypadają


# ---------------------------------------------------------------
# 1. CHAR STATS — rozkład znaków w labelach train
# ---------------------------------------------------------------
def char_stats(dirpath):
    counts = Counter()
    examples = {}
    for f in Path(dirpath).glob("*.txt"):
        text = f.read_text(encoding="utf-8").strip()
        counts.update(text)
        for c in set(text):
            examples.setdefault(c, f.name)
    return counts, examples


print("=== CHAR STATS (train) ===")
counts, examples = char_stats(TRAIN_DIR)
total_chars = sum(counts.values())
print(f"Próbki label: {len(list(Path(TRAIN_DIR).glob('*.txt')))}")
print(f"Znaki unikalne: {len(counts)} | łącznie znaków: {total_chars}")

# znaki, które WYPADNĄ z alfabetu po filtrze min_count (decyzja widoczna co run)
dropped = [(c, n) for c, n in counts.items() if n < MIN_COUNT]
print(f"\nZnaki wypadające z alfabetu (n < {MIN_COUNT}): {len(dropped)}")
for c, n in sorted(dropped, key=lambda x: x[1]):
    print(f"  {c!r}  n={n:<4} np. {examples[c]}")

# sanity: brakujące litery, których spodziewamy się w polskim tekście
expected = set(
    "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ"
    "aąbcćdeęfghijklłmnńoóprsśtuwyzźż0123456789"
)
missing = expected - set(counts)
if missing:
    print(f"\n[UWAGA] Brak w train (może być OK, ale zweryfikuj): {sorted(missing)}")
else:
    print("\n[OK] Wszystkie polskie litery i cyfry obecne.")

# ---------------------------------------------------------------
# 2. CHARSET — budowa (z progiem częstości) + walidacja na val
# ---------------------------------------------------------------
print("\n=== CHARSET ===")
# raportuje znaki spoza alfabetu train obecne w val (znak + pliki)
cs = build_charset(TRAIN_DIR, VAL_DIR, min_count=MIN_COUNT)
print(f"Alfabet (z blank): {len(cs)} znaków")

# ---------------------------------------------------------------
# 3. LOADER — jeden batch, kształty, sanity checks
# ---------------------------------------------------------------
print("\n=== LOADER ===")
ds = LineDataset(TRAIN_DIR, cs)
print(
    f"Próbek train: {len(ds)} "
    f"(charset-skip: {ds.skipped_charset}, CTC-skip: {ds.skipped_ctc})"
)
if ds.skipped_names:
    print(f"  charset-skip train (pierwsze 10): {ds.skipped_names[:10]}")

ds_val = LineDataset(VAL_DIR, cs)
print(
    f"Próbek val:   {len(ds_val)} "
    f"(charset-skip: {ds_val.skipped_charset}, CTC-skip: {ds_val.skipped_ctc})"
)
if ds_val.skipped_names:
    print(f"  charset-skip val (pierwsze 10): {ds_val.skipped_names[:10]}")

x, y, input_lens, y_lens = next(iter(
    DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate)
))
print(f"x.shape      = {x.shape}   (oczekiwane: (4, 1, 32, W))")
print(f"y.shape      = {y.shape}")
print(f"input_lens   = {input_lens.tolist()}   (W/4 — kroki czasowe po CNN)")
print(f"y_lens       = {y_lens.tolist()}")

# transkrypcja z powrotem na tekst — sanity check charset + encoding
sample = cs.decode(y[: y_lens[0]].tolist())
print(f"Transkrypcja [0]: {sample!r}")
assert any(c.isalpha() for c in sample), "Transkrypcja pusta/sama interpunkcja?"

# krytyczny check CTC: labelka <= liczby kroków czasowych
bad = [
    (yl, il) for yl, il in zip(y_lens, input_lens, strict=True) if yl > il
]
assert not bad, f"CTC violation (y_len > input_len): {bad}"
print("[OK] CTC check: labelki mieszczą się w sekwencji czasowej")

# rozmiar klasy spójny z modelem
print(f"[OK] Wszystko gra. Liczba klas dla CRNN: {len(cs)}")
