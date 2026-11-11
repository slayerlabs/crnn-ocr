"""Rozkład znaków w labelach + które pliki zawierają rzadkie znaki."""
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows

train = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/train")

counts = Counter()
examples = {}  # znak → pierwszy plik, w którym wystąpił
for f in train.glob("*.txt"):
    # jawne UTF-8 + NFC — przed strip i liczeniem
    text = unicodedata.normalize("NFC", f.read_text(encoding="utf-8")).strip()
    counts.update(text)
    for c in set(text):
        examples.setdefault(c, f.name)

print(f"{'znak':>6} | {'liczba':>7} | przykład pliku")
for c, n in counts.most_common():
    print(f"{c!r:>6} | {n:>7} | {examples[c]}")
