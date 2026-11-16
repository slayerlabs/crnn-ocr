"""Budowa alfabetu (charsetu) z danych treningowych. Blank (CTC) = indeks 0.

Wszystkie wczytania transkrypcji: jawne UTF-8 + normalizacja NFC.
"""
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def read_transcripts(dirpath):
    """Kolejność (ścieżka, tekst): jawne UTF-8 + NFC przed strip/encode."""
    for f in sorted(Path(dirpath).glob("*.txt")):
        text = unicodedata.normalize("NFC", f.read_text(encoding="utf-8"))
        yield f, text.strip()


class Charset:
    def __init__(self, chars: str):
        self.itos = ["<blank>"] + sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.dropped_chars = {}   # uzupełnia build_charset: znaki poniżej progu

    def encode(self, text: str):
        ids = []
        for c in text:
            if c not in self.stoi:
                raise KeyError(
                    f"Znak poza alfabetem: {c!r} (U+{ord(c):04X}). "
                    "Przefiltruj dane (LineDataset.skipped_charset) "
                    "albo przebuduj alfabet: python -m data.charset"
                )
            ids.append(self.stoi[c])
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.itos, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls("")
        obj.itos = itos
        obj.stoi = {c: i for i, c in enumerate(itos)}
        return obj


def build_charset(
    train_dir: str, val_dir: str = None, min_count: int = 25
) -> Charset:
    """Alfabet ze znaków train o częstości >= min_count (blank=0, sorted).

    Odrzucone znaki (n < min_count): atrybut cs.dropped_chars + log.
    Jeśli val_dir: jawnie raportuje znaki spoza alfabetu train (znak,
    hex, n_train, pliki val) — NIE rzuca wyjątku; te próbki val zostaną
    odfiltrowane w LineDataset (skipped_charset).
    """
    counter = Counter()
    for _, text in read_transcripts(train_dir):
        counter.update(text)

    keep = {c for c, n in counter.items() if n >= min_count}
    cs = Charset("".join(sorted(keep)))
    cs.dropped_chars = {
        c: n
        for c, n in sorted(counter.items(), key=lambda kv: -kv[1])
        if n < min_count
    }

    if cs.dropped_chars:
        print(
            f"[charset] odrzucone przez próg n < {min_count}: "
            f"{len(cs.dropped_chars)} znaków"
        )
        for c, n in cs.dropped_chars.items():
            print(f"    {c!r}  U+{ord(c):04X}  n={n}")

    if val_dir:
        foreign = {}
        for f, text in read_transcripts(val_dir):
            for c in set(text) - set(cs.itos):
                foreign.setdefault(c, []).append(f.name)
        if foreign:
            print(
                f"[charset][UWAGA] znaki spoza alfabetu train obecne w val: "
                f"{len(foreign)}"
            )
            for c in sorted(foreign):
                shown = ", ".join(foreign[c][:5])
                extra = f" (+{len(foreign[c]) - 5})" if len(foreign[c]) > 5 else ""
                print(
                    f"    {c!r}  U+{ord(c):04X}  n_train={counter.get(c, 0)}"
                    f"  val: {shown}{extra}"
                )

    return cs


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows

    cs = build_charset("data/raw/train", "data/raw/val")
    print(f"\nRozmiar alfabetu (z blank): {len(cs)}")
    print("Znaki:", "".join(cs.itos))
    cs.save("configs/charset.json")
    print("Zapisano: configs/charset.json")
