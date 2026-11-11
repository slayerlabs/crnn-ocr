## Co to jest

CRNN (CNN + BiLSTM + CTC) do rozpoznawania polskich linii tekstu, trenowany
na [ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines).

**Status: WIP** — gotowe: dane, charset z filtracją rzadkich znaków, loader, model CRNN, dekoder, metryki, sanity-check pipeline, trening. W trakcie: ewaluacja (`evaluate.py`), inferencja (`infer.py`).

## Struktura katalogów

```
crnn-ocr/
│
├── README.md                  # opis, setup, status
├── LICENSE                    # pełny tekst licencji MIT
├── environment.yml            # specyfikacja środowiska conda
├── requirements.txt           # zależności pip (to samo co pip w environment.yml)
├── check_env.py               # weryfikacja zależności (exit 0 = OK, 1 = braki)
├── main.py                    # sanity-check pipeline z configu (check przed treningiem)
├── train.py                   # pętla treningu: CTC + TensorBoard + checkpointy
├── .gitignore
│
├── configs/
│   ├── default.yaml           # hiperparametry + ścieżki (jedno źródło prawdy)
│   ├── charset.json           # zamrożony alfabet: 84 klasy, blank=0 (lista JSON)
│   └── char_stats.txt         # rozkład znaków w labelach (lokalny, ignorowany)
│
├── data/
│   ├── __init__.py
│   ├── charset.py             # budowa alfabetu: NFC + filtr min_count (blank=0)
│   ├── dataset.py             # LineDataset (filtry: alfabet, CTC) + collate (lens, H=32)
│   ├── char_stats.py          # rozkład znaków + gdzie występują rzadkie znaki
│   └── raw/                   # dane: pary {base}.png / {base}.txt (IGNOROWANE przez git)
│
├── model/
│   ├── __init__.py
│   └── crnn.py                # CRNN: CNN -> sekwencja -> BiLSTM -> FC (T = W/4)
│
├── decoding/
│   └── greedy.py              # greedy CTC decode (potem: beam.py)
│
├── metrics.py                 # CER / WER / exact match (rapidfuzz) + batch dla train.py
│
└── tests/
    ├── conftest.py            # sys.path dla pytesta (bez znaczenia, skąd odpalane)
    ├── test_charset.py        # itos/stoi, round-trip encode/decode, save/load, min_count, NFC
    ├── test_greedy.py         # greedy decode: blanki, kolaps powtórek, batch
    ├── test_collate.py        # padding zerami, input_lens = w/4 przed paddingiem
    ├── test_crnn.py           # kontrakt kształtów (T = W/4), przepływ gradientów
    ├── test_dataset.py        # filtry datasetu (charset, CTC) na syntetycznych danych
    ├── test_metrics.py        # CER/WER/exact_match: przypadki brzegowe, batch
    └── test_smoke.py          # smoke na realnych danych (wymaga data/raw)
```

`data/raw/` jest ignorowana przez git — dataset pobiera się osobno (poniżej).
`runs/` i `checkpoints/` też są ignorowane (artefakty treningu, nie kod).

Planowane/docelowe pliki (jeszcze nie istnieją): `evaluate.py`, `infer.py`.

## Setup

```bash
conda env create -f environment.yml
conda activate ocr_env

# weryfikacja zależności (użyteczne też w CI — kod wyjścia 0/1)
python check_env.py
```

Albo bez conda: `pip install -r requirements.txt`.

## Dane

Dataset w formie par plików `{base}.png` + `{base}.txt` umieść w:

```
data/raw/train/
data/raw/val/
```

## Regeneracja konfiguracji

Po zmianie danych (lub progu częstości) przebuduj alfabet:

```bash
python data/charset.py      # równoważnie: python -m data.charset
# zapisuje configs/charset.json; wypisuje odrzucone znaki i raport val
```

## Alfabet a znaki egzotyczne

- Transkrypcje czytane jawnym UTF-8 + normalizacją NFC (przed strip/encode —
  chroni przed podwojonymi literami przy danych w formie NFD).
- `build_charset(..., min_count=25)`: znaki rzadsze niż 25× w train (cyrylica,
  katakana/kanji, pojedyncze znaki obce) wypadają z alfabetu. Pełna lista
  z licznikami przy budowie + w `cs.dropped_chars`. Obecnie: 84 klasy.
- Znaki val spoza alfabetu train są **raportowane** (znak + pliki), bez wyjątku.
- `LineDataset` pomija próbki dwoma filtrami, w kolejności: 1) znak spoza
  alfabetu (`skipped_charset`, nazwy w `skipped_names` — np. polska linia
  z „Märtenstorget"), 2) labelka dłuższa niż W/4 kroków (`skipped_ctc`).
- Ground truth nietknięty — bez `<unk>`, bez transliteracji. `Charset.encode`
  rzuca jawny wyjątek (znak + hex U+XXXX); awaryjne mapowanie nieznanego
  znaku pojawi się dopiero przy inferencji.

## Sanity checks

Komendy przed commitem (z katalogu głównego repo, po `conda activate ocr_env`):

```bash
# 1. Środowisko — wszystkie zależności + GPU
python check_env.py
#    oczekiwane: same [OK], "Wszystkie zależności są dostępne. OK", exit 0

# 2. Linter — jakość kodu
python -m ruff check . --select E,W,F,I,B,UP --exclude data/raw
#    oczekiwane: All checks passed!

# 3. Testy jednostkowe — bez danych, bez GPU (ok. 7 s)
python -m pytest tests/ -q
#    oczekiwane: 35 passed

# 4. Smoke na realnych danych — diagnostyka charsetu + loadera
python tests/test_smoke.py
#    oczekiwane: 59 znaków wypadających (n < 25), raport val (10 znaków),
#    train=1737 (charset-skip: 228, CTC-skip: 35), val=181 (18, 1),
#    na końcu: "[OK] Wszystko gra. Liczba klas dla CRNN: 84"

# 5. Pełny pipeline przez model (GPU)
python main.py --config configs/default.yaml
#    oczekiwane: 84 klasy, skończony CTC loss (bez inf/NaN),
#    "Metryki batcha: CER=... WER=... exact_match=..." (bez treningu),
#    "[OK] Pipeline działa end-to-end (config -> dane -> model -> CTC -> dekoder -> metryki)"

# 6. Opcjonalnie: determinizm przebudowy alfabetu
python -m data.charset > a.txt && python -m data.charset > b.txt && diff a.txt b.txt && rm a.txt b.txt
#    oczekiwane: diff milczy, "Zapisano: configs/charset.json"
```

Uwagi:

- PowerShell 5.x nie obsługuje `&&` — krok 6 rozpisz linia po linii
  (zamiast `diff` użyj `Compare-Object (Get-Content a.txt) (Get-Content b.txt)`)
  albo odpal go w Git Bash.
- Liczby w oczekiwanych wynikach (35 passed, 84 klasy, liczba skipów) dotyczą
  aktualnego datasetu — po zmianie danych oceń wyniki jakościowo
  (struktura outputu, brak błędów), a nie po dokładnych wartościach.

## Trening

```bash
python train.py --config configs/default.yaml
```

- Checkpointy: `checkpoints/best.pt` (najniższy val CER) i `checkpoints/last.pt`,
- Logi: `tensorboard --logdir runs/` (train/loss, val/loss, val/cer, val/wer, val/exact_match),
- Walidacja po każdej epoce: CER / WER / exact match (`metrics.batch_metrics`, greedy decode),
- Checkpoint zawiera `model`, `itos` (alfabet), `epoch` i metryki walidacji.

## Wyniki

*(do uzupełnienia po pierwszych treningach: CER / WER / exact match na zbiorze walidacyjnym i testowym)*

## Licencja

Kod projektu: licencja **MIT** — pełny tekst w [LICENSE](LICENSE).

Dane treningowe (`data/raw/`) nie są częścią repozytorium. Zbiór
[ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines)
jest udostępniony na licencji **MIT** (zgodnie z kartą datasetu), więc
publikacja artefaktów treningu (modele, checkpointy) również nie jest
ograniczona.

## Autorzy

* **Paweł Aksiutowicz** — Lead Developer

## Podziękowania

* Generowanie i optymalizacja kodu przy wsparciu **Mistral AI**

* Teksty korpusu dzięki uprzejmości **Wolnych Lektur** ([wolnelektury.pl](https://wolnelektury.pl/)) oraz datasetowi [ArielUW/wolnelektury](https://huggingface.co/datasets/ArielUW/wolnelektury) na Hugging Face

* Szczególne podziękowania za nieocenione zasoby, kod i mocne wsparcie w nauce:

  * Arkadiusz Słota [@Maggio333](https://github.com/Maggio333/)
  * Kacper Wikieł [@kwikiel](https://github.com/kwikiel/)
  * **Piotr Styla** [@PiotrStyla](https://github.com/PiotrStyla) — [zbiór linii treningowych](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines)
  * **Slayer AI Labs** [@slayerlabs](https://github.com/slayerlabs)