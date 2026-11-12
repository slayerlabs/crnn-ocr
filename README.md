## Co to jest

CRNN (CNN + BiLSTM + CTC) do rozpoznawania polskich linii tekstu, trenowany
na [ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines).

**Status: WIP** — gotowe: dane, charset z filtracją rzadkich znaków, loader,
model CRNN, dekoder (greedy + beam), metryki, sanity-check pipeline, trening
(+`--resume`), auto-pobranie danych z HF, predykcja (`predict.py`), ewaluacja
(`evaluate.py` → `metrics.json`).

## Komendy (wszystkie w jednym miejscu)

Setup (raz, po klonie):

```bash
conda env create -f environment.yml && conda activate ocr_env   # albo: pip install -r requirements.txt
python check_env.py                                             # weryfikacja zależności (exit 0 = OK)
```

Dane: nic nie robisz — `train.py`/`main.py` pobiorą dataset z HF
automatycznie, gdy `data/raw/{train,val}` są puste (raz, idempotentnie;
URL + SHA-256 przypięte w `configs/default.yaml`).

### Trening i resume

```bash
python train.py --config configs/default.yaml                         # trening
python train.py --config configs/default.yaml --resume checkpoints/last.pt   # kontynuacja
```

### Ewaluacja i predykcja

```bash
python evaluate.py                                # best.pt, greedy -> metrics.json
python evaluate.py --decoder beam --out metrics_beam.json
python evaluate.py --checkpoint inny.pt --device cpu --out raport.json

python predict.py --image linia.png               # greedy (domyślnie)
python predict.py --image linia.png --decoder beam --beam-size 10
python predict.py --image linia.png --checkpoint checkpoints/best.pt --device cpu

# demo bez datasetu: deterministyczna linia testowa -> predykcja
python make_test_line.py && python predict.py --image test_line.png
```

### Sanity checks (przed commitem)

```bash
python check_env.py                                                        # zależności + GPU
python -m ruff check .                                                     # linter (config: ruff.toml)
python -m pytest tests/ -q                                                 # testy (bez danych, <1 min)
python tests/test_smoke.py                                                 # smoke na realnych danych
python main.py --config configs/default.yaml                               # pełny pipeline, metryki batcha
python evaluate.py                                                         # metryki best.pt -> metrics.json
python -m data.charset > a.txt && python -m data.charset > b.txt && diff a.txt b.txt && rm a.txt b.txt   # determinizm przebudowy
```

Oczekiwane: linter `All checks passed!`, testy `58 passed`, smoke:
`train=1737 (charset-skip: 228, CTC-skip: 35)`, `val=181 (18, 1)`,
`Liczba klas dla CRNN: 84`, zero duplikatów PNG train/val,
diff milczy. Liczby dotyczą aktualnego
datasetu (v1) — po zmianie danych oceń jakościowo, nie po wartościach.
PowerShell 5.x nie obsługuje `&&` — ostatnią komendę rozpisz linia po
linii albo odpal w Git Bash.

### Konserwacja danych/alfabetu (rzadko)

```bash
python -m data.charset            # przebuduj alfabet z data/raw (zapis configs/charset.json)
tensorboard --logdir runs/        # logi treningu
```

Uwaga: przebudowa alfabetu zmienia `configs/charset.json` — po niej
stary `best.pt` (84 klasy) przestaje pasować do modelu wytrenowanego
od zera; nie przebudowuj „na zapas" na pustym `data/raw`.

## Struktura katalogów

```
crnn-ocr/
│
├── README.md                  # opis, setup, status
├── LICENSE                    # pełny tekst licencji MIT
├── environment.yml            # specyfikacja środowiska conda
├── requirements.txt           # zależności pip (to samo co pip w environment.yml)
├── ruff.toml                  # config lintera: select (z E402) + exclude data/raw
├── check_env.py               # weryfikacja zależności (exit 0 = OK, 1 = braki)
├── main.py                    # sanity-check pipeline z configu (check przed treningiem)
├── train.py                   # pętla treningu: CTC + TensorBoard + checkpointy + --resume
├── evaluate.py                # metryki checkpointu -> konsola + metrics.json (źródło prawdy)
├── predict.py                 # inferencja: obraz -> tekst (model+alfabet z best.pt)
├── make_test_line.py          # deterministyczna linia testowa (pangram) -> PNG
├── metrics.json               # raport z evaluate.py (greedy) — artefakt dokumentacyjny (git!)
├── metrics_beam.json          # raport z evaluate.py (beam k=10) — jak wyżej
├── .gitignore
│
├── configs/
│   ├── default.yaml           # hiperparametry + ścieżki (jedno źródło prawdy)
│   ├── charset.json           # zamrożony alfabet: 84 klasy, blank=0 (lista JSON)
│   └── char_stats.txt         # rozkład znaków w labelach (lokalny, ignorowany)
│
├── data/
│   ├── charset.py             # budowa alfabetu: NFC + filtr min_count (blank=0)
│   ├── dataset.py             # LineDataset (filtry: alfabet, CTC) + collate (lens, H=32)
│   ├── download.py            # auto-pobranie datasetu z HF (idempotentne, SHA-256)
│   ├── char_stats.py          # rozkład znaków + gdzie występują rzadkie znaki
│   └── raw/                   # dane: pary {base}.png / {base}.txt (IGNOROWANE przez git)
│
├── model/
│   └── crnn.py                # CRNN: CNN -> sekwencja -> BiLSTM -> FC (T = W/4)
│
├── decoding/
│   ├── greedy.py              # greedy CTC decode (argmax + kolaps powtórek)
│   └── beam.py                # prefix beam search CTC (kontrakt jak greedy.py)
│
├── metrics.py                 # CER / WER / exact match (rapidfuzz) + batch dla train.py
│
└── tests/                     # unit: charset, greedy, beam, collate, crnn, dataset,
                               #       metrics, download, predict, resume, make_test_line;
                               #       smoke: realne dane
```

`data/raw/`, `runs/`, `checkpoints/` — ignorowane przez git (dane /
artefakty).

## Dane

Pary `{base}.png` + `{base}.txt` w `data/raw/train/` i `data/raw/val/`.
Jeśli oba katalogi są puste — `train.py`/`main.py` pobierają dataset
z HF automatycznie (raz, idempotentnie; URL + SHA-256 w configu).

**Kontaminacja train/val** (audyt w `tests/test_smoke.py`): zero
identycznych renderów (PNG) między splitami, ale 138/200 transkrypcji val
występuje też w train — generator losował z puli z powtórzeniami.
Rendery się różnią, więc nie ma wycieku pikseli; overlap daje modelowi
znajomość tekstów waluacyjnych. Kontrolnie: na 62 czystych (niepokrywających
się z train; po filtrach `LineDataset` zostaje z nich 59 — to „n" w tabeli
„Wyniki") próbkach wyniki są *lepsze* niż na całym val (krótkie
boilerplate-y z train są trudniejsze), więc metryki z tabeli nie są
kontaminacją zawyżone. Docelowo: dedup tekstów przed splitem
w [OCR_engine](https://github.com/PiotrStyla/OCR_engine).

## Alfabet

- Transkrypcje: jawne UTF-8 + NFC przed strip/encode (bez podwojonych liter z NFD).
- `build_charset(min_count=25)`: rzadsze znaki (cyrylica, katakana itp.)
  wypadają; lista z licznikami wypisywana przy budowie (`cs.dropped_chars`).
- `LineDataset` pomija próbki: 1) znak spoza alfabetu (`skipped_charset`),
  2) labelka dłuższa niż W/4 (`skipped_ctc`). GT nietknięty, bez `<unk>`.

## Wyniki

Źródło prawdy: `metrics.json` / `metrics_beam.json` (artefakty
`evaluate.py`; kopiuj z nich, nie z konsoli).

| decoder | n | val CER | val WER | exact match | val loss |
|---------|---|---------|---------|-------------|----------|
| greedy  | 181 | 0.0452 | 0.1996 | 0.5746 | 0.2431 |
| beam (k=10) | 181 | 0.0451 | 0.1926 | 0.5912 | 0.2431 |

Kontrolnie na 59 czystych (bez kontaminacji tekstowej, patrz „Dane"):

| decoder | n | val CER | val WER | exact match |
|---------|---|---------|---------|-------------|
| greedy  | 59 | 0.0308 | 0.1748 | 0.6949 |
| beam (k=10) | 59 | 0.0299 | 0.1695 | 0.6949 |

Checkpoint: `checkpoints/best.pt` (epoch 40). Selekcja best.pt po walidacyjnym
greedy CER; metryki zapisane w checkpoincie zgodne z `metrics.json`
(ten sam dekoder z cięciem po `input_lens`).

## Licencja

Kod: **MIT** — pełny tekst w [LICENSE](LICENSE). Dane nie są częścią repo;
ich licencja i atrybucja — patrz karta
[ocr-pl-lines](https://huggingface.co/datasets/PiotrSty/ocr-pl-lines):
próbki wygenerowane ze zdań Wikipedii dziedziczą CC BY-SA 4.0.

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
