"""Punkt wejścia pipeline'u, sterowany plikiem YAML.

Uruchomienie:
    python main.py --config configs/default.yaml

Na razie: pełny sanity-check pipeline'u na jednym batchu —
config -> charset -> dataset -> CRNN -> CTC loss -> greedy decode.
Docelowo: pętla treningu (train.py) czytająca ten sam plik configu.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.charset import Charset
from data.dataset import LineDataset, collate
from decoding.greedy import greedy_decode
from metrics import batch_metrics
from model.crnn import CRNN

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            sys.exit("[ERROR] device: cuda w configu, ale CUDA niedostępne")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser(
        description="CRNN OCR — sanity-check pipeline'u z configu"
    )
    ap.add_argument(
        "--config", default="configs/default.yaml", help="plik YAML z hiperparametrami"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config:   {args.config}")

    # --- charset + dane
    charset_path = cfg["data"]["charset_file"]
    cs = Charset.load(charset_path)
    print(f"Alfabet:  {len(cs)} klas (z blank), plik {charset_path}")

    for key in ("train_dir", "val_dir"):
        p = Path(cfg["data"][key])
        if not p.is_dir():
            sys.exit(f"[ERROR] Brak katalogu {p} — patrz README, sekcja Dane")

    train_ds = LineDataset(cfg["data"]["train_dir"], cs)
    val_ds = LineDataset(cfg["data"]["val_dir"], cs)
    print(
        f"Próbki:   train={len(train_ds)} "
        f"(charset-skip: {train_ds.skipped_charset}, "
        f"CTC-skip: {train_ds.skipped_ctc})  "
        f"val={len(val_ds)} "
        f"(charset-skip: {val_ds.skipped_charset}, "
        f"CTC-skip: {val_ds.skipped_ctc})"
    )

    # --- model
    mcfg = cfg.get("model", {})
    device = pick_device(cfg["train"].get("device", "auto"))
    model = CRNN(
        num_classes=len(cs),
        lstm_hidden=mcfg.get("lstm_hidden", 256),
        lstm_layers=mcfg.get("lstm_layers", 2),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:    CRNN, {n_params / 1e6:.2f} M parametrów, device={device}")

    # --- jeden batch przez cały pipeline
    loader = DataLoader(
        train_ds,
        batch_size=cfg["data"].get("batch_size", 32),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        collate_fn=collate,
    )
    x, y, input_lens, y_lens = next(iter(loader))
    x = x.to(device)
    y, input_lens, y_lens = y.to(device), input_lens.to(device), y_lens.to(device)

    logits = model(x)                       # (B, T, C), T = W/4
    print(f"Batch:    x {tuple(x.shape)}  ->  logits {tuple(logits.shape)}  (T = W/4)")

    # sanity: kontrakt CTC — labelka musi mieścić się w sekwencji czasowej
    assert (y_lens <= input_lens).all(), "y_len > input_len — kontrakt CTC złamany"

    log_probs = logits.log_softmax(-1).permute(1, 0, 2)   # (T, B, C)
    loss = F.ctc_loss(
        log_probs, y, input_lens, y_lens,
        blank=0, zero_infinity=True,
    )
    print(f"CTC loss: {loss.item():.3f}  (model nietrenowany — wartość nic nie znaczy)")
    assert torch.isfinite(loss), "CTC loss nie jest skończone (inf/NaN)"

    preds = greedy_decode(logits.cpu(), cs)
    print("Predykcje bez treningu (pierwsze 3):")
    for i, p in enumerate(preds[:3]):
        print(f"  [{i}] {p!r}")

    # metryki batcha (model nietrenowany — wartości bez znaczenia,
    # sprawdza spójność całej ścieżki: encode -> model -> decode -> metryki)
    refs, off = [], 0
    for length in y_lens.tolist():
        refs.append(cs.decode(y[off:off + length].tolist()))
        off += length
    m = batch_metrics(refs, preds)
    print(
        f"Metryki batcha: CER={m['cer']:.3f}  WER={m['wer']:.3f}  "
        f"exact_match={m['exact_match']:.3f}  (n={m['n']})"
    )

    print(
        "[OK] Pipeline działa end-to-end "
        "(config -> dane -> model -> CTC -> dekoder -> metryki)"
    )


if __name__ == "__main__":
    main()
