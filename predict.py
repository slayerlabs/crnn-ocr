"""Predykcja OCR na pojedynczym obrazie linii — wystarczy sam checkpoint.

Uruchomienie:
    python predict.py --image linia.png [--decoder beam]

Architektura i alfabet odtwarzane z best.pt (kształty wag + itos) — bez
configu i bez charset.json. Preprocess identyczny z LineDataset (H=32);
przy B=1 nie ma paddingu, dekoder dostaje pełne T kroków.
"""
import argparse
import re
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows
sys.stderr.reconfigure(encoding="utf-8")   # sys.exit() pisze na stderr

from data.charset import Charset  # noqa: E402  (po reconfigure)
from data.dataset import TARGET_H  # noqa: E402
from decoding.greedy import greedy_decode  # noqa: E402
from model.crnn import CRNN  # noqa: E402


def pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            sys.exit("[ERROR] device: cuda, ale CUDA niedostępne")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_from_ckpt(ckpt, device) -> CRNN:
    """Architektura z kształtów wag (bez configu):
    weight_ih_l0 ma (4*hidden, in), num_layers z sufiksów _lN, klasy z fc.
    """
    sd = ckpt["model"]
    lstm_hidden = sd["lstm.weight_ih_l0"].shape[0] // 4
    n_layers = 1 + max(
        int(m.group(1))
        for k in sd
        if (m := re.fullmatch(r"lstm\.weight_ih_l(\d+)", k))
    )
    model = CRNN(num_classes=sd["fc.weight"].shape[0],
                 lstm_hidden=int(lstm_hidden), lstm_layers=n_layers)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def charset_from_ckpt(ckpt) -> Charset:
    """Alfabet z itos zapisanego w checkpointcie (blank=0)."""
    cs = Charset("")
    cs.itos = ckpt["itos"]
    cs.stoi = {c: i for i, c in enumerate(cs.itos)}
    return cs


def preprocess(path: str) -> torch.Tensor:
    """Obraz -> (1, 1, 32, W), dokładnie jak LineDataset.__getitem__."""
    img = Image.open(path).convert("L")
    w = max(1, round(img.width * TARGET_H / img.height))
    img = img.resize((w, TARGET_H))
    return TF.to_tensor(img).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(
        description="OCR jednej linii tekstu z checkpointu (best.pt)"
    )
    ap.add_argument("--image", required=True, help="obraz linii (PNG)")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt",
                    help="checkpoint z wagami + itos")
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beam"],
                    help="greedy | beam (beam wymaga decoding/beam.py)")
    ap.add_argument("--beam-size", type=int, default=10,
                    help="szerokość wiązki (tylko decoder=beam)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        sys.exit(f"[ERROR] Brak checkpointu: {ckpt_path}"
                 " — najpierw python train.py --config configs/default.yaml")
    device = pick_device(args.device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model_from_ckpt(ckpt, device)
    cs = charset_from_ckpt(ckpt)
    assert model.fc.out_features == len(cs.itos), (
        "checkpoint niespójny: liczba klas w wagach ≠ len(itos)")

    x = preprocess(args.image)
    T = x.shape[3] // 4
    if T < 1:
        sys.exit("[ERROR] Obraz za wąski — T = W/4 musi być >= 1 (W >= 4 px)")

    with torch.no_grad():
        logits = model(x.to(device))               # (1, T, C)
    logits = logits.cpu()

    if args.decoder == "beam":
        try:
            from decoding.beam import beam_decode
        except ImportError:
            sys.exit("[ERROR] decoder=beam, ale decoding/beam.py nie istnieje "
                     "jeszcze w repo — na razie tylko greedy")
        text = beam_decode(logits, cs, beam_size=args.beam_size)[0]
    else:
        text = greedy_decode(logits, cs)[0]

    val = ckpt.get("val") or {}
    cer = f"  val CER={val['cer']:.4f}" if val else ""
    print(f"Checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch', '?')}{cer})")
    print(f"Obraz:      {args.image}  (W={x.shape[3]} -> T={T})")
    print(f"Dekoder:    {args.decoder}"
          + (f" (k={args.beam_size})" if args.decoder == "beam" else ""))
    print(f"Tekst:      {text}")


if __name__ == "__main__":
    main()
