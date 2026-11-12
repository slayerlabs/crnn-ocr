"""Ewaluacja checkpointu na walidacji — raport konsolowy + metrics.json.

metrics.json = jedyne źródło prawdy dla README/model card. Model i alfabet
z checkpointu (jak predict.py); config daje tylko dane i parametry loadera.
Walidacja: evaluate() reużyta z train.py (cięcie po input_lens włącznie).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows
sys.stderr.reconfigure(encoding="utf-8")

from data.dataset import LineDataset, collate  # noqa: E402  (po reconfigure)
from predict import build_model_from_ckpt, charset_from_ckpt, pick_device  # noqa: E402
from train import evaluate, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Metryki checkpointu na walidacji + raport metrics.json"
    )
    ap.add_argument("--config", default="configs/default.yaml",
                    help="YAML: val_dir + parametry loadera (reszta z checkpointu)")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt",
                    help="checkpoint z wagami + itos")
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam-size", type=int, default=10)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--out", default="metrics.json",
                    help="gdzie zapisać raport (JSON)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    if args.decoder == "beam":
        try:
            import decoding.beam  # noqa: F401  — sprawdzamy dostępność z góry
        except ImportError:
            sys.exit("[ERROR] decoder=beam, ale decoding/beam.py nie istnieje "
                     "jeszcze w repo — na razie tylko greedy")
    decoder = args.decoder

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        sys.exit(f"[ERROR] Brak checkpointu: {ckpt_path}"
                 " — najpierw python train.py --config configs/default.yaml")
    device = pick_device(args.device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model_from_ckpt(ckpt, device)   # .eval() w środku
    cs = charset_from_ckpt(ckpt)

    val_ds = LineDataset(dcfg["val_dir"], cs)
    val_loader = DataLoader(
        val_ds, batch_size=dcfg.get("batch_size", 32), shuffle=False,
        num_workers=dcfg.get("num_workers", 0), collate_fn=collate,
    )

    m = evaluate(model, val_loader, cs, device,
                decoder=decoder, beam_size=args.beam_size)

    # raport dla człowieka (maszynowa kopia: --out)
    print(f"Checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch', '?')}"
          f"  best_cer={ckpt.get('best_cer', '?')})")
    print(f"Próbki val: {m['n']}  (decoder={decoder}"
          + (f", k={args.beam_size}" if decoder == "beam" else "") + ")")
    for k in ("loss", "cer", "wer", "exact_match"):
        print(f"  {k:12s} {m[k]:.4f}")

    report = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "n_val": m["n"],
        "decoder": decoder,
        "metrics": {k: round(float(v), 4) for k, v in m.items() if k != "n"},
    }
    Path(args.out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Raport: {args.out}  <- źródło prawdy dla README/model card")


if __name__ == "__main__":
    main()
