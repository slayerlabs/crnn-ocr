"""Pętla treningu CRNN (CTC loss), sterowana plikiem YAML.

Uruchomienie:
    python train.py --config configs/default.yaml

TensorBoard:  tensorboard --logdir runs/
Checkpointy:  checkpoints/best.pt (najniższy val CER) + checkpoints/last.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows

from data.charset import Charset  # noqa: E402  (po reconfigure)
from data.dataset import LineDataset, collate  # noqa: E402
from decoding.greedy import greedy_decode  # noqa: E402
from metrics import batch_metrics  # noqa: E402
from model.crnn import CRNN  # noqa: E402


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


def split_targets(y, y_lens, cs):
    """Referencje ze spłaszczonych labeli batcha ((B, sum_len) + lens)."""
    refs, off = [], 0
    for length in y_lens.tolist():
        refs.append(cs.decode(y[off:off + length].tolist()))
        off += length
    return refs


@torch.no_grad()
def evaluate(model, loader, cs, device) -> dict:
    """Metryki (batch_metrics) + loss na całym zbiorze walidacyjnym."""
    model.eval()
    refs_all, hyps_all = [], []
    loss_sum, n_sum = 0.0, 0
    for x, y, input_lens, y_lens in loader:
        x = x.to(device)
        y = y.to(device)
        input_lens, y_lens = input_lens.to(device), y_lens.to(device)

        logits = model(x)                                # (B, T, C)
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)
        loss = F.ctc_loss(
            log_probs, y, input_lens, y_lens, blank=0, zero_infinity=True
        )
        loss_sum += loss.item() * x.size(0)
        n_sum += x.size(0)

        hyps_all.extend(greedy_decode(logits.cpu(), cs))
        refs_all.extend(split_targets(y, y_lens, cs))

    m = batch_metrics(refs_all, hyps_all)
    m["loss"] = loss_sum / max(1, n_sum)
    return m


def main():
    ap = argparse.ArgumentParser(description="Trening CRNN (CTC) z configu")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="plik YAML z hiperparametrami")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, tcfg, mcfg = cfg["data"], cfg["train"], cfg.get("model", {})

    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- dane + model (identycznie jak w main.py)
    cs = Charset.load(dcfg["charset_file"])
    train_ds = LineDataset(dcfg["train_dir"], cs)
    val_ds = LineDataset(dcfg["val_dir"], cs)
    print(f"Config:    {args.config}  (seed={seed})")
    print(f"Alfabet:   {len(cs)} klas (z blank)")
    print(f"Próbki:    train={len(train_ds)}  val={len(val_ds)}")

    device = pick_device(tcfg.get("device", "auto"))
    model = CRNN(
        num_classes=len(cs),
        lstm_hidden=mcfg.get("lstm_hidden", 256),
        lstm_layers=mcfg.get("lstm_layers", 2),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:     CRNN, {n_params / 1e6:.2f} M parametrów, device={device}")

    bs = dcfg.get("batch_size", 32)
    nw = dcfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.get("lr", 1e-3))

    epochs = tcfg.get("epochs", 20)
    log_every = tcfg.get("log_every", 20)
    grad_clip = tcfg.get("grad_clip")          # np. 5.0; brak/nul = bez clipowania
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tcfg.get("log_dir", "runs")) / time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"Artefakty: {ckpt_dir}/  +  TensorBoard: {log_dir}")
    print(f"Optimizer: Adam lr={tcfg.get('lr', 1e-3)}"
          f"  grad_clip={'off' if grad_clip is None else grad_clip}")

    best_cer = float("inf")
    steps_per_epoch = len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y, input_lens, y_lens) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)
            input_lens, y_lens = input_lens.to(device), y_lens.to(device)

            logits = model(x)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            loss = F.ctc_loss(log_probs, y, input_lens, y_lens,
                              blank=0, zero_infinity=True)
            if not torch.isfinite(loss):
                print(f"  [UWAGA] nieskończony loss (epoch {epoch}, step {step})"
                      " — batch pominięty")
                continue

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item()

            if step % log_every == 0 or step == steps_per_epoch:
                mean = running / step
                gstep = (epoch - 1) * steps_per_epoch + step
                print(f"  epoch {epoch}/{epochs}  step {step}/{steps_per_epoch}"
                      f"  train_loss={mean:.3f}")
                writer.add_scalar("train/loss", mean, gstep)

        val = evaluate(model, val_loader, cs, device)
        writer.add_scalar("val/loss", val["loss"], epoch)
        writer.add_scalar("val/cer", val["cer"], epoch)
        writer.add_scalar("val/wer", val["wer"], epoch)
        writer.add_scalar("val/exact_match", val["exact_match"], epoch)
        print(f"Epoch {epoch}/{epochs}  val: loss={val['loss']:.3f}"
              f"  CER={val['cer']:.4f}  WER={val['wer']:.4f}"
              f"  exact_match={val['exact_match']:.4f}")

        ckpt = {"model": model.state_dict(), "itos": cs.itos,
                "epoch": epoch, "val": val}
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val["cer"] < best_cer:
            best_cer = val["cer"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  [OK] nowy najlepszy: val CER={val['cer']:.4f}"
                  f" -> {ckpt_dir / 'best.pt'}")

    writer.close()
    print(f"[OK] Trening zakończony. Najlepszy val CER={best_cer:.4f}"
          f" ({ckpt_dir / 'best.pt'})")


if __name__ == "__main__":
    main()
