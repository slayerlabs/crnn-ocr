"""Testy --resume: restore_training odtwarza wagi, optimizer, epokę, best_cer."""
import torch

from model.crnn import CRNN
from train import restore_training


def _ckpt_after_step(path):
    """Checkpoint po realnym kroku Adam — state optimizer nie jest pusty."""
    torch.manual_seed(0)
    m = CRNN(num_classes=6, lstm_hidden=32, lstm_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.zeros(1, 1, 32, 16)
    x.normal_()
    m(x).log_softmax(-1).sum().backward()
    opt.step()
    ckpt = {"model": m.state_dict(), "optimizer": opt.state_dict(),
            "itos": ["<blank>"] + list("abcde"), "epoch": 7,
            "val": {"cer": 0.1}, "best_cer": 0.09}
    torch.save(ckpt, path)
    return m, opt


def test_restore_training_roundtrip(tmp_path):
    p = tmp_path / "ck.pt"
    m1, opt1 = _ckpt_after_step(p)

    m2 = CRNN(num_classes=6, lstm_hidden=32, lstm_layers=1)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    start_epoch, best_cer = restore_training(str(p), m2, opt2)

    assert start_epoch == 8                     # epoch 7 -> od 8
    assert best_cer == 0.09
    for k in m1.state_dict():
        assert torch.equal(m1.state_dict()[k], m2.state_dict()[k])
    s1, s2 = opt1.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()               # optimizer nie świeży
    for pid in s1:
        for k in s1[pid]:
            assert torch.equal(s1[pid][k], s2[pid][k])


def test_restore_training_legacy_ckpt_without_opt_and_best(tmp_path):
    """Stary format checkpointu — resume działa, zamiast wybuchać."""
    m = CRNN(num_classes=4, lstm_hidden=16, lstm_layers=1)
    ckpt = {"model": m.state_dict(), "itos": ["<blank>", "a", "b", "c"],
            "epoch": 3, "val": {"cer": 0.2}}
    p = tmp_path / "old.pt"
    torch.save(ckpt, p)

    m2 = CRNN(num_classes=4, lstm_hidden=16, lstm_layers=1)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    start_epoch, best_cer = restore_training(str(p), m2, opt)

    assert start_epoch == 4
    assert best_cer == float("inf")
    assert opt.state_dict()["state"] == {}      # optimizer od zera
