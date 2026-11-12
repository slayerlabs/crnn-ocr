"""Testy predict.py — architektura z kształtów wag, preprocess = LineDataset."""
import torch
from PIL import Image

from data.charset import Charset
from data.dataset import LineDataset
from model.crnn import CRNN
from predict import build_model_from_ckpt, charset_from_ckpt, preprocess


def test_build_model_roundtrip():
    m = CRNN(num_classes=6, lstm_hidden=32, lstm_layers=1)
    ckpt = {"model": m.state_dict(), "itos": ["<blank>"] + list("abcde")}
    m2 = build_model_from_ckpt(ckpt, torch.device("cpu"))
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(),
                                  m2.state_dict().items(), strict=True):
        assert k1 == k2
        assert v1.shape == v2.shape
        assert torch.equal(v1, v2)         # wagi naprawdę wgrane, nie defaulty


def test_build_model_infers_hidden_and_layers():
    for hidden, layers in ((64, 1), (128, 2), (256, 3)):
        m = CRNN(num_classes=5, lstm_hidden=hidden, lstm_layers=layers)
        ckpt = {"model": m.state_dict(), "itos": ["<blank>", "a", "b", "c", "d"]}
        m2 = build_model_from_ckpt(ckpt, torch.device("cpu"))
        assert m2.lstm.hidden_size == hidden
        assert m2.lstm.num_layers == layers


def test_charset_from_ckpt_decodes():
    cs = charset_from_ckpt({"itos": ["<blank>", "a", "b"]})
    assert len(cs) == 3
    assert cs.decode([1, 2, 1]) == "aba"
    assert cs.encode("ba") == [2, 1]


def test_preprocess_matches_linedataset(tmp_path):
    """Identyczny plik przez preprocess() i LineDataset -> identyczny tensor."""
    Image.new("L", (100, 40), 128).save(tmp_path / "00000.png")
    (tmp_path / "00000.txt").write_text("ab", encoding="utf-8")

    x = preprocess(str(tmp_path / "00000.png"))
    assert x.shape == (1, 1, 32, 80)          # w = round(100 * 32/40) = 80

    ds = LineDataset(str(tmp_path), Charset("ab"))
    x_ds, _ = ds[0]
    assert x_ds.shape == (1, 32, 80)
    assert torch.equal(x[0], x_ds)             # co do piksela


def test_preprocess_tall_narrow_image_assumes_width(tmp_path):
    """Wysoki wąski obraz: w = max(1, ...) pilnuje degeneracji."""
    Image.new("L", (1, 400), 200).save(tmp_path / "thin.png")
    x = preprocess(str(tmp_path / "thin.png"))
    assert x.shape[-1] >= 1
