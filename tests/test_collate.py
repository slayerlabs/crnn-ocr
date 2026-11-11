"""Testy jednostkowe collate: padding zerami + prawdziwe długości per próbka."""
import torch

from data.dataset import collate


def test_padding_to_max_width():
    x1, x2 = torch.randn(1, 32, 40), torch.randn(1, 32, 55)
    y1, y2 = torch.tensor([1, 2, 3]), torch.tensor([4, 5])
    xs, ys, input_lens, y_lens = collate([(x1, y1), (x2, y2)])

    assert xs.shape == (2, 1, 32, 55)
    assert (xs[0, :, :, 40:] == 0).all()      # krótsza: zera na końcu
    assert (xs[1] == x2).all()                # najszersza: bez zmian


def test_lens_computed_before_padding():
    x1, x2 = torch.randn(1, 32, 40), torch.randn(1, 32, 55)
    y1, y2 = torch.tensor([1, 2, 3]), torch.tensor([4, 5])
    _, ys, input_lens, y_lens = collate([(x1, y1), (x2, y2)])

    assert input_lens.tolist() == [10, 13]               # w // 4, nie pad_W // 4
    assert y_lens.tolist() == [3, 2]
    assert ys.tolist() == [1, 2, 3, 4, 5]                 # konkatenacja w kolejności
