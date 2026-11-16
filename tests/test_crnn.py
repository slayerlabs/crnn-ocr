"""Testy jednostkowe CRNN: kontrakt kształtów (T = W/4) i przepływ gradientów."""
import torch

from model.crnn import CRNN


def test_shape_contract_odd_widths():
    m = CRNN(num_classes=143)
    m.eval()
    for W in (33, 100, 777):
        with torch.no_grad():
            y = m(torch.randn(2, 1, 32, W))
        assert y.shape == (2, W // 4, 143), f"W={W}: {y.shape}"


def test_gradient_flows_through_model():
    m = CRNN(num_classes=20)
    m(torch.randn(1, 1, 32, 64)).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
