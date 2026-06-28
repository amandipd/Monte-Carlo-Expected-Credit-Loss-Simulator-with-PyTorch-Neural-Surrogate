"""Tests for ECLSurrogate model architecture."""
import sys
from pathlib import Path

import torch

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.model import ECLSurrogate


def test_forward_pass_single_sample():
    model = ECLSurrogate()
    x = torch.tensor([[4.0, 3.0, 100.0]], dtype=torch.float32)
    y = model(x)
    assert y.shape == (1, 1)
    assert torch.isfinite(y).all()


def test_forward_pass_batch():
    model = ECLSurrogate()
    x = torch.randn(16, 3)
    y = model(x)
    assert y.shape == (16, 1)


def test_model_architecture_dims():
    model = ECLSurrogate()
    layers = list(model.network.children())

    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].in_features == 3
    assert layers[0].out_features == 64

    assert isinstance(layers[2], torch.nn.Linear)
    assert layers[2].in_features == 64
    assert layers[2].out_features == 32

    assert isinstance(layers[4], torch.nn.Linear)
    assert layers[4].in_features == 32
    assert layers[4].out_features == 1
