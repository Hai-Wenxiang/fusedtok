"""Tests for LayerNorm (naive version).

Reference (computed independently in Python):
    mean = sum(row) / cols
    var  = sum((x - mean)^2) / cols        (biased / population variance)
    y    = (x - mean) / sqrt(var + eps) * w + b
"""

import math
import random

import pytest

import _fusedtok


def ref_layernorm(x, w, b, rows, cols, eps):
    y = []
    for r in range(rows):
        row = x[r * cols:(r + 1) * cols]
        mean = sum(row) / cols
        var = sum((v - mean) ** 2 for v in row) / cols
        inv = 1.0 / math.sqrt(var + eps)
        y.extend((v - mean) * inv * w[i] + b[i] for i, v in enumerate(row))
    return y


def make_case(rows, cols):
    x = [random.uniform(-4, 4) for _ in range(rows * cols)]
    w = [random.uniform(0.5, 1.5) for _ in range(cols)]
    b = [random.uniform(-0.5, 0.5) for _ in range(cols)]
    return x, w, b


def test_hand_checkable():
    # x = [1, 2, 3]: mean = 2, var = 2/3, unit weight, zero bias, eps -> 0
    # y = [-1, 0, 1] / sqrt(2/3)
    y = _fusedtok.layernorm([1.0, 2.0, 3.0], [1.0] * 3, [0.0] * 3, 1, 3, 1e-12)
    s = math.sqrt(2.0 / 3.0)
    assert y == pytest.approx([-1.0 / s, 0.0, 1.0 / s], abs=1e-5)


def test_matches_reference():
    x, w, b = make_case(6, 65)
    y = _fusedtok.layernorm(x, w, b, 6, 65, 1e-5)
    assert y == pytest.approx(ref_layernorm(x, w, b, 6, 65, 1e-5), abs=1e-4)


def test_constant_row_only_bias():
    # Constant row: variance 0 -> y = 0 * w + b = b (modulo eps)
    b = [0.1, -0.2, 0.3]
    y = _fusedtok.layernorm([5.0] * 3, [2.0] * 3, b, 1, 3, 1e-5)
    assert y == pytest.approx(b, abs=1e-3)


def test_empty():
    assert _fusedtok.layernorm([], [1.0], [0.0], 0, 1, 1e-5) == []


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _fusedtok.layernorm([1.0] * 5, [1.0] * 3, [0.0] * 3, 1, 3, 1e-5)
    with pytest.raises(ValueError):
        _fusedtok.layernorm([1.0] * 3, [1.0] * 2, [0.0] * 3, 1, 3, 1e-5)


@pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu(self):
        x, w, b = make_case(8, 257)
        cpu = _fusedtok.layernorm(x, w, b, 8, 257, 1e-5)
        cuda = _fusedtok.layernorm(x, w, b, 8, 257, 1e-5, cuda=True)
        assert cuda == pytest.approx(cpu, abs=1e-4)
