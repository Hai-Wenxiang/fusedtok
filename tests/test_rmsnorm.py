"""Tests for RMSNorm (naive version).

Reference formula (computed independently in Python):
    y = (x + r) * rsqrt(mean((x + r)^2) + eps) * w
"""

import math
import random

import pytest

import _fusedtok


def ref_rmsnorm(x, w, rows, cols, eps, residual=None):
    """Pure-Python reference implementation."""
    y = []
    for row in range(rows):
        vals = [x[row * cols + i] + (residual[row * cols + i] if residual else 0.0)
                for i in range(cols)]
        ms = sum(v * v for v in vals) / cols
        inv = 1.0 / math.sqrt(ms + eps)
        y.extend(v * inv * w[i] for i, v in enumerate(vals))
    return y


def make_case(rows, cols):
    x = [random.uniform(-3, 3) for _ in range(rows * cols)]
    w = [random.uniform(0.5, 1.5) for _ in range(cols)]
    return x, w


def test_hand_checkable():
    # One row [3, 4], unit weight, eps ~ 0:
    #   rms = sqrt((9 + 16) / 2) = sqrt(12.5), y = x / rms
    y = _fusedtok.rmsnorm([3.0, 4.0], [1.0, 1.0], 1, 2, 1e-12)
    inv = 1.0 / math.sqrt(12.5)
    assert y == pytest.approx([3.0 * inv, 4.0 * inv], abs=1e-5)


def test_matches_reference():
    x, w = make_case(4, 65)          # odd cols on purpose
    y = _fusedtok.rmsnorm(x, w, 4, 65, 1e-6)
    assert y == pytest.approx(ref_rmsnorm(x, w, 4, 65, 1e-6), abs=1e-5)


def test_residual_equals_shifted_input():
    # rmsnorm(x, r) must equal rmsnorm(x + r) without residual
    x, w = make_case(3, 32)
    r = [random.uniform(-1, 1) for _ in range(96)]
    shifted = [x[i] + r[i] for i in range(96)]
    with_r = _fusedtok.rmsnorm(x, w, 3, 32, 1e-6, residual=r)
    plain = _fusedtok.rmsnorm(shifted, w, 3, 32, 1e-6)
    assert with_r == pytest.approx(plain, abs=1e-5)


def test_empty():
    assert _fusedtok.rmsnorm([], [1.0], 0, 1, 1e-6) == []


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _fusedtok.rmsnorm([1.0, 2.0, 3.0], [1.0, 1.0], 1, 2, 1e-6)  # x.size != rows*cols
    with pytest.raises(ValueError):
        _fusedtok.rmsnorm([1.0, 2.0], [1.0], 1, 2, 1e-6)             # w.size != cols


@pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu(self):
        x, w = make_case(8, 257)
        r = [random.uniform(-1, 1) for _ in range(8 * 257)]
        cpu = _fusedtok.rmsnorm(x, w, 8, 257, 1e-6, residual=r)
        cuda = _fusedtok.rmsnorm(x, w, 8, 257, 1e-6, residual=r, cuda=True)
        assert cuda == pytest.approx(cpu, abs=1e-5)

    def test_single_col(self):
        # cols = 1 edge case: every row normalizes to +-w[0]
        y = _fusedtok.rmsnorm([2.0, -3.0], [1.5], 2, 1, 1e-6, cuda=True)
        assert y == pytest.approx([1.5, -1.5], abs=1e-5)
