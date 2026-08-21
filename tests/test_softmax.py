"""Tests for row-wise Softmax (naive version).

Reference (computed independently in Python):
    y = exp(x - max(row)) / sum(exp(x - max(row)))
"""

import math
import random

import pytest

import _fusedtok


def ref_softmax(x, rows, cols):
    y = []
    for r in range(rows):
        row = x[r * cols:(r + 1) * cols]
        m = max(row)
        es = [math.exp(v - m) for v in row]
        s = sum(es)
        y.extend(e / s for e in es)
    return y


def make_input(rows, cols):
    return [random.uniform(-10, 10) for _ in range(rows * cols)]


def test_hand_checkable():
    # Uniform row -> uniform distribution
    y = _fusedtok.softmax([2.0, 2.0, 2.0, 2.0], 1, 4)
    assert y == pytest.approx([0.25] * 4, abs=1e-6)


def test_rows_sum_to_one():
    x = make_input(8, 33)
    y = _fusedtok.softmax(x, 8, 33)
    for r in range(8):
        s = sum(y[r * 33:(r + 1) * 33])
        assert s == pytest.approx(1.0, abs=1e-5)


def test_matches_reference():
    x = make_input(5, 64)
    assert _fusedtok.softmax(x, 5, 64) == pytest.approx(ref_softmax(x, 5, 64), abs=1e-5)


def test_numerical_stability_large_values():
    # Without max-subtraction, exp(1000) would overflow to inf/nan
    x = [1000.0, 1000.0, 999.0]
    y = _fusedtok.softmax(x, 1, 3)
    assert all(math.isfinite(v) for v in y)
    assert y[0] == pytest.approx(y[1], rel=1e-6)
    assert y[2] == pytest.approx(y[0] * math.exp(-1.0), rel=1e-5)


def test_empty():
    assert _fusedtok.softmax([], 0, 7) == []


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _fusedtok.softmax([1.0] * 5, 2, 3)


@pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu(self):
        x = make_input(16, 129)
        cpu = _fusedtok.softmax(x, 16, 129)
        cuda = _fusedtok.softmax(x, 16, 129, cuda=True)
        assert cuda == pytest.approx(cpu, abs=1e-5)
