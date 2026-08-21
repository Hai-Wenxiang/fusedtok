"""Tests for SwiGLU (naive version).

Reference formula: out[i] = silu(gate[i]) * up[i], silu(v) = v * sigmoid(v).
"""

import math
import random

import pytest

import _fusedtok


def ref_swiglu(gate, up):
    return [g / (1.0 + math.exp(-g)) * u for g, u in zip(gate, up)]


def make_pair(n):
    return ([random.uniform(-6, 6) for _ in range(n)],
            [random.uniform(-3, 3) for _ in range(n)])


def test_hand_checkable():
    # gate = 0 -> silu(0) = 0 regardless of up
    assert _fusedtok.swiglu([0.0], [5.0]) == [0.0]
    # gate large positive -> silu(v) ~ v, so out ~ v * u
    y = _fusedtok.swiglu([100.0], [2.0])
    assert y == pytest.approx([200.0], rel=1e-5)
    # gate large negative -> silu(v) ~ 0
    y = _fusedtok.swiglu([-100.0], [2.0])
    assert abs(y[0]) < 1e-5


def test_matches_reference():
    gate, up = make_pair(1000)
    y = _fusedtok.swiglu(gate, up)
    assert y == pytest.approx(ref_swiglu(gate, up), abs=1e-5)


def test_empty():
    assert _fusedtok.swiglu([], []) == []


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        _fusedtok.swiglu([1.0, 2.0], [1.0])


@pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu(self):
        gate, up = make_pair(10000)
        cpu = _fusedtok.swiglu(gate, up)
        cuda = _fusedtok.swiglu(gate, up, cuda=True)
        assert cuda == pytest.approx(cpu, abs=1e-5)
