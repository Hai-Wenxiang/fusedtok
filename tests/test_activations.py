"""Tests for elementwise activations: SiLU and GeLU (naive versions).

References (computed independently in Python):
    silu(v) = v * sigmoid(v)
    gelu(v) = 0.5 * v * (1 + erf(v / sqrt(2)))   (exact erf form)
"""

import math
import random

import pytest

import _fusedtok


def sigmoid(v):
    # Numerically stable pure-Python sigmoid
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def ref_silu(x):
    return [v * sigmoid(v) for v in x]


def ref_gelu(x):
    return [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x]


def make_input(n):
    return [random.uniform(-8, 8) for _ in range(n)]


class TestSilu:
    def test_hand_checkable(self):
        assert _fusedtok.silu([0.0]) == [0.0]
        # Large positive -> ~identity; large negative -> ~0
        assert _fusedtok.silu([50.0])[0] == pytest.approx(50.0, rel=1e-4)
        assert abs(_fusedtok.silu([-50.0])[0]) < 1e-5

    def test_matches_reference(self):
        x = make_input(2000)
        assert _fusedtok.silu(x) == pytest.approx(ref_silu(x), abs=1e-5)

    def test_empty(self):
        assert _fusedtok.silu([]) == []

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(10000)
        assert _fusedtok.silu(x, cuda=True) == pytest.approx(_fusedtok.silu(x), abs=1e-5)


class TestGelu:
    def test_hand_checkable(self):
        assert _fusedtok.gelu([0.0]) == [0.0]
        # Large positive -> ~identity; large negative -> ~0
        assert _fusedtok.gelu([50.0])[0] == pytest.approx(50.0, rel=1e-4)
        assert abs(_fusedtok.gelu([-50.0])[0]) < 1e-5

    def test_matches_reference(self):
        x = make_input(2000)
        assert _fusedtok.gelu(x) == pytest.approx(ref_gelu(x), abs=1e-5)

    def test_empty(self):
        assert _fusedtok.gelu([]) == []

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(10000)
        assert _fusedtok.gelu(x, cuda=True) == pytest.approx(_fusedtok.gelu(x), abs=1e-5)
