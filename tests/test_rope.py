"""Tests for RoPE, interleaved-pair variant (naive version).

Reference (computed independently in Python):
    angle(m, j) = m * theta ** (-2j / dim)
    x'[2j]   = x[2j] * cos - x[2j+1] * sin
    x'[2j+1] = x[2j] * sin + x[2j+1] * cos
"""

import math
import random

import pytest

import _fusedtok


def ref_rope(x, seq, dim, theta):
    y = [0.0] * len(x)
    for m in range(seq):
        for j in range(dim // 2):
            angle = m * theta ** (-2.0 * j / dim)
            c, s = math.cos(angle), math.sin(angle)
            e = m * dim + 2 * j
            o = e + 1
            y[e] = x[e] * c - x[o] * s
            y[o] = x[e] * s + x[o] * c
    return y


def make_q(seq, dim):
    return [random.uniform(-2, 2) for _ in range(seq * dim)]


def test_position_zero_is_identity():
    # angle = 0 for every pair -> output equals input
    q = make_q(1, 8)
    out, k = _fusedtok.rope(q, None, 1, 8, 10000.0)
    assert out == pytest.approx(q, abs=1e-6)
    assert k is None


def test_matches_reference():
    q = make_q(5, 64)
    out, _ = _fusedtok.rope(q, None, 5, 64, 10000.0)
    assert out == pytest.approx(ref_rope(q, 5, 64, 10000.0), abs=1e-4)


def test_q_and_k_both_rotated():
    q, k = make_q(3, 16), make_q(3, 16)
    qo, ko = _fusedtok.rope(q, k, 3, 16, 10000.0)
    assert qo == pytest.approx(ref_rope(q, 3, 16, 10000.0), abs=1e-4)
    assert ko == pytest.approx(ref_rope(k, 3, 16, 10000.0), abs=1e-4)


def test_norm_preserved():
    # RoPE is a rotation per pair: vector norm per row must be preserved
    q = make_q(4, 32)
    out, _ = _fusedtok.rope(q, None, 4, 32, 10000.0)
    for m in range(4):
        n_in = math.sqrt(sum(v * v for v in q[m * 32:(m + 1) * 32]))
        n_out = math.sqrt(sum(v * v for v in out[m * 32:(m + 1) * 32]))
        assert n_out == pytest.approx(n_in, rel=1e-4)


def test_empty():
    out, k = _fusedtok.rope([], None, 0, 4, 10000.0)
    assert out == [] and k is None


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        _fusedtok.rope([1.0, 2.0, 3.0], None, 1, 3, 10000.0)   # dim odd
    with pytest.raises(ValueError):
        _fusedtok.rope([1.0] * 5, None, 2, 4, 10000.0)          # seq*dim mismatch


@pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu(self):
        q, k = make_q(16, 128), make_q(16, 128)
        qo_c, ko_c = _fusedtok.rope(q, k, 16, 128, 10000.0)
        qo_g, ko_g = _fusedtok.rope(q, k, 16, 128, 10000.0, cuda=True)
        assert qo_g == pytest.approx(qo_c, abs=1e-4)
        assert ko_g == pytest.approx(ko_c, abs=1e-4)
