"""Tests for sampling helpers (argmax, temperature) and the NeoX-layout RoPE.

References (computed independently in Python):
    argmax      : earliest index of the maximum
    temperature : y = x / t
    rope_neox   : halves paired, angle(m, j) = m * theta ** (-2j / dim)
"""

import math
import random

import pytest

import _fusedtok


def make_input(n, lo=-5, hi=5):
    return [random.uniform(lo, hi) for _ in range(n)]


class TestArgmax:
    def test_hand_checkable(self):
        assert _fusedtok.argmax([1.0, 3.0, 2.0]) == 1
        # tie -> earliest index
        assert _fusedtok.argmax([5.0, 5.0, 1.0]) == 0

    def test_matches_python(self):
        x = make_input(500)
        assert _fusedtok.argmax(x) == x.index(max(x))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _fusedtok.argmax([])

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(1000)
        assert _fusedtok.argmax(x, cuda=True) == _fusedtok.argmax(x)


class TestTemperature:
    def test_hand_checkable(self):
        assert _fusedtok.temperature([2.0, -4.0], 2.0) == pytest.approx([1.0, -2.0])

    def test_matches_reference(self):
        x = make_input(100)
        t = 0.7
        assert _fusedtok.temperature(x, t) == pytest.approx([v / t for v in x], rel=1e-6)

    def test_empty(self):
        assert _fusedtok.temperature([], 1.0) == []

    def test_invalid_t_raises(self):
        with pytest.raises(ValueError):
            _fusedtok.temperature([1.0], 0.0)

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(1000)
        assert _fusedtok.temperature(x, 0.8, cuda=True) == pytest.approx(
            _fusedtok.temperature(x, 0.8), rel=1e-6)


def ref_rope_neox(x, seq, dim, theta):
    y = [0.0] * len(x)
    half = dim // 2
    for m in range(seq):
        for j in range(half):
            angle = m * theta ** (-2.0 * j / dim)
            c, s = math.cos(angle), math.sin(angle)
            i1 = m * dim + j
            i2 = i1 + half
            y[i1] = x[i1] * c - x[i2] * s
            y[i2] = x[i1] * s + x[i2] * c
    return y


class TestRopeNeoX:
    def test_position_zero_is_identity(self):
        q = make_input(8)
        out, k = _fusedtok.rope_neox(q, None, 1, 8, 10000.0)
        assert out == pytest.approx(q, abs=1e-6)
        assert k is None

    def test_matches_reference(self):
        q = make_input(5 * 64)
        out, _ = _fusedtok.rope_neox(q, None, 5, 64, 10000.0)
        assert out == pytest.approx(ref_rope_neox(q, 5, 64, 10000.0), abs=1e-4)

    def test_norm_preserved(self):
        q = make_input(4 * 32)
        out, _ = _fusedtok.rope_neox(q, None, 4, 32, 10000.0)
        for m in range(4):
            n_in = math.sqrt(sum(v * v for v in q[m * 32:(m + 1) * 32]))
            n_out = math.sqrt(sum(v * v for v in out[m * 32:(m + 1) * 32]))
            assert n_out == pytest.approx(n_in, rel=1e-4)

    def test_is_permutation_of_interleaved(self):
        # The two layouts assign physical positions to logical channels
        # differently: interleaved uses (2j, 2j+1), NeoX uses (j, half+j).
        # Applying the same position permutation to BOTH input and output
        # must make the two variants agree.
        q = make_input(3 * 8)
        half = 4
        perm = [0] * 8
        for j in range(half):
            perm[j] = 2 * j          # neox pos j <- interleaved pos 2j
            perm[half + j] = 2 * j + 1

        inter, _ = _fusedtok.rope(q, None, 3, 8, 10000.0)
        q_perm = [q[m * 8 + perm[i]] for m in range(3) for i in range(8)]
        neox, _ = _fusedtok.rope_neox(q_perm, None, 3, 8, 10000.0)
        inter_perm = [[inter[m * 8 + perm[i]] for i in range(8)] for m in range(3)]
        for m in range(3):
            assert neox[m * 8:(m + 1) * 8] == pytest.approx(inter_perm[m], abs=1e-4)

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        q = make_input(16 * 128)
        qo_c, _ = _fusedtok.rope_neox(q, None, 16, 128, 10000.0)
        qo_g, _ = _fusedtok.rope_neox(q, None, 16, 128, 10000.0, cuda=True)
        assert qo_g == pytest.approx(qo_c, abs=1e-4)
