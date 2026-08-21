"""Tests for ReLU, Tanh and top-k selection (naive versions).

References (computed independently in Python):
    relu(v) = max(v, 0)
    tanh(v) = math.tanh(v)
    topk    = sorted(values, descending) with deterministic earliest-index ties
"""

import math
import random

import pytest

import _fusedtok


def make_input(n, lo=-5, hi=5):
    return [random.uniform(lo, hi) for _ in range(n)]


class TestRelu:
    def test_hand_checkable(self):
        assert _fusedtok.relu([-2.0, 0.0, 3.5]) == [0.0, 0.0, 3.5]

    def test_matches_reference(self):
        # approx, not ==: inputs are float64 in Python but float32 in C++
        x = make_input(2000)
        assert _fusedtok.relu(x) == pytest.approx([max(v, 0.0) for v in x], rel=1e-6)

    def test_empty(self):
        assert _fusedtok.relu([]) == []

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(10000)
        assert _fusedtok.relu(x, cuda=True) == _fusedtok.relu(x)


class TestTanh:
    def test_hand_checkable(self):
        y = _fusedtok.tanh([0.0, 100.0, -100.0])
        assert y[0] == 0.0
        assert y[1] == pytest.approx(1.0)
        assert y[2] == pytest.approx(-1.0)

    def test_matches_reference(self):
        x = make_input(2000)
        assert _fusedtok.tanh(x) == pytest.approx([math.tanh(v) for v in x], abs=1e-6)

    def test_empty(self):
        assert _fusedtok.tanh([]) == []

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(10000)
        assert _fusedtok.tanh(x, cuda=True) == pytest.approx(_fusedtok.tanh(x), abs=1e-6)


class TestTopK:
    def test_hand_checkable(self):
        vals, idxs = _fusedtok.topk([1.0, 5.0, 3.0, 2.0], 2)
        assert vals == [5.0, 3.0]
        assert idxs == [1, 2]

    def test_matches_sorted_reference(self):
        # distinct random values: compare against sorted() descending
        x = make_input(100)
        vals, idxs = _fusedtok.topk(x, 7)
        expect = sorted(x, reverse=True)[:7]
        assert vals == pytest.approx(expect, abs=1e-6)
        assert [x[i] for i in idxs] == pytest.approx(expect, abs=1e-6)

    def test_ties_deterministic_earliest_index(self):
        vals, idxs = _fusedtok.topk([7.0, 7.0, 7.0, 1.0], 3)
        assert vals == [7.0, 7.0, 7.0]
        assert idxs == [0, 1, 2]

    def test_k_equals_n(self):
        x = make_input(10)
        vals, _ = _fusedtok.topk(x, 10)
        assert vals == pytest.approx(sorted(x, reverse=True))

    def test_k_zero(self):
        vals, idxs = _fusedtok.topk([1.0, 2.0], 0)
        assert vals == [] and idxs == []

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            _fusedtok.topk([1.0, 2.0], 3)

    @pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
    def test_cuda_matches_cpu(self):
        x = make_input(300)
        vcpu, icpu = _fusedtok.topk(x, 5)
        vgpu, igpu = _fusedtok.topk(x, 5, cuda=True)
        assert vgpu == pytest.approx(vcpu)
        assert igpu == icpu
