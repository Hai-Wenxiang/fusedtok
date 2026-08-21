import random

import pytest

import _fusedtok


def test_cpu_exact():
    assert _fusedtok.axpy([1.0, 2.0, 3.0], 2.0, 1.0) == [3.0, 5.0, 7.0]


def test_cpu_empty():
    assert _fusedtok.axpy([], 2.0, 1.0) == []


@pytest.mark.skipif(not _fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    def test_matches_cpu(self):
        x = [random.uniform(-10, 10) for _ in range(10000)]
        cpu = _fusedtok.axpy(x, 1.5, -2.0)
        cuda = _fusedtok.axpy(x, 1.5, -2.0, cuda=True)
        assert cuda == pytest.approx(cpu, abs=1e-5)

    def test_empty(self):
        assert _fusedtok.axpy([], 1.0, 0.0, cuda=True) == []
