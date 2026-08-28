"""Fused decode_step: penalty -> temperature -> nucleus sample in one call.

Parity contract: for the same seed, decode_step returns the SAME token
as the composed reference (repetition_penalty -> temperature ->
sample_topp), on CPU and on every GPU path. The GPU applies the penalty
to the raw logit and then the temperature scale - the exact composed
order - so the sampled CDFs agree up to float rounding away from exact
mass-boundary draws.
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False


def composed(logits, ids, penalty, p, t, seed):
    pen = fusedtok.repetition_penalty(logits, ids, penalty)
    return fusedtok.sample_topp(pen, p, temperature=t, seed=seed)


@pytest.mark.parametrize("penalty", [1.0, 1.3, 0.7])
@pytest.mark.parametrize("seed", [0, 7, 123])
def test_cpu_matches_composed(penalty, seed):
    rng = np.random.default_rng(31)
    logits = (rng.standard_normal(900) * 2).astype(np.float32)
    ids = rng.integers(0, 900, size=40).tolist()
    assert fusedtok.decode_step(logits, ids, penalty, p=0.9,
                                temperature=0.8, seed=seed) == \
        composed(logits, ids, penalty, 0.9, 0.8, seed)


def test_penalty_changes_distribution():
    # heavy penalty on the top token pushes sampling away from it
    rng = np.random.default_rng(32)
    logits = (rng.standard_normal(300) * 2).astype(np.float32)
    top = int(np.argmax(logits))
    draws_plain = {fusedtok.decode_step(logits, [], 1.0, p=0.8, seed=s)
                   for s in range(40)}
    draws_pen = {fusedtok.decode_step(logits, [top] * 5, 100.0, p=0.8, seed=s)
                 for s in range(40)}
    assert top in draws_plain or len(draws_plain) > 1
    assert top not in draws_pen


def test_empty_ids_and_disabled_penalty():
    rng = np.random.default_rng(33)
    logits = (rng.standard_normal(500) * 2).astype(np.float32)
    for seed in (1, 9):
        a = fusedtok.decode_step(logits, [], 2.0, p=0.9, seed=seed)
        b = fusedtok.decode_step(logits, [7], 1.0, p=0.9, seed=seed)
        ref = fusedtok.sample_topp(logits, 0.9, seed=seed)
        assert a == ref and b == ref


def test_errors():
    x = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.decode_step(x, [0], 0.0)
    with pytest.raises(ValueError):
        fusedtok.decode_step(x, [0], 1.1, p=0.0)
    with pytest.raises(ValueError):
        fusedtok.decode_step(x, [0], 1.1, temperature=0.0)
    with pytest.raises(ValueError):
        fusedtok.decode_step(x, [99], 1.1)
    with pytest.raises(ValueError):
        fusedtok.decode_step(np.ones((2, 2), dtype=np.float32), [0], 1.1)


@pytest.mark.skipif(not fusedtok.cuda_available(), reason="no GPU")
class TestCuda:
    @pytest.mark.parametrize("penalty", [1.0, 1.3])
    def test_staged_matches_cpu(self, penalty):
        rng = np.random.default_rng(34)
        logits = (rng.standard_normal(4000) * 2).astype(np.float32)
        ids = rng.integers(0, 4000, size=120).tolist()
        for seed in (0, 5, 77):
            assert fusedtok.decode_step(logits, ids, penalty, p=0.9,
                                        temperature=0.8, seed=seed,
                                        cuda=True) == \
                fusedtok.decode_step(logits, ids, penalty, p=0.9,
                                     temperature=0.8, seed=seed)

    def test_deterministic_repeats(self):
        rng = np.random.default_rng(35)
        logits = (rng.standard_normal(2048) * 2).astype(np.float32)
        ids = [3, 900, 2047]
        first = fusedtok.decode_step(logits, ids, 1.2, p=0.85, seed=42,
                                     cuda=True)
        assert all(fusedtok.decode_step(logits, ids, 1.2, p=0.85, seed=42,
                                        cuda=True) == first
                   for _ in range(3))

    def test_full_generation_loop(self):
        # a 40-token greedy-ish generation loop purely through decode_step
        rng = np.random.default_rng(36)
        vocab = 1024
        logits = (rng.standard_normal(vocab) * 3).astype(np.float32)
        history = [int(np.argmax(logits))]
        for step in range(40):
            tok = fusedtok.decode_step(logits, history, 1.15, p=0.9,
                                       temperature=0.9, seed=step,
                                       cuda=True)
            assert 0 <= tok < vocab
            history.append(tok)
        assert len(history) == 41
        # heavy penalty must break greedy repetition
        assert len(set(history[1:20])) > 1


@pytest.mark.skipif(not (HAS_TORCH and fusedtok.cuda_available()),
                    reason="no torch/GPU")
class TestTorchZeroCopy:
    def test_matches_cpu(self):
        rng = np.random.default_rng(37)
        logits = (rng.standard_normal(3000) * 2).astype(np.float32)
        ids = rng.integers(0, 3000, size=50).tolist()
        t = torch.from_numpy(logits).cuda()
        ti = torch.tensor(ids, dtype=torch.int64).cuda()
        assert fusedtok.decode_step(t, ti, 1.25, p=0.9, temperature=0.7,
                                    seed=11) == \
            fusedtok.decode_step(logits, ids, 1.25, p=0.9, temperature=0.7,
                                 seed=11)
