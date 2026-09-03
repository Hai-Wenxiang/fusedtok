"""Regression probes for the 1.2.1 selection-workspace fix (vocab > 131072).

The nucleus-scan scratch between the key buffers and the SelArgs block
was sized for 512 partial floats - exactly ceil(131072/256) blocks - so
every vocabulary ABOVE 131072 (Qwen's 152064, ...) overflowed into the
args block and corrupted the output pointers. These probes exercise the
crossing path at 152064 on the ops that run the partials:
`topp` (the crossing scan), `sample_topp` (workspace growth + the
full-vocab fast path) and `decode_step` (penalty bitmap + widening).

References are numpy (argsort-based), not the O(k*n) C++ top-k loop -
152064**2 would be minutes per call.
"""

import numpy as np
import pytest

import fusedtok

HAS_TORCH = True
try:
    import torch
except ImportError:
    HAS_TORCH = False

needs_gpu = pytest.mark.skipif(
    not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch/GPU")

# Qwen-family vocabulary: comfortably past the 131072 boundary (595
# partial blocks > the old 512-float reserve) but a modest allocation
N = 152064


def _topp_reference(probs, p):
    """Smallest top-p set (crossing element included), ties -> earliest."""
    order = np.argsort(-probs, kind="stable")
    csum = np.cumsum(probs[order])
    cut = int(np.searchsorted(csum, p, side="left")) + 1
    return order[:cut]


@needs_gpu
def test_topp_full_vocabulary_p_one():
    # p=1.0 keeps everything: the crossing scan runs at k=n=152064 where
    # the old reserve overflowed into SelArgs (count_out corrupted)
    rng = np.random.default_rng(0)
    probs = np.abs(rng.standard_normal(N).astype(np.float32))
    probs /= probs.sum()
    vals, idxs = fusedtok.topp(torch.from_numpy(probs).cuda(), 1.0)
    assert vals.shape[0] == N
    assert idxs.shape[0] == N
    ref = _topp_reference(probs, 1.0)
    got_idx = idxs.cpu().numpy()
    got_val = vals.cpu().numpy()
    assert np.array_equal(np.sort(got_idx), np.arange(N))  # a permutation
    assert got_val == pytest.approx(probs[ref], abs=1e-6)


@needs_gpu
def test_topp_peaked_nucleus_matches_reference():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(N).astype(np.float32) * 0.5
    logits[7] += 6.0                      # a dominant token
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    p = 0.9
    ref_idx = _topp_reference(probs, p)
    vals, idxs = fusedtok.topp(torch.from_numpy(probs).cuda(), p)
    assert idxs.shape[0] == ref_idx.shape[0]
    assert np.array_equal(idxs.cpu().numpy(), ref_idx)
    assert vals.cpu().numpy() == pytest.approx(probs[ref_idx], abs=1e-6)


@needs_gpu
def test_sample_topp_full_vocabulary_cpu_gpu_parity():
    # near-uniform logits force the adaptive widening to (nearly) the
    # full vocabulary: the biggest workspace + the fast path, at a vocab
    # that used to overflow the scan scratch
    rng = np.random.default_rng(2)
    logits = (rng.standard_normal(N).astype(np.float32) * 1e-3)
    tok_cpu = fusedtok.sample_topp(logits, 0.95, seed=123)
    tok_gpu = fusedtok.sample_topp(torch.from_numpy(logits).cuda(), 0.95,
                                   seed=123)
    # documented boundary caveat: near-uniform draws sit on the exp
    # rounding boundary (CPU exact exp vs GPU __expf, ~2 ulp each), and
    # the drift ACCUMULATES along the ~1e5-element serial CDF walk, so
    # CPU/GPU pick tokens a small RANK window apart (measured 14 ranks
    # at n=152064; their indices are far apart in a random permutation).
    # The GPU result itself must be a valid index and per-seed stable.
    assert 0 <= tok_gpu < N
    assert fusedtok.sample_topp(torch.from_numpy(logits).cuda(), 0.95,
                                seed=123) == tok_gpu
    order = np.argsort(-logits, kind="stable")
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)
    assert abs(rank[tok_cpu] - rank[tok_gpu]) <= 64


@needs_gpu
def test_sample_topp_peaked_parity_at_qwen_vocab():
    # peaked logits: the first window covers the nucleus, so the draw is
    # boundary-free and CPU/GPU tokens must match exactly
    rng = np.random.default_rng(3)
    logits = rng.standard_normal(N).astype(np.float32) * 0.5
    logits[7] += 20.0
    tok_cpu = fusedtok.sample_topp(logits, 0.9, seed=7)
    tok_gpu = fusedtok.sample_topp(torch.from_numpy(logits).cuda(), 0.9,
                                   seed=7)
    assert tok_gpu == tok_cpu


@needs_gpu
def test_decode_step_at_qwen_vocab():
    # the penalty bitmap rides after the args block in the same
    # workspace: a vocab-past-131072 decode with penalties exercises the
    # grown layout end to end
    rng = np.random.default_rng(4)
    logits_t = torch.tensor(rng.standard_normal(N).astype(np.float32),
                            device="cuda")
    history = [3, 11, 3]
    tok = fusedtok.decode_step(logits_t, history, penalty=1.3,
                               p=0.9, temperature=0.8, seed=5)
    assert isinstance(tok, int) and 0 <= tok < N
    # composed reference on the host, same order/seed
    host = logits_t.cpu().numpy()
    penalized = fusedtok.repetition_penalty(host, history, 1.3)
    ref = fusedtok.sample_topp(penalized, p=0.9, temperature=0.8, seed=5)
    assert tok == ref


# ---------------------------------------------------------------------------
# stride >= 2 checkpoint walks (1.3.1 fix). walk_cp_stride scales the
# checkpoint recording so the shared-memory slot array stays within
# kWalkCpMax = 8192 entries: a window of more than 8192 * 32 = 262144
# elements records every SECOND batch boundary (stride 2), and the
# walk_from_cp resume index used to sit (stride - 1) batches BEFORE the
# checkpoint boundary - those elements were counted twice (once in the
# checkpoint, once re-walked), inflating the cum and drawing a token
# (stride - 1) * 32 ranks early. Latent until v1.3.1 because no real
# vocabulary exceeded 262144.
# ---------------------------------------------------------------------------

N_WIDE = 300000      # > 262144 -> walk_cp_stride == 2 on the full window


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@needs_gpu
def test_sample_topp_stride2_checkpoints_match_cpu(seed):
    # all-zero logits: every exp is EXACTLY 1.0 on both sides (std::exp
    # and __expf agree at 0), and with n = 300000 < 2**24 the serial
    # CDF is exact integer arithmetic - so CPU and GPU tokens must be
    # bit-identical, with zero rounding-drift margin. p=0.9 widens the
    # window to the full vocabulary (stride 2); the pre-1.3.1 resume
    # bug shifted the GPU token exactly (stride - 1) * 32 = 32 ranks.
    logits = np.zeros(N_WIDE, dtype=np.float32)
    dev = torch.from_numpy(logits).cuda()
    tok_cpu = fusedtok.sample_topp(logits, 0.9, seed=seed)
    tok_gpu = fusedtok.sample_topp(dev, 0.9, seed=seed)
    assert tok_gpu == tok_cpu
    # per-seed stability on the same build
    assert fusedtok.sample_topp(dev, 0.9, seed=seed) == tok_gpu
