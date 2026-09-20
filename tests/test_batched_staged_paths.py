"""Staged execution paths of every batched sampler (numpy + cuda=True).

The staged path uploads a host array to a scratch device buffer and
launches the same kernels as the zero-copy path. Until 2.2.1 it was
only covered for the v1.4 trio (topp/minp/topk) - which is exactly how
two binding defects shipped unnoticed: `sample_xtc_batched` had no
staged binding at all (AttributeError) and the `sample_tfs_batched`
staged binding computed on the CPU after a dead device upload. This
file pins the staged contract for ALL nine batched samplers so a new
sampler cannot skip it again:

- staged output exists and is a numpy ndarray (not an AttributeError)
- staged per-row parity vs the row-wise CPU singles, with the
  documented neighbor-rank tolerance (block-atomic accumulation order
  differs between paths, so a draw landing exactly on a CDF boundary
  may pick a neighbor - same contract as the single-row samplers)
- staged determinism: identical repeated calls give identical tokens
- empty batch on the staged path returns an empty result
- the CPU direct surface (numpy, no cuda) matches the row-wise CPU
  singles bit-exactly for all nine samplers
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

# (name, batched callable, single callable, positional args tuple) for
# every batched sampler in the API. The args tuple is passed after the
# logits matrix / row to both the batched and the single-row form.
SAMPLERS = [
    ("topp", fusedtok.sample_topp_batched, fusedtok.sample_topp, (0.9,)),
    ("minp", fusedtok.sample_minp_batched, fusedtok.sample_minp, (0.05,)),
    ("topk", fusedtok.sample_topk_batched, fusedtok.sample_topk, (50,)),
    ("eta", fusedtok.sample_eta_batched, fusedtok.sample_eta, (1e-3,)),
    ("typical", fusedtok.sample_typical_batched, fusedtok.sample_typical,
     (0.9,)),
    ("topa", fusedtok.sample_topa_batched, fusedtok.sample_topa, (0.2,)),
    ("nsigma", fusedtok.sample_nsigma_batched, fusedtok.sample_nsigma,
     (1.5,)),
    ("tfs", fusedtok.sample_tfs_batched, fusedtok.sample_tfs, (0.95,)),
    ("xtc", fusedtok.sample_xtc_batched, fusedtok.sample_xtc, (3, 0.8)),
]


def _batch(rng, b, n):
    """Mixed-shape batch: spiked / plain randn rows."""
    x = rng.standard_normal((b, n)).astype(np.float32)
    for r in range(0, b, 2):
        x[r, 7] += 8.0
    return x


def _assert_neighbor_rank(logits_row, got, want, what):
    """Exact per-row parity with the documented ulp fallback: the GPU
    total is accumulated with per-block float atomics whose arrival
    order differs between launch shapes, so a draw landing exactly on a
    CDF boundary may pick a neighbor rank (<= 2 apart in stable order)."""
    if got == want:
        return
    order = np.argsort(-logits_row, kind="stable")
    rank = {int(t): i for i, t in enumerate(order)}
    assert got in rank and want in rank, what
    assert abs(rank[got] - rank[want]) <= 2, (what, got, want,
                                              rank[got], rank[want])


@pytest.mark.parametrize("name,batched,single,args", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_cpu_direct_matches_rowwise_singles(name, batched, single, args):
    rng = np.random.default_rng(400)
    x = _batch(rng, 5, 4096)
    seeds = np.arange(5, dtype=np.int64)
    got = batched(x, *args, seeds=seeds)
    want = [int(single(x[r], *args, seed=int(s)))
            for r, s in enumerate(seeds)]
    assert isinstance(got, np.ndarray), name
    assert got.tolist() == want, name


@needs_gpu
@pytest.mark.parametrize("name,batched,single,args", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_staged_matches_rowwise_singles(name, batched, single, args):
    rng = np.random.default_rng(401)
    x = _batch(rng, 6, 4096)
    seeds = np.arange(6, dtype=np.int64)
    got = batched(x, *args, seeds=seeds, cuda=True)
    assert isinstance(got, np.ndarray), name
    assert got.shape == (6,), name
    for r in range(6):
        want = int(single(x[r], *args, seed=int(seeds[r])))
        _assert_neighbor_rank(x[r], int(got[r]), want, (name, r))


@needs_gpu
@pytest.mark.parametrize("name,batched,single,args", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_staged_determinism(name, batched, single, args):
    rng = np.random.default_rng(402)
    x = _batch(rng, 4, 2048)
    seeds = np.arange(4, dtype=np.int64)
    first = batched(x, *args, seeds=seeds, cuda=True)
    second = batched(x, *args, seeds=seeds, cuda=True)
    assert first.tolist() == second.tolist(), name


@needs_gpu
@pytest.mark.parametrize("name,batched,single,args", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_staged_empty_batch(name, batched, single, args):
    x = np.empty((0, 512), dtype=np.float32)
    got = batched(x, *args, cuda=True)
    assert isinstance(got, np.ndarray), name
    assert got.shape == (0,), name


@needs_gpu
@pytest.mark.parametrize("name,batched,single,args", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_torch_cpu_tensor_host_path(name, batched, single, args):
    """A torch CPU tensor is a supported host input for every batched
    sampler - the 2.4.2 dispatcher delegation briefly rejected it for
    dry with a misleading zero-copy TypeError (2.4.3 fix)."""
    import torch
    rng = np.random.default_rng(403)
    x = _batch(rng, 4, 2048)
    seeds = np.arange(4, dtype=np.int64)
    xt = torch.from_numpy(x)                     # CPU torch tensor
    got = batched(xt, *args, seeds=seeds)
    want = [int(single(x[r], *args, seed=int(seeds[r])))
            for r in range(4)]
    got_l = got.tolist() if hasattr(got, "tolist") else list(got)
    assert got_l == want, name


@needs_gpu
def test_dry_batched_torch_cpu_tensor():
    import torch
    rng = np.random.default_rng(404)
    x = rng.standard_normal((4, 2048)).astype(np.float32)
    hists = [[5, 9, 5, 9], [], [7] * 8, [1, 2, 3]]
    seeds = np.arange(4, dtype=np.int64)
    got = fusedtok.sample_dry_batched(torch.from_numpy(x), hists, 2,
                                      1.75, seeds=seeds)
    want = [fusedtok.sample_dry(x[r], hists[r], 2, 1.75,
                                seed=int(seeds[r])) for r in range(4)]
    assert got.tolist() == want
