"""logit_penalties (1.6.1): the HF-style combined penalty operator.

Covers the composition order (repetition scale -> presence shift ->
count-weighted frequency shift), the once-per-distinct-id rule for
duplicate ids, the exact CPU/GPU contract (integer counts, no output
atomics -> bit-equal across numpy / staged-torch / direct-cuda paths),
the validation guards, and the raw-launch in-place support.
"""

import numpy as np
import pytest

import fusedtok

try:
    import torch
    HAS_TORCH = True
except ImportError:          # torch is optional; CI runs without it
    torch = None
    HAS_TORCH = False

GPU = pytest.mark.skipif(
    not (HAS_TORCH and fusedtok.cuda_available()), reason="no torch / no GPU")


def f32(x):
    return np.float32(x)


def ref_penalties(logits, ids, repetition=1.0, presence=0.0, frequency=0.0):
    """NumPy mirror of the documented formula, float32 throughout.

    Mirrors the C++ exactly: per DISTINCT id, repetition scale with the
    CTRL sign rule, then the presence shift, then c * frequency with c
    as a float32 cast (the kernel computes (float)c * frequency)."""
    y = np.array(logits, dtype=np.float32, copy=True)
    counts = {}
    for i in ids:
        counts[int(i)] = counts.get(int(i), 0) + 1
    for tok, c in counts.items():
        v = y[tok]
        if f32(repetition) != f32(1.0):
            v = v / f32(repetition) if v > 0 else v * f32(repetition)
        if f32(presence) != f32(0.0):
            v = v - f32(presence)
        if f32(frequency) != f32(0.0):
            v = v - np.float32(c) * f32(frequency)
        y[tok] = v
    return y


# ---------------------------------------------------------------------------
# composition semantics (host path)
# ---------------------------------------------------------------------------


def test_repetition_only_matches_repetition_penalty():
    # with presence/frequency at their defaults the op must agree
    # bit-for-bit with the 1.0 repetition_penalty operator
    rng = np.random.default_rng(11)
    lg = rng.standard_normal(512).astype(np.float32)
    ids = [0, 1, 2, 500, 500]
    a = fusedtok.logit_penalties(lg, ids, repetition=1.3)
    b = fusedtok.repetition_penalty(lg, ids, 1.3)
    assert (a == b).all()


def test_presence_subtracts_once_per_distinct_id():
    lg = np.ones(8, dtype=np.float32) * 4.0
    y = fusedtok.logit_penalties(lg, [3, 3, 3], presence=0.5)
    # duplicates must not stack: 4.0 - 0.5, not 4.0 - 1.5
    assert y[3] == pytest.approx(3.5, abs=1e-6)
    assert y[0] == pytest.approx(4.0, abs=1e-6)
    # a negative logit shifts the same way
    y2 = fusedtok.logit_penalties(np.full(8, -4.0, dtype=np.float32),
                                  [7], presence=0.5)
    assert y2[7] == pytest.approx(-4.5, abs=1e-6)


def test_frequency_scales_with_count():
    lg = np.ones(8, dtype=np.float32) * 4.0
    y = fusedtok.logit_penalties(lg, [5, 5, 5], frequency=0.25)
    # c = 3 -> subtract 0.75 once
    assert y[5] == pytest.approx(3.25, abs=1e-6)
    y1 = fusedtok.logit_penalties(lg, [5], frequency=0.25)
    assert y1[5] == pytest.approx(3.75, abs=1e-6)


def test_composition_order_rep_then_presence_then_frequency():
    # positive logit: divide, then shift, then count-weighted shift;
    # order is observable (e.g. presence before vs after the scale differ)
    lg = np.full(8, 4.0, dtype=np.float32)
    y = fusedtok.logit_penalties(lg, [2, 2], repetition=2.0,
                                 presence=0.5, frequency=0.5)
    assert y[2] == pytest.approx(4.0 / 2.0 - 0.5 - 2 * 0.5, abs=1e-6)
    # negative logit: multiply, then the two shifts
    lg_neg = np.full(8, -4.0, dtype=np.float32)
    y_neg = fusedtok.logit_penalties(lg_neg, [2, 2], repetition=2.0,
                                     presence=0.5, frequency=0.5)
    assert y_neg[2] == pytest.approx(-8.0 - 0.5 - 1.0, abs=1e-5)


def test_zero_logit_boundary():
    # v == 0 takes the multiply branch (0 * rep == 0) but still receives
    # both shifts
    lg = np.zeros(8, dtype=np.float32)
    y = fusedtok.logit_penalties(lg, [1], repetition=3.0,
                                 presence=0.25, frequency=0.5)
    assert y[1] == pytest.approx(-0.25 - 0.5, abs=1e-6)
    assert y[0] == 0.0


def test_noop_defaults_and_empty_ids():
    rng = np.random.default_rng(12)
    lg = rng.standard_normal(64).astype(np.float32)
    # all three penalties at their defaults: exact pass-through
    assert (fusedtok.logit_penalties(lg, [0, 1, 2]) == lg).all()
    # empty id list with active penalties: exact pass-through
    assert (fusedtok.logit_penalties(lg, [], repetition=2.0,
                                     presence=1.0, frequency=1.0) == lg).all()


def test_identity_repetition_is_exact():
    # repetition == 1.0 with presence active must not touch the values
    # through the division (v / 1.0 == v is exact anyway, the kernel
    # also skips the branch)
    rng = np.random.default_rng(13)
    lg = rng.standard_normal(128).astype(np.float32)
    a = fusedtok.logit_penalties(lg, [3, 3, 9], repetition=1.0, presence=0.1)
    b = ref_penalties(lg, [3, 3, 9], repetition=1.0, presence=0.1)
    assert (a == b).all()


def test_full_matrix_against_reference():
    rng = np.random.default_rng(14)
    for n, m in ((1, 0), (1, 1), (7, 4), (256, 32), (4096, 500), (30011, 2000)):
        lg = (rng.standard_normal(n) * 3.0).astype(np.float32)
        ids = rng.integers(0, n, size=m).tolist()
        for rep, pres, freq in ((1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                                (1.0, 0.8, 0.0), (1.0, 0.0, 0.3),
                                (0.7, 0.4, 0.2), (2.0, 1.0, 0.5)):
            got = fusedtok.logit_penalties(lg, ids, repetition=rep,
                                           presence=pres, frequency=freq)
            want = ref_penalties(lg, ids, rep, pres, freq)
            assert got.dtype == np.float32
            assert (got == want).all(), (n, m, rep, pres, freq)


def test_guards():
    lg = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError):
        fusedtok.logit_penalties(lg, [0], repetition=0.0)
    with pytest.raises(ValueError):
        fusedtok.logit_penalties(lg, [0], repetition=-1.0)
    with pytest.raises(ValueError):
        fusedtok.logit_penalties(lg, [4], repetition=1.0)   # id == vocab
    with pytest.raises(ValueError):
        fusedtok.logit_penalties(lg, [-1], repetition=1.0)
    with pytest.raises(ValueError):
        fusedtok.logit_penalties(np.ones((2, 2), dtype=np.float32), [0])
    with pytest.raises(TypeError):
        fusedtok.logit_penalties(lg, [0.5])                  # float ids
    with pytest.raises(ValueError):
        fusedtok.logit_penalties(lg, [[0, 1]])               # not 1-D


# ---------------------------------------------------------------------------
# GPU paths: staged (torch-cpu host operands), direct (torch-cuda), and
# the raw launcher, all bit-equal to the host reference
# ---------------------------------------------------------------------------


@GPU
def test_gpu_bit_exact_against_reference():
    rng = np.random.default_rng(15)
    n, m = 8192, 300
    host = (rng.standard_normal(n) * 2.0).astype(np.float32)
    ids_host = rng.integers(0, n, size=m)
    want = ref_penalties(host, ids_host, 1.4, 0.3, 0.2)
    # staged path: torch CPU tensors go host -> device -> host
    staged = fusedtok.logit_penalties(torch.from_numpy(host),
                                      torch.from_numpy(ids_host.astype(np.int64)),
                                      repetition=1.4, presence=0.3,
                                      frequency=0.2)
    assert (staged.numpy() == want).all()
    # direct path: device tensors, zero copy
    dev_lg = torch.tensor(host, device="cuda")
    dev_ids = torch.tensor(ids_host.astype(np.int64), device="cuda")
    direct = fusedtok.logit_penalties(dev_lg, dev_ids, repetition=1.4,
                                      presence=0.3, frequency=0.2)
    assert (direct.cpu().numpy() == want).all()
    # the three paths agree bit-for-bit with each other too
    got_host = fusedtok.logit_penalties(host, ids_host.tolist(),
                                        repetition=1.4, presence=0.3,
                                        frequency=0.2)
    assert (got_host == want).all()


@GPU
def test_gpu_duplicates_and_empty_ids():
    lg = torch.full((1024,), 4.0, device="cuda")
    y = fusedtok.logit_penalties(lg, [7, 7, 7], repetition=2.0,
                                 presence=0.5, frequency=0.5)
    assert y[7].item() == pytest.approx(2.0 - 0.5 - 1.5, abs=1e-6)
    assert y[0].item() == pytest.approx(4.0, abs=1e-6)
    ids_empty = torch.zeros(0, dtype=torch.int64, device="cuda")
    y2 = fusedtok.logit_penalties(lg, ids_empty, repetition=2.0,
                                  presence=1.0)
    assert (y2 == lg).all()


@GPU
def test_raw_launch_inplace_alias_supported():
    # on the cached-workspace path the launcher may write in place
    # (counts live in the workspace, never in y)
    from fusedtok import _fusedtok as native
    lg = torch.full((2048,), 4.0, device="cuda")
    ids = torch.tensor([3, 3, 5], dtype=torch.int64, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    native.logit_penalties_launch(lg.data_ptr(), ids.data_ptr(),
                                  lg.data_ptr(), 2048, 3, 2.0, 0.5, 0.0,
                                  stream)
    assert lg[3].item() == pytest.approx(2.0 - 0.5, abs=1e-6)
    assert lg[5].item() == pytest.approx(2.0 - 0.5, abs=1e-6)
    assert lg[0].item() == pytest.approx(4.0, abs=1e-6)


@GPU
def test_cuda_graph_capture_replay():
    # capture after warmup: the counts workspace is hot, the memsets and
    # both kernels ride the capture; replay after mutating the logits
    # and the ids must recompute everything
    from fusedtok import _fusedtok as native

    def run(x, ids, out):
        stream = torch.cuda.current_stream().cuda_stream
        native.logit_penalties_launch(x.data_ptr(), ids.data_ptr(),
                                      out.data_ptr(), x.numel(),
                                      ids.numel(), 2.0, 0.5, 0.25, stream)

    x = torch.full((4096,), 4.0, device="cuda")
    ids = torch.tensor([1, 1, 2], dtype=torch.int64, device="cuda")
    out = torch.empty_like(x)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(x, ids, out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(x, ids, out)
    g.replay()
    torch.cuda.synchronize()
    assert out[1].item() == pytest.approx(2.0 - 0.5 - 0.5, abs=1e-6)
    assert out[2].item() == pytest.approx(2.0 - 0.5 - 0.25, abs=1e-6)
    assert out[100].item() == pytest.approx(4.0, abs=1e-6)
    # mutate between replays: a vacuously-captured graph would keep the
    # warm-up values. all three slots now point at id 100, so its count
    # is c = 3 and the frequency term is 3 * 0.25
    x.fill_(-4.0)
    ids.fill_(100)
    g.replay()
    torch.cuda.synchronize()
    assert out[100].item() == pytest.approx(-8.0 - 0.5 - 0.75, abs=1e-5)
    assert out[1].item() == pytest.approx(-4.0, abs=1e-6)
