"""logit_penalties_batched (1.7): the combined penalty operator for a
whole [rows, vocab] batch with ragged per-row histories.

Covers the batched contract (each row bit-identical to the single-row
op on that row, every path), the three ragged id input forms, per-row
count independence, the error contract shared with decode_step_batched,
the raw-launch in-place support on the warm path, and graph capture.
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


def ref_batched(logits, row_ids, repetition=1.0, presence=0.0,
                frequency=0.0):
    """NumPy reference: the single-row semantics per row."""
    out = np.array(logits, dtype=np.float32, copy=True)
    for r, ids in enumerate(row_ids):
        counts = {}
        for i in ids:
            counts[int(i)] = counts.get(int(i), 0) + 1
        for tok, c in counts.items():
            v = out[r, tok]
            if np.float32(repetition) != np.float32(1.0):
                v = (v / np.float32(repetition) if v > 0
                     else v * np.float32(repetition))
            if np.float32(presence) != np.float32(0.0):
                v = v - np.float32(presence)
            if np.float32(frequency) != np.float32(0.0):
                v = v - np.float32(c) * np.float32(frequency)
            out[r, tok] = v
    return out


def make_batch(rng, rows, n, m_per_row):
    x = (rng.standard_normal((rows, n)) * 2.0).astype(np.float32)
    ids = [rng.integers(0, n, size=m).tolist() for m in m_per_row]
    return x, ids


def flat_of(ids):
    flat = [i for row in ids for i in row]
    offs = np.zeros(len(ids) + 1, dtype=np.int64)
    for r, row in enumerate(ids):
        offs[r + 1] = offs[r] + len(row)
    return np.asarray(flat, dtype=np.int64), offs


# ---------------------------------------------------------------------------
# semantics (host path)
# ---------------------------------------------------------------------------


def test_batched_rows_match_single_row_bit_exact():
    # the batched contract: each row IS the single-row op on that row
    rng = np.random.default_rng(170)
    x, ids = make_batch(rng, 5, 2048, [0, 1, 7, 500, 3000])
    flat, offs = flat_of(ids)
    got = fusedtok.logit_penalties_batched(x, flat, ids_offsets=offs,
                                           repetition=1.4, presence=0.3,
                                           frequency=0.2)
    want = ref_batched(x, ids, 1.4, 0.3, 0.2)
    assert (got == want).all()


def test_ragged_input_forms_agree():
    # list-of-lists, 2-D, and flat+offsets must produce identical output
    rng = np.random.default_rng(171)
    x, ids = make_batch(rng, 4, 512, [3, 0, 64, 10])
    flat, offs = flat_of(ids)
    kw = dict(repetition=1.5, presence=0.5, frequency=0.25)
    a = fusedtok.logit_penalties_batched(x, ids, **kw)
    b = fusedtok.logit_penalties_batched(x, flat, ids_offsets=offs, **kw)
    assert (a == b).all()
    # a 2-D id array: every row contributes ALL its columns - pad rows
    # with a valid id and expect the pad id to carry its own per-row
    # count (that is the documented 2-D semantics)
    m_max = max(len(r) for r in ids)
    ids2d = np.zeros((4, m_max), dtype=np.int64)
    for r, row in enumerate(ids):
        ids2d[r, :len(row)] = row
    ids2d_ref = [list(row) + [0] * (m_max - len(row)) for row in ids]
    c = fusedtok.logit_penalties_batched(x, ids2d, **kw)
    want = ref_batched(x, ids2d_ref, 1.5, 0.5, 0.25)
    assert (c == want).all()
    assert c.shape == a.shape == b.shape == (4, 512)


def test_counts_stay_per_row():
    # the same id in two rows carries different counts; one row's
    # penalties must not leak into the neighbor
    x = np.full((2, 16), 4.0, dtype=np.float32)
    ids = [[3, 3, 3], [5]]
    got = fusedtok.logit_penalties_batched(x, ids, frequency=0.5)
    # row 0: c=3 -> 4 - 1.5 ; row 1: id 3 unpenalized there
    assert got[0, 3] == pytest.approx(2.5, abs=1e-6)
    assert got[1, 3] == pytest.approx(4.0, abs=1e-6)
    assert got[1, 5] == pytest.approx(3.5, abs=1e-6)


def test_empty_rows_and_empty_batch():
    x = np.ones((3, 8), dtype=np.float32)
    got = fusedtok.logit_penalties_batched(x, [[], [0], []],
                                           repetition=2.0, presence=1.0)
    assert (got[0] == x[0]).all() and (got[2] == x[2]).all()
    # row 1: 1.0 is positive -> 1.0 / 2.0, minus presence -> -0.5
    assert got[1, 0] == pytest.approx(-0.5, abs=1e-6)
    empty = fusedtok.logit_penalties_batched(
        np.empty((0, 8), dtype=np.float32),
        np.zeros(0, dtype=np.int64), ids_offsets=np.zeros(1, dtype=np.int64))
    assert empty.shape == (0, 8)


def test_error_contract():
    x = np.ones((2, 8), dtype=np.float32)
    ids = np.zeros(3, dtype=np.int64)
    offs = np.array([0, 2, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        fusedtok.logit_penalties_batched(x, ids, ids_offsets=offs,
                                         repetition=0.0)
    with pytest.raises(ValueError):
        fusedtok.logit_penalties_batched(x, ids, ids_offsets=offs,
                                         repetition=-1.0)
    bad = np.array([8], dtype=np.int64)              # == vocab
    with pytest.raises(ValueError):
        fusedtok.logit_penalties_batched(x, bad,
                                         ids_offsets=np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        # wrong offsets length (rows + 1 required)
        fusedtok.logit_penalties_batched(x, ids,
                                         ids_offsets=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        # non-monotonic offsets
        fusedtok.logit_penalties_batched(
            x, np.zeros(3, dtype=np.int64),
            ids_offsets=np.array([0, 2, 1, 3], dtype=np.int64))
    with pytest.raises(TypeError):
        # float ids would silently truncate before the range check
        fusedtok.logit_penalties_batched(
            x, np.zeros(3, dtype=np.float64),
            ids_offsets=np.array([0, 1, 3], dtype=np.int64))
    with pytest.raises(ValueError):
        # 2-D id array with a rows mismatch against the logits
        fusedtok.logit_penalties_batched(x, np.zeros((3, 2),
                                                     dtype=np.int64))
    with pytest.raises(ValueError):
        fusedtok.logit_penalties_batched(np.ones((4, 8), dtype=np.float32),
                                         [[0], [1]])                    # rows mismatch


def test_row0_only_padding_row_consistency():
    # a 2-D id array where one row is all-padding: the pad id gets its
    # own per-row count there and nowhere else
    x = np.full((2, 8), 4.0, dtype=np.float32)
    ids2d = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64)
    got = fusedtok.logit_penalties_batched(x, ids2d, presence=1.0)
    assert got[0, 1] == pytest.approx(3.0, abs=1e-6)
    assert got[0, 0] == pytest.approx(4.0, abs=1e-6)
    assert got[1, 0] == pytest.approx(3.0, abs=1e-6)
    assert got[1, 1] == pytest.approx(4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GPU paths: staged, direct, raw launch (in-place), graph capture
# ---------------------------------------------------------------------------


@GPU
def test_gpu_bit_exact_matches_host():
    rng = np.random.default_rng(172)
    x, ids = make_batch(rng, 8, 8192,
                        [0, 1, 5, 64, 700, 4096, 3, 8000])
    flat, offs = flat_of(ids)
    want = ref_batched(x, ids, 1.3, 0.4, 0.15)
    # staged (torch-cpu input)
    staged = fusedtok.logit_penalties_batched(
        torch.from_numpy(x), torch.from_numpy(flat),
        ids_offsets=torch.from_numpy(offs), repetition=1.3, presence=0.4,
        frequency=0.15)
    assert (staged.numpy() == want).all()
    # direct (device tensors, zero copy)
    dev = fusedtok.logit_penalties_batched(
        torch.from_numpy(x).cuda(), torch.from_numpy(flat).cuda(),
        ids_offsets=torch.from_numpy(offs).cuda(), repetition=1.3,
        presence=0.4, frequency=0.15)
    assert dev.is_cuda
    assert (dev.cpu().numpy() == want).all()
    # host numpy input
    host = fusedtok.logit_penalties_batched(x, flat, ids_offsets=offs,
                                            repetition=1.3, presence=0.4,
                                            frequency=0.15)
    assert (host == want).all()


@GPU
def test_raw_launch_inplace_warm_path():
    from fusedtok import _fusedtok
    lg = torch.full((4, 1024), 4.0, device="cuda")
    ids = torch.tensor([3, 3, 5], dtype=torch.int64, device="cuda")
    offs = torch.tensor([0, 2, 2, 3, 3], dtype=torch.int64, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    _fusedtok.logit_penalties_batched_launch(
        lg.data_ptr(), ids.data_ptr(), offs.data_ptr(), lg.data_ptr(),
        4, 1024, 2.0, 0.5, 0.0, stream)
    # offs [0,2,2,3,3]: row 0 = ids [3,3] (c=2), row 1 empty, row 2 = [5],
    # row 3 empty. 4.0 / 2 = 2.0, minus presence 0.5 -> 1.5
    assert lg[0, 3].item() == pytest.approx(1.5, abs=1e-6)
    assert lg[1, 3].item() == pytest.approx(4.0, abs=1e-6)
    assert lg[2, 5].item() == pytest.approx(1.5, abs=1e-6)
    assert lg[3, 7].item() == pytest.approx(4.0, abs=1e-6)


@GPU
def test_cuda_graph_capture_replay():
    from fusedtok import _fusedtok

    def run(x, ids, offs, out):
        stream = torch.cuda.current_stream().cuda_stream
        _fusedtok.logit_penalties_batched_launch(
            x.data_ptr(), ids.data_ptr(), offs.data_ptr(), out.data_ptr(),
            2, 2048, 2.0, 0.5, 0.25, stream)

    x = torch.full((2, 2048), 4.0, device="cuda")
    # device-resident ids/offs: the captured graph bakes their pointers
    # in, so they must live on the device and stay stable across replays
    ids = torch.tensor([1, 1, 2], dtype=torch.int64, device="cuda")
    offs = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")
    out = torch.empty_like(x)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(x, ids, offs, out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(x, ids, offs, out)
    g.replay()
    torch.cuda.synchronize()
    assert out[0, 1].item() == pytest.approx(2.0 - 0.5 - 0.5, abs=1e-6)
    assert out[0, 100].item() == pytest.approx(4.0, abs=1e-6)
    # mutate between replays: a vacuous capture would keep warm-up values
    x.fill_(-4.0)
    ids.fill_(100)
    g.replay()
    torch.cuda.synchronize()
    # row 0: c=2 -> -8 - 0.5 - 0.5 ; row 1: c=1 -> -8 - 0.5 - 0.25
    assert out[0, 100].item() == pytest.approx(-9.0, abs=1e-5)
    assert out[1, 100].item() == pytest.approx(-8.75, abs=1e-5)


@GPU
def test_single_token_rows():
    x = np.zeros((3, 1), dtype=np.float32)
    got = fusedtok.logit_penalties_batched(
        torch.from_numpy(x).cuda(), [[0], [0, 0], []],
        repetition=2.0, presence=0.5, frequency=0.25)
    got = got.cpu().numpy()
    assert got[0, 0] == pytest.approx(-0.5 - 0.25, abs=1e-6)
    assert got[1, 0] == pytest.approx(-0.5 - 0.5, abs=1e-6)
    assert got[2, 0] == 0.0
