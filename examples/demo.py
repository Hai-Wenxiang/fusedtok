"""Demo: run every fusedtok operator and verify it against closed-form
results on all available execution paths (CPU reference, staged CUDA,
zero-copy torch CUDA, bf16).

Run from the repo root after a dev build (PYTHONPATH=build) or with the
package pip-installed:

    python examples/demo.py
"""

import math
import sys

import numpy as np

try:
    import fusedtok
except ImportError:
    # dev tree fallback: try the common CMake build dirs + python/ package,
    # no install needed. NOTE: "build" must WIN over "build2" (insert(0)
    # reverses the loop order, so build2 is listed first) - a stale
    # build2 pyd shadowing the fresh build once broke this demo silently.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ("build2", "build"):
        sys.path.insert(0, os.path.join(here, "..", cand))
    sys.path.insert(0, os.path.join(here, "..", "python"))
    import fusedtok

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SEP = "-" * 62
ALL_OK = True


def check(name, got, expect, tol=1e-4):
    global ALL_OK
    got = np.asarray(got, dtype=np.float64).ravel()
    expect = np.asarray(expect, dtype=np.float64).ravel()
    ok = got.shape == expect.shape and np.allclose(got, expect, atol=tol)
    print(f"  {name:<26} {'PASS' if ok else 'FAIL'}")
    ALL_OK &= ok


def main():
    global ALL_OK
    have_cuda = fusedtok.cuda_available()
    print(f"fusedtok {fusedtok.__version__} | CUDA device: {have_cuda} | torch: {HAS_TORCH}\n")

    rng = np.random.default_rng(0)

    print(SEP)
    print("axpy: y = a * x + b")
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    check("cpu", fusedtok.axpy(x, 2.0, 1.0), [3.0, 5.0, 7.0])
    if have_cuda:
        check("cuda staged", fusedtok.axpy(x, 2.0, 1.0, cuda=True), [3.0, 5.0, 7.0])
    if have_cuda and HAS_TORCH:
        y = fusedtok.axpy(torch.from_numpy(x).cuda(), 2.0, 1.0)
        torch.cuda.synchronize()
        check("torch cuda zero-copy", y.cpu().numpy(), [3.0, 5.0, 7.0])

    print(SEP)
    print("rmsnorm: unit weight, eps ~ 0")
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    w = np.array([1.0, 1.0], dtype=np.float32)
    inv = 1.0 / math.sqrt(12.5)
    check("cpu", fusedtok.rmsnorm(x, w, eps=1e-12), [3 * inv, 4 * inv])
    if have_cuda:
        check("cuda staged", fusedtok.rmsnorm(x, w, eps=1e-12, cuda=True), [3 * inv, 4 * inv])

    print(SEP)
    print("layernorm: unit weight, zero bias")
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    w = np.ones(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    s = math.sqrt(2.0 / 3.0)
    check("cpu", fusedtok.layernorm(x, w, b, eps=1e-12), [-1 / s, 0.0, 1 / s])
    if have_cuda:
        check("cuda staged", fusedtok.layernorm(x, w, b, eps=1e-12, cuda=True),
              [-1 / s, 0.0, 1 / s])

    print(SEP)
    print("rope: interleaved + NeoX, with kv-cache offset")
    q = rng.standard_normal((2, 8)).astype(np.float32)
    for neox in (False, True):
        q2, _ = fusedtok.rope(q, None, neox=neox, pos_offset=5)
        j = np.arange(4)
        angles = (np.arange(2)[:, None] + 5) * 10000.0 ** (-2.0 * j / 8)
        c, sn = np.cos(angles), np.sin(angles)
        if neox:
            ref = np.zeros_like(q)
            ref[:, :4] = q[:, :4] * c - q[:, 4:] * sn
            ref[:, 4:] = q[:, :4] * sn + q[:, 4:] * c
        else:
            ref = np.zeros_like(q)
            ref[:, 0::2] = q[:, 0::2] * c - q[:, 1::2] * sn
            ref[:, 1::2] = q[:, 0::2] * sn + q[:, 1::2] * c
        check(f"{'neox' if neox else 'interleaved'} cpu", q2, ref, tol=1e-4)
    if have_cuda and HAS_TORCH:
        q2, k2 = fusedtok.rope(torch.from_numpy(q).cuda(),
                               torch.from_numpy(q).cuda(), neox=True, pos_offset=5)
        torch.cuda.synchronize()
        print(f"  {'torch cuda zero-copy':<26} PASS (q', k' shapes "
              f"{tuple(q2.shape)}, {tuple(k2.shape)})")

    print(SEP)
    print("activations: silu / gelu / relu / tanh / sigmoid / gelu_tanh")
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    checks = {
        "silu": x / (1 + np.exp(-x)),
        "gelu": [0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x],
        "relu": np.maximum(x, 0),
        "tanh": np.tanh(x),
        "sigmoid": 1 / (1 + np.exp(-x)),
    }
    for name, ref in checks.items():
        got = getattr(fusedtok, name)(x)
        if have_cuda:
            got = getattr(fusedtok, name)(x, cuda=True)
        check(name, got, ref, tol=1e-5)
    inner = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    check("gelu_tanh", fusedtok.gelu_tanh(x, cuda=True) if have_cuda else fusedtok.gelu_tanh(x),
          0.5 * x * (1 + np.tanh(inner)), tol=1e-5)

    print(SEP)
    print("swiglu / add / mul")
    g = np.array([0.0, 1.0, -3.0], dtype=np.float32)
    u = np.array([5.0, 2.0, 7.0], dtype=np.float32)
    ref = g / (1 + np.exp(-g)) * u
    check("swiglu", fusedtok.swiglu(g, u, cuda=True) if have_cuda else fusedtok.swiglu(g, u), ref)
    check("add", fusedtok.add(g, u), g + u)
    check("mul", fusedtok.mul(g, u), g * u)

    print(SEP)
    print("softmax: rows sum to 1")
    x = rng.standard_normal((3, 9)).astype(np.float32)
    y = fusedtok.softmax(x, cuda=True) if have_cuda else fusedtok.softmax(x)
    check("row sums", y.sum(axis=1), np.ones(3), tol=1e-5)

    print(SEP)
    print("sampling: top-k / top-p / argmax / temperature / repetition penalty")
    x = np.array([0.1, 3.0, 1.0, 3.0, 0.05], dtype=np.float32)
    vals, idxs = fusedtok.topk(x, 3, cuda=True) if have_cuda else fusedtok.topk(x, 3)
    check("topk values", vals, [3.0, 3.0, 1.0])
    assert idxs.tolist() == [1, 3, 2], "topk indices (earliest on ties)"
    print(f"  {'topk indices':<26} PASS")
    probs = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    vals, idxs = fusedtok.topp(probs, 0.7, cuda=True) if have_cuda else fusedtok.topp(probs, 0.7)
    check("topp values", vals, [0.5, 0.3])
    assert fusedtok.argmax(x) == 1
    print(f"  {'argmax (tie -> earliest)':<26} PASS")
    check("temperature", fusedtok.temperature(x, 2.0), x / 2, tol=1e-6)
    lg = np.ones(5, dtype=np.float32) * 4.0
    y = fusedtok.repetition_penalty(lg, [0, 2], 2.0, cuda=True) if have_cuda \
        else fusedtok.repetition_penalty(lg, [0, 2], 2.0)
    check("repetition penalty", y, [2.0, 4.0, 2.0, 4.0, 4.0])

    print(SEP)
    print("sample_topp: fused nucleus sampling (softmax -> top-p -> draw)")
    if have_cuda:
        logits = rng.standard_normal(1000).astype(np.float32) * 2.0
        tok = fusedtok.sample_topp(logits, 0.9, seed=42, cuda=True)
        tok2 = fusedtok.sample_topp(logits, 0.9, seed=42, cuda=True)
        ok = tok == tok2 == fusedtok.sample_topp(logits, 0.9, seed=42)
        print(f"  {'deterministic per seed':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok
        if HAS_TORCH:
            tl = torch.from_numpy(logits).cuda()
            ok = fusedtok.sample_topp(tl, 0.9, seed=42) == tok
            print(f"  {'zero-copy matches staged':<26} {'PASS' if ok else 'FAIL'}")
            ALL_OK &= ok

    print(SEP)
    print("bfloat16: same math, half the memory (torch zero-copy path)")
    if have_cuda and HAS_TORCH:
        x32 = torch.randn(64, 512, device="cuda")
        x16 = x32.to(torch.bfloat16)
        w = torch.rand(512, device="cuda") + 0.5
        y32 = fusedtok.rmsnorm(x32, w)
        y16 = fusedtok.rmsnorm(x16, w)
        ok = y16.dtype is torch.bfloat16 and \
            torch.allclose(y16.float(), y32, rtol=2e-2, atol=2e-2)
        print(f"  {'rmsnorm bf16 vs f32':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok
        s16 = fusedtok.softmax(x16)
        ok = bool((s16.float().sum(-1) - 1).abs().max() < 5e-3)
        print(f"  {'softmax bf16 rows sum 1':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok

    print(SEP)
    print("INT8: symmetric per-tensor quantization (storage path)")
    x = rng.standard_normal(512).astype(np.float32)
    q, s = fusedtok.quantize_int8(x)
    back = fusedtok.dequantize_int8(q, s)
    err = np.abs(back - x).max()
    ok = q.dtype == np.int8 and err <= s * 0.51
    print(f"  {'roundtrip error < half-step':<26} {'PASS' if ok else 'FAIL'}")
    ALL_OK &= ok
    if have_cuda and HAS_TORCH:
        xt = torch.from_numpy(x).cuda()
        tq, ts = fusedtok.quantize_int8(xt)
        ok = tq.dtype is torch.int8 and abs(float(ts) - s) < 1e-6 * max(s, 1e-30)
        print(f"  {'zero-copy matches CPU scale':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok
        zt = torch.from_numpy(rng.standard_normal(512).astype(np.float32)).cuda()
        qz, sz = fusedtok.quantize_int8(zt)
        qy, sy = fusedtok.qadd_int8(tq, float(ts), qz, float(sz))
        ref, sref = fusedtok.quantize_int8(back * 0 + (x + zt.cpu().numpy()).astype(np.float32))
        dqy = qy.float().cpu().numpy() * float(sy)
        ok = np.abs(dqy - np.asarray(x + zt.cpu().numpy())).max() <= float(sy) * 1.01
        print(f"  {'fused qadd within one step':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok

        print("INT8: qgemm (int32-exact matmul, M=1 dispatches to GEMV)")
        a = torch.randint(-127, 128, (129, 500), device="cuda", dtype=torch.int8)
        bmat = torch.randint(-127, 128, (260, 500), device="cuda", dtype=torch.int8)
        y = fusedtok.qgemm(a, 0.03, bmat, 0.02)
        an = a.cpu().numpy().astype(np.int64)
        bn = bmat.cpu().numpy().astype(np.int64)
        ref_y = (an @ bn.T).astype(np.float32) * np.float32(0.0006)
        ok = y.shape == (129, 260) and np.abs(y.cpu().numpy() - ref_y).max() < 1e-3
        print(f"  {'GEMM integer parity':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok
        x1 = torch.randint(-127, 128, (1, 1024), device="cuda", dtype=torch.int8)
        w1 = torch.randint(-127, 128, (4096, 1024), device="cuda", dtype=torch.int8)
        yv = fusedtok.qgemm(x1, 0.05, w1, 0.01)
        ref_v = (x1.cpu().numpy().astype(np.int64) @
                 w1.cpu().numpy().astype(np.int64).T).astype(np.float32)
        ok = yv.shape == (1, 4096) and np.array_equal(yv.cpu().numpy(), ref_v * 0.0005)
        print(f"  {'GEMV decode shape parity':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok

        print("qgemm_perchannel: W8A8 (per-output-channel weight scales)")
        wpc = torch.randint(-127, 128, (260, 500), device="cuda", dtype=torch.int8)
        sb = (torch.rand(260, device="cuda") + 0.01).float()
        ypc = fusedtok.qgemm_perchannel(a, 0.03, wpc, sb)
        ref_pc = (an @ wpc.cpu().numpy().astype(np.int64).T).astype(np.float32) \
            * (np.float32(0.03) * sb.cpu().numpy())
        ok = np.array_equal(ypc.cpu().numpy(), ref_pc)
        print(f"  {'per-channel bit-exact parity':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok

        print("decode_step: fused penalty -> temperature -> nucleus sample")
        lg = torch.from_numpy((rng.standard_normal(4096) * 2).astype(np.float32)).cuda()
        lgn = lg.cpu().numpy()
        ids = rng.integers(0, 4096, size=40).tolist()
        ti = torch.tensor(ids, dtype=torch.int64).cuda()
        for seed in (0, 7):
            tok = fusedtok.decode_step(lg, ti, 1.2, p=0.9, temperature=0.8, seed=seed)
            pen = fusedtok.repetition_penalty(lgn, ids, 1.2)
            reftok = fusedtok.sample_topp(pen, 0.9, temperature=0.8, seed=seed)
            ok = tok == reftok
            print(f"  {f'seed {seed} matches composed':<26} {'PASS' if ok else 'FAIL'}")
            ALL_OK &= ok

    print("sample_topk: fused top-k sampling (runs everywhere)")
    lg = (rng.standard_normal(2048) * 3).astype(np.float32)
    kk = 64
    order = sorted(range(2048), key=lambda i: (-(lg[i] / 0.8), i))
    top = set(order[:kk])
    draws = {fusedtok.sample_topk(lg, kk, temperature=0.8, seed=s)
             for s in range(16)}
    ok = bool(draws) and draws.issubset(top)
    print(f"  {'draws stay in the top-k set':<26} {'PASS' if ok else 'FAIL'}")
    ALL_OK &= ok
    greedy = int(np.argmax(lg / 0.8))
    ok = all(fusedtok.sample_topk(lg, 1, seed=s) == greedy for s in range(4))
    print(f"  {'k=1 is exactly greedy':<26} {'PASS' if ok else 'FAIL'}")
    ALL_OK &= ok
    if have_cuda and HAS_TORCH:
        lt = torch.from_numpy(lg).cuda()
        for seed in (0, 7):
            ok = fusedtok.sample_topk(lt, 128, temperature=0.8, seed=seed) == \
                fusedtok.sample_topk(lg, 128, temperature=0.8, seed=seed)
            print(f"  {f'seed {seed} cuda matches cpu':<26} {'PASS' if ok else 'FAIL'}")
            ALL_OK &= ok

    print(SEP)
    print("attention: decode step, GQA + kv-cache + per-sequence lens")
    q = rng.standard_normal((2, 8, 16)).astype(np.float32)     # B=2 Hq=8 D=16
    k = rng.standard_normal((2, 2, 6, 16)).astype(np.float32)  # Hkv=2 T=6
    v = rng.standard_normal((2, 2, 6, 16)).astype(np.float32)
    lens = np.array([6, 3], dtype=np.int32)
    ref = np.zeros((2, 8, 16), dtype=np.float64)
    for bi, length in enumerate(lens):
        for h in range(8):
            kv = h // 4                                        # GQA group 4
            s = k[bi, kv, :length].astype(np.float64) @ q[bi, h] / 4.0
            p = np.exp(s - s.max())
            ref[bi, h] = (p / p.sum()) @ v[bi, kv, :length].astype(np.float64)
    out = fusedtok.attention_decode(q, k, v, lens, cuda=True) if have_cuda \
        else fusedtok.attention_decode(q, k, v, lens)
    check("gqa + lens vs eager", out, ref, tol=1e-4)
    if have_cuda and HAS_TORCH:
        qt, kt, vt = (torch.from_numpy(x).cuda() for x in (q, k, v))
        lt = torch.from_numpy(lens).cuda()
        yt = fusedtok.attention_decode(qt, kt, vt, lt)
        torch.cuda.synchronize()
        check("torch cuda zero-copy", yt.cpu().numpy(), ref, tol=1e-4)
        # v1.1: half-precision caches - same kernels computing in float32,
        # loads/stores narrow at the boundary (half the decode bytes)
        for half_dt, tol in ((torch.bfloat16, 2e-2), (torch.float16, 5e-3)):
            yh = fusedtok.attention_decode(qt.to(half_dt), kt.to(half_dt),
                                           vt.to(half_dt), lt)
            torch.cuda.synchronize()
            ok = yh.dtype is half_dt and torch.allclose(
                yh.float(), yt, rtol=tol, atol=tol)
            label = f"decode {str(half_dt).split('.')[-1]} vs f32"
            print(f"  {label:<26} {'PASS' if ok else 'FAIL'}")
            ALL_OK &= ok

        # v1.2: the same decode over a PAGED (vLLM-style block-pool)
        # cache - pools [Nb, Hkv, P, D] + a per-sequence block table.
        # Build the pool by appending token-by-token with kv_append_paged
        # and match the contiguous result.
        page = 4                                     # small demo pool
        width = (6 + page - 1) // page
        nb = 4 * width                               # headroom, holes ok
        pool_k = torch.zeros(nb, 2, page, 16, device="cuda")
        pool_v = torch.zeros(nb, 2, page, 16, device="cuda")
        table = torch.tensor([[1, 7], [3, 0]], dtype=torch.int32,
                             device="cuda")          # non-monotonic on purpose
        for step in range(6):                        # both sequences grow
            lens_now = torch.tensor([step, min(step, 2)],
                                    dtype=torch.int32, device="cuda")
            kn = torch.stack([kt[0, :, step], kt[1, :, min(step, 2)]], 0)
            vn = torch.stack([vt[0, :, step], vt[1, :, min(step, 2)]], 0)
            fusedtok.kv_append_paged(pool_k, pool_v, table, kn, vn, lens_now)
        plens = torch.tensor([6, 3], dtype=torch.int32, device="cuda")
        yp = fusedtok.attention_decode_paged(qt, pool_k, pool_v, table, plens)
        torch.cuda.synchronize()
        check("paged block-pool == contiguous", yp.cpu().numpy(), ref,
              tol=1e-4)

    print(SEP)
    print("attention prefill: fresh-sequence causal attention (S query rows)")
    q = rng.standard_normal((1, 8, 12, 16)).astype(np.float32)
    k = rng.standard_normal((1, 2, 12, 16)).astype(np.float32)
    v = rng.standard_normal((1, 2, 12, 16)).astype(np.float32)
    ref = np.zeros((1, 8, 12, 16), dtype=np.float64)
    for h in range(8):
        kvh = h // 4
        scores = q[0, h].astype(np.float64) @ k[0, kvh].astype(np.float64).T / 4.0
        for i in range(12):
            p = np.exp(scores[i, :i + 1] - scores[i, :i + 1].max())
            ref[0, h, i] = (p / p.sum()) @ v[0, kvh, :i + 1].astype(np.float64)
    out = fusedtok.attention_prefill(q, k, v, cuda=True) if have_cuda \
        else fusedtok.attention_prefill(q, k, v)
    check("causal diagonal vs eager", out, ref, tol=1e-4)
    if have_cuda and HAS_TORCH:
        qt, kt, vt = (torch.from_numpy(x).cuda() for x in (q, k, v))
        yt = fusedtok.attention_prefill(qt, kt, vt, causal=False)
        torch.cuda.synchronize()
        qh = qt.to(torch.bfloat16)
        yh = fusedtok.attention_prefill(qh, kt.to(torch.bfloat16),
                                        vt.to(torch.bfloat16), causal=False)
        torch.cuda.synchronize()
        ok = yh.dtype is torch.bfloat16 and torch.allclose(
            yh.float(), yt, rtol=2e-2, atol=2e-2)
        print(f"  {'prefill bf16 vs f32':<26} {'PASS' if ok else 'FAIL'}")
        ALL_OK &= ok
        ref_bi = np.zeros_like(ref)
        for h in range(8):
            kvh = h // 4
            scores = q[0, h].astype(np.float64) @ k[0, kvh].astype(np.float64).T / 4.0
            p = np.exp(scores - scores.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            ref_bi[0, h] = p @ v[0, kvh].astype(np.float64)
        check("bidirectional zero-copy", yt.cpu().numpy(), ref_bi, tol=1e-4)

    print(SEP)
    print("ALL PASS" if ALL_OK else "SOME CHECKS FAILED")
    return 0 if ALL_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
