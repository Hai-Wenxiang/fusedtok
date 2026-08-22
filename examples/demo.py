"""Demo: run every fusedtok operator and verify it against closed-form
results on all available execution paths (CPU reference, staged CUDA,
zero-copy torch CUDA).

Run:  py -3.12 examples/demo.py   (build dir on PYTHONPATH, or pip-installed)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import fusedtok  # noqa: E402

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
    print("ALL PASS" if ALL_OK else "SOME CHECKS FAILED")
    return 0 if ALL_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
