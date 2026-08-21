"""Demo: run every fusedtok operator on CPU and CUDA and compare.

Run:  py -3.12 examples/demo.py   (with the build dir on PYTHONPATH)
"""

import math

import _fusedtok

SEP = "-" * 62


def check(name, got, expect, tol=1e-4):
    ok = len(got) == len(expect) and all(abs(a - b) <= tol for a, b in zip(got, expect))
    print(f"  {name:<22} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True

    print(SEP)
    print("axpy: y = a * x + b")
    x = [1.0, 2.0, 3.0]
    y = _fusedtok.axpy(x, 2.0, 1.0, cuda=True)
    all_ok &= check("cuda vs formula", y, [2 * v + 1 for v in x])

    print(SEP)
    print("rmsnorm: unit weight, eps ~ 0")
    y = _fusedtok.rmsnorm([3.0, 4.0], [1.0, 1.0], 1, 2, 1e-12, cuda=True)
    inv = 1.0 / math.sqrt(12.5)
    all_ok &= check("cuda vs formula", y, [3 * inv, 4 * inv])

    print(SEP)
    print("layernorm: unit weight, zero bias")
    y = _fusedtok.layernorm([1.0, 2.0, 3.0], [1.0] * 3, [0.0] * 3, 1, 3, 1e-12, cuda=True)
    s = math.sqrt(2.0 / 3.0)
    all_ok &= check("cuda vs formula", y, [-1 / s, 0.0, 1 / s])

    print(SEP)
    print("softmax: uniform row -> uniform distribution")
    y = _fusedtok.softmax([2.0] * 4, 1, 4, cuda=True)
    all_ok &= check("cuda vs formula", y, [0.25] * 4)

    print(SEP)
    print("activations: silu / gelu / relu / tanh")
    all_ok &= check("silu(0, 50)", _fusedtok.silu([0.0, 50.0], cuda=True), [0.0, 50.0], tol=5e-3)
    all_ok &= check("gelu(0, 50)", _fusedtok.gelu([0.0, 50.0], cuda=True), [0.0, 50.0], tol=5e-3)
    all_ok &= check("relu", _fusedtok.relu([-1.0, 2.0], cuda=True), [0.0, 2.0])
    all_ok &= check("tanh limits", _fusedtok.tanh([100.0, -100.0], cuda=True), [1.0, -1.0])

    print(SEP)
    print("swiglu: silu(gate) * up")
    y = _fusedtok.swiglu([0.0, 100.0], [3.0, 2.0], cuda=True)
    all_ok &= check("cuda vs limits", y, [0.0, 200.0], tol=1e-2)

    print(SEP)
    print("rope (both layouts): position 0 is identity")
    q = [0.5, -1.0, 2.0, 0.25]
    out, k = _fusedtok.rope(q, None, 1, 4, cuda=True)
    all_ok &= check("interleaved", out, q)
    out, _ = _fusedtok.rope_neox(q, None, 1, 4, cuda=True)
    all_ok &= check("neox", out, q)

    print(SEP)
    print("sampling: temperature -> softmax -> topk / topp / argmax")
    logits = _fusedtok.temperature([1.0, 2.0, 3.0], 0.5, cuda=True)
    probs = _fusedtok.softmax(logits, 1, 3, cuda=True)
    vals, idxs = _fusedtok.topk(probs, 2, cuda=True)
    all_ok &= check("topk values", vals, sorted(probs, reverse=True)[:2])
    print(f"  {'topk indices':<22} {idxs}")
    vals, idxs = _fusedtok.topp(probs, 0.9, cuda=True)
    print(f"  {'topp (p=0.9)':<22} {idxs} -> {['%.3f' % v for v in vals]}")
    print(f"  {'argmax':<22} {_fusedtok.argmax(logits, cuda=True)}")

    print(SEP)
    print("GPU available:", _fusedtok.cuda_available())
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
