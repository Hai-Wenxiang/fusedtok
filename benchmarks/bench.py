"""fusedtok benchmark suite: fusedtok zero-copy torch path vs PyTorch eager.

Timing uses CUDA events (wall-clock-free, WDDM-safe). Every
configuration is measured over THREE independent timed rounds (each with
its own warmup) and averaged; the per-round values travel with the JSON
so variance stays auditable.
Results are printed as a table, dumped to JSON, and rendered into ONE
single-panel chart per GPU next to this script under ../docs/.

The chart is a horizontal speedup chart sorted from best to worst: every
bar carries the fusedtok and reference microsecond values at the bar end,
a parity line marks 1.0x, and bar colors separate wins (green), ties
(amber) and losses (red). One figure per GPU, no panels, no log axes.

Usage:
    python benchmarks/bench.py [--iters N] [--out docs]

The torch "eager" references are the composite expressions an inference
loop would write by hand (or the closest native op); RoPE's reference
computes frequencies inside the timed region, matching fusedtok (no
precomputed cos/sin cache on either side). The attention references use
PRE-EXPANDED heads (no repeat_interleave inside the timed region), the
fair comparison for what fusedtok computes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import fusedtok  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = []


def bench(fn, iters, warmup=10, rounds=3):
    """Average GPU time per call in microseconds over `rounds`
    independent timed runs (each with its own warmup), via CUDA events.
    Returns (mean, per-round values) so the JSON dump keeps the variance
    auditable instead of hiding it inside a single number."""
    times = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters * 1000.0)
    return sum(times) / len(times), times


def record(op, shape, ft_fn, torch_fn, iters, bytes_moved=None, ops=None):
    t_ft, ft_rounds = bench(ft_fn, iters)
    t_tr, tr_rounds = bench(torch_fn, iters)
    row = {
        "op": op,
        "shape": shape,
        "fusedtok_us": round(t_ft, 2),
        "torch_us": round(t_tr, 2),
        "speedup": round(t_tr / t_ft, 2),
        "fusedtok_rounds_us": [round(v, 2) for v in ft_rounds],
        "torch_rounds_us": [round(v, 2) for v in tr_rounds],
    }
    if bytes_moved:
        row["bandwidth_gbs"] = round(bytes_moved / (t_ft * 1e-6) / 1e9, 1)
    if ops:
        # dense MACs*2/s - the honest metric for compute-bound ops (int8
        # GEMM); fusedtok value only, the torch reference tops ride in
        # the round data
        row["fusedtok_tops"] = round(ops / (t_ft * 1e-6) / 1e12, 1)
    RESULTS.append(row)
    spread = (max(ft_rounds) - min(ft_rounds)) / t_ft * 100.0
    extra = f"  {row['bandwidth_gbs']:6.1f} GB/s" if bytes_moved else ""
    if ops:
        extra += f"  {row['fusedtok_tops']:5.1f} TOPS"
    print(f"{op:<16} {shape:<18} fusedtok {t_ft:9.1f} us | torch {t_tr:9.1f} us"
          f" | {t_tr / t_ft:5.2f}x | round spread {spread:4.1f}%{extra}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "docs"))
    args = parser.parse_args()
    iters = args.iters

    if not fusedtok.cuda_available():
        print("No CUDA device - benchmark requires a GPU")
        return 1
    torch.cuda.init()
    dev_name = torch.cuda.get_device_name(0)
    print(f"fusedtok {fusedtok.__version__} | torch {torch.__version__} | {dev_name}\n")

    rng = np.random.default_rng(0)

    # --- norms / softmax / activations over [batch, hidden] -------------------
    for batch, hidden in [(256, 4096), (1024, 4096), (4096, 4096)]:
        x = torch.randn(batch, hidden, device="cuda")
        r = torch.randn(batch, hidden, device="cuda")
        w = torch.rand(hidden, device="cuda") + 0.5
        b = torch.zeros(hidden, device="cuda")
        shape = f"[{batch}x{hidden}]"
        io_bytes = (x.numel() * 4) * (2 if r is not None else 1) + x.numel() * 4

        record("rmsnorm+res", shape,
               lambda: fusedtok.rmsnorm(x, w, residual=r),
               lambda: (lambda v: v * torch.rsqrt(
                   v.pow(2).mean(-1, keepdim=True)) * w)(x + r),
               iters, bytes_moved=io_bytes)

        record("layernorm", shape,
               lambda: fusedtok.layernorm(x, w, b),
               lambda: torch.nn.functional.layer_norm(x, (hidden,), w, b),
               iters, bytes_moved=io_bytes)

        record("softmax", shape,
               lambda: fusedtok.softmax(x),
               lambda: torch.softmax(x, -1),
               iters, bytes_moved=io_bytes)

        record("silu", shape,
               lambda: fusedtok.silu(x),
               lambda: torch.nn.functional.silu(x),
               iters, bytes_moved=io_bytes)

        record("gelu(erf)", shape,
               lambda: fusedtok.gelu(x),
               lambda: torch.nn.functional.gelu(x),
               iters, bytes_moved=io_bytes)

        record("swiglu", shape,
               lambda: fusedtok.swiglu(x, r),
               lambda: torch.nn.functional.silu(x) * r,
               iters, bytes_moved=x.numel() * 4 * 3)

        record("add", shape,
               lambda: fusedtok.add(x, r),
               lambda: torch.add(x, r),
               iters, bytes_moved=x.numel() * 4 * 3)

    # --- RoPE (NeoX layout) over [seq, hidden] --------------------------------
    for seq, dim in [(512, 4096), (2048, 4096), (8192, 4096)]:
        q = torch.randn(seq, dim, device="cuda")
        k = torch.randn(seq, dim, device="cuda")
        half = dim // 2

        def torch_rope_neox(q, k, half):
            inv = torch.pow(10000.0, -2.0 * torch.arange(
                0, half, device="cuda", dtype=torch.float32) / dim)
            pos = torch.arange(seq, device="cuda", dtype=torch.float32)
            ang = pos[:, None] * inv[None, :]
            c, s = torch.cos(ang), torch.sin(ang)
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            qr = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], -1)
            kr = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], -1)
            return qr, kr

        record("rope neox q+k", f"[{seq}x{dim}]",
               lambda: fusedtok.rope(q, k, neox=True),
               lambda: torch_rope_neox(q, k, half),
               iters, bytes_moved=q.numel() * 4 * 2 * 2)

    # --- sampling over vocab ----------------------------------------------------
    for vocab in [32000, 131072]:
        logits = torch.randn(vocab, device="cuda")
        probs = torch.softmax(torch.randn(vocab, device="cuda") * 2.5, -1)
        record("topk k=50", f"[{vocab}]",
               lambda: fusedtok.topk(logits, 50),
               lambda: torch.topk(logits, 50),
               max(20, iters // 4))
        # the mid-k window: k large enough that the selection leaves the
        # in-block-sort path, small enough that torch's CUB select is
        # still in its fast regime - the honest comparison point for the
        # chunk/merge tail
        record("topk k=4096", f"[{vocab}]",
               lambda: fusedtok.topk(logits, 4096),
               lambda: torch.topk(logits, 4096),
               max(20, iters // 4))
        record("argmax", f"[{vocab}]",
               lambda: fusedtok.argmax(logits),
               lambda: int(logits.argmax()),
               max(20, iters // 2))

    # --- INT8 compute: qgemm vs cuBLASLt (torch._int_mm) -----------------------
    # The tensor-core path; TOPS is the metric, bandwidth is reported for
    # context. torch._int_mm is cuBLASLt's int8 GEMM - the hardest
    # possible comparison on this dtype.
    if hasattr(torch, "_int_mm"):
        for m, n, k in [(512, 4096, 4096), (4096, 4096, 4096),
                        (4096, 11008, 4096)]:
            a = torch.randint(-127, 128, (m, k), device="cuda",
                              dtype=torch.int8)
            b = torch.randint(-127, 128, (n, k), device="cuda",
                              dtype=torch.int8)
            record("int8 qgemm", f"[{m}x{n}x{k}]",
                   lambda: fusedtok.qgemm(a, 0.02, b, 0.01),
                   lambda: torch._int_mm(a, b.t()),
                   max(20, iters // 4),
                   bytes_moved=a.numel() + b.numel() + m * n * 4,
                   ops=2 * m * n * k)
        # per-channel weight scales (W8A8): the torch reference is what
        # real code writes - cuBLASLt _int_mm plus the scale broadcast
        # (that epilogue runs inside the fusedtok kernel for free)
        for m, n, k in [(512, 4096, 4096), (4096, 4096, 4096)]:
            a = torch.randint(-127, 128, (m, k), device="cuda",
                              dtype=torch.int8)
            b = torch.randint(-127, 128, (n, k), device="cuda",
                              dtype=torch.int8)
            sb = (torch.rand(n, device="cuda") + 0.5) * 0.1
            record("int8 qgemm pc", f"[{m}x{n}x{k}]",
                   lambda: fusedtok.qgemm_perchannel(a, 0.02, b, sb),
                   lambda: torch._int_mm(a, b.t()) * (0.02 * sb),
                   max(20, iters // 4),
                   bytes_moved=a.numel() + b.numel() + m * n * 4,
                   ops=2 * m * n * k)

    # --- attention (decode step over a kv-cache; fresh-sequence prefill) ------
    # references use PRE-EXPANDED heads (repeat_interleave outside the
    # timed region) - the fair fight; fusedtok reads the GQA cache as-is
    for cache_rows in [4096, 16384]:
        b, hq, hkv, d = 1, 32, 8, 128
        q = torch.randn(b, hq, d, device="cuda")
        k = torch.randn(b, hkv, cache_rows, d, device="cuda")
        v = torch.randn(b, hkv, cache_rows, d, device="cuda")
        kk = k.repeat_interleave(hq // hkv, dim=1)
        vv = v.repeat_interleave(hq // hkv, dim=1)
        kv_bytes = (2 * k.numel() + q.numel() * 2) * 4
        record("attn decode", f"T={cache_rows}",
               lambda: fusedtok.attention_decode(q, k, v),
               lambda: torch.nn.functional.scaled_dot_product_attention(
                   q.unsqueeze(2), kk, vv).squeeze(2),
               max(20, iters // 2), bytes_moved=kv_bytes)

    b, hq, hkv, d, s = 1, 32, 8, 128, 1024
    qp = torch.randn(b, hq, s, d, device="cuda")
    kp = torch.randn(b, hkv, s, d, device="cuda")
    vp = torch.randn(b, hkv, s, d, device="cuda")
    kkp = kp.repeat_interleave(hq // hkv, dim=1)
    vvp = vp.repeat_interleave(hq // hkv, dim=1)
    record("attn prefill", f"S={s}",
           lambda: fusedtok.attention_prefill(qp, kp, vp, causal=True),
           lambda: torch.nn.functional.scaled_dot_product_attention(
               qp, kkp, vvp, is_causal=True),
           max(10, iters // 8))

    # --- outputs ------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    # device-derived file slug: charts/JSONs from different GPUs never
    # overwrite each other
    dev_slug = (dev_name.lower()
                .replace("geforce", "").replace("nvidia", "")
                .replace(" ", ""))
    json_path = os.path.join(args.out, f"benchmark_{dev_slug}.json")
    payload = {
        "device": dev_name,
        "torch": torch.__version__,
        "fusedtok": fusedtok.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timing": "mean over 3 independent timed rounds; per-round values in each row",
        "results": RESULTS,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # ONE single-panel chart per GPU (device name in the file name so
    # different GPUs never overwrite each other): horizontal speedup
    # bars sorted best-first, absolute microsecond values at the bar
    # end, a 1.0x parity line, color-coded win/tie/loss. Each op is
    # represented by its LAST measured shape (the suites above measure
    # progressively larger shapes, so last = largest = the headline
    # configuration).
    rows_by_op = {}
    for row in RESULTS:
        rows_by_op[row["op"]] = row          # last shape per op wins
    ops = sorted(rows_by_op.values(),
                 key=lambda r: r["speedup"], reverse=True)

    n = len(ops)
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * n + 1.6), dpi=150)
    y = np.arange(n)[::-1]
    speedups = [r["speedup"] for r in ops]
    colors = ["#16a34a" if sp >= 1.05 else
              ("#d97706" if sp >= 0.9 else "#dc2626") for sp in speedups]
    bars = ax.barh(y, speedups, height=0.62, color=colors, zorder=3)

    # absolute values at each bar end (or inside when the bar is long)
    x_max = max(max(speedups) * 1.06, 1.35)
    for yi, r, sp in zip(y, ops, speedups):
        label = (f"{r['fusedtok_us']:.0f} vs {r['torch_us']:.0f} us"
                 f"  ({r['shape']})")
        inside = sp > x_max * 0.55
        ax.annotate(label,
                    xy=(sp, yi),
                    xytext=(6 if not inside else -6, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if not inside else "right",
                    fontsize=8.5,
                    color="#111827",
                    zorder=4)

    ax.axvline(1.0, color="#6b7280", linewidth=1.1,
               linestyle="--", zorder=2)
    ax.annotate("parity (1.0x)", xy=(1.0, y[-1] - 0.55),
                fontsize=8, color="#6b7280", ha="center")
    # ticks must ALWAYS include 1 (the parity anchor): matplotlib's
    # default step-2 ticks at x_max ~ 9.5 read 0/2/4/6/8 and lose the
    # reference point the whole chart is judged against
    if x_max <= 10:
        ticks = list(range(0, int(x_max) + 1))
    else:
        ticks = [0, 1] + list(range(2, int(x_max) + 1, 2))
    ax.set_xticks(ticks)
    ax.set_yticks(y)
    ax.set_yticklabels([r["op"] for r in ops], fontsize=9.5)
    ax.set_xlabel("speedup vs PyTorch reference (linear, higher is better)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.9, n - 0.1 + 0.55)
    ax.set_title(f"fusedtok vs PyTorch reference - {dev_name} "
                 f"(float32; largest shape per op; value = fusedtok vs ref)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png_path = os.path.join(args.out, f"benchmark_{dev_slug}.png")
    fig.savefig(png_path)

    print(f"\nwrote {json_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
