"""fusedtok benchmark suite: fusedtok zero-copy torch path vs PyTorch eager.

Timing uses CUDA events (wall-clock-free, WDDM-safe). Each configuration is
warmed up, then measured over enough iterations to average out jitter.
Results are printed as a table, dumped to JSON, and rendered into a grouped
bar chart next to this script under ../docs/.

The chart uses LINEAR time axes on two panels (fast ops / slow ops, split
at the largest gap in the sorted times): every bar is annotated with its
direct microsecond value and the per-op speedup, so no log-scale powers
of ten have to be decoded by eye.

Usage:
    python benchmarks/bench.py [--iters N] [--out docs]

The torch "eager" references are the composite expressions an inference
loop would write by hand (or the closest native op); RoPE's reference
computes frequencies inside the timed region, matching fusedtok (no
precomputed cos/sin cache on either side).
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


def bench(fn, iters, warmup=10):
    """Average GPU time per call in microseconds, via CUDA events."""
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
    return start.elapsed_time(end) / iters * 1000.0


def record(op, shape, ft_fn, torch_fn, iters, bytes_moved=None):
    t_ft = bench(ft_fn, iters)
    t_tr = bench(torch_fn, iters)
    row = {
        "op": op,
        "shape": shape,
        "fusedtok_us": round(t_ft, 2),
        "torch_us": round(t_tr, 2),
        "speedup": round(t_tr / t_ft, 2),
    }
    if bytes_moved:
        row["bandwidth_gbs"] = round(bytes_moved / (t_ft * 1e-6) / 1e9, 1)
    RESULTS.append(row)
    extra = f"  {row['bandwidth_gbs']:6.1f} GB/s" if bytes_moved else ""
    print(f"{op:<16} {shape:<18} fusedtok {t_ft:9.1f} us | torch {t_tr:9.1f} us"
          f" | {t_tr / t_ft:5.2f}x{extra}")
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
        record("argmax", f"[{vocab}]",
               lambda: fusedtok.argmax(logits),
               lambda: int(logits.argmax()),
               max(20, iters // 2))

    # --- outputs ------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "benchmark_results.json")
    payload = {
        "device": dev_name,
        "torch": torch.__version__,
        "fusedtok": fusedtok.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": RESULTS,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Grouped bar chart with LINEAR axes, one group per op (largest
    # shape). Times span two orders of magnitude, so the ops split into
    # two panels at the largest gap of the sorted per-op maxima - each
    # panel stays directly comparable in plain microseconds and every
    # bar carries its value plus the speedup.
    ops, ft_times, tr_times = [], [], []
    for row in RESULTS:
        if row["shape"] in ("[4096x4096]", "[8192x4096]", "[131072]"):
            label = row["op"]
            if label not in ops:
                ops.append(label)
                ft_times.append(row["fusedtok_us"])
                tr_times.append(row["torch_us"])

    # natural-break split: sort ops by their slower bar, cut where the
    # ratio between consecutive maxima is largest (both sides non-empty)
    order = sorted(range(len(ops)), key=lambda i: max(ft_times[i],
                                                      tr_times[i]))
    split_at = order[0]
    best_ratio = 1.0
    for a, b in zip(order, order[1:]):
        hi = max(ft_times[b], tr_times[b])
        lo = max(ft_times[a], tr_times[a])
        if hi / lo > best_ratio:
            best_ratio = hi / lo
            split_at = b
    slow_idx = set(i for i in order[order.index(split_at):])
    groups = [("slow ops (linear us)", [i for i in range(len(ops))
                                         if i in slow_idx]),
              ("fast ops (linear us)", [i for i in range(len(ops))
                                        if i not in slow_idx])]

    # output filename derives from the actual device so charts from
    # different GPUs never overwrite each other
    dev_slug = dev_name.lower().replace(" ", "").replace("geforce", "")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.5), dpi=150)
    width = 0.38
    for ax, (title, idxs) in zip(axes, groups):
        if not idxs:
            ax.set_visible(False)
            continue
        x = np.arange(len(idxs))
        fts = [ft_times[i] for i in idxs]
        trs = [tr_times[i] for i in idxs]
        b1 = ax.bar(x - width / 2, fts, width, label="fusedtok",
                    color="#3b82f6")
        b2 = ax.bar(x + width / 2, trs, width, label="PyTorch eager",
                    color="#9ca3af")
        ax.set_ylabel("time per call (us)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([ops[i] for i in idxs], rotation=18, ha="right")
        # annotate every bar with its direct microsecond value; the
        # speedup badge above each pair carries the ratio
        for bars, vals in ((b1, fts), (b2, trs)):
            for rect, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}",
                            xy=(rect.get_x() + rect.get_width() / 2, v),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7)
        # speedup badge centered above each op pair
        for j, i in enumerate(idxs):
            sp = tr_times[i] / ft_times[i]
            ax.annotate(f"{sp:.2f}x",
                        xy=(j, max(ft_times[i], tr_times[i])),
                        xytext=(0, 16), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#166534" if sp >= 1.05 else
                        ("#92400e" if sp >= 0.9 else "#991b1b"))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.22)
    fig.suptitle(f"fusedtok vs PyTorch eager - {dev_name} "
                 f"(float32; largest shape per op)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png_path = os.path.join(args.out, f"benchmark_{dev_slug}.png")
    fig.savefig(png_path)

    print(f"\nwrote {json_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
