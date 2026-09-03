"""fusedtok benchmark suite: fusedtok zero-copy torch path vs PyTorch eager.

Timing uses CUDA events (wall-clock-free, WDDM-safe). Every
configuration is measured over THREE independent timed rounds (each with
its own warmup) and averaged; the per-round values travel with the JSON
so variance stays auditable.
Results are printed as a table, dumped to JSON, and rendered into ONE
single-panel chart per GPU under ../docs/benchmarks/.

The chart is a horizontal speedup chart sorted from best to worst: every
bar carries the fusedtok and reference microsecond values at the bar end,
a parity line marks 1.0x, and bar colors separate wins (green), ties
(amber) and losses (red). One figure per GPU, no panels, no log axes.

Usage:
    python benchmarks/bench.py [--iters N] [--out docs/benchmarks]

The torch "eager" references are the composite expressions an inference
loop would write by hand (or the closest native op); RoPE's reference
computes frequencies inside the timed region, matching fusedtok (no
precomputed cos/sin cache on either side). The attention references use
PRE-EXPANDED heads (no repeat_interleave inside the timed region), the
fair comparison for what fusedtok computes.

Sampling rows carry a timing asymmetry that is CONSERVATIVE against
fusedtok: its samplers return a Python int (one device synchronisation
per call, inside the timed loop) while the torch composites return
device tensors whose draws synchronise once per timed round.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

try:
    import fusedtok  # noqa: E402  (an installed wheel wins, if any)
except ImportError:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "python"))
    import fusedtok  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = []

# one constant per parameter that appears on BOTH sides of a comparison
# (fusedtok call and torch reference) so the two can never drift apart
TOPK_SMALL = 50          # small-k selection / sampling window
TOPK_MID = 4096          # mid-k chunk/merge tail
TOPP_P = 0.9             # nucleus mass threshold
MINP = 0.05              # min-p fraction of the max probability


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
    print(f"{op:<16} {shape:<18} fusedtok {t_ft:9.1f} us"
          f" | torch {t_tr:9.1f} us"
          f" | {t_tr / t_ft:5.2f}x | round spread {spread:4.1f}%{extra}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "docs", "benchmarks"))
    args = parser.parse_args()
    iters = args.iters

    if not fusedtok.cuda_available():
        print("No CUDA device - benchmark requires a GPU")
        return 1
    torch.cuda.init()
    dev_name = torch.cuda.get_device_name(0)
    print(f"fusedtok {fusedtok.__version__} | torch {torch.__version__}"
          f" | {dev_name}\n")

    # torch's global RNG drives the sampling logits; seeding it keeps the
    # DRAWS identical across runs/machines so JSON numbers stay comparable
    # (unseeded, the peaked-row nucleus width varies per run and swings
    # the sample_topp timing 2-3x with the same code)
    torch.manual_seed(0)

    # --- norms / softmax / activations over [batch, hidden] ---------------
    for batch, hidden in [(256, 4096), (1024, 4096), (4096, 4096)]:
        x = torch.randn(batch, hidden, device="cuda")
        r = torch.randn(batch, hidden, device="cuda")
        w = torch.rand(hidden, device="cuda") + 0.5
        b = torch.zeros(hidden, device="cuda")
        shape = f"[{batch}x{hidden}]"
        f32 = x.numel() * 4
        # honest bytes per op: rmsnorm+res and swiglu stream three
        # tensors (x, r/x2, y); layernorm/softmax/silu/gelu only two
        # (x in, y out) - the 1.2.0 suite billed all of them for three
        # and over-reported their GB/s by 1.5x
        io3 = 3 * f32
        io2 = 2 * f32

        record("rmsnorm+res", shape,
               lambda: fusedtok.rmsnorm(x, w, residual=r),
               lambda: (lambda v: v * torch.rsqrt(
                   v.pow(2).mean(-1, keepdim=True)) * w)(x + r),
               iters, bytes_moved=io3)

        record("layernorm", shape,
               lambda: fusedtok.layernorm(x, w, b),
               lambda: torch.nn.functional.layer_norm(x, (hidden,), w, b),
               iters, bytes_moved=io2)

        record("softmax", shape,
               lambda: fusedtok.softmax(x),
               lambda: torch.softmax(x, -1),
               iters, bytes_moved=io2)

        record("silu", shape,
               lambda: fusedtok.silu(x),
               lambda: torch.nn.functional.silu(x),
               iters, bytes_moved=io2)

        record("gelu(erf)", shape,
               lambda: fusedtok.gelu(x),
               lambda: torch.nn.functional.gelu(x),
               iters, bytes_moved=io2)

        record("swiglu", shape,
               lambda: fusedtok.swiglu(x, r),
               lambda: torch.nn.functional.silu(x) * r,
               iters, bytes_moved=x.numel() * 4 * 3)

        record("add", shape,
               lambda: fusedtok.add(x, r),
               lambda: torch.add(x, r),
               iters, bytes_moved=x.numel() * 4 * 3)

    # --- RoPE (NeoX layout) over [seq, hidden] -----------------------------
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

    # --- sampling over vocab -----------------------------------------------
    for vocab in [32000, 131072]:
        logits = torch.randn(vocab, device="cuda")
        probs = torch.softmax(torch.randn(vocab, device="cuda") * 2.5, -1)
        record("topk k=50", f"[{vocab}]",
               lambda: fusedtok.topk(logits, TOPK_SMALL),
               lambda: torch.topk(logits, TOPK_SMALL),
               max(20, iters // 4))
        # the mid-k window: k large enough that the selection leaves the
        # in-block-sort path, small enough that torch's CUB select is
        # still in its fast regime - the honest comparison point for the
        # chunk/merge tail
        record("topk k=4096", f"[{vocab}]",
               lambda: fusedtok.topk(logits, TOPK_MID),
               lambda: torch.topk(logits, TOPK_MID),
               max(20, iters // 4))
        # fused top-k sampling vs the composite an inference loop would
        # write (topk + softmax + multinomial). SEMANTICS NOTE: the torch
        # side draws from its own global RNG - the timing is the fair
        # part, not the seed determinism (fusedtok is seed-reproducible,
        # the composite is not).
        def torch_sample_topk():
            v, i = torch.topk(logits, TOPK_SMALL)
            return i[torch.multinomial(torch.softmax(v, -1), 1)]
        record("sample_topk k=50", f"[{vocab}]",
               lambda: fusedtok.sample_topk(logits, TOPK_SMALL),
               torch_sample_topk,
               max(20, iters // 4))

        # fused min-p sampling (v1.3) vs the composite an inference loop
        # would write (softmax + boolean mask + renormalize + multinomial;
        # same RNG caveat as the topk row). min_p=0.05 on plain randn
        # keeps a mid-sized value-threshold nucleus - the typical
        # creative-generation setting.
        def torch_sample_minp(src):
            p = torch.softmax(src, -1)
            sel = p >= MINP * p.max()
            return torch.multinomial(p * sel, 1)
        record("sample_minp p=0.05", f"[{vocab}]",
               lambda: fusedtok.sample_minp(logits, MINP),
               lambda: torch_sample_minp(logits),
               max(20, iters // 4))

        def torch_sample_topp_flat(src):
            probs = torch.softmax(src, -1)
            sp, si = torch.sort(probs, descending=True)
            cum = sp.cumsum(-1)
            sel = si[cum - sp < TOPP_P]
            return sel[torch.multinomial(probs[sel], 1)]

        # TWO regimes, honestly labeled: PEAKED logits (a dominant token -
        # the decode-time case, covered by the first sampling window) and
        # FLAT logits (worst case: the nucleus spans ~p*n of the vocab, so
        # the first window cannot cover it). v1.2 notes: the peaked spike
        # is +20 because +10 at n=131072 sits ON the coverage boundary -
        # the max-normalized tail carries ~11% of the mass and p=0.9
        # needs 10%, so the row was a per-seed coin flip between the
        # fast first-window path and a several-thousand-token nucleus
        # (measured: nucleus 6855 wide on one seed, instant on another).
        # +20 makes the top-1 mass ~1/T with a vanishing tail: the row
        # measures what its label says. The "flat" row uses near-uniform
        # logits (randn * 1e-3) for the same reason (v1.1's plain randn
        # nucleus was only ~8k wide - a mid-tail, not the worst case).
        peaked = torch.randn(vocab, device="cuda")
        peaked[peaked.argmax()] += 20.0
        record("sample_topp peaked", f"[{vocab}]",
               lambda: fusedtok.sample_topp(peaked, TOPP_P),
               lambda: torch_sample_topp_flat(peaked),
               max(20, iters // 4))
        # the same peaked logits through min-p: the nucleus is a handful
        # of tokens and the value-threshold walk stops immediately - the
        # decode-time min-p case (this is the row the README prose cites
        # for min-p's win scenario)
        record("sample_minp peaked", f"[{vocab}]",
               lambda: fusedtok.sample_minp(peaked, MINP),
               lambda: torch_sample_minp(peaked),
               max(20, iters // 4))
        flat_logits = torch.randn(vocab, device="cuda") * 1e-3
        record("sample_topp flat (worst)", f"[{vocab}]",
               lambda: fusedtok.sample_topp(flat_logits, TOPP_P),
               lambda: torch_sample_topp_flat(flat_logits),
               max(10, iters // 6))

        record("argmax", f"[{vocab}]",
               lambda: fusedtok.argmax(logits),
               lambda: int(logits.argmax()),
               max(20, iters // 2))

    # --- INT8 compute: qgemm vs cuBLASLt (torch._int_mm) ------------------
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

    # --- attention (decode step over a kv-cache; fresh-sequence prefill) --
    # references use PRE-EXPANDED heads (repeat_interleave outside the
    # timed region) - the fair fight; fusedtok reads the GQA cache as-is.
    # bf16 rows (v1.1): half-width kv-cache = half the bytes; the SDPA
    # reference runs the same dtype so the fight stays fair.
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
        qb = q.to(torch.bfloat16)
        kb = k.to(torch.bfloat16)
        vb = v.to(torch.bfloat16)
        kkb = kk.to(torch.bfloat16)
        vvb = vv.to(torch.bfloat16)
        record("attn decode bf16", f"T={cache_rows}",
               lambda: fusedtok.attention_decode(qb, kb, vb),
               lambda: torch.nn.functional.scaled_dot_product_attention(
                   qb.unsqueeze(2), kkb, vvb).squeeze(2),
               max(20, iters // 2),
               bytes_moved=(2 * kb.numel() + qb.numel() * 2) * 2)
        # paged variant (v1.2): the same cache content in a block pool
        # (P=16) walked through a shuffled block table; the SDPA
        # reference reads the materialized contiguous cache - the row
        # measures the paging indirection against the same math
        if cache_rows == 16384:
            page = 16
            width = cache_rows // page
            perm = torch.randperm(width)
            pool_k = torch.zeros(width, hkv, page, d, device="cuda")
            pool_v = torch.zeros(width, hkv, page, d, device="cuda")
            for sidx in range(width):
                pool_k[perm[sidx]] = k[:, :, sidx * page:(sidx + 1) * page]
                pool_v[perm[sidx]] = v[:, :, sidx * page:(sidx + 1) * page]
            table = perm.unsqueeze(0).to(torch.int32)
            record("attn decode paged", f"T={cache_rows}",
                   lambda: fusedtok.attention_decode_paged(q, pool_k,
                                                           pool_v, table),
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

    # --- kv-cache append (v1.3): the contiguous cache-write side of the
    # decode loop vs the advanced-indexing scatter a hand-written loop
    # would use (index_copy_ cannot express "sequence b writes row
    # lens[b]"). The op is tiny and launch-bound at these sizes - the
    # point is parity-level cost per decode step, not a headline speedup.
    ab, ahkv, arows, ad = 8, 8, 4096, 128
    akc = torch.randn(ab, ahkv, arows, ad, device="cuda")
    avc = torch.randn_like(akc)
    akn = torch.randn(ab, ahkv, ad, device="cuda")
    avn = torch.randn(ab, ahkv, ad, device="cuda")
    alens = torch.randint(0, arows, (ab,), dtype=torch.int32,
                          device="cuda")

    def torch_kv_append():
        bi = torch.arange(ab, device="cuda")
        pos = alens.long()
        akc[bi, :, pos] = akn
        avc[bi, :, pos] = avn

    record("kv_append", f"B={ab} T={arows}",
           lambda: fusedtok.kv_append(akc, avc, akn, avn, alens),
           torch_kv_append,
           max(50, iters // 2),
           bytes_moved=4 * ab * ahkv * ad * 4)

    # --- outputs -----------------------------------------------------------
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
        "timing": ("mean over 3 independent timed rounds; "
                   "per-round values in each row"),
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
                 "(largest shape per op; dtype as noted per row)",
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
