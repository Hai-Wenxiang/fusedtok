# Sampling and selection

The selection operators (top-k, top-p, argmax) and the fused samplers
(`sample_topp`, `sample_topk`, `sample_minp`, `decode_step`) share one
pipeline and one determinism contract. This page explains both, plus
the exact boundary where CPU and GPU draws may differ.

**Other languages:** [中文：采样与选择](../zh/sampling.md)

- [Selection operators](#selection-operators)
- [The fused samplers](#the-fused-samplers)
- [sample_minp - threshold-by-max sampling (v1.3)](#sample_minp---threshold-by-max-sampling-v13)
- [The same-token guarantee](#the-same-token-guarantee)
- [Flat distributions - the honest worst case](#flat-distributions---the-honest-worst-case)
- [How the pipeline works](#how-the-pipeline-works)

## Selection operators

All selection ops resolve ties toward the **earliest index**.

```python
i = fusedtok.argmax(logits)                # earliest index on ties
vals, idxs = fusedtok.topk(logits, 50)     # descending, (values, indices)
vals, idxs = fusedtok.topp(probs, 0.9)     # input = probabilities
```

- `topk` takes raw scores, `k` in `[0, n]`.
- `topp` takes **probabilities** (already softmaxed), `p` in `(0, 1]`;
  it returns the smallest top-p set, crossing element included.

`argmax` returns one `int` - including the single device-to-host
readback that a host return value inherently needs. On the zero-copy
path `argmax` keeps two self-resetting workspace slots so the hot path
is one kernel launch and no extra allocation (v1.2).

## The fused samplers

The samplers run logits -> token in one GPU round trip:

```python
tok = fusedtok.sample_topp(logits, p=0.9, temperature=0.8, seed=step)
tok = fusedtok.sample_topk(logits, k=50, temperature=0.8, seed=step)
tok = fusedtok.decode_step(logits, history, penalty=1.1,
                           p=0.9, temperature=0.8, seed=step)
```

- `sample_topp`: softmax(logits / T) -> nucleus cut with a
  **global-mass** threshold -> inverse-CDF draw. If the first
  candidate window does not cover the nucleus, the window widens -
  adaptively since v1.2 (see below).
- `sample_topk`: softmax(logits / T) -> keep k -> renormalize **within
  the k survivors** -> draw. The window is covered by construction, so
  there is no threshold and no widening loop. `k = 1` is exactly
  greedy; `k >= vocab` samples the whole distribution.
- `decode_step`: CTRL-style `repetition_penalty` over `history`, then
  temperature, then nucleus sampling - one call, one readback, and the
  identical result as composing the three ops in that order with the
  same seed.
- `repetition_penalty(logits, token_ids, penalty)` is also exposed
  standalone: positive logits are divided by `penalty`, negative
  logits multiplied (`penalty=1.0` disables it).

Sampling is **deterministic per seed**: the draw uses a splitmix-style
hash uniform (reproducible, not cryptographically secure). Host-origin
token ids are validated against the vocab before upload; CUDA id
tensors are trusted (no stream sync).

## sample_minp - threshold-by-max sampling (v1.3)

```python
tok = fusedtok.sample_minp(logits, min_p=0.1, temperature=0.8, seed=step)
```

Min-p (from "Turning Up the Heat: Min-p Sampling for Creative and
Coherent LLM Outputs", Nguyen et al. 2024 - already deployed in
llama.cpp and vLLM) truncates by a value threshold relative to the peak
instead of a cumulative mass: keep every token whose probability is at
least `min_p` times the maximum probability, renormalize within that
nucleus, draw with the same seeded hash.

- `min_p` must be in `(0, 1]` (`ValueError` otherwise); `temperature`
  must be greater than 0.
- Adaptive by construction: peaked decode logits get a tiny nucleus,
  near-uniform logits a wide one - no window guessing on the caller's
  side.
- `min_p = 1.0` keeps only the tokens at the maximum (unique max =
  exactly greedy; tied maxima stay tied in the draw).
- Deterministic per seed, same RNG and same-token guarantee as the
  other samplers (CPU exact-`exp` vs GPU `__expf` boundary caveat
  included).
- Implementation note: the exp column is already max-normalized
  (`exps[0] == 1.0` exactly), so the nucleus is simply the prefix
  cut at the first element below `min_p` - no global-mass reduction
  is needed, and the serial walk inherits the v1.3 checkpoint
  bisection.

## The same-token guarantee

For a fixed seed, the CPU / staged / zero-copy paths draw the **same
token** - this is tested, not aspirational. The one documented
boundary: the CPU reference uses exact `exp`, the GPU kernels use
`__expf` (~2 ulp). A draw that lands exactly on an exp-rounding
boundary of the CDF can differ by one element between CPU and GPU -
both draws are valid samples of the distribution.

At very large vocabularies with near-uniform logits, this boundary
effect occurs at scale: the tiny per-element rounding differences
accumulate along the strictly-sequential CDF walk, so CPU and GPU may
pick tokens a small **rank window** apart (measured ~14 ranks at
n=152064; ~1 at 32k vocabularies). The GPU result itself is always
bit-identical per seed, and the window-widening schedule never changes
the sampled token.

Why not fix it? The strictly-sequential float adds are the determinism
contract - parallelizing the sum would change every token ever drawn
with any given seed. The v1.2 optimization batch kept the add order
untouched and only pipelined the loads (an 8.5x cut of the flat-case
worst time with bit-identical tokens).

## Flat distributions - the honest worst case

When the nucleus spans most of the vocabulary (uniform-ish logits),
`sample_topp` must effectively order the whole thing, and torch's
fully parallel sort stays ahead - the benchmark tables carry the
honest 0.17-0.37x. v1.2 cut this worst case ~8.5x (18.2ms -> 2.2ms at
n=131072 on a 3060) with three contract-preserving changes:

1. **Adaptive widening jump** - a failed window attempt leaves its
   cumulated mass in a workspace slot; combined with the global
   softmax total this yields the necessary bound
   `w >= W * p * T / C` (every element past rank W is at most C/W),
   so flat distributions jump straight to (nearly) the full
   vocabulary instead of stepping the widening ladder.
2. **Full-vocabulary fast path** - when the window equals the
   vocabulary, the radix selection is pure waste (every key
   survives); a plain parallel pack replaces it.
3. **Batched serial walk** - the contract-bearing sequential adds stay
   sequential, but their loads are pipelined in branch-free batches
   (the naive one-load-one-add walk was pure L2 latency - 97% of the
   flat-case time).

Real decode logits are peaked; the flat case is the worst case, not
the typical one. v1.3 added one more token-preserving cut: walk 1
records its prefix sums at batch boundaries into shared memory and
walk 2 binary-searches those checkpoints instead of scanning from
index 0 (the resumed prefix is bit-identical - same adds in the same
order), taking the flat worst case down another ~1.6x.

## How the pipeline works

Background for the curious (nothing here is needed to use the ops):

- **Arrival-ticket radix rounds**: candidate keys are histogrammed in
  radix rounds; the last block to arrive per round decides the
  boundary - plain kernel launches, no grid-wide barriers, no
  cooperative launch.
- **Early-exit compaction**: when a radix boundary bin holds at most
  1024 candidates, one block sorts the survivors in shared memory
  instead of further rounds.
- **Merge-ladder sort**: for larger k, per-block chunk sorts merge
  through levels with one launch per level.
- **Cached CUDA graph**: the whole sequence is captured once per
  (n, k, mode) and replays as a single graph launch; per-call
  pointers travel through a device-side argument block, so replays
  pick up fresh tensors.
