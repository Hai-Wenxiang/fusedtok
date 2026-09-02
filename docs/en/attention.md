# Attention operators

fusedtok ships four attention entry points: the decode-step workhorse
(`attention_decode`), its paged-cache variant (v1.2, with the
`kv_append_paged` write side), and a prefill convenience path
(`attention_prefill`). This page covers the layouts, the GQA mapping,
per-sequence lengths, and the honest performance framing.

**Other languages:** [中文：注意力算子](../zh/attention.md)

- [attention_decode - the decode step](#attention_decode---the-decode-step)
- [attention_decode_paged - the vLLM-style block pool](#attention_decode_paged---the-vllm-style-block-pool)
- [kv_append_paged - writing tokens into the pool](#kv_append_paged---writing-tokens-into-the-pool)
- [kv_append - writing tokens into the contiguous cache (v1.3)](#kv_append---writing-tokens-into-the-contiguous-cache-v13)
- [attention_prefill - fresh sequences](#attention_prefill---fresh-sequences)
- [Performance framing](#performance-framing)

## attention_decode - the decode step

```python
out = fusedtok.attention_decode(q, k_cache, v_cache, lens)
```

- `q`: `[B, Hq, D]` - the new token's query heads.
- `k_cache` / `v_cache`: `[B, Hkv, T, D]` - a **contiguous** kv-cache
  (`T` = allocated rows, not necessarily used rows).
- `lens`: optional `[B]` int32 - per-sequence valid cache length.
  `None` means all `T` rows are valid. Sequences with length 0 produce
  zero output rows.

GQA mapping is **contiguous groups**: q head `h` attends with kv head
`h // (Hq // Hkv)`; `Hq == Hkv` degenerates to plain MHA. This is the
layout LLaMA-style checkpoints use (a group's q heads sit next to each
other).

Constraints: `Hq % Hkv == 0`; `D` a multiple of 4 and at most 512.
Long caches split automatically flash-decoding style: the sequence is
cut into slices, partial results are computed in parallel, then
reduced - one call regardless of `T`. Scores never materialize; q/K/V
are each read exactly once.

`lens` values are validated for host-side inputs (lists, numpy, CPU
tensors) before the upload; a CUDA `lens` tensor is trusted as-is -
reading it back would sync the stream and break CUDA-graph capture
(the same trust boundary as a raw device pointer). CUDA-graph capture
works after a warm-up call outside the capture (the split workspace
must pre-exist).

Storage dtypes on the zero-copy path: float32, bfloat16, float16.
Half-precision caches halve the decode bytes (the decode-step
bottleneck) while the softmax stays float32; output matches the input
dtype.

## attention_decode_paged - the vLLM-style block pool

```python
out = fusedtok.attention_decode_paged(q, k_pool, v_pool, block_table, lens)
```

The same math over a **paged** cache layout - the memory shape real
serving stacks use:

- `k_pool` / `v_pool`: `[Nb, Hkv, P, D]` - a pool of fixed-size token
  blocks (`P` = tokens per block, read from the pool shape). Instead
  of a preallocated contiguous span per sequence, tokens live in
  blocks anywhere in the pool, so growing, shrinking and evicting
  sequences never fragments the cache.
- `block_table`: `[B, S]` int32 - sequence `b`'s token `t` lives in
  pool block `block_table[b, t // P]` at offset `t % P`. **Any valid
  table is honored** - non-monotonic block ids, shared blocks, holes;
  the kernel walks the indirection.
- `lens`: optional `[B]` - valid length per sequence (`None` = every
  sequence uses its full table width `S * P`).

Same GQA mapping, zero-row convention and dtype matrix as the
contiguous op; the split pipeline and its workspace are shared.
Paged-specific rules:

- GQA group sizes `Hq // Hkv` are limited to 1/2/4/8/16 (other
  divisors: use the contiguous op).
- Host-origin `block_table` and `lens` values are validated before the
  upload (`ValueError`); device-resident tensors are trusted (no
  stream sync).
- CUDA-graph capture needs one warm-up call outside the capture.

Measured cost of the indirection (3060, b=1, GQA 32/8, D=128, T=16384,
P=16): **1.06-1.07x** the contiguous op, with bit-identical output on
matching slice schedules.

## kv_append_paged - writing tokens into the pool

```python
fusedtok.kv_append_paged(k_pool, v_pool, block_table, k_new, v_new, lens)
```

The cache-write side of the paged loop: sequence `b`'s fresh rows
`k_new[b]` / `v_new[b]` (each `[Hkv, D]`) land at pool block
`block_table[b, lens[b] // P]`, offset `lens[b] % P`.

- **In place** (returns `None`). The block table belongs to the
  scheduler: this writes data into already-mapped blocks and never
  touches table entries.
- Host paths require float32 C-contiguous pools - a dtype/layout
  conversion would view a copy and silently drop the writes, so it is
  rejected with `TypeError` instead. The torch path supports the full
  f32/bf16/fp16 storage matrix.
- One tiny kernel, stream-ordered, graph-capturable (after the usual
  warm-up).

The typical loop: append at `lens[b]`, then decode with `lens + 1`:

```python
for step in range(n_steps):
    fusedtok.kv_append_paged(k_pool, v_pool, block_table,
                             k_new, v_new, lens)          # write at lens
    out = fusedtok.attention_decode_paged(q, k_pool, v_pool,
                                          block_table, lens + 1)
    lens += 1
    # ... k_new/v_new for the next token come from the model
```

## kv_append - writing tokens into the contiguous cache (v1.3)

```python
fusedtok.kv_append(k_cache, v_cache, k_new, v_new, lens)
```

The cache-write side of the CONTIGUOUS decode loop (the twin of
`kv_append_paged`): sequence `b`'s fresh rows `k_new[b]` / `v_new[b]`
(each `[Hkv, D]`) land at cache row `lens[b]` of `k_cache[b]` /
`v_cache[b]`.

- **In place** (returns `None`); `lens` is REQUIRED (the write position
  is each sequence's current length by definition).
- Host paths require float32 C-contiguous caches (a conversion would
  view a copy and silently drop the writes - rejected with
  `TypeError`). The torch path supports f32/bf16/fp16 storage.
- One tiny kernel, stream-ordered, CUDA-graph capturable.
- Host-origin `lens` values are validated in `[0, T)`; device-resident
  tensors are trusted (the standard zero-copy boundary).

The typical loop mirrors the paged one: append at `lens[b]`, then
decode with `lens + 1`:

```python
for step in range(n_steps):
    fusedtok.kv_append(k_cache, v_cache, k_new, v_new, lens)
    out = fusedtok.attention_decode(q, k_cache, v_cache, lens + 1)
    lens += 1
    # ... k_new/v_new for the next token come from the model
```

## attention_prefill - fresh sequences

```python
ctx = fusedtok.attention_prefill(q_all, k_all, v_all, causal=True)
```

- `q`: `[B, Hq, S, D]`, `k` / `v`: `[B, Hkv, S, D]`; query row `i`
  attends to key rows `[0, i]` when `causal=True` (the prefill
  diagonal), to all rows when `causal=False`.
- Same GQA/dtype/dim rules as decode.

Honest scope: this is the **convenience path** - one tiled kernel, no
tensor cores. It exists so small prefills and mixed workloads stay
inside fusedtok; heavyweight prefill belongs to SDPA /
FlashAttention (the benchmark tables carry the honest ~0.45x ratio).

## Performance framing

Decode attention is **bandwidth-bound**: every token streams the whole
kv-cache once. What to expect:

- f32 decode runs at effective-bandwidth parity or better vs SDPA at
  long caches (the README tables show up to 8.79x on an RTX 3060 at
  T=16384 - the reference pays head expansion or small-query
  inefficiency there).
- bf16/fp16 caches halve the bytes. At batch 1 the kernel is
  latency-bound, so the absolute win is modest and grows with batch.
- The paged indirection costs 1.06-1.07x over the contiguous op.
- Prefill is deliberately not competitive with flash backends.

See [benchmarks.md](benchmarks.md) for the measurement protocol and
how to reproduce the tables.
