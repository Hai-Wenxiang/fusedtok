# Benchmarks: protocol and how to read them

fusedtok's published numbers come from one script, one protocol and an
honesty policy. This page documents all three so you can reproduce
every table row and interpret the losses as well as the wins.

**Other languages:** [中文：基准测试](../zh/benchmarks.md)

## The protocol

```bash
python benchmarks/bench.py            # full suite, a few minutes
```

- **CUDA-event timing** on the zero-copy torch path, never wall clock
  (Windows WDDM makes host-side timing of GPU work meaningless).
- Every configuration runs **3 independent timed rounds**, each with
  its own warmup; the tables report the **mean** and the JSON keeps
  every per-round value, so variance stays auditable.
- The torch references are the composite expressions an inference
  loop would actually write (eager composites; attention references
  use **pre-expanded** heads - `repeat_interleave` outside the timed
  region - because that is the fair comparison for what fusedtok
  computes).
- Sampling rows measure **fixed logits**: the bench seeds torch's RNG
  so the draws are identical across runs and machines. Since 1.2 the
  peaked row uses a +20 spike and the flat row uses near-uniform
  logits, so both regimes are deterministically what their labels say.
- `--iters N` adjusts the per-round iteration count (60 for the
  published runs, 100 by default for quick checks).

Output lands in `docs/benchmarks/`: one JSON + one single-panel
speedup chart per GPU (file names carry the device name). The README
benchmark tables are generated from these JSONs - same source, same
rounding.

## How to read the tables

- **Largest shape per op** is what the headline tables show; smaller
  shapes live in the JSON (on Blackwell the small-shape wins are
  bigger - launch overhead fades as shapes grow).
- **(honest)** marks rows where fusedtok loses: attention_prefill vs
  SDPA's flash backend (~0.45x), the INT8 GEMMs vs cuBLASLt
  (0.40-0.57x), flat-distribution sample_topp vs torch's fully
  parallel sort (0.17-0.29x) and the wide-nucleus sample_minp row
  (0.2-0.31x - two widening retries plus a 64k sort; the torch
  boolean-mask composite never sorts). These are design-scope
  statements, not measurement noise - the [topic pages](usage.md)
  explain each one.
- The **bandwidth column** (GB/s) counts only the bytes the op must
  move (e.g. 2 tensors for softmax, 3 for rmsnorm+residual); the
  **TOPS** figure on INT8 rows is dense-MACs-times-two per second.
- **argmax** rows are event-timed over a host-synchronized call and
  swing on WDDM (0.73-1.24x across runs); wall-clock probes with the
  sync excluded from the timed loop measure 1.12x (3060) / 0.96x
  (5060 Ti). The row documents both.
- Composite **sampling references** (sort+mask+multinomial and
  friends) themselves swing ~15% run-to-run on WDDM; per-round values
  in the JSON show which side moved.

## Current published numbers

See the benchmark section of the
[README](../../README.md#benchmarks) for the tables on the RTX 3060
and RTX 5060 Ti. To reproduce:

```bash
git clone https://github.com/Hai-Wenxiang/fusedtok.git
cd fusedtok
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
PYTHONPATH=$PWD/build python benchmarks/bench.py --iters 60
```

(On Windows run the cmake configure inside a VS developer prompt; see
[CONTRIBUTING](../../CONTRIBUTING.md).)

## The honesty policy

1. Every number in the README is regenerable from the JSONs in the
   same tree, on the protocol above.
2. Losses are published next to wins, with the reason (design scope
   vs measured deficit) spelled out in the prose.
3. When a protocol or input distribution changes, the change is
   recorded in the CHANGELOG (e.g. the 1.1.1 seeding fix and the 1.2
   peaked/flat regime fix, both of which corrected numbers this
   repo had previously published).
