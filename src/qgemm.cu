// INT8 quantized matmul (the compute half of the v0.4 INT8 path; the
// storage half - quantize/dequantize/qadd - lives in quantize.cu).
//
//   C[M,N] = scale * ( A_q[M,K] (int8)  .  B_q[N,K] (int8) ^T )
//
// Both operands are row-major along K - the LLM-friendly layout: with
// A = activations [tokens, hidden] and B = a Linear weight [out, in],
// C = A @ B.T without any transposes. The kernel accumulates exact int32
// dot products and applies the combined per-tensor scale
// (sa*sb) once at the store, so CPU and GPU results are BIT-IDENTICAL
// to numpy's int32 matmul times a float scale - tests assert exact
// parity, no quantization tolerance games. (Integer wmma accumulation is
// exact under ANY summation order, so tile/slab configs cannot perturb
// the result: every k-split lands in the same int32 accumulator.)
//
// Kernel: qgemm_pipe_kernel - M > 1. Output tiles of TILE_M x TILE_N per
// block, K streamed in SLAB-wide slabs through DOUBLE-BUFFERED shared
// memory fed by cp.async (global -> shared DMA; the v0.4 kernel
// round-tripped every byte through registers and stalled the whole block
// on a barrier per 64-element slab):
//
//   zero both smem buffers (boundary-tile padding, done once)
//   issue slab 0 (cp.async group); per full slab s:
//     issue slab s+1 into the other buffer     (DMA in flight)
//     wait for group s; __syncthreads()
//     mma over smem[s & 1]; __syncthreads()
//
// so the global-load latency of slab s+1 hides behind the tensor-core
// work of slab s. K-tail slabs take a scalar synchronous stage with zero
// padding; shapes whose row stride cannot feed cp.async's 4-byte minimum
// alignment (k % 4 != 0 or misaligned base pointers) run a fully scalar
// synchronous loop - rare, correctness-identical.
//
// Tile configs (template parameters; the launch micro-benchmarks them on
// the caller's REAL buffers at full size on first sight of a shape and
// caches the winner per process - the v0.4.1 autotune pattern; stream
// captures skip tuning and take the default):
//
//   64x64, SLAB 64 or 128 - 256 threads (8 warps, 2x4, each 32x16 =
//     2 stacked mma tiles). Small-tile default for small M/N; 32 KB of
//     dynamic shared memory.
//   128x128, SLAB 64 - 512 threads (16 warps, 4x4, each 32x32 = 2x2 mma
//     tiles). The DRAM-intensity config: a 128x128 tile moves 4x more
//     MACs per operand byte than 64x64, which is what lets this kernel
//     approach cuBLASLt on large square GEMMs (the 64x64 tile saturates
//     around DRAM+L2 replay on a 3060). 96 KB of dynamic shared memory
//     (opt-in via cudaFuncSetAttribute, done inside the tuner - never
//     during a capture; a first-call capture always takes the default
//     config, and reaching the 128-tile in production implies its tuner
//     ran outside a capture and already raised the limit).
//
// Exactness is config-independent.
//
// GEMV - M == 1 (decode step): one warp per output row, lanes stream the
// K dimension with char4 loads and a warp-shuffle reduction. Entirely
// bandwidth-bound by design - the point of INT8 weights is halving the
// bytes moved per token - and it already runs at DRAM speed, so the
// pipeline work does not touch it.
//
// K must be >= 1 for the compute kernels; K == 0 zero-fills the output
// (an empty dot product is zero - the CPU reference has always said so;
// the v0.4 launcher silently skipped the write and left torch.empty
// garbage, which the tests never covered). Any M/N/K, including
// non-multiples of the tile sizes. Stream-ordered on the caller's stream;
// CUDA-graph capturable (no allocations, no syncs on the hot path).
//
// Register-array discipline (the v0.5 CONTRIBUTING lesson): every staged
// value lives in an array indexed ONLY by a compile-time-unrolled loop
// variable. Runtime loop bounds appear exclusively in the smem tile
// walks, where wmma fragments are re-declared per iteration (no dynamic
// indexing of register arrays anywhere in this file).

#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fusedtok {

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kQgBlockSmall = 256;   // threads for the 64x64 tile
constexpr int kQgBlockLarge = 512;   // threads for the 128x128 tile
constexpr int kQgTileSmall = 64;     // small tile edge
constexpr int kQgTileLarge = 128;    // large tile edge
constexpr int kQgSlabDefault = 64;   // K elements staged per iteration
constexpr int kQgSlabWide = 128;     // alternate slab for the small tile

// Shared-memory byte needs of a config (dynamic smem, partitioned as
// as[2][TILE_M][SLAB] | bs[2][TILE_N][SLAB] | cs[TILE_M][TILE_N]).
template <int TILE_M, int TILE_N, int SLAB>
constexpr int qgemm_smem_bytes() {
    return 2 * TILE_M * SLAB + 2 * TILE_N * SLAB + TILE_M * TILE_N * 4;
}

// ---------------------------------------------------------------------------
// GEMM: cp.async double-buffered pipeline, IMMA wmma, templated tile
// ---------------------------------------------------------------------------

template <int TILE_M, int TILE_N, int SLAB, int MT, int NT, int BLOCK, bool PC>
__global__ __launch_bounds__(BLOCK) void qgemm_pipe_kernel(
    const signed char* __restrict__ aq, const signed char* __restrict__ bq,
    const float* __restrict__ sb_vec, float* __restrict__ y,
    int m, int n, int k, float scale) {
    // Warp tiling: warps_m x warps_n over the block tile; each warp owns
    // (MT x NT) m16n16k16 mma tiles (MT*16 x NT*16 outputs). The config
    // constants below make the sub-tiles cover the block exactly.
    constexpr int WARPS_M = TILE_M / (MT * 16);
    constexpr int WARPS_N = TILE_N / (NT * 16);
    static_assert(WARPS_M * WARPS_N * 32 == BLOCK, "warp tiling mismatch");
    // cp.async fan-out per thread (compile-time, PER CHUNK WIDTH - using
    // the 4-byte worst case for the 16-byte loops would run the offsets
    // past the slab). The chunk sizes 16/8/4 bytes keep the global side
    // aligned for k % 16 / 8 / 4. All divide evenly by BLOCK threads.
    constexpr int A16 = TILE_M * SLAB / 16 / BLOCK;
    constexpr int A8 = TILE_M * SLAB / 8 / BLOCK;
    constexpr int A4 = TILE_M * SLAB / 4 / BLOCK;
    constexpr int B16 = TILE_N * SLAB / 16 / BLOCK;
    constexpr int B8 = TILE_N * SLAB / 8 / BLOCK;
    constexpr int B4 = TILE_N * SLAB / 4 / BLOCK;
    // K-tail / scalar-path staging (scalar bytes per thread).
    constexpr int A_SCL = TILE_M * SLAB / BLOCK;
    constexpr int B_SCL = TILE_N * SLAB / BLOCK;
    // Boundary-padding sweep (char4 stores per thread per buffer pair).
    constexpr int Z_IT = 2 * (TILE_M + TILE_N) * SLAB / 4 / BLOCK;
    static_assert(A16 >= 1 && B16 >= 1, "slab too small for the block");
    static_assert(A_SCL >= 4, "scalar staging degenerate");

    const int warp = threadIdx.x >> 5;
    const int warp_m = (warp % WARPS_M) * (MT * 16);
    const int warp_n = (warp / WARPS_M) * (NT * 16);
    // L2-aware rasterization: remap the linear block index so a wave of
    // co-resident blocks covers a GROUP_M x several-N panel instead of a
    // 1 x whole-row strip. With row-major ordering every wave re-reads
    // the entire B matrix (the m=4096 GEMM has 32 y-waves); grouped
    // ordering reads A 4x and B (tiles_x/wave) x - on a 3 MB-L2 GPU this
    // is the difference between DRAM-bound and compute-bound.
    const int tiles_x = gridDim.x, tiles_y = gridDim.y;
    const int b = blockIdx.y * tiles_x + blockIdx.x;
    const int GROUP_M = 4;
    const int per_group = GROUP_M * tiles_x;
    const int first_y = (b / per_group) * GROUP_M;
    const int size_y = min(tiles_y - first_y, GROUP_M);
    const int ty = first_y + (b % size_y);
    const int tx = (b % per_group) / size_y;
    const int gm0 = ty * TILE_M;
    const int gn0 = tx * TILE_N;
    const int tid = threadIdx.x;

    // Dynamic shared memory, partitioned by hand (the 128x128 config
    // needs 96 KB, past the 48 KB static limit; sharing one extern
    // buffer keeps ONE kernel body for every config).
    extern __shared__ signed char qg_smem[];
    signed char(*as)[SLAB] = reinterpret_cast<signed char(*)[SLAB]>(qg_smem);
    signed char(*bs)[SLAB] = reinterpret_cast<signed char(*)[SLAB]>(
        qg_smem + 2 * TILE_M * SLAB);
    int* cs = reinterpret_cast<int*>(qg_smem + 2 * TILE_M * SLAB +
                                     2 * TILE_N * SLAB);

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[MT][NT];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            wmma::fill_fragment(acc[i][j], 0);

    // Zero BOTH staging buffers up front, so boundary tiles (rows past
    // M / cols past N) read zeros without any predicate in the hot path:
    // the cp.async issue below simply skips out-of-range rows and the
    // pre-zeroed smem does the padding.
    {
        const char4 z = {0, 0, 0, 0};
        char4* az = reinterpret_cast<char4*>(qg_smem);
        #pragma unroll
        for (int it = 0; it < Z_IT; ++it)
            az[(tid + it * BLOCK)] = z;
    }
    __syncthreads();

    // Issue one full slab of both operands into staging buffer `buf` via
    // cp.async (global -> shared DMA, no register detour, no stall).
    // Rows outside M/N are skipped (pre-zeroed above). `chunk` is the
    // caller-computed width; every branch here is uniform across the
    // block. NOT followed by a commit - the caller commits the group.
    auto issue_slab = [&](int s, int buf, int chunk) {
        const int k0 = s * SLAB;
        if (chunk == 16) {
            #pragma unroll
            for (int it = 0; it < A16; ++it) {
                const int off = (tid + it * BLOCK) * 16;
                const int r = off / SLAB, c = off % SLAB;
                if (gm0 + r < m)
                    __pipeline_memcpy_async(&as[buf * TILE_M + r][c],
                                            aq + (size_t)(gm0 + r) * k + k0 + c,
                                            16);
            }
            #pragma unroll
            for (int it = 0; it < B16; ++it) {
                const int off = (tid + it * BLOCK) * 16;
                const int r = off / SLAB, c = off % SLAB;
                if (gn0 + r < n)
                    __pipeline_memcpy_async(&bs[buf * TILE_N + r][c],
                                            bq + (size_t)(gn0 + r) * k + k0 + c,
                                            16);
            }
        } else if (chunk == 8) {
            #pragma unroll
            for (int it = 0; it < A8; ++it) {
                const int off = (tid + it * BLOCK) * 8;
                const int r = off / SLAB, c = off % SLAB;
                if (gm0 + r < m)
                    __pipeline_memcpy_async(&as[buf * TILE_M + r][c],
                                            aq + (size_t)(gm0 + r) * k + k0 + c,
                                            8);
            }
            #pragma unroll
            for (int it = 0; it < B8; ++it) {
                const int off = (tid + it * BLOCK) * 8;
                const int r = off / SLAB, c = off % SLAB;
                if (gn0 + r < n)
                    __pipeline_memcpy_async(&bs[buf * TILE_N + r][c],
                                            bq + (size_t)(gn0 + r) * k + k0 + c,
                                            8);
            }
        } else {
            #pragma unroll
            for (int it = 0; it < A4; ++it) {
                const int off = (tid + it * BLOCK) * 4;
                const int r = off / SLAB, c = off % SLAB;
                if (gm0 + r < m)
                    __pipeline_memcpy_async(&as[buf * TILE_M + r][c],
                                            aq + (size_t)(gm0 + r) * k + k0 + c,
                                            4);
            }
            #pragma unroll
            for (int it = 0; it < B4; ++it) {
                const int off = (tid + it * BLOCK) * 4;
                const int r = off / SLAB, c = off % SLAB;
                if (gn0 + r < n)
                    __pipeline_memcpy_async(&bs[buf * TILE_N + r][c],
                                            bq + (size_t)(gn0 + r) * k + k0 + c,
                                            4);
            }
        }
    };

    // Tensor-core pass over one staged slab. `kk` bounds the valid K
    // columns (SLAB for full slabs, the tail width for the last one);
    // the guard is uniform across the block (same k everywhere). B[N,K]
    // row-major IS the col-major view of B^T (K x N), so no transpose is
    // ever materialized.
    auto compute = [&](int buf, int kk) {
        #pragma unroll
        for (int ks = 0; ks < SLAB / 16; ++ks) {
            if (ks * 16 >= kk) break;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                           wmma::row_major> fa[MT];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                           wmma::col_major> fb[NT];
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                wmma::load_matrix_sync(
                    fa[i], &as[buf * TILE_M + warp_m + i * 16][ks * 16], SLAB);
            #pragma unroll
            for (int j = 0; j < NT; ++j)
                wmma::load_matrix_sync(
                    fb[j], &bs[buf * TILE_N + warp_n + j * 16][ks * 16], SLAB);
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
    };

    // Scalar stage of the slab starting at k0 with kk valid columns
    // (zero-padded past kk). Single-buffered; callers sync around it.
    auto stage_scalar = [&](int k0, int kk) {
        signed char ascl[A_SCL], bscl[B_SCL];
        #pragma unroll
        for (int it = 0; it < A_SCL; ++it) {
            const int t = tid + it * BLOCK;
            const int r = t / SLAB, c = t % SLAB;
            ascl[it] = (gm0 + r < m && c < kk)
                ? aq[(size_t)(gm0 + r) * k + k0 + c] : (signed char)0;
        }
        #pragma unroll
        for (int it = 0; it < B_SCL; ++it) {
            const int t = tid + it * BLOCK;
            const int r = t / SLAB, c = t % SLAB;
            bscl[it] = (gn0 + r < n && c < kk)
                ? bq[(size_t)(gn0 + r) * k + k0 + c] : (signed char)0;
        }
        #pragma unroll
        for (int it = 0; it < A_SCL; ++it) {
            const int t = tid + it * BLOCK;
            as[t / SLAB][t % SLAB] = ascl[it];
        }
        #pragma unroll
        for (int it = 0; it < B_SCL; ++it) {
            const int t = tid + it * BLOCK;
            bs[t / SLAB][t % SLAB] = bscl[it];
        }
    };

    // --- the pipeline -----------------------------------------------------
    // cp.async needs BOTH sides aligned to the chunk width: the global
    // row bases advance by k bytes (int8 rows), so k % 16 / 8 / 4 gate
    // the 16 / 8 / 4-byte chunks, and the operand base pointers must be
    // at least as aligned (a byte-shifted view can misalign any base).
    // chunk == 0 means cp.async is unusable for this shape - everything
    // takes the scalar synchronous loop (rare: odd K; performance is
    // irrelevant there, correctness is identical).
    const size_t both = (size_t)aq | (size_t)bq;
    const int chunk = (k % 16 == 0 && (both & 15) == 0) ? 16
                    : (k % 8 == 0 && (both & 7) == 0) ? 8
                    : (k % 4 == 0 && (both & 3) == 0) ? 4 : 0;
    const int nfull = (chunk > 0) ? k / SLAB : 0;   // uniform per block
    const int kr = k - nfull * SLAB;                // tail in [0, SLAB)
    if (nfull > 0) {
        issue_slab(0, 0, chunk);
        __pipeline_commit();
        for (int s = 0; s < nfull; ++s) {
            if (s + 1 < nfull) {
                issue_slab(s + 1, (s + 1) & 1, chunk);
                __pipeline_commit();
            }
            // group s is done once at most (s+1 pending) remains
            __pipeline_wait_prior((s + 1 < nfull) ? 1 : 0);
            __syncthreads();                 // slab s visible to the block
            compute(s & 1, SLAB);
            __syncthreads();                 // all reads done before reuse
        }
    }
    // K tail: one scalar, synchronous stage (zero-padded), then one mma
    // pass. Register arrays are indexed only by unrolled loop variables.
    if (chunk > 0 && kr > 0) {
        stage_scalar(nfull * SLAB, kr);
        __syncthreads();
        compute(0, kr);
    }
    // Fully scalar fallback (chunk == 0): every slab staged synchronously.
    if (chunk == 0) {
        const int ns = (k + SLAB - 1) / SLAB;
        for (int s = 0; s < ns; ++s) {
            const int kk = min(SLAB, k - s * SLAB);
            stage_scalar(s * SLAB, kk);
            __syncthreads();
            compute(0, kk);
            __syncthreads();    // slab consumed before the next stage
        }
    }

    // Epilogue: int32 accumulators through shared (coalesced global
    // stores), one float scale applied exactly once - the bit-identical
    // contract lives or dies with this single application. PC (per-
    // channel) composes the output scale as f32(sa * sb_vec[gn]): one
    // rounding for the scale, one for the product - the CPU reference
    // performs the identical two-rounding sequence per element, and
    // consecutive lanes read consecutive sb entries (coalesced).
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            wmma::store_matrix_sync(
                &cs[(warp_m + i * 16) * TILE_N + (warp_n + j * 16)],
                acc[i][j], TILE_N, wmma::mem_row_major);
    __syncthreads();
    for (int t = threadIdx.x; t < TILE_M * TILE_N; t += BLOCK) {
        const int r = t / TILE_N, c = t % TILE_N;
        const int gm = gm0 + r, gn = gn0 + c;
        if (gm < m && gn < n) {
            float s = scale;
            if (PC)
                s *= __ldg(&sb_vec[gn]);
            y[(size_t)gm * n + gn] = (float)cs[t] * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Config selection: micro-benchmark the tile/slab candidates on the
// caller's real buffers at full size (the v0.4.1 lesson: a truncated
// problem misleads). Cached per (m, n, k) for the process. Captures skip
// tuning and take the default (events + syncs are illegal mid-capture).
// The 128x128 candidate needs > 48 KB of dynamic shared memory: the
// opt-in attribute is raised here, INSIDE the tuner - never during a
// capture (a capture-path launch can only reach that config after its
// tuner ran, so the attribute is already in place).
// ---------------------------------------------------------------------------

struct QgConfig {
    int tile;                       // TILE_M == TILE_N (64 or 128)
    int slab;
    int block;                      // threads per block
    int mt, nt;                     // mma tiles per warp
};

// The candidates the tuner races. Index 0 is the capture/default config.
constexpr QgConfig kQgCands[3] = {
    {64, kQgSlabDefault, kQgBlockSmall, 2, 1},   // default small tile
    {64, kQgSlabWide, kQgBlockSmall, 2, 1},      // small tile, wide slab
    {kQgTileLarge, kQgSlabDefault, kQgBlockLarge, 2, 2},  // DRAM-intensity tile
};

int qgemm_smem_for(const QgConfig& cfg) {
    if (cfg.tile == 128)
        return qgemm_smem_bytes<kQgTileLarge, kQgTileLarge, 64>();
    if (cfg.slab == 128)
        return qgemm_smem_bytes<64, 64, 128>();
    return qgemm_smem_bytes<64, 64, 64>();
}

void qgemm_launch_config(const QgConfig& cfg, bool pc,
                         const signed char* aq, const signed char* bq,
                         const float* sb_vec, float* y,
                         int m, int n, int k, float scale,
                         dim3 grid, cudaStream_t cs) {
    const int smem = qgemm_smem_for(cfg);
    if (cfg.tile == 128) {
        if (pc)
            qgemm_pipe_kernel<kQgTileLarge, kQgTileLarge, 64, 2, 2, kQgBlockLarge, true>
                <<<grid, cfg.block, smem, cs>>>(aq, bq, sb_vec, y, m, n, k, scale);
        else
            qgemm_pipe_kernel<kQgTileLarge, kQgTileLarge, 64, 2, 2, kQgBlockLarge, false>
                <<<grid, cfg.block, smem, cs>>>(aq, bq, nullptr, y, m, n, k, scale);
    } else if (cfg.slab == 128) {
        if (pc)
            qgemm_pipe_kernel<kQgTileSmall, kQgTileSmall, kQgSlabWide,
                              2, 1, kQgBlockSmall, true>
                <<<grid, cfg.block, smem, cs>>>(aq, bq, sb_vec, y, m, n, k, scale);
        else
            qgemm_pipe_kernel<kQgTileSmall, kQgTileSmall, kQgSlabWide,
                              2, 1, kQgBlockSmall, false>
                <<<grid, cfg.block, smem, cs>>>(aq, bq, nullptr, y, m, n, k, scale);
    } else {
        if (pc)
            qgemm_pipe_kernel<kQgTileSmall, kQgTileSmall, kQgSlabDefault,
                              2, 1, kQgBlockSmall, true>
                <<<grid, cfg.block, smem, cs>>>(aq, bq, sb_vec, y, m, n, k, scale);
        else
            qgemm_pipe_kernel<kQgTileSmall, kQgTileSmall, kQgSlabDefault,
                              2, 1, kQgBlockSmall, false>
                <<<grid, cfg.block, smem, cs>>>(aq, bq, nullptr, y, m, n, k, scale);
    }
}

int qgemm_pick_config(const signed char* aq, const signed char* bq,
                      const float* sb_vec, float* y,
                      int m, int n, int k, float scale, bool pc,
                      cudaStream_t cs) {
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, bool>, int> cache;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple(m, n, k, pc);
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;                // winning candidate index

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (cudaEventCreate(&ev0) != cudaSuccess ||
        cudaEventCreate(&ev1) != cudaSuccess) {
        cudaEventDestroy(ev0);
        cudaGetLastError();
        return 0;                     // default config; tuning impossible
    }
    float best_ms = 1e30f;
    int best_idx = 0;
    for (int ci = 0; ci < 3; ++ci) {
        const QgConfig& cfg = kQgCands[ci];
        if (cfg.tile == 128) {
            // opt-in dynamic smem above the 48 KB static ceiling; safe to
            // repeat. Never executed during a capture (tuning is skipped).
            const void* kfn = pc
                ? (const void*)qgemm_pipe_kernel<kQgTileLarge,
                                                kQgTileLarge, 64, 2, 2,
                                                kQgBlockLarge, true>
                : (const void*)qgemm_pipe_kernel<kQgTileLarge,
                                                kQgTileLarge, 64, 2, 2,
                                                kQgBlockLarge, false>;
            if (cudaFuncSetAttribute(
                    kfn,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    qgemm_smem_for(cfg)) != cudaSuccess) {
                cudaGetLastError();
                continue;                 // config unavailable on this GPU
            }
        }
        dim3 grid((n + cfg.tile - 1) / cfg.tile, (m + cfg.tile - 1) / cfg.tile);
        // A structurally unlaunchable candidate (or one hitting an async
        // fault) scores as slow instead of failing the call. Faults can
        // surface asynchronously - the sticky error is checked after the
        // warmups AND after the timed section, and a candidate that
        // reports any error is discarded. Every exit path clears the
        // sticky error so the production launch below starts clean.
        bool ok = true;
        try {
            for (int i = 0; i < 3; ++i)
                qgemm_launch_config(cfg, pc, aq, bq, sb_vec, y, m, n, k,
                                    scale, grid, cs);
        } catch (const std::runtime_error&) {
            cudaGetLastError();
            ok = false;
        }
        if (ok) {
            cudaEventRecord(ev0, cs);
            try {
                for (int i = 0; i < 8; ++i)
                    qgemm_launch_config(cfg, pc, aq, bq, sb_vec, y, m, n, k,
                                        scale, grid, cs);
            } catch (const std::runtime_error&) {
                ok = false;
            }
            cudaEventRecord(ev1, cs);
            cudaEventSynchronize(ev1);
            if (cudaGetLastError() != cudaSuccess)
                ok = false;              // late-surfacing async fault
        }
        float ms = 1e30f;
        if (ok) {
            cudaEventElapsedTime(&ms, ev0, ev1);
            if (ms < best_ms) {
                best_ms = ms;
                best_idx = ci;
            }
        }
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaGetLastError();                   // clear benign residue
    cache.emplace(key, best_idx);
    return best_idx;
}

// ---------------------------------------------------------------------------
// GEMV: M == 1 decode step, one warp per output row. PC composes the
// output scale as f32(sa * sb_vec[warp]) - the same two-rounding
// sequence as the GEMM epilogue and the CPU reference.
// ---------------------------------------------------------------------------

template <bool PC>
__global__ void qgemv_kernel(const signed char* __restrict__ xq,
                             const signed char* __restrict__ wq,
                             const float* __restrict__ sb_vec,
                             float* __restrict__ y,
                             int n, int k, float scale) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= n) return;
    const int lane = threadIdx.x & 31;
    const signed char* row = wq + (size_t)warp * k;
    int acc = 0;
    // vectorized loads: char4 = 4 int8 values (4-byte alignment always
    // holds - rows start at multiples of k bytes from an aligned base).
    // The warp covers 128 bytes per iteration; the scalar tail (and any
    // misalignment) falls back to single bytes. NOTE: char4 fields are
    // PLAIN char - unsigned on MSVC hosts - so every lane value passes
    // through an explicit (signed char) cast (a negative int8 read
    // through an unsigned char would flip the product's sign).
    const int k4 = (k / 4) * 4;
    if (((size_t)row & 3) == 0 && ((size_t)xq & 3) == 0 && k4 > 0) {
        for (int c = lane * 4; c < k4; c += 32 * 4) {
            const char4 xv = *reinterpret_cast<const char4*>(xq + c);
            const char4 wv = *reinterpret_cast<const char4*>(row + c);
            acc += (int)(signed char)xv.x * (int)(signed char)wv.x
                 + (int)(signed char)xv.y * (int)(signed char)wv.y
                 + (int)(signed char)xv.z * (int)(signed char)wv.z
                 + (int)(signed char)xv.w * (int)(signed char)wv.w;
        }
    }
    for (int c = k4 + lane; c < k; c += 32)
        acc += (int)xq[c] * (int)row[c];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float s = scale;
        if (PC)
            s *= __ldg(&sb_vec[warp]);
        y[warp] = (float)acc * s;
    }
}

void qgemm_check(int m, int n, int k) {
    if (m < 0 || n < 0 || k < 0)
        throw std::invalid_argument("m, n, k must be >= 0");
    if ((long long)m * n * k > (1LL << 38))
        throw std::invalid_argument("qgemm operands too large");
}

} // namespace

// ---------------------------------------------------------------------------
// CPU reference: exact int32 dot products + one float scale
// ---------------------------------------------------------------------------

std::vector<float> qgemm_cpu(const std::vector<signed char>& aq,
                             const std::vector<signed char>& bq,
                             int m, int n, int k, float sa, float sb) {
    qgemm_check(m, n, k);
    if ((size_t)m * k > aq.size() || (size_t)n * k > bq.size())
        throw std::invalid_argument("qgemm operand size mismatch");
    const float scale = sa * sb;
    std::vector<float> y((size_t)m * n, 0.0f);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int acc = 0;
            const signed char* arow = aq.data() + (size_t)i * k;
            const signed char* brow = bq.data() + (size_t)j * k;
            for (int p = 0; p < k; ++p) acc += (int)arow[p] * (int)brow[p];
            y[(size_t)i * n + j] = (float)acc * scale;
        }
    return y;
}

std::vector<float> qgemm_perchannel_cpu(const std::vector<signed char>& aq,
                                        const std::vector<signed char>& bq,
                                        const std::vector<float>& sb,
                                        int m, int n, int k, float sa) {
    qgemm_check(m, n, k);
    if ((size_t)m * k > aq.size() || (size_t)n * k > bq.size())
        throw std::invalid_argument("qgemm operand size mismatch");
    if (sb.size() != (size_t)n)
        throw std::invalid_argument("per-channel scale vector must have n entries");
    // Per element: f32(sa * sb[j]) once, then one product by the exact
    // int32 accumulator - identical operation order to the GPU epilogue.
    std::vector<float> y((size_t)m * n, 0.0f);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int acc = 0;
            const signed char* arow = aq.data() + (size_t)i * k;
            const signed char* brow = bq.data() + (size_t)j * k;
            for (int p = 0; p < k; ++p) acc += (int)arow[p] * (int)brow[p];
            y[(size_t)i * n + j] = (float)acc * (sa * sb[j]);
        }
    return y;
}

// ---------------------------------------------------------------------------
// launcher: dispatches M == 1 to the GEMV kernel, M > 1 to the pipelined
// IMMA kernel with runtime config selection
// ---------------------------------------------------------------------------

void qgemm_launch(const signed char* aq, const signed char* bq,
                  float* y, int m, int n, int k, float sa, float sb,
                  std::uintptr_t stream) {
    qgemm_check(m, n, k);
    if (m == 0 || n == 0) return;
    cudaStream_t cs = (cudaStream_t)stream;
    // K == 0: every dot product is empty -> the output is zeros. The
    // float scale is irrelevant (0 * anything = 0; the CPU reference
    // multiplies the zero accumulator the same way).
    if (k == 0) {
        cudaMemsetAsync(y, 0, (size_t)m * n * sizeof(float), cs);
        return;
    }
    if (m == 1) {
        // one warp per output row; 256-thread blocks = 8 rows per block
        const int grid = (n + 7) / 8;
        qgemv_kernel<false><<<grid, kQgBlockSmall, 0, cs>>>(
            aq, bq, nullptr, y, n, k, sa * sb);
        check_launch("qgemv kernel launch");
        return;
    }
    const float scale = sa * sb;
    if (stream_is_capturing(cs)) {
        // graph capture: no events/syncs during tuning - take the default
        dim3 grid((n + kQgTileSmall - 1) / kQgTileSmall,
                  (m + kQgTileSmall - 1) / kQgTileSmall);
        qgemm_pipe_kernel<64, 64, 64, 2, 1, 256, false>
            <<<grid, kQgBlockSmall, qgemm_smem_bytes<64, 64, 64>(), cs>>>(
                aq, bq, nullptr, y, m, n, k, scale);
        check_launch("qgemm pipe kernel launch");
        return;
    }
    const int ci = qgemm_pick_config(aq, bq, nullptr, y, m, n, k, scale,
                                     false, cs);
    const QgConfig& cfg = kQgCands[ci];
    dim3 grid((n + cfg.tile - 1) / cfg.tile, (m + cfg.tile - 1) / cfg.tile);
    qgemm_launch_config(cfg, false, aq, bq, nullptr, y, m, n, k, scale,
                        grid, cs);
    check_launch("qgemm pipe kernel launch");
}

void qgemm_perchannel_launch(const signed char* aq, const signed char* bq,
                             const float* sb_vec, float* y,
                             int m, int n, int k, float sa,
                             std::uintptr_t stream) {
    qgemm_check(m, n, k);
    if (!sb_vec)
        throw std::invalid_argument("per-channel scale vector is required");
    if (m == 0 || n == 0) return;
    cudaStream_t cs = (cudaStream_t)stream;
    if (k == 0) {
        cudaMemsetAsync(y, 0, (size_t)m * n * sizeof(float), cs);
        return;
    }
    if (m == 1) {
        const int grid = (n + 7) / 8;
        qgemv_kernel<true><<<grid, kQgBlockSmall, 0, cs>>>(
            aq, bq, sb_vec, y, n, k, sa);
        check_launch("qgemv per-channel kernel launch");
        return;
    }
    if (stream_is_capturing(cs)) {
        dim3 grid((n + kQgTileSmall - 1) / kQgTileSmall,
                  (m + kQgTileSmall - 1) / kQgTileSmall);
        qgemm_pipe_kernel<64, 64, 64, 2, 1, 256, true>
            <<<grid, kQgBlockSmall, qgemm_smem_bytes<64, 64, 64>(), cs>>>(
                aq, bq, sb_vec, y, m, n, k, sa);
        check_launch("qgemm pipe per-channel kernel launch");
        return;
    }
    const int ci = qgemm_pick_config(aq, bq, sb_vec, y, m, n, k, sa,
                                     true, cs);
    const QgConfig& cfg = kQgCands[ci];
    dim3 grid((n + cfg.tile - 1) / cfg.tile, (m + cfg.tile - 1) / cfg.tile);
    qgemm_launch_config(cfg, true, aq, bq, sb_vec, y, m, n, k, sa,
                        grid, cs);
    check_launch("qgemm pipe per-channel kernel launch");
}

} // namespace fusedtok
