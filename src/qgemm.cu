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
// parity, no quantization tolerance games.
//
// Two kernels:
//   qgemm_kernel - M > 1: 64x64 output tile per block, K staged in
//     32-element slabs through shared memory (char4 vector loads with
//     scalar bounds tails), each of the 256 threads owns a 4x4 sub-tile
//     and accumulates in registers.
//   qgemv_kernel - M == 1 (decode step): one warp per output row, lanes
//     stream the K dimension with char4 loads and a warp-shuffle
//     reduction. Entirely bandwidth-bound by design - the point of INT8
//     weights is halving the bytes moved per token.
//
// K must be >= 1; any M/N/K (including non-multiples of the tile sizes).
// Stream-ordered on the caller's stream; CUDA-graph capturable (no
// allocations, no syncs).

#include "fusedtok/cuda_launch.hpp"
#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace fusedtok {

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kQgTile = 64;      // output tile: M x N per block
constexpr int kQgSlab = 64;      // K elements staged per iteration (4 mma k-steps)
constexpr int kQgBlock = 256;    // 8 warps, each owns a 32x32 sub-tile (2x2 mma tiles)

// ---------------------------------------------------------------------------
// GEMM: 64x64 tile, tensor-core IMMA (wmma s8 x s8 -> s32), 32x32 per warp
// ---------------------------------------------------------------------------

__global__ void qgemm_kernel(const signed char* __restrict__ aq,
                             const signed char* __restrict__ bq,
                             float* __restrict__ y,
                             int m, int n, int k, float scale) {
    // warp's 32x16 sub-tile within the 64x64 block tile: warps tile as
    // 2 (M) x 4 (N); each owns 2x1 m16n16k16 mma tiles = 32x16 outputs,
    // 8 warps * 512 = 4096 = 64*64 exactly
    const int warp = threadIdx.x >> 5;
    const int warp_m = (warp >> 2) * 32;   // 0 or 32
    const int warp_n = (warp & 3) * 16;    // 0, 16, 32, 48
    const int gm0 = blockIdx.y * kQgTile;
    const int gn0 = blockIdx.x * kQgTile;

    __shared__ signed char as[kQgTile][kQgSlab];
    __shared__ signed char bs[kQgTile][kQgSlab];
    __shared__ int cs[kQgTile][kQgTile];

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[2][1];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        wmma::fill_fragment(acc[i][0], 0);

    for (int k0 = 0; k0 < k; k0 += kQgSlab) {
        const int kk = min(kQgSlab, k - k0);
        // stage one 64x16 slab per operand (1KB each): one 4-byte store
        // per thread when the rows are 4-byte aligned (k % 4 == 0 and
        // base-aligned bases), scalar bytes otherwise; rows beyond M/N
        // and the K tail pad with zeros
        const bool vec = ((k % 4) == 0) &&
                         (((size_t)aq & 3) == 0) && (((size_t)bq & 3) == 0);
        const int stage_items = kQgTile * kQgSlab / 4;
        for (int t = threadIdx.x; t < stage_items; t += kQgBlock) {
            const int r = (t * 4) / kQgSlab;
            const int c = (t * 4) % kQgSlab;
            const int gmr = gm0 + r;
            const int gnc = gn0 + r;
            if (vec && c + 4 <= kk && gmr < m && gnc < n) {
                *reinterpret_cast<char4*>(&as[r][c]) =
                    *reinterpret_cast<const char4*>(aq + (size_t)gmr * k + k0 + c);
                *reinterpret_cast<char4*>(&bs[r][c]) =
                    *reinterpret_cast<const char4*>(bq + (size_t)gnc * k + k0 + c);
            } else {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int cc = c + u;
                    const signed char av =
                        (gmr < m && cc < kk)
                            ? aq[(size_t)gmr * k + k0 + cc] : (signed char)0;
                    const signed char bv =
                        (gnc < n && cc < kk)
                            ? bq[(size_t)gnc * k + k0 + cc] : (signed char)0;
                    as[r][cc] = av;
                    bs[r][cc] = bv;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < kQgSlab / 16; ++ks) {
            if (ks * 16 >= kk) break;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                               wmma::row_major> fa;
                wmma::load_matrix_sync(fa, &as[warp_m + i * 16][ks * 16],
                                       kQgSlab);
                // B[N, K] row-major IS the col-major view of B^T (K x N)
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                               wmma::col_major> fb;
                wmma::load_matrix_sync(fb, &bs[warp_n][ks * 16], kQgSlab);
                wmma::mma_sync(acc[i][0], fa, fb, acc[i][0]);
            }
        }
        __syncthreads();
    }

    // epilogue: int32 accumulators through shared, then one float scale
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        wmma::store_matrix_sync(&cs[warp_m + i * 16][warp_n],
                                acc[i][0], kQgTile,
                                wmma::mem_row_major);
    __syncthreads();
    for (int t = threadIdx.x; t < kQgTile * kQgTile; t += kQgBlock) {
        const int r = t / kQgTile, c = t % kQgTile;
        const int gm = gm0 + r, gn = gn0 + c;
        if (gm < m && gn < n)
            y[(size_t)gm * n + gn] = (float)cs[r][c] * scale;
    }
}

// ---------------------------------------------------------------------------
// GEMV: M == 1 decode step, one warp per output row
// ---------------------------------------------------------------------------

__global__ void qgemv_kernel(const signed char* __restrict__ xq,
                             const signed char* __restrict__ wq,
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
    if (lane == 0) y[warp] = (float)acc * scale;
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

// ---------------------------------------------------------------------------
// launcher: dispatches M == 1 to the GEMV kernel
// ---------------------------------------------------------------------------

void qgemm_launch(const signed char* aq, const signed char* bq,
                  float* y, int m, int n, int k, float sa, float sb,
                  std::uintptr_t stream) {
    qgemm_check(m, n, k);
    if (m == 0 || n == 0 || k == 0) return;
    cudaStream_t cs = (cudaStream_t)stream;
    if (m == 1) {
        // one warp per output row; 256-thread blocks = 8 rows per block
        const int grid = (n + 7) / 8;
        qgemv_kernel<<<grid, kQgBlock, 0, cs>>>(aq, bq, y, n, k, sa * sb);
        check_launch("qgemv kernel launch");
        return;
    }
    dim3 grid((n + kQgTile - 1) / kQgTile, (m + kQgTile - 1) / kQgTile);
    qgemm_kernel<<<grid, kQgBlock, 0, cs>>>(aq, bq, y, m, n, k, sa * sb);
    check_launch("qgemm kernel launch");
}

} // namespace fusedtok
