// AxonML CUDA — FUSED VQ sub-bit matmul (TRAINING-capable, forward + backward).
//
// Computes  out[n,out_f] = x[n,in_f] · recon^T   where the dense weight
//   recon[o,c] = scale[o] * codebook[ idx[o*vpr + c/dim]*dim + (c%dim) ]
// is NEVER materialized — codebook vectors are gathered by index inside the
// dot-product loop. This is the on-GPU, grad-capable twin of
// PackedVqLinear::forward_fused (pack.rs): a d=16384 block runs with the
// ~0.5 GiB packed working set instead of its 12.8 GiB dense fp32, so the
// offload streaming loop collapses to one launch per layer -> resident op-count.
//
// vpr = in_f / dim   (vectors per output row).  Indices are u16 (k<=65536).
// Pure f32, deterministic. Compile:
//   clang -O3 -x cuda --cuda-device-only --cuda-gpu-arch=sm_89 \
//         --cuda-path=/usr/local/cuda-12.9 -nocudalib -S -o vq_fused_matmul.ptx vq_fused_matmul.cu

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// ---- forward: out[i,o] = scale[o] * sum_v <x[i,v], C[idx[o,v]]> ----
extern "C" __global__ void vq_fused_matmul_f32(
    const float*          __restrict__ x,      // [n, in_f]
    const unsigned int* __restrict__ idx,    // [out_f * vpr]
    const float*          __restrict__ cb,     // [k * dim]
    const float*          __restrict__ scale,  // [out_f]
    float*                __restrict__ out,    // [n, out_f]
    int n, int in_f, int out_f, int dim, int vpr
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    int o = blockIdx.y * blockDim.y + threadIdx.y;  // output feature
    if (i >= n || o >= out_f) return;
    const float* xrow          = x   + (long)i * in_f;
    const unsigned int* irow = idx + (long)o * vpr;
    float acc = 0.f;
    for (int v = 0; v < vpr; ++v) {
        const float* cv = cb + (long)irow[v] * dim;
        const float* xb = xrow + v * dim;
        for (int t = 0; t < dim; ++t) acc += xb[t] * cv[t];
    }
    out[(long)i * out_f + o] = scale[o] * acc;
}

// ---- backward wrt x: grad_x[i, v*dim + t] = sum_o g[i,o]*scale[o]*C[idx[o,v]][t]
// One thread per (i, input-column). Loops over out_f (reduction).           ----
extern "C" __global__ void vq_fused_matmul_dx_f32(
    const float*          __restrict__ g,      // grad_out [n, out_f]
    const unsigned int* __restrict__ idx,    // [out_f * vpr]
    const float*          __restrict__ cb,     // [k * dim]
    const float*          __restrict__ scale,  // [out_f]
    float*                __restrict__ dx,     // [n, in_f]
    int n, int in_f, int out_f, int dim, int vpr
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    int c = blockIdx.y * blockDim.y + threadIdx.y;  // input column [0,in_f)
    if (i >= n || c >= in_f) return;
    int v = c / dim;        // which vector
    int t = c - v * dim;    // lane within vector
    const float* grow = g + (long)i * out_f;
    float acc = 0.f;
    for (int o = 0; o < out_f; ++o) {
        unsigned int code = idx[(long)o * vpr + v];
        acc += grow[o] * scale[o] * cb[(long)code * dim + t];
    }
    dx[(long)i * in_f + c] = acc;
}

// ---- backward wrt codebook: grad_cb[code][t] += sum_{o,v: idx==code} sum_i g[i,o]*scale[o]*x[i,v*dim+t]
// STE to the codebook. One thread per (o, v); atomicAdd into grad_cb (k*dim).
// (dcb is the codebook gradient; the assignment idx is treated as constant — STE.)
extern "C" __global__ void vq_fused_matmul_dcb_f32(
    const float*          __restrict__ g,      // grad_out [n, out_f]
    const float*          __restrict__ x,      // [n, in_f]
    const unsigned int* __restrict__ idx,    // [out_f * vpr]
    const float*          __restrict__ scale,  // [out_f]
    float*                __restrict__ dcb,    // [k * dim]  (must be pre-zeroed)
    int n, int in_f, int out_f, int dim, int vpr
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    int v = blockIdx.y * blockDim.y + threadIdx.y;  // vector index
    if (o >= out_f || v >= vpr) return;
    unsigned int code = idx[(long)o * vpr + v];
    float s = scale[o];
    // accumulate over rows for this (o,v)'s dim lanes, atomically into the shared code
    for (int t = 0; t < dim; ++t) {
        float acc = 0.f;
        for (int i = 0; i < n; ++i) {
            acc += g[(long)i * out_f + o] * x[(long)i * in_f + v * dim + t];
        }
        atomicAdd(&dcb[(long)code * dim + t], s * acc);
    }
}

// ---- dcb SCATTER: fold a precomputed dW tile into the codebook grad ----
// Splits the fused dcb kernel so the g^T@x GEMM can run on cuBLAS-tf32 and only
// the scatter stays hand-written. dW[lo, i] = (g^T @ x)[o0+lo, i]; accumulate
// dcb[code*dim+t] += scale[o] * dW[lo, v*dim+t], code = idx[o*vpr+v], o=o0+lo.
// Matches vq_fused_matmul_dcb_f32's scatter exactly.
extern "C" __global__ void vq_dcb_scatter_f32(
    const float*        __restrict__ dW,     // [ot, in_f] tile of g^T@x
    const unsigned int* __restrict__ idx,    // [out_f * vpr]
    const float*        __restrict__ scale,  // [out_f]
    float*              __restrict__ dcb,    // [k * dim]  (pre-zeroed)
    int ot, int in_f, int dim, int vpr, int o0
) {
    int lo = blockIdx.x * blockDim.x + threadIdx.x;  // local output row within the tile
    int v  = blockIdx.y * blockDim.y + threadIdx.y;  // vector index
    if (lo >= ot || v >= vpr) return;
    int o = o0 + lo;
    unsigned int code = idx[(long)o * vpr + v];
    float s = scale[o];
    for (int t = 0; t < dim; ++t) {
        atomicAdd(&dcb[(long)code * dim + t], s * dW[(long)lo * in_f + v * dim + t]);
    }
}

// ---- TILED (shared-memory) fused VQ matmul forward — real throughput ----
// out[n,out_f] = x @ recon^T,  recon^T[k,o] = scale[o]*cb[idx[o*vpr+k/dim]][k%dim].
// 16x16 output tile; the gathered weight tile is loaded to smem once and reused
// across the tile's rows (dim=16 => one codebook vector per k-tile).
#define VQ_TILE 16
// Codebook floats (k*dim) cached in shared memory once per block. The sub-bit
// production config is k<=256, dim<=8 => k*dim<=2048 (8 KiB). A block whose
// codebook exceeds this cap (or with the cache toggled off) transparently falls
// back to gathering from global — same math, different memory source.
#define VQ_CB_SMEM_MAX 2048
extern "C" __global__ void vq_fused_matmul_tiled_f32(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr, int k, int use_smem
) {
    __shared__ float As[VQ_TILE][VQ_TILE];
    __shared__ float Bs[VQ_TILE][VQ_TILE];
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * VQ_TILE + ty;   // i in [0,n)
    int col = blockIdx.x * VQ_TILE + tx;   // o in [0,out_f)
    // Stage the whole codebook into shared memory once (memory-bound decode GEMV:
    // the per-column gather otherwise re-reads codebook entries from global for
    // every output column of every block). `cache` is uniform across the block,
    // so the __syncthreads() is reached by all-or-no threads. Values are copied
    // bit-for-bit, so the gather result is byte-identical to the global path.
    int cbn = k * dim;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        for (int li = ty * VQ_TILE + tx; li < cbn; li += VQ_TILE * VQ_TILE) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;
    float acc = 0.f;
    for (int k0 = 0; k0 < in_f; k0 += VQ_TILE) {
        int kA = k0 + tx;                  // As[ty][tx] = x[row, kA]
        As[ty][tx] = (row < n && kA < in_f) ? x[(long)row * in_f + kA] : 0.f;
        int kB = k0 + ty;                  // Bs[ty][tx] = recon^T[kB, col]
        if (col < out_f && kB < in_f) {
            unsigned int code = idx[(long)col * vpr + kB / dim];
            Bs[ty][tx] = scale[col] * cbp[(long)code * dim + (kB % dim)];
        } else {
            Bs[ty][tx] = 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < VQ_TILE; ++kk) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (row < n && col < out_f) out[(long)row * out_f + col] = acc;
}

// ---- WARP-PER-COLUMN fused VQ GEMV (decode, NON-grouped) --------------------
// Batch-1 decode twin of vq_fused_matmul_tiled_f32 for the single (non-MoE)
// packed weight — the attention projections (fused-QKV, o_proj) and the whole
// 1.7B dense model. One WARP reduces one output column: lane l sums the strided
// vq-vectors v = l, l+32, ... into a partial, then a shfl_down tree combines the
// 32 lane partials. At n==1 the tiled path wasted 15/16 threads and ran a
// __syncthreads() per k-tile; this gives every column 32 independent in-flight
// load streams (idx read coalesced across lanes) + 32x resident threads to hide
// the dependent gather latency, with NO per-k-tile sync. Mirrors the grouped-warp
// arithmetic EXACTLY (scale folded into each B term before the FFMA, smem codebook
// cache, float4 x/codebook loads) but on u32 indices with no group/expert offset.
// Re-associates the length-in_f dot at the 32-lane boundaries + tree (NOT bit-
// identical) so it is FNV-gated: accept only if the token stream stays argmax-
// identical. Same signature as vq_fused_matmul_tiled_f32 (k, use_smem trailing).
extern "C" __global__ void __launch_bounds__(512, 3)
vq_fused_matmul_tiled_warp_f32(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr, int k, int use_smem
) {
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int lane   = threadIdx.x;                 // 0..31
    int warp   = threadIdx.y;                 // 0..nwarps-1
    int nwarps = blockDim.y;
    // Stage the whole codebook (k*dim<=2048 f32 = 8 KiB) into smem once per block;
    // `cache` is uniform so the __syncthreads() is all-or-no-threads. Byte-identical
    // values, only the memory source changes.
    int cbn = k * dim;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;

    int o   = blockIdx.x * nwarps + warp;     // one output column per warp
    int row = blockIdx.y;                     // i in [0,n)  (n==1 at decode)
    if (o >= out_f || row >= n) return;       // o uniform per warp -> whole-warp exit
    const float* xrow = x + (long)row * in_f;
    const unsigned int* irow = idx + (long)o * vpr;
    float s    = scale[o];
    float part = 0.f;
    if (dim == 8) {
        // dim==8 production path: float4x2 x/codebook loads (offsets v*8, code*8 are
        // 32 B-aligned). The 8 FFMAs stay in k-order with scale folded per term, so
        // each lane's partial matches the tiled/grouped scalar path bit-for-bit; only
        // the cross-lane reassociation differs.
        for (int v = lane; v < vpr; v += 32) {          // coalesced idx across lanes
            const float* cv = cbp  + (long)irow[v] * 8;
            const float* xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b;
            b = s * c0.x; part += x0.x * b;
            b = s * c0.y; part += x0.y * b;
            b = s * c0.z; part += x0.z * b;
            b = s * c0.w; part += x0.w * b;
            b = s * c1.x; part += x1.x * b;
            b = s * c1.y; part += x1.y * b;
            b = s * c1.z; part += x1.z * b;
            b = s * c1.w; part += x1.w * b;
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cbp  + (long)irow[v] * dim;
            const float* xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) { float b = s * cv[t]; part += xb[t] * b; }
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        part += __shfl_down_sync(0xffffffff, part, off);
    if (lane == 0) out[(long)row * out_f + o] = part;
}

// ---- fp16-CODEBOOK sibling of vq_fused_matmul_tiled_warp_f32 -----------------
// OPT-IN (env PRG_SUBBIT_F16_CB): codebook staged to smem as __half (4 KiB) and
// gathered 16 B/vector; x, scale, accumulation stay fp32, SAME k-order + fold. Only
// the cb operand is fp16-rounded. Not argmax-identical. Covers the attention Q/K/V/O
// projections and the whole 1.7B dense model.
extern "C" __global__ void __launch_bounds__(512, 3)
vq_fused_matmul_tiled_warp_f16cb(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr, int k, int use_smem
) {
    __shared__ __half __align__(16) cb_s[VQ_CB_SMEM_MAX];   // fp16 codebook cache (4 KiB)
    int lane   = threadIdx.x;
    int warp   = threadIdx.y;
    int nwarps = blockDim.y;
    int cbn = k * dim;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) cb_s[li] = __float2half(cb[li]);
        __syncthreads();
    }

    int o   = blockIdx.x * nwarps + warp;
    int row = blockIdx.y;
    if (o >= out_f || row >= n) return;
    const float* xrow = x + (long)row * in_f;
    const unsigned int* irow = idx + (long)o * vpr;
    float s    = scale[o];
    float part = 0.f;
    if (cache && dim == 8) {
        for (int v = lane; v < vpr; v += 32) {
            const __half* cv = cb_s + (long)irow[v] * 8;
            const float*  xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 cpk = *(const float4*)cv;
            const __half2* ch = (const __half2*)&cpk;
            float2 c0 = __half22float2(ch[0]);
            float2 c1 = __half22float2(ch[1]);
            float2 c2 = __half22float2(ch[2]);
            float2 c3 = __half22float2(ch[3]);
            float b;
            b = s * c0.x; part += x0.x * b;
            b = s * c0.y; part += x0.y * b;
            b = s * c1.x; part += x0.z * b;
            b = s * c1.y; part += x0.w * b;
            b = s * c2.x; part += x1.x * b;
            b = s * c2.y; part += x1.y * b;
            b = s * c3.x; part += x1.z * b;
            b = s * c3.y; part += x1.w * b;
        }
    } else if (cache) {
        for (int v = lane; v < vpr; v += 32) {
            const __half* cv = cb_s + (long)irow[v] * dim;
            const float*  xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) { float b = s * __half2float(cv[t]); part += xb[t] * b; }
        }
    } else if (dim == 8) {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cb  + (long)irow[v] * 8;
            const float* xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b;
            b = s * c0.x; part += x0.x * b;
            b = s * c0.y; part += x0.y * b;
            b = s * c0.z; part += x0.z * b;
            b = s * c0.w; part += x0.w * b;
            b = s * c1.x; part += x1.x * b;
            b = s * c1.y; part += x1.y * b;
            b = s * c1.z; part += x1.z * b;
            b = s * c1.w; part += x1.w * b;
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cb  + (long)irow[v] * dim;
            const float* xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) { float b = s * cv[t]; part += xb[t] * b; }
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        part += __shfl_down_sync(0xffffffff, part, off);
    if (lane == 0) out[(long)row * out_f + o] = part;
}

// ---- REGISTER-BLOCKED fused VQ matmul forward — cuBLAS-class throughput ----
// 64x64 block tile, BK=16 depth, each of 256 threads computes a 4x4 register
// microtile. Weight tile gathered from the packed codebook into smem.
#define VQ_BM 64
#define VQ_BN 64
#define VQ_BK 16
#define VQ_TM 4
#define VQ_TN 4
extern "C" __global__ void vq_fused_matmul_rb_f32(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr
) {
    __shared__ float As[VQ_BK][VQ_BM];   // transposed: As[k][m]
    __shared__ float Bs[VQ_BK][VQ_BN];   // Bs[k][o]
    int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
    int block_row = blockIdx.y * VQ_BM;
    int block_col = blockIdx.x * VQ_BN;
    int tr = tid / (VQ_BN / VQ_TN);      // 0..15
    int tc = tid % (VQ_BN / VQ_TN);      // 0..15
    float acc[VQ_TM][VQ_TN];
    #pragma unroll
    for (int im = 0; im < VQ_TM; ++im)
        #pragma unroll
        for (int jn = 0; jn < VQ_TN; ++jn) acc[im][jn] = 0.f;

    for (int k0 = 0; k0 < in_f; k0 += VQ_BK) {
        for (int li = tid; li < VQ_BM * VQ_BK; li += 256) {
            int m = li / VQ_BK, kk = li % VQ_BK;
            int gr = block_row + m, gk = k0 + kk;
            As[kk][m] = (gr < n && gk < in_f) ? x[(long)gr * in_f + gk] : 0.f;
        }
        for (int li = tid; li < VQ_BK * VQ_BN; li += 256) {
            int kk = li / VQ_BN, j = li % VQ_BN;
            int go = block_col + j, gk = k0 + kk;
            float val = 0.f;
            if (go < out_f && gk < in_f) {
                unsigned int code = idx[(long)go * vpr + gk / dim];
                val = scale[go] * cb[(long)code * dim + gk % dim];
            }
            Bs[kk][j] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < VQ_BK; ++kk) {
            float ar[VQ_TM], br[VQ_TN];
            #pragma unroll
            for (int im = 0; im < VQ_TM; ++im) ar[im] = As[kk][tr * VQ_TM + im];
            #pragma unroll
            for (int jn = 0; jn < VQ_TN; ++jn) br[jn] = Bs[kk][tc * VQ_TN + jn];
            #pragma unroll
            for (int im = 0; im < VQ_TM; ++im)
                #pragma unroll
                for (int jn = 0; jn < VQ_TN; ++jn) acc[im][jn] += ar[im] * br[jn];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int im = 0; im < VQ_TM; ++im) {
        int gr = block_row + tr * VQ_TM + im;
        if (gr >= n) continue;
        #pragma unroll
        for (int jn = 0; jn < VQ_TN; ++jn) {
            int go = block_col + tc * VQ_TN + jn;
            if (go < out_f) out[(long)gr * out_f + go] = acc[im][jn];
        }
    }
}

// ---- REGISTER-BLOCKED backward wrt x: grad_x[n,in_f] = grad_out @ recon ----
// A = grad_out [n,out_f], B = recon [out_f,in_f] (gathered), K = out_f.
extern "C" __global__ void vq_fused_matmul_dx_rb_f32(
    const float*        __restrict__ g,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ dx,
    int n, int in_f, int out_f, int dim, int vpr
) {
    __shared__ float As[VQ_BK][VQ_BM];   // g^T tile: As[k][m]
    __shared__ float Bs[VQ_BK][VQ_BN];   // recon tile: Bs[k][c]
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_row = blockIdx.y * VQ_BM;  // n
    int block_col = blockIdx.x * VQ_BN;  // in_f
    int tr = tid / (VQ_BN / VQ_TN);
    int tc = tid % (VQ_BN / VQ_TN);
    float acc[VQ_TM][VQ_TN];
    #pragma unroll
    for (int im = 0; im < VQ_TM; ++im)
        #pragma unroll
        for (int jn = 0; jn < VQ_TN; ++jn) acc[im][jn] = 0.f;

    for (int k0 = 0; k0 < out_f; k0 += VQ_BK) {
        for (int li = tid; li < VQ_BM * VQ_BK; li += 256) {
            int m = li / VQ_BK, kk = li % VQ_BK;
            int gr = block_row + m, go = k0 + kk;
            As[kk][m] = (gr < n && go < out_f) ? g[(long)gr * out_f + go] : 0.f;
        }
        for (int li = tid; li < VQ_BK * VQ_BN; li += 256) {
            int kk = li / VQ_BN, j = li % VQ_BN;
            int go = k0 + kk, gc = block_col + j;
            float val = 0.f;
            if (go < out_f && gc < in_f) {
                unsigned int code = idx[(long)go * vpr + gc / dim];
                val = scale[go] * cb[(long)code * dim + gc % dim];
            }
            Bs[kk][j] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < VQ_BK; ++kk) {
            float ar[VQ_TM], br[VQ_TN];
            #pragma unroll
            for (int im = 0; im < VQ_TM; ++im) ar[im] = As[kk][tr * VQ_TM + im];
            #pragma unroll
            for (int jn = 0; jn < VQ_TN; ++jn) br[jn] = Bs[kk][tc * VQ_TN + jn];
            #pragma unroll
            for (int im = 0; im < VQ_TM; ++im)
                #pragma unroll
                for (int jn = 0; jn < VQ_TN; ++jn) acc[im][jn] += ar[im] * br[jn];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int im = 0; im < VQ_TM; ++im) {
        int gr = block_row + tr * VQ_TM + im;
        if (gr >= n) continue;
        #pragma unroll
        for (int jn = 0; jn < VQ_TN; ++jn) {
            int gc = block_col + tc * VQ_TN + jn;
            if (gc < in_f) dx[(long)gr * in_f + gc] = acc[im][jn];
        }
    }
}

// ---- REGISTER-BLOCKED backward wrt codebook: G = g^T@x tiled, scatter to dcb --
// C = G[out_f,in_f] = sum_i g[i,o]*x[i,c], K = n. G never materialized globally:
// each thread's 4x4 microtile is scatter-added into grad_cb via idx.
extern "C" __global__ void vq_fused_matmul_dcb_rb_f32(
    const float*        __restrict__ g,     // [n, out_f]
    const float*        __restrict__ x,     // [n, in_f]
    const unsigned int* __restrict__ idx,   // [out_f*vpr]
    const float*        __restrict__ scale, // [out_f]
    float*              __restrict__ dcb,   // [k*dim] pre-zeroed
    int n, int in_f, int out_f, int dim, int vpr
) {
    __shared__ float As[VQ_BK][VQ_BM];   // g tile: As[kk][m] = g[k0+kk, block_o+m]
    __shared__ float Bs[VQ_BK][VQ_BN];   // x tile: Bs[kk][j] = x[k0+kk, block_c+j]
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_o = blockIdx.y * VQ_BM;    // out_f
    int block_c = blockIdx.x * VQ_BN;    // in_f
    int tr = tid / (VQ_BN / VQ_TN);
    int tc = tid % (VQ_BN / VQ_TN);
    float acc[VQ_TM][VQ_TN];
    #pragma unroll
    for (int im = 0; im < VQ_TM; ++im)
        #pragma unroll
        for (int jn = 0; jn < VQ_TN; ++jn) acc[im][jn] = 0.f;

    for (int k0 = 0; k0 < n; k0 += VQ_BK) {   // K = n (batch rows)
        for (int li = tid; li < VQ_BM * VQ_BK; li += 256) {
            int m = li / VQ_BK, kk = li % VQ_BK;
            int go = block_o + m, gk = k0 + kk;
            As[kk][m] = (go < out_f && gk < n) ? g[(long)gk * out_f + go] : 0.f;
        }
        for (int li = tid; li < VQ_BK * VQ_BN; li += 256) {
            int kk = li / VQ_BN, j = li % VQ_BN;
            int gc = block_c + j, gk = k0 + kk;
            Bs[kk][j] = (gc < in_f && gk < n) ? x[(long)gk * in_f + gc] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < VQ_BK; ++kk) {
            float ar[VQ_TM], br[VQ_TN];
            #pragma unroll
            for (int im = 0; im < VQ_TM; ++im) ar[im] = As[kk][tr * VQ_TM + im];
            #pragma unroll
            for (int jn = 0; jn < VQ_TN; ++jn) br[jn] = Bs[kk][tc * VQ_TN + jn];
            #pragma unroll
            for (int im = 0; im < VQ_TM; ++im)
                #pragma unroll
                for (int jn = 0; jn < VQ_TN; ++jn) acc[im][jn] += ar[im] * br[jn];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int im = 0; im < VQ_TM; ++im) {
        int go = block_o + tr * VQ_TM + im;
        if (go >= out_f) continue;
        float s = scale[go];
        #pragma unroll
        for (int jn = 0; jn < VQ_TN; ++jn) {
            int gc = block_c + tc * VQ_TN + jn;
            if (gc >= in_f) continue;
            unsigned int code = idx[(long)go * vpr + gc / dim];
            atomicAdd(&dcb[(long)code * dim + gc % dim], s * acc[im][jn]);
        }
    }
}

// ---- 8x8-microtile / 128x128-tile fused VQ matmul forward (max throughput) --
// 256 threads, each computes an 8x8 register microtile; BK=dim=16 so one code
// index per column per k-tile. scale folded into the B tile. Targets >cuBLAS by
// avoiding the dense-weight DRAM traffic (B is gathered from the tiny codebook).
#define R8_BM 128
#define R8_BN 128
#define R8_BK 16
#define R8_TM 8
#define R8_TN 8
extern "C" __global__ void vq_fused_matmul_rb8_f32(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr
) {
    __shared__ float As[R8_BK][R8_BM];
    __shared__ float Bs[R8_BK][R8_BN];
    __shared__ float sc_s[R8_BN];
    __shared__ unsigned int code_s[R8_BN];   // one code per column per k-tile (BK==dim)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_row = blockIdx.y * R8_BM;   // n
    int block_col = blockIdx.x * R8_BN;   // out_f
    int tr = tid / (R8_BN / R8_TN);       // 0..15
    int tc = tid % (R8_BN / R8_TN);       // 0..15
    float acc[R8_TM][R8_TN];
    #pragma unroll
    for (int i = 0; i < R8_TM; ++i)
        #pragma unroll
        for (int j = 0; j < R8_TN; ++j) acc[i][j] = 0.f;
    for (int li = tid; li < R8_BN; li += 256) {
        int go = block_col + li;
        sc_s[li] = (go < out_f) ? scale[go] : 0.f;
    }
    for (int k0 = 0; k0 < in_f; k0 += R8_BK) {
        for (int li = tid; li < R8_BM * R8_BK; li += 256) {
            int m = li / R8_BK, kk = li % R8_BK;
            int gr = block_row + m, gk = k0 + kk;
            As[kk][m] = (gr < n && gk < in_f) ? x[(long)gr * in_f + gk] : 0.f;
        }
        // hoist the codebook index: BK==dim so gk/dim == k0/dim for the whole tile
        for (int o = tid; o < R8_BN; o += 256) {
            int go = block_col + o;
            code_s[o] = (go < out_f && k0 < in_f) ? idx[(long)go * vpr + k0 / dim] : 0u;
        }
        __syncthreads();
        for (int li = tid; li < R8_BK * R8_BN; li += 256) {
            int kk = li / R8_BN, o = li % R8_BN;
            int go = block_col + o, gk = k0 + kk;
            Bs[kk][o] = (go < out_f && gk < in_f)
                ? sc_s[o] * cb[(long)code_s[o] * dim + kk]
                : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < R8_BK; ++kk) {
            float ar[R8_TM], br[R8_TN];
            #pragma unroll
            for (int im = 0; im < R8_TM; ++im) ar[im] = As[kk][tr * R8_TM + im];
            #pragma unroll
            for (int jn = 0; jn < R8_TN; ++jn) br[jn] = Bs[kk][tc * R8_TN + jn];
            #pragma unroll
            for (int im = 0; im < R8_TM; ++im)
                #pragma unroll
                for (int jn = 0; jn < R8_TN; ++jn) acc[im][jn] += ar[im] * br[jn];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int im = 0; im < R8_TM; ++im) {
        int gr = block_row + tr * R8_TM + im;
        if (gr >= n) continue;
        #pragma unroll
        for (int jn = 0; jn < R8_TN; ++jn) {
            int go = block_col + tc * R8_TN + jn;
            if (go < out_f) out[(long)gr * out_f + go] = acc[im][jn];
        }
    }
}

// ---- TENSOR-CORE (tf32) fused VQ matmul forward — padded smem, BK=32 ---------
// 64x64 tile, 4 warps (each 2x2 wmma m16n16k8 tf32). Smem leading dims PADDED to
// avoid bank conflicts; BK=32 so 4 k-steps of tensor-core work amortize each
// gather/load. tf32 ~10-bit mantissa, QAT-tolerant.
#define TC_M 64
#define TC_N 64
#define TC_K 32
#define AS_LD (TC_K + 8)
#define BS_LD (TC_N + 8)
extern "C" __global__ void vq_fused_matmul_tc_f32(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr
) {
    __shared__ float As[TC_M][AS_LD];
    __shared__ float Bs[TC_K][BS_LD];
    __shared__ float sc_s[TC_N];
    int tid = threadIdx.x, warp = tid >> 5, wr = warp >> 1, wc = warp & 1;
    int block_row = blockIdx.y * TC_M, block_col = blockIdx.x * TC_N;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(cf[i][j], 0.0f);
    for (int o = tid; o < TC_N; o += 128) { int go = block_col + o; sc_s[o] = (go < out_f) ? scale[go] : 0.f; }
    for (int k0 = 0; k0 < in_f; k0 += TC_K) {
        for (int li = tid; li < TC_M * TC_K; li += 128) {
            int m = li / TC_K, kk = li % TC_K; int gr = block_row + m, gk = k0 + kk;
            As[m][kk] = (gr < n && gk < in_f) ? x[(long)gr * in_f + gk] : 0.f;
        }
        for (int li = tid; li < TC_K * TC_N; li += 128) {
            int kk = li / TC_N, o = li % TC_N; int go = block_col + o, gk = k0 + kk; float val = 0.f;
            if (go < out_f && gk < in_f) { unsigned int code = idx[(long)go * vpr + gk / dim]; val = sc_s[o] * cb[(long)code * dim + gk % dim]; }
            Bs[kk][o] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TC_K; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(af[i], &As[wr * 32 + i * 16][kk], AS_LD);
                #pragma unroll
                for (int e = 0; e < af[i].num_elements; ++e) af[i].x[e] = wmma::__float_to_tf32(af[i].x[e]);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc * 32 + j * 16], BS_LD);
                #pragma unroll
                for (int e = 0; e < bf[j].num_elements; ++e) bf[j].x[e] = wmma::__float_to_tf32(bf[j].x[e]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int gr = block_row + wr * 32 + i * 16, go = block_col + wc * 32 + j * 16;
            if (gr + 16 <= n && go + 16 <= out_f) {
                wmma::store_matrix_sync(&out[(long)gr * out_f + go], cf[i][j], out_f, wmma::mem_row_major);
            } else if (gr < n && go < out_f) {
                __shared__ float stg[4][16][16];
                wmma::store_matrix_sync(&stg[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = (tid & 31); e < 256; e += 32) {
                    int rr = e >> 4, cc = e & 15;
                    if (gr + rr < n && go + cc < out_f) out[(long)(gr + rr) * out_f + go + cc] = stg[warp][rr][cc];
                }
            }
        }
    }
}

// ---- reconstruct the dense recon[out_f,in_f] from the packed weight ----------
// recon[o,c] = scale[o]*cb[idx[o*vpr + c/dim]*dim + (c%dim)]. Transient per layer
// (fits: attn 1GB, mlp 4GB) so cuBLAS(tf32) can then do the GEMM at tensor-core
// speed off the resident packed weight — no dense fp32 master ever resident.
extern "C" __global__ void vq_reconstruct_f32(
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ recon,
    int out_f, int in_f, int dim, int vpr, int o_base
) {
    // reconstruct a row-tile: `recon` is [out_f, in_f] (out_f = tile rows), but
    // idx/scale are indexed at global row (o_base + o) so a wide layer can be
    // reconstructed in slices that fit VRAM.
    long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)out_f * in_f;
    if (e >= total) return;
    int o = e / in_f, c = e % in_f;
    int go = o_base + o;
    unsigned int code = idx[(long)go * vpr + c / dim];
    recon[e] = scale[go] * cb[(long)code * dim + c % dim];
}

// ---- cp.async PIPELINED tf32 fused VQ matmul forward ------------------------
// Codebook (k=256*dim=16=4096 floats) resident in smem; the activation tile is
// cp.async-loaded (global->smem, overlapped with tensor-core compute); the recon
// tile is gathered from the resident codebook (can't ride cp.async). 64x64 tile,
// 4 warps x 2x2 wmma m16n16k8 tf32, 2-stage activation pipeline. Assumes k*dim<=4096.
#include <cuda_pipeline.h>
#define CA_M 64
#define CA_N 64
#define CA_K 16
extern "C" __global__ void vq_fused_matmul_ca_f32(
    const float*        __restrict__ x,
    const unsigned int* __restrict__ idx,
    const float*        __restrict__ cb,
    const float*        __restrict__ scale,
    float*              __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr
) {
    __shared__ float cb_s[4096];
    __shared__ float As[2][CA_M][CA_K];
    __shared__ float Bs[CA_K][CA_N];
    __shared__ float sc_s[CA_N];
    __shared__ unsigned int code_s[CA_N];
    const int NT = 128;
    int tid = threadIdx.x, warp = tid >> 5, wr = warp >> 1, wc = warp & 1;
    int block_row = blockIdx.y * CA_M, block_col = blockIdx.x * CA_N;
    for (int i = tid; i < 4096; i += NT) cb_s[i] = cb[i];
    for (int o = tid; o < CA_N; o += NT) { int go = block_col + o; sc_s[o] = (go < out_f) ? scale[go] : 0.f; }
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(cf[i][j], 0.0f);
    // prologue: async-load activation tile 0
    for (int li = tid; li < CA_M * CA_K; li += NT) {
        int m = li / CA_K, kk = li % CA_K; int gr = block_row + m, gk = kk;
        if (gr < n && gk < in_f) __pipeline_memcpy_async(&As[0][m][kk], &x[(long)gr * in_f + gk], sizeof(float));
        else As[0][m][kk] = 0.f;
    }
    __pipeline_commit();
    int cur = 0;
    for (int k0 = 0; k0 < in_f; k0 += CA_K) {
        int nk = k0 + CA_K, nb = cur ^ 1;
        if (nk < in_f) {
            for (int li = tid; li < CA_M * CA_K; li += NT) {
                int m = li / CA_K, kk = li % CA_K; int gr = block_row + m, gk = nk + kk;
                if (gr < n && gk < in_f) __pipeline_memcpy_async(&As[nb][m][kk], &x[(long)gr * in_f + gk], sizeof(float));
                else As[nb][m][kk] = 0.f;
            }
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        // gather recon tile from resident codebook (BK==dim -> one code per column)
        for (int o = tid; o < CA_N; o += NT) { int go = block_col + o; code_s[o] = (go < out_f && k0 < in_f) ? idx[(long)go * vpr + k0 / dim] : 0u; }
        __syncthreads();
        for (int li = tid; li < CA_K * CA_N; li += NT) {
            int kk = li / CA_N, o = li % CA_N;
            Bs[kk][o] = sc_s[o] * cb_s[code_s[o] * dim + kk];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < CA_K; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(af[i], &As[cur][wr * 32 + i * 16][kk], CA_K);
                #pragma unroll
                for (int e = 0; e < af[i].num_elements; ++e) af[i].x[e] = wmma::__float_to_tf32(af[i].x[e]);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc * 32 + j * 16], CA_N);
                #pragma unroll
                for (int e = 0; e < bf[j].num_elements; ++e) bf[j].x[e] = wmma::__float_to_tf32(bf[j].x[e]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncthreads();
        cur = nb;
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int gr = block_row + wr * 32 + i * 16, go = block_col + wc * 32 + j * 16;
            if (gr + 16 <= n && go + 16 <= out_f)
                wmma::store_matrix_sync(&out[(long)gr * out_f + go], cf[i][j], out_f, wmma::mem_row_major);
        }
    }
}

// ---- fused MoE top-k combine ------------------------------------------------
// out[j] = sum_{g<G} rows[g*h + j] * w[g], accumulated in g-order from acc=0.
// Byte-identical to the sequential `acc += w[g]*down[g]` combine (a chain of
// scaled_add_inplace_ launches: `dst = dst + src*scalar` => one FFMA per step,
// dst starting at 0). Same g-order, same fma, same f32 accumulation — so the
// token stream is argmax-identical — but ONE launch replaces the top_k separate
// scaled-add launches per MoE layer.
extern "C" __global__ void vq_moe_combine_f32(
    const float* __restrict__ rows,   // [G, h] per-expert (down) outputs
    const float* __restrict__ w,      // [G] router weights, route order
    float*       __restrict__ out,    // [h]
    int h, int g_count
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= h) return;
    float acc = 0.f;
    for (int g = 0; g < g_count; ++g) {
        acc = acc + rows[(long)g * h + j] * w[g];   // FFMA, matches scaled_add
    }
    out[j] = acc;
}

// ---- DEVICE MoE router top-k (kills the per-layer logits D2H + host top-k) ---
// Softmax over `ne` router logits, select top_k, optionally renormalize — writing
// the expert indices and weights to DEVICE buffers that feed the grouped expert
// GEMV and the combine, so no host round-trip per layer. One block per row;
// serial in thread 0 (ne<=256, top_k<=32 — tiny). Matches host topk_route
// (model.rs) modulo the exp libm's last ULP: denom summed in double in index
// order; e_i = (float)exp((double)(L[i]-max)); inv = (float)(1.0/sum);
// prob_i = e_i * inv (both f32); top_k chosen by LOGIT (monotonic in prob),
// lowest index wins exact ties; renorm divides by the f32 sum of the selected
// probs accumulated in selection order.
// exp via the hardware ex2.approx.f32 (inline PTX) — no libdevice, so this file
// still JITs cleanly under -nocudalib. exp(x) = 2^(x·log2 e). ~2^-22 rel error,
// which is fine: it only feeds the router WEIGHTS (argmax-stable, not bit-exact);
// selection is by logit (exp-free & exact).
__device__ __forceinline__ float ex2_expf(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x * 1.4426950408889634f));
    return r;
}

extern "C" __global__ void vq_router_topk_f32(
    const float* __restrict__ logits,  // [rows, ne]
    int*         __restrict__ sel,     // [rows, top_k]  (out) expert indices
    float*       __restrict__ wout,    // [rows, top_k]  (out) router weights
    int ne, int top_k, int norm
) {
    int row = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float* L = logits + (long)row * ne;
    int*   S = sel  + (long)row * top_k;
    float* W = wout + (long)row * top_k;

    float maxv = -INFINITY;
    for (int i = 0; i < ne; ++i) { float v = L[i]; if (v > maxv) maxv = v; }

    double sum = 0.0;
    for (int i = 0; i < ne; ++i) sum += (double)ex2_expf(L[i] - maxv);
    float inv = (float)(1.0 / sum);

    bool chosen[256];
    for (int i = 0; i < ne; ++i) chosen[i] = false;
    float wsum = 0.f;
    for (int k = 0; k < top_k; ++k) {
        int best = -1; float bestv = -INFINITY;
        for (int i = 0; i < ne; ++i) {
            if (!chosen[i] && L[i] > bestv) { bestv = L[i]; best = i; }
        }
        chosen[best] = true;
        float prob = ex2_expf(L[best] - maxv) * inv;
        S[k] = best;
        W[k] = prob;
        wsum += prob;
    }
    if (norm && wsum > 0.f) {
        for (int k = 0; k < top_k; ++k) W[k] = W[k] / wsum;
    }
}

// ---- GROUPED tiled fused VQ matmul (decode MoE experts) ---------------------
// Runs G independent VQ GEMVs (one per selected expert) in ONE launch instead of
// G separate launches: the memory-bound dim=8 decode GEMV under-occupies the GPU
// as a lone launch, so G experts packed across a grid.z dimension fill the SMs
// and cut launch/host overhead (the 8 gate + 8 up + 8 down per MoE layer collapse
// to 3 launches). Per (group,row,col) the arithmetic is BYTE-IDENTICAL to
// vq_fused_matmul_tiled_f32 run once for that expert — same tile loop, same
// scale-into-B fold, same f32 accumulation order — so the token stream is
// argmax-identical to the per-expert path. blockIdx.z = group g; the weights are
// selected from concatenated per-expert buffers via sel[g] (expert id):
//   idx = idx_all + sel[g]*idx_stride, cb = cb_all + sel[g]*cb_stride,
//   scale = scale_all + sel[g]*scale_stride.
// x is either broadcast to every group (x_stride==0, the SAME input row for
// gate/up) or per-group (x_stride==n*in_f, the per-expert SwiGLU act for down).
// out is [G, n, out_f]. Strides are int (all < 2^31 for the 30B) but every
// address is formed in 64-bit ((long) casts) so the gather never overflows.
extern "C" __global__ void vq_fused_matmul_grouped_f32(
    const float*         __restrict__ x,         // broadcast [n,in_f] or [G,n,in_f]
    const unsigned char* __restrict__ idx_all,   // [n_experts * idx_stride], u8 (k<=256)
    const float*         __restrict__ cb_all,    // [n_experts * cb_stride]
    const float*         __restrict__ scale_all, // [n_experts * scale_stride]
    const int*           __restrict__ sel,       // [G] selected expert id per group
    float*               __restrict__ out,       // [G, n, out_f]
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    // DECODE-SPECIALIZED (batch-1) thread-per-output-column GEMV.
    // Order-preserving twin of the tiled path: each thread owns ONE output column
    // and sums k=0..in_f-1 into a SINGLE acc, folding scale into the B term BEFORE
    // the FFMA exactly as the tiled kernel does (Bs = scale*cb rounded to f32, then
    // acc += As*Bs). No 16x16 tile, no wasted lanes (the tiled path left 15/16
    // threads idle at n=1), no per-k-tile __syncthreads. The codebook (k*dim<=2048
    // f32 = 8 KiB) is cached once in smem; the x row stays L1/L2-resident across the
    // block's columns. Bit-identical => argmax-identical token stream.
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int g = blockIdx.z;
    int e = sel[g];
    const unsigned char* idx = idx_all + (long)e * idx_stride;
    const float*        cb  = cb_all    + (long)e * cb_stride;
    const float*        scale = scale_all + (long)e * scale_stride;
    const float*        xg  = x   + (long)g * x_stride;      // x_stride==0 => broadcast
    float*              outg = out + (long)g * n * out_f;
    int tid = threadIdx.x, nthreads = blockDim.x;
    // Stage this expert's codebook into shared memory once per block (cb_stride ==
    // k*dim). Byte-identical values, same gather math — only the memory source
    // changes; `cache` is uniform so the __syncthreads() is safe.
    int cbn = cb_stride;                   // k * dim floats for expert e
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        for (int li = tid; li < cbn; li += nthreads) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;

    int col = blockIdx.x * nthreads + tid;   // o in [0,out_f)
    int row = blockIdx.y;                    // i in [0,n)  (n==1 at decode)
    if (col >= out_f || row >= n) return;    // col uniform-free; whole-warp-safe
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* irow = idx + (long)col * vpr;
    float s   = scale[col];
    float acc = 0.f;
    if (dim == 8) {
        // dim==8 production path: load x + cb as float4x2 (only the load WIDTH
        // changes; the 8 FFMAs stay in k-order with scale folded per term, so
        // bit-identical to the scalar/tiled path). Offsets v*8 and code*8 are
        // 32 B-aligned => valid float4.
        for (int v = 0; v < vpr; ++v) {
            const float* cv = cbp  + (long)irow[v] * 8;
            const float* xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b;
            b = s * c0.x; acc += x0.x * b;
            b = s * c0.y; acc += x0.y * b;
            b = s * c0.z; acc += x0.z * b;
            b = s * c0.w; acc += x0.w * b;
            b = s * c1.x; acc += x1.x * b;
            b = s * c1.y; acc += x1.y * b;
            b = s * c1.z; acc += x1.z * b;
            b = s * c1.w; acc += x1.w * b;
        }
    } else {
        for (int v = 0; v < vpr; ++v) {      // ascending k = v*dim+t
            const float* cv = cbp  + (long)irow[v] * dim;
            const float* xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) {
                float b = s * cv[t];         // Bs = scale*cb, rounded to f32
                acc += xb[t] * b;            // FFMA into one acc, k-increasing
            }
        }
    }
    outg[(long)row * out_f + col] = acc;
}

#define VQ_MLP4_STEP(K, VV, ACC)                                              \
    {                                                                          \
        const float* xb_ = xrow + (long)(VV) * 8;                              \
        float4 xa_ = *(const float4*)(xb_);                                    \
        float4 xb2_ = *(const float4*)(xb_ + 4);                               \
        const __half2* ch_ = (const __half2*)&(K);                             \
        float2 d0_ = __half22float2(ch_[0]);                                   \
        float2 d1_ = __half22float2(ch_[1]);                                   \
        float2 d2_ = __half22float2(ch_[2]);                                   \
        float2 d3_ = __half22float2(ch_[3]);                                   \
        float bb_;                                                             \
        bb_ = s * d0_.x; (ACC) += xa_.x * bb_;                                 \
        bb_ = s * d0_.y; (ACC) += xa_.y * bb_;                                 \
        bb_ = s * d1_.x; (ACC) += xa_.z * bb_;                                 \
        bb_ = s * d1_.y; (ACC) += xa_.w * bb_;                                 \
        bb_ = s * d2_.x; (ACC) += xb2_.x * bb_;                                \
        bb_ = s * d2_.y; (ACC) += xb2_.y * bb_;                                \
        bb_ = s * d3_.x; (ACC) += xb2_.z * bb_;                                \
        bb_ = s * d3_.y; (ACC) += xb2_.w * bb_;                                \
    }

#define VQ_C4_COL(IR, SC, ACC)                                                \
    {                                                                          \
        float4 kk_ = *(const float4*)(cb_s + (long)(IR)[v] * 8);               \
        const __half2* ch_ = (const __half2*)&kk_;                             \
        float2 e0_ = __half22float2(ch_[0]);                                   \
        float2 e1_ = __half22float2(ch_[1]);                                   \
        float2 e2_ = __half22float2(ch_[2]);                                   \
        float2 e3_ = __half22float2(ch_[3]);                                   \
        float bq_;                                                             \
        bq_ = (SC) * e0_.x; (ACC) += xa.x * bq_;                               \
        bq_ = (SC) * e0_.y; (ACC) += xa.y * bq_;                               \
        bq_ = (SC) * e1_.x; (ACC) += xa.z * bq_;                               \
        bq_ = (SC) * e1_.y; (ACC) += xa.w * bq_;                               \
        bq_ = (SC) * e2_.x; (ACC) += xc.x * bq_;                               \
        bq_ = (SC) * e2_.y; (ACC) += xc.y * bq_;                               \
        bq_ = (SC) * e3_.x; (ACC) += xc.z * bq_;                               \
        bq_ = (SC) * e3_.y; (ACC) += xc.w * bq_;                               \
    }

// ---- GROUPED WARP f16-CB, 4 COLUMNS PER WARP (decode) -----------------------
// What this kernel actually moves, per expert group (up: in_f 2688, out_f 1856):
//
//   packed indices   out_f * vpr        = 623 KB
//   x, re-read once per output column   = out_f * in_f * 4 B = 20 MB
//
// The INPUT ROW is the traffic, not the weights — every column re-reads all of
// x. That is why expert-major dedup changed nothing: it shares weight reads
// across rows while leaving the (row, column) count, and therefore the x reads,
// exactly as they were.
//
// So reuse x instead: each warp takes FOUR consecutive output columns and loads
// each x vector once for all of them, cutting x traffic 4x. The four columns
// have different codes, so the codebook gathers stay independent — the same
// work, a quarter of the reads.
extern "C" __global__ void vq_fused_matmul_grouped_warp_f16cb_c4(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ __half __align__(16) cb_s[VQ_CB_SMEM_MAX];
    int g = blockIdx.z;
    int e = sel[g];
    const unsigned char* idx = idx_all + (long)e * idx_stride;
    const float*        cb  = cb_all    + (long)e * cb_stride;
    const float*        scale = scale_all + (long)e * scale_stride;
    const float*        xg  = x   + (long)g * x_stride;
    float*              outg = out + (long)g * n * out_f;

    int lane = threadIdx.x, warp = threadIdx.y, nwarps = blockDim.y;
    {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cb_stride; li += tpb) cb_s[li] = __float2half(cb[li]);
    }
    __syncthreads();

    int o0  = (blockIdx.x * nwarps + warp) * 4;
    int row = blockIdx.y;
    if (o0 >= out_f || row >= n) return;
    int cols = out_f - o0; if (cols > 4) cols = 4;
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* ir0 = idx + (long)o0 * vpr;
    const unsigned char* ir1 = ir0 + vpr;
    const unsigned char* ir2 = ir0 + 2 * vpr;
    const unsigned char* ir3 = ir0 + 3 * vpr;
    float s0 = scale[o0];
    float s1 = (cols > 1) ? scale[o0 + 1] : 0.f;
    float s2 = (cols > 2) ? scale[o0 + 2] : 0.f;
    float s3 = (cols > 3) ? scale[o0 + 3] : 0.f;
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;

    for (int v = lane; v < vpr; v += 32) {
        const float* xb = xrow + (long)v * 8;
        float4 xa = *(const float4*)(xb);          // x loaded ONCE for 4 columns
        float4 xc = *(const float4*)(xb + 4);
        VQ_C4_COL(ir0, s0, p0)
        if (cols > 1) VQ_C4_COL(ir1, s1, p1)
        if (cols > 2) VQ_C4_COL(ir2, s2, p2)
        if (cols > 3) VQ_C4_COL(ir3, s3, p3)
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        p0 += __shfl_down_sync(0xffffffff, p0, off);
        p1 += __shfl_down_sync(0xffffffff, p1, off);
        p2 += __shfl_down_sync(0xffffffff, p2, off);
        p3 += __shfl_down_sync(0xffffffff, p3, off);
    }
    if (lane == 0) {
        float* ob = outg + (long)row * out_f + o0;
        ob[0] = p0;
        if (cols > 1) ob[1] = p1;
        if (cols > 2) ob[2] = p2;
        if (cols > 3) ob[3] = p3;
    }
}

// Scale is folded into the codebook term BEFORE the FMA, exactly as the
// warp-per-column kernel does (`b = s*cb; acc += x*b`), so a column's value is
// bit-identical to what that kernel produces. Folding happens once per (v,
// column) and is reused across the four rows.
#define VQ_RC_SCALE(CA, CB, SC)                                               \
    {                                                                          \
        (CA).x *= (SC); (CA).y *= (SC); (CA).z *= (SC); (CA).w *= (SC);        \
        (CB).x *= (SC); (CB).y *= (SC); (CB).z *= (SC); (CB).w *= (SC);        \
    }

#define VQ_RC_DOT(C, CA, CB)                                                  \
    {                                                                          \
        acc[r][C] += xa.x * (CA).x;                                            \
        acc[r][C] += xa.y * (CA).y;                                            \
        acc[r][C] += xa.z * (CA).z;                                            \
        acc[r][C] += xa.w * (CA).w;                                            \
        acc[r][C] += xc.x * (CB).x;                                            \
        acc[r][C] += xc.y * (CB).y;                                            \
        acc[r][C] += xc.z * (CB).z;                                            \
        acc[r][C] += xc.w * (CB).w;                                            \
    }

// ---- NON-GROUPED WARP GEMV, 4 ROWS x 4 COLUMNS PER WARP (decode) ------------
// The non-grouped projections (mamba in/out, attention qkv/o, the shared expert,
// lm_head) run the 16x16 tile kernel once n exceeds the warp-GEMV threshold.
// That kernel reconstructs the weight ELEMENT BY ELEMENT: it re-reads the packed
// code for every (column, k) pair, so at dim=8 each code is fetched eight times,
// four bytes wide. The register-blocked rb8 kernel exists precisely to hoist the
// code per k-tile, but it requires BK == dim (16) and is shaped BM=128 for
// prefill — at n=16 it would idle 112 of its 128 row slots.
//
// This kernel blocks in both directions at decode scale: each warp owns FOUR
// output columns and FOUR rows, so one loaded x vector serves four columns and
// one gathered codebook vector serves four rows, and each code is read once per
// eight k rather than eight times.
extern "C" __global__ void vq_fused_matmul_warp_rc_f32(
    const float*        __restrict__ x,       // [n, in_f]
    const unsigned int* __restrict__ idx,     // [out_f, vpr]
    const float*        __restrict__ cb,      // [k, dim]
    const float*        __restrict__ scale,   // [out_f]
    float*              __restrict__ out,     // [n, out_f]
    int n, int in_f, int out_f, int dim, int vpr, int k_codes, int use_smem
) {
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int lane = threadIdx.x, warp = threadIdx.y, nwarps = blockDim.y;
    int cbn = k_codes * dim;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;

    int o0 = (blockIdx.x * nwarps + warp) * 4;
    int r0 = blockIdx.y * 4;
    if (o0 >= out_f || r0 >= n) return;
    int cols = out_f - o0; if (cols > 4) cols = 4;
    int rows = n - r0;     if (rows > 4) rows = 4;

    const unsigned int* ir0 = idx + (long)o0 * vpr;
    float sc0 = scale[o0];
    float sc1 = (cols > 1) ? scale[o0 + 1] : 0.f;
    float sc2 = (cols > 2) ? scale[o0 + 2] : 0.f;
    float sc3 = (cols > 3) ? scale[o0 + 3] : 0.f;
    float acc[4][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a)
        #pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.f;

    for (int v = lane; v < vpr; v += 32) {
        const float* cv0 = cbp + (long)ir0[v] * 8;
        const float* cv1 = (cols > 1) ? cbp + (long)ir0[vpr + v] * 8 : cv0;
        const float* cv2 = (cols > 2) ? cbp + (long)ir0[2 * vpr + v] * 8 : cv0;
        const float* cv3 = (cols > 3) ? cbp + (long)ir0[3 * vpr + v] * 8 : cv0;
        float4 c0a = *(const float4*)cv0, c0b = *(const float4*)(cv0 + 4);
        float4 c1a = *(const float4*)cv1, c1b = *(const float4*)(cv1 + 4);
        float4 c2a = *(const float4*)cv2, c2b = *(const float4*)(cv2 + 4);
        float4 c3a = *(const float4*)cv3, c3b = *(const float4*)(cv3 + 4);
        VQ_RC_SCALE(c0a, c0b, sc0)
        VQ_RC_SCALE(c1a, c1b, sc1)
        VQ_RC_SCALE(c2a, c2b, sc2)
        VQ_RC_SCALE(c3a, c3b, sc3)
        for (int r = 0; r < rows; ++r) {
            const float* xb = x + (long)(r0 + r) * in_f + (long)v * 8;
            float4 xa = *(const float4*)(xb);
            float4 xc = *(const float4*)(xb + 4);
            VQ_RC_DOT(0, c0a, c0b)
            if (cols > 1) VQ_RC_DOT(1, c1a, c1b)
            if (cols > 2) VQ_RC_DOT(2, c2a, c2b)
            if (cols > 3) VQ_RC_DOT(3, c3a, c3b)
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                acc[a][b] += __shfl_down_sync(0xffffffff, acc[a][b], off);
    }
    if (lane == 0) {
        for (int r = 0; r < rows; ++r) {
            float* ob = out + (long)(r0 + r) * out_f + o0;
            for (int c = 0; c < cols; ++c) ob[c] = acc[r][c];
        }
    }
}

// ---- GROUPED WARP f16-CB, 8 COLUMNS PER WARP (decode) -----------------------
// Same x-reuse idea as the 4-column kernel, taken one step further: eight
// columns share each loaded x vector, cutting the dominant traffic 8x.
extern "C" __global__ void vq_fused_matmul_grouped_warp_f16cb_c8(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ __half __align__(16) cb_s[VQ_CB_SMEM_MAX];
    int g = blockIdx.z;
    int e = sel[g];
    const unsigned char* idx = idx_all + (long)e * idx_stride;
    const float*        cb  = cb_all    + (long)e * cb_stride;
    const float*        scale = scale_all + (long)e * scale_stride;
    const float*        xg  = x   + (long)g * x_stride;
    float*              outg = out + (long)g * n * out_f;

    int lane = threadIdx.x, warp = threadIdx.y, nwarps = blockDim.y;
    {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cb_stride; li += tpb) cb_s[li] = __float2half(cb[li]);
    }
    __syncthreads();

    int o0  = (blockIdx.x * nwarps + warp) * 8;
    int row = blockIdx.y;
    if (o0 >= out_f || row >= n) return;
    int cols = out_f - o0; if (cols > 8) cols = 8;
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* ir0 = idx + (long)o0 * vpr;
    const unsigned char* ir1 = ir0 + vpr;
    const unsigned char* ir2 = ir0 + 2 * vpr;
    const unsigned char* ir3 = ir0 + 3 * vpr;
    const unsigned char* ir4 = ir0 + 4 * vpr;
    const unsigned char* ir5 = ir0 + 5 * vpr;
    const unsigned char* ir6 = ir0 + 6 * vpr;
    const unsigned char* ir7 = ir0 + 7 * vpr;
    float s0 = scale[o0];
    float s1 = (cols > 1) ? scale[o0 + 1] : 0.f;
    float s2 = (cols > 2) ? scale[o0 + 2] : 0.f;
    float s3 = (cols > 3) ? scale[o0 + 3] : 0.f;
    float s4 = (cols > 4) ? scale[o0 + 4] : 0.f;
    float s5 = (cols > 5) ? scale[o0 + 5] : 0.f;
    float s6 = (cols > 6) ? scale[o0 + 6] : 0.f;
    float s7 = (cols > 7) ? scale[o0 + 7] : 0.f;
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    float p4 = 0.f, p5 = 0.f, p6 = 0.f, p7 = 0.f;

    for (int v = lane; v < vpr; v += 32) {
        const float* xb = xrow + (long)v * 8;
        float4 xa = *(const float4*)(xb);          // x loaded ONCE for 4 columns
        float4 xc = *(const float4*)(xb + 4);
        VQ_C4_COL(ir0, s0, p0)
        if (cols > 1) VQ_C4_COL(ir1, s1, p1)
        if (cols > 2) VQ_C4_COL(ir2, s2, p2)
        if (cols > 3) VQ_C4_COL(ir3, s3, p3)
        if (cols > 4) VQ_C4_COL(ir4, s4, p4)
        if (cols > 5) VQ_C4_COL(ir5, s5, p5)
        if (cols > 6) VQ_C4_COL(ir6, s6, p6)
        if (cols > 7) VQ_C4_COL(ir7, s7, p7)
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        p0 += __shfl_down_sync(0xffffffff, p0, off);
        p1 += __shfl_down_sync(0xffffffff, p1, off);
        p2 += __shfl_down_sync(0xffffffff, p2, off);
        p3 += __shfl_down_sync(0xffffffff, p3, off);
        p4 += __shfl_down_sync(0xffffffff, p4, off);
        p5 += __shfl_down_sync(0xffffffff, p5, off);
        p6 += __shfl_down_sync(0xffffffff, p6, off);
        p7 += __shfl_down_sync(0xffffffff, p7, off);
    }
    if (lane == 0) {
        float* ob = outg + (long)row * out_f + o0;
        ob[0] = p0;
        if (cols > 1) ob[1] = p1;
        if (cols > 2) ob[2] = p2;
        if (cols > 3) ob[3] = p3;
        if (cols > 4) ob[4] = p4;
        if (cols > 5) ob[5] = p5;
        if (cols > 6) ob[6] = p6;
        if (cols > 7) ob[7] = p7;
    }
}

// ---- GROUPED WARP f16-CB, 4-WAY MEMORY-LEVEL PARALLELISM (decode) -----------
// The expert GEMV is latency bound, not bandwidth bound: measured ~14 GB/s on a
// ~900 GB/s part and ~1% of peak FLOPs. Each loop iteration carries a dependent
// chain — load the u8 code, use it to index the codebook, then FMA — and the
// strided one-v-per-iteration loop leaves at most one such chain in flight per
// lane.
//
// This variant issues FOUR chains at once: four codes are read, then four
// codebook gathers are launched before any of them is consumed, so the gather
// latencies overlap instead of serialising. Lane coalescing is unchanged (each
// of the four sub-steps still covers one contiguous 32-lane block).
//
// The dot is re-associated into four partials, which the warp kernel's contract
// already permits (it is argmax-gated, not bit-exact).
extern "C" __global__ void vq_fused_matmul_grouped_warp_f16cb_mlp4(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ __half __align__(16) cb_s[VQ_CB_SMEM_MAX];
    int g = blockIdx.z;
    int e = sel[g];
    const unsigned char* idx = idx_all + (long)e * idx_stride;
    const float*        cb  = cb_all    + (long)e * cb_stride;
    const float*        scale = scale_all + (long)e * scale_stride;
    const float*        xg  = x   + (long)g * x_stride;
    float*              outg = out + (long)g * n * out_f;

    int lane = threadIdx.x, warp = threadIdx.y, nwarps = blockDim.y;
    {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cb_stride; li += tpb) cb_s[li] = __float2half(cb[li]);
    }
    __syncthreads();

    int o = blockIdx.x * nwarps + warp;
    int row = blockIdx.y;
    if (o >= out_f || row >= n) return;
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* irow = idx + (long)o * vpr;
    float s = scale[o];

    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    int v = lane;
    for (; v + 96 < vpr; v += 128) {
        int a0 = irow[v], a1 = irow[v + 32], a2 = irow[v + 64], a3 = irow[v + 96];
        float4 k0 = *(const float4*)(cb_s + (long)a0 * 8);
        float4 k1 = *(const float4*)(cb_s + (long)a1 * 8);
        float4 k2 = *(const float4*)(cb_s + (long)a2 * 8);
        float4 k3 = *(const float4*)(cb_s + (long)a3 * 8);
        VQ_MLP4_STEP(k0, v, p0)
        VQ_MLP4_STEP(k1, v + 32, p1)
        VQ_MLP4_STEP(k2, v + 64, p2)
        VQ_MLP4_STEP(k3, v + 96, p3)
    }
    for (; v < vpr; v += 32) {
        float4 kk = *(const float4*)(cb_s + (long)irow[v] * 8);
        VQ_MLP4_STEP(kk, v, p0)
    }
    float part = (p0 + p1) + (p2 + p3);
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
    if (lane == 0) outg[(long)row * out_f + o] = part;
}

// ---- GROUPED MULTI-ROW fused VQ GEMV (expert-major decode) -------------------
// Batching a sparse MoE does not amortize the routed experts: each agent picks
// its own top-k, and the grouped kernel above runs one group per (row, expert)
// pair even when many rows chose the SAME expert. Measured on a cohort of agents
// doing similar work, 85% of their top-k selections coincide — so the same
// 1.25 MB expert is re-read up to M times per layer.
//
// This kernel is expert-major instead: the caller buckets the pairs by expert,
// and each block walks ONE expert's packed weights once while accumulating into
// every row that selected it. The index byte and codebook vector for a column
// are fetched a single time and feed R independent accumulators.
//
// Bit-identical to the thread-per-column kernel: for a given (row, column) the
// FFMA sequence is unchanged — same k order, same `Bs = scale*cb` folding — only
// the loop nesting and the number of live accumulators differ.
// ---- GROUPED WARP MULTI-ROW fused VQ GEMV (expert-major decode) -------------
// The warp-per-column twin of the multi-row kernel below, and the one the batched
// MoE path actually uses. Lane l walks v = l, l+32, ... exactly as the
// single-row warp kernel does, but holds R accumulators — one per row that
// selected this expert — so the idx byte and codebook vector fetched for a
// column feed R rows instead of one. R shfl trees reduce at the end.
//
// Same re-association as the single-row warp kernel (partials combined at the
// 32-lane boundary and through the tree), so a row's value is bit-identical to
// what that kernel produces for the same row.
#define VQ_WMULTIROW_MAX 16
extern "C" __global__ void vq_fused_matmul_grouped_warp_multirow_f32(
    const float*         __restrict__ x,         // [P, in_f]
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel_u,     // [U] expert id per bucket
    const int*           __restrict__ row_off,   // [U+1] CSR offsets
    const int*           __restrict__ pair_g,    // [P] original group index
    float*               __restrict__ out,       // [P, out_f]
    int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int use_smem
) {
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int u = blockIdx.z;
    int e = sel_u[u];
    int start = row_off[u];
    int R = row_off[u + 1] - start;
    if (R <= 0) return;
    if (R > VQ_WMULTIROW_MAX) R = VQ_WMULTIROW_MAX;

    const unsigned char* idx   = idx_all   + (long)e * idx_stride;
    const float*         cb    = cb_all    + (long)e * cb_stride;
    const float*         scale = scale_all + (long)e * scale_stride;

    int lane   = threadIdx.x;
    int warp   = threadIdx.y;
    int nwarps = blockDim.y;
    int cbn = cb_stride;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;

    int o = blockIdx.x * nwarps + warp;
    if (o >= out_f) return;
    const unsigned char* irow = idx + (long)o * vpr;
    float s = scale[o];

    int grp[VQ_WMULTIROW_MAX];
    float part[VQ_WMULTIROW_MAX];
    #pragma unroll
    for (int r = 0; r < VQ_WMULTIROW_MAX; ++r) part[r] = 0.f;
    for (int r = 0; r < R; ++r) grp[r] = pair_g[start + r];

    if (dim == 8) {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cbp + (long)irow[v] * 8;
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b0 = s * c0.x, b1 = s * c0.y, b2 = s * c0.z, b3 = s * c0.w;
            float b4 = s * c1.x, b5 = s * c1.y, b6 = s * c1.z, b7 = s * c1.w;
            for (int r = 0; r < R; ++r) {
                const float* xb = x + (long)grp[r] * in_f + (long)v * 8;
                float4 x0 = *(const float4*)(xb);
                float4 x1 = *(const float4*)(xb + 4);
                float a = part[r];
                a += x0.x * b0;
                a += x0.y * b1;
                a += x0.z * b2;
                a += x0.w * b3;
                a += x1.x * b4;
                a += x1.y * b5;
                a += x1.z * b6;
                a += x1.w * b7;
                part[r] = a;
            }
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cbp + (long)irow[v] * dim;
            for (int r = 0; r < R; ++r) {
                const float* xb = x + (long)grp[r] * in_f + (long)v * dim;
                float a = part[r];
                for (int t = 0; t < dim; ++t) {
                    float b = s * cv[t];
                    a += xb[t] * b;
                }
                part[r] = a;
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        float a = part[r];
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) out[(long)grp[r] * out_f + o] = a;
    }
}

#define VQ_MULTIROW_MAX 16
extern "C" __global__ void vq_fused_matmul_grouped_multirow_f32(
    const float*         __restrict__ x,         // [P, in_f] one row per pair slot
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel_u,     // [U] expert id per bucket
    const int*           __restrict__ row_off,   // [U+1] CSR offsets into pair_g
    const int*           __restrict__ pair_g,    // [P] original group index
    float*               __restrict__ out,       // [P, out_f]
    int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int use_smem
) {
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int u = blockIdx.z;
    int e = sel_u[u];
    int start = row_off[u];
    int end   = row_off[u + 1];
    int R = end - start;
    if (R <= 0) return;
    if (R > VQ_MULTIROW_MAX) R = VQ_MULTIROW_MAX;

    const unsigned char* idx   = idx_all   + (long)e * idx_stride;
    const float*         cb    = cb_all    + (long)e * cb_stride;
    const float*         scale = scale_all + (long)e * scale_stride;

    int tid = threadIdx.x, nthreads = blockDim.x;
    int cbn = cb_stride;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        for (int li = tid; li < cbn; li += nthreads) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;

    int col = blockIdx.x * nthreads + tid;
    if (col >= out_f) return;
    const unsigned char* irow = idx + (long)col * vpr;
    float s = scale[col];

    int grp[VQ_MULTIROW_MAX];
    float acc[VQ_MULTIROW_MAX];
    #pragma unroll
    for (int r = 0; r < VQ_MULTIROW_MAX; ++r) acc[r] = 0.f;
    for (int r = 0; r < R; ++r) grp[r] = pair_g[start + r];

    if (dim == 8) {
        for (int v = 0; v < vpr; ++v) {
            const float* cv = cbp + (long)irow[v] * 8;
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b0 = s * c0.x, b1 = s * c0.y, b2 = s * c0.z, b3 = s * c0.w;
            float b4 = s * c1.x, b5 = s * c1.y, b6 = s * c1.z, b7 = s * c1.w;
            for (int r = 0; r < R; ++r) {
                const float* xb = x + (long)grp[r] * in_f + (long)v * 8;
                float4 x0 = *(const float4*)(xb);
                float4 x1 = *(const float4*)(xb + 4);
                float a = acc[r];
                a += x0.x * b0;
                a += x0.y * b1;
                a += x0.z * b2;
                a += x0.w * b3;
                a += x1.x * b4;
                a += x1.y * b5;
                a += x1.z * b6;
                a += x1.w * b7;
                acc[r] = a;
            }
        }
    } else {
        for (int v = 0; v < vpr; ++v) {
            const float* cv = cbp + (long)irow[v] * dim;
            for (int r = 0; r < R; ++r) {
                const float* xb = x + (long)grp[r] * in_f + (long)v * dim;
                float a = acc[r];
                for (int t = 0; t < dim; ++t) {
                    float b = s * cv[t];
                    a += xb[t] * b;
                }
                acc[r] = a;
            }
        }
    }
    for (int r = 0; r < R; ++r) out[(long)grp[r] * out_f + col] = acc[r];
}

// ---- GROUPED WARP-PER-COLUMN fused VQ GEMV (decode) --------------------------
// One WARP reduces one output column: lane l sums the strided vq-vectors
// v = l, l+32, ... into a partial, then a shfl_down tree combines the 32 lane
// partials. This is NOT arithmetic-order-preserving (the length-in_f dot is
// re-associated at the 32 lane boundaries + the tree), so it is gated on the FNV
// hash: accept only if the token stream stays argmax-identical. Wins by giving
// each column 32 independent in-flight load streams (idx read is coalesced across
// lanes) and 32x the resident threads => hides the gather latency the thread-per-
// column path can't at batch=1. Scale folded per term to stay as close to the
// reference sum as possible (minimizes argmax perturbation).
extern "C" __global__ void vq_fused_matmul_grouped_warp_f32(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ float cb_s[VQ_CB_SMEM_MAX];
    int g = blockIdx.z;
    int e = sel[g];
    const unsigned char* idx = idx_all + (long)e * idx_stride;
    const float*        cb  = cb_all    + (long)e * cb_stride;
    const float*        scale = scale_all + (long)e * scale_stride;
    const float*        xg  = x   + (long)g * x_stride;
    float*              outg = out + (long)g * n * out_f;

    int lane   = threadIdx.x;                 // 0..31
    int warp   = threadIdx.y;                 // 0..nwarps-1
    int nwarps = blockDim.y;
    int cbn = cb_stride;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) cb_s[li] = cb[li];
        __syncthreads();
    }
    const float* cbp = cache ? cb_s : cb;

    int o   = blockIdx.x * nwarps + warp;     // one output column per warp
    int row = blockIdx.y;
    if (o >= out_f || row >= n) return;       // o uniform per warp -> whole-warp exit
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* irow = idx + (long)o * vpr;
    float s    = scale[o];
    float part = 0.f;
    if (dim == 8) {
        for (int v = lane; v < vpr; v += 32) {          // coalesced idx across lanes
            const float* cv = cbp  + (long)irow[v] * 8;
            const float* xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b;
            b = s * c0.x; part += x0.x * b;
            b = s * c0.y; part += x0.y * b;
            b = s * c0.z; part += x0.z * b;
            b = s * c0.w; part += x0.w * b;
            b = s * c1.x; part += x1.x * b;
            b = s * c1.y; part += x1.y * b;
            b = s * c1.z; part += x1.z * b;
            b = s * c1.w; part += x1.w * b;
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cbp  + (long)irow[v] * dim;
            const float* xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) { float b = s * cv[t]; part += xb[t] * b; }
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        part += __shfl_down_sync(0xffffffff, part, off);
    if (lane == 0) outg[(long)row * out_f + o] = part;
}

// ---- fp16-CODEBOOK sibling of vq_fused_matmul_grouped_warp_f32 ---------------
// OPT-IN fast path (env PRG_SUBBIT_F16_CB): the codebook is staged to smem as
// __half (4 KiB vs 8 KiB) and gathered 16 B/vq-vector (vs 32 B); x, scale, and the
// accumulation stay fp32 with the SAME k-order + scale-into-b fold. Only the cb
// operand is fp16-rounded. NOT argmax-identical (flips the greedy FNV) so it is
// never the default — selected only when the caller opts in.
extern "C" __global__ void vq_fused_matmul_grouped_warp_f16cb(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ idx_all,
    const float*         __restrict__ cb_all,
    const float*         __restrict__ scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ out,
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ __half __align__(16) cb_s[VQ_CB_SMEM_MAX];   // fp16 codebook cache (4 KiB)
    int g = blockIdx.z;
    int e = sel[g];
    const unsigned char* idx = idx_all + (long)e * idx_stride;
    const float*        cb  = cb_all    + (long)e * cb_stride;
    const float*        scale = scale_all + (long)e * scale_stride;
    const float*        xg  = x   + (long)g * x_stride;
    float*              outg = out + (long)g * n * out_f;

    int lane   = threadIdx.x;
    int warp   = threadIdx.y;
    int nwarps = blockDim.y;
    int cbn = cb_stride;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) cb_s[li] = __float2half(cb[li]);
        __syncthreads();
    }

    int o   = blockIdx.x * nwarps + warp;
    int row = blockIdx.y;
    if (o >= out_f || row >= n) return;
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* irow = idx + (long)o * vpr;
    float s    = scale[o];
    float part = 0.f;
    if (cache && dim == 8) {
        for (int v = lane; v < vpr; v += 32) {
            const __half* cv = cb_s + (long)irow[v] * 8;
            const float*  xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 cpk = *(const float4*)cv;             // 8 halfs = 16 B
            const __half2* ch = (const __half2*)&cpk;
            float2 c0 = __half22float2(ch[0]);
            float2 c1 = __half22float2(ch[1]);
            float2 c2 = __half22float2(ch[2]);
            float2 c3 = __half22float2(ch[3]);
            float b;
            b = s * c0.x; part += x0.x * b;
            b = s * c0.y; part += x0.y * b;
            b = s * c1.x; part += x0.z * b;
            b = s * c1.y; part += x0.w * b;
            b = s * c2.x; part += x1.x * b;
            b = s * c2.y; part += x1.y * b;
            b = s * c3.x; part += x1.z * b;
            b = s * c3.y; part += x1.w * b;
        }
    } else if (cache) {
        for (int v = lane; v < vpr; v += 32) {
            const __half* cv = cb_s + (long)irow[v] * dim;
            const float*  xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) { float b = s * __half2float(cv[t]); part += xb[t] * b; }
        }
    } else if (dim == 8) {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cb  + (long)irow[v] * 8;
            const float* xb = xrow + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            float4 c0 = *(const float4*)(cv);
            float4 c1 = *(const float4*)(cv + 4);
            float b;
            b = s * c0.x; part += x0.x * b;
            b = s * c0.y; part += x0.y * b;
            b = s * c0.z; part += x0.z * b;
            b = s * c0.w; part += x0.w * b;
            b = s * c1.x; part += x1.x * b;
            b = s * c1.y; part += x1.y * b;
            b = s * c1.z; part += x1.z * b;
            b = s * c1.w; part += x1.w * b;
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* cv = cb  + (long)irow[v] * dim;
            const float* xb = xrow + (long)v * dim;
            for (int t = 0; t < dim; ++t) { float b = s * cv[t]; part += xb[t] * b; }
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        part += __shfl_down_sync(0xffffffff, part, off);
    if (lane == 0) outg[(long)row * out_f + o] = part;
}

// ---- FUSED gate + up + SwiGLU grouped-warp GEMV (decode MoE) -----------------
// One WARP reduces column `o` of BOTH gate and up for group g (same routed expert
// sel[g], same broadcast x row), then applies SwiGLU in registers and writes
// act[g, o] = silu(gate_o) * up_o directly. The [top_k, inter] gate/up
// intermediates never touch DRAM — the separate up GEMV launch and the swiglu
// launch collapse into this one kernel.
//
// BIT-IDENTICAL to (grouped_warp gate) -> (grouped_warp up) -> (swiglu_f32):
//   * pg accumulates in the SAME per-term (scale-into-b) fold, SAME lane-strided
//     k-order, and SAME shfl_down tree as the standalone gate GEMV; the
//     interleaved pu FFMAs use a separate accumulator and never perturb pg.
//   * pu likewise reproduces the standalone up GEMV bit-for-bit.
//   * the tail silu = pg/(1+ex2_expf(-pg)) reproduces swiglu_f32's
//     g/(1+__expf(-g)) exactly (ex2_expf and __expf are the same ex2.approx.f32
//     op on the same 1.4426950408889634*x argument), and silu*pu matches.
// => token stream unchanged, FNV gate holds trivially (order-preserving fusion).
//
// Win: interleaving the gate & up codebook gathers gives each warp TWO
// independent dependent-load->FFMA chains, so the up work fills the gate gather's
// stall bubbles that bottleneck the batch-1 GEMV; the broadcast x row is loaded
// ONCE and fed to both dots. gate & up share shapes/strides (idx/cb/scale).
extern "C" __global__ void vq_fused_gate_up_swiglu_grouped_warp_f32(
    const float*         __restrict__ x,           // broadcast [n, in_f] (x_stride==0)
    const unsigned char* __restrict__ g_idx_all,
    const float*         __restrict__ g_cb_all,
    const float*         __restrict__ g_scale_all,
    const unsigned char* __restrict__ u_idx_all,
    const float*         __restrict__ u_cb_all,
    const float*         __restrict__ u_scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ act,         // [G, n, out_f] = silu(gate)*up
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ float g_cb_s[VQ_CB_SMEM_MAX];       // gate codebook (8 KiB)
    __shared__ float u_cb_s[VQ_CB_SMEM_MAX];       // up   codebook (8 KiB)
    int grp = blockIdx.z;
    int e   = sel[grp];
    const unsigned char* g_idx = g_idx_all + (long)e * idx_stride;
    const float*         g_cb  = g_cb_all  + (long)e * cb_stride;
    const float*         g_sc  = g_scale_all + (long)e * scale_stride;
    const unsigned char* u_idx = u_idx_all + (long)e * idx_stride;
    const float*         u_cb  = u_cb_all  + (long)e * cb_stride;
    const float*         u_sc  = u_scale_all + (long)e * scale_stride;
    const float*         xg    = x + (long)grp * x_stride;   // x_stride==0 => broadcast
    float*               actg  = act + (long)grp * n * out_f;

    int lane   = threadIdx.x;
    int warp   = threadIdx.y;
    int nwarps = blockDim.y;
    int cbn = cb_stride;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) { g_cb_s[li] = g_cb[li]; u_cb_s[li] = u_cb[li]; }
        __syncthreads();
    }
    const float* g_cbp = cache ? g_cb_s : g_cb;
    const float* u_cbp = cache ? u_cb_s : u_cb;

    int o   = blockIdx.x * nwarps + warp;
    int row = blockIdx.y;
    if (o >= out_f || row >= n) return;
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* g_ir = g_idx + (long)o * vpr;
    const unsigned char* u_ir = u_idx + (long)o * vpr;
    float gs = g_sc[o], us = u_sc[o];
    float pg = 0.f, pu = 0.f;
    if (dim == 8) {
        for (int v = lane; v < vpr; v += 32) {          // coalesced idx across lanes
            const float* xb  = xrow  + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            const float* gcv = g_cbp + (long)g_ir[v] * 8;   // two INDEPENDENT gathers
            const float* ucv = u_cbp + (long)u_ir[v] * 8;
            float4 gc0 = *(const float4*)(gcv);
            float4 gc1 = *(const float4*)(gcv + 4);
            float4 uc0 = *(const float4*)(ucv);
            float4 uc1 = *(const float4*)(ucv + 4);
            float b;
            b = gs*gc0.x; pg += x0.x*b;   b = us*uc0.x; pu += x0.x*b;
            b = gs*gc0.y; pg += x0.y*b;   b = us*uc0.y; pu += x0.y*b;
            b = gs*gc0.z; pg += x0.z*b;   b = us*uc0.z; pu += x0.z*b;
            b = gs*gc0.w; pg += x0.w*b;   b = us*uc0.w; pu += x0.w*b;
            b = gs*gc1.x; pg += x1.x*b;   b = us*uc1.x; pu += x1.x*b;
            b = gs*gc1.y; pg += x1.y*b;   b = us*uc1.y; pu += x1.y*b;
            b = gs*gc1.z; pg += x1.z*b;   b = us*uc1.z; pu += x1.z*b;
            b = gs*gc1.w; pg += x1.w*b;   b = us*uc1.w; pu += x1.w*b;
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* xb  = xrow  + (long)v * dim;
            const float* gcv = g_cbp + (long)g_ir[v] * dim;
            const float* ucv = u_cbp + (long)u_ir[v] * dim;
            for (int t = 0; t < dim; ++t) { float xt = xb[t]; pg += xt*(gs*gcv[t]); pu += xt*(us*ucv[t]); }
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        pg += __shfl_down_sync(0xffffffff, pg, off);
        pu += __shfl_down_sync(0xffffffff, pu, off);
    }
    if (lane == 0) {
        float silu = pg / (1.0f + ex2_expf(-pg));
        actg[(long)row * out_f + o] = silu * pu;
    }
}

// ---- fp16-CODEBOOK sibling of the fused gate+up+SwiGLU grouped-warp GEMV ------
// OPT-IN fast path (env PRG_SUBBIT_F16_CB): both the gate and up codebooks are
// staged to smem as __half (2x4 KiB = 8 KiB vs 16 KiB) and gathered 16 B/vector;
// x, scale, accumulation, the SwiGLU tail all stay fp32 with the SAME order as the
// f32 fused kernel. Only the two cb operands are fp16-rounded. NOT argmax-identical.
extern "C" __global__ void vq_fused_gate_up_swiglu_grouped_warp_f16cb(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ g_idx_all,
    const float*         __restrict__ g_cb_all,
    const float*         __restrict__ g_scale_all,
    const unsigned char* __restrict__ u_idx_all,
    const float*         __restrict__ u_cb_all,
    const float*         __restrict__ u_scale_all,
    const int*           __restrict__ sel,
    float*               __restrict__ act,
    int n, int in_f, int out_f, int dim, int vpr,
    int idx_stride, int cb_stride, int scale_stride, int x_stride, int use_smem
) {
    __shared__ __half __align__(16) g_cb_s[VQ_CB_SMEM_MAX];   // gate cb fp16 (4 KiB)
    __shared__ __half __align__(16) u_cb_s[VQ_CB_SMEM_MAX];   // up   cb fp16 (4 KiB)
    int grp = blockIdx.z;
    int e   = sel[grp];
    const unsigned char* g_idx = g_idx_all + (long)e * idx_stride;
    const float*         g_cb  = g_cb_all  + (long)e * cb_stride;
    const float*         g_sc  = g_scale_all + (long)e * scale_stride;
    const unsigned char* u_idx = u_idx_all + (long)e * idx_stride;
    const float*         u_cb  = u_cb_all  + (long)e * cb_stride;
    const float*         u_sc  = u_scale_all + (long)e * scale_stride;
    const float*         xg    = x + (long)grp * x_stride;
    float*               actg  = act + (long)grp * n * out_f;

    int lane   = threadIdx.x;
    int warp   = threadIdx.y;
    int nwarps = blockDim.y;
    int cbn = cb_stride;
    bool cache = (use_smem != 0) && (cbn <= VQ_CB_SMEM_MAX);
    if (cache) {
        int tpb = blockDim.x * nwarps;
        int lid = warp * blockDim.x + lane;
        for (int li = lid; li < cbn; li += tpb) { g_cb_s[li] = __float2half(g_cb[li]); u_cb_s[li] = __float2half(u_cb[li]); }
        __syncthreads();
    }

    int o   = blockIdx.x * nwarps + warp;
    int row = blockIdx.y;
    if (o >= out_f || row >= n) return;
    const float* xrow = xg + (long)row * in_f;
    const unsigned char* g_ir = g_idx + (long)o * vpr;
    const unsigned char* u_ir = u_idx + (long)o * vpr;
    float gs = g_sc[o], us = u_sc[o];
    float pg = 0.f, pu = 0.f;
    if (cache && dim == 8) {
        for (int v = lane; v < vpr; v += 32) {
            const float* xb  = xrow  + (long)v * 8;
            float4 x0 = *(const float4*)(xb);
            float4 x1 = *(const float4*)(xb + 4);
            const __half* gcv = g_cb_s + (long)g_ir[v] * 8;
            const __half* ucv = u_cb_s + (long)u_ir[v] * 8;
            float4 gpk = *(const float4*)gcv;
            float4 upk = *(const float4*)ucv;
            const __half2* gh = (const __half2*)&gpk;
            const __half2* uh = (const __half2*)&upk;
            float2 gc0 = __half22float2(gh[0]), gc1 = __half22float2(gh[1]);
            float2 gc2 = __half22float2(gh[2]), gc3 = __half22float2(gh[3]);
            float2 uc0 = __half22float2(uh[0]), uc1 = __half22float2(uh[1]);
            float2 uc2 = __half22float2(uh[2]), uc3 = __half22float2(uh[3]);
            float b;
            b = gs*gc0.x; pg += x0.x*b;   b = us*uc0.x; pu += x0.x*b;
            b = gs*gc0.y; pg += x0.y*b;   b = us*uc0.y; pu += x0.y*b;
            b = gs*gc1.x; pg += x0.z*b;   b = us*uc1.x; pu += x0.z*b;
            b = gs*gc1.y; pg += x0.w*b;   b = us*uc1.y; pu += x0.w*b;
            b = gs*gc2.x; pg += x1.x*b;   b = us*uc2.x; pu += x1.x*b;
            b = gs*gc2.y; pg += x1.y*b;   b = us*uc2.y; pu += x1.y*b;
            b = gs*gc3.x; pg += x1.z*b;   b = us*uc3.x; pu += x1.z*b;
            b = gs*gc3.y; pg += x1.w*b;   b = us*uc3.y; pu += x1.w*b;
        }
    } else if (cache) {
        for (int v = lane; v < vpr; v += 32) {
            const float* xb  = xrow  + (long)v * dim;
            const __half* gcv = g_cb_s + (long)g_ir[v] * dim;
            const __half* ucv = u_cb_s + (long)u_ir[v] * dim;
            for (int t = 0; t < dim; ++t) { float xt = xb[t]; pg += xt*(gs*__half2float(gcv[t])); pu += xt*(us*__half2float(ucv[t])); }
        }
    } else {
        for (int v = lane; v < vpr; v += 32) {
            const float* xb  = xrow  + (long)v * dim;
            const float* gcv = g_cb + (long)g_ir[v] * dim;
            const float* ucv = u_cb + (long)u_ir[v] * dim;
            for (int t = 0; t < dim; ++t) { float xt = xb[t]; pg += xt*(gs*gcv[t]); pu += xt*(us*ucv[t]); }
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        pg += __shfl_down_sync(0xffffffff, pg, off);
        pu += __shfl_down_sync(0xffffffff, pu, off);
    }
    if (lane == 0) {
        float silu = pg / (1.0f + ex2_expf(-pg));
        actg[(long)row * out_f + o] = silu * pu;
    }
}

// ---- ROUTER GEMV (dense fp32, row-major weight) -----------------------------
// out[i,c] = sum_k x[i,k] * w[k*N + c]  for a small [n,K]x[K,N] matmul (the MoE
// router projection normed[1,K]·router_t[K,N], N=num_experts). Replaces the
// cuBLAS `Tensor::matmul` on the decode (n==1) path so the MoE layer is entirely
// cuBLAS-free and the whole decode step becomes CUDA-graph-capturable. One block
// per row; each thread owns output column(s) c and accumulates over k in
// ASCENDING order into one f32 acc (matches the fp32 cuBLAS reduction closely —
// FNV-gated). For a fixed k the threads read w[k*N + c] for consecutive c, so the
// weight load is fully coalesced; x[i,k] is a broadcast load. Kept deliberately
// simple (1D block, no shared mem): a 2D/1024-thread variant broke CUDA-graph
// capture on this box and — since grid is one block per row (n==1 → one SM) — did
// not actually speed up the skinny decode GEMV. The single-SM launch latency this
// costs in EAGER decode is fully erased once the step is replayed from a graph.
extern "C" __global__ void vq_router_gemv_f32(
    const float* __restrict__ x,   // [n, K]
    const float* __restrict__ w,   // [K, N] row-major
    float*       __restrict__ out, // [n, N]
    int n, int K, int N
) {
    int row = blockIdx.x;
    if (row >= n) return;
    const float* xrow = x + (long)row * K;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        float acc = 0.f;
        for (int kk = 0; kk < K; ++kk) acc += xrow[kk] * w[(long)kk * N + c];
        out[(long)row * N + c] = acc;
    }
}

// ---- FUSED DECODE ATTENTION (single query position, fixed-KV) ---------------
// One block per query head h computes the whole per-head SDPA over the fixed
// [nkv, max_ctx, hd] KV buffers with the additive device mask — q·Kᵀ (scaled) +
// mask, softmax, ·V — in ONE launch, with NO cuBLAS and NO repeat_kv
// materialization (GQA is resolved in-kernel: query head h reads kv head
// h/(nh/nkv)). This is the decode-path replacement for
//   scores = q.matmul(kᵀ); attn = (scores*scale + mask).softmax; out = attn.matmul(v)
// so the fixed-KV decode step is cuBLAS-free and CUDA-graph-capturable. It is not
// bit-identical to the cuBLAS+softmax reference (fp reassociation + ex2.approx exp,
// the same approx the router uses) but matches the masked-softmax math and is
// FNV-gated: accept only if the token stream stays argmax-identical.
//
// q:    [nh, hd]            (contiguous, head h at h*hd)
// k,v:  [nkv, max_ctx, hd]  (the fixed KV buffers; unwritten positions are zero)
// mask: [max_ctx]           (additive: 0 for valid, ~-1e30 for future positions)
// out:  [nh, hd]            (== [1,1,nh*hd] for the o_proj)
// Dynamic shared mem holds the [max_ctx] score row followed by the [hd] q row.
extern "C" __global__ void vq_decode_attn_f32(
    const float* __restrict__ q,     // [nh, hd]
    const float* __restrict__ k,     // [nkv, max_ctx, hd]
    const float* __restrict__ v,     // [nkv, max_ctx, hd]
    const float* __restrict__ mask,  // [max_ctx]
    float*       __restrict__ out,   // [nh, hd]
    int nh, int nkv, int max_ctx, int hd, float scale
) {
    extern __shared__ float sm[];      // [max_ctx] scores, then [hd] q cache
    float* sc = sm;                    // scores / probs
    float* qs = sm + max_ctx;          // q row cache
    __shared__ float warp_red[32];

    int h = blockIdx.x;
    if (h >= nh) return;
    int rep = nh / nkv;                // GQA repeat factor
    int kvh = h / rep;
    const float* qh = q + (long)h * hd;
    const float* kh = k + (long)kvh * max_ctx * hd;
    const float* vh = v + (long)kvh * max_ctx * hd;
    int tid = threadIdx.x, nt = blockDim.x;
    int lane = tid & 31, warp = tid >> 5, nwarps = nt >> 5;

    for (int d = tid; d < hd; d += nt) qs[d] = qh[d];
    __syncthreads();

    // Pass 1: scores s_j = scale*(q·k_j) + mask_j, and per-thread max.
    float lmax = -INFINITY;
    for (int j = tid; j < max_ctx; j += nt) {
        const float* kj = kh + (long)j * hd;
        float acc = 0.f;
        for (int d = 0; d < hd; ++d) acc += qs[d] * kj[d];
        float s = acc * scale + mask[j];
        sc[j] = s;
        if (s > lmax) lmax = s;
    }
    // block-reduce max -> warp_red[0]
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        float o = __shfl_down_sync(0xffffffff, lmax, off);
        if (o > lmax) lmax = o;
    }
    if (lane == 0) warp_red[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < nwarps) ? warp_red[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            float o = __shfl_down_sync(0xffffffff, m, off);
            if (o > m) m = o;
        }
        if (lane == 0) warp_red[0] = m;
    }
    __syncthreads();
    float gmax = warp_red[0];
    __syncthreads();  // all threads have read gmax before warp_red is reused

    // Pass 2: exp(s_j - gmax) (ex2.approx, no libdevice) and per-thread sum.
    float lsum = 0.f;
    for (int j = tid; j < max_ctx; j += nt) {
        float e = ex2_expf(sc[j] - gmax);
        sc[j] = e;
        lsum += e;
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) lsum += __shfl_down_sync(0xffffffff, lsum, off);
    if (lane == 0) warp_red[warp] = lsum;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < nwarps) ? warp_red[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) warp_red[0] = s;
    }
    __syncthreads();
    float inv = 1.f / warp_red[0];

    // Pass 3: out[h,d] = inv * sum_j prob_j * v[j,d]. For fixed j the threads read
    // v[j*hd + d] for consecutive d, so the value load is coalesced.
    for (int d = tid; d < hd; d += nt) {
        float acc = 0.f;
        for (int j = 0; j < max_ctx; ++j) acc += sc[j] * vh[(long)j * hd + d];
        out[(long)h * hd + d] = acc * inv;
    }
}

// SFU approx intrinsics via inline PTX (no libdevice; matches transformer_ops.ptx
// rope which lowers powf→lg2.approx/ex2.approx and sincosf→sin.approx/cos.approx).
// The reference uses the `.ftz` variants; for the normal-range operands here
// flush-to-zero is a no-op, so these produce the identical bit pattern.
__device__ __forceinline__ float lg2_approx(float x) {
    float r; asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float ex2_approx(float x) {
    float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float sin_approx(float x) {
    float r; asm("sin.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float cos_approx(float x) {
    float r; asm("cos.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}

// ---- DEVICE-POS split-halves RoPE (CUDA-graph decode replay) -----------------
// Bit-identical to rope_split_halves_bhsd_f32 (transformer_ops.cu) EXCEPT the
// starting position is read from a device i32 scalar (`pos_ptr[0]`) instead of a
// launch-arg constant. That is the whole point: one captured graph can replay at
// every decode step because RoPE reads `pos` from a fixed device address the
// eager prep updates per token, rather than baking `pos` into the launch record.
// The float math `(float)pos * powf(theta, exponent)` is identical whether `pos`
// arrives as a param or a memory load, so the rotated Q/K are byte-for-byte the
// same as the param-pos kernel — the argmax stream (and its FNV hash) is preserved.
// Input/output shape [bs, n_heads, seq, head_dim] row-major (head-major, Qwen3).
// Launch: grid = (seq, n_heads, bs), block = (head_dim/2, 1, 1), shmem = 0.
extern "C" __global__ void rope_split_halves_bhsd_devpos_f32(
    const float* __restrict__ src,
    float* __restrict__ out,
    const float* __restrict__ pos_ptr, // device scalar: starting position (pos),
                                       // stored as f32 (exact for pos < 2^24)
    unsigned int seq,
    unsigned int n_heads,
    unsigned int head_dim,
    float theta
) {
    const unsigned int t = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int pair = threadIdx.x;
    const unsigned int half = head_dim >> 1;
    if (pair >= half) return;

    const unsigned int pos = (unsigned int)((int)pos_ptr[0]) + t;
    const float exponent = -(float)(2u * pair) / (float)head_dim;
    // powf(theta, exponent) via the SFU approx path (ex2(exponent*lg2(theta))) —
    // matching transformer_ops.ptx's rope (lg2.approx/ex2.approx/sin.approx/
    // cos.approx), NOT libdevice powf/sincosf, so this kernel needs no libdevice
    // (built with -nocudalib) and stays bit-identical to the eager reference RoPE.
    const float angle = (float)pos * ex2_approx(exponent * lg2_approx(theta));
    const float s = sin_approx(angle);
    const float c = cos_approx(angle);

    const size_t base = ((size_t)b * (size_t)n_heads * (size_t)seq
                        + (size_t)h * (size_t)seq
                        + (size_t)t) * (size_t)head_dim
                      + (size_t)pair;
    const float a = src[base];
    const float bv = src[base + half];
    out[base]        = c * a  - s * bv;
    out[base + half] = s * a  + c * bv;
}

// ---- DEVICE-POS KV scatter (CUDA-graph decode replay) -----------------------
// Writes this step's K (or V) into the fixed [nkv, max_ctx, hd] cache buffer at
// sequence position `pos_ptr[0]` (a device i32 scalar). This is the KERNEL twin
// of the memcpy_2d_dtod scatter (`scatter_heads_at`): a captured memcpy node
// bakes its destination offset (pos*hd) into the launch record and would write
// the wrong position on replay, whereas this kernel recomputes the destination
// from the device `pos` every launch — so one captured graph advances the KV
// correctly across decode steps. Pure copy, so it is byte-identical to the
// memcpy scatter (same source bytes → same destination addresses).
// src: [nkv, t, hd] contiguous (this token's t positions × hd per head).
// dst: [nkv, max_ctx, hd]. head h's row starts at h*max_ctx*hd; position pos at
//      + pos*hd. Launch: 1D grid-stride over nkv*t*hd elements.
extern "C" __global__ void kv_scatter_devpos_f32(
    const float* __restrict__ src,     // [nkv, t, hd]
    float*       __restrict__ dst,     // [nkv, max_ctx, hd]
    const float* __restrict__ pos_ptr, // device scalar: destination position (f32)
    int nkv, int t, int hd, int max_ctx
) {
    const int pos = (int)pos_ptr[0];
    const long width = (long)t * hd;          // values per head (contiguous)
    const long total = (long)nkv * width;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int head = (int)(i / width);
        const long off = i - (long)head * width;    // 0 .. t*hd-1
        const long dst_i = (long)head * max_ctx * hd + (long)pos * hd + off;
        dst[dst_i] = src[i];
    }
}

// ── NemotronH (Mamba2) fused decode-step kernels ──
// Single-token decode: the depthwise causal conv (CIRCULAR ring, no shifting),
// the per-head SSD recurrence, and the gated group-RMSNorm, all device-side so
// the hidden never round-trips the host between the in/out projections.
// exp/log1p via ex2/lg2.approx (no libdevice, JITs under -nocudalib).

__device__ __forceinline__ float lg2_log1pf(float x) {
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(1.0f + x));
    return r * 0.6931471805599453f;
}

extern "C" __global__ void nh_conv_silu_f32(
    const float* __restrict__ zxbcdt,   // [d_inner + conv_dim + heads]
    float*       __restrict__ ring,     // [K * conv_dim] circular, slot `pos` written
    const float* __restrict__ conv_w,   // [conv_dim * K] (HF [c][1][k], taps oldest→newest)
    const float* __restrict__ conv_b,   // [conv_dim]
    float*       __restrict__ xbc,      // out [conv_dim] = silu(conv)
    int d_inner, int conv_dim, int K, int pos
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    ring[(long)pos * conv_dim + c] = zxbcdt[d_inner + c];
    float acc = conv_b[c];
    for (int t = 0; t < K; ++t) {
        int slot = pos + 1 + t; if (slot >= K) slot -= K; if (slot >= K) slot -= K;
        acc += ring[(long)slot * conv_dim + c] * conv_w[c * K + t];
    }
    xbc[c] = acc / (1.0f + ex2_expf(-acc));
}

extern "C" __global__ void nh_ssd_scan_f32(
    const float* __restrict__ zxbcdt,   // dt tail at [d_inner + conv_dim + h]
    const float* __restrict__ xbc,      // [conv_dim] = [x(d_inner) | B(ng·ns) | C(ng·ns)]
    float*       __restrict__ state,    // [heads * hd * ns], updated in place
    const float* __restrict__ a_log,    // [heads]
    const float* __restrict__ dvec,     // [heads]
    const float* __restrict__ dt_bias,  // [heads]
    float*       __restrict__ y,        // out [d_inner]
    int d_inner, int conv_dim, int heads, int hd, int ns, int ngroups, float dt_min
) {
    int h = blockIdx.x;
    int di = threadIdx.x;
    if (h >= heads || di >= hd) return;
    float dtr = zxbcdt[d_inner + conv_dim + h] + dt_bias[h];
    float dt = (dtr > 20.0f) ? dtr : lg2_log1pf(ex2_expf(dtr));
    if (dt < dt_min) dt = dt_min;
    float da = ex2_expf(dt * -ex2_expf(a_log[h]));
    int g = h / (heads / ngroups);
    const float* Bg = xbc + d_inner + g * ns;
    const float* Cg = xbc + d_inner + ngroups * ns + g * ns;
    float xv = xbc[h * hd + di];
    float dtx = dt * xv;
    float* srow = state + ((long)(h * hd + di)) * ns;
    float acc = 0.0f;
    for (int s = 0; s < ns; ++s) {
        float sv = srow[s] * da + dtx * Bg[s];
        srow[s] = sv;
        acc += sv * Cg[s];
    }
    y[h * hd + di] = acc + dvec[h] * xv;
}

extern "C" __global__ void nh_gated_gnorm_f32(
    const float* __restrict__ y,        // [d_inner] scan output
    const float* __restrict__ zxbcdt,   // gate at [0 .. d_inner)
    const float* __restrict__ nw,       // [d_inner] group-norm weight
    float*       __restrict__ out,      // [d_inner]
    int d_inner, int ngroups, float eps
) {
    int grp = blockIdx.x;
    int gs = d_inner / ngroups;
    __shared__ float red[256];
    float local = 0.0f;
    for (int j = threadIdx.x; j < gs; j += blockDim.x) {
        int i = grp * gs + j;
        float gate = zxbcdt[i];
        float v = y[i] * (gate / (1.0f + ex2_expf(-gate)));
        out[i] = v;
        local += v * v;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float ms = red[0] / (float)gs;
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(ms + eps));
    for (int j = threadIdx.x; j < gs; j += blockDim.x) {
        int i = grp * gs + j;
        out[i] = out[i] * r * nw[i];
    }
}

// NemotronH sigmoid router: score = sigmoid(logit); top-k CHOSEN by
// score + e_score_correction_bias, WEIGHTED by score (renormed to sum 1 when
// norm!=0, then × routed_scaling). One thread per row (ne<=256), same shape
// contract as vq_router_topk_f32 — device sel/weights so the MoE layer keeps
// zero host syncs.
extern "C" __global__ void nh_router_topk_f32(
    const float* __restrict__ logits,  // [rows, ne]
    const float* __restrict__ bias,    // [ne] correction bias (choice only)
    int*         __restrict__ sel,     // [rows, top_k] (out)
    float*       __restrict__ wout,    // [rows, top_k] (out)
    int ne, int top_k, int norm, float scaling
) {
    int row = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float* L = logits + (long)row * ne;
    int*   S = sel  + (long)row * top_k;
    float* W = wout + (long)row * top_k;

    float score[256];
    float choice[256];
    bool  chosen[256];
    for (int i = 0; i < ne; ++i) {
        float s = 1.0f / (1.0f + ex2_expf(-L[i]));
        score[i] = s;
        choice[i] = s + bias[i];
        chosen[i] = false;
    }
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int best = -1; float bestv = -INFINITY;
        for (int i = 0; i < ne; ++i) {
            if (!chosen[i] && choice[i] > bestv) { bestv = choice[i]; best = i; }
        }
        // A NaN row loses every comparison and would leave best = -1, which
        // indexes the expert weights out of bounds. Fall back to the first
        // unchosen expert so a poisoned row degrades numerically, not fatally.
        if (best < 0) {
            for (int i = 0; i < ne; ++i) { if (!chosen[i]) { best = i; break; } }
            if (best < 0) best = 0;
        }
        chosen[best] = true;
        S[k] = best;
        W[k] = score[best];
        wsum += score[best];
    }
    if (norm) {
        float inv = 1.0f / (wsum + 1e-20f);
        for (int k = 0; k < top_k; ++k) W[k] *= inv;
    }
    for (int k = 0; k < top_k; ++k) W[k] *= scaling;
}

// Own dense f32 GEMV (decode n=1): y[o] = Σ_k x[k]·W[o,k], W row-major [N,K] so
// each warp STREAMS its output row (a streaming GEMV shape, ~416 GB/s class).
// No cuBLAS → capture-safe; the captured-decode replacement for the dense-mirror
// matmuls. 4 warps/block, one output row per warp, warp-shuffle reduce.
extern "C" __global__ void nh_dense_gemv_f32(
    const float* __restrict__ x,   // [K]
    const float* __restrict__ w,   // [N, K] row-major
    float*       __restrict__ y,   // [N]
    int n, int k
) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n) return;
    const float* row = w + (long)warp * k;
    float acc = 0.0f;
    for (int i = lane; i < k; i += 32) acc += x[i] * row[i];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[warp] = acc;
}

// Device-pos twin of nh_conv_silu_f32: ring slot read from the f32 scalar
// `pos[0]` so a captured graph advances the conv window across replays.
extern "C" __global__ void nh_conv_silu_devpos_f32(
    const float* __restrict__ zxbcdt,
    float*       __restrict__ ring,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_b,
    const float* __restrict__ pos,     // [1] f32 scalar, integer value mod K
    float*       __restrict__ xbc,
    int d_inner, int conv_dim, int K
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int p = ((int)pos[0]) % K;
    ring[(long)p * conv_dim + c] = zxbcdt[d_inner + c];
    float acc = conv_b[c];
    for (int t = 0; t < K; ++t) {
        int slot = p + 1 + t; if (slot >= K) slot -= K; if (slot >= K) slot -= K;
        acc += ring[(long)slot * conv_dim + c] * conv_w[c * K + t];
    }
    xbc[c] = acc / (1.0f + ex2_expf(-acc));
}

// In-graph position advance: pos[0] += 1 (single thread). Lets one captured
// token-graph replay repeatedly with the conv ring/KV walking forward.
extern "C" __global__ void nh_pos_incr_f32(float* __restrict__ pos) {
    if (blockIdx.x == 0 && threadIdx.x == 0) pos[0] += 1.0f;
}

// ── NemotronH BATCHED decode (M agents/sequences per step) ──
// Multi-agent serving: M sequences advance one token each in ONE launch, sharing
// the weight read. State is SLOTTED — one contiguous per-layer buffer indexed by
// slot b (ring[b][K][conv_dim], state[b][heads][hd][ns]) — so a single kernel can
// address every agent's recurrent state without a pointer array. Per (b, ...) the
// arithmetic is IDENTICAL to the M=1 kernels above.

extern "C" __global__ void nh_conv_silu_batch_f32(
    const float* __restrict__ zxbcdt,   // [M, d_inner + conv_dim + heads]
    float*       __restrict__ ring,     // [M, K, conv_dim] circular
    const float* __restrict__ conv_w,   // [conv_dim, K]
    const float* __restrict__ conv_b,   // [conv_dim]
    float*       __restrict__ xbc,      // out [M, conv_dim]
    const float* __restrict__ active,   // [M] 0 = slot parked, leave state alone
    const float* __restrict__ ring_pos, // [M] per-slot ring index: a parked slot
                                        //  does not advance, so it cannot share
                                        //  a global step counter
    int d_inner, int conv_dim, int K, int m_total, int zx_stride
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (c >= conv_dim || b >= m_total) return;
    if (active[b] == 0.f) return;
    const float* z = zxbcdt + (long)b * zx_stride;
    float* r = ring + (long)b * K * conv_dim;
    int p = ((int)ring_pos[b]) % K;
    r[(long)p * conv_dim + c] = z[d_inner + c];
    float acc = conv_b[c];
    for (int t = 0; t < K; ++t) {
        int slot = p + 1 + t; if (slot >= K) slot -= K; if (slot >= K) slot -= K;
        acc += r[(long)slot * conv_dim + c] * conv_w[c * K + t];
    }
    xbc[(long)b * conv_dim + c] = acc / (1.0f + ex2_expf(-acc));
}

extern "C" __global__ void nh_ssd_scan_batch_f32(
    const float* __restrict__ zxbcdt,   // [M, zx_stride]
    const float* __restrict__ xbc,      // [M, conv_dim]
    float*       __restrict__ state,    // [M, heads*hd*ns]
    const float* __restrict__ a_log,
    const float* __restrict__ dvec,
    const float* __restrict__ dt_bias,
    float*       __restrict__ y,        // out [M, d_inner]
    const float* __restrict__ active,   // [M] 0 = slot parked, leave state alone
    int d_inner, int conv_dim, int heads, int hd, int ns, int ngroups,
    float dt_min, int m_total, int zx_stride
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int di = threadIdx.x;
    if (h >= heads || di >= hd || b >= m_total) return;
    if (active[b] == 0.f) return;
    const float* z = zxbcdt + (long)b * zx_stride;
    const float* xb = xbc + (long)b * conv_dim;
    float* st = state + (long)b * ((long)heads * hd * ns);
    float dtr = z[d_inner + conv_dim + h] + dt_bias[h];
    float dt = (dtr > 20.0f) ? dtr : lg2_log1pf(ex2_expf(dtr));
    if (dt < dt_min) dt = dt_min;
    float da = ex2_expf(dt * -ex2_expf(a_log[h]));
    int g = h / (heads / ngroups);
    const float* Bg = xb + d_inner + g * ns;
    const float* Cg = xb + d_inner + ngroups * ns + g * ns;
    float xv = xb[h * hd + di];
    float dtx = dt * xv;
    // State is laid out [head][state][dim], NOT [head][dim][state]: at a given s
    // the threads of a warp (consecutive di) then touch consecutive addresses and
    // coalesce into full 128 B transactions. The natural per-thread-row layout
    // puts neighbouring threads ns*4 = 512 B apart, which turns every warp access
    // into 32 separate transactions using 4 of every 32 bytes — 12.5% of the
    // bandwidth, on the largest single data movement in the model (2.1 MB of
    // state per slot, read AND written every step).
    float* sbase = st + (long)h * ns * hd + di;
    float acc = 0.0f;
    for (int s = 0; s < ns; ++s) {
        float* cell = sbase + (long)s * hd;
        float sv = *cell * da + dtx * Bg[s];
        *cell = sv;
        acc += sv * Cg[s];
    }
    y[(long)b * d_inner + h * hd + di] = acc + dvec[h] * xv;
}

// ---- BATCHED SSD SCAN, float4 state (decode) --------------------------------
// The scan moves more bytes than anything else in the model: 2.1 MB of state per
// slot, read AND written every step, 23 mamba layers deep. With the [h][s][di]
// layout a warp's accesses already coalesce, but each thread still issues 128
// separate 4-byte load/store pairs for its one di.
//
// Here each thread owns FOUR consecutive di and moves them as a float4, so the
// same bytes cost a quarter of the memory instructions. Arithmetic per element
// is untouched (same order, same accumulator), so results are bit-identical.
// Block is (hd/4, 4): sixteen threads cover one head's 64 dims, four heads deep.
extern "C" __global__ void nh_ssd_scan_batch_v4_f32(
    const float* __restrict__ zxbcdt,
    const float* __restrict__ xbc,
    float*       __restrict__ state,
    const float* __restrict__ a_log,
    const float* __restrict__ dvec,
    const float* __restrict__ dt_bias,
    float*       __restrict__ y,
    const float* __restrict__ active,
    int d_inner, int conv_dim, int heads, int hd, int ns, int ngroups,
    float dt_min, int m_total, int zx_stride
) {
    int h = blockIdx.x * blockDim.y + threadIdx.y;
    int b = blockIdx.y;
    int d4 = threadIdx.x * 4;
    if (h >= heads || d4 >= hd || b >= m_total) return;
    if (active[b] == 0.f) return;
    const float* z = zxbcdt + (long)b * zx_stride;
    const float* xb = xbc + (long)b * conv_dim;
    float* st = state + (long)b * ((long)heads * hd * ns);
    float dtr = z[d_inner + conv_dim + h] + dt_bias[h];
    float dt = (dtr > 20.0f) ? dtr : lg2_log1pf(ex2_expf(dtr));
    if (dt < dt_min) dt = dt_min;
    float da = ex2_expf(dt * -ex2_expf(a_log[h]));
    int g = h / (heads / ngroups);
    const float* Bg = xb + d_inner + g * ns;
    const float* Cg = xb + d_inner + ngroups * ns + g * ns;
    float4 xv = *(const float4*)(xb + h * hd + d4);
    float4 dtx = make_float4(dt * xv.x, dt * xv.y, dt * xv.z, dt * xv.w);
    float* sbase = st + (long)h * ns * hd + d4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < ns; ++s) {
        float4* cell = (float4*)(sbase + (long)s * hd);
        float4 sv = *cell;
        float bs = Bg[s], cs = Cg[s];
        sv.x = sv.x * da + dtx.x * bs;
        sv.y = sv.y * da + dtx.y * bs;
        sv.z = sv.z * da + dtx.z * bs;
        sv.w = sv.w * da + dtx.w * bs;
        *cell = sv;
        acc.x += sv.x * cs;
        acc.y += sv.y * cs;
        acc.z += sv.z * cs;
        acc.w += sv.w * cs;
    }
    float dv = dvec[h];
    float* yb = y + (long)b * d_inner + h * hd + d4;
    *(float4*)yb = make_float4(acc.x + dv * xv.x, acc.y + dv * xv.y,
                               acc.z + dv * xv.z, acc.w + dv * xv.w);
}

extern "C" __global__ void nh_gated_gnorm_batch_f32(
    const float* __restrict__ y,        // [M, d_inner]
    const float* __restrict__ zxbcdt,   // [M, zx_stride] (gate at [0..d_inner))
    const float* __restrict__ nw,       // [d_inner]
    float*       __restrict__ out,      // [M, d_inner]
    const float* __restrict__ active,   // [M] 0 = slot parked
    int d_inner, int ngroups, float eps, int m_total, int zx_stride
) {
    int grp = blockIdx.x;
    int b = blockIdx.y;
    if (b >= m_total) return;
    int gs = d_inner / ngroups;
    // A parked slot skipped the conv/scan kernels, so its scratch holds stale
    // values. Emit zeros instead of letting them reach the router.
    if (active[b] == 0.f) {
        float* oz = out + (long)b * d_inner;
        for (int j = threadIdx.x; j < gs; j += blockDim.x) oz[grp * gs + j] = 0.f;
        return;
    }
    const float* z = zxbcdt + (long)b * zx_stride;
    const float* yb = y + (long)b * d_inner;
    float* ob = out + (long)b * d_inner;
    __shared__ float red[256];
    float local = 0.0f;
    for (int j = threadIdx.x; j < gs; j += blockDim.x) {
        int i = grp * gs + j;
        float gate = z[i];
        float v = yb[i] * (gate / (1.0f + ex2_expf(-gate)));
        ob[i] = v;
        local += v * v;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float ms = red[0] / (float)gs;
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(ms + eps));
    for (int j = threadIdx.x; j < gs; j += blockDim.x) {
        int i = grp * gs + j;
        ob[i] = ob[i] * r * nw[i];
    }
}

// Row-wise RMSNorm over [M, n]: one block per row (the generic rms_norm asserts
// weight-length == total length, which only holds at M=1).
extern "C" __global__ void nh_rmsnorm_rows_f32(
    const float* __restrict__ x,   // [M, n]
    const float* __restrict__ w,   // [n]
    float*       __restrict__ out, // [M, n]
    int n, float eps, int m_total
) {
    int b = blockIdx.x;
    if (b >= m_total) return;
    const float* xr = x + (long)b * n;
    float* orow = out + (long)b * n;
    __shared__ float red[256];
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += xr[i] * xr[i];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float ms = red[0] / (float)n;
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(ms + eps));
    for (int i = threadIdx.x; i < n; i += blockDim.x) orow[i] = xr[i] * r * w[i];
}

// Row repeat for batched MoE: out[g] = x[g / rep]. The grouped expert kernel
// indexes its input PER GROUP, but with M agents each selecting top_k experts the
// group→row map is g/top_k, so the agent rows are materialized once per layer
// (M*top_k*in_f floats, ~0.5 MB at M=8) instead of changing the shared kernel's
// signature (Qwen3 uses it too).
extern "C" __global__ void nh_row_repeat_f32(
    const float* __restrict__ x,   // [M, n]
    float*       __restrict__ out, // [M*rep, n]
    int n, int rep, int total_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (i >= n || g >= total_rows) return;
    out[(long)g * n + i] = x[(long)(g / rep) * n + i];
}

// Batched MoE combine: rows [M*top_k, h] weighted by w[M*top_k] and summed within
// each agent's top_k block → [M, h]. Same g-order accumulation as the M=1
// vq_moe_combine, so a single-agent batch is byte-identical.
extern "C" __global__ void nh_moe_combine_batch_f32(
    const float* __restrict__ rows, // [M*top_k, h]
    const float* __restrict__ w,    // [M*top_k]
    float*       __restrict__ out,  // [M, h]
    int h, int top_k, int m_total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (i >= h || b >= m_total) return;
    float acc = 0.0f;
    for (int g = 0; g < top_k; ++g) {
        int r = b * top_k + g;
        acc += rows[(long)r * h + i] * w[r];
    }
    out[(long)b * h + i] = acc;
}

// Batched decode attention for M agents: grid.y = slot, each slot addressing its
// OWN KV in one slotted buffer [M, nkv, max_ctx, hd]. Per (slot, head) the math is
// identical to vq_decode_attn_f32 — same 3-pass max/exp/weighted-sum, same
// ex2.approx — so M=1 is byte-identical to the single-agent kernel. The mask is
// shared because all slots step in lockstep.
extern "C" __global__ void nh_decode_attn_batch_f32(
    const float* __restrict__ q,     // [M, nh, hd]
    const float* __restrict__ k,     // [M, nkv, max_ctx, hd]
    const float* __restrict__ v,     // [M, nkv, max_ctx, hd]
    const float* __restrict__ mask,  // [M, max_ctx] per-slot: agents join at
                                     //  different times, so each slot masks its
                                     //  own history and never sees a prior
                                     //  occupant's KV
    const float* __restrict__ pos,   // [M] per-slot current position
    float*       __restrict__ out,   // [M, nh, hd]
    int nh, int nkv, int max_ctx, int hd, float scale, int m_total
) {
    extern __shared__ float sm[];
    float* sc = sm;
    float* qs = sm + max_ctx;
    __shared__ float warp_red[32];

    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= nh || b >= m_total) return;
    int rep = nh / nkv;
    int kvh = h / rep;
    const float* qh = q + (long)b * nh * hd + (long)h * hd;
    const float* kh = k + (long)b * nkv * max_ctx * hd + (long)kvh * max_ctx * hd;
    const float* vh = v + (long)b * nkv * max_ctx * hd + (long)kvh * max_ctx * hd;
    float* ob = out + (long)b * nh * hd;
    const float* mb = mask + (long)b * max_ctx;
    // Only this slot's OWN history is live; the rest of the window is masked to
    // -inf and contributes exactly zero to both the max and the softmax sum, so
    // stopping at pos+1 is bit-identical and skips the dead tail. At a 1024
    // window with 30 tokens generated that is 30x less attention work.
    int lim = (int)pos[b] + 1;
    if (lim > max_ctx) lim = max_ctx;
    if (lim < 1) lim = 1;
    int tid = threadIdx.x, nt = blockDim.x;
    int lane = tid & 31, warp = tid >> 5, nwarps = nt >> 5;

    for (int d = tid; d < hd; d += nt) qs[d] = qh[d];
    __syncthreads();

    float lmax = -INFINITY;
    for (int j = tid; j < lim; j += nt) {
        const float* kj = kh + (long)j * hd;
        float acc = 0.f;
        for (int d = 0; d < hd; ++d) acc += qs[d] * kj[d];
        float s = acc * scale + mb[j];
        sc[j] = s;
        if (s > lmax) lmax = s;
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        float o = __shfl_down_sync(0xffffffff, lmax, off);
        if (o > lmax) lmax = o;
    }
    if (lane == 0) warp_red[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float mx = (lane < nwarps) ? warp_red[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            float o = __shfl_down_sync(0xffffffff, mx, off);
            if (o > mx) mx = o;
        }
        if (lane == 0) warp_red[0] = mx;
    }
    __syncthreads();
    float gmax = warp_red[0];
    __syncthreads();

    float lsum = 0.f;
    for (int j = tid; j < lim; j += nt) {
        float e = ex2_expf(sc[j] - gmax);
        sc[j] = e;
        lsum += e;
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) lsum += __shfl_down_sync(0xffffffff, lsum, off);
    if (lane == 0) warp_red[warp] = lsum;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < nwarps) ? warp_red[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) warp_red[0] = s;
    }
    __syncthreads();
    float inv = 1.f / warp_red[0];

    for (int d = tid; d < hd; d += nt) {
        float acc = 0.f;
        for (int j = 0; j < lim; ++j) acc += sc[j] * vh[(long)j * hd + d];
        ob[(long)h * hd + d] = acc * inv;
    }
}

// Batched KV append: write M agents' new K (or V) rows at `pos` into the slotted
// [M, nkv, max_ctx, hd] buffer. src is [M, nkv*hd] (this step's projection rows).
extern "C" __global__ void nh_kv_scatter_batch_f32(
    const float* __restrict__ src,  // [M, nkv*hd]
    float*       __restrict__ dst,  // [M, nkv, max_ctx, hd]
    const float* __restrict__ pos,  // [M] per-slot write position
    const float* __restrict__ active,  // [M] 0 = slot parked, do not append
    int nkv, int hd, int max_ctx, int m_total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int width = nkv * hd;
    if (i >= width || b >= m_total) return;
    if (active[b] == 0.f) return;
    int head = i / hd;
    int d = i - head * hd;
    int p = (int)pos[b];
    if (p < 0) p = 0;
    if (p >= max_ctx) p = max_ctx - 1;
    dst[(long)b * nkv * max_ctx * hd + (long)head * max_ctx * hd + (long)p * hd + d] =
        src[(long)b * width + i];
}
