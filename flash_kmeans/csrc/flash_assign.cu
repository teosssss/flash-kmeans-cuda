#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda/barrier>
#include <cuda_pipeline_primitives.h>

#include <cfloat>
#include <cstddef>

using namespace nvcuda;

#ifndef AUTOKERNEL_ENABLE_PERSISTENT_GEMM
#define AUTOKERNEL_ENABLE_PERSISTENT_GEMM 0
#endif

// ------------- CONFIGURATION -------------
constexpr int WARP_SIZE = 32;
constexpr int THREAD_COUNT = 256;  // 8 warps -> 4x2 warp layout over a 256x128 CTA tile
constexpr int WMMA = 16;
constexpr int SPLIT_K = 4;
#ifndef AUTOKERNEL_WS_PRODUCER_WARPS
#define AUTOKERNEL_WS_PRODUCER_WARPS 3
#endif
#ifndef AUTOKERNEL_WS_CONSUMER_WARPS
#define AUTOKERNEL_WS_CONSUMER_WARPS 4
#endif
constexpr int WARP_SPECIALIZED_PRODUCER_WARPS = AUTOKERNEL_WS_PRODUCER_WARPS;
constexpr int WARP_SPECIALIZED_CONSUMER_WARPS = AUTOKERNEL_WS_CONSUMER_WARPS;
constexpr int WARP_SPECIALIZED_THREAD_COUNT =
    (WARP_SPECIALIZED_PRODUCER_WARPS + WARP_SPECIALIZED_CONSUMER_WARPS) * WARP_SIZE;

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816_F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                                          \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"                     \
                 : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                  \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0), "f"(RC1), "f"(RC2),\
                   "f"(RC3))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define AUTOKERNEL_CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define AUTOKERNEL_CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

namespace {

constexpr int PAD_A = 8;
constexpr int PAD_B = 8;
constexpr int K_TILE = 32;
constexpr int TILE_M = 256;
constexpr int TILE_N = 128;
constexpr int A_STRIDE = K_TILE + PAD_A;
constexpr int B_STRIDE = K_TILE + PAD_B;
constexpr int A_STAGE_ELEMS = TILE_M * A_STRIDE;
constexpr int B_STAGE_ELEMS = TILE_N * B_STRIDE;
constexpr int STAGE_COUNT = 3;

template <int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    AUTOKERNEL_CP_ASYNC_CG(smem_addr, src, 16);
}

__device__ __forceinline__ void zero_16B(half* dst) {
    uint4 zeros = make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(dst) = zeros;
}

__device__ __forceinline__ void rotate_stage_triplet_3(int& current_stage, int& next_stage, int& prefetch_stage) {
    const int old_current = current_stage;
    current_stage = next_stage;
    next_stage = prefetch_stage;
    prefetch_stage = old_current;
}

__device__ __forceinline__ void prefetch_a_stage_256x32(
    const half* A,
    half* sA_stage,
    int block_row_start,
    int row_A,
    int col_A,
    int K,
    int k_base,
    int M
) {
    // Each thread owns one 16-byte vector per 64-row pass. Together the CTA covers
    // a full 256x32 slice of the point tile for the current K_TILE position.
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int local_row = pass * 64 + row_A;
        half* dst = sA_stage + local_row * A_STRIDE + col_A;
        const int global_row = block_row_start + local_row;
        const int global_col = k_base + col_A;

        if (global_row < M && (global_col + 7) < K) {
            const half* src = A + global_row * K + global_col;
            cp_async_cg_16B(dst, src);
        } else {
            // Tail handling is explicit so the ldmatrix path always sees valid shared data.
            // This is only taken on edge tiles along M or K.
            if (global_row < M) {
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int d = global_col + t;
                    dst[t] = (d < K) ? A[global_row * K + d] : __float2half(0.0f);
                }
            } else {
                zero_16B(dst);
            }
        }
    }
}

__device__ __forceinline__ void prefetch_b_stage_128x32(
    const half* B_col_major,
    half* sB_stage,
    int block_col_start,
    int row_B,
    int col_B,
    int K,
    int k_base,
    int N
) {
    // Each thread owns one 16-byte vector per 64-row pass. Together the CTA covers
    // a full 128x32 slice of the centroid tile for the current K_TILE position.
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int local_row = pass * 64 + row_B;
        half* dst = sB_stage + local_row * B_STRIDE + col_B;
        const int global_row = block_col_start + local_row;
        const int global_col = k_base + col_B;

        if (global_row < N && (global_col + 7) < K) {
            const half* src = B_col_major + global_row * K + global_col;
            cp_async_cg_16B(dst, src);
        } else {
            // Tail handling mirrors the A path so partial centroid tiles are still safe.
            if (global_row < N) {
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int d = global_col + t;
                    dst[t] = (d < K) ? B_col_major[global_row * K + d] : __float2half(0.0f);
                }
            } else {
                zero_16B(dst);
            }
        }
    }
}

__device__ __forceinline__ void prefetch_a_stage_256x32_aligned(
    const half* A,
    half* sA_stage,
    int block_row_start,
    int row_A,
    int col_A,
    int K,
    int k_base
) {
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int local_row = pass * 64 + row_A;
        half* dst = sA_stage + local_row * A_STRIDE + col_A;
        const half* src = A + (block_row_start + local_row) * K + (k_base + col_A);
        cp_async_cg_16B(dst, src);
    }
}

__device__ __forceinline__ void prefetch_b_stage_128x32_aligned(
    const half* B_col_major,
    half* sB_stage,
    int block_col_start,
    int row_B,
    int col_B,
    int K,
    int k_base
) {
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int local_row = pass * 64 + row_B;
        half* dst = sB_stage + local_row * B_STRIDE + col_B;
        const half* src = B_col_major + (block_col_start + local_row) * K + (k_base + col_B);
        cp_async_cg_16B(dst, src);
    }
}

__device__ __forceinline__ void gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
    const half* sA_stage,
    const half* sB_stage,
    int warp_row,
    int warp_col,
    int lane_id,
    int reg_slot,
    int k_step,
    uint32_t (&a_regs)[2][4][4],
    uint32_t (&b_regs)[2][8][2]
) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int smem_row = warp_row + i * 16;
        const half* tile_ptr_A =
            sA_stage + (smem_row + (lane_id % 16)) * A_STRIDE + k_step + (lane_id / 16) * 8;
        const uint32_t A_smem_lane_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile_ptr_A));
        LDMATRIX_X4(
            a_regs[reg_slot][i][0],
            a_regs[reg_slot][i][1],
            a_regs[reg_slot][i][2],
            a_regs[reg_slot][i][3],
            A_smem_lane_addr
        );
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int smem_col = warp_col + j * 8;
        const half* tile_ptr_B =
            sB_stage + (smem_col + (lane_id % 8)) * B_STRIDE + k_step + ((lane_id / 8) % 2) * 8;
        const uint32_t B_smem_lane_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile_ptr_B));
        LDMATRIX_X2(
            b_regs[reg_slot][j][0],
            b_regs[reg_slot][j][1],
            B_smem_lane_addr
        );
    }
}

__device__ __forceinline__ void gemm_rotate3_mma_reg_pingpong_256_colb_MMA(
    int reg_slot,
    uint32_t (&a_regs)[2][4][4],
    uint32_t (&b_regs)[2][8][2],
    float (&c_regs)[4][8][4]
) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int j_s = (i % 2) ? (8 - j - 1) : j;
            float d0, d1, d2, d3;
            HMMA16816_F32(
                d0,
                d1,
                d2,
                d3,
                a_regs[reg_slot][i][0],
                a_regs[reg_slot][i][1],
                a_regs[reg_slot][i][2],
                a_regs[reg_slot][i][3],
                b_regs[reg_slot][j_s][0],
                b_regs[reg_slot][j_s][1],
                c_regs[i][j_s][0],
                c_regs[i][j_s][1],
                c_regs[i][j_s][2],
                c_regs[i][j_s][3]
            );
            c_regs[i][j_s][0] = d0;
            c_regs[i][j_s][1] = d1;
            c_regs[i][j_s][2] = d2;
            c_regs[i][j_s][3] = d3;
        }
    }
}

}  // namespace

template <int NUM_THREADS = THREAD_COUNT>
__global__ void row_l2_norm_kernel(
    const half* __restrict__ x,
    float* __restrict__ norms,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;

    // Each thread walks the row with a block-stride loop so the kernel works for any cols.
    for (int col = tid; col < cols; col += NUM_THREADS) {
        const float v = __half2float(x[row * cols + col]);
        sum += v * v;
    }

    sum = warp_reduce_sum(sum);

    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];

    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            norms[row] = block_sum;
        }
    }
}

__global__ void flash_assign_kernel_256x128x32(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K
) {
    // For flash-assign reinterpret the GEMM operands as:
    // - A          : points      [M, K] where K is the feature dimension D
    // - B_col_major: centroids   [N, K]
    // - output_ids : final centroid assignment for each point row in the CTA tile
    // - output_dists: optional final best distance for each point row
    //
    // The CTA owns one point tile [256, K] and streams centroid tiles [128, K].
    // The control flow is:
    //   outer loop over centroid tiles (N dimension)
    //   inner loop over feature tiles of size K_TILE=32 (K dimension)
    //
    // Both A and B are asynchronously triple-buffered over the K dimension.
    const int block_row = blockIdx.x;
    const int block_row_start = block_row * TILE_M;
    if (block_row_start >= M) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = (warp_id / 2) * 64;
    const int warp_col = (warp_id % 2) * 64;

    const int row_A = tid / 4;
    const int col_A = (tid % 4) * 8;
    const int row_B = tid / 4;
    const int col_B = (tid % 4) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);
    float* s_x_norm = reinterpret_cast<float*>(smem + STAGE_COUNT * A_STAGE_ELEMS + STAGE_COUNT * B_STAGE_ELEMS);
    float* s_c_norm = s_x_norm + TILE_M;
    float* s_running_best_dist = s_c_norm + TILE_N;
    int* s_running_best_idx = reinterpret_cast<int*>(s_running_best_dist + TILE_M);
    float* s_warp_row_min = reinterpret_cast<float*>(s_running_best_idx + TILE_M);
    int* s_warp_row_idx = reinterpret_cast<int*>(s_warp_row_min + 8 * 64);

    uint32_t a_regs[2][4][4];
    uint32_t b_regs[2][8][2];
    float c_regs[4][8][4];
    float lane_row_min[8];
    int lane_row_idx[8];

    // Norm staging:
    // - point norms are invariant across the outer centroid-tile loop, so load the 256-entry
    //   tile once per CTA before entering the loop
    // - centroid norms depend on the current 128-column centroid tile, so load them inside
    //   the outer loop at the beginning of each iteration
    for (int idx = tid; idx < TILE_M; idx += THREAD_COUNT) {
        const int global_row = block_row_start + idx;
        s_x_norm[idx] = (global_row < M) ? x_norm[global_row] : 0.0f;
        s_running_best_dist[idx] = FLT_MAX;
        s_running_best_idx[idx] = -1;
    }
    __syncthreads();

    const int k_tile_count = (K + K_TILE - 1) / K_TILE;
    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    // Global prologue:
    // Prime the async pipeline once. If the current centroid tile has only one K-slice,
    // use the second stage slot to start prefetching the first K-slice of the next centroid
    // tile instead of leaving the pipeline idle.
    bool have_second_stage = false;
    if (k_tile_count > 0) {
        prefetch_a_stage_256x32(A, sA[current_stage], block_row_start, row_A, col_A, K, 0, M);
        prefetch_b_stage_128x32(B_col_major, sB[current_stage], 0, row_B, col_B, K, 0, N);
        __pipeline_commit();
    }
    if (k_tile_count > 1) {
        prefetch_a_stage_256x32(A, sA[next_stage], block_row_start, row_A, col_A, K, K_TILE, M);
        prefetch_b_stage_128x32(B_col_major, sB[next_stage], 0, row_B, col_B, K, K_TILE, N);
        __pipeline_commit();
        have_second_stage = true;
    } else if (N > TILE_N) {
        prefetch_a_stage_256x32(A, sA[next_stage], block_row_start, row_A, col_A, K, 0, M);
        prefetch_b_stage_128x32(B_col_major, sB[next_stage], TILE_N, row_B, col_B, K, 0, N);
        __pipeline_commit();
        have_second_stage = true;
    }

    __pipeline_wait_prior(have_second_stage ? 1 : 0);
    __syncthreads();

    // Stage 1:
    // Walk centroid tiles along N. Each iteration computes the dot-product tile
    // between one fixed point tile [256, K] and one centroid tile [128, K].
    for (int block_col_start = 0; block_col_start < N; block_col_start += TILE_N) {
        for (int idx = tid; idx < TILE_N; idx += THREAD_COUNT) {
            const int global_col = block_col_start + idx;
            s_c_norm[idx] = (global_col < N) ? c_norm[global_col] : 0.0f;
        }
        __syncthreads();

        // Stage 2:
        // Reset the accumulator tile for the current pair (X_i, C_j).
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                c_regs[i][j][0] = 0.0f;
                c_regs[i][j][1] = 0.0f;
                c_regs[i][j][2] = 0.0f;
                c_regs[i][j][3] = 0.0f;
            }
        }

        // Stage 3:
        // Inner loop over K_TILE slices. This is the exact place where the point tile does
        // not have to live entirely in shared memory. Only one 256x32 slice of A and one
        // 128x32 slice of B are resident per stage.
        for (int k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
            // Load the current stage into registers in two 16-wide ldmatrix steps.
            gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
                sA[current_stage],
                sB[current_stage],
                warp_row,
                warp_col,
                lane_id,
                reg_curr,
                0,
                a_regs,
                b_regs
            );
            gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
                sA[current_stage],
                sB[current_stage],
                warp_row,
                warp_col,
                lane_id,
                reg_next,
                WMMA,
                a_regs,
                b_regs
            );

            // Issue mma on the current K_TILE=32 slice.
            gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);
            gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

            // Steady-state prefetch:
            // Keep the three-stage buffer alive across centroid-tile boundaries.
            // Once the current centroid tile is almost done, the lookahead loads start
            // prefetching the first K-slices of the next centroid tile instead.
            bool issued_prefetch = false;
            int prefetch_block_col_start = block_col_start;
            int k_prefetch = 0;

            if (k_tile_idx + 2 < k_tile_count) {
                k_prefetch = (k_tile_idx + 2) * K_TILE;
                issued_prefetch = true;
            } else {
                const int next_block_col_start = block_col_start + TILE_N;
                const int next_tile_k_idx = k_tile_idx + 2 - k_tile_count;
                if (next_block_col_start < N && next_tile_k_idx < k_tile_count) {
                    prefetch_block_col_start = next_block_col_start;
                    k_prefetch = next_tile_k_idx * K_TILE;
                    issued_prefetch = true;
                }
            }

            if (issued_prefetch) {
                prefetch_a_stage_256x32(
                    A,
                    sA[prefetch_stage],
                    block_row_start,
                    row_A,
                    col_A,
                    K,
                    k_prefetch,
                    M
                );
                prefetch_b_stage_128x32(
                    B_col_major,
                    sB[prefetch_stage],
                    prefetch_block_col_start,
                    row_B,
                    col_B,
                    K,
                    k_prefetch,
                    N
                );
                __pipeline_commit();
            }

            // Before the next iteration, wait until the next stage is ready, then rotate
            // the stage indices:
            //   current <- next
            //   next    <- prefetch
            //   prefetch<- old current
            if (k_tile_idx + 1 < k_tile_count) {
                __pipeline_wait_prior(issued_prefetch ? 1 : 0);
                __syncthreads();
                rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
            }
        }

        // Stage 4:
        // At this point c_regs contains the full dot-product tile for:
        //   points   [block_row_start : block_row_start + 256)
        //   centroids[block_col_start : block_col_start + 128)
        //
        // Step 1:
        // Convert the FP32 dot-product accumulators into squared Euclidean distances
        // in place. We intentionally reuse c_regs so the next epilogue stages can reduce
        // directly over distances without allocating another register tile.
        //
        // Verified fragment mapping for one m16n8 accumulator fragment:
        //   row0 = warp_row + i * 16 + lane_id / 4
        //   row1 = row0 + 8
        //   col0 = warp_col + j * 8 + (lane_id % 4) * 2
        //
        // Then the four accumulator values owned by the lane map to:
        //   c_regs[i][j][0] -> (row0, col0 + 0)
        //   c_regs[i][j][1] -> (row0, col0 + 1)
        //   c_regs[i][j][2] -> (row1, col0 + 0)
        //   c_regs[i][j][3] -> (row1, col0 + 1)
        //
        // Therefore the norm lookup is:
        //   s_x_norm[row0], s_x_norm[row1], s_c_norm[col0 + 0], s_c_norm[col0 + 1]
        //
        // Tail rows/cols are clamped to +inf so later min-reduction logic can ignore
        // them naturally without requiring special cases in the reduction tree.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row0 = warp_row + i * 16 + lane_id / 4;
            const int row1 = row0 + 8;
            const bool row0_valid = (block_row_start + row0) < M;
            const bool row1_valid = (block_row_start + row1) < M;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col0 = warp_col + j * 8 + (lane_id % 4) * 2;
                const int col1 = col0 + 1;
                const bool col0_valid = (block_col_start + col0) < N;
                const bool col1_valid = (block_col_start + col1) < N;

                const float x_norm_row0 = row0_valid ? s_x_norm[row0] : FLT_MAX;
                const float x_norm_row1 = row1_valid ? s_x_norm[row1] : FLT_MAX;
                const float c_norm_col0 = col0_valid ? s_c_norm[col0] : FLT_MAX;
                const float c_norm_col1 = col1_valid ? s_c_norm[col1] : FLT_MAX;

                const float dot00 = c_regs[i][j][0];
                const float dot01 = c_regs[i][j][1];
                const float dot10 = c_regs[i][j][2];
                const float dot11 = c_regs[i][j][3];

                c_regs[i][j][0] = (row0_valid && col0_valid)
                    ? (x_norm_row0 + c_norm_col0 - 2.0f * dot00)
                    : FLT_MAX;
                c_regs[i][j][1] = (row0_valid && col1_valid)
                    ? (x_norm_row0 + c_norm_col1 - 2.0f * dot01)
                    : FLT_MAX;
                c_regs[i][j][2] = (row1_valid && col0_valid)
                    ? (x_norm_row1 + c_norm_col0 - 2.0f * dot10)
                    : FLT_MAX;
                c_regs[i][j][3] = (row1_valid && col1_valid)
                    ? (x_norm_row1 + c_norm_col1 - 2.0f * dot11)
                    : FLT_MAX;
            }
        }

        // Step 2:
        // Do the first reduction stage entirely in registers.
        //
        // One lane touches 8 distinct rows total:
        //   for each i in [0, 3]:
        //     slot 2*i + 0 -> row0 = warp_row + i*16 + lane_id/4
        //     slot 2*i + 1 -> row1 = row0 + 8
        //
        // For each of those rows the lane sees 16 candidate distances:
        //   8 j-iterations * 2 columns per iteration.
        //
        // Here we compress those 16 candidates down to one (min distance, centroid index)
        // pair per row touched by the lane. This is only the lane-local reduction.
        // The next step will reduce across the 4 consecutive lanes that share a row.
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            lane_row_min[r] = FLT_MAX;
            lane_row_idx[r] = -1;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col0 = warp_col + j * 8 + (lane_id % 4) * 2;
                const int global_col0 = block_col_start + col0;
                const int global_col1 = global_col0 + 1;

                const float dist00 = c_regs[i][j][0];
                const float dist01 = c_regs[i][j][1];
                const float dist10 = c_regs[i][j][2];
                const float dist11 = c_regs[i][j][3];

                // slot 2*i + 0 corresponds to row0 for this i
                if (dist00 < lane_row_min[2 * i + 0]) {
                    lane_row_min[2 * i + 0] = dist00;
                    lane_row_idx[2 * i + 0] = global_col0;
                }
                if (dist01 < lane_row_min[2 * i + 0]) {
                    lane_row_min[2 * i + 0] = dist01;
                    lane_row_idx[2 * i + 0] = global_col1;
                }

                // slot 2*i + 1 corresponds to row1 for this i
                if (dist10 < lane_row_min[2 * i + 1]) {
                    lane_row_min[2 * i + 1] = dist10;
                    lane_row_idx[2 * i + 1] = global_col0;
                }
                if (dist11 < lane_row_min[2 * i + 1]) {
                    lane_row_min[2 * i + 1] = dist11;
                    lane_row_idx[2 * i + 1] = global_col1;
                }
            }
        }

        // Step 3:
        // Reduce across the 4 consecutive lanes that share each row.
        //
        // For a fixed row, the owning lanes are exactly one width-4 subgroup:
        //   lane_id = 4*g + {0,1,2,3}
        // where g identifies the row inside the current 8-row block.
        //
        // After Step 2, each of those 4 lanes holds its own local minimum over the 16
        // candidates it saw for that row. Now we reduce those 4 local minima down to one
        // warp-local minimum for that row over the full 64 columns owned by the warp.
        //
        // We use width=4 shuffles so communication stays inside each consecutive 4-lane
        // subgroup. Only subgroup leader lane (lane_id % 4 == 0) is considered to own the
        // final reduced value after this step.
        const int lane_in_row_group = lane_id % 4;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float reduced_min = lane_row_min[r];
            int reduced_idx = lane_row_idx[r];

            float other_min = __shfl_down_sync(0xffffffffu, reduced_min, 2, 4);
            int other_idx = __shfl_down_sync(0xffffffffu, reduced_idx, 2, 4);
            if (other_min < reduced_min) {
                reduced_min = other_min;
                reduced_idx = other_idx;
            }

            other_min = __shfl_down_sync(0xffffffffu, reduced_min, 1, 4);
            other_idx = __shfl_down_sync(0xffffffffu, reduced_idx, 1, 4);
            if (other_min < reduced_min) {
                reduced_min = other_min;
                reduced_idx = other_idx;
            }

            lane_row_min[r] = reduced_min;
            lane_row_idx[r] = reduced_idx;
        }

        // Step 4:
        // Materialize the warp-local row minima into shared memory.
        //
        // Only the subgroup leader lane owns the logical result for each row after the
        // width-4 reduction above. Each leader writes 8 rows, one for every row slot it
        // touched:
        //   row_local = i*16 + subgroup_id [+8 for the row1 slot]
        //
        // Shared-memory layout is [warp_id][64 rows within that warp tile].
        if (lane_in_row_group == 0) {
            const int subgroup_id = lane_id / 4;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row_local0 = i * 16 + subgroup_id;
                const int row_local1 = row_local0 + 8;

                s_warp_row_min[warp_id * 64 + row_local0] = lane_row_min[2 * i + 0];
                s_warp_row_idx[warp_id * 64 + row_local0] = lane_row_idx[2 * i + 0];

                s_warp_row_min[warp_id * 64 + row_local1] = lane_row_min[2 * i + 1];
                s_warp_row_idx[warp_id * 64 + row_local1] = lane_row_idx[2 * i + 1];
            }
        }
        __syncthreads();

        // Step 5:
        // Merge the two warps that correspond to the same CTA row range, then compare the
        // merged result against the running best over all previously processed centroid tiles.
        //
        // Warp layout across the CTA is:
        //   warp 0,1 -> rows   0..63,  cols  0..63 and  64..127
        //   warp 2,3 -> rows  64..127, cols  0..63 and  64..127
        //   warp 4,5 -> rows 128..191, cols  0..63 and  64..127
        //   warp 6,7 -> rows 192..255, cols  0..63 and  64..127
        //
        // One thread per CTA row performs:
        //   best_tile_row = min(best_from_left_warp, best_from_right_warp)
        //   running_best_row = min(running_best_row, best_tile_row)
        for (int row = tid; row < TILE_M; row += THREAD_COUNT) {
            const int global_row = block_row_start + row;
            if (global_row < M) {
                const int warp_row_group = row / 64;
                const int row_in_warp = row % 64;
                const int left_warp = warp_row_group * 2 + 0;
                const int right_warp = warp_row_group * 2 + 1;

                float best_dist = s_warp_row_min[left_warp * 64 + row_in_warp];
                int best_idx = s_warp_row_idx[left_warp * 64 + row_in_warp];

                const float right_dist = s_warp_row_min[right_warp * 64 + row_in_warp];
                const int right_idx = s_warp_row_idx[right_warp * 64 + row_in_warp];

                if (right_dist < best_dist) {
                    best_dist = right_dist;
                    best_idx = right_idx;
                }

                if (best_dist < s_running_best_dist[row]) {
                    s_running_best_dist[row] = best_dist;
                    s_running_best_idx[row] = best_idx;
                }
            }
        }
        __syncthreads();

        // Boundary transition:
        // The last two iterations above may already have prefetched the first K-slices
        // of the next centroid tile. Rotate once more so the next outer-loop iteration
        // starts directly from those prefetched stages.
        if (block_col_start + TILE_N < N) {
            __pipeline_wait_prior(1);
            __syncthreads();
            rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
        }

    }

    // Step 6:
    // The outer centroid-tile loop is complete, so s_running_best_idx / s_running_best_dist
    // now hold the final assignment result for each valid row of the CTA point tile.
    //
    // Write the final centroid id to output_ids and, if requested, the final minimum distance
    // to output_dists.
    for (int row = tid; row < TILE_M; row += THREAD_COUNT) {
        const int global_row = block_row_start + row;
        if (global_row < M) {
            output_ids[global_row] = s_running_best_idx[row];
            if (output_dists != nullptr) {
                output_dists[global_row] = s_running_best_dist[row];
            }
        }
    }
}

__global__ void flash_assign_kernel_256x128x32_aligned(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K
) {
    const int block_row = blockIdx.x;
    const int block_row_start = block_row * TILE_M;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = (warp_id / 2) * 64;
    const int warp_col = (warp_id % 2) * 64;

    const int row_A = tid / 4;
    const int col_A = (tid % 4) * 8;
    const int row_B = tid / 4;
    const int col_B = (tid % 4) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);
    float* s_x_norm = reinterpret_cast<float*>(smem + STAGE_COUNT * A_STAGE_ELEMS + STAGE_COUNT * B_STAGE_ELEMS);
    float* s_c_norm = s_x_norm + TILE_M;
    float* s_running_best_dist = s_c_norm + TILE_N;
    int* s_running_best_idx = reinterpret_cast<int*>(s_running_best_dist + TILE_M);
    float* s_warp_row_min = reinterpret_cast<float*>(s_running_best_idx + TILE_M);
    int* s_warp_row_idx = reinterpret_cast<int*>(s_warp_row_min + 8 * 64);

    uint32_t a_regs[2][4][4];
    uint32_t b_regs[2][8][2];
    float c_regs[4][8][4];
    float lane_row_min[8];
    int lane_row_idx[8];

    for (int idx = tid; idx < TILE_M; idx += THREAD_COUNT) {
        s_x_norm[idx] = x_norm[block_row_start + idx];
        s_running_best_dist[idx] = FLT_MAX;
        s_running_best_idx[idx] = -1;
    }
    __syncthreads();

    const int k_tile_count = K / K_TILE;
    const int centroid_tile_count = N / TILE_N;
    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    prefetch_a_stage_256x32_aligned(A, sA[current_stage], block_row_start, row_A, col_A, K, 0);
    prefetch_b_stage_128x32_aligned(B_col_major, sB[current_stage], 0, row_B, col_B, K, 0);
    __pipeline_commit();

    // Fast-path assumptions guarantee K >= 128 and K % 32 == 0, so the startup always
    // consists of two valid slices of the first centroid tile.
    prefetch_a_stage_256x32_aligned(A, sA[next_stage], block_row_start, row_A, col_A, K, K_TILE);
    prefetch_b_stage_128x32_aligned(B_col_major, sB[next_stage], 0, row_B, col_B, K, K_TILE);
    __pipeline_commit();

    __pipeline_wait_prior(1);

    
    for (int block_col_start = 0; block_col_start < N; block_col_start += TILE_N) {
        for (int idx = tid; idx < TILE_N; idx += THREAD_COUNT) {
            s_c_norm[idx] = c_norm[block_col_start + idx];
        }
        __syncthreads();
        
        auto load_stage_half = [&](int stage, int reg_slot, int k_step) {
            gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
                sA[stage], sB[stage], warp_row, warp_col, lane_id, reg_slot, k_step, a_regs, b_regs);
        };
        
        load_stage_half(current_stage, reg_curr, 0);


        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c_regs[i][j][0] = 0.0f;
                c_regs[i][j][1] = 0.0f;
                c_regs[i][j][2] = 0.0f;
                c_regs[i][j][3] = 0.0f;
            }
        }

        const int outer_tile_idx = block_col_start / TILE_N;
        const bool has_next_centroid_tile = (outer_tile_idx + 1) < centroid_tile_count;

        

        // GEMM-style register ping-pong:
        // - preload reg_curr with the first half of the current K_TILE
        // - in steady state:
        //     load reg_next (second half of current stage)
        //     mma reg_curr
        //     prefetch k+2
        //     rotate stages
        //     load reg_curr (first half of new current stage)
        //     mma reg_next

        for (int k_tile_idx = 0; k_tile_idx < k_tile_count - 2; ++k_tile_idx) {
            load_stage_half(current_stage, reg_next, WMMA);
            gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);

            const int k_prefetch = (k_tile_idx + 2) * K_TILE;
            prefetch_a_stage_256x32_aligned(
                A, sA[prefetch_stage], block_row_start, row_A, col_A, K, k_prefetch);
            prefetch_b_stage_128x32_aligned(
                B_col_major, sB[prefetch_stage], block_col_start, row_B, col_B, K, k_prefetch);
            __pipeline_commit();

            __pipeline_wait_prior(1);
            __syncthreads();
            rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

            load_stage_half(current_stage, reg_curr, 0);
            gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);
        }

        // Tail 1: consume slice k_count-2, optionally bootstrap next centroid tile k=0.
        load_stage_half(current_stage, reg_next, WMMA);
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);
        if (has_next_centroid_tile) {
            prefetch_a_stage_256x32_aligned(
                A, sA[prefetch_stage], block_row_start, row_A, col_A, K, 0);
            prefetch_b_stage_128x32_aligned(
                B_col_major, sB[prefetch_stage], block_col_start + TILE_N, row_B, col_B, K, 0);
            __pipeline_commit();
        }
        __pipeline_wait_prior(has_next_centroid_tile ? 1 : 0);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
        load_stage_half(current_stage, reg_curr, 0);
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

        // Tail 2: consume slice k_count-1, optionally bootstrap next centroid tile k=1.
        load_stage_half(current_stage, reg_next, WMMA);
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);
        if (has_next_centroid_tile) {
            prefetch_a_stage_256x32_aligned(
                A, sA[prefetch_stage], block_row_start, row_A, col_A, K, K_TILE);
            prefetch_b_stage_128x32_aligned(
                B_col_major, sB[prefetch_stage], block_col_start + TILE_N, row_B, col_B, K, K_TILE);
            __pipeline_commit();
        }
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            lane_row_min[r] = FLT_MAX;
            lane_row_idx[r] = -1;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row0 = warp_row + i * 16 + lane_id / 4;
            const int row1 = row0 + 8;
            const float x_norm_row0 = s_x_norm[row0];
            const float x_norm_row1 = s_x_norm[row1];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col0 = warp_col + j * 8 + (lane_id % 4) * 2;
                const int global_col0 = block_col_start + col0;
                const int global_col1 = global_col0 + 1;
                const float c_norm_col0 = s_c_norm[col0];
                const float c_norm_col1 = s_c_norm[col0 + 1];

                const float dist00 = fmaf(-2.0f, c_regs[i][j][0], x_norm_row0 + c_norm_col0);
                const float dist01 = fmaf(-2.0f, c_regs[i][j][1], x_norm_row0 + c_norm_col1);
                const float dist10 = fmaf(-2.0f, c_regs[i][j][2], x_norm_row1 + c_norm_col0);
                const float dist11 = fmaf(-2.0f, c_regs[i][j][3], x_norm_row1 + c_norm_col1);

                if (dist00 < lane_row_min[2 * i + 0]) {
                    lane_row_min[2 * i + 0] = dist00;
                    lane_row_idx[2 * i + 0] = global_col0;
                }
                if (dist01 < lane_row_min[2 * i + 0]) {
                    lane_row_min[2 * i + 0] = dist01;
                    lane_row_idx[2 * i + 0] = global_col1;
                }
                if (dist10 < lane_row_min[2 * i + 1]) {
                    lane_row_min[2 * i + 1] = dist10;
                    lane_row_idx[2 * i + 1] = global_col0;
                }
                if (dist11 < lane_row_min[2 * i + 1]) {
                    lane_row_min[2 * i + 1] = dist11;
                    lane_row_idx[2 * i + 1] = global_col1;
                }
            }
        }

        const int lane_in_row_group = lane_id % 4;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float reduced_min = lane_row_min[r];
            int reduced_idx = lane_row_idx[r];
            float other_min = __shfl_down_sync(0xffffffffu, reduced_min, 2, 4);
            int other_idx = __shfl_down_sync(0xffffffffu, reduced_idx, 2, 4);
            if (other_min < reduced_min) {
                reduced_min = other_min;
                reduced_idx = other_idx;
            }
            other_min = __shfl_down_sync(0xffffffffu, reduced_min, 1, 4);
            other_idx = __shfl_down_sync(0xffffffffu, reduced_idx, 1, 4);
            if (other_min < reduced_min) {
                reduced_min = other_min;
                reduced_idx = other_idx;
            }
            lane_row_min[r] = reduced_min;
            lane_row_idx[r] = reduced_idx;
        }

        if (lane_in_row_group == 0) {
            const int subgroup_id = lane_id / 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row_local0 = i * 16 + subgroup_id;
                const int row_local1 = row_local0 + 8;
                s_warp_row_min[warp_id * 64 + row_local0] = lane_row_min[2 * i + 0];
                s_warp_row_idx[warp_id * 64 + row_local0] = lane_row_idx[2 * i + 0];
                s_warp_row_min[warp_id * 64 + row_local1] = lane_row_min[2 * i + 1];
                s_warp_row_idx[warp_id * 64 + row_local1] = lane_row_idx[2 * i + 1];
            }
        }
        __syncthreads();

        for (int row = tid; row < TILE_M; row += THREAD_COUNT) {
            const int warp_row_group = row / 64;
            const int row_in_warp = row % 64;
            const int left_warp = warp_row_group * 2 + 0;
            const int right_warp = warp_row_group * 2 + 1;

            float best_dist = s_warp_row_min[left_warp * 64 + row_in_warp];
            int best_idx = s_warp_row_idx[left_warp * 64 + row_in_warp];
            const float right_dist = s_warp_row_min[right_warp * 64 + row_in_warp];
            const int right_idx = s_warp_row_idx[right_warp * 64 + row_in_warp];
            if (right_dist < best_dist) {
                best_dist = right_dist;
                best_idx = right_idx;
            }
            if (best_dist < s_running_best_dist[row]) {
                s_running_best_dist[row] = best_dist;
                s_running_best_idx[row] = best_idx;
            }
        }
        __syncthreads();

        if (block_col_start + TILE_N < N) {
            __pipeline_wait_prior(1);
            __syncthreads();
            rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
        }
    }

    for (int row = tid; row < TILE_M; row += THREAD_COUNT) {
        output_ids[block_row_start + row] = s_running_best_idx[row];
        if (output_dists != nullptr) {
            output_dists[block_row_start + row] = s_running_best_dist[row];
        }
    }
}

__global__ void flash_assign_kernel_256x128x32_aligned_deferred_reduce(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K
) {
    const int block_row = blockIdx.x;
    const int block_row_start = block_row * TILE_M;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = (warp_id / 2) * 64;
    const int warp_col = (warp_id % 2) * 64;

    const int row_A = tid / 4;
    const int col_A = (tid % 4) * 8;
    const int row_B = tid / 4;
    const int col_B = (tid % 4) * 8;

    extern __shared__ half smem[];
    auto sA = reinterpret_cast<half (*)[A_STAGE_ELEMS]>(smem);
    auto sB = reinterpret_cast<half (*)[B_STAGE_ELEMS]>(smem + STAGE_COUNT * A_STAGE_ELEMS);
    float* s_x_norm = reinterpret_cast<float*>(smem + STAGE_COUNT * A_STAGE_ELEMS + STAGE_COUNT * B_STAGE_ELEMS);
    float* s_c_norm = s_x_norm + TILE_M;
    float* s_warp_row_min = reinterpret_cast<float*>(s_c_norm + TILE_N);
    int* s_warp_row_idx = reinterpret_cast<int*>(s_warp_row_min + 8 * 64);

    uint32_t a_regs[2][4][4];
    uint32_t b_regs[2][8][2];
    float c_regs[4][8][4];
    float lane_row_min[8];
    int lane_row_idx[8];

    for (int idx = tid; idx < TILE_M; idx += THREAD_COUNT) {
        s_x_norm[idx] = x_norm[block_row_start + idx];
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        lane_row_min[r] = FLT_MAX;
        lane_row_idx[r] = -1;
    }

    const int k_tile_count = K / K_TILE;
    const int centroid_tile_count = N / TILE_N;
    int current_stage = 0;
    int next_stage = 1;
    int prefetch_stage = 2;
    int reg_curr = 0;
    int reg_next = 1;

    prefetch_a_stage_256x32_aligned(A, sA[current_stage], block_row_start, row_A, col_A, K, 0);
    prefetch_b_stage_128x32_aligned(B_col_major, sB[current_stage], 0, row_B, col_B, K, 0);
    __pipeline_commit();
    prefetch_a_stage_256x32_aligned(A, sA[next_stage], block_row_start, row_A, col_A, K, K_TILE);
    prefetch_b_stage_128x32_aligned(B_col_major, sB[next_stage], 0, row_B, col_B, K, K_TILE);
    __pipeline_commit();
    __pipeline_wait_prior(1);


    #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c_regs[i][j][0] = 0.0f;
                c_regs[i][j][1] = 0.0f;
                c_regs[i][j][2] = 0.0f;
                c_regs[i][j][3] = 0.0f;
            }
        }

    for (int block_col_start = 0; block_col_start < N; block_col_start += TILE_N) {
        for (int idx = tid; idx < TILE_N; idx += THREAD_COUNT) {
            s_c_norm[idx] = c_norm[block_col_start + idx];
        }
        __syncthreads();

        

        const int outer_tile_idx = block_col_start / TILE_N;
        const bool has_next_centroid_tile = (outer_tile_idx + 1) < centroid_tile_count;

        auto load_stage_half = [&](int stage, int reg_slot, int k_step) {
            gemm_rotate3_load_stage_fragments_reg_pingpong_256_colb_MMA(
                sA[stage], sB[stage], warp_row, warp_col, lane_id, reg_slot, k_step, a_regs, b_regs);
        };

        load_stage_half(current_stage, reg_curr, 0);

        for (int k_tile_idx = 0; k_tile_idx < k_tile_count - 2; ++k_tile_idx) {
            load_stage_half(current_stage, reg_next, WMMA);
            gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);

            const int k_prefetch = (k_tile_idx + 2) * K_TILE;
            prefetch_a_stage_256x32_aligned(
                A, sA[prefetch_stage], block_row_start, row_A, col_A, K, k_prefetch);
            prefetch_b_stage_128x32_aligned(
                B_col_major, sB[prefetch_stage], block_col_start, row_B, col_B, K, k_prefetch);
            __pipeline_commit();

            __pipeline_wait_prior(1);
            __syncthreads();
            rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);

            load_stage_half(current_stage, reg_curr, 0);
            gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);
        }

        load_stage_half(current_stage, reg_next, WMMA);
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);
        if (has_next_centroid_tile) {
            prefetch_a_stage_256x32_aligned(
                A, sA[prefetch_stage], block_row_start, row_A, col_A, K, 0);
            prefetch_b_stage_128x32_aligned(
                B_col_major, sB[prefetch_stage], block_col_start + TILE_N, row_B, col_B, K, 0);
            __pipeline_commit();
        }
        __pipeline_wait_prior(has_next_centroid_tile ? 1 : 0);
        __syncthreads();
        rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
        load_stage_half(current_stage, reg_curr, 0);
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

        load_stage_half(current_stage, reg_next, WMMA);
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_curr, a_regs, b_regs, c_regs);
        if (has_next_centroid_tile) {
            prefetch_a_stage_256x32_aligned(
                A, sA[prefetch_stage], block_row_start, row_A, col_A, K, K_TILE);
            prefetch_b_stage_128x32_aligned(
                B_col_major, sB[prefetch_stage], block_col_start + TILE_N, row_B, col_B, K, K_TILE);
            __pipeline_commit();
        }
        gemm_rotate3_mma_reg_pingpong_256_colb_MMA(reg_next, a_regs, b_regs, c_regs);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row0 = warp_row + i * 16 + lane_id / 4;
            const int row1 = row0 + 8;
            const float x_norm_row0 = s_x_norm[row0];
            const float x_norm_row1 = s_x_norm[row1];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col0 = warp_col + j * 8 + (lane_id % 4) * 2;
                const int global_col0 = block_col_start + col0;
                const int global_col1 = global_col0 + 1;
                const float c_norm_col0 = s_c_norm[col0];
                const float c_norm_col1 = s_c_norm[col0 + 1];

                const float dist00 = fmaf(-2.0f, c_regs[i][j][0], x_norm_row0 + c_norm_col0);
                const float dist01 = fmaf(-2.0f, c_regs[i][j][1], x_norm_row0 + c_norm_col1);
                const float dist10 = fmaf(-2.0f, c_regs[i][j][2], x_norm_row1 + c_norm_col0);
                const float dist11 = fmaf(-2.0f, c_regs[i][j][3], x_norm_row1 + c_norm_col1);

                if (dist00 < lane_row_min[2 * i + 0]) {
                    lane_row_min[2 * i + 0] = dist00;
                    lane_row_idx[2 * i + 0] = global_col0;
                }
                if (dist01 < lane_row_min[2 * i + 0]) {
                    lane_row_min[2 * i + 0] = dist01;
                    lane_row_idx[2 * i + 0] = global_col1;
                }
                if (dist10 < lane_row_min[2 * i + 1]) {
                    lane_row_min[2 * i + 1] = dist10;
                    lane_row_idx[2 * i + 1] = global_col0;
                }
                if (dist11 < lane_row_min[2 * i + 1]) {
                    lane_row_min[2 * i + 1] = dist11;
                    lane_row_idx[2 * i + 1] = global_col1;
                }
            }
        }



        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c_regs[i][j][0] = 0.0f;
                c_regs[i][j][1] = 0.0f;
                c_regs[i][j][2] = 0.0f;
                c_regs[i][j][3] = 0.0f;
            }
        }

        if (block_col_start + TILE_N < N) {
            __pipeline_wait_prior(1);
            rotate_stage_triplet_3(current_stage, next_stage, prefetch_stage);
        }

        
    }

    const int lane_in_row_group = lane_id % 4;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        float reduced_min = lane_row_min[r];
        int reduced_idx = lane_row_idx[r];

        float other_min = __shfl_down_sync(0xffffffffu, reduced_min, 2, 4);
        int other_idx = __shfl_down_sync(0xffffffffu, reduced_idx, 2, 4);
        if (other_min < reduced_min) {
            reduced_min = other_min;
            reduced_idx = other_idx;
        }

        other_min = __shfl_down_sync(0xffffffffu, reduced_min, 1, 4);
        other_idx = __shfl_down_sync(0xffffffffu, reduced_idx, 1, 4);
        if (other_min < reduced_min) {
            reduced_min = other_min;
            reduced_idx = other_idx;
        }

        lane_row_min[r] = reduced_min;
        lane_row_idx[r] = reduced_idx;
    }

    if (lane_in_row_group == 0) {
        const int subgroup_id = lane_id / 4;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row_local0 = i * 16 + subgroup_id;
            const int row_local1 = row_local0 + 8;

            s_warp_row_min[warp_id * 64 + row_local0] = lane_row_min[2 * i + 0];
            s_warp_row_idx[warp_id * 64 + row_local0] = lane_row_idx[2 * i + 0];
            s_warp_row_min[warp_id * 64 + row_local1] = lane_row_min[2 * i + 1];
            s_warp_row_idx[warp_id * 64 + row_local1] = lane_row_idx[2 * i + 1];
        }
    }
    __syncthreads();

    for (int row = tid; row < TILE_M; row += THREAD_COUNT) {
        const int warp_row_group = row / 64;
        const int row_in_warp = row % 64;
        const int left_warp = warp_row_group * 2 + 0;
        const int right_warp = warp_row_group * 2 + 1;

        float best_dist = s_warp_row_min[left_warp * 64 + row_in_warp];
        int best_idx = s_warp_row_idx[left_warp * 64 + row_in_warp];
        const float right_dist = s_warp_row_min[right_warp * 64 + row_in_warp];
        const int right_idx = s_warp_row_idx[right_warp * 64 + row_in_warp];
        if (right_dist < best_dist) {
            best_dist = right_dist;
            best_idx = right_idx;
        }

        output_ids[block_row_start + row] = best_idx;
        if (output_dists != nullptr) {
            output_dists[block_row_start + row] = best_dist;
        }
    }
}

size_t flash_assign_smem_bytes_256x128x32() {
    // Dynamic shared memory layout used by the kernel:
    // 1. sA[3]                : 3 stages of the 256x32 point tile
    // 2. sB[3]                : 3 stages of the 128x32 centroid tile
    // 3. s_x_norm[256]        : point norms for the CTA tile
    // 4. s_c_norm[128]        : centroid norms for the current centroid tile
    // 5. s_running_best_dist  : running best distance per CTA row
    // 6. s_running_best_idx   : running best centroid index per CTA row
    // 7. s_warp_row_min       : warp-local row minima before cross-warp merge
    // 8. s_warp_row_idx       : warp-local centroid indices before cross-warp merge
    return
        STAGE_COUNT * A_STAGE_ELEMS * sizeof(half) +
        STAGE_COUNT * B_STAGE_ELEMS * sizeof(half) +
        TILE_M * sizeof(float) +
        TILE_N * sizeof(float) +
        TILE_M * sizeof(float) +
        TILE_M * sizeof(int) +
        8 * 64 * sizeof(float) +
        8 * 64 * sizeof(int);
}

cudaError_t launch_point_l2_norm_kernel(
    const half* points,
    float* point_norms,
    int num_points,
    int dim,
    cudaStream_t stream
) {
    const dim3 block(THREAD_COUNT);
    const dim3 grid(num_points);
    row_l2_norm_kernel<<<grid, block, 0, stream>>>(
        points,
        point_norms,
        num_points,
        dim
    );
    return cudaGetLastError();
}

cudaError_t launch_centroid_l2_norm_kernel(
    const half* centroids,
    float* centroid_norms,
    int num_centroids,
    int dim,
    cudaStream_t stream
) {
    const dim3 block(THREAD_COUNT);
    const dim3 grid(num_centroids);
    row_l2_norm_kernel<<<grid, block, 0, stream>>>(
        centroids,
        centroid_norms,
        num_centroids,
        dim
    );
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_kernel_256x128x32(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    const dim3 block(THREAD_COUNT);
    const dim3 grid((M + TILE_M - 1) / TILE_M);
    const size_t smem_bytes = flash_assign_smem_bytes_256x128x32();

    cudaError_t err = cudaFuncSetAttribute(
        flash_assign_kernel_256x128x32,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)
    );
    if (err != cudaSuccess) {
        return err;
    }

    const bool use_aligned_fast_path =
        (M % TILE_M == 0) &&
        (N % TILE_N == 0) &&
        (K >= 128) &&
        (K % K_TILE == 0);

    if (use_aligned_fast_path) {
        err = cudaFuncSetAttribute(
            flash_assign_kernel_256x128x32_aligned,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (err != cudaSuccess) {
            return err;
        }
        flash_assign_kernel_256x128x32_aligned<<<grid, block, smem_bytes, stream>>>(
            A, B_col_major, x_norm, c_norm, output_ids, output_dists, M, N, K);
    } else {
        flash_assign_kernel_256x128x32<<<grid, block, smem_bytes, stream>>>(
            A, B_col_major, x_norm, c_norm, output_ids, output_dists, M, N, K);
    }
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_kernel_256x128x32_aligned_deferred_reduce(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    const dim3 block(THREAD_COUNT);
    const dim3 grid((M + TILE_M - 1) / TILE_M);
    const size_t smem_bytes = flash_assign_smem_bytes_256x128x32();

    cudaError_t err = cudaFuncSetAttribute(
        flash_assign_kernel_256x128x32_aligned_deferred_reduce,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)
    );
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_kernel_256x128x32_aligned_deferred_reduce<<<grid, block, smem_bytes, stream>>>(
        A, B_col_major, x_norm, c_norm, output_ids, output_dists, M, N, K);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_complete_256x128x32(
    const half* points,
    const half* centroids,
    float* point_norms,
    float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int num_points,
    int num_centroids,
    int dim,
    cudaStream_t stream
) {
    // Full launch order for the current implementation:
    // 1. Compute squared L2 norms for all points.
    // 2. Compute squared L2 norms for all centroids.
    // 3. Launch the assignment kernel using those norm buffers.
    //
    // The norm buffers are caller-owned scratch/output buffers so this helper does not
    // allocate device memory internally.
    cudaError_t err = launch_point_l2_norm_kernel(
        points,
        point_norms,
        num_points,
        dim,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = launch_centroid_l2_norm_kernel(
        centroids,
        centroid_norms,
        num_centroids,
        dim,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = launch_flash_assign_kernel_256x128x32(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    return cudaSuccess;
}

cudaError_t launch_flash_assign_complete_256x128x32_aligned_deferred_reduce(
    const half* points,
    const half* centroids,
    float* point_norms,
    float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int num_points,
    int num_centroids,
    int dim,
    cudaStream_t stream
) {
    cudaError_t err = launch_point_l2_norm_kernel(
        points,
        point_norms,
        num_points,
        dim,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = launch_centroid_l2_norm_kernel(
        centroids,
        centroid_norms,
        num_centroids,
        dim,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = launch_flash_assign_kernel_256x128x32_aligned_deferred_reduce(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    return cudaSuccess;
}
