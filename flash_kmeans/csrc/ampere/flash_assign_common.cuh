#pragma once

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
