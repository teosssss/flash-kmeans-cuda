#include "flash_assign_hopper_common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_kmeans::hopper {

namespace {

// Common helpers and shared-storage layouts.

__device__ __forceinline__ void consumer_barrier() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Synchronize only consumer warpgroups (256 threads) to avoid divergent CTA barriers.
    asm volatile("bar.sync 1, 256;" : : : "memory");
#endif
}

struct HopperSharedStorage {
    alignas(128) half a_tiles[kStageCount][kTileM * kPaddedTileK];
    alignas(128) half b_tiles[kStageCount][kTileN * kPaddedTileK];
    alignas(16) float x_norm[kTileM];
    alignas(16) float c_norm[kTileN];
    alignas(16) float running_best_dist[kTileM];
    alignas(16) int running_best_idx[kTileM];
    alignas(16) float warp_row_min[kConsumerWarpgroups * kConsumerRows];
    alignas(16) int warp_row_idx[kConsumerWarpgroups * kConsumerRows];
    alignas(8) uint64_t full[kStageCount];
    alignas(8) uint64_t empty[kStageCount];
};

// Norm kernels.

__global__ void row_l2_norm_kernel_hopper(
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
    for (int col = tid; col < cols; col += kThreadCount) {
        const float v = __half2float(x[row * cols + col]);
        sum += v * v;
    }

    sum = warp_reduce_sum(sum);
    __shared__ float warp_sums[kThreadCount / kWarpSize];
    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = (lane < (kThreadCount / kWarpSize)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            norms[row] = block_sum;
        }
    }
}

__device__ __forceinline__ void consumer_compute_rows(
    HopperSharedStorage& smem,
    int consumer_group,
    int consumer_wg_tid,
    int block_col_start,
    int N,
    float (&accum)[2][8][8]
) {
    const int wg_lane = consumer_wg_tid % kWarpSize;
    const int wg_warp = consumer_wg_tid / kWarpSize;
    const int row_group_base = consumer_group * kConsumerRows + wg_warp * 16;
    float lane_min[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    int lane_idx[4] = {-1, -1, -1, -1};

    #pragma unroll
    for (int m_it = 0; m_it < 2; ++m_it) {
        const int yo = row_group_base + m_it * kWgmmaM;
        const int row0 = yo + wg_lane / 4;
        const int row1 = row0 + 8;
        const bool row0_valid = row0 < kTileM;
        const bool row1_valid = row1 < kTileM;

        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int col = w * 16 + 2 * (consumer_wg_tid % 4);
            const int global_col0 = block_col_start + col;
            const int global_col1 = global_col0 + 1;
            const int global_col8 = global_col0 + 8;
            const int global_col9 = global_col0 + 9;

            const float x0 = row0_valid ? smem.x_norm[row0] : 0.0f;
            const float x1 = row1_valid ? smem.x_norm[row1] : 0.0f;

            float v00 = FLT_MAX;
            float v01 = FLT_MAX;
            float v08 = FLT_MAX;
            float v09 = FLT_MAX;
            float v10 = FLT_MAX;
            float v11 = FLT_MAX;
            float v18 = FLT_MAX;
            float v19 = FLT_MAX;

            if (row0_valid && global_col0 < N) v00 = x0 + smem.c_norm[col + 0] - 2.0f * accum[m_it][w][0];
            if (row0_valid && global_col1 < N) v01 = x0 + smem.c_norm[col + 1] - 2.0f * accum[m_it][w][1];
            if (row1_valid && global_col0 < N) v10 = x1 + smem.c_norm[col + 0] - 2.0f * accum[m_it][w][2];
            if (row1_valid && global_col1 < N) v11 = x1 + smem.c_norm[col + 1] - 2.0f * accum[m_it][w][3];
            if (row0_valid && global_col8 < N) v08 = x0 + smem.c_norm[col + 8] - 2.0f * accum[m_it][w][4];
            if (row0_valid && global_col9 < N) v09 = x0 + smem.c_norm[col + 9] - 2.0f * accum[m_it][w][5];
            if (row1_valid && global_col8 < N) v18 = x1 + smem.c_norm[col + 8] - 2.0f * accum[m_it][w][6];
            if (row1_valid && global_col9 < N) v19 = x1 + smem.c_norm[col + 9] - 2.0f * accum[m_it][w][7];

            if (better_candidate(v00, global_col0, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v00;
                lane_idx[m_it * 2 + 0] = global_col0;
            }
            if (better_candidate(v01, global_col1, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v01;
                lane_idx[m_it * 2 + 0] = global_col1;
            }
            if (better_candidate(v08, global_col8, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v08;
                lane_idx[m_it * 2 + 0] = global_col8;
            }
            if (better_candidate(v09, global_col9, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v09;
                lane_idx[m_it * 2 + 0] = global_col9;
            }
            if (better_candidate(v10, global_col0, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v10;
                lane_idx[m_it * 2 + 1] = global_col0;
            }
            if (better_candidate(v11, global_col1, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v11;
                lane_idx[m_it * 2 + 1] = global_col1;
            }
            if (better_candidate(v18, global_col8, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v18;
                lane_idx[m_it * 2 + 1] = global_col8;
            }
            if (better_candidate(v19, global_col9, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v19;
                lane_idx[m_it * 2 + 1] = global_col9;
            }
        }
    }

    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        reduce_group_min(lane_min[slot], lane_idx[slot]);
        if ((consumer_wg_tid % 4) == 0) {
            const int local_row = row_group_base + (slot / 2) * kWgmmaM + (wg_lane / 4) + (slot % 2) * 8;
            if (local_row < kTileM) {
                const int row_offset = local_row;
                const float candidate_dist = lane_min[slot];
                const int candidate_idx = lane_idx[slot];
                if (better_candidate(candidate_dist, candidate_idx, smem.running_best_dist[local_row], smem.running_best_idx[local_row])) {
                    smem.running_best_dist[local_row] = candidate_dist;
                    smem.running_best_idx[local_row] = candidate_idx;
                }
                smem.warp_row_min[row_offset] = candidate_dist;
                smem.warp_row_idx[row_offset] = candidate_idx;
            }
        }
    }
}




__device__ __forceinline__ void consumer_compute_rows_fast(
    HopperSharedStorage& smem,
    int consumer_group,
    int consumer_wg_tid,
    int block_col_start,
    int N,
    float (&accum)[2][8][8]
) {
    const int wg_lane = consumer_wg_tid % kWarpSize;
    const int wg_warp = consumer_wg_tid / kWarpSize;
    const int row_group_base = consumer_group * kConsumerRows + wg_warp * 16;
    float lane_min[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    int lane_idx[4] = {-1, -1, -1, -1};

    #pragma unroll
    for (int m_it = 0; m_it < 2; ++m_it) {
        const int yo = row_group_base + m_it * kWgmmaM;
        const int row0 = yo + wg_lane / 4;
        const int row1 = row0 + 8;
        const bool row0_valid = row0 < kTileM;
        const bool row1_valid = row1 < kTileM;

        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int col = w * 16 + 2 * (consumer_wg_tid % 4);
            const int global_col0 = block_col_start + col;
            const int global_col1 = global_col0 + 1;
            const int global_col8 = global_col0 + 8;
            const int global_col9 = global_col0 + 9;

            const float x0 = row0_valid ? smem.x_norm[row0] : 0.0f;
            const float x1 = row1_valid ? smem.x_norm[row1] : 0.0f;

            float v00 = FLT_MAX;
            float v01 = FLT_MAX;
            float v08 = FLT_MAX;
            float v09 = FLT_MAX;
            float v10 = FLT_MAX;
            float v11 = FLT_MAX;
            float v18 = FLT_MAX;
            float v19 = FLT_MAX;

            v00 = x0 + smem.c_norm[col + 0] - 2.0f * accum[m_it][w][0];
            v01 = x0 + smem.c_norm[col + 1] - 2.0f * accum[m_it][w][1];
            v10 = x1 + smem.c_norm[col + 0] - 2.0f * accum[m_it][w][2];
            v11 = x1 + smem.c_norm[col + 1] - 2.0f * accum[m_it][w][3];
            v08 = x0 + smem.c_norm[col + 8] - 2.0f * accum[m_it][w][4];
            v09 = x0 + smem.c_norm[col + 9] - 2.0f * accum[m_it][w][5];
            v18 = x1 + smem.c_norm[col + 8] - 2.0f * accum[m_it][w][6];
            v19 = x1 + smem.c_norm[col + 9] - 2.0f * accum[m_it][w][7];

            if (better_candidate(v00, global_col0, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v00;
                lane_idx[m_it * 2 + 0] = global_col0;
            }
            if (better_candidate(v01, global_col1, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v01;
                lane_idx[m_it * 2 + 0] = global_col1;
            }
            if (better_candidate(v08, global_col8, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v08;
                lane_idx[m_it * 2 + 0] = global_col8;
            }
            if (better_candidate(v09, global_col9, lane_min[m_it * 2 + 0], lane_idx[m_it * 2 + 0])) {
                lane_min[m_it * 2 + 0] = v09;
                lane_idx[m_it * 2 + 0] = global_col9;
            }
            if (better_candidate(v10, global_col0, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v10;
                lane_idx[m_it * 2 + 1] = global_col0;
            }
            if (better_candidate(v11, global_col1, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v11;
                lane_idx[m_it * 2 + 1] = global_col1;
            }
            if (better_candidate(v18, global_col8, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v18;
                lane_idx[m_it * 2 + 1] = global_col8;
            }
            if (better_candidate(v19, global_col9, lane_min[m_it * 2 + 1], lane_idx[m_it * 2 + 1])) {
                lane_min[m_it * 2 + 1] = v19;
                lane_idx[m_it * 2 + 1] = global_col9;
            }
        }
    }

    // This can be done outside , here we just do lane reduction , which is the only things that matters for correctness, this can be done once at the end.
    // See what i do flash-assign.cu for the Ampere architecture.
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        reduce_group_min(lane_min[slot], lane_idx[slot]);
        if ((consumer_wg_tid % 4) == 0) {
            const int local_row = row_group_base + (slot / 2) * kWgmmaM + (wg_lane / 4) + (slot % 2) * 8;
            if (local_row < kTileM) {
                const int row_offset = local_row;
                const float candidate_dist = lane_min[slot];
                const int candidate_idx = lane_idx[slot];
                if (better_candidate(candidate_dist, candidate_idx, smem.running_best_dist[local_row], smem.running_best_idx[local_row])) {
                    smem.running_best_dist[local_row] = candidate_dist;
                    smem.running_best_idx[local_row] = candidate_idx;
                }
                smem.warp_row_min[row_offset] = candidate_dist;
                smem.warp_row_idx[row_offset] = candidate_idx;
            }
        }
    }
}


// Kernel 5 structure with Kernel 7 barriers.

__global__ void flash_assign_hopper_k5_k7_v1_kernel(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
    (void)points;
    (void)centroids;
    (void)point_norms;
    (void)centroid_norms;
    (void)output_ids;
    (void)output_dists;
    (void)M;
    (void)N;
    (void)K;
    (void)tensorMapA;
    (void)tensorMapB;
    return;
#else
    extern __shared__ __align__(128) unsigned char shared_raw[];
    HopperSharedStorage& smem = *reinterpret_cast<HopperSharedStorage*>(shared_raw);
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int warpgroup = warp / kWarpsPerWarpgroup;
    const int warpgroup_tid = tid % kWarpgroupThreads;
    const int warpgroup_lane = warpgroup_tid % kWarpSize;
    const int warpgroup_warp = warpgroup_tid / kWarpSize;
    const int block_row_start = blockIdx.x * kTileM;
    if (block_row_start >= M) {
        return;
    }

    if (tid == 0) {
        for (int i = 0; i < kStageCount; ++i) {
            init_barrier(&smem.full[i], 1, 0);
            init_barrier(&smem.empty[i], kConsumerWarpgroups, 0);
        }
    }
    __syncthreads();

    for (int row = tid; row < kTileM; row += kThreadCount) {
        const int global_row = block_row_start + row;
        smem.x_norm[row] = (global_row < M) ? point_norms[global_row] : 0.0f;
        smem.running_best_dist[row] = FLT_MAX;
        smem.running_best_idx[row] = -1;
    }
    __syncthreads();

    const int k_tiles = K / kTileK;
    const int n_tiles = (N + kTileN - 1) / kTileN;

    if (warpgroup == 0) {
        warpgroup_reg_dealloc<32>();
        if (warpgroup_tid == 0) {
            int phase = 0;
            int stage = 0;
            for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    if (stage == kStageCount) {
                        stage = 0;
                        phase ^= 1;
                    }
                    wait(&smem.empty[stage], phase);
                    expect_bytes(&smem.full[stage], static_cast<uint32_t>((kTileM * kTileK + kTileN * kTileK) * sizeof(half)));
                    load_async(&smem.a_tiles[stage][0], &tensorMapA, &smem.full[stage], k_tile * kTileK, block_row_start);
                    load_async(&smem.b_tiles[stage][0], &tensorMapB, &smem.full[stage], k_tile * kTileK, n_tile * kTileN);
                    ++stage;
                }
            }
        }
    } else {
        warpgroup_reg_alloc<160>();
        const int consumer_group = warpgroup - 1;
        const int consumer_wg_tid = warpgroup_tid;
        const int consumer_global_tid = consumer_group * kWarpgroupThreads + consumer_wg_tid;
        if (warpgroup_lane == 0 && warpgroup_warp == 0) {
            #pragma unroll
            for (int i = 0; i < kStageCount; ++i) {
                arrive(&smem.empty[i], 1);
            }
        }

        int phase = 0;
        int stage = 0;
        for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
            for (int col = consumer_global_tid; col < kTileN; col += kWarpgroupThreads * kConsumerWarpgroups) {
                const int global_col = n_tile * kTileN + col;
                smem.c_norm[col] = (global_col < N) ? centroid_norms[global_col] : 0.0f;
            }

            float accum[2][8][8];
            #pragma unroll
            for (int m_it = 0; m_it < 2; ++m_it) {
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        accum[m_it][w][r] = 0.0f;
                    }
                }
            }

            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                if (stage == kStageCount) {
                    stage = 0;
                    phase ^= 1;
                }
                wait(&smem.full[stage], phase);
                warpgroup_arrive();

                #pragma unroll
                for (int m_it = 0; m_it < 2; ++m_it) {
                    const half* wgmma_a = &smem.a_tiles[stage][(consumer_group * kConsumerRows + m_it * kWgmmaM) * kPaddedTileK];
                    const half* wgmma_b = &smem.b_tiles[stage][0];
                    wgmma128<1, 1, 1, 0, 0>(accum[m_it], wgmma_a + 0 * kWgmmaK, wgmma_b + 0 * kWgmmaK);
                    wgmma128<1, 1, 1, 0, 0>(accum[m_it], wgmma_a + 1 * kWgmmaK, wgmma_b + 1 * kWgmmaK);
                    wgmma128<1, 1, 1, 0, 0>(accum[m_it], wgmma_a + 2 * kWgmmaK, wgmma_b + 2 * kWgmmaK);
                    wgmma128<1, 1, 1, 0, 0>(accum[m_it], wgmma_a + 3 * kWgmmaK, wgmma_b + 3 * kWgmmaK);
                }

                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                    arrive(&smem.empty[stage], 1);
                }
                ++stage;
            }
            consumer_compute_rows_fast(smem, consumer_group, consumer_wg_tid, n_tile * kTileN, N, accum);
            consumer_barrier();
        }
    }

    // is this synchreads necessary?
    __syncthreads();
    for (int row = tid; row < kTileM; row += kThreadCount) {
        const int global_row = block_row_start + row;
        if (global_row < M) {
            output_ids[global_row] = smem.running_best_idx[row];
            if (output_dists != nullptr) {
                output_dists[global_row] = smem.running_best_dist[row];
            }
        }
    }
#endif
}

// WGMMA256 storage and reduction helpers.

struct HopperSharedStorageWgmma256 {
    static constexpr int kTileM256 = 128;
    static constexpr int kTileN256 = 256;
    static constexpr int kConsumerRows256 = kTileM256 / kConsumerWarpgroups;
    static constexpr int kStageCountWgmma256 = 3;
    alignas(128) half a_tiles[kStageCountWgmma256][kTileM256 * kPaddedTileK];
    alignas(128) half b_tiles[kStageCountWgmma256][kTileN256 * kPaddedTileK];
    alignas(16) float x_norm[kTileM256];
    alignas(16) float c_norm[256];
    alignas(16) float running_best_dist[kTileM256];
    alignas(16) int running_best_idx[kTileM256];
    alignas(16) float warp_row_min[kConsumerWarpgroups * kConsumerRows256];
    alignas(16) int warp_row_idx[kConsumerWarpgroups * kConsumerRows256];
    alignas(8) uint64_t full[kStageCountWgmma256];
    alignas(8) uint64_t empty[kStageCountWgmma256];
};

struct HopperSharedStorageWgmma256ACache {
    static constexpr int kTileM256 = 128;
    static constexpr int kTileN256 = 256;
    static constexpr int kConsumerRows256 = kTileM256 / kConsumerWarpgroups;
    static constexpr int kStageCountWgmma256 = 3;
    static constexpr int kMaxKTiles = 4;
    alignas(128) half a_cache[kMaxKTiles][kTileM256 * kPaddedTileK];
    alignas(128) half b_tiles[kStageCountWgmma256][kTileN256 * kPaddedTileK];
    alignas(16) float x_norm[kTileM256];
    alignas(16) float c_norm[kTileN256];
    alignas(16) float running_best_dist[kTileM256];
    alignas(16) int running_best_idx[kTileM256];
    alignas(16) float warp_row_min[kConsumerWarpgroups * kConsumerRows256];
    alignas(16) int warp_row_idx[kConsumerWarpgroups * kConsumerRows256];
    alignas(8) uint64_t a_full[kMaxKTiles];
    alignas(8) uint64_t full[kStageCountWgmma256];
    alignas(8) uint64_t empty[kStageCountWgmma256];
};

template <typename Storage>
__device__ __forceinline__ void consumer_update_lane_min_wgmma256(
    Storage& smem,
    int consumer_group,
    int consumer_wg_tid,
    int block_col_start,
    int N,
    float (&accum)[16][8],
    float (&lane_min)[2],
    int (&lane_idx)[2]
) {
    const int wg_lane = consumer_wg_tid % kWarpSize;
    const int wg_warp = consumer_wg_tid / kWarpSize;
    const int row_group_base = consumer_group * Storage::kConsumerRows256 + wg_warp * 16;

    const int row0 = row_group_base + wg_lane / 4;
    const int row1 = row0 + 8;

    #pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int col = w * 16 + 2 * (consumer_wg_tid % 4);
        const int global_col0 = block_col_start + col;
        const int global_col1 = global_col0 + 1;
        const int global_col8 = global_col0 + 8;
        const int global_col9 = global_col0 + 9;

        const float x0 = smem.x_norm[row0] ;
        const float x1 = smem.x_norm[row1] ;

        float v00 = FLT_MAX;
        float v01 = FLT_MAX;
        float v08 = FLT_MAX;
        float v09 = FLT_MAX;
        float v10 = FLT_MAX;
        float v11 = FLT_MAX;
        float v18 = FLT_MAX;
        float v19 = FLT_MAX;

        v00 = x0 + smem.c_norm[col + 0] - 2.0f * accum[w][0];
        v01 = x0 + smem.c_norm[col + 1] - 2.0f * accum[w][1];
        v10 = x1 + smem.c_norm[col + 0] - 2.0f * accum[w][2];
        v11 = x1 + smem.c_norm[col + 1] - 2.0f * accum[w][3];
        v08 = x0 + smem.c_norm[col + 8] - 2.0f * accum[w][4];
        v09 = x0 + smem.c_norm[col + 9] - 2.0f * accum[w][5];
        v18 = x1 + smem.c_norm[col + 8] - 2.0f * accum[w][6];
        v19 = x1 + smem.c_norm[col + 9] - 2.0f * accum[w][7];

        if (better_candidate(v00, global_col0, lane_min[0], lane_idx[0])) {
            lane_min[0] = v00;
            lane_idx[0] = global_col0;
        }
        if (better_candidate(v01, global_col1, lane_min[0], lane_idx[0])) {
            lane_min[0] = v01;
            lane_idx[0] = global_col1;
        }
        if (better_candidate(v08, global_col8, lane_min[0], lane_idx[0])) {
            lane_min[0] = v08;
            lane_idx[0] = global_col8;
        }
        if (better_candidate(v09, global_col9, lane_min[0], lane_idx[0])) {
            lane_min[0] = v09;
            lane_idx[0] = global_col9;
        }
        if (better_candidate(v10, global_col0, lane_min[1], lane_idx[1])) {
            lane_min[1] = v10;
            lane_idx[1] = global_col0;
        }
        if (better_candidate(v11, global_col1, lane_min[1], lane_idx[1])) {
            lane_min[1] = v11;
            lane_idx[1] = global_col1;
        }
        if (better_candidate(v18, global_col8, lane_min[1], lane_idx[1])) {
            lane_min[1] = v18;
            lane_idx[1] = global_col8;
        }
        if (better_candidate(v19, global_col9, lane_min[1], lane_idx[1])) {
            lane_min[1] = v19;
            lane_idx[1] = global_col9;
        }
    }
}

template <typename Storage>
__device__ __forceinline__ void consumer_finalize_lane_min_wgmma256(
    Storage& smem,
    int consumer_group,
    int consumer_wg_tid,
    float (&lane_min)[2],
    int (&lane_idx)[2]
) {
    const int wg_lane = consumer_wg_tid % kWarpSize;
    const int wg_warp = consumer_wg_tid / kWarpSize;
    const int row_group_base = consumer_group * Storage::kConsumerRows256 + wg_warp * 16;

    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        reduce_group_min(lane_min[slot], lane_idx[slot]);
        if ((consumer_wg_tid % 4) == 0) {
            const int local_row = row_group_base + (wg_lane / 4) + slot * 8;
            if (local_row < Storage::kTileM256) {
                const float candidate_dist = lane_min[slot];
                const int candidate_idx = lane_idx[slot];
                if (better_candidate(candidate_dist, candidate_idx, smem.running_best_dist[local_row], smem.running_best_idx[local_row])) {
                    smem.running_best_dist[local_row] = candidate_dist;
                    smem.running_best_idx[local_row] = candidate_idx;
                }
                smem.warp_row_min[local_row] = candidate_dist;
                smem.warp_row_idx[local_row] = candidate_idx;
            }
        }
    }
}

// WGMMA256 kernel variants.

__global__ void flash_assign_hopper_k5_k7_wgmma256_kernel(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
    (void)points;
    (void)centroids;
    (void)point_norms;
    (void)centroid_norms;
    (void)output_ids;
    (void)output_dists;
    (void)M;
    (void)N;
    (void)K;
    (void)tensorMapA;
    (void)tensorMapB;
    return;
#else
    extern __shared__ __align__(128) unsigned char shared_raw[];
    HopperSharedStorageWgmma256& smem = *reinterpret_cast<HopperSharedStorageWgmma256*>(shared_raw);
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int warpgroup = warp / kWarpsPerWarpgroup;
    const int warpgroup_tid = tid % kWarpgroupThreads;
    const int warpgroup_lane = warpgroup_tid % kWarpSize;
    const int warpgroup_warp = warpgroup_tid / kWarpSize;
    const int block_row_start = blockIdx.x * HopperSharedStorageWgmma256::kTileM256;
    if (block_row_start >= M) {
        return;
    }

    if (tid == 0) {
        for (int i = 0; i < HopperSharedStorageWgmma256::kStageCountWgmma256; ++i) {
            init_barrier(&smem.full[i], 1, 0);
            init_barrier(&smem.empty[i], kConsumerWarpgroups, 0);
        }
    }
    __syncthreads();

    for (int row = tid; row < HopperSharedStorageWgmma256::kTileM256; row += kThreadCount) {
        const int global_row = block_row_start + row;
        smem.x_norm[row] = (global_row < M) ? point_norms[global_row] : 0.0f;
        smem.running_best_dist[row] = FLT_MAX;
        smem.running_best_idx[row] = -1;
    }
    __syncthreads();

    const int k_tiles = K / kTileK;
    const int n_tiles = (N + 255) / 256;

    if (warpgroup == 0) {
        warpgroup_reg_dealloc<32>();
        if (warpgroup_tid == 0) {
            int phase = 0;
            int stage = 0;
            for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    if (stage == HopperSharedStorageWgmma256::kStageCountWgmma256) {
                        stage = 0;
                        phase ^= 1;
                    }
                    wait(&smem.empty[stage], phase);
                    expect_bytes(&smem.full[stage], static_cast<uint32_t>((HopperSharedStorageWgmma256::kTileM256 * kTileK + HopperSharedStorageWgmma256::kTileN256 * kTileK) * sizeof(half)));
                    load_async(&smem.a_tiles[stage][0], &tensorMapA, &smem.full[stage], k_tile * kTileK, block_row_start);
                    load_async(&smem.b_tiles[stage][0], &tensorMapB, &smem.full[stage], k_tile * kTileK, n_tile * HopperSharedStorageWgmma256::kTileN256);
                    ++stage;
                }
            }
        }
    } else {
        warpgroup_reg_alloc<160>();
        const int consumer_group = warpgroup - 1;
        const int consumer_wg_tid = warpgroup_tid;
        const int consumer_global_tid = consumer_group * kWarpgroupThreads + consumer_wg_tid;
        if (warpgroup_lane == 0 && warpgroup_warp == 0) {
            #pragma unroll
            for (int i = 0; i < HopperSharedStorageWgmma256::kStageCountWgmma256; ++i) {
                arrive(&smem.empty[i], 1);
            }
        }

        int phase = 0;
        int stage = 0;
        float lane_best_dist[2] = {FLT_MAX, FLT_MAX};
        int lane_best_idx[2] = {-1, -1};
        for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
            for (int col = consumer_global_tid; col < HopperSharedStorageWgmma256::kTileN256; col += kWarpgroupThreads * kConsumerWarpgroups) {
                const int global_col = n_tile * HopperSharedStorageWgmma256::kTileN256 + col;
                smem.c_norm[col] = (global_col < N) ? centroid_norms[global_col] : 0.0f;
            }
            consumer_barrier();
            float accum[16][8];
            #pragma unroll
            for (int w = 0; w < 16; ++w) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    accum[w][r] = 0.0f;
                }
            }

            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                if (stage == HopperSharedStorageWgmma256::kStageCountWgmma256) {
                    stage = 0;
                    phase ^= 1;
                }
                wait(&smem.full[stage], phase);
                warpgroup_arrive();

                const half* wgmma_a = &smem.a_tiles[stage][consumer_group * HopperSharedStorageWgmma256::kConsumerRows256 * kPaddedTileK];
                const half* wgmma_b = &smem.b_tiles[stage][0];
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 0 * kWgmmaK, wgmma_b + 0 * kWgmmaK);
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 1 * kWgmmaK, wgmma_b + 1 * kWgmmaK);
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 2 * kWgmmaK, wgmma_b + 2 * kWgmmaK);
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 3 * kWgmmaK, wgmma_b + 3 * kWgmmaK);

                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                    arrive(&smem.empty[stage], 1);
                }
                ++stage;
            }

            consumer_update_lane_min_wgmma256(
                smem,
                consumer_group,
                consumer_wg_tid,
                n_tile * HopperSharedStorageWgmma256::kTileN256,
                N,
                accum,
                lane_best_dist,
                lane_best_idx);
            consumer_barrier();
        }
        consumer_finalize_lane_min_wgmma256(smem, consumer_group, consumer_wg_tid, lane_best_dist, lane_best_idx);
    }

    __syncthreads();
    for (int row = tid; row < HopperSharedStorageWgmma256::kTileM256; row += kThreadCount) {
        const int global_row = block_row_start + row;
        if (global_row < M) {
            output_ids[global_row] = smem.running_best_idx[row];
            if (output_dists != nullptr) {
                output_dists[global_row] = smem.running_best_dist[row];
            }
        }
    }
#endif
}

__global__ void flash_assign_hopper_k5_k7_wgmma256_persistent_kernel(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
    (void)points;
    (void)centroids;
    (void)point_norms;
    (void)centroid_norms;
    (void)output_ids;
    (void)output_dists;
    (void)M;
    (void)N;
    (void)K;
    (void)tensorMapA;
    (void)tensorMapB;
    return;
#else
    extern __shared__ __align__(128) unsigned char shared_raw[];
    HopperSharedStorageWgmma256& smem = *reinterpret_cast<HopperSharedStorageWgmma256*>(shared_raw);
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int warpgroup = warp / kWarpsPerWarpgroup;
    const int warpgroup_tid = tid % kWarpgroupThreads;
    const int warpgroup_lane = warpgroup_tid % kWarpSize;
    const int warpgroup_warp = warpgroup_tid / kWarpSize;

    const int m_tiles = (M + HopperSharedStorageWgmma256::kTileM256 - 1) / HopperSharedStorageWgmma256::kTileM256;
    const int k_tiles = K / kTileK;
    const int n_tiles = (N + HopperSharedStorageWgmma256::kTileN256 - 1) / HopperSharedStorageWgmma256::kTileN256;

    for (int m_tile = blockIdx.x; m_tile < m_tiles; m_tile += gridDim.x) {
        const int block_row_start = m_tile * HopperSharedStorageWgmma256::kTileM256;

        if (tid == 0) {
            for (int i = 0; i < HopperSharedStorageWgmma256::kStageCountWgmma256; ++i) {
                init_barrier(&smem.full[i], 1, 0);
                init_barrier(&smem.empty[i], kConsumerWarpgroups, 0);
            }
        }
        __syncthreads();

        for (int row = tid; row < HopperSharedStorageWgmma256::kTileM256; row += kThreadCount) {
            const int global_row = block_row_start + row;
            smem.x_norm[row] = (global_row < M) ? point_norms[global_row] : 0.0f;
            smem.running_best_dist[row] = FLT_MAX;
            smem.running_best_idx[row] = -1;
        }
        __syncthreads();

        if (warpgroup == 0) {
            warpgroup_reg_dealloc<32>();
            if (warpgroup_tid == 0) {
                int phase = 0;
                int stage = 0;
                for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                        if (stage == HopperSharedStorageWgmma256::kStageCountWgmma256) {
                            stage = 0;
                            phase ^= 1;
                        }
                        wait(&smem.empty[stage], phase);
                        expect_bytes(&smem.full[stage], static_cast<uint32_t>((HopperSharedStorageWgmma256::kTileM256 * kTileK + HopperSharedStorageWgmma256::kTileN256 * kTileK) * sizeof(half)));
                        load_async(&smem.a_tiles[stage][0], &tensorMapA, &smem.full[stage], k_tile * kTileK, block_row_start);
                        load_async(&smem.b_tiles[stage][0], &tensorMapB, &smem.full[stage], k_tile * kTileK, n_tile * HopperSharedStorageWgmma256::kTileN256);
                        ++stage;
                    }
                }
            }
        } else {
            warpgroup_reg_alloc<160>();
            const int consumer_group = warpgroup - 1;
            const int consumer_wg_tid = warpgroup_tid;
            const int consumer_global_tid = consumer_group * kWarpgroupThreads + consumer_wg_tid;
            if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                #pragma unroll
                for (int i = 0; i < HopperSharedStorageWgmma256::kStageCountWgmma256; ++i) {
                    arrive(&smem.empty[i], 1);
                }
            }

            int phase = 0;
            int stage = 0;
            float lane_best_dist[2] = {FLT_MAX, FLT_MAX};
            int lane_best_idx[2] = {-1, -1};
            for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                for (int col = consumer_global_tid; col < HopperSharedStorageWgmma256::kTileN256; col += kWarpgroupThreads * kConsumerWarpgroups) {
                    const int global_col = n_tile * HopperSharedStorageWgmma256::kTileN256 + col;
                    smem.c_norm[col] = (global_col < N) ? centroid_norms[global_col] : 0.0f;
                }
                consumer_barrier();
                float accum[16][8];
                #pragma unroll
                for (int w = 0; w < 16; ++w) {
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        accum[w][r] = 0.0f;
                    }
                }

                for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    if (stage == HopperSharedStorageWgmma256::kStageCountWgmma256) {
                        stage = 0;
                        phase ^= 1;
                    }
                    wait(&smem.full[stage], phase);
                    warpgroup_arrive();

                    const half* wgmma_a = &smem.a_tiles[stage][consumer_group * HopperSharedStorageWgmma256::kConsumerRows256 * kPaddedTileK];
                    const half* wgmma_b = &smem.b_tiles[stage][0];
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 0 * kWgmmaK, wgmma_b + 0 * kWgmmaK);
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 1 * kWgmmaK, wgmma_b + 1 * kWgmmaK);
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 2 * kWgmmaK, wgmma_b + 2 * kWgmmaK);
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 3 * kWgmmaK, wgmma_b + 3 * kWgmmaK);

                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                        arrive(&smem.empty[stage], 1);
                    }
                    ++stage;
                }

                consumer_update_lane_min_wgmma256(
                    smem,
                    consumer_group,
                    consumer_wg_tid,
                    n_tile * HopperSharedStorageWgmma256::kTileN256,
                    N,
                    accum,
                    lane_best_dist,
                    lane_best_idx);
                consumer_barrier();
            }
            consumer_finalize_lane_min_wgmma256(smem, consumer_group, consumer_wg_tid, lane_best_dist, lane_best_idx);
        }

        __syncthreads();
        for (int row = tid; row < HopperSharedStorageWgmma256::kTileM256; row += kThreadCount) {
            const int global_row = block_row_start + row;
            if (global_row < M) {
                output_ids[global_row] = smem.running_best_idx[row];
                if (output_dists != nullptr) {
                    output_dists[global_row] = smem.running_best_dist[row];
                }
            }
        }
        __syncthreads();
    }
#endif
}

template <int kClusterSize>
__global__ __launch_bounds__(kThreadCount) void __cluster_dims__(kClusterSize, 1, 1)
flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
    (void)points;
    (void)centroids;
    (void)point_norms;
    (void)centroid_norms;
    (void)output_ids;
    (void)output_dists;
    (void)M;
    (void)N;
    (void)K;
    (void)tensorMapA;
    (void)tensorMapB;
    return;
#else
    constexpr uint16_t kClusterMask = (1u << kClusterSize) - 1u;

    extern __shared__ __align__(128) unsigned char shared_raw[];
    HopperSharedStorageWgmma256& smem = *reinterpret_cast<HopperSharedStorageWgmma256*>(shared_raw);
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int warpgroup = warp / kWarpsPerWarpgroup;
    const int warpgroup_tid = tid % kWarpgroupThreads;
    const int warpgroup_lane = warpgroup_tid % kWarpSize;
    const int warpgroup_warp = warpgroup_tid / kWarpSize;
    const int rank = static_cast<int>(cluster_rank());
    const int cluster_base_block = blockIdx.x - rank;

    const int m_tiles = (M + HopperSharedStorageWgmma256::kTileM256 - 1) / HopperSharedStorageWgmma256::kTileM256;
    const int k_tiles = K / kTileK;
    const int n_tiles = (N + HopperSharedStorageWgmma256::kTileN256 - 1) / HopperSharedStorageWgmma256::kTileN256;

    for (int cluster_tile = cluster_base_block; cluster_tile < m_tiles; cluster_tile += gridDim.x) {
        const int m_tile = cluster_tile + rank;
        const int block_row_start = m_tile * HopperSharedStorageWgmma256::kTileM256;
        const bool tile_valid = m_tile < m_tiles;

        if (tid == 0) {
            for (int i = 0; i < HopperSharedStorageWgmma256::kStageCountWgmma256; ++i) {
                init_barrier(&smem.full[i], 1, 0);
                init_barrier(&smem.empty[i], kConsumerWarpgroups * kClusterSize, 0);
            }
        }
        __syncthreads();
        cluster_sync();

        for (int row = tid; row < HopperSharedStorageWgmma256::kTileM256; row += kThreadCount) {
            const int global_row = block_row_start + row;
            smem.x_norm[row] = (tile_valid && global_row < M) ? point_norms[global_row] : 0.0f;
            smem.running_best_dist[row] = FLT_MAX;
            smem.running_best_idx[row] = -1;
        }
        __syncthreads();

        if (warpgroup == 0) {
            warpgroup_reg_dealloc<32>();
            if (warpgroup_tid == 0) {
                int phase = 0;
                int stage = 0;
                for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                        if (stage == HopperSharedStorageWgmma256::kStageCountWgmma256) {
                            stage = 0;
                            phase ^= 1;
                        }
                        wait(&smem.empty[stage], phase);
                        expect_bytes(&smem.full[stage], static_cast<uint32_t>((HopperSharedStorageWgmma256::kTileM256 * kTileK + HopperSharedStorageWgmma256::kTileN256 * kTileK) * sizeof(half)));
                        load_async(&smem.a_tiles[stage][0], &tensorMapA, &smem.full[stage], k_tile * kTileK, block_row_start);
                        if (rank == 0) {
                            load_async_multicast(
                                &smem.b_tiles[stage][0],
                                &tensorMapB,
                                &smem.full[stage],
                                k_tile * kTileK,
                                n_tile * HopperSharedStorageWgmma256::kTileN256,
                                kClusterMask);
                        }
                        ++stage;
                    }
                }
            }
        } else {
            warpgroup_reg_alloc<160>();
            const int consumer_group = warpgroup - 1;
            const int consumer_wg_tid = warpgroup_tid;
            const int consumer_global_tid = consumer_group * kWarpgroupThreads + consumer_wg_tid;
            if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                #pragma unroll
                for (int i = 0; i < HopperSharedStorageWgmma256::kStageCountWgmma256; ++i) {
                    #pragma unroll
                    for (int cta = 0; cta < kClusterSize; ++cta) {
                        arrive_cluster(&smem.empty[i], cta, 1);
                    }
                }
            }

            int phase = 0;
            int stage = 0;
            float lane_best_dist[2] = {FLT_MAX, FLT_MAX};
            int lane_best_idx[2] = {-1, -1};
            for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                for (int col = consumer_global_tid; col < HopperSharedStorageWgmma256::kTileN256; col += kWarpgroupThreads * kConsumerWarpgroups) {
                    const int global_col = n_tile * HopperSharedStorageWgmma256::kTileN256 + col;
                    smem.c_norm[col] = (global_col < N) ? centroid_norms[global_col] : 0.0f;
                }
                consumer_barrier();
                float accum[16][8];
                #pragma unroll
                for (int w = 0; w < 16; ++w) {
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        accum[w][r] = 0.0f;
                    }
                }

                for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    if (stage == HopperSharedStorageWgmma256::kStageCountWgmma256) {
                        stage = 0;
                        phase ^= 1;
                    }
                    wait(&smem.full[stage], phase);
                    warpgroup_arrive();

                    const half* wgmma_a = &smem.a_tiles[stage][consumer_group * HopperSharedStorageWgmma256::kConsumerRows256 * kPaddedTileK];
                    const half* wgmma_b = &smem.b_tiles[stage][0];
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 0 * kWgmmaK, wgmma_b + 0 * kWgmmaK);
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 1 * kWgmmaK, wgmma_b + 1 * kWgmmaK);
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 2 * kWgmmaK, wgmma_b + 2 * kWgmmaK);
                    wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 3 * kWgmmaK, wgmma_b + 3 * kWgmmaK);

                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                        #pragma unroll
                        for (int cta = 0; cta < kClusterSize; ++cta) {
                            arrive_cluster(&smem.empty[stage], cta, 1);
                        }
                    }
                    ++stage;
                }

                consumer_update_lane_min_wgmma256(
                    smem,
                    consumer_group,
                    consumer_wg_tid,
                    n_tile * HopperSharedStorageWgmma256::kTileN256,
                    N,
                    accum,
                    lane_best_dist,
                    lane_best_idx);
                consumer_barrier();
            }
            consumer_finalize_lane_min_wgmma256(smem, consumer_group, consumer_wg_tid, lane_best_dist, lane_best_idx);
        }

        __syncthreads();
        for (int row = tid; row < HopperSharedStorageWgmma256::kTileM256; row += kThreadCount) {
            const int global_row = block_row_start + row;
            if (tile_valid && global_row < M) {
                output_ids[global_row] = smem.running_best_idx[row];
                if (output_dists != nullptr) {
                    output_dists[global_row] = smem.running_best_dist[row];
                }
            }
        }
        __syncthreads();
        cluster_sync();
    }
#endif
}

__global__ void flash_assign_hopper_k5_k7_wgmma256_acache_kernel(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
    (void)points;
    (void)centroids;
    (void)point_norms;
    (void)centroid_norms;
    (void)output_ids;
    (void)output_dists;
    (void)M;
    (void)N;
    (void)K;
    (void)tensorMapA;
    (void)tensorMapB;
    return;
#else
    extern __shared__ __align__(128) unsigned char shared_raw[];
    HopperSharedStorageWgmma256ACache& smem = *reinterpret_cast<HopperSharedStorageWgmma256ACache*>(shared_raw);
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int warpgroup = warp / kWarpsPerWarpgroup;
    const int warpgroup_tid = tid % kWarpgroupThreads;
    const int warpgroup_lane = warpgroup_tid % kWarpSize;
    const int warpgroup_warp = warpgroup_tid / kWarpSize;
    const int block_row_start = blockIdx.x * HopperSharedStorageWgmma256ACache::kTileM256;
    if (block_row_start >= M) {
        return;
    }

    const int k_tiles = K / kTileK;
    const int n_tiles = (N + HopperSharedStorageWgmma256ACache::kTileN256 - 1) / HopperSharedStorageWgmma256ACache::kTileN256;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < HopperSharedStorageWgmma256ACache::kMaxKTiles; ++i) {
            init_barrier(&smem.a_full[i], 1, 0);
        }
        #pragma unroll
        for (int i = 0; i < HopperSharedStorageWgmma256ACache::kStageCountWgmma256; ++i) {
            init_barrier(&smem.full[i], 1, 0);
            init_barrier(&smem.empty[i], kConsumerWarpgroups, 0);
        }
    }
    __syncthreads();

    for (int row = tid; row < HopperSharedStorageWgmma256ACache::kTileM256; row += kThreadCount) {
        const int global_row = block_row_start + row;
        smem.x_norm[row] = (global_row < M) ? point_norms[global_row] : 0.0f;
        smem.running_best_dist[row] = FLT_MAX;
        smem.running_best_idx[row] = -1;
    }
    __syncthreads();

    if (warpgroup == 0) {
        warpgroup_reg_dealloc<32>();
        if (warpgroup_tid == 0) {
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                expect_bytes(&smem.a_full[k_tile], static_cast<uint32_t>(HopperSharedStorageWgmma256ACache::kTileM256 * kTileK * sizeof(half)));
                load_async(&smem.a_cache[k_tile][0], &tensorMapA, &smem.a_full[k_tile], k_tile * kTileK, block_row_start);
            }

            int phase = 0;
            int stage = 0;
            for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    if (stage == HopperSharedStorageWgmma256ACache::kStageCountWgmma256) {
                        stage = 0;
                        phase ^= 1;
                    }
                    wait(&smem.empty[stage], phase);
                    expect_bytes(&smem.full[stage], static_cast<uint32_t>(HopperSharedStorageWgmma256ACache::kTileN256 * kTileK * sizeof(half)));
                    load_async(&smem.b_tiles[stage][0], &tensorMapB, &smem.full[stage], k_tile * kTileK, n_tile * HopperSharedStorageWgmma256ACache::kTileN256);
                    ++stage;
                }
            }
        }
    } else {
        warpgroup_reg_alloc<160>();
        const int consumer_group = warpgroup - 1;
        const int consumer_wg_tid = warpgroup_tid;
        const int consumer_global_tid = consumer_group * kWarpgroupThreads + consumer_wg_tid;
        if (warpgroup_lane == 0 && warpgroup_warp == 0) {
            #pragma unroll
            for (int i = 0; i < HopperSharedStorageWgmma256ACache::kStageCountWgmma256; ++i) {
                arrive(&smem.empty[i], 1);
            }
        }

        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            wait(&smem.a_full[k_tile], 0);
        }

        int phase = 0;
        int stage = 0;
        float lane_best_dist[2] = {FLT_MAX, FLT_MAX};
        int lane_best_idx[2] = {-1, -1};
        for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
            for (int col = consumer_global_tid; col < HopperSharedStorageWgmma256ACache::kTileN256; col += kWarpgroupThreads * kConsumerWarpgroups) {
                const int global_col = n_tile * HopperSharedStorageWgmma256ACache::kTileN256 + col;
                smem.c_norm[col] = (global_col < N) ? centroid_norms[global_col] : 0.0f;
            }

            float accum[16][8];
            #pragma unroll
            for (int w = 0; w < 16; ++w) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    accum[w][r] = 0.0f;
                }
            }
            consumer_barrier();

            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                if (stage == HopperSharedStorageWgmma256ACache::kStageCountWgmma256) {
                    stage = 0;
                    phase ^= 1;
                }
                wait(&smem.full[stage], phase);
                warpgroup_arrive();

                const half* wgmma_a = &smem.a_cache[k_tile][consumer_group * HopperSharedStorageWgmma256ACache::kConsumerRows256 * kPaddedTileK];
                const half* wgmma_b = &smem.b_tiles[stage][0];
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 0 * kWgmmaK, wgmma_b + 0 * kWgmmaK);
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 1 * kWgmmaK, wgmma_b + 1 * kWgmmaK);
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 2 * kWgmmaK, wgmma_b + 2 * kWgmmaK);
                wgmma256<1, 1, 1, 0, 0>(accum, wgmma_a + 3 * kWgmmaK, wgmma_b + 3 * kWgmmaK);

                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (warpgroup_lane == 0 && warpgroup_warp == 0) {
                    arrive(&smem.empty[stage], 1);
                }
                ++stage;
            }

            consumer_update_lane_min_wgmma256(
                smem,
                consumer_group,
                consumer_wg_tid,
                n_tile * HopperSharedStorageWgmma256ACache::kTileN256,
                N,
                accum,
                lane_best_dist,
                lane_best_idx);
            consumer_barrier();
        }
        consumer_finalize_lane_min_wgmma256(smem, consumer_group, consumer_wg_tid, lane_best_dist, lane_best_idx);
    }

    __syncthreads();
    for (int row = tid; row < HopperSharedStorageWgmma256ACache::kTileM256; row += kThreadCount) {
        const int global_row = block_row_start + row;
        if (global_row < M) {
            output_ids[global_row] = smem.running_best_idx[row];
            if (output_dists != nullptr) {
                output_dists[global_row] = smem.running_best_dist[row];
            }
        }
    }
#endif
}

}  // namespace

// Shared-memory sizing helpers.

size_t flash_assign_hopper_smem_bytes() {
    return sizeof(HopperSharedStorage);
}

size_t flash_assign_hopper_smem_bytes_wgmma256() {
    return sizeof(HopperSharedStorageWgmma256);
}

size_t flash_assign_hopper_smem_bytes_wgmma256_acache() {
    return sizeof(HopperSharedStorageWgmma256ACache);
}

// Public launchers.

cudaError_t launch_point_l2_norm_kernel_hopper(
    const half* points,
    float* point_norms,
    int num_points,
    int dim,
    cudaStream_t stream
) {
    row_l2_norm_kernel_hopper<<<num_points, kThreadCount, 0, stream>>>(points, point_norms, num_points, dim);
    return cudaGetLastError();
}

cudaError_t launch_centroid_l2_norm_kernel_hopper(
    const half* centroids,
    float* centroid_norms,
    int num_centroids,
    int dim,
    cudaStream_t stream
) {
    row_l2_norm_kernel_hopper<<<num_centroids, kThreadCount, 0, stream>>>(centroids, centroid_norms, num_centroids, dim);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_k5_k7_v1(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9) {
        return cudaErrorNotSupported;
    }
    if ((K % kTileK) != 0) {
        return cudaErrorInvalidValue;
    }

    const CUtensorMap tensor_map_a = create_tensor_map<kTileM, kTileK>(points, M, K);
    const CUtensorMap tensor_map_b = create_tensor_map<kTileN, kTileK>(centroids, N, K);
    const dim3 block(kThreadCount);
    const dim3 grid((M + kTileM - 1) / kTileM);
    const size_t smem_bytes = flash_assign_hopper_smem_bytes();

    err = cudaFuncSetAttribute(
        flash_assign_hopper_k5_k7_v1_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_hopper_k5_k7_v1_kernel<<<grid, block, smem_bytes, stream>>>(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        M,
        N,
        K,
        tensor_map_a,
        tensor_map_b);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_complete_k5_k7_v1(
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
    cudaError_t err = launch_point_l2_norm_kernel_hopper(points, point_norms, num_points, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = launch_centroid_l2_norm_kernel_hopper(centroids, centroid_norms, num_centroids, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return launch_flash_assign_hopper_k5_k7_v1(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream);
}

cudaError_t launch_flash_assign_hopper_k5_k7_wgmma256(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9) {
        return cudaErrorNotSupported;
    }
    if ((K % kTileK) != 0) {
        return cudaErrorInvalidValue;
    }

    const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
    const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
    const dim3 block(kThreadCount);
    const dim3 grid((M + HopperSharedStorageWgmma256::kTileM256 - 1) / HopperSharedStorageWgmma256::kTileM256);
    const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();

    err = cudaFuncSetAttribute(
        flash_assign_hopper_k5_k7_wgmma256_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_hopper_k5_k7_wgmma256_kernel<<<grid, block, smem_bytes, stream>>>(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        M,
        N,
        K,
        tensor_map_a,
        tensor_map_b);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_complete_k5_k7_wgmma256(
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
    cudaError_t err = launch_point_l2_norm_kernel_hopper(points, point_norms, num_points, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = launch_centroid_l2_norm_kernel_hopper(centroids, centroid_norms, num_centroids, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return launch_flash_assign_hopper_k5_k7_wgmma256(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream);
}

cudaError_t launch_flash_assign_hopper_k5_k7_wgmma256_persistent(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9) {
        return cudaErrorNotSupported;
    }
    if ((K % kTileK) != 0) {
        return cudaErrorInvalidValue;
    }

    const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
    const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
    const dim3 block(kThreadCount);
    const dim3 grid(props.multiProcessorCount);
    const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();

    err = cudaFuncSetAttribute(
        flash_assign_hopper_k5_k7_wgmma256_persistent_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_hopper_k5_k7_wgmma256_persistent_kernel<<<grid, block, smem_bytes, stream>>>(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        M,
        N,
        K,
        tensor_map_a,
        tensor_map_b);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_complete_k5_k7_wgmma256_persistent(
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
    cudaError_t err = launch_point_l2_norm_kernel_hopper(points, point_norms, num_points, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = launch_centroid_l2_norm_kernel_hopper(centroids, centroid_norms, num_centroids, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return launch_flash_assign_hopper_k5_k7_wgmma256_persistent(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream);
}

cudaError_t launch_flash_assign_hopper_k5_k7_wgmma256_persistent_cluster4(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9) {
        return cudaErrorNotSupported;
    }
    if ((K % kTileK) != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kClusterSize = 4;
    const int cluster_count = props.multiProcessorCount / kClusterSize;
    if (cluster_count <= 0) {
        return cudaErrorInvalidValue;
    }

    const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
    const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
    const dim3 block(kThreadCount);
    const dim3 grid(cluster_count * kClusterSize);
    const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();

    err = cudaFuncSetAttribute(
        flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<4><<<grid, block, smem_bytes, stream>>>(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        M,
        N,
        K,
        tensor_map_a,
        tensor_map_b);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_complete_k5_k7_wgmma256_persistent_cluster4(
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
    cudaError_t err = launch_point_l2_norm_kernel_hopper(points, point_norms, num_points, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = launch_centroid_l2_norm_kernel_hopper(centroids, centroid_norms, num_centroids, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return launch_flash_assign_hopper_k5_k7_wgmma256_persistent_cluster4(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream);
}

cudaError_t launch_flash_assign_hopper_k5_k7_wgmma256_persistent_cluster8(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9) {
        return cudaErrorNotSupported;
    }
    if ((K % kTileK) != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kClusterSize = 8;
    const int cluster_count = props.multiProcessorCount / kClusterSize;
    if (cluster_count <= 0) {
        return cudaErrorInvalidValue;
    }

    const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
    const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
    const dim3 block(kThreadCount);
    const dim3 grid(cluster_count * kClusterSize);
    const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();

    err = cudaFuncSetAttribute(
        flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<8><<<grid, block, smem_bytes, stream>>>(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        M,
        N,
        K,
        tensor_map_a,
        tensor_map_b);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_complete_k5_k7_wgmma256_persistent_cluster8(
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
    cudaError_t err = launch_point_l2_norm_kernel_hopper(points, point_norms, num_points, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = launch_centroid_l2_norm_kernel_hopper(centroids, centroid_norms, num_centroids, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return launch_flash_assign_hopper_k5_k7_wgmma256_persistent_cluster8(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream);
}

cudaError_t launch_flash_assign_hopper_k5_k7_wgmma256_acache(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9) {
        return cudaErrorNotSupported;
    }
    if ((K % kTileK) != 0 || K > HopperSharedStorageWgmma256ACache::kMaxKTiles * kTileK) {
        return cudaErrorInvalidValue;
    }

    const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256ACache::kTileM256, kTileK>(points, M, K);
    const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256ACache::kTileN256, kTileK>(centroids, N, K);
    const dim3 block(kThreadCount);
    const dim3 grid((M + HopperSharedStorageWgmma256ACache::kTileM256 - 1) / HopperSharedStorageWgmma256ACache::kTileM256);
    const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256_acache();

    err = cudaFuncSetAttribute(
        flash_assign_hopper_k5_k7_wgmma256_acache_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return err;
    }

    flash_assign_hopper_k5_k7_wgmma256_acache_kernel<<<grid, block, smem_bytes, stream>>>(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        M,
        N,
        K,
        tensor_map_a,
        tensor_map_b);
    return cudaGetLastError();
}

cudaError_t launch_flash_assign_hopper_complete_k5_k7_wgmma256_acache(
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
    cudaError_t err = launch_point_l2_norm_kernel_hopper(points, point_norms, num_points, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = launch_centroid_l2_norm_kernel_hopper(centroids, centroid_norms, num_centroids, dim, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return launch_flash_assign_hopper_k5_k7_wgmma256_acache(
        points,
        centroids,
        point_norms,
        centroid_norms,
        output_ids,
        output_dists,
        num_points,
        num_centroids,
        dim,
        stream);
}

// Benchmark entrypoint.

cudaError_t benchmark_flash_assign_hopper_precomputed(
    const half* points,
    const half* centroids,
    const float* point_norms,
    const float* centroid_norms,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    int variant,
    int iters,
    float* elapsed_ms,
    cudaStream_t stream
) {
    if (iters <= 0 || elapsed_ms == nullptr) {
        return cudaErrorInvalidValue;
    }

    int device_index = -1;
    cudaError_t err = cudaGetDevice(&device_index);
    if (err != cudaSuccess) {
        return err;
    }

    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device_index);
    if (err != cudaSuccess) {
        return err;
    }
    if (props.major < 9 || (K % kTileK) != 0) {
        return cudaErrorInvalidValue;
    }

    cudaEvent_t start{};
    cudaEvent_t end{};
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaEventCreate(&end);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        return err;
    }

    const dim3 block(kThreadCount);
    if (variant == 0) {
        const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
        const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
        const dim3 grid((M + HopperSharedStorageWgmma256::kTileM256 - 1) / HopperSharedStorageWgmma256::kTileM256);
        const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();
        err = cudaFuncSetAttribute(
            flash_assign_hopper_k5_k7_wgmma256_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        for (int i = 0; i < iters; ++i) {
            flash_assign_hopper_k5_k7_wgmma256_kernel<<<grid, block, smem_bytes, stream>>>(
                points,
                centroids,
                point_norms,
                centroid_norms,
                output_ids,
                output_dists,
                M,
                N,
                K,
                tensor_map_a,
                tensor_map_b);
        }
    } else if (variant == 1) {
        if (K > HopperSharedStorageWgmma256ACache::kMaxKTiles * kTileK) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return cudaErrorInvalidValue;
        }
        const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256ACache::kTileM256, kTileK>(points, M, K);
        const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256ACache::kTileN256, kTileK>(centroids, N, K);
        const dim3 grid((M + HopperSharedStorageWgmma256ACache::kTileM256 - 1) / HopperSharedStorageWgmma256ACache::kTileM256);
        const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256_acache();
        err = cudaFuncSetAttribute(
            flash_assign_hopper_k5_k7_wgmma256_acache_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        for (int i = 0; i < iters; ++i) {
            flash_assign_hopper_k5_k7_wgmma256_acache_kernel<<<grid, block, smem_bytes, stream>>>(
                points,
                centroids,
                point_norms,
                centroid_norms,
                output_ids,
                output_dists,
                M,
                N,
                K,
                tensor_map_a,
                tensor_map_b);
        }
    } else if (variant == 2) {
        const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
        const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
        const dim3 grid(props.multiProcessorCount);
        const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();
        err = cudaFuncSetAttribute(
            flash_assign_hopper_k5_k7_wgmma256_persistent_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        for (int i = 0; i < iters; ++i) {
            flash_assign_hopper_k5_k7_wgmma256_persistent_kernel<<<grid, block, smem_bytes, stream>>>(
                points,
                centroids,
                point_norms,
                centroid_norms,
                output_ids,
                output_dists,
                M,
                N,
                K,
                tensor_map_a,
                tensor_map_b);
        }
    } else if (variant == 3) {
        constexpr int kClusterSize = 4;
        const int cluster_count = props.multiProcessorCount / kClusterSize;
        if (cluster_count <= 0) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return cudaErrorInvalidValue;
        }
        const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
        const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
        const dim3 grid(cluster_count * kClusterSize);
        const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();
        err = cudaFuncSetAttribute(
            flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        for (int i = 0; i < iters; ++i) {
            flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<4><<<grid, block, smem_bytes, stream>>>(
                points,
                centroids,
                point_norms,
                centroid_norms,
                output_ids,
                output_dists,
                M,
                N,
                K,
                tensor_map_a,
                tensor_map_b);
        }
    } else if (variant == 4) {
        constexpr int kClusterSize = 8;
        const int cluster_count = props.multiProcessorCount / kClusterSize;
        if (cluster_count <= 0) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return cudaErrorInvalidValue;
        }
        const CUtensorMap tensor_map_a = create_tensor_map<HopperSharedStorageWgmma256::kTileM256, kTileK>(points, M, K);
        const CUtensorMap tensor_map_b = create_tensor_map<HopperSharedStorageWgmma256::kTileN256, kTileK>(centroids, N, K);
        const dim3 grid(cluster_count * kClusterSize);
        const size_t smem_bytes = flash_assign_hopper_smem_bytes_wgmma256();
        err = cudaFuncSetAttribute(
            flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return err;
        }
        for (int i = 0; i < iters; ++i) {
            flash_assign_hopper_k5_k7_wgmma256_persistent_cluster_kernel<8><<<grid, block, smem_bytes, stream>>>(
                points,
                centroids,
                point_norms,
                centroid_norms,
                output_ids,
                output_dists,
                M,
                N,
                K,
                tensor_map_a,
                tensor_map_b);
        }
    } else {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return cudaErrorInvalidValue;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return err;
    }
    err = cudaEventRecord(end, stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return err;
    }
    err = cudaEventSynchronize(end);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return err;
    }

    float total_ms = 0.0f;
    err = cudaEventElapsedTime(&total_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    if (err != cudaSuccess) {
        return err;
    }
    *elapsed_ms = total_ms / static_cast<float>(iters);
    return cudaSuccess;
}

}  // namespace flash_kmeans::hopper
