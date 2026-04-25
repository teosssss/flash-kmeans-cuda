#pragma once

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
