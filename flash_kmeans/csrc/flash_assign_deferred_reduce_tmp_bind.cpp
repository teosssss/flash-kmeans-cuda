#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

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
);

std::vector<torch::Tensor> flash_assign_deferred_reduce_tmp_cuda(torch::Tensor points, torch::Tensor centroids) {
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
    TORCH_CHECK(points.dtype() == torch::kFloat16, "points must be float16");
    TORCH_CHECK(centroids.dtype() == torch::kFloat16, "centroids must be float16");
    TORCH_CHECK(points.dim() == 2, "points must have shape [num_points, dim]");
    TORCH_CHECK(centroids.dim() == 2, "centroids must have shape [num_centroids, dim]");
    TORCH_CHECK(points.size(1) == centroids.size(1), "points and centroids must have the same dim");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");

    const auto num_points = static_cast<int>(points.size(0));
    const auto num_centroids = static_cast<int>(centroids.size(0));
    const auto dim = static_cast<int>(points.size(1));

    auto point_norms = torch::empty({num_points}, points.options().dtype(torch::kFloat32));
    auto centroid_norms = torch::empty({num_centroids}, centroids.options().dtype(torch::kFloat32));
    auto output_ids = torch::empty({num_points}, points.options().dtype(torch::kInt32));
    auto output_dists = torch::empty({num_points}, points.options().dtype(torch::kFloat32));

    const c10::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(points.device().index()).stream();

    cudaError_t err = launch_flash_assign_complete_256x128x32_aligned_deferred_reduce(
        reinterpret_cast<const half*>(points.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(centroids.data_ptr<at::Half>()),
        point_norms.data_ptr<float>(),
        centroid_norms.data_ptr<float>(),
        output_ids.data_ptr<int>(),
        output_dists.data_ptr<float>(),
        num_points,
        num_centroids,
        dim,
        stream
    );
    TORCH_CHECK(
        err == cudaSuccess,
        "launch_flash_assign_complete_256x128x32_aligned_deferred_reduce failed: ",
        cudaGetErrorString(err)
    );

    return {output_ids, output_dists, point_norms, centroid_norms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_assign_deferred_reduce_tmp_cuda", &flash_assign_deferred_reduce_tmp_cuda,
          "Flash assign CUDA deferred-reduce tmp");
}
