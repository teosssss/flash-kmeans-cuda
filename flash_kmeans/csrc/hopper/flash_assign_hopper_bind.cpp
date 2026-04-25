#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace flash_kmeans::hopper {

cudaError_t launch_point_l2_norm_kernel_hopper(
    const half* points,
    float* point_norms,
    int num_points,
    int dim,
    cudaStream_t stream
);

cudaError_t launch_centroid_l2_norm_kernel_hopper(
    const half* centroids,
    float* centroid_norms,
    int num_centroids,
    int dim,
    cudaStream_t stream
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

}  // namespace flash_kmeans::hopper

namespace {

void check_common_inputs(const torch::Tensor& points, const torch::Tensor& centroids, const std::string& kernel_name) {
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
    TORCH_CHECK(points.dtype() == torch::kFloat16, "points must be float16");
    TORCH_CHECK(centroids.dtype() == torch::kFloat16, "centroids must be float16");
    TORCH_CHECK(points.dim() == 2, "points must have shape [num_points, dim]");
    TORCH_CHECK(centroids.dim() == 2, "centroids must have shape [num_centroids, dim]");
    TORCH_CHECK(points.size(1) == centroids.size(1), "points and centroids must have the same dim");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
    TORCH_CHECK(
        kernel_name == "hopper_k5_k7_v1" ||
            kernel_name == "hopper_k5_k7_wgmma256" ||
            kernel_name == "hopper_k5_k7_wgmma256_acache" ||
            kernel_name == "hopper_k5_k7_wgmma256_persistent" ||
            kernel_name == "hopper_k5_k7_wgmma256_persistent_cluster4" ||
            kernel_name == "hopper_k5_k7_wgmma256_persistent_cluster8",
        "Unknown Hopper kernel_name: ",
        kernel_name);
}

cudaError_t launch_assign_by_name(
    const torch::Tensor& points,
    const torch::Tensor& centroids,
    const torch::Tensor& point_norms,
    const torch::Tensor& centroid_norms,
    torch::Tensor& output_ids,
    torch::Tensor& output_dists,
    const std::string& kernel_name,
    cudaStream_t stream
) {
    const auto num_points = static_cast<int>(points.size(0));
    const auto num_centroids = static_cast<int>(centroids.size(0));
    const auto dim = static_cast<int>(points.size(1));

    if (kernel_name == "hopper_k5_k7_v1") {
        return flash_kmeans::hopper::launch_flash_assign_hopper_k5_k7_v1(
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
    }
    if (kernel_name == "hopper_k5_k7_wgmma256") {
        return flash_kmeans::hopper::launch_flash_assign_hopper_k5_k7_wgmma256(
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
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_persistent") {
        return flash_kmeans::hopper::launch_flash_assign_hopper_k5_k7_wgmma256_persistent(
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
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_persistent_cluster4") {
        return flash_kmeans::hopper::launch_flash_assign_hopper_k5_k7_wgmma256_persistent_cluster4(
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
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_persistent_cluster8") {
        return flash_kmeans::hopper::launch_flash_assign_hopper_k5_k7_wgmma256_persistent_cluster8(
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
    }
    return flash_kmeans::hopper::launch_flash_assign_hopper_k5_k7_wgmma256_acache(
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
}

}  // namespace

int hopper_kernel_variant(const std::string& kernel_name) {
    if (kernel_name == "hopper_k5_k7_wgmma256") {
        return 0;
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_acache") {
        return 1;
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_persistent") {
        return 2;
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_persistent_cluster4") {
        return 3;
    }
    if (kernel_name == "hopper_k5_k7_wgmma256_persistent_cluster8") {
        return 4;
    }
    return -1;
}

std::vector<torch::Tensor> flash_assign_hopper_cuda(
    torch::Tensor points,
    torch::Tensor centroids,
    std::string kernel_name
) {
    check_common_inputs(points, centroids, kernel_name);

    const auto num_points = static_cast<int>(points.size(0));
    const auto num_centroids = static_cast<int>(centroids.size(0));
    const auto dim = static_cast<int>(points.size(1));

    auto point_norms = torch::empty({num_points}, points.options().dtype(torch::kFloat32));
    auto centroid_norms = torch::empty({num_centroids}, centroids.options().dtype(torch::kFloat32));
    auto output_ids = torch::empty({num_points}, points.options().dtype(torch::kInt32));
    auto output_dists = torch::empty({num_points}, points.options().dtype(torch::kFloat32));

    const c10::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(points.device().index()).stream();

    cudaError_t err = flash_kmeans::hopper::launch_point_l2_norm_kernel_hopper(
        reinterpret_cast<const half*>(points.data_ptr<at::Half>()),
        point_norms.data_ptr<float>(),
        num_points,
        dim,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "launch_point_l2_norm_kernel_hopper failed: ", cudaGetErrorString(err));

    err = flash_kmeans::hopper::launch_centroid_l2_norm_kernel_hopper(
        reinterpret_cast<const half*>(centroids.data_ptr<at::Half>()),
        centroid_norms.data_ptr<float>(),
        num_centroids,
        dim,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "launch_centroid_l2_norm_kernel_hopper failed: ", cudaGetErrorString(err));

    err = launch_assign_by_name(points, centroids, point_norms, centroid_norms, output_ids, output_dists, kernel_name, stream);
    TORCH_CHECK(err == cudaSuccess, "launch assign failed for ", kernel_name, ": ", cudaGetErrorString(err));

    return {output_ids, output_dists, point_norms, centroid_norms};
}

void flash_assign_hopper_precomputed_cuda(
    torch::Tensor points,
    torch::Tensor centroids,
    torch::Tensor point_norms,
    torch::Tensor centroid_norms,
    torch::Tensor output_ids,
    torch::Tensor output_dists,
    std::string kernel_name
) {
    check_common_inputs(points, centroids, kernel_name);
    TORCH_CHECK(point_norms.is_cuda() && point_norms.dtype() == torch::kFloat32, "point_norms must be CUDA float32");
    TORCH_CHECK(centroid_norms.is_cuda() && centroid_norms.dtype() == torch::kFloat32, "centroid_norms must be CUDA float32");
    TORCH_CHECK(output_ids.is_cuda() && output_ids.dtype() == torch::kInt32, "output_ids must be CUDA int32");
    TORCH_CHECK(output_dists.is_cuda() && output_dists.dtype() == torch::kFloat32, "output_dists must be CUDA float32");
    TORCH_CHECK(point_norms.is_contiguous(), "point_norms must be contiguous");
    TORCH_CHECK(centroid_norms.is_contiguous(), "centroid_norms must be contiguous");
    TORCH_CHECK(output_ids.is_contiguous(), "output_ids must be contiguous");
    TORCH_CHECK(output_dists.is_contiguous(), "output_dists must be contiguous");
    TORCH_CHECK(point_norms.numel() == points.size(0), "point_norms shape mismatch");
    TORCH_CHECK(centroid_norms.numel() == centroids.size(0), "centroid_norms shape mismatch");
    TORCH_CHECK(output_ids.numel() == points.size(0), "output_ids shape mismatch");
    TORCH_CHECK(output_dists.numel() == points.size(0), "output_dists shape mismatch");

    const c10::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(points.device().index()).stream();
    cudaError_t err = launch_assign_by_name(points, centroids, point_norms, centroid_norms, output_ids, output_dists, kernel_name, stream);
    TORCH_CHECK(err == cudaSuccess, "launch precomputed assign failed for ", kernel_name, ": ", cudaGetErrorString(err));
}

double flash_assign_hopper_bench_precomputed_cuda(
    torch::Tensor points,
    torch::Tensor centroids,
    torch::Tensor point_norms,
    torch::Tensor centroid_norms,
    torch::Tensor output_ids,
    torch::Tensor output_dists,
    std::string kernel_name,
    int iters
) {
    check_common_inputs(points, centroids, kernel_name);
    TORCH_CHECK(point_norms.is_cuda() && point_norms.dtype() == torch::kFloat32, "point_norms must be CUDA float32");
    TORCH_CHECK(centroid_norms.is_cuda() && centroid_norms.dtype() == torch::kFloat32, "centroid_norms must be CUDA float32");
    TORCH_CHECK(output_ids.is_cuda() && output_ids.dtype() == torch::kInt32, "output_ids must be CUDA int32");
    TORCH_CHECK(output_dists.is_cuda() && output_dists.dtype() == torch::kFloat32, "output_dists must be CUDA float32");
    TORCH_CHECK(point_norms.is_contiguous(), "point_norms must be contiguous");
    TORCH_CHECK(centroid_norms.is_contiguous(), "centroid_norms must be contiguous");
    TORCH_CHECK(output_ids.is_contiguous(), "output_ids must be contiguous");
    TORCH_CHECK(output_dists.is_contiguous(), "output_dists must be contiguous");
    TORCH_CHECK(point_norms.numel() == points.size(0), "point_norms shape mismatch");
    TORCH_CHECK(centroid_norms.numel() == centroids.size(0), "centroid_norms shape mismatch");
    TORCH_CHECK(output_ids.numel() == points.size(0), "output_ids shape mismatch");
    TORCH_CHECK(output_dists.numel() == points.size(0), "output_dists shape mismatch");
    TORCH_CHECK(iters > 0, "iters must be positive");

    const int variant = hopper_kernel_variant(kernel_name);
    TORCH_CHECK(variant >= 0, "C++ precomputed benchmark supports only wgmma256 variants, got: ", kernel_name);

    const c10::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(points.device().index()).stream();
    float elapsed_ms = 0.0f;
    cudaError_t err = flash_kmeans::hopper::benchmark_flash_assign_hopper_precomputed(
        reinterpret_cast<const half*>(points.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(centroids.data_ptr<at::Half>()),
        point_norms.data_ptr<float>(),
        centroid_norms.data_ptr<float>(),
        output_ids.data_ptr<int>(),
        output_dists.data_ptr<float>(),
        static_cast<int>(points.size(0)),
        static_cast<int>(centroids.size(0)),
        static_cast<int>(points.size(1)),
        variant,
        iters,
        &elapsed_ms,
        stream);
    TORCH_CHECK(err == cudaSuccess, "C++ precomputed benchmark failed for ", kernel_name, ": ", cudaGetErrorString(err));
    return static_cast<double>(elapsed_ms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_assign_hopper_cuda", &flash_assign_hopper_cuda, "Flash assign Hopper kernels");
    m.def("flash_assign_hopper_precomputed_cuda", &flash_assign_hopper_precomputed_cuda, "Flash assign Hopper kernels with precomputed norms and outputs");
    m.def("flash_assign_hopper_bench_precomputed_cuda", &flash_assign_hopper_bench_precomputed_cuda, "Benchmark Hopper assign kernels with precomputed norms and outputs");
}
