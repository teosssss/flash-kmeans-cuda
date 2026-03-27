from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from flash_kmeans.assign_euclid_triton import euclid_assign_triton


def _load_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    repo_root = Path(__file__).resolve().parents[1]
    csrc = repo_root / "flash_kmeans" / "csrc"
    return load(
        name="flash_assign_deferred_reduce_tmp_ext_bench",
        sources=[
            str(csrc / "flash_assign_deferred_reduce_tmp_bind.cpp"),
            str(csrc / "flash_assign.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench_case(ext, num_points: int, num_centroids: int, dim: int, warmup: int, iters: int, seed: int) -> None:
    torch.manual_seed(seed)
    points = torch.randn(num_points, dim, device="cuda", dtype=torch.float16).contiguous()
    centroids = torch.randn(num_centroids, dim, device="cuda", dtype=torch.float16).contiguous()

    x = points.unsqueeze(0)
    c = centroids.unsqueeze(0)
    x_sq = (points.float() ** 2).sum(dim=-1).unsqueeze(0)
    c_sq = (centroids.float() ** 2).sum(dim=-1).unsqueeze(0)

    def run_cuda():
        return ext.flash_assign_deferred_reduce_tmp_cuda(points, centroids)

    def run_triton():
        return euclid_assign_triton(x, c, x_sq, c_sq=c_sq, use_heuristic=True)

    cuda_ms = _time_ms(run_cuda, warmup=warmup, iters=iters)
    triton_ms = _time_ms(run_triton, warmup=warmup, iters=iters)
    speedup = triton_ms / cuda_ms

    ids_cuda, _, _, _ = run_cuda()
    ids_triton = run_triton().squeeze(0)
    mismatch = int((ids_cuda != ids_triton).sum().item())

    print(
        f"BENCH_DEFERRED shape=(M={num_points}, N={num_centroids}, K={dim}) "
        f"cuda_ms={cuda_ms:.3f} triton_ms={triton_ms:.3f} speedup={speedup:.3f}x "
        f"mismatch={mismatch}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark flash_assign deferred-reduce CUDA vs Triton")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["4096,1024,256", "262144,8192,256"],
        help="Cases as M,N,K triplets",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ext = _load_extension()
    for case in args.cases:
        m_str, n_str, k_str = case.split(",")
        _bench_case(ext, int(m_str), int(n_str), int(k_str), args.warmup, args.iters, args.seed)


if __name__ == "__main__":
    main()
