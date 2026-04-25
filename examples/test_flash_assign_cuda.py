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
    csrc = repo_root / "flash_kmeans" / "csrc" / "ampere"
    return load(
        name="flash_assign_cuda_ext",
        sources=[
            str(csrc / "flash_assign_bind.cpp"),
            str(csrc / "flash_assign.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def _run_case(ext, num_points: int, num_centroids: int, dim: int, seed: int) -> None:
    torch.manual_seed(seed)
    points = torch.randn(num_points, dim, device="cuda", dtype=torch.float16).contiguous()
    centroids = torch.randn(num_centroids, dim, device="cuda", dtype=torch.float16).contiguous()

    ids_cuda, dists_cuda, _, _ = ext.flash_assign_cuda(points, centroids)

    x = points.unsqueeze(0)
    c = centroids.unsqueeze(0)
    x_sq = (points.float() ** 2).sum(dim=-1).unsqueeze(0)
    ids_triton = euclid_assign_triton(x, c, x_sq, use_heuristic=True).squeeze(0)

    torch.cuda.synchronize()

    mismatch = (ids_cuda.cpu() != ids_triton.cpu()).sum().item()
    if mismatch != 0:
        mismatch_idx = int((ids_cuda != ids_triton).nonzero(as_tuple=False)[0].item())
        raise AssertionError(
            f"Mismatch for shape (M={num_points}, N={num_centroids}, K={dim}). "
            f"{mismatch} ids differ. First mismatch at row {mismatch_idx}: "
            f"cuda={int(ids_cuda[mismatch_idx])}, triton={int(ids_triton[mismatch_idx])}"
        )

    print(
        f"PASS shape=(M={num_points}, N={num_centroids}, K={dim}) "
        f"ids_match={num_points} dists_dtype={dists_cuda.dtype}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compile and correctness-check flash_assign.cu")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["256,128,128", "300,145,128", "512,256,256"],
        help="Cases as M,N,K triplets",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ext = _load_extension()

    for case in args.cases:
        m_str, n_str, k_str = case.split(",")
        _run_case(ext, int(m_str), int(n_str), int(k_str), args.seed)


if __name__ == "__main__":
    main()
