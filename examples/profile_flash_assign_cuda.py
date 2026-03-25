from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _load_extension():
    repo_root = Path(__file__).resolve().parents[1]
    csrc = repo_root / "flash_kmeans" / "csrc"
    return load(
        name="flash_assign_cuda_ext_profile",
        sources=[
            str(csrc / "flash_assign_bind.cpp"),
            str(csrc / "flash_assign.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Profile flash assign CUDA kernel stack")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-file", type=str, default="flash_assign_trace.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    ext = _load_extension()

    points = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16).contiguous()
    centroids = torch.randn(args.n, args.k, device="cuda", dtype=torch.float16).contiguous()

    def run_once():
        return ext.flash_assign_cuda(points, centroids)

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.iters):
            run_once()
        torch.cuda.synchronize()

    print("PROFILE_TABLE_START")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
            max_name_column_width=80,
        )
    )
    print("PROFILE_TABLE_END")

    prof.export_chrome_trace(args.trace_file)
    print(f"TRACE_FILE {args.trace_file}")


if __name__ == "__main__":
    main()
