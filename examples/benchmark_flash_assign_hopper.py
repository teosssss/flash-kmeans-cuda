import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

import torch
from torch.utils.cpp_extension import load

from flash_kmeans.assign_euclid_triton import euclid_assign_triton


ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "flash_kmeans" / "csrc" / "hopper"
KERNELS = [
    "hopper_k5_k7_v1",
    "hopper_k5_k7_wgmma256",
    "hopper_k5_k7_wgmma256_acache",
    "hopper_k5_k7_wgmma256_persistent",
    "hopper_k5_k7_wgmma256_persistent_cluster4",
    "hopper_k5_k7_wgmma256_persistent_cluster8",
]

SHAPES = [
    (4096, 1024, 128),
    (8192, 2048, 128),
    (32768, 4096, 128),
    (4096, 1024, 256),
    (8192, 2048, 256),
    (32768, 4096, 256),
    (4096, 1024, 512),
    (8192, 2048, 512),
    (32768, 4096, 512),
]


def parse_shape(shape):
    try:
        parts = tuple(int(x) for x in shape.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {shape!r}, expected M,N,K") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"invalid shape {shape!r}, expected M,N,K")
    return parts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", type=parse_shape, default=SHAPES)
    parser.add_argument("--kernels", nargs="*", default=KERNELS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=("assign-only", "full", "both"), default="assign-only")
    return parser.parse_args()


def load_ext():
    return load(
        name="flash_assign_hopper_bench_ext",
        sources=[
            str(CSRC / "flash_assign_hopper_bind.cpp"),
            str(CSRC / "flash_assign_hopper.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def bench_ms(fn, warmup=5, iters=20):
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


def bench_wall_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def triton_ref(points, centroids):
    x = points.unsqueeze(0)
    c = centroids.unsqueeze(0)
    x_sq = (x.float() * x.float()).sum(dim=-1)
    c_sq = (c.float() * c.float()).sum(dim=-1)
    return euclid_assign_triton(x, c, x_sq, c_sq=c_sq).squeeze(0)


def make_triton_assign_only(points, centroids):
    x = points.unsqueeze(0)
    c = centroids.unsqueeze(0)
    x_sq = (x.float() * x.float()).sum(dim=-1)
    c_sq = (c.float() * c.float()).sum(dim=-1)
    out = torch.empty((1, points.shape[0]), device=points.device, dtype=torch.int32)

    def run():
        return euclid_assign_triton(x, c, x_sq, out=out, c_sq=c_sq)

    return run, out.squeeze(0)


def make_hopper_assign_only(ext, points, centroids, kernel_name):
    point_norms = (points.float() * points.float()).sum(dim=-1).contiguous()
    centroid_norms = (centroids.float() * centroids.float()).sum(dim=-1).contiguous()
    output_ids = torch.empty((points.shape[0],), device=points.device, dtype=torch.int32)
    output_dists = torch.empty((points.shape[0],), device=points.device, dtype=torch.float32)

    def run():
        ext.flash_assign_hopper_precomputed_cuda(
            points,
            centroids,
            point_norms,
            centroid_norms,
            output_ids,
            output_dists,
            kernel_name,
        )
        return output_ids

    return run, output_ids, output_dists, point_norms, centroid_norms


def make_inputs(M, N, K, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    points = torch.randn((M, K), generator=gen, device="cuda", dtype=torch.float16).contiguous()
    centroids = torch.randn((N, K), generator=gen, device="cuda", dtype=torch.float16).contiguous()
    return points, centroids


def main():
    args = parse_args()
    ext = load_ext()

    for shape_idx, (M, N, K) in enumerate(args.cases):
        seed = 1234 + shape_idx
        points, centroids = make_inputs(M, N, K, seed)

        if args.mode in ("assign-only", "both"):
            triton_run, triton_out = make_triton_assign_only(points, centroids)
            triton_run()
            torch.cuda.synchronize()
            ref = triton_out.detach().clone()
            triton_ms = bench_ms(triton_run, args.warmup, args.iters)
            triton_wall_ms = bench_wall_ms(triton_run, args.warmup, args.iters)

            print(f"ASSIGN_ONLY shape=({M},{N},{K}) seed={seed} triton_event_ms={triton_ms:.3f} triton_wall_ms={triton_wall_ms:.3f}")
            for kernel_name in args.kernels:
                hopper_run, ids, dists, point_norms, centroid_norms = make_hopper_assign_only(ext, points, centroids, kernel_name)
                hopper_run()
                torch.cuda.synchronize()
                mismatch = int(torch.ne(ids.cpu(), ref.cpu()).sum().item())
                if kernel_name in (
                    "hopper_k5_k7_wgmma256",
                    "hopper_k5_k7_wgmma256_acache",
                    "hopper_k5_k7_wgmma256_persistent",
                    "hopper_k5_k7_wgmma256_persistent_cluster4",
                    "hopper_k5_k7_wgmma256_persistent_cluster8",
                ):
                    cuda_ms = ext.flash_assign_hopper_bench_precomputed_cuda(
                        points,
                        centroids,
                        point_norms,
                        centroid_norms,
                        ids,
                        dists,
                        kernel_name,
                        args.iters,
                    )
                else:
                    cuda_ms = bench_ms(hopper_run, args.warmup, args.iters)
                wall_ms = bench_wall_ms(hopper_run, args.warmup, args.iters)
                print(
                    "HOPPER_ASSIGN_ONLY "
                    f"kernel={kernel_name} "
                    f"shape=({M},{N},{K}) "
                    f"event_ms={cuda_ms:.3f} "
                    f"wall_ms={wall_ms:.3f} "
                    f"speedup={triton_ms / cuda_ms:.3f} "
                    f"mismatch={mismatch}"
                )

        if args.mode in ("full", "both"):
            ref = triton_ref(points, centroids)
            triton_ms = bench_ms(lambda: triton_ref(points, centroids), args.warmup, args.iters)

            print(f"FULL_WRAPPER shape=({M},{N},{K}) seed={seed} triton_ms={triton_ms:.3f}")
            for kernel_name in args.kernels:
                ids, dists, *_ = ext.flash_assign_hopper_cuda(points, centroids, kernel_name)
                cuda_ms = bench_ms(lambda: ext.flash_assign_hopper_cuda(points, centroids, kernel_name), args.warmup, args.iters)
                mismatch = int(torch.ne(ids.cpu(), ref.cpu()).sum().item())
                print(
                    "HOPPER_FULL_WRAPPER "
                    f"kernel={kernel_name} "
                    f"shape=({M},{N},{K}) "
                    f"cuda_ms={cuda_ms:.3f} "
                    f"speedup={triton_ms / cuda_ms:.3f} "
                    f"mismatch={mismatch}"
                )


if __name__ == "__main__":
    main()
