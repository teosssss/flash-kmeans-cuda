from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import geometric_mean

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

import torch
from torch.utils.cpp_extension import load

from flash_kmeans.assign_euclid_triton import euclid_assign_triton


ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "flash_kmeans" / "csrc"

KERNELS = [
    "generic_main",
    "aligned_generic_main",
    "aligned_static_main",
    "deferred_generic",
    "deferred_static",
]

DEFAULT_SHAPES = [
    (4096, 1024, 128),
    (8192, 2048, 128),
    (32768, 4096, 128),
    (131072, 16384, 128),
    (4096, 1024, 256),
    (8192, 2048, 256),
    (32768, 4096, 256),
    (131072, 16384, 256),
    (262144, 32768, 256),
    (4096, 1024, 512),
    (8192, 2048, 512),
    (32768, 4096, 512),
    (131072, 16384, 512),
]

KERNEL_LABELS = {
    "generic_main": "Generic pipeline",
    "aligned_generic_main": "Aligned pipeline",
    "aligned_static_main": "Aligned + static K",
    "deferred_generic": "Deferred reduce",
    "deferred_static": "Deferred reduce + static K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CUDA flash-assign variants against the Triton baseline.")
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=[f"{m},{n},{k}" for m, n, k in DEFAULT_SHAPES],
        help="Cases as M,N,K triplets.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "cuda_vs_triton",
        help="Directory where raw results, summary tables, and plots are written.",
    )
    parser.add_argument(
        "--require-exact-match",
        action="store_true",
        help="Only summarize shapes where at least one CUDA kernel has zero mismatch against Triton.",
    )
    return parser.parse_args()


def parse_shapes(raw_shapes: list[str]) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for raw in raw_shapes:
        m_str, n_str, k_str = raw.split(",")
        shapes.append((int(m_str), int(n_str), int(k_str)))
    return shapes


def load_ext():
    return load(
        name="flash_assign_cuda_vs_triton_ext",
        sources=[
            str(CSRC / "flash_assign_all_kernels_tmp_bind.cpp"),
            str(CSRC / "flash_assign.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def bench_ms(fn, warmup: int, iters: int) -> float:
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


def triton_assign(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    x = points.unsqueeze(0)
    c = centroids.unsqueeze(0)
    x_sq = (x.float() * x.float()).sum(dim=-1)
    c_sq = (c.float() * c.float()).sum(dim=-1)
    return euclid_assign_triton(x, c, x_sq, c_sq=c_sq, use_heuristic=True).squeeze(0)


def make_inputs(num_points: int, num_centroids: int, dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    points = torch.randn((num_points, dim), generator=gen, device="cuda", dtype=torch.float16).contiguous()
    centroids = torch.randn((num_centroids, dim), generator=gen, device="cuda", dtype=torch.float16).contiguous()
    return points, centroids


def shape_label(shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    return f"M={m:,} N={n:,} K={k}"


def kernel_sort_key(name: str) -> tuple[int, str]:
    return (KERNELS.index(name), name)


def choose_best_kernel(rows: list[dict], require_exact_match: bool) -> dict | None:
    valid_rows = [row for row in rows if row["mismatch"] == 0]
    if require_exact_match and not valid_rows:
        return None
    candidate_rows = valid_rows if valid_rows else rows
    return min(candidate_rows, key=lambda row: row["cuda_ms"])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, summary: dict, best_rows: list[dict], skipped_rows: list[dict]) -> None:
    lines = [
        "# CUDA vs Triton Summary",
        "",
        f"- Shapes requested: {summary['num_shapes_requested']}",
        f"- Shapes summarized: {summary['num_shapes_summarized']}",
        f"- Shapes skipped for correctness: {summary['num_shapes_skipped']}",
        "",
        "## Best CUDA kernel per shape",
        "",
        "| Shape | Best CUDA kernel | CUDA ms | Triton ms | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    if best_rows:
        lines[4:4] = [
            f"- CUDA faster on: {summary['cuda_faster_shapes']}/{summary['num_shapes_summarized']} summarized shapes",
            f"- Mean speedup (Triton / CUDA): {summary['mean_speedup']:.3f}x",
            f"- Geometric mean speedup: {summary['geomean_speedup']:.3f}x",
            f"- Best speedup: {summary['best_speedup']:.3f}x on `{summary['best_shape']}` with `{summary['best_kernel']}`",
            f"- Worst speedup: {summary['worst_speedup']:.3f}x on `{summary['worst_shape']}` with `{summary['worst_kernel']}`",
        ]
    for row in best_rows:
        lines.append(
            f"| `{row['shape_label']}` | `{row['best_kernel']}` | {row['cuda_ms']:.3f} | "
            f"{row['triton_ms']:.3f} | {row['speedup_vs_triton']:.3f}x |"
        )
    if skipped_rows:
        lines.extend(
            [
                "",
                "## Skipped shapes",
                "",
                "| Shape | Reason |",
                "| --- | --- |",
            ]
        )
        for row in skipped_rows:
            lines.append(f"| `{row['shape_label']}` | {row['reason']} |")
    path.write_text("\n".join(lines) + "\n")


def format_svg_text(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_speedup_svg(path: Path, best_rows: list[dict], summary: dict) -> None:
    if not best_rows:
        path.write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"180\" viewBox=\"0 0 960 180\">\n"
            "<rect width=\"100%\" height=\"100%\" fill=\"#fffdf8\"/>\n"
            "<text x=\"40\" y=\"80\" font-size=\"28\" font-family=\"Helvetica, Arial, sans-serif\" fill=\"#1f2937\">"
            "No zero-mismatch shapes were available for summary.</text>\n"
            "</svg>\n"
        )
        return

    width = 1560
    height = 900
    left = 90
    right = 120
    top = 70
    bottom = 260
    plot_width = width - left - right
    plot_height = height - top - bottom
    label_pad = 36

    min_speed = min(row["speedup_vs_triton"] for row in best_rows)
    max_speed = max(row["speedup_vs_triton"] for row in best_rows)
    y_min = min(0.9, math.floor(min_speed * 10.0) / 10.0)
    y_max = max(1.1, math.ceil(max_speed * 10.0) / 10.0)
    y_span = max(y_max - y_min, 0.2)

    def x_pos(idx: int) -> float:
        if len(best_rows) == 1:
            return left + plot_width / 2
        usable_width = max(plot_width - 2 * label_pad, 1)
        return left + label_pad + idx * usable_width / (len(best_rows) - 1)

    def y_pos(val: float) -> float:
        return top + plot_height - ((val - y_min) / y_span) * plot_height

    grid_values = []
    tick = math.floor(y_min * 10.0) / 10.0
    while tick <= y_max + 1e-6:
        grid_values.append(round(tick, 1))
        tick += 0.1

    points = " ".join(f"{x_pos(i):.1f},{y_pos(row['speedup_vs_triton']):.1f}" for i, row in enumerate(best_rows))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="36" font-size="28" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">CUDA vs Triton: best kernel per shape</text>',
        f'<text x="{left}" y="58" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">Speedup is Triton time divided by CUDA time. Values above 1.0 mean CUDA is faster.</text>',
    ]

    for val in grid_values:
        y = y_pos(val)
        color = "#d97706" if abs(val - 1.0) < 1e-6 else "#e5e7eb"
        stroke_width = 2 if abs(val - 1.0) < 1e-6 else 1
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{color}" stroke-width="{stroke_width}"/>')
        elements.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13" '
            'font-family="Helvetica, Arial, sans-serif" fill="#6b7280">'
            f"{val:.1f}</text>"
        )

    elements.append(f'<polyline fill="none" stroke="#0f766e" stroke-width="3" points="{points}"/>')

    for idx, row in enumerate(best_rows):
        x = x_pos(idx)
        y = y_pos(row["speedup_vs_triton"])
        color = "#0f766e" if row["speedup_vs_triton"] >= 1.0 else "#b91c1c"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        elements.append(
            f'<text transform="translate({x:.1f},{height - bottom + 48}) rotate(42)" text-anchor="end" font-size="13" '
            'font-family="Helvetica, Arial, sans-serif" fill="#374151">'
            f"{format_svg_text(row['shape_label'])}</text>"
        )

    legend_x = left + plot_width - 280
    legend_y = top + 18
    elements.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="240" height="90" rx="10" fill="#ffffff" stroke="#e5e7eb"/>',
            f'<text x="{legend_x + 16}" y="{legend_y + 28}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">Mean speedup: {summary["mean_speedup"]:.3f}x</text>',
            f'<text x="{legend_x + 16}" y="{legend_y + 50}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">Geomean speedup: {summary["geomean_speedup"]:.3f}x</text>',
            f'<text x="{legend_x + 16}" y="{legend_y + 72}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">Best speedup: {summary["best_speedup"]:.3f}x</text>',
        ]
    )

    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    shapes = parse_shapes(args.shapes)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = load_ext()

    raw_rows: list[dict] = []
    best_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for shape_idx, (num_points, num_centroids, dim) in enumerate(shapes):
        seed = args.seed + shape_idx
        points, centroids = make_inputs(num_points, num_centroids, dim, seed)

        ref = triton_assign(points, centroids)
        triton_ms = bench_ms(lambda: triton_assign(points, centroids), args.warmup, args.iters)

        shape_rows: list[dict] = []
        for kernel_name in KERNELS:
            ids, _dists, *_ = ext.flash_assign_all_kernels_tmp_cuda(points, centroids, kernel_name)
            cuda_ms = bench_ms(
                lambda kernel=kernel_name: ext.flash_assign_all_kernels_tmp_cuda(points, centroids, kernel),
                args.warmup,
                args.iters,
            )
            mismatch = int(torch.ne(ids, ref).sum().item())
            row = {
                "shape_label": shape_label((num_points, num_centroids, dim)),
                "M": num_points,
                "N": num_centroids,
                "K": dim,
                "seed": seed,
                "kernel": kernel_name,
                "kernel_label": KERNEL_LABELS[kernel_name],
                "cuda_ms": cuda_ms,
                "triton_ms": triton_ms,
                "speedup_vs_triton": triton_ms / cuda_ms,
                "mismatch": mismatch,
            }
            raw_rows.append(row)
            shape_rows.append(row)
            print(
                "CUDA_KERNEL "
                f"shape=({num_points},{num_centroids},{dim}) "
                f"kernel={kernel_name} cuda_ms={cuda_ms:.3f} triton_ms={triton_ms:.3f} "
                f"speedup={row['speedup_vs_triton']:.3f}x mismatch={mismatch}"
            )

        best_kernel_row = choose_best_kernel(shape_rows, require_exact_match=args.require_exact_match)
        if best_kernel_row is None:
            skipped_rows.append(
                {
                    "shape_label": shape_label((num_points, num_centroids, dim)),
                    "M": num_points,
                    "N": num_centroids,
                    "K": dim,
                    "reason": "all CUDA kernels had non-zero mismatch against Triton in exact-match mode",
                }
            )
            print(
                "SKIP_SHAPE "
                f"shape=({num_points},{num_centroids},{dim}) "
                "reason=no_zero_mismatch_kernel"
            )
            continue

        best_row = dict(best_kernel_row)
        best_row["best_kernel"] = best_row["kernel"]
        best_rows.append(best_row)
        print(
            "BEST_CUDA "
            f"shape=({num_points},{num_centroids},{dim}) "
            f"kernel={best_row['best_kernel']} cuda_ms={best_row['cuda_ms']:.3f} "
            f"triton_ms={best_row['triton_ms']:.3f} speedup={best_row['speedup_vs_triton']:.3f}x"
        )

    summary = {
        "num_shapes_requested": len(shapes),
        "num_shapes_summarized": len(best_rows),
        "num_shapes_skipped": len(skipped_rows),
        "total_best_kernel_mismatches": sum(row["mismatch"] for row in best_rows),
        "cuda_faster_shapes": 0,
        "mean_speedup": None,
        "geomean_speedup": None,
        "best_speedup": None,
        "best_shape": None,
        "best_kernel": None,
        "worst_speedup": None,
        "worst_shape": None,
        "worst_kernel": None,
    }
    if best_rows:
        speedups = [row["speedup_vs_triton"] for row in best_rows]
        best_speed_row = max(best_rows, key=lambda row: row["speedup_vs_triton"])
        worst_speed_row = min(best_rows, key=lambda row: row["speedup_vs_triton"])
        summary.update(
            {
                "cuda_faster_shapes": sum(1 for row in best_rows if row["speedup_vs_triton"] > 1.0),
                "mean_speedup": sum(speedups) / len(speedups),
                "geomean_speedup": geometric_mean(speedups),
                "best_speedup": best_speed_row["speedup_vs_triton"],
                "best_shape": best_speed_row["shape_label"],
                "best_kernel": best_speed_row["best_kernel"],
                "worst_speedup": worst_speed_row["speedup_vs_triton"],
                "worst_shape": worst_speed_row["shape_label"],
                "worst_kernel": worst_speed_row["best_kernel"],
            }
        )

    raw_json_path = out_dir / "raw_results.json"
    raw_json_path.write_text(
        json.dumps({"summary": summary, "raw_rows": raw_rows, "best_rows": best_rows, "skipped_rows": skipped_rows}, indent=2) + "\n"
    )

    write_csv(
        out_dir / "all_kernels.csv",
        raw_rows,
        ["shape_label", "M", "N", "K", "seed", "kernel", "kernel_label", "cuda_ms", "triton_ms", "speedup_vs_triton", "mismatch"],
    )
    write_csv(
        out_dir / "best_vs_triton.csv",
        best_rows,
        ["shape_label", "M", "N", "K", "seed", "best_kernel", "kernel_label", "cuda_ms", "triton_ms", "speedup_vs_triton", "mismatch"],
    )
    write_csv(
        out_dir / "skipped_shapes.csv",
        skipped_rows,
        ["shape_label", "M", "N", "K", "reason"],
    )
    write_markdown_summary(out_dir / "summary.md", summary, best_rows, skipped_rows)
    write_speedup_svg(out_dir / "best_vs_triton_speedup.svg", best_rows, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
