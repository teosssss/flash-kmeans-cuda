from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from benchmark_cuda_vs_triton import (
    KERNELS,
    bench_ms,
    choose_best_kernel,
    load_ext,
    make_inputs,
    triton_assign,
)


ROOT = Path(__file__).resolve().parents[1]

REGIMES = {
    "large_m_large_n": [
        (262144, 16384, 256),
        (524288, 32768, 256),
        (1048576, 65536, 256),
        (262144, 32768, 512),
    ],
    "large_m_small_n": [
        (1048576, 1024, 128),
        (2097152, 1024, 128),
        (1048576, 2048, 128),
        (2097152, 2048, 256),
    ],
    "small_m_small_n": [
        (4096, 1024, 128),
        (8192, 1024, 128),
        (16384, 2048, 128),
        (4096, 1024, 256),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CUDA flash-assign against Triton on representative assignment regimes."
    )
    parser.add_argument(
        "--regimes",
        nargs="*",
        choices=sorted(REGIMES.keys()),
        default=list(REGIMES.keys()),
        help="Subset of representative regimes to run.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "cuda_vs_triton_regimes",
    )
    parser.add_argument(
        "--require-exact-match",
        action="store_true",
        help="Only summarize shapes where at least one CUDA kernel has zero mismatch against Triton.",
    )
    return parser.parse_args()


def shape_label(shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    return f"M={m:,} N={n:,} K={k}"


def regime_title(name: str) -> str:
    return name.replace("_", " ").title()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary_rows: list[dict], best_rows: list[dict]) -> None:
    lines = [
        "# CUDA vs Triton Representative Regimes",
        "",
        "These are assignment-only benchmarks grouped into representative shape regimes, not end-to-end k-means runs.",
        "",
        "| Regime | Shapes | CUDA wins | Mean speedup | Geomean | Best speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['regime']}` | {row['num_shapes']} | {row['cuda_faster_shapes']}/{row['num_shapes']} | "
            f"{row['mean_speedup']:.3f}x | {row['geomean_speedup']:.3f}x | "
            f"{row['best_speedup']:.3f}x on `{row['best_shape']}` |"
        )

    lines.extend(
        [
            "",
            "## Best CUDA kernel per shape",
            "",
            "| Regime | Shape | Best CUDA kernel | CUDA ms | Triton ms | Speedup |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in best_rows:
        lines.append(
            f"| `{row['regime']}` | `{row['shape_label']}` | `{row['best_kernel']}` | "
            f"{row['cuda_ms']:.3f} | {row['triton_ms']:.3f} | {row['speedup_vs_triton']:.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_svg(path: Path, summary_rows: list[dict]) -> None:
    if not summary_rows:
        path.write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"180\" viewBox=\"0 0 900 180\">\n"
            "<rect width=\"100%\" height=\"100%\" fill=\"#fffdf8\"/>\n"
            "<text x=\"40\" y=\"80\" font-size=\"28\" font-family=\"Helvetica, Arial, sans-serif\" fill=\"#1f2937\">No regime data available.</text>\n"
            "</svg>\n"
        )
        return

    width = 1180
    height = 240 + 110 * len(summary_rows)
    left = 280
    right = 180
    top = 90
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_h = 28
    row_gap = 110
    max_speed = max(row["best_speedup"] for row in summary_rows)
    x_max = max(1.2, math.ceil(max_speed * 10.0) / 10.0)

    def x_pos(val: float) -> float:
        return left + (val / x_max) * plot_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="36" font-size="28" font-family="Helvetica, Arial, sans-serif" fill="#1f2937">CUDA vs Triton: representative assignment regimes</text>',
        f'<text x="{left}" y="58" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">Bars show mean speedup by regime. Labels on the right show the best point in each regime.</text>',
    ]

    tick = 0.0
    while tick <= x_max + 1e-6:
        x = x_pos(tick)
        color = "#d97706" if abs(tick - 1.0) < 1e-6 else "#e5e7eb"
        stroke_width = 2 if abs(tick - 1.0) < 1e-6 else 1
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="{color}" stroke-width="{stroke_width}"/>')
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="13" '
            'font-family="Helvetica, Arial, sans-serif" fill="#6b7280">'
            f"{tick:.1f}</text>"
        )
        tick += 0.2

    for idx, row in enumerate(summary_rows):
        y = top + idx * row_gap + 30
        x_end = x_pos(row["mean_speedup"])
        best_x = x_pos(row["best_speedup"])
        mean_label_x = min(x_end + 10, left + plot_width - 120)
        best_label_x = min(best_x + 12, left + plot_width - 210)
        elements.append(
            f'<text x="{left - 18}" y="{y + 6:.1f}" text-anchor="end" font-size="17" '
            'font-family="Helvetica, Arial, sans-serif" fill="#374151">'
            f"{regime_title(row['regime'])}</text>"
        )
        elements.append(f'<rect x="{left}" y="{y - bar_h / 2:.1f}" width="{x_end - left:.1f}" height="{bar_h}" rx="6" fill="#0f766e" opacity="0.92"/>')
        elements.append(
            f'<text x="{mean_label_x:.1f}" y="{y - 20:.1f}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">'
            f"mean {row['mean_speedup']:.3f}x</text>"
        )
        elements.append(f'<line x1="{best_x:.1f}" y1="{y - 24:.1f}" x2="{best_x:.1f}" y2="{y + 24:.1f}" stroke="#1d4ed8" stroke-width="3"/>')
        elements.append(
            f'<text x="{best_label_x:.1f}" y="{y + 28:.1f}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#1d4ed8">'
            f"best {row['best_speedup']:.3f}x</text>"
        )
        elements.append(
            f'<text x="{best_label_x:.1f}" y="{y + 46:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">'
            f"{row['best_shape']}</text>"
        )

    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    ext = load_ext()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    best_rows: list[dict] = []
    regime_rows: list[dict] = []

    for regime in args.regimes:
        shapes = REGIMES[regime]
        regime_best_rows: list[dict] = []
        for shape_idx, (num_points, num_centroids, dim) in enumerate(shapes):
            seed = args.seed + len(raw_rows) + shape_idx
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
                mismatch = int((ids != ref).sum().item())
                row = {
                    "regime": regime,
                    "shape_label": shape_label((num_points, num_centroids, dim)),
                    "M": num_points,
                    "N": num_centroids,
                    "K": dim,
                    "seed": seed,
                    "kernel": kernel_name,
                    "cuda_ms": cuda_ms,
                    "triton_ms": triton_ms,
                    "speedup_vs_triton": triton_ms / cuda_ms,
                    "mismatch": mismatch,
                }
                raw_rows.append(row)
                shape_rows.append(row)
                print(
                    f"CUDA_KERNEL regime={regime} shape=({num_points},{num_centroids},{dim}) "
                    f"kernel={kernel_name} cuda_ms={cuda_ms:.3f} triton_ms={triton_ms:.3f} "
                    f"speedup={triton_ms / cuda_ms:.3f}x mismatch={mismatch}"
                )

            best = choose_best_kernel(shape_rows, args.require_exact_match)
            if best is None:
                continue
            best_row = {
                **best,
                "best_kernel": best["kernel"],
            }
            best_rows.append(best_row)
            regime_best_rows.append(best_row)
            print(
                f"BEST_CUDA regime={regime} shape=({num_points},{num_centroids},{dim}) "
                f"kernel={best_row['best_kernel']} cuda_ms={best_row['cuda_ms']:.3f} "
                f"triton_ms={best_row['triton_ms']:.3f} speedup={best_row['speedup_vs_triton']:.3f}x"
            )

        if not regime_best_rows:
            continue

        speeds = [row["speedup_vs_triton"] for row in regime_best_rows]
        best_regime_row = max(regime_best_rows, key=lambda row: row["speedup_vs_triton"])
        regime_rows.append(
            {
                "regime": regime,
                "num_shapes": len(regime_best_rows),
                "cuda_faster_shapes": sum(1 for row in regime_best_rows if row["speedup_vs_triton"] > 1.0),
                "mean_speedup": sum(speeds) / len(speeds),
                "geomean_speedup": math.exp(sum(math.log(x) for x in speeds) / len(speeds)),
                "best_speedup": best_regime_row["speedup_vs_triton"],
                "best_shape": best_regime_row["shape_label"],
            }
        )

    write_csv(
        args.out_dir / "all_kernels.csv",
        raw_rows,
        ["regime", "shape_label", "M", "N", "K", "seed", "kernel", "cuda_ms", "triton_ms", "speedup_vs_triton", "mismatch"],
    )
    write_csv(
        args.out_dir / "best_vs_triton.csv",
        best_rows,
        ["regime", "shape_label", "M", "N", "K", "seed", "best_kernel", "cuda_ms", "triton_ms", "speedup_vs_triton", "mismatch"],
    )
    write_summary(args.out_dir / "summary.md", regime_rows, best_rows)
    write_svg(args.out_dir / "regime_speedup.svg", regime_rows)
    (args.out_dir / "regimes.json").write_text(json.dumps({"regimes": args.regimes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
