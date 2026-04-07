# Flash-KMeans

<p align="left">
| <a href="https://svg-project.github.io/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2603.09229"><b>Paper</b></a> | <a href="https://x.com/HaochengXiUCB/status/2033693755791052804"><b>Twitter/X</b></a> |
</p>

IO-aware batched K-Means clustering implemented with Triton GPU kernels. This repository provides the official K-Means implementation of [Sparse VideoGen2](https://arxiv.org/pdf/2505.18875).

![Teasor](assets/FlashAssignAndTime.png)


## Installation

Install flash-kmeans with `pip`:

```bash
pip install flash-kmeans
```

From source:

```bash
git clone https://github.com/svg-project/flash-kmeans.git
cd flash-kmeans
pip install -e .
```

## Usage

```python
import torch
from flash_kmeans import batch_kmeans_Euclid

x = torch.randn(32, 75600, 128, device="cuda", dtype=torch.float16)
cluster_ids, centers, _ = batch_kmeans_Euclid(x, n_clusters=1000, tol=1e-4, verbose=True)
```

We also provide a API interface similar to `faiss/sklearn`, see [API docs](https://github.com/svg-project/flash-kmeans/blob/main/flash_kmeans/interface.py) for details.

## Benchmark


We compare the performance of our Triton implementation with the following baselines:
  - [fast_pytorch_kmeans](https://github.com/DeMoriarty/fast_pytorch_kmeans) a Pytorch implmentation of K-Means clustering.
  - [fastkmeans(triton) / fastkmeans(torch)](https://github.com/AnswerDotAI/fastkmeans) another triton implementation of K-Means clustering. (and its Pytorch fallback)
  - flash-kmeans(triton) / flash-kmeans(torch): our implementation in Triton and Pytorch fallback.
  - batched torch kmeans: a naive batch implementation without considering OOM. 

Tested on NVIDIA H200 GPU with FP16 precision, 128 demensional data, varying number of clusters (k), data points (n) and batch size (b). Our Triton implementation brings significant performance improvements. 

![Benchmark result 1](assets/bench_k100.png)
![Benchmark result 2](assets/bench_k128.png)

Note: fastkmeans(triton) get error when k=100 or k=1000 in figure 1.

### Large tensor Benchmark

For large input that cannot fit in GPU memory, we compare the performance with fastkmeans(triton) with FP32 precision, 128 demensional data, number if data points scaling from 256K to 268M  (N = 2^18, 2^20, 2^22, 2^24, 2^26, 2^28) with cluster counts following K = √N (512, 1024, 2048, 4096, 8192, 16384).

Input tensor is generated randomly in CPU pinned memory. both flash-kmeans and fastkmeans transfer data from CPU to GPU in chunk and compute.

![benchmark large N](assets/benchmark_large.png)

### CUDA flash-assign vs Triton

We benchmarked the CUDA flash-assign kernels against the Triton `euclid_assign_triton` baseline on Modal with an NVIDIA L4 GPU, FP16 inputs, and a 13-shape sweep covering `K in {128, 256, 512}`. For each shape, we report the fastest CUDA kernel among `generic_main`, `aligned_generic_main`, `aligned_static_main`, `deferred_generic`, and `deferred_static`.

Across this sweep, the best CUDA kernel per shape won on all 13 tested shapes:
  - mean speedup: `1.748x`
  - geometric-mean speedup: `1.667x`
  - best speedup: `3.635x` on `M=4,096, N=1,024, K=128`
  - worst speedup: `1.077x` on `M=32,768, N=4,096, K=128`

The strongest kernel is shape-dependent. `deferred_generic` remains the most reliable general-purpose variant, `deferred_static` wins the smallest `K=128` cases in this run, and `aligned_generic_main` is strongest on several aligned medium and large shapes.

A `3.635x` speedup here means `3.635x` over Triton, not over the older baselines above. Since Flash-KMeans Triton already reports up to `17.9x` over the best prior baselines, `33x` over cuML, and `200x+` over FAISS, this implies roughly `65.1x`, `120.0x`, and `727.1x` stacked speedups at the best point, or about `31.3x`, `57.7x`, and `349.6x` using the mean CUDA gain (`1.748x`). These stacked numbers are directional only, since the Triton paper results and this CUDA benchmark were not measured on the same hardware and benchmark suite.

![CUDA vs Triton speedup](assets/cuda_vs_triton_modal_13_final.svg)

The exact best-vs-best results from this Modal run are:

| Shape | Best CUDA kernel | CUDA ms | Triton ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| `M=4,096 N=1,024 K=128` | `deferred_static` | 0.057 | 0.206 | 3.635x |
| `M=8,192 N=2,048 K=128` | `deferred_static` | 0.115 | 0.201 | 1.750x |
| `M=32,768 N=4,096 K=128` | `deferred_generic` | 0.769 | 0.828 | 1.077x |
| `M=131,072 N=16,384 K=128` | `aligned_generic_main` | 10.907 | 13.139 | 1.205x |
| `M=4,096 N=1,024 K=256` | `deferred_generic` | 0.091 | 0.197 | 2.176x |
| `M=8,192 N=2,048 K=256` | `aligned_generic_main` | 0.191 | 0.281 | 1.467x |
| `M=32,768 N=4,096 K=256` | `deferred_generic` | 1.370 | 2.206 | 1.610x |
| `M=131,072 N=16,384 K=256` | `deferred_generic` | 18.846 | 26.240 | 1.392x |
| `M=262,144 N=32,768 K=256` | `deferred_generic` | 74.662 | 99.031 | 1.326x |
| `M=4,096 N=1,024 K=512` | `deferred_generic` | 0.158 | 0.294 | 1.859x |
| `M=8,192 N=2,048 K=512` | `aligned_generic_main` | 0.380 | 0.652 | 1.714x |
| `M=32,768 N=4,096 K=512` | `aligned_generic_main` | 2.671 | 5.048 | 1.890x |
| `M=131,072 N=16,384 K=512` | `deferred_generic` | 36.332 | 59.092 | 1.626x |

This repository also includes a dedicated benchmark for rerunning or extending the comparison:

```bash
python3 examples/benchmark_cuda_vs_triton.py \
  --out-dir /path/to/output_dir
```

The default sweep in `benchmark_cuda_vs_triton.py` matches the 13-shape Modal figure above:
  - `K=128`: `(4096,1024)`, `(8192,2048)`, `(32768,4096)`, `(131072,16384)`
  - `K=256`: `(4096,1024)`, `(8192,2048)`, `(32768,4096)`, `(131072,16384)`, `(262144,32768)`
  - `K=512`: `(4096,1024)`, `(8192,2048)`, `(32768,4096)`, `(131072,16384)`

This is the canonical experiment used for the README summary and table.

It writes:
  - `all_kernels.csv`: every CUDA kernel variant against Triton for every tested shape.
  - `best_vs_triton.csv`: the fastest CUDA kernel per shape.
  - `summary.md`: a ready-to-paste Markdown summary with mean/geomean/best advantage.
  - `best_vs_triton_speedup.svg`: a plot of `Triton time / CUDA time` per shape.

The main README figure intentionally stays best-vs-best. The full kernel sweep still lives in `all_kernels.csv` for anyone who wants the tuning details without cluttering the top-level story.

#### CUDA kernel notes

| Kernel | Main optimization ideas |
| --- | --- |
| `generic_main` | `cp.async` triple buffering over `K`, WMMA tensor-core MMA, register-resident partial minima, shared-memory row reduction, explicit tail handling for non-aligned tiles. |
| `aligned_generic_main` | Same pipeline as `generic_main`, but removes edge guards on aligned shapes so the load path is cheaper and more regular. |
| `aligned_static_main` | Adds compile-time specialization for common `K` values (`128/256/512`) so the compiler can unroll the `K` loop more aggressively and simplify address arithmetic. |
| `deferred_generic` | Keeps the tensor-core tiled pipeline, but defers the row-min writeback so more of the distance reduction stays in registers before shared-memory merge. |
| `deferred_static` | Combines deferred reduction with static-`K` specialization; this is the highest-ceiling version on the common aligned shapes. |


## Citation

If you use this codebase, or otherwise found our work valuable, please cite:

```
@article{yang2026flash,
  title={Flash-KMeans: Fast and Memory-Efficient Exact K-Means},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Fan, Xiaoze and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Keutzer, Kurt and others},
  journal={arXiv preprint arXiv:2603.09229},
  year={2026}
}

@article{yang2025sparse,
  title={Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Xu, Chenfeng and Peng, Kelly and others},
  journal={arXiv preprint arXiv:2505.18875},
  year={2025}
}
```
