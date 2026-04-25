# CUDA source layout

- `ampere/`: Ampere / Ada flash-assign kernels, helpers, and binders.
- `hopper/`: Hopper-only experiments, binders, and helper utilities.
- `gemm.cu`: separate local GEMM experiment file.

The production-style Ampere/Ada path now lives under `flash_kmeans/csrc/ampere/`, while Hopper experiments stay isolated under `flash_kmeans/csrc/hopper/`.
