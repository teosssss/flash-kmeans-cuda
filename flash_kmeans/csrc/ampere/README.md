# Ampere / Ada CUDA source layout

- `flash_assign.cu`: assign kernels and launchers.
- `flash_assign_common.cuh`: shared tile constants, `cp.async` helpers, WMMA helpers, and shared-memory staging utilities.
- `flash_assign_norm_kernels.cuh`: row-wise L2 norm kernel used by both points and centroids.
- `flash_assign_bind.cpp`: default PyTorch extension entry point.

The temporary `*_tmp.cu` files used by some local benchmark scripts are separate experiments. The main Ampere/Ada comparison path is `flash_assign.cu` plus the matching benchmark binders.
