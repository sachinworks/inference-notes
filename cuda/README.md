# CUDA experiments

Small kernels written to build intuition for GPU performance: memory bandwidth, occupancy, tiling and tensor cores. These are learning exercises, not production code.

## Plan

| # | Experiment | What it teaches |
|---|---|---|
| 01 | Vector add + bandwidth test | Achieved vs. peak memory bandwidth, coalescing |
| 02 | Parallel reduction | Warp shuffles, shared memory, bank conflicts |
| 03 | Matmul: naive → tiled (shared memory) → register blocking | Arithmetic intensity, roofline |
| 04 | Softmax: naive vs. online (one pass) | The trick behind FlashAttention |
| 05 | Same matmul in Triton | How Triton maps blocks to hardware |

Each experiment records the GPU, the measured numbers, and what a profiler (Nsight Compute) showed.
