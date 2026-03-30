# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmark for FlashMoE EP communication strategies.

Compares V1 (NCCL AllReduce) vs V4 (fused GEMM+AllReduce) latency.

Usage (single GPU, V4 kernel-only benchmark):
  python bench_flashmoe_ep.py --mode kernel --m 1024 --n 7168 --k 2048 --experts 8

Multi-GPU (requires MPI):
  mpirun -n 4 python bench_flashmoe_ep.py --mode full --m 1024 --n 7168 --k 2048 --experts 8
"""

import argparse

import torch


def bench_allreduce_kernel_single_gpu(m, n, k, l, top_k=2, tile_size=128, warmup=10, iters=100):  # noqa: E741
    """Benchmark the 11-warp AllReduce kernel on a single GPU.

    Measures kernel launch overhead vs the base finalize-fusion kernel.
    AR warps are no-op when world_size=1.
    """
    try:
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
            Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
        )
    except ImportError as e:
        print(f"Cannot import required modules: {e}")
        return

    device = torch.device("cuda")
    sf_vec_size = 16
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size
    mma_tiler_mn = (tile_size, 128)
    cluster_shape_mn = (tile_size // 128, 1)

    make_ptr = cutlass.cute.runtime.make_ptr

    # Allocate inputs
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (l, n, k // 2), dtype=torch.uint8, device=device)
    a_sf = torch.randint(0, 256, (m * scale_k,), dtype=torch.uint8, device=device)
    b_sf = torch.randint(0, 256, (l, n, scale_k), dtype=torch.uint8, device=device)
    alpha = torch.ones(l, dtype=torch.float32, device=device)
    tile_idx_to_expert_idx = torch.arange(num_tiles, dtype=torch.int32, device=device) % l
    tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32, device=device)
    permuted_idx = torch.arange(m, dtype=torch.int32, device=device)
    num_non_exit = torch.tensor([num_tiles], dtype=torch.int32, device=device)
    token_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)
    staging = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    out = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])

    def make_ptrs():
        return (
            make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(
                cutlass.Float8E4M3FN, a_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float8E4M3FN, b_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, permuted_idx.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, num_non_exit.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Float32, token_scales.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(
                cutlass.BFloat16, staging.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        )

    print(f"\nBenchmark: M={m}, N={n}, K={k}, L={l}, top_k={top_k}")
    print(f"  tile_size={tile_size}, mma_tiler_mn={mma_tiler_mn}")
    print(f"  warmup={warmup}, iters={iters}")

    # Compile and benchmark base kernel
    p = make_ptrs()
    base = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    compiled_base = cute.compile(
        base.wrapper,
        p[0],
        p[1],
        p[2],
        p[3],
        p[11],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_clusters,
        stream=stream,
    )

    for _ in range(warmup):
        compiled_base(
            p[0],
            p[1],
            p[2],
            p[3],
            p[11],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            stream=stream,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled_base(
            p[0],
            p[1],
            p[2],
            p[3],
            p[11],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            stream=stream,
        )
    end.record()
    torch.cuda.synchronize()
    base_ms = start.elapsed_time(end) / iters
    print(f"  Base kernel (7-warp):     {base_ms:.3f} ms")

    # Compile and benchmark AllReduce kernel (world_size=1)
    ar = Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    compiled_ar = cute.compile(
        ar.wrapper,
        p[0],
        p[1],
        p[2],
        p[3],
        p[10],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        0,
        0,
        0,
        0,
        p[11],
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        0,
        1,  # rank=0, world_size=1
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_clusters,
        stream=stream,
    )

    for _ in range(warmup):
        compiled_ar(
            p[0],
            p[1],
            p[2],
            p[3],
            p[10],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
            0,
            0,
            0,
            0,
            p[11],
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            # rank and world_size are Constexpr, baked in at compile time
            stream=stream,
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        compiled_ar(
            p[0],
            p[1],
            p[2],
            p[3],
            p[10],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
            0,
            0,
            0,
            0,
            p[11],
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            # rank and world_size are Constexpr, baked in at compile time
            stream=stream,
        )
    end.record()
    torch.cuda.synchronize()
    ar_ms = start.elapsed_time(end) / iters
    print(f"  AR kernel (11-warp, ws=1): {ar_ms:.3f} ms")
    print(f"  Overhead: {(ar_ms / base_ms - 1) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="FlashMoE EP benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        default="kernel",
        choices=["kernel", "full"],
        help="kernel: single-GPU kernel benchmark, full: multi-GPU EP",
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "kernel":
        bench_allreduce_kernel_single_gpu(
            m=args.m,
            n=args.n,
            k=args.k,
            l=args.experts,
            top_k=args.top_k,
            tile_size=args.tile_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        print("Full multi-GPU EP benchmark requires MPI launch.")
        print("Will be added when multi-GPU IPC path is validated.")


if __name__ == "__main__":
    main()
