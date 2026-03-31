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
Benchmark for FlashMoE fused GEMM+AllReduce kernel strategies.

Modes:
  kernel:    Single-GPU overhead: 7-warp base vs 11-warp (world_size=1)
  strategy:  Simulated multi-GPU: compare AR strategy 0 (batch) vs 1 (overlapped)
  ipc:       Real multi-GPU IPC: uses MoEEpAllReduceMnnvlMemory for cross-GPU AllReduce
  ipc_sweep: Multi-config sweep with real IPC

Usage:
  python bench_flashmoe_ep.py --mode kernel --m 1024 --n 7168 --k 2048 --experts 8
  python bench_flashmoe_ep.py --mode strategy --m 1024 --n 7168 --k 2048 --experts 8 --world_size 2
  python bench_flashmoe_ep.py --mode strategy --m 1024 --n 7168 --k 2048 --experts 8 --world_size 4
  python bench_flashmoe_ep.py --mode sweep

Real IPC (requires MPI launch, e.g., NTASKS=3 for 2 GPUs + 1 controller):
  mpirun -n 3 python bench_flashmoe_ep.py --mode ipc --m 1024 --n 7168 --k 2048 --experts 8
  mpirun -n 5 python bench_flashmoe_ep.py --mode ipc_sweep
"""

import argparse

import torch


def _import_kernels():
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
    )

    return (
        cutlass,
        cute,
        cuda,
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
    )


def _time_kernel(compiled_fn, args, warmup, iters):
    """Time a compiled kernel, returning average ms per iteration."""
    for _ in range(warmup):
        compiled_fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled_fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_kernel_overhead(m, n, k, l, top_k=2, tile_size=128, warmup=10, iters=100):  # noqa: E741
    """Benchmark 11-warp vs 7-warp overhead (world_size=1, AR warps idle)."""
    try:
        cutlass, cute, cuda, ARKernel, BaseKernel = _import_kernels()
    except ImportError as e:
        print(f"Cannot import: {e}")
        return

    device = torch.device("cuda")
    sf_vec_size = 16
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size
    mma_tiler_mn = (tile_size, 128)
    cluster_shape_mn = (tile_size // 128, 1)

    make_ptr = cutlass.cute.runtime.make_ptr

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

    def _ptr(dtype, tensor, align=None):
        kw = {"assumed_align": align} if align else {}
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, **kw)

    p_a = _ptr(cutlass.Float4E2M1FN, a, 32)
    p_b = _ptr(cutlass.Float4E2M1FN, b, 32)
    p_asf = _ptr(cutlass.Float8E4M3FN, a_sf, 16)
    p_bsf = _ptr(cutlass.Float8E4M3FN, b_sf, 16)
    p_alpha = _ptr(cutlass.Float32, alpha)
    p_tig = _ptr(cutlass.Int32, tile_idx_to_expert_idx)
    p_tmn = _ptr(cutlass.Int32, tile_idx_to_mn_limit)
    p_pie = _ptr(cutlass.Int32, permuted_idx)
    p_nne = _ptr(cutlass.Int32, num_non_exit)
    p_tfs = _ptr(cutlass.Float32, token_scales)
    p_stg = _ptr(cutlass.BFloat16, staging, 16)
    p_out = _ptr(cutlass.BFloat16, out, 16)

    print(f"\n=== Kernel Overhead: M={m}, N={n}, K={k}, L={l}, top_k={top_k} ===")
    print(f"    tile_size={tile_size}, warmup={warmup}, iters={iters}")

    # Base 7-warp kernel
    base = BaseKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    base_args_compile = [
        p_a,
        p_b,
        p_asf,
        p_bsf,
        p_out,
        p_alpha,
        p_tig,
        p_tmn,
        p_pie,
        p_nne,
        p_tfs,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
    ]
    compiled_base = cute.compile(
        base.wrapper,
        *base_args_compile,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_clusters,
        stream=stream,
    )
    base_ms = _time_kernel_kw(compiled_base, base_args_compile, stream, warmup, iters)
    print(f"    Base (7-warp):      {base_ms:.3f} ms")

    # 11-warp kernel (world_size=1, ar_strategy=0)
    ar = ARKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    ar_args_compile = [
        p_a,
        p_b,
        p_asf,
        p_bsf,
        p_stg,
        p_alpha,
        p_tig,
        p_tmn,
        p_pie,
        p_nne,
        p_tfs,
        0,
        0,
        0,
        0,  # mc pointers (unused)
        0,
        0,  # rank strides (unused)
        p_out,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        0,
        1,
        0,  # rank=0, world_size=1, ar_strategy=0
    ]
    compiled_ar = cute.compile(
        ar.wrapper,
        *ar_args_compile,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_clusters,
        stream=stream,
    )
    # Runtime args: same but without Constexpr (rank, world_size, ar_strategy)
    ar_run_args = [
        p_a,
        p_b,
        p_asf,
        p_bsf,
        p_stg,
        p_alpha,
        p_tig,
        p_tmn,
        p_pie,
        p_nne,
        p_tfs,
        0,
        0,
        0,
        0,
        0,
        0,
        p_out,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
    ]
    ar_ms = _time_kernel_kw(compiled_ar, ar_run_args, stream, warmup, iters)
    print(f"    AR (11-warp, ws=1): {ar_ms:.3f} ms")
    print(f"    Overhead:           {(ar_ms / base_ms - 1) * 100:.1f}%")

    return base_ms, ar_ms


def _time_kernel_kw(compiled_fn, args, stream, warmup, iters):
    """Time a compiled kernel with stream as keyword arg."""
    for _ in range(warmup):
        compiled_fn(*args, stream=stream)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled_fn(*args, stream=stream)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_ar_strategies(
    m,
    n,
    k,
    l,  # noqa: E741
    top_k=2,
    tile_size=128,
    world_size=2,
    warmup=10,
    iters=100,
):
    """Benchmark AR strategy 0 (batch) vs 1 (overlapped) with simulated multi-GPU.

    Pre-populates staging buffers and pre-sets barriers to world_size so that
    AR warps run the full reduce path without spin-waiting. This measures:
      GEMM + epilogue (writes to staging) + AR reduce (reads staging, writes output)
    """
    try:
        cutlass, cute, cuda, ARKernel, _ = _import_kernels()
    except ImportError as e:
        print(f"Cannot import: {e}")
        return

    device = torch.device("cuda")
    sf_vec_size = 16
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size
    tile_n = 128
    num_2d_tiles = num_tiles * (n // tile_n)
    mma_tiler_mn = (tile_size, tile_n)
    cluster_shape_mn = (tile_size // 128, 1)

    make_ptr = cutlass.cute.runtime.make_ptr

    # Shared GEMM inputs (rank 0)
    torch.manual_seed(42)
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (l, n, k // 2), dtype=torch.uint8, device=device)
    a_sf = torch.randint(0, 8, (m * scale_k,), dtype=torch.uint8, device=device)
    b_sf = torch.randint(0, 8, (l, n, scale_k), dtype=torch.uint8, device=device)
    alpha = torch.ones(l, dtype=torch.float32, device=device) * 0.1
    tile_idx_to_expert_idx = torch.arange(num_tiles, dtype=torch.int32, device=device) % l
    tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32, device=device)
    permuted_idx = torch.arange(m, dtype=torch.int32, device=device)
    num_non_exit = torch.tensor([num_tiles], dtype=torch.int32, device=device)
    token_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

    # Staging: world_size separate buffers laid out contiguously (IPC simulation)
    staging_rank_stride = m * n * 2  # bytes (bf16)
    out_rank_stride = staging_rank_stride
    staging_all = torch.zeros(world_size * m * n, dtype=torch.bfloat16, device=device)
    output_all = torch.zeros(world_size * m * n, dtype=torch.bfloat16, device=device)
    staging_gemm = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    out_test = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

    # Barriers: one i32 per 2D tile
    barrier_bytes = max(num_2d_tiles * 4, 4096)
    tile_barriers = torch.zeros(barrier_bytes // 4, dtype=torch.int32, device=device)
    completion_barriers = torch.zeros(barrier_bytes // 4, dtype=torch.int32, device=device)

    # Populate staging with small random data for each rank
    for rank_i in range(world_size):
        torch.manual_seed(100 + rank_i)
        staging_all[rank_i * m * n : (rank_i + 1) * m * n].copy_(
            torch.randn(m * n, dtype=torch.bfloat16, device=device) * 0.01
        )

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])

    def _ptr(dtype, tensor, align=None):
        kw = {"assumed_align": align} if align else {}
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, **kw)

    p_a = _ptr(cutlass.Float4E2M1FN, a, 32)
    p_b = _ptr(cutlass.Float4E2M1FN, b, 32)
    p_asf = _ptr(cutlass.Float8E4M3FN, a_sf, 16)
    p_bsf = _ptr(cutlass.Float8E4M3FN, b_sf, 16)
    p_alpha = _ptr(cutlass.Float32, alpha)
    p_tig = _ptr(cutlass.Int32, tile_idx_to_expert_idx)
    p_tmn = _ptr(cutlass.Int32, tile_idx_to_mn_limit)
    p_pie = _ptr(cutlass.Int32, permuted_idx)
    p_nne = _ptr(cutlass.Int32, num_non_exit)
    p_tfs = _ptr(cutlass.Float32, token_scales)
    p_stg_gemm = _ptr(cutlass.BFloat16, staging_gemm, 16)
    p_out = _ptr(cutlass.BFloat16, out_test, 16)

    print(f"\n=== AR Strategy Benchmark: M={m}, N={n}, K={k}, L={l}, world_size={world_size} ===")
    print(
        f"    tiles={num_tiles}(M)×{n // tile_n}(N)={num_2d_tiles}(2D), "
        f"warmup={warmup}, iters={iters}"
    )

    results = {}
    for strategy in [0, 1]:
        # Compile for this strategy
        ar = ARKernel(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_blkred=True,
            raster_along_m=False,
        )
        compile_args = [
            p_a,
            p_b,
            p_asf,
            p_bsf,
            p_stg_gemm,
            p_alpha,
            p_tig,
            p_tmn,
            p_pie,
            p_nne,
            p_tfs,
            staging_all.data_ptr(),
            output_all.data_ptr(),
            tile_barriers.data_ptr(),
            completion_barriers.data_ptr(),
            staging_rank_stride,
            out_rank_stride,
            p_out,
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            0,
            world_size,
            strategy,  # rank=0, world_size, ar_strategy
        ]
        compiled = cute.compile(
            ar.wrapper,
            *compile_args,
            tile_size=tile_size,
            scaling_vector_size=sf_vec_size,
            max_active_clusters=max_clusters,
            stream=stream,
        )
        # Runtime args (no Constexpr: rank, world_size, ar_strategy are baked in)
        run_args = [
            p_a,
            p_b,
            p_asf,
            p_bsf,
            p_stg_gemm,
            p_alpha,
            p_tig,
            p_tmn,
            p_pie,
            p_nne,
            p_tfs,
            staging_all.data_ptr(),
            output_all.data_ptr(),
            tile_barriers.data_ptr(),
            completion_barriers.data_ptr(),
            staging_rank_stride,
            out_rank_stride,
            p_out,
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
        ]

        # Before each iteration, reset barriers to world_size and clear output
        def run_once():
            tile_barriers[:num_2d_tiles] = world_size
            completion_barriers.zero_()
            output_all.zero_()
            compiled(*run_args, stream=stream)

        # Warmup (includes barrier reset overhead)
        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()

        # Timed runs: reset barriers in bulk, then time kernel iterations.
        # To avoid measuring barrier reset overhead, pre-set barriers for all
        # iterations and run the kernel. But completion_barriers need reset too.
        # Compromise: measure full run_once() and subtract barrier overhead.
        #
        # Actually, a simpler approach: for benchmarking, we can set barriers
        # once before the timed loop. The AR warps read barriers and reduce,
        # but the epilogue also writes to barriers (incrementing them). So after
        # each kernel run, tile_barriers will be world_size + (number of CTAs
        # that wrote to each barrier). For strategy correctness we'd need reset,
        # but for timing purposes this is fine since AR warps just check >= world_size.
        #
        # The completion_barrier is incremented by AR warps, which we don't wait
        # on in this benchmark.
        tile_barriers[:num_2d_tiles] = world_size
        completion_barriers.zero_()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            compiled(*run_args, stream=stream)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        results[strategy] = ms
        sname = "batch" if strategy == 0 else "overlapped"
        print(f"    Strategy {strategy} ({sname:>10}): {ms:.3f} ms")

    if results:
        s0 = results.get(0, 0)
        s1 = results.get(1, 0)
        if s0 > 0 and s1 > 0:
            faster = 0 if s0 < s1 else 1
            slower = 1 - faster
            speedup = results[slower] / results[faster]
            fname = "batch" if faster == 0 else "overlapped"
            print(f"    Winner: strategy {faster} ({fname}), {speedup:.2f}x faster")

    return results


def bench_sweep(warmup=5, iters=50):
    """Run a sweep of configs comparing strategies."""
    configs = [
        # (m, n, k, l, top_k, world_size)
        (256, 2048, 2048, 8, 2, 2),
        (256, 2048, 2048, 8, 2, 4),
        (512, 2048, 2048, 8, 2, 2),
        (512, 2048, 2048, 8, 2, 4),
        (1024, 7168, 2048, 8, 2, 2),
        (1024, 7168, 2048, 8, 2, 4),
        (2048, 7168, 2048, 8, 2, 2),
        (2048, 7168, 2048, 8, 2, 4),
        (4096, 7168, 2048, 64, 8, 2),
        (4096, 7168, 2048, 64, 8, 4),
    ]

    print("=" * 80)
    print("FlashMoE Fused GEMM+AllReduce Strategy Sweep")
    print("=" * 80)

    all_results = []
    for cfg in configs:
        c_m, c_n, c_k, c_experts, c_topk, c_ws = cfg
        r = bench_ar_strategies(
            m=c_m,
            n=c_n,
            k=c_k,
            l=c_experts,
            top_k=c_topk,
            world_size=c_ws,
            warmup=warmup,
            iters=iters,
        )
        if r:
            all_results.append((*cfg, r))

    # Summary table
    if all_results:
        print("\n" + "=" * 80)
        print(
            f"{'M':>6} {'N':>6} {'K':>6} {'L':>4} {'k':>2} {'ws':>3} "
            f"{'S0(ms)':>8} {'S1(ms)':>8} {'Winner':>10} {'Speedup':>8}"
        )
        print("-" * 80)
        for c_m, c_n, c_k, c_experts, c_topk, c_ws, r in all_results:
            s0 = r.get(0, 0)
            s1 = r.get(1, 0)
            if s0 > 0 and s1 > 0:
                faster = 0 if s0 < s1 else 1
                speedup = max(s0, s1) / min(s0, s1)
                fname = "batch" if faster == 0 else "overlap"
                print(
                    f"{c_m:>6} {c_n:>6} {c_k:>6} {c_experts:>4} "
                    f"{c_topk:>2} {c_ws:>3} "
                    f"{s0:>8.3f} {s1:>8.3f} {fname:>10} {speedup:>7.2f}x"
                )
        print("=" * 80)


def _bench_ipc_worker(m, n, k, num_experts, top_k, world_size, warmup, iters):
    """MPI worker: benchmark 11-warp kernel with real IPC AllReduce.

    Each rank allocates MoEEpAllReduceMnnvlMemory, runs the kernel with
    actual IPC pointers, and measures real cross-GPU AllReduce latency.
    """
    try:
        return _bench_ipc_worker_impl(m, n, k, num_experts, top_k, world_size, warmup, iters)
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def _bench_ipc_worker_impl(m, n, k, num_experts, top_k, world_size, warmup, iters):
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from tensorrt_llm._mnnvl_utils import MoEEpAllReduceMnnvlMemory
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
    )
    from tensorrt_llm.mapping import Mapping

    try:
        from mpi4py.MPI import COMM_WORLD
    except ImportError:
        print("mpi4py not available, cannot run IPC benchmark")
        return None

    rank = COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # EP mapping: all ranks in one EP group
    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    mapping.rank = rank

    sf_vec_size = 16
    tile_size = 128
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size
    tile_n = mma_tiler_mn[1]
    num_2d_tiles = num_tiles * (n // tile_n)

    # Allocate NVLS IPC memory (cross-rank fabric handles via MPI allgather)
    staging_bytes = m * n * 2  # bf16
    output_bytes = m * n * 2
    barrier_bytes = max(num_2d_tiles * 4, 4096)

    staging_mem = MoEEpAllReduceMnnvlMemory(mapping, staging_bytes)
    output_mem = MoEEpAllReduceMnnvlMemory(mapping, output_bytes)
    tile_barrier_mem = MoEEpAllReduceMnnvlMemory(mapping, barrier_bytes)
    completion_barrier_mem = MoEEpAllReduceMnnvlMemory(mapping, barrier_bytes)

    staging_tensor = staging_mem.as_torch_strided_tensor(torch.bfloat16)
    output_tensor = output_mem.as_torch_strided_tensor(torch.bfloat16)
    tile_barrier_tensor = tile_barrier_mem.as_torch_strided_tensor(torch.int32)

    ep_comm = MoEEpAllReduceMnnvlMemory.get_comm(mapping)

    # Generate per-rank GEMM data
    with torch.device(device):
        torch.manual_seed(42 + rank)
        a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8)
        b = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8)
        a_sf = torch.randint(0, 8, (m * scale_k,), dtype=torch.uint8)
        b_sf = torch.randint(0, 8, (num_experts, n, scale_k), dtype=torch.uint8)
        alpha = torch.ones(num_experts, dtype=torch.float32) * 0.1
        tile_idx_to_expert_idx = torch.arange(num_tiles, dtype=torch.int32) % num_experts
        tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32)
        permuted_idx = torch.arange(m, dtype=torch.int32)
        num_non_exit = torch.tensor([num_tiles], dtype=torch.int32)
        token_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        staging_gemm = torch.zeros(m, n, dtype=torch.bfloat16)
        out_test = torch.zeros(num_tokens, n, dtype=torch.bfloat16)

    # Populate staging with known data for each rank
    torch.manual_seed(100 + rank)
    staging_data = torch.randn(m * n, dtype=torch.bfloat16, device=device) * 0.01
    staging_tensor[rank, : m * n].copy_(staging_data)
    torch.cuda.synchronize(device)
    ep_comm.barrier()

    make_ptr = cutlass.cute.runtime.make_ptr
    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])

    a_ptr = make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    a_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN, a_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN, b_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    tig_ptr = make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem)
    tmn_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem)
    pie_ptr = make_ptr(cutlass.Int32, permuted_idx.data_ptr(), cute.AddressSpace.gmem)
    nne_ptr = make_ptr(cutlass.Int32, num_non_exit.data_ptr(), cute.AddressSpace.gmem)
    tfs_ptr = make_ptr(cutlass.Float32, token_scales.data_ptr(), cute.AddressSpace.gmem)
    stg_ptr = make_ptr(
        cutlass.BFloat16, staging_gemm.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    out_ptr = make_ptr(
        cutlass.BFloat16, out_test.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )

    results = {}
    for strategy in [0, 1]:
        # Compile kernel for this strategy
        ar_kernel = Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_blkred=True,
            raster_along_m=False,
        )
        compile_args = [
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            stg_ptr,
            alpha_ptr,
            tig_ptr,
            tmn_ptr,
            pie_ptr,
            nne_ptr,
            tfs_ptr,
            staging_mem.ptr,
            output_mem.ptr,
            tile_barrier_mem.ptr,
            completion_barrier_mem.ptr,
            staging_mem.rank_stride,
            output_mem.rank_stride,
            out_ptr,
            m,
            n,
            k,
            num_experts,
            num_tokens,
            top_k,
            rank,
            world_size,
            strategy,
        ]
        compiled = cute.compile(
            ar_kernel.wrapper,
            *compile_args,
            tile_size=tile_size,
            scaling_vector_size=sf_vec_size,
            max_active_clusters=max_clusters,
            stream=stream,
        )
        # Runtime args (Constexpr rank/world_size/strategy are baked in)
        run_args = [
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            stg_ptr,
            alpha_ptr,
            tig_ptr,
            tmn_ptr,
            pie_ptr,
            nne_ptr,
            tfs_ptr,
            staging_mem.ptr,
            output_mem.ptr,
            tile_barrier_mem.ptr,
            completion_barrier_mem.ptr,
            staging_mem.rank_stride,
            output_mem.rank_stride,
            out_ptr,
            m,
            n,
            k,
            num_experts,
            num_tokens,
            top_k,
        ]

        # Warmup (all ranks synchronize between iterations via barrier)
        for _ in range(warmup):
            # Reset staging + barriers
            staging_tensor[rank, : m * n].copy_(staging_data)
            output_tensor.zero_()
            torch.cuda.synchronize(device)
            ep_comm.barrier()
            if rank == 0:
                barrier_view = tile_barrier_tensor[0]
                barrier_view[:num_2d_tiles] = world_size
            torch.cuda.synchronize(device)
            ep_comm.barrier()
            compiled(*run_args, stream=stream)
            torch.cuda.synchronize(device)
            ep_comm.barrier()

        # Timed runs
        times_ms = []
        for _ in range(iters):
            # Reset barriers and staging for each iteration
            staging_tensor[rank, : m * n].copy_(staging_data)
            output_tensor.zero_()
            torch.cuda.synchronize(device)
            ep_comm.barrier()
            if rank == 0:
                barrier_view = tile_barrier_tensor[0]
                barrier_view[:num_2d_tiles] = world_size
            torch.cuda.synchronize(device)
            ep_comm.barrier()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled(*run_args, stream=stream)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
            ep_comm.barrier()

        avg_ms = sum(times_ms) / len(times_ms)
        median_ms = sorted(times_ms)[len(times_ms) // 2]
        min_ms = min(times_ms)
        results[strategy] = {
            "avg": avg_ms,
            "median": median_ms,
            "min": min_ms,
        }
        if rank == 0:
            sname = "batch" if strategy == 0 else "overlapped"
            print(
                f"    Strategy {strategy} ({sname:>10}): "
                f"avg={avg_ms:.3f}ms  median={median_ms:.3f}ms  min={min_ms:.3f}ms"
            )

    if rank == 0 and len(results) == 2:
        s0_avg = results[0]["avg"]
        s1_avg = results[1]["avg"]
        faster = 0 if s0_avg < s1_avg else 1
        slower = 1 - faster
        speedup = results[slower]["avg"] / results[faster]["avg"]
        fname = "batch" if faster == 0 else "overlapped"
        print(f"    Winner: strategy {faster} ({fname}), {speedup:.2f}x faster")

    return results if rank == 0 else None


def bench_ipc(m, n, k, num_experts, top_k, world_size, warmup=10, iters=50):
    """Launch real IPC benchmark via MPIPoolExecutor."""

    from mpi4py.futures import MPIPoolExecutor

    if world_size is None:
        world_size = min(torch.cuda.device_count(), 4)

    print(
        f"\n=== Real IPC Benchmark: M={m}, N={n}, K={k}, L={num_experts}, world_size={world_size} ==="
    )
    print(f"    warmup={warmup}, iters={iters}")

    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = list(
            executor.map(
                _bench_ipc_worker,
                *zip(*[(m, n, k, num_experts, top_k, world_size, warmup, iters)] * world_size),
            )
        )
    # Only rank 0 returns non-None
    return next((r for r in results if r is not None), None)


def bench_ipc_sweep(warmup=5, iters=30):
    """Multi-config sweep with real IPC AllReduce."""

    from mpi4py.futures import MPIPoolExecutor

    world_size = min(torch.cuda.device_count(), 4)

    configs = [
        # (m, n, k, num_experts, top_k)
        (256, 2048, 2048, 8, 2),
        (512, 2048, 2048, 8, 2),
        (1024, 2048, 2048, 8, 2),
        (1024, 7168, 2048, 8, 2),
        (2048, 7168, 2048, 8, 2),
        (4096, 7168, 2048, 64, 8),
    ]

    print("=" * 90)
    print(f"FlashMoE Real IPC AllReduce Sweep  (world_size={world_size})")
    print("=" * 90)

    all_results = []
    for c_m, c_n, c_k, c_exp, c_topk in configs:
        print(f"\n=== M={c_m}, N={c_n}, K={c_k}, L={c_exp}, top_k={c_topk}, ws={world_size} ===")

        with MPIPoolExecutor(max_workers=world_size) as executor:
            results = list(
                executor.map(
                    _bench_ipc_worker,
                    *zip(*[(c_m, c_n, c_k, c_exp, c_topk, world_size, warmup, iters)] * world_size),
                )
            )
        r = next((x for x in results if x is not None), None)
        if r:
            all_results.append((c_m, c_n, c_k, c_exp, c_topk, world_size, r))

    # Summary table
    if all_results:
        print("\n" + "=" * 90)
        print(
            f"{'M':>6} {'N':>6} {'K':>6} {'L':>4} {'k':>2} {'ws':>3} "
            f"{'S0_avg':>8} {'S1_avg':>8} {'S0_med':>8} {'S1_med':>8} "
            f"{'Winner':>10} {'Speedup':>8}"
        )
        print("-" * 90)
        for c_m, c_n, c_k, c_exp, c_topk, c_ws, r in all_results:
            s0 = r.get(0, {})
            s1 = r.get(1, {})
            s0_avg = s0.get("avg", 0)
            s1_avg = s1.get("avg", 0)
            s0_med = s0.get("median", 0)
            s1_med = s1.get("median", 0)
            if s0_avg > 0 and s1_avg > 0:
                faster = 0 if s0_avg < s1_avg else 1
                speedup = max(s0_avg, s1_avg) / min(s0_avg, s1_avg)
                fname = "batch" if faster == 0 else "overlap"
                print(
                    f"{c_m:>6} {c_n:>6} {c_k:>6} {c_exp:>4} "
                    f"{c_topk:>2} {c_ws:>3} "
                    f"{s0_avg:>8.3f} {s1_avg:>8.3f} "
                    f"{s0_med:>8.3f} {s1_med:>8.3f} "
                    f"{fname:>10} {speedup:>7.2f}x"
                )
        print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="FlashMoE EP benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        default="kernel",
        choices=["kernel", "strategy", "sweep", "ipc", "ipc_sweep"],
        help="kernel: overhead test, strategy: simulated multi-GPU, "
        "sweep: multi-config simulated, ipc: real IPC multi-GPU, "
        "ipc_sweep: multi-config real IPC",
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "kernel":
        bench_kernel_overhead(
            m=args.m,
            n=args.n,
            k=args.k,
            l=args.experts,
            top_k=args.top_k,
            tile_size=args.tile_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    elif args.mode == "strategy":
        bench_ar_strategies(
            m=args.m,
            n=args.n,
            k=args.k,
            l=args.experts,
            top_k=args.top_k,
            tile_size=args.tile_size,
            world_size=args.world_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    elif args.mode == "sweep":
        bench_sweep(warmup=args.warmup, iters=args.iters)
    elif args.mode == "ipc":
        bench_ipc(
            m=args.m,
            n=args.n,
            k=args.k,
            num_experts=args.experts,
            top_k=args.top_k,
            world_size=args.world_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    elif args.mode == "ipc_sweep":
        bench_ipc_sweep(warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
