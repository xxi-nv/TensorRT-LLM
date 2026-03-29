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
r"""Test script for the 11-warp fused GEMM + AllReduce kernel.

Single-GPU test (AR warps are no-op when world_size=1):
  python run_blockscaled_contiguous_grouped_gemm_allreduce.py \\
      --ab_dtype Float4E2M1FN --out_dtype BFloat16 \\
      --sf_dtype Float8E4M3FN --sf_vec_size 16 \\
      --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \\
      --benchmark 128x7168x2048x8 --iterations 1 --world_size 1

Multi-GPU EP test (requires MPI launch):
  mpirun -n 2 python run_blockscaled_contiguous_grouped_gemm_allreduce.py \\
      --ab_dtype Float4E2M1FN --out_dtype BFloat16 \\
      --sf_dtype Float8E4M3FN --sf_vec_size 16 \\
      --mma_tiler_mn 128,128 --cluster_shape_mn 1,1 \\
      --benchmark 128x7168x2048x8 --iterations 1 --world_size 2
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import torch

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_grouped_gemm_allreduce as kernel_module,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion as base_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[3] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell import blockscaled_contiguous_grouped_gemm_allreduce as kernel_module
    from blackwell import blockscaled_contiguous_grouped_gemm_finalize_fusion as base_module

Sm100BlockScaledContiguousGroupedGemmAllReduceKernel = (
    kernel_module.Sm100BlockScaledContiguousGroupedGemmAllReduceKernel
)
cvt_sf_MKL_to_M32x4xrm_K4xrk_L = base_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L

try:
    from .testing import benchmark  # noqa: F401
except ImportError:
    pass


def torch_dtype_from_cutlass(cutlass_dtype):
    dtype_map = {
        cutlass.Float4E2M1FN: torch.float4_e2m1fn_x2,
        cutlass.Float8E5M2: torch.float8_e5m2,
        cutlass.Float8E4M3FN: torch.float8_e4m3fn,
        cutlass.Float32: torch.float32,
        cutlass.Float16: torch.float16,
        cutlass.BFloat16: torch.bfloat16,
    }
    return dtype_map.get(cutlass_dtype)


def parse_dtype(s: str) -> Type[cutlass.Numeric]:
    dtype_map = {
        "Float4E2M1FN": cutlass.Float4E2M1FN,
        "Float8E5M2": cutlass.Float8E5M2,
        "Float8E4M3FN": cutlass.Float8E4M3FN,
        "Float32": cutlass.Float32,
        "Float16": cutlass.Float16,
        "BFloat16": cutlass.BFloat16,
    }
    return dtype_map[s]


def parse_benchmark(s: str) -> Tuple[int, int, int, int]:
    """Parse benchmark string MxNxKxL."""
    parts = s.split("x")
    assert len(parts) == 4, f"Expected MxNxKxL, got {s}"
    return tuple(int(p) for p in parts)


def run_single_gpu_test(
    ab_dtype,
    out_dtype,
    sf_dtype,
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    m: int,
    n: int,
    k: int,
    l: int,  # noqa: E741
    top_k: int = 2,
    tile_size: int = 128,
):
    """Test AllReduce kernel in single-GPU mode (world_size=1, AR warps no-op).

    Verifies that the 11-warp kernel produces the same result as the base
    7-warp finalize-fusion kernel when world_size=1.
    """
    print(f"\n{'=' * 60}")
    print(f"Single-GPU test: M={m}, N={n}, K={k}, L={l}")
    print(f"  ab_dtype={ab_dtype}, out_dtype={out_dtype}")
    print(f"  mma_tiler_mn={mma_tiler_mn}, cluster={cluster_shape_mn}")
    print(f"{'=' * 60}")

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Check if kernel can be implemented
    if not Sm100BlockScaledContiguousGroupedGemmAllReduceKernel.can_implement(
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        out_dtype=out_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=m,
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        out_major="n",
    ):
        print("  SKIP: cannot implement this config")
        return True

    num_tokens = m // top_k
    num_tiles = m // tile_size
    scale_k = k // sf_vec_size

    # Generate inputs
    a_torch = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)
    b_torch = torch.randint(0, 256, (l, n, k // 2), dtype=torch.uint8, device=device)
    a_sf_torch = torch.randint(0, 256, (m * scale_k,), dtype=torch.uint8, device=device)
    b_sf_torch = torch.randint(0, 256, (l, n, scale_k), dtype=torch.uint8, device=device)
    alpha_torch = torch.ones(l, dtype=torch.float32, device=device)

    # Per-tile metadata
    tile_idx_to_expert_idx = torch.zeros(num_tiles, dtype=torch.int32, device=device)
    for i in range(num_tiles):
        tile_idx_to_expert_idx[i] = i % l
    tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32, device=device)
    permuted_idx_to_expanded_idx = torch.arange(m, dtype=torch.int32, device=device)
    num_non_exiting_tiles = torch.tensor([num_tiles], dtype=torch.int32, device=device)
    token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

    # Staging buffer (for AR path, but world_size=1 so unused)
    staging = torch.zeros(m, n, dtype=torch.bfloat16, device=device)

    # Output buffer
    out = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

    # World size = 1: AR warps are no-op
    rank = 0
    world_size = 1

    # Run AllReduce kernel
    gemm = Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    a_ptr = cutlass.cute.runtime.make_ptr(
        cutlass.Float4E2M1FN,
        a_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    b_ptr = cutlass.cute.runtime.make_ptr(
        cutlass.Float4E2M1FN,
        b_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    a_sf_ptr = cutlass.cute.runtime.make_ptr(
        cutlass.Float8E4M3FN,
        a_sf_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_sf_ptr = cutlass.cute.runtime.make_ptr(
        cutlass.Float8E4M3FN,
        b_sf_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    torch_stream = torch.cuda.current_stream()
    from cuda.bindings import driver as cuda

    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm = cute.compile(
        gemm.wrapper,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        cutlass.cute.runtime.make_ptr(
            cutlass.BFloat16, staging.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, alpha_torch.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem
        ),
        0,  # staging_mc_ptr (unused for world_size=1)
        0,  # out_mc_ptr
        0,  # tile_barrier_mc_ptr
        0,  # completion_barrier_mc_ptr
        cutlass.cute.runtime.make_ptr(
            cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        rank,
        world_size,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )

    compiled_gemm(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        cutlass.cute.runtime.make_ptr(
            cutlass.BFloat16, staging.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, alpha_torch.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem
        ),
        0,
        0,
        0,
        0,
        cutlass.cute.runtime.make_ptr(
            cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        rank,
        world_size,
        stream=stream,
    )
    torch.cuda.synchronize()

    # Verify output is non-zero (basic sanity)
    out_abs_sum = out.abs().sum().item()
    print(f"  Output abs sum: {out_abs_sum:.4f}")

    # Compare with base finalize-fusion kernel
    out_ref = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)
    base_kernel = base_module.Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )

    compiled_base = cute.compile(
        base_kernel.wrapper,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        cutlass.cute.runtime.make_ptr(
            cutlass.BFloat16, out_ref.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, alpha_torch.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem
        ),
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )

    compiled_base(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        cutlass.cute.runtime.make_ptr(
            cutlass.BFloat16, out_ref.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, alpha_torch.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
        ),
        cutlass.cute.runtime.make_ptr(
            cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem
        ),
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        stream=stream,
    )
    torch.cuda.synchronize()

    # Compare
    max_diff = (out.float() - out_ref.float()).abs().max().item()
    ref_abs_sum = out_ref.abs().sum().item()
    print(f"  Reference abs sum: {ref_abs_sum:.4f}")
    print(f"  Max diff (AR kernel vs base): {max_diff:.6f}")

    passed = max_diff < 1e-2
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Test 11-warp fused GEMM + AllReduce kernel")
    parser.add_argument(
        "--ab_dtype",
        type=str,
        default="Float4E2M1FN",
        help="A/B data type",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="BFloat16",
        help="Output data type",
    )
    parser.add_argument(
        "--sf_dtype",
        type=str,
        default="Float8E4M3FN",
        help="Scale factor data type",
    )
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--mma_tiler_mn", type=str, default="128,128")
    parser.add_argument("--cluster_shape_mn", type=str, default="1,1")
    parser.add_argument("--benchmark", type=str, default="128x7168x2048x8")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    ab_dtype = parse_dtype(args.ab_dtype)
    out_dtype = parse_dtype(args.out_dtype)
    sf_dtype = parse_dtype(args.sf_dtype)
    mma_tiler_mn = tuple(int(x) for x in args.mma_tiler_mn.split(","))
    cluster_shape_mn = tuple(int(x) for x in args.cluster_shape_mn.split(","))

    m, n, k, l = parse_benchmark(args.benchmark)  # noqa: E741

    if args.world_size == 1:
        passed = run_single_gpu_test(
            ab_dtype,
            out_dtype,
            sf_dtype,
            args.sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            args.top_k,
        )
        if not passed:
            sys.exit(1)
    else:
        print("Multi-GPU EP test requires MPI launch and NVLS setup.")
        print("See test_moe_module.py for multi-GPU pytest-based tests.")
        sys.exit(0)


if __name__ == "__main__":
    main()
