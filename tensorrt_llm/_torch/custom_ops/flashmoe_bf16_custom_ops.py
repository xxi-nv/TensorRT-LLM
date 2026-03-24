# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""FlashMoE BF16 Grouped GEMM Custom Ops.

These ops encapsulate the bf16 grouped GEMM fusion boundary for FlashMoE:
- FC1: Gather + GroupedGEMM + SwiGLU
- FC2: GroupedGEMM + Scale + Scatter-Add

Two compute paths:
1. cuteDSL (SM100+): Uses compiled kernels for GEMM operations:
   - FC1: Pre-gather + Sm100Bf16ContiguousGroupedGemmKernel + Python SwiGLU
   - FC2: Sm100Bf16ContiguousGroupedGemmFinalizeFusionKernel
     (GEMM + scale + scatter-add in one kernel)
2. Fallback (all GPUs): torch.mm() per expert with expert-merged ranges.

The cuteDSL path is automatically selected on SM100+ when cutlass-dsl is available.
"""

import os
from functools import lru_cache

import torch

# Global cache for compiled cuteDSL kernels
_compiled_kernel_cache = {}
_compiled_finalize_kernel_cache = {}
_compiled_gather_swiglu_kernel_cache = {}

# Environment variable to force-disable cuteDSL path
_FORCE_DISABLE_CUTEDSL = os.environ.get("FLASHMOE_DISABLE_CUTEDSL", "0") == "1"


@lru_cache(maxsize=1)
def _cutedsl_available() -> bool:
    """Check if cuteDSL bf16 grouped GEMM is available (SM100+ and cutlass-dsl)."""
    if _FORCE_DISABLE_CUTEDSL:
        return False
    try:
        from ..._utils import get_sm_version

        sm_version = get_sm_version()
        if sm_version < 100:
            return False
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        import cutlass.utils.blackwell_helpers  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


def _run_cutedsl_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    num_experts: int,
    tile_size: int,
):
    """Run the cuteDSL bf16 grouped GEMM kernel.

    Args:
        a: [M, K] bf16, pre-gathered activations (contiguous, K-major)
        b: [num_experts, N, K] bf16, expert weights (contiguous within each expert, K-major)
        c: [M, N] bf16, output buffer (pre-allocated)
        tile_idx_to_expert_idx: [num_tiles] int32
        num_non_exiting_tiles: [1] int32
        num_experts: number of experts
        tile_size: MMA tile M dimension
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.utils import make_ptr

    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    # Alpha = 1.0 for all experts
    alpha = torch.ones(num_experts, dtype=torch.float32, device=a.device)

    # Get current CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # Create pointer objects (same pattern as cute_dsl_custom_ops.py)
    a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(cutlass.BFloat16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    tile_ptr = make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem)
    nnet_ptr = make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem)

    mma_tiler_mn = (tile_size, 128)
    cluster_shape_mn = (tile_size // 128, 1)

    cache_key = (tile_size,)
    if cache_key not in _compiled_kernel_cache:
        from ..cute_dsl_kernels.blackwell.bf16_contiguous_grouped_gemm import (
            Sm100Bf16ContiguousGroupedGemmKernel,
        )

        gemm = Sm100Bf16ContiguousGroupedGemmKernel(
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        compiled_gemm = cute.compile(
            gemm.wrapper,
            a_ptr,
            b_ptr,
            c_ptr,
            alpha_ptr,
            tile_ptr,
            nnet_ptr,
            m,
            n,
            k,
            num_experts,
            tile_size=tile_size,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        _compiled_kernel_cache[cache_key] = compiled_gemm
    else:
        compiled_gemm = _compiled_kernel_cache[cache_key]

    compiled_gemm(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha_ptr,
        tile_ptr,
        nnet_ptr,
        m,
        n,
        k,
        num_experts,
        stream=stream,
    )


def _merge_expert_ranges(tile_idx_to_expert_idx, tile_idx_to_mn_limit, n_valid_tiles, tile_size):
    """Merge consecutive tiles belonging to the same expert into ranges.

    Since moe_sort groups tiles by expert, consecutive tiles for the same
    expert can be merged into a single contiguous range for one GEMM call.

    Returns list of (expert_idx, row_start, row_end) tuples.
    """
    if n_valid_tiles == 0:
        return []

    ranges = []
    expert_ids = tile_idx_to_expert_idx[:n_valid_tiles]
    mn_limits = tile_idx_to_mn_limit[:n_valid_tiles]

    cur_expert = expert_ids[0].item()
    cur_start = 0
    cur_end = mn_limits[0].item()

    for tile_idx in range(1, n_valid_tiles):
        expert_idx = expert_ids[tile_idx].item()
        mn_limit = mn_limits[tile_idx].item()

        if expert_idx == cur_expert:
            # Same expert: extend the range
            cur_end = mn_limit
        else:
            # Different expert: emit the current range and start new one
            ranges.append((cur_expert, cur_start, cur_end))
            cur_expert = expert_idx
            cur_start = tile_idx * tile_size
            cur_end = mn_limit

    ranges.append((cur_expert, cur_start, cur_end))
    return ranges


def _fc1_torch_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    top_k: int,
    tile_size: int,
    is_gated_activation: bool,
) -> torch.Tensor:
    """FC1 implementation using torch.mm() per expert (fallback path)."""
    total_permuted = permuted_idx_to_expanded_idx.shape[0]
    interm_size = weight.shape[1] // 2
    output = torch.zeros(total_permuted, interm_size, dtype=input.dtype, device=input.device)
    n_valid_tiles = num_non_exiting_tiles.item()

    expert_ranges = _merge_expert_ranges(
        tile_idx_to_expert_idx, tile_idx_to_mn_limit, n_valid_tiles, tile_size
    )

    for local_expert_idx, row_start, row_end in expert_ranges:
        perm_indices = permuted_idx_to_expanded_idx[row_start:row_end]
        token_indices = perm_indices // top_k
        gathered_input = input[token_indices]

        gate_up = torch.mm(gathered_input, weight[local_expert_idx].t())

        up_proj, gate_proj = gate_up.chunk(2, dim=-1)
        if is_gated_activation:
            activated = torch.nn.functional.silu(gate_proj) * up_proj
        else:
            activated = up_proj

        output[row_start:row_end] = activated

    return output


def _run_cutedsl_gather_gemm_swiglu(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    token_id_mapping: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    num_experts: int,
    top_k: int,
    tile_size: int,
):
    """Run the cuteDSL fused gather + GEMM + SwiGLU kernel.

    Args:
        input: [orig_m, K] bf16, un-gathered input tokens
        weight: [num_experts, 2*I, K] bf16, interleaved gate+up weights
        output: [M, I] bf16, output buffer (pre-allocated)
        tile_idx_to_expert_idx: [num_tiles] int32
        tile_idx_to_mn_limit: [num_tiles] int32
        token_id_mapping: [M] int32, token indices for gather (clamped)
        num_non_exiting_tiles: [1] int32
        num_experts: number of experts
        top_k: number of experts per token
        tile_size: MMA tile M dimension
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.utils import make_ptr

    orig_m = input.shape[0]
    k = input.shape[1]
    m = output.shape[0]  # total_permuted (padded)
    n = weight.shape[1]  # 2 * interm_size

    # Alpha = 1.0 for all experts
    alpha = torch.ones(num_experts, dtype=torch.float32, device=input.device)

    # Get current CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # Create pointer objects
    a_ptr = make_ptr(cutlass.BFloat16, input.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(cutlass.BFloat16, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    tile_expert_ptr = make_ptr(
        cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
    )
    tile_mn_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem)
    token_map_ptr = make_ptr(cutlass.Int32, token_id_mapping.data_ptr(), cute.AddressSpace.gmem)
    nnet_ptr = make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem)

    mma_tiler_mn = (tile_size, 128)
    cluster_shape_mn = (tile_size // 128, 1)

    cache_key = ("gather_swiglu", tile_size, top_k)
    if cache_key not in _compiled_gather_swiglu_kernel_cache:
        from ..cute_dsl_kernels.blackwell.bf16_contiguous_gather_grouped_gemm_swiglu_fusion import (
            Sm100Bf16ContiguousGatherGroupedGemmSwigluFusionKernel,
        )

        gemm = Sm100Bf16ContiguousGatherGroupedGemmSwigluFusionKernel(
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        compiled_gemm = cute.compile(
            gemm.wrapper,
            a_ptr,
            b_ptr,
            c_ptr,
            alpha_ptr,
            tile_expert_ptr,
            tile_mn_ptr,
            token_map_ptr,
            nnet_ptr,
            orig_m,
            m,
            n,
            k,
            num_experts,
            tile_size=tile_size,
            top_k=top_k,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        _compiled_gather_swiglu_kernel_cache[cache_key] = compiled_gemm
    else:
        compiled_gemm = _compiled_gather_swiglu_kernel_cache[cache_key]

    compiled_gemm(
        a_ptr,
        b_ptr,
        c_ptr,
        alpha_ptr,
        tile_expert_ptr,
        tile_mn_ptr,
        token_map_ptr,
        nnet_ptr,
        orig_m,
        m,
        n,
        k,
        num_experts,
        stream=stream,
    )


def _fc1_cutedsl(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    top_k: int,
    tile_size: int,
    is_gated_activation: bool,
) -> torch.Tensor:
    """FC1 implementation using cuteDSL grouped GEMM kernel.

    Pre-gather in Python + cuteDSL GEMM + Python SwiGLU.
    The fused gather+SwiGLU kernel (11 warps) exceeds SM100's register
    budget for bf16; this 7-warp path trades 2 extra Python ops for
    reliable compilation.
    """
    total_permuted = permuted_idx_to_expanded_idx.shape[0]
    n_2i = weight.shape[1]
    num_experts = weight.shape[0]

    # Step 1: Pre-gather tokens into contiguous buffer
    # Clamp indices for safe gather: padding positions in
    # permuted_idx_to_expanded_idx may contain sentinel values.
    # The kernel only reads valid tile rows, so padding data in
    # gathered_a is harmless (never contributes to final output).
    token_indices = permuted_idx_to_expanded_idx // top_k
    token_indices = token_indices.clamp(0, input.shape[0] - 1)
    gathered_a = input[token_indices]  # [total_permuted, hidden_size]

    # Step 2: Run cuteDSL grouped GEMM
    # A: [total_permuted, hidden_size]
    # B: [num_experts, 2*I, hidden_size] - already in correct layout
    # C: [total_permuted, 2*I]
    gate_up = torch.empty(total_permuted, n_2i, dtype=input.dtype, device=input.device)

    _run_cutedsl_gemm(
        a=gathered_a,
        b=weight,
        c=gate_up,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        num_experts=num_experts,
        tile_size=tile_size,
    )

    # Step 3: SwiGLU activation on all rows (including padding)
    # FC2 needs the full padded tensor. Padding rows produce garbage
    # but FC2's scatter step only uses valid rows.
    up_proj, gate_proj = gate_up.chunk(2, dim=-1)
    if is_gated_activation:
        output = torch.nn.functional.silu(gate_proj) * up_proj
    else:
        output = up_proj

    return output


def _fc2_torch_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tile_size: int,
) -> None:
    """FC2 implementation using torch.mm() per expert (fallback path)."""
    n_valid_tiles = num_non_exiting_tiles.item()
    flat_scales = token_final_scales.float().view(-1)

    expert_ranges = _merge_expert_ranges(
        tile_idx_to_expert_idx, tile_idx_to_mn_limit, n_valid_tiles, tile_size
    )

    for local_expert_idx, row_start, row_end in expert_ranges:
        expert_input = input[row_start:row_end]
        expert_output = torch.mm(expert_input, weight[local_expert_idx].t())

        perm_indices = permuted_idx_to_expanded_idx[row_start:row_end]
        token_indices = perm_indices // top_k
        scales = flat_scales[perm_indices].unsqueeze(1)
        scaled_output = expert_output * scales.to(expert_output.dtype)
        output.index_add_(0, token_indices, scaled_output)


def _run_cutedsl_finalize_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    tile_size: int,
):
    """Run the cuteDSL fused finalize GEMM kernel (GEMM + scale + scatter-add).

    Args:
        a: [M, K] bf16, permuted activations (contiguous, K-major)
        b: [num_experts, N, K] bf16, expert weights (K-major within each expert)
        out: [num_tokens, N] bf16, output buffer (pre-zeroed, scatter-add target)
        tile_idx_to_expert_idx: [num_tiles] int32
        tile_idx_to_mn_limit: [num_tiles] int32
        permuted_idx_to_expanded_idx: [M] int32
        num_non_exiting_tiles: [1] int32
        token_final_scales: [num_tokens, top_k] float32
        num_experts: number of experts
        num_tokens: number of output tokens
        top_k: number of experts per token
        tile_size: MMA tile M dimension
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.utils import make_ptr

    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    # Alpha = 1.0 for all experts
    alpha = torch.ones(num_experts, dtype=torch.float32, device=a.device)

    # Ensure token_final_scales is [num_tokens, top_k] and float32
    scales = token_final_scales.float().view(num_tokens, top_k).contiguous()

    # Get current CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # Create pointer objects
    a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    out_ptr = make_ptr(cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
    tile_expert_ptr = make_ptr(
        cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
    )
    tile_mn_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem)
    perm_ptr = make_ptr(
        cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
    )
    nnet_ptr = make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem)
    scales_ptr = make_ptr(cutlass.Float32, scales.data_ptr(), cute.AddressSpace.gmem)

    mma_tiler_mn = (tile_size, 128)
    cluster_shape_mn = (tile_size // 128, 1)

    cache_key = ("finalize", tile_size)
    if cache_key not in _compiled_finalize_kernel_cache:
        from ..cute_dsl_kernels.blackwell.bf16_contiguous_grouped_gemm_finalize_fusion import (
            Sm100Bf16ContiguousGroupedGemmFinalizeFusionKernel,
        )

        gemm = Sm100Bf16ContiguousGroupedGemmFinalizeFusionKernel(
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        compiled_gemm = cute.compile(
            gemm.wrapper,
            a_ptr,
            b_ptr,
            out_ptr,
            alpha_ptr,
            tile_expert_ptr,
            tile_mn_ptr,
            perm_ptr,
            nnet_ptr,
            scales_ptr,
            m,
            n,
            k,
            num_experts,
            num_tokens,
            top_k,
            tile_size=tile_size,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        _compiled_finalize_kernel_cache[cache_key] = compiled_gemm
    else:
        compiled_gemm = _compiled_finalize_kernel_cache[cache_key]

    compiled_gemm(
        a_ptr,
        b_ptr,
        out_ptr,
        alpha_ptr,
        tile_expert_ptr,
        tile_mn_ptr,
        perm_ptr,
        nnet_ptr,
        scales_ptr,
        m,
        n,
        k,
        num_experts,
        num_tokens,
        top_k,
        stream=stream,
    )


def _fc2_cutedsl(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tile_size: int,
) -> None:
    """FC2 implementation using cuteDSL fused finalize kernel.

    Fuses GEMM + scale + scatter-add into a single kernel launch.
    No intermediate gemm_output buffer needed.
    """
    num_tokens = output.shape[0]
    num_experts = weight.shape[0]

    _run_cutedsl_finalize_gemm(
        a=input,
        b=weight,
        out=output,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        tile_size=tile_size,
    )


@torch.library.custom_op(
    "trtllm::flashmoe_bf16_gather_gemm_swiglu", mutates_args=(), device_types="cuda"
)
def flashmoe_bf16_gather_gemm_swiglu(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    top_k: int,
    tile_size: int,
    is_gated_activation: bool,
) -> torch.Tensor:
    """FC1: Gather + GroupedGEMM + SwiGLU for bf16 FlashMoE.

    Automatically selects cuteDSL kernel path on SM100+ or falls back to
    torch.mm() per expert.

    Args:
        input: [num_tokens, hidden_size], bf16
        weight: [num_local_experts, 2*intermediate_size, hidden_size], bf16
        tile_idx_to_expert_idx: [num_tiles], int32 (local expert indices from moe_sort)
        tile_idx_to_mn_limit: [num_tiles], int32 (absolute cumulative boundary from moe_sort)
        permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
        num_non_exiting_tiles: scalar int32
        top_k: number of experts per token
        tile_size: MMA tile M dimension (128 or 256)
        is_gated_activation: whether to apply SwiGLU (True) or identity (False)
    Returns:
        fc1_output: [total_permuted_tokens, intermediate_size], bf16
    """
    if _cutedsl_available():
        return _fc1_cutedsl(
            input,
            weight,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            top_k,
            tile_size,
            is_gated_activation,
        )
    return _fc1_torch_fallback(
        input,
        weight,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        top_k,
        tile_size,
        is_gated_activation,
    )


@torch.library.register_fake("trtllm::flashmoe_bf16_gather_gemm_swiglu")
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    top_k: int,
    tile_size: int,
    is_gated_activation: bool,
) -> torch.Tensor:
    total_permuted = permuted_idx_to_expanded_idx.size(0)
    interm_size = weight.size(1) // 2
    return torch.empty(total_permuted, interm_size, dtype=input.dtype, device=input.device)


@torch.library.custom_op(
    "trtllm::flashmoe_bf16_gemm_finalize_inplace", mutates_args=("output",), device_types="cuda"
)
def flashmoe_bf16_gemm_finalize_inplace(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tile_size: int,
) -> None:
    """FC2: GroupedGEMM + Scale + Scatter-Add for bf16 FlashMoE (in-place).

    Automatically selects cuteDSL kernel path on SM100+ or falls back to
    torch.mm() per expert.

    Args:
        input: [total_permuted_tokens, intermediate_size], bf16
        weight: [num_local_experts, hidden_size, intermediate_size], bf16
        output: [num_tokens, hidden_size], bf16 (pre-zeroed, mutated in-place)
        tile_idx_to_expert_idx: [num_tiles], int32 (local expert indices from moe_sort)
        tile_idx_to_mn_limit: [num_tiles], int32 (absolute cumulative boundary from moe_sort)
        permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
        num_non_exiting_tiles: scalar int32
        token_final_scales: [num_tokens, top_k] or [num_tokens * top_k], float32
        top_k: number of experts per token
        tile_size: MMA tile M dimension (128 or 256)
    """
    if _cutedsl_available():
        _fc2_cutedsl(
            input,
            weight,
            output,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            token_final_scales,
            top_k,
            tile_size,
        )
    else:
        _fc2_torch_fallback(
            input,
            weight,
            output,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            token_final_scales,
            top_k,
            tile_size,
        )


@torch.library.register_fake("trtllm::flashmoe_bf16_gemm_finalize_inplace")
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tile_size: int,
) -> None:
    pass


@torch.library.custom_op(
    "trtllm::flashmoe_bf16_gemm_finalize", mutates_args=(), device_types="cuda"
)
def flashmoe_bf16_gemm_finalize(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tile_size: int,
    num_tokens: int,
    hidden_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """FC2: Non-inplace version that allocates and returns output."""
    output = torch.zeros(num_tokens, hidden_size, dtype=output_dtype, device=input.device)
    torch.ops.trtllm.flashmoe_bf16_gemm_finalize_inplace(
        input=input,
        weight=weight,
        output=output,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        token_final_scales=token_final_scales,
        top_k=top_k,
        tile_size=tile_size,
    )
    return output


@torch.library.register_fake("trtllm::flashmoe_bf16_gemm_finalize")
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    top_k: int,
    tile_size: int,
    num_tokens: int,
    hidden_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(num_tokens, hidden_size, dtype=output_dtype, device=input.device)
