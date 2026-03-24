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
1. cuteDSL (SM100+): Uses Sm100Bf16ContiguousGroupedGemmKernel for the GEMM
   with pre/post PyTorch operations for gather/SwiGLU/scatter-add.
2. Fallback (all GPUs): torch.mm() per expert with expert-merged ranges.

The cuteDSL path is automatically selected on SM100+ when cutlass-dsl is available.
"""

import os
from functools import lru_cache

import torch

# Global cache for compiled cuteDSL kernels
_compiled_kernel_cache = {}

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
    torch.cuda.synchronize()

    # --- DEBUG: Compare cuteDSL output with torch.mm reference ---
    import os

    if os.environ.get("FLASHMOE_DEBUG_GEMM", "0") == "1":
        n_valid_tiles_val = num_non_exiting_tiles.item()
        tile_experts = tile_idx_to_expert_idx[:n_valid_tiles_val].cpu().tolist()
        c_ref = torch.zeros_like(c)
        row = 0
        for t in range(n_valid_tiles_val):
            expert = tile_experts[t]
            row_end = min(row + tile_size, m)
            a_tile = a[row:row_end]
            b_expert = b[expert]  # [N, K]
            c_ref[row:row_end] = torch.mm(a_tile, b_expert.t())
            row = row_end
        c_out_valid = c[:row].float()
        c_ref_valid = c_ref[:row].float()
        max_abs_err = (c_out_valid - c_ref_valid).abs().max().item()
        c_out_norm = c_out_valid.abs().mean().item()
        c_ref_norm = c_ref_valid.abs().mean().item()
        print(
            f"[GEMM DEBUG] m={m} n={n} k={k} L={num_experts} "
            f"n_tiles={n_valid_tiles_val} valid_rows={row} "
            f"c_out_norm={c_out_norm:.6f} c_ref_norm={c_ref_norm:.6f} "
            f"max_abs_err={max_abs_err:.6f}"
        )
    # --- END DEBUG ---


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
    """FC1 implementation using cuteDSL grouped GEMM kernel."""
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
    """FC2 implementation using cuteDSL grouped GEMM kernel."""
    total_permuted = permuted_idx_to_expanded_idx.shape[0]
    hidden_size = weight.shape[1]
    num_experts = weight.shape[0]

    # Step 1: Run cuteDSL grouped GEMM
    # A: [total_permuted, interm_size]
    # B: [num_experts, hidden_size, interm_size] - already in correct layout
    # C: [total_permuted, hidden_size]
    gemm_output = torch.empty(total_permuted, hidden_size, dtype=input.dtype, device=input.device)

    _run_cutedsl_gemm(
        a=input,
        b=weight,
        c=gemm_output,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        num_experts=num_experts,
        tile_size=tile_size,
    )

    # Step 2: Scale by routing weights and scatter-add (valid rows only)
    # moe_sort pads each tile to tile_size; padding rows have sentinel
    # indices and must NOT participate in scatter-add.
    # tile_idx_to_mn_limit[t] is the absolute position in the padded
    # layout where real rows end for tile t. For example, with tile_size=128
    # and 2 tiles with 1 real token each: mn_limit = [1, 129].
    # Build a boolean mask over all positions identifying valid rows.
    n_valid_tiles = num_non_exiting_tiles.item()
    if n_valid_tiles == 0:
        return

    # Vectorized valid-row mask: position p is valid iff p < mn_limit[p // tile_size]
    positions = torch.arange(total_permuted, device=gemm_output.device)
    tile_indices = positions // tile_size
    # Only check tiles within n_valid_tiles; padding tiles are invalid
    valid_mask = (tile_indices < n_valid_tiles) & (
        positions < tile_idx_to_mn_limit[tile_indices.clamp(max=n_valid_tiles - 1)]
    )

    valid_positions = valid_mask.nonzero(as_tuple=True)[0]
    flat_scales = token_final_scales.float().view(-1)
    perm_indices = permuted_idx_to_expanded_idx[valid_positions]
    token_indices = perm_indices // top_k
    scales = flat_scales[perm_indices].unsqueeze(1)
    scaled_output = gemm_output[valid_positions] * scales.to(gemm_output.dtype)
    output.index_add_(0, token_indices, scaled_output)


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
