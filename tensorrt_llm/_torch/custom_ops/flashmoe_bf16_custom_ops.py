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

Uses expert-merged GEMM: tiles are grouped by expert (from moe_sort) so that
all tiles for the same expert are processed in a single torch.mm() call.
This reduces CUDA kernel launch overhead from O(num_tiles) to O(num_active_experts).

The compute path can be replaced by compiled cuteDSL kernels when bf16 grouped
GEMM support is ready (CUTLASS GroupedGemmKernel supports bf16 on SM100+).

No cutlass-dsl dependency - these ops use pure PyTorch operations.
"""

import torch


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

    Merges consecutive tiles by expert for efficient GEMM execution:
    1. Group tiles by expert (O(num_active_experts) GEMM calls instead of O(num_tiles))
    2. For each expert group: gather input tokens, GEMM, apply SwiGLU
    3. Write results to contiguous output buffer

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
    total_permuted = permuted_idx_to_expanded_idx.shape[0]
    interm_size = weight.shape[1] // 2
    output = torch.zeros(total_permuted, interm_size, dtype=input.dtype, device=input.device)
    n_valid_tiles = num_non_exiting_tiles.item()

    expert_ranges = _merge_expert_ranges(
        tile_idx_to_expert_idx, tile_idx_to_mn_limit, n_valid_tiles, tile_size
    )

    for local_expert_idx, row_start, row_end in expert_ranges:
        # Gather input tokens for all tiles of this expert
        perm_indices = permuted_idx_to_expanded_idx[row_start:row_end]
        token_indices = perm_indices // top_k
        gathered_input = input[token_indices]

        # Single GEMM for all tokens assigned to this expert
        gate_up = torch.mm(gathered_input, weight[local_expert_idx].t())

        # SwiGLU activation
        up_proj, gate_proj = gate_up.chunk(2, dim=-1)
        if is_gated_activation:
            activated = torch.nn.functional.silu(gate_proj) * up_proj
        else:
            activated = up_proj

        output[row_start:row_end] = activated

    return output


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

    Merges consecutive tiles by expert for efficient GEMM execution:
    1. Group tiles by expert (O(num_active_experts) GEMM calls instead of O(num_tiles))
    2. For each expert group: GEMM, scale by routing weights, scatter-add

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
    n_valid_tiles = num_non_exiting_tiles.item()
    flat_scales = token_final_scales.float().view(-1)

    expert_ranges = _merge_expert_ranges(
        tile_idx_to_expert_idx, tile_idx_to_mn_limit, n_valid_tiles, tile_size
    )

    for local_expert_idx, row_start, row_end in expert_ranges:
        # Single GEMM for all tokens assigned to this expert
        expert_input = input[row_start:row_end]
        expert_output = torch.mm(expert_input, weight[local_expert_idx].t())

        # Scale by routing weights and scatter-add to output
        perm_indices = permuted_idx_to_expanded_idx[row_start:row_end]
        token_indices = perm_indices // top_k
        scales = flat_scales[perm_indices].unsqueeze(1)
        scaled_output = expert_output * scales.to(expert_output.dtype)
        output.index_add_(0, token_indices, scaled_output)


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
