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

"""CuTe DSL helpers for MegaMoE direct-input route building."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils.distributed import atomicAdd

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    make_ptr,
)

_THREADS_PER_BLOCK = 256
_direct_route_compile_cache: dict[tuple[object, ...], tuple[object, ...]] = {}


@cute.kernel
def _direct_route_init_kernel(
    expert_route_offsets: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_experts_per_rank: cutlass.Constexpr,
    route_layout_capacity: cutlass.Constexpr,
    total_work_items: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    idx = block_idx * threads_per_block + thread_idx
    stride = threads_per_block * num_blocks

    while idx < total_work_items:
        if idx < num_experts_per_rank:
            expert_route_offsets[idx] = cutlass.Int32(0)
        if idx < route_layout_capacity:
            tile_idx_to_expert_idx[idx] = cutlass.Int32(-1)
            tile_idx_to_mn_limit[idx] = cutlass.Int32(0)
        idx = idx + stride

    if block_idx == 0 and thread_idx == 0:
        num_non_exiting_tiles[0] = cutlass.Int32(0)

    griddepcontrol_launch_dependents()


@cute.kernel
def _direct_route_copy_count_kernel(
    input: cute.Tensor,
    input_sf: cute.Tensor,
    topk_idx: cute.Tensor,
    token_counts: cute.Tensor,
    direct_input: cute.Tensor,
    direct_input_sf: cute.Tensor,
    expert_route_offsets: cute.Tensor,
    ep_size: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    hidden_packed_size: cutlass.Constexpr,
    sf_hidden_size: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    max_num_tokens_per_rank: cutlass.Constexpr,
    total_routes: cutlass.Constexpr,
    max_copy_work_items: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    first_idx = block_idx * threads_per_block + thread_idx
    stride = threads_per_block * num_blocks

    for source_rank in range(ep_size):
        rank_token_count = token_counts[source_rank]
        if rank_token_count < cutlass.Int32(0):
            rank_token_count = cutlass.Int32(0)
        if rank_token_count > cutlass.Int32(max_num_tokens_per_rank):
            rank_token_count = cutlass.Int32(max_num_tokens_per_rank)

        input_idx = first_idx
        input_elements = rank_token_count * cutlass.Int32(hidden_packed_size)
        while input_idx < input_elements:
            source_token_idx = input_idx // hidden_packed_size
            hidden_idx = input_idx - source_token_idx * hidden_packed_size
            direct_input[source_rank * max_num_tokens_per_rank + source_token_idx, hidden_idx] = (
                input[
                    source_rank,
                    source_token_idx,
                    hidden_idx,
                ]
            )
            input_idx = input_idx + stride

        sf_idx = first_idx
        input_sf_elements = rank_token_count * cutlass.Int32(sf_hidden_size)
        while sf_idx < input_sf_elements:
            source_token_idx = sf_idx // sf_hidden_size
            hidden_idx = sf_idx - source_token_idx * sf_hidden_size
            direct_input_sf[
                source_rank * max_num_tokens_per_rank + source_token_idx, hidden_idx
            ] = input_sf[
                source_rank,
                source_token_idx,
                hidden_idx,
            ]
            sf_idx = sf_idx + stride

    route_idx = first_idx
    routes_per_rank = max_num_tokens_per_rank * top_k
    first_local_expert = cutlass.Int64(local_rank * num_experts_per_rank)
    last_local_expert = cutlass.Int64((local_rank + 1) * num_experts_per_rank)
    while route_idx < total_routes:
        source_rank = route_idx // routes_per_rank
        rank_route_idx = route_idx - source_rank * routes_per_rank
        source_token_idx = rank_route_idx // top_k
        topk_ordinal = rank_route_idx - source_token_idx * top_k
        token_count = token_counts[source_rank]
        if source_token_idx < token_count and source_token_idx < max_num_tokens_per_rank:
            selected_expert = topk_idx[source_rank, source_token_idx, topk_ordinal]
            if selected_expert >= first_local_expert and selected_expert < last_local_expert:
                local_expert_idx = cutlass.Int32(selected_expert - first_local_expert)
                atomicAdd(expert_route_offsets.iterator + local_expert_idx, cutlass.Int32(1))
        route_idx = route_idx + stride

    griddepcontrol_launch_dependents()


@cute.kernel
def _direct_route_prefix_tiles_kernel(
    expert_route_offsets: cute.Tensor,
    expert_route_base_offsets: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_experts_per_rank: cutlass.Constexpr,
    tile_size: cutlass.Constexpr,
    num_pool_slots: cutlass.Constexpr,
    route_layout_capacity: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    if block_idx == 0 and thread_idx == 0:
        running_offset = cutlass.Int32(0)
        for expert_idx in range(num_experts_per_rank):
            count = expert_route_offsets[expert_idx]
            if count < cutlass.Int32(0):
                count = cutlass.Int32(0)
            aligned_count = ((count + tile_size - 1) // tile_size) * tile_size
            expert_route_base_offsets[expert_idx] = running_offset
            expert_route_offsets[expert_idx] = cutlass.Int32(0)

            tile_start = running_offset // tile_size
            tile_count = aligned_count // tile_size
            routes_remaining = count
            tile_offset = cutlass.Int32(0)
            while tile_offset < tile_count:
                tile_idx = tile_start + tile_offset
                if tile_idx < route_layout_capacity:
                    tile_routes = routes_remaining
                    if tile_routes > cutlass.Int32(tile_size):
                        tile_routes = cutlass.Int32(tile_size)
                    if tile_routes < cutlass.Int32(0):
                        tile_routes = cutlass.Int32(0)
                    tile_idx_to_expert_idx[tile_idx] = cutlass.Int32(expert_idx)
                    tile_idx_to_mn_limit[tile_idx] = (
                        running_offset + tile_offset * tile_size + tile_routes
                    )
                routes_remaining = routes_remaining - tile_size
                tile_offset = tile_offset + cutlass.Int32(1)

            running_offset = running_offset + aligned_count

        if running_offset > cutlass.Int32(num_pool_slots):
            num_non_exiting_tiles[0] = (num_pool_slots + tile_size - 1) // tile_size + 1
        else:
            num_non_exiting_tiles[0] = (running_offset + tile_size - 1) // tile_size

    griddepcontrol_launch_dependents()


@cute.kernel
def _direct_route_fill_kernel(
    topk_idx: cute.Tensor,
    topk_scales: cute.Tensor,
    token_counts: cute.Tensor,
    expert_route_offsets: cute.Tensor,
    expert_route_base_offsets: cute.Tensor,
    token_id_mapping: cute.Tensor,
    output_mapping: cute.Tensor,
    output_scales: cute.Tensor,
    ep_size: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    max_num_tokens_per_rank: cutlass.Constexpr,
    num_pool_slots: cutlass.Constexpr,
    combine_layout_rows: cutlass.Constexpr,
    output_mapping_rows: cutlass.Constexpr,
    output_scale_rows: cutlass.Constexpr,
    direct_atomic_output: cutlass.Constexpr,
    direct_token_major_output: cutlass.Constexpr,
    total_routes: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    route_idx = block_idx * threads_per_block + thread_idx
    stride = threads_per_block * num_blocks
    routes_per_rank = max_num_tokens_per_rank * top_k
    first_local_expert = cutlass.Int64(local_rank * num_experts_per_rank)
    last_local_expert = cutlass.Int64((local_rank + 1) * num_experts_per_rank)

    while route_idx < total_routes:
        source_rank = route_idx // routes_per_rank
        rank_route_idx = route_idx - source_rank * routes_per_rank
        source_token_idx = rank_route_idx // top_k
        topk_ordinal = rank_route_idx - source_token_idx * top_k
        token_count = token_counts[source_rank]
        if source_token_idx < token_count and source_token_idx < max_num_tokens_per_rank:
            selected_expert = topk_idx[source_rank, source_token_idx, topk_ordinal]
            if selected_expert >= first_local_expert and selected_expert < last_local_expert:
                local_expert_idx = cutlass.Int32(selected_expert - first_local_expert)
                expert_base = expert_route_base_offsets[local_expert_idx]
                ordinal = atomicAdd(
                    expert_route_offsets.iterator + local_expert_idx, cutlass.Int32(1)
                )
                pool_slot = expert_base + ordinal
                if pool_slot >= cutlass.Int32(0) and pool_slot < cutlass.Int32(num_pool_slots):
                    token_row = source_rank * max_num_tokens_per_rank + source_token_idx
                    if cutlass.const_expr(direct_atomic_output or direct_token_major_output):
                        combine_row = token_row * top_k + topk_ordinal
                        combine_row_limit = output_scale_rows
                    else:
                        combine_row = (
                            source_rank * top_k + topk_ordinal
                        ) * max_num_tokens_per_rank + source_token_idx
                        combine_row_limit = combine_layout_rows

                    if (
                        pool_slot < cutlass.Int32(output_mapping_rows)
                        and combine_row < combine_row_limit
                    ):
                        output_mapping[pool_slot] = combine_row
                    if cutlass.const_expr(direct_atomic_output):
                        token_id_mapping[pool_slot] = combine_row
                    else:
                        token_id_mapping[pool_slot] = token_row
                    if combine_row < output_scale_rows:
                        output_scales[combine_row, 0] = topk_scales[
                            source_rank, source_token_idx, topk_ordinal
                        ]
        route_idx = route_idx + stride

    griddepcontrol_launch_dependents()


@cute.jit
def launch_direct_route(
    input_ptr: cute.Pointer,
    input_sf_ptr: cute.Pointer,
    topk_idx_ptr: cute.Pointer,
    topk_scales_ptr: cute.Pointer,
    token_counts_ptr: cute.Pointer,
    direct_input_ptr: cute.Pointer,
    direct_input_sf_ptr: cute.Pointer,
    expert_route_offsets_ptr: cute.Pointer,
    expert_route_base_offsets_ptr: cute.Pointer,
    token_id_mapping_ptr: cute.Pointer,
    output_mapping_ptr: cute.Pointer,
    output_scales_ptr: cute.Pointer,
    tile_idx_to_expert_idx_ptr: cute.Pointer,
    tile_idx_to_mn_limit_ptr: cute.Pointer,
    num_non_exiting_tiles_ptr: cute.Pointer,
    input_stride_0: cutlass.Constexpr,
    input_stride_1: cutlass.Constexpr,
    input_stride_2: cutlass.Constexpr,
    input_sf_stride_0: cutlass.Constexpr,
    input_sf_stride_1: cutlass.Constexpr,
    input_sf_stride_2: cutlass.Constexpr,
    topk_idx_stride_0: cutlass.Constexpr,
    topk_idx_stride_1: cutlass.Constexpr,
    topk_idx_stride_2: cutlass.Constexpr,
    topk_scales_stride_0: cutlass.Constexpr,
    topk_scales_stride_1: cutlass.Constexpr,
    topk_scales_stride_2: cutlass.Constexpr,
    direct_input_stride_0: cutlass.Constexpr,
    direct_input_stride_1: cutlass.Constexpr,
    direct_input_sf_stride_0: cutlass.Constexpr,
    direct_input_sf_stride_1: cutlass.Constexpr,
    output_scales_stride_0: cutlass.Constexpr,
    output_scales_stride_1: cutlass.Constexpr,
    ep_size: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    hidden_packed_size: cutlass.Constexpr,
    sf_hidden_size: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    tile_size: cutlass.Constexpr,
    num_pool_slots: cutlass.Constexpr,
    max_num_tokens_per_rank: cutlass.Constexpr,
    combine_layout_rows: cutlass.Constexpr,
    output_mapping_rows: cutlass.Constexpr,
    output_scale_rows: cutlass.Constexpr,
    output_scales_cols: cutlass.Constexpr,
    route_layout_capacity: cutlass.Constexpr,
    direct_atomic_output: cutlass.Constexpr,
    direct_token_major_output: cutlass.Constexpr,
    total_routes: cutlass.Constexpr,
    max_copy_work_items: cutlass.Constexpr,
    init_work_items: cutlass.Constexpr,
    init_blocks: cutlass.Constexpr,
    route_blocks: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    input = cute.make_tensor(
        input_ptr,
        cute.make_layout(
            (ep_size, max_num_tokens_per_rank, hidden_packed_size),
            stride=(input_stride_0, input_stride_1, input_stride_2),
        ),
    )
    input_sf = cute.make_tensor(
        input_sf_ptr,
        cute.make_layout(
            (ep_size, max_num_tokens_per_rank, sf_hidden_size),
            stride=(input_sf_stride_0, input_sf_stride_1, input_sf_stride_2),
        ),
    )
    topk_idx = cute.make_tensor(
        topk_idx_ptr,
        cute.make_layout(
            (ep_size, max_num_tokens_per_rank, top_k),
            stride=(topk_idx_stride_0, topk_idx_stride_1, topk_idx_stride_2),
        ),
    )
    topk_scales = cute.make_tensor(
        topk_scales_ptr,
        cute.make_layout(
            (ep_size, max_num_tokens_per_rank, top_k),
            stride=(topk_scales_stride_0, topk_scales_stride_1, topk_scales_stride_2),
        ),
    )
    token_counts = cute.make_tensor(token_counts_ptr, cute.make_layout((ep_size,)))
    direct_input = cute.make_tensor(
        direct_input_ptr,
        cute.make_layout(
            (num_pool_slots, hidden_packed_size),
            stride=(direct_input_stride_0, direct_input_stride_1),
        ),
    )
    direct_input_sf = cute.make_tensor(
        direct_input_sf_ptr,
        cute.make_layout(
            (num_pool_slots, sf_hidden_size),
            stride=(direct_input_sf_stride_0, direct_input_sf_stride_1),
        ),
    )
    expert_route_offsets = cute.make_tensor(
        expert_route_offsets_ptr, cute.make_layout((num_experts_per_rank,))
    )
    expert_route_base_offsets = cute.make_tensor(
        expert_route_base_offsets_ptr, cute.make_layout((num_experts_per_rank,))
    )
    token_id_mapping = cute.make_tensor(token_id_mapping_ptr, cute.make_layout((num_pool_slots,)))
    output_mapping = cute.make_tensor(output_mapping_ptr, cute.make_layout((output_mapping_rows,)))
    output_scales = cute.make_tensor(
        output_scales_ptr,
        cute.make_layout(
            (output_scale_rows, output_scales_cols),
            stride=(output_scales_stride_0, output_scales_stride_1),
        ),
    )
    tile_idx_to_expert_idx = cute.make_tensor(
        tile_idx_to_expert_idx_ptr, cute.make_layout((route_layout_capacity,))
    )
    tile_idx_to_mn_limit = cute.make_tensor(
        tile_idx_to_mn_limit_ptr, cute.make_layout((route_layout_capacity,))
    )
    num_non_exiting_tiles = cute.make_tensor(num_non_exiting_tiles_ptr, cute.make_layout((1,)))

    _direct_route_init_kernel(
        expert_route_offsets,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
        num_experts_per_rank,
        route_layout_capacity,
        init_work_items,
        threads_per_block,
        init_blocks,
    ).launch(
        grid=(init_blocks, 1, 1),
        block=[threads_per_block, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )
    _direct_route_copy_count_kernel(
        input,
        input_sf,
        topk_idx,
        token_counts,
        direct_input,
        direct_input_sf,
        expert_route_offsets,
        ep_size,
        local_rank,
        num_experts_per_rank,
        hidden_packed_size,
        sf_hidden_size,
        top_k,
        max_num_tokens_per_rank,
        total_routes,
        max_copy_work_items,
        threads_per_block,
        route_blocks,
    ).launch(
        grid=(route_blocks, 1, 1),
        block=[threads_per_block, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )
    _direct_route_prefix_tiles_kernel(
        expert_route_offsets,
        expert_route_base_offsets,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
        num_experts_per_rank,
        tile_size,
        num_pool_slots,
        route_layout_capacity,
    ).launch(
        grid=(1, 1, 1),
        block=[threads_per_block, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )
    _direct_route_fill_kernel(
        topk_idx,
        topk_scales,
        token_counts,
        expert_route_offsets,
        expert_route_base_offsets,
        token_id_mapping,
        output_mapping,
        output_scales,
        ep_size,
        local_rank,
        num_experts_per_rank,
        top_k,
        max_num_tokens_per_rank,
        num_pool_slots,
        combine_layout_rows,
        output_mapping_rows,
        output_scale_rows,
        direct_atomic_output,
        direct_token_major_output,
        total_routes,
        threads_per_block,
        route_blocks,
    ).launch(
        grid=(route_blocks, 1, 1),
        block=[threads_per_block, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


def _tensor_compile_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device.type,
        tensor.device.index,
    )


def _make_gmem_ptr(tensor: torch.Tensor) -> cute.Pointer:
    if tensor.dtype == torch.uint8:
        return make_ptr(cutlass.Uint8, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=1)
    if tensor.dtype == torch.int32:
        return make_ptr(cutlass.Int32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    if tensor.dtype == torch.int64:
        return make_ptr(cutlass.Int64, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=8)
    if tensor.dtype == torch.float32:
        return make_ptr(cutlass.Float32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    raise ValueError(f"unsupported tensor dtype for CUTEDSL direct route pointer: {tensor.dtype}")


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on the route device")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}")
    if tensor.dim() != dim:
        raise ValueError(f"{name} must be {dim}D")


def build_direct_input_route_from_ranked_topk(
    input: torch.Tensor,
    input_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    token_counts: torch.Tensor,
    direct_input: torch.Tensor,
    direct_input_sf: torch.Tensor,
    expert_route_offsets: torch.Tensor,
    expert_route_base_offsets: torch.Tensor,
    token_id_mapping: torch.Tensor,
    output_mapping: torch.Tensor,
    output_scales: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    local_rank: int,
    tile_size: int,
    combine_layout_rows: int,
    direct_atomic_output: bool = False,
    direct_token_major_output: bool = False,
) -> None:
    """Build direct-input route metadata with CuTe DSL kernels."""
    device = input.device
    _check_tensor("input", input, device=device, dtype=torch.uint8, dim=3)
    _check_tensor("input_sf", input_sf, device=device, dtype=torch.uint8, dim=3)
    _check_tensor("topk_idx", topk_idx, device=device, dtype=torch.int64, dim=3)
    _check_tensor("topk_scales", topk_scales, device=device, dtype=torch.float32, dim=3)
    _check_tensor("token_counts", token_counts, device=device, dtype=torch.int32, dim=1)
    _check_tensor("direct_input", direct_input, device=device, dtype=torch.uint8, dim=2)
    _check_tensor("direct_input_sf", direct_input_sf, device=device, dtype=torch.uint8, dim=2)
    _check_tensor(
        "expert_route_offsets", expert_route_offsets, device=device, dtype=torch.int32, dim=1
    )
    _check_tensor(
        "expert_route_base_offsets",
        expert_route_base_offsets,
        device=device,
        dtype=torch.int32,
        dim=1,
    )
    _check_tensor("token_id_mapping", token_id_mapping, device=device, dtype=torch.int32, dim=1)
    _check_tensor("output_mapping", output_mapping, device=device, dtype=torch.int32, dim=1)
    _check_tensor("output_scales", output_scales, device=device, dtype=torch.float32, dim=2)
    _check_tensor(
        "tile_idx_to_expert_idx", tile_idx_to_expert_idx, device=device, dtype=torch.int32, dim=1
    )
    _check_tensor(
        "tile_idx_to_mn_limit", tile_idx_to_mn_limit, device=device, dtype=torch.int32, dim=1
    )
    _check_tensor(
        "num_non_exiting_tiles", num_non_exiting_tiles, device=device, dtype=torch.int32, dim=1
    )

    ep_size = int(input.size(0))
    max_num_tokens_per_rank = int(input.size(1))
    hidden_packed_size = int(input.size(2))
    sf_hidden_size = int(input_sf.size(2))
    top_k = int(topk_idx.size(2))
    local_rank = int(local_rank)
    tile_size = int(tile_size)
    combine_layout_rows = int(combine_layout_rows)
    direct_atomic_output = bool(direct_atomic_output)
    direct_token_major_output = bool(direct_token_major_output)

    if ep_size <= 0 or max_num_tokens_per_rank <= 0 or hidden_packed_size <= 0:
        raise ValueError("input dimensions must be positive")
    if input_sf.shape[:2] != input.shape[:2]:
        raise ValueError("input_sf leading dimensions must match input")
    if topk_idx.shape[:2] != input.shape[:2] or topk_scales.shape != topk_idx.shape:
        raise ValueError("topk tensors must match input leading dimensions and each other")
    if token_counts.numel() != ep_size:
        raise ValueError("token_counts must have one entry per EP rank")
    if local_rank < 0 or local_rank >= ep_size:
        raise ValueError("local_rank must fit ep_size")
    if tile_size <= 0 or combine_layout_rows <= 0:
        raise ValueError("tile_size and combine_layout_rows must be positive")

    flat_input_rows = ep_size * max_num_tokens_per_rank
    if (
        int(direct_input.size(0)) < flat_input_rows
        or int(direct_input.size(1)) != hidden_packed_size
    ):
        raise ValueError("direct_input must cover all flattened input rows")
    if (
        int(direct_input_sf.size(0)) < flat_input_rows
        or int(direct_input_sf.size(1)) != sf_hidden_size
    ):
        raise ValueError("direct_input_sf must cover all flattened input rows")

    num_experts_per_rank = int(expert_route_offsets.numel())
    num_pool_slots = int(token_id_mapping.numel())
    output_mapping_rows = int(output_mapping.numel())
    output_scale_rows = (
        int(output_scales.numel()) if direct_atomic_output else int(output_scales.size(0))
    )
    route_layout_capacity = min(
        int(tile_idx_to_expert_idx.numel()), int(tile_idx_to_mn_limit.numel())
    )
    if num_experts_per_rank <= 0 or num_pool_slots <= 0 or route_layout_capacity <= 0:
        raise ValueError("route metadata tensors must be non-empty")
    if int(expert_route_base_offsets.numel()) != num_experts_per_rank:
        raise ValueError("expert_route_base_offsets must match expert_route_offsets")
    if output_mapping_rows < num_pool_slots:
        raise ValueError("output_mapping must cover token_id_mapping rows")
    if direct_atomic_output:
        if int(output_scales.size(1)) != top_k or output_scale_rows < flat_input_rows * top_k:
            raise ValueError("atomic output_scales must cover flattened input rows and top_k")
    else:
        if int(output_scales.size(1)) != 1 or output_scale_rows < combine_layout_rows:
            raise ValueError("output_scales must cover combine-layout rows")

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability[0] < 10:
        raise ValueError("CUTEDSL MegaMoE direct route requires Blackwell or newer")

    total_routes = ep_size * max_num_tokens_per_rank * top_k
    flat_input_elements = flat_input_rows * hidden_packed_size
    flat_input_sf_elements = flat_input_rows * sf_hidden_size
    max_copy_work_items = max(total_routes, flat_input_elements, flat_input_sf_elements)
    if max_copy_work_items <= 0:
        return

    props = torch.cuda.get_device_properties(device_index)
    platform_blocks = int(props.multi_processor_count) * 8
    route_blocks = max(
        1,
        min(platform_blocks, (max_copy_work_items + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK),
    )
    init_work_items = max(num_experts_per_rank, route_layout_capacity)
    init_blocks = max(
        1, min(platform_blocks, (init_work_items + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK)
    )

    tensors = (
        input,
        input_sf,
        topk_idx,
        topk_scales,
        token_counts,
        direct_input,
        direct_input_sf,
        expert_route_offsets,
        expert_route_base_offsets,
        token_id_mapping,
        output_mapping,
        output_scales,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
    )
    ptrs = tuple(_make_gmem_ptr(tensor) for tensor in tensors)
    strides = (
        int(input.stride(0)),
        int(input.stride(1)),
        int(input.stride(2)),
        int(input_sf.stride(0)),
        int(input_sf.stride(1)),
        int(input_sf.stride(2)),
        int(topk_idx.stride(0)),
        int(topk_idx.stride(1)),
        int(topk_idx.stride(2)),
        int(topk_scales.stride(0)),
        int(topk_scales.stride(1)),
        int(topk_scales.stride(2)),
        int(direct_input.stride(0)),
        int(direct_input.stride(1)),
        int(direct_input_sf.stride(0)),
        int(direct_input_sf.stride(1)),
        int(output_scales.stride(0)),
        int(output_scales.stride(1)),
    )
    output_scales_cols = int(output_scales.size(1))
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    compile_key = (
        capability,
        tuple(_tensor_compile_signature(tensor) for tensor in tensors),
        strides,
        ep_size,
        local_rank,
        num_experts_per_rank,
        hidden_packed_size,
        sf_hidden_size,
        top_k,
        tile_size,
        num_pool_slots,
        max_num_tokens_per_rank,
        combine_layout_rows,
        output_mapping_rows,
        output_scale_rows,
        output_scales_cols,
        route_layout_capacity,
        direct_atomic_output,
        direct_token_major_output,
        total_routes,
        max_copy_work_items,
        init_work_items,
        init_blocks,
        route_blocks,
        _THREADS_PER_BLOCK,
    )
    if compile_key not in _direct_route_compile_cache:
        _direct_route_compile_cache[compile_key] = cute.compile(
            launch_direct_route,
            *ptrs,
            *strides,
            ep_size,
            local_rank,
            num_experts_per_rank,
            hidden_packed_size,
            sf_hidden_size,
            top_k,
            tile_size,
            num_pool_slots,
            max_num_tokens_per_rank,
            combine_layout_rows,
            output_mapping_rows,
            output_scale_rows,
            output_scales_cols,
            route_layout_capacity,
            direct_atomic_output,
            direct_token_major_output,
            total_routes,
            max_copy_work_items,
            init_work_items,
            init_blocks,
            route_blocks,
            _THREADS_PER_BLOCK,
            stream,
        )

    _direct_route_compile_cache[compile_key](
        *ptrs,
        stream,
    )


__all__ = ["build_direct_input_route_from_ranked_topk"]
