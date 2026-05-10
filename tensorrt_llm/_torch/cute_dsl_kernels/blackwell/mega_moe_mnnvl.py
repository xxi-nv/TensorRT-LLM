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
"""MegaMoE MNNVL CuTe DSL debug kernels and launch helpers."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    ld_acquire_sys_u32,
    red_add_release_sys_u32,
)

_MAX_ROUTE_PAYLOAD_ROW_BYTES = 1024
_route_payload_pull_compile_cache: dict[tuple[object, ...], object] = {}
_topk_route_count_compile_cache: dict[tuple[object, ...], object] = {}
_topk_route_metadata_compile_cache: dict[tuple[object, ...], object] = {}
_pool_route_metadata_compile_cache: dict[tuple[object, ...], object] = {}


@cute.kernel
def _mnnvl_topk_route_metadata_kernel(
    topk_idx_workspace: cute.Tensor,
    token_counts: cute.Tensor,
    route_counts: cute.Tensor,
    src_token_topk_idx: cute.Tensor,
    token_src_metadata: cute.Tensor,
    local_rank: cutlass.Constexpr,
    topk_idx_offset: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    max_routes_per_pair: cutlass.Constexpr,
    ep_size: cutlass.Constexpr,
) -> None:
    pair_idx, _, _ = cute.arch.block_idx()
    source_rank = pair_idx // num_experts_per_rank
    local_expert_idx = pair_idx - source_rank * num_experts_per_rank
    global_expert_idx = local_rank * num_experts_per_rank + local_expert_idx
    route_count = cutlass.Int32(0)
    source_token_count = token_counts[source_rank]

    for token_idx in range(num_tokens):
        if cutlass.Int32(token_idx) < source_token_count:
            for topk_ordinal in range(top_k):
                src_idx = token_idx * top_k + topk_ordinal
                expert_idx = topk_idx_workspace[source_rank, topk_idx_offset + src_idx]
                if expert_idx == global_expert_idx:
                    route_slot = (
                        local_expert_idx * ep_size + source_rank
                    ) * max_routes_per_pair + route_count
                    if route_count < max_routes_per_pair:
                        src_token_topk_idx[route_slot] = cutlass.Int32(src_idx)
                        token_src_metadata[route_slot, 0] = cutlass.Int32(source_rank)
                        token_src_metadata[route_slot, 1] = cutlass.Int32(token_idx)
                        token_src_metadata[route_slot, 2] = cutlass.Int32(topk_ordinal)
                    route_count = route_count + 1

    route_counts[pair_idx] = route_count


@cute.kernel
def _mnnvl_pool_route_metadata_kernel(
    route_counts: cute.Tensor,
    src_token_topk_idx: cute.Tensor,
    token_src_metadata: cute.Tensor,
    l1_arrival_count: cute.Tensor,
    top_k: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    ep_size: cutlass.Constexpr,
    max_routes_per_pair: cutlass.Constexpr,
    tile_size: cutlass.Constexpr,
    num_max_pool_tokens: cutlass.Constexpr,
) -> None:
    local_expert_idx, _, _ = cute.arch.block_idx()

    pool_start = cutlass.Int32(0)
    for previous_expert_idx in range(num_experts_per_rank):
        if cutlass.Int32(previous_expert_idx) < cutlass.Int32(local_expert_idx):
            expert_routes = cutlass.Int32(0)
            for source_rank in range(ep_size):
                count_offset = source_rank * num_experts_per_rank + previous_expert_idx
                expert_routes = expert_routes + route_counts[count_offset]
            pool_start = ((pool_start + expert_routes + tile_size - 1) // tile_size) * tile_size

    pool_slot = pool_start
    for route_ordinal in range(max_routes_per_pair):
        for source_rank in range(ep_size):
            count_offset = source_rank * num_experts_per_rank + local_expert_idx
            route_count = route_counts[count_offset]
            if cutlass.Int32(route_ordinal) < route_count:
                table_base = (
                    local_expert_idx * ep_size + source_rank
                ) * max_routes_per_pair + route_ordinal
                src_idx = src_token_topk_idx[table_base]
                token_idx = src_idx // top_k
                topk_ordinal = src_idx - token_idx * top_k
                if pool_slot < num_max_pool_tokens:
                    token_src_metadata[pool_slot, 0] = cutlass.Int32(source_rank)
                    token_src_metadata[pool_slot, 1] = token_idx
                    token_src_metadata[pool_slot, 2] = topk_ordinal
                    tile_idx = pool_slot // tile_size
                    l1_arrival_count[tile_idx] = l1_arrival_count[tile_idx] + cutlass.Int32(1)
                pool_slot = pool_slot + 1


@cute.kernel
def _mnnvl_topk_route_count_kernel(
    topk_idx_workspace: cute.Tensor,
    route_counts: cute.Tensor,
    local_rank: cutlass.Constexpr,
    topk_idx_offset: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
) -> None:
    route_linear_idx, _, _ = cute.arch.block_idx()

    routes_per_rank = num_tokens * top_k
    source_rank = route_linear_idx // routes_per_rank
    route_in_rank = route_linear_idx - source_rank * routes_per_rank
    token_idx = route_in_rank // top_k
    topk_ordinal = route_in_rank - token_idx * top_k

    expert_idx = topk_idx_workspace[source_rank, topk_idx_offset + token_idx * top_k + topk_ordinal]
    local_expert_start = local_rank * num_experts_per_rank
    local_expert_idx = cutlass.Int32(expert_idx - local_expert_start)

    if local_expert_idx >= 0 and local_expert_idx < num_experts_per_rank:
        count_offset = source_rank * num_experts_per_rank + local_expert_idx
        count_ptr = route_counts.iterator + count_offset
        red_add_release_sys_u32(count_ptr, cutlass.Uint32(1))


@cute.kernel
def _mnnvl_route_payload_pull_kernel(
    workspace: cute.Tensor,
    signal: cute.Tensor,
    route_metadata: cute.Tensor,
    pulled: cute.Tensor,
    observed_signal: cute.Tensor,
    peer_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    payload_byte_offset: cutlass.Constexpr,
    row_bytes: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    token_stride_rows: cutlass.Constexpr,
    topk_stride_rows: cutlass.Constexpr,
    num_routes: cutlass.Constexpr,
    done_count_signal_u32_offset: cutlass.Constexpr,
    peer_done_signal_u32_offset: cutlass.Constexpr,
) -> None:
    route_idx, _, _ = cute.arch.block_idx()
    byte_idx, _, _ = cute.arch.thread_idx()

    src_rank = route_metadata[route_idx, 0]
    token_idx = route_metadata[route_idx, 1]
    topk_idx = route_metadata[route_idx, 2]
    if src_rank < 0:
        pulled[route_idx, byte_idx] = cute.Uint8(0)
    else:
        row_idx = token_idx * token_stride_rows + topk_idx * topk_stride_rows
        source_offset = payload_byte_offset + row_idx * row_bytes + byte_idx
        pulled[route_idx, byte_idx] = workspace[src_rank, source_offset]

    if byte_idx == 0:
        local_count_ptr = signal.iterator + cute.crd2idx(
            (local_rank, done_count_signal_u32_offset), signal.layout
        )
        red_add_release_sys_u32(local_count_ptr, cutlass.Uint32(1))

        if route_idx == 0:
            local_count = cutlass.Uint32(0)
            while local_count < cutlass.Uint32(num_routes):
                local_count = ld_acquire_sys_u32(local_count_ptr)

            peer_done_ptr = signal.iterator + cute.crd2idx(
                (peer_rank, peer_done_signal_u32_offset), signal.layout
            )
            local_done_ptr = signal.iterator + cute.crd2idx(
                (local_rank, peer_done_signal_u32_offset), signal.layout
            )
            red_add_release_sys_u32(peer_done_ptr, cutlass.Uint32(1))

            peer_done = cutlass.Uint32(0)
            while peer_done < cutlass.Uint32(1):
                peer_done = ld_acquire_sys_u32(local_done_ptr)
            observed_signal[0] = peer_done
            observed_signal[1] = local_count


@cute.jit
def launch_mnnvl_topk_route_metadata(
    topk_idx_workspace: cute.Tensor,
    token_counts: cute.Tensor,
    route_counts: cute.Tensor,
    src_token_topk_idx: cute.Tensor,
    token_src_metadata: cute.Tensor,
    local_rank: cutlass.Constexpr,
    topk_idx_offset: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    max_routes_per_pair: cutlass.Constexpr,
    ep_size: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch a top-k route-metadata debug kernel over MNNVL staged metadata."""
    _mnnvl_topk_route_metadata_kernel(
        topk_idx_workspace,
        token_counts,
        route_counts,
        src_token_topk_idx,
        token_src_metadata,
        local_rank,
        topk_idx_offset,
        num_tokens,
        top_k,
        num_experts_per_rank,
        max_routes_per_pair,
        ep_size,
    ).launch(
        grid=(ep_size * num_experts_per_rank, 1, 1),
        block=[1, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


@cute.jit
def launch_mnnvl_pool_route_metadata(
    route_counts: cute.Tensor,
    src_token_topk_idx: cute.Tensor,
    token_src_metadata: cute.Tensor,
    l1_arrival_count: cute.Tensor,
    top_k: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    ep_size: cutlass.Constexpr,
    max_routes_per_pair: cutlass.Constexpr,
    tile_size: cutlass.Constexpr,
    num_max_pool_tokens: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch pool-ordered route metadata construction from source-major routes."""
    _mnnvl_pool_route_metadata_kernel(
        route_counts,
        src_token_topk_idx,
        token_src_metadata,
        l1_arrival_count,
        top_k,
        num_experts_per_rank,
        ep_size,
        max_routes_per_pair,
        tile_size,
        num_max_pool_tokens,
    ).launch(
        grid=(num_experts_per_rank, 1, 1),
        block=[1, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


@cute.jit
def launch_mnnvl_topk_route_count(
    topk_idx_workspace: cute.Tensor,
    route_counts: cute.Tensor,
    local_rank: cutlass.Constexpr,
    topk_idx_offset: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    num_experts_per_rank: cutlass.Constexpr,
    ep_size: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch a top-k route-count debug kernel over MNNVL rank-strided metadata."""
    _mnnvl_topk_route_count_kernel(
        topk_idx_workspace,
        route_counts,
        local_rank,
        topk_idx_offset,
        num_tokens,
        top_k,
        num_experts_per_rank,
    ).launch(
        grid=(ep_size * num_tokens * top_k, 1, 1),
        block=[1, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


@cute.jit
def launch_mnnvl_route_payload_pull(
    workspace: cute.Tensor,
    signal: cute.Tensor,
    route_metadata: cute.Tensor,
    pulled: cute.Tensor,
    observed_signal: cute.Tensor,
    peer_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    payload_byte_offset: cutlass.Constexpr,
    row_bytes: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    token_stride_rows: cutlass.Constexpr,
    topk_stride_rows: cutlass.Constexpr,
    num_routes: cutlass.Constexpr,
    done_count_signal_u32_offset: cutlass.Constexpr,
    peer_done_signal_u32_offset: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch a route-metadata-driven MNNVL byte-row pull debug kernel."""
    _mnnvl_route_payload_pull_kernel(
        workspace,
        signal,
        route_metadata,
        pulled,
        observed_signal,
        peer_rank,
        local_rank,
        payload_byte_offset,
        row_bytes,
        num_tokens,
        token_stride_rows,
        topk_stride_rows,
        num_routes,
        done_count_signal_u32_offset,
        peer_done_signal_u32_offset,
    ).launch(
        grid=(num_routes, 1, 1),
        block=[row_bytes, 1, 1],
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


def _to_dynamic_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach()).mark_layout_dynamic()


def _validate_topk_route_metadata_inputs(
    topk_idx_workspace: torch.Tensor,
    token_counts: torch.Tensor,
    route_counts: torch.Tensor,
    src_token_topk_idx: torch.Tensor,
    token_src_metadata: torch.Tensor,
    local_rank: int,
    topk_idx_offset: int,
    num_tokens: int,
    top_k: int,
    num_experts_per_rank: int,
    max_routes_per_pair: int,
    ep_size: int,
) -> None:
    tensors = {
        "topk_idx_workspace": topk_idx_workspace,
        "token_counts": token_counts,
        "route_counts": route_counts,
        "src_token_topk_idx": src_token_topk_idx,
        "token_src_metadata": token_src_metadata,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    if topk_idx_workspace.dtype != torch.int64 or topk_idx_workspace.dim() != 2:
        raise ValueError("topk_idx_workspace must be a rank-strided 2D int64 tensor")
    if token_counts.dtype != torch.int32 or token_counts.dim() != 1:
        raise ValueError("token_counts must be a 1D int32 tensor")
    if route_counts.dtype != torch.int32 or route_counts.dim() != 1:
        raise ValueError("route_counts must be a 1D int32 tensor")
    if src_token_topk_idx.dtype != torch.int32 or src_token_topk_idx.dim() != 1:
        raise ValueError("src_token_topk_idx must be a 1D int32 tensor")
    if token_src_metadata.dtype != torch.int32 or token_src_metadata.dim() != 2:
        raise ValueError("token_src_metadata must be a 2D int32 tensor")
    if token_src_metadata.shape[1] < 3:
        raise ValueError("token_src_metadata must have at least three columns")
    if ep_size <= 0 or ep_size > topk_idx_workspace.shape[0]:
        raise ValueError("ep_size must be positive and fit topk_idx_workspace ranks")
    if local_rank < 0 or local_rank >= ep_size:
        raise ValueError("local_rank must fit ep_size")
    if topk_idx_offset < 0 or min(num_tokens, top_k, num_experts_per_rank) <= 0:
        raise ValueError("topk_idx_offset must be non-negative and shape parameters positive")
    if max_routes_per_pair <= 0:
        raise ValueError("max_routes_per_pair must be positive")
    routes_per_rank = num_tokens * top_k
    if topk_idx_offset + routes_per_rank > topk_idx_workspace.shape[1]:
        raise ValueError("topk_idx range must fit topk_idx_workspace")
    pair_count = ep_size * num_experts_per_rank
    required_route_slots = pair_count * max_routes_per_pair
    if token_counts.numel() < ep_size:
        raise ValueError("token_counts must fit ep_size")
    if route_counts.numel() < pair_count:
        raise ValueError("route_counts must fit ep_size * num_experts_per_rank")
    if src_token_topk_idx.numel() < required_route_slots:
        raise ValueError("src_token_topk_idx must fit all route slots")
    if token_src_metadata.shape[0] < required_route_slots:
        raise ValueError("token_src_metadata must fit all route slots")


def _validate_pool_route_metadata_inputs(
    route_counts: torch.Tensor,
    src_token_topk_idx: torch.Tensor,
    token_src_metadata: torch.Tensor,
    l1_arrival_count: torch.Tensor,
    top_k: int,
    num_experts_per_rank: int,
    ep_size: int,
    max_routes_per_pair: int,
    tile_size: int,
    num_max_pool_tokens: int,
) -> None:
    tensors = {
        "route_counts": route_counts,
        "src_token_topk_idx": src_token_topk_idx,
        "token_src_metadata": token_src_metadata,
        "l1_arrival_count": l1_arrival_count,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    if route_counts.dtype != torch.int32 or route_counts.dim() != 1:
        raise ValueError("route_counts must be a 1D int32 tensor")
    if src_token_topk_idx.dtype != torch.int32 or src_token_topk_idx.dim() != 1:
        raise ValueError("src_token_topk_idx must be a 1D int32 tensor")
    if token_src_metadata.dtype != torch.int32 or token_src_metadata.dim() != 2:
        raise ValueError("token_src_metadata must be a 2D int32 tensor")
    if token_src_metadata.shape[1] < 3:
        raise ValueError("token_src_metadata must have at least three columns")
    if l1_arrival_count.dtype != torch.int32 or l1_arrival_count.dim() != 1:
        raise ValueError("l1_arrival_count must be a 1D int32 tensor")
    if min(top_k, num_experts_per_rank, ep_size, max_routes_per_pair, tile_size) <= 0:
        raise ValueError("route metadata shape parameters must be positive")
    if num_max_pool_tokens <= 0 or token_src_metadata.shape[0] < num_max_pool_tokens:
        raise ValueError("num_max_pool_tokens must fit token_src_metadata")

    pair_count = ep_size * num_experts_per_rank
    required_route_slots = pair_count * max_routes_per_pair
    if route_counts.numel() < pair_count:
        raise ValueError("route_counts must fit ep_size * num_experts_per_rank")
    if src_token_topk_idx.numel() < required_route_slots:
        raise ValueError("src_token_topk_idx must fit all route slots")
    required_arrival_counts = (num_max_pool_tokens + tile_size - 1) // tile_size
    if l1_arrival_count.numel() < required_arrival_counts:
        raise ValueError("l1_arrival_count must fit all pool tiles")


def _validate_topk_route_count_inputs(
    topk_idx_workspace: torch.Tensor,
    route_counts: torch.Tensor,
    local_rank: int,
    topk_idx_offset: int,
    num_tokens: int,
    top_k: int,
    num_experts_per_rank: int,
    ep_size: int,
) -> None:
    if not topk_idx_workspace.is_cuda:
        raise ValueError("topk_idx_workspace must be a CUDA tensor")
    if not route_counts.is_cuda:
        raise ValueError("route_counts must be a CUDA tensor")
    if topk_idx_workspace.dtype != torch.int64 or topk_idx_workspace.dim() != 2:
        raise ValueError("topk_idx_workspace must be a rank-strided 2D int64 tensor")
    if route_counts.dtype != torch.uint32 or route_counts.dim() != 1:
        raise ValueError("route_counts must be a 1D uint32 tensor")
    if ep_size <= 0 or ep_size > topk_idx_workspace.shape[0]:
        raise ValueError("ep_size must be positive and fit topk_idx_workspace ranks")
    if local_rank < 0 or local_rank >= ep_size:
        raise ValueError("local_rank must fit ep_size")
    if topk_idx_offset < 0 or min(num_tokens, top_k, num_experts_per_rank) <= 0:
        raise ValueError("topk_idx_offset must be non-negative and shape parameters positive")
    routes_per_rank = num_tokens * top_k
    if topk_idx_offset + routes_per_rank > topk_idx_workspace.shape[1]:
        raise ValueError("topk_idx range must fit topk_idx_workspace")
    if route_counts.numel() < ep_size * num_experts_per_rank:
        raise ValueError("route_counts must fit ep_size * num_experts_per_rank")


def _validate_route_payload_pull_inputs(
    workspace: torch.Tensor,
    signal: torch.Tensor,
    route_metadata: torch.Tensor,
    pulled: torch.Tensor,
    observed_signal: torch.Tensor,
    peer_rank: int,
    local_rank: int,
    row_bytes: int,
    num_routes: int,
    done_count_signal_u32_offset: int,
    peer_done_signal_u32_offset: int,
) -> None:
    tensors = {
        "workspace": workspace,
        "signal": signal,
        "route_metadata": route_metadata,
        "pulled": pulled,
        "observed_signal": observed_signal,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    if workspace.dtype != torch.uint8 or workspace.dim() != 2:
        raise ValueError("workspace must be a rank-strided 2D uint8 tensor")
    if signal.dtype != torch.uint32 or signal.dim() != 2:
        raise ValueError("signal must be a rank-strided 2D uint32 tensor")
    if route_metadata.dtype != torch.int32 or route_metadata.dim() != 2:
        raise ValueError("route_metadata must be a 2D int32 tensor")
    if route_metadata.shape[1] < 3:
        raise ValueError("route_metadata must have at least three columns")
    if pulled.dtype != torch.uint8 or pulled.dim() != 2:
        raise ValueError("pulled must be a 2D uint8 tensor")
    if observed_signal.dtype != torch.uint32 or observed_signal.numel() < 2:
        raise ValueError("observed_signal must contain at least two uint32 elements")
    if num_routes <= 0 or num_routes > route_metadata.shape[0] or num_routes > pulled.shape[0]:
        raise ValueError("num_routes must be positive and fit route_metadata and pulled")
    if row_bytes <= 0 or row_bytes > pulled.shape[1]:
        raise ValueError("row_bytes must be positive and fit pulled rows")
    if row_bytes > _MAX_ROUTE_PAYLOAD_ROW_BYTES:
        raise ValueError(f"row_bytes must not exceed {_MAX_ROUTE_PAYLOAD_ROW_BYTES}")
    if peer_rank < 0 or local_rank < 0:
        raise ValueError("peer_rank and local_rank must be non-negative")
    max_rank = max(peer_rank, local_rank)
    if max_rank >= workspace.shape[0] or max_rank >= signal.shape[0]:
        raise ValueError("peer_rank and local_rank must fit workspace and signal ranks")
    if min(done_count_signal_u32_offset, peer_done_signal_u32_offset) < 0:
        raise ValueError("signal offsets must be non-negative")
    if max(done_count_signal_u32_offset, peer_done_signal_u32_offset) >= signal.shape[1]:
        raise ValueError("signal offsets must fit the signal tensor")


def build_mnnvl_topk_route_metadata(
    topk_idx_workspace: torch.Tensor,
    token_counts: torch.Tensor,
    route_counts: torch.Tensor,
    src_token_topk_idx: torch.Tensor,
    token_src_metadata: torch.Tensor,
    *,
    local_rank: int,
    topk_idx_offset: int,
    num_tokens: int,
    top_k: int,
    num_experts_per_rank: int,
    max_routes_per_pair: int,
    ep_size: int | None = None,
) -> None:
    """Compile/cache and launch the MNNVL top-k route-metadata debug kernel."""
    if ep_size is None:
        ep_size = int(topk_idx_workspace.shape[0])
    local_rank = int(local_rank)
    topk_idx_offset = int(topk_idx_offset)
    num_tokens = int(num_tokens)
    top_k = int(top_k)
    num_experts_per_rank = int(num_experts_per_rank)
    max_routes_per_pair = int(max_routes_per_pair)
    ep_size = int(ep_size)

    _validate_topk_route_metadata_inputs(
        topk_idx_workspace,
        token_counts,
        route_counts,
        src_token_topk_idx,
        token_src_metadata,
        local_rank,
        topk_idx_offset,
        num_tokens,
        top_k,
        num_experts_per_rank,
        max_routes_per_pair,
        ep_size,
    )

    topk_idx_workspace_cute = _to_dynamic_cute_tensor(topk_idx_workspace)
    token_counts_cute = _to_dynamic_cute_tensor(token_counts)
    route_counts_cute = _to_dynamic_cute_tensor(route_counts)
    src_token_topk_idx_cute = _to_dynamic_cute_tensor(src_token_topk_idx)
    token_src_metadata_cute = _to_dynamic_cute_tensor(token_src_metadata)
    stream = cuda.CUstream(torch.cuda.current_stream(topk_idx_workspace.device).cuda_stream)

    device_index = topk_idx_workspace.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compile_key = (
        torch.cuda.get_device_capability(device_index),
        _tensor_compile_signature(topk_idx_workspace),
        _tensor_compile_signature(token_counts),
        _tensor_compile_signature(route_counts),
        _tensor_compile_signature(src_token_topk_idx),
        _tensor_compile_signature(token_src_metadata),
        local_rank,
        topk_idx_offset,
        num_tokens,
        top_k,
        num_experts_per_rank,
        max_routes_per_pair,
        ep_size,
    )
    if compile_key not in _topk_route_metadata_compile_cache:
        _topk_route_metadata_compile_cache[compile_key] = cute.compile(
            launch_mnnvl_topk_route_metadata,
            topk_idx_workspace_cute,
            token_counts_cute,
            route_counts_cute,
            src_token_topk_idx_cute,
            token_src_metadata_cute,
            local_rank,
            topk_idx_offset,
            num_tokens,
            top_k,
            num_experts_per_rank,
            max_routes_per_pair,
            ep_size,
            stream,
        )

    _topk_route_metadata_compile_cache[compile_key](
        topk_idx_workspace_cute,
        token_counts_cute,
        route_counts_cute,
        src_token_topk_idx_cute,
        token_src_metadata_cute,
        stream,
    )


def build_mnnvl_pool_route_metadata(
    route_counts: torch.Tensor,
    src_token_topk_idx: torch.Tensor,
    token_src_metadata: torch.Tensor,
    l1_arrival_count: torch.Tensor,
    *,
    top_k: int,
    num_experts_per_rank: int,
    max_routes_per_pair: int,
    tile_size: int,
    num_max_pool_tokens: int,
    ep_size: int | None = None,
) -> None:
    """Compile/cache and launch pool-ordered route metadata construction."""
    if ep_size is None:
        ep_size = int(route_counts.numel()) // int(num_experts_per_rank)
    top_k = int(top_k)
    num_experts_per_rank = int(num_experts_per_rank)
    max_routes_per_pair = int(max_routes_per_pair)
    tile_size = int(tile_size)
    num_max_pool_tokens = int(num_max_pool_tokens)
    ep_size = int(ep_size)

    _validate_pool_route_metadata_inputs(
        route_counts,
        src_token_topk_idx,
        token_src_metadata,
        l1_arrival_count,
        top_k,
        num_experts_per_rank,
        ep_size,
        max_routes_per_pair,
        tile_size,
        num_max_pool_tokens,
    )

    route_counts_cute = _to_dynamic_cute_tensor(route_counts)
    src_token_topk_idx_cute = _to_dynamic_cute_tensor(src_token_topk_idx)
    token_src_metadata_cute = _to_dynamic_cute_tensor(token_src_metadata)
    l1_arrival_count_cute = _to_dynamic_cute_tensor(l1_arrival_count)
    stream = cuda.CUstream(torch.cuda.current_stream(route_counts.device).cuda_stream)

    device_index = route_counts.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compile_key = (
        torch.cuda.get_device_capability(device_index),
        _tensor_compile_signature(route_counts),
        _tensor_compile_signature(src_token_topk_idx),
        _tensor_compile_signature(token_src_metadata),
        _tensor_compile_signature(l1_arrival_count),
        top_k,
        num_experts_per_rank,
        ep_size,
        max_routes_per_pair,
        tile_size,
        num_max_pool_tokens,
    )
    if compile_key not in _pool_route_metadata_compile_cache:
        _pool_route_metadata_compile_cache[compile_key] = cute.compile(
            launch_mnnvl_pool_route_metadata,
            route_counts_cute,
            src_token_topk_idx_cute,
            token_src_metadata_cute,
            l1_arrival_count_cute,
            top_k,
            num_experts_per_rank,
            ep_size,
            max_routes_per_pair,
            tile_size,
            num_max_pool_tokens,
            stream,
        )

    _pool_route_metadata_compile_cache[compile_key](
        route_counts_cute,
        src_token_topk_idx_cute,
        token_src_metadata_cute,
        l1_arrival_count_cute,
        stream,
    )


def count_mnnvl_topk_routes(
    topk_idx_workspace: torch.Tensor,
    route_counts: torch.Tensor,
    *,
    local_rank: int,
    topk_idx_offset: int,
    num_tokens: int,
    top_k: int,
    num_experts_per_rank: int,
    ep_size: int | None = None,
) -> None:
    """Compile/cache and launch the MNNVL top-k route-count debug kernel."""
    if ep_size is None:
        ep_size = int(topk_idx_workspace.shape[0])
    local_rank = int(local_rank)
    topk_idx_offset = int(topk_idx_offset)
    num_tokens = int(num_tokens)
    top_k = int(top_k)
    num_experts_per_rank = int(num_experts_per_rank)
    ep_size = int(ep_size)

    _validate_topk_route_count_inputs(
        topk_idx_workspace,
        route_counts,
        local_rank,
        topk_idx_offset,
        num_tokens,
        top_k,
        num_experts_per_rank,
        ep_size,
    )

    topk_idx_workspace_cute = _to_dynamic_cute_tensor(topk_idx_workspace)
    route_counts_cute = _to_dynamic_cute_tensor(route_counts)
    stream = cuda.CUstream(torch.cuda.current_stream(topk_idx_workspace.device).cuda_stream)

    device_index = topk_idx_workspace.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compile_key = (
        torch.cuda.get_device_capability(device_index),
        _tensor_compile_signature(topk_idx_workspace),
        _tensor_compile_signature(route_counts),
        local_rank,
        topk_idx_offset,
        num_tokens,
        top_k,
        num_experts_per_rank,
        ep_size,
    )
    if compile_key not in _topk_route_count_compile_cache:
        _topk_route_count_compile_cache[compile_key] = cute.compile(
            launch_mnnvl_topk_route_count,
            topk_idx_workspace_cute,
            route_counts_cute,
            local_rank,
            topk_idx_offset,
            num_tokens,
            top_k,
            num_experts_per_rank,
            ep_size,
            stream,
        )

    _topk_route_count_compile_cache[compile_key](
        topk_idx_workspace_cute,
        route_counts_cute,
        stream,
    )


def pull_mnnvl_route_payload(
    workspace: torch.Tensor,
    signal: torch.Tensor,
    route_metadata: torch.Tensor,
    pulled: torch.Tensor,
    observed_signal: torch.Tensor,
    *,
    peer_rank: int,
    local_rank: int,
    payload_byte_offset: int,
    row_bytes: int,
    num_tokens: int,
    token_stride_rows: int = 1,
    topk_stride_rows: int = 0,
    num_routes: int | None = None,
    done_count_signal_u32_offset: int,
    peer_done_signal_u32_offset: int,
) -> None:
    """Compile/cache and launch the route-metadata-driven MNNVL byte-row pull kernel."""
    if num_routes is None:
        num_routes = int(route_metadata.shape[0])
    payload_byte_offset = int(payload_byte_offset)
    row_bytes = int(row_bytes)
    num_tokens = int(num_tokens)
    token_stride_rows = int(token_stride_rows)
    topk_stride_rows = int(topk_stride_rows)
    num_routes = int(num_routes)
    peer_rank = int(peer_rank)
    local_rank = int(local_rank)
    done_count_signal_u32_offset = int(done_count_signal_u32_offset)
    peer_done_signal_u32_offset = int(peer_done_signal_u32_offset)

    if payload_byte_offset < 0 or num_tokens < 0 or token_stride_rows < 0 or topk_stride_rows < 0:
        raise ValueError("payload offset and row-count parameters must be non-negative")

    _validate_route_payload_pull_inputs(
        workspace,
        signal,
        route_metadata,
        pulled,
        observed_signal,
        peer_rank,
        local_rank,
        row_bytes,
        num_routes,
        done_count_signal_u32_offset,
        peer_done_signal_u32_offset,
    )

    workspace_cute = _to_dynamic_cute_tensor(workspace)
    signal_cute = _to_dynamic_cute_tensor(signal)
    route_metadata_cute = _to_dynamic_cute_tensor(route_metadata)
    pulled_cute = _to_dynamic_cute_tensor(pulled)
    observed_signal_cute = _to_dynamic_cute_tensor(observed_signal)
    stream = cuda.CUstream(torch.cuda.current_stream(workspace.device).cuda_stream)

    device_index = workspace.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compile_key = (
        torch.cuda.get_device_capability(device_index),
        _tensor_compile_signature(workspace),
        _tensor_compile_signature(signal),
        _tensor_compile_signature(route_metadata),
        _tensor_compile_signature(pulled),
        _tensor_compile_signature(observed_signal),
        peer_rank,
        local_rank,
        payload_byte_offset,
        row_bytes,
        num_tokens,
        token_stride_rows,
        topk_stride_rows,
        num_routes,
        done_count_signal_u32_offset,
        peer_done_signal_u32_offset,
    )
    if compile_key not in _route_payload_pull_compile_cache:
        _route_payload_pull_compile_cache[compile_key] = cute.compile(
            launch_mnnvl_route_payload_pull,
            workspace_cute,
            signal_cute,
            route_metadata_cute,
            pulled_cute,
            observed_signal_cute,
            peer_rank,
            local_rank,
            payload_byte_offset,
            row_bytes,
            num_tokens,
            token_stride_rows,
            topk_stride_rows,
            num_routes,
            done_count_signal_u32_offset,
            peer_done_signal_u32_offset,
            stream,
        )

    _route_payload_pull_compile_cache[compile_key](
        workspace_cute,
        signal_cute,
        route_metadata_cute,
        pulled_cute,
        observed_signal_cute,
        stream,
    )


__all__ = [
    "build_mnnvl_pool_route_metadata",
    "build_mnnvl_topk_route_metadata",
    "count_mnnvl_topk_routes",
    "launch_mnnvl_pool_route_metadata",
    "launch_mnnvl_route_payload_pull",
    "launch_mnnvl_topk_route_count",
    "launch_mnnvl_topk_route_metadata",
    "pull_mnnvl_route_payload",
]
