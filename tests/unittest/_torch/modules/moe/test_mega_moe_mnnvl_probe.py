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
"""M4 probe for MegaMoE symmetric-memory peer access from CuTe DSL."""

import socket

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack
from mpi4py import MPI

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.mega_moe_mnnvl import (
    build_mnnvl_pool_route_metadata,
    build_mnnvl_topk_route_metadata,
    count_mnnvl_topk_routes,
    pull_mnnvl_route_payload,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    ld_acquire_sys_u32,
    red_add_release_sys_u32,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

_SEGMENT_BYTES = 4096
_PROBE_BYTES = 256
_PAYLOAD_BYTE_OFFSET = 512
_SIGNAL_U32_OFFSET = 0
_ROUTE_PAYLOAD_BYTE_OFFSET = 1024
_TOKEN_PAYLOAD_BYTE_OFFSET = 2048
_WEIGHT_PAYLOAD_BYTE_OFFSET = 3072
_ROUTE_ROW_BYTES = 64
_WEIGHT_ROW_BYTES = 4
_ROUTE_NUM_TOKENS = 4
_ROUTE_TOP_K = 2
_ROUTE_NUM_ROUTES = 7
_ROUTE_DONE_COUNT_SIGNAL_U32_OFFSET = 2
_ROUTE_PEER_DONE_SIGNAL_U32_OFFSET = 3
_TOKEN_DONE_COUNT_SIGNAL_U32_OFFSET = 4
_TOKEN_PEER_DONE_SIGNAL_U32_OFFSET = 5
_WEIGHT_DONE_COUNT_SIGNAL_U32_OFFSET = 6
_WEIGHT_PEER_DONE_SIGNAL_U32_OFFSET = 7
_TOPK_IDX_I64_OFFSET = 448
_COUNT_NUM_TOKENS = 4
_COUNT_TOP_K = 2
_COUNT_NUM_EXPERTS_PER_RANK = 2
_COUNT_MAX_ROUTES_PER_PAIR = _COUNT_NUM_TOKENS * _COUNT_TOP_K
_COUNT_POOL_TILE_SIZE = 4


def _skip_if_not_blackwell() -> None:
    sm = get_sm_version()
    if sm not in {100, 103}:
        pytest.skip(f"MegaMoE requires SM100/103, got SM{sm}")


def _local_rank_and_world_size(comm: MPI.Comm) -> tuple[int, int]:
    hostname = socket.gethostname()
    hostnames = comm.allgather(hostname)
    rank = comm.Get_rank()
    local_rank = sum(1 for other in hostnames[:rank] if other == hostname)
    local_world_size = hostnames.count(hostname)
    return local_rank, local_world_size


@cute.kernel
def _mnnvl_peer_pull_probe_kernel(
    workspace: cute.Tensor,
    signal: cute.Tensor,
    pulled: cute.Tensor,
    observed_signal: cute.Tensor,
    peer_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()

    pulled[tidx] = workspace[peer_rank, _PAYLOAD_BYTE_OFFSET + tidx]

    if tidx == 0:
        peer_signal_ptr = signal.iterator + cute.crd2idx(
            (peer_rank, _SIGNAL_U32_OFFSET), signal.layout
        )
        local_signal_ptr = signal.iterator + cute.crd2idx(
            (local_rank, _SIGNAL_U32_OFFSET), signal.layout
        )
        red_add_release_sys_u32(peer_signal_ptr, cutlass.Uint32(1))

        cached = cutlass.Uint32(0)
        while cached < cutlass.Uint32(1):
            cached = ld_acquire_sys_u32(local_signal_ptr)
        observed_signal[0] = cached


@cute.jit
def _launch_mnnvl_peer_pull_probe(
    workspace: cute.Tensor,
    signal: cute.Tensor,
    pulled: cute.Tensor,
    observed_signal: cute.Tensor,
    peer_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    _mnnvl_peer_pull_probe_kernel(
        workspace, signal, pulled, observed_signal, peer_rank, local_rank
    ).launch(
        grid=(1, 1, 1),
        block=[_PROBE_BYTES, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


@pytest.mark.gpu
def test_megamoe_m4_cute_dsl_mnnvl_peer_pull_probe() -> None:
    """Prove the M4 peer-pull substrate needed before in-kernel dispatch fusion.

    Each rank writes a deterministic byte payload into its local MNNVL segment.
    The CuTe DSL kernel then reads the next rank's segment through the rank-
    strided symmetric-memory mapping and performs a sys-scope release/acquire
    signal exchange. This is the smallest executable check for the M5/M6
    assumption that a MegaMoE kernel can directly address peer rank workspace.
    """
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size < 2:
        pytest.skip("requires static MPI launch with at least 2 ranks")

    local_rank, local_world_size = _local_rank_and_world_size(comm)
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"need {local_world_size} local GPUs, have {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    _skip_if_not_blackwell()
    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("MNNVL symmetric memory is not supported on this platform")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
    )
    mnnvl_memory = MnnvlMemory(mapping, _SEGMENT_BYTES)
    workspace = mnnvl_memory.as_torch_strided_tensor(torch.uint8)
    signal = mnnvl_memory.as_torch_strided_tensor(torch.uint32)

    workspace[rank].zero_()
    local_payload = (
        (torch.arange(_PROBE_BYTES, device="cuda", dtype=torch.int16) + rank * 37) % 256
    ).to(torch.uint8)
    workspace[rank, _PAYLOAD_BYTE_OFFSET : _PAYLOAD_BYTE_OFFSET + _PROBE_BYTES].copy_(local_payload)
    torch.cuda.synchronize()
    comm.barrier()

    peer_rank = (rank + 1) % world_size
    pulled_torch = torch.empty(_PROBE_BYTES, dtype=torch.uint8, device="cuda")
    observed_signal_torch = torch.zeros(1, dtype=torch.uint32, device="cuda")

    workspace_cute = from_dlpack(workspace).mark_layout_dynamic()
    signal_cute = from_dlpack(signal).mark_layout_dynamic()
    pulled_cute = from_dlpack(pulled_torch).mark_layout_dynamic()
    observed_signal_cute = from_dlpack(observed_signal_torch).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = cute.compile(
        _launch_mnnvl_peer_pull_probe,
        workspace_cute,
        signal_cute,
        pulled_cute,
        observed_signal_cute,
        peer_rank,
        rank,
        stream,
    )
    compiled(
        workspace_cute,
        signal_cute,
        pulled_cute,
        observed_signal_cute,
        stream,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_peer_payload = (
        (torch.arange(_PROBE_BYTES, dtype=torch.int16) + peer_rank * 37) % 256
    ).to(torch.uint8)
    assert torch.equal(pulled_torch.cpu(), expected_peer_payload)
    assert int(observed_signal_torch.cpu().item()) == 1
    assert int(signal[rank, _SIGNAL_U32_OFFSET].cpu().item()) == 1

    comm.barrier()


@pytest.mark.gpu
def test_megamoe_m4_cute_dsl_mnnvl_route_payload_pull_probe() -> None:
    """Prove route-metadata-driven peer payload pulls from MNNVL segments.

    This is the next scaffold after the fixed-offset peer probe: each CTA reads
    one route descriptor shaped like MegaMoE ``token_src_metadata`` and pulls a
    top-k/token payload row from the selected source rank's symmetric segment.
    """
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size < 2:
        pytest.skip("requires static MPI launch with at least 2 ranks")

    local_rank, local_world_size = _local_rank_and_world_size(comm)
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"need {local_world_size} local GPUs, have {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    _skip_if_not_blackwell()
    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("MNNVL symmetric memory is not supported on this platform")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
    )
    mnnvl_memory = MnnvlMemory(mapping, _SEGMENT_BYTES)
    workspace = mnnvl_memory.as_torch_strided_tensor(torch.uint8)
    signal = mnnvl_memory.as_torch_strided_tensor(torch.uint32)

    workspace[rank].zero_()
    byte_ids = torch.arange(_ROUTE_ROW_BYTES, device="cuda", dtype=torch.int16).view(
        1, 1, _ROUTE_ROW_BYTES
    )
    topk_ids = torch.arange(_ROUTE_TOP_K, device="cuda", dtype=torch.int16).view(_ROUTE_TOP_K, 1, 1)
    token_ids = torch.arange(_ROUTE_NUM_TOKENS, device="cuda", dtype=torch.int16).view(
        1, _ROUTE_NUM_TOKENS, 1
    )
    rank_payload = (rank * 53 + topk_ids * 17 + token_ids * 7 + byte_ids).to(torch.uint8)
    workspace[
        rank, _ROUTE_PAYLOAD_BYTE_OFFSET : _ROUTE_PAYLOAD_BYTE_OFFSET + rank_payload.numel()
    ].copy_(rank_payload.reshape(-1))
    torch.cuda.synchronize()
    comm.barrier()

    peer_rank = (rank + 1) % world_size
    route_metadata_cpu = torch.tensor(
        (
            (rank, 0, 0),
            (peer_rank, 1, 1),
            (peer_rank, 2, 0),
            (-1, -1, -1),
            (rank, 3, 1),
            (peer_rank, 0, 1),
            (rank, 2, 0),
        ),
        dtype=torch.int32,
    )
    route_metadata_torch = route_metadata_cpu.to(device="cuda")
    pulled_torch = torch.empty(
        (_ROUTE_NUM_ROUTES, _ROUTE_ROW_BYTES), dtype=torch.uint8, device="cuda"
    )
    observed_signal_torch = torch.zeros(2, dtype=torch.uint32, device="cuda")

    pull_mnnvl_route_payload(
        workspace,
        signal,
        route_metadata_torch,
        pulled_torch,
        observed_signal_torch,
        peer_rank=peer_rank,
        local_rank=rank,
        payload_byte_offset=_ROUTE_PAYLOAD_BYTE_OFFSET,
        row_bytes=_ROUTE_ROW_BYTES,
        num_tokens=_ROUTE_NUM_TOKENS,
        token_stride_rows=1,
        topk_stride_rows=_ROUTE_NUM_TOKENS,
        num_routes=_ROUTE_NUM_ROUTES,
        done_count_signal_u32_offset=_ROUTE_DONE_COUNT_SIGNAL_U32_OFFSET,
        peer_done_signal_u32_offset=_ROUTE_PEER_DONE_SIGNAL_U32_OFFSET,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_rows = []
    for src_rank, token_idx, topk_idx in route_metadata_cpu.tolist():
        if src_rank < 0:
            expected_rows.append(torch.zeros((1, _ROUTE_ROW_BYTES), dtype=torch.uint8))
        else:
            expected_rows.append(
                (src_rank * 53 + topk_idx * 17 + token_idx * 7 + torch.arange(_ROUTE_ROW_BYTES))
                .to(torch.uint8)
                .view(1, _ROUTE_ROW_BYTES)
            )
    expected = torch.cat(expected_rows, dim=0)
    assert torch.equal(pulled_torch.cpu(), expected)
    assert int(observed_signal_torch[0].cpu().item()) == 1
    assert int(observed_signal_torch[1].cpu().item()) == _ROUTE_NUM_ROUTES
    assert int(signal[rank, _ROUTE_PEER_DONE_SIGNAL_U32_OFFSET].cpu().item()) == 1
    assert int(signal[rank, _ROUTE_DONE_COUNT_SIGNAL_U32_OFFSET].cpu().item()) == _ROUTE_NUM_ROUTES

    token_payload = (rank * 61 + token_ids * 11 + byte_ids).to(torch.uint8)
    workspace[
        rank, _TOKEN_PAYLOAD_BYTE_OFFSET : _TOKEN_PAYLOAD_BYTE_OFFSET + token_payload.numel()
    ].copy_(token_payload.reshape(-1))
    signal[rank, _TOKEN_DONE_COUNT_SIGNAL_U32_OFFSET] = 0
    signal[rank, _TOKEN_PEER_DONE_SIGNAL_U32_OFFSET] = 0
    pulled_token_torch = torch.full(
        (_ROUTE_NUM_ROUTES, _ROUTE_ROW_BYTES), 0xEF, dtype=torch.uint8, device="cuda"
    )
    observed_token_signal_torch = torch.zeros(2, dtype=torch.uint32, device="cuda")
    torch.cuda.synchronize()
    comm.barrier()

    pull_mnnvl_route_payload(
        workspace,
        signal,
        route_metadata_torch,
        pulled_token_torch,
        observed_token_signal_torch,
        peer_rank=peer_rank,
        local_rank=rank,
        payload_byte_offset=_TOKEN_PAYLOAD_BYTE_OFFSET,
        row_bytes=_ROUTE_ROW_BYTES,
        num_tokens=_ROUTE_NUM_TOKENS,
        token_stride_rows=1,
        topk_stride_rows=0,
        num_routes=_ROUTE_NUM_ROUTES,
        done_count_signal_u32_offset=_TOKEN_DONE_COUNT_SIGNAL_U32_OFFSET,
        peer_done_signal_u32_offset=_TOKEN_PEER_DONE_SIGNAL_U32_OFFSET,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_token_rows = []
    for src_rank, token_idx, _ in route_metadata_cpu.tolist():
        if src_rank < 0:
            expected_token_rows.append(torch.zeros((1, _ROUTE_ROW_BYTES), dtype=torch.uint8))
        else:
            expected_token_rows.append(
                (src_rank * 61 + token_idx * 11 + torch.arange(_ROUTE_ROW_BYTES))
                .to(torch.uint8)
                .view(1, _ROUTE_ROW_BYTES)
            )
    expected_token = torch.cat(expected_token_rows, dim=0)
    assert torch.equal(pulled_token_torch.cpu(), expected_token)
    assert int(observed_token_signal_torch[0].cpu().item()) == 1
    assert int(observed_token_signal_torch[1].cpu().item()) == _ROUTE_NUM_ROUTES
    assert int(signal[rank, _TOKEN_PEER_DONE_SIGNAL_U32_OFFSET].cpu().item()) == 1
    assert int(signal[rank, _TOKEN_DONE_COUNT_SIGNAL_U32_OFFSET].cpu().item()) == _ROUTE_NUM_ROUTES

    weight_byte_ids = torch.arange(_WEIGHT_ROW_BYTES, device="cuda", dtype=torch.int16).view(
        1, 1, _WEIGHT_ROW_BYTES
    )
    weight_token_ids = torch.arange(_ROUTE_NUM_TOKENS, device="cuda", dtype=torch.int16).view(
        _ROUTE_NUM_TOKENS, 1, 1
    )
    weight_topk_ids = torch.arange(_ROUTE_TOP_K, device="cuda", dtype=torch.int16).view(
        1, _ROUTE_TOP_K, 1
    )
    weight_payload = (
        rank * 67 + weight_token_ids * 13 + weight_topk_ids * 19 + weight_byte_ids
    ).to(torch.uint8)
    workspace[
        rank, _WEIGHT_PAYLOAD_BYTE_OFFSET : _WEIGHT_PAYLOAD_BYTE_OFFSET + weight_payload.numel()
    ].copy_(weight_payload.reshape(-1))
    signal[rank, _WEIGHT_DONE_COUNT_SIGNAL_U32_OFFSET] = 0
    signal[rank, _WEIGHT_PEER_DONE_SIGNAL_U32_OFFSET] = 0
    pulled_weight_torch = torch.full(
        (_ROUTE_NUM_ROUTES, _WEIGHT_ROW_BYTES), 0x7D, dtype=torch.uint8, device="cuda"
    )
    observed_weight_signal_torch = torch.zeros(2, dtype=torch.uint32, device="cuda")
    torch.cuda.synchronize()
    comm.barrier()

    pull_mnnvl_route_payload(
        workspace,
        signal,
        route_metadata_torch,
        pulled_weight_torch,
        observed_weight_signal_torch,
        peer_rank=peer_rank,
        local_rank=rank,
        payload_byte_offset=_WEIGHT_PAYLOAD_BYTE_OFFSET,
        row_bytes=_WEIGHT_ROW_BYTES,
        num_tokens=_ROUTE_NUM_TOKENS,
        token_stride_rows=_ROUTE_TOP_K,
        topk_stride_rows=1,
        num_routes=_ROUTE_NUM_ROUTES,
        done_count_signal_u32_offset=_WEIGHT_DONE_COUNT_SIGNAL_U32_OFFSET,
        peer_done_signal_u32_offset=_WEIGHT_PEER_DONE_SIGNAL_U32_OFFSET,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_weight_rows = []
    for src_rank, token_idx, topk_idx in route_metadata_cpu.tolist():
        if src_rank < 0:
            expected_weight_rows.append(torch.zeros((1, _WEIGHT_ROW_BYTES), dtype=torch.uint8))
        else:
            expected_weight_rows.append(
                (src_rank * 67 + token_idx * 13 + topk_idx * 19 + torch.arange(_WEIGHT_ROW_BYTES))
                .to(torch.uint8)
                .view(1, _WEIGHT_ROW_BYTES)
            )
    expected_weight = torch.cat(expected_weight_rows, dim=0)
    assert torch.equal(pulled_weight_torch.cpu(), expected_weight)
    assert int(observed_weight_signal_torch[0].cpu().item()) == 1
    assert int(observed_weight_signal_torch[1].cpu().item()) == _ROUTE_NUM_ROUTES
    assert int(signal[rank, _WEIGHT_PEER_DONE_SIGNAL_U32_OFFSET].cpu().item()) == 1
    assert int(signal[rank, _WEIGHT_DONE_COUNT_SIGNAL_U32_OFFSET].cpu().item()) == _ROUTE_NUM_ROUTES

    comm.barrier()


@pytest.mark.gpu
def test_megamoe_m5_cute_dsl_mnnvl_topk_route_count_probe() -> None:
    """Count local-expert routes from rank-strided MNNVL top-k metadata.

    This M5 scaffold verifies the next metadata-ownership step after payload
    pulls: every rank stages its ``topk_idx`` rows in symmetric memory, then
    each destination rank counts only the routes assigned to its local experts.
    """
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size < 2:
        pytest.skip("requires static MPI launch with at least 2 ranks")

    local_rank, local_world_size = _local_rank_and_world_size(comm)
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"need {local_world_size} local GPUs, have {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    _skip_if_not_blackwell()
    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("MNNVL symmetric memory is not supported on this platform")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
    )
    mnnvl_memory = MnnvlMemory(mapping, _SEGMENT_BYTES)
    topk_idx_workspace = mnnvl_memory.as_torch_strided_tensor(torch.int64)

    topk_idx_workspace[rank].zero_()
    token_ids = torch.arange(_COUNT_NUM_TOKENS, device="cuda", dtype=torch.int64).view(
        _COUNT_NUM_TOKENS, 1
    )
    topk_ordinals = torch.arange(_COUNT_TOP_K, device="cuda", dtype=torch.int64).view(
        1, _COUNT_TOP_K
    )
    num_experts = world_size * _COUNT_NUM_EXPERTS_PER_RANK
    local_topk_idx = (rank * 3 + token_ids * 2 + topk_ordinals) % num_experts
    topk_idx_workspace[
        rank, _TOPK_IDX_I64_OFFSET : _TOPK_IDX_I64_OFFSET + local_topk_idx.numel()
    ].copy_(local_topk_idx.reshape(-1))
    torch.cuda.synchronize()
    comm.barrier()

    route_counts = torch.zeros(
        world_size * _COUNT_NUM_EXPERTS_PER_RANK, dtype=torch.uint32, device="cuda"
    )
    count_mnnvl_topk_routes(
        topk_idx_workspace,
        route_counts,
        local_rank=rank,
        topk_idx_offset=_TOPK_IDX_I64_OFFSET,
        num_tokens=_COUNT_NUM_TOKENS,
        top_k=_COUNT_TOP_K,
        num_experts_per_rank=_COUNT_NUM_EXPERTS_PER_RANK,
        ep_size=world_size,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_counts = torch.zeros((world_size, _COUNT_NUM_EXPERTS_PER_RANK), dtype=torch.int64)
    expert_start = rank * _COUNT_NUM_EXPERTS_PER_RANK
    expert_end = expert_start + _COUNT_NUM_EXPERTS_PER_RANK
    for source_rank in range(world_size):
        for token_idx in range(_COUNT_NUM_TOKENS):
            for topk_ordinal in range(_COUNT_TOP_K):
                expert_idx = (source_rank * 3 + token_idx * 2 + topk_ordinal) % num_experts
                if expert_start <= expert_idx < expert_end:
                    expected_counts[source_rank, expert_idx - expert_start] += 1

    observed_counts = (
        route_counts.reshape(world_size, _COUNT_NUM_EXPERTS_PER_RANK).cpu().to(torch.int64)
    )
    assert torch.equal(observed_counts, expected_counts)

    comm.barrier()


@pytest.mark.gpu
def test_megamoe_m5_cute_dsl_mnnvl_topk_route_metadata_probe() -> None:
    """Build route metadata tables from rank-strided MNNVL top-k metadata."""
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size < 2:
        pytest.skip("requires static MPI launch with at least 2 ranks")

    local_rank, local_world_size = _local_rank_and_world_size(comm)
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"need {local_world_size} local GPUs, have {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    _skip_if_not_blackwell()
    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("MNNVL symmetric memory is not supported on this platform")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
    )
    mnnvl_memory = MnnvlMemory(mapping, _SEGMENT_BYTES)
    topk_idx_workspace = mnnvl_memory.as_torch_strided_tensor(torch.int64)

    topk_idx_workspace[rank].zero_()
    token_counts_cpu = torch.tensor(
        [max(1, _COUNT_NUM_TOKENS - (source_rank % 2)) for source_rank in range(world_size)],
        dtype=torch.int32,
    )
    token_counts = token_counts_cpu.to(device="cuda")
    token_ids = torch.arange(_COUNT_NUM_TOKENS, device="cuda", dtype=torch.int64).view(
        _COUNT_NUM_TOKENS, 1
    )
    topk_ordinals = torch.arange(_COUNT_TOP_K, device="cuda", dtype=torch.int64).view(
        1, _COUNT_TOP_K
    )
    num_experts = world_size * _COUNT_NUM_EXPERTS_PER_RANK
    local_topk_idx = (rank * 3 + token_ids * 2 + topk_ordinals) % num_experts
    local_token_count = int(token_counts_cpu[rank].item())
    poison_expert_idx = rank * _COUNT_NUM_EXPERTS_PER_RANK
    local_topk_idx[local_token_count:, :] = poison_expert_idx
    topk_idx_workspace[
        rank, _TOPK_IDX_I64_OFFSET : _TOPK_IDX_I64_OFFSET + local_topk_idx.numel()
    ].copy_(local_topk_idx.reshape(-1))
    torch.cuda.synchronize()
    comm.barrier()

    pair_count = world_size * _COUNT_NUM_EXPERTS_PER_RANK
    route_slot_count = pair_count * _COUNT_MAX_ROUTES_PER_PAIR
    route_counts = torch.zeros(pair_count, dtype=torch.int32, device="cuda")
    src_token_topk_idx = torch.full((route_slot_count,), -1, dtype=torch.int32, device="cuda")
    token_src_metadata = torch.full((route_slot_count, 3), -1, dtype=torch.int32, device="cuda")

    build_mnnvl_topk_route_metadata(
        topk_idx_workspace,
        token_counts,
        route_counts,
        src_token_topk_idx,
        token_src_metadata,
        local_rank=rank,
        topk_idx_offset=_TOPK_IDX_I64_OFFSET,
        num_tokens=_COUNT_NUM_TOKENS,
        top_k=_COUNT_TOP_K,
        num_experts_per_rank=_COUNT_NUM_EXPERTS_PER_RANK,
        max_routes_per_pair=_COUNT_MAX_ROUTES_PER_PAIR,
        ep_size=world_size,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_counts = torch.zeros((world_size, _COUNT_NUM_EXPERTS_PER_RANK), dtype=torch.int32)
    expected_src_token_topk_idx = torch.full(
        (
            _COUNT_NUM_EXPERTS_PER_RANK,
            world_size,
            _COUNT_MAX_ROUTES_PER_PAIR,
        ),
        -1,
        dtype=torch.int32,
    )
    expected_token_src_metadata = torch.full((route_slot_count, 3), -1, dtype=torch.int32)
    expert_start = rank * _COUNT_NUM_EXPERTS_PER_RANK
    expert_end = expert_start + _COUNT_NUM_EXPERTS_PER_RANK
    for source_rank in range(world_size):
        source_token_count = int(token_counts_cpu[source_rank].item())
        for token_idx in range(source_token_count):
            for topk_ordinal in range(_COUNT_TOP_K):
                expert_idx = (source_rank * 3 + token_idx * 2 + topk_ordinal) % num_experts
                if not (expert_start <= expert_idx < expert_end):
                    continue

                local_expert_idx = expert_idx - expert_start
                route_ordinal = int(expected_counts[source_rank, local_expert_idx].item())
                src_idx = token_idx * _COUNT_TOP_K + topk_ordinal
                expected_counts[source_rank, local_expert_idx] += 1
                expected_src_token_topk_idx[local_expert_idx, source_rank, route_ordinal] = src_idx
                route_slot = (
                    local_expert_idx * world_size + source_rank
                ) * _COUNT_MAX_ROUTES_PER_PAIR + route_ordinal
                expected_token_src_metadata[route_slot] = torch.tensor(
                    (source_rank, token_idx, topk_ordinal), dtype=torch.int32
                )

    observed_counts = route_counts.reshape(world_size, _COUNT_NUM_EXPERTS_PER_RANK).cpu()
    observed_src_token_topk_idx = src_token_topk_idx.reshape(
        _COUNT_NUM_EXPERTS_PER_RANK, world_size, _COUNT_MAX_ROUTES_PER_PAIR
    ).cpu()
    assert torch.equal(observed_counts, expected_counts)
    assert torch.equal(observed_src_token_topk_idx, expected_src_token_topk_idx)
    assert torch.equal(token_src_metadata.cpu(), expected_token_src_metadata)

    comm.barrier()


@pytest.mark.gpu
def test_megamoe_m5_cute_dsl_mnnvl_pool_route_metadata_probe() -> None:
    """Build pool-ordered route metadata from source-major M5 route tables."""
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    if world_size < 2:
        pytest.skip("requires static MPI launch with at least 2 ranks")

    local_rank, local_world_size = _local_rank_and_world_size(comm)
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"need {local_world_size} local GPUs, have {torch.cuda.device_count()}")

    torch.cuda.set_device(local_rank)
    _skip_if_not_blackwell()
    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("MNNVL symmetric memory is not supported on this platform")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
    )
    mnnvl_memory = MnnvlMemory(mapping, _SEGMENT_BYTES)
    topk_idx_workspace = mnnvl_memory.as_torch_strided_tensor(torch.int64)

    topk_idx_workspace[rank].zero_()
    token_counts_cpu = torch.tensor(
        [max(1, _COUNT_NUM_TOKENS - (source_rank % 2)) for source_rank in range(world_size)],
        dtype=torch.int32,
    )
    token_counts = token_counts_cpu.to(device="cuda")
    token_ids = torch.arange(_COUNT_NUM_TOKENS, device="cuda", dtype=torch.int64).view(
        _COUNT_NUM_TOKENS, 1
    )
    topk_ordinals = torch.arange(_COUNT_TOP_K, device="cuda", dtype=torch.int64).view(
        1, _COUNT_TOP_K
    )
    num_experts = world_size * _COUNT_NUM_EXPERTS_PER_RANK
    local_topk_idx = (rank * 3 + token_ids * 2 + topk_ordinals) % num_experts
    local_token_count = int(token_counts_cpu[rank].item())
    poison_expert_idx = rank * _COUNT_NUM_EXPERTS_PER_RANK
    local_topk_idx[local_token_count:, :] = poison_expert_idx
    topk_idx_workspace[
        rank, _TOPK_IDX_I64_OFFSET : _TOPK_IDX_I64_OFFSET + local_topk_idx.numel()
    ].copy_(local_topk_idx.reshape(-1))
    torch.cuda.synchronize()
    comm.barrier()

    pair_count = world_size * _COUNT_NUM_EXPERTS_PER_RANK
    route_slot_count = pair_count * _COUNT_MAX_ROUTES_PER_PAIR
    route_counts = torch.zeros(pair_count, dtype=torch.int32, device="cuda")
    src_token_topk_idx = torch.full((route_slot_count,), -1, dtype=torch.int32, device="cuda")
    source_major_metadata = torch.full((route_slot_count, 3), -1, dtype=torch.int32, device="cuda")

    build_mnnvl_topk_route_metadata(
        topk_idx_workspace,
        token_counts,
        route_counts,
        src_token_topk_idx,
        source_major_metadata,
        local_rank=rank,
        topk_idx_offset=_TOPK_IDX_I64_OFFSET,
        num_tokens=_COUNT_NUM_TOKENS,
        top_k=_COUNT_TOP_K,
        num_experts_per_rank=_COUNT_NUM_EXPERTS_PER_RANK,
        max_routes_per_pair=_COUNT_MAX_ROUTES_PER_PAIR,
        ep_size=world_size,
    )

    pool_metadata = torch.full((route_slot_count, 3), -1, dtype=torch.int32, device="cuda")
    l1_arrival_count = torch.zeros(
        (route_slot_count + _COUNT_POOL_TILE_SIZE - 1) // _COUNT_POOL_TILE_SIZE,
        dtype=torch.int32,
        device="cuda",
    )
    build_mnnvl_pool_route_metadata(
        route_counts,
        src_token_topk_idx,
        pool_metadata,
        l1_arrival_count,
        top_k=_COUNT_TOP_K,
        num_experts_per_rank=_COUNT_NUM_EXPERTS_PER_RANK,
        max_routes_per_pair=_COUNT_MAX_ROUTES_PER_PAIR,
        tile_size=_COUNT_POOL_TILE_SIZE,
        num_max_pool_tokens=route_slot_count,
        ep_size=world_size,
    )
    torch.cuda.synchronize()
    comm.barrier()

    expected_counts = torch.zeros((world_size, _COUNT_NUM_EXPERTS_PER_RANK), dtype=torch.int32)
    expected_src_token_topk_idx = torch.full(
        (_COUNT_NUM_EXPERTS_PER_RANK, world_size, _COUNT_MAX_ROUTES_PER_PAIR),
        -1,
        dtype=torch.int32,
    )
    expert_start = rank * _COUNT_NUM_EXPERTS_PER_RANK
    expert_end = expert_start + _COUNT_NUM_EXPERTS_PER_RANK
    for source_rank in range(world_size):
        source_token_count = int(token_counts_cpu[source_rank].item())
        for token_idx in range(source_token_count):
            for topk_ordinal in range(_COUNT_TOP_K):
                expert_idx = (source_rank * 3 + token_idx * 2 + topk_ordinal) % num_experts
                if not (expert_start <= expert_idx < expert_end):
                    continue
                local_expert_idx = expert_idx - expert_start
                route_ordinal = int(expected_counts[source_rank, local_expert_idx].item())
                expected_counts[source_rank, local_expert_idx] += 1
                expected_src_token_topk_idx[local_expert_idx, source_rank, route_ordinal] = (
                    token_idx * _COUNT_TOP_K + topk_ordinal
                )

    expected_pool_metadata = torch.full((route_slot_count, 3), -1, dtype=torch.int32)
    expected_l1_arrival_count = torch.zeros_like(l1_arrival_count.cpu())
    pool_slot = 0
    for local_expert_idx in range(_COUNT_NUM_EXPERTS_PER_RANK):
        expert_pool_start = pool_slot
        max_routes = int(expected_counts[:, local_expert_idx].max().item())
        for route_ordinal in range(max_routes):
            for source_rank in range(world_size):
                if route_ordinal >= int(expected_counts[source_rank, local_expert_idx].item()):
                    continue
                src_idx = int(
                    expected_src_token_topk_idx[local_expert_idx, source_rank, route_ordinal].item()
                )
                expected_pool_metadata[pool_slot] = torch.tensor(
                    (source_rank, src_idx // _COUNT_TOP_K, src_idx % _COUNT_TOP_K),
                    dtype=torch.int32,
                )
                expected_l1_arrival_count[pool_slot // _COUNT_POOL_TILE_SIZE] += 1
                pool_slot += 1
        pool_slot = (
            (expert_pool_start + (pool_slot - expert_pool_start) + _COUNT_POOL_TILE_SIZE - 1)
            // _COUNT_POOL_TILE_SIZE
        ) * _COUNT_POOL_TILE_SIZE

    assert torch.equal(pool_metadata.cpu(), expected_pool_metadata)
    assert torch.equal(l1_arrival_count.cpu(), expected_l1_arrival_count)

    comm.barrier()
