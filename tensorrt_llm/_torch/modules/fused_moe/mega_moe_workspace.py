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
"""Workspace layout helpers for the MegaMoE full-fusion path.

This module is intentionally CPU-only. It records the M4 full-fusion workspace
contract before M5 dispatch-pull and M6 combine-push code starts using the
layout from CUDA/CuTe DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

BARRIER_REGION_BYTES = 512
BF16_BYTES = 2
FLOAT32_BYTES = 4
FP4X2_BYTES = 1
UINT8_BYTES = 1
UINT32_BYTES = 4
UINT64_BYTES = 8
INVALID_UINT32 = (1 << 32) - 1
TOKEN_SRC_METADATA_BYTES = 3 * UINT32_BYTES
DEFAULT_REGION_ALIGNMENT = 128
TMA_DESCRIPTOR_ALIGNMENT = 16

__all__ = [
    "MegaMoeFullFusionWorkspaceConfig",
    "INVALID_UINT32",
    "MegaMoeFullFusionRuntimeGate",
    "MegaMoeFullFusionWorkspaceDescriptor",
    "MegaMoeCombinePushRecord",
    "MegaMoeCombinePushReduceProof",
    "MegaMoeFullFusionWorkspaceLayout",
    "MegaMoeSingleRankDispatchPullProof",
    "MegaMoeSingleRankFullFusionProof",
    "MegaMoeSingleRankRoute",
    "MegaMoeTokenSourceMetadata",
    "MegaMoeWorkspaceRegion",
    "MegaMoeWorkspaceRegionDescriptor",
    "build_megamoe_combine_push_reduce_proof",
    "build_megamoe_full_fusion_runtime_gate",
    "build_megamoe_full_fusion_workspace_descriptor",
    "build_megamoe_full_fusion_workspace_layout",
    "build_megamoe_single_rank_dispatch_pull_proof",
    "build_megamoe_single_rank_full_fusion_proof",
]


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class MegaMoeFullFusionWorkspaceConfig:
    """Shape parameters for a full-fusion MegaMoE symmetric workspace.

    Args:
        ep_size: Number of expert-parallel ranks.
        max_num_tokens_per_rank: Padded token capacity for one rank.
        num_experts: Global expert count.
        top_k: Number of selected experts per token.
        hidden_size: Model hidden dimension in elements.
        intermediate_size: Per-expert post-SwiGLU intermediate dimension.
        tile_size: M tile size, also used as the pool-block token count.
        scaling_vector_size: NVFP4 scale-vector width in elements.
    """

    ep_size: int
    max_num_tokens_per_rank: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    tile_size: int = 128
    scaling_vector_size: int = 16

    def __post_init__(self) -> None:
        for name in (
            "ep_size",
            "max_num_tokens_per_rank",
            "num_experts",
            "top_k",
            "hidden_size",
            "intermediate_size",
            "tile_size",
            "scaling_vector_size",
        ):
            _require_positive(name, getattr(self, name))

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        if self.max_num_tokens_per_rank % self.tile_size != 0:
            raise ValueError(
                "max_num_tokens_per_rank must be padded to tile_size: "
                f"{self.max_num_tokens_per_rank} vs {self.tile_size}"
            )
        if self.hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even for FP4x2 packing, got {self.hidden_size}")
        if self.intermediate_size % 2 != 0:
            raise ValueError(
                f"intermediate_size must be even for FP4x2 packing, got {self.intermediate_size}"
            )
        if self.hidden_size % self.scaling_vector_size != 0:
            raise ValueError(
                "hidden_size must be divisible by scaling_vector_size: "
                f"{self.hidden_size} vs {self.scaling_vector_size}"
            )
        if self.intermediate_size % self.scaling_vector_size != 0:
            raise ValueError(
                "intermediate_size must be divisible by scaling_vector_size: "
                f"{self.intermediate_size} vs {self.scaling_vector_size}"
            )

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def max_recv_tokens_per_expert(self) -> int:
        return self.ep_size * self.max_num_tokens_per_rank

    @property
    def max_experts_per_token_on_rank(self) -> int:
        return min(self.top_k, self.num_experts_per_rank)

    @property
    def num_max_pool_tokens(self) -> int:
        unpadded_tokens = (
            self.max_recv_tokens_per_expert * self.max_experts_per_token_on_rank
            + self.num_experts_per_rank * (self.tile_size - 1)
        )
        return _align_up(unpadded_tokens, self.tile_size)

    @property
    def num_max_pool_blocks(self) -> int:
        return self.num_max_pool_tokens // self.tile_size

    @property
    def num_padded_sf_pool_tokens(self) -> int:
        sf_block_tokens = _align_up(self.tile_size, 128)
        return self.num_max_pool_blocks * sf_block_tokens

    @property
    def l1_arrival_count_entries(self) -> int:
        return _align_up(self.num_max_pool_blocks, 2)


@dataclass(frozen=True)
class MegaMoeWorkspaceRegion:
    """Byte range for one full-fusion workspace region."""

    name: str
    offset: int
    size_bytes: int
    alignment: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.size_bytes


@dataclass(frozen=True)
class MegaMoeWorkspaceRegionDescriptor:
    """Rank-aware byte range for one full-fusion workspace region."""

    name: str
    rank: int
    offset: int
    size_bytes: int
    alignment: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.size_bytes


@dataclass(frozen=True)
class MegaMoeTokenSourceMetadata:
    """Origin metadata stored next to one expert-major pool slot."""

    src_rank: int
    token_idx: int
    topk_idx: int


@dataclass(frozen=True)
class MegaMoeSingleRankRoute:
    """One active selected route in the single-rank full-fusion proof."""

    expert_idx: int
    pool_slot: int
    token_idx: int
    topk_idx: int
    src_token_topk_idx: int
    route_weight: float
    unweighted_output: tuple[float, ...]
    weighted_output: tuple[float, ...]


@dataclass(frozen=True)
class MegaMoeSingleRankFullFusionProof:
    """CPU proof artifact for the single-rank M5/M6 full-fusion contract.

    ``reduced_output`` models the local FP32 accumulation result before the
    final BF16 store. The helper intentionally does not allocate runtime
    symmetric memory or change MegaMoE dispatch; it only makes the metadata and
    route-weight/combine semantics executable in unit tests.
    """

    config: MegaMoeFullFusionWorkspaceConfig
    expert_send_count: tuple[int, ...]
    expert_recv_count: tuple[tuple[int, ...], ...]
    expert_recv_count_sum: tuple[int, ...]
    src_token_topk_idx: tuple[int, ...]
    token_src_metadata: tuple[MegaMoeTokenSourceMetadata, ...]
    l1_arrival_count: tuple[int, ...]
    l1_topk_weights_pool: tuple[float, ...]
    combine_token_buffer: tuple[tuple[tuple[float, ...], ...], ...]
    reduced_output: tuple[tuple[float, ...], ...]
    routes: tuple[MegaMoeSingleRankRoute, ...]
    num_active_pool_tokens: int
    num_materialized_pool_slots: int


@dataclass(frozen=True)
class MegaMoeSingleRankDispatchPullProof:
    """CPU proof artifact for the single-rank M5 dispatch-pull contract.

    ``metadata`` is the existing M5/M6 route proof. ``l1_acts_pool`` and
    ``l1_acts_sf_pool`` materialize the selected token payloads into the
    expert-major pool layout that FC1 will consume after dispatch fusion.
    """

    metadata: MegaMoeSingleRankFullFusionProof
    l1_acts_pool: tuple[tuple[int, ...], ...]
    l1_acts_sf_pool: tuple[tuple[int, ...], ...]
    pool_slot_to_sf_slot: tuple[int, ...]


@dataclass(frozen=True)
class MegaMoeCombinePushRecord:
    """One FC2 partial pushed into its source rank's combine buffer."""

    pool_slot: int
    dst_rank: int
    token_idx: int
    topk_idx: int
    weighted_output: tuple[float, ...]


@dataclass(frozen=True)
class MegaMoeCombinePushReduceProof:
    """CPU proof artifact for the M6 combine push and local reduce contract.

    ``combine_token_buffer`` is indexed as
    ``[dst_rank][topk_idx][token_idx][hidden_idx]``. A rank's
    ``barrier_ready_by_rank`` entry becomes true only when every expected
    ``[token, topk]`` slot for that rank has been peer-pushed.
    """

    config: MegaMoeFullFusionWorkspaceConfig
    num_tokens_per_rank: tuple[int, ...]
    combine_token_buffer: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
    combine_slot_ready: tuple[tuple[tuple[bool, ...], ...], ...]
    barrier_ready_by_rank: tuple[bool, ...]
    reduced_output: tuple[tuple[tuple[float, ...], ...], ...]
    routes: tuple[MegaMoeCombinePushRecord, ...]


@dataclass(frozen=True)
class MegaMoeFullFusionRuntimeGate:
    """Runtime decision for the guarded full-fusion path.

    The gate is deliberately separate from the CPU proof helpers: it records
    readiness for M5 dispatch-pull, M6 combine-push, and the final output path
    separately. Runtime code may bypass the compatibility path only when all
    three pieces are explicitly ready.
    """

    requested: bool
    use_full_fusion: bool
    fallback_reason: str | None
    workspace_layout: "MegaMoeFullFusionWorkspaceLayout | None"
    workspace_descriptor: "MegaMoeFullFusionWorkspaceDescriptor | None"
    m5_dispatch_pull_ready: bool
    m6_combine_push_ready: bool
    output_path_ready: bool


@dataclass(frozen=True)
class MegaMoeFullFusionWorkspaceLayout:
    """Computed byte layout for one rank's full-fusion MegaMoE workspace."""

    config: MegaMoeFullFusionWorkspaceConfig
    regions: tuple[MegaMoeWorkspaceRegion, ...]
    size_bytes: int

    def region(self, name: str) -> MegaMoeWorkspaceRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown MegaMoE workspace region: {name}")


@dataclass(frozen=True)
class MegaMoeFullFusionWorkspaceDescriptor:
    """Rank-aware descriptor for an allocated full-fusion workspace.

    ``rank_stride_bytes`` matches the stride between peer rank segments in the
    symmetric-memory allocation. Tests may use the compact layout size as the
    stride; MNNVL-backed runtime allocation can pass the actual CUDA VMM stride.
    """

    layout: MegaMoeFullFusionWorkspaceLayout
    rank: int
    ep_size: int
    rank_stride_bytes: int

    @property
    def local_rank_offset(self) -> int:
        return self.rank * self.rank_stride_bytes

    @property
    def size_bytes_per_rank(self) -> int:
        return self.layout.size_bytes

    def region(self, name: str, *, rank: int | None = None) -> MegaMoeWorkspaceRegionDescriptor:
        target_rank = self.rank if rank is None else rank
        if target_rank < 0 or target_rank >= self.ep_size:
            raise ValueError(f"rank must be in [0, {self.ep_size}), got {target_rank}")

        region = self.layout.region(name)
        return MegaMoeWorkspaceRegionDescriptor(
            name=region.name,
            rank=target_rank,
            offset=target_rank * self.rank_stride_bytes + region.offset,
            size_bytes=region.size_bytes,
            alignment=region.alignment,
        )

    def local_region(self, name: str) -> MegaMoeWorkspaceRegionDescriptor:
        return self.region(name)

    def peer_region(self, name: str, rank: int) -> MegaMoeWorkspaceRegionDescriptor:
        return self.region(name, rank=rank)


class _LayoutBuilder:
    def __init__(self) -> None:
        self._offset = 0
        self._regions: list[MegaMoeWorkspaceRegion] = []

    def add(self, name: str, size_bytes: int, alignment: int = DEFAULT_REGION_ALIGNMENT) -> None:
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative for {name}, got {size_bytes}")
        self._offset = _align_up(self._offset, alignment)
        self._regions.append(MegaMoeWorkspaceRegion(name, self._offset, size_bytes, alignment))
        self._offset += size_bytes

    def finish(self) -> tuple[tuple[MegaMoeWorkspaceRegion, ...], int]:
        size_bytes = _align_up(self._offset, DEFAULT_REGION_ALIGNMENT)
        return tuple(self._regions), size_bytes


def build_megamoe_full_fusion_workspace_layout(
    config: MegaMoeFullFusionWorkspaceConfig,
) -> MegaMoeFullFusionWorkspaceLayout:
    """Build the full-fusion workspace layout for one rank.

    The layout follows the M4 checkpoint contract and the DeepGEMM workspace
    formulas for count tables, pool capacity, and arrival-mask alignment. It is
    not a runtime allocator; M5/M6 code should consume this contract when wiring
    symmetric memory and kernel parameters.
    """
    builder = _LayoutBuilder()
    c = config

    builder.add("control_barrier", BARRIER_REGION_BYTES, TMA_DESCRIPTOR_ALIGNMENT)
    builder.add("expert_send_count", c.num_experts * UINT64_BYTES, UINT64_BYTES)
    builder.add(
        "expert_recv_count",
        c.ep_size * c.num_experts_per_rank * UINT64_BYTES,
        UINT64_BYTES,
    )
    builder.add("expert_recv_count_sum", c.num_experts_per_rank * UINT64_BYTES, UINT64_BYTES)
    builder.add("l1_arrival_count", c.l1_arrival_count_entries * UINT32_BYTES, UINT32_BYTES)
    builder.add("l2_arrival_mask", c.num_max_pool_blocks * UINT64_BYTES, UINT64_BYTES)
    builder.add(
        "src_token_topk_idx",
        c.num_experts_per_rank * c.ep_size * c.max_recv_tokens_per_expert * UINT32_BYTES,
    )
    builder.add("token_src_metadata", c.num_max_pool_tokens * TOKEN_SRC_METADATA_BYTES)
    builder.add("x_buf", c.max_num_tokens_per_rank * (c.hidden_size // 2) * FP4X2_BYTES)
    builder.add(
        "x_sf_buf",
        c.max_num_tokens_per_rank * (c.hidden_size // c.scaling_vector_size) * UINT8_BYTES,
    )
    builder.add("topk_idx_buf", c.max_num_tokens_per_rank * c.top_k * UINT64_BYTES)
    builder.add("topk_weights_buf", c.max_num_tokens_per_rank * c.top_k * FLOAT32_BYTES)
    builder.add("l1_acts_pool", c.num_max_pool_tokens * (c.hidden_size // 2) * FP4X2_BYTES)
    builder.add(
        "l1_acts_sf_pool",
        c.num_padded_sf_pool_tokens * (c.hidden_size // c.scaling_vector_size) * UINT8_BYTES,
    )
    builder.add("l1_topk_weights_pool", c.num_max_pool_tokens * FLOAT32_BYTES)
    builder.add(
        "l2_acts_pool",
        c.num_max_pool_tokens * (c.intermediate_size // 2) * FP4X2_BYTES,
    )
    builder.add(
        "l2_acts_sf_pool",
        c.num_padded_sf_pool_tokens * (c.intermediate_size // c.scaling_vector_size) * UINT8_BYTES,
    )
    builder.add(
        "ranked_route_output_buf",
        c.top_k * c.max_num_tokens_per_rank * c.hidden_size * BF16_BYTES,
    )
    builder.add(
        "combine_token_buffer",
        c.top_k * c.max_num_tokens_per_rank * c.hidden_size * BF16_BYTES,
    )

    regions, size_bytes = builder.finish()
    return MegaMoeFullFusionWorkspaceLayout(config=config, regions=regions, size_bytes=size_bytes)


def build_megamoe_full_fusion_workspace_descriptor(
    layout: MegaMoeFullFusionWorkspaceLayout,
    *,
    rank: int,
    ep_size: int,
    rank_stride_bytes: int | None = None,
) -> MegaMoeFullFusionWorkspaceDescriptor:
    """Build a rank-aware descriptor for a full-fusion workspace allocation.

    The descriptor does not allocate memory. It binds the M4 byte layout to the
    rank stride of the runtime symmetric allocation so M5/M6 wiring can address
    local and peer workspace regions with the same contract.
    """
    if ep_size != layout.config.ep_size:
        raise ValueError(f"ep_size must match layout config: {ep_size} vs {layout.config.ep_size}")
    if rank < 0 or rank >= ep_size:
        raise ValueError(f"rank must be in [0, {ep_size}), got {rank}")

    stride = layout.size_bytes if rank_stride_bytes is None else rank_stride_bytes
    if stride < layout.size_bytes:
        raise ValueError(
            f"rank_stride_bytes must cover one workspace layout: {stride} vs {layout.size_bytes}"
        )
    if stride % DEFAULT_REGION_ALIGNMENT != 0:
        raise ValueError(
            f"rank_stride_bytes must be {DEFAULT_REGION_ALIGNMENT}-byte aligned, got {stride}"
        )

    return MegaMoeFullFusionWorkspaceDescriptor(
        layout=layout,
        rank=rank,
        ep_size=ep_size,
        rank_stride_bytes=stride,
    )


def build_megamoe_full_fusion_runtime_gate(
    config: MegaMoeFullFusionWorkspaceConfig,
    *,
    requested: bool,
    workspace_runtime_ready: bool = False,
    workspace_rank: int = 0,
    workspace_rank_stride_bytes: int | None = None,
    workspace_fallback_reason: str | None = None,
    m5_dispatch_pull_ready: bool = False,
    m6_combine_push_ready: bool = False,
    output_path_ready: bool = False,
) -> MegaMoeFullFusionRuntimeGate:
    """Build the guarded runtime decision for the full-fusion path.

    ``requested=False`` preserves the current compute-fused compatibility path.
    When requested, the helper computes the M4 workspace layout but still keeps
    ``use_full_fusion=False`` until runtime workspace allocation, M5 dispatch-pull,
    M6 combine-push, and the final output path are all explicitly marked ready.
    ``workspace_fallback_reason`` lets runtime allocation report a concrete
    unsupported-platform or allocation failure reason while keeping the descriptor
    available for diagnostics.
    """
    if not requested:
        return MegaMoeFullFusionRuntimeGate(
            requested=False,
            use_full_fusion=False,
            fallback_reason="full-fusion runtime gate disabled",
            workspace_layout=None,
            workspace_descriptor=None,
            m5_dispatch_pull_ready=False,
            m6_combine_push_ready=False,
            output_path_ready=False,
        )

    layout = build_megamoe_full_fusion_workspace_layout(config)
    descriptor = build_megamoe_full_fusion_workspace_descriptor(
        layout,
        rank=workspace_rank,
        ep_size=config.ep_size,
        rank_stride_bytes=workspace_rank_stride_bytes,
    )
    if not workspace_runtime_ready:
        fallback_reason = (
            workspace_fallback_reason or "full-fusion workspace runtime allocation is not wired"
        )
    elif not m5_dispatch_pull_ready:
        fallback_reason = "M5 dispatch-pull runtime wiring is not ready"
    elif not m6_combine_push_ready:
        fallback_reason = "M6 combine-push runtime wiring is not ready"
    elif not output_path_ready:
        fallback_reason = "full-fusion output path wiring is not ready"
    else:
        fallback_reason = None

    return MegaMoeFullFusionRuntimeGate(
        requested=True,
        use_full_fusion=fallback_reason is None,
        fallback_reason=fallback_reason,
        workspace_layout=layout,
        workspace_descriptor=descriptor,
        m5_dispatch_pull_ready=m5_dispatch_pull_ready,
        m6_combine_push_ready=m6_combine_push_ready,
        output_path_ready=output_path_ready,
    )


def _invalid_metadata() -> MegaMoeTokenSourceMetadata:
    return MegaMoeTokenSourceMetadata(
        src_rank=INVALID_UINT32,
        token_idx=INVALID_UINT32,
        topk_idx=INVALID_UINT32,
    )


def _normalize_expert_table(
    name: str, values: Sequence[Sequence[int]], expected_cols: int
) -> tuple[tuple[int, ...], ...]:
    table = tuple(tuple(row) for row in values)
    if not table:
        raise ValueError(f"{name} must contain at least one token")
    for row_idx, row in enumerate(table):
        if len(row) != expected_cols:
            raise ValueError(
                f"{name}[{row_idx}] must contain {expected_cols} entries, got {len(row)}"
            )
    return table


def _normalize_float_table(
    name: str, values: Sequence[Sequence[float]], expected_cols: int
) -> tuple[tuple[float, ...], ...]:
    table = tuple(tuple(float(value) for value in row) for row in values)
    if not table:
        raise ValueError(f"{name} must contain at least one token")
    for row_idx, row in enumerate(table):
        if len(row) != expected_cols:
            raise ValueError(
                f"{name}[{row_idx}] must contain {expected_cols} entries, got {len(row)}"
            )
    return table


def _normalize_byte_table(
    name: str,
    values: Sequence[Sequence[int]],
    expected_rows: int,
    expected_cols: int,
) -> tuple[tuple[int, ...], ...]:
    table = tuple(tuple(int(value) for value in row) for row in values)
    if len(table) != expected_rows:
        raise ValueError(
            f"{name} row count must match token_selected_experts: {len(table)} vs {expected_rows}"
        )
    for row_idx, row in enumerate(table):
        if len(row) != expected_cols:
            raise ValueError(
                f"{name}[{row_idx}] must contain {expected_cols} entries, got {len(row)}"
            )
        for col_idx, value in enumerate(row):
            if value < 0 or value > 0xFF:
                raise ValueError(
                    f"{name}[{row_idx}][{col_idx}] must fit in uint8 payload storage, got {value}"
                )
    return table


def _zero_route_outputs(
    num_tokens: int, top_k: int, hidden_size: int
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    zero_route = tuple(0.0 for _ in range(hidden_size))
    return tuple(tuple(zero_route for _ in range(top_k)) for _ in range(num_tokens))


def _normalize_route_outputs(
    values: Sequence[Sequence[Sequence[float]]],
    expected_tokens: int,
    expected_top_k: int,
    expected_hidden_size: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    outputs = tuple(
        tuple(tuple(float(value) for value in route) for route in token_routes)
        for token_routes in values
    )
    if len(outputs) != expected_tokens:
        raise ValueError(
            "unweighted_route_outputs token count must match token_selected_experts: "
            f"{len(outputs)} vs {expected_tokens}"
        )
    for token_idx, token_routes in enumerate(outputs):
        if len(token_routes) != expected_top_k:
            raise ValueError(
                f"unweighted_route_outputs[{token_idx}] must contain {expected_top_k} routes, "
                f"got {len(token_routes)}"
            )
        for topk_idx, route in enumerate(token_routes):
            if len(route) != expected_hidden_size:
                raise ValueError(
                    f"unweighted_route_outputs[{token_idx}][{topk_idx}] hidden_size must be "
                    f"{expected_hidden_size}, got {len(route)}"
                )
    return outputs


def _normalize_token_counts(
    values: Sequence[int], expected_ranks: int, max_tokens_per_rank: int
) -> tuple[int, ...]:
    counts = tuple(int(value) for value in values)
    if len(counts) != expected_ranks:
        raise ValueError(
            f"num_tokens_per_rank must contain {expected_ranks} entries, got {len(counts)}"
        )
    for rank, count in enumerate(counts):
        if count < 0 or count > max_tokens_per_rank:
            raise ValueError(
                f"num_tokens_per_rank[{rank}] must be in [0, {max_tokens_per_rank}], got {count}"
            )
    return counts


def _normalize_weighted_route_outputs(
    values: Sequence[Sequence[float]], expected_rows: int, expected_hidden_size: int
) -> tuple[tuple[float, ...], ...]:
    outputs = tuple(tuple(float(value) for value in row) for row in values)
    if len(outputs) != expected_rows:
        raise ValueError(
            "weighted_route_outputs row count must match token_src_metadata: "
            f"{len(outputs)} vs {expected_rows}"
        )
    for row_idx, row in enumerate(outputs):
        if len(row) != expected_hidden_size:
            raise ValueError(
                f"weighted_route_outputs[{row_idx}] hidden_size must be "
                f"{expected_hidden_size}, got {len(row)}"
            )
    return outputs


def _is_invalid_metadata(metadata: MegaMoeTokenSourceMetadata) -> bool:
    return (
        metadata.src_rank == INVALID_UINT32
        and metadata.token_idx == INVALID_UINT32
        and metadata.topk_idx == INVALID_UINT32
    )


def _has_partial_invalid_metadata(metadata: MegaMoeTokenSourceMetadata) -> bool:
    invalid_fields = (
        metadata.src_rank == INVALID_UINT32,
        metadata.token_idx == INVALID_UINT32,
        metadata.topk_idx == INVALID_UINT32,
    )
    return any(invalid_fields) and not all(invalid_fields)


def build_megamoe_combine_push_reduce_proof(
    config: MegaMoeFullFusionWorkspaceConfig,
    num_tokens_per_rank: Sequence[int],
    token_src_metadata: Sequence[MegaMoeTokenSourceMetadata],
    weighted_route_outputs: Sequence[Sequence[float]],
) -> MegaMoeCombinePushReduceProof:
    """Build a CPU-only proof for M6 combine peer-push and local reduce.

    The helper models FC2 epilogue peer-pushes into the source rank's
    ``combine_token_buffer``. ``weighted_route_outputs`` must already include
    the full-fusion route weight from the FC1-before-requant contract, so M6
    materialization copies the partials without multiplying by route weights
    again.

    Args:
        config: Full-fusion workspace config.
        num_tokens_per_rank: Local token count for each source rank.
        token_src_metadata: Pool-slot source metadata. Invalid sentinel entries
            are treated as padded/inactive pool slots.
        weighted_route_outputs: FC2 route outputs shaped
            ``[num_pool_slots, hidden_size]`` after route-weight application.

    Returns:
        A proof artifact containing peer-pushed combine buffers, per-slot ready
        bits, per-rank barrier readiness, and local top-k reductions.
    """
    if len(token_src_metadata) > config.num_max_pool_tokens:
        raise ValueError(
            "token_src_metadata length exceeds workspace pool capacity: "
            f"{len(token_src_metadata)} vs {config.num_max_pool_tokens}"
        )

    token_counts = _normalize_token_counts(
        num_tokens_per_rank, config.ep_size, config.max_num_tokens_per_rank
    )
    route_outputs = _normalize_weighted_route_outputs(
        weighted_route_outputs, len(token_src_metadata), config.hidden_size
    )

    zero_output = tuple(0.0 for _ in range(config.hidden_size))
    combine_token_buffer = [
        [
            [list(zero_output) for _ in range(config.max_num_tokens_per_rank)]
            for _ in range(config.top_k)
        ]
        for _ in range(config.ep_size)
    ]
    combine_slot_ready = [
        [[False for _ in range(config.max_num_tokens_per_rank)] for _ in range(config.top_k)]
        for _ in range(config.ep_size)
    ]

    routes: list[MegaMoeCombinePushRecord] = []
    for pool_slot, metadata in enumerate(token_src_metadata):
        route_output = route_outputs[pool_slot]
        if _has_partial_invalid_metadata(metadata):
            raise ValueError(
                f"token_src_metadata[{pool_slot}] must be fully valid or the invalid sentinel"
            )
        if _is_invalid_metadata(metadata):
            if any(value != 0.0 for value in route_output):
                raise ValueError(
                    f"weighted_route_outputs[{pool_slot}] must be zero for an inactive pool slot"
                )
            continue

        if metadata.src_rank < 0 or metadata.src_rank >= config.ep_size:
            raise ValueError(
                f"token_src_metadata[{pool_slot}].src_rank must be in [0, {config.ep_size}), "
                f"got {metadata.src_rank}"
            )
        if metadata.token_idx < 0 or metadata.token_idx >= token_counts[metadata.src_rank]:
            raise ValueError(
                f"token_src_metadata[{pool_slot}].token_idx must be within rank "
                f"{metadata.src_rank} token count {token_counts[metadata.src_rank]}, "
                f"got {metadata.token_idx}"
            )
        if metadata.topk_idx < 0 or metadata.topk_idx >= config.top_k:
            raise ValueError(
                f"token_src_metadata[{pool_slot}].topk_idx must be in [0, {config.top_k}), "
                f"got {metadata.topk_idx}"
            )
        if combine_slot_ready[metadata.src_rank][metadata.topk_idx][metadata.token_idx]:
            raise ValueError(
                "duplicate combine push for "
                f"rank {metadata.src_rank}, token {metadata.token_idx}, topk {metadata.topk_idx}"
            )

        combine_token_buffer[metadata.src_rank][metadata.topk_idx][metadata.token_idx] = list(
            route_output
        )
        combine_slot_ready[metadata.src_rank][metadata.topk_idx][metadata.token_idx] = True
        routes.append(
            MegaMoeCombinePushRecord(
                pool_slot=pool_slot,
                dst_rank=metadata.src_rank,
                token_idx=metadata.token_idx,
                topk_idx=metadata.topk_idx,
                weighted_output=route_output,
            )
        )

    barrier_ready_by_rank = []
    reduced_output = []
    for rank, token_count in enumerate(token_counts):
        rank_ready = True
        rank_output = []
        for token_idx in range(token_count):
            for topk_idx in range(config.top_k):
                rank_ready = rank_ready and combine_slot_ready[rank][topk_idx][token_idx]
            reduced_token = []
            for hidden_idx in range(config.hidden_size):
                reduced_token.append(
                    sum(
                        combine_token_buffer[rank][topk_idx][token_idx][hidden_idx]
                        for topk_idx in range(config.top_k)
                    )
                )
            rank_output.append(tuple(reduced_token))
        barrier_ready_by_rank.append(rank_ready)
        reduced_output.append(tuple(rank_output))

    frozen_buffer = tuple(
        tuple(tuple(tuple(hidden_values) for hidden_values in topk_rows) for topk_rows in rank_rows)
        for rank_rows in combine_token_buffer
    )
    frozen_ready = tuple(
        tuple(tuple(token_ready for token_ready in topk_rows) for topk_rows in rank_rows)
        for rank_rows in combine_slot_ready
    )

    return MegaMoeCombinePushReduceProof(
        config=config,
        num_tokens_per_rank=token_counts,
        combine_token_buffer=frozen_buffer,
        combine_slot_ready=frozen_ready,
        barrier_ready_by_rank=tuple(barrier_ready_by_rank),
        reduced_output=tuple(reduced_output),
        routes=tuple(routes),
    )


def build_megamoe_single_rank_full_fusion_proof(
    config: MegaMoeFullFusionWorkspaceConfig,
    token_selected_experts: Sequence[Sequence[int]],
    token_final_scales: Sequence[Sequence[float]],
    unweighted_route_outputs: Sequence[Sequence[Sequence[float]]],
) -> MegaMoeSingleRankFullFusionProof:
    """Build a CPU-only single-rank proof for M5/M6 full-fusion semantics.

    The proof models the degenerate ``ep_size == 1`` path: dispatch metadata is
    still written into the same expert-major tables that a multi-rank full
    fusion kernel would consume, route weights are placed in the FC1-side pool,
    and M6 combine writes weighted FC2 partials into
    ``combine_token_buffer[topk, token, hidden]`` before a local reduction.

    Args:
        config: Full-fusion workspace config. Must use ``ep_size == 1``.
        token_selected_experts: Expert ids shaped ``[num_tokens, top_k]``.
        token_final_scales: Route weights shaped ``[num_tokens, top_k]``.
        unweighted_route_outputs: Linearized FC2 route outputs shaped
            ``[num_tokens, top_k, hidden_size]`` before the full-fusion route
            weight is applied.

    Returns:
        A proof artifact containing populated metadata tables, route-weight pool
        entries, combine-token buffer contents, and reduced output.
    """
    if config.ep_size != 1:
        raise ValueError(f"single-rank proof requires ep_size == 1, got {config.ep_size}")

    selected_experts = _normalize_expert_table(
        "token_selected_experts", token_selected_experts, config.top_k
    )
    route_weights = _normalize_float_table("token_final_scales", token_final_scales, config.top_k)

    num_tokens = len(selected_experts)
    if len(route_weights) != num_tokens:
        raise ValueError(
            "token_final_scales token count must match token_selected_experts: "
            f"{len(route_weights)} vs {num_tokens}"
        )
    if num_tokens > config.max_num_tokens_per_rank:
        raise ValueError(
            "token count exceeds max_num_tokens_per_rank: "
            f"{num_tokens} vs {config.max_num_tokens_per_rank}"
        )

    route_outputs = _normalize_route_outputs(
        unweighted_route_outputs, num_tokens, config.top_k, config.hidden_size
    )

    expert_routes: list[list[tuple[int, int, int]]] = [
        [] for _ in range(config.num_experts_per_rank)
    ]
    for token_idx, expert_row in enumerate(selected_experts):
        if len(set(expert_row)) != len(expert_row):
            raise ValueError(
                f"token_selected_experts[{token_idx}] must not contain duplicate experts"
            )
        for topk_idx, expert_idx in enumerate(expert_row):
            if expert_idx < 0 or expert_idx >= config.num_experts_per_rank:
                raise ValueError(
                    "single-rank proof only supports local experts in "
                    f"[0, {config.num_experts_per_rank}), got {expert_idx}"
                )
            src_token_topk_idx = token_idx * config.top_k + topk_idx
            expert_routes[expert_idx].append((token_idx, topk_idx, src_token_topk_idx))

    expert_send_count = tuple(len(routes) for routes in expert_routes)
    expert_recv_count = (expert_send_count,)
    expert_recv_count_sum = expert_send_count

    src_token_topk_idx = [
        INVALID_UINT32
        for _ in range(
            config.num_experts_per_rank * config.ep_size * config.max_recv_tokens_per_expert
        )
    ]
    token_src_metadata = [_invalid_metadata() for _ in range(config.num_max_pool_tokens)]
    l1_topk_weights_pool = [0.0 for _ in range(config.num_max_pool_tokens)]
    l1_arrival_count = [0 for _ in range(config.l1_arrival_count_entries)]
    combine_token_buffer = [
        [[0.0 for _ in range(config.hidden_size)] for _ in range(config.max_num_tokens_per_rank)]
        for _ in range(config.top_k)
    ]

    routes: list[MegaMoeSingleRankRoute] = []
    pool_slot = 0
    active_pool_slots: set[int] = set()
    for expert_idx, expert_route_list in enumerate(expert_routes):
        if len(expert_route_list) > config.max_recv_tokens_per_expert:
            raise ValueError(
                f"expert {expert_idx} has {len(expert_route_list)} routes, exceeding "
                f"max_recv_tokens_per_expert {config.max_recv_tokens_per_expert}"
            )

        expert_pool_start = pool_slot
        table_base = expert_idx * config.ep_size * config.max_recv_tokens_per_expert
        for route_ordinal, (token_idx, topk_idx, src_idx) in enumerate(expert_route_list):
            current_pool_slot = expert_pool_start + route_ordinal
            if current_pool_slot >= config.num_max_pool_tokens:
                raise ValueError(
                    "single-rank metadata proof exceeded workspace pool capacity: "
                    f"slot {current_pool_slot} vs capacity {config.num_max_pool_tokens}"
                )

            table_idx = table_base + route_ordinal
            src_token_topk_idx[table_idx] = src_idx
            token_src_metadata[current_pool_slot] = MegaMoeTokenSourceMetadata(
                src_rank=0, token_idx=token_idx, topk_idx=topk_idx
            )

            route_weight = route_weights[token_idx][topk_idx]
            unweighted_output = route_outputs[token_idx][topk_idx]
            weighted_output = tuple(value * route_weight for value in unweighted_output)

            l1_topk_weights_pool[current_pool_slot] = route_weight
            combine_token_buffer[topk_idx][token_idx] = list(weighted_output)
            active_pool_slots.add(current_pool_slot)
            routes.append(
                MegaMoeSingleRankRoute(
                    expert_idx=expert_idx,
                    pool_slot=current_pool_slot,
                    token_idx=token_idx,
                    topk_idx=topk_idx,
                    src_token_topk_idx=src_idx,
                    route_weight=route_weight,
                    unweighted_output=unweighted_output,
                    weighted_output=weighted_output,
                )
            )

        pool_slot = _align_up(expert_pool_start + len(expert_route_list), config.tile_size)

    if pool_slot > config.num_max_pool_tokens:
        raise ValueError(
            "single-rank metadata proof exceeded workspace pool capacity after padding: "
            f"{pool_slot} vs {config.num_max_pool_tokens}"
        )

    for active_pool_slot in active_pool_slots:
        block_idx = active_pool_slot // config.tile_size
        l1_arrival_count[block_idx] = 1

    reduced_output = []
    for token_idx in range(num_tokens):
        reduced_token = []
        for hidden_idx in range(config.hidden_size):
            reduced_token.append(
                sum(
                    combine_token_buffer[topk_idx][token_idx][hidden_idx]
                    for topk_idx in range(config.top_k)
                )
            )
        reduced_output.append(tuple(reduced_token))

    frozen_combine_buffer = tuple(
        tuple(tuple(hidden_values) for hidden_values in topk_rows)
        for topk_rows in combine_token_buffer
    )

    return MegaMoeSingleRankFullFusionProof(
        config=config,
        expert_send_count=expert_send_count,
        expert_recv_count=expert_recv_count,
        expert_recv_count_sum=expert_recv_count_sum,
        src_token_topk_idx=tuple(src_token_topk_idx),
        token_src_metadata=tuple(token_src_metadata),
        l1_arrival_count=tuple(l1_arrival_count),
        l1_topk_weights_pool=tuple(l1_topk_weights_pool),
        combine_token_buffer=frozen_combine_buffer,
        reduced_output=tuple(reduced_output),
        routes=tuple(routes),
        num_active_pool_tokens=len(routes),
        num_materialized_pool_slots=pool_slot,
    )


def build_megamoe_single_rank_dispatch_pull_proof(
    config: MegaMoeFullFusionWorkspaceConfig,
    x_buf: Sequence[Sequence[int]],
    x_sf_buf: Sequence[Sequence[int]],
    token_selected_experts: Sequence[Sequence[int]],
    token_final_scales: Sequence[Sequence[float]],
    unweighted_route_outputs: Sequence[Sequence[Sequence[float]]] | None = None,
) -> MegaMoeSingleRankDispatchPullProof:
    """Build a CPU-only single-rank proof for M5 dispatch payload pull.

    Args:
        config: Full-fusion workspace config. Must use ``ep_size == 1``.
        x_buf: Packed FP4x2 activation bytes shaped
            ``[num_tokens, hidden_size / 2]``.
        x_sf_buf: Activation scale-factor bytes shaped
            ``[num_tokens, hidden_size / scaling_vector_size]``.
        token_selected_experts: Expert ids shaped ``[num_tokens, top_k]``.
        token_final_scales: Route weights shaped ``[num_tokens, top_k]``.
        unweighted_route_outputs: Optional FC2 route outputs forwarded to the
            existing M6 combine proof. When omitted, zero outputs are used so
            the helper focuses only on dispatch materialization.

    Returns:
        A proof artifact with expert-major activation/SF pools materialized
        according to the same route metadata and arrival-count contract used by
        ``build_megamoe_single_rank_full_fusion_proof``.
    """
    selected_experts = _normalize_expert_table(
        "token_selected_experts", token_selected_experts, config.top_k
    )
    num_tokens = len(selected_experts)
    route_weights = _normalize_float_table("token_final_scales", token_final_scales, config.top_k)
    if len(route_weights) != num_tokens:
        raise ValueError(
            "token_final_scales token count must match token_selected_experts: "
            f"{len(route_weights)} vs {num_tokens}"
        )

    x_rows = _normalize_byte_table("x_buf", x_buf, num_tokens, config.hidden_size // 2)
    x_sf_rows = _normalize_byte_table(
        "x_sf_buf",
        x_sf_buf,
        num_tokens,
        config.hidden_size // config.scaling_vector_size,
    )
    if unweighted_route_outputs is None:
        route_outputs = _zero_route_outputs(num_tokens, config.top_k, config.hidden_size)
    else:
        route_outputs = unweighted_route_outputs

    metadata = build_megamoe_single_rank_full_fusion_proof(
        config, selected_experts, route_weights, route_outputs
    )

    zero_x_row = tuple(0 for _ in range(config.hidden_size // 2))
    zero_sf_row = tuple(0 for _ in range(config.hidden_size // config.scaling_vector_size))
    l1_acts_pool = [zero_x_row for _ in range(config.num_max_pool_tokens)]
    l1_acts_sf_pool = [zero_sf_row for _ in range(config.num_padded_sf_pool_tokens)]
    pool_slot_to_sf_slot = [INVALID_UINT32 for _ in range(config.num_max_pool_tokens)]

    sf_block_tokens = _align_up(config.tile_size, 128)
    for route in metadata.routes:
        sf_slot = (route.pool_slot // config.tile_size) * sf_block_tokens + (
            route.pool_slot % config.tile_size
        )
        l1_acts_pool[route.pool_slot] = x_rows[route.token_idx]
        l1_acts_sf_pool[sf_slot] = x_sf_rows[route.token_idx]
        pool_slot_to_sf_slot[route.pool_slot] = sf_slot

    return MegaMoeSingleRankDispatchPullProof(
        metadata=metadata,
        l1_acts_pool=tuple(l1_acts_pool),
        l1_acts_sf_pool=tuple(l1_acts_sf_pool),
        pool_slot_to_sf_slot=tuple(pool_slot_to_sf_slot),
    )
