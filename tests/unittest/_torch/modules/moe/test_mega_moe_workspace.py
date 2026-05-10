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
"""CPU layout tests for MegaMoE full-fusion workspace planning."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_mega import MegaMoE
from tensorrt_llm._torch.modules.fused_moe.mega_moe_workspace import (
    BF16_BYTES,
    FLOAT32_BYTES,
    INVALID_UINT32,
    TOKEN_SRC_METADATA_BYTES,
    UINT8_BYTES,
    UINT32_BYTES,
    UINT64_BYTES,
    MegaMoeFullFusionWorkspaceConfig,
    MegaMoeTokenSourceMetadata,
    build_megamoe_combine_push_reduce_proof,
    build_megamoe_full_fusion_runtime_gate,
    build_megamoe_full_fusion_workspace_descriptor,
    build_megamoe_full_fusion_workspace_layout,
    build_megamoe_single_rank_dispatch_pull_proof,
    build_megamoe_single_rank_full_fusion_proof,
)

_PayloadRows = tuple[tuple[int, ...], ...]


_REGION_ORDER = (
    "control_barrier",
    "expert_send_count",
    "expert_recv_count",
    "expert_recv_count_sum",
    "l1_arrival_count",
    "l2_arrival_mask",
    "src_token_topk_idx",
    "token_src_metadata",
    "x_buf",
    "x_sf_buf",
    "topk_idx_buf",
    "topk_weights_buf",
    "l1_acts_pool",
    "l1_acts_sf_pool",
    "l1_topk_weights_pool",
    "l2_acts_pool",
    "l2_acts_sf_pool",
    "ranked_route_output_buf",
    "combine_token_buffer",
)


def _small_single_rank_config() -> MegaMoeFullFusionWorkspaceConfig:
    return MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=3,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )


def _route_outputs(
    num_tokens: int, top_k: int, hidden_size: int
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                float(token_idx * 100 + topk_idx * 10 + hidden_idx)
                for hidden_idx in range(hidden_size)
            )
            for topk_idx in range(top_k)
        )
        for token_idx in range(num_tokens)
    )


def _payload_rows(num_tokens: int, row_width: int) -> _PayloadRows:
    return tuple(
        tuple(token_idx * 17 + value_idx for value_idx in range(row_width))
        for token_idx in range(num_tokens)
    )


def _scale_payload_rows(num_tokens: int, row_width: int) -> _PayloadRows:
    return tuple(
        tuple(200 + token_idx * 7 + value_idx for value_idx in range(row_width))
        for token_idx in range(num_tokens)
    )


def _weighted_output(base: float, hidden_size: int) -> tuple[float, ...]:
    return tuple(base + float(hidden_idx) for hidden_idx in range(hidden_size))


def test_full_fusion_workspace_layout_order_and_alignment() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=8,
        max_num_tokens_per_rank=512,
        num_experts=64,
        top_k=8,
        hidden_size=512,
        intermediate_size=1024,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)

    assert tuple(region.name for region in layout.regions) == _REGION_ORDER
    assert layout.size_bytes % 128 == 0
    assert layout.region("control_barrier").size_bytes >= (
        MegaMoE._FULL_FUSION_M6_CONTROL_WORDS * UINT64_BYTES
    )

    previous_end = 0
    for region in layout.regions:
        assert region.offset >= previous_end
        assert region.offset % region.alignment == 0
        previous_end = region.end_offset

    assert layout.region("l2_arrival_mask").offset % UINT64_BYTES == 0


def test_pool_capacity_and_region_sizes_match_m4_formulas() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=4,
        max_num_tokens_per_rank=256,
        num_experts=16,
        top_k=2,
        hidden_size=512,
        intermediate_size=1024,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)

    assert config.num_experts_per_rank == 4
    assert config.max_recv_tokens_per_expert == 1024
    assert config.num_max_pool_tokens == 2560
    assert config.num_max_pool_blocks == 20
    assert config.l1_arrival_count_entries == 20
    assert config.num_padded_sf_pool_tokens == 2560

    assert layout.region("expert_send_count").size_bytes == 16 * UINT64_BYTES
    assert layout.region("expert_recv_count").size_bytes == 4 * 4 * UINT64_BYTES
    assert layout.region("expert_recv_count_sum").size_bytes == 4 * UINT64_BYTES
    assert layout.region("l1_arrival_count").size_bytes == 20 * UINT32_BYTES
    assert layout.region("l2_arrival_mask").size_bytes == 20 * UINT64_BYTES
    assert layout.region("src_token_topk_idx").size_bytes == 4 * 4 * 1024 * UINT32_BYTES
    assert layout.region("token_src_metadata").size_bytes == 2560 * TOKEN_SRC_METADATA_BYTES
    assert layout.region("x_buf").size_bytes == 256 * (512 // 2)
    assert layout.region("x_sf_buf").size_bytes == 256 * (512 // 16) * UINT8_BYTES
    assert layout.region("topk_idx_buf").size_bytes == 256 * 2 * UINT64_BYTES
    assert layout.region("topk_weights_buf").size_bytes == 256 * 2 * FLOAT32_BYTES
    assert layout.region("l1_acts_pool").size_bytes == 2560 * (512 // 2)
    assert layout.region("l1_acts_sf_pool").size_bytes == 2560 * (512 // 16) * UINT8_BYTES
    assert layout.region("l1_topk_weights_pool").size_bytes == 2560 * FLOAT32_BYTES
    assert layout.region("l2_acts_pool").size_bytes == 2560 * (1024 // 2)
    assert layout.region("l2_acts_sf_pool").size_bytes == 2560 * (1024 // 16) * UINT8_BYTES
    assert layout.region("ranked_route_output_buf").size_bytes == 2 * 256 * 512 * BF16_BYTES
    assert layout.region("combine_token_buffer").size_bytes == 2 * 256 * 512 * BF16_BYTES


def test_arrival_count_padding_keeps_l2_mask_aligned() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=128,
        num_experts=2,
        top_k=1,
        hidden_size=512,
        intermediate_size=512,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)

    assert config.num_max_pool_blocks == 3
    assert config.l1_arrival_count_entries == 4
    assert layout.region("l2_arrival_mask").offset % UINT64_BYTES == 0


def test_region_lookup_rejects_unknown_names() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=128,
        num_experts=1,
        top_k=1,
        hidden_size=512,
        intermediate_size=512,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)

    with pytest.raises(KeyError, match="unknown MegaMoE workspace region"):
        layout.region("not_a_region")


def test_full_fusion_runtime_gate_defaults_to_fallback() -> None:
    config = _small_single_rank_config()

    disabled = build_megamoe_full_fusion_runtime_gate(config, requested=False)

    assert disabled.requested is False
    assert disabled.use_full_fusion is False
    assert disabled.workspace_layout is None
    assert disabled.workspace_descriptor is None
    assert disabled.output_path_ready is False
    assert disabled.fallback_reason == "full-fusion runtime gate disabled"

    requested = build_megamoe_full_fusion_runtime_gate(config, requested=True)

    assert requested.requested is True
    assert requested.use_full_fusion is False
    assert requested.workspace_layout is not None
    assert requested.workspace_descriptor is not None
    assert requested.workspace_layout.config == config
    assert requested.workspace_descriptor.layout == requested.workspace_layout
    assert requested.workspace_descriptor.rank == 0
    assert requested.output_path_ready is False
    assert requested.fallback_reason == "full-fusion workspace runtime allocation is not wired"


def test_full_fusion_runtime_gate_preserves_workspace_allocation_fallback() -> None:
    config = _small_single_rank_config()

    requested = build_megamoe_full_fusion_runtime_gate(
        config,
        requested=True,
        workspace_fallback_reason="full-fusion workspace allocation requires pure EP mapping",
    )

    assert requested.requested is True
    assert requested.use_full_fusion is False
    assert requested.workspace_descriptor is not None
    assert requested.fallback_reason == "full-fusion workspace allocation requires pure EP mapping"


def test_full_fusion_runtime_gate_uses_allocated_rank_stride() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)
    rank_stride_bytes = layout.size_bytes + 128

    requested = build_megamoe_full_fusion_runtime_gate(
        config,
        requested=True,
        workspace_runtime_ready=True,
        workspace_rank=1,
        workspace_rank_stride_bytes=rank_stride_bytes,
    )

    assert requested.use_full_fusion is False
    assert requested.fallback_reason == "M5 dispatch-pull runtime wiring is not ready"
    assert requested.workspace_descriptor is not None
    assert requested.workspace_descriptor.rank == 1
    assert requested.workspace_descriptor.rank_stride_bytes == rank_stride_bytes
    assert requested.workspace_descriptor.local_rank_offset == rank_stride_bytes


def test_full_fusion_workspace_descriptor_maps_rank_strided_regions() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)
    rank_stride_bytes = layout.size_bytes + 128

    descriptor = build_megamoe_full_fusion_workspace_descriptor(
        layout, rank=1, ep_size=2, rank_stride_bytes=rank_stride_bytes
    )

    local_l1_pool = descriptor.local_region("l1_acts_pool")
    peer_l1_pool = descriptor.peer_region("l1_acts_pool", 0)
    layout_l1_pool = layout.region("l1_acts_pool")

    assert descriptor.local_rank_offset == rank_stride_bytes
    assert descriptor.size_bytes_per_rank == layout.size_bytes
    assert local_l1_pool.rank == 1
    assert local_l1_pool.offset == rank_stride_bytes + layout_l1_pool.offset
    assert local_l1_pool.size_bytes == layout_l1_pool.size_bytes
    assert local_l1_pool.end_offset == local_l1_pool.offset + local_l1_pool.size_bytes
    assert peer_l1_pool.rank == 0
    assert peer_l1_pool.offset == layout_l1_pool.offset


@pytest.mark.parametrize(
    "rank, ep_size, rank_stride_delta, match",
    [
        (2, 2, 0, "rank must be"),
        (0, 3, 0, "ep_size must match"),
        (0, 2, -128, "rank_stride_bytes must cover"),
        (0, 2, 64, "rank_stride_bytes must be"),
    ],
)
def test_full_fusion_workspace_descriptor_validation(
    rank: int, ep_size: int, rank_stride_delta: int, match: str
) -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    layout = build_megamoe_full_fusion_workspace_layout(config)

    with pytest.raises(ValueError, match=match):
        build_megamoe_full_fusion_workspace_descriptor(
            layout,
            rank=rank,
            ep_size=ep_size,
            rank_stride_bytes=layout.size_bytes + rank_stride_delta,
        )


def _fake_megamoe_with_workspace(
    config: MegaMoeFullFusionWorkspaceConfig | None = None,
    *,
    rank: int = 1,
) -> tuple[SimpleNamespace, int]:
    if config is None:
        config = MegaMoeFullFusionWorkspaceConfig(
            ep_size=2,
            max_num_tokens_per_rank=4,
            num_experts=4,
            top_k=2,
            hidden_size=16,
            intermediate_size=16,
            tile_size=4,
        )
    layout = build_megamoe_full_fusion_workspace_layout(config)
    rank_stride_bytes = layout.size_bytes + 128
    descriptor = build_megamoe_full_fusion_workspace_descriptor(
        layout, rank=rank, ep_size=config.ep_size, rank_stride_bytes=rank_stride_bytes
    )
    fake = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_rank=rank),
        slot_start=rank * config.num_experts_per_rank,
        expert_size_per_partition=config.num_experts_per_rank,
        num_slots=config.num_experts,
        tile_size=config.tile_size,
        hidden_size=config.hidden_size,
        _full_fusion_runtime_gate=SimpleNamespace(workspace_descriptor=descriptor),
        _FULL_FUSION_DISPATCH_STAGING_ZERO_REGIONS=MegaMoE._FULL_FUSION_DISPATCH_STAGING_ZERO_REGIONS,
        _full_fusion_workspace=torch.full(
            (config.ep_size, rank_stride_bytes), 0xA5, dtype=torch.uint8
        ),
        _full_fusion_m5_producer_epoch=0,
        _full_fusion_combine_push_fallback_reason=None,
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout=None,
        _full_fusion_m6_combine_reduce_kernel=None,
        _full_fusion_fallback_diagnostics={},
        _full_fusion_force_output_path_fallback_reason=None,
        _full_fusion_profile_enabled=False,
        _full_fusion_profile_events={},
        _full_fusion_m5_direct_pool_fc_route_enabled=False,
        _full_fusion_m5_route_pull_verify_enabled=False,
        _full_fusion_m5_route_pull_materialize_enabled=False,
        _full_fusion_m5_route_metadata_verify_enabled=False,
        _full_fusion_m5_route_metadata_materialize_enabled=False,
        _full_fusion_m5_pool_metadata_verify_enabled=False,
        _full_fusion_m5_pool_metadata_materialize_enabled=False,
        _full_fusion_m5_helper_materialize_enabled=False,
        _full_fusion_m5_reconstruction_materialize_enabled=False,
        _full_fusion_m6_direct_pool_combine_push_enabled=False,
        _full_fusion_m6_direct_combine_layout_output_enabled=False,
        _full_fusion_m6_direct_combine_buffer_output_enabled=False,
        _full_fusion_m6_output_plan=None,
        _full_fusion_m6_route_output_layout="pool",
        _full_fusion_m6_route_output_layout_rows=None,
        _full_fusion_m6_route_output_active_rows=None,
        _full_fusion_m5_direct_pool_fc_route_layout=None,
        _full_fusion_m5_direct_pool_fc_route_active_pool_limit=None,
        _full_fusion_m5_active_pool_slots=None,
        _full_fusion_m5_direct_combine_rows=None,
        _full_fusion_m5_direct_combine_output_mapping=None,
        _full_fusion_m5_direct_combine_output_scales=None,
        _FULL_FUSION_M5_CONTROL_WORDS=MegaMoE._FULL_FUSION_M5_CONTROL_WORDS,
        _FULL_FUSION_M6_CONTROL_WORDS=MegaMoE._FULL_FUSION_M6_CONTROL_WORDS,
        _FULL_FUSION_M5_READY_MAGIC=MegaMoE._FULL_FUSION_M5_READY_MAGIC,
        _FULL_FUSION_M5_READY_FLAG=MegaMoE._FULL_FUSION_M5_READY_FLAG,
        _FULL_FUSION_ROUTE_OUTPUT_READY_FLAG=MegaMoE._FULL_FUSION_ROUTE_OUTPUT_READY_FLAG,
        _FULL_FUSION_M6_READY_FLAG=MegaMoE._FULL_FUSION_M6_READY_FLAG,
        _FULL_FUSION_M5_SYNC_TIMEOUT_S=0.01,
        _FULL_FUSION_M5_SYNC_POLL_INTERVAL_S=0.0001,
    )
    fake._full_fusion_fallback_diagnostics_snapshot = MethodType(
        MegaMoE._full_fusion_fallback_diagnostics_snapshot, fake
    )
    fake._record_full_fusion_fallback = MethodType(MegaMoE._record_full_fusion_fallback, fake)
    fake._set_full_fusion_output_path_fallback = MethodType(
        MegaMoE._set_full_fusion_output_path_fallback, fake
    )
    fake._full_fusion_profile_device = MethodType(MegaMoE._full_fusion_profile_device, fake)
    fake._synchronize_full_fusion_profile_device = MethodType(
        MegaMoE._synchronize_full_fusion_profile_device, fake
    )
    fake._start_full_fusion_profile_event = MethodType(
        MegaMoE._start_full_fusion_profile_event, fake
    )
    fake._finish_full_fusion_profile_event = MethodType(
        MegaMoE._finish_full_fusion_profile_event, fake
    )
    fake.reset_full_fusion_profile_events = MethodType(
        MegaMoE.reset_full_fusion_profile_events, fake
    )
    fake._full_fusion_workspace_region = MethodType(MegaMoE._full_fusion_workspace_region, fake)
    fake._full_fusion_workspace_region_as = MethodType(
        MegaMoE._full_fusion_workspace_region_as, fake
    )
    fake._full_fusion_workspace_region_all_ranks_as = MethodType(
        MegaMoE._full_fusion_workspace_region_all_ranks_as, fake
    )
    fake._full_fusion_local_workspace_region = MethodType(
        MegaMoE._full_fusion_local_workspace_region, fake
    )
    fake._copy_to_full_fusion_local_region = MethodType(
        MegaMoE._copy_to_full_fusion_local_region, fake
    )
    fake._full_fusion_local_workspace_region_as = MethodType(
        MegaMoE._full_fusion_local_workspace_region_as, fake
    )
    fake._stage_full_fusion_dispatch_inputs = MethodType(
        MegaMoE._stage_full_fusion_dispatch_inputs, fake
    )
    fake._stage_full_fusion_dispatch_inputs_for_m5 = MethodType(
        MegaMoE._stage_full_fusion_dispatch_inputs_for_m5, fake
    )
    fake._full_fusion_m5_control_words = MethodType(MegaMoE._full_fusion_m5_control_words, fake)
    fake._publish_full_fusion_m5_producer_ready = MethodType(
        MegaMoE._publish_full_fusion_m5_producer_ready, fake
    )
    fake._collect_full_fusion_m5_ready_token_counts = MethodType(
        MegaMoE._collect_full_fusion_m5_ready_token_counts, fake
    )
    fake._wait_full_fusion_m5_ready_token_counts = MethodType(
        MegaMoE._wait_full_fusion_m5_ready_token_counts, fake
    )
    fake._publish_full_fusion_m5_consumer_ready = MethodType(
        MegaMoE._publish_full_fusion_m5_consumer_ready, fake
    )
    fake._collect_full_fusion_m5_consumers_ready = MethodType(
        MegaMoE._collect_full_fusion_m5_consumers_ready, fake
    )
    fake._wait_full_fusion_m5_consumers_ready = MethodType(
        MegaMoE._wait_full_fusion_m5_consumers_ready, fake
    )
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = MethodType(
        MegaMoE._sync_full_fusion_m5_producers_and_materialize_with_counts, fake
    )
    fake._sync_full_fusion_m5_producers_and_materialize = MethodType(
        MegaMoE._sync_full_fusion_m5_producers_and_materialize, fake
    )
    fake._finalize_full_fusion_m5_route_pull_if_requested = MethodType(
        MegaMoE._finalize_full_fusion_m5_route_pull_if_requested, fake
    )
    fake._full_fusion_m6_control_words = MethodType(MegaMoE._full_fusion_m6_control_words, fake)
    fake._full_fusion_combine_buffer = MethodType(MegaMoE._full_fusion_combine_buffer, fake)
    fake._full_fusion_combine_buffers_all_ranks = MethodType(
        MegaMoE._full_fusion_combine_buffers_all_ranks, fake
    )
    fake._full_fusion_combine_buffer_output_view = MethodType(
        MegaMoE._full_fusion_combine_buffer_output_view, fake
    )
    fake._clear_full_fusion_local_direct_combine_buffer_output = MethodType(
        MegaMoE._clear_full_fusion_local_direct_combine_buffer_output, fake
    )
    fake._full_fusion_ranked_route_output_buffer = MethodType(
        MegaMoE._full_fusion_ranked_route_output_buffer, fake
    )
    fake._full_fusion_ranked_route_output_buffers_all_ranks = MethodType(
        MegaMoE._full_fusion_ranked_route_output_buffers_all_ranks, fake
    )
    fake._publish_full_fusion_route_output_producer_ready = MethodType(
        MegaMoE._publish_full_fusion_route_output_producer_ready, fake
    )
    fake._collect_full_fusion_route_output_producers_ready = MethodType(
        MegaMoE._collect_full_fusion_route_output_producers_ready, fake
    )
    fake._wait_full_fusion_route_output_producers_ready = MethodType(
        MegaMoE._wait_full_fusion_route_output_producers_ready, fake
    )
    fake._stage_full_fusion_ranked_route_outputs = MethodType(
        MegaMoE._stage_full_fusion_ranked_route_outputs, fake
    )
    fake._publish_full_fusion_ranked_route_outputs_from_weighted_route_outputs = MethodType(
        MegaMoE._publish_full_fusion_ranked_route_outputs_from_weighted_route_outputs, fake
    )
    fake._sync_full_fusion_ranked_route_outputs_from_weighted_route_outputs = MethodType(
        MegaMoE._sync_full_fusion_ranked_route_outputs_from_weighted_route_outputs, fake
    )
    fake._materialize_full_fusion_weighted_route_outputs_from_staged_ranked_route_outputs = MethodType(
        MegaMoE._materialize_full_fusion_weighted_route_outputs_from_staged_ranked_route_outputs,
        fake,
    )
    fake._build_full_fusion_m5_pool_route_inputs = MethodType(
        MegaMoE._build_full_fusion_m5_pool_route_inputs, fake
    )
    fake._publish_full_fusion_m5_trusted_direct_topk_route_layout = MethodType(
        MegaMoE._publish_full_fusion_m5_trusted_direct_topk_route_layout, fake
    )
    fake._publish_full_fusion_m5_direct_pool_fc_route_layout = MethodType(
        MegaMoE._publish_full_fusion_m5_direct_pool_fc_route_layout, fake
    )
    fake._build_full_fusion_m5_active_pool_slots = MethodType(
        MegaMoE._build_full_fusion_m5_active_pool_slots, fake
    )
    fake._build_full_fusion_m5_direct_pool_fc_route_metadata = MethodType(
        MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata, fake
    )
    fake._build_full_fusion_m5_direct_combine_output_metadata = MethodType(
        MegaMoE._build_full_fusion_m5_direct_combine_output_metadata, fake
    )
    fake._build_full_fusion_m5_route_error_masks = MegaMoE._build_full_fusion_m5_route_error_masks
    fake._build_full_fusion_m5_moe_sort_counts = MegaMoE._build_full_fusion_m5_moe_sort_counts
    fake._plan_full_fusion_m6_output = MethodType(MegaMoE._plan_full_fusion_m6_output, fake)
    fake._apply_full_fusion_m6_output_plan = MethodType(
        MegaMoE._apply_full_fusion_m6_output_plan, fake
    )
    fake._run_full_fusion_m5_pool_route_outputs = MethodType(
        MegaMoE._run_full_fusion_m5_pool_route_outputs, fake
    )
    fake._run_full_fusion_m5_pool_route_output_producer = MethodType(
        MegaMoE._run_full_fusion_m5_pool_route_output_producer, fake
    )
    fake._run_full_fusion_m5_m6_output_path = MethodType(
        MegaMoE._run_full_fusion_m5_m6_output_path, fake
    )
    fake._run_full_fusion_m6_output_plan = MethodType(MegaMoE._run_full_fusion_m6_output_plan, fake)
    fake._run_full_fusion_distributed_output_path = MethodType(
        MegaMoE._run_full_fusion_distributed_output_path, fake
    )
    fake._sync_full_fusion_route_output_producers_and_materialize = MethodType(
        MegaMoE._sync_full_fusion_route_output_producers_and_materialize, fake
    )
    fake._is_full_fusion_invalid_token_metadata = MegaMoE._is_full_fusion_invalid_token_metadata
    fake._has_full_fusion_partial_invalid_token_metadata = (
        MegaMoE._has_full_fusion_partial_invalid_token_metadata
    )
    fake._validate_full_fusion_route_metadata = MethodType(
        MegaMoE._validate_full_fusion_route_metadata, fake
    )
    fake._trusted_full_fusion_route_metadata = MegaMoE._trusted_full_fusion_route_metadata
    fake._materialize_full_fusion_combine_push = MethodType(
        MegaMoE._materialize_full_fusion_combine_push, fake
    )
    fake._materialize_full_fusion_combine_layout_push = MethodType(
        MegaMoE._materialize_full_fusion_combine_layout_push, fake
    )
    fake._publish_full_fusion_m6_combine_ready = MethodType(
        MegaMoE._publish_full_fusion_m6_combine_ready, fake
    )
    fake._collect_full_fusion_m6_combine_ready = MethodType(
        MegaMoE._collect_full_fusion_m6_combine_ready, fake
    )
    fake._wait_full_fusion_m6_combine_ready = MethodType(
        MegaMoE._wait_full_fusion_m6_combine_ready, fake
    )
    fake._reduce_full_fusion_combine_buffer = MethodType(
        MegaMoE._reduce_full_fusion_combine_buffer, fake
    )
    fake._sync_full_fusion_m6_combine_push_and_reduce = MethodType(
        MegaMoE._sync_full_fusion_m6_combine_push_and_reduce, fake
    )
    fake._sync_full_fusion_m6_combine_layout_and_reduce = MethodType(
        MegaMoE._sync_full_fusion_m6_combine_layout_and_reduce, fake
    )
    fake._sync_full_fusion_m6_direct_combine_buffer_and_reduce = MethodType(
        MegaMoE._sync_full_fusion_m6_direct_combine_buffer_and_reduce, fake
    )
    fake._sync_full_fusion_m5_m6_materialize_and_reduce = MethodType(
        MegaMoE._sync_full_fusion_m5_m6_materialize_and_reduce, fake
    )
    fake._materialize_full_fusion_weighted_route_outputs_from_topk_route_outputs = MethodType(
        MegaMoE._materialize_full_fusion_weighted_route_outputs_from_topk_route_outputs, fake
    )
    fake._materialize_full_fusion_weighted_route_outputs_from_ranked_topk_route_outputs = (
        MethodType(
            MegaMoE._materialize_full_fusion_weighted_route_outputs_from_ranked_topk_route_outputs,
            fake,
        )
    )
    fake._materialize_full_fusion_dispatch_pull = MethodType(
        MegaMoE._materialize_full_fusion_dispatch_pull, fake
    )
    fake._normalize_full_fusion_dispatch_token_counts = MethodType(
        MegaMoE._normalize_full_fusion_dispatch_token_counts, fake
    )
    fake._full_fusion_round_robin_rank_routes = MegaMoE._full_fusion_round_robin_rank_routes
    return fake, rank_stride_bytes


def _attach_disabled_full_fusion_profile_helpers(fake: SimpleNamespace) -> None:
    fake._full_fusion_profile_enabled = False
    fake._full_fusion_profile_events = {}
    fake._full_fusion_profile_device = MethodType(MegaMoE._full_fusion_profile_device, fake)
    fake._synchronize_full_fusion_profile_device = MethodType(
        MegaMoE._synchronize_full_fusion_profile_device, fake
    )
    fake._start_full_fusion_profile_event = MethodType(
        MegaMoE._start_full_fusion_profile_event, fake
    )
    fake._finish_full_fusion_profile_event = MethodType(
        MegaMoE._finish_full_fusion_profile_event, fake
    )


def test_megamoe_output_path_diagnostics_default_to_not_used() -> None:
    fake = SimpleNamespace()

    assert MegaMoE.full_fusion_output_path_used.fget(fake) is False
    assert MegaMoE.full_fusion_output_path_layout.fget(fake) is None
    assert MegaMoE.full_fusion_output_path_status.fget(fake) == {
        "requested": False,
        "eligible": False,
        "runtime_requested": False,
        "runtime_eligible": False,
        "output_path_ready": False,
        "planned_layout": None,
        "pre_dispatch_used": False,
        "used": False,
        "layout": None,
        "m5_dispatch_materialize_kernel": None,
        "m5_dispatch_materialize_strategy": None,
        "m5_standalone_materialization_scope": None,
        "m5_debug_materialization_gates": (),
        "m6_combine_reduce_kernel": None,
        "final_kernel_path": None,
        "final_kernel_ready": False,
        "fallback_reason": None,
        "fallback_stage": None,
        "fallback_code": None,
        "dispatch_stage_fallback_reason": None,
        "staged_direct_topk_fallback_reason": None,
        "m5_pre_materialization_support_reason": None,
        "dispatch_pull_fallback_reason": None,
        "combine_push_fallback_reason": None,
        "diagnostics": {},
    }


def test_megamoe_output_path_status_reports_current_snapshot() -> None:
    fake = SimpleNamespace(
        _full_fusion_output_path_requested=True,
        _full_fusion_runtime_gate=SimpleNamespace(
            requested=True, use_full_fusion=True, output_path_ready=True
        ),
        _full_fusion_m6_output_plan=SimpleNamespace(layout="combine_buffer"),
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout=None,
        _full_fusion_m6_combine_reduce_kernel=None,
        _full_fusion_m5_dispatch_materialize_strategy="helper_only",
        _full_fusion_m5_standalone_materialization_scope="post_dispatch_output_path",
        _full_fusion_m5_route_pull_verify_enabled=True,
        _full_fusion_m5_helper_materialize_enabled=True,
        _full_fusion_output_path_fallback_reason="route output unavailable",
        _full_fusion_output_path_fallback_stage="m6_output_path",
        _full_fusion_output_path_fallback_code="route_output_unavailable",
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_staged_direct_topk_fallback_reason="monolithic staged direct-topk unavailable",
        _full_fusion_m5_pre_materialization_support_reason=(
            "explicit M5 debug/materialization gate requested"
        ),
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_combine_push_fallback_reason=None,
        _full_fusion_fallback_diagnostics={},
    )
    MegaMoE._record_full_fusion_fallback(
        fake, "m6_output_path", "route_output_unavailable", "route output unavailable"
    )

    assert MegaMoE.full_fusion_output_path_status.fget(fake) == {
        "requested": True,
        "eligible": True,
        "runtime_requested": True,
        "runtime_eligible": True,
        "output_path_ready": True,
        "planned_layout": "combine_buffer",
        "pre_dispatch_used": False,
        "used": False,
        "layout": None,
        "m5_dispatch_materialize_kernel": None,
        "m5_dispatch_materialize_strategy": "helper_only",
        "m5_standalone_materialization_scope": "post_dispatch_output_path",
        "m5_debug_materialization_gates": ("route_pull_verify", "helper_materialize"),
        "m6_combine_reduce_kernel": None,
        "final_kernel_path": None,
        "final_kernel_ready": False,
        "fallback_reason": "route output unavailable",
        "fallback_stage": "m6_output_path",
        "fallback_code": "route_output_unavailable",
        "dispatch_stage_fallback_reason": None,
        "staged_direct_topk_fallback_reason": "monolithic staged direct-topk unavailable",
        "m5_pre_materialization_support_reason": (
            "explicit M5 debug/materialization gate requested"
        ),
        "dispatch_pull_fallback_reason": None,
        "combine_push_fallback_reason": None,
        "diagnostics": {
            "m6_output_path.route_output_unavailable": {
                "stage": "m6_output_path",
                "code": "route_output_unavailable",
                "message": "route output unavailable",
                "count": 1,
            }
        },
    }


def test_megamoe_output_path_status_reports_reconstruction_materialize_gate() -> None:
    fake = SimpleNamespace(
        _full_fusion_runtime_gate=SimpleNamespace(
            requested=True, use_full_fusion=True, output_path_ready=True
        ),
        _full_fusion_m5_reconstruction_materialize_enabled=True,
        _full_fusion_fallback_diagnostics={},
    )

    assert MegaMoE._active_full_fusion_m5_debug_materialization_gates(fake) == (
        "reconstruction_materialize",
    )
    assert MegaMoE.full_fusion_m5_debug_materialization_gates.fget(fake) == (
        "reconstruction_materialize",
    )
    assert MegaMoE.full_fusion_output_path_status.fget(fake)["m5_debug_materialization_gates"] == (
        "reconstruction_materialize",
    )
    assert MegaMoE._full_fusion_m5_validation_gate_enabled(fake) is True
    assert (
        MegaMoE._get_full_fusion_m5_pre_materialization_support_reason(
            fake, "staged direct-topk failed"
        )
        == "explicit M5 debug/materialization gate requested"
    )


def test_megamoe_output_path_fallback_helpers_track_stage_and_code() -> None:
    fake = SimpleNamespace(_full_fusion_fallback_diagnostics={})

    reason = MegaMoE._set_full_fusion_output_path_fallback(
        fake, "m6_output_path", "route_output_unavailable", "route output unavailable"
    )

    assert reason == "route output unavailable"
    assert fake._full_fusion_output_path_fallback_reason == "route output unavailable"
    assert fake._full_fusion_output_path_fallback_stage == "m6_output_path"
    assert fake._full_fusion_output_path_fallback_code == "route_output_unavailable"
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["fallback_reason"] == "route output unavailable"
    assert status["fallback_stage"] == "m6_output_path"
    assert status["fallback_code"] == "route_output_unavailable"

    assert (
        MegaMoE._finish_full_fusion_output_path_attempt(fake, "m6_output_path", "unused", None)
        is None
    )
    assert fake._full_fusion_output_path_fallback_reason is None
    assert fake._full_fusion_output_path_fallback_stage is None
    assert fake._full_fusion_output_path_fallback_code is None
    assert MegaMoE.full_fusion_output_path_status.fget(fake)["fallback_reason"] is None
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_path.route_output_unavailable"] == {
        "stage": "m6_output_path",
        "code": "route_output_unavailable",
        "message": "route output unavailable",
        "count": 1,
    }


def test_megamoe_output_path_attempt_reset_clears_stale_status() -> None:
    fake = SimpleNamespace(
        _full_fusion_pre_dispatch_output_path_used=True,
        _full_fusion_output_path_used=True,
        _full_fusion_output_path_layout="combine_buffer",
        _full_fusion_staged_direct_topk_fallback_reason="stale staged fallback",
        _full_fusion_m5_pre_materialization_support_reason="stale M5 support reason",
        _full_fusion_m5_standalone_materialization_scope="stale M5 scope",
        _full_fusion_m5_dispatch_materialize_strategy="stale M5 strategy",
        _full_fusion_m6_output_plan=SimpleNamespace(layout="combine_buffer"),
    )

    MegaMoE._reset_full_fusion_output_path_attempt(fake)

    assert fake._full_fusion_pre_dispatch_output_path_used is False
    assert fake._full_fusion_output_path_used is False
    assert fake._full_fusion_output_path_layout is None
    assert fake._full_fusion_staged_direct_topk_fallback_reason is None
    assert fake._full_fusion_m5_pre_materialization_support_reason is None
    assert fake._full_fusion_m5_standalone_materialization_scope is None
    assert fake._full_fusion_m5_dispatch_materialize_strategy is None
    assert fake._full_fusion_m6_output_plan is None


def test_megamoe_full_fusion_profile_events_are_explicitly_enabled() -> None:
    fake, _ = _fake_megamoe_with_workspace()

    disabled_start = MegaMoE._start_full_fusion_profile_event(fake)
    MegaMoE._finish_full_fusion_profile_event(fake, "disabled_event", disabled_start)

    assert disabled_start is None
    assert MegaMoE.full_fusion_profile_events.fget(fake) == {}

    fake._full_fusion_profile_enabled = True
    enabled_start = MegaMoE._start_full_fusion_profile_event(fake)
    MegaMoE._finish_full_fusion_profile_event(fake, "enabled_event", enabled_start)

    events = MegaMoE.full_fusion_profile_events.fget(fake)
    assert enabled_start is not None
    assert len(events["enabled_event"]) == 1
    assert events["enabled_event"][0] >= 0.0

    fake.reset_full_fusion_profile_events()

    assert MegaMoE.full_fusion_profile_events.fget(fake) == {}


def test_megamoe_m5_route_error_masks_detect_invalid_and_duplicate_routes() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=3,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    all_topk_idx = torch.tensor(
        (
            (0, 1, 2),
            (1, 1, 3),
            (-1, 2, 3),
            (0, 4, 2),
        ),
        dtype=torch.int64,
    )

    invalid_expert, duplicate_expert = MegaMoE._build_full_fusion_m5_route_error_masks(
        all_topk_idx, config
    )

    assert invalid_expert.tolist() == [False, False, True, True]
    assert duplicate_expert.tolist() == [False, True, False, False]


def test_megamoe_m5_local_expert_route_counts_only_counts_local_experts() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=3,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    all_topk_idx = torch.tensor(
        (
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 2),
            (3, 1, 0),
        ),
        dtype=torch.int64,
    )

    counts = MegaMoE._build_full_fusion_m5_local_expert_route_counts(
        all_topk_idx,
        config,
        slot_start=2,
    )

    assert counts.tolist() == [3, 3]


def test_megamoe_m5_pool_sf_slots_handle_padded_blocks() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    pool_slots = torch.tensor((0, 1, 2, 3, 4, 5, 6, 7, 8), dtype=torch.int64)

    sf_slots = MegaMoE._full_fusion_m5_pool_sf_slots(pool_slots, config)

    assert sf_slots.tolist() == [0, 1, 2, 3, 128, 129, 130, 131, 256]

    l1_acts_sf_pool = torch.arange(config.num_padded_sf_pool_tokens, dtype=torch.uint8).reshape(
        -1, 1
    )
    pool_x_sf = MegaMoE._full_fusion_m5_pool_sf_rows(l1_acts_sf_pool, 9, config)

    assert torch.equal(pool_x_sf, l1_acts_sf_pool.index_select(0, sf_slots))


def test_megamoe_m5_pool_sf_rows_reuse_aligned_tile_view() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=128,
        num_experts=1,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        tile_size=128,
    )
    l1_acts_sf_pool = torch.arange(config.num_padded_sf_pool_tokens, dtype=torch.uint8).reshape(
        -1, 1
    )
    pool_slots = torch.tensor((0, 1, 127, 128, 129), dtype=torch.int64)

    sf_slots = MegaMoE._full_fusion_m5_pool_sf_slots(pool_slots, config)
    pool_x_sf = MegaMoE._full_fusion_m5_pool_sf_rows(l1_acts_sf_pool, 130, config)

    assert sf_slots is pool_slots
    assert pool_x_sf.data_ptr() == l1_acts_sf_pool.data_ptr()
    assert torch.equal(pool_x_sf, l1_acts_sf_pool[:130])


def test_megamoe_m5_moe_sort_count_builder_handles_uniform_token_counts() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    all_topk_idx = torch.tensor(
        (
            (2, 3),
            (0, 1),
            (3, 2),
            (2, 3),
        ),
        dtype=torch.int64,
    )

    send_counts, recv_counts = MegaMoE._build_full_fusion_m5_moe_sort_counts(
        all_topk_idx,
        (2, 2),
        config,
        local_rank=1,
        slot_start=2,
    )

    assert send_counts.tolist() == [0, 0, 2, 2]
    assert recv_counts.tolist() == [[1, 1], [2, 2]]


def test_megamoe_m5_moe_sort_count_builder_handles_nonuniform_token_counts() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    all_topk_idx = torch.tensor(
        (
            (2, 0),
            (3, 1),
            (2, 3),
            (3, 2),
            (1, 2),
        ),
        dtype=torch.int64,
    )

    send_counts, recv_counts = MegaMoE._build_full_fusion_m5_moe_sort_counts(
        all_topk_idx,
        (3, 2),
        config,
        local_rank=1,
        slot_start=2,
    )

    assert send_counts.tolist() == [0, 1, 2, 1]
    assert recv_counts.tolist() == [[2, 2], [2, 1]]


def test_megamoe_m5_moe_sort_count_builder_handles_single_rank() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    all_topk_idx = torch.tensor(
        (
            (0, 2),
            (1, 3),
            (2, 3),
        ),
        dtype=torch.int64,
    )

    send_counts, recv_counts = MegaMoE._build_full_fusion_m5_moe_sort_counts(
        all_topk_idx,
        (3,),
        config,
        local_rank=0,
        slot_start=0,
    )

    assert send_counts.tolist() == [1, 1, 2, 2]
    assert recv_counts.tolist() == [[1, 1, 2, 2]]


def test_megamoe_full_fusion_local_region_view_maps_descriptor_offsets() -> None:
    fake, _ = _fake_megamoe_with_workspace()
    descriptor = fake._full_fusion_runtime_gate.workspace_descriptor
    x_buf_region = descriptor.layout.region("x_buf")

    local_x_buf = MegaMoE._full_fusion_local_workspace_region(fake, "x_buf")

    assert local_x_buf is not None
    assert local_x_buf.numel() == x_buf_region.size_bytes
    local_x_buf.fill_(0x11)
    assert torch.all(
        fake._full_fusion_workspace[
            fake.mapping.moe_ep_rank,
            x_buf_region.offset : x_buf_region.end_offset,
        ]
        == 0x11
    )
    assert torch.all(
        fake._full_fusion_workspace[0, x_buf_region.offset : x_buf_region.end_offset] == 0xA5
    )


def test_megamoe_stages_dispatch_inputs_into_local_workspace_regions() -> None:
    fake, _ = _fake_megamoe_with_workspace()
    descriptor = fake._full_fusion_runtime_gate.workspace_descriptor
    config = descriptor.layout.config
    num_tokens = 3
    x_fp4 = torch.arange(num_tokens * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        num_tokens, config.hidden_size // 2
    )
    x_sf = torch.arange(num_tokens, dtype=torch.uint8).reshape(num_tokens, 1)
    token_selected_experts = torch.tensor(
        ((0, 1), (2, 3), (1, 0)),
        dtype=torch.int32,
    )
    token_final_scales = torch.tensor(
        ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125)),
        dtype=torch.float32,
    )

    ok, reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )

    assert ok is True
    assert reason is None

    x_buf = MegaMoE._full_fusion_local_workspace_region(fake, "x_buf")
    x_sf_buf = MegaMoE._full_fusion_local_workspace_region(fake, "x_sf_buf")
    topk_idx_buf = MegaMoE._full_fusion_local_workspace_region(fake, "topk_idx_buf")
    topk_weights_buf = MegaMoE._full_fusion_local_workspace_region(fake, "topk_weights_buf")
    l1_arrival_count = MegaMoE._full_fusion_local_workspace_region(fake, "l1_arrival_count")

    assert x_buf is not None
    assert x_sf_buf is not None
    assert topk_idx_buf is not None
    assert topk_weights_buf is not None
    assert l1_arrival_count is not None
    assert torch.equal(x_buf[: x_fp4.numel()], x_fp4.flatten())
    assert torch.all(x_buf[x_fp4.numel() :] == 0)
    assert torch.equal(x_sf_buf[: x_sf.numel()], x_sf.flatten())
    assert torch.all(x_sf_buf[x_sf.numel() :] == 0)
    assert torch.equal(
        topk_idx_buf[: token_selected_experts.numel() * UINT64_BYTES].view(torch.int64),
        token_selected_experts.to(dtype=torch.int64).flatten(),
    )
    torch.testing.assert_close(
        topk_weights_buf[: token_final_scales.numel() * FLOAT32_BYTES].view(torch.float32),
        token_final_scales.flatten(),
    )
    assert torch.all(l1_arrival_count == 0)


def test_megamoe_dispatch_input_staging_rejects_over_capacity_tokens() -> None:
    fake, _ = _fake_megamoe_with_workspace()
    descriptor = fake._full_fusion_runtime_gate.workspace_descriptor
    config = descriptor.layout.config
    num_tokens = config.max_num_tokens_per_rank + 1
    x_fp4 = torch.zeros((num_tokens, config.hidden_size // 2), dtype=torch.uint8)
    x_sf = torch.zeros(
        (num_tokens, config.hidden_size // config.scaling_vector_size), dtype=torch.uint8
    )
    token_selected_experts = torch.zeros((num_tokens, config.top_k), dtype=torch.int64)
    token_final_scales = torch.ones((num_tokens, config.top_k), dtype=torch.float32)

    ok, reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )

    assert ok is False
    assert reason is not None
    assert "exceeds workspace capacity" in reason


def test_megamoe_m5_dispatch_staging_returns_source_domain_token_count() -> None:
    fake, _ = _fake_megamoe_with_workspace()
    descriptor = fake._full_fusion_runtime_gate.workspace_descriptor
    config = descriptor.layout.config
    num_tokens = 3
    x_fp4 = torch.arange(num_tokens * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        num_tokens, config.hidden_size // 2
    )
    x_sf = torch.arange(num_tokens, dtype=torch.uint8).reshape(num_tokens, 1)
    token_selected_experts = torch.tensor(((0, 1), (2, 3), (1, 0)), dtype=torch.int32)
    token_final_scales = torch.ones((num_tokens, config.top_k), dtype=torch.float32)

    local_num_tokens, reason = MegaMoE._stage_full_fusion_dispatch_inputs_for_m5(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )

    assert reason is None
    assert local_num_tokens == num_tokens
    x_buf = MegaMoE._full_fusion_local_workspace_region(fake, "x_buf")
    assert x_buf is not None
    assert torch.equal(x_buf[: x_fp4.numel()], x_fp4.flatten())


def _copy_tensor_to_workspace_rank(
    fake: SimpleNamespace, rank: int, region_name: str, source: torch.Tensor
) -> None:
    region = MegaMoE._full_fusion_workspace_region(fake, region_name, rank=rank)
    assert region is not None
    source_bytes = source.contiguous().view(torch.uint8).flatten()
    assert source_bytes.numel() <= region.numel()
    region.zero_()
    region[: source_bytes.numel()].copy_(source_bytes)


def _write_m5_ready_control(
    fake: SimpleNamespace, *, rank: int, epoch: int, num_tokens: int
) -> None:
    control, reason = MegaMoE._full_fusion_m5_control_words(fake, rank=rank)
    assert reason is None
    assert control is not None
    control.copy_(
        torch.tensor(
            (
                MegaMoE._FULL_FUSION_M5_READY_MAGIC,
                epoch,
                num_tokens,
                MegaMoE._FULL_FUSION_M5_READY_FLAG,
                0,
                0,
            ),
            dtype=control.dtype,
            device=control.device,
        )
    )


def _write_m5_consumer_ready_control(fake: SimpleNamespace, *, rank: int, epoch: int) -> None:
    control, reason = MegaMoE._full_fusion_m5_control_words(fake, rank=rank)
    assert reason is None
    assert control is not None
    control[4:6].copy_(
        torch.tensor(
            (epoch, MegaMoE._FULL_FUSION_M5_READY_FLAG),
            dtype=control.dtype,
            device=control.device,
        )
    )


def _write_route_output_ready_control(fake: SimpleNamespace, *, rank: int, epoch: int) -> None:
    control, reason = MegaMoE._full_fusion_m6_control_words(fake, rank=rank)
    assert reason is None
    assert control is not None
    control[6:8].copy_(
        torch.tensor(
            (epoch, MegaMoE._FULL_FUSION_ROUTE_OUTPUT_READY_FLAG),
            dtype=control.dtype,
            device=control.device,
        )
    )


def _write_m6_ready_control(fake: SimpleNamespace, *, rank: int, epoch: int) -> None:
    control, reason = MegaMoE._full_fusion_m6_control_words(fake, rank=rank)
    assert reason is None
    assert control is not None
    control[8:10].copy_(
        torch.tensor(
            (epoch, MegaMoE._FULL_FUSION_M6_READY_FLAG),
            dtype=control.dtype,
            device=control.device,
        )
    )


def _write_runtime_token_src_metadata(
    fake: SimpleNamespace, metadata: tuple[tuple[int, int, int], ...]
) -> None:
    region = MegaMoE._full_fusion_local_workspace_region(fake, "token_src_metadata")
    assert region is not None
    metadata_region = region.view(torch.int32).reshape(-1, 3)
    metadata_region.fill_(-1)
    for pool_slot, values in enumerate(metadata):
        metadata_region[pool_slot].copy_(torch.tensor(values, dtype=torch.int32))


def _zero_all_combine_buffers(fake: SimpleNamespace, ep_size: int) -> None:
    for rank in range(ep_size):
        combine_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=rank)
        assert reason is None
        assert combine_buffer is not None
        combine_buffer.zero_()


def test_megamoe_materializes_single_rank_dispatch_pull_from_staged_regions() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=3,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    fake._full_fusion_m5_reconstruction_materialize_enabled = True
    num_tokens = 3
    x_fp4 = torch.arange(num_tokens * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        num_tokens, config.hidden_size // 2
    )
    x_sf = torch.arange(
        num_tokens * (config.hidden_size // config.scaling_vector_size), dtype=torch.uint8
    ).reshape(num_tokens, config.hidden_size // config.scaling_vector_size)
    token_selected_experts = torch.tensor(((2, 0), (1, 2), (0, 1)), dtype=torch.int32)
    token_final_scales = torch.tensor(
        ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125)),
        dtype=torch.float32,
    )

    staged, stage_reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )
    materialized, materialize_reason = MegaMoE._materialize_full_fusion_dispatch_pull(
        fake, (num_tokens,)
    )

    assert staged is True
    assert stage_reason is None
    assert materialized is True
    assert materialize_reason is None
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) == "python_reconstruct"

    expert_send_count = MegaMoE._full_fusion_local_workspace_region(fake, "expert_send_count").view(
        torch.int64
    )
    expert_recv_count = MegaMoE._full_fusion_local_workspace_region(fake, "expert_recv_count").view(
        torch.int64
    )
    expert_recv_count_sum = MegaMoE._full_fusion_local_workspace_region(
        fake, "expert_recv_count_sum"
    ).view(torch.int64)
    src_token_topk_idx = MegaMoE._full_fusion_local_workspace_region(
        fake, "src_token_topk_idx"
    ).view(torch.int32)
    token_src_metadata = MegaMoE._full_fusion_local_workspace_region(
        fake, "token_src_metadata"
    ).view(torch.int32)
    l1_arrival_count = MegaMoE._full_fusion_local_workspace_region(fake, "l1_arrival_count").view(
        torch.int32
    )
    l1_topk_weights_pool = MegaMoE._full_fusion_local_workspace_region(
        fake, "l1_topk_weights_pool"
    ).view(torch.float32)
    l1_acts_pool = MegaMoE._full_fusion_local_workspace_region(fake, "l1_acts_pool").reshape(
        config.num_max_pool_tokens, config.hidden_size // 2
    )
    l1_acts_sf_pool = MegaMoE._full_fusion_local_workspace_region(fake, "l1_acts_sf_pool").reshape(
        config.num_padded_sf_pool_tokens, config.hidden_size // config.scaling_vector_size
    )

    assert expert_send_count[: config.num_experts].tolist() == [2, 2, 2]
    assert expert_recv_count.reshape(config.ep_size, config.num_experts_per_rank)[0].tolist() == [
        2,
        2,
        2,
    ]
    assert expert_recv_count_sum[: config.num_experts_per_rank].tolist() == [2, 2, 2]

    expected_src_idx = [-1] * 12
    expected_src_idx[0:2] = [1, 4]
    expected_src_idx[4:6] = [2, 5]
    expected_src_idx[8:10] = [0, 3]
    assert src_token_topk_idx[:12].tolist() == expected_src_idx

    token_src_metadata = token_src_metadata.reshape(config.num_max_pool_tokens, 3)
    assert token_src_metadata[0].tolist() == [0, 0, 1]
    assert token_src_metadata[1].tolist() == [0, 2, 0]
    assert token_src_metadata[4].tolist() == [0, 1, 0]
    assert token_src_metadata[5].tolist() == [0, 2, 1]
    assert token_src_metadata[8].tolist() == [0, 0, 0]
    assert token_src_metadata[9].tolist() == [0, 1, 1]
    assert token_src_metadata[2].tolist() == [-1, -1, -1]

    assert l1_arrival_count[: config.l1_arrival_count_entries].tolist() == [2, 2, 2, 0, 0, 0]
    torch.testing.assert_close(l1_topk_weights_pool[0], token_final_scales[0, 1])
    torch.testing.assert_close(l1_topk_weights_pool[1], token_final_scales[2, 0])
    torch.testing.assert_close(l1_topk_weights_pool[4], token_final_scales[1, 0])
    torch.testing.assert_close(l1_topk_weights_pool[5], token_final_scales[2, 1])
    torch.testing.assert_close(l1_topk_weights_pool[8], token_final_scales[0, 0])
    torch.testing.assert_close(l1_topk_weights_pool[9], token_final_scales[1, 1])

    assert torch.equal(l1_acts_pool[0], x_fp4[0])
    assert torch.equal(l1_acts_pool[1], x_fp4[2])
    assert torch.equal(l1_acts_pool[4], x_fp4[1])
    assert torch.equal(l1_acts_pool[5], x_fp4[2])
    assert torch.equal(l1_acts_pool[8], x_fp4[0])
    assert torch.equal(l1_acts_pool[9], x_fp4[1])
    assert torch.all(l1_acts_pool[2] == 0)

    assert torch.equal(l1_acts_sf_pool[0], x_sf[0])
    assert torch.equal(l1_acts_sf_pool[1], x_sf[2])
    assert torch.equal(l1_acts_sf_pool[128], x_sf[1])
    assert torch.equal(l1_acts_sf_pool[129], x_sf[2])
    assert torch.equal(l1_acts_sf_pool[256], x_sf[0])
    assert torch.equal(l1_acts_sf_pool[257], x_sf[1])
    assert torch.all(l1_acts_sf_pool[2] == 0)


def test_megamoe_materializes_multi_rank_dispatch_pull_from_peer_staged_regions() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_reconstruction_materialize_enabled = True
    rank0_tokens = 3
    rank1_tokens = 2
    rank0_x = (
        torch.arange(rank0_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 10
    ).reshape(rank0_tokens, config.hidden_size // 2)
    rank1_x = (
        torch.arange(rank1_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 100
    ).reshape(rank1_tokens, config.hidden_size // 2)
    rank0_sf = (torch.arange(rank0_tokens, dtype=torch.uint8) + 1).reshape(rank0_tokens, 1)
    rank1_sf = (torch.arange(rank1_tokens, dtype=torch.uint8) + 50).reshape(rank1_tokens, 1)
    rank0_experts = torch.tensor(((2, 0), (3, 1), (2, 3)), dtype=torch.int64)
    rank1_experts = torch.tensor(((3, 2), (1, 2)), dtype=torch.int64)
    rank0_weights = torch.tensor(((0.2, 9.0), (0.3, 8.0), (0.4, 0.5)), dtype=torch.float32)
    rank1_weights = torch.tensor(((0.6, 0.7), (8.0, 0.8)), dtype=torch.float32)

    _copy_tensor_to_workspace_rank(fake, 0, "x_buf", rank0_x)
    _copy_tensor_to_workspace_rank(fake, 0, "x_sf_buf", rank0_sf)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_idx_buf", rank0_experts)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_weights_buf", rank0_weights)
    _copy_tensor_to_workspace_rank(fake, 1, "x_buf", rank1_x)
    _copy_tensor_to_workspace_rank(fake, 1, "x_sf_buf", rank1_sf)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_idx_buf", rank1_experts)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_weights_buf", rank1_weights)

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (rank0_tokens, rank1_tokens))

    assert ok is True
    assert reason is None
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) == "python_reconstruct"

    expert_send_count = MegaMoE._full_fusion_local_workspace_region(fake, "expert_send_count").view(
        torch.int64
    )
    expert_recv_count = MegaMoE._full_fusion_local_workspace_region(fake, "expert_recv_count").view(
        torch.int64
    )
    expert_recv_count_sum = MegaMoE._full_fusion_local_workspace_region(
        fake, "expert_recv_count_sum"
    ).view(torch.int64)
    src_token_topk_idx = MegaMoE._full_fusion_local_workspace_region(
        fake, "src_token_topk_idx"
    ).view(torch.int32)
    token_src_metadata = MegaMoE._full_fusion_local_workspace_region(
        fake, "token_src_metadata"
    ).view(torch.int32)
    l1_arrival_count = MegaMoE._full_fusion_local_workspace_region(fake, "l1_arrival_count").view(
        torch.int32
    )
    l1_topk_weights_pool = MegaMoE._full_fusion_local_workspace_region(
        fake, "l1_topk_weights_pool"
    ).view(torch.float32)
    l1_acts_pool = MegaMoE._full_fusion_local_workspace_region(fake, "l1_acts_pool").reshape(
        config.num_max_pool_tokens, config.hidden_size // 2
    )
    l1_acts_sf_pool = MegaMoE._full_fusion_local_workspace_region(fake, "l1_acts_sf_pool").reshape(
        config.num_padded_sf_pool_tokens, config.hidden_size // config.scaling_vector_size
    )

    assert expert_send_count[: config.num_experts].tolist() == [0, 1, 2, 1]
    assert expert_recv_count.reshape(config.ep_size, config.num_experts_per_rank).tolist() == [
        [2, 2],
        [2, 1],
    ]
    assert expert_recv_count_sum[: config.num_experts_per_rank].tolist() == [4, 3]

    max_recv = config.max_recv_tokens_per_expert
    assert src_token_topk_idx[0:2].tolist() == [0, 4]
    assert src_token_topk_idx[max_recv : max_recv + 2].tolist() == [1, 3]
    expert1_base = config.ep_size * max_recv
    assert src_token_topk_idx[expert1_base : expert1_base + 2].tolist() == [2, 5]
    assert src_token_topk_idx[expert1_base + max_recv].item() == 0

    token_src_metadata = token_src_metadata.reshape(config.num_max_pool_tokens, 3)
    assert token_src_metadata[0].tolist() == [0, 0, 0]
    assert token_src_metadata[1].tolist() == [1, 0, 1]
    assert token_src_metadata[2].tolist() == [0, 2, 0]
    assert token_src_metadata[3].tolist() == [1, 1, 1]
    assert token_src_metadata[4].tolist() == [0, 1, 0]
    assert token_src_metadata[5].tolist() == [1, 0, 0]
    assert token_src_metadata[6].tolist() == [0, 2, 1]
    assert token_src_metadata[7].tolist() == [-1, -1, -1]

    assert l1_arrival_count[: config.l1_arrival_count_entries].tolist() == [4, 3, 0, 0, 0, 0]
    torch.testing.assert_close(l1_topk_weights_pool[0], rank0_weights[0, 0])
    torch.testing.assert_close(l1_topk_weights_pool[1], rank1_weights[0, 1])
    torch.testing.assert_close(l1_topk_weights_pool[2], rank0_weights[2, 0])
    torch.testing.assert_close(l1_topk_weights_pool[3], rank1_weights[1, 1])
    torch.testing.assert_close(l1_topk_weights_pool[4], rank0_weights[1, 0])
    torch.testing.assert_close(l1_topk_weights_pool[5], rank1_weights[0, 0])
    torch.testing.assert_close(l1_topk_weights_pool[6], rank0_weights[2, 1])

    assert torch.equal(l1_acts_pool[0], rank0_x[0])
    assert torch.equal(l1_acts_pool[1], rank1_x[0])
    assert torch.equal(l1_acts_pool[2], rank0_x[2])
    assert torch.equal(l1_acts_pool[3], rank1_x[1])
    assert torch.equal(l1_acts_pool[4], rank0_x[1])
    assert torch.equal(l1_acts_pool[5], rank1_x[0])
    assert torch.equal(l1_acts_pool[6], rank0_x[2])
    assert torch.all(l1_acts_pool[7] == 0)

    assert torch.equal(l1_acts_sf_pool[0], rank0_sf[0])
    assert torch.equal(l1_acts_sf_pool[1], rank1_sf[0])
    assert torch.equal(l1_acts_sf_pool[2], rank0_sf[2])
    assert torch.equal(l1_acts_sf_pool[3], rank1_sf[1])
    assert torch.equal(l1_acts_sf_pool[128], rank0_sf[1])
    assert torch.equal(l1_acts_sf_pool[129], rank1_sf[0])
    assert torch.equal(l1_acts_sf_pool[130], rank0_sf[2])
    assert torch.all(l1_acts_sf_pool[131] == 0)

    cached_route_layout = fake._full_fusion_m5_direct_pool_fc_route_layout
    assert cached_route_layout is not None
    cached_tile_idx_to_expert_idx, cached_tile_idx_to_mn_limit, cached_num_tiles = (
        cached_route_layout
    )
    assert cached_tile_idx_to_expert_idx.tolist() == [0, 1]
    assert cached_tile_idx_to_mn_limit.tolist() == [4, 8]
    assert cached_num_tiles.tolist() == [2]
    assert l1_arrival_count[: cached_num_tiles.item()].tolist() == [4, 3]

    expert_recv_count_sum[: config.num_experts_per_rank].fill_(-1)
    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=8
    )
    assert reason is None
    assert route_metadata is not None
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    ) = route_metadata
    assert tile_idx_to_expert_idx.tolist() == [0, 1]
    assert tile_idx_to_mn_limit.tolist() == [4, 8]
    assert permuted_idx_to_expanded_idx.tolist() == list(range(8))
    assert num_non_exiting_tiles.tolist() == [2]


def test_megamoe_dispatch_pull_materialization_rejects_invalid_token_counts() -> None:
    fake, _ = _fake_megamoe_with_workspace()

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (1,))

    assert ok is False
    assert reason is not None
    assert "must contain 2 entries" in reason


def test_megamoe_dispatch_pull_helper_materialization_requires_all_helper_gates() -> None:
    fake, _ = _fake_megamoe_with_workspace()
    fake._full_fusion_m5_helper_materialize_enabled = True

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (1, 1))

    assert ok is False
    assert reason is not None
    assert "helper-only materialization requires route-pull" in reason
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) is None


def test_megamoe_dispatch_pull_helper_materialization_records_strategy_after_required_gates() -> (
    None
):
    fake, _ = _fake_megamoe_with_workspace()
    fake._full_fusion_m5_helper_materialize_enabled = True
    fake._full_fusion_m5_route_pull_materialize_enabled = True
    fake._full_fusion_m5_route_metadata_materialize_enabled = True
    fake._full_fusion_m5_pool_metadata_materialize_enabled = True
    finalized_token_counts = []

    def finalize_route_pull(token_counts: tuple[int, ...]) -> tuple[bool, str | None]:
        finalized_token_counts.append(tuple(token_counts))
        return True, None

    fake._finalize_full_fusion_m5_route_pull_if_requested = finalize_route_pull

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (1, 1))

    assert ok is True
    assert reason is None
    assert finalized_token_counts == [(1, 1)]
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) == "helper_only"


def _stage_dispatch_pull_for_materialization_strategy(
    strategy: str,
) -> tuple[SimpleNamespace, MegaMoeFullFusionWorkspaceConfig, tuple[int, ...]]:
    config = _small_single_rank_config()
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    if strategy != "direct_topk":
        fake._full_fusion_m5_direct_pool_fc_route_enabled = True
    num_tokens = 2
    x_fp4 = torch.arange(num_tokens * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        num_tokens, config.hidden_size // 2
    )
    x_sf = torch.arange(
        num_tokens * (config.hidden_size // config.scaling_vector_size), dtype=torch.uint8
    ).reshape(num_tokens, config.hidden_size // config.scaling_vector_size)
    token_selected_experts = torch.tensor(((0, 1), (1, 2)), dtype=torch.int64)
    token_final_scales = torch.tensor(((0.25, 1.0), (0.5, 0.75)), dtype=torch.float32)

    staged, reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )

    assert staged is True
    assert reason is None
    return fake, config, (num_tokens,)


def test_megamoe_dispatch_pull_rejects_python_reconstruction_without_debug_gate() -> None:
    fake, _config, token_counts = _stage_dispatch_pull_for_materialization_strategy(
        "python_reconstruct"
    )

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, token_counts)

    assert ok is False
    assert reason == (
        "M5 dispatch-pull Python reconstruction requires an explicit M5 debug/materialization gate"
    )
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) is None


def _fake_moe_sort_for_m5_strategy(
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del token_final_scales, num_experts, top_k, local_expert_offset, local_num_experts
    device = token_selected_experts.device
    return (
        torch.tensor((0,), dtype=torch.int32, device=device),
        torch.tensor((2,), dtype=torch.int32, device=device),
        torch.arange(tile_tokens_dim, dtype=torch.int64, device=device),
        torch.arange(tile_tokens_dim, dtype=torch.int64, device=device),
        torch.tensor(tile_tokens_dim, dtype=torch.int32, device=device),
        torch.tensor((1,), dtype=torch.int32, device=device),
    )


def _fill_direct_materialization_outputs(
    active_pool_slots_buffer: torch.Tensor,
    active_combine_rows_buffer: torch.Tensor,
    active_route_count: torch.Tensor,
    active_combine_output_mapping: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
) -> None:
    active_pool_slots_buffer[:2].copy_(torch.tensor((0, 1), dtype=torch.int64))
    active_combine_rows_buffer[:2].copy_(torch.tensor((0, 1), dtype=torch.int64))
    active_route_count[0] = 2
    active_combine_output_mapping[:4].fill_(8)
    active_combine_output_mapping[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    tile_idx_to_expert_idx[0] = 0
    tile_idx_to_mn_limit[0] = 2
    num_non_exiting_tiles[0] = 1


def test_megamoe_dispatch_pull_strategy_tracks_direct_topk_materializer(monkeypatch) -> None:
    fake, _config, token_counts = _stage_dispatch_pull_for_materialization_strategy("direct_topk")
    calls = []

    def direct_topk_materialize(*args) -> None:
        calls.append("direct_topk")
        _fill_direct_materialization_outputs(
            args[12],
            args[13],
            args[14],
            args[15],
            args[17],
            args[18],
            args[19],
        )

    monkeypatch.setattr(
        torch.ops.trtllm,
        "mega_moe_m5_materialize_direct_from_topk",
        direct_topk_materialize,
        raising=False,
    )

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull_with_moe_sort(fake, token_counts)

    assert ok is True
    assert reason is None
    assert calls == ["direct_topk"]
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) == "direct_topk"
    assert MegaMoE.full_fusion_m5_dispatch_materialize_kernel.fget(fake) == "direct_topk"
    tile_idx_to_expert_idx, tile_idx_to_mn_limit, num_non_exiting_tiles = (
        fake._full_fusion_m5_direct_pool_fc_route_layout
    )
    assert tile_idx_to_expert_idx[0].item() == 0
    assert tile_idx_to_expert_idx[1:].tolist() == [-1] * (_config.num_max_pool_blocks - 1)
    assert tile_idx_to_mn_limit[0].item() == 2
    assert tile_idx_to_mn_limit[1:].tolist() == [0] * (_config.num_max_pool_blocks - 1)
    assert num_non_exiting_tiles.tolist() == [1]
    assert (
        fake._full_fusion_m5_direct_pool_fc_route_active_pool_limit == _config.num_max_pool_tokens
    )
    assert fake._full_fusion_m5_active_pool_slots is None
    assert fake._full_fusion_m5_direct_combine_rows is None
    output_mapping = fake._full_fusion_m5_direct_combine_output_mapping
    inactive_combine_row = _config.ep_size * _config.top_k * _config.max_num_tokens_per_rank
    assert output_mapping[:2].tolist() == [0, 1]
    assert output_mapping[2:].tolist() == [inactive_combine_row] * (_config.num_max_pool_tokens - 2)


def test_megamoe_dispatch_pull_strategy_tracks_direct_moe_sort_materializer(monkeypatch) -> None:
    fake, _config, token_counts = _stage_dispatch_pull_for_materialization_strategy(
        "direct_moe_sort"
    )
    calls = []

    def direct_moe_sort_materialize(*args) -> None:
        calls.append("direct_moe_sort")
        _fill_direct_materialization_outputs(
            args[12],
            args[13],
            args[14],
            args[15],
            torch.tensor((0,), dtype=torch.int32),
            args[4],
            args[6],
        )

    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_topk", None, raising=False
    )
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", _fake_moe_sort_for_m5_strategy, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "mega_moe_m5_materialize_direct_from_moe_sort",
        direct_moe_sort_materialize,
        raising=False,
    )

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull_with_moe_sort(fake, token_counts)

    assert ok is True
    assert reason is None
    assert calls == ["direct_moe_sort"]
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) == "direct_moe_sort"
    assert MegaMoE.full_fusion_m5_dispatch_materialize_kernel.fget(fake) == "direct_moe_sort"
    assert fake._full_fusion_m5_active_pool_slots.tolist() == [0, 1]
    assert fake._full_fusion_m5_direct_combine_rows.tolist() == [0, 1]


def test_megamoe_dispatch_pull_strategy_tracks_moe_sort_materializer(monkeypatch) -> None:
    fake, _config, token_counts = _stage_dispatch_pull_for_materialization_strategy("moe_sort")
    calls = []

    def moe_sort_materialize(*args) -> None:
        calls.append("moe_sort")
        token_src_metadata = args[10].reshape(-1, 3)
        token_src_metadata.fill_(-1)
        token_src_metadata[0].copy_(torch.tensor((0, 0, 0), dtype=torch.int32))
        token_src_metadata[1].copy_(torch.tensor((0, 1, 0), dtype=torch.int32))

    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_topk", None, raising=False
    )
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", _fake_moe_sort_for_m5_strategy, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_moe_sort", None, raising=False
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "mega_moe_m5_materialize_from_moe_sort",
        moe_sort_materialize,
        raising=False,
    )

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull_with_moe_sort(fake, token_counts)

    assert ok is True
    assert reason is None
    assert calls == ["moe_sort"]
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) == "moe_sort"
    assert MegaMoE.full_fusion_m5_dispatch_materialize_kernel.fget(fake) == "moe_sort"
    assert fake._full_fusion_m5_active_pool_slots.tolist() == [0, 1]
    assert fake._full_fusion_m5_direct_combine_rows.tolist() == [0, 1]


def test_megamoe_dispatch_pull_rejects_torch_moe_sort_reconstruction_without_debug_gate(
    monkeypatch,
) -> None:
    fake, _config, token_counts = _stage_dispatch_pull_for_materialization_strategy(
        "torch_moe_sort_reconstruct"
    )
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_topk", None, raising=False
    )
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", _fake_moe_sort_for_m5_strategy, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_moe_sort", None, raising=False
    )
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_from_moe_sort", None, raising=False
    )

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull_with_moe_sort(fake, token_counts)

    assert ok is False
    assert reason == (
        "M5 dispatch-pull torch moe_sort reconstruction requires an explicit "
        "M5 debug/materialization gate"
    )
    assert MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake) is None
    assert MegaMoE.full_fusion_m5_dispatch_materialize_kernel.fget(fake) is None


def test_megamoe_dispatch_pull_strategy_tracks_torch_moe_sort_reconstruction_under_debug_gate(
    monkeypatch,
) -> None:
    fake, _config, token_counts = _stage_dispatch_pull_for_materialization_strategy(
        "torch_moe_sort_reconstruct"
    )
    fake._full_fusion_m5_reconstruction_materialize_enabled = True
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_topk", None, raising=False
    )
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", _fake_moe_sort_for_m5_strategy, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_direct_from_moe_sort", None, raising=False
    )
    monkeypatch.setattr(
        torch.ops.trtllm, "mega_moe_m5_materialize_from_moe_sort", None, raising=False
    )

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull_with_moe_sort(fake, token_counts)

    assert ok is True
    assert reason is None
    assert (
        MegaMoE.full_fusion_m5_dispatch_materialize_strategy.fget(fake)
        == "torch_moe_sort_reconstruct"
    )
    assert MegaMoE.full_fusion_m5_dispatch_materialize_kernel.fget(fake) is None
    assert fake._full_fusion_m5_active_pool_slots.tolist() == [0, 1]
    assert fake._full_fusion_m5_direct_combine_rows.tolist() == [0, 4]


def test_megamoe_m5_producer_readiness_collects_peer_token_counts() -> None:
    fake, _ = _fake_megamoe_with_workspace(rank=1)

    epoch, reason = MegaMoE._publish_full_fusion_m5_producer_ready(fake, 2)
    assert reason is None
    assert epoch is not None
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=3)

    token_counts, reason = MegaMoE._collect_full_fusion_m5_ready_token_counts(fake, epoch, (3, 2))

    assert reason is None
    assert token_counts == (3, 2)


def test_megamoe_m5_producer_readiness_rejects_missing_or_stale_peer() -> None:
    fake, _ = _fake_megamoe_with_workspace(rank=1)

    epoch, reason = MegaMoE._publish_full_fusion_m5_producer_ready(fake, 2)
    assert reason is None
    assert epoch is not None

    token_counts, reason = MegaMoE._collect_full_fusion_m5_ready_token_counts(fake, epoch, (3, 2))
    assert token_counts is None
    assert reason == "M5 producer rank 0 is not ready"

    _write_m5_ready_control(fake, rank=0, epoch=epoch + 1, num_tokens=3)
    token_counts, reason = MegaMoE._collect_full_fusion_m5_ready_token_counts(fake, epoch, (3, 2))
    assert token_counts is None
    assert reason == f"M5 producer rank 0 has epoch {epoch + 1}, expected {epoch}"


def test_megamoe_m5_consumer_readiness_waits_for_peer_consumers() -> None:
    fake, _ = _fake_megamoe_with_workspace(rank=1)

    epoch, reason = MegaMoE._publish_full_fusion_m5_producer_ready(fake, 2)
    assert reason is None
    assert epoch is not None
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=3)
    _write_m5_consumer_ready_control(fake, rank=1, epoch=epoch)

    ready, reason = MegaMoE._collect_full_fusion_m5_consumers_ready(fake, epoch)
    assert ready is False
    assert reason == "M5 consumer rank 0 is not ready"

    _write_m5_consumer_ready_control(fake, rank=0, epoch=epoch)
    ready, reason = MegaMoE._collect_full_fusion_m5_consumers_ready(fake, epoch)
    assert ready is True
    assert reason is None


def test_megamoe_single_rank_m5_sync_materializes_after_local_ready() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    x_fp4 = torch.arange(2 * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        2, config.hidden_size // 2
    )
    x_sf = torch.arange(2, dtype=torch.uint8).reshape(2, 1)
    token_selected_experts = torch.tensor(((0,), (1,)), dtype=torch.int64)
    token_final_scales = torch.tensor(((0.5,), (0.75,)), dtype=torch.float32)

    staged, stage_reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )
    materialized, materialize_reason = MegaMoE._sync_full_fusion_m5_producers_and_materialize(
        fake, 2, (2,)
    )

    assert staged is True
    assert stage_reason is None
    assert materialized is True
    assert materialize_reason is None
    control, reason = MegaMoE._full_fusion_m5_control_words(fake)
    assert reason is None
    assert control is not None
    assert control.tolist() == [
        MegaMoE._FULL_FUSION_M5_READY_MAGIC,
        1,
        2,
        MegaMoE._FULL_FUSION_M5_READY_FLAG,
        1,
        MegaMoE._FULL_FUSION_M5_READY_FLAG,
    ]


def test_megamoe_m6_combine_push_materializes_peer_buffers_and_reduces_after_barrier() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    epoch = 7
    num_tokens_per_rank = (1, 2)
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=num_tokens_per_rank[0])
    _write_m5_ready_control(fake, rank=1, epoch=epoch, num_tokens=num_tokens_per_rank[1])
    _zero_all_combine_buffers(fake, config.ep_size)
    _write_runtime_token_src_metadata(
        fake,
        (
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 1),
        ),
    )
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    for pool_slot, base in enumerate((1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0)):
        route_outputs[pool_slot] = torch.tensor(
            _weighted_output(base, config.hidden_size), dtype=torch.bfloat16
        )

    pushed, reason = MegaMoE._materialize_full_fusion_combine_push(
        fake, num_tokens_per_rank, route_outputs
    )
    assert pushed is True
    assert reason is None

    rank0_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=0)
    assert reason is None
    assert rank0_buffer is not None
    rank1_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=1)
    assert reason is None
    assert rank1_buffer is not None
    torch.testing.assert_close(rank0_buffer[0, 0], route_outputs[4])
    torch.testing.assert_close(rank0_buffer[1, 0], route_outputs[5])
    torch.testing.assert_close(rank1_buffer[0, 0], route_outputs[0])
    torch.testing.assert_close(rank1_buffer[1, 0], route_outputs[1])
    torch.testing.assert_close(rank1_buffer[0, 1], route_outputs[2])
    torch.testing.assert_close(rank1_buffer[1, 1], route_outputs[3])

    ready, reason = MegaMoE._publish_full_fusion_m6_combine_ready(fake, epoch)
    assert ready is True
    assert reason is None
    ready, reason = MegaMoE._collect_full_fusion_m6_combine_ready(fake, epoch)
    assert ready is False
    assert reason == "M6 combine rank 0 is not ready"

    _write_m6_ready_control(fake, rank=0, epoch=epoch)
    ready, reason = MegaMoE._collect_full_fusion_m6_combine_ready(fake, epoch)
    assert ready is True
    assert reason is None

    reduced, reason = MegaMoE._reduce_full_fusion_combine_buffer(fake, num_tokens_per_rank)
    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(reduced[0], route_outputs[0].float() + route_outputs[1].float())
    torch.testing.assert_close(reduced[1], route_outputs[2].float() + route_outputs[3].float())


def test_megamoe_single_rank_m6_sync_reduces_preweighted_outputs_without_double_scaling() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    epoch = 3
    route_weights = (0.25, 0.5)
    unweighted_outputs = (
        torch.tensor(_weighted_output(2.0, config.hidden_size), dtype=torch.float32),
        torch.tensor(_weighted_output(20.0, config.hidden_size), dtype=torch.float32),
    )
    preweighted_outputs = (
        unweighted_outputs[0] * route_weights[0],
        unweighted_outputs[1] * route_weights[1],
    )
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[0] = preweighted_outputs[0].to(dtype=torch.bfloat16)
    route_outputs[1] = preweighted_outputs[1].to(dtype=torch.bfloat16)
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=1)
    _zero_all_combine_buffers(fake, config.ep_size)
    _write_runtime_token_src_metadata(fake, ((0, 0, 0), (0, 0, 1)))

    reduced, reason = MegaMoE._sync_full_fusion_m6_combine_push_and_reduce(
        fake, (1,), route_outputs, epoch
    )

    assert reason is None
    assert reduced is not None
    expected = route_outputs[0].float() + route_outputs[1].float()
    double_scaled = (
        unweighted_outputs[0] * route_weights[0] * route_weights[0]
        + unweighted_outputs[1] * route_weights[1] * route_weights[1]
    )
    torch.testing.assert_close(reduced[0], expected)
    assert not torch.allclose(reduced[0], double_scaled)
    control, reason = MegaMoE._full_fusion_m6_control_words(fake)
    assert reason is None
    assert control is not None
    assert control[8:10].tolist() == [epoch, MegaMoE._FULL_FUSION_M6_READY_FLAG]
    assert fake._full_fusion_combine_push_fallback_reason is None


def test_megamoe_output_path_materializes_topk_route_outputs_for_m6() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    x_fp4 = torch.arange(2 * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        2, config.hidden_size // 2
    )
    x_sf = torch.arange(2, dtype=torch.uint8).reshape(2, 1)
    token_selected_experts = torch.tensor(((0, 1), (1, 0)), dtype=torch.int64)
    token_final_scales = torch.tensor(((0.25, 0.5), (0.75, 1.25)), dtype=torch.float32)

    staged, stage_reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )
    materialized, pull_reason = MegaMoE._sync_full_fusion_m5_producers_and_materialize(
        fake, 2, (2,)
    )
    assert staged is True
    assert stage_reason is None
    assert materialized is True
    assert pull_reason is None

    topk_route_outputs = torch.zeros((config.top_k, 2, config.hidden_size), dtype=torch.bfloat16)
    topk_route_outputs[0, 0] = torch.tensor(
        _weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16
    )
    topk_route_outputs[0, 1] = torch.tensor(
        _weighted_output(10.0, config.hidden_size), dtype=torch.bfloat16
    )
    topk_route_outputs[1, 0] = torch.tensor(
        _weighted_output(100.0, config.hidden_size), dtype=torch.bfloat16
    )
    topk_route_outputs[1, 1] = torch.tensor(
        _weighted_output(1000.0, config.hidden_size), dtype=torch.bfloat16
    )

    weighted_route_outputs, reason = (
        MegaMoE._materialize_full_fusion_weighted_route_outputs_from_topk_route_outputs(
            fake, (2,), topk_route_outputs
        )
    )

    assert reason is None
    assert weighted_route_outputs is not None
    torch.testing.assert_close(weighted_route_outputs[0], topk_route_outputs[0, 0])
    torch.testing.assert_close(weighted_route_outputs[1], topk_route_outputs[1, 1])
    torch.testing.assert_close(weighted_route_outputs[4], topk_route_outputs[1, 0])
    torch.testing.assert_close(weighted_route_outputs[5], topk_route_outputs[0, 1])
    assert torch.count_nonzero(weighted_route_outputs[2:4]).item() == 0
    assert torch.count_nonzero(weighted_route_outputs[6:]).item() == 0


def test_megamoe_output_path_materializes_ranked_topk_route_outputs_for_multi_rank_m6() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    rank0_tokens = 2
    rank1_tokens = 2
    rank0_x = (
        torch.arange(rank0_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 10
    ).reshape(rank0_tokens, config.hidden_size // 2)
    rank1_x = (
        torch.arange(rank1_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 100
    ).reshape(rank1_tokens, config.hidden_size // 2)
    rank0_sf = (torch.arange(rank0_tokens, dtype=torch.uint8) + 1).reshape(rank0_tokens, 1)
    rank1_sf = (torch.arange(rank1_tokens, dtype=torch.uint8) + 50).reshape(rank1_tokens, 1)
    rank0_experts = torch.tensor(((2, 3), (0, 1)), dtype=torch.int64)
    rank1_experts = torch.tensor(((3, 2), (2, 3)), dtype=torch.int64)
    rank0_weights = torch.tensor(((0.2, 0.3), (9.0, 8.0)), dtype=torch.float32)
    rank1_weights = torch.tensor(((0.4, 0.5), (0.6, 0.7)), dtype=torch.float32)

    _copy_tensor_to_workspace_rank(fake, 0, "x_buf", rank0_x)
    _copy_tensor_to_workspace_rank(fake, 0, "x_sf_buf", rank0_sf)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_idx_buf", rank0_experts)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_weights_buf", rank0_weights)
    _copy_tensor_to_workspace_rank(fake, 1, "x_buf", rank1_x)
    _copy_tensor_to_workspace_rank(fake, 1, "x_sf_buf", rank1_sf)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_idx_buf", rank1_experts)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_weights_buf", rank1_weights)

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (rank0_tokens, rank1_tokens))
    assert ok is True
    assert reason is None

    ranked_topk_route_outputs = torch.zeros(
        (config.ep_size, config.top_k, config.max_num_tokens_per_rank, config.hidden_size),
        dtype=torch.bfloat16,
    )
    active_routes = (
        (0, 0, 0, 1.0),
        (1, 1, 0, 10.0),
        (1, 0, 1, 100.0),
        (0, 1, 0, 1000.0),
        (1, 0, 0, 10000.0),
        (1, 1, 1, 20000.0),
    )
    for src_rank, topk_idx, token_idx, base in active_routes:
        ranked_topk_route_outputs[src_rank, topk_idx, token_idx] = torch.tensor(
            _weighted_output(base, config.hidden_size), dtype=torch.bfloat16
        )

    weighted_route_outputs, reason = (
        MegaMoE._materialize_full_fusion_weighted_route_outputs_from_ranked_topk_route_outputs(
            fake, (rank0_tokens, rank1_tokens), ranked_topk_route_outputs
        )
    )

    assert reason is None
    assert weighted_route_outputs is not None
    torch.testing.assert_close(weighted_route_outputs[0], ranked_topk_route_outputs[0, 0, 0])
    torch.testing.assert_close(weighted_route_outputs[1], ranked_topk_route_outputs[1, 1, 0])
    torch.testing.assert_close(weighted_route_outputs[2], ranked_topk_route_outputs[1, 0, 1])
    torch.testing.assert_close(weighted_route_outputs[4], ranked_topk_route_outputs[0, 1, 0])
    torch.testing.assert_close(weighted_route_outputs[5], ranked_topk_route_outputs[1, 0, 0])
    torch.testing.assert_close(weighted_route_outputs[6], ranked_topk_route_outputs[1, 1, 1])
    assert torch.count_nonzero(weighted_route_outputs[3]).item() == 0
    assert torch.count_nonzero(weighted_route_outputs[7:]).item() == 0

    epoch = 13
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=rank0_tokens)
    _write_m5_ready_control(fake, rank=1, epoch=epoch, num_tokens=rank1_tokens)
    _write_m6_ready_control(fake, rank=0, epoch=epoch)
    _zero_all_combine_buffers(fake, config.ep_size)

    reduced, reason = MegaMoE._sync_full_fusion_m6_combine_push_and_reduce(
        fake, (rank0_tokens, rank1_tokens), weighted_route_outputs, epoch
    )

    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(
        reduced[0], weighted_route_outputs[5].float() + weighted_route_outputs[1].float()
    )
    torch.testing.assert_close(
        reduced[1], weighted_route_outputs[2].float() + weighted_route_outputs[6].float()
    )
    rank0_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=0)
    assert reason is None
    assert rank0_buffer is not None
    torch.testing.assert_close(rank0_buffer[0, 0], weighted_route_outputs[0])
    torch.testing.assert_close(rank0_buffer[1, 0], weighted_route_outputs[4])


def test_megamoe_route_output_producer_stages_ranked_outputs_for_multi_rank_m6() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    rank0_tokens = 2
    rank1_tokens = 2
    rank0_x = (
        torch.arange(rank0_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 10
    ).reshape(rank0_tokens, config.hidden_size // 2)
    rank1_x = (
        torch.arange(rank1_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 100
    ).reshape(rank1_tokens, config.hidden_size // 2)
    rank0_sf = (torch.arange(rank0_tokens, dtype=torch.uint8) + 1).reshape(rank0_tokens, 1)
    rank1_sf = (torch.arange(rank1_tokens, dtype=torch.uint8) + 50).reshape(rank1_tokens, 1)
    rank0_experts = torch.tensor(((2, 3), (0, 1)), dtype=torch.int64)
    rank1_experts = torch.tensor(((3, 2), (2, 3)), dtype=torch.int64)
    rank0_weights = torch.tensor(((0.2, 0.3), (9.0, 8.0)), dtype=torch.float32)
    rank1_weights = torch.tensor(((0.4, 0.5), (0.6, 0.7)), dtype=torch.float32)

    _copy_tensor_to_workspace_rank(fake, 0, "x_buf", rank0_x)
    _copy_tensor_to_workspace_rank(fake, 0, "x_sf_buf", rank0_sf)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_idx_buf", rank0_experts)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_weights_buf", rank0_weights)
    _copy_tensor_to_workspace_rank(fake, 1, "x_buf", rank1_x)
    _copy_tensor_to_workspace_rank(fake, 1, "x_sf_buf", rank1_sf)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_idx_buf", rank1_experts)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_weights_buf", rank1_weights)

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (rank0_tokens, rank1_tokens))
    assert ok is True
    assert reason is None

    epoch = 17
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=rank0_tokens)
    _write_m5_ready_control(fake, rank=1, epoch=epoch, num_tokens=rank1_tokens)

    rank0_route_outputs = torch.zeros(
        (config.top_k, config.max_num_tokens_per_rank, config.hidden_size), dtype=torch.bfloat16
    )
    rank1_route_outputs = torch.zeros_like(rank0_route_outputs)
    active_routes = (
        (0, 0, 0, 1.0),
        (1, 1, 0, 10.0),
        (1, 0, 1, 100.0),
        (0, 1, 0, 1000.0),
        (1, 0, 0, 10000.0),
        (1, 1, 1, 20000.0),
    )
    for src_rank, topk_idx, token_idx, base in active_routes:
        target = rank0_route_outputs if src_rank == 0 else rank1_route_outputs
        target[topk_idx, token_idx] = torch.tensor(
            _weighted_output(base, config.hidden_size), dtype=torch.bfloat16
        )

    _copy_tensor_to_workspace_rank(fake, 0, "ranked_route_output_buf", rank0_route_outputs)
    _write_route_output_ready_control(fake, rank=0, epoch=epoch)
    staged, reason = MegaMoE._stage_full_fusion_ranked_route_outputs(
        fake, rank1_tokens, rank1_route_outputs, epoch
    )
    assert staged is True
    assert reason is None

    ready, reason = MegaMoE._collect_full_fusion_route_output_producers_ready(fake, epoch)
    assert ready is True
    assert reason is None

    weighted_route_outputs, reason = (
        MegaMoE._sync_full_fusion_route_output_producers_and_materialize(
            fake, (rank0_tokens, rank1_tokens), epoch
        )
    )
    assert reason is None
    assert weighted_route_outputs is not None
    torch.testing.assert_close(weighted_route_outputs[0], rank0_route_outputs[0, 0])
    torch.testing.assert_close(weighted_route_outputs[1], rank1_route_outputs[1, 0])
    torch.testing.assert_close(weighted_route_outputs[2], rank1_route_outputs[0, 1])
    torch.testing.assert_close(weighted_route_outputs[4], rank0_route_outputs[1, 0])
    torch.testing.assert_close(weighted_route_outputs[5], rank1_route_outputs[0, 0])
    torch.testing.assert_close(weighted_route_outputs[6], rank1_route_outputs[1, 1])
    assert torch.count_nonzero(weighted_route_outputs[3]).item() == 0
    assert torch.count_nonzero(weighted_route_outputs[7:]).item() == 0

    _write_m6_ready_control(fake, rank=0, epoch=epoch)
    _zero_all_combine_buffers(fake, config.ep_size)
    reduced, reason = MegaMoE._sync_full_fusion_m6_combine_push_and_reduce(
        fake, (rank0_tokens, rank1_tokens), weighted_route_outputs, epoch
    )
    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(
        reduced[0], weighted_route_outputs[5].float() + weighted_route_outputs[1].float()
    )
    torch.testing.assert_close(
        reduced[1], weighted_route_outputs[2].float() + weighted_route_outputs[6].float()
    )


def test_megamoe_builds_m5_pool_route_inputs_for_fc2_route_producer() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    rank0_tokens = 2
    rank1_tokens = 2
    rank0_x = (
        torch.arange(rank0_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 10
    ).reshape(rank0_tokens, config.hidden_size // 2)
    rank1_x = (
        torch.arange(rank1_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 100
    ).reshape(rank1_tokens, config.hidden_size // 2)
    rank0_sf = (torch.arange(rank0_tokens, dtype=torch.uint8) + 1).reshape(rank0_tokens, 1)
    rank1_sf = (torch.arange(rank1_tokens, dtype=torch.uint8) + 50).reshape(rank1_tokens, 1)
    rank0_experts = torch.tensor(((2, 3), (0, 1)), dtype=torch.int64)
    rank1_experts = torch.tensor(((3, 2), (2, 3)), dtype=torch.int64)
    rank0_weights = torch.tensor(((0.2, 0.3), (9.0, 8.0)), dtype=torch.float32)
    rank1_weights = torch.tensor(((0.4, 0.5), (0.6, 0.7)), dtype=torch.float32)

    _copy_tensor_to_workspace_rank(fake, 0, "x_buf", rank0_x)
    _copy_tensor_to_workspace_rank(fake, 0, "x_sf_buf", rank0_sf)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_idx_buf", rank0_experts)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_weights_buf", rank0_weights)
    _copy_tensor_to_workspace_rank(fake, 1, "x_buf", rank1_x)
    _copy_tensor_to_workspace_rank(fake, 1, "x_sf_buf", rank1_sf)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_idx_buf", rank1_experts)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_weights_buf", rank1_weights)

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (rank0_tokens, rank1_tokens))
    assert ok is True
    assert reason is None

    route_inputs, reason = MegaMoE._build_full_fusion_m5_pool_route_inputs(fake)

    assert reason is None
    assert route_inputs is not None
    pool_x, pool_x_sf, pool_selected_experts, pool_final_scales, active_pool_limit = route_inputs
    assert active_pool_limit == 8
    assert pool_selected_experts.dtype == torch.int32
    assert pool_selected_experts[:, 0].tolist() == [2, 2, 2, 2, 3, 3, 3, 3]
    torch.testing.assert_close(
        pool_final_scales[:, 0],
        torch.tensor((0.2, 0.5, 0.6, 0.0, 0.3, 0.4, 0.7, 0.0), dtype=torch.float32),
    )

    assert torch.equal(pool_x[0], rank0_x[0])
    assert torch.equal(pool_x[1], rank1_x[0])
    assert torch.equal(pool_x[2], rank1_x[1])
    assert torch.all(pool_x[3] == 0)
    assert torch.equal(pool_x[4], rank0_x[0])
    assert torch.equal(pool_x[5], rank1_x[0])
    assert torch.equal(pool_x[6], rank1_x[1])
    assert torch.all(pool_x[7] == 0)

    assert torch.equal(pool_x_sf[0], rank0_sf[0])
    assert torch.equal(pool_x_sf[1], rank1_sf[0])
    assert torch.equal(pool_x_sf[2], rank1_sf[1])
    assert torch.all(pool_x_sf[3] == 0)
    assert torch.equal(pool_x_sf[4], rank0_sf[0])
    assert torch.equal(pool_x_sf[5], rank1_sf[0])
    assert torch.equal(pool_x_sf[6], rank1_sf[1])
    assert torch.all(pool_x_sf[7] == 0)


def test_megamoe_reuses_cached_m5_pool_route_inputs_without_expert_counts() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
    )
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].fill_(-1)
    l1_acts_pool = MegaMoE._full_fusion_local_workspace_region(fake, "l1_acts_pool")
    assert l1_acts_pool is not None
    l1_acts_pool = l1_acts_pool.reshape(config.num_max_pool_tokens, config.hidden_size // 2)
    l1_acts_pool.copy_(
        torch.arange(l1_acts_pool.numel(), dtype=torch.uint8).reshape_as(l1_acts_pool)
    )
    l1_acts_sf_pool = MegaMoE._full_fusion_local_workspace_region(fake, "l1_acts_sf_pool")
    assert l1_acts_sf_pool is not None
    l1_acts_sf_pool = l1_acts_sf_pool.reshape(
        config.num_padded_sf_pool_tokens, config.hidden_size // config.scaling_vector_size
    )
    l1_acts_sf_pool.copy_(
        (torch.arange(l1_acts_sf_pool.numel(), dtype=torch.uint8) + 10).reshape_as(l1_acts_sf_pool)
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:8].copy_(
        torch.tensor((0.2, 0.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0), dtype=torch.float32)
    )

    route_inputs, reason = MegaMoE._build_full_fusion_m5_pool_route_inputs(
        fake, include_selected_experts=False, full_capacity=True
    )

    assert reason is None
    assert route_inputs is not None
    pool_x, pool_x_sf, pool_selected_experts, pool_final_scales, active_pool_limit = route_inputs
    assert active_pool_limit == 8
    assert pool_x.shape[0] == config.num_max_pool_tokens
    assert pool_x_sf.shape[0] == config.num_max_pool_tokens
    assert pool_selected_experts is None
    torch.testing.assert_close(
        pool_final_scales[:8, 0],
        torch.tensor((0.2, 0.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0), dtype=torch.float32),
    )
    assert torch.count_nonzero(pool_final_scales[8:]).item() == 0
    assert torch.equal(pool_x[:8], l1_acts_pool[:8])
    sf_block_tokens = ((config.tile_size + 127) // 128) * 128
    sf_indices = (torch.arange(8) // config.tile_size) * sf_block_tokens + (
        torch.arange(8) % config.tile_size
    )
    assert torch.equal(pool_x_sf[:8], l1_acts_sf_pool.index_select(0, sf_indices))


def test_megamoe_cached_m5_pool_route_inputs_can_fill_selected_experts() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 4, 5, 6), dtype=torch.int64)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
    )
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].fill_(-1)
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:8].copy_(
        torch.tensor((0.2, 0.5, 0.0, 0.0, 0.3, 0.4, 0.7, 0.0), dtype=torch.float32)
    )

    route_inputs, reason = MegaMoE._build_full_fusion_m5_pool_route_inputs(fake)

    assert reason is None
    assert route_inputs is not None
    _, _, pool_selected_experts, pool_final_scales, active_pool_limit = route_inputs
    assert active_pool_limit == 8
    assert pool_selected_experts is not None
    assert pool_selected_experts[:, 0].tolist() == [2, 2, 2, 2, 3, 3, 3, 3]
    torch.testing.assert_close(
        pool_final_scales[:, 0],
        torch.tensor((0.2, 0.5, 0.0, 0.0, 0.3, 0.4, 0.7, 0.0), dtype=torch.float32),
    )


def test_megamoe_cached_m5_pool_route_descriptor_validates_active_limit() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 8), dtype=torch.int64)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
    )

    descriptor = MegaMoE._cached_full_fusion_m5_pool_route_descriptor(fake, config)

    assert descriptor is None

    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 4), dtype=torch.int64)
    descriptor = MegaMoE._cached_full_fusion_m5_pool_route_descriptor(fake, config)

    assert descriptor is not None
    active_pool_slots, route_layout, active_pool_limit = descriptor
    assert active_pool_slots.tolist() == [0, 1, 4]
    assert route_layout is not None
    assert active_pool_limit == 8


def test_megamoe_cached_m5_pool_route_inputs_fallback_on_invalid_descriptor() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 8), dtype=torch.int64)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
    )
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].fill_(-1)

    route_inputs, reason = MegaMoE._build_full_fusion_m5_pool_route_inputs(
        fake, include_selected_experts=False
    )

    assert route_inputs is None
    assert reason == "expert 0 has negative route count -1"


def test_megamoe_builds_direct_m5_pool_fc_route_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((3, 5), dtype=torch.int64)
    )

    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=12
    )

    assert reason is None
    assert route_metadata is not None
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    ) = route_metadata
    assert tile_idx_to_expert_idx.dtype == torch.int32
    assert tile_idx_to_mn_limit.dtype == torch.int32
    assert permuted_idx_to_expanded_idx.dtype == torch.int32
    assert num_non_exiting_tiles.dtype == torch.int32
    assert tile_idx_to_expert_idx.tolist() == [0, 1, 1]
    assert tile_idx_to_mn_limit.tolist() == [4, 8, 12]
    assert permuted_idx_to_expanded_idx.tolist() == list(range(12))
    assert num_non_exiting_tiles.tolist() == [3]


def test_megamoe_reuses_cached_direct_m5_pool_fc_route_layout() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].fill_(-1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor(
        (0, 1, 2, 4, 5, 6, 7, 8), dtype=torch.int64
    )
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1, 1), dtype=torch.int32),
        torch.tensor((4, 8, 12), dtype=torch.int32),
        torch.tensor((3,), dtype=torch.int32),
    )

    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=12
    )

    assert reason is None
    assert route_metadata is not None
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    ) = route_metadata
    assert tile_idx_to_expert_idx.tolist() == [0, 1, 1]
    assert tile_idx_to_mn_limit.tolist() == [4, 8, 12]
    assert permuted_idx_to_expanded_idx.tolist() == list(range(12))
    assert num_non_exiting_tiles.tolist() == [3]


def test_megamoe_publishes_validated_direct_m5_pool_fc_route_layout() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m6_direct_combine_layout_output_enabled = True
    active_pool_slots = torch.tensor((0, 1, 4), dtype=torch.int64)
    active_combine_rows = torch.tensor((4, 10, 3), dtype=torch.int64)
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.3, 0.0, 0.0, 0.4)))

    published, reason = MegaMoE._publish_full_fusion_m5_direct_pool_fc_route_layout(
        fake,
        config,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
        active_pool_slots=active_pool_slots,
        active_combine_rows=active_combine_rows,
    )

    assert published is True
    assert reason is None
    cached_route_layout = MegaMoE._cached_full_fusion_m5_direct_pool_fc_route_layout(fake, config)
    assert cached_route_layout is not None
    tile_idx_to_expert_idx, tile_idx_to_mn_limit, num_non_exiting_tiles, active_pool_limit = (
        cached_route_layout
    )
    assert tile_idx_to_expert_idx.tolist() == [0, 1]
    assert tile_idx_to_mn_limit.tolist() == [4, 8]
    assert num_non_exiting_tiles.tolist() == [2]
    assert active_pool_limit == 8
    cached_active_pool_slots = MegaMoE._cached_full_fusion_m5_active_pool_slots(fake, config)
    assert cached_active_pool_slots is not None
    assert cached_active_pool_slots.tolist() == [0, 1, 4]
    cached_combine_rows = MegaMoE._cached_full_fusion_m5_direct_combine_rows(
        fake, active_pool_slots, combine_layout_rows=16
    )
    assert cached_combine_rows is active_combine_rows
    assert cached_combine_rows.tolist() == [4, 10, 3]
    cached_output_mapping = MegaMoE._cached_full_fusion_m5_direct_combine_output_mapping(
        fake,
        active_pool_slots,
        active_pool_limit=active_pool_limit,
        combine_layout_rows=16,
        inactive_row=16,
        expected_device=active_pool_slots.device,
    )
    assert cached_output_mapping is fake._full_fusion_m5_direct_combine_output_mapping
    assert cached_output_mapping is not None
    assert cached_output_mapping.tolist() == [4, 10, 16, 16, 3, 16, 16, 16]
    cached_output_scales = MegaMoE._cached_full_fusion_m5_direct_combine_output_scales(
        fake,
        num_max_pool_tokens=config.num_max_pool_tokens,
        expected_device=active_pool_slots.device,
    )
    assert cached_output_scales is fake._full_fusion_m5_direct_combine_output_scales
    assert cached_output_scales is not None
    torch.testing.assert_close(
        cached_output_scales[torch.tensor((4, 10, 3), dtype=torch.int64), 0],
        torch.tensor((0.2, 0.3, 0.4)),
    )
    assert torch.count_nonzero(cached_output_scales).item() == 3


def test_megamoe_publishes_prebuilt_direct_m5_output_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m6_direct_combine_layout_output_enabled = True
    active_pool_slots = torch.tensor((4, 0), dtype=torch.int64)
    active_combine_rows = torch.tensor((11, 4), dtype=torch.int64)
    prebuilt_mapping = torch.full((8,), 16, dtype=torch.int32)
    prebuilt_mapping.index_copy_(0, active_pool_slots, active_combine_rows.to(dtype=torch.int32))
    prebuilt_scales = torch.zeros((config.num_max_pool_tokens, 1), dtype=torch.float32)
    prebuilt_scales[11, 0] = 0.7
    prebuilt_scales[4, 0] = 0.2
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].fill_(9.0)

    published, reason = MegaMoE._publish_full_fusion_m5_direct_pool_fc_route_layout(
        fake,
        config,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
        active_pool_slots=active_pool_slots,
        active_combine_rows=active_combine_rows,
        active_combine_output_mapping=prebuilt_mapping,
        active_combine_output_scales=prebuilt_scales,
    )

    assert published is True
    assert reason is None
    descriptor = MegaMoE._cached_full_fusion_m5_direct_materialization_descriptor(
        fake, config, output_active_pool_limit=8, expected_device=torch.device("cpu")
    )
    assert descriptor is not None
    assert descriptor.active_pool_slots is active_pool_slots
    assert descriptor.active_combine_rows is active_combine_rows
    assert descriptor.output_mapping is prebuilt_mapping
    assert descriptor.output_scales is prebuilt_scales
    torch.testing.assert_close(descriptor.output_scales[11, 0], torch.tensor(0.7))
    torch.testing.assert_close(descriptor.output_scales[4, 0], torch.tensor(0.2))


def test_megamoe_rejects_invalid_published_direct_m5_pool_fc_route_layout() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0,), dtype=torch.int32),
        torch.tensor((4,), dtype=torch.int32),
        torch.tensor((1,), dtype=torch.int32),
    )
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0,), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4,), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_output_mapping = torch.tensor((4,), dtype=torch.int32)
    fake._full_fusion_m5_direct_combine_output_scales = torch.ones((16, 1), dtype=torch.float32)

    published, reason = MegaMoE._publish_full_fusion_m5_direct_pool_fc_route_layout(
        fake,
        config,
        torch.tensor((9, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
        active_pool_slots=torch.tensor((0, 1), dtype=torch.int64),
    )

    assert published is False
    assert reason == "invalid M5 direct pool FC route layout cache"
    assert fake._full_fusion_m5_direct_pool_fc_route_layout is None
    assert fake._full_fusion_m5_active_pool_slots is None
    assert fake._full_fusion_m5_direct_combine_rows is None
    assert fake._full_fusion_m5_direct_combine_output_mapping is None
    assert fake._full_fusion_m5_direct_combine_output_scales is None


def test_megamoe_direct_m5_pool_fc_route_metadata_fallback_on_invalid_descriptor() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 12), dtype=torch.int64)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1, 1), dtype=torch.int32),
        torch.tensor((4, 8, 12), dtype=torch.int32),
        torch.tensor((3,), dtype=torch.int32),
    )
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].fill_(-1)

    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=12
    )

    assert route_metadata is None
    assert reason == "expert 0 has negative route count -1"


def test_megamoe_cached_direct_m5_pool_fc_route_layout_mismatch_falls_back() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((9, 9), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
    )
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((3, 5), dtype=torch.int64)
    )

    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=12
    )

    assert reason is None
    assert route_metadata is not None
    tile_idx_to_expert_idx, tile_idx_to_mn_limit, _, num_non_exiting_tiles = route_metadata
    assert tile_idx_to_expert_idx.tolist() == [0, 1, 1]
    assert tile_idx_to_mn_limit.tolist() == [4, 8, 12]
    assert num_non_exiting_tiles.tolist() == [3]


def test_megamoe_invalid_cached_direct_m5_pool_fc_route_layout_same_limit_falls_back() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((9, 1, 1), dtype=torch.int32),
        torch.tensor((4, 8, 12), dtype=torch.int32),
        torch.tensor((3,), dtype=torch.int32),
    )
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((3, 5), dtype=torch.int64)
    )

    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=12
    )

    assert reason is None
    assert route_metadata is not None
    tile_idx_to_expert_idx, tile_idx_to_mn_limit, _, num_non_exiting_tiles = route_metadata
    assert tile_idx_to_expert_idx.tolist() == [0, 1, 1]
    assert tile_idx_to_mn_limit.tolist() == [4, 8, 12]
    assert num_non_exiting_tiles.tolist() == [3]


def test_megamoe_direct_m5_pool_fc_route_bypasses_pool_moe_sort() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_direct_pool_fc_route_enabled = True
    rank0_tokens = 2
    rank1_tokens = 2
    rank0_x = (
        torch.arange(rank0_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 10
    ).reshape(rank0_tokens, config.hidden_size // 2)
    rank1_x = (
        torch.arange(rank1_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 100
    ).reshape(rank1_tokens, config.hidden_size // 2)
    rank0_sf = (torch.arange(rank0_tokens, dtype=torch.uint8) + 1).reshape(rank0_tokens, 1)
    rank1_sf = (torch.arange(rank1_tokens, dtype=torch.uint8) + 50).reshape(rank1_tokens, 1)
    rank0_experts = torch.tensor(((2, 3), (0, 1)), dtype=torch.int64)
    rank1_experts = torch.tensor(((3, 2), (2, 3)), dtype=torch.int64)
    rank0_weights = torch.tensor(((0.2, 0.3), (9.0, 8.0)), dtype=torch.float32)
    rank1_weights = torch.tensor(((0.4, 0.5), (0.6, 0.7)), dtype=torch.float32)

    _copy_tensor_to_workspace_rank(fake, 0, "x_buf", rank0_x)
    _copy_tensor_to_workspace_rank(fake, 0, "x_sf_buf", rank0_sf)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_idx_buf", rank0_experts)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_weights_buf", rank0_weights)
    _copy_tensor_to_workspace_rank(fake, 1, "x_buf", rank1_x)
    _copy_tensor_to_workspace_rank(fake, 1, "x_sf_buf", rank1_sf)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_idx_buf", rank1_experts)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_weights_buf", rank1_weights)

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (rank0_tokens, rank1_tokens))
    assert ok is True
    assert reason is None

    captured = {}

    def run_fused_fc1_fc2_combine(
        *,
        x_fp4,
        x_sf,
        token_final_scales,
        output,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        effective_top_k,
        output_permuted_idx_to_expanded_idx=None,
        direct_combine_buffer_output_config=None,
        monolithic_output=None,
        monolithic_control=None,
        monolithic_local_rank=0,
        monolithic_local_tokens=0,
    ):
        assert output_permuted_idx_to_expanded_idx is None
        assert direct_combine_buffer_output_config is None
        captured["x_shape"] = tuple(x_fp4.shape)
        captured["x_sf_shape"] = tuple(x_sf.shape)
        captured["token_final_scales"] = token_final_scales[:, 0].clone()
        captured["tile_idx_to_expert_idx"] = tile_idx_to_expert_idx.clone()
        captured["tile_idx_to_mn_limit"] = tile_idx_to_mn_limit.clone()
        captured["permuted_idx_to_expanded_idx"] = permuted_idx_to_expanded_idx.clone()
        captured["num_non_exiting_tiles"] = num_non_exiting_tiles.clone()
        captured["effective_top_k"] = effective_top_k
        output.copy_(
            torch.arange(output.numel(), dtype=torch.float32)
            .to(dtype=output.dtype)
            .reshape_as(output)
        )

    fake._run_fused_fc1_fc2_combine = run_fused_fc1_fc2_combine

    weighted_route_outputs, reason = MegaMoE._run_full_fusion_m5_pool_route_outputs(fake)

    assert reason is None
    assert weighted_route_outputs is not None
    assert captured["x_shape"] == (8, config.hidden_size // 2)
    assert captured["x_sf_shape"] == (8, config.hidden_size // config.scaling_vector_size)
    torch.testing.assert_close(
        captured["token_final_scales"],
        torch.tensor((0.2, 0.5, 0.6, 0.0, 0.3, 0.4, 0.7, 0.0), dtype=torch.float32),
    )
    assert captured["tile_idx_to_expert_idx"].tolist() == [0, 1]
    assert captured["tile_idx_to_mn_limit"].tolist() == [4, 8]
    assert captured["permuted_idx_to_expanded_idx"].tolist() == list(range(8))
    assert captured["num_non_exiting_tiles"].tolist() == [2]
    assert captured["effective_top_k"] == 1
    torch.testing.assert_close(
        weighted_route_outputs[:8],
        torch.arange(8 * config.hidden_size, dtype=torch.float32)
        .to(dtype=torch.bfloat16)
        .reshape(8, config.hidden_size),
    )
    assert torch.count_nonzero(weighted_route_outputs[8:]).item() == 0


def test_megamoe_m5_pool_route_outputs_falls_back_from_direct_combine_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=4,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    active_pool_limit = 3
    calls = []

    def build_route_inputs(*, include_selected_experts, full_capacity):
        calls.append(("route_inputs", include_selected_experts, full_capacity))
        row_count = config.num_max_pool_tokens if full_capacity else active_pool_limit
        pool_x = torch.zeros((row_count, config.hidden_size // 2), dtype=torch.uint8)
        pool_x_sf = torch.zeros(
            (row_count, config.hidden_size // config.scaling_vector_size), dtype=torch.uint8
        )
        pool_final_scales = torch.ones((row_count, 1), dtype=torch.float32)
        return (pool_x, pool_x_sf, None, pool_final_scales, active_pool_limit), None

    def build_route_metadata(active_limit, *, expected_device=None):
        calls.append(("route_metadata", active_limit, expected_device))
        return (
            (
                torch.tensor((0,), dtype=torch.int32),
                torch.tensor((active_limit,), dtype=torch.int32),
                torch.arange(active_limit, dtype=torch.int32),
                torch.tensor((1,), dtype=torch.int32),
            ),
            None,
        )

    def build_combine_output_metadata(active_limit, *, expected_device=None):
        calls.append(("combine_metadata", active_limit, expected_device))
        return None, "direct combine metadata unavailable"

    def combine_buffer_output_view():
        raise AssertionError("direct combine-buffer view should not be requested after fallback")

    def run_fused_fc1_fc2_combine(
        *,
        x_fp4,
        x_sf,
        token_final_scales,
        output,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        effective_top_k,
        output_permuted_idx_to_expanded_idx=None,
        direct_combine_buffer_output_config=None,
        monolithic_output=None,
        monolithic_control=None,
        monolithic_local_rank=0,
        monolithic_local_tokens=0,
    ):
        calls.append(("fused", tuple(x_fp4.shape), tuple(output.shape)))
        assert output_permuted_idx_to_expanded_idx is None
        assert direct_combine_buffer_output_config is None
        assert tuple(output.shape) == (active_pool_limit, config.hidden_size)
        output.copy_(
            torch.arange(output.numel(), dtype=torch.float32)
            .to(dtype=output.dtype)
            .reshape_as(output)
        )

    fake = SimpleNamespace(
        _full_fusion_runtime_gate=SimpleNamespace(
            workspace_descriptor=SimpleNamespace(layout=SimpleNamespace(config=config))
        ),
        _full_fusion_workspace=object(),
        _full_fusion_m5_direct_pool_fc_route_enabled=True,
        _full_fusion_m6_direct_combine_layout_output_enabled=True,
        _full_fusion_m6_direct_combine_buffer_output_enabled=True,
        _full_fusion_m6_route_output_layout="stale",
        _full_fusion_m6_route_output_layout_rows=99,
        _full_fusion_m6_route_output_active_rows=torch.tensor((1,), dtype=torch.int64),
        _build_full_fusion_m5_pool_route_inputs=build_route_inputs,
        _build_full_fusion_m5_direct_pool_fc_route_metadata=build_route_metadata,
        _build_full_fusion_m5_direct_combine_output_metadata=build_combine_output_metadata,
        _full_fusion_combine_buffer_output_view=combine_buffer_output_view,
        _run_fused_fc1_fc2_combine=run_fused_fc1_fc2_combine,
    )

    _attach_disabled_full_fusion_profile_helpers(fake)

    weighted_route_outputs, reason = MegaMoE._run_full_fusion_m5_pool_route_outputs(fake)

    assert reason is None
    assert weighted_route_outputs is not None
    assert fake._full_fusion_m6_route_output_layout == "pool"
    assert fake._full_fusion_m6_route_output_layout_rows is None
    assert fake._full_fusion_m6_route_output_active_rows is None
    assert calls == [
        ("route_inputs", False, True),
        ("combine_metadata", active_pool_limit, torch.device("cpu")),
        ("route_inputs", False, False),
        ("route_metadata", active_pool_limit, torch.device("cpu")),
        (
            "fused",
            (active_pool_limit, config.hidden_size // 2),
            (active_pool_limit, config.hidden_size),
        ),
    ]
    torch.testing.assert_close(
        weighted_route_outputs[:active_pool_limit],
        torch.arange(active_pool_limit * config.hidden_size, dtype=torch.float32)
        .to(dtype=torch.bfloat16)
        .reshape(active_pool_limit, config.hidden_size),
    )
    assert torch.count_nonzero(weighted_route_outputs[active_pool_limit:]).item() == 0
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_plan.missing_combine_metadata"] == {
        "stage": "m6_output_plan",
        "code": "missing_combine_metadata",
        "message": "direct combine metadata unavailable",
        "count": 1,
    }


def test_megamoe_m6_output_plan_records_forced_fallback() -> None:
    fake = SimpleNamespace(
        _full_fusion_force_output_path_fallback_reason="forced by test",
        _full_fusion_fallback_diagnostics={},
    )

    plan = MegaMoE._plan_full_fusion_m6_output(fake)
    second_plan = MegaMoE._plan_full_fusion_m6_output(fake)

    assert plan.layout == "pool"
    assert plan.use_direct_pool_fc_route is False
    assert plan.use_direct_combine_layout_output is False
    assert plan.use_direct_combine_buffer_output is False
    assert plan.fatal_fallback_reason == "forced by test"
    assert second_plan.fatal_fallback_reason == "forced by test"
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_plan.forced_fallback"] == {
        "stage": "m6_output_plan",
        "code": "forced_fallback",
        "message": "forced by test",
        "count": 2,
    }


def test_megamoe_m5_pool_route_outputs_uses_pool_layout_for_empty_direct_combine_batch() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    calls = []

    def build_route_inputs(*, include_selected_experts, full_capacity):
        calls.append((include_selected_experts, full_capacity))
        assert include_selected_experts is False
        assert full_capacity is True
        return (
            torch.zeros((config.num_max_pool_tokens, config.hidden_size // 2), dtype=torch.uint8),
            torch.zeros(
                (config.num_max_pool_tokens, config.hidden_size // config.scaling_vector_size),
                dtype=torch.uint8,
            ),
            None,
            torch.ones((config.num_max_pool_tokens, 1), dtype=torch.float32),
            0,
        ), None

    fake = SimpleNamespace(
        _full_fusion_runtime_gate=SimpleNamespace(
            workspace_descriptor=SimpleNamespace(layout=SimpleNamespace(config=config))
        ),
        _full_fusion_workspace=object(),
        _full_fusion_fallback_diagnostics={},
        _full_fusion_force_output_path_fallback_reason=None,
        _full_fusion_m5_direct_pool_fc_route_enabled=True,
        _full_fusion_m6_direct_combine_layout_output_enabled=True,
        _full_fusion_m6_direct_combine_buffer_output_enabled=True,
        _full_fusion_m6_route_output_layout="stale",
        _full_fusion_m6_route_output_layout_rows=99,
        _full_fusion_m6_route_output_active_rows=torch.tensor((1,), dtype=torch.int64),
        _build_full_fusion_m5_pool_route_inputs=build_route_inputs,
    )

    weighted_route_outputs, reason = MegaMoE._run_full_fusion_m5_pool_route_outputs(fake)

    assert reason is None
    assert weighted_route_outputs is not None
    assert tuple(weighted_route_outputs.shape) == (config.num_max_pool_tokens, config.hidden_size)
    assert fake._full_fusion_m6_route_output_layout == "pool"
    assert fake._full_fusion_m6_route_output_layout_rows is None
    assert fake._full_fusion_m6_route_output_active_rows is None
    assert calls == [(False, True)]


def test_megamoe_m5_pool_route_outputs_records_invalid_direct_pool_route_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    active_pool_limit = 3

    def build_route_inputs(*, include_selected_experts, full_capacity):
        assert include_selected_experts is False
        assert full_capacity is False
        return (
            torch.zeros((active_pool_limit, config.hidden_size // 2), dtype=torch.uint8),
            torch.zeros(
                (active_pool_limit, config.hidden_size // config.scaling_vector_size),
                dtype=torch.uint8,
            ),
            None,
            torch.ones((active_pool_limit, 1), dtype=torch.float32),
            active_pool_limit,
        ), None

    def build_route_metadata(active_limit, *, expected_device=None):
        assert active_limit == active_pool_limit
        assert expected_device == torch.device("cpu")
        return None, "invalid M5 direct pool FC route layout cache"

    fake = SimpleNamespace(
        _full_fusion_runtime_gate=SimpleNamespace(
            workspace_descriptor=SimpleNamespace(layout=SimpleNamespace(config=config))
        ),
        _full_fusion_workspace=object(),
        _full_fusion_fallback_diagnostics={},
        _full_fusion_force_output_path_fallback_reason=None,
        _full_fusion_m5_direct_pool_fc_route_enabled=True,
        _full_fusion_m6_direct_combine_layout_output_enabled=False,
        _full_fusion_m6_direct_combine_buffer_output_enabled=False,
        _full_fusion_m6_route_output_layout="stale",
        _full_fusion_m6_route_output_layout_rows=99,
        _full_fusion_m6_route_output_active_rows=torch.tensor((1,), dtype=torch.int64),
        _build_full_fusion_m5_pool_route_inputs=build_route_inputs,
        _build_full_fusion_m5_direct_pool_fc_route_metadata=build_route_metadata,
    )

    weighted_route_outputs, reason = MegaMoE._run_full_fusion_m5_pool_route_outputs(fake)

    assert weighted_route_outputs is None
    assert reason == "invalid M5 direct pool FC route layout cache"
    assert fake._full_fusion_m6_route_output_layout == "pool"
    assert fake._full_fusion_m6_route_output_layout_rows is None
    assert fake._full_fusion_m6_route_output_active_rows is None
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_plan.invalid_direct_pool_route_metadata"] == {
        "stage": "m6_output_plan",
        "code": "invalid_direct_pool_route_metadata",
        "message": "invalid M5 direct pool FC route layout cache",
        "count": 1,
    }


def test_megamoe_m5_pool_route_outputs_falls_back_from_direct_combine_buffer_view() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=4,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    active_pool_limit = 3
    active_combine_rows = torch.tensor((1, 5, 9), dtype=torch.int64)
    combine_layout_rows = config.ep_size * config.top_k * config.max_num_tokens_per_rank
    calls = []

    def build_route_inputs(*, include_selected_experts, full_capacity):
        calls.append(("route_inputs", include_selected_experts, full_capacity))
        assert include_selected_experts is False
        row_count = config.num_max_pool_tokens if full_capacity else active_pool_limit
        return (
            torch.zeros((row_count, config.hidden_size // 2), dtype=torch.uint8),
            torch.zeros(
                (row_count, config.hidden_size // config.scaling_vector_size), dtype=torch.uint8
            ),
            None,
            torch.ones((row_count, 1), dtype=torch.float32),
            active_pool_limit,
        ), None

    def build_combine_output_metadata(active_limit, *, expected_device=None):
        calls.append(("combine_metadata", active_limit, expected_device))
        assert active_limit == active_pool_limit
        return (
            torch.arange(active_pool_limit, dtype=torch.int32),
            torch.ones((config.num_max_pool_tokens, 1), dtype=torch.float32),
            combine_layout_rows,
            active_combine_rows,
        ), None

    def combine_buffer_output_view():
        calls.append("combine_buffer_view")
        return None, "direct combine-buffer output view unavailable"

    def build_route_metadata(active_limit, *, expected_device=None):
        calls.append(("route_metadata", active_limit, expected_device))
        return (
            (
                torch.tensor((0,), dtype=torch.int32),
                torch.tensor((active_limit,), dtype=torch.int32),
                torch.arange(active_limit, dtype=torch.int32),
                torch.tensor((1,), dtype=torch.int32),
            ),
            None,
        )

    def run_fused_fc1_fc2_combine(
        *,
        x_fp4,
        x_sf,
        token_final_scales,
        output,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        effective_top_k,
        output_permuted_idx_to_expanded_idx=None,
        direct_combine_buffer_output_config=None,
        monolithic_output=None,
        monolithic_control=None,
        monolithic_local_rank=0,
        monolithic_local_tokens=0,
    ):
        calls.append(("fused", tuple(output.shape), direct_combine_buffer_output_config))
        assert output_permuted_idx_to_expanded_idx is not None
        assert direct_combine_buffer_output_config is None
        assert tuple(output.shape) == (config.num_max_pool_tokens, config.hidden_size)
        output[:active_pool_limit].copy_(
            torch.arange(active_pool_limit * config.hidden_size, dtype=torch.float32)
            .to(dtype=output.dtype)
            .reshape(active_pool_limit, config.hidden_size)
        )

    fake = SimpleNamespace(
        _full_fusion_runtime_gate=SimpleNamespace(
            workspace_descriptor=SimpleNamespace(layout=SimpleNamespace(config=config))
        ),
        _full_fusion_workspace=object(),
        _full_fusion_fallback_diagnostics={},
        _full_fusion_force_output_path_fallback_reason=None,
        _full_fusion_m5_direct_pool_fc_route_enabled=True,
        _full_fusion_m6_direct_combine_layout_output_enabled=True,
        _full_fusion_m6_direct_combine_buffer_output_enabled=True,
        _full_fusion_m6_route_output_layout="stale",
        _full_fusion_m6_route_output_layout_rows=99,
        _full_fusion_m6_route_output_active_rows=torch.tensor((1,), dtype=torch.int64),
        _build_full_fusion_m5_pool_route_inputs=build_route_inputs,
        _build_full_fusion_m5_direct_combine_output_metadata=build_combine_output_metadata,
        _full_fusion_combine_buffer_output_view=combine_buffer_output_view,
        _build_full_fusion_m5_direct_pool_fc_route_metadata=build_route_metadata,
        _run_fused_fc1_fc2_combine=run_fused_fc1_fc2_combine,
    )

    _attach_disabled_full_fusion_profile_helpers(fake)

    weighted_route_outputs, reason = MegaMoE._run_full_fusion_m5_pool_route_outputs(fake)

    assert reason is None
    assert weighted_route_outputs is not None
    assert fake._full_fusion_m6_route_output_layout == "combine"
    assert fake._full_fusion_m6_route_output_layout_rows == combine_layout_rows
    assert fake._full_fusion_m6_route_output_active_rows is active_combine_rows
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_plan.missing_combine_buffer_output"] == {
        "stage": "m6_output_plan",
        "code": "missing_combine_buffer_output",
        "message": "direct combine-buffer output view unavailable",
        "count": 1,
    }
    assert calls == [
        ("route_inputs", False, True),
        ("combine_metadata", active_pool_limit, torch.device("cpu")),
        "combine_buffer_view",
        ("route_metadata", active_pool_limit, torch.device("cpu")),
        ("fused", (config.num_max_pool_tokens, config.hidden_size), None),
    ]


def test_megamoe_route_output_producer_peer_writes_weighted_fc2_outputs() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    rank0_tokens = 2
    rank1_tokens = 2
    rank0_x = (
        torch.arange(rank0_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 10
    ).reshape(rank0_tokens, config.hidden_size // 2)
    rank1_x = (
        torch.arange(rank1_tokens * (config.hidden_size // 2), dtype=torch.uint8) + 100
    ).reshape(rank1_tokens, config.hidden_size // 2)
    rank0_sf = (torch.arange(rank0_tokens, dtype=torch.uint8) + 1).reshape(rank0_tokens, 1)
    rank1_sf = (torch.arange(rank1_tokens, dtype=torch.uint8) + 50).reshape(rank1_tokens, 1)
    rank0_experts = torch.tensor(((2, 3), (0, 1)), dtype=torch.int64)
    rank1_experts = torch.tensor(((3, 2), (2, 3)), dtype=torch.int64)
    rank0_weights = torch.tensor(((0.2, 0.3), (9.0, 8.0)), dtype=torch.float32)
    rank1_weights = torch.tensor(((0.4, 0.5), (0.6, 0.7)), dtype=torch.float32)

    _copy_tensor_to_workspace_rank(fake, 0, "x_buf", rank0_x)
    _copy_tensor_to_workspace_rank(fake, 0, "x_sf_buf", rank0_sf)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_idx_buf", rank0_experts)
    _copy_tensor_to_workspace_rank(fake, 0, "topk_weights_buf", rank0_weights)
    _copy_tensor_to_workspace_rank(fake, 1, "x_buf", rank1_x)
    _copy_tensor_to_workspace_rank(fake, 1, "x_sf_buf", rank1_sf)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_idx_buf", rank1_experts)
    _copy_tensor_to_workspace_rank(fake, 1, "topk_weights_buf", rank1_weights)

    ok, reason = MegaMoE._materialize_full_fusion_dispatch_pull(fake, (rank0_tokens, rank1_tokens))
    assert ok is True
    assert reason is None

    epoch = 23
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=rank0_tokens)
    _write_m5_ready_control(fake, rank=1, epoch=epoch, num_tokens=rank1_tokens)
    _write_route_output_ready_control(fake, rank=0, epoch=epoch)

    weighted_route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    active_pool_outputs = {
        0: 1.0,
        1: 10.0,
        2: 100.0,
        4: 1000.0,
        5: 10000.0,
        6: 20000.0,
    }
    for pool_slot, base in active_pool_outputs.items():
        weighted_route_outputs[pool_slot] = torch.tensor(
            _weighted_output(base, config.hidden_size), dtype=torch.bfloat16
        )

    materialized, reason = (
        MegaMoE._sync_full_fusion_ranked_route_outputs_from_weighted_route_outputs(
            fake, (rank0_tokens, rank1_tokens), weighted_route_outputs, epoch
        )
    )

    assert reason is None
    assert materialized is not None
    torch.testing.assert_close(materialized[0], weighted_route_outputs[0])
    torch.testing.assert_close(materialized[1], weighted_route_outputs[1])
    torch.testing.assert_close(materialized[2], weighted_route_outputs[2])
    torch.testing.assert_close(materialized[4], weighted_route_outputs[4])
    torch.testing.assert_close(materialized[5], weighted_route_outputs[5])
    torch.testing.assert_close(materialized[6], weighted_route_outputs[6])

    rank0_route_outputs, reason = MegaMoE._full_fusion_ranked_route_output_buffer(fake, rank=0)
    assert reason is None
    assert rank0_route_outputs is not None
    rank1_route_outputs, reason = MegaMoE._full_fusion_ranked_route_output_buffer(fake, rank=1)
    assert reason is None
    assert rank1_route_outputs is not None
    torch.testing.assert_close(rank0_route_outputs[0, 0], weighted_route_outputs[0])
    torch.testing.assert_close(rank0_route_outputs[1, 0], weighted_route_outputs[4])
    torch.testing.assert_close(rank1_route_outputs[1, 0], weighted_route_outputs[1])
    torch.testing.assert_close(rank1_route_outputs[0, 1], weighted_route_outputs[2])
    torch.testing.assert_close(rank1_route_outputs[0, 0], weighted_route_outputs[5])
    torch.testing.assert_close(rank1_route_outputs[1, 1], weighted_route_outputs[6])

    control, control_reason = MegaMoE._full_fusion_m6_control_words(fake, rank=1)
    assert control_reason is None
    assert control is not None
    assert control[6:8].tolist() == [epoch, MegaMoE._FULL_FUSION_ROUTE_OUTPUT_READY_FLAG]
    assert int(control[9].item()) != MegaMoE._FULL_FUSION_M6_READY_FLAG


def test_megamoe_distributed_output_path_pushes_m5_pool_outputs_directly_to_m6() -> None:
    weighted_route_outputs = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    reduced_output = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    calls = []

    def run_m5_pool_route_outputs(*, token_counts=None, producer_epoch=None):
        calls.append("m5_pool")
        return weighted_route_outputs, None

    def run_legacy_route_output_producer(num_tokens_per_rank, producer_epoch):
        raise AssertionError("distributed output path should bypass route-output producer")

    def sync_m6_combine_push_and_reduce(
        num_tokens_per_rank,
        route_outputs,
        producer_epoch,
        *,
        trusted_route_metadata=False,
        trusted_active_pool_slots=None,
    ):
        calls.append(
            (
                tuple(num_tokens_per_rank),
                route_outputs is weighted_route_outputs,
                producer_epoch,
                trusted_route_metadata,
                trusted_active_pool_slots,
            )
        )
        return reduced_output, None

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _run_full_fusion_m5_pool_route_outputs=run_m5_pool_route_outputs,
        _run_full_fusion_m5_pool_route_output_producer=run_legacy_route_output_producer,
        _sync_full_fusion_m6_combine_push_and_reduce=sync_m6_combine_push_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m5_m6_output_path(
        fake, token_counts=(2, 3), producer_epoch=17
    )

    assert reason is None
    assert reduced is reduced_output
    assert fake._full_fusion_output_path_fallback_reason is None
    assert calls == ["m5_pool", ((2, 3), True, 17, True, None)]


def test_megamoe_distributed_output_path_can_pass_direct_m5_pool_slots_to_m6() -> None:
    weighted_route_outputs = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    reduced_output = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    active_pool_slots = torch.tensor((0, 2), dtype=torch.int64)
    calls = []

    def run_m5_pool_route_outputs(*, token_counts=None, producer_epoch=None):
        calls.append("m5_pool")
        return weighted_route_outputs, None

    def build_active_pool_slots():
        calls.append("active_slots")
        return active_pool_slots, None

    def sync_m6_combine_push_and_reduce(
        num_tokens_per_rank,
        route_outputs,
        producer_epoch,
        *,
        trusted_route_metadata=False,
        trusted_active_pool_slots=None,
    ):
        calls.append(
            (
                tuple(num_tokens_per_rank),
                route_outputs is weighted_route_outputs,
                producer_epoch,
                trusted_route_metadata,
                trusted_active_pool_slots is active_pool_slots,
            )
        )
        return reduced_output, None

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_m6_direct_pool_combine_push_enabled=True,
        _run_full_fusion_m5_pool_route_outputs=run_m5_pool_route_outputs,
        _build_full_fusion_m5_active_pool_slots=build_active_pool_slots,
        _sync_full_fusion_m6_combine_push_and_reduce=sync_m6_combine_push_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m5_m6_output_path(
        fake, token_counts=(2, 3), producer_epoch=17
    )

    assert reason is None
    assert reduced is reduced_output
    assert fake._full_fusion_output_path_fallback_reason is None
    assert calls == ["m5_pool", "active_slots", ((2, 3), True, 17, True, True)]


def test_megamoe_distributed_output_path_uses_direct_combine_layout_outputs() -> None:
    weighted_route_outputs = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    reduced_output = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    calls = []
    active_combine_rows = torch.tensor((1, 3), dtype=torch.int64)

    def run_m5_pool_route_outputs(*, token_counts=None, producer_epoch=None):
        calls.append("m5_pool")
        fake._full_fusion_m6_route_output_layout = "combine"
        fake._full_fusion_m6_route_output_layout_rows = 4
        fake._full_fusion_m6_route_output_active_rows = active_combine_rows
        return weighted_route_outputs, None

    def sync_m6_combine_layout_and_reduce(
        num_tokens_per_rank,
        route_outputs,
        producer_epoch,
        combine_layout_rows,
        active_rows,
    ):
        calls.append(
            (
                tuple(num_tokens_per_rank),
                route_outputs is weighted_route_outputs,
                producer_epoch,
                combine_layout_rows,
                active_rows is active_combine_rows,
            )
        )
        return reduced_output, None

    def sync_m6_combine_push_and_reduce(*args, **kwargs):
        raise AssertionError("direct combine-layout output should bypass M6 metadata scatter")

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_m6_route_output_layout="pool",
        _full_fusion_m6_route_output_layout_rows=None,
        _run_full_fusion_m5_pool_route_outputs=run_m5_pool_route_outputs,
        _sync_full_fusion_m6_combine_layout_and_reduce=sync_m6_combine_layout_and_reduce,
        _sync_full_fusion_m6_combine_push_and_reduce=sync_m6_combine_push_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m5_m6_output_path(
        fake, token_counts=(2, 3), producer_epoch=17
    )

    assert reason is None
    assert reduced is reduced_output
    assert fake._full_fusion_output_path_fallback_reason is None
    assert calls == ["m5_pool", ((2, 3), True, 17, 4, True)]


def test_megamoe_distributed_output_path_uses_direct_combine_buffer_output() -> None:
    direct_output = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    reduced_output = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    calls = []

    def run_m5_pool_route_outputs(*, token_counts=None, producer_epoch=None):
        calls.append("m5_pool")
        fake._full_fusion_m6_route_output_layout = "combine_buffer"
        return direct_output, None

    def sync_m6_direct_combine_buffer_and_reduce(
        num_tokens_per_rank, producer_epoch, *, profile_device=None
    ):
        calls.append(
            (tuple(num_tokens_per_rank), producer_epoch, profile_device == direct_output.device)
        )
        return reduced_output, None

    def sync_m6_combine_layout_and_reduce(*args, **kwargs):
        raise AssertionError("direct combine-buffer output should bypass combine-layout scatter")

    def sync_m6_combine_push_and_reduce(*args, **kwargs):
        raise AssertionError("direct combine-buffer output should bypass M6 metadata scatter")

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_m6_route_output_layout="pool",
        _run_full_fusion_m5_pool_route_outputs=run_m5_pool_route_outputs,
        _sync_full_fusion_m6_direct_combine_buffer_and_reduce=(
            sync_m6_direct_combine_buffer_and_reduce
        ),
        _sync_full_fusion_m6_combine_layout_and_reduce=sync_m6_combine_layout_and_reduce,
        _sync_full_fusion_m6_combine_push_and_reduce=sync_m6_combine_push_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m5_m6_output_path(
        fake, token_counts=(2, 3), producer_epoch=17
    )

    assert reason is None
    assert reduced is reduced_output
    assert fake._full_fusion_output_path_fallback_reason is None
    assert calls == ["m5_pool", ((2, 3), 17, True)]


def test_megamoe_m5_m6_output_path_records_route_output_unavailable_diagnostic() -> None:
    def run_m5_pool_route_outputs(*, token_counts=None, producer_epoch=None):
        return None, "M5 route output cache is stale"

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_fallback_diagnostics={},
        _run_full_fusion_m5_pool_route_outputs=run_m5_pool_route_outputs,
    )

    reduced, reason = MegaMoE._run_full_fusion_m5_m6_output_path(
        fake, token_counts=(2, 3), producer_epoch=17
    )

    assert reduced is None
    assert reason == "M5 route output cache is stale"
    assert fake._full_fusion_output_path_fallback_reason == "M5 route output cache is stale"
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_path.route_output_unavailable"] == {
        "stage": "m6_output_path",
        "code": "route_output_unavailable",
        "message": "M5 route output cache is stale",
        "count": 1,
    }


def test_megamoe_m6_output_plan_records_direct_combine_buffer_sync_failure() -> None:
    route_outputs = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    def sync_m6_direct_combine_buffer_and_reduce(
        num_tokens_per_rank, producer_epoch, *, profile_device=None
    ):
        assert tuple(num_tokens_per_rank) == (2, 3)
        assert producer_epoch == 17
        assert profile_device == route_outputs.device
        return None, "direct combine-buffer wait failed"

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_fallback_diagnostics={},
        _full_fusion_m6_route_output_layout="combine_buffer",
        _full_fusion_output_path_layout="stale",
        _sync_full_fusion_m6_direct_combine_buffer_and_reduce=(
            sync_m6_direct_combine_buffer_and_reduce
        ),
    )

    reduced, reason = MegaMoE._run_full_fusion_m6_output_plan(
        fake, token_counts=(2, 3), weighted_route_outputs=route_outputs, producer_epoch=17
    )

    assert reduced is None
    assert reason == "direct combine-buffer wait failed"
    assert fake._full_fusion_output_path_fallback_reason == "direct combine-buffer wait failed"
    assert fake._full_fusion_output_path_layout is None
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_path.direct_combine_buffer_sync_failed"] == {
        "stage": "m6_output_path",
        "code": "direct_combine_buffer_sync_failed",
        "message": "direct combine-buffer wait failed",
        "count": 1,
    }


def test_megamoe_m6_output_plan_records_successful_direct_combine_buffer_layout() -> None:
    route_outputs = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    reduced_output = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    def sync_m6_direct_combine_buffer_and_reduce(
        num_tokens_per_rank, producer_epoch, *, profile_device=None
    ):
        assert tuple(num_tokens_per_rank) == (2, 3)
        assert producer_epoch == 17
        assert profile_device == route_outputs.device
        return reduced_output, None

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_m6_route_output_layout="combine_buffer",
        _full_fusion_output_path_layout="stale",
        _sync_full_fusion_m6_direct_combine_buffer_and_reduce=(
            sync_m6_direct_combine_buffer_and_reduce
        ),
    )

    reduced, reason = MegaMoE._run_full_fusion_m6_output_plan(
        fake, token_counts=(2, 3), weighted_route_outputs=route_outputs, producer_epoch=17
    )

    assert reduced is reduced_output
    assert reason is None
    assert fake._full_fusion_output_path_fallback_reason is None
    assert fake._full_fusion_output_path_layout == "combine_buffer"


def test_megamoe_m6_output_plan_records_combine_layout_sync_failure() -> None:
    route_outputs = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    active_combine_rows = torch.tensor((1, 3), dtype=torch.int64)

    def sync_m6_combine_layout_and_reduce(
        num_tokens_per_rank, route_outputs_arg, producer_epoch, combine_layout_rows, active_rows
    ):
        assert tuple(num_tokens_per_rank) == (2, 3)
        assert route_outputs_arg is route_outputs
        assert producer_epoch == 17
        assert combine_layout_rows == 4
        assert active_rows is active_combine_rows
        return None, "combine-layout reduce failed"

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_fallback_diagnostics={},
        _full_fusion_m6_route_output_layout="combine",
        _full_fusion_m6_route_output_layout_rows=4,
        _full_fusion_m6_route_output_active_rows=active_combine_rows,
        _sync_full_fusion_m6_combine_layout_and_reduce=sync_m6_combine_layout_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m6_output_plan(
        fake, token_counts=(2, 3), weighted_route_outputs=route_outputs, producer_epoch=17
    )

    assert reduced is None
    assert reason == "combine-layout reduce failed"
    assert fake._full_fusion_output_path_fallback_reason == "combine-layout reduce failed"
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_path.combine_layout_sync_failed"] == {
        "stage": "m6_output_path",
        "code": "combine_layout_sync_failed",
        "message": "combine-layout reduce failed",
        "count": 1,
    }


def test_megamoe_m6_output_plan_records_pool_combine_push_sync_failure() -> None:
    route_outputs = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    def sync_m6_combine_push_and_reduce(
        num_tokens_per_rank,
        route_outputs_arg,
        producer_epoch,
        *,
        trusted_route_metadata=False,
        trusted_active_pool_slots=None,
    ):
        assert tuple(num_tokens_per_rank) == (2, 3)
        assert route_outputs_arg is route_outputs
        assert producer_epoch == 17
        assert trusted_route_metadata is True
        assert trusted_active_pool_slots is None
        return None, "pool combine-push wait failed"

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _full_fusion_fallback_diagnostics={},
        _full_fusion_m6_route_output_layout="pool",
        _full_fusion_m6_direct_pool_combine_push_enabled=False,
        _sync_full_fusion_m6_combine_push_and_reduce=sync_m6_combine_push_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m6_output_plan(
        fake, token_counts=(2, 3), weighted_route_outputs=route_outputs, producer_epoch=17
    )

    assert reduced is None
    assert reason == "pool combine-push wait failed"
    assert fake._full_fusion_output_path_fallback_reason == "pool combine-push wait failed"
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m6_output_path.pool_combine_push_sync_failed"] == {
        "stage": "m6_output_path",
        "code": "pool_combine_push_sync_failed",
        "message": "pool combine-push wait failed",
        "count": 1,
    }


def test_megamoe_single_rank_output_path_uses_m5_m6_pool_route() -> None:
    weighted_route_outputs = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    reduced_output = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    calls = []

    def run_m5_pool_route_outputs(*, token_counts=None, producer_epoch=None):
        calls.append("m5_pool")
        return weighted_route_outputs, None

    def sync_m6_combine_push_and_reduce(
        num_tokens_per_rank,
        route_outputs,
        producer_epoch,
        *,
        trusted_route_metadata=False,
        trusted_active_pool_slots=None,
    ):
        calls.append(
            (
                tuple(num_tokens_per_rank),
                route_outputs is weighted_route_outputs,
                producer_epoch,
                trusted_route_metadata,
                trusted_active_pool_slots,
            )
        )
        return reduced_output, None

    def legacy_m6_output_path(*args, **kwargs):
        raise AssertionError("single-rank output path should use M5/M6 pool route flow")

    fake = SimpleNamespace(
        _full_fusion_output_path_fallback_reason="stale",
        _run_full_fusion_m5_pool_route_outputs=run_m5_pool_route_outputs,
        _run_full_fusion_m6_output_path=legacy_m6_output_path,
        _sync_full_fusion_m6_combine_push_and_reduce=sync_m6_combine_push_and_reduce,
    )

    reduced, reason = MegaMoE._run_full_fusion_m5_m6_output_path(
        fake, token_counts=(2,), producer_epoch=17
    )

    assert reason is None
    assert reduced is reduced_output
    assert fake._full_fusion_output_path_fallback_reason is None
    assert fake._full_fusion_output_path_layout == "pool"
    assert calls == ["m5_pool", ((2,), True, 17, True, None)]


def test_megamoe_distributed_output_path_is_legacy_alias() -> None:
    reduced_output = torch.ones((1, 8), dtype=torch.float32)
    calls = []

    def run_m5_m6_output_path(*, token_counts, producer_epoch):
        calls.append((tuple(token_counts), producer_epoch))
        return reduced_output, None

    fake = SimpleNamespace(_run_full_fusion_m5_m6_output_path=run_m5_m6_output_path)

    reduced, reason = MegaMoE._run_full_fusion_distributed_output_path(
        fake, token_counts=(2, 3), producer_epoch=17
    )

    assert reason is None
    assert reduced is reduced_output
    assert calls == [((2, 3), 17)]


def test_megamoe_route_output_producer_readiness_reports_missing_peer() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    epoch = 5
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=2)
    _write_m5_ready_control(fake, rank=1, epoch=epoch, num_tokens=2)
    _write_route_output_ready_control(fake, rank=1, epoch=epoch)

    ready, reason = MegaMoE._collect_full_fusion_route_output_producers_ready(fake, epoch)

    assert ready is False
    assert reason == "route-output producer rank 0 is not ready"


def test_megamoe_single_rank_m5_m6_sync_materializes_and_reduces() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    x_fp4 = torch.arange(2 * (config.hidden_size // 2), dtype=torch.uint8).reshape(
        2, config.hidden_size // 2
    )
    x_sf = torch.arange(2, dtype=torch.uint8).reshape(2, 1)
    token_selected_experts = torch.tensor(((0, 1), (1, 0)), dtype=torch.int64)
    token_final_scales = torch.tensor(((0.25, 0.5), (0.75, 1.25)), dtype=torch.float32)
    weighted_route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    for pool_slot, base in ((0, 1.0), (1, 10.0), (4, 100.0), (5, 1000.0)):
        weighted_route_outputs[pool_slot] = torch.tensor(
            _weighted_output(base, config.hidden_size), dtype=torch.bfloat16
        )

    staged, stage_reason = MegaMoE._stage_full_fusion_dispatch_inputs(
        fake, x_fp4, x_sf, token_selected_experts, token_final_scales
    )
    _zero_all_combine_buffers(fake, config.ep_size)
    reduced, reason = MegaMoE._sync_full_fusion_m5_m6_materialize_and_reduce(
        fake, 2, weighted_route_outputs, (2,)
    )

    assert staged is True
    assert stage_reason is None
    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(
        reduced[0], weighted_route_outputs[0].float() + weighted_route_outputs[4].float()
    )
    torch.testing.assert_close(
        reduced[1], weighted_route_outputs[5].float() + weighted_route_outputs[1].float()
    )

    token_src_metadata, metadata_reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "token_src_metadata", torch.int32
    )
    assert metadata_reason is None
    assert token_src_metadata is not None
    token_src_metadata = token_src_metadata[: config.num_max_pool_tokens * 3].reshape(
        config.num_max_pool_tokens, 3
    )
    assert token_src_metadata[0].tolist() == [0, 0, 0]
    assert token_src_metadata[1].tolist() == [0, 1, 1]
    assert token_src_metadata[4].tolist() == [0, 0, 1]
    assert token_src_metadata[5].tolist() == [0, 1, 0]

    control, control_reason = MegaMoE._full_fusion_m6_control_words(fake)
    assert control_reason is None
    assert control is not None
    assert control.tolist() == [
        MegaMoE._FULL_FUSION_M5_READY_MAGIC,
        1,
        2,
        MegaMoE._FULL_FUSION_M5_READY_FLAG,
        1,
        MegaMoE._FULL_FUSION_M5_READY_FLAG,
        0,
        0,
        1,
        MegaMoE._FULL_FUSION_M6_READY_FLAG,
    ]
    assert fake._full_fusion_combine_push_fallback_reason is None


def test_megamoe_records_m6_combine_push_profile_events() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    fake._full_fusion_profile_enabled = True
    _write_m5_ready_control(fake, rank=0, epoch=1, num_tokens=2)
    _zero_all_combine_buffers(fake, config.ep_size)
    _write_runtime_token_src_metadata(fake, ((0, 0, 0), (0, 1, 1)))
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[0] = torch.tensor(_weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[1] = torch.tensor(_weighted_output(2.0, config.hidden_size), dtype=torch.bfloat16)

    reduced, reason = MegaMoE._sync_full_fusion_m6_combine_push_and_reduce(
        fake, (2,), route_outputs, 1
    )

    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(reduced[0], route_outputs[0].to(dtype=torch.float32))
    torch.testing.assert_close(reduced[1], route_outputs[1].to(dtype=torch.float32))
    events = MegaMoE.full_fusion_profile_events.fget(fake)
    for event_name in (
        "m6_combine_push.metadata_validate",
        "m6_combine_push.scatter",
        "m6_combine_push.materialize_total",
        "m6_combine_push.wait_ready",
        "m6_combine_push.reduce",
    ):
        assert len(events[event_name]) == 1
        assert events[event_name][0] >= 0.0


def test_megamoe_records_m6_combine_layout_profile_events() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    fake._full_fusion_profile_enabled = True
    _write_m5_ready_control(fake, rank=0, epoch=1, num_tokens=2)
    _zero_all_combine_buffers(fake, config.ep_size)
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[0] = torch.tensor(_weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[5] = torch.tensor(_weighted_output(5.0, config.hidden_size), dtype=torch.bfloat16)

    reduced, reason = MegaMoE._sync_full_fusion_m6_combine_layout_and_reduce(
        fake,
        (2,),
        route_outputs,
        1,
        config.ep_size * config.top_k * config.max_num_tokens_per_rank,
        torch.tensor((0, 5), dtype=torch.int64),
    )

    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(reduced[0], route_outputs[0].to(dtype=torch.float32))
    torch.testing.assert_close(reduced[1], route_outputs[5].to(dtype=torch.float32))
    events = MegaMoE.full_fusion_profile_events.fget(fake)
    for event_name in (
        "m6_combine_layout.scatter",
        "m6_combine_layout.materialize_total",
        "m6_combine_layout.wait_ready",
        "m6_combine_layout.reduce",
    ):
        assert len(events[event_name]) == 1
        assert events[event_name][0] >= 0.0


def test_megamoe_builds_m5_active_pool_slots_from_expert_counts() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((2, 3), dtype=torch.int64)
    )

    active_pool_slots, reason = MegaMoE._build_full_fusion_m5_active_pool_slots(fake)

    assert reason is None
    assert active_pool_slots is not None
    assert active_pool_slots.tolist() == [0, 1, 4, 5, 6]


def test_megamoe_reuses_cached_m5_active_pool_slots() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1, 4), dtype=torch.int64)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((-1, -1), dtype=torch.int64)
    )

    active_pool_slots, reason = MegaMoE._build_full_fusion_m5_active_pool_slots(fake)

    assert reason is None
    assert active_pool_slots is fake._full_fusion_m5_active_pool_slots
    assert active_pool_slots.tolist() == [0, 1, 4]


def test_megamoe_invalid_cached_m5_active_pool_slots_falls_back() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor(((0, 1), (4, 5)), dtype=torch.int64)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((2, 3), dtype=torch.int64)
    )

    active_pool_slots, reason = MegaMoE._build_full_fusion_m5_active_pool_slots(fake)

    assert reason is None
    assert active_pool_slots is not None
    assert active_pool_slots.tolist() == [0, 1, 4, 5, 6]


def test_megamoe_direct_combine_output_metadata_reuses_cached_m5_active_pool_slots() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((-1, -1), dtype=torch.int64)
    )
    _write_runtime_token_src_metadata(
        fake,
        (
            (0, 0, 1),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (1, 3, 0),
        ),
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.0, 0.0, 0.0, 0.7)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8
    )

    assert reason is None
    assert metadata is not None
    output_mapping, output_scales, combine_layout_rows, active_combine_rows = metadata
    assert combine_layout_rows == 16
    assert output_mapping.tolist() == [4, 16, 16, 16, 11, 16, 16, 16]
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.2))
    torch.testing.assert_close(output_scales[11, 0], torch.tensor(0.7))
    assert torch.count_nonzero(output_scales).item() == 2
    assert active_combine_rows.tolist() == [4, 11]


def test_megamoe_direct_combine_output_metadata_reuses_cached_m5_combine_rows() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4, 11), dtype=torch.int64)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((-1, -1), dtype=torch.int64)
    )
    _write_runtime_token_src_metadata(
        fake,
        (
            (0, -1, 1),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (1, -1, 0),
        ),
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.0, 0.0, 0.0, 0.7)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8
    )

    assert reason is None
    assert metadata is not None
    output_mapping, output_scales, combine_layout_rows, active_combine_rows = metadata
    assert combine_layout_rows == 16
    assert output_mapping.tolist() == [4, 16, 16, 16, 11, 16, 16, 16]
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.2))
    torch.testing.assert_close(output_scales[11, 0], torch.tensor(0.7))
    assert torch.count_nonzero(output_scales).item() == 2
    assert active_combine_rows is fake._full_fusion_m5_direct_combine_rows
    assert active_combine_rows.tolist() == [4, 11]


def test_megamoe_direct_combine_output_metadata_reuses_cached_output_mapping() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4, 11), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_output_mapping = torch.tensor(
        (4, 16, 16, 16, 11, 16, 16, 16), dtype=torch.int32
    )
    _write_runtime_token_src_metadata(
        fake,
        (
            (0, -1, 1),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (1, -1, 0),
        ),
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.0, 0.0, 0.0, 0.7)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8
    )

    assert reason is None
    assert metadata is not None
    output_mapping, output_scales, combine_layout_rows, active_combine_rows = metadata
    assert output_mapping is fake._full_fusion_m5_direct_combine_output_mapping
    assert combine_layout_rows == 16
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.2))
    torch.testing.assert_close(output_scales[11, 0], torch.tensor(0.7))
    assert active_combine_rows is fake._full_fusion_m5_direct_combine_rows


def test_megamoe_direct_combine_output_metadata_reuses_cached_output_scales() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4, 11), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_output_mapping = torch.tensor(
        (4, 16, 16, 16, 11, 16, 16, 16), dtype=torch.int32
    )
    fake._full_fusion_m5_direct_combine_output_scales = torch.zeros(
        (config.num_max_pool_tokens, 1), dtype=torch.float32
    )
    fake._full_fusion_m5_direct_combine_output_scales[4, 0] = 0.4
    fake._full_fusion_m5_direct_combine_output_scales[11, 0] = 0.9
    _write_runtime_token_src_metadata(
        fake,
        (
            (0, -1, 1),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (1, -1, 0),
        ),
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.0, 0.0, 0.0, 0.7)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8
    )

    assert reason is None
    assert metadata is not None
    _, output_scales, _, active_combine_rows = metadata
    assert output_scales is fake._full_fusion_m5_direct_combine_output_scales
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.4))
    torch.testing.assert_close(output_scales[11, 0], torch.tensor(0.9))
    assert active_combine_rows is fake._full_fusion_m5_direct_combine_rows


def test_megamoe_direct_combine_output_metadata_reuses_complete_cached_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4, 11), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_output_mapping = torch.tensor(
        (4, 16, 16, 16, 11, 16, 16, 16), dtype=torch.int32
    )
    fake._full_fusion_m5_direct_combine_output_scales = torch.zeros(
        (config.num_max_pool_tokens, 1), dtype=torch.float32
    )
    fake._full_fusion_m5_direct_combine_output_scales[4, 0] = 0.4
    fake._full_fusion_m5_direct_combine_output_scales[11, 0] = 0.9

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8, expected_device=torch.device("cpu")
    )

    assert reason is None
    assert metadata is not None
    output_mapping, output_scales, combine_layout_rows, active_combine_rows = metadata
    assert output_mapping is fake._full_fusion_m5_direct_combine_output_mapping
    assert output_scales is fake._full_fusion_m5_direct_combine_output_scales
    assert active_combine_rows is fake._full_fusion_m5_direct_combine_rows
    assert combine_layout_rows == 16
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.4))
    torch.testing.assert_close(output_scales[11, 0], torch.tensor(0.9))


def test_megamoe_direct_materialization_descriptor_feeds_route_and_output_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_pool_fc_route_layout = (
        torch.tensor((0, 1), dtype=torch.int32),
        torch.tensor((4, 8), dtype=torch.int32),
        torch.tensor((2,), dtype=torch.int32),
    )
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4, 11), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_output_mapping = torch.tensor(
        (4, 16, 16, 16, 11, 16, 16, 16), dtype=torch.int32
    )
    fake._full_fusion_m5_direct_combine_output_scales = torch.zeros(
        (config.num_max_pool_tokens, 1), dtype=torch.float32
    )
    fake._full_fusion_m5_direct_combine_output_scales[4, 0] = 0.4
    fake._full_fusion_m5_direct_combine_output_scales[11, 0] = 0.9

    descriptor = MegaMoE._cached_full_fusion_m5_direct_materialization_descriptor(
        fake,
        config,
        expected_device=torch.device("cpu"),
        output_active_pool_limit=8,
        require_route_layout=True,
    )

    assert descriptor is not None
    assert descriptor.active_pool_slots is fake._full_fusion_m5_active_pool_slots
    assert descriptor.active_pool_limit == 8
    assert descriptor.tile_idx_to_expert_idx is not None
    assert descriptor.tile_idx_to_mn_limit is not None
    assert (
        descriptor.tile_idx_to_expert_idx.data_ptr()
        == fake._full_fusion_m5_direct_pool_fc_route_layout[0].data_ptr()
    )
    assert (
        descriptor.tile_idx_to_mn_limit.data_ptr()
        == fake._full_fusion_m5_direct_pool_fc_route_layout[1].data_ptr()
    )
    assert descriptor.num_non_exiting_tiles is fake._full_fusion_m5_direct_pool_fc_route_layout[2]
    assert descriptor.active_combine_rows is fake._full_fusion_m5_direct_combine_rows
    assert descriptor.output_mapping is fake._full_fusion_m5_direct_combine_output_mapping
    assert descriptor.output_scales is fake._full_fusion_m5_direct_combine_output_scales

    route_metadata, reason = MegaMoE._build_full_fusion_m5_direct_pool_fc_route_metadata(
        fake, active_pool_limit=8, expected_device=torch.device("cpu")
    )
    assert reason is None
    assert route_metadata is not None
    tile_idx_to_expert_idx, tile_idx_to_mn_limit, _, num_non_exiting_tiles = route_metadata
    assert descriptor.tile_idx_to_expert_idx is not None
    assert descriptor.tile_idx_to_mn_limit is not None
    assert tile_idx_to_expert_idx.data_ptr() == descriptor.tile_idx_to_expert_idx.data_ptr()
    assert tile_idx_to_mn_limit.data_ptr() == descriptor.tile_idx_to_mn_limit.data_ptr()
    assert num_non_exiting_tiles is descriptor.num_non_exiting_tiles

    output_metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8, expected_device=torch.device("cpu")
    )
    assert reason is None
    assert output_metadata is not None
    output_mapping, output_scales, _, active_combine_rows = output_metadata
    assert output_mapping is descriptor.output_mapping
    assert output_scales is descriptor.output_scales
    assert active_combine_rows is descriptor.active_combine_rows


def test_megamoe_direct_combine_output_metadata_rebuilds_invalid_cached_output_mapping() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 4), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_rows = torch.tensor((4, 11), dtype=torch.int64)
    fake._full_fusion_m5_direct_combine_output_mapping = torch.tensor(
        (4, 7, 16, 16, 11, 16, 16, 16), dtype=torch.int32
    )
    fake._full_fusion_m5_direct_combine_output_scales = torch.zeros((4,), dtype=torch.float32)
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.0, 0.0, 0.0, 0.7)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8
    )

    assert reason is None
    assert metadata is not None
    output_mapping, output_scales, combine_layout_rows, active_combine_rows = metadata
    assert output_mapping is not fake._full_fusion_m5_direct_combine_output_mapping
    assert output_scales is not fake._full_fusion_m5_direct_combine_output_scales
    assert output_mapping.tolist() == [4, 16, 16, 16, 11, 16, 16, 16]
    assert combine_layout_rows == 16
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.2))
    torch.testing.assert_close(output_scales[11, 0], torch.tensor(0.7))
    assert active_combine_rows is fake._full_fusion_m5_direct_combine_rows


def test_megamoe_builds_direct_m5_combine_layout_output_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    expert_recv_count_sum, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "expert_recv_count_sum", torch.int64
    )
    assert reason is None
    assert expert_recv_count_sum is not None
    expert_recv_count_sum[: config.num_experts_per_rank].copy_(
        torch.tensor((2, 1), dtype=torch.int64)
    )
    _write_runtime_token_src_metadata(
        fake,
        (
            (0, 0, 1),
            (1, 2, 0),
            (-1, -1, -1),
            (-1, -1, -1),
            (0, 3, 0),
        ),
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:5].copy_(torch.tensor((0.2, 0.5, 0.0, 0.0, 0.7)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=8
    )

    assert reason is None
    assert metadata is not None
    output_mapping, output_scales, combine_layout_rows, active_combine_rows = metadata
    assert combine_layout_rows == 16
    assert output_mapping.tolist() == [4, 10, 16, 16, 3, 16, 16, 16]
    torch.testing.assert_close(output_scales[4, 0], torch.tensor(0.2))
    torch.testing.assert_close(output_scales[10, 0], torch.tensor(0.5))
    torch.testing.assert_close(output_scales[3, 0], torch.tensor(0.7))
    assert torch.count_nonzero(output_scales).item() == 3
    assert active_combine_rows.tolist() == [4, 10, 3]


def test_megamoe_direct_combine_output_metadata_rejects_duplicate_rows() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_active_pool_slots = torch.tensor((0, 1), dtype=torch.int64)
    _write_runtime_token_src_metadata(
        fake,
        (
            (0, 0, 1),
            (0, 0, 1),
        ),
    )
    l1_topk_weights_pool, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "l1_topk_weights_pool", torch.float32
    )
    assert reason is None
    assert l1_topk_weights_pool is not None
    l1_topk_weights_pool[:2].copy_(torch.tensor((0.2, 0.5)))

    metadata, reason = MegaMoE._build_full_fusion_m5_direct_combine_output_metadata(
        fake, active_pool_limit=4
    )

    assert metadata is None
    assert reason == "duplicate direct combine-layout output row 4"


def test_megamoe_m6_combine_layout_push_copies_flattened_rows() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    _zero_all_combine_buffers(fake, config.ep_size)
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[3] = torch.tensor(_weighted_output(3.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[4] = torch.tensor(_weighted_output(4.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[10] = torch.tensor(
        _weighted_output(10.0, config.hidden_size), dtype=torch.bfloat16
    )

    pushed, reason = MegaMoE._materialize_full_fusion_combine_layout_push(
        fake,
        (4, 4),
        route_outputs,
        combine_layout_rows=config.ep_size * config.top_k * config.max_num_tokens_per_rank,
        active_combine_rows=torch.tensor((3, 4, 10), dtype=torch.int64),
    )

    assert pushed is True
    assert reason is None
    combine_buffers, reason = MegaMoE._full_fusion_combine_buffers_all_ranks(fake)
    assert reason is None
    assert combine_buffers is not None
    torch.testing.assert_close(combine_buffers[0, 0, 3], route_outputs[3])
    torch.testing.assert_close(combine_buffers[0, 1, 0], route_outputs[4])
    torch.testing.assert_close(combine_buffers[1, 0, 2], route_outputs[10])


def test_megamoe_combine_buffer_output_view_preserves_rank_stride() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, rank_stride_bytes = _fake_megamoe_with_workspace(config, rank=1)

    output_view, reason = MegaMoE._full_fusion_combine_buffer_output_view(fake)

    assert reason is None
    assert output_view is not None
    assert output_view.shape == (
        config.ep_size,
        config.top_k * config.max_num_tokens_per_rank * config.hidden_size,
    )
    assert output_view.stride() == (rank_stride_bytes // BF16_BYTES, 1)
    output_view.zero_()
    output_view[1, 0] = torch.tensor(7.0, dtype=torch.bfloat16)
    combine_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=1)
    assert reason is None
    assert combine_buffer is not None
    torch.testing.assert_close(combine_buffer[0, 0, 0], torch.tensor(7.0, dtype=torch.bfloat16))


def test_megamoe_direct_combine_buffer_output_clear_is_noop_for_unique_row_store() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    fake._full_fusion_m5_direct_pool_fc_route_enabled = True
    fake._full_fusion_m6_direct_combine_layout_output_enabled = True
    fake._full_fusion_m6_direct_combine_buffer_output_enabled = True

    rank0_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=0)
    assert reason is None
    assert rank0_buffer is not None
    rank1_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=1)
    assert reason is None
    assert rank1_buffer is not None
    rank0_buffer.fill_(3.0)
    rank1_buffer.fill_(7.0)

    cleared, reason = MegaMoE._clear_full_fusion_local_direct_combine_buffer_output(fake)

    assert cleared is True
    assert reason is None
    torch.testing.assert_close(rank0_buffer, torch.full_like(rank0_buffer, 3.0))
    torch.testing.assert_close(rank1_buffer, torch.full_like(rank1_buffer, 7.0))


def test_megamoe_direct_combine_buffer_output_clear_is_gated() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    combine_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=0)
    assert reason is None
    assert combine_buffer is not None
    combine_buffer.fill_(5.0)

    cleared, reason = MegaMoE._clear_full_fusion_local_direct_combine_buffer_output(fake)

    assert cleared is True
    assert reason is None
    torch.testing.assert_close(combine_buffer, torch.full_like(combine_buffer, 5.0))


def test_megamoe_m6_direct_combine_buffer_output_reduces_after_barrier() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=1)
    epoch = 11
    _write_m5_ready_control(fake, rank=0, epoch=epoch, num_tokens=1)
    _write_m5_ready_control(fake, rank=1, epoch=epoch, num_tokens=1)
    _write_m6_ready_control(fake, rank=0, epoch=epoch)
    _zero_all_combine_buffers(fake, config.ep_size)
    combine_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake, rank=1)
    assert reason is None
    assert combine_buffer is not None
    route0 = torch.tensor(_weighted_output(3.0, config.hidden_size), dtype=torch.bfloat16)
    route1 = torch.tensor(_weighted_output(30.0, config.hidden_size), dtype=torch.bfloat16)
    combine_buffer[0, 0].copy_(route0)
    combine_buffer[1, 0].copy_(route1)

    reduced, reason = MegaMoE._sync_full_fusion_m6_direct_combine_buffer_and_reduce(
        fake, (1, 1), epoch
    )

    assert reason is None
    assert reduced is not None
    torch.testing.assert_close(reduced[0], route0.float() + route1.float())
    control, reason = MegaMoE._full_fusion_m6_control_words(fake)
    assert reason is None
    assert control is not None
    assert control[8:10].tolist() == [epoch, MegaMoE._FULL_FUSION_M6_READY_FLAG]


def test_megamoe_trusted_m6_combine_push_can_use_m5_active_pool_slots() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    _zero_all_combine_buffers(fake, config.ep_size)
    _write_runtime_token_src_metadata(fake, ((0, 0, 0), (0, 1, 0)))
    token_src_metadata, reason = MegaMoE._full_fusion_local_workspace_region_as(
        fake, "token_src_metadata", torch.int32
    )
    assert reason is None
    assert token_src_metadata is not None
    token_src_metadata = token_src_metadata[: config.num_max_pool_tokens * 3].reshape(
        config.num_max_pool_tokens, 3
    )
    token_src_metadata[4].copy_(torch.tensor((0, 0, 1), dtype=torch.int32))

    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[0] = torch.tensor(_weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[1] = torch.tensor(
        _weighted_output(100.0, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[4] = torch.tensor(_weighted_output(2.0, config.hidden_size), dtype=torch.bfloat16)

    pushed, reason = MegaMoE._materialize_full_fusion_combine_push(
        fake,
        (2,),
        route_outputs,
        trusted_route_metadata=True,
        trusted_active_pool_slots=torch.tensor((0, 4), dtype=torch.int64),
    )

    assert pushed is True
    assert reason is None
    combine_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake)
    assert reason is None
    assert combine_buffer is not None
    torch.testing.assert_close(combine_buffer[0, 0], route_outputs[0])
    torch.testing.assert_close(combine_buffer[1, 0], route_outputs[4])
    assert torch.count_nonzero(combine_buffer[0, 1]).item() == 0


def test_megamoe_trusted_m6_combine_push_scatters_m5_metadata() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    _zero_all_combine_buffers(fake, config.ep_size)
    _write_runtime_token_src_metadata(fake, ((0, 0, 0), (0, 1, 1)))
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[0] = torch.tensor(_weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[1] = torch.tensor(_weighted_output(2.0, config.hidden_size), dtype=torch.bfloat16)

    pushed, reason = MegaMoE._materialize_full_fusion_combine_push(
        fake, (2,), route_outputs, trusted_route_metadata=True
    )

    assert pushed is True
    assert reason is None
    combine_buffer, reason = MegaMoE._full_fusion_combine_buffer(fake)
    assert reason is None
    assert combine_buffer is not None
    torch.testing.assert_close(combine_buffer[0, 0], route_outputs[0])
    torch.testing.assert_close(combine_buffer[1, 1], route_outputs[1])


def test_megamoe_m6_combine_push_rejects_duplicate_and_inactive_nonzero_rows() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    route_outputs = torch.zeros(
        (config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16
    )
    route_outputs[0] = torch.tensor(_weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16)
    route_outputs[1] = torch.tensor(_weighted_output(2.0, config.hidden_size), dtype=torch.bfloat16)
    _write_runtime_token_src_metadata(fake, ((0, 0, 0), (0, 0, 0)))

    pushed, reason = MegaMoE._materialize_full_fusion_combine_push(fake, (1,), route_outputs)
    assert pushed is False
    assert reason is not None
    assert "duplicate combine push" in reason

    route_outputs.zero_()
    route_outputs[0] = torch.tensor(_weighted_output(1.0, config.hidden_size), dtype=torch.bfloat16)
    _write_runtime_token_src_metadata(fake, ())

    pushed, reason = MegaMoE._materialize_full_fusion_combine_push(fake, (1,), route_outputs)
    assert pushed is False
    assert reason is not None
    assert "must be zero for an inactive pool slot" in reason


def test_megamoe_skips_external_comm_combine_after_full_fusion_output_path() -> None:
    fake = SimpleNamespace(comm=object())

    assert MegaMoE._should_run_post_moe_comm_combine(fake, False) is True
    assert MegaMoE._should_run_post_moe_comm_combine(fake, True) is False

    fake.comm = None
    assert MegaMoE._should_run_post_moe_comm_combine(fake, False) is False


def test_megamoe_skips_post_dispatch_output_retry_after_pre_dispatch_attempt() -> None:
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
    )

    assert (
        MegaMoE._should_try_post_dispatch_full_fusion_output_path(
            fake, pre_dispatch_output_attempted=False
        )
        is True
    )
    assert (
        MegaMoE._should_try_post_dispatch_full_fusion_output_path(
            fake, pre_dispatch_output_attempted=True
        )
        is False
    )

    fake.comm = None
    assert (
        MegaMoE._should_try_post_dispatch_full_fusion_output_path(
            fake, pre_dispatch_output_attempted=True
        )
        is True
    )

    fake._full_fusion_runtime_gate = SimpleNamespace(use_full_fusion=False)
    assert (
        MegaMoE._should_try_post_dispatch_full_fusion_output_path(
            fake, pre_dispatch_output_attempted=False
        )
        is False
    )


def test_megamoe_pre_dispatch_output_path_returns_before_compat_dispatch() -> None:
    calls = []
    reduced = torch.ones((2, 4), dtype=torch.float32)
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout="stale",
        _full_fusion_output_path_fallback_stage="stale_stage",
        _full_fusion_output_path_fallback_code="stale_code",
        _full_fusion_m6_output_plan=SimpleNamespace(layout="stale"),
        _full_fusion_m5_pool_metadata_verify_enabled=True,
    )

    def staged_direct_topk(**_kwargs):
        calls.append("staged_direct_topk")
        return None, None

    def stage(x_fp4, x_sf, token_selected_experts, token_final_scales):
        calls.append("stage")
        return x_fp4.size(0), None

    def materialize(local_num_tokens, all_rank_num_tokens, *, materialization_scope=None):
        fake._full_fusion_m5_standalone_materialization_scope = materialization_scope
        calls.append(
            ("materialize", local_num_tokens, tuple(all_rank_num_tokens), materialization_scope)
        )
        return tuple(all_rank_num_tokens), 7, None

    def run_output_path(*, token_counts, producer_epoch):
        calls.append(("output", tuple(token_counts), producer_epoch))
        return reduced, None

    fake._run_full_fusion_monolithic_staged_direct_topk_output = staged_direct_topk
    fake._stage_full_fusion_dispatch_inputs_for_m5 = stage
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = materialize
    fake._run_full_fusion_m5_m6_output_path = run_output_path

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    output = MegaMoE._try_full_fusion_pre_dispatch_output_path(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=[2, 2],
    )

    assert output is not None
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, reduced.to(dtype=torch.bfloat16))
    assert fake._full_fusion_pre_dispatch_output_path_used is True
    assert fake._full_fusion_output_path_used is True
    assert fake._full_fusion_output_path_layout is None
    assert fake._full_fusion_output_path_fallback_stage is None
    assert fake._full_fusion_output_path_fallback_code is None
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["pre_dispatch_used"] is True
    assert status["m5_pre_materialization_support_reason"] == (
        "explicit M5 debug/materialization gate requested"
    )
    assert status["m5_standalone_materialization_scope"] == "pre_dispatch_output_path"
    assert fake._full_fusion_m6_output_plan is None
    assert fake._full_fusion_dispatch_stage_fallback_reason is None
    assert fake._full_fusion_staged_direct_topk_fallback_reason is None
    assert fake._full_fusion_dispatch_pull_fallback_reason is None
    assert fake._full_fusion_output_path_fallback_reason is None
    assert calls == [
        "staged_direct_topk",
        "stage",
        ("materialize", 2, (2, 2), "pre_dispatch_output_path"),
        ("output", (2, 2), 7),
    ]


def _enable_monolithic_direct_topk_fake(fake: SimpleNamespace) -> None:
    fake._full_fusion_output_path_requested = True
    fake._full_fusion_monolithic_direct_topk_output_enabled = True
    fake._full_fusion_monolithic_direct_topk_materialize_enabled = True
    fake._full_fusion_monolithic_m6_reduce_enabled = True
    fake._full_fusion_m5_direct_pool_fc_route_enabled = True
    fake._full_fusion_m6_direct_combine_layout_output_enabled = True
    fake._full_fusion_m6_direct_combine_buffer_output_enabled = True


def test_megamoe_monolithic_direct_topk_output_gate_disabled(monkeypatch) -> None:
    config = _small_single_rank_config()
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    _enable_monolithic_direct_topk_fake(fake)
    fake._full_fusion_monolithic_direct_topk_output_enabled = False
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce",
        object(),
        raising=False,
    )

    ready, reason = MegaMoE._can_use_full_fusion_monolithic_direct_topk_output(fake, (1,))

    assert ready is False
    assert reason == "monolithic direct-topk output gate disabled"


def test_megamoe_monolithic_direct_topk_output_gate_allows_large_batch(
    monkeypatch,
) -> None:
    config = _small_single_rank_config()
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    _enable_monolithic_direct_topk_fake(fake)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce",
        object(),
        raising=False,
    )

    ready, reason = MegaMoE._can_use_full_fusion_monolithic_direct_topk_output(
        fake, (config.tile_size,)
    )

    assert ready is True
    assert reason is None


def test_megamoe_m5_direct_topk_materialize_cta_plan_scales_with_problem_size() -> None:
    small = MegaMoE._plan_full_fusion_m5_direct_topk_materialize_ctas(
        4, sm_count=4, max_active_blocks_per_sm=8
    )
    large = MegaMoE._plan_full_fusion_m5_direct_topk_materialize_ctas(
        512 * 4, sm_count=4, max_active_blocks_per_sm=8
    )

    assert small["planned_ctas"] == 4
    assert large["planned_ctas"] == large["platform_cta_cap"]
    assert large["planned_ctas"] > small["planned_ctas"]


def test_megamoe_m6_combine_reduce_cta_plan_scales_with_problem_size() -> None:
    small = MegaMoE._plan_full_fusion_m6_combine_reduce_ctas(
        1, 128, sm_count=4, max_active_blocks_per_sm=8
    )
    large = MegaMoE._plan_full_fusion_m6_combine_reduce_ctas(
        512, 2048, sm_count=4, max_active_blocks_per_sm=8
    )

    assert small["planned_ctas"] == 1
    assert large["planned_ctas"] == large["platform_cta_cap"]
    assert large["planned_ctas"] > small["planned_ctas"]


def test_megamoe_monolithic_local_tokens_is_runtime_compile_arg() -> None:
    source_path = (
        Path(__file__).resolve().parents[5]
        / "tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py"
    )
    source = source_path.read_text()
    runner_start = source.index("class Sm100BlockScaledMegaMoeBlackwellRunner")
    cache_start = source.index("cache_key = (", runner_start)
    cache_end = source.index("if cache_key not in", cache_start)
    compile_args_start = source.index("compile_args = [", runner_start)
    compile_args_end = source.index("]", compile_args_start)
    compile_kwargs_start = source.index("compiled_gemm = cute.compile", runner_start)
    compile_kwargs_end = source.index(
        "self.__class__.kernel_cache[cache_key]", compile_kwargs_start
    )

    assert "monolithic_local_tokens" not in source[cache_start:cache_end]
    assert "monolithic_local_tokens" in source[compile_args_start:compile_args_end]
    assert "monolithic_local_tokens=" not in source[compile_kwargs_start:compile_kwargs_end]


def test_megamoe_monolithic_staged_direct_topk_defaults_single_rank_counts(
    monkeypatch,
) -> None:
    config = _small_single_rank_config()
    fake, _ = _fake_megamoe_with_workspace(config, rank=0)
    fake._full_fusion_monolithic_direct_topk_stage_enabled = True
    fake._full_fusion_output_path_requested = True
    x_fp4 = torch.zeros((3, config.hidden_size // 2), dtype=torch.uint8)
    x_sf = torch.zeros((3, config.hidden_size // config.scaling_vector_size), dtype=torch.uint8)
    selected = torch.tensor(((0, 1), (1, 2), (2, 0)), dtype=torch.int64)
    scales = torch.ones((3, config.top_k), dtype=torch.float32)
    captured = {}

    def can_use(_self, token_counts):
        captured["can_use_counts"] = tuple(token_counts)
        return True, None

    def rank_views(_config):
        return (
            (
                torch.zeros(
                    (1, config.max_num_tokens_per_rank, config.hidden_size // 2), dtype=torch.uint8
                ),
                torch.zeros(
                    (
                        1,
                        config.max_num_tokens_per_rank,
                        config.hidden_size // config.scaling_vector_size,
                    ),
                    dtype=torch.uint8,
                ),
                torch.zeros((1, config.max_num_tokens_per_rank, config.top_k), dtype=torch.int64),
                torch.ones((1, config.max_num_tokens_per_rank, config.top_k), dtype=torch.float32),
            ),
            None,
        )

    def run_kernel(**kwargs):
        captured["kernel_counts"] = tuple(kwargs["token_counts"])
        captured["staged_local_tokens"] = kwargs["staged_local_inputs"][0].size(0)
        return torch.ones((3, config.hidden_size), dtype=torch.bfloat16), None

    monkeypatch.setattr(MegaMoE, "_can_use_full_fusion_monolithic_direct_topk_output", can_use)
    fake._full_fusion_monolithic_direct_topk_rank_views = rank_views
    fake._full_fusion_combine_buffer_output_view = lambda: (
        torch.zeros((config.num_max_pool_tokens, config.hidden_size), dtype=torch.bfloat16),
        None,
    )
    fake._full_fusion_workspace_region_all_ranks_as = lambda _name, dtype: (
        torch.zeros((1, 8), dtype=dtype),
        None,
    )
    fake._run_full_fusion_monolithic_direct_topk_reduce_kernel = run_kernel

    output, reason = MegaMoE._run_full_fusion_monolithic_staged_direct_topk_output(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=None,
    )

    assert reason is None
    assert output is not None
    assert captured == {
        "can_use_counts": (3,),
        "kernel_counts": (3,),
        "staged_local_tokens": 3,
    }


def test_megamoe_pre_dispatch_output_path_accepts_no_comm_direct_topk() -> None:
    calls = []
    reduced = torch.ones((2, 4), dtype=torch.float32)
    fake = SimpleNamespace(
        comm=None,
        _full_fusion_output_path_requested=True,
        _full_fusion_runtime_gate=SimpleNamespace(
            requested=True, use_full_fusion=True, output_path_ready=True
        ),
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout=None,
        _full_fusion_output_path_fallback_stage=None,
        _full_fusion_output_path_fallback_code=None,
        _full_fusion_m6_output_plan=None,
        _full_fusion_fallback_diagnostics={},
    )

    def staged_direct_topk(**kwargs):
        calls.append(("staged_direct_topk", kwargs["all_rank_num_tokens"]))
        fake._full_fusion_m5_dispatch_materialize_kernel = "in_kernel_stage_direct_topk"
        fake._full_fusion_m6_combine_reduce_kernel = "in_kernel_direct_buffer"
        fake._full_fusion_output_path_layout = "combine_buffer"
        return reduced, None

    def unreachable(*_args, **_kwargs):
        raise AssertionError("no-comm direct top-k success must not use standalone M5 fallback")

    fake._run_full_fusion_monolithic_staged_direct_topk_output = staged_direct_topk
    fake._stage_full_fusion_dispatch_inputs_for_m5 = unreachable
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = unreachable
    fake._run_full_fusion_m5_m6_output_path = unreachable

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    output = MegaMoE._try_full_fusion_pre_dispatch_output_path(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=None,
    )

    assert output is not None
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, reduced.to(dtype=torch.bfloat16))
    assert fake._full_fusion_pre_dispatch_output_path_used is True
    assert fake._full_fusion_output_path_used is True
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["pre_dispatch_used"] is True
    assert status["final_kernel_path"] == "in_kernel_stage_direct_topk+in_kernel_direct_buffer"
    assert status["final_kernel_ready"] is True
    assert calls == [("staged_direct_topk", None)]


def test_megamoe_forward_impl_no_comm_returns_pre_dispatch_output_before_moe_sort(
    monkeypatch,
) -> None:
    calls = []
    expected_output = torch.full((2, 4), 3.0, dtype=torch.bfloat16)

    class _Routing:
        @staticmethod
        def apply(_router_logits):
            calls.append("routing")
            return (
                torch.zeros((2, 1), dtype=torch.int64),
                torch.ones((2, 1), dtype=torch.float32),
            )

    def fp4_quantize(input_tensor, *_args, **_kwargs):
        calls.append("fp4_quantize")
        return (
            torch.zeros((input_tensor.size(0), input_tensor.size(1) // 2), dtype=torch.uint8),
            torch.zeros((input_tensor.size(0), 1), dtype=torch.uint8),
        )

    def moe_sort_unreachable(*_args, **_kwargs):
        raise AssertionError("no-comm one-kernel path should return before moe_sort")

    def pre_dispatch_output(**kwargs):
        calls.append(("pre_dispatch", kwargs["all_rank_num_tokens"]))
        return expected_output

    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", fp4_quantize, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", moe_sort_unreachable, raising=False)

    fake = SimpleNamespace(
        _weights_created=True,
        comm=None,
        routing_method=_Routing(),
        fc31_input_scale=torch.ones((), dtype=torch.float32),
        scaling_vector_size=16,
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True, requested=True),
        _try_full_fusion_pre_dispatch_output_path=pre_dispatch_output,
    )

    output = MegaMoE.forward_impl(
        fake,
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((2, 3), dtype=torch.float32),
        all_rank_num_tokens=None,
    )

    assert output is expected_output
    assert calls == ["routing", "fp4_quantize", ("pre_dispatch", None)]


def test_megamoe_pre_dispatch_output_path_skips_m5_fallback_without_debug_gate() -> None:
    calls = []
    reason = "staged direct-topk was not attempted"
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
        _full_fusion_dispatch_stage_fallback_reason="stale_stage",
        _full_fusion_dispatch_pull_fallback_reason="stale_pull",
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout=None,
        _full_fusion_output_path_fallback_stage=None,
        _full_fusion_output_path_fallback_code=None,
        _full_fusion_m6_output_plan=None,
        _full_fusion_fallback_diagnostics={},
    )

    def staged_direct_topk(**_kwargs):
        calls.append("staged_direct_topk")
        return None, None

    def unreachable(*args, **kwargs):
        raise AssertionError("M5 fallback should require an explicit debug gate")

    fake._run_full_fusion_monolithic_staged_direct_topk_output = staged_direct_topk
    fake._full_fusion_m5_validation_gate_enabled = lambda: False
    fake._stage_full_fusion_dispatch_inputs_for_m5 = unreachable
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = unreachable
    fake._run_full_fusion_m5_m6_output_path = unreachable

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    output = MegaMoE._try_full_fusion_pre_dispatch_output_path(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=[2, 2],
    )

    assert output is None
    assert fake._full_fusion_pre_dispatch_output_path_used is False
    assert fake._full_fusion_output_path_used is False
    assert fake._full_fusion_dispatch_stage_fallback_reason is None
    assert fake._full_fusion_dispatch_pull_fallback_reason is None
    assert fake._full_fusion_output_path_fallback_reason == reason
    assert fake._full_fusion_output_path_fallback_stage == "staged_direct_topk"
    assert fake._full_fusion_output_path_fallback_code == "unavailable"
    assert fake._full_fusion_staged_direct_topk_fallback_reason == reason
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["staged_direct_topk_fallback_reason"] == reason
    assert status["m5_pre_materialization_support_reason"] is None
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["staged_direct_topk.unavailable"] == {
        "stage": "staged_direct_topk",
        "code": "unavailable",
        "message": reason,
        "count": 1,
    }
    assert calls == ["staged_direct_topk"]


def test_megamoe_pre_dispatch_output_path_records_staged_direct_topk_fallback() -> None:
    calls = []
    staged_direct_reason = "monolithic staged direct-topk unavailable"
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
        _full_fusion_dispatch_stage_fallback_reason="stale_stage",
        _full_fusion_dispatch_pull_fallback_reason="stale_pull",
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout=None,
        _full_fusion_output_path_fallback_stage=None,
        _full_fusion_output_path_fallback_code=None,
        _full_fusion_m6_output_plan=None,
        _full_fusion_fallback_diagnostics={},
    )

    def staged_direct_topk(**_kwargs):
        calls.append("staged_direct_topk")
        return None, staged_direct_reason

    def unreachable(*args, **kwargs):
        raise AssertionError("staged direct-topk miss should not run M5 fallback by default")

    fake._run_full_fusion_monolithic_staged_direct_topk_output = staged_direct_topk
    fake._full_fusion_m5_validation_gate_enabled = lambda: False
    fake._stage_full_fusion_dispatch_inputs_for_m5 = unreachable
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = unreachable
    fake._run_full_fusion_m5_m6_output_path = unreachable

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    output = MegaMoE._try_full_fusion_pre_dispatch_output_path(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=[2, 2],
    )

    assert output is None
    assert fake._full_fusion_pre_dispatch_output_path_used is False
    assert fake._full_fusion_output_path_used is False
    assert fake._full_fusion_dispatch_stage_fallback_reason is None
    assert fake._full_fusion_dispatch_pull_fallback_reason is None
    assert fake._full_fusion_output_path_fallback_reason == staged_direct_reason
    assert fake._full_fusion_output_path_fallback_stage == "staged_direct_topk"
    assert fake._full_fusion_output_path_fallback_code == "unavailable"
    assert fake._full_fusion_staged_direct_topk_fallback_reason == staged_direct_reason
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["staged_direct_topk_fallback_reason"] == staged_direct_reason
    assert status["m5_pre_materialization_support_reason"] is None
    assert status["dispatch_stage_fallback_reason"] is None
    assert status["dispatch_pull_fallback_reason"] is None
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["staged_direct_topk.unavailable"] == {
        "stage": "staged_direct_topk",
        "code": "unavailable",
        "message": staged_direct_reason,
        "count": 1,
    }
    assert calls == ["staged_direct_topk"]


def test_megamoe_pre_dispatch_output_path_uses_m5_fallback_under_validation_gate() -> None:
    calls = []
    staged_direct_reason = "monolithic staged direct-topk unavailable"
    reduced = torch.ones((2, 4), dtype=torch.float32)
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=False,
        _full_fusion_output_path_layout=None,
        _full_fusion_output_path_fallback_stage=None,
        _full_fusion_output_path_fallback_code=None,
        _full_fusion_m6_output_plan=None,
        _full_fusion_fallback_diagnostics={},
        _full_fusion_m5_pool_metadata_verify_enabled=True,
    )

    def staged_direct_topk(**_kwargs):
        calls.append("staged_direct_topk")
        return None, staged_direct_reason

    def stage(x_fp4, x_sf, token_selected_experts, token_final_scales):
        calls.append("stage")
        return x_fp4.size(0), None

    def materialize(local_num_tokens, all_rank_num_tokens, *, materialization_scope=None):
        fake._full_fusion_m5_standalone_materialization_scope = materialization_scope
        calls.append(
            ("materialize", local_num_tokens, tuple(all_rank_num_tokens), materialization_scope)
        )
        return tuple(all_rank_num_tokens), 7, None

    def run_output_path(*, token_counts, producer_epoch):
        calls.append(("output", tuple(token_counts), producer_epoch))
        return reduced, None

    fake._run_full_fusion_monolithic_staged_direct_topk_output = staged_direct_topk
    fake._stage_full_fusion_dispatch_inputs_for_m5 = stage
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = materialize
    fake._run_full_fusion_m5_m6_output_path = run_output_path

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    output = MegaMoE._try_full_fusion_pre_dispatch_output_path(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=[2, 2],
    )

    assert output is not None
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, reduced.to(dtype=torch.bfloat16))
    assert fake._full_fusion_pre_dispatch_output_path_used is True
    assert fake._full_fusion_output_path_used is True
    assert fake._full_fusion_dispatch_stage_fallback_reason is None
    assert fake._full_fusion_dispatch_pull_fallback_reason is None
    assert fake._full_fusion_output_path_fallback_reason is None
    assert fake._full_fusion_staged_direct_topk_fallback_reason == staged_direct_reason
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["staged_direct_topk_fallback_reason"] == staged_direct_reason
    assert status["m5_pre_materialization_support_reason"] == (
        "explicit M5 debug/materialization gate requested"
    )
    assert status["m5_standalone_materialization_scope"] == "pre_dispatch_output_path"
    assert status["m5_debug_materialization_gates"] == ("pool_metadata_verify",)
    assert status["dispatch_stage_fallback_reason"] is None
    assert calls == [
        "staged_direct_topk",
        "stage",
        ("materialize", 2, (2, 2), "pre_dispatch_output_path"),
        ("output", (2, 2), 7),
    ]


def test_megamoe_pre_dispatch_output_path_records_stage_fallback() -> None:
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=True,
        _full_fusion_output_path_layout="stale",
        _full_fusion_fallback_diagnostics={},
    )
    fake._full_fusion_m5_validation_gate_enabled = lambda: True
    fake._run_full_fusion_monolithic_staged_direct_topk_output = lambda **_kwargs: (None, None)

    def stage(x_fp4, x_sf, token_selected_experts, token_final_scales):
        return None, "stage failed"

    def unreachable(*args, **kwargs):
        raise AssertionError("pre-dispatch output path should stop after stage failure")

    fake._stage_full_fusion_dispatch_inputs_for_m5 = stage
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = unreachable
    fake._run_full_fusion_m5_m6_output_path = unreachable

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    output = MegaMoE._try_full_fusion_pre_dispatch_output_path(
        fake,
        x_fp4=x_fp4,
        x_sf=x_sf,
        token_selected_experts=selected,
        token_final_scales=scales,
        all_rank_num_tokens=[2, 2],
    )

    assert output is None
    assert fake._full_fusion_pre_dispatch_output_path_used is False
    assert fake._full_fusion_output_path_used is False
    assert fake._full_fusion_output_path_layout is None
    assert fake._full_fusion_dispatch_stage_fallback_reason == "stage failed"
    assert fake._full_fusion_dispatch_pull_fallback_reason == "stage failed"
    assert fake._full_fusion_output_path_fallback_reason == "stage failed"
    assert fake._full_fusion_output_path_fallback_stage == "m5_dispatch_stage"
    assert fake._full_fusion_output_path_fallback_code == "stage_failed"
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m5_dispatch_stage.stage_failed"] == {
        "stage": "m5_dispatch_stage",
        "code": "stage_failed",
        "message": "stage failed",
        "count": 1,
    }


def test_megamoe_m5_validation_gate_excludes_direct_pool_route() -> None:
    fake = SimpleNamespace(
        _full_fusion_m5_route_pull_verify_enabled=False,
        _full_fusion_m5_route_pull_materialize_enabled=False,
        _full_fusion_m5_route_metadata_verify_enabled=False,
        _full_fusion_m5_route_metadata_materialize_enabled=False,
        _full_fusion_m5_pool_metadata_verify_enabled=False,
        _full_fusion_m5_pool_metadata_materialize_enabled=False,
        _full_fusion_m5_helper_materialize_enabled=False,
        _full_fusion_m5_direct_pool_fc_route_enabled=True,
    )

    assert MegaMoE._active_full_fusion_m5_debug_materialization_gates(fake) == ()
    assert MegaMoE.full_fusion_m5_debug_materialization_gates.fget(fake) == ()
    assert MegaMoE._full_fusion_m5_validation_gate_enabled(fake) is False
    assert MegaMoE._get_full_fusion_m5_pre_materialization_support_reason(fake, None) is None
    assert (
        MegaMoE._get_full_fusion_m5_pre_materialization_support_reason(
            fake, "staged direct-topk failed"
        )
        is None
    )

    fake._full_fusion_m5_pool_metadata_verify_enabled = True
    fake._full_fusion_m5_helper_materialize_enabled = True
    assert MegaMoE._active_full_fusion_m5_debug_materialization_gates(fake) == (
        "pool_metadata_verify",
        "helper_materialize",
    )
    assert MegaMoE.full_fusion_m5_debug_materialization_gates.fget(fake) == (
        "pool_metadata_verify",
        "helper_materialize",
    )
    assert MegaMoE._full_fusion_m5_validation_gate_enabled(fake) is True
    assert (
        MegaMoE._get_full_fusion_m5_pre_materialization_support_reason(
            fake, "staged direct-topk failed"
        )
        == "explicit M5 debug/materialization gate requested"
    )


def test_megamoe_pre_dispatch_output_path_records_materialize_fallback() -> None:
    fake = SimpleNamespace(
        comm=object(),
        _full_fusion_runtime_gate=SimpleNamespace(use_full_fusion=True),
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_pre_dispatch_output_path_used=False,
        _full_fusion_output_path_used=True,
        _full_fusion_output_path_layout="stale",
        _full_fusion_fallback_diagnostics={},
    )

    def stage(x_fp4, x_sf, token_selected_experts, token_final_scales):
        return x_fp4.size(0), None

    def materialize(local_num_tokens, all_rank_num_tokens, *, materialization_scope=None):
        assert local_num_tokens == 2
        assert tuple(all_rank_num_tokens) == (2, 2)
        assert materialization_scope == "pre_dispatch_output_path"
        fake._full_fusion_m5_standalone_materialization_scope = materialization_scope
        return None, 7, "M5 materialize failed"

    def unreachable(*args, **kwargs):
        raise AssertionError("pre-dispatch output path should stop after materialize failure")

    fake._run_full_fusion_monolithic_staged_direct_topk_output = lambda **_kwargs: (None, None)
    fake._stage_full_fusion_dispatch_inputs_for_m5 = stage
    fake._sync_full_fusion_m5_producers_and_materialize_with_counts = materialize
    fake._run_full_fusion_m5_m6_output_path = unreachable
    fake._full_fusion_m5_validation_gate_enabled = lambda: True

    x_fp4 = torch.zeros((2, 2), dtype=torch.uint8)
    x_sf = torch.zeros((2, 1), dtype=torch.uint8)
    selected = torch.zeros((2, 1), dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="M5 materialize failed"):
        MegaMoE._try_full_fusion_pre_dispatch_output_path(
            fake,
            x_fp4=x_fp4,
            x_sf=x_sf,
            token_selected_experts=selected,
            token_final_scales=scales,
            all_rank_num_tokens=[2, 2],
        )

    assert fake._full_fusion_pre_dispatch_output_path_used is False
    assert fake._full_fusion_output_path_used is False
    assert fake._full_fusion_output_path_layout is None
    assert fake._full_fusion_dispatch_stage_fallback_reason is None
    assert fake._full_fusion_dispatch_pull_fallback_reason == "M5 materialize failed"
    assert fake._full_fusion_output_path_fallback_reason == "M5 materialize failed"
    assert fake._full_fusion_output_path_fallback_stage == "m5_dispatch_pull"
    assert fake._full_fusion_output_path_fallback_code == "materialize_failed"
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["m5_standalone_materialization_scope"] == "pre_dispatch_output_path"
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["m5_dispatch_pull.materialize_failed"] == {
        "stage": "m5_dispatch_pull",
        "code": "materialize_failed",
        "message": "M5 materialize failed",
        "count": 1,
    }


def test_full_fusion_runtime_gate_enables_explicit_attention_dp_output_path() -> None:
    fake = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
        comm=object(),
        tile_size=4,
        num_slots=4,
        routing_method=SimpleNamespace(top_k=2),
        hidden_size=16,
        intermediate_size_per_partition=16,
        scaling_vector_size=16,
        _full_fusion_output_path_fallback_reason=None,
        _try_alloc_full_fusion_workspace=lambda layout: (True, layout.size_bytes, None),
    )
    fake._disabled_full_fusion_runtime_gate = MethodType(
        MegaMoE._disabled_full_fusion_runtime_gate, fake
    )
    model_config = SimpleNamespace(
        extra_attrs={
            "megamoe_enable_full_fusion_runtime": True,
            "megamoe_enable_full_fusion_output_path": True,
        },
        max_num_tokens=4,
    )

    attention_dp_gate = MegaMoE._build_full_fusion_runtime_gate(fake, model_config)

    assert attention_dp_gate.use_full_fusion is True
    assert attention_dp_gate.m5_dispatch_pull_ready is True
    assert attention_dp_gate.m6_combine_push_ready is True
    assert attention_dp_gate.output_path_ready is True
    assert attention_dp_gate.fallback_reason is None
    assert fake._full_fusion_output_path_fallback_reason is None


def test_full_fusion_runtime_gate_enables_explicit_pure_ep_output_path() -> None:
    fake = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
        comm=None,
        tile_size=4,
        num_slots=4,
        routing_method=SimpleNamespace(top_k=2),
        hidden_size=16,
        intermediate_size_per_partition=16,
        scaling_vector_size=16,
        _full_fusion_output_path_fallback_reason=None,
        _try_alloc_full_fusion_workspace=lambda layout: (True, layout.size_bytes, None),
    )
    fake._disabled_full_fusion_runtime_gate = MethodType(
        MegaMoE._disabled_full_fusion_runtime_gate, fake
    )
    model_config = SimpleNamespace(
        extra_attrs={
            "megamoe_enable_full_fusion_runtime": True,
            "megamoe_enable_full_fusion_output_path": True,
        },
        max_num_tokens=4,
    )

    multi_rank_gate = MegaMoE._build_full_fusion_runtime_gate(fake, model_config)

    assert multi_rank_gate.use_full_fusion is True
    assert multi_rank_gate.m5_dispatch_pull_ready is True
    assert multi_rank_gate.m6_combine_push_ready is True
    assert multi_rank_gate.output_path_ready is True
    assert multi_rank_gate.fallback_reason is None
    assert fake._full_fusion_output_path_fallback_reason is None
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["requested"] is True
    assert status["eligible"] is True
    assert status["planned_layout"] is None
    assert status["fallback_reason"] is None


def test_full_fusion_output_path_default_keeps_runtime_gate_disabled() -> None:
    fake = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
        comm=object(),
        tile_size=4,
        num_slots=4,
        routing_method=SimpleNamespace(top_k=2),
        hidden_size=16,
        intermediate_size_per_partition=16,
        scaling_vector_size=16,
        _full_fusion_output_path_fallback_reason=None,
        _try_alloc_full_fusion_workspace=lambda layout: (True, layout.size_bytes, None),
    )
    fake._disabled_full_fusion_runtime_gate = MethodType(
        MegaMoE._disabled_full_fusion_runtime_gate, fake
    )
    model_config = SimpleNamespace(extra_attrs={}, max_num_tokens=4)

    gate = MegaMoE._build_full_fusion_runtime_gate(fake, model_config)

    assert gate.requested is False
    assert gate.use_full_fusion is False
    assert gate.output_path_ready is False
    assert fake._full_fusion_output_path_fallback_reason == "full-fusion runtime gate disabled"
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["requested"] is False
    assert status["runtime_requested"] is False
    assert status["eligible"] is False
    assert status["fallback_reason"] == "full-fusion runtime gate disabled"


def test_full_fusion_output_path_status_reports_final_kernel_path() -> None:
    fake = SimpleNamespace(
        _full_fusion_runtime_gate=SimpleNamespace(
            requested=True,
            use_full_fusion=True,
            output_path_ready=True,
        ),
        _full_fusion_output_path_requested=True,
        _full_fusion_m6_output_plan=SimpleNamespace(layout="combine_buffer"),
        _full_fusion_pre_dispatch_output_path_used=True,
        _full_fusion_output_path_used=True,
        _full_fusion_output_path_layout="combine_buffer",
        _full_fusion_m5_dispatch_materialize_kernel="direct_topk",
        _full_fusion_m6_combine_reduce_kernel="direct_buffer",
        _full_fusion_output_path_fallback_reason=None,
        _full_fusion_output_path_fallback_stage=None,
        _full_fusion_output_path_fallback_code=None,
        _full_fusion_dispatch_stage_fallback_reason=None,
        _full_fusion_dispatch_pull_fallback_reason=None,
        _full_fusion_combine_push_fallback_reason=None,
        _full_fusion_fallback_diagnostics={},
    )

    status = MegaMoE.full_fusion_output_path_status.fget(fake)

    assert MegaMoE.full_fusion_final_kernel_path.fget(fake) == "direct_topk+direct_buffer"
    assert MegaMoE.full_fusion_final_kernel_ready.fget(fake) is False
    assert status["m5_dispatch_materialize_kernel"] == "direct_topk"
    assert status["m6_combine_reduce_kernel"] == "direct_buffer"
    assert status["final_kernel_path"] == "direct_topk+direct_buffer"
    assert status["final_kernel_ready"] is False

    fake._full_fusion_m6_combine_reduce_kernel = "in_kernel_direct_buffer"
    status = MegaMoE.full_fusion_output_path_status.fget(fake)

    assert MegaMoE.full_fusion_final_kernel_path.fget(fake) == (
        "direct_topk+in_kernel_direct_buffer"
    )
    assert MegaMoE.full_fusion_final_kernel_ready.fget(fake) is False
    assert status["m6_combine_reduce_kernel"] == "in_kernel_direct_buffer"
    assert status["final_kernel_path"] == "direct_topk+in_kernel_direct_buffer"
    assert status["final_kernel_ready"] is False

    fake._full_fusion_m5_dispatch_materialize_kernel = "in_kernel_direct_topk"
    status = MegaMoE.full_fusion_output_path_status.fget(fake)

    assert MegaMoE.full_fusion_final_kernel_path.fget(fake) == (
        "in_kernel_direct_topk+in_kernel_direct_buffer"
    )
    assert MegaMoE.full_fusion_final_kernel_ready.fget(fake) is True
    assert status["m5_dispatch_materialize_kernel"] == "in_kernel_direct_topk"
    assert status["final_kernel_path"] == "in_kernel_direct_topk+in_kernel_direct_buffer"
    assert status["final_kernel_ready"] is True

    fake._full_fusion_m5_dispatch_materialize_kernel = "in_kernel_stage_direct_topk"
    status = MegaMoE.full_fusion_output_path_status.fget(fake)

    assert MegaMoE.full_fusion_final_kernel_path.fget(fake) == (
        "in_kernel_stage_direct_topk+in_kernel_direct_buffer"
    )
    assert MegaMoE.full_fusion_final_kernel_ready.fget(fake) is True
    assert status["m5_dispatch_materialize_kernel"] == "in_kernel_stage_direct_topk"
    assert status["final_kernel_path"] == "in_kernel_stage_direct_topk+in_kernel_direct_buffer"
    assert status["final_kernel_ready"] is True


def test_megamoe_runtime_gate_records_output_path_disabled_diagnostic() -> None:
    fake = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_size=2, moe_ep_rank=1),
        comm=None,
        tile_size=4,
        num_slots=4,
        routing_method=SimpleNamespace(top_k=2),
        hidden_size=16,
        intermediate_size_per_partition=16,
        scaling_vector_size=16,
        _full_fusion_output_path_fallback_reason=None,
        _try_alloc_full_fusion_workspace=lambda layout: (True, layout.size_bytes, None),
    )
    fake._disabled_full_fusion_runtime_gate = MethodType(
        MegaMoE._disabled_full_fusion_runtime_gate, fake
    )
    model_config = SimpleNamespace(
        extra_attrs={
            "megamoe_enable_full_fusion_runtime": True,
            "megamoe_enable_full_fusion_output_path": False,
        },
        max_num_tokens=4,
    )

    gate = MegaMoE._build_full_fusion_runtime_gate(fake, model_config)

    assert gate.use_full_fusion is False
    assert gate.output_path_ready is False
    assert gate.fallback_reason == "full-fusion output path wiring is not ready"
    assert fake._full_fusion_output_path_fallback_reason == (
        "full-fusion output path gate disabled"
    )
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    assert diagnostics["runtime_gate.output_path_disabled"] == {
        "stage": "runtime_gate",
        "code": "output_path_disabled",
        "message": "full-fusion output path gate disabled",
        "count": 1,
    }
    status = MegaMoE.full_fusion_output_path_status.fget(fake)
    assert status["requested"] is False
    assert status["eligible"] is False
    assert status["runtime_requested"] is True
    assert status["fallback_reason"] == "full-fusion output path gate disabled"
    assert status["diagnostics"] == diagnostics


def test_megamoe_runtime_gate_records_unsupported_shape_diagnostic() -> None:
    fake = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_size=3, moe_ep_rank=0),
        comm=None,
        tile_size=4,
        num_slots=4,
        routing_method=SimpleNamespace(top_k=2),
        hidden_size=16,
        intermediate_size_per_partition=16,
        scaling_vector_size=16,
        _full_fusion_output_path_fallback_reason=None,
        _try_alloc_full_fusion_workspace=lambda layout: (True, layout.size_bytes, None),
    )
    fake._disabled_full_fusion_runtime_gate = MethodType(
        MegaMoE._disabled_full_fusion_runtime_gate, fake
    )
    model_config = SimpleNamespace(
        extra_attrs={
            "megamoe_enable_full_fusion_runtime": True,
            "megamoe_enable_full_fusion_output_path": True,
        },
        max_num_tokens=4,
    )

    gate = MegaMoE._build_full_fusion_runtime_gate(fake, model_config)

    assert gate.use_full_fusion is False
    assert gate.workspace_layout is None
    assert gate.output_path_ready is False
    assert gate.fallback_reason.startswith("full-fusion workspace config invalid")
    assert fake._full_fusion_output_path_fallback_reason == gate.fallback_reason
    diagnostics = MegaMoE._full_fusion_fallback_diagnostics_snapshot(fake)
    unsupported_shape = diagnostics["runtime_gate.unsupported_shape"]
    assert unsupported_shape["stage"] == "runtime_gate"
    assert unsupported_shape["code"] == "unsupported_shape"
    assert unsupported_shape["count"] == 1
    assert str(unsupported_shape["message"]).startswith("full-fusion workspace config invalid")


def test_full_fusion_runtime_gate_requires_m5_m6_and_output_wiring() -> None:
    config = _small_single_rank_config()

    missing_dispatch = build_megamoe_full_fusion_runtime_gate(
        config, requested=True, workspace_runtime_ready=True
    )
    missing_combine = build_megamoe_full_fusion_runtime_gate(
        config,
        requested=True,
        workspace_runtime_ready=True,
        m5_dispatch_pull_ready=True,
    )
    missing_output = build_megamoe_full_fusion_runtime_gate(
        config,
        requested=True,
        workspace_runtime_ready=True,
        m5_dispatch_pull_ready=True,
        m6_combine_push_ready=True,
    )
    ready = build_megamoe_full_fusion_runtime_gate(
        config,
        requested=True,
        workspace_runtime_ready=True,
        m5_dispatch_pull_ready=True,
        m6_combine_push_ready=True,
        output_path_ready=True,
    )

    assert missing_dispatch.use_full_fusion is False
    assert missing_dispatch.fallback_reason == "M5 dispatch-pull runtime wiring is not ready"
    assert missing_combine.use_full_fusion is False
    assert missing_combine.fallback_reason == "M6 combine-push runtime wiring is not ready"
    assert missing_output.use_full_fusion is False
    assert missing_output.fallback_reason == "full-fusion output path wiring is not ready"
    assert ready.use_full_fusion is True
    assert ready.fallback_reason is None
    assert ready.output_path_ready is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ep_size": 0}, "ep_size must be positive"),
        ({"num_experts": 10, "ep_size": 4}, "num_experts .* must be divisible"),
        ({"max_num_tokens_per_rank": 129}, "padded to tile_size"),
        ({"hidden_size": 513}, "hidden_size must be even"),
        ({"intermediate_size": 513}, "intermediate_size must be even"),
        ({"hidden_size": 514}, "hidden_size must be divisible"),
        ({"intermediate_size": 514}, "intermediate_size must be divisible"),
    ],
)
def test_workspace_config_validation(kwargs: dict[str, int], match: str) -> None:
    base_kwargs = dict(
        ep_size=1,
        max_num_tokens_per_rank=128,
        num_experts=1,
        top_k=1,
        hidden_size=512,
        intermediate_size=512,
    )
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        MegaMoeFullFusionWorkspaceConfig(**base_kwargs)


def test_single_rank_full_fusion_proof_populates_dispatch_metadata() -> None:
    config = _small_single_rank_config()
    selected_experts = ((2, 0), (1, 2), (0, 1))
    route_weights = ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125))
    outputs = _route_outputs(len(selected_experts), config.top_k, config.hidden_size)

    proof = build_megamoe_single_rank_full_fusion_proof(
        config, selected_experts, route_weights, outputs
    )

    assert proof.expert_send_count == (2, 2, 2)
    assert proof.expert_recv_count == ((2, 2, 2),)
    assert proof.expert_recv_count_sum == (2, 2, 2)
    assert proof.num_active_pool_tokens == 6
    assert proof.num_materialized_pool_slots == 12

    expected_src_idx = [INVALID_UINT32] * 12
    expected_src_idx[0:2] = [1, 4]
    expected_src_idx[4:6] = [2, 5]
    expected_src_idx[8:10] = [0, 3]
    assert proof.src_token_topk_idx == tuple(expected_src_idx)

    assert proof.token_src_metadata[0] == MegaMoeTokenSourceMetadata(0, 0, 1)
    assert proof.token_src_metadata[1] == MegaMoeTokenSourceMetadata(0, 2, 0)
    assert proof.token_src_metadata[4] == MegaMoeTokenSourceMetadata(0, 1, 0)
    assert proof.token_src_metadata[5] == MegaMoeTokenSourceMetadata(0, 2, 1)
    assert proof.token_src_metadata[8] == MegaMoeTokenSourceMetadata(0, 0, 0)
    assert proof.token_src_metadata[9] == MegaMoeTokenSourceMetadata(0, 1, 1)
    assert proof.token_src_metadata[2] == MegaMoeTokenSourceMetadata(
        INVALID_UINT32, INVALID_UINT32, INVALID_UINT32
    )

    assert proof.l1_arrival_count == (1, 1, 1, 0, 0, 0)
    assert [
        (
            route.expert_idx,
            route.pool_slot,
            route.token_idx,
            route.topk_idx,
            route.src_token_topk_idx,
        )
        for route in proof.routes
    ] == [
        (0, 0, 0, 1, 1),
        (0, 1, 2, 0, 4),
        (1, 4, 1, 0, 2),
        (1, 5, 2, 1, 5),
        (2, 8, 0, 0, 0),
        (2, 9, 1, 1, 3),
    ]


def test_single_rank_full_fusion_route_weight_and_combine_semantics() -> None:
    config = _small_single_rank_config()
    selected_experts = ((2, 0), (1, 2), (0, 1))
    route_weights = ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125))
    outputs = _route_outputs(len(selected_experts), config.top_k, config.hidden_size)

    proof = build_megamoe_single_rank_full_fusion_proof(
        config, selected_experts, route_weights, outputs
    )

    assert proof.l1_topk_weights_pool[8] == route_weights[0][0]
    assert proof.l1_topk_weights_pool[0] == route_weights[0][1]
    assert proof.l1_topk_weights_pool[4] == route_weights[1][0]
    assert proof.l1_topk_weights_pool[9] == route_weights[1][1]
    assert proof.l1_topk_weights_pool[1] == route_weights[2][0]
    assert proof.l1_topk_weights_pool[5] == route_weights[2][1]

    expected_topk0_token0 = tuple(value * route_weights[0][0] for value in outputs[0][0])
    expected_topk1_token0 = tuple(value * route_weights[0][1] for value in outputs[0][1])
    assert proof.combine_token_buffer[0][0] == expected_topk0_token0
    assert proof.combine_token_buffer[1][0] == expected_topk1_token0

    expected_reduced = tuple(
        outputs[0][0][hidden_idx] * route_weights[0][0]
        + outputs[0][1][hidden_idx] * route_weights[0][1]
        for hidden_idx in range(config.hidden_size)
    )
    assert proof.reduced_output[0] == expected_reduced

    double_scaled = tuple(
        outputs[0][0][hidden_idx] * route_weights[0][0] * route_weights[0][0]
        + outputs[0][1][hidden_idx] * route_weights[0][1] * route_weights[0][1]
        for hidden_idx in range(config.hidden_size)
    )
    assert proof.reduced_output[0] != double_scaled


def test_single_rank_full_fusion_topk1_weight_one_is_identity() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    selected_experts = ((0,), (1,))
    route_weights = ((1.0,), (1.0,))
    outputs = _route_outputs(len(selected_experts), config.top_k, config.hidden_size)

    proof = build_megamoe_single_rank_full_fusion_proof(
        config, selected_experts, route_weights, outputs
    )

    assert proof.reduced_output == tuple(token_routes[0] for token_routes in outputs)
    assert proof.combine_token_buffer[0][0] == outputs[0][0]
    assert proof.combine_token_buffer[0][1] == outputs[1][0]


def test_single_rank_dispatch_pull_materializes_payload_pools() -> None:
    config = _small_single_rank_config()
    selected_experts = ((2, 0), (1, 2), (0, 1))
    route_weights = ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125))
    x_buf = _payload_rows(len(selected_experts), config.hidden_size // 2)
    x_sf_buf = _scale_payload_rows(
        len(selected_experts), config.hidden_size // config.scaling_vector_size
    )

    proof = build_megamoe_single_rank_dispatch_pull_proof(
        config, x_buf, x_sf_buf, selected_experts, route_weights
    )

    assert proof.metadata.l1_arrival_count == (1, 1, 1, 0, 0, 0)
    assert proof.metadata.l1_topk_weights_pool[0] == route_weights[0][1]
    assert proof.metadata.l1_topk_weights_pool[1] == route_weights[2][0]
    assert proof.metadata.l1_topk_weights_pool[4] == route_weights[1][0]
    assert proof.metadata.l1_topk_weights_pool[5] == route_weights[2][1]
    assert proof.metadata.l1_topk_weights_pool[8] == route_weights[0][0]
    assert proof.metadata.l1_topk_weights_pool[9] == route_weights[1][1]

    assert proof.l1_acts_pool[0] == x_buf[0]
    assert proof.l1_acts_pool[1] == x_buf[2]
    assert proof.l1_acts_pool[4] == x_buf[1]
    assert proof.l1_acts_pool[5] == x_buf[2]
    assert proof.l1_acts_pool[8] == x_buf[0]
    assert proof.l1_acts_pool[9] == x_buf[1]
    assert proof.l1_acts_pool[2] == (0,) * (config.hidden_size // 2)

    assert proof.pool_slot_to_sf_slot[0] == 0
    assert proof.pool_slot_to_sf_slot[1] == 1
    assert proof.pool_slot_to_sf_slot[4] == 128
    assert proof.pool_slot_to_sf_slot[5] == 129
    assert proof.pool_slot_to_sf_slot[8] == 256
    assert proof.pool_slot_to_sf_slot[9] == 257
    assert proof.pool_slot_to_sf_slot[2] == INVALID_UINT32

    assert proof.l1_acts_sf_pool[0] == x_sf_buf[0]
    assert proof.l1_acts_sf_pool[1] == x_sf_buf[2]
    assert proof.l1_acts_sf_pool[128] == x_sf_buf[1]
    assert proof.l1_acts_sf_pool[129] == x_sf_buf[2]
    assert proof.l1_acts_sf_pool[256] == x_sf_buf[0]
    assert proof.l1_acts_sf_pool[257] == x_sf_buf[1]
    assert proof.l1_acts_sf_pool[2] == (0,) * (config.hidden_size // config.scaling_vector_size)


def test_single_rank_dispatch_pull_reuses_m6_combine_outputs() -> None:
    config = _small_single_rank_config()
    selected_experts = ((2, 0), (1, 2), (0, 1))
    route_weights = ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125))
    outputs = _route_outputs(len(selected_experts), config.top_k, config.hidden_size)
    x_buf = _payload_rows(len(selected_experts), config.hidden_size // 2)
    x_sf_buf = _scale_payload_rows(
        len(selected_experts), config.hidden_size // config.scaling_vector_size
    )

    proof = build_megamoe_single_rank_dispatch_pull_proof(
        config,
        x_buf,
        x_sf_buf,
        selected_experts,
        route_weights,
        unweighted_route_outputs=outputs,
    )

    expected_reduced = tuple(
        outputs[0][0][hidden_idx] * route_weights[0][0]
        + outputs[0][1][hidden_idx] * route_weights[0][1]
        for hidden_idx in range(config.hidden_size)
    )
    assert proof.metadata.reduced_output[0] == expected_reduced


def test_combine_push_materializes_peer_buffers_and_barrier_ready() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    zero_output = (0.0,) * config.hidden_size
    metadata = (
        MegaMoeTokenSourceMetadata(0, 0, 0),
        MegaMoeTokenSourceMetadata(1, 0, 0),
        MegaMoeTokenSourceMetadata(0, 0, 1),
        MegaMoeTokenSourceMetadata(1, 0, 1),
        MegaMoeTokenSourceMetadata(INVALID_UINT32, INVALID_UINT32, INVALID_UINT32),
    )
    outputs = (
        _weighted_output(1.0, config.hidden_size),
        _weighted_output(100.0, config.hidden_size),
        _weighted_output(10.0, config.hidden_size),
        _weighted_output(1000.0, config.hidden_size),
        zero_output,
    )

    proof = build_megamoe_combine_push_reduce_proof(config, (1, 1), metadata, outputs)

    assert proof.combine_token_buffer[0][0][0] == outputs[0]
    assert proof.combine_token_buffer[0][1][0] == outputs[2]
    assert proof.combine_token_buffer[1][0][0] == outputs[1]
    assert proof.combine_token_buffer[1][1][0] == outputs[3]
    assert proof.combine_token_buffer[0][0][1] == zero_output
    assert proof.combine_slot_ready[0][0][0] is True
    assert proof.combine_slot_ready[0][1][0] is True
    assert proof.combine_slot_ready[0][0][1] is False
    assert proof.barrier_ready_by_rank == (True, True)

    expected_rank0 = tuple(
        outputs[0][hidden_idx] + outputs[2][hidden_idx] for hidden_idx in range(config.hidden_size)
    )
    expected_rank1 = tuple(
        outputs[1][hidden_idx] + outputs[3][hidden_idx] for hidden_idx in range(config.hidden_size)
    )
    assert proof.reduced_output[0][0] == expected_rank0
    assert proof.reduced_output[1][0] == expected_rank1
    assert [
        (route.pool_slot, route.dst_rank, route.token_idx, route.topk_idx) for route in proof.routes
    ] == [(0, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 1), (3, 1, 0, 1)]


def test_combine_push_barrier_waits_for_every_expected_topk_slot() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=2,
        max_num_tokens_per_rank=4,
        num_experts=4,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    metadata = (
        MegaMoeTokenSourceMetadata(0, 0, 0),
        MegaMoeTokenSourceMetadata(0, 0, 1),
        MegaMoeTokenSourceMetadata(1, 0, 0),
    )
    outputs = (
        _weighted_output(1.0, config.hidden_size),
        _weighted_output(10.0, config.hidden_size),
        _weighted_output(100.0, config.hidden_size),
    )

    proof = build_megamoe_combine_push_reduce_proof(config, (1, 1), metadata, outputs)

    assert proof.barrier_ready_by_rank == (True, False)
    assert proof.combine_slot_ready[1][0][0] is True
    assert proof.combine_slot_ready[1][1][0] is False


def test_combine_push_consumes_preweighted_outputs_without_double_scaling() -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    route_weights = (0.25, 0.5)
    unweighted_outputs = (
        _weighted_output(2.0, config.hidden_size),
        _weighted_output(20.0, config.hidden_size),
    )
    preweighted_outputs = tuple(
        tuple(value * route_weights[topk_idx] for value in unweighted_output)
        for topk_idx, unweighted_output in enumerate(unweighted_outputs)
    )
    metadata = (
        MegaMoeTokenSourceMetadata(0, 0, 0),
        MegaMoeTokenSourceMetadata(0, 0, 1),
    )

    proof = build_megamoe_combine_push_reduce_proof(config, (1,), metadata, preweighted_outputs)

    expected_reduced = tuple(
        preweighted_outputs[0][hidden_idx] + preweighted_outputs[1][hidden_idx]
        for hidden_idx in range(config.hidden_size)
    )
    double_scaled = tuple(
        unweighted_outputs[0][hidden_idx] * route_weights[0] * route_weights[0]
        + unweighted_outputs[1][hidden_idx] * route_weights[1] * route_weights[1]
        for hidden_idx in range(config.hidden_size)
    )
    assert proof.combine_token_buffer[0][0][0] == preweighted_outputs[0]
    assert proof.combine_token_buffer[0][1][0] == preweighted_outputs[1]
    assert proof.reduced_output[0][0] == expected_reduced
    assert proof.reduced_output[0][0] != double_scaled


@pytest.mark.parametrize(
    "metadata, outputs, match",
    [
        (
            (
                MegaMoeTokenSourceMetadata(0, 0, 0),
                MegaMoeTokenSourceMetadata(0, 0, 0),
            ),
            (
                _weighted_output(1.0, 16),
                _weighted_output(2.0, 16),
            ),
            "duplicate combine push",
        ),
        (
            (MegaMoeTokenSourceMetadata(0, 4, 0),),
            (_weighted_output(1.0, 16),),
            "token_idx",
        ),
        (
            (MegaMoeTokenSourceMetadata(INVALID_UINT32, INVALID_UINT32, INVALID_UINT32),),
            (_weighted_output(1.0, 16),),
            "inactive pool slot",
        ),
    ],
)
def test_combine_push_validation(
    metadata: tuple[MegaMoeTokenSourceMetadata, ...],
    outputs: tuple[tuple[float, ...], ...],
    match: str,
) -> None:
    config = MegaMoeFullFusionWorkspaceConfig(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=2,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )

    with pytest.raises(ValueError, match=match):
        build_megamoe_combine_push_reduce_proof(config, (1,), metadata, outputs)


@pytest.mark.parametrize(
    "x_buf_update, x_sf_buf_update, match",
    [
        (lambda rows: (rows[0][:-1], *rows[1:]), None, "x_buf\\[0\\]"),
        (None, lambda rows: ((), *rows[1:]), "x_sf_buf\\[0\\]"),
        (lambda rows: ((256, *rows[0][1:]), *rows[1:]), None, "uint8 payload"),
        (lambda rows: rows[:-1], None, "x_buf row count"),
    ],
)
def test_single_rank_dispatch_pull_validation(
    x_buf_update: Callable[[_PayloadRows], _PayloadRows] | None,
    x_sf_buf_update: Callable[[_PayloadRows], _PayloadRows] | None,
    match: str,
) -> None:
    config = _small_single_rank_config()
    selected_experts = ((2, 0), (1, 2), (0, 1))
    route_weights = ((0.25, 1.0), (0.5, 0.75), (1.25, 0.125))
    x_buf = _payload_rows(len(selected_experts), config.hidden_size // 2)
    x_sf_buf = _scale_payload_rows(
        len(selected_experts), config.hidden_size // config.scaling_vector_size
    )
    if x_buf_update is not None:
        x_buf = x_buf_update(x_buf)
    if x_sf_buf_update is not None:
        x_sf_buf = x_sf_buf_update(x_sf_buf)

    with pytest.raises(ValueError, match=match):
        build_megamoe_single_rank_dispatch_pull_proof(
            config, x_buf, x_sf_buf, selected_experts, route_weights
        )


@pytest.mark.parametrize(
    "kwargs, selected_experts, route_weights, outputs, match",
    [
        ({"ep_size": 2, "num_experts": 4}, ((0, 1),), ((1.0, 1.0),), None, "ep_size == 1"),
        (
            {"max_num_tokens_per_rank": 4},
            ((0, 1),) * 5,
            ((1.0, 1.0),) * 5,
            None,
            "max_num_tokens_per_rank",
        ),
        ({}, ((0, 0),), ((1.0, 1.0),), None, "duplicate experts"),
        ({}, ((0, 3),), ((1.0, 1.0),), None, "local experts"),
        ({}, ((0, 1),), ((1.0, 1.0),), (((1.0,), (2.0,)),), "hidden_size"),
    ],
)
def test_single_rank_full_fusion_proof_validation(
    kwargs: dict[str, int],
    selected_experts: tuple[tuple[int, ...], ...],
    route_weights: tuple[tuple[float, ...], ...],
    outputs: tuple[tuple[tuple[float, ...], ...], ...] | None,
    match: str,
) -> None:
    base_kwargs = dict(
        ep_size=1,
        max_num_tokens_per_rank=4,
        num_experts=3,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        tile_size=4,
    )
    base_kwargs.update(kwargs)
    config = MegaMoeFullFusionWorkspaceConfig(**base_kwargs)
    if outputs is None:
        outputs = _route_outputs(len(selected_experts), config.top_k, config.hidden_size)

    with pytest.raises(ValueError, match=match):
        build_megamoe_single_rank_full_fusion_proof(
            config, selected_experts, route_weights, outputs
        )
