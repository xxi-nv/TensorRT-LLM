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
"""Pure-PyTorch references the CuteDSL grouped-GEMM tests compare against."""

from typing import List, Tuple

import torch


def _expert_count_tile_plan(
    expert_counts: List[int],
    capacity: int,
    tile_size: int,
) -> List[Tuple[int, int, int]]:
    """Return the reference tile plan for count-native scheduling.

    Each tuple is ``(expert_idx, permuted_mn_limit, expanded_row_start)`` for
    one MMA M tile. Source-contract tests use this helper to cover empty
    experts and boundary counts without requiring a GPU build.
    """
    if capacity <= 0 or tile_size <= 0:
        raise ValueError("capacity and tile_size must be positive")

    plan = []
    tile_idx = 0
    for expert_idx, raw_count in enumerate(expert_counts):
        count = min(max(raw_count, 0), capacity)
        for row_in_expert in range(0, count, tile_size):
            rows_in_tile = min(tile_size, count - row_in_expert)
            plan.append(
                (
                    expert_idx,
                    tile_idx * tile_size + rows_in_tile,
                    expert_idx * capacity + row_in_expert,
                )
            )
            tile_idx += 1
    return plan


def cute_dsl_nvfp4_grouped_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    output_dtype: torch.dtype,
    scaling_vector_size: int = 16,
):
    assert a.dtype == torch.float4_e2m1fn_x2
    assert a.dim() == 2
    assert b.dtype == torch.float4_e2m1fn_x2
    assert b.dim() == 3
    assert a_sf.dtype == torch.uint8
    assert a_sf.dim() == 1
    assert b_sf.dtype == torch.uint8
    assert b_sf.dim() == 3
    assert alpha.dtype == torch.float32
    assert alpha.dim() == 1

    m, k = a.size(0), a.size(1) * 2
    num_groups, n = b.size(0), b.size(1)
    scale_k = k // scaling_vector_size
    assert m % tile_size == 0
    assert k % (scaling_vector_size * 4) == 0
    assert b.size(2) * 2 == k
    assert a_sf.size(0) == m * scale_k
    assert b_sf.size(0) == num_groups
    assert b_sf.size(1) == n
    assert b_sf.size(2) == scale_k
    assert alpha.size(0) == num_groups

    num_tiles = m // tile_size
    assert tile_idx_to_group_idx.dtype == torch.int32
    assert tile_idx_to_group_idx.size() == (num_tiles,)
    assert num_non_exiting_tiles.dtype == torch.int32
    assert num_non_exiting_tiles.size() == (1,)

    num_tiles_per_expert = torch.bincount(
        tile_idx_to_group_idx[: num_non_exiting_tiles[0].item()], minlength=num_groups
    )
    offsets = [0] + num_tiles_per_expert.cumsum(dim=0).tolist()

    ref = torch.empty(m, n, dtype=output_dtype, device="cuda")
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end <= start:
            continue
        a_sliced = a[start * tile_size : end * tile_size]
        a_sf_sliced = a_sf[
            start * tile_size * k // scaling_vector_size : end
            * tile_size
            * k
            // scaling_vector_size
        ]
        ref[start * tile_size : end * tile_size] = torch.ops.trtllm.nvfp4_gemm(
            a_sliced.view(torch.uint8),
            b[i].view(torch.uint8),
            a_sf_sliced,
            b_sf[i],
            alpha[i],
            output_dtype,
        )

    return ref
