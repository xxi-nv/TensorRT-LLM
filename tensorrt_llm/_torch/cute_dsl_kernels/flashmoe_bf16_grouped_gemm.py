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
"""
FlashMoE bf16 Grouped GEMM Kernels using cuteDSL.

Provides two fused kernel operations for FlashMoE:

1. FC1: Gather + GroupedGEMM + SwiGLU
   - Gathers tokens by expert assignment (using permuted indices from moe_sort)
   - Grouped GEMM: [T_e, H] x [H, 2*I] -> [T_e, 2*I] per expert
   - SwiGLU activation fused in epilogue: silu(gate) * up

2. FC2: GroupedGEMM + Scale + Scatter-Add (Finalize)
   - Grouped GEMM: [T_e, I] x [I, H] -> [T_e, H] per expert
   - Scales output by routing weights
   - Scatter-adds back to original token positions

Both kernels use:
- moe_sort() for tile-to-expert mapping and permutation indices
- cuteDSL TiledMma for bf16 HMMA/WGMMA instructions
- Persistent tile scheduling for SM90+ (Hopper/Blackwell)

Target architectures:
- SM90 (Hopper): WGMMA-based bf16 GEMM
- SM100 (Blackwell): tcgen05.mma-based bf16 GEMM (TMEM accumulator)
"""

from typing import List, Optional, Tuple

import torch

from ..._utils import get_sm_version


def _check_bf16_kernel_support() -> Tuple[bool, Optional[str]]:
    """Check if the current GPU supports bf16 cuteDSL kernels."""
    sm_version = get_sm_version()
    if sm_version < 90:
        return False, f"FlashMoE bf16 cuteDSL requires SM >= 90, got SM {sm_version}"
    return True, None


class FlashMoEBf16GatherGemmSwigluKernel:
    """
    cuteDSL kernel for FC1: Gather + GroupedGEMM + SwiGLU.

    This kernel performs:
    1. Gather input tokens using permuted_idx_to_expanded_idx (from moe_sort)
    2. Per-expert GEMM: A[gathered] @ W_gate_up^T -> [T_e, 2*I]
    3. SwiGLU: silu(gate_part) * up_part -> [T_e, I]

    Memory layout:
    - A (input activations): [num_tokens, hidden_size], bf16, row-major
    - B (weights): [num_experts, 2*intermediate_size, hidden_size], bf16
      - Layout: [w3(gate/up), w1(gate/silu)] interleaved or contiguous
    - C (output): [total_permuted_tokens, intermediate_size], bf16

    Tile management (from moe_sort):
    - tile_idx_to_expert_idx: maps each tile to its expert
    - tile_idx_to_mn_limit: number of valid tokens in each tile
    - permuted_idx_to_expanded_idx: gather indices for A

    For SM100 (Blackwell):
    - Uses TMA for B matrix loading
    - Uses LDGSTS for A matrix loading with gather
    - Uses tcgen05.mma for bf16 GEMM (accumulator in TMEM)
    - SwiGLU fused in epilogue warps

    For SM90 (Hopper):
    - Uses TMA for B matrix loading
    - Uses LDGSTS/cp.async for A matrix loading
    - Uses WGMMA for bf16 GEMM
    - SwiGLU fused after GEMM epilogue
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        raster_along_m: bool = False,
    ):
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m

    @staticmethod
    def can_implement(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
    ) -> bool:
        """Check if this kernel can be used with the given problem size."""
        sm_version = get_sm_version()
        if sm_version < 90:
            return False

        tile_m, tile_n = mma_tiler_mn
        # Tile M must be 128 or 256 (for 2CTA on Blackwell)
        if tile_m not in (128, 256):
            return False
        # Tile N must be power-of-2 multiple of 64
        if tile_n not in (64, 128, 192, 256):
            return False
        # M must be aligned to tile_m
        if m % tile_m != 0:
            return False
        # K must be at least 16 (bf16 alignment)
        if k < 16 or k % 16 != 0:
            return False
        # N (after SwiGLU halving) must be aligned
        if n % (tile_n * 2) != 0:
            return False

        cluster_m, cluster_n = cluster_shape_mn
        if cluster_m * cluster_n > 16:
            return False
        if tile_m == 256 and cluster_m % 2 != 0:
            return False

        return True

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute FC1 kernel: gather + grouped GEMM + SwiGLU.

        This is a reference PyTorch implementation that will be replaced by
        a compiled cuteDSL kernel once the cutlass-dsl package supports bf16
        grouped GEMM compilation.

        Args:
            a: Input activations [num_tokens, hidden_size], bf16
            b: Expert weights [num_experts, 2*interm_size, hidden_size], bf16
            tile_idx_to_expert_idx: [num_tiles], int32, expert id per tile
            tile_idx_to_mn_limit: [num_tiles], int32, valid tokens per tile
            permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
            num_non_exiting_tiles: scalar int32, number of valid tiles

        Returns:
            output: [total_permuted_tokens, interm_size], bf16 (after SwiGLU)
        """
        num_tokens, hidden_size = a.shape
        num_experts, n_2x, _ = b.shape
        interm_size = n_2x // 2
        tile_m = self.mma_tiler_mn[0]
        total_permuted = permuted_idx_to_expanded_idx.shape[0]

        output = torch.zeros(
            total_permuted, interm_size, dtype=a.dtype, device=a.device
        )

        n_valid_tiles = num_non_exiting_tiles.item()
        for tile_idx in range(n_valid_tiles):
            expert_idx = tile_idx_to_expert_idx[tile_idx].item()
            mn_limit = tile_idx_to_mn_limit[tile_idx].item()
            row_start = tile_idx * tile_m
            row_end = mn_limit  # mn_limit is absolute cumulative boundary

            # Gather input tokens
            perm_indices = permuted_idx_to_expanded_idx[row_start:row_end]
            gathered_a = a[perm_indices]  # [mn_limit, hidden_size]

            # GEMM: gathered_a @ b[expert_idx].T -> [mn_limit, 2*interm_size]
            gate_up = torch.mm(gathered_a, b[expert_idx].t())

            # SwiGLU: split into gate and up, apply silu(gate) * up
            # Weight layout: [w3(up), w1(gate)] -> first half is up, second is gate
            up_proj = gate_up[:, :interm_size]
            gate_proj = gate_up[:, interm_size:]
            activated = torch.nn.functional.silu(gate_proj) * up_proj

            output[row_start:row_end] = activated.to(output.dtype)

        return output


class FlashMoEBf16GemmFinalizeKernel:
    """
    cuteDSL kernel for FC2: GroupedGEMM + Scale + Scatter-Add.

    This kernel performs:
    1. Per-expert GEMM: intermediate @ W_down^T -> [T_e, H]
    2. Scale by routing weights (token_final_scales)
    3. Scatter-add to output at original token positions

    Memory layout:
    - A (intermediate activations): [total_permuted_tokens, interm_size], bf16
    - B (weights): [num_experts, hidden_size, interm_size], bf16
    - output: [num_tokens, hidden_size], bf16 (scatter-add target, pre-zeroed)
    - token_final_scales: [num_tokens * top_k], float32

    The scatter-add uses expanded_idx_to_permuted_idx to map permuted
    positions back to original token positions, scaling by routing weights.
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        raster_along_m: bool = False,
    ):
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m

    @staticmethod
    def can_implement(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
    ) -> bool:
        """Check if this kernel can be used with the given problem size."""
        sm_version = get_sm_version()
        if sm_version < 90:
            return False

        tile_m, tile_n = mma_tiler_mn
        if tile_m not in (128, 256):
            return False
        if tile_n not in (64, 128, 192, 256):
            return False
        if m % tile_m != 0:
            return False
        if k < 16 or k % 16 != 0:
            return False
        if n % tile_n != 0:
            return False

        cluster_m, cluster_n = cluster_shape_mn
        if cluster_m * cluster_n > 16:
            return False
        if tile_m == 256 and cluster_m % 2 != 0:
            return False

        return True

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Execute FC2 kernel: grouped GEMM + scale + scatter-add.

        This is a reference PyTorch implementation that will be replaced by
        a compiled cuteDSL kernel once the cutlass-dsl package supports bf16
        grouped GEMM compilation.

        Args:
            a: Intermediate activations [total_permuted_tokens, interm_size], bf16
            b: Expert weights [num_experts, hidden_size, interm_size], bf16
            output: Pre-zeroed output buffer [num_tokens, hidden_size], bf16
            tile_idx_to_expert_idx: [num_tiles], int32
            tile_idx_to_mn_limit: [num_tiles], int32
            permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
            expanded_idx_to_permuted_idx: [num_tokens * top_k], int32
            num_non_exiting_tiles: scalar int32
            token_final_scales: [num_tokens * top_k], float32
            top_k: number of experts per token

        Returns:
            output: [num_tokens, hidden_size], bf16, with scatter-added results
        """
        tile_m = self.mma_tiler_mn[0]
        n_valid_tiles = num_non_exiting_tiles.item()

        for tile_idx in range(n_valid_tiles):
            expert_idx = tile_idx_to_expert_idx[tile_idx].item()
            mn_limit = tile_idx_to_mn_limit[tile_idx].item()
            row_start = tile_idx * tile_m
            row_end = mn_limit  # mn_limit is absolute cumulative boundary

            tile_input = a[row_start:row_end]  # [mn_limit, interm_size]

            # GEMM: tile_input @ b[expert_idx].T -> [mn_limit, hidden_size]
            tile_output = torch.mm(tile_input, b[expert_idx].t())

            # Get permuted indices for this tile
            perm_indices = permuted_idx_to_expanded_idx[row_start:row_end]

            # Scale by routing weights and scatter-add
            for local_idx in range(mn_limit):
                expanded_idx = perm_indices[local_idx].item()
                if expanded_idx < 0:
                    continue
                token_idx = expanded_idx // top_k
                scale = token_final_scales[expanded_idx].item()
                scaled_output = tile_output[local_idx] * scale
                output[token_idx] += scaled_output.to(output.dtype)

        return output
