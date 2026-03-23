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
"""FlashMoE bf16 Grouped GEMM using CUTLASS GroupedGemmKernel.

Wraps the CUTLASS cuteDSL GroupedGemmKernel for bf16 grouped GEMM operations
needed by FlashMoE:

1. FC1: Gather + GroupedGEMM + SwiGLU
   - Pre-gather tokens using permuted indices from moe_sort
   - Grouped GEMM: per-expert [T_e, H] x [H, 2*I] -> [T_e, 2*I]
   - SwiGLU activation applied after GEMM

2. FC2: GroupedGEMM + Scale + Scatter-Add (Finalize)
   - Grouped GEMM: per-expert [T_e, I] x [I, H] -> [T_e, H]
   - Scale by routing weights
   - Scatter-add to original token positions

The CUTLASS GroupedGemmKernel (from cutlass-dsl >= 4.3) supports:
- bf16/fp16 A/B inputs with fp32 accumulator
- Per-group pointer arrays with variable problem sizes
- Persistent tile scheduling on SM100 (Blackwell)
- Warp specialization (TMA + MMA + Epilogue warps)

Architecture:
- SM100 (Blackwell): tcgen05.mma-based bf16 GEMM (TMEM accumulator)
- SM90 (Hopper): WGMMA-based bf16 GEMM (planned, not yet implemented)

Integration approach:
- Convert FlashMoE's contiguous weight layout [num_experts, N, K] to
  per-group pointer/stride/shape arrays expected by GroupedGemmKernel
- Pre-gather activations using moe_sort indices before GEMM
- Apply SwiGLU/scale+scatter-add as post-GEMM epilogues
"""

from typing import Optional, Tuple

import torch

from ..._utils import get_sm_version


def _check_bf16_kernel_support() -> Tuple[bool, Optional[str]]:
    """Check if the current GPU supports bf16 cuteDSL kernels."""
    sm_version = get_sm_version()
    if sm_version < 100:
        return False, (f"FlashMoE bf16 cuteDSL requires SM >= 100 (Blackwell), got SM {sm_version}")
    return True, None


def _check_cutlass_dsl_available() -> Tuple[bool, Optional[str]]:
    """Check if cutlass-dsl is available with GroupedGemmKernel support."""
    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        import cutlass.utils.blackwell_helpers  # noqa: F401

        return True, None
    except ImportError as e:
        return False, f"cutlass-dsl not available: {e}"


class FlashMoEBf16GroupedGemmWrapper:
    """Wrapper around CUTLASS GroupedGemmKernel for FlashMoE bf16 GEMM.

    Converts FlashMoE's contiguous expert weight layout to the per-group
    pointer/stride/shape arrays expected by the CUTLASS grouped GEMM kernel.

    For FlashMoE, all experts share the same N and K dimensions (only M varies).
    The A activations are pre-gathered into contiguous per-expert segments using
    moe_sort indices.

    Usage:
        wrapper = FlashMoEBf16GroupedGemmWrapper(mma_tiler_mn=(128, 128))
        # Pre-gather A using moe_sort indices
        gathered_a = input[permuted_idx_to_expanded_idx // top_k]
        # Set up per-expert problem sizes (M varies per expert)
        # Run grouped GEMM
        c = wrapper(gathered_a, weights_3d, expert_ranges, ...)
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        use_2cta_instrs: bool = False,
    ):
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.use_2cta_instrs = use_2cta_instrs
        self._compiled_kernel = None
        self._kernel_instance = None

    def _ensure_compiled(self):
        """Lazily compile the CUTLASS GroupedGemmKernel on first use."""
        if self._compiled_kernel is not None:
            return

        import cutlass

        # Import the GroupedGemmKernel from the CUTLASS examples path
        # This requires cutlass-dsl >= 4.3 which ships the grouped_gemm example
        from cutlass.examples.blackwell.grouped_gemm import GroupedGemmKernel

        self._kernel_instance = GroupedGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
        )

    @staticmethod
    def can_implement() -> Tuple[bool, Optional[str]]:
        """Check if this wrapper can be used."""
        ok, reason = _check_bf16_kernel_support()
        if not ok:
            return ok, reason
        ok, reason = _check_cutlass_dsl_available()
        if not ok:
            return ok, reason
        return True, None


class FlashMoEBf16GatherGemmSwigluKernel:
    """FC1: Gather + GroupedGEMM + SwiGLU for bf16 FlashMoE.

    Uses expert-merged GEMM: tiles are grouped by expert from moe_sort,
    so all tiles for the same expert are processed in one torch.mm() call.
    This reduces CUDA kernel launches from O(num_tiles) to O(num_active_experts).

    Memory layout:
    - A (input activations): [num_tokens, hidden_size], bf16, row-major
    - B (weights): [num_experts, 2*intermediate_size, hidden_size], bf16
    - C (output): [total_permuted_tokens, intermediate_size], bf16

    Tile management (from moe_sort):
    - tile_idx_to_expert_idx: maps each tile to its LOCAL expert index
    - tile_idx_to_mn_limit: absolute cumulative boundary per tile
    - permuted_idx_to_expanded_idx: gather indices for A
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
        num_groups: int,
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
        # N (2*intermediate_size) must be aligned to 2*tile_n for SwiGLU
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
        top_k: int = 1,
        is_gated_activation: bool = True,
    ) -> torch.Tensor:
        """Execute FC1: gather + grouped GEMM + SwiGLU.

        Uses expert-merged approach: merges consecutive tiles belonging
        to the same expert into a single GEMM call.

        Args:
            a: Input activations [num_tokens, hidden_size], bf16
            b: Expert weights [num_experts, 2*interm_size, hidden_size], bf16
            tile_idx_to_expert_idx: [num_tiles], int32, local expert ids
            tile_idx_to_mn_limit: [num_tiles], int32, absolute boundaries
            permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
            num_non_exiting_tiles: scalar int32
            top_k: number of experts per token
            is_gated_activation: whether to apply SwiGLU

        Returns:
            output: [total_permuted_tokens, interm_size], bf16
        """
        return torch.ops.trtllm.flashmoe_bf16_gather_gemm_swiglu(
            input=a,
            weight=b,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            top_k=top_k,
            tile_size=self.mma_tiler_mn[0],
            is_gated_activation=is_gated_activation,
        )


class FlashMoEBf16GemmFinalizeKernel:
    """FC2: GroupedGEMM + Scale + Scatter-Add for bf16 FlashMoE.

    Uses expert-merged GEMM: tiles are grouped by expert from moe_sort,
    so all tiles for the same expert are processed in one torch.mm() call.

    Memory layout:
    - A (intermediate activations): [total_permuted_tokens, interm_size], bf16
    - B (weights): [num_experts, hidden_size, interm_size], bf16
    - output: [num_tokens, hidden_size], bf16 (scatter-add target, pre-zeroed)
    - token_final_scales: [num_tokens * top_k], float32
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
        num_groups: int,
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
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Execute FC2: grouped GEMM + scale + scatter-add.

        Uses expert-merged approach: merges consecutive tiles belonging
        to the same expert into a single GEMM call.

        Args:
            a: Intermediate activations [total_permuted_tokens, interm_size], bf16
            b: Expert weights [num_experts, hidden_size, interm_size], bf16
            output: Pre-zeroed output buffer [num_tokens, hidden_size], bf16
            tile_idx_to_expert_idx: [num_tiles], int32
            tile_idx_to_mn_limit: [num_tiles], int32
            permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
            num_non_exiting_tiles: scalar int32
            token_final_scales: [num_tokens * top_k], float32
            top_k: number of experts per token

        Returns:
            output: [num_tokens, hidden_size], bf16, with scatter-added results
        """
        torch.ops.trtllm.flashmoe_bf16_gemm_finalize_inplace(
            input=a,
            weight=b,
            output=output,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            top_k=top_k,
            tile_size=self.mma_tiler_mn[0],
        )
        return output
