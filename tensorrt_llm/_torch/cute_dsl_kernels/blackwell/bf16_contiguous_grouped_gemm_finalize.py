# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""BF16 Contiguous Grouped GEMM + Finalize Fusion (FC2) for Blackwell.

FC2 operator: GroupedGEMM + Scale + Scatter-Add.

Architecture:
  1. Grouped GEMM: uses Sm100Bf16ContiguousGroupedGemmKernel
     [total_permuted_tokens, I] x [H, I, num_experts] -> [total_permuted_tokens, H]
  2. Scale: multiply by routing weights per token
  3. Scatter-Add: accumulate to original token positions in output

This is an operator-level fusion wrapping the base bf16 contiguous grouped
GEMM kernel. Full kernel-level scatter-add fusion (as in
blockscaled_contiguous_grouped_gemm_finalize_fusion.py) is planned for
a follow-up optimization with LDMC comm warps in Phase V3.2.
"""

from typing import Optional, Tuple

import torch

from .bf16_contiguous_grouped_gemm import Sm100Bf16ContiguousGroupedGemmKernel


class Sm100Bf16GroupedGemmFinalizeOp:
    """FC2: bf16 Grouped GEMM + Scale + Scatter-Add.

    Uses the base Sm100Bf16ContiguousGroupedGemmKernel for the GEMM,
    with post-scale and scatter-add as PyTorch operations.

    :param mma_tiler_mn: MMA tile shape (M, N).
    :param cluster_shape_mn: Cluster shape (M, N).
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
    ):
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self._kernel = None

    def _ensure_kernel(self):
        """Lazily create kernel instance."""
        if self._kernel is not None:
            return
        self._kernel = Sm100Bf16ContiguousGroupedGemmKernel(
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
        )

    def __call__(
        self,
        intermediate: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        token_final_scales: torch.Tensor,
        top_k: int,
        stream: Optional[int] = None,
    ) -> torch.Tensor:
        """Execute FC2: grouped GEMM + scale + scatter-add.

        Args:
            intermediate: [total_permuted_tokens, intermediate_size], bf16
            weight: [num_local_experts, hidden_size, intermediate_size], bf16
            output: [num_tokens, hidden_size], bf16 (pre-zeroed, modified in-place)
            tile_idx_to_expert_idx: [num_tiles], int32
            num_non_exiting_tiles: [1], int32
            permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
            token_final_scales: [num_tokens * top_k] or [num_tokens, top_k], float32
            top_k: number of experts per token
            stream: Optional CUDA stream handle

        Returns:
            output: [num_tokens, hidden_size], bf16, with scatter-added results
        """
        self._ensure_kernel()

        total_permuted = permuted_idx_to_expanded_idx.shape[0]
        hidden_size = weight.shape[1]
        interm_size = weight.shape[2]
        num_experts = weight.shape[0]

        # Step 1: Grouped GEMM via cuteDSL kernel
        # A: [total_permuted_tokens, interm_size] -> CuTe layout [M, K, 1]
        # B: [hidden_size, interm_size, num_experts] -> CuTe layout [N, K, L]
        # C: [total_permuted_tokens, hidden_size] -> CuTe layout [M, N, 1]
        gemm_output = torch.empty(
            total_permuted, hidden_size, dtype=intermediate.dtype, device=intermediate.device
        )

        # Alpha = 1.0 for each expert
        alpha = torch.ones(num_experts, dtype=torch.float32, device=intermediate.device)

        # Get CUDA stream
        if stream is None:
            cu_stream = torch.cuda.current_stream().cuda_stream
        else:
            cu_stream = stream

        # Call the kernel via the wrapper
        import cutlass
        from cutlass.cute.typing import Pointer

        self._kernel.wrapper(
            Pointer(intermediate[:total_permuted].data_ptr(), cutlass.BFloat16),
            Pointer(weight.data_ptr(), cutlass.BFloat16),
            Pointer(gemm_output.data_ptr(), cutlass.BFloat16),
            Pointer(alpha.data_ptr(), cutlass.Float32),
            Pointer(tile_idx_to_expert_idx.data_ptr(), cutlass.Int32),
            Pointer(num_non_exiting_tiles.data_ptr(), cutlass.Int32),
            m=total_permuted,
            n=hidden_size,
            k=interm_size,
            l=num_experts,
            tile_size=self.mma_tiler_mn[0],
            max_active_clusters=1,
            stream=cu_stream,
        )

        # Step 2: Scale by routing weights and scatter-add to output
        flat_scales = token_final_scales.float().view(-1)
        perm_indices = permuted_idx_to_expanded_idx[:total_permuted]
        token_indices = perm_indices // top_k
        scales = flat_scales[perm_indices].unsqueeze(1)
        scaled_output = gemm_output * scales.to(gemm_output.dtype)
        output.index_add_(0, token_indices, scaled_output)

        return output
