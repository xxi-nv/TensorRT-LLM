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

"""BF16 Contiguous Gather Grouped GEMM + SwiGLU Fusion (FC1) for Blackwell.

FC1 operator: Gather + GroupedGEMM + SwiGLU activation.

Architecture:
  1. Pre-gather: tokens are gathered from input using permuted indices
  2. Grouped GEMM: uses Sm100Bf16ContiguousGroupedGemmKernel
     [total_permuted_tokens, H] x [2*I, H, num_experts] -> [total_permuted_tokens, 2*I]
  3. SwiGLU: silu(gate) * up, halving N from 2*I to I

This is an operator-level fusion wrapping the base bf16 contiguous grouped
GEMM kernel. Full kernel-level gather fusion (LDGSTS gather warps as in
blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py) is planned
for a follow-up optimization.
"""

from typing import Optional, Tuple

import torch

from .bf16_contiguous_grouped_gemm import Sm100Bf16ContiguousGroupedGemmKernel


class Sm100Bf16GatherGroupedGemmSwigluOp:
    """FC1: Gather + bf16 Grouped GEMM + SwiGLU.

    Uses the base Sm100Bf16ContiguousGroupedGemmKernel for the GEMM,
    with pre-gather and post-SwiGLU as PyTorch operations.

    :param mma_tiler_mn: MMA tile shape (M, N). M is the tile size along
        the token dimension, N along the intermediate dimension.
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
        input_tokens: torch.Tensor,
        weight: torch.Tensor,
        gathered_input: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        top_k: int,
        is_gated_activation: bool = True,
        stream: Optional[int] = None,
    ) -> torch.Tensor:
        """Execute FC1: gather + grouped GEMM + SwiGLU.

        Args:
            input_tokens: [num_tokens, hidden_size], bf16
            weight: [num_local_experts, 2*intermediate_size, hidden_size], bf16
            gathered_input: Pre-allocated buffer [total_permuted_tokens, hidden_size], bf16
            tile_idx_to_expert_idx: [num_tiles], int32
            num_non_exiting_tiles: [1], int32
            permuted_idx_to_expanded_idx: [total_permuted_tokens], int32
            top_k: number of experts per token
            is_gated_activation: whether to apply SwiGLU
            stream: Optional CUDA stream handle

        Returns:
            output: [total_permuted_tokens, intermediate_size], bf16
        """
        self._ensure_kernel()

        total_permuted = permuted_idx_to_expanded_idx.shape[0]
        n_2i = weight.shape[1]  # 2 * intermediate_size
        hidden_size = weight.shape[2]
        num_experts = weight.shape[0]

        # Step 1: Pre-gather tokens into contiguous buffer
        token_indices = permuted_idx_to_expanded_idx // top_k
        gathered_input[:total_permuted].copy_(input_tokens[token_indices])

        # Step 2: Grouped GEMM via cuteDSL kernel
        # A: [total_permuted_tokens, hidden_size] -> CuTe layout [M, K, 1]
        # B: [2*intermediate_size, hidden_size, num_experts] -> CuTe layout [N, K, L]
        # C: [total_permuted_tokens, 2*intermediate_size] -> CuTe layout [M, N, 1]
        gate_up_output = torch.empty(
            total_permuted, n_2i, dtype=input_tokens.dtype, device=input_tokens.device
        )

        # Alpha = 1.0 for each expert
        alpha = torch.ones(num_experts, dtype=torch.float32, device=input_tokens.device)

        # Get CUDA stream
        if stream is None:
            from cuda.bindings import driver as cuda_driver

            err, cu_stream = cuda_driver.cuStreamCreate(0)
            # Use the current PyTorch stream
            cu_stream = torch.cuda.current_stream().cuda_stream
        else:
            cu_stream = stream

        # Call the kernel via the wrapper
        import cutlass
        from cutlass.cute.typing import Pointer

        self._kernel.wrapper(
            Pointer(gathered_input[:total_permuted].data_ptr(), cutlass.BFloat16),
            Pointer(weight.data_ptr(), cutlass.BFloat16),
            Pointer(gate_up_output.data_ptr(), cutlass.BFloat16),
            Pointer(alpha.data_ptr(), cutlass.Float32),
            Pointer(tile_idx_to_expert_idx.data_ptr(), cutlass.Int32),
            Pointer(num_non_exiting_tiles.data_ptr(), cutlass.Int32),
            m=total_permuted,
            n=n_2i,
            k=hidden_size,
            l=num_experts,
            tile_size=self.mma_tiler_mn[0],
            max_active_clusters=1,
            stream=cu_stream,
        )

        # Step 3: SwiGLU activation
        # Weight layout: [up_proj, gate_proj] concatenated along N
        up_proj, gate_proj = gate_up_output.chunk(2, dim=-1)
        if is_gated_activation:
            output = torch.nn.functional.silu(gate_proj) * up_proj
        else:
            output = up_proj

        return output
