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
"""``trtllm.triton.blockscale_gemm.fp8_block_scales``."""

from typing import Optional, Tuple, Union

import torch

from ....utils import Fp4QuantizedTensor
from ..fused_moe_triton_fp8_block_scale import run_triton_fp8_block_scale_moe
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERunContext,
    require_comm_plan,
)
from ..impl_identity import register_moe_impl
from ..quantization import DeepSeekFP8BlockScalesFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import BF16_ONLY, SmSupport, check_cutlass_leaf
from .grouped_gemm import local_expert_ids
from .identity import (
    KERNEL_BLOCKSCALE_GEMM,
    PROVIDER_TRTLLM,
    TECHNIQUE_TRITON,
    blockscale_descriptor,
)
from .input_quant import quantize_noop


@register_moe_impl
class TrtllmTritonFp8BlockScalesImpl(CutlassFusedMoEBase):
    """``trtllm.triton.blockscale_gemm.fp8_block_scales``.

    FP8 block scales on SM120, where CUTLASS TMA fails for large token counts
    (``cuTensorMapEncodeTiled`` limitations). The whole MoE runs in
    ``..fused_moe_triton_fp8_block_scale.run_triton_fp8_block_scale_moe`` and
    never enters C++, which is why neither ``cutlass`` nor ``deepgemm`` appears
    in this identity: it shares no kernel with either.

    Same quantization format as
    ``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``, different kernel and
    different SM -- the two used to be one legacy name whose choice was made by
    a ``get_sm_version() == 120`` branch inside the forward path. Splitting the
    identity is what lets that branch go away and lets the resolution report
    name the kernel that actually ran.

    It shares ``CutlassFusedMoEBase`` for construction and the weight lifecycle
    only; no ``GROUPED_GEMM_FLAGS`` here, because it never issues the CUTLASS op.
    """

    descriptor = blockscale_descriptor(
        PROVIDER_TRTLLM,
        TECHNIQUE_TRITON,
        KERNEL_BLOCKSCALE_GEMM,
        "Internal Triton FP8 block-scale MoE, SM120.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(allowed=frozenset({120}))
    #: The Triton kernel is instantiated for BF16 activations only.
    supported_dtypes = BF16_ONLY

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return DeepSeekFP8BlockScalesFusedMoEMethod()

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pass through: the Triton kernel quantizes activations itself."""
        del kwargs
        return quantize_noop(self, x, post_quant_comm)

    def run_moe(self, ctx: MoERunContext, *, workspace: Optional[dict] = None) -> torch.Tensor:
        """Run the internal Triton block-scale MoE. Never enters C++.

        This is the branch that used to live inside the shared ``run_moe`` as
        ``if has_deepseek_fp8_block_scales and get_sm_version() == 120``.
        """
        del workspace  # The Triton kernel allocates its own intermediates.
        plan = require_comm_plan(self, ctx)
        enable_alltoall = plan.enable_alltoall

        # ``forward_chunk`` leaves token_final_scales as None when the router
        # weights were already folded into x; ones make the per-token scaling a
        # no-op rather than a missing operand.
        token_final_scales = ctx.token_final_scales
        if token_final_scales is None:
            token_final_scales = torch.ones_like(ctx.token_selected_experts, dtype=torch.float32)

        # The Triton kernel indexes local experts (0 .. expert_size-1), so remap
        # and zero-scale any token-expert pair owned by another rank to suppress
        # its contribution.
        local_n = self.expert_size_per_partition
        ids = ctx.token_selected_experts
        if enable_alltoall:
            is_local = ids < local_n
        else:
            is_local = (ids >= self.slot_start) & (ids < self.slot_start + local_n)
        local_ids = local_expert_ids(self, ctx, enable_alltoall)
        local_scales = token_final_scales * is_local.to(token_final_scales.dtype)

        return run_triton_fp8_block_scale_moe(
            ctx.x,
            local_ids,
            local_scales,
            self.w3_w1_weight,
            self.quant_scales.fc_weight_scales,
            self.w2_weight,
            self.quant_scales.proj_weight_scales,
            activation_type=self.activation_type,
            output_dtype=ctx.output_dtype,
        )
