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
"""``trtllm.cutlass.grouped_gemm.w4a16_nvfp4``."""

from typing import Optional, Tuple, Union

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERunContext,
    require_comm_plan,
)
from ..impl_identity import register_moe_impl
from ..quantization import W4A16NVFP4CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .grouped_gemm import local_expert_ids, run_dequantized_grouped_gemm
from .identity import cutlass_descriptor
from .input_quant import quantize_noop


@register_moe_impl
class TrtllmCutlassW4a16Nvfp4Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a16_nvfp4``.

    Weights stay NVFP4 on disk but are dequantized into a high-precision
    workspace per forward, so what finally runs is the unquantized
    kernel. The SM range therefore tracks that path's limits, not NVFP4
    tensor-core support.

    AUTO only routes here outside SM90-99 (MARLIN) and SM120/121
    (CuteDSL B12x); see ``ModelConfig.resolve_moe_backend``.
    """

    descriptor = cutlass_descriptor(
        "w4a16_nvfp4",
        "NVFP4 weights dequantized to the activation dtype every forward, SM80+.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(minimum=80)
    supported_dtypes = HP_DTYPES

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return W4A16NVFP4CutlassFusedMoEMethod()

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        return quantize_noop(self, x, post_quant_comm)

    def run_moe(self, ctx: MoERunContext, *, workspace: Optional[dict] = None) -> torch.Tensor:
        """Active-mask dequant into a high-precision workspace, then bf16 GEMM.

        CUDA-graph capturable: the workspace is static and the scatter is
        in-bounds because the ids are clamped to the local range.
        """
        del workspace  # Cutlass allocates its own intermediates.
        plan = require_comm_plan(self, ctx)
        output_dtype = ctx.output_dtype if ctx.output_dtype is not None else ctx.x.dtype
        # Non-local tokens collapse onto a boundary expert (one extra dequant
        # per rank); the op below still receives the original global ids and
        # does its own remap.
        local_ids = local_expert_ids(self, ctx, plan.enable_alltoall)
        w3_w1_hp, w2_hp = self.quant_method.dequant_active_experts_to_hp(
            self, local_ids, output_dtype
        )
        return run_dequantized_grouped_gemm(self, ctx, w3_w1_hp, w2_hp, output_dtype)
