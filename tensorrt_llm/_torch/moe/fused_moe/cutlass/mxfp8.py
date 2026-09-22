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
"""``trtllm.cutlass.grouped_gemm.mxfp8``."""

from typing import Optional, Tuple, Union

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..quantization import MXFP8CutlassFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .grouped_gemm import GroupedGemmFlags, run_grouped_gemm
from .identity import cutlass_descriptor
from .input_quant import quantize_mxfp8


@register_moe_impl
class TrtllmCutlassMxfp8Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.mxfp8``.

    W8A8 e4m3 x e4m3 with UE8M0 1x32 block scales on both sides. Reuses
    the MXFP4 family's activation quantization; only the weight element
    widens from fp4 to fp8. No other MoE implementation serves this
    format.
    """

    descriptor = cutlass_descriptor(
        "mxfp8",
        "CUTLASS grouped GEMM over MXFP8 weights and dynamic MXFP8 activations.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(allowed=frozenset({100, 103}))
    supported_dtypes = HP_DTYPES

    supports_gptoss_style = True
    #: Kernel-selection flags for this format.
    GROUPED_GEMM_FLAGS = GroupedGemmFlags(use_mxfp8_act_scaling=True, use_mxfp8_weight_scaling=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return MXFP8CutlassFusedMoEMethod()

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        return quantize_mxfp8(self, x, post_quant_comm)

    def run_moe(self, ctx: MoERunContext, *, workspace: Optional[dict] = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        return run_grouped_gemm(self, ctx, self.GROUPED_GEMM_FLAGS)
