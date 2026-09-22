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
"""``trtllm.cutlass.grouped_gemm.w4a16_mxfp4``."""

from typing import Optional, Tuple, Union

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..quantization import WFP4A16FusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .grouped_gemm import GroupedGemmFlags, run_grouped_gemm
from .identity import cutlass_descriptor
from .input_quant import quantize_noop


@register_moe_impl
class TrtllmCutlassW4a16Mxfp4Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a16_mxfp4``.

    SM90 only -- the TRTLLM-Gen leaves cover this format on the Blackwell
    family, and the weight method raises outside SM90. Pads hidden and
    intermediate sizes up to 128 at construction.
    """

    descriptor = cutlass_descriptor(
        "w4a16_mxfp4",
        "CUTLASS grouped GEMM over MXFP4 weights with high-precision activations, SM90.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(allowed=frozenset({90}))
    supported_dtypes = HP_DTYPES

    supports_gptoss_style = True
    #: Kernel-selection flags for this format.
    GROUPED_GEMM_FLAGS = GroupedGemmFlags(weight_dtype=torch.uint8, use_w4_group_scaling=True)

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return WFP4A16FusedMoEMethod()

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        return quantize_noop(self, x, post_quant_comm)

    def run_moe(self, ctx: MoERunContext, *, workspace: Optional[dict] = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        # Pad to the 128-aligned hidden size here rather than in
        # ``quantize_input``, so dispatch sends unpadded tensors and does not
        # overallocate the NVLink workspace.
        x = ctx.x
        pad_size = self.hidden_size - x.shape[-1]
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size))
        return run_grouped_gemm(self, ctx, self.GROUPED_GEMM_FLAGS, x=x)
