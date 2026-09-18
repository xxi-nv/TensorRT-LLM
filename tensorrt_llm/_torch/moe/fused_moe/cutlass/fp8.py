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
"""``trtllm.cutlass.grouped_gemm.fp8``."""

from typing import Optional, Tuple, Union

import torch

from ....utils import Fp4QuantizedTensor
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERunContext
from ..impl_identity import register_moe_impl
from ..quantization import FP8QDQFusedMoEMethod
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES_WITH_FP32, SmSupport, check_cutlass_leaf
from .grouped_gemm import DEFAULT_FLAGS, run_grouped_gemm
from .identity import CUTLASS_LORA_CAPABILITIES, cutlass_descriptor
from .input_quant import quantize_static_e4m3


@register_moe_impl
class TrtllmCutlassFp8Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.fp8``.

    Per-tensor FP8, quantized and dequantized around the GEMM rather than
    block-scaled. The second of the two leaves that fuse routed-expert
    LoRA; the C++ op rejects LoRA for every other format.
    """

    descriptor = cutlass_descriptor(
        "fp8",
        "CUTLASS grouped GEMM over per-tensor FP8 (qdq) weights, SM89+.",
        capabilities=CUTLASS_LORA_CAPABILITIES,
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(minimum=89)
    supported_dtypes = HP_DTYPES_WITH_FP32

    #: Kernel-selection flags for this format.
    GROUPED_GEMM_FLAGS = DEFAULT_FLAGS

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)

    def _get_quant_method(self) -> object:
        return FP8QDQFusedMoEMethod()

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        return quantize_static_e4m3(self, x, post_quant_comm)

    def run_moe(self, ctx: MoERunContext, *, workspace: Optional[dict] = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        return run_grouped_gemm(self, ctx, self.GROUPED_GEMM_FLAGS)
