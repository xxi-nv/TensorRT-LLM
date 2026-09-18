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
"""``trtllm.cutlass.grouped_gemm.nvfp4``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .base import CutlassFusedMoEBase
from .eligibility import (
    HP_DTYPES_WITH_FP8,
    SmSupport,
    check_cutlass_leaf,
    check_nvfp4_shard_alignment,
)
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassNvfp4Impl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.nvfp4``.

    W4A4: both operands are FP4, so ``quantize_input`` quantizes the
    activations too. Admits the NVFP4_AWQ / NVFP4_ARC calibration
    aliases, which ``MoEProblem.identity_quant`` folds onto ``nvfp4``.

    Serves MiniMax-style SwigluBias but not a real 1-D gpt-oss expert
    bias, because the NVFP4 weight pad asserts 2-D.
    """

    descriptor = cutlass_descriptor(
        "nvfp4",
        "CUTLASS grouped GEMM over NVFP4 weights and activations, Blackwell.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(allowed=frozenset({100, 103, 107, 120, 121}))
    supported_dtypes = HP_DTYPES_WITH_FP8

    supports_gptoss_style = True
    rejects_gptoss_expert_bias = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d, check_nvfp4_shard_alignment)
