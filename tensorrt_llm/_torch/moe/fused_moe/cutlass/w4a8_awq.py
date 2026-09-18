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
"""``trtllm.cutlass.grouped_gemm.w4a8_awq``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


@register_moe_impl
class TrtllmCutlassW4a8AwqImpl(CutlassFusedMoEBase):
    """``trtllm.cutlass.grouped_gemm.w4a8_awq``.

    The DeepSeek-R1 W4A8 mixed-precision recipe on Hopper. No other MoE
    implementation serves this format, so this leaf is the only one.
    """

    descriptor = cutlass_descriptor(
        "w4a8_awq",
        "CUTLASS grouped GEMM over INT4 group-scaled weights with FP8 activations.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(allowed=frozenset({89, 90}))
    supported_dtypes = HP_DTYPES

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)
