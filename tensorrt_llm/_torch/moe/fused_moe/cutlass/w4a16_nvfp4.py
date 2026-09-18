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

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .base import CutlassFusedMoEBase
from .eligibility import HP_DTYPES, SmSupport, check_cutlass_leaf
from .identity import cutlass_descriptor


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
