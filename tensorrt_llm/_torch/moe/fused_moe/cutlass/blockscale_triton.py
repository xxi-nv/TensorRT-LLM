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

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .base import CutlassFusedMoEBase
from .eligibility import BF16_ONLY, SmSupport, check_cutlass_leaf
from .identity import (
    KERNEL_BLOCKSCALE_GEMM,
    PROVIDER_TRTLLM,
    TECHNIQUE_TRITON,
    blockscale_descriptor,
)


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
