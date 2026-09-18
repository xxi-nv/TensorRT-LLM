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
"""``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``."""

from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .base import CutlassFusedMoEBase
from .eligibility import BF16_ONLY, SmSupport, check_cutlass_leaf
from .identity import (
    KERNEL_HOPPER_GROUPED_GEMM,
    PROVIDER_DEEPGEMM,
    TECHNIQUE_CUDA,
    blockscale_descriptor,
)


@register_moe_impl
class DeepgemmCudaHopperFp8BlockScalesImpl(CutlassFusedMoEBase):
    """``deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales``.

    FP8 block scales on SM90. The identity says ``deepgemm`` because the GEMM
    is the vendored DeepSeek DeepGEMM, not a CUTLASS grouped GEMM: passing
    ``use_deepseek_fp8_block_scale=True`` into ``torch.ops.trtllm.fused_moe``
    makes ``CutlassMoeFCRunner`` route FC1/FC2 through
    ``CutlassFp8BlockScaleGemmRunner::moeGemm``, whose default path JIT-compiles
    and launches ``deep_gemm::runGemm(..., GemmType::GroupedWithOffset, ...)``
    (``cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/``, MIT
    sources under ``cpp/include/tensorrt_llm/deep_gemm/``).

    Only ``kernel_name`` separates it from
    ``deepgemm.cuda.grouped_gemm.fp8_block_scales`` in ``..fused_moe_deepgemm``,
    which reaches the same library through its Python package on SM100/103.
    The MoE orchestration differs -- permute, activation and finalize stay in
    ``CutlassMoeFCRunner`` here -- which is why this leaf lives in the
    ``cutlass`` subpackage and shares its base.

    Moving it next to the SM100/103 leaf, with the ``BACKEND_FAMILY`` and
    config changes that requires, is a follow-up; see the plan for
    TRTLLM-14960 section 4.
    """

    descriptor = blockscale_descriptor(
        PROVIDER_DEEPGEMM,
        TECHNIQUE_CUDA,
        KERNEL_HOPPER_GROUPED_GEMM,
        "DeepGEMM grouped GEMM over FP8 block scales, driven by CutlassMoeFCRunner, SM90.",
    )

    # Taken off the descriptor, not restated: the scheduler reads these three
    # attributes and the registry publishes the descriptor, so a second literal
    # would let the two drift apart.
    scheduler_kind = descriptor.scheduler_kind
    capabilities = descriptor.capabilities
    input_requirement = descriptor.input_requirement

    sm_support = SmSupport(allowed=frozenset({90}))
    #: The block-scale GEMM runner only has BF16 A / output instantiations.
    supported_dtypes = BF16_ONLY

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_cutlass_leaf(cls, p, d)
