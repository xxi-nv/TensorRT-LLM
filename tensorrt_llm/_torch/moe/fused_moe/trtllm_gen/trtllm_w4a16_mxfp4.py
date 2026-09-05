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
"""``trtllm.trtllm_gen.fused_moe.w4a16_mxfp4``."""

from ..fused_moe_trtllm_gen import PROVIDER_TRTLLM, check_trtllm_gen_leaf, trtllm_gen_descriptor
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .bases import TRTLLMGenW4a16Mxfp4Base
from .mixins import TrtllmProviderMixin


@register_moe_impl
class TrtllmTrtllmGenW4a16Mxfp4Impl(TrtllmProviderMixin, TRTLLMGenW4a16Mxfp4Base):
    """``trtllm.trtllm_gen.fused_moe.w4a16_mxfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM,
        "w4a16_mxfp4",
        "TRTLLM-Gen batched-GEMM cubins over MXFP4 weights with bf16 activations.",
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)
