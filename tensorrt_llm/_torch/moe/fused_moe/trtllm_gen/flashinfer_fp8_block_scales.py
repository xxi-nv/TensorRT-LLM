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
"""``flashinfer.trtllm_gen.fused_moe.fp8_block_scales``."""

from ..fused_moe_trtllm_gen import (
    PROVIDER_FLASHINFER,
    check_flashinfer_provider,
    check_trtllm_gen_leaf,
    trtllm_gen_descriptor,
)
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from .bases import TRTLLMGenFp8BlockScalesBase
from .mixins import FlashinferProviderMixin


@register_moe_impl
class FlashinferTrtllmGenFp8BlockScalesImpl(FlashinferProviderMixin, TRTLLMGenFp8BlockScalesBase):
    """``flashinfer.trtllm_gen.fused_moe.fp8_block_scales``.

    No alignment gate: the old check reached its unconditional ``return True``
    for this format, because the padding rules it enforced belong to the FP4
    weight layouts.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER,
        "fp8_block_scales",
        "FlashInfer's TRTLLM-Gen FP8 block-scale fused MoE, SM100 family.",
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d, check_flashinfer_provider(cls, p, d))
