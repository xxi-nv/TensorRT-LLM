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
"""``flashinfer.trtllm_gen.fused_moe.nvfp4``."""

from typing import Optional

from ..fused_moe_trtllm_gen import (
    PROVIDER_FLASHINFER,
    check_flashinfer_provider,
    check_mxfp4_flashinfer_shape,
    check_trtllm_gen_leaf,
    nvfp4_needs_padded_method,
    trtllm_gen_descriptor,
)
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem
from ..impl_identity import register_moe_impl
from ..quantization import NVFP4TRTLLMGenFusedMoEMethod
from .bases import TRTLLMGenNvfp4Base
from .mixins import FlashinferProviderMixin


@register_moe_impl
class FlashinferTrtllmGenNvfp4Impl(FlashinferProviderMixin, TRTLLMGenNvfp4Base):
    """``flashinfer.trtllm_gen.fused_moe.nvfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER, "nvfp4", "FlashInfer's TRTLLM-Gen NVFP4 fused MoE, SM100 family."
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(
            cls, p, d, check_flashinfer_provider(cls, p, d), cls._check_shape(p, d)
        )

    @classmethod
    def _check_shape(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
        """Alignment gate, and only for the padded quant method.

        The unpadded ``NVFP4TRTLLMGenFusedMoEBaseMethod`` lays out any shape,
        so it was admitted unconditionally and still is.

        The alignments are the padded method's *class* attributes, not what
        ``resolve_alignments`` would pick for this shape. That is what the old
        instance-level check read -- it built a fresh quant method in
        ``__init__``, before ``create_weights`` replaced the class attributes
        with resolved instance ones -- and reading the resolved values here
        would reject shapes this provider currently serves. Left as it was on
        purpose: this change is a split, not a fix.
        """
        if not nvfp4_needs_padded_method(p.activation_type, "alpha" in p.activation_constants):
            return None
        return check_mxfp4_flashinfer_shape(
            cls,
            p,
            d,
            weight_alignment=NVFP4TRTLLMGenFusedMoEMethod.weight_alignment,
            input_hidden_alignment=NVFP4TRTLLMGenFusedMoEMethod.input_hidden_alignment,
        )
