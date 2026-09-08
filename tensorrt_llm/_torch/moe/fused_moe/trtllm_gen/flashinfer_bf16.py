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
"""``flashinfer.trtllm_gen.fused_moe.none`` -- the unquantized path."""

from typing import Optional, Union

import torch

from ....utils import ActivationType
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
)
from ..impl_environment import MoEDep
from ..impl_identity import register_moe_impl
from ..interface import _reject
from ..quantization import BF16TRTLLMGenFusedMoEMethod
from ..routing import DeepSeekV3MoeRoutingMethod
from .eligibility import check_trtllm_gen_leaf
from .family import TrtllmGenFusedMoEBase
from .identity import PROVIDER_FLASHINFER, FlashinferProviderTraits, trtllm_gen_descriptor
from .kernel_inputs import prepare_kernel_inputs, to_trtllm_gen_act_type


@register_moe_impl
class FlashinferTrtllmGenBf16Impl(FlashinferProviderTraits, TrtllmGenFusedMoEBase):
    """``flashinfer.trtllm_gen.fused_moe.none`` -- the unquantized path.

    FlashInfer-exclusive, and the one leaf reached without the opt-in flag:
    ``trtllm_bf16_moe`` has no counterpart in TRT-LLM's own cubins, so there is
    no native leaf to prefer and nothing for a flag to switch between. It is
    also the only leaf whose kernel does not route internally, which is what
    ``_requires_separated_routing`` below says.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER,
        "none",
        "FlashInfer's TRTLLM-Gen bf16 fused MoE, unquantized, SM100 family.",
    )

    #: What the FlashInfer bf16 kernels implement. Lives on this leaf rather
    #: than beside the family's shared activation declaration because it is the
    #: only leaf that reads it, from both sides: the gate below and
    #: ``_check_configs``.
    _BF16_SUPPORTED_ACTIVATIONS = {
        ActivationType.Swiglu,
        ActivationType.Relu2,
    }

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d, cls._check_bf16_path(p, d))

    def _requires_separated_routing(self) -> bool:
        """``trtllm_bf16_moe`` has no internal routing, so top-k comes from the host.

        The override is the whole content of what used to be
        ``self.use_flashinfer and self._is_unquantized_path()`` on the family
        base: exactly one of the eleven leaves runs a kernel with no internal
        routing, and the other ten take the default of False.

        The DeepSeekV3 exception stays a runtime test because routing is a
        per-layer argument, not part of the identity -- that kernel's separated
        variant has accuracy issues, so its fused form is used instead.
        """
        return not isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod)

    def _check_configs(self) -> None:
        """The unquantized kernels take no activation constants at all.

        Restates the gate above against the constructed instance, because
        direct construction reaches here without a resolution query, and adds
        the clamp -- which ``swiglu_gptoss_style`` does not cover.
        """
        assert self.activation_type in self._BF16_SUPPORTED_ACTIVATIONS, (
            f"{type(self).__name__} only supports "
            f"{[a.name for a in self._BF16_SUPPORTED_ACTIVATIONS]} activations, "
            f"got {self.activation_type.name}."
        )
        assert (
            not self.bias
            and self.act_alpha is None
            and self.act_beta is None
            and self.act_clamp is None
        ), f"{type(self).__name__} does not support bias/swiglu custom parameters."

    @classmethod
    def _check_bf16_path(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support bias/swiglu custom parameters.",
            )
        # Same set _check_configs asserts on, so the verdict and the
        # constructor agree instead of failing later at create_weights.
        if p.activation_type not in cls._BF16_SUPPORTED_ACTIVATIONS:
            supported = ", ".join(
                sorted(activation.name for activation in cls._BF16_SUPPORTED_ACTIVATIONS)
            )
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} only supports {supported} activations, got {p.activation}",
            )
        # Stronger than MoEDep.FLASHINFER: the wheel has to expose the two
        # bf16 entry points, which older ones do not.
        if not d.env.has_dep(MoEDep.FLASHINFER_BF16_MOE):
            return _reject(
                MoERejectReason.DEP_MISSING,
                f"{cls.__name__} requires FlashInfer fused MoE with trtllm_bf16_moe support.",
            )
        # FlashInfer BF16 kernels require the per-rank intermediate size
        # to be a multiple of 128.
        if p.intermediate_size is not None:
            inter = p.intermediate_size
            if d.tp_size > 1:
                if inter % d.tp_size != 0:
                    return _reject(
                        MoERejectReason.SHAPE_UNALIGNED,
                        f"{cls.__name__} requires intermediate_size ({inter}) "
                        f"divisible by moe_tp_size ({d.tp_size})",
                    )
                inter = inter // d.tp_size
            if inter % 128 != 0:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"{cls.__name__} requires "
                    "intermediate_size_per_partition % 128 == 0; "
                    f"got {inter} "
                    f"(full intermediate_size={p.intermediate_size}, "
                    f"moe_tp_size={d.tp_size})",
                )
        return None

    def _get_quant_method(self):
        return BF16TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = prepare_kernel_inputs(self, ctx)

        result = self.op_backend.run_bf16_moe(
            router_logits=k.router_logits,
            routing_bias=k.routing_bias,
            hidden_states=k.x,
            gemm1_weights=self.w3_w1_weight,
            gemm2_weights=self.w2_weight,
            num_experts=self.num_slots,
            top_k=k.top_k,
            n_group=k.n_group,
            topk_group=k.topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            routed_scaling_factor=k.routed_scaling_factor,
            routing_method_type=self.routing_method.routing_method_type,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            gated_act_type=to_trtllm_gen_act_type(self.activation_type),
            output=k.moe_output,
            use_shuffled_weight=getattr(self.quant_method, "use_shuffled_weight", False),
            weight_layout=getattr(self.quant_method, "weight_layout", 0),
            do_finalize=k.do_finalize,
        )
        if not k.do_finalize:
            return self._unfinalized(result)
        return result
