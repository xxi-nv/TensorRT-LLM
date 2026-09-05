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
"""Per-quant abstract classes between ``TRTLLMGenFusedMoE`` and the leaves.

Each carries ``_get_quant_method`` / ``quantize_input`` / ``run_moe`` for one
weight-and-activation format, so a leaf that has both a native and a FlashInfer
provider adds only its identity and its provider gates. The two formats with a
single provider have no class here -- a parent implementing three methods for
exactly one subclass buys nothing -- and inherit ``TRTLLMGenFusedMoE`` directly.
"""

from typing import Optional, Union

import torch

from ....utils import ActType_TrtllmGen, Fp4QuantizedTensor, MxFp8QuantizedTensor
from ..fused_moe_trtllm_gen import TRTLLMGenFusedMoE, nvfp4_needs_padded_method
from ..impl_contract import MoERunContext

# isort: off
from ..quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEBaseMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
    W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
    W4A16MXFP4TRTLLMGenFusedMoEMethod,
)
# isort: on


class TRTLLMGenFp4BlockScaleBase(TRTLLMGenFusedMoE):
    """``run_fp4_block_scale_moe`` for the three formats that share it.

    NVFP4, W4A16_MXFP4 and W4A8_MXFP4_MXFP8 differ in how weights and inputs
    are prepared and not at all in how the kernel is called, so the call lives
    here once and each subclass supplies the preparation. Six of the eleven
    leaves reach the kernel through this body.
    """

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = self._prepare_kernel_inputs(ctx)

        act_type = self._to_trtllm_gen_activation_type(self.activation_type)
        factor = 1 if act_type in [ActType_TrtllmGen.Relu2, ActType_TrtllmGen.Silu] else 2
        intermediate_size_per_partition_padded = self.w3_w1_weight.shape[-2] // factor
        # Holds SwiGLU's per-expert alpha/beta, or SiTu's backend-local
        # activation parameters (which reuse this storage; see create_weights).
        gemm1_alpha, gemm1_beta = self.act_alpha, self.act_beta

        output1_scale_scalar = self._get_data_or_none("fc31_scale_c")
        output1_scale_gate_scalar = self._get_data_or_none("fc31_alpha")
        output2_scale_scalar = self._get_data_or_none("fc2_alpha")

        outputs = self.op_backend.run_fp4_block_scale_moe(
            k.router_logits,
            k.routing_bias,
            k.x,
            k.x_sf,
            self.w3_w1_weight,
            self.w3_w1_weight_scale,
            self.w3_w1_bias if self.bias else None,
            gemm1_alpha,
            gemm1_beta,
            self.act_clamp,
            self.w2_weight,
            self.w2_weight_scale,
            self.w2_bias if self.bias else None,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            self.num_slots,
            k.top_k,
            k.n_group,
            k.topk_group,
            intermediate_size_per_partition_padded,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            do_finalize=k.do_finalize,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            valid_hidden_size=self.hidden_size,
            valid_intermediate_size=getattr(
                self.quant_method, "intermediate_size_per_partition_lean", None
            ),
            gated_act_type=act_type,
            output=k.moe_output,
            # Pass that to the autotuner so the top bucket profiles per-expert load at runtime scale.
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )

        if not k.do_finalize:
            return self._unfinalized(outputs)

        # When output is provided, use it directly as the result
        final_hidden_states = k.moe_output if k.moe_output is not None else outputs
        # Slice output if it was padded (only needed when moe_output is not provided)
        if k.moe_output is None and final_hidden_states.shape[1] > self.hidden_size:
            final_hidden_states = final_hidden_states[:, : self.hidden_size].contiguous()
        return final_hidden_states


class TRTLLMGenNvfp4Base(TRTLLMGenFp4BlockScaleBase):
    """NVFP4 weights and activations, group-16 block scales."""

    supports_gptoss_style = True

    def _get_quant_method(self):
        # ``is_situ_activation`` (not ``act_alpha is not None``): SiTu fills the
        # act_alpha/act_beta slots from create_weights, i.e. after this runs, so
        # keying off the tensor would make the selected method depend on *when*
        # _get_quant_method is called. Like the SwiGLU-alpha and element-wise
        # cases, SiTu needs the padded method's alignment handling.
        needs_padded_method = nvfp4_needs_padded_method(
            self.activation_type, self.act_alpha is not None
        )
        return (
            NVFP4TRTLLMGenFusedMoEMethod()
            if needs_padded_method
            else NVFP4TRTLLMGenFusedMoEBaseMethod()
        )

    def quantize_input(self, x, post_quant_comm: bool = True):
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, (
                "Fp4QuantizedTensor should not be swizzled before communication"
            )
            x_row = x.shape[0]
            x, x_sf = x.fp4_tensor, x.scaling_factor
        elif isinstance(x, MxFp8QuantizedTensor):
            assert not x.is_sf_swizzled, (
                "MxFp8QuantizedTensor should not be swizzled before communication"
            )
            x_row = x.shape[0]
            x, x_sf = x.fp8_tensor, x.scaling_factor
        else:
            # Apply pre_quant_scale if it exists (for NVFP4_AWQ)
            # fc31_act_scale shape: (1, hidden_size)
            # x shape: (num_tokens, hidden_size)
            if hasattr(self, "fc31_act_scale") and self.fc31_act_scale is not None:
                x = x * self.fc31_act_scale

            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            if pad_size > 0:
                x = torch.nn.functional.pad(x, (0, pad_size))

            x_row = x.shape[0]
            x, x_sf = self.op_backend.fp4_quantize(
                x, self.fc31_input_scale, self.scaling_vector_size, False, False
            )
        return x, x_sf.view(x_row, -1)


class TRTLLMGenW4a16Mxfp4Base(TRTLLMGenFp4BlockScaleBase):
    """MXFP4 weights, bfloat16 activations."""

    supports_gptoss_style = True

    def _get_quant_method(self):
        return W4A16MXFP4TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        # Weight-only: the activation is padded to the packed weight width and
        # stays bfloat16, so there is no scaling factor to hand back.
        pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad_size)), None


class TRTLLMGenW4a8Mxfp4Mxfp8Base(TRTLLMGenFp4BlockScaleBase):
    """MXFP4 weights, MXFP8 activations, group-32 block scales."""

    supports_gptoss_style = True

    def _get_quant_method(self):
        return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        x, x_sf = self.op_backend.mxfp8_quantize(
            x, False, alignment=self.quant_method.input_hidden_alignment
        )
        return x, x_sf.view(x.shape[0], -1)


class TRTLLMGenFp8BlockScalesBase(TRTLLMGenFusedMoE):
    """DeepSeek-style FP8 with 1x128 block scales.

    The only format whose activation is quantized inside ``run_moe`` rather
    than in ``quantize_input``: ``fp8_quantize_1x128`` returns scales shaped
    ``(blocked_n, num_tokens)``, and the all-to-all dispatch needs every
    payload's first dimension to be ``num_tokens``. Transposing around the
    dispatch would cost more than it saves, so this format simply does not
    offer post-quant communication.
    """

    def _get_quant_method(self):
        return DeepSeekFP8BlockScalesFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = self._prepare_kernel_inputs(ctx)

        assert k.do_finalize, "fp8_block_scale_moe_runner does not support do_finalize=False"
        x, x_sf = k.x, k.x_sf
        # fp8_quantize_1x128 returns 2D x_sf on SM100+, 1D on SM90
        if x_sf is None:
            x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)

        result = self.op_backend.run_fp8_block_scale_moe(
            k.router_logits,
            k.routing_bias,
            x,
            x_sf,
            self.w3_w1_weight,
            self.w3_w1_weight_scaling_factor,
            self.w2_weight,
            self.w2_weight_scaling_factor,
            self.num_slots,
            k.top_k,
            self.num_fused_shared_expert,
            k.n_group,
            k.topk_group,
            self.intermediate_size_per_partition,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            gemm1_clamp_limit=self.act_clamp,
            output=k.moe_output,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )
        # When output is provided, use it directly as the result
        return k.moe_output if k.moe_output is not None else result
