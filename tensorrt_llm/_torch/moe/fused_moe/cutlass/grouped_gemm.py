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
"""The one ``torch.ops.trtllm.fused_moe`` call the grouped-GEMM leaves share.

Ten leaves issue the same op with the same operands; what differs between them
is a handful of kernel-selection flags and, for two of them, a pre-pass over
``x``. Those differences are declared per leaf as a :class:`GroupedGemmFlags`
and read here, which is what replaces the chain of ``self.has_*`` tests the
single pre-split ``run_moe`` ran on every call.

The flags are a frozen dataclass rather than keyword arguments so that a leaf
declares them once, next to its identity, and a reader can see the whole kernel
configuration for a format in one place.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..impl_contract import MoERunContext, require_comm_plan


@dataclass(frozen=True)
class GroupedGemmFlags:
    """Kernel-selection flags for one quantization format.

    Every field maps to an argument of ``torch.ops.trtllm.fused_moe`` whose
    value used to be computed from a ``self.has_*`` property. Defaults describe
    the unquantized path, so a leaf only states what makes it different.
    """

    #: Reinterpret the weight buffers as this dtype before the call. ``None``
    #: keeps ``w3_w1_weight.dtype``, which is what every path but the packed
    #: 4-bit ones wants.
    weight_dtype: Optional[torch.dtype] = None
    #: Route FC1/FC2 through the DeepSeek block-scale GEMM runner instead of the
    #: CUTLASS grouped GEMM. Only the SM90 FP8-block-scale leaf sets this.
    use_deepseek_fp8_block_scale: bool = False
    #: Weights carry per-group scales (packed 4-bit formats).
    use_w4_group_scaling: bool = False
    use_int8_woq_per_channel: bool = False
    #: Quantize activations to MXFP8 inside the kernel before the GEMM.
    use_mxfp8_act_scaling: bool = False
    #: Select the MXFP8xMXFP8 block-scaled path inside the <e4m3, e4m3> runner
    #: template; per-tensor FP8 otherwise.
    use_mxfp8_weight_scaling: bool = False


#: The unquantized / per-tensor-FP8 shape: no flag set, native weight dtype.
DEFAULT_FLAGS = GroupedGemmFlags()


def run_grouped_gemm(
    impl,
    ctx: MoERunContext,
    flags: GroupedGemmFlags,
    *,
    x: Optional[torch.Tensor] = None,
    extra_quant_scales: Sequence[torch.Tensor] = (),
    use_dynamic_fc2_scale: bool = False,
) -> torch.Tensor:
    """Issue the CUTLASS grouped-GEMM MoE op and return the finalized output.

    ``x`` overrides ``ctx.x`` for the one leaf that pads its activations; the
    other nine pass the context through untouched.
    """
    plan = require_comm_plan(impl, ctx)
    enable_alltoall = plan.enable_alltoall
    tuner_num_tokens, tuner_top_k = impl._tuner_shapes(ctx, enable_alltoall)
    moe_output = plan.moe_output

    weight_dtype = flags.weight_dtype or impl.w3_w1_weight.dtype

    lora_kwargs = impl._extract_moe_lora_tensors(ctx.lora_params) or {}

    result = torch.ops.trtllm.fused_moe(
        ctx.x if x is None else x,
        ctx.token_selected_experts,
        ctx.token_final_scales,
        impl.w3_w1_weight.view(weight_dtype),
        impl.w3_w1_bias,
        impl.w2_weight.view(weight_dtype),
        impl.w2_bias,
        ctx.output_dtype,
        quant_scales=list(impl.quant_scales) + list(extra_quant_scales),
        input_sf=ctx.x_sf,
        swizzled_input_sf=plan.input_sf_swizzled,
        # ``swiglu_*`` are the moe_op schema's names for these registers
        # (``ActivationParams`` in moe_kernels.h); SiTU fills the same three
        # with tanh soft-caps.
        swiglu_alpha=impl.act_alpha,
        swiglu_beta=impl.act_beta,
        swiglu_limit=impl.act_clamp,
        tp_size=impl.tp_size,
        tp_rank=impl.tp_rank,
        ep_size=impl.ep_size,
        ep_rank=impl.ep_rank,
        cluster_size=impl.cluster_size,
        cluster_rank=impl.cluster_rank,
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=flags.use_deepseek_fp8_block_scale,
        use_w4_group_scaling=flags.use_w4_group_scaling,
        use_int8_woq_per_channel=flags.use_int8_woq_per_channel,
        use_mxfp8_act_scaling=flags.use_mxfp8_act_scaling,
        min_latency_mode=False,
        use_fused_finalize=impl.use_fused_finalize,
        tune_max_num_tokens=impl.tune_max_num_tokens,
        tuner_num_tokens=tuner_num_tokens,
        tuner_top_k=tuner_top_k,
        activation_type=impl.activation_type,
        unpadded_hidden_size=impl.unpadded_hidden_size,
        out_tensor=moe_output,
        use_dynamic_fc2_scale=use_dynamic_fc2_scale,
        use_mxfp8_weight_scaling=flags.use_mxfp8_weight_scaling,
        **lora_kwargs,
    )
    # With moe_output supplied the result is written in place and the op returns
    # an empty list to avoid an aliasing-constraint violation; otherwise the
    # single output tensor comes back in a list.
    return moe_output if moe_output is not None else result[0]


def run_dequantized_grouped_gemm(
    impl,
    ctx: MoERunContext,
    w3_w1_hp: torch.Tensor,
    w2_hp: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Issue the op against already-dequantized high-precision weights.

    The W4A16 path builds its operands per forward instead of reading the
    quantized buffers, so it cannot go through :func:`run_grouped_gemm`: the
    weights are arguments rather than module state, ``quant_scales`` is empty,
    and there is no input scaling factor. Everything else matches the
    unquantized call.
    """
    plan = require_comm_plan(impl, ctx)
    enable_alltoall = plan.enable_alltoall
    tuner_num_tokens, tuner_top_k = impl._tuner_shapes(ctx, enable_alltoall)
    moe_output = plan.moe_output

    result = torch.ops.trtllm.fused_moe(
        ctx.x,
        ctx.token_selected_experts,
        ctx.token_final_scales,
        w3_w1_hp,
        impl.w3_w1_bias,
        w2_hp,
        impl.w2_bias,
        output_dtype,
        quant_scales=[],
        input_sf=None,
        swizzled_input_sf=False,
        swiglu_alpha=impl.act_alpha,
        swiglu_beta=impl.act_beta,
        swiglu_limit=impl.act_clamp,
        tp_size=impl.tp_size,
        tp_rank=impl.tp_rank,
        ep_size=impl.ep_size,
        ep_rank=impl.ep_rank,
        cluster_size=impl.cluster_size,
        cluster_rank=impl.cluster_rank,
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_int8_woq_per_channel=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        use_fused_finalize=impl.use_fused_finalize,
        tune_max_num_tokens=impl.tune_max_num_tokens,
        tuner_num_tokens=tuner_num_tokens,
        tuner_top_k=tuner_top_k,
        activation_type=impl.activation_type,
        unpadded_hidden_size=impl.unpadded_hidden_size,
        out_tensor=moe_output,
        use_dynamic_fc2_scale=False,
    )
    return moe_output if moe_output is not None else result[0]


def local_expert_ids(impl, ctx: MoERunContext, enable_alltoall: bool) -> torch.Tensor:
    """Map the context's expert ids into this rank's local slot range.

    Two leaves need this and neither can guess ``enable_alltoall``: after an
    alltoall dispatch the ids are already local, otherwise they are global and
    have to be shifted by ``slot_start``. Getting it wrong shifts every id
    without failing, so the caller passes the plan's value explicitly.
    """
    local_n = impl.expert_size_per_partition
    ids = ctx.token_selected_experts
    if enable_alltoall:
        return ids.clamp(0, local_n - 1)
    return (ids - impl.slot_start).clamp(0, local_n - 1)
