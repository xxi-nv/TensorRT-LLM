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

from typing import Dict, Optional, Tuple

import torch

from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, EventType
from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import MoERunContext

# isort: off
from ..quantization import (
    MoEWeightLoadingMode,
)

# isort: on
from ..routing import BaseMoeRoutingMethod
from .eligibility import SmSupport


class CutlassFusedMoEBase(MoEImplBase):
    """Abstract root of the Cutlass-family implementations.

    Holds what every leaf shares: construction, the routed-expert LoRA
    plumbing, the weight lifecycle, input quantization, and the
    ``torch.ops.trtllm.fused_moe`` call. Abstract and unregistered -- it
    implements none of ``can_implement`` and publishes no descriptor, because a
    descriptor is exactly one identity and this class stands for twelve.

    What a leaf declares, and what the family gates in :mod:`.eligibility`
    read off it:

    * ``descriptor`` -- its one identity, which is also what
      ``check_quant_matches_identity`` compares a problem against.
    * ``sm_support`` -- the SM versions its kernel is built for.
    * ``supported_dtypes`` -- activation dtypes (BEFORE quantization) its
      kernel has instantiations for.
    * ``supports_gptoss_style`` / ``rejects_gptoss_expert_bias`` -- whether its
      weight method can load a gpt-oss / MiniMax SwiGLU shape.

    These replace the ``_QUANT_SUPPORT_TABLE`` this class used to interpret: a
    row that described a format the class did not implement is no longer
    expressible.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    MoE torch custom op:
        In max-throughput mode:
        Quant:
            fp8 block scales (SM90 Hopper only):
                FusedMoE Op: dynamic quant + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
            p8 qdq, nvfp4:
                FusedMoE Op: scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)

    FusedMoE module:
        max-throughput mode, with AttentionDP on:
            routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4]
            [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter,
            which equals: dynamic quant + routing(topK, etc.)
            [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 +
            finalizeMoeRoute [no allreduce] + reducescatter
    """

    # Declared by every leaf; no default, because a wrong SM set or dtype set
    # silently admits a layer whose kernel does not exist.
    sm_support: SmSupport
    supported_dtypes: frozenset
    # Whether the leaf's weight method can load a gpt-oss expert bias at all,
    # and -- for NVFP4, whose weight pad asserts 2-D -- whether a real 1-D
    # expert bias is still out while MiniMax-style SwigluBias is fine.
    supports_gptoss_style: bool = False
    rejects_gptoss_expert_bias: bool = False

    # The CUTLASS epilogue has an adaptor per kind (``moe_kernels.cuh``) and
    # takes all three constants as ``float*`` indexed by expert.
    activation_support = MoEActivationSupport(
        kinds=frozenset(
            {
                ActivationType.Swiglu,
                ActivationType.SwigluBias,
                ActivationType.Geglu,
                ActivationType.SiTu,
                ActivationType.Relu2,
                ActivationType.Silu,
            }
        ),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        bias: bool = False,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ):
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            activation=activation,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
        )

        # ``unpadded_hidden_size`` is captured by
        # apply_moe_impl_construction_state() above, before the padding below.
        if (
            model_config.quant_config
            and model_config.quant_config.layer_quant_mode.has_w4a16_mxfp4()
        ):
            self.hidden_size = ((self.hidden_size + 127) // 128) * 128
            self.intermediate_size_per_partition = (
                (self.intermediate_size_per_partition + 127) // 128
            ) * 128

        # The EPLB layout (num_slots, expert_size_per_partition, slot_start,
        # slot_end, initial_*) comes from apply_moe_impl_construction_state();
        # ConfigurableMoE then overwrites it via _BACKEND_SYNC_ATTRS.

        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        # The auxiliary CUDA stream and CUDA events are only used when MoE chunking is applied
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size
        if self.moe_max_num_tokens < default_moe_max_num_tokens:
            self.aux_stream = (
                aux_stream_dict[AuxStreamType.MoeChunkingOverlap]
                if aux_stream_dict is not None
                else torch.cuda.Stream()
            )
            self.event_dict = {
                key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeChunkingOverlap]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        # The profiler converges on the same best tactic when the number of tokens is large enough.
        # To avoid long profiling time, the max number of tokens used in the profiling is capped to
        # around 16k tokens per expert, which is well into the compute bound domain.
        self.tune_max_num_tokens = min(
            self.moe_max_num_tokens,
            16384 * self.num_slots // routing_method.get_experts_per_token(),
        )
        self.has_been_profiled = False
        self.has_been_profiled_min_latency = False

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Finalize fusion should be disabled if Lora is used.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )

        # Only the two leaves whose format the C++ op can fuse LoRA into
        # override this; for everyone else it is a no-op and the LoRA
        # attributes are never set. Called here, before create_weights(), to
        # keep the original ordering.
        self._init_moe_lora(model_config)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _init_moe_lora(self, model_config: ModelConfig) -> None:
        """Set up routed-expert MoE LoRA state. No-op unless the leaf can fuse it.

        The seam ``CutlassMoELoraMixin`` overrides. Readers of the LoRA state
        outside this package already probe for it rather than assume it --
        ``interface.forward_impl`` via ``getattr(self, "_moe_lora_enabled",
        False)`` and ``CudaGraphLoraManager`` via ``getattr(module,
        "reserve_moe_lora_cuda_graph_workspace", None)`` -- so a leaf that
        never sets it behaves as "no MoE LoRA here".
        """

    def _check_configs(self):
        assert self._weights_created

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

        if self.quant_config and self.quant_config.quant_mode.has_any_quant(exclude_kv_cache=True):
            if not (
                self.quant_config.quant_mode.has_nvfp4()
                | self.quant_config.quant_mode.has_fp8_block_scales()
                | self.quant_config.quant_mode.has_fp8_qdq()
                | self.quant_config.quant_mode.is_weight_only()
                | self.quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                | self.quant_config.quant_mode.has_w4a16_mxfp4()
                | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()
                | self.quant_config.quant_mode.has_mxfp8()
            ):
                raise ValueError(f"unsupported quantization mode: {self.quant_config.quant_mode}")

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group()

    @property
    def has_int8_woq_per_channel(self):
        return (
            self.quant_config
            and self.quant_config.layer_quant_mode.is_int8_weight_only()
            and not self.quant_config.layer_quant_mode.has_per_group_scaling()
        )

    def _supports_load_balancer(self) -> bool:
        """Every Cutlass leaf supports the load balancer."""
        return True

    def supports_moe_output_in_alltoall_workspace(self):
        return True

    def _tuner_shapes(
        self,
        ctx: MoERunContext,
        enable_alltoall: Optional[bool],
    ) -> Tuple[Optional[int], Optional[int]]:
        """Token/top-k shapes the profiling tuner should key on.

        Only meaningful under alltoall: the tuner must see pre-alltoall token
        counts so tactics cached during the no-alltoall warmup still apply at
        runtime. Without alltoall the kernel derives both from ``x`` itself.
        """
        if not enable_alltoall:
            return None, None
        if ctx.all_rank_num_tokens is not None:
            tuner_num_tokens = sum(ctx.all_rank_num_tokens)
        else:
            tuner_num_tokens = ctx.x.shape[0] * self.mapping.tp_size
        return tuner_num_tokens, self.routing_method.top_k
