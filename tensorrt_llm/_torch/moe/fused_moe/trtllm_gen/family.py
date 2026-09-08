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
"""The abstract root the eleven TRTLLM-Gen leaves share.

Carries only what all eleven need: construction, the weight-creation skeleton,
the two routing predicates the framework reads off a module, and the fake-output
shapes. Anything that varies by quantization format lives on a class in
:mod:`.quant_bases`, and anything that varies by provider on one of the trait
classes in :mod:`.identity` -- so nothing here branches on either axis.

It implements none of the four abstract methods of ``MoEImplBase`` and declares
no ``MoEImplDescriptor``, so it is abstract at the type level and unaddressable
at the registry level: a descriptor is exactly one identity, and a class
standing for eleven has none to publish.
"""

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger

from ....custom_ops.trtllm_gen_custom_ops import fp4_block_scale_fake_output_without_finalize
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import canonical_quant
from ..interface import FORCE_SEPARATED_ROUTING, MoEWeightLoadingMode
from ..moe_op_backend import MoEOpBackend, get_op_backend
from ..routing import BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod, MiniMaxM2MoeRoutingMethod
from .eligibility import identity_quant_of, normalize_quant
from .identity import TRTLLM_GEN_CAPABILITIES, TRTLLM_GEN_INPUT_REQUIREMENT


class TrtllmGenFusedMoEBase(MoEImplBase):
    """Abstract root of the eleven ``*.trtllm_gen.fused_moe.*`` implementations.

    The split below this class is by ``quant`` x ``provider``, because those are
    the two axes the old single class switched on at runtime. The two axes divide
    the work differently, and the class layout follows that rather than the
    identity grid:

    - ``quant`` decides which kernel ``run_moe`` calls, how weights and inputs
      are prepared, and which activation ABI applies. That is held by a
      per-quant class in :mod:`.quant_bases`, one for each format two providers
      share, so the ``run_fp4_block_scale_moe`` body is written once and not six
      times. A format only one provider serves has no such class: below one leaf
      a parent would implement something with exactly one subclass.
    - ``provider`` decides which op backend executes, which configurations are
      eligible, and whether the kernel writes into the caller's workspace. That
      is held by the traits mixed into each registered leaf, which is therefore
      small: an identity, its provider traits, and a ``can_implement``.

    One kernel call per forward for all eleven: routing (unless a leaf routes
    outside it), scatter, gemm1, the fused activation, gemm2, and the combine
    reduction are a single cubin. That is why ``run_moe`` returns a finalized
    tensor by default; eight of the eleven can be asked to stop short and hand
    back per-expert outputs for the caller to combine, through
    ``_unfinalized`` below.

    No AllReduce is issued from here. With ``reduce_results=False`` the model
    definition owns it; ``ConfigurableMoE`` supplies it through the
    communication strategy otherwise.

    Slots, not experts, size the weight buffers: EPLB may give a hot expert
    several replicas, so ``num_slots >= num_experts`` and the kernel is told
    both. ``expert_size_per_partition`` and ``slot_start`` are this rank's
    window into the slot array.
    """

    # ---- identity-derived declarations, set by each registered leaf ------
    #: ``MoEImplId.provider``, and the ``moe_op_backend`` registry key. Named
    #: once per leaf; ``__init__`` builds the op backend from it, so a leaf
    #: cannot end up executing a provider other than the one it publishes.
    provider: str
    #: ``provider == PROVIDER_FLASHINFER``, restated because it is read from
    #: outside as a plain attribute (``ConfigurableMoE`` passes it to the
    #: communication factory, and the backend tests read it off an instance).
    use_flashinfer: bool
    #: Whether this leaf's quantization format has a fused cubin for the gpt-oss
    #: SwiGLU package (expert bias plus alpha/beta). Read off the class by
    #: ``check_trtllm_gen_capabilities``, which runs at resolution time with no
    #: instance to ask, so it has to be a class attribute rather than a method.
    supports_gptoss_style: bool = False
    #: Whether this leaf can run SiTu: its format must have a fused SiTu FC1
    #: cubin *and* its provider must call that cubin, so it is declared on the
    #: leaf and not on the format base a FlashInfer sibling shares. Read off the
    #: class at resolution and off the instance by the format base's
    #: construction check, so what the registry admits and what the constructor
    #: accepts cannot drift.
    supports_situ: bool = False
    #: Whether this format's cubins always read an FC bias, so a model without
    #: one still needs a zeroed buffer allocated. Stays declarative data rather
    #: than an override because its four setters sit on two different
    #: inheritance levels -- two format bases and two single-provider leaves.
    needs_zero_expert_bias: bool = False

    # Inherited by all eleven leaves and restated in each descriptor from the
    # same module-level constants, so what the registry publishes and what the
    # scheduler reads cannot drift apart.
    capabilities = TRTLLM_GEN_CAPABILITIES
    input_requirement = TRTLLM_GEN_INPUT_REQUIREMENT

    # The fused-activation cubins index alpha/beta/clamp by expert
    # (``gemm1_alpha`` / ``gemm1_beta`` / per-expert clamp tensor). The FP8
    # block-scale format runs its clamp in a separate activation kernel that
    # takes a scalar instead, which is why that format overrides
    # ``resolve_activation_support`` and this declaration does not mention it.
    activation_support = MoEActivationSupport(
        kinds=frozenset(
            {
                ActivationType.Swiglu,
                ActivationType.SwigluBias,
                ActivationType.Relu2,
                ActivationType.Silu,
                ActivationType.SiTu,
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
        layer_idx: Optional[int] = None,
        bias: bool = False,
        init_load_balancer: bool = False,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
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
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation=activation,
        )

        self._check_before_weights()

        # Cached for autotune profile sizing (forward path passes
        # tune_max_num_tokens to the MoE op).
        self.max_num_tokens = model_config.max_num_tokens

        # The provider is the leaf's identity, not a runtime choice: which
        # kernels can serve this configuration is what ``can_implement``
        # answered, and which of the two providers to prefer among those that
        # can is what ``IMPL_PRIORITY`` answered. By the time a leaf is
        # constructed both questions are settled, so this only names the op
        # backend its identity already committed to.
        self.op_backend: MoEOpBackend = get_op_backend(self.provider)

        self._weights_created = False
        # Read from outside through ``ConfigurableMoE.num_fused_shared_expert``
        # on every leaf, so it is initialized here even though only one format
        # can raise it above zero.
        self.num_fused_shared_expert = 0
        self._configure_shared_expert_fusion(model_config)

        # create_weights must see the final fused-expert count so the fused shared
        # slots are allocated when fusion is enabled.
        if not model_config.skip_create_weights_in_init:
            self.create_weights()
        self.layer_idx = layer_idx

    @property
    def is_situ_activation(self) -> bool:
        return self.activation.kind is ActivationType.SiTu

    def _check_before_weights(self) -> None:
        """Preconditions the SiTu cubins carry, checked where the caller asked.

        Separate from ``_check_configs`` by when it runs, which is the reason it
        exists: this fires from ``__init__`` even when
        ``skip_create_weights_in_init`` defers weight creation. It also has to
        run for all eleven and not just the two formats that serve SiTu -- the
        rejection below is the only thing standing between a directly
        constructed leaf and the wrong cubin, because two of the leaves declare
        no ``_check_configs`` at all. Resolution rejects the same configurations
        earlier, from ``supports_situ`` on the class.

        Kind and constant shape are already settled: the activation carrier only
        admits two positive soft-caps for SiTu and no clamp, and
        ``install_activation_params`` has materialized them as the per-expert
        ``gemm1_alpha`` / ``gemm1_beta`` buffers the cubin indexes.
        """
        if not self.is_situ_activation:
            return
        if self.dtype != torch.bfloat16:
            raise ValueError(f"TRTLLM-Gen SiTu requires bfloat16 activations, got {self.dtype}.")
        if not is_sm_100f():
            raise ValueError(
                f"TRTLLM-Gen SiTu requires the SM100 family, got SM{get_sm_version()}."
            )
        if not self.supports_situ:
            quant_algo = None if self.quant_config is None else self.quant_config.quant_algo
            raise ValueError(
                f"{type(self).__name__} reaches no fused SiTu cubin "
                f"(quant_algo={quant_algo}, provider={self.provider})."
            )
        if self.tp_size > 1:
            # Intra-expert MoE TP: w1/w3 column-shard and w2 row-shard along
            # the intermediate dim (the stock MXFP4/NVFP4 quant-method loaders
            # slice the packed bytes and scales per rank). Require the
            # per-rank shard to stay a whole multiple of the quant method's
            # weight alignment so per-shard scale groups and the padded
            # weight buffers line up without fractional groups.
            alignment = self._situ_tp_weight_alignment()
            if (
                self.intermediate_size % self.tp_size != 0
                or self.intermediate_size_per_partition % alignment != 0
            ):
                raise ValueError(
                    "TRTLLM-Gen SiTu MoE TP requires intermediate_size "
                    f"({self.intermediate_size}) divisible by moe_tp_size "
                    f"({self.tp_size}) with the per-rank shard a multiple of "
                    f"{alignment}, got {self.intermediate_size_per_partition}."
                )
        if self.bias:
            raise ValueError(
                "TRTLLM-Gen SiTu does not support expert bias; the cubin adds "
                "no FC1 bias before the soft-caps."
            )

    def _situ_tp_weight_alignment(self) -> int:
        """The weight alignment a SiTu MoE-TP shard has to be a multiple of.

        The one format-dependent step of the check above, so it is answered by
        the format base rather than switched on here. Only reachable with
        ``supports_situ`` set, which is why the two formats that set it are the
        only two that implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares supports_situ but no SiTu MoE-TP alignment."
        )

    @classmethod
    def fused_shared_expert_count(cls, model_config: ModelConfig) -> int:
        """How many shared experts this format folds into the routed GEMM.

        Zero here: only one of the seven formats has a grouped GEMM that can
        append shared-expert slots, and it is the class implementing that
        format that knows the environment flag, the parallel restrictions, and
        the weight layout involved.

        A classmethod rather than only a constructor side effect because the
        answer is needed before any leaf exists:
        ``ConfigurableMoE.will_fuse_shared_expert`` asks it while the model is
        still in ``__init__``, where DeepseekV3 sizes the shared-expert TP that
        its ``GatedMLP`` is then built with. Keeping the rule in one place is
        what stops that prediction and this layer's actual count from drifting.
        """
        del model_config  # a format that cannot fuse has nothing to read
        return 0

    def _configure_shared_expert_fusion(self, model_config: ModelConfig) -> None:
        """Adopt this format's fused-shared-expert count for this instance."""
        self.num_fused_shared_expert = type(self).fused_shared_expert_count(model_config)
        if self.num_fused_shared_expert > 0:
            logger.info_once(
                f"Shared-expert fusion enabled: folding "
                f"{self.num_fused_shared_expert} shared expert(s) into the "
                f"routed-expert grouped GEMM.",
                key="trtllm_gen_shared_expert_fusion",
            )

    def _requires_separated_routing(self) -> bool:
        """Whether this leaf's kernel takes top-k from the host, not the logits.

        False for ten of the eleven: their cubins route internally from the
        logits. The one leaf whose kernel has no internal routing overrides this.
        """
        return False

    def _supports_load_balancer(self) -> bool:
        """Whether separated routing (top-k outside the kernel) is used.

        ConfigurableMoE uses this flag to decide whether routing is separated
        (top-k ids/scales computed outside backend) or fused inside the kernel.
        """
        if self._requires_separated_routing():
            return True
        return self.use_dp and self.parallel_size > 1

    def _routes_outside_the_kernel(self) -> bool:
        """Whether top-k is precomputed, so the kernel must not route again.

        Three independent triggers, none of which subsumes the others: a
        kernel or parallel layout that forces it (both folded into
        ``_supports_load_balancer``), a routing algorithm no C++ kernel
        implements, and the host-routing override.
        """
        return (
            self._supports_load_balancer()
            or self.routing_method.requires_separated_routing
            or FORCE_SEPARATED_ROUTING
        )

    def create_weights(self):
        """The allocation skeleton; the variations are hooks, not tests here.

        What differs between formats is either an override of
        ``_create_quant_method_weights`` or the declarative
        ``needs_zero_expert_bias``, because the old form read ``quant_config``
        back at allocation time to rediscover a format the identity already
        fixed. ``_check_configs`` is not implemented at this level at all: what
        used to be one method with six quant allow-lists is now each format or
        leaf asserting only its own.
        """
        if self._weights_created:
            return

        self._check_quant_config_is_my_format()
        self.quant_method = self._get_quant_method()
        self._create_quant_method_weights()

        self._weights_created = True
        self._check_configs()

        if self.needs_zero_expert_bias and not self.bias:
            self._allocate_zero_expert_bias()

    def _check_quant_config_is_my_format(self) -> None:
        """Fail loudly if this layer ended up on a leaf of the wrong format.

        The same comparison ``can_implement`` makes, repeated once weights are
        about to be laid out, because the two happen at different times against
        different inputs. Resolution keys on the model-level ``quant_algo``,
        while ``apply_layerwise_quant_config`` and
        ``apply_quant_config_exclude_modules`` give a layer its own
        ``quant_config`` afterwards -- so a layer can be admitted by a leaf
        whose format it no longer has.

        ``ConfigurableMoE.create_weights`` re-resolves for exactly that case
        and this never fires under it. What it covers is direct construction,
        which has no such step: without it, the mismatch would surface as
        weights the checkpoint cannot fill, or silently wrong numerics.
        """
        expected = identity_quant_of(type(self))
        actual = normalize_quant(
            canonical_quant(None if self.quant_config is None else self.quant_config.quant_algo)
        )
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__} implements quant={expected}, but layer "
                f"{self.layer_idx}'s quant_config resolves to {actual}. The "
                f"implementation was picked from the model-level quant_algo; "
                f"layerwise quantization or a module exclusion moved this layer "
                f"afterwards, and the layer must be re-resolved for the format "
                f"it actually has."
            )

    def _create_quant_method_weights(self) -> None:
        """Hand the module to its quantization method, and settle the layout.

        The one hook the weight skeleton needs: a format may pass extra
        arguments its method takes, or install backend-owned parameters that
        ``_check_configs`` then validates. Overriders that do the latter must
        call ``super()`` first.
        """
        self.quant_method.create_weights(self)

    def _allocate_zero_expert_bias(self) -> None:
        """Give the FC epilogues a bias buffer the checkpoint does not carry.

        The cubins for these formats always read a bias, so a model without one
        still needs the registers filled.
        """
        self.w3_w1_bias = nn.Parameter(
            torch.zeros(
                (self.w3_w1_weight.shape[0], self.w3_w1_weight.shape[1]), dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.register_parameter("w3_w1_bias", self.w3_w1_bias)
        self.w2_bias = nn.Parameter(
            torch.zeros((self.w2_weight.shape[0], self.w2_weight.shape[1]), dtype=torch.float32),
            requires_grad=False,
        )
        self.register_parameter("w2_bias", self.w2_bias)

    def supports_moe_output_in_alltoall_workspace(self):
        """Whether ``run_moe`` fills a caller-supplied output buffer.

        Only the native provider's runners take an output tensor;
        ``self.has_any_quant and not self.use_flashinfer`` reduced to the second
        half of itself, because the only unquantized leaf is a FlashInfer one.
        A future leaf that diverges from its provider overrides this.
        """
        return not self.use_flashinfer

    def _unfinalized(self, outputs):
        """Hand back per-expert outputs for the caller to combine."""
        assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
        return outputs

    def forward_fake(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Meta-tensor shapes for both the finalized and un-finalized ABIs.

        Both halves are family-wide. The finalized output is bf16 for all eleven
        because the shared eligibility gate admits only bfloat16 activations. The
        un-finalized layout is reached by eight of them -- through the FP4
        block-scale format base, the unquantized leaf, and the NVFP4/FP8 leaf --
        which is why it is answered here rather than on any one of those.
        """
        if do_finalize:
            return super().forward_fake(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=torch.bfloat16,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )

        is_deepseek_v3_routing = isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod)
        is_minimax_routing = isinstance(self.routing_method, MiniMaxM2MoeRoutingMethod)
        top_k = (
            self.routing_method.routing_impl.top_k
            if is_deepseek_v3_routing
            else self.routing_method.top_k
        )
        routing_bias = (
            self.routing_method.e_score_correction_bias
            if (is_deepseek_v3_routing or is_minimax_routing)
            else None
        )
        return fp4_block_scale_fake_output_without_finalize(
            x,
            self.num_experts,
            top_k,
            routing_bias,
        )
