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

import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...custom_ops.trtllm_gen_custom_ops import \
    fp4_block_scale_fake_output_without_finalize
from ...model_config import ModelConfig
from ...modules.gated_mlp import GatedMLP
from ...utils import (ActivationType, ActType_TrtllmGen, AuxStreamType,
                      Fp4QuantizedTensor, MxFp8QuantizedTensor)
from .activation import (DEFAULT_MOE_ACTIVATION, ActivationParamShape,
                         MoEActivation, MoEActivationSupport,
                         materialize_activation_params)
from .impl_base import MoEImplBase, apply_moe_impl_construction_state
from .impl_contract import (MoEDeployment, MoEEligibility, MoEInputRequirement,
                            MoEProblem, MoERejectReason, MoERunContext,
                            MoEStaticCapability, require_comm_plan)
from .impl_environment import MoEDep, MoEEnvFlag
from .impl_identity import (MOE_IMPL_REGISTRY, MoEImplDescriptor, MoEImplId,
                            register_moe_impl)
from .interface import (FORCE_SEPARATED_ROUTING, MoESchedulerKind,
                        MoEWeightLoadingMode, _reject)
from .moe_op_backend import MoEOpBackend, TRTLLMOpBackend, get_op_backend

# isort: off
from .quantization import (
    BF16TRTLLMGenFusedMoEMethod, DeepSeekFP8BlockScalesFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEBaseMethod, NVFP4TRTLLMGenFusedMoEMethod,
    W4A8MXFP4FP8TRTLLMGenFusedMoEMethod, W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
    W4A8NVFP4FP8TRTLLMGenFusedMoEMethod, W4A16MXFP4TRTLLMGenFusedMoEMethod)
# isort: on
from .routing import (BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod,
                      DeepSeekV4MoeRoutingMethod, MiniMaxM2MoeRoutingMethod,
                      MiniMaxM3MoeRoutingMethod)


@dataclass
class RoutingParams:
    top_k: int
    routing_bias: Optional[torch.Tensor]
    n_group: Optional[int]
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]


@dataclass
class KernelInputs:
    """What every TRTLLM-Gen kernel call needs, derived once from a run context.

    The seven ``run_moe`` bodies used to sit in one ``elif`` chain under a
    shared prologue. Splitting them by quantization format would have copied
    that prologue seven times, so it is a parent method returning this instead:
    the routing arguments each kernel takes positionally, resolved once, plus
    the three context values every branch reads.

    ``routing_bias`` is already ``None`` when routing happened outside the
    kernel, and ``top_k`` already reflects a caller-supplied top-k width. Both
    are decisions, not raw inputs, which is why they are settled here rather
    than in each branch.
    """

    x: Union[torch.Tensor, Fp4QuantizedTensor]
    x_sf: Optional[torch.Tensor]  # flattened to 1D for the kernel ABI
    router_logits: Optional[torch.Tensor]  # None when top-k is precomputed
    token_selected_experts: Optional[torch.Tensor]
    token_final_scales: Optional[torch.Tensor]
    moe_output: Optional[torch.Tensor]
    do_finalize: bool
    top_k: int
    routing_bias: Optional[torch.Tensor]
    n_group: Optional[int]
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]


# Kernel-lineage segment of the eleven identities. The same TRTLLM-Gen
# algorithm ships in TRT-LLM's own cubins and in the FlashInfer wheel, so
# provider is what tells two otherwise identical implementations apart -- the
# only place in the registry where that is true. These strings are also the op
# backend registry keys (``moe_op_backend.get_op_backend``), which is why a leaf
# needs to name its provider exactly once.
PROVIDER_TRTLLM = "trtllm"
PROVIDER_FLASHINFER = "flashinfer"

# Technique and kernel segments. Identical across the eleven, because the
# algorithm is: one batched-GEMM cubin family, reached as a single fused MoE op.
TECHNIQUE_TRTLLM_GEN = "trtllm_gen"
KERNEL_FUSED_MOE = "fused_moe"

# Declared once and referenced from both the parent class attributes and every
# descriptor. All eleven publish the same two, so a per-leaf literal would be
# eleven chances for the published contract and the executed one to disagree.
TRTLLM_GEN_CAPABILITIES = MoEStaticCapability(supports_expert_bias=True,
                                              supports_eplb=True)

# bfloat16 routing scales are what these kernels read, and the DeepEP
# dispatch has to mark unfilled rows before they reach them.
TRTLLM_GEN_INPUT_REQUIREMENT = MoEInputRequirement(
    routing_scales_dtype=torch.bfloat16,
    requires_sanitized_expert_ids=True,
    # The combine reduction runs in bf16 regardless of the model's output
    # dtype, so the NVLink one-sided payload buffer must be bf16 too.
    onesided_workspace_dtype=torch.bfloat16,
)


def trtllm_gen_leaf(quant_algo: Optional[QuantAlgo],
                    *,
                    provider: Optional[str] = None) -> type:
    """The leaf implementing ``quant_algo``, on ``provider`` when given.

    For callers that already know the format and want the class rather than a
    verdict: benchmarks that construct a backend directly, and tests that ask
    one leaf a question. Resolution itself does not use this -- it walks
    ``IMPL_PRIORITY`` and asks ``can_implement`` -- so this stays a lookup and
    never becomes a second selection path.

    With ``provider`` left open, the native leaf wins where there is one, which
    matches what a deployment without the FlashInfer opt-in flag would run. The
    unquantized format has no native leaf, so there it resolves to the
    FlashInfer one -- the only implementation of it either way.

    Goes through the registry rather than a table of its own, so a leaf that is
    renamed or unregistered disappears from here too.
    """
    quant = "none" if quant_algo is None else str(quant_algo.value)
    providers = ((provider, ) if provider is not None else
                 (PROVIDER_TRTLLM, PROVIDER_FLASHINFER))
    for candidate in providers:
        cls = MOE_IMPL_REGISTRY.lookup(
            MoEImplId(candidate, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE, quant))
        if cls is not None:
            return cls
    registered = sorted(identity.canonical()
                        for identity in MOE_IMPL_REGISTRY.identities()
                        if identity.technique == TECHNIQUE_TRTLLM_GEN)
    raise ValueError(
        f"no TRTLLM-Gen implementation for quant={quant} on "
        f"provider={'|'.join(providers)}; registered: {registered}")


def trtllm_gen_descriptor(provider: str, quant: str,
                          doc: str) -> MoEImplDescriptor:
    """Build one leaf's descriptor; only provider and quant ever differ."""
    return MoEImplDescriptor(
        identity=MoEImplId(provider, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE,
                           quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=TRTLLM_GEN_CAPABILITIES,
        input_requirement=TRTLLM_GEN_INPUT_REQUIREMENT,
        doc=doc,
    )


def check_trtllm_gen_capabilities(cls: type, p: MoEProblem,
                                  d: MoEDeployment) -> Optional[MoEEligibility]:
    """Capability gates every TRTLLM-Gen leaf shares, or ``None`` to admit.

    A free function rather than a base-class method: the abstract parent must
    not implement ``can_implement``, and eleven leaves restating the same three
    gates is how the SM and dtype answers drift apart. Each leaf checks its own
    quantization format, then defers here.

    Reads only ``p`` and ``d``. No ``get_sm_version()``, no ``os.environ``, and
    no import probe -- SM and dependency availability arrive through
    ``d.env``, collected once, so an offline tuner on a GPU-less host gets the
    same verdict a serving process does.
    """
    # The cubin drop is sm_100f (family-compatible) plus arch-specific
    # sm_100a/sm_103a, so the whole SM100 family is servable; the C++
    # selector (KernelRunner.cpp isSMCompatible) picks sm_100f on family
    # members without their own arch build.
    if not is_sm_100f(d.env.sm):
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} requires the SM100 family, got SM{d.env.sm}")

    # run_moe asserts x.dtype == torch.bfloat16
    if p.dtype_act != torch.bfloat16:
        return _reject(
            MoERejectReason.DTYPE_UNSUPPORTED,
            f"{cls.__name__} only supports bfloat16 activation, got {p.dtype_act}"
        )

    if d.smart_router:
        return _reject(
            MoERejectReason.TOPOLOGY_UNSUPPORTED,
            f"{cls.__name__} has no smart-router path (moe_cluster_size="
            f"{d.cluster_size})")

    # Whether the gpt-oss SwiGLU package (expert bias plus alpha/beta) has a
    # fused cubin is a per-quant fact, so the leaf declares the answer and this
    # gate only applies it.
    if p.swiglu_gptoss_style and not cls.supports_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} has no fused bias/swiglu-parameter cubin for its "
            f"quantization format (quant={p.quant})")

    return None


def check_flashinfer_provider(cls: type, p: MoEProblem,
                              d: MoEDeployment) -> Optional[MoEEligibility]:
    """Gates that separate the FlashInfer provider from the ``trtllm`` one.

    Mirrors what ``_check_flashinfer_backend_support`` used to answer from an
    instance in ``__init__``. Every input it read is available statically now:
    the opt-in flag and the wheel's presence come from ``d.env``, the activation
    and routing shapes from ``p``.

    Only the quantized leaves come through here. The unquantized one is
    FlashInfer-exclusive and reached without the opt-in flag, so it states its
    own dependency gate instead.
    """
    if not d.env.has_dep(MoEDep.FLASHINFER):
        return _reject(MoERejectReason.DEP_MISSING,
                       f"{cls.__name__} requires the FlashInfer wheel")

    # Opt-in, not a capability: the trtllm provider serves these formats too,
    # so routing traffic here without being asked would change which kernel a
    # previously-working deployment runs.
    if d.env.env_flag(MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER) != "1":
        return _reject(
            MoERejectReason.PATH_NOT_ENABLED, f"{cls.__name__} is opt-in; set "
            f"{MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER.value}=1 to select the "
            f"FlashInfer provider for quantized TRTLLM-Gen")

    # SiTu is a native TRTLLM-Gen cubin and is absent from FlashInfer's
    # activation enum; Relu2 has no FlashInfer path either.
    if p.activation_type in (ActivationType.SiTu, ActivationType.Relu2):
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} does not implement {p.activation} "
            f"(FlashInfer's activation enum has no such kernel)")

    # These two fuse routing in a form only the native runner accepts. Named by
    # RoutingMethodType rather than by routing class, which is what a problem
    # carries.
    #
    # One consequence is deliberate and worth stating: DeepSeekV4MoeRoutingMethod
    # reports RoutingMethodType.DeepSeekV3 (routing.py:595-597, so the C++ MoE
    # kernels get an encoding they recognize), so it lands in this rejection
    # where the old instance-level isinstance check let it through. It only
    # changes anything under the opt-in flag above, and the effect is that a
    # DeepSeek-V4 layer stays on the trtllm provider rather than moving to
    # FlashInfer -- a provider choice, not a capability loss.
    if p.routing_method_type is not None and p.routing_method_type.name in (
            "DeepSeekV3", "Default"):
        return _reject(
            MoERejectReason.ROUTING_UNSUPPORTED,
            f"{cls.__name__} does not implement {p.routing} routing")

    return None


def check_mxfp4_flashinfer_shape(
        cls: type, p: MoEProblem, d: MoEDeployment, *, weight_alignment: int,
        input_hidden_alignment: int) -> Optional[MoEEligibility]:
    """Per-rank shard alignment the FlashInfer path needs, or ``None``.

    Absent shapes abstain rather than reject: a call site that did not say what
    ``hidden_size`` is has not said the layer is misaligned.
    """
    if p.bias:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} takes no expert bias on the FlashInfer provider")

    if p.intermediate_size is not None:
        inter = p.intermediate_size
        if d.tp_size > 1:
            if inter % d.tp_size != 0:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"{cls.__name__} requires intermediate_size ({inter}) "
                    f"divisible by moe_tp_size ({d.tp_size})")
            inter = inter // d.tp_size
        if inter % weight_alignment != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"{cls.__name__} requires intermediate_size_per_partition "
                f"({inter}) to be a multiple of {weight_alignment}")

    if p.hidden_size is not None and p.hidden_size % input_hidden_alignment != 0:
        return _reject(
            MoERejectReason.SHAPE_UNALIGNED,
            f"{cls.__name__} requires hidden_size ({p.hidden_size}) to be a "
            f"multiple of {input_hidden_alignment}")

    return None


def nvfp4_needs_padded_method(activation_type: ActivationType,
                              has_alpha_constant: bool) -> bool:
    """Whether NVFP4 needs the padded quant method rather than the base one.

    One definition read from two sides: ``can_implement`` asks it of the
    problem, ``_get_quant_method`` of the instance. The two answers decide
    whether the alignment gates above apply, so a second copy would let a
    configuration be admitted by one and laid out by the other.
    """
    return (has_alpha_constant or activation_type is ActivationType.SiTu
            or activation_type in (ActivationType.Relu2, ActivationType.Silu))


def check_quant_matches_identity(cls: type,
                                 p: MoEProblem) -> Optional[MoEEligibility]:
    """Reject any format other than the one in this leaf's own identity.

    Read off ``cls.descriptor`` rather than restated as a set, so the format a
    leaf admits and the format it publishes are the same string. This replaces
    the ``_SUPPORTED_QUANT_ALGOS`` membership test, which had to admit all
    seven because one class served all seven.
    """
    expected = cls.descriptor.identity.quant
    actual = "none" if p.quant is None else p.quant.lower()
    if actual != expected:
        return _reject(
            MoERejectReason.QUANT_UNSUPPORTED,
            f"{cls.__name__} implements quant={expected}, got {actual}")
    return None


def check_trtllm_gen_leaf(
        cls: type, p: MoEProblem, d: MoEDeployment, *provider_gates:
    Optional[MoEEligibility]) -> MoEEligibility:
    """Compose one leaf's verdict: identity, shared capability, then provider.

    First rejection wins, and the order is fixed so that two leaves differing
    only in provider give the same reason for a problem neither can serve.
    ``provider_gates`` are already-evaluated verdicts rather than callables:
    every gate is a pure function of ``p`` and ``d``, so there is nothing to
    defer, and passing values keeps the leaf bodies to one expression.
    """
    for verdict in (check_quant_matches_identity(cls, p),
                    check_trtllm_gen_capabilities(cls, p, d), *provider_gates):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()


class TRTLLMGenFusedMoE(MoEImplBase):
    """Abstract parent of the eleven TRTLLM-Gen implementations.

    Carries what the leaves share and nothing they differ on: construction, the
    weight lifecycle, the routing-parameter extraction, the activation ABI, and
    the shared-expert fusion. It implements none of the four abstract methods of
    ``MoEImplBase`` and declares no ``MoEImplDescriptor``, so it is abstract at
    the type level and unaddressable at the registry level -- a descriptor is
    exactly one identity, and a class standing for eleven has none to publish.

    The split is by ``quant`` x ``provider`` because those are the two axes the
    old single class switched on at runtime: seven quantization variants, each
    of which then chose between TRT-LLM's own cubins and the FlashInfer wheel.
    The two axes divide the work differently, and the class layout follows that
    rather than the identity grid:

    - ``quant`` decides which kernel ``run_moe`` calls, and how weights and
      inputs are prepared. That is held by a per-quant abstract class, one for
      each format two providers share, so the ``run_fp4_block_scale_moe`` body
      is written once and not six times. A format only one provider serves has
      no such class: below one leaf a parent would implement something with
      exactly one subclass.
    - ``provider`` decides only which op backend executes and which
      configurations are eligible. That is held by the registered leaf, which is
      therefore small: an identity, a provider string, and a ``can_implement``.

    The name survives as the parent because 160-odd call sites, five
    ``isinstance`` gates, and the comments across the MoE tree name it. Two
    ``__class__ ==`` checks did too, and are now ``issubclass``
    (``create_moe.py``, ``modeling_qwen3_moe.py``) -- an equality check against
    a parent silently misses every leaf.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.

    MoE torch custom op:
        Only support min-latency mode now (SM100 Blackwell only).
        Quant: fp8 block scales quant and nvfp4 quant and w4a16_mxfp4 quant
            FusedMoE Op: routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    FusedMoE module:
        min-latency mode:
            dynamic quant + FusedMoe Op
            equals to: dynamic quant + routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
    AttentionDP should be turned off for min-latency mode.

    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
    There should be at lease `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
    """

    # ---- identity-derived declarations, set by each registered leaf ------
    #: ``MoEImplId.provider``, and the ``moe_op_backend`` registry key. Named
    #: once per leaf; ``__init__`` builds the op backend from it, so a leaf
    #: cannot end up executing a provider other than the one it publishes.
    provider: str
    #: ``provider == PROVIDER_FLASHINFER``, restated because it is read from
    #: outside as a plain attribute (``ConfigurableMoE`` passes it to the
    #: communication factory, and the backend tests read it off an instance).
    #: It used to be an instance field computed in ``__init__``; with provider
    #: in the identity it is a constant per leaf.
    use_flashinfer: bool
    #: Whether this leaf's quantization format has a fused cubin for the gpt-oss
    #: SwiGLU package (expert bias plus alpha/beta). Read by
    #: ``check_trtllm_gen_capabilities``; was the ``_GPTOSS_SUPPORTED_ALGOS``
    #: membership test inside the single ``can_implement``.
    supports_gptoss_style: bool = False
    #: Whether ``run_moe`` writes into the caller's workspace-backed output
    #: buffer. Was ``self.has_any_quant and not self.use_flashinfer``, read off
    #: an instance behind an assert that weights already existed; both halves
    #: are identity now.
    writes_moe_output_into_workspace: bool = False
    #: Whether the kernel this leaf runs takes top-k ids/scales from the host
    #: instead of routing internally. Was
    #: ``self.use_flashinfer and self._is_unquantized_path()``, which is true of
    #: exactly one leaf.
    provider_requires_separated_routing: bool = False

    # Inherited by all eleven leaves and restated in each descriptor from the
    # same module-level constants, so what the registry publishes and what the
    # scheduler reads cannot drift apart.
    capabilities = TRTLLM_GEN_CAPABILITIES
    input_requirement = TRTLLM_GEN_INPUT_REQUIREMENT

    # The set that used to live here as ``_SUPPORTED_QUANT_ALGOS`` is now the
    # ``quant`` segment of the eleven identities, and the gpt-oss subset is
    # ``supports_gptoss_style`` on each leaf. Neither survives as a set: a leaf
    # that admitted a format outside its own identity would run the wrong
    # kernel, and there is nothing left for one class to enumerate.

    # Activations supported by the FlashInfer BF16 kernels: Swiglu and Relu2.
    _BF16_SUPPORTED_ACTIVATIONS = {
        ActivationType.Swiglu,
        ActivationType.Relu2,
    }

    # Quantization algorithms with fused SiTu FC1 cubins in the TRTLLM-Gen
    # batched-GEMM kernel drop. NVFP4 feeds the group-16 `Bmm_E2m1_E2m1E2m1_...
    # _siTuGlu_*` kernels; W4A8_MXFP4_MXFP8 the group-32
    # `Bmm_MxE4m3_MxE2m1MxE4m3_..._siTuGlu_*` ones. There is no standalone
    # SiTu activation kernel, so anything without a fused cubin is rejected.
    _SITU_SUPPORTED_QUANT_ALGOS = {
        QuantAlgo.NVFP4,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    }

    # The fused-activation cubins index alpha/beta/clamp by expert
    # (``gemm1_alpha`` / ``gemm1_beta`` / per-expert clamp tensor). The clamp is
    # quant-dependent, so ``resolve_activation_support`` narrows it per instance.
    activation_support = MoEActivationSupport(
        kinds=frozenset({
            ActivationType.Swiglu,
            ActivationType.SwigluBias,
            ActivationType.Relu2,
            ActivationType.Silu,
            ActivationType.SiTu,
        }),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

    # ActivationType -> the batched-GEMM ``ActType`` encoding the cubins are
    # keyed by. SwigluBias shares the SwiGlu kernel (the gpt-oss constants
    # travel as separate per-expert tensors, not as a distinct act type).
    _TRTLLM_GEN_ACT_TYPE = {
        ActivationType.Swiglu: ActType_TrtllmGen.SwiGlu,
        ActivationType.SwigluBias: ActType_TrtllmGen.SwiGlu,
        ActivationType.Relu2: ActType_TrtllmGen.Relu2,
        ActivationType.Silu: ActType_TrtllmGen.Silu,
        ActivationType.SiTu: ActType_TrtllmGen.SiTu,
    }

    def resolve_activation_support(self) -> MoEActivationSupport:
        """Narrow the clamp ABI to what this instance's quant path reads.

        The DeepSeek FP8 block-scale path runs the clamp in a separate
        activation kernel (``DevKernel.cu::activationDeepSeekKernel``) that
        takes one ``double`` by value, so a per-expert tensor would be silently
        ignored there. Every other quant path consumes the per-expert tensor
        the class attribute declares.
        """
        support = type(self).activation_support
        # Reads ``quant_config`` rather than ``has_deepseek_fp8_block_scales``:
        # this runs while construction state is being installed, before
        # create_weights sets the ``_weights_created`` those properties assert.
        quant_config = self.quant_config
        if (quant_config is not None
                and quant_config.layer_quant_mode.has_fp8_block_scales()):
            return replace(support, limit=ActivationParamShape.UNIFORM_SCALAR)
        return support

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
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
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

        self._validate_situ_activation()

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
        self.num_fused_shared_expert = 0

        # Fusing the shared experts into the routed-expert grouped GEMM is opt-in:
        # set TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1 to enable it. The benefit is
        # workload-dependent (small decode batches gain, large prefill chunks lose the
        # aux-stream overlap of the unfused path), and the fused path additionally
        # restricts tactics to tileN>=32 to avoid a small-tile dynB kernel defect.
        fusion_enabled = os.environ.get("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION",
                                        "0") == "1"
        # Only the trtllm op backend implements fused shared experts
        on_trtllm_backend = isinstance(self.op_backend, TRTLLMOpBackend)
        # Expert parallelism (moe_ep_size > 1) is not supported by the fused path yet
        # (the routing kernel's shared-expert append assumes the full expert set is
        # local); gate it out here so EP configs fall back to the unfused path instead
        # of tripping the runtime EP check in the TRTLLM-Gen runner.
        fusion_supported = (
            fusion_enabled and on_trtllm_backend
            and model_config.mapping.dp_size == 1
            and model_config.mapping.moe_ep_size == 1
            and self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp8_block_scales())
        if fusion_supported:
            # Not all models that use this backend define shared experts (e.g. non-DeepSeek
            # MoEs), so fall back to 0 when the config has no `n_shared_experts`.
            self.num_fused_shared_expert = getattr(
                model_config.pretrained_config, "n_shared_experts", 0) or 0
            if self.num_fused_shared_expert > 0:
                logger.info_once(
                    f"Shared-expert fusion enabled: folding "
                    f"{self.num_fused_shared_expert} shared expert(s) into the "
                    f"routed-expert grouped GEMM.",
                    key="trtllm_gen_shared_expert_fusion")

        # create_weights must see the final fused-expert count so the fused shared
        # slots are allocated when fusion is enabled.
        if not model_config.skip_create_weights_in_init:
            self.create_weights()
        self.layer_idx = layer_idx

    def _to_trtllm_gen_activation_type(self,
                                       activation_type: ActivationType) -> int:
        act_type = self._TRTLLM_GEN_ACT_TYPE.get(
            ActivationType(activation_type))
        if act_type is None:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        return int(act_type)

    @property
    def is_situ_activation(self) -> bool:
        return self.activation.kind is ActivationType.SiTu

    def _validate_situ_activation(self) -> None:
        """Hardware / quant preconditions the SiTu cubins carry.

        Kind and constant shape are already settled: the activation carrier only
        admits two positive soft-caps for SiTu and no clamp, and
        ``install_activation_params`` has materialized them as the per-expert
        ``gemm1_alpha`` / ``gemm1_beta`` buffers the cubin indexes. What remains
        is that trtllm-gen ships SiTu for exactly one dtype/quant combination.
        """
        if not self.is_situ_activation:
            return
        if self.dtype != torch.bfloat16:
            raise ValueError(
                "TRTLLM-Gen SiTu requires bfloat16 activations, got "
                f"{self.dtype}.")
        if not is_sm_100f():
            raise ValueError("TRTLLM-Gen SiTu requires the SM100 family, got "
                             f"SM{get_sm_version()}.")
        quant_algo = (None if self.quant_config is None else
                      self.quant_config.quant_algo)
        if quant_algo not in self._SITU_SUPPORTED_QUANT_ALGOS:
            supported = ", ".join(
                sorted(algo.name for algo in self._SITU_SUPPORTED_QUANT_ALGOS))
            raise ValueError(
                f"TRTLLM-Gen SiTu requires one of {supported} quantization, "
                f"got {quant_algo}.")
        if self.tp_size > 1:
            # Intra-expert MoE TP: w1/w3 column-shard and w2 row-shard along
            # the intermediate dim (the stock MXFP4/NVFP4 quant-method loaders
            # slice the packed bytes and scales per rank). Require the
            # per-rank shard to stay a whole multiple of the quant method's
            # weight alignment so per-shard scale groups and the padded
            # weight buffers line up without fractional groups.
            if quant_algo == QuantAlgo.NVFP4:
                # NVFP4 picks its alignment from the layer shape in
                # create_weights (32 -> 128 or 256), which runs after this
                # check. Ask for the resolved value: the class attribute is
                # only the starting point, and validating against it would
                # admit shards the loader cannot lay out.
                alignment, _ = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(
                    self.hidden_size, self.intermediate_size_per_partition)
            else:
                # MXFP4's alignment is a fixed class attribute.
                alignment = (
                    W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.weight_alignment)
            if (self.intermediate_size % self.tp_size != 0
                    or self.intermediate_size_per_partition % alignment != 0):
                raise ValueError(
                    "TRTLLM-Gen SiTu MoE TP requires intermediate_size "
                    f"({self.intermediate_size}) divisible by moe_tp_size "
                    f"({self.tp_size}) with the per-rank shard a multiple of "
                    f"{alignment}, got "
                    f"{self.intermediate_size_per_partition}.")
        if self.bias:
            raise ValueError(
                "TRTLLM-Gen SiTu does not support expert bias; the cubin adds "
                "no FC1 bias before the soft-caps.")

    def _requires_separated_routing(self) -> bool:
        """Whether this leaf's kernel takes top-k from the host, not the logits.

        What used to be ``self.use_flashinfer and self._is_unquantized_path()``
        is now one class attribute: exactly one of the eleven leaves runs a
        kernel with no internal routing. The DeepSeekV3 exception stays a
        runtime test because routing is a per-layer argument, not part of the
        identity -- that kernel's separated variant has accuracy issues, so its
        fused form is used instead.
        """
        if not self.provider_requires_separated_routing:
            return False
        return not isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod)

    def _get_data_or_none(self, attr_name: str) -> Optional[torch.Tensor]:
        attr = getattr(self, attr_name, None)
        return attr.data if attr is not None else None

    def _supports_load_balancer(self) -> bool:
        """Whether separated routing (top-k outside the kernel) is used.

        ConfigurableMoE uses this flag to decide whether routing is separated
        (top-k ids/scales computed outside backend) or fused inside the kernel.
        BF16 FlashInfer path always requires separated routing.
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
        return (self._supports_load_balancer()
                or self.routing_method.requires_separated_routing
                or FORCE_SEPARATED_ROUTING)

    def _check_configs(self):
        assert not self.has_any_quant \
            or self.has_deepseek_fp8_block_scales \
            or self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8 \
            or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, \
            "TRTLLMGenFusedMoE only supports bf16 (FlashInfer), fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8 dtypes."

        if not self.has_any_quant:
            assert self.activation_type in self._BF16_SUPPORTED_ACTIVATIONS, \
                ("TRTLLMGenFusedMoE BF16 path only supports "
                 f"{[a.name for a in self._BF16_SUPPORTED_ACTIVATIONS]} activations, "
                 f"got {self.activation_type.name}.")
            assert not self.bias and self.act_alpha is None and self.act_beta is None and self.act_clamp is None, \
                "TRTLLMGenFusedMoE BF16 path does not support bias/swiglu custom parameters."

        if self.bias or self.act_alpha is not None or self.act_beta is not None:
            assert self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, \
                "TRTLLMGenFusedMoE supports bias and the alpha/beta activation constants only for nvfp4 and mxfp4 variants."
        if self.act_clamp is not None:
            # Whether the clamp arrives as a scalar or a per-expert tensor is
            # settled by ``resolve_activation_support``; which algorithms have a
            # clamp at all is a quant fact, so it stays here.
            assert self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 \
                or self.has_w4a8_mxfp4_mxfp8 or self.has_deepseek_fp8_block_scales, \
                "TRTLLMGenFusedMoE supports an activation clamp only for nvfp4, mxfp4, and fp8_block_scale variants."

        if self.is_situ_activation:
            if not isinstance(self.op_backend, TRTLLMOpBackend):
                raise ValueError(
                    "TRTLLM-Gen SiTu requires the native TRTLLM op backend.")
            if not (self.has_nvfp4 or self.has_w4a8_mxfp4_mxfp8):
                raise ValueError("TRTLLM-Gen SiTu requires the NVFP4 or "
                                 "W4A8_MXFP4_MXFP8 path.")
            expected_scaling_vector_size = 16 if self.has_nvfp4 else 32
            if self.scaling_vector_size != expected_scaling_vector_size:
                raise ValueError(
                    "TRTLLM-Gen SiTu requires scaling vector size "
                    f"{expected_scaling_vector_size} for this quantization "
                    f"mode, got {self.scaling_vector_size}.")
            # For SiTu these hold the backend-local activation parameters
            # (populated by create_weights, which runs before this check).
            for name in ("act_alpha", "act_beta"):
                value = getattr(self, name)
                if (value.dtype != torch.float32
                        or value.shape != (self.expert_size_per_partition, )
                        or not value.is_contiguous()):
                    raise ValueError(
                        f"{name} must be a contiguous float32 tensor with "
                        "one value per local expert/slot.")

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_block_scales(
        ):
            self.quant_method.create_weights(self, self.num_fused_shared_expert)
        else:
            self.quant_method.create_weights(self)

        # SiTu's two soft-caps ride in the same gemm1_alpha/gemm1_beta op slots
        # SwiGLU's alpha/beta use -- the kinds are mutually exclusive. They are
        # backend configuration, not checkpoint weights, so ``cache_derived_state``
        # refills them if meta-device materialization wipes them.
        if self.is_situ_activation:
            self.act_alpha = nn.Parameter(self.act_alpha, requires_grad=False)
            self.act_beta = nn.Parameter(self.act_beta, requires_grad=False)

        self._weights_created = True
        self._check_configs()

        if (self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8
                or self.has_w4a8_mxfp4_fp8
                or self.has_w4a8_mxfp4_mxfp8) and not self.bias:
            self.w3_w1_bias = nn.Parameter(torch.zeros(
                (self.w3_w1_weight.shape[0], self.w3_w1_weight.shape[1]),
                dtype=torch.float32),
                                           requires_grad=False)
            self.register_parameter("w3_w1_bias", self.w3_w1_bias)
            self.w2_bias = nn.Parameter(torch.zeros(
                (self.w2_weight.shape[0], self.w2_weight.shape[1]),
                dtype=torch.float32),
                                        requires_grad=False)
            self.register_parameter("w2_bias", self.w2_bias)

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        if self.is_situ_activation:
            # Reinitialize constants after meta-device materialization. These
            # are backend configuration, not checkpoint weights, so nothing
            # else refills them.
            #
            # Re-materialized from ``self.activation`` rather than from a
            # snapshot of the slots taken in create_weights: under meta init
            # those slots are themselves meta at that point, so the snapshot
            # would carry no values. ``self.activation`` is the declaration and
            # is always real.
            params = materialize_activation_params(
                self.activation,
                self.resolve_activation_support(),
                num_local_experts=self.expert_size_per_partition,
                device=self.act_alpha.device,
                owner=type(self).__name__,
            )
            self.act_alpha.data.copy_(params.alpha)
            self.act_beta.data.copy_(params.beta)

    def try_fused_route_quant(
        self,
        x: Union[torch.Tensor, MxFp8QuantizedTensor],
        router_logits: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor]]:
        """Fuse Kimi K3 no-aux routing and MXFP8 input quantization.

        This launch-overhead optimization is deliberately specialized to the
        K3 decode shape: the op below hardcodes 896 experts, top-16, hidden
        3584 and at most 64 tokens. The checks here mirror its ``TORCH_CHECK``s
        so a miss declines quietly instead of raising. Returning ``None`` keeps
        every other model, shape, architecture, and op backend on the existing
        unfused path.
        """
        if (os.environ.get("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", "0") == "1"
                or isinstance(x, MxFp8QuantizedTensor)):
            return None

        if (not is_sm_100f() or not self.has_w4a8_mxfp4_mxfp8
                or not isinstance(self.op_backend, TRTLLMOpBackend)
                or not isinstance(self.routing_method,
                                  DeepSeekV3MoeRoutingMethod)):
            return None

        routing = self.routing_method.routing_impl
        bias = self.routing_method.e_score_correction_bias
        if (not routing.is_fused or routing.n_group != 1
                or routing.topk_group != 1 or routing.top_k != 16
                or router_logits.ndim != 2 or router_logits.shape[1] != 896
                or router_logits.dtype != torch.float32
                or not router_logits.is_contiguous()
                or bias.dtype != torch.float32 or not bias.is_contiguous()
                or x.ndim != 2 or x.shape != (router_logits.shape[0], 3584)
                or not 0 < x.shape[0] <= 64 or x.dtype != torch.bfloat16
                or not x.is_contiguous()):
            return None

        return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
            router_logits, bias, x, routing.routed_scaling_factor)

    def supports_moe_output_in_alltoall_workspace(self):
        """Whether ``run_moe`` fills a caller-supplied output buffer.

        ``self.has_any_quant and not self.use_flashinfer`` reduced to one
        constant: the only unquantized leaf is a FlashInfer one, so the old
        conjunction is exactly "this is a native-provider leaf", which the
        provider mixin states once instead of eleven times.
        """
        return self.writes_moe_output_into_workspace

    def _extract_routing_params(self) -> RoutingParams:
        if isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.routing_impl.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=self.routing_method.routing_impl.n_group,
                topk_group=self.routing_method.routing_impl.topk_group,
                routed_scaling_factor=self.routing_method.routing_impl.
                routed_scaling_factor,
            )
        elif isinstance(self.routing_method, MiniMaxM3MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=self.routing_method.routed_scaling_factor,
            )
        elif isinstance(self.routing_method, MiniMaxM2MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
            )
        elif isinstance(self.routing_method, DeepSeekV4MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=self.routing_method.n_group,
                topk_group=self.routing_method.topk_group,
                routed_scaling_factor=self.routing_method.routed_scaling_factor,
            )
        else:
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=None,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
            )

    def fuse_shared_expert(self, shared_experts: GatedMLP):
        assert self._weights_created
        self.quant_method.fuse_shared_expert(self, shared_experts,
                                             self.num_fused_shared_expert)

    def _prepare_kernel_inputs(self, ctx: MoERunContext) -> KernelInputs:
        """Resolve the arguments every leaf's kernel call shares."""
        plan = require_comm_plan(self, ctx)

        # The caller used to apply this filter before handing over the kwargs.
        if self._routes_outside_the_kernel():
            if ctx.router_logits is not None and ctx.token_selected_experts is None:
                raise ValueError(
                    f"{type(self).__name__} requires separated routing for this "
                    "config, so ctx.router_logits is ignored, but "
                    "ctx.token_selected_experts is None -- there is nothing left "
                    "to route with. Supply precomputed top-k ids and scales.")
            router_logits = None
        else:
            router_logits = ctx.router_logits

        routing_params = self._extract_routing_params()
        top_k = routing_params.top_k
        if ctx.token_selected_experts is not None:
            # for cases like deepep low latency where fake top_k=1 might be used
            top_k = ctx.token_selected_experts.shape[-1]

        x_sf = ctx.x_sf
        if x_sf is not None:
            # Ensure x_sf is 2D before flattening
            assert len(
                x_sf.shape
            ) == 2, f"x_sf should be 2D tensor, got shape {x_sf.shape}"
            x_sf = x_sf.flatten()

        return KernelInputs(
            x=ctx.x,
            x_sf=x_sf,
            router_logits=router_logits,
            token_selected_experts=ctx.token_selected_experts,
            token_final_scales=ctx.token_final_scales,
            moe_output=plan.moe_output,
            do_finalize=ctx.do_finalize,
            top_k=top_k,
            routing_bias=(routing_params.routing_bias
                          if router_logits is not None else None),
            n_group=routing_params.n_group,
            topk_group=routing_params.topk_group,
            routed_scaling_factor=routing_params.routed_scaling_factor,
        )

    def _unfinalized(self, outputs):
        """Hand back per-expert outputs for the caller to combine."""
        assert not self.reduce_results, \
            "reduce_results must be False when do_finalize is False"
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
        if do_finalize:
            # TRTLLMGenFusedMoE only supports bfloat16 output
            return super().forward_fake(x,
                                        router_logits,
                                        do_finalize=do_finalize,
                                        output_dtype=torch.bfloat16,
                                        all_rank_num_tokens=all_rank_num_tokens,
                                        use_dp_padding=use_dp_padding,
                                        **kwargs)
        else:
            is_deepseek_v3_routing = isinstance(self.routing_method,
                                                DeepSeekV3MoeRoutingMethod)
            is_minimax_routing = isinstance(self.routing_method,
                                            MiniMaxM2MoeRoutingMethod)
            top_k = self.routing_method.routing_impl.top_k if is_deepseek_v3_routing else self.routing_method.top_k
            routing_bias = self.routing_method.e_score_correction_bias if (
                is_deepseek_v3_routing or is_minimax_routing) else None
            return fp4_block_scale_fake_output_without_finalize(
                x,
                self.num_experts,
                top_k,
                routing_bias,
            )


# ---------------------------------------------------------------------------
# Provider mixins
# ---------------------------------------------------------------------------
#
# Three constants that follow from the provider alone. As mixins rather than
# per-leaf attributes because "use_flashinfer == (provider is flashinfer)" is an
# invariant, and eleven copies of it is eleven chances to break it. They carry
# no methods and do not subclass ``MoEImplBase``: a leaf is
# ``(provider mixin, per-quant class)``, and only the second half is an impl.


class TrtllmProviderMixin:
    """The native TRT-LLM cubins, reached through ``TRTLLMOpBackend``."""

    provider = PROVIDER_TRTLLM
    use_flashinfer = False
    # Every native leaf is quantized (the unquantized kernel exists only in the
    # FlashInfer wheel), which is what the old
    # ``has_any_quant and not use_flashinfer`` amounted to.
    writes_moe_output_into_workspace = True


class FlashinferProviderMixin:
    """The same algorithm as shipped in the FlashInfer wheel."""

    provider = PROVIDER_FLASHINFER
    use_flashinfer = True
    writes_moe_output_into_workspace = False


# ---------------------------------------------------------------------------
# Per-quant abstract classes
# ---------------------------------------------------------------------------


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
        factor = 1 if act_type in [
            ActType_TrtllmGen.Relu2, ActType_TrtllmGen.Silu
        ] else 2
        intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
            -2] // factor
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
                self.quant_method, 'intermediate_size_per_partition_lean',
                None),
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
        if k.moe_output is None and final_hidden_states.shape[
                1] > self.hidden_size:
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
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
            self.activation_type, self.act_alpha is not None)
        return (NVFP4TRTLLMGenFusedMoEMethod()
                if needs_padded_method else NVFP4TRTLLMGenFusedMoEBaseMethod())

    def quantize_input(self, x, post_quant_comm: bool = True):
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
            x_row = x.shape[0]
            x, x_sf = x.fp4_tensor, x.scaling_factor
        elif isinstance(x, MxFp8QuantizedTensor):
            assert not x.is_sf_swizzled, "MxFp8QuantizedTensor should not be swizzled before communication"
            x_row = x.shape[0]
            x, x_sf = x.fp8_tensor, x.scaling_factor
        else:
            # Apply pre_quant_scale if it exists (for NVFP4_AWQ)
            # fc31_act_scale shape: (1, hidden_size)
            # x shape: (num_tokens, hidden_size)
            if hasattr(self,
                       'fc31_act_scale') and self.fc31_act_scale is not None:
                x = x * self.fc31_act_scale

            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            if pad_size > 0:
                x = torch.nn.functional.pad(x, (0, pad_size))

            x_row = x.shape[0]
            x, x_sf = self.op_backend.fp4_quantize(x, self.fc31_input_scale,
                                                   self.scaling_vector_size,
                                                   False, False)
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
            x, False, alignment=self.quant_method.input_hidden_alignment)
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


# ---------------------------------------------------------------------------
# Registered leaves -- trtllm provider (TRTLLM-14968)
# ---------------------------------------------------------------------------


@register_moe_impl
class TrtllmTrtllmGenNvfp4Impl(TrtllmProviderMixin, TRTLLMGenNvfp4Base):
    """``trtllm.trtllm_gen.fused_moe.nvfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "nvfp4",
        "TRTLLM-Gen batched-GEMM cubins over NVFP4, SM100 family.")

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)


@register_moe_impl
class TrtllmTrtllmGenFp8BlockScalesImpl(TrtllmProviderMixin,
                                        TRTLLMGenFp8BlockScalesBase):
    """``trtllm.trtllm_gen.fused_moe.fp8_block_scales``.

    ``supports_gptoss_style`` stays False: this format's separate-activation
    path takes the DSV4-style scalar clamp, but no bias and no alpha/beta.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "fp8_block_scales",
        "TRTLLM-Gen batched-GEMM cubins over FP8 1x128 block scales, SM100 family."
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)


@register_moe_impl
class TrtllmTrtllmGenW4a16Mxfp4Impl(TrtllmProviderMixin,
                                    TRTLLMGenW4a16Mxfp4Base):
    """``trtllm.trtllm_gen.fused_moe.w4a16_mxfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "w4a16_mxfp4",
        "TRTLLM-Gen batched-GEMM cubins over MXFP4 weights with bf16 activations."
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)


@register_moe_impl
class TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl(TrtllmProviderMixin,
                                        TRTLLMGenW4a8Mxfp4Mxfp8Base):
    """``trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "w4a8_mxfp4_mxfp8",
        "TRTLLM-Gen batched-GEMM cubins over MXFP4 weights with MXFP8 activations."
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)


@register_moe_impl
class TrtllmTrtllmGenW4a8Nvfp4Fp8Impl(TrtllmProviderMixin, TRTLLMGenFusedMoE):
    """``trtllm.trtllm_gen.fused_moe.w4a8_nvfp4_fp8``.

    No per-quant parent: this format has one provider, so a parent would
    implement three methods for exactly one subclass. It also calls its runner
    op directly instead of going through ``self.op_backend`` -- the FlashInfer
    wheel has no equivalent, which is the same fact from the other side.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "w4a8_nvfp4_fp8",
        "TRTLLM-Gen fp8_fp4 block-scale runner: NVFP4 weights, FP8 activations."
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def _get_quant_method(self):
        return W4A8NVFP4FP8TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, 1.0 / self.fc31_input_scale)
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = self._prepare_kernel_inputs(ctx)

        outputs = torch.ops.trtllm.fp8_fp4_block_scale_moe_runner(
            k.router_logits,
            k.routing_bias,
            k.x,
            self.w3_w1_weight,
            self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
            self.w2_weight,
            self.w2_weight_scale.view(torch.float8_e4m3fn),
            self.fc31_scale_c.data,
            self.fc31_alpha.data,
            self.fc2_alpha.data,
            self.num_slots,
            k.top_k,
            k.n_group,
            k.topk_group,
            self.intermediate_size_per_partition,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            do_finalize=k.do_finalize,
            act_type=0,
            topk_weights=k.token_final_scales,
            topk_ids=k.token_selected_experts,
            output=k.moe_output,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )

        if not k.do_finalize:
            return self._unfinalized(outputs)
        # When output is provided, use it directly as the result
        return k.moe_output if k.moe_output is not None else outputs[0]


@register_moe_impl
class TrtllmTrtllmGenW4a8Mxfp4Fp8Impl(TrtllmProviderMixin, TRTLLMGenFusedMoE):
    """``trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_fp8``.

    Single-provider like the NVFP4/FP8 leaf above, and likewise on its own
    runner op rather than ``self.op_backend``.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_TRTLLM, "w4a8_mxfp4_fp8",
        "TRTLLM-Gen e4m3_mxe2m1 block-scale runner: MXFP4 weights, FP8 activations."
    )

    supports_gptoss_style = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d)

    def _get_quant_method(self):
        return W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()

    def quantize_input(self, x, post_quant_comm: bool = True):
        pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad_size))
        # Two static per-tensor scales, one per side of the fused FC1: the
        # post-communication path dequantizes with the gate-less scale.
        scale = (self.fc31_input_dequant[0]
                 if post_quant_comm else self.fc31_input_gate_dequant[0])
        x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)
        return x, None

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        k = self._prepare_kernel_inputs(ctx)

        intermediate_size_per_partition_padded = self.w3_w1_weight.shape[-2] // 2

        result = torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner(
            k.router_logits,
            k.routing_bias,
            k.x,
            self.w3_w1_weight,
            self.w3_w1_weight_scale,
            self.w3_w1_bias,
            self.act_alpha,
            self.act_beta,
            self.act_clamp,
            self.w2_weight,
            self.w2_weight_scale,
            self.w2_bias,
            self.fc31_input_dequant,
            self.fc31_input_gate_dequant,
            self.fc2_input_dequant,
            self.num_slots,
            k.top_k,
            k.n_group,
            k.topk_group,
            intermediate_size_per_partition_padded,
            self.hidden_size,
            self.quant_method.intermediate_size_per_partition_lean,
            self.slot_start,
            self.expert_size_per_partition,
            k.routed_scaling_factor,
            self.routing_method.routing_method_type,
            0,  # act_type
            k.token_final_scales,
            k.token_selected_experts,
            output=k.moe_output,
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )
        # When output is provided, use it directly as the result
        if k.moe_output is not None:
            return k.moe_output
        return result[:, :self.hidden_size].contiguous()


# ---------------------------------------------------------------------------
# Registered leaves -- flashinfer provider (TRTLLM-14969)
# ---------------------------------------------------------------------------


@register_moe_impl
class FlashinferTrtllmGenNvfp4Impl(FlashinferProviderMixin, TRTLLMGenNvfp4Base):
    """``flashinfer.trtllm_gen.fused_moe.nvfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER, "nvfp4",
        "FlashInfer's TRTLLM-Gen NVFP4 fused MoE, SM100 family.")

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d,
                                     check_flashinfer_provider(cls, p, d),
                                     cls._check_shape(p, d))

    @classmethod
    def _check_shape(cls, p: MoEProblem,
                     d: MoEDeployment) -> Optional[MoEEligibility]:
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
        if not nvfp4_needs_padded_method(p.activation_type, "alpha"
                                         in p.activation_constants):
            return None
        return check_mxfp4_flashinfer_shape(
            cls,
            p,
            d,
            weight_alignment=NVFP4TRTLLMGenFusedMoEMethod.weight_alignment,
            input_hidden_alignment=NVFP4TRTLLMGenFusedMoEMethod.
            input_hidden_alignment,
        )


@register_moe_impl
class FlashinferTrtllmGenFp8BlockScalesImpl(FlashinferProviderMixin,
                                            TRTLLMGenFp8BlockScalesBase):
    """``flashinfer.trtllm_gen.fused_moe.fp8_block_scales``.

    No alignment gate: the old check reached its unconditional ``return True``
    for this format, because the padding rules it enforced belong to the FP4
    weight layouts.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER, "fp8_block_scales",
        "FlashInfer's TRTLLM-Gen FP8 block-scale fused MoE, SM100 family.")

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d,
                                     check_flashinfer_provider(cls, p, d))


@register_moe_impl
class FlashinferTrtllmGenW4a16Mxfp4Impl(FlashinferProviderMixin,
                                        TRTLLMGenW4a16Mxfp4Base):
    """``flashinfer.trtllm_gen.fused_moe.w4a16_mxfp4``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER, "w4a16_mxfp4",
        "FlashInfer's TRTLLM-Gen MXFP4-weight fused MoE with bf16 activations.")

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(
            cls, p, d, check_flashinfer_provider(cls, p, d),
            check_mxfp4_flashinfer_shape(
                cls,
                p,
                d,
                weight_alignment=W4A16MXFP4TRTLLMGenFusedMoEMethod.
                weight_alignment,
                input_hidden_alignment=W4A16MXFP4TRTLLMGenFusedMoEMethod.
                input_hidden_alignment,
            ))


@register_moe_impl
class FlashinferTrtllmGenW4a8Mxfp4Mxfp8Impl(FlashinferProviderMixin,
                                            TRTLLMGenW4a8Mxfp4Mxfp8Base):
    """``flashinfer.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8``."""

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER, "w4a8_mxfp4_mxfp8",
        "FlashInfer's TRTLLM-Gen MXFP4-weight fused MoE with MXFP8 activations."
    )

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(
            cls, p, d, check_flashinfer_provider(cls, p, d),
            check_mxfp4_flashinfer_shape(
                cls,
                p,
                d,
                weight_alignment=W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.
                weight_alignment,
                input_hidden_alignment=W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.
                input_hidden_alignment,
            ))


@register_moe_impl
class FlashinferTrtllmGenBf16Impl(FlashinferProviderMixin, TRTLLMGenFusedMoE):
    """``flashinfer.trtllm_gen.fused_moe.none`` -- the unquantized path.

    FlashInfer-exclusive, and the one leaf reached without the opt-in flag:
    ``trtllm_bf16_moe`` has no counterpart in TRT-LLM's own cubins, so there is
    no native leaf to prefer and nothing for a flag to switch between. It is
    also the only leaf whose kernel does not route internally, which is what
    ``provider_requires_separated_routing`` says.
    """

    descriptor = trtllm_gen_descriptor(
        PROVIDER_FLASHINFER, "none",
        "FlashInfer's TRTLLM-Gen bf16 fused MoE, unquantized, SM100 family.")

    provider_requires_separated_routing = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        return check_trtllm_gen_leaf(cls, p, d, cls._check_bf16_path(p, d))

    @classmethod
    def _check_bf16_path(cls, p: MoEProblem,
                         d: MoEDeployment) -> Optional[MoEEligibility]:
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support bias/swiglu custom parameters."
            )
        # Same set _check_configs asserts on, so the verdict and the
        # constructor agree instead of failing later at create_weights.
        if p.activation_type not in cls._BF16_SUPPORTED_ACTIVATIONS:
            supported = ", ".join(
                sorted(activation.name
                       for activation in cls._BF16_SUPPORTED_ACTIVATIONS))
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} only supports {supported} activations, got "
                f"{p.activation}")
        # Stronger than MoEDep.FLASHINFER: the wheel has to expose the two
        # bf16 entry points, which older ones do not.
        if not d.env.has_dep(MoEDep.FLASHINFER_BF16_MOE):
            return _reject(
                MoERejectReason.DEP_MISSING,
                f"{cls.__name__} requires FlashInfer fused MoE with "
                "trtllm_bf16_moe support.")
        # FlashInfer BF16 kernels require the per-rank intermediate size
        # to be a multiple of 128.
        if p.intermediate_size is not None:
            inter = p.intermediate_size
            if d.tp_size > 1:
                if inter % d.tp_size != 0:
                    return _reject(
                        MoERejectReason.SHAPE_UNALIGNED,
                        f"{cls.__name__} requires intermediate_size ({inter}) "
                        f"divisible by moe_tp_size ({d.tp_size})")
                inter = inter // d.tp_size
            if inter % 128 != 0:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED, f"{cls.__name__} requires "
                    "intermediate_size_per_partition % 128 == 0; "
                    f"got {inter} "
                    f"(full intermediate_size={p.intermediate_size}, "
                    f"moe_tp_size={d.tp_size})")
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
        k = self._prepare_kernel_inputs(ctx)

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
            gated_act_type=self._to_trtllm_gen_activation_type(
                self.activation_type),
            output=k.moe_output,
            use_shuffled_weight=getattr(self.quant_method,
                                        "use_shuffled_weight", False),
            weight_layout=getattr(self.quant_method, "weight_layout", 0),
            do_finalize=k.do_finalize,
        )
        if not k.do_finalize:
            return self._unfinalized(result)
        return result
