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
"""The eligibility gates the leaves compose into ``can_implement``.

Free functions rather than methods on :class:`.TrtllmGenFusedMoEBase`: the
family base must not implement ``can_implement``, or all eleven leaves would
inherit an answer that belongs to no single identity. They live in their own
module because the family base never calls them -- every caller is a leaf.

Composition stays explicit at each leaf rather than moving into a shared base.
Nothing here is guaranteed to stay universal: a future kernel may serve a
different SM set or evolve at a different granularity, and a leaf that spells
out which gates it runs can diverge without first having to disentangle itself
from an inherited default.

Each reads only ``cls``, ``MoEProblem`` and ``MoEDeployment``. No
``get_sm_version()``, no ``os.environ``, and no import probe, so an offline tuner
on a GPU-less host gets the same verdict a serving process does.
"""

from typing import Optional

import torch

from tensorrt_llm._utils import is_sm_100f

from ....utils import ActivationType
from ..impl_contract import MoEDeployment, MoEEligibility, MoEProblem, MoERejectReason
from ..impl_environment import MoEDep, MoEEnvFlag
from ..interface import _reject


def check_trtllm_gen_capabilities(
    cls: type, p: MoEProblem, d: MoEDeployment
) -> Optional[MoEEligibility]:
    """Capability gates every TRTLLM-Gen leaf shares, or ``None`` to admit.

    Eleven leaves restating the same three gates is how the SM and dtype answers
    drift apart, so each leaf checks its own quantization format and then defers
    here.
    """
    # The cubin drop is sm_100f (family-compatible) plus arch-specific
    # sm_100a/sm_103a, so the whole SM100 family is servable; the C++
    # selector (KernelRunner.cpp isSMCompatible) picks sm_100f on family
    # members without their own arch build.
    if not is_sm_100f(d.env.sm):
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} requires the SM100 family, got SM{d.env.sm}",
        )

    # run_moe asserts x.dtype == torch.bfloat16
    if p.dtype_act != torch.bfloat16:
        return _reject(
            MoERejectReason.DTYPE_UNSUPPORTED,
            f"{cls.__name__} only supports bfloat16 activation, got {p.dtype_act}",
        )

    if d.smart_router:
        return _reject(
            MoERejectReason.TOPOLOGY_UNSUPPORTED,
            f"{cls.__name__} has no smart-router path (moe_cluster_size={d.cluster_size})",
        )

    # Whether the gpt-oss SwiGLU package (expert bias plus alpha/beta) has a
    # fused cubin is a per-quant fact, so the leaf declares the answer and this
    # gate only applies it.
    if p.swiglu_gptoss_style and not cls.supports_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} has no fused bias/swiglu-parameter cubin for its "
            f"quantization format (quant={p.quant})",
        )

    # SiTu exists only as a fused FC1 epilogue -- there is no standalone
    # activation kernel to fall back to -- and reaching one takes both a cubin
    # family that ships it and a provider that calls it, so unlike the gpt-oss
    # package above the leaf and not the format holds the answer. Read here so
    # a resolution query is turned down rather than admitted and then raised on
    # by the constructor.
    if p.activation_type is ActivationType.SiTu and not cls.supports_situ:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} reaches no fused SiTu cubin (quant={p.quant}, "
            f"provider={cls.provider})",
        )

    return None


def check_flashinfer_provider(
    cls: type, p: MoEProblem, d: MoEDeployment
) -> Optional[MoEEligibility]:
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
        return _reject(MoERejectReason.DEP_MISSING, f"{cls.__name__} requires the FlashInfer wheel")

    # Opt-in, not a capability: the trtllm provider serves these formats too,
    # so routing traffic here without being asked would change which kernel a
    # previously-working deployment runs.
    if d.env.env_flag(MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER) != "1":
        return _reject(
            MoERejectReason.PATH_NOT_ENABLED,
            f"{cls.__name__} is opt-in; set "
            f"{MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER.value}=1 to select the "
            f"FlashInfer provider for quantized TRTLLM-Gen",
        )

    # SiTu is a native TRTLLM-Gen cubin and is absent from FlashInfer's
    # activation enum; Relu2 has no FlashInfer path either.
    if p.activation_type in (ActivationType.SiTu, ActivationType.Relu2):
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} does not implement {p.activation} "
            f"(FlashInfer's activation enum has no such kernel)",
        )

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
        "DeepSeekV3",
        "Default",
    ):
        return _reject(
            MoERejectReason.ROUTING_UNSUPPORTED,
            f"{cls.__name__} does not implement {p.routing} routing",
        )

    return None


def check_mxfp4_flashinfer_shape(
    cls: type,
    p: MoEProblem,
    d: MoEDeployment,
    *,
    weight_alignment: int,
    input_hidden_alignment: int,
) -> Optional[MoEEligibility]:
    """Per-rank shard alignment the FlashInfer path needs, or ``None``.

    Absent shapes abstain rather than reject: a call site that did not say what
    ``hidden_size`` is has not said the layer is misaligned.
    """
    if p.bias:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} takes no expert bias on the FlashInfer provider",
        )

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
        if inter % weight_alignment != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"{cls.__name__} requires intermediate_size_per_partition "
                f"({inter}) to be a multiple of {weight_alignment}",
            )

    if p.hidden_size is not None and p.hidden_size % input_hidden_alignment != 0:
        return _reject(
            MoERejectReason.SHAPE_UNALIGNED,
            f"{cls.__name__} requires hidden_size ({p.hidden_size}) to be a "
            f"multiple of {input_hidden_alignment}",
        )

    return None


def nvfp4_needs_padded_method(activation_type: ActivationType, has_alpha_constant: bool) -> bool:
    """Whether NVFP4 needs the padded quant method rather than the base one.

    One definition read from two sides: ``can_implement`` asks it of the
    problem, ``_get_quant_method`` of the instance. The two answers decide
    whether the alignment gates above apply, so a second copy would let a
    configuration be admitted by one and laid out by the other.
    """
    return (
        has_alpha_constant
        or activation_type is ActivationType.SiTu
        or activation_type in (ActivationType.Relu2, ActivationType.Silu)
    )


def identity_quant_of(cls: type) -> str:
    """The single format ``cls`` publishes, spelled as the identities spell it.

    Read off ``cls.descriptor`` rather than restated as a set, so the format a
    leaf admits and the format it publishes are the same string. This replaces
    the ``_SUPPORTED_QUANT_ALGOS`` membership test, which had to admit all
    seven because one class served all seven.
    """
    return cls.descriptor.identity.quant


def normalize_quant(quant: Optional[str]) -> str:
    """A ``canonical_quant`` result as an identity spells it.

    ``canonical_quant`` yields the ``QuantAlgo`` value, which is upper case,
    and ``None`` for unquantized; identities are lower case and spell that
    ``"none"``. One definition because the same comparison is made twice, at
    two different times: on a problem before the class is picked, and on the
    instance once its layer's quant config is final.
    """
    return "none" if quant is None else quant.lower()


def check_quant_matches_identity(cls: type, p: MoEProblem) -> Optional[MoEEligibility]:
    """Reject any format other than the one in this leaf's own identity."""
    expected = identity_quant_of(cls)
    actual = normalize_quant(p.quant)
    if actual != expected:
        return _reject(
            MoERejectReason.QUANT_UNSUPPORTED,
            f"{cls.__name__} implements quant={expected}, got {actual}",
        )
    return None


def check_trtllm_gen_leaf(
    cls: type, p: MoEProblem, d: MoEDeployment, *provider_gates: Optional[MoEEligibility]
) -> MoEEligibility:
    """Compose one leaf's verdict: identity, shared capability, then provider.

    First rejection wins, and the order is fixed so that two leaves differing
    only in provider give the same reason for a problem neither can serve.
    ``provider_gates`` are already-evaluated verdicts rather than callables:
    every gate is a pure function of ``p`` and ``d``, so there is nothing to
    defer, and passing values keeps the leaf bodies to one expression.
    """
    for verdict in (
        check_quant_matches_identity(cls, p),
        check_trtllm_gen_capabilities(cls, p, d),
        *provider_gates,
    ):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()
