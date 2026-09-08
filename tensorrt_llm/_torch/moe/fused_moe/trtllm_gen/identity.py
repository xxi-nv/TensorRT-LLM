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
"""What a TRTLLM-Gen leaf *is*: the eleven identities, in three forms.

The values, the provider segment written as class attributes, and the factory
that stamps the whole thing into a registry descriptor. One subject, so one
module -- and the module every leaf imports, which is why nothing that varies
by quantization format is allowed in here.

Not to be confused with ``..impl_identity``, which holds the machinery every
backend family shares (``MoEImplId``, ``MoEImplDescriptor``, the registry).
This file holds TRTLLM-Gen's values of those types. It is the bottom of the
subpackage: every other module here reads from it and it imports nothing from
the package, so deleting the deprecated ``fused_moe_trtllm_gen`` module path
above moves nothing.
"""

import torch

from ..impl_contract import MoEInputRequirement, MoEStaticCapability
from ..impl_identity import MoEImplDescriptor, MoEImplId
from ..interface import MoESchedulerKind

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

# Declared once and referenced from both the base class attributes and every
# descriptor. All eleven publish the same two, so a per-leaf literal would be
# eleven chances for the published contract and the executed one to disagree.
TRTLLM_GEN_CAPABILITIES = MoEStaticCapability(supports_expert_bias=True, supports_eplb=True)

# bfloat16 routing scales are what these kernels read, and the DeepEP
# dispatch has to mark unfilled rows before they reach them.
TRTLLM_GEN_INPUT_REQUIREMENT = MoEInputRequirement(
    routing_scales_dtype=torch.bfloat16,
    requires_sanitized_expert_ids=True,
    # The combine reduction runs in bf16 regardless of the model's output
    # dtype, so the NVLink one-sided payload buffer must be bf16 too.
    onesided_workspace_dtype=torch.bfloat16,
)


class TrtllmProviderTraits:
    """The native TRT-LLM cubins, reached through ``TRTLLMOpBackend``."""

    provider = PROVIDER_TRTLLM
    use_flashinfer = False


class FlashinferProviderTraits:
    """The same algorithm as shipped in the FlashInfer wheel."""

    provider = PROVIDER_FLASHINFER
    use_flashinfer = True


# The two classes above are the provider segment as class attributes, named
# once per provider rather than once per leaf because
# ``use_flashinfer == (provider is flashinfer)`` is an invariant and eleven
# copies of it is eleven chances to break it.
#
# Traits rather than mixins: they contribute no behavior, only values for class
# attributes ``TrtllmGenFusedMoEBase`` declares and deliberately leaves unset.
# They also do not subclass ``MoEImplBase`` -- a leaf is ``(provider traits,
# per-quant class)`` and only the second half is an impl. They sit first in the
# base list so their values win the MRO over the family base's defaults.


def trtllm_gen_descriptor(provider: str, quant: str, doc: str) -> MoEImplDescriptor:
    """Build one leaf's descriptor; only provider and quant ever differ.

    A factory because this family has eleven identities to publish. The other
    backend families each have exactly one and build ``MoEImplDescriptor``
    inline, so there is no shared helper to reach for and nothing above this
    package for this to live in.

    Distinct from :mod:`.eligibility`, which is where a leaf says what it will
    *accept*: the registry reads a descriptor at import time, and calls a gate
    only at resolution time.
    """
    return MoEImplDescriptor(
        identity=MoEImplId(provider, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=TRTLLM_GEN_CAPABILITIES,
        input_requirement=TRTLLM_GEN_INPUT_REQUIREMENT,
        doc=doc,
    )
