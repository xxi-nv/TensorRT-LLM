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
"""Everything TRTLLM-Gen: the eleven registered leaves and the layers under them.

Self-contained, so the deprecated ``..fused_moe_trtllm_gen`` module path above
it holds nothing but a name. Bottom to top:

* :mod:`.identity` -- what a leaf *is*, in the three forms every leaf needs:
  the provider / technique / kernel strings and the two published contracts as
  values, the provider segment as trait classes, and the factory that stamps
  the whole thing into a registry descriptor. Imports nothing from here, and
  holds nothing that varies by quantization format.
* :mod:`.family` -- :class:`.TrtllmGenFusedMoEBase`, the abstract root all
  eleven share: construction, the weight-creation skeleton, the routing
  predicates the framework reads off a module, fake-output shapes. Branches on
  neither quant nor provider; both axes are hooks it declares and the layers
  below override.
* :mod:`.quant_bases` -- one class per format two providers share, holding
  ``run_moe`` and everything downstream of the weight layout. A format with a
  single provider has none and its leaf inherits the family base directly,
  which is why nothing all eleven need may live here.
* :mod:`.eligibility` -- the checks each ``can_implement`` composes. Beside
  the leaves rather than under the family base, because neither the family
  base nor a quant base calls a gate; every caller is a leaf.
* :mod:`.kernel_inputs` -- the ``run_moe`` prologue, as free functions.

One module per leaf, named ``<provider>_<quant>``, so the file name spells out
the two identity segments that actually differ across the eleven. The technique
and kernel segments are ``trtllm_gen`` / ``fused_moe`` for all of them, which is
what this package name says once.

Importing this package is what registers the leaves, which is why
``trtllm_gen_leaf`` is defined here: the lookup is only correct once every leaf
module has been imported, and at the bottom of this file it is.

* ``trtllm`` provider (TRTLLM-14968): the native cubins, six formats.
* ``flashinfer`` provider (TRTLLM-14969): the same algorithm out of the
  FlashInfer wheel, five formats. NVFP4, FP8_BLOCK_SCALES, W4A16_MXFP4 and
  W4A8_MXFP4_MXFP8 exist on both providers; W4A8_NVFP4_FP8 and W4A8_MXFP4_FP8
  are native-only, and the unquantized bf16 path is FlashInfer-only.
"""

from typing import Optional

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..impl_contract import canonical_quant
from ..impl_identity import MOE_IMPL_REGISTRY, MoEImplId
from .eligibility import (
    check_flashinfer_provider,
    check_mxfp4_flashinfer_shape,
    check_quant_matches_identity,
    check_trtllm_gen_capabilities,
    check_trtllm_gen_leaf,
    normalize_quant,
    nvfp4_needs_padded_method,
)
from .family import TrtllmGenFusedMoEBase
from .flashinfer_bf16 import FlashinferTrtllmGenBf16Impl
from .flashinfer_fp8_block_scales import FlashinferTrtllmGenFp8BlockScalesImpl
from .flashinfer_nvfp4 import FlashinferTrtllmGenNvfp4Impl
from .flashinfer_w4a8_mxfp4_mxfp8 import FlashinferTrtllmGenW4a8Mxfp4Mxfp8Impl
from .flashinfer_w4a16_mxfp4 import FlashinferTrtllmGenW4a16Mxfp4Impl
from .identity import (
    KERNEL_FUSED_MOE,
    PROVIDER_FLASHINFER,
    PROVIDER_TRTLLM,
    TECHNIQUE_TRTLLM_GEN,
    TRTLLM_GEN_CAPABILITIES,
    TRTLLM_GEN_INPUT_REQUIREMENT,
    FlashinferProviderTraits,
    TrtllmProviderTraits,
    trtllm_gen_descriptor,
)
from .quant_bases import (
    TRTLLMGenFp4BlockScaleBase,
    TRTLLMGenFp8BlockScalesBase,
    TRTLLMGenNvfp4Base,
    TRTLLMGenW4a8Mxfp4Mxfp8Base,
    TRTLLMGenW4a16Mxfp4Base,
)
from .trtllm_fp8_block_scales import TrtllmTrtllmGenFp8BlockScalesImpl
from .trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl
from .trtllm_w4a8_mxfp4_fp8 import TrtllmTrtllmGenW4a8Mxfp4Fp8Impl
from .trtllm_w4a8_mxfp4_mxfp8 import TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
from .trtllm_w4a8_nvfp4_fp8 import TrtllmTrtllmGenW4a8Nvfp4Fp8Impl
from .trtllm_w4a16_mxfp4 import TrtllmTrtllmGenW4a16Mxfp4Impl


def trtllm_gen_leaf(quant_algo: Optional[QuantAlgo], *, provider: Optional[str] = None) -> type:
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

    Raises when no leaf publishes the format, for callers that are about to use
    the class. A caller enumerating formats -- a parameter generator deciding
    what is worth running -- wants ``find_trtllm_gen_leaf`` instead: absence is
    an ordinary answer there, and reading it off an exception makes a registry
    regression look like a case that was never meant to run.
    """
    cls = find_trtllm_gen_leaf(quant_algo, provider=provider)
    if cls is not None:
        return cls
    registered = sorted(
        identity.canonical()
        for identity in MOE_IMPL_REGISTRY.identities()
        if identity.technique == TECHNIQUE_TRTLLM_GEN
    )
    raise ValueError(
        f"no TRTLLM-Gen implementation for quant="
        f"{normalize_quant(canonical_quant(quant_algo))} on "
        f"provider={'|'.join(_providers_to_try(provider))}; registered: {registered}"
    )


def _providers_to_try(provider: Optional[str]) -> tuple:
    """The providers a lookup walks, in the order it walks them."""
    return (provider,) if provider is not None else (PROVIDER_TRTLLM, PROVIDER_FLASHINFER)


def find_trtllm_gen_leaf(
    quant_algo: Optional[QuantAlgo], *, provider: Optional[str] = None
) -> Optional[type]:
    """The leaf implementing ``quant_algo``, or ``None`` if none publishes it.

    Goes through the registry rather than a table of its own, so a leaf that is
    renamed or unregistered disappears from here too.
    """
    # The same two steps resolution takes (``moe_resolution`` canonicalizes,
    # ``MoEImplId`` folds case), so a caller naming NVFP4_AWQ or MIXED_PRECISION
    # here reaches the leaf resolution would pick rather than a lookup miss.
    quant = normalize_quant(canonical_quant(quant_algo))
    for candidate in _providers_to_try(provider):
        cls = MOE_IMPL_REGISTRY.lookup(
            MoEImplId(candidate, TECHNIQUE_TRTLLM_GEN, KERNEL_FUSED_MOE, quant)
        )
        if cls is not None:
            return cls
    return None


__all__ = [
    # trtllm provider
    "TrtllmTrtllmGenNvfp4Impl",
    "TrtllmTrtllmGenFp8BlockScalesImpl",
    "TrtllmTrtllmGenW4a16Mxfp4Impl",
    "TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl",
    "TrtllmTrtllmGenW4a8Nvfp4Fp8Impl",
    "TrtllmTrtllmGenW4a8Mxfp4Fp8Impl",
    # flashinfer provider
    "FlashinferTrtllmGenNvfp4Impl",
    "FlashinferTrtllmGenFp8BlockScalesImpl",
    "FlashinferTrtllmGenW4a16Mxfp4Impl",
    "FlashinferTrtllmGenW4a8Mxfp4Mxfp8Impl",
    "FlashinferTrtllmGenBf16Impl",
    # shared layers
    "TrtllmGenFusedMoEBase",
    "TrtllmProviderTraits",
    "FlashinferProviderTraits",
    "TRTLLMGenFp4BlockScaleBase",
    "TRTLLMGenNvfp4Base",
    "TRTLLMGenW4a16Mxfp4Base",
    "TRTLLMGenW4a8Mxfp4Mxfp8Base",
    "TRTLLMGenFp8BlockScalesBase",
    # identity
    "PROVIDER_TRTLLM",
    "PROVIDER_FLASHINFER",
    "TECHNIQUE_TRTLLM_GEN",
    "KERNEL_FUSED_MOE",
    "TRTLLM_GEN_CAPABILITIES",
    "TRTLLM_GEN_INPUT_REQUIREMENT",
    "trtllm_gen_leaf",
    "find_trtllm_gen_leaf",
    "trtllm_gen_descriptor",
    "check_trtllm_gen_leaf",
    "check_trtllm_gen_capabilities",
    "check_quant_matches_identity",
    "check_flashinfer_provider",
    "check_mxfp4_flashinfer_shape",
    "nvfp4_needs_padded_method",
]
