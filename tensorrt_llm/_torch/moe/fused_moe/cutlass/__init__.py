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
"""Everything Cutlass: the registered leaves and the layers under them.

Self-contained, so the ``..fused_moe_cutlass`` module path above it holds
nothing but the ``CutlassFusedMoE`` name. Bottom to top:

* :mod:`.identity` -- the provider / technique / kernel strings, the two
  published contracts, and the descriptor factories.
* :mod:`.base` -- :class:`.CutlassFusedMoEBase`, the abstract root every leaf
  shares. Reads the SM set, dtype set and activation limits as leaf-declared
  attributes, never as a branch on ``quant_config``.
* :mod:`.eligibility` -- the checks each ``can_implement`` composes, replacing
  the ``_QUANT_SUPPORT_TABLE`` interpreter.

One module per leaf, named for its quantization format. Importing this package
is what registers them.

Ten leaves run the CUTLASS grouped GEMM. Two do not, and say so in their
identity: ``fp8_block_scales`` reaches DeepGEMM on SM90
(:mod:`.blockscale_hopper`) and an internal Triton kernel on SM120
(:mod:`.blockscale_triton`). That format used to be one name whose kernel was
picked by an SM branch inside the forward path.
"""

from typing import Optional, Tuple, Type

from tensorrt_llm.models.modeling_utils import QuantAlgo

from ..impl_contract import canonical_quant, normalize_quant
from ..impl_identity import MOE_IMPL_REGISTRY, MoEImplId
from .base import CutlassFusedMoEBase
from .blockscale_hopper import DeepgemmCudaHopperFp8BlockScalesImpl
from .blockscale_triton import TrtllmTritonFp8BlockScalesImpl
from .eligibility import (
    BF16_ONLY,
    HP_DTYPES,
    HP_DTYPES_WITH_FP8,
    HP_DTYPES_WITH_FP32,
    SmSupport,
    check_cutlass_leaf,
)
from .fp8 import TrtllmCutlassFp8Impl
from .identity import (
    CUTLASS_CAPABILITIES,
    CUTLASS_INPUT_REQUIREMENT,
    CUTLASS_LORA_CAPABILITIES,
    KERNEL_GROUPED_GEMM,
    PROVIDER_TRTLLM,
    TECHNIQUE_CUTLASS,
    cutlass_descriptor,
)
from .lora import CutlassMoELoraMixin, raise_moe_lora_multichunk_unsupported
from .mxfp8 import TrtllmCutlassMxfp8Impl
from .nvfp4 import TrtllmCutlassNvfp4Impl
from .unquantized import TrtllmCutlassUnquantizedImpl
from .w4a8_awq import TrtllmCutlassW4a8AwqImpl
from .w4a8_mxfp4_fp8 import TrtllmCutlassW4a8Mxfp4Fp8Impl
from .w4a8_mxfp4_mxfp8 import TrtllmCutlassW4a8Mxfp4Mxfp8Impl
from .w4a16_mxfp4 import TrtllmCutlassW4a16Mxfp4Impl
from .w4a16_nvfp4 import TrtllmCutlassW4a16Nvfp4Impl
from .w8a16 import TrtllmCutlassW8a16Impl

#: Every leaf in this family, in the order ``IMPL_PRIORITY`` ranks them.
#: Ordering within the family does not affect correctness -- the ``quant``
#: segments are disjoint, so at most one leaf can admit a given problem -- but
#: it is the order a resolution report lists them in, and it is what
#: ``moe_resolution`` uses as the family's fallback set.
CUTLASS_LEAVES: Tuple[Type[CutlassFusedMoEBase], ...] = (
    # The two that are not CUTLASS go first: their SM gates are the narrowest,
    # so nothing else in the family can shadow them.
    DeepgemmCudaHopperFp8BlockScalesImpl,
    TrtllmTritonFp8BlockScalesImpl,
    # The CUTLASS grouped-GEMM leaves.
    TrtllmCutlassNvfp4Impl,
    TrtllmCutlassW4a16Nvfp4Impl,
    TrtllmCutlassMxfp8Impl,
    TrtllmCutlassW4a8Mxfp4Fp8Impl,
    TrtllmCutlassW4a8Mxfp4Mxfp8Impl,
    TrtllmCutlassW4a16Mxfp4Impl,
    TrtllmCutlassW4a8AwqImpl,
    TrtllmCutlassW8a16Impl,
    TrtllmCutlassFp8Impl,
    # Unquantized last: the widest SM and dtype coverage, so it is the one a
    # heuristic walk should reach only after every narrower leaf declined.
    TrtllmCutlassUnquantizedImpl,
)


def find_cutlass_grouped_gemm_leaf(quant_algo: Optional[QuantAlgo]) -> Optional[type]:
    """The CUTLASS grouped-GEMM leaf implementing ``quant_algo``, or ``None``.

    Goes through the registry rather than a table of its own, so a leaf that is
    renamed or unregistered disappears from here too. Does NOT find the two
    ``fp8_block_scales`` leaves -- they are not grouped-GEMM identities, and a
    caller naming that format has to say which kernel it means.
    """
    quant = normalize_quant(canonical_quant(quant_algo))
    return MOE_IMPL_REGISTRY.lookup(
        MoEImplId(PROVIDER_TRTLLM, TECHNIQUE_CUTLASS, KERNEL_GROUPED_GEMM, quant)
    )


__all__ = [
    # leaves
    "TrtllmCutlassUnquantizedImpl",
    "TrtllmCutlassFp8Impl",
    "TrtllmCutlassNvfp4Impl",
    "TrtllmCutlassW4a16Nvfp4Impl",
    "TrtllmCutlassW4a8AwqImpl",
    "TrtllmCutlassW8a16Impl",
    "TrtllmCutlassW4a16Mxfp4Impl",
    "TrtllmCutlassW4a8Mxfp4Fp8Impl",
    "TrtllmCutlassW4a8Mxfp4Mxfp8Impl",
    "TrtllmCutlassMxfp8Impl",
    "DeepgemmCudaHopperFp8BlockScalesImpl",
    "TrtllmTritonFp8BlockScalesImpl",
    # shared layers
    "CutlassFusedMoEBase",
    "CUTLASS_LEAVES",
    "CutlassMoELoraMixin",
    "raise_moe_lora_multichunk_unsupported",
    # identity
    "PROVIDER_TRTLLM",
    "TECHNIQUE_CUTLASS",
    "KERNEL_GROUPED_GEMM",
    "CUTLASS_CAPABILITIES",
    "CUTLASS_LORA_CAPABILITIES",
    "CUTLASS_INPUT_REQUIREMENT",
    "cutlass_descriptor",
    "find_cutlass_grouped_gemm_leaf",
    # eligibility
    "SmSupport",
    "check_cutlass_leaf",
    "BF16_ONLY",
    "HP_DTYPES",
    "HP_DTYPES_WITH_FP32",
    "HP_DTYPES_WITH_FP8",
]
