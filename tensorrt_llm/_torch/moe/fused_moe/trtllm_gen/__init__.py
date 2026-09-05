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
"""The eleven registered TRTLLM-Gen leaves, one module each.

The abstract parent :class:`..fused_moe_trtllm_gen.TRTLLMGenFusedMoE` stays
outside this package: it owns construction, weight creation, and the shared
helpers, and it is what the rest of the codebase means by "a TRTLLM-Gen MoE".
Only the leaves that carry an identity live here, plus the two layers that
exist solely to serve them -- :mod:`.mixins` for the provider constants and
:mod:`.bases` for the per-quant execution bodies.

One module per leaf, named ``<provider>_<quant>``, so the file name spells out
the two identity segments that actually differ across the eleven. The technique
and kernel segments are ``trtllm_gen`` / ``fused_moe`` for all of them, which is
what this package name says once.

Importing this package is what registers the leaves. ``MOE_IMPL_REGISTRY``
lookups -- including ``trtllm_gen_leaf`` in the parent module -- therefore
depend on it having been imported, which ``moe_resolution`` guarantees for the
resolution path.

* ``trtllm`` provider (TRTLLM-14968): the native cubins, six formats.
* ``flashinfer`` provider (TRTLLM-14969): the same algorithm out of the
  FlashInfer wheel, five formats. NVFP4, FP8_BLOCK_SCALES, W4A16_MXFP4 and
  W4A8_MXFP4_MXFP8 exist on both providers; W4A8_NVFP4_FP8 and W4A8_MXFP4_FP8
  are native-only, and the unquantized bf16 path is FlashInfer-only.
"""

from .bases import (
    TRTLLMGenFp4BlockScaleBase,
    TRTLLMGenFp8BlockScalesBase,
    TRTLLMGenNvfp4Base,
    TRTLLMGenW4a8Mxfp4Mxfp8Base,
    TRTLLMGenW4a16Mxfp4Base,
)
from .flashinfer_bf16 import FlashinferTrtllmGenBf16Impl
from .flashinfer_fp8_block_scales import FlashinferTrtllmGenFp8BlockScalesImpl
from .flashinfer_nvfp4 import FlashinferTrtllmGenNvfp4Impl
from .flashinfer_w4a8_mxfp4_mxfp8 import FlashinferTrtllmGenW4a8Mxfp4Mxfp8Impl
from .flashinfer_w4a16_mxfp4 import FlashinferTrtllmGenW4a16Mxfp4Impl
from .mixins import FlashinferProviderMixin, TrtllmProviderMixin
from .trtllm_fp8_block_scales import TrtllmTrtllmGenFp8BlockScalesImpl
from .trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl
from .trtllm_w4a8_mxfp4_fp8 import TrtllmTrtllmGenW4a8Mxfp4Fp8Impl
from .trtllm_w4a8_mxfp4_mxfp8 import TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
from .trtllm_w4a8_nvfp4_fp8 import TrtllmTrtllmGenW4a8Nvfp4Fp8Impl
from .trtllm_w4a16_mxfp4 import TrtllmTrtllmGenW4a16Mxfp4Impl

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
    "TrtllmProviderMixin",
    "FlashinferProviderMixin",
    "TRTLLMGenFp4BlockScaleBase",
    "TRTLLMGenNvfp4Base",
    "TRTLLMGenW4a16Mxfp4Base",
    "TRTLLMGenW4a8Mxfp4Mxfp8Base",
    "TRTLLMGenFp8BlockScalesBase",
]
