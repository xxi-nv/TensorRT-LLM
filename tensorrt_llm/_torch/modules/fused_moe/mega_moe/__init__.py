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
"""MegaMoE — fused MoE kernels exposed as first-class MoE backends.

Two backends live here, one per kernel family:

- :class:`MegaMoEDeepGemm` wraps DeepGEMM's W4A8_MXFP4_MXFP8
  ``fp8_fp4_mega_moe`` (FUSED_COMM; the DG kernel owns its own NVLink
  SymmBuffer dispatch + combine).
  :class:`W4A8MXFP4MXFP8MegaMoEDeepGemmMethod` owns its weight tensors,
  scale conversion, and DeepGEMM weight transform.

- :class:`MegaMoECuteDSL` wraps the CuTeDSL NVFP4 SM100/SM103
  ``cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce`` op
  (FUSED_COMM; the monolithic kernel owns dispatch + FC1 + SwiGLU +
  FC2 + reduce across the EP world via NVLink symmetric memory).
  :class:`NVFP4MegaMoECuteDSLMethod` owns the gate/up-interleaved NVFP4
  weight layout that the fused kernel reads.
"""

from ..quantization import NVFP4MegaMoECuteDSLMethod, W4A8MXFP4MXFP8MegaMoEDeepGemmMethod
from .mega_moe_cute_dsl import MegaMoECuteDSL
from .mega_moe_deepgemm import MegaMoEDeepGemm

__all__ = [
    "MegaMoECuteDSL",
    "MegaMoEDeepGemm",
    "NVFP4MegaMoECuteDSLMethod",
    "W4A8MXFP4MXFP8MegaMoEDeepGemmMethod",
]
