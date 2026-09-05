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
"""Provider mixins shared by the TRTLLM-Gen leaves.

Three constants that follow from the provider alone. As mixins rather than
per-leaf attributes because "use_flashinfer == (provider is flashinfer)" is an
invariant, and eleven copies of it is eleven chances to break it. They carry
no methods and do not subclass ``MoEImplBase``: a leaf is
``(provider mixin, per-quant class)``, and only the second half is an impl.
"""

from ..fused_moe_trtllm_gen import PROVIDER_FLASHINFER, PROVIDER_TRTLLM


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
