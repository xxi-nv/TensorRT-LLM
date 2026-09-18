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
"""The ``CutlassFusedMoE`` family name, kept importable from its own path.

The implementation lives in the :mod:`.cutlass` subpackage. The split moved the
code out and left the name behind on purpose: the leaves are the addressable
implementations, and this name stands for all twelve at once, which no
descriptor can publish.

Following the ``TRTLLMGenFusedMoE`` precedent in :mod:`.fused_moe_trtllm_gen`,
it is an alias and not a parent, so there is no second class to keep in step.

Every gate against it has to be ``issubclass`` / ``isinstance`` and not an
equality check: this name is the family, and the leaves are what resolution
actually hands over, so ``type(x) is CutlassFusedMoE`` matches nothing.

``raise_moe_lora_multichunk_unsupported`` is forwarded because
:mod:`.moe_scheduler` imports it from this path; the function itself lives with
the LoRA plumbing in :mod:`.cutlass.base`.
"""

from .cutlass import (CUTLASS_LEAVES, CutlassFusedMoEBase,
                      find_cutlass_grouped_gemm_leaf,
                      raise_moe_lora_multichunk_unsupported)

#: The family base under its pre-split name. Not a subclass of it: an alias, so
#: that ``isinstance``/``issubclass`` against either name give the same answer
#: and there is no second class to keep in step.
CutlassFusedMoE = CutlassFusedMoEBase

__all__ = [
    "CutlassFusedMoE",
    "CUTLASS_LEAVES",
    "find_cutlass_grouped_gemm_leaf",
    "raise_moe_lora_multichunk_unsupported",
]
