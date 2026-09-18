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
"""Cutlass's values of the identity types ``..impl_identity`` defines.

Every leaf imports this module, so nothing that varies by quantization format
belongs here.

Three ``(provider, technique, kernel_name)`` triples live here, not one,
because this family's leaves do not all run a CUTLASS kernel:

* ``trtllm.cutlass.grouped_gemm`` -- the CUTLASS grouped GEMM reached through
  ``torch.ops.trtllm.fused_moe``. Ten quantization formats.
* ``deepgemm.cuda.hopper_grouped_gemm`` -- FP8 block scales on SM90. The MoE
  orchestration is still ``CutlassMoeFCRunner``, but the GEMM itself is the
  vendored DeepSeek DeepGEMM, JIT-compiled
  (``cpp/.../fp8_blockscale_gemm/`` -> ``deep_gemm::runGemm``). Shares
  ``provider`` / ``technique`` with the SM100/103 leaf in
  ``..fused_moe_deepgemm``, which runs the same library through its Python
  package; only ``kernel_name`` separates the two.
* ``trtllm.triton.blockscale_gemm`` -- FP8 block scales on SM120, an internal
  Triton kernel that never enters C++ at all.
"""

import torch

from ..impl_contract import MoEInputRequirement, MoEStaticCapability
from ..impl_identity import MoEImplDescriptor, MoEImplId
from ..interface import MoESchedulerKind

PROVIDER_TRTLLM = "trtllm"
PROVIDER_DEEPGEMM = "deepgemm"

TECHNIQUE_CUTLASS = "cutlass"
TECHNIQUE_CUDA = "cuda"
TECHNIQUE_TRITON = "triton"

KERNEL_GROUPED_GEMM = "grouped_gemm"
KERNEL_HOPPER_GROUPED_GEMM = "hopper_grouped_gemm"
KERNEL_BLOCKSCALE_GEMM = "blockscale_gemm"

# Published family-wide, because every leaf declares the same three. LoRA is
# the exception and is handled by CUTLASS_LORA_CAPABILITIES below.
CUTLASS_CAPABILITIES = MoEStaticCapability(
    supports_expert_bias=True,
    supports_eplb=True,
    supports_apply_router_weight_on_input=True,
)

# Routed-expert MoE LoRA is fused into ``torch.ops.trtllm.fused_moe`` for
# unquantized and per-tensor-FP8 weights only -- every other format is rejected
# by the C++ op. Declaring it per leaf rather than family-wide is what lets
# ``MoEScheduler`` keep its ``capabilities.supports_moe_lora`` short-circuits
# (``..moe_scheduler`` lines 154 and 836) working: a leaf that cannot fuse LoRA
# says so, and the scheduler never reaches for the helpers it does not carry.
CUTLASS_LORA_CAPABILITIES = MoEStaticCapability(
    supports_moe_lora=True,
    supports_expert_bias=True,
    supports_eplb=True,
    supports_apply_router_weight_on_input=True,
)

CUTLASS_INPUT_REQUIREMENT = MoEInputRequirement(routing_scales_dtype=torch.float32)


def cutlass_descriptor(
    quant: str,
    doc: str,
    *,
    capabilities: MoEStaticCapability = CUTLASS_CAPABILITIES,
) -> MoEImplDescriptor:
    """Build one CUTLASS grouped-GEMM leaf's descriptor."""
    return MoEImplDescriptor(
        identity=MoEImplId(PROVIDER_TRTLLM, TECHNIQUE_CUTLASS, KERNEL_GROUPED_GEMM, quant),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=capabilities,
        input_requirement=CUTLASS_INPUT_REQUIREMENT,
        doc=doc,
    )


def blockscale_descriptor(
    provider: str,
    technique: str,
    kernel_name: str,
    doc: str,
) -> MoEImplDescriptor:
    """Build one FP8-block-scale leaf's descriptor.

    Takes all three leading segments because the two leaves share none of them:
    the SM90 one is DeepGEMM reached from C++, the SM120 one is an internal
    Triton kernel.
    """
    return MoEImplDescriptor(
        identity=MoEImplId(provider, technique, kernel_name, "fp8_block_scales"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=CUTLASS_CAPABILITIES,
        input_requirement=CUTLASS_INPUT_REQUIREMENT,
        doc=doc,
    )
