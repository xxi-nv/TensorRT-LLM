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
"""The checks each Cutlass-family ``can_implement`` composes.

Replaces the ``_QUANT_SUPPORT_TABLE`` interpreter the single ``CutlassFusedMoE``
class used to run. The table's rows became per-leaf declarations -- ``sm_support``
and ``supported_dtypes`` -- so a row can no longer describe a format its class
does not implement, and reading one leaf tells the whole story for that format.
"""

from dataclasses import dataclass
from typing import FrozenSet, Optional

import torch

from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    nvfp4_fc1_row_alignment_rejection,
)
from ..interface import _reject

# Every kernel in this family is a Hopper-or-later CUTLASS build.
CUTLASS_MINIMUM_SM = 80


@dataclass(frozen=True)
class SmSupport:
    """SM versions a leaf's kernel is built and validated for.

    Exactly one of the two fields is set. ``minimum`` is an open range for the
    formats whose kernel is compiled for every later architecture; ``allowed``
    enumerates the ones whose cubins exist only for specific families, which is
    the common case on the quantized paths.
    """

    minimum: Optional[int] = None
    allowed: Optional[FrozenSet[int]] = None

    def __post_init__(self) -> None:
        if (self.minimum is None) == (self.allowed is None):
            raise ValueError("SmSupport takes exactly one of minimum / allowed")

    def accepts(self, sm: int) -> bool:
        if self.minimum is not None:
            return sm >= self.minimum
        return sm in self.allowed

    def describe(self) -> str:
        if self.minimum is not None:
            return f"SM >= {self.minimum}"
        if len(self.allowed) == 1:
            return f"SM{next(iter(self.allowed))} only"
        return "/".join(f"SM{v}" for v in sorted(self.allowed))


def check_sm(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
    """Reject an SM version this leaf's kernel was not built for."""
    del p
    sm = d.env.sm
    if sm < CUTLASS_MINIMUM_SM:
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} requires SM >= {CUTLASS_MINIMUM_SM}, got SM{sm}",
        )
    if not cls.sm_support.accepts(sm):
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} supports {cls.sm_support.describe()}, got SM{sm}",
        )
    return None


def check_quant_matches_identity(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
    """Reject a format this leaf is not the registered implementation of.

    Compares against the descriptor rather than a separate table, so the gate
    and the registry lookup that routes a problem here cannot disagree.
    ``MoEProblem.identity_quant`` folds the calibration aliases the same way
    the lookup does, so a layer declaring NVFP4_AWQ reaches the NVFP4 leaf and
    is admitted by it.
    """
    del d
    expected = cls.descriptor.identity.quant
    actual = p.identity_quant
    if actual != expected:
        return _reject(
            MoERejectReason.QUANT_UNSUPPORTED,
            f"{cls.__name__} implements quant={expected}, got {actual}",
        )
    return None


def check_dtype(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
    """Reject an activation dtype this leaf's kernel has no instantiation for."""
    del d
    if p.dtype_act not in cls.supported_dtypes:
        supported = ", ".join(str(dtype) for dtype in sorted(cls.supported_dtypes, key=str))
        return _reject(
            MoERejectReason.DTYPE_UNSUPPORTED,
            f"{cls.__name__} requires {supported}, got {p.dtype_act}",
        )
    return None


def check_gptoss_activation(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
    """Reject a gpt-oss / MiniMax SwiGLU shape this leaf's loader cannot take.

    ``supports_gptoss_style`` is about the weight method, not the kernel: the
    unquantized and MXFP4-family methods can load a 1-D gpt-oss expert bias,
    and MXFP8 inherits the generic ``FusedMoEMethodBase`` path that does the
    same. NVFP4 is a partial case -- it serves MiniMax-style SwigluBias, but
    its weight pad asserts 2-D, so a real expert bias is out.
    """
    del d
    if not p.swiglu_gptoss_style:
        return None
    if not cls.supports_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} cannot load a gpt-oss bias for quant={cls.descriptor.identity.quant}",
        )
    if cls.rejects_gptoss_expert_bias and p.bias is True:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} cannot load a 1-D gpt-oss expert bias "
            f"(weight-pad assert is 2-D); MiniMax-style SwigluBias without "
            f"bias is eligible",
        )
    return None


def check_moe_lora(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
    """Reject routed-expert MoE LoRA on a leaf whose format the op cannot fuse.

    The resolver already drops candidates whose ``capabilities`` decline LoRA
    before ``can_implement`` runs, so this only fires for a direct call -- unit
    tests and microbenchmarks reach a leaf without going through resolution.
    """
    del p
    if d.moe_lora_enabled and not cls.capabilities.supports_moe_lora:
        return _reject(
            MoERejectReason.LORA_UNSUPPORTED,
            f"{cls.__name__} does not fuse routed-expert MoE LoRA; only "
            f"unquantized fp16/bf16 and per-tensor FP8 (qdq) do",
        )
    return None


def check_nvfp4_shard_alignment(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
    """Reject an NVFP4 shard whose gated FC1 rows would be padded mid-operand.

    The loader bottom-pads the concatenated ``[w3; w1]`` FC1 buffer while the
    kernel splits gate/up at half the padded size, so an unaligned shard runs
    and returns garbage instead of failing. Shared with the other NVFP4
    backends through ``..impl_contract``.
    """
    del cls
    return nvfp4_fc1_row_alignment_rejection(p, d)


def check_cutlass_leaf(cls, p: MoEProblem, d: MoEDeployment, *extra_gates) -> MoEEligibility:
    """Run the family gates in order, then any the leaf adds, and admit.

    Order matters for the message an operator reads: identity first, so a
    format this leaf does not implement is never described in terms of an SM or
    dtype it also would not satisfy.
    """
    for gate in (
        check_quant_matches_identity,
        check_sm,
        check_dtype,
        check_gptoss_activation,
        check_moe_lora,
        *extra_gates,
    ):
        verdict = gate(cls, p, d)
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()


# dtype sets the table used, named so a leaf reads as a declaration.
HP_DTYPES = frozenset({torch.float16, torch.bfloat16})
HP_DTYPES_WITH_FP32 = frozenset({torch.float16, torch.bfloat16, torch.float32})
HP_DTYPES_WITH_FP8 = frozenset({torch.float16, torch.bfloat16, torch.float8_e4m3fn})
BF16_ONLY = frozenset({torch.bfloat16})
