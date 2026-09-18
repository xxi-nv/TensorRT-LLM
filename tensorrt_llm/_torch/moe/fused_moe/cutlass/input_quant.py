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
"""Activation-quantization strategies the grouped-GEMM leaves choose between.

Four strategies cover all twelve leaves, which is why they live here rather
than being inlined per leaf: the pre-split ``quantize_input`` was one
``elif`` chain over ``self.has_*``, and most branches were shared by two
formats. A leaf now names the strategy it uses instead of re-deriving it from
``quant_config`` on every forward.
"""

from typing import Optional, Tuple, Union

import torch

from ....utils import Fp4QuantizedTensor


def quantize_noop(
    impl, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Hand the activations through untouched.

    Used by the unquantized leaf and by every format whose kernel quantizes
    activations itself (FP8 block scales, W4A8-AWQ, INT8 weight-only) or keeps
    them high precision (W4A16 variants). ``w4a16_mxfp4`` is here too: its
    padding is deliberately deferred to ``run_moe`` so that dispatch sends
    unpadded tensors and does not overallocate the NVLink workspace.
    """
    del impl, post_quant_comm
    return x, None


def quantize_static_e4m3(
    impl, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-tensor FP8: scale by the static FC1 input scale, cast to e4m3."""
    del post_quant_comm
    x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, impl.fc31_input_dequant)
    return x, None


def quantize_mxfp8(
    impl, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dynamic MXFP8 activation quantization.

    Shared by W4A8 MXFP4xMXFP8 and W8A8 MXFP8xMXFP8 -- only the weight side
    differs between them, the activation kernel is the same.
    """
    if post_quant_comm:
        x, x_sf = torch.ops.trtllm.mxfp8_quantize(
            x, False, alignment=impl.quant_method.weight_alignment
        )
        # Reshape to 2D for post-quant communication; x.shape[0] is padded.
        if x_sf is not None:
            x_sf = x_sf.view((x.shape[0], -1))
        return x, x_sf
    return torch.ops.trtllm.mxfp8_quantize(x, True, alignment=impl.quant_method.weight_alignment)


def quantize_nvfp4(
    impl, x: Union[torch.Tensor, Fp4QuantizedTensor], post_quant_comm: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """NVFP4: optional AWQ pre-scale, optional dynamic scale, then FP4 quant."""
    if getattr(impl, "fc31_act_scale", None) is not None:
        assert not isinstance(x, Fp4QuantizedTensor), (
            "Fp4QuantizedTensor is not expected for AWQ quantization."
        )
        x = x * impl.fc31_act_scale

    # Dynamic quantization: derive input_scale from this input and update alpha
    # in place, keeping the same tensor addresses so CUDA graphs stay valid.
    if impl.force_dynamic_quantization and hasattr(impl, "fc31_weight_scale_2"):
        FP8_MAX, E2M1_MAX = 448.0, 6.0
        amax_input = torch.amax(torch.abs(x)).float()
        dyn_input_scale = FP8_MAX * E2M1_MAX / amax_input
        # fc31_alpha[e] = weight_scale_2[e] / dyn_input_scale
        impl.fc31_alpha.data.copy_(impl.fc31_weight_scale_2.data / dyn_input_scale)
        impl.fc31_input_scale.data.copy_(dyn_input_scale)

    if post_quant_comm:
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, (
                "Fp4QuantizedTensor should not be swizzled before communication"
            )
            x, x_sf = x.fp4_tensor, x.scaling_factor
            x_row = x.shape[0]
        else:
            x_row = x.shape[0]
            x, x_sf = torch.ops.trtllm.fp4_quantize(
                x, impl.fc31_input_scale, impl.scaling_vector_size, False, False
            )
        # Reshape x_sf to 2D for post-quant communication.
        if x_sf is not None:
            x_sf = x_sf.view((x_row, -1))
        return x, x_sf

    if not isinstance(x, Fp4QuantizedTensor):
        return torch.ops.trtllm.fp4_quantize(
            x, impl.fc31_input_scale, impl.scaling_vector_size, False, True
        )
    return x, None
