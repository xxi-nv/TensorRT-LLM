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

"""CuTe DSL output zero helpers for MoE finalize paths."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    make_ptr,
)

_THREADS_PER_BLOCK = 256
_zero_no_local_rows_compile_cache: dict[tuple[object, ...], object] = {}


@cute.kernel
def _zero_no_local_rows_kernel(
    output: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    num_tokens: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    token_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    if token_idx < num_tokens:
        has_local_route = cutlass.Boolean(False)
        for topk_idx in range(top_k):
            if expanded_idx_to_permuted_idx[token_idx, topk_idx] >= cutlass.Int32(0):
                has_local_route = cutlass.Boolean(True)
        if not has_local_route:
            hidden_idx = thread_idx
            while hidden_idx < hidden_size:
                output[token_idx, hidden_idx] = cutlass.BFloat16(0.0)
                hidden_idx = hidden_idx + threads_per_block

    griddepcontrol_launch_dependents()


@cute.jit
def launch_zero_no_local_rows(
    output_ptr: cute.Pointer,
    expanded_idx_to_permuted_idx_ptr: cute.Pointer,
    output_stride_0: cutlass.Constexpr,
    output_stride_1: cutlass.Constexpr,
    route_stride_0: cutlass.Constexpr,
    route_stride_1: cutlass.Constexpr,
    num_tokens: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    output = cute.make_tensor(
        output_ptr,
        cute.make_layout(
            (num_tokens, hidden_size),
            stride=(output_stride_0, output_stride_1),
        ),
    )
    expanded_idx_to_permuted_idx = cute.make_tensor(
        expanded_idx_to_permuted_idx_ptr,
        cute.make_layout(
            (num_tokens, top_k),
            stride=(route_stride_0, route_stride_1),
        ),
    )
    _zero_no_local_rows_kernel(
        output,
        expanded_idx_to_permuted_idx,
        num_tokens,
        hidden_size,
        top_k,
        threads_per_block,
    ).launch(
        grid=(num_tokens, 1, 1),
        block=[threads_per_block, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


def _tensor_compile_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device.type,
        tensor.device.index,
    )


def _make_output_ptr(tensor: torch.Tensor) -> cute.Pointer:
    if tensor.dtype == torch.bfloat16:
        return make_ptr(
            cutlass.BFloat16, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=2
        )
    raise ValueError(f"unsupported output dtype for no-local zero: {tensor.dtype}")


def _make_int32_ptr(tensor: torch.Tensor) -> cute.Pointer:
    if tensor.dtype == torch.int32:
        return make_ptr(cutlass.Int32, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    raise ValueError(f"unsupported route dtype for no-local zero: {tensor.dtype}")


def zero_no_local_moe_output_rows(
    output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
) -> None:
    """Zero output rows with no local route in ``expanded_idx_to_permuted_idx``."""
    if not output.is_cuda or not expanded_idx_to_permuted_idx.is_cuda:
        raise ValueError("output and expanded_idx_to_permuted_idx must be CUDA tensors")
    if output.device != expanded_idx_to_permuted_idx.device:
        raise ValueError("output and expanded_idx_to_permuted_idx must be on the same device")
    if output.dtype != torch.bfloat16:
        raise ValueError("output must use bfloat16 dtype")
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise ValueError("expanded_idx_to_permuted_idx must use int32 dtype")
    if output.dim() != 2 or expanded_idx_to_permuted_idx.dim() != 2:
        raise ValueError("output and expanded_idx_to_permuted_idx must be 2D")
    if int(output.size(0)) != int(expanded_idx_to_permuted_idx.size(0)):
        raise ValueError("route rows must match output rows")
    if output.numel() == 0:
        return

    device = output.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability[0] < 10:
        raise ValueError("CUTEDSL no-local output zero requires Blackwell or newer")

    num_tokens = int(output.size(0))
    hidden_size = int(output.size(1))
    top_k = int(expanded_idx_to_permuted_idx.size(1))
    if top_k <= 0 or hidden_size <= 0:
        return

    output_ptr = _make_output_ptr(output)
    route_ptr = _make_int32_ptr(expanded_idx_to_permuted_idx)
    output_stride_0 = int(output.stride(0))
    output_stride_1 = int(output.stride(1))
    route_stride_0 = int(expanded_idx_to_permuted_idx.stride(0))
    route_stride_1 = int(expanded_idx_to_permuted_idx.stride(1))
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    compile_key = (
        capability,
        _tensor_compile_signature(output),
        _tensor_compile_signature(expanded_idx_to_permuted_idx),
        output_stride_0,
        output_stride_1,
        route_stride_0,
        route_stride_1,
        num_tokens,
        hidden_size,
        top_k,
        _THREADS_PER_BLOCK,
    )
    if compile_key not in _zero_no_local_rows_compile_cache:
        _zero_no_local_rows_compile_cache[compile_key] = cute.compile(
            launch_zero_no_local_rows,
            output_ptr,
            route_ptr,
            output_stride_0,
            output_stride_1,
            route_stride_0,
            route_stride_1,
            num_tokens,
            hidden_size,
            top_k,
            _THREADS_PER_BLOCK,
            stream,
        )

    _zero_no_local_rows_compile_cache[compile_key](output_ptr, route_ptr, stream)


__all__ = ["zero_no_local_moe_output_rows"]
