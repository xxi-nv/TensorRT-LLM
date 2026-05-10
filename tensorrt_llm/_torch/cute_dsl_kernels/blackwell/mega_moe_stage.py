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

"""CuTe DSL helpers for MegaMoE dispatch staging."""

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
_BYTES_PER_WORD = 8
_stage_dispatch_compile_cache: dict[tuple[object, ...], object] = {}


@cute.kernel
def _stage_dispatch_inputs_u64_kernel(
    input_words: cute.Tensor,
    input_sf_words: cute.Tensor,
    topk_idx_words: cute.Tensor,
    topk_scales_words: cute.Tensor,
    input_buffer_words: cute.Tensor,
    input_sf_buffer_words: cute.Tensor,
    topk_idx_buffer_words: cute.Tensor,
    topk_scales_buffer_words: cute.Tensor,
    input_word_count: cutlass.Constexpr,
    input_sf_word_count: cutlass.Constexpr,
    topk_idx_word_count: cutlass.Constexpr,
    topk_scales_word_count: cutlass.Constexpr,
    total_word_count: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    idx = block_idx * threads_per_block + thread_idx
    stride = threads_per_block * num_blocks

    while idx < total_word_count:
        if idx < input_word_count:
            input_buffer_words[idx] = input_words[idx]
        if idx < input_sf_word_count:
            input_sf_buffer_words[idx] = input_sf_words[idx]
        if idx < topk_idx_word_count:
            topk_idx_buffer_words[idx] = topk_idx_words[idx]
        if idx < topk_scales_word_count:
            topk_scales_buffer_words[idx] = topk_scales_words[idx]
        idx = idx + stride

    griddepcontrol_launch_dependents()


@cute.jit
def launch_stage_dispatch_inputs_u64(
    input_ptr: cute.Pointer,
    input_sf_ptr: cute.Pointer,
    topk_idx_ptr: cute.Pointer,
    topk_scales_ptr: cute.Pointer,
    input_buffer_ptr: cute.Pointer,
    input_sf_buffer_ptr: cute.Pointer,
    topk_idx_buffer_ptr: cute.Pointer,
    topk_scales_buffer_ptr: cute.Pointer,
    input_word_count: cutlass.Constexpr,
    input_sf_word_count: cutlass.Constexpr,
    topk_idx_word_count: cutlass.Constexpr,
    topk_scales_word_count: cutlass.Constexpr,
    total_word_count: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    input_words = cute.make_tensor(
        input_ptr, cute.make_ordered_layout((input_word_count,), order=(0,))
    )
    input_sf_words = cute.make_tensor(
        input_sf_ptr, cute.make_ordered_layout((input_sf_word_count,), order=(0,))
    )
    topk_idx_words = cute.make_tensor(
        topk_idx_ptr, cute.make_ordered_layout((topk_idx_word_count,), order=(0,))
    )
    topk_scales_words = cute.make_tensor(
        topk_scales_ptr, cute.make_ordered_layout((topk_scales_word_count,), order=(0,))
    )
    input_buffer_words = cute.make_tensor(
        input_buffer_ptr, cute.make_ordered_layout((input_word_count,), order=(0,))
    )
    input_sf_buffer_words = cute.make_tensor(
        input_sf_buffer_ptr, cute.make_ordered_layout((input_sf_word_count,), order=(0,))
    )
    topk_idx_buffer_words = cute.make_tensor(
        topk_idx_buffer_ptr, cute.make_ordered_layout((topk_idx_word_count,), order=(0,))
    )
    topk_scales_buffer_words = cute.make_tensor(
        topk_scales_buffer_ptr,
        cute.make_ordered_layout((topk_scales_word_count,), order=(0,)),
    )

    _stage_dispatch_inputs_u64_kernel(
        input_words,
        input_sf_words,
        topk_idx_words,
        topk_scales_words,
        input_buffer_words,
        input_sf_buffer_words,
        topk_idx_buffer_words,
        topk_scales_buffer_words,
        input_word_count,
        input_sf_word_count,
        topk_idx_word_count,
        topk_scales_word_count,
        total_word_count,
        threads_per_block,
        num_blocks,
    ).launch(
        grid=(num_blocks, 1, 1),
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


def _byte_size(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _check_source_tensor(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on the staging device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_buffer(
    name: str, tensor: torch.Tensor, device: torch.device, required_bytes: int
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on the staging device")
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous uint8 tensor")
    if int(tensor.numel()) < required_bytes:
        raise ValueError(f"{name} is too small for {required_bytes} staged bytes")


def _make_u64_ptr(tensor: torch.Tensor) -> cute.Pointer:
    return make_ptr(
        cutlass.Uint64,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=_BYTES_PER_WORD,
    )


def stage_dispatch_inputs_u64(
    input: torch.Tensor,
    input_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    input_buffer: torch.Tensor,
    input_sf_buffer: torch.Tensor,
    topk_idx_buffer: torch.Tensor,
    topk_scales_buffer: torch.Tensor,
) -> None:
    """Stage MegaMoE dispatch inputs with an 8-byte CuTe DSL copy kernel."""
    device = input.device
    _check_source_tensor("input", input, device)
    _check_source_tensor("input_sf", input_sf, device)
    _check_source_tensor("topk_idx", topk_idx, device)
    _check_source_tensor("topk_scales", topk_scales, device)
    if topk_idx.dtype != torch.int64:
        raise ValueError("topk_idx must use int64 dtype")
    if topk_scales.dtype != torch.float32:
        raise ValueError("topk_scales must use float32 dtype")

    input_bytes = _byte_size(input)
    input_sf_bytes = _byte_size(input_sf)
    topk_idx_bytes = _byte_size(topk_idx)
    topk_scales_bytes = _byte_size(topk_scales)
    for name, byte_count in (
        ("input", input_bytes),
        ("input_sf", input_sf_bytes),
        ("topk_idx", topk_idx_bytes),
        ("topk_scales", topk_scales_bytes),
    ):
        if byte_count % _BYTES_PER_WORD != 0:
            raise ValueError(f"{name} byte size must be divisible by {_BYTES_PER_WORD}")

    _check_buffer("input_buffer", input_buffer, device, input_bytes)
    _check_buffer("input_sf_buffer", input_sf_buffer, device, input_sf_bytes)
    _check_buffer("topk_idx_buffer", topk_idx_buffer, device, topk_idx_bytes)
    _check_buffer("topk_scales_buffer", topk_scales_buffer, device, topk_scales_bytes)

    word_counts = (
        input_bytes // _BYTES_PER_WORD,
        input_sf_bytes // _BYTES_PER_WORD,
        topk_idx_bytes // _BYTES_PER_WORD,
        topk_scales_bytes // _BYTES_PER_WORD,
    )
    total_word_count = max(word_counts)
    if total_word_count <= 0:
        return

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability[0] < 10:
        raise ValueError("CUTEDSL MegaMoE staging requires Blackwell or newer")

    element_blocks = (total_word_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    props = torch.cuda.get_device_properties(device_index)
    num_blocks = max(1, min(int(props.multi_processor_count) * 8, element_blocks))

    input_ptr = _make_u64_ptr(input)
    input_sf_ptr = _make_u64_ptr(input_sf)
    topk_idx_ptr = _make_u64_ptr(topk_idx)
    topk_scales_ptr = _make_u64_ptr(topk_scales)
    input_buffer_ptr = _make_u64_ptr(input_buffer)
    input_sf_buffer_ptr = _make_u64_ptr(input_sf_buffer)
    topk_idx_buffer_ptr = _make_u64_ptr(topk_idx_buffer)
    topk_scales_buffer_ptr = _make_u64_ptr(topk_scales_buffer)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

    compile_key = (
        capability,
        _tensor_compile_signature(input),
        _tensor_compile_signature(input_sf),
        _tensor_compile_signature(topk_idx),
        _tensor_compile_signature(topk_scales),
        _tensor_compile_signature(input_buffer),
        _tensor_compile_signature(input_sf_buffer),
        _tensor_compile_signature(topk_idx_buffer),
        _tensor_compile_signature(topk_scales_buffer),
        word_counts,
        total_word_count,
        _THREADS_PER_BLOCK,
        num_blocks,
    )
    if compile_key not in _stage_dispatch_compile_cache:
        _stage_dispatch_compile_cache[compile_key] = cute.compile(
            launch_stage_dispatch_inputs_u64,
            input_ptr,
            input_sf_ptr,
            topk_idx_ptr,
            topk_scales_ptr,
            input_buffer_ptr,
            input_sf_buffer_ptr,
            topk_idx_buffer_ptr,
            topk_scales_buffer_ptr,
            word_counts[0],
            word_counts[1],
            word_counts[2],
            word_counts[3],
            total_word_count,
            _THREADS_PER_BLOCK,
            num_blocks,
            stream,
        )

    _stage_dispatch_compile_cache[compile_key](
        input_ptr,
        input_sf_ptr,
        topk_idx_ptr,
        topk_scales_ptr,
        input_buffer_ptr,
        input_sf_buffer_ptr,
        topk_idx_buffer_ptr,
        topk_scales_buffer_ptr,
        stream,
    )


__all__ = ["stage_dispatch_inputs_u64"]
