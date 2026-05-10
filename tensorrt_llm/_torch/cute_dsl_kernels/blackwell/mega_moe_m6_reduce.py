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

"""CuTe DSL helpers for MegaMoE M6 combine-buffer reduction."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

_THREADS_PER_BLOCK = 256
_m6_reduce_compile_cache: dict[tuple[object, ...], object] = {}


@cute.kernel
def _m6_reduce_combine_buffer_bf16_out_kernel(
    combine_buffer: cute.Tensor,
    output: cute.Tensor,
    top_k: cutlass.Constexpr,
    local_tokens: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
    token_major: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
) -> None:
    griddepcontrol_wait()

    block_idx, _, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    linear_idx = block_idx * threads_per_block + thread_idx
    stride = threads_per_block * num_blocks
    total_elements = local_tokens * hidden_size

    while linear_idx < total_elements:
        token_idx = linear_idx // hidden_size
        hidden_idx = linear_idx - token_idx * hidden_size
        accum = cutlass.Float32(0.0)
        for topk_idx in range(top_k):
            if cutlass.const_expr(token_major):
                accum = accum + combine_buffer[token_idx, topk_idx, hidden_idx].to(cutlass.Float32)
            else:
                accum = accum + combine_buffer[topk_idx, token_idx, hidden_idx].to(cutlass.Float32)
        output[token_idx, hidden_idx] = accum.to(cutlass.BFloat16)
        linear_idx = linear_idx + stride

    griddepcontrol_launch_dependents()


@cute.jit
def launch_m6_reduce_combine_buffer_bf16_out(
    combine_buffer: cute.Tensor,
    output: cute.Tensor,
    top_k: cutlass.Constexpr,
    local_tokens: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
    token_major: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    _m6_reduce_combine_buffer_bf16_out_kernel(
        combine_buffer,
        output,
        top_k,
        local_tokens,
        hidden_size,
        token_major,
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


def _to_dynamic_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach()).mark_layout_dynamic()


def _validate_reduce_inputs(
    combine_buffer: torch.Tensor,
    output: torch.Tensor,
    local_tokens: int,
    token_major: bool,
) -> tuple[int, int, int]:
    if not combine_buffer.is_cuda:
        raise ValueError("combine_buffer must be a CUDA tensor")
    if not output.is_cuda:
        raise ValueError("output must be a CUDA tensor")
    if combine_buffer.dtype != torch.bfloat16:
        raise ValueError("combine_buffer must use bfloat16 dtype")
    if output.dtype != torch.bfloat16:
        raise ValueError("output must use bfloat16 dtype")
    if combine_buffer.dim() != 3:
        raise ValueError("combine_buffer must be a 3D tensor")
    if output.dim() != 2:
        raise ValueError("output must be a 2D tensor")
    if local_tokens < 0:
        raise ValueError("local_tokens must be non-negative")

    if token_major:
        max_num_tokens_per_rank = int(combine_buffer.size(0))
        top_k = int(combine_buffer.size(1))
        hidden_size = int(combine_buffer.size(2))
    else:
        top_k = int(combine_buffer.size(0))
        max_num_tokens_per_rank = int(combine_buffer.size(1))
        hidden_size = int(combine_buffer.size(2))

    if top_k <= 0 or max_num_tokens_per_rank <= 0 or hidden_size <= 0:
        raise ValueError("combine_buffer dimensions must be positive")
    if local_tokens > max_num_tokens_per_rank:
        raise ValueError(
            f"local_tokens ({local_tokens}) exceeds combine buffer capacity "
            f"({max_num_tokens_per_rank})"
        )
    if output.size(0) < local_tokens or output.size(1) != hidden_size:
        raise ValueError("output shape must cover local_tokens and match hidden_size")

    return top_k, max_num_tokens_per_rank, hidden_size


def reduce_m6_combine_buffer_bf16_out(
    combine_buffer: torch.Tensor,
    output: torch.Tensor,
    local_tokens: int,
    *,
    token_major: bool = False,
) -> None:
    """Reduce a rank-local MegaMoE direct combine buffer into BF16 output.

    ``combine_buffer`` is either ``[top_k, max_tokens, hidden]`` or, when
    ``token_major=True``, ``[max_tokens, top_k, hidden]``. The kernel writes
    ``sum(top_k)`` for the first ``local_tokens`` rows into ``output``.
    """
    local_tokens = int(local_tokens)
    token_major = bool(token_major)
    top_k, _, hidden_size = _validate_reduce_inputs(
        combine_buffer, output, local_tokens, token_major
    )
    if local_tokens == 0:
        return

    device_index = combine_buffer.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability[0] < 10:
        raise ValueError("CUTEDSL MegaMoE M6 reduce requires Blackwell or newer")

    total_elements = local_tokens * hidden_size
    element_blocks = (total_elements + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    props = torch.cuda.get_device_properties(device_index)
    num_blocks = max(1, min(int(props.multi_processor_count) * 8, element_blocks))

    combine_buffer_cute = _to_dynamic_cute_tensor(combine_buffer)
    output_cute = _to_dynamic_cute_tensor(output)
    stream = cuda.CUstream(torch.cuda.current_stream(combine_buffer.device).cuda_stream)

    compile_key = (
        capability,
        _tensor_compile_signature(combine_buffer),
        _tensor_compile_signature(output),
        top_k,
        local_tokens,
        hidden_size,
        token_major,
        _THREADS_PER_BLOCK,
        num_blocks,
    )
    if compile_key not in _m6_reduce_compile_cache:
        _m6_reduce_compile_cache[compile_key] = cute.compile(
            launch_m6_reduce_combine_buffer_bf16_out,
            combine_buffer_cute,
            output_cute,
            top_k,
            local_tokens,
            hidden_size,
            token_major,
            _THREADS_PER_BLOCK,
            num_blocks,
            stream,
        )

    _m6_reduce_compile_cache[compile_key](
        combine_buffer_cute,
        output_cute,
        stream,
    )


__all__ = ["reduce_m6_combine_buffer_bf16_out"]
