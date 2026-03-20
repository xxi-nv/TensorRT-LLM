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
"""
NVSHMEM buffer management utilities for FlashMoE kernel-level communication.

Provides symmetric heap allocation and lifecycle management for NVSHMEM
buffers used by the FlashMoE fused communication+computation kernel.

Key NVSHMEM operations used:
- nvshmem_malloc / nvshmem_free: Symmetric heap allocation
- nvshmemx_putmem_nbi_on_stream: Non-blocking put (dispatch tokens)
- nvshmemx_getmem_nbi_on_stream: Non-blocking get (combine results)
- nvshmem_quiet: Ensure all operations complete
- nvshmem_barrier_all: Global synchronization

The symmetric buffers are allocated lazily on first use and freed when
the owning FlashMoECuteDsl module is destroyed. This avoids the GC-ordering
hang seen with DeepEP (same issue as configurable_moe.py:381).
"""

from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger


class NvshmemBufferManager:
    """Manages NVSHMEM symmetric heap buffers for FlashMoE dispatch/combine.

    Allocates dispatch and combine buffers on the NVSHMEM symmetric heap
    for kernel-level communication. Buffers are sized for the maximum
    number of tokens that can be dispatched to/from each GPU.

    The manager handles:
    - Lazy initialization (allocate on first forward pass)
    - Buffer reuse across forward passes
    - Proper cleanup to avoid GC-ordering hangs

    Usage:
        manager = NvshmemBufferManager(ep_size=4, hidden_size=7168,
                                       max_num_tokens=8192)
        dispatch_buf, combine_buf = manager.get_buffers(device)
        ...
        manager.free()  # explicit cleanup before module destruction
    """

    def __init__(
        self,
        ep_size: int,
        hidden_size: int,
        intermediate_size: int,
        max_num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.ep_size = ep_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = max_num_tokens
        self.dtype = dtype

        self._dispatch_buffer: Optional[torch.Tensor] = None
        self._combine_buffer: Optional[torch.Tensor] = None
        self._initialized = False

    def _allocate(self, device: torch.device) -> None:
        """Allocate symmetric heap buffers.

        dispatch_buffer: holds input tokens being sent to remote experts
            Shape: [max_tokens_per_rank, hidden_size]
        combine_buffer: holds output tokens being sent back
            Shape: [max_tokens_per_rank, hidden_size]

        For now, use regular CUDA memory as a placeholder.
        True NVSHMEM allocation requires nvshmem_malloc which needs
        NVSHMEM initialization at process startup.
        """
        max_tokens_per_rank = self.max_num_tokens
        self._dispatch_buffer = torch.zeros(
            max_tokens_per_rank,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._combine_buffer = torch.zeros(
            max_tokens_per_rank,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._initialized = True
        logger.info(
            f"NvshmemBufferManager: allocated buffers "
            f"({max_tokens_per_rank} x {self.hidden_size}) on {device}"
        )

    def get_buffers(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dispatch and combine buffers, allocating on first use."""
        if not self._initialized:
            self._allocate(device)
        return self._dispatch_buffer, self._combine_buffer

    def free(self) -> None:
        """Explicitly free buffers. Must be called before module destruction
        to avoid GC-ordering hang (same issue as DeepEP)."""
        if self._initialized:
            self._dispatch_buffer = None
            self._combine_buffer = None
            self._initialized = False
            logger.info("NvshmemBufferManager: freed buffers")

    @property
    def initialized(self) -> bool:
        return self._initialized
