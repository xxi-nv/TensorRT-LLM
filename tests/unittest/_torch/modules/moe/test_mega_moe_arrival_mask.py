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
"""Proof probe for MegaMoE FC1-to-FC2 arrival-mask synchronization."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    fence_acq_rel_gpu,
    ld_acquire_gpu_u64,
    red_or_release_gpu_u64,
)
from tensorrt_llm._utils import get_sm_version

_PAYLOAD0 = 0x1234567
_PAYLOAD1 = 0x7654321
_EXPECTED_MASK = (1 << 40) | 0x5


def _skip_if_not_blackwell() -> None:
    sm = get_sm_version()
    if sm not in {100, 103}:
        pytest.skip(f"MegaMoE requires SM100/103, got SM{sm}")


@cute.kernel
def _arrival_mask_probe_kernel(
    mask: cute.Tensor, payload: cute.Tensor, observed: cute.Tensor
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    expected_mask = cutlass.Uint64(_EXPECTED_MASK)

    if bidx == 0:
        if tidx == 0:
            payload[0] = cutlass.Int32(_PAYLOAD0)
            payload[1] = cutlass.Int32(_PAYLOAD1)
            fence_acq_rel_gpu()
            red_or_release_gpu_u64(mask.iterator, expected_mask)
    elif bidx == 1:
        if tidx == 0:
            cached = cutlass.Uint64(0)
            while (cached & expected_mask) != expected_mask:
                cached = ld_acquire_gpu_u64(mask.iterator)
            observed[0] = payload[0]
            observed[1] = payload[1]
            observed[2] = cutlass.Int64(cached)


@cute.jit
def _launch_arrival_mask_probe(
    mask: cute.Tensor,
    payload: cute.Tensor,
    observed: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    _arrival_mask_probe_kernel(mask, payload, observed).launch(
        grid=(2, 1, 1),
        block=[32, 1, 1],
        stream=stream,
        use_pdl=TRTLLM_ENABLE_PDL,
    )


@pytest.mark.gpu
def test_mega_moe_l2_arrival_mask_release_acquire_probe() -> None:
    """Prove the b64 release/acquire arrival-mask primitive used by MegaMoE.

    The producer CTA writes ordinary global payload data, fences, and publishes
    a high 64-bit arrival bit via ``red.release.gpu.global.or.b64``. The
    consumer CTA spins with ``ld.acquire.gpu.global.u64`` until the full mask is
    visible, then reads the payload. This isolates the synchronization contract
    before the main MegaMoE kernel relies on it for FC1-to-FC2 pool hand-off.
    """
    _skip_if_not_blackwell()

    mask_torch = torch.zeros(1, dtype=torch.int64, device="cuda")
    payload_torch = torch.zeros(2, dtype=torch.int32, device="cuda")
    observed_torch = torch.zeros(3, dtype=torch.int64, device="cuda")

    assert mask_torch.data_ptr() % 8 == 0

    mask = from_dlpack(mask_torch).mark_layout_dynamic()
    payload = from_dlpack(payload_torch).mark_layout_dynamic()
    observed = from_dlpack(observed_torch).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_launch_arrival_mask_probe, mask, payload, observed, stream)

    for _ in range(200):
        mask_torch.zero_()
        payload_torch.zero_()
        observed_torch.fill_(-1)
        compiled(mask, payload, observed, stream)
        torch.cuda.synchronize()

        assert int(mask_torch.cpu().item()) == _EXPECTED_MASK
        assert observed_torch.cpu().tolist() == [_PAYLOAD0, _PAYLOAD1, _EXPECTED_MASK]
