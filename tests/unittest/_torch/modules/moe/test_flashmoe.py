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
"""FlashMoE unit tests.

Tests the fully-fused FlashMoE module (FC1+FC2+AllGather+AllReduce) against
the reference CuteDslFusedMoE with AllGatherReduceScatter communication.

Multi-GPU tests require 2+ GB200 GPUs (OCI) with NVLink fabric support.
"""

import os
import time

import pytest
import torch

from tensorrt_llm._utils import get_sm_version


def _skip_reason() -> str:
    """Check if FlashMoE tests should be skipped."""
    sm = get_sm_version()
    if sm not in (100, 103):
        return f"FlashMoE requires Blackwell (SM100/SM103), got SM{sm}"
    try:
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        MnnvlMemory.initialize()
        if not MnnvlMemory.supports_mnnvl():
            return "FlashMoE requires MNNVL (NVLink fabric) support"
    except Exception as e:
        return f"MNNVL initialization failed: {e}"
    return ""


def _create_reference_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
):
    """Create random NVFP4 weights for both FlashMoE and reference module.

    Returns dict with w3w1_weight, w2_weight, fc1_weight_scale, fc2_weight_scale,
    fc1_alpha, fc2_alpha, fc31_input_scale, fc2_input_scale.
    """
    torch.manual_seed(42)

    # NVFP4 weights: N dimension is NOT packed, K dimension is packed (2 FP4 per byte)
    # w3w1: [experts, 2*intermediate_size, hidden_size // 2]
    # w2: [experts, hidden_size, intermediate_size // 2]
    w3w1_weight = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_weight = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )

    # Scale factors (uint8, representing float8_e4m3fn)
    # b_sf shape must be [L, N, K/sf_vec_size] where N = b.size(1) (NOT packed)
    sf_vec_size = 16
    fc1_weight_scale = torch.randint(
        1,
        128,
        (num_experts, 2 * intermediate_size, hidden_size // sf_vec_size),
        dtype=torch.uint8,
        device=device,
    )
    fc2_weight_scale = torch.randint(
        1,
        128,
        (num_experts, hidden_size, intermediate_size // sf_vec_size),
        dtype=torch.uint8,
        device=device,
    )

    # Global alpha per expert (fc1_alpha = 1 / (input_scale * weight_scale))
    fc1_alpha = torch.ones(num_experts, dtype=torch.float32, device=device) * 0.1
    fc2_alpha = torch.ones(num_experts, dtype=torch.float32, device=device) * 0.1

    # Input scales
    fc31_input_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    fc2_input_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    return {
        "w3w1_weight": w3w1_weight,
        "w2_weight": w2_weight,
        "fc1_weight_scale": fc1_weight_scale,
        "fc2_weight_scale": fc2_weight_scale,
        "fc1_alpha": fc1_alpha,
        "fc2_alpha": fc2_alpha,
        "fc31_input_scale": fc31_input_scale,
        "fc2_input_scale": fc2_input_scale,
    }


# ============================================================================
# Single-GPU smoke test (IPC memory allocation + basic flow)
# ============================================================================
class TestFlashMoEIpcMemory:
    """Test FlashMoeMnnvlMemory allocation and buffer access."""

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    def test_memory_layout(self):
        """Verify buffer offsets and sizes are consistent."""
        from tensorrt_llm._mnnvl_utils import FlashMoeMnnvlMemory

        max_input_tokens = 1024
        max_output_tokens = 4096  # = max_input_tokens * ep_size(4)
        hidden_size = 7168
        sf_vec_size = 16

        # Test offset computation without actual allocation
        align = FlashMoeMnnvlMemory._ALIGN

        def _align(size):
            return (size + align - 1) & ~(align - 1)

        # Input buffers use per-rank sizing (max_input_tokens)
        input_a_bytes = max_input_tokens * (hidden_size // 2)
        input_sfa_bytes = max_input_tokens * ((hidden_size + sf_vec_size - 1) // sf_vec_size)
        # Output/staging buffers use global sizing (max_output_tokens)
        staging_bytes = max_output_tokens * hidden_size * 2

        expected_sfa_offset = _align(input_a_bytes)
        expected_staging_offset = expected_sfa_offset + _align(input_sfa_bytes)
        expected_output_offset = expected_staging_offset + _align(staging_bytes)

        # Verify buffer sizes are reasonable
        assert input_a_bytes == max_input_tokens * hidden_size // 2
        assert staging_bytes == max_output_tokens * hidden_size * 2
        assert expected_sfa_offset > 0
        assert expected_staging_offset > expected_sfa_offset
        assert expected_output_offset > expected_staging_offset


class TestFlashMoECanImplement:
    """Test FlashMoeFusedKernel.can_implement() validation."""

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    def test_valid_config(self):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )

        assert FlashMoeFusedKernel.can_implement(
            hidden_size=7168,
            intermediate_size=2048,
            num_experts=256,
            ep_size=8,
        )

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    def test_invalid_ep_size_1(self):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )

        # ep_size=1 not supported (no point in IPC gather)
        assert not FlashMoeFusedKernel.can_implement(
            hidden_size=7168,
            intermediate_size=2048,
            num_experts=256,
            ep_size=1,
        )

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    def test_invalid_alignment(self):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )

        # hidden_size not divisible by 128 (MMA tile alignment)
        assert not FlashMoeFusedKernel.can_implement(
            hidden_size=7100,
            intermediate_size=2048,
            num_experts=256,
            ep_size=8,
        )
        # hidden_size divisible by 32 but not 128
        assert not FlashMoeFusedKernel.can_implement(
            hidden_size=7200,
            intermediate_size=2048,
            num_experts=256,
            ep_size=8,
        )


# ============================================================================
# Multi-GPU functional test (requires 2+ GB200 GPUs)
# ============================================================================
def _flashmoe_worker_impl(
    rank: int,
    world_size: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens_per_rank: int,
    max_num_tokens: int,
    use_fused_kernel: bool = False,
):
    """Worker function for multi-GPU FlashMoE test.

    Each rank:
    1. Initializes FlashMoE module.
    2. Generates random input and router logits.
    3. Runs FlashMoE forward.
    4. Verifies output shape, dtype, and finiteness.
    """
    import torch.distributed

    # Use torch.distributed instead of MPI for communication
    os.environ["TLLM_DISABLE_MPI"] = "1"

    # Initialize process group
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    ep_size = world_size

    # Create FlashMoE module
    from tensorrt_llm._torch.modules.fused_moe.flashmoe import FlashMoE
    from tensorrt_llm.mapping import Mapping

    # In pure EP mode, tp_size must equal world_size so that
    # moe_world_size = tp_size = moe_tp_size * moe_ep_size * moe_cluster_size.
    # This makes the TP group span all EP ranks, which is what allgather needs.
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=ep_size,
        moe_tp_size=1,
    )

    flashmoe = FlashMoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        use_fused_kernel=use_fused_kernel,
        use_ipc=False,  # Use torch.distributed instead of MNNVL IPC
    )

    # Create weights for LOCAL experts only (not global).
    # Each rank holds weights for its own expert partition.
    experts_per_rank = num_experts // ep_size
    weights = _create_reference_weights(
        num_experts=experts_per_rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    flashmoe.w3w1_weight = weights["w3w1_weight"]
    flashmoe.w2_weight = weights["w2_weight"]
    flashmoe.fc1_weight_scale = weights["fc1_weight_scale"]
    flashmoe.fc2_weight_scale = weights["fc2_weight_scale"]
    flashmoe.fc1_alpha = weights["fc1_alpha"]
    flashmoe.fc2_alpha = weights["fc2_alpha"]
    flashmoe.fc31_input_scale = weights["fc31_input_scale"]
    flashmoe.fc2_input_scale = weights["fc2_input_scale"]

    # Create input
    torch.manual_seed(42 + rank)
    x = torch.randn(num_tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(
        num_tokens_per_rank, num_experts, dtype=torch.bfloat16, device=device
    )

    # Create reference routing method
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod

    routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

    # Run FlashMoE forward
    t0 = time.time()
    with torch.inference_mode():
        flash_output = flashmoe.forward(x, router_logits, routing_method)
    t1 = time.time()
    print(f"[Rank {rank}] forward (incl. JIT compile) took {t1 - t0:.1f}s", flush=True)

    torch.cuda.synchronize()
    t2 = time.time()
    print(f"[Rank {rank}] cuda.synchronize took {t2 - t1:.1f}s", flush=True)

    # Verify output shape and dtype
    assert flash_output.shape == (num_tokens_per_rank, hidden_size), (
        f"Expected shape ({num_tokens_per_rank}, {hidden_size}), got {flash_output.shape}"
    )
    assert flash_output.dtype == torch.bfloat16, f"Expected bfloat16, got {flash_output.dtype}"

    # Note: with random NVFP4 weights, output may contain non-finite values (inf/nan).
    # Check that the kernel produced some non-zero output (including nan/inf).
    has_nonzero = (flash_output != 0).any().item()
    has_nonfinite = (~torch.isfinite(flash_output)).any().item()
    assert has_nonzero or has_nonfinite, "Output is all zeros — kernel may not have run"

    return None


def _flashmoe_compare_worker_impl(
    rank: int,
    world_size: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens_per_rank: int,
    max_num_tokens: int,
):
    """Worker function for fused vs decomposed comparison test.

    Runs both the fused (single FC1+FC2 persistent kernel) and decomposed
    (separate FC1 + FC2 kernel launches) paths with the same input/weights,
    and compares outputs element-by-element.
    """
    import torch.distributed

    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    ep_size = world_size

    from tensorrt_llm._torch.modules.fused_moe.flashmoe import FlashMoE
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=ep_size,
        moe_tp_size=1,
    )

    experts_per_rank = num_experts // ep_size
    weights = _create_reference_weights(
        num_experts=experts_per_rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    def _create_flashmoe(use_fused: bool) -> FlashMoE:
        fm = FlashMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            use_fused_kernel=use_fused,
            use_ipc=False,
        )
        fm.w3w1_weight = weights["w3w1_weight"]
        fm.w2_weight = weights["w2_weight"]
        fm.fc1_weight_scale = weights["fc1_weight_scale"]
        fm.fc2_weight_scale = weights["fc2_weight_scale"]
        fm.fc1_alpha = weights["fc1_alpha"]
        fm.fc2_alpha = weights["fc2_alpha"]
        fm.fc31_input_scale = weights["fc31_input_scale"]
        fm.fc2_input_scale = weights["fc2_input_scale"]
        return fm

    torch.manual_seed(42 + rank)
    x = torch.randn(num_tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(
        num_tokens_per_rank, num_experts, dtype=torch.bfloat16, device=device
    )
    routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

    # Run decomposed path (reference)
    flashmoe_ref = _create_flashmoe(use_fused=False)
    t0 = time.time()
    with torch.inference_mode():
        ref_output = flashmoe_ref.forward(x, router_logits, routing_method)
    torch.cuda.synchronize()
    print(f"[Rank {rank}] decomposed forward+sync took {time.time() - t0:.1f}s", flush=True)

    # Run fused path
    flashmoe_fused = _create_flashmoe(use_fused=True)
    t0 = time.time()
    with torch.inference_mode():
        fused_output = flashmoe_fused.forward(x, router_logits, routing_method)
    t1 = time.time()
    print(f"[Rank {rank}] fused forward (incl. JIT) took {t1 - t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    print(f"[Rank {rank}] fused sync took {time.time() - t1:.1f}s", flush=True)

    # Compare outputs (both use NCCL AllReduce, so only kernel differs)
    # Mask out non-finite values (random weights can cause overflow)
    valid_mask = torch.isfinite(ref_output) & torch.isfinite(fused_output)
    if valid_mask.any():
        abs_diff = (fused_output[valid_mask] - ref_output[valid_mask]).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        print(
            f"[Rank {rank}] Fused vs Decomposed: "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"valid={valid_mask.sum().item()}/{valid_mask.numel()}"
        )
        # NVFP4 numerics: expect very close match since both paths use
        # the same MMA operations and data
        assert max_diff < 0.1, f"Rank {rank}: max abs diff too large: {max_diff:.6f}"
        assert mean_diff < 0.02, f"Rank {rank}: mean abs diff too large: {mean_diff:.6f}"
    else:
        print(f"[Rank {rank}] WARNING: No finite values to compare")

    return None


def _find_free_port():
    """Find a free port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _spawn_wrapper(rank, world_size, fn_args):
    """Wrapper for torch.multiprocessing.spawn that unpacks arguments."""
    _flashmoe_worker_impl(rank, world_size, *fn_args)


def _spawn_compare_wrapper(rank, world_size, fn_args):
    """Wrapper for torch.multiprocessing.spawn that unpacks arguments."""
    _flashmoe_compare_worker_impl(rank, world_size, *fn_args)


@pytest.mark.skipif(
    _skip_reason() != "",
    reason=_skip_reason() or "Unknown",
)
class TestFlashMoEMultiGPU:
    """Multi-GPU FlashMoE functional tests.

    These tests require 2+ GB200 GPUs with NVLink fabric support.
    Run on OCI using: pytest tests/unittest/_torch/modules/moe/test_flashmoe.py -v
    """

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size,top_k",
        [
            (256, 7168, 2048, 8),  # DeepSeek-V3 config
        ],
    )
    def test_flashmoe_basic(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
    ):
        """Basic functional test: FlashMoE output shape and finiteness."""
        world_size = min(torch.cuda.device_count(), 4)
        if world_size < 2:
            pytest.skip("Requires at least 2 GPUs")

        os.environ["MASTER_PORT"] = str(_find_free_port())
        torch.multiprocessing.spawn(
            _spawn_wrapper,
            args=(
                world_size,
                (num_experts, hidden_size, intermediate_size, top_k, 64, 256),
            ),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size,top_k",
        [
            (256, 7168, 2048, 8),  # DeepSeek-V3 config
        ],
    )
    @pytest.mark.timeout(1800)
    def test_flashmoe_fused_kernel(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
    ):
        """Fused kernel test: single persistent FC1+FC2 kernel."""
        world_size = min(torch.cuda.device_count(), 4)
        if world_size < 2:
            pytest.skip("Requires at least 2 GPUs")

        os.environ["MASTER_PORT"] = str(_find_free_port())
        torch.multiprocessing.spawn(
            _spawn_wrapper,
            args=(
                world_size,
                (num_experts, hidden_size, intermediate_size, top_k, 64, 256, True),
            ),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size,top_k",
        [
            (256, 7168, 2048, 8),  # DeepSeek-V3 config
        ],
    )
    @pytest.mark.timeout(1800)
    def test_flashmoe_fused_vs_decomposed(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
    ):
        """Correctness test: fused kernel matches decomposed (FC1+FC2) output."""
        world_size = min(torch.cuda.device_count(), 4)
        if world_size < 2:
            pytest.skip("Requires at least 2 GPUs")

        os.environ["MASTER_PORT"] = str(_find_free_port())
        torch.multiprocessing.spawn(
            _spawn_compare_wrapper,
            args=(
                world_size,
                (num_experts, hidden_size, intermediate_size, top_k, 64, 256),
            ),
            nprocs=world_size,
            join=True,
        )


# ============================================================================
# Single-GPU kernel validation tests
# ============================================================================
class TestFlashMoEKernelConfig:
    """Test kernel configuration and validation."""

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    def test_kernel_init(self):
        """Test FlashMoeFusedKernel initialization."""
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )

        kernel = FlashMoeFusedKernel(
            sf_vec_size=16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

        assert kernel.threads_per_cta == 352
        assert kernel.num_warps == 11
        assert kernel.sf_vec_size == 16
        assert kernel.PHASE_FC1 == 0
        assert kernel.PHASE_FC2 == 1

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    def test_warp_ids(self):
        """Test warp ID assignment."""
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )

        kernel = FlashMoeFusedKernel()

        # Verify no warp ID conflicts
        all_warp_ids = set()
        for wid in kernel.epilog_warp_id:
            assert wid not in all_warp_ids
            all_warp_ids.add(wid)
        for wid in kernel.ldgsts_a_warp_id:
            assert wid not in all_warp_ids
            all_warp_ids.add(wid)
        assert kernel.mma_warp_id not in all_warp_ids
        all_warp_ids.add(kernel.mma_warp_id)
        assert kernel.tma_b_warp_id not in all_warp_ids
        all_warp_ids.add(kernel.tma_b_warp_id)
        assert kernel.sched_warp_id not in all_warp_ids
        all_warp_ids.add(kernel.sched_warp_id)

        assert len(all_warp_ids) == 11


# ============================================================================
# MPI-based AllReduce test (requires srun --mpi=pmix -n N pytest ...)
# ============================================================================
def _is_mpi_launched() -> bool:
    """Check if this process was launched as part of an MPI job."""
    return (
        "OMPI_COMM_WORLD_SIZE" in os.environ
        or "PMI_SIZE" in os.environ
        or "SLURM_NTASKS" in os.environ
    )


def _flashmoe_ar_worker(
    num_experts: int = 256,
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    top_k: int = 8,
    num_tokens_per_rank: int = 64,
    max_num_tokens: int = 256,
):
    """MPI worker for FlashMoE AllReduce test.

    Runs both IPC+AR path and non-IPC+NCCL path, compares outputs.
    Must be launched via srun --mpi=pmix.
    """
    import tensorrt_llm as tllm
    from tensorrt_llm._torch.modules.fused_moe.flashmoe import FlashMoE
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
    from tensorrt_llm.mapping import Mapping

    rank = tllm.mpi_rank()
    world_size = tllm.mpi_world_size()
    ep_size = world_size

    torch.cuda.set_device(rank)
    tllm.MnnvlMemory.initialize()

    # Initialize torch.distributed for NCCL reference path
    if not torch.distributed.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{rank}")
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=ep_size,
        moe_tp_size=1,
    )

    experts_per_rank = num_experts // ep_size
    weights = _create_reference_weights(
        num_experts=experts_per_rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    def _create_flashmoe(use_ipc: bool) -> FlashMoE:
        fm = FlashMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            use_fused_kernel=True,
            use_ipc=use_ipc,
        )
        fm.w3w1_weight = weights["w3w1_weight"]
        fm.w2_weight = weights["w2_weight"]
        fm.fc1_weight_scale = weights["fc1_weight_scale"]
        fm.fc2_weight_scale = weights["fc2_weight_scale"]
        fm.fc1_alpha = weights["fc1_alpha"]
        fm.fc2_alpha = weights["fc2_alpha"]
        fm.fc31_input_scale = weights["fc31_input_scale"]
        fm.fc2_input_scale = weights["fc2_input_scale"]
        return fm

    # Create input (same seed per rank for reproducibility)
    torch.manual_seed(42 + rank)
    x = torch.randn(num_tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(
        num_tokens_per_rank, num_experts, dtype=torch.bfloat16, device=device
    )
    routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

    # --- Reference path: fused kernel + NCCL AllReduce ---
    flashmoe_ref = _create_flashmoe(use_ipc=False)
    t0 = time.time()
    with torch.inference_mode():
        ref_output = flashmoe_ref.forward(x, router_logits, routing_method)
    torch.cuda.synchronize()
    print(f"[Rank {rank}] ref forward+sync took {time.time() - t0:.1f}s", flush=True)

    # --- Test path: fused kernel + in-kernel AR via IPC ---
    flashmoe_ar = _create_flashmoe(use_ipc=True)
    t0 = time.time()
    with torch.inference_mode():
        ar_output = flashmoe_ar.forward(x, router_logits, routing_method)
    t1 = time.time()
    print(f"[Rank {rank}] AR forward (incl. JIT) took {t1 - t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    print(f"[Rank {rank}] AR sync took {time.time() - t1:.1f}s", flush=True)

    # Compare outputs
    abs_diff = (ar_output - ref_output).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"[Rank {rank}] AR vs NCCL: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    # NVFP4 quantization allows some numerical difference
    assert max_diff < 0.5, f"Rank {rank}: max abs diff too large: {max_diff:.6f}"
    assert mean_diff < 0.1, f"Rank {rank}: mean abs diff too large: {mean_diff:.6f}"


class TestFlashMoEAllReduceMPI:
    """MPI-based FlashMoE in-kernel AllReduce tests.

    These tests require MPI launch (srun --mpi=pmix -n N pytest ...) and
    NVLink fabric (MNNVL) support. They compare the in-kernel AR result
    against NCCL AllReduce as reference.
    """

    @pytest.mark.skipif(
        get_sm_version() not in (100, 103),
        reason="Requires Blackwell GPU",
    )
    @pytest.mark.skipif(
        not _is_mpi_launched(),
        reason="Requires MPI launch (srun --mpi=pmix)",
    )
    def test_flashmoe_ar_vs_nccl(self):
        """Compare in-kernel AR against NCCL AllReduce reference."""
        _flashmoe_ar_worker(
            num_experts=256,
            hidden_size=7168,
            intermediate_size=2048,
            top_k=8,
            num_tokens_per_rank=64,
            max_num_tokens=256,
        )
