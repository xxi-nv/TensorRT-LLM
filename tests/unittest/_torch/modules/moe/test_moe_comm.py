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
Unified Communication Strategy Tests for MoE

Tests all Communication subclasses (AllGatherReduceScatter, DeepEP,
DeepEPLowLatency, NVLinkOneSided, NVLinkTwoSided) via the public
dispatch() and combine() interfaces defined in Communication base class.

Each test runs the full pipeline: dispatch -> verify dispatch -> simple_moe
-> combine -> verify combine.  Dispatch errors are caught early with clear
diagnostics before combine verification runs.

Dispatch verification:
  - AllGatherRS: verifies allgathered data is bitwise exact concatenation
  - AllToAll comms: encodes (rank_id, token_idx) in hidden_states bytes,
    then checks routing correctness, content integrity, and completeness

Combine verification:
  Uses simple_moe (weighted sum of hidden_states, no expert-specific
  computation).  Routing correctness is already covered by dispatch
  verification.  Dispatch-returned token_final_scales handle the
  scale-application asymmetry across comms (DeepEPLL returns ones,
  others return real scales).

Singleton safety:
  NVLinkOneSided._WORKSPACE is reset before each creation to avoid
  assertion failures from varying params across MPI process reuse.
  All tests use num_experts=32 to avoid DeepEP buffer_pool num_experts
  assertion failures.

Run with: mpirun -np 8 pytest test_moe_comm.py -x -v
"""

import os
import pickle
import struct
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm.mapping import Mapping

# NOTE: Communication subclass imports (DeepEP, NVLink, etc.) and platform
# utility imports (MnnvlMemory, deep_ep_installed) are intentionally lazy-loaded
# inside the functions that use them.  These modules depend on optional
# libraries / hardware and would cause the entire test file to fail to import
# on machines without them.  Lazy loading allows unsupported tests to be
# gracefully skipped instead.

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# ============================================================================
# Constants
# ============================================================================

COMM_ALLGATHER_RS = "AllGatherReduceScatter"
COMM_DEEP_EP = "DeepEP"
COMM_DEEP_EP_LL = "DeepEPLowLatency"
COMM_NVLINK_ONE_SIDED = "NVLinkOneSided"
COMM_NVLINK_TWO_SIDED = "NVLinkTwoSided"

ALL_COMM_TYPES = [
    COMM_ALLGATHER_RS,
    COMM_DEEP_EP,
    COMM_DEEP_EP_LL,
    COMM_NVLINK_ONE_SIDED,
    COMM_NVLINK_TWO_SIDED,
]

# Must be in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES
DEFAULT_HIDDEN_SIZE = 4096

# Fixed across all tests to avoid DeepEP buffer_pool singleton conflicts
# (VariableLengthLowLatencyBuffer.reserve asserts num_experts is consistent).
FIXED_NUM_EXPERTS = 32

# Force consistent NVLinkOneSided workspace size across varying top_k
# to avoid _WORKSPACE singleton assertion failures.
NVLINK_WORKSPACE_MB = "512"


# ============================================================================
# Test Configuration
# ============================================================================


@dataclass
class CommTestConfig:
    """Configuration for a single comm test case."""

    comm_type: str
    ep_size: int
    num_experts: int
    top_k: int
    hidden_size: int
    all_num_tokens: List[int]

    def __str__(self) -> str:
        tokens_str = "x".join(str(t) for t in self.all_num_tokens)
        return (
            f"{self.comm_type}_ep{self.ep_size}_e{self.num_experts}"
            f"_k{self.top_k}_h{self.hidden_size}_t{tokens_str}"
        )


# ============================================================================
# Source Encoding Utilities
# ============================================================================


def encode_source_info(
    hidden_states: torch.Tensor,
    rank_id: int,
) -> torch.Tensor:
    """Encode (rank_id, token_idx) into the last 4 bytes of each row.

    Format: [rank_id_u16, token_idx_u16] in big-endian.
    Works regardless of dtype because we operate on the raw byte view.
    """
    hs = hidden_states.clone()
    flat_bytes = hs.view(torch.uint8)
    row_bytes = hidden_states.shape[1] * hidden_states.element_size()
    num_tokens = hidden_states.shape[0]

    for i in range(num_tokens):
        offset = i * row_bytes + (row_bytes - 4)
        packed = struct.pack(">HH", rank_id, i)
        for j, b in enumerate(packed):
            flat_bytes[offset + j] = b

    return hs


def decode_source_info(
    hidden_states: torch.Tensor,
    dtype: torch.dtype,
    hidden_size: int,
) -> List[Tuple[int, int]]:
    """Decode (rank_id, token_idx) from the last 4 bytes of each row."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    row_bytes = hidden_size * element_size
    flat_bytes = hidden_states.contiguous().view(torch.uint8)
    num_rows = flat_bytes.numel() // row_bytes
    results = []

    for i in range(num_rows):
        offset = i * row_bytes + (row_bytes - 4)
        raw = bytes(flat_bytes[offset : offset + 4].cpu().tolist())
        rank_id, token_idx = struct.unpack(">HH", raw)
        results.append((rank_id, token_idx))

    return results


# ============================================================================
# Simple MoE Substitute (for combine verification)
# ============================================================================


def simple_moe(
    hidden_states: torch.Tensor,
    token_selected_slots: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_slots: int,
    ep_rank: int,
    experts_per_rank: int,
) -> torch.Tensor:
    """Trivial MoE: weighted sum of hidden_states for local experts only.

    Each expert applies: output += hidden_states * scale.
    Dispatch verification already covers routing correctness, so experts
    do not need distinct computations.

    Uses dispatch-returned token_final_scales which handles the
    scale-application asymmetry:
    - AllGatherRS / NVLink / DeepEP: real scales (MoE applies them)
    - DeepEPLL: all-ones (combine applies real scales internally)
    """
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    slot_start = ep_rank * experts_per_rank
    slot_end = slot_start + experts_per_rank

    for i in range(hidden_states.shape[0]):
        for k in range(token_selected_slots.shape[1]):
            eid = token_selected_slots[i, k].item()
            if not (slot_start <= eid < slot_end):
                continue
            output[i] += hidden_states[i].float() * token_final_scales[i, k].float()

    return output.to(hidden_states.dtype)


def simple_moe_reference(
    hidden_states: torch.Tensor,
    token_selected_slots: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Single-card reference: weighted sum with ALL experts (no EP)."""
    output = torch.zeros_like(hidden_states, dtype=torch.float32)

    for i in range(hidden_states.shape[0]):
        for k in range(token_selected_slots.shape[1]):
            eid = token_selected_slots[i, k].item()
            if eid < 0 or eid >= num_experts:
                continue
            output[i] += hidden_states[i].float() * token_final_scales[i, k].float()

    return output.to(hidden_states.dtype)


# ============================================================================
# Communication Object Factory
# ============================================================================


def create_comm_object(
    comm_type: str,
    mapping: Mapping,
    config: CommTestConfig,
):
    """Create a Communication object for the given type and config."""
    num_experts = config.num_experts
    num_slots = num_experts
    max_num_tokens = max(config.all_num_tokens)

    if comm_type == COMM_ALLGATHER_RS:
        from tensorrt_llm._torch.modules.fused_moe.communication.allgather_reducescatter import (
            AllGatherReduceScatter,
        )

        return AllGatherReduceScatter(mapping=mapping)

    elif comm_type == COMM_DEEP_EP:
        from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep import DeepEP

        return DeepEP(
            mapping=mapping,
            num_slots=num_slots,
            hidden_size=config.hidden_size,
            weight_dtype=torch.bfloat16,
            quant_config=None,
            expert_size_per_partition=num_experts // config.ep_size,
        )

    elif comm_type == COMM_DEEP_EP_LL:
        from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency import (
            DeepEPLowLatency,
        )

        return DeepEPLowLatency(
            mapping=mapping,
            num_slots=num_slots,
            hidden_size=config.hidden_size,
            weight_dtype=torch.bfloat16,
            quant_config=None,
            expert_size_per_partition=num_experts // config.ep_size,
            max_num_tokens=max_num_tokens,
            moe_max_num_tokens=max_num_tokens,
        )

    elif comm_type == COMM_NVLINK_ONE_SIDED:
        from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import (
            NVLinkOneSided,
        )

        # Reset class-level singleton to avoid assertion failures when
        # test params change across MPI process reuse.
        NVLinkOneSided._WORKSPACE = None
        os.environ["TRTLLM_MOE_A2A_WORKSPACE_MB"] = NVLINK_WORKSPACE_MB

        return NVLinkOneSided(
            mapping=mapping,
            num_slots=num_slots,
            top_k=config.top_k,
            max_num_tokens_per_rank=max_num_tokens,
            hidden_size=config.hidden_size,
            dtype=torch.bfloat16,
        )

    elif comm_type == COMM_NVLINK_TWO_SIDED:
        from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided import (
            NVLinkTwoSided,
        )

        return NVLinkTwoSided(
            mapping=mapping,
            num_experts=num_experts,
            num_slots=num_slots,
            top_k=config.top_k,
        )

    else:
        raise ValueError(f"Unknown comm type: {comm_type}")


# ============================================================================
# Platform / Feasibility Checks
# ============================================================================


def _check_mnnvl_support() -> Optional[str]:
    """Return skip reason if MNNVL is not supported, else None."""
    from tensorrt_llm._mnnvl_utils import MnnvlMemory

    try:
        MnnvlMemory.initialize()
        if not MnnvlMemory.supports_mnnvl():
            return "MNNVL not supported"
    except Exception:
        return "MNNVL initialization failed"
    return None


def check_platform_support(comm_type: str) -> Optional[str]:
    """Return skip reason string if comm type is unsupported, else None."""
    if comm_type == COMM_ALLGATHER_RS:
        return None

    if comm_type in (COMM_DEEP_EP, COMM_DEEP_EP_LL):
        try:
            from tensorrt_llm._torch.modules.fused_moe.deep_ep_utils import deep_ep_installed

            if not deep_ep_installed:
                return "DeepEP library not installed"
        except ImportError:
            return "DeepEP library not importable"
        return _check_mnnvl_support()

    if comm_type in (COMM_NVLINK_ONE_SIDED, COMM_NVLINK_TWO_SIDED):
        return _check_mnnvl_support()

    return f"Unknown comm type: {comm_type}"


def check_feasibility(comm_type: str, config: CommTestConfig) -> Optional[str]:
    """Return skip reason string if config is infeasible, else None."""
    if config.num_experts % config.ep_size != 0:
        return f"num_experts={config.num_experts} not divisible by ep_size={config.ep_size}"

    if comm_type in (COMM_DEEP_EP, COMM_DEEP_EP_LL):
        from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep import DeepEP

        if not DeepEP._is_deepep_feasible(config.ep_size):
            return f"DeepEP not feasible for ep_size={config.ep_size}"

    if comm_type == COMM_DEEP_EP_LL:
        from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency import (
            DeepEPLowLatency,
        )

        if config.hidden_size not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES:
            return f"DeepEPLL does not support hidden_size={config.hidden_size}"

    if comm_type == COMM_NVLINK_ONE_SIDED:
        from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import (
            NVLinkOneSided,
        )

        if config.top_k > NVLinkOneSided.MAX_TOP_K:
            return f"NVLinkOneSided MAX_TOP_K={NVLinkOneSided.MAX_TOP_K}, got top_k={config.top_k}"

    return None


# ============================================================================
# Worker Function (runs on each MPI rank)
# ============================================================================


def _generate_test_data(
    rank: int,
    config: CommTestConfig,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test data for a single rank.

    Returns (hidden_states, token_selected_slots, token_final_scales).
    hidden_states has source info encoded in last 4 bytes of each row.
    """
    num_tokens = config.all_num_tokens[rank]

    torch.manual_seed(seed + rank)

    hidden_states = torch.randn(num_tokens, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    hidden_states = encode_source_info(hidden_states, rank)

    token_selected_slots = torch.randint(
        0, config.num_experts, (num_tokens, config.top_k), dtype=torch.int32, device="cuda"
    )

    token_final_scales = torch.rand(num_tokens, config.top_k, dtype=torch.bfloat16, device="cuda")
    # Avoid near-zero scales that amplify relative error in bf16 verification.
    token_final_scales = token_final_scales.clamp(min=0.1)

    return hidden_states, token_selected_slots, token_final_scales


def _worker_full_pipeline(config: CommTestConfig) -> dict:
    """Run dispatch -> simple_moe -> combine on a single MPI rank.

    Returns both dispatch intermediate results (for dispatch verification)
    and final combine output (for combine verification).
    """
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    comm = None
    try:
        mapping = Mapping(
            rank=rank,
            tp_size=config.ep_size,
            moe_ep_size=config.ep_size,
            world_size=config.ep_size,
        )

        comm = create_comm_object(config.comm_type, mapping, config)
        hs, slots, scales = _generate_test_data(rank, config)

        if config.comm_type == COMM_NVLINK_TWO_SIDED:
            comm.prepare_dispatch(slots, config.all_num_tokens)

        recv_hs, recv_sf, recv_slots, recv_scales = comm.dispatch(
            hs,
            None,
            slots,
            scales,
            config.all_num_tokens,
        )

        experts_per_rank = config.num_experts // config.ep_size
        moe_output = simple_moe(
            recv_hs,
            recv_slots,
            recv_scales,
            config.num_experts,
            rank,
            experts_per_rank,
        )

        combined = comm.combine(
            moe_output,
            all_rank_max_num_tokens=max(config.all_num_tokens),
        )

        return {
            "rank": rank,
            "original_hs": hs.cpu(),
            "original_slots": slots.cpu(),
            "original_scales": scales.cpu(),
            "recv_hs": recv_hs.cpu(),
            "recv_slots": recv_slots.cpu(),
            "recv_scales": recv_scales.cpu() if recv_scales is not None else None,
            "combined": combined.cpu(),
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if comm is not None and hasattr(comm, "destroy"):
            comm.destroy()


# ============================================================================
# Verification Functions
# ============================================================================


def verify_dispatch_allgather_rs(
    all_results: List[dict],
    config: CommTestConfig,
):
    """Verify AllGatherRS: allgathered data is bitwise-exact concatenation."""
    total_tokens = sum(config.all_num_tokens)

    for result in all_results:
        recv_hs = result["recv_hs"]
        assert recv_hs.shape[0] == total_tokens, (
            f"Rank {result['rank']}: expected {total_tokens} tokens, got {recv_hs.shape[0]}"
        )

        offset = 0
        for src_rank in range(config.ep_size):
            src_hs = all_results[src_rank]["original_hs"]
            n = config.all_num_tokens[src_rank]
            assert torch.equal(recv_hs[offset : offset + n], src_hs), (
                f"Rank {result['rank']}: chunk for source rank {src_rank} mismatch"
            )
            offset += n


def _compute_expected_tokens_per_rank(
    all_results: List[dict],
    config: CommTestConfig,
) -> Dict[int, Set[Tuple[int, int]]]:
    """Compute the set of (src_rank, token_idx) each rank should receive.

    A token is expected at dest_rank if at least one of its top_k expert
    selections falls in dest_rank's expert range.
    """
    experts_per_rank = config.num_experts // config.ep_size

    expected: Dict[int, Set[Tuple[int, int]]] = {r: set() for r in range(config.ep_size)}
    for result in all_results:
        src_rank = result["rank"]
        slots = result["original_slots"]

        for i in range(slots.shape[0]):
            for k in range(slots.shape[1]):
                eid = slots[i, k].item()
                if 0 <= eid < config.num_experts:
                    expected[eid // experts_per_rank].add((src_rank, i))

    return expected


def verify_dispatch_alltoall(
    all_results: List[dict],
    config: CommTestConfig,
):
    """Verify AllToAll dispatch: slot validity, content integrity, completeness.

    1. All received slots are local experts or invalid (-1 / >= num_experts)
    2. Encoded (rank_id, token_idx) in received data matches original content
    3. All tokens that should be routed to this rank are present (completeness)
    """
    num_experts = config.num_experts
    experts_per_rank = num_experts // config.ep_size

    # Global lookup: (src_rank, token_idx) -> original hidden_states row
    original_data: Dict[Tuple[int, int], torch.Tensor] = {}
    for result in all_results:
        src_rank = result["rank"]
        orig_hs = result["original_hs"]
        for i in range(orig_hs.shape[0]):
            original_data[(src_rank, i)] = orig_hs[i]

    expected_per_rank = _compute_expected_tokens_per_rank(all_results, config)

    for result in all_results:
        recv_rank = result["rank"]
        recv_hs = result["recv_hs"]
        recv_slots = result["recv_slots"]

        slot_start = recv_rank * experts_per_rank
        slot_end = slot_start + experts_per_rank

        decoded = decode_source_info(recv_hs, torch.bfloat16, config.hidden_size)
        top_k = recv_slots.shape[1]

        actually_received: Set[Tuple[int, int]] = set()

        for i in range(recv_hs.shape[0]):
            for k in range(top_k):
                slot = recv_slots[i, k].item()
                is_local = slot_start <= slot < slot_end
                is_invalid = slot < 0 or slot >= num_experts
                assert is_local or is_invalid, (
                    f"Rank {recv_rank}, token {i}, k={k}: slot={slot} "
                    f"not local [{slot_start},{slot_end}) and not invalid"
                )

            has_valid = any(slot_start <= recv_slots[i, k].item() < slot_end for k in range(top_k))
            if has_valid:
                src_rank, src_idx = decoded[i]
                key = (src_rank, src_idx)
                actually_received.add(key)
                if key in original_data:
                    assert torch.equal(recv_hs[i], original_data[key]), (
                        f"Rank {recv_rank}, token {i}: content mismatch. "
                        f"Source: rank={src_rank}, idx={src_idx}"
                    )

        # Completeness check.
        # Skip for DeepEPLL: tokens are duplicated per expert and top_k is
        # collapsed to 1, so source encoding cannot distinguish duplicates.
        if config.comm_type != COMM_DEEP_EP_LL:
            expected_tokens = expected_per_rank[recv_rank]
            missing = expected_tokens - actually_received
            assert not missing, (
                f"Rank {recv_rank}: {len(missing)} expected tokens not received. "
                f"Expected {len(expected_tokens)}, got {len(actually_received)}. "
                f"Missing (first 5): {list(missing)[:5]}"
            )


def verify_dispatch_results(
    all_results: List[dict],
    config: CommTestConfig,
):
    """Route to the appropriate dispatch verification based on comm type."""
    if config.comm_type == COMM_ALLGATHER_RS:
        verify_dispatch_allgather_rs(all_results, config)
    else:
        verify_dispatch_alltoall(all_results, config)


def verify_combine_results(
    all_results: List[dict],
    config: CommTestConfig,
    rtol: float = 0.05,
    atol: float = 0.1,
):
    """Verify combine results against single-card reference.

    Computes simple_moe_reference (all experts, no EP) and compares
    with the distributed dispatch+simple_moe+combine pipeline output.
    """
    for result in all_results:
        rank = result["rank"]
        original_hs = result["original_hs"]
        original_slots = result["original_slots"]
        original_scales = result["original_scales"]
        combined = result["combined"]

        num_tokens = original_hs.shape[0]
        if num_tokens == 0:
            continue

        ref = simple_moe_reference(
            original_hs,
            original_slots,
            original_scales,
            config.num_experts,
        )

        combined_tokens = combined[:num_tokens]
        ref_tokens = ref[:num_tokens]

        assert combined_tokens.shape == ref_tokens.shape, (
            f"Rank {rank}: shape mismatch. combined={combined_tokens.shape}, ref={ref_tokens.shape}"
        )

        try:
            torch.testing.assert_close(
                combined_tokens.float(),
                ref_tokens.float(),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as e:
            abs_diff = (combined_tokens.float() - ref_tokens.float()).abs()
            max_idx = abs_diff.argmax().item()
            token_idx = max_idx // combined_tokens.shape[1]
            elem_idx = max_idx % combined_tokens.shape[1]
            raise AssertionError(
                f"Rank {rank}: combine mismatch at token={token_idx}, elem={elem_idx}. "
                f"combined={combined_tokens[token_idx, elem_idx]:.6f}, "
                f"ref={ref_tokens[token_idx, elem_idx]:.6f}, "
                f"max_abs_diff={abs_diff.max():.6f}\n"
                f"Original error: {e}"
            ) from e


# ============================================================================
# Test Parameter Generation
# ============================================================================


def _make_workloads(ep_size: int) -> List[List[int]]:
    """Generate token distributions: uniform, non-uniform, minimal."""
    workloads = [[32] * ep_size]

    if ep_size == 2:
        workloads.append([16, 48])
    elif ep_size == 4:
        workloads.append([16, 32, 48, 64])

    workloads.append([1] * ep_size)
    return workloads


def _make_test_params():
    """Generate full-pipeline test parameters.

    Each entry is (ep_size, config).  ep_size is passed to mpi_pool_executor
    via indirect parametrization; config is passed directly to the test.
    """
    params = []
    for comm_type in ALL_COMM_TYPES:
        for ep_size in [2, 4]:
            for top_k in [2, 4, 8]:
                for workload in _make_workloads(ep_size):
                    config = CommTestConfig(
                        comm_type=comm_type,
                        ep_size=ep_size,
                        num_experts=FIXED_NUM_EXPERTS,
                        top_k=top_k,
                        hidden_size=DEFAULT_HIDDEN_SIZE,
                        all_num_tokens=workload,
                    )
                    params.append(pytest.param(ep_size, config, id=str(config)))
    return params


def _make_boundary_test_params():
    """Generate boundary / edge-case test parameters."""
    params = []
    for comm_type in ALL_COMM_TYPES:
        params.append(
            pytest.param(
                2,
                CommTestConfig(
                    comm_type=comm_type,
                    ep_size=2,
                    num_experts=FIXED_NUM_EXPERTS,
                    top_k=1,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    all_num_tokens=[8, 8],
                ),
                id=f"{comm_type}_topk1",
            )
        )

        params.append(
            pytest.param(
                2,
                CommTestConfig(
                    comm_type=comm_type,
                    ep_size=2,
                    num_experts=FIXED_NUM_EXPERTS,
                    top_k=2,
                    hidden_size=2048,
                    all_num_tokens=[16, 16],
                ),
                id=f"{comm_type}_h2048",
            )
        )

        params.append(
            pytest.param(
                4,
                CommTestConfig(
                    comm_type=comm_type,
                    ep_size=4,
                    num_experts=FIXED_NUM_EXPERTS,
                    top_k=2,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    all_num_tokens=[1, 1, 1, 1],
                ),
                id=f"{comm_type}_single_token",
            )
        )

        # Zero tokens on some ranks (DeepEPLL kernel does not support this).
        if comm_type != COMM_DEEP_EP_LL:
            params.append(
                pytest.param(
                    4,
                    CommTestConfig(
                        comm_type=comm_type,
                        ep_size=4,
                        num_experts=FIXED_NUM_EXPERTS,
                        top_k=2,
                        hidden_size=DEFAULT_HIDDEN_SIZE,
                        all_num_tokens=[32, 0, 16, 0],
                    ),
                    id=f"{comm_type}_zero_tokens",
                )
            )

    return params


# ============================================================================
# Pytest Fixtures & Test Runner
# ============================================================================


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0x1234)
    tllm.logger.set_level("error")


def _run_full_test(mpi_pool_executor, config: CommTestConfig):
    """Run dispatch -> verify dispatch -> simple_moe -> combine -> verify combine."""
    skip_reason = check_platform_support(config.comm_type)
    if skip_reason:
        pytest.skip(skip_reason)

    skip_reason = check_feasibility(config.comm_type, config)
    if skip_reason:
        pytest.skip(skip_reason)

    if config.ep_size > torch.cuda.device_count():
        pytest.skip(f"Need {config.ep_size} GPUs but only {torch.cuda.device_count()} available")

    results = mpi_pool_executor.map(
        _worker_full_pipeline,
        *zip(*[(config,)] * config.ep_size),
    )
    all_results = list(results)

    verify_dispatch_results(all_results, config)

    # DeepEPLL: combine applies real scales internally, which introduces
    # additional FP rounding vs our reference.
    if config.comm_type == COMM_DEEP_EP_LL:
        verify_combine_results(all_results, config, rtol=0.1, atol=0.5)
    else:
        verify_combine_results(all_results, config, rtol=0.05, atol=0.1)


# ============================================================================
# Test Class
# ============================================================================


class TestMoEComm:
    """Full-pipeline tests for all MoE Communication types.

    Each test: dispatch -> verify dispatch -> simple_moe -> combine
    -> verify combine.  Dispatch errors are caught early with clear
    diagnostics before combine verification runs.
    """

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,config",
        _make_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm(self, mpi_pool_executor, config: CommTestConfig):
        """Verify full dispatch -> compute -> combine pipeline."""
        _run_full_test(mpi_pool_executor, config)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,config",
        _make_boundary_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_boundary(self, mpi_pool_executor, config: CommTestConfig):
        """Test full pipeline with boundary / edge-case parameters."""
        _run_full_test(mpi_pool_executor, config)
