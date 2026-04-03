#!/usr/bin/env python3
"""FlashMoE correctness test: compare fused vs decomposed output.

Uses the FlashMoE module to run both paths with the same inputs and
compares the FC2 output.

Prerequisites:
  - Container with tensorrt_llm installed (provides C++ bindings, moe_sort, etc.)
  - Our new files copied into container's site-packages (done by slurm script)

Usage: torchrun --nproc_per_node=2 test_correctness.py
"""

import os
import sys
import time

# Must be set BEFORE any import of mpi4py.  Prevents MPI_Init_thread
# which fails inside enroot containers launched with --mpi=none.
os.environ["MPI4PY_RC_INITIALIZE"] = "0"
os.environ["MPI4PY_RC_FINALIZE"] = "0"
os.environ["TRTLLM_ENABLE_PDL"] = "0"
os.environ["TLLM_DISABLE_MPI"] = "1"

import torch
import torch.distributed as dist


def run_test():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    sys.stderr.write(f"[Rank {rank}] Starting correctness test\n")
    sys.stderr.flush()

    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
    )

    hidden_size = 7168
    intermediate_size = 2048
    num_experts = 128
    top_k = 8
    experts_per_rank = num_experts // world_size
    max_tokens = 64
    sf_vec_size = 16

    from tensorrt_llm._torch.modules.fused_moe.flashmoe import FlashMoE
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod

    # Use fixed seed for reproducibility across runs (but different per rank)
    torch.manual_seed(42 + rank)

    # Non-zero NVFP4 weights: fill with 0x11 (two FP4 values of 0.5 each).
    # This gives non-trivial GEMM results.
    w3w1 = torch.full(
        (experts_per_rank, intermediate_size * 2, hidden_size // 2),
        0x11,
        dtype=torch.uint8,
        device=dev,
    )
    w2 = torch.full(
        (experts_per_rank, hidden_size, intermediate_size // 2),
        0x11,
        dtype=torch.uint8,
        device=dev,
    )
    fc1_ws = torch.ones(
        experts_per_rank,
        intermediate_size * 2,
        hidden_size // sf_vec_size,
        dtype=torch.float8_e4m3fn,
        device=dev,
    )
    fc2_ws = torch.ones(
        experts_per_rank,
        hidden_size,
        intermediate_size // sf_vec_size,
        dtype=torch.float8_e4m3fn,
        device=dev,
    )
    fc1_alpha = torch.ones(experts_per_rank, dtype=torch.float32, device=dev)
    fc2_alpha = torch.ones(experts_per_rank, dtype=torch.float32, device=dev)
    fc31_input_scale = torch.ones(1, dtype=torch.float32, device=dev)
    fc2_input_scale = torch.ones(1, dtype=torch.float32, device=dev)

    x = torch.randn(max_tokens, hidden_size, dtype=torch.bfloat16, device=dev)
    router_logits = torch.randn(max_tokens, num_experts, dtype=torch.float32, device=dev)
    routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

    def _setup_moe(use_fused, phase_mode=0):
        moe = FlashMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            mapping=mapping,
            max_num_tokens=max_tokens,
            use_ipc=False,
            use_fused_kernel=use_fused,
        )
        moe.phase_mode = phase_mode
        moe.w3w1_weight = w3w1
        moe.w2_weight = w2
        moe.fc1_weight_scale = fc1_ws
        moe.fc2_weight_scale = fc2_ws
        moe.fc1_alpha = fc1_alpha
        moe.fc2_alpha = fc2_alpha
        moe.fc31_input_scale = fc31_input_scale
        moe.fc2_input_scale = fc2_input_scale
        return moe

    # --- Run decomposed path ---
    sys.stderr.write(f"[Rank {rank}] Running decomposed path...\n")
    sys.stderr.flush()
    t0 = time.monotonic()
    moe_decomposed = _setup_moe(use_fused=False)
    try:
        output_decomposed = moe_decomposed(x, router_logits, routing_method)
        t1 = time.monotonic()
        sys.stderr.write(
            f"[Rank {rank}] Decomposed: shape={output_decomposed.shape}, "
            f"abs_max={output_decomposed.abs().max().item():.4f}, "
            f"time={t1 - t0:.1f}s\n"
        )
        sys.stderr.flush()
    except Exception as e:
        import traceback

        sys.stderr.write(f"[Rank {rank}] Decomposed FAILED: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(1)

    # --- Run fused path ---
    sys.stderr.write(f"[Rank {rank}] Running fused path (phase_mode=0)...\n")
    sys.stderr.flush()
    t0 = time.monotonic()
    moe_fused = _setup_moe(use_fused=True, phase_mode=0)
    try:
        output_fused = moe_fused(x, router_logits, routing_method)
        t1 = time.monotonic()
        sys.stderr.write(
            f"[Rank {rank}] Fused: shape={output_fused.shape}, "
            f"abs_max={output_fused.abs().max().item():.4f}, "
            f"time={t1 - t0:.1f}s\n"
        )
        sys.stderr.flush()
    except Exception as e:
        import traceback

        sys.stderr.write(f"[Rank {rank}] Fused FAILED: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(1)

    # --- Compare ---
    abs_diff = (output_fused - output_decomposed).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative error: denominator is max of decomposed output abs value
    denom = output_decomposed.abs().max().item()
    rel_max = max_diff / denom if denom > 1e-6 else max_diff

    sys.stderr.write(
        f"[Rank {rank}] Comparison:\n"
        f"  abs_max_diff={max_diff:.6f}\n"
        f"  abs_mean_diff={mean_diff:.6f}\n"
        f"  rel_max_diff={rel_max:.6f}\n"
        f"  decomposed abs_max={denom:.4f}\n"
    )
    sys.stderr.flush()

    # Tolerance for NVFP4 numerical precision
    # NVFP4 has very low precision (4 bits), so larger tolerance is expected
    tol_max = 1.0
    tol_mean = 0.1
    passed = max_diff < tol_max and mean_diff < tol_mean

    if passed:
        sys.stderr.write(f"[Rank {rank}] CORRECTNESS TEST PASSED\n")
    else:
        sys.stderr.write(
            f"[Rank {rank}] CORRECTNESS TEST FAILED "
            f"(max_diff={max_diff:.6f} > {tol_max} or "
            f"mean_diff={mean_diff:.6f} > {tol_mean})\n"
        )
    sys.stderr.flush()

    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    run_test()
