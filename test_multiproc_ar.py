#!/usr/bin/env python3
"""FlashMoE multi-process IPC AllReduce correctness test.

Validates real multi-process CUDA IPC AllReduce by running independent
kernel instances on each GPU, communicating via IPC-mapped staging buffers.

Each rank:
  1. Allocates staging, output, flags on its own GPU.
  2. Shares staging and flag tensors via torch.multiprocessing Queue (CUDA IPC).
  3. Builds IPC pointer arrays from received tensors' data_ptr().
  4. Compiles and runs the fused kernel independently.
  5. Kernel's AR warps signal ready, wait for all ranks, then reduce.
  6. Validates: output = sum of all ranks' FC2 outputs.

Since all ranks use identical data and weights (same seed), each rank's FC2
output is identical (~0.047 per element). Expected AR result = N * 0.047.

No C++ bindings, MPI, or TRT-LLM build needed -- uses package shim + CUDA IPC.

Usage: python test_multiproc_ar.py [--num-gpus N]
"""

import argparse
import math
import os
import sys
import time
import types

os.environ["TRTLLM_ENABLE_PDL"] = "0"
os.environ["TLLM_DISABLE_MPI"] = "1"

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

# Ensure repo root is on PYTHONPATH for relative imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ============================================================
# Bypass tensorrt_llm/__init__.py
# ============================================================
def _setup_package_shim():
    """Pre-populate sys.modules with lightweight stubs."""
    pkg_chain = [
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.cute_dsl_kernels",
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell",
    ]
    for pkg_name in pkg_chain:
        if pkg_name in sys.modules:
            continue
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [os.path.join(REPO_ROOT, pkg_name.replace(".", "/"))]
        mod.__package__ = pkg_name
        sys.modules[pkg_name] = mod


# ============================================================
# Pure Python moe_sort
# ============================================================
def python_moe_sort(
    token_selected_experts,
    token_final_scales,
    num_experts,
    top_k,
    local_expert_offset,
    local_num_experts,
    tile_tokens_dim,
):
    """Pure Python moe_sort implementation."""
    num_tokens = token_selected_experts.shape[0]
    device = token_selected_experts.device

    expert_to_expanded = {
        e: [] for e in range(local_expert_offset, local_expert_offset + local_num_experts)
    }
    for t in range(num_tokens):
        for k in range(top_k):
            expert_id = token_selected_experts[t, k].item()
            if local_expert_offset <= expert_id < local_expert_offset + local_num_experts:
                expanded_idx = t * top_k + k
                expert_to_expanded[expert_id].append(expanded_idx)

    tile_idx_to_expert_list = []
    tile_idx_to_mn_limit_list = []
    permuted_idx_list = []

    for expert_id in range(local_expert_offset, local_expert_offset + local_num_experts):
        expanded_indices = expert_to_expanded[expert_id]
        n_tokens = len(expanded_indices)
        if n_tokens == 0:
            continue
        n_padded = math.ceil(n_tokens / tile_tokens_dim) * tile_tokens_dim
        num_tiles = n_padded // tile_tokens_dim
        padded_indices = expanded_indices + [expanded_indices[-1]] * (n_padded - n_tokens)
        permuted_idx_list.extend(padded_indices)
        for tile_i in range(num_tiles):
            tile_idx_to_expert_list.append(expert_id - local_expert_offset)
            mn_limit = (
                len(permuted_idx_list) - n_padded + min((tile_i + 1) * tile_tokens_dim, n_tokens)
            )
            tile_idx_to_mn_limit_list.append(mn_limit)

    num_non_exiting_tiles_val = len(tile_idx_to_expert_list)
    total_num_padded_tokens_val = len(permuted_idx_list)

    expanded_idx_to_permuted = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=device)
    for perm_idx, exp_idx in enumerate(permuted_idx_list):
        t = exp_idx // top_k
        k = exp_idx % top_k
        if expanded_idx_to_permuted[t, k].item() == -1:
            expanded_idx_to_permuted[t, k] = perm_idx

    max_num_tiles = max(num_non_exiting_tiles_val, 1)
    max_num_permuted_tokens = max(total_num_padded_tokens_val, 1)

    tile_idx_to_expert_idx = torch.tensor(
        tile_idx_to_expert_list + [0] * (max_num_tiles - len(tile_idx_to_expert_list)),
        dtype=torch.int32,
        device=device,
    )
    tile_idx_to_mn_limit = torch.tensor(
        tile_idx_to_mn_limit_list + [0] * (max_num_tiles - len(tile_idx_to_mn_limit_list)),
        dtype=torch.int32,
        device=device,
    )
    permuted_idx_to_expanded_idx = torch.tensor(
        permuted_idx_list + [0] * (max_num_permuted_tokens - len(permuted_idx_list)),
        dtype=torch.int32,
        device=device,
    )
    total_num_padded_tokens = torch.tensor(
        [total_num_padded_tokens_val], dtype=torch.int32, device=device
    )
    num_non_exiting_tiles = torch.tensor(
        [num_non_exiting_tiles_val], dtype=torch.int32, device=device
    )

    return (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    )


def compute_analytical_expected(hidden_size, intermediate_size, sf_vec_size):
    """Compute analytical expected FC2 output per element for uniform data."""
    fp8_sf = 0.0078125  # FP8 E4M3 exact 1/128
    num_blocks_k1 = hidden_size // sf_vec_size
    block_inner_k1 = sf_vec_size * 0.5 * 0.5
    fc1_val = num_blocks_k1 * block_inner_k1 * fp8_sf * fp8_sf
    gate = fc1_val
    value = fc1_val
    sigmoid_gate = 1.0 / (1.0 + math.exp(-gate))
    swiglu_val = gate * sigmoid_gate * value  # noqa: F841
    fp4_requant = 3.0
    fp8_scale_fc1c = 0.001953125
    num_blocks_k2 = intermediate_size // sf_vec_size
    block_inner_k2 = sf_vec_size * fp4_requant * 0.5
    fc2_val = num_blocks_k2 * block_inner_k2 * fp8_scale_fc1c * fp8_sf
    return fc2_val


# ============================================================
# Worker function for each rank
# ============================================================
def _worker(
    rank: int,
    world_size: int,
    staging_queues: list,
    flag_queues: list,
    barrier: mp.Barrier,
    result_queue: mp.Queue,
):
    """Multi-process worker: compile kernel, exchange IPC, run AR, validate."""
    try:
        _setup_package_shim()

        import cutlass
        import cutlass.cute as cute

        try:
            from cuda.bindings import driver as cuda_driver
        except ImportError:
            from cuda import cuda as cuda_driver

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

        def log(msg):
            sys.stderr.write(f"[Rank {rank}] {msg}\n")
            sys.stderr.flush()

        log(f"Worker started on GPU {rank}")

        # ---- Model config (identical across ranks) ----
        hidden_size = 7168
        intermediate_size = 2048
        num_experts = 128
        top_k = 8
        sf_vec_size = 16
        tile_size = 128
        experts_per_rank = num_experts
        max_tokens = 64
        ar_num_ranks = world_size

        # ---- Create identical data on each rank (same seed) ----
        torch.manual_seed(42)

        w3w1_weight = torch.full(
            (experts_per_rank, intermediate_size * 2, hidden_size // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        )
        w2_weight = torch.full(
            (experts_per_rank, hidden_size, intermediate_size // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        )
        fc1_ws_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
        fc1_weight_scale = (
            fc1_ws_val.expand(experts_per_rank, intermediate_size * 2, hidden_size // sf_vec_size)
            .contiguous()
            .to(device)
        )
        fc2_ws_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
        fc2_weight_scale = (
            fc2_ws_val.expand(experts_per_rank, hidden_size, intermediate_size // sf_vec_size)
            .contiguous()
            .to(device)
        )
        fc1_alpha = torch.ones(experts_per_rank, dtype=torch.float32, device=device)
        fc2_alpha = torch.ones(experts_per_rank, dtype=torch.float32, device=device)
        fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)
        gathered_a = torch.full(
            (max_tokens, hidden_size // 2), 0x11, dtype=torch.uint8, device=device
        )
        sfa_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
        gathered_sfa = (
            sfa_val.expand(max_tokens * hidden_size // sf_vec_size).contiguous().to(device)
        )

        # ---- Routing (identical across ranks) ----
        router_logits = torch.randn(max_tokens, num_experts, dtype=torch.float32, device=device)
        topk_vals, topk_indices = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        token_selected_experts = topk_indices.to(torch.int32)
        token_final_scales = topk_weights.to(torch.float32)

        # ---- moe_sort ----
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            _,
            permuted_idx_to_expanded_idx,
            _,
            num_non_exiting_tiles,
        ) = python_moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=0,
            local_num_experts=experts_per_rank,
            tile_tokens_dim=tile_size,
        )
        m = permuted_idx_to_expanded_idx.shape[0]
        log(f"moe_sort: tiles={num_non_exiting_tiles.item()}, m_padded={m}")

        orig_m = gathered_a.shape[0]
        k1 = hidden_size
        fc1_n = 2 * intermediate_size
        k2 = intermediate_size
        fc2_n = hidden_size

        # ---- Buffers ----
        fc1_c = torch.empty(m, intermediate_size // 2, dtype=torch.uint8, device=device)
        fc1_c_sf = torch.empty(
            m * intermediate_size // sf_vec_size, dtype=torch.float8_e4m3fn, device=device
        )

        # Staging: FC2 output goes here, AR reads from all ranks
        staging = torch.zeros(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)

        # Ready flag: kernel sets this to 1 when FC2 is done
        ready_flag = torch.zeros(1, dtype=torch.int32, device=device)

        # AR output
        ar_output = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)

        # CTA exit counter
        cta_exit_counter = torch.zeros(1, dtype=torch.int32, device=device)

        # FC1 done counter
        fc1_done_counter = torch.zeros(8, dtype=torch.int32, device=device)

        # ---- Exchange staging + flag tensors via CUDA IPC ----
        # Put this rank's tensors into all queues
        for r in range(world_size):
            staging_queues[r].put((rank, staging))
            flag_queues[r].put((rank, ready_flag))

        barrier.wait()  # Ensure all ranks have published

        # Collect from own queue
        remote_staging = {}
        remote_flags = {}
        for _ in range(world_size):
            r, s = staging_queues[rank].get()
            remote_staging[r] = s
        for _ in range(world_size):
            r, f = flag_queues[rank].get()
            remote_flags[r] = f

        barrier.wait()  # Ensure all receives done before kernel launch

        # Build IPC pointer arrays (on local device)
        staging_ptrs = [remote_staging[r].data_ptr() for r in range(world_size)]
        flag_ptrs = [remote_flags[r].data_ptr() for r in range(world_size)]

        staging_ipc_ptrs = torch.tensor(staging_ptrs, dtype=torch.int64, device=device)
        flag_ipc_ptrs = torch.tensor(flag_ptrs, dtype=torch.int64, device=device)

        log(f"IPC staging ptrs: {[f'0x{p:x}' for p in staging_ptrs]}")
        log(f"IPC flag ptrs: {[f'0x{p:x}' for p in flag_ptrs]}")

        # ---- Build CuTe pointers ----
        def mk_ptr(dtype, data_ptr, align=32):
            return make_ptr(dtype, data_ptr, cute.AddressSpace.gmem, assumed_align=align)

        fc1_a_ptr = mk_ptr(cutlass.Float4E2M1FN, gathered_a.data_ptr())
        fc1_b_ptr = mk_ptr(cutlass.Float4E2M1FN, w3w1_weight.data_ptr())
        fc1_c_ptr = mk_ptr(cutlass.Float4E2M1FN, fc1_c.data_ptr())
        fc1_sfa_ptr = mk_ptr(cutlass.Float8E4M3FN, gathered_sfa.data_ptr(), align=16)
        fc1_sfb_ptr = mk_ptr(cutlass.Float8E4M3FN, fc1_weight_scale.data_ptr(), align=16)
        fc1_sfc_ptr = mk_ptr(cutlass.Float8E4M3FN, fc1_c_sf.data_ptr(), align=16)
        fc1_norm_const_ptr = make_ptr(
            cutlass.Float32, fc2_input_scale.data_ptr(), cute.AddressSpace.gmem
        )
        fc1_alpha_ptr = make_ptr(cutlass.Float32, fc1_alpha.data_ptr(), cute.AddressSpace.gmem)
        fc2_a_ptr = mk_ptr(cutlass.Float4E2M1FN, fc1_c.data_ptr())
        fc2_b_ptr = mk_ptr(cutlass.Float4E2M1FN, w2_weight.data_ptr())
        fc2_out_ptr = mk_ptr(cutlass.BFloat16, staging.data_ptr())
        fc2_sfa_ptr = mk_ptr(cutlass.Float8E4M3FN, fc1_c_sf.data_ptr(), align=16)
        fc2_sfb_ptr = mk_ptr(cutlass.Float8E4M3FN, fc2_weight_scale.data_ptr(), align=16)
        fc2_alpha_ptr = make_ptr(cutlass.Float32, fc2_alpha.data_ptr(), cute.AddressSpace.gmem)
        tile_idx_to_expert_idx_ptr = make_ptr(
            cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem
        )
        tile_idx_to_mn_limit_ptr = make_ptr(
            cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
        )
        token_id_mapping_ptr = make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        )
        num_non_exiting_tiles_ptr = make_ptr(
            cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
        )
        permuted_idx_ptr = make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        )
        token_final_scales_ptr = make_ptr(
            cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem
        )
        fc1_done_counter_ptr = make_ptr(
            cutlass.Int32, fc1_done_counter.data_ptr(), cute.AddressSpace.gmem
        )
        ar_staging_ipc_ptrs_ptr = make_ptr(
            cutlass.Int64, staging_ipc_ptrs.data_ptr(), cute.AddressSpace.gmem
        )
        ar_output_ptr = mk_ptr(cutlass.BFloat16, ar_output.data_ptr())
        ar_cta_exit_counter_ptr = make_ptr(
            cutlass.Int32, cta_exit_counter.data_ptr(), cute.AddressSpace.gmem
        )
        ar_rank_ready_flag_ptrs_ptr = make_ptr(
            cutlass.Int64, flag_ipc_ptrs.data_ptr(), cute.AddressSpace.gmem
        )
        ar_local_rank = cutlass.Int32(rank)

        # ---- Compile kernel ----
        log("Compiling fused kernel...")
        t0 = time.monotonic()
        kernel = FlashMoeFusedKernel(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=(tile_size, 128),
            cluster_shape_mn=(1, 1),
            vectorized_f32=True,
            topk=top_k,
            raster_along_m=False,
            enable_ar=True,
            ar_num_ranks=ar_num_ranks,
            phase_mode=0,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(1)
        torch_stream = torch.cuda.current_stream()
        stream = cuda_driver.CUstream(torch_stream.cuda_stream)

        compiled = cute.compile(
            kernel.wrapper,
            fc1_a_ptr,
            fc1_b_ptr,
            fc1_c_ptr,
            fc1_sfa_ptr,
            fc1_sfb_ptr,
            fc1_sfc_ptr,
            fc1_norm_const_ptr,
            fc1_alpha_ptr,
            fc2_a_ptr,
            fc2_b_ptr,
            fc2_out_ptr,
            fc2_sfa_ptr,
            fc2_sfb_ptr,
            fc2_alpha_ptr,
            tile_idx_to_expert_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            token_id_mapping_ptr,
            num_non_exiting_tiles_ptr,
            permuted_idx_ptr,
            token_final_scales_ptr,
            orig_m,
            m,
            fc1_n,
            fc2_n,
            k1,
            k2,
            experts_per_rank,
            max_tokens,
            top_k,
            fc1_done_counter_ptr=fc1_done_counter_ptr,
            ar_staging_ipc_ptrs_ptr=ar_staging_ipc_ptrs_ptr,
            ar_output_ptr=ar_output_ptr,
            ar_cta_exit_counter_ptr=ar_cta_exit_counter_ptr,
            ar_rank_ready_flag_ptrs_ptr=ar_rank_ready_flag_ptrs_ptr,
            ar_local_rank=ar_local_rank,
            scaling_vector_size=sf_vec_size,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        t1 = time.monotonic()
        log(f"Compilation done in {t1 - t0:.1f}s")

        # ---- All ranks sync before kernel launch ----
        barrier.wait()
        log("Launching kernel...")

        compiled(
            fc1_a_ptr,
            fc1_b_ptr,
            fc1_c_ptr,
            fc1_sfa_ptr,
            fc1_sfb_ptr,
            fc1_sfc_ptr,
            fc1_norm_const_ptr,
            fc1_alpha_ptr,
            fc2_a_ptr,
            fc2_b_ptr,
            fc2_out_ptr,
            fc2_sfa_ptr,
            fc2_sfb_ptr,
            fc2_alpha_ptr,
            tile_idx_to_expert_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            token_id_mapping_ptr,
            num_non_exiting_tiles_ptr,
            permuted_idx_ptr,
            token_final_scales_ptr,
            orig_m,
            m,
            fc1_n,
            fc2_n,
            k1,
            k2,
            experts_per_rank,
            max_tokens,
            top_k,
            fc1_done_counter_ptr=fc1_done_counter_ptr,
            ar_staging_ipc_ptrs_ptr=ar_staging_ipc_ptrs_ptr,
            ar_output_ptr=ar_output_ptr,
            ar_cta_exit_counter_ptr=ar_cta_exit_counter_ptr,
            ar_rank_ready_flag_ptrs_ptr=ar_rank_ready_flag_ptrs_ptr,
            ar_local_rank=ar_local_rank,
            stream=stream,
        )

        # ---- Wait with timeout ----
        sync_event = torch.cuda.Event()
        sync_event.record()
        deadline = time.monotonic() + 120
        while not sync_event.query():
            if time.monotonic() > deadline:
                log("DEADLOCK: kernel did not complete within 120s")
                result_queue.put((rank, False, "DEADLOCK"))
                os._exit(42)
            time.sleep(0.1)
        log("Kernel completed")

        # ---- Validate ----
        output_f32 = ar_output.float()
        output_mean = output_f32.mean().item()
        has_nan = torch.isnan(output_f32).any().item()
        has_inf = torch.isinf(output_f32).any().item()
        staging_mean = staging.float().mean().item()

        # Expected: all ranks produce identical FC2 output (~0.047)
        # AR reduces: sum = world_size * 0.047
        expected_fc2 = compute_analytical_expected(hidden_size, intermediate_size, sf_vec_size)
        expected_ar = world_size * expected_fc2

        ratio = output_mean / expected_ar if abs(expected_ar) > 1e-10 else float("inf")

        log(
            f"staging_mean={staging_mean:.6f}, ar_mean={output_mean:.6f}, "
            f"expected={expected_ar:.6f}, ratio={ratio:.4f}, nan={has_nan}, inf={has_inf}"
        )

        passed = True
        msg = "OK"
        if has_nan or has_inf:
            passed, msg = False, "NaN/Inf in output"
        elif abs(output_mean) < 1e-6:
            passed, msg = False, "Output all zeros"
        elif not (0.7 < ratio < 1.3):
            passed, msg = False, f"Magnitude mismatch: ratio={ratio:.4f}"
        elif output_mean < expected_fc2 * 1.5:
            passed, msg = False, f"Output too low: {output_mean:.4f} (cross-rank read failed?)"

        if passed:
            log("PASSED")
        else:
            log(f"FAILED: {msg}")

        result_queue.put((rank, passed, msg))

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        sys.stderr.write(f"[Rank {rank}] EXCEPTION: {e}\n{tb}\n")
        sys.stderr.flush()
        result_queue.put((rank, False, str(e)))


# ============================================================
# Main
# ============================================================
def run_multiproc_ar_test(num_gpus: int = 2):
    """Launch multi-process IPC AllReduce test."""
    gpu_count = torch.cuda.device_count()
    if gpu_count < num_gpus:
        print(f"[SKIP] Need {num_gpus} GPUs, only {gpu_count} available")
        return True

    print(f"=== Multi-process IPC AR test: {num_gpus} GPUs ===")

    # Pre-create queues and barrier for cross-process communication
    staging_queues = [mp.Queue() for _ in range(num_gpus)]
    flag_queues = [mp.Queue() for _ in range(num_gpus)]
    barrier = mp.Barrier(num_gpus)
    result_queue = mp.Queue()

    # Spawn workers
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=_worker,
            args=(rank, num_gpus, staging_queues, flag_queues, barrier, result_queue),
        )
        p.start()
        processes.append(p)

    # Collect results with timeout
    results = {}
    deadline = time.monotonic() + 300  # 5min total timeout
    while len(results) < num_gpus and time.monotonic() < deadline:
        try:
            r, passed, msg = result_queue.get(timeout=5)
            results[r] = (passed, msg)
        except Exception:
            pass

    # Wait for processes
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Check results
    all_passed = True
    for rank in range(num_gpus):
        if rank not in results:
            print(f"[Rank {rank}] NO RESULT (timeout/crash)")
            all_passed = False
        else:
            passed, msg = results[rank]
            status = "PASSED" if passed else f"FAILED: {msg}"
            print(f"[Rank {rank}] {status}")
            if not passed:
                all_passed = False

    if all_passed:
        print(f"=== ALL {num_gpus} RANKS PASSED ===")
    else:
        print("=== SOME RANKS FAILED ===")
    return all_passed


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-process IPC AR test")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    args = parser.parse_args()

    success = run_multiproc_ar_test(num_gpus=args.num_gpus)
    sys.exit(0 if success else 1)
