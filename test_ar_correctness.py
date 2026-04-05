#!/usr/bin/env python3
"""FlashMoE single-GPU AllReduce simulation test.

Validates the in-kernel AllReduce logic by simulating 2 ranks on a single GPU.
Uses the same package shim as test_correctness.py to avoid heavy dependencies.

Approach:
  1. Create two staging buffers on the same GPU (simulating 2 EP ranks).
  2. Pre-fill staging[1] (remote rank) with constant bf16 1.0.
  3. Pre-set rank_ready_flag[1] = 1 (remote rank "ready").
  4. Run the fused kernel with enable_ar=True, ar_num_ranks=2, ar_local_rank=0.
  5. FC2 writes to staging[0], AR reduces staging[0]+staging[1] → ar_output.
  6. Validate: ar_output ≈ FC2_single_rank_output + 1.0.

The FC2 single-rank output is ~0.047 per element (from analytical computation
with uniform FP4=0.5, sf=1/128 data). So ar_output should be ~1.047.

No C++ bindings or multi-GPU setup needed -- uses the package shim.

Usage: python test_ar_correctness.py
"""

import math
import os
import sys
import time
import types

os.environ["TRTLLM_ENABLE_PDL"] = "0"
os.environ["TLLM_DISABLE_MPI"] = "1"

import torch  # noqa: E402

# Ensure repo root is on PYTHONPATH for relative imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ============================================================
# Bypass tensorrt_llm/__init__.py (same shim as test_correctness.py)
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


_setup_package_shim()


# ============================================================
# Pure Python moe_sort (same as test_correctness.py)
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


# ============================================================
# Analytical expected value (same as test_correctness.py)
# ============================================================
def compute_analytical_expected(hidden_size, intermediate_size, sf_vec_size):
    """Compute analytical expected FC2 output per element for uniform data."""
    fp8_sf = 0.0078125  # FP8 E4M3 exact 1/128

    num_blocks_k1 = hidden_size // sf_vec_size
    block_inner_k1 = sf_vec_size * 0.5 * 0.5  # = 4.0
    fc1_val = num_blocks_k1 * block_inner_k1 * fp8_sf * fp8_sf

    gate = fc1_val
    value = fc1_val
    sigmoid_gate = 1.0 / (1.0 + math.exp(-gate))
    swiglu_val = gate * sigmoid_gate * value

    fp4_requant = 3.0
    fp8_scale_fc1c = 0.001953125  # FP8 E4M3 subnormal (e=0, m=1)

    num_blocks_k2 = intermediate_size // sf_vec_size
    block_inner_k2 = sf_vec_size * fp4_requant * 0.5  # = 24.0
    fc2_val = num_blocks_k2 * block_inner_k2 * fp8_scale_fc1c * fp8_sf

    return fc2_val, fc1_val, swiglu_val


# ============================================================
# Main test
# ============================================================
def run_ar_correctness_test():
    """Run fused kernel with simulated 2-rank AllReduce and validate output."""
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

    device = "cuda:0"
    torch.cuda.set_device(0)

    # Model config
    hidden_size = 7168
    intermediate_size = 2048
    num_experts = 128
    top_k = 8
    sf_vec_size = 16
    tile_size = 128
    experts_per_rank = num_experts
    max_tokens = 64

    # AR config
    ar_num_ranks = 2
    ar_local_rank = cutlass.Int32(0)
    # Constant value for remote rank's staging buffer (exact in bf16)
    remote_staging_fill_value = 1.0

    sys.stderr.write(
        f"[AR Test] Config: hidden={hidden_size}, intermediate={intermediate_size}, "
        f"experts={num_experts}, top_k={top_k}, tokens={max_tokens}, "
        f"ar_num_ranks={ar_num_ranks}\n"
    )
    sys.stderr.flush()

    torch.manual_seed(42)

    # --- Create uniform NVFP4 data (same as test_correctness.py) ---
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

    # Input: uniform NVFP4
    gathered_a = torch.full((max_tokens, hidden_size // 2), 0x11, dtype=torch.uint8, device=device)
    sfa_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
    gathered_sfa = sfa_val.expand(max_tokens * hidden_size // sf_vec_size).contiguous().to(device)

    # --- Routing ---
    router_logits = torch.randn(max_tokens, num_experts, dtype=torch.float32, device=device)
    topk_vals, topk_indices = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    token_selected_experts = topk_indices.to(torch.int32)
    token_final_scales = topk_weights.to(torch.float32)

    # --- moe_sort ---
    sys.stderr.write("[AR Test] Running moe_sort...\n")
    sys.stderr.flush()
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
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
    sys.stderr.write(f"[AR Test] moe_sort: tiles={num_non_exiting_tiles.item()}, m_padded={m}\n")
    sys.stderr.flush()

    orig_m = gathered_a.shape[0]
    k1 = hidden_size
    fc1_n = 2 * intermediate_size
    k2 = intermediate_size
    fc2_n = hidden_size

    # --- Allocate FC1 output buffer ---
    fc1_c = torch.empty(m, intermediate_size // 2, dtype=torch.uint8, device=device)
    fc1_c_sf = torch.empty(
        m * intermediate_size // sf_vec_size, dtype=torch.float8_e4m3fn, device=device
    )

    # --- AllReduce buffers ---
    # Staging buffer for local rank (rank 0): FC2 writes here via scatter-add
    staging_0 = torch.zeros(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    # Staging buffer for remote rank (rank 1): pre-filled with constant 1.0
    staging_1 = torch.full(
        (max_tokens, hidden_size), remote_staging_fill_value, dtype=torch.bfloat16, device=device
    )

    # IPC pointer array: Int64 device pointers to each rank's staging
    staging_ipc_ptrs = torch.tensor(
        [staging_0.data_ptr(), staging_1.data_ptr()],
        dtype=torch.int64,
        device=device,
    )

    # AR output buffer
    ar_output = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # CTA exit counter (zeroed before launch)
    cta_exit_counter = torch.zeros(1, dtype=torch.int32, device=device)

    # Rank ready flags: Int32 array, one per rank
    # rank 0's flag = 0 (will be set by kernel)
    # rank 1's flag = 1 (pre-set, simulating remote rank is ready)
    rank_ready_flags = torch.tensor([0, 1], dtype=torch.int32, device=device)
    # IPC pointers to individual flags (4 bytes apart)
    flag_ipc_ptrs = torch.tensor(
        [rank_ready_flags.data_ptr() + i * 4 for i in range(ar_num_ranks)],
        dtype=torch.int64,
        device=device,
    )

    # FC1 done counter
    fc1_done_counter = torch.zeros(8, dtype=torch.int32, device=device)

    # --- Build CuTe pointers ---
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
    # FC2 A = FC1 output
    fc2_a_ptr = mk_ptr(cutlass.Float4E2M1FN, fc1_c.data_ptr())
    fc2_b_ptr = mk_ptr(cutlass.Float4E2M1FN, w2_weight.data_ptr())
    # FC2 output goes to staging[0] (local rank's staging buffer)
    fc2_out_ptr = mk_ptr(cutlass.BFloat16, staging_0.data_ptr())
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

    # AR pointers
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

    # --- Create and compile kernel ---
    sys.stderr.write(
        f"[AR Test] Compiling fused kernel (enable_ar=True, ar_num_ranks={ar_num_ranks})...\n"
    )
    sys.stderr.flush()
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
    sys.stderr.write(f"[AR Test] Compilation done in {t1 - t0:.1f}s\n")
    sys.stderr.flush()

    # --- Launch kernel ---
    sys.stderr.write("[AR Test] Launching fused kernel with AllReduce...\n")
    sys.stderr.flush()

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

    # --- Wait with timeout ---
    sync_event = torch.cuda.Event()
    sync_event.record()
    deadline = time.monotonic() + 120
    while not sync_event.query():
        if time.monotonic() > deadline:
            sys.stderr.write("[AR Test] DEADLOCK: kernel did not complete within 120s\n")
            sys.stderr.flush()
            os._exit(42)
        time.sleep(0.1)

    sys.stderr.write("[AR Test] Kernel completed successfully\n")
    sys.stderr.flush()

    # ================================================================
    # Validation
    # ================================================================
    output_f32 = ar_output.float()
    abs_max = output_f32.abs().max().item()
    has_nan = torch.isnan(output_f32).any().item()
    has_inf = torch.isinf(output_f32).any().item()
    nonzero_count = (output_f32.abs() > 1e-8).sum().item()

    sys.stderr.write(
        f"[AR Test] AR output: shape={ar_output.shape}, "
        f"abs_max={abs_max:.6f}, nonzero={nonzero_count}/{output_f32.numel()}, "
        f"nan={has_nan}, inf={has_inf}\n"
    )
    sys.stderr.flush()

    # Also check staging[0] (FC2 output before AR)
    staging_f32 = staging_0.float()
    staging_mean = staging_f32.mean().item()
    staging_abs_max = staging_f32.abs().max().item()
    sys.stderr.write(
        f"[AR Test] Staging[0] (FC2 output): mean={staging_mean:.6f}, "
        f"abs_max={staging_abs_max:.6f}\n"
    )
    sys.stderr.flush()

    # Check 1: No NaN or Inf
    if has_nan or has_inf:
        sys.stderr.write("[FAIL] AR output contains NaN or Inf\n")
        sys.stderr.flush()
        return False

    # Check 2: Output is non-zero
    if abs_max < 1e-6:
        sys.stderr.write("[FAIL] AR output is all zeros\n")
        sys.stderr.flush()
        return False

    # Check 3: Analytical validation
    # AR output = staging[0] + staging[1]
    #           = FC2_output + remote_staging_fill_value
    # FC2_output ≈ 0.047 per element (analytical), staging[1] = 1.0
    # Expected mean ≈ 0.047 + 1.0 = 1.047
    expected_fc2_val, fc1_val, swiglu_val = compute_analytical_expected(
        hidden_size, intermediate_size, sf_vec_size
    )
    expected_ar_val = expected_fc2_val + remote_staging_fill_value

    output_mean = output_f32.mean().item()
    output_std = output_f32.std().item()
    cv = output_std / abs(output_mean) if abs(output_mean) > 1e-10 else float("inf")

    sys.stderr.write(
        f"[AR Test] Analytical: FC2_expected={expected_fc2_val:.6f}, "
        f"remote_fill={remote_staging_fill_value}, "
        f"AR_expected={expected_ar_val:.6f}\n"
    )
    sys.stderr.write(
        f"[AR Test] AR output: mean={output_mean:.6f}, std={output_std:.6f}, cv={cv:.6f}\n"
    )
    sys.stderr.flush()

    ratio = output_mean / expected_ar_val if abs(expected_ar_val) > 1e-10 else float("inf")
    sys.stderr.write(
        f"[AR Test] Magnitude: ar_mean={output_mean:.6f}, "
        f"expected={expected_ar_val:.6f}, ratio={ratio:.4f}\n"
    )
    sys.stderr.flush()

    # Check 4: Uniformity (all elements should be approximately the same)
    if cv > 0.05:
        sys.stderr.write(f"[FAIL] AR output not uniform: cv={cv:.4f} > 0.05\n")
        sys.stderr.flush()
        return False

    # Check 5: Magnitude matches expected (allow 30% tolerance for FP4/FP8 quantization)
    if not (0.7 < ratio < 1.3):
        sys.stderr.write(f"[FAIL] AR magnitude mismatch: ratio={ratio:.4f} not in [0.7, 1.3]\n")
        sys.stderr.flush()
        return False

    # Check 6: Verify AR actually added the remote staging data
    # If AR were broken (not reading rank 1), output would be ~0.047, not ~1.047
    # Check that output is significantly different from FC2-only output (~0.047)
    if output_mean < 0.5:
        sys.stderr.write(
            f"[FAIL] AR output mean={output_mean:.4f} too low — "
            f"AllReduce may not be reading remote staging\n"
        )
        sys.stderr.flush()
        return False

    # Check 7: Cross-check staging[0] (FC2 output) is reasonable
    fc2_ratio = staging_mean / expected_fc2_val if abs(expected_fc2_val) > 1e-10 else float("inf")
    sys.stderr.write(
        f"[AR Test] Staging cross-check: staging_mean={staging_mean:.6f}, "
        f"expected_fc2={expected_fc2_val:.6f}, ratio={fc2_ratio:.4f}\n"
    )
    sys.stderr.flush()

    if not (0.5 < fc2_ratio < 2.0):
        sys.stderr.write(f"[WARN] Staging[0] doesn't match FC2 expected: ratio={fc2_ratio:.4f}\n")
        # Not a hard failure — the AR test is the main validation

    sys.stderr.write("[PASSED] AllReduce correctness test passed\n")
    sys.stderr.flush()
    return True


if __name__ == "__main__":
    success = run_ar_correctness_test()
    sys.exit(0 if success else 1)
