#!/usr/bin/env python3
"""FlashMoE single-GPU correctness test.

Validates the fused FC1+FC2 kernel output by running phase_mode=0 with
uniform NVFP4 data (all FP4=0.5, all scale factors=1/128) and checking:
1. Output is non-zero, non-NaN, non-Inf.
2. All output elements are approximately the same (uniform-input symmetry).
3. Output magnitude matches analytical expected value within tolerance.

The analytical derivation (for FP4=0.5, sf=FP8(1/128)):
  FC1: 448 blocks * 4.0 inner * (1/128)^2 = 0.109375 per element
  SwiGLU: silu(0.109375) * 0.109375 ≈ 0.00631
  Requant to NVFP4: FP4(3.0) * sf ≈ 0.00586 (quantization error)
  FC2: 128 blocks * 24.0 inner * sf_a * sf_b ≈ 0.047
  Routing: softmax top_k weights sum to 1.0 → output ≈ 0.047

No C++ bindings or multi-GPU setup needed -- uses the package shim.

Usage: python test_correctness.py
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
# Bypass tensorrt_llm/__init__.py (which imports heavy deps like
# transformers, mpi4py, etc.). We only need the CuTe DSL kernel
# module which has no dependency on those.
# ============================================================
def _setup_package_shim():
    """Pre-populate sys.modules with lightweight stubs.

    Sets up the tensorrt_llm package hierarchy so that relative imports
    inside the blackwell kernel directory resolve correctly without
    executing tensorrt_llm/__init__.py.
    """
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
# Pure Python moe_sort (reused from test_kernel_direct.py)
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
# Analytical expected value computation
# ============================================================
def compute_analytical_expected(hidden_size, intermediate_size, sf_vec_size):
    """Compute analytical expected output per element for uniform data.

    All FP4 values = 0.5 (nibble 0x1), all scale factors = FP8(1/128).

    Returns:
        Tuple of (expected_value, fc1_val, swiglu_val) for diagnostics.
    """
    # FP8 E4M3 exact representation of 1/128 = 2^(-7)
    fp8_sf = 0.0078125

    # FC1 GEMM: [tokens, hidden] @ [2*intermediate, hidden]^T
    # num_blocks = hidden_size / sf_vec_size
    # Per block: sf_vec_size * FP4(0.5) * FP4(0.5) = sf_vec_size * 0.25
    # Scaled: block_inner * sf_a * sf_b
    num_blocks_k1 = hidden_size // sf_vec_size
    block_inner_k1 = sf_vec_size * 0.5 * 0.5  # = 4.0
    fc1_val = num_blocks_k1 * block_inner_k1 * fp8_sf * fp8_sf

    # SwiGLU: gate and value are identical (same weights for w3 and w1 halves)
    # silu(x) = x * sigmoid(x)
    gate = fc1_val
    value = fc1_val
    sigmoid_gate = 1.0 / (1.0 + math.exp(-gate))
    swiglu_val = gate * sigmoid_gate * value

    # NVFP4 requantization of SwiGLU output:
    # The kernel quantizes to FP4 E2M1 with FP8 E4M3 scale factor.
    # For uniform data where all elements = swiglu_val:
    #   raw_scale = swiglu_val / 6.0 (max FP4 value)
    #   FP8 scale ≈ nearest representable FP8 E4M3
    #   quantized FP4 = round(swiglu_val / fp8_scale)
    #
    # From debug output: fc1_c = 0x55 → FP4(3.0) everywhere.
    # This means fp8_scale ≈ swiglu_val / 3.0
    # Closest FP8 E4M3 subnormal to swiglu_val/3.0 ≈ 0.002102:
    #   m=1 subnormal: 2^(-6) * 1/8 = 0.001953125
    fp4_requant = 3.0
    fp8_scale_fc1c = 0.001953125  # FP8 E4M3 subnormal (e=0, m=1)

    # FC2 GEMM: [tokens, intermediate] @ [hidden, intermediate]^T
    # A = requantized FC1 output: FP4(3.0) with sf = fp8_scale_fc1c
    # B = w2 weights: FP4(0.5) with sf = fp8_sf (1/128)
    num_blocks_k2 = intermediate_size // sf_vec_size
    block_inner_k2 = sf_vec_size * fp4_requant * 0.5  # = 16 * 3.0 * 0.5 = 24.0
    fc2_val = num_blocks_k2 * block_inner_k2 * fp8_scale_fc1c * fp8_sf

    # With fc2_alpha = 1.0 and softmax routing weights summing to 1.0:
    # output per element = fc2_val
    return fc2_val, fc1_val, swiglu_val


# ============================================================
# Main test
# ============================================================
def run_correctness_test():
    """Run fused kernel and validate output against analytical expectation."""
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

    # Model config (smaller intermediate for faster test)
    hidden_size = 7168
    intermediate_size = 2048
    num_experts = 128
    top_k = 8
    sf_vec_size = 16
    tile_size = 128
    experts_per_rank = num_experts
    max_tokens = 64

    sys.stderr.write(
        f"[Correctness] Config: hidden={hidden_size}, intermediate={intermediate_size}, "
        f"experts={num_experts}, top_k={top_k}, tokens={max_tokens}\n"
    )
    sys.stderr.flush()

    torch.manual_seed(42)

    # --- Create uniform NVFP4 data ---
    # 0x11: each byte = two FP4 values of 0.5 (nibble 0001)
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

    # Scale factors: FP8 E4M3 representation of 1/128
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
    topk_weights = torch.softmax(topk_vals, dim=-1)  # sum to 1.0 per token
    token_selected_experts = topk_indices.to(torch.int32)
    token_final_scales = topk_weights.to(torch.float32)

    # --- moe_sort ---
    sys.stderr.write("[Correctness] Running moe_sort...\n")
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
    sys.stderr.write(
        f"[Correctness] moe_sort: tiles={num_non_exiting_tiles.item()}, "
        f"m_padded={m}, orig_m={max_tokens}\n"
    )
    sys.stderr.flush()

    orig_m = gathered_a.shape[0]
    k1 = hidden_size
    fc1_n = 2 * intermediate_size
    k2 = intermediate_size
    fc2_n = hidden_size

    # --- Allocate buffers ---
    fc1_c = torch.empty(m, intermediate_size // 2, dtype=torch.uint8, device=device)
    fc1_c_sf = torch.empty(
        m * intermediate_size // sf_vec_size, dtype=torch.float8_e4m3fn, device=device
    )
    output = torch.zeros(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)
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
    fc2_a_ptr = mk_ptr(cutlass.Float4E2M1FN, fc1_c.data_ptr())
    fc2_b_ptr = mk_ptr(cutlass.Float4E2M1FN, w2_weight.data_ptr())
    fc2_out_ptr = mk_ptr(cutlass.BFloat16, output.data_ptr())
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

    # --- Create and compile kernel ---
    sys.stderr.write("[Correctness] Compiling fused kernel (phase_mode=0)...\n")
    sys.stderr.flush()
    t0 = time.monotonic()
    kernel = FlashMoeFusedKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(tile_size, 128),
        cluster_shape_mn=(1, 1),
        vectorized_f32=True,
        topk=top_k,
        raster_along_m=False,
        enable_ar=False,
        ar_num_ranks=1,
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
        ar_staging_ipc_ptrs_ptr=None,
        ar_output_ptr=None,
        ar_cta_exit_counter_ptr=None,
        ar_rank_ready_flag_ptrs_ptr=None,
        ar_local_rank=None,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    t1 = time.monotonic()
    sys.stderr.write(f"[Correctness] Compilation done in {t1 - t0:.1f}s\n")
    sys.stderr.flush()

    # --- Launch kernel ---
    sys.stderr.write("[Correctness] Launching fused kernel...\n")
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
        ar_staging_ipc_ptrs_ptr=None,
        ar_output_ptr=None,
        ar_cta_exit_counter_ptr=None,
        ar_rank_ready_flag_ptrs_ptr=None,
        ar_local_rank=None,
        stream=stream,
    )

    # --- Wait with timeout ---
    sync_event = torch.cuda.Event()
    sync_event.record()
    deadline = time.monotonic() + 120
    while not sync_event.query():
        if time.monotonic() > deadline:
            sys.stderr.write("[Correctness] DEADLOCK: kernel did not complete within 120s\n")
            sys.stderr.flush()
            os._exit(42)
        time.sleep(0.1)

    sys.stderr.write("[Correctness] Kernel completed successfully\n")
    sys.stderr.flush()

    # ================================================================
    # Validation
    # ================================================================
    kernel_output = output.float()
    abs_max = kernel_output.abs().max().item()
    has_nan = torch.isnan(kernel_output).any().item()
    has_inf = torch.isinf(kernel_output).any().item()
    nonzero_count = (kernel_output.abs() > 1e-8).sum().item()

    sys.stderr.write(
        f"[Correctness] Kernel output: shape={output.shape}, "
        f"abs_max={abs_max:.6f}, nonzero={nonzero_count}/{kernel_output.numel()}, "
        f"nan={has_nan}, inf={has_inf}\n"
    )
    sys.stderr.flush()

    # Check 1: No NaN or Inf
    if has_nan or has_inf:
        sys.stderr.write("[FAIL] Output contains NaN or Inf\n")
        sys.stderr.flush()
        return False

    # Check 2: Output is non-zero
    if abs_max < 1e-8:
        sys.stderr.write("[FAIL] Output is all zeros\n")
        sys.stderr.flush()
        return False

    # Check 3: Uniformity — since all FP4 inputs/weights are identical and
    # softmax routing weights sum to 1.0, every output element should be
    # approximately the same value.
    output_mean = kernel_output.mean().item()
    output_std = kernel_output.std().item()
    cv = output_std / abs(output_mean) if abs(output_mean) > 1e-10 else float("inf")

    sys.stderr.write(
        f"[Correctness] Uniformity: mean={output_mean:.6f}, std={output_std:.6f}, "
        f"coeff_of_variation={cv:.6f}\n"
    )
    sys.stderr.flush()

    if cv > 0.05:
        sys.stderr.write(f"[FAIL] Output not uniform: coefficient of variation {cv:.4f} > 0.05\n")
        sys.stderr.flush()
        return False

    # Check 4: Magnitude matches analytical expected value
    expected_val, fc1_val, swiglu_val = compute_analytical_expected(
        hidden_size, intermediate_size, sf_vec_size
    )
    sys.stderr.write(
        f"[Correctness] Analytical: FC1={fc1_val:.6f}, SwiGLU={swiglu_val:.6f}, "
        f"expected_output={expected_val:.6f}\n"
    )
    sys.stderr.flush()

    ratio = output_mean / expected_val if abs(expected_val) > 1e-10 else float("inf")
    sys.stderr.write(
        f"[Correctness] Magnitude: kernel_mean={output_mean:.6f}, "
        f"analytical={expected_val:.6f}, ratio={ratio:.4f}\n"
    )
    sys.stderr.flush()

    # Allow 30% tolerance for FP4/FP8 quantization rounding
    if not (0.7 < ratio < 1.3):
        sys.stderr.write(f"[FAIL] Output magnitude mismatch: ratio={ratio:.4f} not in [0.7, 1.3]\n")
        sys.stderr.flush()
        return False

    # Check 5: Inspect FC1 intermediate buffer for sanity
    fc1_c_sample = fc1_c[0, :4].tolist()
    fc1_c_nonzero = (fc1_c != 0).sum().item()
    sys.stderr.write(
        f"[Correctness] FC1 intermediate: nonzero_bytes={fc1_c_nonzero}/{fc1_c.numel()}, "
        f"sample_row0_hex=[{', '.join(f'0x{b:02x}' for b in fc1_c_sample)}]\n"
    )
    sys.stderr.flush()

    sys.stderr.write("[PASSED] All correctness checks passed\n")
    sys.stderr.flush()
    return True


if __name__ == "__main__":
    success = run_correctness_test()
    sys.exit(0 if success else 1)
