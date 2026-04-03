#!/usr/bin/env python3
"""FlashMoE single-GPU correctness test.

Validates the fused FC1+FC2 kernel output by:
1. Running the fused kernel (phase_mode=0) with non-zero NVFP4 data.
2. Reading the FC1 intermediate output (fc1_c buffer persists after kernel).
3. Dequantizing FC1 output + w2 weights to BF16.
4. Computing the reference FC2 output in pure PyTorch.
5. Comparing the kernel's FC2 output with the PyTorch reference.

No C++ bindings or multi-GPU setup needed -- uses the package shim
from test_kernel_direct.py.

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
# FP4 E2M1FN dequantization (pure Python/PyTorch)
# ============================================================
# Lookup table: 4-bit value -> FP32 value
# FP4 E2M1FN: sign(1) + exponent(2) + mantissa(1), bias=1
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,  # 0000: +0
        0.5,  # 0001: subnormal +0.5
        1.0,  # 0010: 2^(1-1) * (1+0/2) = 1.0
        1.5,  # 0011: 2^(1-1) * (1+1/2) = 1.5
        2.0,  # 0100: 2^(2-1) * (1+0/2) = 2.0
        3.0,  # 0101: 2^(2-1) * (1+1/2) = 3.0
        4.0,  # 0110: 2^(3-1) * (1+0/2) = 4.0
        6.0,  # 0111: 2^(3-1) * (1+1/2) = 6.0
        -0.0,  # 1000: -0
        -0.5,  # 1001: subnormal -0.5
        -1.0,  # 1010
        -1.5,  # 1011
        -2.0,  # 1100
        -3.0,  # 1101
        -4.0,  # 1110
        -6.0,  # 1111
    ],
    dtype=torch.float32,
)


def dequantize_nvfp4(
    packed: torch.Tensor,
    scale_factors: torch.Tensor,
    sf_vec_size: int,
    num_rows: int,
    num_cols: int,
) -> torch.Tensor:
    """Dequantize NVFP4 packed data to BF16.

    Args:
        packed: [num_rows, num_cols // 2] uint8 tensor (2 FP4 values per byte)
        scale_factors: FP8 E4M3 scale factors, shape varies:
            - For input: [num_rows * num_cols // sf_vec_size] (flat)
            - For weights: [num_rows, num_cols // sf_vec_size] (2D)
        sf_vec_size: number of elements per scale factor block (e.g., 16)
        num_rows: M dimension
        num_cols: K dimension (unpacked)

    Returns:
        BF16 tensor of shape [num_rows, num_cols]
    """
    device = packed.device
    lut = _FP4_E2M1_LUT.to(device)

    # Unpack: low nibble first, high nibble second
    low_nibbles = packed & 0x0F  # [num_rows, num_cols//2]
    high_nibbles = (packed >> 4) & 0x0F
    # Interleave: for each byte, low nibble is even index, high is odd
    unpacked_flat = torch.stack([low_nibbles, high_nibbles], dim=-1)  # [..., 2]
    unpacked = unpacked_flat.reshape(num_rows, num_cols).long()

    # Lookup FP4 -> FP32
    fp32_vals = lut[unpacked]  # [num_rows, num_cols]

    # Apply scale factors
    sf = scale_factors.to(torch.float32).reshape(num_rows, num_cols // sf_vec_size)
    # Broadcast scale: each scale applies to sf_vec_size consecutive elements
    sf_expanded = sf.repeat_interleave(sf_vec_size, dim=1)  # [num_rows, num_cols]

    result = fp32_vals * sf_expanded
    return result.to(torch.bfloat16)


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
# Reference FC2 computation (pure PyTorch)
# ============================================================
def compute_reference_fc2(
    fc1_c_packed,  # [m_padded, intermediate_size//2] uint8 (NVFP4)
    fc1_c_sf,  # FP8 E4M3 scale factors
    w2_weight,  # [experts, hidden_size, intermediate_size//2] uint8 (NVFP4)
    fc2_weight_scale,  # [experts, hidden_size, intermediate_size//sf_vec_size] FP8
    fc2_alpha,  # [experts] float32
    token_final_scales,  # [num_tokens, top_k] float32
    permuted_idx_to_expanded_idx,  # [m_padded] int32
    tile_idx_to_expert_idx,  # [num_tiles] int32
    tile_idx_to_mn_limit,  # [num_tiles] int32
    num_non_exiting_tiles,  # [1] int32
    num_tokens,  # original number of tokens
    hidden_size,
    intermediate_size,
    sf_vec_size,
    top_k,
    tile_size,
):
    """Compute reference FC2 output using dequantized data and BF16 matmul."""
    device = fc1_c_packed.device
    num_tiles = num_non_exiting_tiles.item()
    m_padded = permuted_idx_to_expanded_idx.shape[0]

    # Dequantize FC1 output: [m_padded, intermediate_size]
    fc1_bf16 = dequantize_nvfp4(
        packed=fc1_c_packed,
        scale_factors=fc1_c_sf,
        sf_vec_size=sf_vec_size,
        num_rows=m_padded,
        num_cols=intermediate_size,
    )

    # Output buffer: scatter-add target
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Process tiles (matching the kernel's tile-based execution)
    for tile_i in range(num_tiles):
        expert_local = tile_idx_to_expert_idx[tile_i].item()
        mn_limit = tile_idx_to_mn_limit[tile_i].item()
        # This tile's token range in permuted space
        tile_start = tile_i * tile_size
        tile_end = tile_start + tile_size

        # Dequantize this expert's w2 weight: [hidden_size, intermediate_size]
        w2_bf16 = dequantize_nvfp4(
            packed=w2_weight[expert_local],
            scale_factors=fc2_weight_scale[expert_local],
            sf_vec_size=sf_vec_size,
            num_rows=hidden_size,
            num_cols=intermediate_size,
        )

        alpha = fc2_alpha[expert_local].item()

        # Process each token in this tile
        for perm_idx in range(tile_start, min(tile_end, mn_limit)):
            expanded_idx = permuted_idx_to_expanded_idx[perm_idx].item()
            token_idx = expanded_idx // top_k
            expert_slot = expanded_idx % top_k
            if token_idx >= num_tokens:
                continue

            # FC2 GEMM for this token: [1, intermediate_size] @ [intermediate_size, hidden_size]
            fc1_row = fc1_bf16[perm_idx].unsqueeze(0).float()  # [1, intermediate_size]
            w2_t = w2_bf16.T.float()  # [intermediate_size, hidden_size]
            fc2_out = torch.matmul(fc1_row, w2_t).squeeze(0)  # [hidden_size]

            # Apply alpha and token_final_scales
            scale = alpha * token_final_scales[token_idx, expert_slot].item()
            fc2_out = fc2_out * scale

            # Scatter-add
            output[token_idx] += fc2_out.to(torch.bfloat16)

    return output


# ============================================================
# Main test
# ============================================================
def run_correctness_test():
    """Run fused kernel and compare FC2 output with PyTorch reference."""
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

    # Model config (smaller for faster test)
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

    # --- Create non-zero NVFP4 weights ---
    # Fill with 0x11: each byte = two FP4 values of 0.5
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

    # Weight scale factors: use small values to keep GEMM output in range
    # FC1: hidden_size=7168 elements × 0.5 × 0.5 × sf_w = output per element
    # Want output ~1.0, so sf_w ≈ 1/(7168*0.25) ≈ 0.00056
    # Use sf_w = 1/128 ≈ 0.0078 (representable in FP8) → output ≈ 7168*0.5*0.5*0.0078 ≈ 14
    # That's OK for SwiGLU + NVFP4 requant
    fc1_ws_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
    fc1_weight_scale = (
        fc1_ws_val.expand(
            experts_per_rank,
            intermediate_size * 2,
            hidden_size // sf_vec_size,
        )
        .contiguous()
        .to(device)
    )

    # FC2 weight scale: similar reasoning
    fc2_ws_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
    fc2_weight_scale = (
        fc2_ws_val.expand(
            experts_per_rank,
            hidden_size,
            intermediate_size // sf_vec_size,
        )
        .contiguous()
        .to(device)
    )

    fc1_alpha = torch.ones(experts_per_rank, dtype=torch.float32, device=device)
    fc2_alpha = torch.ones(experts_per_rank, dtype=torch.float32, device=device)
    fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)

    # --- Create non-zero NVFP4 input ---
    # Fill with 0x11: each byte = two FP4 values of 0.5
    gathered_a = torch.full(
        (max_tokens, hidden_size // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    # Input scale factors: use small values
    sfa_val = torch.tensor([1.0 / 128], dtype=torch.float32).to(torch.float8_e4m3fn)
    gathered_sfa = (
        sfa_val.expand(
            max_tokens * hidden_size // sf_vec_size,
        )
        .contiguous()
        .to(device)
    )

    # --- Routing ---
    router_logits = torch.randn(max_tokens, num_experts, dtype=torch.float32, device=device)
    topk_vals, topk_indices = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
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
        m * intermediate_size // sf_vec_size,
        dtype=torch.float8_e4m3fn,
        device=device,
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

    # --- Check kernel output ---
    kernel_output = output.clone()
    abs_max = kernel_output.abs().max().item()
    nonzero_count = (kernel_output.abs() > 1e-8).sum().item()
    has_nan = torch.isnan(kernel_output).any().item()
    has_inf = torch.isinf(kernel_output).any().item()

    sys.stderr.write(
        f"[Correctness] Kernel output: shape={kernel_output.shape}, "
        f"abs_max={abs_max:.6f}, nonzero={nonzero_count}/{kernel_output.numel()}, "
        f"nan={has_nan}, inf={has_inf}\n"
    )
    sys.stderr.flush()

    if has_nan or has_inf:
        sys.stderr.write("[Correctness] FAIL: output contains NaN or Inf\n")
        sys.stderr.flush()
        return False

    if abs_max < 1e-8:
        sys.stderr.write("[Correctness] FAIL: output is all zeros (trivial)\n")
        sys.stderr.flush()
        return False

    # --- Compute PyTorch reference for FC2 ---
    sys.stderr.write("[Correctness] Computing PyTorch reference for FC2...\n")
    sys.stderr.flush()
    t0 = time.monotonic()

    ref_output = compute_reference_fc2(
        fc1_c_packed=fc1_c,
        fc1_c_sf=fc1_c_sf,
        w2_weight=w2_weight,
        fc2_weight_scale=fc2_weight_scale,
        fc2_alpha=fc2_alpha,
        token_final_scales=token_final_scales,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        num_tokens=max_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        sf_vec_size=sf_vec_size,
        top_k=top_k,
        tile_size=tile_size,
    )
    t1 = time.monotonic()
    sys.stderr.write(f"[Correctness] Reference computed in {t1 - t0:.1f}s\n")
    sys.stderr.flush()

    ref_abs_max = ref_output.abs().max().item()
    ref_nonzero = (ref_output.abs() > 1e-8).sum().item()
    sys.stderr.write(
        f"[Correctness] Reference output: abs_max={ref_abs_max:.6f}, "
        f"nonzero={ref_nonzero}/{ref_output.numel()}\n"
    )
    sys.stderr.flush()

    # --- Compare ---
    abs_diff = (kernel_output - ref_output).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    denom = max(ref_abs_max, abs_max, 1e-6)
    rel_max = max_diff / denom

    sys.stderr.write(
        f"[Correctness] Comparison:\n"
        f"  abs_max_diff={max_diff:.6f}\n"
        f"  abs_mean_diff={mean_diff:.6f}\n"
        f"  rel_max_diff={rel_max:.6f}\n"
        f"  kernel abs_max={abs_max:.6f}\n"
        f"  ref abs_max={ref_abs_max:.6f}\n"
    )
    sys.stderr.flush()

    # Tolerance: NVFP4 has very low precision. The kernel uses hardware MMA
    # with NVFP4 inputs, while reference uses dequantized BF16 matmul.
    # Differences come from: (1) FP4 rounding in MMA vs LUT dequant,
    # (2) BF16 accumulation order, (3) scatter-add order.
    # Use generous tolerance: relative max < 50%, absolute mean < 20% of ref.
    tol_rel_max = 0.5
    tol_rel_mean = 0.2
    rel_mean = mean_diff / denom if denom > 1e-6 else mean_diff

    passed = rel_max < tol_rel_max and rel_mean < tol_rel_mean

    if passed:
        sys.stderr.write(
            f"[Correctness] PASSED (rel_max={rel_max:.4f} < {tol_rel_max}, "
            f"rel_mean={rel_mean:.4f} < {tol_rel_mean})\n"
        )
    else:
        sys.stderr.write(
            f"[Correctness] FAILED (rel_max={rel_max:.4f} vs {tol_rel_max}, "
            f"rel_mean={rel_mean:.4f} vs {tol_rel_mean})\n"
        )
    sys.stderr.flush()
    return passed


if __name__ == "__main__":
    success = run_correctness_test()
    if not success:
        sys.exit(1)
