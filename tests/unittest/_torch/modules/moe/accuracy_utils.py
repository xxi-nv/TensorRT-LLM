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
Unified adaptive accuracy checking framework for MoE tests.

Replaces scattered hardcoded thresholds with a single function that derives
tolerances from problem parameters (quant_algo, hidden_size, top_k, etc.).

Error budget model:
    Total_Error ≈ Quant_Error × Activation_Amplification × Routing_Factor
                  + GEMM_Accumulation_Error

Each component is estimated from first principles:
- Quant_Error:  mantissa bits → per-element error bound
- Accum_Error:  sqrt(K) × accumulator_eps
- Activation:   SwiGLU empirical amplification ~1.5x
- Routing:      softmax averages (1/sqrt(top_k)), sigmoid sums (sqrt(top_k))

Threshold derivation uses 3-sigma coverage (99.7%) on the estimated error
distribution, with additional tail relaxation for larger problem sizes.

A cosine-similarity hard-fail gate catches catastrophic shape distortions
that element-wise checks might miss (e.g., wrong expert routing).
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm.models.modeling_utils import QuantAlgo

G_LOGGER = logging.getLogger(__name__)

# Per-element quantization error estimate (as a fraction of typical element magnitude).
#
# For raw formats, error ≈ 2^(-mantissa_bits).
# For block-scaled formats (NVFP4, MXFP4, FP8_BLOCK_SCALES), the block scale
# compensates much of the raw error.  The residual is the within-block
# quantization noise, which is roughly quant_step / sqrt(block_size) for each
# dot-product element.  We use empirically calibrated values that account for
# this compensation while still covering observed worst cases.
#
# These values are multiplied by activation_factor and routing_factor, then
# given a 3-sigma safety margin, so they should represent the ~1-sigma error.
_QUANT_ERROR_TABLE = {
    None: 0.0,  # unquantized — kernel tiling diffs handled by min atol
    QuantAlgo.FP8: 0.06,  # E4M3 per-tensor: 1/8 raw, scale compensates ~50%
    QuantAlgo.NVFP4: 0.15,  # E2M1 block-scaled: 1/2 raw, scale compensates ~70%
    QuantAlgo.FP8_BLOCK_SCALES: 0.06,  # E4M3 per-block: similar to FP8
    QuantAlgo.W4A8_NVFP4_FP8: 0.15,  # W4 block-scaled dominates
    QuantAlgo.W4A16_MXFP4: 0.15,  # MXFP4 block-scaled
    QuantAlgo.W4A8_MXFP4_MXFP8: 0.15,  # MXFP4 weight dominates
    QuantAlgo.W8A16: 1.0 / 128,  # INT8 per-channel (well-characterized)
    QuantAlgo.W4A8_AWQ: 0.06,  # INT4 weight + FP8 activation QDQ
}

# Routing type classification for error aggregation model.
# "softmax": weights sum to 1 → errors averaged → factor = 1/sqrt(top_k)
# "sigmoid": weights independent in (0,1) → errors sum → factor = sqrt(top_k)*E[w]
ROUTING_TYPE_SOFTMAX = "softmax"
ROUTING_TYPE_SIGMOID = "sigmoid"


def _quant_error_estimate(quant_algo: Optional[QuantAlgo]) -> float:
    """Estimate per-element quantization error (relative to element magnitude)."""
    return _QUANT_ERROR_TABLE.get(quant_algo, 0.1)


def _accum_error_estimate(K: int, accum_eps: float = 1.19e-7) -> float:
    """GEMM accumulation rounding error (statistical model).

    For a dot product of K elements accumulated in a dtype with epsilon
    `accum_eps`, the expected rounding error is O(sqrt(K) * eps).
    Most MoE kernels use FP32 accumulators (eps ≈ 1.19e-7).
    """
    return math.sqrt(max(K, 1)) * accum_eps


def compute_adaptive_thresholds(
    hidden_size: int,
    intermediate_size: int,
    top_k: int = 2,
    quant_algo: Optional[QuantAlgo] = None,
    routing_type: str = ROUTING_TYPE_SOFTMAX,
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "CUTLASS",
    swiglu_gptoss_style: bool = False,
):
    """Compute (atol, rtol, percent) from problem parameters.

    Returns:
        (atol, rtol, max_mismatch_rate):
            atol - absolute tolerance for element-wise |a - b| check
            rtol - relative tolerance for element-wise check (|a - b| <= atol + rtol*|b|)
            max_mismatch_rate - maximum allowed fraction of elements exceeding tolerance
    """
    K_eff = hidden_size + intermediate_size

    # --- atol: absolute tolerance ---
    q_err = _quant_error_estimate(quant_algo)
    accum_err = _accum_error_estimate(K_eff)

    # SwiGLU activation amplification (empirical 1.5x for standard, 2.0x for gptoss)
    activation_factor = 2.0 if swiglu_gptoss_style else 1.5

    # Routing aggregation factor
    if routing_type == ROUTING_TYPE_SIGMOID:
        # Sigmoid: independent weights in (0,1), variance sums across top_k experts.
        # E[w^2] ≈ 0.25 for sigmoid, so std ≈ 0.5 * sqrt(top_k)
        routing_factor = math.sqrt(max(top_k, 1)) * 0.5
    else:
        # Softmax: weights sum to 1, Jensen's inequality gives sum(w_i^2) >= 1/top_k.
        # Noise is averaged, so effective factor ≈ 1/sqrt(top_k).
        routing_factor = max(1.0 / math.sqrt(max(top_k, 1)), 0.3)

    # Combine with 3-sigma safety margin (covers 99.7% of normal distribution)
    base_atol = (q_err + accum_err) * activation_factor * routing_factor
    atol = base_atol * 3.0

    # Minimum floor: even for unquantized, kernel tiling vs cuBLAS reference
    # produces rounding differences from FP32 accumulation ordering.
    # Statistical model: O(sqrt(K) * fp32_eps) ≈ 0.00001 for K~3000.
    # We use a generous constant floor to cover edge cases across HW.
    atol = max(atol, 0.01)

    # Backend-specific adjustments
    # DEEPGEMM uses E8M0 scale format which introduces extra quantization step
    if backend == "DEEPGEMM" and quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
        atol = max(atol, 0.3)
    # TRTLLM Gen kernels have different tiling strategies
    if backend == "TRTLLM" and quant_algo in (
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A8_NVFP4_FP8,
    ):
        atol = max(atol, 0.15)

    # --- rtol: relative tolerance ---
    dtype_eps = torch.finfo(dtype).eps
    # Scale with log2(K_eff/512) to account for K-dependent accumulation ordering diffs
    log_scale = max(0, math.log2(max(1, K_eff / 512.0)))
    rtol = dtype_eps * (10 + 2 * log_scale)
    rtol = max(rtol, 1e-3)

    # For heavily quantized formats, rtol needs to account for quantization noise
    # relative to output magnitude (not just accumulation ordering)
    if quant_algo in (
        QuantAlgo.NVFP4,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    ):
        rtol = max(rtol, 0.05 + 0.01 * log_scale)
    elif quant_algo in (QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.W4A8_AWQ):
        rtol = max(rtol, 0.02 + 0.005 * log_scale)

    # --- max_mismatch_rate: allowed element failure fraction ---
    # Based on tail probability of error distribution:
    # Larger K -> wider distribution -> more tail outliers
    base_mismatch = 0.01  # 1% baseline
    tail_growth = 0.005 * log_scale  # +0.5% per doubling of K

    # Quantization adds independent noise, widening the tail
    if quant_algo is not None:
        tail_growth += 0.01

    # Sigmoid routing amplifies tail (independent expert errors sum)
    if routing_type == ROUTING_TYPE_SIGMOID:
        tail_growth += 0.02

    # Heavy quantization (4-bit) has wider tails
    if quant_algo in (
        QuantAlgo.NVFP4,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    ):
        tail_growth += 0.03

    # swiglu_gptoss_style has custom activation that can widen error distribution
    if swiglu_gptoss_style:
        tail_growth += 0.02

    max_mismatch_rate = min(base_mismatch + tail_growth, 0.20)  # cap at 20%

    return atol, rtol, max_mismatch_rate


def moe_check_accuracy(
    output: torch.Tensor,
    ref_output: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int = 2,
    quant_algo: Optional[QuantAlgo] = None,
    routing_type: str = ROUTING_TYPE_SOFTMAX,
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "CUTLASS",
    swiglu_gptoss_style: bool = False,
    # Override: if set, these replace the computed thresholds
    atol_override: Optional[float] = None,
    rtol_override: Optional[float] = None,
    max_mismatch_override: Optional[float] = None,
    # Cosine similarity hard-fail gate
    cosine_sim_threshold: float = 0.95,
    # Logging control
    log_metrics: bool = True,
):
    """Unified adaptive accuracy check for MoE kernel vs reference.

    Performs three checks in order:
    1. Cosine similarity (catches catastrophic shape distortions)
    2. Element-wise |a - b| <= atol + rtol * |b| with mismatch rate threshold
    3. Logs detailed metrics for debugging

    Raises:
        AssertionError with detailed diagnostics on failure.
    """
    assert output.shape == ref_output.shape, (
        f"Shape mismatch: output={output.shape}, ref={ref_output.shape}"
    )

    # Compute adaptive thresholds
    atol, rtol, max_mismatch_rate = compute_adaptive_thresholds(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        quant_algo=quant_algo,
        routing_type=routing_type,
        dtype=dtype,
        backend=backend,
        swiglu_gptoss_style=swiglu_gptoss_style,
    )

    # Apply overrides
    if atol_override is not None:
        atol = atol_override
    if rtol_override is not None:
        rtol = rtol_override
    if max_mismatch_override is not None:
        max_mismatch_rate = max_mismatch_override

    # Cast to float32 for comparison
    out_f = output.float()
    ref_f = ref_output.float()

    # --- Check 1: Cosine similarity (global shape check) ---
    out_flat = out_f.flatten()
    ref_flat = ref_f.flatten()
    # Handle zero vectors
    out_norm = torch.norm(out_flat)
    ref_norm = torch.norm(ref_flat)
    if out_norm > 0 and ref_norm > 0:
        cos_sim = F.cosine_similarity(
            out_flat.unsqueeze(0), ref_flat.unsqueeze(0)
        ).item()
    else:
        cos_sim = 1.0 if (out_norm == 0 and ref_norm == 0) else 0.0

    # --- Check 2: Element-wise mismatch ---
    diff = torch.abs(out_f - ref_f)
    threshold = atol + rtol * torch.abs(ref_f)
    mismatches = diff > threshold
    mismatch_rate = mismatches.float().mean().item()

    # --- Compute diagnostic metrics ---
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    # RMSE normalized by output scale
    rmse = torch.sqrt((diff**2).mean()).item()
    output_scale = torch.abs(ref_f).mean().item() + 1e-8
    nrmse = rmse / output_scale

    # --- Build diagnostics string ---
    diagnostics = (
        f"\n  Thresholds: atol={atol:.4f}, rtol={rtol:.4f}, "
        f"max_mismatch_rate={max_mismatch_rate:.4f}"
        f"\n  Cosine similarity: {cos_sim:.6f} (threshold={cosine_sim_threshold})"
        f"\n  Mismatch rate: {mismatch_rate:.4f} ({mismatch_rate*100:.2f}%)"
        f"\n  Max abs diff: {max_abs_diff:.6f}"
        f"\n  Mean abs diff: {mean_abs_diff:.6f}"
        f"\n  NRMSE: {nrmse:.6f}"
        f"\n  Config: hidden={hidden_size}, inter={intermediate_size}, "
        f"top_k={top_k}, quant={quant_algo}, routing={routing_type}, "
        f"dtype={dtype}, backend={backend}"
    )

    if log_metrics:
        G_LOGGER.info(f"[MoE Accuracy] {diagnostics}")

    # --- Verdict ---
    # Hard fail: cosine similarity too low
    assert cos_sim >= cosine_sim_threshold, (
        f"Cosine similarity {cos_sim:.4f} < {cosine_sim_threshold} "
        f"(catastrophic shape distortion){diagnostics}"
    )

    # Soft fail: element-wise mismatch rate too high
    assert mismatch_rate <= max_mismatch_rate, (
        f"Mismatch rate {mismatch_rate:.4f} ({mismatch_rate*100:.2f}%) > "
        f"{max_mismatch_rate:.4f} ({max_mismatch_rate*100:.2f}%){diagnostics}"
    )

    return {
        "cosine_similarity": cos_sim,
        "mismatch_rate": mismatch_rate,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "nrmse": nrmse,
        "atol": atol,
        "rtol": rtol,
        "max_mismatch_rate": max_mismatch_rate,
    }
