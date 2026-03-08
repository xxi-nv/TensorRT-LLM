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

    # Sigma multiplier for atol derivation.
    #
    # This controls the tradeoff between per-element tolerance and mismatch-rate
    # detection sensitivity:
    #   - High sigma (3σ): generous atol, low baseline mismatch, but mismatch
    #     is INSENSITIVE to precision regressions (few elements near boundary).
    #   - Low sigma (1.5σ): tight atol, higher baseline mismatch, but mismatch
    #     is SENSITIVE to precision regressions (many elements near boundary).
    #
    # For unquantized: 3σ is fine (Gaussian distribution, few outliers).
    # For 8-bit quantized: 2σ balances coverage and detection.
    # For 4-bit quantized: 1.5σ gives detection sensitivity — a 2× error
    #   increase pushes P(|Z|>0.75) ≈ 45% vs P(|Z|>1.5) ≈ 13%, yielding a
    #   large mismatch jump that the threshold reliably catches.
    #   The resulting higher baseline mismatch (~15%) is accommodated by a
    #   correspondingly higher mismatch threshold.
    if quant_algo is None:
        sigma = 3.0
    elif quant_algo in (
        QuantAlgo.NVFP4,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    ):
        sigma = 1.5
    else:
        sigma = 2.0

    base_atol = (q_err + accum_err) * activation_factor * routing_factor
    atol = base_atol * sigma

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
    #
    # Design rationale: the atol above is derived from a 3-sigma Gaussian model,
    # covering ~99.7% of the WELL-QUANTIZED elements.  But quantization error
    # distributions are NOT Gaussian — they are bimodal for block-scaled formats:
    #
    #   Main peak (~85-90% of elements): quantization error < atol.
    #     These are elements in blocks with moderate dynamic range, where the
    #     block scale provides good coverage of all values.
    #
    #   Heavy tail (~10-15% for 4-bit, ~2-5% for 8-bit, ~0.3% for unquantized):
    #     Elements in blocks with HIGH dynamic range, where small values are
    #     coarsely quantized (error up to block_scale × max_quant_step).
    #     For MXFP4 (E2M1, block_size=32): ~15% of elements live in blocks
    #     where max(|w|)/median(|w|) > 3.7, producing per-element errors that
    #     far exceed the atol derived from the Gaussian model.
    #
    # The max_mismatch_rate must accommodate this structural tail WITHOUT being
    # so loose that it masks real accuracy bugs.
    #
    # Threshold = expected_mismatch + detection_gap, where:
    #   - expected_mismatch: predicted from quantization error distribution theory
    #   - detection_gap: minimum headroom to detect a 2× error increase
    #     (e.g., FP32→FP16 accumulator regression, the subtlest detectable bug)
    #
    # A 2× per-element error increase roughly doubles the mismatch rate, so
    # the detection gap must be >= expected_mismatch to catch it.
    # We use: threshold = expected × (1 + safety_factor), safety_factor >= 1.0.
    #
    # What real bugs look like (all detectable):
    #   - Wrong block index/scale:  mismatch > 50%, cos_sim < 0.95
    #   - Missing scale dimension:  mismatch > 80%, cos_sim < 0.90
    #   - Accumulator regression (FP32→FP16): mismatch ~2× normal
    #   - Off-by-one in expert routing: cos_sim < 0.5
    #
    # The cosine_sim gate (0.95 default) is the PRIMARY defense against
    # catastrophic bugs.  The mismatch check catches subtler precision
    # regressions that don't distort the overall output shape.
    #
    # Expected mismatch per tier (with tier-specific sigma for atol):
    #
    # For unquantized (3σ atol): virtually all main-peak elements pass.
    #   Expected mismatch ~0.3% (kernel tiling diffs, accumulator ordering).
    #
    # For 8-bit (2σ atol): P(|Z|>2) = 4.6% of main-peak elements fail,
    #   plus ~2% from quantization tail.  Expected mismatch ~5%.
    #
    # For 4-bit (1.5σ atol): P(|Z|>1.5) = 13.4% of main-peak elements fail,
    #   plus ~12% from block-quantization tail (block dynamic range outliers).
    #   Expected mismatch ~18%.  Old code used percent=0.85 (15% allowed) with
    #   atol≈0.2 (~1σ); our 1.5σ gives slightly higher atol → slightly lower
    #   mismatch, but the tail contribution is similar.
    #
    # Threshold = expected × headroom_factor.  The headroom must be large
    # enough that a 2× per-element error increase pushes mismatch past the
    # threshold.  With Nσ atol, a 2× error shifts the CDF boundary to N/2 σ,
    # causing P(|Z|>N/2) >> P(|Z|>N) for the main peak.
    #
    # Detection verification (2× error → new mismatch):
    #   unquantized: P(|Z|>1.5)=13% from main peak → well above 1% threshold ✓
    #   8-bit: P(|Z|>1)=32% from main peak → well above 8% threshold ✓
    #   4-bit: P(|Z|>0.75)=45% + 15% tail → ~50% total → above 25% threshold ✓
    _EXPECTED_MISMATCH = {
        "unquantized": 0.005,   # ~0.5% with 3σ atol
        "8bit":        0.05,    # ~5% with 2σ atol
        "4bit":        0.20,    # ~20% with 1.5σ atol (main peak ~13% + tail ~7%)
                                # Worst case at high K + high top_k where routing
                                # factor compresses atol significantly.
    }
    _HEADROOM_FACTOR = {
        "unquantized": 2.0,     # 0.5% × 2.0 = 1.0% threshold
        "8bit":        1.6,     # 5% × 1.6 = 8% threshold
        "4bit":        1.4,     # 20% × 1.4 = 28% threshold
    }

    # Classify quantization format into tier
    if quant_algo is None:
        tier = "unquantized"
    elif quant_algo in (
        QuantAlgo.NVFP4,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    ):
        tier = "4bit"
    else:
        # FP8, FP8_BLOCK_SCALES, W8A16, W4A8_AWQ (8-bit precision dominates)
        tier = "8bit"

    expected = _EXPECTED_MISMATCH[tier]
    headroom = _HEADROOM_FACTOR[tier]
    max_mismatch_rate = expected * headroom

    # Sigmoid routing: independent expert weights don't average errors like
    # softmax does, increasing the effective outlier fraction by ~30%.
    if routing_type == ROUTING_TYPE_SIGMOID:
        max_mismatch_rate *= 1.3

    # swiglu_gptoss_style: custom activation widens error distribution (~20%)
    if swiglu_gptoss_style:
        max_mismatch_rate *= 1.2

    max_mismatch_rate = min(max_mismatch_rate, 0.40)  # cap at 40%

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
