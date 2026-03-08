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
"""Validation script for the adaptive accuracy framework.

Runs a representative subset of MoE test configurations and compares
the new adaptive thresholds against old hardcoded values.

Usage:
    cd tests/unittest
    python -m pytest _torch/modules/moe/test_accuracy_framework.py -v -s

Or run directly:
    cd tests/unittest
    python _torch/modules/moe/test_accuracy_framework.py
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from _torch.modules.moe.accuracy_utils import (
    ROUTING_TYPE_SIGMOID,
    ROUTING_TYPE_SOFTMAX,
    compute_adaptive_thresholds,
    moe_check_accuracy,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
G_LOGGER = logging.getLogger(__name__)


# ──────────────────── Pure-math threshold sanity checks ────────────────────


@dataclass
class ThresholdTestCase:
    """A single threshold calibration test case."""
    name: str
    hidden: int
    inter: int
    top_k: int
    quant: Optional[str]  # None or QuantAlgo name
    backend: str
    routing: str
    # Expected bounds for atol (inclusive)
    atol_min: float
    atol_max: float


def _get_quant_algo(name: Optional[str]):
    """Convert string name to QuantAlgo enum."""
    if name is None:
        return None
    from tensorrt_llm.models.modeling_utils import QuantAlgo
    return getattr(QuantAlgo, name)


# Representative threshold calibration test cases.
# atol_min/atol_max define the acceptable range for computed atol.
# Derived from old hardcoded thresholds + error budget analysis.
THRESHOLD_CASES = [
    # Unquantized: 3σ atol, floor dominates → 0.01
    ThresholdTestCase("unquant_small", 512, 512, 1, None, "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.005, 0.05),
    ThresholdTestCase("unquant_large", 7168, 2048, 8, None, "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.005, 0.05),

    # FP8: 2σ atol (was 3σ). atol = q_err(0.06) × activation × routing × 2.0
    ThresholdTestCase("fp8_k4", 2048, 1408, 4, "FP8", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.05, 0.20),
    ThresholdTestCase("fp8_k1", 512, 512, 1, "FP8", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.10, 0.30),
    ThresholdTestCase("fp8_k8", 7168, 2048, 8, "FP8", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.03, 0.15),

    # NVFP4: 1.5σ atol. atol = q_err(0.15) × activation × routing × 1.5
    ThresholdTestCase("nvfp4_k4", 2048, 1408, 4, "NVFP4", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.10, 0.40),
    ThresholdTestCase("nvfp4_k1", 512, 512, 1, "NVFP4", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.20, 0.60),

    # FP8_BLOCK_SCALES: 2σ atol. Backend floor may dominate.
    ThresholdTestCase("fp8bs_cutlass", 2880, 2880, 4, "FP8_BLOCK_SCALES", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.05, 0.20),
    ThresholdTestCase("fp8bs_deepgemm", 2880, 2880, 4, "FP8_BLOCK_SCALES", "DEEPGEMM", ROUTING_TYPE_SOFTMAX, 0.25, 0.60),
    ThresholdTestCase("fp8bs_trtllm", 2880, 2880, 4, "FP8_BLOCK_SCALES", "TRTLLM", ROUTING_TYPE_SOFTMAX, 0.10, 0.30),

    # Sigmoid routing (Llama4/DeepSeek): amplified errors
    ThresholdTestCase("fp8_sigmoid_k4", 2880, 2880, 4, "FP8", "CUTLASS", ROUTING_TYPE_SIGMOID, 0.10, 0.40),
    ThresholdTestCase("nvfp4_sigmoid_k8", 7168, 2048, 8, "NVFP4", "CUTLASS", ROUTING_TYPE_SIGMOID, 0.30, 1.00),

    # W8A16: 2σ atol, very low quantization error
    ThresholdTestCase("w8a16_k4", 2048, 1408, 4, "W8A16", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.005, 0.05),

    # W4A8_AWQ: 2σ atol, FP8-like
    ThresholdTestCase("w4a8awq_k4", 2048, 1408, 4, "W4A8_AWQ", "CUTLASS", ROUTING_TYPE_SOFTMAX, 0.05, 0.20),
]


@pytest.mark.parametrize("tc", THRESHOLD_CASES, ids=[tc.name for tc in THRESHOLD_CASES])
def test_threshold_calibration(tc: ThresholdTestCase):
    """Verify computed thresholds fall within expected calibration range."""
    quant_algo = _get_quant_algo(tc.quant)
    atol, rtol, mm = compute_adaptive_thresholds(
        hidden_size=tc.hidden,
        intermediate_size=tc.inter,
        top_k=tc.top_k,
        quant_algo=quant_algo,
        routing_type=tc.routing,
        dtype=torch.bfloat16,
        backend=tc.backend,
    )
    G_LOGGER.info(
        f"  {tc.name}: atol={atol:.4f} rtol={rtol:.4f} mm={mm:.4f} "
        f"(expected atol in [{tc.atol_min:.3f}, {tc.atol_max:.3f}])"
    )
    assert tc.atol_min <= atol <= tc.atol_max, (
        f"{tc.name}: atol={atol:.4f} out of expected range "
        f"[{tc.atol_min:.3f}, {tc.atol_max:.3f}]"
    )
    assert 0 < rtol < 1.0, f"{tc.name}: rtol={rtol} out of reasonable range"
    assert 0 < mm < 0.45, f"{tc.name}: max_mismatch={mm} out of reasonable range"


# ──────────────────── GPU accuracy checks (require CUDA) ────────────────────


def _skip_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def cuda_device():
    _skip_no_cuda()
    return torch.device("cuda")


def _make_synthetic_output(
    shape, dtype, device, noise_level=0.0, seed=42
):
    """Create synthetic MoE output pair (output, ref_output) with controlled noise.

    ref_output is random normal, output = ref_output + noise.
    This simulates kernel vs reference differences.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    ref_output = torch.randn(shape, dtype=torch.float32, device=device, generator=gen)
    noise = torch.randn_like(ref_output) * noise_level
    output = (ref_output + noise).to(dtype)
    ref_output = ref_output.to(dtype)
    return output, ref_output


@pytest.mark.parametrize("quant_name,noise", [
    (None, 0.001),       # Unquantized: very small noise
    ("FP8", 0.05),       # FP8: moderate noise
    ("NVFP4", 0.15),     # FP4: larger noise
    ("FP8_BLOCK_SCALES", 0.05),
    ("W8A16", 0.005),    # INT8: small noise
])
def test_synthetic_pass(cuda_device, quant_name, noise):
    """Verify that synthetic outputs with expected noise levels pass the check."""
    _skip_no_cuda()
    quant_algo = _get_quant_algo(quant_name)
    shape = (32, 2048)  # typical MoE output shape

    output, ref_output = _make_synthetic_output(
        shape, torch.bfloat16, cuda_device, noise_level=noise
    )

    # Should pass without assertion error
    metrics = moe_check_accuracy(
        output, ref_output,
        hidden_size=2048,
        intermediate_size=1408,
        top_k=4,
        quant_algo=quant_algo,
        routing_type=ROUTING_TYPE_SOFTMAX,
        dtype=torch.bfloat16,
        backend="CUTLASS",
    )
    G_LOGGER.info(
        f"  quant={quant_name}: cos_sim={metrics['cosine_similarity']:.6f} "
        f"mismatch={metrics['mismatch_rate']:.4f} max_diff={metrics['max_abs_diff']:.4f}"
    )
    # NVFP4 noise_level=0.15 produces cos_sim ~0.989 on GB200 due to bf16 casting;
    # 0.98 is a safe bound that still catches catastrophic failures.
    assert metrics["cosine_similarity"] > 0.98


@pytest.mark.parametrize("failure_mode", [
    "catastrophic_shape",     # Wrong routing: output is a permutation of ref
    "high_noise",             # Too much noise for the quant type
    "systematic_offset",      # Constant offset (wrong scale)
])
def test_synthetic_fail(cuda_device, failure_mode):
    """Verify that truly broken outputs are caught by the framework."""
    _skip_no_cuda()
    shape = (32, 2048)
    gen = torch.Generator(device=cuda_device).manual_seed(123)
    ref_output = torch.randn(shape, dtype=torch.bfloat16, device=cuda_device, generator=gen)

    if failure_mode == "catastrophic_shape":
        # Shuffle rows: simulates wrong expert routing
        output = ref_output[torch.randperm(shape[0], device=cuda_device)]
    elif failure_mode == "high_noise":
        # 10x expected noise for unquantized
        noise = torch.randn_like(ref_output) * 5.0
        output = ref_output + noise
    elif failure_mode == "systematic_offset":
        # Constant offset: scales output by 2x
        output = ref_output * 2.0

    with pytest.raises(AssertionError):
        moe_check_accuracy(
            output, ref_output,
            hidden_size=2048,
            intermediate_size=1408,
            top_k=4,
            quant_algo=None,
            routing_type=ROUTING_TYPE_SOFTMAX,
            dtype=torch.bfloat16,
            backend="CUTLASS",
        )


# ──────────────────── Full MoE integration test (optional) ────────────────────


def _run_full_moe_integration():
    """Run the actual MoE test with the new accuracy framework.

    This is the most important validation: run the real test_moe_module tests
    and confirm they pass with the new adaptive thresholds.

    Invoke with:
        cd tests/unittest
        python -m pytest _torch/modules/moe/test_moe_module.py -v -s \
            -k "test_moe_module" --timeout=600
    """
    pass  # Placeholder — actual validation runs via test_moe_module.py


# ──────────────────── CLI entry point ────────────────────


if __name__ == "__main__":
    print("=" * 80)
    print("MoE Adaptive Accuracy Framework — Threshold Calibration Report")
    print("=" * 80)

    from tensorrt_llm.models.modeling_utils import QuantAlgo

    # Run threshold calibration
    for tc in THRESHOLD_CASES:
        quant_algo = _get_quant_algo(tc.quant)
        atol, rtol, mm = compute_adaptive_thresholds(
            hidden_size=tc.hidden,
            intermediate_size=tc.inter,
            top_k=tc.top_k,
            quant_algo=quant_algo,
            routing_type=tc.routing,
            dtype=torch.bfloat16,
            backend=tc.backend,
        )
        in_range = tc.atol_min <= atol <= tc.atol_max
        status = "PASS" if in_range else "FAIL"
        print(
            f"  [{status}] {tc.name:<25} "
            f"atol={atol:.4f} rtol={rtol:.4f} mm%={mm:.4f} "
            f"(expected [{tc.atol_min:.3f}, {tc.atol_max:.3f}])"
        )

    # Run GPU tests if available
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Running GPU synthetic tests...")
        print("=" * 80)
        sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
    else:
        print("\nNo CUDA device available. Skipping GPU tests.")
        print("Run on GPU node with: python -m pytest _torch/modules/moe/test_accuracy_framework.py -v -s")
