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
MoE Backend Unit Tests

This module provides a unified test framework for testing different MoE backends
through the backend-level interfaces (quantize_input + run_moe), rather than
the high-level forward() interface.

Design Goals:
1. Test backend interfaces directly: routing_method.apply -> quantize_input -> run_moe
2. Cover all quantization + backend combinations
3. Use can_implement() interface to determine test skip logic
4. Support autotune and tactic capture testing
"""

import itertools
import logging
import os
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
from _torch.modules.moe.moe_test_utils import (
    IS_CI_MODE,
    MoeBackendType,
    MoeModelConfig,
    create_test_param,
    get_backend_class,
    iter_base_test_configs,
    replay_tactics_and_check,
    should_skip_to_accelerate_ci,
    skip_if_insufficient_gpu_memory,
    supports_autotuner_capture,
)
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.interface import MoE, MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoEDeepGemm
from tensorrt_llm._torch.modules.fused_moe.quantization import W4A8MXFP4MXFP8MegaMoEDeepGemmMethod
from tensorrt_llm._torch.utils import ActivationType, is_gated_activation
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

logger = logging.getLogger(__name__)


_MEGAMOE_BACKEND_TYPES = {
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
}


def _ensure_single_proc_dist_for_megamoe(backend_type: MoeBackendType, rank: int) -> None:
    """Every MegaMoE backend (DG + CuteDSL) resolves an EP ProcessGroup
    at construction time via ``_resolve_ep_pg``. Single-process tests
    must therefore initialise ``torch.distributed`` even when the test
    only exercises ``ep_size == 1`` -- otherwise the constructor raises
    ``MegaMoe*Unavailable``. Both MegaMoE backends need the same fixture
    so the dist helper must accept the full set."""
    if backend_type not in _MEGAMOE_BACKEND_TYPES:
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MegaMoE tests")
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)


def should_skip_gptoss(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    swiglu_gptoss_style: bool,
) -> Optional[str]:
    """
    Check if swiglu_gptoss_style test should be skipped for this backend.

    Only CUTLASS and TRTLLM backends support swiglu_gptoss_style (SwiGlu with custom
    alpha/beta/limit parameters and bias).

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        swiglu_gptoss_style: Whether swiglu_gptoss_style is enabled

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if not swiglu_gptoss_style:
        return None

    # Only CUTLASS and TRTLLM backends support swiglu_gptoss_style
    supported_backends = {MoeBackendType.CUTLASS, MoeBackendType.TRTLLM}
    if backend_type not in supported_backends:
        return (
            f"swiglu_gptoss_style is only supported by CUTLASS and TRTLLM backends "
            f"(got backend_type={backend_type.value})"
        )

    return None


def create_test_backend(
    backend_type: MoeBackendType,
    routing_method: RenormalizeMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    quant_config,
    mapping: Mapping,
    bias: bool = False,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    activation_type: ActivationType = ActivationType.Swiglu,
) -> MoE:
    """Create a MoE backend for testing."""
    backend_cls = get_backend_class(backend_type)

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=mapping,
        moe_backend=backend_type.value,
    )

    return create_moe_backend(
        moe_cls=backend_cls,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
        bias=bias,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        weight_loading_mode=weight_loading_mode,
        activation_type=activation_type,
    )


def test_megamoe_init_rejects_uneven_num_slots_with_value_error():
    routing_method = RenormalizeMoeRoutingMethod(top_k=1)
    model_config = ModelConfig(
        mapping=Mapping(
            world_size=4,
            rank=0,
            tp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
        ),
        moe_backend=MoeBackendType.MEGAMOE_DEEPGEMM.value,
    )

    with pytest.raises(
        ValueError,
        match=r"MegaMoEDeepGemm requires num_slots \(10\) divisible by ep_size \(4\)",
    ):
        MegaMoEDeepGemm(
            routing_method=routing_method,
            num_experts=10,
            hidden_size=512,
            intermediate_size=512,
            dtype=torch.bfloat16,
            model_config=model_config,
            init_load_balancer=False,
        )


def test_megamoe_post_load_rejects_uneven_num_slots_with_value_error(monkeypatch):
    import tensorrt_llm._torch.modules.fused_moe.quantization as quantization_module

    class DummyModule:
        _weights_loaded = True
        num_slots = 10
        ep_size = 4

    monkeypatch.setattr(quantization_module, "_import_deep_gemm", lambda: object())
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()

    with pytest.raises(
        ValueError,
        match=r"MegaMoEDeepGemm requires num_slots \(10\) divisible by ep_size \(4\)",
    ):
        method.post_load_weights(DummyModule())


def run_backend_moe(
    backend: MoE,
    backend_type: MoeBackendType,
    x_quantized: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    dtype: torch.dtype,
    router_logits: torch.Tensor = None,
    trtllm_use_router_logits: bool = True,
) -> torch.Tensor:
    """
    Run MoE computation with backend-specific parameters.

    Each backend has different requirements:
    - CUTLASS: output_dtype, token_final_scales=float32
    - TRTLLM: token_final_scales=bfloat16, optionally router_logits
    - CUTEDSL: token_final_scales=float32
    - DEEPGEMM: workspace, token_final_scales=float32
    - MegaMoE backends: token_selected_experts=int64, output_dtype

    Args:
        trtllm_use_router_logits: If True, TRTLLM backend uses router_logits for routing.
            If False, uses token_selected_experts and token_final_scales.
            Note: When both are provided, TRTLLM only uses (topk_ids and topk_weights).
    """
    # Common args for all backends (default: token_final_scales=float32)
    args = dict(
        x=x_quantized,
        token_selected_experts=token_selected_experts.to(torch.int32),
        token_final_scales=token_final_scales.to(torch.float32),
        x_sf=x_sf,
    )

    # Backend-specific overrides
    if backend_type == MoeBackendType.CUTLASS:
        args["output_dtype"] = dtype
    elif backend_type == MoeBackendType.TRTLLM:
        args["token_final_scales"] = token_final_scales.to(torch.bfloat16)
        if trtllm_use_router_logits:
            # Use router_logits for routing (TRTLLM will compute topk internally)
            args["router_logits"] = router_logits
            args["token_selected_experts"] = None
            args["token_final_scales"] = None
        # else: use token_selected_experts and token_final_scales (already set)
    elif backend_type == MoeBackendType.DEEPGEMM:
        import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils

        m_max = fp8_utils.align(x_quantized.shape[0], 128)
        args["workspace"] = backend.get_workspace(m_max, 128)
    elif backend_type in _MEGAMOE_BACKEND_TYPES:
        args["token_selected_experts"] = token_selected_experts.to(torch.int64)
        args["output_dtype"] = dtype

    return backend.run_moe(**args)


# ============================================================================
# Test Parameters
# ============================================================================

# Quantization algorithms to test
QUANT_ALGOS_TO_TEST = [
    None,  # Unquantized
    QuantAlgo.FP8,
    QuantAlgo.NVFP4,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.W4A8_NVFP4_FP8,
    QuantAlgo.W4A16_MXFP4,
    QuantAlgo.W4A8_MXFP4_FP8,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.W8A16,
    QuantAlgo.W4A8_AWQ,
]

# Backend types to test
BACKEND_TYPES_TO_TEST = [
    MoeBackendType.CUTLASS,
    MoeBackendType.TRTLLM,
    MoeBackendType.CUTEDSL,
    MoeBackendType.DEEPGEMM,
    MoeBackendType.DENSEGEMM,
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
]

# Data types to test
DTYPES_TO_TEST = [
    torch.float16,
    torch.bfloat16,
]

# Format: (num_experts, top_k, hidden_size, intermediate_size)
#
# Default runs the CI subset (TRTLLM_TEST_MOE_CI=1).
# Set TRTLLM_TEST_MOE_CI=0 for the full local config matrix.
CI_MOE_MODEL_CONFIGS = [
    # Real models (small/medium — tactic replay is model-size-independent,
    # e256 is covered by test_moe_module integration tests)
    MoeModelConfig(60, 4, 2048, 1408),  # Qwen1.5-MoE-A2.7B
    MoeModelConfig(128, 4, 2880, 2880),  # GPT-OSS-120B
    MoeModelConfig(8, 1, 512, 512),  # boundary: top_k=1, single expert activated
    # Boundary tests for tactic correctness
    MoeModelConfig(4, 4, 512, 512),  # top_k=num_experts, all experts activated
    MoeModelConfig(7, 2, 256, 512),  # prime num_experts
    MoeModelConfig(13, 3, 256, 512),  # prime num_experts, odd top_k
]

LOCAL_MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS + [
    MoeModelConfig(256, 8, 7168, 2048),  # DeepSeek-V3
    MoeModelConfig(256, 6, 4096, 2048),  # DeepSeek-V4-Flash
    MoeModelConfig(8, 2, 4096, 14336),  # Mixtral-8x7B
    MoeModelConfig(64, 6, 2048, 1408),  # DeepSeek-MoE-16B / DeepSeek-V2-Lite
    MoeModelConfig(8, 2, 6144, 32768),  # Grok-1
    # === Boundary Tests: small sizes ===
    MoeModelConfig(4, 2, 64, 128),  # very small hidden_size
    MoeModelConfig(4, 2, 128, 64),  # intermediate < hidden
]

MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS if IS_CI_MODE else LOCAL_MOE_MODEL_CONFIGS

# Sequence lengths to test
SEQ_LENS_TO_TEST = [1, 8]

# SwiGLU parameters for swiglu_gptoss_style testing
SWIGLU_ALPHAS = [1, 1.702]  # default, GPT-OSS (modeling_gpt_oss.py)
SWIGLU_BETAS = [0, 1.0]  # default, GPT-OSS
SWIGLU_LIMITS = [float("inf"), 7.0]  # default, GPT-OSS

# Full product of all SwiGLU combos (local exhaustive testing only)
LOCAL_SWIGLU_COMBOS = list(itertools.product(SWIGLU_ALPHAS, SWIGLU_BETAS, SWIGLU_LIMITS))

# CI: only non-gptoss (default) and one gptoss combo
# All non-default combos trigger the same swiglu_gptoss_style=True code path;
# different alpha/beta/limit values are just kernel parameters, not code branches.
CI_SWIGLU_COMBOS = [
    (1, 0, float("inf")),  # non-gptoss (default SwiGLU)
    (1.702, 1.0, 7.0),  # gptoss style (GPT-OSS real values)
]

SWIGLU_COMBOS = CI_SWIGLU_COMBOS if IS_CI_MODE else LOCAL_SWIGLU_COMBOS


def generate_test_params() -> List:
    """
    Generate test parameter combinations, filtering out unsupported configurations.

    Unsupported combinations (those with a skip_reason from get_quick_skip_reason)
    are excluded entirely so they never appear in pytest collection output.

    Returns:
        List of pytest.param objects for runnable test configurations only
    """
    params: List = []
    for (
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        model_config,
        seq_len,
        dtype,
        backend_type,
        quant_algo,
        routing_method_cls,
        skip_reason,
        test_id,
    ) in iter_base_test_configs(
        SWIGLU_COMBOS,
        MOE_MODEL_CONFIGS,
        SEQ_LENS_TO_TEST,
        DTYPES_TO_TEST,
        BACKEND_TYPES_TO_TEST,
        QUANT_ALGOS_TO_TEST,
    ):
        if skip_reason:
            continue
        param_values = (
            dtype,
            backend_type,
            quant_algo,
            seq_len,
            model_config,
            routing_method_cls,
            ActivationType.Swiglu,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
        )
        params.append(create_test_param(param_values, test_id))

    return params


# Pre-generate test parameters at module load time
TEST_PARAMS = generate_test_params()


def generate_element_wise_test_params() -> List:
    params: List = []
    for activation_type in [ActivationType.Silu, ActivationType.Relu2]:
        for (
            _,  # swiglu_alpha  (ignored)
            _,  # swiglu_beta   (ignored)
            _,  # swiglu_limit  (ignored)
            model_config,
            seq_len,
            dtype,
            backend_type,
            quant_algo,
            routing_method_cls,
            skip_reason,
            base_test_id,
        ) in iter_base_test_configs(
            [(1, 0, float("inf"))],  # swiglu parameters are irrelevant
            MOE_MODEL_CONFIGS,
            SEQ_LENS_TO_TEST,
            DTYPES_TO_TEST,
            [MoeBackendType.CUTLASS, MoeBackendType.TRTLLM],
            [None, QuantAlgo.NVFP4],
        ):
            if skip_reason:
                continue
            if backend_type == MoeBackendType.CUTLASS and activation_type == ActivationType.Silu:
                continue
            if backend_type == MoeBackendType.TRTLLM and quant_algo is None:
                continue
            test_id = f"act={activation_type.name}-{base_test_id}"
            param_values = (
                dtype,
                backend_type,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                activation_type,
                None,
                None,
                None,
            )
            params.append(create_test_param(param_values, test_id))
    return params


TEST_PARAMS += generate_element_wise_test_params()


# ============================================================================
# Test Implementation
# ============================================================================
#
# This file provides a UNIFIED TEST FRAMEWORK for testing all MoE backend
# implementations through their backend-level interfaces.
#
# =============================================================================
# Purpose & Scope
# =============================================================================
# - Test MoE backends via: routing_method.apply -> quantize_input -> run_moe
# - Single GPU execution (no multi-GPU/distributed testing)
# - Accuracy validation against reference implementations
#
# =============================================================================
# Test Coverage Matrix
# =============================================================================
# 1. BACKENDS: CUTLASS, TRTLLM, CUTEDSL, DEEPGEMM
#    - When using element wise activations (Relu2, Silu), only CUTLASS and TRTLLM
#      are supported
#
# 2. QUANTIZATION ALGORITHMS:
#    - When using Swiglu:
#      - Unquantized (None)
#      - FP8, FP8_BLOCK_SCALES
#      - NVFP4, W4A8_NVFP4_FP8
#      - W4A16_MXFP4, W4A8_MXFP4_MXFP8
#      - W8A16, W4A8_AWQ
#    - When using element-wise activations
#      - Unquantized (CUTLASS)
#      - NVFP4 (TRTLLM, CUTLASS)
#
# 3. ACTIVATION DTYPES: float16, bfloat16
#
# 4. AUTOTUNER TACTICS:
#    - Autotune phase: find optimal tactics via AutoTuner
#    - Capture phase: record all tactics used
#    - Replay phase: verify each tactic produces correct results
#
# 5. GPTOSS_STYLE (SwiGLU with custom parameters):
#    - swiglu_alpha: scaling factor (default=1)
#    - swiglu_beta: bias term (default=0)
#    - swiglu_limit: clipping limit (default=inf)
#    - Supported by: CUTLASS (W4A8_MXFP4_MXFP8), TRTLLM (W4A8_MXFP4_MXFP8)
#
# 6. MODEL CONFIGURATIONS:
#    - Real models: Mixtral, DeepSeek, Qwen, Grok, GPT-OSS
#    - Boundary cases: prime num_experts, small sizes, top_k=1, top_k=num_experts
#
# =============================================================================
# Skip Logic
# =============================================================================
# Tests are automatically skipped for unsupported configurations using:
# - backend.can_implement(): Check dtype/quant_algo/swiglu_gptoss_style support
# - should_skip_trtllm(): TRTLLM-specific constraints (num_experts % 4, etc.)
# - should_skip_cutedsl(): CuteDSL-specific accuracy issues
# - 128-alignment requirements for quantization
#
# =============================================================================
@pytest.mark.parametrize(
    "dtype_activation,backend_type,quant_algo,seq_len,model_config,"
    "routing_method_cls,activation_type,swiglu_alpha,swiglu_beta,swiglu_limit",
    TEST_PARAMS,
)
def test_moe_backend(
    dtype_activation: torch.dtype,
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    seq_len: int,
    model_config: MoeModelConfig,
    routing_method_cls,
    activation_type: ActivationType,
    swiglu_alpha: Optional[float],
    swiglu_beta: Optional[float],
    swiglu_limit: Optional[float],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test MoE backend with autotune to capture all tactics.

    This test verifies:
    1. Autotune works correctly with the backend
    2. All tactics are captured properly
    3. Different sequence lengths use appropriate tactics
    4. swiglu_gptoss_style (SwiGlu with custom parameters) works correctly
    """
    # DENSEGEMM: disable fused fc2_alpha path for backend-level testing.
    if backend_type == MoeBackendType.DENSEGEMM:
        monkeypatch.setenv("TRTLLM_MOE_FUSED_FC2_ALPHA", "0")

    # MEGAMOE_CUTEDSL v1 alpha gate (see
    # NVFP4MegaMoECuteDslMethod._check_v1_alpha_gate) rejects any checkpoint
    # whose fc31_alpha / fc2_alpha / fc2_input_scale deviates from 1.0
    # because the ported kernel hard-codes alpha=1 / norm_const=1. The
    # backend test uses NVFP4QuantizeUtil which always produces non-1
    # weight_scale_2 values; bypass the gate so the load -> post-load ->
    # run_moe path can be exercised end-to-end here. Production paths leave
    # the env var unset and the gate stays enforced.
    if backend_type == MoeBackendType.MEGAMOE_CUTEDSL:
        monkeypatch.setenv("TRTLLM_MEGAMOE_CUTEDSL_BYPASS_V1_ALPHA_GATE", "1")

    is_gated = is_gated_activation(activation_type)
    swiglu_gptoss_style = False
    if is_gated:
        # Determine swiglu_gptoss_style based on swiglu parameters
        # swiglu_gptoss_style is True when any swiglu parameter deviates from default
        # Default values: alpha=1, beta=0, limit=inf
        swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")

    ci_skip = should_skip_to_accelerate_ci(
        backend_type=backend_type,
        quant_algo=quant_algo,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
        dtype=dtype_activation,
        seq_len=seq_len,
        swiglu_gptoss_style=swiglu_gptoss_style,
        activation_type=activation_type,
    )
    if ci_skip:
        pytest.skip(ci_skip)

    # Extract model parameters
    num_experts = model_config.num_experts
    top_k = model_config.top_k
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype_activation)

    # Create mapping
    mapping = Mapping()
    mapping.rank = mpi_rank()
    _ensure_single_proc_dist_for_megamoe(backend_type, mapping.rank)

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Setup autotuner distributed state
        AutoTuner.get().setup_distributed_state(mapping)

        # Create routing method from parametrized class
        routing_method = routing_method_cls(top_k=top_k)

        # Create test inputs
        x = torch.randn((seq_len, hidden_size), dtype=dtype_activation, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype_activation, device="cuda")

        # Get quantization parameters
        # Pass backend_type to determine scale format (DEEPGEMM/TRTLLM need E8M0 scale)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )

        # Create quantize utility with swiglu_gptoss_style parameters
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype_activation,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=swiglu_gptoss_style,
            swiglu_gptoss_style=swiglu_gptoss_style,
            swiglu_alpha=swiglu_alpha if swiglu_gptoss_style else None,
            swiglu_beta=swiglu_beta if swiglu_gptoss_style else None,
            swiglu_limit=swiglu_limit if swiglu_gptoss_style else None,
            activation_type=activation_type,
        )

        # Get swiglu tensors if swiglu_gptoss_style is enabled
        swiglu_tensors = quantize_util.get_swiglu_tensors()

        # Determine weight loading mode based on quantization algorithm
        weight_loading_mode = MoEWeightLoadingMode.VANILLA
        if hasattr(quantize_util, "weight_loading_mode"):
            weight_loading_mode = quantize_util.weight_loading_mode

        # Clear class-level permute indices cache between parametrized test cases
        # to work around a B200-specific kernel bug (tactic [32,5] illegal memory access)
        from tensorrt_llm._torch.modules.fused_moe.quantization import (
            NVFP4TRTLLMGenFusedMoEBaseMethod,
        )

        NVFP4TRTLLMGenFusedMoEBaseMethod._cache_permute_indices.clear()

        # Create backend first (needed for MXFP4_MXFP8 to get shapes)
        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype_activation,
            quant_config=quant_config,
            mapping=mapping,
            bias=swiglu_gptoss_style,
            swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
            swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
            swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
            weight_loading_mode=weight_loading_mode,
            activation_type=activation_type,
        )

        # W4A8_MXFP4_MXFP8 / W4A8_MXFP4_FP8 require backend-layout-aware
        # weights. CUTLASS and MegaMoE use 128 hidden alignment; TRTLLMGen
        # pads FC1 input to 512. MXFP4FP8QuantizeUtil inherits
        # prepare_weights_from_backend from MXFP4MXFP8QuantizeUtil so the
        # backend-vs-reference weight split applies to both variants.
        ref_cls = quant_kwargs.pop("ref_cls", None)
        ref_module_kwargs = {}
        if quant_algo in (QuantAlgo.W4A8_MXFP4_MXFP8, QuantAlgo.W4A8_MXFP4_FP8):
            weights, ref_weights, ref_module_kwargs = quantize_util.prepare_weights_from_backend(
                backend, **quant_kwargs
            )
        else:
            weights = quantize_util.create_weights(**quant_kwargs)
            ref_weights = weights

        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        # Create reference
        if ref_cls is not None:
            ref_fused_moe = quantize_util.create_ref_module(
                routing_method, ref_cls=ref_cls, **ref_module_kwargs
            )
        else:
            ref_fused_moe = quantize_util.create_ref_module(routing_method, **ref_module_kwargs)
        ref_fused_moe.load_weights([ref_weights])
        ref_fused_moe.cuda()

        # Clear autotuner cache before autotune phase
        AutoTuner.get().clear_cache()

        # Get reference output first
        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        # Helper to run MoE computation
        def run_moe():
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            return run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype_activation,
                router_logits,
            )

        # Configure AutoTuner for faster profiling (reduce warmup/repeat for unit tests)
        autotuner = AutoTuner.get()
        autotuner.warmup = 0  # default: 2
        autotuner.repeat = 1  # default: 10
        autotuner.stream_delay_micro_secs = 10  # default: 1000

        # Autotune phase: tune kernels to find best tactics
        # Use cache_path to speed up subsequent runs by reusing tuning results
        with torch.inference_mode(), autotune(cache_path="/tmp/moe_autotuner_cache.json"):
            _ = run_moe()

        # flashinfer has no capture and replay mechanisms, so we skip test_all_kernels
        use_flashinfer = getattr(backend, "use_flashinfer", False)

        # Check if this backend+quant_algo combination supports autotuner capture/replay
        if supports_autotuner_capture(backend_type, quant_algo, use_flashinfer):
            # Capture phase: record which tactics are used (requires actual execution)
            with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
                _ = run_moe()

            # Replay phase: test each tactic for correctness
            # Set fail_fast=True to stop on first failure, False to run all and report summary
            replay_tactics_and_check(
                all_tactics=all_tactics,
                run_moe_fn=run_moe,
                check_accuracy_fn=ref_fused_moe.check_accuracy,
                ref_output=ref_output,
                backend_type=backend_type,
                quant_algo=quant_algo,
                fail_fast=False,  # Change to True to fail on first error
            )
        else:
            # For backends that don't support autotuner capture/replay,
            # just run a simple accuracy check
            with torch.inference_mode():
                output = run_moe()
                ref_fused_moe.check_accuracy(output, ref_output)


# ============================================================================
# MegaMoECuteDsl focused tests
#
# The MegaMoECuteDsl backend is gated behind a CUDA 13 Cutlass DSL runtime
# (PR #14354) and an NVSHMEM-backed symmetric-memory provider (hard gate
# in MEGAMOE_CUTEDSL_DESIGN.md). Until both land, the full ``run_moe`` path
# is unreachable in production. The tests below cover the contract that IS
# implementable today:
#   * package import / module-level constants
#   * ``can_implement`` positive and negative cases
#   * ``to_blocked`` / ``from_blocked`` byte-equivalence
#   * tactic representation: validation, JSON serialization, ``repr``
#     round-trip, ``group_hint`` resolution
#   * ``quantize_input`` zero-token short-circuit (also exercises the
#     scheduler refactor that no longer special-cases zero-token chunks)
#   * v1 alpha-gate rejection inside ``NVFP4MegaMoECuteDslMethod``
#   * ``run_moe`` raises the clear ``MegaMoeCuteDslUnavailable`` error
#     until the provider lands
# ============================================================================


def _skip_if_no_megamoe_cutedsl_runtime():
    from tensorrt_llm._torch.modules.fused_moe.mega_moe.mega_moe_cute_dsl import (
        is_megamoe_cute_dsl_runtime_available,
    )

    ok, reason = is_megamoe_cute_dsl_runtime_available()
    if not ok:
        pytest.skip(reason)


def test_megamoe_cutedsl_kernel_package_imports():
    """Importing the lazy package surface must not pull the heavyweight
    kernel module on environments without the CUDA 13 Cutlass DSL runtime;
    only the constants + blocked_scale helpers are loaded eagerly.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.mega_moe_nvfp4 import (
        Nvfp4BlockSize,
        SfPaddingBlock,
        from_blocked,
        to_blocked,
    )

    assert Nvfp4BlockSize == 16
    assert SfPaddingBlock == 128
    assert callable(to_blocked)
    assert callable(from_blocked)


@pytest.mark.parametrize(
    "rows,cols",
    [(8, 4), (128, 16), (256, 32), (1, 1), (0, 0)],
)
def test_megamoe_cutedsl_to_blocked_roundtrip(rows, cols):
    """``from_blocked(to_blocked(x))`` must recover the original raw scale
    bytes for representative shapes used by FC1/FC2 SF tensors.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.mega_moe_nvfp4 import from_blocked, to_blocked

    if rows == 0 or cols == 0:
        raw = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device="cpu")
        flat = to_blocked(raw)
        # Empty input must short-circuit to a length-0 view.
        assert flat.numel() == 0
        return

    # Use bytewise-deterministic values so the byte-equivalence test is
    # robust to FP8 NaN canonicalization.
    raw_uint8 = (
        (torch.arange(rows * cols, dtype=torch.int32) % 200).to(torch.uint8).reshape(rows, cols)
    )
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    flat = to_blocked(raw_fp8)
    recovered = from_blocked(flat, rows, cols)
    assert recovered.shape == (rows, cols)
    assert torch.equal(recovered.view(torch.uint8), raw_uint8), (
        f"to_blocked/from_blocked roundtrip mismatch for ({rows}, {cols})"
    )


def test_megamoe_cutedsl_can_implement_positive_and_negative():
    """``MegaMoECuteDsl.can_implement`` is a pure-Python capability query
    that does not require CUDA. It should:

    * Accept ``NVFP4 + bf16 + SM100-aligned shapes`` when running on an
      SM100 GPU with the required cu13 Cutlass DSL symbols available.
    * Reject non-NVFP4 quant, non-bf16 activations, ``swiglu_gptoss_style``,
      and unaligned hidden/intermediate shapes regardless of host
      environment (the negative checks short-circuit before the
      SM/runtime probe).
    """
    from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoECuteDsl

    ok, reason = MegaMoECuteDsl.can_implement(QuantAlgo.W4A8_MXFP4_MXFP8)
    assert not ok, "must reject non-NVFP4 quant"
    assert "NVFP4" in reason

    ok, reason = MegaMoECuteDsl.can_implement(QuantAlgo.NVFP4, dtype_activation=torch.float16)
    assert not ok, "must reject non-bf16 activation"
    assert "bfloat16" in reason or "bf16" in reason.lower()

    ok, reason = MegaMoECuteDsl.can_implement(QuantAlgo.NVFP4, swiglu_gptoss_style=True)
    assert not ok, "must reject swiglu_gptoss_style"

    ok, reason = MegaMoECuteDsl.can_implement(QuantAlgo.NVFP4, hidden_size=33)
    assert not ok, "must reject unaligned hidden_size"
    assert "32" in reason

    ok, reason = MegaMoECuteDsl.can_implement(QuantAlgo.NVFP4, intermediate_size=15)
    assert not ok, "must reject unaligned intermediate_size"
    assert "16" in reason

    # Positive case: requires SM100 + cu13 cutlass-dsl. Run if available.
    if not torch.cuda.is_available():
        pytest.skip("positive can_implement check needs an SM100 GPU")
    sm = torch.cuda.get_device_capability(0)
    if sm[0] != 10:
        pytest.skip(f"positive can_implement check needs SM100 family, got {sm}")
    _skip_if_no_megamoe_cutedsl_runtime()
    ok, reason = MegaMoECuteDsl.can_implement(
        QuantAlgo.NVFP4, hidden_size=2048, intermediate_size=2048
    )
    assert ok, f"positive can_implement failed: {reason}"


def test_megamoe_cutedsl_tactic_validation():
    """Tactic tuples must pass the kernel-side validation (see
    MEGAMOE_CUTEDSL_DESIGN.md "MegaMoECuteDsl tactic representation").

    Tactics live in :mod:`tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op`
    (the runner module), not in the backend module — matching the
    boundary used by ``fused_moe_cute_dsl.py`` for its inner runners
    (``cute_dsl_custom_ops.py`` owns the Runner + op + tactic).
    """
    from tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op import (
        DEFAULT_MEGAMOE_TACTIC,
        enumerate_megamoe_candidate_tactics,
        resolve_megamoe_group_hint,
        validate_megamoe_tactic,
    )

    # Default tactic must be valid once group_hint is resolved.
    tactic = (
        list(DEFAULT_MEGAMOE_TACTIC[0]),
        list(DEFAULT_MEGAMOE_TACTIC[1]),
        DEFAULT_MEGAMOE_TACTIC[2],
        max(1, resolve_megamoe_group_hint(tuple(DEFAULT_MEGAMOE_TACTIC[1]))),
        DEFAULT_MEGAMOE_TACTIC[4],
        DEFAULT_MEGAMOE_TACTIC[5],
    )
    validate_megamoe_tactic(tactic)

    # All run_mega_tests.sh-derived candidate tactics must validate.
    for cand in enumerate_megamoe_candidate_tactics():
        validate_megamoe_tactic(cand)

    def _patched(idx, value):
        new = list(tactic)
        new[idx] = value
        return tuple(new)

    # Negative cases (positional index in the 6-tuple):
    with pytest.raises(ValueError, match="mma_tiler_mnk"):
        validate_megamoe_tactic(_patched(0, [64, 128, 256]))
    with pytest.raises(ValueError, match="cluster_shape_mnk"):
        validate_megamoe_tactic(_patched(1, [1, 2, 1]))
    with pytest.raises(ValueError, match="use_2cta_instrs"):
        validate_megamoe_tactic(_patched(2, True))
    with pytest.raises(ValueError, match="resolved_group_hint"):
        validate_megamoe_tactic(_patched(3, None))
    with pytest.raises(ValueError, match="load_balance_mode"):
        validate_megamoe_tactic(_patched(4, "clc"))


def test_megamoe_cutedsl_tactic_json_and_repr_roundtrip():
    """Tactic tuples must round-trip through ``eval(repr(tactic))`` so
    the AutoTuner cache can serialize them. Inner fields are
    JSON-friendly primitives so ``json.dumps`` also works (the tuple
    casts to a list at the top level, but the round-tripped value
    compares structurally equal when re-wrapped as tuple).
    """
    import json

    from tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op import DEFAULT_MEGAMOE_TACTIC

    tactic = (
        list(DEFAULT_MEGAMOE_TACTIC[0]),
        list(DEFAULT_MEGAMOE_TACTIC[1]),
        DEFAULT_MEGAMOE_TACTIC[2],
        4,
        DEFAULT_MEGAMOE_TACTIC[4],
        DEFAULT_MEGAMOE_TACTIC[5],
    )

    # repr round-trip preserves both tuple structure and inner list values.
    restored = eval(repr(tactic))
    assert restored == tactic

    # json round-trip: tuple becomes list at top level, re-wrap as tuple.
    serialized = json.dumps(list(tactic))
    restored_json = tuple(json.loads(serialized))
    assert restored_json == tactic


def test_megamoe_cutedsl_quantize_input_zero_tokens():
    """``MegaMoECuteDsl.quantize_input`` must accept zero-token input
    without launching ``fp4_quantize`` so the FusedCommMoEScheduler can
    call it uniformly for empty chunks. The empty layout is NVFP4 packed
    bytes + plain K-major FP8 SF bytes (uint8 alias).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for quantize_input")
    _skip_if_no_megamoe_cutedsl_runtime()
    sm = torch.cuda.get_device_capability(0)
    if sm[0] != 10:
        pytest.skip(f"MegaMoECuteDsl requires SM100 family, got {sm}")

    from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoECuteDsl

    routing_method = RenormalizeMoeRoutingMethod(top_k=2)
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = 4
    pretrained_config.hidden_size = 1024
    pretrained_config.intermediate_size = 1024
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
    )
    backend = MegaMoECuteDsl(
        routing_method=routing_method,
        num_experts=4,
        hidden_size=1024,
        intermediate_size=1024,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
    )

    from tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op import (
        megamoe_activation_sf_bytes_per_row,
    )

    empty_x = torch.empty((0, 1024), dtype=torch.bfloat16, device="cuda")
    x_fp4, x_sf = backend.quantize_input(empty_x)
    assert x_fp4.shape[0] == 0
    assert x_fp4.shape[1] == 1024 // 2
    assert x_sf.shape[0] == 0
    # hidden=1024 -> ceil(1024/16)=64, round_up(64, 4)=64 (no pad needed)
    assert x_sf.shape[1] == megamoe_activation_sf_bytes_per_row(1024) == 64


def test_megamoe_cutedsl_run_moe_multi_rank_requires_ep_pg():
    """Multi-rank (``ep_size > 1``) ``run_moe`` requires a real EP
    process group so the ``MegaMoeSymmMemProvider`` can run its
    rendezvous collective. The provider is backed by PyTorch's
    ``torch.distributed._symmetric_memory`` (cuMem-based NVSHMEM
    equivalent already used by ``SymmetricMemoryAllReduce``).

    Without a live ProcessGroup, ``_get_symm_provider`` raises
    ``MegaMoeCuteDslUnavailable`` with an actionable message so the
    user can switch to Ray / DeviceMesh / mpirun.

    Single-rank degenerate path (``ep_size == 1``) is supported and
    exercised by the module-level
    ``test_megamoe_cutedsl_factory_routing_and_scheduler`` test on a
    real GPU. Real multi-rank end-to-end is covered by the multi-GPU
    EP tests once the OCI worktree is rebuilt.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for backend instantiation")
    _skip_if_no_megamoe_cutedsl_runtime()
    sm = torch.cuda.get_device_capability(0)
    if sm[0] != 10:
        pytest.skip(f"MegaMoECuteDsl requires SM100 family, got {sm}")

    from tensorrt_llm._torch.modules.fused_moe.mega_moe import (
        MegaMoECuteDsl,
        MegaMoeCuteDslUnavailable,
    )

    routing_method = RenormalizeMoeRoutingMethod(top_k=2)
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = 4
    pretrained_config.hidden_size = 1024
    pretrained_config.intermediate_size = 1024
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
    )
    backend = MegaMoECuteDsl(
        routing_method=routing_method,
        num_experts=4,
        hidden_size=1024,
        intermediate_size=1024,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
    )

    # Force multi-rank ep_size AFTER the backend was constructed
    # single-rank: the symm provider was therefore NOT allocated at
    # ``create_weights`` time (per design, provider rendezvous is a
    # build-time collective). ``run_moe`` must raise a clear
    # MegaMoeCuteDslUnavailable when the cached provider is missing.
    backend.ep_size = 4
    backend.ep_rank = 0
    backend._symm_provider = None  # ensure no leftover

    # Non-empty inputs so run_moe reaches the multi-rank branch
    # (zero-token short-circuit returns before the provider check).
    x_buf = torch.zeros((1, 1024 // 2), dtype=torch.uint8, device="cuda")
    sf_buf = torch.zeros((1, 1024 // 16), dtype=torch.uint8, device="cuda")
    topk_ids = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    topk_w = torch.zeros((1, 2), dtype=torch.float32, device="cuda")
    with pytest.raises(MegaMoeCuteDslUnavailable, match="symmetric-memory"):
        backend.run_moe(
            x=x_buf,
            token_selected_experts=topk_ids,
            token_final_scales=topk_w,
            x_sf=sf_buf,
            output_dtype=torch.bfloat16,
        )


def test_megamoe_cutedsl_alpha_gate_rejects_non_one():
    """``NVFP4MegaMoECuteDslMethod._check_v1_alpha_gate`` must reject any
    checkpoint whose ``fc31_alpha`` / ``fc2_alpha`` / ``fc2_input_scale``
    deviates from 1.0 within FP32 tolerance, because the ported kernel
    hard-codes those values (see MEGAMOE_CUTEDSL_DESIGN.md "NVFP4 scale
    and alpha ABI"). Once the kernel ABI is extended, this gate is
    removed and the test should be updated to assert pass-through.
    """
    from tensorrt_llm._torch.modules.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    class _StubModule:
        # Mimic the minimal nn.Module surface the gate needs (Parameter-like).
        class _Param:
            def __init__(self, data):
                self.data = data

        def __init__(self, fc31_alpha, fc2_alpha, fc2_input_scale):
            self.fc31_alpha = _StubModule._Param(fc31_alpha)
            self.fc2_alpha = _StubModule._Param(fc2_alpha)
            self.fc2_input_scale = _StubModule._Param(fc2_input_scale)

    method = NVFP4MegaMoECuteDslMethod()

    # All-ones must pass.
    method._check_v1_alpha_gate(
        _StubModule(
            torch.ones(4, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
    )

    # One non-1 fc31_alpha entry must raise.
    bad_fc31 = torch.ones(4, dtype=torch.float32)
    bad_fc31[2] = 0.5
    with pytest.raises(NotImplementedError, match="fc31_alpha"):
        method._check_v1_alpha_gate(
            _StubModule(
                bad_fc31, torch.ones(4, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
            )
        )

    # Non-1 fc2_input_scale must raise.
    with pytest.raises(NotImplementedError, match="fc2_input_scale"):
        method._check_v1_alpha_gate(
            _StubModule(
                torch.ones(4, dtype=torch.float32),
                torch.ones(4, dtype=torch.float32),
                torch.tensor(2.0, dtype=torch.float32),
            )
        )


def test_megamoe_cutedsl_sf_byte_width_helper():
    """``megamoe_activation_sf_bytes_per_row`` must match
    ``round_up(ceil(hidden / 16), 4)`` so that backend SF staging and
    ``quantize_input`` output match the kernel TMA load expectation
    (``sf_uint32_per_token = ceil(hidden / 64)`` uint32 per row,
    i.e. ``ceil(hidden / 64) * 4`` bytes). Hidden sizes that are 32-
    aligned but not 64-aligned (1568, 1632, 2080) must round up by 2.
    """
    from tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op import (
        megamoe_activation_sf_bytes_per_row,
    )

    # Aligned-to-64 hidden sizes: row width == hidden // 16 exactly.
    for hidden in (1024, 2048, 4096):
        assert megamoe_activation_sf_bytes_per_row(hidden) == hidden // 16
        assert megamoe_activation_sf_bytes_per_row(hidden) % 4 == 0

    # 32-aligned-only hidden sizes: row width pads up by 2 bytes to the
    # next multiple of 4 columns (one uint32 column covers 64 elements).
    for hidden in (1568, 1632, 2080):
        sf_cols = megamoe_activation_sf_bytes_per_row(hidden)
        raw_cols = (hidden + 15) // 16
        assert sf_cols == raw_cols + 2, (
            f"hidden={hidden}: expected {raw_cols + 2} byte cols, got {sf_cols}"
        )
        assert sf_cols % 4 == 0
        # And it must equal `ceil(hidden / 64) * 4` (kernel formula).
        assert sf_cols == ((hidden + 63) // 64) * 4

    # Sanity: non-positive or non-32-aligned hidden must raise.
    with pytest.raises(ValueError):
        megamoe_activation_sf_bytes_per_row(0)
    with pytest.raises(ValueError):
        megamoe_activation_sf_bytes_per_row(48)


def test_megamoe_cutedsl_atomic_counter_in_candidates():
    """Sweep ``load_balance_mode in {"static", "atomic_counter"}`` per
    the design (``MegaMoECuteDsl tactic representation`` /
    ``LoadBalanceMode`` enum). The previous v1 sweep only emitted
    ``static`` tactics; ``atomic_counter`` is kernel-supported (see
    ImplDesc.__post_init__ in fc1_fc2_fuse_sched.py) and was being
    skipped by the autotuner.
    """
    from tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op import (
        enumerate_megamoe_candidate_tactics,
    )

    tactics = enumerate_megamoe_candidate_tactics()
    modes = {t[4] for t in tactics}
    assert "static" in modes, f"static load_balance_mode missing from candidates: {tactics!r}"
    assert "atomic_counter" in modes, (
        f"atomic_counter load_balance_mode missing from candidates: {tactics!r}"
    )
    # ``clc`` was intentionally excluded (routes through a different
    # scheduler not wired through FC12 fused kernel here).
    assert "clc" not in modes


def test_megamoe_cutedsl_fc1_gate_up_interleave_byte_equivalence():
    """Byte-equivalence check for ``_build_mega_format_weights`` FC1
    gate/up 16-atom interleave.

    Per MEGAMOE_CUTEDSL_DESIGN.md "FC1 gate/up interleave" and upstream
    ``swiglu_fold_interleave_16`` in ``runner_fc12.py``, the kernel's
    FC1 epilogue indexes the output column ``c`` as
    ``gate_col = 2*(c//16)*16 + c%16``, ``up_col = (2*(c//16)+1)*16 + c%16``.
    That means the storage layout along the ``expand_intermediate``
    axis MUST be
    ``[gate[0:16], up[0:16], gate[16:32], up[16:32], ...]``.

    The parent ``w3_w1_weight`` stores ``[w3 | w1]`` cat'd along M;
    ``up = w3 = w3_w1[:intermediate]``, ``gate = w1 = w3_w1[intermediate:]``.

    This test fills the parent buffer with a deterministic byte pattern
    that encodes (slot, M, K) so any wrong axis / wrong gate-vs-up
    assignment / wrong stride shows up as a mismatch.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for tensor allocation")
    _skip_if_no_megamoe_cutedsl_runtime()

    from tensorrt_llm._torch.modules.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    # Pick small shapes that exercise multiple gate/up pairs (n_pairs >= 2)
    # and a non-trivial slot dimension. intermediate must be a multiple of 16.
    num_local_slots = 2
    hidden = 64  # h_bytes = 32 (NVFP4 packed); 4 SF cols per row at sf_vec=16
    intermediate = 32  # n_pairs = 2
    expand_intermediate = 2 * intermediate  # 64
    h_bytes = hidden // 2

    # Encode each byte as ``(slot+1) << 24 | (m+1) << 12 | (k+1) >> 4`` style
    # — we just want every byte to be uniquely traceable. Use a numpy-style
    # arange-based fill instead.
    w3_w1 = torch.zeros(
        (num_local_slots, expand_intermediate, h_bytes), dtype=torch.uint8, device="cuda"
    )
    for s in range(num_local_slots):
        for m in range(expand_intermediate):
            for k in range(h_bytes):
                # 8 bits is too narrow for unique triplets at full shape,
                # but for (2,64,32) we have 4096 cells and only 1 byte; we
                # encode position modulo 251 (largest prime < 256) so
                # collisions are extremely unlikely along the axes we slice.
                w3_w1[s, m, k] = (s * 7919 + m * 251 + k * 19) & 0xFF

    # Parent stores [w3 | w1]: w3 = up_part (first half along M), w1 = gate_part.
    up_part_ref = w3_w1[:, :intermediate, :].clone()
    gate_part_ref = w3_w1[:, intermediate:, :].clone()

    # Minimal stub module that mimics the surface
    # ``_build_mega_format_weights`` reads from.
    mega_fc1_weight = torch.zeros(
        (num_local_slots, expand_intermediate, h_bytes), dtype=torch.uint8, device="cuda"
    )
    mega_fc2_weight = torch.zeros(
        (num_local_slots, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
    )
    # The SF arrays are exercised by other tests; fill with zeros and supply
    # the helper a matching empty SF input via a smaller pattern.
    w2_weight = torch.empty(
        (num_local_slots, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
    )
    w2_weight.random_(0, 256)

    # Build a minimal SF parent matching the expected shape (slots,
    # expand_intermediate, hidden // 16) so the SF leg does not raise.
    sf_cols = hidden // 16
    w3_w1_sf = torch.zeros(
        (num_local_slots, expand_intermediate, sf_cols),
        dtype=torch.uint8,
        device="cuda",
    )
    w2_sf = torch.zeros(
        (num_local_slots, hidden, intermediate // 16),
        dtype=torch.uint8,
        device="cuda",
    )
    # mega_fc1_weight_sf / mega_fc2_weight_sf live at the kernel-side
    # ``round_up`` shape; pass an oversized destination so the SF copy
    # path does not raise.
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        NVFP4MegaMoECuteDslMethod as _Method,
    )

    # ``mega_fc{1,2}_weight_sf`` are 2D ``(num_local_slots, flat_size)``
    # parameters created by ``NVFP4MegaMoECuteDslMethod.create_weights``.
    # The helpers return the per-slot flat size given (intermediate, hidden)
    # / (hidden, intermediate); pass that exact layout here.
    fc1_sf_dst = torch.zeros(
        (num_local_slots, _Method.fc1_sf_flat_size(intermediate, hidden)),
        dtype=torch.uint8,
        device="cuda",
    )
    fc2_sf_dst = torch.zeros(
        (num_local_slots, _Method.fc2_sf_flat_size(hidden, intermediate)),
        dtype=torch.uint8,
        device="cuda",
    )

    class _StubParam:
        def __init__(self, data):
            self.data = data

    class _StubModule:
        def __init__(self):
            self.expert_size_per_partition = num_local_slots
            self.intermediate_size_per_partition = intermediate
            self.hidden_size = hidden
            self.expand_intermediate_size_per_partition = expand_intermediate
            self.w3_w1_weight = _StubParam(w3_w1)
            self.w2_weight = _StubParam(w2_weight)
            self.w3_w1_weight_scale = _StubParam(w3_w1_sf)
            self.w2_weight_scale = _StubParam(w2_sf)
            self.mega_fc1_weight = _StubParam(mega_fc1_weight)
            self.mega_fc2_weight = _StubParam(mega_fc2_weight)
            self.mega_fc1_weight_sf = _StubParam(fc1_sf_dst)
            self.mega_fc2_weight_sf = _StubParam(fc2_sf_dst)

    method = NVFP4MegaMoECuteDslMethod()
    module = _StubModule()
    method._build_mega_format_weights(module)
    torch.cuda.synchronize()

    # ============== FC1 byte-equivalence check ==============
    # Reconstruct the expected layout from gate_part_ref / up_part_ref
    # using the design's documented indexing:
    #   mega[slot, 2*i*16 + j, :]     == gate_part[slot, i*16 + j, :]
    #   mega[slot, (2*i+1)*16 + j, :] == up_part[slot, i*16 + j, :]
    n_pairs = intermediate // 16
    expected_fc1 = torch.empty_like(mega_fc1_weight)
    for s in range(num_local_slots):
        for i in range(n_pairs):
            for j in range(16):
                expected_fc1[s, 2 * i * 16 + j, :] = gate_part_ref[s, i * 16 + j, :]
                expected_fc1[s, (2 * i + 1) * 16 + j, :] = up_part_ref[s, i * 16 + j, :]

    assert torch.equal(mega_fc1_weight, expected_fc1), (
        "mega_fc1_weight byte pattern does not match the "
        "[gate[0:16], up[0:16], gate[16:32], ...] contract documented "
        "in MEGAMOE_CUTEDSL_DESIGN.md 'FC1 gate/up interleave'. Either "
        "the gate <-> up assignment is flipped (gate must be w1 = "
        "w3_w1[intermediate:], up must be w3 = w3_w1[:intermediate]) "
        "or the M-axis stride / chunking is wrong."
    )

    # ============== FC2 byte-equivalence check ==============
    # FC2 is a byte-copy of w2_weight (the kernel boundary applies
    # permute(0,2,1) on consumption; storage is preserved).
    assert torch.equal(mega_fc2_weight, w2_weight), (
        "mega_fc2_weight must be byte-equal to w2_weight (kernel does "
        "the permute at consumption time, not at staging time)."
    )


def test_megamoe_deepgemm_quantize_input_zero_tokens():
    """Regression test for the FusedCommMoEScheduler refactor that now
    calls ``backend.quantize_input`` for zero-token chunks too. The DG
    backend must return its DG-specific empty layout (FP8 + packed-UE8M0
    int32 SF) so the scheduler stays layout-agnostic.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for backend instantiation")

    from tensorrt_llm._torch.modules.fused_moe.mega_moe.mega_moe_deepgemm import (
        MegaMoEDeepGemm as _DG,
    )

    # Build the backend without weights; we only exercise the empty path
    # which short-circuits before the DG kernel is touched.
    routing_method = RenormalizeMoeRoutingMethod(top_k=2)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = 4
    pretrained_config.hidden_size = 1024
    pretrained_config.intermediate_size = 1024
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_DEEPGEMM.value,
        skip_create_weights_in_init=True,
    )

    # The DG capability probe needs the bundled deep_gemm module. Skip if
    # not available; the empty-path semantic is then unreachable here but
    # documented behaviour is unaffected.
    try:
        backend = _DG(
            routing_method=routing_method,
            num_experts=4,
            hidden_size=1024,
            intermediate_size=1024,
            dtype=torch.bfloat16,
            model_config=model_config,
            init_load_balancer=False,
        )
    except Exception as e:
        pytest.skip(f"MegaMoEDeepGemm cannot be instantiated on this host: {e}")

    empty_x = torch.empty((0, 1024), dtype=torch.bfloat16, device="cuda")
    x_fp8, x_sf = backend.quantize_input(empty_x)
    assert x_fp8.shape[0] == 0
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert x_sf.shape[0] == 0
    assert x_sf.dtype == torch.int32
    assert x_sf.shape[1] == 1024 // 128
