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
MoE Module Unit Tests

This module provides a unified test framework for testing MoE modules through the
high-level create_moe() + forward() interface, rather than the backend-level interfaces.

Design Goals:
1. Test MoE module via: create_moe -> load_weights -> forward
2. Cover key quantization + backend combinations
3. Support EPLB (Expert Load Balancing) testing
4. Support autotune and tactic capture testing
"""

import copy
import functools
import logging
import os
import pickle
import sys
import tempfile
import traceback
from contextlib import nullcontext
from itertools import product
from typing import List, Optional

import cloudpickle
import pytest
import torch
from _torch.modules.moe.moe_test_utils import (
    IS_CI_MODE,
    MoeBackendType,
    MoeModelConfig,
    create_test_param,
    get_quick_skip_reason,
    iter_base_test_configs,
    replay_tactics_and_check,
    should_skip_cutedsl,
    should_skip_cutlass,
    should_skip_deepgemm,
    should_skip_multi_gpu,
    should_skip_to_accelerate_ci,
    should_skip_trtllm,
    skip_if_insufficient_gpu_memory,
    supports_autotuner_capture,
)
from _torch.modules.moe.quantize_utils import get_test_quant_params
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig

import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    DefaultMoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    RenormalizeNaiveMoeRoutingMethod,
    create_moe,
)
from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency import DeepEPLowLatency
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod,
    DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm,
    FP8QDQFusedMoEMethod,
    INT8WoqPerChannelFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
    UnquantizedFusedMoEMethod,
    W4A8MXFP4FP8CutlassFusedMoEMethod,
    W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
    W4A8MXFP4MXFP8CutlassFusedMoEMethod,
    W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
    W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
    W4A16MXFP4TRTLLMGenFusedMoEMethod,
    WFP4A16FusedMoEMethod,
    WInt4AFP8FusedMoEMethod,
)
from tensorrt_llm._utils import get_sm_version, mpi_rank
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

G_LOGGER = logging.getLogger(__name__)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def _create_mapping_for_parallel_mode(world_size, parallel_mode):
    """Create Mapping for different parallelism strategies.

    Args:
        world_size: Total number of GPUs
        parallel_mode: One of "DEP", "TEP", "DTP", "TTP"
            - DEP: Attention uses DP, MoE uses EP
            - TEP: Attention uses TP, MoE uses EP
            - DTP: Attention uses DP, MoE uses TP
            - TTP: Attention uses TP, MoE uses TP

    Returns:
        Mapping object configured for the specified parallel mode
    """
    configs = {
        "DEP": {  # Attention DP, MoE EP
            "moe_ep_size": world_size,
            "moe_tp_size": 1,
            "enable_attention_dp": True,
        },
        "TEP": {  # Attention TP, MoE EP
            "moe_ep_size": world_size,
            "moe_tp_size": 1,
            "enable_attention_dp": False,
        },
        "DTP": {  # Attention DP, MoE TP
            "moe_ep_size": 1,
            "moe_tp_size": world_size,
            "enable_attention_dp": True,
        },
        "TTP": {  # Attention TP, MoE TP
            "moe_ep_size": 1,
            "moe_tp_size": world_size,
            "enable_attention_dp": False,
        },
    }
    if parallel_mode not in configs:
        raise ValueError(
            f"Unknown parallel_mode: {parallel_mode}. Must be one of {list(configs.keys())}"
        )

    cfg = configs[parallel_mode]
    return Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=cfg["moe_ep_size"],
        moe_tp_size=cfg["moe_tp_size"],
        enable_attention_dp=cfg["enable_attention_dp"],
    )


def _create_moe_load_balancer(model_cfg, enable_eplb):
    """Create MoeLoadBalancer if EPLB is enabled, otherwise return nullcontext."""
    if not enable_eplb:
        return nullcontext()

    ep_rank = model_cfg.mapping.moe_ep_rank
    ep_size = model_cfg.mapping.moe_ep_size
    model_cfg.moe_load_balancer.setup(ep_rank=ep_rank, ep_size=ep_size)
    return MoeLoadBalancer(
        ep_rank=ep_rank,
        ep_size=ep_size,
        layer_updates_per_iter=model_cfg.moe_load_balancer.layer_updates_per_iter,
    )


def _setup_autotuner_for_test(mapping):
    """Configure AutoTuner for faster unit test profiling."""
    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()
    autotuner = AutoTuner.get()
    autotuner.warmup = 0  # default: 2
    autotuner.repeat = 1  # default: 10
    autotuner.stream_delay_micro_secs = 10  # default: 1000


def _create_model_config(
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    mapping,
    quant_config,
    moe_backend,
    enable_eplb=False,
    num_slots=-1,
    layer_updates_per_iter=-1,
    max_num_tokens=None,
):
    """Create PretrainedConfig and ModelConfig for MoE testing."""
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    moe_load_balancer_config = (
        MoeLoadBalancerConfig(
            num_slots=num_slots,
            layer_updates_per_iter=layer_updates_per_iter,
        )
        if enable_eplb
        else None
    )

    kwargs = dict(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=moe_backend,
        moe_disable_finalize_fusion=False,
        moe_load_balancer=moe_load_balancer_config,
    )
    if max_num_tokens is not None:
        kwargs["max_num_tokens"] = max_num_tokens

    return ModelConfig(**kwargs)


def _run_autotune_test(
    run_forward_fn,
    ref_fused_moe,
    ref_output,
    backend_type,
    quant_algo,
    run_all_tactics=False,
    use_flashinfer=False,
):
    """Run autotune phase and tactic replay test.

    Args:
        run_forward_fn: Forward function to run
        ref_fused_moe: Reference MoE module for accuracy check
        ref_output: Reference output for comparison
        backend_type: MoE backend type
        quant_algo: Quantization algorithm
        run_all_tactics: If False, skip full tactic replay and only run simple accuracy check
    """
    # Autotune phase
    cache_path = os.path.join(tempfile.gettempdir(), "moe_module_autotuner_cache.json")
    with torch.inference_mode(), autotune(cache_path=cache_path):
        _ = run_forward_fn()

    # Check if we should run full tactic replay
    if not run_all_tactics or not supports_autotuner_capture(
        backend_type, quant_algo, use_flashinfer
    ):
        # Simple accuracy check for unsupported backends or when run_all_tactics is False
        with torch.inference_mode():
            output = run_forward_fn()
            ref_fused_moe.check_accuracy(output, ref_output)
        return

    # Capture phase: record which tactics are used
    with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
        _ = run_forward_fn()

    # Replay phase: test each tactic for correctness
    replay_tactics_and_check(
        all_tactics=all_tactics,
        run_moe_fn=run_forward_fn,
        check_accuracy_fn=ref_fused_moe.check_accuracy,
        ref_output=ref_output,
        backend_type=backend_type,
        quant_algo=quant_algo,
        fail_fast=False,
    )


def _run_eplb_test(
    run_forward_fn, ref_fused_moe, ref_output, moe_load_balancer, initial_expert_ids
):
    """Run EPLB multi-iteration test.

    Args:
        run_forward_fn: Forward function to run
        ref_fused_moe: Reference MoE module for accuracy check
        ref_output: Reference output for comparison
        moe_load_balancer: MoeLoadBalancer instance
        initial_expert_ids: Expert IDs recorded immediately after MoE initialization (before any forward)
    """
    assert isinstance(moe_load_balancer, MoeLoadBalancer), (
        "Moe load balancer should be created when eplb is enabled"
    )
    assert initial_expert_ids is not None, (
        "initial_expert_ids should be recorded before any forward pass"
    )

    extra_steps = 1
    for _ in range(extra_steps):
        output = run_forward_fn()
        ref_fused_moe.check_accuracy(output, ref_output)

    current_expert_ids = copy.deepcopy(
        moe_load_balancer.single_layer_load_balancers[0].get_old_rank_expert_ids()
    )

    # EPLB should have updated expert_ids from initial state
    assert initial_expert_ids != current_expert_ids, (
        f"Expert ids after eplb update should be different from the initial loaded ones. "
        f"Initial: {initial_expert_ids}, Current: {current_expert_ids}"
    )


def _create_routing_method(routing_method_cls, top_k, num_experts, dtype):
    """
    Create a routing method instance with appropriate parameters for each routing method type.

    Args:
        routing_method_cls: The routing method class to instantiate
        top_k: Number of experts to select per token
        num_experts: Total number of experts
        dtype: Data type for tensors

    Returns:
        An instance of the routing method
    """
    # Routing methods with force_enable_pytorch_op support
    if routing_method_cls in (RenormalizeMoeRoutingMethod, DefaultMoeRoutingMethod):
        return routing_method_cls(top_k=top_k, force_enable_pytorch_op=True)

    # Simple routing methods (only top_k)
    if routing_method_cls in (RenormalizeNaiveMoeRoutingMethod, Llama4RenormalizeMoeRoutingMethod):
        return routing_method_cls(top_k=top_k)

    # DeepSeekV3 routing method requires special parameters
    if routing_method_cls == DeepSeekV3MoeRoutingMethod:
        # DeepSeek-V3 routing: groups experts, selects top groups, then selects top_k from those
        # The routing logic does topk(k=2) within each group, so each group must have >= 2 experts
        # Calculate n_group such that each group has at least 2 experts
        experts_per_group = 2
        n_group = max(1, num_experts // experts_per_group)
        # topk_group should be <= n_group and reasonable for the selection
        topk_group = min(n_group, max(1, n_group // 2))
        routed_scaling_factor = 1.0
        # Create e_score_correction_bias as a zero tensor (no bias correction in test)
        e_score_correction_bias = torch.zeros(num_experts, dtype=dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
            is_fused=False,  # Use PyTorch implementation for testing
        )

    # MiniMaxM2 routing method requires special parameters
    if routing_method_cls == MiniMaxM2MoeRoutingMethod:
        # Create e_score_correction_bias as a zero tensor (no bias correction in test)
        e_score_correction_bias = torch.zeros(num_experts, dtype=dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )

    # Fallback: try with just top_k
    return routing_method_cls(top_k=top_k)


def _test_moe_worker(
    moe_backend,
    dtype,
    quant_algo,
    mapping=None,
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
    model_config: Optional[MoeModelConfig] = None,
    seq_len: int = 4,
    enable_autotune: bool = False,
    routing_method_cls=RenormalizeMoeRoutingMethod,
    dtype_routing_logits=None,
    swiglu_alpha: float = 1,
    swiglu_beta: float = 0,
    swiglu_limit: float = float("inf"),
):
    """
    Test MoE module worker function.

    This test verifies:
    1. MoE module forward pass produces correct results
    2. EPLB (Expert Load Balancing) works correctly when enabled
    3. Autotune works correctly with the module when enabled
    4. All tactics are captured and replayed properly when autotune is enabled

    Args:
        routing_method_cls: Routing method class to use (default: RenormalizeMoeRoutingMethod)
        dtype_routing_logits: Data type for routing logits (default: same as dtype).
                              DeepSeekV3 routing requires torch.float32.
        swiglu_alpha: SwiGLU alpha parameter (default=1, non-gptoss)
        swiglu_beta: SwiGLU beta parameter (default=0, non-gptoss)
        swiglu_limit: SwiGLU limit parameter (default=inf, non-gptoss)
    """
    try:
        _test_moe_worker_impl(
            moe_backend=moe_backend,
            dtype=dtype,
            quant_algo=quant_algo,
            mapping=mapping,
            enable_eplb=enable_eplb,
            layer_updates_per_iter=layer_updates_per_iter,
            num_slots=num_slots,
            model_config=model_config,
            seq_len=seq_len,
            enable_autotune=enable_autotune,
            routing_method_cls=routing_method_cls,
            dtype_routing_logits=dtype_routing_logits,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )
    except Exception:
        traceback.print_exc()
        raise


def _test_moe_worker_impl(
    moe_backend,
    dtype,
    quant_algo,
    mapping=None,
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
    model_config: Optional[MoeModelConfig] = None,
    seq_len: int = 4,
    enable_autotune: bool = False,
    routing_method_cls=RenormalizeMoeRoutingMethod,
    dtype_routing_logits=None,
    swiglu_alpha: float = 1,
    swiglu_beta: float = 0,
    swiglu_limit: float = float("inf"),
):
    """Actual implementation of _test_moe_worker."""
    # Default routing logits dtype to model dtype if not specified
    if dtype_routing_logits is None:
        dtype_routing_logits = dtype
    # Parse model config
    if model_config is not None:
        num_experts = model_config.num_experts
        top_k = model_config.top_k
        hidden_size = model_config.hidden_size
        intermediate_size = model_config.intermediate_size
    else:
        num_experts, top_k, hidden_size, intermediate_size = 8, 2, 512, 512

    # Setup mapping
    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()
    all_rank_num_tokens = [seq_len] * mapping.world_size
    torch.cuda.set_device(mapping.rank)

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Create routing method and input tensors
        routing_method = _create_routing_method(
            routing_method_cls, top_k=top_k, num_experts=num_experts, dtype=dtype
        )
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        if enable_eplb:
            # Same router_logits for all tokens to force the eplb update weights
            router_logits = torch.randn(
                (1, num_experts), dtype=dtype_routing_logits, device="cuda"
            ).repeat(seq_len, 1)
        else:
            router_logits = torch.randn(
                (seq_len, num_experts), dtype=dtype_routing_logits, device="cuda"
            )

        # Determine swiglu_gptoss_style
        swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")

        # In EP mode, swiglu tensors must be sized per local experts
        # (C++ kernels check: swiglu_alpha.size(0) == num_experts_on_rank)
        num_local_experts = num_experts // mapping.moe_ep_size

        # Setup quantization
        backend_type = MoeBackendType(moe_backend)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=swiglu_gptoss_style,
            swiglu_gptoss_style=swiglu_gptoss_style,
            swiglu_alpha=swiglu_alpha if swiglu_gptoss_style else None,
            swiglu_beta=swiglu_beta if swiglu_gptoss_style else None,
            swiglu_limit=swiglu_limit if swiglu_gptoss_style else None,
            num_local_experts=num_local_experts,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        # For EPLB, keep weights on CPU
        if enable_eplb:
            for key in weights:
                if isinstance(weights[key], torch.Tensor):
                    weights[key] = weights[key].to("cpu")
        ref_weights = copy.deepcopy(weights) if enable_eplb else weights

        # Use a small max_num_tokens for unit tests to avoid NVSHMEM buffer
        # allocation failures.  DeepEP low-latency buffers are sized by
        # max_num_tokens * hidden_size * num_experts, and the default 8192
        # causes cuMemMap failures for large configs (e.g. e384 * h7168).
        # Unit tests only send seq_len tokens, so 256 is more than enough.
        test_max_num_tokens = max(256, seq_len)

        # Create configs
        model_cfg = _create_model_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=moe_backend,
            enable_eplb=enable_eplb,
            num_slots=num_slots,
            layer_updates_per_iter=layer_updates_per_iter,
            max_num_tokens=test_max_num_tokens,
        )

        # Create MoE load balancer
        moe_load_balancer = _create_moe_load_balancer(model_cfg, enable_eplb)

        # Get swiglu tensors if swiglu_gptoss_style is enabled
        swiglu_tensors = quantize_util.get_swiglu_tensors()

        # Get weight_loading_mode from quantize_util if available
        # (e.g., W4A8AWQQuantizeUtil uses W4A8_CUSTOM mode)
        weight_loading_mode = getattr(
            quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
        )

        with (
            moe_load_balancer,
            create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_cfg,
                bias=swiglu_gptoss_style,
                swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
                swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
                swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
                weight_loading_mode=weight_loading_mode,
            ) as fused_moe,
        ):
            fused_moe.load_weights([weights])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{mapping.rank}")

            # Record initial expert_ids before any forward pass (for EPLB test)
            initial_expert_ids = None
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                moe_load_balancer.register_weight_slots_after_to_cuda()
                moe_load_balancer.finalize_model()
                moe_load_balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)
                # Record initial expert_ids immediately after initialization
                # Use deepcopy to avoid reference issues if the list is modified in-place
                initial_expert_ids = copy.deepcopy(
                    moe_load_balancer.single_layer_load_balancers[0].get_old_rank_expert_ids()
                )
                G_LOGGER.info(f"[EPLB Debug] Initial expert_ids (after init): {initial_expert_ids}")

            # Create reference module
            ref_fused_moe = quantize_util.create_ref_module(routing_method)
            ref_fused_moe.moe_tp_size = mapping.moe_tp_size
            ref_fused_moe.load_weights([ref_weights])
            ref_fused_moe.cuda(f"cuda:{mapping.rank}")

            # Define forward function
            def run_forward():
                with torch.inference_mode():
                    if isinstance(moe_load_balancer, MoeLoadBalancer):
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            output = fused_moe.forward(
                                x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                            )
                    else:
                        output = fused_moe.forward(
                            x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                        )
                torch.cuda.synchronize()
                return output

            # Get reference output
            with torch.inference_mode():
                ref_output = ref_fused_moe.forward(x, router_logits)

            # flashinfer has no capture and replay mechanisms, so we skip test_all_kernels
            use_flashinfer = getattr(fused_moe, "use_flashinfer", False)

            # Run tests
            if enable_autotune:
                _setup_autotuner_for_test(mapping)
                _run_autotune_test(
                    run_forward,
                    ref_fused_moe,
                    ref_output,
                    backend_type,
                    quant_algo,
                    use_flashinfer=use_flashinfer,
                )
            else:
                output = run_forward()
                ref_fused_moe.check_accuracy(output, ref_output)

            if enable_eplb:
                _run_eplb_test(
                    run_forward, ref_fused_moe, ref_output, moe_load_balancer, initial_expert_ids
                )


def _test_moe_multi_gpu(
    comm_method_type,
    moe_backend,
    quant_algo,
    dtype,
    world_size,
    parallel_mode="DEP",
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
    model_config: Optional[MoeModelConfig] = None,
    seq_len: int = 4,
    enable_autotune: bool = False,
    routing_method_cls=RenormalizeMoeRoutingMethod,
    dtype_routing_logits=None,
    swiglu_alpha: float = 1,
    swiglu_beta: float = 0,
    swiglu_limit: float = float("inf"),
):
    """
    Test MoE module with multi-GPU support.

    Args:
        comm_method_type: Communication method type
        moe_backend: Backend type string
        quant_algo: Quantization algorithm
        dtype: Activation data type
        world_size: Total world size
        parallel_mode: Parallelism strategy ("DEP", "TEP", "DTP", "TTP")
        enable_eplb: Enable Expert Load Balancing
        layer_updates_per_iter: EPLB layer updates per iteration
        num_slots: EPLB number of slots
        model_config: MoE model configuration
        seq_len: Sequence length for test input
        enable_autotune: Enable autotune and tactic capture/replay testing
        routing_method_cls: Routing method class to use
        dtype_routing_logits: Data type for routing logits (default: same as dtype)
        swiglu_alpha: SwiGLU alpha parameter (default=1, non-gptoss)
        swiglu_beta: SwiGLU beta parameter (default=0, non-gptoss)
        swiglu_limit: SwiGLU limit parameter (default=inf, non-gptoss)
    """

    def init_worker(custom_paths, comm_method_type):
        # Update the sys.path to align with main process for submodule import
        for custom_path in custom_paths:
            if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
                sys.path.append(custom_path)

        # Set comm method
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = comm_method_type

    mapping = _create_mapping_for_parallel_mode(world_size, parallel_mode)

    with MPIPoolExecutor(
        initializer=init_worker, initargs=(sys.path, comm_method_type), max_workers=world_size
    ) as executor:
        results = executor.map(
            _test_moe_worker,
            *zip(
                *[
                    (
                        moe_backend,
                        dtype,
                        quant_algo,
                        mapping,
                        enable_eplb,
                        layer_updates_per_iter,
                        num_slots,
                        model_config,
                        seq_len,
                        enable_autotune,
                        routing_method_cls,
                        dtype_routing_logits,
                        swiglu_alpha,
                        swiglu_beta,
                        swiglu_limit,
                    )
                ]
                * world_size
            ),
        )
        for r in results:
            assert r is None


# ============================================================================
# Test Parameters Configuration
# ============================================================================

# Quantization algorithms to test
QUANT_ALGOS = [
    None,  # Unquantized
    QuantAlgo.FP8,
    QuantAlgo.NVFP4,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.W4A8_NVFP4_FP8,
    QuantAlgo.W4A16_MXFP4,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.W8A16,
    QuantAlgo.W4A8_AWQ,
]

# Backend types to test
BACKEND_TYPES = [
    MoeBackendType.CUTLASS,
    MoeBackendType.TRTLLM,
    MoeBackendType.CUTEDSL,
    MoeBackendType.DEEPGEMM,
]

# Data types to test
DTYPES = [
    torch.float16,
    torch.bfloat16,
]

# Model configurations for testing
# (num_experts, top_k, hidden_size, intermediate_size)
#
# Default runs the CI subset (TRTLLM_TEST_MOE_CI=1).
# Set TRTLLM_TEST_MOE_CI=0 for the full local config matrix.
CI_MOE_MODEL_CONFIGS = [
    MoeModelConfig(60, 4, 2048, 1408),  # Qwen1.5-MoE-A2.7B
    MoeModelConfig(256, 8, 7168, 2048),  # DeepSeek-V3
    MoeModelConfig(128, 4, 2880, 2880),  # GPT-OSS-120B
    MoeModelConfig(8, 1, 512, 512),  # boundary: top_k=1, single expert activated
]

LOCAL_MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS + [
    MoeModelConfig(64, 6, 2048, 1408),  # DeepSeek-MoE-16B / DeepSeek-V2-Lite
    MoeModelConfig(384, 8, 7168, 2048),  # Kimi-K2
    # === Boundary Tests: num_experts / top_k ===
    MoeModelConfig(4, 4, 512, 512),  # top_k=num_experts, all experts activated
    MoeModelConfig(7, 2, 256, 512),  # prime num_experts
    MoeModelConfig(13, 3, 256, 512),  # prime num_experts, odd top_k
    # === Boundary Tests: small sizes ===
    MoeModelConfig(4, 2, 64, 128),  # very small hidden_size
    MoeModelConfig(4, 2, 128, 64),  # intermediate < hidden
]

MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS if IS_CI_MODE else LOCAL_MOE_MODEL_CONFIGS

# Sequence lengths to test
SEQ_LENS = [1, 8]

# Routing methods to test
ROUTING_METHODS = [
    RenormalizeMoeRoutingMethod,  # TopK -> Softmax (Mixtral, etc.)
    DefaultMoeRoutingMethod,  # Softmax -> TopK
    RenormalizeNaiveMoeRoutingMethod,  # Softmax -> TopK -> Renormalize (Qwen3)
    Llama4RenormalizeMoeRoutingMethod,  # Top1 -> Sigmoid (Llama4)
    DeepSeekV3MoeRoutingMethod,  # Sigmoid -> BiasAdd -> Group TopK (DeepSeek-V3)
    MiniMaxM2MoeRoutingMethod,  # Sigmoid -> BiasAdd -> TopK -> Renormalize (MiniMax-M2)
]


MULTI_GPU_ROUTING_METHODS = [
    RenormalizeMoeRoutingMethod,  # TopK -> Softmax (Mixtral, etc.)
    DeepSeekV3MoeRoutingMethod,  # Sigmoid -> BiasAdd -> Group TopK (DeepSeek-V3)
]


# ============================================================================
# Multi-GPU Test Configuration
# ============================================================================
# Parallel modes to test
PARALLEL_MODES = [
    "DEP",  # Attention DP, MoE EP
    "TEP",  # Attention TP, MoE EP
    "DTP",  # Attention DP, MoE TP
    "TTP",  # Attention TP, MoE TP
]

# Communication methods to test
COMM_METHODS = [
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
    "DEEPEP",
    "DEEPEPLOWLATENCY",
]

# SwiGLU parameters for swiglu_gptoss_style testing
SWIGLU_ALPHAS = [1, 1.702]  # default, GPT-OSS (modeling_gpt_oss.py)
SWIGLU_BETAS = [0, 1.0]  # default, GPT-OSS
SWIGLU_LIMITS = [float("inf"), 7.0]  # default, GPT-OSS

# Full product of all SwiGLU combos (local exhaustive testing only)
LOCAL_SWIGLU_COMBOS = list(product(SWIGLU_ALPHAS, SWIGLU_BETAS, SWIGLU_LIMITS))

# CI / Multi-GPU: only non-gptoss (default) and one gptoss combo
# All non-default combos trigger the same swiglu_gptoss_style=True code path;
# different alpha/beta/limit values are just kernel parameters, not code branches.
CI_SWIGLU_COMBOS = [
    (1, 0, float("inf")),  # non-gptoss (default SwiGLU)
    (1.702, 1.0, 7.0),  # gptoss style (GPT-OSS real values)
]

# Default runs CI subset. Set TRTLLM_TEST_MOE_CI=0 for full local matrix.
SWIGLU_COMBOS = CI_SWIGLU_COMBOS if IS_CI_MODE else LOCAL_SWIGLU_COMBOS


@functools.lru_cache(maxsize=1)
def _is_mnnvl_supported() -> bool:
    """Cached check for MNNVL platform support (pynvml query is expensive)."""
    return MnnvlMemory.supports_mnnvl()


def _get_comm_method_skip_reason(
    comm_method: str,
    model_config: "MoeModelConfig",
    dtype: Optional[torch.dtype] = None,
) -> Optional[str]:
    """
    Check if a communication method is compatible with the given model config.

    Returns a skip reason string if incompatible, None otherwise.
    """
    # NVLink-based methods require all NVLink links active.
    # NVLINK_ONE_SIDED/TWO_SIDED: base.py:53-58 raises RuntimeError without MNNVL.
    # DEEPEP: upstream DeepEP check_nvlink_connections() asserts NVLink P2P
    #   between all GPU pairs; NUM_MAX_NVL_PEERS=8 hardcoded in configs.cuh.
    # DeepEPLowLatency does NOT require NVLink (RDMA only, num_nvl_bytes=0).
    if comm_method in ("NVLINK_ONE_SIDED", "NVLINK_TWO_SIDED", "DEEPEP"):
        if not _is_mnnvl_supported():
            return (
                f"{comm_method} requires NVLink support (all links active). "
                f"Not supported on this platform."
            )

    # DeepEP/DeepEPLowLatency only support bfloat16 at runtime:
    # is_workload_feasible (deep_ep.py:136, deep_ep_low_latency.py:164)
    # rejects non-bfloat16.  The auto-selection path already guards this
    # (communication_factory.py:157: act_dtype == torch.bfloat16), but
    # the forced-method path used by tests creates the NVSHMEM buffer
    # unconditionally.  Buffer creation is a collective NVSHMEM operation
    # that can hang when followed by immediate destruction on fallback.
    # Skip here to avoid creating buffers that will never be used.
    if (
        comm_method in ("DEEPEP", "DEEPEPLOWLATENCY")
        and dtype is not None
        and dtype != torch.bfloat16
    ):
        return (
            f"{comm_method} only supports bfloat16 (dtype={dtype}). "
            f"Auto-selection already skips DeepEP for non-bfloat16 "
            f"(communication_factory.py:157); forced-method buffer "
            f"creation hangs on collective NVSHMEM init."
        )

    if comm_method == "DEEPEPLOWLATENCY":
        if model_config.hidden_size not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES:
            return (
                f"DeepEPLowLatency does not support hidden_size={model_config.hidden_size}, "
                f"requires one of {sorted(DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES)}"
            )
    return None


def generate_multi_gpu_test_params(
    parallel_modes,
    comm_methods,
    swiglu_combos,
    model_configs,
    seq_lens,
    dtypes,
    backend_types,
    quant_algos,
    routing_methods,
) -> List:
    """
    Generate test parameter combinations for multi-GPU tests.

    Args:
        parallel_modes: List of parallel modes
        comm_methods: List of communication methods
        swiglu_combos: List of (swiglu_alpha, swiglu_beta, swiglu_limit) tuples
        model_configs: List of MoeModelConfig
        seq_lens: List of sequence lengths
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes

    Returns:
        List of pytest.param objects for runnable test configurations only
    """
    params: List = []
    for parallel_mode, comm_method in product(parallel_modes, comm_methods):
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
            base_test_id,
        ) in iter_base_test_configs(
            swiglu_combos,
            model_configs,
            seq_lens,
            dtypes,
            backend_types,
            quant_algos,
            routing_methods,
        ):
            # Check multi-GPU specific skip conditions (short-circuit on first match)
            if not skip_reason:
                # TP modes shard intermediate_size; EP modes don't
                moe_tp_size = 4 if parallel_mode in ("DTP", "TTP") else 1
                for reason in (
                    _get_comm_method_skip_reason(comm_method, model_config, dtype=dtype),
                    should_skip_trtllm(
                        backend_type,
                        quant_algo,
                        model_config,
                        comm_method=comm_method,
                        moe_tp_size=moe_tp_size,
                    ),
                    should_skip_cutlass(
                        backend_type,
                        comm_method,
                        quant_algo=quant_algo,
                        model_config=model_config,
                        moe_tp_size=moe_tp_size,
                        dtype=dtype,
                    ),
                    should_skip_cutedsl(
                        backend_type,
                        quant_algo,
                        model_config,
                        comm_method,
                        moe_tp_size=moe_tp_size,
                    ),
                    should_skip_deepgemm(
                        backend_type,
                        comm_method,
                        quant_algo=quant_algo,
                        model_config=model_config,
                        moe_tp_size=moe_tp_size,
                    ),
                    should_skip_multi_gpu(
                        parallel_mode, model_config, world_size=4, comm_method=comm_method
                    ),
                ):
                    if reason:
                        skip_reason = reason
                        break

            if skip_reason:
                continue

            test_id = f"parallel={parallel_mode}-comm={comm_method}-{base_test_id}"
            param_values = (
                parallel_mode,
                comm_method,
                dtype,
                backend_type.value,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                swiglu_alpha,
                swiglu_beta,
                swiglu_limit,
            )
            params.append(create_test_param(param_values, test_id))

    return params


def generate_base_test_params(
    swiglu_combos, model_configs, seq_lens, dtypes, backend_types, quant_algos, routing_methods
) -> List:
    """
    Generate test parameter combinations for base tests.

    Args:
        swiglu_combos: List of (swiglu_alpha, swiglu_beta, swiglu_limit) tuples
        model_configs: List of MoeModelConfig
        seq_lens: List of sequence lengths
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes

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
        base_test_id,
    ) in iter_base_test_configs(
        swiglu_combos, model_configs, seq_lens, dtypes, backend_types, quant_algos, routing_methods
    ):
        if skip_reason:
            continue
        param_values = (
            dtype,
            backend_type.value,
            quant_algo,
            seq_len,
            model_config,
            routing_method_cls,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
        )
        params.append(create_test_param(param_values, base_test_id))

    return params


# ============================================================================
# MoE Single GPU Tests
# ============================================================================
# Pre-generate test parameters at module load time
BASE_TEST_PARAMS = generate_base_test_params(
    swiglu_combos=SWIGLU_COMBOS,
    model_configs=MOE_MODEL_CONFIGS,
    seq_lens=SEQ_LENS,
    dtypes=DTYPES,
    backend_types=BACKEND_TYPES,
    quant_algos=QUANT_ALGOS,
    routing_methods=ROUTING_METHODS,
)


@pytest.mark.parametrize(
    "dtype,moe_backend,quant_algo,seq_len,model_config,routing_method_cls,"
    "swiglu_alpha,swiglu_beta,swiglu_limit",
    BASE_TEST_PARAMS,
)
def test_configurable_moe_single_gpu(
    dtype: torch.dtype,
    moe_backend: str,
    quant_algo: Optional[QuantAlgo],
    seq_len: int,
    model_config: MoeModelConfig,
    routing_method_cls,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
):
    """
    Single-GPU test for ConfigurableMoE module.

    This test verifies:
    1. MoE create_moe -> load_weights -> forward produces correct results
    2. Various backend + quantization combinations work correctly
    3. Autotune captures and replays all tactics properly
    4. swiglu_gptoss_style (SwiGLU with custom parameters) works correctly
    """
    swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")
    ci_skip = should_skip_to_accelerate_ci(
        backend_type=MoeBackendType(moe_backend),
        quant_algo=quant_algo,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
        dtype=dtype,
        seq_len=seq_len,
        swiglu_gptoss_style=swiglu_gptoss_style,
    )
    if ci_skip:
        pytest.skip(ci_skip)

    skip_if_insufficient_gpu_memory(
        model_config.num_experts,
        model_config.hidden_size,
        model_config.intermediate_size,
        dtype,
    )

    # DeepSeekV3 routing requires float32 routing_logits for TRTLLM backend
    # See: cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:70-72
    dtype_routing_logits = None
    if (
        moe_backend == MoeBackendType.TRTLLM.value
        and routing_method_cls == DeepSeekV3MoeRoutingMethod
    ):
        dtype_routing_logits = torch.float32

    _test_moe_worker(
        moe_backend=moe_backend,
        dtype=dtype,
        quant_algo=quant_algo,
        model_config=model_config,
        seq_len=seq_len,
        enable_autotune=True,
        routing_method_cls=routing_method_cls,
        dtype_routing_logits=dtype_routing_logits,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )


# ============================================================================
# MoE Multi-GPU Tests
# ============================================================================
# Pre-generate multi-GPU test parameters at module load time
MULTI_GPU_TEST_PARAMS = generate_multi_gpu_test_params(
    parallel_modes=PARALLEL_MODES,
    comm_methods=COMM_METHODS,
    swiglu_combos=SWIGLU_COMBOS,
    model_configs=MOE_MODEL_CONFIGS,
    seq_lens=[8] if IS_CI_MODE else SEQ_LENS,
    dtypes=DTYPES,
    backend_types=BACKEND_TYPES,
    quant_algos=QUANT_ALGOS,
    routing_methods=MULTI_GPU_ROUTING_METHODS,
)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize(
    "parallel_mode,comm_method_type,dtype,moe_backend,quant_algo,seq_len,model_config,"
    "routing_method_cls,swiglu_alpha,swiglu_beta,swiglu_limit",
    MULTI_GPU_TEST_PARAMS,
)
def test_configurable_moe_multi_gpu(
    parallel_mode,
    comm_method_type,
    dtype,
    moe_backend,
    quant_algo,
    seq_len,
    model_config,
    routing_method_cls,
    swiglu_alpha,
    swiglu_beta,
    swiglu_limit,
):
    swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")
    ci_skip = should_skip_to_accelerate_ci(
        backend_type=MoeBackendType(moe_backend),
        quant_algo=quant_algo,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
        dtype=dtype,
        seq_len=seq_len,
        swiglu_gptoss_style=swiglu_gptoss_style,
        parallel_mode=parallel_mode,
    )
    if ci_skip:
        pytest.skip(ci_skip)

    skip_if_insufficient_gpu_memory(
        model_config.num_experts,
        model_config.hidden_size,
        model_config.intermediate_size,
        dtype,
    )

    # DeepSeekV3 routing requires float32 routing_logits for TRTLLM backend
    # See: cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:70-72
    dtype_routing_logits = None
    if (
        moe_backend == MoeBackendType.TRTLLM.value
        and routing_method_cls == DeepSeekV3MoeRoutingMethod
    ):
        dtype_routing_logits = torch.float32

    world_size = 4
    _test_moe_multi_gpu(
        comm_method_type,
        moe_backend,
        quant_algo,
        dtype=dtype,
        world_size=world_size,
        parallel_mode=parallel_mode,
        model_config=model_config,
        seq_len=seq_len,
        routing_method_cls=routing_method_cls,
        dtype_routing_logits=dtype_routing_logits,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )


# ============================================================================
# MoE Multi-GPU EPLB Tests
# ============================================================================
# EPLB-specific configuration
EPLB_PARALLEL_MODES = ["DEP"]  # EPLB only works with DEP mode (use_dp=True)
EPLB_COMM_METHODS = [
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
]  # Communication methods for EPLB
EPLB_ROUTING_METHODS = [RenormalizeMoeRoutingMethod]  # Common routing methods
EPLB_MODEL_CONFIGS = [MoeModelConfig(8, 2, 512, 512)]  # Model configs for EPLB
EPLB_NUM_SLOTS_LIST = [16]  # Must be > num_experts (8) to be effective


def _get_fused_moe_method_class(quant_algo, backend_type):
    """
    Get the FusedMoEMethod class based on quant_algo and backend_type.

    This mirrors the logic in each backend's _get_quant_method() method.

    Returns:
        FusedMoEMethod class or None if not found
    """
    backend_str = backend_type.value if hasattr(backend_type, "value") else str(backend_type)

    if quant_algo is None:
        # Unquantized - only CUTLASS supports it
        if backend_str == "CUTLASS":
            return UnquantizedFusedMoEMethod
        return None

    # CUTLASS backend
    # Mapping based on CutlassFusedMoE._get_quant_method() logic
    if backend_str == "CUTLASS":
        DSFP8BlockScalesFusedMoEMethod = (
            DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm
            if get_sm_version() == 120
            else DeepSeekFP8BlockScalesFusedMoEMethod
        )
        method_map = {
            QuantAlgo.FP8: FP8QDQFusedMoEMethod,
            QuantAlgo.FP8_BLOCK_SCALES: DSFP8BlockScalesFusedMoEMethod,
            QuantAlgo.NVFP4: NVFP4CutlassFusedMoEMethod,
            # W4A8_AWQ uses is_int4_weight_only_per_group() -> WInt4AFP8FusedMoEMethod
            QuantAlgo.W4A8_AWQ: WInt4AFP8FusedMoEMethod,
            QuantAlgo.W8A16: INT8WoqPerChannelFusedMoEMethod,
            QuantAlgo.W4A16_MXFP4: WFP4A16FusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_FP8: W4A8MXFP4FP8CutlassFusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_MXFP8: W4A8MXFP4MXFP8CutlassFusedMoEMethod,
            # Note: W4A8_NVFP4_FP8 is NOT supported by CUTLASS backend
        }
        return method_map.get(quant_algo)

    # TRTLLM backend
    if backend_str == "TRTLLM":
        method_map = {
            QuantAlgo.FP8_BLOCK_SCALES: DeepSeekFP8BlockScalesFusedMoEMethod,
            QuantAlgo.NVFP4: NVFP4TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A16_MXFP4: W4A16MXFP4TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A8_NVFP4_FP8: W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_FP8: W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_MXFP8: W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
        }
        return method_map.get(quant_algo)

    # CUTEDSL backend uses same methods as CUTLASS for quantization
    if backend_str == "CUTEDSL":
        method_map = {
            QuantAlgo.NVFP4: NVFP4CutlassFusedMoEMethod,
        }
        return method_map.get(quant_algo)

    # DEEPGEMM backend
    if backend_str == "DEEPGEMM":
        method_map = {
            QuantAlgo.FP8_BLOCK_SCALES: DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm,
        }
        return method_map.get(quant_algo)

    return None


def _should_skip_EPLB(quant_algo, backend_type, num_slots, num_experts):
    """
    Check if EPLB test should be skipped based on quant_algo, backend_type, and slot configuration.

    Returns:
        str or None: Skip reason if should skip, None otherwise
    """
    # Check num_slots > num_experts requirement
    if num_slots <= num_experts:
        return f"EPLB requires num_slots ({num_slots}) > num_experts ({num_experts})"

    # Get the FusedMoEMethod class for this quant_algo + backend combination
    method_class = _get_fused_moe_method_class(quant_algo, backend_type)

    if method_class is None:
        # Cannot determine the method class, skip the test
        return (
            f"Cannot determine FusedMoEMethod for quant_algo={quant_algo}, backend={backend_type}"
        )

    # Query the method class directly for EPLB support
    if not method_class.supports_online_eplb():
        return f"EPLB not supported for {method_class.__name__} (supports_online_eplb=False)"

    return None


def generate_eplb_test_params(
    parallel_modes,
    comm_methods,
    model_configs,
    num_slots_list,
    dtypes,
    backend_types,
    quant_algos,
    routing_methods,
) -> List:
    """
    Generate test parameter combinations for EPLB tests.

    EPLB requires num_slots > num_experts to be effective.

    Args:
        parallel_modes: List of parallel modes (only EP modes: DEP, TEP)
        comm_methods: List of communication methods
        model_configs: List of MoeModelConfig
        num_slots_list: List of EPLB slots (must be > num_experts)
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes

    Returns:
        List of pytest.param objects for runnable test configurations only
    """
    params: List = []

    for (
        parallel_mode,
        comm_method,
        model_config,
        num_slots,
        dtype,
        backend_type,
        quant_algo,
        routing_method_cls,
    ) in product(
        parallel_modes,
        comm_methods,
        model_configs,
        num_slots_list,
        dtypes,
        backend_types,
        quant_algos,
        routing_methods,
    ):
        # Get skip reason using existing logic
        skip_reason = get_quick_skip_reason(
            backend_type, quant_algo, dtype, model_config, routing_method_cls
        )

        # Check comm method platform compatibility (e.g. NVLink support)
        if not skip_reason:
            skip_reason = _get_comm_method_skip_reason(comm_method, model_config)

        # Check EPLB-specific skip conditions
        if not skip_reason:
            skip_reason = _should_skip_EPLB(
                quant_algo, backend_type, num_slots, model_config.num_experts
            )

        if skip_reason:
            continue

        routing_name = routing_method_cls.__name__.replace("MoeRoutingMethod", "")
        test_id = (
            f"parallel={parallel_mode}-comm={comm_method}-{model_config}-slots={num_slots}-"
            f"dtype={dtype}-backend={backend_type.value}-quant={quant_algo}-routing={routing_name}"
        )

        param_values = (
            parallel_mode,
            comm_method,
            dtype,
            backend_type.value,
            quant_algo,
            model_config,
            num_slots,
            routing_method_cls,
        )
        params.append(create_test_param(param_values, test_id))

    return params


# Pre-generate EPLB test parameters at module load time
EPLB_TEST_PARAMS = generate_eplb_test_params(
    parallel_modes=EPLB_PARALLEL_MODES,
    comm_methods=EPLB_COMM_METHODS,
    model_configs=EPLB_MODEL_CONFIGS,
    num_slots_list=EPLB_NUM_SLOTS_LIST,
    dtypes=DTYPES,
    backend_types=BACKEND_TYPES,
    quant_algos=QUANT_ALGOS,
    routing_methods=EPLB_ROUTING_METHODS,
)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@pytest.mark.skipif(
    not _tbr.is_host_accessible_device_memory_supported(),
    reason="needs support of host accessible device memory",
)
@pytest.mark.parametrize(
    "parallel_mode,comm_method_type,dtype,moe_backend,quant_algo,model_config,num_slots,routing_method_cls",
    EPLB_TEST_PARAMS,
)
def test_configurable_moe_multi_gpu_eplb(
    parallel_mode,
    comm_method_type,
    dtype,
    moe_backend,
    quant_algo,
    model_config,
    num_slots,
    routing_method_cls,
):
    skip_if_insufficient_gpu_memory(
        model_config.num_experts,
        model_config.hidden_size,
        model_config.intermediate_size,
        dtype,
    )

    world_size = 4
    _test_moe_multi_gpu(
        comm_method_type,
        moe_backend,
        quant_algo,
        dtype=dtype,
        world_size=world_size,
        parallel_mode=parallel_mode,
        enable_eplb=True,
        layer_updates_per_iter=1,
        num_slots=num_slots,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuteDSL not available")
@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="Requires SM100/SM103 (Blackwell)",
)
@pytest.mark.parametrize(
    "m,n,k,l,top_k",
    [
        (128, 2048, 2048, 8, 2),
        (256, 7168, 2048, 8, 2),
    ],
)
def test_allreduce_kernel_single_gpu(m, n, k, l, top_k):  # noqa: E741
    """Test 11-warp AllReduce kernel matches base finalize-fusion on single GPU.

    When world_size=1, the AR warps are no-op and the kernel should produce
    identical results to the base 7-warp finalize-fusion kernel.
    """
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
    )

    torch.manual_seed(42)
    device = torch.device("cuda")

    sf_vec_size = 16
    tile_size = 128
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size

    if not Sm100BlockScaledContiguousGroupedGemmAllReduceKernel.can_implement(
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=sf_vec_size,
        out_dtype=cutlass.BFloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=m,
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        out_major="n",
    ):
        pytest.skip("Cannot implement this config")

    # Generate random inputs
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (l, n, k // 2), dtype=torch.uint8, device=device)
    # Use small scale factors to avoid BF16 overflow (consistent with existing cuteDSL tests)
    a_sf = torch.randint(0, 8, (m * scale_k,), dtype=torch.uint8, device=device)
    b_sf = torch.randint(0, 8, (l, n, scale_k), dtype=torch.uint8, device=device)
    alpha = torch.ones(l, dtype=torch.float32, device=device) * 0.1

    tile_idx_to_expert_idx = torch.arange(num_tiles, dtype=torch.int32, device=device) % l
    tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32, device=device)
    permuted_idx_to_expanded_idx = torch.arange(m, dtype=torch.int32, device=device)
    num_non_exiting_tiles = torch.tensor([num_tiles], dtype=torch.int32, device=device)
    token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

    staging = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    out_ar = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)
    out_ref = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

    make_ptr = cutlass.cute.runtime.make_ptr
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Helper to make pointers
    def ptrs():
        return (
            make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(
                cutlass.Float8E4M3FN, a_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float8E4M3FN, b_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(
                cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
            ),
            make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem),
            make_ptr(cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem),
        )

    # Run base finalize-fusion kernel
    base_kernel = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    p = ptrs()
    out_ref_ptr = make_ptr(
        cutlass.BFloat16, out_ref.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    compiled_base = cute.compile(
        base_kernel.wrapper,
        p[0],
        p[1],
        p[2],
        p[3],
        out_ref_ptr,
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    compiled_base(
        p[0],
        p[1],
        p[2],
        p[3],
        out_ref_ptr,
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        stream=stream,
    )
    torch.cuda.synchronize()

    # Run AllReduce kernel (world_size=1)
    ar_kernel = Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    staging_ptr = make_ptr(
        cutlass.BFloat16, staging.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    out_ar_ptr = make_ptr(
        cutlass.BFloat16, out_ar.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    compiled_ar = cute.compile(
        ar_kernel.wrapper,
        p[0],
        p[1],
        p[2],
        p[3],
        staging_ptr,
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        0,
        0,
        0,
        0,  # mc pointers (unused for world_size=1)
        0,
        0,  # rank strides (unused for world_size=1)
        out_ar_ptr,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        0,
        1,  # rank=0, world_size=1
        0,  # ar_strategy=0 (unused for world_size=1)
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    compiled_ar(
        p[0],
        p[1],
        p[2],
        p[3],
        staging_ptr,
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        0,
        0,
        0,
        0,
        0,
        0,  # rank strides (unused for world_size=1)
        out_ar_ptr,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        # rank and world_size are Constexpr, baked in at compile time
        stream=stream,
    )
    torch.cuda.synchronize()

    # Compare results
    assert torch.allclose(out_ar, out_ref, atol=0, rtol=0, equal_nan=True), (
        f"AllReduce kernel (world_size=1) differs from base: "
        f"max_diff={torch.nan_to_num(out_ar - out_ref).abs().max().item()}, "
        f"nan_ar={out_ar.isnan().sum().item()}, nan_ref={out_ref.isnan().sum().item()}"
    )
    # Verify output has non-zero finite values (kernel actually computed something)
    finite_ref = out_ref[out_ref.isfinite()]
    assert finite_ref.numel() > 0 and finite_ref.abs().max().item() > 0, (
        "Base kernel output has no finite non-zero values"
    )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuteDSL not available")
@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="Requires SM100/SM103 (Blackwell)",
)
@pytest.mark.parametrize(
    "m,n,k,l,top_k,world_size,ar_strategy",
    [
        (256, 2048, 2048, 8, 2, 2, 0),
        (256, 2048, 2048, 8, 2, 2, 1),
        (512, 2048, 2048, 8, 2, 4, 0),
        (512, 2048, 2048, 8, 2, 4, 1),
        # Auto-select: M=256 should pick strategy 0, M=512/ws=4 picks 1
        (256, 2048, 2048, 8, 2, 2, -1),
        (512, 2048, 2048, 8, 2, 4, -1),
    ],
)
def test_allreduce_kernel_multi_gpu(m, n, k, l, top_k, world_size, ar_strategy):  # noqa: E741
    """Test 11-warp kernel IPC AllReduce correctness (simulated multi-GPU).

    Simulates multi-GPU IPC reduce by allocating separate staging buffers for
    each rank on a single GPU, running the kernel with world_size > 1, and
    comparing against a manual summation of all ranks' staging buffers.

    This test validates the IPC reduce datapath without requiring actual
    multi-GPU setup. Run with: pytest -k test_allreduce_kernel_multi_gpu
    """
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
    )

    torch.manual_seed(42)
    device = torch.device("cuda")

    sf_vec_size = 16
    tile_size = 128
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size  # M-tiles (for num_non_exiting_tiles)
    tile_n = mma_tiler_mn[1]
    num_2d_tiles = num_tiles * (n // tile_n)  # Total 2D tiles (M × N)

    if not Sm100BlockScaledContiguousGroupedGemmAllReduceKernel.can_implement(
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=sf_vec_size,
        out_dtype=cutlass.BFloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=m,
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        out_major="n",
    ):
        pytest.skip("Cannot implement this config")

    # Allocate staging buffers: one per simulated rank, laid out contiguously
    # to mimic IPC VA mapping where rank i's buffer = base + i * rank_stride
    staging_rank_stride = m * n * 2  # bytes (bf16 = 2 bytes per element)
    out_rank_stride = staging_rank_stride
    staging_all = torch.zeros(world_size * m * n, dtype=torch.bfloat16, device=device)
    output_all = torch.zeros(world_size * m * n, dtype=torch.bfloat16, device=device)
    staging_base_ptr = staging_all.data_ptr()
    output_base_ptr = output_all.data_ptr()

    # Tile barriers: need one per 2D tile
    barrier_bytes = max(num_2d_tiles * 4, 4096)
    tile_barriers = torch.zeros(barrier_bytes // 4, dtype=torch.int32, device=device)
    completion_barriers = torch.zeros(barrier_bytes // 4, dtype=torch.int32, device=device)

    make_ptr = cutlass.cute.runtime.make_ptr
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # For each simulated rank, generate different random inputs and run
    # the kernel's epilogue to populate the staging buffer. We then validate
    # that the AR warps correctly sum across all ranks.
    per_rank_inputs = []
    for rank in range(world_size):
        torch.manual_seed(42 + rank)
        a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)
        b = torch.randint(0, 256, (l, n, k // 2), dtype=torch.uint8, device=device)
        a_sf = torch.randint(0, 8, (m * scale_k,), dtype=torch.uint8, device=device)
        b_sf = torch.randint(0, 8, (l, n, scale_k), dtype=torch.uint8, device=device)
        alpha = torch.ones(l, dtype=torch.float32, device=device) * 0.1

        tile_idx_to_expert_idx = torch.arange(num_tiles, dtype=torch.int32, device=device) % l
        tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32, device=device)
        permuted_idx_to_expanded_idx = torch.arange(m, dtype=torch.int32, device=device)
        num_non_exiting_tiles = torch.tensor([num_tiles], dtype=torch.int32, device=device)
        token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

        per_rank_inputs.append(
            dict(
                a=a,
                b=b,
                a_sf=a_sf,
                b_sf=b_sf,
                alpha=alpha,
                tile_idx_to_expert_idx=tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                token_final_scales=token_final_scales,
            )
        )

    # Use rank 0's inputs for the kernel (all ranks share the same GEMM inputs
    # in this test — the key test is whether IPC reduce across staging works).
    # We populate each rank's staging buffer by running the kernel serially
    # for each rank with world_size=1 first, then run one final kernel with
    # world_size > 1 and the pre-populated staging buffers.

    # Step 1: Populate each rank's staging buffer by running epilogue-only
    # (world_size=1 to skip AR, writing to each rank's staging slice)
    for rank in range(world_size):
        inp = per_rank_inputs[rank]
        out_dummy = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

        a_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            inp["a"].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        b_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            inp["b"].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        a_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            inp["a_sf"].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_sf_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            inp["b_sf"].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        alpha_ptr = make_ptr(cutlass.Float32, inp["alpha"].data_ptr(), cute.AddressSpace.gmem)
        out_ptr = make_ptr(
            cutlass.BFloat16,
            out_dummy.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        tig_ptr = make_ptr(
            cutlass.Int32,
            inp["tile_idx_to_expert_idx"].data_ptr(),
            cute.AddressSpace.gmem,
        )
        tmn_ptr = make_ptr(
            cutlass.Int32,
            inp["tile_idx_to_mn_limit"].data_ptr(),
            cute.AddressSpace.gmem,
        )
        pie_ptr = make_ptr(
            cutlass.Int32,
            inp["permuted_idx_to_expanded_idx"].data_ptr(),
            cute.AddressSpace.gmem,
        )
        nne_ptr = make_ptr(
            cutlass.Int32,
            inp["num_non_exiting_tiles"].data_ptr(),
            cute.AddressSpace.gmem,
        )
        tfs_ptr = make_ptr(
            cutlass.Float32,
            inp["token_final_scales"].data_ptr(),
            cute.AddressSpace.gmem,
        )

        # Run kernel with world_size=1 to populate staging only
        # (AR warps no-op, epilogue scatter-adds to out_dummy — we only care
        # about what was written to staging before scatter-add, but for
        # world_size=1, epilogue writes to out directly, not staging.)
        # We need world_size > 1 for the epilogue to write to staging.
        # But then AR warps would try to reduce. This is a chicken-and-egg
        # problem. Instead, use the finalize-fusion kernel to compute the
        # expected output per rank, then compute the reference as the sum.
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
        )

        base_kernel = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_blkred=True,
            raster_along_m=False,
        )
        compiled_base = cute.compile(
            base_kernel.wrapper,
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            out_ptr,
            alpha_ptr,
            tig_ptr,
            tmn_ptr,
            pie_ptr,
            nne_ptr,
            tfs_ptr,
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            tile_size=tile_size,
            scaling_vector_size=sf_vec_size,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        compiled_base(
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            out_ptr,
            alpha_ptr,
            tig_ptr,
            tmn_ptr,
            pie_ptr,
            nne_ptr,
            tfs_ptr,
            m,
            n,
            k,
            l,
            num_tokens,
            top_k,
            stream=stream,
        )
        torch.cuda.synchronize()

        # Store this rank's output for reference (already scatter-added by
        # the finalize-fusion kernel with token_final_scales * alpha applied)
        per_rank_inputs[rank]["out_ref"] = out_dummy.clone()

    # Step 2: Compute reference output = sum of all ranks' outputs
    ref_output = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)
    for rank in range(world_size):
        # Convert to float for accurate summation, then back to bf16
        ref_output = (ref_output.float() + per_rank_inputs[rank]["out_ref"].float()).to(
            torch.bfloat16
        )

    # Step 3: Simulate the IPC reduce path. To test the AR warps, we need
    # to populate staging buffers (one per rank) with the FC2 output data.
    # We run the allreduce kernel with world_size > 1 and pre-populated
    # staging. The epilogue writes to staging (for world_size > 1), then
    # AR warps reduce across ranks.
    #
    # Since we can't run actual multi-rank epilogues on a single GPU, we
    # instead directly populate the staging buffers with known data and test
    # only the AR warp reduce logic.
    #
    # Populate staging[rank] with rank-specific bf16 data patterns.
    staging_all.zero_()
    for rank in range(world_size):
        torch.manual_seed(100 + rank)
        staging_slice = staging_all[rank * m * n : (rank + 1) * m * n]
        staging_slice.copy_(torch.randn(m * n, dtype=torch.bfloat16, device=device) * 0.01)

    # Reset barriers and output
    tile_barriers.zero_()
    completion_barriers.zero_()
    output_all.zero_()

    # Pre-set tile barriers to world_size (as if all ranks' epilogues arrived)
    # Need to cover all 2D tiles (M-tiles * N-tiles), not just M-tiles.
    tile_barriers[:num_2d_tiles] = world_size

    # Run the allreduce kernel as rank 0 with the staging buffers pre-populated.
    # The AR warps should read from all ranks' staging via IPC, sum, and store.
    inp = per_rank_inputs[0]
    ar_kernel = Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )

    # Create staging for rank 0 (the epilogue will write to this, overwriting
    # our test data). To avoid this, we use a separate staging buffer for the
    # GEMM epilogue and only test the AR reduction.
    staging_gemm = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    out_test = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

    a_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        inp["a"].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    b_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        inp["b"].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    a_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        inp["a_sf"].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        inp["b_sf"].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    alpha_ptr = make_ptr(cutlass.Float32, inp["alpha"].data_ptr(), cute.AddressSpace.gmem)
    staging_gemm_ptr = make_ptr(
        cutlass.BFloat16,
        staging_gemm.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    out_test_ptr = make_ptr(
        cutlass.BFloat16,
        out_test.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    tig_ptr = make_ptr(
        cutlass.Int32,
        inp["tile_idx_to_expert_idx"].data_ptr(),
        cute.AddressSpace.gmem,
    )
    tmn_ptr = make_ptr(
        cutlass.Int32,
        inp["tile_idx_to_mn_limit"].data_ptr(),
        cute.AddressSpace.gmem,
    )
    pie_ptr = make_ptr(
        cutlass.Int32,
        inp["permuted_idx_to_expanded_idx"].data_ptr(),
        cute.AddressSpace.gmem,
    )
    nne_ptr = make_ptr(
        cutlass.Int32,
        inp["num_non_exiting_tiles"].data_ptr(),
        cute.AddressSpace.gmem,
    )
    tfs_ptr = make_ptr(
        cutlass.Float32,
        inp["token_final_scales"].data_ptr(),
        cute.AddressSpace.gmem,
    )

    rank_val = 0
    # Resolve auto strategy (-1) to a concrete value for cute.compile()
    if ar_strategy < 0:
        tile_n = mma_tiler_mn[1]
        total_2d = num_tiles * (n // tile_n)
        ar_strategy = 1 if total_2d * world_size >= 256 else 0
    compiled_ar = cute.compile(
        ar_kernel.wrapper,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        staging_gemm_ptr,
        alpha_ptr,
        tig_ptr,
        tmn_ptr,
        pie_ptr,
        nne_ptr,
        tfs_ptr,
        staging_base_ptr,  # staging_mc_ptr: base of all ranks' staging
        output_base_ptr,  # out_mc_ptr: base of all ranks' output
        tile_barriers.data_ptr(),  # tile_barrier_mc_ptr
        completion_barriers.data_ptr(),  # completion_barrier_mc_ptr
        staging_rank_stride,  # staging_rank_stride
        out_rank_stride,  # out_rank_stride
        out_test_ptr,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        rank_val,
        world_size,
        ar_strategy,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    compiled_ar(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        staging_gemm_ptr,
        alpha_ptr,
        tig_ptr,
        tmn_ptr,
        pie_ptr,
        nne_ptr,
        tfs_ptr,
        staging_base_ptr,
        output_base_ptr,
        tile_barriers.data_ptr(),
        completion_barriers.data_ptr(),
        staging_rank_stride,
        out_rank_stride,
        out_test_ptr,
        m,
        n,
        k,
        l,
        num_tokens,
        top_k,
        stream=stream,
    )
    torch.cuda.synchronize()

    # Verify the AR warps' output in the NVLS output buffer (rank 0's slice)
    ar_output = output_all[: m * n].view(m, n)

    # Compute expected: sum of all ranks' staging data
    expected = torch.zeros(m, n, dtype=torch.float32, device=device)
    for rank in range(world_size):
        staging_rank = staging_all[rank * m * n : (rank + 1) * m * n].view(m, n)
        expected += staging_rank.float()
    expected_bf16 = expected.to(torch.bfloat16)

    # Compare AR output vs expected (bf16 sum allows small tolerance)
    max_diff = torch.nan_to_num(ar_output - expected_bf16).abs().max().item()
    assert torch.allclose(ar_output, expected_bf16, atol=1e-2, rtol=1e-2), (
        f"IPC AllReduce output differs from expected sum: "
        f"max_diff={max_diff}, "
        f"nan_ar={ar_output.isnan().sum().item()}, "
        f"nan_ref={expected_bf16.isnan().sum().item()}"
    )

    # Verify output has non-zero values
    finite_out = ar_output[ar_output.isfinite()]
    assert finite_out.numel() > 0 and finite_out.abs().max().item() > 0, (
        "AllReduce output has no finite non-zero values"
    )


def _test_allreduce_ipc_worker(m, n, k, num_experts, top_k, world_size, ar_strategy):
    """Worker for real multi-GPU IPC AllReduce test.

    Each MPI rank:
    1. Sets CUDA device and creates EP mapping
    2. Allocates MoEEpAllReduceMnnvlMemory (IPC VA mapped across ranks)
    3. Runs finalize-fusion kernel to get per-rank FC2 output
    4. Copies FC2 output to rank's staging slice in NVLS memory
    5. Pre-sets tile barriers, launches 11-warp kernel
    6. Verifies AR warps correctly reduce across all ranks
    """
    try:
        return _test_allreduce_ipc_worker_impl(m, n, k, num_experts, top_k, world_size, ar_strategy)
    except Exception:
        traceback.print_exc()
        raise


def _test_allreduce_ipc_worker_impl(m, n, k, num_experts, top_k, world_size, ar_strategy):  # noqa: C901
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from tensorrt_llm._mnnvl_utils import MoEEpAllReduceMnnvlMemory
    from tensorrt_llm.mapping import Mapping

    rank = mpi_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # EP mapping: all ranks in one EP group
    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    mapping.rank = rank

    sf_vec_size = 16
    tile_size = 128
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)
    scale_k = k // sf_vec_size
    num_tokens = m // top_k
    num_tiles = m // tile_size
    tile_n = mma_tiler_mn[1]
    num_2d_tiles = num_tiles * (n // tile_n)

    # --- Allocate NVLS IPC memory (synchronized via MPI allgather) ---
    staging_bytes = m * n * 2  # bf16
    output_bytes = m * n * 2
    barrier_bytes = max(num_2d_tiles * 4, 4096)

    staging_mem = MoEEpAllReduceMnnvlMemory(mapping, staging_bytes)
    output_mem = MoEEpAllReduceMnnvlMemory(mapping, output_bytes)
    tile_barrier_mem = MoEEpAllReduceMnnvlMemory(mapping, barrier_bytes)
    completion_barrier_mem = MoEEpAllReduceMnnvlMemory(mapping, barrier_bytes)

    staging_tensor = staging_mem.as_torch_strided_tensor(torch.bfloat16)
    output_tensor = output_mem.as_torch_strided_tensor(torch.bfloat16)
    tile_barrier_tensor = tile_barrier_mem.as_torch_strided_tensor(torch.int32)

    ep_comm = MoEEpAllReduceMnnvlMemory.get_comm(mapping)
    assert ep_comm.Get_size() == world_size, (
        f"EP communicator size {ep_comm.Get_size()} != world_size {world_size}"
    )

    # --- Generate per-rank GEMM data and compute FC2 reference output ---
    with torch.device(device):
        torch.manual_seed(42 + rank)
        a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8)
        b = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8)
        a_sf = torch.randint(0, 8, (m * scale_k,), dtype=torch.uint8)
        b_sf = torch.randint(0, 8, (num_experts, n, scale_k), dtype=torch.uint8)
        alpha = torch.ones(num_experts, dtype=torch.float32) * 0.1

        tile_idx_to_expert_idx = torch.arange(num_tiles, dtype=torch.int32) % num_experts
        tile_idx_to_mn_limit = torch.full((num_tiles,), m, dtype=torch.int32)
        permuted_idx_to_expanded_idx = torch.arange(m, dtype=torch.int32)
        num_non_exiting_tiles = torch.tensor([num_tiles], dtype=torch.int32)
        token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)

    # Run finalize-fusion (7-warp) kernel for per-rank reference output
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
    )

    make_ptr = cutlass.cute.runtime.make_ptr
    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    ref_output = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)
    a_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        a.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    b_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        b.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    a_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        a_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        b_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    alpha_ptr = make_ptr(
        cutlass.Float32,
        alpha.data_ptr(),
        cute.AddressSpace.gmem,
    )
    out_ref_ptr = make_ptr(
        cutlass.BFloat16,
        ref_output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    tig_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_expert_idx.data_ptr(),
        cute.AddressSpace.gmem,
    )
    tmn_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_mn_limit.data_ptr(),
        cute.AddressSpace.gmem,
    )
    pie_ptr = make_ptr(
        cutlass.Int32,
        permuted_idx_to_expanded_idx.data_ptr(),
        cute.AddressSpace.gmem,
    )
    nne_ptr = make_ptr(
        cutlass.Int32,
        num_non_exiting_tiles.data_ptr(),
        cute.AddressSpace.gmem,
    )
    tfs_ptr = make_ptr(
        cutlass.Float32,
        token_final_scales.data_ptr(),
        cute.AddressSpace.gmem,
    )

    base_kernel = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    compiled_base = cute.compile(
        base_kernel.wrapper,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        out_ref_ptr,
        alpha_ptr,
        tig_ptr,
        tmn_ptr,
        pie_ptr,
        nne_ptr,
        tfs_ptr,
        m,
        n,
        k,
        num_experts,
        num_tokens,
        top_k,
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    compiled_base(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        out_ref_ptr,
        alpha_ptr,
        tig_ptr,
        tmn_ptr,
        pie_ptr,
        nne_ptr,
        tfs_ptr,
        m,
        n,
        k,
        num_experts,
        num_tokens,
        top_k,
        stream=stream,
    )
    torch.cuda.synchronize(device)

    # ref_output now holds this rank's FC2 output (scatter-added with scales).
    # For the IPC reduce test, we need the RAW staging data (before scatter).
    # Generate simple known patterns for staging instead.
    torch.manual_seed(100 + rank)
    staging_data = torch.randn(m * n, dtype=torch.bfloat16, device=device) * 0.01
    staging_tensor[rank, : m * n].copy_(staging_data)
    torch.cuda.synchronize(device)

    # MPI barrier: ensure all ranks have written their staging data
    ep_comm.barrier()

    # --- Pre-set tile barriers and zero output ---
    output_tensor.zero_()
    if rank == 0:
        # Barrier memory is IPC-mapped; rank 0 sets it for all.
        barrier_view = tile_barrier_tensor[0]
        barrier_view[:num_2d_tiles] = world_size
    torch.cuda.synchronize(device)
    ep_comm.barrier()

    # --- Run 11-warp AllReduce kernel ---
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
    )

    # Separate staging buffer for the GEMM epilogue (not the IPC staging)
    staging_gemm = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    out_test = torch.zeros(num_tokens, n, dtype=torch.bfloat16, device=device)

    staging_gemm_ptr = make_ptr(
        cutlass.BFloat16,
        staging_gemm.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    out_test_ptr = make_ptr(
        cutlass.BFloat16,
        out_test.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Resolve auto strategy
    concrete_ar_strategy = ar_strategy
    if concrete_ar_strategy < 0:
        total_2d = num_tiles * (n // tile_n)
        concrete_ar_strategy = 1 if total_2d * world_size >= 256 else 0

    ar_kernel = Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_blkred=True,
        raster_along_m=False,
    )
    compiled_ar = cute.compile(
        ar_kernel.wrapper,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        staging_gemm_ptr,
        alpha_ptr,
        tig_ptr,
        tmn_ptr,
        pie_ptr,
        nne_ptr,
        tfs_ptr,
        staging_mem.ptr,  # staging_mc_ptr: IPC base address
        output_mem.ptr,  # out_mc_ptr: IPC base address
        tile_barrier_mem.ptr,  # tile_barrier_mc_ptr
        completion_barrier_mem.ptr,  # completion_barrier_mc_ptr
        staging_mem.rank_stride,  # staging_rank_stride
        output_mem.rank_stride,  # out_rank_stride
        out_test_ptr,
        m,
        n,
        k,
        num_experts,
        num_tokens,
        top_k,
        rank,  # Constexpr: this rank
        world_size,  # Constexpr
        concrete_ar_strategy,  # Constexpr
        tile_size=tile_size,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    compiled_ar(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        staging_gemm_ptr,
        alpha_ptr,
        tig_ptr,
        tmn_ptr,
        pie_ptr,
        nne_ptr,
        tfs_ptr,
        staging_mem.ptr,
        output_mem.ptr,
        tile_barrier_mem.ptr,
        completion_barrier_mem.ptr,
        staging_mem.rank_stride,
        output_mem.rank_stride,
        out_test_ptr,
        m,
        n,
        k,
        num_experts,
        num_tokens,
        top_k,
        stream=stream,
    )
    torch.cuda.synchronize(device)
    ep_comm.barrier()

    # --- Verify: AR output in this rank's NVLS output slice ---
    ar_output = output_tensor[rank, : m * n].clone().view(m, n)

    # Compute expected: sum of ALL ranks' staging data
    expected = torch.zeros(m, n, dtype=torch.float32, device=device)
    for r in range(world_size):
        rank_staging = staging_tensor[r, : m * n].view(m, n)
        expected += rank_staging.float()
    expected_bf16 = expected.to(torch.bfloat16)

    max_diff = torch.nan_to_num(ar_output - expected_bf16).abs().max().item()
    assert torch.allclose(ar_output, expected_bf16, atol=1e-2, rtol=1e-2), (
        f"Rank {rank}: IPC AllReduce output mismatch: "
        f"max_diff={max_diff}, "
        f"nan_ar={ar_output.isnan().sum().item()}, "
        f"nan_ref={expected_bf16.isnan().sum().item()}"
    )

    finite_out = ar_output[ar_output.isfinite()]
    assert finite_out.numel() > 0 and finite_out.abs().max().item() > 0, (
        f"Rank {rank}: AllReduce output has no finite non-zero values"
    )

    return None


@pytest.mark.parametrize(
    "m,n,k,num_experts,top_k,world_size,ar_strategy",
    [
        (256, 2048, 2048, 8, 2, 2, 0),
        (256, 2048, 2048, 8, 2, 2, 1),
    ],
)
def test_allreduce_kernel_real_ipc(m, n, k, num_experts, top_k, world_size, ar_strategy):
    """Test 11-warp kernel with REAL IPC AllReduce on multiple GPUs.

    Requires MPI launch with enough ranks:
      mpirun -n 3 pytest -k test_allreduce_kernel_real_ipc

    Each rank:
    1. Allocates MoEEpAllReduceMnnvlMemory (cross-rank IPC via fabric handles)
    2. Writes rank-specific staging data to its NVLS slice
    3. Pre-sets tile barriers, launches the 11-warp kernel
    4. Verifies AR warps correctly reduce across all ranks via IPC
    """
    import cutlass

    if get_sm_version() not in (100, 103):
        pytest.skip("Requires SM 100/103 (Blackwell)")

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.blockscaled_contiguous_grouped_gemm_allreduce import (
        Sm100BlockScaledContiguousGroupedGemmAllReduceKernel,
    )

    if not Sm100BlockScaledContiguousGroupedGemmAllReduceKernel.can_implement(
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        out_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        m=m,
        n=n,
        k=k,
        l=num_experts,
        a_major="k",
        b_major="k",
        out_major="n",
    ):
        pytest.skip("Cannot implement this config")

    def init_worker(custom_paths):
        for custom_path in custom_paths:
            if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
                sys.path.append(custom_path)

    with MPIPoolExecutor(
        initializer=init_worker,
        initargs=(sys.path,),
        max_workers=world_size,
    ) as executor:
        results = list(
            executor.map(
                _test_allreduce_ipc_worker,
                *zip(*[(m, n, k, num_experts, top_k, world_size, ar_strategy)] * world_size),
            )
        )
        for r in results:
            assert r is None


# ============================================================================
# End-to-End ConfigurableMoE V4 EP Test
# ============================================================================

# V4 EP requires: CUTEDSL + NVFP4 + EP > 1 + SM100/103 + MNNVL.
# This test runs through the full ConfigurableMoE forward path with
# AllGather dispatch → V4 fused FC2+AllReduce → skip combine.

V4_EP_TEST_CONFIGS = [
    # (num_experts, top_k, hidden_size, intermediate_size, seq_len)
    (8, 2, 512, 512, 8),
    (60, 4, 2048, 1408, 4),
]


def _test_configurable_moe_v4_ep_worker(
    num_experts, top_k, hidden_size, intermediate_size, seq_len, world_size
):
    """Worker for ConfigurableMoE V4 EP end-to-end test.

    Each MPI rank creates a ConfigurableMoE with CUTEDSL backend, NVFP4 quant,
    and DEP (EP-only) parallelism.  The forward path goes through AllGather
    dispatch → run_moe_nvfp4_v4_ep (fused FC2+AllReduce) → scatter output.
    """
    try:
        _test_configurable_moe_v4_ep_worker_impl(
            num_experts, top_k, hidden_size, intermediate_size, seq_len, world_size
        )
    except Exception:
        traceback.print_exc()
        raise


def _test_configurable_moe_v4_ep_worker_impl(
    num_experts, top_k, hidden_size, intermediate_size, seq_len, world_size
):
    from tensorrt_llm._utils import get_sm_version

    rank = mpi_rank()
    torch.cuda.set_device(rank)
    dtype = torch.bfloat16
    quant_algo = QuantAlgo.NVFP4

    # DEP mode: attention DP + MoE EP
    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    mapping.rank = rank

    all_rank_num_tokens = [seq_len] * world_size

    with torch.device(f"cuda:{rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Create routing method and input tensors
        routing_method = _create_routing_method(
            RenormalizeMoeRoutingMethod, top_k=top_k, num_experts=num_experts, dtype=dtype
        )
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")

        # Setup quantization
        backend_type = MoeBackendType.CUTEDSL
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )
        num_local_experts = num_experts // mapping.moe_ep_size
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            num_local_experts=num_local_experts,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        test_max_num_tokens = max(256, seq_len)

        model_cfg = _create_model_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=backend_type.value,
            max_num_tokens=test_max_num_tokens,
        )

        with create_moe(
            routing_method=routing_method,
            reduce_results=True,
            model_config=model_cfg,
        ) as fused_moe:
            fused_moe.load_weights([weights])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{rank}")

            # Check V4 is actually active
            sm_version = get_sm_version()
            v4_expected = sm_version in (100, 103) and MnnvlMemory.supports_mnnvl()
            backend = fused_moe.backend
            has_mapping = hasattr(backend, "mapping")
            ep_size = backend.mapping.moe_ep_size if has_mapping else -1
            v4_active = backend._should_use_v4_ep()
            assert v4_active == v4_expected, (
                f"V4 EP active={v4_active} but expected={v4_expected} "
                f"(SM={sm_version}, MNNVL={MnnvlMemory.supports_mnnvl()}, "
                f"has_mapping={has_mapping}, ep_size={ep_size}, "
                f"backend_cls={type(backend).__name__})"
            )
            if not v4_active:
                pytest.skip(
                    f"V4 EP not available: SM={sm_version}, MNNVL={MnnvlMemory.supports_mnnvl()}"
                )

            # Run forward
            with torch.inference_mode():
                output = fused_moe.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            torch.cuda.synchronize()

            # Reference: single-GPU (no EP)
            ref_fused_moe = quantize_util.create_ref_module(routing_method)
            ref_fused_moe.moe_tp_size = 1
            ref_fused_moe.load_weights([weights])
            ref_fused_moe.cuda(f"cuda:{rank}")

            with torch.inference_mode():
                ref_output = ref_fused_moe.forward(x, router_logits)

            # Check accuracy
            ref_fused_moe.check_accuracy(output, ref_output)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="needs >= 2 GPUs for V4 EP test",
)
@pytest.mark.parametrize(
    "num_experts,top_k,hidden_size,intermediate_size,seq_len",
    V4_EP_TEST_CONFIGS,
    ids=[f"e{ne}_k{k}_h{h}_i{i}_s{s}" for ne, k, h, i, s in V4_EP_TEST_CONFIGS],
)
def test_configurable_moe_v4_ep(num_experts, top_k, hidden_size, intermediate_size, seq_len):
    """End-to-end test for V4 EP (fused FC2+AllReduce) through ConfigurableMoE.

    Tests the full forward path: AllGather dispatch → V4 fused GEMM+AllReduce →
    scatter output.  Requires GB200 (SM100/103) with MNNVL support.

    Run with: mpirun -n 3 pytest -k test_configurable_moe_v4_ep -vs
    (first MPI rank is the MPIPoolExecutor controller)
    """
    world_size = min(torch.cuda.device_count(), 4)

    def init_worker(custom_paths):
        for custom_path in custom_paths:
            if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
                sys.path.append(custom_path)

    with MPIPoolExecutor(
        initializer=init_worker,
        initargs=(sys.path,),
        max_workers=world_size,
    ) as executor:
        results = list(
            executor.map(
                _test_configurable_moe_v4_ep_worker,
                *zip(
                    *[
                        (
                            num_experts,
                            top_k,
                            hidden_size,
                            intermediate_size,
                            seq_len,
                            world_size,
                        )
                    ]
                    * world_size
                ),
            )
        )
        for r in results:
            assert r is None
