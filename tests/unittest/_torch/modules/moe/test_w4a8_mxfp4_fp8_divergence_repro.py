# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Element-wise divergence reproducer for W4A8_MXFP4_FP8 TRTLLM-Gen failing cases.

Builds the smallest failing single_gpu case
(``gpt-oss e60_k4_h2048_i1408 seq=1`` + W4A8_MXFP4_FP8 TRTLLM backend) and
compares the actual TRTLLM-Gen kernel output against (a) the baseline
``MXFP4FP8RefGatedMLPFusedMoE`` and (b) several variants that probe specific
hypotheses about where the ref vs. kernel divergence comes from.

Not a regular pytest test - this file is a diagnostic. Run it on a GB200 OCI
node with ``pytest -s
tests/unittest/_torch/modules/moe/test_w4a8_mxfp4_fp8_divergence_repro.py``.
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
import tempfile
from typing import Callable, Dict

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from _torch.modules.moe.moe_test_utils import MoeBackendType
from _torch.modules.moe.quantize_utils import MXFP4FP8RefGatedMLPFusedMoE, get_test_quant_params
from mpi4py import MPI
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig  # noqa: F401
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# ---------------------------------------------------------------------------
# Cases:
#   - FAILING_CFG: smallest module-level failing case from Phase A.
#   - PASSING_CFG: known-good gpt-oss config to sanity-check the reproducer.
# Magnitudes between fused-kernel and ref should agree for PASSING_CFG; any
# >100x ratio on FAILING_CFG alone implicates the kernel itself.
# ---------------------------------------------------------------------------
DTYPE = torch.bfloat16
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

FAILING_CFG = dict(
    label="FAILING_e60_k4_h2048_i1408",
    num_experts=60,
    top_k=4,
    hidden_size=2048,
    intermediate_size=1408,
    seq_len=1,
)

PASSING_CFG = dict(
    label="PASSING_e256_k8_h2048_i2048",
    num_experts=256,
    top_k=8,
    hidden_size=2048,
    intermediate_size=2048,
    seq_len=1,
)


# ---------------------------------------------------------------------------
# Variant ref subclasses. Each subclass overrides exactly one aspect of
# MXFP4FP8RefGatedMLPFusedMoE so we can A/B-test which difference, if any,
# accounts for the ~93% mismatch against the fused TRTLLM-Gen kernel.
# ---------------------------------------------------------------------------


def _build_swiglu_fn(
    *,
    alpha: float,
    beta: float,
    limit: float,
    gate_clamp: str,  # "one_sided" | "two_sided" | "none"
    value_clamp: str,  # "two_sided" | "none"
    add_beta: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a closure that performs gpt-oss SwiGLU with knobs."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        if gate_clamp == "one_sided":
            gate = gate.clamp(max=limit)
        elif gate_clamp == "two_sided":
            gate = gate.clamp(min=-limit, max=limit)
        elif gate_clamp == "none":
            pass
        else:
            raise ValueError(gate_clamp)

        if value_clamp == "two_sided":
            value = value.clamp(min=-limit, max=limit)
        elif value_clamp == "none":
            pass
        else:
            raise ValueError(value_clamp)

        gate_act = gate * torch.sigmoid(gate * alpha)
        rhs = (value + beta) if add_beta else value
        return gate_act * rhs

    return fn


class VariantRef(MXFP4FP8RefGatedMLPFusedMoE):
    """Configurable ref that lets us toggle FP8 round-trips and swiglu knobs."""

    def __init__(
        self,
        *args,
        skip_fc1_roundtrip: bool = False,
        skip_fc2_roundtrip: bool = False,
        gate_clamp: str = "one_sided",
        value_clamp: str = "two_sided",
        add_beta: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._skip_fc1_roundtrip = skip_fc1_roundtrip
        self._skip_fc2_roundtrip = skip_fc2_roundtrip
        # Replace each expert's activation with our configurable variant.
        swiglu_fn = _build_swiglu_fn(
            alpha=SWIGLU_ALPHA,
            beta=SWIGLU_BETA,
            limit=SWIGLU_LIMIT,
            gate_clamp=gate_clamp,
            value_clamp=value_clamp,
            add_beta=add_beta,
        )
        for expert in self.experts:
            expert.activation = swiglu_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size_unpadded

        if self.hidden_size_unpadded < self.hidden_size:
            pad_size = self.hidden_size - self.hidden_size_unpadded
            hidden_states = F.pad(hidden_states, (0, pad_size))

        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        selected_experts, routing_weights = self.routing_method.apply(router_logits)

        final_hidden_states = torch.zeros(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx]
            expert = self.experts[expert_id]

            if not self._skip_fc1_roundtrip:
                expert_inputs = self._fp8_round_trip(expert_inputs, self.fc31_input_gate_dequant)

            h1 = expert.gate_up_proj(expert_inputs)
            h2 = expert._apply_activation(h1)

            if not self._skip_fc2_roundtrip:
                h2 = self._fp8_round_trip(h2, self.fc2_input_dequant)

            output = expert.down_proj(h2)
            final_hidden_states[batch_idx] += (
                routing_weights[batch_idx, nth_expert, None] * output.float()
            )

        final_hidden_states = final_hidden_states.reshape(original_shape)
        if self.hidden_size_unpadded < self.hidden_size:
            final_hidden_states = final_hidden_states[..., : self.hidden_size_unpadded]
        return final_hidden_states


# Each variant = (name, kwargs passed into VariantRef).
# V0 reproduces current production ref exactly.
VARIANTS: Dict[str, Dict[str, object]] = {
    "V0_baseline": dict(),
    "V1_gate_two_sided_clamp": dict(gate_clamp="two_sided"),
    "V2_no_value_plus_beta": dict(add_beta=False),
    "V3_no_clamp": dict(gate_clamp="none", value_clamp="none"),
    "V4_two_sided_no_beta": dict(gate_clamp="two_sided", add_beta=False),
    "V5_no_fc2_roundtrip": dict(skip_fc2_roundtrip=True),
    "V6_no_fc1_roundtrip": dict(skip_fc1_roundtrip=True),
    "V7_no_roundtrips": dict(skip_fc1_roundtrip=True, skip_fc2_roundtrip=True),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_routing_method(cfg) -> RenormalizeMoeRoutingMethod:
    return RenormalizeMoeRoutingMethod(top_k=cfg["top_k"], force_enable_pytorch_op=True)


def _build_model_config(cfg, quant_config) -> ModelConfig:
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = cfg["num_experts"]
    pretrained_config.hidden_size = cfg["hidden_size"]
    pretrained_config.intermediate_size = cfg["intermediate_size"]
    pretrained_config.torch_dtype = DTYPE
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(),
        quant_config=quant_config,
        moe_backend="TRTLLM",
        moe_disable_finalize_fusion=False,
        moe_load_balancer=None,
        max_num_tokens=max(256, cfg["seq_len"]),
    )


def _build_quantize_util(x: torch.Tensor):
    return get_test_quant_params(QuantAlgo.W4A8_MXFP4_FP8, x, MoeBackendType.TRTLLM)


def _build_variant_ref(cfg, quantize_util, routing_method, variant_kwargs) -> VariantRef:
    ref = VariantRef(
        num_experts=cfg["num_experts"],
        routing_method=routing_method,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        dtype=DTYPE,
        model_config=ModelConfig(quant_config=quantize_util.quant_config),
        bias=quantize_util.bias,
        swiglu_gptoss_style=quantize_util.swiglu_gptoss_style,
        swiglu_alpha=quantize_util.swiglu_alpha,
        swiglu_beta=quantize_util.swiglu_beta,
        swiglu_limit=quantize_util.swiglu_limit,
        **variant_kwargs,
    )
    return ref


def _diff_stats(out_a: torch.Tensor, out_b: torch.Tensor) -> Dict[str, float]:
    """Element-wise diff stats: A vs B."""
    a = out_a.detach().to(torch.float32).flatten()
    b = out_b.detach().to(torch.float32).flatten()
    diff = (a - b).abs()
    eps = 1e-6
    rel = diff / b.abs().clamp_min(eps)
    # Mismatch percentage uses the same definition as check_accuracy:
    # |a - b| > atol + rtol * |b|.
    atol = 0.3
    rtol = 0.15
    threshold = atol + rtol * b.abs()
    mismatch_frac = (diff > threshold).float().mean().item()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "median_abs_diff": diff.median().item(),
        "max_rel_diff": rel.max().item(),
        "mismatch_frac@atol0.3_rtol0.15": mismatch_frac,
        "b_abs_max": b.abs().max().item(),
        "b_abs_mean": b.abs().mean().item(),
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def _run_one_config(cfg):
    """Run all variants for a single config and print a summary."""
    label = cfg["label"]
    seq_len = cfg["seq_len"]
    hidden_size = cfg["hidden_size"]
    num_experts = cfg["num_experts"]
    intermediate_size = cfg["intermediate_size"]
    top_k = cfg["top_k"]

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((seq_len, hidden_size), dtype=DTYPE, device="cuda")
    router_logits = torch.randn((seq_len, num_experts), dtype=DTYPE, device="cuda")

    quantize_util_cls, quant_config, quant_kwargs = _build_quantize_util(x)
    quantize_util = quantize_util_cls(
        num_experts=num_experts,
        dtype=DTYPE,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        quant_config=quant_config,
        bias=True,
        swiglu_gptoss_style=True,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_beta=SWIGLU_BETA,
        swiglu_limit=SWIGLU_LIMIT,
        num_local_experts=num_experts,
    )
    weights = quantize_util.create_weights(**quant_kwargs)
    variant_weights = {k: copy.deepcopy(v) for k, v in weights.items()}

    model_cfg = _build_model_config(cfg, quant_config)
    routing_method = _build_routing_method(cfg)
    swiglu_tensors = quantize_util.get_swiglu_tensors()

    print(f"\n========== {label} ==========")
    print(
        f"Config: e{num_experts}_k{top_k}_h{hidden_size}_i{intermediate_size} "
        f"seq={seq_len}, gpt-oss alpha={SWIGLU_ALPHA} beta={SWIGLU_BETA} "
        f"limit={SWIGLU_LIMIT}"
    )

    with create_moe(
        routing_method=routing_method,
        reduce_results=True,
        model_config=model_cfg,
        bias=True,
        swiglu_alpha=swiglu_tensors["swiglu_alpha"],
        swiglu_beta=swiglu_tensors["swiglu_beta"],
        swiglu_limit=swiglu_tensors["swiglu_limit"],
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
    ) as fused_moe:
        fused_moe.load_weights([weights])
        fused_moe.post_load_weights()
        fused_moe.cuda("cuda:0")

        cache_path = os.path.join(tempfile.gettempdir(), "moe_repro_autotuner_cache.json")
        # Production tests always run an autotune pass before the simple
        # accuracy check; without it, the kernel uses an untuned default
        # tactic that can produce ~600x-magnitude-off output.
        with torch.inference_mode(), autotune(cache_path=cache_path):
            _ = fused_moe.forward(x, router_logits, all_rank_num_tokens=[seq_len])
            torch.cuda.synchronize()

        with torch.inference_mode():
            k_full = fused_moe.forward(
                x,
                router_logits,
                all_rank_num_tokens=[seq_len],
            )
            torch.cuda.synchronize()

    k_full = k_full.detach()
    print(
        f"K_full stats: abs_max={k_full.abs().max().item():.6f}, "
        f"abs_mean={k_full.abs().mean().item():.6f}, "
        f"abs_median={k_full.abs().float().median().item():.6f}, "
        f"shape={tuple(k_full.shape)}"
    )
    print()

    # Build V0 once for both vs-K_full and pairwise comparisons.
    ref0 = _build_variant_ref(cfg, quantize_util, routing_method, VARIANTS["V0_baseline"])
    ref0.moe_tp_size = 1
    ref0.load_weights([{k: v.clone() for k, v in variant_weights.items()}])
    ref0.cuda("cuda:0")
    with torch.inference_mode():
        r0 = ref0.forward(x, router_logits)
        torch.cuda.synchronize()
    r0 = r0.detach()
    print(
        f"V0 ref stats: abs_max={r0.abs().max().item():.6f}, "
        f"abs_mean={r0.abs().float().mean().item():.6f}"
    )
    ratio = r0.abs().mean().item() / max(k_full.abs().mean().item(), 1e-12)
    print(f"Magnitude ratio (V0 / K_full): {ratio:.2f}x")
    print()

    print("  -- Variant vs K_full --")
    for variant_name, variant_kwargs in VARIANTS.items():
        this_weights = {k: v.clone() for k, v in variant_weights.items()}
        ref = _build_variant_ref(cfg, quantize_util, routing_method, variant_kwargs)
        ref.moe_tp_size = 1
        ref.load_weights([this_weights])
        ref.cuda("cuda:0")
        with torch.inference_mode():
            r_out = ref.forward(x, router_logits)
            torch.cuda.synchronize()
        stats_vs_k = _diff_stats(r_out, k_full)
        print(f"  [{variant_name}]")
        for key, value in stats_vs_k.items():
            print(f"      {key:34s} = {value:.6f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GB200 required for TRTLLM-Gen W4A8 path")
def test_w4a8_mxfp4_fp8_divergence_repro():
    torch.cuda.set_device(0)
    with torch.device("cuda:0"):
        for cfg in (FAILING_CFG, PASSING_CFG):
            _run_one_config(cfg)

    print(
        "\nLegend: 'mismatch_frac@atol0.3_rtol0.15' is the production "
        "check_accuracy mismatch fraction; <0.15 means the variant would "
        "pass the test."
    )
    print("========================================================\n")
