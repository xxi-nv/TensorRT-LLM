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

# ruff: noqa: E501

"""MoE microbenchmark (MPI) — whole MoE module forward, all backends.

Times ``ConfigurableMoE.forward`` (routing + dispatch + GroupGEMM + activation +
combine) across multiple model profiles, parallel modes, backends, and
synthetic load-imbalance configurations.

See ``tests/microbenchmarks/BENCH_MOE_DESIGN.md`` for the full design.

Launch (examples):

```bash
# Single-rank smoke (eager)
python tests/microbenchmarks/bench_moe.py \
    --world_size 1 --model qwen1.5_moe --backend CUTLASS \
    --num_tokens 16 64 --no_cuda_graph

# Multi-rank winner selection (CUDA graph is the default; pass --no_cuda_graph
# to fall back to eager timing, e.g. for dynamic EPLB).
python tests/microbenchmarks/bench_moe.py \
    --world_size 4 --parallel_mode DEP --model deepseek_v3 \
    --backend best --num_tokens 64 256

# mpirun (multi-node)
mpirun -n 8 python tests/microbenchmarks/bench_moe.py \
    --model deepseek_v3 --backend best --num_tokens 64 256 \
    --parallel_mode DEP
```
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import json
import os
import pickle
import socket
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.autograd import DeviceType

# ``cloudpickle`` and ``mpi4py.futures.MPIPoolExecutor`` are only used in the
# self-spawning launcher path (single ``python3 bench_moe.py`` invocation that
# pops up its own MPI pool). When started under external mpirun/srun they are
# never touched, so import them lazily inside ``main()`` to avoid hard
# dependencies on container images that ship only the minimal mpi4py.

# When invoked as ``python tests/microbenchmarks/bench_moe.py`` (rather than
# ``python -m`` or via pytest), Python sets ``sys.path[0]`` to the script's
# directory (``tests/microbenchmarks/``), not the repo root. Adding the repo
# root first makes ``import tensorrt_llm`` resolve to the in-tree checkout
# without requiring an installed wheel or a manual ``PYTHONPATH``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ``quantize_utils.py`` lives under ``tests/unittest/_torch/modules/moe/`` and
# uses pytest-style relative imports such as ``from _torch.helpers import ...``;
# the project's ``tests/unittest/conftest.py`` puts ``tests/unittest`` on
# sys.path. Replicate that here so the benchmark works without pytest.
_TESTS_UNITTEST_DIR = Path(__file__).resolve().parent.parent / "unittest"
if str(_TESTS_UNITTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_UNITTEST_DIR))

from _torch.modules.moe.moe_test_utils import (  # noqa: E402
    MoeBackendType,
    MoeModelConfig,
    get_backend_class,
    resolve_deepseek_group_config,
)
from _torch.modules.moe.quantize_utils import get_test_quant_params  # noqa: E402
from transformers.configuration_utils import PretrainedConfig  # noqa: E402

import tensorrt_llm as tllm  # noqa: E402
from tensorrt_llm._torch.autotuner import AutoTuner, autotune  # noqa: E402
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe import (  # noqa: E402
    DeepSeekV3MoeRoutingMethod,
    DefaultMoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    RenormalizeNaiveMoeRoutingMethod,
    SigmoidRenormMoeRoutingMethod,
    TRTLLMGenFusedMoE,
    create_moe,
)
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (  # noqa: E402
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._utils import (  # noqa: E402
    local_mpi_rank,
    mpi_allgather,
    mpi_barrier,
    mpi_rank,
    mpi_world_size,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig  # noqa: E402
from tensorrt_llm.tools.layer_wise_benchmarks.runner import (  # noqa: E402
    BalanceMethod,
    make_balanced_routing_method,
    make_balanced_run_moe,
    make_forward_impl_check,
)

# ---------------------------------------------------------------------------
# Routing method registry
# ---------------------------------------------------------------------------

_ROUTING_METHODS = {
    "DEFAULT": DefaultMoeRoutingMethod,
    "RENORMALIZE": RenormalizeMoeRoutingMethod,
    "RENORMALIZE_NAIVE": RenormalizeNaiveMoeRoutingMethod,
    "LLAMA4_RENORMALIZE": Llama4RenormalizeMoeRoutingMethod,
    "DEEPSEEK_V3": DeepSeekV3MoeRoutingMethod,
    "MINIMAX_M2": MiniMaxM2MoeRoutingMethod,
    "SIGMOID_RENORM": SigmoidRenormMoeRoutingMethod,
}


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """Static model description used to build a synthetic MoE module."""

    name: str
    moe_model_config: MoeModelConfig
    quant_algo: Optional[QuantAlgo]
    routing_method_cls: type
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    swiglu_limit: float = float("inf")


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "qwen1.5_moe": ModelProfile(
        "qwen1.5_moe",
        MoeModelConfig(60, 4, 2048, 1408),
        QuantAlgo.FP8,
        RenormalizeMoeRoutingMethod,
    ),
    "deepseek_v2_lite": ModelProfile(
        "deepseek_v2_lite",
        MoeModelConfig(64, 6, 2048, 1408),
        QuantAlgo.FP8_BLOCK_SCALES,
        DeepSeekV3MoeRoutingMethod,
    ),
    "deepseek_v3": ModelProfile(
        "deepseek_v3",
        MoeModelConfig(256, 8, 7168, 2048, n_group=8, topk_group=4),
        QuantAlgo.FP8_BLOCK_SCALES,
        DeepSeekV3MoeRoutingMethod,
    ),
    "kimi_k2": ModelProfile(
        "kimi_k2",
        MoeModelConfig(384, 8, 7168, 2048),
        QuantAlgo.FP8_BLOCK_SCALES,
        DeepSeekV3MoeRoutingMethod,
    ),
    # DeepSeek-V4-Pro: 1.6T total / 49B activated. MoE config sourced from
    # https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json
    # (n_routed_experts=384, num_experts_per_tok=6, hidden_size=7168,
    # moe_intermediate_size=3072, topk_method="noaux_tc").
    # quant_algo intentionally left None: pass --quant on the CLI to pin the
    # mode (e.g. FP8_BLOCK_SCALES); the released checkpoint mixes FP4 experts
    # with FP8 elsewhere which has no single QuantAlgo match.
    "deepseek_v4_pro": ModelProfile(
        "deepseek_v4_pro",
        MoeModelConfig(384, 6, 7168, 3072),
        None,
        RenormalizeMoeRoutingMethod,
    ),
    # DeepSeek-V4-Flash: 284B total / 13B activated. MoE config sourced from
    # https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json
    # and matches the unit-test annotation in
    # tests/unittest/_torch/modules/moe/test_moe_backend.py:322
    # (DeepSeek-V4-Flash = MoeModelConfig(256, 6, 4096, 2048)).
    # quant_algo intentionally left None: pass --quant on the CLI.
    "deepseek_v4_flash": ModelProfile(
        "deepseek_v4_flash",
        MoeModelConfig(256, 6, 4096, 2048),
        None,
        RenormalizeMoeRoutingMethod,
    ),
    "mixtral_8x7b": ModelProfile(
        "mixtral_8x7b",
        MoeModelConfig(8, 2, 4096, 14336),
        QuantAlgo.FP8,
        RenormalizeMoeRoutingMethod,
    ),
    "gpt_oss_120b": ModelProfile(
        "gpt_oss_120b",
        MoeModelConfig(128, 4, 2880, 2880),
        QuantAlgo.W4A8_MXFP4_MXFP8,
        RenormalizeMoeRoutingMethod,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    ),
}


# All ConfigurableMoE-eligible backends.
_ALL_BACKENDS = [b.value for b in MoeBackendType]


def _check_backend_can_implement(
    backend_str: str,
    quant_algo: Optional[QuantAlgo],
    dtype_activation: torch.dtype,
    swiglu_gptoss_style: bool,
) -> Tuple[bool, Optional[str]]:
    """Resolve ``backend_str`` to its MoE class and forward to ``can_implement``.

    Returns ``(False, reason)`` on every failure mode so callers can format a
    single, uniform error message regardless of whether the backend name was
    unknown, the class lookup failed, or the backend itself reported that the
    requested ``(quant_algo, dtype, swiglu_gptoss_style)`` triple is unsupported
    on the current hardware.
    """
    try:
        backend_cls = get_backend_class(MoeBackendType(backend_str.upper()))
    except (KeyError, ValueError) as exc:
        return False, f"unknown MoE backend {backend_str!r}: {exc}"
    try:
        return backend_cls.can_implement(
            quant_algo=quant_algo,
            dtype_activation=dtype_activation,
            swiglu_gptoss_style=swiglu_gptoss_style,
        )
    except Exception as exc:
        return False, (f"{backend_cls.__name__}.can_implement raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _maybe_print_rank0(msg: str) -> None:
    if mpi_rank() == 0:
        print(msg, flush=True)


def _sync() -> None:
    torch.cuda.synchronize()
    mpi_barrier()


def _set_device_from_local_rank() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    local_rank = local_mpi_rank()
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            "Detected GPU oversubscription: "
            f"local_mpi_rank={local_rank} >= cuda_device_count={device_count}."
        )
    dev = local_rank % device_count
    torch.cuda.set_device(dev)
    return dev


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_dist_for_megamoe(moe_backend: str, rank: int, world_size: int) -> None:
    """Initialize the torch.distributed NCCL ProcessGroup for MegaMoE."""
    if moe_backend.upper() != MoeBackendType.MEGAMOE.value:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for MegaMoE backend")
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_get_free_tcp_port()))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_mpi_rank())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "mean": mean,
        "median": s[n // 2],
        "stdev": variance**0.5,
        "min": s[0],
        "max": s[-1],
    }


def _gather_per_rank(times_us: List[float], iter_stats: bool = False) -> Dict[str, Any]:
    all_times = mpi_allgather(times_us)
    if iter_stats:
        return {f"rank{i}": _compute_stats(t) for i, t in enumerate(all_times)}
    return {f"rank{i}": (sum(t) / len(t) if t else 0.0) for i, t in enumerate(all_times)}


# ---------------------------------------------------------------------------
# Token distribution & inputs
# ---------------------------------------------------------------------------


def _distribute_tokens(total: int, world_size: int, ratio: float) -> List[int]:
    """Distribute ``total`` global tokens across ``world_size`` ranks.

    ``ratio == 0`` -> evenly distributed (remainder on rank 0).
    ``ratio == 1`` -> rank 0 takes everything; other ranks get 0.
    """
    if world_size <= 0 or total < 0 or not (0.0 <= ratio <= 1.0):
        raise ValueError(f"invalid args: total={total}, world_size={world_size}, ratio={ratio}")
    if world_size == 1:
        return [total]

    base = total // world_size
    if ratio == 0.0:
        out = [base] * world_size
        out[0] += total - base * world_size
        return out
    rank0 = base + round((total - base) * ratio)
    rank0 = min(rank0, total)
    rest_total = total - rank0
    others = rest_total // (world_size - 1)
    out = [rank0] + [others] * (world_size - 1)
    out[1] += rest_total - others * (world_size - 1)
    return out


def _make_inputs(
    local_num_tokens: int,
    hidden_size: int,
    num_experts: int,
    act_dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic synthetic hidden states + router logits."""
    if local_num_tokens == 0:
        x = torch.empty((0, hidden_size), dtype=act_dtype, device=device)
        logits = torch.empty((0, num_experts), dtype=routing_logits_dtype, device=device)
        return x, logits
    x = torch.randn((local_num_tokens, hidden_size), dtype=act_dtype, device=device)
    logits = torch.randn((local_num_tokens, num_experts), dtype=routing_logits_dtype, device=device)
    return x, logits


# ---------------------------------------------------------------------------
# Mapping / ModelConfig / Routing-method construction
# ---------------------------------------------------------------------------


def _create_mapping_for_parallel_mode(world_size: int, parallel_mode: str) -> Mapping:
    """Return a freshly-built Mapping for ``parallel_mode``.

    Caller MUST overwrite ``mapping.rank = mpi_rank()`` inside each worker.
    """
    configs = {
        "DEP": {"moe_ep_size": world_size, "moe_tp_size": 1, "enable_attention_dp": True},
        "TEP": {"moe_ep_size": world_size, "moe_tp_size": 1, "enable_attention_dp": False},
        "DTP": {"moe_ep_size": 1, "moe_tp_size": world_size, "enable_attention_dp": True},
        "TTP": {"moe_ep_size": 1, "moe_tp_size": world_size, "enable_attention_dp": False},
    }
    if parallel_mode not in configs:
        raise ValueError(
            f"Unknown parallel_mode={parallel_mode!r}; expected one of {list(configs)}"
        )
    cfg = configs[parallel_mode]
    return Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=cfg["moe_ep_size"],
        moe_tp_size=cfg["moe_tp_size"],
        enable_attention_dp=cfg["enable_attention_dp"],
    )


def _build_mapping(args: argparse.Namespace, world_size: int) -> Mapping:
    """Build ``Mapping`` from CLI args, with optional fine-grained overrides."""
    if args.moe_ep_size is not None or args.moe_tp_size is not None:
        moe_ep = int(args.moe_ep_size) if args.moe_ep_size is not None else world_size
        moe_tp = int(args.moe_tp_size) if args.moe_tp_size is not None else 1
        if moe_ep * moe_tp != world_size:
            raise ValueError(
                f"moe_ep_size * moe_tp_size = {moe_ep * moe_tp} must equal world_size={world_size}"
            )
        mapping = Mapping(
            world_size=world_size,
            tp_size=world_size,
            moe_ep_size=moe_ep,
            moe_tp_size=moe_tp,
            enable_attention_dp=bool(args.enable_attention_dp),
        )
    else:
        mapping = _create_mapping_for_parallel_mode(world_size, args.parallel_mode)
    mapping.rank = mpi_rank()
    return mapping


def _create_routing_method(
    routing_method_cls,
    top_k: int,
    num_experts: int,
    bias_dtype: torch.dtype,
    profile_model_config: MoeModelConfig,
):
    """Create a routing-method instance mirroring ``test_moe_module._create_routing_method``."""
    if routing_method_cls in (RenormalizeMoeRoutingMethod, DefaultMoeRoutingMethod):
        return routing_method_cls(top_k=top_k, force_enable_pytorch_op=True)

    if routing_method_cls in (RenormalizeNaiveMoeRoutingMethod, Llama4RenormalizeMoeRoutingMethod):
        return routing_method_cls(top_k=top_k)

    if routing_method_cls is DeepSeekV3MoeRoutingMethod:
        n_group, topk_group = resolve_deepseek_group_config(profile_model_config)
        e_score_correction_bias = torch.zeros(num_experts, dtype=bias_dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=1.0,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
            is_fused=False,
        )

    if routing_method_cls is MiniMaxM2MoeRoutingMethod:
        e_score_correction_bias = torch.zeros(num_experts, dtype=bias_dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )

    if routing_method_cls is SigmoidRenormMoeRoutingMethod:
        return routing_method_cls(top_k=top_k, num_experts=num_experts)

    return routing_method_cls(top_k=top_k)


def _build_pretrained_config(
    num_experts: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype
) -> PretrainedConfig:
    """Construct a HF-style ``PretrainedConfig`` for ``ConfigurableMoE``."""
    pc = PretrainedConfig()
    pc.num_experts = num_experts
    pc.hidden_size = hidden_size
    pc.intermediate_size = intermediate_size
    pc.torch_dtype = dtype
    return pc


def _build_model_config(
    *,
    profile: ModelProfile,
    mapping: Mapping,
    quant_algo: Optional[QuantAlgo],
    moe_backend: str,
    use_cuda_graph: bool,
    max_num_tokens: int,
    use_low_precision_moe_combine: bool,
    enable_eplb: bool,
    num_slots: int,
    layer_updates_per_iter: int,
    dtype: torch.dtype,
) -> ModelConfig:
    """Build ``ModelConfig`` plumbed into ``create_moe``."""
    mc = profile.moe_model_config
    pretrained_config = _build_pretrained_config(
        mc.num_experts, mc.hidden_size, mc.intermediate_size, dtype
    )

    quant_config = (
        QuantConfig(quant_algo=None) if quant_algo is None else QuantConfig(quant_algo=quant_algo)
    )

    moe_load_balancer_config = (
        MoeLoadBalancerConfig(
            num_slots=num_slots,
            layer_updates_per_iter=layer_updates_per_iter,
        )
        if enable_eplb
        else None
    )

    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=moe_backend,
        moe_disable_finalize_fusion=False,
        moe_load_balancer=moe_load_balancer_config,
        max_num_tokens=max(int(max_num_tokens), 1),
        use_cuda_graph=use_cuda_graph,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
    )


# ---------------------------------------------------------------------------
# MoE construction (build phase)
# ---------------------------------------------------------------------------


def _backend_name_from_module(moe) -> str:
    """Resolve ``actual_backend`` for both ConfigurableMoE and legacy modules."""
    backend_attr = getattr(moe, "backend", None)
    if backend_attr is not None and backend_attr is not moe:
        backend_cls = type(backend_attr).__name__
    else:
        backend_cls = type(moe).__name__
    # Strip suffixes from class name so downstream comparison is stable.
    aliases = {
        "CutlassFusedMoE": "CUTLASS",
        "TRTLLMGenFusedMoE": "TRTLLM",
        "CuteDslFusedMoE": "CUTEDSL",
        "DeepGemmFusedMoE": "DEEPGEMM",
        "DenseGEMMFusedMoE": "DENSEGEMM",
        "MegaMoEDeepGemm": "MEGAMOE_DEEPGEMM",
        "VanillaMoE": "VANILLA",
    }
    return aliases.get(backend_cls, backend_cls.upper())


def _build_moe_module(
    *,
    profile: ModelProfile,
    mapping: Mapping,
    quant_algo: Optional[QuantAlgo],
    moe_backend: str,
    use_cuda_graph: bool,
    max_num_tokens: int,
    use_low_precision_moe_combine: bool,
    enable_eplb: bool,
    num_slots: int,
    layer_updates_per_iter: int,
    enable_perfect_router: bool,
    dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    device: torch.device,
):
    """Build a fresh ``ConfigurableMoE`` for a single ``(backend, num_tokens)`` case.

    Returns ``(moe_module, moe_load_balancer or None, initial_expert_ids or None)``.
    """
    # Perfect-router toggle must be applied to the env BEFORE create_moe(...),
    # so the module's ``_init_perfect_router`` picks it up at construction.
    if enable_perfect_router:
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    else:
        os.environ.pop("ENABLE_PERFECT_ROUTER", None)

    mc = profile.moe_model_config
    swiglu_gptoss_style = (
        profile.swiglu_alpha != 1.0
        or profile.swiglu_beta != 0.0
        or profile.swiglu_limit != float("inf")
    )

    routing_method = _create_routing_method(
        profile.routing_method_cls,
        top_k=mc.top_k,
        num_experts=mc.num_experts,
        bias_dtype=dtype,
        profile_model_config=mc,
    )

    model_config = _build_model_config(
        profile=profile,
        mapping=mapping,
        quant_algo=quant_algo,
        moe_backend=moe_backend,
        use_cuda_graph=use_cuda_graph,
        max_num_tokens=max_num_tokens,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
        enable_eplb=enable_eplb,
        num_slots=num_slots,
        layer_updates_per_iter=layer_updates_per_iter,
        dtype=dtype,
    )

    # MegaMoE backend requires torch.distributed NCCL group ahead of construction.
    _ensure_dist_for_megamoe(moe_backend, mapping.rank, mapping.world_size)

    # Probe input used by get_test_quant_params to derive activation scales.
    probe_x = torch.randn(
        (max(1, mc.hidden_size // 32), mc.hidden_size), dtype=dtype, device=device
    )
    backend_type = MoeBackendType(moe_backend.upper())
    quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
        quant_algo, probe_x, backend_type
    )
    quant_kwargs.pop("ref_cls", None)  # reference module not needed in benchmark mode

    num_local_experts = mc.num_experts // max(mapping.moe_ep_size, 1)
    quantize_util = quantize_util_cls(
        num_experts=mc.num_experts,
        dtype=dtype,
        intermediate_size=mc.intermediate_size,
        hidden_size=mc.hidden_size,
        quant_config=quant_config,
        bias=swiglu_gptoss_style,
        swiglu_gptoss_style=swiglu_gptoss_style,
        swiglu_alpha=profile.swiglu_alpha if swiglu_gptoss_style else None,
        swiglu_beta=profile.swiglu_beta if swiglu_gptoss_style else None,
        swiglu_limit=profile.swiglu_limit if swiglu_gptoss_style else None,
        num_local_experts=num_local_experts,
    )

    weight_loading_mode = getattr(
        quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
    )

    # EPLB context: a no-op if not enabled.
    if enable_eplb:
        model_config.moe_load_balancer.setup(
            ep_rank=mapping.moe_ep_rank, ep_size=mapping.moe_ep_size
        )
        moe_load_balancer = MoeLoadBalancer(
            ep_rank=mapping.moe_ep_rank,
            ep_size=mapping.moe_ep_size,
            layer_updates_per_iter=model_config.moe_load_balancer.layer_updates_per_iter,
        )
    else:
        moe_load_balancer = None

    eplb_ctx = moe_load_balancer if moe_load_balancer is not None else contextlib.nullcontext()
    swiglu_tensors = quantize_util.get_swiglu_tensors()

    with eplb_ctx:
        moe = create_moe(
            routing_method=routing_method,
            num_experts=mc.num_experts,
            hidden_size=mc.hidden_size,
            intermediate_size=mc.intermediate_size,
            dtype=dtype,
            reduce_results=True,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=swiglu_gptoss_style,
            swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
            swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
            swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
        )

        if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
            weights, _ref_weights, _ref_kwargs = quantize_util.prepare_weights_from_backend(
                moe, **quant_kwargs
            )
        else:
            weights = quantize_util.create_weights(**quant_kwargs)

        if enable_eplb:
            for key, value in weights.items():
                if isinstance(value, torch.Tensor):
                    weights[key] = value.to("cpu")

        moe.load_weights([weights])
        moe.post_load_weights()
        moe.cuda(f"cuda:{torch.cuda.current_device()}")

        initial_expert_ids = None
        if moe_load_balancer is not None:
            moe_load_balancer.register_weight_slots_after_to_cuda()
            moe_load_balancer.finalize_model()
            moe_load_balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)
            if moe_load_balancer.single_layer_load_balancers:
                initial_expert_ids = copy.deepcopy(
                    moe_load_balancer.single_layer_load_balancers[0].get_old_rank_expert_ids()
                )

    return moe, moe_load_balancer, initial_expert_ids, routing_logits_dtype


# ---------------------------------------------------------------------------
# Routing-imbalance patch
# ---------------------------------------------------------------------------


@dataclass
class _PatchState:
    moe: Any
    routing_target: Any
    apply_method_orig: Any
    run_moe_orig: Any
    forward_impl_orig: Any


@contextlib.contextmanager
def _maybe_install_balance_patch(
    moe,
    mapping: Mapping,
    num_experts: int,
    top_k: int,
    experts_hot_imbalane_ratio: float,
    force_balance_patch: bool,
):
    """Patch routing/run_moe for synthetic per-expert hot-spot imbalance.

    A pass-through context manager when both ``experts_hot_imbalane_ratio`` and
    ``force_balance_patch`` are unset. When active, follows the runner.py
    pattern and restores all overridden methods on exit.
    """
    if experts_hot_imbalane_ratio == 0.0 and not force_balance_patch:
        yield
        return

    if experts_hot_imbalane_ratio == 0.0:
        balance_method = BalanceMethod.Balanced
        balance_ratio = 1.0
    else:
        balance_method = BalanceMethod.ImbalancedExperts
        # Inverse semantics: ``balance_ratio=0`` is fully imbalanced.
        balance_ratio = 1.0 - experts_hot_imbalane_ratio

    dp_size = mapping.dp_size
    dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
    ep_size = mapping.moe_ep_size

    # ConfigurableMoE keeps the routing_method on the outer module; the inner
    # backend (e.g. TRTLLMGen) accesses the same instance via .backend.
    routing_target = moe
    apply_method_orig = routing_target.routing_method.apply
    routing_target.routing_method.apply = make_balanced_routing_method(
        routing_target,
        apply_method_orig,
        num_experts,
        balance_method,
        balance_ratio,
        dp_size,
        dp_rank,
        ep_size,
    )

    inner_backend = getattr(moe, "backend", moe)
    run_moe_orig = None
    if isinstance(inner_backend, TRTLLMGenFusedMoE):
        run_moe_orig = inner_backend.run_moe
        inner_backend.run_moe = make_balanced_run_moe(
            inner_backend,
            run_moe_orig,
            top_k,
            num_experts,
            balance_method,
            balance_ratio,
            dp_size,
            dp_rank,
            ep_size,
        )

    forward_impl_orig = moe.forward_impl
    moe.forward_impl = make_forward_impl_check(moe, forward_impl_orig)

    state = _PatchState(
        moe=moe,
        routing_target=routing_target,
        apply_method_orig=apply_method_orig,
        run_moe_orig=run_moe_orig,
        forward_impl_orig=forward_impl_orig,
    )
    try:
        yield
    finally:
        state.routing_target.routing_method.apply = state.apply_method_orig
        if state.run_moe_orig is not None:
            getattr(moe, "backend", moe).run_moe = state.run_moe_orig
        state.moe.forward_impl = state.forward_impl_orig


# ---------------------------------------------------------------------------
# Autotune (untimed pre-pass)
# ---------------------------------------------------------------------------


def _run_autotune(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    eplb: Optional[MoeLoadBalancer],
    fast_autotune: bool,
) -> None:
    """One untimed forward pass under ``autotune(...)`` to populate kernel caches."""
    if fast_autotune:
        AutoTuner.get().warmup = 0
        AutoTuner.get().repeat = 1
        AutoTuner.get().stream_delay_micro_secs = 10

    cache_path = os.path.join(tempfile.gettempdir(), "bench_moe_autotuner_cache.json")
    iter_ctx = MoeLoadBalancerIterContext(eplb) if eplb is not None else contextlib.nullcontext()
    with torch.inference_mode(), autotune(cache_path=cache_path), iter_ctx:
        moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Eager timing
# ---------------------------------------------------------------------------


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """Allocate a 2x-L2 flush buffer to clear L2 between iterations."""
    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    l2_flush_size = (l2_size * 2) // 4
    return torch.empty(l2_flush_size, dtype=torch.int32, device=device)


def _time_moe_forward_eager(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    *,
    warmup: int,
    iters: int,
    eplb: Optional[MoeLoadBalancer],
    flush_l2: bool = True,
) -> Tuple[List[float], Dict[str, Any]]:
    """Time eager ``ConfigurableMoE.forward`` with Kineto + ``record_function``."""
    device = x.device if x.numel() > 0 else torch.device("cuda")
    l2_buffer = _l2_flush_buffer(device) if flush_l2 else None

    def _do_forward():
        ctx = MoeLoadBalancerIterContext(eplb) if eplb is not None else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            _ = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        _sync()
        for _ in range(warmup):
            if l2_buffer is not None:
                l2_buffer.zero_()
            _do_forward()
        for _ in range(iters):
            if l2_buffer is not None:
                l2_buffer.zero_()
            with torch.profiler.record_function("moe_forward"):
                _do_forward()

    _sync()
    return _parse_profiler_events_moe(list(prof.events()))


def _parse_profiler_events_moe(events_list: list) -> Tuple[List[float], Dict[str, Any]]:
    """Parse Kineto events with ``moe_forward`` ranges.

    Returns ``(moe_forward_times_us, detailed_stats)`` where ``detailed_stats``
    contains ``moe_forward_kernels`` (within the range) and ``other_kernels``.
    """

    def _is_gpu_event(evt) -> bool:
        return getattr(evt, "device_type", None) == DeviceType.CUDA

    gpu_moe_intervals: List[Tuple[int, int]] = []
    for evt in events_list:
        if not _is_gpu_event(evt) or evt.name != "moe_forward":
            continue
        tr = getattr(evt, "time_range", None)
        if tr is None or tr.end <= tr.start:
            continue
        gpu_moe_intervals.append((tr.start, tr.end))
    gpu_moe_intervals.sort()

    def _scope(evt) -> Optional[str]:
        tr = getattr(evt, "time_range", None)
        if tr is None:
            return None
        for s, e in gpu_moe_intervals:
            if s <= tr.start and tr.end <= e:
                return "moe_forward"
        return None

    moe_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}
    for evt in events_list:
        if not _is_gpu_event(evt):
            continue
        if evt.device_time <= 0 or evt.name == "moe_forward":
            continue
        scope = _scope(evt)
        bucket = moe_kernel_times if scope == "moe_forward" else other_kernel_times
        bucket.setdefault(evt.name, []).append(evt.device_time)

    forward_times_us: List[float] = []
    for evt in events_list:
        if _is_gpu_event(evt) and evt.name == "moe_forward":
            forward_times_us.append(evt.device_time)

    def _build(ktimes: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        out = [{"name": n, "count": len(t), "_times": t} for n, t in ktimes.items()]
        out.sort(
            key=lambda x: (sum(x["_times"]) / len(x["_times"])) if x["_times"] else 0.0,
            reverse=True,
        )
        return out

    detailed_stats = {
        "moe_forward_kernels": _build(moe_kernel_times),
        "other_kernels": _build(other_kernel_times),
    }
    return forward_times_us, detailed_stats


# ---------------------------------------------------------------------------
# CUPTI helpers (must init BEFORE the CUDA context)
# ---------------------------------------------------------------------------


def _try_init_cupti():
    """Initialize CUPTI before any CUDA context. Mirrors bench_moe_comm."""
    try:
        from functools import partial as _partial

        from cupti import cupti as _cupti

        _cupti_kernels: List[Tuple[str, int, int]] = []
        _cupti_events: List[int] = []

        def _buf_requested():
            return 8 * 1024 * 1024, 0

        def _buf_completed(kernels, events, activities):
            for act in activities:
                if act.kind == _cupti.ActivityKind.CONCURRENT_KERNEL:
                    kernels.append((act.name, act.start, act.end))
                elif act.kind == _cupti.ActivityKind.CUDA_EVENT:
                    events.append(act.device_timestamp)

        _cupti.activity_enable(_cupti.ActivityKind.CONCURRENT_KERNEL)
        _cupti.activity_enable(_cupti.ActivityKind.CUDA_EVENT)
        _cupti.activity_enable_cuda_event_device_timestamps(1)
        _cupti.activity_register_callbacks(
            _buf_requested, _partial(_buf_completed, _cupti_kernels, _cupti_events)
        )
        return _cupti, _cupti_kernels, _cupti_events, True
    except Exception:
        return None, [], [], False


def _demangle_names(names: List[str]) -> Dict[str, str]:
    try:
        import cxxfilt

        return {n: cxxfilt.demangle(n) for n in names}
    except Exception:
        return {n: n for n in names}


def _build_cuda_graph_kernel_stats_cupti(
    cupti_kernels: List[Tuple[str, int, int]],
    cupti_events: List[int],
    iters: int,
) -> Optional[Dict[str, Any]]:
    """Categorize replay kernels into moe_forward/other windows via CUPTI.

    The graph records 2 CUDA EXTERNAL events per timed iteration (none during
    warmup): event ``2*i+0`` -> moe_start, ``2*i+1`` -> moe_end.
    """
    expected_events = 2 * iters
    if len(cupti_events) != expected_events:
        _maybe_print_rank0(
            f"[bench_moe] CUPTI breakdown skipped: expected {expected_events} CUDA_EVENT "
            f"records ({iters} iters x 2) but got {len(cupti_events)}. "
            "Most likely CUPTI was registered after CUDA context creation."
        )
        return None
    if not cupti_kernels:
        return None

    starts_abs = [cupti_events[2 * i + 0] for i in range(iters)]
    ends_abs = [cupti_events[2 * i + 1] for i in range(iters)]

    unique_names = list({name for name, _, _ in cupti_kernels})
    dm = _demangle_names(unique_names)

    moe_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}

    iter_span: List[List[Optional[int]]] = [[None, None] for _ in range(iters)]

    for name, k_start, k_end in cupti_kernels:
        demangled = dm.get(name, name)
        device_time_us = (k_end - k_start) / 1e3

        iter_idx = -1
        for i in range(iters):
            if k_start >= starts_abs[i] and k_end <= ends_abs[i]:
                iter_idx = i
                break

        if iter_idx >= 0:
            span = iter_span[iter_idx]
            span[0] = k_start if span[0] is None else min(span[0], k_start)
            span[1] = k_end if span[1] is None else max(span[1], k_end)
            moe_kernel_times.setdefault(demangled, []).append(device_time_us)
        else:
            other_kernel_times.setdefault(demangled, []).append(device_time_us)

    def _build(ktimes: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        result = [{"name": n, "count": len(t), "_times": t} for n, t in ktimes.items()]
        result.sort(
            key=lambda x: (sum(x["_times"]) / len(x["_times"])) if x["_times"] else 0.0,
            reverse=True,
        )
        return result

    moe_times_us = [
        (span[1] - span[0]) / 1e3 if span[0] is not None else None for span in iter_span
    ]

    return {
        "moe_forward_kernels": _build(moe_kernel_times),
        "other_kernels": _build(other_kernel_times),
        "moe_times_us": moe_times_us,
    }


# ---------------------------------------------------------------------------
# CUDA Graph timing
# ---------------------------------------------------------------------------


def _time_moe_forward_cuda_graph(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    *,
    warmup: int,
    iters: int,
    cupti_ctx: Optional[Any] = None,
    flush_l2: bool = True,
) -> Tuple[List[float], Dict[str, Any]]:
    """Time ``moe.forward`` inside an unrolled CUDA graph with EXTERNAL events."""
    device = x.device if x.numel() > 0 else torch.device("cuda")
    l2_buffer = _l2_flush_buffer(device) if flush_l2 else None

    if cupti_ctx is not None:
        _cupti, _cupti_kernels, _cupti_events, _cupti_available = cupti_ctx
    else:
        _cupti = None
        _cupti_kernels = []
        _cupti_events = []
        _cupti_available = False

    # ---- 1. Shape-discovery eager pass ----
    with torch.inference_mode():
        eager_out = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
    if not isinstance(eager_out, torch.Tensor):
        # Fallback: backends that return tuple/list; not supported by graph capture here.
        raise RuntimeError(
            "CUDA-Graph timing requires a tensor output from moe.forward; got "
            f"{type(eager_out).__name__}. Use --no_cuda_graph for this case."
        )
    torch.cuda.synchronize()

    # ---- 2. Pre-create events; ``cudaEventRecordWithFlags`` makes graph-internal
    # events queryable via elapsed_time().
    _cudart = ctypes.CDLL("libcudart.so")
    _cudart.cudaEventRecordWithFlags.restype = ctypes.c_int
    _cudart.cudaEventRecordWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    _CUDA_EVENT_RECORD_EXTERNAL = 0x1

    def _record_external(event: torch.cuda.Event) -> None:
        stream = torch.cuda.current_stream()
        ret = _cudart.cudaEventRecordWithFlags(
            event.cuda_event, stream.cuda_stream, _CUDA_EVENT_RECORD_EXTERNAL
        )
        if ret != 0:
            raise RuntimeError(f"cudaEventRecordWithFlags failed with code {ret}")

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # Force lazy creation of the underlying CUDA events before capture.
    for evt in starts + ends:
        evt.record()
    torch.cuda.synchronize()

    # ---- 3. Capture big graph (warmup + iters). Only iters carry events. ----
    # ``inference_mode`` wraps the full capture region; toggling it per-iter
    # would add unnecessary Python work and is irrelevant inside replay.
    big_graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(big_graph):
        for _ in range(warmup):
            if l2_buffer is not None:
                l2_buffer.zero_()
            moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
        for i in range(iters):
            if l2_buffer is not None:
                l2_buffer.zero_()
            _record_external(starts[i])
            moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
            _record_external(ends[i])

    # ---- 4. Drain CUPTI activities before replay so the bucket starts clean ----
    if _cupti_available:
        _cupti.activity_flush_all(0)
        _cupti_kernels.clear()
        _cupti_events.clear()

    _sync()
    big_graph.replay()
    _sync()

    if _cupti_available:
        _cupti.activity_flush_all(0)

    forward_times_us = [starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters)]

    if _cupti_available:
        _cupti_kernels.sort(key=lambda k: k[1])
        _cupti_events.sort()
        cupti_stats = _build_cuda_graph_kernel_stats_cupti(_cupti_kernels, _cupti_events, iters)
        if cupti_stats is not None:
            cupti_times = cupti_stats.pop("moe_times_us")
            forward_times_us = [
                ct if ct is not None else et for ct, et in zip(cupti_times, forward_times_us)
            ]
            detailed_stats = cupti_stats
        else:
            detailed_stats = {"moe_forward_kernels": [], "other_kernels": []}
    else:
        detailed_stats = {"moe_forward_kernels": [], "other_kernels": []}

    return forward_times_us, detailed_stats


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def _gather_kernel_breakdown(
    detailed_stats: Dict[str, Any], iter_stats: bool
) -> Dict[str, List[Dict[str, Any]]]:
    """All-gather per-kernel timings across ranks and produce per-rank summaries."""
    categories = ("moe_forward_kernels", "other_kernels")
    local_payload: Dict[str, Dict[str, List[float]]] = {}
    for cat in categories:
        local_payload[cat] = {
            kernel["name"]: kernel.get("_times", []) for kernel in detailed_stats.get(cat, [])
        }
    all_payload = mpi_allgather(local_payload)

    merged: Dict[str, List[Dict[str, Any]]] = {}
    for cat in categories:
        seen = set()
        kernel_names: List[str] = []
        for rank_payload in all_payload:
            for name in rank_payload.get(cat, {}):
                if name not in seen:
                    seen.add(name)
                    kernel_names.append(name)

        kernels: List[Dict[str, Any]] = []
        for name in kernel_names:
            per_rank_times: List[List[float]] = []
            for rank_payload in all_payload:
                times = rank_payload.get(cat, {}).get(name, [])
                per_rank_times.append(times if isinstance(times, list) else [])

            if iter_stats:
                per_rank = {
                    f"rank{i}": _compute_stats(times) for i, times in enumerate(per_rank_times)
                }
            else:
                per_rank = {
                    f"rank{i}": (sum(times) / len(times) if times else 0.0)
                    for i, times in enumerate(per_rank_times)
                }

            kernels.append(
                {
                    "name": name,
                    "count": max((len(times) for times in per_rank_times), default=0),
                    "per_rank": per_rank,
                }
            )
        merged[cat] = kernels
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE module microbenchmark (MPI). Times ConfigurableMoE.forward."
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of MPI worker ranks to spawn (ignored under external mpirun).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_PROFILES.keys()),
        help="Named model profile (overridable via individual flags below).",
    )
    # Per-field overrides
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--n_group", type=int, default=None)
    parser.add_argument("--topk_group", type=int, default=None)
    parser.add_argument(
        "--quant",
        type=lambda s: QuantAlgo[str(s).upper()] if s is not None else None,
        default=None,
        choices=[q.name for q in QuantAlgo],
        help="Quantization algorithm.",
    )
    parser.add_argument(
        "--routing_method",
        type=lambda s: str(s).upper(),
        default="RENORMALIZE",
        choices=sorted(_ROUTING_METHODS),
        help=(
            "Routing method. Defaults to RENORMALIZE for every model; pass "
            "DEEPSEEK_V3 / MINIMAX_M2 / etc. to switch grouped or sigmoid "
            "routing for the same model profile."
        ),
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        nargs="+",
        required=True,
        help="GLOBAL total tokens per case; per-rank counts are derived via imbalance ratio.",
    )

    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="DEP",
        choices=("DEP", "TEP", "DTP", "TTP"),
        help="Parallel mode (combined with world_size).",
    )
    parser.add_argument("--moe_ep_size", type=int, default=None)
    parser.add_argument("--moe_tp_size", type=int, default=None)
    parser.add_argument("--enable_attention_dp", action="store_true")

    parser.add_argument("--enable_eplb", action="store_true")
    parser.add_argument("--num_slots", type=int, default=-1)
    parser.add_argument("--layer_updates_per_iter", type=int, default=-1)

    parser.add_argument(
        "--backend",
        type=lambda s: str(s).upper(),
        default="BEST",
        choices=_ALL_BACKENDS + ["BEST"],
        help=(
            "Backend to bench. With ``BEST`` (default) every backend is "
            "pre-checked via ``can_implement``; unsupported backends are "
            "skipped, runnable ones are timed, and the fastest is reported "
            "per num_tokens with unsupported / failed backends surfaced as "
            "``time_us=None`` plus a skip_reason. A specific backend value "
            "(CUTLASS, TRTLLM, ...) errors out immediately when "
            "``can_implement`` rejects the configuration."
        ),
    )

    parser.add_argument(
        "--tokens_per_rank_imbalance_ratio",
        type=float,
        default=0.0,
        help="In [0, 1]. 0=evenly distributed; 1=rank 0 takes everything.",
    )
    parser.add_argument(
        "--experts_hot_imbalane_ratio",
        type=float,
        default=0.0,
        help="In [0, 1]. 0=perfect router; >0 concentrates tokens on hot experts.",
    )

    parser.add_argument(
        "--no_cuda_graph",
        dest="cuda_graph",
        action="store_false",
        default=True,
        help=(
            "Disable CUDA-Graph capture and use eager timing. CUDA-Graph "
            "capture is ON by default; pass this flag to opt out (required "
            "for dynamic EPLB)."
        ),
    )

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--fast_autotune", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16"),
    )
    parser.add_argument("--use_low_precision_moe_combine", action="store_true")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--iter_stats", action="store_true")
    parser.add_argument(
        "--kernel_breakdown",
        dest="kernel_breakdown",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no_kernel_breakdown", dest="kernel_breakdown", action="store_false")
    parser.add_argument(
        "--force_balance_patch",
        action="store_true",
        help="Force the balanced routing patch even when experts_hot_imbalane_ratio == 0.",
    )
    parser.add_argument(
        "--force_comm_method",
        type=str,
        default=None,
        choices=(
            "NVLINK_ONE_SIDED",
            "NVLINK_TWO_SIDED",
            "DEEPEP",
            "DEEPEPLOWLATENCY",
            "ALLGATHER",
        ),
        help="Force a specific MoE comm method via TRTLLM_FORCE_COMM_METHOD.",
    )
    parser.add_argument("--output_file", type=str, default=None)

    return parser.parse_args()


def _resolve_profile(args: argparse.Namespace) -> ModelProfile:
    """Merge ``--model`` profile with per-field overrides into a final profile."""
    base = MODEL_PROFILES.get(args.model) if args.model is not None else None

    if base is None:
        missing = [
            f
            for f in ("num_experts", "top_k", "hidden_size", "intermediate_size", "quant")
            if getattr(args, f) is None
        ]
        if missing:
            raise ValueError(f"No --model selected; you must also pass: {', '.join(missing)}")
        moe_model_config = MoeModelConfig(
            num_experts=int(args.num_experts),
            top_k=int(args.top_k),
            hidden_size=int(args.hidden_size),
            intermediate_size=int(args.intermediate_size),
            n_group=args.n_group,
            topk_group=args.topk_group,
        )
        routing_cls = (
            _ROUTING_METHODS[args.routing_method]
            if args.routing_method
            else RenormalizeMoeRoutingMethod
        )
        return ModelProfile(
            name="custom",
            moe_model_config=moe_model_config,
            quant_algo=args.quant,
            routing_method_cls=routing_cls,
        )

    # Per-field overrides
    moe_model_config = MoeModelConfig(
        num_experts=int(args.num_experts)
        if args.num_experts is not None
        else base.moe_model_config.num_experts,
        top_k=int(args.top_k) if args.top_k is not None else base.moe_model_config.top_k,
        hidden_size=int(args.hidden_size)
        if args.hidden_size is not None
        else base.moe_model_config.hidden_size,
        intermediate_size=int(args.intermediate_size)
        if args.intermediate_size is not None
        else base.moe_model_config.intermediate_size,
        n_group=args.n_group if args.n_group is not None else base.moe_model_config.n_group,
        topk_group=args.topk_group
        if args.topk_group is not None
        else base.moe_model_config.topk_group,
    )
    routing_cls = (
        _ROUTING_METHODS[args.routing_method] if args.routing_method else base.routing_method_cls
    )
    return ModelProfile(
        name=base.name,
        moe_model_config=moe_model_config,
        quant_algo=args.quant if args.quant is not None else base.quant_algo,
        routing_method_cls=routing_cls,
        swiglu_alpha=base.swiglu_alpha,
        swiglu_beta=base.swiglu_beta,
        swiglu_limit=base.swiglu_limit,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.tokens_per_rank_imbalance_ratio <= 1.0):
        raise ValueError("--tokens_per_rank_imbalance_ratio must be in [0, 1]")
    if not (0.0 <= args.experts_hot_imbalane_ratio <= 1.0):
        raise ValueError("--experts_hot_imbalane_ratio must be in [0, 1]")
    if args.enable_eplb and args.experts_hot_imbalane_ratio > 0.0:
        raise ValueError(
            "--enable_eplb is incompatible with --experts_hot_imbalane_ratio > 0; "
            "they have conflicting goals (rebalance vs. force imbalance)."
        )
    if args.cuda_graph and args.enable_eplb and int(args.layer_updates_per_iter) > 0:
        raise ValueError(
            "Dynamic EPLB (layer_updates_per_iter > 0) is incompatible with "
            "the default CUDA-Graph timing path; pass --no_cuda_graph to "
            "enable eager timing for dynamic EPLB runs."
        )
    if any(t < 0 for t in args.num_tokens):
        raise ValueError("--num_tokens entries must be >= 0")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_benchmark_worker_under_current_mpi(args: argparse.Namespace, launcher: str) -> None:
    # CUPTI MUST be initialized before the CUDA context is created.
    _early_cupti_ctx: Optional[Any] = None
    if args.cuda_graph:
        _cupti_module, _cupti_kernels_list, _cupti_events_list, _cupti_ok = _try_init_cupti()
        if _cupti_ok:
            _early_cupti_ctx = (_cupti_module, _cupti_kernels_list, _cupti_events_list, True)

    tllm.logger.set_level("error")

    world_size = mpi_world_size()
    rank = mpi_rank()
    _set_device_from_local_rank()
    device = torch.device("cuda")
    seed = int(args.random_seed) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    profile = _resolve_profile(args)
    _validate_args(args)

    act_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # DeepSeekV3 routing on the TRTLLM backend requires fp32 logits.
    routing_logits_dtype = (
        torch.float32 if profile.routing_method_cls is DeepSeekV3MoeRoutingMethod else act_dtype
    )
    # Mirror the swiglu_gptoss_style derivation from ``_build_moe_module`` so
    # the pre-flight ``can_implement`` check evaluates the same triple that the
    # actual MoE construction will use later.
    swiglu_gptoss_style = (
        profile.swiglu_alpha != 1.0
        or profile.swiglu_beta != 0.0
        or profile.swiglu_limit != float("inf")
    )

    mapping = _build_mapping(args, world_size)
    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()

    # Pre-flight ``can_implement`` gate: every requested backend must report
    # support for ``(quant_algo, act_dtype, swiglu_gptoss_style)`` before we
    # spend the construction / autotune cost. The explicit branch matters:
    #   * fixed backend  -> raise so the user gets a clear, immediate error;
    #   * --backend BEST -> filter out unsupported backends, but remember them
    #                       so they still appear in the final ranking with a
    #                       ``None`` time and a skip reason.
    unsupported_backends: Dict[str, str] = {}
    if args.backend == "BEST":
        backends_to_run: List[str] = []
        for candidate in _ALL_BACKENDS:
            ok, skip_reason = _check_backend_can_implement(
                candidate, profile.quant_algo, act_dtype, swiglu_gptoss_style
            )
            if ok:
                backends_to_run.append(candidate)
            else:
                unsupported_backends[candidate] = skip_reason or "unsupported"
                _maybe_print_rank0(
                    "[bench_moe] BEST: skipping backend "
                    f"{candidate!r} (can_implement=False): {skip_reason}"
                )
        if not backends_to_run:
            raise RuntimeError(
                "No MoE backend can implement this configuration: "
                + ", ".join(f"{b}: {r}" for b, r in unsupported_backends.items())
            )
    else:
        ok, skip_reason = _check_backend_can_implement(
            args.backend, profile.quant_algo, act_dtype, swiglu_gptoss_style
        )
        if not ok:
            raise RuntimeError(
                f"--backend {args.backend!r} cannot implement this configuration: {skip_reason}"
            )
        backends_to_run = [args.backend]

    max_global_tokens = max(args.num_tokens) if args.num_tokens else 0
    max_local_tokens = max(
        _distribute_tokens(t, world_size, args.tokens_per_rank_imbalance_ratio)[mapping.rank]
        if t > 0
        else 0
        for t in args.num_tokens
    )
    max_num_tokens = max(max_local_tokens, 1)

    benchmark_metadata = {
        "bench": "bench_moe",
        "launcher": launcher,
        "model": profile.name,
        "moe_model_config": str(profile.moe_model_config),
        "world_size": world_size,
        "parallel_mode": args.parallel_mode,
        "moe_ep_size": mapping.moe_ep_size,
        "moe_tp_size": mapping.moe_tp_size,
        "enable_attention_dp": bool(mapping.enable_attention_dp),
        "tokens_per_rank_imbalance_ratio": float(args.tokens_per_rank_imbalance_ratio),
        "experts_hot_imbalane_ratio": float(args.experts_hot_imbalane_ratio),
        "perfect_router_used": bool(
            args.experts_hot_imbalane_ratio == 0.0 and not args.force_balance_patch
        ),
        "cuda_graph": bool(args.cuda_graph),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "enable_eplb": bool(args.enable_eplb),
        "quant_algo": (profile.quant_algo.name if profile.quant_algo is not None else None),
        "dtype": str(act_dtype),
        "routing_method": profile.routing_method_cls.__name__,
        "num_tokens_list": list(map(int, args.num_tokens)),
        "max_global_tokens": int(max_global_tokens),
        "random_seed": int(args.random_seed),
        "device_count": torch.cuda.device_count(),
    }
    if rank == 0:
        print(json.dumps(benchmark_metadata, indent=2), flush=True)

    # Optional force-comm env hook.
    prev_force_comm = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
    if args.force_comm_method is not None:
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = args.force_comm_method

    iter_stats = bool(args.iter_stats)
    all_results: List[Dict[str, Any]] = []
    # ``case_outcome[(backend, total_tokens)]`` holds the slowest-rank-mean in
    # microseconds for successful runs, or ``None`` when the backend could not
    # produce a timing for that token count (unsupported, build failure,
    # silent fallback to another backend, or runtime exception in the timed
    # phase). Tracking ``None`` makes every backend visible in the final BEST
    # ranking instead of silently dropping it.
    case_outcome: Dict[Tuple[str, int], Optional[float]] = {}
    skip_reasons: Dict[Tuple[str, int], str] = {}
    for skipped_backend, skip_reason_text in unsupported_backends.items():
        for total_tokens in args.num_tokens:
            case_outcome[(skipped_backend, total_tokens)] = None
            skip_reasons[(skipped_backend, total_tokens)] = skip_reason_text

    try:
        for backend in backends_to_run:
            for total_tokens in args.num_tokens:
                per_rank = _distribute_tokens(
                    total_tokens, world_size, args.tokens_per_rank_imbalance_ratio
                )
                local_num_tokens = per_rank[mapping.rank]
                all_rank_num_tokens = list(per_rank)

                case_label = f"backend={backend} total_tokens={total_tokens} per_rank={per_rank}"

                moe = moe_load_balancer = None
                try:
                    moe, moe_load_balancer, _initial_expert_ids, _ = _build_moe_module(
                        profile=profile,
                        mapping=mapping,
                        quant_algo=profile.quant_algo,
                        moe_backend=backend,
                        use_cuda_graph=bool(args.cuda_graph),
                        max_num_tokens=max_num_tokens,
                        use_low_precision_moe_combine=bool(args.use_low_precision_moe_combine),
                        enable_eplb=bool(args.enable_eplb),
                        num_slots=int(args.num_slots),
                        layer_updates_per_iter=int(args.layer_updates_per_iter),
                        enable_perfect_router=(
                            args.experts_hot_imbalane_ratio == 0.0 and not args.force_balance_patch
                        ),
                        dtype=act_dtype,
                        routing_logits_dtype=routing_logits_dtype,
                        device=device,
                    )
                except Exception as exc:
                    reason = f"build error: {type(exc).__name__}: {exc}"
                    _maybe_print_rank0(f"[bench_moe] Skipping {case_label}: {reason}")
                    case_outcome[(backend, total_tokens)] = None
                    skip_reasons[(backend, total_tokens)] = reason
                    continue

                actual_backend = _backend_name_from_module(moe)
                if actual_backend != backend.upper():
                    reason = (
                        f"requested backend {backend!r} fell back to "
                        f"{actual_backend!r}; not ranking under requested name."
                    )
                    _maybe_print_rank0(f"[bench_moe] Skipping {case_label}: {reason}")
                    case_outcome[(backend, total_tokens)] = None
                    skip_reasons[(backend, total_tokens)] = reason
                    try:
                        moe.destroy()
                    except Exception:
                        pass
                    continue

                try:
                    x, router_logits = _make_inputs(
                        local_num_tokens,
                        profile.moe_model_config.hidden_size,
                        profile.moe_model_config.num_experts,
                        act_dtype,
                        routing_logits_dtype,
                        device,
                    )

                    with _maybe_install_balance_patch(
                        moe,
                        mapping,
                        profile.moe_model_config.num_experts,
                        profile.moe_model_config.top_k,
                        float(args.experts_hot_imbalane_ratio),
                        bool(args.force_balance_patch),
                    ):
                        # Autotune (untimed)
                        try:
                            _run_autotune(
                                moe,
                                x,
                                router_logits,
                                all_rank_num_tokens,
                                moe_load_balancer,
                                bool(args.fast_autotune),
                            )
                        except Exception as exc:
                            _maybe_print_rank0(
                                f"[bench_moe] Autotune skipped for {case_label}: "
                                f"{type(exc).__name__}: {exc}"
                            )

                        # Timed phase
                        try:
                            if args.cuda_graph:
                                fwd_times_us, detailed_stats = _time_moe_forward_cuda_graph(
                                    moe,
                                    x,
                                    router_logits,
                                    all_rank_num_tokens,
                                    warmup=int(args.warmup),
                                    iters=int(args.iters),
                                    cupti_ctx=_early_cupti_ctx,
                                )
                            else:
                                fwd_times_us, detailed_stats = _time_moe_forward_eager(
                                    moe,
                                    x,
                                    router_logits,
                                    all_rank_num_tokens,
                                    warmup=int(args.warmup),
                                    iters=int(args.iters),
                                    eplb=moe_load_balancer,
                                )
                        except Exception as exc:
                            reason = f"timed phase error: {type(exc).__name__}: {exc}"
                            _maybe_print_rank0(
                                f"[bench_moe] Skipping {case_label}: {reason}\n"
                                f"{traceback.format_exc()}"
                            )
                            case_outcome[(backend, total_tokens)] = None
                            skip_reasons[(backend, total_tokens)] = reason
                            continue

                    per_rank_stats = _gather_per_rank(fwd_times_us, iter_stats=iter_stats)

                    output: Dict[str, Any] = {
                        "num_tokens": int(total_tokens),
                        "per_rank_num_tokens": [int(v) for v in per_rank],
                        "requested_backend": backend,
                        "actual_backend": actual_backend,
                        "moe_forward_us": per_rank_stats,
                    }

                    if args.kernel_breakdown:
                        kb = _gather_kernel_breakdown(detailed_stats, iter_stats=iter_stats)
                        output.update(kb)

                    if rank == 0:
                        print(json.dumps(output, indent=2), flush=True)
                        all_results.append(output)

                        # Track per-num_tokens slowest-rank-mean for winner selection.
                        if isinstance(per_rank_stats, dict) and per_rank_stats:
                            if iter_stats:
                                rank_means = [v.get("mean", 0.0) for v in per_rank_stats.values()]
                            else:
                                rank_means = list(per_rank_stats.values())
                            score = max(rank_means) if rank_means else 0.0
                            case_outcome[(backend, total_tokens)] = float(score)

                finally:
                    if moe is not None:
                        try:
                            moe.destroy()
                        except Exception:
                            pass
                    if moe_load_balancer is not None:
                        try:
                            moe_load_balancer.shutdown()
                        except Exception:
                            pass
    finally:
        if args.force_comm_method is not None:
            if prev_force_comm is None:
                os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
            else:
                os.environ["TRTLLM_FORCE_COMM_METHOD"] = prev_force_comm

    # Emit best-backend ranking when requested. Every candidate backend is
    # surfaced for each ``num_tokens``: successful timings sort ascending,
    # unsupported / failed backends sort to the tail with ``time_us=None`` and
    # the collected ``skip_reason`` so the operator can see *why* a backend
    # was excluded instead of guessing from missing rows.
    if rank == 0 and args.backend == "BEST":
        for total_tokens in args.num_tokens:
            per_backend: List[Tuple[str, Optional[float]]] = [
                (b, case_outcome.get((b, total_tokens))) for b in _ALL_BACKENDS
            ]
            ranking = sorted(
                per_backend,
                key=lambda p: (p[1] is None, p[1] if p[1] is not None else 0.0),
            )
            best_backend = next((b for b, v in ranking if v is not None), None)
            entry: Dict[str, Any] = {
                "num_tokens": int(total_tokens),
                "best_backend": best_backend,
                "ranking": [
                    {
                        "backend": b,
                        "time_us": (float(v) if v is not None else None),
                        "skip_reason": skip_reasons.get((b, total_tokens)),
                    }
                    for b, v in ranking
                ],
            }
            print(json.dumps(entry, indent=2), flush=True)
            all_results.append(entry)

    if rank == 0 and args.output_file and all_results:
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(
                {"benchmark_metadata": benchmark_metadata, "results": all_results}, f, indent=2
            )
        print(f"Report written to {args.output_file}", flush=True)


# ---------------------------------------------------------------------------
# MPI launchers
# ---------------------------------------------------------------------------


_WORKER_ENV = {
    "TRTLLM_CAN_USE_DEEP_EP": "1",
    "TRTLLM_ENABLE_PDL": "0",
}


def _spawn_worker_main(args_blob: bytes) -> List[Dict[str, Any]]:
    args = pickle.loads(args_blob)
    try:
        _run_benchmark_worker_under_current_mpi(args, launcher="spawn")
    except Exception as exc:
        rank = mpi_rank()
        size = mpi_world_size()
        msg = (
            "[bench_moe worker] uncaught exception:\n"
            f"rank={rank}/{size} local_rank={local_mpi_rank()} pid={os.getpid()}\n"
            f"{traceback.format_exc()}"
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        raise RuntimeError(msg) from exc
    return []


def main() -> None:
    args = parse_args()

    external_world_size = mpi_world_size()
    if external_world_size > 1:
        # Running under external mpirun/srun: no spawn pool, no cloudpickle.
        if args.world_size is not None and int(args.world_size) != external_world_size:
            raise ValueError(
                f"--world_size ({args.world_size}) must match external MPI world size "
                f"({external_world_size}) under mpirun."
            )
        os.environ.update(_WORKER_ENV)
        _run_benchmark_worker_under_current_mpi(args, launcher="external_mpi")
        return

    world_size = int(args.world_size or 1)
    if world_size <= 0:
        raise ValueError("--world_size must be > 0")

    # Single-rank fast path: there is nothing to spawn, and on some sites
    # MPI_Comm_Spawn is disabled under srun/pmix containers. Run the worker
    # inline so single-rank cases (design §11 case 1/2/4/5) do not require
    # cloudpickle or a spawn-capable PMIx layer.
    if world_size == 1:
        if mpi_rank() == 0:
            print(
                json.dumps(
                    {
                        "bench": "bench_moe",
                        "launcher": "inline_single_rank",
                        "world_size": 1,
                        "model": args.model,
                        "backend": args.backend,
                        "num_tokens": list(map(int, args.num_tokens)),
                    },
                    indent=2,
                ),
                flush=True,
            )
        os.environ.update(_WORKER_ENV)
        args.world_size = 1
        _run_benchmark_worker_under_current_mpi(args, launcher="inline_single_rank")
        return

    # Multi-rank self-spawn launcher: pull in cloudpickle + MPIPoolExecutor only here.
    import cloudpickle
    from mpi4py.futures import MPIPoolExecutor

    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(  # type: ignore[attr-defined]
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )

    if mpi_rank() == 0:
        print(
            json.dumps(
                {
                    "bench": "bench_moe",
                    "launcher": "spawn",
                    "world_size": world_size,
                    "model": args.model,
                    "backend": args.backend,
                    "num_tokens": list(map(int, args.num_tokens)),
                },
                indent=2,
            ),
            flush=True,
        )

    args_blob = cloudpickle.dumps(args)
    executor = MPIPoolExecutor(max_workers=world_size, env=_WORKER_ENV)
    try:
        _ = list(executor.map(_spawn_worker_main, [args_blob] * world_size))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
