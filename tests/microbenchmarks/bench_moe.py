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

"""MoE microbenchmark (MPI) — dashboard-oriented data generator.

Times ``ConfigurableMoE.forward`` (routing + dispatch + GroupGEMM + activation
+ combine) across a structured search space of MoE configurations and emits
JSON suitable for dashboard ingestion.

The search space is:

```text
backend x communication x EPLB x parallel mode x CUDA graph x combine precision
```

For each ``(model, num_tokens, workload modifier)`` point the benchmark
generates a list of candidate ``ConfigSpec`` instances, prunes invalid ones
via backend ``can_implement`` and capability checks, runs the survivors, and
records both the requested and the actually executed configuration alongside
timing.

See ``tests/microbenchmarks/BENCH_MOE_DASHBOARD_DESIGN.md`` for the full
design.

Launch examples::

    # Single-rank fixed run (eager).
    python tests/microbenchmarks/bench_moe.py \
        --world_size 1 --model qwen1.5_moe \
        --backend CUTLASS --num_tokens 16 64 --no_cuda_graph

    # Backend search over a token sweep.
    python tests/microbenchmarks/bench_moe.py \
        --world_size 4 --parallel_mode DEP --model deepseek_v3 \
        --search backend --num_tokens 64 256

    # Full dashboard sweep driven by a JSON config file.
    python tests/microbenchmarks/bench_moe.py \
        --config_file configs/moe_dashboard_deepseek_v3.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import getpass
import itertools
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.autograd import DeviceType

# ``cloudpickle`` and ``mpi4py.futures.MPIPoolExecutor`` are only used in the
# self-spawning launcher path (single ``python3 bench_moe.py`` invocation that
# pops up its own MPI pool). When started under external mpirun/srun they are
# never touched, so import them lazily inside ``main()`` to avoid hard
# dependencies on container images that ship only the minimal mpi4py.

# ``quantize_utils.py`` lives under ``tests/unittest/_torch/modules/moe/`` and
# uses pytest-style relative imports such as ``from _torch.helpers import ...``;
# the project's ``tests/unittest/conftest.py`` puts ``tests/unittest`` on
# sys.path. Replicate that here so the benchmark works without pytest.
_TESTS_UNITTEST_DIR = Path(__file__).resolve().parent.parent / "unittest"
if str(_TESTS_UNITTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_UNITTEST_DIR))

# When invoked as ``python tests/microbenchmarks/bench_moe.py`` (rather than
# ``python -m`` or via pytest), Python sets ``sys.path[0]`` to the script's
# directory (``tests/microbenchmarks/``), not the repo root. Adding the repo
# root first makes ``import tensorrt_llm`` resolve to the in-tree checkout
# without requiring an installed wheel or a manual ``PYTHONPATH``.
#
# Only do this when the in-tree ``tensorrt_llm`` package can be imported as a
# fully built package (i.e., the worktree contains compiled ``bindings``). On
# OCI / pre-built container environments where ``tensorrt_llm`` is installed
# system-wide and the worktree is just a source checkout, leaving the worktree
# off sys.path is correct: the installed wheel is used and bindings resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if (_REPO_ROOT / "tensorrt_llm" / "bindings").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
from tensorrt_llm._torch.modules.fused_moe.interface import (  # noqa: E402
    MoESchedulerKind,
    MoEWeightLoadingMode,
)
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
# Output schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Routing method registry
# ---------------------------------------------------------------------------

_ROUTING_METHODS: Dict[str, type] = {
    "DEFAULT": DefaultMoeRoutingMethod,
    "RENORMALIZE": RenormalizeMoeRoutingMethod,
    "RENORMALIZE_NAIVE": RenormalizeNaiveMoeRoutingMethod,
    "LLAMA4_RENORMALIZE": Llama4RenormalizeMoeRoutingMethod,
    "DEEPSEEK_V3": DeepSeekV3MoeRoutingMethod,
    "MINIMAX_M2": MiniMaxM2MoeRoutingMethod,
    "SIGMOID_RENORM": SigmoidRenormMoeRoutingMethod,
}

_ROUTING_NAME_BY_CLS: Dict[type, str] = {cls: name for name, cls in _ROUTING_METHODS.items()}


# All ConfigurableMoE-eligible backends (string values such as "CUTLASS").
_ALL_BACKENDS: List[str] = [b.value for b in MoeBackendType]

# All comm method names accepted by the per-case ``comm_method`` field. ``AUTO``
# defers to ``CommunicationFactory``; ``NONE`` is the introspection sentinel
# emitted when no host-side Communication object exists (single rank, non-DP
# mappings, FUSED_COMM scheduler).
_COMM_METHODS: Tuple[str, ...] = (
    "AUTO",
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
    "DEEPEP",
    "DEEPEPLOWLATENCY",
    "ALLGATHER",
)

# Forced communication methods that ``CommunicationFactory`` understands via
# ``TRTLLM_FORCE_COMM_METHOD``. ``AUTO`` and ``ALLGATHER`` map to no force /
# implicit fallback paths and are not pushed through the env var.
_FORCED_COMM_ENV_VALUES: Dict[str, str] = {
    "NVLINK_ONE_SIDED": "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED": "NVLINK_TWO_SIDED",
    "DEEPEP": "DEEPEP",
    "DEEPEPLOWLATENCY": "DEEPEPLOWLATENCY",
    "ALLGATHER": "ALLGATHER",
}


# ---------------------------------------------------------------------------
# Structured specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Static MoE model description.

    A built-in name resolves to one of the entries in ``BUILT_IN_MODELS``.
    Custom shapes pass ``name="custom"`` and fill the remaining fields
    explicitly. ``routing_method`` is the registry key from
    ``_ROUTING_METHODS``; resolution to a concrete class happens lazily so the
    spec stays JSON-serializable.
    """

    name: str
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    quant_algo: Optional[str]
    routing_method: str
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    swiglu_limit: float = float("inf")

    @property
    def routing_method_cls(self) -> type:
        return _ROUTING_METHODS[self.routing_method]

    @property
    def quant_algo_enum(self) -> Optional[QuantAlgo]:
        return QuantAlgo[self.quant_algo] if self.quant_algo is not None else None

    @property
    def swiglu_gptoss_style(self) -> bool:
        return (
            self.swiglu_alpha != 1.0 or self.swiglu_beta != 0.0 or self.swiglu_limit != float("inf")
        )

    def to_moe_model_config(self) -> MoeModelConfig:
        return MoeModelConfig(
            num_experts=int(self.num_experts),
            top_k=int(self.top_k),
            hidden_size=int(self.hidden_size),
            intermediate_size=int(self.intermediate_size),
            n_group=self.n_group,
            topk_group=self.topk_group,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
            "hidden_size": int(self.hidden_size),
            "intermediate_size": int(self.intermediate_size),
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "quant_algo": self.quant_algo,
            "routing_method": self.routing_method,
            "routing_method_class": self.routing_method_cls.__name__,
            "swiglu_alpha": float(self.swiglu_alpha),
            "swiglu_beta": float(self.swiglu_beta),
            "swiglu_limit": float(self.swiglu_limit),
        }


@dataclass(frozen=True)
class WorkloadSpec:
    """Workload for one timing case after the model is fixed."""

    num_tokens: int
    token_unit: str = "global"  # "global" | "per_rank"
    rank_distribution: str = "balanced"  # "balanced" | "rank0_hot"
    rank_hotspot_ratio: float = 0.0
    expert_distribution: str = "balanced_patch"  # "balanced_patch" | "hotspot"
    expert_hotspot_ratio: float = 0.0

    def to_dict(self, per_rank_num_tokens: Optional[List[int]] = None) -> Dict[str, Any]:
        return {
            "num_tokens": int(self.num_tokens),
            "token_unit": self.token_unit,
            "rank_distribution": self.rank_distribution,
            "rank_hotspot_ratio": float(self.rank_hotspot_ratio),
            "expert_distribution": self.expert_distribution,
            "expert_hotspot_ratio": float(self.expert_hotspot_ratio),
            "per_rank_num_tokens": (
                [int(v) for v in per_rank_num_tokens] if per_rank_num_tokens is not None else None
            ),
        }


@dataclass(frozen=True)
class ConfigSpec:
    """One executable MoE runtime configuration."""

    backend: str
    parallel_mode: str  # "DEP" | "TEP" | "DTP" | "TTP" | "CUSTOM"
    moe_ep_size: Optional[int] = None
    moe_tp_size: Optional[int] = None
    enable_attention_dp: Optional[bool] = None
    comm_method: str = "AUTO"
    eplb_mode: str = "off"  # "off" | "static" | "dynamic"
    num_slots: Optional[int] = None
    layer_updates_per_iter: Optional[int] = None
    cuda_graph: bool = True
    use_low_precision_moe_combine: bool = False
    per_case_subprocess: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "parallel_mode": self.parallel_mode,
            "moe_ep_size": self.moe_ep_size,
            "moe_tp_size": self.moe_tp_size,
            "enable_attention_dp": self.enable_attention_dp,
            "comm_method": self.comm_method,
            "eplb_mode": self.eplb_mode,
            "num_slots": self.num_slots,
            "layer_updates_per_iter": self.layer_updates_per_iter,
            "cuda_graph": bool(self.cuda_graph),
            "use_low_precision_moe_combine": bool(self.use_low_precision_moe_combine),
            "per_case_subprocess": bool(self.per_case_subprocess),
        }


@dataclass(frozen=True)
class SearchSpec:
    """Description of which ConfigSpec axes to expand into candidates."""

    mode: str = "none"  # "none" | "backend" | "comm" | "parallel" | "eplb" | "full"
    backends: Tuple[str, ...] = ()
    parallel_modes: Tuple[str, ...] = ()
    comm_methods: Tuple[str, ...] = ()
    eplb_modes: Tuple[str, ...] = ()
    cuda_graph_options: Tuple[bool, ...] = ()
    combine_precision_options: Tuple[bool, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "backends": list(self.backends),
            "parallel_modes": list(self.parallel_modes),
            "comm_methods": list(self.comm_methods),
            "eplb_modes": list(self.eplb_modes),
            "cuda_graph_options": [bool(v) for v in self.cuda_graph_options],
            "combine_precision_options": [bool(v) for v in self.combine_precision_options],
        }


@dataclass
class RunResult:
    """Result of timing a single ``(model, workload, config)`` triple.

    Mirrors the dashboard schema described in ``BENCH_MOE_DASHBOARD_DESIGN.md``
    but stays a regular dataclass on the worker side so we can mutate fields
    while incrementally collecting data.
    """

    model: ModelSpec
    workload: WorkloadSpec
    config: ConfigSpec
    status: str = "success"  # "success" | "skipped" | "failed"
    skip_reason: Optional[str] = None
    actual_backend: Optional[str] = None
    actual_comm_method: Optional[str] = None
    actual_comm_fallback_reason: Optional[str] = None
    scheduler_kind: Optional[str] = None
    moe_ep_size: Optional[int] = None
    moe_tp_size: Optional[int] = None
    enable_attention_dp: Optional[bool] = None
    num_chunks: Optional[int] = None
    per_rank_num_tokens: List[int] = field(default_factory=list)
    status_per_rank: Dict[str, str] = field(default_factory=dict)
    instrumentation: Dict[str, Any] = field(default_factory=dict)
    latency_us: Dict[str, Any] = field(default_factory=dict)
    phase_times_us: Dict[str, Any] = field(default_factory=dict)
    kernel_breakdown: Dict[str, Any] = field(default_factory=dict)
    overlap: Dict[str, Any] = field(default_factory=dict)
    bottleneck: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in model registry
# ---------------------------------------------------------------------------

BUILT_IN_MODELS: Dict[str, ModelSpec] = {
    "qwen1.5_moe": ModelSpec(
        name="qwen1.5_moe",
        num_experts=60,
        top_k=4,
        hidden_size=2048,
        intermediate_size=1408,
        quant_algo="FP8",
        routing_method="RENORMALIZE",
    ),
    "deepseek_v2_lite": ModelSpec(
        name="deepseek_v2_lite",
        num_experts=64,
        top_k=6,
        hidden_size=2048,
        intermediate_size=1408,
        quant_algo="FP8_BLOCK_SCALES",
        routing_method="DEEPSEEK_V3",
    ),
    "deepseek_v3": ModelSpec(
        name="deepseek_v3",
        num_experts=256,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        quant_algo="FP8_BLOCK_SCALES",
        routing_method="DEEPSEEK_V3",
        n_group=8,
        topk_group=4,
    ),
    "kimi_k2": ModelSpec(
        name="kimi_k2",
        num_experts=384,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        quant_algo="FP8_BLOCK_SCALES",
        routing_method="DEEPSEEK_V3",
    ),
    # DeepSeek-V4-Pro: 1.6T total / 49B activated. quant_algo intentionally
    # left None: pass --quant on the CLI to pin the mode (the released
    # checkpoint mixes FP4 experts with FP8 elsewhere which has no single
    # QuantAlgo match).
    "deepseek_v4_pro": ModelSpec(
        name="deepseek_v4_pro",
        num_experts=384,
        top_k=6,
        hidden_size=7168,
        intermediate_size=3072,
        quant_algo=None,
        routing_method="RENORMALIZE",
    ),
    # DeepSeek-V4-Flash: 284B total / 13B activated.
    "deepseek_v4_flash": ModelSpec(
        name="deepseek_v4_flash",
        num_experts=256,
        top_k=6,
        hidden_size=4096,
        intermediate_size=2048,
        quant_algo=None,
        routing_method="RENORMALIZE",
    ),
    "mixtral_8x7b": ModelSpec(
        name="mixtral_8x7b",
        num_experts=8,
        top_k=2,
        hidden_size=4096,
        intermediate_size=14336,
        quant_algo="FP8",
        routing_method="RENORMALIZE",
    ),
    "gpt_oss_120b": ModelSpec(
        name="gpt_oss_120b",
        num_experts=128,
        top_k=4,
        hidden_size=2880,
        intermediate_size=2880,
        quant_algo="W4A8_MXFP4_MXFP8",
        routing_method="RENORMALIZE",
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    ),
}


# ---------------------------------------------------------------------------
# Backend capability gate
# ---------------------------------------------------------------------------


def _check_backend_can_implement(
    backend_str: str,
    quant_algo: Optional[QuantAlgo],
    dtype_activation: torch.dtype,
    swiglu_gptoss_style: bool,
) -> Tuple[bool, Optional[str]]:
    """Resolve ``backend_str`` to its MoE class and forward to ``can_implement``."""
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
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p90": 0.0,
        }
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    p90_idx = max(0, min(n - 1, int(round(0.9 * (n - 1)))))
    return {
        "mean": mean,
        "median": s[n // 2],
        "stdev": variance**0.5,
        "min": s[0],
        "max": s[-1],
        "p90": s[p90_idx],
    }


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


def _per_rank_tokens(workload: WorkloadSpec, world_size: int) -> List[int]:
    """Materialize the ``per_rank_num_tokens`` list for a workload + world size."""
    if workload.token_unit == "per_rank":
        # Same local count on every rank, no distribution.
        return [int(workload.num_tokens)] * world_size
    # Global: split per ``rank_hotspot_ratio``.
    ratio = float(workload.rank_hotspot_ratio) if workload.rank_distribution == "rank0_hot" else 0.0
    return _distribute_tokens(int(workload.num_tokens), world_size, ratio)


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


_PARALLEL_MODE_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "DEP": {"moe_ep_size": "world", "moe_tp_size": 1, "enable_attention_dp": True},
    "TEP": {"moe_ep_size": "world", "moe_tp_size": 1, "enable_attention_dp": False},
    "DTP": {"moe_ep_size": 1, "moe_tp_size": "world", "enable_attention_dp": True},
    "TTP": {"moe_ep_size": 1, "moe_tp_size": "world", "enable_attention_dp": False},
}


def _resolve_mapping_layout(config: ConfigSpec, world_size: int) -> Tuple[int, int, bool]:
    """Resolve ``(moe_ep_size, moe_tp_size, enable_attention_dp)`` for a ConfigSpec."""
    if config.parallel_mode == "CUSTOM":
        if config.moe_ep_size is None or config.moe_tp_size is None:
            raise ValueError("parallel_mode=CUSTOM requires explicit moe_ep_size and moe_tp_size")
        moe_ep = int(config.moe_ep_size)
        moe_tp = int(config.moe_tp_size)
        enable_dp = (
            bool(config.enable_attention_dp) if config.enable_attention_dp is not None else False
        )
    else:
        layout = _PARALLEL_MODE_LAYOUTS.get(config.parallel_mode)
        if layout is None:
            raise ValueError(f"Unknown parallel_mode={config.parallel_mode!r}")
        moe_ep = world_size if layout["moe_ep_size"] == "world" else int(layout["moe_ep_size"])
        moe_tp = world_size if layout["moe_tp_size"] == "world" else int(layout["moe_tp_size"])
        enable_dp = bool(layout["enable_attention_dp"])
    if moe_ep * moe_tp != world_size:
        raise ValueError(
            f"moe_ep_size * moe_tp_size = {moe_ep * moe_tp} must equal world_size={world_size}"
        )
    return moe_ep, moe_tp, enable_dp


def _build_mapping_from_config(config: ConfigSpec, world_size: int) -> Mapping:
    """Build ``Mapping`` from a ``ConfigSpec`` + world size; sets ``rank=mpi_rank()``."""
    moe_ep, moe_tp, enable_dp = _resolve_mapping_layout(config, world_size)
    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=moe_ep,
        moe_tp_size=moe_tp,
        enable_attention_dp=enable_dp,
    )
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


def _resolve_eplb_internals(
    config: ConfigSpec, model: ModelSpec, mapping: Mapping
) -> Tuple[bool, int, int]:
    """Map ``eplb_mode`` to ``(enable_eplb, num_slots, layer_updates_per_iter)``.

    ``static`` enables the load balancer with ``layer_updates_per_iter=0`` and
    a friendly slot default; ``dynamic`` enables runtime updates with a larger
    slot capacity. An explicit ``num_slots`` / ``layer_updates_per_iter`` on
    the ConfigSpec overrides the friendly defaults.
    """
    if config.eplb_mode == "off":
        return False, -1, -1

    ep_size = max(int(mapping.moe_ep_size), 1)
    num_experts = int(model.num_experts)

    def _round_up(value: int, multiple: int) -> int:
        if multiple <= 0:
            return value
        return ((value + multiple - 1) // multiple) * multiple

    if config.eplb_mode == "static":
        default_slots = _round_up(num_experts + ep_size, ep_size)
        layer_updates = 0
    elif config.eplb_mode == "dynamic":
        default_slots = _round_up(num_experts * 2, ep_size)
        layer_updates = 1
    else:
        raise ValueError(f"Unknown eplb_mode={config.eplb_mode!r}")

    num_slots = int(config.num_slots) if config.num_slots is not None else default_slots
    if config.layer_updates_per_iter is not None:
        layer_updates = int(config.layer_updates_per_iter)
    return True, num_slots, layer_updates


def _build_model_config(
    *,
    model: ModelSpec,
    config: ConfigSpec,
    mapping: Mapping,
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
    pretrained_config = _build_pretrained_config(
        model.num_experts, model.hidden_size, model.intermediate_size, dtype
    )

    quant_algo = model.quant_algo_enum
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


def _scheduler_kind_name(moe) -> Optional[str]:
    """Return ``"EXTERNAL_COMM"`` / ``"FUSED_COMM"`` for the underlying backend."""
    backend = getattr(moe, "backend", None) or moe
    kind = getattr(backend, "scheduler_kind", None)
    if isinstance(kind, MoESchedulerKind):
        return kind.name
    return None


def _comm_method_name(moe) -> str:
    """Return the actual communication strategy class name, or ``"NONE"``."""
    if _scheduler_kind_name(moe) == "FUSED_COMM":
        return "NONE"
    comm = getattr(moe, "comm", None)
    if comm is None:
        return "NONE"
    return type(comm).__name__


def _calculate_num_chunks_safe(moe, all_rank_num_tokens: List[int]) -> Optional[int]:
    """Best-effort lookup of ``num_chunks`` for the case we are about to time."""
    scheduler = getattr(moe, "scheduler", None)
    if scheduler is None:
        return None
    fn = getattr(scheduler, "calculate_num_chunks", None)
    if fn is None:
        return None
    try:
        return int(fn(all_rank_num_tokens))
    except Exception:
        return None


def _build_moe_module(
    *,
    model: ModelSpec,
    config: ConfigSpec,
    mapping: Mapping,
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
    """Build a fresh ``ConfigurableMoE`` for one ``(backend, num_tokens)`` case.

    Returns ``(moe_module, moe_load_balancer or None, initial_expert_ids or None,
    routing_logits_dtype)``.
    """
    if enable_perfect_router:
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    else:
        os.environ.pop("ENABLE_PERFECT_ROUTER", None)

    mc = model.to_moe_model_config()
    swiglu_gptoss_style = model.swiglu_gptoss_style

    routing_method = _create_routing_method(
        model.routing_method_cls,
        top_k=mc.top_k,
        num_experts=mc.num_experts,
        bias_dtype=dtype,
        profile_model_config=mc,
    )

    model_config = _build_model_config(
        model=model,
        config=config,
        mapping=mapping,
        moe_backend=moe_backend,
        use_cuda_graph=use_cuda_graph,
        max_num_tokens=max_num_tokens,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
        enable_eplb=enable_eplb,
        num_slots=num_slots,
        layer_updates_per_iter=layer_updates_per_iter,
        dtype=dtype,
    )

    _ensure_dist_for_megamoe(moe_backend, mapping.rank, mapping.world_size)

    probe_x = torch.randn(
        (max(1, mc.hidden_size // 32), mc.hidden_size), dtype=dtype, device=device
    )
    backend_type = MoeBackendType(moe_backend.upper())
    quant_algo = model.quant_algo_enum
    quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
        quant_algo, probe_x, backend_type
    )
    quant_kwargs.pop("ref_cls", None)

    num_local_experts = mc.num_experts // max(mapping.moe_ep_size, 1)
    quantize_util = quantize_util_cls(
        num_experts=mc.num_experts,
        dtype=dtype,
        intermediate_size=mc.intermediate_size,
        hidden_size=mc.hidden_size,
        quant_config=quant_config,
        bias=swiglu_gptoss_style,
        swiglu_gptoss_style=swiglu_gptoss_style,
        swiglu_alpha=model.swiglu_alpha if swiglu_gptoss_style else None,
        swiglu_beta=model.swiglu_beta if swiglu_gptoss_style else None,
        swiglu_limit=model.swiglu_limit if swiglu_gptoss_style else None,
        num_local_experts=num_local_experts,
    )

    weight_loading_mode = getattr(
        quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
    )

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
    expert_hotspot_ratio: float,
    force_balance_patch: bool,
):
    """Patch routing/run_moe for synthetic per-expert hot-spot imbalance.

    A pass-through context manager when both ``expert_hotspot_ratio`` and
    ``force_balance_patch`` are unset. When active, follows the runner.py
    pattern and restores all overridden methods on exit.
    """
    if expert_hotspot_ratio == 0.0 and not force_balance_patch:
        yield
        return

    if expert_hotspot_ratio == 0.0:
        balance_method = BalanceMethod.Balanced
        balance_ratio = 1.0
    else:
        balance_method = BalanceMethod.ImbalancedExperts
        # Inverse semantics: ``balance_ratio=0`` is fully imbalanced.
        balance_ratio = 1.0 - expert_hotspot_ratio

    dp_size = mapping.dp_size
    dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
    ep_size = mapping.moe_ep_size

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
    """Categorize replay kernels into moe_forward/other windows via CUPTI."""
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
    for evt in starts + ends:
        evt.record()
    torch.cuda.synchronize()

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
# Per-rank gather + scoring
# ---------------------------------------------------------------------------


def _gather_per_iteration_times(times_us: List[float]) -> List[List[float]]:
    """All-gather raw per-iteration latencies; returns ``[ [rank0_iters], ... ]``."""
    return mpi_allgather(times_us)


def _slowest_rank_mean_score(per_rank_iters: List[List[float]]) -> float:
    """Compute ``mean_i(max_r(latency_us[rank=r][iteration=i]))``.

    Falls back gracefully when ranks reported different iteration counts; the
    common length is used and trailing entries are ignored.
    """
    if not per_rank_iters:
        return 0.0
    lengths = [len(r) for r in per_rank_iters if r]
    if not lengths:
        return 0.0
    n = min(lengths)
    if n == 0:
        return 0.0
    iter_max: List[float] = []
    for i in range(n):
        per_iter_vals = [r[i] for r in per_rank_iters if i < len(r)]
        iter_max.append(max(per_iter_vals))
    return sum(iter_max) / len(iter_max)


def _build_latency_block(per_rank_iters: List[List[float]]) -> Dict[str, Any]:
    score = _slowest_rank_mean_score(per_rank_iters)
    return {
        "score": float(score),
        "score_type": "slowest_rank_mean",
        "per_rank": {f"rank{i}": _compute_stats(times) for i, times in enumerate(per_rank_iters)},
    }


def _gather_kernel_breakdown(
    detailed_stats: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """All-gather per-kernel timings and produce per-rank summary stats."""
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

            per_rank = {f"rank{i}": _compute_stats(times) for i, times in enumerate(per_rank_times)}
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
# Bottleneck classification (best-effort, phase-data dependent)
# ---------------------------------------------------------------------------


def _classify_bottleneck(
    phase_times_us_agg: Dict[str, Dict[str, float]],
    kernel_breakdown: Dict[str, Any],
    forward_score_us: float,
) -> Optional[str]:
    """Return a coarse bottleneck label.

    The classification is intentionally minimal: with no scheduler-side phase
    markers (Phase 5 of the design) we can only inspect kernel breakdown. When
    we can identify dispatch/combine/GEMM heuristically by name we use that;
    otherwise we return ``None`` so the dashboard can show ``unknown``.
    """
    if phase_times_us_agg:
        comm_us = sum(
            v.get("score", 0.0)
            for k, v in phase_times_us_agg.items()
            if k in ("dispatch", "combine", "all_reduce_or_reduce_results")
        )
        gemm_us = phase_times_us_agg.get("backend_run_moe", {}).get(
            "score", 0.0
        ) or phase_times_us_agg.get("fused_comm_backend_run_moe", {}).get("score", 0.0)
        routing_us = phase_times_us_agg.get("routing", {}).get("score", 0.0)
        eplb_us = sum(
            v.get("score", 0.0) for k, v in phase_times_us_agg.items() if k.startswith("eplb_")
        )
        total = comm_us + gemm_us + routing_us + eplb_us
        if total <= 0:
            return None
        if comm_us / total > 0.5:
            return "communication_bound"
        if gemm_us / total > 0.5:
            return "compute_bound"
        if routing_us / total > 0.4:
            return "routing_bound"
        if eplb_us / total > 0.4:
            return "eplb_bound"
        return "unknown"

    # No phase markers: inspect kernel breakdown for a rough hint.
    moe_kernels = kernel_breakdown.get("moe_forward_kernels", [])
    if not moe_kernels:
        return None
    total_count = sum(k.get("count", 0) for k in moe_kernels)
    # Average kernel duration in us per launch on rank0.
    rank0_total = 0.0
    rank0_count = 0
    for k in moe_kernels:
        rank0_stats = k.get("per_rank", {}).get("rank0", {})
        mean_us = float(rank0_stats.get("mean", 0.0))
        count = int(k.get("count", 0))
        rank0_total += mean_us * count
        rank0_count += count
    avg_us = (rank0_total / rank0_count) if rank0_count else 0.0
    if total_count > 50 and avg_us > 0.0 and avg_us < 5.0 and forward_score_us > 0:
        return "launch_overhead_bound"
    return None


# ---------------------------------------------------------------------------
# Search expansion + candidate validation
# ---------------------------------------------------------------------------


def _expand_axis(values: Iterable[Any], default: Any) -> Tuple[Any, ...]:
    out = tuple(values)
    return out if out else (default,)


def expand_search(
    base_config: ConfigSpec,
    search: SearchSpec,
    world_size: int,
) -> List[ConfigSpec]:
    """Cartesian-product candidate generation, then explicit pruning.

    ``base_config`` carries the *non-search* fields (cuda_graph default,
    combine precision default, eplb num_slots overrides). Search axes
    explicitly listed on ``search`` override the base values.
    """
    backends = _expand_axis(search.backends, base_config.backend)
    parallel_modes = _expand_axis(search.parallel_modes, base_config.parallel_mode)
    comm_methods = _expand_axis(search.comm_methods, base_config.comm_method)
    eplb_modes = _expand_axis(search.eplb_modes, base_config.eplb_mode)
    cuda_graph_options = _expand_axis(search.cuda_graph_options, base_config.cuda_graph)
    combine_options = _expand_axis(
        search.combine_precision_options, base_config.use_low_precision_moe_combine
    )

    candidates: List[ConfigSpec] = []
    for backend, pmode, comm, eplb, cgraph, combine in itertools.product(
        backends, parallel_modes, comm_methods, eplb_modes, cuda_graph_options, combine_options
    ):
        candidate = replace(
            base_config,
            backend=str(backend).upper(),
            parallel_mode=str(pmode).upper(),
            comm_method=str(comm).upper(),
            eplb_mode=str(eplb).lower(),
            cuda_graph=bool(cgraph),
            use_low_precision_moe_combine=bool(combine),
        )
        candidates.append(candidate)
    return candidates


def is_candidate_valid(
    config: ConfigSpec,
    model: ModelSpec,
    world_size: int,
    act_dtype: torch.dtype,
) -> Tuple[bool, Optional[str]]:
    """Return ``(ok, reason)`` based on backend / mapping / EPLB / comm gates."""
    # Backend can_implement gate.
    ok, reason = _check_backend_can_implement(
        config.backend, model.quant_algo_enum, act_dtype, model.swiglu_gptoss_style
    )
    if not ok:
        return False, reason

    # Mapping layout gate.
    try:
        moe_ep, moe_tp, enable_dp = _resolve_mapping_layout(config, world_size)
    except ValueError as exc:
        return False, str(exc)

    # EPLB / CUDA Graph compatibility.
    if config.eplb_mode == "dynamic" and config.cuda_graph:
        return False, "dynamic EPLB is incompatible with CUDA Graph timing"

    # Forced communication on non-DP / MoE-TP paths.
    forced = config.comm_method.upper()
    if forced not in ("AUTO", "NONE"):
        if not enable_dp:
            return False, f"comm_method={forced} requires enable_attention_dp=True"
        if moe_tp != 1 and forced != "ALLGATHER":
            return False, f"comm_method={forced} requires moe_tp_size=1 (got {moe_tp})"
        if world_size == 1:
            return False, f"comm_method={forced} has no effect at world_size=1"

    return True, None


def expand_and_prune(
    base_config: ConfigSpec,
    search: SearchSpec,
    model: ModelSpec,
    world_size: int,
    act_dtype: torch.dtype,
    max_configs: Optional[int] = None,
) -> Tuple[List[ConfigSpec], Dict[ConfigSpec, str]]:
    """Expand search and split into ``(valid_candidates, skip_reasons_for_invalid)``.

    ``max_configs`` truncates the *valid* list after pruning; the skipped /
    invalid candidates are reported in full so dashboard rows do not silently
    disappear.
    """
    raw = expand_search(base_config, search, world_size)
    valid: List[ConfigSpec] = []
    skipped: Dict[ConfigSpec, str] = {}
    for cand in raw:
        ok, reason = is_candidate_valid(cand, model, world_size, act_dtype)
        if ok:
            valid.append(cand)
        else:
            skipped[cand] = reason or "invalid"
    if max_configs is not None and max_configs >= 0 and len(valid) > max_configs:
        truncated = valid[max_configs:]
        valid = valid[:max_configs]
        for cand in truncated:
            skipped[cand] = f"truncated by --max_configs={max_configs}"
    return valid, skipped


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------


def _force_comm_env(comm_method: str, prev: Optional[str]) -> None:
    """Push or restore ``TRTLLM_FORCE_COMM_METHOD`` for one case.

    The design notes that env-var forcing is the only available per-case knob
    today; subprocess isolation is the recommended long-term path.
    """
    upper = comm_method.upper()
    if upper in _FORCED_COMM_ENV_VALUES:
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = _FORCED_COMM_ENV_VALUES[upper]
    else:
        if prev is None:
            os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
        else:
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = prev


def _gather_status_per_rank(local_status: str) -> Dict[str, str]:
    payload = mpi_allgather(local_status)
    return {f"rank{i}": s for i, s in enumerate(payload)}


def _run_one_candidate(
    *,
    model: ModelSpec,
    workload: WorkloadSpec,
    config: ConfigSpec,
    world_size: int,
    rank: int,
    device: torch.device,
    act_dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    warmup: int,
    iters: int,
    fast_autotune: bool,
    analysis: Tuple[str, ...],
    cupti_ctx: Optional[Any],
    force_balance_patch: bool,
) -> RunResult:
    """Build, autotune, and time one ``ConfigSpec`` candidate.

    Always returns a ``RunResult``; failures are encoded in the ``status`` /
    ``skip_reason`` fields so the caller can write a row even for the failed
    case.
    """
    result = RunResult(model=model, workload=workload, config=config)
    per_rank = _per_rank_tokens(workload, world_size)
    result.per_rank_num_tokens = list(per_rank)
    local_num_tokens = per_rank[rank]
    all_rank_num_tokens = list(per_rank)

    instrumentation: Dict[str, Any] = {
        "level": ",".join(sorted(analysis)) if analysis else "summary",
        "cuda_graph": bool(config.cuda_graph),
        "cupti_available": bool(cupti_ctx is not None and cupti_ctx[3]),
        "phase_timing_available": False,
        "kernel_breakdown_available": "kernels" in analysis,
    }
    result.instrumentation = instrumentation

    # Build mapping for this candidate.
    try:
        mapping = _build_mapping_from_config(config, world_size)
    except ValueError as exc:
        result.status = "skipped"
        result.skip_reason = str(exc)
        result.status_per_rank = _gather_status_per_rank("skipped")
        return result

    result.moe_ep_size = int(mapping.moe_ep_size)
    result.moe_tp_size = int(mapping.moe_tp_size)
    result.enable_attention_dp = bool(mapping.enable_attention_dp)

    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()

    # Resolve EPLB internals.
    try:
        enable_eplb, num_slots, layer_updates = _resolve_eplb_internals(config, model, mapping)
    except ValueError as exc:
        result.status = "skipped"
        result.skip_reason = str(exc)
        result.status_per_rank = _gather_status_per_rank("skipped")
        return result

    # Force comm method via env var (per-case).
    prev_force_comm = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
    _force_comm_env(config.comm_method, prev_force_comm)

    moe = moe_load_balancer = None
    local_status = "success"
    try:
        try:
            moe, moe_load_balancer, _initial_expert_ids, _ = _build_moe_module(
                model=model,
                config=config,
                mapping=mapping,
                moe_backend=config.backend,
                use_cuda_graph=bool(config.cuda_graph),
                max_num_tokens=max(int(local_num_tokens), 1),
                use_low_precision_moe_combine=bool(config.use_low_precision_moe_combine),
                enable_eplb=enable_eplb,
                num_slots=int(num_slots),
                layer_updates_per_iter=int(layer_updates),
                enable_perfect_router=(
                    workload.expert_distribution == "balanced_patch"
                    and workload.expert_hotspot_ratio == 0.0
                    and not force_balance_patch
                ),
                dtype=act_dtype,
                routing_logits_dtype=routing_logits_dtype,
                device=device,
            )
        except Exception as exc:
            reason = f"build error: {type(exc).__name__}: {exc}"
            _maybe_print_rank0(f"[bench_moe] build failed: {reason}")
            result.status = "failed"
            result.skip_reason = reason
            result.status_per_rank = _gather_status_per_rank("failed")
            return result

        result.actual_backend = _backend_name_from_module(moe)
        result.scheduler_kind = _scheduler_kind_name(moe)
        result.actual_comm_method = _comm_method_name(moe)
        result.num_chunks = _calculate_num_chunks_safe(moe, all_rank_num_tokens)

        if result.actual_backend != config.backend.upper():
            reason = f"requested backend {config.backend!r} fell back to {result.actual_backend!r}"
            _maybe_print_rank0(f"[bench_moe] {reason}")
            result.status = "skipped"
            result.skip_reason = reason
            result.status_per_rank = _gather_status_per_rank("skipped")
            return result

        x, router_logits = _make_inputs(
            local_num_tokens,
            model.hidden_size,
            model.num_experts,
            act_dtype,
            routing_logits_dtype,
            device,
        )

        with _maybe_install_balance_patch(
            moe,
            mapping,
            model.num_experts,
            model.top_k,
            float(workload.expert_hotspot_ratio)
            if workload.expert_distribution == "hotspot"
            else 0.0,
            bool(force_balance_patch),
        ):
            try:
                _run_autotune(
                    moe,
                    x,
                    router_logits,
                    all_rank_num_tokens,
                    moe_load_balancer,
                    bool(fast_autotune),
                )
            except Exception as exc:
                _maybe_print_rank0(f"[bench_moe] autotune skipped: {type(exc).__name__}: {exc}")

            try:
                if config.cuda_graph:
                    fwd_times_us, detailed_stats = _time_moe_forward_cuda_graph(
                        moe,
                        x,
                        router_logits,
                        all_rank_num_tokens,
                        warmup=int(warmup),
                        iters=int(iters),
                        cupti_ctx=cupti_ctx,
                    )
                else:
                    fwd_times_us, detailed_stats = _time_moe_forward_eager(
                        moe,
                        x,
                        router_logits,
                        all_rank_num_tokens,
                        warmup=int(warmup),
                        iters=int(iters),
                        eplb=moe_load_balancer,
                    )
            except Exception as exc:
                reason = f"timed phase error: {type(exc).__name__}: {exc}"
                _maybe_print_rank0(f"[bench_moe] {reason}\n{traceback.format_exc()}")
                local_status = "failed"
                result.status = "failed"
                result.skip_reason = reason
                result.status_per_rank = _gather_status_per_rank(local_status)
                return result

        # Refresh actual_comm_method after the first forward (the factory may
        # swap moe.comm to AllGatherReduceScatter inside dispatch).
        result.actual_comm_method = _comm_method_name(moe)

        per_rank_iters = _gather_per_iteration_times(fwd_times_us)
        result.latency_us = _build_latency_block(per_rank_iters)

        if "kernels" in analysis:
            kb_payload = _gather_kernel_breakdown(detailed_stats)
            result.kernel_breakdown = kb_payload
        else:
            result.kernel_breakdown = {"moe_forward_kernels": [], "other_kernels": []}

        # Phase markers live in moe_scheduler.py (Phase 5 of the design); not
        # implemented yet, so emit empty agg/per_rank with a stable shape.
        result.phase_times_us = {"agg": {}, "per_rank": {}}
        result.overlap = {"overlap_us": None, "overlap_ratio": None}
        result.bottleneck = _classify_bottleneck(
            result.phase_times_us["agg"], result.kernel_breakdown, result.latency_us["score"]
        )
        result.status_per_rank = _gather_status_per_rank(local_status)
        return result
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
        # Restore TRTLLM_FORCE_COMM_METHOD.
        if prev_force_comm is None:
            os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
        else:
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = prev_force_comm


# ---------------------------------------------------------------------------
# Output schema v2 serialization
# ---------------------------------------------------------------------------


def _runresult_to_row(result: RunResult) -> Dict[str, Any]:
    """Convert ``RunResult`` to the v2 row schema."""
    return {
        "workload": result.workload.to_dict(per_rank_num_tokens=result.per_rank_num_tokens),
        "requested_config": result.config.to_dict(),
        "actual_config": {
            "backend": result.actual_backend,
            "comm_method": result.actual_comm_method,
            "comm_fallback_reason": result.actual_comm_fallback_reason,
            "scheduler_kind": result.scheduler_kind,
            "moe_ep_size": result.moe_ep_size,
            "moe_tp_size": result.moe_tp_size,
            "enable_attention_dp": result.enable_attention_dp,
            "num_chunks": result.num_chunks,
        },
        "status": result.status,
        "skip_reason": result.skip_reason,
        "status_per_rank": result.status_per_rank,
        "instrumentation": result.instrumentation,
        "latency_us": result.latency_us
        or {
            "score": None,
            "score_type": "slowest_rank_mean",
            "per_rank": {},
        },
        "phase_times_us": result.phase_times_us or {"agg": {}, "per_rank": {}},
        "overlap": result.overlap or {"overlap_us": None, "overlap_ratio": None},
        "bottleneck": result.bottleneck,
        "kernel_breakdown": result.kernel_breakdown
        or {
            "moe_forward_kernels": [],
            "other_kernels": [],
        },
    }


def _build_rankings(results: List[RunResult]) -> List[Dict[str, Any]]:
    """Group results by ``(num_tokens, parallel_mode)`` and rank by score."""
    grouped: Dict[Tuple[int, str], List[RunResult]] = {}
    for r in results:
        key = (int(r.workload.num_tokens), r.config.parallel_mode)
        grouped.setdefault(key, []).append(r)

    rankings: List[Dict[str, Any]] = []
    for (num_tokens, parallel_mode), items in sorted(grouped.items()):
        ranking_entries: List[Dict[str, Any]] = []
        for r in items:
            score = r.latency_us.get("score") if r.latency_us else None
            ranking_entries.append(
                {
                    "backend": r.actual_backend or r.config.backend,
                    "requested_backend": r.config.backend,
                    "comm_method": r.actual_comm_method,
                    "eplb_mode": r.config.eplb_mode,
                    "cuda_graph": r.config.cuda_graph,
                    "use_low_precision_moe_combine": r.config.use_low_precision_moe_combine,
                    "score_us": float(score) if isinstance(score, (int, float)) else None,
                    "status": r.status,
                    "skip_reason": r.skip_reason,
                }
            )
        ranking_entries.sort(
            key=lambda e: (
                e["score_us"] is None,
                e["score_us"] if e["score_us"] is not None else 0.0,
            )
        )
        best = next(
            (e for e in ranking_entries if e["score_us"] is not None and e["status"] == "success"),
            None,
        )
        rankings.append(
            {
                "num_tokens": int(num_tokens),
                "parallel_mode": parallel_mode,
                "best": best,
                "ranking": ranking_entries,
            }
        )
    return rankings


def _trtllm_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _build_environment_block(world_size: int, cuda_graph_default: bool) -> Dict[str, Any]:
    device_name = None
    sm = None
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            device_name = props.name
            sm = int(getattr(props, "major", 0) * 10 + getattr(props, "minor", 0))
        except Exception:
            pass
    cuda_version = None
    try:
        cuda_version = torch.version.cuda
    except Exception:
        pass
    driver_version = None
    try:
        from torch.cuda import _get_driver_version  # type: ignore

        driver_version = _get_driver_version()
    except Exception:
        pass

    try:
        host = socket.gethostname()
    except Exception:
        host = None
    try:
        user = getpass.getuser()
    except Exception:
        user = None

    return {
        "world_size": int(world_size),
        "world_size_per_node": int(min(world_size, max(torch.cuda.device_count(), 1))),
        "hostname": host,
        "username": user,
        "device_name": device_name,
        "sm": sm,
        "cuda_version": str(cuda_version) if cuda_version else None,
        "driver_version": str(driver_version) if driver_version else None,
        "torch_version": str(torch.__version__),
        "trtllm_commit": _trtllm_commit(),
        "platform": platform.platform(),
        "nvlink_topology": "unknown",
        "memory_type": "unknown",
        "clock_locked": False,
        "cuda_graph_default": bool(cuda_graph_default),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEPRECATED_HELP = " (deprecated; see new flag in --help)"


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--search",
        type=lambda s: str(s).lower(),
        default="none",
        choices=("none", "backend", "comm", "parallel", "eplb", "full"),
        help="Search preset; expands the corresponding axis of the candidate space.",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Truncate the valid candidate list to at most this many entries.",
    )
    parser.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Stop launching new candidates once this wall-clock budget elapses.",
    )
    parser.add_argument(
        "--per_case_subprocess",
        action="store_true",
        help=(
            "Recommended for large dashboard sweeps: runs each candidate in a "
            "fresh worker process. Currently a no-op stub that warns; falls "
            "back to in-process execution."
        ),
    )


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

    # ---- Model / shape ----
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(BUILT_IN_MODELS.keys()),
        help="Named model spec (overridable via individual flags below).",
    )
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
        default="AUTO",
        choices=sorted(_ROUTING_METHODS) + ["AUTO"],
        help=(
            "Routing method. Defaults to AUTO: built-in models use the spec "
            "default; custom shapes must specify an explicit method."
        ),
    )

    # ---- Workload ----
    parser.add_argument(
        "--num_tokens",
        type=int,
        nargs="+",
        required=False,
        help=(
            "Token counts to sweep. Interpreted as global tokens by default; "
            "see --token_unit per_rank for the alternate semantics."
        ),
    )
    parser.add_argument(
        "--token_unit",
        type=lambda s: str(s).lower(),
        default="global",
        choices=("global", "per_rank"),
        help="Whether each --num_tokens entry is global or per-rank tokens.",
    )
    parser.add_argument(
        "--rank_distribution",
        type=lambda s: str(s).lower(),
        default="balanced",
        choices=("balanced", "rank0_hot"),
        help="How global tokens are split across ranks.",
    )
    parser.add_argument(
        "--rank_hotspot_ratio",
        type=float,
        default=0.0,
        help="In [0, 1]. Used when --rank_distribution=rank0_hot.",
    )
    parser.add_argument(
        "--tokens_per_rank_imbalance_ratio",
        type=float,
        default=None,
        help="Deprecated alias for --rank_hotspot_ratio." + _DEPRECATED_HELP,
    )
    parser.add_argument(
        "--expert_distribution",
        type=lambda s: str(s).lower(),
        default="balanced_patch",
        choices=("balanced_patch", "hotspot"),
        help="Synthetic expert assignment used for routing logits.",
    )
    parser.add_argument(
        "--expert_hotspot_ratio",
        type=float,
        default=0.0,
        help="In [0, 1]. 0=balanced; >0 concentrates tokens on hot experts.",
    )
    parser.add_argument(
        "--experts_hot_imbalane_ratio",
        type=float,
        default=None,
        help="Deprecated alias for --expert_hotspot_ratio." + _DEPRECATED_HELP,
    )
    parser.add_argument(
        "--force_balance_patch",
        action="store_true",
        help="Force the balanced routing patch even when expert_hotspot_ratio == 0.",
    )

    # ---- Parallel mode ----
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="DEP",
        choices=("DEP", "TEP", "DTP", "TTP", "CUSTOM"),
        help="Parallel mode (combined with world_size).",
    )
    parser.add_argument("--moe_ep_size", type=int, default=None)
    parser.add_argument("--moe_tp_size", type=int, default=None)
    parser.add_argument("--enable_attention_dp", action="store_true")

    # ---- EPLB ----
    parser.add_argument(
        "--eplb_mode",
        type=lambda s: str(s).lower(),
        default="off",
        choices=("off", "static", "dynamic"),
        help="EPLB mode (replaces --enable_eplb / --layer_updates_per_iter).",
    )
    parser.add_argument("--num_slots", type=int, default=None)
    parser.add_argument("--layer_updates_per_iter", type=int, default=None)
    parser.add_argument(
        "--enable_eplb",
        action="store_true",
        default=False,
        help="Deprecated; use --eplb_mode static|dynamic." + _DEPRECATED_HELP,
    )

    # ---- Backend ----
    parser.add_argument(
        "--backend",
        type=lambda s: str(s).upper(),
        default="TRTLLM",
        choices=_ALL_BACKENDS + ["BEST", "ALL"],
        help=(
            "Backend to bench. ``BEST`` is a deprecated shortcut for "
            "``--search backend --backend ALL``; ``ALL`` expands to every "
            "ConfigurableMoE-eligible backend when --search backend is set."
        ),
    )

    # ---- Communication ----
    parser.add_argument(
        "--comm_method",
        type=lambda s: str(s).upper(),
        default="AUTO",
        choices=_COMM_METHODS,
        help="Per-case forced communication method (replaces --force_comm_method).",
    )
    parser.add_argument(
        "--force_comm_method",
        type=lambda s: str(s).upper(),
        default=None,
        choices=tuple(_FORCED_COMM_ENV_VALUES.keys()),
        help="Deprecated alias for --comm_method." + _DEPRECATED_HELP,
    )

    # ---- Timing ----
    parser.add_argument(
        "--no_cuda_graph",
        dest="cuda_graph",
        action="store_false",
        default=True,
        help=("Disable CUDA-Graph capture and use eager timing. Required for dynamic EPLB."),
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

    # ---- Analysis ----
    parser.add_argument(
        "--analysis",
        type=str,
        default="kernels",
        help=(
            "Comma-separated analysis dimensions to enable: 'summary' | "
            "'phases' | 'kernels' | 'phases,kernels'. Phase markers require "
            "moe_scheduler.py instrumentation (Phase 5 of the design) and "
            "currently report phase_timing_available=false."
        ),
    )

    # ---- Search / sweep ----
    _add_search_arguments(parser)

    # ---- Output / pipeline ----
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="JSON config file describing model / workload / search / output (overrides CLI).",
    )
    parser.add_argument("--output_file", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Spec resolution from CLI args / config file
# ---------------------------------------------------------------------------


def _emit_deprecation(field_name: str, replacement: str) -> None:
    warnings.warn(
        f"--{field_name} is deprecated; use --{replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _resolve_model_from_args(args: argparse.Namespace) -> ModelSpec:
    base = BUILT_IN_MODELS.get(args.model) if args.model is not None else None

    routing = args.routing_method
    if base is None:
        # Custom shape requires explicit fields.
        missing = [
            f
            for f in ("num_experts", "top_k", "hidden_size", "intermediate_size")
            if getattr(args, f) is None
        ]
        if missing:
            raise ValueError("No --model selected; you must also pass: " + ", ".join(missing))
        if routing == "AUTO":
            raise ValueError(
                "Custom shapes (no --model) require an explicit --routing_method; "
                "AUTO has no safe default."
            )
        quant_name = args.quant.name if args.quant is not None else None
        return ModelSpec(
            name="custom",
            num_experts=int(args.num_experts),
            top_k=int(args.top_k),
            hidden_size=int(args.hidden_size),
            intermediate_size=int(args.intermediate_size),
            quant_algo=quant_name,
            routing_method=routing,
            n_group=args.n_group,
            topk_group=args.topk_group,
        )

    # Built-in model with optional per-field overrides.
    if routing == "AUTO":
        routing = base.routing_method

    quant_name: Optional[str]
    if args.quant is not None:
        quant_name = args.quant.name
    else:
        quant_name = base.quant_algo
    return ModelSpec(
        name=base.name,
        num_experts=int(args.num_experts) if args.num_experts is not None else base.num_experts,
        top_k=int(args.top_k) if args.top_k is not None else base.top_k,
        hidden_size=int(args.hidden_size) if args.hidden_size is not None else base.hidden_size,
        intermediate_size=int(args.intermediate_size)
        if args.intermediate_size is not None
        else base.intermediate_size,
        quant_algo=quant_name,
        routing_method=routing,
        n_group=args.n_group if args.n_group is not None else base.n_group,
        topk_group=args.topk_group if args.topk_group is not None else base.topk_group,
        swiglu_alpha=base.swiglu_alpha,
        swiglu_beta=base.swiglu_beta,
        swiglu_limit=base.swiglu_limit,
    )


def _resolve_workloads_from_args(args: argparse.Namespace) -> List[WorkloadSpec]:
    if not args.num_tokens:
        raise ValueError("--num_tokens is required (or supply via --config_file)")
    rank_ratio = args.rank_hotspot_ratio
    if args.tokens_per_rank_imbalance_ratio is not None:
        _emit_deprecation("tokens_per_rank_imbalance_ratio", "rank_hotspot_ratio")
        rank_ratio = float(args.tokens_per_rank_imbalance_ratio)
    expert_ratio = args.expert_hotspot_ratio
    if args.experts_hot_imbalane_ratio is not None:
        _emit_deprecation("experts_hot_imbalane_ratio", "expert_hotspot_ratio")
        expert_ratio = float(args.experts_hot_imbalane_ratio)
    rank_distribution = args.rank_distribution
    if rank_ratio > 0.0 and rank_distribution == "balanced":
        rank_distribution = "rank0_hot"
    expert_distribution = args.expert_distribution
    if expert_ratio > 0.0 and expert_distribution == "balanced_patch":
        expert_distribution = "hotspot"

    if not (0.0 <= rank_ratio <= 1.0):
        raise ValueError("rank_hotspot_ratio must be in [0, 1]")
    if not (0.0 <= expert_ratio <= 1.0):
        raise ValueError("expert_hotspot_ratio must be in [0, 1]")
    if any(t < 0 for t in args.num_tokens):
        raise ValueError("--num_tokens entries must be >= 0")

    return [
        WorkloadSpec(
            num_tokens=int(t),
            token_unit=args.token_unit,
            rank_distribution=rank_distribution,
            rank_hotspot_ratio=float(rank_ratio),
            expert_distribution=expert_distribution,
            expert_hotspot_ratio=float(expert_ratio),
        )
        for t in args.num_tokens
    ]


def _resolve_base_config_from_args(args: argparse.Namespace) -> ConfigSpec:
    # Map deprecated --force_comm_method -> --comm_method.
    comm_method = args.comm_method
    if args.force_comm_method is not None:
        _emit_deprecation("force_comm_method", "comm_method")
        if comm_method == "AUTO":
            comm_method = args.force_comm_method

    # Map deprecated --enable_eplb -> --eplb_mode.
    eplb_mode = args.eplb_mode
    if args.enable_eplb and eplb_mode == "off":
        _emit_deprecation("enable_eplb", "eplb_mode")
        # ``layer_updates_per_iter`` legacy semantics: <=0 => static, >0 => dynamic.
        legacy_layer = args.layer_updates_per_iter if args.layer_updates_per_iter is not None else 0
        eplb_mode = "dynamic" if int(legacy_layer) > 0 else "static"

    # Backend handling: BEST is mapped later (after parallel_mode is known) by
    # the caller. Here we just translate BEST -> a placeholder backend so the
    # base ConfigSpec is concrete.
    backend = args.backend
    if backend in ("BEST", "ALL"):
        backend = MoeBackendType.CUTLASS.value  # placeholder; overwritten by search expansion

    # parallel_mode CUSTOM if explicit EP/TP overrides are present.
    parallel_mode = args.parallel_mode
    if (args.moe_ep_size is not None or args.moe_tp_size is not None) and parallel_mode in (
        "DEP",
        "TEP",
        "DTP",
        "TTP",
    ):
        # Treat explicit overrides as opting into CUSTOM so output metadata is honest.
        parallel_mode = "CUSTOM"

    if parallel_mode == "CUSTOM" and (args.moe_ep_size is None or args.moe_tp_size is None):
        raise ValueError("--parallel_mode=CUSTOM requires both --moe_ep_size and --moe_tp_size")

    return ConfigSpec(
        backend=backend,
        parallel_mode=parallel_mode,
        moe_ep_size=args.moe_ep_size,
        moe_tp_size=args.moe_tp_size,
        enable_attention_dp=bool(args.enable_attention_dp) if parallel_mode == "CUSTOM" else None,
        comm_method=comm_method,
        eplb_mode=eplb_mode,
        num_slots=args.num_slots,
        layer_updates_per_iter=args.layer_updates_per_iter,
        cuda_graph=bool(args.cuda_graph),
        use_low_precision_moe_combine=bool(args.use_low_precision_moe_combine),
        per_case_subprocess=bool(args.per_case_subprocess),
    )


def _resolve_search_from_args(args: argparse.Namespace, base_config: ConfigSpec) -> SearchSpec:
    # ``--backend BEST`` is a compatibility alias for ``--search backend --backend ALL``.
    mode = args.search
    backends_arg = args.backend
    if backends_arg == "BEST":
        _emit_deprecation("backend BEST", "search backend --backend ALL")
        mode = "backend"
        backends_arg = "ALL"

    backends: Tuple[str, ...] = ()
    parallel_modes: Tuple[str, ...] = ()
    comm_methods: Tuple[str, ...] = ()
    eplb_modes: Tuple[str, ...] = ()
    cuda_graph_options: Tuple[bool, ...] = ()
    combine_options: Tuple[bool, ...] = ()

    if mode == "none":
        return SearchSpec(mode="none")

    if mode in ("backend", "full"):
        if backends_arg == "ALL":
            backends = tuple(_ALL_BACKENDS)
        else:
            backends = (backends_arg,)
    if mode in ("parallel", "full"):
        parallel_modes = ("DEP", "TEP", "DTP", "TTP")
    if mode in ("comm", "full"):
        comm_methods = tuple(_FORCED_COMM_ENV_VALUES.keys()) + ("AUTO",)
    if mode in ("eplb", "full"):
        eplb_modes = ("off", "static", "dynamic")
    # ``cuda_graph`` and ``combine_precision`` axes are reserved for ``full``;
    # leave them empty by default so the base value is used.
    if mode == "full":
        cuda_graph_options = (True, False)

    return SearchSpec(
        mode=mode,
        backends=backends,
        parallel_modes=parallel_modes,
        comm_methods=comm_methods,
        eplb_modes=eplb_modes,
        cuda_graph_options=cuda_graph_options,
        combine_precision_options=combine_options,
    )


def _parse_analysis(value: str) -> Tuple[str, ...]:
    parts = [p.strip().lower() for p in str(value).split(",") if p.strip()]
    valid = {"summary", "phases", "kernels"}
    out: List[str] = []
    for p in parts:
        if p not in valid:
            raise ValueError(f"unknown --analysis dimension {p!r}; valid: {sorted(valid)}")
        if p == "summary":
            continue
        if p not in out:
            out.append(p)
    return tuple(out)


def _maybe_load_config_file(args: argparse.Namespace) -> argparse.Namespace:
    """Overlay ``--config_file`` JSON onto ``args``; the file wins on conflict."""
    if not args.config_file:
        return args
    with open(args.config_file) as f:
        cfg = json.load(f)
    if "model" in cfg:
        args.model = cfg["model"]
    workload_cfg = cfg.get("workload", {}) or {}
    if "num_tokens" in workload_cfg:
        args.num_tokens = list(workload_cfg["num_tokens"])
    if "token_unit" in workload_cfg:
        args.token_unit = workload_cfg["token_unit"]
    if "rank_distribution" in workload_cfg:
        args.rank_distribution = workload_cfg["rank_distribution"]
    if "rank_hotspot_ratio" in workload_cfg:
        args.rank_hotspot_ratio = float(workload_cfg["rank_hotspot_ratio"])
    if "expert_distribution" in workload_cfg:
        args.expert_distribution = workload_cfg["expert_distribution"]
    if "expert_hotspot_ratio" in workload_cfg:
        args.expert_hotspot_ratio = float(workload_cfg["expert_hotspot_ratio"])
    search_cfg = cfg.get("search", {}) or {}
    # When a search block is present we promote --search to an inferred mode.
    if search_cfg:
        # The simplest mapping: if search lists multiple backends, that's at least
        # ``backend`` mode; ``full`` if multiple axes specified.
        axis_count = sum(
            1
            for k in ("backend", "parallel_mode", "comm_method", "eplb_mode")
            if k in search_cfg and isinstance(search_cfg[k], list) and len(search_cfg[k]) > 1
        )
        if axis_count >= 2:
            args.search = "full"
        elif "backend" in search_cfg:
            args.search = "backend"
        elif "parallel_mode" in search_cfg:
            args.search = "parallel"
        elif "comm_method" in search_cfg:
            args.search = "comm"
        elif "eplb_mode" in search_cfg:
            args.search = "eplb"
        if "backend" in search_cfg:
            backends = search_cfg["backend"]
            if isinstance(backends, list) and len(backends) == 1:
                args.backend = str(backends[0]).upper()
            elif backends == "ALL":
                args.backend = "ALL"
        if "parallel_mode" in search_cfg and isinstance(search_cfg["parallel_mode"], list):
            if len(search_cfg["parallel_mode"]) == 1:
                args.parallel_mode = search_cfg["parallel_mode"][0]
        if "comm_method" in search_cfg and isinstance(search_cfg["comm_method"], list):
            if len(search_cfg["comm_method"]) == 1:
                args.comm_method = str(search_cfg["comm_method"][0]).upper()
        if "eplb_mode" in search_cfg and isinstance(search_cfg["eplb_mode"], list):
            if len(search_cfg["eplb_mode"]) == 1:
                args.eplb_mode = str(search_cfg["eplb_mode"][0]).lower()
    if "analysis" in cfg:
        analysis = cfg["analysis"]
        if isinstance(analysis, list):
            args.analysis = ",".join(analysis)
        else:
            args.analysis = str(analysis)
    if "per_case_subprocess" in cfg:
        args.per_case_subprocess = bool(cfg["per_case_subprocess"])
    if "max_configs" in cfg:
        args.max_configs = int(cfg["max_configs"])
    if "time_budget_minutes" in cfg:
        args.time_budget_minutes = float(cfg["time_budget_minutes"])
    if "output_file" in cfg and not args.output_file:
        args.output_file = cfg["output_file"]
    return args


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

    args = _maybe_load_config_file(args)

    model = _resolve_model_from_args(args)
    workloads = _resolve_workloads_from_args(args)
    base_config = _resolve_base_config_from_args(args)
    search = _resolve_search_from_args(args, base_config)
    analysis = _parse_analysis(args.analysis)

    if args.per_case_subprocess:
        _maybe_print_rank0(
            "[bench_moe] --per_case_subprocess is currently a no-op stub; "
            "running all candidates in-process."
        )

    act_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    routing_logits_dtype = (
        torch.float32 if model.routing_method_cls is DeepSeekV3MoeRoutingMethod else act_dtype
    )

    # Early header (rank 0) for stdout consumers.
    header = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "bench_moe",
        "launcher": launcher,
        "model": model.to_dict(),
        "search": search.to_dict(),
        "world_size": world_size,
        "analysis": list(analysis) or ["summary"],
        "workloads": [
            w.to_dict(per_rank_num_tokens=_per_rank_tokens(w, world_size)) for w in workloads
        ],
        "base_config": base_config.to_dict(),
    }
    if rank == 0:
        print(json.dumps(header, indent=2), flush=True)

    results: List[RunResult] = []
    skipped_global: List[Tuple[WorkloadSpec, ConfigSpec, str]] = []

    deadline = None
    if args.time_budget_minutes is not None and args.time_budget_minutes > 0:
        deadline = time.monotonic() + args.time_budget_minutes * 60.0

    for workload in workloads:
        candidates, skipped = expand_and_prune(
            base_config=base_config,
            search=search,
            model=model,
            world_size=world_size,
            act_dtype=act_dtype,
            max_configs=args.max_configs,
        )

        # Record every pruned candidate as a skipped result row so dashboards
        # see the full search space.
        for cand, reason in skipped.items():
            r = RunResult(model=model, workload=workload, config=cand)
            r.status = "skipped"
            r.skip_reason = reason
            r.per_rank_num_tokens = _per_rank_tokens(workload, world_size)
            r.status_per_rank = {f"rank{i}": "skipped" for i in range(world_size)}
            r.instrumentation = {
                "level": ",".join(sorted(analysis)) if analysis else "summary",
                "cuda_graph": bool(cand.cuda_graph),
                "cupti_available": False,
                "phase_timing_available": False,
                "kernel_breakdown_available": False,
            }
            results.append(r)
            skipped_global.append((workload, cand, reason))

        for cand in candidates:
            if deadline is not None and time.monotonic() > deadline:
                _maybe_print_rank0(
                    "[bench_moe] --time_budget_minutes exceeded; remaining candidates "
                    "will be reported as skipped."
                )
                r = RunResult(model=model, workload=workload, config=cand)
                r.status = "skipped"
                r.skip_reason = "time_budget_exceeded"
                r.per_rank_num_tokens = _per_rank_tokens(workload, world_size)
                r.status_per_rank = {f"rank{i}": "skipped" for i in range(world_size)}
                results.append(r)
                continue

            case_label = (
                f"backend={cand.backend} parallel_mode={cand.parallel_mode} "
                f"comm={cand.comm_method} eplb={cand.eplb_mode} "
                f"num_tokens={workload.num_tokens}"
            )
            _maybe_print_rank0(f"[bench_moe] running {case_label}")

            r = _run_one_candidate(
                model=model,
                workload=workload,
                config=cand,
                world_size=world_size,
                rank=rank,
                device=device,
                act_dtype=act_dtype,
                routing_logits_dtype=routing_logits_dtype,
                warmup=int(args.warmup),
                iters=int(args.iters),
                fast_autotune=bool(args.fast_autotune),
                analysis=analysis,
                cupti_ctx=_early_cupti_ctx,
                force_balance_patch=bool(args.force_balance_patch),
            )
            results.append(r)
            if rank == 0:
                print(json.dumps(_runresult_to_row(r), indent=2), flush=True)

    if rank == 0:
        rows = [_runresult_to_row(r) for r in results]
        rankings = _build_rankings(results)
        out_payload = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "bench_moe",
            "environment": _build_environment_block(world_size, bool(args.cuda_graph)),
            "model": model.to_dict(),
            "search": search.to_dict(),
            "base_config": base_config.to_dict(),
            "results": rows,
            "rankings": rankings,
        }
        if args.output_file:
            out_dir = os.path.dirname(args.output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_file, "w") as f:
                json.dump(out_payload, f, indent=2)
            print(f"Report written to {args.output_file}", flush=True)
        else:
            # Echo a one-shot summary block at the end so headless invocations
            # still produce machine-readable output on stdout.
            print(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "rankings": rankings,
                    },
                    indent=2,
                ),
                flush=True,
            )


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

    if world_size == 1:
        if mpi_rank() == 0:
            print(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "bench": "bench_moe",
                        "launcher": "inline_single_rank",
                        "world_size": 1,
                        "model": args.model,
                        "backend": args.backend,
                        "search": args.search,
                    },
                    indent=2,
                ),
                flush=True,
            )
        os.environ.update(_WORKER_ENV)
        args.world_size = 1
        _run_benchmark_worker_under_current_mpi(args, launcher="inline_single_rank")
        return

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
                    "schema_version": SCHEMA_VERSION,
                    "bench": "bench_moe",
                    "launcher": "spawn",
                    "world_size": world_size,
                    "model": args.model,
                    "backend": args.backend,
                    "search": args.search,
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
