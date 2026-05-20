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
backend x communication x parallel mode x CUDA graph x combine precision
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
from tensorrt_llm._utils import (  # noqa: E402
    local_mpi_rank,
    mpi_allgather,
    mpi_barrier,
    mpi_rank,
    mpi_world_size,
)
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig  # noqa: E402
from tensorrt_llm.tools.layer_wise_benchmarks.runner import make_forward_impl_check  # noqa: E402

# ---------------------------------------------------------------------------
# Output schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 5


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
class RoutingControlSpec:
    """Advanced routing-control knobs for one workload.

    See ``tests/microbenchmarks/BENCH_MOE_ROUTING_CONTROL_DESIGN.md`` for the
    full design. Briefly: ``comm_pattern`` and ``expert_pattern`` describe the
    requested traffic shape; ``routing_mode`` picks between native logits
    realization and forced supplied-topk; ``projection_policy`` controls what
    happens when native logits cannot exactly express the requested pattern.

    All fields default to "balanced everything via native logits", which keeps
    the benchmark behaviour identical to legacy invocations that did not
    request routing control.
    """

    routing_mode: str = "native"  # "native" | "forced"
    projection_policy: str = "project"  # "project" | "reject"
    comm_pattern: str = "balanced_alltoall"
    expert_pattern: str = "balanced"
    routing_pattern_file: Optional[str] = None
    per_rank_num_tokens: Optional[Tuple[int, ...]] = None
    routing_dump_matrix: bool = False
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_mode": self.routing_mode,
            "projection_policy": self.projection_policy,
            "comm_pattern": self.comm_pattern,
            "expert_pattern": self.expert_pattern,
            "routing_pattern_file": self.routing_pattern_file,
            "per_rank_num_tokens": (
                list(int(v) for v in self.per_rank_num_tokens)
                if self.per_rank_num_tokens is not None
                else None
            ),
            "routing_dump_matrix": bool(self.routing_dump_matrix),
            "seed": int(self.seed),
        }

    @property
    def is_active(self) -> bool:
        """True when this spec asks for non-default routing behaviour.

        Used to decide whether to dispatch through routing-control planning or
        keep the normal benchmark path.
        """
        return (
            self.routing_mode != "native"
            or self.comm_pattern != "balanced_alltoall"
            or self.expert_pattern != "balanced"
            or self.routing_pattern_file is not None
            or self.per_rank_num_tokens is not None
        )


@dataclass(frozen=True)
class WorkloadSpec:
    """Workload for one timing case after the model is fixed."""

    num_tokens: int
    routing_control: RoutingControlSpec = field(default_factory=RoutingControlSpec)

    def to_dict(self, per_rank_num_tokens: Optional[List[int]] = None) -> Dict[str, Any]:
        return {
            "num_tokens": int(self.num_tokens),
            "per_rank_num_tokens": (
                [int(v) for v in per_rank_num_tokens] if per_rank_num_tokens is not None else None
            ),
            "routing_control": self.routing_control.to_dict(),
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
    cuda_graph: bool = True
    use_low_precision_moe_combine: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "parallel_mode": self.parallel_mode,
            "moe_ep_size": self.moe_ep_size,
            "moe_tp_size": self.moe_tp_size,
            "enable_attention_dp": self.enable_attention_dp,
            "comm_method": self.comm_method,
            "cuda_graph": bool(self.cuda_graph),
            "use_low_precision_moe_combine": bool(self.use_low_precision_moe_combine),
        }


@dataclass(frozen=True)
class SearchSpec:
    """Description of which ConfigSpec axes to expand into candidates."""

    mode: str = "none"  # "none" | "backend" | "comm" | "parallel" | "full" | comma-joined axes
    backends: Tuple[str, ...] = ()
    parallel_modes: Tuple[str, ...] = ()
    comm_methods: Tuple[str, ...] = ()
    cuda_graph_options: Tuple[bool, ...] = ()
    combine_precision_options: Tuple[bool, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "backends": list(self.backends),
            "parallel_modes": list(self.parallel_modes),
            "comm_methods": list(self.comm_methods),
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
    routing_control: Dict[str, Any] = field(default_factory=dict)


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


def _distribute_tokens(total: int, world_size: int) -> List[int]:
    """Distribute ``total`` global tokens evenly across ``world_size`` ranks."""
    if world_size <= 0 or total < 0:
        raise ValueError(f"invalid args: total={total}, world_size={world_size}")
    if world_size == 1:
        return [total]
    base = total // world_size
    out = [base] * world_size
    out[0] += total - base * world_size
    return out


def _per_rank_tokens(workload: WorkloadSpec, world_size: int) -> List[int]:
    """Materialize the ``per_rank_num_tokens`` list for a workload + world size."""
    per_rank = workload.routing_control.per_rank_num_tokens
    if per_rank is None:
        return _distribute_tokens(int(workload.num_tokens), world_size)
    out = [int(v) for v in per_rank]
    if len(out) != world_size:
        raise ValueError(
            f"per_rank_num_tokens has length {len(out)}, expected world_size={world_size}"
        )
    if sum(out) != int(workload.num_tokens):
        raise ValueError(
            f"sum(per_rank_num_tokens)={sum(out)} must equal num_tokens={workload.num_tokens}"
        )
    if any(v < 0 for v in out):
        raise ValueError("per_rank_num_tokens entries must be >= 0")
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


def _build_model_config(
    *,
    model: ModelSpec,
    mapping: Mapping,
    moe_backend: str,
    use_cuda_graph: bool,
    max_num_tokens: int,
    use_low_precision_moe_combine: bool,
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

    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=moe_backend,
        moe_disable_finalize_fusion=False,
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
    enable_perfect_router: bool,
    dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    device: torch.device,
):
    """Build a fresh ``ConfigurableMoE`` for one ``(backend, num_tokens)`` case.

    Returns ``(moe_module, routing_logits_dtype)``.
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
        mapping=mapping,
        moe_backend=moe_backend,
        use_cuda_graph=use_cuda_graph,
        max_num_tokens=max_num_tokens,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
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

    swiglu_tensors = quantize_util.get_swiglu_tensors()

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

    moe.load_weights([weights])
    moe.post_load_weights()
    moe.cuda(f"cuda:{torch.cuda.current_device()}")

    return moe, routing_logits_dtype


# ---------------------------------------------------------------------------
# Routing control: pattern parsing, plan building, materialization
#
# See ``tests/microbenchmarks/BENCH_MOE_ROUTING_CONTROL_DESIGN.md`` for the
# full design. Highlights:
#   - ``RoutingPlan`` is the canonical normalised form (per-rank tokens +
#     slot dispatch matrix + expert histogram).
#   - Pattern parsers turn user-facing strings like
#     ``"receiver_hotspot,hotness=0.75,rank=0"`` into structured kwargs.
#   - Plan builders generate ``dispatch_matrix`` and ``expert_histogram`` for
#     the requested pattern.
#   - The materializer turns the plan into a per-rank ``selected_experts``
#     tensor and reports the observed slot/token traffic + expert histogram.
#   - The native logits planner reverse-engineers a ``router_logits`` tensor
#     that drives the actual production routing kernels towards the requested
#     plan. The exact-reachability table follows the design document.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingPlan:
    """Canonical normalised routing plan.

    ``per_rank_num_tokens[src]`` is the local input token count on source rank
    ``src``. ``dispatch_matrix[src][dst]`` is the *slot* count (each selected
    (token, expert) slot counts once) sent from ``src`` to ``dst``. Row sums
    are ``per_rank_num_tokens[src] * top_k``. ``expert_histogram[dst][le]`` is
    the global slot count owned by local expert ``le`` on rank ``dst``.
    """

    per_rank_num_tokens: Tuple[int, ...]
    dispatch_matrix: Tuple[Tuple[int, ...], ...]
    expert_histogram: Tuple[Tuple[int, ...], ...]
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_rank_num_tokens": [int(v) for v in self.per_rank_num_tokens],
            "dispatch_matrix": [[int(v) for v in row] for row in self.dispatch_matrix],
            "expert_histogram": [[int(v) for v in row] for row in self.expert_histogram],
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class RoutingProjectionResult:
    """Outcome of trying to realise a ``RoutingPlan`` for a given routing method."""

    router_logits: Optional[torch.Tensor]
    status: str  # "exact" | "projected" | "rejected" | "forced_exact" | "not_applicable"
    reason: str
    observed_slot_dispatch_matrix: Tuple[Tuple[int, ...], ...]
    observed_token_dispatch_matrix: Tuple[Tuple[int, ...], ...]
    observed_expert_histogram: Tuple[Tuple[int, ...], ...]
    max_abs_slot_error: int
    max_relative_slot_error: float
    selected_experts: Optional[torch.Tensor]
    selected_scales: Optional[torch.Tensor]
    warnings: Tuple[str, ...] = ()


# --- pattern parsing -------------------------------------------------------


_COMM_PATTERN_NAMES: Tuple[str, ...] = (
    "balanced_alltoall",
    "receiver_hotspot",
    "pair_hotspot",
    "local_only",
    "ring",
)

_EXPERT_PATTERN_NAMES: Tuple[str, ...] = ("balanced", "hotspot")


def _parse_pattern_spec(spec: str) -> Tuple[str, Dict[str, str]]:
    """Parse ``name,k1=v1,k2=v2`` into ``(name, {k1: v1, k2: v2})``.

    File-based routing control is handled by ``--routing_pattern_file``.
    """
    raw = str(spec).strip()
    if not raw:
        raise ValueError("empty pattern spec")
    if raw.startswith("file:"):
        return "file", {"path": raw[len("file:") :]}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"invalid pattern spec: {spec!r}")
    name = parts[0]
    kwargs: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"invalid pattern fragment {part!r} in {spec!r}; expected k=v")
        k, v = part.split("=", 1)
        kwargs[k.strip()] = v.strip()
    return name, kwargs


def _parse_comm_pattern(spec: str) -> Tuple[str, Dict[str, Any]]:
    name, raw = _parse_pattern_spec(spec)
    if name == "file":
        raise ValueError("comm_pattern no longer accepts file:<path>; use --routing_pattern_file")
    if name not in _COMM_PATTERN_NAMES:
        raise ValueError(f"unknown comm_pattern {name!r}; supported: {_COMM_PATTERN_NAMES}")
    kwargs: Dict[str, Any] = {}
    if "hotness" in raw:
        kwargs["hotness"] = float(raw["hotness"])
        if not 0.0 <= kwargs["hotness"] <= 1.0:
            raise ValueError(f"comm_pattern hotness must be in [0, 1]; got {kwargs['hotness']}")
    if "rank" in raw:
        kwargs["rank"] = int(raw["rank"])
    if "src" in raw:
        kwargs["src"] = int(raw["src"])
    if "dst" in raw:
        kwargs["dst"] = int(raw["dst"])
    if name == "receiver_hotspot":
        if "hotness" not in kwargs:
            raise ValueError("receiver_hotspot requires hotness=<ratio>")
        kwargs.setdefault("rank", 0)
    if name == "pair_hotspot":
        if "hotness" not in kwargs or "src" not in kwargs or "dst" not in kwargs:
            raise ValueError("pair_hotspot requires hotness=<ratio>, src=<src>, dst=<dst>")
    return name, kwargs


def _parse_expert_pattern(spec: str) -> Tuple[str, Dict[str, Any]]:
    name, raw = _parse_pattern_spec(spec)
    if name == "file":
        raise ValueError("expert_pattern no longer accepts file:<path>; use --routing_pattern_file")
    if name not in _EXPERT_PATTERN_NAMES:
        raise ValueError(f"unknown expert_pattern {name!r}; supported: {_EXPERT_PATTERN_NAMES}")
    kwargs: Dict[str, Any] = {}
    if "hotness" in raw:
        kwargs["hotness"] = float(raw["hotness"])
        if not 0.0 <= kwargs["hotness"] <= 1.0:
            raise ValueError(f"expert_pattern hotness must be in [0, 1]; got {kwargs['hotness']}")
    if "active_experts" in raw:
        kwargs["active_experts"] = int(raw["active_experts"])
        if kwargs["active_experts"] <= 0:
            raise ValueError("expert_pattern active_experts must be > 0")
    if name == "hotspot" and "hotness" not in kwargs and "active_experts" not in kwargs:
        raise ValueError(
            "expert_pattern hotspot requires hotness=<ratio> or active_experts=<count>"
        )
    return name, kwargs


# --- per-rank token / dispatch matrix / expert histogram builders ----------


def _largest_remainder_split(total: int, weights: List[float]) -> List[int]:
    """Split ``total`` integer units among bins using largest-remainder method.

    All weights must be non-negative. Zero-weight bins always receive zero
    units. The result is deterministic for ties (ties broken by lower index).
    """
    n = len(weights)
    if n == 0:
        return []
    total = int(total)
    if total <= 0:
        return [0] * n
    s = sum(weights)
    if s <= 0.0:
        # Distribute evenly when all weights are zero.
        base = total // n
        rem = total - base * n
        out = [base] * n
        for i in range(rem):
            out[i] += 1
        return out
    raw = [total * (w / s) for w in weights]
    floors = [int(x) for x in raw]
    used = sum(floors)
    remainders = sorted(
        ((raw[i] - floors[i], -i) for i in range(n)), key=lambda pair: pair[0], reverse=True
    )
    for k in range(total - used):
        _, neg_idx = remainders[k % n]
        floors[-neg_idx] += 1
    return floors


def _build_per_rank_num_tokens(
    spec: RoutingControlSpec,
    num_tokens: int,
    world_size: int,
) -> List[int]:
    """Resolve ``per_rank_num_tokens`` for a workload.

    Explicit ``spec.per_rank_num_tokens`` wins; otherwise tokens are split
    evenly across ranks with any remainder on rank 0.
    """
    if spec.per_rank_num_tokens is not None:
        per_rank = [int(v) for v in spec.per_rank_num_tokens]
        if len(per_rank) != world_size:
            raise ValueError(
                f"per_rank_num_tokens has length {len(per_rank)}, expected world_size={world_size}"
            )
        if any(v < 0 for v in per_rank):
            raise ValueError("per_rank_num_tokens entries must be >= 0")
        if sum(per_rank) != int(num_tokens):
            raise ValueError(
                f"sum(per_rank_num_tokens)={sum(per_rank)} must equal num_tokens={num_tokens}"
            )
        return per_rank
    return _distribute_tokens(int(num_tokens), world_size)


def _build_dispatch_matrix(
    comm_pattern: str,
    per_rank_num_tokens: List[int],
    top_k: int,
    ep_size: int,
) -> List[List[int]]:
    """Build the canonical slot ``dispatch_matrix`` for ``comm_pattern``.

    Row sums always equal ``per_rank_num_tokens[src] * top_k``. The matrix is
    a pure planning artefact: it does not enforce per-token uniqueness yet.
    That constraint is checked at materialisation time.
    """
    name, kwargs = _parse_comm_pattern(comm_pattern)
    matrix: List[List[int]] = [[0] * ep_size for _ in range(ep_size)]
    for src in range(ep_size):
        row_total = int(per_rank_num_tokens[src]) * int(top_k)
        if row_total == 0:
            continue
        if name == "file":
            # Loaded separately by ``_load_dispatch_matrix_file``.
            raise ValueError(
                "file:<path> dispatch matrices are loaded via _load_dispatch_matrix_file"
            )
        elif name == "balanced_alltoall":
            weights = [1.0] * ep_size
        elif name == "receiver_hotspot":
            hot_rank = int(kwargs["rank"])
            if not 0 <= hot_rank < ep_size:
                raise ValueError(f"receiver_hotspot rank={hot_rank} out of range [0, {ep_size})")
            hotness = float(kwargs["hotness"])
            weights = [(1.0 - hotness) / max(ep_size, 1)] * ep_size
            weights[hot_rank] += hotness
        elif name == "pair_hotspot":
            pair_src = int(kwargs["src"])
            pair_dst = int(kwargs["dst"])
            if not 0 <= pair_src < ep_size or not 0 <= pair_dst < ep_size:
                raise ValueError(
                    f"pair_hotspot src/dst must be in [0, {ep_size}); got src={pair_src}, dst={pair_dst}"
                )
            hotness = float(kwargs["hotness"])
            if src == pair_src:
                weights = [(1.0 - hotness) / max(ep_size, 1)] * ep_size
                weights[pair_dst] += hotness
            else:
                weights = [1.0] * ep_size
        elif name == "local_only":
            weights = [0.0] * ep_size
            weights[src] = 1.0
        elif name == "ring":
            weights = [0.0] * ep_size
            weights[(src + 1) % ep_size] = 1.0
        else:
            raise ValueError(f"unknown comm_pattern {name!r}")
        matrix[src] = _largest_remainder_split(row_total, weights)
    return matrix


def _build_expert_histogram(
    expert_pattern: str,
    dispatch_matrix: List[List[int]],
    experts_per_rank: int,
    ep_size: int,
) -> List[List[int]]:
    """Build the canonical global ``expert_histogram[ep_size][experts_per_rank]``."""
    name, kwargs = _parse_expert_pattern(expert_pattern)
    histogram: List[List[int]] = [[0] * experts_per_rank for _ in range(ep_size)]
    # Per-target slot totals come from column sums of the dispatch matrix.
    col_sums = [sum(dispatch_matrix[src][dst] for src in range(ep_size)) for dst in range(ep_size)]
    for dst in range(ep_size):
        target_total = int(col_sums[dst])
        if target_total <= 0:
            continue
        if name == "file":
            raise ValueError(
                "file:<path> expert histograms are loaded via _load_expert_histogram_file"
            )
        elif name == "balanced":
            weights = [1.0] * experts_per_rank
        elif name == "hotspot":
            if "active_experts" in kwargs:
                active = int(kwargs["active_experts"])
                active = max(1, min(active, experts_per_rank))
                weights = [1.0 if le < active else 0.0 for le in range(experts_per_rank)]
            else:
                hotness = float(kwargs["hotness"])
                # Concentrate ``hotness`` fraction onto expert 0; spread the
                # remainder uniformly.
                weights = [(1.0 - hotness) / max(experts_per_rank, 1)] * experts_per_rank
                weights[0] += hotness
        else:
            raise ValueError(f"unknown expert_pattern {name!r}")
        histogram[dst] = _largest_remainder_split(target_total, weights)
    return histogram


def _load_dispatch_matrix_file(path: str, ep_size: int) -> List[List[int]]:
    """Load and validate a ``file:<path>`` slot dispatch matrix."""
    with open(path) as f:
        payload = json.load(f)
    if int(payload.get("ep_size", -1)) != ep_size:
        raise ValueError(
            f"dispatch matrix {path!r}: ep_size={payload.get('ep_size')} mismatches runtime ep_size={ep_size}"
        )
    matrix = payload.get("slot_dispatch_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != ep_size
        or any(not isinstance(row, list) or len(row) != ep_size for row in matrix)
    ):
        raise ValueError(
            f"dispatch matrix {path!r}: slot_dispatch_matrix must be a {ep_size}x{ep_size} integer list"
        )
    return [[int(v) for v in row] for row in matrix]


def _load_expert_histogram_file(path: str, ep_size: int, experts_per_rank: int) -> List[List[int]]:
    """Load and validate a ``file:<path>`` expert histogram."""
    with open(path) as f:
        payload = json.load(f)
    if int(payload.get("ep_size", -1)) != ep_size:
        raise ValueError(
            f"expert histogram {path!r}: ep_size={payload.get('ep_size')} mismatches runtime ep_size={ep_size}"
        )
    if int(payload.get("experts_per_rank", -1)) != experts_per_rank:
        raise ValueError(
            f"expert histogram {path!r}: experts_per_rank={payload.get('experts_per_rank')} mismatches "
            f"runtime experts_per_rank={experts_per_rank}"
        )
    histogram = payload.get("expert_histogram")
    if (
        not isinstance(histogram, list)
        or len(histogram) != ep_size
        or any(not isinstance(row, list) or len(row) != experts_per_rank for row in histogram)
    ):
        raise ValueError(
            f"expert histogram {path!r}: expert_histogram must be a {ep_size}x{experts_per_rank} integer list"
        )
    return [[int(v) for v in row] for row in histogram]


def _load_routing_pattern_file(
    path: str, ep_size: int, experts_per_rank: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """Load a single file that fixes both dispatch traffic and expert histogram."""
    return (
        _load_dispatch_matrix_file(path, ep_size),
        _load_expert_histogram_file(path, ep_size, experts_per_rank),
    )


def _build_routing_plan(
    spec: RoutingControlSpec,
    num_tokens: int,
    world_size: int,
    top_k: int,
    num_experts: int,
    moe_ep_size: int,
) -> RoutingPlan:
    """Translate a ``RoutingControlSpec`` into a canonical normalised plan."""
    if moe_ep_size <= 0 or num_experts % moe_ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by moe_ep_size ({moe_ep_size})"
        )
    experts_per_rank = num_experts // moe_ep_size
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    per_rank = _build_per_rank_num_tokens(spec, num_tokens, world_size)
    # ``moe_ep_size`` may be smaller than ``world_size`` when MoE is replicated
    # across attention DP slots; the dispatch matrix indexing follows the EP
    # axis, but materialisation is still per source rank. For the v1 dispatch
    # matrix abstraction we use ``moe_ep_size`` as both axes.
    if spec.routing_pattern_file:
        if spec.comm_pattern != "balanced_alltoall" or spec.expert_pattern != "balanced":
            raise ValueError(
                "--routing_pattern_file cannot be combined with non-default "
                "--comm_pattern or --expert_pattern"
            )
        dispatch_matrix, expert_histogram = _load_routing_pattern_file(
            spec.routing_pattern_file, moe_ep_size, experts_per_rank
        )
    else:
        dispatch_matrix = _build_dispatch_matrix(spec.comm_pattern, per_rank, top_k, moe_ep_size)
        expert_histogram = _build_expert_histogram(
            spec.expert_pattern, dispatch_matrix, experts_per_rank, moe_ep_size
        )

    # Per-row sums are an invariant; emit a clearer error than the materialiser would.
    for src in range(moe_ep_size):
        expected = int(per_rank[src]) * int(top_k) if src < len(per_rank) else 0
        actual = sum(dispatch_matrix[src])
        if actual != expected:
            raise ValueError(
                f"dispatch_matrix row {src} sums to {actual}, expected per_rank_num_tokens[{src}] * top_k = {expected}"
            )
    # Global expert histogram total must match total slots.
    total_slots = sum(int(t) for t in per_rank) * int(top_k)
    hist_total = sum(sum(row) for row in expert_histogram)
    if hist_total != total_slots:
        raise ValueError(
            f"expert_histogram sum={hist_total} must equal sum(per_rank_num_tokens) * top_k = {total_slots}"
        )

    return RoutingPlan(
        per_rank_num_tokens=tuple(int(v) for v in per_rank),
        dispatch_matrix=tuple(tuple(int(v) for v in row) for row in dispatch_matrix),
        expert_histogram=tuple(tuple(int(v) for v in row) for row in expert_histogram),
        seed=int(spec.seed),
    )


# --- materialisation -------------------------------------------------------


def _split_slot_count_to_experts(
    slot_count: int,
    target_histogram_row: List[int],
) -> List[int]:
    """Allocate ``slot_count`` slots across local experts proportionally.

    Largest-remainder over ``target_histogram_row`` ensures the per-local-expert
    distribution within this (src, dst) cell tracks the global histogram for
    the target rank. Returns a list of length ``len(target_histogram_row)``.
    """
    weights = [float(v) for v in target_histogram_row]
    return _largest_remainder_split(int(slot_count), weights)


def _materialize_selected_experts_for_rank(
    plan: RoutingPlan,
    src_rank: int,
    top_k: int,
    experts_per_rank: int,
    moe_ep_size: int,
    device: torch.device,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise ``[local_num_tokens, top_k]`` expert ids + uniform scales.

    The algorithm:
      1. Flatten ``dispatch_matrix[src_rank]`` into a slot-count-per-(dst, le)
         table by splitting the row counts across local experts proportional
         to the target rank's global histogram.
      2. Build a flat list of expert ids of length ``local_num_tokens * top_k``.
      3. Reshape column-major (k=0 first across tokens, then k=1, ...) so
         that within a row consecutive slots come from different "buckets" and
         per-token expert ids stay distinct in practice.
      4. Run a small repair pass that swaps duplicated expert ids between
         rows until each token has ``top_k`` distinct experts.
    """
    local_num_tokens = int(plan.per_rank_num_tokens[src_rank])
    if local_num_tokens == 0:
        ids = torch.zeros((0, top_k), dtype=torch.int32, device=device)
        scales = torch.zeros((0, top_k), dtype=scale_dtype, device=device)
        return ids, scales

    row = list(plan.dispatch_matrix[src_rank])
    if sum(row) != local_num_tokens * top_k:
        raise ValueError(
            f"dispatch_matrix row sum ({sum(row)}) must equal local_num_tokens*top_k "
            f"({local_num_tokens * top_k}) for rank {src_rank}"
        )

    flat: List[int] = []
    for dst in range(moe_ep_size):
        cell = int(row[dst])
        if cell == 0:
            continue
        target_hist = list(plan.expert_histogram[dst])
        per_le = _split_slot_count_to_experts(cell, target_hist)
        for le, cnt in enumerate(per_le):
            if cnt <= 0:
                continue
            expert_id = dst * experts_per_rank + le
            flat.extend([expert_id] * int(cnt))

    if len(flat) != local_num_tokens * top_k:
        raise ValueError(
            f"materialiser flat length {len(flat)} != local_num_tokens*top_k={local_num_tokens * top_k}"
        )

    # Reshape column-major: pos i -> (token_idx=i%num_tokens, k_idx=i//num_tokens).
    out = [[0] * top_k for _ in range(local_num_tokens)]
    for i, val in enumerate(flat):
        k_idx = i // local_num_tokens
        t_idx = i % local_num_tokens
        out[t_idx][k_idx] = val

    # Repair duplicates: for any token with duplicated expert ids, swap one of
    # the duplicates with a different expert id from another token.
    max_passes = 4
    for _pass in range(max_passes):
        any_repair = False
        for t in range(local_num_tokens):
            seen: Dict[int, int] = {}
            for k in range(top_k):
                eid = out[t][k]
                if eid in seen:
                    # Find a swap partner: another row whose (k slot) has a
                    # different expert id and that does not introduce a new
                    # duplicate.
                    target_k = k
                    swapped = False
                    for t2 in range(local_num_tokens):
                        if t2 == t:
                            continue
                        partner = out[t2][target_k]
                        if partner == eid:
                            continue
                        if partner in seen:
                            continue
                        out[t][target_k], out[t2][target_k] = partner, eid
                        swapped = True
                        any_repair = True
                        break
                    if not swapped:
                        # Try swapping the k-th slot with a different k slot
                        # inside the same row.
                        for k2 in range(top_k):
                            if k2 == k:
                                continue
                            alt = out[t][k2]
                            if alt == eid or alt in seen:
                                continue
                            out[t][k], out[t][k2] = alt, eid
                            any_repair = True
                            break
                    seen[out[t][k]] = k
                else:
                    seen[eid] = k
        if not any_repair:
            break

    ids = torch.tensor(out, dtype=torch.int32, device=device)
    scales = torch.full(
        (local_num_tokens, top_k), 1.0 / max(top_k, 1), dtype=scale_dtype, device=device
    )
    return ids, scales


def _observe_routing_metrics(
    plan: RoutingPlan,
    selected_experts_per_rank: List[torch.Tensor],
    experts_per_rank: int,
    moe_ep_size: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Derive observed slot/token traffic and expert histogram from materialised ids."""
    slot_traffic = [[0] * moe_ep_size for _ in range(moe_ep_size)]
    token_traffic = [[0] * moe_ep_size for _ in range(moe_ep_size)]
    expert_hist = [[0] * experts_per_rank for _ in range(moe_ep_size)]
    for src, ids in enumerate(selected_experts_per_rank):
        if ids is None or ids.numel() == 0:
            continue
        ids_cpu = ids.detach().cpu().numpy() if not isinstance(ids, list) else ids
        for row in ids_cpu:
            dst_visited = set()
            for eid in row:
                eid_int = int(eid)
                dst = eid_int // experts_per_rank
                le = eid_int % experts_per_rank
                if 0 <= dst < moe_ep_size and 0 <= le < experts_per_rank:
                    slot_traffic[src][dst] += 1
                    expert_hist[dst][le] += 1
                    if dst not in dst_visited:
                        token_traffic[src][dst] += 1
                        dst_visited.add(dst)
    return slot_traffic, token_traffic, expert_hist


def _observe_summary(
    requested_slot: List[List[int]],
    observed_slot: List[List[int]],
) -> Tuple[int, float]:
    """Return ``(max_abs_slot_error, max_relative_slot_error)``."""
    max_abs = 0
    max_rel = 0.0
    for src in range(len(observed_slot)):
        for dst in range(len(observed_slot[src])):
            req = int(requested_slot[src][dst]) if src < len(requested_slot) else 0
            obs = int(observed_slot[src][dst])
            abs_err = abs(obs - req)
            if abs_err > max_abs:
                max_abs = abs_err
            denom = max(req, 1)
            rel_err = abs_err / denom
            if rel_err > max_rel:
                max_rel = rel_err
    return int(max_abs), float(max_rel)


# --- native logits projection ---------------------------------------------


_NATIVE_PROJECTION_CAPABILITIES: Dict[str, str] = {
    "DefaultMoeRoutingMethod": "exact_ids",
    "RenormalizeMoeRoutingMethod": "exact",
    "RenormalizeNaiveMoeRoutingMethod": "exact",
    "SigmoidRenormMoeRoutingMethod": "exact_ids",
    "Llama4RenormalizeMoeRoutingMethod": "top1_exact",
    "MiniMaxM2MoeRoutingMethod": "exact_with_zero_bias",
    "DeepSeekV3MoeRoutingMethod": "projected_or_exact",
    "SparseMixerMoeRoutingMethod": "unsupported",
}


def _project_router_logits_for_plan(
    plan: RoutingPlan,
    src_rank: int,
    routing_method,
    num_experts: int,
    top_k: int,
    experts_per_rank: int,
    moe_ep_size: int,
    device: torch.device,
    dtype: torch.dtype,
    high_logit: float = 10.0,
    low_logit: float = -10.0,
) -> Tuple[torch.Tensor, str, str]:
    """Build router_logits for ``routing_method.apply`` matching the plan.

    Construct logits such that ``routing_method.apply`` yields a top-k matching
    ``plan``'s ``[local_num_tokens, top_k]`` materialised expert ids.

    Returns ``(router_logits, status, reason)`` where ``status`` is one of
    ``"exact"``, ``"projected"``, or ``"rejected"``.
    """
    local_num_tokens = int(plan.per_rank_num_tokens[src_rank])
    if local_num_tokens == 0:
        return (
            torch.empty((0, num_experts), dtype=dtype, device=device),
            "exact",
            "no local tokens; trivial logits",
        )

    method_name = type(routing_method).__name__
    capability = _NATIVE_PROJECTION_CAPABILITIES.get(method_name, "unsupported")

    # Materialise target experts using the canonical plan, then construct
    # logits that drive the routing kernels towards those experts.
    ids, _ = _materialize_selected_experts_for_rank(
        plan,
        src_rank=src_rank,
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        moe_ep_size=moe_ep_size,
        device=device,
        scale_dtype=dtype if dtype.is_floating_point else torch.bfloat16,
    )

    # Base low / high pattern with a small monotone perturbation by k index so
    # that ties inside a row are broken consistently.
    logits = torch.full((local_num_tokens, num_experts), low_logit, dtype=dtype, device=device)
    k_offsets = torch.linspace(0.0, 1.0, steps=top_k, device=device, dtype=dtype) * 0.01
    # Score per slot decreases with k_idx so that top-k tie-breaking yields the
    # same expert ordering when scales are derived from logits.
    score_per_k = high_logit + (k_offsets.flip(dims=(0,)) - 0.005)
    row_idx = (
        torch.arange(local_num_tokens, device=device).unsqueeze(1).expand(local_num_tokens, top_k)
    )
    logits[row_idx, ids.long()] = score_per_k.unsqueeze(0).expand_as(ids).to(dtype)

    status = "exact"
    reason = "high/low logits drive top-k to plan"

    if capability == "exact":
        pass
    elif capability == "exact_ids":
        reason = (
            "expert ids exactly realised; selected_scales follow native routing kernel and are not "
            "matrix-controlled"
        )
    elif capability == "top1_exact":
        # Llama4 only uses top1 routing semantically; if the plan asks for top-k
        # > 1 we cannot exactly express the same workload via native logits.
        if top_k > 1:
            status = "projected"
            reason = (
                "Llama4 native realisation is only exact for top1; multi-target plans are projected"
            )
    elif capability == "exact_with_zero_bias":
        # MiniMax2 selects via ``sigmoid(logits) + bias``; with zero bias the
        # ids are still exact but final scales come from a bias-less sigmoid.
        reason = "MiniMax2 exact realisation assumes zero score-correction bias"
    elif capability == "projected_or_exact":
        # DeepSeekV3 grouped routing: every selected expert must belong to one
        # of the top ``topk_group`` groups (out of ``n_group``). Detect a
        # violation by counting groups per token.
        routing_impl = getattr(routing_method, "routing_impl", None)
        n_group = getattr(routing_impl, "n_group", 1) if routing_impl is not None else 1
        topk_group = getattr(routing_impl, "topk_group", 1) if routing_impl is not None else 1
        if n_group > 1 and topk_group >= 1:
            experts_per_group = num_experts // n_group
            ids_cpu = ids.detach().cpu().tolist()
            for row in ids_cpu:
                groups = {int(eid) // experts_per_group for eid in row}
                if len(groups) > topk_group:
                    status = "projected"
                    reason = (
                        f"DeepSeekV3 grouped routing: row needs experts in {len(groups)} groups "
                        f"but topk_group={topk_group}"
                    )
                    break
    elif capability == "unsupported":
        status = "projected"
        reason = f"{method_name} native logits realisation is unsupported in v1; falling back to high/low logits"
    else:
        status = "projected"
        reason = f"{method_name}: unknown capability"

    return logits, status, reason


def _make_supplied_topk_run_moe(
    moe_module,
    run_moe_orig,
    materialized_ids: torch.Tensor,
    materialized_scales: torch.Tensor,
):
    """Return a ``run_moe`` wrapper that injects pre-materialised top-k tensors.

    Mirrors the ``make_balanced_run_moe`` helper in the layer-wise benchmark
    runner, but feeds the routing-control plan instead of the legacy
    balanced/imbalanced selection helpers.
    """

    def supplied_run_moe(
        x, token_selected_experts, token_final_scales, x_sf, router_logits, do_finalize, moe_output
    ):
        if getattr(moe_module, "_routing_results_replaced_at", None) is not None:
            return run_moe_orig(
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                router_logits,
                do_finalize,
                moe_output,
            )
        local = materialized_ids
        scales = materialized_scales
        # If the engine resized the local batch (e.g. CUDA graph padding), align
        # to the runtime ``x.shape[0]`` so we never index out of bounds.
        if local.shape[0] != x.shape[0]:
            if local.shape[0] >= x.shape[0]:
                local = local[: x.shape[0]]
                scales = scales[: x.shape[0]]
            else:
                # Pad by repeating the last row; rare, only happens when the
                # backend over-allocates the batch dimension.
                pad_rows = x.shape[0] - local.shape[0]
                local = torch.cat([local, local[-1:].expand(pad_rows, -1).clone()], dim=0)
                scales = torch.cat([scales, scales[-1:].expand(pad_rows, -1).clone()], dim=0)
        local = local.to(device=x.device, dtype=torch.int32)
        scales = scales.to(device=x.device)
        final_hidden_states = run_moe_orig(x, local, scales, x_sf, None, do_finalize, moe_output)
        if not do_finalize:
            final_hidden_states = (
                final_hidden_states[0],
                scales,
                final_hidden_states[2],
            )
        moe_module._routing_results_replaced_at = "make_supplied_topk_run_moe"
        return final_hidden_states

    return supplied_run_moe


def _make_supplied_topk_apply(
    moe_module,
    apply_orig,
    materialized_ids: torch.Tensor,
    materialized_scales: torch.Tensor,
):
    """Return a ``routing_method.apply`` wrapper returning the plan directly."""

    def supplied_apply(router_logits):
        # Honour the original apply to preserve dtypes/device when possible,
        # but discard its result.
        try:
            _ = apply_orig(router_logits)
        except Exception:
            pass
        local = materialized_ids
        scales = materialized_scales
        if router_logits is not None and local.shape[0] != router_logits.shape[0]:
            if local.shape[0] >= router_logits.shape[0]:
                local = local[: router_logits.shape[0]]
                scales = scales[: router_logits.shape[0]]
            else:
                pad_rows = router_logits.shape[0] - local.shape[0]
                local = torch.cat([local, local[-1:].expand(pad_rows, -1).clone()], dim=0)
                scales = torch.cat([scales, scales[-1:].expand(pad_rows, -1).clone()], dim=0)
        device = router_logits.device if router_logits is not None else local.device
        moe_module._routing_results_replaced_at = "make_supplied_topk_apply"
        return local.to(device=device, dtype=torch.int32), scales.to(device=device)

    return supplied_apply


@contextlib.contextmanager
def _maybe_install_routing_control_patch(
    moe,
    materialized_ids: Optional[torch.Tensor],
    materialized_scales: Optional[torch.Tensor],
    active: bool,
):
    """Install supplied-topk patches when routing control is active in forced mode.

    For non-TRTLLM backends we override ``routing_method.apply`` to return the
    pre-materialised ``(ids, scales)`` pair. For ``TRTLLMGenFusedMoE`` the
    fused TEP path needs ``run_moe`` to be patched as well, mirroring the
    layer-wise benchmark's ``make_balanced_run_moe`` flow.

    ``active=False`` makes this a no-op pass-through so legacy callers keep
    behaving exactly as before.
    """
    if not active or materialized_ids is None or materialized_scales is None:
        yield
        return

    routing_target = moe
    apply_method_orig = routing_target.routing_method.apply
    routing_target.routing_method.apply = _make_supplied_topk_apply(
        routing_target, apply_method_orig, materialized_ids, materialized_scales
    )

    inner_backend = getattr(moe, "backend", moe)
    run_moe_orig = None
    if isinstance(inner_backend, TRTLLMGenFusedMoE):
        run_moe_orig = inner_backend.run_moe
        inner_backend.run_moe = _make_supplied_topk_run_moe(
            inner_backend, run_moe_orig, materialized_ids, materialized_scales
        )

    forward_impl_orig = moe.forward_impl
    moe.forward_impl = make_forward_impl_check(moe, forward_impl_orig)

    try:
        yield
    finally:
        routing_target.routing_method.apply = apply_method_orig
        if run_moe_orig is not None:
            getattr(moe, "backend", moe).run_moe = run_moe_orig
        moe.forward_impl = forward_impl_orig


# ---------------------------------------------------------------------------
# Autotune (untimed pre-pass)
# ---------------------------------------------------------------------------


def _run_autotune(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    fast_autotune: bool,
) -> str:
    """One untimed forward pass under ``autotune(...)`` to populate kernel caches.

    Returns an autotune status string, one of:
      - ``"success"``       : ran with the project default tuner settings
      - ``"success:fast"``  : ran with ``--fast_autotune`` overrides (lower quality)
      - ``"failed:<reason>"``: the autotune pass raised; caller decides whether to
                              trust the subsequent timings

    The function always restores ``AutoTuner`` singleton state on exit so that
    ``--fast_autotune`` set for one case does not leak into the next.
    """
    tuner = AutoTuner.get()
    saved_warmup = tuner.warmup
    saved_repeat = tuner.repeat
    saved_stream_delay = tuner.stream_delay_micro_secs
    if fast_autotune:
        tuner.warmup = 0
        tuner.repeat = 1
        tuner.stream_delay_micro_secs = 10

    cache_path = os.path.join(tempfile.gettempdir(), "bench_moe_autotuner_cache.json")
    try:
        with torch.inference_mode(), autotune(cache_path=cache_path):
            moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
        torch.cuda.synchronize()
        return "success:fast" if fast_autotune else "success"
    finally:
        tuner.warmup = saved_warmup
        tuner.repeat = saved_repeat
        tuner.stream_delay_micro_secs = saved_stream_delay


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
    flush_l2: bool = True,
    collect_kernels: bool = True,
) -> Tuple[List[float], Dict[str, Any]]:
    """Time eager ``ConfigurableMoE.forward``.

    Latency is ALWAYS measured with pure ``torch.cuda.Event`` records so the
    reported ``score_us`` is comparable across ``cuda_graph`` and eager paths
    (the CUDA-Graph path also uses external CUDA events for its per-iter
    window). When ``collect_kernels=True`` a separate, shorter profiler pass is
    run only to gather the kernel breakdown; profiler-derived numbers are not
    used to score the candidate.
    """
    device = x.device if x.numel() > 0 else torch.device("cuda")
    l2_buffer = _l2_flush_buffer(device) if flush_l2 else None

    def _do_forward():
        with torch.inference_mode():
            _ = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    _sync()
    for _ in range(warmup):
        if l2_buffer is not None:
            l2_buffer.zero_()
        _do_forward()
    for i in range(iters):
        if l2_buffer is not None:
            l2_buffer.zero_()
        starts[i].record()
        _do_forward()
        ends[i].record()
    _sync()
    forward_times_us = [starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters)]

    detailed_stats: Dict[str, Any] = {
        "moe_forward_kernels": [],
        "other_kernels": [],
    }
    if not collect_kernels:
        return forward_times_us, detailed_stats

    # Separate profiler-only pass for kernel breakdown. Use a small fixed iter
    # count (capped by ``iters``) so the profiler overhead does not dominate
    # the case wall-clock budget. The latencies produced here are intentionally
    # discarded; only the kernel categorisation is kept.
    breakdown_iters = max(1, min(iters, 3))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            _sync()
            if l2_buffer is not None:
                l2_buffer.zero_()
            _do_forward()  # one warmup under profiler
            for _ in range(breakdown_iters):
                if l2_buffer is not None:
                    l2_buffer.zero_()
                with torch.profiler.record_function("moe_forward"):
                    _do_forward()
        _sync()
        _, detailed_stats = _parse_profiler_events_moe(list(prof.events()))
    except Exception as exc:
        # Breakdown is best-effort; do not fail the case if Kineto misbehaves.
        _maybe_print_rank0(f"[bench_moe] kernel breakdown skipped: {type(exc).__name__}: {exc}")
    return forward_times_us, detailed_stats


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

    # ---- 1. Pre-capture eager pass for shape discovery and lazy initialization.
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
    for _ in range(warmup):
        big_graph.replay()
    _sync()

    if _cupti_available:
        _cupti.activity_flush_all(0)
        _cupti_kernels.clear()
        _cupti_events.clear()

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
        total = comm_us + gemm_us + routing_us
        if total <= 0:
            return None
        if comm_us / total > 0.5:
            return "communication_bound"
        if gemm_us / total > 0.5:
            return "compute_bound"
        if routing_us / total > 0.4:
            return "routing_bound"
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
    combine precision default). Search axes
    explicitly listed on ``search`` override the base values.
    """
    backends = _expand_axis(search.backends, base_config.backend)
    parallel_modes = _expand_axis(search.parallel_modes, base_config.parallel_mode)
    comm_methods = _expand_axis(search.comm_methods, base_config.comm_method)
    cuda_graph_options = _expand_axis(search.cuda_graph_options, base_config.cuda_graph)
    combine_options = _expand_axis(
        search.combine_precision_options, base_config.use_low_precision_moe_combine
    )

    candidates: List[ConfigSpec] = []
    for backend, pmode, comm, cgraph, combine in itertools.product(
        backends, parallel_modes, comm_methods, cuda_graph_options, combine_options
    ):
        candidate = replace(
            base_config,
            backend=str(backend).upper(),
            parallel_mode=str(pmode).upper(),
            comm_method=str(comm).upper(),
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
    """Return ``(ok, reason)`` based on backend / mapping / comm gates."""
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


def _build_routing_control_block(
    *,
    spec: RoutingControlSpec,
    plan: RoutingPlan,
    observed_slot: List[List[int]],
    observed_token: List[List[int]],
    observed_hist: List[List[int]],
    routing_path: Optional[str],
    realization_status: str,
    realization_reason: str,
    enable_perfect_router: bool,
    max_num_tokens_per_rank: int,
    num_chunks_observed: Optional[int],
    warnings: List[str],
    scale_dtype: torch.dtype,
    moe_ep_size: int,
    dump_full: bool,
    routing_mode: str,
) -> Dict[str, Any]:
    """Compose the ``routing_control`` block for a result row.

    Always includes ``requested`` and an ``actual`` summary; full slot/token
    matrices and histograms are included only when ``dump_full`` is set, to
    avoid JSON bloat during large sweeps.

    NOTE on ``observed_*`` fields: the dispatch / histogram numbers reported
    here are derived from a deterministic re-materialisation of the canonical
    ``RoutingPlan`` -- they describe *the plan the bench asked the kernel to
    realise*, not what the kernel actually emitted at runtime. The
    ``actual.observation_source`` field documents this. In ``forced`` mode the
    kernel is patched to consume the exact materialised top-k, so plan ==
    kernel output by construction. In ``native`` mode the kernel routes via
    projected logits and may produce slightly different top-k due to fp ties,
    quantisation, or projection-status='projected'; a warning is added so the
    consumer does not over-trust the slot/histogram numbers.
    """
    requested_slot = [list(row) for row in plan.dispatch_matrix]
    max_abs, max_rel = _observe_summary(requested_slot, observed_slot)
    row_sums = [sum(row) for row in observed_slot]
    col_sums = [sum(observed_slot[s][d] for s in range(moe_ep_size)) for d in range(moe_ep_size)]
    diag = sum(observed_slot[i][i] for i in range(moe_ep_size)) if moe_ep_size > 0 else 0
    total = sum(row_sums)
    off_diag_ratio = 0.0 if total <= 0 else (1.0 - diag / total)

    flat_hist = [v for row in observed_hist for v in row]
    hist_min = min(flat_hist) if flat_hist else 0
    hist_max = max(flat_hist) if flat_hist else 0
    active_experts = sum(1 for v in flat_hist if v > 0)

    warnings_out = list(warnings)
    if routing_mode != "forced":
        warnings_out.append(
            "observed_* fields are derived from RoutingPlan re-materialisation, "
            "not from the kernel's actual selected_experts output; in native mode "
            "the real top-k may differ from the plan."
        )

    block: Dict[str, Any] = {
        "requested": {
            "routing_mode": spec.routing_mode,
            "projection_policy": spec.projection_policy,
            "comm_pattern": spec.comm_pattern,
            "expert_pattern": spec.expert_pattern,
            "routing_pattern_file": spec.routing_pattern_file,
            "per_rank_num_tokens": list(plan.per_rank_num_tokens),
            "seed": int(spec.seed),
        },
        "actual": {
            "routing_path": routing_path,
            "routing_realization": {
                "status": realization_status,
                "reason": realization_reason,
                "max_abs_slot_error": int(max_abs),
                "max_relative_slot_error": float(max_rel),
            },
            "enable_perfect_router": bool(enable_perfect_router),
            "effective_src_axis": "dp_rank",
            "max_num_tokens_per_rank": int(max_num_tokens_per_rank),
            "num_chunks_observed": int(num_chunks_observed)
            if isinstance(num_chunks_observed, int)
            else None,
            "use_dp_padding": False,
            # "plan_exact": forced mode (kernel is patched to the materialised
            # plan, so observed == plan by construction).
            # "plan_simulation": native mode (numbers are a deterministic
            # re-materialisation of the plan, NOT the kernel's actual top-k).
            "observation_source": ("plan_exact" if routing_mode == "forced" else "plan_simulation"),
            "observed_dispatch_matrix_summary": {
                "row_sums": row_sums,
                "col_sums": col_sums,
                "off_diagonal_ratio": float(off_diag_ratio),
                "max_abs_slot_error": int(max_abs),
                "matrix_dump_path": None,
            },
            "observed_expert_histogram_summary": {
                "min": int(hist_min),
                "max": int(hist_max),
                "active_experts": int(active_experts),
            },
            "selected_scales": {
                "distribution": "uniform",
                "dtype": str(scale_dtype),
                "seed": int(spec.seed),
            },
            "warnings": warnings_out,
        },
    }
    if dump_full:
        block["actual"]["observed_slot_dispatch_matrix"] = [list(r) for r in observed_slot]
        block["actual"]["observed_token_dispatch_matrix"] = [list(r) for r in observed_token]
        block["actual"]["observed_expert_histogram"] = [list(r) for r in observed_hist]
        block["actual"]["requested_slot_dispatch_matrix"] = [list(r) for r in requested_slot]
    return block


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
) -> RunResult:
    """Build, autotune, and time one ``ConfigSpec`` candidate.

    Always returns a ``RunResult``; failures are encoded in the ``status`` /
    ``skip_reason`` fields so the caller can write a row even for the failed
    case.
    """
    result = RunResult(model=model, workload=workload, config=config)
    rc_spec = workload.routing_control
    rc_active = rc_spec.is_active

    # Resolve EP-axis ``moe_ep_size`` from the candidate now so routing-plan
    # building can use it. The mapping object created later will agree.
    try:
        moe_ep_size, _moe_tp_size, _enable_dp = _resolve_mapping_layout(config, world_size)
    except ValueError as exc:
        result.status = "skipped"
        result.skip_reason = str(exc)
        result.status_per_rank = _gather_status_per_rank("skipped")
        return result

    # Routing-control's dispatch_matrix axis is ``moe_ep_size`` while
    # ``per_rank_num_tokens`` follows the world (DP source) axis. When the two
    # axes disagree (DTP / TTP / CUSTOM with ``moe_ep_size != world_size``) the
    # plan would either crash inside ``_build_routing_plan`` (slot total
    # mismatch) or silently drop the tokens of world ranks beyond
    # ``moe_ep_size``. Skip cleanly so dashboards do not record meaningless
    # latencies for an unsupported combination.
    if rc_active and int(moe_ep_size) != int(world_size):
        reason = (
            f"routing-control requires moe_ep_size == world_size "
            f"(got moe_ep_size={moe_ep_size}, world_size={world_size}); "
            "the dispatch_matrix axis would not align with the per-rank token "
            "distribution. Use parallel_mode in {DEP, TEP} or drop routing-control."
        )
        result.status = "skipped"
        result.skip_reason = reason
        result.status_per_rank = _gather_status_per_rank("skipped")
        return result

    routing_plan: Optional[RoutingPlan] = None
    materialized_ids: Optional[torch.Tensor] = None
    materialized_scales: Optional[torch.Tensor] = None
    routing_realization_status: Optional[str] = None
    routing_realization_reason: Optional[str] = None
    routing_warnings: List[str] = []

    if rc_active:
        try:
            routing_plan = _build_routing_plan(
                rc_spec,
                num_tokens=int(workload.num_tokens),
                world_size=world_size,
                top_k=int(model.top_k),
                num_experts=int(model.num_experts),
                moe_ep_size=int(moe_ep_size),
            )
        except Exception as exc:
            reason = f"routing plan error: {type(exc).__name__}: {exc}"
            _maybe_print_rank0(f"[bench_moe] {reason}")
            result.status = "skipped"
            result.skip_reason = reason
            result.status_per_rank = _gather_status_per_rank("skipped")
            return result
        per_rank = list(routing_plan.per_rank_num_tokens)
        # Pad ``per_rank`` to ``world_size`` when ``moe_ep_size < world_size``;
        # MoE-replicated source ranks share the same EP-axis bucket.
        if len(per_rank) < world_size:
            base = per_rank + [0] * (world_size - len(per_rank))
            per_rank = base
    else:
        per_rank = _per_rank_tokens(workload, world_size)

    result.per_rank_num_tokens = list(per_rank)
    local_num_tokens = per_rank[rank] if rank < len(per_rank) else 0
    all_rank_num_tokens = list(per_rank)

    instrumentation: Dict[str, Any] = {
        "level": ",".join(sorted(analysis)) if analysis else "summary",
        "cuda_graph": bool(config.cuda_graph),
        "cupti_available": bool(cupti_ctx is not None and cupti_ctx[3]),
        "phase_timing_available": False,
        "kernel_breakdown_available": "kernels" in analysis,
        "autotune_status": "not_run",
        "latency_source": "cuda_event_external" if config.cuda_graph else "cuda_event_eager",
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

    # Force comm method via env var (per-case).
    prev_force_comm = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
    _force_comm_env(config.comm_method, prev_force_comm)

    # ``enable_perfect_router`` selection per design table.
    if rc_active:
        # ``forced`` always disables perfect router; ``native`` enables it
        # only for ``balanced_alltoall`` + ``balanced``.
        if rc_spec.routing_mode == "forced":
            enable_perfect_router = False
        else:
            enable_perfect_router = (
                rc_spec.comm_pattern == "balanced_alltoall" and rc_spec.expert_pattern == "balanced"
            )
    else:
        enable_perfect_router = True

    moe = None
    local_status = "success"
    try:
        try:
            moe, _ = _build_moe_module(
                model=model,
                config=config,
                mapping=mapping,
                moe_backend=config.backend,
                use_cuda_graph=bool(config.cuda_graph),
                max_num_tokens=max(int(local_num_tokens), 1),
                use_low_precision_moe_combine=bool(config.use_low_precision_moe_combine),
                enable_perfect_router=enable_perfect_router,
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

        routing_path: Optional[str] = None
        if rc_active and routing_plan is not None:
            experts_per_rank = int(model.num_experts) // int(moe_ep_size)
            ep_axis_rank = rank if rank < int(moe_ep_size) else (rank % int(moe_ep_size))
            if rc_spec.routing_mode == "forced":
                try:
                    materialized_ids, materialized_scales = _materialize_selected_experts_for_rank(
                        routing_plan,
                        src_rank=ep_axis_rank,
                        top_k=int(model.top_k),
                        experts_per_rank=experts_per_rank,
                        moe_ep_size=int(moe_ep_size),
                        device=device,
                        scale_dtype=act_dtype,
                    )
                except Exception as exc:
                    reason = f"routing materialise error: {type(exc).__name__}: {exc}"
                    _maybe_print_rank0(f"[bench_moe] {reason}")
                    result.status = "skipped"
                    result.skip_reason = reason
                    result.status_per_rank = _gather_status_per_rank("skipped")
                    return result
                routing_realization_status = "forced_exact"
                routing_realization_reason = (
                    "forced routing_mode: top-k ids and uniform 1/top_k scales materialised "
                    "from RoutingPlan; native fused scoring is intentionally bypassed"
                )
                inner_backend = getattr(moe, "backend", moe)
                if isinstance(inner_backend, TRTLLMGenFusedMoE):
                    routing_path = "supplied_topk_run_moe"
                else:
                    routing_path = "supplied_topk_apply"
            else:
                try:
                    new_logits, projection_status, projection_reason = (
                        _project_router_logits_for_plan(
                            routing_plan,
                            src_rank=ep_axis_rank,
                            routing_method=moe.routing_method,
                            num_experts=int(model.num_experts),
                            top_k=int(model.top_k),
                            experts_per_rank=experts_per_rank,
                            moe_ep_size=int(moe_ep_size),
                            device=device,
                            dtype=routing_logits_dtype,
                        )
                    )
                except Exception as exc:
                    reason = f"native logits projection error: {type(exc).__name__}: {exc}"
                    _maybe_print_rank0(f"[bench_moe] {reason}")
                    result.status = "skipped"
                    result.skip_reason = reason
                    result.status_per_rank = _gather_status_per_rank("skipped")
                    return result
                if projection_status != "exact" and rc_spec.projection_policy == "reject":
                    routing_realization_status = "rejected"
                    routing_realization_reason = projection_reason
                    result.status = "skipped"
                    result.skip_reason = (
                        f"routing_realization rejected by projection_policy=reject: "
                        f"{projection_reason}"
                    )
                    result.status_per_rank = _gather_status_per_rank("skipped")
                    # Still record routing-control summary so dashboards see the case.
                    result.routing_control = _build_routing_control_block(
                        spec=rc_spec,
                        plan=routing_plan,
                        observed_slot=[[0] * int(moe_ep_size) for _ in range(int(moe_ep_size))],
                        observed_token=[[0] * int(moe_ep_size) for _ in range(int(moe_ep_size))],
                        observed_hist=[[0] * experts_per_rank for _ in range(int(moe_ep_size))],
                        routing_path=None,
                        realization_status="rejected",
                        realization_reason=projection_reason,
                        enable_perfect_router=enable_perfect_router,
                        max_num_tokens_per_rank=max(per_rank) if per_rank else 0,
                        num_chunks_observed=result.num_chunks,
                        warnings=routing_warnings,
                        scale_dtype=act_dtype,
                        moe_ep_size=int(moe_ep_size),
                        dump_full=bool(rc_spec.routing_dump_matrix),
                        routing_mode=rc_spec.routing_mode,
                    )
                    return result
                router_logits = new_logits
                routing_realization_status = projection_status
                routing_realization_reason = projection_reason
                routing_path = (
                    "logits_native" if projection_status == "exact" else "logits_projected"
                )
                if projection_status != "exact":
                    routing_warnings.append(
                        f"routing_realization={projection_status}: {projection_reason}"
                    )

        with _maybe_install_routing_control_patch(
            moe,
            materialized_ids,
            materialized_scales,
            active=(rc_active and rc_spec.routing_mode == "forced"),
        ):
            try:
                autotune_status = _run_autotune(
                    moe,
                    x,
                    router_logits,
                    all_rank_num_tokens,
                    bool(fast_autotune),
                )
            except Exception as exc:
                autotune_status = f"failed:{type(exc).__name__}: {exc}"
                _maybe_print_rank0(f"[bench_moe] autotune skipped: {type(exc).__name__}: {exc}")
            result.instrumentation["autotune_status"] = autotune_status

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
                        collect_kernels="kernels" in analysis,
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

        if rc_active and routing_plan is not None:
            experts_per_rank = int(model.num_experts) // int(moe_ep_size)
            # Observation uses the canonical plan to materialise *every* EP
            # source independently. This avoids MPI gather double-counting in
            # mapping modes where multiple world ranks share the same EP rank
            # (e.g. DTP/TTP with moe_ep_size < world_size), and lets the rank-0
            # writer report consistent slot/token/histogram numbers regardless
            # of which world rank executed which EP axis.
            per_rank_ids: List[Any] = []
            for src in range(int(moe_ep_size)):
                try:
                    src_ids, _ = _materialize_selected_experts_for_rank(
                        routing_plan,
                        src_rank=src,
                        top_k=int(model.top_k),
                        experts_per_rank=experts_per_rank,
                        moe_ep_size=int(moe_ep_size),
                        device=torch.device("cpu"),
                        scale_dtype=torch.float32,
                    )
                except Exception:
                    src_ids = torch.empty((0, int(model.top_k)), dtype=torch.int32)
                per_rank_ids.append(src_ids)
            observed_slot, observed_token, observed_hist = _observe_routing_metrics(
                routing_plan, per_rank_ids, experts_per_rank, int(moe_ep_size)
            )
            result.routing_control = _build_routing_control_block(
                spec=rc_spec,
                plan=routing_plan,
                observed_slot=observed_slot,
                observed_token=observed_token,
                observed_hist=observed_hist,
                routing_path=routing_path,
                realization_status=routing_realization_status or "exact",
                realization_reason=routing_realization_reason or "balanced default",
                enable_perfect_router=enable_perfect_router,
                max_num_tokens_per_rank=max(per_rank) if per_rank else 0,
                num_chunks_observed=result.num_chunks,
                warnings=routing_warnings,
                scale_dtype=act_dtype,
                moe_ep_size=int(moe_ep_size),
                dump_full=bool(rc_spec.routing_dump_matrix),
                routing_mode=rc_spec.routing_mode,
            )

        result.status_per_rank = _gather_status_per_rank(local_status)
        return result
    finally:
        if moe is not None:
            try:
                moe.destroy()
            except Exception:
                pass
        # Restore TRTLLM_FORCE_COMM_METHOD.
        if prev_force_comm is None:
            os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
        else:
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = prev_force_comm


# ---------------------------------------------------------------------------
# Output schema serialization
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
        "routing_control": result.routing_control or None,
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
            autotune_status = (
                r.instrumentation.get("autotune_status") if r.instrumentation else None
            )
            ranking_entries.append(
                {
                    "backend": r.actual_backend or r.config.backend,
                    "requested_backend": r.config.backend,
                    "comm_method": r.actual_comm_method,
                    "cuda_graph": r.config.cuda_graph,
                    "use_low_precision_moe_combine": r.config.use_low_precision_moe_combine,
                    "score_us": float(score) if isinstance(score, (int, float)) else None,
                    "status": r.status,
                    "skip_reason": r.skip_reason,
                    "autotune_status": autotune_status,
                }
            )
        ranking_entries.sort(
            key=lambda e: (
                e["score_us"] is None,
                e["score_us"] if e["score_us"] is not None else 0.0,
            )
        )
        # ``best`` excludes cases whose autotune pass failed: their score is
        # likely a not-yet-tuned slow path and would mislead the
        # "which config is fastest" question.
        best = next(
            (
                e
                for e in ranking_entries
                if e["score_us"] is not None
                and e["status"] == "success"
                and not (
                    isinstance(e.get("autotune_status"), str)
                    and e["autotune_status"].startswith("failed")
                )
            ),
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


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Search and sweep")
    group.add_argument(
        "--search",
        type=lambda s: str(s).lower(),
        nargs="+",
        default=("none",),
        help=(
            "Expand one or more runtime axes. Examples: --search backend --backend ALL; "
            "--search backend comm; --search full. Comma-separated input is also accepted."
        ),
    )
    group.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Run at most this many valid candidate configs after pruning. Example: --max_configs 32.",
    )
    group.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Stop launching new candidates after this wall-clock budget. Example: --time_budget_minutes 30.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE module microbenchmark (MPI). Times ConfigurableMoE.forward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    launch_group = parser.add_argument_group("Launch")
    launch_group.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of MPI worker ranks to spawn (ignored under external mpirun).",
    )

    model_group = parser.add_argument_group("Model and shape")
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(BUILT_IN_MODELS.keys()),
        help=(
            "Built-in model shape. Examples: deepseek_v3, qwen1.5_moe. "
            "Omit only when passing all custom shape fields below."
        ),
    )
    model_group.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Custom-shape total expert count. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Custom-shape experts selected per token. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Custom-shape hidden size. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="Custom-shape MoE intermediate size. Required when --model is omitted.",
    )
    model_group.add_argument(
        "--n_group",
        type=int,
        default=None,
        help="DeepSeek-style routing group count for custom grouped routing.",
    )
    model_group.add_argument(
        "--topk_group",
        type=int,
        default=None,
        help="DeepSeek-style number of routing groups kept per token.",
    )
    model_group.add_argument(
        "--quant",
        type=lambda s: QuantAlgo[str(s).upper()] if s is not None else None,
        default=None,
        choices=[q.name for q in QuantAlgo],
        help="Quantization algorithm. Example: --quant FP8_BLOCK_SCALES.",
    )
    model_group.add_argument(
        "--routing_method",
        type=lambda s: str(s).upper(),
        default="AUTO",
        choices=sorted(_ROUTING_METHODS) + ["AUTO"],
        help=(
            "Routing method. Defaults to AUTO: built-in models use the spec "
            "default; custom shapes must specify an explicit method."
        ),
    )

    workload_group = parser.add_argument_group("Workload shape")
    workload_group.add_argument(
        "--balanced_total_num_tokens",
        "--num_tokens",
        dest="balanced_total_num_tokens",
        type=int,
        nargs="+",
        required=False,
        help=(
            "Global token counts to sweep. Each value is balanced across ranks "
            "with any remainder on rank 0. Example: --balanced_total_num_tokens 64 256 1024."
        ),
    )

    routing_group = parser.add_argument_group("Routing control")
    routing_group.add_argument(
        "--routing_mode",
        type=lambda s: str(s).lower(),
        default="native",
        choices=("native", "forced"),
        help=(
            "native: route through production logits kernels (default); "
            "forced: supply top-k ids/scales directly (skips fused scoring)."
        ),
    )
    routing_group.add_argument(
        "--projection_policy",
        type=lambda s: str(s).lower(),
        default="project",
        choices=("project", "reject"),
        help=(
            "project: when native logits cannot exactly realise the plan, run with "
            "the closest legal projection and warn; reject: skip the case instead."
        ),
    )
    routing_group.add_argument(
        "--comm_pattern",
        type=str,
        default="balanced_alltoall",
        help=(
            "Source-to-target slot dispatch pattern. Examples: balanced_alltoall, "
            "receiver_hotspot,hotness=0.75,rank=0, pair_hotspot,hotness=0.5,src=0,dst=1, "
            "local_only, ring."
        ),
    )
    routing_group.add_argument(
        "--expert_pattern",
        type=str,
        default="balanced",
        help=(
            "Per-target-rank local expert histogram pattern. Examples: balanced, "
            "hotspot,hotness=0.5, hotspot,active_experts=2."
        ),
    )
    routing_group.add_argument(
        "--routing_pattern_file",
        type=str,
        default=None,
        help="JSON file that provides both slot_dispatch_matrix and expert_histogram.",
    )
    routing_group.add_argument(
        "--per_rank_num_tokens",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit per-rank input token counts. Length must equal world_size "
            "and sum defines the workload total. Mutually exclusive with --balanced_total_num_tokens."
        ),
    )
    routing_group.add_argument(
        "--routing_dump_matrix",
        action="store_true",
        help="Include the full observed slot/token matrix and expert histogram in each result row.",
    )
    routing_group.add_argument(
        "--routing_seed",
        type=int,
        default=0,
        help="Seed for deterministic routing-plan materialisation; independent from --random_seed.",
    )

    parallel_group = parser.add_argument_group("Parallel layout")
    parallel_group.add_argument(
        "--parallel_mode",
        type=str,
        default="DEP",
        choices=("DEP", "TEP", "DTP", "TTP", "CUSTOM"),
        help=(
            "Parallel layout. DEP=attention DP + MoE EP; TEP=attention TP + MoE EP; "
            "DTP/TTP use MoE TP; CUSTOM requires --moe_ep_size and --moe_tp_size."
        ),
    )
    parallel_group.add_argument(
        "--moe_ep_size",
        type=int,
        default=None,
        help="CUSTOM only: MoE expert-parallel size. Must multiply with --moe_tp_size to world_size.",
    )
    parallel_group.add_argument(
        "--moe_tp_size",
        type=int,
        default=None,
        help="CUSTOM only: MoE tensor-parallel size. Must multiply with --moe_ep_size to world_size.",
    )
    parallel_group.add_argument(
        "--enable_attention_dp",
        action="store_true",
        help="CUSTOM only: enable attention data parallelism for the mapping.",
    )

    runtime_group = parser.add_argument_group("Runtime backend and communication")
    runtime_group.add_argument(
        "--backend",
        type=lambda s: str(s).upper(),
        default="TRTLLM",
        choices=_ALL_BACKENDS + ["ALL"],
        help=(
            "MoE backend to benchmark. Use --search backend --backend ALL to scan all "
            "ConfigurableMoE-eligible backends."
        ),
    )
    runtime_group.add_argument(
        "--comm_method",
        type=lambda s: str(s).upper(),
        default="AUTO",
        choices=_COMM_METHODS,
        help="Communication method. AUTO lets TensorRT-LLM select; other values force a specific path.",
    )

    timing_group = parser.add_argument_group("Timing")
    timing_group.add_argument(
        "--no_cuda_graph",
        dest="cuda_graph",
        action="store_false",
        default=True,
        help="Disable CUDA-Graph capture and use eager timing.",
    )
    timing_group.add_argument("--warmup", type=int, default=1, help="Warmup iterations per case.")
    timing_group.add_argument("--iters", type=int, default=5, help="Timed iterations per case.")
    timing_group.add_argument(
        "--fast_autotune",
        action="store_true",
        help="Use a short autotune pass for smoke tests; may reduce measurement quality.",
    )
    timing_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16"),
        help="Activation dtype for synthetic inputs.",
    )
    timing_group.add_argument(
        "--use_low_precision_moe_combine",
        action="store_true",
        help="Use low-precision combine where the selected backend supports it.",
    )
    timing_group.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="Seed for synthetic hidden states/router logits; routing-control plans use --routing_seed.",
    )

    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--analysis",
        nargs="+",
        default=("kernels",),
        choices=("none", "kernels"),
        help="Analysis data to collect. Use --analysis none for latency-only output.",
    )

    # ---- Search / sweep ----
    _add_search_arguments(parser)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help="JSON config file. CLI flags override matching config-file fields.",
    )
    output_group.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Write the final dashboard JSON report to this path.",
    )
    args = parser.parse_args()
    provided = set()
    argv = sys.argv[1:]
    for action in parser._actions:
        for option in action.option_strings:
            if option in argv or any(arg.startswith(option + "=") for arg in argv):
                provided.add(action.dest)
    args._cli_provided = provided
    args.search = _parse_search_axes(args.search)
    return args


# ---------------------------------------------------------------------------
# Spec resolution from CLI args / config file
# ---------------------------------------------------------------------------


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
    balanced_total_num_tokens = getattr(args, "balanced_total_num_tokens", None)
    if balanced_total_num_tokens is None:
        balanced_total_num_tokens = getattr(args, "num_tokens", None)

    # Resolve RoutingControlSpec from explicit CLI/config fields.
    per_rank_num_tokens: Optional[Tuple[int, ...]] = None
    if getattr(args, "per_rank_num_tokens", None):
        if balanced_total_num_tokens:
            raise ValueError(
                "--balanced_total_num_tokens and --per_rank_num_tokens are mutually exclusive"
            )
        raw = args.per_rank_num_tokens
        if isinstance(raw, str):
            try:
                parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
            except ValueError as exc:
                raise ValueError(f"--per_rank_num_tokens must be integers; got {raw!r}") from exc
        else:
            parts = [int(v) for v in raw]
        per_rank_num_tokens = tuple(parts)
        if any(v < 0 for v in per_rank_num_tokens):
            raise ValueError("--per_rank_num_tokens entries must be >= 0")
        token_values = [sum(per_rank_num_tokens)]
    else:
        if not balanced_total_num_tokens:
            raise ValueError(
                "--balanced_total_num_tokens or --per_rank_num_tokens is required "
                "(or supply via --config_file)"
            )
        token_values = [int(t) for t in balanced_total_num_tokens]
        if any(t < 0 for t in token_values):
            raise ValueError("--balanced_total_num_tokens entries must be >= 0")

    routing_spec = RoutingControlSpec(
        routing_mode=str(getattr(args, "routing_mode", "native")),
        projection_policy=str(getattr(args, "projection_policy", "project")),
        comm_pattern=str(getattr(args, "comm_pattern", "balanced_alltoall")),
        expert_pattern=str(getattr(args, "expert_pattern", "balanced")),
        routing_pattern_file=getattr(args, "routing_pattern_file", None),
        per_rank_num_tokens=per_rank_num_tokens,
        routing_dump_matrix=bool(getattr(args, "routing_dump_matrix", False)),
        seed=int(getattr(args, "routing_seed", 0)),
    )

    return [
        WorkloadSpec(
            num_tokens=int(t),
            routing_control=routing_spec,
        )
        for t in token_values
    ]


def _resolve_base_config_from_args(args: argparse.Namespace) -> ConfigSpec:
    comm_method = args.comm_method

    backend = args.backend
    if backend == "ALL":
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
        cuda_graph=bool(args.cuda_graph),
        use_low_precision_moe_combine=bool(args.use_low_precision_moe_combine),
    )


_SEARCH_AXES = ("backend", "comm", "parallel")
_SEARCH_MODES = ("none",) + _SEARCH_AXES + ("full",)


def _parse_search_axes(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [
            part.strip().lower() for part in str(value).replace(",", " ").split() if part.strip()
        ]
    if not parts:
        return ("none",)

    out: List[str] = []
    for part in parts:
        if part not in _SEARCH_MODES:
            raise ValueError(f"unknown --search axis {part!r}; valid: {list(_SEARCH_MODES)}")
        if part not in out:
            out.append(part)

    if "none" in out and len(out) > 1:
        raise ValueError("--search none cannot be combined with other axes")
    if "full" in out and len(out) > 1:
        raise ValueError("--search full cannot be combined with other axes")
    return tuple(out)


def _resolve_search_from_args(args: argparse.Namespace, base_config: ConfigSpec) -> SearchSpec:
    search_axes = _parse_search_axes(args.search)
    backends_arg = args.backend
    config_axes = getattr(args, "_config_search_axes", {}) or {}

    backends: Tuple[str, ...] = ()
    parallel_modes: Tuple[str, ...] = ()
    comm_methods: Tuple[str, ...] = ()
    cuda_graph_options: Tuple[bool, ...] = ()
    combine_options: Tuple[bool, ...] = ()

    if search_axes == ("none",) and backends_arg == "ALL":
        search_axes = ("backend",)

    if search_axes == ("none",):
        return SearchSpec(mode="none")

    full_search = search_axes == ("full",)
    enabled_axes = set(_SEARCH_AXES if full_search else search_axes)
    mode = "full" if full_search else ",".join(search_axes)

    if "backend" in enabled_axes:
        if "backend" in config_axes:
            backends = tuple(config_axes["backend"])
        elif backends_arg == "ALL":
            backends = tuple(_ALL_BACKENDS)
        else:
            backends = (backends_arg,)
    if "parallel" in enabled_axes:
        parallel_modes = tuple(config_axes.get("parallel_mode", ("DEP", "TEP", "DTP", "TTP")))
    if "comm" in enabled_axes:
        comm_methods = tuple(
            config_axes.get("comm_method", tuple(_FORCED_COMM_ENV_VALUES.keys()) + ("AUTO",))
        )
    # ``cuda_graph`` and ``combine_precision`` axes are reserved for ``full``;
    # leave them empty by default so the base value is used.
    if full_search:
        cuda_graph_options = (True, False)

    return SearchSpec(
        mode=mode,
        backends=backends,
        parallel_modes=parallel_modes,
        comm_methods=comm_methods,
        cuda_graph_options=cuda_graph_options,
        combine_precision_options=combine_options,
    )


def _parse_analysis(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [str(p).strip().lower() for p in value if str(p).strip()]
    else:
        parts = [p.strip().lower() for p in str(value).replace(",", " ").split() if p.strip()]
    valid = {"none", "kernels"}
    out: List[str] = []
    for p in parts:
        if p not in valid:
            raise ValueError(f"unknown --analysis dimension {p!r}; valid: {sorted(valid)}")
        if p == "none":
            continue
        if p not in out:
            out.append(p)
    return tuple(out)


def _maybe_load_config_file(args: argparse.Namespace) -> argparse.Namespace:
    """Overlay ``--config_file`` JSON onto ``args``; explicit CLI flags win."""
    if not args.config_file:
        args._config_search_axes = {}
        return args
    with open(args.config_file) as f:
        cfg = json.load(f)

    provided = set(getattr(args, "_cli_provided", set()))

    def set_if_unset(dest: str, value: Any) -> None:
        if dest not in provided:
            setattr(args, dest, value)

    if "model" in cfg:
        set_if_unset("model", cfg["model"])
    workload_cfg = cfg.get("workload", {}) or {}
    if "balanced_total_num_tokens" in workload_cfg:
        set_if_unset("balanced_total_num_tokens", list(workload_cfg["balanced_total_num_tokens"]))
    elif "num_tokens" in workload_cfg:
        set_if_unset("balanced_total_num_tokens", list(workload_cfg["num_tokens"]))
    if "per_rank_num_tokens" in workload_cfg:
        prnt = workload_cfg["per_rank_num_tokens"]
        set_if_unset(
            "per_rank_num_tokens",
            [int(v) for v in prnt] if isinstance(prnt, list) else str(prnt),
        )
    routing_cfg = workload_cfg.get("routing_control", {}) or {}
    if "routing_mode" in routing_cfg:
        set_if_unset("routing_mode", str(routing_cfg["routing_mode"]).lower())
    if "projection_policy" in routing_cfg:
        set_if_unset("projection_policy", str(routing_cfg["projection_policy"]).lower())
    if "comm_pattern" in routing_cfg:
        set_if_unset("comm_pattern", str(routing_cfg["comm_pattern"]))
    if "expert_pattern" in routing_cfg:
        set_if_unset("expert_pattern", str(routing_cfg["expert_pattern"]))
    if "routing_pattern_file" in routing_cfg:
        set_if_unset("routing_pattern_file", str(routing_cfg["routing_pattern_file"]))
    if "routing_dump_matrix" in routing_cfg:
        set_if_unset("routing_dump_matrix", bool(routing_cfg["routing_dump_matrix"]))
    if "seed" in routing_cfg:
        set_if_unset("routing_seed", int(routing_cfg["seed"]))
    search_cfg = cfg.get("search", {}) or {}
    unsupported_search_keys = set(search_cfg) - {"backend", "parallel_mode", "comm_method"}
    if unsupported_search_keys:
        raise ValueError(f"unsupported search key(s): {sorted(unsupported_search_keys)}")
    config_search_axes: Dict[str, Tuple[str, ...]] = {}

    def normalize_search_axis(key: str, value: Any) -> Optional[Tuple[str, ...]]:
        if key in provided:
            return None
        if key == "backend":
            if value == "ALL":
                set_if_unset("backend", "ALL")
                return None
            values = (
                tuple(str(v).upper() for v in value)
                if isinstance(value, list)
                else (str(value).upper(),)
            )
            if len(values) == 1:
                if values[0] == "ALL":
                    set_if_unset("backend", "ALL")
                    return None
                set_if_unset("backend", values[0])
            return values
        if key == "parallel_mode":
            values = (
                tuple(str(v).upper() for v in value)
                if isinstance(value, list)
                else (str(value).upper(),)
            )
            if len(values) == 1:
                set_if_unset("parallel_mode", values[0])
            return values
        if key == "comm_method":
            values = (
                tuple(str(v).upper() for v in value)
                if isinstance(value, list)
                else (str(value).upper(),)
            )
            if len(values) == 1:
                set_if_unset("comm_method", values[0])
            return values
        return None

    if search_cfg:
        for key in ("backend", "parallel_mode", "comm_method"):
            if key in search_cfg:
                axis = normalize_search_axis(key, search_cfg[key])
                if axis:
                    config_search_axes[key] = axis
        if "search" not in provided:
            axes = []
            if "backend" in search_cfg and "backend" not in provided:
                axes.append("backend")
            if "parallel_mode" in search_cfg and "parallel_mode" not in provided:
                axes.append("parallel")
            if "comm_method" in search_cfg and "comm_method" not in provided:
                axes.append("comm")
            if axes:
                args.search = tuple(axes)
    args._config_search_axes = config_search_axes
    if "analysis" in cfg:
        set_if_unset("analysis", cfg["analysis"])
    if "max_configs" in cfg:
        set_if_unset("max_configs", int(cfg["max_configs"]))
    if "time_budget_minutes" in cfg:
        set_if_unset("time_budget_minutes", float(cfg["time_budget_minutes"]))
    if "output_file" in cfg:
        set_if_unset("output_file", cfg["output_file"])
    return args


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_benchmark_worker_under_current_mpi(args: argparse.Namespace, launcher: str) -> None:
    args = _maybe_load_config_file(args)
    analysis = _parse_analysis(args.analysis)

    # CUPTI MUST be initialized before the CUDA context is created.
    _early_cupti_ctx: Optional[Any] = None
    if args.cuda_graph and "kernels" in analysis:
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

    model = _resolve_model_from_args(args)
    workloads = _resolve_workloads_from_args(args)
    base_config = _resolve_base_config_from_args(args)
    search = _resolve_search_from_args(args, base_config)

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
                f"comm={cand.comm_method} num_tokens={workload.num_tokens}"
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
