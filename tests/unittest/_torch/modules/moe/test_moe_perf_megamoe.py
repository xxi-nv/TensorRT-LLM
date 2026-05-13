# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MoE module forward-perf benchmark on a fixed multi-GPU DEP=4 setup.

Compares ``ConfigurableMoE.forward`` latency under CUDA Graph replay across
three MoE backends on the DeepSeek-V4-Flash MoE shape:
    - MEGAMOE_CUTEDSL  + NVFP4
    - MEGAMOE          + W4A8_MXFP4_MXFP8 (a.k.a. MegaMoEDeepGemm "MXFP4")
    - CUTEDSL          + NVFP4            (FUSEDMOE_CUTEDSL)

For each backend the worker module is instantiated once and CUDA Graphs are
captured per ``num_tokens`` value. ``num_tokens`` is interpreted as the
per-rank sequence length (matches the convention in ``test_moe_module.py``,
where ``all_rank_num_tokens = [seq_len] * world_size``).

Layout follows ``test_moe_module.py`` so all helper functions
(``_create_mapping_for_parallel_mode``, ``_create_routing_method``,
``_create_model_config``, ``_ensure_dist_for_megamoe``,
``MEGAMOE_CUTEDSL_IGNORE_COMM_METHOD``) are reused directly.

Result emission:
    - JSON written to $TRTLLM_MOE_PERF_RESULT_JSON (default
      ``moe_perf_megamoe_v4flash.json`` in CWD).
    - Human-readable summary printed at the end of the test.

Invocation:
    pytest -s tests/unittest/_torch/modules/moe/test_moe_perf_megamoe.py
"""

from __future__ import annotations

import json
import os
import pickle
import socket
import statistics
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import cloudpickle
import pytest
import torch
from _torch.modules.moe.moe_test_utils import (
    MoeBackendType,
    MoeModelConfig,
    should_skip_cutedsl,
    should_skip_megamoe,
    should_skip_megamoe_cutedsl,
)
from _torch.modules.moe.quantize_utils import get_test_quant_params
from _torch.modules.moe.test_moe_module import (
    MEGAMOE_CUTEDSL_IGNORE_COMM_METHOD,
    _create_mapping_for_parallel_mode,
    _create_model_config,
    _create_moe_load_balancer,
    _create_routing_method,
    _ensure_dist_for_megamoe,
    _get_free_tcp_port,
)
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.models.modeling_utils import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WORLD_SIZE = 4
PARALLEL_MODE = "DEP"
DTYPE = torch.bfloat16
ROUTING_METHOD_CLS = RenormalizeMoeRoutingMethod  # default routing
DEEPSEEK_V4_FLASH = MoeModelConfig(256, 6, 4096, 2048)

NUM_TOKENS_LIST: List[int] = [4, 64, 256, 512, 1024, 4096]

WARMUP_ITERS = 5
TIMED_ITERS = 20

# (backend_value, quant_algo, label) — MEGAMOE_CUTEDSL last so an early
# CUDA context corruption (e.g. workspace bug at large max_num_tokens) does
# not block the other two backends from producing data.
BACKEND_CASES: List[Tuple[str, Optional[QuantAlgo], str]] = [
    (MoeBackendType.MEGAMOE.value, QuantAlgo.W4A8_MXFP4_MXFP8, "MEGAMOE_DEEPGEMM_MXFP4"),
    (MoeBackendType.CUTEDSL.value, QuantAlgo.NVFP4, "FUSEDMOE_CUTEDSL_NVFP4"),
    (MoeBackendType.MEGAMOE_CUTEDSL.value, QuantAlgo.NVFP4, "MEGAMOE_CUTEDSL_NVFP4"),
]

RESULT_JSON_PATH = os.environ.get("TRTLLM_MOE_PERF_RESULT_JSON", "moe_perf_megamoe_v4flash.json")


# -----------------------------------------------------------------------------
# Per-rank perf worker
# -----------------------------------------------------------------------------


def _perf_worker(
    moe_backend: str,
    quant_algo: Optional[QuantAlgo],
    mapping,
    model_config: MoeModelConfig,
    num_tokens_list: List[int],
    max_num_tokens: int,
    warmup_iters: int,
    timed_iters: int,
    routing_method_cls=ROUTING_METHOD_CLS,
    dtype: torch.dtype = DTYPE,
):
    """Per-rank: setup module once, capture+time CUDA graph per num_tokens.

    Returns a dict ``{num_tokens: List[float] (per-iter latency in ms)}``.
    """

    try:
        return _perf_worker_impl(
            moe_backend=moe_backend,
            quant_algo=quant_algo,
            mapping=mapping,
            model_config=model_config,
            num_tokens_list=num_tokens_list,
            max_num_tokens=max_num_tokens,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            routing_method_cls=routing_method_cls,
            dtype=dtype,
        )
    except Exception:
        traceback.print_exc()
        raise


def _perf_worker_impl(
    moe_backend: str,
    quant_algo: Optional[QuantAlgo],
    mapping,
    model_config: MoeModelConfig,
    num_tokens_list: List[int],
    max_num_tokens: int,
    warmup_iters: int,
    timed_iters: int,
    routing_method_cls,
    dtype: torch.dtype,
):
    num_experts = model_config.num_experts
    top_k = model_config.top_k
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)
    _ensure_dist_for_megamoe(moe_backend, mapping.rank, mapping.world_size)

    # DeepSeekV3 routing on TRTLLM backend would need fp32 logits, but we test
    # only Renormalize / DeepSeekV3-on-non-TRTLLM here, so dtype matches model.
    dtype_routing_logits = dtype

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        routing_method = _create_routing_method(
            routing_method_cls,
            top_k=top_k,
            num_experts=num_experts,
            dtype=dtype,
            model_config=model_config,
        )

        # Initial probe tensors used for weight prep (max-shape is enough).
        probe_x = torch.randn((max_num_tokens, hidden_size), dtype=dtype, device="cuda")

        backend_type = MoeBackendType(moe_backend)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, probe_x, backend_type
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
        quant_kwargs.pop("ref_cls", None)

        model_cfg = _create_model_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=moe_backend,
            enable_eplb=False,
            num_slots=-1,
            layer_updates_per_iter=-1,
            max_num_tokens=max_num_tokens,
        )
        moe_load_balancer = _create_moe_load_balancer(model_cfg, enable_eplb=False)
        weight_loading_mode = getattr(
            quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
        )

        with (
            moe_load_balancer,
            create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_cfg,
                bias=False,
                weight_loading_mode=weight_loading_mode,
            ) as fused_moe,
        ):
            if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
                weights, _ref_weights, _ = quantize_util.prepare_weights_from_backend(
                    fused_moe, **quant_kwargs
                )
            else:
                weights = quantize_util.create_weights(**quant_kwargs)

            fused_moe.load_weights([weights])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{mapping.rank}")

            results: Dict[int, List[float]] = {}
            errors: Dict[int, str] = {}
            for num_tokens in num_tokens_list:
                try:
                    results[num_tokens] = _time_forward_cuda_graph(
                        fused_moe=fused_moe,
                        num_tokens=num_tokens,
                        hidden_size=hidden_size,
                        num_experts=num_experts,
                        world_size=mapping.world_size,
                        rank=mapping.rank,
                        dtype=dtype,
                        dtype_routing_logits=dtype_routing_logits,
                        warmup_iters=warmup_iters,
                        timed_iters=timed_iters,
                    )
                except Exception as exc:  # noqa: BLE001 — keep partial results
                    err_str = f"{type(exc).__name__}: {exc}"
                    errors[num_tokens] = err_str
                    results[num_tokens] = []
                    # CUDA illegal-memory errors poison the context, so
                    # all later num_tokens would also fail. Exit early to
                    # avoid spamming the same traceback.
                    if "CUDA error" in err_str or isinstance(exc, torch.AcceleratorError):
                        for remaining in num_tokens_list[num_tokens_list.index(num_tokens) + 1 :]:
                            errors[remaining] = "skipped: prior CUDA error poisoned context"
                            results[remaining] = []
                        break

            return {
                "rank": mapping.rank,
                "world_size": mapping.world_size,
                "moe_backend": moe_backend,
                "quant_algo": str(quant_algo) if quant_algo is not None else None,
                "model_config": str(model_config),
                "results": results,
                "errors": errors,
            }


def _time_forward_cuda_graph(
    fused_moe,
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    dtype_routing_logits: torch.dtype,
    warmup_iters: int,
    timed_iters: int,
) -> List[float]:
    """Capture one CUDA graph with one forward call, then warmup+timed replay.

    Each replay is timed with a CUDA Event pair to give per-iteration latency.
    """
    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    router_logits = torch.randn(
        (num_tokens, num_experts), dtype=dtype_routing_logits, device="cuda"
    )
    all_rank_num_tokens = [num_tokens] * world_size

    # Eager warmup: lets the autotuner / kernel-cache settle before capture.
    with torch.inference_mode():
        for _ in range(2):
            fused_moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
    torch.cuda.synchronize()

    # Capture one forward into a CUDA graph.
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        fused_moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)

    # Warmup replays.
    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    # Timed replays with per-iter event pairs (replay-then-record cadence).
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    for i in range(timed_iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()

    return [starts[i].elapsed_time(ends[i]) for i in range(timed_iters)]


# -----------------------------------------------------------------------------
# Top-level perf entry — fans out per backend to MPIPoolExecutor(world=4)
# -----------------------------------------------------------------------------


def _init_worker(custom_paths, comm_method_type: str, master_port: int) -> None:
    for custom_path in custom_paths:
        if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
            sys.path.append(custom_path)

    if comm_method_type == MEGAMOE_CUTEDSL_IGNORE_COMM_METHOD:
        os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    else:
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = comm_method_type
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)


def _run_one_backend(
    moe_backend: str,
    quant_algo: Optional[QuantAlgo],
    label: str,
) -> Dict:
    """Run one backend across all NUM_TOKENS_LIST values.

    For MEGAMOE_CUTEDSL we recreate the MoE module per num_tokens with
    ``max_num_tokens = that num_tokens``. The fused mega kernel has a known
    correctness issue when ``max_num_tokens`` is significantly larger than
    the actual local token count (the staging-loop OOB read is one site;
    there is at least one more site we have not located yet — bisect shows
    max_num_tokens=4096 fails even after the staging-loop fix). Sizing the
    workspace exactly to ``num_tokens`` sidesteps both issues.

    For DEEPGEMM / CUTEDSL backends the single-module sweep is fine.
    """
    mapping = _create_mapping_for_parallel_mode(WORLD_SIZE, PARALLEL_MODE)
    master_port = _get_free_tcp_port()

    is_mcd = moe_backend == MoeBackendType.MEGAMOE_CUTEDSL.value
    if is_mcd:
        # One MoE-module instance per num_tokens; max_num_tokens == num_tokens.
        token_groups = [[n] for n in NUM_TOKENS_LIST]
    else:
        # Single module with max_num_tokens covering the entire sweep.
        token_groups = [list(NUM_TOKENS_LIST)]

    aggregated_per_rank: Dict[int, Dict] = {}  # rank -> partial results dict

    with MPIPoolExecutor(
        initializer=_init_worker,
        initargs=(sys.path, MEGAMOE_CUTEDSL_IGNORE_COMM_METHOD, master_port),
        max_workers=WORLD_SIZE,
    ) as executor:
        for group in token_groups:
            args_per_rank = (
                moe_backend,
                quant_algo,
                mapping,
                DEEPSEEK_V4_FLASH,
                group,
                max(group),
                WARMUP_ITERS,
                TIMED_ITERS,
                ROUTING_METHOD_CLS,
                DTYPE,
            )
            futures = [executor.submit(_perf_worker, *args_per_rank) for _ in range(WORLD_SIZE)]
            for idx, fut in enumerate(futures):
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = {
                        "rank": idx,
                        "world_size": WORLD_SIZE,
                        "moe_backend": moe_backend,
                        "quant_algo": str(quant_algo) if quant_algo is not None else None,
                        "model_config": str(DEEPSEEK_V4_FLASH),
                        "results": {n: [] for n in group},
                        "errors": {n: f"{type(exc).__name__}: {exc}" for n in group},
                        "fatal": True,
                    }
                # Merge per-num_tokens results into the rank's accumulator.
                rank_id = res.get("rank", idx)
                acc = aggregated_per_rank.setdefault(
                    rank_id,
                    {
                        "rank": rank_id,
                        "world_size": WORLD_SIZE,
                        "moe_backend": moe_backend,
                        "quant_algo": str(quant_algo) if quant_algo is not None else None,
                        "model_config": str(DEEPSEEK_V4_FLASH),
                        "results": {},
                        "errors": {},
                    },
                )
                acc["results"].update(res.get("results", {}))
                acc["errors"].update(res.get("errors", {}))

    results = list(aggregated_per_rank.values())
    rank_results = sorted(results, key=lambda r: r["rank"])
    return {
        "label": label,
        "moe_backend": moe_backend,
        "quant_algo": str(quant_algo) if quant_algo is not None else None,
        "model_config": str(DEEPSEEK_V4_FLASH),
        "world_size": WORLD_SIZE,
        "parallel_mode": PARALLEL_MODE,
        "ranks": rank_results,
    }


def _aggregate_per_num_tokens(
    rank_results: List[Dict],
) -> Dict[int, Dict[str, float]]:
    """Reduce per-rank lists of latencies to ``rank-max(median, mean)``.

    Skips ranks / shapes whose timing list is empty (e.g. a CUDA error
    poisoned the rank's context for that num_tokens).
    """
    out: Dict[int, Dict[str, float]] = {}
    for num_tokens in NUM_TOKENS_LIST:
        per_rank_means: List[float] = []
        per_rank_medians: List[float] = []
        per_rank_errors: List[str] = []
        for r in rank_results:
            iters = r["results"].get(num_tokens, [])
            err = r.get("errors", {}).get(num_tokens)
            if err:
                per_rank_errors.append(f"rank{r['rank']}: {err}")
            if iters:
                per_rank_means.append(statistics.fmean(iters))
                per_rank_medians.append(statistics.median(iters))

        if per_rank_means:
            out[num_tokens] = {
                "rank_max_mean_ms": max(per_rank_means),
                "rank_max_median_ms": max(per_rank_medians),
                "rank_avg_mean_ms": statistics.fmean(per_rank_means),
                "per_rank_mean_ms": per_rank_means,
                "per_rank_median_ms": per_rank_medians,
                "errors": per_rank_errors,
            }
        else:
            out[num_tokens] = {
                "rank_max_mean_ms": float("nan"),
                "rank_max_median_ms": float("nan"),
                "rank_avg_mean_ms": float("nan"),
                "per_rank_mean_ms": [],
                "per_rank_median_ms": [],
                "errors": per_rank_errors or ["no data"],
            }
    return out


def _format_table(all_backend_results: List[Dict]) -> str:
    """Render a code-block-friendly text table comparing the three backends."""
    headers = ["num_tokens"] + [r["label"] + " (ms)" for r in all_backend_results]
    col_widths = [max(len(h), 12) for h in headers]

    rows: List[List[str]] = []
    for num_tokens in NUM_TOKENS_LIST:
        row = [str(num_tokens)]
        for r in all_backend_results:
            agg = r["aggregated"][num_tokens]
            median = agg["rank_max_median_ms"]
            mean = agg["rank_max_mean_ms"]
            if median != median:  # NaN
                row.append("ERROR")
            else:
                row.append(f"{median:.4f} (mean {mean:.4f})")
        rows.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt(cells):
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = [
        f"DeepSeek-V4-Flash MoE perf (ConfigurableMoE.forward, "
        f"DEP={WORLD_SIZE}, dtype={DTYPE}, routing={ROUTING_METHOD_CLS.__name__}, "
        f"CUDA Graph, warmup={WARMUP_ITERS} timed={TIMED_ITERS})",
        f"Shape: {DEEPSEEK_V4_FLASH}",
        "",
        _fmt(headers),
        "  ".join("-" * w for w in col_widths),
    ]
    for r in rows:
        lines.append(_fmt(r))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Pytest entry
# -----------------------------------------------------------------------------


def _quick_skip_check_for_backend(
    moe_backend: str, quant_algo: Optional[QuantAlgo]
) -> Optional[str]:
    """Pre-check skip reasons before paying the 4-rank MPI startup cost."""
    backend_type = MoeBackendType(moe_backend)
    if backend_type == MoeBackendType.MEGAMOE_CUTEDSL:
        return should_skip_megamoe_cutedsl(
            backend_type,
            quant_algo=quant_algo,
            dtype=DTYPE,
            model_config=DEEPSEEK_V4_FLASH,
            swiglu_gptoss_style=False,
            routing_method_cls=ROUTING_METHOD_CLS,
        )
    if backend_type == MoeBackendType.MEGAMOE:
        # Mirror generate_megamoe_deepgemm_multi_gpu_test_params: do NOT pass
        # comm_method / parallel_mode so the "generic multi-GPU coverage
        # requires a torch.distributed EP ProcessGroup-aware launcher" early-
        # exit doesn't fire — MegaMoEDeepGemm is supported on this MPIPool
        # path (it just runs through the dedicated DeepGemm test param
        # generator in test_moe_module.py).
        return should_skip_megamoe(
            backend_type,
            quant_algo=quant_algo,
            dtype=DTYPE,
            model_config=DEEPSEEK_V4_FLASH,
            moe_tp_size=1,
            swiglu_gptoss_style=False,
        )
    if backend_type == MoeBackendType.CUTEDSL:
        return should_skip_cutedsl(
            backend_type,
            quant_algo,
            DEEPSEEK_V4_FLASH,
            comm_method=None,
            moe_tp_size=1,
        )
    return None


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason=f"needs {WORLD_SIZE} GPUs")
def test_perf_moe_deepseek_v4_flash_dep4():
    """Sweep three backends x six num_tokens; print + write JSON."""
    all_backend_results: List[Dict] = []

    for moe_backend, quant_algo, label in BACKEND_CASES:
        skip_reason = _quick_skip_check_for_backend(moe_backend, quant_algo)
        if skip_reason:
            print(f"\n[skip] {label}: {skip_reason}", flush=True)
            all_backend_results.append(
                {
                    "label": label,
                    "moe_backend": moe_backend,
                    "quant_algo": str(quant_algo) if quant_algo is not None else None,
                    "model_config": str(DEEPSEEK_V4_FLASH),
                    "world_size": WORLD_SIZE,
                    "parallel_mode": PARALLEL_MODE,
                    "skipped": skip_reason,
                    "aggregated": {
                        n: {
                            "rank_max_mean_ms": float("nan"),
                            "rank_max_median_ms": float("nan"),
                            "rank_avg_mean_ms": float("nan"),
                            "per_rank_mean_ms": [],
                            "per_rank_median_ms": [],
                        }
                        for n in NUM_TOKENS_LIST
                    },
                }
            )
            continue

        t_start = time.time()
        try:
            backend_result = _run_one_backend(moe_backend, quant_algo, label)
            backend_result["aggregated"] = _aggregate_per_num_tokens(backend_result["ranks"])
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            backend_result = {
                "label": label,
                "moe_backend": moe_backend,
                "quant_algo": str(quant_algo) if quant_algo is not None else None,
                "model_config": str(DEEPSEEK_V4_FLASH),
                "world_size": WORLD_SIZE,
                "parallel_mode": PARALLEL_MODE,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "ranks": [],
                "aggregated": {
                    n: {
                        "rank_max_mean_ms": float("nan"),
                        "rank_max_median_ms": float("nan"),
                        "rank_avg_mean_ms": float("nan"),
                        "per_rank_mean_ms": [],
                        "per_rank_median_ms": [],
                        "errors": [str(exc)],
                    }
                    for n in NUM_TOKENS_LIST
                },
            }
        backend_result["elapsed_s"] = time.time() - t_start
        all_backend_results.append(backend_result)

        # Per-backend summary print so we get partial output even if a later
        # backend crashes the process.
        print(
            f"\n[{label}] elapsed={backend_result['elapsed_s']:.1f}s",
            flush=True,
        )
        for n in NUM_TOKENS_LIST:
            agg = backend_result["aggregated"][n]
            print(
                f"  num_tokens={n:>5d}  "
                f"median={agg['rank_max_median_ms']:.4f} ms  "
                f"mean={agg['rank_max_mean_ms']:.4f} ms",
                flush=True,
            )

    # Write JSON for downstream inspection.
    payload = {
        "host": socket.gethostname(),
        "world_size": WORLD_SIZE,
        "parallel_mode": PARALLEL_MODE,
        "model_config": str(DEEPSEEK_V4_FLASH),
        "routing_method": ROUTING_METHOD_CLS.__name__,
        "dtype": str(DTYPE),
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "num_tokens_list": NUM_TOKENS_LIST,
        "results": all_backend_results,
    }
    with open(RESULT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[perf] JSON written to {RESULT_JSON_PATH}", flush=True)

    # Final comparison table.
    table = _format_table(all_backend_results)
    print("\n" + table, flush=True)
