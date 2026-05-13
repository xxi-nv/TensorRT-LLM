# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Bisect MEGAMOE_CUTEDSL CUDA-illegal-memory-access by max_num_tokens.

Builds the same DeepSeek-V4-Flash MoE shape used by the perf benchmark,
then for each candidate ``max_num_tokens`` value creates a fresh
``ConfigurableMoE`` and runs ONE eager forward at ``num_tokens=4``.  We
record:
    - PASS / FAIL per max_num_tokens value
    - On FAIL: the truncated exception message (for triage when run
      with ``CUDA_LAUNCH_BLOCKING=1``)

The goal is to identify the smallest ``max_num_tokens`` where the
MEGAMOE_CUTEDSL kernel starts to crash, which is the first signal of
where the workspace allocation / kernel grid sizing breaks.

Run with:
    CUDA_LAUNCH_BLOCKING=1 mpirun --oversubscribe -n 1 \
        python -m pytest <this file>::test_mcd_bisect_max_num_tokens -vs
"""

from __future__ import annotations

import pickle
import sys
import traceback
from typing import Dict, List, Tuple

import cloudpickle
import pytest
import torch
from _torch.modules.moe.moe_test_utils import MoeBackendType, MoeModelConfig
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
from _torch.modules.moe.test_moe_perf_megamoe import _init_worker
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

WORLD_SIZE = 4
PARALLEL_MODE = "DEP"
DTYPE = torch.bfloat16
ROUTING_METHOD_CLS = RenormalizeMoeRoutingMethod
DEEPSEEK_V4_FLASH = MoeModelConfig(256, 6, 4096, 2048)

# Bisect candidates. 256 known PASS via test_moe_module.py. 4096 known FAIL
# via the perf run.
MAX_NUM_TOKENS_CANDIDATES: List[int] = [256, 512, 1024, 2048, 4096]
PROBE_NUM_TOKENS = 4  # tiny: just one forward, not perf measurement


def _probe_one_max(
    max_num_tokens: int,
    mapping,
    moe_backend: str = MoeBackendType.MEGAMOE_CUTEDSL.value,
    quant_algo: QuantAlgo = QuantAlgo.NVFP4,
) -> Tuple[bool, str]:
    """Per-rank: build module with ``max_num_tokens``, run 1 eager forward."""
    try:
        mapping.rank = mpi_rank()
        torch.cuda.set_device(mapping.rank)
        _ensure_dist_for_megamoe(moe_backend, mapping.rank, mapping.world_size)

        model_config = DEEPSEEK_V4_FLASH
        with torch.device(f"cuda:{mapping.rank}"):
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)

            routing_method = _create_routing_method(
                ROUTING_METHOD_CLS,
                top_k=model_config.top_k,
                num_experts=model_config.num_experts,
                dtype=DTYPE,
                model_config=model_config,
            )
            probe_x = torch.randn(
                (max_num_tokens, model_config.hidden_size),
                dtype=DTYPE,
                device="cuda",
            )
            backend_type = MoeBackendType(moe_backend)
            quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
                quant_algo, probe_x, backend_type
            )
            num_local_experts = model_config.num_experts // mapping.moe_ep_size
            quantize_util = quantize_util_cls(
                num_experts=model_config.num_experts,
                dtype=DTYPE,
                intermediate_size=model_config.intermediate_size,
                hidden_size=model_config.hidden_size,
                quant_config=quant_config,
                bias=False,
                swiglu_gptoss_style=False,
                num_local_experts=num_local_experts,
            )
            quant_kwargs.pop("ref_cls", None)

            model_cfg = _create_model_config(
                num_experts=model_config.num_experts,
                hidden_size=model_config.hidden_size,
                intermediate_size=model_config.intermediate_size,
                dtype=DTYPE,
                mapping=mapping,
                quant_config=quant_config,
                moe_backend=moe_backend,
                enable_eplb=False,
                num_slots=-1,
                layer_updates_per_iter=-1,
                max_num_tokens=max_num_tokens,
            )
            mlb = _create_moe_load_balancer(model_cfg, enable_eplb=False)
            wlm = getattr(quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA)

            with (
                mlb,
                create_moe(
                    routing_method=routing_method,
                    reduce_results=True,
                    model_config=model_cfg,
                    bias=False,
                    weight_loading_mode=wlm,
                ) as fused_moe,
            ):
                weights = quantize_util.create_weights(**quant_kwargs)
                fused_moe.load_weights([weights])
                fused_moe.post_load_weights()
                fused_moe.cuda(f"cuda:{mapping.rank}")

                # One small eager forward
                x = torch.randn(
                    (PROBE_NUM_TOKENS, model_config.hidden_size),
                    dtype=DTYPE,
                    device="cuda",
                )
                router_logits = torch.randn(
                    (PROBE_NUM_TOKENS, model_config.num_experts),
                    dtype=DTYPE,
                    device="cuda",
                )
                all_rank_num_tokens = [PROBE_NUM_TOKENS] * mapping.world_size
                with torch.inference_mode():
                    _ = fused_moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
                torch.cuda.synchronize()
        return True, "OK"
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return False, f"{type(exc).__name__}: {exc}\n--- traceback ---\n{tb[:2500]}"


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason=f"needs {WORLD_SIZE} GPUs")
def test_mcd_bisect_max_num_tokens():
    mapping = _create_mapping_for_parallel_mode(WORLD_SIZE, PARALLEL_MODE)
    master_port = _get_free_tcp_port()

    summary: Dict[int, Dict] = {}
    for max_num_tokens in MAX_NUM_TOKENS_CANDIDATES:
        with MPIPoolExecutor(
            initializer=_init_worker,
            initargs=(sys.path, MEGAMOE_CUTEDSL_IGNORE_COMM_METHOD, master_port + max_num_tokens),
            max_workers=WORLD_SIZE,
        ) as executor:
            args = (max_num_tokens, mapping)
            futures = [executor.submit(_probe_one_max, *args) for _ in range(WORLD_SIZE)]
            per_rank: List[Tuple[bool, str]] = []
            for fut in futures:
                try:
                    per_rank.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    per_rank.append((False, f"submit err: {type(exc).__name__}: {exc}"))

        ok_ranks = [r for r in per_rank if r[0]]
        fail_ranks = [r for r in per_rank if not r[0]]
        summary[max_num_tokens] = {
            "all_ok": len(fail_ranks) == 0,
            "ok_count": len(ok_ranks),
            "fail_count": len(fail_ranks),
            "first_fail_msg": fail_ranks[0][1] if fail_ranks else "",
        }
        status = "PASS" if len(fail_ranks) == 0 else "FAIL"
        print(
            f"\n[bisect max_num_tokens={max_num_tokens}] {status} "
            f"(ok={len(ok_ranks)}/{WORLD_SIZE})",
            flush=True,
        )
        if fail_ranks:
            msg = fail_ranks[0][1]
            for line in msg.split("\n"):
                print(f"    {line}", flush=True)
        # If first PASS, keep going (we want the breakpoint).
        # If first FAIL, also keep going to confirm later values also fail.

    print("\n=== bisect summary ===", flush=True)
    for m, s in summary.items():
        print(
            f"  max_num_tokens={m}: {'PASS' if s['all_ok'] else 'FAIL'} "
            f"({s['ok_count']}/{WORLD_SIZE} ok)",
            flush=True,
        )
