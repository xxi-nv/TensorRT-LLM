# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""MegaMoE single-GPU microbench for Phase C-a profiling.

Runs `MegaMoE.forward` N times under `TLLM_NVTX_DEBUG=1` so each step
(routing / quant / moe_sort / fc1_swiglu / memset_overlap / fc2_combine / reduce)
shows up as an nvtx range under nsys. Outputs wall-clock per-iter and
saves a nsys-stats summary if `--nsys-stats` is passed.

Invocation (inside container on B200):
    nsys profile -o mega_bench --trace=cuda,nvtx --force-overwrite=true \\
        -c cudaProfilerApi python bench_mega_moe.py \\
        --num-experts 8 --hidden 512 --intermediate 512 --top-k 2 \\
        --seq-len 256 --iters 50 --warmup 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Force nvtx_range_debug to activate.
os.environ.setdefault("TLLM_NVTX_DEBUG", "1")

# Prevent mpi4py from auto-initialising Open MPI at import time.
# We run this bench as a plain single-GPU Python process (srun --mpi=none),
# so OMPI's PMI handshake would fail and tear down the process. This must
# happen BEFORE any tensorrt_llm import pulls mpi4py in.
try:
    import mpi4py  # noqa: E402

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
except ImportError:
    pass

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# tests/unittest/_torch/modules/moe/ -> tests/unittest (3 levels up)
_UNITTEST_DIR = os.path.abspath(os.path.join(_TEST_DIR, "../../.."))
if _UNITTEST_DIR not in sys.path:
    sys.path.append(_UNITTEST_DIR)

import torch  # noqa: E402
from _torch.modules.moe.quantize_utils import NVFP4QuantizeUtil, get_test_quant_params  # noqa: E402

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe.fused_moe_mega import MegaMoE  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantAlgo  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--profile-range",
        action="store_true",
        help="Wrap the timed region in torch.cuda.cudart().cudaProfilerStart/Stop.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dtype = torch.bfloat16

    with torch.device("cuda:0"):
        x = torch.randn((args.seq_len, args.hidden), dtype=dtype)
        router_logits = torch.randn((args.seq_len, args.num_experts), dtype=dtype)

        routing_method = RenormalizeMoeRoutingMethod(top_k=args.top_k, force_enable_pytorch_op=True)

        _, quant_config, quant_kwargs = get_test_quant_params(QuantAlgo.NVFP4, x)
        quantize_util = NVFP4QuantizeUtil(
            num_experts=args.num_experts,
            dtype=dtype,
            intermediate_size=args.intermediate,
            hidden_size=args.hidden,
            quant_config=quant_config,
            num_local_experts=args.num_experts,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        mapping = Mapping()
        model_cfg = ModelConfig(mapping=mapping, quant_config=quant_config)
        mega = MegaMoE(
            routing_method=routing_method,
            num_experts=args.num_experts,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            dtype=dtype,
            reduce_results=False,
            model_config=model_cfg,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        )
        mega.load_weights([weights])
        mega.post_load_weights()
        mega.cuda("cuda:0")

        mega.eval()
        with torch.inference_mode():
            for _ in range(args.warmup):
                mega.forward(x, router_logits)
            torch.cuda.synchronize()

            if args.profile_range:
                torch.cuda.cudart().cudaProfilerStart()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                mega.forward(x, router_logits)
            end_evt.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            if args.profile_range:
                torch.cuda.cudart().cudaProfilerStop()

        gpu_ms = start_evt.elapsed_time(end_evt)

    per_iter_gpu_us = gpu_ms * 1000.0 / args.iters
    per_iter_wall_us = wall * 1e6 / args.iters
    print(
        f"[bench] E={args.num_experts} H={args.hidden} I={args.intermediate} "
        f"topk={args.top_k} S={args.seq_len} iters={args.iters} "
        f"gpu_us/iter={per_iter_gpu_us:.1f} wall_us/iter={per_iter_wall_us:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
