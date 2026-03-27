# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""FlashMoE EP Communication Benchmark: NCCL (V1) vs Symmetric Memory (V2) vs Pipelined RS (V3).

Usage:
    # Basic benchmark (4 GPUs, mp.spawn):
    trtllm-llmapi-launch python tests/unittest/_torch/modules/moe/bench_flashmoe_ep.py

    # With NSYS profiling:
    nsys profile -w true -t cuda,nvtx --force-overwrite true \
        -o flashmoe_ep_bench \
        trtllm-llmapi-launch python tests/unittest/_torch/modules/moe/bench_flashmoe_ep.py --nsys

    # Custom parameters:
    trtllm-llmapi-launch python tests/unittest/_torch/modules/moe/bench_flashmoe_ep.py \
        --num-experts 256 --top-k 8 --hidden-size 7168 --intermediate-size 2048 \
        --seq-len 32 --warmup 10 --iters 50

    # Benchmark specific versions only:
    trtllm-llmapi-launch python tests/unittest/_torch/modules/moe/bench_flashmoe_ep.py --versions v1 v3
"""

import argparse
import os
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _bench_worker(rank, world_size, port, args):
    """Worker process for EP benchmark."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        _bench_worker_impl(rank, world_size, args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _create_moe_weights(num_experts, hidden_size, intermediate_size, dtype):
    """Create random MoE expert weights (w1, w2, w3 for SwiGLU)."""
    weights = {}
    for eid in range(num_experts):
        weights[f"{eid}.w1.weight"] = torch.randn(
            (intermediate_size, hidden_size), dtype=dtype, device="cuda"
        )
        weights[f"{eid}.w2.weight"] = torch.randn(
            (hidden_size, intermediate_size), dtype=dtype, device="cuda"
        )
        weights[f"{eid}.w3.weight"] = torch.randn(
            (intermediate_size, hidden_size), dtype=dtype, device="cuda"
        )
    return weights


def _bench_worker_impl(rank, world_size, args):
    """Benchmark V1 (NCCL) vs V2 (symm mem) vs V3 (pipelined RS) EP communication."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_flashmoe import FlashMoECuteDsl
    from tensorrt_llm.mapping import Mapping

    dtype = torch.bfloat16
    num_experts = args.num_experts
    top_k = args.top_k
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    seq_len = args.seq_len
    warmup_iters = args.warmup
    bench_iters = args.iters
    use_nsys = args.nsys
    versions = args.versions

    with torch.device(f"cuda:{rank}"):
        # Deterministic data across ranks
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        total_tokens = seq_len * world_size
        all_x = torch.randn((total_tokens, hidden_size), dtype=dtype, device="cuda")
        all_router_logits = torch.randn((total_tokens, num_experts), dtype=dtype, device="cuda")

        my_x = all_x[rank * seq_len : (rank + 1) * seq_len].contiguous()
        my_logits = all_router_logits[rank * seq_len : (rank + 1) * seq_len].contiguous()

        # Shared weights (inline, no test-util dependency)
        weights = _create_moe_weights(num_experts, hidden_size, intermediate_size, dtype)

        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)

        ep_mapping = Mapping(
            world_size=world_size,
            tp_size=world_size,
            moe_ep_size=world_size,
            moe_tp_size=1,
            enable_attention_dp=True,
        )
        ep_mapping.rank = rank

        ep_model_cfg = ModelConfig(
            pretrained_config=pretrained_config,
            mapping=ep_mapping,
        )

        all_rank_num_tokens = [seq_len] * world_size

        # --- Create models for requested versions ---
        models = {}
        model_labels = {
            "v1": "V1 NCCL",
            "v2": "V2 SymmMem",
            "v3": "V3 PipeRS",
            "graph": "Graph",
        }

        if "v1" in versions:
            m = FlashMoECuteDsl(
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                reduce_results=True,
                model_config=ep_model_cfg,
                use_symm_mem_ep=False,
                ep_comm_version="v1",
            )
            m.load_weights([weights])
            m.cuda(f"cuda:{rank}")
            models["v1"] = m

        if "v2" in versions:
            m = FlashMoECuteDsl(
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                reduce_results=True,
                model_config=ep_model_cfg,
                use_symm_mem_ep=True,
                ep_comm_version="v2",
            )
            m.load_weights([weights])
            m.cuda(f"cuda:{rank}")
            models["v2"] = m

        if "v3" in versions:
            m = FlashMoECuteDsl(
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                reduce_results=True,
                model_config=ep_model_cfg,
                ep_comm_version="v3",
            )
            m.load_weights([weights])
            m.cuda(f"cuda:{rank}")
            models["v3"] = m

        if "graph" in versions:
            m = FlashMoECuteDsl(
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                reduce_results=True,
                model_config=ep_model_cfg,
                ep_comm_version="graph",
            )
            m.load_weights([weights])
            m.cuda(f"cuda:{rank}")
            models["graph"] = m

        def _run_forward(model, label):
            """Run forward pass with optional NVTX annotation."""
            if use_nsys:
                torch.cuda.nvtx.range_push(label)
            out = model.forward(
                my_x.clone(),
                my_logits.clone(),
                all_rank_num_tokens=all_rank_num_tokens,
            )
            if use_nsys:
                torch.cuda.nvtx.range_pop()
            return out

        # --- Warmup ---
        if rank == 0:
            print(f"\n{'=' * 60}")
            print("FlashMoE EP Benchmark")
            print(f"  experts={num_experts}, top_k={top_k}")
            print(f"  hidden={hidden_size}, intermediate={intermediate_size}")
            print(f"  seq_len={seq_len} per rank, total={total_tokens}")
            print(f"  world_size={world_size}")
            print(f"  versions={versions}")
            print(f"  warmup={warmup_iters}, bench_iters={bench_iters}")
            print(f"{'=' * 60}")
            print(f"\nWarming up ({warmup_iters} iters)...")

        with torch.inference_mode():
            for _ in range(warmup_iters):
                for v, model in models.items():
                    _run_forward(model, f"warmup_{v}")
            torch.cuda.synchronize()

        dist.barrier()

        # --- Benchmark each version ---
        all_times = {}
        for v, model in models.items():
            label = model_labels[v]
            if rank == 0:
                print(f"\nBenchmarking {label}: {bench_iters} iters...")

            starts = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

            with torch.inference_mode():
                for i in range(bench_iters):
                    if use_nsys and i == 0:
                        torch.cuda.nvtx.range_push(f"bench_{v}")
                    starts[i].record()
                    _run_forward(model, f"{v}_iter_{i}")
                    ends[i].record()
                    if use_nsys and i == bench_iters - 1:
                        torch.cuda.nvtx.range_pop()

            torch.cuda.synchronize()
            dist.barrier()

            if rank == 0:
                all_times[v] = [starts[i].elapsed_time(ends[i]) for i in range(bench_iters)]

        # --- Results (rank 0 only) ---
        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"RESULTS (rank 0, {bench_iters} iterations)")
            print(f"{'=' * 70}")

            # Build header
            header = f"{'Metric':<12}"
            for v in versions:
                if v in all_times:
                    header += f" {model_labels[v] + ' (ms)':>16}"
            print(header)
            print("-" * len(header))

            def _stats(times):
                s = sorted(times)
                return {
                    "mean": sum(s) / len(s),
                    "median": s[len(s) // 2],
                    "min": s[0],
                    "max": s[-1],
                    "p95": s[int(len(s) * 0.95)],
                }

            stats = {v: _stats(t) for v, t in all_times.items()}

            for metric in ["mean", "median", "min", "max", "p95"]:
                row = f"{metric.capitalize():<12}"
                for v in versions:
                    if v in stats:
                        row += f" {stats[v][metric]:>16.3f}"
                print(row)

            # Speedup vs V1 (if V1 is included)
            if "v1" in stats:
                print()
                for v in versions:
                    if v != "v1" and v in stats:
                        speedup = stats["v1"]["median"] / stats[v]["median"]
                        print(f"  {model_labels[v]} vs V1: {speedup:.2f}x (median)")

            print(f"{'=' * 70}")

            if args.verbose:
                print("\nPer-iteration times (ms):")
                header = f"{'Iter':<6}"
                for v in versions:
                    if v in all_times:
                        header += f" {model_labels[v]:>12}"
                print(header)
                for i in range(bench_iters):
                    row = f"{i:<6}"
                    for v in versions:
                        if v in all_times:
                            row += f" {all_times[v][i]:>12.3f}"
                    print(row)


def main():
    parser = argparse.ArgumentParser(description="FlashMoE EP Benchmark")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=32, help="Tokens per rank")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--nsys", action="store_true", help="Add NVTX annotations for NSYS profiling"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration times")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3", "graph"],
        choices=["v1", "v2", "v3", "graph"],
        help="EP versions to benchmark (default: v1 v2 v3 graph)",
    )
    args = parser.parse_args()

    from tensorrt_llm._utils import get_free_port

    port = get_free_port()
    mp.spawn(
        _bench_worker,
        args=(args.world_size, port, args),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
