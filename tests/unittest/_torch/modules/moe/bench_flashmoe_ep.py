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
"""FlashMoE EP Communication Benchmark: NCCL (V1) vs Symmetric Memory (V2).

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
    """Benchmark V1 (NCCL) vs V2 (symm mem) EP communication."""
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

        # --- Create V1 (NCCL) model ---
        model_v1 = FlashMoECuteDsl(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=True,
            model_config=ep_model_cfg,
            use_symm_mem_ep=False,
        )
        model_v1.load_weights([weights])
        model_v1.cuda(f"cuda:{rank}")

        # --- Create V2 (symm mem) model ---
        model_v2 = FlashMoECuteDsl(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=True,
            model_config=ep_model_cfg,
            use_symm_mem_ep=True,
        )
        model_v2.load_weights([weights])
        model_v2.cuda(f"cuda:{rank}")

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
            print(f"  warmup={warmup_iters}, bench_iters={bench_iters}")
            print(f"{'=' * 60}")
            print(f"\nWarming up ({warmup_iters} iters)...")

        with torch.inference_mode():
            for _ in range(warmup_iters):
                _run_forward(model_v1, "warmup_v1")
                _run_forward(model_v2, "warmup_v2")
            torch.cuda.synchronize()

        dist.barrier()

        # --- Benchmark V1 (NCCL) ---
        if rank == 0:
            print(f"\nBenchmarking V1 (NCCL) EP: {bench_iters} iters...")

        start_events_v1 = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
        end_events_v1 = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

        with torch.inference_mode():
            for i in range(bench_iters):
                if use_nsys and i == 0:
                    torch.cuda.nvtx.range_push("bench_v1_nccl")
                start_events_v1[i].record()
                _run_forward(model_v1, f"v1_iter_{i}")
                end_events_v1[i].record()
                if use_nsys and i == bench_iters - 1:
                    torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        dist.barrier()

        # --- Benchmark V2 (symm mem) ---
        if rank == 0:
            print(f"Benchmarking V2 (Symmetric Memory) EP: {bench_iters} iters...")

        start_events_v2 = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
        end_events_v2 = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

        with torch.inference_mode():
            for i in range(bench_iters):
                if use_nsys and i == 0:
                    torch.cuda.nvtx.range_push("bench_v2_symm_mem")
                start_events_v2[i].record()
                _run_forward(model_v2, f"v2_iter_{i}")
                end_events_v2[i].record()
                if use_nsys and i == bench_iters - 1:
                    torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        dist.barrier()

        # --- Results (rank 0 only) ---
        if rank == 0:
            times_v1 = [
                start_events_v1[i].elapsed_time(end_events_v1[i]) for i in range(bench_iters)
            ]
            times_v2 = [
                start_events_v2[i].elapsed_time(end_events_v2[i]) for i in range(bench_iters)
            ]

            avg_v1 = sum(times_v1) / len(times_v1)
            avg_v2 = sum(times_v2) / len(times_v2)
            min_v1 = min(times_v1)
            min_v2 = min(times_v2)
            max_v1 = max(times_v1)
            max_v2 = max(times_v2)

            # Median
            s_v1 = sorted(times_v1)
            s_v2 = sorted(times_v2)
            med_v1 = s_v1[len(s_v1) // 2]
            med_v2 = s_v2[len(s_v2) // 2]

            # P95
            p95_v1 = s_v1[int(len(s_v1) * 0.95)]
            p95_v2 = s_v2[int(len(s_v2) * 0.95)]

            print(f"\n{'=' * 60}")
            print(f"RESULTS (rank 0, {bench_iters} iterations)")
            print(f"{'=' * 60}")
            print(f"{'Metric':<12} {'V1 NCCL (ms)':>14} {'V2 SymmMem (ms)':>16} {'Speedup':>10}")
            print(f"{'-' * 12} {'-' * 14} {'-' * 16} {'-' * 10}")
            print(f"{'Mean':<12} {avg_v1:>14.3f} {avg_v2:>16.3f} {avg_v1 / avg_v2:>9.2f}x")
            print(f"{'Median':<12} {med_v1:>14.3f} {med_v2:>16.3f} {med_v1 / med_v2:>9.2f}x")
            print(f"{'Min':<12} {min_v1:>14.3f} {min_v2:>16.3f} {min_v1 / min_v2:>9.2f}x")
            print(f"{'Max':<12} {max_v1:>14.3f} {max_v2:>16.3f} {max_v1 / max_v2:>9.2f}x")
            print(f"{'P95':<12} {p95_v1:>14.3f} {p95_v2:>16.3f} {p95_v1 / p95_v2:>9.2f}x")
            print(f"{'=' * 60}")

            # Print individual iteration times for detailed analysis
            if args.verbose:
                print("\nPer-iteration times (ms):")
                print(f"{'Iter':<6} {'V1 NCCL':>10} {'V2 SymmMem':>12}")
                for i in range(bench_iters):
                    print(f"{i:<6} {times_v1[i]:>10.3f} {times_v2[i]:>12.3f}")


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
