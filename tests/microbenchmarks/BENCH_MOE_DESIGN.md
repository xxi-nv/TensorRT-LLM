<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `bench_moe.py` — Design Doc

> Status: **draft, revised after design review** (see §10 *Default Decisions*).
>
> Scope: a new MPI-based microbenchmark `tests/microbenchmarks/bench_moe/__main__.py` that
> times the **whole MoE module** (`ConfigurableMoE.forward`) — routing + dispatch
> + GroupGEMM + activation + combine — across multiple model profiles, parallel
> modes, backends, and synthetic load-imbalance configurations.
>
> Companion files this doc references (all read; line numbers below are evidence):
> - `tests/microbenchmarks/bench_moe_comm.py`
> - `tests/unittest/_torch/modules/moe/test_moe_module.py`
> - `tests/unittest/_torch/modules/moe/moe_test_utils.py`
> - `tensorrt_llm/_torch/modules/fused_moe/{create_moe.py, configurable_moe.py, interface.py}`
> - `tensorrt_llm/tools/layer_wise_benchmarks/{runner.py, calibrator.py, mark_utils.py}`

---

## 1. Goal & Non-Goals

### Goal

A single-file CLI microbenchmark that answers, on demand and without any
HuggingFace checkpoint:

> *"On these GPUs, with this parallel mode and this load imbalance, which MoE
> backend (CUTLASS / TRTLLM-Gen / CuteDSL / DeepGEMM / DenseGEMM /
> MegaMoE-DeepGEMM) runs the fastest end-to-end MoE forward, and which kernels
> dominate the time?"*

It must natively support:

1. Named model profiles with one-shot field overrides.
2. Synthetic per-rank token-count imbalance.
3. Synthetic per-expert hot-spot imbalance (with a fast-path equivalence to the
   existing `ENABLE_PERFECT_ROUTER` route).
4. Global `num_tokens` lists that fan out across ranks per (2).
5. Free choice of parallel mode (DEP / TEP / DTP / TTP) + EP/TP size + EPLB.
6. `ConfigurableMoE` as the **only** forward entry; backend selectable, or
   `best` to scan all and return the winner.
7. CUDA-Graph on/off for the timed region.
8. Per-kernel breakdown + total MoE-forward time, default `warmup=1 / iters=5`,
   with **autotune executed first as an untimed phase**.

### Non-Goals

- Loading a real HuggingFace checkpoint (that's `tools/layer_wise_benchmarks/`).
- Running attention / KV-cache / norm / residual (that's also
  `tools/layer_wise_benchmarks/`).
- Measuring just the dispatch/combine collectives in isolation (that's already
  `tests/microbenchmarks/bench_moe_comm.py`).
- Accuracy / numerical correctness (covered by
  `tests/unittest/_torch/modules/moe/test_moe_module.py`).
- Multi-node launch automation beyond what the existing
  `MPIPoolExecutor + external mpirun` pattern in `bench_moe_comm.py` already
  supports.

---

## 2. Why a New File (vs. extending peers)

| Peer | Why it does not fit |
|---|---|
| `tests/microbenchmarks/bench_moe_comm.py` | Bypasses the MoE module entirely; calls `CommunicationFactory._create_forced_method(...)` to time only `dispatch()` + `combine()` (`bench_moe_comm.py:1127-1146, 228-329`). Does not exercise GEMM / activation / combine inside a real backend. |
| `tools/layer_wise_benchmarks/` | A library, not a CLI: `__init__.py:1-3` exports only `get_calibrator`; `runner.py` has no `main()`/argparse. `Runner.__init__` strictly requires a real HF checkpoint, KV cache, and attention metadata (`runner.py:415-444, 595-678, 760-886`) — too heavy to ask for the question above. |
| `tests/unittest/_torch/modules/moe/test_moe_module.py` | Pytest-driven correctness suite, not a perf tool. No CUPTI/Kineto, no per-kernel breakdown, no backend winner selection. |

Conclusion: write a new benchmark entrypoint that **reuses stable utilities**
where they already exist, but does **not** import pytest test modules or another
executable benchmark as implementation dependencies.

Stable imports:

- `tools/layer_wise_benchmarks/runner.py` for routing-imbalance helpers.
- `moe_test_utils.py` / `quantize_utils.py` for shared MoE model/quant helpers.

Implementation-local or newly extracted helpers:

- Small builders currently embedded in `test_moe_module.py` (`Mapping`,
  `ModelConfig`, routing-method construction, MegaMoE dist setup) should either
  be moved to a shared non-pytest helper or reimplemented locally in
  `bench_moe.py`.
- Timing/CUPTI helpers currently embedded in `bench_moe_comm.py` should either
  be extracted to `tests/microbenchmarks/moe_bench_timing.py` and shared, or
  copied locally with narrow scope. Do not import the `bench_moe_comm.py`
  executable script as a runtime dependency.

---

## 3. Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ bench_moe.py (single file, CLI executable)                               │
│                                                                          │
│  argparse ──▶ main() ──▶ MPIPoolExecutor(ep_size)  OR  external mpirun   │
│                              │                                           │
│                              ▼                                           │
│  per-rank worker:                                                        │
│   1. cupti_early_init           (only if --cuda_graph)                   │
│   2. set device + seed                                                   │
│   3. resolve model profile + mapping                                     │
│   4. for backend in backends_to_run:                                     │
│        for total_tokens in args.num_tokens:                              │
│          ┌──── fresh case build phase ───────────────────────────────┐  │
│          │ a. per_rank = _distribute_tokens(total, ws, t_imb_ratio) │  │
│          │ b. set ENABLE_PERFECT_ROUTER iff experts_hot_imbalane==0 │  │
│          │ c. create_moe(...) ─▶ ConfigurableMoE                    │  │
│          │ d. validate requested_backend == actual_backend          │  │
│          │ e. quantize_util.create_weights() + load + .cuda()       │  │
│          │ f. (EPLB) MoeLoadBalancer ctx + register/finalize        │  │
│          │ g. install routing patch iff experts_hot_imbalane > 0    │  │
│          └──────────────────────────────────────────────────────────┘  │
│          ┌──── measure phase ────────────────────────────────────────┐  │
│          │ inputs   = _make_inputs(per_rank[rank], ...)             │  │
│          │ AUTOTUNE: with autotune(cache_path=...): forward()       │  │
│          │ TIMED:    _time_forward_{eager|cuda_graph}(warmup,iters) │  │
│          │ gather per-rank stats; emit JSON entry                   │  │
│          │ moe.destroy()  # release Communication / NVSHMEM         │  │
│          └──────────────────────────────────────────────────────────┘  │
│   5. (if --backend best) emit per-num_tokens winner ranking             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Reused Code And Helper Boundaries

| Symbol | From | Use |
|---|---|---|
| MPI launcher: `MPIPoolExecutor`, `cloudpickle.register_pickle_by_value`, `MPI.pickle.__init__`, external-mpirun branch | `bench_moe_comm.py:1298-1389` | reuse the same pattern, but keep implementation local or move shared pieces to a non-executable helper |
| `_set_device_from_local_rank`, `_sync`, `_compute_stats`, `_gather_per_rank` | `bench_moe_comm.py:115-136, 799-825` | copy locally or extract to `tests/microbenchmarks/moe_bench_common.py`; do not import `bench_moe_comm.py` |
| Eager Kineto timing + `record_function` + `_parse_profiler_events` | `bench_moe_comm.py:228-459` | adapt locally or extract to shared timing helper |
| CUDA-Graph capture + `cudaEventRecordWithFlags(EXTERNAL)` + CUPTI activity callbacks + `_build_cuda_graph_kernel_stats_cupti` + `_try_init_cupti` | `bench_moe_comm.py:472-796, 1019-1037` | adapt locally or extract to shared timing helper |
| `BalanceMethod`, `get_balanced_selection`, `get_all_to_one_selection`, `get_balanced_rank_imbalanced_expert_selection`, `make_balanced_routing_method`, `make_balanced_run_moe`, `make_forward_impl_check` | `tools/layer_wise_benchmarks/runner.py:46-49, 61-227, 230-392` | inject per-token expert assignments to control hot-spot distribution |
| `_create_mapping_for_parallel_mode(world_size, parallel_mode)` logic | `test_moe_module.py:146-194` | reimplement locally or move to shared helper; must set `mapping.rank = mpi_rank()` per worker |
| `_create_routing_method(routing_method_cls, top_k, num_experts, dtype, model_config)` logic | `test_moe_module.py:350-407` | reimplement locally or move to shared helper; includes DeepSeekV3 `n_group`/`topk_group` handling |
| `MoeBackendType` enum, `MoeModelConfig` dataclass, `resolve_deepseek_group_config` | `tests/unittest/_torch/modules/moe/moe_test_utils.py:60-115` | backend names + shape parameters |
| `get_test_quant_params(quant_algo, x, backend_type)` + `quantize_util.create_weights(...)` + `prepare_weights_from_backend(...)` | `tests/unittest/_torch/modules/moe/quantize_utils.py` (called by `test_moe_module.py:540-657`) | synthetic weights for any quant_algo |
| `_ensure_dist_for_megamoe(moe_backend, rank, world_size)` logic | `test_moe_module.py:129-143` | reimplement locally or move to shared helper; NCCL `ProcessGroup` for MegaMoE only |
| `with autotune(cache_path=...)` | `tensorrt_llm._torch.autotuner` (used by `test_moe_module.py:284-287`) | autotune phase (untimed) |
| `MoeLoadBalancer(...)`, `MoeLoadBalancerIterContext(...)`, `MoeLoadBalancerConfig(...)` | `test_moe_module.py:197-209, 595-666` + `tensorrt_llm.llmapi.llm_args` | EPLB toggle |
| `create_moe(routing_method, num_experts, hidden_size, intermediate_size, dtype, reduce_results, model_config, ...)` | `tensorrt_llm/_torch/modules/fused_moe/create_moe.py:399-526` | sole MoE entry; returns `ConfigurableMoE` when `ENABLE_CONFIGURABLE_MOE=1` (default) and `moe_cls ∈ {CUTLASS, TRTLLM-Gen, CuteDSL, DeepGEMM, DenseGEMM, MegaMoE-DeepGEMM}` |
| `ConfigurableMoE.forward(x, router_logits, all_rank_num_tokens=..., use_dp_padding=...)` | `configurable_moe.py:508-557` + `interface.py:842-888` | the timed callable |
| `ENABLE_PERFECT_ROUTER=1` | `interface.py:375-387, 855` | fast path for `experts_hot_imbalane_ratio == 0` |
| `moe.destroy()` (mandatory at backend swap to avoid DeepEP barrier hang) | `configurable_moe.py:461-475` | release NVSHMEM/DeepEP between backends |

> The `_DummyPretrainedConfig` workaround in `bench_moe_comm.py:76-80` is **not**
> reused. The new file should locally construct a real `PretrainedConfig` using
> the same fields as `test_moe_module.py:222-262`, so `ConfigurableMoE` can read
> `pretrained_config.num_experts / hidden_size / torch_dtype` directly without
> importing the pytest module.

---

## 5. Requirements → Design Mapping

### Req 1 — Embedded model profiles, name-selectable

```python
@dataclass(frozen=True)
class ModelProfile:
    name: str
    moe_model_config: MoeModelConfig          # num_experts, top_k, hidden, intermediate (+ optional n_group/topk_group)
    quant_algo: Optional[QuantAlgo]
    routing_method_cls: type
    swiglu_alpha: float = 1.0
    swiglu_beta:  float = 0.0
    swiglu_limit: float = float("inf")

MODEL_PROFILES: Dict[str, ModelProfile] = {
    "qwen1.5_moe":      ModelProfile("qwen1.5_moe",      MoeModelConfig( 60, 4, 2048, 1408), QuantAlgo.FP8,                 RenormalizeMoeRoutingMethod),
    "deepseek_v2_lite": ModelProfile("deepseek_v2_lite", MoeModelConfig( 64, 6, 2048, 1408), QuantAlgo.FP8_BLOCK_SCALES,    DeepSeekV3MoeRoutingMethod),
    "deepseek_v3":      ModelProfile("deepseek_v3",      MoeModelConfig(256, 8, 7168, 2048, n_group=8, topk_group=4),
                                                                                            QuantAlgo.FP8_BLOCK_SCALES,    DeepSeekV3MoeRoutingMethod),
    "kimi_k2":          ModelProfile("kimi_k2",          MoeModelConfig(384, 8, 7168, 2048), QuantAlgo.FP8_BLOCK_SCALES,    DeepSeekV3MoeRoutingMethod),
    "mixtral_8x7b":     ModelProfile("mixtral_8x7b",     MoeModelConfig(  8, 2, 4096,14336), QuantAlgo.FP8,                 RenormalizeMoeRoutingMethod),
    "gpt_oss_120b":     ModelProfile("gpt_oss_120b",     MoeModelConfig(128, 4, 2880, 2880), QuantAlgo.W4A8_MXFP4_MXFP8,    RenormalizeMoeRoutingMethod,
                                     swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0),
}
```

Shape evidence: `test_moe_module.py:831-849` (CI / LOCAL configs) +
`bench_moe_comm.py:92-107`.

CLI: `--model deepseek_v3` selects a profile; per-field overrides
`--num_experts / --top_k / --hidden_size / --intermediate_size / --quant /
--routing_method` follow the `_resolve_profile_args` pattern in
`bench_moe_comm.py:1000-1016`.

### Req 2 — `--rank_imbalance_ratio ∈ [0, 1]`

```python
def _distribute_tokens(total: int, world_size: int, ratio: float) -> List[int]:
    if world_size <= 0 or total < 0 or not (0.0 <= ratio <= 1.0):
        raise ValueError(...)
    if world_size == 1:
        return [total]

    base = total // world_size
    if ratio == 0.0:
        out = [base] * world_size
        out[0] += total - base * world_size           # remainder to rank 0
        return out
    rank0 = base + round((total - base) * ratio)
    rest_total = total - rank0
    others = rest_total // (world_size - 1)
    out = [rank0] + [others] * (world_size - 1)
    out[1] += rest_total - others * (world_size - 1)  # remainder to rank 1
    return out
```

Output is the per-rank `local_num_tokens` and the `all_rank_num_tokens` list
required by `ConfigurableMoE.forward(...)`.

`ratio == 1` → rank 0 takes everything; other ranks get 0. Zero-token forwards
are legal: `interface.py:836-840` `forward_fake` handles it; backends that
cannot handle it will be detected via `is_workload_feasible(...)` and skipped
with a printed reason (mirrors `bench_moe_comm.py:1169-1173`).

### Req 3 — `--experts_hot_imbalane_ratio ∈ [0, 1]`

| User value | Internal action | Implementation |
|---|---|---|
| `0.0` | Perfectly-balanced routing, no synthetic skew | Set `os.environ["ENABLE_PERFECT_ROUTER"]="1"` **before** `create_moe(...)`; `interface.py:375-387` then takes over |
| `(0, 1]` | Concentrate fraction of tokens onto hot experts | Install `BalanceMethod.ImbalancedExperts` patch with `internal_balance_ratio = 1.0 - ratio`; calls `make_balanced_routing_method(...)` (`runner.py:230-283`) and `make_forward_impl_check(...)` (`runner.py:382-392`); for `TRTLLMGenFusedMoE`, also wrap `run_moe` via `make_balanced_run_moe(...)` (`runner.py:291-379`) |

> **Semantic flip is intentional and documented**: `runner.py:158-165` defines
> `balance_ratio=0` → fully imbalanced and `balance_ratio=1` → fully balanced.
> The user-facing `experts_hot_imbalane_ratio` reads as *"how hot"*, which is
> the inverse, hence `internal_balance_ratio = 1 - experts_hot_imbalane_ratio`.

`rank_imbalance_ratio` and `experts_hot_imbalane_ratio` are
**orthogonal**: the first controls inter-rank token counts; the second controls
intra-rank token-to-expert assignment. Any combination is allowed.

### Req 4 — `--num_tokens` list (global tokens, per-rank derived)

```text
--num_tokens 16 64 256 1024
```

For each value, per-rank distribution = `_distribute_tokens(total, world_size,
rank_imbalance_ratio)`. Example with `world_size=4`:

| total | ratio | per-rank |
|---|---|---|
| 16 | 0.0 | `[4, 4, 4, 4]` |
| 16 | 0.5 | `[10, 2, 2, 2]` |
| 16 | 1.0 | `[16, 0, 0, 0]` |

### Req 5 — Parallel mode + EP/TP size + EPLB

- `--parallel_mode {DEP, TEP, DTP, TTP}` calls
  `_create_mapping_for_parallel_mode(world_size, parallel_mode)` logic
  (`test_moe_module.py:146-194`). The benchmark must set
  `mapping.rank = mpi_rank()` inside every worker before constructing
  `ModelConfig`; otherwise all ranks behave as rank 0.
- `--moe_ep_size N --moe_tp_size M --enable_attention_dp` for fine-grained
  override of the resulting `Mapping`. Validate
  `moe_ep_size * moe_tp_size * moe_cluster_size == tp_size` and
  `tp_size * pp_size * cp_size == world_size`, matching `Mapping` invariants.
- `--enable_eplb --num_slots K --layer_updates_per_iter U`:
  - feed `MoeLoadBalancerConfig(num_slots=K, layer_updates_per_iter=U)` into
    `_create_model_config(...)` (`test_moe_module.py:242-249`);
  - wrap `create_moe(...)` and weight load in `MoeLoadBalancer(ep_rank, ep_size,
    layer_updates_per_iter)` (`test_moe_module.py:595-645`);
  - wrap each timed forward in
    `with MoeLoadBalancerIterContext(moe_load_balancer):`
    (`test_moe_module.py:660-665`).
- `_build_model_config()` must set `use_cuda_graph=not args.no_cuda_graph`,
  because communication/backend selection can inspect `ModelConfig.use_cuda_graph`.

### Req 6 — `--backend {CUTLASS | TRTLLM | CUTEDSL | DEEPGEMM | DENSEGEMM | MEGAMOE_DEEPGEMM | best}`

- Single value → build once **per `num_tokens` value**. This avoids
  cross-size contamination from in-place communication fallback.
- `best` → loop over all six backends as in `bench_moe_comm.py:1100-1146`,
  recording mean MoE-forward time per `(backend, num_tokens)`. `try/except`
  unsupported `(backend, quant)` combinations and print the skip reason.
- All backends are constructed via the same
  `create_moe(routing_method=..., model_config=..., reduce_results=True, ...)`
  call; `create_moe.py:464-487` returns `ConfigurableMoE` for every supported
  backend when `ENABLE_CONFIGURABLE_MOE=1` (the default).
- `MEGAMOE_DEEPGEMM` selection triggers `_ensure_dist_for_megamoe(...)`
  (`test_moe_module.py:129-143`) before `create_moe(...)`.
- Winner is reported per `num_tokens`, not globally — backends scale very
  differently across token counts.
- After `create_moe(...)`, record and validate the **actual backend**. Some
  requested backend/quant/hardware combinations fall back to another backend in
  `create_moe.py`. If `requested_backend != actual_backend`, either skip that
  result with a clear fallback reason or report it under `actual_backend`; never
  rank a fallback result under the requested backend name.
- When a forced communication method is needed, use the existing
  `TRTLLM_FORCE_COMM_METHOD` environment hook consumed by
  `CommunicationFactory.create_strategy(...)`; restore the previous env value
  after the benchmark case.

### Req 7 — `--cuda_graph` / `--no_cuda_graph` (default ON)

Two timing functions, mirroring `bench_moe_comm.py` but with the timed region
collapsed from `dispatch + combine` to a single `moe_forward`:

- **Eager** — `_time_moe_forward_eager`: rewrite of
  `bench_moe_comm.py:228-329`. Replace the two `record_function("dispatch")` /
  `record_function("combine")` blocks with a single
  `record_function("moe_forward")` wrapping `moe.forward(...)`.
- **CUDA Graph** — `_time_moe_forward_cuda_graph`: rewrite of
  `bench_moe_comm.py:618-796`. Same structure:
  1. one eager forward to discover output shape and allocate `static_output`;
  2. unrolled graph capture with `warmup + iters` iterations, only the timed
     iterations record `_record_external(start[i]) / _record_external(end[i])`
     around the single `moe.forward(...)` call;
  3. CUPTI bucketing (`_build_cuda_graph_kernel_stats_cupti`) collapses the
     dispatch/combine windows into a single `moe_forward` window.
- **Hard constraints carried over verbatim**:
  - CUPTI must be initialized **before** the CUDA context — call
    `_try_init_cupti()` at the very top of the worker (mirrors
    `bench_moe_comm.py:1028-1037`).
  - `cudaEventRecordWithFlags(..., cudaEventRecordExternal=0x1)` is required
    for `elapsed_time()` on graph-internal events
    (`bench_moe_comm.py:686-712`).
  - `evt.record()` must be called once before capture to force lazy event
    creation (`bench_moe_comm.py:708-712`).
- CUDA Graph mode is incompatible with **dynamic EPLB updates** in this
  benchmark. `MoeLoadBalancerIterContext` and `ConfigurableMoE.repeat_idx`
  bookkeeping are Python-side state transitions, while graph replay does not
  re-enter Python. If `--enable_eplb` uses dynamic updates, require
  `--no_cuda_graph`. Static EPLB/no-update modes may be allowed only if the
  implementation documents that the captured graph measures a fixed routing /
  fixed assignment state.

### Req 8 — Breakdown + autotune-first + default `warmup=1, iters=5`

Phase order per `(backend, num_tokens)`:

```text
0. build      : create_moe → validate actual backend → load weights → post_load_weights → .cuda()
1. patch      : install routing imbalance (or set ENABLE_PERFECT_ROUTER env in build)
2. AUTOTUNE   : with autotune(cache_path=...): one untimed forward      ← NOT measured
3. WARMUP     : args.warmup forwards (default 1), untimed
4. TIMED      : args.iters forwards (default 5), each wrapped in record_function("moe_forward")
5. PARSE      : Kineto / CUPTI events → per-kernel + total moe_forward times
6. GATHER+EMIT: mpi_allgather per-rank stats → JSON
7. CLEANUP    : moe.destroy(); restore env knobs; shutdown EPLB if needed
```

Evidence:

- autotune call site: `test_moe_module.py:284-287`.
- per-kernel + per-rank gather: `bench_moe_comm.py:412-457, 799-825`.
- only the kernel-category names change vs. `bench_moe_comm.py`: from
  `{dispatch_kernels, combine_kernels, other_kernels}` to
  `{moe_forward_kernels, other_kernels}`.

---

## 6. CLI Surface (proposed)

```text
python tests/microbenchmarks/bench_moe/__main__.py \
    --model deepseek_v3                              # Req 1; or --num_experts/--top_k/--hidden_size/--intermediate_size/--quant
    [--routing_method DEFAULT|RENORMALIZE|RENORMALIZE_NAIVE|LLAMA4_RENORMALIZE|DEEPSEEK_V3|MINIMAX_M2|SIGMOID_RENORM]
    --num_tokens 16 64 256 1024                      # Req 4 (global; per-rank derived)
    --world_size 4                                   # = MPI world; required when not under mpirun
    --parallel_mode DEP                              # Req 5 — DEP|TEP|DTP|TTP
    [--moe_ep_size N --moe_tp_size M --enable_attention_dp]   # Req 5 — fine override
    [--enable_eplb --num_slots K --layer_updates_per_iter U]  # Req 5 — EPLB; dynamic EPLB requires --no_cuda_graph
    --backend best                                   # Req 6 — CUTLASS|TRTLLM|CUTEDSL|DEEPGEMM|DENSEGEMM|MEGAMOE_DEEPGEMM|best
    --rank_imbalance_ratio 0.0                       # Req 2 ∈ [0,1]
    --experts_hot_imbalane_ratio 0.0                 # Req 3 ∈ [0,1]
    [--cuda_graph | --no_cuda_graph]                 # Req 7 — default cuda_graph ON, except dynamic EPLB
    --warmup 1 --iters 5                             # Req 8 defaults
    [--fast_autotune]                                # AutoTuner.warmup=0 / repeat=1 / stream_delay=10us
    [--dtype bfloat16|float16]
    [--use_low_precision_moe_combine]
    [--random_seed 1234]
    [--iter_stats]                                   # show per-iter mean/median/stdev/min/max
    [--kernel_breakdown]                             # default ON for this bench
    [--force_comm_method NVLINK_ONE_SIDED|NVLINK_TWO_SIDED|DEEPEP|DEEPEPLOWLATENCY|ALLGATHER]
    [--output_file out.json]
```

`mpirun` mode (multi-node):

```bash
mpirun -n 4 python tests/microbenchmarks/bench_moe/__main__.py \
    --model deepseek_v3 --backend best --num_tokens 64 256 \
    --parallel_mode DEP --cuda_graph
```

`MPIPoolExecutor` mode (single-node, single-launcher):

```bash
python tests/microbenchmarks/bench_moe/__main__.py \
    --world_size 4 --model deepseek_v3 --backend best --num_tokens 64 256 \
    --parallel_mode DEP --cuda_graph
```

---

## 7. File Skeleton (single file, ~700–900 LOC, English comments only)

```text
tests/microbenchmarks/bench_moe/__main__.py
├── License header + module docstring (with mpirun & spawn examples)
├── Imports
│     ├── from tensorrt_llm.tools.layer_wise_benchmarks.runner import (
│     │       BalanceMethod, get_balanced_selection,
│     │       make_balanced_routing_method, make_balanced_run_moe,
│     │       make_forward_impl_check)
│     ├── from tests.unittest._torch.modules.moe.moe_test_utils import (
│     │       MoeBackendType, MoeModelConfig, resolve_deepseek_group_config)
│     ├── from tests.unittest._torch.modules.moe.quantize_utils import get_test_quant_params
│     ├── from tensorrt_llm._torch.modules.fused_moe import create_moe
│     ├── from tensorrt_llm._torch.autotuner import autotune, AutoTuner
│     ├── from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
│     ├── from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
│     │       MoeLoadBalancer, MoeLoadBalancerIterContext)
│     └── local or shared non-executable timing helpers adapted from bench_moe_comm
│
├── ModelProfile dataclass + MODEL_PROFILES dict        (Req 1)
├── _resolve_profile_args(args)                         (Req 1)
├── _distribute_tokens(total, ws, ratio)                (Req 2/4)
├── _build_mapping(args)                                (Req 5)
│     └── must set mapping.rank = mpi_rank()
├── _build_model_config(args, mapping, profile)         (Req 5 + EPLB)
│     └── must pass use_cuda_graph=not args.no_cuda_graph
├── _build_moe_module(...)  → ConfigurableMoE
│     ├── set ENABLE_PERFECT_ROUTER env BEFORE create_moe iff Req 3 ratio == 0
│     ├── create_moe(...)
│     ├── synthetic weights via quantize_util (mirrors test_moe_module.py)
│     ├── post_load_weights() + .cuda()
│     └── (EPLB) register_weight_slots_after_to_cuda(); finalize_model(); set_iter_info()
│
├── _maybe_install_balance_patch(moe, args, mapping)    (Req 3) — context manager
├── _make_inputs(local_num_tokens, hidden_size, num_experts, dtype, device)
│
├── _validate_actual_backend(requested, moe)                         (Req 6)
├── _run_autotune(moe, inputs, all_rank_num_tokens, eplb_ctx)        (Req 8)
├── _time_moe_forward_eager(...)                                     (Req 7/8)
├── _time_moe_forward_cuda_graph(...)                                (Req 7/8)
│     # both wrap forward in record_function("moe_forward")
│     # return (forward_times_us, detailed_stats)
│
├── _build_output(backend, num_tokens, per_rank, fwd_times, kernel_stats, args)
│
├── parse_args()                                        (§6)
├── _run_benchmark_worker_under_current_mpi(args, launcher)
│     ├── cupti early-init iff cuda_graph
│     ├── tllm.logger.set_level("error")
│     ├── device + seed
│     ├── for backend in backends_to_run:
│     │   for total_tokens in args.num_tokens:
│     │     try:
│     │         build fresh moe for this backend/token case
│     │         validate actual backend or skip fallback
│     │         autotune (untimed)
│     │         with balance_patch_ctx + eplb_ctx:
│     │             inputs = _make_inputs(...)
│     │             times, breakdown = _time_moe_forward_*(...)
│     │             emit JSON entry
│     │     finally:
│     │         moe.destroy()        # configurable_moe.py:461-475
│     └── if --backend best: emit per-num_tokens winner ranking
│
├── _spawn_worker_main(args_blob)
└── main() (cloudpickle hookup + MPIPoolExecutor or external mpirun)
```

---

## 8. Output Schema

```jsonc
{
  "benchmark_metadata": {
    "bench": "bench_moe",
    "model": "deepseek_v3",
    "world_size": 4,
    "parallel_mode": "DEP",
    "moe_ep_size": 4, "moe_tp_size": 1, "enable_attention_dp": true,
    "rank_imbalance_ratio": 0.0,
    "experts_hot_imbalane_ratio": 0.0,
    "perfect_router_used": true,
    "cuda_graph": true,
    "warmup": 1, "iters": 5,
    "enable_eplb": false,
    "quant_algo": "FP8_BLOCK_SCALES",
    "dtype": "torch.bfloat16"
  },
  "results": [
    {
      "num_tokens": 1024,
      "per_rank_num_tokens": [256, 256, 256, 256],
      "requested_backend": "TRTLLM",
      "actual_backend": "TRTLLM",
      "moe_forward_us": {
        "rank0": {"mean": 850.2, "median": 848.0, "stdev": 6.4, "min": 842.1, "max": 859.5},
        "rank1": { "...": "..." }
      },
      "kernel_breakdown": [
        {"name": "...gemm_grouped_fp8...", "count": 5, "per_rank": { "...": "..." }},
        {"name": "...topk_softmax...",     "count": 5, "per_rank": { "...": "..." }}
      ]
    },
    {
      "num_tokens": 1024,
      "best_backend": "TRTLLM",
      "ranking": [["TRTLLM", 850.2], ["CUTLASS", 1102.4], ["CUTEDSL", 1457.0]]
    }
  ]
}
```

`rank0` of `MPIPoolExecutor` writes the JSON file; all ranks print structured
log lines to stdout (the Kineto / CUPTI traces themselves are kept rank-local
to avoid prohibitive I/O).

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Different backends have different `quant_algo` compatibility (e.g. `DEEPGEMM` accepts only `FP8_BLOCK_SCALES`). | Reuse the `should_skip_*` functions exposed by `moe_test_utils.py`. In bench mode, do not pytest-skip — print the reason and continue to the next backend, mirroring `bench_moe_comm.py:1144-1146`. |
| `MEGAMOE_DEEPGEMM` requires `dist.init_process_group("nccl")`. | Call `_ensure_dist_for_megamoe(...)` only when that backend is selected; other paths pay nothing. |
| CUPTI must be initialized before the CUDA context. | Strictly follow `bench_moe_comm.py:1028-1037`: `_try_init_cupti()` is the first call in the worker, before `_set_device_from_local_rank()`. |
| Routing patches must reset their assertion flag every forward. | Use `make_forward_impl_check` (`runner.py:382-392`) which restores `_routing_results_replaced_at = None` per call. New backend → new module → fresh patch. |
| `rank_imbalance_ratio == 1` produces ranks with `local_num_tokens = 0`. | Most backends accept zero-token forward. For ones that don't, surface `is_workload_feasible(...) == False` and print skip reason (mirrors `bench_moe_comm.py:1169-1173`). |
| `ConfigurableMoE` holds NVSHMEM / DeepEP buffers that must be released collectively. | After each `(backend, num_tokens)` case, `try / finally moe.destroy()` (`configurable_moe.py:461-475`); without it, switching cases can hang at the next collective. |
| `experts_hot_imbalane_ratio == 0` via `ENABLE_PERFECT_ROUTER` should be equivalent to `BalanceMethod.Balanced` patch — but they exercise different code paths. | `--force_balance_patch` debug switch routes the `ratio==0` case through the patch as well, so both paths can be diff'd kernel-by-kernel. |
| Router-method dtype quirks (e.g. `DeepSeekV3 + TRTLLM` requires fp32 routing logits, see `cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:70-72`). | Mirror `test_moe_module.py:1336-1343` — set `dtype_routing_logits=torch.float32` for that combination. |
| Importing pytest test modules from a benchmark CLI can fail outside pytest and triggers large parameter-matrix construction at import time. | Do not import `test_moe_module.py`; move small reusable builders to a shared helper or reimplement them locally. |
| `create_moe(...)` can fall back from the requested backend to a different backend. | Validate `actual_backend` after construction and skip/report under actual backend, never under the requested name. |
| `ConfigurableMoE.determine_communication_method(...)` mutates `moe.comm` when a workload is infeasible. | Build a fresh MoE per `(backend, num_tokens)` case, or explicitly recreate comm before each token case. Fresh build is the default design. |
| CUDA Graph replay does not re-enter Python EPLB iteration context or advance Python-side state. | Reject `--cuda_graph` with dynamic EPLB; static/no-update EPLB graph mode must be explicitly documented if enabled later. |
| `Mapping` defaults `rank=0`; shared helper returns a rankless mapping. | Set `mapping.rank = mpi_rank()` in every worker before `ModelConfig` / MoE construction. |
| `ModelConfig.use_cuda_graph` defaults to `False`. | Pass `use_cuda_graph=not args.no_cuda_graph` into `ModelConfig` so backend/comm choices match the timing mode. |

---

## 10. Default Decisions

These defaults are part of the revised design unless the user overrides them:

1. **`--num_tokens` means global token count.** The example
   *"16 + 4 GPUs => 4 tokens/rank"* is interpreted as `total_tokens=16`.
   A future `--tokens_unit {global,per_rank}` can be added, but the initial
   benchmark keeps one meaning.
2. **`best` ranking metric is slowest-rank latency.** For each timed iteration
   use the per-rank MoE-forward time; backend score is the mean of
   `max_across_ranks(iter_time)`. This matches serving step latency.
3. **Dynamic EPLB and expert hot imbalance are mutually exclusive.** EPLB's goal
   is to rebalance experts at runtime, while `experts_hot_imbalane_ratio` fixes
   a synthetic skew. If both are requested, error out. Static/no-update EPLB may
   be allowed with a warning only after implementation proves the routing state
   stays fixed.
4. **CUDA Graph and dynamic EPLB are mutually exclusive.** Dynamic EPLB depends
   on Python iteration context and `repeat_idx` state; graph replay bypasses
   Python. Require `--no_cuda_graph` for dynamic EPLB.
5. **Scope is MoE module only.** The benchmark does not include `gate + MoE`
   block, attention, residual, KV cache, or real checkpoint loading.
6. **NVTX annotation is optional and off by default.** `record_function` is the
   primary attribution mechanism. Add `--nvtx` later if external nsys workflow
   needs it.
7. **Fresh MoE per `(backend, num_tokens)` case.** This avoids communication
   fallback and backend state pollution across token sizes.

---

## 11. Validation Plan (post-implementation, not part of this design step)

1. `--world_size 1 --backend CUTLASS --model qwen1.5_moe --num_tokens 8
   --no_cuda_graph` — single-rank smoke, validates phases 0–6.
2. Same as (1) but `--cuda_graph` — validates CUDA-Graph + CUPTI breakdown.
3. `--world_size 4 --parallel_mode DEP --backend best --num_tokens 64 256
   --cuda_graph` — multi-rank + winner selection.
4. `--experts_hot_imbalane_ratio 0.0` vs.
   `--experts_hot_imbalane_ratio 0.0 --force_balance_patch` on the same config —
   per-kernel times must agree within noise (validates equivalence of the two
   `ratio==0` paths).
5. `--rank_imbalance_ratio 1.0 --num_tokens 64 --world_size 4` —
   expect `per_rank_num_tokens == [64, 0, 0, 0]`; backend either runs or emits
   `is_workload_feasible=False` skip.
6. `--enable_eplb --num_slots 16 --backend CUTLASS --model qwen1.5_moe` —
   `num_slots > num_experts` constraint per `test_moe_module.py:1640-1641`;
   verifies the EPLB code path does not crash.

---

## 12. Implementation Roadmap (after §10 is settled)

1. Land `bench_moe.py` skeleton (imports + `parse_args` + `main` + worker
   shell).
2. Add `ModelProfile` table + `_distribute_tokens` + local/shared
   `_build_mapping` / `_build_model_config` / `_build_moe_module` helpers
   (no direct import from `test_moe_module.py`; no backend loop yet).
3. Implement `_run_autotune` + `_time_moe_forward_eager` end-to-end; validate
   with single-GPU smoke.
4. Add `_time_moe_forward_cuda_graph` (adapt or extract CUPTI plumbing from
   `bench_moe_comm.py`; do not import `bench_moe_comm.py` as a script module).
5. Add `_maybe_install_balance_patch` + perfect-router env shortcut.
6. Add actual-backend validation to catch `create_moe(...)` fallback.
7. Add fresh-build multi-backend loop + `best` winner selection.
8. Add EPLB toggle with `cuda_graph` / hot-imbalance mutual exclusion.
9. Add JSON output + per-rank gather.
10. Run §11 validation; iterate on backend-specific skip reasons.

Each step is a single PR-sized commit, signed off; total estimated 700–900 LOC
in one file plus this design doc.
