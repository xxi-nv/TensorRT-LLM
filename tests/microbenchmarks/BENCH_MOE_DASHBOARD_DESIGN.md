<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `bench_moe.py` Dashboard-Oriented Design

## Goal

The goal is to evolve `tests/microbenchmarks/bench_moe.py` from a standalone
MoE microbenchmark into a data generator for a user-facing MoE performance
dashboard.

The dashboard should let a user choose a MoE model shape, sweep token counts,
search valid runtime configurations, and answer:

> For this model and token count, which MoE configuration is fastest, which
> backend / communication / EPLB / parallel-mode choices were used, where did
> the time go, and what is the bottleneck?

The primary output should be structured data that can power dashboards and
automated performance comparisons. Human-readable stdout is secondary.

## Current State

`bench_moe.py` already has several useful pieces:

- It times `ConfigurableMoE.forward`, not a full decoder layer
  (`bench_moe.py:18-22`).
- It has built-in model shape presets and field overrides
  (`bench_moe.py:157-235`, `bench_moe.py:1235-1276`).
- It supports global `num_tokens` sweeps and derives per-rank token counts
  (`bench_moe.py:350-372`, `bench_moe.py:1278-1284`).
- It supports DEP / TEP / DTP / TTP mappings
  (`bench_moe.py:398-442`).
- It can scan MoE backends with `--backend BEST`
  (`bench_moe.py:1518-1547`, `bench_moe.py:1771-1799`).
- It can force a single communication method through `TRTLLM_FORCE_COMM_METHOD`
  (`bench_moe.py:1366-1378`, `bench_moe.py:1587-1590`).
- It has basic eager and CUDA Graph timing paths with kernel breakdown
  (`bench_moe.py:851-888`, `bench_moe.py:1070-1172`).

The main gap is not the ability to run a single benchmark. The main gap is that
the benchmark dimensions are still CLI flags, not a structured search space and
not a dashboard schema.

## Scope

In scope:

- Pure MoE module performance, using `ConfigurableMoE.forward`.
- Synthetic MoE weights and inputs; no HuggingFace checkpoint dependency.
- Model shape presets plus explicit shape overrides.
- Token-count sweeps after the model is chosen.
- Search over valid runtime configurations:
  - backend
  - communication method
  - EPLB mode
  - parallel mode
  - CUDA Graph on/off
  - low precision MoE combine
- Per-configuration timing and bottleneck attribution.
- JSON output that is stable enough for dashboard ingestion.

Out of scope:

- Full decoder-layer performance with attention / KV cache / residual / norm.
  That remains the role of `examples/layer_wise_benchmarks`.
- End-to-end serving performance.
- Accuracy validation; correctness is covered by unit tests.
- Reproducing full user traffic distributions in this file. For this benchmark,
  after the model is fixed, the main workload axis is token count.

## Core Mental Model

### Model

The selected model determines the static MoE shape:

- `num_experts`
- `top_k`
- `hidden_size`
- `intermediate_size`
- default quantization
- default routing method
- optional model-specific activation style

This corresponds to `ModelProfile` in the current file
(`bench_moe.py:157-167`).

### Workload

For this MoE-only benchmark, the workload should stay intentionally simple.
After the model is selected, the primary workload axis is:

```text
num_tokens
```

The current input construction confirms this: the forward input is just
`[local_num_tokens, hidden_size]` plus router logits
`[local_num_tokens, num_experts]` (`bench_moe.py:375-390`).

Prefill/decode, batch size, sequence length, and concurrency are full-model
concepts. In this pure MoE benchmark, they should be reduced to token counts
before entering the benchmark.

Workload modifiers are still useful, but they should not become a separate
"workload suite" concept:

- token distribution across ranks
- expert distribution / hotspot behavior
- optional future replay of routing assignments

### Runtime Configuration

Runtime configuration is the true search space:

```text
backend x communication x EPLB x parallel mode x CUDA graph x combine precision
```

The dashboard should search this space for each `(model, num_tokens, workload
modifier)` point.

## Target Data Model

### ModelSpec

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    quant_algo: Optional[str]
    routing_method: str
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    swiglu_limit: float = float("inf")
```

Rules:

- `--model` selects a built-in `ModelSpec`.
- Shape flags override fields only when explicitly passed.
- `--routing_method AUTO` should use the profile default. This avoids the
  current trap where the CLI default can override a model-specific routing
  method (`bench_moe.py:1266-1275`, `bench_moe.py:1433-1440`).
- If no `--model` is selected, the user must provide enough shape fields to
  build a custom spec.
- For a custom shape without `--model`, `--routing_method AUTO` should fail
  with a clear error. Falling back to `RENORMALIZE` would silently create a
  benchmark row whose routing does not correspond to any explicit user choice.

### WorkloadSpec

```python
@dataclass(frozen=True)
class WorkloadSpec:
    num_tokens: int
    rank_imbalance_ratio: float = 0.0
    expert_distribution: Literal["balanced_patch", "hotspot"] = "balanced_patch"
    expert_hotspot_ratio: float = 0.0
```

Rules:

- The dashboard's x-axis is normally `num_tokens`, and `num_tokens` is always a
  global token count.
- `rank_imbalance_ratio` controls how global tokens are split across ranks:
  `0` is balanced and `1` puts all tokens on rank 0.
- `rank_imbalance_ratio` replaces `rank_hotspot_ratio` and
  `tokens_per_rank_imbalance_ratio`.
- `expert_hotspot_ratio` should replace the misspelled
  `experts_hot_imbalane_ratio`; the old flag can remain as a deprecated alias.
- `expert_distribution` should start with only values that map to implemented
  routing-logit generation:
  - `balanced_patch`: current balanced routing patch.
  - `hotspot`: current hotspot patch controlled by `expert_hotspot_ratio`.
  `random` can be added later only with a documented random-logit algorithm and
  deterministic seed handling.

### ConfigSpec

```python
@dataclass(frozen=True)
class ConfigSpec:
    backend: str
    parallel_mode: str
    moe_ep_size: Optional[int] = None
    moe_tp_size: Optional[int] = None
    enable_attention_dp: Optional[bool] = None
    comm_method: Literal[
        "AUTO",
        "NVLINK_ONE_SIDED",
        "NVLINK_TWO_SIDED",
        "DEEPEP",
        "DEEPEPLOWLATENCY",
        "ALLGATHER",
    ] = "AUTO"
    eplb_mode: Literal["off", "static", "dynamic"] = "off"
    num_slots: Optional[int] = None
    layer_updates_per_iter: Optional[int] = None
    cuda_graph: bool = True
    use_low_precision_moe_combine: bool = False
    per_case_subprocess: bool = False
```

Rules:

- `parallel_mode` should be the normal user interface.
- Explicit `moe_ep_size` / `moe_tp_size` should create `parallel_mode="CUSTOM"`
  in output metadata.
- `comm_method="AUTO"` means allow `CommunicationFactory` to choose.
- `comm_method` must become a real per-case configuration field before
  `--search comm` is implemented. The current `TRTLLM_FORCE_COMM_METHOD`
  environment variable is process-global and is only acceptable as a deprecated
  compatibility path, not as the internal search mechanism.
- `eplb_mode` should replace the lower-level `enable_eplb` +
  `layer_updates_per_iter` mental model.
- Dynamic EPLB means weight updates are allowed at runtime.
- Static EPLB means extra slots / routing through slots are enabled, but no
  per-iteration migration is performed.
- `per_case_subprocess=True` means each candidate is launched in a fresh worker
  process. This is the safe mode for large searches because backend autotune
  caches, CUDA workspaces, CUPTI buffers, and OOM state are otherwise shared
  across cases in the same process.

### RunResult

```python
@dataclass
class RunResult:
    model: ModelSpec
    workload: WorkloadSpec
    config: ConfigSpec
    status: Literal["success", "skipped", "failed"]
    skip_reason: Optional[str]
    actual_backend: Optional[str]
    actual_comm_method: Optional[str]
    actual_comm_fallback_reason: Optional[str]
    scheduler_kind: Optional[str]
    num_chunks: Optional[int]
    per_rank_num_tokens: list[int]
    status_per_rank: dict
    instrumentation: dict
    latency_us: dict
    phase_times_us: dict
    kernel_breakdown: dict
    bottleneck: Optional[str]
```

The dashboard should treat this as the core row schema.

`actual_comm_method` should use `"NONE"` when no external communication object
exists, for example single-rank runs, non-DP mappings, or fused-communication
backends where cross-rank exchange is inside the kernel.

The timing dictionaries must have stable shapes:

```text
latency_us:
  score: float
  score_type: str
  per_rank: {rank_i: {mean, median, min, max, stdev, p90}}

phase_times_us:
  agg: {phase: {score, score_type}}
  per_rank: {rank_i: {phase: {mean, median, min, max, stdev, p90}}}

kernel_breakdown:
  per_rank: {rank_i: [{name, calls, total_us, mean_us, max_us}]}
```

The default score formula is:

```text
score = mean_i(max_r(latency_us[rank=r][iteration=i]))
```

In other words, compute the slowest rank for every measured iteration, then
average those per-iteration slowest-rank values. This is different from taking
the maximum of per-rank means and is more faithful when the slowest rank changes
across iterations.

## Search Space

### Search Presets

The CLI should distinguish fixed runs from searches:

```text
--search none       # run exactly the supplied config
--search backend    # search backend only
--search comm       # fixed backend, search communication methods
--search parallel   # search DEP / TEP / DTP / TTP
--search eplb       # search off / static / dynamic EPLB
--search full       # search all valid dimensions
```

The current `--backend BEST` should be treated as a compatibility shortcut for:

```text
--search backend --backend ALL
```

### Candidate Generation

For each `WorkloadSpec`, generate `ConfigSpec` candidates in this order:

1. Expand backend candidates.
2. Expand parallel-mode candidates.
3. Expand communication candidates.
4. Expand EPLB candidates.
5. Expand CUDA Graph and combine-precision candidates.
6. Prune invalid combinations before construction.
7. Run valid candidates and keep skip reasons for invalid or failed ones.

Every skipped candidate should still appear in output. Missing rows make a
dashboard misleading.

Candidate pruning should be explicit and source-backed. A first-pass matrix:

| Backend family | Scheduler kind | Communication search | Parallel modes | EPLB modes |
|---|---|---|---|---|
| External-comm backends | `EXTERNAL_COMM` | `AUTO`, NVLink, DeepEP, AllGather as allowed by platform/workload feasibility | `DEP`, `TEP`, `DTP`, `TTP`, `CUSTOM` | `off`, `static`, `dynamic` if backend and quantization support EPLB |
| Fused-comm backends | `FUSED_COMM` | no host communication object; output `actual_comm_method="NONE"` | only mappings supported by the backend capability checks | `off`, `static`, `dynamic` only when backend-specific EPLB constraints pass |
| Dense or local-only fallbacks | usually `EXTERNAL_COMM` | `AUTO` or `NONE`, depending on mapping | capability-check dependent | usually `off` unless explicitly supported |

The matrix is a guide, not a hard-coded replacement for capability checks. The
implementation should still call backend `can_implement()`, backend validation,
communication `is_platform_supported()`, communication `is_workload_feasible()`,
and EPLB validation before constructing a case.

### Ranking

For a serving-like dashboard, the default score should remain:

```text
score = mean_i(max_r(latency_us[rank=r][iteration=i]))
```

This matches the current slowest-rank winner logic
(`bench_moe.py:1744-1751`).

Future ranking modes:

- `slowest_rank_mean` (default)
- `rank0_mean`
- `median_of_slowest_rank`
- `p90_of_slowest_rank`
- `tokens_per_second`

### Process Lifetime & Isolation

Search runs may construct many `ConfigurableMoE` instances in one invocation.
That is convenient, but it is not always a clean experimental boundary.

Risks:

- backend autotune caches can survive across candidates;
- CUDA workspaces and temporary tensors can keep memory pressure high;
- CUPTI activity buffers can survive beyond a single case;
- an OOM or CUDA error in one candidate can poison later candidates.

Rules:

- Always call `moe.destroy()` and release obvious case-local tensors after each
  candidate, including successful candidates.
- Provide `--per_case_subprocess` for large searches. In this mode, the parent
  process generates candidates and each candidate runs in a fresh worker process
  that returns one `RunResult`.
- Provide safety limits such as `--max_configs` and `--time_budget_minutes` so
  accidental full Cartesian products do not monopolize a node.
- Treat subprocess isolation as the recommended mode for dashboard production
  sweeps. In-process execution remains useful for local quick runs.

## Validity Rules

The benchmark should classify invalid combinations before running whenever
possible.

### Backend Capability

Use backend `can_implement()` as the first gate. The current helper already
does this (`bench_moe.py:242-267`, `bench_moe.py:1518-1547`).

This gate must receive the exact tuple used for construction:

- quant algorithm
- activation dtype
- `swiglu_gptoss_style`

### Communication Rules

Communication is only meaningful for external-communication backends and DP/EP
workloads.

Rules:

- Forced communication should be passed through an explicit per-case config
  field. Environment-variable forcing can remain as a compatibility alias, but
  search code must not rely on mutating `os.environ` between cases.
- If `enable_attention_dp=False`, forced communication should be rejected or
  warned, because `CommunicationFactory` returns `None` when attention does not
  use DP (`communication_factory.py:105-107`).
- If `dp_size == 1`, the actual communication method is `"NONE"` even if the
  requested method is `AUTO`.
- If `moe_tp_size != 1`, all-to-all communication is not supported and the
  factory falls back to `AllGatherReduceScatter`
  (`communication_factory.py:109-112`).
- If backend scheduler kind is `FUSED_COMM`, host-side `Communication.dispatch`
  / `combine` must not be layered on top. The output should record
  `actual_comm_method="NONE"` and `scheduler_kind="FUSED_COMM"` rather than
  pretending dispatch/combine were free. The MoE guide states this invariant for
  MegaMoE-style backends (`MOE_DEVELOPER_GUIDE.md:204-212`).
- If `comm_method=AUTO`, the result must record the actual communication class.
  The factory can select NVLinkOneSided, NVLinkTwoSided, DeepEP,
  DeepEPLowLatency, or AllGather (`communication_factory.py:131-213`).
- The factory should expose a structured selection trace or fallback reason for
  benchmark use. Prefer an optional trace/debug object over changing the public
  return type for production callers.

### EPLB Rules

Rules:

- `eplb_mode=off`: no load balancer.
- `eplb_mode=static`: alias for `enable_eplb=True` with
  `layer_updates_per_iter=0`. It is not a separate current runtime feature. It
  enables routing through slots and EPLB bookkeeping, but it does not migrate
  weights during timed iterations.
- `eplb_mode=dynamic`: allow runtime statistics and weight updates.
- `static` may be slower than `off` because statistics, slot routing, and CPU
  stage hooks can still run. The dashboard should show those phases explicitly.
- `num_slots` must be valid when EPLB is enabled. `num_slots == num_experts`
  should not be the default for `static` because it degenerates toward
  "EPLB bookkeeping without extra capacity". Use a friendly default such as
  `num_slots = round_up(num_experts + ep_size, ep_size)` for `static` and
  `num_slots = round_up(num_experts * slot_multiplier, ep_size)` for `dynamic`.
- `dynamic` EPLB is incompatible with CUDA Graph timing until graph replay can
  safely model Python-side state changes. The current validation already guards
  the low-level form (`bench_moe.py:1457-1462`).
- Synthetic expert hotspot and dynamic EPLB should not be searched together by
  default. They have conflicting goals: one forces imbalance, the other tries
  to rebalance it. The current file already rejects this combination
  (`bench_moe.py:1452-1456`).

### Parallel-Mode Rules

Rules:

- `DEP`: `moe_ep_size=world_size`, `moe_tp_size=1`, `enable_attention_dp=True`.
- `TEP`: `moe_ep_size=world_size`, `moe_tp_size=1`, `enable_attention_dp=False`.
- `DTP`: `moe_ep_size=1`, `moe_tp_size=world_size`, `enable_attention_dp=True`.
- `TTP`: `moe_ep_size=1`, `moe_tp_size=world_size`, `enable_attention_dp=False`.

The current implementation follows this mapping (`bench_moe.py:398-420`).

Explicit EP/TP overrides should be either:

- mutually exclusive with `--parallel_mode`, or
- represented as `parallel_mode="CUSTOM"` in all metadata and output rows.

The second option is more flexible for advanced users.

## Runtime Introspection

The dashboard must record what actually happened, not only what was requested.

Required fields:

- requested backend
- actual backend
- requested communication method
- actual communication class
- communication selection trace or fallback reason
- scheduler kind (`EXTERNAL_COMM` or `FUSED_COMM`)
- parallel mode and resolved mapping
- EPLB mode and resolved slot count
- CUDA Graph mode and whether CUPTI breakdown was available
- `num_chunks`
- fallback reason when communication or backend changes path
- status per rank when rank-local status can be collected

Current `bench_moe.py` records actual backend (`bench_moe.py:1645-1658`) but
does not record actual communication or scheduler kind.

Implementation options:

- Inspect `moe.comm.__class__.__name__` after construction and after the first
  forward, because `determine_communication_method` can mutate `moe.comm`.
- Inspect `moe.backend.scheduler_kind`.
- Add a lightweight debug/introspection method to `ConfigurableMoE` if direct
  attribute reads become brittle.
- Write structured JSON only from rank 0. Non-zero ranks should send their
  latency, phase, kernel, and status payloads to rank 0 through collective
  gather operations before rank 0 emits the result row.

## Phase-Level Analysis

Kernel names alone are not enough to explain bottlenecks. The dashboard needs
phase-level timing.

### Required Phases

For `EXTERNAL_COMM` scheduler:

```text
routing
eplb_wait_gpu
eplb_update_statistic
eplb_route
comm_prepare
quantize
dispatch
backend_run_moe
eplb_set_cpu_start
combine
eplb_set_cpu_done
all_reduce_or_reduce_results
```

For `FUSED_COMM` scheduler:

```text
routing
eplb_wait_gpu
eplb_update_statistic
eplb_route
quantize
fused_comm_backend_run_moe
eplb_set_cpu
```

The scheduler source already defines these conceptual phases
(`moe_scheduler.py:295-500`, `moe_scheduler.py:759-830`).

### Instrumentation Strategy

Add phase markers in the scheduler path, not through benchmark monkey-patching.
`MoEScheduler` owns forward-time policy, so it is the stable place to keep
phase names aligned with the real execution order.

Use a lightweight marker abstraction:

```python
with moe_perf_phase("dispatch"):
    ...
```

The marker should support:

- `torch.profiler.record_function` for eager timing.
- NVTX ranges for nsys traces. These may be enabled by a profiling flag or kept
  as lightweight default annotations if measurement impact is negligible.
- CUDA event ranges for phase timing only when `--analysis phases` is enabled.
- CUDA Graph compatible external event ranges for graph timing when feasible.

The first implementation can support eager phase breakdown and CUDA Graph
kernel breakdown. CUDA Graph phase breakdown can follow once event range
nesting is reliable.

`FUSED_COMM` backends have coarser phase visibility. Because dispatch and
combine are inside the fused kernel, the dashboard should not emit
`dispatch_us=0` or `combine_us=0`. It should instead report a fused phase such
as `fused_comm_backend_run_moe` and mark dispatch/combine as `not_applicable`.

Suggested phase-to-code anchors:

```text
routing                       -> routing_method.apply(...)
eplb_wait_gpu                 -> _load_balancer_start_wait_gpu_stage / done_wait_gpu_stage
eplb_update_statistic         -> _load_balancer_update_statistic(...)
eplb_route                    -> _load_balancer_route(...)
comm_prepare                  -> comm.prepare_dispatch(...)
quantize                      -> backend.quantize_input(...)
dispatch                      -> comm.dispatch(...)
backend_run_moe               -> backend.run_moe(...)
eplb_set_cpu_start            -> _load_balancer_start_set_cpu_stage(...)
combine                       -> comm.combine(...)
eplb_set_cpu_done             -> _load_balancer_done_set_cpu_stage(...)
all_reduce_or_reduce_results  -> wrapper all_reduce path when reduce_results applies
fused_comm_backend_run_moe    -> backend.run_moe(...) for FUSED_COMM scheduler
```

### Analysis Overhead

Analysis must be explicit because instrumentation can perturb the measured
latency.

Rules:

- Default `--analysis summary`: no CUPTI activity tracing and no phase event
  timing.
- `--analysis phases`: enable phase markers/events and record
  `instrumentation.level="phases"`.
- `--analysis phases,kernels`: enable phase markers and kernel breakdown, record
  `instrumentation.level="phases,kernels"`.
- Dashboard comparisons should group or filter by instrumentation level. A run
  measured with phase/kernel tracing should not be directly compared against a
  summary-only latency row unless explicitly requested.

### Overlap Reporting

Overlap matters because chunking can use an auxiliary stream
(`moe_scheduler.py:540-632`).

The output should report:

- per-phase summed time
- forward wall time
- overlap estimate:

```text
overlap_us = sum(phase_times_us) - moe_forward_wall_time_us
overlap_ratio = overlap_us / sum(phase_times_us)
```

This is not a perfect dependency graph, but it is useful for dashboard-level
diagnosis.

### Bottleneck Classification

Use simple first-pass rules:

- `communication_bound`: dispatch + combine dominates.
- `compute_bound`: backend GEMM / fused kernel dominates.
- `routing_bound`: routing/topk dominates small-token cases.
- `eplb_bound`: EPLB wait/stat/route/set CPU dominates.
- `launch_overhead_bound`: many small kernels and low GPU time per kernel.
- `unknown`: missing phase markers or ambiguous split.

Do not invent hardware metrics. MFU/SOL requires NCU or additional profiling.

## CLI Direction

### Current CLI Problems

The current CLI mixes model, workload, runtime config, search, and analysis in
one flat namespace (`bench_moe.py:1235-1381`).

Specific issues:

- `--routing_method` default should be `AUTO`, not `RENORMALIZE`.
- `--experts_hot_imbalane_ratio` is misspelled and should become
  `--expert_hotspot_ratio`.
- `--backend BEST` is too narrow for the dashboard goal.
- `--force_comm_method` is a fixed config knob, not a search knob.
- `--enable_eplb`, `--num_slots`, and `--layer_updates_per_iter` expose
  implementation details instead of an EPLB mode.
- Explicit EP/TP overrides can silently override `--parallel_mode` semantics.

### Proposed CLI

Quick fixed run:

```bash
python tests/microbenchmarks/bench_moe.py \
  --model deepseek_v3 \
  --num_tokens 1 2 4 8 16 32 64 128 \
  --backend TRTLLM \
  --parallel_mode DEP \
  --comm_method AUTO \
  --eplb_mode off
```

Backend search:

```bash
python tests/microbenchmarks/bench_moe.py \
  --model deepseek_v3 \
  --num_tokens 1 2 4 8 16 32 64 128 \
  --search backend
```

Full dashboard data generation:

```bash
python tests/microbenchmarks/bench_moe.py \
  --model deepseek_v3 \
  --num_tokens 1 2 4 8 16 32 64 128 256 512 1024 \
  --search full \
  --analysis phases,kernels \
  --output_file out/moe_dashboard_deepseek_v3.json
```

Dashboard pipeline run from a config file:

```bash
python tests/microbenchmarks/bench_moe.py \
  --config_file configs/moe_dashboard_deepseek_v3.json
```

The first implementation should support JSON config files because that requires
only the Python standard library. YAML can be added later if the repository
already has an accepted YAML dependency in the benchmark environment.

Example config file:

```json
{
  "model": "deepseek_v3",
  "workload": {
    "num_tokens": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "rank_imbalance_ratio": 0.0
  },
  "search": {
    "backend": "ALL",
    "parallel_mode": ["DEP", "TEP"],
    "comm_method": ["AUTO"],
    "eplb_mode": ["off", "static", "dynamic"]
  },
  "analysis": ["phases"],
  "per_case_subprocess": true,
  "output_file": "out/moe_dashboard_deepseek_v3.json"
}
```

Advanced custom shape:

```bash
python tests/microbenchmarks/bench_moe.py \
  --num_experts 384 \
  --top_k 6 \
  --hidden_size 7168 \
  --intermediate_size 3072 \
  --quant FP8_BLOCK_SCALES \
  --routing_method AUTO \
  --num_tokens 1 2 4 8 16 32
```

The last example should error until `--routing_method` is set to an explicit
method. `AUTO` only has a safe default when a built-in model profile is selected.

Deprecated aliases:

- `--experts_hot_imbalane_ratio` should warn and map to
  `--expert_hotspot_ratio`.
- `--force_comm_method` should warn and map to `--comm_method`.

Use `warnings.warn(..., DeprecationWarning)` and document a removal window so
compatibility aliases do not become permanent.

## Output Schema

Top-level JSON:

```json
{
  "schema_version": 2,
  "benchmark": "bench_moe",
  "environment": {
    "world_size": 2,
    "world_size_per_node": 2,
    "hostname": "...",
    "device_name": "NVIDIA ...",
    "sm": 100,
    "cuda_version": "...",
    "driver_version": "...",
    "torch_version": "...",
    "trtllm_commit": "...",
    "nvlink_topology": "unknown",
    "memory_type": "unknown",
    "clock_locked": false,
    "cuda_graph_default": true
  },
  "model": {
    "name": "deepseek_v3",
    "num_experts": 256,
    "top_k": 8,
    "hidden_size": 7168,
    "intermediate_size": 2048,
    "quant_algo": "FP8_BLOCK_SCALES",
    "routing_method": "DeepSeekV3MoeRoutingMethod"
  },
  "results": []
}
```

Each result row:

```json
{
  "workload": {
    "num_tokens": 128,
    "rank_imbalance_ratio": 0.0,
    "per_rank_num_tokens": [64, 64],
    "expert_distribution": "balanced_patch",
    "expert_hotspot_ratio": 0.0
  },
  "requested_config": {
    "backend": "TRTLLM",
    "parallel_mode": "DEP",
    "comm_method": "AUTO",
    "eplb_mode": "off",
    "cuda_graph": true,
    "use_low_precision_moe_combine": false
  },
  "actual_config": {
    "backend": "TRTLLM",
    "comm_method": "NVLinkOneSided",
    "comm_fallback_reason": null,
    "scheduler_kind": "EXTERNAL_COMM",
    "moe_ep_size": 2,
    "moe_tp_size": 1,
    "enable_attention_dp": true,
    "num_chunks": 1
  },
  "status": "success",
  "status_per_rank": {
    "rank0": "success",
    "rank1": "success"
  },
  "instrumentation": {
    "level": "phases,kernels",
    "cuda_graph": true,
    "cupti_available": true,
    "phase_timing_available": true,
    "kernel_breakdown_available": true
  },
  "latency_us": {
    "score": 850.2,
    "score_type": "slowest_rank_mean",
    "per_rank": {
      "rank0": {"mean": 842.1, "median": 841.9, "min": 838.0, "max": 849.0, "stdev": 2.0, "p90": 847.0},
      "rank1": {"mean": 850.2, "median": 849.7, "min": 844.0, "max": 859.0, "stdev": 3.0, "p90": 856.0}
    }
  },
  "phase_times_us": {
    "agg": {
      "routing": {"score": 12.0, "score_type": "slowest_rank_mean"},
      "dispatch": {"score": 80.0, "score_type": "slowest_rank_mean"},
      "backend_run_moe": {"score": 650.0, "score_type": "slowest_rank_mean"},
      "combine": {"score": 70.0, "score_type": "slowest_rank_mean"}
    },
    "per_rank": {
      "rank0": {
        "routing": {"mean": 11.0, "median": 11.0, "min": 10.0, "max": 13.0, "stdev": 1.0, "p90": 12.5}
      }
    }
  },
  "overlap": {
    "overlap_us": 25.0,
    "overlap_ratio": 0.03
  },
  "bottleneck": "compute_bound",
  "kernel_breakdown": {
    "per_rank": {
      "rank0": []
    }
  }
}
```

Ranking rows should be derived from result rows by post-processing, not mixed
into the same list as ad-hoc records. The current output appends both run rows
and ranking entries into `all_results` (`bench_moe.py:1728-1742`,
`bench_moe.py:1776-1799`); this should become:

```json
{
  "results": [],
  "rankings": []
}
```

## Implementation Plan

### Phase 1: Fix Current CLI Semantics

Changes:

1. Add `--routing_method AUTO` and make it the default.
   - Built-in model profile: `AUTO` resolves to the profile default.
   - Custom shape: `AUTO` errors and asks for an explicit routing method.
2. Add `--expert_hotspot_ratio` and keep `--experts_hot_imbalane_ratio` as a
   deprecated alias.
3. Add `--rank_imbalance_ratio`; keep `--rank_hotspot_ratio` and
   `--tokens_per_rank_imbalance_ratio` as deprecated aliases.
4. Add `--comm_method`, keep `--force_comm_method` as a deprecated alias.
   - Internally pass the requested communication method through config, not
     `TRTLLM_FORCE_COMM_METHOD`.
5. Add `--eplb_mode {off,static,dynamic}` and map it to current load-balancer
   internals.
6. Record `parallel_mode="CUSTOM"` when explicit EP/TP values are used.
7. Add validation / warnings for forced communication on non-DP or MoE-TP paths.
8. Add `--analysis {summary,phases,phases,kernels}` with `summary` as the
   default.
9. Add `--per_case_subprocess`, `--max_configs`, and `--time_budget_minutes`.
10. Add `--config_file` for dashboard pipelines.

### Phase 2: Introduce Structured Specs

Add dataclasses:

- `ModelSpec`
- `WorkloadSpec`
- `ConfigSpec`
- `SearchSpec`
- `Candidate`
- `RunResult`

Replace direct use of `argparse.Namespace` in the worker with these structures.
The CLI becomes one way to create specs, not the internal data model.

### Phase 3: Generalize Search

Replace backend-only `BEST` with candidate generation:

```text
SearchSpec -> list[ConfigSpec] -> prune -> run -> rank
```

Keep compatibility:

- `--backend BEST` maps to backend search.
- fixed `--backend X` with `--search none` still runs exactly one backend.

Search execution rules:

- Run in-process by default for quick local runs.
- Use subprocess isolation for dashboard sweeps or whenever
  `--per_case_subprocess` is set.
- Always produce a row for skipped, failed, and successful candidates.
- Store communication selection and backend fallback reasons in the row.

### Phase 4: Runtime Introspection

Record:

- `actual_backend`
- `actual_comm_method`
- `actual_comm_fallback_reason`
- `scheduler_kind`
- `num_chunks`
- `status_per_rank`
- `instrumentation`
- CUDA Graph / CUPTI breakdown status

### Phase 5: Phase-Level Instrumentation

Add markers inside `moe_scheduler.py`, because scheduler code owns forward-time
policy. Do not monkey-patch scheduler methods from the benchmark.

Start with eager phase markers. Add CUDA Graph phase attribution later.

Implementation should be opt-in to avoid perturbing default timing:

```text
--analysis summary
--analysis phases
--analysis phases,kernels
```

`FUSED_COMM` rows should report fused-kernel-level timing and mark external
dispatch/combine phases as `not_applicable`, not zero.

### Phase 6: Dashboard Schema

Move output to schema version 2:

- `environment`
- `model`
- `results`
- `rankings`
- `skips`
- `errors`
- fixed sub-schema for `latency_us`, `phase_times_us`, `kernel_breakdown`, and
  `instrumentation`

Keep a small compatibility converter if existing scripts depend on the current
output.

## Validation Plan

Smoke tests:

1. Single-rank fixed backend:
   - `world_size=1`
   - `backend=CUTLASS`
   - `num_tokens=[1, 8]`
   - `search=none`
2. Single-rank backend search:
   - verifies unsupported backends appear as skipped.
3. Multi-rank DEP backend search:
   - verifies per-rank score and ranking.
4. `parallel_mode=CUSTOM`:
   - explicit `moe_ep_size` / `moe_tp_size`.
5. `comm_method=AUTO`:
   - verifies actual communication is recorded.
6. Forced communication on invalid mapping:
   - verifies friendly skip/error.
7. `eplb_mode=static`:
   - verifies slot setup without dynamic updates.
8. `eplb_mode=dynamic --no_cuda_graph`:
   - verifies dynamic EPLB path remains eager.
9. `expert_hotspot_ratio > 0`:
   - verifies routing patch behavior.
10. `routing_method=AUTO --model deepseek_v3`:
    - verifies DeepSeek routing is preserved.
11. custom shape with `routing_method=AUTO`:
    - verifies a clear error asks for an explicit routing method.
12. `comm_method=AUTO` on `world_size=1`:
    - verifies `actual_comm_method="NONE"`.
13. `--per_case_subprocess`:
    - verifies one failed/skipped candidate does not drop other result rows.
14. `--config_file`:
    - verifies JSON config and equivalent CLI produce the same candidate list.

## Design Principles

- Keep workload simple. For pure MoE, once model shape is selected, token count
  is the main workload axis.
- Put complexity in configuration search, not in workload modeling.
- Always record requested and actual config.
- Never silently drop skipped configurations.
- Prefer stable JSON rows over human-only logs.
- Treat process lifetime as part of benchmark correctness. Large dashboard
  sweeps should support per-case subprocess isolation.
- Keep measurement overhead explicit. Dashboard comparisons must know whether a
  row used summary-only timing, phase markers, or kernel tracing.
- Keep `ConfigurableMoE` as the execution entry and align with the
  backend / communication / scheduler architecture described in
  `MOE_DEVELOPER_GUIDE.md`.
- Do not invent performance metrics. Bottleneck labels must be derived from
  measured phase or kernel times.

## Resolved Design Decisions

- `num_tokens` is always global tokens; rank imbalance is controlled by
  `rank_imbalance_ratio`.
- Phase instrumentation lives in `moe_scheduler.py`, not in benchmark
  monkey-patches.
- Communication selection should expose a benchmark-readable trace or fallback
  reason, preferably without changing the production return type.
- Model profiles should move out of the script into a shared MoE benchmark/test
  helper module once structured specs are introduced.

## Open Questions

1. Should `static` EPLB mean `num_slots > num_experts` with no weight migration,
   or should it also support a fixed precomputed slot assignment?
2. Should dashboard production sweeps always force `--per_case_subprocess`, or
   should this remain a recommendation?
3. Should YAML config files be supported in addition to JSON if the benchmark
   environment already has a YAML parser available?
