<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `bench_moe.py` Routing Control 设计

## 目标

为 `tests/microbenchmarks/bench_moe.py` 增加高级 routing 控制模式，让 benchmark 能有意生成通信流量和 expert 热点程度可控的 MoE routing workload。

这个目标不是替换模型原生 routing，而是让性能实验可复现：

- 控制 source rank 到 target rank 的 dispatch / all-to-all 通信流量。
- 控制每个 expert 的 token histogram，用于 MoE compute 研究。
- 在 JSON 输出中记录实际生成的 routing matrix 和 expert histogram。
- 明确 TRTLLM TEP 行为，因为它的 fused routing path 和 DEP、supplied-topk routing 都不同。

## 非目标

- 不改变 production MoE routing、communication、scheduler 或 backend 行为。
- 不把 benchmark-only routing helper 变成公开 LLM API。
- 不承诺任意通信矩阵都可以由模型原生 logits 表达，尤其是 DeepSeekV3 这类 grouped router。
- 不把 source-rank token imbalance 作为 rank imbalance 的默认含义。Source token imbalance 有用，但它不是 dispatch traffic imbalance。

## 当前状态

`WorkloadSpec` 目前的 workload knob 混合了几个不同概念：

```python
@dataclass(frozen=True)
class WorkloadSpec:
    num_tokens: int
    rank_imbalance_ratio: float = 0.0
    expert_distribution: str = "balanced_patch"
    expert_hotspot_ratio: float = 0.0
```

`rank_imbalance_ratio` 当前控制的是全局输入 token 如何切分到 source ranks：

```python
def _per_rank_tokens(workload: WorkloadSpec, world_size: int) -> List[int]:
    ratio = float(workload.rank_imbalance_ratio)
    return _distribute_tokens(int(workload.num_tokens), world_size, ratio)
```

### `rank_imbalance_ratio` 的用途和用法

`rank_imbalance_ratio` 是历史 workload knob，语义是 **source rank local token count skew**，不是 communication target skew。

它的输入范围是 `[0, 1]`：

- `0.0`：全局 `num_tokens` 尽量均匀切到所有 ranks；余数放到 rank0。
- `1.0`：rank0 拿走全部 `num_tokens`，其他 ranks 的 local token 数为 0。
- `(0, 1)`：在均匀切分和 rank0 全热点之间插值。

例如 `world_size=4`、`num_tokens=16`：

```text
rank_imbalance_ratio=0.0 -> per_rank_num_tokens=[4, 4, 4, 4]
rank_imbalance_ratio=0.5 -> per_rank_num_tokens=[10, 2, 2, 2]
rank_imbalance_ratio=1.0 -> per_rank_num_tokens=[16, 0, 0, 0]
```

它适合研究：

- source rank 本地 routing / quantize / dispatch pack 工作量不均衡；
- `max(all_rank_num_tokens)` 变化带来的 workspace、padding、chunking 变化；
- slowest-rank synchronization 和 source-side tail latency。

它不适合用来研究：

- target rank receive hotspot；
- `src -> dst` pair hotspot；
- expert 内部 token histogram skew。

这些通信和 compute 形态应该由 `comm_pattern`、`expert_pattern` 或显式 `dispatch_matrix` / `expert_histogram` 控制。新设计中保留 `rank_imbalance_ratio` 只是为了兼容旧命令；新用法应优先使用 `--per_rank_num_tokens` 来显式表达 source-rank token 分布。

这对研究 source-side skew 和 slowest-rank synchronization 有价值，但它不能直接控制 dispatch traffic matrix：

```text
dispatch_matrix[src_rank][dst_rank]
```

当前 expert hotspot 控制是通过 layerwise benchmark helper patch routing 实现的。它可以影响 expert selection，但不会把生成出的 source-to-target matrix 或 expert histogram 作为一等 benchmark 数据暴露出来。

## 性能模型

### Dispatch / All-to-All

Dispatch 通信由 top-k expert selection 之后的 routing 决定。最重要的对象是 source-to-target matrix：

```text
D[src][dst] = 从 source rank src 发到 target rank dst 的 selected expert slots
              或 distinct token payload 数量
```

不同通信路径会对 `D` 的不同投影敏感：

- Row sums：每个 source rank 的总发送工作量。
- Column sums：每个 target rank 的总接收工作量。
- Pair hot spots：单个 `src -> dst` pair 可能主导某条 peer link。
- Off-diagonal traffic：local-only routing 和 full all-to-all 即使 row / column sums 相同，通信行为也可能完全不同。
- Runtime max tokens per rank：多条路径会围绕 `max(all_rank_num_tokens)` 分配 workspace 或做运行时规划。

因此，单个标量如 `rank_imbalance_ratio` 无法精确描述 dispatch traffic。高级控制应该直接生成并报告 `D`。

### MoE Compute

MoE compute 由 dispatch 后的 per-expert token histogram 决定：

```text
H[target_rank][local_expert] = target_rank 上 local_expert 拥有的 selected expert slots 数量
```

两个 workload 可以有相同的 target-rank receive total，但 compute 行为不同：

```text
balanced experts: [256, 256, 256, 256]
hot experts:      [1024, 0, 0, 0]
```

第一个 workload 有多个大小接近的 expert group；第二个 workload 有一个很大的 expert group 和多个空 expert group。这会改变 grouped-GEMM tile 分布、padding waste 和 tail behavior。

## 面向用户的高级模式

使用紧凑的高级 knobs。它们描述 workload shape，而不是 row sums / column sums 这类实现细节，同时避免给已经很大的 benchmark CLI 再增加大量平铺参数。

```bash
--routing_mode native|forced
--projection_policy project|reject
--comm_pattern balanced_alltoall
--comm_pattern receiver_hotspot,hotness=0.75,rank=0
--comm_pattern pair_hotspot,hotness=0.5,src=0,dst=1
--comm_pattern local_only
--comm_pattern ring
--comm_pattern file:path/to/matrix.json

--expert_pattern balanced
--expert_pattern hotspot,hotness=0.75
--expert_pattern hotspot,active_experts=2
--expert_pattern file:path/to/histogram.json

--routing_scenario balanced_baseline|receiver_hotspot_75|pair_hotspot_50|local_only_baseline
--per_rank_num_tokens 128,128,512,128
--routing_dump_matrix
```

`--comm_pattern` 的输入矩阵总是按 **slot traffic** 解释：

```text
slot_traffic[src][dst] = dst 拥有的 selected expert slots 数量
```

benchmark 也必须在输出里报告 distinct-token traffic，但输入文件格式使用 slot-based 语义，因为 `selected_experts` 本身是 slot-based。

`--per_rank_num_tokens` 有意和 `--comm_pattern` 分离。它控制 source-rank local input token counts、`max(all_rank_num_tokens)` 和 chunking 相关行为；它本身不定义 routed tokens 被发往哪里。

### Routing Mode

`routing_mode=native`

- 保留模型原生 routing kernels。
- 根据 requested `comm_pattern` / `expert_pattern` 反推 `router_logits`。
- 最适合测量真实 TRTLLM TEP fused routing path。
- 不保证可以精确实现任意 traffic matrix；不能精确满足时，benchmark 必须生成最接近的 routing logits，并在 actual output 中记录 projection warning 和偏差。

`routing_mode=forced`

- 直接 materialize `topk_ids` 和 `topk_weights`。
- 最适合精确控制 dispatch 和 compute workload。
- 在 TRTLLM TEP 下，这会改变 routing path：scoring / top-k 会被跳过，但后续 routing metadata、permutation 和 grouped-GEMM setup 仍然会执行。
- 这是显式 opt-in 模式，不能作为 `auto` 的隐式 fallback。

`projection_policy=project`

- 仅对 `routing_mode=native` 生效。
- 当 requested pattern 无法被当前 routing type 精确表示时，使用 closest projection 继续运行。
- actual output 必须记录 `routing_realization.status="projected"`、projection reason 和 observed deviation。

`projection_policy=reject`

- 仅对 `routing_mode=native` 生效。
- 当 requested pattern 无法被当前 routing type 精确表达时，直接跳过该 case，并给出清晰原因。
- 适合用户不希望 benchmark 静默改变请求 workload 的调试场景。

`--routing_scenario`

- 是常用组合的 preset，展开后仍然落到 `routing_mode`、`projection_policy`、`comm_pattern` 和 `expert_pattern`。
- `balanced_baseline` 等价于 `routing_mode=native --comm_pattern balanced_alltoall --expert_pattern balanced`。
- `receiver_hotspot_75` 等价于 `routing_mode=native --projection_policy project --comm_pattern receiver_hotspot,hotness=0.75,rank=0`。
- `pair_hotspot_50` 等价于 `routing_mode=native --projection_policy project --comm_pattern pair_hotspot,hotness=0.5,src=0,dst=1`。
- `local_only_baseline` 等价于 `routing_mode=native --projection_policy reject --comm_pattern local_only`。

### 通信模式

`balanced_alltoall`

- 每个 source rank 将 selected slots 尽量均匀地发送到所有 target ranks。
- 这是公平通信对比的默认模式。

`receiver_hotspot`

- 可配置比例的 traffic 发往一个 target rank。
- 由 `receiver_hotspot,hotness=<ratio>,rank=<rank>` 控制。
- `hotness` 是 per-row 比例：每个 source row 都把对应比例的 slot traffic 转向热点 receiver；它不是全局 traffic 比例。
- 用于研究接收端带宽和排队瓶颈。

`pair_hotspot`

- 可配置比例的 traffic 发往一个 `src:dst` pair。
- 由 `pair_hotspot,hotness=<ratio>,src=<src>,dst=<dst>` 控制。
- `hotness` 是指定 source row 内的比例。
- 用于研究 peer-link hotspot。

`local_only`

- 尽量选择本 rank 上的 experts。
- 用于最小通信量 baseline。

`ring`

- Source rank `i` 优先发往 target rank `(i + 1) % ep_size`。
- 用于结构化 peer traffic，区别于 full all-to-all。

`file:path/to/matrix.json`

- 从 JSON 加载显式 source-to-target matrix。
- 这是精确实验的 escape hatch。
- 文件格式固定为：

```json
{
  "schema": "v1",
  "ep_size": 4,
  "slot_dispatch_matrix": [
    [64, 64, 64, 64],
    [64, 64, 64, 64],
    [64, 64, 64, 64],
    [64, 64, 64, 64]
  ]
}
```

### Expert 模式

`balanced`

- 在每个 target rank 内，将 slots 尽量均匀分配给 local experts。

`hotspot`

- 将 slots 集中到 local experts 的一个子集上。
- 由 `hotspot,hotness=<ratio>` 或 `hotspot,active_experts=<count>` 控制。
- `active_experts` 是 per-rank 数量，不是全局数量。

`file:path/to/histogram.json`

- 从 JSON 加载显式 target-rank expert histogram。
- 这是精确 compute-shape 实验的 escape hatch。
- 文件格式固定为：

```json
{
  "schema": "v1",
  "ep_size": 4,
  "experts_per_rank": 16,
  "expert_histogram": [
    [16, 16, 16, 16],
    [16, 16, 16, 16],
    [16, 16, 16, 16],
    [16, 16, 16, 16]
  ]
}
```

## 规范化内部表示

每个 routing-control 请求都会被规范化为一个 canonical plan：

```python
@dataclass
class RoutingPlan:
    per_rank_num_tokens: List[int]
    dispatch_matrix: List[List[int]]
    expert_histogram: List[List[int]]
    seed: int
```

然后每个 MPI rank materialize：

```python
selected_experts: torch.Tensor  # [local_num_tokens, top_k], int32
selected_scales: torch.Tensor   # [local_num_tokens, top_k], v1 固定为 uniform 1/top_k
```

生成的 `selected_experts` 必须满足：

- Shape 是 `[local_num_tokens, top_k]`。
- Expert ids 位于 `[0, num_experts)`。
- 同一个 token 内的 expert ids 应该唯一；如果显式请求的矩阵违反基本合法性约束（例如要求单 token 重复同一个 expert），应该 reject 该 case，而不是静默生成非法 routing。
- `num_experts % moe_ep_size == 0`.
- `top_k <= num_experts`.
- 对于要把一个 token 的所有 top-k experts 放到单个 target rank 上的 patterns，要求 `top_k <= experts_per_rank`。
- 显式提供 `per_rank_num_tokens` 时，它的长度必须等于 `world_size`。
- 对 global-token sweeps，`sum(per_rank_num_tokens)` 必须等于 `num_tokens`。
- 输入 `dispatch_matrix` 使用 slot traffic；对每个 source row，必须满足
  `sum(dispatch_matrix[src]) == per_rank_num_tokens[src] * top_k`。
- 输入 `expert_histogram` 的全局总和必须等于 `sum(per_rank_num_tokens) * top_k`；每个 target rank 的 histogram sum 应等于 `sum_src dispatch_matrix[src][target]`。

### `comm_method` 与 `comm_pattern` 的性能耦合

`comm_pattern` 控制的是 selected experts 对应的 routing 形态，但不同 communication backend 对这个形态的敏感度不同。benchmark 输出应该同时记录 requested `comm_pattern` 和 actual communication path。

| actual communication path | 对 `comm_pattern` 的预期敏感度 |
|---|---|
| `NVLINK_TWO_SIDED` / MNNVL two-sided | 对 receiver hotspot、pair hotspot、off-diagonal traffic 敏感；prepare 阶段也依赖 `token_selected_experts`、`top_k` 和 `runtime_max_tokens_per_rank`。 |
| `NVLINK_ONE_SIDED` | 对 receiver hotspot 和 pair hotspot 敏感，但 workspace / payload layout 与 two-sided 不同。 |
| `DEEPEP` / `DEEPEPLOWLATENCY` | 对 target expert 分布和 token volume 敏感，具体瓶颈可能体现为 dispatch buffer / handle 路径。 |
| allgather post-quant path | 即使 `comm_pattern=local_only`，也可能仍然 allgather 全量 hidden states；此时 pattern 对通信量的影响弱于 all-to-all path。 |
| no-comm / TEP non-DP local path | `comm_pattern` 不代表真实跨 rank 通信，只影响 native routing / compute 形态。actual output 应设置 `effective_src_axis="collapsed"` 或给出 warning。 |

### EPLB 交互

v1 中 routing control 与 `eplb_mode != off` 互斥并 reject。

原因是 EPLB 会通过 `_load_balancer_route(token_selected_experts, self.use_dp)` 将 expert ids 映射到 slots。此时用户请求的 expert-domain `dispatch_matrix` 不一定仍然对应同一个 target rank。后续如果要支持 EPLB，应新增 slot-domain 语义，并在 slot routing 之后采集 observed matrix。

## Per-Rank Tokens 和 Chunking

`per_rank_num_tokens` 是一等 workload 轴。它控制 source-rank local work、`max(all_rank_num_tokens)`，并可能改变依赖 runtime token counts 的下游执行路径。

benchmark 应该支持：

```bash
--per_rank_num_tokens 128,128,512,128
```

以及 JSON config：

```json
"workload": {
  "num_tokens": [896],
  "per_rank_num_tokens": [128, 128, 512, 128]
}
```

开启 routing control 时，actual output 应包含：

```json
"token_shape": {
  "per_rank_num_tokens": [128, 128, 512, 128],
  "max_num_tokens_per_rank": 512,
  "num_chunks_observed": 2,
  "use_dp_padding": false,
  "chunking_warning": "per-rank token skew changed the observed chunk count"
}
```

Canonical `dispatch_matrix` 描述的是聚合 workload。如果 backend 将 work 切成多个 chunks，聚合矩阵不保证每个 chunk 内都有同样的分布。后续实现可以在 `num_chunks_observed > 1` 时报告 per-chunk matrices；在此之前，发生 chunking 的 rows 应该携带 warning。

## Distinct Token Traffic vs Expert-Slot Traffic

通信指标中有两个相关但不同的计数：

```text
slot_traffic[src][dst] = number of selected expert slots owned by dst
token_traffic[src][dst] = number of distinct tokens that need to be sent to dst
```

如果一个 token 在同一个 target rank 上选择了两个 experts，`slot_traffic` 增加 2，而 `token_traffic` 可能只增加 1。通信 payload 更接近 token traffic；expert compute 更接近 slot traffic。

输入 `dispatch_matrix` 文件使用 **slot traffic**。benchmark 应该在 actual output 中同时记录 slot 和 token traffic：

```json
"routing_control": {
  "observed_token_dispatch_matrix": [[...]],
  "observed_slot_dispatch_matrix": [[...]],
  "observed_expert_histogram": [[...]]
}
```

这不是 open question：输入文件使用 slot traffic，token traffic 从 materialized `selected_experts` tensor 派生。

## TRTLLM TEP 路径注意事项

TRTLLM TEP 比较特殊，因为默认路径会把 routing 融合进 MoE kernel。在 `TRTLLMGenFusedMoE.forward_impl` 中，当 post-quant communication 关闭时，routing 可能不会调用 `routing_method.apply()`；相反，`router_logits` 会传给 `run_moe`，由 backend kernel 在内部执行 routing。

现有 layerwise benchmark 通过 patch `TRTLLMGenFusedMoE.run_moe` 处理 TEP-like cases，因为只 patch `routing_method.apply()` 不够。

`enable_perfect_router` 必须和 routing control 协调。现有 benchmark 已经在 balanced case 使用 production perfect-router planner。Routing control 不应该引入第二套 balanced-logits 实现与它互相覆盖。

| routing_mode | comm_pattern | enable_perfect_router |
|---|---|---|
| `native` | `balanced_alltoall` | `true` |
| `native` | non-balanced patterns | `false` |
| `forced` | any pattern | `false` |
| `auto` | `balanced_alltoall` | `true` |
| `auto` | non-balanced patterns | `false` |

当 `native` / `auto` mode 用于非 balanced pattern 时，benchmark 应该尝试从 requested plan 反推 `router_logits`。如果 routing type 无法精确表达该 plan，benchmark 应选择最接近的合法 projection，继续运行，并在 actual output 中明确记录 warning 和偏差。它不能静默回退到 perfect-router balanced logits。

### Native Logits Realization 约束

不同 routing type 从目标 `selected_experts` 反推 `router_logits` 的能力不同。`native` / `auto` mode 必须把这个能力显式记录在 actual output 中：

| routing type | 反推能力 | 约束 |
|---|---|---|
| `Default` | `exact` for ids | 目标 experts 给高 logits，其它 experts 给低 logits；scales 来自 full softmax，不能任意指定。 |
| `Renormalize` | `exact` | 目标 experts 给高 logits；top-k 内 scales 可用 logits 差值近似/精确控制。 |
| `RenormalizeNaive` | `exact` | 数学上接近 `Renormalize`；仍需避免 top-k tie。 |
| `SigmoidRenorm` | `exact` for ids | 目标 experts 给高 logits；scales 受 sigmoid range 约束。 |
| `Llama4` | `exact` only for top1 | 只能表达 top1 routing；多 expert / 多 target per token 的请求需要 project。 |
| `MiniMax2` | `exact_with_zero_bias` 或 `projected_with_bias` | 选择 top-k 使用 `sigmoid(logits) + bias`，final scales 使用无 bias sigmoid；保留真实 bias 时可能无法精确满足。 |
| `DeepSeekV3` | `projected_or_exact` | 受 `n_group` / `topk_group` 约束；每个 token 的 experts 必须落在可选择的 group 集合内。任意矩阵不保证可达。 |
| `SparseMixer` | `unsupported_initially` | iterative argmax + epsilon mask，第一版不承诺支持 native 反推。 |
| `Static` / `LoadBalanced` / `Unspecified` | `not_applicable` | 不是 native logits 反推路径；`Static` 属于 forced/supplied-topk，`LoadBalanced` 忽略 logits。 |

如果 realization 不是 `exact`，actual output 必须记录：

```json
"routing_realization": {
  "status": "projected",
  "reason": "DeepSeekV3 group constraints require projection",
  "requested_slot_dispatch_matrix": [[...]],
  "observed_slot_dispatch_matrix": [[...]],
  "max_abs_slot_error": 3,
  "max_relative_slot_error": 0.004
}
```

### Projection 接口和目标函数

native logits planner 使用明确的请求 / 响应接口：

```python
@dataclass
class RoutingProjectionRequest:
    plan: RoutingPlan
    routing_method: BaseMoeRoutingMethod
    num_experts: int
    top_k: int
    moe_ep_size: int
    ep_rank: int
    device: torch.device
    dtype: torch.dtype
    projection_policy: str  # "project" | "reject"

@dataclass
class RoutingProjectionResult:
    router_logits: torch.Tensor
    realization_status: str  # "exact" | "projected" | "rejected"
    reason: str
    observed_slot_dispatch_matrix: List[List[int]]
    observed_token_dispatch_matrix: List[List[int]]
    observed_expert_histogram_summary: Dict[str, Any]
    max_abs_slot_error: int
    max_relative_slot_error: float
```

Projection 目标函数按优先级处理：

1. 基本合法性约束必须满足：expert id range、token 内不重复、matrix row sum、histogram sum。
2. 优先匹配 `dispatch_matrix`，因为它直接决定通信 workload。
3. 在不增加 dispatch error 的前提下，尽量匹配 `expert_histogram`。
4. 对 DeepSeekV3 这类 grouped routing，如果 group 约束无法同时满足 comm 与 expert 目标，优先保留 comm matrix 的 column / pair 形态，再最小化 expert histogram 偏差。
5. 如果 `projection_policy=reject` 且无法 exact，则该 case skipped，并将原因写入 `skip_reason`。

`comm_pattern` 和 `expert_pattern` 冲突时，以 `comm_pattern` 为第一优先级。`expert_pattern` 只在实际到达某个 target rank 的 slots 内生效；如果某个 target rank 没有足够 slots，expert histogram 会退化，并在 actual warnings 中说明。

因此设计上有两条执行路径：

### Native Logits Path

```text
router_logits -> TRTLLM fused routing -> routing metadata -> grouped GEMM
```

属性：

- 测量原生 TRTLLM TEP routing 和 MoE kernel 行为。
- 只支持能由 logits 表达的 routing patterns。
- DeepSeekV3 grouped routing 可能因为 `n_group` 和 `topk_group` 约束而 project 或 reject 请求的 pattern。
- 由于 top-k tie breaking、dtype、grouped routing 约束或 routing bias，observed traffic matrix 可能偏离 requested pattern。当 max-min 或百分比偏差超过很小的 tolerance 时，actual output 必须包含 warning。

### Forced Supplied Top-k Path

```text
selected_experts + selected_scales -> TRTLLM supplied-topk routing path -> routing metadata -> grouped GEMM
```

属性：

- 可以精确控制 routing workload。
- 会跳过 fused routing path 里的 scoring / top-k 部分。
- 仍然会执行 supplied top-k ids 之后的 metadata generation、permutation、grouped-GEMM planning 和 compute。
- 必须在输出中标记，避免被当成 native TEP fused top-k scoring 结果比较。
- 对 TRTLLM TEP，这条路径不是默认 TEP fused scoring path；它只适合用户明确要求“精确控制 `topk_ids`”的实验。
- 对 DEP / post-quant-comm path，forced 可以通过 patch `routing_method.apply()` 或 `StaticMoeRoutingMethod` 生效；对 TRTLLM TEP，则必须 patch `TRTLLMGenFusedMoE.run_moe`，让 `topk_ids` / `topk_weights` 传入 backend kernel。

## Selected Scale Distribution

v1 不开放 `--scale_distribution`。supplied-topk mode 固定使用 `uniform` scales，也就是每个 selected expert 使用 `1 / top_k`。这和 layerwise benchmark helper 一致，可以减少 CLI 维度和 dashboard 搜索空间。

如果未来 low-precision combine 或数值稳定性研究需要控制 scales，再扩展 `scale_distribution`。在此之前，它只作为 actual output 中的 metadata 存在，不作为用户输入。

actual output 应包含：

```json
"selected_scales": {
  "distribution": "uniform",
  "dtype": "torch.bfloat16",
  "seed": 1234
}
```

第一版实现应默认使用 `uniform`，并记录选择的 dtype。Backend-specific dtype 要求应该显式处理，不应该通过对 dummy logits 调用 `routing_method.apply()` 来推断，因为这可能触发 CUDA routing kernels 或修改 routing state。

## Output Schema 补充

增加 `SCHEMA_VERSION`，并在每一行 result 中加入 routing-control block：

```json
"routing_control": {
  "requested": {
    "routing_mode": "native",
    "projection_policy": "project",
    "comm_pattern": "receiver_hotspot,hotness=0.75,rank=0",
    "expert_pattern": "hotspot,active_experts=2",
    "per_rank_num_tokens": [256, 256, 256, 256],
    "seed": 1234
  },
  "actual": {
    "routing_path": "logits_projected",
    "routing_realization": {
      "status": "projected",
      "reason": "DeepSeekV3 group constraints require projection",
      "max_abs_slot_error": 3,
      "max_relative_slot_error": 0.004
    },
    "enable_perfect_router": false,
    "effective_src_axis": "dp_rank",
    "max_num_tokens_per_rank": 256,
    "num_chunks_observed": 1,
    "use_dp_padding": false,
    "observed_dispatch_matrix_summary": {
      "row_sums": [2048, 2048, 2048, 2048],
      "col_sums": [6144, 682, 683, 683],
      "off_diagonal_ratio": 0.75,
      "max_abs_slot_error": 3,
      "matrix_dump_path": null
    },
    "observed_expert_histogram_summary": {
      "min": 0,
      "max": 512,
      "active_experts": 8
    },
    "selected_scales": {
      "distribution": "uniform",
      "dtype": "torch.bfloat16",
      "seed": 1234
    },
    "warnings": []
  }
}
```

默认只在 row 里输出 summary，避免大规模 sweep 时 JSON 膨胀。只有用户显式传 `--routing_dump_matrix` 时，才输出完整 `observed_token_dispatch_matrix`、`observed_slot_dispatch_matrix` 和 `observed_expert_histogram`，或者写入伴生 JSON/NPZ 文件并在 row 中记录路径。

建议的 `routing_path` 取值：

- `logits_native`
- `logits_projected`
- `supplied_topk_apply`
- `supplied_topk_run_moe`

建议的 `routing_realization.status` 取值：

- `exact`：requested plan 被 native logits 精确实现。
- `projected`：requested plan 无法精确表达，benchmark 使用最接近的合法 projection。
- `rejected`：requested plan 连 projection 都无法合法生成，应跳过该 case 并给出原因。
- `forced_exact`：用户显式选择 `routing_mode=forced`，直接供应 `topk_ids`。

建议的 `effective_src_axis` 取值：

- `dp_rank`：source 轴对应 data-parallel local token ownership。
- `ep_rank`：source 轴对应显式 benchmark mapping 后的 EP rank local routing。
- `collapsed`：所选模式下 source 轴没有明确意义，benchmark 应在 `warnings` 中解释原因。

## 实现草案

### Data Classes

在 `WorkloadSpec` 附近增加 `RoutingControlSpec` 和 `RoutingPlan`：

```python
@dataclass(frozen=True)
class RoutingControlSpec:
    routing_mode: str = "native"
    projection_policy: str = "project"
    comm_pattern: str = "balanced_alltoall"
    expert_pattern: str = "balanced"
    per_rank_num_tokens: Optional[Tuple[int, ...]] = None
    routing_dump_matrix: bool = False
    seed: int = 0
```

把 `WorkloadSpec` 更新成包含：

```python
num_tokens: int
routing_control: RoutingControlSpec
```

过渡期保留旧 knobs 作为 deprecated aliases：

- `rank_imbalance_ratio` 映射为 source-token-skew compatibility mode，并应在 actual output 中报告为 `per_rank_num_tokens`。
- `expert_hotspot_ratio` 映射为 `expert_pattern=hotspot,hotness=<value>`。
- `force_balance_patch` 等价于 `routing_mode=forced --comm_pattern balanced_alltoall --expert_pattern balanced`。这是为了保留旧路径的性能语义：历史上 forced balance patch 会绕过部分 native routing 行为。

### Matrix Generation

增加 benchmark-local helper 生成 canonical plan：

```python
def _build_dispatch_matrix(
    spec: RoutingControlSpec,
    per_rank_num_tokens: List[int],
    top_k: int,
    ep_size: int,
) -> List[List[int]]:
    ...

def _build_expert_histogram(
    spec: RoutingControlSpec,
    dispatch_matrix: List[List[int]],
    experts_per_rank: int,
) -> List[List[int]]:
    ...
```

`balanced_alltoall` 应均匀分配每一行。`receiver_hotspot,hotness=<ratio>,rank=<rank>` 应把每一行的一部分流量转移到指定 receiver。`pair_hotspot,hotness=<ratio>,src=<src>,dst=<dst>` 应把某个 source row 的一部分流量转移到指定 destination。`local_only` 应尽量把 traffic 放在对角线。`ring` 应优先发往 `(src + 1) % ep_size`。

第一版实现应把这些 helper 保持在 `bench_moe.py` 或 benchmark-local module 中。现有 layerwise routing patch helper 应复用于 execution injection，但新的 matrix / histogram planning 语义不应强行塞进 `tools/layer_wise_benchmarks/runner.py`，除非两个 benchmark 后续都需要同一套 public helper。

### Materialization

增加：

```python
def _materialize_selected_experts(
    plan: RoutingPlan,
    rank: int,
    top_k: int,
    experts_per_rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    ...
```

这个函数通过把 target ranks 翻译成 expert ids 来创建 `[local_num_tokens, top_k]` expert ids：

```python
expert_id = dst_rank * experts_per_rank + local_expert_id
```

它还应该从实际生成的 tensor 中计算 observed token dispatch matrix、slot dispatch matrix 和 expert histogram。

当 materialization 需要确定性选择时，使用 `RoutingControlSpec.seed` 与 rank 组合，确保重复运行能生成相同的 selected experts 和相同的 observed matrices。

### 复用已有 Patch Helpers

不要重复实现 layerwise benchmark 已经验证过的 patch 机制：

- 复用或适配 `make_balanced_routing_method`，用于会调用 `routing_method.apply()` 的路径。
- 复用或适配 `make_balanced_run_moe`，用于需要 supplied top-k ids 的 `TRTLLMGenFusedMoE` 路径。
- 复用 `make_forward_impl_check` 或等价检查，确保 benchmark 在没有走到预期 supplied-routing path 时直接失败。

新工作应该是 pattern planner 和 materializer，而不是另一套独立 routing patch wrapper。

### Execution Integration

在 `_run_one_candidate` 中：

1. 在 input generation 前 resolve `RoutingPlan`。
2. routing control active 时，使用 `plan.per_rank_num_tokens`，而不是只用 `_per_rank_tokens(...)`。
3. `routing_mode=native` 默认走 native logits planner，根据 routing type 尝试生成能实现 requested plan 的 `router_logits`。
4. 如果 native logits planner 不能精确满足 requested plan，则按 `projection_policy` 处理：`project` 生成最接近的合法 projection，`reject` 跳过该 case。
5. 只有用户显式选择 `routing_mode=forced` 时，才生成 `selected_experts` / `selected_scales` 并进入 supplied-topk 路径。
6. 对非 TRTLLM 或 post-quant-comm path，forced 模式可 patch `routing_method.apply()` 或切换为 `StaticMoeRoutingMethod`。
7. 对 `TRTLLMGenFusedMoE` TEP，forced 模式按 layerwise pattern patch `run_moe`，让 supplied ids 传到 `topk_ids`；这会跳过 fused scoring / top-k。
8. 只在 supplied-topk `run_moe` patch 中设置 `router_logits=None`。
9. 在 `_build_moe_module` 前，根据 `routing_mode` 和 `comm_pattern` 显式计算 `enable_perfect_router`。
10. 在 `RunResult` 里记录 `routing_path`、`routing_realization`、observed matrices、token-shape fields、scale metadata 和 warnings。
11. 如果 `eplb_mode != off` 且 routing control active，v1 直接 reject，并在 skip reason 中说明 EPLB slot remapping 暂不支持。

## 验证计划

增加 routing-plan helper 的 CPU-only tests：

- `balanced_alltoall` 生成近似均匀的 row / column distributions。
- `receiver_hotspot` 增加指定 receiver column。
- `pair_hotspot` 增加指定 source-target cell。
- `local_only` 最大化 diagonal entries。
- `ring` 主要把 source `i` 发往 `(i + 1) % ep_size`。
- `expert_pattern=hotspot` 限制每个 rank 的 active experts 数。
- Materialized `selected_experts` shape、dtype、range 正确，并且每个 token 内没有重复 expert id。
- 当 `top_k > experts_per_rank` 且请求 single-target case 时，给出清晰错误。
- Multi-rank realization：模拟 `ep_size=4`，分别为每个 rank materialize selected experts，聚合 observed slot dispatch matrix，并验证它和 planned matrix 完全一致。
- Per-rank token escape hatch：验证 `per_rank_num_tokens` 可以独立于 `comm_pattern` 控制 `max_num_tokens_per_rank`。
- Native balanced path：验证 `routing_mode=native` 且 `comm_pattern=balanced_alltoall` 时选择 `enable_perfect_router=true`。
- Projection policy：同一个不可精确实现的 DeepSeekV3 case，在 `projection_policy=project` 下继续运行并输出偏差，在 `projection_policy=reject` 下 skipped。
- Native projection path：对 DeepSeekV3 grouped routing 构造一个不能精确满足的 `comm_pattern`，验证 benchmark 继续运行、`routing_realization.status="projected"`，并输出 requested/observed matrix 偏差。
- Routing type coverage：分别覆盖 `Default`、`Renormalize`、`RenormalizeNaive`、`SigmoidRenorm`、`Llama4`、`MiniMax2`、`DeepSeekV3` 的 native logits realization 策略。
- Forced TEP path：验证 `routing_mode=forced`、TRTLLM TEP 时输出 `routing_realization.status="forced_exact"`，并明确标记不测 native fused scoring。
- EPLB 互斥：验证 routing control active 且 `eplb_mode=static|dynamic` 时直接 reject。
- Matrix file schema：验证 `slot_dispatch_matrix` row sum、`ep_size`、`schema`、`expert_histogram` sum 校验。

轻量检查：

```bash
python3 -m py_compile tests/microbenchmarks/bench_moe.py
git diff --check -- tests/microbenchmarks/bench_moe.py tests/microbenchmarks/*.md
```

GPU validation 单独做：

- `routing_mode=native`、`parallel_mode=TEP`、TRTLLM backend：验证 native fused routing 仍可运行。
- `routing_mode=native`、`parallel_mode=TEP`、DeepSeekV3：验证不可精确满足的 pattern 会 projection，并输出 warning。
- `routing_mode=forced`、`parallel_mode=TEP`、TRTLLM backend：验证 supplied-topk run_moe patch。
- `routing_mode=forced`、`parallel_mode=DEP`：验证 apply/static path。

## 已关闭的设计决策

1. 输入 `dispatch_matrix` 文件解释为 slot traffic。actual output 默认记录 summary；启用 `--routing_dump_matrix` 时记录完整 slot traffic 和 distinct-token traffic。
2. `rank_imbalance_ratio` 过渡期保留为 source-token skew 的兼容 alias，但新文档应该推荐 `per_rank_num_tokens`。
3. 如果 planning helpers 被抽出文件，helper tests 应放在 `tests/unittest/_torch/modules/moe/`；如果 helpers 仍是 `bench_moe.py` 私有实现，则可以放在 benchmark 旁边。
4. v1 中 routing control 与 `eplb_mode != off` 互斥。
5. v1 中 `scale_distribution` 固定为 `uniform`，不作为用户输入。

## 推荐默认值

- 默认 `routing_mode=native`。
- 默认 `projection_policy=project`。
- 默认 `comm_pattern=balanced_alltoall`。
- 默认 `expert_pattern=balanced`。
- 默认不输出完整矩阵，只输出 summary；需要完整矩阵时使用 `--routing_dump_matrix`。
- native 默认走 logits。无法精确满足时使用 closest projection 并报告 warning，不隐式切换到 forced。
- 精确 routing matrix 需要用户显式选择 `routing_mode=forced`。
- Supplied-topk TEP rows 必须明确标记为不测 native fused top-k scoring。
