<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MegaMoE CuteDSL NVFP4 Design

## MoE Design Gate

- Change area: backend, quantization-weights, MoEScheduler fused-communication execution, routing-factory, test-matrix, and MoE docs.
- Owner boundary: `ConfigurableMoE` remains the assembler. `FusedCommMoEScheduler` owns fused forward policy. `MegaMoECuteDsl` owns capability checks, activation quantization, workspace staging, and kernel launch. A new quantization method owns NVFP4 weight layout and post-load transforms.
- Main APIs touched: `MegaMoECuteDsl.can_implement`, `MegaMoECuteDsl.validate_configurable_moe`, `MegaMoECuteDsl.create_weights`, `MegaMoECuteDsl.load_weights`, `MegaMoECuteDsl.process_weights_after_loading`, `MegaMoECuteDsl.quantize_input`, `MegaMoECuteDsl.run_moe`, `create_moe.get_moe_cls`, `create_moe.create_moe_backend`, `FusedCommMoEScheduler._forward_chunk`, MoE test helpers, `test_moe_backend.py`, `test_moe_module.py`.
- Reference pattern: `MegaMoEDeepGemm` for `MoESchedulerKind.FUSED_COMM`; `CuteDslFusedMoE` for CuteDSL availability and autotuner-style runner boundaries; external `Sm100MegaMoEKernel` for the concrete kernel ABI.
- Guide sections used: `MOE_DEVELOPER_GUIDE.md` architecture, scheduler selection, fused-comm execution flow, file map, backend capability matrix, canonical examples, anti-patterns, and tests.
- Guide update needed: yes. Add `MegaMoECuteDsl` to the backend list, capability matrix, MegaMoE file map, canonical fused-comm examples, and tests.
- Refactor needed: yes. The existing fused scheduler has DeepGEMM-specific empty quantization assumptions; the better contract is to call `backend.quantize_input(...)` for zero-token chunks and require every fused-comm backend to return its own empty layout.
- Test plan: backend unit tests for capability, quantization, tactic selection, file importability, and single-rank `run_moe`; module tests for `ConfigurableMoE` factory/scheduler integration; multi-rank EP tests for fused-kernel lockstep; explicit skips for unsupported hardware, quantization, and missing CUDA 13 Cutlass DSL runtime symbols.

## Source Evidence

This design is grounded in these code contracts:

- `ConfigurableMoE` constructs a backend, creates a communication strategy, and later constructs a scheduler selected from the backend's `scheduler_kind`. This makes backend registration and `scheduler_kind` the correct integration point, not a backend-specific `forward_impl` branch in `ConfigurableMoE`.
- `MOE_DEVELOPER_GUIDE.md` states that `FUSED_COMM` backends use `FusedCommMoEScheduler`, skip host `Communication.dispatch` / `combine`, and still launch zero-token chunks so peer ranks can cross the in-kernel barrier.
- `FusedCommMoEScheduler._forward_chunk` currently performs routing, EPLB stat update, backend `quantize_input`, then backend `run_moe`. The new backend must fit this public backend contract.
- `MegaMoEDeepGemm` declares `scheduler_kind = MoESchedulerKind.FUSED_COMM`, performs static capability checks, and launches a fused dispatch/GEMM/activation/GEMM/combine kernel through `run_moe`.
- `MegaMoEDeepGemm` allocates symmetric activation workspace from `create_weights()`, not from `run_moe`, because symmetric-memory rendezvous is a build-time collective and can deadlock under PP/layer-skip or fail under CUDA graph capture if performed at runtime.
- `Sm100MegaMoEKernel.__call__` from `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/megamoe_kernel.py` requires NVFP4 activation, padded FP8 activation scales, `topk_idx`, `topk_weights`, local weights, symmetric combine output, local workspace, shared workspace, `SymBufferHost`, `max_active_clusters`, and stream.
- The external runner computes `group_hint` from `HardwareInfo().get_max_active_clusters(cluster_size)` when unset and passes `mma_tiler_mnk`, `cluster_shape_mnk`, `use_2cta_instrs`, `group_hint`, and `load_balance_mode` into `Sm100MegaMoEKernel`.
- The external kernel import graph spans both `moe_nvfp4_swapab/` and `src/`; all runtime files must be flattened into one TRT-LLM package and their imports rewritten together.
- The external fused FC12 kernel currently calls its epilogue with `alpha=1.0` and `norm_const=1.0`. TRT-LLM NVFP4 paths have per-expert alpha tensors and `fc2_input_scale`; correctness for real checkpoints requires extending the kernel ABI rather than silently assuming those values are one.
- PR `https://github.com/NVIDIA/TensorRT-LLM/pull/14354` is titled `[None][chore] Use CUDA 13 CUTLASS DSL package`; current `requirements.txt` pins `nvidia-cutlass-dsl[cu13]==4.5.0`.

## Goals

`MegaMoECuteDsl` adds an NVFP4 MegaMoE backend that uses the CuteDSL fused dispatch + FC1 + activation + FC2 + combine kernel from `/home/xxi/sc2/cutedsl_megamoe`.

The backend should:

- Be selected by `model_config.moe_backend == "MEGAMOE_CUTEDSL"`.
- Run through `ConfigurableMoE` and `FusedCommMoEScheduler`.
- Support NVFP4 weights and BF16 activations on SM100-family GPUs.
- Reuse the MoE backend lifecycle contract: capability check, weight creation/loading, activation quantization, `run_moe`.
- Bring the external CuteDSL kernel sources into `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4`.
- Use an autotuner-compatible runner whose tunable parameters include `mma_tiler_mnk`, `cluster_shape_mnk`, `use_2cta_instrs`, `group_hint`, and `load_balance_mode`.
- Preserve NVFP4 scale semantics for real checkpoints: `fc31_alpha`, `fc2_alpha`, and `fc2_input_scale` must reach the fused kernel via `MegaMoECuteDslWeightView.{fc31_alpha, fc2_alpha, fc2_input_scale}` and the runner ABI, or be explicitly rejected by `can_implement`.
- Keep FUSED_COMM invariants explicit: no host-side MoE communication, all EP ranks launch each chunk, and shared workspace is reset before the kernel relies on counters/signals.
- Allocate EP symmetric memory during the build-time weight lifecycle, before CUDA graph capture and before any non-lockstep forward path can occur.

Non-goals for the first implementation:

- No legacy standalone `forward` path.
- No host `Communication.dispatch` / `combine` layering around the MegaMoE kernel.
- No dynamic per-rank compile shape. Runtime token counts may differ per rank, including zero-token chunks, but all ranks must use the same static `max_tokens_per_rank` buffer shape for a compiled kernel.
- No GPT-OSS-style SwiGLU support unless the kernel explicitly adds that path.
- No dynamic-shape MegaMoE kernel compile path; the external kernel currently requires `static_expert_shape`.

Hard gates before product use:

- Kernel ABI extension is a prerequisite for real NVFP4 checkpoints. Until the ported kernel accepts `fc31_alpha`, `fc2_alpha`, and `fc2_input_scale` / `norm_const`, this backend is limited to checkpoints whose per-expert / per-layer scales are all equal to 1. The checkpoint-value check belongs in `post_load_weights()` (or `process_weights_after_loading()`), not in `can_implement()`; `can_implement` is a static capability query that does not see checkpoint tensor values.
- NVSHMEM provider design is a prerequisite for multi-rank execution. The implementation must settle dependency ownership, process-level init/finalize, interaction with the existing DeepEP NVSHMEM runtime, and PP/layer-skip fallback before backend wiring proceeds.
- Production memory and compile-time budgets must be measured before enabling this backend as a default path. Form A is a correctness-first candidate, not automatically the v1 production default.
- **Always-pad-to-`max_tokens_per_rank` launch contract**. `_dispatch_prep` round 3 in `dispatch_kernel.py` writes per-(expert, rank) advertise cards at stride `num_tokens * num_topk`, but the corresponding `src_token_topk_idx` symmetric buffer is allocated at stride `max_tokens_per_rank * num_topk` (see `_RegionSpec("src_token_topk_idx", ..., (num_experts_per_rank, world_size, max_slot))` in `megamoe_kernel.py`). The two strides only agree when every launch passes `num_tokens == max_tokens_per_rank` as the constexpr leading dim. The backend must therefore always launch the kernel with full-size `max_tokens_per_rank` activation / SF / topk_weights / topk_idx / combine_output tensors and mask the live region with `topk_idx[live_T:] = -1` (which `_dispatch_prep` skips at line `if expert_id >= Int32(0):`). Slicing to `[:num_tokens]` before the op call is a multi-rank silent-corruption bug.
- **Activation SF row width must be `round_up(ceil(hidden_size / 16), 4)` FP8 bytes**, not `hidden_size // 16`. The kernel computes `sf_uint32_per_token = ceil(hidden / 64) * 4` bytes per row; for hidden sizes that are 32-aligned but not 64-aligned (1568, 1632, 2080, ...) the naive `hidden // 16` is 2 bytes short and causes the TMA load to read uninitialized bytes. The backend's symmetric / local SF allocations and `quantize_input` output must both use the helper `megamoe_activation_sf_bytes_per_row(hidden_size)`. Note that `torch.ops.trtllm.fp4_quantize(..., is_sf_swizzled=False)` returns LINEAR layout `(rows, ceil(hidden/16))` with no column pad, so `quantize_input` must pad the SF tail columns before returning.
- **Custom-ops import guard must be stricter than `IS_CUTLASS_DSL_AVAILABLE`**. The probe in `cute_dsl_utils.py` only imports `cutlass` and `cutlass.cute`; a half-installed cutlass-dsl wheel can still satisfy that yet fail on `cutlass.torch`, `cutlass._mlir`, the `cute_nvgpu` MMA atoms used by `kernel_fc12.py`, or the symm-memory adapter symbols used by `sym_buffer.py`. The op registration module exports its own `IS_MEGAMOE_OP_AVAILABLE` flag, set by a strict try/except that imports every symbol the op needs; `tensorrt_llm._torch.custom_ops.__init__` only re-exports the op when that flag is `True`. The factory must accept that a host can have `IS_CUTLASS_DSL_AVAILABLE == True` but `IS_MEGAMOE_OP_AVAILABLE == False` and fall back to CutlassFusedMoE with a clear warning instead of crashing.
- **Dynamic EPLB is rejected at validate time** (see EPLB status below). Static EPLB is allowed.

## Proposed File Layout

Runtime backend:

- `tensorrt_llm/_torch/modules/fused_moe/mega_moe/mega_moe_cute_dsl.py`
- `tensorrt_llm/_torch/modules/fused_moe/mega_moe/__init__.py`
- `tensorrt_llm/_torch/modules/fused_moe/quantization.py`
- `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`
- `tensorrt_llm/_torch/modules/fused_moe/moe_scheduler.py`

CuteDSL kernel package:

All runtime kernel files should be flattened into `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/`. The upstream project splits code between `moe_nvfp4_swapab/` and `src/`; leaving that split in TRT-LLM would make imports fragile and would require adding a top-level `src` package. The port must rewrite imports to package-relative imports inside `mega_moe_nvfp4`.

| Source file | Target file | Import rewrite |
|---|---|---|
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/megamoe_kernel.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/megamoe_kernel.py` | `from .kernel_fc12 ...`; `from .moe_utils ...`; `from .dispatch_kernel ...`; `from .iket_compat ...` |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/kernel_fc12.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/kernel_fc12.py` | `from .epilogue ...`; `from .fc1_fc2_fuse_sched ...`; `from .custom_ext ...`; `from .megamoe_constants ...`; `from .moe_utils ...`; `from .iket_compat ...` |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/custom_ext.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/custom_ext.py` | `from .fc1_fc2_fuse_sched ...`; `from .moe_utils ...`; `from .moe_persistent_scheduler ...` |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/fc1_fc2_fuse_sched.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/fc1_fc2_fuse_sched.py` | `from .moe_persistent_scheduler ...`; `from .moe_utils ...`; `from .iket_compat ...` |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/moe_persistent_scheduler.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/moe_persistent_scheduler.py` | `from .iket_compat ...` |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/moe_utils.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/moe_utils.py` | keep package-local |
| subset of `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/runner_fc12.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/blocked_scale.py` | extract `to_blocked`, `from_blocked`, and byte-reinterpretable stack helpers only; no runner/test harness |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/epilogue.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/epilogue.py` | `from .contract ...`; `from .fc1_fc2_fuse_sched ...`; `from .megamoe_constants ...`; `from .iket_compat ...` |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/contract.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/contract.py` | keep package-local |
| `/home/xxi/sc2/cutedsl_megamoe/moe_nvfp4_swapab/megamoe_constants.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/megamoe_constants.py` | keep package-local |
| `/home/xxi/sc2/cutedsl_megamoe/src/config.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/config.py` | keep only runtime constants and helpers needed by `dispatch_kernel.py`; remove bootstrap/test-only config if unused |
| `/home/xxi/sc2/cutedsl_megamoe/src/dispatch_kernel.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/dispatch_kernel.py` | `from .config ...`; `from .grid_sync ...`; `from .ptx_helpers ...`; `from .sf_swizzle ...`; `from .iket_compat ...` |
| `/home/xxi/sc2/cutedsl_megamoe/src/grid_sync.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/grid_sync.py` | keep package-local |
| `/home/xxi/sc2/cutedsl_megamoe/src/ptx_helpers.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/ptx_helpers.py` | keep package-local |
| `/home/xxi/sc2/cutedsl_megamoe/src/sf_swizzle.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/sf_swizzle.py` | keep package-local |
| `/home/xxi/sc2/cutedsl_megamoe/src/sym_buffer.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/sym_buffer.py` | keep package-local; imports runtime Cutlass DSL adapter APIs |
| `/home/xxi/sc2/cutedsl_megamoe/src/iket_compat.py` | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/iket_compat.py` | keep package-local |
| new file | `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/__init__.py` | `__all__ = ["Sm100MegaMoEKernel", "SymBufferHost", "to_blocked", "from_blocked", "SfPaddingBlock", "Fc1GateUpInterleave"]` |

Test and documentation updates:

- `tests/unittest/_torch/modules/moe/moe_test_utils.py`
- `tests/unittest/_torch/modules/moe/quantize_utils.py`
- `tests/unittest/_torch/modules/moe/test_moe_backend.py`
- `tests/unittest/_torch/modules/moe/test_moe_module.py`
- `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md`

The external `mega_runner.py`, reference implementation, and bootstrap utilities should not be copied into the runtime package. Their useful host-side logic should be rewritten into TRT-LLM backend tests or runner helpers.
The only `runner_fc12.py` code allowed in runtime is the small blocked-scale helper subset listed in the table.

## End-to-End Call Chain

Expected runtime flow:

```text
create_moe()
  -> resolve_moe_cls()
  -> get_moe_cls(model_config.moe_backend="MEGAMOE_CUTEDSL")
  -> ConfigurableMoE(...)
  -> ConfigurableMoE._create_and_sync_backend(...)
  -> create_moe_backend(MegaMoECuteDsl, ...)
  -> ConfigurableMoE._create_comm_strategy_auto() returns None for FUSED_COMM
  -> create_moe_scheduler(...) returns FusedCommMoEScheduler
  -> ConfigurableMoE.forward_impl(...)
  -> FusedCommMoEScheduler.forward(...)
  -> FusedCommMoEScheduler._forward_chunk(...)
  -> MegaMoECuteDsl.quantize_input(...)
  -> MegaMoECuteDsl.run_moe(...)
  -> torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(...)
  -> Sm100MegaMoENvfp4Runner(...)
  -> Sm100MegaMoEKernel.__call__(...)
```

The scheduler remains the only forward-policy owner. The backend does not split chunks, advance `repeat_idx`, run EPLB CPU migration, or select host communication.

## Backend API Design

### `class MegaMoECuteDsl(MoE)`

Public class attributes:

```python
class MegaMoECuteDsl(MoE):
    scheduler_kind = MoESchedulerKind.FUSED_COMM
    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})
```

The SM gate should use the same SM100-family helper as `MegaMoEDeepGemm` (`is_sm_100f`). Do not add a second `_SUPPORTED_SM_VERSIONS` fact source; future SM-family expansion should happen in the helper.

### `can_implement`

Signature:

```python
@classmethod
def can_implement(
    cls,
    quant_algo: Optional[QuantAlgo],
    dtype_activation: torch.dtype = torch.bfloat16,
    swiglu_gptoss_style: bool = False,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
```

Checks:

- `IS_CUTLASS_DSL_AVAILABLE` must be true.
- CUDA 13 Cutlass DSL capability must be probed by importing the symbols the kernel actually uses. Do not rely only on pip package metadata, because container installs may not expose reliable package extras. A practical probe should import `cutlass.cute`, `cutlass.torch`, the required `cutlass._mlir` APIs used by `sym_buffer.py`, and any `cute_nvgpu` / async-copy APIs used by the ported kernel.
- GPU must be SM100-family.
- `quant_algo == QuantAlgo.NVFP4`.
- `dtype_activation == torch.bfloat16`.
- `swiglu_gptoss_style` must be false.
- `hidden_size` must be a positive multiple of 32. If the activation SF output is stored as FP8 bytes, the padded SF column count must be `round_up(ceil(hidden / 16), 4)`.
- `intermediate_size` is TRT-LLM's down-projection width. It must be positive and `intermediate_size % 16 == 0`, which is equivalent to the external kernel's gate/up width constraint `expand_intermediate_size_per_partition % 32 == 0`.

`can_implement()` does NOT check checkpoint tensor values. The "all alphas == 1 fallback" until the kernel ABI extension lands is enforced in `post_load_weights()` / `process_weights_after_loading()`; see `NVFP4 scale and alpha ABI` below.

Return `(False, reason)` for every unsupported combination. Do not raise from `can_implement` except for unexpected internal errors.

The CUDA 13 Cutlass DSL probe must not be served by the existing global
`IS_CUTLASS_DSL_AVAILABLE` flag alone. That flag only imports `cutlass` and
`cutlass.cute` (see `tensorrt_llm/_torch/cute_dsl_utils.py`), so it returns
`True` on environments where the MegaMoE-required symbols
(`cutlass.torch.from_dlpack`, the `cutlass._mlir` APIs used by `sym_buffer.py`,
the `cute_nvgpu` MMA atoms used by `kernel_fc12.py`, and the async-copy
helpers used by `dispatch_kernel.py`) are missing. Add a backend-local probe
that imports those exact attributes and caches the result.

### `__init__`

Signature should mirror other `MoE` backends:

```python
def __init__(
    self,
    *,
    routing_method: BaseMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: Optional[torch.dtype] = None,
    reduce_results: bool = False,
    model_config: ModelConfig = ModelConfig(),
    aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    apply_router_weight_on_input: bool = False,
    layer_idx: Optional[int] = None,
    init_load_balancer: bool = True,
    without_comm: bool = False,
    activation_type: ActivationType = ActivationType.Swiglu,
) -> None:
```

Initialization responsibilities:

- Call `super().__init__`.
- Reject `apply_router_weight_on_input=True`.
- Save `mapping`, `rank`, `ep_size`, `ep_rank`, `num_slots`, `hidden_size`, and `intermediate_size`.
- Use `self.expand_intermediate_size_per_partition` inherited from `MoE` for the gate/up width. Do not introduce a second field that can drift from `intermediate_size_per_partition * intermediate_size_expand_ratio`.
- Select `NVFP4MegaMoECuteDslMethod` as `self.quant_method`.
- Resolve the EP process group during construction, following `MegaMoEDeepGemm._resolve_ep_pg`. Resolving a group during forward would be a non-lockstep collective.
- Initialize a lazy `MegaMoECuteDslNvfp4Runner`.
- Initialize symmetric-buffer state to `None`. Allocate symmetric EP buffers in `create_weights()` after `ConfigurableMoE` has synchronized EPLB-derived attributes such as `num_slots`.
- Keep local CUDA workspaces lazy and shape-cached; those do not perform host-side symmetric-memory rendezvous and can be allocated from `run_moe`.
- Accept `aux_stream_dict` for `create_moe_backend` signature uniformity, but ignore it. `FUSED_COMM` kernels must not use multi-stream chunk overlap.
- Respect `model_config.skip_create_weights_in_init`.

Backend-specific configuration fields should be pulled from a narrow model config extension or environment-backed internal config. The first implementation can use constants that match the external runner defaults and expose a follow-up to add Pydantic config fields:

```python
mma_tiler_mnk = (128, 128, 256)
cluster_shape_mnk = (1, 1, 1)
use_2cta_instrs = False
group_hint = None
load_balance_mode = "static"
```

Validation of these fields belongs in a small dataclass, not scattered inside `run_moe`.

### `validate_configurable_moe`

Signature:

```python
def validate_configurable_moe(self, moe: "ConfigurableMoE") -> None:
```

Checks:

- `moe.comm is None`, because `FUSED_COMM` backends must not use external communication.
- `moe.mapping.moe_tp_size == 1` for v1.
- `moe.mapping.tp_size == 1` for v1, unless the copied kernel is proven TP-safe.
- `moe.num_slots % moe.mapping.moe_ep_size == 0`.
- If `moe.use_dp and moe.parallel_size > 1`, require `moe.mapping.moe_ep_size == moe.parallel_size`. The fused kernel exchanges only inside the EP group; ADP > EP would require an outer allgather before the kernel and reducescatter after it.
- `moe.routing_method.experts_per_token <= 13` for v1 unless implementation adds a kernel-side capacity proof or tests for larger top-k values. This matches the current external coverage boundary and avoids documenting an unproven `<=16` limit.
- `moe.use_dp_padding` does not affect fused-comm behavior; scheduler strips ADP padding before chunking.
- `moe.moe_max_num_tokens > 0`.

### Weight lifecycle APIs

The backend should expose all standard weight lifecycle hooks and delegate layout-specific work to `NVFP4MegaMoECuteDslMethod`:

```python
def create_weights(self) -> None
def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None
def post_load_weights(self) -> None
def process_weights_after_loading(self) -> None
def pre_reload_weights(self) -> None
```

Rules:

- Weight lifecycle remains backend-owned at the API boundary.
- The quantization method owns parameter registration, checkpoint loading details, scale conversion, `to_blocked`/swizzle transforms, and EPLB fix-up registration.
- Prefer one concrete post-load finalization owner for transformed weights and scales. Match `MegaMoEDeepGemm` by using `post_load_weights()` for the quantization method; if `process_weights_after_loading()` remains on the backend for interface compatibility, it must be idempotent and must not repeat the same tensor transforms.
- `create_weights()` must allocate or fetch symmetric EP buffers after the backend has the final `num_slots`, `expert_size_per_partition`, and `routing_method.experts_per_token`.
- `create_weights()` order must match the MegaMoEDeepGemm safety pattern:
  1. `_alloc_symm_buffer()` or equivalent build-time symmetric allocation;
  2. `_get_quant_method()`;
  3. `self.quant_method.create_weights(self)`;
  4. mark `_weights_created = True`.
  This lets quantization code see any symmetric views it needs while preserving the build-time rendezvous window.
- If dynamic EPLB is not fully implemented in v1, declare `EplbSupportStatus.NOT_VERIFIED` with concrete missing items: transformed `fc1_weight_sf` / `fc2_weight_sf` staging buffers, fix-up functions for `to_blocked` outputs, and tests proving migrated slots keep the same blocked scale layout. Do not use a vague "not verified" skip reason.

### NVFP4 scale and alpha ABI

The external kernel is not numerically complete for TRT-LLM NVFP4 checkpoints until its ABI carries the same scale data as existing NVFP4 backends. Treat this as an independent kernel-side work item before product backend integration, not as a small backend glue task. V1 must choose the kernel-ABI path:

- Add `fc1_alpha` / `fc31_alpha` input for the FC1 global scale used by the first grouped GEMM path.
- Add `fc2_alpha` input for the FC2 global scale used by the second grouped GEMM path.
- Add `fc2_input_scale` as the FC1-output quantization `norm_const` equivalent.
- Add these inputs to the ported copy under `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/` by threading them through `Sm100MegaMoEKernel.__call__`, `kernel_fc12.py`, and `epilogue.py`.
- Expect this to touch multiple CuteDSL files (`megamoe_kernel.py`, `kernel_fc12.py`, `epilogue.py`, and any contract/scheduler metadata that needs the new runtime values). This should be tracked as its own implementation phase or PR.
- Keep GPT-OSS SwiGLU `alpha` separate from NVFP4 per-expert alpha. The current epilogue `alpha` controls SwiGLU math; it is not a substitute for `fc31_alpha` or `fc2_alpha`.
- If the kernel ABI extension is not present in the active build, `post_load_weights()` / `process_weights_after_loading()` must inspect `fc31_alpha`, `fc2_alpha`, and `fc2_input_scale` after checkpoint load and raise (or fall back) when any value differs from 1.0 within FP32 tolerance. `can_implement()` cannot enforce this because it does not see checkpoint tensors; the post-load gate is the correct owner.
- The post-load gate must read tensors via `.detach()` and `torch.allclose(..., torch.ones_like(...), atol=0)` so a single non-1 expert short-circuits the gate. Document this as an explicit v1 product limitation and not as a permanent product gate.

Do not fold these values into FP8 scale factors in v1. Folding would hide the semantic mismatch, changes FP8 dynamic range, and would need a separate numerical-error study. Do not apply a host-side post-correction path for form A as the primary solution because it does not naturally extend to form B in-kernel top-k reduction and cannot fix FC1 quantization scale semantics after the fact.

### `quantize_input`

Signature:

```python
def quantize_input(
    self,
    x: Union[torch.Tensor, Fp4QuantizedTensor],
    *,
    post_quant_comm: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

Contract:

- Input is BF16 `(num_tokens, hidden_size)`.
- Output activation is NVFP4 packed tensor in the layout consumed by `Sm100MegaMoEKernel`.
- Output activation SF is FP8 scale tensor in plain K-major layout, padded on the last dimension to a multiple of 4 FP8 scale values.
- Empty input returns empty NVFP4 activation and correctly padded empty SF shape.
- The method must accept `x.shape[0] == 0`. It should early-return correctly shaped empty tensors without launching a quantization kernel.

Implementation detail:

- Reuse existing TRT-LLM NVFP4 quantization utilities used by `CuteDslFusedMoE` where the output layout matches.
- `quantize_input` reads `self.fc31_input_scale`, the per-tensor FP32 input scale set up during the quantization method's post-load finalization, and passes it to `torch.ops.trtllm.fp4_quantize(...)`. This mirrors `CuteDslFusedMoE.quantize_input`.
- If `CuteDslFusedMoE` emits a scale layout that differs from MegaMoE's plain K-major SF contract, add an explicit conversion implementation in the backend or quantization method and test it directly. Do not assume this is a trivial view/reshape; if the layout requires swizzle or rebroadcast work, use a dedicated CUDA/Triton/custom-op path instead of a host-side copy.
- Add a unit test that compares activation SF bytes against the external runner's raw scale construction for representative hidden sizes.
- Do not return FP8 activations or packed UE8M0 int32 scales; those are DeepGEMM-specific and are currently hard-coded in the empty path of `FusedCommMoEScheduler`.

Needed scheduler refactor:

`FusedCommMoEScheduler` should always call `moe.backend.quantize_input(x_chunk_real)`, even when `num_tokens == 0`. This keeps the existing backend API surface and avoids a second empty-layout method. The scheduler should not synthesize backend-specific empty tensors.

### `run_moe`

Signature:

```python
def run_moe(
    self,
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    x_sf: Optional[torch.Tensor] = None,
    *,
    output_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> torch.Tensor:
```

Contract:

- `x` is quantized NVFP4 activation from `quantize_input`.
- `x_sf` is padded FP8 activation scale tensor.
- `token_selected_experts` is slot ID tensor in `[0, num_slots)`, shape `(T, top_k)`, from `FusedCommMoEScheduler` after optional EPLB remap.
- `token_final_scales` is FP32 top-k weight tensor, shape `(T, top_k)`.
- Return shape is `(T, hidden_size)` with dtype `output_dtype or self.dtype`.

Steps:

1. Validate shapes and dtypes.
2. Convert `token_selected_experts` to `torch.int64` at the `run_moe` boundary, because `FusedCommMoEScheduler` keeps routing output as `torch.int32` while the external `Sm100MegaMoEKernel` uses `topk_idx: Int64`.
3. Stage activation, activation SF, top-k weights, and combine output into preallocated symmetric memory reachable through `SymBufferHost`.
4. Keep `topk_idx` in local CUDA memory; the kernel contract reads it only on the local rank. Weights also remain local CUDA memory.
5. Allocate or reuse local CUDA workspace sized by `Sm100MegaMoEKernel.get_workspace_sizes()`. The shared symmetric workspace capacity must already exist from `create_weights()`.
6. Zero the symmetric shared workspace before each launch. This is a lockstep launch precondition: every EP rank must finish its local zero before entering the fused dispatch barrier. If using form-B top-k reduction later, also zero `combine_output`.
7. Invoke `MegaMoECuteDslNvfp4Runner.run(...)`.
8. For v1 form A, run `combine_output.sum(dim=1)` after the kernel returns and return `(T, hidden_size)`. Top-k weights are already applied inside the kernel's STG path.

The v1 design should require all EP ranks to use the same `max_tokens_per_rank` compile shape. Runtime token counts may differ per rank, including zero-token chunks, as long as each rank stages into the fixed symmetric buffer shape. If the kernel does not yet support masked rows, the backend should reject runtime shapes where a rank would need to expose invalid rows to peers; it should not reject merely because per-rank token counts differ.

Form A memory footprint must be checked before enabling large production shapes. The combine buffer size is only one part of the total:

```text
max_tokens_per_rank * top_k * hidden_size * sizeof(output_dtype)
```

For `max_tokens_per_rank=8192`, `top_k=8`, `hidden_size=7168`, BF16 form A needs about `8192 * 8 * 7168 * 2 = 0.88 GiB` per rank for `combine_output` alone. The implementation must also budget the local and shared workspaces returned by `Sm100MegaMoEKernel.get_workspace_sizes()`:

| Region | Memory class | Size driver |
|---|---|---|
| `combine_output` | symmetric or local single-rank | `max_tokens_per_rank * top_k * hidden_size * sizeof(output_dtype)` |
| staged activation | symmetric user-domain | `max_tokens_per_rank * hidden_size / 2` bytes for packed NVFP4 |
| staged activation SF | symmetric user-domain | `max_tokens_per_rank * round_up(ceil(hidden_size / 16), 4)` FP8 bytes |
| staged top-k weights | symmetric user-domain | `max_tokens_per_rank * top_k * sizeof(float)` |
| `shared_workspace` | symmetric workspace | at least `src_token_topk_idx`, receive counters, and NVLink barrier state from `_build_shared_region_specs()` |
| `local_workspace` / `l1_token_buffer` | local CUDA workspace | `pool_token_capacity * hidden_size / 2` bytes |
| `local_workspace` / `l1_sf_buffer` | local CUDA workspace | `pool_sf_capacity * sf_uint32_per_token * sizeof(int32)` |
| `local_workspace` / `fc1_output` | local CUDA workspace | `pool_token_capacity * intermediate_size_per_partition / 2` bytes for packed NVFP4 |
| `local_workspace` / `fc1_output_sf` | local CUDA workspace | `(pool_token_capacity + num_experts_per_rank * sf_padding_block) * sf_block_cols` FP8 bytes |

where `pool_token_capacity = round_up(world_size * max_tokens_per_rank * min(top_k, num_experts_per_rank) + num_experts_per_rank * (token_padding_block - 1), token_padding_block)`. The design must include a concrete production-shape budget before choosing form A as default. V1 can start with form A for correctness, but form B should be pulled into v1 if the target workload would otherwise exceed the memory budget.

## Runner And Autotuner Design

### MegaMoECuteDsl tactic representation

Tactic values must be compatible with TRT-LLM autotuner cache serialization, which requires both `json.dumps` / `json.loads` and `eval(repr(tactic))` round-trip. The original design proposed a JSON-friendly dict; the implementation upgrades that to a **JSON-friendly tuple** because the AutoTuner cache also requires the tactic to be **hashable** (dict is not). Tuple of (list-of-int, list-of-int, bool, int, str, bool) satisfies both.

Implementation lives in `tensorrt_llm/_torch/custom_ops/cute_dsl_megamoe_custom_op.py` (helper functions `DEFAULT_MEGAMOE_TACTIC`, `validate_megamoe_tactic`, `enumerate_megamoe_candidate_tactics`, `resolve_megamoe_group_hint`).

Tactic shape:

```python
DEFAULT_MEGAMOE_TACTIC = (
    [128, 128, 256],   # mma_tiler_mnk
    [1, 1, 1],         # cluster_shape_mnk
    False,             # use_2cta_instrs
    1,                 # resolved_group_hint (placeholder; runner resolves)
    "static",          # load_balance_mode
    False,             # use_bf16_redg
)
```

Validation (must match what the ported kernel enforces; see evidence below):

- `mma_tiler_mnk[0] in {128, 256}` and `mma_tiler_mnk[1] in {128, 256}`, matching `SupportedMmaTileM` / `SupportedMmaTileN` in `runner_fc12.py`.
- `mma_tiler_mnk[2] % (sf_vec_size * 4) == 0`, where `sf_vec_size == Nvfp4BlockSize == 16` (kernel_fc12.py `_validate_mma_*`).
- `use_2cta_instrs == (mma_tiler_mnk[0] == 256)`. This is a kernel law: `ImplDesc.__post_init__` and `Sm100SwapABSwigluFp4Fc12Kernel._validate_*` both raise if it is violated. After the swap, per-CTA M must be 128.
- `cluster_shape_mnk[2] == 1` (L axis).
- `cluster_shape_mnk[1] == 1` for v1. The kernel raises `NotImplementedError` for `cluster_n > 1`.
- `cluster_shape_mnk[0]` and `cluster_shape_mnk[1]` are powers of two, each at most 4, and `cluster_shape_mnk[0] * cluster_shape_mnk[1] <= 16`. Matches both `ImplDesc.__post_init__` and `Sm100SwapABSwigluFp4Fc12Kernel._validate_mma_*`.
- `cluster_shape_mnk[0] % (2 if use_2cta_instrs else 1) == 0`.
- For the first autotuner sweep, only enumerate combinations covered by the external `run_mega_tests.sh` (M01..M20). Anything outside that envelope must add a kernel-side smoke test before entering the sweep.
- `load_balance_mode in {"static", "atomic_counter"}`. The `"clc"` mode in `ImplDesc.__post_init__` is handled by a separate scheduler class and is not wired through the fused FC12 kernel.
- `resolved_group_hint > 0`. Resolve `group_hint=None` to `HardwareInfo().get_max_active_clusters(cluster_size)` before building the tactic/cache key so the cache key contains the actual integer used by `Sm100MegaMoEKernel`.

Fields that look tunable but are NOT in the tactic:

- `token_padding_block = EpilogueTokenTile` (= 64) and `sf_padding_block = SfPaddingBlock` (= 128) are passed to `Sm100MegaMoEKernel` as constants by upstream `mega_runner.py`. They are not autotuner dimensions; they live in `megamoe_constants.py` (ported) and are reused.
- `force_static_sched`, `clc_bundle_size`, `num_sched_stages`, `enable_static_expert_shape` are kernel-construction knobs in `ImplDesc` but are not part of the v1 tactic. Lock them to their upstream defaults (`force_static_sched=True`, `clc_bundle_size=None`, `num_sched_stages=None`, `enable_static_expert_shape=True`) so the compile-cache key stays small. Expose them only if a measured perf gap justifies adding them to the sweep.

### `class Sm100MegaMoENvfp4Runner(TunableRunner)`

The runner lives in `tensorrt_llm/_torch/custom_ops/cute_dsl_megamoe_custom_op.py` (not in the backend file) to match the boundary used by `CuteDslFusedMoE` / `cute_dsl_custom_ops.py`. The `MegaMoECuteDsl` backend only ever calls `torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell`, never the runner directly. The op invokes `AutoTuner.choose_one` and forwards to the runner with the selected tactic plus `peer_offsets` / `shared_workspace` as kwargs.

Primary APIs:

```python
def __init__(
    self,
    *,
    world_size: int,
    local_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    expand_intermediate_size_per_partition: int,
    max_tokens_per_rank: int,
    output_dtype: torch.dtype,
) -> None:
```

```python
def run(
    self,
    *,
    activation: torch.Tensor,
    activation_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    fc31_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    combine_output: torch.Tensor,
    local_workspace: torch.Tensor,
    shared_workspace: torch.Tensor,
    peer_rank_ptr_mapper_host: SymBufferHost,
    tactic: MegaMoECuteDslTactic,
) -> None:
```

Runner responsibilities:

- Convert torch tensors to `cute.Tensor` via `cutlass.torch.from_dlpack`.
- Mark dynamic leading dimensions where needed.
- Resolve `group_hint` before cache lookup: if user config has `group_hint=None`, compute `HardwareInfo().get_max_active_clusters(cluster_size)` and store the resolved integer in the tactic/cache key. `group_hint` is a kernel construction-time constant, so caching under `None` would be incorrect.
- Instantiate `Sm100MegaMoEKernel` with static expert shape:

```python
static_expert_shape = (
    num_experts_per_rank,
    expand_intermediate_size_per_partition,
    hidden_size,
)
```

- Compute workspace sizes through `kernel.get_workspace_sizes()`.
- Compile and cache `cute.compile(kernel, **compile_kwargs)`.
- Launch on `torch.cuda.current_stream()`.

Compile cache key should include:

- `world_size`
- `local_rank`
- `num_topk`
- `num_experts_per_rank`
- `hidden_size`
- `intermediate_size_per_partition`
- `expand_intermediate_size_per_partition`
- `max_tokens_per_rank`; this is the static compile shape and should normally equal the configured `moe_max_num_tokens`
- `output_dtype`
- resolved tactic dict, including `use_bf16_redg`

Autotuning dimensions:

- Primary key: `(max_tokens_per_rank, hidden_size, intermediate_size_per_partition, expand_intermediate_size_per_partition, num_experts_per_rank, experts_per_token, world_size)`.
- Candidate tactics: begin with the combinations from external `run_mega_tests.sh`, then narrow by `ImplDesc.__post_init__` constraints and compile-time budget.
- Objective: elapsed kernel time for `run_moe` excluding weight transforms and one-time compile.
- Fallback tactic: `{"mma_tiler_mnk": [128, 128, 256], "cluster_shape_mnk": [1, 1, 1], "use_2cta_instrs": False, "resolved_group_hint": <resolved>, "load_balance_mode": "static", "use_bf16_redg": False}`.

Compile budget:

- CuteDSL compile can dominate startup time. The first implementation should cap autotuner candidates to the small external-runner set, log compile time per tactic, and cache compiled kernels across layers with identical static shapes.
- Provide a warmup mode that compiles the selected/fallback tactic once per unique shape before serving traffic.
- Do not enable broad tactic sweeps by default for large models. Require an explicit profiling/autotune path for multi-candidate searches.

The compiled kernel should not be recompiled per chunk. Every EP rank should use the same static `max_tokens_per_rank` cache key for a layer; shorter chunks and zero-token chunks are runtime occupancy cases inside the same compiled shape.

## Symmetric Memory Design

The implementation should use the external kernel's `SymBufferHost` ABI and an NVSHMEM-backed provider for v1. This is the only provider currently known to expose exactly what the CuteDSL kernel needs: a local symmetric base address and per-peer byte offsets that `SymBufferHost` can pack into the generated host wrapper. The DeepGEMM `SymmBuffer` path is not selected for v1 because it hides peer offsets behind DeepGEMM APIs and does not expose the `SymBufferHost(base_addr, offsets, rank_idx, num_max_ranks)` payload required by the ported CuteDSL kernel.

Provider requirements:

- Use an NVSHMEM Python binding equivalent to the external runner's `nvshmem.core.tensor` and `nvshmem.core.get_peer_tensor`, or a TRT-LLM-owned wrapper that exposes the same tensor and peer-pointer semantics.
- Treat dependency approval as a hard gate. Confirm with TRT-LLM owners whether `nvshmem4py-cu13` and its `nvidia-nvshmem-cu13` C-library dependency can be added to the runtime image. If this package is not acceptable as a TRT-LLM dependency, implementation must stop at the provider spike and not proceed to backend wiring.
- Define a process-level NVSHMEM singleton: init once, allocate/free symmetric tensors through tracked handles, and finalize only at engine/process teardown after all tensors are explicitly freed.
- Reconcile ownership with existing DeepEP NVSHMEM usage before adding a second initialization path. The design must state whether MegaMoE CuteDSL shares the same runtime initialization or uses an isolated provider that is proven not to interfere.
- Decide PP/layer-skip behavior before multi-rank enablement. If build-time symmetric allocation or NVSHMEM init cannot be made lockstep under those paths, `MEGAMOE_CUTEDSL` must be disabled for those configurations.
- Allocate same-size per-rank symmetric buffers for activation, activation SF, top-k weights, combine output, and shared workspace.
- Provide `base_addr=int(local_tensor.data_ptr())` and `offsets=(peer_ptr - base_addr for peer in ep_ranks)`.
- Guarantee identical allocation order and compatible offsets across EP ranks.
- Reuse buffers across calls when shape and dtype match.
- Avoid storing immutable weights in symmetric scratch; weights are local-only according to the kernel ABI.

Allocation timing:

- Resolve the EP process group in `__init__`, following `MegaMoEDeepGemm._resolve_ep_pg`.
- Allocate or fetch symmetric buffers from `create_weights()`, after `ConfigurableMoE` has synchronized `num_slots` and `expert_size_per_partition` onto the backend.
- Do not allocate symmetric memory in `run_moe`. Symmetric-memory rendezvous is host-side collective IPC and is unsafe under PP/layer-skip paths or CUDA graph capture.
- Local CUDA-only workspace may be allocated lazily in `run_moe`, because it does not participate in cross-rank rendezvous.

Cache storage:

- The cache must live at module scope as a process-global dictionary, matching `MegaMoEDeepGemm`:

```python
_MEGA_MOE_CUTEDSL_SYMM_CACHE: Dict[tuple, SymBufferHandle] = {}
```

- The cached object is mutable forward-time symmetric scratch, not weight state. Reuse across 60+ MoE layers is required to avoid multiplying the form-A combine buffer by layer count.
- The concurrency contract matches `MegaMoEDeepGemm`: reuse assumes MoE layers are executed serially within one forward pass. Concurrent forwards that share the same cache key would race on the same scratch buffers unless a future implementation adds per-forward cache partitioning or locking.

Cache key:

```python
(
    ep_group_signature,
    self.num_experts,
    self.num_slots,
    self.max_tokens_per_rank,
    self.routing_method.experts_per_token,
    self.hidden_size,
    self.intermediate_size_per_partition,
    self.expand_intermediate_size_per_partition,
    self.dtype,
)
```

`ep_group_signature` must be stable for the process lifetime and should not rely on `id(self._ep_pg)`. Prefer an explicit tuple such as `(world_size, ep_rank, sorted_global_ranks)` or a named process-group identifier when available. If the implementation cannot obtain stable global ranks, wrap the cache in a weakref-aware structure tied to the process group object and document the lifetime.

Provider spike exit criteria before backend implementation:

- TRT-LLM owners accept the dependency/runtime ownership model or an equivalent in-tree provider.
- Init-once/finalize-on-engine-destroy behavior is specified and tested alongside existing DeepEP NVSHMEM usage.
- A two-rank test allocates symmetric tensors through the chosen provider.
- `SymBufferHost` receives `base_addr`, `offsets`, `rank_idx`, and `num_max_ranks` and a tiny CuteDSL kernel can peer-map a pointer.
- The provider works from the same EP process group topology that `ConfigurableMoE` / `Mapping` will use.
- The allocation path is build-time only and does not run under CUDA graph capture.
- A negative test or capability check disables `MEGAMOE_CUTEDSL` for PP/layer-skip configurations if the provider cannot guarantee lockstep symmetric allocation.

## Quantization And Weight Layout

This section is the authoritative spec for `NVFP4MegaMoECuteDslMethod`. Every shape, stride, and swizzle rule below is grounded in the upstream CuteDSL kernel sources at `/home/xxi/sc2/cutedsl_megamoe`. File:line citations appear after each statement so reviewers can verify by diff.

### Symbol mapping: TRT-LLM ↔ upstream MegaMoE

The two codebases use overlapping but mismatched names. Use this mapping consistently, especially when computing scale shapes:

| Upstream symbol | TRT-LLM symbol | Meaning |
|---|---|---|
| `experts` (kernel `experts` template param) | `num_local_slots = num_slots // mapping.moe_ep_size` | Per-rank expert/slot count |
| `hidden` | `hidden_size` | Token feature dim |
| `intermediate` (also called `intermediate_gateup`) | `expand_intermediate_size_per_partition` | FC1 N-axis = 2 × down-proj width |
| `intermediate_downproj` (= `intermediate // 2`) | `intermediate_size_per_partition` | FC2 K-axis = down-proj width |
| `Nvfp4BlockSize` | `scaling_vector_size` (16) | NVFP4 SF block size |
| `Fc1GateUpInterleave` | `_FC1_GATE_UP_INTERLEAVE` (16) | Gate/up interleave atom size |
| `SfPaddingBlock` | `_SF_PADDING_BLOCK` (128) | to_blocked row-pad atom |
| `EpilogueTokenTile` | `_TOKEN_PADDING_BLOCK` (64) | Pool token tile size |

Hard alignment requirements (from `ProblemDesc.__post_init__` at `mega_runner.py:312-339`):

- `hidden_size % (2 * Nvfp4BlockSize) == 0` → `hidden_size % 32 == 0`.
- `expand_intermediate_size_per_partition % (2 * Fc1GateUpInterleave) == 0` → `% 32 == 0`.
- `num_experts % world_size == 0` → `num_slots % mapping.moe_ep_size == 0`.

### `class NVFP4MegaMoECuteDslMethod(FusedMoEMethodBase)`

Responsibilities (each owns a step in the transform pipeline below):

- Register the `nn.Parameter` set on the module in the shapes required by `Sm100MegaMoEKernel.__call__`.
- Load checkpoint `w1` / `w3` / `w2` using the same public weight names as existing NVFP4 MoE paths.
- Apply gate/up interleave at 16-element atom granularity along the `expand_intermediate` axis (NOT the parent NVFP4 method's `group_size=64` convention).
- Apply the 32×4×4 `to_blocked` swizzle to every per-slot raw weight scale tensor (FC1 + FC2).
- Keep `fc31_alpha`, `fc2_alpha`, `fc2_input_scale`, and `fc31_input_scale` as first-class tensors.
- Expose `MegaMoECuteDslWeightView` for `run_moe`.
- Declare EPLB status explicitly (target: SUPPORTED with transformed-scale staging; v1 may declare NOT_VERIFIED with specific missing artifacts).

Do not subclass `NVFP4CuteDslFusedMoEMethod`. Evidence: its `_interleave_w3_w1_weight` uses `interleave_linear_and_gate(group_size=64, dim=0)` (`quantization.py:2916-2924`), which is the byte layout of `cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`. MegaMoE FC12 reads gate/up at 16-element granularity along the intermediate axis (`epilogue.py:38` defines `Fc1GateUpInterleave = 16`; `runner_fc12.py:387-411` references the canonical layout `c_fp32.view(M, n_pairs, 2, 16)`). The two conventions are not equivalent.

### Parameter registration (`create_weights`)

The quantization method registers six per-expert `nn.Parameter`s plus per-tensor / per-expert scalar parameters. All sizes use `num_local_slots = module.expert_size_per_partition`.

| Parameter | Shape (logical NVFP4 elements) | Storage shape (bytes) | dtype | Stride-1 axis (post-transform) |
|---|---|---|---|---|
| `fc1_weight` | `(num_local_slots, hidden_size, expand_intermediate)` | `(num_local_slots, hidden_size // 2, expand_intermediate)` | `float4_e2m1fn_x2` | hidden_size (K-major) |
| `fc1_weight_sf` | n/a (1-D flat per slot) | `(num_local_slots, fc1_sf_flat_size)` | `float8_e4m3fn` (atom layout via `uint8` view) | flat |
| `fc2_weight` | `(num_local_slots, intermediate_size_per_partition, hidden_size)` | `(num_local_slots, intermediate_size_per_partition // 2, hidden_size)` | `float4_e2m1fn_x2` | intermediate_size_per_partition (K-major) |
| `fc2_weight_sf` | n/a (1-D flat per slot) | `(num_local_slots, fc2_sf_flat_size)` | `float8_e4m3fn` | flat |
| `fc31_alpha` | `(num_local_slots,)` | same | `float32` | scalar per slot |
| `fc2_alpha` | `(num_local_slots,)` | same | `float32` | scalar per slot |
| `fc31_input_scale` | `()` (per-layer) | same | `float32` | scalar |
| `fc2_input_scale` | `()` (per-layer) | same | `float32` | scalar |

`fc1_sf_flat_size` and `fc2_sf_flat_size` are deterministic functions of layer shapes (kernel-side evidence `kernel_fc12.py:880-890` and `runner_fc12.py:1298-1310`):

```python
fc1_sf_flat_size = (
    round_up(expand_intermediate, SfPaddingBlock=128)
    * round_up(ceil(hidden_size / scaling_vector_size=16), 4)
)
fc2_sf_flat_size = (
    round_up(hidden_size, SfPaddingBlock=128)
    * round_up(ceil(intermediate_size_per_partition / scaling_vector_size=16), 4)
)
```

K-major storage detail (verified at `runner_fc12.py:1270-1290`): the parent class stores
`(num_local_slots, intermediate, hidden // 2)` then `.permute(0, 2, 1)` so that the
hidden axis is stride-1 in byte terms. The TRT-LLM loader must produce a tensor
whose `.stride(-2) == 1` view exposes the kernel-required `(slots, hidden, intermediate)` logical shape. The same applies to FC2 with `(slots, hidden, intermediate_downproj // 2)` underlying storage permuted to expose intermediate_downproj as stride-1.

### Weight view passed to `run_moe`

```python
@dataclass(frozen=True)
class MegaMoECuteDslWeightView:
    # Per-slot tensors registered as nn.Parameter, sliced to the rank's local
    # experts. The kernel reads these as local-only (NOT through symmetric heap).
    fc1_weight: torch.Tensor              # (num_local_slots, H, expand_I) NVFP4 packed
    fc1_weight_sf: torch.Tensor           # (num_local_slots, fc1_sf_flat_size) FP8 atom-swizzled
    fc2_weight: torch.Tensor              # (num_local_slots, I, H) NVFP4 packed
    fc2_weight_sf: torch.Tensor           # (num_local_slots, fc2_sf_flat_size) FP8 atom-swizzled
    # Scales required by the extended kernel ABI (see "NVFP4 scale and alpha ABI").
    fc31_alpha: torch.Tensor              # (num_local_slots,) fp32
    fc2_alpha: torch.Tensor               # (num_local_slots,) fp32
    fc2_input_scale: torch.Tensor         # () fp32 (per-layer scalar)
```

`fc31_input_scale` is consumed by `quantize_input(...)` for FC1 input quantization and is not part of the FC12 kernel ABI; it stays on `self` and does not enter the view.

### Required transform pipeline (`load_weights` → `post_load_weights` → `process_weights_after_loading`)

Pipeline owner: `NVFP4MegaMoECuteDslMethod`. The backend calls `quant_method.create_weights/load_weights/post_load_weights` and provides EPLB shared-staging hooks.

1. **Load raw checkpoint tensors.** Use the existing NVFP4 checkpoint utilities (parent class `NVFP4CutlassFusedMoEMethod.load_expert_w3_w1_weight*` and `load_expert_w2_weight*`). The raw per-expert layout follows TRT-LLM convention: `w1` and `w3` are each `(intermediate_size_per_partition, hidden_size)`; `w2` is `(hidden_size, intermediate_size_per_partition)`. Keep the raw tensors untransformed until the post-load step.

2. **Build FC1 gate/up tensor with 16-atom interleave.** The external epilogue (`epilogue.py:38`) defines `Fc1GateUpInterleave = 16`, and `runner_fc12.py:387-411` documents the canonical reference layout:

   ```text
   intermediate_axis = 2 * Fc1GateUpInterleave * n_pairs
   layout per (slot, hidden_row):
   [gate[0:16], up[0:16],
    gate[16:32], up[16:32],
    ...,
    gate[(n_pairs-1)*16 : n_pairs*16], up[(n_pairs-1)*16 : n_pairs*16]]
   ```

   For TRT-LLM, `n_pairs = intermediate_size_per_partition // 16` and the resulting axis has length `2 * 16 * n_pairs = expand_intermediate_size_per_partition`. Gate vs up ordering must match what the kernel's epilogue consumes; pick the order experimentally and lock it down with the byte-equivalence test below.

   Algorithm (NVFP4-packed bytes, do not call `.contiguous()` on the final permute or strides will be wrong):

   ```python
   # raw_gate, raw_up: per-expert NVFP4 weights, logical shape
   #   (num_local_slots, intermediate_size_per_partition, hidden_size)
   # Physical NVFP4 byte storage (packed along hidden):
   #   (num_local_slots, intermediate_size_per_partition, hidden_size // 2)
   I = intermediate_size_per_partition
   n_pairs = I // 16
   H_bytes = hidden_size // 2  # NVFP4 packed-byte width

   # Reshape M-axis into 16-element chunks for both gate and up.
   gate_p = raw_gate.view(num_local_slots, n_pairs, 16, H_bytes)
   up_p   = raw_up.view(num_local_slots, n_pairs, 16, H_bytes)

   # Interleave at pair granularity: pair i = [gate_chunk_i, up_chunk_i].
   # torch.stack(dim=2) allocates a fresh contiguous buffer, so the next
   # .view(...) is safe.
   interleaved = torch.stack([gate_p, up_p], dim=2).contiguous()
   # Shape: (slots, n_pairs, 2, 16, H_bytes); merge (n_pairs, 2, 16) -> expand_intermediate.
   interleaved = interleaved.view(num_local_slots, expand_intermediate, H_bytes)

   # Permute to expose H_bytes as stride-1 along the GEMM K axis.
   # Do NOT call .contiguous() after this permute; the goal is the same
   # non-contiguous (stride[1]==1) view the upstream constructor produces
   # at runner_fc12.py L1273-1280.
   fc1_weight = interleaved.permute(0, 2, 1)
   # Logical fp4 shape: (slots, hidden_size, expand_intermediate);
   # storage shape  : (slots, H_bytes, expand_intermediate);
   # strides        : (expand_intermediate * H_bytes, 1, H_bytes).
   ```

   Whichever path produces the byte sequence must pass the byte-equivalence test against a tiny reference that mirrors the upstream `_make_nvfp4_tensor_from_rng` packed-dim convention (`mega_runner.py:823-826`, `packed_dim=2`).

3. **Build FC2 weight in SwapAB orientation.** The kernel consumes
   `fc2_weight.shape == (experts, intermediate_downproj, hidden)` with `intermediate_downproj` as stride-1 (`kernel_fc12.py:732`, `runner_fc12.py:1282-1290`). TRT-LLM's `w2_weight` per expert is `(hidden_size, intermediate_size_per_partition)` with hidden as stride-1; the loader must transpose to `(intermediate_size_per_partition, hidden_size)` and then pack so that the packed-K axis is stride-1.

4. **Build FC1 raw weight scale per slot.** Shape `(expand_intermediate, ceil(hidden_size / 16))` FP8. The non-K (M-axis of the GEMM) is `expand_intermediate` and must already be gate/up-interleaved with the same 16-atom pattern; that means the loader applies the same gate/up interleave to the scale tensors as to the weight tensors before swizzle. Evidence: kernel uses `fc1_weight_sf` on TMA SFA descriptor for `M=expand_intermediate, K=hidden` (`kernel_fc12.py:870-890`).

5. **Build FC2 raw weight scale per slot.** Shape `(hidden_size, ceil(intermediate_size_per_partition / 16))` FP8. The non-K is `hidden_size` (kernel's FC2 M-axis), K is `intermediate_size_per_partition` (FC2 K-axis). Evidence: `runner_fc12.py:1217-1224` (`non_k=hidden, k=intermediate // 2`).

   Common mistake to avoid: do not use `intermediate_size_per_partition * 2 / 16` for the scale K axis. The FC2 GEMM consumes the post-SwiGLU activation whose width is `intermediate_size_per_partition` (down-proj width), not the gate-up sum.

6. **Apply `to_blocked` swizzle to each per-slot 2-D scale tensor.** Provide a TRT-LLM-side `_torch/cute_dsl_kernels/mega_moe_nvfp4/blocked_scale.py` that mirrors `runner_fc12.py:271-298` byte-for-byte:

   ```python
   def to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
       rows, cols = scale_2d.shape
       if rows == 0 or cols == 0:
           return scale_2d.new_empty((0,))
       row_blocks = ceil_div(rows, _SF_PADDING_BLOCK)    # 128
       col_blocks = ceil_div(cols, 4)
       padded_rows = row_blocks * _SF_PADDING_BLOCK
       padded_cols = col_blocks * 4
       padded = scale_2d
       if (rows, cols) != (padded_rows, padded_cols):
           padded = torch.zeros((padded_rows, padded_cols), dtype=scale_2d.dtype, device=scale_2d.device)
           padded[:rows, :cols] = scale_2d
       blocks = padded.view(row_blocks, _SF_PADDING_BLOCK, col_blocks, 4).permute(0, 2, 1, 3)
       rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
       return rearranged.flatten()
   ```

   Output is 1-D length `padded_rows * padded_cols` FP8 bytes. Empty tensors are allowed and must short-circuit.

7. **Stack blocked per-slot scales.** Final `fc1_weight_sf.shape == (num_local_slots, fc1_sf_flat_size)`, `fc2_weight_sf.shape == (num_local_slots, fc2_sf_flat_size)`. Use a byte-reinterpretable stack helper (`_stack_byte_reinterpretable_tensors` in `runner_fc12.py:316-326`) because `torch.stack` on FP8 dtypes is not universally supported across torch versions.

8. **Keep alpha and norm_const as first-class tensors.** Do not fold `fc31_alpha`, `fc2_alpha`, or `fc2_input_scale` into FP8 scale factors. Folding would corrupt FP8 dynamic range and is incompatible with form B in-kernel topk reduction (where the per-expert alpha is consumed inside the device fc2 epilogue).

9. **Register EPLB shared-staging buffers.** For every per-expert `nn.Parameter` registered in step (1)-(7), add the matching CPU shared staging buffer following the `quantization.py` family contract (see `MOE_DEVELOPER_GUIDE.md` and the skill's "CPU shared-staging buffer family" section):

   | GPU `nn.Parameter` | CPU staging | Sized by |
   |---|---|---|
   | `fc1_weight` | `module.local_shared_fc1_weight_tensors` | `num_shared` |
   | `fc2_weight` | `module.local_shared_fc2_weight_tensors` | `num_shared` |
   | `fc1_weight_sf` (post-blocked) | `module.local_shared_fc1_weight_sf_tensors` | `num_shared` |
   | `fc2_weight_sf` (post-blocked) | `module.local_shared_fc2_weight_sf_tensors` | `num_shared` |
   | `fc31_alpha` | `shared_fc31_alpha` (local var in `process_weights_after_loading`) | `num_shared` |
   | `fc2_alpha` | `shared_fc2_alpha` (local var) | `num_shared` |

   Migration fix-up functions must take the destination tensor as an explicit parameter, never write into `module.<param>.data[expert_idx]` directly from the staging path. `num_shared = len(local_shared_load_expert_ids)` differs from `expert_size_per_partition` on multi-node setups; mixing the two index spaces is the NVBug 6130334 / PR #13856 failure mode.

   Per-layer `fc31_input_scale` and `fc2_input_scale` do not need per-expert EPLB migration because they are layer-scope scalars, but they still need shared-load on the same rank for hot-reload completeness.

10. **Reject non-1 alpha when kernel ABI extension is absent.** At the end of `post_load_weights()`, if the ported kernel package signals that `fc31_alpha` / `fc2_alpha` / `fc2_input_scale` are not threaded through, assert each tensor equals 1.0 within FP32 tolerance and raise a clear error otherwise. This is the v1 product gate.

### EPLB status

- The target state is `EplbSupportStatus.SUPPORTED`, matching `MegaMoEDeepGemm`, because the fused kernel consumes slot IDs after scheduler/EPLB routing.
- V1 may declare `EplbSupportStatus.NOT_VERIFIED` with explicit missing artifacts: shared CPU buffers for transformed `fc1_weight_sf` / `fc2_weight_sf` and the corresponding migration fix-up functions that re-run `to_blocked` on the post-migration raw scales.
- Do not mark EPLB unsupported merely because the backend is fused-comm; fused-comm and EPLB are compatible when the transformed per-slot tensors migrate correctly.
- **V1 hard gate**: `MegaMoECuteDsl.validate_configurable_moe` rejects any `ConfigurableMoE` whose `_using_dynamic_load_balancer()` returns `True`. This guards against the silent staleness bug where dynamic EPLB migrates the parent `w3_w1_weight` / `w2_weight` in-place but the derived `mega_fc*_weight*` buffers (which depend on a 16-atom gate/up interleave + `to_blocked` swizzle of the parent) keep their pre-migration contents. Static EPLB (`MoeLoadBalancer.is_static_routing()`) is allowed because slot IDs are remapped only at construction time. The gate is removed once the shared-staging buffers and the `to_blocked` re-run migration hook are wired (TODO in `NVFP4MegaMoECuteDslMethod`).

### Byte-equivalence test plan for quantization

Required tests in `test_moe_backend.py` (run on a single SM100 GPU with cu13 Cutlass DSL available):

- `to_blocked(raw_scale)` matches the upstream `runner_fc12.to_blocked` byte-for-byte for `hidden in {1024, 1568, 2048, 7168}` and `intermediate_size in {1088, 2112, 2368, 4096}`.
- `fc1_weight` gate/up interleave matches a reference built directly from `raw_w1`, `raw_w3` via `runner_fc12.swiglu_fold_interleave_16`-equivalent indexing (gate at `[2*(c//16)*16 + c%16]`, up at `[(2*(c//16)+1)*16 + c%16]`).
- `fc2_weight` stride matches `intermediate_size_per_partition` stride-1; raise with a clear message if the loader produced an `(hidden, intermediate)` storage without the required permute.
- `fc1_sf_flat_size` and `fc2_sf_flat_size` match the kernel's `fc1_weight_sf.shape[1]` and `fc2_weight_sf.shape[1]` after running the constructor in single-rank degenerate mode.
- Non-1 `fc31_alpha`, `fc2_alpha`, `fc2_input_scale` round-trip through `MegaMoECuteDslWeightView` unchanged and reach the runner.

## Factory And Config Integration

`create_moe.py` changes:

- Import `MegaMoECuteDsl`.
- Add backend string:

```python
elif moe_backend.upper() == "MEGAMOE_CUTEDSL":
    ...
```

- Require `quant_config.quant_mode.has_nvfp4()`.
- Call `MegaMoECuteDsl.can_implement(...)` with pretrained dtype, hidden size, and MoE intermediate size.
- Fall back to `CutlassFusedMoE` with a precise warning when unsupported.
- Add `MegaMoECuteDsl` to the `ConfigurableMoE` allowlist.
- Update `create_moe_backend` to instantiate `MegaMoECuteDsl` through the same keyword set as `MegaMoEDeepGemm`.
- Prefer an explicit `elif moe_cls in (MegaMoEDeepGemm, MegaMoECuteDsl)` branch over adding more logic to the final `else` fall-through. This keeps new MegaMoE backends visible and avoids growing an implicit catch-all branch.

Public config surface:

- v1 can use `model_config.moe_backend = "MEGAMOE_CUTEDSL"` plus internal default tactic.
- Follow-up should add structured tactic override fields instead of environment variables. These fields should be Pydantic-validated if exposed through user-facing config.

## Scheduler Integration

The new backend should not require a new scheduler kind.

Required scheduler change:

- Replace the current DeepGEMM-specific empty quantization fallback with an unconditional backend call:

```python
x_quant, x_sf = moe.backend.quantize_input(x_chunk_real)
```

Each fused-comm backend must make `quantize_input` tolerate `x.shape[0] == 0` and return its own empty tensor layout. This keeps `FusedCommMoEScheduler` layout-agnostic and avoids adding a new `make_empty_quantized_input` method.

Before changing the scheduler, add a regression test for `MegaMoEDeepGemm.quantize_input` with `x.shape[0] == 0`. The scheduler refactor must not rely on undocumented behavior of `torch.ops.trtllm.mxfp8_quantize` or the DeepGEMM fallback path.

Optional scheduler change (blocked by a precondition):

- Passing `all_rank_num_tokens` through `run_moe` as a kwarg is tempting (`MegaMoECuteDsl` can use it to validate lockstep assumptions and choose fixed staging shapes), but `MegaMoEDeepGemm.run_moe` currently has `assert not unused_kwargs` (see `mega_moe_deepgemm.py:555-557`). Adding the kwarg from the scheduler without first relaxing that assert would crash every DeepGEMM forward.
- The safe order is:
  1. Relax `MegaMoEDeepGemm.run_moe` to accept and ignore `all_rank_num_tokens` via a typed kwarg (and add a regression test that DG accepts it).
  2. Then update the scheduler to pass it for fused-comm backends.
  3. Then have `MegaMoECuteDsl.run_moe` consume it.
- If 1. is not in scope for the v1 PR, the v1 backend should derive lockstep token counts inside `run_moe` from `self._symm_buffer` shape and `x.shape[0]`, not from a new scheduler kwarg.

Do not add `MegaMoECuteDsl`-specific forward branches in `ConfigurableMoE`.

### FusedCommMoEScheduler invariant checklist

`MegaMoECuteDsl` must satisfy every invariant currently documented on `FusedCommMoEScheduler`:

1. Reject pre-quantized `Fp4QuantizedTensor` activation; backend `quantize_input` owns BF16 to NVFP4.
2. Ignore `use_dp_padding`; no host-side shape alignment is available for fused comm.
3. Use `mapping.moe_ep_rank` for local token counts because symmetric exchange is EP-scoped.
4. Strip ADP padding before chunking.
5. If `all_rank_num_tokens is None`, allow scheduler fallback to `x.shape[0]`; backend must not index `all_rank_num_tokens` unconditionally.
6. Compile with static `max_tokens_per_rank >= moe.moe_max_num_tokens`; runtime `num_chunks` is derived from the max real token count across EP ranks.
7. Launch every chunk on every EP rank, including zero-token chunks, using the same compiled static shape and runtime zero-token metadata.
8. Do not call external `Communication.dispatch` or `Communication.combine`.
9. Do not use multi-stream chunk overlap for fused-comm kernels.

The implementation should add focused tests for invariants 5, 6, and 7 because they are easy to break when adding a new fused-comm backend.

## Test Plan

### Shared helpers

Update `moe_test_utils.py`:

- `MoeBackendType` exposes both `MEGAMOE_DEEPGEMM = "MEGAMOE_DEEPGEMM"` and `MEGAMOE_CUTEDSL = "MEGAMOE_CUTEDSL"` as canonical members. The legacy asymmetric pair `MEGAMOE = "MEGAMOE_DEEPGEMM"` was removed in the v1 PR alongside a full use-site migration across `test_moe_backend.py`, `test_moe_module.py`, `quantize_utils.py`, `tests/microbenchmarks/bench_moe/`, and the `get_backend_class` / `should_skip_*` helper branches. Do not reintroduce `MoeBackendType.MEGAMOE` as a runtime alias: enum aliases with the same value silently break `value -> member` lookup tables and make grep noisy.
- Add `get_backend_class` mapping.
- Skip helpers are split into `should_skip_megamoe_deepgemm(...)` and `should_skip_megamoe_cutedsl(...)`. The grouped-GEMM CuteDSL backend and MegaMoE CuteDSL backend have different shape and routing constraints, so the helpers must stay independent.
- Make any generic `should_skip_cutedsl(...)` path early-return or dispatch away from `MEGAMOE_CUTEDSL`.
- Add skip logic for:
  - no CUDA
  - non-SM100-family
  - missing Cutlass DSL capability probe
  - missing CUDA 13 Cutlass DSL runtime symbols required by the kernel
  - non-NVFP4 quantization
  - unsupported hidden/intermediate shape
  - unsupported TP / mixed parallelism
  - `experts_per_token > 13` unless larger top-k coverage is added

Update `quantize_utils.py`:

- Add NVFP4 test quantization parameters for `MegaMoECuteDsl`.
- Generate or transform weights into the exact `MegaMoECuteDslWeightView` layout.
- Update backend-name branches that currently special-case `"MEGAMOE_DEEPGEMM"` so the new enum naming remains explicit and the CuteDSL path can select its own reference class.

### Backend tests

Add focused coverage in `test_moe_backend.py`:

- `can_implement` accepts NVFP4 BF16 SM100-family shapes.
- `can_implement` rejects non-NVFP4, non-BF16 activation, unsupported SM, GPT-OSS SwiGLU, and bad hidden/intermediate sizes.
- `quantize_input` returns NVFP4 activation plus padded FP8 SF layout.
- `quantize_input` accepts zero-token input and returns dtype/shape compatible with `run_moe`.
- `run_moe` single-rank output matches a reference implementation for small deterministic inputs.
- Tactic override changes compile/autotune key and reaches the runner.
- Tactic serialization passes `json.dumps`/`json.loads` and `eval(repr(tactic))` compatibility checks required by `TunableRunner`.
- `group_hint=None` is resolved before compile-cache lookup, so the cache key contains the real integer used by `Sm100MegaMoEKernel`.
- The flattened kernel package imports cleanly after all `src.*` and `moe_nvfp4_swapab.*` imports are rewritten.
- `blocked_scale.to_blocked(raw_scale)` and `from_blocked(...)` are byte-identical to the upstream `runner_fc12.py` helpers for representative FC1 and FC2 scale shapes.
- Non-one `fc31_alpha`, `fc2_alpha`, and `fc2_input_scale` affect output as expected; tests must fail against a kernel that still hard-codes `alpha=1.0` / `norm_const=1.0`.
- If the kernel ABI extension is not present, `post_load_weights()` (or `process_weights_after_loading()`) detects non-1 `fc31_alpha`, `fc2_alpha`, or `fc2_input_scale` after checkpoint load and raises a clear error. `can_implement` is not the gate, because it cannot see checkpoint values.
- `MegaMoEDeepGemm.quantize_input` zero-token behavior is verified before the shared scheduler empty-path refactor lands.
- Activation SF conversion is byte-equivalent to the external runner for representative hidden sizes; if no view-only transform exists, the dedicated conversion kernel is tested directly.

### Module tests

Add integration coverage in `test_moe_module.py`:

- `create_moe` with `model_config.moe_backend="MEGAMOE_CUTEDSL"` returns `ConfigurableMoE` whose backend is `MegaMoECuteDsl`.
- `ConfigurableMoE.forward` calls the fused scheduler path and does not create external communication.
- Single-rank forward matches a reference within NVFP4 tolerance.
- Multi-rank EP launches all ranks for each chunk and returns correctly shaped local output.
- Invariant tests cover `all_rank_num_tokens=None`, per-rank zero-token chunks, and `moe_max_num_tokens`-driven chunking with a single static compile shape.
- Dynamic EPLB tests either pass with transformed scale migration enabled or skip with the exact missing staging/fix-up buffers listed.

### CuTe DSL cu13 gate

Testing should check both:

- `IS_CUTLASS_DSL_AVAILABLE`
- direct import/capability probes for the CUDA 13 Cutlass DSL symbols used by `sym_buffer.py`, `dispatch_kernel.py`, and `megamoe_kernel.py`

Skip with an actionable reason if the runtime environment cannot import the required CUDA 13 Cutlass DSL symbols from PR 14354.

## Implementation Phases

Status legend: ✅ done · ⚠️ partial · ⏳ pending hard gate.

0. ✅ Symmetric-memory provider lands as `MegaMoeSymmMemProvider` in `custom_ops/cute_dsl_megamoe_custom_op.py`, backed by PyTorch's `torch.distributed._symmetric_memory` (cuMem-based NVSHMEM-equivalent already used elsewhere in TRT-LLM by `SymmetricMemoryAllReduce`). The provider allocates ONE rendezvous'd buffer per (group, layout) and carves out five regions (activation / activation_sf / topk_weights / combine_output / shared_workspace) that share the same `peer_offsets` because they live in the same allocation. Process-scoped cache (`_MEGAMOE_SYMM_PROVIDER_CACHE`) shares the buffer across MoE layers. Init/finalize lifetime: enabled lazily on first multi-rank `run_moe`; idempotent enable on the same group; init runs collectively. NOTE: PP/layer-skip behavior depends on the underlying `enable_symm_mem_for_group` semantics — this matches the existing `SymmetricMemoryAllReduce` constraints.
1. ✅ Port kernel sources into `cute_dsl_kernels/mega_moe_nvfp4` and fix imports according to the source-to-target table.
2. ⏳ Extend the ported kernel ABI for `fc31_alpha`, `fc2_alpha`, and `fc2_input_scale` / `norm_const`. **Status**: pending; `NVFP4MegaMoECuteDslMethod._check_v1_alpha_gate` rejects checkpoints with non-1 values so production paths fall back via the factory instead of silently producing wrong results.
3. ✅ Add `blocked_scale.py` (`to_blocked` / `from_blocked` / `stack_byte_reinterpretable_tensors`) with byte-equivalence roundtrip tests in `test_moe_backend.py`.
4. ⚠️ Production memory-budget report for form A vs form B. **Status**: form A is the v1 default; explicit memory-budget measurement is deferred.
5. ✅ Tactic representation + Runner moved to `tensorrt_llm/_torch/custom_ops/cute_dsl_megamoe_custom_op.py` per the standard CuteDSL pattern (mirrors `cute_dsl_custom_ops.py` for `CuteDslFusedMoE`). The runner owns `kernel_cache`, `tuning_config`, `get_valid_tactics`, `forward`, and `unique_id`. Tactic is a JSON-friendly 6-tuple validated against the kernel-side constraints.
6. ✅ Backend (`mega_moe_cute_dsl.py`) implements capability gating, config validation, lifecycle hooks (`create_weights`, `load_weights`, `post_load_weights`, idempotent `process_weights_after_loading`, `pre_reload_weights`), EP process-group resolution, and a real `run_moe` that calls `torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell`. Both single-rank (peer_offsets=[0]) and multi-rank (peer_offsets from `MegaMoeSymmMemProvider.peer_offsets`) paths are wired. `create_weights()` follows the design's ordered safety pattern: (1) `_alloc_symm_provider` build-time collective rendezvous when `ep_size > 1`, (2) `_get_quant_method`, (3) `quant_method.create_weights`, (4) flip `_weights_created`. `run_moe` rejects per-rank token counts above `max_num_tokens` so peers do not read invalid rows. `MegaMoECuteDslWeightView` bundles `fc1_weight`, `fc1_weight_sf`, `fc2_weight`, `fc2_weight_sf`, and the alpha / norm_const tensors (`fc31_alpha`, `fc2_alpha`, `fc2_input_scale`) per the design contract; the kernel ABI extension will consume the alpha fields without touching the backend boundary.
7. ✅ `NVFP4MegaMoECuteDslMethod` implements the full blocked scale / weight transform pipeline (16-atom gate/up interleave, FC2 byte copy, per-slot `to_blocked` swizzle, flat per-slot stacked SF) and v1 alpha gate. `MegaMoECuteDslWeightView` bundles the four MegaMoE-format tensors for `run_moe`.
8. ✅ `MegaMoEDeepGemm.quantize_input` zero-token regression test landed; `FusedCommMoEScheduler` refactored to call `backend.quantize_input` for zero-token chunks unconditionally.
9. ✅ Factory integration (`create_moe.py` accepts `MEGAMOE_CUTEDSL`) and shared test helper entries (`MoeBackendType.MEGAMOE_CUTEDSL`, `should_skip_megamoe_cutedsl`, `get_backend_class`, autotune-capable gate).
10. ✅ Backend-level tests: `test_moe_backend.py` covers `can_implement`, tactic validation, tactic JSON/`repr` round-trip, blocked-scale byte roundtrip, `quantize_input` zero-token, multi-rank `run_moe` gate, v1 alpha gate, kernel package import.
11. ✅ Module-level `ConfigurableMoE` test: `test_megamoe_cutedsl_factory_routing_and_scheduler` in `test_moe_module.py`.
12. ✅ `MOE_DEVELOPER_GUIDE.md` updated (backend file map, quant capability matrix, FUSED_COMM scheduler invariant about uniform `quantize_input`, anti-patterns about allocating symmetric memory from `run_moe`, dataclass tactic anti-pattern).
13. ⏳ Targeted unit tests on a CUDA 13 Cutlass DSL environment. **Status**: blocked on OCI worktree needing a fresh C++ build (main repo `.so` is too old; missing `LinearCacheType` from `tensorrt_llm.bindings.internal.batch_manager`). Multi-rank GPU tests now possible end-to-end via `MegaMoeSymmMemProvider`; pending the same OCI rebuild.

## Open Questions

- Should v1 expose tactic overrides through public `MoeConfig`, or keep them internal until autotuner coverage is stable?
- Should form-B in-kernel top-k reduction (`use_bf16_redg`) be enabled in v1 for large target models where form A symmetric memory exceeds budget?
- Is dynamic EPLB required for the first PR, or can it be explicitly marked `NOT_VERIFIED` until transformed NVFP4 scale migration is covered?
- Should the kernel ABI extension evolve only in the TRT-LLM ported copy, or should it be coordinated upstream in `/home/xxi/sc2/cutedsl_megamoe` before re-porting? Who owns coordination with the upstream owner?
