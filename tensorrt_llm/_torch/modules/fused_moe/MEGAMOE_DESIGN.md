# MegaMoE Design

A "mega-fused" MoE path that inlines routing, quantize, communication, grouped-GEMM × 2, SwiGLU, combine, and reduce in a **single `forward_impl`**, inspired by DeepGEMM PR #304.

## Hard requirements (see `MOE_AGENT_EVALUATION.md` for full rubric)

- cuteDSL kernels only (NVFP4, Blackwell sm_100/103). Inline PTX allowed if unavoidable.
- Inherit directly from `MoE` (in `interface.py`). **Never** inherit from or compose with `ConfigurableMoE`.
- A single `forward_impl`; no dispatch to separate `quantize_input → comm.dispatch → run_moe → comm.combine` method calls.
- Tests added in `tests/unittest/_torch/modules/moe/test_mega_moe.py`, single + multi GPU, pass on OCI B200.

## Design choice: Mega-flow (initial) → Mega-kernel (aspirational)

DeepGEMM PR #304 is a **single CUDA C++ kernel** (1364 lines) fusing EP dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine using tcgen05 / UTCCP / 2-CTA UMMA / NVLink symm memory. Direct 1:1 port to cuteDSL is multi-week effort.

The pragmatic first cut reuses existing cuteDSL NVFP4 kernels:

- `torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell` — fuses gather + FC1 + SwiGLU.
- `torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell` — fuses FC2 + combine (token unpermute + scale).

These are cuteDSL kernels (see `custom_ops/cute_dsl_custom_ops.py`). MegaMoE calls both inside one `forward_impl` with no ConfigurableMoE orchestration. That satisfies C1/C2/C3.

**Progressive fusion roadmap:**

1. Phase A (this PR): 2 cuteDSL kernels (gather-GEMM-SwiGLU, GEMM-finalize), inlined in `forward_impl`. Single-GPU first.
2. Phase B: inline communication (all-gather / alltoall) in `forward_impl` without going through `Communication.dispatch/combine` method boundaries; overlap with GEMMs via CUDA streams.
3. Phase C (aspirational): merge the two kernels into a single persistent 2-CTA cuteDSL kernel matching DeepGEMM's fusion scope.

## Architecture

```
MegaMoE(MoE)
├── can_implement()            # NVFP4 + SM100/103 + bf16 + no gptoss-style
├── __init__()                 # inherits MoE fields; no ConfigurableMoE wrap
├── create_weights()           # delegates to NVFP4MegaMoEMethod (new, or reuse NVFP4CuteDslFusedMoEMethod)
├── load_weights()             # delegates to quant_method
├── quantize_input()           # REQUIRED by ABC but NOT called from forward_impl — raises to enforce non-split use
├── run_moe()                  # REQUIRED by ABC but NOT called from forward_impl — raises to enforce non-split use
└── forward_impl()             # THE FUSED FLOW — one method, all steps inline:
        1. routing (self.routing_method.apply)
        2. fp4 quantize (torch.ops.trtllm.fp4_quantize) — inline
        3. [multi-GPU] inline all-gather / alltoall (future: comm-GEMM overlap)
        4. moe_sort for permutation indices
        5. cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell (FC1 + SwiGLU, cuteDSL)
        6. moe_output_memset_inplace
        7. cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell (FC2 + combine, cuteDSL)
        8. [multi-GPU] reducescatter_or_allreduce
```

## Weight layout (NVFP4, same as `NVFP4CuteDslFusedMoEMethod`)

- `w3_w1_weight`: `[E_local, 2I, H/16]` float4_e2m1fn_x2
- `w2_weight`: `[E_local, H, I/16]` float4_e2m1fn_x2
- `*_weight_scale`: fp8_e4m3 block scales (UE8M0-style)
- `fc31_input_scale`, `fc2_input_scale`, `fc31_alpha`, `fc2_alpha`: fp32
- `scaling_vector_size = 16`

## Communication (Phase A: single-GPU; multi-GPU via Phase B)

Phase A: no EP/TP communication; run single GPU end-to-end.

Phase B: direct calls to `torch.ops.trtllm.allgather` / alltoall inside `forward_impl`, not via `Communication.dispatch()`. TP reduce via `self.reducescatter_or_allreduce` (already on base class).

## Current M5 Output-Path Status Contract

The optimized output path is identified by `full_fusion_final_kernel_path`, not by the standalone M5 strategy field.
MegaMoE requests the full-fusion output path by default for eligible runtime configs; explicit
`megamoe_enable_full_fusion_output_path=False` or `megamoe_enable_full_fusion_runtime=False` is
the rollback switch. Today the ready fast paths are:

- Multi-rank staged direct top-k: `in_kernel_stage_direct_topk+in_kernel_direct_buffer`.
- Single-rank/no-comm staged direct top-k: `in_kernel_stage_direct_topk+in_kernel_direct_buffer`.

The single-rank/no-comm path is attempted immediately after FP4 quantization and before `moe_sort`.
When it succeeds, the monolithic direct-top-k CuTe op owns direct dispatch materialization, FC1,
SwiGLU, FC2, direct combine-buffer writes, and local reduce in one kernel launch. A failed
pre-dispatch attempt keeps the existing post-dispatch compatibility fallback available.

Standalone M5 materialization is fallback/debug support. It is intentionally reachable only after an explicit
debug/materialization gate. If staged direct top-k is unavailable and no debug gate is active, MegaMoE falls
back to the compatibility output path instead of pre-materializing standalone M5 state:

- `direct_topk`, `direct_moe_sort`, and `moe_sort` use concrete materializer ops after the pre-materialization
  fallback is accepted.
- `helper_only` requires the route-pull, route-metadata, and pool-metadata materialization gates.
- `python_reconstruct` and `torch_moe_sort_reconstruct` require `reconstruction_materialize`.

Diagnostics:

- `m5_standalone_materialization_scope` records the caller that intentionally requested standalone
  fallback/debug M5 materialization.
- `m5_debug_materialization_gates` records the gates that made debug materialization legal.
- `m5_dispatch_materialize_strategy` is set only after branch prerequisites pass; rejected pre-gate candidates
  leave it `None` and report through fallback diagnostics.
- Clean optimized paths clear the standalone scope and strategy.
- `final_kernel_ready` is true only for in-kernel M5 plus direct M6 optimized paths. Standalone
  `direct_topk` fallback/debug paths can appear in `final_kernel_path`, but they are not ready.

## Test plan (see `test_mega_moe.py`)

- Single-GPU: random weights, random topk routing, compare MegaMoE output to `CuteDslFusedMoE` reference within NVFP4 tolerance.
- Multi-GPU (TP=2, EP=2): MPI pool harness from `moe_test_utils.py::_test_moe_worker`.
- Target: OCI B200 (sm_100). Computelab is insufficient per evaluator C6.
