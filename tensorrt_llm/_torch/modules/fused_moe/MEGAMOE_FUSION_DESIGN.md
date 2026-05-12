# MegaMoE Fusion Kernel Design

**Scope:** engineering design for the single fused **dispatch + FC1 + FC2 + combine** cuteDSL
kernel that replaces the current multi-kernel Python pipeline in `MegaMoE.forward_impl`.
Port target is DeepGEMM `sm100_fp8_fp4_mega_moe.cuh` (1364 lines CUDA C++ + inline PTX);
everything here is a Blackwell sm_100/103 NVFP4 design.

**Authoritative reference:** `/home/scratch.xxi_sw_2/trtllm/deepseek_mega_moe_discussion.md`
(§一-§八) is the semantics ground truth for dispatch, overlap graph, scheduler, combine.

**Status:** M1 done (commit `48891fb02`). M2.0-M2.4 done (commit `feba558af`, OCI 4/4 green).
M2.5-M7 planned below.

**Scope decision 2026-04-22 (revised):** dispatch and combine are **now in scope** of the
fused kernel, matching DeepGEMM PR#304 end-to-end. Earlier drafts deliberately excluded
dispatch/combine and left them at the Python-layer `CommunicationFactory`. That scope is
rejected — see §10 (updated) and §11-§13 for the new milestones.

---

## 1. Milestone table (extended for dispatch + combine)

| Milestone | Deliverable | Perf impact |
|-----------|-------------|-------------|
| **M1** ✅ | Skeleton: `blockscaled_contiguous_mega_moe_fusion.py` = verbatim copy of FC1, class renamed `BlockScaledMegaMoeFusionKernel`. | zero |
| **M2.0-M2.4** ✅ | Interface extended for Linear2 (flag, Optional tensors, sInfo 6-slot, 10 consumer warps const_expr-gated phase reads). `enable_linear2=False` → DCE → bit-equal with FC1. | zero |
| **M2.5** ✅ | Port FC2 epilogue (topk-weighted combine + BF16 scatter-add) into the mega kernel, wrapped in `const_expr(self.enable_linear2)` so anchor stays byte-equal with FC1 baseline. | zero (code dormant) |
| **M2.6 + M2.7** 🟡 | Path-1 merge: flip `enable_linear2=True` + add Linear2 TMA-B loader + UMMA + HBM pool + FC1→FC2 hand-off. S1-S2d ✅, S3a-d ✅, S3e-fix1-6 ✅ @ `73587c70d7` (**SMEM overflow cleared** — split `num_smem_capacity` 50/50 between L1 and L2 when `enable_linear2=True`). Fused kernel now launches and runs to completion. S3e-fix7 pending: `cudaErrorMisalignedAddress` surfaces on async CUDA sync after fused op returns (kernel ran for 11+ s on a 1-tile case before the error manifested — likely a pool/TMA descriptor or phase-branch barrier bug that the bigger shapes mask). linear1 anchor 4/4 PASSED byte-equal across 11 commits. | small — 1 kernel launch saved (when green) |
| **M3** ⬜ | **FC1↔FC2 tile-level overlap.** Wave-interleaved `MegaMoEScheduler` (port of `mega_moe.cuh::for_each_block`), `l2_arrival_mask[pool_block]` 64-bit bitmask sync (`red.or.release.gpu.b64` via `llvm.inline_asm`), HBM pool with same-SM wave scheduling → L2-cache hand-off. | **Perf milestone #1** — FC1→FC2 DRAM round-trip collapses to L2. |
| **M4** ⬜ **PoC** | Infrastructure probe for dispatch fusion: (a) cuteDSL can issue remote TMA loads via `sym_buffer.map`-equivalent address translation, (b) NVLink barrier primitive reachable (`red.release.sys.global.add.s32`), (c) symm memory allocator integrates with TRT-LLM's `NVLinkOneSided` / `DeepEPLowLatency` backend. Output: 1 hello-world kernel that pulls from a peer rank's symm buffer and writes a local HBM buffer. | gate for M5 |
| **M5** ⬜ | **Dispatch warp group (kNumDispatchWarps, 48 regs each).** Two-phase dispatch body: (1) write `src_token_topk_idx` metadata to peer ranks, (2) per-token TMA `cp.async.bulk` pull into local SMEM + TMA store into HBM `l1_token_buffer`, increment `l1_arrival_count[pool_block]`. FC1 A-loader changes from "token gather via `token_id_mapping_tensor`" to "contiguous read from pool, spin on `l1_arrival_count`". Python layer's `CommunicationFactory.dispatch()` removed. | **Perf milestone #2** — tile-level 3-way overlap (dispatch↔FC1↔FC2). |
| **M6** ⬜ | **Combine warp body** (piggybacks on epilogue 208-reg budget). FC2 epilogue scatters BF16 partials across peer ranks' `combine_token_buffer` via `sym_buffer.map`. NVLink barrier `kBeforeCombineReduceBarrierTag`, then local combine warp reduces topk partials and TMA-stores to `y`. Python layer's `reducescatter_or_allreduce` removed. | small-medium — 1-2 kernel launches saved, cross-rank roundtrip pulls into kernel. |
| **M7** ⬜ | Autotune `kNumExpertsPerWave` / `kNumDispatchWarps` / pool sizing / warp register split; pathological routing imbalance stress test; NSys validation of 3-way overlap. | last 5-10% |

### 1.1 M2.6 Path-1 status (2026-04-23) — SMEM overflow cleared, misaligned address surfaces

S3e-fix5 @ `bccf7d943f` cleared the MLIR region isolation blocker by
threading 5 L2 `TiledMma` / `ComposedLayout` / `Layout` attributes as
explicit kernel args (MLIR verification now passes). S3e-fix6 @
`73587c70d7` cleared the follow-on `CUDA_ERROR_INVALID_VALUE` at
launch: `_setup_attributes` was calling `_compute_stages` with the full
`num_smem_capacity` (228 KB) once for L1 and again for L2, so the CTA
SharedStorage allocation exceeded the Blackwell SM100a per-SM cap. The
fix splits the budget -- when `enable_linear2=True`, each pipeline
receives `num_smem_capacity // 2`. On the `enable_linear2=False` anchor
path nothing shares SMEM with L1, so the legacy full budget is
preserved and the linear1 byte-equal invariant holds across 11
consecutive commits (fix1..fix6).

OCI 2422097 verified that the fused kernel now launches, compiles, and
runs to completion on Blackwell. The residual failure is a
`cudaErrorMisalignedAddress` detected on `torch.cuda.synchronize()`
after the first fused test case `[2-512-512-2]` (the smallest shape,
orig_m=4, 1 tile). Because CUDA errors are sticky, every subsequent
test inherits the poisoned context and errors out during fixture
setup. The 11+ s elapsed time on a 1-tile case suggests the kernel
spins on a barrier before the bad access -- likely a pool-side TMA
descriptor stride or a phase-branch barrier wiring bug in the
single-tile corner case. Next step (S3e-fix7) is to re-run with
`CUDA_LAUNCH_BLOCKING=1` to localize the offending launch, then audit
the pool-side TMA descriptor setup (`pool_tensor` / `pool_sfc_tensor`
in `wrapper_fused`, tile partitions threaded into the kernel) and the
L1→L2 phase-transition barrier count in the single-tile case.

**Scope compression decision:** each milestone keeps an anchor test green
(M1 == FC1 byte-equal; M2.0-M2.7 == MegaMoE forward byte-equal with 2-kernel
reference via flag toggle; M3-M7 == MegaMoE forward output ≤ NVFP4 atol against
reference; M5-M7 on multi-rank must also match single-rank reference under
`MPIPoolExecutor`). Trying to land dispatch/combine in a single milestone
multiplies the debug surface beyond reason.

---

## 2. DeepGEMM reference — component map

Single-kernel fusion of 6 phases (DeepGEMM `sm100_fp8_fp4_mega_moe.cuh`):

| Phase | DeepGEMM warp role (regs) | Our milestone | Key primitive |
|-------|---------------------------|---------------|---------------|
| 1. EP dispatch (NVLink pull FP8 tokens + SF) | Dispatch warps (48 regs) | M5 | `ptx::tma_load_1d(..., sym_buffer.map(src, rank), mbarrier, kHidden)` + `red.add.release.gpu` to `l1_arrival_count` |
| 2. L1 TMA-A load (tokens from pool) | TMA-A loader (40 regs) | M2.7 (replaces LDGSTS) | `ld.acq.gpu` spin on `l1_arrival_count`, then TMA load |
| 3. L1 TMA-B load (weights + SFB) | TMA-B loader (40 regs) | M1 ✅ | TMA multicast `b_full_mcast_mask` |
| 4. 2-CTA UMMA (MXF8F6F4) | MMA warp (40 regs) | M1 ✅ | `tcgen05.mma.cta_group::2.kind::mxf8f6f4` + UTCCP |
| 5. L1 epilogue (SwiGLU + route-weight + FP4 requant + TMA store to pool) | Epilogue warps (208 regs) | M1 ✅ (SwiGLU+requant in epi); route-weight mul to L1 epi in M2.5 | `red.or.release.gpu.b64` to `l2_arrival_mask` at tile done |
| 6. L2 GEMM + BF16 combine (NVLink push + barrier + local reduce) | Combine warps (share 208-reg epi group) | M2.5-M2.7 (local FC2 epi); M6 (NVLink combine) | `sym_buffer.map(dst, rank) = partial` + NVLink barrier + local TMA reduce |

Everything outside this single kernel in our design is Python layer:
`routing_method.apply`, FP4 `fp4_quantize`, `moe_sort` (for in-rank permutation
before dispatch knows expert ordering — still needed; see §11.5). **No
`CommunicationFactory` call remains in `forward_impl` once M6 ships.**

---

## 3. Structural comparison: FC1 vs FC2

Both kernels already are 2-CTA cluster, warp-specialized, persistent
blockscaled NVFP4 GEMMs using `tcgen05.MmaMXF4NVF4Op(CtaGroup.TWO)` +
`Cp4x32x128bOp(CtaGroup.TWO)` + `TmemAllocator(is_two_cta=True)`. The template
is **already correct** for both; the divergence is in 6 sub-components listed
below. Line numbers are against the M1 baseline, which matches FC1's file.

Anchors (FC1 file = `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py`,
FC2 file = `blockscaled_contiguous_grouped_gemm_finalize_fusion.py`, worktree
`/home/scratch.xxi_sw_2/trtllm/.claude/worktrees/megamoe`):

| Component | FC1 (gather SwiGLU) | FC2 (finalize) | Delta |
|-----------|---------------------|----------------|-------|
| Class name | `BlockScaledContiguousGatherGroupedGemmKernel` (L296) | `Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel` (L299) | naming only |
| Warp layout | 12 warps: epilog(0-3) · **ldgsts_a(4-7)** · mma(8) · tma_b(9) · sched(10) · sync_transform(11) (L445-455) | 7 warps: epilog(0-3) · mma(4) · tma(5) · sched(6) (L390-393) | **LARGE** — FC1 has 4 extra LDGSTS warps + 1 sync_transform warp |
| `__call__` signature | `(a, b, c, sfa, sfb, sfc_tensor, norm_const_tensor, tile_idx_to_expert_idx, tile_idx_to_mn_limit, token_id_mapping_tensor, num_non_exiting_tiles, alpha, max_active_clusters, stream, epilogue_op)` (L723) | `(a, b, out, sfa, sfb, tile_idx_to_expert_idx, num_non_exiting_tiles, tile_idx_to_mn_limit, alpha, max_active_clusters, stream, permuted_idx_to_expanded_idx, token_final_scales, epilogue_op)` (L629) | union required |
| A-side loading | **LDGSTS + cpasync** with token-based gather (4 warps, L1670-1789); uses `token_id_mapping_tensor` to remap rows; no TMA for A | **TMA** (1 warp, L1556-1635) with `a_full_mcast_mask` | **LARGE — structural divergence** |
| B-side loading | TMA multicast, per-expert L-slice, SwiGLU-interleaved weights (`num_b_tensors` 1-4) (L1998-2080) | TMA multicast, per-expert L-slice (L1636+) | small — same pattern, different descriptor |
| SF A/B loading | UTCCP via mainloop; both SFA and SFB via TMA for SFB, via cpasync-to-SMEM for SFA | Same UTCCP path, but SFA via TMA too | small — SFA descriptor swap |
| MMA loop | `tcgen05.MmaMXF4NVF4Op(CtaGroup.TWO)`, k_tile loop, UTCCP SF, TMEM accumulator | Same primitive, same k_tile loop | micro-small — identical template, different K / shapes |
| Epilogue | **SwiGLU + FP4 requant** — up×silu(gate), optional FP4 quantize with SFC store | **topk-weighted combine + BF16 store** — per-row scale, reverse permutation via `permuted_idx_to_expanded_idx`, atomic-add / scatter to `out` | **LARGE** |
| Scheduler (tile order) | `utils.StaticPersistentTileScheduler` persistent | Same scheduler class | medium — need phase-aware augmentation in mega |

Key observation: **FC1 and FC2 are the same warp-specialized 2-CTA persistent
template** with two divergences: A-load (LDGSTS vs TMA) and epilogue (SwiGLU+FP4
vs combine+BF16). The mega kernel absorbs both.

---

## 4. Merge strategy per component (M2.5 — M2.7)

The mega kernel keeps FC1's 12-warp layout for Linear1+Linear2. Dispatch and
combine add new warp classes on top (§11, §12).

1. **A-side (warp 4-7, `ldgsts_a_warp_id`):** repurposed per phase.
   - Linear1 (M2.6): today's token-based gather with `token_id_mapping_tensor`.
     Once M5 lands, Linear1 A is read from the HBM pool (`l1_token_buffer`,
     contiguous) without `token_id_mapping` indirection.
   - Linear2: contiguous read from the same pool with `pool_ptr + tile_m *
     m_stride`. **M2.7 decision:** keep LDGSTS path (pool access transparent
     to LDGSTS). M3 may revisit and route through the TMA warp.
2. **B-side (warp 9, `tma_b_warp_id`):** two TMA descriptors (L1, L2 weights)
   created in host prolog; in-kernel `TensorMapManager.update_tensormap` swaps
   between them at phase boundary. Both are `[E, N, K/16]` FP4 grouped. Same
   multicast mask (`b_full_mcast_mask`) in both phases.
3. **SF A/B:** FC1 SFA via cpasync + UTCCP; Linear2 SFA is the Linear1 output
   SFC — generated in Linear1 epilogue and **stored to the HBM pool** (always
   HBM; M3 adds L2-cache residency via wave scheduling, not SMEM).
4. **MMA loop (warp 8):** identical primitive and k_tile loop; only difference
   is K dimension (`intermediate_size * 2 // 16` for L1, `intermediate_size //
   16` for L2). Use `BlockPhase` constexpr branch on `k_tile_cnt` and per-phase
   `tCtAcc` layout.
5. **Epilogue (warps 0-3):** branch on `BlockPhase`:
   - `Linear1`: today's SwiGLU + FP4 requant + SFC + **route-weight multiply
     applied to up×silu(gate) product before FP4 requant** (DeepGEMM L1 epi,
     not combine). Store destination = `l1_token_buffer` in the HBM pool.
   - `Linear2`: FC2's topk scatter + BF16 store to **`combine_token_buffer`**
     (per-topk slot, NOT `out`) when M6 runs; falls back to `out` at M2.6 for
     single-rank smoke tests.
6. **Scheduler (warp 10):** port `MegaMoEScheduler::for_each_block` (221 lines,
   port verbatim). State machine: `next_phase ∈ {Linear1, Linear2}`, wave
   interleaving controlled by `kNumExpertsPerWave`. Scheduler reads
   `expert_recv_count_sum[e]` spin-waiting until all ranks × all SMs have
   contributed (high 32 bits = `kNumSMs × kNumRanks`).

---

## 5. M2 edit plan — specific changes to `blockscaled_contiguous_mega_moe_fusion.py`

All line numbers below are the expected location in the M1 baseline
(current file is a verbatim copy of FC1 with class renamed).

**E1. Imports** (L29-50): add
```python
from cutlass.utils import TensorMapManager  # in-kernel descriptor swap
```
and an enum for `BlockPhase` (local module const, not a real Enum class to keep
`constexpr` friendly):
```python
PHASE_LINEAR1 = cutlass.const_expr(0)
PHASE_LINEAR2 = cutlass.const_expr(1)
```

**E2-E14** — see previous doc revision preserved in memory
(`project_megamoe_fusion_design.md`). Detail unchanged for M2.5-M2.7. Summary:
interface widening, host-side TMA descriptor prolog doubling, scheduler `phase`
field, LDGSTS/TMA/MMA/epilogue per-phase branches. Estimated delta +700-1000
lines on top of M1 baseline; no deletions.

---

## 6. Python op registration (M2.7 onwards)

Add `torch.ops.trtllm.cute_dsl_nvfp4_mega_moe_blackwell` in
`custom_ops/cute_dsl_custom_ops.py` parallel to:
- existing `cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell` (L2221)
- existing `cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell` (L1442)

New runner class `MegaMoeKernelRunner(TunableRunner)` with
`kernel_class = BlockScaledMegaMoeFusionKernel`. Inputs = union of FC1+FC2
buffers + **workspace + sym-buffer offsets** (M5+). Pool_tensor allocated
inside the op (`torch.empty` at M2.7 and M3 — the pool stays in HBM; M3's
wave-scheduling change governs which SM reads the pool so that HBM lines land
in L2 cache).

`MegaMoE.forward_impl` evolution:
- M2.6 + M2.7: replace FC1→memset→FC2 trio with single call, keep
  `CommunicationFactory` for dispatch/combine.
- M5: drop `comm.dispatch()` call; instead pre-publish quantized input to
  `input_token_buffer` (symm memory) and pass pointer + rank offsets.
- M6: drop `comm.combine()` / `reducescatter_or_allreduce`.

---

## 7. M3 preview — HBM pool + wave scheduler + L2 cache residency

> **⚠️ Corrected 2026-04-22 after reading
> `/home/scratch.xxi_sw_2/trtllm/deepseek_mega_moe_discussion.md`. Earlier
> drafts said "SMEM pool in M3"; that is architecturally impossible because
> FC1 and FC2 tiles run on different SMs under wave scheduling, and SMEM is
> per-SM. The DeepGEMM reference kernel keeps the pool in HBM
> (`l1_token_buffer`); M3's optimization is to route producer/consumer tiles
> to the same SM so the HBM transfer hits L2 cache.**

Perf-critical. Three coupled changes:

1. **Pool layout — stays in HBM** (ref: DeepGEMM
   `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` `l1_token_buffer`). The
   pool is a HBM ring of `kPoolBlocks` tiles of FP8-quantised Linear1 output.
   Linear1 epilogue writes via TMA store; Linear2 A-loader reads via TMA load
   (or LDGSTS at M2.7 transitional). The L2 cache is the de-facto "fast
   storage" when same-SM scheduling applies.

2. **Arrival synchronisation — two distinct primitives** (see
   `reference_deepseek_mega_moe.md` §4 for the contract):
   - `l1_arrival_count[pool_block]` — a **counter** (uint32) incremented by
     the Linear1 epilogue each time a tile is written. FC2 A-loader spins
     until it crosses the threshold (M3). Same counter is also touched by
     the dispatch warp when it finishes pulling a token (+1 per token) —
     at M5 FC1 A-loader spins on the same counter waiting for dispatch, and
     the Linear1 epilogue increment moves to `l2_arrival_mask` instead.
   - `l2_arrival_mask[pool_block]` — a **bitmask** (uint64), NOT a counter.
     Each bit represents "one FC1 N-block satisfied". FC2 K-block needs 2
     adjacent bits (`const uint64_t needed = 3ull << (k_block_idx * 2);`).
     FC2 A-loader does `while ((ptx::ld_acq_gpu(ptr) & needed) != needed);`.
   Both use `llvm.inline_asm` to emit `red.or.release.gpu.b64` /
   `red.add.release.gpu.u32` (cuteDSL has no first-class binding; matches
   DeepGEMM approach in `ptx/ld_st.cuh`).

3. **Wave-interleaved scheduler** (port of
   `scheduler/mega_moe.cuh::MegaMoEScheduler::for_each_block`): block index
   order is `Wave 0 FC1 tiles → Wave 0 FC2 tiles → Wave 1 FC1 tiles → ...`
   where each wave spans `kNumExpertsPerWave` experts. FC1 and FC2 tiles of
   the same wave can land on the same SM, so the HBM pool read by Linear2
   hits L2 cache. Parameter `kNumExpertsPerWave` tunable per
   `(E, H, I, topk)`.

M3 is perf milestone #1. Expected gain split by scenario
(ref: `reference_deepseek_mega_moe.md` §6.3):
- **Prefill (large batch):** tile-level 3-way overlap (dispatch↔FC1↔FC2) +
  wave-quant amortisation. Dominant in compute-bound regimes.
- **Decode (small batch):** ~5 saved kernel launches at ~10-20 µs each
  (~50-100 µs absolute savings). Tile-level overlap window is too narrow to
  matter; SM utilisation is already low due to token scarcity.

---

## 8. Test plan

**Per-milestone anchor tests** (live at `tests/unittest/_torch/modules/moe/test_mega_moe.py`):

| Milestone | Anchor name | Reference | Constraint |
|-----------|-------------|-----------|-----------|
| M1-M2.4 | `test_mega_moe_fusion_kernel_linear1_matches_fc1_baseline` | standalone FC1 kernel | byte-equal (uint8 view, trim `[:max_valid_row]`) |
| M2.5 | same + new `_with_linear2_gated` variant | FC1 baseline | byte-equal (Linear2 DCE'd under const_expr) |
| M2.6 | `test_mega_moe_fused_kernel_matches_two_kernel` | FC1+FC2 baseline through `MegaMoE.forward_impl` | NVFP4 atol |
| M2.7 | same | same | NVFP4 atol |
| M3 | same | same | NVFP4 atol; nsys trace shows FC1/FC2 MMA overlap |
| M4 | `test_mega_moe_m4_hello_world_remote_pull` | hand-written expected buffer | byte-equal (1 rank pulls from peer) |
| M5 | `test_mega_moe_fused_dispatch_matches_reference[world_size∈{2,4}]` | `MegaMoE` with `CommunicationFactory.dispatch` still at Python | NVFP4 atol on multi-rank |
| M6 | `test_mega_moe_fused_combine_matches_reference[world_size∈{2,4}]` | same as above + Python combine | NVFP4 atol on multi-rank |
| M7 | same + `test_mega_moe_pathological_routing` (all tokens→1 expert) | | perf degrades gracefully, no hang |

**OCI B200 execution:** `sbatch --partition=batch --qos=short --time=02:00:00
--gres=gpu:4` of `run-unit-megamoe.slurm` for every milestone. Computelab is
insufficient per evaluation rubric C6.

---

## 9. Open questions / risks

- **LDGSTS in Linear2 is wasteful but correct** — idle warps 4-7 in Linear2?
  Or repurpose them to drive pre-fetch of the next Linear1 wave? Deferred to
  M3.
- **TMEM size budget**: `tCtAcc` must hold the max-N of L1 and L2. For E=8,
  H=2048, I=4096, top_k=2 the two phases have comparable N. Validate on the
  `(16, 1024, 1024, 2)` case first.
- **SFC layout compat**: Linear1 SFC layout must be a legal SFA input for
  Linear2 UTCCP. Highest-probability M2.6 bug source.
- **`TensorMapManager` interaction with 2-CTA multicast**: descriptor update
  must be performed by a single warp then made visible to both CTAs in the
  cluster. Use `cute.arch.cluster_sync` around the swap.
- **(M4 PoC) cuteDSL cross-rank TMA**: is there a first-class primitive, or
  must we emit PTX via `llvm.inline_asm`? See §11.1.
- **(M4 PoC) Symm memory ownership**: TRT-LLM's `NVLinkOneSided` /
  `DeepEPLowLatency` already allocate symm buffers — can we borrow them or
  must we allocate independently? If NVSHMEM-based, does the kernel have
  access to `nvshmem_ptr`? See §11.2.
- **(M5 risk) register budget**: dispatch(48) + {ldgsts(40), mma(40), tma_b(40),
  sched(40), sync_transform(40)} + epilogue(208) @ 12 epilogue warps exceeds
  64512 on some cluster geometries. Budget calc must be redone per-config.
- **(M6 risk) combine warp piggybacks on epilogue 208-reg group.** That means
  epilogue warps run FC2 BF16 store → switch to combine loop in the same
  warps. Mixing kernels on same register budget requires careful live-range
  management. Same approach DeepGEMM takes (1 warp class, 2 loop bodies).
- **Redundant NVLink pull** (`deepseek_mega_moe_discussion.md §2.3`): each
  token pulled ~2× by routing design. DeepGEMM inherits, we inherit.
  `pull-unique + local scatter` optimization saves 50% NVLink at cost of
  +T·hidden HBM — deferred beyond M7.
- **Work-stealing absent** (§5.2): SM locks `(expert, m, n)` and spin-waits.
  Pathological routing imbalance → serial tail. Inherited.

---

## 10. Scope boundaries (revised 2026-04-22)

> **This section replaces the earlier "NOT fused — delegated to outer Python
> layer" bullet list. Dispatch and combine are now fused.**

### 10.1 Fused INSIDE the kernel (M1 through M6)

- **Dispatch Step 1** — write routing metadata (uint32 `token_topk_idx`) to
  each destination rank's symm-mem `src_token_topk_idx` table (M5)
- **Dispatch Step 2** — NVLink barrier `kBeforeDispatchPullBarrierTag` (M5)
- **Dispatch Step 3** — per-token `cp.async.bulk` one-sided TMA pull of FP8
  hidden row (`kHidden` bytes) from peer rank to local SMEM, then TMA store to
  HBM `l1_token_buffer`; plus manual copy of SF and topk-weight from peer HBM
  to local HBM (M5)
- **`l1_arrival_count` counter** — dispatch `red.add.release.gpu.u32`; FC1
  A-loader `ld.acquire.gpu` spin (M5, integrating with M3 infrastructure)
- **Linear1 (FC1) GEMM**: hidden → 2·intermediate NVFP4, 2-CTA UMMA (M1)
- **SwiGLU activation** (halving N) (M1)
- **Route-weight multiply** (applied in L1 epilogue before FP4 requant,
  matching DeepGEMM) (M2.5)
- **FP4 requant** of Linear1 output + SFC generation (M1)
- **`l2_arrival_mask` bitmask** — L1 epilogue `red.or.release.gpu.b64`; FC2
  A-loader `ld.acquire.gpu` spin on per-K-block mask (M3)
- **Linear2 (FC2) GEMM**: intermediate → hidden, 2-CTA UMMA (M2.7)
- **Topk-weighted scatter + BF16 store to local `combine_token_buffer`**
  (Linear2 epilogue, M2.7 local / M6 symm mem destination)
- **NVLink combine fan-out** — FC2 epilogue `sym_buffer.map(dst, rank) = part`
  writes partials to owning rank's combine slot (M6)
- **NVLink barrier `kBeforeCombineReduceBarrierTag`** (M6)
- **Combine reduction** — combine warp (on epilogue warpgroup) reads local
  `combine_token_buffer` slots, reduces topk partials, TMA-stores to `y` (M6)
- **Wave-interleaved scheduler** (M3)

### 10.2 Still OUTSIDE the kernel (stays in `MegaMoE.forward_impl`)

- **Routing**: `routing_method.apply` — Python. Produces
  `token_selected_experts` and `token_final_scales`.
- **FP4 input quantization**: `torch.ops.trtllm.fp4_quantize`. Output is
  published to `input_token_buffer` in symm memory (§11.2).
- **moe_sort** for per-rank permutation of local tokens → `tile_idx_to_*`
  scheduling metadata (still needed so that the scheduler's initial state
  knows per-expert counts after routing).
- **Workspace zero-init** — `cudaMemsetAsync` before launch.

### 10.3 Accepted limitations (documented for reviewers)

| Limitation | Why we accept it |
|-----------|-----------------|
| Combine has no overlap with FC2 | Inherent to global NVLink barrier; same as DeepGEMM (§4.4) |
| Redundant NVLink pull (~2× traffic) | Deferred; DeepGEMM inherits |
| SM-locked tile dispatch (no work-stealing) | Relies on routing load-balance statistics |
| Wave-boundary serialisation | Inherent to scheduler structure |
| Training path not supported | Per-token scale + fused intermediate tensors complicate backward |
| No cross-layer overlap | Fully-fused kernel cannot overlap with adjacent layer compute |

---

## 11. M5 design — Dispatch warp group

### 11.1 cuteDSL primitives required

| Primitive | PTX | Status in cuteDSL |
|-----------|-----|-------------------|
| One-sided TMA load | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` | `cute.arch.cp_async_bulk_*` exists for local; remote via `sym_buffer.map`'d pointer untested. **M4 PoC item.** |
| NVLink barrier signal | `red.release.sys.global.add.s32` | No first-class; emit via `llvm.inline_asm` (same as DeepGEMM's `ptx/ld_st.cuh:red_add_rel_sys`) |
| NVLink barrier wait | `ld.acquire.sys.global.u32` with 30s timeout | Same: `llvm.inline_asm` |
| Arrival counter increment | `red.release.gpu.global.add.u32` | `llvm.inline_asm` (same mechanism as M3's bitmask) |
| Arrival counter wait | `ld.acquire.gpu.global.u32` | Same |
| mbarrier arrive-with-tx-bytes | `mbarrier.arrive.expect_tx` | `cute.arch.mbarrier_arrive_and_expect_tx` exists |
| mbarrier wait-and-flip | `mbarrier.try_wait.parity` | `cute.arch.mbarrier_wait` exists |

**M4 PoC result decides M5 viability.** If cross-rank TMA pull fails in
cuteDSL (e.g., address translation through `sym_buffer.map` not validated),
fallback is to write the dispatch warp body as ~100 lines of `llvm.inline_asm`
PTX wrapped in a cute.jit function. That is feasible but raises maintenance
cost — document the fallback path explicitly before committing to M5.

### 11.2 Symmetric memory integration

TRT-LLM already has `NVLinkOneSided` (`communication/nvlink_one_sided.py`)
and `DeepEPLowLatency` that allocate symmetric NVLink buffers. The plan:

1. **M4 step 1** — identify the allocation backend (NVSHMEM? CUDA VMM peer?).
   Grep `nvshmem` / `cuMemSetAccess` in
   `tensorrt_llm/_torch/modules/fused_moe/communication/`.
2. **M4 step 2** — if NVSHMEM: we get `nvshmem_ptr(local_ptr, peer_rank)`
   giving a translated pointer usable in `cp.async.bulk`. If CUDA VMM peer
   mapping: we get a single VA range visible on all ranks, and "translation"
   is just `base[peer] - base[self] + local_ptr` — our own `sym_buffer.map`
   replica of DeepGEMM's.
3. **M5** — pass `(base_ptrs: List[Tensor], self_rank: int)` as a
   TensorMapManager-wrapped tuple into the kernel. The tensor layer already
   supports passing List[Tensor] as a Dict of DeviceTensors.

Buffers to place in symm memory (total per-rank size ≤ few hundred MB):

| Buffer | Size per rank | Role |
|--------|--------------|------|
| `input_token_buffer` | `num_max_tokens_per_rank × kHidden` FP8 bytes | source for dispatch pull (this rank's post-fp4_quantize activation) |
| `input_sf_buffer` | `num_max_tokens_per_rank × kHidden / scaling_vector_size` FP8 bytes | source SFs |
| `input_topk_idx_buffer` | `num_max_tokens_per_rank × kNumTopk × int64` | routing decisions |
| `input_topk_weights_buffer` | `num_max_tokens_per_rank × kNumTopk × float32` | route weights |
| `combine_token_buffer` | `num_max_tokens_per_rank × kNumTopk × kHidden` BF16 bytes | combine partial store |
| `workspace` | `layout::Workspace::get_num_bytes()` ≈ a few MB | barrier counters, arrival counts, src_token_topk_idx table, token_src_metadata |

### 11.3 Warp layout expansion (M1 → M5)

| Warp id range | Role | Regs |
|---------------|------|------|
| `[0, 4)` | Epilogue + combine (M1 + M2.7 + M6) | 208 |
| `[4, 4 + kNumDispatchWarps)` | **Dispatch** (M5) | 48 |
| next warp | LDGSTS-A (M1) | 40 |
| next warp | TMA-B (M1) | 40 |
| next warp | MMA (M1) | 40 |
| next warp | Scheduler (M1) | 40 |
| next warp | Sync-transform (M1) | 40 |

`kNumDispatchWarps` ≥ 4 (a full warpgroup, DeepGEMM constraint); commonly
`kNumDispatchWarps = 4` or 8 depending on pool fill rate. Register budget
check (64512 per block):

    208 × (4 × 32) + 48 × (kNDW × 32) + 40 × (5 × 32) ≤ 64512
    → kNDW ≤ ~24, easily satisfied with kNDW=4.

### 11.4 Dispatch warp body (port of DeepGEMM L340-L600)

**Phase A — metadata publish** (all dispatch warps in parallel):
```text
For each (token_idx, topk_slot) assigned to this warp:
    expert_idx   = input_topk_idx_buffer[token_idx, topk_slot]
    if expert_idx < 0: skip
    dst_rank     = expert_idx / kNumExpertsPerRank
    dst_local_e  = expert_idx % kNumExpertsPerRank
    dst_slot     = atomicAdd_block(smem_expert_count[expert_idx], 1)  # init = global expert_send_count atomic
    dst_ptr      = workspace.src_token_topk_idx_ptr(dst_local_e, my_rank, dst_slot)
    * sym_map(dst_ptr, dst_rank) = token_idx * kNumTopk + topk_slot   # uint32 write
ptx::sync_aligned(kDispatchThreads, kDispatchBarrierIdx)
```

**Phase B — NVLink barrier** (SM 0 only, `comm::nvlink_barrier`):
```text
if sm_idx == 0:
    if thread_idx < kNumRanks:
        red_add_rel_sys(sym_map(signal_ptr, thread_idx), +1 or -1)
    if thread_idx == 0:
        spin_until ld_acq_sys(signal_ptr) == target  (30s timeout)
grid_sync()
```

**Phase C — per-token pull** (round-robin over flat pool_token_idx):
```text
token_idx = sm_idx * kNumDispatchWarps + warp_idx
stride    = kNumSMs * kNumDispatchWarps
while True:
    # advance to the expert that owns this token_idx (cached per-expert counts)
    while token_idx >= expert_end_idx:
        if ++current_expert_idx >= kNumExpertsPerRank: exit
        ...
    # min-peel round-robin rank selection using stored_rank_count[]
    (src_rank, token_idx_in_rank) = peel_next_rank(current_expert_idx, token_idx)
    src_token_topk_idx = workspace.src_token_topk_idx_ptr(local_e, src_rank, token_idx_in_rank)
    src_token_idx      = src_token_topk_idx / kNumTopk
    src_topk_idx       = src_token_topk_idx % kNumTopk
    # TMA one-sided pull (hidden bytes) via mbarrier completion
    if elect_one:
        cp_async_bulk(
            pull_buffer.get_base_ptr(),                       # dst SMEM
            sym_map(input_token_buffer.get_data_buffer(src_token_idx).ptr, src_rank),
            pull_mbarrier, kHidden)
    mbarrier_arrive_expect_tx(pull_mbarrier, kHidden)
    mbarrier_wait(pull_mbarrier, phase)
    # Plain lane-striped HBM→HBM copy for SF (kHidden / 128 uint32s)
    for u in uint32_sf_words: local_sf_buf[u] = sym_map(remote_sf_buf, src_rank)[u]
    # HBM→HBM weight (float, 1 word)
    local_weight = sym_map(remote_weight_ptr, src_rank)[0]
    # Stage pull_buffer → HBM l1_token_buffer[pool_token_idx] via TMA store
    cp_async_bulk_store(l1_token_buffer_ptr(pool_token_idx), pull_buffer)
    tma_store_arrive(); tma_store_wait<0>()
    # Publish arrival
    red_add_rel_gpu(l1_arrival_count_ptr[pool_block_idx], 1)  # +1 per token
    token_idx += stride
```

### 11.5 FC1 A-loader modification (M5)

Current (M1-M2.7): token-gather via `token_id_mapping_tensor` from raw
`a_l1` tensor.

After M5: pool-read from `l1_token_buffer` at `pool_base + pool_block_idx ×
BLOCK_M × kHidden`, preceded by:
```text
# M5 addition to LDGSTS-A warp body (Linear1 path)
ptr = workspace.l1_arrival_count_ptr(pool_block_idx)
expected = scheduler.get_valid_m<false>()  # num tokens assigned to this pool_block
while ld_acq_gpu(ptr) != expected: spin
```
Then LDGSTS from the pool (contiguous, no `token_id_mapping` needed — the
pool is already in permuted layout).

### 11.6 Python layer changes (M5)

This subsection describes the pure M5 milestone before M6 direct combine is enabled. The current
transitional runtime can already pair M5 dispatch diagnostics with M6 direct combine; see §11.7.

```python
# fused_moe_mega.py forward_impl (M5 version)
# Step 1 — routing (unchanged)
# Step 2 — FP4 quantize
#          → **instead of writing to a local tensor, publish to
#            self._input_token_buffer (symm memory, pre-registered)**
# Step 3 — REMOVED: CommunicationFactory.dispatch() no longer called
# Step 4 — moe_sort (unchanged, produces scheduling metadata)
# Step 5 — single cute_dsl_nvfp4_mega_moe_blackwell kernel
# Step 6 — REMOVED: memset (now inlined in dispatch phase B)
# Step 7 — combine still in Python at M5; merged into kernel at M6
# Step 8 — reducescatter_or_allreduce stays at M5
```

`torch.ops.trtllm.cute_dsl_nvfp4_mega_moe_blackwell` gains:
- `input_token_buffer: Tensor` (symm, shape `[num_ranks, num_max_tokens_per_rank, kHidden]`; caller passes own rank's slice as input plus all-ranks base-pointer table)
- `input_sf_buffer`, `input_topk_idx_buffer`, `input_topk_weights_buffer`
- `workspace: Tensor` (uninitialised, zero'd by kernel prolog)
- `base_ptrs: List[int]` + `self_rank: int` (for `sym_map` table)

### 11.7 Transitional runtime policy

- The default target remains in-kernel M5 dispatch plus direct M6 combine, reported by
  `full_fusion_final_kernel_path`.
- The standalone M5 materializers are not the success criterion for the optimized path. They exist for
  fallback/debug validation while the in-kernel path is staged in.
- `m5_dispatch_materialize_strategy` reports which standalone fallback/debug implementation ran after its
  prerequisites passed. Use `full_fusion_final_kernel_path` and `full_fusion_final_kernel_ready` to identify the
  optimized M5/M6 path.
- Reconstruction materializers must stay behind `reconstruction_materialize` so normal fallback does not silently
  rebuild the dispatch pool in Python/Torch.

---

## 12. M6 design — Combine warp body

Combine lives in the same kernel, sharing the 208-register epilogue warpgroup.
Body executes **after** the Linear2 epilogue has fanned out BF16 partials to
peer ranks via `sym_map`.

### 12.1 Linear2 epilogue modification (M6)

Current (M2.5): scatter-add to local `out` using `permuted_idx_to_expanded_idx`.

After M6: scatter via `sym_map` to **owning rank's combine slot**:
```text
dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx)
dst_ptr   = advance(dst_token.ptr, n_idx * sizeof(bf16) + lane16 * sizeof(float4))
* sym_map(dst_ptr, dst_rank) = bf16x8_packed
```
where `(dst_rank, dst_token_idx, dst_topk_idx) = TokenSrcMetadata[pool_token_idx]`
written during dispatch (§11.4, last line).

### 12.2 NVLink barrier `kBeforeCombineReduceBarrierTag`

Symmetric with dispatch's Phase B: SM 0 signals, all SMs grid-sync. After
this barrier, every rank's `combine_token_buffer` has all topk partials for
every local token.

### 12.3 Combine reduction loop

```text
# Per-token local reduction (lane-token striped)
for token_idx in strided range:
    # topk mask: which topk slots actually routed
    topk_mask = ballot(lane_idx < kNumTopk and input_topk_idx_buffer[token_idx, lane_idx] >= 0)
    acc = float4(0)
    for slot in iterate_bits(topk_mask):
        partial = load(combine_token_buffer.rank_buffer(slot).data_buffer(token_idx))  # via TMA into SMEM stage
        acc += bf16x8_to_float(partial)
    # Cast to bf16, TMA store to final y
    y[token_idx] = float_to_bf16x8(acc)
```

Uses 2-stage SMEM buffer (`combine_load_buffer`) per warp and per-warp
mbarriers (`combine_barriers`). Same 208-reg budget as the epilogue.

### 12.4 Python layer changes (M6)

```python
# fused_moe_mega.py forward_impl (M6 version)
# Step 8 REMOVED (reducescatter_or_allreduce)
# Kernel output directly writes y
```

---

## 13. M4 Infrastructure PoC — sequenced checklist

Before writing any M5/M6 kernel code, the following must work end-to-end
in isolation.

### 13.1 PoC kernel spec

File: `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/mega_moe_dispatch_poc.py`
(new, M4 lifetime only; delete or move contents into mega kernel at M5).

Behaviour: 1 cuteDSL kernel, 1 warp, does:
1. For token_idx 0..127, pull `kHidden` FP8 bytes from peer rank
   (`sym_map(peer_input_buffer, (my_rank + 1) % num_ranks)`) into local SMEM
   via `cp.async.bulk` with mbarrier completion.
2. TMA-store to local HBM buffer.
3. At end, `red.add.release.gpu.u32` to a counter; SM 0 spins on it until
   `== num_tokens`.

### 13.2 Test

`tests/unittest/_torch/modules/moe/test_mega_moe_m4_poc.py` (MPIPoolExecutor
world_size=2 and 4). Expected: each rank's local HBM buffer contains the
exact bytes of `(self+1)%num_ranks`'s input buffer. Byte-equal assertion.

### 13.3 Symm memory source

- If TRT-LLM's `NVLinkOneSided` exposes a symm-allocator hook:
  borrow via
  `from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided
  import NVLinkOneSided; comm = NVLinkOneSided(...); sym_buf =
  comm.alloc_symm(num_bytes)`.
- If it does not: call the underlying lib directly (NVSHMEM via
  `tensorrt_llm._torch.utils.nvshmem` if present, else CUDA VMM via
  `torch.cuda.CUDAMemPool` peer mapping).
- Record the chosen path in this doc §11.2 and in memory
  `feedback_symm_memory_trtllm.md`.

### 13.4 Risk-resolved output

After M4 completes, the following statements must be answered Yes/No in a
short followup doc (`tests/unittest/.../test_mega_moe_m4_poc.md`):
1. Does cuteDSL `cp.async.bulk` accept a remotely-translated pointer via
   `sym_map` style offset addition? If No, is `llvm.inline_asm` PTX path
   viable?
2. Does `red.release.sys.global.add.s32` work from within a cuteDSL kernel?
3. Does the TRT-LLM stack already have a symm allocator we can use, or
   must we introduce one?
4. Is the dispatch warpgroup register budget (48 regs × up to 8 warps =
   ≤ 12288 regs within the 64512 total) feasible alongside M3's warp
   layout?

**Gate**: if all four Yes, proceed to M5. If any No, write a remediation
mini-design (inline PTX fallback, or symm allocator port) before M5.

---

## 14. Next-session playbook

### 14.1 In-flight (M2.5)
> **Before writing any cuteDSL code: consult `reference_cutedsl_examples.md`**
> (grep `/home/xxi/sc2/cute_dsl_kernel_library/dsl_kernels/` first, then
> `/home/xxi/sc2/cutlass/examples/python/CuTeDSL/blackwell/`, then
> `/home/xxi/sc2/dynamic-kernel-generator/cutegen/include/`). Do **not** write
> cuteDSL API from memory — find precedent in these paths.

1. Read this doc plus `project_megamoe_m25_plan.md` (memory). Open
   `blockscaled_contiguous_mega_moe_fusion.py` in the megamoe worktree.
2. Implement M2.5 in 3 sub-commits (partition helper / setup block / main
   tile loop const_expr branch). Every commit runs OCI anchor 4/4.
3. M2.6 — flip `enable_linear2=True`, update `fused_moe_mega.py` Step 5
   to new op, add `test_mega_moe_fused_kernel_matches_two_kernel`.
4. M2.7 — add `TensorMapManager` + second B TMA descriptor + Linear2 SFA.
5. M3 — port `MegaMoEScheduler::for_each_block` + `l2_arrival_mask` bitmask
   via `llvm.inline_asm`.

### 14.2 After M3 (starting the dispatch branch)
6. **M4 PoC week** — answer §13.4 Yes/No. Don't touch the mega kernel file
   during this week; it's a parallel probe.
7. **M5** — add Dispatch warp group to the mega kernel (§11). Remove
   `CommunicationFactory.dispatch` from `forward_impl`. First validate on
   single rank (dispatch warp exits trivially with `kNumRanks=1`).
8. **M6** — Combine warp body (§12). Remove `reducescatter_or_allreduce`.
9. **M7** — autotune + pathological routing + NSys overlap validation.

### 14.3 Acceptance

Project closes when:
- All anchor tests in §8 green on OCI B200 (4-GPU, 2-GPU, 1-GPU).
- No `CommunicationFactory.*` or `reducescatter_or_allreduce` calls remain
  in `MegaMoE.forward_impl`.
- NSys trace (M7) confirms 3-way overlap: dispatch warps running while
  MMA warps running while FC2 A-loader warps running, on the same SM.
- Design doc + memory updated with M3-M7 outcome metrics.
