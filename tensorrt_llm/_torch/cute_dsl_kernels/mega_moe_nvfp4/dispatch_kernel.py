# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuTeDSL replica of mega_moe's dispatch kernel.

This file owns the single-kernel device-side replication of the dispatch slice
of `sm100_fp8_fp4_mega_moe.cuh` (lines 432-766). The mega_moe kernel runs four
sub-flows back-to-back inside one grid: ``Dispatch_Prep`` (CTA-local routing
metadata), ``Dispatch_Barrier`` (publish counts + grid sync + NVLink barrier),
``Dispatch_Pull`` (TMA pull tokens + SF + weights from peer ranks into the
local L1 pool), and an end-of-kernel NVLink barrier replacing the original
``Dispatch_Cleanup`` (we do not yet clear workspace here; cleanup is a separate
kernel in DSV4).

All four sub-flows are implemented: ``_dispatch_prep`` (3-round topk scan),
``_dispatch_barrier`` (peer count publish + grid sync + 3-stage NVLink barrier),
``_dispatch_pull`` (TMA pull token / SF LDG-STG / weight + arrival count), and
the kernel-tail NVLink barrier on slot 1.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint8, Uint32

try:
    from cutlass.cute import iket as _iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from .iket_compat import iket as _iket

from .grid_sync import software_grid_sync
from .ptx_helpers import (fns_b32, ldg_b32_raw, ldg_f32_raw,
                          red_add_release_sys_s32_raw,
                          red_add_release_sys_u64_raw, stg_b32_raw, stg_b64_raw,
                          tma_load_1d_raw, tma_store_1d)
from .sf_swizzle import sf_atom_int32_offset

# -----------------------------------------------------------------------------
# Shared-memory layout
# -----------------------------------------------------------------------------

# 4 dispatch warps per CTA, mirroring mega_moe `kNumDispatchWarps`.
_NUM_DISPATCH_WARPS = 4
# Per-warp pull-buffer staging slot is one full token body (in bytes). The
# byte count depends on the dispatch wire format (FP8 = 7168 B / token, NVFP4
# = 3584 B / token), so the SharedStorage layout is parameterised by
# ``hidden_bytes`` -- see ``_make_shared_storage`` below.
# Per-CTA expert histogram capacity. 512 covers DeepSeek V4 (384 experts)
# with ~25% headroom; ~0.5 KB unused SMEM is negligible.
_NUM_TOTAL_EXPERTS_MAX = 512
# CTA-wide thread layout, matches mega_moe (`sm100_fp8_fp4_mega_moe.cuh`):
# 4 dispatch warps + 8 epilogue warps = 12 warps = 384 threads per CTA. The
# epilogue slot is currently a no-op placeholder in this dispatch-only
# kernel; it participates only in the cross-warp-group named barrier
# (`kDispatchWithEpilogueBarrierIdx`) so the thread layout is identical to
# mega_moe -- letting this kernel drop into a fused MoE kernel later without
# changing dispatch logic.
_NUM_DISPATCH_THREADS = _NUM_DISPATCH_WARPS * 32  # 128
_NUM_EPILOGUE_WARPS = 8
_NUM_EPILOGUE_THREADS = _NUM_EPILOGUE_WARPS * 32  # 256
_NUM_TOTAL_THREADS = _NUM_DISPATCH_THREADS + _NUM_EPILOGUE_THREADS  # 384
# Named barrier id for dispatch <-> epilogue handshake. Distinct from
# barrier_id=0 used by `_sync_aligned_first_4_warps` (dispatch-internal sync).
_KDISPATCH_WITH_EPILOGUE_BARRIER_IDX = 1


def _make_shared_storage(hidden_bytes: int):
    """Factory: returns the CTA-wide SMEM layout for the given token wire size.

    Mirrors the relevant chunk of mega_moe's
    ``SharedStorage<UMMA_M=128, K=64, BLOCK_M=192, ...>`` from
    ``sm100_fp8_fp4_mega_moe.cuh``:

        * ``smem_send_buffers`` -> ``pull_buffer`` (per-warp ``hidden_bytes`` B).
          SF flows GMEM-to-GMEM via 32-lane LDG/STG with no SMEM hop, so
          mega_moe (and this replica) skip an SF SMEM staging buffer.
        * ``dispatch_barriers[kNumDispatchWarps]`` -> ``pull_mbar`` (one mbar
          per warp, 8 B each).
        * ``smem_expert_count[kNumExperts]`` -> the 1024 B histogram used to
          atomically count dispatch tokens per global expert (line 455) and
          later atomically allocate per-expert slot indices (line 471).

    Per-warp mbarrier phase bits live in a lane-0 register (matching mega_moe
    line 544 ``uint32_t pull_mbarrier_phase = 0;``). The XOR is hoisted out
    of the per-iteration ``if lane_idx == 0`` body so all lanes increment
    a uniform value -- only lane 0 reads it via ``mbarrier_wait``.

    The pull buffer width is parameterised by ``hidden_bytes`` so that FP8
    (7168 B/token) and NVFP4 (3584 B/token) JIT instances each get exactly
    the SMEM they need.
    """

    @cute.struct
    class SharedStorage:
        # 1 mbarrier per dispatch warp. Each is an 8-byte object.
        pull_mbar: cute.struct.MemRange[Int64, _NUM_DISPATCH_WARPS]
        # CTA-local histogram for the 256 global experts.
        smem_expert_count: cute.struct.MemRange[Int32, _NUM_TOTAL_EXPERTS_MAX]
        # 4 warps x hidden_bytes B token body (28 KB for FP8, 14 KB for NVFP4).
        # Holds the in-flight TMA load before it is TMA-stored into the L1 pool.
        pull_buffer: cute.struct.MemRange[Uint8,
                                          _NUM_DISPATCH_WARPS * hidden_bytes]

    return SharedStorage


# -----------------------------------------------------------------------------
# Inline-PTX helpers
# -----------------------------------------------------------------------------


def _atomic_add_block_u32(ptr, val: Int32) -> Int32:
    """CTA-scope u32 atomic add. Mirrors mega_moe ``atomicAdd_block`` at L455/L471."""
    return cute.arch.atomic_add(ptr, val, sem="relaxed", scope="cta")


def _load_smem_u32(ptr) -> Int32:
    """Non-atomic SMEM 32-bit load (``ld.shared.u32``)."""
    return cute.arch.load(ptr, Int32, ss="cta")


def _store_smem_u32(ptr, val: Int32) -> None:
    """Non-atomic SMEM 32-bit store (``st.shared.u32``)."""
    cute.arch.store(ptr, val, ss="cta")


def _atomic_add_relaxed_gpu_u64(ptr, val: Int64) -> Int64:
    """GPU-scope u64 atomic add (relaxed). Mirrors mega_moe L464 base-slot FAA."""
    return cute.arch.atomic_add(ptr, val, sem="relaxed", scope="gpu")


def _load_relaxed_gpu_u64(ptr) -> Int64:
    """Relaxed-order GPU-scope u64 global load. Mirrors mega_moe L507."""
    return cute.arch.load(ptr, Int64, sem="relaxed", scope="gpu")


def _atomic_add_release_sys_u64(ptr, val: Int64) -> Int64:
    """Cross-rank u64 atomic add (release+sys). Mirrors mega_moe L511-513."""
    return cute.arch.atomic_add(ptr, val, sem="release", scope="sys")


def _atomic_add_release_sys_s32(ptr, val: Int32) -> Int32:
    """Cross-rank s32 atomic add (release+sys). Mirrors barrier.cuh L50 NVLink signal fan-out."""
    return cute.arch.atomic_add(ptr, val, sem="release", scope="sys")


def _atomic_add_release_gpu_u32(ptr, val: Int32) -> Int32:
    """GPU-scope u32 atomic add (release). Mirrors mega_moe L707-708 arrival_count release-add."""
    return cute.arch.atomic_add(ptr, val, sem="release", scope="gpu")


def _load_acquire_sys_s32(ptr) -> Int32:
    """Acquire-order sys-scope s32 global load. Mirrors barrier.cuh L59 spin target."""
    return cute.arch.load(ptr, Int32, sem="acquire", scope="sys")


# TODO: This should be placed in the class with awareness of base fc12
_DISPATCH_INTRA_CTA_BAR_ID: int = 10


def _sync_aligned_first_4_warps() -> None:
    """``bar.sync <_DISPATCH_INTRA_CTA_BAR_ID>, 128`` aligned barrier across
    the 128 dispatch threads.

    See ``_DISPATCH_INTRA_CTA_BAR_ID`` comment above for why this is not NB 0.
    """
    cute.arch.barrier(
        barrier_id=_DISPATCH_INTRA_CTA_BAR_ID,
        number_of_threads=128,
    )


# -----------------------------------------------------------------------------
# Sub-flow inner functions
# -----------------------------------------------------------------------------


@cute.jit
def _dispatch_prep(
    storage,
    input_topk_idx_buffer,
    expert_send_count,
    src_token_topk_idx,
    peer_rank_ptr_mapper,
    sm_idx,
    warp_idx,
    lane_idx,
    *,
    num_tokens: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    num_total_experts: cutlass.Constexpr[int],
    local_rank: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
):
    """CTA-local topk scan + per-rank base-slot allocation + cross-rank advertise.

    Mirrors lines 432-475 of ``sm100_fp8_fp4_mega_moe.cuh``:

        * lines 442-450  -- the ``read_topk_idx`` lambda: each (warp, lane)
          reads one ``token, topk_slot`` and runs ``process(token_topk_idx,
          expert_idx)`` if the slot is active and the expert id is non-negative
          (mega_moe stores ``-1`` for masked-out tokens).
        * lines 454-456  -- first invocation: ``atomicAdd_block`` into the
          CTA-local SMEM histogram (one slot per global expert).
        * lines 460-466  -- second round: u64-packed FAA into per-rank
          ``expert_send_count``. Each SM contributes ``(1ull << 32) |
          local_count``; the fetched-old low32 is this SM's base slot, written
          back into ``smem_expert_count`` for use as the round-3 cursor.
        * lines 469-475  -- third round: re-scan topk and write per-token
          advertise cards to every peer rank's ``src_token_topk_idx`` view at
          ``[local_expert, local_rank, slot]``.

    All 128 dispatch threads (4 warps) must enter and leave together; lane and
    warp indexing follow mega_moe so the math here is identical to the C++
    version.
    """
    # Initialise the SMEM histogram to zero, cooperatively across all 128
    # dispatch threads (mirrors mega_moe ``st_shared_bulk`` semantically:
    # one st.shared.u32 per (thread, stride) instead of every thread writing
    # the whole region). cute.Tensor.fill(0) lowers to a vector splat that
    # each thread executes in full -- correct (all-zeros) but wasteful.
    thread_idx_in_dispatch = Int32(warp_idx * 32 + lane_idx)
    smem_count_ptr = storage.smem_expert_count.data_ptr()
    KNUM_DISPATCH_THREADS: cutlass.Constexpr[
        int] = _NUM_DISPATCH_WARPS * 32  # 128
    i = thread_idx_in_dispatch
    while i < Int32(num_total_experts):
        _store_smem_u32(smem_count_ptr + i, Int32(0))
        i = i + Int32(KNUM_DISPATCH_THREADS)
    _sync_aligned_first_4_warps()

    # Tokens-per-warp packing: lane 0..(num_topk-1) handles token (i+0)'s
    # topk slots, lane num_topk..(2*num_topk-1) handles token (i+1)'s, etc.
    # mega_moe defines `kNumTokensPerWarp = 32 / kNumTopk` and only the first
    # `kNumTokensPerWarp * kNumTopk` lanes are active per warp. For DSV4
    # (num_topk=6) that is 5 tokens/warp and 30 active lanes.
    tokens_per_warp: cutlass.Constexpr[int] = 32 // num_topk
    active_lanes: cutlass.Constexpr[int] = tokens_per_warp * num_topk
    num_dispatch_warps_per_grid: cutlass.Constexpr[
        int] = num_sms * _NUM_DISPATCH_WARPS

    base_token_for_warp = (sm_idx * _NUM_DISPATCH_WARPS +
                           warp_idx) * tokens_per_warp
    grid_token_stride = num_dispatch_warps_per_grid * tokens_per_warp

    # First round: count expert -> token edges.
    # Loop bound is dynamic (`num_tokens` is a runtime value passed via the
    # kernel signature), so this is a runtime `range`.
    t = base_token_for_warp
    while t < num_tokens:
        # Active lane test mirrors line 443:
        #     if (i + (lane_idx / kNumTopk) < num_tokens
        #         and lane_idx < kNumActivateLanes)
        token_offset_in_warp = lane_idx // num_topk
        token_global = t + token_offset_in_warp
        if lane_idx < active_lanes and token_global < num_tokens:
            # token_topk_idx_buffer is shaped [num_tokens, num_topk] in
            # mega_moe; the C++ reads the contiguous int64 element at
            # `i * kNumTopk + lane_idx`. We use the cuTeDSL 2D access which
            # produces the same address sequence.
            topk_slot = lane_idx % num_topk
            expert_id = Int32(input_topk_idx_buffer[token_global, topk_slot])
            if expert_id >= Int32(0):
                # CTA-scope atomic add into SMEM histogram (line 455).
                _atomic_add_block_u32(smem_count_ptr + expert_id, Int32(1))
        cute.arch.sync_warp()
        t += grid_token_stride

    _sync_aligned_first_4_warps()

    # ------------------------------------------------------------------
    # Second round: bit-packed FAA into per-rank ``expert_send_count``
    # ------------------------------------------------------------------
    # Mirrors lines 460-466 of ``sm100_fp8_fp4_mega_moe.cuh``:
    #
    #     for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
    #         const uint64_t send_value = (1ull << 32) | smem_expert_count[i];
    #         smem_expert_count[i] = static_cast<uint32_t>(
    #             ptx::atomic_add(workspace.get_expert_send_count_ptr(i),
    #                             send_value));
    #     }
    #
    # FAA delta high32 = ``1`` (each SM contributes a single publisher --
    # confirmed at line 462: ``(1ull << 32) | smem_expert_count[i]``), so after
    # all ``num_sms`` SMs finalise, ``expert_send_count[i].high32 == num_sms``
    # and ``.low32 == sum_over_SMs(local_token_count_for_expert_i)``.
    #
    # The fetched-old low32 is this SM's *base slot* for the round-3 advertise
    # write. We overwrite ``smem_expert_count[expert_id]`` with the base slot
    # so the round-3 ``atomicAdd_block`` returns ``base_slot + n_th-token-from
    # -this-SM-this-expert`` directly.
    EXPERTS_PER_PASS: cutlass.Constexpr[int] = _NUM_DISPATCH_WARPS * 32  # 128
    for offset in cutlass.range_constexpr(0, num_total_experts,
                                          EXPERTS_PER_PASS):
        expert_id = Int32(offset + warp_idx * 32 + lane_idx)
        if expert_id < Int32(num_total_experts):
            slot_ptr = smem_count_ptr + expert_id
            local_count = _load_smem_u32(slot_ptr)
            # Pack: high32=1 (this SM's publisher contribution), low32=local
            # token count. Cast through Int64 to avoid sign-extension on the
            # 32-bit count.
            delta = (Int64(1) << Int64(32)) | (Int64(local_count)
                                               & Int64(0xFFFFFFFF))
            old_packed = _atomic_add_relaxed_gpu_u64(
                expert_send_count.iterator + expert_id, delta)
            # ``Int64.to(Int32)`` truncates the high 32 bits and matches the
            # C++ ``static_cast<uint32_t>(old)``. Use the explicit Int32
            # constructor to make the lowering deterministic across cuTeDSL
            # versions where ``.to`` may not be defined on every wrapper.
            base_slot = Int32(old_packed & Int64(0xFFFFFFFF))
            _store_smem_u32(slot_ptr, base_slot)
    _sync_aligned_first_4_warps()

    # ------------------------------------------------------------------
    # Third round: cross-rank ``src_token_topk_idx`` advertise
    # ------------------------------------------------------------------
    # Mirrors lines 469-475 of ``sm100_fp8_fp4_mega_moe.cuh``:
    #
    #     read_topk_idx([&](token_topk_idx, expert_idx) {
    #         dst_rank_idx = expert_idx / kNumExpertsPerRank;
    #         dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
    #         dst_ptr = workspace.get_src_token_topk_idx_ptr(
    #             expert_idx % kNumExpertsPerRank, peer_rank_ptr_mapper.rank_idx,
    #             dst_slot_idx);
    #         *peer_rank_ptr_mapper.map(dst_ptr, dst_rank_idx) = token_topk_idx;
    #     });
    #
    # After round 2, ``smem_expert_count[expert_id]`` *holds the base slot*
    # for this SM. ``atomicAdd_block`` returns the previous value, which is
    # ``base_slot + n_th-token-from-this-SM-this-expert``. The cross-rank
    # write goes via ``peer_rank_ptr_mapper.map(src_token_topk_idx.iterator, dst_rank,
    # elem_off)``: same byte address on the destination rank's
    # symmetric heap, since all ranks share an identical layout.
    t = base_token_for_warp
    while t < num_tokens:
        token_offset_in_warp = lane_idx // num_topk
        token_global = t + token_offset_in_warp
        if lane_idx < active_lanes and token_global < num_tokens:
            topk_slot = lane_idx % num_topk
            expert_id = Int32(input_topk_idx_buffer[token_global, topk_slot])
            if expert_id >= Int32(0):
                dst_rank = expert_id // Int32(num_experts_per_rank)
                local_expert = expert_id % Int32(num_experts_per_rank)
                slot = _atomic_add_block_u32(smem_count_ptr + expert_id,
                                             Int32(1))
                token_topk_word = Int32(token_global * num_topk + topk_slot)
                # Cross-rank STG: peer rank ``dst_rank``'s
                # ``src_token_topk_idx[local_expert, local_rank, slot] =
                # token_topk_word``. Uses mega_moe ``peer_rank_ptr_mapper.map`` style:
                # one dynamic offsets[dst_rank] lookup against the param
                # bank instead of constexpr-fanout over ``world_size``
                # Python list entries.
                # Element offset = ((local_expert * R + local_rank) *
                # MAX_SLOT + slot) * 4 bytes (int32 element).
                MAX_SLOT_C: cutlass.Constexpr[int] = num_tokens * num_topk
                elem_off = (
                    (local_expert * Int32(world_size) + Int32(local_rank)) *
                    Int32(MAX_SLOT_C) + slot) * Int32(4)
                peer_addr = peer_rank_ptr_mapper.map(
                    src_token_topk_idx.iterator.toint(),
                    dst_rank,
                    Int64(elem_off),
                )
                stg_b32_raw(peer_addr, token_topk_word)
        cute.arch.sync_warp()
        t += grid_token_stride

    # No final _sync_aligned_first_4_warps() -- the dispatch_barrier entry runs
    # software_grid_sync (grid-wide), which subsumes the CTA-local barrier.


@cute.jit
def _dispatch_barrier(
    expert_send_count,
    expert_recv_count,
    expert_recv_count_sum,
    nvlink_barrier_signal,
    grid_sync_counter,
    peer_rank_ptr_mapper,
    sm_idx,
    warp_idx,
    lane_idx,
    *,
    num_sms: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    num_total_experts: cutlass.Constexpr[int],
    local_rank: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
):
    """Publish per-rank expert recv counts and execute the NVLink barrier.

    Mirrors lines 488-532 of ``sm100_fp8_fp4_mega_moe.cuh``:

        * lines 489-492 -- ``comm::grid_sync<kDispatchGridSyncIndex>``: cross-
          CTA grid sync inside this rank. We already have ``software_grid_sync``
          in ``grid_sync.py`` for this.
        * lines 502-516 -- SM-0 fan-out: read each global expert's
          ``expert_send_count`` (u64), split into ``recv_count`` (low 32 bits)
          and ``recv_sum`` (full 64 bits), and publish to each peer rank's
          ``expert_recv_count`` (rank,expert) and ``expert_recv_count_sum``
          (expert) via ``peer_rank_ptr_mapper.map(...)`` -> peer view.
        * lines 526-532 -- ``comm::nvlink_barrier<kBeforeDispatchPullBarrierTag>``
          three-stage NVLink barrier ensuring all ranks finished the publish.

    The mega_moe sequence is: grid_sync, then SM-0 publishes counts to peers,
    then the NVLink barrier itself runs with ``sync_prologue=False`` (the grid
    sync above already covers the prologue) and ``sync_epilogue=True`` (the
    pull stage that follows must see every peer's publish complete).
    """
    # Logical thread index within the dispatch warp group: ``warp_idx`` is
    # dispatch-LOCAL ([0, 4)) on every caller (standalone dispatch_kernel
    # places its dispatch warps at warp 0..3 and feeds ``warp_idx`` directly;
    # MegaMoE rebases by subtracting ``dispatch_warp_id[0] == 8`` before
    # calling this helper).  ``tid_in_group ∈ [0, 128)`` and the thread
    # with tid_in_group == 0 is the grid-sync leader.  See the docstring
    # of ``software_grid_sync`` for why this MUST NOT come from the hardware
    # ``%tid.x`` register.
    tid_in_group = warp_idx * Int32(32) + lane_idx

    # 1) Intra-rank grid sync (mega_moe line 489-492). After this point every
    #    SM in this rank has finished its round-3 advertise STGs to all peers.
    software_grid_sync(grid_sync_counter,
                       sm_idx,
                       num_sms,
                       tid_in_group,
                       num_threads=_NUM_DISPATCH_THREADS)

    # 2) Publish per-rank ``expert_send_count`` to every peer's
    #    ``expert_recv_count`` and ``expert_recv_count_sum`` (lines 502-516).
    #    Only SM 0 participates -- the work is O(kNumExperts) and the grid
    #    sync above guarantees all 256 entries are finalised.
    if sm_idx == 0:
        EXPERTS_PER_PASS: cutlass.Constexpr[
            int] = _NUM_DISPATCH_WARPS * 32  # 128
        for offset in cutlass.range_constexpr(0, num_total_experts,
                                              EXPERTS_PER_PASS):
            expert_id = Int32(offset + warp_idx * 32 + lane_idx)
            if expert_id < Int32(num_total_experts):
                dst_rank = expert_id // Int32(num_experts_per_rank)
                dst_local_expert = expert_id % Int32(num_experts_per_rank)
                # Relaxed load -- the grid sync above already paired with the
                # round-2 atomic FAA's release semantics (the grid sync ends
                # with a release-ordered atomic_add followed by an acquire-
                # ordered ld), so a relaxed load here is sufficient.
                status_u64 = _load_relaxed_gpu_u64(expert_send_count.iterator +
                                                   expert_id)
                # Cross-rank publish via ``peer_rank_ptr_mapper.map``: take the
                # local-rank base address and add ``offsets[dst_rank]``
                # for the peer-rank delta. Both targets are int64 element
                # types, so element offsets multiply by 8.
                #
                # Site #2: low 32 bits = token count -> peer ``expert_recv_count
                # [local_rank, dst_local_expert]``. Stored as u64 (zero-ext) to
                # match the existing layout (8 B/element).
                token_count_u32 = Int32(status_u64 & Int64(0xFFFFFFFF))
                erc_local_base = expert_recv_count.iterator.toint()
                erc_elem_off = (Int32(local_rank) * Int32(num_experts_per_rank)
                                + dst_local_expert) * Int32(8)
                erc_peer_addr = peer_rank_ptr_mapper.map(
                    erc_local_base,
                    dst_rank,
                    Int64(erc_elem_off),
                )
                stg_b64_raw(erc_peer_addr, Int64(token_count_u32))
                # Site #3: u64 sys-atomic add at peer ``expert_recv_count_sum
                # [dst_local_expert]``. After all ``world_size`` ranks add,
                # high 32 = world_size * num_sms, low 32 = total tokens.
                ercs_local_base = expert_recv_count_sum.iterator.toint()
                ercs_peer_addr = peer_rank_ptr_mapper.map(
                    ercs_local_base,
                    dst_rank,
                    Int64(dst_local_expert * Int32(8)),
                )
                red_add_release_sys_u64_raw(ercs_peer_addr, status_u64)
    _sync_aligned_first_4_warps()

    # 3) Cross-rank NVLink barrier (lines 526-532). Slot 0 is reserved for the
    #    pre-pull barrier in DSV4 (slot 1 will be used for the kernel-tail
    #    barrier). ``prologue_grid_sync=False`` because step 1 above already
    #    serves that purpose; ``epilogue_grid_sync=True`` to fan completion
    #    out across all CTAs before the pull stage starts.
    _nvlink_barrier_3stage(
        nvlink_barrier_signal,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        slot=0,
        num_sms=num_sms,
        world_size=world_size,
        local_rank=local_rank,
        prologue_grid_sync=False,
        epilogue_grid_sync=True,
    )


@cute.jit
def _dispatch_pull(
    storage,
    input_token_buffer,
    input_sf_buffer,
    input_topk_weights_buffer,
    src_token_topk_idx,
    expert_recv_count,
    expert_recv_count_sum,
    l1_token_buffer,
    l1_sf_buffer,
    l1_topk_weights_buffer,
    l1_arrival_count,
    token_src_metadata,
    peer_rank_ptr_mapper,
    sm_idx,
    warp_idx,
    lane_idx,
    *,
    num_sms: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    block_m: cutlass.Constexpr[int],
    sf_block_m: cutlass.Constexpr[int],
    # Granularity at which ``l1_arrival_count`` release-add notifies the
    # downstream GEMM consumer.  Decoupled from ``block_m`` (which only
    # controls how tokens are LAID OUT inside ``l1_token_buffer``) so the
    # fused fc12 kernel's per-task-tile spin sees exactly one counter slot
    # per work tile; see ``fc12_integrate_comm.md`` §4 (constraint C3) for
    # the full derivation.  When the standalone dispatch kernel is used
    # without a fused downstream, set ``cluster_tile_m == block_m`` to
    # recover the legacy per-pool-block counter layout.
    cluster_tile_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    hidden_bytes: cutlass.Constexpr[int],
    sf_uint32_per_token: cutlass.Constexpr[int],
    num_padded_sf_pool_tokens: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
    local_rank: cutlass.Constexpr[int],
):
    """Pull tokens, SF, and weights from peer ranks into the L1 pool.

    Mirrors lines 542-711 of ``sm100_fp8_fp4_mega_moe.cuh``. The body has six
    logical chunks:

        * lines 544-558  -- per-warp register state init:
          ``pull_mbarrier_phase = 0``, ``pull_buffer = warp's smem slot``,
          ``current_expert_idx = -1``, etc. We additionally init
          ``storage.pull_mbar[warp_idx]`` via ``mbarrier_init`` (1 thread
          arrival count). The phase bit is a lane-local Int32 register
          (matching mega_moe), not a SMEM slot.
        * lines 559-588  -- outer ``for token_idx = sm_idx * 4 + warp_idx; ;
          token_idx += kNumGlobalWarps`` loop with the "advance current expert
          until token_idx is in range" peeling and per-rank-count reload on
          expert change (uses ``expert_recv_count`` view).
        * lines 590-638  -- round-robin rank selection via iterative
          min-peeling.
        * lines 640-654  -- read ``src_token_topk_idx`` advertise card and
          ``tma_load_1d`` from peer rank's ``input_token_buffer`` into
          ``pull_buffer[warp_idx]``.
        * lines 656-671  -- SF LDG/STG: lane-parallel read of ``sf_uint32_per_token``
          uint32s from peer's ``input_sf_buffer`` and write into the
          atom-flat 1D ``l1_sf_buffer`` at the linear Int32 offset returned
          by ``sf_atom_int32_offset(sf_token_in_pool_axis, k_atom_idx,
          num_k_atoms=sf_uint32_per_token)``.  Source M position is
          ``sf_token_in_pool_axis = expert_sf_pool_block_offset * sf_block_m +
          token_idx_in_expert``; the swizzle helper takes that absolute M
          coord and the K-atom index and emits the cute SFA atom layout's
          linear Int32 position so the mma TMA-SFB load reads back the same
          bytes via ``tile_atom_to_shape_SF``.
        * lines 673-710  -- elect_one leader writes weight, waits on mbar,
          ``tma_store_1d`` into ``l1_token_buffer``, writes
          ``token_src_metadata``, ``cp_async_bulk_commit_group`` +
          ``cp_async_bulk_wait_group(0)``, then ``red_add_rel`` into
          ``l1_arrival_count`` (release-semantic atomic add) -- this is what
          the GEMM stage spins on.

    Round-robin rank-selection strategy
    -----------------------------------
    Warp-cooperative path matching mega_moe lines 590-638. Each lane in the
    warp caches *one* rank's remaining count in a register (lane ``r`` for
    ``r < world_size``; lanes ``>= world_size`` hold 0 so they fall out of
    every ballot). The round decision uses three SIMT primitives:

      * ``cute.arch.vote_ballot_sync(remaining_lane > 0)`` -> active mask
        (``__ballot_sync`` equivalent).
      * ``cute.arch.popc(mask)`` -> ``num_active_ranks``.
      * ``cute.arch.warp_redux_sync(v_for_min, "min")`` -> ``length``
        (``__reduce_min_sync`` equivalent; inactive lanes contribute INT_MAX).
      * ``fns_b32(mask, 0, slot_idx_in_round + 1)`` -> picked rank index in
        a single PTX instruction (mega_moe ``__fns`` equivalent).

    Peel-step is local to each lane: ``remaining_lane`` either drops by
    ``length`` (active) or stays at 0 (inactive). All 32 lanes exit each
    round with the same warp-broadcast scalars (``num_active_ranks``,
    ``length``, ``picked``), so the elected leader's downstream
    ``current_rank_in_expert_idx`` and ``token_idx_in_rank`` are identical
    across lanes; no shuffle broadcast is needed.
    """
    # ------------------------------------------------------------------
    # Per-warp mbarrier init (mega_moe lines 544 + dispatch_barriers[warp_idx]
    # construction earlier in the kernel)
    # ------------------------------------------------------------------
    # mega_moe pre-creates per-warp dispatch_barriers with arrive-count 1
    # in shared memory at kernel start. The cuTeDSL replica defers that
    # construction until just before the pull loop so the SMEM region's
    # lifetime is identical to mega_moe's. ``mbarrier_init`` initialises a
    # single mbarrier object with the given arrival count; the leader lane
    # of each warp owns its own mbar slot.
    #
    # ``MemRange`` does not support pointer arithmetic with a dynamic offset
    # nor dynamic ``__getitem__``; route reads/writes through a 1D tensor view
    # and route atomic-target / mbar-slot addresses through the raw data
    # pointer that ``MemRange.data_ptr()`` exposes.
    pull_mbar_ptr = storage.pull_mbar.data_ptr()
    pull_buffer_ptr = storage.pull_buffer.data_ptr()
    if lane_idx == Int32(0):
        cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
    # Make the init visible to the warp's elect_one consumer below. mega_moe
    # uses ``ptx::sync_unaligned`` at line 528; in the cuTeDSL replica the
    # round-3 close of ``_dispatch_barrier`` already aligned all 4 dispatch
    # warps via ``_sync_aligned_first_4_warps``, but the per-warp mbarrier
    # init still needs an intra-warp fence so the lane-0 store is observed.
    cute.arch.sync_warp()

    # NOTE: a cluster-level ``mbarrier_init_fence + cluster_arrive +
    # cluster_wait`` lived here previously, anticipating future 2-CTA cluster
    # fusion. With the placeholder epilogue warp group added (12 warps / CTA
    # but only the 4 dispatch warps enter this branch), only 128 of the 384
    # CTA threads would call ``cluster_arrive``; the cluster_wait would block
    # indefinitely on the missing arrivals. For cluster=(1,1,1) (current
    # standalone launch) the cluster sync is a no-op anyway, so it's removed
    # here. When fusing into a larger MoE kernel that uses 2-CTA cluster
    # TMEM allocation, re-add the cluster sync at kernel entry (before the
    # warp_idx dispatch/epilogue split) so all warps participate.

    # Per-warp mbarrier phase bit, register-resident on every lane (only
    # lane 0 reads it; the XOR is hoisted out of the elect_one body so all
    # lanes track the same value uniformly across iterations -- avoids an
    # scf.if SSA carry like OI-12 / OI-19). Mirrors mega_moe line 544.
    phase_bit = Int32(0)

    # ------------------------------------------------------------------
    # Per-warp register state. Mirrors mega_moe lines 552-557 verbatim.
    # ------------------------------------------------------------------
    # ``stored_rank_count[r]`` holds the un-peeled token count for source rank
    # ``r`` of the warp's current expert. World_size is at most 4 in DSV4 so
    # we keep the array Pythonically as four scalars (no kNumRanksPerLane
    # split is needed because all 4 fit in any single lane).
    current_expert_idx = Int32(-1)
    expert_start_idx = Int32(0)
    expert_end_idx = Int32(0)
    expert_pool_block_offset = Int32(0)
    # Running cumul of task-tile slots consumed by experts already finished
    # (analogous to ``expert_pool_block_offset`` but with ``cluster_tile_m``
    # granularity instead of ``block_m``).  Drives the release-add target
    # index ``task_tile_idx`` below so the fused fc12 fc1 spin sees one
    # counter slot per work tile.
    expert_task_tile_offset = Int32(0)
    # Independent SF-axis cumul.  Mirrors ``expert_pool_block_offset`` but
    # uses ``sf_block_m`` granularity instead of ``block_m``.  Needed because
    # ``block_m`` (token pool padding) and ``sf_block_m`` (SF pool padding,
    # dictated by the UTCCP 4x32 atom layout) are decoupled in MegaMoE:
    # reusing ``expert_pool_block_offset`` for the SF axis silently assumes
    # ``block_m == sf_block_m`` and misplaces every non-leading expert's SFs
    # by ``ceil(prev_valid, block_m) * sf_block_m - ceil(prev_valid,
    # sf_block_m) * sf_block_m`` SF rows.  Must stay in lockstep with
    # ``MoEFusedFc12SchedulerParams.current_sf_cumul`` (which advances by
    # ``ceil(prev_valid, sf_padding_block) * sf_padding_block``) -- the two
    # are the same quantity expressed in different units (sf-block count vs
    # absolute SF M-axis row).
    expert_sf_pool_block_offset = Int32(0)

    # Per-lane remaining count: lane ``r`` for ``r < world_size`` holds the
    # un-peeled token count for source rank ``r`` of the warp's current
    # expert; lanes ``r >= world_size`` hold 0 so they are filtered out by
    # the ballot. Mirrors mega_moe ``remaining[kNumRanksPerLane]`` register
    # array (with ``kNumRanksPerLane = ceil(world_size / 32) == 1`` for
    # world_size <= 32).
    stored_rank_count_lane = Int32(0)

    # ------------------------------------------------------------------
    # Per-expert total token count cache. Mirrors
    # ``MegaMoEScheduler::fetch_expert_recv_count`` +
    # ``get_num_tokens(e)`` from
    # ``DeepGEMM/deep_gemm/scheduler/mega_moe.cuh``.
    #
    # Layout: ``stored_num_tokens_per_expert[i]`` on lane ``j`` holds the
    # low-32 bits of ``expert_recv_count_sum[i * 32 + j]`` -- i.e. the
    # total token count for expert ``i*32 + j`` (across all source ranks),
    # which was finalised by ``_dispatch_barrier``'s sys-scope u64 add.
    # ``kNumExpertsPerLane`` chunks let us cover ``num_experts_per_rank``
    # > 32 (V2: 96 -> 3 chunks).
    #
    # Filling is one LDG per (lane, chunk) -- the 3-stage NVLink barrier
    # at the close of ``_dispatch_barrier`` already establishes
    # acquire+sys ordering w.r.t. every peer's release-add into
    # ``expert_recv_count_sum``, so we don't need mega_moe's spin-on-
    # high32 loop (which guards against starting dispatch_pull before
    # all peers have published).
    NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (num_experts_per_rank +
                                                    31) // 32
    stored_num_tokens_per_expert = []
    for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
        stored_num_tokens_per_expert.append(Int32(0))
    for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
        e_idx_for_lane = Int32(i * 32) + lane_idx
        if e_idx_for_lane < Int32(num_experts_per_rank):
            sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
            stored_num_tokens_per_expert[i] = Int32(
                Int64(sum_packed_init) & Int64(0xFFFFFFFF))
    cute.arch.sync_warp()

    # ------------------------------------------------------------------
    # Outer token loop: mega_moe line 559.
    # ------------------------------------------------------------------
    # ``num_global_warps = num_sms * 4`` (kNumDispatchWarps). Each global
    # warp gets a strided slice of the local pool (token_idx 0,1,2,...).
    num_global_warps: cutlass.Constexpr[int] = num_sms * _NUM_DISPATCH_WARPS
    token_idx = sm_idx * Int32(_NUM_DISPATCH_WARPS) + warp_idx

    # IKET fine-grain emitter predicate: only the very first thread of the
    # whole grid (sm 0, warp 0, lane 0) writes events. Multiplied across
    # 152 SM x 4 warp x 32 lane the per-iter markers would explode the
    # trace buffer; gating like this keeps total events per launch =
    # num_outer_iter * markers_per_iter (DSV4: ~3 * 6 = ~18; large config:
    # ~1.7k * 6 ~= 10k events / launch -- still cheap).
    # NOTE: previously emitted from all 4 dispatch warps (the version6
    # "allwarps" variant) to see per-warp SF/Weight cost differences. That
    # quadrupled probe count and triggered an IKET tracker overflow under
    # NVFP4 (sf_passes=4 vs FP8's 2 doubles SF-loop unroll, and the larger
    # cubin caused 3 markers -- Dispatch_Prep / Pull.ChooseToken /
    # Pull.TMA_NVLink_Roundtrip -- to drop). Gate down to warp 0 only.
    _iket_pull_emit = (sm_idx == Int32(0)) and (warp_idx
                                                == Int32(0)) and (lane_idx
                                                                  == Int32(0))

    # mega_moe uses an unbounded ``for(;;)`` with an inner ``break``. cuTeDSL
    # disallows ``break`` in dynamic control flow, so we use a bounded
    # *dynamic* ``cutlass.range`` loop with a ``done`` state flag. Switching
    # from ``range_constexpr`` -> ``range`` keeps the body emitted ONCE in
    # PTX (no 81x unroll), eliminating the icache pressure described in
    # perf_diff.md row 1.9.1. The upper bound ``num_padded_sf_pool_tokens //
    # num_global_warps + 1`` is a safe limit on the warp's strided slice.
    # Python ``while`` -- cuTeDSL frontend lowers to scf.while via
    # while_generate. Replaces ``for _iter in cutlass.range(upper_bound)``
    # which always ran the bounded iter count (~1766 at T=32k). The
    # while exits as soon as ExpertWalk advances current_expert_idx past
    # the last expert -- semantically identical to mega_moe's
    # ``while(true) { ... if (...) break; }`` outer pull loop.
    while current_expert_idx < Int32(num_experts_per_rank):
        # Advance ``current_expert_idx`` until the warp's token_idx falls
        # within ``[expert_start_idx, expert_end_idx)``. mega_moe lines
        # 561-572 (note: mega_moe peels ``expert_pool_block_offset += ceil_
        # div(end - start, BLOCK_M)`` *before* advancing the indices, but
        # it uses the *previous* expert's ``end - start``, which is the
        # count for the expert that just finished -- equivalent to "after
        # closing expert E, bump pool block offset by E's block count").
        if _iket_pull_emit:
            _iket.range_push("Pull.ChooseToken")
        old_expert_idx = current_expert_idx
        # Software fast-path: most outer pull iterations don't need to
        # advance (we still process tokens for the same current_expert_idx).
        # Wrap the entire constexpr-unrolled walk in a single dynamic
        # ``if token_idx >= expert_end_idx`` so the common case (no advance
        # needed) pays just one setp+bra instead of 65/97 setp+bra from
        # the unroll. Only iters that actually need to advance pay the
        # full unroll. Without this, the constexpr unroll executes every
        # outer iter regardless of whether real work is needed -- IKET
        # measured 1.3 us / call avg with 97-unroll at T=32k single-rank,
        # contributing 38 % of the kernel time.
        # Direct mega_moe equivalent: ``while (token_idx >= expert_end_idx)
        # { advance; if (cur_e >= N) break; }``. cuTeDSL has no dynamic
        # ``break``, but a Python ``while`` with the combined condition
        # ``token_idx >= expert_end_idx and current_expert_idx < N`` is
        # semantically identical -- the inner loop exits on either
        # token landing in range OR walking past the last expert.
        # No constexpr unroll → SASS instruction count is bounded by
        # actual #advances per token, matching mega_moe's `while+break`.
        while (token_idx >= expert_end_idx) and (current_expert_idx
                                                 < Int32(num_experts_per_rank)):
            prev_valid_count = expert_end_idx - expert_start_idx
            prev_block_count = (prev_valid_count + Int32(block_m) -
                                Int32(1)) // Int32(block_m)
            expert_pool_block_offset = expert_pool_block_offset + prev_block_count
            # Mirror cumul for the release-counter granularity (cluster_tile_m).
            prev_task_tile_count = (prev_valid_count + Int32(cluster_tile_m) -
                                    Int32(1)) // Int32(cluster_tile_m)
            expert_task_tile_offset = expert_task_tile_offset + prev_task_tile_count
            # Mirror cumul for the SF axis granularity (sf_block_m).
            # ``sf_block_m`` is decoupled from ``block_m``; see the comment
            # on ``expert_sf_pool_block_offset`` initialisation above.
            prev_sf_block_count = (prev_valid_count + Int32(sf_block_m) -
                                   Int32(1)) // Int32(sf_block_m)
            expert_sf_pool_block_offset = expert_sf_pool_block_offset + prev_sf_block_count
            current_expert_idx = current_expert_idx + Int32(1)
            if current_expert_idx < Int32(num_experts_per_rank):
                expert_start_idx = expert_end_idx
                valid_value = Int32(0)
                for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                    if current_expert_idx == Int32(i * 32) + lane_idx:
                        valid_value = stored_num_tokens_per_expert[i]
                total_for_expert = cute.arch.shuffle_sync(
                    valid_value, current_expert_idx % Int32(32))
                expert_end_idx = expert_end_idx + total_for_expert

        # If ExpertWalk just advanced past the last expert, skip the
        # rest of this iteration (token-pull body); the outer ``while``
        # will exit on its next condition check.
        if current_expert_idx < Int32(num_experts_per_rank):
            # Per-rank count reload on expert change (mega_moe lines
            # 579-588). One lane per source rank: lane ``r`` reads
            # ``expert_recv_count[r, current_expert_idx]`` for
            # ``r < world_size``; lanes ``>= world_size`` keep 0. The
            # 32-way warp load coalesces into a single 128-byte
            # cache-line read since the elements are contiguous along
            # the rank axis (the leading dim) for a fixed expert.
            if old_expert_idx != current_expert_idx:
                if lane_idx < Int32(world_size):
                    stored_rank_count_lane = Int32(
                        expert_recv_count[lane_idx, current_expert_idx])
                else:
                    stored_rank_count_lane = Int32(0)

            # ----------------------------------------------------------
            # Round-robin rank selection (mega_moe lines 590-638;
            # warp-cooperative path -- see strategy note in the
            # docstring).
            # ----------------------------------------------------------
            token_idx_in_expert = token_idx - expert_start_idx
            slot_idx = token_idx_in_expert
            offset = Int32(0)
            # Working copy of the per-lane remaining count. Mutated by
            # the per-round peel below; ``stored_rank_count_lane`` stays
            # untouched so it can serve subsequent tokens of the same
            # expert without reloading.
            remaining_lane = stored_rank_count_lane

            current_rank_in_expert_idx = Int32(0)
            token_idx_in_rank = Int32(0)

            # mega_moe inner ``while(true)`` -- bounded by ``world_size``
            # rounds because each round either decides the answer or peels
            # at least one rank to 0. ``decided`` flag replaces the
            # original ``break``.
            decided = Int32(0)
            for _round in cutlass.range_constexpr(0, world_size + 1, 1):
                if decided == Int32(0):
                    # Warp-cooperative ballot: bit ``r`` set iff lane
                    # ``r``'s remaining > 0. Lanes ``>= world_size``
                    # hold 0 so their bits are always clear.
                    active = remaining_lane > Int32(0)
                    mask = cute.arch.vote_ballot_sync(active)
                    num_active_ranks = Int32(cute.arch.popc(Int32(mask)))
                    # Min reduction over actives. Inactive lanes feed
                    # INT_MAX so they don't poison the ``min``.
                    v_for_min = Int32(0x7FFFFFFF)
                    if active:
                        v_for_min = remaining_lane
                    length = Int32(cute.arch.warp_redux_sync(v_for_min, "min"))

                    if num_active_ranks > Int32(0):
                        num_round_tokens = length * num_active_ranks
                        if slot_idx < num_round_tokens:
                            # Hit in this round (mega_moe lines 614-630).
                            slot_idx_in_round = slot_idx % num_active_ranks
                            # Single-PTX-instr Nth-set-bit (mega_moe
                            # ``__fns(mask, 0, n)``). ``n`` is 1-indexed.
                            current_rank_in_expert_idx = fns_b32(
                                Int32(mask),
                                Int32(0),
                                slot_idx_in_round + Int32(1),
                            )
                            token_idx_in_rank = offset + (slot_idx //
                                                          num_active_ranks)
                            decided = Int32(1)
                        else:
                            # Peel one round and continue. mega_moe
                            # lines 632-637. Each lane peels its own
                            # ``remaining_lane``.
                            slot_idx = slot_idx - num_round_tokens
                            offset = offset + length
                            if remaining_lane > length:
                                remaining_lane = remaining_lane - length
                            else:
                                remaining_lane = Int32(0)
                    else:
                        # Defensive: all ranks empty. The outer ``done``
                        # flag guarantees we never enter this branch with
                        # a valid ``slot_idx``; mark ``decided`` so we
                        # skip the remaining round iterations.
                        decided = Int32(1)

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.ChooseToken
                # Production layout (v9): TMA load is issued first, then
                # SF/Weight LDG-STG runs in parallel with the in-flight
                # TMA, and mbarrier_arrive+wait closes the range. Range
                # wraps: TMA issue + SF LDG + Weight LDG + SF STG + Weight
                # STG + mbar wait. SF_LDG_STG / Weight_LDG are nested
                # sub-ranges (= LD phase / ST phase from V7 refactor).
                _iket.range_push("Pull.TMA_NVLink_Roundtrip")

            # ----------------------------------------------------------
            # Read src_token_topk advertise card (mega_moe lines 641-
            # 644). The local ``src_token_topk_idx`` view was populated
            # by the round-3 advertise STG of every peer rank into this
            # rank during prep.
            # ----------------------------------------------------------
            src_token_topk = Uint32(src_token_topk_idx[
                current_expert_idx,
                current_rank_in_expert_idx,
                token_idx_in_rank,
            ])
            src_token = Int32(src_token_topk // Uint32(num_topk))
            src_topk = Int32(src_token_topk % Uint32(num_topk))

            # ``peer_rank_ptr_mapper.map`` style: cache the per-token peer offset
            # once after round-robin decides ``current_rank_in_expert_idx``,
            # then reuse for TMA-load source, SF LDG, weight LDG. One
            # GMEM/L1 read per token (warp-broadcast since all lanes share
            # the same ``current_rank_in_expert_idx``).
            # peer_rank_ptr_mapper.map(0, idx, 0) returns offsets[idx]; lowers to a
            # single indexed ld.param.b64 (= LDC.U64) from the param bank.
            cur_peer_offset = peer_rank_ptr_mapper.map(
                Int64(0), current_rank_in_expert_idx, Int64(0))
            # Local-rank base addresses; cross-rank access adds the peer
            # delta via ``peer_rank_ptr_mapper.map`` (or by reusing the cached
            # ``cur_peer_offset`` above) at each TMA load / scalar LDG
            # site below.
            inp_tok_local_base = input_token_buffer.iterator.toint()
            inp_sf_local_base = input_sf_buffer.iterator.toint()
            inp_w_local_base = input_topk_weights_buffer.iterator.toint()

            # ----------------------------------------------------------
            # TMA 1D pull token body (mega_moe lines 647-654). Pulls
            # ``hidden_bytes`` (7168 B for FP8, 3584 B for NVFP4) from peer
            # rank's ``input_token_buffer[src_token]`` row into this warp's
            # pull-buffer slot. Source addr = local_base +
            # peer_offset[current_rank] + src_token * hidden_bytes.
            # ----------------------------------------------------------
            with cute.arch.elect_one():
                pull_buffer_warp_ptr = pull_buffer_ptr + (warp_idx *
                                                          Int32(hidden_bytes))
                tma_src_addr = (inp_tok_local_base + cur_peer_offset +
                                Int64(src_token * Int32(hidden_bytes)))
                tma_load_1d_raw(
                    pull_buffer_warp_ptr,
                    tma_src_addr,
                    pull_mbar_ptr + warp_idx,
                    Int32(hidden_bytes),
                )
            cute.arch.sync_warp()

            if _iket_pull_emit:
                _iket.range_push("Pull.SF_LDG_STG")

            # ----------------------------------------------------------
            # Combined SF + Weight LD-then-ST phase (v7 layout).
            # Mega_moe lines 656-679. Hoist all peer NVLink LDGs to the
            # front (SF x sf_passes + Weight) so PTX scoreboard can keep
            # them all in flight in parallel, then issue all local STGs.
            # Baseline V5 stalled on each LDG before the next due to
            # the dependent STG; this layout removes that dependency
            # chain. IKET ranges keep their old names but the contents
            # shift: ``Pull.SF_LDG_STG`` now covers the LD phase (SF + W
            # LDGs); ``Pull.Weight_LDG`` covers the ST phase.
            # ----------------------------------------------------------
            # ``sf_token_in_pool_axis`` is the absolute M-axis position of this
            # token inside the pool's SF tensor.  ``expert_sf_pool_block_offset
            # * sf_block_m`` lands on the expert's atom-aligned start (since
            # ``sf_block_m`` is itself a multiple of ``SF_ATOM_BLOCK_TOKENS``);
            # adding ``token_idx_in_expert`` gives the per-token M position
            # which the cute SFA atom layout consumes via
            # ``sf_atom_int32_offset`` below.  MUST use the SF-granularity
            # cumul (``expert_sf_pool_block_offset``), NOT the token-granularity
            # one (``expert_pool_block_offset``) -- ``block_m`` and ``sf_block_m``
            # are decoupled in MegaMoE so the two cumuls advance independently.
            sf_token_in_pool_axis = (
                expert_sf_pool_block_offset * Int32(sf_block_m) +
                token_idx_in_expert)
            pool_token_idx = expert_pool_block_offset * Int32(
                block_m) + token_idx_in_expert
            # ceil(56 / 32) == 2 in DSV4; constexpr so loop fully unrolls.
            sf_passes: cutlass.Constexpr[int] = (sf_uint32_per_token + 31) // 32

            # Register-resident SF holds (per-lane Int32 per pass).
            sf_vals = []
            for _ in cutlass.range_constexpr(0, sf_passes, 1):
                sf_vals.append(Int32(0))

            # === LD phase: issue all NVLink LDGs back-to-back ===
            for i in cutlass.range_constexpr(0, sf_passes, 1):
                j = Int32(i * 32) + lane_idx
                if j < Int32(sf_uint32_per_token):
                    sf_addr = (inp_sf_local_base + cur_peer_offset + Int64(
                        (src_token * Int32(sf_uint32_per_token) + j) *
                        Int32(4)))
                    sf_vals[i] = ldg_b32_raw(sf_addr)

            # Weight LDG (single lane via dynamic if; SSA carry pattern
            # matches stored_rank_count_lane elsewhere).
            weight = Float32(0.0)
            if lane_idx == Int32(0):
                weight_addr = (inp_w_local_base + cur_peer_offset + Int64(
                    (src_token * Int32(num_topk) + src_topk) * Int32(4)))
                weight = ldg_f32_raw(weight_addr)

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.SF_LDG_STG  (= LD phase)
                _iket.range_push("Pull.Weight_LDG")  # (= ST phase)

            # === ST phase: all local stores ===
            # ``l1_sf_buffer`` is a 1D Int32 atom-flat buffer; each Int32
            # holds 4 K-bank fp8 SFs (= one K-atom column of width 1) at the
            # cute SFA atom layout's atom-inner byte position.  See
            # ``src/sf_swizzle.py:sf_atom_int32_offset`` for the layout.
            for i in cutlass.range_constexpr(0, sf_passes, 1):
                j = Int32(i * 32) + lane_idx
                if j < Int32(sf_uint32_per_token):
                    sf_int32_pos = sf_atom_int32_offset(
                        sf_token_in_pool_axis,
                        j,
                        num_k_atoms=sf_uint32_per_token,
                    )
                    l1_sf_buffer[sf_int32_pos] = sf_vals[i]
            cute.arch.sync_warp()

            if lane_idx == Int32(0):
                l1_topk_weights_buffer[pool_token_idx] = weight

            # mbarrier arrive+wait restored to its V4 production position:
            # AFTER SF/Weight LD-then-ST so that SF+Weight LDG-STG fully
            # overlaps the in-flight TMA load. Mega_moe lines 683-684.
            # Profiling-only V5 layout (mbar before SF/Weight) sacrificed
            # this overlap and dropped perf by ~36% per IKET trace; this
            # restores production layout.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    pull_mbar_ptr + warp_idx, Int32(hidden_bytes))
                cute.arch.mbarrier_wait(
                    pull_mbar_ptr + warp_idx,
                    phase_bit,
                )

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.Weight_LDG (ST phase)
                _iket.range_pop()  # Pull.TMA_NVLink_Roundtrip (outer)
                _iket.range_push("Pull.TMA_Store")

            with cute.arch.elect_one():
                # 3) TMA store the hidden_bytes-long token body into the L1
                #    pool. mega_moe lines 687-689:
                #    ``cp.async.bulk.global.shared::cta.bulk_group``.
                pull_buffer_warp_ptr = pull_buffer_ptr + (warp_idx *
                                                          Int32(hidden_bytes))
                tma_store_1d(
                    l1_token_buffer.iterator
                    # Int64 multiply: pool_token_idx (up to ~3.16 M at
                    # T=128k) × hidden_bytes overflows int32 (max 2.1 G).
                    # 64-bit math is required for the tma store destination
                    # address at large T.
                    + (Int64(pool_token_idx) * Int64(hidden_bytes)),
                    pull_buffer_warp_ptr,
                    Int32(hidden_bytes),
                )

            # TMA_Store range covers issue + metadata STG + commit/wait;
            # individual sub-ranges removed per profile-simplification.

            with cute.arch.elect_one():
                # 4) Metadata pack: 12-byte
                #    ``{src_rank, src_token, src_topk}`` little-endian
                #    record. Three ``st.global.b32`` at row offsets
                #    ``0 / 4 / 8`` (12 B records aren't 8 B-aligned for
                #    odd indices; v2.b32 + b32 isn't safe).
                _store_token_src_metadata_u32x3(
                    token_src_metadata,
                    pool_token_idx,
                    Uint32(current_rank_in_expert_idx),
                    Uint32(src_token),
                    Uint32(src_topk),
                )

            # (Metadata_STG inside the merged Pull.TMA_Store range.)

            with cute.arch.elect_one():
                # 5) Commit the bulk store group and wait so the GMEM
                #    write is in flight before the arrival-count
                #    release. mega_moe lines 696-697:
                #    ``cute::tma_store_arrive`` +
                #    ``ptx::tma_store_wait<0>``. The cuTeDSL
                #    equivalents are ``cp_async_bulk_commit_group``
                #    and ``cp_async_bulk_wait_group(0)``.
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.TMA_Store
                _iket.range_push("Pull.Arrival_Atomic")

            with cute.arch.elect_one():
                # 6) Release-semantic atomic add into
                #    ``l1_arrival_count``. The GEMM consumer (post-DSV4)
                #    spins on this counter with an acquire load; the
                #    release here pairs with that acquire so the SF /
                #    weight / token / metadata writes above are
                #    observable once the counter ticks.
                #
                # Counter granularity is ``cluster_tile_m``, not ``block_m``
                # -- ``block_m`` only controls how tokens are LAID OUT in
                # ``l1_token_buffer`` (and pool padding granularity); the
                # downstream consumer's spin slot index is at task-tile
                # granularity so that one task tile maps to exactly one
                # counter (no K-aggregate spin).  When
                # ``cluster_tile_m == block_m`` this degenerates to the
                # legacy per-pool-block counter.  See
                # ``fc12_integrate_comm.md`` §4 (C3) for the contract.
                task_tile_idx = expert_task_tile_offset + (
                    token_idx_in_expert // Int32(cluster_tile_m))
                _atomic_add_release_gpu_u32(
                    l1_arrival_count.iterator + task_tile_idx, Int32(1))
            cute.arch.sync_warp()

            if _iket_pull_emit:
                _iket.range_pop()  # Pull.Arrival_Atomic

            # Flip the mbarrier phase bit for the next iteration. Hoisted
            # out of the elect_one block so all lanes see the uniform
            # update; only lane 0 actually reads the bit on the next
            # mbarrier_wait. mega_moe does this inline in the leader's
            # register at line 684 (``pull_mbarrier_phase ^= 1``).
            phase_bit = phase_bit ^ Int32(1)

            # Advance to this warp's next token slot.
            token_idx = token_idx + Int32(num_global_warps)


def _store_token_src_metadata_u32x3(
    token_src_metadata,
    pool_token_idx,
    src_rank: Uint32,
    src_token: Uint32,
    src_topk: Uint32,
) -> None:
    """Store the 12-byte ``TokenSrcMetadata`` record as three u32 GMEM stores.

    The kernel-facing tensor is shaped ``(num_pool_tokens, 12)`` uint8 but
    the payload is three u32 fields ``{rank, token, topk}`` each 4 B LE.
    Mirrors mega_moe ``sm100_fp8_fp4_mega_moe.cuh:692-693``. (Note: PTX has
    no ``v3.b32`` form, and the 12 B records are not 8 B-aligned for odd
    pool indices, so a mixed ``v2.b32 + b32`` pair is unsafe; three
    32-bit stores remain the safest emission.)
    """
    base_ptr = token_src_metadata.iterator + (pool_token_idx * Int32(12))
    cute.arch.store(base_ptr, src_rank, scope="gpu")
    cute.arch.store(base_ptr + Int32(4), src_token, scope="gpu")
    cute.arch.store(base_ptr + Int32(8), src_topk, scope="gpu")


@cute.jit
def _nvlink_barrier_3stage(
    nvlink_barrier_signal,
    grid_sync_counter,
    peer_rank_ptr_mapper,
    sm_idx,
    warp_idx,
    lane_idx,
    *,
    slot: cutlass.Constexpr[int],
    # ``num_sms`` is forwarded as-is to ``software_grid_sync``, which
    # accepts either a codegen-time Python int or a runtime ``Int32`` SSA
    # (the ``isinstance(num_sms, int)`` branch in ``grid_sync.py`` picks
    # the right delta arithmetic).  No PTX-level reason to require a
    # codegen const here; the standalone dispatch_kernel happens to feed
    # a Python int because its top-level kwarg is already Constexpr, but
    # a fused consumer that gets the CTA count from launch-time inputs
    # can pass an ``Int32`` SSA without any further indirection.
    num_sms,
    world_size: cutlass.Constexpr[int],
    local_rank: cutlass.Constexpr[int],
    prologue_grid_sync: cutlass.Constexpr[bool],
    epilogue_grid_sync: cutlass.Constexpr[bool],
):
    """Three-stage cross-rank NVLink barrier.

    Mirrors ``comm::nvlink_barrier`` from
    ``DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh`` (lines 28-72).
    The mega_moe implementation toggles between two ``signal_phase`` slots and
    uses a sign bit to alternate ``+1`` / ``-1`` so the slots can be reused
    indefinitely. DSV4 simplifies that scheme: the per-rank signal tensor has
    one slot per barrier call (DSV4 uses ``slot=0`` for the pre-pull barrier
    and ``slot=1`` for the kernel-tail barrier). Each slot is hit once per
    kernel, so no phase toggle is needed and the signal payload is always
    ``+1``.

    Three logical stages:
      A. Prologue grid sync (optional). Ensures every SM in this rank has
         finished any pre-barrier writes before SM 0 fires the cross-rank
         signal.
      B. Cross-rank fan-out: SM 0 warp 0 does, on lane ``r`` for ``r in
         [0, world_size)``,
         ``atom.release.sys.global.add.s32(peer_signal[r][slot], 1)``,
         then spin-waits on the local signal until it equals ``world_size``.
         This is functionally equivalent to mega_moe's lines 49-66 with the
         ``signal_sign=0`` branch hard-coded.
      C. Epilogue grid sync (optional). Fans completion out across all CTAs
         so that, after this call returns, every CTA on this rank can
         observe the barrier as released.

    ``nvlink_barrier_signal`` is the local-rank view of the symmetric signal
    buffer; cross-rank fan-out goes through ``peer_rank_ptr_mapper.map(local_base,
    peer_rank, slot * 4)``, the spin-wait reads the local view directly.
    """
    # Logical thread index within the dispatch warp group ([0, 128)); leader
    # is tid_in_group == 0.  See ``_dispatch_barrier`` / ``software_grid_sync``
    # docstrings for why this MUST be group-relative, not the raw ``%tid.x``.
    tid_in_group = warp_idx * Int32(32) + lane_idx

    if prologue_grid_sync:
        software_grid_sync(grid_sync_counter,
                           sm_idx,
                           num_sms,
                           tid_in_group,
                           num_threads=_NUM_DISPATCH_THREADS)

    if sm_idx == 0:
        # Stage B: only SM 0, only warp 0 fans out and waits.
        if warp_idx == 0:
            # Each of the first ``world_size`` lanes writes to a distinct
            # peer rank (matches mega_moe line 49-50:
            # ``if (thread_idx < kNumRanks)
            #     ptx::red_add_rel_sys(peer_rank_ptr_mapper.map(signal_ptr, thread_idx), 1)``).
            # Each lane targets one peer rank: lane r writes to
            # ``peer_rank_ptr_mapper.map(local_signal, r, slot * 4)``. The warp
            # issues ``world_size`` distinct cross-rank atomic adds in
            # parallel, slot 0 = pre-pull barrier, slot 1 = kernel-tail.
            # ``peer_rank_ptr_mapper.map`` style: local base + ``offsets[lane_idx]``
            # + slot. Each lane issues a distinct cross-rank atomic add
            # (one per peer rank). Element type is i32, so slot offset
            # is *4.
            nbs_local_base = nvlink_barrier_signal.iterator.toint()
            if lane_idx < Int32(world_size):
                lane_peer_addr = peer_rank_ptr_mapper.map(
                    nbs_local_base,
                    lane_idx,
                    Int64(Int32(slot * 4)),
                )
                red_add_release_sys_s32_raw(lane_peer_addr, Int32(1))
            cute.arch.sync_warp()

            # Stage B continued: lane 0 spin-waits on the local signal until
            # every peer rank (including this rank) has fanned a +1 in.
            if lane_idx == 0:
                target = Int32(world_size)
                # Local view of this rank's signal -- the spin target.
                local_signal_ptr = nvlink_barrier_signal.iterator + Int32(slot)
                # Spin until the signal hits ``world_size``. The acquire
                # load pairs with the peer release stores above so any
                # writes the peers issued *before* their barrier arrive are
                # observable here.
                while _load_acquire_sys_s32(local_signal_ptr) < target:
                    pass
        # Other warps in SM 0 idle until the epilogue grid sync below.

    if epilogue_grid_sync:
        software_grid_sync(grid_sync_counter,
                           sm_idx,
                           num_sms,
                           tid_in_group,
                           num_threads=_NUM_DISPATCH_THREADS)


# -----------------------------------------------------------------------------
# Top-level kernel
# -----------------------------------------------------------------------------


@cute.kernel
def dispatch_kernel(
    # Per-rank input region.
    input_token_buffer: cute.Tensor,
    input_sf_buffer: cute.Tensor,
    input_topk_idx_buffer: cute.Tensor,
    input_topk_weights_buffer: cute.Tensor,
    # Counter region.
    expert_send_count: cute.Tensor,
    expert_recv_count: cute.Tensor,
    expert_recv_count_sum: cute.Tensor,
    # Routing workspace.
    src_token_topk_idx: cute.Tensor,
    token_src_metadata: cute.Tensor,
    # L1 pool region (local rank's receive-side buffers).
    l1_arrival_count: cute.Tensor,
    l1_token_buffer: cute.Tensor,
    l1_sf_buffer: cute.Tensor,
    l1_topk_weights_buffer: cute.Tensor,
    # Cross-rank barrier signal (single local view; peer access via
    # peer_rank_ptr_mapper.map).
    nvlink_barrier_signal: cute.Tensor,
    grid_sync_counter: cute.Tensor,
    # Symmetric-heap peer-pointer table.  A concrete ``SymBuffer{N}``
    # value is passed by CUDA byval ABI (param bank → ``LDC.U64`` on
    # each peer lookup).
    # All cross-rank pointer arithmetic in this kernel goes through
    # ``peer_rank_ptr_mapper.map(local_ptr, peer_rank, byte_off)``.  The same delta
    # works for every symmetric sub-region because all ranks share an
    # identical heap layout (single-allocation byte buffer, see
    # ``bootstrap.py``).  Type is intentionally untyped here: ``SymBuffer{N}``
    # is parametrised on ``num_max_ranks`` so this kernel is duck-typed.
    peer_rank_ptr_mapper,
    # Compile-time configuration.
    local_rank: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
    num_tokens: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    num_total_experts: cutlass.Constexpr[int],
    block_m: cutlass.Constexpr[int],
    # Release-add counter granularity (task-tile size of the downstream
    # consumer).  ``l1_arrival_count`` is sized ``num_max_task_tiles``,
    # not ``num_max_pool_blocks``.  When run as a standalone dispatch
    # without a fused downstream, set ``cluster_tile_m == block_m`` to
    # recover the legacy per-pool-block counter layout.  See
    # ``fc12_integrate_comm.md`` §4 (C3).
    cluster_tile_m: cutlass.Constexpr[int],
    sf_block_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    hidden_bytes: cutlass.Constexpr[int],
    sf_uint32_per_token: cutlass.Constexpr[int],
    num_padded_sf_pool_tokens: cutlass.Constexpr[int],
):
    """Single-kernel cuTeDSL replica of mega_moe's dispatch slice.

    Replicates ``sm100_fp8_fp4_mega_moe.cuh`` lines 432-766:

        ``dispatch_prep`` -> ``dispatch_barrier`` -> ``dispatch_pull``
        -> end-of-kernel NVLink barrier (DSV4 leaves ``Cleanup`` to a separate
        kernel).

    Only the first 4 warps (the dispatch warps) participate in DSV4; mega_moe
    keeps a 5th warp for the ``Epilogue`` slot (line 770-) which has no role
    in dispatch-only DSV4.
    """
    # SMEM allocation -- size depends on `hidden_bytes` (28 KB pull buffer for
    # FP8, 14 KB for NVFP4). SharedStorage is JIT-built per kernel instance.
    smem = cutlass.utils.SmemAllocator()
    SharedStorage = _make_shared_storage(hidden_bytes)
    storage = smem.allocate(SharedStorage)

    sm_idx = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    warp_idx = tid // 32
    lane_idx = tid % 32

    if warp_idx < _NUM_DISPATCH_WARPS:
        # mega_moe line 419-422 deallocates registers down to 48 for the
        # dispatch warps so Epilogue/Combine warps can use them. DSV4 has no
        # epilogue but we keep the dealloc to mirror mega_moe register
        # pressure exactly.
        cute.arch.warpgroup_reg_dealloc(48)

        # IKET coarse-grain regions. We only emit on the leader CTA (sm 0)
        # and warp 0 to keep the trace-event volume bounded -- mirrors
        # mega_moe's ``IKET_RANGE_START`` gating at line 422-425
        # ``(sm_idx == 0) && is_leader_cta && (warp_idx == 0)``. Without
        # this guard 152 SMs * 4 warps * 4 regions = 2432 events per
        # kernel-launch flood the trace buffer.
        # Reverted from version6-allwarps back to warp 0 only -- the
        # 4-warp variant works for FP8 but causes IKET tracker overflow
        # under NVFP4 (drops 3 markers, see _iket_pull_emit comment).
        _iket_active = (sm_idx == Int32(0)) and (warp_idx == Int32(0))
        if _iket_active:
            _iket.range_push("Dispatch_Prep")

        _dispatch_prep(
            storage,
            input_topk_idx_buffer,
            expert_send_count,
            src_token_topk_idx,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_tokens=num_tokens,
            num_topk=num_topk,
            num_sms=num_sms,
            num_experts_per_rank=num_experts_per_rank,
            num_total_experts=num_total_experts,
            local_rank=local_rank,
            world_size=world_size,
        )

        if _iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Barrier")

        # Dispatch_Barrier: publishes per-rank ``expert_send_count`` to all
        # peers as ``expert_recv_count`` / ``expert_recv_count_sum``, runs
        # the intra-rank grid sync, and finishes with a 3-stage NVLink
        # barrier so the pull stage on every rank only starts after all
        # peers' counts are visible. Mega_moe lines 488-532.
        _dispatch_barrier(
            expert_send_count,
            expert_recv_count,
            expert_recv_count_sum,
            nvlink_barrier_signal,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_sms=num_sms,
            num_experts_per_rank=num_experts_per_rank,
            num_total_experts=num_total_experts,
            local_rank=local_rank,
            world_size=world_size,
        )

        if _iket_active:
            _iket.range_pop()
            _iket.range_push("Dispatch_Pull")

        # Dispatch_Pull: each dispatch warp peels its strided slice of the
        # local pool, replays the round-robin oracle to identify (src_rank,
        # token_idx_in_rank) for each slot, and TMA-pulls the token body /
        # SF / weight from the appropriate peer rank. The release-add into
        # ``l1_arrival_count`` at the end of each iteration pairs with the
        # downstream GEMM consumer's acquire load. Mega_moe lines 542-711.
        _dispatch_pull(
            storage,
            input_token_buffer,
            input_sf_buffer,
            input_topk_weights_buffer,
            src_token_topk_idx,
            expert_recv_count,
            expert_recv_count_sum,
            l1_token_buffer,
            l1_sf_buffer,
            l1_topk_weights_buffer,
            l1_arrival_count,
            token_src_metadata,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            num_sms=num_sms,
            num_experts_per_rank=num_experts_per_rank,
            num_topk=num_topk,
            block_m=block_m,
            sf_block_m=sf_block_m,
            cluster_tile_m=cluster_tile_m,
            hidden=hidden,
            hidden_bytes=hidden_bytes,
            sf_uint32_per_token=sf_uint32_per_token,
            num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
            world_size=world_size,
            local_rank=local_rank,
        )

        if _iket_active:
            _iket.range_pop()
            _iket.range_push("Kernel_Tail")

        # Kernel-tail NVLink barrier (DSV4 dispatch-only termination). Mega_moe
        # closes the dispatch slice with a sync_unaligned + Cleanup pairing
        # against the Combine warps; DSV4 has no Combine, so we stand the
        # cross-rank barrier up directly on slot 1 to guarantee every peer
        # has finished its pull stage before the kernel exits. The
        # downstream GEMM stage waits on ``l1_arrival_count``, which already
        # carries per-pool-block release semantics; this final barrier just
        # ensures kernel-exit is observable to the host's stream barrier.
        # Cross-warp-group named barrier with the (placeholder) epilogue
        # warps. Mirrors mega_moe ``sm100_fp8_fp4_mega_moe.cuh:727``
        # ``ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads,
        # kDispatchWithEpilogueBarrierIdx)``. In mega_moe this gates the
        # epilogue's own barrier slot from running concurrently with the
        # pull barrier; in DSV4 the epilogue side is a no-op placeholder,
        # but we keep the barrier so the thread layout is shape-compatible
        # with the fused MoE kernel (4 dispatch + 8 epilogue warps).
        cute.arch.barrier(
            barrier_id=_KDISPATCH_WITH_EPILOGUE_BARRIER_IDX,
            number_of_threads=_NUM_TOTAL_THREADS,
        )

        _nvlink_barrier_3stage(
            nvlink_barrier_signal,
            grid_sync_counter,
            peer_rank_ptr_mapper,
            sm_idx,
            warp_idx,
            lane_idx,
            slot=1,
            num_sms=num_sms,
            world_size=world_size,
            local_rank=local_rank,
            prologue_grid_sync=True,
            epilogue_grid_sync=True,
        )

        if _iket_active:
            _iket.range_pop()
    else:
        # Placeholder "epilogue" warp group (8 warps, warp_idx in [4, 12)).
        # mega_moe's epilogue/combine warps live here; DSV4 dispatch-only has
        # no real work for them, but they participate in the cross-warp-group
        # named barrier so ``bar.sync 1, 384`` is well-defined and the thread
        # layout matches mega_moe. Future fusion replaces this branch with
        # the real epilogue logic without touching the dispatch path.
        cute.arch.barrier(
            barrier_id=_KDISPATCH_WITH_EPILOGUE_BARRIER_IDX,
            number_of_threads=_NUM_TOTAL_THREADS,
        )
