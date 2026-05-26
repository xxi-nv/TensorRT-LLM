# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sm100MegaMoEKernel -- MegaMoE-complete fused dispatch + fc1 + fc2 (+ combine).

This subclass plugs the dispatch three-stage flow (dispatch_prep /
dispatch_barrier / dispatch_pull) into the lean fused fc1+fc2 base
kernel (``Sm100SwapABSwigluFp4Fc12Kernel``).  The base owns all GEMM /
sched / epi specialization; this file only owns:

  1. ``TokenCommArgs`` -- the MegaMoE-only argument bundle carried
     into the device kernel as an opaque MLIR-serialized struct.
     Field naming follows the fc12-kernel viewpoint (``fc1_input_*``
     for buffers the fc1 phase reads as activations / SF / topk
     weights) rather than the dispatch-side ``l1_*`` term.  Inside
     this subclass the workspace partition produces two cute views
     over identical bytes (one for each consumer) so the dispatch
     and GEMM sides each see the element-type they expect.
  2. ``Sm100MegaMoEKernel`` -- the subclass that flips
     ``self.enable_token_comm = True``, hosts the MegaMoE codegen
     constants (independent: ``world_size`` / ``local_rank`` /
     ``num_topk`` / ``max_tokens_per_rank`` / ``hidden``; the rest
     are derived from these and from ``static_expert_shape``).
     Exposes ``get_workspace_sizes()`` for the host to query the two
     opaque byte budgets, partitions the opaque workspaces inside
     ``__call__`` into typed cute views, and overrides the six base
     hook methods that wire dispatch / fc1 spin / NamedBarrier 9 /
     kernel-tail NVLink slot=1, plus the ``_smem_misc_budget_bytes``
     hook so the base's AB-stage SMEM budget accounts for the
     dispatch warps' SMEM region.

Host-facing contract:

  ``Sm100MegaMoEKernel.__init__(..., world_size, local_rank,
                                num_topk, max_tokens_per_rank,
                                hidden)``
      Five independent MegaMoE-specific constants.  ``hidden`` is
      separate from ``static_expert_shape[2]`` (and validated to
      match) because SMEM static sizing needs it before
      ``static_expert_shape`` is decoded.  ``static_expert_shape``
      itself is *required* (the dynamic-shape path is future work).
      All other shape constants (``num_experts_per_rank``,
      ``intermediate_gateup``, ``hidden_bytes``,
      ``sf_uint32_per_token``, ``num_total_experts``,
      ``pool_*_capacity``) are derived.

  ``kernel.get_workspace_sizes() -> (local_ws_bytes, shared_ws_bytes)``
      Host allocates two opaque byte buffers (``local`` in CUDA-local
      memory, ``shared`` in NVSHMEM symmetric heap).  Per-launch
      ``T = activation.shape[0]`` may be <= ``max_tokens_per_rank``.

  ``kernel.__call__(activation, ..., local_workspace, shared_workspace,
                    peer_rank_ptr_mapper_host, max_active_clusters, stream)``
      ``sm_count`` is derived inside as
      ``max_active_clusters * cluster_size`` -- not a kwarg.

      ``peer_rank_ptr_mapper_host`` carries runtime base/offset values;
      ``__call__`` packs them into ``SymBuffer{world_size}`` before device
      code uses ``peer_rank_ptr_mapper.map(...)``.

Shared / local region split (first-principles -- a region is shared
iff some peer rank can reach it via ``peer_rank_ptr_mapper.map(local_addr,
peer_rank, byte_off)`` inside ``src/dispatch_kernel.py``):

  SHARED  : src_token_topk_idx, expert_recv_count, expert_recv_count_sum,
            nvlink_barrier_signal
  LOCAL   : expert_send_count, grid_sync_counter, l1_token_buffer,
            l1_sf_buffer, l1_topk_weights_buffer, l1_arrival_count,
            token_src_metadata, fc1_output, fc1_output_sf,
            fc1_done_counter, (optionally) load_balance_counter

User-domain inputs (NOT in the opaque workspaces): activation,
activation_sf, topk_idx, topk_weights, the four weight tensors, and
combine_output.  ``activation / activation_sf / topk_weights`` must
satisfy the sym-map contract too (peers reach them during pull);
``topk_idx`` and the weights are local-only.  ``combine_output`` must
be sym-mapped (peer write target under S3).

S3 (combine STG redirect) is intentionally NOT implemented here.

Reference: ``moe_nvfp4_swapab/fc12_integrate_comm.md`` (sections
1 / 2 / 3 / 4 / 6 / 7 / 8) for design rationale.  The C1 / C2 / C3
alignment constraints (§4) are unified at construction time:
``token_padding_block`` (base) and ``block_m`` (dispatch) become the
same constant, similarly for ``sf_padding_block`` / ``sf_block_m``;
C3 reduces to a divisibility check that ``cluster_tile_tokens`` is a
multiple of ``token_padding_block``.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here.
# PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``'s
# element-type introspection (it reads ``__annotations__`` and demands the
# values be live ``cute.struct.MemRange[...] / struct / array / base_dsl
# scalar`` objects, not their string forms).  The lean fc1+fc2 base
# (``kernel_fc12.py``) and the dispatch standalone (``src/dispatch_kernel.py``)
# both already follow this convention.  Self-references (the single
# ``"TokenCommArgs"`` forward ref on ``__new_from_mlir_values__``) stay
# quoted explicitly.

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from .iket_compat import iket

import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (Int32, Int64, Uint8, extract_mlir_values,
                                 new_from_mlir_values)

from .dispatch_kernel import (_dispatch_barrier, _dispatch_prep, _dispatch_pull,
                              _nvlink_barrier_3stage)
from .kernel_fc12 import Sm100SwapABSwigluFp4Fc12Kernel
from .moe_utils import spin_wait

# =============================================================================
# Module-level constants (PascalCase per project convention; see
# moe_nvfp4_swapab/megamoe_constants.py for the same style: Nvfp4BlockSize,
# SupportedMmaTileM, ...).
# =============================================================================

# NamedBarrier IDs.  Base reserves 1-7; this subclass uses 8 and 9.
# Barrier 0 is implicitly reused by the standalone dispatch primitives'
# CTA-local ``_sync_aligned_first_4_warps`` / ``software_grid_sync``
# wrappers (128-thread participants); GEMM / epi warps are concurrently
# on different barrier ids at that point so the reuse is safe.
_KernelTailNamedBarrierId = 8  # 12-warp rendezvous (384 threads)
_DispatchToSchedNamedBarrierId = 9  # 4 dispatch + 1 sched (160 threads)

# Dispatch warp count + per-CTA expert histogram capacity.  Redeclared
# here rather than imported from src/dispatch_kernel.py: the standalone
# constants are implementation details of that kernel; the 4 / 512
# numbers are part of *this* subclass's contract too.
_DispatchWarpCount = 4
_TotalExpertsCapacity = 512

# Per-pool-slot provenance record consumed by combine STG redirect (S3).
# Three packed Uint32 fields = 12 bytes: ``{src_rank, src_token, src_topk}``.
_TokenMetadataBytes = 12

# NVLink slot count: slot 0 = pre-pull (dispatch_barrier internal),
# slot 1 = kernel-tail (combine release).
_NvlinkSlotCount = 2

# Grid-sync counter slot count.  ``software_grid_sync`` phase-flips bit 31
# so a single slot suffices; 2 slots keeps the layout 8-byte aligned.
_GridSyncSlotCount = 2

# =============================================================================
# TokenCommArgs -- MegaMoE argument bundle
# =============================================================================

# Field-name lists drive both serialization and ``__init__`` so the two
# sides cannot drift; order is the canonical MLIR-extract order.
# All listed fields support ``__extract_mlir_values__`` /
# ``__new_from_mlir_values__`` -- ``cute.Tensor`` natively,
# ``peer_rank_ptr_mapper`` via the ``@native_struct`` protocol -- so a
# single serialization loop handles both.
_MLIR_VALUE_FIELDS: Tuple[str, ...] = (
    "input_token_buffer",
    "input_sf_buffer",
    "input_topk_idx_buffer",
    "input_topk_weights_buffer",
    "expert_send_count",
    "expert_recv_count",
    "expert_recv_count_sum",
    "src_token_topk_idx",
    "fc1_input_token_buffer",
    "fc1_input_sf_buffer",
    "fc1_input_topk_weights_buffer",
    "fc1_ready_counter",
    "token_src_metadata",
    "combine_output",
    "nvlink_barrier_signal",
    "grid_sync_counter",
    "peer_rank_ptr_mapper",
)

# Codegen-time Python int fields -- never MLIR-serialized; round-trip
# via the prototype in ``__new_from_mlir_values__``.  ``sm_count`` is
# always a Python int now (derived from ``max_active_clusters *
# cluster_size``), so it lives here and flows cleanly to dispatch
# primitives' ``num_sms: Constexpr[int]`` slots.
_CONST_FIELDS: Tuple[str, ...] = (
    "world_size",
    "local_rank",
    "num_total_experts",
    "num_experts_per_rank",
    "num_topk",
    "hidden_bytes",
    "sf_uint32_per_token",
    "token_padding_block",
    "sf_padding_block",
    "sm_count",
)


class TokenCommArgs:
    """MegaMoE argument bundle.  Carried into ``fc1fc2_kernel_impl`` and
    forwarded as-is to the ``token_comm_hook_*`` methods.

    Serialization:
      * Every ``_MLIR_VALUE_FIELDS`` member extends MLIR values via
        ``extract_mlir_values`` in the listed order.  ``cute.Tensor``
        fields contribute their tensor primitives; ``peer_rank_ptr_mapper``
        contributes a single struct ``ir.Value`` (it is a
        ``@native_struct`` instance).
      * ``_CONST_FIELDS`` (Python int codegen constants) are passthrough
        -- copied from the prototype during ``__new_from_mlir_values__``.

    Combine-side fields (``combine_output``, ``token_src_metadata``)
    are carried in S2 even though no S2 hook consumes them; S3 will
    wire them through ``Fc2UnpackPermuteStg``.
    """

    def __init__(
        self,
        *,
        input_token_buffer: cute.Tensor,
        input_sf_buffer: cute.Tensor,
        input_topk_idx_buffer: cute.Tensor,
        input_topk_weights_buffer: cute.Tensor,
        expert_send_count: cute.Tensor,
        expert_recv_count: cute.Tensor,
        expert_recv_count_sum: cute.Tensor,
        src_token_topk_idx: cute.Tensor,
        fc1_input_token_buffer: cute.Tensor,
        fc1_input_sf_buffer: cute.Tensor,
        fc1_input_topk_weights_buffer: cute.Tensor,
        fc1_ready_counter: cute.Tensor,
        token_src_metadata: cute.Tensor,
        combine_output: cute.Tensor,
        nvlink_barrier_signal: cute.Tensor,
        grid_sync_counter: cute.Tensor,
        peer_rank_ptr_mapper: Any,
        world_size: int,
        local_rank: int,
        num_total_experts: int,
        num_experts_per_rank: int,
        num_topk: int,
        hidden_bytes: int,
        sf_uint32_per_token: int,
        token_padding_block: int,
        sf_padding_block: int,
        sm_count: int,
    ):
        self.input_token_buffer = input_token_buffer
        self.input_sf_buffer = input_sf_buffer
        self.input_topk_idx_buffer = input_topk_idx_buffer
        self.input_topk_weights_buffer = input_topk_weights_buffer
        self.expert_send_count = expert_send_count
        self.expert_recv_count = expert_recv_count
        self.expert_recv_count_sum = expert_recv_count_sum
        self.src_token_topk_idx = src_token_topk_idx
        self.fc1_input_token_buffer = fc1_input_token_buffer
        self.fc1_input_sf_buffer = fc1_input_sf_buffer
        self.fc1_input_topk_weights_buffer = fc1_input_topk_weights_buffer
        self.fc1_ready_counter = fc1_ready_counter
        self.token_src_metadata = token_src_metadata
        self.combine_output = combine_output
        self.nvlink_barrier_signal = nvlink_barrier_signal
        self.grid_sync_counter = grid_sync_counter
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_total_experts = num_total_experts
        self.num_experts_per_rank = num_experts_per_rank
        self.num_topk = num_topk
        self.hidden_bytes = hidden_bytes
        self.sf_uint32_per_token = sf_uint32_per_token
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.sm_count = sm_count

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for name in _MLIR_VALUE_FIELDS:
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "TokenCommArgs":
        idx = 0
        rebuilt: Dict[str, Any] = {}
        for name in _MLIR_VALUE_FIELDS:
            proto = getattr(self, name)
            n = len(extract_mlir_values(proto))
            rebuilt[name] = new_from_mlir_values(proto, values[idx:idx + n])
            idx += n
        assert idx == len(values), (
            f"TokenCommArgs serialization mismatch: consumed={idx} provided={len(values)}"
        )
        const_kwargs = {name: getattr(self, name) for name in _CONST_FIELDS}
        return TokenCommArgs(**rebuilt, **const_kwargs)


# =============================================================================
# Region spec + layout helpers
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _RegionSpec:
    """One region in either the local or shared workspace.

    Byte size = ``ceil(numel * cute_dtype.width / 8)``.  ``align`` is
    the region's start-byte alignment (TMA store / load destinations
    want 128 B; counters / metadata want 16 B).
    """

    name: str
    cute_dtype: Any
    shape: Tuple[int, ...]
    align: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def stride_row_major(self) -> Tuple[int, ...]:
        """Row-major stride matching ``shape`` (rightmost dim contiguous)."""
        if len(self.shape) == 0:
            return ()
        out: List[int] = [1]
        for d in reversed(self.shape[1:]):
            out.append(out[-1] * d)
        out.reverse()
        return tuple(out)

    @property
    def nbytes(self) -> int:
        bits = self.numel * int(self.cute_dtype.width)
        return (bits + 7) // 8


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _layout_regions(regions: List[_RegionSpec], ) -> Tuple[Dict[str, int], int]:
    """Place ``regions`` sequentially honouring each region's ``align``.
    Returns ``(name -> byte_offset)`` and the total byte count (rounded
    up to 16 B for downstream safety).

    Drives both ``get_workspace_sizes()`` (total only) and the
    ``__call__`` partition (offsets) -- keeping the host allocation
    and the device view construction in sync without any explicit
    handshake.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    for r in regions:
        cursor = _round_up(cursor, r.align)
        offsets[r.name] = cursor
        cursor += r.nbytes
    total = _round_up(cursor, 16)
    return offsets, total


# =============================================================================
# Sm100MegaMoEKernel
# =============================================================================


class Sm100MegaMoEKernel(Sm100SwapABSwigluFp4Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 (+ combine in S3).

    See module docstring for the full host contract.  Internal flow:

      * ``__init__`` validates inputs, derives the dependent shape /
        pool constants, builds the region tables and caches their
        offsets + totals.
      * ``_smem_misc_budget_bytes`` is overridden to add the dispatch
        warps' SMEM footprint to the base's 1024 B reservation; the
        base's ``_compute_stages`` then subtracts the new total from
        the AB-stage SMEM budget.
      * ``get_workspace_sizes`` returns the (local, shared) byte
        budgets computed at construction time.
      * ``__call__`` partitions the two opaque workspaces, builds the
        dual-dtype views where dispatch and GEMM cannot share an
        iterator dtype, derives ``sm_count`` from ``max_active_clusters
        * cluster_size``, assembles ``TokenCommArgs``, and forwards
        to ``super().__call__``.
    """

    def __init__(
        self,
        # Base-class kwargs (forwarded 1:1 to ``super().__init__``).
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 16,
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants (the only 5 that are not
        # derivable from base kwargs / static_expert_shape).
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        # MegaMoE form B opt-in (per-token cross-rank
        # ``red.global.add.v2.bf16x2`` reduce on fc2 output --
        # combine_output shape collapses to (token, 1, hidden); the
        # per-(src_token, src_topk) cells are atomic-added into the
        # single per-(src_rank, src_token) accumulator row).
        # Form A (False, default) writes one BF16 cell per (src_rank,
        # src_token, src_topk) into combine_output (token, topk, hidden)
        # and the host reduces the topk axis after the kernel returns.
        #
        # Form B caller contract additions (NOT enforced here -- caller /
        # runner responsibility):
        #   * ``combine_output`` MUST be zero-init before launch (REDG
        #     accumulates onto the existing cells).
        #   * ``combine_output.shape[1]`` SHOULD be 1.
        fc2_in_kernel_topk_reduce: bool = False,
    ) -> None:
        # MegaMoE currently requires ``static_expert_shape`` because the
        # workspace-sizing pool formulas need ``num_experts_per_rank`` /
        # ``intermediate_gateup`` at construction time, and the dispatch
        # primitives' inner loops are also coded against codegen-time
        # expert counts.  Dynamic-shape support is future work.
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm100MegaMoEKernel currently requires "
                "static_expert_shape != None (dynamic-shape MegaMoE is "
                "future work).")
        # ``hidden`` is taken as a separate kwarg because SMEM static
        # sizing inside ``_dispatch_smem_bytes`` needs it before the
        # static_expert_shape tuple is decoded; the consistency check
        # below catches caller drift between the two sources.
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal static_expert_shape[2] ({static_expert_shape[2]})."
            )

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            sf_vec_size=sf_vec_size,
            scenario=scenario,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
        )

        # Flip the MegaMoE toggle BEFORE the base's ``_setup_attributes``
        # runs (which it does inside ``__call__``).  All subclass attrs
        # accessed by hooks / ``_smem_misc_budget_bytes`` must be set
        # by the end of this method.
        self.enable_token_comm = True

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden

        # Derived from static_expert_shape (= (num_experts_per_rank,
        # intermediate_gateup, hidden) per fc1_weight.shape convention).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2

        # NVFP4: 4 bits/elem -> 2 elements per byte.
        self.hidden_bytes = self.hidden // 2
        # NVFP4 SF: one FP8 scale per ``sf_vec_size`` elements, 4 FP8
        # scales packed into each uint32.  Ceil-divide so that ``hidden``
        # values which round up to a non-integer SF uint32 count (e.g.
        # ``hidden=1632`` -> 25.5 uint32) still allocate one extra trailing
        # uint32 slot per token.  The caller's contract (see
        # ``__call__``'s ``activation_sf`` docstring + ``megamoe_design.md``
        # §11) requires the host-supplied ``activation_sf`` row stride
        # already pads to that same ceiling with zero-filled bytes so the
        # dispatch_pull LDG.32 byte stride matches the host wire format.
        sf_atom_k_elements = 4 * self.sf_vec_size
        self.sf_uint32_per_token = (self.hidden + sf_atom_k_elements -
                                    1) // sf_atom_k_elements
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank

        # Tokens covered by one cluster tile = ``mma_tiler_n *
        # cluster_n``.  This is the per-task-tile release-counter
        # granularity used by ``_dispatch_pull`` (see
        # ``fc12_integrate_comm.md`` §4 C3); v1's ``cluster_n == 1``
        # constraint (enforced by ``_validate_mma_tiler_and_cluster_shape``)
        # reduces it to ``mma_tiler_n``.
        self.cluster_tile_tokens = mma_tiler_mnk[1] * cluster_shape_mnk[1]

        # C1 / C2 are tautologically satisfied now that block_m /
        # sf_block_m are unified with token_padding_block / sf_padding_block
        # (we never carry the dispatch-side names separately).  C3 reduces
        # to a divisibility check: the dispatch_pull release-add granularity
        # (token-axis cluster tile) must be a multiple of the pool-block
        # granularity so that one task tile maps cleanly to a contiguous
        # run of pool blocks.
        if self.cluster_tile_tokens % self.token_padding_block != 0:
            raise ValueError(
                f"C3 violated: cluster_tile_tokens "
                f"({self.cluster_tile_tokens}) must be a multiple of "
                f"token_padding_block ({self.token_padding_block}); "
                f"otherwise pool row offsets and release counter slots "
                f"will not align.")

        # Pool sizes derived from first principles -- cached so
        # ``_build_*_region_specs`` and ``__call__`` (and the
        # ``num_padded_sf_pool_tokens`` kwarg passed to _dispatch_pull)
        # all read from one source of truth.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()

        # Region layout (same call drives both get_workspace_sizes() and
        # the __call__ partition).
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs)
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs)
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r
            for r in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r
            for r in self._shared_region_specs
        }

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM bytes consumed by the dispatch warps' extra storage region
        (see ``token_comm_extra_smem_storage_class`` for the struct).

        Three fields, each rounded up to its own alignment to mirror
        ``cute.struct`` packing inside SMEM:

          * pull_mbar     : Int64 x _DispatchWarpCount   (mbar align 16)
          * smem_expert_count : Int32 x _TotalExpertsCapacity  (align 16)
          * pull_buffer   : Uint8 x (_DispatchWarpCount * hidden_bytes)
                            (TMA target -> align 128)
        """
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = _TotalExpertsCapacity * 4
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
        return (_round_up(pull_mbar_bytes, 16) +
                _round_up(expert_count_bytes, 16) +
                _round_up(pull_buffer_bytes, 128))

    def _smem_misc_budget_bytes(self) -> int:
        """MegaMoE-augmented SMEM misc budget: base reservation + the
        dispatch warps' pull_buffer / pull_mbar / smem_expert_count
        region.  Without this override the base's AB-stage budget
        calculation would over-count by ~16 KB on NVFP4 (4-warp,
        3584 B/token) -- the resulting ``num_ab_stage`` would exceed
        the available SMEM and either fail to launch or scribble over
        the dispatch region.
        """
        return super()._smem_misc_budget_bytes() + self._dispatch_smem_bytes()

    # =========================================================================
    # Pool sizing (first-principles)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        """Worst-case pool sizes.

        ``pool_token_capacity``: every received token from any peer can
        replicate to ``min(num_topk, num_experts_per_rank)`` local
        experts; worst case is ``world_size * max_tokens_per_rank``
        tokens received, each replicated up to that bound.  Each of
        the ``num_experts_per_rank`` experts wastes up to
        ``token_padding_block - 1`` rows at its tail; round the whole
        sum up to the pool-layout granularity ``token_padding_block``.

        ``pool_sf_capacity``: same number of expert blocks as the data
        pool, each padded to ``sf_padding_block`` rows (UTCCP 4x32
        swizzle that the SF TMA load expects).

        ``pool_task_tile_capacity``: ``ceil(pool_token_capacity,
        cluster_tile_tokens)``.  C3 makes ``cluster_tile_tokens`` a
        multiple of ``token_padding_block`` so this stays exact.
        """
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = max_recv * max_per_token + num_experts_per_rank * (
            token_padding_block - 1)
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_sf_capacity = (pool_token_capacity //
                            token_padding_block) * sf_padding_block
        # ``pool_task_tile_capacity`` = an upper bound on the number of
        # distinct (expert, task-tile) counter slots ``l1_arrival_count``
        # will ever be addressed by.  The total comes from
        # ``sum_e ceil(valid_e, cluster_tile_tokens)``, which the
        # arithmetic identity bounds as:
        #
        #     sum_e ceil(t_e, c) <= ceil(sum_t, c) + (num_experts_per_rank)
        #
        # The ``+ num_experts_per_rank`` slack covers the per-expert
        # ceiling waste (each expert may "round up" at its tail by up
        # to one slot).  Without this slack ``l1_arrival_count`` is
        # sized too small, dispatch's per-task-tile release-add
        # overruns into the adjacent ``token_src_metadata`` region,
        # and fc1's TMA-B spin reads garbage bytes -- hang.  This
        # mirrors the base ``get_workspace_size_in_bytes`` formula
        # for ``fc1_done_counter`` (``kernel_fc12.py:557-566``), which
        # uses the same ``+ experts`` slack.
        pool_task_tile_capacity = (
            pool_token_capacity + cluster_tile_tokens -
            1) // cluster_tile_tokens + num_experts_per_rank
        return (
            pool_token_capacity,
            pool_sf_capacity,
            pool_task_tile_capacity,
        )

    # =========================================================================
    # Region tables
    # =========================================================================

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        """Local-only regions (no peer access via ``peer_rank_ptr_mapper.map`` in
        ``src/dispatch_kernel.py``).
        """
        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        sf_uint32_per_token = self.sf_uint32_per_token
        intermediate_downproj = self.intermediate_downproj
        mma_tiler_n = self.mma_tiler_mnk[1]
        sf_vec_size = self.sf_vec_size
        sf_padding_block = self.sf_padding_block

        # fc1_output_sf / fc1_done_counter sizing mirrors base
        # ``get_workspace_size_in_bytes`` (kernel_fc12.py ~lines 525-543).
        sf_total_rows_upper = pool_token_capacity + num_experts_per_rank * sf_padding_block
        sf_block_cols = (((intermediate_downproj // sf_vec_size) + 3) // 4) * 4
        fc1_done_slots = (pool_token_capacity + mma_tiler_n -
                          1) // mma_tiler_n + num_experts_per_rank

        specs: List[_RegionSpec] = [
            # L1 input pool (dispatch_pull writes -> fc1 reads).  Stored
            # as Uint8 bytes; the NVFP4 view at the same offset is
            # built inside ``__call__``.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            # Stored as Int32 (dispatch_pull's 32 b read/write); the FP8
            # view for activation_sf is built at the same offset.
            # 1D Int32 atom-flat buffer.  Total Int32 count = pool_sf_capacity
            # (M-axis token positions) * sf_uint32_per_token (K-atom count),
            # laid out atom-by-atom per cute SFA layout.  dispatch writes
            # individual Int32 slots via the linear offset returned by
            # ``src/sf_swizzle.py:sf_atom_int32_offset``; the mma side
            # re-views this same byte buffer through ``tile_atom_to_shape_SF``
            # which reads back the atom-swizzled bytes.
            _RegionSpec(
                "l1_sf_buffer",
                cutlass.Int32,
                (pool_sf_capacity * sf_uint32_per_token, ),
                16,
            ),
            _RegionSpec(
                "l1_topk_weights_buffer",
                cutlass.Float32,
                (pool_token_capacity, ),
                16,
            ),
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity, ),
                16,
            ),
            _RegionSpec(
                "token_src_metadata",
                cutlass.Uint8,
                (pool_token_capacity, _TokenMetadataBytes),
                16,
            ),
            _RegionSpec(
                "expert_send_count",
                cutlass.Int64,
                (num_total_experts, ),
                16,
            ),
            _RegionSpec(
                "grid_sync_counter",
                cutlass.Int32,
                (_GridSyncSlotCount, ),
                16,
            ),
            _RegionSpec(
                "fc1_output",
                cutlass.Float4E2M1FN,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
            _RegionSpec(
                "fc1_output_sf",
                cutlass.Float8E4M3FN,
                (sf_total_rows_upper, sf_block_cols),
                128,
            ),
            _RegionSpec(
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots, ),
                16,
            ),
        ]

        if self.load_balance_mode == "atomic_counter":
            specs.append(
                _RegionSpec(
                    "load_balance_counter",
                    cutlass.Int32,
                    (1, ),
                    16,
                ))

        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        """Shared (peer-mapped) regions -- every entry is reached from
        some ``peer_rank_ptr_mapper.map(local_ptr, peer_rank, byte_off)``
        call site inside ``src/dispatch_kernel.py``:

          * ``src_token_topk_idx`` -- ``_dispatch_prep`` round 3
          * ``expert_recv_count`` / ``expert_recv_count_sum``
            -- ``_dispatch_barrier`` step 2 (b64 store + sys-atomic-add)
          * ``nvlink_barrier_signal``
            -- ``_nvlink_barrier_3stage`` stage B (slot=0 and slot=1)
        """
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        # ``MAX_SLOT`` in ``_dispatch_prep`` round 3: every (token, topk)
        # edge any peer might publish for this rank's local experts.
        max_slot = max_tokens_per_rank * num_topk

        return [
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
            _RegionSpec(
                "expert_recv_count",
                cutlass.Int64,
                (world_size, num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum",
                cutlass.Int64,
                (num_experts_per_rank, ),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount, ),
                16,
            ),
        ]

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)`` -- the byte
        budgets for the two opaque workspaces the host must allocate.
        Both totals are invariant across launches; per-launch ``T``
        may be <= ``max_tokens_per_rank``.
        """
        return self._local_total, self._shared_total

    # =========================================================================
    # Workspace partition helpers
    # =========================================================================

    @staticmethod
    def _make_typed_view(
        byte_workspace: cute.Tensor,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
    ) -> cute.Tensor:
        """Build a typed cute view at ``byte_offset`` of the opaque
        Uint8 workspace.  ``+byte_offset`` advances byte_offset bytes
        (workspace iterator is Uint8 typed); ``cute.recast_ptr`` then
        re-interprets the element type without moving the address.
        """
        typed_iter = cute.recast_ptr(
            byte_workspace.iterator + byte_offset,
            dtype=cute_dtype,
        )
        return cute.make_tensor(typed_iter,
                                cute.make_layout(shape, stride=stride))

    def _view_local(
        self,
        local_workspace: cute.Tensor,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        """Partition a region of the local workspace.  With no overrides,
        uses the region's declared dtype + shape + row-major stride;
        overrides let dual-view callers build alternate-dtype views at
        the same byte offset.
        """
        return self._partition_region(
            local_workspace,
            self._local_offsets,
            self._local_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_shared(
        self,
        shared_workspace: cute.Tensor,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            shared_workspace,
            self._shared_offsets,
            self._shared_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _partition_region(
        self,
        byte_workspace: cute.Tensor,
        offsets: Dict[str, int],
        spec: _RegionSpec,
        *,
        cute_dtype: Optional[Any],
        shape: Optional[Tuple[int, ...]],
        stride: Optional[Tuple[int, ...]],
    ) -> cute.Tensor:
        dt = cute_dtype if cute_dtype is not None else spec.cute_dtype
        sh = shape if shape is not None else spec.shape
        st = stride
        if st is None:
            if cute_dtype is None and shape is None:
                st = spec.stride_row_major
            else:
                # Derive row-major from the (possibly overridden) shape.
                out: List[int] = [1]
                for d in reversed(list(sh)[1:]):
                    out.append(out[-1] * d)
                out.reverse()
                st = tuple(out)
        return self._make_typed_view(
            byte_workspace,
            offsets[spec.name],
            dt,
            sh,
            st,
        )

    # =========================================================================
    # SMEM extra storage
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        """Per-CTA SMEM block for the dispatch warps -- mirrors
        ``src/dispatch_kernel.py:_make_shared_storage``.  Factory
        pattern so the inner ``@cute.struct`` captures
        ``self.hidden_bytes`` as a monomorphic int per kernel instance.
        """
        hidden_bytes = self.hidden_bytes

        # ``@cute.struct`` checks element types by identity; use the
        # ``cutlass.cutlass_dsl`` scalar aliases (Int64 / Int32 / Uint8)
        # rather than the ``cutlass.*`` re-exports, matching the working
        # pattern in ``src/dispatch_kernel.py:_make_shared_storage``.  The
        # ``cutlass.Int64`` etc. variants are accepted by other cuTeDSL
        # surfaces but not by ``@cute.struct`` element validation.
        @cute.struct
        class TokenCommStorage:
            pull_mbar: cute.struct.MemRange[Int64, _DispatchWarpCount]
            smem_expert_count: cute.struct.MemRange[Int32,
                                                    _TotalExpertsCapacity]
            pull_buffer: cute.struct.MemRange[Uint8,
                                              _DispatchWarpCount * hidden_bytes]

        return TokenCommStorage

    # =========================================================================
    # Hook 1: dispatch -> fc1 release counter pointer
    # =========================================================================

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Expose the dispatch -> fc1 release counter to the sched
        extension.  Returning a non-None pointer toggles the fc1-phase
        peek inside ``SwapABSwigluFp4Fc12SchedExtension.enrich_work_tile_info``
        (mirrors the fc1 -> fc2 path's ``fc1_done_counter_ptr``).
        """
        return token_comm_args.fc1_ready_counter.iterator

    # =========================================================================
    # Hook 2: sched warp pre-init wait (NamedBarrier 9)
    # =========================================================================

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Sched-warp blocks on NamedBarrier 9 until this CTA's dispatch
        warps have crossed the cross-rank NVLink slot=0 acquire fence
        inside ``_dispatch_barrier``.  Only then is
        ``expert_recv_count_sum`` finalised for the sched warp's late
        ``internal_init`` to read.  The fence covers dispatch warps
        but does not auto-propagate to other warp groups on the same
        CTA, hence this intra-CTA handshake.
        """
        nb = pipeline.NamedBarrier(
            barrier_id=_DispatchToSchedNamedBarrierId,
            num_threads=5 * 32,
        )
        nb.arrive_and_wait()

    # =========================================================================
    # Hook 3: fc1 TMA-B pre-dispatch spin
    # =========================================================================

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self,
        token_comm_args,
        work_tile_info,
    ):
        """TMA-B blocking spin at each fc1 task tile head on
        ``fc1_ready_counter[cumulative_token_block_count + tile_n_idx]``
        until it reaches ``valid_tokens_in_tile`` (dynamic threshold:
        dispatch does NOT pull padding rows).  Short-circuited by
        ``work_tile_info.peek_ready`` from the sched warp's
        enrich-time peek.
        """
        counter_slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
        counter_ptr = token_comm_args.fc1_ready_counter.iterator + counter_slot
        if not work_tile_info.peek_ready:
            iket.range_push("tma_token_fc1_wait")
            spin_wait(
                counter_ptr,
                lambda v: v >= work_tile_info.valid_tokens_in_tile,
                fail_sleep_cycles=20,
            )
            iket.range_pop()

    # =========================================================================
    # Hook 4: dispatch warp body (warps 8-11)
    # =========================================================================

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Dispatch warp body (global warps 8-11).  Runs the three-stage
        flow back-to-back; the NamedBarrier 9 ``arrive()`` between
        barrier and pull releases the sched warp's pre-init wait
        without blocking dispatch.

        Indexing notes:

          * ``sm_idx`` passed to dispatch primitives is the CTA-linear
            id (always dense in ``[0, num_active_clusters * cluster_size)``
            with id 0 present; unlike ``%smid`` which may skip 0 and
            deadlock the SM-0-only branches inside ``_dispatch_barrier``
            / ``_nvlink_barrier_3stage``).  Swap-AB grid layout:
            ``gdx = cluster_shape_mn[1]``, ``gdy = cluster_shape_mn[0]``,
            ``gdz = max_active_clusters``.
          * ``warp_idx`` arrives in GLOBAL ``[8, 12)``; dispatch
            primitives index by dispatch-LOCAL ``[0, 4)`` -- rebase
            by subtracting ``self.dispatch_warp_id[0]``.

        Reg dealloc to 48 (per ``fc12_integrate_comm.md`` §3) is NOT
        emitted here -- TODO until the matching epi / mma / tma /
        sched reg alloc is added in the base.
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx) + Int32(self.cluster_shape_mn[1]) * Int32(bidy) +
            Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0]) *
            Int32(bidz))
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_id[0])

        iket_active = (cta_linear_id == Int32(0)) and (local_warp_idx
                                                       == Int32(0))
        if iket_active:
            iket.range_push("Dispatch_Prep")

        _dispatch_prep(
            token_comm_storage,
            token_comm_args.input_topk_idx_buffer,
            token_comm_args.expert_send_count,
            token_comm_args.src_token_topk_idx,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_tokens=token_comm_args.input_token_buffer.shape[0],
            num_topk=self.num_topk,
            num_sms=token_comm_args.sm_count,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

        if iket_active:
            iket.range_pop()
            iket.range_push("Dispatch_Barrier")

        _dispatch_barrier(
            token_comm_args.expert_send_count,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.nvlink_barrier_signal,
            token_comm_args.grid_sync_counter,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

        # Non-blocking arrive: releases the sched warp's pre-init wait;
        # dispatch warps continue into pull without blocking.
        nb_dispatch_to_sched = pipeline.NamedBarrier(
            barrier_id=_DispatchToSchedNamedBarrierId,
            num_threads=5 * 32,
        )
        nb_dispatch_to_sched.arrive()

        if iket_active:
            iket.range_pop()
            iket.range_push("Dispatch_Pull")

        # ``block_m`` / ``sf_block_m`` kwargs at the dispatch_pull call
        # site keep the standalone dispatch_kernel.py API stable -- we
        # just feed it our unified ``token_padding_block`` /
        # ``sf_padding_block`` constants.
        _dispatch_pull(
            token_comm_storage,
            token_comm_args.input_token_buffer,
            token_comm_args.input_sf_buffer,
            token_comm_args.input_topk_weights_buffer,
            token_comm_args.src_token_topk_idx,
            token_comm_args.expert_recv_count,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.fc1_input_token_buffer,
            token_comm_args.fc1_input_sf_buffer,
            token_comm_args.fc1_input_topk_weights_buffer,
            token_comm_args.fc1_ready_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            block_m=self.token_padding_block,
            sf_block_m=self.sf_padding_block,
            # ``_dispatch_pull`` still names the kwarg ``cluster_tile_m``
            # (post-swap M in scheduler-speak); we feed it our
            # ``cluster_tile_tokens``, which carries the same number with
            # the axis-explicit name.
            cluster_tile_m=self.cluster_tile_tokens,
            hidden=self.hidden,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            num_padded_sf_pool_tokens=self.pool_sf_capacity,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )

        if iket_active:
            iket.range_pop()

    # =========================================================================
    # Hook 5: kernel-tail (NamedBarrier 8 + NVLink slot=1)
    # =========================================================================

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """12-warp rendezvous on NamedBarrier 8, then dispatch warps
        drive ``_nvlink_barrier_3stage(slot=1)`` with prologue +
        epilogue grid syncs.  The rendezvous establishes the
        happens-before edge from epi STG into ``combine_output`` to
        the cross-rank release.
        """
        nb_kernel_tail = pipeline.NamedBarrier(
            barrier_id=_KernelTailNamedBarrierId,
            num_threads=12 * 32,
        )
        nb_kernel_tail.arrive_and_wait()

        if warp_idx >= self.dispatch_warp_id[0]:
            bidx, bidy, bidz = cute.arch.block_idx()
            cta_linear_id = (
                Int32(bidx) + Int32(self.cluster_shape_mn[1]) * Int32(bidy) +
                Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0]) *
                Int32(bidz))
            local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_id[0])
            _nvlink_barrier_3stage(
                token_comm_args.nvlink_barrier_signal,
                token_comm_args.grid_sync_counter,
                token_comm_args.peer_rank_ptr_mapper,
                cta_linear_id,
                local_warp_idx,
                lane_idx,
                slot=1,
                num_sms=token_comm_args.sm_count,
                world_size=self.world_size,
                local_rank=self.local_rank,
                prologue_grid_sync=True,
                epilogue_grid_sync=True,
            )

    # =========================================================================
    # __call__
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,  # (T, hidden) NVFP4
        activation_sf: cute.
        Tensor,  # (T, round_up(hidden, sf_atom_block_k)) FP8
        topk_idx: cute.Tensor,  # (T, num_topk) Int64
        topk_weights: cute.Tensor,  # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        # Combine destination (peer write target under S3; local fc2
        # output region under S2 -- same memory, same caller).
        combine_output: cute.Tensor,  # (T, num_topk, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Tensor,  # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,  # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``
        # before entering the device kernel.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``activation_sf`` / ``topk_weights`` MUST
            point into memory reachable via ``peer_rank_ptr_mapper.map(...)``
            (typically NVSHMEM symmetric heap).  Single-rank degenerate
            runs (``peer_rank_ptr_mapper.offsets[local_rank] == 0`` by NVSHMEM
            convention) are allowed.
          * ``topk_idx`` is read on the local rank only; placement is
            unconstrained (cuda local or sym heap).
          * ``fc1_weight`` / ``fc1_weight_sf`` / ``fc2_weight`` /
            ``fc2_weight_sf`` are local-only.
          * ``combine_output`` is the per-rank S3 combine STG target;
            under S2 it acts as the rank's local BF16 fc2 output.
            Placement: sym heap (peer write target) or local in the
            single-rank degenerate case.

        Workspace zero-init contract: caller is currently expected to
        zero ``shared_workspace`` before launch (the dispatch
        primitives' counters / signals rely on a clean state).  This
        contract may be tightened later to have the kernel take
        ownership of the reset.
        """
        # ``max_active_clusters`` and ``cluster_size`` are both Python ints
        # at trace time, so the product folds to a Python int that flows
        # cleanly to every dispatch primitive's ``num_sms: Constexpr[int]``
        # slot.
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        hidden = self.hidden
        sf_per_token_fp8 = self.sf_uint32_per_token * 4  # 4 FP8 SFs per Int32

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + NVFP4
        # view (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(
            local_workspace,
            "l1_token_buffer",
        )
        l1_token_buffer_nvfp4 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            cutlass.Float4E2M1FN,
            (pool_token_capacity, hidden),
            (hidden, 1),
        )

        # L1 SF buffer: Int32 view (dispatch_pull's [j, t] 2D indexing) +
        # FP8 view (base.activation_sf re-views via tile_atom_to_shape_SF
        # off the iterator, so the stride here is informational only).
        l1_sf_buffer_i32 = self._view_local(
            local_workspace,
            "l1_sf_buffer",
        )
        l1_sf_buffer_fp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_sf_buffer"],
            cutlass.Float8E4M3FN,
            (pool_sf_capacity, sf_per_token_fp8),
            (sf_per_token_fp8, 1),
        )

        l1_topk_weights_buffer = self._view_local(
            local_workspace,
            "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(
            local_workspace,
            "l1_arrival_count",
        )
        # token_src_metadata storage = (pool_token_capacity, 12) Uint8;
        # dispatch_pull writes three Uint32 fields per pool token row via
        # byte-stepped pointer arithmetic on this Uint8 view (so its
        # element-width-1 ``+ pool_token_idx * 12`` matches a 12-byte row
        # stride).  The fc2 epilogue's metadata-LDG path wants a logical
        # ``(N, 3) Uint32`` view of the same bytes -- it does that recast
        # itself inside ``_run_fc2_task_tile`` to keep the dispatch-side
        # Uint8 ABI intact (dispatch_kernel.py is a standalone module
        # whose API the fused kernel does not mutate).
        token_src_metadata = self._view_local(
            local_workspace,
            "token_src_metadata",
        )
        expert_send_count = self._view_local(
            local_workspace,
            "expert_send_count",
        )
        grid_sync_counter = self._view_local(
            local_workspace,
            "grid_sync_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local(
            local_workspace,
            "fc1_done_counter",
        )

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace,
                "load_balance_counter",
            )

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace,
            "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(
            shared_workspace,
            "expert_recv_count",
        )
        expert_recv_count_sum = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace,
            "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum``
        # buffer -- low32 bits hold per-expert total token count after
        # _dispatch_barrier; zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank, ),
            stride=(2, ),
        )

        token_comm_args = TokenCommArgs(
            input_token_buffer=activation,
            input_sf_buffer=activation_sf,
            input_topk_idx_buffer=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_buffer_i32,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output,
            nvlink_barrier_signal=nvlink_barrier_signal,
            grid_sync_counter=grid_sync_counter,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=self.local_rank,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )

        # C1 / C2 are tautological (token_padding_block == "block_m";
        # sf_padding_block == "sf_block_m") so the pool layout and the
        # sched cumulative-row offsets align by construction.
        #
        # ``combine_output`` is already in the MoE-domain
        # ``(max_tokens_per_rank, num_topk, hidden)`` layout the base
        # expects for ``fc2_output``, so it can be forwarded as-is.  The
        # base's fc2 epilogue uses ``Fc2OutputDest`` with the
        # ``token_comm_args`` we hand it to compute per-(src_rank,
        # src_token, src_topk) STG destinations via ``peer_rank_ptr_mapper.map``, so
        # the axis 0 of ``combine_output`` indexes the *target* rank's
        # per-rank token row (NOT the local pool token row); the host
        # reduces the topk axis after the kernel returns.
        super().__call__(
            activation=l1_token_buffer_nvfp4,
            fc1_weight=fc1_weight,
            activation_sf=l1_sf_buffer_fp8,
            fc1_weight_sf=fc1_weight_sf,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=combine_output,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )
