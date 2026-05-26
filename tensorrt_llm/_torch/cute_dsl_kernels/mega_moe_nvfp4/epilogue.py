# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Autonomous epilogue for the fused fc1+fc2 swap-AB MegaMoE kernel.

Component boundaries use ``TensorWithContract`` to keep per-thread RMEM layout
semantics explicit.  See ``megamoe_design.md`` for the epilogue dataflow.
"""

import dataclasses
from typing import Any, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from .iket_compat import iket

import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import dsl_user_op

from .contract import (Contract, FunctionMapping, Space, TensorWithContract,
                       assert_contract_equivalent)
from .fc1_fc2_fuse_sched import BlockPhase
from .megamoe_constants import Nvfp4BlockSize

Fc1GateUpInterleave = 16
EpilogueTokenTile = 64
Fc1EpilogueOutputTile = 64
WarpThreadCount = 32
EpiWarpCount = 4

# =============================================================================
# Module-local helpers
# =============================================================================


@dsl_user_op
def _red_add_relaxed_sys_v2_bf16x2(
    addr,
    val0_packed_bf16x2,
    val1_packed_bf16x2,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.relaxed.sys.global.add.v2.bf16x2 [addr], {v0, v1};``.

    Used by the fc2 epi REDG path (form B) to atomic-add a 2-element
    vector of packed bf16x2 cells (= 4 bf16 = 8 B) onto a peer rank's
    combine-output row.  ``red`` (vs ``atom``) drops the return value,
    saving the return-path round-trip; ``relaxed.sys`` is the weakest
    qualifier that still publishes the write cross-rank -- ordering
    against the kernel-tail NVLink barrier (slot=1, system-scope
    release) covers the final visibility fence, so REDG itself doesn't
    need release semantics.

    Per-thread cost: 1 PTX instruction, 8 B atomic-add.  Per-warp
    coalescing happens inside the memory subsystem -- 4 consecutive
    threads writing contiguous 8 B segments of the same destination
    row get merged into a single sector-aligned 32 B atomic
    transaction (this is the whole reason ``Fc2UnpackPermuteStg``'s
    REDG path does the STTM + LDTM shuffle before issuing REDGs).

    Direct inline asm rather than ``cute.arch.red`` because the latter
    routes a scalar BFloat16 ``val`` through ``nvvm.red(type_=bf16x2)``
    (type-mismatched against the dtype enum) and has no vector-form
    surface for the ``.v2`` qualifier.  Constructing a vector ir.Value
    from two packed-bf16x2 fp32 RMEM slots via ``llvm.bitcast`` is
    feasible but not shipped in any cuTeDSL example we can copy from,
    so we go with the same hand-rolled-PTX pattern as
    ``_red_add_release_gpu_s32`` below.

    ``val{0,1}_packed_bf16x2`` are two 32-bit ``cutlass.Float32``
    values whose bit patterns are the packed bf16x2 cells (i.e. the
    recast view of the transpose's packed-bf16x2 RMEM slots).  PTX
    constraint ``r`` only cares about the 32-bit register payload,
    not the source dtype.

    ``.noftz`` is REQUIRED by PTX for any ``red`` with type
    ``.bf16`` / ``.bf16x2`` / ``.f16`` / ``.f16x2`` (per ISA spec).
    """
    llvm.inline_asm(
        None,
        [
            addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            val0_packed_bf16x2.ir_value(loc=loc, ip=ip),
            val1_packed_bf16x2.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.sys.global.add.noftz.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_release_gpu_s32(
    counter_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.release.gpu.add.s32`` to a GMEM int32 location.

    Used by the fused fc1+fc2 epilogue to publish the per-(expert,
    token_block) fc1-done counter increment that the fc2 loader warp
    spins on.

    ``red`` (vs ``atom``) drops the return value, saving a return-path
    round-trip; ``release.gpu`` orders prior GMEM writes from this thread
    (the per-thread STG SFC + per-warp TMA bulk store of the fc1 output)
    before the counter increment becomes visible to readers, forming a
    release-acquire pair with the fc2 loader's ``ld.acquire.gpu.b32``.

    Caller's responsibility: ensure all per-CTA GMEM writes for the fc1
    output of ``(expert_idx, token_block_idx)`` are flushed before
    invoking this helper (e.g. via ``cp_async_bulk_wait_group(0,
    read=True)`` for TMA stores plus a 4-warp NamedBarrier so per-thread
    STG SFC instructions of all CTA threads are issued).  This helper
    does NOT add its own fence -- ``release.gpu`` ordering is sufficient
    given the upstream caller-side flush.

    The helper is single-thread (caller must guard with a thread predicate
    such as ``if tidx == 0`` to avoid 128 redundant atomic ops per CTA).
    """
    # ``Pointer.toint(...)`` returns a cutlass Numeric (Int64) wrapper, not
    # a raw ``ir.Value``; call ``.ir_value(...)`` to unwrap before handing
    # it to ``llvm.inline_asm`` (which validates operand types against MLIR
    # ``ir.Value``).  Mirrors the cutlass ``utils/distributed.py`` idioms
    # for ``multimem.red.release.*`` family helpers.
    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Region tag
# =============================================================================


class Region:
    """Codegen-time region tag for a 16x32 sub-region within a 32x32 tile."""

    Top = 0
    Bottom = 1


# =============================================================================
# TmemTranspose16x32
# =============================================================================


class _TmemTranspose16x32Core:
    """Contract-naive physical implementation of the 16x32 -> 32x16 TMEM
    in-place transpose.  Shared by:

      - ``TmemTranspose16x32``       : fc1 epi codomain naming
                                       (``intermediate_output_idx``);
                                       elements are fp32 (swiglu fold output).
      - ``TmemTranspose16x32Packed`` : fc2 epi codomain naming
                                       (``hidden_pair_idx``); elements are
                                       32-bit packed ``(bf16, bf16)`` pairs.

    The (lane_idx, elem_idx) physical distribution is identical for both
    subclasses -- the underlying tcgen05 atoms are 32-bit element atoms,
    agnostic to whether each 32-bit slot holds an fp32 or a packed bf16x2.
    Only the codomain semantic names differ, expressed via the subclass's
    ``InputContract`` / ``OutputContract`` class attributes.

    Per-thread RMEM coordinate convention (used by both subclasses' contracts):

      - ``lane_idx`` -- warp lane id (= thread index within warp), in [0, 32).
      - ``elem_idx`` -- per-thread reg index, in [0, 16).

    Subclasses MUST override these two class attributes:
      ``InputContract``  -- (lane_idx, elem_idx) -> codomain mapping after
                            R1.Load (or after ``reg_tensor`` is fed in for
                            skip-R1.Load mode).
      ``OutputContract`` -- (lane_idx, elem_idx) -> codomain mapping after
                            ``r4_perm`` has run all four rounds.

    The Core's ``__init__`` reads ``self.InputContract`` / ``self.OutputContract``
    via Python's normal MRO attribute lookup; the subclass's overrides take
    precedence at construction time.
    """

    # Subclasses MUST override these.
    InputContract: Contract
    OutputContract: Contract

    _PermR1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    _PermR3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    _PermR4 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)

    _TmemRowStride = 1 << 16
    _io_dtype = cutlass.Float32

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1), ),
            stride=(((_TmemTranspose16x32Core._TmemRowStride, 1), 0), ),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor,
                        num_regs: int,
                        offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset,
            cute.make_layout((((num_regs, ), 1), ), stride=(((1, ), 0), )),
        )

    @staticmethod
    def load_subtile_raw_acc(
        tmem_subtile_tensor: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """LDTM the entire 32-lane x 64-col raw acc region of one epi
        subtile into 4 independent (16,) fp32 RMEM tensors.

        Used by the overlap-acc unroll path in
        ``_run_fc{1,2}_task_tile`` to extract all raw acc data of the
        first 2 subtiles up front, so that the acc TMEM can be released
        to the next mma right after the first subtile's 4 LDTMs (instead
        of waiting for a full subtile body to complete).

        ``tmem_subtile_tensor`` is the (32 lanes, 64 cols) view onto a
        single epi subtile's acc TMEM region (already offset by
        ``warp_lane_offset + acc_stage_col_offset + subtile_col_offset``;
        see ``SwapABSwigluFp4Epilogue._subtile_local_tmem_tensor``).

        Returns a 4-tuple of (16,) fp32 RMEM tensors, each carrying
        the (lane_idx, elem_idx) -> codomain distribution described by
        ``TmemTranspose16x32.InputContract`` /
        ``TmemTranspose16x32Packed.InputContract`` (physically identical
        for fc1 and fc2, only codomain semantic names differ):

          [0] gate_lo / first-half top   -- subtile cols 0..31, lanes 0..15
          [1] up_lo   / first-half bot   -- subtile cols 0..31, lanes 16..31
          [2] raw_top / second-half top  -- subtile cols 32..63, lanes 0..15
          [3] raw_bot / second-half bot  -- subtile cols 32..63, lanes 16..31

        4 atom calls of ``Ld16x64bOp(Repetition.x16) Float32`` -- the
        same atom currently used by the per-subtile entry LDTM in
        ``_run_fc1_subtile`` and by ``second_t.r1_load`` /
        ``Fc2AccLoadAndPack`` per-half LDTMs.  Caller is expected to
        wrap each output in ``TensorWithContract`` with
        ``TmemTranspose16x32{,Packed}.InputContract`` before handing
        them downstream.
        """
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            _TmemTranspose16x32Core._io_dtype,
        )

        ptr = tmem_subtile_tensor.iterator
        half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride

        # 4 source 16-lane x 32-col views over the (32, 64) subtile region:
        #   first  half (cols 0..31): top  lanes 0..15  / bot lanes 16..31
        #   second half (cols 32..63): top lanes 0..15  / bot lanes 16..31
        # All offsets are Python ints (compile-time const) so cute can
        # const-fold them and infer the correct (>= 8 B / 2 col) ptr
        # alignment that the LDTM atom requires.  Using ``cutlass.Int32``
        # offsets here would wrap them as SSA values that cute treats as
        # alignment-unknown, tripping the atom's verifier.
        first_top_view = cute.make_tensor(
            ptr,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        first_bot_view = cute.make_tensor(
            ptr + half_lane_off,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        second_top_view = cute.make_tensor(
            ptr + 32,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        second_bot_view = cute.make_tensor(
            ptr + 32 + half_lane_off,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )

        first_top = cute.make_rmem_tensor((16, ),
                                          _TmemTranspose16x32Core._io_dtype)
        first_bot = cute.make_rmem_tensor((16, ),
                                          _TmemTranspose16x32Core._io_dtype)
        second_top = cute.make_rmem_tensor((16, ),
                                           _TmemTranspose16x32Core._io_dtype)
        second_bot = cute.make_rmem_tensor((16, ),
                                           _TmemTranspose16x32Core._io_dtype)

        cute.copy(
            atom_ld16x64,
            first_top_view,
            _TmemTranspose16x32Core._rmem_copy_view(first_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            first_bot_view,
            _TmemTranspose16x32Core._rmem_copy_view(first_bot, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_top_view,
            _TmemTranspose16x32Core._rmem_copy_view(second_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_bot_view,
            _TmemTranspose16x32Core._rmem_copy_view(second_bot, 16),
        )

        return (first_top, first_bot, second_top, second_bot)

    def __init__(
        self,
        tmem_ptr,
        region: int,
        reg_tensor: Optional[TensorWithContract] = None,
    ) -> None:
        half_lane_off = 16 * self._TmemRowStride
        if region == Region.Top:
            src_ptr = tmem_ptr
            dst_ptr = tmem_ptr
        elif region == Region.Bottom:
            src_ptr = tmem_ptr + half_lane_off
            dst_ptr = tmem_ptr + 16
        else:
            raise ValueError("region must be Region.Top or Region.Bottom")

        self.region = region

        self._tmem_src_full = cute.make_tensor(src_ptr,
                                               self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(dst_ptr,
                                               self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(dst_ptr,
                                              self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(dst_ptr + half_lane_off,
                                              self._tmem_layout(16, 16))

        self._atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            self._io_dtype,
        )
        self._atom_st16x128 = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8),
            self._io_dtype,
        )
        self._atom_st32x32 = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16),
            self._io_dtype,
        )
        self._atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2),
            self._io_dtype,
        )
        self._atom_ld16x128 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition.x4),
            self._io_dtype,
        )

        self._src_regs = cute.make_rmem_tensor((16, ), self._io_dtype)
        output_tensor = cute.make_rmem_tensor((16, ), self._io_dtype)
        self.output = TensorWithContract(
            tensor=output_tensor,
            contract=self.OutputContract,
        )

        self._reg_tensor = reg_tensor
        if reg_tensor is not None:
            assert_contract_equivalent(
                reg_tensor.contract,
                self.InputContract,
                context=f"{type(self).__name__} skip-R1.Load reg_tensor",
            )
            for r in range(16):
                self._src_regs[r] = reg_tensor.tensor[r]

    # -- R1 ------------------------------------------------------------------

    def r1_load(self) -> None:
        """LDTM src region -> ``_src_regs``.  No-op in skip-R1.Load mode."""
        if self._reg_tensor is not None:
            return
        cute.copy(
            self._atom_ld16x64,
            self._tmem_src_full,
            self._rmem_copy_view(self._src_regs, 16),
        )

    def r1_perm(self) -> None:
        for r in range(16):
            self.output.tensor[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(
            self._atom_st16x128,
            self._rmem_copy_view(self.output.tensor, 16),
            self._tmem_src_full,
        )

    # -- R2 ------------------------------------------------------------------

    def r2_load(self) -> None:
        cute.copy(
            self._atom_ld16x64,
            self._tmem_src_full,
            self._rmem_copy_view(self._src_regs, 16),
        )

    def r2_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self._src_regs, 16),
            self._tmem_dst_full,
        )

    # -- R3 ------------------------------------------------------------------

    def r3_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r3_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r3_perm(self) -> None:
        for r in range(16):
            self.output.tensor[r] = self._src_regs[self._PermR3[r]]

    def r3_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output.tensor, 16),
            self._tmem_dst_full,
        )

    # -- R4 ------------------------------------------------------------------

    def r4_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r4_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r4_perm(self) -> None:
        for r in range(16):
            self.output.tensor[r] = self._src_regs[self._PermR4[r]]

    def r4_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output.tensor, 16),
            self._tmem_dst_full,
        )


class TmemTranspose16x32(_TmemTranspose16x32Core):
    """fc1 epi 16x32 -> 32x16 TMEM in-place transpose.

    Contract summary:
      - input : ``token_idx = elem_idx * 2 + ((lane_idx // 2) % 2)``
      - output: ``token_idx = lane_idx``
    The second codomain axis is ``intermediate_output_idx``.
    """

    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "intermediate_output_idx"), (32, 16))

    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": elem_idx * 2 + ((lane_idx // 2) % 2),
                "intermediate_output_idx": (lane_idx % 2) * 8 + lane_idx // 4,
            }),
    )
    OutputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(lambda lane_idx, elem_idx: {
            "token_idx": lane_idx,
            "intermediate_output_idx": elem_idx,
        }),
    )


class TmemTranspose16x32Packed(_TmemTranspose16x32Core):
    """fc2 epi 16x32 -> 32x16 TMEM in-place transpose, 32-bit packed
    bf16x2 elements.

    Same physical atom sequence as ``TmemTranspose16x32``; codomain is
    ``(token_idx, hidden_pair_idx)`` and each slot holds one packed bf16x2.
    """

    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "hidden_pair_idx"), (32, 16))

    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": elem_idx * 2 + ((lane_idx // 2) % 2),
                "hidden_pair_idx": (lane_idx % 2) * 8 + lane_idx // 4,
            }),
    )
    OutputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(lambda lane_idx, elem_idx: {
            "token_idx": lane_idx,
            "hidden_pair_idx": elem_idx,
        }),
    )


# =============================================================================
# TmemTranspose32x32Inplace
# =============================================================================


class TmemTranspose32x32Inplace:
    """fc1 epi 32x32 in-place TMEM transpose: two ``TmemTranspose16x32``
    sub-instances (``top`` = lanes 0..15, ``bot`` = lanes 16..31).

    Optional ``reg_tensor_top`` / ``reg_tensor_bot`` enable skip-R1.Load mode
    for both halves; they must be provided or omitted together.
    """

    def __init__(
        self,
        tmem_ptr,
        reg_tensor_top: Optional[TensorWithContract] = None,
        reg_tensor_bot: Optional[TensorWithContract] = None,
    ) -> None:
        if (reg_tensor_top is None) != (reg_tensor_bot is None):
            raise ValueError(
                "TmemTranspose32x32Inplace: reg_tensor_top and reg_tensor_bot "
                "must be provided or omitted together (both halves either "
                "skip-R1.Load or do R1.Load).")
        self.top = TmemTranspose16x32(tmem_ptr,
                                      Region.Top,
                                      reg_tensor=reg_tensor_top)
        self.bot = TmemTranspose16x32(tmem_ptr,
                                      Region.Bottom,
                                      reg_tensor=reg_tensor_bot)


# =============================================================================
# SwigluCompute
# =============================================================================


class SwigluCompute:
    """Element-wise SwiGLU fold over a configurable reg range
    (packed_f32x2 path).

    SwigluCompute does NOT have a fixed ``InputContract``: the caller
    determines the input distribution by what it hands in.  At
    construction time we validate that ``gate.contract`` and ``up.contract``
    are equal -- the fold is element-wise, so they must share the same
    physical (lane_idx, elem_idx) -> (logical) distribution; only the
    semantic label differs (gate slice vs up slice).

    ``self.output`` inherits the input contract: the fold is element-wise,
    so the output has the same (lane_idx, elem_idx) -> physical mapping
    as the input.  Only the codomain semantic label changes (intermediate
    input slot -> intermediate output slot); since both labels are
    logically the same axis, the contract object is reused as-is.

    ``fold(start, end)`` writes ``self.output.tensor[start:end]`` only.
    The caller may invoke ``fold`` with disjoint ranges to disperse SwiGLU's
    MUFU traffic across surrounding transpose STTM boundaries.
    """

    _Log2E = 1.4426950408889634

    def __init__(
        self,
        gate: TensorWithContract,
        up: TensorWithContract,
        alpha,
    ) -> None:
        assert_contract_equivalent(
            gate.contract,
            up.contract,
            context="SwigluCompute gate/up contract",
        )

        self._gate = gate.tensor
        self._up = up.tensor
        self._alpha = alpha

        output_tensor = cute.make_rmem_tensor((16, ), cutlass.Float32)
        self.output = TensorWithContract(
            tensor=output_tensor,
            contract=gate.contract,
        )

    def fold(self, start: int = 0, end: int = 16) -> None:
        """Fold pairs ``(i, i+1)`` for ``i in range(start, end, 2)``::

            out[i, i+1] = (alpha^2) * up[i, i+1] * gate[i, i+1] *
                          sigmoid(alpha * gate[i, i+1])
            sigmoid(x)  = rcp(1 + exp2(-x * log2(e)))

        Reassociated to put the FMUL2 first so the inner mul collapses to
        one FMUL2 per pair.  ``mul``/``add`` go through packed_f32x2;
        ``exp2``/``rcp_approx`` run scalar but on adjacent pairs ptxas
        back-to-backs them on the MUFU pipe.

        ``start`` / ``end`` must be pair-aligned.
        """
        alpha_f32 = cutlass.Float32(self._alpha)
        neg_alpha_log2e = alpha_f32 * cutlass.Float32(-self._Log2E)
        neg_alpha_log2e_pair = (neg_alpha_log2e, neg_alpha_log2e)
        alpha_sq = alpha_f32 * alpha_f32
        alpha_sq_pair = (alpha_sq, alpha_sq)
        one_pair = (cutlass.Float32(1.0), cutlass.Float32(1.0))

        out = self.output.tensor
        for i in range(start, end, 2):
            ug = cute.arch.mul_packed_f32x2(
                (self._up[i], self._up[i + 1]),
                (self._gate[i], self._gate[i + 1]),
            )

            neg_g_log2e = cute.arch.mul_packed_f32x2(
                (self._gate[i], self._gate[i + 1]), neg_alpha_log2e_pair)
            exp_pair = (
                cute.math.exp2(neg_g_log2e[0], fastmath=True),
                cute.math.exp2(neg_g_log2e[1], fastmath=True),
            )
            one_plus_exp = cute.arch.add_packed_f32x2(exp_pair, one_pair)
            sigmoid_pair = (
                cute.arch.rcp_approx(one_plus_exp[0]),
                cute.arch.rcp_approx(one_plus_exp[1]),
            )

            ug_sig = cute.arch.mul_packed_f32x2(ug, sigmoid_pair)
            out_pair = cute.arch.mul_packed_f32x2(ug_sig, alpha_sq_pair)

            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]


# =============================================================================
# PostSwigluHalf
# =============================================================================


class PostSwigluHalf:
    """Per-half SwiGLU finalize: topk-weight broadcast mul + gen_sf + quantize
    at construction, then ``stg_sfc`` / ``r2s`` as later atomic actions.

    The post-transpose contract gives each thread one token's 16 values, so
    topk broadcast is one scalar multiply across local regs.  See
    ``megamoe_design.md`` for Path-A semantics.
    """

    # InputContract: explicit definition (NOT an alias of
    # ``TmemTranspose16x32.OutputContract``).  Two distinct Contract objects
    # carrying the same mapping function let the construct-time
    # ``assert_contract_equivalent(swiglu.contract, self.InputContract)``
    # check actually exercise the equivalence comparison: if anyone later
    # mutates the upstream OutputContract without updating this one, the
    # mismatch is caught at codegen time.  Aliasing would silently
    # short-circuit the validation.
    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "intermediate_output_idx"), (32, 16))
    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(lambda lane_idx, elem_idx: {
            "token_idx": lane_idx,
            "intermediate_output_idx": elem_idx,
        }),
    )

    _Nvfp4RcpLimit = 1.0 / 6.0  # 1 / max abs of Float4E2M1FN (= 6.0)
    _Fp32Max = 3.40282346638528859812e38

    def __init__(
        self,
        swiglu: TensorWithContract,
        *,
        sC: cute.Tensor,
        gSFC: cute.Tensor,
        preloaded_topk_weight: cutlass.Float32,
        warp_idx,
        norm_const,
        sf_vec_size: int,
        half_idx: int,
        token_idx,
        thread_in_warp,
        intermediate_downproj_idx,
        intermediate_downproj,
        cga_cluster_tile_intermediate_downproj: int,
    ) -> None:
        assert_contract_equivalent(
            swiglu.contract,
            self.InputContract,
            context="PostSwigluHalf swiglu input",
        )

        self._sC = sC
        self._gSFC = gSFC
        self._warp_idx = warp_idx
        self._sf_vec_size = sf_vec_size

        self._half_idx = half_idx
        self._token_idx = token_idx
        self._thread_in_warp = thread_in_warp
        self._intermediate_downproj_idx = intermediate_downproj_idx
        self._intermediate_downproj = intermediate_downproj
        self._cga_cluster_tile_intermediate_downproj = cga_cluster_tile_intermediate_downproj

        self._sfc_reg, self._scaled_regs = self._gen_sfc_quantize(
            swiglu.tensor, norm_const, preloaded_topk_weight)

    def _gen_sfc_quantize(self, swiglu_rmem: cute.Tensor, norm_const,
                          topk_weight):
        """Compute SFC + pre-quantized scaled fp32 regs in RMEM."""
        sfc_reg = cute.make_rmem_tensor((1, ), cutlass.Float8E4M3FN)
        weighted_regs = cute.make_rmem_tensor((16, ), cutlass.Float32)
        scaled_regs = cute.make_rmem_tensor((16, ), cutlass.Float32)

        # Path A: multiply topk before NVFP4 quantize.
        topk_pair = (topk_weight, topk_weight)
        for i in range(0, 16, 2):
            w0, w1 = cute.arch.mul_packed_f32x2(
                (cutlass.Float32(
                    swiglu_rmem[i]), cutlass.Float32(swiglu_rmem[i + 1])),
                topk_pair,
            )
            weighted_regs[i] = w0
            weighted_regs[i + 1] = w1

        # Step 1: absmax over the weighted regs.
        absmax = cutlass.Float32(0.0)
        for i in range(16):
            v = weighted_regs[i]
            abs_v = cute.arch.fmax(v, -v)
            absmax = cute.arch.fmax(absmax, abs_v)

        sfc_fp32 = absmax * cutlass.Float32(self._Nvfp4RcpLimit) * norm_const
        sfc_reg[0] = sfc_fp32.to(cutlass.Float8E4M3FN)
        sfc_fp32_rt = cutlass.Float32(sfc_reg[0])

        acc_scale = norm_const * cute.arch.rcp_approx(sfc_fp32_rt)
        acc_scale = cute.arch.fmin(acc_scale, cutlass.Float32(self._Fp32Max))
        mask = cute.arch.fmin(sfc_fp32_rt * cutlass.Float32(1e30),
                              cutlass.Float32(1.0))
        acc_scale = acc_scale * mask

        acc_scale_pair = (acc_scale, acc_scale)
        for i in range(0, 16, 2):
            s0, s1 = cute.arch.mul_packed_f32x2(
                (weighted_regs[i], weighted_regs[i + 1]),
                acc_scale_pair,
            )
            scaled_regs[i] = s0
            scaled_regs[i + 1] = s1

        return sfc_reg, scaled_regs

    @cute.jit
    def stg_sfc(self) -> None:
        """RMEM -> GMEM: 1 fp8 SFC byte for this token/SF block.

        Skip when the warp's intermediate_downproj position is past the
        valid bound; corresponding fp4 is TMA-OOB-fill-0 on fc2 data leg.

        ``self._intermediate_downproj`` is one of three flavors:

          * Python int that is a multiple of
            ``_cga_cluster_tile_intermediate_downproj``: the predicate is
            statically True; const_expr collapses the entire branch into
            an unconditional STG (no runtime cmp / branch).

          * Python int that is NOT statically aligned (static_expert_shape
            path with e.g. ``intermediate_downproj == 96`` and
            ``cga_cluster_tile_intermediate_downproj == 64``): the
            second branch runs a runtime predicate against a const SSA
            (the Python int folds to an immediate cmp).

          * Int32 SSA (static_expert_shape is None, i.e. dynamic-shape
            mode): the second branch runs a full runtime cmp.  The
            ``isinstance`` short-circuit prevents the first branch from
            trying to ``%`` an SSA value at trace time (which would
            either raise or, worse, silently fall back as it did under
            the old ``-1`` sentinel).
        """
        if cutlass.const_expr(
                isinstance(self._intermediate_downproj, int)
                and self._intermediate_downproj %
                self._cga_cluster_tile_intermediate_downproj == 0):
            self._gSFC[self._token_idx, self._intermediate_downproj_idx,
                       0] = self._sfc_reg[0]
            return

        # Runtime predicate.  Works for both the unaligned-static and the
        # dynamic-shape paths; subtile size assumption (1 fp8 / warp /
        # subtile) is unchanged from the previous implementation.
        if self._intermediate_downproj_idx < self._intermediate_downproj:
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    cutlass.Uint8,
                                    num_bits_per_copy=8),
                self._sfc_reg,
                cute.composition(
                    self._gSFC[self._token_idx, self._intermediate_downproj_idx,
                               None],
                    self._sfc_reg.shape,
                ),
            )

    def r2s(self, subtile_idx) -> None:
        """RMEM -> SMEM: per-thread STS.64 of 16 fp4 to ``sC[subtile_idx]``."""
        fp4_regs = cute.make_rmem_tensor((16, ), cutlass.Float4E2M1FN)
        fp4_vec = self._scaled_regs.load().to(cutlass.Float4E2M1FN)
        fp4_regs.store(fp4_vec)

        sC_stage = cute.slice_(self._sC, (None, None, subtile_idx))
        token_coord = self._thread_in_warp + 32 * self._half_idx
        sC_thread_row = cute.local_tile(
            sC_stage,
            (1, 16),
            (token_coord, self._warp_idx),
        )

        copy_atom_64b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float4E2M1FN,
            num_bits_per_copy=64,
        )
        cute.copy(
            copy_atom_64b,
            cute.coalesce(fp4_regs),
            cute.coalesce(sC_thread_row),
        )


# =============================================================================
# Fc2AccLoadAndPack
# =============================================================================


class Fc2AccLoadAndPack:
    """fc2 epi: LDTM x 2 + cvt.rn.bf16x2.f32 fuse + pair packing.

    Output is a ``TensorWithContract`` matching
    ``TmemTranspose16x32Packed.InputContract``; each 32-bit slot stores one
    bf16x2 pair ``(hidden_i, hidden_i + 16)``.
    """

    # OutputContract: explicit definition (NOT an alias of
    # ``TmemTranspose16x32Packed.InputContract``).  Two distinct Contract
    # objects carrying the same mapping function let the construct-time
    # equivalence check inside ``TmemTranspose16x32Packed.__init__`` (when
    # given ``reg_tensor=self.output``) actually exercise the check: if
    # either side's mapping drifts, the codegen-time assertion catches it.
    # Aliasing would silently short-circuit the validation.
    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "hidden_pair_idx"), (32, 16))
    OutputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": elem_idx * 2 + ((lane_idx // 2) % 2),
                "hidden_pair_idx": (lane_idx % 2) * 8 + lane_idx // 4,
            }),
    )

    _io_dtype = cutlass.Float32  # 32-bit slot dtype for downstream atoms
    _TmemRowStride = _TmemTranspose16x32Core._TmemRowStride

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return _TmemTranspose16x32Core._tmem_layout(num_lanes, num_cols)

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor,
                        num_regs: int,
                        offset: int = 0) -> cute.Tensor:
        return _TmemTranspose16x32Core._rmem_copy_view(rmem, num_regs, offset)

    def __init__(
        self,
        tmem_ptr=None,
        *,
        preload_acc: Optional[Tuple[cute.Tensor, cute.Tensor]] = None,
    ) -> None:
        """Create the packed-bf16x2 RMEM view from TMEM or preloaded acc."""
        if (tmem_ptr is None) == (preload_acc is None):
            raise ValueError(
                "Fc2AccLoadAndPack: exactly one of tmem_ptr / preload_acc "
                "must be provided (LDTM mode vs skip-LDTM preload mode).")

        # Gather top/bottom hidden halves into one 32-reg fp32 vector.
        acc_full = cute.make_rmem_tensor((32, ), self._io_dtype)
        if preload_acc is None:
            # Two LDTMs cover top and bottom hidden halves.
            atom_ld16x64 = cute.make_copy_atom(
                tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
                self._io_dtype,
            )
            tmem_top_view = cute.make_tensor(
                tmem_ptr,
                self._tmem_layout(16, 32),
            )
            tmem_bot_view = cute.make_tensor(
                tmem_ptr + 16 * self._TmemRowStride,
                self._tmem_layout(16, 32),
            )
            cute.copy(
                atom_ld16x64,
                tmem_top_view,
                self._rmem_copy_view(acc_full, 16, offset=0),
            )
            cute.copy(
                atom_ld16x64,
                tmem_bot_view,
                self._rmem_copy_view(acc_full, 16, offset=16),
            )
        else:
            top_reg, bot_reg = preload_acc
            for i in range(16):
                acc_full[i] = top_reg[i]
                acc_full[i + 16] = bot_reg[i]

        # Interleave so bf16x2 pairs become (hidden_i, hidden_i+16).
        reordered_fp32 = cute.make_rmem_tensor((32, ), self._io_dtype)
        for i in range(16):
            reordered_fp32[2 * i] = acc_full[i]
            reordered_fp32[2 * i + 1] = acc_full[i + 16]

        # Bulk cast lets NVVM form cvt.rn.bf16x2.f32 for adjacent pairs.
        packed_bf16 = cute.make_rmem_tensor((32, ), cutlass.BFloat16)
        packed_bf16.store(reordered_fp32.load().to(cutlass.BFloat16))

        # Recast 32 bf16 elements as 16 32-bit slots for downstream atoms.
        packed_fp32 = cute.recast_tensor(packed_bf16, self._io_dtype)

        # Contract matches what the packed transpose expects as input.
        self.output = TensorWithContract(
            tensor=packed_fp32,
            contract=self.OutputContract,
        )


# =============================================================================
# Fc2OutputDest -- MoE-domain fc2 output destination descriptor
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Fc2OutputDest:
    """fc2 output destination, in MoE-domain ``(token_max, topk, hidden)``
    layout.

    Optional ``metadata`` and ``peer_rank_ptr_mapper`` together drive two
    base destination modes (toggled at trace time by their Python-level
    None-ness, so PTX overhead is zero in the direct mode):

    * Both ``None`` -- **direct mode**: ``Fc2UnpackPermuteStg`` writes to
      ``tensor[pool_token_global, 0, hidden_off]`` on the local rank.
      This is the lean fc1+fc2 path with codegen-time ``K=1``; the
      identity metadata ``(src_rank=local_rank, src_token=pool_token_global,
      src_topk=0)`` collapses to a direct ptr arithmetic chain.

    * Both non-``None`` -- **indirect mode**: each lane LDGs
      ``(src_rank, src_token, src_topk)`` from ``metadata[pool_token_global, :]``
      and writes to ``peer(src_rank).tensor[src_token, src_topk, hidden_off]``
      via ``peer_rank_ptr_mapper.map(local_addr, src_rank, 0)`` (one int64 add
      against the in-param-bank peer offset table).  This is MegaMoE
      form A: per-(token, topk) STG to the symmetric-heap combine output
      of the source rank, no atomic, no reduce -- the host reduces the
      topk axis on its side.

    Mixed states (one ``None`` and one not) are rejected in
    ``__post_init__``.  The combined None-ness flow simplifies the
    base-mode state space to {lean, MegaMoE-A} which is exactly what
    callers distinguish in practice.

    Orthogonal mode toggle ``fc2_output_with_redg`` (default ``False``)
    selects the per-fc2-cell store flavour:

    * ``False`` -- **STG.256 mode** (default): every per-(src_token,
      src_topk) destination cell is written by a single warp lane via
      a non-atomic STG.256.  This is the form A contract -- host sums
      the topk axis after the kernel returns.

    * ``True`` -- **REDG mode** (form B): the kernel does the topk axis
      reduce on-device by atomic add (``red.global.add.v2.bf16x2``) into
      ``peer(src_rank).tensor[src_token, 0, hidden_off]``.  The
      ``src_topk`` metadata field is ignored on this path (every topk
      slot of a given ``(src_rank, src_token)`` collapses to the same
      destination row).  Caller contract: ``tensor`` MUST be zero-init
      before the kernel launches (REDG accumulates onto the existing
      cells), and ``tensor.shape[1]`` should be 1 (codegen-time
      collapsed topk axis -- form B doesn't need K storage slots).
      The lean / direct base mode with ``fc2_output_with_redg=True``
      is not currently meaningful (no metadata = no cross-rank /
      topk reduce to do), and is rejected in ``__post_init__``.

    ``tensor`` is the underlying storage tensor, shape
    ``(token_max, topk, hidden)`` BF16/FP16 in row-major; ``token_max``
    is ``pool_token_capacity`` (lean) or ``max_tokens_per_rank`` (MegaMoE).
    Address arithmetic follows ``tensor.layout``'s stride; the consumer
    never hard-codes ``K * H`` or ``H``.

    ``metadata`` is a ``(pool_token_count, 3)`` Uint32 view -- the same
    ``token_src_metadata`` byte buffer the dispatch warps write
    (recast from the underlying ``(pool_token_count, 12)`` Uint8 storage).
    Field order: ``[src_rank, src_token, src_topk]`` per pool token row.

    ``peer_rank_ptr_mapper`` is a ``SymBuffer{world_size}`` ``@native_struct``
    instance carrying the per-peer byte deltas in the kernel parameter
    bank (CUDA byval ABI; see ``src/sym_buffer.py``).  Cross-rank
    redirect uses ``peer_rank_ptr_mapper.map(local_addr, src_rank, byte_off)``;
    by NVSHMEM convention ``peer_rank_ptr_mapper`` returns a zero delta when
    ``src_rank == local_rank``, so a single-rank degenerate MegaMoE run
    is a legal indirect mode (the redirect folds to a no-op at runtime).
    Untyped here because ``SymBuffer{N}`` is parametric on
    ``num_max_ranks``; duck-typed via ``.map(...)``.
    """

    tensor: cute.Tensor
    metadata: Optional[cute.Tensor] = None
    peer_rank_ptr_mapper: Any = None
    fc2_output_with_redg: bool = False

    def __post_init__(self) -> None:
        # Both fields must be either set or both None; mixed states are
        # rejected so callers cannot accidentally request cross-rank
        # redirect without metadata, or vice versa.
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputDest: ``metadata`` and ``peer_rank_ptr_mapper`` must be "
                "both None (lean / direct mode) or both non-None (MegaMoE / "
                "indirect mode).  Got metadata="
                f"{'set' if self.metadata is not None else 'None'}, "
                f"peer_rank_ptr_mapper={'set' if self.peer_rank_ptr_mapper is not None else 'None'}."
            )
        # REDG mode requires metadata + peer_rank_ptr_mapper (it is a MegaMoE-only
        # form B specialisation; the lean / direct path has no topk axis
        # to reduce and no cross-rank visibility issue, so a redg toggle
        # without metadata makes no sense).
        if self.fc2_output_with_redg and self.metadata is None:
            raise ValueError(
                "Fc2OutputDest: fc2_output_with_redg=True is only valid in "
                "indirect mode (metadata + peer_rank_ptr_mapper both set).  The "
                "lean / direct path has no topk axis to reduce.")

    @cute.jit
    def resolve_token_row(self, pool_token_global) -> cute.Tensor:
        """Return the per-lane ``(hidden,)`` BF16 GMEM row view this dest
        wants the lane's STG / REDG to land on.

        Direct mode (``metadata is None``): the row is simply
        ``tensor[pool_token_global, 0, :]`` on the local rank -- one
        ``cute.slice_`` does the (token, topk, hidden) layout walk so
        ``K * H``-style multiplication is never hard-coded.

        Indirect mode (both fields set): per-lane LDG
        ``(src_rank, src_token, src_topk)`` from
        ``metadata[pool_token_global, :]`` (the dispatch warps' three-u32
        token-source record), build the local-rank row view at the new
        ``(src_token, src_topk)`` coordinates, then rebase the row's GMEM
        pointer through ``peer_rank_ptr_mapper.map(local_addr, src_rank, 0)``
        (NVSHMEM symmetric-heap byte delta sitting in the kernel
        parameter bank).  The result is a ``cute.Tensor`` whose iterator
        already points to the destination peer rank's combine slot, so
        downstream STG / REDG sees a normal cute tensor with no awareness
        of the cross-rank mechanics.

        REDG mode (``fc2_output_with_redg=True``): the ``src_topk`` LDG
        is elided and the slice uses ``(src_token, 0, None)``; every
        topk slot for a given ``(src_rank, src_token)`` therefore lands
        on the same destination row and the in-kernel REDG accumulates
        the topk axis on device.

        The const_expr branch keeps the lean path metadata-LDG- and
        peer_rank_ptr_mapper-free in PTX; the metadata LDG (when present) is
        issued at the call site so callers can place it early to overlap
        with downstream RMEM / TMEM work before the STG / REDG.
        """
        if cutlass.const_expr(self.metadata is None):
            return cute.slice_(self.tensor, (pool_token_global, 0, None))

        src_rank = cutlass.Int32(self.metadata[pool_token_global, 0])
        src_token = cutlass.Int32(self.metadata[pool_token_global, 1])
        if cutlass.const_expr(self.fc2_output_with_redg):
            local_row = cute.slice_(self.tensor, (src_token, 0, None))
        else:
            src_topk = cutlass.Int32(self.metadata[pool_token_global, 2])
            local_row = cute.slice_(self.tensor, (src_token, src_topk, None))
        # One LDC.U64 against the in-param-bank offsets table + one int64
        # add; preserves dtype/memspace/alignment of the local iterator.
        peer_iter = self.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator,
            src_rank,
        )
        return cute.make_tensor(peer_iter, local_row.layout)


# =============================================================================
# Fc2UnpackPermuteStg
# =============================================================================


class Fc2UnpackPermuteStg:
    """fc2 epi RMEM -> GMEM dispatcher: STG.256 (default) or topk-collapsing
    REDG (form B), const_expr-switched on
    ``fc2_output_dest.fc2_output_with_redg``.

    Input contract maps ``lane_idx`` to token and ``elem_idx`` to hidden
    pair (one packed bf16x2 = 2 bf16 per 32-bit slot).  This is the
    output contract of ``TmemTranspose16x32Packed`` -- the per-half fc2
    transpose hands the packed regs to this class verbatim.

    The destination is described by a ``Fc2OutputDest`` value carrying:
      * ``metadata`` + ``peer_rank_ptr_mapper`` -- both None (lean / direct) or
        both set (MegaMoE indirect).  Decides whether the lane's GMEM
        row is ``tensor[pool_token_global, 0, :]`` (direct) or
        ``peer(src_rank).tensor[src_token, src_topk, :]`` (indirect,
        ``peer_rank_ptr_mapper.map`` redirect).
      * ``fc2_output_with_redg`` -- chooses between two write flavours:

        - **STG.256 mode (default)**: 2 x STG.256 (32 B each = 16 bf16)
          per thread; each warp lane lands its 32 hidden in 64 B onto a
          unique destination row.  No cross-thread coalescing across
          lanes (each lane targets a distinct token).

        - **REDG mode**: STTM-then-2xLDTM-then-8xREDG.v2.bf16x2 per
          thread (8 B atomic-add per call).  The STTM+LDTM shuffle
          (see below) puts 4 consecutive lanes on the SAME token row's
          contiguous 32 B segment, so the warp-wide
          ``red.global.add.v2.bf16x2`` traffic naturally coalesces
          into sector-aligned 32 B atomic transactions.  Used by MegaMoE form
          B to fold the topk axis on device (every (src_rank, src_token)
          row receives atomic adds from every topk slot on every peer).
          Requires ``fc2_output_with_redg=True`` on the dest descriptor;
          the dest contract additionally requires the underlying tensor
          to be zero-init before the kernel launches (the host / runner
          owns this responsibility) and shape ``(token_max, 1, hidden)``
          (the topk axis collapses to 1 -- ``src_topk`` is ignored when
          resolving the destination row).

    REDG-path register reshuffle -- post-LDTM contract
    ===================================================

    Source TMEM: after STTM ``St32x32b(Repetition.x16)``, this warp's
    32 token rows x 32 hidden cells (packed as 32 lanes x 16
    bf16x2 cols) sit in a contiguous 32-lane x 16-col TMEM slab.

    The slab is then read back through TWO calls of
    ``Ld16x256b(Repetition.x2)``:

      iter 0: source = (TMEM lanes  0..15, cols 0..15)  -- first 16 tokens
      iter 1: source = (TMEM lanes 16..31, cols 0..15)  -- second 16 tokens

    For each call (Rep.x2 = the 4-reg image-1 16dp pattern in cols 0..7
    followed by the same pattern in cols 8..15, yielding 8 reg/thread),
    the per-thread register layout is:

      tmem_lane_in_ldtm = (lane_idx // 4) + 8 * ((elem_idx // 2) % 2)
      tmem_col          = (lane_idx %  4) * 2 + (elem_idx % 2)
                                              + 8 * (elem_idx // 4)

    -- captured as ``_RedgPerLdtmContract`` below for grep / future
    drift detection.  Concretely, the 8 regs split into FOUR adjacent
    8 B pairs, each pair = 2 contiguous bf16x2 cells along the hidden
    axis of a single token row:

      pair 0 = (Rx+0, Rx+1) -> row (lane//4),     cols (lane%4)*2 .. +1
      pair 1 = (Rx+2, Rx+3) -> row (lane//4)+8,   cols (lane%4)*2 .. +1
      pair 2 = (Rx+4, Rx+5) -> row (lane//4),     cols (lane%4)*2+8 .. +9
      pair 3 = (Rx+6, Rx+7) -> row (lane//4)+8,   cols (lane%4)*2+8 .. +9

    Coalescing invariant (the whole reason we do this shuffle):
    for each fixed (ldtm_iter, pair_idx) the four lanes {T_{4k+0..3}}
    sweep the SAME token row's 8 cells (= 32 B) with stride (2 cells
    = 8 B) per lane.  When all 4 lanes simultaneously fire
    ``red.global.add.v2.bf16x2`` (8 B each), the memory subsystem
    coalesces into one sector-aligned 32 B atomic transaction.

    Per-thread REDG count per subtile-half: 2 ldtm_iter * 4 reg-pair =
    8 REDG.v2.bf16x2 = 64 B (same byte budget as the STG.256-mode
    path's 2 x STG.256).

    Per-thread metadata LDG count: 4 distinct token rows
    (= ``(lane//4) + {0, 8, 16, 24}``); no cross-thread shuffle, every
    thread LDGs its own copy (4 lanes per token row redundantly read
    the same 3 uint32s -- acceptable now, would be the first thing to
    optimise if metadata LDG ever shows up as a profile bottleneck).
    """

    # Direct-path input contract -- shared by both modes, both consume
    # ``TmemTranspose16x32Packed.OutputContract``-shaped RMEM.
    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "hidden_pair_idx"), (32, 16))
    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(lambda lane_idx, elem_idx: {
            "token_idx": lane_idx,
            "hidden_pair_idx": elem_idx,
        }),
    )

    # REDG-path per-LDTM RMEM contract: the (lane_idx, elem_idx) ->
    # (TMEM lane within the LDTM source slab, TMEM col within the slab)
    # distribution that ONE ``Ld16x256b(Repetition.x2)`` call produces
    # off a 16-lane x 16-col TMEM slab.  See the class docstring for
    # the pair semantics + coalescing invariant; this Contract object
    # exists so any future drift in the LDTM atom's register layout is
    # caught at codegen time by an explicit equivalence assertion (and
    # so future readers can grep for the mapping without diving into
    # the LDTM atom internals).
    _redg_per_ldtm_domain = Space(("lane_idx", "elem_idx"), (32, 8))
    _redg_per_ldtm_codomain = Space(("tmem_lane_in_ldtm", "tmem_col"), (16, 16))
    _RedgPerLdtmContract = Contract(
        domain=_redg_per_ldtm_domain,
        codomain=_redg_per_ldtm_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "tmem_lane_in_ldtm": (lane_idx // 4) + 8 * (
                    (elem_idx // 2) % 2),
                "tmem_col": (lane_idx % 4) * 2 + (elem_idx % 2) + 8 *
                (elem_idx // 4),
            }),
    )

    # Per-warp hidden tile width (warp w handles hidden [w*32, w*32+32)).
    _HiddenPerWarp = 32
    # Bits per STG.256 store.
    _StgBitsPerCopy = 256
    # REDG-path layout constants.
    _RedgLdtmIterCount = 2  # 2 x Ld16x256b(Repetition.x2)
    _RedgRegPerLdtm = 8  # reg/thread per LDTM call
    _RedgRegPairsPerLdtm = 4  # = _RedgRegPerLdtm / 2
    _RedgBf16PerPair = 4  # 8 B pair = 2 bf16x2 = 4 bf16

    def __init__(
        self,
        packed: TensorWithContract,
        *,
        fc2_output_dest: Fc2OutputDest,
        subtile_pool_token_base,
        warp_idx,
        half_idx: int,
        lane_idx,
        valid_hidden,
        tile_hidden_idx,
        hidden_tile_size: int,
        needs_hidden_predicate: bool,
        valid_token_row_end,
        # REDG-only: the (32 lanes, 64 cols) TMEM view onto this warp's
        # acc subtile region (i.e. the ``tmem_subtile_tensor`` the caller
        # built via ``_subtile_local_tmem_tensor``).  We carve a
        # 32-lane x 16-col STTM+LDTM reshuffle slab out of it based on
        # ``half_idx`` (cols 0..15 for half 0, cols 32..47 for half 1 --
        # matching the per-half ``tmem_first_ptr`` / ``tmem_second_ptr``
        # split inside ``_run_fc2_subtile``).  The slab is overwritten
        # by this class's STTM and consumed by its two LDTM reads, both
        # inside the same task tile body -- the next mma (and any acc
        # release) only fires after this body returns, so the reuse is
        # race-free (see ``_run_fc2_subtile`` comments).  Required iff
        # ``fc2_output_dest.fc2_output_with_redg=True``; rejected if
        # that flag is False AND a non-None value is passed (to catch
        # confused call-sites).
        tmem_subtile_scratch: Optional[cute.Tensor] = None,
    ) -> None:
        assert_contract_equivalent(
            packed.contract,
            self.InputContract,
            context="Fc2UnpackPermuteStg packed input",
        )
        if cutlass.const_expr(fc2_output_dest.fc2_output_with_redg):
            if tmem_subtile_scratch is None:
                raise ValueError(
                    "Fc2UnpackPermuteStg: tmem_subtile_scratch is required "
                    "when fc2_output_dest.fc2_output_with_redg=True "
                    "(needs the (32, 64) subtile TMEM view to carve a "
                    "STTM+LDTM reshuffle slab out of).")
        else:
            if tmem_subtile_scratch is not None:
                raise ValueError(
                    "Fc2UnpackPermuteStg: tmem_subtile_scratch must be None "
                    "in STG.256 mode (fc2_output_with_redg=False).  Got "
                    "a non-None scratch; caller probably passed the wrong "
                    "dest descriptor.")

        self._dest = fc2_output_dest
        # ``subtile_pool_token_base`` = pool token offset of this subtile's
        # token-row 0 within the rank's pool (= current expert start
        # ``cumulative_data_physical_row`` + ``tile_n_idx * cta_tile_n`` +
        # ``subtile_idx * EpilogueTokenTile``).  In STG.256 mode adding
        # ``lane_idx + 32 * half_idx`` 0..63 gives the lane's pool-token
        # row global index (lane <-> token 1:1).  In REDG mode each lane
        # carries 4 distinct tokens recovered as ``32 * half_idx +
        # (lane_idx // 4) + {0, 8, 16, 24}``.  Caller-supplied so this
        # class stays oblivious to subtile / task-tile geometry.
        self._subtile_pool_token_base = subtile_pool_token_base
        self._warp_idx = warp_idx
        self._half_idx = half_idx
        self._lane_idx = lane_idx
        self._valid_hidden = valid_hidden
        self._tile_hidden_base = hidden_tile_size * tile_hidden_idx
        self._needs_hidden_predicate = needs_hidden_predicate

        # Lane-global token row index + valid upper bound for the
        # task-tile-wide token-row predicate.  ``valid_token_row_end`` =
        # ``cumulative_data_physical_row + tile_n_idx * cta_tile_n +
        # valid_tokens_in_tile`` -- the absolute end (exclusive) of the
        # current task tile's valid token-row range (caller-computed).
        # ``token_row_global >= valid_token_row_end`` marks a per-expert
        # padding row whose metadata is uninitialised
        # (``src_token`` defaults to 0); ``stg()`` MUST gate every STG /
        # REDG on this predicate because the destination is a user-
        # supplied tensor and we have no out-of-bounds safety net.
        # Predicate is ALWAYS on (no const_expr short-circuit):
        # ``valid_tokens_in_tile`` is a dispatch-time runtime value that
        # can be strictly less than ``cta_tile_n`` on the last task tile
        # of any expert, and a single padding write would corrupt
        # ``combine_output[0]`` (or any row whose ``src_token`` slot is
        # 0 / stale).
        self._valid_token_row_end = valid_token_row_end

        if cutlass.const_expr(fc2_output_dest.fc2_output_with_redg):
            self._init_redg(packed, tmem_subtile_scratch)
        else:
            self._init_direct(packed)

    # ------------------------------------------------------------------
    # STG.256 mode -- original direct path
    # ------------------------------------------------------------------

    def _init_direct(self, packed: TensorWithContract) -> None:
        """Direct-path setup: unpack + permute packed bf16x2 RMEM back
        to natural hidden order and pre-resolve the destination row.

        Mirrors the legacy ``__init__`` behaviour 1:1 -- the
        non-REDG path is bit-for-bit unchanged from before the REDG
        toggle landed.
        """
        # Direct path: lane <-> token 1:1, one lane-global token row.
        # token-in-subtile = lane_idx + 32 * half_idx (∈ [0, 64) over
        # the 64-token subtile, with half_idx ∈ {0, 1}).
        self._token_row_global = (self._subtile_pool_token_base +
                                  self._lane_idx +
                                  cutlass.Int32(32 * self._half_idx))

        # Resolve this lane's destination row eagerly.  In MegaMoE mode
        # this issues the three-u32 metadata LDG + the int64
        # ``peer_rank_ptr_mapper.map`` add right here at construction time, so the
        # GMEM-load latency
        # overlaps with the caller's subsequent RMEM transpose / TMEM
        # bookkeeping before ``stg()`` actually starts firing STG.256.
        # Lean mode collapses to a single ``cute.slice_`` -- no PTX.
        self._token_row_1d = self._dest.resolve_token_row(
            self._token_row_global)

        # -- Step 2: recast (16, Float32) view back to (32, BFloat16) ------
        packed_bf16 = cute.recast_tensor(packed.tensor, cutlass.BFloat16)

        # -- Step 3: unpack + permute back to natural hidden order ---------
        natural = cute.make_rmem_tensor((32, ), cutlass.BFloat16)
        for h in range(16):
            natural[h] = packed_bf16[2 * h]  # lo lane = hidden h
            natural[h + 16] = packed_bf16[2 * h + 1]  # hi lane = hidden h+16

        # -- Step 4: cache for stg() ---------------------------------------
        self._bf16_regs = natural

    @cute.jit
    def _stg_direct(self) -> None:
        """RMEM -> GMEM: 2 x STG.256 of 16 bf16 each.

        Slices:
          - STG #1: ``natural[0 : 16]``  -> hidden cols ``[w*32, w*32+16)``
          - STG #2: ``natural[16 : 32]`` -> hidden cols ``[w*32+16, w*32+32)``
          (where ``w = warp_idx``).

        The destination ``(hidden,)`` row view was pre-resolved in
        ``_init_direct`` and cached as ``self._token_row_1d``; this
        body just slices it at the per-warp / per-STG hidden offset
        and fires STG.256.  No LDG / no peer arithmetic here.
        """
        copy_atom_256b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=self._StgBitsPerCopy,
        )

        token_row_1d = self._token_row_1d

        # FIXME: every STG.256 below is gated by hand-rolled predicates +
        # hand-computed offsets:
        #   * ``hidden_base`` = ``tile_hidden_base + warp_hidden_base +
        #     stg_idx*16`` indexed into ``token_row_1d``, a hand-resolved
        #     FULL ``(hidden,)`` row from ``Fc2OutputDest.resolve_token_row``
        #     (token / topk resolved, hidden NOT shifted).  This is how the
        #     ``hidden > cta_tile_m`` bug landed -- the base lean path used
        #     a caller-shifted ``_gOut_subtile`` so the stg body only saw
        #     subtile-local offsets, but the MegaMoE rewrite dropped that
        #     shift on the floor.
        #   * ``token_row_global < valid_token_row_end`` is the (always-on)
        #     row predicate that filters out per-expert padding rows whose
        #     destination ``src_token`` metadata is uninitialised.  Without
        #     it last-task-tile padding STGs corrupt ``combine_output[0]``
        #     (or any row whose ``src_token`` slot is 0 / stale).
        # The whole offset + predicate chain should be redone properly:
        # either a ``TensorWithContract`` that pins the per-tile hidden /
        # token / peer axes (and the valid range) by construction, or a
        # tile-level cute tensor the caller hands down already sliced down
        # to the valid (token, hidden) window.  Until then this is glue.
        if self._token_row_global < self._valid_token_row_end:
            warp_hidden_base = self._warp_idx * self._HiddenPerWarp

            for stg_idx in range(2):
                reg_view = cute.make_tensor(
                    self._bf16_regs.iterator + stg_idx * 16,
                    cute.make_layout((((16, ), 1), ), stride=(((1, ), 0), )),
                )

                hidden_base = self._tile_hidden_base + warp_hidden_base + stg_idx * 16

                gOut_thread_row = cute.local_tile(
                    token_row_1d,
                    (16, ),
                    (hidden_base // 16, ),
                )

                aligned_row_iter = cute.make_ptr(
                    gOut_thread_row.element_type,
                    gOut_thread_row.iterator.toint(),
                    AddressSpace.gmem,
                    assumed_align=32,
                )
                gOut_thread_row = cute.make_tensor(aligned_row_iter,
                                                   gOut_thread_row.layout)

                if cutlass.const_expr(self._needs_hidden_predicate):
                    if hidden_base < self._valid_hidden:
                        cute.copy(
                            copy_atom_256b,
                            cute.coalesce(reg_view),
                            cute.coalesce(gOut_thread_row),
                        )
                else:
                    cute.copy(
                        copy_atom_256b,
                        cute.coalesce(reg_view),
                        cute.coalesce(gOut_thread_row),
                    )

    # ------------------------------------------------------------------
    # REDG mode -- STTM + 2x LDTM + 8x REDG.v2.bf16x2
    # ------------------------------------------------------------------

    def _init_redg(
        self,
        packed: TensorWithContract,
        tmem_subtile_scratch: cute.Tensor,
    ) -> None:
        """REDG-path setup: fire STTM(St32x32b.x16) immediately so its
        latency overlaps with the upcoming 4 metadata LDGs + 2 LDTMs.

        Also pre-resolves the 4 destination rows (one per distinct
        token this lane holds across both LDTM iterations) so the
        metadata LDGs are issued early too.  Row resolution + STTM
        are independent (the row resolution reads ``metadata`` GMEM +
        the in-param-bank ``peer_rank_ptr_mapper``; STTM writes TMEM), so the
        two latency chains run in parallel.

        ``tmem_subtile_scratch`` is the caller's (32 lanes, 64 cols)
        ``tmem_subtile_tensor`` view onto this warp's acc TMEM subtile
        region.  We carve a 32-lane x 16-col slab out of it at column
        offset ``32 * half_idx`` -- matching the per-half base used by
        the upstream in-place transpose (``tmem_first_ptr`` at col 0,
        ``tmem_second_ptr`` at col 32).  By the time this body runs,
        the transpose has already finished reading the slab back into
        RMEM (``r4_perm`` was the last consumer), so STTM-overwriting
        the slab is race-free; the next mma cannot reclaim the subtile
        until acc release fires at task-tile boundary, well after both
        per-half ``stg()`` calls have completed.
        """
        # Carve the per-half 32x16 slab.  Col stride in the
        # ``_tmem_layout`` is 1 (see ``_TmemTranspose16x32Core``), so
        # a per-col pointer offset of ``32 * half_idx`` matches the
        # ``tmem_second_ptr = tmem_first_ptr + 32`` arithmetic in
        # ``_run_fc2_subtile``.
        scratch_iter = tmem_subtile_scratch.iterator + 32 * self._half_idx
        # Per-LDTM source views: each is 16 lanes x 16 cols, addressing
        # rows 0..15 (iter 0) and rows 16..31 (iter 1) of the slab.
        # The 16-lane offset in TMEM cell units = 16 *
        # ``_TmemTranspose16x32Core._TmemRowStride``.
        ldtm_half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride
        self._redg_ldtm_src_views = (
            cute.make_tensor(
                scratch_iter,
                _TmemTranspose16x32Core._tmem_layout(16, 16),
            ),
            cute.make_tensor(
                scratch_iter + ldtm_half_lane_off,
                _TmemTranspose16x32Core._tmem_layout(16, 16),
            ),
        )

        # -- Natural-order permute + STTM to TMEM scratch -----------------
        packed_bf16 = cute.recast_tensor(packed.tensor, cutlass.BFloat16)
        natural = cute.make_rmem_tensor((32, ), cutlass.BFloat16)
        for h in range(16):
            natural[h] = packed_bf16[2 * h]
            natural[h + 16] = packed_bf16[2 * h + 1]
        natural_fp32 = cute.recast_tensor(natural, cutlass.Float32)

        sttm_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16),
            cutlass.Float32,
        )
        rmem_view = cute.make_tensor(
            natural_fp32.iterator,
            cute.make_layout((((16, ), 1), ), stride=(((1, ), 0), )),
        )
        sttm_dst_view = cute.make_tensor(
            scratch_iter,
            _TmemTranspose16x32Core._tmem_layout(32, 16),
        )
        cute.copy(sttm_atom, rmem_view, sttm_dst_view)

        # -- Pre-resolve the 4 destination rows ----------------------------
        # Lane t (in half) holds 4 distinct token rows across the two
        # LDTM iterations:
        #
        #   ldtm_iter 0:  token_in_half = (lane % 32 // 4) + {0, 8}
        #   ldtm_iter 1:  token_in_half = (lane % 32 // 4) + {16, 24}
        #
        # The 4 lanes {T_{4k+0..3}} all share the SAME 4 tokens (one
        # for each (iter, pair_parity) slot), so 4 lanes redundantly
        # LDG the same metadata + ``peer_rank_ptr_mapper.map`` redirect; no shuffle,
        # simple but suboptimal (acceptable per design decision).
        #
        # Per-half token base = 32 * half_idx (NOT lane-dependent --
        # ``lane // 4 + {0, 8, 16, 24}`` covers all 32 token slots
        # within the half).
        half_token_base = cutlass.Int32(32 * self._half_idx)
        base_in_half = self._lane_idx // cutlass.Int32(4)
        # Per-lane pool-token-row globals + matching dest-row tensors.
        # Storing them as 4-tuples keeps the addressing arithmetic
        # explicit (one (slot) -> (token_row_global, dest_row) pair,
        # no implicit positional ambiguity).
        token_in_half_offsets = (0, 8, 16, 24)
        self._redg_token_row_globals = tuple(
            self._subtile_pool_token_base + half_token_base + base_in_half +
            cutlass.Int32(off) for off in token_in_half_offsets)
        self._redg_token_rows_1d = tuple(
            self._dest.resolve_token_row(tok_row_global)
            for tok_row_global in self._redg_token_row_globals)

    @cute.jit
    def _stg_redg(self) -> None:
        """TMEM scratch -> RMEM -> GMEM REDG path.

        For each of 2 LDTM iterations, this body:

          1. Calls ``Ld16x256b(Repetition.x2)`` on the per-iter
             16-lane x 16-col TMEM slab.  Each call produces 8
             reg/thread following ``_RedgPerLdtmContract`` (see the
             class docstring for the (lane, elem) -> (tmem_lane, tmem_col)
             mapping and the reg-pair semantics).
          2. For each of 4 reg-pairs, emits 1
             ``red.global.add.v2.bf16x2`` call (= 2 packed bf16x2 = 4
             bf16 = 8 B atomic-add) onto the destination row
             corresponding to that pair's (token, hidden_seg) slot.

        Total per thread per subtile-half: 2 LDTM + 8 REDG.v2.bf16x2
        = 64 B written = same byte budget as the STG.256-mode path.

        Predicates: a single per-token-row valid predicate gates the
        REDGs targeting that row.  Hidden predicate (when the warp's
        hidden segment is past ``valid_hidden``) is checked once per
        pair (every pair = 1 hidden segment of 4 contiguous bf16).
        """
        ldtm_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2),
            cutlass.Float32,
        )

        warp_hidden_base = self._warp_idx * self._HiddenPerWarp

        # Per-pair token-row global + dest-row picker.  Pairs 0/2 share
        # token A = base_in_half + 0/16; pairs 1/3 share token B = +8/+24.
        # Combined into a (iter, pair_idx_in_iter) -> token_slot table
        # so the inner loop can index it directly.
        #
        # pair_idx_in_iter -> token_slot in self._redg_token_rows_1d:
        #   iter 0 pair 0: slot 0  (base + 0)
        #   iter 0 pair 1: slot 1  (base + 8)
        #   iter 0 pair 2: slot 0
        #   iter 0 pair 3: slot 1
        #   iter 1 pair 0: slot 2  (base + 16)
        #   iter 1 pair 1: slot 3  (base + 24)
        #   iter 1 pair 2: slot 2
        #   iter 1 pair 3: slot 3
        pair_to_token_slot = (0, 1, 0, 1)

        # Address-side per-pair hidden offset in bf16 elements:
        #   pair 0/1 -> low  segment, starting at  hidden_pair_col * 2
        #               (= (lane%4) * 2 * 2 = (lane%4) * 4 bf16 from
        #               the warp's hidden base)
        #   pair 2/3 -> high segment, +16 bf16 from low
        # Computed once outside the inner loop because lane-relative.
        lane_quad = self._lane_idx % cutlass.Int32(4)
        # Per-pair low-end hidden offset within the warp's 32-hidden
        # segment, in bf16 element units.
        per_pair_hidden_in_warp_lo = (
            lane_quad * cutlass.Int32(4),  # pair 0
            lane_quad * cutlass.Int32(4),  # pair 1
            lane_quad * cutlass.Int32(4) + cutlass.Int32(16),  # pair 2
            lane_quad * cutlass.Int32(4) + cutlass.Int32(16),  # pair 3
        )

        for ldtm_iter in cutlass.range_constexpr(self._RedgLdtmIterCount):
            regs = cute.make_rmem_tensor(
                (self._RedgRegPerLdtm, ),
                cutlass.Float32,
            )
            cute.copy(
                ldtm_atom,
                self._redg_ldtm_src_views[ldtm_iter],
                cute.make_tensor(
                    regs.iterator,
                    cute.make_layout(
                        (((self._RedgRegPerLdtm, ), 1), ),
                        stride=(((1, ), 0), ),
                    ),
                ),
            )
            # ``regs`` is 8 fp32 slots; each slot's 32-bit bit-pattern
            # is one packed bf16x2 (= 2 bf16 along hidden), inherited
            # from ``Fc2AccLoadAndPack`` upstream.  We pass the fp32
            # ir.Value straight into the bf16x2 REDG inline asm -- PTX
            # constraint ``r`` only cares about the 32-bit register
            # payload, see ``_red_add_relaxed_sys_v2_bf16x2`` docstring.

            for pair_idx in cutlass.range_constexpr(self._RedgRegPairsPerLdtm):
                token_slot = pair_to_token_slot[pair_idx] + (2 if ldtm_iter == 1
                                                             else 0)
                token_row_global = self._redg_token_row_globals[token_slot]
                token_row_1d = self._redg_token_rows_1d[token_slot]
                hidden_in_warp_lo = per_pair_hidden_in_warp_lo[pair_idx]
                hidden_base = (cutlass.Int32(self._tile_hidden_base) +
                               cutlass.Int32(warp_hidden_base) +
                               hidden_in_warp_lo)

                # One v2.bf16x2 REDG = the pair's full 8 B payload (4
                # bf16) in a single PTX op.  4 consecutive lanes hold
                # adjacent (hidden_base + lane_quad * 4 ..
                # + (lane_quad+1)*4) ranges of the same row, which the
                # memory subsystem coalesces into a sector-aligned 32 B
                # atomic transaction (the whole reason we did the
                # STTM+LDTM shuffle).  Pair occupies fp32 regs
                # ``[pair_idx*2, pair_idx*2 + 1]`` -- each fp32 slot is
                # one packed bf16x2 cell along the hidden axis.
                if token_row_global < self._valid_token_row_end:
                    addr = token_row_1d.iterator + hidden_base
                    val0 = cutlass.Float32(regs[pair_idx * 2])
                    val1 = cutlass.Float32(regs[pair_idx * 2 + 1])
                    if cutlass.const_expr(self._needs_hidden_predicate):
                        if hidden_base < self._valid_hidden:
                            _red_add_relaxed_sys_v2_bf16x2(
                                addr,
                                val0,
                                val1,
                            )
                    else:
                        _red_add_relaxed_sys_v2_bf16x2(
                            addr,
                            val0,
                            val1,
                        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    @cute.jit
    def stg(self) -> None:
        """Const_expr dispatch to the STG.256 (default) or REDG path."""
        if cutlass.const_expr(self._dest.fc2_output_with_redg):
            self._stg_redg()
        else:
            self._stg_direct()


# =============================================================================
# SwapABSwigluFp4Epilogue
# =============================================================================


class SwapABSwigluFp4Epilogue:
    """Autonomous epilogue for the swap-AB SwiGLU NVFP4 kernel.

    ``run()`` is the single entry point the kernel calls inside the epi
    warp body.  The kernel's responsibility is reduced to:

      - allocate / free TMEM and build ``acc_tensor``
      - construct the AB / acc pipelines
      - obtain the scheduler consumer

    Everything else (acc consumer state, task-tile loop, overlap rotation,
    early release, TMA store commit / drain, per-subtile dispatch) lives
    inside this class.
    """

    # Per-subtile rotated-leader sync constants.
    _SubtileBarIdBase = 4

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        sf_vec_size: int,
        fc1_output_dtype: Type[cutlass.Numeric],
        fc1_output_layout: utils.LayoutEnum,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        allow_overlap_acc: bool = True,
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        fc2_in_kernel_topk_reduce: bool = False,
    ) -> None:
        # ``fc2_in_kernel_topk_reduce=True`` selects form B: device-side
        # cross-rank ``red.global.add.v2.bf16x2`` reduce over the topk axis
        # (combine output shape ``(token_max, 1, hidden)`` -- the topk
        # axis collapses to a single accumulator slot; the kernel does
        # the topk fold on device via atomic adds).  Form A
        # (``fc2_in_kernel_topk_reduce=False``, default) writes one BF16
        # cell per ``(src_rank, src_token, src_topk)`` slot and the host
        # reduces the K axis after the kernel returns; see
        # ``Fc2OutputDest`` docstring for the full semantic contract.
        #
        # Form B caller contract additions (NOT enforced here -- caller
        # responsibility):
        #   * ``fc2_output`` MUST be zero-init before launch (REDG is
        #     atomic add; non-zero starting cells will accumulate
        #     stale data).
        #   * ``fc2_output.shape[1]`` SHOULD be 1 (the topk axis is
        #     collapsed by the device-side reduce).
        self._fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce

        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self._sf_vec_size = sf_vec_size
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids

        atom_thr_size = 2 if use_2cta_instrs else 1
        self._cta_tile_m = mma_tiler_mnk[0] // atom_thr_size
        self._cta_tile_n = mma_tiler_mnk[1]
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if (static_expert_shape is not None and static_expert_shape[2] %
            (self._cta_tile_m * cluster_shape_mn[0]) == 0):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate: bool = True

        # K-padding gate for fc1 epi SF writes; see PostSwigluHalf.stg_sfc.
        # ``cga_cluster_tile_intermediate_downproj`` is the CGA-level
        # alignment unit on the intermediate_downproj axis; when
        # ``intermediate_downproj`` is an integer multiple of it the
        # predicate is statically True and elided.
        self._cga_cluster_tile_intermediate_downproj: int = (
            self._cta_tile_m // 2) * cluster_shape_mn[0]
        if static_expert_shape is not None:
            intermediate_downproj = static_expert_shape[1] // 2
            if intermediate_downproj % sf_vec_size != 0:
                raise NotImplementedError(
                    f"intermediate_downproj ({intermediate_downproj}) must "
                    f"be a multiple of sf_vec_size ({sf_vec_size}); sub-SF-"
                    f"block K-masking is not implemented.")
            self._intermediate_downproj: Optional[int] = intermediate_downproj
        else:
            self._intermediate_downproj: Optional[int] = None

        self._epi_tile = (EpilogueTokenTile, Fc1EpilogueOutputTile)
        self._subtile_cnt = self._cta_tile_n // EpilogueTokenTile

        self._overlapping_accum = allow_overlap_acc and (
            self._cta_tile_n == EpiWarpCount * EpilogueTokenTile)

        self._num_acc_stage = 2
        self._num_acc_pipeline_stages = 1 if self._overlapping_accum else self._num_acc_stage

        k = self._mma_tiler_k
        self._num_sfa_tmem_cols = self._cta_tile_m * k // sf_vec_size * 4 // 4 // 128
        self._num_sfb_tmem_cols = self._cta_tile_n_sfb * k // sf_vec_size * 4 // 4 // 128
        self._num_sf_tmem_cols = self._num_sfa_tmem_cols + self._num_sfb_tmem_cols

        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage - (
            self._num_sf_tmem_cols if self._overlapping_accum else 0)

        self._iter_acc_early_release = (
            self._num_sf_tmem_cols + EpilogueTokenTile - 1) // EpilogueTokenTile

    # -- Codegen-time queries (read by kernel) --------------------------------

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

    @property
    def overlapping_accum(self) -> bool:
        return self._overlapping_accum

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self._num_acc_pipeline_stages

    @property
    def num_acc_stage(self) -> int:
        return self._num_acc_stage

    @property
    def iter_acc_early_release(self) -> int:
        return self._iter_acc_early_release

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def cta_tile_n(self) -> int:
        return self._cta_tile_n

    @property
    def num_sf_tmem_cols(self) -> int:
        return self._num_sf_tmem_cols

    @property
    def num_sfa_tmem_cols(self) -> int:
        return self._num_sfa_tmem_cols

    @property
    def num_sfb_tmem_cols(self) -> int:
        return self._num_sfb_tmem_cols

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self._num_accumulator_tmem_cols

    # -- sC SMEM layout queries -----------------------------------------------

    def staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self._epi_tile,
            n_stages,
        )

    @property
    def smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self.fc1_output_dtype,
                                  self.smem_layout_one_stage)

    # -- Subtile-local TMEM view helper ---------------------------------------

    def _subtile_local_tmem_tensor(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
        acc_stage_col_offset,
    ) -> cute.Tensor:
        """Build a (32 lanes, 64 cols) cute.Tensor view onto one epi
        subtile's per-warp acc TMEM region.

        Owns the per-warp lane offset, per-stage col offset (overlap-acc
        phase aware), and per-subtile col offset arithmetic.  Returned
        tensor is what ``_run_fc{1,2}_subtile`` and
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` consume.

        ``cute.assume(divby=16)`` is applied here once -- callees can
        derive ``+32`` first/second-half ptrs from the returned tensor's
        iterator without re-asserting alignment (16-aligned base + 32 is
        still 16-aligned).

        ``divby=16`` (instead of 64) so the assume holds even under
        ``overlapping_accum=True`` with phase=1, where
        ``acc_stage_col_offset = phase * (256 - num_sf_tmem_cols) = 208``
        (when ``num_sf_tmem_cols = 48``) and ``208 % 64 = 16``.
        ``divby=16`` still satisfies the downstream alignment check that
        ``cute.assume`` exists to bypass; the codegen optimisation
        difference between ``divby=16`` and ``divby=64`` is negligible
        for this offset arithmetic.
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTokenTile
        total = (warp_lane_off << 16) + acc_stage_col_offset + subtile_col_off
        subtile_ptr = base + cute.assume(total, divby=16)
        return cute.make_tensor(
            subtile_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTokenTile),
        )

    # -- Subtile-level TMA store cmd issue ------------------------------------

    @staticmethod
    @dsl_user_op
    @cute.jit
    def tma_store_fc1_output(
        warp_idx,
        sC,
        subtile_idx,
        tma_atom_fc1_output,
        g_fc1_output_subtile_view: cute.Tensor,
        *,
        loc=None,
        ip=None,
    ) -> None:
        """Per-subtile fence + rotated-leader TMA store cmd issue.

        Subtile-level operation -- it is not per-half, so it lives on the
        epilogue rather than on ``PostSwigluHalf``.  All 4 epi warps call
        this; ``subtile_idx`` doubles as the sC stage index AND drives both
        the leader-warp choice and the NamedBarrier id::

            leader_warp_idx = subtile_idx              (warp s leads sC[s])
            subtile_bar_id  = _SubtileBarIdBase + subtile_idx

        Each subtile owns its own bar id, so producer warps fire-and-forget
        arrive on this bar and race ahead into the next subtile without
        phase-mismatch on a shared bar.  The leader does ``arrive_and_wait``
        and issues the bulk-tensor store; the other 3 warps only ``arrive``.
        No commit / acquire here -- task-tile-boundary commit + drain lives
        inside ``run()``.

        ``@staticmethod @dsl_user_op @cute.jit``: jit is required for the
        ``if warp_idx == leader_warp_idx`` runtime conditional to lower to
        scf.if; making it a free-form static keeps live-locals serialization
        simple (only cute-native types in scope).
        """
        cute.arch.fence_proxy("async.shared", space="cta")
        sC_stage = cute.slice_(sC, (None, None, subtile_idx))
        g_fc1_output_2d = cute.slice_(g_fc1_output_subtile_view,
                                      (None, None, 0))
        bSG_sC, bSG_g_fc1_output = cpasync.tma_partition(
            tma_atom_fc1_output,
            0,
            cute.make_layout(1),
            cute.group_modes(sC_stage, 0, 2),
            cute.group_modes(g_fc1_output_2d, 0, 2),
        )

        # FIXME: previously ``leader_warp_idx = subtile_idx`` -- each
        # subtile was issued by a different epi warp.  This is a race:
        # ``cp.async.bulk.{commit_group,wait_group}`` is per-thread, so
        # the task-tile-boundary commit/drain in ``run()`` only flushed
        # the issuing warp's own in-flight TMA stores; stores issued by
        # the OTHER warps could still be in flight when warp-0 lane-0
        # subsequently fired ``red.release.gpu.add.s32`` on the
        # fc1_done_counter, letting the fc2 phase spin out while fc1
        # TMA writes were unobservable.  The named barrier here is
        # program-order only and does not enforce cross-warp TMA store
        # visibility.  Hard-pin the leader to warp 0 so the task-tile-
        # boundary commit/drain fully captures every fc1 TMA store
        # before the release-add.  Costs serial issue of the 4 subtile
        # TMA stores by warp 0; a perf-conscious rewrite should batch
        # all 4 subtile bulk-tensor copies into a single warp-0 group
        # and keep the dispatch warps independent.
        leader_warp_idx = cutlass.Int32(0)
        subtile_bar_id = subtile_idx + cutlass.Int32(
            SwapABSwigluFp4Epilogue._SubtileBarIdBase)
        subtile_bar = pipeline.NamedBarrier(
            barrier_id=subtile_bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        if warp_idx == leader_warp_idx:
            subtile_bar.arrive_and_wait()
            cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g_fc1_output)
        else:
            subtile_bar.arrive()

    # -- Full task-tile loop --------------------------------------------------

    @cute.jit
    def run(
        self,
        # ── Acc TMEM + acc pipeline (shared, both phases) ────────────────
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        # ── Sched ────────────────────────────────────────────────────────
        sched_consumer,
        sched_ext,
        # ── fc1 (Linear1 phase) outputs ──────────────────────────────────
        # NVFP4 packed output staged through SMEM and dispatched by TMA
        # bulk store.
        smem_fc1_output_buffer: cute.Tensor,  # sC SMEM (subtile_cnt slots)
        tma_atom_fc1_output: cute.CopyAtom,  # TMA store atom for fc1 NVFP4 out
        gmem_fc1_output: cute.Tensor,  # GMEM target for the TMA store
        gmem_fc1_output_sf: cute.
        Tensor,  # GMEM fp8 SFC, written by per-thread STG
        # ── topk weights (Path A: fc1 epi mixes them into swiglu fp32 ─
        # before NVFP4 quantize).  Per-token fp32 scalar GMEM tensor of
        # shape (token_sum,).  Used only by the Linear1 phase; ignored
        # under Linear2.
        gmem_topk_scores: cute.Tensor,
        # ── fc2 (Linear2 phase) output ───────────────────────────────────
        # bf16/fp16 output, MoE-domain layout ``(token_max, topk, hidden)``,
        # hidden stride-1.  Written DIRECTLY by per-thread STG.256 (no TMA,
        # no SMEM staging).  Lean fc1+fc2 path: ``K==1`` codegen-time const,
        # axis 0 indexes pool-token rows.  MegaMoE: ``K==num_topk``, axis 0
        # is per-rank ``max_tokens_per_rank``; per-(src_rank, src_token,
        # src_topk) destination is decoded from ``token_comm_args`` below.
        gmem_fc2_output: cute.Tensor,
        # ── fc1 -> fc2 release-acquire signal ──────────────────────────
        # GMEM int32 counter, 1D shape (max_token_block_per_rank,). Indexed by
        # ``cumulative_token_block_count + tile_n_idx`` as carried by
        # ``SwapABSwigluFp4Fc12WorkTileInfo``.
        gmem_fc1_done_counter: cute.Tensor,
        # ── Per-warp / per-thread ────────────────────────────────────────
        warp_idx: int,
        tidx,
        # ── Epi-side runtime scaling (fc1 only) ─────────────────────────
        alpha,
        norm_const,
        # ── MegaMoE-only routing bundle (Optional) ──────────────────────
        # When None the fc2 STG path runs in direct mode: writes go to
        # ``gmem_fc2_output[pool_token_global, 0, hidden_off]`` on the
        # local rank (the lean fc1+fc2 contract).  When non-None, the
        # subclass routing metadata + in-param-bank ``peer_rank_ptr_mapper`` carried
        # by the bundle drive the per-(token, topk) ``peer_rank_ptr_mapper.map``
        # redirect described in ``Fc2OutputDest``.  Duck-typed: we read
        # only ``.token_src_metadata`` and ``.peer_rank_ptr_mapper`` here.
        token_comm_args=None,
    ) -> None:
        """Run the full fc1+fc2-fused epilogue task-tile loop.

        Per work tile, the body executes (in this exact order):

          1. ``run_subtiles``    : phase-dispatched task-tile body
                                   (``_run_fc1_task_tile`` /
                                   ``_run_fc2_task_tile``); does
                                   ``acc_pipeline.consumer_wait``,
                                   subtile loop, ``consumer_release``.
          2. ``snapshot``        : remember (phase, fc1 counter slot) for
                                   the piggyback atomic add below.
          3. ``advance``         : ``acc_consumer_state.advance()`` and
                                   (under overlap) flip ``is_odd_turn``.
          4. ``consume_sched``   : fetch next work tile (sched-warp wait
                                   overlaps with the TMA drain in 5/6).
          5. ``commit_tma`` +
             ``wait_tma``        : commit the current task tile's bulk
                                   store batch and drain everything in
                                   flight.  Harmless no-op for fc2 task
                                   tiles (which use STG.256, not TMA).
          6. ``boundary_bar``    : 4-epi-warp NamedBarrier sync; ensures
                                   every CTA thread has issued its STG /
                                   TMA before the atomic add fires.
          7. ``red_add``         : ``red.release.gpu.add.s32`` from CTA
                                   thread 0 to ``fc1_done_counter[slot]``,
                                   only if the just-processed work tile
                                   was Linear1 (fc1).  Provides the
                                   release-side of the fc1->fc2 sync
                                   protocol.

        The piggyback ordering (``5..7`` after ``4`` consume_sched) lets
        the TMA drain wait overlap with the sched warp's pipeline ack
        latency on consume_work, and folds the "final flush + atomic
        add for the last fc1 work tile" naturally into the loop body
        (when prev was the last fc1 work tile, ``consume_sched`` returns
        an invalid sentinel; the atomic add still fires here, then the
        while-header sees invalid and exits -- no separate post-loop
        flush needed).
        """
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_acc_pipeline_stages)
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=32 * len(self._epilogue_warp_ids),
        )

        # gmem_fc2_output is MoE-domain (token_max, topk, hidden); valid
        # hidden cols = axis 2 size.
        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[2])

        # In-bound fc1 tile count (= ext_fc2_spin_threshold mirror).
        # gmem_fc1_output is (tokens_sum_padded, intermediate_downproj);
        # tile step (post-swiglu) = cta_tile_m // 2.
        intermediate_downproj_tile_count = (gmem_fc1_output.shape[1] +
                                            (self._cta_tile_m // 2) -
                                            1) // (self._cta_tile_m // 2)

        # Init=1 (= reverse): under overlapping_accum the first task tile
        # walks subtiles N-1, 0, 1, ..., N-2 so the rightmost subtile
        # (containing the staggered overlap region cols) is processed first
        # and its TMEM cols are released to the next phase's mma immediately.
        # Shared across phases -- fc2 inherits the same overlap rotation.
        # Constexpr-elided under non-overlap.
        is_odd_turn = cutlass.Int32(1)

        work_tile_info = sched_consumer.consume_work()
        while work_tile_info.is_valid_tile:
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                self._run_fc1_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    gmem_fc1_output_sf=gmem_fc1_output_sf,
                    gmem_topk_scores=gmem_topk_scores,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    alpha=alpha,
                    norm_const=norm_const,
                )
            else:
                self._run_fc2_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    token_comm_args=token_comm_args,
                )
            iket.range_pop()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(
                BlockPhase.Linear1)
            cur_fc1_counter_slot = (
                work_tile_info.cumulative_token_block_count +
                work_tile_info.tile_n_idx)
            # Considering possible future mix-cga, let only valid tiles
            # signal the buffer ready.  ``tile_m_idx`` in swap_ab is the
            # ``intermediate_downproj_tile_idx``.
            cur_intermediate_downproj_tile_in_bound = (
                work_tile_info.tile_m_idx < intermediate_downproj_tile_count)

            acc_consumer_state.advance()
            if cutlass.const_expr(self._overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn

            work_tile_info = sched_consumer.consume_work()

            # Drain fc1 TMA stores and sf stores before publishing the fc1-done counter.
            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.fence_acq_rel_gpu()
            task_tile_boundary_bar.arrive_and_wait()

            # Publish completion for the work tile snapshotted above.
            if cur_was_linear1 and cur_intermediate_downproj_tile_in_bound:
                if tidx == 0:
                    _red_add_release_gpu_s32(
                        gmem_fc1_done_counter.iterator + cur_fc1_counter_slot,
                        cutlass.Int32(1),
                    )

    # -- Per-phase task-tile dispatch ------------------------------------------

    @cute.jit
    def _run_fc1_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
    ) -> None:
        """Linear1 task-tile body."""
        real_fc1_output, _ = sched_ext.get_gmem_tensor(
            "c",
            gmem_fc1_output,
            work_tile_info,
        )
        real_fc1_output_sf, _ = sched_ext.get_gmem_tensor(
            "sfc",
            gmem_fc1_output_sf,
            work_tile_info,
        )
        # Shifted topk view; downstream indexing is expert-local.
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk",
            gmem_topk_scores,
            work_tile_info,
        )

        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("fc1_epi")

        if cutlass.const_expr(self._overlapping_accum):
            acc_stage_col_offset = cutlass.Int32(
                acc_consumer_state.phase) * (256 - self._num_sf_tmem_cols)
        else:
            acc_stage_col_offset = cutlass.Int32(
                acc_consumer_state.index) * self._cta_tile_n

        valid_tokens = work_tile_info.valid_tokens_in_tile
        subtile_cnt = self._subtile_cnt

        # Overlap path preloads two subtiles before releasing acc TMEM.
        unroll_tile_cnt = 2 if cutlass.const_expr(
            self._overlapping_accum) else 0
        remain_subtile_cnt = subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (cutlass.Int32(subtile_cnt) -
                                 is_odd_turn) % cutlass.Int32(subtile_cnt)
            subtile_idx_second = (cutlass.Int32(subtile_cnt + 1) -
                                  is_odd_turn) % cutlass.Int32(subtile_cnt)

            tmem_subtile_first = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_first,
                warp_idx,
                acc_stage_col_offset,
            )
            tmem_subtile_second = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_second,
                warp_idx,
                acc_stage_col_offset,
            )

            # Always preload before unconditional acc release.
            preload_subtile_first = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_first)

            # Release acc to next MMA unconditionally.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            preload_subtile_second = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_second)

            # Both unrolled subtiles borrow tmem_subtile_second as workspace.
            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(64) < valid_tokens:
                    self._run_fc1_subtile(
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_subtile_second,
                        real_fc1_output=real_fc1_output,
                        real_fc1_output_sf=real_fc1_output_sf,
                        real_topk_scores=real_topk_scores,
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        warp_idx=warp_idx,
                        tidx=tidx,
                        alpha=alpha,
                        norm_const=norm_const,
                        preload_acc=preload_pair[i],
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = (cutlass.Int32(real_i + subtile_cnt) -
                               is_odd_turn) % cutlass.Int32(subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)

            if subtile_idx * cutlass.Int32(64) < valid_tokens:
                self._run_fc1_subtile(
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=self._subtile_local_tmem_tensor(
                        tmem_acc_tensor,
                        subtile_idx,
                        warp_idx,
                        acc_stage_col_offset,
                    ),
                    real_fc1_output=real_fc1_output,
                    real_fc1_output_sf=real_fc1_output_sf,
                    real_topk_scores=real_topk_scores,
                    work_tile_info=work_tile_info,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    alpha=alpha,
                    norm_const=norm_const,
                )

        # Non-overlap-path release: at the natural task-tile boundary.
        # (Overlap path's release fires inside the unroll prologue at ②
        # above, replacing the old "release at i==0 inside loop" pattern.)
        if cutlass.const_expr(not self._overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_fc2_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
    ) -> None:
        """fc2 (Linear2) task-tile body.

        Mirrors ``_run_fc1_task_tile``'s shape -- same acc_pipeline
        wait/release lifecycle, same subtile loop with overlap rotation
        and valid_tokens early-exit -- only the per-subtile body differs
        (``_run_fc2_subtile`` instead of ``_run_fc1_subtile``).

        Path A: fc2 epi takes no topk weight (the per-token scalar was
        already pre-multiplied into the swiglu fp32 output by the upstream
        fc1 ``PostSwigluHalf``).  fc2's data path is just LDTM + cvt + pack
        + transpose + unpack-permute + STG.256, no SMEM staging, no TMA.

        ``gmem_fc2_output`` is the MoE-domain ``(token_max, topk, hidden)``
        tensor; the sched ext's ``"c"`` token-offset shift is intentionally
        bypassed -- per-token offset is recovered from
        ``work_tile_info.cumulative_data_physical_row`` plus the per-tile /
        per-subtile / per-lane arithmetic inside ``_run_fc2_subtile``.
        For MegaMoE form A, the bundle's ``token_src_metadata`` +
        ``peer_rank_ptr_mapper`` redirect each lane's STG to the symmetric-heap
        combine output of the right source rank; see ``Fc2OutputDest``.
        """
        # MoE-domain dest descriptor.  Lean fc1+fc2 path: token_comm_args
        # is None and the dest collapses to direct STG into
        # ``gmem_fc2_output[pool_token_global, 0, :]``.  MegaMoE form A:
        # subclass-supplied metadata + peer_rank_ptr_mapper are forwarded so
        # ``Fc2UnpackPermuteStg`` can sym-map each lane's STG to the right
        # peer rank's combine slot.
        #
        # ``token_comm_args.token_src_metadata`` is the dispatch-side ABI
        # view (``(pool_token_capacity, 12)`` Uint8, byte-stepped); we
        # recast to ``(pool_token_capacity, 3)`` Uint32 here so the
        # per-lane LDG inside ``Fc2OutputDest.resolve_token_row`` indexes
        # the three (src_rank, src_token, src_topk) fields as natural
        # 4-byte elements.  Zero-copy alias -- same byte addresses.
        if cutlass.const_expr(token_comm_args is None):
            # Lean / direct mode: REDG toggle is not meaningful here
            # (no metadata = no topk reduce to fold), so the dest is
            # always STG.256.
            fc2_output_dest = Fc2OutputDest(tensor=gmem_fc2_output)
        else:
            metadata_u32 = cute.recast_tensor(
                token_comm_args.token_src_metadata,
                cutlass.Uint32,
            )
            fc2_output_dest = Fc2OutputDest(
                tensor=gmem_fc2_output,
                metadata=metadata_u32,
                peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                fc2_output_with_redg=self._fc2_in_kernel_topk_reduce,
            )

        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("fc2_epi")

        if cutlass.const_expr(self._overlapping_accum):
            acc_stage_col_offset = cutlass.Int32(
                acc_consumer_state.phase) * (256 - self._num_sf_tmem_cols)
        else:
            acc_stage_col_offset = cutlass.Int32(
                acc_consumer_state.index) * self._cta_tile_n

        valid_tokens = work_tile_info.valid_tokens_in_tile
        subtile_cnt = self._subtile_cnt

        # Same overlap-acc structure as fc1; downstream subtile body differs.
        unroll_tile_cnt = 2 if cutlass.const_expr(
            self._overlapping_accum) else 0
        remain_subtile_cnt = subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (cutlass.Int32(subtile_cnt) -
                                 is_odd_turn) % cutlass.Int32(subtile_cnt)
            subtile_idx_second = (cutlass.Int32(subtile_cnt + 1) -
                                  is_odd_turn) % cutlass.Int32(subtile_cnt)

            tmem_subtile_first = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_first,
                warp_idx,
                acc_stage_col_offset,
            )
            tmem_subtile_second = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_second,
                warp_idx,
                acc_stage_col_offset,
            )

            preload_subtile_first = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_first)

            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            preload_subtile_second = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_second)

            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(64) < valid_tokens:
                    self._run_fc2_subtile(
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_subtile_second,
                        fc2_output_dest=fc2_output_dest,
                        work_tile_info=work_tile_info,
                        valid_hidden=valid_hidden,
                        warp_idx=warp_idx,
                        tidx=tidx,
                        preload_acc=preload_pair[i],
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = (cutlass.Int32(real_i + subtile_cnt) -
                               is_odd_turn) % cutlass.Int32(subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)

            if subtile_idx * cutlass.Int32(64) < valid_tokens:
                self._run_fc2_subtile(
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=self._subtile_local_tmem_tensor(
                        tmem_acc_tensor,
                        subtile_idx,
                        warp_idx,
                        acc_stage_col_offset,
                    ),
                    fc2_output_dest=fc2_output_dest,
                    work_tile_info=work_tile_info,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                )

        if cutlass.const_expr(not self._overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    # -- Per-subtile dispatch -------------------------------------------------

    @cute.jit
    def _run_fc1_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        real_topk_scores: cute.Tensor,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        *,
        preload_acc: Optional[Tuple[cute.Tensor, cute.Tensor, cute.Tensor,
                                    cute.Tensor]] = None,
    ) -> None:
        """Run one fc1 epi subtile with contract-backed component wiring.

        ``tmem_subtile_tensor`` is a (32 lanes, 64 cols) view onto the
        per-warp acc TMEM region (or, in the overlap-acc unroll path,
        onto a *workspace* subtile's region that this subtile is
        borrowing for its in-place transpose STTMs).  Caller-built via
        ``_subtile_local_tmem_tensor``.

        ``preload_acc`` (4-tuple of (16,) fp32 RMEM tensors) is
        non-None **only** in the overlap-acc unroll path -- it carries
        the raw acc data that has already been LDTM'd out by
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` *before* the
        acc TMEM was released to the next mma.  Tuple ordering:

            preload_acc[0]: gate_lo  (subtile cols 0..31, lanes 0..15)
            preload_acc[1]: up_lo    (subtile cols 0..31, lanes 16..31)
            preload_acc[2]: raw_top  (subtile cols 32..63, lanes 0..15)
            preload_acc[3]: raw_bot  (subtile cols 32..63, lanes 16..31)

        When provided, the per-subtile entry LDTM x 2 (gate / up) is
        skipped and ``second_t`` is constructed in skip-R1.Load mode
        with raw_top / raw_bot fed through ``reg_tensor_top`` /
        ``reg_tensor_bot``; the rest of the body (transpose rounds,
        SwiGLU folds, post-half quantize, R2S, TMA cmd) is identical.

        When ``preload_acc is None`` the body matches the original
        sequential-per-subtile path bit-for-bit -- fine-grained ILP
        interleaving of transpose rounds, SwiGLU folds, and post-SwiGLU
        tasks is preserved exactly.
        """
        # -- Per-half TMEM ptrs derived from the (32, 64) subtile view ------
        #
        # ``tmem_subtile_tensor.iterator`` is the (lane 0, col 0) corner
        # of this subtile's 64-col region, already 16-aligned by the
        # ``cute.assume(divby=16)`` inside ``_subtile_local_tmem_tensor``.
        # +32 second-half offset uses a Python int (compile-time const)
        # so cute const-folds it into the ptr and propagates the 16-col
        # alignment to the LDTM/STTM atoms; a ``cutlass.Int32(32)`` here
        # would be an SSA value that cute treats as alignment-unknown,
        # tripping the LDTM atom's "tmem aligned at >= 2 cols" verifier.
        tmem_first_ptr = tmem_subtile_tensor.iterator
        tmem_second_ptr = tmem_first_ptr + 32

        # -- Per-subtile GMEM views and per-half meta -------------------------

        subtile_token_tile_idx = (work_tile_info.tile_n_idx *
                                  (self._cta_tile_n // EpilogueTokenTile) +
                                  subtile_idx)
        g_fc1_output_subtile_view = cute.local_tile(
            real_fc1_output,
            (EpilogueTokenTile, Fc1EpilogueOutputTile, 1),
            (subtile_token_tile_idx, work_tile_info.tile_m_idx, 0),
        )

        thread_in_warp = tidx % 32
        subtile_token_start = (work_tile_info.tile_n_idx * self._cta_tile_n +
                               subtile_idx * EpilogueTokenTile)
        token_left = subtile_token_start + thread_in_warp
        token_right = token_left + 32
        intermediate_downproj_idx = (work_tile_info.tile_m_idx *
                                     (self._cta_tile_m // 2) +
                                     warp_idx * Nvfp4BlockSize)

        # Resolve ``intermediate_downproj`` for the PostSwigluHalf SFC
        # predicate.
        if cutlass.const_expr(self._intermediate_downproj is not None):
            intermediate_downproj_value = self._intermediate_downproj
        else:
            intermediate_downproj_value = real_fc1_output.shape[1]

        # -- Pre-LDG topk weights ---------------------------------------------
        #
        # Hoisted ahead of the LDTM x 2 below so the per-thread topk-weight
        # GMEM round trip overlaps with the entire LDTM + transpose +
        # quantize pipeline.  The PostSwigluHalf instances at the end of
        # this subtile receive these as ``preloaded_topk_weight`` instead
        # of running their own LDG inside __init__ -- saving the LDG +
        # cvt latency from the critical path.
        #
        # Each thread reads exactly 2 fp32: ``token_left`` for the
        # left-half post (covering tokens 0..31 of this subtile) and
        # ``token_right`` for the right-half post (tokens 32..63).
        # ``real_topk_scores`` is the per-expert-shifted view of the
        # global topk_scores 1D tensor (produced by
        # ``SwapABSwigluFp4Fc12SchedExtension.get_gmem_tensor("topk", ...)``).
        # Indexing it with the EXPERT-LOCAL token coord matches the SFC
        # write-side coord convention.
        topk_left = cutlass.Float32(real_topk_scores[token_left])
        topk_right = cutlass.Float32(real_topk_scores[token_right])

        # -- raw f_gate / f_up: either LDTM here or take from preload_acc ----
        #
        # Both paths produce RMEM tensors that, by definition, carry
        # ``TmemTranspose16x32.InputContract`` -- the (lane_idx, elem_idx)
        # -> (token_idx, intermediate_output_idx) distribution that
        # ``Ld16x64bOp(Repetition.x16) Float32`` LDTM produces, and that
        # ``load_subtile_raw_acc`` (using the very same atom) reproduces.
        # Wrap them as TensorWithContract so they cross the boundary into
        # SwigluCompute under the contract-backed handoff rule.
        if cutlass.const_expr(preload_acc is None):
            f_gate_tensor = cute.make_rmem_tensor((16, ), cutlass.Float32)
            f_up_tensor = cute.make_rmem_tensor((16, ), cutlass.Float32)
            atom_ld16x64 = cute.make_copy_atom(
                tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
                cutlass.Float32,
            )
            tmem_gate_view = cute.make_tensor(
                tmem_first_ptr,
                TmemTranspose16x32._tmem_layout(16, 32),
            )
            tmem_up_view = cute.make_tensor(
                tmem_first_ptr + (16 << 16),
                TmemTranspose16x32._tmem_layout(16, 32),
            )
            cute.copy(
                atom_ld16x64,
                tmem_gate_view,
                TmemTranspose16x32._rmem_copy_view(f_gate_tensor, 16),
            )
            cute.copy(
                atom_ld16x64,
                tmem_up_view,
                TmemTranspose16x32._rmem_copy_view(f_up_tensor, 16),
            )
        else:
            f_gate_tensor = preload_acc[0]
            f_up_tensor = preload_acc[1]

        f_gate = TensorWithContract(
            tensor=f_gate_tensor,
            contract=TmemTranspose16x32.InputContract,
        )
        f_up = TensorWithContract(
            tensor=f_up_tensor,
            contract=TmemTranspose16x32.InputContract,
        )

        # -- SwiGLU fold first half (0..8 now; 8..16 dispersed below) --------

        first_swiglu = SwigluCompute(gate=f_gate, up=f_up, alpha=alpha)
        first_swiglu.fold(0, 8)

        # -- Second 32x32 in-place transpose (R4 has no STTM) ----------------
        #
        # preload_acc=None: standard path (R1.Load LDTMs from TMEM).
        # preload_acc!=None: skip-R1.Load mode -- raw_top / raw_bot have
        # already been LDTM'd out by ``load_subtile_raw_acc`` and are
        # being fed in via reg_tensor_{top,bot}.  R1.Store then writes
        # them into the borrowed workspace cols (= ``tmem_second_ptr``,
        # which under the unroll path points to the *workspace* subtile's
        # second-half cols, NOT this subtile's own second-half cols).

        if cutlass.const_expr(preload_acc is None):
            second_t = TmemTranspose32x32Inplace(tmem_second_ptr)
            second_t.bot.r1_load()
            second_t.top.r1_load()
        else:
            second_t = TmemTranspose32x32Inplace(
                tmem_second_ptr,
                reg_tensor_top=TensorWithContract(
                    tensor=preload_acc[2],
                    contract=TmemTranspose16x32.InputContract,
                ),
                reg_tensor_bot=TensorWithContract(
                    tensor=preload_acc[3],
                    contract=TmemTranspose16x32.InputContract,
                ),
            )
            # skip-R1.Load: r1_load is a no-op inside _TmemTranspose16x32Core,
            # so we don't call it here.
        second_t.bot.r1_perm()
        second_t.top.r1_perm()
        second_t.bot.r1_store()
        second_t.top.r1_store()

        second_t.bot.r2_load()
        second_t.top.r2_load()
        second_t.top.r2_store()
        second_t.bot.r2_store()

        second_t.top.r3_load_top()
        second_t.top.r3_load_bot()
        second_t.bot.r3_load_top()
        second_t.bot.r3_load_bot()
        first_swiglu.fold(8, 16)
        second_t.top.r3_perm()
        second_t.bot.r3_perm()
        second_t.top.r3_store()
        second_t.bot.r3_store()

        second_t.top.r4_load_top()
        second_t.top.r4_load_bot()
        second_t.bot.r4_load_top()
        second_t.bot.r4_load_bot()
        second_t.top.r4_perm()
        second_t.bot.r4_perm()

        # -- SwiGLU fold second half (0..8 now; 8..16 dispersed below) ------
        #
        # second_t.top.output / second_t.bot.output carry
        # ``TmemTranspose16x32.OutputContract``; SwigluCompute validates
        # gate/up contracts are equal at construction time.

        second_swiglu = SwigluCompute(
            gate=second_t.top.output,
            up=second_t.bot.output,
            alpha=alpha,
        )
        second_swiglu.fold(0, 8)

        # -- First 16x32 transpose (skip-R1.Load; R4 has no STTM) -----------
        #
        # ``first_swiglu.output.contract == TmemTranspose16x32.InputContract``
        # (SwigluCompute inherits from f_gate's contract, which IS
        # ``InputContract``).  TmemTranspose16x32.__init__ validates the
        # reg_tensor contract against InputContract.

        first_t = TmemTranspose16x32(
            tmem_first_ptr,
            Region.Top,
            reg_tensor=first_swiglu.output,
        )
        first_t.r1_perm()
        first_t.r1_store()
        first_t.r2_load()
        first_t.r2_store()
        first_t.r3_load_top()
        first_t.r3_load_bot()
        second_swiglu.fold(8, 16)
        first_t.r3_perm()
        first_t.r3_store()
        first_t.r4_load_top()
        first_t.r4_load_bot()

        # gen_sf + quantize for the right half (second_swiglu.output is
        # ready; first_t.r4_perm hasn't run yet, so left half waits below).
        # The per-thread topk-weight LDG was hoisted to the subtile prologue
        # (``topk_right`` is now a ready fp32 register); PostSwigluHalf
        # consumes it via ``preloaded_topk_weight`` and pre-multiplies it
        # into the swiglu fp32 values BEFORE NVFP4 quantize (Path A).
        # ``token_right`` is the EXPERT-LOCAL token coord (same coord
        # that indexes ``real_fc1_output_sf`` for the SFC write).
        post_right = PostSwigluHalf(
            second_swiglu.output,
            sC=smem_fc1_output_buffer,
            gSFC=real_fc1_output_sf,
            warp_idx=warp_idx,
            norm_const=norm_const,
            sf_vec_size=Nvfp4BlockSize,
            half_idx=1,
            token_idx=token_right,
            thread_in_warp=thread_in_warp,
            preloaded_topk_weight=topk_right,
            intermediate_downproj_idx=intermediate_downproj_idx,
            intermediate_downproj=intermediate_downproj_value,
            cga_cluster_tile_intermediate_downproj=(
                self._cga_cluster_tile_intermediate_downproj),
        )

        first_t.r4_perm()

        # -- Quant + R2S + STG SFC, right then left half ---------------------
        #
        # No per-subtile-entry barrier wait: each subtile owns its own bar
        # id (``subtile_bar_id``), so producer warps can fire-and-forget
        # arrive on the previous subtile's bar and immediately start the
        # next subtile's R2S without throttle.  Phase correctness is
        # guaranteed by single-arrive-per-warp-per-bar within a task tile.

        post_right.stg_sfc()

        post_left = PostSwigluHalf(
            first_t.output,
            sC=smem_fc1_output_buffer,
            gSFC=real_fc1_output_sf,
            warp_idx=warp_idx,
            norm_const=norm_const,
            sf_vec_size=Nvfp4BlockSize,
            half_idx=0,
            token_idx=token_left,
            thread_in_warp=thread_in_warp,
            preloaded_topk_weight=topk_left,
            intermediate_downproj_idx=intermediate_downproj_idx,
            intermediate_downproj=intermediate_downproj_value,
            cga_cluster_tile_intermediate_downproj=(
                self._cga_cluster_tile_intermediate_downproj),
        )
        post_left.stg_sfc()

        post_right.r2s(subtile_idx)
        post_left.r2s(subtile_idx)

        # -- TMA store C cmd issue (per-subtile; task-tile commit lives in
        #    the run() body, not here) -------------------------------------

        SwapABSwigluFp4Epilogue.tma_store_fc1_output(
            warp_idx,
            smem_fc1_output_buffer,
            subtile_idx,
            tma_atom_fc1_output,
            g_fc1_output_subtile_view,
        )

    @cute.jit
    def _run_fc2_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        fc2_output_dest: Fc2OutputDest,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        *,
        preload_acc: Optional[Tuple[cute.Tensor, cute.Tensor, cute.Tensor,
                                    cute.Tensor]] = None,
    ) -> None:
        """Run one fc2 epi subtile: LDTM + cvt + pack + transpose + STG.

        Per fc2 epi subtile (64 token x cta_tile_m hidden):

          - 4 epi warp share the subtile.  Per warp = 32 hidden x 64 token.
          - Split into first/second halves (32 token left / 32 token right);
            each half is 32 hidden x 32 token (= one warp-region of acc TMEM).

        Per-half pipeline:

          1. ``Fc2AccLoadAndPack``           : LDTM x 2 + cvt.rn.bf16x2.f32 +
                                               pack pairs into (16, Float32)
                                               packed-bf16x2 RMEM tensor
                                               carrying TmemTranspose16x32-
                                               Packed.InputContract.
          2. ``TmemTranspose16x32Packed``    : 4-round in-place transpose
                                               (skip-R1.Load mode -- the
                                               packed tensor is fed directly
                                               via reg_tensor=).  Output
                                               carries TmemTranspose16x32-
                                               Packed.OutputContract.
          3. ``Fc2UnpackPermuteStg``         : recast packed view back to
                                               (32, BFloat16), permute to
                                               natural hidden order, fire
                                               2 x STG.256 to GMEM out2.

        first half (token 0..31) and second half (token 32..63) are
        sequential; we let the compiler interleave LDTMs / STTMs / STGs
        across the two halves for ILP rather than driving an explicit
        cross-half action ordering (fc2 has no SwiGLU coupling between
        halves, so a simpler structure is fine here).

        ``tmem_subtile_tensor`` (32 hidden lanes, 64 token cols): per-warp
        acc TMEM region of this fc2 subtile (or, in the overlap-acc unroll
        path, a workspace subtile's region this subtile is borrowing).
        Caller-built via ``_subtile_local_tmem_tensor``.

        ``preload_acc`` (4-tuple of (16,) fp32 RMEM tensors): non-None
        only in the overlap-acc unroll path.  Carries the raw acc data
        of this subtile already LDTM'd out by
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` *before* the
        acc TMEM was released.  Tuple ordering (matches fc1):

            preload_acc[0]: first-half top  (cols 0..31, hidden lanes 0..15)
            preload_acc[1]: first-half bot  (cols 0..31, hidden lanes 16..31)
            preload_acc[2]: second-half top (cols 32..63, hidden lanes 0..15)
            preload_acc[3]: second-half bot (cols 32..63, hidden lanes 16..31)

        When provided, ``Fc2AccLoadAndPack`` for each half is constructed
        with ``preload_acc=(top, bot)`` instead of ``tmem_ptr=...``,
        skipping the per-half LDTM x 2.  The downstream
        ``TmemTranspose16x32Packed`` (already skip-R1.Load) and
        ``Fc2UnpackPermuteStg`` are unchanged.
        """
        # -- Per-half TMEM ptrs derived from the (32, 64) subtile view ------
        # Same Python-int alignment propagation convention as fc1.
        tmem_first_ptr = tmem_subtile_tensor.iterator
        tmem_second_ptr = tmem_first_ptr + 32

        # -- Per-subtile pool-token base for the destination row index ---
        #
        # ``fc2_output_dest`` (MoE-domain ``(token_max, topk, hidden)``) is
        # addressed per-lane inside ``Fc2UnpackPermuteStg.stg()``: the
        # caller-supplied ``subtile_pool_token_base`` is the pool-token row
        # of the subtile's token 0, and ``stg()`` adds the lane-mapped
        # ``token_idx`` (∈ [0, 64)) to recover ``pool_token_global``.
        #
        # The pool-token offset accumulates three levels: (1) the current
        # expert's start row (``cumulative_data_physical_row``, expert-
        # padded), (2) the task tile's start within the expert
        # (``tile_n_idx * cta_tile_n``, swap-AB N axis = token axis), and
        # (3) the subtile start within the task tile
        # (``subtile_idx * EpilogueTokenTile``).  No "GEMM-domain"
        # local_tile shift is built here -- the dest descriptor handles
        # all axis-0 / axis-1 / axis-2 stride arithmetic.
        subtile_pool_token_base = (
            work_tile_info.cumulative_data_physical_row +
            work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n) +
            subtile_idx * cutlass.Int32(EpilogueTokenTile))

        # Task-tile-wide valid token-row upper bound (exclusive).
        # ``cumulative_data_physical_row`` is the current expert's start
        # row in the pool; ``tile_n_idx * cta_tile_n`` advances by full
        # task tiles within that expert; ``valid_tokens_in_tile`` is the
        # per-task-tile valid count (= ``cta_tile_n`` except on the last
        # task tile of an expert, where it can be strictly less).  Padding
        # rows in ``[task_tile_data_row_start + valid_tokens_in_tile,
        # task_tile_data_row_start + cta_tile_n)`` have uninitialised
        # ``token_src_metadata`` -- dispatch never writes them, so their
        # ``src_token`` field is whatever the workspace was initialised
        # with (zero in the runner's torch.zeros path, but undefined in
        # general).  ``Fc2UnpackPermuteStg`` uses this bound to gate every
        # STG.256 so a padding lane's stale ``src_token`` never aliases
        # back into ``combine_output[stale_src_token, ...]``.
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row +
            work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n))
        valid_token_row_end = task_tile_data_row_start + work_tile_info.valid_tokens_in_tile

        # Lane within warp (∈ [0, 32)).  In STG.256 mode the per-half
        # token-in-subtile coord is ``thread_in_warp + 32 * half_idx``
        # (lane <-> token 1:1); in REDG mode each lane holds 4 distinct
        # tokens and the (lane // 4) + {0, 8, 16, 24} mapping is
        # recovered inside ``Fc2UnpackPermuteStg._init_redg``.  Either
        # way the class only needs the raw lane index from the caller.
        thread_in_warp = tidx % 32

        # ``redg_subtile_scratch``: the same (32 lanes, 64 cols) view
        # the caller already built for the upstream transpose
        # (= ``tmem_subtile_tensor`` from ``_subtile_local_tmem_tensor``).
        # In REDG mode ``Fc2UnpackPermuteStg`` carves a per-half
        # 32-lane x 16-col STTM+LDTM reshuffle slab out of it at col
        # offset ``32 * half_idx``, matching the per-half tmem_*_ptr
        # split used by the transpose.  Race-free reasoning:
        #   * tile_n=256 (overlap-acc unroll path): in-place transpose
        #     runs on the NEXT subtile's TMEM region, so the per-half
        #     slab here belongs to that next-subtile region and the
        #     next acc mma does not touch it until after the epilogue
        #     body returns.
        #   * tile_n=128: transpose runs on the CURRENT subtile's TMEM
        #     region, but acc_pipeline.consumer_release is hoisted to
        #     AFTER the full per-subtile loop (see
        #     ``_run_fc2_task_tile``), so the next mma is gated by the
        #     same task-tile-boundary fence the REDG STTM has already
        #     cleared by the time release fires.
        # In STG.256 mode the kwarg must be None (the class rejects a
        # non-None scratch when its dest descriptor has
        # ``fc2_output_with_redg=False``).
        if cutlass.const_expr(fc2_output_dest.fc2_output_with_redg):
            redg_subtile_scratch = tmem_subtile_tensor
        else:
            redg_subtile_scratch = None

        # -- First half (token 0..31): LDTM + pack + transpose + STG -------
        if cutlass.const_expr(preload_acc is None):
            first_pack = Fc2AccLoadAndPack(tmem_first_ptr)
        else:
            first_pack = Fc2AccLoadAndPack(preload_acc=(preload_acc[0],
                                                        preload_acc[1]), )
        first_t = TmemTranspose16x32Packed(
            tmem_first_ptr,
            Region.Top,
            reg_tensor=first_pack.output,
        )
        first_t.r1_perm()
        first_t.r1_store()
        first_t.r2_load()
        first_t.r2_store()
        first_t.r3_load_top()
        first_t.r3_load_bot()
        first_t.r3_perm()
        first_t.r3_store()
        first_t.r4_load_top()
        first_t.r4_load_bot()
        first_t.r4_perm()

        first_stg = Fc2UnpackPermuteStg(
            first_t.output,
            fc2_output_dest=fc2_output_dest,
            subtile_pool_token_base=subtile_pool_token_base,
            warp_idx=warp_idx,
            half_idx=0,
            lane_idx=thread_in_warp,
            valid_hidden=valid_hidden,
            tile_hidden_idx=work_tile_info.tile_m_idx,
            hidden_tile_size=self._cta_tile_m,
            needs_hidden_predicate=self._fc2_stg_needs_predicate,
            valid_token_row_end=valid_token_row_end,
            tmem_subtile_scratch=redg_subtile_scratch,
        )
        first_stg.stg()

        # -- Second half (token 32..63): same pipeline -----------------------
        if cutlass.const_expr(preload_acc is None):
            second_pack = Fc2AccLoadAndPack(tmem_second_ptr)
        else:
            second_pack = Fc2AccLoadAndPack(preload_acc=(preload_acc[2],
                                                         preload_acc[3]), )
        second_t = TmemTranspose16x32Packed(
            tmem_second_ptr,
            Region.Top,
            reg_tensor=second_pack.output,
        )
        second_t.r1_perm()
        second_t.r1_store()
        second_t.r2_load()
        second_t.r2_store()
        second_t.r3_load_top()
        second_t.r3_load_bot()
        second_t.r3_perm()
        second_t.r3_store()
        second_t.r4_load_top()
        second_t.r4_load_bot()
        second_t.r4_perm()

        second_stg = Fc2UnpackPermuteStg(
            second_t.output,
            fc2_output_dest=fc2_output_dest,
            subtile_pool_token_base=subtile_pool_token_base,
            warp_idx=warp_idx,
            half_idx=1,
            lane_idx=thread_in_warp,
            valid_hidden=valid_hidden,
            tile_hidden_idx=work_tile_info.tile_m_idx,
            hidden_tile_size=self._cta_tile_m,
            needs_hidden_predicate=self._fc2_stg_needs_predicate,
            valid_token_row_end=valid_token_row_end,
            tmem_subtile_scratch=redg_subtile_scratch,
        )
        second_stg.stg()
