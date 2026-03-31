# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Inline PTX helpers for NVLink multicast (multimem) operations and
multicast barrier synchronization used in fused GEMM + AllReduce kernels.

These helpers wrap SM90+ multimem.ld_reduce / multimem.st instructions and
atomic barrier operations on multicast addresses, following the same
@dsl_user_op + llvm.inline_asm pattern used in utils.py.

Two paths are provided:
  1. Multicast path (>= 2 GPUs with NVSwitch): multimem_ld_reduce + multimem_st
  2. IPC unicast path (2 GPUs without NVSwitch): regular loads from IPC
     pointers + local add + regular store
"""

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _llvm_struct(*element_types):
    """Create LLVM literal struct type, compatible across cutlass DSL versions.

    Some cutlass DSL versions provide T.struct(); older ones do not.
    Falls back to ir.Type.parse() which is always available.
    """
    if hasattr(T, "struct"):
        return T.struct(list(element_types))
    type_strs = [str(t) for t in element_types]
    return ir.Type.parse(f"!llvm.struct<({', '.join(type_strs)})>")


# ---------------------------------------------------------------------------
# Multicast reduce-load: multimem.ld_reduce.global.add.v4.f32
# The NVLink switch performs the reduction across all ranks.
# ---------------------------------------------------------------------------


@dsl_user_op
def multimem_ld_reduce_add_v4_f32(mc_addr, *, loc=None, ip=None):
    """Load 4xf32 from a multicast address with NVLink reduce-add.

    The NVLink switch sums the values written by all ranks at *mc_addr* and
    returns the reduced result.

    Args:
        mc_addr: Pointer to multicast global memory (cutlass Int64 / ptr).

    Returns:
        Tuple of 4 Float32 values (128-bit vector load).
    """
    results = llvm.inline_asm(
        _llvm_struct(T.f32(), T.f32(), T.f32(), T.f32()),
        [mc_addr.ir_value()],
        "multimem.ld_reduce.global.add.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    v0 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [0]))
    v1 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [1]))
    v2 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [2]))
    v3 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [3]))
    return v0, v1, v2, v3


# ---------------------------------------------------------------------------
# Multicast store: multimem.st.global.v4.f32
# Writes to a multicast address visible to all ranks via NVLink switch.
# ---------------------------------------------------------------------------


@dsl_user_op
def multimem_st_v4_f32(mc_addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """Store 4xf32 to a multicast global address.

    All ranks' stores are visible to subsequent multimem.ld_reduce operations.

    Args:
        mc_addr: Pointer to multicast global memory.
        v0, v1, v2, v3: Four Float32 values to store (128-bit vector store).
    """
    llvm.inline_asm(
        None,
        [
            mc_addr.ir_value(),
            v0.ir_value(),
            v1.ir_value(),
            v2.ir_value(),
            v3.ir_value(),
        ],
        "multimem.st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# BF16 vector store to global memory (non-multicast, for staging buffer)
# st.global.v4.b32  stores 4×b32 = 8×bf16 = 128 bits
# ---------------------------------------------------------------------------


@dsl_user_op
def st_global_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """Store 4×b32 (8×bf16) to global memory.

    Args:
        addr: Global memory pointer (Int64).
        v0, v1, v2, v3: Four Int32 values (each holding 2 bf16).
    """
    llvm.inline_asm(
        None,
        [
            addr.ir_value(),
            v0.ir_value(),
            v1.ir_value(),
            v2.ir_value(),
            v3.ir_value(),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# ld.global.v4.b32  loads 4×b32 = 128 bits (for IPC unicast path)
# ---------------------------------------------------------------------------


@dsl_user_op
def ld_global_v4_f32(addr, *, loc=None, ip=None):
    """Load 4xf32 (128 bits) from global memory.

    Used in the IPC unicast path to load from each rank's IPC pointer.

    Args:
        addr: Global memory pointer.

    Returns:
        Tuple of 4 Float32 values.
    """
    results = llvm.inline_asm(
        _llvm_struct(T.f32(), T.f32(), T.f32(), T.f32()),
        [addr.ir_value()],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    v0 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [0]))
    v1 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [1]))
    v2 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [2]))
    v3 = cutlass.Float32(llvm.extractvalue(T.f32(), results, [3]))
    return v0, v1, v2, v3


# ---------------------------------------------------------------------------
# Multicast barrier: arrive via atomicAdd on a multicast flag
# ---------------------------------------------------------------------------


@dsl_user_op
def barrier_arrive_mc(flag_addr, *, loc=None, ip=None):
    """Atomic increment on a multicast barrier flag (arrive phase).

    Each rank increments the flag at the multicast address; when all ranks
    have arrived the flag equals world_size.

    Args:
        flag_addr: Pointer to the int32 flag in multicast memory.
    """
    # atom.sys.global.add.u32 — system scope atomic for cross-GPU visibility.
    # Default (unscoped) atom.global.add defaults to GPU scope in PTX,
    # which is insufficient for IPC/MNNVL cross-GPU barrier signaling.
    llvm.inline_asm(
        T.i32(),
        [flag_addr.ir_value()],
        "atom.relaxed.sys.global.add.u32 $0, [$1], 1;",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Barrier wait: spin-read a unicast flag until == expected, then reset
# ---------------------------------------------------------------------------


@dsl_user_op
def barrier_try_wait_eq(flag_addr, expected, *, loc=None, ip=None):
    """Non-blocking check if barrier flag equals expected value.

    Args:
        flag_addr: Pointer to the int32 flag in (unicast) memory.
        expected: Expected value (Int32).

    Returns:
        Int32: Current value of the flag.
    """
    # ld.global.acquire.sys — system scope acquire for cross-GPU visibility.
    # GPU-scope acquire only orders loads within the same GPU, not across
    # GPUs. System scope is required to see staging data written by remote
    # GPUs (after their threadfence_system + barrier arrive).
    result = llvm.inline_asm(
        T.i32(),
        [flag_addr.ir_value()],
        "ld.global.acquire.sys.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def barrier_reset(flag_addr, *, loc=None, ip=None):
    """Reset a barrier flag to 0 after all ranks have arrived.

    Uses release semantics to ensure all prior writes are visible.

    Args:
        flag_addr: Pointer to the int32 flag.
    """
    # st.global.release.sys — system scope release for cross-GPU visibility.
    # Ensures the zero write is visible to remote GPUs' acquire loads.
    llvm.inline_asm(
        None,
        [flag_addr.ir_value()],
        "st.global.release.sys.b32 [$0], 0;",
        "l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# ld.global.v4.b32  loads 4×b32 = 128 bits (as raw bit patterns)
# ---------------------------------------------------------------------------


@dsl_user_op
def ld_global_v4_b32(addr, *, loc=None, ip=None):
    """Load 4×b32 (128 bits) from global memory as Int32 values.

    Used in the IPC unicast path to load bf16 data from each rank's buffer.
    Returns raw b32 values (each containing 2 packed bf16 values).

    Args:
        addr: Global memory pointer.

    Returns:
        Tuple of 4 Int32 values (each holding 2 packed bf16).
    """
    results = llvm.inline_asm(
        _llvm_struct(T.i32(), T.i32(), T.i32(), T.i32()),
        [addr.ir_value()],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    v0 = cutlass.Int32(llvm.extractvalue(T.i32(), results, [0]))
    v1 = cutlass.Int32(llvm.extractvalue(T.i32(), results, [1]))
    v2 = cutlass.Int32(llvm.extractvalue(T.i32(), results, [2]))
    v3 = cutlass.Int32(llvm.extractvalue(T.i32(), results, [3]))
    return v0, v1, v2, v3


# ---------------------------------------------------------------------------
# BF16 unpack/pack and conversion helpers for IPC reduce path
#
# Each b32 word holds 2 packed bf16 values. For cross-rank accumulation
# we unpack to f32, sum, and re-pack to bf16.
# ---------------------------------------------------------------------------


@dsl_user_op
def bf16x2_to_f32x2(packed_b32, *, loc=None, ip=None):
    """Unpack one b32 word containing 2 bf16 into 2 f32 values.

    Uses mov.b32 to split the packed value, then cvt.f32.bf16 for each half.

    Args:
        packed_b32: Int32 holding 2 packed bf16 values.

    Returns:
        Tuple (f32_lo, f32_hi) — the two bf16 values converted to f32.
    """
    results = llvm.inline_asm(
        _llvm_struct(T.f32(), T.f32()),
        [packed_b32.ir_value()],
        "{.reg .b16 %lo, %hi;mov.b32 {%lo, %hi}, $2;cvt.f32.bf16 $0, %lo;cvt.f32.bf16 $1, %hi;}",
        "=f,=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    lo = cutlass.Float32(llvm.extractvalue(T.f32(), results, [0]))
    hi = cutlass.Float32(llvm.extractvalue(T.f32(), results, [1]))
    return lo, hi


@dsl_user_op
def f32x2_to_bf16x2(f32_lo, f32_hi, *, loc=None, ip=None):
    """Pack 2 f32 values into one b32 word as 2 bf16 (with rounding).

    Uses cvt.rn.bf16.f32 + mov.b32 to pack.

    Args:
        f32_lo: Float32 — lower bf16 slot.
        f32_hi: Float32 — upper bf16 slot.

    Returns:
        Int32 holding 2 packed bf16.
    """
    result = llvm.inline_asm(
        T.i32(),
        [f32_lo.ir_value(), f32_hi.ir_value()],
        "{"
        ".reg .b16 %lo, %hi;"
        "cvt.rn.bf16.f32 %lo, $1;"
        "cvt.rn.bf16.f32 %hi, $2;"
        "mov.b32 $0, {%lo, %hi};"
        "}",
        "=r,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


# ---------------------------------------------------------------------------
# Threadfence for cross-rank visibility
# ---------------------------------------------------------------------------


@dsl_user_op
def threadfence_system(*, loc=None, ip=None):
    """Issue membar.sys to ensure cross-GPU memory ordering."""
    llvm.inline_asm(
        None,
        [],
        "membar.sys;",
        "",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Device-local atomic add returning old value (for CTA exit counter)
# ---------------------------------------------------------------------------


@dsl_user_op
def atomic_add_return_old(addr, *, loc=None, ip=None):
    """Atomic increment on a device-local global memory flag, returning old value.

    Unlike barrier_arrive_mc which targets multicast addresses, this is for
    regular device memory used for intra-GPU CTA coordination.

    Args:
        addr: Pointer to int32 flag in device global memory (NOT multicast).

    Returns:
        Int32: The old value before the increment.
    """
    result = llvm.inline_asm(
        T.i32(),
        [addr.ir_value()],
        "atom.global.add.u32 $0, [$1], 1;",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


# ---------------------------------------------------------------------------
# Grid dimension query (for CTA exit counter coordination)
# ---------------------------------------------------------------------------


@dsl_user_op
def get_num_ctas(*, loc=None, ip=None):
    """Get total number of CTAs in the grid (nctaid.x * nctaid.y).

    Used to detect the last CTA that finishes its epilogue tiles
    for CTA-level completion barrier signaling.

    Returns:
        Int32: nctaid.x * nctaid.y (total CTA count).
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "{"
        ".reg .u32 %nx, %ny;"
        "mov.u32 %nx, %nctaid.x;"
        "mov.u32 %ny, %nctaid.y;"
        "mul.lo.u32 $0, %nx, %ny;"
        "}",
        "=r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)
