# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is copied and modified from cutlass https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py

import ctypes
import os
from typing import Union

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type
from cutlass.cutlass_dsl import T, dsl_user_op

TRTLLM_ENABLE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


# WAR for CuTeDSL make_ptr implementation
class _Pointer(Pointer):
    """Represents a runtime pointer that can interoperate with various data structures,
    including numpy arrays and device memory.

    Args:
        pointer (int or pointer-like object): The pointer to the data.
        dtype (Type): Data type of the elements pointed to.
        mem_space (_cute_ir.AddressSpace, optional): Memory space where the pointer resides. Defaults to generic.
        assumed_align (int, optional): Alignment of the input pointer in bytes. Defaults to None.

    Attributes:
        _pointer: The underlying pointer.
        _dtype: Data type of the elements.
        _addr_space: Memory space of the pointer.
        _assumed_align: Alignment of the pointer in bytes.
        _desc: C-type descriptor for the pointer.
        _c_pointer: C-compatible pointer representation.
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned")

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(self._dtype.mlir_type, self._addr_space,
                                    self._assumed_align)

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        if expected_py_type is Pointer or (isinstance(
                expected_py_type, ir.Value) and expected_py_type.ty is Pointer):
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Creates a pointer from a memory address.

    Args:
        dtype (Type[Numeric]): Data type of the pointer elements.
        value (Union[int, ctypes._Pointer]): Memory address as an integer or ctypes pointer.
        mem_space (AddressSpace, optional): Memory address space. Defaults to AddressSpace.generic.
        assumed_align (int, optional): Alignment in bytes. Defaults to None.

    Returns:
        Pointer: A pointer object.

    Example:
        ```python
        import numpy as np
        import ctypes
        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)
        # Get pointer address as ctypes pointer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)
        ```
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}")

    return _Pointer(address_value,
                    dtype,
                    mem_space,
                    assumed_align=assumed_align)


def is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@dsl_user_op
def fmin(a: Union[float, cutlass.Float32],
         b: Union[float, cutlass.Float32],
         *,
         nan=False,
         loc=None,
         ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmin(
            T.f32(),
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        ))


def sigmoid_f32(a: Union[float, cutlass.Float32],
                fastmath: bool = False) -> Union[float, cutlass.Float32]:
    """
    Compute the sigmoid of the input tensor.
    """
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(a: Union[float, cutlass.Float32],
             fastmath: bool = False) -> Union[float, cutlass.Float32]:
    """
    Compute the silu of the input tensor.
    """
    return a * sigmoid_f32(a, fastmath=fastmath)


# TODO(zhichenj): try to move these to NVVM wrapper or helper functions
@dsl_user_op
def vectorized_atomic_add_bf16x8(rOut_epi_packed,
                                 scatter_out_offset,
                                 loc=None,
                                 ip=None):
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            llvm.bitcast(T.i32(), rOut_epi_packed[0, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[1, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[2, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[3, None].load().ir_value()),
        ],
        "red.global.v4.bf16x2.add.noftz [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def vectorized_atomic_add_fp32x2(rOut_epi_packed,
                                 scatter_out_offset,
                                 loc=None,
                                 ip=None):
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            rOut_epi_packed[0].ir_value(),
            rOut_epi_packed[1].ir_value(),
        ],
        "red.global.v2.f32.add [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_func(rOut_epi_packed, scatter_out_offset, loc=None, ip=None):
    if cutlass.const_expr(rOut_epi_packed.dtype == cutlass.Float32):
        llvm.inline_asm(
            None,
            [
                scatter_out_offset.iterator.llvm_ptr,
                rOut_epi_packed.ir_value(),
            ],
            "red.global.add.f32 [$0], $1;",
            "l,f",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    elif cutlass.const_expr(rOut_epi_packed.dtype == cutlass.BFloat16):
        llvm.inline_asm(
            None,
            [
                scatter_out_offset.iterator.llvm_ptr,
                llvm.bitcast(T.i16(), rOut_epi_packed.ir_value()),
            ],
            "red.add.noftz.bf16 [$0], $1;",
            "l,h",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp32(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp16(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None) -> None:
    """
    This instruction is used to wait for the previous kernel's grid ending
    (all blocks of the previous kernel have finished and memflushed), i.e.,
    the instruction after this instruction will not be issued until the previous
    grid has finished.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.wait;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None) -> None:
    """
    Issuing the launch_dependents instruction hints a dependent kernel to launch earlier.
    launch_dependents doesn't impact the functionality but the performance:
    Launching a dependent kernel too early can compete with current kernels,
    while launching too late can lead to a long latency.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.launch_dependents;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =========================================================================
# AllReduce PTX helpers for FlashMoE in-kernel cross-rank reduction
# =========================================================================


@dsl_user_op
def atomic_add_global_i32_return(addr, val, *, loc=None, ip=None):
    """Atomically add val to i32 at global addr and return old value."""
    result = llvm.inline_asm(
        T.i32(),
        [addr.iterator.llvm_ptr, val.ir_value()],
        "atom.global.add.s32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def fence_sc_sys(*, loc=None, ip=None):
    """System-scope sequentially-consistent memory fence.

    Ensures all prior memory operations (from this thread) are visible
    to all threads across all GPUs before any subsequent operations.
    Required before signaling readiness to remote ranks.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="fence.sc.sys;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_release_sys_i32(addr, val, *, loc=None, ip=None):
    """Store i32 with release semantics, system scope.

    Ensures all prior writes from this thread are visible to any thread
    (including remote GPUs) that performs an acquire load of this location.
    """
    llvm.inline_asm(
        None,
        [addr.iterator.llvm_ptr, val.ir_value()],
        "st.release.sys.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_sys_i32(addr, *, loc=None, ip=None):
    """Load i32 with acquire semantics, system scope.

    Pairs with st_release_sys_i32. After this load returns a value written
    by a release store, all writes that happened-before that store are
    guaranteed visible to this thread.
    """
    result = llvm.inline_asm(
        T.i32(),
        [addr.iterator.llvm_ptr],
        "ld.acquire.sys.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def ld_volatile_global_i32(addr, *, loc=None, ip=None):
    """Volatile load i32 from global memory.

    Bypasses L1/L2 caches. Used for polling CTA exit counters where
    the writer is on the same device (no need for system-scope ordering).
    """
    result = llvm.inline_asm(
        T.i32(),
        [addr.iterator.llvm_ptr],
        "ld.volatile.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def ld_global_128b(addr, *, loc=None, ip=None):
    """Load 128 bits (4 x i32) from global memory.

    Used for vectorized bf16 loads during AllReduce: 128 bits = 8 bf16 values.
    Returns 4 i32 values (each containing 2 packed bf16).
    """
    r0, r1, r2, r3 = llvm.inline_asm(
        [T.i32(), T.i32(), T.i32(), T.i32()],
        [addr.iterator.llvm_ptr],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return r0, r1, r2, r3


@dsl_user_op
def st_global_128b(addr, r0, r1, r2, r3, *, loc=None, ip=None):
    """Store 128 bits (4 x i32) to global memory.

    Used for vectorized bf16 stores during AllReduce.
    """
    llvm.inline_asm(
        None,
        [addr.iterator.llvm_ptr, r0, r1, r2, r3],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_i64(addr, *, loc=None, ip=None):
    """Load i64 from global memory. Used for reading IPC pointer values."""
    result = llvm.inline_asm(
        T.i64(),
        [addr.iterator.llvm_ptr],
        "ld.global.b64 $0, [$1];",
        "=l,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(result)


@dsl_user_op
def add_bf16x2(a_packed, b_packed, *, loc=None, ip=None):
    """Add two packed bf16x2 values natively on SM90+.

    Takes two i32 values (each containing 2 packed bf16), adds them
    element-wise, and returns one i32 (packed bf16x2 result).
    Uses native bf16x2 add instruction available on Blackwell (SM100).
    """
    result = llvm.inline_asm(
        T.i32(),
        [a_packed, b_packed],
        "add.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def ptr_add_i64(base, byte_offset, *, loc=None, ip=None):
    """Add byte offset to a 64-bit pointer/address. Returns i64."""
    result = llvm.inline_asm(
        T.i64(),
        [base, byte_offset],
        "add.u64 $0, $1, $2;",
        "=l,l,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def ld_global_128b_from_i64_addr(addr_i64, *, loc=None, ip=None):
    """Load 128 bits (4 x i32) from a 64-bit address value.

    Unlike ld_global_128b which takes a cute pointer, this takes a raw
    i64 address value (e.g., from reading an IPC pointer array).
    """
    r0, r1, r2, r3 = llvm.inline_asm(
        [T.i32(), T.i32(), T.i32(), T.i32()],
        [addr_i64],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return r0, r1, r2, r3


@dsl_user_op
def st_global_128b_to_i64_addr(addr_i64, r0, r1, r2, r3, *, loc=None, ip=None):
    """Store 128 bits (4 x i32) to a 64-bit address value."""
    llvm.inline_asm(
        None,
        [addr_i64, r0, r1, r2, r3],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---- Address-based variants (take raw i64 addresses) ----


@dsl_user_op
def st_release_sys_i32_addr(addr_i64, val, *, loc=None, ip=None):
    """Store i32 with release semantics at a raw i64 address.

    Like st_release_sys_i32 but takes a raw i64 address value instead
    of a cute tensor pointer.
    """
    llvm.inline_asm(
        None,
        [addr_i64, val.ir_value()],
        "st.release.sys.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_sys_i32_addr(addr_i64, *, loc=None, ip=None):
    """Load i32 with acquire semantics from a raw i64 address.

    Like ld_acquire_sys_i32 but takes a raw i64 address value.
    """
    result = llvm.inline_asm(
        T.i32(),
        [addr_i64],
        "ld.acquire.sys.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def ld_volatile_global_i32_addr(addr_i64, *, loc=None, ip=None):
    """Volatile load i32 from a raw i64 address."""
    result = llvm.inline_asm(
        T.i32(),
        [addr_i64],
        "ld.volatile.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def i32_to_i64(val, *, loc=None, ip=None):
    """Zero-extend i32 to i64 for use in address arithmetic."""
    result = llvm.inline_asm(
        T.i64(),
        [val.ir_value()],
        "cvt.u64.u32 $0, $1;",
        "=l,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def i64_mul(a, b, *, loc=None, ip=None):
    """Multiply two i64 values."""
    result = llvm.inline_asm(
        T.i64(),
        [a, b],
        "mul.lo.u64 $0, $1, $2;",
        "=l,l,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result
