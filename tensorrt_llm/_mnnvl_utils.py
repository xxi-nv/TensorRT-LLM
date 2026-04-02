# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import pynvml
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from ._dlpack_utils import pack_strided_memory
from ._utils import mpi_comm
from .logger import logger
from .mapping import Mapping


def _check_cu_result(cu_func_ret):
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None


class MnnvlMemory:
    """MNNVL memory management for tensor parallel (TP) operations."""

    # Shared across all subclasses (global/device state).
    initialized: bool = False
    allocation_granularity: int = 0
    fabric_page_size: int = 1 << 29  # 512 MB.
    dev_id: int = None

    # Per-class state attributes. These will be auto-initialized for each subclass
    # to avoid polluting the parent class's state. Use callable (e.g., dict) for mutable defaults.
    _per_class_attrs = {
        "current_mem_offset": 0,
        "current_rank_stride": 0,  # stride for ranks and also address space size.
        "current_start_address": 0,
        "comm": None,  # MPI communicator.
        "allocated_map": dict,  # callable for fresh dict.
        "address_refcnt": dict,  # callable for fresh dict.
    }

    # Initialize per-class state for the base class.
    current_mem_offset: int = 0
    current_rank_stride: int = 0
    current_start_address: int = 0
    comm = None
    allocated_map = {}
    address_refcnt = {}

    def __init_subclass__(cls, **kwargs):
        """Auto-initialize per-class attributes for each subclass to avoid sharing state with parent."""
        super().__init_subclass__(**kwargs)
        for attr, default in cls._per_class_attrs.items():
            if callable(default):
                setattr(cls, attr, default())  # e.g., dict() creates a fresh dict.
            else:
                setattr(cls, attr, default)

    def __init__(self, mapping: Mapping, size: int):
        self.mapping = mapping
        self.segment_size = size
        self.ptr, self.rank_stride = type(self).open_mnnvl_memory(self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            if hasattr(self, "ptr"):
                type(self).close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        num_segments = type(self).comm.Get_size()
        return pack_strided_memory(
            self.ptr, self.segment_size, self.rank_stride, num_segments, dtype, MnnvlMemory.dev_id
        )

    @staticmethod
    def initialize():
        if not MnnvlMemory.initialized:
            # use a dummy torch CUDA tensor to trigger CUDA context initialization
            _ = torch.empty(1, device="cuda")
            # ensure nvml is initialized.
            try:
                pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError_Uninitialized:
                pynvml.nvmlInit()
            MnnvlMemory.initialized = True

    @classmethod
    def get_comm(cls, mapping: Mapping):
        """Get TP-based communicator (ranks grouped by PP+CP+MOE_TP, ordered by TP rank)."""
        if cls.comm is not None:
            return cls.comm
        comm = mpi_comm().Split(
            (mapping.pp_rank * mapping.cp_size + mapping.cp_rank) * mapping.moe_tp_size
            + mapping.moe_tp_rank,
            mapping.tp_rank,
        )
        cls.comm = comm
        return comm

    @staticmethod
    def get_allocation_prop(dev_id: int):
        location = cuda.CUmemLocation()
        location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = dev_id
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

        # TODO: We differentiate FABRIC for GB200 (aarch64) and POSIX_FILE_DESCRIPTOR for BB200 (x86_64).
        # May need to find a better way to handle this.
        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        if is_on_aarch64:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        allocation_prop.location = location
        return allocation_prop

    @staticmethod
    def get_allocation_granularity(dev_id: int):
        if MnnvlMemory.allocation_granularity != 0:
            return MnnvlMemory.allocation_granularity
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        option = cuda.CUmemAllocationGranularity_flags(
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        granularity = _check_cu_result(
            cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option)
        )
        MnnvlMemory.allocation_granularity = granularity
        return MnnvlMemory.allocation_granularity

    @classmethod
    def new_mnnvl_memory_address(cls, mapping: Mapping, size: int):
        page_count = (size + MnnvlMemory.fabric_page_size - 1) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        logger.info(f"[{cls.__name__}] creating address with stride={current_rank_stride}")
        comm = cls.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = _check_cu_result(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size, 0, 0)
        )
        cls.current_start_address = int(ptr)
        cls.current_rank_stride = current_rank_stride
        cls.current_mem_offset = 0

    @classmethod
    def open_mnnvl_memory(cls, mapping: Mapping, size: int):
        # Ensure MnnvlMemory is initialized (for dev_id and allocation_granularity)
        MnnvlMemory.initialize()

        dev = _check_cu_result(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = cls.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(x == size for x in all_rank_allocate_sizes), "Not all rank allocating same size."
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        if cls.current_mem_offset + aligned_size > cls.current_rank_stride:
            cls.new_mnnvl_memory_address(mapping, aligned_size)

        assert cls.current_mem_offset + aligned_size <= cls.current_rank_stride

        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        allocated_mem_handle = _check_cu_result(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
        )
        exported_fabric_handle = _check_cu_result(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle, allocation_prop.requestedHandleTypes, 0
            )
        )
        pidfds = []
        remote_fds = []
        try:
            if (
                allocation_prop.requestedHandleTypes
                == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            ):
                all_handles_data = comm.allgather(exported_fabric_handle.data)
            else:
                all_handles_data = comm.allgather(exported_fabric_handle)
                all_pids = comm.allgather(os.getpid())
                libc = ctypes.CDLL(None, use_errno=True)
                syscall = libc.syscall
                SYS_pidfd_open = 434
                SYS_pidfd_getfd = 438
                for i, pid in enumerate(all_pids):
                    pidfd = syscall(SYS_pidfd_open, pid, 0)
                    if pidfd < 0:
                        err = ctypes.get_errno()
                        raise RuntimeError(
                            f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
                        )
                    pidfds.append(pidfd)

                for i, (pidfd, fd) in enumerate(zip(pidfds, all_handles_data)):
                    remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
                    if remote_fd < 0:
                        err = ctypes.get_errno()
                        error_msg = f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with errno {err}: {os.strerror(err)}."
                        if err == 1:  # EPERM
                            error_msg += (
                                " Permission denied. If running in a container, try adding --cap-add=SYS_PTRACE "
                                "to your docker run command."
                            )
                        else:
                            error_msg += " This may be due to kernel version (requires Linux 5.6+)."
                        raise RuntimeError(error_msg)
                    remote_fds.append(remote_fd)

                all_handles_data = remote_fds
        except Exception:
            # Release resources on failure path to avoid leaks; then re-raise.
            if isinstance(exported_fabric_handle, int):
                try:
                    os.close(exported_fabric_handle)
                except OSError as e:
                    logger.warning(
                        "Failed to close exported shareable handle on error: %s",
                        e,
                    )
            try:
                _check_cu_result(cuda.cuMemRelease(allocated_mem_handle))
            except RuntimeError as e:
                logger.warning(
                    "cuMemRelease failed during error cleanup (original error will be raised): %s",
                    e,
                )
            for _pidfd in pidfds:
                try:
                    os.close(_pidfd)
                except OSError:
                    pass
            for _rfd in remote_fds:
                try:
                    os.close(_rfd)
                except OSError:
                    pass
            raise
        # all_handles_data like b'\x00\x00\x00 \x00\x00\x00\x00\x8f\xec\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # noqa: E501
        # can use buf = memoryview(data) to import if using plain buffer for data.

        madesc = cuda.CUmemAccessDesc()
        madesc.location = allocation_prop.location
        madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        mem_handles = [None] * comm_size

        for i, remote_handle_data in enumerate(all_handles_data):
            rank_ptr = (
                cls.current_start_address + cls.current_rank_stride * i + cls.current_mem_offset
            )
            if i == comm_rank:
                # Local memory mapping
                mem_handles[i] = allocated_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, allocated_mem_handle, 0))
            else:
                # Fabric memory mapping
                imported_mem_handle = _check_cu_result(
                    cuda.cuMemImportFromShareableHandle(
                        remote_handle_data, allocation_prop.requestedHandleTypes
                    )
                )
                mem_handles[i] = imported_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, imported_mem_handle, 0))

            _check_cu_result(cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1))

        ptr = cls.current_start_address + cls.current_mem_offset
        stride = cls.current_rank_stride
        cls.allocated_map[ptr] = (
            mapping,
            aligned_size,
            mem_handles,
            cls.current_start_address,
            cls.current_rank_stride,
            cls.current_mem_offset,
        )
        cls.address_refcnt[cls.current_start_address] = (
            cls.address_refcnt.get(cls.current_start_address, 0) + 1
        )

        cls.current_mem_offset += aligned_size
        return ptr, stride

    @classmethod
    def close_mnnvl_memory(cls, ptr: int):
        mapping, aligned_size, mem_handles, start_address, rank_stride, address_offset = (
            cls.allocated_map.pop(ptr)
        )
        comm = cls.get_comm(mapping)
        comm_size = comm.Get_size()
        for i in range(comm_size):
            rank_ptr = start_address + i * rank_stride + address_offset
            _check_cu_result(cuda.cuMemUnmap(rank_ptr, aligned_size))
            _check_cu_result(cuda.cuMemRelease(mem_handles[i]))
        cls.address_refcnt[start_address] -= 1

        if cls.address_refcnt[start_address] == 0:
            cls.address_refcnt.pop(start_address)
            device_ptr = cuda.CUdeviceptr(start_address)
            _check_cu_result(cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride))
            if start_address == cls.current_start_address:
                cls.current_start_address = 0
                cls.current_rank_stride = 0
                cls.current_mem_offset = 0

    @staticmethod
    def support_nvlink(need_all_up: bool = True):
        dev_id = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        link_count = pynvml.NVML_NVLINK_MAX_LINKS
        active_links = 0
        available_links = 0
        for link_idx in range(link_count):
            try:
                if pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                ):
                    available_links += 1
                    is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                    if is_active:
                        active_links += 1
            except pynvml.NVMLError_NotSupported:
                continue
        return (
            active_links == available_links and available_links > 0
            if need_all_up
            else available_links > 0
        )

    @staticmethod
    def supports_mnnvl() -> bool:
        # TODO:
        # We check if it has all NVLink up now.
        # But it is not equivalent to MNNVL support.
        # May need better support check.
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return support_nvlink_and_all_up


class HelixCpMnnvlMemory(MnnvlMemory):
    """MNNVL memory management for Helix context parallel (CP) operations.

    Per-class state (current_mem_offset, comm, allocated_map, etc.) is automatically
    initialized via __init_subclass__ in the parent class, ensuring this class has
    its own isolated state separate from MnnvlMemory.
    """

    @classmethod
    def get_comm(cls, mapping: Mapping):
        """Get CP-based communicator (ranks grouped by PP+TP+MOE_TP, ordered by CP rank)."""
        if cls.comm is not None:
            return cls.comm
        comm = mpi_comm().Split(
            mapping.pp_rank * mapping.tp_size + mapping.tp_rank,
            mapping.cp_rank,
        )
        cls.comm = comm
        return comm


def init_helix_cp_comm(mapping: Mapping) -> None:
    """Pre-initialize the Helix CP communicator.

    This function MUST be called during model initialization when all ranks
    are synchronized (before any PP pipeline divergence). The MPI Split operation
    is collective and requires all ranks in the communicator to participate.

    In PP (pipeline parallel) mode, different PP stages execute different parts
    of the model at different times. If the communicator is initialized lazily
    during the first forward pass, ranks in different PP stages may not reach
    the Split operation at the same time, causing a deadlock.

    Args:
        mapping: The mapping object containing parallelism configuration.
    """
    if mapping.has_cp_helix() and not mapping.cp_config.get("use_nccl_for_alltoall", True):
        HelixCpMnnvlMemory.get_comm(mapping)


class FlashMoeMnnvlMemory(MnnvlMemory):
    """MNNVL memory for FlashMoE fully-fused EP execution.

    Allocates IPC-accessible buffers per rank for:
    - input_a: NVFP4-quantized activation (packed 2 values per byte)
    - input_sfa: Scale factors for input_a (one uint8 per sf_vec_size elements)
    - staging: BF16 staging buffer for FC2 output before AllReduce
    - output: BF16 final output buffer after AllReduce
    - fc1_counters: Per-expert-group atomic counters for FC1->FC2 barrier
    - barriers: Completion barrier + CTA exit counter for AR synchronization

    Input buffers are sized for max_input_tokens (per-rank local tokens).
    Output/staging buffers are sized for max_output_tokens (global tokens across
    all EP ranks), since FC2 scatter-adds to global token positions.

    Per-class state is automatically isolated via __init_subclass__.
    """

    # Alignment for each sub-buffer within the per-rank region
    _ALIGN = 256

    def __init__(
        self,
        mapping: Mapping,
        max_input_tokens: int,
        max_output_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_expert_groups: int,
        sf_vec_size: int = 16,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_expert_groups = num_expert_groups
        self.sf_vec_size = sf_vec_size
        self.sfa_cols = (hidden_size + sf_vec_size - 1) // sf_vec_size

        align = self._ALIGN

        def _align(size):
            return (size + align - 1) & ~(align - 1)

        # Input buffers: sized per-rank (max_input_tokens)
        # input_a: NVFP4 packed, 2 values per byte
        self._input_a_bytes = max_input_tokens * (hidden_size // 2)
        # input_sfa: one uint8 scale per sf_vec_size elements
        self._input_sfa_bytes = max_input_tokens * self.sfa_cols

        # Output buffers: sized for global tokens (max_output_tokens)
        # staging: bf16 for FC2 output before AllReduce (IPC-readable)
        self._staging_bytes = max_output_tokens * hidden_size * 2
        # output: bf16 final output after AllReduce
        self._output_bytes = max_output_tokens * hidden_size * 2

        # Per-expert-group FC1 tile counters: int32 each
        self._counters_bytes = num_expert_groups * 4
        # Completion barrier (int32) + CTA exit counter (int32)
        self._barriers_bytes = 8

        # Compute offsets (each sub-buffer aligned)
        self.input_a_offset = 0
        self.input_sfa_offset = _align(self._input_a_bytes)
        self.staging_offset = self.input_sfa_offset + _align(self._input_sfa_bytes)
        self.output_offset = self.staging_offset + _align(self._staging_bytes)
        self.counters_offset = self.output_offset + _align(self._output_bytes)
        self.barriers_offset = self.counters_offset + _align(self._counters_bytes)

        total_per_rank = self.barriers_offset + _align(self._barriers_bytes)

        super().__init__(mapping, total_per_rank)

        self.local_rank = mapping.moe_ep_rank
        self.num_ranks = mapping.moe_ep_size

    @classmethod
    def get_comm(cls, mapping: Mapping):
        """Get EP-based communicator (ranks grouped by PP+CP+TP, ordered by EP rank)."""
        if cls.comm is not None:
            return cls.comm
        comm = mpi_comm().Split(
            (mapping.pp_rank * mapping.cp_size + mapping.cp_rank) * mapping.tp_size
            + mapping.tp_rank,
            mapping.moe_ep_rank,
        )
        cls.comm = comm
        return comm

    def _rank_ptr(self, rank: int, offset: int) -> int:
        """Compute raw device pointer to rank's buffer at given offset."""
        return self.ptr + (rank - self.local_rank) * self.rank_stride + offset

    def get_input_ipc_ptrs(self) -> torch.Tensor:
        """Return Int64 tensor of IPC pointers to each rank's input A buffer."""
        return torch.tensor(
            [self._rank_ptr(i, self.input_a_offset) for i in range(self.num_ranks)],
            dtype=torch.int64,
            device="cuda",
        )

    def get_sfa_ipc_ptrs(self) -> torch.Tensor:
        """Return Int64 tensor of IPC pointers to each rank's SFA buffer."""
        return torch.tensor(
            [self._rank_ptr(i, self.input_sfa_offset) for i in range(self.num_ranks)],
            dtype=torch.int64,
            device="cuda",
        )

    def get_staging_ipc_ptrs(self) -> torch.Tensor:
        """Return Int64 tensor of IPC pointers to each rank's staging buffer."""
        return torch.tensor(
            [self._rank_ptr(i, self.staging_offset) for i in range(self.num_ranks)],
            dtype=torch.int64,
            device="cuda",
        )

    def get_local_input_a(self) -> torch.Tensor:
        """Return local rank's input A buffer as uint8 [max_input_tokens, K/2]."""
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.input_a_offset
        end = start + self._input_a_bytes
        return full[self.local_rank, start:end].view(self.max_input_tokens, self.hidden_size // 2)

    def get_local_input_sfa(self) -> torch.Tensor:
        """Return local rank's SFA buffer as uint8 [max_input_tokens, sfa_cols]."""
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.input_sfa_offset
        end = start + self._input_sfa_bytes
        return full[self.local_rank, start:end].view(self.max_input_tokens, self.sfa_cols)

    def get_local_output(self) -> torch.Tensor:
        """Return local rank's output buffer as bf16 [max_output_tokens, hidden]."""
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.output_offset
        end = start + self._output_bytes
        buf = full[self.local_rank, start:end]
        return buf.view(torch.bfloat16).view(self.max_output_tokens, self.hidden_size)

    def get_local_staging(self) -> torch.Tensor:
        """Return local rank's staging buffer as bf16 [max_output_tokens, hidden]."""
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.staging_offset
        end = start + self._staging_bytes
        buf = full[self.local_rank, start:end]
        return buf.view(torch.bfloat16).view(self.max_output_tokens, self.hidden_size)

    def get_counters_ptr(self) -> int:
        """Return pointer to local rank's FC1 tile counters array."""
        return self._rank_ptr(self.local_rank, self.counters_offset)

    def get_barriers_ptr(self) -> int:
        """Return pointer to local rank's barrier+exit_counter region."""
        return self._rank_ptr(self.local_rank, self.barriers_offset)

    def get_cta_exit_counter(self) -> torch.Tensor:
        """Return local rank's CTA exit counter as a 1-element Int32 tensor.

        The CTA exit counter is at offset 0 within the barriers region.
        """
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.barriers_offset
        buf = full[self.local_rank, start : start + 4]
        return buf.view(torch.int32)

    def get_rank_ready_flag_ipc_ptrs(self) -> torch.Tensor:
        """Return Int64 tensor of IPC pointers to each rank's ready flag.

        Each rank's ready flag is a single Int32 at offset 4 within its
        barriers region (right after the cta_exit_counter). The flag is
        written by the owning rank and read by all ranks via IPC.
        """
        flag_offset = self.barriers_offset + 4
        return torch.tensor(
            [self._rank_ptr(i, flag_offset) for i in range(self.num_ranks)],
            dtype=torch.int64,
            device="cuda",
        )

    def reset_ar_state(self) -> None:
        """Reset AllReduce synchronization state before kernel launch.

        Zeros the local rank's CTA exit counter and rank ready flag.
        Must be followed by a cross-rank barrier() to ensure all ranks
        have reset before the kernel starts.
        """
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.barriers_offset
        end = start + self._barriers_bytes
        full[self.local_rank, start:end].zero_()

    def write_input(self, x_nvfp4: torch.Tensor, x_sf: torch.Tensor) -> None:
        """Copy locally quantized NVFP4 input and SFA into IPC-accessible buffers."""
        local_a = self.get_local_input_a()
        local_sfa = self.get_local_input_sfa()
        num_tokens = x_nvfp4.shape[0]
        local_a[:num_tokens].copy_(x_nvfp4[:num_tokens].view_as(local_a[:num_tokens]))
        local_sfa[:num_tokens].copy_(x_sf[:num_tokens].view_as(local_sfa[:num_tokens]))

    def gather_all_input_a(self, tokens_per_rank: int) -> torch.Tensor:
        """Gather NVFP4 input from all ranks via fabric memory.

        Uses the strided tensor view to directly read each rank's input_a
        buffer (IPC-accessible via NVLink fabric memory).

        Args:
            tokens_per_rank: Number of active tokens per rank to gather.

        Returns:
            Contiguous uint8 tensor [tokens_per_rank * num_ranks, hidden_size/2].
        """
        full = self.as_torch_strided_tensor(torch.uint8)
        k_half = self.hidden_size // 2
        start = self.input_a_offset
        parts = []
        for r in range(self.num_ranks):
            end = start + tokens_per_rank * k_half
            rank_data = full[r, start:end].reshape(tokens_per_rank, k_half)
            parts.append(rank_data.clone())
        return torch.cat(parts, dim=0)

    def gather_all_input_sfa(self, tokens_per_rank: int) -> torch.Tensor:
        """Gather SFA (scale factors for A) from all ranks via fabric memory.

        Args:
            tokens_per_rank: Number of active tokens per rank to gather.

        Returns:
            Contiguous uint8 tensor [tokens_per_rank * num_ranks, sfa_cols].
        """
        full = self.as_torch_strided_tensor(torch.uint8)
        start = self.input_sfa_offset
        parts = []
        for r in range(self.num_ranks):
            end = start + tokens_per_rank * self.sfa_cols
            rank_data = full[r, start:end].reshape(tokens_per_rank, self.sfa_cols)
            parts.append(rank_data.clone())
        return torch.cat(parts, dim=0)

    def barrier(self) -> None:
        """Cross-rank barrier ensuring IPC data is visible to all ranks."""
        torch.cuda.synchronize()
        type(self).comm.Barrier()


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: torch.Tensor
    send_rank_count_cumsum: torch.Tensor
    send_rank_local_indices: torch.Tensor
    recv_rank_count_cumsum: torch.Tensor
    recv_rank_local_indices: torch.Tensor
    backward_recv_rank_local_indices: torch.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_prepare_workspace: MnnvlMemory = None
    moe_workspace_tensor: torch.Tensor = None
    moe_prepare_workspace_tensor: torch.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor

        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(
            mapping.moe_ep_size
        )
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(torch.uint64)
        torch.ops.trtllm.moe_initialize_workspace(
            MnnvlMoe.moe_workspace_tensor, mapping.moe_ep_rank, mapping.moe_ep_size
        )
        torch.cuda.synchronize()
        MnnvlMoe.moe_workspace.comm.barrier()
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def get_moe_prepare_workspace(mapping: Mapping):
        if MnnvlMoe.moe_prepare_workspace_tensor is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_prepare_workspace_tensor
        workspace_size_per_rank = torch.ops.trtllm.get_moe_prepare_workspace_size_per_rank(
            mapping.moe_ep_size
        )
        MnnvlMoe.moe_prepare_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_prepare_workspace_tensor = (
            MnnvlMoe.moe_prepare_workspace.as_torch_strided_tensor(torch.uint64)
        )
        return MnnvlMoe.moe_prepare_workspace_tensor

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: torch.Tensor, expert_count: int, ep_size: int
    ):
        assert expert_count % ep_size == 0, "expert_count should be divisible by ep_size"
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare_without_allgather(
        expert_ids: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ):
        (
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            gathered_expert_statics,
        ) = torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size
        # Looks like we don't need this.
        local_gather_indices = None

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            local_token_allocation_count,
        )

        return alltoall_info, gathered_expert_statics

    @staticmethod
    def mnnvl_moe_expert_static_allgather(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
    ):
        gathered_expert_ids = torch.ops.trtllm.mnnvl_moe_expert_static_allgather(
            expert_ids, workspace, ep_rank, ep_size, expert_count
        )
        return gathered_expert_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cumsum: Optional[torch.Tensor],
        gathered_expert_ids: torch.Tensor,
        gathered_scales: Optional[torch.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = torch.ops.trtllm.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size

        local_expert_ids = torch.empty(
            local_token_allocation_count, top_k, dtype=torch.int32, device=torch.device("cuda")
        )
        if gathered_scales is None:
            local_scales = None
        else:
            local_scales = torch.empty(
                local_token_allocation_count,
                top_k,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )

        torch.ops.trtllm.moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: Union[torch.Tensor, List[Optional[torch.Tensor]]],
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # Convert single tensor to list for unified handling
        is_single_tensor = not isinstance(x, list)
        if is_single_tensor:
            assert x.dim() == 2, "only 2D tensor supported, please reshape."
            x = [x]

        assert len(x) > 0, "Empty tensor list not supported"

        # Filter out None values
        valid_list = [tensor is not None for tensor in x]
        valid_tensors = [tensor for tensor in x if tensor is not None]

        if len(valid_tensors) == 0:
            # All tensors are None, return list of None
            result = [None] * len(x)
        else:
            first_dim = None
            for tensor in valid_tensors:
                # Validate dimensions of valid tensors
                assert tensor.dim() == 2, "only 2D tensor supported, please reshape."
                if first_dim is None:
                    first_dim = tensor.shape[0]
                else:
                    assert tensor.shape[0] == first_dim, (
                        f"All tensors must have the same first dimension, got {tensor.shape[0]} vs {first_dim}"
                    )

            # Process only valid tensors
            output_tensors = torch.ops.trtllm.moe_comm(
                valid_tensors,
                alltoall_info.send_rank_count_cumsum,
                alltoall_info.send_rank_local_indices,
                alltoall_info.recv_rank_count_cumsum,
                alltoall_info.recv_rank_local_indices,
                workspace,
                alltoall_info.local_token_allocation_count,
                ep_rank,
                ep_size,
            )

            # Restore None positions in output
            idx = 0
            result = []
            for is_valid in valid_list:
                if is_valid:
                    result.append(output_tensors[idx])
                    idx += 1
                else:
                    result.append(None)

        # If input was a single tensor, return a single tensor
        if is_single_tensor:
            result = result[0]

        return result

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
        use_low_precision_combine: bool = False,
        do_reduce: bool = True,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensors = torch.ops.trtllm.moe_comm(
            [x],
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            token_count * top_k,
            ep_rank,
            ep_size,
            [True],
            use_low_precision_combine,
        )
        output_tensor = output_tensors[0].reshape(token_count, top_k, x.shape[1])
        if do_reduce:
            return torch.sum(output_tensor, dim=1, keepdim=False)
        else:
            return output_tensor
