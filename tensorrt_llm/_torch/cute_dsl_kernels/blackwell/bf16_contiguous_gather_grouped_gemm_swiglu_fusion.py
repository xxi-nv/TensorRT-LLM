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

"""BF16 Contiguous Grouped GEMM with Gather + SwiGLU Fusion for Blackwell (SM100).

Fused FC1 operator: Gather + GroupedGEMM + SwiGLU in a single kernel.

Architecture: 11 specialized warps per CTA
  Warp 0-3:  Epilogue warps (TMEM -> register -> SwiGLU -> SMEM -> TMA store)
  Warp 4-7:  LDGSTS A warps (load A from GMEM to SMEM with gather via token_id_mapping)
  Warp 8:    MMA warp (tcgen05.mma bf16 GEMM)
  Warp 9:    TMA B warp (TMA load B to SMEM)
  Warp 10:   Scheduler warp (contiguous grouped tile scheduling with mn_limit)

Key features:
  - LDGSTS gather for A: non-contiguous token gather via token_id_mapping
  - TMA load for B: standard contiguous TMA with multicast
  - Separate A/B pipelines: A=PipelineCpAsyncUmma, B=PipelineTmaUmma
  - SwiGLU epilogue: pairs up/gate subtiles, output = up * silu(gate)
  - B weights interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
  - C output has halved N: C is (M, N/2, 1) due to SwiGLU fusion
  - 5-element sInfo: (bidx, bidy, expert_idx, valid, mn_limit)
"""

from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    silu_f32,
)


class Sm100Bf16ContiguousGatherGroupedGemmSwigluFusionKernel:
    """BF16 contiguous grouped GEMM with gather + SwiGLU fusion for Blackwell GPUs.

    Implements fused FC1: C = up * silu(gate) where [up, gate] = alpha * A[token_ids] x B
    A is gathered via token_id_mapping, B weights are interleaved for SwiGLU.

    :param mma_tiler_mn: Shape of the MMA tile (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M, N)
    :type cluster_shape_mn: Tuple[int, int]
    :param topk: Number of experts selected per token
    :type topk: int

    :note: Supported A/B data types: BFloat16, Float16
    :note: Supported accumulator: Float32
    :note: C output data type: BFloat16 or Float16
    :note: Only 1CTA mode (M=128) supported
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        topk: int = 1,
    ):
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.topk = topk

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1

        # 11 warps: epilogue(0-3), LDGSTS A(4-7), MMA(8), TMA B(9), scheduler(10)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                *self.ldgsts_a_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                *self.ldgsts_a_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
            )
        )
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_epilogue_warps = 216

        # Barriers
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.tmem_final_offset = 384
        self.num_tmem_alloc_cols = 512

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs.

        Configures tiled MMA, MMA/cluster/tile shapes, cluster layout,
        multicast CTAs for B, epilogue subtile (fixed 128x64 for SwiGLU),
        A/B/C stage counts, A/B/C SMEM layouts, and LDGSTS parameters.
        """
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )

        # Configure tiled mma for bf16 (no blockscaling)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        # SwiGLU output has halved N
        self.mma_tiler_c = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1] // 2,
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Only B uses TMA multicast (A uses LDGSTS gather)
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Fixed epilogue tile for SwiGLU output (halved N)
        self.epi_tile = (128, 64)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1],
        )

        # Compute stages
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.num_smem_capacity,
            self.occupancy,
        )

        # Compute A/B/C shared memory layouts
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Accumulator staging
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage

        # LDGSTS parameters for bf16
        # LDGSTS.128 loads 128 bits = 8 bf16 elements per copy
        self.ldgsts_elements_per_copy = 8
        self.ldgsts_k_threads = 8  # from a_thread_layout (16, 8)
        self.ldgsts_k_per_iter = self.ldgsts_elements_per_copy * self.ldgsts_k_threads  # 64
        self.k_sub_iters = self.cta_tile_shape_mnk[2] // self.ldgsts_k_per_iter

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the fused gather + GroupedGEMM + SwiGLU kernel.

        :param a: Input tensor A (bf16/fp16, row-major [orig_m, K, 1])
        :param b: Input tensor B (bf16/fp16, row-major [N, K, L]), interleaved weights
        :param c: Output tensor C (row-major [M, N/2, 1])
        :param tile_idx_to_group_idx: Mapping from tile to expert [num_tiles]
        :param tile_idx_to_mn_limit: Absolute position boundary per tile [num_tiles]
        :param token_id_mapping_tensor: Token IDs for gather [M]
        :param num_non_exiting_tiles: Number of valid tiles [1]
        :param alpha: Alpha per group [L]
        :param max_active_clusters: Maximum active clusters
        :param stream: CUDA stream
        :param epilogue_op: Optional elementwise epilogue
        """
        # Setup static attributes
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for B only (A uses LDGSTS gather)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = b_copy_size * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size using C tensor (halved N)
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage (separate A/B mbar for separate pipelines)
        @cute.struct
        class SharedStorage:
            # 5 elements per tile: (bidx, bidy, expert_idx, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel for persistent bf16 gather + GEMM + SwiGLU."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors (B load + C store)
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init barriers
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # A pipeline: LDGSTS producer (4 warps, 128 threads) + MMA consumer
        a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 4,
        )
        a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # B pipeline: TMA producer (1 warp) + MMA consumer
        b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_b
        b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # ACC pipeline
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Tile info pipeline
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory allocator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Setup SMEM tensors
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        # Multicast mask for B
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Partition global tensors
        # A: tiled by cta_tile_shape (LDGSTS)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        # B: tiled by mma_tiler (TMA)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # Token mapping: tiled by cta_tile M
        gToken_ml = cute.local_tile(
            token_id_mapping_tensor, cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)), (None,)
        )
        # C: tiled by mma_tiler_c (halved N)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # Partition for TiledMMA B/C (A uses LDGSTS, not TMA)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partition B
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # MMA fragments: A from SMEM (swizzled view), B from SMEM
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before tensor memory alloc
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        griddepcontrol_wait()

        # =========================================================
        # Scheduler warp (warp 10) - 5-element tile_info with mn_limit
        # =========================================================
        if warp_idx == self.sched_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_sched_warps)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_valid_tiles = num_non_exiting_tiles[0]

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)

                if mma_tile_coord_m < num_valid_tiles:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    expert_idx = tile_idx_to_group_idx[mma_tile_coord_m]
                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                        sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                            work_tile.is_valid_tile
                        )
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Sentinel for end
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # =========================================================
        # LDGSTS A warps (warps 4-7) - gather A from GMEM to SMEM
        # =========================================================
        if warp_idx <= self.ldgsts_a_warp_id[-1] and warp_idx >= self.ldgsts_a_warp_id[0]:
            # Setup LDGSTS copy atom for bf16 A
            # 128-bit copy = 8 bf16 elements per LDGSTS instruction
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mA_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout(
                (1, self.ldgsts_elements_per_copy),
                stride=(self.ldgsts_elements_per_copy, 1),
            )
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )

            tidx_in_warpgroup = tidx % 128

            # Flat M×K×stages view of sA for LDGSTS addressing
            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2], self.num_ab_stage),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

            # Pre-allocate register tensors for token offsets and predicates
            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Boolean,
            )

            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Load token IDs for gather: 8 per thread (one per M-iteration)
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                for i in range(8):
                    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                    a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = (
                        a_token_offset_tensor[i] // self.topk
                        if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                        < tile_info[4]
                        else 0
                    )

                tAgA = gA_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume(
                    (tidx_in_warpgroup % 8) * self.ldgsts_elements_per_copy,
                    divby=self.ldgsts_elements_per_copy,
                )

                # Peek A buffer empty
                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < k_tile_cnt:
                    peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                # LDGSTS load loop: load A with gather per K-tile
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

                    tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                    tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]

                    # 8 M-iterations × k_sub_iters K sub-iterations
                    for i in range(8):
                        for k_sub in range(self.k_sub_iters):
                            # GMEM source: A[token_row, k_tile_offset + thread_k + k_sub*64]
                            k_offset = A_gmem_thread_offset + k_sub * self.ldgsts_k_per_iter
                            A_gmem_slice_offset = k_offset + cute.assume(
                                a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                                divby=self.ldgsts_elements_per_copy,
                            )
                            A_gmem_slice_offset = cute.assume(
                                A_gmem_slice_offset,
                                divby=self.ldgsts_elements_per_copy,
                            )
                            tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                            tAgA_slice = cute.make_tensor(
                                tAgA_slice_ptr,
                                layout=cute.make_layout((self.ldgsts_elements_per_copy,)),
                            )

                            # SMEM dest: from tiled copy partition
                            tAsA_slice = cute.make_tensor(
                                tAsA_ktile[(None, i, k_sub)].iterator,
                                layout=cute.make_layout((self.ldgsts_elements_per_copy,)),
                            )

                            a_predicate_slice = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean
                            )
                            a_predicate_slice[0] = a_predicate_tensor[i]

                            cute.copy_atom_call(
                                a_atom_copy, tAgA_slice, tAsA_slice, pred=a_predicate_slice
                            )

                    a_pipeline.producer_commit(a_producer_state)

                    a_producer_state.advance()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            a_pipeline.producer_tail(a_producer_state)

        # =========================================================
        # TMA B load warp (warp 9)
        # =========================================================
        if warp_idx == self.tma_b_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info (only need first 4 elements for B)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                b_producer_state.reset_count()
                peek_b_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_b_empty_status = b_pipeline.producer_try_acquire(b_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    b_pipeline.producer_acquire(b_producer_state, peek_b_empty_status)

                    tBgB_k = tBgB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = tBsB[(None, b_producer_state.index)]

                    tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    b_producer_state.advance()
                    peek_b_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_b_empty_status = b_pipeline.producer_try_acquire(b_producer_state)

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            b_pipeline.producer_tail(b_producer_state)

        # =========================================================
        # MMA warp (warp 8) - consumes from separate A and B pipelines
        # =========================================================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek A and B buffers full
                a_consumer_state.reset_count()
                peek_a_full_status = cutlass.Boolean(1)
                if a_consumer_state.count < k_tile_cnt:
                    peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if a_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(k_tile_cnt):
                    if is_leader_cta:
                        # Wait for both A and B
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                        b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Release both A and B
                        a_pipeline.consumer_release(a_consumer_state)
                        b_pipeline.consumer_release(b_consumer_state)

                    # Peek next
                    a_consumer_state.advance()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                    b_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                # Commit ACC
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                acc_producer_state.advance()

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        # =========================================================
        # Epilogue warps (warps 0-3) - SwiGLU activation + TMA store
        # =========================================================
        if warp_idx <= self.epilog_warp_id[-1]:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                # Slice global C for this tile
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # SwiGLU: process pairs of subtiles (up, gate)
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                    real_subtile_idx = subtile_idx // 2

                    # Load up and gate accumulator subtiles from TMEM
                    tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, subtile_idx)]
                    tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, subtile_idx + 1)]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                    acc_vec_up = tTR_rAcc_up.load()
                    acc_vec_gate = tTR_rAcc_gate.load()

                    # SwiGLU: output = (alpha * up) * silu(alpha * gate)
                    tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                    for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                        acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(alpha_val)
                        acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(alpha_val)
                        tCompute[i] = acc_vec_up_alpha * silu_f32(acc_vec_gate_alpha, fastmath=True)

                    # Convert to C type and store to SMEM
                    acc_vec = tiled_copy_r2s.retile(tCompute).load()
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    num_prev_subtiles = num_prev_subtiles + 1
                    c_buffer = num_prev_subtiles % self.num_c_stage

                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    # TMA store C to global memory
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                # Release ACC
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Dealloc tensor memory
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

            # Wait for C store complete
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """Partition tensor memory for SwiGLU epilogue T2R copy.

        Returns separate up/gate register tensors for SwiGLU activation.
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # Separate register tensors for up and gate
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition shared memory for epilogue R2S copy."""
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Partition global memory for epilogue TMA store."""
        gC_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        num_smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int, int]:
        """Compute number of stages for A/B/C."""
        num_acc_stage = 2
        num_c_stage = 2
        num_tile_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)

        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid size using persistent tile scheduler on C (halved N)."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the MMA tiler and cluster shape are valid."""
        is_valid = True

        if mma_tiler_mn[0] not in (128, 256):
            is_valid = False
        if mma_tiler_mn[1] not in (64, 128, 192, 256):
            is_valid = False

        if (mma_tiler_mn[0] // cluster_shape_mn[0]) != 128:
            is_valid = False

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False

        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if this kernel can be used."""
        can_impl = True

        # Only K-major A and B supported
        if not (a_major == "k" and b_major == "k"):
            can_impl = False

        # BF16/FP16 only
        if ab_dtype not in {cutlass.BFloat16, cutlass.Float16}:
            can_impl = False

        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            can_impl = False

        if not Sm100Bf16ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn,
            cluster_shape_mn,
        ):
            can_impl = False

        # N must be even (SwiGLU halves it)
        if n % 2 != 0:
            can_impl = False

        # Check 16B alignment
        def check_16b_alignment(dtype, is_mode0_major, shape):
            major_idx = 0 if is_mode0_major else 1
            num_major = shape[major_idx]
            num_contiguous = 16 * 8 // dtype.width
            return num_major % num_contiguous == 0

        if (
            not check_16b_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_16b_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_16b_alignment(c_dtype, c_major == "m", (m, n // 2, l))
        ):
            can_impl = False

        return can_impl

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        top_k: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Wrapper that constructs tensors from pointers and calls the kernel."""
        interm_size = n // 2
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(token_id_mapping_ptr, layout=cute.make_layout((m,)))
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        return self(
            a,
            b,
            c,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )
