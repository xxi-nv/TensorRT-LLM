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
11-warp fused GEMM + AllReduce kernel for FlashMoE Expert Parallelism.

Extends the 7-warp finalize-fusion kernel with 4 dedicated AllReduce warps
that overlap NVLink ReduceScatter with GEMM epilogue. Architecture:

    Warp 0-3:  Epilogue   — TMEM → reg → scale → TMA store to staging buffer
                             + arrive(tile_barrier)
    Warp 4:    MMA        — tcgen05.mma grouped GEMM
    Warp 5:    TMA        — TMA load A/B → SMEM
    Warp 6:    Scheduler  — contiguous grouped tile scheduling
    Warp 7-10: AllReduce  — wait(tile_barrier) → multimem_ld_reduce →
                             multimem_st → arrive(completion_barrier)

The epilogue of tile N and the AllReduce of tile N-1 run concurrently,
hiding ReduceScatter latency behind the GEMM epilogue.

Two AllReduce paths:
  1. Multicast path (>= 2 GPUs with NVSwitch): multimem.ld_reduce + multimem.st
  2. IPC unicast path (2 GPUs): regular loads from IPC pointers + local add + store

When world_size == 1, the AR warps immediately exit (no-op), making this
kernel equivalent to the base finalize-fusion kernel for single-GPU operation.
"""

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)
from .multimem_helpers import (
    barrier_arrive_mc,
    barrier_try_wait_eq,
    bf16x2_to_f32x2,
    f32x2_to_bf16x2,
    ld_global_v4_b32,
    st_global_v4_b32,
    threadfence_system,
)
from .utils import (
    TRTLLM_ENABLE_PDL,
    atomic_add_func,
    blk_reduce_bf16,
    blk_reduce_fp16,
    blk_reduce_fp32,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    vectorized_atomic_add_bf16x8,
    vectorized_atomic_add_fp32x2,
)


class Sm100BlockScaledContiguousGroupedGemmAllReduceKernel(
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
):
    """11-warp fused GEMM + AllReduce kernel.

    Extends the base 7-warp finalize-fusion with 4 AllReduce warps.
    Inherits all MMA/TMA/scheduler/epilogue logic and adds:
      - AR warp IDs 7-10 (128 threads)
      - Tile barrier in shared memory for epilogue→AR synchronization
      - Multicast or IPC reduce across EP ranks
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_blkred: bool = False,
        raster_along_m: bool = False,
    ):
        super().__init__(
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_blkred=use_blkred,
            raster_along_m=raster_along_m,
        )

        # Add 4 AllReduce warps
        self.ar_warp_ids = (7, 8, 9, 10)
        self.num_ar_warps = len(self.ar_warp_ids)
        self.ar_threads = self.num_ar_warps * self.threads_per_warp  # 128

        # Override thread count: 7 base + 4 AR = 11 warps = 352 threads
        self.threads_per_cta = self.threads_per_warp * 11  # 352
        # threads_wo_sched stays the same (epilogue + MMA + TMA = 6 warps)
        # but we need all 10 non-scheduler warps to sync on tile_info
        self.threads_wo_sched = self.threads_per_warp * 10  # 320

        # Register budget: AR warps use uniform (64) regs
        self.num_regs_ar_warps = 64

        # Update cta_sync_barrier to include all 11 warps
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        # AR-specific barrier for tile ready notification
        # AR warps wait on this, epilogue warps arrive on this
        self.ar_tile_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=32 * (len(self.epilog_warp_id) + self.num_ar_warps),
        )

        # AR completion barrier: AR warps arrive when reduce is done
        self.ar_completion_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=32 * self.num_ar_warps,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        staging: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        # AllReduce parameters
        staging_mc_ptr: cutlass.Int64,
        out_mc_ptr: cutlass.Int64,
        tile_barrier_mc_ptr: cutlass.Int64,
        completion_barrier_mc_ptr: cutlass.Int64,
        staging_rank_stride: cutlass.Int64,
        out_rank_stride: cutlass.Int64,
        total_2d_tiles: cutlass.Int32,
        n_tiles: cutlass.Int32,
        staging_n: cutlass.Int32,
        rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        ar_strategy: cutlass.Constexpr,
        out: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute fused GEMM + AllReduce.

        Args:
            a: Input tensor A [permuted_m, k]
            b: Weight tensor B [n, k, l]
            staging: Staging buffer for FC2 output before reduce [permuted_m, n]
            sfa: Scale factor A
            sfb: Scale factor B
            tile_idx_to_expert_idx: Tile to expert mapping
            num_non_exiting_tiles: Number of valid tiles
            tile_idx_to_mn_limit: Per-tile MN limit
            alpha: Per-expert scaling
            max_active_clusters: Max active clusters
            stream: CUDA stream
            permuted_idx_to_expanded_idx: Permuted to expanded index mapping
            token_final_scales: Router scales [num_tokens, top_k]
            staging_mc_ptr: Base IPC VA for staging buffer (rank 0's offset)
            out_mc_ptr: Base IPC VA for reduced output
            tile_barrier_mc_ptr: Base IPC VA for tile barrier flags
            completion_barrier_mc_ptr: Base IPC VA for completion barrier flags
            staging_rank_stride: Byte stride between ranks in staging IPC buffer
            out_rank_stride: Byte stride between ranks in output IPC buffer
            total_2d_tiles: Total number of 2D tiles (M-tiles * N-tiles)
            n_tiles: Number of N-tiles (for 2D tile addressing)
            staging_n: N dimension of staging buffer (for 2D addressing)
            rank: Current EP rank
            world_size: Number of EP ranks
            ar_strategy: AllReduce strategy (0=batch, 1=overlapped)
            out: Final output tensor (for scatter-add after reduce)
            epilogue_op: Optional epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.out_dtype = staging.element_type
        self.sf_dtype = sfa.element_type
        self.final_scale_dtype = token_final_scales.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.gemm_output_layout = utils.LayoutEnum.ROW_MAJOR

        self.topK = token_final_scales.shape[1]
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor layouts
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA atoms (same as parent)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

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

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Handle sfb N=192 special case
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        self.tile_sched_params, grid = self._compute_grid(
            (a.shape[0], b.shape[0], a.shape[2]),
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Epilogue layout setup
        epi_tile_m = cute.size(self.epi_tile[0])
        epi_tile_n = cute.size(self.epi_tile[1])
        epi_tile_size = epi_tile_m * epi_tile_n
        num_epilogue_threads = 32 * len(self.epilog_warp_id)
        self.ttr_racc_size = epi_tile_size // num_epilogue_threads
        self.copy_size = self.cta_tile_shape_mnk[1] * (self.out_dtype.width // 8)

        if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
            self.epi_layout = cute.make_layout(
                shape=(self.ttr_racc_size // 8, 4, 2), stride=(8, 2, 1)
            )
            self.epi_loop_size = self.ttr_racc_size // 8
            self.element_offset = 8
        elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
            self.epi_layout = cute.make_layout(shape=(self.ttr_racc_size // 2, 2), stride=(2, 1))
            self.epi_loop_size = self.ttr_racc_size // 2
            self.element_offset = 2
        else:
            self.epi_layout = cute.make_layout(shape=(self.ttr_racc_size,), stride=(1,))
            self.epi_loop_size = self.ttr_racc_size
            self.element_offset = 1

        # Define shared storage including AR tile counter
        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            if cutlass.const_expr(self.use_blkred):
                sC: cute.struct.Align[
                    cute.struct.MemRange[self.out_dtype, cute.cosize(self.c_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            staging,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.epi_layout,
            self.topK,
            self.tile_sched_params,
            # AR params
            staging_mc_ptr,
            out_mc_ptr,
            tile_barrier_mc_ptr,
            completion_barrier_mc_ptr,
            staging_rank_stride,
            out_rank_stride,
            total_2d_tiles,
            n_tiles,
            staging_n,
            rank,
            world_size,
            ar_strategy,
            out,
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
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        staging: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        epi_layout: cute.Layout,
        topK: cutlass.Int32,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        # AllReduce parameters
        staging_mc_ptr: cutlass.Int64,
        out_mc_ptr: cutlass.Int64,
        tile_barrier_mc_ptr: cutlass.Int64,
        completion_barrier_mc_ptr: cutlass.Int64,
        staging_rank_stride: cutlass.Int64,
        out_rank_stride: cutlass.Int64,
        total_2d_tiles: cutlass.Int32,
        n_tiles: cutlass.Int32,
        staging_n: cutlass.Int32,
        rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        ar_strategy: cutlass.Constexpr,
        out: cute.Tensor,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU kernel: 11-warp fused GEMM + AllReduce."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors (TMA warp only)
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # CTA/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        tidx, _, _ = cute.arch.thread_idx()

        # Alloc shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize pipelines (same as parent for base warps)
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

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

        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        # Consumer group: original 6 warps + no AR warps on tile_info
        # (AR warps use a separate mechanism)
        tile_info_consumer_threads = self.threads_per_warp * 6  # epi(4) + mma(1) + tma(1)
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            tile_info_consumer_threads,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # TMEM allocator
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
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        if cutlass.const_expr(self.use_blkred):
            sC = storage.sC.get_tensor(c_smem_layout_staged)

        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        # Multicast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        # Global tensor partitioning
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # MMA partitions
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        # TMA partitions
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # MMA fragment tensors
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Staging buffer partition (for epilogue writes)
        gC_mnl = cute.local_tile(
            staging, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCgC = thr_mma.partition_C(gC_mnl)

        # Cluster wait
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        griddepcontrol_wait()

        # ===================================================================
        # Warp dispatch
        # ===================================================================

        # --- Scheduler warp (6) ---
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

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
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
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
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
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            # Sentinel tile (not valid)
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # --- TMA warp (5) ---
        if warp_idx == self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

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
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, 0)]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # Load A, B, SFA, SFB via TMA
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                # Next tile
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

            ab_pipeline.producer_tail(ab_producer_state)

        # --- MMA warp (4) ---
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

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
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                acc_producer_state.advance()

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

            acc_pipeline.producer_tail(acc_producer_state)

        # --- Epilogue warps (0-3) ---
        # Modified epilogue: writes to staging buffer (contiguous), then signals AR warps
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.out_dtype)
            if cutlass.const_expr(self.use_blkred):
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    epi_tidx, tTR_rC, sC, tiled_copy_t2r
                )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            token_idx = cutlass.Int32(0)
            token_scale = self.final_scale_dtype(0.0)

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

            epi_tile_count = cutlass.Int32(0)

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                permuted_row = tile_m_start + epi_tidx
                expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                is_valid_row = permuted_row < tile_info[4]

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                if is_valid_row:
                    token_idx = expanded_idx // topK
                    topk_idx = expanded_idx % topK
                    token_scale = token_final_scales[(token_idx, topk_idx)]
                    alpha_val = alpha_val * token_scale

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = subtile_cnt - 1 - subtile_idx

                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    acc_vec = tTR_rAcc.load()
                    acc_vec_final = alpha_val * acc_vec

                    # Write to staging buffer (contiguous, not scatter-add)
                    # This is the key difference from the base finalize-fusion kernel.
                    # For single-GPU (world_size==1), we still scatter-add directly.
                    if cutlass.const_expr(world_size > 1):
                        # Write to contiguous staging buffer
                        if cutlass.const_expr(self.use_blkred):
                            tRS_rC.store(acc_vec_final.to(self.out_dtype))
                            if is_valid_row:
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC[(None, None, real_subtile_idx, None)],
                                )
                        else:
                            tTR_rC.store(acc_vec_final.to(self.out_dtype))
                            if is_valid_row:
                                rOut_epi = cute.make_tensor(tTR_rC.iterator, epi_layout)
                                base_coord_n = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[
                                    1
                                ] + real_subtile_idx * cute.size(tTR_rC)
                                # Write to staging buffer at permuted_row position
                                staging_out = cute.domain_offset(
                                    (permuted_row, 0, 0),
                                    staging,
                                )
                                for index in cutlass.range(self.epi_loop_size, unroll_full=True):
                                    coord_n = base_coord_n + index * self.element_offset
                                    staging_out_offset = cute.domain_offset(
                                        (0, coord_n, 0), staging_out
                                    )
                                    if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
                                        rOut_epi_packed = rOut_epi[index, None, None]
                                        vectorized_atomic_add_bf16x8(
                                            rOut_epi_packed, staging_out_offset
                                        )
                                    elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
                                        rOut_epi_packed = rOut_epi[index, None]
                                        vectorized_atomic_add_fp32x2(
                                            rOut_epi_packed, staging_out_offset
                                        )
                                    else:
                                        rOut_epi_packed = rOut_epi[index]
                                        atomic_add_func(rOut_epi_packed, staging_out_offset)
                    else:
                        # Single-GPU: scatter-add directly (same as base)
                        if cutlass.const_expr(self.use_blkred):
                            tRS_rC.store(acc_vec_final.to(self.out_dtype))
                            if is_valid_row:
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC[(None, None, real_subtile_idx, None)],
                                )
                        else:
                            tTR_rC.store(acc_vec_final.to(self.out_dtype))
                            if is_valid_row:
                                rOut_epi = cute.make_tensor(tTR_rC.iterator, epi_layout)
                                base_coord_n = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[
                                    1
                                ] + real_subtile_idx * cute.size(tTR_rC)
                                scatter_out = cute.domain_offset(
                                    (token_idx, 0, 0),
                                    out,
                                )
                                for index in cutlass.range(self.epi_loop_size, unroll_full=True):
                                    coord_n = base_coord_n + index * self.element_offset
                                    scatter_out_offset = cute.domain_offset(
                                        (0, coord_n, 0), scatter_out
                                    )
                                    if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
                                        rOut_epi_packed = rOut_epi[index, None, None]
                                        vectorized_atomic_add_bf16x8(
                                            rOut_epi_packed, scatter_out_offset
                                        )
                                    elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
                                        rOut_epi_packed = rOut_epi[index, None]
                                        vectorized_atomic_add_fp32x2(
                                            rOut_epi_packed, scatter_out_offset
                                        )
                                    else:
                                        rOut_epi_packed = rOut_epi[index]
                                        atomic_add_func(rOut_epi_packed, scatter_out_offset)

                if cutlass.const_expr(self.use_blkred):
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                if cutlass.const_expr(not self.overlapping_accum):
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                if cutlass.const_expr(self.use_blkred):
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    if is_valid_row:
                        if cutlass.const_expr(world_size > 1):
                            coord_n = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                            staging_out_offset = cute.domain_offset(
                                (permuted_row, coord_n, 0), staging
                            )
                            if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
                                blk_reduce_bf16(
                                    staging_out_offset,
                                    sC[epi_tidx, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                            elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
                                blk_reduce_fp32(
                                    staging_out_offset,
                                    sC[epi_tidx, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                            elif cutlass.const_expr(self.out_dtype == cutlass.Float16):
                                blk_reduce_fp16(
                                    staging_out_offset,
                                    sC[epi_tidx, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                        else:
                            coord_n = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                            scatter_out_offset = cute.domain_offset((token_idx, coord_n, 0), out)
                            if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
                                blk_reduce_bf16(
                                    scatter_out_offset,
                                    sC[epi_tidx, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                            elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
                                blk_reduce_fp32(
                                    scatter_out_offset,
                                    sC[epi_tidx, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                            elif cutlass.const_expr(self.out_dtype == cutlass.Float16):
                                blk_reduce_fp16(
                                    scatter_out_offset,
                                    sC[epi_tidx, None, 0],
                                    cutlass.Int32(self.copy_size),
                                )
                    self.epilog_sync_barrier.arrive_and_wait()

                # Signal AR warps that this tile is ready
                if cutlass.const_expr(world_size > 1):
                    # Use global 2D tile index for barrier addressing.
                    # Each tile maps to a unique barrier slot regardless
                    # of which CTA processes it.
                    global_tile_idx = mma_tile_coord_mnl[0] * n_tiles + mma_tile_coord_mnl[1]
                    threadfence_system()
                    if epi_tidx == 0:
                        barrier_arrive_mc(tile_barrier_mc_ptr + global_tile_idx * 4)

                epi_tile_count = epi_tile_count + 1

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

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # --- AllReduce warps (7-10) ---
        if warp_idx >= 7:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_ar_warps)

            if cutlass.const_expr(world_size > 1):
                # AR thread index across 4 warps: 0-127
                ar_thread_idx = (warp_idx - 7) * 32 + (tidx % 32)

                # Elements per tile: cta_tile_m * cta_tile_n (bf16 elements)
                cta_tile_m = cutlass.Int32(self.cta_tile_shape_mnk[0])
                cta_tile_n = cutlass.Int32(self.cta_tile_shape_mnk[1])
                elements_per_tile = cta_tile_m * cta_tile_n

                # Each 128-bit load covers 4 b32 words = 8 bf16 elements.
                vec_width = cutlass.Int32(8)  # 8 bf16 = 128 bits
                num_ar_threads = cutlass.Int32(128)  # 4 warps * 32 threads

                if cutlass.const_expr(ar_strategy == 0):
                    # --------------------------------------------------
                    # Strategy 0 (batch): Wait ALL tiles, then linear
                    # reduce the entire staging buffer at once.
                    # Simple but no overlap with epilogue.
                    # --------------------------------------------------
                    wait_idx = cutlass.Int32(0)
                    while wait_idx < total_2d_tiles:
                        tile_barrier_addr = tile_barrier_mc_ptr + wait_idx * 4
                        flag_val = barrier_try_wait_eq(tile_barrier_addr, world_size)
                        while flag_val < world_size:
                            flag_val = barrier_try_wait_eq(tile_barrier_addr, world_size)
                        wait_idx = wait_idx + 1

                    total_elements = total_2d_tiles * elements_per_tile
                    elems_per_thread = total_elements // num_ar_threads

                    for i in cutlass.range(0, elems_per_thread, vec_width):
                        elem_offset = ar_thread_idx * elems_per_thread + i
                        byte_offset = elem_offset * 2  # bf16

                        rank0_addr = staging_mc_ptr + byte_offset
                        w0, w1, w2, w3 = ld_global_v4_b32(rank0_addr)
                        acc0, acc1 = bf16x2_to_f32x2(w0)
                        acc2, acc3 = bf16x2_to_f32x2(w1)
                        acc4, acc5 = bf16x2_to_f32x2(w2)
                        acc6, acc7 = bf16x2_to_f32x2(w3)

                        for rank_i in cutlass.range_constexpr(1, world_size):
                            rank_addr = staging_mc_ptr + rank_i * staging_rank_stride + byte_offset
                            rw0, rw1, rw2, rw3 = ld_global_v4_b32(rank_addr)
                            f0, f1 = bf16x2_to_f32x2(rw0)
                            f2, f3 = bf16x2_to_f32x2(rw1)
                            f4, f5 = bf16x2_to_f32x2(rw2)
                            f6, f7 = bf16x2_to_f32x2(rw3)
                            acc0 = acc0 + f0
                            acc1 = acc1 + f1
                            acc2 = acc2 + f2
                            acc3 = acc3 + f3
                            acc4 = acc4 + f4
                            acc5 = acc5 + f5
                            acc6 = acc6 + f6
                            acc7 = acc7 + f7

                        out_w0 = f32x2_to_bf16x2(acc0, acc1)
                        out_w1 = f32x2_to_bf16x2(acc2, acc3)
                        out_w2 = f32x2_to_bf16x2(acc4, acc5)
                        out_w3 = f32x2_to_bf16x2(acc6, acc7)

                        out_addr = out_mc_ptr + rank * out_rank_stride + byte_offset
                        st_global_v4_b32(out_addr, out_w0, out_w1, out_w2, out_w3)

                    threadfence_system()
                    if ar_thread_idx == 0:
                        barrier_arrive_mc(completion_barrier_mc_ptr)

                elif cutlass.const_expr(ar_strategy == 1):
                    # --------------------------------------------------
                    # Strategy 1 (overlapped): Wait per-tile barrier,
                    # then reduce that tile using 2D addressing.
                    # Overlaps reduce of tile N with epilogue of tile N+1.
                    # All CTAs' AR warps process all tiles (redundant but
                    # correct — all produce identical results).
                    # --------------------------------------------------
                    elems_per_thread_tile = elements_per_tile // num_ar_threads

                    tile_idx = cutlass.Int32(0)
                    while tile_idx < total_2d_tiles:
                        # Wait for this specific tile
                        tile_barrier_addr = tile_barrier_mc_ptr + tile_idx * 4
                        flag_val = barrier_try_wait_eq(tile_barrier_addr, world_size)
                        while flag_val < world_size:
                            flag_val = barrier_try_wait_eq(tile_barrier_addr, world_size)

                        # 2D tile coordinates from linear tile index
                        m_tile_idx = tile_idx // n_tiles
                        n_tile_idx = tile_idx - m_tile_idx * n_tiles
                        tile_row_start = m_tile_idx * cta_tile_m
                        tile_col_start = n_tile_idx * cta_tile_n

                        for i in cutlass.range(0, elems_per_thread_tile, vec_width):
                            flat_idx = ar_thread_idx * elems_per_thread_tile + i
                            row_in_tile = flat_idx // cta_tile_n
                            col_in_tile = flat_idx - row_in_tile * cta_tile_n
                            row = tile_row_start + row_in_tile
                            col = tile_col_start + col_in_tile
                            # Byte offset in row-major [m, n] bf16 buffer
                            byte_offset = (row * staging_n + col) * 2

                            rank0_addr = staging_mc_ptr + byte_offset
                            w0, w1, w2, w3 = ld_global_v4_b32(rank0_addr)
                            acc0, acc1 = bf16x2_to_f32x2(w0)
                            acc2, acc3 = bf16x2_to_f32x2(w1)
                            acc4, acc5 = bf16x2_to_f32x2(w2)
                            acc6, acc7 = bf16x2_to_f32x2(w3)

                            for rank_i in cutlass.range_constexpr(1, world_size):
                                rank_addr = (
                                    staging_mc_ptr + rank_i * staging_rank_stride + byte_offset
                                )
                                rw0, rw1, rw2, rw3 = ld_global_v4_b32(rank_addr)
                                f0, f1 = bf16x2_to_f32x2(rw0)
                                f2, f3 = bf16x2_to_f32x2(rw1)
                                f4, f5 = bf16x2_to_f32x2(rw2)
                                f6, f7 = bf16x2_to_f32x2(rw3)
                                acc0 = acc0 + f0
                                acc1 = acc1 + f1
                                acc2 = acc2 + f2
                                acc3 = acc3 + f3
                                acc4 = acc4 + f4
                                acc5 = acc5 + f5
                                acc6 = acc6 + f6
                                acc7 = acc7 + f7

                            out_w0 = f32x2_to_bf16x2(acc0, acc1)
                            out_w1 = f32x2_to_bf16x2(acc2, acc3)
                            out_w2 = f32x2_to_bf16x2(acc4, acc5)
                            out_w3 = f32x2_to_bf16x2(acc6, acc7)

                            out_addr = out_mc_ptr + rank * out_rank_stride + byte_offset
                            st_global_v4_b32(
                                out_addr,
                                out_w0,
                                out_w1,
                                out_w2,
                                out_w3,
                            )

                        tile_idx = tile_idx + 1

                    threadfence_system()
                    if ar_thread_idx == 0:
                        barrier_arrive_mc(completion_barrier_mc_ptr)
            # else: world_size == 1, AR warps do nothing

        griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        staging_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        # AR pointers
        staging_mc_ptr: cutlass.Int64,
        out_mc_ptr: cutlass.Int64,
        tile_barrier_mc_ptr: cutlass.Int64,
        completion_barrier_mc_ptr: cutlass.Int64,
        staging_rank_stride: cutlass.Int64,
        out_rank_stride: cutlass.Int64,
        out_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        ar_strategy: cutlass.Constexpr,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        scale_k = k // scaling_vector_size
        num_tiles = m // tile_size
        # Total 2D tiles for AR warps: M-tiles * N-tiles
        # Use self.mma_tiler[1] (set in __init__) because cta_tile_shape_mnk
        # is only available after _setup_attributes() runs inside __call__().
        cta_n = cutlass.Int64(self.mma_tiler[1])
        n_tiles_i64 = (n + cta_n - 1) // cta_n
        total_2d_tiles = cutlass.Int32(num_tiles * n_tiles_i64)
        n_tiles = cutlass.Int32(n_tiles_i64)
        staging_n = cutlass.Int32(n)

        a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        # Staging buffer: contiguous [permuted_m, n]
        staging = cute.make_tensor(
            staging_ptr, layout=cute.make_ordered_layout((m, n, 1), order=(1, 0, 2))
        )
        # Output for scatter-add (or for single-GPU path)
        out = cute.make_tensor(
            out_ptr, layout=cute.make_ordered_layout((num_tokens, n, 1), order=(1, 0, 2))
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
        )

        return self(
            a,
            b,
            staging,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            staging_mc_ptr=staging_mc_ptr,
            out_mc_ptr=out_mc_ptr,
            tile_barrier_mc_ptr=tile_barrier_mc_ptr,
            completion_barrier_mc_ptr=completion_barrier_mc_ptr,
            staging_rank_stride=staging_rank_stride,
            out_rank_stride=out_rank_stride,
            total_2d_tiles=total_2d_tiles,
            n_tiles=n_tiles,
            staging_n=staging_n,
            rank=rank,
            world_size=world_size,
            ar_strategy=ar_strategy,
            out=out,
            epilogue_op=epilogue_op,
        )
