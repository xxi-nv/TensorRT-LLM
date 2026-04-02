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

"""FlashMoE Fully-Fused Two-Phase Persistent Kernel for Blackwell.

This module implements a single persistent CuTe DSL kernel that fuses FC1 and FC2
phases of a Mixture-of-Experts FFN layer on Blackwell GPUs.

Architecture (per CTA, 11 warps = 352 threads):
  Phase 1 -- FC1:
    LDGSTS warps (4-7) gather A+SFA, TMA warp (9) loads B+SFB weights,
    MMA warp (8) computes GEMM, epilogue warps (0-3) apply SwiGLU + NVFP4 quant,
    scheduler warp (10) dispatches FC1 tiles.

  Phase transition: CTA-wide named barrier (all warps sync).

  Phase 2 -- FC2:
    TMA warp (9) loads A (FC1 output) + B (w2) + SFA + SFB,
    MMA warp (8) computes GEMM, epilogue warps (0-3) apply scale + scatter-add,
    scheduler warp (10) dispatches FC2 tiles. Warps 4-7 idle.

SMEM data buffers (sA, sB, sSFA, sSFB) are shared between phases since both
use the same NVFP4 data type with identical MMA tile configuration.

Current version (v1): No in-kernel IPC gather or AllReduce. Input is
pre-gathered, AllReduce is done via NCCL after kernel completion.
"""

from typing import Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import blockscaled_utils, cpasync, sm100_utils, tcgen05

from .custom_pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineCpAsyncUmma,
    PipelineTmaUmma,
    PipelineUmmaAsync,
)
from .utils import (
    TRTLLM_ENABLE_PDL,
    atomic_add_func,
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    silu_f32,
    vectorized_atomic_add_bf16x8,
    vectorized_atomic_add_fp32x2,
)


class FlashMoeFusedKernel:
    """Two-phase persistent cuteDSL kernel: FC1 (gather+GEMM+SwiGLU) -> FC2 (GEMM+scatter-add).

    Warp layout (1CTA mode, mma_tiler_mn=(128,128)):
      0-3   Epilogue   -- SwiGLU+quant (FC1) / scale+scatter-add (FC2)
      4-7   LDGSTS     -- Gather A+SFA (FC1) / idle (FC2)
      8     MMA        -- tcgen05.mma GEMM (both phases)
      9     TMA        -- TMA B+SFB (FC1) / TMA A+B+SFA+SFB (FC2)
      10    Scheduler  -- Dispatch tiles (both phases, sequential)
    """

    PHASE_FC1 = 0
    PHASE_FC2 = 1

    def __init__(
        self,
        sf_vec_size: int = 16,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        vectorized_f32: bool = True,
        topk: int = 8,
        raster_along_m: bool = False,
    ):
        self.sf_vec_size = sf_vec_size
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = False  # 1CTA mode only
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)  # K deferred to _setup_attributes
        self.raster_along_m = raster_along_m
        self.cta_group = tcgen05.CtaGroup.ONE
        self.occupancy = 1

        # Warp ID assignment (11 warps, 352 threads)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10

        self.threads_per_warp = 32
        self.num_warps = 11
        self.threads_per_cta = self.threads_per_warp * self.num_warps  # 352
        self.warps_wo_sched = len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_b_warp_id,
                *self.ldgsts_a_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Named barriers
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len(self.epilog_warp_id)
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.threads_per_warp
        )
        # Phase transition barrier (all warps sync between FC1 and FC2)
        self.phase_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=self.threads_per_cta)

        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512
        self.vectorized_f32 = vectorized_f32

    # =================================================================
    # Attribute setup (called from __call__ after dtypes are known)
    # =================================================================

    def _setup_attributes(self):
        """Set up MMA config, tile shapes, SMEM layouts, and TMEM columns.

        Both FC1 and FC2 use the same NVFP4 MMA configuration, so the MMA tiler,
        tile shapes, and mainloop SMEM layouts are shared between phases.
        Only the epilogue configuration differs (FC1: SwiGLU+quant, FC2: scatter-add).
        """
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0],
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Create tiled MMA (identical for both phases)
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
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Resolve K dimension from MMA instruction
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfa = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # FC1 output tile (N/2 due to SwiGLU halving)
        self.fc1_mma_tiler_c = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1] // 2,
            mma_inst_shape_k * mma_inst_tile_k,
        )

        # CTA tile shapes (shared between FC1 and FC2 mainloop)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // atom_thr_size,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // atom_thr_size,
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // atom_thr_size,
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )
        self.fc1_cta_tile_shape_mnk_c = (
            self.fc1_mma_tiler_c[0] // atom_thr_size,
            self.fc1_mma_tiler_c[1],
            self.fc1_mma_tiler_c[2],
        )

        # Cluster layout (trivial for (1,1))
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # FC1 epilogue tile (SwiGLU + NVFP4 quantization)
        self.fc1_epi_tile = (128, 64)
        self.fc1_epi_tile_cnt = (
            self.fc1_cta_tile_shape_mnk_c[0] // self.fc1_epi_tile[0],
            self.fc1_cta_tile_shape_mnk_c[1] // self.fc1_epi_tile[1],
        )

        # FC2 epilogue tile (scatter-add with bf16 output)
        self.fc2_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.fc2_gemm_output_layout,
            self.fc2_out_dtype,
        )
        self.fc2_epi_tile_n = cute.size(self.fc2_epi_tile[1])

        # Pipeline stages (computed from FC1 SMEM needs which include sC)
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.fc1_num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.fc1_epi_tile,
            self.fc1_c_dtype,
            self.fc1_c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
        )

        # Mainloop SMEM layouts (shared between FC1 and FC2)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        # FC1 epilogue SMEM layout (TMA store staging for NVFP4 output)
        self.fc1_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.fc1_c_dtype,
            self.fc1_c_layout,
            self.fc1_epi_tile,
            self.fc1_num_c_stage,
        )

        # Accumulator overlap and TMEM columns
        self.overlapping_accum = self.num_acc_stage == 1

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.epi_tile_n_required = 2 * cute.size(self.fc1_epi_tile[1])
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n_required

        # FC2 TMEM final offset
        self.fc2_tmem_final_offset = 384
        self.fc2_iter_acc_early_release = self.num_sf_tmem_cols // self.fc2_epi_tile_n

    # =================================================================
    # Static helpers
    # =================================================================

    @staticmethod
    def can_implement(
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        ep_size: int,
        sf_vec_size: int = 16,
    ) -> bool:
        """Check if FlashMoE fused kernel can run on this configuration."""
        from tensorrt_llm._utils import get_sm_version

        sm = get_sm_version()
        if sm not in (100, 103):
            return False
        if ep_size <= 1:
            return False
        if hidden_size % 32 != 0:
            return False
        if intermediate_size % 128 != 0:
            return False
        if num_experts % ep_size != 0:
            return False
        return True

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int, int]:
        """Compute pipeline stage counts for SMEM allocation.

        Extra 2048B reserved for dual-phase pipeline barrier storage
        (FC1 + FC2 mbar storage) vs 1024B in the single-phase kernels.
        """
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 2
        num_tile_stage = 2

        a_smem_1 = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_1 = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_1 = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        sfb_smem_1 = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        c_smem_1 = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)

        ab_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_1)
            + cute.size_in_bytes(b_dtype, b_smem_1)
            + cute.size_in_bytes(sf_dtype, sfa_smem_1)
            + cute.size_in_bytes(sf_dtype, sfb_smem_1)
        )
        # Extra space for dual-phase pipeline mbar storage
        mbar_helpers_bytes = 2048
        c_per_stage = cute.size_in_bytes(c_dtype, c_smem_1)
        c_total = c_per_stage * num_c_stage

        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_total)
        ) // ab_per_stage

        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_total)
        ) // (occupancy * c_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute persistent tile scheduler params and grid shape."""
        (m, n, l) = gemm_shape  # noqa: E741
        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l
        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[
        cpasync.CopyBulkTensorTileG2SMulticastOp,
        cpasync.CopyBulkTensorTileG2SOp,
    ]:
        """Select TMA copy atom based on SM count and multicast flag."""
        if atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        if atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        if atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        raise ValueError(f"Invalid: atom_sm_cnt={atom_sm_cnt}, mcast={mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """Reciprocal of max absolute value for NVFP4/FP8 quantization."""
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    # =================================================================
    # Mainloop helper (used by MMA warp in both phases)
    # =================================================================

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition scale factor tensors for SMEM -> TMEM copy.

        Used by MMA warp to copy SFA/SFB from SMEM to TMEM before GEMM.
        Same pattern for both FC1 and FC2 phases.
        """
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # =================================================================
    # FC1 epilogue helpers (SwiGLU + NVFP4 quant + TMA store)
    # =================================================================

    def fc1_epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple:
        """Partition TMEM accumulator for FC1 SwiGLU epilogue (up + gate)."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.fc1_c_layout,
            self.fc1_c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def fc1_epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple:
        """Partition register/SMEM for FC1 epilogue R2S copy."""
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.fc1_c_layout,
            self.fc1_c_dtype,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def fc1_epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tma_atom_c: cute.CopyAtom,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple:
        """Partition SMEM/GMEM for FC1 epilogue TMA S2G store."""
        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma,
            gC_for_tma,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    # =================================================================
    # FC2 epilogue helper (scatter-add)
    # =================================================================

    def fc2_epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple:
        """Partition TMEM accumulator for FC2 scatter-add epilogue."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.fc2_gemm_output_layout,
            self.fc2_out_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    # =================================================================
    # __call__: Setup TMA atoms, SharedStorage, launch kernel
    # =================================================================

    @cute.jit
    def __call__(
        self,
        # FC1 tensors
        fc1_a: cute.Tensor,  # [M, K1, 1] NVFP4 gathered input
        fc1_b: cute.Tensor,  # [2*N1, K1, L] NVFP4 w3w1 weights
        fc1_c: cute.Tensor,  # [M, N1, 1] NVFP4 FC1 output buffer
        fc1_sfa: cute.Tensor,  # [M, K1/sf, 1] scale factors for A
        fc1_sfb: cute.Tensor,  # [..., L] scale factors for w3w1
        fc1_sfc: Optional[cute.Tensor],  # [M, N1/sf, 1] FC1 output scales
        fc1_norm_const: Optional[cute.Tensor],  # normalization constant
        # FC2 tensors (FC2 A = fc1_c, FC2 SFA = fc1_sfc)
        fc2_a: cute.Tensor,  # [M, K2, L] NVFP4 FC1 output (same data as fc1_c)
        fc2_b: cute.Tensor,  # [N2, K2, L] NVFP4 w2 weights
        fc2_out: cute.Tensor,  # [M', N2] BFloat16 output buffer
        fc2_sfa: cute.Tensor,  # [..., L] scale factors for FC2 A
        fc2_sfb: cute.Tensor,  # [..., L] scale factors for w2
        # Shared metadata (from moe_sort, M-tile based)
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        # Per-phase metadata
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        # Config
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Setup TMA atoms for both phases, define SharedStorage, launch kernel."""
        # --- Set data types ---
        self.a_dtype: Type[cutlass.Numeric] = fc1_a.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_b.element_type
        self.fc1_c_dtype: Type[cutlass.Numeric] = fc1_c.element_type
        self.fc2_out_dtype: Type[cutlass.Numeric] = fc2_out.element_type
        self.sf_dtype: Type[cutlass.Numeric] = fc1_sfa.element_type
        self.fc2_final_scale_dtype: Type[cutlass.Numeric] = token_final_scales.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(fc1_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(fc1_b).mma_major_mode()
        self.fc1_c_layout = utils.LayoutEnum.from_tensor(fc1_c)
        self.fc2_gemm_output_layout = utils.LayoutEnum.ROW_MAJOR

        self.generate_sfc = fc1_sfc is not None and fc1_norm_const is not None

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/B dtype must match: {self.a_dtype} != {self.b_dtype}")

        # --- Setup attributes (MMA, tiles, SMEM layouts) ---
        self._setup_attributes()

        # --- Reshape scale factor tensors ---
        fc1_sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(fc1_b.shape, self.sf_vec_size)
        fc1_sfb = cute.make_tensor(fc1_sfb.iterator, fc1_sfb_layout)

        if cutlass.const_expr(self.generate_sfc):
            fc1_sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(fc1_c.shape, self.sf_vec_size)
            fc1_sfc = cute.make_tensor(fc1_sfc.iterator, fc1_sfc_layout)

        fc2_sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(fc2_a.shape, self.sf_vec_size)
        fc2_sfa = cute.make_tensor(fc2_sfa.iterator, fc2_sfa_layout)

        fc2_sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(fc2_b.shape, self.sf_vec_size)
        fc2_sfb = cute.make_tensor(fc2_sfb.iterator, fc2_sfb_layout)

        # --- Create tiled MMA (needed for TMA atom creation) ---
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
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # ============================================================
        # FC1 TMA atoms: B (w3w1), SFB, C (TMA store for NVFP4 output)
        # ============================================================
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_b, tma_tensor_fc1_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            fc1_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_sfb, tma_tensor_fc1_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            fc1_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.fc1_num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

        # FC1 TMA store for C (NVFP4 output)
        fc1_epi_smem_layout = cute.slice_(self.fc1_c_smem_layout_staged, (None, None, 0))
        tma_atom_fc1_c, tma_tensor_fc1_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            fc1_c,
            fc1_epi_smem_layout,
            self.fc1_epi_tile,
        )

        # ============================================================
        # FC2 TMA atoms: A (FC1 output), B (w2), SFA, SFB
        # ============================================================
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc2_a, tma_tensor_fc2_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            fc2_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_fc2_b, tma_tensor_fc2_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            fc2_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc2_sfa, tma_tensor_fc2_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            fc2_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        tma_atom_fc2_sfb, tma_tensor_fc2_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            fc2_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.fc2_num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # ============================================================
        # FC2 epilogue layout (scatter-add vectorization)
        # ============================================================
        fc2_epi_tile_m = cute.size(self.fc2_epi_tile[0])
        fc2_epi_tile_n = cute.size(self.fc2_epi_tile[1])
        fc2_epi_tile_size = fc2_epi_tile_m * fc2_epi_tile_n
        num_epilogue_threads = 32 * len(self.epilog_warp_id)
        self.fc2_ttr_racc_size = fc2_epi_tile_size // num_epilogue_threads
        self.fc2_copy_size = self.cta_tile_shape_mnk[1] * (self.fc2_out_dtype.width // 8)

        if cutlass.const_expr(self.fc2_out_dtype == cutlass.BFloat16):
            self.fc2_epi_layout = cute.make_layout(
                shape=(self.fc2_ttr_racc_size // 8, 4, 2), stride=(8, 2, 1)
            )
            self.fc2_epi_loop_size = self.fc2_ttr_racc_size // 8
            self.fc2_element_offset = 8
        elif cutlass.const_expr(self.fc2_out_dtype == cutlass.Float32):
            self.fc2_epi_layout = cute.make_layout(
                shape=(self.fc2_ttr_racc_size // 2, 2), stride=(2, 1)
            )
            self.fc2_epi_loop_size = self.fc2_ttr_racc_size // 2
            self.fc2_element_offset = 2
        else:
            self.fc2_epi_layout = cute.make_layout(shape=(self.fc2_ttr_racc_size,), stride=(1,))
            self.fc2_epi_loop_size = self.fc2_ttr_racc_size
            self.fc2_element_offset = 1

        # ============================================================
        # Compute grid (use max of FC1 and FC2 tile counts)
        # ============================================================
        # FC1 grid: based on FC1 output shape (M, intermediate_size)
        self.fc1_tile_sched_params, fc1_grid = self._compute_grid(
            (fc1_c.shape[0], fc1_c.shape[1], 1),
            self.fc1_cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )
        # FC2 grid: based on FC2 GEMM shape (M, hidden_size)
        self.fc2_tile_sched_params, fc2_grid = self._compute_grid(
            (fc2_a.shape[0], fc2_b.shape[0], fc2_a.shape[2]),
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )
        # Use max grid to ensure enough CTAs for both phases
        grid = (max(fc1_grid[0], fc2_grid[0]), 1, 1)

        # ============================================================
        # SharedStorage definition (dual-phase pipeline barriers)
        # ============================================================
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # FC1 tile info (bidx, bidy, expert_idx, valid, mn_limit)
            fc1_sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            # FC2 tile info
            fc2_sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            # FC1 pipeline barriers
            fc1_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc1_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc1_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            fc1_tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            # FC2 pipeline barriers
            fc2_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            fc2_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            fc2_tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            # Shared TMEM management
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # FC1 epilogue staging (TMA store for NVFP4 output)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.fc1_c_dtype,
                    cute.cosize(self.fc1_c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # Mainloop data buffers (shared between FC1 and FC2)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfa_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfb_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # ============================================================
        # Launch the fused kernel
        # ============================================================
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            # FC1 tensors
            fc1_a,
            tma_atom_fc1_b,
            tma_tensor_fc1_b,
            fc1_sfa,
            tma_atom_fc1_sfb,
            tma_tensor_fc1_sfb,
            tma_atom_fc1_c,
            tma_tensor_fc1_c,
            fc1_sfc,
            fc1_norm_const,
            # FC2 tensors
            tma_atom_fc2_a,
            tma_tensor_fc2_a,
            tma_atom_fc2_b,
            tma_tensor_fc2_b,
            tma_atom_fc2_sfa,
            tma_tensor_fc2_sfa,
            tma_atom_fc2_sfb,
            tma_tensor_fc2_sfb,
            fc2_out,
            # Shared metadata
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            fc1_alpha,
            fc2_alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            # Layouts and config
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.fc1_c_smem_layout_staged,
            self.fc1_epi_tile,
            self.fc2_epi_tile,
            self.fc2_epi_layout,
            self.fc1_tile_sched_params,
            self.fc2_tile_sched_params,
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

    # =================================================================
    # GPU device kernel: two-phase persistent GEMM
    # =================================================================

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # FC1 tensors
        fc1_mA_mkl: cute.Tensor,
        tma_atom_fc1_b: cute.CopyAtom,
        fc1_mB_nkl: cute.Tensor,
        fc1_mSFA_mkl: cute.Tensor,
        tma_atom_fc1_sfb: cute.CopyAtom,
        fc1_mSFB_nkl: cute.Tensor,
        tma_atom_fc1_c: cute.CopyAtom,
        fc1_mC_mnl: cute.Tensor,
        fc1_mSFC_mnl: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        # FC2 tensors
        tma_atom_fc2_a: cute.CopyAtom,
        fc2_mA_mkl: cute.Tensor,
        tma_atom_fc2_b: cute.CopyAtom,
        fc2_mB_nkl: cute.Tensor,
        tma_atom_fc2_sfa: cute.CopyAtom,
        fc2_mSFA_mkl: cute.Tensor,
        tma_atom_fc2_sfb: cute.CopyAtom,
        fc2_mSFB_nkl: cute.Tensor,
        fc2_out: cute.Tensor,
        # Shared metadata
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        # Layouts and config
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        fc1_c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        fc1_epi_tile: cute.Tile,
        fc2_epi_tile: cute.Tile,
        fc2_epi_layout: cute.Layout,
        fc1_tile_sched_params: utils.PersistentTileSchedulerParams,
        fc2_tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """Two-phase persistent GEMM kernel: FC1 -> barrier -> FC2."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors for both phases
        if warp_idx == self.tma_b_warp_id:
            # FC1 TMA descriptors
            cpasync.prefetch_descriptor(tma_atom_fc1_b)
            cpasync.prefetch_descriptor(tma_atom_fc1_sfb)
            cpasync.prefetch_descriptor(tma_atom_fc1_c)
            # FC2 TMA descriptors
            cpasync.prefetch_descriptor(tma_atom_fc2_a)
            cpasync.prefetch_descriptor(tma_atom_fc2_b)
            cpasync.prefetch_descriptor(tma_atom_fc2_sfa)
            cpasync.prefetch_descriptor(tma_atom_fc2_sfb)

        # CTA/thread coordinate setup
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # ============================================================
        # SMEM allocation
        # ============================================================
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ============================================================
        # FC1 pipeline creation
        # ============================================================
        # FC1 A pipeline: PipelineCpAsyncUmma (LDGSTS warps 4-7 produce)
        fc1_a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.fc1_a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=CooperativeGroup(Agent.Thread, self.threads_per_warp * 4),
            consumer_group=CooperativeGroup(Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # FC1 B pipeline: PipelineTmaUmma (TMA warp 9 produces)
        fc1_b_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.fc1_b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, self.num_mcast_ctas_b),
            tx_count=self.fc1_num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # FC1 accumulator pipeline
        fc1_acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.fc1_acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, len(self.epilog_warp_id)),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # FC1 tile info pipeline
        fc1_tile_info_pipeline = PipelineAsync.create(
            barrier_storage=storage.fc1_tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=CooperativeGroup(Agent.Thread, self.threads_per_warp),
            consumer_group=CooperativeGroup(Agent.Thread, self.threads_wo_sched),
        )

        # ============================================================
        # FC2 pipeline creation
        # ============================================================
        # FC2 AB pipeline: PipelineTmaUmma (TMA warp loads A+B+SFA+SFB)
        fc2_ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.fc2_ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.fc2_num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # FC2 accumulator pipeline
        fc2_acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.fc2_acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, len(self.epilog_warp_id)),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # FC2 tile info pipeline
        fc2_tile_info_pipeline = PipelineAsync.create(
            barrier_storage=storage.fc2_tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=CooperativeGroup(Agent.Thread, self.threads_per_warp),
            consumer_group=CooperativeGroup(Agent.Thread, self.threads_wo_sched),
        )

        # ============================================================
        # TMEM allocation (shared between phases)
        # ============================================================
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster sync after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ============================================================
        # Setup SMEM tensors (shared between phases)
        # ============================================================
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sC = storage.sC.get_tensor(
            fc1_c_smem_layout_staged.outer,
            swizzle=fc1_c_smem_layout_staged.inner,
        )

        fc1_info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        fc1_sInfo = storage.fc1_sInfo.get_tensor(fc1_info_layout)
        fc2_sInfo = storage.fc2_sInfo.get_tensor(fc1_info_layout)

        # Multicast masks (trivial for cluster (1,1))
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None

        # ============================================================
        # FC1 global tensor partitioning
        # ============================================================
        # FC1 A: gathered input (for LDGSTS)
        fc1_gA_mkl = cute.local_tile(
            fc1_mA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # FC1 B: w3w1 weights (for TMA)
        fc1_gB_nkl = cute.local_tile(
            fc1_mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        fc1_gSFA_mkl = cute.local_tile(
            fc1_mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None)),
            (None, None, None),
        )
        fc1_gSFB_nkl = cute.local_tile(
            fc1_mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        fc1_gToken_ml = cute.local_tile(
            token_id_mapping,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )
        fc1_gC_mnl = cute.local_tile(
            fc1_mC_mnl,
            cute.slice_(self.fc1_mma_tiler_c, (None, None, 0)),
            (None, None, None),
        )
        fc1_k_tile_cnt = cutlass.Int32(cute.size(fc1_gA_mkl, mode=[3]))

        # ============================================================
        # FC2 global tensor partitioning
        # ============================================================
        fc2_gA_mkl = cute.local_tile(
            fc2_mA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        fc2_gB_nkl = cute.local_tile(
            fc2_mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        fc2_gSFA_mkl = cute.local_tile(
            fc2_mSFA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        fc2_gSFB_nkl = cute.local_tile(
            fc2_mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        fc2_k_tile_cnt = cutlass.Int32(cute.size(fc2_gA_mkl, mode=[3]))

        # ============================================================
        # MMA partitions (shared tiled_mma, different global tensors)
        # ============================================================
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)

        # FC1 MMA partitions (A/SFA partitions reserved for future IPC TMA path)
        fc1_tCgA = thr_mma.partition_A(fc1_gA_mkl)  # noqa: F841
        fc1_tCgB = thr_mma.partition_B(fc1_gB_nkl)
        fc1_tCgSFA = thr_mma.partition_A(fc1_gSFA_mkl)  # noqa: F841
        fc1_tCgSFB = thr_mma_sfb.partition_B(fc1_gSFB_nkl)

        # FC2 MMA partitions
        fc2_tCgA = thr_mma.partition_A(fc2_gA_mkl)
        fc2_tCgB = thr_mma.partition_B(fc2_gB_nkl)
        fc2_tCgSFA = thr_mma.partition_A(fc2_gSFA_mkl)
        fc2_tCgSFB = thr_mma_sfb.partition_B(fc2_gSFB_nkl)

        # ============================================================
        # TMA partitions for FC1 B/SFB (same pattern as FC1 reference)
        # ============================================================
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        fc1_tBsB, fc1_tBgB = cpasync.tma_partition(
            tma_atom_fc1_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(fc1_tCgB, 0, 3),
        )

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        fc1_tBsSFB, fc1_tBgSFB = cpasync.tma_partition(
            tma_atom_fc1_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(fc1_tCgSFB, 0, 3),
        )

        # ============================================================
        # TMA partitions for FC2 A/B/SFA/SFB
        # ============================================================
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        fc2_tAsA, fc2_tAgA = cpasync.tma_partition(
            tma_atom_fc2_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(fc2_tCgA, 0, 3),
        )
        fc2_tBsB, fc2_tBgB = cpasync.tma_partition(
            tma_atom_fc2_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(fc2_tCgB, 0, 3),
        )
        fc2_tAsSFA, fc2_tAgSFA = cpasync.tma_partition(
            tma_atom_fc2_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(fc2_tCgSFA, 0, 3),
        )
        fc2_tBsSFB, fc2_tBgSFB = cpasync.tma_partition(
            tma_atom_fc2_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(fc2_tCgSFB, 0, 3),
        )

        # ============================================================
        # PDL: Wait for previous kernel to finish
        # ============================================================
        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_wait()

        # ============================================================
        # Common setup: MMA fragments & TMEM accumulator layout
        # ============================================================
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

        # FC1/FC2 output partitions for epilogue
        fc1_tCgC = thr_mma.partition_C(fc1_gC_mnl)
        fc2_gC_mnl = cute.local_tile(
            fc2_out,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        fc2_tCgC = thr_mma.partition_C(fc2_gC_mnl)

        # CTA sync before warp specialization
        self.cta_sync_barrier.arrive_and_wait()

        # ============================================================
        # ==================== FC1 PHASE =============================
        # ============================================================

        # --- FC1 Scheduler warp (warp 10) ---
        if warp_idx == self.sched_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                fc1_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
            )
            work_tile = tile_sched.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            is_continue = cutlass.Boolean(1)
            while work_tile.is_valid_tile and is_continue:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                if mma_tile_coord_m < num_non_exiting_tiles_value:
                    fc1_tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    cur_tile_coord = work_tile.tile_idx
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                    with cute.arch.elect_one():
                        fc1_sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                        fc1_sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                        fc1_sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        fc1_sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                            work_tile.is_valid_tile
                        )
                        fc1_sInfo[(4, tile_info_producer_state.index)] = mn_limit
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.sched_sync_barrier.arrive_and_wait()
                    fc1_tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                else:
                    is_continue = cutlass.Boolean(0)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Final sentinel tile
            fc1_tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                fc1_sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                fc1_sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                fc1_sInfo[(2, tile_info_producer_state.index)] = -1
                fc1_sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                fc1_sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.sched_sync_barrier.arrive_and_wait()
            fc1_tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            fc1_tile_info_pipeline.producer_tail(tile_info_producer_state)

        # --- FC1 LDGSTS warps (warps 4-7): Gather A+SFA ---
        if warp_idx >= self.ldgsts_a_warp_id[0] and warp_idx <= self.ldgsts_a_warp_id[-1]:
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                fc1_mA_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                fc1_mSFA_mkl.element_type,
                num_bits_per_copy=32,
            )
            tidx_in_warpgroup = tidx % 128
            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)
            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((8,)),
                cutlass.Boolean,
            )
            sfa_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Int32,
            )
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Boolean,
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                fc1_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
            )
            work_tile = tile_sched.initial_work_tile_info()
            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                gToken_ml_tile = fc1_gToken_ml[(None, tile_info[0])]
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
                token_ml_tile_offset = (
                    8 * (tidx_in_warpgroup // 32)
                    + 32 * ((tidx_in_warpgroup % 32) // 8)
                    + (tidx_in_warpgroup % 8)
                )
                sfa_token_offset_tensor[0] = gToken_ml_tile[token_ml_tile_offset] // self.topk
                sfa_predicate_tensor[0] = (
                    cutlass.Boolean(1)
                    if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                    < tile_info[4]
                    else cutlass.Boolean(0)
                )
                relative_sfa_token_offset = sfa_token_offset_tensor[0]

                tAgA = fc1_gA_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 32, divby=32)
                tAgSFA = fc1_gSFA_mkl[(relative_sfa_token_offset, None, 0, None, 0)]
                tAsSFA = sSFA[
                    (
                        (
                            (
                                (
                                    8 * (tidx_in_warpgroup // 32) + (tidx_in_warpgroup % 8),
                                    (tidx_in_warpgroup % 32) // 8,
                                ),
                                None,
                            ),
                            None,
                        ),
                        None,
                        None,
                        None,
                    )
                ]

                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < fc1_k_tile_cnt:
                    peek_a_empty_status = fc1_a_pipeline.producer_try_acquire(a_producer_state)

                for k_tile in cutlass.range(0, fc1_k_tile_cnt, 1, unroll=1):
                    fc1_a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)
                    tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                    tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]
                    tAgSFA_ktile = tAgSFA[(None, a_producer_state.count)]
                    tAsSFA_ktile = tAsSFA[
                        (
                            None,
                            None,
                            None,
                            None,
                            a_producer_state.index,
                        )
                    ]

                    for i in range(8):
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                            divby=32,
                        )
                        A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                        tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                        tAgA_slice = cute.make_tensor(
                            tAgA_slice_ptr,
                            layout=cute.make_layout((32,)),
                        )
                        tAsA_slice = cute.make_tensor(
                            tAsA_ktile[(None, i, None)].iterator,
                            layout=cute.make_layout((32,)),
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]
                        cute.copy_atom_call(
                            a_atom_copy,
                            tAgA_slice,
                            tAsA_slice,
                            pred=a_predicate_slice,
                        )

                    for i in range(4):
                        swizzled_iterator = (tidx_in_warpgroup % 32) // 8 ^ i
                        tAgSFA_slice_ptr = tAgSFA_ktile.iterator + 4 * swizzled_iterator
                        tAgSFA_slice = cute.make_tensor(
                            tAgSFA_slice_ptr,
                            layout=cute.make_layout((4,)),
                        )
                        tAsSFA_slice_ptr = tAsSFA_ktile.iterator + 512 * swizzled_iterator
                        tAsSFA_slice = cute.make_tensor(tAsSFA_slice_ptr, cute.make_layout((4,)))
                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tAgSFA_slice,
                            tAsSFA_slice,
                            pred=sfa_predicate_tensor,
                        )

                    fc1_a_pipeline.producer_commit(a_producer_state)
                    a_producer_state.advance()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < fc1_k_tile_cnt:
                        peek_a_empty_status = fc1_a_pipeline.producer_try_acquire(a_producer_state)

                fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            fc1_a_pipeline.producer_tail(a_producer_state)

        # --- FC1 TMA warp (warp 9): Load B+SFB ---
        if warp_idx == self.tma_b_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                fc1_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
            )
            work_tile = tile_sched.initial_work_tile_info()
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                tBgB_slice = fc1_tBgB[
                    (
                        None,
                        mma_tile_coord_mnl[1],
                        None,
                        mma_tile_coord_mnl[2],
                    )
                ]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB_slice = fc1_tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                b_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < fc1_k_tile_cnt:
                    peek_ab_empty_status = fc1_b_pipeline.producer_try_acquire(b_producer_state)

                for k_tile in cutlass.range(0, fc1_k_tile_cnt, 1, unroll=1):
                    fc1_b_pipeline.producer_acquire(b_producer_state, peek_ab_empty_status)
                    tBgB_k = tBgB_slice[(None, b_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = fc1_tBsB[(None, b_producer_state.index)]
                    tBsSFB_pipe = fc1_tBsSFB[(None, b_producer_state.index)]
                    tma_bar = fc1_b_pipeline.producer_get_barrier(b_producer_state)
                    cute.copy(
                        tma_atom_fc1_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_fc1_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    b_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < fc1_k_tile_cnt:
                        peek_ab_empty_status = fc1_b_pipeline.producer_try_acquire(b_producer_state)

                fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            fc1_b_pipeline.producer_tail(b_producer_state)

        # --- FC1 MMA warp (warp 8) ---
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

            tile_sched = utils.StaticPersistentTileScheduler.create(
                fc1_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
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

            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                a_consumer_state.reset_count()
                peek_a_full_status = cutlass.Boolean(1)
                if a_consumer_state.count < fc1_k_tile_cnt:
                    peek_a_full_status = fc1_a_pipeline.consumer_try_wait(a_consumer_state)
                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < fc1_k_tile_cnt and is_leader_cta:
                    peek_b_full_status = fc1_b_pipeline.consumer_try_wait(b_consumer_state)

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
                    fc1_acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(fc1_k_tile_cnt):
                    if is_leader_cta:
                        fc1_a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                        fc1_b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            b_consumer_state.index,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_consumer_state.index,
                            )
                            sf_kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        fc1_a_pipeline.consumer_release(a_consumer_state)
                        fc1_b_pipeline.consumer_release(b_consumer_state)

                    a_consumer_state.advance()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < fc1_k_tile_cnt:
                        peek_a_full_status = fc1_a_pipeline.consumer_try_wait(a_consumer_state)
                    b_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < fc1_k_tile_cnt:
                        if is_leader_cta:
                            peek_b_full_status = fc1_b_pipeline.consumer_try_wait(b_consumer_state)

                if is_leader_cta:
                    fc1_acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            fc1_acc_pipeline.producer_tail(acc_producer_state)

        # --- FC1 Epilogue warps (warps 0-3): SwiGLU + NVFP4 quant + TMA store ---
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
            ) = self.fc1_epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                fc1_tCgC,
                fc1_epi_tile,
                self.use_2cta_instrs,
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.fc1_c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.fc1_epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c_epi,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.fc1_epilog_gmem_copy_and_partition(
                epi_tidx,
                tma_atom_fc1_c,
                fc1_tCgC,
                fc1_epi_tile,
                sC,
            )

            if cutlass.const_expr(self.generate_sfc):
                norm_const = fc1_norm_const[0]
                gSFC_mnl = cute.local_tile(
                    fc1_mSFC_mnl,
                    fc1_epi_tile,
                    (None, None, None),
                )
                thr_copy_t2r_epi = tiled_copy_t2r.get_slice(tidx)
                tCgSFC_mnl = thr_copy_t2r_epi.partition_D(gSFC_mnl)
                tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                tCrSFC = cute.make_rmem_tensor(
                    tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout,
                    self.sf_dtype,
                )
                tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                fc1_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.fc1_num_c_stage,
                producer_group=c_producer_group,
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = fc1_alpha[expert_idx]

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

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                tTR_tAcc = tTR_tAcc_base[
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        acc_stage_index,
                    )
                ]

                if cutlass.const_expr(self.generate_sfc):
                    tCgSFC_mn = tCgSFC_mnl[(None, None, None, None, None, 0)]

                fc1_acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                    real_subtile_idx = subtile_idx // 2
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n_required
                                - 1
                                - subtile_idx // 2
                            )

                    tTR_tAcc_mn_up = tTR_tAcc[
                        (
                            None,
                            None,
                            None,
                            real_subtile_idx * 2,
                        )
                    ]
                    tTR_tAcc_mn_gate = tTR_tAcc[
                        (
                            None,
                            None,
                            None,
                            real_subtile_idx * 2 + 1,
                        )
                    ]
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_mn_up,
                        tTR_rAcc_up,
                    )
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_mn_gate,
                        tTR_rAcc_gate,
                    )

                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx // 2 == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                fc1_acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    acc_vec_up = tTR_rAcc_up.load()
                    acc_vec_gate = tTR_rAcc_gate.load()

                    # SwiGLU: output = (alpha * up) * silu(alpha * gate)
                    tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                    if cutlass.const_expr(self.vectorized_f32):
                        LOG2_E = cutlass.Float32(1.4426950408889634)
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                            acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                (
                                    acc_vec_up[i],
                                    acc_vec_up[i + 1],
                                ),
                                (
                                    cutlass.Float32(alpha_val),
                                    cutlass.Float32(alpha_val),
                                ),
                            )
                            acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                (
                                    acc_vec_gate[i],
                                    acc_vec_gate[i + 1],
                                ),
                                (
                                    cutlass.Float32(alpha_val),
                                    cutlass.Float32(alpha_val),
                                ),
                            )
                            tCompute_log2e = cute.arch.mul_packed_f32x2(
                                (
                                    acc_vec_gate_alpha[0],
                                    acc_vec_gate_alpha[1],
                                ),
                                (-LOG2_E, -LOG2_E),
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.add_packed_f32x2(
                                (
                                    cute.math.exp2(
                                        tCompute_log2e[0],
                                        fastmath=True,
                                    ),
                                    cute.math.exp2(
                                        tCompute_log2e[1],
                                        fastmath=True,
                                    ),
                                ),
                                (1.0, 1.0),
                            )
                            tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                            tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (
                                    acc_vec_gate_alpha[0],
                                    acc_vec_gate_alpha[1],
                                ),
                            )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(
                                (tCompute[i], tCompute[i + 1]),
                                (
                                    acc_vec_up_alpha[0],
                                    acc_vec_up_alpha[1],
                                ),
                            )
                    else:
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                            acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(alpha_val)
                            acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(alpha_val)
                            tCompute[i] = acc_vec_up_alpha * silu_f32(
                                acc_vec_gate_alpha,
                                fastmath=True,
                            )

                    if cutlass.const_expr(self.generate_sfc):
                        # NVFP4 quantization: compute SFC + quantize
                        sfc_subtile_idx_mn = (
                            tile_info[0] * self.fc1_epi_tile_cnt[0],
                            tile_info[1] * self.fc1_epi_tile_cnt[1] + real_subtile_idx,
                        )
                        tCgSFC = tCgSFC_mn[
                            (
                                None,
                                None,
                                None,
                                *sfc_subtile_idx_mn,
                            )
                        ]
                        tTR_rAcc_frg = cute.logical_divide(
                            tCompute,
                            cute.make_layout(self.sf_vec_size),
                        )
                        acc_frg = tTR_rAcc_frg.load()
                        abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                        abs_acc_frg = type(acc_frg)(
                            abs_acc_frg_ir,
                            acc_frg.shape,
                            acc_frg.dtype,
                        )
                        if cutlass.const_expr(self.vectorized_f32):
                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,
                                )
                            for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                                (
                                    tCrSFC_pvscale[vi],
                                    tCrSFC_pvscale[vi + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (
                                        tCrSFC_pvscale[vi],
                                        tCrSFC_pvscale[vi + 1],
                                    ),
                                    (
                                        self.get_dtype_rcp_limits(self.fc1_c_dtype),
                                        self.get_dtype_rcp_limits(self.fc1_c_dtype),
                                    ),
                                )
                                (
                                    tCrSFC_pvscale[vi],
                                    tCrSFC_pvscale[vi + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (
                                        tCrSFC_pvscale[vi],
                                        tCrSFC_pvscale[vi + 1],
                                    ),
                                    (norm_const, norm_const),
                                )
                        else:
                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrSFC_pvscale[vi] = (
                                    abs_acc_frg[None, vi].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,
                                    )
                                    * self.get_dtype_rcp_limits(self.fc1_c_dtype)
                                    * norm_const
                                )
                        tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))
                        cute.autovec_copy(tCrSFC, tCgSFC)

                        # Quantize output
                        tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                        fp32_max = cutlass.Float32(3.40282346638528859812e38)
                        if cutlass.const_expr(self.vectorized_f32):
                            for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                                acc_scale = cute.arch.mul_packed_f32x2(
                                    (
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi]),
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi + 1]),
                                    ),
                                    (
                                        norm_const,
                                        norm_const,
                                    ),
                                )
                                acc_scale_min0 = fmin(
                                    acc_scale[0],
                                    fp32_max,
                                    nan=True,
                                )
                                acc_scale_min1 = fmin(
                                    acc_scale[1],
                                    fp32_max,
                                    nan=True,
                                )
                                vec0 = tTR_rAcc_frg[None, vi]
                                vec1 = tTR_rAcc_frg[None, vi + 1]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    (
                                        vec0[ei],
                                        vec1[ei],
                                    ) = cute.arch.mul_packed_f32x2(
                                        (vec0[ei], vec1[ei]),
                                        (
                                            acc_scale_min0,
                                            acc_scale_min1,
                                        ),
                                    )
                        else:
                            for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                                acc_scale = norm_const * cute.arch.rcp_approx(
                                    tCrSFC_qpvscale_up[vi]
                                )
                                acc_scale = fmin(
                                    acc_scale,
                                    fp32_max,
                                    nan=True,
                                )
                                vec = tTR_rAcc_frg[None, vi]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    vec[ei] = vec[ei] * acc_scale

                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.fc1_c_dtype))
                    else:
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.fc1_c_dtype))

                    # Store C via TMA S2G
                    num_prev_subtiles = num_prev_subtiles + 1
                    c_buffer = num_prev_subtiles % self.fc1_num_c_stage
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
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c_epi,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        fc1_acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                fc1_tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = fc1_sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc1_tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()

        # ============================================================
        # ==================== PHASE BARRIER =========================
        # ============================================================
        # All warps synchronize before transitioning to FC2 phase.
        # FC1 pipelines are fully drained at this point.
        self.phase_barrier.arrive_and_wait()

        # ============================================================
        # ==================== FC2 PHASE =============================
        # ============================================================
        # FC2 warp-specialized execution follows the same pattern as
        # blockscaled_contiguous_grouped_gemm_finalize_fusion.py.
        # TMA warp loads A (FC1 output) + B (w2) + SFA + SFB,
        # MMA warp computes GEMM, epilogue warps apply scatter-add.
        # Warps 4-7 (LDGSTS) are idle during FC2.
        #
        # TODO: FC2 warp bodies will be added in subsequent edits.
        # ============================================================

        # --- FC2 Scheduler warp (warp 10) ---
        if warp_idx == self.sched_warp_id:
            fc2_tile_sched = utils.StaticPersistentTileScheduler.create(
                fc2_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
            )
            fc2_work_tile = fc2_tile_sched.initial_work_tile_info()
            fc2_tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_tile_stage,
            )
            fc2_num_valid_tiles = num_non_exiting_tiles[0]

            fc2_is_continue = cutlass.Boolean(1)
            while fc2_work_tile.is_valid_tile and fc2_is_continue:
                fc2_cur_tile_coord = fc2_work_tile.tile_idx
                fc2_mma_tile_coord_m = fc2_cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                if fc2_mma_tile_coord_m < fc2_num_valid_tiles:
                    fc2_tile_info_pipeline.producer_acquire(fc2_tile_info_producer_state)
                    fc2_cur_tile_coord = fc2_work_tile.tile_idx
                    fc2_expert_idx = tile_idx_to_expert_idx[fc2_mma_tile_coord_m]
                    fc2_mn_limit = tile_idx_to_mn_limit[fc2_mma_tile_coord_m]
                    with cute.arch.elect_one():
                        fc2_sInfo[
                            (
                                0,
                                fc2_tile_info_producer_state.index,
                            )
                        ] = fc2_cur_tile_coord[0]
                        fc2_sInfo[
                            (
                                1,
                                fc2_tile_info_producer_state.index,
                            )
                        ] = fc2_cur_tile_coord[1]
                        fc2_sInfo[
                            (
                                2,
                                fc2_tile_info_producer_state.index,
                            )
                        ] = fc2_expert_idx
                        fc2_sInfo[
                            (
                                3,
                                fc2_tile_info_producer_state.index,
                            )
                        ] = cutlass.Int32(fc2_work_tile.is_valid_tile)
                        fc2_sInfo[
                            (
                                4,
                                fc2_tile_info_producer_state.index,
                            )
                        ] = fc2_mn_limit
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.sched_sync_barrier.arrive_and_wait()
                    fc2_tile_info_pipeline.producer_commit(fc2_tile_info_producer_state)
                    fc2_tile_info_producer_state.advance()
                else:
                    fc2_is_continue = cutlass.Boolean(0)
                fc2_tile_sched.advance_to_next_work()
                fc2_work_tile = fc2_tile_sched.get_current_work()

            # Sentinel tile to signal end of FC2 tiles
            fc2_tile_info_pipeline.producer_acquire(fc2_tile_info_producer_state)
            with cute.arch.elect_one():
                fc2_sInfo[(0, fc2_tile_info_producer_state.index)] = fc2_work_tile.tile_idx[0]
                fc2_sInfo[(1, fc2_tile_info_producer_state.index)] = fc2_work_tile.tile_idx[1]
                fc2_sInfo[(2, fc2_tile_info_producer_state.index)] = -1
                fc2_sInfo[(3, fc2_tile_info_producer_state.index)] = cutlass.Int32(0)
                fc2_sInfo[(4, fc2_tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.sched_sync_barrier.arrive_and_wait()
            fc2_tile_info_pipeline.producer_commit(fc2_tile_info_producer_state)
            fc2_tile_info_producer_state.advance()
            fc2_tile_info_pipeline.producer_tail(fc2_tile_info_producer_state)

        # --- FC2 LDGSTS warps (warps 4-7): Idle during FC2 ---
        # Warps 4-7 have no work in FC2 phase. They will wait at
        # the final CTA sync barrier after FC2 completes.

        # --- FC2 TMA warp (warp 9): Load A+B+SFA+SFB ---
        if warp_idx == self.tma_warp_id:
            fc2_ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_ab_stage,
            )
            fc2_tma_tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_tile_stage,
            )

            # Get first FC2 tile info
            fc2_tma_tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            fc2_tile_info_pipeline.consumer_wait(fc2_tma_tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                fc2_tma_tile_info[idx] = fc2_sInfo[(idx, fc2_tma_tile_info_consumer_state.index)]
            fc2_tma_is_valid = fc2_tma_tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc2_tile_info_pipeline.consumer_release(fc2_tma_tile_info_consumer_state)
            fc2_tma_tile_info_consumer_state.advance()

            while fc2_tma_is_valid:
                fc2_tma_mnl = (
                    fc2_tma_tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    fc2_tma_tile_info[1],
                    fc2_tma_tile_info[2],
                )
                # Slice to per-MMA tile index
                fc2_tAgA_slice = fc2_tAgA[(None, fc2_tma_mnl[0], None, 0)]
                fc2_tBgB_slice = fc2_tBgB[(None, fc2_tma_mnl[1], None, fc2_tma_mnl[2])]
                fc2_tAgSFA_slice = fc2_tAgSFA[(None, fc2_tma_mnl[0], None, 0)]
                fc2_tBgSFB_slice = fc2_tBgSFB[(None, fc2_tma_mnl[1], None, fc2_tma_mnl[2])]

                # Peek AB buffer empty
                fc2_ab_producer_state.reset_count()
                fc2_peek_empty = cutlass.Boolean(1)
                if fc2_ab_producer_state.count < fc2_k_tile_cnt:
                    fc2_peek_empty = fc2_ab_pipeline.producer_try_acquire(fc2_ab_producer_state)

                # TMA load loop over K tiles
                for fc2_k_tile in cutlass.range(0, fc2_k_tile_cnt, 1, unroll=1):
                    fc2_tAgA_k = fc2_tAgA_slice[(None, fc2_ab_producer_state.count)]
                    fc2_tBgB_k = fc2_tBgB_slice[(None, fc2_ab_producer_state.count)]
                    fc2_tAgSFA_k = fc2_tAgSFA_slice[(None, fc2_ab_producer_state.count)]
                    fc2_tBgSFB_k = fc2_tBgSFB_slice[(None, fc2_ab_producer_state.count)]
                    fc2_tAsA_pipe = fc2_tAsA[(None, fc2_ab_producer_state.index)]
                    fc2_tBsB_pipe = fc2_tBsB[(None, fc2_ab_producer_state.index)]
                    fc2_tAsSFA_pipe = fc2_tAsSFA[(None, fc2_ab_producer_state.index)]
                    fc2_tBsSFB_pipe = fc2_tBsSFB[(None, fc2_ab_producer_state.index)]

                    fc2_tma_bar = fc2_ab_pipeline.producer_get_barrier(fc2_ab_producer_state)

                    # Wait for AB buffer empty
                    fc2_ab_pipeline.producer_acquire(fc2_ab_producer_state, fc2_peek_empty)

                    # TMA copy A, B, SFA, SFB
                    cute.copy(
                        tma_atom_fc2_a,
                        fc2_tAgA_k,
                        fc2_tAsA_pipe,
                        tma_bar_ptr=fc2_tma_bar,
                        mcast_mask=None,
                    )
                    cute.copy(
                        tma_atom_fc2_b,
                        fc2_tBgB_k,
                        fc2_tBsB_pipe,
                        tma_bar_ptr=fc2_tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_fc2_sfa,
                        fc2_tAgSFA_k,
                        fc2_tAsSFA_pipe,
                        tma_bar_ptr=fc2_tma_bar,
                        mcast_mask=None,
                    )
                    cute.copy(
                        tma_atom_fc2_sfb,
                        fc2_tBgSFB_k,
                        fc2_tBsSFB_pipe,
                        tma_bar_ptr=fc2_tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Advance and peek next
                    fc2_ab_producer_state.advance()
                    fc2_peek_empty = cutlass.Boolean(1)
                    if fc2_ab_producer_state.count < fc2_k_tile_cnt:
                        fc2_peek_empty = fc2_ab_pipeline.producer_try_acquire(fc2_ab_producer_state)

                # Advance to next FC2 tile
                fc2_tile_info_pipeline.consumer_wait(fc2_tma_tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    fc2_tma_tile_info[idx] = fc2_sInfo[
                        (
                            idx,
                            fc2_tma_tile_info_consumer_state.index,
                        )
                    ]
                fc2_tma_is_valid = fc2_tma_tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc2_tile_info_pipeline.consumer_release(fc2_tma_tile_info_consumer_state)
                fc2_tma_tile_info_consumer_state.advance()

            # Wait AB buffer empty
            fc2_ab_pipeline.producer_tail(fc2_ab_producer_state)

        # --- FC2 MMA warp (warp 8): GEMM mainloop ---
        if warp_idx == self.mma_warp_id:
            # Retrieve TMEM pointer (already allocated in FC1)
            tmem.wait_for_alloc()
            fc2_acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            fc2_tCtAcc_base = cute.make_tensor(fc2_acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA tmem tensor
            fc2_sfa_tmem_ptr = cute.recast_ptr(
                fc2_acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            fc2_tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            fc2_tCtSFA = cute.make_tensor(fc2_sfa_tmem_ptr, fc2_tCtSFA_layout)

            # SFB tmem tensor
            fc2_sfb_tmem_ptr = cute.recast_ptr(
                fc2_acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            fc2_tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(
                    sfb_smem_layout_staged,
                    (None, None, None, 0),
                ),
            )
            fc2_tCtSFB = cute.make_tensor(fc2_sfb_tmem_ptr, fc2_tCtSFB_layout)

            # S2T copy partitions for SFA/SFB
            (
                fc2_tiled_copy_s2t_sfa,
                fc2_tCsSFA_compact_s2t,
                fc2_tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, fc2_tCtSFA)
            (
                fc2_tiled_copy_s2t_sfb,
                fc2_tCsSFB_compact_s2t,
                fc2_tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, fc2_tCtSFB)

            fc2_ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_ab_stage,
            )
            fc2_acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_acc_stage,
            )
            fc2_mma_tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_tile_stage,
            )

            # Get first FC2 tile info
            fc2_mma_tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            fc2_tile_info_pipeline.consumer_wait(fc2_mma_tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                fc2_mma_tile_info[idx] = fc2_sInfo[
                    (
                        idx,
                        fc2_mma_tile_info_consumer_state.index,
                    )
                ]
            fc2_mma_is_valid = fc2_mma_tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc2_tile_info_pipeline.consumer_release(fc2_mma_tile_info_consumer_state)
            fc2_mma_tile_info_consumer_state.advance()

            while fc2_mma_is_valid:
                # Peek AB buffer full
                fc2_ab_consumer_state.reset_count()
                fc2_peek_full = cutlass.Boolean(1)
                if fc2_ab_consumer_state.count < fc2_k_tile_cnt and is_leader_cta:
                    fc2_peek_full = fc2_ab_pipeline.consumer_try_wait(fc2_ab_consumer_state)

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    fc2_acc_stage = fc2_acc_producer_state.phase ^ 1
                else:
                    fc2_acc_stage = fc2_acc_producer_state.index

                fc2_tCtAcc = fc2_tCtAcc_base[(None, None, None, fc2_acc_stage)]

                fc2_tCtSFB_mma = fc2_tCtSFB

                # Wait for accumulator buffer empty
                if is_leader_cta:
                    fc2_acc_pipeline.producer_acquire(fc2_acc_producer_state)

                # Reset ACCUMULATE for each new tile
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # MMA mainloop over K tiles
                for fc2_k_tile in cutlass.range(fc2_k_tile_cnt):
                    if is_leader_cta:
                        # Wait for AB buffer full
                        fc2_ab_pipeline.consumer_wait(
                            fc2_ab_consumer_state,
                            fc2_peek_full,
                        )

                        # S2T copy SFA/SFB
                        fc2_s2t_coord = (
                            None,
                            None,
                            None,
                            None,
                            fc2_ab_consumer_state.index,
                        )
                        fc2_tCsSFA_staged = fc2_tCsSFA_compact_s2t[fc2_s2t_coord]
                        fc2_tCsSFB_staged = fc2_tCsSFB_compact_s2t[fc2_s2t_coord]
                        cute.copy(
                            fc2_tiled_copy_s2t_sfa,
                            fc2_tCsSFA_staged,
                            fc2_tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            fc2_tiled_copy_s2t_sfb,
                            fc2_tCsSFB_staged,
                            fc2_tCtSFB_compact_s2t,
                        )

                        # GEMM over kblocks
                        fc2_num_kblocks = cute.size(tCrA, mode=[2])
                        for fc2_kblock_idx in cutlass.range(fc2_num_kblocks, unroll_full=True):
                            fc2_kblock_coord = (
                                None,
                                None,
                                fc2_kblock_idx,
                                fc2_ab_consumer_state.index,
                            )
                            fc2_sf_kblock_coord = (
                                None,
                                None,
                                fc2_kblock_idx,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                fc2_tCtSFA[fc2_sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                fc2_tCtSFB_mma[fc2_sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                fc2_tCtAcc,
                                tCrA[fc2_kblock_coord],
                                tCrB[fc2_kblock_coord],
                                fc2_tCtAcc,
                            )

                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Release AB buffer
                        fc2_ab_pipeline.consumer_release(fc2_ab_consumer_state)

                    # Peek next AB full
                    fc2_ab_consumer_state.advance()
                    fc2_peek_full = cutlass.Boolean(1)
                    if fc2_ab_consumer_state.count < fc2_k_tile_cnt:
                        if is_leader_cta:
                            fc2_peek_full = fc2_ab_pipeline.consumer_try_wait(fc2_ab_consumer_state)

                # Commit accumulator
                if is_leader_cta:
                    fc2_acc_pipeline.producer_commit(fc2_acc_producer_state)

                fc2_acc_producer_state.advance()

                # Advance to next FC2 tile
                fc2_tile_info_pipeline.consumer_wait(fc2_mma_tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    fc2_mma_tile_info[idx] = fc2_sInfo[
                        (
                            idx,
                            fc2_mma_tile_info_consumer_state.index,
                        )
                    ]
                fc2_mma_is_valid = fc2_mma_tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc2_tile_info_pipeline.consumer_release(fc2_mma_tile_info_consumer_state)
                fc2_mma_tile_info_consumer_state.advance()

            # Wait accumulator buffer empty
            fc2_acc_pipeline.producer_tail(fc2_acc_producer_state)

        # --- FC2 Epilogue warps (warps 0-3): scatter-add ---
        if warp_idx <= self.epilog_warp_id[-1]:
            # Re-allocate TMEM (after FC1 freed it)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            fc2_epi_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            fc2_epi_tCtAcc_base = cute.make_tensor(fc2_epi_tmem_ptr, tCtAcc_fake.layout)

            fc2_epi_tidx = tidx % 128
            (
                fc2_tiled_copy_t2r,
                fc2_tTR_tAcc_base,
                fc2_tTR_rAcc,
            ) = self.fc2_epilog_tmem_copy_and_partition(
                fc2_epi_tidx,
                fc2_epi_tCtAcc_base,
                fc2_tCgC,
                fc2_epi_tile,
                self.use_2cta_instrs,
            )

            fc2_tTR_rC = cute.make_rmem_tensor(fc2_tTR_rAcc.shape, self.fc2_out_dtype)

            fc2_epi_acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_acc_stage,
            )
            fc2_epi_tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_tile_stage,
            )

            fc2_token_idx = cutlass.Int32(0)
            fc2_token_scale = self.fc2_final_scale_dtype(0.0)

            # Get first FC2 tile info
            fc2_epi_tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            fc2_tile_info_pipeline.consumer_wait(fc2_epi_tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                fc2_epi_tile_info[idx] = fc2_sInfo[
                    (
                        idx,
                        fc2_epi_tile_info_consumer_state.index,
                    )
                ]
            fc2_epi_is_valid = fc2_epi_tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            fc2_tile_info_pipeline.consumer_release(fc2_epi_tile_info_consumer_state)
            fc2_epi_tile_info_consumer_state.advance()

            while fc2_epi_is_valid:
                fc2_epi_mnl = (
                    fc2_epi_tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    fc2_epi_tile_info[1],
                    fc2_epi_tile_info[2],
                )

                # Get alpha for current expert group
                fc2_expert_idx_epi = fc2_epi_mnl[2]
                fc2_alpha_val = fc2_alpha[fc2_expert_idx_epi]

                fc2_tile_m_start = fc2_epi_tile_info[0] * self.cta_tile_shape_mnk[0]
                fc2_permuted_row = fc2_tile_m_start + fc2_epi_tidx
                fc2_expanded_idx = permuted_idx_to_expanded_idx[fc2_permuted_row]
                fc2_is_valid_row = fc2_permuted_row < fc2_epi_tile_info[4]

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    fc2_epi_acc_stage = fc2_epi_acc_consumer_state.phase
                    fc2_reverse_subtile = (
                        cutlass.Boolean(True) if fc2_epi_acc_stage == 0 else cutlass.Boolean(False)
                    )
                else:
                    fc2_epi_acc_stage = fc2_epi_acc_consumer_state.index

                fc2_tTR_tAcc = fc2_tTR_tAcc_base[
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        fc2_epi_acc_stage,
                    )
                ]

                # Wait for accumulator buffer full
                fc2_acc_pipeline.consumer_wait(fc2_epi_acc_consumer_state)

                fc2_tTR_tAcc = cute.group_modes(fc2_tTR_tAcc, 3, cute.rank(fc2_tTR_tAcc))
                fc2_subtile_cnt = cute.size(fc2_tTR_tAcc.shape, mode=[3])

                if fc2_is_valid_row:
                    fc2_token_idx = fc2_expanded_idx // self.topk
                    fc2_topk_idx = fc2_expanded_idx % self.topk
                    fc2_token_scale = token_final_scales[(fc2_token_idx, fc2_topk_idx)]
                    fc2_alpha_val = fc2_alpha_val * fc2_token_scale

                for fc2_subtile_idx in cutlass.range(fc2_subtile_cnt):
                    fc2_real_subtile_idx = fc2_subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if fc2_reverse_subtile:
                            fc2_real_subtile_idx = fc2_subtile_cnt - 1 - fc2_subtile_idx

                    # Load accumulator from TMEM to register
                    fc2_tTR_tAcc_mn = fc2_tTR_tAcc[(None, None, None, fc2_real_subtile_idx)]
                    cute.copy(
                        fc2_tiled_copy_t2r,
                        fc2_tTR_tAcc_mn,
                        fc2_tTR_rAcc,
                    )

                    # Early release accumulator if overlapping
                    if cutlass.const_expr(self.overlapping_accum):
                        if fc2_subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                fc2_acc_pipeline.consumer_release(fc2_epi_acc_consumer_state)
                            fc2_epi_acc_consumer_state.advance()

                    # Scale and scatter-add to output
                    fc2_acc_vec = fc2_tTR_rAcc.load()
                    fc2_acc_vec_final = fc2_alpha_val * fc2_acc_vec
                    fc2_tTR_rC.store(fc2_acc_vec_final.to(self.fc2_out_dtype))

                    if fc2_is_valid_row:
                        fc2_rOut_epi = cute.make_tensor(
                            fc2_tTR_rC.iterator,
                            fc2_epi_layout,
                        )
                        fc2_base_coord_n = fc2_epi_mnl[1] * self.cta_tile_shape_mnk[
                            1
                        ] + fc2_real_subtile_idx * cute.size(fc2_tTR_rC)
                        fc2_scatter_out = cute.domain_offset(
                            (fc2_token_idx, 0, 0),
                            fc2_out,
                        )

                        for fc2_epi_idx in cutlass.range(
                            self.fc2_epi_loop_size,
                            unroll_full=True,
                        ):
                            fc2_coord_n = fc2_base_coord_n + fc2_epi_idx * self.fc2_element_offset
                            fc2_scatter_offset = cute.domain_offset(
                                (0, fc2_coord_n, 0),
                                fc2_scatter_out,
                            )
                            if cutlass.const_expr(self.fc2_out_dtype == cutlass.BFloat16):
                                fc2_rOut_packed = fc2_rOut_epi[fc2_epi_idx, None, None]
                                vectorized_atomic_add_bf16x8(
                                    fc2_rOut_packed,
                                    fc2_scatter_offset,
                                )
                            elif cutlass.const_expr(self.fc2_out_dtype == cutlass.Float32):
                                fc2_rOut_packed = fc2_rOut_epi[fc2_epi_idx, None]
                                vectorized_atomic_add_fp32x2(
                                    fc2_rOut_packed,
                                    fc2_scatter_offset,
                                )
                            else:
                                fc2_rOut_packed = fc2_rOut_epi[fc2_epi_idx]
                                atomic_add_func(
                                    fc2_rOut_packed,
                                    fc2_scatter_offset,
                                )

                # Release accumulator (non-overlapping path)
                if cutlass.const_expr(not self.overlapping_accum):
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        fc2_acc_pipeline.consumer_release(fc2_epi_acc_consumer_state)
                    fc2_epi_acc_consumer_state.advance()

                # Advance to next FC2 tile
                fc2_tile_info_pipeline.consumer_wait(fc2_epi_tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    fc2_epi_tile_info[idx] = fc2_sInfo[
                        (
                            idx,
                            fc2_epi_tile_info_consumer_state.index,
                        )
                    ]
                fc2_epi_is_valid = fc2_epi_tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                fc2_tile_info_pipeline.consumer_release(fc2_epi_tile_info_consumer_state)
                fc2_epi_tile_info_consumer_state.advance()

            # Dealloc TMEM
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(fc2_epi_tmem_ptr)

        # ============================================================
        # PDL: Signal dependent kernels
        # ============================================================
        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_launch_dependents()

        # TMEM deallocation
        tmem.free(self.num_tmem_alloc_cols)

    # =================================================================
    # wrapper: Convert raw pointers to CuTe tensors and invoke kernel
    # =================================================================

    def wrapper(
        self,
        # FC1 pointers
        fc1_a_ptr: cute.Pointer,
        fc1_b_ptr: cute.Pointer,
        fc1_c_ptr: cute.Pointer,
        fc1_sfa_ptr: cute.Pointer,
        fc1_sfb_ptr: cute.Pointer,
        fc1_sfc_ptr: Optional[cute.Pointer],
        fc1_norm_const_ptr: Optional[cute.Pointer],
        fc1_alpha_ptr: cute.Pointer,
        # FC2 pointers
        fc2_a_ptr: cute.Pointer,
        fc2_b_ptr: cute.Pointer,
        fc2_out_ptr: cute.Pointer,
        fc2_sfa_ptr: cute.Pointer,
        fc2_sfb_ptr: cute.Pointer,
        fc2_alpha_ptr: cute.Pointer,
        # Shared metadata pointers
        tile_idx_to_expert_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        # Dimensions
        orig_m: cutlass.Int64,  # original input tokens (before gather)
        m: cutlass.Int64,  # permuted/padded M dimension
        fc1_n: cutlass.Int64,  # 2 * intermediate_size (w3w1 N dim)
        fc2_n: cutlass.Int64,  # hidden_size (w2 N dim)
        k1: cutlass.Int64,  # hidden_size (FC1 K dim)
        k2: cutlass.Int64,  # intermediate_size (FC2 K dim)
        l: cutlass.Int64,  # number of local experts  # noqa: E741
        num_tokens: cutlass.Int64,  # total global tokens
        top_k: cutlass.Int64,  # number of experts per token
        # Constexpr config
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Convert raw pointers to CuTe tensors and invoke the fused kernel."""
        num_tiles = m // self.tile_size
        scale_k1 = k1 // scaling_vector_size
        scale_k2 = k2 // scaling_vector_size
        fc1_intermediate_sz = fc1_n // 2  # intermediate_size

        # FC1 A: [orig_m, K1, 1] row-major
        fc1_a = cute.make_tensor(
            fc1_a_ptr,
            layout=cute.make_ordered_layout((orig_m, k1, 1), order=(1, 0, 2)),
        )
        # FC1 B: [2*N1, K1, L] row-major
        fc1_b = cute.make_tensor(
            fc1_b_ptr,
            layout=cute.make_ordered_layout((fc1_n, k1, l), order=(1, 0, 2)),
        )
        # FC1 C: [M, N1, 1] row-major (intermediate_size, not 2x)
        fc1_c = cute.make_tensor(
            fc1_c_ptr,
            layout=cute.make_ordered_layout((m, fc1_intermediate_sz, 1), order=(1, 0, 2)),
        )
        # FC1 SFA: [orig_m, scale_k1, 1] row-major
        fc1_sfa = cute.make_tensor(
            fc1_sfa_ptr,
            layout=cute.make_ordered_layout((orig_m, scale_k1, 1), order=(1, 0, 2)),
        )
        # FC1 SFB: MMA-friendly layout [32, 4, n//128, 4, scale_k1//4, L]
        fc1_sfb = cute.make_tensor(
            fc1_sfb_ptr,
            layout=cute.make_ordered_layout(
                (
                    32,
                    4,
                    fc1_n // 128,
                    4,
                    scale_k1 // 4,
                    l,
                ),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        # FC1 SFC: MMA-friendly layout (output scale factors)
        fc1_sfc = None
        if fc1_sfc_ptr is not None:
            fc1_sfc = cute.make_tensor(
                fc1_sfc_ptr,
                layout=cute.make_ordered_layout(
                    (
                        32,
                        4,
                        m // 128,
                        4,
                        fc1_intermediate_sz // (scaling_vector_size * 4),
                        l,
                    ),
                    order=(2, 1, 4, 0, 3, 5),
                ),
            )
        # FC1 norm const
        fc1_norm_const = None
        if fc1_norm_const_ptr is not None:
            fc1_norm_const = cute.make_tensor(
                fc1_norm_const_ptr,
                layout=cute.make_layout((1,)),
            )
        # FC1 alpha: [L]
        fc1_alpha = cute.make_tensor(fc1_alpha_ptr, layout=cute.make_layout((l,)))

        # FC2 A: [M, K2, L] row-major
        fc2_a = cute.make_tensor(
            fc2_a_ptr,
            layout=cute.make_ordered_layout((m, k2, l), order=(1, 0, 2)),
        )
        # FC2 B: [N2, K2, L] row-major
        fc2_b = cute.make_tensor(
            fc2_b_ptr,
            layout=cute.make_ordered_layout((fc2_n, k2, l), order=(1, 0, 2)),
        )
        # FC2 output: [num_tokens, N2, 1] row-major
        fc2_out = cute.make_tensor(
            fc2_out_ptr,
            layout=cute.make_ordered_layout((num_tokens, fc2_n, 1), order=(1, 0, 2)),
        )
        # FC2 SFA: MMA-friendly layout
        fc2_sfa = cute.make_tensor(
            fc2_sfa_ptr,
            layout=cute.make_ordered_layout(
                (
                    32,
                    4,
                    m // 128,
                    4,
                    scale_k2 // 4,
                    l,
                ),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        # FC2 SFB: MMA-friendly layout
        fc2_sfb = cute.make_tensor(
            fc2_sfb_ptr,
            layout=cute.make_ordered_layout(
                (
                    32,
                    4,
                    fc2_n // 128,
                    4,
                    scale_k2 // 4,
                    l,
                ),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        # FC2 alpha: [L]
        fc2_alpha = cute.make_tensor(fc2_alpha_ptr, layout=cute.make_layout((l,)))

        # Shared metadata tensors
        tile_idx_to_expert_idx = cute.make_tensor(
            tile_idx_to_expert_idx_ptr,
            layout=cute.make_layout((num_tiles,)),
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr,
            layout=cute.make_layout((num_tiles,)),
        )
        token_id_mapping = cute.make_tensor(
            token_id_mapping_ptr,
            layout=cute.make_layout((m,)),
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr,
            layout=cute.make_layout((1,)),
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr,
            layout=cute.make_layout((m,)),
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
        )

        return self(
            fc1_a,
            fc1_b,
            fc1_c,
            fc1_sfa,
            fc1_sfb,
            fc1_sfc,
            fc1_norm_const,
            fc2_a,
            fc2_b,
            fc2_out,
            fc2_sfa,
            fc2_sfb,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            fc1_alpha,
            fc2_alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
