# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    TRTLLM_ENABLE_PDL,
    atomic_add_release_sys_u32,
    fence_acq_rel_gpu,
    fence_acq_rel_sys,
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    ld_acquire_gpu_u64,
    ld_acquire_sys_u32,
    ld_acquire_sys_u64,
    red_or_release_gpu_u64,
    silu_f32,
    st_release_sys_u64,
    vectorized_atomic_add_bf16x8,
    vectorized_store_bf16x8,
)

# ---------------------------------------------------------------------------
# MegaMoE fused FC1+FC2 phase tags. The scheduler warp writes exactly one phase
# per emitted logical tile; consumer warps read slot 5 to choose the Linear1 or
# Linear2 body.
# ---------------------------------------------------------------------------
PHASE_LINEAR1 = 0
PHASE_LINEAR2 = 1


# ---------------------------------------------------------------------------
# MegaMoE fused FC1+FC2 phase scheduler.
#
# The fused kernel uses one logical tile slot per emitted tile and stores the
# phase (Linear1 or Linear2) in the tile key. The scheduler walks expert waves:
# for each wave it emits all Linear1 tiles first, then the matching Linear2 tiles
# from separate FC2 geometry so later waves can overlap with earlier FC2 work.
# ---------------------------------------------------------------------------

"""
MegaMoE fused FC1+FC2 NVFP4 grouped GEMM kernel for Blackwell
(port of DeepGEMM's sm100_fp8_fp4_mega_moe.cuh to CuTe DSL).

Current fused design:
  - Linear1 performs gather grouped GEMM, SwiGLU, FP4 requant, and HBM pool publication.
  - Linear2 waits on the pool arrival mask, consumes activation and scale-factor pool data,
    performs FC2, and atomically combines the top-k output into BF16.

Upstream C++ references for the fused design:
  impls/sm100_fp8_fp4_mega_moe.cuh   (1364-line mega kernel)
  scheduler/mega_moe.cuh             (BlockPhase state machine, wave scheduler)
  layout/mega_moe.cuh                (arrival counters, pool layout)

Original FC1-only documentation follows.

High-performance persistent blockscaled contiguous grouped dense GEMM with gather and SwiGLU fusion
(C = up * silu(gate), where up and gate come from interleaved weight matrix B)
example for the NVIDIA Blackwell architecture using CUTE DSL.

This kernel performs FC1 layer computation with SwiGLU activation fusion:
1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
3. Optional Quant: When c_dtype is Float4E2M1FN, generates scale factor C and quantizes output

- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension (number of experts)
  - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
- Matrix C is Mx(N/2)x1, C can be row-major("N"), N is halved due to SwiGLU fusion
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk,
  which has M×ceil_div(K, sf_vec_size)×1 elements
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk,
  which has N×ceil_div(K, sf_vec_size)×L elements
- Token ID mapping tensor enables gather operation for A and SFA

Matrix A/C Memory Layout Diagrams:

   ```
    Group 0    Group 1   Group 2
   -+---------+---------+---------+
    |         |         |         |
   K| ValidM0 | ValidM1 | ValidM2 |
    |         |         |         |
   -+---------+---------+---------+
    |<-        ValidM           ->|
   ```
   Note: the Group(L) dimension will be flatted into M dimension, and the rest Group(L) size is 1.
         each ValidM will be aligned to 256 or 128. The alignment is determined by the mma_tiler_mn parameter.
         For NVFP4, 2CTA, the alignment is 256. For NVFP4, 1CTA, the alignment is 128.

This GEMM kernel supports the following features:
    - Utilizes LDGSTS (Load Global to Shared with Swizzle) for A and SFA with gather operation
    - Utilizes Tensor Memory Access (TMA) for B and SFB matrices
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. SCHEDULER warp (warp 10): Dispatches tile information to all consumer warps via tile_info_pipeline.
2. LDGSTS A/SFA warps (warps 4-7):
    - Load A matrix from global memory (GMEM) to shared memory (SMEM) using LDGSTS instructions with gather.
    - Load SFA (scale factor A) from GMEM to SMEM using LDGSTS instructions.
    - Uses token_id_mapping to perform permutation/gather during load.
3. TMA B/SFB warp (warp 9):
    - Load B and SFB matrices from GMEM to SMEM using TMA operations with multicast.
4. MMA warp (warp 8):
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
5. EPILOGUE warps (warps 0-3):
    - Load two accumulator subtiles (up and gate) from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Apply alpha scaling: up_scaled = alpha * up, gate_scaled = alpha * gate
    - Compute SwiGLU activation: output = up_scaled * silu(gate_scaled), where silu(x) = x * sigmoid(x)
    - If c_dtype is Float4E2M1FN: generate scale factor C (SFC) and quantize output
    - Type convert output to c_dtype.
    - Store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the tile_idx_to_expert_idx, token_id_mapping, A/C matrices,
  and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(256-1),
  example: 4096*8 + (256/32)*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - tile_idx_to_expert_idx: padded for invalid tiles (set to -2e9 for padding tiles)
  - token_id_mapping: padded to permuted_m size (invalid tokens set to -1)
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - C matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding:
  - Scheduler warp checks if tile_idx >= num_non_exiting_tiles to exit
  - Only valid tiles (tile_idx < num_non_exiting_tiles) are written to tile_info pipeline
  - LDGSTS warps use token_id_mapping predicates to skip invalid tokens (token_id == -1)
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Consumer warps check initial tile against num_non_exiting_tiles and set
  is_valid_tile=False if tile_idx >= num_non_exiting_tiles
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in C matrix will not be written by the kernel
"""


# TODO: Remove this hook helper function after nvidia-cutlass-dsl 4.4 is released.
def hooked_PersistentTileSchedulerParams_init(
    self,
    problem_shape_ntile_mnl: cute.Shape,
    cluster_shape_mnk: cute.Shape,
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    *,
    loc=None,
    ip=None,
):
    if cluster_shape_mnk[2] != 1:
        raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")
    if swizzle_size < 1:
        raise ValueError(f"expect swizzle_size >= 1, but get {swizzle_size}")

    self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
    # cluster_shape_mnk is kept for reconstruction
    self._cluster_shape_mnk = cluster_shape_mnk
    self.cluster_shape_mn = cluster_shape_mnk[:2]
    self.swizzle_size = swizzle_size
    self._raster_along_m = raster_along_m
    self._loc = loc

    # Apply swizzle if swizzle_size > 1
    if swizzle_size > 1:
        problem_shape_ncluster_mnl = cute.round_up(
            self.problem_layout_ncluster_mnl.shape,
            (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),
        )

        if raster_along_m:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                (
                    problem_shape_ncluster_mnl[0],
                    (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),
                    problem_shape_ncluster_mnl[2],
                ),
                stride=(
                    swizzle_size,
                    (1, swizzle_size * problem_shape_ncluster_mnl[0]),
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        else:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                (
                    (swizzle_size, problem_shape_ncluster_mnl[0] // swizzle_size),
                    problem_shape_ncluster_mnl[1],
                    problem_shape_ncluster_mnl[2],
                ),
                stride=(
                    (1, swizzle_size * problem_shape_ncluster_mnl[1]),
                    swizzle_size,
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )

    # Create FastDivmod divisors (only when swizzle_size == 1 for correctness)
    # FastDivmod assumes simple col-major/row-major layout, incompatible with swizzled layouts
    if swizzle_size == 1:
        problem_shape_ncluster_mnl = cute.ceil_div(
            self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip
        )
        if raster_along_m:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                problem_shape_ncluster_mnl,
                stride=(
                    1,
                    problem_shape_ncluster_mnl[0],
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        else:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                problem_shape_ncluster_mnl,
                stride=(
                    problem_shape_ncluster_mnl[1],
                    1,
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        problem_layout_size = cute.size(self.problem_layout_ncluster_mnl, loc=loc, ip=ip)
        cluster_count_m = self.problem_layout_ncluster_mnl.shape[0]
        cluster_count_n = self.problem_layout_ncluster_mnl.shape[1]

        # batch_fdd: Used to map linear_idx to work_unit_id (handles persistent scheduling)
        self.batch_fdd = cute.fast_divmod_create_divisor(problem_layout_size, loc=loc, ip=ip)

        # cluster_shape_m_fdd: Used to decode work_unit_id to cluster coordinates
        self.cluster_shape_m_fdd = cute.fast_divmod_create_divisor(cluster_count_m, loc=loc, ip=ip)

        # cluster_shape_n_fdd: Used for the second level decomposition
        self.cluster_shape_n_fdd = cute.fast_divmod_create_divisor(cluster_count_n, loc=loc, ip=ip)
    else:
        # FastDivmod not applicable with swizzling, set to None
        self.batch_fdd = None
        self.cluster_shape_m_fdd = None
        self.cluster_shape_n_fdd = None


def hooked_get_cluster_work_idx_with_fastdivmod(
    self, current_work_linear_idx: Int32, *, loc=None, ip=None
) -> Tuple[Int32, Int32, Int32]:
    work_iteration, work_unit_id = divmod(current_work_linear_idx, self.params.batch_fdd)

    if self.params._raster_along_m:
        # raster_along_m=True means column major (m is fastest)
        # First, get cluster_m using cluster_shape_m_fdd
        cluster_n_batch, cluster_m = divmod(work_unit_id, self.params.cluster_shape_m_fdd)

        # Then decode cluster_n_batch to get cluster_n and batch_l using FastDivmod
        batch_l, cluster_n = divmod(cluster_n_batch, self.params.cluster_shape_n_fdd)
    else:
        # raster_along_m=False means row major (n is fastest)
        # First, get cluster_n using cluster_shape_n_fdd
        cluster_m_batch, cluster_n = divmod(work_unit_id, self.params.cluster_shape_n_fdd)

        # Then decode cluster_m_batch to get cluster_m and batch_l using FastDivmod
        batch_l, cluster_m = divmod(cluster_m_batch, self.params.cluster_shape_m_fdd)

    return (cluster_m, cluster_n, batch_l)


cutlass.utils.PersistentTileSchedulerParams.__init__ = hooked_PersistentTileSchedulerParams_init
cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod = (
    hooked_get_cluster_work_idx_with_fastdivmod
)


class BlockScaledMegaMoeFusionKernel:
    """MegaMoE fused FC1+FC2 NVFP4 grouped GEMM kernel for Blackwell.

    Port of DeepGEMM's ``sm100_fp8_fp4_mega_moe.cuh`` to the CuTe DSL.
    The kernel supports a Linear1-only path for regression coverage and a fused
    Linear1+Linear2 path selected by ``enable_linear2``. In fused mode, Linear1
    publishes activation and scale-factor data to HBM pool buffers; Linear2 waits
    on the per-K arrival mask before loading those buffers through its FC2 TMA
    pipeline and combining the top-k output.

    For the original FC1 kernel docstring (gather semantics, data-layout
    constraints, warp-specialisation layout, supported dtypes), see
    ``BlockScaledContiguousGatherGroupedGemmKernel`` in
    ``blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py``.

    This class implements contiguous grouped matrix multiplication with gather operation and SwiGLU fusion
    for FC1 layer computation (C = up * silu(gate), where up/gate come from interleaved GEMM result).

    The computation flow:
    1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
    2. SwiGLU: C = up * silu(gate), extracted from interleaved acc with granularity=64
    3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

    Note: Output C has N/2 columns since pairs of (up, gate) are combined by SwiGLU.

    Key Features:
    - Uses LDGSTS instructions for loading A and SFA matrices with gather/permutation capability
    - Uses TMA (Tensor Memory Access) for loading B and SFB matrices with multicast
    - Token ID mapping enables efficient gather operation during A/SFA load
    - SwiGLU activation fusion in epilogue (up * silu(gate) with interleaved weights)
    - Optional quantization fusion for Float4E2M1FN output with scale factor generation
    - Warp specialization: Scheduler (warp 10), A Sync Transform (warp 11, only used when
      use_2cta_instrs is True), LDGSTS A/SFA (warps 4-7), TMA B/SFB (warp 9), MMA (warp 8),
      Epilogue (warps 0-3)

    :param sf_vec_size: Scalefactor vector size (16 for NVF4, 32 for MXF4/MXF8).
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N).
        Note: use_2cta_instrs is automatically inferred from mma_tiler_mn[0]
        (True when M=256, False when M=128).
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param vectorized_f32: Whether to use vectorized f32x2 operations for better performance.
    :type vectorized_f32: bool

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
        # Note: Float4E2M1FN output includes SFC generation and quantization support for internal testing.
        - Float4E2M1FN (with scale factor generation)

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> # Note: use_2cta_instrs is auto-inferred from mma_tiler_mn[0]
        >>> # (True when M=256, False when M=128)
        >>> gemm = BlockScaledMegaMoeFusionKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),  # use_2cta_instrs=True since M=256
        ...     cluster_shape_mn=(2, 1),
        ...     vectorized_f32=True,
        ... )
        >>> gemm(
        ...     a=a_tensor,
        ...     b=b_tensor,
        ...     c=c_tensor,
        ...     sfa=sfa_tensor,
        ...     sfb=sfb_tensor,
        ...     sfc_tensor=None,
        ...     norm_const_tensor=None,
        ...     tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        ...     tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        ...     token_id_mapping_tensor=token_id_mapping_tensor,
        ...     num_non_exiting_tiles=num_non_exiting_tiles,
        ...     alpha=alpha,
        ...     max_active_clusters=max_active_clusters,
        ...     stream=stream,
        ... )
    """

    # Maximum number of B tensors supported
    MAX_B_TENSORS = 4

    @staticmethod
    def _select_num_experts_per_wave(num_experts: int) -> int:
        """Pick the first M-PA-6 expert-wave size."""
        wave_size = min(4, num_experts)
        while wave_size > 1 and num_experts % wave_size != 0:
            wave_size -= 1
        return wave_size

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        raster_along_m: bool = False,
        b_tensor_l_sizes: Optional[Tuple[int, ...]] = None,
        enable_linear2: bool = False,
        b_tensor_l_sizes_l2: Optional[Tuple[int, ...]] = None,
        mma_tiler_mn_l2: Optional[Tuple[int, int]] = None,
    ):
        """Initializes the configuration for a Blackwell blockscaled dense GEMM kernel with
        gather operation and SwiGLU fusion.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Automatically inferred from mma_tiler_mn[0]
              (True when M=256, False when M=128).

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Scale Factor Configuration:
            - sf_vec_size: Vector size for block-scaled quantization.

        4.  Performance Optimization:
            - vectorized_f32: Enable vectorized f32x2 operations.

        5.  MoE Configuration:
            - topk: Number of experts selected per token (used for token ID mapping).

        :param sf_vec_size: Vector size for scale factors (16 for NVF4, 32 for MXF4/MXF8).
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
            use_2cta_instrs is automatically set based on M (True if M=256, False if M=128).
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param vectorized_f32: Enable vectorized f32x2 operations for better performance.
        :type vectorized_f32: bool
        :param topk: Number of experts selected per token (used for token ID mapping).
        :type topk: cutlass.Int64
        :param b_tensor_l_sizes: Optional tuple of L sizes for each B tensor (FC1 weights).
            E.g., (8, 8, 16) means 3 B tensors with L=8, 8, 16. Sum equals total L.
            If None, single B tensor mode (backward compatible).
        :type b_tensor_l_sizes: Optional[Tuple[int, ...]]
        :param enable_linear2: Master switch for the Linear2 (FC2) fused path.
            False (default) keeps the kernel on the Linear1-only code path.
            True enables the fused FC1+FC2 phase branches and HBM pool hand-off.
        :type enable_linear2: bool
        :param b_tensor_l_sizes_l2: Parallel to ``b_tensor_l_sizes`` for the
            Linear2 weights (FC2). Ignored while ``enable_linear2=False``.
        :type b_tensor_l_sizes_l2: Optional[Tuple[int, ...]]
        :param mma_tiler_mn_l2: Tuple (M, N) shape of the Linear2 (FC2) MMA
            instruction tiler. FC2 has a different N from FC1 (hidden vs
            2*intermediate), so geometry cannot be shared. When
            ``enable_linear2=False`` (or this arg is ``None``), falls back to
            the FC1 shape as a placeholder so ``_setup_attributes`` can run
            unconditionally; all ``_l2`` attributes derived from it live
            behind ``cutlass.const_expr(self.enable_linear2)`` gates on the
            device side and DCE on the Linear1-only path.
        :type mma_tiler_mn_l2: Optional[Tuple[int, int]]
        """

        self.sf_vec_size = sf_vec_size
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        # Linear2 configuration is stored here so the fused path can be
        # configured without changing the Linear1-only entry point.
        self.enable_linear2 = enable_linear2
        self.b_tensor_l_sizes_l2 = b_tensor_l_sizes_l2
        # Linear2 epilogue shape constants are populated only when
        # ``enable_linear2=True`` in ``__call__``. The Linear1-only path keeps
        # these as placeholders behind const_expr-gated code.
        self.final_scale_dtype: Optional[Type[cutlass.Numeric]] = None
        self.epi_loop_size_l2: Optional[int] = None
        self.element_offset_l2: Optional[int] = None
        self.epi_layout_l2: Optional[cute.Layout] = None
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        # Linear2 (FC2) MMA tile shape. FC2 has N = hidden whereas FC1 has
        # N = 2 * intermediate, so geometry cannot be shared. When the caller
        # does not opt into the fused path, fall back to the FC1 shape as a
        # safe placeholder; all ``_l2`` attributes derived from this value
        # are only consumed from ``cutlass.const_expr(self.enable_linear2)``-
        # gated device code, which DCEs cleanly on the Linear1-only path.
        if mma_tiler_mn_l2 is not None:
            self.mma_tiler_l2 = (*mma_tiler_mn_l2, 1)
        else:
            self.mma_tiler_l2 = self.mma_tiler
        self.raster_along_m = raster_along_m

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.sync_transform_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.ldgsts_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                self.sync_transform_warp_id,
            )
        )
        self.warps_wo_sched = (
            len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    self.sync_transform_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
            if self.use_2cta_instrs
            else len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
        )
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
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
        # Synchronizes FC2 consumers after FC1 pool TMA stores are complete.
        phase_sync_warps = 2 + len(self.ldgsts_a_warp_id) + len(self.epilog_warp_id)
        self.phase_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.threads_per_warp * phase_sync_warps,
        )

        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.vectorized_f32 = vectorized_f32

        # Multi-B tensor configuration
        if b_tensor_l_sizes is None:
            self.num_b_tensors = 1
            self.b_tensor_l_sizes = None
            # Offsets padded for safe indexing in kernel
            self.b_tensor_l_offsets = (0,) + (2**30,) * self.MAX_B_TENSORS
        else:
            assert len(b_tensor_l_sizes) <= self.MAX_B_TENSORS, (
                f"Max {self.MAX_B_TENSORS} B tensors, got {len(b_tensor_l_sizes)}"
            )
            self.num_b_tensors = len(b_tensor_l_sizes)
            self.b_tensor_l_sizes = b_tensor_l_sizes
            offsets = [0]
            for l_size in b_tensor_l_sizes:
                offsets.append(offsets[-1] + l_size)
            # Pad to MAX_B_TENSORS + 1 for safe indexing
            while len(offsets) < self.MAX_B_TENSORS + 1:
                offsets.append(2**30)
            self.b_tensor_l_offsets = tuple(offsets)

        self.total_num_experts = (
            self.b_tensor_l_offsets[self.num_b_tensors] if b_tensor_l_sizes is not None else 1
        )
        self.num_experts_per_wave = (
            self._select_num_experts_per_wave(self.total_num_experts)
            if enable_linear2
            else self.total_num_experts
        )

        # combine body dereferences ``self.b_tensor_l_offsets_l2[...]`` to
        # pick the correct FC2 alpha from ``alpha_l2_tuple``; without this
        # the fused path fails with ``AttributeError: no attribute
        # 'b_tensor_l_offsets_l2'``. Fall back to the Linear1 offsets when
        # no dedicated Linear2 sizes are supplied so the placeholder attr
        # is always non-None.
        if b_tensor_l_sizes_l2 is None:
            self.b_tensor_l_offsets_l2 = self.b_tensor_l_offsets
        else:
            assert len(b_tensor_l_sizes_l2) <= self.MAX_B_TENSORS, (
                f"Max {self.MAX_B_TENSORS} FC2 B tensors, got {len(b_tensor_l_sizes_l2)}"
            )
            offsets_l2 = [0]
            for l_size in b_tensor_l_sizes_l2:
                offsets_l2.append(offsets_l2[-1] + l_size)
            while len(offsets_l2) < self.MAX_B_TENSORS + 1:
                offsets_l2.append(2**30)
            self.b_tensor_l_offsets_l2 = tuple(offsets_l2)

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Configure tiled mma
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

        # Compute mma/cluster/tile shapes
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

        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
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

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = (128, 64)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1],
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory.
        # Linear2 coexist inside the same CTA's SharedStorage, so
        # ``_compute_stages`` must not hand the full HW SMEM budget to L1 (it
        # would fill ~228KB by itself and leave L2's own ``_compute_stages``
        # call below to request another ~228KB -> cuLaunchKernel returns
        # CUDA_ERROR_INVALID_VALUE). On the Linear1-only path
        # (``enable_linear2=False``) nothing shares SMEM with L1, so we keep
        # the legacy full budget -- this preserves byte-equal output vs the
        # standalone FC1 kernel.
        smem_capacity_l1 = (
            self.num_smem_capacity // 2 if self.enable_linear2 else self.num_smem_capacity
        )
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
            self.sf_dtype,
            self.sf_vec_size,
            smem_capacity_l1,
            self.occupancy,
        )

        # Compute A/B/C/Scale shared memory layout
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
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.epi_tile_n_required = 2 * cute.size(self.epi_tile[1])
        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n_required

        # ------------------------------------------------------------------
        # Linear2 (FC2) geometry. Python-level ``if`` gate
        # (not ``const_expr``) because this runs inside the compile-time
        # Python interpretation of ``_setup_attributes``; ``self.mma_tiler_l2``
        # etc. are plain Python attrs consumed by subsequent cuteDSL code
        # that is itself gated on ``cutlass.const_expr(self.enable_linear2)``
        # at use sites, so the Linear1-only path (enable_linear2=False) sees
        # neither these attribute writes nor any downstream device code.
        # SMEM layouts, stage counts and TMA descriptors for Linear2 are
        # prepared here and consumed by the phase-specialized kernel paths.
        # ------------------------------------------------------------------
        if self.enable_linear2:
            self.mma_inst_shape_mn_l2 = (
                self.mma_tiler_l2[0],
                self.mma_tiler_l2[1],
            )
            self.mma_inst_shape_mn_sfb_l2 = (
                self.mma_inst_shape_mn_l2[0] // (2 if self.use_2cta_instrs else 1),
                cute.round_up(self.mma_inst_shape_mn_l2[1], 128),
            )

            tiled_mma_l2 = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mn_l2,
            )
            tiled_mma_sfb_l2 = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                self.mma_inst_shape_mn_sfb_l2,
            )
            self.tiled_mma_l2 = tiled_mma_l2
            self.tiled_mma_sfb_l2 = tiled_mma_sfb_l2

            # Fill Linear2 K extent. Same MMA-inst K as FC1 (NVFP4 dictates
            # ``mma_inst_shape_k * mma_inst_tile_k`` per tile); the outer K
            # loop count varies with the FC2 input K (intermediate_size)
            # but that is a runtime shape, not a tile shape.
            self.mma_tiler_l2 = (
                self.mma_tiler_l2[0],
                self.mma_tiler_l2[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )
            self.mma_tiler_sfa_l2 = (
                self.mma_inst_shape_mn_l2[0],
                self.mma_inst_shape_mn_l2[1],
                mma_inst_shape_k * mma_inst_tile_k // 16,
            )
            self.mma_tiler_sfb_l2 = (
                self.mma_inst_shape_mn_sfb_l2[0],
                self.mma_inst_shape_mn_sfb_l2[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )

            self.cta_tile_shape_mnk_l2 = (
                self.mma_tiler_l2[0] // cute.size(tiled_mma_l2.thr_id.shape),
                self.mma_tiler_l2[1],
                self.mma_tiler_l2[2],
            )
            self.cta_tile_shape_mnk_sfa_l2 = (
                self.mma_tiler_sfa_l2[0] // cute.size(tiled_mma_l2.thr_id.shape),
                self.mma_tiler_sfa_l2[1],
                self.mma_tiler_sfa_l2[2],
            )
            self.cta_tile_shape_mnk_sfb_l2 = (
                self.mma_tiler_sfb_l2[0] // cute.size(tiled_mma_l2.thr_id.shape),
                self.mma_tiler_sfb_l2[1],
                self.mma_tiler_sfb_l2[2],
            )
            # FC2 produces a full-N output (no SwiGLU halving), so reuse the
            # MxN tile directly for epilogue accounting. Explicit aliases
            # keep future s2/s3 edits grep-friendly against the FC1 ``_c``
            # variants.
            self.mma_tiler_c_l2 = self.mma_tiler_l2
            self.cta_tile_shape_mnk_c_l2 = self.cta_tile_shape_mnk_l2
            # FC1 publishes one arrival bit per pool-store N tile. FC2 consumes
            # K tiles from the same pool, so each FC2 K tile must wait for all
            # FC1 pool N tiles covered by that K span. With the current default
            # tactic this is 256 / 64 = four bits per FC2 K tile.
            if self.cta_tile_shape_mnk_l2[2] % self.cta_tile_shape_mnk_c[1] != 0:
                raise ValueError(
                    "FC2 K tile must cover an integer number of FC1 pool N tiles: "
                    f"{self.cta_tile_shape_mnk_l2[2]} vs {self.cta_tile_shape_mnk_c[1]}"
                )
            self.l2_arrival_bits_per_k_tile = (
                self.cta_tile_shape_mnk_l2[2] // self.cta_tile_shape_mnk_c[1]
            )
            self.l2_arrival_mask_per_k_tile = (1 << self.l2_arrival_bits_per_k_tile) - 1

            self.cluster_layout_vmnk_l2 = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma_l2.thr_id.shape,),
            )
            self.cluster_layout_sfb_vmnk_l2 = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma_sfb_l2.thr_id.shape,),
            )

            self.num_mcast_ctas_a_l2 = cute.size(self.cluster_layout_vmnk_l2.shape[2])
            self.num_mcast_ctas_b_l2 = cute.size(self.cluster_layout_vmnk_l2.shape[1])
            self.is_a_mcast_l2 = self.num_mcast_ctas_a_l2 > 1
            self.is_b_mcast_l2 = self.num_mcast_ctas_b_l2 > 1

            # TMEM column accounting mirrors FC1 structure for the FC2
            # accumulator and scale-factor fragments.
            sf_atom_mn_l2 = 32
            self.num_sfa_tmem_cols_l2 = (
                self.cta_tile_shape_mnk_l2[0] // sf_atom_mn_l2
            ) * mma_inst_tile_k
            self.num_sfb_tmem_cols_l2 = (
                self.cta_tile_shape_mnk_sfb_l2[1] // sf_atom_mn_l2
            ) * mma_inst_tile_k
            self.num_sf_tmem_cols_l2 = self.num_sfa_tmem_cols_l2 + self.num_sfb_tmem_cols_l2

            # --------------------------------------------------------------
            # Linear2 stage counts + staged SMEM layouts.
            # ``_compute_stages`` uses the L2 MMA tiler and the L2 output
            # dtype (seeded in ``__call__`` before this method runs) to
            # size SMEM budget for the FC2 pipeline. Staged SMEM layouts
            # are computed here so the ``__call__`` setup block can build
            # TMA descriptors; actual ``SharedStorage`` slots referencing
            # them feed the Linear2 shared-storage and TMA setup below.
            # --------------------------------------------------------------
            # the mirror comment above the L1 ``_compute_stages`` call), so we
            # pass the complementary half of the HW budget here. A 50/50 split
            # is conservative: FC1 typically has a larger N (hidden * 2 for
            # SwiGLU) than FC2, so the L1 pipeline may occupy slightly more
            # than half the share, but the residual slack in both budgets is
            # enough to keep ``num_ab_stage_l2 >= 2`` on the tested
            # (H=512/I=512, H=1024/I=1024) shapes. Future work can split
            # proportional to ``ab_bytes_per_stage`` once FC2 tiling lands.
            smem_capacity_l2 = self.num_smem_capacity // 2
            (
                self.num_acc_stage_l2,
                self.num_ab_stage_l2,
                self.num_c_stage_l2,
                self.num_tile_stage_l2,
            ) = self._compute_stages(
                tiled_mma_l2,
                self.mma_tiler_l2,
                self.a_dtype,
                self.b_dtype,
                self.epi_tile,
                self.l2_out_dtype,
                self.l2_output_layout,
                self.sf_dtype,
                self.sf_vec_size,
                smem_capacity_l2,
                self.occupancy,
            )

            self.a_smem_layout_staged_l2 = sm100_utils.make_smem_layout_a(
                tiled_mma_l2,
                self.mma_tiler_l2,
                self.a_dtype,
                self.num_ab_stage_l2,
            )
            self.b_smem_layout_staged_l2 = sm100_utils.make_smem_layout_b(
                tiled_mma_l2,
                self.mma_tiler_l2,
                self.b_dtype,
                self.num_ab_stage_l2,
            )
            self.sfa_smem_layout_staged_l2 = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_l2,
                self.mma_tiler_l2,
                self.sf_vec_size,
                self.num_ab_stage_l2,
            )
            self.sfb_smem_layout_staged_l2 = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma_l2,
                self.mma_tiler_l2,
                self.sf_vec_size,
                self.num_ab_stage_l2,
            )
            self.c_smem_layout_staged_l2 = sm100_utils.make_smem_layout_epi(
                self.l2_out_dtype,
                self.l2_output_layout,
                self.epi_tile,
                self.num_c_stage_l2,
            )

            # L1 and L2 keep their separately computed AB stage counts. Both
            # non-2CTA and 2CTA paths create phase-specific L2 A pipeline and
            # sync-transform state below, so FC2 no longer has to inherit the
            # Linear1 AB stage count.

            self.overlapping_accum_l2 = self.num_acc_stage_l2 == 1
            self.num_accumulator_tmem_cols_l2 = (
                self.cta_tile_shape_mnk_l2[1] * self.num_acc_stage_l2
                if not self.overlapping_accum_l2
                else self.cta_tile_shape_mnk_l2[1] * 2 - self.num_sf_tmem_cols_l2
            )
            self.iter_acc_early_release_in_epilogue_l2 = (
                self.num_sf_tmem_cols_l2 // self.epi_tile_n_required
            )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: Union[cute.Tensor, Tuple[cute.Tensor, ...]],
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: Union[cute.Tensor, Tuple[cute.Tensor, ...]],
        sfc_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: Union[cute.Tensor, Tuple[cute.Tensor, ...]],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        a_bytes: Optional[cute.Tensor] = None,
        # ------------------------------------------------------------------
        # Linear2 (FC2) inputs. Defaults keep the Linear1-only path active.
        # ------------------------------------------------------------------
        b_l2: Optional[Union[cute.Tensor, Tuple[cute.Tensor, ...]]] = None,
        sfb_l2: Optional[Union[cute.Tensor, Tuple[cute.Tensor, ...]]] = None,
        out: Optional[cute.Tensor] = None,
        scatter_out: Optional[cute.Tensor] = None,
        alpha_l2: Optional[Union[cute.Tensor, Tuple[cute.Tensor, ...]]] = None,
        permuted_idx_to_expanded_idx: Optional[cute.Tensor] = None,
        token_final_scales: Optional[cute.Tensor] = None,
        pool_tensor: Optional[cute.Tensor] = None,
        pool_sfc_tensor: Optional[cute.Tensor] = None,
        l2_arrival_mask: Optional[cute.Tensor] = None,
        direct_combine_output: cutlass.Constexpr = False,
        direct_combine_atomic_output: cutlass.Constexpr = False,
        direct_combine_token_major_output: cutlass.Constexpr = False,
        combine_output_ep_size: cutlass.Constexpr = 1,
        combine_output_top_k: cutlass.Constexpr = 1,
        combine_output_max_num_tokens_per_rank: cutlass.Constexpr = 0,
        combine_output_hidden_size: cutlass.Constexpr = 0,
        monolithic_reduce_output: cutlass.Constexpr = False,
        monolithic_final_output: Optional[cute.Tensor] = None,
        monolithic_control: Optional[cute.Tensor] = None,
        monolithic_local_rank: cutlass.Constexpr = 0,
        monolithic_local_tokens: cutlass.Int64 = 0,
        monolithic_grid_sync_blocks: cutlass.Constexpr = 0,
        monolithic_direct_topk_materialize: cutlass.Constexpr = False,
        monolithic_direct_topk_source_input: cutlass.Constexpr = False,
        monolithic_direct_topk_input: Optional[cute.Tensor] = None,
        monolithic_direct_topk_input_fp4: Optional[cute.Tensor] = None,
        monolithic_direct_topk_input_scale: Optional[cute.Tensor] = None,
        monolithic_direct_topk_input_rank_stride_fp4: cutlass.Constexpr = 0,
        monolithic_direct_topk_input_scale_rank_stride_elements: cutlass.Constexpr = 0,
        monolithic_direct_topk_idx: Optional[cute.Tensor] = None,
        monolithic_direct_topk_scales: Optional[cute.Tensor] = None,
        monolithic_direct_topk_token_counts: Optional[cute.Tensor] = None,
        monolithic_direct_topk_local_input: Optional[cute.Tensor] = None,
        monolithic_direct_topk_local_input_scale: Optional[cute.Tensor] = None,
        monolithic_direct_topk_local_idx: Optional[cute.Tensor] = None,
        monolithic_direct_topk_local_scales: Optional[cute.Tensor] = None,
        monolithic_direct_topk_local_expert_offset: cutlass.Constexpr = 0,
        monolithic_direct_topk_num_local_experts: cutlass.Constexpr = 0,
        monolithic_direct_topk_stage_inputs: cutlass.Constexpr = False,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        This method performs FC1 layer computation:
        1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
        2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded using LDGSTS instructions with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - LDGSTS warps: Load A and SFA with gather
           - A Sync Transform warps: Transform the sync signal of A and SFA from global to
             shared memory when use_2cta_instrs is True
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply SwiGLU activation, optional quantization, and store results

        :param a: Input tensor A (MxKx1), will be gathered using token_id_mapping
        :type a: cute.Tensor
        :param b: Input tensor B (NxKxL), L is the number of experts/groups, weights are interleaved for SwiGLU
        :type b: cute.Tensor
        :param c: Output tensor C (Mx(N/2)x1), N is halved due to SwiGLU fusion
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A, will be gathered using token_id_mapping
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param sfc_tensor: Scale factor tensor C for quantized output (None if not quantizing)
        :type sfc_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Normalization constant for scale factor generation
            (None if not quantizing)
        :type norm_const_tensor: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param token_id_mapping_tensor: Token ID mapping for gather operation, shape (permuted_m,)
        :type token_id_mapping_tensor: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        # Handle tuple of B tensors
        b_tuple = b if isinstance(b, tuple) else (b,)
        sfb_tuple = sfb if isinstance(sfb, tuple) else (sfb,)
        self.b_dtype: Type[cutlass.Numeric] = b_tuple[0].element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tuple[0]).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        # Linear2 (FC2) output tensor metadata. Derived from the optional
        # ``out`` argument only when ``enable_linear2=True``; otherwise seeded
        # with safe Linear1-path defaults so that the device-side helper
        # ``epilog_tmem_copy_and_partition_linear2`` has attribute targets to
        # bind against. Its invocation is gated on
        # ``cutlass.const_expr(self.enable_linear2)`` and therefore DCE'd on
        # the Linear1-only path. The fused path supplies a real BF16 output tensor.
        self.l2_out_dtype: Type[cutlass.Numeric] = (
            out.element_type if cutlass.const_expr(self.enable_linear2) else cutlass.BFloat16
        )
        self.l2_output_layout = self.c_layout
        if cutlass.const_expr(self.enable_linear2):
            if cutlass.const_expr(direct_combine_output):
                self.l2_output_layout = self.c_layout
            else:
                self.l2_output_layout = utils.LayoutEnum.from_tensor(out)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensors - create layout for each B tensor (use const_expr, not loop)
        sfb_layout_0 = blockscaled_utils.tile_atom_to_shape_SF(b_tuple[0].shape, self.sf_vec_size)
        sfb_tensor_0 = cute.make_tensor(sfb_tuple[0].iterator, sfb_layout_0)
        sfb_tensors = [sfb_tensor_0]
        if cutlass.const_expr(self.num_b_tensors >= 2):
            sfb_layout_1 = blockscaled_utils.tile_atom_to_shape_SF(
                b_tuple[1].shape, self.sf_vec_size
            )
            sfb_tensors.append(cute.make_tensor(sfb_tuple[1].iterator, sfb_layout_1))
        if cutlass.const_expr(self.num_b_tensors >= 3):
            sfb_layout_2 = blockscaled_utils.tile_atom_to_shape_SF(
                b_tuple[2].shape, self.sf_vec_size
            )
            sfb_tensors.append(cute.make_tensor(sfb_tuple[2].iterator, sfb_layout_2))
        if cutlass.const_expr(self.num_b_tensors >= 4):
            sfb_layout_3 = blockscaled_utils.tile_atom_to_shape_SF(
                b_tuple[3].shape, self.sf_vec_size
            )
            sfb_tensors.append(cute.make_tensor(sfb_tuple[3].iterator, sfb_layout_3))
        sfb_tuple = tuple(sfb_tensors)
        # Backward compat alias
        sfb = sfb_tuple[0]

        # Setup sfc tensor by filling C tensor to scale factor atom layout.
        # in fused mode (``enable_linear2=True``) the FC1
        # requant SFC path writes into ``pool_sfc_tensor`` instead of a
        # standalone ``sfc_tensor``; ``norm_const_tensor`` is still required
        # because the epilogue references it via ``norm_const_tensor[0]``.
        # Extending the gate keeps the Linear1-only path behaviour unchanged
        # (``enable_linear2=False`` short-circuits the fused disjunct) and
        # prevents the pool SFC store path from being DCE'd when only the
        # pool buffer is supplied.
        self.generate_sfc = norm_const_tensor is not None and (
            sfc_tensor is not None or (self.enable_linear2 and pool_sfc_tensor is not None)
        )
        if cutlass.const_expr(self.generate_sfc):
            sfc_ref = sfc_tensor if sfc_tensor is not None else pool_sfc_tensor
            sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(c.shape, self.sf_vec_size)
            sfc_tensor = cute.make_tensor(sfc_ref.iterator, sfc_layout)

        # into the kernel twice — as the FC1 epilogue SFC destination
        # (``mSFC_eff = pool_sfc_tensor_l1`` under the ``enable_linear2``
        # const_expr branch) AND as the FC2 LDGSTS-A source
        # (``gPoolSFC_mkl = cute.local_tile(pool_sfc_tensor_l1, ...)``). The
        # FC2 path expects the same 3D (M, K, L) logical shape as
        # ``mSFA_mkl`` (built via ``tile_atom_to_shape_SF``), whereas the
        # wrapper's raw 6D ordered-layout make_tensor (convenient for the
        # C_SF ABI) does not satisfy that contract. Rebuild the pool SFC
        # layout here with the same shape derivation as ``sfc_tensor``
        # above; the Linear1-only path DCE's this branch because
        # ``self.enable_linear2`` is False.
        if cutlass.const_expr(self.enable_linear2 and pool_sfc_tensor is not None):
            pool_sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(c.shape, self.sf_vec_size)
            pool_sfc_tensor = cute.make_tensor(pool_sfc_tensor.iterator, pool_sfc_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
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

        # Setup TMA ops (shared across all B tensors)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))

        # Helper to create TMA for one B tensor
        def _make_tma_b(b_tensor, sfb_tensor):
            atom_b, tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                b_tensor,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            atom_sfb, tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # Handle overlapping layout for SFB when cta_tile_shape_n=192
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                x = tensor_sfb.stride[0][1]
                y = cute.ceil_div(tensor_sfb.shape[0][1], 4)
                new_shape = (
                    (tensor_sfb.shape[0][0], ((2, 2), y)),
                    tensor_sfb.shape[1],
                    tensor_sfb.shape[2],
                )
                x_times_3 = 3 * x
                new_stride = (
                    (tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                    tensor_sfb.stride[1],
                    tensor_sfb.stride[2],
                )
                tensor_sfb = cute.make_tensor(
                    tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
                )
            return atom_b, tensor_b, atom_sfb, tensor_sfb

        # Create TMA for all B tensors (use const_expr, not loop)
        atom_b_0, tensor_b_0, atom_sfb_0, tensor_sfb_0 = _make_tma_b(b_tuple[0], sfb_tuple[0])
        tma_atoms_b = [atom_b_0]
        tma_tensors_b = [tensor_b_0]
        tma_atoms_sfb = [atom_sfb_0]
        tma_tensors_sfb = [tensor_sfb_0]
        if cutlass.const_expr(self.num_b_tensors >= 2):
            atom_b_1, tensor_b_1, atom_sfb_1, tensor_sfb_1 = _make_tma_b(b_tuple[1], sfb_tuple[1])
            tma_atoms_b.append(atom_b_1)
            tma_tensors_b.append(tensor_b_1)
            tma_atoms_sfb.append(atom_sfb_1)
            tma_tensors_sfb.append(tensor_sfb_1)
        if cutlass.const_expr(self.num_b_tensors >= 3):
            atom_b_2, tensor_b_2, atom_sfb_2, tensor_sfb_2 = _make_tma_b(b_tuple[2], sfb_tuple[2])
            tma_atoms_b.append(atom_b_2)
            tma_tensors_b.append(tensor_b_2)
            tma_atoms_sfb.append(atom_sfb_2)
            tma_tensors_sfb.append(tensor_sfb_2)
        if cutlass.const_expr(self.num_b_tensors >= 4):
            atom_b_3, tensor_b_3, atom_sfb_3, tensor_sfb_3 = _make_tma_b(b_tuple[3], sfb_tuple[3])
            tma_atoms_b.append(atom_b_3)
            tma_tensors_b.append(tensor_b_3)
            tma_atoms_sfb.append(atom_sfb_3)
            tma_tensors_sfb.append(tensor_sfb_3)
        tma_atoms_b = tuple(tma_atoms_b)
        tma_tensors_b = tuple(tma_tensors_b)
        tma_atoms_sfb = tuple(tma_atoms_sfb)
        tma_tensors_sfb = tuple(tma_tensors_sfb)

        # Handle alpha tuple (convert to tuple if single tensor)
        alpha_tuple = alpha if isinstance(alpha, tuple) else (alpha,)

        # build ``alpha_l2_tuple`` for the Linear2 epilogue. When
        # ``enable_linear2=True`` the caller must supply ``alpha_l2``; we
        # normalize to a tuple for the L2 ``alpha_l2_tuple[0][...]`` subscript.
        # On the Linear1-only path ``alpha_l2`` is ``None``; fall back to
        # ``alpha_tuple`` so the kernel receives a well-typed
        # ``Tuple[cute.Tensor, ...]`` that DCE elides inside the L2 ``else:``
        # branch via ``cutlass.const_expr(not self.enable_linear2)``.
        if cutlass.const_expr(self.enable_linear2):
            alpha_l2_tuple = alpha_l2 if isinstance(alpha_l2, tuple) else (alpha_l2,)
        else:
            alpha_l2_tuple = alpha_tuple

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size
        self.num_tma_load_bytes_l2 = self.num_tma_load_bytes

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # ------------------------------------------------------------------
        # Linear2 (FC2) TMA B/SFB descriptors, TMA store
        # for ``out``, and final-scale / epi_layout metadata. All work is
        # gated on ``cutlass.const_expr(self.enable_linear2)`` so the FC1
        # Linear1-only path walks a no-op: locals seed to ``None`` and the four
        # ``Optional`` epi attrs stay as their ``__init__`` placeholders.
        # ``SharedStorage`` slots for B/SFB, pool activation, and output tiles
        # are gated under the same Linear2 enable flag.
        # ------------------------------------------------------------------
        tma_atoms_b_l2 = None
        tma_tensors_b_l2 = None
        tma_atoms_sfb_l2 = None
        tma_tensors_sfb_l2 = None
        tma_atom_pool_sfa_l2 = None
        tma_tensor_pool_sfa_l2 = None
        self.num_tma_sfa_pool_load_bytes = 0
        tma_atom_out = None
        tma_tensor_out = None
        # pool store atoms start as ``None``; populated
        # inside the enable_linear2 block below.
        tma_atom_pool_store = None
        tma_tensor_pool_store = None
        if cutlass.const_expr(self.enable_linear2):
            b_l2_input = b_l2 if isinstance(b_l2, tuple) else (b_l2,)
            sfb_l2_input = sfb_l2 if isinstance(sfb_l2, tuple) else (sfb_l2,)

            # Apply block-scaled SF atom layout to each FC2 SFB tensor, mirroring
            # the FC1 handling above. ``num_b_tensors`` is shared between FC1
            # and FC2 today; per-shard L2 weights would extend this when the
            # caller supplies ``b_tensor_l_sizes_l2``.
            sfb_l2_layout_0 = blockscaled_utils.tile_atom_to_shape_SF(
                b_l2_input[0].shape, self.sf_vec_size
            )
            sfb_l2_tensor_0 = cute.make_tensor(sfb_l2_input[0].iterator, sfb_l2_layout_0)
            sfb_l2_tensors = [sfb_l2_tensor_0]
            if cutlass.const_expr(self.num_b_tensors >= 2):
                sfb_l2_layout_1 = blockscaled_utils.tile_atom_to_shape_SF(
                    b_l2_input[1].shape, self.sf_vec_size
                )
                sfb_l2_tensors.append(cute.make_tensor(sfb_l2_input[1].iterator, sfb_l2_layout_1))
            if cutlass.const_expr(self.num_b_tensors >= 3):
                sfb_l2_layout_2 = blockscaled_utils.tile_atom_to_shape_SF(
                    b_l2_input[2].shape, self.sf_vec_size
                )
                sfb_l2_tensors.append(cute.make_tensor(sfb_l2_input[2].iterator, sfb_l2_layout_2))
            if cutlass.const_expr(self.num_b_tensors >= 4):
                sfb_l2_layout_3 = blockscaled_utils.tile_atom_to_shape_SF(
                    b_l2_input[3].shape, self.sf_vec_size
                )
                sfb_l2_tensors.append(cute.make_tensor(sfb_l2_input[3].iterator, sfb_l2_layout_3))
            sfb_l2_tuple = tuple(sfb_l2_tensors)

            b_l2_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, self.tiled_mma_l2.thr_id
            )
            sfb_l2_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, self.tiled_mma_l2.thr_id
            )
            b_l2_smem_layout = cute.slice_(self.b_smem_layout_staged_l2, (None, None, None, 0))
            sfb_l2_smem_layout = cute.slice_(self.sfb_smem_layout_staged_l2, (None, None, None, 0))
            b_l2_copy_size = cute.size_in_bytes(self.b_dtype, b_l2_smem_layout)
            sfb_l2_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_l2_smem_layout)
            self.num_tma_load_bytes_l2 = (b_l2_copy_size + sfb_l2_copy_size) * atom_thr_size
            pool_sfa_l2_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, self.tiled_mma_l2.thr_id
            )
            pool_sfa_l2_smem_layout = cute.slice_(
                self.sfa_smem_layout_staged_l2, (None, None, None, 0)
            )
            pool_sfa_l2_source = pool_sfc_tensor if pool_sfc_tensor is not None else sfa
            tma_atom_pool_sfa_l2, tma_tensor_pool_sfa_l2 = cute.nvgpu.make_tiled_tma_atom_A(
                pool_sfa_l2_op,
                pool_sfa_l2_source,
                pool_sfa_l2_smem_layout,
                self.mma_tiler_l2,
                self.tiled_mma_l2,
                self.cluster_layout_vmnk_l2.shape,
                internal_type=cutlass.Int16,
            )
            sfa_pool_copy_size = cute.size_in_bytes(self.sf_dtype, pool_sfa_l2_smem_layout)
            self.num_tma_sfa_pool_load_bytes = sfa_pool_copy_size * atom_thr_size

            def _make_tma_b_l2(b_tensor, sfb_tensor):
                atom_b, tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                    b_l2_op,
                    b_tensor,
                    b_l2_smem_layout,
                    self.mma_tiler_l2,
                    self.tiled_mma_l2,
                    self.cluster_layout_vmnk_l2.shape,
                )
                atom_sfb, tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                    sfb_l2_op,
                    sfb_tensor,
                    sfb_l2_smem_layout,
                    self.mma_tiler_sfb_l2,
                    self.tiled_mma_sfb_l2,
                    self.cluster_layout_sfb_vmnk_l2.shape,
                    internal_type=cutlass.Int16,
                )
                return atom_b, tensor_b, atom_sfb, tensor_sfb

            atom_b_l2_0, tensor_b_l2_0, atom_sfb_l2_0, tensor_sfb_l2_0 = _make_tma_b_l2(
                b_l2_input[0], sfb_l2_tuple[0]
            )
            tma_atoms_b_l2_list = [atom_b_l2_0]
            tma_tensors_b_l2_list = [tensor_b_l2_0]
            tma_atoms_sfb_l2_list = [atom_sfb_l2_0]
            tma_tensors_sfb_l2_list = [tensor_sfb_l2_0]
            if cutlass.const_expr(self.num_b_tensors >= 2):
                a_b, t_b, a_s, t_s = _make_tma_b_l2(b_l2_input[1], sfb_l2_tuple[1])
                tma_atoms_b_l2_list.append(a_b)
                tma_tensors_b_l2_list.append(t_b)
                tma_atoms_sfb_l2_list.append(a_s)
                tma_tensors_sfb_l2_list.append(t_s)
            if cutlass.const_expr(self.num_b_tensors >= 3):
                a_b, t_b, a_s, t_s = _make_tma_b_l2(b_l2_input[2], sfb_l2_tuple[2])
                tma_atoms_b_l2_list.append(a_b)
                tma_tensors_b_l2_list.append(t_b)
                tma_atoms_sfb_l2_list.append(a_s)
                tma_tensors_sfb_l2_list.append(t_s)
            if cutlass.const_expr(self.num_b_tensors >= 4):
                a_b, t_b, a_s, t_s = _make_tma_b_l2(b_l2_input[3], sfb_l2_tuple[3])
                tma_atoms_b_l2_list.append(a_b)
                tma_tensors_b_l2_list.append(t_b)
                tma_atoms_sfb_l2_list.append(a_s)
                tma_tensors_sfb_l2_list.append(t_s)
            # Locals are threaded into ``self.kernel(...)`` below. ``noqa: F841``
            # is the same idiom used for placeholder locals.
            tma_atoms_b_l2 = tuple(tma_atoms_b_l2_list)  # noqa: F841
            tma_tensors_b_l2 = tuple(tma_tensors_b_l2_list)  # noqa: F841
            tma_atoms_sfb_l2 = tuple(tma_atoms_sfb_l2_list)  # noqa: F841
            tma_tensors_sfb_l2 = tuple(tma_tensors_sfb_l2_list)  # noqa: F841

            # TMA store for the Linear2 BF16 output tensor. FC2 scatters at
            # the full (M, N) tile granularity (no SwiGLU halving), so we
            # slice ``c_smem_layout_staged_l2`` with the same epi_tile used
            # for FC1.
            epi_smem_layout_l2 = cute.slice_(self.c_smem_layout_staged_l2, (None, None, 0))
            tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out,
                epi_smem_layout_l2,
                self.epi_tile,
            )

            # FC1 epilogue HBM pool store TMA atom. Shares
            # the FC1 ``epi_smem_layout`` (``sC`` staging) because the FC1
            # requant path writes FP4 to ``sC`` before S2G streaming; for
            # ``enable_linear2=True`` we redirect that S2G stream from the
            # standalone FC1 ``c`` tensor to the permuted HBM pool consumed
            # by the FC2 LDGSTS-A warp. Built only under the enable_linear2
            # gate so the Linear1-only path allocates no extra TMA descriptor
            # bytes. ``epi_smem_layout`` is valid for both destinations.
            tma_atom_pool_store, tma_tensor_pool_store = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                pool_tensor,
                epi_smem_layout,
                self.epi_tile,
            )

            # Populate the four ``Optional`` L2 epilogue metadata attrs
            # declared in ``__init__``. These drive the tmem->gmem copy
            # layout inside the Linear2 combine body.
            self.final_scale_dtype = token_final_scales.element_type
            epi_tile_size_l2 = cute.size(self.epi_tile[0]) * cute.size(self.epi_tile[1])
            num_epilogue_threads_l2 = 32 * len(self.epilog_warp_id)
            ttr_racc_size_l2 = epi_tile_size_l2 // num_epilogue_threads_l2
            if cutlass.const_expr(self.l2_out_dtype == cutlass.BFloat16):
                self.epi_layout_l2 = cute.make_layout(
                    shape=(ttr_racc_size_l2 // 8, 4, 2), stride=(8, 2, 1)
                )
                self.epi_loop_size_l2 = ttr_racc_size_l2 // 8
                self.element_offset_l2 = 8
            elif cutlass.const_expr(self.l2_out_dtype == cutlass.Float32):
                self.epi_layout_l2 = cute.make_layout(
                    shape=(ttr_racc_size_l2 // 2, 2), stride=(2, 1)
                )
                self.epi_loop_size_l2 = ttr_racc_size_l2 // 2
                self.element_offset_l2 = 2
            else:
                self.epi_layout_l2 = cute.make_layout(shape=(ttr_racc_size_l2,), stride=(1,))
                self.epi_loop_size_l2 = ttr_racc_size_l2
                self.element_offset_l2 = 1

        # Compute grid size for Linear1. Linear2 uses the same M tile map but
        # must derive its N geometry from the FC2 output width, not from the
        # FC1 pool width. Keep the L2 params behind the enable gate so the
        # Linear1-only path receives only typed placeholders.
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )
        self.tile_sched_params_l2 = self.tile_sched_params
        if cutlass.const_expr(self.enable_linear2):
            self.tile_sched_params_l2, _ = self._compute_grid_from_shape(
                (c.shape[0], out.shape[1], c.shape[2]),
                self.cta_tile_shape_mnk_c_l2,
                self.cluster_shape_mn,
                max_active_clusters,
                self.raster_along_m,
            )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage1cta:
            # (bidx, bidy, bidz, valid, mn_limit, phase)  - slot 5 stores the phase
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            # --------------------------------------------------------------
            # Linear2 SMEM slots. Guarded on
            # ``const_expr(self.enable_linear2)`` so the Linear1-only path's
            # ``SharedStorage`` layout is bit-identical to the pre-S1c
            # version (cute.struct respects the const_expr gate and elides
            # the fields entirely when the flag is False -- same pattern
            # the FC2 reference kernel uses for its ``use_blkred`` slot).
            # --------------------------------------------------------------
            if cutlass.const_expr(self.enable_linear2):
                a_pool_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage_l2 * 2]
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_l2_mbar_ptr: cute.struct.MemRange[
                        cutlass.Int64,
                        self.num_ab_stage_l2 * 2,
                    ]
                b_l2_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage_l2 * 2]
                acc_l2_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage_l2 * 2]
                sfa_pool_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage_l2 * 2]
                # FC2 weights and scale-factor B tiles.
                # (MMA, MMA_N_L2, MMA_K, STAGE_L2)
                sB_l2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype,
                        cute.cosize(self.b_smem_layout_staged_l2.outer),
                    ],
                    self.buffer_align_bytes,
                ]
                # (granularity_n_l2, repeat_n), (granularity_k, repeat_k), STAGE_L2)
                sSFB_l2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype,
                        cute.cosize(self.sfb_smem_layout_staged_l2),
                    ],
                    self.buffer_align_bytes,
                ]
                # FC1 -> FC2 hand-off staging. Loader warps pull FC1's
                # requantized FP4 activations from the HBM pool into these
                # SMEM slots before the FC2 MMA consumes them. Under the
                # naive ``all-L1-then-all-L2 per wave`` scheduler (plumbed
                # in S1d) the A/SFA MMA pipeline is quiesced between
                # phases, so these staging slots are dedicated FC2 real
                # estate rather than aliases of sA/sSFA.
                sA_pool: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype,
                        cute.cosize(self.a_smem_layout_staged_l2.outer),
                    ],
                    self.buffer_align_bytes,
                ]
                sSFA_pool: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype,
                        cute.cosize(self.sfa_smem_layout_staged_l2),
                    ],
                    self.buffer_align_bytes,
                ]
                # FC2 BF16 output staging ahead of the TMA S2G store built
                # in ``__call__`` (``tma_atom_out``/``tma_tensor_out``).
                sC_l2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.l2_out_dtype,
                        cute.cosize(self.c_smem_layout_staged_l2.outer),
                    ],
                    self.buffer_align_bytes,
                ]

        @cute.struct
        class SharedStorage2cta:
            # (bidx, bidy, bidz, valid, mn_limit, phase)  - slot 5 stores the phase
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            # --------------------------------------------------------------
            # Linear2 SMEM slots. Guarded on
            # ``const_expr(self.enable_linear2)`` so the Linear1-only path's
            # ``SharedStorage`` layout is bit-identical to the pre-S1c
            # version (cute.struct respects the const_expr gate and elides
            # the fields entirely when the flag is False -- same pattern
            # the FC2 reference kernel uses for its ``use_blkred`` slot).
            # --------------------------------------------------------------
            if cutlass.const_expr(self.enable_linear2):
                a_pool_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage_l2 * 2]
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_l2_mbar_ptr: cute.struct.MemRange[
                        cutlass.Int64,
                        self.num_ab_stage_l2 * 2,
                    ]
                b_l2_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage_l2 * 2]
                acc_l2_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage_l2 * 2]
                sfa_pool_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage_l2 * 2]
                # FC2 weights and scale-factor B tiles.
                # (MMA, MMA_N_L2, MMA_K, STAGE_L2)
                sB_l2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype,
                        cute.cosize(self.b_smem_layout_staged_l2.outer),
                    ],
                    self.buffer_align_bytes,
                ]
                # (granularity_n_l2, repeat_n), (granularity_k, repeat_k), STAGE_L2)
                sSFB_l2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype,
                        cute.cosize(self.sfb_smem_layout_staged_l2),
                    ],
                    self.buffer_align_bytes,
                ]
                # FC1 -> FC2 hand-off staging. Loader warps pull FC1's
                # requantized FP4 activations from the HBM pool into these
                # SMEM slots before the FC2 MMA consumes them. Under the
                # naive ``all-L1-then-all-L2 per wave`` scheduler (plumbed
                # in S1d) the A/SFA MMA pipeline is quiesced between
                # phases, so these staging slots are dedicated FC2 real
                # estate rather than aliases of sA/sSFA.
                sA_pool: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype,
                        cute.cosize(self.a_smem_layout_staged_l2.outer),
                    ],
                    self.buffer_align_bytes,
                ]
                sSFA_pool: cute.struct.Align[
                    cute.struct.MemRange[
                        self.sf_dtype,
                        cute.cosize(self.sfa_smem_layout_staged_l2),
                    ],
                    self.buffer_align_bytes,
                ]
                # FC2 BF16 output staging ahead of the TMA S2G store built
                # in ``__call__`` (``tma_atom_out``/``tma_tensor_out``).
                sC_l2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.l2_out_dtype,
                        cute.cosize(self.c_smem_layout_staged_l2.outer),
                    ],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = (
            SharedStorage2cta if cutlass.const_expr(self.use_2cta_instrs) else SharedStorage1cta
        )

        # ------------------------------------------------------------------
        # Linear2 TMA descriptors + L2 cluster/SMEM layouts
        # threaded into the device kernel. On the Linear1-only path the
        # Linear2 attrs live only on ``self`` (populated inside the
        # ``if self.enable_linear2`` branch in ``_setup_attributes``); we
        # bind reused Linear1 values here so the kernel signature is
        # well-typed regardless of ``enable_linear2``. The device-side L2
        # partitions are gated on ``cutlass.const_expr(self.enable_linear2)``
        # so DCE folds the whole block away on the Linear1-only path.
        # ------------------------------------------------------------------
        if cutlass.const_expr(self.enable_linear2):
            kernel_tma_atoms_b_l2 = tma_atoms_b_l2
            kernel_tma_tensors_b_l2 = tma_tensors_b_l2
            kernel_tma_atoms_sfb_l2 = tma_atoms_sfb_l2
            kernel_tma_tensors_sfb_l2 = tma_tensors_sfb_l2
            kernel_tma_atom_pool_sfa_l2 = tma_atom_pool_sfa_l2
            kernel_tma_tensor_pool_sfa_l2 = tma_tensor_pool_sfa_l2
            kernel_tma_atom_out = tma_atom_out
            kernel_tma_tensor_out = tma_tensor_out
            kernel_cluster_layout_vmnk_l2 = self.cluster_layout_vmnk_l2
            kernel_cluster_layout_sfb_vmnk_l2 = self.cluster_layout_sfb_vmnk_l2
            kernel_b_smem_layout_staged_l2 = self.b_smem_layout_staged_l2
            kernel_sfb_smem_layout_staged_l2 = self.sfb_smem_layout_staged_l2
            kernel_c_smem_layout_staged_l2 = self.c_smem_layout_staged_l2
            # pool store TMA atom built above; drives the
            # FC1 epilogue S2G stream to the permuted HBM pool.
            kernel_tma_atom_pool_store = tma_atom_pool_store
            kernel_tma_tensor_pool_store = tma_tensor_pool_store
            # TiledMma / Layout attributes as kernel args instead of
            # dereferencing them via ``self`` inside the device region.
            # MLIR region isolation rejects ``self.<composed_layout>``
            # reads from the kernel body (the value is defined outside
            # the isolated region); threading them as explicit kernel
            # arguments mirrors the L1 pattern where ``tiled_mma`` and
            # ``a_smem_layout_staged`` are passed positionally.
            kernel_tiled_mma_l2 = self.tiled_mma_l2
            kernel_tiled_mma_sfb_l2 = self.tiled_mma_sfb_l2
            kernel_a_smem_layout_staged_l2 = self.a_smem_layout_staged_l2
            kernel_sfa_smem_layout_staged_l2 = self.sfa_smem_layout_staged_l2
            kernel_epi_layout_l2 = self.epi_layout_l2
        else:
            # Linear1-only path: reuse FC1 atoms so the signature stays
            # uniformly typed; the device-side const_expr gate ensures
            # these values are never read.
            kernel_tma_atoms_b_l2 = tma_atoms_b
            kernel_tma_tensors_b_l2 = tma_tensors_b
            kernel_tma_atoms_sfb_l2 = tma_atoms_sfb
            kernel_tma_tensors_sfb_l2 = tma_tensors_sfb
            kernel_tma_atom_pool_sfa_l2 = tma_atoms_sfb[0]
            kernel_tma_tensor_pool_sfa_l2 = sfa
            kernel_tma_atom_out = tma_atom_c
            kernel_tma_tensor_out = tma_tensor_c
            kernel_cluster_layout_vmnk_l2 = self.cluster_layout_vmnk
            kernel_cluster_layout_sfb_vmnk_l2 = self.cluster_layout_sfb_vmnk
            kernel_b_smem_layout_staged_l2 = self.b_smem_layout_staged
            kernel_sfb_smem_layout_staged_l2 = self.sfb_smem_layout_staged
            kernel_c_smem_layout_staged_l2 = self.c_smem_layout_staged
            # Linear1-only path reuses the FC1 ``tma_atom_c`` /
            # ``tma_tensor_c`` to keep the kernel signature uniformly
            # typed; the device-side epilogue picks FC1 vs pool via the
            # ``cutlass.const_expr(self.enable_linear2)`` gate, so these
            # reused values are never actually used for a pool store.
            kernel_tma_atom_pool_store = tma_atom_c
            kernel_tma_tensor_pool_store = tma_tensor_c
            # composed_layout / TiledMma / Layout kernel args. ``tiled_mma``
            # and ``tiled_mma_sfb`` are the FC1 locals already bound above
            # in this ``__call__``; ``self.a_smem_layout_staged`` /
            # ``self.sfa_smem_layout_staged`` are the FC1 staged SMEM
            # layouts. ``epi_layout_l2`` has no FC1 analog, so use a
            # trivial 1-element layout placeholder; Linear1-only path never
            # reaches the L2 combine body that reads this param.
            kernel_tiled_mma_l2 = tiled_mma
            kernel_tiled_mma_sfb_l2 = tiled_mma_sfb
            kernel_a_smem_layout_staged_l2 = self.a_smem_layout_staged
            kernel_sfa_smem_layout_staged_l2 = self.sfa_smem_layout_staged
            kernel_epi_layout_l2 = cute.make_layout(shape=(1,), stride=(1,))

        # ------------------------------------------------------------------
        # HBM pool tensors threaded into the device kernel.
        # ``pool_tensor``/``pool_sfc_tensor`` carry the FC1->FC2 activation /
        # SF hand-off buffer; when the caller has not allocated them yet
        # (fused-path transition) or when ``enable_linear2=False`` we reuse
        # ``a``/``sfa`` as typed placeholders. The
        # device-side FC2 LDGSTS body reads from these only inside the
        # ``cutlass.const_expr(self.enable_linear2)`` gate, so the FC1
        # Linear1-only path never dereferences the fallback.
        # ------------------------------------------------------------------
        kernel_pool_tensor_l1 = pool_tensor if pool_tensor is not None else a
        kernel_pool_sfc_tensor_l1 = pool_sfc_tensor if pool_sfc_tensor is not None else sfa
        kernel_a_bytes = a_bytes if a_bytes is not None else a
        kernel_l2_arrival_mask = (
            l2_arrival_mask if l2_arrival_mask is not None else num_non_exiting_tiles
        )
        kernel_out_tensor = scatter_out if scatter_out is not None else out
        kernel_monolithic_final_output = (
            monolithic_final_output if monolithic_final_output is not None else kernel_out_tensor
        )
        kernel_monolithic_control = (
            monolithic_control if monolithic_control is not None else kernel_l2_arrival_mask
        )
        kernel_grid = grid
        kernel_monolithic_grid_sync_blocks = monolithic_grid_sync_blocks

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            a,
            kernel_a_bytes,
            tma_atoms_b,  # Tuple of TMA atoms for B
            tma_tensors_b,  # Tuple of TMA tensors for B
            sfa,
            tma_atoms_sfb,  # Tuple of TMA atoms for SFB
            tma_tensors_sfb,  # Tuple of TMA tensors for SFB
            tma_atom_c,
            tma_tensor_c,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha_tuple,
            # Linear2 epilogue tensors threaded through to the
            # device kernel. On the Linear1-only path these are
            # ``None`` / a reused Linear1 ``alpha_tuple``; the L2 tile body is
            # DCE'd away so they are never read.
            alpha_l2_tuple,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            kernel_out_tensor,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            self.tile_sched_params_l2,
            epilogue_op,
            # Linear2 TMA atoms/tensors + cluster/SMEM
            # layouts. Unused on the Linear1-only path via const_expr.
            kernel_tma_atoms_b_l2,
            kernel_tma_tensors_b_l2,
            kernel_tma_atoms_sfb_l2,
            kernel_tma_tensors_sfb_l2,
            kernel_tma_atom_pool_sfa_l2,
            kernel_tma_tensor_pool_sfa_l2,
            kernel_tma_atom_out,
            kernel_tma_tensor_out,
            kernel_cluster_layout_vmnk_l2,
            kernel_cluster_layout_sfb_vmnk_l2,
            kernel_b_smem_layout_staged_l2,
            kernel_sfb_smem_layout_staged_l2,
            kernel_c_smem_layout_staged_l2,
            # HBM pool hand-off tensors for the FC2
            # LDGSTS-A phase. Linear1-only path reuses ``a``/``sfa``; gated
            # device-side via ``cutlass.const_expr(self.enable_linear2)``.
            kernel_pool_tensor_l1,
            kernel_pool_sfc_tensor_l1,
            # FC1 epilogue pool store TMA atom/tensor.
            # Used by the FC1 epilogue warp to stream requantized FP4 rows
            # to the HBM pool when ``enable_linear2=True``; Linear1-only path
            # reuses ``tma_atom_c`` / ``tma_tensor_c`` as typed
            # placeholders.
            kernel_tma_atom_pool_store,
            kernel_tma_tensor_pool_store,
            kernel_l2_arrival_mask,
            # Layout kernel args. These previously lived on ``self`` and
            # were dereferenced from the device region, violating MLIR
            # region isolation. Threading them as kernel arguments mirrors
            # the L1 ``tiled_mma`` / ``a_smem_layout_staged`` pattern and
            # makes the FC2 phase body MLIR-legal.
            kernel_tiled_mma_l2,
            kernel_tiled_mma_sfb_l2,
            kernel_a_smem_layout_staged_l2,
            kernel_sfa_smem_layout_staged_l2,
            kernel_epi_layout_l2,
            monolithic_direct_topk_input if monolithic_direct_topk_input is not None else a,
            monolithic_direct_topk_input_fp4 if monolithic_direct_topk_input_fp4 is not None else a,
            monolithic_direct_topk_input_scale
            if monolithic_direct_topk_input_scale is not None
            else sfa,
            monolithic_direct_topk_source_input,
            monolithic_direct_topk_input_rank_stride_fp4,
            monolithic_direct_topk_input_scale_rank_stride_elements,
            monolithic_direct_topk_idx
            if monolithic_direct_topk_idx is not None
            else monolithic_control,
            monolithic_direct_topk_scales
            if monolithic_direct_topk_scales is not None
            else token_final_scales,
            monolithic_direct_topk_token_counts
            if monolithic_direct_topk_token_counts is not None
            else num_non_exiting_tiles,
            monolithic_direct_topk_local_input
            if monolithic_direct_topk_local_input is not None
            else a,
            monolithic_direct_topk_local_input_scale
            if monolithic_direct_topk_local_input_scale is not None
            else sfa,
            monolithic_direct_topk_local_idx
            if monolithic_direct_topk_local_idx is not None
            else monolithic_control,
            monolithic_direct_topk_local_scales
            if monolithic_direct_topk_local_scales is not None
            else token_final_scales,
            monolithic_direct_topk_local_expert_offset,
            monolithic_direct_topk_num_local_experts,
            monolithic_direct_topk_stage_inputs,
            direct_combine_output,
            direct_combine_atomic_output,
            direct_combine_token_major_output,
            combine_output_ep_size,
            combine_output_top_k,
            combine_output_max_num_tokens_per_rank,
            monolithic_reduce_output,
            monolithic_direct_topk_materialize,
            kernel_monolithic_final_output,
            kernel_monolithic_control,
            monolithic_local_rank,
            monolithic_local_tokens,
            kernel_monolithic_grid_sync_blocks,
            combine_output_hidden_size,
        ).launch(
            grid=kernel_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to
        partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mA_mkl: cute.Tensor,
        mA_bytes: cute.Tensor,
        tma_atoms_b: Tuple[cute.CopyAtom, ...],
        mB_nkl_tuple: Tuple[cute.Tensor, ...],
        mSFA_mkl: cute.Tensor,
        tma_atoms_sfb: Tuple[cute.CopyAtom, ...],
        mSFB_nkl_tuple: Tuple[cute.Tensor, ...],
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha_tuple: Tuple[cute.Tensor, ...],
        # Linear2 epilogue references four names that previously
        # lived under ``# noqa: F821`` inside the const_expr-DCE'd ``else:``
        # branch (L2 tile body). Threading them through the kernel signature
        # brings them into lexical scope so the noqa markers can be dropped
        # while preserving DCE on the ``enable_linear2=False`` Linear1-only path.
        alpha_l2_tuple: Tuple[cute.Tensor, ...],
        permuted_idx_to_expanded_idx: Optional[cute.Tensor],
        token_final_scales: Optional[cute.Tensor],
        out_tensor: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        tile_sched_params_l2: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        # Linear2 (FC2) TMA atoms/tensors + cluster/SMEM
        # layouts. The Linear1-only path (enable_linear2=False) receives
        # reused Linear1 values here; device-side L2 partitions + phase
        # branches are all gated on ``cutlass.const_expr(self.enable_linear2)``
        # so these parameters are read only when the FC2 path is live.
        tma_atoms_b_l2: Tuple[cute.CopyAtom, ...],
        mB_l2_nkl_tuple: Tuple[cute.Tensor, ...],
        tma_atoms_sfb_l2: Tuple[cute.CopyAtom, ...],
        mSFB_l2_nkl_tuple: Tuple[cute.Tensor, ...],
        tma_atom_pool_sfa_l2: cute.CopyAtom,
        mPoolSFA_l2_mkl: cute.Tensor,
        tma_atom_out: cute.CopyAtom,
        mOut_mnl: cute.Tensor,
        cluster_layout_vmnk_l2: cute.Layout,
        cluster_layout_sfb_vmnk_l2: cute.Layout,
        b_smem_layout_staged_l2: cute.ComposedLayout,
        sfb_smem_layout_staged_l2: cute.Layout,
        c_smem_layout_staged_l2: Union[cute.Layout, cute.ComposedLayout, None],
        # FC1 -> FC2 activation hand-off pool tensors.
        # Reads are gated on ``cutlass.const_expr(self.enable_linear2)`` in
        # the LDGSTS-A warp body below so the Linear1-only path never touches
        # them (and ``__call__`` threads ``a``/``sfa`` as typed placeholders
        # when the caller has not wired pool buffers yet).
        pool_tensor_l1: cute.Tensor,
        pool_sfc_tensor_l1: cute.Tensor,
        # FC1 epilogue pool-store TMA descriptor. When
        # ``enable_linear2=True`` the FC1 epilogue warp routes the
        # SwiGLU+requant FP4 S2G stream to the HBM pool via this atom
        # instead of to the standalone FC1 ``c`` output tensor. Linear1-only path
        # reuses ``tma_atom_c`` / ``tma_tensor_c`` as typed placeholders;
        # device-side reads are gated on
        # ``cutlass.const_expr(self.enable_linear2)``.
        tma_atom_pool_store: cute.CopyAtom,
        mPool_store_mnl: cute.Tensor,
        l2_arrival_mask: cute.Tensor,
        # ``ComposedLayout`` / ``Layout`` + epilogue scatter ``Layout``
        # threaded in as kernel parameters. The previous layout of
        # dereferencing ``self.tiled_mma_l2`` / ``self.a_smem_layout_staged_l2``
        # / ``self.sfa_smem_layout_staged_l2`` / ``self.epi_layout_l2``
        # from the device kernel region tripped MLIR's ``region isolation``
        # verifier (''cute.composed_get_outer' using value defined outside
        # the region'); these kernel arguments bring the values into
        # lexical scope, mirroring the L1 ``tiled_mma``
        # / ``a_smem_layout_staged`` / ``sfa_smem_layout_staged`` params
        # above. Linear1-only path fallbacks are bound in ``__call__`` to reused
        # L1 values / trivial placeholders; device-side reads are gated
        # on ``cutlass.const_expr(self.enable_linear2)``.
        tiled_mma_l2: cute.TiledMma,
        tiled_mma_sfb_l2: cute.TiledMma,
        a_smem_layout_staged_l2: cute.ComposedLayout,
        sfa_smem_layout_staged_l2: cute.Layout,
        epi_layout_l2: cute.Layout,
        monolithic_direct_topk_input: cute.Tensor,
        monolithic_direct_topk_input_fp4: cute.Tensor,
        monolithic_direct_topk_input_scale: cute.Tensor,
        monolithic_direct_topk_source_input: cutlass.Constexpr,
        monolithic_direct_topk_input_rank_stride_fp4: cutlass.Constexpr,
        monolithic_direct_topk_input_scale_rank_stride_elements: cutlass.Constexpr,
        monolithic_direct_topk_idx: cute.Tensor,
        monolithic_direct_topk_scales: cute.Tensor,
        monolithic_direct_topk_token_counts: cute.Tensor,
        monolithic_direct_topk_local_input: cute.Tensor,
        monolithic_direct_topk_local_input_scale: cute.Tensor,
        monolithic_direct_topk_local_idx: cute.Tensor,
        monolithic_direct_topk_local_scales: cute.Tensor,
        monolithic_direct_topk_local_expert_offset: cutlass.Constexpr,
        monolithic_direct_topk_num_local_experts: cutlass.Constexpr,
        monolithic_direct_topk_stage_inputs: cutlass.Constexpr,
        direct_combine_output: cutlass.Constexpr,
        direct_combine_atomic_output: cutlass.Constexpr,
        direct_combine_token_major_output: cutlass.Constexpr,
        combine_output_ep_size: cutlass.Constexpr,
        combine_output_top_k: cutlass.Constexpr,
        combine_output_max_num_tokens_per_rank: cutlass.Constexpr,
        monolithic_reduce_output: cutlass.Constexpr,
        monolithic_direct_topk_materialize: cutlass.Constexpr,
        monolithic_final_output: cute.Tensor,
        monolithic_control: cute.Tensor,
        monolithic_local_rank: cutlass.Constexpr,
        monolithic_local_tokens: cutlass.Int64,
        monolithic_grid_sync_blocks: cutlass.Constexpr,
        monolithic_hidden_size: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            # Prefetch TMA descriptors for all B tensors using const_expr conditions
            cpasync.prefetch_descriptor(tma_atoms_b[0])
            cpasync.prefetch_descriptor(tma_atoms_sfb[0])
            if cutlass.const_expr(self.num_b_tensors >= 2):
                cpasync.prefetch_descriptor(tma_atoms_b[1])
                cpasync.prefetch_descriptor(tma_atoms_sfb[1])
            if cutlass.const_expr(self.num_b_tensors >= 3):
                cpasync.prefetch_descriptor(tma_atoms_b[2])
                cpasync.prefetch_descriptor(tma_atoms_sfb[2])
            if cutlass.const_expr(self.num_b_tensors >= 4):
                cpasync.prefetch_descriptor(tma_atoms_b[3])
                cpasync.prefetch_descriptor(tma_atoms_sfb[3])
            cpasync.prefetch_descriptor(tma_atom_c)
            # prefetch Linear2 (FC2) TMA descriptors so the
            # FC2 tile-loads inside the phase-branch below can stream without
            # a cold-start stall. DCE-elided on the Linear1-only path.
            if cutlass.const_expr(self.enable_linear2):
                cpasync.prefetch_descriptor(tma_atoms_b_l2[0])
                cpasync.prefetch_descriptor(tma_atoms_sfb_l2[0])
                if cutlass.const_expr(self.num_b_tensors >= 2):
                    cpasync.prefetch_descriptor(tma_atoms_b_l2[1])
                    cpasync.prefetch_descriptor(tma_atoms_sfb_l2[1])
                if cutlass.const_expr(self.num_b_tensors >= 3):
                    cpasync.prefetch_descriptor(tma_atoms_b_l2[2])
                    cpasync.prefetch_descriptor(tma_atoms_sfb_l2[2])
                if cutlass.const_expr(self.num_b_tensors >= 4):
                    cpasync.prefetch_descriptor(tma_atoms_b_l2[3])
                    cpasync.prefetch_descriptor(tma_atoms_sfb_l2[3])
                cpasync.prefetch_descriptor(tma_atom_pool_sfa_l2)
                cpasync.prefetch_descriptor(tma_atom_out)
                # prefetch FC1 epilogue pool-store TMA
                # descriptor so the first L1-phase epilogue S2G stream to
                # the HBM pool does not pay a cold-start stall. DCE'd on
                # the Linear1-only path via the outer enable_linear2 gate.
                cpasync.prefetch_descriptor(tma_atom_pool_store)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # matching coord projections onto the Linear2
        # cluster layouts. FC1 and FC2 share ``cluster_shape_mn`` so the
        # flat-coord is structurally identical today; keeping separate names
        # documents intent and leaves room for M3 wave scheduling where the
        # two layouts may diverge.
        block_in_cluster_coord_vmnk_l2 = cluster_layout_vmnk_l2.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk_l2 = cluster_layout_sfb_vmnk_l2.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline Init: Initialize A pipeline for LDGSTS operations
        # Producer: 4 warps (warps 4-7) with 128 threads total for LDGSTS operations
        # Consumer: MMA warp for consuming A/SFA data
        a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 4,
        )

        a_pipeline_l1 = PipelineCpAsyncUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        a_pipeline_l2 = a_pipeline_l1
        if cutlass.const_expr(self.enable_linear2):
            a_pipeline_l2 = PipelineCpAsyncUmma.create(
                barrier_storage=storage.a_pool_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage_l2,
                producer_group=a_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk_l2,
                defer_sync=True,
            )
        a_pipeline = a_pipeline_l1

        # Pipeline Init: Initialize A SYNC Transform pipelines when use_2cta_instrs is True.
        # L2 owns a separate transform barrier/state so FC2 can use its own AB stage count.
        if cutlass.const_expr(self.use_2cta_instrs):
            a_sync_transform_pipeline_producer_group_l1 = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * cute.size(cluster_layout_vmnk, mode=[0]),
            )
            a_sync_transform_pipeline_l1 = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.a_sync_transform_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=a_sync_transform_pipeline_producer_group_l1,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            a_sync_transform_pipeline_l2 = a_sync_transform_pipeline_l1
            if cutlass.const_expr(self.enable_linear2):
                a_sync_transform_pipeline_producer_group_l2 = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * cute.size(cluster_layout_vmnk_l2, mode=[0]),
                )
                a_sync_transform_pipeline_l2 = pipeline.PipelineAsyncUmma.create(
                    barrier_storage=storage.a_sync_transform_l2_mbar_ptr.data_ptr(),
                    num_stages=self.num_ab_stage_l2,
                    producer_group=a_sync_transform_pipeline_producer_group_l2,
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    cta_layout_vmnk=cluster_layout_vmnk_l2,
                    defer_sync=True,
                )
            a_sync_transform_pipeline = a_sync_transform_pipeline_l1

        # Pipeline Init: Initialize B pipelines for TMA operations.
        # L2 owns its barrier group and tx_count so FC2 B/SFB geometry no longer
        # relies on FC1's B tile bytes or AB stage state.
        b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        b_pipeline_consumer_group_l1 = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mcast_ctas_b
        )
        b_pipeline_l1 = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group_l1,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_pipeline_l2 = b_pipeline_l1
        if cutlass.const_expr(self.enable_linear2):
            b_pipeline_consumer_group_l2 = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b_l2
            )
            b_pipeline_l2 = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_l2_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage_l2,
                producer_group=b_pipeline_producer_group,
                consumer_group=b_pipeline_consumer_group_l2,
                tx_count=self.num_tma_load_bytes_l2,
                cta_layout_vmnk=cluster_layout_vmnk_l2,
            )
        b_pipeline = b_pipeline_l1

        sfa_pool_pipeline_l2 = None
        if cutlass.const_expr(self.enable_linear2):
            sfa_pool_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            sfa_pool_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_a_l2
            )
            sfa_pool_pipeline_l2 = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.sfa_pool_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage_l2,
                producer_group=sfa_pool_pipeline_producer_group,
                consumer_group=sfa_pool_pipeline_consumer_group,
                tx_count=self.num_tma_sfa_pool_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk_l2,
            )

        # Pipeline Init: Initialize accumulator pipelines and states.
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline_l1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline_l2 = acc_pipeline_l1
        if cutlass.const_expr(self.enable_linear2):
            acc_pipeline_l2 = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_l2_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage_l2,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk_l2,
            )
        acc_pipeline = acc_pipeline_l1

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
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

        # Tensor memory dealloc barrier init
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

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # ------------------------------------------------------------------
        # Bind Linear2 SMEM slots to staged layouts under the same
        # ``cutlass.const_expr(self.enable_linear2)`` gate used by storage.
        # Assigned to ``None`` on the Linear1-only path so the name is defined
        # regardless.
        # ------------------------------------------------------------------
        sB_l2 = None
        sSFB_l2 = None
        # FC1 -> FC2 hand-off staging. ``sA_pool`` and
        # ``sSFA_pool`` are the SMEM destinations the LDGSTS-A warp writes
        # into during the FC2 phase (source = HBM pool written by the FC1
        # epilogue in S2-d). Linear1-only path DCE keeps these as ``None``.
        sA_pool = None
        sSFA_pool = None
        if cutlass.const_expr(self.enable_linear2):
            sB_l2 = storage.sB_l2.get_tensor(
                b_smem_layout_staged_l2.outer, swizzle=b_smem_layout_staged_l2.inner
            )
            sSFB_l2 = storage.sSFB_l2.get_tensor(sfb_smem_layout_staged_l2)
            sA_pool = storage.sA_pool.get_tensor(
                a_smem_layout_staged_l2.outer,
                swizzle=a_smem_layout_staged_l2.inner,
            )
            sSFA_pool = storage.sSFA_pool.get_tensor(sfa_smem_layout_staged_l2)
        # (bidx, bidy, bidz, valid, mn_limit, phase)  - slot 5 stores the phase
        info_layout = cute.make_layout((6, self.num_tile_stage), stride=(1, 6))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        # Linear2 multicast masks. Derived from the L2
        # cluster layouts; falls back to FC1 atoms when Linear2 is off so
        # the symbols are always defined but DCE-elided.
        b_l2_full_mcast_mask = b_full_mcast_mask
        sfb_l2_full_mcast_mask = sfb_full_mcast_mask
        sfa_pool_l2_full_mcast_mask = None
        if cutlass.const_expr(self.enable_linear2):
            if cutlass.const_expr(self.is_b_mcast_l2 or use_2cta_instrs):
                b_l2_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk_l2,
                    block_in_cluster_coord_vmnk_l2,
                    mcast_mode=1,
                )
                sfb_l2_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk_l2,
                    block_in_cluster_coord_sfb_vmnk_l2,
                    mcast_mode=1,
                )
            if cutlass.const_expr(self.is_a_mcast_l2 or use_2cta_instrs):
                sfa_pool_l2_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk_l2,
                    block_in_cluster_coord_vmnk_l2,
                    mcast_mode=2,
                )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL) - Use const_expr conditions for tuple indexing
        gB_nkl_0 = cute.local_tile(
            mB_nkl_tuple[0], cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        if cutlass.const_expr(self.num_b_tensors >= 2):
            gB_nkl_1 = cute.local_tile(
                mB_nkl_tuple[1], cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
        if cutlass.const_expr(self.num_b_tensors >= 3):
            gB_nkl_2 = cute.local_tile(
                mB_nkl_tuple[2], cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
        if cutlass.const_expr(self.num_b_tensors >= 4):
            gB_nkl_3 = cute.local_tile(
                mB_nkl_tuple[3], cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None)), (None, None, None)
        )

        # (bN, bK, RestN, RestK, RestL) - Use const_expr conditions for tuple indexing
        gSFB_nkl_0 = cute.local_tile(
            mSFB_nkl_tuple[0],
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(self.num_b_tensors >= 2):
            gSFB_nkl_1 = cute.local_tile(
                mSFB_nkl_tuple[1],
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
        if cutlass.const_expr(self.num_b_tensors >= 3):
            gSFB_nkl_2 = cute.local_tile(
                mSFB_nkl_tuple[2],
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
        if cutlass.const_expr(self.num_b_tensors >= 4):
            gSFB_nkl_3 = cute.local_tile(
                mSFB_nkl_tuple[3],
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )

        gToken_ml = cute.local_tile(
            token_id_mapping_tensor, cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)), (None,)
        )

        # ------------------------------------------------------------------
        # FC1 -> FC2 HBM pool local_tile slices. The pool
        # is already in permuted layout (FC1 epilogue S2-d scatters the
        # requantized FP4 rows with no gather), so the FC2 LDGSTS-A body
        # mirrors the FC1 gather pattern but with ``token_offset = 0``.
        # Both pool views use the full Linear2 MMA-M tile. In 2CTA mode each
        # CTA copies only its 128-row half into local SMEM, but the global
        # source tile spans the 256-row MMA tile so peer CTA-v lanes can add
        # their row-block offset exactly like the standalone TMA path.
        # ------------------------------------------------------------------
        gPool_mkl = None
        gPoolSFC_mkl = None
        if cutlass.const_expr(self.enable_linear2):
            gPool_mkl = cute.local_tile(
                pool_tensor_l1,
                cute.slice_(self.mma_tiler_l2, (None, 0, None)),
                (None, None, None),
            )
            gPoolSFC_mkl = cute.local_tile(
                mPoolSFA_l2_mkl,
                cute.slice_(self.mma_tiler_l2, (None, 0, None)),
                (None, None, None),
            )

        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt_l1 = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL) - const_expr conditions
        tCgB_0 = thr_mma.partition_B(gB_nkl_0)
        if cutlass.const_expr(self.num_b_tensors >= 2):
            tCgB_1 = thr_mma.partition_B(gB_nkl_1)
        if cutlass.const_expr(self.num_b_tensors >= 3):
            tCgB_2 = thr_mma.partition_B(gB_nkl_2)
        if cutlass.const_expr(self.num_b_tensors >= 4):
            tCgB_3 = thr_mma.partition_B(gB_nkl_3)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL) - const_expr conditions
        tCgSFB_0 = thr_mma_sfb.partition_B(gSFB_nkl_0)
        if cutlass.const_expr(self.num_b_tensors >= 2):
            tCgSFB_1 = thr_mma_sfb.partition_B(gSFB_nkl_1)
        if cutlass.const_expr(self.num_b_tensors >= 3):
            tCgSFB_2 = thr_mma_sfb.partition_B(gSFB_nkl_2)
        if cutlass.const_expr(self.num_b_tensors >= 4):
            tCgSFB_3 = thr_mma_sfb.partition_B(gSFB_nkl_3)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        # ------------------------------------------------------------------
        # FC1 epilogue pool store partition. Mirrors
        # ``tCgC`` but uses ``mPool_store_mnl`` (the permuted HBM pool
        # activation tensor) as the global destination. Only meaningful
        # when ``enable_linear2=True``; on the Linear1-only path
        # ``mPool_store_mnl`` is the reused FC1 ``c`` tensor so the
        # partition is structurally identical and the device-side epi
        # body picks ``tCgC`` anyway (gated on
        # ``cutlass.const_expr(self.enable_linear2)``).
        # ------------------------------------------------------------------
        tCgPool_store = tCgC
        tCgOut = tCgC
        if cutlass.const_expr(self.enable_linear2):
            gPool_store_mnl = cute.local_tile(
                mPool_store_mnl,
                cute.slice_(self.mma_tiler_c, (None, None, 0)),
                (None, None, None),
            )
            tCgPool_store = thr_mma.partition_C(gPool_store_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        sB_grouped = cute.group_modes(sB, 0, 3)
        sSFB_grouped = cute.group_modes(sSFB, 0, 3)

        # TMA partition for B tensor 0
        tBsB_0, tBgB_0 = cpasync.tma_partition(
            tma_atoms_b[0],
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            sB_grouped,
            cute.group_modes(tCgB_0, 0, 3),
        )
        tBsSFB_0, tBgSFB_0 = cute.nvgpu.cpasync.tma_partition(
            tma_atoms_sfb[0],
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            sSFB_grouped,
            cute.group_modes(tCgSFB_0, 0, 3),
        )
        tBsSFB_0 = cute.filter_zeros(tBsSFB_0)
        tBgSFB_0 = cute.filter_zeros(tBgSFB_0)

        # TMA partition for B tensor 1 (tBsB shared memory partition is same for all, use _ to ignore)
        if cutlass.const_expr(self.num_b_tensors >= 2):
            _, tBgB_1 = cpasync.tma_partition(
                tma_atoms_b[1],
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                sB_grouped,
                cute.group_modes(tCgB_1, 0, 3),
            )
            _, tBgSFB_1 = cute.nvgpu.cpasync.tma_partition(
                tma_atoms_sfb[1],
                block_in_cluster_coord_sfb_vmnk[1],
                sfb_cta_layout,
                sSFB_grouped,
                cute.group_modes(tCgSFB_1, 0, 3),
            )
            tBgSFB_1 = cute.filter_zeros(tBgSFB_1)

        # TMA partition for B tensor 2
        if cutlass.const_expr(self.num_b_tensors >= 3):
            _, tBgB_2 = cpasync.tma_partition(
                tma_atoms_b[2],
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                sB_grouped,
                cute.group_modes(tCgB_2, 0, 3),
            )
            _, tBgSFB_2 = cute.nvgpu.cpasync.tma_partition(
                tma_atoms_sfb[2],
                block_in_cluster_coord_sfb_vmnk[1],
                sfb_cta_layout,
                sSFB_grouped,
                cute.group_modes(tCgSFB_2, 0, 3),
            )
            tBgSFB_2 = cute.filter_zeros(tBgSFB_2)

        # TMA partition for B tensor 3
        if cutlass.const_expr(self.num_b_tensors >= 4):
            _, tBgB_3 = cpasync.tma_partition(
                tma_atoms_b[3],
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                sB_grouped,
                cute.group_modes(tCgB_3, 0, 3),
            )
            _, tBgSFB_3 = cute.nvgpu.cpasync.tma_partition(
                tma_atoms_sfb[3],
                block_in_cluster_coord_sfb_vmnk[1],
                sfb_cta_layout,
                sSFB_grouped,
                cute.group_modes(tCgSFB_3, 0, 3),
            )
            tBgSFB_3 = cute.filter_zeros(tBgSFB_3)

        # ------------------------------------------------------------------
        # Linear2 (FC2) global/shared partitions. Mirrors
        # the FC1 pattern above but reads from ``mB_l2_nkl_tuple`` /
        # ``mSFB_l2_nkl_tuple`` through the FC2 ``tiled_mma_l2`` slice. All
        # work is gated on ``cutlass.const_expr(self.enable_linear2)`` so
        # the Linear1-only path compiles the whole block down to nothing.
        # The TMA-B warp body below picks between FC1 and FC2 tensors via
        # the per-tile phase tag emitted by the scheduler.
        # ------------------------------------------------------------------
        gB_l2_nkl_0 = None
        gB_l2_nkl_1 = None
        gB_l2_nkl_2 = None
        gB_l2_nkl_3 = None
        gSFB_l2_nkl_0 = None
        gSFB_l2_nkl_1 = None
        gSFB_l2_nkl_2 = None
        gSFB_l2_nkl_3 = None
        tBgB_l2_0 = None
        tBgB_l2_1 = None
        tBgB_l2_2 = None
        tBgB_l2_3 = None
        tBgSFB_l2_0 = None
        tBgSFB_l2_1 = None
        tBgSFB_l2_2 = None
        tBgSFB_l2_3 = None
        tBsB_l2 = None
        tBsSFB_l2 = None
        tAsSFA_pool_tma = None
        tAgSFA_pool_tma = None
        k_tile_cnt_l2 = cutlass.Int32(0)
        if cutlass.const_expr(self.enable_linear2):
            gB_l2_nkl_0 = cute.local_tile(
                mB_l2_nkl_tuple[0],
                cute.slice_(self.mma_tiler_l2, (0, None, None)),
                (None, None, None),
            )
            if cutlass.const_expr(self.num_b_tensors >= 2):
                gB_l2_nkl_1 = cute.local_tile(
                    mB_l2_nkl_tuple[1],
                    cute.slice_(self.mma_tiler_l2, (0, None, None)),
                    (None, None, None),
                )
            if cutlass.const_expr(self.num_b_tensors >= 3):
                gB_l2_nkl_2 = cute.local_tile(
                    mB_l2_nkl_tuple[2],
                    cute.slice_(self.mma_tiler_l2, (0, None, None)),
                    (None, None, None),
                )
            if cutlass.const_expr(self.num_b_tensors >= 4):
                gB_l2_nkl_3 = cute.local_tile(
                    mB_l2_nkl_tuple[3],
                    cute.slice_(self.mma_tiler_l2, (0, None, None)),
                    (None, None, None),
                )

            gSFB_l2_nkl_0 = cute.local_tile(
                mSFB_l2_nkl_tuple[0],
                cute.slice_(self.mma_tiler_sfb_l2, (0, None, None)),
                (None, None, None),
            )
            if cutlass.const_expr(self.num_b_tensors >= 2):
                gSFB_l2_nkl_1 = cute.local_tile(
                    mSFB_l2_nkl_tuple[1],
                    cute.slice_(self.mma_tiler_sfb_l2, (0, None, None)),
                    (None, None, None),
                )
            if cutlass.const_expr(self.num_b_tensors >= 3):
                gSFB_l2_nkl_2 = cute.local_tile(
                    mSFB_l2_nkl_tuple[2],
                    cute.slice_(self.mma_tiler_sfb_l2, (0, None, None)),
                    (None, None, None),
                )
            if cutlass.const_expr(self.num_b_tensors >= 4):
                gSFB_l2_nkl_3 = cute.local_tile(
                    mSFB_l2_nkl_tuple[3],
                    cute.slice_(self.mma_tiler_sfb_l2, (0, None, None)),
                    (None, None, None),
                )

            thr_mma_l2 = tiled_mma_l2.get_slice(mma_tile_coord_v)
            thr_mma_sfb_l2 = tiled_mma_sfb_l2.get_slice(mma_tile_coord_v)
            tCgB_l2_0 = thr_mma_l2.partition_B(gB_l2_nkl_0)
            if cutlass.const_expr(self.num_b_tensors >= 2):
                tCgB_l2_1 = thr_mma_l2.partition_B(gB_l2_nkl_1)
            if cutlass.const_expr(self.num_b_tensors >= 3):
                tCgB_l2_2 = thr_mma_l2.partition_B(gB_l2_nkl_2)
            if cutlass.const_expr(self.num_b_tensors >= 4):
                tCgB_l2_3 = thr_mma_l2.partition_B(gB_l2_nkl_3)
            tCgSFB_l2_0 = thr_mma_sfb_l2.partition_B(gSFB_l2_nkl_0)
            tCgPoolSFA_l2 = thr_mma_l2.partition_A(gPoolSFC_mkl)
            gOut_mnl = cute.local_tile(
                mOut_mnl,
                cute.slice_(self.mma_tiler_c_l2, (None, None, 0)),
                (None, None, None),
            )
            tCgOut = thr_mma_l2.partition_C(gOut_mnl)
            if cutlass.const_expr(self.num_b_tensors >= 2):
                tCgSFB_l2_1 = thr_mma_sfb_l2.partition_B(gSFB_l2_nkl_1)
            if cutlass.const_expr(self.num_b_tensors >= 3):
                tCgSFB_l2_2 = thr_mma_sfb_l2.partition_B(gSFB_l2_nkl_2)
            if cutlass.const_expr(self.num_b_tensors >= 4):
                tCgSFB_l2_3 = thr_mma_sfb_l2.partition_B(gSFB_l2_nkl_3)

            b_cta_layout_l2 = cute.make_layout(
                cute.slice_(cluster_layout_vmnk_l2, (0, None, 0, 0)).shape
            )
            sfb_cta_layout_l2 = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk_l2, (0, None, 0, 0)).shape
            )
            sB_l2_grouped = cute.group_modes(sB_l2, 0, 3)
            sSFB_l2_grouped = cute.group_modes(sSFB_l2, 0, 3)

            tBsB_l2, tBgB_l2_0 = cpasync.tma_partition(
                tma_atoms_b_l2[0],
                block_in_cluster_coord_vmnk_l2[1],
                b_cta_layout_l2,
                sB_l2_grouped,
                cute.group_modes(tCgB_l2_0, 0, 3),
            )
            tBsSFB_l2, tBgSFB_l2_0 = cute.nvgpu.cpasync.tma_partition(
                tma_atoms_sfb_l2[0],
                block_in_cluster_coord_sfb_vmnk_l2[1],
                sfb_cta_layout_l2,
                sSFB_l2_grouped,
                cute.group_modes(tCgSFB_l2_0, 0, 3),
            )
            tBsSFB_l2 = cute.filter_zeros(tBsSFB_l2)
            tBgSFB_l2_0 = cute.filter_zeros(tBgSFB_l2_0)

            sfa_pool_cta_layout_l2 = cute.make_layout(
                cute.slice_(cluster_layout_vmnk_l2, (0, 0, None, 0)).shape
            )
            tAsSFA_pool_tma, tAgSFA_pool_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_pool_sfa_l2,
                block_in_cluster_coord_vmnk_l2[2],
                sfa_pool_cta_layout_l2,
                cute.group_modes(sSFA_pool, 0, 3),
                cute.group_modes(tCgPoolSFA_l2, 0, 3),
            )
            tAsSFA_pool_tma = cute.filter_zeros(tAsSFA_pool_tma)
            tAgSFA_pool_tma = cute.filter_zeros(tAgSFA_pool_tma)

            if cutlass.const_expr(self.num_b_tensors >= 2):
                _, tBgB_l2_1 = cpasync.tma_partition(
                    tma_atoms_b_l2[1],
                    block_in_cluster_coord_vmnk_l2[1],
                    b_cta_layout_l2,
                    sB_l2_grouped,
                    cute.group_modes(tCgB_l2_1, 0, 3),
                )
                _, tBgSFB_l2_1 = cute.nvgpu.cpasync.tma_partition(
                    tma_atoms_sfb_l2[1],
                    block_in_cluster_coord_sfb_vmnk_l2[1],
                    sfb_cta_layout_l2,
                    sSFB_l2_grouped,
                    cute.group_modes(tCgSFB_l2_1, 0, 3),
                )
                tBgSFB_l2_1 = cute.filter_zeros(tBgSFB_l2_1)

            if cutlass.const_expr(self.num_b_tensors >= 3):
                _, tBgB_l2_2 = cpasync.tma_partition(
                    tma_atoms_b_l2[2],
                    block_in_cluster_coord_vmnk_l2[1],
                    b_cta_layout_l2,
                    sB_l2_grouped,
                    cute.group_modes(tCgB_l2_2, 0, 3),
                )
                _, tBgSFB_l2_2 = cute.nvgpu.cpasync.tma_partition(
                    tma_atoms_sfb_l2[2],
                    block_in_cluster_coord_sfb_vmnk_l2[1],
                    sfb_cta_layout_l2,
                    sSFB_l2_grouped,
                    cute.group_modes(tCgSFB_l2_2, 0, 3),
                )
                tBgSFB_l2_2 = cute.filter_zeros(tBgSFB_l2_2)

            if cutlass.const_expr(self.num_b_tensors >= 4):
                _, tBgB_l2_3 = cpasync.tma_partition(
                    tma_atoms_b_l2[3],
                    block_in_cluster_coord_vmnk_l2[1],
                    b_cta_layout_l2,
                    sB_l2_grouped,
                    cute.group_modes(tCgB_l2_3, 0, 3),
                )
                _, tBgSFB_l2_3 = cute.nvgpu.cpasync.tma_partition(
                    tma_atoms_sfb_l2[3],
                    block_in_cluster_coord_sfb_vmnk_l2[1],
                    sfb_cta_layout_l2,
                    sSFB_l2_grouped,
                    cute.group_modes(tCgSFB_l2_3, 0, 3),
                )
                tBgSFB_l2_3 = cute.filter_zeros(tBgSFB_l2_3)

            # Outer K-loop count for FC2 tiles (intermediate dim / K-tile).
            # Match the finalize kernel and derive it from the A/pool tile's
            # loopK axis. The B tile layout is (bN, bK, loopN, loopK, loopL),
            # so mode 2 is output-N tiling and is not a K-loop count.
            k_tile_cnt_l2 = cutlass.Int32(cute.size(gPool_mkl, mode=[3]))

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
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
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # ------------------------------------------------------------------
        # Linear2 MMA fragments + accumulator layout.
        # Built under ``cutlass.const_expr(self.enable_linear2)`` so the
        # Linear1-only path compiles these references away entirely.
        #
        # SMEM staging slot written by the LDGSTS-A warp in the FC2 phase),
        # NOT the FC1 ``sA`` slot. The FC2 path requires this swap, and
        # the swap never actually landed; FC2 MMA was reading stale FC1
        # activations via an ``sA`` layout whose swizzle (derived from
        # ``a_smem_layout_staged``) did not match the swizzle
        # ``a_smem_layout_staged_l2`` that the FC2 LDGSTS-A warp uses. The
        # swizzle mismatch causes UMMA to access misaligned SMEM cells
        # (async fault surfacing at ``torch.cuda.synchronize()`` as
        # ``cudaErrorMisalignedAddress``). ``sA_pool`` is sized with
        # ``a_smem_layout_staged_l2.outer`` (L1690), so the fragment layout
        # matches the producer.
        # ------------------------------------------------------------------
        tCrA_l2 = None
        tCrB_l2 = None
        tCtAcc_fake_l2 = None
        if cutlass.const_expr(self.enable_linear2):
            tCrA_l2 = tiled_mma_l2.make_fragment_A(sA_pool)
            tCrB_l2 = tiled_mma_l2.make_fragment_B(sB_l2)
            acc_shape_l2 = tiled_mma_l2.partition_shape_C(self.mma_tiler_l2[:2])
            if cutlass.const_expr(self.overlapping_accum_l2):
                num_acc_stage_overlapped_l2 = 2
                tCtAcc_fake_l2 = tiled_mma_l2.make_fragment_C(
                    cute.append(acc_shape_l2, num_acc_stage_overlapped_l2)
                )
                tCtAcc_fake_l2 = cute.make_tensor(
                    tCtAcc_fake_l2.iterator,
                    cute.make_layout(
                        tCtAcc_fake_l2.shape,
                        stride=(
                            tCtAcc_fake_l2.stride[0],
                            tCtAcc_fake_l2.stride[1],
                            tCtAcc_fake_l2.stride[2],
                            (256 - self.num_sf_tmem_cols_l2) * tCtAcc_fake_l2.stride[0][1],
                        ),
                    ),
                )
            else:
                tCtAcc_fake_l2 = tiled_mma_l2.make_fragment_C(
                    cute.append(acc_shape_l2, self.num_acc_stage_l2)
                )

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        griddepcontrol_wait()

        runtime_monolithic_grid_sync_blocks = cutlass.Int32(cute.size(cute.arch.grid_dim()))
        if cutlass.const_expr(monolithic_grid_sync_blocks > 0):
            runtime_monolithic_grid_sync_blocks = cutlass.Int32(monolithic_grid_sync_blocks)
        _, monolithic_grid_dim_y, monolithic_grid_dim_z = cute.arch.grid_dim()
        monolithic_linear_block_idx = (
            bidx * monolithic_grid_dim_y + bidy
        ) * monolithic_grid_dim_z + bidz
        stage_grid_stride = (
            cutlass.Int32(self.threads_per_cta) * runtime_monolithic_grid_sync_blocks
        )

        byte_idx = cutlass.Int32(0)
        sf_linear_idx = cutlass.Int32(0)
        route_linear_idx = cutlass.Int32(0)

        if cutlass.const_expr(monolithic_direct_topk_stage_inputs):
            # All CTAs stage this rank's direct-topk payload, then CTA0 publishes
            # the per-rank M5 ready words after a device-side grid barrier.
            byte_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
            max_input_bytes = combine_output_max_num_tokens_per_rank * (monolithic_hidden_size // 2)
            while byte_idx < max_input_bytes:
                token_idx = byte_idx // (monolithic_hidden_size // 2)
                hidden_byte_idx = byte_idx - token_idx * (monolithic_hidden_size // 2)
                monolithic_direct_topk_input[
                    monolithic_local_rank, token_idx, hidden_byte_idx, 0
                ] = monolithic_direct_topk_local_input[token_idx, hidden_byte_idx, 0]
                byte_idx = byte_idx + stage_grid_stride

            sf_linear_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
            local_sf_size = monolithic_hidden_size // self.sf_vec_size
            max_sf_items = combine_output_max_num_tokens_per_rank * local_sf_size
            while sf_linear_idx < max_sf_items:
                token_idx = sf_linear_idx // local_sf_size
                sf_idx = sf_linear_idx - token_idx * local_sf_size
                monolithic_direct_topk_input_scale[monolithic_local_rank, token_idx, sf_idx, 0] = (
                    monolithic_direct_topk_local_input_scale[token_idx, sf_idx, 0]
                )
                sf_linear_idx = sf_linear_idx + stage_grid_stride

            route_linear_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
            max_route_items = combine_output_max_num_tokens_per_rank * combine_output_top_k
            while route_linear_idx < max_route_items:
                token_idx = route_linear_idx // combine_output_top_k
                topk_idx = route_linear_idx - token_idx * combine_output_top_k
                monolithic_direct_topk_idx[monolithic_local_rank, token_idx, topk_idx] = (
                    monolithic_direct_topk_local_idx[token_idx, topk_idx]
                )
                monolithic_direct_topk_scales[monolithic_local_rank, token_idx, topk_idx] = (
                    monolithic_direct_topk_local_scales[token_idx, topk_idx]
                )
                route_linear_idx = route_linear_idx + stage_grid_stride

            # Phase 1: all CTAs finished local staging before CTA0 publishes M5 ready.
            fence_acq_rel_sys()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(1)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, cutlass.Uint64(1))
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

            if monolithic_linear_block_idx == 0:
                if tidx == 0:
                    magic_ptr = cute.domain_offset(
                        (monolithic_local_rank, 0), monolithic_control
                    ).iterator
                    epoch_ptr = cute.domain_offset(
                        (monolithic_local_rank, 1), monolithic_control
                    ).iterator
                    token_count_ptr = cute.domain_offset(
                        (monolithic_local_rank, 2), monolithic_control
                    ).iterator
                    producer_flag_ptr = cute.domain_offset(
                        (monolithic_local_rank, 3), monolithic_control
                    ).iterator
                    consumer_epoch_ptr = cute.domain_offset(
                        (monolithic_local_rank, 4), monolithic_control
                    ).iterator
                    consumer_flag_ptr = cute.domain_offset(
                        (monolithic_local_rank, 5), monolithic_control
                    ).iterator
                    previous_epoch = ld_acquire_sys_u64(epoch_ptr)
                    next_epoch = previous_epoch + cutlass.Uint64(1)
                    st_release_sys_u64(magic_ptr, cutlass.Uint64(0x4D35445245414459))
                    st_release_sys_u64(epoch_ptr, next_epoch)
                    st_release_sys_u64(
                        token_count_ptr, cutlass.Uint64(combine_output_max_num_tokens_per_rank)
                    )
                    st_release_sys_u64(producer_flag_ptr, cutlass.Uint64(1))
                    st_release_sys_u64(consumer_epoch_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(consumer_flag_ptr, cutlass.Uint64(0))

                    for ready_rank in range(combine_output_ep_size):
                        peer_magic_ptr = cute.domain_offset(
                            (ready_rank, 0), monolithic_control
                        ).iterator
                        peer_epoch_ptr = cute.domain_offset(
                            (ready_rank, 1), monolithic_control
                        ).iterator
                        peer_flag_ptr = cute.domain_offset(
                            (ready_rank, 3), monolithic_control
                        ).iterator
                        peer_magic = cutlass.Uint64(0)
                        peer_epoch = cutlass.Uint64(0)
                        peer_flag = cutlass.Uint64(0)
                        while (
                            peer_magic != cutlass.Uint64(0x4D35445245414459)
                            or peer_epoch != next_epoch
                            or peer_flag != cutlass.Uint64(1)
                        ):
                            peer_magic = ld_acquire_sys_u64(peer_magic_ptr)
                            peer_epoch = ld_acquire_sys_u64(peer_epoch_ptr)
                            peer_flag = ld_acquire_sys_u64(peer_flag_ptr)

                    st_release_sys_u64(consumer_epoch_ptr, next_epoch)
                    st_release_sys_u64(consumer_flag_ptr, cutlass.Uint64(1))

                self.cta_sync_barrier.arrive_and_wait()

            # Phase 2: all CTAs wait for CTA0 to finish the cross-rank M5 ready wait.
            fence_acq_rel_gpu()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(2)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, cutlass.Uint64(2))
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

        if cutlass.const_expr(monolithic_direct_topk_materialize):
            if cutlass.const_expr(not monolithic_direct_topk_source_input):
                flat_input_linear_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
                flat_input_items = (
                    combine_output_ep_size
                    * combine_output_max_num_tokens_per_rank
                    * (monolithic_hidden_size // 2)
                )
                while flat_input_linear_idx < flat_input_items:
                    token_row = flat_input_linear_idx // (monolithic_hidden_size // 2)
                    hidden_byte_idx = flat_input_linear_idx - token_row * (
                        monolithic_hidden_size // 2
                    )
                    source_rank = token_row // combine_output_max_num_tokens_per_rank
                    source_token_idx = (
                        token_row - source_rank * combine_output_max_num_tokens_per_rank
                    )
                    mA_bytes[token_row, hidden_byte_idx, 0] = monolithic_direct_topk_input[
                        source_rank, source_token_idx, hidden_byte_idx, 0
                    ]
                    flat_input_linear_idx = flat_input_linear_idx + stage_grid_stride

                flat_sf_linear_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
                flat_sf_items = (
                    combine_output_ep_size
                    * combine_output_max_num_tokens_per_rank
                    * (monolithic_hidden_size // self.sf_vec_size)
                )
                while flat_sf_linear_idx < flat_sf_items:
                    token_row = flat_sf_linear_idx // (monolithic_hidden_size // self.sf_vec_size)
                    sf_idx = flat_sf_linear_idx - token_row * (
                        monolithic_hidden_size // self.sf_vec_size
                    )
                    source_rank = token_row // combine_output_max_num_tokens_per_rank
                    source_token_idx = (
                        token_row - source_rank * combine_output_max_num_tokens_per_rank
                    )
                    mSFA_mkl[token_row, sf_idx, 0] = monolithic_direct_topk_input_scale[
                        source_rank, source_token_idx, sf_idx, 0
                    ]
                    flat_sf_linear_idx = flat_sf_linear_idx + stage_grid_stride

            # Grid-parallel monolithic mode absorbs the old M5 direct-topk
            # materialization boundary here. CTAs first count local routes,
            # block0 builds compact expert tile metadata, and CTAs then fill
            # route mappings in parallel.
            route_count_base_ptr = cute.domain_offset(
                (monolithic_local_rank, 10), monolithic_control
            ).iterator
            route_base_base_ptr = cute.domain_offset(
                (monolithic_local_rank, 10 + monolithic_direct_topk_num_local_experts),
                monolithic_control,
            ).iterator
            route_cursor_base_ptr = cute.domain_offset(
                (monolithic_local_rank, 10 + 2 * monolithic_direct_topk_num_local_experts),
                monolithic_control,
            ).iterator
            direct_topk_tile_size = self.mma_tiler_l2[0]

            if monolithic_linear_block_idx == 0:
                if tidx == 0:
                    num_non_exiting_tiles[0] = cutlass.Int32(0)
                    for local_expert_idx in cutlass.range_constexpr(
                        monolithic_direct_topk_num_local_experts
                    ):
                        st_release_sys_u64(
                            route_count_base_ptr + local_expert_idx, cutlass.Uint64(0)
                        )
                        st_release_sys_u64(
                            route_base_base_ptr + local_expert_idx, cutlass.Uint64(0)
                        )
                        st_release_sys_u64(
                            route_cursor_base_ptr + local_expert_idx, cutlass.Uint64(0)
                        )

            # Phase 3: counters and cursors are zeroed before any CTA counts routes.
            fence_acq_rel_gpu()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(3)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, cutlass.Uint64(3))
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

            flat_route_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
            total_route_items = (
                combine_output_ep_size
                * combine_output_max_num_tokens_per_rank
                * combine_output_top_k
            )
            while flat_route_idx < total_route_items:
                route_token_linear_idx = flat_route_idx // combine_output_top_k
                source_topk_idx = flat_route_idx - route_token_linear_idx * combine_output_top_k
                source_rank = route_token_linear_idx // combine_output_max_num_tokens_per_rank
                source_token_idx = (
                    route_token_linear_idx - source_rank * combine_output_max_num_tokens_per_rank
                )
                source_token_count = monolithic_direct_topk_token_counts[source_rank]
                if source_token_idx < source_token_count:
                    selected_expert = monolithic_direct_topk_idx[
                        source_rank, source_token_idx, source_topk_idx
                    ]
                    local_expert_dynamic = selected_expert - cutlass.Int64(
                        monolithic_direct_topk_local_expert_offset
                    )
                    if (local_expert_dynamic >= cutlass.Int64(0)) and (
                        local_expert_dynamic
                        < cutlass.Int64(monolithic_direct_topk_num_local_experts)
                    ):
                        local_expert_i32 = cutlass.Int32(local_expert_dynamic)
                        atomic_add_release_sys_u32(
                            route_count_base_ptr + local_expert_i32, cutlass.Uint32(1)
                        )
                flat_route_idx = flat_route_idx + stage_grid_stride

            # Phase 4: route counts are complete before block0 writes prefix metadata.
            fence_acq_rel_gpu()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(4)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, cutlass.Uint64(4))
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

            if monolithic_linear_block_idx == 0:
                if tidx == 0:
                    pool_start = cutlass.Int32(0)
                    for local_expert_idx in cutlass.range_constexpr(
                        monolithic_direct_topk_num_local_experts
                    ):
                        num_routes = cutlass.Int32(
                            ld_acquire_sys_u64(route_count_base_ptr + local_expert_idx)
                        )
                        st_release_sys_u64(
                            route_base_base_ptr + local_expert_idx,
                            cutlass.Uint64(pool_start),
                        )
                        st_release_sys_u64(
                            route_cursor_base_ptr + local_expert_idx, cutlass.Uint64(0)
                        )
                        padded_routes = (
                            (num_routes + direct_topk_tile_size - 1) // direct_topk_tile_size
                        ) * direct_topk_tile_size
                        tile_start = pool_start // direct_topk_tile_size
                        tile_count = padded_routes // direct_topk_tile_size
                        if tile_count > 0:
                            num_non_exiting_tiles[0] = tile_start + tile_count
                        routes_remaining = num_routes
                        tile_offset = cutlass.Int32(0)
                        while tile_offset < tile_count:
                            tile_idx = tile_start + tile_offset
                            tile_routes = routes_remaining
                            if tile_routes > direct_topk_tile_size:
                                tile_routes = cutlass.Int32(direct_topk_tile_size)
                            tile_idx_to_expert_idx[tile_idx] = cutlass.Int32(local_expert_idx)
                            tile_idx_to_mn_limit[tile_idx] = (
                                pool_start + tile_offset * direct_topk_tile_size + tile_routes
                            )
                            routes_remaining = routes_remaining - tile_routes
                            tile_offset = tile_offset + cutlass.Int32(1)
                        pool_start = pool_start + padded_routes

            # Phase 5: prefix metadata is visible before CTAs fill route mappings.
            fence_acq_rel_gpu()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(5)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, cutlass.Uint64(5))
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

            flat_route_fill_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
            while flat_route_fill_idx < total_route_items:
                route_token_linear_idx = flat_route_fill_idx // combine_output_top_k
                source_topk_idx = (
                    flat_route_fill_idx - route_token_linear_idx * combine_output_top_k
                )
                source_rank = route_token_linear_idx // combine_output_max_num_tokens_per_rank
                source_token_idx = (
                    route_token_linear_idx - source_rank * combine_output_max_num_tokens_per_rank
                )
                source_token_count = monolithic_direct_topk_token_counts[source_rank]
                if source_token_idx < source_token_count:
                    selected_expert = monolithic_direct_topk_idx[
                        source_rank, source_token_idx, source_topk_idx
                    ]
                    local_expert_dynamic = selected_expert - cutlass.Int64(
                        monolithic_direct_topk_local_expert_offset
                    )
                    if (local_expert_dynamic >= cutlass.Int64(0)) and (
                        local_expert_dynamic
                        < cutlass.Int64(monolithic_direct_topk_num_local_experts)
                    ):
                        local_expert_i32 = cutlass.Int32(local_expert_dynamic)
                        route_ordinal = cutlass.Int32(
                            atomic_add_release_sys_u32(
                                route_cursor_base_ptr + local_expert_i32,
                                cutlass.Uint32(1),
                            )
                        )
                        pool_start = cutlass.Int32(
                            ld_acquire_sys_u64(route_base_base_ptr + local_expert_i32)
                        )
                        pool_slot = pool_start + route_ordinal
                        combine_row = (
                            source_rank * combine_output_top_k + source_topk_idx
                        ) * combine_output_max_num_tokens_per_rank + source_token_idx
                        token_row = (
                            source_rank * combine_output_max_num_tokens_per_rank + source_token_idx
                        )
                        if cutlass.const_expr(monolithic_direct_topk_source_input):
                            token_id_mapping_tensor[pool_slot] = cutlass.Int32(source_rank) * (
                                cutlass.Int32(monolithic_direct_topk_input_rank_stride_fp4)
                            ) + cutlass.Int32(source_token_idx) * cutlass.Int32(
                                monolithic_hidden_size
                            )
                        else:
                            token_id_mapping_tensor[pool_slot] = token_row
                        permuted_idx_to_expanded_idx[pool_slot] = combine_row
                        token_final_scales[(combine_row, 0)] = monolithic_direct_topk_scales[
                            source_rank, source_token_idx, source_topk_idx
                        ]
                flat_route_fill_idx = flat_route_fill_idx + stage_grid_stride

            # Phase 6: all route mappings are materialized before persistent FC1/FC2 scheduling.
            fence_acq_rel_gpu()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(6)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, cutlass.Uint64(6))
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            # M-PA-6 wave-interleaved scheduler. Each expert wave emits all
            # Linear1 tiles first, then Linear2 tiles for the same wave. This
            # mirrors DeepGEMM's phase order while keeping the existing
            # StaticPersistentTileScheduler and tile-info pipeline.
            for wave_start in range(0, self.total_num_experts, self.num_experts_per_wave):
                wave_start_i32 = cutlass.Int32(wave_start)
                wave_end_i32 = cutlass.Int32(wave_start + self.num_experts_per_wave)

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                if cutlass.const_expr(self.raster_along_m):
                    while work_tile.is_valid_tile:
                        cur_tile_coord = work_tile.tile_idx
                        mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                        if mma_tile_coord_m < num_non_exiting_tiles_value:
                            expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                            expert_in_wave = (expert_idx >= wave_start_i32) and (
                                expert_idx < wave_end_i32
                            )
                            if expert_in_wave:
                                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                                cur_tile_coord = work_tile.tile_idx
                                mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                                with cute.arch.elect_one():
                                    sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                                    sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                                    sInfo[(2, tile_info_producer_state.index)] = expert_idx
                                    sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                        work_tile.is_valid_tile
                                    )
                                    sInfo[(4, tile_info_producer_state.index)] = mn_limit
                                    sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(
                                        PHASE_LINEAR1
                                    )
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
                        if mma_tile_coord_m < num_non_exiting_tiles_value:
                            expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                            expert_in_wave = (expert_idx >= wave_start_i32) and (
                                expert_idx < wave_end_i32
                            )
                            if expert_in_wave:
                                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                                cur_tile_coord = work_tile.tile_idx
                                mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                                with cute.arch.elect_one():
                                    sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                                    sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                                    sInfo[(2, tile_info_producer_state.index)] = expert_idx
                                    sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                        work_tile.is_valid_tile
                                    )
                                    sInfo[(4, tile_info_producer_state.index)] = mn_limit
                                    sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(
                                        PHASE_LINEAR1
                                    )
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

                if cutlass.const_expr(self.enable_linear2):
                    tile_sched_l2 = utils.StaticPersistentTileScheduler.create(
                        tile_sched_params_l2, cute.arch.block_idx(), cute.arch.grid_dim()
                    )
                    work_tile_l2 = tile_sched_l2.initial_work_tile_info()

                    if cutlass.const_expr(self.raster_along_m):
                        while work_tile_l2.is_valid_tile:
                            cur_tile_coord = work_tile_l2.tile_idx
                            mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                                tiled_mma_l2.thr_id.shape
                            )
                            if mma_tile_coord_m < num_non_exiting_tiles_value:
                                expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                                expert_in_wave = (expert_idx >= wave_start_i32) and (
                                    expert_idx < wave_end_i32
                                )
                                if expert_in_wave:
                                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                                    cur_tile_coord = work_tile_l2.tile_idx
                                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                                    with cute.arch.elect_one():
                                        sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                            0
                                        ]
                                        sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                            1
                                        ]
                                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                            work_tile_l2.is_valid_tile
                                        )
                                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                                        sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(
                                            PHASE_LINEAR2
                                        )
                                    cute.arch.fence_proxy(
                                        cute.arch.ProxyKind.async_shared,
                                        space=cute.arch.SharedSpace.shared_cta,
                                    )
                                    self.sched_sync_barrier.arrive_and_wait()
                                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                                    tile_info_producer_state.advance()

                            tile_sched_l2.advance_to_next_work()
                            work_tile_l2 = tile_sched_l2.get_current_work()
                    else:
                        is_continue_l2 = cutlass.Boolean(1)
                        while work_tile_l2.is_valid_tile and is_continue_l2:
                            cur_tile_coord = work_tile_l2.tile_idx
                            mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                                tiled_mma_l2.thr_id.shape
                            )
                            if mma_tile_coord_m < num_non_exiting_tiles_value:
                                expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                                expert_in_wave = (expert_idx >= wave_start_i32) and (
                                    expert_idx < wave_end_i32
                                )
                                if expert_in_wave:
                                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                                    cur_tile_coord = work_tile_l2.tile_idx
                                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                                    with cute.arch.elect_one():
                                        sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                            0
                                        ]
                                        sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                            1
                                        ]
                                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                            work_tile_l2.is_valid_tile
                                        )
                                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                                        sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(
                                            PHASE_LINEAR2
                                        )
                                    cute.arch.fence_proxy(
                                        cute.arch.ProxyKind.async_shared,
                                        space=cute.arch.SharedSpace.shared_cta,
                                    )
                                    self.sched_sync_barrier.arrive_and_wait()
                                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                                    tile_info_producer_state.advance()
                            else:
                                is_continue_l2 = cutlass.Boolean(0)

                            tile_sched_l2.advance_to_next_work()
                            work_tile_l2 = tile_sched_l2.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
                sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(PHASE_LINEAR1)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized LDGSTS A/SFA warps (warps 4-7)
        # These warps use LDGSTS instructions to load A and SFA from global to shared memory
        # with gather/permutation capability enabled by token_id_mapping
        #
        if warp_idx <= self.ldgsts_a_warp_id[-1] and warp_idx >= self.ldgsts_a_warp_id[0]:
            #
            # Setup LDGSTS copy atoms for A and SFA
            # A: 8x LDGSTS.128 per thread with swizzle_128B for A matrix (32 elements per thread)
            # SFA: 4x LDGSTS.32 per thread with 512-element block swizzling for scale factor A (4 elements per thread)
            #
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mA_mkl.element_type,
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
                mSFA_mkl.element_type,
                num_bits_per_copy=32,
            )
            tidx_in_warpgroup = tidx % 128

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
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            a_producer_state_l2 = a_producer_state
            if cutlass.const_expr(self.enable_linear2):
                a_producer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage_l2
                )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            # Slot 5 carries the per-tile phase for the fused path. Keep a
            # Linear1 sentinel when Linear2 is disabled so the FC2 branch is
            # folded away by const_expr.
            tile_phase = cutlass.Int32(PHASE_LINEAR1)
            if cutlass.const_expr(self.enable_linear2):
                tile_phase = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            did_l2_phase_sync = cutlass.Boolean(False)
            while is_valid_tile:
                # per-tile FC1 vs FC2 LDGSTS dispatch. On
                # the Linear1-only path (enable_linear2=False) ``do_phase_l2`` is a
                # constant False so the FC2 branch collapses via const_expr
                # DCE and the runtime ``if do_phase_l2:`` folds to the FC1
                # else-branch.
                do_phase_l2 = cutlass.Boolean(False)
                if cutlass.const_expr(self.enable_linear2):
                    do_phase_l2 = tile_phase == cutlass.Int32(PHASE_LINEAR2)

                if do_phase_l2:
                    if not did_l2_phase_sync:
                        did_l2_phase_sync = cutlass.Boolean(True)
                        self.phase_sync_barrier.arrive_and_wait()
                    # ------------------------------------------------------
                    # FC2 LDGSTS mainloop. Reads from the HBM pool written
                    # by the FC1 epilogue in permuted layout (S2-d), so no
                    # ``token_id_mapping`` gather is needed. K-loop iterates
                    # ``k_tile_cnt_l2`` and the destination SMEM slots are
                    # ``sA_pool`` / ``sSFA_pool``. The inner const_expr
                    # guard is the const_expr gate for the FC1 compile path.
                    # ------------------------------------------------------
                    if cutlass.const_expr(self.enable_linear2):
                        sA_pool_tiled = cute.make_tensor(
                            sA_pool.iterator,
                            layout=cute.make_layout(
                                (
                                    self.cta_tile_shape_mnk_l2[0],
                                    self.cta_tile_shape_mnk_l2[2],
                                    self.num_ab_stage_l2,
                                ),
                                stride=(
                                    self.cta_tile_shape_mnk_l2[2],
                                    1,
                                    self.cta_tile_shape_mnk_l2[0] * self.cta_tile_shape_mnk_l2[2],
                                ),
                            ),
                        )
                        tAsA_pool_tiled = a_thr_copy.partition_D(sA_pool_tiled)

                        # cover all M_cta_l2 rows of the pool tile at
                        # ``tile_info[0]``. Mirror FC1's per-i row-stride
                        # advance (FC1 uses ``a_token_offset_tensor[i]``
                        # to gather scattered token rows; here the pool is
                        # already in permuted layout so the row index is
                        # the identity ``(tidx//8) + i*16`` within the
                        # tile, plus a tile-level base offset selecting
                        # the right M-tile of ``gPool_mkl``). The previous
                        # body hardcoded ``loopM=0`` and reused the same
                        # GMEM iterator across all 8 i-iterations,
                        # covering only one pool row per thread and no
                        # tile-level M selection -- wrong data on every
                        # non-zero M-tile, and interacts with
                        # LDGSTS.128's address alignment requirements via
                        # the SMEM partition destination (different i
                        # iteration targets different SMEM rows, so the
                        # mismatched GMEM broadcast produced sporadic
                        # ``cudaErrorMisalignedAddress`` at the HW level).
                        l2_pool_block_idx = tile_info[0] // cute.size(tiled_mma_l2.thr_id.shape)
                        l2_pool_cta_m_offset = mma_tile_coord_v * self.cta_tile_shape_mnk_l2[0]
                        tAgA_l2 = gPool_mkl[(None, None, l2_pool_block_idx, None, 0)]
                        A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 32, divby=32)
                        # Pool SFA is loaded by the dedicated FC2 A-side TMA pipeline.
                        # This LDGSTS warp only streams the FP4 pool activation tile.

                        a_producer_state_l2.reset_count()
                        peek_a_empty_status_l2 = cutlass.Boolean(1)
                        if a_producer_state_l2.count < k_tile_cnt_l2:
                            peek_a_empty_status_l2 = a_pipeline_l2.producer_try_acquire(
                                a_producer_state_l2
                            )

                        cached_l2_arrival_mask = cutlass.Uint64(0)
                        l2_arrival_mask_ptr = l2_arrival_mask.iterator + l2_pool_block_idx
                        for k_tile in cutlass.range(0, k_tile_cnt_l2, 1, unroll=1):
                            a_pipeline_l2.producer_acquire(
                                a_producer_state_l2, peek_a_empty_status_l2
                            )

                            needed_l2_arrival_mask = cutlass.Uint64(
                                self.l2_arrival_mask_per_k_tile
                            ) << cutlass.Uint64(
                                a_producer_state_l2.count * self.l2_arrival_bits_per_k_tile
                            )
                            while (
                                cached_l2_arrival_mask & needed_l2_arrival_mask
                            ) != needed_l2_arrival_mask:
                                cached_l2_arrival_mask = ld_acquire_gpu_u64(l2_arrival_mask_ptr)

                            tAgA_l2_ktile = tAgA_l2[(None, None, a_producer_state_l2.count)]
                            tAsA_pool_ktile = tAsA_pool_tiled[
                                (None, None, None, a_producer_state_l2.index)
                            ]

                            for i in range(8):
                                # Per-i row index within the M-tile; each
                                # thread covers 8 rows across the 8
                                # iterations (``128 threads // 8 lanes-
                                # per-row-group * 8 i = 128 rows``), so
                                # the warpgroup fully covers M_cta_l2 =
                                # 128 rows per K-tile.
                                row_idx_i = l2_pool_cta_m_offset + (tidx_in_warpgroup // 8) + i * 16
                                A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                                    row_idx_i * tAgA_l2_ktile.layout[0].stride, divby=32
                                )
                                A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                                tAgA_slice_ptr = tAgA_l2_ktile.iterator + A_gmem_slice_offset
                                tAgA_slice = cute.make_tensor(
                                    tAgA_slice_ptr, layout=cute.make_layout((32,))
                                )
                                tAsA_slice = cute.make_tensor(
                                    tAsA_pool_ktile[(None, i, None)].iterator,
                                    layout=cute.make_layout((32,)),
                                )
                                cute.copy_atom_call(a_atom_copy, tAgA_slice, tAsA_slice)

                            # Pool SFA is produced by the TMA-B warp through
                            # ``sfa_pool_pipeline``. Keep the A pipeline commit
                            # scoped to the FP4 pool activation cp.async writes.
                            cute.arch.fence_proxy(
                                cute.arch.ProxyKind.async_shared,
                                space=cute.arch.SharedSpace.shared_cta,
                            )
                            a_pipeline_l2.producer_commit(a_producer_state_l2)

                            a_producer_state_l2.advance()
                            peek_a_empty_status_l2 = cutlass.Boolean(1)
                            if a_producer_state_l2.count < k_tile_cnt_l2:
                                peek_a_empty_status_l2 = a_pipeline_l2.producer_try_acquire(
                                    a_producer_state_l2
                                )
                else:
                    # FC1 LDGSTS gather body.
                    # Load token IDs for gather operation
                    # For A matrix: each thread loads 8 token offsets (for 8 LDGSTS.128 operations)
                    # For SFA matrix: each thread loads 1 token offset (for 4 LDGSTS.32 operations)
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
                    direct_sfa_source_offset = relative_sfa_token_offset
                    if cutlass.const_expr(monolithic_direct_topk_source_input):
                        direct_sfa_source_rank = relative_sfa_token_offset // cutlass.Int32(
                            monolithic_direct_topk_input_rank_stride_fp4
                        )
                        direct_sfa_source_token_fp4_offset = relative_sfa_token_offset - (
                            direct_sfa_source_rank
                            * cutlass.Int32(monolithic_direct_topk_input_rank_stride_fp4)
                        )
                        direct_sfa_source_token = (
                            direct_sfa_source_token_fp4_offset
                            // cutlass.Int32(monolithic_hidden_size)
                        )
                        direct_sfa_source_offset = direct_sfa_source_rank * cutlass.Int32(
                            monolithic_direct_topk_input_scale_rank_stride_elements
                        ) + direct_sfa_source_token * cutlass.Int32(
                            monolithic_hidden_size // self.sf_vec_size
                        )

                    tAgA = gA_mkl[(None, None, 0, None, 0)]
                    A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 32, divby=32)
                    tAgSFA = gSFA_mkl[(relative_sfa_token_offset, None, 0, None, 0)]
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

                    # Peek (try_wait) SCALE buffer empty
                    a_producer_state.reset_count()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt_l1:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                    #
                    # Load A and SFA with LDGSTS and gather/permutation
                    # Each K-tile iteration loads one K-tile of A and SFA from GMEM to SMEM
                    # using LDGSTS instructions with token-based gather addressing
                    #
                    for k_tile in cutlass.range(0, k_tile_cnt_l1, 1, unroll=1):
                        # Conditionally wait for AB buffer empty
                        a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

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
                            #
                            # Load A matrix: 8x LDGSTS.128 per thread with swizzle_128B
                            # Each LDGSTS.128 loads 32 elements (128 bits) from GMEM to SMEM
                            # Global memory address is computed using token offset for gather operation
                            # Predicate mask guards against invalid token IDs (padding tokens marked as -1)
                            #
                            A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                                a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride, divby=32
                            )
                            A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                            tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                            if cutlass.const_expr(monolithic_direct_topk_source_input):
                                direct_a_k_offset = (
                                    a_producer_state.count * self.cta_tile_shape_mnk[2]
                                )
                                direct_a_slice_offset = cute.assume(
                                    a_token_offset_tensor[i]
                                    + cutlass.Int32(direct_a_k_offset)
                                    + A_gmem_thread_offset,
                                    divby=32,
                                )
                                tAgA_slice_ptr = (
                                    monolithic_direct_topk_input_fp4.iterator
                                    + direct_a_slice_offset
                                )
                            tAgA_slice = cute.make_tensor(
                                tAgA_slice_ptr, layout=cute.make_layout((32,))
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
                                a_atom_copy, tAgA_slice, tAsA_slice, pred=a_predicate_slice
                            )

                        for i in range(4):
                            #
                            # Load SFA: 4x LDGSTS.32 per thread with 512-element block swizzling
                            # Each LDGSTS.32 loads 4 scale factor elements (32 bits) from GMEM to SMEM
                            # Uses same token offset as A matrix for consistent gather operation
                            #
                            swizzled_iterator = (tidx_in_warpgroup % 32) // 8 ^ i
                            tAgSFA_slice_ptr = tAgSFA_ktile.iterator + 4 * swizzled_iterator
                            if cutlass.const_expr(monolithic_direct_topk_source_input):
                                direct_sfa_k_offset = (
                                    a_producer_state.count * self.cta_tile_shape_mnk_sfa[2]
                                )
                                tAgSFA_slice_ptr = (
                                    monolithic_direct_topk_input_scale.iterator
                                    + direct_sfa_source_offset
                                    + cutlass.Int32(direct_sfa_k_offset)
                                    + 4 * swizzled_iterator
                                )
                            tAgSFA_slice = cute.make_tensor(
                                tAgSFA_slice_ptr, layout=cute.make_layout((4,))
                            )

                            tAsSFA_slice_ptr = tAsSFA_ktile.iterator + 512 * swizzled_iterator
                            tAsSFA_slice = cute.make_tensor(
                                tAsSFA_slice_ptr, cute.make_layout((4,))
                            )

                            cute.copy_atom_call(
                                sfa_atom_copy,
                                tAgSFA_slice,
                                tAsSFA_slice,
                                pred=sfa_predicate_tensor,
                            )

                        a_pipeline.producer_commit(a_producer_state)

                        # Peek (try_wait) A buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if a_producer_state.count < k_tile_cnt_l1:
                            peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                # refresh ``tile_phase`` for the next tile
                # the LDGSTS-A warp is about to handle. Linear1-only path DCE.
                if cutlass.const_expr(self.enable_linear2):
                    tile_phase = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait A pipeline buffer empty
            #
            if cutlass.const_expr(self.enable_linear2):
                a_pipeline_l2.producer_tail(a_producer_state_l2)
            a_pipeline.producer_tail(a_producer_state)

        #
        # Specialized A/SFA Sync Transform Warp (warp 11) when use_2cta_instrs is True
        # This warp serve as sync transformation for A and SFA
        #
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.use_2cta_instrs):
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                a_consumer_state_l2 = a_consumer_state
                a_sync_transform_producer_state_l2 = a_sync_transform_producer_state
                if cutlass.const_expr(self.enable_linear2):
                    a_consumer_state_l2 = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_ab_stage_l2
                    )
                    a_sync_transform_producer_state_l2 = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_ab_stage_l2
                    )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # Get the first tile info. Read the phase unconditionally so
                # the loop-carried value follows the same SSA pattern as the
                # MMA warp's phase dispatch.
                valid_tile_info = cute.make_rmem_tensor((1,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                tile_phase = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = valid_tile_info[0] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    do_phase_l2 = cutlass.Boolean(False)
                    if cutlass.const_expr(self.enable_linear2):
                        do_phase_l2 = tile_phase == cutlass.Int32(PHASE_LINEAR2)

                    if do_phase_l2:
                        if cutlass.const_expr(self.enable_linear2):
                            a_consumer_state_l2.reset_count()
                            peek_a_full_status_l2 = cutlass.Boolean(1)
                            if a_consumer_state_l2.count < k_tile_cnt_l2:
                                peek_a_full_status_l2 = a_pipeline_l2.consumer_try_wait(
                                    a_consumer_state_l2
                                )
                            a_sync_transform_producer_state_l2.reset_count()

                            for k_tile in cutlass.range(0, k_tile_cnt_l2, 1, unroll=1):
                                a_pipeline_l2.consumer_wait(
                                    a_consumer_state_l2, peek_a_full_status_l2
                                )

                                a_sync_transform_pipeline_l2.producer_commit(
                                    a_sync_transform_producer_state_l2
                                )
                                a_sync_transform_producer_state_l2.advance()

                                a_consumer_state_l2.advance()
                                peek_a_full_status_l2 = cutlass.Boolean(1)
                                if a_consumer_state_l2.count < k_tile_cnt_l2:
                                    peek_a_full_status_l2 = a_pipeline_l2.consumer_try_wait(
                                        a_consumer_state_l2
                                    )
                    else:
                        # Peek (try_wait) A buffer full for k_tile = 0
                        a_consumer_state.reset_count()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt_l1:
                            peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)
                        # Peek (try_wait) a sync transform buffer empty
                        a_sync_transform_producer_state.reset_count()

                        for k_tile in cutlass.range(0, k_tile_cnt_l1, 1, unroll=1):
                            # Conditionally wait for A buffer full
                            a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)

                            a_sync_transform_pipeline.producer_commit(
                                a_sync_transform_producer_state
                            )
                            a_sync_transform_producer_state.advance()

                            # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                            a_consumer_state.advance()
                            peek_a_full_status = cutlass.Boolean(1)
                            if a_consumer_state.count < k_tile_cnt_l1:
                                peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                    tile_phase = sInfo[(5, tile_info_consumer_state.index)]
                    is_valid_tile = valid_tile_info[0] == 1
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait A sync transform buffer empty
                #
                if cutlass.const_expr(self.enable_linear2):
                    a_sync_transform_pipeline_l2.producer_tail(a_sync_transform_producer_state_l2)
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)

        #
        # Specialized TMA B/SFB load warp (warp 9)
        # This warp uses TMA instructions to load B and SFB from global to shared memory
        # with multicast support to reduce L2 memory traffic
        #
        if warp_idx == self.tma_b_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            b_producer_state_l2 = b_producer_state
            sfa_pool_producer_state_l2 = b_producer_state_l2
            if cutlass.const_expr(self.enable_linear2):
                b_producer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage_l2
                )
                sfa_pool_producer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage_l2
                )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            # Slot 5 carries the per-tile phase for the fused path. Keep a
            # Linear1 sentinel when Linear2 is disabled so the FC2 branch is
            # folded away by const_expr.
            tile_phase = cutlass.Int32(PHASE_LINEAR1)
            if cutlass.const_expr(self.enable_linear2):
                tile_phase = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            did_l2_phase_sync = cutlass.Boolean(False)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                expert_idx = mma_tile_coord_mnl[2]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # per-tile phase dispatch. ``do_phase_l2``
                # is a runtime Boolean on the fused path (enable_linear2=True)
                # and a constant False on the Linear1-only path, so the FC2
                # branch below gets constant-folded out of the FC1 kernel.
                do_phase_l2 = cutlass.Boolean(False)
                if cutlass.const_expr(self.enable_linear2):
                    do_phase_l2 = tile_phase == cutlass.Int32(PHASE_LINEAR2)

                if do_phase_l2:
                    if not did_l2_phase_sync:
                        did_l2_phase_sync = cutlass.Boolean(True)
                        self.phase_sync_barrier.arrive_and_wait()
                b_producer_state.reset_count()

                if do_phase_l2:
                    # ------------------------------------------------------
                    # FC2 TMA load loop. Mirrors the FC1
                    # routing below but targets the Linear2 descriptors,
                    # SMEM slots, mcast masks and K-tile count. The inner
                    # ``cutlass.const_expr(self.enable_linear2)`` guard is
                    # the const_expr gate that lets the FC1 path compile away
                    # the whole block even though ``if do_phase_l2:`` is
                    # a runtime branch.
                    # ------------------------------------------------------
                    if cutlass.const_expr(self.enable_linear2):
                        b_producer_state_l2.reset_count()
                        slice_n_l2 = mma_tile_coord_mnl[1]
                        if cutlass.const_expr(self.cta_tile_shape_mnk_l2[1] == 64):
                            slice_n_l2 = mma_tile_coord_mnl[1] // 2
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if b_producer_state_l2.count < k_tile_cnt_l2:
                            peek_ab_empty_status = b_pipeline_l2.producer_try_acquire(
                                b_producer_state_l2
                            )
                        sfa_pool_producer_state_l2.reset_count()
                        peek_sfa_pool_empty_status = cutlass.Boolean(1)
                        if sfa_pool_producer_state_l2.count < k_tile_cnt_l2:
                            peek_sfa_pool_empty_status = sfa_pool_pipeline_l2.producer_try_acquire(
                                sfa_pool_producer_state_l2
                            )
                        l2_pool_block_idx = tile_info[0] // cute.size(tiled_mma_l2.thr_id.shape)
                        l2_arrival_mask_ptr = l2_arrival_mask.iterator + l2_pool_block_idx
                        cached_sfa_pool_l2_arrival_mask = cutlass.Uint64(0)
                        for k_tile in cutlass.range(0, k_tile_cnt_l2, 1, unroll=1):
                            b_pipeline_l2.producer_acquire(
                                b_producer_state_l2, peek_ab_empty_status
                            )
                            sfa_pool_pipeline_l2.producer_acquire(
                                sfa_pool_producer_state_l2, peek_sfa_pool_empty_status
                            )
                            tBsB_l2_pipe = tBsB_l2[(None, b_producer_state_l2.index)]
                            tBsSFB_l2_pipe = tBsSFB_l2[(None, b_producer_state_l2.index)]
                            tAsSFA_pool_tma_pipe = tAsSFA_pool_tma[
                                (None, sfa_pool_producer_state_l2.index)
                            ]
                            tma_bar_l2 = b_pipeline_l2.producer_get_barrier(b_producer_state_l2)
                            tma_bar_sfa_pool_l2 = sfa_pool_pipeline_l2.producer_get_barrier(
                                sfa_pool_producer_state_l2
                            )
                            tAgSFA_pool_l2_slice = tAgSFA_pool_tma[
                                (None, mma_tile_coord_mnl[0], None, 0)
                            ]

                            if cutlass.const_expr(self.num_b_tensors == 1):
                                tBgB_l2_slice = tBgB_l2_0[
                                    (None, mma_tile_coord_mnl[1], None, expert_idx)
                                ]
                                tBgSFB_l2_slice = tBgSFB_l2_0[(None, slice_n_l2, None, expert_idx)]
                                cute.copy(
                                    tma_atoms_b_l2[0],
                                    tBgB_l2_slice[(None, b_producer_state_l2.count)],
                                    tBsB_l2_pipe,
                                    tma_bar_ptr=tma_bar_l2,
                                    mcast_mask=b_l2_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atoms_sfb_l2[0],
                                    tBgSFB_l2_slice[(None, b_producer_state_l2.count)],
                                    tBsSFB_l2_pipe,
                                    tma_bar_ptr=tma_bar_l2,
                                    mcast_mask=sfb_l2_full_mcast_mask,
                                )
                            else:
                                # Expert-routed L2 weight selection. The
                                # ``b_tensor_l_offsets`` table is shared with
                                # FC1 (today both sides share the sharding
                                # fanout); per-L2 offsets are applied when
                                # introduces ``b_tensor_l_offsets_l2``.
                                if cutlass.const_expr(self.num_b_tensors == 2):
                                    if expert_idx < self.b_tensor_l_offsets[1]:
                                        local_l_0 = expert_idx - self.b_tensor_l_offsets[0]
                                        cute.copy(
                                            tma_atoms_b_l2[0],
                                            tBgB_l2_0[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_0,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[0],
                                            tBgSFB_l2_0[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_0,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                    else:
                                        local_l_1 = expert_idx - self.b_tensor_l_offsets[1]
                                        cute.copy(
                                            tma_atoms_b_l2[1],
                                            tBgB_l2_1[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_1,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[1],
                                            tBgSFB_l2_1[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_1,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                elif cutlass.const_expr(self.num_b_tensors == 3):
                                    if expert_idx < self.b_tensor_l_offsets[1]:
                                        local_l_0 = expert_idx - self.b_tensor_l_offsets[0]
                                        cute.copy(
                                            tma_atoms_b_l2[0],
                                            tBgB_l2_0[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_0,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[0],
                                            tBgSFB_l2_0[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_0,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                    elif expert_idx < self.b_tensor_l_offsets[2]:
                                        local_l_1 = expert_idx - self.b_tensor_l_offsets[1]
                                        cute.copy(
                                            tma_atoms_b_l2[1],
                                            tBgB_l2_1[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_1,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[1],
                                            tBgSFB_l2_1[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_1,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                    else:
                                        local_l_2 = expert_idx - self.b_tensor_l_offsets[2]
                                        cute.copy(
                                            tma_atoms_b_l2[2],
                                            tBgB_l2_2[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_2,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[2],
                                            tBgSFB_l2_2[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_2,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                else:
                                    # 4 L2 B tensors
                                    if expert_idx < self.b_tensor_l_offsets[1]:
                                        local_l_0 = expert_idx - self.b_tensor_l_offsets[0]
                                        cute.copy(
                                            tma_atoms_b_l2[0],
                                            tBgB_l2_0[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_0,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[0],
                                            tBgSFB_l2_0[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_0,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                    elif expert_idx < self.b_tensor_l_offsets[2]:
                                        local_l_1 = expert_idx - self.b_tensor_l_offsets[1]
                                        cute.copy(
                                            tma_atoms_b_l2[1],
                                            tBgB_l2_1[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_1,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[1],
                                            tBgSFB_l2_1[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_1,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                    elif expert_idx < self.b_tensor_l_offsets[3]:
                                        local_l_2 = expert_idx - self.b_tensor_l_offsets[2]
                                        cute.copy(
                                            tma_atoms_b_l2[2],
                                            tBgB_l2_2[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_2,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[2],
                                            tBgSFB_l2_2[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_2,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )
                                    else:
                                        local_l_3 = expert_idx - self.b_tensor_l_offsets[3]
                                        cute.copy(
                                            tma_atoms_b_l2[3],
                                            tBgB_l2_3[
                                                (
                                                    None,
                                                    mma_tile_coord_mnl[1],
                                                    b_producer_state_l2.count,
                                                    local_l_3,
                                                )
                                            ],
                                            tBsB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=b_l2_full_mcast_mask,
                                        )
                                        cute.copy(
                                            tma_atoms_sfb_l2[3],
                                            tBgSFB_l2_3[
                                                (
                                                    None,
                                                    slice_n_l2,
                                                    b_producer_state_l2.count,
                                                    local_l_3,
                                                )
                                            ],
                                            tBsSFB_l2_pipe,
                                            tma_bar_ptr=tma_bar_l2,
                                            mcast_mask=sfb_l2_full_mcast_mask,
                                        )

                            # Pool SFA is published by the FC1 epilogue together
                            # with the FP4 pool activation. Match the activation
                            # LDGSTS path and wait for the release-published arrival
                            # bit before issuing the SFA TMA read from HBM.
                            needed_sfa_pool_l2_arrival_mask = cutlass.Uint64(
                                self.l2_arrival_mask_per_k_tile
                            ) << cutlass.Uint64(
                                sfa_pool_producer_state_l2.count * self.l2_arrival_bits_per_k_tile
                            )
                            while (
                                cached_sfa_pool_l2_arrival_mask & needed_sfa_pool_l2_arrival_mask
                            ) != needed_sfa_pool_l2_arrival_mask:
                                cached_sfa_pool_l2_arrival_mask = ld_acquire_gpu_u64(
                                    l2_arrival_mask_ptr
                                )

                            cute.copy(
                                tma_atom_pool_sfa_l2,
                                tAgSFA_pool_l2_slice[(None, sfa_pool_producer_state_l2.count)],
                                tAsSFA_pool_tma_pipe,
                                tma_bar_ptr=tma_bar_sfa_pool_l2,
                                mcast_mask=sfa_pool_l2_full_mcast_mask,
                            )

                            b_producer_state_l2.advance()
                            sfa_pool_producer_state_l2.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if b_producer_state_l2.count < k_tile_cnt_l2:
                                peek_ab_empty_status = b_pipeline_l2.producer_try_acquire(
                                    b_producer_state_l2
                                )
                            peek_sfa_pool_empty_status = cutlass.Boolean(1)
                            if sfa_pool_producer_state_l2.count < k_tile_cnt_l2:
                                peek_sfa_pool_empty_status = (
                                    sfa_pool_pipeline_l2.producer_try_acquire(
                                        sfa_pool_producer_state_l2
                                    )
                                )
                else:
                    # FC1 TMA load loop.
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt_l1:
                        peek_ab_empty_status = b_pipeline.producer_try_acquire(b_producer_state)
                    for k_tile in cutlass.range(0, k_tile_cnt_l1, 1, unroll=1):
                        b_pipeline.producer_acquire(b_producer_state, peek_ab_empty_status)
                        tBsB_pipe = tBsB_0[(None, b_producer_state.index)]
                        tBsSFB_pipe = tBsSFB_0[(None, b_producer_state.index)]
                        tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                        # Select correct B tensor based on expert_idx
                        if cutlass.const_expr(self.num_b_tensors == 1):
                            # Single B tensor - original logic
                            tBgB_slice = tBgB_0[(None, mma_tile_coord_mnl[1], None, expert_idx)]
                            tBgSFB_slice = tBgSFB_0[(None, slice_n, None, expert_idx)]
                            cute.copy(
                                tma_atoms_b[0],
                                tBgB_slice[(None, b_producer_state.count)],
                                tBsB_pipe,
                                tma_bar_ptr=tma_bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atoms_sfb[0],
                                tBgSFB_slice[(None, b_producer_state.count)],
                                tBsSFB_pipe,
                                tma_bar_ptr=tma_bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                        else:
                            # Multi-B tensor - select based on expert_idx
                            # Use nested const_expr ifs to avoid index out of range at compile time
                            if cutlass.const_expr(self.num_b_tensors == 2):
                                # Exactly 2 B tensors
                                if expert_idx < self.b_tensor_l_offsets[1]:
                                    local_l_0 = expert_idx - self.b_tensor_l_offsets[0]
                                    cute.copy(
                                        tma_atoms_b[0],
                                        tBgB_0[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_0,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[0],
                                        tBgSFB_0[
                                            (None, slice_n, b_producer_state.count, local_l_0)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                                else:
                                    local_l_1 = expert_idx - self.b_tensor_l_offsets[1]
                                    cute.copy(
                                        tma_atoms_b[1],
                                        tBgB_1[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_1,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[1],
                                        tBgSFB_1[
                                            (None, slice_n, b_producer_state.count, local_l_1)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                            elif cutlass.const_expr(self.num_b_tensors == 3):
                                # Exactly 3 B tensors
                                if expert_idx < self.b_tensor_l_offsets[1]:
                                    local_l_0 = expert_idx - self.b_tensor_l_offsets[0]
                                    cute.copy(
                                        tma_atoms_b[0],
                                        tBgB_0[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_0,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[0],
                                        tBgSFB_0[
                                            (None, slice_n, b_producer_state.count, local_l_0)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                                elif expert_idx < self.b_tensor_l_offsets[2]:
                                    local_l_1 = expert_idx - self.b_tensor_l_offsets[1]
                                    cute.copy(
                                        tma_atoms_b[1],
                                        tBgB_1[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_1,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[1],
                                        tBgSFB_1[
                                            (None, slice_n, b_producer_state.count, local_l_1)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                                else:
                                    local_l_2 = expert_idx - self.b_tensor_l_offsets[2]
                                    cute.copy(
                                        tma_atoms_b[2],
                                        tBgB_2[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_2,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[2],
                                        tBgSFB_2[
                                            (None, slice_n, b_producer_state.count, local_l_2)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                            else:
                                # 4 B tensors
                                if expert_idx < self.b_tensor_l_offsets[1]:
                                    local_l_0 = expert_idx - self.b_tensor_l_offsets[0]
                                    cute.copy(
                                        tma_atoms_b[0],
                                        tBgB_0[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_0,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[0],
                                        tBgSFB_0[
                                            (None, slice_n, b_producer_state.count, local_l_0)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                                elif expert_idx < self.b_tensor_l_offsets[2]:
                                    local_l_1 = expert_idx - self.b_tensor_l_offsets[1]
                                    cute.copy(
                                        tma_atoms_b[1],
                                        tBgB_1[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_1,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[1],
                                        tBgSFB_1[
                                            (None, slice_n, b_producer_state.count, local_l_1)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                                elif expert_idx < self.b_tensor_l_offsets[3]:
                                    local_l_2 = expert_idx - self.b_tensor_l_offsets[2]
                                    cute.copy(
                                        tma_atoms_b[2],
                                        tBgB_2[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_2,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[2],
                                        tBgSFB_2[
                                            (None, slice_n, b_producer_state.count, local_l_2)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )
                                else:
                                    local_l_3 = expert_idx - self.b_tensor_l_offsets[3]
                                    cute.copy(
                                        tma_atoms_b[3],
                                        tBgB_3[
                                            (
                                                None,
                                                mma_tile_coord_mnl[1],
                                                b_producer_state.count,
                                                local_l_3,
                                            )
                                        ],
                                        tBsB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                                    cute.copy(
                                        tma_atoms_sfb[3],
                                        tBgSFB_3[
                                            (None, slice_n, b_producer_state.count, local_l_3)
                                        ],
                                        tBsSFB_pipe,
                                        tma_bar_ptr=tma_bar,
                                        mcast_mask=sfb_full_mcast_mask,
                                    )

                        b_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if b_producer_state.count < k_tile_cnt_l1:
                            peek_ab_empty_status = b_pipeline.producer_try_acquire(b_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                # refresh ``tile_phase`` for the next tile
                # so the top of the while loop sees a coherent phase tag.
                if cutlass.const_expr(self.enable_linear2):
                    tile_phase = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            if cutlass.const_expr(self.enable_linear2):
                sfa_pool_pipeline_l2.producer_tail(sfa_pool_producer_state_l2)
                b_pipeline_l2.producer_tail(b_producer_state_l2)
            b_pipeline.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB
            #
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

            # ------------------------------------------------------------------
            # Linear2 tmem pointers + S2T copy partitions.
            # The accumulator / SFA / SFB reuse the same tmem columns as FC1
            # since the naive scheduler serialises FC1 then FC2 per tile; the
            # layouts switch to the L2 variants so the FC2 MMA sees the right
            # MMA_M/MMA_N shapes. The fused path rewires the SFA s2t copy
            # to read from ``sSFA_pool`` (populated by the LDGSTS-A FC2
            # phase that streams activations from the HBM pool), matching
            # the FC2 SFB side which reads from ``sSFB_l2``.
            # ------------------------------------------------------------------
            tCtAcc_base_l2 = None
            tCtSFA_l2 = None
            tCtSFB_l2 = None
            tCtSFA_layout_l2 = None
            tCtSFB_layout_l2 = None
            tiled_copy_s2t_sfa_l2 = None
            tCsSFA_compact_s2t_l2 = None
            tCtSFA_compact_s2t_l2 = None
            tiled_copy_s2t_sfb_l2 = None
            tCsSFB_compact_s2t_l2 = None
            tCtSFB_compact_s2t_l2 = None
            if cutlass.const_expr(self.enable_linear2):
                tCtAcc_base_l2 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_l2.layout)
                sfa_tmem_ptr_l2 = cute.recast_ptr(
                    acc_tmem_ptr + self.num_accumulator_tmem_cols_l2,
                    dtype=self.sf_dtype,
                )
                tCtSFA_layout_l2 = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma_l2,
                    self.mma_tiler_l2,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged_l2, (None, None, None, 0)),
                )
                tCtSFA_l2 = cute.make_tensor(sfa_tmem_ptr_l2, tCtSFA_layout_l2)
                sfb_tmem_ptr_l2 = cute.recast_ptr(
                    acc_tmem_ptr + self.num_accumulator_tmem_cols_l2 + self.num_sfa_tmem_cols_l2,
                    dtype=self.sf_dtype,
                )
                tCtSFB_layout_l2 = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma_l2,
                    self.mma_tiler_l2,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged_l2, (None, None, None, 0)),
                )
                tCtSFB_l2 = cute.make_tensor(sfb_tmem_ptr_l2, tCtSFB_layout_l2)

                (
                    tiled_copy_s2t_sfa_l2,
                    tCsSFA_compact_s2t_l2,
                    tCtSFA_compact_s2t_l2,
                ) = self.mainloop_s2t_copy_and_partition(sSFA_pool, tCtSFA_l2)
                (
                    tiled_copy_s2t_sfb_l2,
                    tCsSFB_compact_s2t_l2,
                    tCtSFB_compact_s2t_l2,
                ) = self.mainloop_s2t_copy_and_partition(sSFB_l2, tCtSFB_l2)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_consumer_state_l2 = a_sync_transform_consumer_state
                if cutlass.const_expr(self.enable_linear2):
                    a_sync_transform_consumer_state_l2 = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_ab_stage_l2
                    )
            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            a_consumer_state_l2 = a_consumer_state
            if cutlass.const_expr(self.enable_linear2):
                a_consumer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage_l2
                )

            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            b_consumer_state_l2 = b_consumer_state
            sfa_pool_consumer_state_l2 = b_consumer_state_l2
            if cutlass.const_expr(self.enable_linear2):
                b_consumer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage_l2
                )
                sfa_pool_consumer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage_l2
                )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            acc_producer_state_l2 = acc_producer_state
            if cutlass.const_expr(self.enable_linear2):
                acc_producer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage_l2
                )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            # Read the phase unconditionally so Cute DSL carries it around
            # the loop back-edge like the other tile-info fields. Linear1-only
            # kernels still fold the FC2 branch away because ``do_phase_l2`` is
            # a const_expr False constant there.
            tile_phase = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            did_l2_phase_sync = cutlass.Boolean(False)
            while is_valid_tile:
                # per-tile FC1 vs FC2 MMA dispatch. On
                # the Linear1-only path (enable_linear2=False) ``do_phase_l2``
                # is a constant False so the FC2 branch collapses via
                # const_expr DCE and the runtime ``if do_phase_l2:`` folds
                # to the FC1 else-branch. Compute the shared tile coord
                # once and branch on the phase tag.
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                do_phase_l2 = cutlass.Boolean(False)
                if cutlass.const_expr(self.enable_linear2):
                    do_phase_l2 = tile_phase == cutlass.Int32(PHASE_LINEAR2)

                if do_phase_l2:
                    if not did_l2_phase_sync:
                        did_l2_phase_sync = cutlass.Boolean(True)
                        self.phase_sync_barrier.arrive_and_wait()
                # Keep the phase branch immediately after the loop-carried tile
                # metadata refresh so L1 and L2 scheduler slots dispatch through
                # distinct mainloop states.
                if do_phase_l2:
                    # ------------------------------------------------------
                    # FC2 MMA mainloop. Mirrors the FC1 body below but uses
                    # ``self.tiled_mma_l2`` / ``tCrA_l2`` / ``tCrB_l2`` /
                    # ``tCtAcc_base_l2`` / ``tCtSFA_l2`` / ``tCtSFB_l2`` /
                    # ``tiled_copy_s2t_sfa_l2`` / ``tiled_copy_s2t_sfb_l2``
                    # and iterates ``k_tile_cnt_l2`` K-tiles. The inner
                    # ``cutlass.const_expr(self.enable_linear2)`` guard is
                    # the const_expr gate for the FC1 compile path.
                    # ------------------------------------------------------
                    if cutlass.const_expr(self.enable_linear2):
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_consumer_state_l2.reset_count()
                            peek_a_sync_transform_full_status_l2 = cutlass.Boolean(1)
                            if (
                                a_sync_transform_consumer_state_l2.count < k_tile_cnt_l2
                                and is_leader_cta
                            ):
                                peek_a_sync_transform_full_status_l2 = (
                                    a_sync_transform_pipeline_l2.consumer_try_wait(
                                        a_sync_transform_consumer_state_l2
                                    )
                                )
                            a_consumer_state_l2.reset_count()
                        else:
                            a_consumer_state_l2.reset_count()
                            peek_a_full_status = cutlass.Boolean(1)
                            if a_consumer_state_l2.count < k_tile_cnt_l2:
                                peek_a_full_status = a_pipeline_l2.consumer_try_wait(
                                    a_consumer_state_l2
                                )

                        b_consumer_state_l2.reset_count()
                        peek_b_full_status = cutlass.Boolean(1)
                        if b_consumer_state_l2.count < k_tile_cnt_l2 and is_leader_cta:
                            peek_b_full_status = b_pipeline_l2.consumer_try_wait(
                                b_consumer_state_l2
                            )
                        sfa_pool_consumer_state_l2.reset_count()
                        peek_sfa_pool_full_status = cutlass.Boolean(1)
                        if sfa_pool_consumer_state_l2.count < k_tile_cnt_l2 and is_leader_cta:
                            peek_sfa_pool_full_status = sfa_pool_pipeline_l2.consumer_try_wait(
                                sfa_pool_consumer_state_l2
                            )

                        if cutlass.const_expr(self.overlapping_accum_l2):
                            acc_stage_index_l2 = acc_producer_state_l2.phase ^ 1
                        else:
                            acc_stage_index_l2 = acc_producer_state_l2.index

                        tCtAcc_l2 = tCtAcc_base_l2[(None, None, None, acc_stage_index_l2)]

                        tCtSFB_mma_l2 = tCtSFB_l2
                        if cutlass.const_expr(self.cta_tile_shape_mnk_l2[1] == 192):
                            offset = (
                                cutlass.Int32(2)
                                if mma_tile_coord_mnl[1] % 2 == 1
                                else cutlass.Int32(0)
                            )
                            shifted_ptr = cute.recast_ptr(
                                acc_tmem_ptr
                                + self.num_accumulator_tmem_cols_l2
                                + self.num_sfa_tmem_cols_l2
                                + offset,
                                dtype=self.sf_dtype,
                            )
                            tCtSFB_mma_l2 = cute.make_tensor(shifted_ptr, tCtSFB_layout_l2)
                        elif cutlass.const_expr(self.cta_tile_shape_mnk_l2[1] == 64):
                            offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                            shifted_ptr = cute.recast_ptr(
                                acc_tmem_ptr
                                + self.num_accumulator_tmem_cols_l2
                                + self.num_sfa_tmem_cols_l2
                                + offset,
                                dtype=self.sf_dtype,
                            )
                            tCtSFB_mma_l2 = cute.make_tensor(shifted_ptr, tCtSFB_layout_l2)

                        if is_leader_cta:
                            acc_pipeline_l2.producer_acquire(acc_producer_state_l2)

                        tiled_mma_l2.set(tcgen05.Field.ACCUMULATE, False)

                        # FC2 consumes its own A/B/SF stage states and writes to
                        # the L2 accumulator pipeline.
                        for k_tile in cutlass.range(k_tile_cnt_l2):
                            if is_leader_cta:
                                if cutlass.const_expr(self.use_2cta_instrs):
                                    a_sync_transform_pipeline_l2.consumer_wait(
                                        a_sync_transform_consumer_state_l2,
                                        peek_a_sync_transform_full_status_l2,
                                    )
                                else:
                                    a_pipeline_l2.consumer_wait(
                                        a_consumer_state_l2, peek_a_full_status
                                    )
                                b_pipeline_l2.consumer_wait(b_consumer_state_l2, peek_b_full_status)
                                sfa_pool_pipeline_l2.consumer_wait(
                                    sfa_pool_consumer_state_l2, peek_sfa_pool_full_status
                                )

                                # FC2 A, pool-SFA, and B/SFB are produced by separate
                                # pipelines. Keep their stage indexes separate at the
                                # S2T and MMA boundaries.
                                s2t_stage_coord_sfa_l2 = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    sfa_pool_consumer_state_l2.index,
                                )
                                s2t_stage_coord_sfb_l2 = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    b_consumer_state_l2.index,
                                )
                                num_kblocks_l2 = cute.size(tCrA_l2, mode=[2])
                                tCsSFA_compact_s2t_staged_l2 = tCsSFA_compact_s2t_l2[
                                    s2t_stage_coord_sfa_l2
                                ]
                                tCsSFB_compact_s2t_staged_l2 = tCsSFB_compact_s2t_l2[
                                    s2t_stage_coord_sfb_l2
                                ]
                                cute.copy(
                                    tiled_copy_s2t_sfa_l2,
                                    tCsSFA_compact_s2t_staged_l2,
                                    tCtSFA_compact_s2t_l2,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfb_l2,
                                    tCsSFB_compact_s2t_staged_l2,
                                    tCtSFB_compact_s2t_l2,
                                )

                                for kblock_idx in cutlass.range(num_kblocks_l2, unroll_full=True):
                                    kblock_coord_a_l2 = (
                                        None,
                                        None,
                                        kblock_idx,
                                        a_consumer_state_l2.index,
                                    )
                                    kblock_coord_b_l2 = (
                                        None,
                                        None,
                                        kblock_idx,
                                        b_consumer_state_l2.index,
                                    )
                                    sf_kblock_coord_l2 = (None, None, kblock_idx)
                                    tiled_mma_l2.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA_l2[sf_kblock_coord_l2].iterator,
                                    )
                                    tiled_mma_l2.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma_l2[sf_kblock_coord_l2].iterator,
                                    )
                                    cute.gemm(
                                        tiled_mma_l2,
                                        tCtAcc_l2,
                                        tCrA_l2[kblock_coord_a_l2],
                                        tCrB_l2[kblock_coord_b_l2],
                                        tCtAcc_l2,
                                    )
                                    tiled_mma_l2.set(tcgen05.Field.ACCUMULATE, True)

                                a_pipeline_l2.consumer_release(a_consumer_state_l2)
                                if cutlass.const_expr(self.use_2cta_instrs):
                                    a_sync_transform_pipeline_l2.consumer_release(
                                        a_sync_transform_consumer_state_l2
                                    )
                                b_pipeline_l2.consumer_release(b_consumer_state_l2)
                                sfa_pool_pipeline_l2.consumer_release(sfa_pool_consumer_state_l2)

                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_consumer_state_l2.advance()
                                peek_a_sync_transform_full_status_l2 = cutlass.Boolean(1)
                                if a_sync_transform_consumer_state_l2.count < k_tile_cnt_l2:
                                    if is_leader_cta:
                                        peek_a_sync_transform_full_status_l2 = (
                                            a_sync_transform_pipeline_l2.consumer_try_wait(
                                                a_sync_transform_consumer_state_l2
                                            )
                                        )
                                a_consumer_state_l2.advance()
                            else:
                                a_consumer_state_l2.advance()
                                peek_a_full_status = cutlass.Boolean(1)
                                if a_consumer_state_l2.count < k_tile_cnt_l2:
                                    peek_a_full_status = a_pipeline_l2.consumer_try_wait(
                                        a_consumer_state_l2
                                    )

                            b_consumer_state_l2.advance()
                            sfa_pool_consumer_state_l2.advance()
                            peek_b_full_status = cutlass.Boolean(1)
                            if b_consumer_state_l2.count < k_tile_cnt_l2:
                                if is_leader_cta:
                                    peek_b_full_status = b_pipeline_l2.consumer_try_wait(
                                        b_consumer_state_l2
                                    )
                            peek_sfa_pool_full_status = cutlass.Boolean(1)
                            if sfa_pool_consumer_state_l2.count < k_tile_cnt_l2:
                                if is_leader_cta:
                                    peek_sfa_pool_full_status = (
                                        sfa_pool_pipeline_l2.consumer_try_wait(
                                            sfa_pool_consumer_state_l2
                                        )
                                    )
                else:
                    # FC1 MMA mainloop.
                    # Peek (try_wait) AB buffer full for k_tile = 0
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.reset_count()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if a_sync_transform_consumer_state.count < k_tile_cnt_l1 and is_leader_cta:
                            peek_a_sync_transform_full_status = (
                                a_sync_transform_pipeline.consumer_try_wait(
                                    a_sync_transform_consumer_state
                                )
                            )
                        a_consumer_state.reset_count()
                    else:
                        a_consumer_state.reset_count()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt_l1:
                            peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                    b_consumer_state.reset_count()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt_l1 and is_leader_cta:
                        peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                    # ``mma_tile_coord_mnl`` is now computed above the phase
                    # dispatch so both branches see the same value; the
                    # duplicate assignment is removed.

                    # Get accumulator stage index
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_producer_state.phase ^ 1
                    else:
                        acc_stage_index = acc_producer_state.index

                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                    # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or
                    # cta_tile_shape_n=64
                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                        # If this is an ODD tile, shift the TMEM start address for
                        # cta_tile_shape_n=192 case by two words
                        # (ignores first 64 columns of SFB)
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
                        # Move in increments of 64 columns of SFB
                        offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                        #
                    # Wait for accumulator buffer empty
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)
                    #
                    # Mma mainloop
                    #

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    # Linear1 consumes the original A/B/SF stage states and writes
                    # to the L1 accumulator pipeline.
                    for k_tile in cutlass.range(k_tile_cnt_l1):
                        # Set tensor memory buffer for current tile
                        # (MMA, MMA_M, MMA_N)

                        if is_leader_cta:
                            # Conditionally wait for AB buffer full
                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_pipeline.consumer_wait(
                                    a_sync_transform_consumer_state,
                                    peek_a_sync_transform_full_status,
                                )
                            else:
                                a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                            b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)

                            #  Copy SFA/SFB from smem to tmem
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                b_consumer_state.index,
                            )
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

                            # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                            num_kblocks = cute.size(tCrA, mode=[2])

                            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    b_consumer_state.index,
                                )

                                # Set SFA/SFB tensor to tiled_mma
                                sf_kblock_coord = (None, None, kblock_idx)
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
                                # Enable accumulate on tCtAcc after first kblock
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            # Async arrive AB buffer empty
                            a_pipeline.consumer_release(a_consumer_state)
                            if cutlass.const_expr(self.use_2cta_instrs):
                                a_sync_transform_pipeline.consumer_release(
                                    a_sync_transform_consumer_state
                                )
                            b_pipeline.consumer_release(b_consumer_state)

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_consumer_state.advance()
                            peek_a_sync_transform_full_status = cutlass.Boolean(1)
                            if a_sync_transform_consumer_state.count < k_tile_cnt_l1:
                                if is_leader_cta:
                                    peek_a_sync_transform_full_status = (
                                        a_sync_transform_pipeline.consumer_try_wait(
                                            a_sync_transform_consumer_state
                                        )
                                    )
                            a_consumer_state.advance()
                        else:
                            a_consumer_state.advance()
                            peek_a_full_status = cutlass.Boolean(1)
                            if a_consumer_state.count < k_tile_cnt_l1:
                                peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)

                        b_consumer_state.advance()
                        peek_b_full_status = cutlass.Boolean(1)
                        if b_consumer_state.count < k_tile_cnt_l1:
                            if is_leader_cta:
                                peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                #
                # Async arrive accumulator buffer full(each phase)
                #
                if do_phase_l2:
                    if cutlass.const_expr(self.enable_linear2):
                        if is_leader_cta:
                            acc_pipeline_l2.producer_commit(acc_producer_state_l2)
                        acc_producer_state_l2.advance()
                else:
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                # refresh ``tile_phase`` for the next tile
                # so the top of the MMA while loop dispatches correctly.
                #
                # ``cutlass.const_expr(self.enable_linear2)`` earlier and
                # failed to propagate across the loop back-edge via SSA
                # φ-node). See the matching pre-load comment above.
                tile_phase = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                # Must fire AFTER consumer_wait (L4973) so the read is
                # properly synced with scheduler's commit.
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            if cutlass.const_expr(self.enable_linear2):
                acc_pipeline_l2.producer_tail(acc_producer_state_l2)
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_base_l2 = None
            if cutlass.const_expr(self.enable_linear2):
                tCtAcc_base_l2 = cute.make_tensor(tmem_ptr, tCtAcc_fake_l2.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            # ----------------------------------------------------------------
            # FC1 epilogue pool-store partition. When
            # ``enable_linear2=True`` the FC1 epilogue warp redirects the
            # S2G stream from ``tma_atom_c`` -> ``tCgC`` to
            # ``tma_atom_pool_store`` -> ``tCgPool_store`` (the permuted
            # HBM pool tile that the FC2 LDGSTS-A phase will read). The
            # SMEM source partition ``bSG_sC_pool`` is structurally
            # identical to ``bSG_sC`` because both use the same
            # ``epi_smem_layout`` (sC staging), but we call the helper
            # again so the descriptor / partition tuple binds to the
            # pool-store atom. Gated on ``cutlass.const_expr`` so the FC1
            # Linear1-only path never builds this partition.
            # ----------------------------------------------------------------
            bSG_sC_pool = bSG_sC
            bSG_gC_pool_partitioned = bSG_gC_partitioned
            if cutlass.const_expr(self.enable_linear2):
                (
                    _,
                    bSG_sC_pool,
                    bSG_gC_pool_partitioned,
                ) = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_pool_store, tCgPool_store, epi_tile, sC
                )

            # Linear2 (FC2) epilogue partition setup. The helper
            # returns the TMEM->register partition for a full-N accumulator
            # tile (no SwiGLU halving). The block is gated on
            # ``enable_linear2`` so the Linear1-only path DCEs it cleanly;
            # ``tCgC`` is a trace-time placeholder for the global output
            # tensor until the fused path threads a dedicated ``mOut_mnl`` through the
            # device kernel signature. The register output tensor
            # (``tTR_rC_l2 = cute.make_rmem_tensor(tTR_rAcc_l2.shape,
            # self.l2_out_dtype)``) is created where the
            # combine epilogue body consumes it.
            tiled_copy_t2r_l2 = None
            tTR_tAcc_base_l2 = None
            tTR_rAcc_l2 = None
            if cutlass.const_expr(self.enable_linear2):
                (
                    tiled_copy_t2r_l2,
                    tTR_tAcc_base_l2,
                    tTR_rAcc_l2,
                ) = self.epilog_tmem_copy_and_partition_linear2(
                    epi_tidx, tCtAcc_base_l2, tCgOut, epi_tile, use_2cta_instrs
                )

            # ----------------------------------------------------------------
            # FC1 epilogue SFC destination selector. The
            # FC1 requant path stores the per-vector scale factor tensor
            # ``SFC`` alongside the FP4 activation. When
            # ``enable_linear2=True`` the SFC rows must land in the HBM
            # pool SFC buffer (``pool_sfc_tensor_l1``) consumed by the FC2
            # LDGSTS-A phase; otherwise they stream to the standalone FC1
            # ``mSFC_mnl`` output tensor. Both source tensors share the
            # same ``tile_atom_to_shape_SF`` layout over the FC1 output
            # shape so the downstream partitioning is structurally
            # identical. Selected via ``cutlass.const_expr`` so the FC1
            # Linear1-only path compiles straight to ``mSFC_mnl``.
            # ----------------------------------------------------------------
            if cutlass.const_expr(self.enable_linear2):
                mSFC_eff = pool_sfc_tensor_l1
            else:
                mSFC_eff = mSFC_mnl

            if cutlass.const_expr(self.generate_sfc):
                norm_const = norm_const_tensor[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFC_mnl = cute.local_tile(mSFC_eff, epi_tile, (None, None, None))

                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
                tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFC = cute.make_rmem_tensor(
                    tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            acc_consumer_state_l2 = acc_consumer_state
            if cutlass.const_expr(self.enable_linear2):
                acc_consumer_state_l2 = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage_l2
                )

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
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

            # Get the first tile info. Slot 4 carries mn_limit for the
            # Linear2 combine path; slot 5 is phase and is read separately.
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            # Slot 5 carries the per-tile phase consumed by the loop-top
            # dispatch below. This mirrors the LDGSTS-A, TMA-B, and MMA warps.
            if cutlass.const_expr(self.enable_linear2):
                tile_phase = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            did_l2_phase_sync = cutlass.Boolean(False)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #
                expert_idx = mma_tile_coord_mnl[2]

                # ----------------------------------------------------------
                # runtime phase dispatch. ``do_phase_l2``
                # is ``cutlass.Boolean(False)`` on the Linear1-only path
                # (``enable_linear2=False``) so the outer
                # ``if not do_phase_l2:`` below folds to the unconditional
                # FC1 body (byte-equal to the standalone FC1 control flow). In
                # ``enable_linear2=True`` mode the scheduler emits
                # ``PHASE_LINEAR1`` + ``PHASE_LINEAR2`` per valid tile
                # (s1d); this signal steers FC1 SwiGLU+requant+pool-store
                # vs the FC2 combine scatter body below.
                # ----------------------------------------------------------
                do_phase_l2 = cutlass.Boolean(False)
                if cutlass.const_expr(self.enable_linear2):
                    do_phase_l2 = tile_phase == cutlass.Int32(PHASE_LINEAR2)

                if do_phase_l2:
                    if not did_l2_phase_sync:
                        did_l2_phase_sync = cutlass.Boolean(True)
                        c_pipeline.producer_tail()
                        self.phase_sync_barrier.arrive_and_wait()
                # wrap the Linear1 tile body (alpha lookup,
                # partition slicing, subtile SwiGLU+FP4+TMA store,
                # consumer_release) in a phase-dispatched branch. On the
                # Linear1-only path ``do_phase_l2`` folds to False so the
                # outer predicate collapses to True -> FC1 body always
                # taken -> byte-equal to the standalone FC1 behavior. The
                # ``else:`` branch implements Linear2 combine+scatter and
                # the predicate switches from
                # ``cutlass.const_expr(not self.enable_linear2)`` to the
                # runtime ``not do_phase_l2`` so the FC1 body also runs
                # for ``PHASE_LINEAR1`` tiles in ``enable_linear2=True``
                # mode.
                if not do_phase_l2:
                    # Select alpha from correct tensor based on expert_idx
                    # Initialize alpha_val first to avoid DSL "None prior to if" error
                    alpha_val = alpha_tuple[0][expert_idx - self.b_tensor_l_offsets[0]]
                    if cutlass.const_expr(self.num_b_tensors == 1):
                        pass  # Already initialized above
                    elif cutlass.const_expr(self.num_b_tensors == 2):
                        if expert_idx >= self.b_tensor_l_offsets[1]:
                            alpha_val = alpha_tuple[1][expert_idx - self.b_tensor_l_offsets[1]]
                    elif cutlass.const_expr(self.num_b_tensors == 3):
                        if (
                            expert_idx >= self.b_tensor_l_offsets[1]
                            and expert_idx < self.b_tensor_l_offsets[2]
                        ):
                            alpha_val = alpha_tuple[1][expert_idx - self.b_tensor_l_offsets[1]]
                        elif expert_idx >= self.b_tensor_l_offsets[2]:
                            alpha_val = alpha_tuple[2][expert_idx - self.b_tensor_l_offsets[2]]
                    else:
                        # 4 B tensors
                        if (
                            expert_idx >= self.b_tensor_l_offsets[1]
                            and expert_idx < self.b_tensor_l_offsets[2]
                        ):
                            alpha_val = alpha_tuple[1][expert_idx - self.b_tensor_l_offsets[1]]
                        elif (
                            expert_idx >= self.b_tensor_l_offsets[2]
                            and expert_idx < self.b_tensor_l_offsets[3]
                        ):
                            alpha_val = alpha_tuple[2][expert_idx - self.b_tensor_l_offsets[2]]
                        elif expert_idx >= self.b_tensor_l_offsets[3]:
                            alpha_val = alpha_tuple[3][expert_idx - self.b_tensor_l_offsets[3]]

                    #
                    # Slice to per mma tile index
                    #
                    # pick FC1 vs pool store destination.
                    # ``enable_linear2=True``: stream to the permuted HBM
                    # pool via ``tma_atom_pool_store`` / ``bSG_sC_pool`` /
                    # ``bSG_gC_pool_partitioned``; Linear1-only path reuses
                    # ``tma_atom_c`` / ``bSG_sC`` / ``bSG_gC_partitioned``.
                    # ``sC`` is shared between FC1 and pool store (same
                    # ``epi_smem_layout``); only the S2G destination and
                    # TMA descriptor atom change.
                    if cutlass.const_expr(self.enable_linear2):
                        tma_atom_store_eff = tma_atom_pool_store
                        bSG_sC_eff = bSG_sC_pool
                        bSG_gC_partitioned_eff = bSG_gC_pool_partitioned
                    else:
                        tma_atom_store_eff = tma_atom_c
                        bSG_sC_eff = bSG_sC
                        bSG_gC_partitioned_eff = bSG_gC_partitioned

                    bSG_gC = None
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    bSG_gC = bSG_gC_partitioned_eff[
                        (
                            None,
                            None,
                            None,
                            mma_tile_coord_mnl[0],
                            mma_tile_coord_mnl[1],
                            0,
                        )
                    ]

                    # Get accumulator stage index
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_consumer_state.phase
                        reverse_subtile = (
                            cutlass.Boolean(True)
                            if acc_stage_index == 0
                            else cutlass.Boolean(False)
                        )
                    else:
                        acc_stage_index = acc_consumer_state.index

                    # Set tensor memory buffer for current tile
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                    if cutlass.const_expr(self.generate_sfc):
                        # (T2R, T2R_M, T2R_N, RestM, RestN)
                        tCgSFC_mn = tCgSFC_mnl[
                            (
                                None,
                                None,
                                None,
                                None,
                                None,
                                0,
                            )
                        ]

                    #
                    # Wait for accumulator buffer full
                    #
                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    #
                    # Process accumulator subtiles with SwiGLU fusion and store to global memory
                    # Each iteration processes a pair of subtiles (up, gate) and computes
                    # up * silu(gate)
                    #
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
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                        tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]

                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                        #
                        # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                        #
                        if cutlass.const_expr(self.overlapping_accum):
                            if subtile_idx // 2 == self.iter_acc_early_release_in_epilogue:
                                # Fence for TMEM load
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()

                        acc_vec_up = tTR_rAcc_up.load()
                        acc_vec_gate = tTR_rAcc_gate.load()

                        #
                        # SwiGLU activation: output = up * silu(gate)
                        # where silu(x) = x * sigmoid(x)
                        # up and gate are extracted from interleaved accumulator subtiles
                        #
                        tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                        if cutlass.const_expr(self.vectorized_f32):
                            # SwiGLU Packed Version: uses f32x2 packed operations for better performance
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            # where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                            LOG2_E = cutlass.Float32(1.4426950408889634)
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                                acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_up[i], acc_vec_up[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                )
                                acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                )
                                tCompute_log2e = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                    (-LOG2_E, -LOG2_E),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.add_packed_f32x2(
                                    (
                                        cute.math.exp2(tCompute_log2e[0], fastmath=True),
                                        cute.math.exp2(tCompute_log2e[1], fastmath=True),
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
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_up_alpha[0], acc_vec_up_alpha[1]),
                                )
                        else:
                            # SwiGLU Unpacked Version: scalar operations
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                                acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(alpha_val)
                                acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(alpha_val)
                                tCompute[i] = acc_vec_up_alpha * silu_f32(
                                    acc_vec_gate_alpha, fastmath=True
                                )

                        if cutlass.const_expr(self.generate_sfc):
                            #
                            # Quantization path for Float4E2M1FN output:
                            # 1. Compute per-vector absolute max from SwiGLU result
                            # 2. Generate scale factor C (SFC) based on max values
                            # 3. Store SFC to global memory
                            # 4. Quantize output by scaling with reciprocal of SFC
                            #
                            # Assume subtile partitioned always happens on n dimension
                            sfc_subtile_idx_mn = (
                                tile_info[0] * self.epi_tile_cnt[0],
                                tile_info[1] * self.epi_tile_cnt[1] + real_subtile_idx,
                            )
                            tCgSFC = tCgSFC_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfc_subtile_idx_mn,
                                )
                            ]

                            #
                            # Get absolute max across a vector and Compute SFC
                            #
                            tTR_rAcc_frg = cute.logical_divide(
                                tCompute, cute.make_layout(self.sf_vec_size)
                            )
                            acc_frg = tTR_rAcc_frg.load()
                            acc_frg = epilogue_op(acc_frg)

                            # Apply element-wise absolute value using math.absf (supports vectors)
                            abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                            abs_acc_frg = type(acc_frg)(
                                abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                            )

                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,  # Use 0.0 as init for abs values
                                    )
                                for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                            (
                                                self.get_dtype_rcp_limits(self.c_dtype),
                                                self.get_dtype_rcp_limits(self.c_dtype),
                                            ),
                                        )
                                    )
                                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                            (norm_const, norm_const),
                                        )
                                    )
                            else:
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrSFC_pvscale[vi] = (
                                        abs_acc_frg[None, vi].reduce(
                                            cute.ReductionOp.MAX,
                                            cutlass.Float32(0.0),
                                            0,  # Use 0.0 as init for abs values
                                        )
                                        * self.get_dtype_rcp_limits(self.c_dtype)
                                        * norm_const
                                    )

                            # TODO: need to add f32x2 -> f8x2 conversion
                            tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))

                            #
                            # Store SFC to global memory
                            #
                            # TODO: Need to think about predicate on it
                            # if cute.elem_less():
                            cute.autovec_copy(tCrSFC, tCgSFC)

                            #
                            # Compute quantized output values and convert to C type
                            #
                            # TODO: need to add f8x2 -> f32x2 conversion
                            tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                            fp32_max = cutlass.Float32(3.40282346638528859812e38)
                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                                    acc_scale = cute.arch.mul_packed_f32x2(
                                        (
                                            cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi]),
                                            cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi + 1]),
                                        ),
                                        (norm_const, norm_const),
                                    )
                                    acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                                    acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)

                                    vec0 = tTR_rAcc_frg[None, vi]
                                    vec1 = tTR_rAcc_frg[None, vi + 1]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                            (vec0[ei], vec1[ei]),
                                            (acc_scale_min0, acc_scale_min1),
                                        )
                            else:
                                for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                                    # TODO:Need to add E8M0 rcp approximation
                                    acc_scale = norm_const * cute.arch.rcp_approx(
                                        tCrSFC_qpvscale_up[vi]
                                    )
                                    acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                    vec = tTR_rAcc_frg[None, vi]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec[ei] = vec[ei] * acc_scale

                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                        else:
                            #
                            # Convert to C type
                            #
                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                            tRS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory (or HBM pool when
                        # ``enable_linear2=True`` -- S2d).
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_store_eff,
                                bSG_sC_eff[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(self.enable_linear2):
                        c_pipeline.producer_tail()
                        self.epilog_sync_barrier.arrive_and_wait()
                        # Pool SFC is written with ordinary global stores by all epilogue
                        # threads. Fence those stores before one elected thread publishes
                        # the FC2 arrival bit consumed with acquire semantics.
                        fence_acq_rel_gpu()
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            with cute.arch.elect_one():
                                l2_arrival_bit = cutlass.Uint64(1) << cutlass.Uint64(
                                    mma_tile_coord_mnl[1]
                                )
                                red_or_release_gpu_u64(
                                    l2_arrival_mask.iterator + mma_tile_coord_mnl[0],
                                    l2_arrival_bit,
                                )
                        self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # Async arrive accumulator buffer empty
                    #
                    if cutlass.const_expr(not self.overlapping_accum):
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                else:
                    # Linear2 (FC2) combine body — topk-weighted
                    # scatter-atomic-add to the dense output tensor. Ported
                    # from ``blockscaled_contiguous_grouped_gemm_finalize_fusion.py``
                    # L2230-L2332 (BF16 path only). Compiled only when
                    # ``enable_linear2=True`` (wired for fused mode); entirely DCE'd
                    # on the Linear1-only path. The fused path threads the four
                    # names (``permuted_idx_to_expanded_idx``, ``token_final_scales``,
                    # ``alpha_l2_tuple``, ``out_tensor``) through as explicit
                    # kernel parameters, so the prior ``# noqa: F821`` markers
                    # are no longer needed.
                    #
                    # FC2 combine const_expr gate: the outer predicate is now
                    # runtime ``not do_phase_l2`` (not compile-time
                    # ``const_expr(not enable_linear2)``), so cuteDSL traces
                    # both branches even on the Linear1-only path. The body below
                    # dereferences ``permuted_idx_to_expanded_idx`` / other
                    # Optional kernel params that are ``None`` on the FC1
                    # Linear1-only path. Wrap the body in
                    # ``cutlass.const_expr(self.enable_linear2)`` so the IR
                    # builder folds it to an empty else block when
                    # ``enable_linear2=False`` (mirroring the FC2 MMA
                    # warp and LDGSTS-A ``if cutlass.const_expr(...)``
                    # const_expr gate idiom).
                    if cutlass.const_expr(self.enable_linear2):
                        # ``reverse_subtile`` in the combine scope because
                        # cuteDSL's if/else tracing does not leak them from
                        # the FC1 branch (same scoping contract that forced
                        # the fused path's ``alpha_val`` drop).
                        if cutlass.const_expr(self.overlapping_accum_l2):
                            acc_stage_index = acc_consumer_state_l2.phase
                            reverse_subtile = (
                                cutlass.Boolean(True)
                                if acc_stage_index == 0
                                else cutlass.Boolean(False)
                            )
                        else:
                            acc_stage_index = acc_consumer_state_l2.index
                            reverse_subtile = cutlass.Boolean(False)

                        tile_m_start = tile_info[0] * self.cta_tile_shape_mnk_l2[0]
                        permuted_row = tile_m_start + epi_tidx
                        expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                        is_valid_row = permuted_row < tile_info[4]

                        # Overwrite alpha_val with the topk-weighted factor for
                        # the current (token, topk) slot. Following DeepGEMM
                        # §一 the route-weight multiply happens in the L2
                        # epilogue; this is the spot where it fuses into the
                        # pre-scatter scale.
                        token_idx = cutlass.Int32(0)
                        topk_idx = cutlass.Int32(0)
                        token_scale = self.final_scale_dtype(0.0)
                        if is_valid_row:
                            token_idx = expanded_idx // self.topk
                            topk_idx = expanded_idx % self.topk
                            token_scale = token_final_scales[(token_idx, topk_idx)]

                        # Re-derive alpha from the Linear2 B-tensor-specific
                        # tuple. The current implementation assumes ``num_b_tensors_l2 == 1``;
                        # multi-tuple expansion is deferred (see
                        # ``project_megamoe_m25_plan.md`` §7.2).
                        #
                        # NOTE (the fused path): the FC1 ``alpha_val`` defined in
                        # the ``if not do_phase_l2:`` branch above does NOT
                        # leak into this combine body scope — cuteDSL's
                        # if/else traces both branches but ``alpha_val`` is
                        # only bound in the FC1 trace, so dereferencing it
                        # here produced ``TypeError: None * Float32``. The
                        # FC2 combine path only needs ``alpha_val_l2``
                        # (applied at the final acc_vec scale below); drop
                        # the stale ``alpha_val * token_scale`` fold.
                        alpha_val_l2 = alpha_l2_tuple[0][expert_idx - self.b_tensor_l_offsets_l2[0]]
                        if is_valid_row:
                            alpha_val_l2 = alpha_val_l2 * token_scale

                        tTR_tAcc_l2 = tTR_tAcc_base_l2[
                            (None, None, None, None, None, acc_stage_index)
                        ]
                        acc_pipeline_l2.consumer_wait(acc_consumer_state_l2)
                        tTR_tAcc_l2 = cute.group_modes(tTR_tAcc_l2, 3, cute.rank(tTR_tAcc_l2))
                        subtile_cnt_l2 = cute.size(tTR_tAcc_l2.shape, mode=[3])
                        tTR_rC_l2 = cute.make_rmem_tensor(tTR_rAcc_l2.shape, self.l2_out_dtype)

                        for subtile_idx in cutlass.range(subtile_cnt_l2):
                            real_subtile_idx = subtile_idx
                            if cutlass.const_expr(self.overlapping_accum_l2):
                                if reverse_subtile:
                                    real_subtile_idx = subtile_cnt_l2 - 1 - subtile_idx
                            tTR_tAcc_l2_mn = tTR_tAcc_l2[(None, None, None, real_subtile_idx)]
                            cute.copy(tiled_copy_t2r_l2, tTR_tAcc_l2_mn, tTR_rAcc_l2)

                            if cutlass.const_expr(self.overlapping_accum_l2):
                                if subtile_idx == self.iter_acc_early_release_in_epilogue_l2:
                                    cute.arch.fence_view_async_tmem_load()
                                    with cute.arch.elect_one():
                                        acc_pipeline_l2.consumer_release(acc_consumer_state_l2)
                                    acc_consumer_state_l2.advance()

                            acc_vec_l2 = tTR_rAcc_l2.load()
                            acc_vec_final_l2 = alpha_val_l2 * acc_vec_l2
                            tTR_rC_l2.store(acc_vec_final_l2.to(self.l2_out_dtype))

                            if is_valid_row:
                                rOut_epi = cute.make_tensor(tTR_rC_l2.iterator, epi_layout_l2)
                                base_coord_n = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk_l2[
                                    1
                                ] + real_subtile_idx * cute.size(tTR_rC_l2)
                                if cutlass.const_expr(direct_combine_output):
                                    rows_per_rank = (
                                        combine_output_top_k
                                        * combine_output_max_num_tokens_per_rank
                                    )
                                    combine_output_rows = combine_output_ep_size * rows_per_rank
                                    if expanded_idx < combine_output_rows:
                                        output_rank = expanded_idx // rows_per_rank
                                        row_offset = expanded_idx - output_rank * rows_per_rank
                                        if cutlass.const_expr(direct_combine_token_major_output):
                                            output_token_idx = row_offset // combine_output_top_k
                                            output_topk_idx = (
                                                row_offset - output_token_idx * combine_output_top_k
                                            )
                                        else:
                                            output_topk_idx = (
                                                row_offset // combine_output_max_num_tokens_per_rank
                                            )
                                            output_token_idx = (
                                                row_offset
                                                - output_topk_idx
                                                * combine_output_max_num_tokens_per_rank
                                            )
                                        if cutlass.const_expr(direct_combine_atomic_output):
                                            atomic_token_row = expanded_idx // combine_output_top_k
                                            atomic_output_rank = (
                                                atomic_token_row
                                                // combine_output_max_num_tokens_per_rank
                                            )
                                            atomic_output_token_idx = (
                                                atomic_token_row
                                                - atomic_output_rank
                                                * combine_output_max_num_tokens_per_rank
                                            )
                                            scatter_out = cute.domain_offset(
                                                (
                                                    atomic_output_rank,
                                                    0,
                                                    atomic_output_token_idx,
                                                    0,
                                                ),
                                                out_tensor,
                                            )
                                            for index in cutlass.range(
                                                self.epi_loop_size_l2, unroll_full=True
                                            ):
                                                coord_n = (
                                                    base_coord_n + index * self.element_offset_l2
                                                )
                                                scatter_out_offset = cute.domain_offset(
                                                    (0, 0, 0, coord_n), scatter_out
                                                )
                                                rOut_epi_packed = rOut_epi[index, None, None]
                                                vectorized_atomic_add_bf16x8(
                                                    rOut_epi_packed, scatter_out_offset
                                                )
                                        else:
                                            if cutlass.const_expr(
                                                direct_combine_token_major_output
                                            ):
                                                scatter_out = cute.domain_offset(
                                                    (
                                                        output_rank,
                                                        output_token_idx,
                                                        output_topk_idx,
                                                        0,
                                                    ),
                                                    out_tensor,
                                                )
                                            else:
                                                scatter_out = cute.domain_offset(
                                                    (
                                                        output_rank,
                                                        output_topk_idx,
                                                        output_token_idx,
                                                        0,
                                                    ),
                                                    out_tensor,
                                                )
                                            for index in cutlass.range(
                                                self.epi_loop_size_l2, unroll_full=True
                                            ):
                                                coord_n = (
                                                    base_coord_n + index * self.element_offset_l2
                                                )
                                                scatter_out_offset = cute.domain_offset(
                                                    (0, 0, 0, coord_n), scatter_out
                                                )
                                                # Direct combine-buffer rows are unique
                                                # per (source-rank, top-k, token) route;
                                                # store is sufficient and avoids a
                                                # separate pre-clear kernel.
                                                rOut_epi_packed = rOut_epi[index, None, None]
                                                vectorized_store_bf16x8(
                                                    rOut_epi_packed, scatter_out_offset
                                                )
                                else:
                                    scatter_out = cute.domain_offset((token_idx, 0, 0), out_tensor)
                                    for index in cutlass.range(
                                        self.epi_loop_size_l2, unroll_full=True
                                    ):
                                        coord_n = base_coord_n + index * self.element_offset_l2
                                        scatter_out_offset = cute.domain_offset(
                                            (0, coord_n, 0), scatter_out
                                        )
                                        rOut_epi_packed = rOut_epi[index, None, None]
                                        vectorized_atomic_add_bf16x8(
                                            rOut_epi_packed, scatter_out_offset
                                        )

                        if cutlass.const_expr(not self.overlapping_accum_l2):
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline_l2.consumer_release(acc_consumer_state_l2)
                            acc_consumer_state_l2.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                # refresh ``tile_phase`` for the next
                # tile so the top of this while loop dispatches correctly
                # via ``do_phase_l2``. Mirrors the sibling MMA / LDGSTS-A
                # next-tile reads.
                if cutlass.const_expr(self.enable_linear2):
                    tile_phase = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        if cutlass.const_expr(monolithic_reduce_output):
            # All persistent CTAs must complete FC2 direct-buffer stores before
            # any CTA starts the M6 top-k reduction. This mirrors DeepGEMM's
            # device-side grid barrier but keeps the state in the existing
            # per-rank control words. Word 6 is the CTA arrival count and word 7
            # is the per-launch phase; the last arriving CTA publishes it after host-side control zeroing.
            fence_acq_rel_gpu()
            self.cta_sync_barrier.arrive_and_wait()
            if tidx == 0:
                grid_sync_count_ptr = cute.domain_offset(
                    (monolithic_local_rank, 6), monolithic_control
                ).iterator
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                target_phase = cutlass.Uint32(4)
                target_phase_u64 = cutlass.Uint64(4)
                if cutlass.const_expr(monolithic_direct_topk_materialize):
                    target_phase = cutlass.Uint32(7)
                    target_phase_u64 = cutlass.Uint64(7)
                old_count = atomic_add_release_sys_u32(grid_sync_count_ptr, cutlass.Uint32(1))
                if old_count + cutlass.Uint32(1) == cutlass.Uint32(
                    runtime_monolithic_grid_sync_blocks
                ):
                    st_release_sys_u64(grid_sync_count_ptr, cutlass.Uint64(0))
                    st_release_sys_u64(grid_sync_phase_ptr, target_phase_u64)
                observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                while observed_phase < target_phase:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
            self.cta_sync_barrier.arrive_and_wait()

            expected_epoch = cutlass.Uint64(1)
            if cutlass.const_expr(monolithic_direct_topk_stage_inputs):
                expected_epoch_ptr = cute.domain_offset(
                    (monolithic_local_rank, 1), monolithic_control
                ).iterator
                expected_epoch = ld_acquire_sys_u64(expected_epoch_ptr)

            # Publish this rank's FC2 stores only after the local grid barrier,
            # then wait for peer ranks entirely on device. Only block0 performs
            # the cross-rank polling and releases local CTAs through the phase
            # word; otherwise every CTA spins on the same peer flags.
            fence_acq_rel_sys()
            peer_ready_phase = cutlass.Uint32(8)
            peer_ready_phase_u64 = cutlass.Uint64(8)
            if tidx == 0:
                grid_sync_phase_ptr = cute.domain_offset(
                    (monolithic_local_rank, 7), monolithic_control
                ).iterator
                if monolithic_linear_block_idx == 0:
                    epoch_ptr = cute.domain_offset(
                        (monolithic_local_rank, 8), monolithic_control
                    ).iterator
                    flag_ptr = cute.domain_offset(
                        (monolithic_local_rank, 9), monolithic_control
                    ).iterator
                    st_release_sys_u64(epoch_ptr, expected_epoch)
                    st_release_sys_u64(flag_ptr, cutlass.Uint64(1))

                    for ready_rank in range(combine_output_ep_size):
                        peer_epoch_ptr = cute.domain_offset(
                            (ready_rank, 8), monolithic_control
                        ).iterator
                        peer_flag_ptr = cute.domain_offset(
                            (ready_rank, 9), monolithic_control
                        ).iterator
                        peer_epoch = cutlass.Uint64(0)
                        peer_flag = cutlass.Uint64(0)
                        while peer_epoch != expected_epoch or peer_flag != cutlass.Uint64(1):
                            peer_epoch = ld_acquire_sys_u64(peer_epoch_ptr)
                            peer_flag = ld_acquire_sys_u64(peer_flag_ptr)
                    st_release_sys_u64(grid_sync_phase_ptr, peer_ready_phase_u64)
                else:
                    observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)
                    while observed_phase < peer_ready_phase:
                        observed_phase = ld_acquire_sys_u32(grid_sync_phase_ptr)

            self.cta_sync_barrier.arrive_and_wait()

            linear_idx = monolithic_linear_block_idx * self.threads_per_cta + tidx
            total_output_elements = cutlass.Int32(
                combine_output_max_num_tokens_per_rank * monolithic_hidden_size
            )
            output_stride = (
                cutlass.Int32(self.threads_per_cta) * runtime_monolithic_grid_sync_blocks
            )
            while linear_idx < total_output_elements:
                token_idx = linear_idx // monolithic_hidden_size
                hidden_idx = linear_idx - token_idx * monolithic_hidden_size
                accum = cutlass.Float32(0.0)
                for topk_idx in range(combine_output_top_k):
                    accum = accum + out_tensor[
                        monolithic_local_rank, topk_idx, token_idx, hidden_idx
                    ].to(cutlass.Float32)
                monolithic_final_output[token_idx, hidden_idx, 0] = accum.to(cutlass.BFloat16)
                linear_idx = linear_idx + output_stride

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def epilog_tmem_copy_and_partition_linear2(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gOut_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Linear2 (Linear2/FC2) counterpart of ``epilog_tmem_copy_and_partition``.

        Ports the FC2 kernel's TMEM → register partition
        (see ``blockscaled_contiguous_grouped_gemm_finalize_fusion.py:2397``).

        Linear2 reads the accumulator as a single full-N register tile (no
        SwiGLU halving), so it returns a 3-tuple
        ``(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)`` instead of the 4-tuple used
        by the Linear1 helper above.

        This method is only reachable when ``self.enable_linear2`` is True.
        ``self.l2_output_layout`` / ``self.l2_out_dtype`` are seeded in
        ``__call__`` regardless of the flag, so attribute access here is
        always safe; the whole call site is gated by
        ``cutlass.const_expr(self.enable_linear2)``.

        :param tidx: Thread index in epilogue warp groups
        :param tAcc: Accumulator tensor to partition
        :param gOut_mnl: Global output tensor used for the destination
            partition shape (mega-kernel-side ``out`` tensor, BF16)
        :param epi_tile: Epilogue tiler
        :param use_2cta_instrs: Whether 2-CTA instructions are enabled

        :return: ``(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)``
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk_l2,
            self.l2_output_layout,
            self.l2_out_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gOut_mnl_epi = cute.flat_divide(gOut_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gOut = thr_copy_t2r.partition_D(gOut_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gOut[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
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
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
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
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid_from_shape(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size from an M/N/L shape."""
        m, n, problem_size_l = gemm_shape

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = problem_size_l

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
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
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
    ) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in (128, 256):
            is_valid = False
        # Skip invalid mma tile n
        # SwiGlu Fusion requires even epi_tile counts,
        # based on epi_tile_n = 64, only mma_tiler_n = 128 and 256 are supported
        if mma_tiler_mn[1] not in (128, 256):
            is_valid = False

        # Skip illegal cluster shape
        if (mma_tiler_mn[0] // cluster_shape_mn[0]) != 128:
            is_valid = False

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False

        # We only support cluster shape n = 1 for now
        # TODO: Support cluster shape n > 1
        if cluster_shape_mn[1] != 1:
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False

        # Skip unsupported layouts
        if not cls.is_valid_layouts(ab_dtype, c_dtype, a_major, b_major, c_major):
            can_implement = False

        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip unsupported A/B layout
        if not (a_major == "k" and b_major == "k"):
            can_implement = False
        return can_implement

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr_tuple: Tuple[cute.Pointer, ...],
        a_sf_ptr: cute.Pointer,
        b_sf_ptr_tuple: Tuple[cute.Pointer, ...],
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr_tuple: Tuple[cute.Pointer, ...],
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Unified wrapper supporting both single-B and multi-B tensors.

        B tensors are always passed as tuples (length 1 for single-B).
        L sizes are configured via b_tensor_l_sizes in __init__.
        """
        scale_k = k // scaling_vector_size
        interm_size = n // 2
        num_tiles = m // tile_size
        total_l = self.b_tensor_l_offsets[self.num_b_tensors]

        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        a_sf = cute.make_tensor(
            a_sf_ptr, layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
        )
        c_sf = cute.make_tensor(
            c_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, interm_size // (scaling_vector_size * 4), total_l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )

        # Create B and alpha tensors using const_expr conditions
        l_0 = self.b_tensor_l_sizes[0]
        alpha_0 = cute.make_tensor(alpha_ptr_tuple[0], layout=cute.make_layout((l_0,)))
        b_0 = cute.make_tensor(
            b_ptr_tuple[0], layout=cute.make_ordered_layout((n, k, l_0), order=(1, 0, 2))
        )
        b_sf_0 = cute.make_tensor(
            b_sf_ptr_tuple[0],
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l_0), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_tuple = [b_0]
        b_sf_tuple = [b_sf_0]
        alpha_tuple = [alpha_0]

        if cutlass.const_expr(self.num_b_tensors >= 2):
            l_1 = self.b_tensor_l_sizes[1]
            alpha_1 = cute.make_tensor(alpha_ptr_tuple[1], layout=cute.make_layout((l_1,)))
            b_1 = cute.make_tensor(
                b_ptr_tuple[1], layout=cute.make_ordered_layout((n, k, l_1), order=(1, 0, 2))
            )
            b_sf_1 = cute.make_tensor(
                b_sf_ptr_tuple[1],
                layout=cute.make_ordered_layout(
                    (32, 4, n // 128, 4, scale_k // 4, l_1), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            b_tuple.append(b_1)
            b_sf_tuple.append(b_sf_1)
            alpha_tuple.append(alpha_1)

        if cutlass.const_expr(self.num_b_tensors >= 3):
            l_2 = self.b_tensor_l_sizes[2]
            alpha_2 = cute.make_tensor(alpha_ptr_tuple[2], layout=cute.make_layout((l_2,)))
            b_2 = cute.make_tensor(
                b_ptr_tuple[2], layout=cute.make_ordered_layout((n, k, l_2), order=(1, 0, 2))
            )
            b_sf_2 = cute.make_tensor(
                b_sf_ptr_tuple[2],
                layout=cute.make_ordered_layout(
                    (32, 4, n // 128, 4, scale_k // 4, l_2), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            b_tuple.append(b_2)
            b_sf_tuple.append(b_sf_2)
            alpha_tuple.append(alpha_2)

        if cutlass.const_expr(self.num_b_tensors >= 4):
            l_3 = self.b_tensor_l_sizes[3]
            alpha_3 = cute.make_tensor(alpha_ptr_tuple[3], layout=cute.make_layout((l_3,)))
            b_3 = cute.make_tensor(
                b_ptr_tuple[3], layout=cute.make_ordered_layout((n, k, l_3), order=(1, 0, 2))
            )
            b_sf_3 = cute.make_tensor(
                b_sf_ptr_tuple[3],
                layout=cute.make_ordered_layout(
                    (32, 4, n // 128, 4, scale_k // 4, l_3), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            b_tuple.append(b_3)
            b_sf_tuple.append(b_sf_3)
            alpha_tuple.append(alpha_3)

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
        global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))

        return self(
            a,
            tuple(b_tuple),
            c,
            a_sf,
            tuple(b_sf_tuple),
            c_sf,
            global_sf,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            tuple(alpha_tuple),
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )

    @cute.jit
    def wrapper_fused(
        self,
        # --- FC1 pointers (same layout as :meth:`wrapper`) ---
        a_ptr: cute.Pointer,
        a_byte_ptr: cute.Pointer,
        b_l1_ptr_tuple: Tuple[cute.Pointer, ...],
        a_sf_ptr: cute.Pointer,
        b_sf_l1_ptr_tuple: Tuple[cute.Pointer, ...],
        alpha_l1_ptr_tuple: Tuple[cute.Pointer, ...],
        # --- FC2 pointers (new) ---
        b_l2_ptr_tuple: Tuple[cute.Pointer, ...],
        b_sf_l2_ptr_tuple: Tuple[cute.Pointer, ...],
        alpha_l2_ptr_tuple: Tuple[cute.Pointer, ...],
        # --- Pool + final output (new) ---
        pool_ptr: cute.Pointer,
        pool_sf_ptr: cute.Pointer,
        l2_arrival_mask_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        monolithic_final_output_ptr: cute.Pointer,
        monolithic_control_ptr: cute.Pointer,
        # --- Routing + metadata ---
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        # --- Optional monolithic direct-topk materialization inputs ---
        monolithic_direct_topk_input_ptr: cute.Pointer,
        monolithic_direct_topk_input_fp4_ptr: cute.Pointer,
        monolithic_direct_topk_input_scale_ptr: cute.Pointer,
        monolithic_direct_topk_idx_ptr: cute.Pointer,
        monolithic_direct_topk_scales_ptr: cute.Pointer,
        monolithic_direct_topk_token_counts_ptr: cute.Pointer,
        monolithic_direct_topk_local_input_ptr: cute.Pointer,
        monolithic_direct_topk_local_input_scale_ptr: cute.Pointer,
        monolithic_direct_topk_local_idx_ptr: cute.Pointer,
        monolithic_direct_topk_local_scales_ptr: cute.Pointer,
        # --- Shapes ---
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n_l1: cutlass.Int64,
        n_l2: cutlass.Int64,
        k: cutlass.Int64,
        monolithic_local_tokens: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        direct_combine_output: cutlass.Constexpr = False,
        direct_combine_atomic_output: cutlass.Constexpr = False,
        direct_combine_token_major_output: cutlass.Constexpr = False,
        combine_output_ep_size: cutlass.Constexpr = 1,
        combine_output_top_k: cutlass.Constexpr = 1,
        combine_output_max_num_tokens_per_rank: cutlass.Constexpr = 0,
        combine_output_rank_stride_elements: cutlass.Constexpr = 0,
        combine_output_hidden_size: cutlass.Constexpr = 0,
        monolithic_reduce_output: cutlass.Constexpr = False,
        monolithic_control_rank_stride_elements: cutlass.Constexpr = 0,
        monolithic_local_rank: cutlass.Constexpr = 0,
        monolithic_grid_sync_blocks: cutlass.Constexpr = 0,
        monolithic_direct_topk_materialize: cutlass.Constexpr = False,
        monolithic_direct_topk_input_rank_stride_elements: cutlass.Constexpr = 0,
        monolithic_direct_topk_input_scale_rank_stride_elements: cutlass.Constexpr = 0,
        monolithic_direct_topk_idx_rank_stride_elements: cutlass.Constexpr = 0,
        monolithic_direct_topk_source_input: cutlass.Constexpr = False,
        monolithic_direct_topk_scales_rank_stride_elements: cutlass.Constexpr = 0,
        monolithic_direct_topk_local_expert_offset: cutlass.Constexpr = 0,
        monolithic_direct_topk_num_local_experts: cutlass.Constexpr = 0,
        monolithic_direct_topk_stage_inputs: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Fused FC1 + SwiGLU + FC2 + combine wrapper.

        Extension of :meth:`wrapper` that additionally plumbs the Linear2
        (FC2) weight / scale-factor / alpha tuples, the HBM activation +
        scale-factor pool used for the FC1->FC2 hand-off, the BF16 final
        output buffer, and the combine metadata (``permuted_idx_to_expanded
        _idx`` + ``token_final_scales``) into :meth:`__call__`. The kernel
        must be instantiated with ``enable_linear2=True`` for these
        arguments to be consumed; otherwise the Linear2 device code stays
        DCE'd and the fused arguments are ignored.

        Shape conventions:
            * ``n_l1`` is the FC1 output N before SwiGLU (``2 * interm_size``)
            * ``n_l2`` is the FC2 output N (= ``hidden_size``)
            * ``k`` is the FC1 input K (= ``hidden_size``)
            * FC2 consumes ``interm_size = n_l1 // 2`` on its K axis
            * Pool layout mirrors FC1 C (``(m, interm_size, 1)``, FP4)
            * Pool-SF layout mirrors FC1 C-SF
            * Final output layout is ``(orig_m, n_l2, 1)`` BF16 by default
            * Direct combine output layout is rank-strided
              ``(ep_size, top_k, max_tokens, n_l2)`` BF16
            * ``token_final_scales`` is indexed as ``(token_idx, topk_idx)``
        """
        scale_k = k // scaling_vector_size
        interm_size = n_l1 // 2
        scale_k_l2 = interm_size // scaling_vector_size
        num_tiles = m // tile_size
        total_l_l1 = self.b_tensor_l_offsets[self.num_b_tensors]

        # --- A / A_SF (FC1 inputs) ---
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        a_bytes = cute.make_tensor(
            a_byte_ptr,
            layout=cute.make_ordered_layout((orig_m, k // 2, 1), order=(1, 0, 2)),
        )
        a_sf = cute.make_tensor(
            a_sf_ptr, layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2))
        )

        # --- Pool activation + pool-SF (FC1 output / FC2 input) ---
        # The pool shares FC1 C's logical shape (m, interm_size, 1). In
        # fused mode ``c`` is effectively replaced by ``pool_tensor`` at
        # the epilogue TMA store path, so we thread the pool
        # tensor in as both the ``c`` placeholder (needed for SFC layout
        # derivation in ``__call__``) and the explicit ``pool_tensor``.
        pool_tensor = cute.make_tensor(
            pool_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
        )
        pool_sfc_tensor = cute.make_tensor(
            pool_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, interm_size // (scaling_vector_size * 4), total_l_l1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        l2_arrival_mask = cute.make_tensor(
            l2_arrival_mask_ptr, layout=cute.make_layout((num_tiles,))
        )

        # --- FC1 weights (B / SFB / alpha) — same fan-out as wrapper() ---
        l_0 = self.b_tensor_l_sizes[0]
        alpha_l1_0 = cute.make_tensor(alpha_l1_ptr_tuple[0], layout=cute.make_layout((l_0,)))
        b_l1_0 = cute.make_tensor(
            b_l1_ptr_tuple[0], layout=cute.make_ordered_layout((n_l1, k, l_0), order=(1, 0, 2))
        )
        b_sf_l1_0 = cute.make_tensor(
            b_sf_l1_ptr_tuple[0],
            layout=cute.make_ordered_layout(
                (32, 4, n_l1 // 128, 4, scale_k // 4, l_0), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_l1_tuple = [b_l1_0]
        b_sf_l1_tuple = [b_sf_l1_0]
        alpha_l1_tuple = [alpha_l1_0]

        if cutlass.const_expr(self.num_b_tensors >= 2):
            l_1 = self.b_tensor_l_sizes[1]
            alpha_l1_1 = cute.make_tensor(alpha_l1_ptr_tuple[1], layout=cute.make_layout((l_1,)))
            b_l1_1 = cute.make_tensor(
                b_l1_ptr_tuple[1],
                layout=cute.make_ordered_layout((n_l1, k, l_1), order=(1, 0, 2)),
            )
            b_sf_l1_1 = cute.make_tensor(
                b_sf_l1_ptr_tuple[1],
                layout=cute.make_ordered_layout(
                    (32, 4, n_l1 // 128, 4, scale_k // 4, l_1), order=(2, 1, 4, 0, 3, 5)
                ),
            )
            b_l1_tuple.append(b_l1_1)
            b_sf_l1_tuple.append(b_sf_l1_1)
            alpha_l1_tuple.append(alpha_l1_1)

        # --- FC2 weights (B_l2 / SFB_l2 / alpha_l2) ---
        # Shape: (n_l2, interm_size, L) with SF scale_k_l2 = interm_size/SV.
        # The current implementation assumes a single FC2 tuple (``num_b_tensors_l2 == 1``);
        # multi-tuple expansion is deferred.
        l_l2_0 = (
            self.b_tensor_l_sizes_l2[0]
            if self.b_tensor_l_sizes_l2 is not None
            else self.b_tensor_l_sizes[0]
        )
        alpha_l2_0 = cute.make_tensor(alpha_l2_ptr_tuple[0], layout=cute.make_layout((l_l2_0,)))
        b_l2_0 = cute.make_tensor(
            b_l2_ptr_tuple[0],
            layout=cute.make_ordered_layout((n_l2, interm_size, l_l2_0), order=(1, 0, 2)),
        )
        b_sf_l2_0 = cute.make_tensor(
            b_sf_l2_ptr_tuple[0],
            layout=cute.make_ordered_layout(
                (32, 4, n_l2 // 128, 4, scale_k_l2 // 4, l_l2_0), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_l2_tuple = [b_l2_0]
        b_sf_l2_tuple = [b_sf_l2_0]
        alpha_l2_tuple = [alpha_l2_0]

        # --- Final BF16 output (combine destination) ---
        out = cute.make_tensor(
            out_ptr, layout=cute.make_ordered_layout((orig_m, n_l2, 1), order=(1, 0, 2))
        )
        monolithic_final_output = cute.make_tensor(
            monolithic_final_output_ptr,
            layout=cute.make_ordered_layout(
                (combine_output_max_num_tokens_per_rank, n_l2, 1), order=(1, 0, 2)
            ),
        )
        monolithic_control = cute.make_tensor(
            monolithic_control_ptr,
            layout=cute.make_layout(
                (combine_output_ep_size, 64),
                stride=(monolithic_control_rank_stride_elements, 1),
            ),
        )
        scatter_out = out
        if cutlass.const_expr(direct_combine_output):
            if cutlass.const_expr(direct_combine_token_major_output):
                scatter_out = cute.make_tensor(
                    out_ptr,
                    layout=cute.make_layout(
                        (
                            combine_output_ep_size,
                            combine_output_max_num_tokens_per_rank,
                            combine_output_top_k,
                            combine_output_hidden_size,
                        ),
                        stride=(
                            combine_output_rank_stride_elements,
                            combine_output_top_k * combine_output_hidden_size,
                            combine_output_hidden_size,
                            1,
                        ),
                    ),
                )
            else:
                scatter_out = cute.make_tensor(
                    out_ptr,
                    layout=cute.make_layout(
                        (
                            combine_output_ep_size,
                            combine_output_top_k,
                            combine_output_max_num_tokens_per_rank,
                            combine_output_hidden_size,
                        ),
                        stride=(
                            combine_output_rank_stride_elements,
                            combine_output_max_num_tokens_per_rank * combine_output_hidden_size,
                            combine_output_hidden_size,
                            1,
                        ),
                    ),
                )

        # --- Routing + metadata ---
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
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((orig_m, self.topk), order=(1, 0)),
        )
        global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))

        direct_topk_max_tokens = combine_output_max_num_tokens_per_rank
        if cutlass.const_expr(not monolithic_direct_topk_materialize):
            direct_topk_max_tokens = 1
        direct_topk_input_rank_stride = monolithic_direct_topk_input_rank_stride_elements
        direct_topk_input_scale_rank_stride = (
            monolithic_direct_topk_input_scale_rank_stride_elements
        )
        direct_topk_idx_rank_stride = monolithic_direct_topk_idx_rank_stride_elements
        direct_topk_scales_rank_stride = monolithic_direct_topk_scales_rank_stride_elements
        if cutlass.const_expr(not monolithic_direct_topk_materialize):
            direct_topk_input_rank_stride = direct_topk_max_tokens * (k // 2)
            direct_topk_input_scale_rank_stride = direct_topk_max_tokens * scale_k
            direct_topk_idx_rank_stride = direct_topk_max_tokens * combine_output_top_k
            direct_topk_scales_rank_stride = direct_topk_max_tokens * combine_output_top_k
        monolithic_direct_topk_input = cute.make_tensor(
            monolithic_direct_topk_input_ptr,
            layout=cute.make_layout(
                (combine_output_ep_size, direct_topk_max_tokens, k // 2, 1),
                stride=(direct_topk_input_rank_stride, k // 2, 1, 1),
            ),
        )
        monolithic_direct_topk_input_fp4 = cute.make_tensor(
            monolithic_direct_topk_input_fp4_ptr,
            layout=cute.make_layout(
                (combine_output_ep_size, direct_topk_max_tokens, k, 1),
                stride=(direct_topk_input_rank_stride * 2, k, 1, 1),
            ),
        )
        monolithic_direct_topk_input_scale = cute.make_tensor(
            monolithic_direct_topk_input_scale_ptr,
            layout=cute.make_layout(
                (combine_output_ep_size, direct_topk_max_tokens, scale_k, 1),
                stride=(direct_topk_input_scale_rank_stride, scale_k, 1, 1),
            ),
        )
        monolithic_direct_topk_idx = cute.make_tensor(
            monolithic_direct_topk_idx_ptr,
            layout=cute.make_layout(
                (combine_output_ep_size, direct_topk_max_tokens, combine_output_top_k),
                stride=(direct_topk_idx_rank_stride, combine_output_top_k, 1),
            ),
        )
        monolithic_direct_topk_scales = cute.make_tensor(
            monolithic_direct_topk_scales_ptr,
            layout=cute.make_layout(
                (combine_output_ep_size, direct_topk_max_tokens, combine_output_top_k),
                stride=(direct_topk_scales_rank_stride, combine_output_top_k, 1),
            ),
        )
        monolithic_direct_topk_token_counts = cute.make_tensor(
            monolithic_direct_topk_token_counts_ptr,
            layout=cute.make_layout((combine_output_ep_size,)),
        )
        monolithic_local_layout_tokens = 1
        if cutlass.const_expr(monolithic_direct_topk_stage_inputs):
            monolithic_local_layout_tokens = combine_output_max_num_tokens_per_rank
        monolithic_direct_topk_local_input = cute.make_tensor(
            monolithic_direct_topk_local_input_ptr,
            layout=cute.make_layout(
                (monolithic_local_layout_tokens, k // 2, 1), stride=(k // 2, 1, 1)
            ),
        )
        monolithic_direct_topk_local_input_scale = cute.make_tensor(
            monolithic_direct_topk_local_input_scale_ptr,
            layout=cute.make_layout(
                (monolithic_local_layout_tokens, scale_k, 1), stride=(scale_k, 1, 1)
            ),
        )
        monolithic_direct_topk_local_idx = cute.make_tensor(
            monolithic_direct_topk_local_idx_ptr,
            layout=cute.make_layout(
                (monolithic_local_layout_tokens, combine_output_top_k),
                stride=(combine_output_top_k, 1),
            ),
        )
        monolithic_direct_topk_local_scales = cute.make_tensor(
            monolithic_direct_topk_local_scales_ptr,
            layout=cute.make_layout(
                (monolithic_local_layout_tokens, combine_output_top_k),
                stride=(combine_output_top_k, 1),
            ),
        )

        return self(
            a,
            tuple(b_l1_tuple),
            # In fused mode the kernel writes FC1 output to ``pool_tensor``
            # rather than a standalone ``c``; thread the pool in as the
            # ``c`` placeholder so ``__call__`` can derive the SFC layout
            # from ``c.shape`` (same logical shape) and no separate ``c``
            # buffer needs to be allocated by the caller.
            pool_tensor,
            a_sf,
            tuple(b_sf_l1_tuple),
            # ``sfc_tensor=None`` forces the s3a gate to pick up
            # ``pool_sfc_tensor`` as the SFC destination.
            None,
            global_sf,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            tuple(alpha_l1_tuple),
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
            a_bytes=a_bytes,
            b_l2=tuple(b_l2_tuple),
            sfb_l2=tuple(b_sf_l2_tuple),
            out=out,
            scatter_out=scatter_out,
            alpha_l2=tuple(alpha_l2_tuple),
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            pool_tensor=pool_tensor,
            pool_sfc_tensor=pool_sfc_tensor,
            l2_arrival_mask=l2_arrival_mask,
            direct_combine_output=direct_combine_output,
            direct_combine_atomic_output=direct_combine_atomic_output,
            direct_combine_token_major_output=direct_combine_token_major_output,
            combine_output_ep_size=combine_output_ep_size,
            combine_output_top_k=combine_output_top_k,
            combine_output_max_num_tokens_per_rank=combine_output_max_num_tokens_per_rank,
            combine_output_hidden_size=combine_output_hidden_size,
            monolithic_reduce_output=monolithic_reduce_output,
            monolithic_final_output=monolithic_final_output,
            monolithic_control=monolithic_control,
            monolithic_local_rank=monolithic_local_rank,
            monolithic_local_tokens=monolithic_local_tokens,
            monolithic_grid_sync_blocks=monolithic_grid_sync_blocks,
            monolithic_direct_topk_materialize=monolithic_direct_topk_materialize,
            monolithic_direct_topk_source_input=monolithic_direct_topk_source_input,
            monolithic_direct_topk_input=monolithic_direct_topk_input,
            monolithic_direct_topk_input_fp4=monolithic_direct_topk_input_fp4,
            monolithic_direct_topk_input_scale=monolithic_direct_topk_input_scale,
            monolithic_direct_topk_input_rank_stride_fp4=direct_topk_input_rank_stride * 2,
            monolithic_direct_topk_input_scale_rank_stride_elements=(
                direct_topk_input_scale_rank_stride
            ),
            monolithic_direct_topk_idx=monolithic_direct_topk_idx,
            monolithic_direct_topk_scales=monolithic_direct_topk_scales,
            monolithic_direct_topk_token_counts=monolithic_direct_topk_token_counts,
            monolithic_direct_topk_local_input=monolithic_direct_topk_local_input,
            monolithic_direct_topk_local_input_scale=monolithic_direct_topk_local_input_scale,
            monolithic_direct_topk_local_idx=monolithic_direct_topk_local_idx,
            monolithic_direct_topk_local_scales=monolithic_direct_topk_local_scales,
            monolithic_direct_topk_local_expert_offset=monolithic_direct_topk_local_expert_offset,
            monolithic_direct_topk_num_local_experts=monolithic_direct_topk_num_local_experts,
            monolithic_direct_topk_stage_inputs=monolithic_direct_topk_stage_inputs,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
    sf_swizzled_tensor: cute.Tensor,
    sf_unswizzled_tensor: cute.Tensor,
):
    """Convert scale factor tensor from mma specification M(32x4xrest_m)xK(4xrest_k)xL layout to MKL layout"""
    # sf_swizzled_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 0, 3)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_unswizzled_tensor)):
        mkl_coord = sf_unswizzled_tensor.layout.get_hier_coord(i)
        sf_unswizzled_tensor[mkl_coord] = sf_swizzled_tensor[mkl_coord]
