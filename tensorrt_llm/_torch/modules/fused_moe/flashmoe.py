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
"""FlashMoE: Fully-fused FC1+FC2+AllGather+AllReduce in a single persistent kernel.

This standalone module replaces the separate AllGather -> FC1 -> FC2 -> ReduceScatter
pipeline with a single persistent cuteDSL kernel on Blackwell that:
  1. Gathers hidden-state data from remote ranks via MNNVL fabric memory.
  2. Computes FC1 (GEMM + SwiGLU + NVFP4 quant) per expert group.
  3. Computes FC2 (GEMM + scatter-add) per expert group.
  4. Optionally performs in-kernel AllReduce across EP ranks via IPC.

Pre-kernel Python orchestration:
  1. Local routing -> AllGather ONLY small routing tensors (~2 MB).
  2. moe_sort with global routing -> token_id_mapping with global token indices.
  3. Quantize local input to NVFP4 -> gather via MNNVL fabric memory.
  4. Launch fused FC1+FC2 kernel (with optional in-kernel AllReduce).

The IPC gather for FC1 input transfers only ~14 MB of NVFP4 data via NVLink
fabric memory instead of ~112 MB of bf16 via NCCL AllGather.
"""

from typing import Optional

import torch

from tensorrt_llm.mapping import Mapping


class FlashMoE(torch.nn.Module):
    """Standalone fully-fused MoE for Blackwell Expert Parallelism.

    Orchestrates the pre-kernel flow (small AllGather + moe_sort + IPC write)
    and launches FC1 + FC2 with IPC-gathered input.

    Two execution modes:
    - Fused (use_fused_kernel=True): FC1+FC2 in a single persistent CuTe DSL
      kernel (FlashMoeFusedKernel). When EP > 1 and use_ipc=True, includes
      in-kernel AllReduce via IPC staging buffers.
    - Decomposed (use_fused_kernel=False): Separate FC1 and FC2 kernel launches
      using existing CuTe DSL kernels, with NCCL AllReduce. Useful as a
      reference and fallback.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MoE intermediate (FFN) dimension.
        num_experts: Total number of experts across all ranks.
        top_k: Number of experts selected per token.
        mapping: Parallelism mapping (EP rank/size, TP, etc.).
        max_num_tokens: Maximum tokens per rank for IPC buffer sizing.
        sf_vec_size: Scale-factor vector size for NVFP4 (default 16).
        use_fused_kernel: If True, use the single fused FC1+FC2 persistent kernel.
            If False, use decomposed FC1 + FC2 kernel launches (default).
        use_ipc: If True, use MNNVL IPC memory for cross-rank data transfer.
            If False, use torch.distributed.all_gather instead (for testing
            without MNNVL/MPI infrastructure). Default True.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        mapping: Mapping,
        max_num_tokens: int = 8192,
        sf_vec_size: int = 16,
        use_fused_kernel: bool = False,
        use_ipc: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.mapping = mapping
        self.max_num_tokens = max_num_tokens
        self.sf_vec_size = sf_vec_size
        self.use_fused_kernel = use_fused_kernel
        self.use_ipc = use_ipc

        self.ep_size = mapping.moe_ep_size
        self.ep_rank = mapping.moe_ep_rank

        assert num_experts % self.ep_size == 0
        self.experts_per_rank = num_experts // self.ep_size
        self.local_expert_offset = self.ep_rank * self.experts_per_rank

        # Tile size for moe_sort (matches existing CuteDSL kernel)
        self.tile_size = 128

        # Number of expert groups = number of local experts
        self.num_expert_groups = self.experts_per_rank

        # Lazily initialized IPC memory
        self._ipc_memory = None

        # Weight buffers (to be loaded externally)
        # NVFP4: N dimension NOT packed, K dimension packed (2 FP4 per byte)
        # w3w1_weight: [num_local_experts, 2*intermediate_size, hidden_size/2]
        # w2_weight: [num_local_experts, hidden_size, intermediate_size/2]
        self.w3w1_weight: Optional[torch.Tensor] = None
        self.w2_weight: Optional[torch.Tensor] = None

        # Quantization scales (to be loaded externally)
        self.fc1_weight_scale: Optional[torch.Tensor] = None
        self.fc2_weight_scale: Optional[torch.Tensor] = None
        self.fc1_alpha: Optional[torch.Tensor] = None
        self.fc2_alpha: Optional[torch.Tensor] = None
        self.fc31_input_scale: Optional[torch.Tensor] = None
        self.fc2_input_scale: Optional[torch.Tensor] = None

    @property
    def ipc_memory(self):
        """Lazily allocate IPC memory on first use."""
        if self._ipc_memory is None:
            from tensorrt_llm._mnnvl_utils import FlashMoeMnnvlMemory, MnnvlMemory

            MnnvlMemory.initialize()
            self._ipc_memory = FlashMoeMnnvlMemory(
                mapping=self.mapping,
                # Input buffers: per-rank (each rank writes its local tokens)
                max_input_tokens=self.max_num_tokens,
                # Output/staging: global (FC2 scatter-adds to all token positions)
                max_output_tokens=self.max_num_tokens * self.ep_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_expert_groups=self.num_expert_groups,
                sf_vec_size=self.sf_vec_size,
            )
        return self._ipc_memory

    def _quantize_input(self, x: torch.Tensor):
        """Quantize BF16 input to NVFP4 + scale factors.

        Returns:
            (x_nvfp4, x_sf): Quantized tensor [num_tokens, K/2] and
                scale factors [num_tokens, K/sf_vec_size] (both uint8).
        """
        num_tokens = x.shape[0]
        x_nvfp4, x_sf = torch.ops.trtllm.fp4_quantize(
            x,
            self.fc31_input_scale,
            self.sf_vec_size,
            False,  # sf_use_ue8m0
            False,  # is_sf_swizzled_layout
        )
        # fp4_quantize returns x_sf as a flat 1D tensor; reshape to 2D
        # so it can be correctly sliced per-token in write_input().
        x_sf = x_sf.view(num_tokens, -1)
        return x_nvfp4, x_sf

    def _allgather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """AllGather a tensor across EP ranks using torch.distributed."""
        if self.ep_size <= 1:
            return t
        gathered = [torch.empty_like(t) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered, t)
        return torch.cat(gathered, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        routing_method,
    ) -> torch.Tensor:
        """Execute fully-fused FlashMoE.

        Args:
            x: Input hidden states [local_tokens, hidden_size] (bf16).
            router_logits: Router output [local_tokens, num_experts].
            routing_method: Routing method with .apply() method.

        Returns:
            output: [local_tokens, hidden_size] (bf16), AllReduced across EP ranks.
        """
        local_num_tokens = x.shape[0]

        # --- Step 1: Local routing ---
        token_selected_experts, token_final_scales = routing_method.apply(router_logits)

        # --- Step 2: AllGather ONLY routing tensors (small, ~2 MB) ---
        if self.ep_size > 1:
            if self.use_ipc:
                from ...distributed import allgather

                token_selected_experts = allgather(token_selected_experts, self.mapping, dim=0)
                token_final_scales = allgather(token_final_scales, self.mapping, dim=0)
            else:
                token_selected_experts = self._allgather_tensor(token_selected_experts)
                token_final_scales = self._allgather_tensor(token_final_scales)

        global_num_tokens = token_selected_experts.shape[0]

        # --- Step 3: moe_sort -> global tile schedule ---
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_experts,
            top_k=self.top_k,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.experts_per_rank,
            tile_tokens_dim=self.tile_size,
        )

        # --- Step 4: Quantize input + gather ---
        x_nvfp4, x_sf = self._quantize_input(x)

        if self.use_ipc:
            # IPC path: write to MNNVL fabric memory, gather via IPC pointers
            ipc_mem = self.ipc_memory
            ipc_mem.write_input(x_nvfp4, x_sf)
            ipc_mem.barrier()  # cross-rank sync
            gathered_a = ipc_mem.gather_all_input_a(local_num_tokens)
            gathered_sfa = ipc_mem.gather_all_input_sfa(local_num_tokens)
        else:
            # Non-IPC path: use torch.distributed.all_gather for quantized data
            gathered_a = self._allgather_tensor(x_nvfp4)
            gathered_sfa = self._allgather_tensor(x_sf)

        # --- Step 5: Launch kernel ---
        launch_fn = self._launch_fused if self.use_fused_kernel else self._launch_decomposed
        output = launch_fn(
            gathered_a=gathered_a,
            gathered_sfa=gathered_sfa,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            global_num_tokens=global_num_tokens,
            local_num_tokens=local_num_tokens,
        )

        # Output is AllReduced; slice to local tokens for this rank
        local_start = self.ep_rank * local_num_tokens
        return output[local_start : local_start + local_num_tokens]

    def _launch_decomposed(
        self,
        gathered_a: torch.Tensor,
        gathered_sfa: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        global_num_tokens: int,
        local_num_tokens: int,
    ) -> torch.Tensor:
        """Decomposed execution: FC1 + FC2 + AllReduce.

        Receives pre-gathered NVFP4 input (via IPC or torch.distributed) and
        runs FC1 + FC2 CuTe DSL kernels followed by AllReduce.

        Returns:
            output: [global_num_tokens, hidden_size] bf16 tensor with
                    AllReduced results.
        """
        device = self.w3w1_weight.device

        # ---------------------------------------------------------------
        # FC1: Gathered GEMM + SwiGLU + NVFP4 quant
        # ---------------------------------------------------------------
        fc1_out, fc1_out_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell(
            input=gathered_a.view(torch.float4_e2m1fn_x2),
            weight=self.w3w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=gathered_sfa.view(torch.uint8),
            weight_scale=self.fc1_weight_scale.view(torch.uint8),
            alpha=self.fc1_alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=self.fc2_input_scale,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.experts_per_rank,
            local_expert_offset=self.local_expert_offset,
            tile_size=self.tile_size,
        )

        # ---------------------------------------------------------------
        # FC2: GEMM + fused finalize (scatter + scale + atomic add)
        # ---------------------------------------------------------------
        # Output sized for global tokens since FC2 scatter-adds to token
        # positions across the full global range.
        output = torch.zeros(
            global_num_tokens, self.hidden_size, dtype=torch.bfloat16, device=device
        )

        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
            input=fc1_out.view(torch.float4_e2m1fn_x2),
            weight=self.w2_weight.view(torch.float4_e2m1fn_x2),
            input_scale=fc1_out_sf.view(torch.uint8),
            weight_scale=self.fc2_weight_scale.view(torch.uint8),
            alpha=self.fc2_alpha,
            output=output,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.experts_per_rank,
            local_expert_offset=self.local_expert_offset,
            tile_size=self.tile_size,
            output_dtype=torch.bfloat16,
        )

        # ---------------------------------------------------------------
        # AllReduce across EP ranks
        # ---------------------------------------------------------------
        # Each rank computed partial results from its local experts.
        # AllReduce sums contributions across all EP ranks.
        # In the fused kernel, this is done in-kernel by warps 4-7 reading
        # from IPC staging buffers and reducing directly.
        if self.ep_size > 1:
            torch.distributed.all_reduce(output, op=torch.distributed.ReduceOp.SUM)

        return output

    def _launch_fused(
        self,
        gathered_a: torch.Tensor,
        gathered_sfa: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        global_num_tokens: int,
        local_num_tokens: int,
    ) -> torch.Tensor:
        """Fused execution: single kernel for FC1 + FC2 [+ in-kernel AllReduce].

        Uses FlashMoeFusedKernel to execute both FC1 (GEMM+SwiGLU+quant) and
        FC2 (GEMM+scatter-add) in a single persistent kernel launch.
        When EP > 1 and IPC is available, the kernel also performs in-kernel
        AllReduce via IPC staging buffers, eliminating the need for NCCL.

        Returns:
            output: [global_num_tokens, hidden_size] bf16 tensor with
                    AllReduced results.
        """
        import cutlass
        import cutlass.cute as cute

        try:
            from cuda.bindings import driver as cuda
        except ImportError:
            from cuda import cuda

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.flashmoe_fused_kernel import (
            FlashMoeFusedKernel,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

        device = self.w3w1_weight.device
        enable_ar = self.ep_size > 1 and self.use_ipc

        # Total padded M dimension (from permuted_idx_to_expanded_idx)
        m = permuted_idx_to_expanded_idx.shape[0]
        orig_m = gathered_a.shape[0]

        # FC1 dimensions
        k1 = self.hidden_size  # FC1 K = hidden_size
        fc1_n = 2 * self.intermediate_size  # w3w1 combined
        intermediate_sz = self.intermediate_size

        # FC2 dimensions
        k2 = self.intermediate_size  # FC2 K = intermediate_size
        fc2_n = self.hidden_size  # w2 output = hidden_size

        # Allocate FC1 output buffer (NVFP4)
        fc1_c = torch.empty(m, intermediate_sz // 2, dtype=gathered_a.dtype, device=device)
        fc1_c_sf = torch.empty(
            m * intermediate_sz // self.sf_vec_size,
            dtype=gathered_sfa.dtype,
            device=device,
        )

        # FC2 output buffer: when AR is enabled, FC2 writes to IPC staging
        # (so other ranks can read via IPC), and AR writes to a separate output.
        # When AR is disabled, FC2 writes directly to the output buffer.
        if enable_ar:
            ipc_mem = self.ipc_memory
            # Reset AR synchronization state and staging buffer
            ipc_mem.reset_ar_state()
            staging = ipc_mem.get_local_staging()
            staging[:global_num_tokens].zero_()  # FC2 scatter-adds to this
            # AR output buffer (not necessarily IPC, just local GMEM)
            output = torch.empty(
                global_num_tokens,
                self.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            output = torch.zeros(
                global_num_tokens,
                self.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )

        # Build CuTe pointers
        fc1_a_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            gathered_a.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        fc1_b_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            self.w3w1_weight.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        fc1_c_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            fc1_c.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        fc1_sfa_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            gathered_sfa.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        fc1_sfb_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            self.fc1_weight_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        fc1_sfc_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            fc1_c_sf.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        fc1_norm_const_ptr = make_ptr(
            cutlass.Float32,
            self.fc2_input_scale.data_ptr(),
            cute.AddressSpace.gmem,
        )
        fc1_alpha_ptr = make_ptr(
            cutlass.Float32,
            self.fc1_alpha.data_ptr(),
            cute.AddressSpace.gmem,
        )
        # FC2 A = FC1 output
        fc2_a_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            fc1_c.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        fc2_b_ptr = make_ptr(
            cutlass.Float4E2M1FN,
            self.w2_weight.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        # FC2 writes to staging (IPC) when AR is enabled, else to output
        fc2_out_data_ptr = staging.data_ptr() if enable_ar else output.data_ptr()
        fc2_out_ptr = make_ptr(
            cutlass.BFloat16,
            fc2_out_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        fc2_sfa_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            fc1_c_sf.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        fc2_sfb_ptr = make_ptr(
            cutlass.Float8E4M3FN,
            self.fc2_weight_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        fc2_alpha_ptr = make_ptr(
            cutlass.Float32,
            self.fc2_alpha.data_ptr(),
            cute.AddressSpace.gmem,
        )
        tile_idx_to_expert_idx_ptr = make_ptr(
            cutlass.Int32,
            tile_idx_to_expert_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        tile_idx_to_mn_limit_ptr = make_ptr(
            cutlass.Int32,
            tile_idx_to_mn_limit.data_ptr(),
            cute.AddressSpace.gmem,
        )
        token_id_mapping_ptr = make_ptr(
            cutlass.Int32,
            permuted_idx_to_expanded_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        num_non_exiting_tiles_ptr = make_ptr(
            cutlass.Int32,
            num_non_exiting_tiles.data_ptr(),
            cute.AddressSpace.gmem,
        )
        permuted_idx_ptr = make_ptr(
            cutlass.Int32,
            permuted_idx_to_expanded_idx.data_ptr(),
            cute.AddressSpace.gmem,
        )
        token_final_scales_ptr = make_ptr(
            cutlass.Float32 if token_final_scales.dtype == torch.float32 else cutlass.BFloat16,
            token_final_scales.data_ptr(),
            cute.AddressSpace.gmem,
        )

        # Cross-CTA FC1→FC2 synchronization counter (GMEM, zeroed before each launch)
        fc1_done_counter = torch.zeros(1, dtype=torch.int32, device=device)
        fc1_done_counter_ptr = make_ptr(
            cutlass.Int32,
            fc1_done_counter.data_ptr(),
            cute.AddressSpace.gmem,
        )

        # Build AR pointers (only when AR is enabled)
        ar_staging_ipc_ptrs_ptr = None
        ar_output_ptr = None
        ar_cta_exit_counter_ptr = None
        ar_rank_ready_flag_ptrs_ptr = None
        ar_local_rank = None

        if enable_ar:
            # IPC pointers to all ranks' staging buffers
            staging_ipc_ptrs = ipc_mem.get_staging_ipc_ptrs()
            ar_staging_ipc_ptrs_ptr = make_ptr(
                cutlass.Int64,
                staging_ipc_ptrs.data_ptr(),
                cute.AddressSpace.gmem,
            )
            # AR output buffer
            ar_output_ptr = make_ptr(
                cutlass.BFloat16,
                output.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            )
            # CTA exit counter (1-element Int32 in MNNVL memory)
            cta_exit_counter = ipc_mem.get_cta_exit_counter()
            ar_cta_exit_counter_ptr = make_ptr(
                cutlass.Int32,
                cta_exit_counter.data_ptr(),
                cute.AddressSpace.gmem,
            )
            # IPC pointers to each rank's ready flag
            rank_ready_flag_ptrs = ipc_mem.get_rank_ready_flag_ipc_ptrs()
            ar_rank_ready_flag_ptrs_ptr = make_ptr(
                cutlass.Int64,
                rank_ready_flag_ptrs.data_ptr(),
                cute.AddressSpace.gmem,
            )
            ar_local_rank = self.ep_rank

        # Create or retrieve cached kernel
        if not hasattr(self, "_fused_kernel_cache"):
            self._fused_kernel_cache = {}

        cache_key = (self.sf_vec_size, self.tile_size, self.top_k, enable_ar)
        if cache_key not in self._fused_kernel_cache:
            kernel = FlashMoeFusedKernel(
                sf_vec_size=self.sf_vec_size,
                mma_tiler_mn=(self.tile_size, 128),
                cluster_shape_mn=(1, 1),
                vectorized_f32=True,
                topk=self.top_k,
                raster_along_m=False,
                enable_ar=enable_ar,
                ar_num_ranks=self.ep_size if enable_ar else 1,
            )
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(1)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            compiled = cute.compile(
                kernel.wrapper,
                fc1_a_ptr,
                fc1_b_ptr,
                fc1_c_ptr,
                fc1_sfa_ptr,
                fc1_sfb_ptr,
                fc1_sfc_ptr,
                fc1_norm_const_ptr,
                fc1_alpha_ptr,
                fc2_a_ptr,
                fc2_b_ptr,
                fc2_out_ptr,
                fc2_sfa_ptr,
                fc2_sfb_ptr,
                fc2_alpha_ptr,
                tile_idx_to_expert_idx_ptr,
                tile_idx_to_mn_limit_ptr,
                token_id_mapping_ptr,
                num_non_exiting_tiles_ptr,
                permuted_idx_ptr,
                token_final_scales_ptr,
                orig_m,
                m,
                fc1_n,
                fc2_n,
                k1,
                k2,
                self.experts_per_rank,
                global_num_tokens,
                self.top_k,
                fc1_done_counter_ptr=fc1_done_counter_ptr,
                ar_staging_ipc_ptrs_ptr=ar_staging_ipc_ptrs_ptr,
                ar_output_ptr=ar_output_ptr,
                ar_cta_exit_counter_ptr=ar_cta_exit_counter_ptr,
                ar_rank_ready_flag_ptrs_ptr=ar_rank_ready_flag_ptrs_ptr,
                ar_local_rank=ar_local_rank,
                scaling_vector_size=self.sf_vec_size,
                max_active_clusters=max_active_clusters,
                stream=stream,
            )
            self._fused_kernel_cache[cache_key] = compiled
        else:
            compiled = self._fused_kernel_cache[cache_key]

        # Cross-rank barrier to ensure AR state reset and staging zeroing
        # are visible before kernel launch
        if enable_ar:
            ipc_mem.barrier()

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        compiled(
            fc1_a_ptr,
            fc1_b_ptr,
            fc1_c_ptr,
            fc1_sfa_ptr,
            fc1_sfb_ptr,
            fc1_sfc_ptr,
            fc1_norm_const_ptr,
            fc1_alpha_ptr,
            fc2_a_ptr,
            fc2_b_ptr,
            fc2_out_ptr,
            fc2_sfa_ptr,
            fc2_sfb_ptr,
            fc2_alpha_ptr,
            tile_idx_to_expert_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            token_id_mapping_ptr,
            num_non_exiting_tiles_ptr,
            permuted_idx_ptr,
            token_final_scales_ptr,
            orig_m,
            m,
            fc1_n,
            fc2_n,
            k1,
            k2,
            self.experts_per_rank,
            global_num_tokens,
            self.top_k,
            fc1_done_counter_ptr=fc1_done_counter_ptr,
            ar_staging_ipc_ptrs_ptr=ar_staging_ipc_ptrs_ptr,
            ar_output_ptr=ar_output_ptr,
            ar_cta_exit_counter_ptr=ar_cta_exit_counter_ptr,
            ar_rank_ready_flag_ptrs_ptr=ar_rank_ready_flag_ptrs_ptr,
            ar_local_rank=ar_local_rank,
            stream=stream,
        )

        # AllReduce: when AR is enabled, the kernel already reduced the output.
        # When AR is disabled, fall back to NCCL AllReduce.
        if not enable_ar and self.ep_size > 1:
            torch.distributed.all_reduce(output, op=torch.distributed.ReduceOp.SUM)

        return output
