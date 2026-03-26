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
FlashMoE Fused Backend for TensorRT-LLM

Implements fused MoE backends inspired by the FlashMoE paper
(https://arxiv.org/html/2506.04667v3), fusing token dispatch, expert FFN
computation, and combine into a single logical operation.

Phase 1: FlashMoEFused - PyTorch + torch.compile implementation for bf16/fp16.
Phase 2: FlashMoECuteDsl - cuteDSL high-performance kernel for bf16 on SM90+.

Core algorithm:
1. Token dispatch: moe_sort -> tile-to-expert mapping + permutation indices
2. FC1: Gather + GroupedGEMM(gate_up) + SwiGLU activation
3. FC2: GroupedGEMM(down) + Scale + Scatter-Add to original positions
4. Multi-GPU: NVSHMEM kernel-level dispatch/combine (Phase 2 only)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# Ensure bf16 FlashMoE custom ops are registered
import tensorrt_llm._torch.custom_ops.flashmoe_bf16_custom_ops  # noqa: F401
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...model_config import ModelConfig
from ...utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from .interface import MoE, MoEWeightLoadingMode, _warn_and_return
from .routing import BaseMoeRoutingMethod

logger = logging.getLogger(__name__)

try:
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as torch_symm_mem

    _SYMM_MEM_AVAILABLE = True
except ImportError:
    _SYMM_MEM_AVAILABLE = False

try:
    from cuda.bindings import driver as cuda_driver  # noqa: F401

    _CUDA_DRIVER_AVAILABLE = True
except ImportError:
    _CUDA_DRIVER_AVAILABLE = False


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up."""
    return _silu(gate) * up


class FlashMoEFused(MoE):
    """
    FlashMoE Fused Backend: fuses dispatch + expert FFN + combine.

    Only supports unquantized bf16/fp16 (Phase 1).
    Uses per-expert loop with torch operations for FFN computation.
    """

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        FlashMoE only supports unquantized bf16/fp16.

        Args:
            quant_algo: Must be None (unquantized).
            dtype_activation: Must be float16 or bfloat16.
            swiglu_gptoss_style: Not supported.

        Returns:
            Tuple[bool, Optional[str]]: (can_implement, skip_reason)
        """
        if quant_algo is not None:
            return _warn_and_return(
                f"FlashMoEFused only supports unquantized mode (got quant_algo={quant_algo})"
            )

        if dtype_activation not in (torch.float16, torch.bfloat16):
            return _warn_and_return(
                f"FlashMoEFused only supports float16/bfloat16 (got {dtype_activation})"
            )

        if swiglu_gptoss_style:
            return _warn_and_return("FlashMoEFused does not support swiglu_gptoss_style")

        return True, None

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        bias: bool = False,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.moe_max_num_tokens = model_config.moe_max_num_tokens

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _supports_load_balancer(self) -> bool:
        """FlashMoE supports load balancer."""
        return True

    def create_weights(self):
        if self._weights_created:
            return

        weight_dtype = self.dtype

        # w3_w1_weight: fused gate_up projection [num_local_experts, 2*I_per_tp, H]
        w3_w1_weight_shape = (
            self.expert_size_per_partition,
            self.expand_intermediate_size_per_partition,
            self.hidden_size,
        )
        w3_w1_weight = nn.Parameter(
            torch.empty(w3_w1_weight_shape, dtype=weight_dtype),
            requires_grad=False,
        )
        self.register_parameter("w3_w1_weight", w3_w1_weight)

        # w2_weight: down projection [num_local_experts, H, I_per_tp]
        w2_weight_shape = (
            self.expert_size_per_partition,
            self.hidden_size,
            self.intermediate_size_per_partition,
        )
        w2_weight = nn.Parameter(
            torch.empty(w2_weight_shape, dtype=weight_dtype),
            requires_grad=False,
        )
        self.register_parameter("w2_weight", w2_weight)

        # Bias support
        if self.bias:
            w3_w1_bias = nn.Parameter(
                torch.empty(w3_w1_weight_shape[:2], dtype=weight_dtype),
                requires_grad=False,
            )
            self.register_parameter("w3_w1_bias", w3_w1_bias)

            w2_bias = nn.Parameter(
                torch.empty(w2_weight_shape[:2], dtype=weight_dtype),
                requires_grad=False,
            )
            self.register_parameter("w2_bias", w2_bias)
        else:
            self.w3_w1_bias = None
            self.w2_bias = None

        self.quant_scales = tuple()
        self._weights_created = True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        """Load weights using the same pattern as other backends."""
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        from .quantization import TensorParallelMode, load_weight_shard

        for local_slot_id, expert_id in enumerate(self.initial_local_expert_ids):
            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight = weights.get(f"{expert_id}.w1.weight")
                w3_weight = weights.get(f"{expert_id}.w3.weight")
                w2_weight = weights.get(f"{expert_id}.w2.weight")
            elif self.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_weight, w3_weight = None, None
                if "gate_up_proj" in weights:
                    w1_w3_weight = weights["gate_up_proj"][expert_id].transpose(0, 1)
                    w1_weight, w3_weight = w1_w3_weight.chunk(2, dim=0)
                w2_weight = (
                    weights["down_proj"][expert_id].transpose(0, 1).contiguous()
                    if "down_proj" in weights
                    else None
                )
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode: {self.weight_loading_mode}"
                )

            device = self.w3_w1_weight.device
            dst_w3_w1 = self.w3_w1_weight.data[local_slot_id]
            dst_w2 = self.w2_weight.data[local_slot_id]

            # Load w1 (up proj) and w3 (gate proj) into fused w3_w1
            if w1_weight is not None:
                w1_shard = load_weight_shard(
                    w1_weight, self.tp_size, self.tp_rank, TensorParallelMode.COLUMN, device=device
                )
                dst_w3_w1_w3, dst_w3_w1_w1 = dst_w3_w1.chunk(2, dim=0)
                dst_w3_w1_w1.copy_(w1_shard.contiguous().view(dst_w3_w1.dtype), non_blocking=True)

            if w3_weight is not None:
                w3_shard = load_weight_shard(
                    w3_weight, self.tp_size, self.tp_rank, TensorParallelMode.COLUMN, device=device
                )
                dst_w3_w1_w3, _ = dst_w3_w1.chunk(2, dim=0)
                dst_w3_w1_w3.copy_(w3_shard.contiguous().view(dst_w3_w1.dtype), non_blocking=True)

            # Load w2 (down proj)
            if w2_weight is not None:
                w2_shard = load_weight_shard(
                    w2_weight, self.tp_size, self.tp_rank, TensorParallelMode.ROW, device=device
                )
                dst_w2.copy_(w2_shard.view(dst_w2.dtype), non_blocking=True)

    def post_load_weights(self):
        """Post-load weight processing. Sets up EPLB if needed."""
        if hasattr(self, "layer_load_balancer") and self.layer_load_balancer:
            weight_fns = {
                "w3_w1_weight": self.w3_w1_weight.data,
                "w2_weight": self.w2_weight.data,
            }
            self.register_all_parameter_slot_and_to_fix_weight_fns(weight_fns)
            self.layer_load_balancer.set_initial_weight_assignments(self.initial_global_assignments)

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """No quantization for FlashMoE (bf16/fp16 only)."""
        return x, None

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Fused dispatch + expert FFN + combine.

        Algorithm:
        1. Flatten (token, top_k) -> sorted by expert ID
        2. For each local expert: GEMM1(gate_up) + SwiGLU + GEMM2(down)
        3. Scatter-add weighted results back to token positions

        Args:
            x: Input [num_tokens, hidden_size]
            token_selected_experts: Expert IDs or slots [num_tokens, top_k]
            token_final_scales: Routing weights [num_tokens, top_k]

        Returns:
            Output [num_tokens, hidden_size]
        """
        num_tokens = x.shape[0]
        hidden_size = x.shape[1]
        top_k = token_selected_experts.shape[1]
        output_dtype = x.dtype

        # Initialize output
        final_output = torch.zeros((num_tokens, hidden_size), dtype=output_dtype, device=x.device)

        # Flatten [num_tokens, top_k] -> [num_tokens * top_k]
        flat_expert_ids = token_selected_experts.view(-1)  # [N*K]
        flat_scales = token_final_scales.float().view(-1)  # [N*K]
        # Token indices: [0,0,..,1,1,..,2,2,..] each repeated top_k times
        flat_token_ids = (
            torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        )  # [N*K]

        # Process each local expert
        for local_expert_idx in range(self.expert_size_per_partition):
            # Global expert slot ID for this local expert
            global_slot_id = self.slot_start + local_expert_idx

            # Find tokens assigned to this expert
            mask = flat_expert_ids == global_slot_id
            if not mask.any():
                continue

            # Gather tokens and scales for this expert
            expert_token_ids = flat_token_ids[mask]  # [T_e]
            expert_scales = flat_scales[mask]  # [T_e]
            expert_input = x[expert_token_ids]  # [T_e, H]

            # Expert FFN: gate_up projection
            # w3_w1_weight[local_expert_idx] shape: [2*I, H]
            w3_w1 = self.w3_w1_weight[local_expert_idx]  # [2*I, H]

            # GEMM1: [T_e, H] x [H, 2*I] -> [T_e, 2*I]
            gate_up = torch.mm(expert_input, w3_w1.t())  # [T_e, 2*I]

            if self.w3_w1_bias is not None:
                gate_up = gate_up + self.w3_w1_bias[local_expert_idx]

            # Split: w3_w1 layout is [w3(up_proj), w1(gate_proj)]
            # w3 = up projection (direct multiply), w1 = gate projection (SiLU)
            # SwiGLU = silu(w1*x) * (w3*x)
            w3_out, w1_out = gate_up.chunk(2, dim=-1)  # each [T_e, I]

            if self.is_gated_activation:
                hidden = _swiglu(w1_out, w3_out)
            else:
                hidden = w3_out  # For non-gated, only use the first projection

            # GEMM2: down projection
            # w2_weight[local_expert_idx] shape: [H, I]
            w2 = self.w2_weight[local_expert_idx]  # [H, I]
            expert_output = torch.mm(hidden, w2.t())  # [T_e, H]

            if self.w2_bias is not None:
                expert_output = expert_output + self.w2_bias[local_expert_idx]

            # Weighted scatter-add back to original positions
            weighted_output = expert_output * expert_scales.unsqueeze(1).to(expert_output.dtype)
            final_output.index_add_(0, expert_token_ids, weighted_output)

        return final_output

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Full forward pass: routing -> run_moe -> reduce.

        This is the legacy single-backend forward path used when NOT wrapped
        by ConfigurableMoE.
        """
        assert isinstance(x, torch.Tensor), (
            "FlashMoEFused does not support Fp4QuantizedTensor input"
        )
        assert do_finalize, "FlashMoEFused does not support do_finalize=False"

        x = x.view(-1, self.hidden_size)

        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        token_selected_experts = token_selected_experts.to(torch.int32)

        # Apply router weight on input if enabled
        if self.apply_router_weight_on_input:
            x = x * token_final_scales.to(x.dtype)
            token_final_scales = None

        # AllGather for attention DP
        if self.use_dp and self.parallel_size > 1:
            from ...distributed import allgather

            x, token_selected_experts, token_final_scales = allgather(
                [x, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens,
            )

        # Quantize (no-op for FlashMoE)
        x, x_sf = self.quantize_input(x)

        # Run fused MoE computation
        final_hidden_states = self.run_moe(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
        )

        # ReduceScatter or AllReduce for TP
        final_hidden_states = self.reducescatter_or_allreduce(
            final_hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        return final_hidden_states


class FlashMoECuteDsl(FlashMoEFused):
    """FlashMoE with cuteDSL kernel-level fusion.

    Standalone MoE module (NOT wrapped by ConfigurableMoE).
    Fuses ALL operations into cuteDSL-backed kernels:
    - Single-GPU: moe_sort -> gather + GEMM + SwiGLU -> GEMM + scatter-add
    - Multi-GPU EP: symmetric memory AllGather + GEMM + ReduceScatter

    Uses moe_sort for tile management (same as CuteDslFusedMoE) and
    bf16 grouped GEMM kernels for compute.

    Multi-GPU EP communication (V1):
    - AllGather input tokens via symmetric memory + cuMemcpyDtoDAsync
    - GEMM on all tokens with local experts only (moe_sort handles partitioning)
    - ReduceScatter output via symmetric memory + cuMemcpyDtoDAsync + local sum

    Only supports unquantized bf16 on SM >= 90 (Hopper/Blackwell).
    """

    # Default tile size for moe_sort tile management
    DEFAULT_TILE_SIZE = 128

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Check if FlashMoECuteDsl can be used.

        Requirements:
        - Unquantized (quant_algo=None)
        - bf16 only (not fp16 - cuteDSL kernels target bf16 HMMA/WGMMA)
        - SM >= 90 (Hopper or Blackwell)
        - No swiglu_gptoss_style
        """
        can, reason = FlashMoEFused.can_implement(quant_algo, dtype_activation, swiglu_gptoss_style)
        if not can:
            return can, reason

        if dtype_activation != torch.bfloat16:
            return _warn_and_return(
                f"FlashMoECuteDsl only supports bfloat16 (got {dtype_activation})"
            )

        if not torch.cuda.is_available():
            return _warn_and_return("FlashMoECuteDsl requires CUDA")

        if torch.cuda.get_device_capability() < (9, 0):
            return _warn_and_return(
                f"FlashMoECuteDsl requires SM >= 90 (got SM {torch.cuda.get_device_capability()})"
            )

        return True, None

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        bias: bool = False,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
        use_symm_mem_ep: bool = False,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            without_comm=without_comm,
            activation_type=activation_type,
        )

        self.tile_size = self.DEFAULT_TILE_SIZE
        self.use_symm_mem_ep = use_symm_mem_ep

        # EP symmetric memory state (lazily initialized in _init_ep_symm_mem)
        self._ep_symm_mem_initialized = False
        self._input_symm = None
        self._input_hdl = None
        self._output_symm = None
        self._output_hdl = None
        self._remote_inputs = None
        self._remote_output_chunk = None
        self._ep_group_cached = None
        self._max_num_tokens_ep = 0

    def _init_ep_symm_mem(self, num_tokens: int):
        """Lazily initialize symmetric memory buffers for EP V2 communication.

        Allocates symmetric memory visible to all EP ranks via NVLink for:
        - Input buffer: each rank writes its tokens, other ranks read via
          cuMemcpyDtoDAsync from buffer_ptrs
        - Output buffer: each rank writes partial GEMM output, other ranks
          read their chunk and sum locally (ReduceScatter)

        Uses torch.distributed._symmetric_memory for buffer allocation and
        rendezvous, and cuda.bindings.driver for direct D2D copies.

        Args:
            num_tokens: Number of tokens for the current batch. Buffers are
                reallocated if this exceeds the previous max.
        """
        if self._ep_symm_mem_initialized and num_tokens <= self._max_num_tokens_ep:
            return

        if self.ep_size <= 1:
            self._ep_symm_mem_initialized = True
            return

        if not _SYMM_MEM_AVAILABLE:
            raise RuntimeError(
                "FlashMoECuteDsl symmetric memory EP requires PyTorch >= 2.5 "
                "with torch.distributed._symmetric_memory support"
            )
        if not _CUDA_DRIVER_AVAILABLE:
            raise RuntimeError(
                "FlashMoECuteDsl symmetric memory EP requires cuda-python "
                "(pip install cuda-python) for cuMemcpyDtoDAsync"
            )

        H = self.hidden_size
        ep_size = self.ep_size
        device = torch.cuda.current_device()

        # Get EP process group and enable symmetric memory
        ep_group = self._get_ep_process_group()
        self._ep_group_cached = ep_group
        group_name = str(ep_group.group_name)
        torch_symm_mem.enable_symm_mem_for_group(group_name)

        # Input symmetric memory: each rank's tokens [num_tokens, H]
        self._input_symm = torch_symm_mem.empty(
            (num_tokens, H), device=device, dtype=torch.bfloat16
        )
        self._input_hdl = torch_symm_mem.rendezvous(self._input_symm, group_name)

        # Output symmetric memory: partial GEMM results [num_tokens * ep_size, H]
        total = num_tokens * ep_size
        self._output_symm = torch_symm_mem.empty((total, H), device=device, dtype=torch.bfloat16)
        self._output_hdl = torch_symm_mem.rendezvous(self._output_symm, group_name)

        # Local buffers for remote ranks' input data (AllGather targets)
        self._remote_inputs = [
            torch.empty((num_tokens, H), device=device, dtype=torch.bfloat16)
            for _ in range(ep_size)
        ]

        # Single buffer for reading remote output chunks (ReduceScatter)
        self._remote_output_chunk = torch.empty(
            (num_tokens, H), device=device, dtype=torch.bfloat16
        )

        self._max_num_tokens_ep = num_tokens

        logger.info(
            "FlashMoECuteDsl symm mem EP initialized: ep_size=%d, ep_rank=%d, max_tokens=%d, H=%d",
            ep_size,
            self.ep_rank,
            num_tokens,
            H,
        )
        self._ep_symm_mem_initialized = True

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Fused dispatch + expert FFN + combine using moe_sort + cuteDSL kernels.

        Algorithm:
        1. moe_sort() -> tile assignments, permutation indices
        2. FC1: Gather + Grouped GEMM + SwiGLU (per expert tiles)
        3. FC2: Grouped GEMM + Scale + Scatter-Add (per expert tiles)

        Falls back to tile-based PyTorch loop (same structure as cuteDSL
        kernel, but using torch.mm) until compiled cuteDSL kernels are ready.

        Args:
            x: Input [num_tokens, hidden_size], bf16
            token_selected_experts: Expert IDs [num_tokens, top_k], int32
            token_final_scales: Routing weights [num_tokens, top_k], float32

        Returns:
            Output [num_tokens, hidden_size], bf16
        """
        num_tokens = x.shape[0]
        hidden_size = x.shape[1]
        top_k = token_selected_experts.shape[1]
        output_dtype = x.dtype

        # Step 1: moe_sort to get tile-to-expert mapping and permutation
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
            num_experts=self.num_slots,
            top_k=top_k,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tile_tokens_dim=self.tile_size,
        )

        # Step 2: FC1 - Gather + Grouped GEMM + SwiGLU
        fc1_output = torch.ops.trtllm.flashmoe_bf16_gather_gemm_swiglu(
            input=x,
            weight=self.w3_w1_weight,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            top_k=top_k,
            tile_size=self.tile_size,
            is_gated_activation=self.is_gated_activation,
        )

        # Step 3: FC2 - Grouped GEMM + Scale + Scatter-Add
        final_output = torch.ops.trtllm.flashmoe_bf16_gemm_finalize(
            input=fc1_output,
            weight=self.w2_weight,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            top_k=top_k,
            tile_size=self.tile_size,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            output_dtype=output_dtype,
        )

        return final_output

    def _get_ep_process_group(self):
        """Get the EP process group for collective communication.

        Tries mapping.moe_ep_group_pg (production), falls back to WORLD
        (standalone tests where world_size == ep_size).
        """
        try:
            return self.mapping.moe_ep_group_pg
        except (AttributeError, RuntimeError):
            return dist.group.WORLD

    def _forward_ep(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """EP communication: AllGather input + local GEMM + ReduceScatter.

        Each rank contributes its input tokens via AllGather so all ranks
        see all tokens. Each rank runs GEMM only on its local experts
        (moe_sort handles partitioning). Partial outputs are reduce-scattered
        (sum) back so each rank gets the full result for its original tokens.

        Requires all ranks to have the same number of tokens (pad if needed).

        Args:
            x: Input tokens [num_tokens, hidden_size], bf16.
            token_selected_experts: Expert IDs [num_tokens, top_k], int32.
            token_final_scales: Routing weights [num_tokens, top_k] or None.

        Returns:
            Output [num_tokens, hidden_size], bf16.
        """
        ep_size = self.ep_size
        ep_group = self._get_ep_process_group()

        # 1. AllGather input tokens across EP ranks
        all_x_list = [torch.empty_like(x) for _ in range(ep_size)]
        dist.all_gather(all_x_list, x.contiguous(), group=ep_group)
        all_x = torch.cat(all_x_list, dim=0)  # [T_total, H]

        # 2. AllGather routing info (small tensors, fine over NCCL)
        all_experts_list = [torch.empty_like(token_selected_experts) for _ in range(ep_size)]
        dist.all_gather(
            all_experts_list,
            token_selected_experts.contiguous(),
            group=ep_group,
        )
        all_experts = torch.cat(all_experts_list, dim=0)

        if token_final_scales is not None:
            all_scales_list = [torch.empty_like(token_final_scales) for _ in range(ep_size)]
            dist.all_gather(
                all_scales_list,
                token_final_scales.contiguous(),
                group=ep_group,
            )
            all_scales = torch.cat(all_scales_list, dim=0)
        else:
            all_scales = None

        # 3. Compute: run_moe on ALL tokens with LOCAL experts only
        partial_output = self.run_moe(all_x, all_experts, all_scales)
        # partial_output: [T_total, H] with contributions from local experts

        # 4. ReduceScatter: sum partial outputs, each rank gets its portion
        # Layout: [rank0_tokens, rank1_tokens, ..., rankN_tokens]
        output = torch.empty_like(x)
        dist.reduce_scatter_tensor(
            output,
            partial_output.contiguous(),
            op=dist.ReduceOp.SUM,
            group=ep_group,
        )

        return output

    @staticmethod
    def _get_remote_buffer_ptr(handle, rank: int) -> int:
        """Get device pointer for remote rank's symmetric memory buffer.

        The symmetric memory handle from rendezvous() exposes each rank's
        buffer device pointer. The attribute name varies by PyTorch version.

        Args:
            handle: Symmetric memory handle from rendezvous().
            rank: Remote rank index.

        Returns:
            CUdeviceptr as int for the remote rank's buffer.
        """
        if hasattr(handle, "buffer_ptrs_dev"):
            return handle.buffer_ptrs_dev[rank]
        if hasattr(handle, "buffer_ptrs"):
            return handle.buffer_ptrs[rank]
        raise AttributeError(
            f"Symmetric memory handle has no buffer_ptrs attribute. "
            f"Available: {[a for a in dir(handle) if not a.startswith('_')]}"
        )

    def _forward_ep_v2(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """EP communication via symmetric memory + cuMemcpyDtoDAsync.

        Replaces NCCL AllGather/ReduceScatter with direct NVLink D2D copies
        through symmetric memory buffer pointers. Lower latency than NCCL
        for small-to-medium token counts.

        Flow:
        1. AllGather: each rank writes tokens to symmetric input buffer,
           barrier, then reads other ranks' buffers via cuMemcpyDtoDAsync
        2. AllGather routing info via NCCL (small tensors, not latency-critical)
        3. Compute: run_moe on all tokens with local experts
        4. ReduceScatter: each rank writes partial output to symmetric output
           buffer, barrier, then reads its chunk from other ranks and sums

        Args:
            x: Input tokens [num_tokens, hidden_size], bf16.
            token_selected_experts: Expert IDs [num_tokens, top_k], int32.
            token_final_scales: Routing weights [num_tokens, top_k] or None.

        Returns:
            Output [num_tokens, hidden_size], bf16.
        """
        ep_size = self.ep_size
        num_tokens, H = x.shape
        ep_group = self._ep_group_cached or self._get_ep_process_group()

        # Lazily initialize symmetric memory buffers
        self._init_ep_symm_mem(num_tokens)

        stream = torch.cuda.current_stream()
        stream_ptr = stream.cuda_stream

        # --- 1. AllGather input tokens via symmetric memory ---
        # Write local tokens to our symmetric buffer
        self._input_symm[:num_tokens].copy_(x)

        # Barrier: ensure all ranks have written before reading
        dist.barrier(group=ep_group)

        # Read remote ranks' tokens via direct D2D NVLink copies
        all_x_parts = []
        for rank in range(ep_size):
            if rank == self.ep_rank:
                all_x_parts.append(x)
            else:
                dst = self._remote_inputs[rank][:num_tokens]
                src_ptr = self._get_remote_buffer_ptr(self._input_hdl, rank)
                nbytes = num_tokens * H * x.element_size()
                cuda_driver.cuMemcpyDtoDAsync(dst.data_ptr(), src_ptr, nbytes, stream_ptr)
                all_x_parts.append(dst)

        # Sync to ensure all D2D copies complete before cat/GEMM
        stream.synchronize()
        all_x = torch.cat(all_x_parts, dim=0)  # [T_total, H]

        # --- 2. AllGather routing info (small tensors, still NCCL) ---
        all_experts_list = [torch.empty_like(token_selected_experts) for _ in range(ep_size)]
        dist.all_gather(
            all_experts_list,
            token_selected_experts.contiguous(),
            group=ep_group,
        )
        all_experts = torch.cat(all_experts_list, dim=0)

        if token_final_scales is not None:
            all_scales_list = [torch.empty_like(token_final_scales) for _ in range(ep_size)]
            dist.all_gather(
                all_scales_list,
                token_final_scales.contiguous(),
                group=ep_group,
            )
            all_scales = torch.cat(all_scales_list, dim=0)
        else:
            all_scales = None

        # --- 3. Compute: run_moe on ALL tokens with LOCAL experts ---
        partial_output = self.run_moe(all_x, all_experts, all_scales)
        # partial_output: [T_total, H] with contributions from local experts

        # --- 4. ReduceScatter via symmetric memory ---
        # Write partial output to symmetric buffer
        total_tokens = partial_output.shape[0]
        self._output_symm[:total_tokens].copy_(partial_output)

        # Barrier: ensure all ranks have written before reading
        dist.barrier(group=ep_group)

        # Each rank reads its chunk from all ranks and sums
        output = torch.zeros_like(x)
        chunk_start = self.ep_rank * num_tokens
        elem_size = x.element_size()

        for rank in range(ep_size):
            if rank == self.ep_rank:
                output += partial_output[chunk_start : chunk_start + num_tokens]
            else:
                dst = self._remote_output_chunk[:num_tokens]
                src_ptr = self._get_remote_buffer_ptr(self._output_hdl, rank)
                # Offset to our chunk within remote rank's output buffer
                src_offset = chunk_start * H * elem_size
                nbytes = num_tokens * H * elem_size
                cuda_driver.cuMemcpyDtoDAsync(
                    dst.data_ptr(), src_ptr + src_offset, nbytes, stream_ptr
                )
                # Must sync before accumulating — next iteration reuses dst
                stream.synchronize()
                output += dst

        return output

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Full forward: routing -> EP comm + GEMM -> TP reduce.

        For ep_size > 1: AllGather + local GEMM + ReduceScatter.
        - V1 (default): NCCL collectives via _forward_ep()
        - V2 (use_symm_mem_ep=True): symmetric memory D2D copies via
          _forward_ep_v2()
        EP handles all inter-GPU communication, so DP allgather and TP
        reduce are skipped.
        For ep_size == 1: direct run_moe with optional DP/TP comm.
        """
        assert isinstance(x, torch.Tensor), (
            "FlashMoECuteDsl does not support Fp4QuantizedTensor input"
        )
        assert do_finalize, "FlashMoECuteDsl does not support do_finalize=False"

        x = x.view(-1, self.hidden_size)

        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        token_selected_experts = token_selected_experts.to(torch.int32)

        # Apply router weight on input if enabled
        if self.apply_router_weight_on_input:
            x = x * token_final_scales.to(x.dtype)
            token_final_scales = None

        if self.ep_size > 1:
            # EP path: AllGather + local GEMM + ReduceScatter.
            # EP handles ALL inter-GPU communication; skip DP/TP comm.
            x, x_sf = self.quantize_input(x)
            if self.use_symm_mem_ep:
                return self._forward_ep_v2(x, token_selected_experts, token_final_scales)
            return self._forward_ep(x, token_selected_experts, token_final_scales)

        # Single-GPU or TP-only path (no EP)
        # DP AllGather (if using attention DP)
        if self.use_dp and self.parallel_size > 1:
            from ...distributed import allgather

            x, token_selected_experts, token_final_scales = allgather(
                [x, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens,
            )

        # Quantize (no-op for FlashMoE)
        x, x_sf = self.quantize_input(x)

        final_hidden_states = self.run_moe(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
        )

        # ReduceScatter or AllReduce for TP
        final_hidden_states = self.reducescatter_or_allreduce(
            final_hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        return final_hidden_states
