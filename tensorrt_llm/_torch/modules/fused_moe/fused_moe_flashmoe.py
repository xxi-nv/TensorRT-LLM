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


def _build_chunk_mappings(
    permuted_idx_to_expanded_idx: torch.Tensor,
    top_k: int,
    seq_len: int,
    ep_size: int,
    num_valid_rows: int,
) -> list:
    """Build per-chunk (per-rank) index mappings for pipelined scatter-add.

    Groups permuted rows by destination rank chunk. For each chunk c
    (rank c's tokens), returns the permuted row indices, local token
    indices within the chunk, and expanded indices for scale lookup.

    Only considers rows within [0, num_valid_rows) to skip padding.

    Args:
        permuted_idx_to_expanded_idx: [T_permuted] int32, from moe_sort
        top_k: experts per token
        seq_len: tokens per rank
        ep_size: number of EP ranks
        num_valid_rows: number of valid (non-padding) permuted rows

    Returns:
        List of (perm_row_indices, local_token_indices, expanded_indices)
        tuples, one per chunk. All tensors are int64 on the same device.
    """
    device = permuted_idx_to_expanded_idx.device

    # Only look at valid rows (exclude padding from moe_sort)
    valid_expanded = permuted_idx_to_expanded_idx[:num_valid_rows].long()
    valid_row_ids = torch.arange(num_valid_rows, device=device)

    # Compute token index and chunk assignment for each valid row
    token_idx = valid_expanded // top_k  # [num_valid_rows]
    chunk_idx = token_idx // seq_len  # which rank's chunk (0..ep_size-1)

    # Build per-chunk mappings
    chunk_mappings = []
    for c in range(ep_size):
        mask = chunk_idx == c
        perm_rows = valid_row_ids[mask]
        local_tokens = (token_idx[mask] - c * seq_len).long()
        expanded_idx = valid_expanded[mask]
        chunk_mappings.append((perm_rows, local_tokens, expanded_idx))

    return chunk_mappings


class FlashMoECuteDsl(FlashMoEFused):
    """FlashMoE with cuteDSL kernel-level fusion.

    Standalone MoE module (NOT wrapped by ConfigurableMoE).
    Fuses ALL operations into cuteDSL-backed kernels:
    - Single-GPU: moe_sort -> gather + GEMM + SwiGLU -> GEMM + scatter-add
    - Multi-GPU EP: symmetric memory AllGather + GEMM + ReduceScatter

    Uses moe_sort for tile management (same as CuteDslFusedMoE) and
    bf16 grouped GEMM kernels for compute.

    Multi-GPU EP communication versions:
    - V1 (default): NCCL AllGather + local GEMM + NCCL ReduceScatter
    - V2 (use_symm_mem_ep=True): symmetric memory P2P for AG/RS
    - V3 (ep_comm_version='v3'): split FC2 + pipelined per-chunk RS
    - Graph (ep_comm_version='graph'): CUDA Graph capture of V1 pipeline

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
        ep_comm_version: str = "v1",
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
        self.ep_comm_version = ep_comm_version

        # EP symmetric memory state (lazily initialized in _init_ep_symm_mem)
        self._ep_symm_mem_initialized = False
        self._input_symm = None
        self._input_hdl = None
        self._output_symm = None
        self._output_hdl = None
        self._ep_group_cached = None
        self._max_num_tokens_ep = 0

        # V3 pipelined RS: comm stream for overlapping scatter-add with RS
        self._comm_stream = None

        # V3.4c CUDA Graph: state for graph capture/replay
        self._ep_graph = None
        self._ep_graph_num_tokens = 0
        # Fixed-shape graph buffers (lazily allocated in _init_graph_buffers)
        self._graph_input = None
        self._graph_experts = None
        self._graph_scales = None
        self._graph_output = None

    def _init_graph_buffers(self, num_tokens: int):
        """Pre-allocate fixed-shape buffers for CUDA Graph EP capture.

        All buffers are sized for num_tokens per rank. The graph is
        captured once and replayed for subsequent forwards with the
        same token count.

        Args:
            num_tokens: Number of tokens per rank (must be constant
                across graph replays).
        """
        if self._graph_input is not None and self._ep_graph_num_tokens == num_tokens:
            return

        ep_size = self.ep_size
        H = self.hidden_size
        top_k = self.routing_method.top_k
        device = torch.cuda.current_device()
        dtype = self.dtype or torch.bfloat16
        total = num_tokens * ep_size

        self._graph_input = torch.zeros(num_tokens, H, dtype=dtype, device=device)
        self._graph_experts = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
        self._graph_scales = torch.zeros(num_tokens, top_k, dtype=torch.float32, device=device)
        self._graph_output = torch.zeros(num_tokens, H, dtype=dtype, device=device)

        # Pre-allocate AllGather destination buffers
        self._graph_all_x = torch.zeros(total, H, dtype=dtype, device=device)
        self._graph_all_experts = torch.zeros(total, top_k, dtype=torch.int32, device=device)
        self._graph_all_scales = torch.zeros(total, top_k, dtype=torch.float32, device=device)

        self._ep_graph_num_tokens = num_tokens
        self._ep_graph = None  # Invalidate any existing graph

        logger.info(
            "FlashMoECuteDsl graph buffers initialized: ep_size=%d, tokens=%d, H=%d",
            ep_size,
            num_tokens,
            H,
        )

    def _capture_ep_graph(self, num_tokens: int):
        """Capture the V1 NCCL EP pipeline as a CUDA Graph.

        Captures: AllGather → moe_sort → FC1 → FC2 → ReduceScatter.

        The graph uses fixed-shape buffers allocated by _init_graph_buffers().
        Data is copied into graph buffers before replay and copied out after.

        NOTE: The cuteDSL kernel path does lazy compilation (Python side
        effects) which is not graph-safe. This method forces the torch.mm
        fallback path during capture by temporarily disabling cuteDSL.

        Args:
            num_tokens: Tokens per rank (must match _init_graph_buffers).
        """

        ep_size = self.ep_size
        ep_group = self._get_ep_process_group()

        self._init_graph_buffers(num_tokens)

        # Force fallback path during graph capture to avoid cuteDSL
        # lazy compilation side effects
        import tensorrt_llm._torch.custom_ops.flashmoe_bf16_custom_ops as _ops

        old_disable = _ops._FORCE_DISABLE_CUTEDSL
        _ops._FORCE_DISABLE_CUTEDSL = True

        # Warmup run (CUDA graph requires at least one eager run)
        self._forward_ep_graph_body(num_tokens, ep_group)

        # Capture
        self._ep_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._ep_graph):
            self._forward_ep_graph_body(num_tokens, ep_group)

        # Restore cuteDSL setting
        _ops._FORCE_DISABLE_CUTEDSL = old_disable

        logger.info(
            "FlashMoECuteDsl CUDA Graph captured: tokens=%d, ep_size=%d",
            num_tokens,
            ep_size,
        )

    def _forward_ep_graph_body(self, num_tokens: int, ep_group):
        """The EP forward body used for both graph capture and eager warmup.

        Reads from self._graph_input/experts/scales and writes to
        self._graph_output. Uses pre-allocated AllGather buffers.
        """
        ep_size = self.ep_size

        x = self._graph_input
        experts = self._graph_experts
        scales = self._graph_scales

        # AllGather input tokens
        all_x_list = list(self._graph_all_x.chunk(ep_size, dim=0))
        dist.all_gather(all_x_list, x.contiguous(), group=ep_group)

        # AllGather routing info
        all_experts_list = list(self._graph_all_experts.chunk(ep_size, dim=0))
        dist.all_gather(all_experts_list, experts.contiguous(), group=ep_group)

        all_scales_list = list(self._graph_all_scales.chunk(ep_size, dim=0))
        dist.all_gather(all_scales_list, scales.contiguous(), group=ep_group)

        # Compute: moe_sort + FC1 + FC2 on all gathered tokens
        partial_output = self.run_moe(
            self._graph_all_x, self._graph_all_experts, self._graph_all_scales
        )

        # ReduceScatter
        dist.reduce_scatter_tensor(
            self._graph_output,
            partial_output.contiguous(),
            op=dist.ReduceOp.SUM,
            group=ep_group,
        )

    def _forward_ep_graph(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """EP forward via CUDA Graph replay.

        Copies inputs to graph buffers, replays the captured graph, and
        returns a clone of the graph output buffer.

        On first call (or when token count changes), captures the graph.

        Args:
            x: Input tokens [num_tokens, hidden_size], bf16.
            token_selected_experts: Expert IDs [num_tokens, top_k], int32.
            token_final_scales: Routing weights [num_tokens, top_k] or None.

        Returns:
            Output [num_tokens, hidden_size], bf16.
        """
        num_tokens = x.shape[0]

        # Capture graph if needed
        if self._ep_graph is None or self._ep_graph_num_tokens != num_tokens:
            self._capture_ep_graph(num_tokens)

        # Copy inputs into graph buffers
        self._graph_input.copy_(x)
        self._graph_experts.copy_(token_selected_experts)
        if token_final_scales is not None:
            self._graph_scales.copy_(token_final_scales)

        # Replay
        self._ep_graph.replay()

        return self._graph_output.clone()

    def _init_ep_symm_mem(self, num_tokens: int):
        """Lazily initialize symmetric memory buffers for EP V2 communication.

        Allocates symmetric memory visible to all EP ranks via NVLink for:
        - Input buffer: each rank writes its tokens, other ranks read via
          handle.get_buffer(rank, shape, dtype) + .copy_()
        - Output buffer: each rank writes partial GEMM output, other ranks
          read their chunk and sum locally (ReduceScatter)

        Uses torch.distributed._symmetric_memory for buffer allocation,
        rendezvous, barrier, and remote buffer access (NVLink P2P).

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

    def _forward_ep_v2(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """EP communication via symmetric memory P2P tensor access.

        Replaces NCCL AllGather/ReduceScatter with direct NVLink reads
        through symmetric memory handle.get_buffer() + .copy_(). Each
        rank can directly read any other rank's symmetric buffer as a
        regular tensor without explicit D2D copy calls.

        Flow:
        1. AllGather: each rank writes tokens to symmetric input buffer,
           barrier, then reads other ranks' buffers via get_buffer + copy
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

        # --- 1. AllGather input tokens via symmetric memory ---
        # Write local tokens to our symmetric buffer
        self._input_symm[:num_tokens].copy_(x)

        # Barrier: ensure all ranks have written before reading
        self._input_hdl.barrier()

        # Read remote ranks' tokens via P2P tensor access
        input_shape = (num_tokens, H)
        all_x_parts = []
        for rank in range(ep_size):
            if rank == self.ep_rank:
                all_x_parts.append(x.clone())
            else:
                src = self._input_hdl.get_buffer(rank, input_shape, x.dtype)
                all_x_parts.append(src.clone())
        all_x = torch.cat(all_x_parts, dim=0)  # [T_total, H]

        # Second barrier: ensure reads are done before writers can modify
        self._input_hdl.barrier()

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
        self._output_hdl.barrier()

        # Each rank reads its chunk from all ranks and sums
        output = torch.zeros_like(x)
        chunk_start = self.ep_rank * num_tokens
        out_total_shape = (total_tokens, H)

        for rank in range(ep_size):
            src = self._output_hdl.get_buffer(rank, out_total_shape, x.dtype)
            output += src[chunk_start : chunk_start + num_tokens]

        # Second barrier: ensure reads are done before writers can modify
        self._output_hdl.barrier()

        return output

    def _forward_ep_v3(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """EP communication with pipelined ReduceScatter (V3.4d).

        Splits FC2 into GEMM-only + per-chunk scatter-add, then overlaps
        the per-chunk scatter-add with ReduceScatter using CUDA streams.

        Flow:
        1. AllGather input tokens + routing (same as V1/V2)
        2. moe_sort + FC1 (same as before)
        3. FC2 GEMM only (no scatter-add) -> [T_permuted, H]
        4. Build per-chunk index mappings
        5. Pipelined loop:
           - Compute stream: scatter-add chunk c
           - Comm stream: reduce(chunk c-1) to destination rank
        6. Final reduce for last chunk

        Args:
            x: Input tokens [num_tokens, hidden_size], bf16.
            token_selected_experts: Expert IDs [num_tokens, top_k], int32.
            token_final_scales: Routing weights [num_tokens, top_k] or None.

        Returns:
            Output [num_tokens, hidden_size], bf16.
        """
        ep_size = self.ep_size
        ep_group = self._get_ep_process_group()
        num_tokens, H = x.shape
        top_k = token_selected_experts.shape[1]

        # --- 1. AllGather input tokens (NCCL) ---
        all_x_list = [torch.empty_like(x) for _ in range(ep_size)]
        dist.all_gather(all_x_list, x.contiguous(), group=ep_group)
        all_x = torch.cat(all_x_list, dim=0)  # [T_total, H]

        # AllGather routing info
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

        # --- 2. moe_sort + FC1 ---
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = torch.ops.trtllm.moe_sort(
            token_selected_experts=all_experts,
            token_final_scales=all_scales,
            num_experts=self.num_slots,
            top_k=top_k,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tile_tokens_dim=self.tile_size,
        )

        fc1_output = torch.ops.trtllm.flashmoe_bf16_gather_gemm_swiglu(
            input=all_x,
            weight=self.w3_w1_weight,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            top_k=top_k,
            tile_size=self.tile_size,
            is_gated_activation=self.is_gated_activation,
        )

        # --- 3. FC2 GEMM only (no scatter-add) ---
        fc2_gemm_out = torch.ops.trtllm.flashmoe_bf16_gemm_only(
            input=fc1_output,
            weight=self.w2_weight,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_size=self.tile_size,
        )
        # fc2_gemm_out: [T_permuted, H] in permuted order

        # --- 4. Build per-chunk index mappings ---
        # Compute total valid rows from tile info
        n_valid_tiles = num_non_exiting_tiles.item()
        if n_valid_tiles > 0:
            num_valid_rows = tile_idx_to_mn_limit[n_valid_tiles - 1].item()
        else:
            num_valid_rows = 0

        chunk_mappings = _build_chunk_mappings(
            permuted_idx_to_expanded_idx, top_k, num_tokens, ep_size, num_valid_rows
        )

        # --- 5. Pipelined scatter-add + ReduceScatter ---
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream()
        comm_stream = self._comm_stream
        compute_stream = torch.cuda.current_stream()

        # Allocate per-chunk output buffers
        chunk_bufs = [
            torch.zeros(num_tokens, H, dtype=x.dtype, device=x.device) for _ in range(ep_size)
        ]
        scatter_events = [torch.cuda.Event() for _ in range(ep_size)]

        for c in range(ep_size):
            perm_rows, local_tokens, expanded_idx = chunk_mappings[c]

            # Scatter-add on compute stream
            torch.ops.trtllm.flashmoe_scatter_add_chunk(
                fc2_output=fc2_gemm_out,
                chunk_output=chunk_bufs[c],
                perm_row_indices=perm_rows,
                local_token_indices=local_tokens,
                expanded_indices=expanded_idx,
                token_final_scales=all_scales if all_scales is not None else token_final_scales,
            )
            scatter_events[c].record(compute_stream)

            # Launch reduce for previous chunk on comm stream (pipelined)
            if c > 0:
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_event(scatter_events[c - 1])
                    dist.reduce(
                        chunk_bufs[c - 1],
                        dst=c - 1,
                        op=dist.ReduceOp.SUM,
                        group=ep_group,
                    )

        # Handle last chunk reduce
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(scatter_events[ep_size - 1])
            dist.reduce(
                chunk_bufs[ep_size - 1],
                dst=ep_size - 1,
                op=dist.ReduceOp.SUM,
                group=ep_group,
            )

        # Wait for all comm to finish
        compute_stream.wait_stream(comm_stream)

        # My output is the chunk corresponding to my rank
        return chunk_bufs[self.ep_rank]

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
        - V2 (use_symm_mem_ep=True): symmetric memory P2P via
          _forward_ep_v2()
        - V3 (ep_comm_version='v3'): pipelined RS via _forward_ep_v3()
        - Graph (ep_comm_version='graph'): CUDA Graph via _forward_ep_graph()
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
            if self.ep_comm_version == "v3":
                return self._forward_ep_v3(x, token_selected_experts, token_final_scales)
            if self.ep_comm_version == "graph":
                return self._forward_ep_graph(x, token_selected_experts, token_final_scales)
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
