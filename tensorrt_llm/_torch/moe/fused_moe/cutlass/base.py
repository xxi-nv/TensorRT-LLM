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

from typing import Dict, List, Optional, Tuple

import torch

from ....model_config import ModelConfig
from ....peft.lora.layer import (
    MOE_LORA_MODULE_NAMES,
    MOE_LORA_MODULE_TO_KERNEL_SLOT,
    LoraModuleType,
    MoeLoraLayer,
)
from ....peft.lora.validation import has_moe_lora_targets
from ....utils import ActivationType, AuxStreamType, EventType
from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import MoERunContext

# isort: off
from ..quantization import (
    MoEWeightLoadingMode,
)

# isort: on
from ..routing import BaseMoeRoutingMethod
from .eligibility import SmSupport


def raise_moe_lora_multichunk_unsupported(num_chunks: int) -> None:
    """Reject multi-chunk execution for routed-expert MoE LoRA.

    Routed-expert MoE LoRA passes per-request/slot adapter metadata that is not
    re-sliced per token-chunk, so multi-chunk execution would mismatch the
    kernel's per-token expansion. Shared by the Cutlass leaves' forward path and the
    MoEScheduler so the message stays in one place.
    """
    raise NotImplementedError(
        f"Routed-expert MoE LoRA does not support multi-chunk execution "
        f"(num_chunks={num_chunks}). Reduce the per-forward token count or "
        f"increase `moe_max_num_tokens` so the MoE runs in a single chunk."
    )


class CutlassFusedMoEBase(MoEImplBase):
    """Abstract root of the Cutlass-family implementations.

    Holds what every leaf shares: construction, the routed-expert LoRA
    plumbing, the weight lifecycle, input quantization, and the
    ``torch.ops.trtllm.fused_moe`` call. Abstract and unregistered -- it
    implements none of ``can_implement`` and publishes no descriptor, because a
    descriptor is exactly one identity and this class stands for twelve.

    What a leaf declares, and what the family gates in :mod:`.eligibility`
    read off it:

    * ``descriptor`` -- its one identity, which is also what
      ``check_quant_matches_identity`` compares a problem against.
    * ``sm_support`` -- the SM versions its kernel is built for.
    * ``supported_dtypes`` -- activation dtypes (BEFORE quantization) its
      kernel has instantiations for.
    * ``supports_gptoss_style`` / ``rejects_gptoss_expert_bias`` -- whether its
      weight method can load a gpt-oss / MiniMax SwiGLU shape.

    These replace the ``_QUANT_SUPPORT_TABLE`` this class used to interpret: a
    row that described a format the class did not implement is no longer
    expressible.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    MoE torch custom op:
        In max-throughput mode:
        Quant:
            fp8 block scales (SM90 Hopper only):
                FusedMoE Op: dynamic quant + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
            p8 qdq, nvfp4:
                FusedMoE Op: scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)

    FusedMoE module:
        max-throughput mode, with AttentionDP on:
            routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4]
            [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter,
            which equals: dynamic quant + routing(topK, etc.)
            [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 +
            finalizeMoeRoute [no allreduce] + reducescatter
    """

    # Declared by every leaf; no default, because a wrong SM set or dtype set
    # silently admits a layer whose kernel does not exist.
    sm_support: SmSupport
    supported_dtypes: frozenset
    # Whether the leaf's weight method can load a gpt-oss expert bias at all,
    # and -- for NVFP4, whose weight pad asserts 2-D -- whether a real 1-D
    # expert bias is still out while MiniMax-style SwigluBias is fine.
    supports_gptoss_style: bool = False
    rejects_gptoss_expert_bias: bool = False

    # The CUTLASS epilogue has an adaptor per kind (``moe_kernels.cuh``) and
    # takes all three constants as ``float*`` indexed by expert.
    activation_support = MoEActivationSupport(
        kinds=frozenset(
            {
                ActivationType.Swiglu,
                ActivationType.SwigluBias,
                ActivationType.Geglu,
                ActivationType.SiTu,
                ActivationType.Relu2,
                ActivationType.Silu,
            }
        ),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

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
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ):
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
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
            activation=activation,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
        )

        # ``unpadded_hidden_size`` is captured by
        # apply_moe_impl_construction_state() above, before the padding below.
        if (
            model_config.quant_config
            and model_config.quant_config.layer_quant_mode.has_w4a16_mxfp4()
        ):
            self.hidden_size = ((self.hidden_size + 127) // 128) * 128
            self.intermediate_size_per_partition = (
                (self.intermediate_size_per_partition + 127) // 128
            ) * 128

        # The EPLB layout (num_slots, expert_size_per_partition, slot_start,
        # slot_end, initial_*) comes from apply_moe_impl_construction_state();
        # ConfigurableMoE then overwrites it via _BACKEND_SYNC_ATTRS.

        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        # The auxiliary CUDA stream and CUDA events are only used when MoE chunking is applied
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size
        if self.moe_max_num_tokens < default_moe_max_num_tokens:
            self.aux_stream = (
                aux_stream_dict[AuxStreamType.MoeChunkingOverlap]
                if aux_stream_dict is not None
                else torch.cuda.Stream()
            )
            self.event_dict = {
                key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeChunkingOverlap]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        # The profiler converges on the same best tactic when the number of tokens is large enough.
        # To avoid long profiling time, the max number of tokens used in the profiling is capped to
        # around 16k tokens per expert, which is well into the compute bound domain.
        self.tune_max_num_tokens = min(
            self.moe_max_num_tokens,
            16384 * self.num_slots // routing_method.get_experts_per_token(),
        )
        self.has_been_profiled = False
        self.has_been_profiled_min_latency = False

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Finalize fusion should be disabled if Lora is used.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )

        # Routed-expert LoRA is fused inside torch.ops.trtllm.fused_moe. This
        # flag records whether the layer was configured with MoE LoRA targets,
        # so forward_impl can reject stray lora_params instead of ignoring them.
        self._moe_lora_enabled = self._has_moe_lora_targets(model_config)

        # Discovery-only marker submodule. The actual LoRA GEMMs are fused into
        # torch.ops.trtllm.fused_moe; MoeLoraLayer exists purely so that
        # CudaGraphLoraManager and the target-module validator can find this MoE
        # layer via isinstance(child, LoraLayer) traversal and read its
        # lora_module_types / output_hidden_sizes when building slot tables.
        self.lora = self._maybe_make_lora_marker(model_config)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ---- Routed-expert LoRA helpers ----

    def _has_moe_lora_targets(self, model_config: ModelConfig) -> bool:
        """Return True iff this MoE layer is in the routed-expert LoRA
        target-module set. The LoRA application itself is fused into
        `torch.ops.trtllm.fused_moe`; no submodule is registered.
        """
        return has_moe_lora_targets(getattr(model_config, "lora_config", None))

    def _maybe_make_lora_marker(self, model_config: ModelConfig) -> Optional[MoeLoraLayer]:
        """Construct a MoeLoraLayer marker iff this MoE layer is in the LoRA
        target-module set. The marker is a discovery-only submodule; the actual
        LoRA application is fused into torch.ops.trtllm.fused_moe.

        The output_hidden_sizes recorded here are the per-token outputs of the
        LoRA-side GEMM (not per-expert weight shapes): MOE_H_TO_4H / MOE_GATE
        produce intermediate_size, MOE_4H_TO_H produces hidden_size.
        """
        lora_config = getattr(model_config, "lora_config", None)
        if lora_config is None:
            return None
        # Normalize to lowercase to match has_moe_lora_targets (which lowercases
        # before comparing), so a mixed-case config marks the layer and builds
        # the discovery marker consistently.
        targets = {name.lower() for name in (getattr(lora_config, "lora_target_modules", []) or [])}
        active_modules: List[LoraModuleType] = []
        active_out_sizes: List[int] = []
        for name in MOE_LORA_MODULE_NAMES:
            if name not in targets:
                continue
            module_type = LoraModuleType.from_string(name)
            if name == "moe_4h_to_h":
                active_out_sizes.append(self.hidden_size)
            else:
                active_out_sizes.append(self.intermediate_size)
            active_modules.append(module_type)
        if not active_modules:
            return None
        return MoeLoraLayer(active_modules, active_out_sizes)

    def reserve_moe_lora_cuda_graph_workspace(
        self, max_num_tokens: int, max_lora_rank: int, max_lora_size: int
    ) -> None:
        """Pre-size the C++ FusedMoeRunner's MoE-LoRA scratch to the engine's
        worst case so no (re)allocation happens during CUDA graph capture or
        replay (which would dangle addresses baked into earlier graphs).

        No-op for layers without MoE LoRA targets and for quantized layers (MoE
        LoRA requires unquantized fp16/bf16); idempotent and grow-only. Call
        during warmup, before any capture that exercises MoE LoRA;
        CudaGraphLoraManager does this automatically.

        Args:
            max_num_tokens: Larger of the captured-decode capacity
                (max_batch_size * max_tokens_per_seq) and the engine-wide token limit.
            max_lora_rank: Largest LoRA rank across adapters.
            max_lora_size: Adapter-slot pool size for the slot-indexed device tables.
        """
        if not self._moe_lora_enabled or max_num_tokens <= 0:
            return
        # MoE LoRA runs on the unquantized fp16/bf16 path or the per-tensor FP8
        # (qdq) path (see moeOp.cpp). Any other quant mode (FP8 block-scale,
        # NVFP4, MXFP8, integer WoQ) is rejected by the C++ op, so a layer in
        # those modes can never reach the LoRA scratch; skip and let the runtime
        # path error loudly.
        if getattr(self, "has_any_quant", False) and not self.has_fp8_qdq:
            return
        # Weights must exist to read the runner's weight dtype. If they have not
        # been created yet, skip; the lazy sizing + in-capture guard still
        # protect correctness.
        if getattr(self, "w3_w1_weight", None) is None:
            return

        # The reservation must cover the engine's worst case, otherwise the first
        # capture hits a lazy allocation that the C++ in-capture guard rejects.
        assert max_lora_rank > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_rank > 0 "
            f"(got {max_lora_rank}); set lora_config.max_lora_rank."
        )
        assert max_lora_size > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_size > 0 "
            f"(got {max_lora_size})."
        )

        # The reservation must land on the *same* cached C++ FusedMoeRunner that
        # the runtime torch.ops.trtllm.fused_moe op uses on this layer, so the
        # MoERunner instance key must match the runtime key exactly. The runtime
        # key uses the activation dtype the op sees:
        #   - per-tensor FP8 (qdq): quantize_input casts activations to e4m3, so
        #     x/weight are fp8 and the output (LoRA compute) dtype is self.dtype;
        #   - unquantized fp16/bf16: x/weight/output all equal self.dtype.
        # Every quant flag in the key is False for both (per-tensor FP8 is not
        # block-scaled / MXFP8 / W4). If a runtime call ever uses a different
        # key, the C++ capture guard surfaces a clear error rather than
        # corrupting replay.
        weight_dtype = self.w3_w1_weight.dtype
        if self.has_fp8_qdq:
            act_dtype = torch.float8_e4m3fn
            output_dtype = self.dtype
        else:
            assert self.dtype in (torch.float16, torch.bfloat16), (
                "MoE LoRA requires fp16/bf16 activations to reserve a "
                f"deterministic FusedMoeRunner key; got {self.dtype}."
            )
            act_dtype = self.dtype
            output_dtype = self.dtype

        from ....custom_ops.torch_custom_ops import MoERunner

        runner = MoERunner(
            x_dtype=act_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            top_k=self.routing_method.experts_per_token,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_int8_woq_per_channel=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            activation_type=self.activation_type,
        )
        runner.fused_moe_runner.reserve_lora_host_buffers(
            int(max_num_tokens),
            int(self.routing_method.experts_per_token),
            int(max_lora_rank),
            int(max_lora_size),
            bool(self.is_gated_activation),
        )

    def _moe_lora_active(self, lora_params: Optional[Dict]) -> bool:
        """Return True when lora_params carries routed-expert MoE LoRA tensors
        for this layer, meaning run_moe would fuse a LoRA delta.
        """
        if not lora_params or self.layer_idx is None:
            return False
        # CUDA-graph slot-indexed mode carries MoE LoRA in cuda_graph_params
        # rather than a per-layer eager dict (mirrors _extract_moe_lora_tensors),
        # so consult the graph layer map to keep the stray-param and multi-chunk
        # guards effective during capture/replay.
        if lora_params.get("use_cuda_graph_mode", False):
            cuda_graph_params = lora_params.get("cuda_graph_params")
            if cuda_graph_params is None:
                return False
            layer_module2key = getattr(cuda_graph_params, "layer_module2key", {})
            return any(
                (self.layer_idx, int(LoraModuleType.from_string(name))) in layer_module2key
                for name in MOE_LORA_MODULE_NAMES
            )
        layer_params = lora_params.get(self.layer_idx, {})
        if not layer_params:
            return False
        return any(
            int(LoraModuleType.from_string(name)) in layer_params for name in MOE_LORA_MODULE_NAMES
        )

    @staticmethod
    def _empty_kernel_slot_dict() -> Dict[str, Optional[torch.Tensor]]:
        return {"fc1": None, "fc2": None, "gated": None}

    def _gather_moe_lora_slots(self, source):
        """Gather per-kernel-slot (ranks, weight_ptrs) tensors.

        `source(module_type)` returns the (ranks, weight_ptrs) pair for an MoE
        LoRA module, or None if absent. Returns (ranks_by_slot, ptrs_by_slot)
        dicts keyed by the kernel slot ("fc1"/"gated"/"fc2"); see
        MOE_LORA_MODULE_TO_KERNEL_SLOT for the module->slot convention. Shared by
        the eager (per-request) and CUDA-graph (slot-indexed) extraction paths.
        """
        ranks = self._empty_kernel_slot_dict()
        ptrs = self._empty_kernel_slot_dict()
        for module_type, slot in MOE_LORA_MODULE_TO_KERNEL_SLOT.items():
            got = source(module_type)
            if got is None:
                continue
            ranks[slot], ptrs[slot] = got
        return ranks, ptrs

    @staticmethod
    def _require_fc1_fc2(ranks: Dict[str, Optional[torch.Tensor]]) -> None:
        """The kernel always dereferences the fc1 and fc2 rank/pointer arrays
        (see setupLoraWorkspace in moe_kernels.cu), so moe_h_to_4h (fc1/gate) and
        moe_4h_to_h (fc2/down) must both be present when MoE LoRA is active. The
        gated slot (moe_gate) is only read for gated activations.
        """
        if ranks["fc1"] is None or ranks["fc2"] is None:
            raise ValueError(
                "MoE LoRA requires both `moe_h_to_4h` (gate/SiLU) and "
                "`moe_4h_to_h` (down) in lora_target_modules."
            )

    def _extract_moe_lora_tensors(self, lora_params: Optional[Dict]) -> Optional[Dict[str, object]]:
        """Pick the MoE-side LoRA tensors out of the global `lora_params` dict
        for this layer. Returns a dict with the kwargs expected by
        `torch.ops.trtllm.fused_moe`, or None when no MoE LoRA applies.

        Each entry is a CPU tensor:
            *_lora_ranks         : int32  [num_seqs]
            *_lora_weight_ptrs   : int64  [num_seqs, 3]   (A, B, DoRA_unused)
            host_request_types   : int32  [num_seqs]      (0=CTX, 1=GEN)
            host_context_lengths : int32  [num_seqs]
            lora_max_low_rank    : int (max rank across the active modules)
        """
        if not lora_params:
            return None
        # Slot-indexed (CUDA-graph decode) path: the per-token expansion is
        # driven inside the op by token_to_slot indexed into stable slot
        # tables owned by CudaGraphLoraParams (see _extract_moe_lora_tensors_cuda_graph).
        if lora_params.get("use_cuda_graph_mode", False):
            return self._extract_moe_lora_tensors_cuda_graph(lora_params)
        layer_params = lora_params.get(self.layer_idx, {}) if self.layer_idx is not None else {}
        if not layer_params:
            return None

        # Gather (ranks, weight_ptrs) per kernel slot. weight_pointers is built
        # flat ([num_seqs * 3], row-major (A, B, DoRA) per seq) in
        # PyTorchModelEngine._build_lora_params; the op expects [num_seqs, 3].
        active_max_rank = 0

        def _source(module_type: LoraModuleType):
            nonlocal active_max_rank
            entry = layer_params.get(int(module_type))
            if entry is None:
                return None
            rank_t = entry["adapter_size"]
            if rank_t.numel() > 0:
                active_max_rank = max(active_max_rank, int(rank_t.max().item()))
            return rank_t, entry["weight_pointers"].reshape(-1, 3)

        ranks, ptrs = self._gather_moe_lora_slots(_source)
        if all(v is None for v in ranks.values()):
            return None
        self._require_fc1_fc2(ranks)

        num_seqs = lora_params["num_seqs"]

        def _slice(t):
            return t[:num_seqs].contiguous() if t is not None else None

        return {
            "fc1_lora_ranks": _slice(ranks["fc1"]),
            "fc1_lora_weight_ptrs": _slice(ptrs["fc1"]),
            "fc2_lora_ranks": _slice(ranks["fc2"]),
            "fc2_lora_weight_ptrs": _slice(ptrs["fc2"]),
            "gated_lora_ranks": _slice(ranks["gated"]),
            "gated_lora_weight_ptrs": _slice(ptrs["gated"]),
            "host_request_types": _slice(lora_params["host_request_types"]),
            "host_context_lengths": _slice(lora_params["prompt_lens_cpu"]),
            "lora_max_low_rank": active_max_rank,
        }

    def _extract_moe_lora_tensors_cuda_graph(
        self, lora_params: Dict
    ) -> Optional[Dict[str, object]]:
        """CUDA-graph slot-indexed extraction for routed-expert MoE LoRA.

        Pulls per-module slot tables and token_to_slot out of
        CudaGraphLoraParams and returns the slot-indexed kwargs accepted by
        torch.ops.trtllm.fused_moe. Returns None when this layer does not
        carry any MoE LoRA modules in the graph layer map.

        Returned tensor addresses are stable across captures and replays: they
        come from persistent pinned host buffers owned by CudaGraphLoraParams
        and the per-module packed pointer cache. Uses the same module->kernel
        slot convention as the per-request path (moe_h_to_4h -> fc1,
        moe_gate -> gated, moe_4h_to_h -> fc2).
        """
        if self.layer_idx is None:
            return None
        cuda_graph_params = lora_params.get("cuda_graph_params")
        if cuda_graph_params is None:
            return None

        def _source(module_type: LoraModuleType):
            return cuda_graph_params.get_moe_slot_inputs(self.layer_idx, int(module_type))

        slot_ranks, slot_ptrs = self._gather_moe_lora_slots(_source)
        if slot_ranks["fc1"] is None or slot_ranks["fc2"] is None:
            return None

        num_seqs = lora_params["num_seqs"]
        tokens_per_seq = getattr(cuda_graph_params, "max_tokens_per_seq", 1)
        num_tokens = num_seqs * max(int(tokens_per_seq), 1)
        token_to_slot = cuda_graph_params.token_to_slot_host[:num_tokens].contiguous()

        # Pass the global max LoRA rank, not the per-step active max: the device
        # path uses it only to size the low-rank workspace strides baked into the
        # captured graph, so the global max keeps them valid for any per-slot
        # rank across replays. The actual per-token rank is read on-device from
        # the slot table, so a smaller rank just runs a smaller GEMM.
        max_rank = int(getattr(cuda_graph_params, "max_rank", 0))
        if max_rank <= 0:
            return None

        return {
            "fc1_slot_lora_ranks": slot_ranks["fc1"].contiguous(),
            "fc1_slot_lora_weight_ptrs": slot_ptrs["fc1"].contiguous(),
            "fc2_slot_lora_ranks": slot_ranks["fc2"].contiguous(),
            "fc2_slot_lora_weight_ptrs": slot_ptrs["fc2"].contiguous(),
            "gated_slot_lora_ranks": (
                slot_ranks["gated"].contiguous() if slot_ranks["gated"] is not None else None
            ),
            "gated_slot_lora_weight_ptrs": (
                slot_ptrs["gated"].contiguous() if slot_ptrs["gated"] is not None else None
            ),
            "token_to_slot": token_to_slot,
            "lora_max_low_rank": max_rank,
        }

    def _check_configs(self):
        assert self._weights_created

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

        if self.quant_config and self.quant_config.quant_mode.has_any_quant(exclude_kv_cache=True):
            if not (
                self.quant_config.quant_mode.has_nvfp4()
                | self.quant_config.quant_mode.has_fp8_block_scales()
                | self.quant_config.quant_mode.has_fp8_qdq()
                | self.quant_config.quant_mode.is_weight_only()
                | self.quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                | self.quant_config.quant_mode.has_w4a16_mxfp4()
                | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()
                | self.quant_config.quant_mode.has_mxfp8()
            ):
                raise ValueError(f"unsupported quantization mode: {self.quant_config.quant_mode}")

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group()

    @property
    def has_int8_woq_per_channel(self):
        return (
            self.quant_config
            and self.quant_config.layer_quant_mode.is_int8_weight_only()
            and not self.quant_config.layer_quant_mode.has_per_group_scaling()
        )

    def _supports_load_balancer(self) -> bool:
        """Every Cutlass leaf supports the load balancer."""
        return True

    def supports_moe_output_in_alltoall_workspace(self):
        return True

    def _tuner_shapes(
        self,
        ctx: MoERunContext,
        enable_alltoall: Optional[bool],
    ) -> Tuple[Optional[int], Optional[int]]:
        """Token/top-k shapes the profiling tuner should key on.

        Only meaningful under alltoall: the tuner must see pre-alltoall token
        counts so tactics cached during the no-alltoall warmup still apply at
        runtime. Without alltoall the kernel derives both from ``x`` itself.
        """
        if not enable_alltoall:
            return None, None
        if ctx.all_rank_num_tokens is not None:
            tuner_num_tokens = sum(ctx.all_rank_num_tokens)
        else:
            tuner_num_tokens = ctx.x.shape[0] * self.mapping.tp_size
        return tuner_num_tokens, self.routing_method.top_k
