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
"""MegaMoE — CuTeDSL ``cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce``.

``MegaMoECuteDSL`` is the single-step fused MegaMoE backend for SM100/SM103.
It wraps ``cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce``,
which fuses dispatch + FC1(NVFP4 GEMM + SwiGLU + requantize) + FC2(NVFP4 GEMM)
+ in-kernel top-k reduce into ONE launch. The kernel owns the cross-rank
NVLink exchange end-to-end, so this backend declares
``scheduler_kind = MoESchedulerKind.FUSED_COMM`` and ``ConfigurableMoE``
must not layer host-side comm on top.

Design mirrors :class:`MegaMoEDeepGemm`:

- Inherits directly from :class:`MoE`. The two-kernel ``CuteDslFusedMoE``
  flow is a *peer* backend; sharing implementation through inheritance
  would muddle the kernel boundary.
- Weight tensors, scale layout, and checkpoint loading are owned by
  :class:`NVFP4MegaMoECuteDSLMethod`. That method enforces the kernel's
  hard gate/up-interleave invariant at ``process_weights_after_loading``
  time so unsupported layouts crash loudly during model build instead
  of silently corrupting math inside the fused op.
- NVLink workspace is allocated once via :class:`MnnvlMemory` and shared
  across layers through a process-global cache keyed on (EP-PG identity,
  shape, top_k, max_tokens). Allocation is deferred to ``create_weights``
  so EPLB-derived attributes (``num_slots`` / ``expert_size_per_partition`` /
  ``slot_start``) are correct before sizing.
- ``quantize_input`` produces NVFP4 ``(x_packed_u8, x_sf_u8)``; the kernel
  reads them via TMA after staging into the NVLink slab.
- ``run_moe`` issues ONE fused launch and returns the per-token result
  read out of the rank-local slot of the monolithic NVLink output.

Variable per-rank token counts are honored: each call passes
``direct_topk_token_counts`` (int32, length ``ep_size``) so the kernel
knows how many rows each peer rank contributed in this chunk. The
``FusedCommMoEScheduler`` pipes ``all_rank_num_tokens`` to ``run_moe``
as a kwarg (see invariant 11 in that scheduler's docstring).
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..interface import MoE, MoESchedulerKind, MoEWeightLoadingMode, _warn_and_return
from ..quantization import NVFP4MegaMoECuteDSLMethod
from ..routing import BaseMoeRoutingMethod

__all__ = ["MegaMoECuteDSL"]


# Process-global NVLink workspace cache. The cached object holds the
# ``MnnvlMemory`` owner plus the sliced rank-strided tensor views so
# successive layers can reuse a single allocation. Sharing is safe only
# while MegaMoE layer forwards execute serially within a forward pass;
# concurrent forwards sharing a key would race on the same scratch
# buffers (see CHUNKING_DESIGN.md §4.2 for the same caveat on the DG
# SymmBuffer cache).
_MEGA_MOE_CUTE_DSL_WORKSPACE_CACHE: Dict[tuple, dict] = {}


# NVLink workspace region layout. Each region is a ``(ep_size, ...)`` view
# into one ``MnnvlMemory`` allocation; ``dim=0`` indexes peer ranks. The
# 128-byte alignment is the MnnvlMemory page granularity (also matches
# TMA's 128B vector load requirement).
_NVLINK_ALIGNMENT_BYTES = 128


def _aligned(nbytes: int) -> int:
    """Round ``nbytes`` up to the next ``_NVLINK_ALIGNMENT_BYTES`` boundary."""
    return (nbytes + _NVLINK_ALIGNMENT_BYTES - 1) & ~(_NVLINK_ALIGNMENT_BYTES - 1)


class MegaMoECuteDSL(MoE):
    """Single-step fused MegaMoE backend on the CuTeDSL NVFP4 monolithic kernel.

    Constraints (enforced in ``can_implement`` + ``__init__``):

    - SM100 / SM103 (Blackwell tensor-core path).
    - ``QuantAlgo.NVFP4`` only; weights flow through
      :class:`NVFP4MegaMoECuteDSLMethod`.
    - Gated activation (SwiGLU). Non-gated would silently diverge from
      the kernel's interleaved ``[up, gate]`` weight layout.
    - ``apply_router_weight_on_input=False``. The fused kernel applies
      router weights at combine time; pre-scaling would double-apply.
    - ``swiglu_gptoss_style=False``. The fused kernel does not implement
      the bias / alpha / limit variant.
    - DWDP (per-expert weight lists) is not supported.
    - Requires a live ``torch.distributed`` group spanning the EP world;
      single-GPU EP works only when MnnvlMemory falls back to a 1-rank
      allocation.
    """

    # Kernel template constants. Both are fixed by the CuTeDSL schedule
    # (Sm100BlockScaledMegaMoeBlackwellRunner); changing either requires a
    # kernel rebuild.
    _TILE_SIZE = 128
    _SCALING_VECTOR_SIZE = 16
    # In-kernel control barrier (NVLink rank-strided int64) used by the
    # fused reduce to synchronize producer FC2 writes against consumer
    # reduce reads. The kernel uses three disjoint regions per rank:
    #   words [0, 6)          : producer/consumer epoch state
    #   words [6, 10)         : grid-sync counter + phase
    #   words [10, 10+N)      : route_count[expert]
    #   words [10+N, 10+2N)   : route_base[expert]
    #   words [10+2N, 10+3N)  : route_cursor[expert]
    # where ``N == self.expert_size_per_partition`` (the kernel parameter
    # ``monolithic_direct_topk_num_local_experts``). See the staging /
    # materialize path in ``blockscaled_contiguous_mega_moe_fusion.py``
    # (3235-3262). The kernel also has an internal
    # ``assert control.size(1) >= 64`` lower bound (cute_dsl_custom_ops),
    # so the per-rank size must be ``max(64, 10 + 3 * N)``.
    _M6_BASE_CONTROL_WORDS = 64
    _M6_KERNEL_STATE_WORDS = 10
    _M6_PER_EXPERT_CONTROL_WORDS = 3

    # Kernel owns dispatch + GEMM1 + SwiGLU + GEMM2 + reduce via NVLink
    # symmetric memory; ConfigurableMoE must NOT layer host-side comm.
    scheduler_kind = MoESchedulerKind.FUSED_COMM

    # ------------------------------------------------------------------
    # Capability gating
    # ------------------------------------------------------------------
    # NVFP4 block-scale group is 16 elements, but the kernel's tile
    # boundaries (mma_tiler_l1 / mma_tiler_l2) require ``hidden_size`` and
    # ``intermediate_size`` to be 128-aligned. ``moe_test_utils`` uses the
    # same threshold; keep them in sync so production fallback selection
    # matches what unit tests actually validate.
    _NVFP4_SIZE_ALIGNMENT = 128

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        del kwargs  # forwards-compatible: ignore unknown capability hints.
        sm = get_sm_version()
        if sm not in (100, 103):
            return _warn_and_return(
                f"MegaMoECuteDSL requires SM100 or SM103 (Blackwell tensor "
                f"cores in cute_dsl_nvfp4_mega_moe_blackwell); got SM{sm}"
            )
        if dtype_activation != torch.bfloat16:
            return _warn_and_return(
                f"MegaMoECuteDSL only supports bfloat16 activations "
                f"(monolithic kernel output is hardcoded to bfloat16); "
                f"got {dtype_activation}"
            )
        if swiglu_gptoss_style:
            return _warn_and_return(
                "MegaMoECuteDSL does not support swiglu_gptoss_style "
                "(bias / custom alpha-beta-limit SwiGLU variant)"
            )
        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"MegaMoECuteDSL only supports QuantAlgo.NVFP4; got {quant_algo}"
            )
        # Shape gating: refuse misaligned models at the factory layer so
        # ``create_moe`` can fall back to CutlassFusedMoE instead of
        # tripping a kernel-level ``assert m % tile_size == 0`` or
        # ``assert n_l1 % (scaling_vector_size * 4 * 2) == 0`` at the
        # first forward (see cute_dsl_custom_ops.py:3654-3659).
        if hidden_size is not None and hidden_size % cls._NVFP4_SIZE_ALIGNMENT != 0:
            return _warn_and_return(
                f"MegaMoECuteDSL requires hidden_size % "
                f"{cls._NVFP4_SIZE_ALIGNMENT} == 0 (NVFP4 block-scale + "
                f"MMA tile alignment); got hidden_size={hidden_size}"
            )
        if intermediate_size is not None and intermediate_size % cls._NVFP4_SIZE_ALIGNMENT != 0:
            return _warn_and_return(
                f"MegaMoECuteDSL requires intermediate_size % "
                f"{cls._NVFP4_SIZE_ALIGNMENT} == 0 (NVFP4 block-scale + "
                f"MMA tile alignment); got intermediate_size={intermediate_size}"
            )
        # The fused monolithic op is registered lazily when
        # ``cute_dsl_custom_ops`` imports. Missing it means the build does
        # not include the Blackwell CuTeDSL kernels (e.g. CPU-only wheel).
        op = getattr(
            torch.ops.trtllm,
            "cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce",
            None,
        )
        if op is None:
            return _warn_and_return(
                "MegaMoECuteDSL requires torch.ops.trtllm."
                "cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce. "
                "This build does not include the CuTeDSL Blackwell mega op."
            )
        return True, None

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
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
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
        **kwargs,
    ) -> None:
        del kwargs  # accept forwards-compatible factory keys without crashing.
        # Reject incompatible knobs BEFORE super().__init__ so we never
        # spend module-build cycles allocating weights for a config the
        # fused mega op cannot honor.
        if apply_router_weight_on_input:
            raise ValueError(
                "MegaMoECuteDSL does not support apply_router_weight_on_input. "
                "cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce "
                "applies routing weights at combine time; pre-scaling would "
                "double-apply them."
            )
        if activation_type != ActivationType.Swiglu:
            raise ValueError(
                f"MegaMoECuteDSL only supports ActivationType.Swiglu "
                f"(got {activation_type}); the fused monolithic kernel "
                f"hardcodes SwiGLU as the activation between FC1 and FC2."
            )

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
            layer_idx=layer_idx,
            activation_type=activation_type,
            init_load_balancer=init_load_balancer,
        )

        # Topology checks: MegaMoECuteDSL is EP-only. The fused kernel
        # routes by ``slot_id`` and assumes a single shard per rank;
        # cluster_size>1 or moe_tp_size>1 would split the per-expert
        # weight slab in ways the kernel does not understand.
        assert self.tp_size == 1, (
            f"MegaMoECuteDSL is EP-only (moe_tp_size=1); got tp_size={self.tp_size}"
        )
        assert self.cluster_size == 1, (
            f"MegaMoECuteDSL assumes cluster_size=1; got cluster_size={self.cluster_size}"
        )
        # ``num_slots`` (= kernel ``num_experts`` template parameter)
        # must divide evenly across EP ranks because each rank owns
        # ``num_slots // ep_size`` slots and the fused kernel reads a
        # contiguous slab per rank.
        if self.num_slots % max(self.ep_size, 1) != 0:
            raise ValueError(
                f"MegaMoECuteDSL requires num_slots ({self.num_slots}) "
                f"divisible by ep_size ({self.ep_size})."
            )

        # ADP semantics: the fused monolithic kernel subsumes cross-rank
        # token dispatch into its NVLink exchange. When EP spans every
        # rank that may carry tokens (``ep_size == parallel_size``), no
        # outer allgather / reducescatter is needed. The strict-subset
        # case (ADP > EP) is not yet supported; pre/post wrappers would
        # need to materialize the missing exchange.
        if self.use_dp and self.parallel_size > 1:
            assert self.ep_size == self.parallel_size, (
                f"MegaMoECuteDSL with enable_attention_dp=True requires "
                f"ep_size == parallel_size (got ep_size={self.ep_size}, "
                f"parallel_size={self.parallel_size}). ADP > EP is not "
                f"supported."
            )

        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.without_comm = without_comm

        # ``top_k`` is fixed at module build; the kernel reads
        # ``combine_output_top_k`` as a template parameter.
        self.top_k = int(self.routing_method.experts_per_token)

        # Buffer sizing matches MegaMoEDeepGemm: a single process-level
        # pool sized to the worst-case per-rank chunk-size serves every
        # MoE layer (layers run serially per forward).
        self.max_num_tokens = int(
            getattr(model_config, "moe_max_num_tokens", 0)
            or getattr(model_config, "max_num_tokens", 0)
            or 4096
        )
        # The kernel reads ``num_pool_tokens`` aligned up to ``tile_size``.
        # Pre-compute so workspace sizing matches kernel invariants.
        # Each local expert reserves up to one full tile (padded routes).
        # Worst case: many active experts each with few routes. Pool must cover:
        #   num_local_experts * TILE_SIZE (tile padding) + total_routes (actual data)
        # The kernel uses this pool both as expert-major routing buffer and FC1 scratch.
        self._num_max_pool_tokens = self._align_up(
            self.ep_size * self.top_k * self.max_num_tokens
            + self.expert_size_per_partition * self._TILE_SIZE,
            self._TILE_SIZE,
        )

        # Resolve the EP ProcessGroup at module construction. Creating a
        # group at forward time would be collective on a non-synchronous
        # call stack and deadlock under PP / layer-skip; building during
        # ``__init__`` runs in the model-build lockstep window.
        self._ep_pg = self._resolve_ep_pg()

        # NVLink workspace state. Actual allocation is deferred to
        # ``create_weights`` so EPLB-derived ``num_slots`` /
        # ``expert_size_per_partition`` / ``slot_start`` are correct.
        self._nvlink: Optional[dict] = None
        self._workspace_key: Optional[tuple] = None
        # Monotonic per-instance counter for the kernel's in-kernel
        # producer/consumer barrier. The kernel uses ``monolithic_epoch``
        # to distinguish consecutive launches without resetting the
        # control region every call; one wrap-around per ~2^63 calls is
        # safe to ignore.
        self._monolithic_epoch = 0

        # Pre-allocated per-rank token-count buffers. Mirrors the
        # MegaMoEDeepGemm SymmBuffer pattern: a stable ``(ep_size,)`` int32
        # tensor pair (pinned-host source + device destination) updated via
        # ``copy_`` each call. Going through ``torch.tensor(list, device=)``
        # instead would do a fresh CUDA allocation per call, which CUDA
        # Graph capture forbids (it shows up as
        # ``cudaErrorStreamCaptureInvalidated`` / ``Offset increment outside
        # graph capture``). Allocation is deferred to ``run_moe`` so it
        # picks up the same device the rest of the workspace uses.
        self._token_counts_cpu: Optional[torch.Tensor] = None
        self._token_counts_gpu: Optional[torch.Tensor] = None

        # Weight tensors are owned by the quant method (mirrors the
        # MegaMoEDeepGemm pattern).
        self.quant_method: Optional[NVFP4MegaMoECuteDSLMethod] = None
        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    @staticmethod
    def _align_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    # ------------------------------------------------------------------
    # Load-balancer / scheduler hooks
    # ------------------------------------------------------------------
    def _supports_load_balancer(self) -> bool:
        # The monolithic kernel routes by ``slot_id`` (range [0, num_slots))
        # once the per-expert weight slab and ``local_expert_offset`` match.
        # Dynamic EPLB migrates the transformed NVFP4 tensors registered by
        # the quantization method, not the raw checkpoint-layout weights.
        return True

    def validate_configurable_moe(self, moe) -> None:
        """Reject configurations the fused kernel cannot honor.

        ``moe`` is the owning ``ConfigurableMoE``; ``num_slots`` /
        ``ep_size`` / load-balancer flags are populated by
        ``MoE.__init__`` before ``validate_backend`` runs, so they're
        stable here.
        """
        if moe.num_slots % moe.ep_size != 0:
            raise ValueError(
                f"MegaMoECuteDSL requires num_slots ({moe.num_slots}) "
                f"divisible by ep_size ({moe.ep_size})."
            )
        # The monolithic kernel reads a single expert-slab per rank; DWDP
        # would hand us per-expert tensor lists that the NVFP4 weight
        # method cannot pack into the contiguous ``w3_w1_weight`` /
        # ``w2_weight`` slabs.
        if getattr(moe, "dwdp_manager", None) is not None:
            raise ValueError(
                "MegaMoECuteDSL does not support DWDP (per-expert weight "
                "lists). Disable dwdp_config to use this backend."
            )
        # ``layer_load_balancer`` is the dynamic EPLB controller; static
        # EPLB (num_slots != num_experts) is still fine.
        balancer = getattr(moe, "layer_load_balancer", None)
        if balancer is not None and getattr(balancer, "is_dynamic_balancer", False):
            raise ValueError(
                "MegaMoECuteDSL does not yet support dynamic EPLB: the "
                "fused kernel reads NVFP4 weights via in-kernel TMA "
                "descriptors that would race with EPLB migration. Use "
                "static EPLB (num_slots > num_experts but fixed mapping) "
                "or pick a different backend."
            )

    # ------------------------------------------------------------------
    # EP process-group resolution (no collective at forward time)
    # ------------------------------------------------------------------
    def _resolve_ep_pg(self):
        """Return the torch.distributed ProcessGroup for the EP sub-world.

        Mirrors :meth:`MegaMoEDeepGemm._resolve_ep_pg` so both fused-comm
        MegaMoE backends pick the same group: prefer
        ``mapping.moe_ep_group_pg`` (built once at Mapping init under the
        DeviceMesh / Ray topology) and fall back to ``dist.group.WORLD``
        only when EP spans the world. Never call ``dist.new_group`` from
        here -- that would be a collective on the model-build call stack
        and deadlock under PP / layer-skip.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "MegaMoECuteDSL requires torch.distributed to be "
                "initialized before module construction (mpirun or Ray)."
            )
        try:
            pg = self.mapping.moe_ep_group_pg
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoECuteDSL] layer={self.layer_idx} using "
                f"mapping.moe_ep_group_pg (DeviceMesh path)"
            )
            return pg
        except (NotImplementedError, AttributeError):
            pass
        world_size = dist.get_world_size()
        if self.ep_size == world_size:
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoECuteDSL] layer={self.layer_idx} using "
                f"dist.group.WORLD (EP == world_size == {world_size})"
            )
            return dist.group.WORLD
        raise RuntimeError(
            f"MegaMoECuteDSL: cannot resolve EP ProcessGroup. The current "
            f"mapping does not expose ``moe_ep_group_pg`` and EP "
            f"({self.ep_size}) is a strict subset of world "
            f"({world_size}). Use DeviceMeshTopology (TLLM_DISABLE_MPI=1) "
            f"so the EP PG is constructed once at Mapping init, or set "
            f"ep_size == world_size."
        )

    # ------------------------------------------------------------------
    # NVLink workspace allocation (collective resource)
    # ------------------------------------------------------------------
    def _control_words(self) -> int:
        """Per-rank ``monolithic_control`` size in int64 words.

        The kernel reads/writes three per-expert tables (route_count,
        route_base, route_cursor) starting at word 10. With
        ``N = expert_size_per_partition`` it touches words up to
        ``10 + 3 * N - 1``. The kernel also asserts a 64-word lower bound
        for its kernel-state region. Size = ``max(64, 10 + 3 * N)``.
        """
        return max(
            self._M6_BASE_CONTROL_WORDS,
            self._M6_KERNEL_STATE_WORDS
            + self._M6_PER_EXPERT_CONTROL_WORDS * self.expert_size_per_partition,
        )

    def _build_nvlink_layout(self) -> List[tuple]:
        """Compute the rank-strided NVLink workspace region table.

        Each entry is ``(name, dtype, shape_per_rank)``. Regions are laid
        out at the offsets returned by ``_alloc_nvlink_workspace`` so the
        kernel sees one ``(ep_size, *shape_per_rank)`` view per region.
        """
        hidden = self.hidden_size
        hidden_packed = hidden // 2  # NVFP4: two FP4 values per byte.
        scale_k = hidden // self._SCALING_VECTOR_SIZE
        max_tok = self.max_num_tokens
        top_k = self.top_k
        return [
            # Staged inputs (kernel.copies local -> rank-r slot).
            ("direct_topk_input", torch.uint8, (max_tok, hidden_packed)),
            ("direct_topk_input_scale", torch.uint8, (max_tok, scale_k)),
            ("direct_topk_idx", torch.int64, (max_tok, top_k)),
            ("direct_topk_scales", torch.float32, (max_tok, top_k)),
            # Cross-rank FC2 combine intermediate. Shape is per-rank;
            # the kernel's ``combine_output_rank_stride_elements`` is
            # this region's stride along ``dim=0`` of the (ep, ...) view.
            ("combine_buffer", torch.bfloat16, (top_k * max_tok, hidden)),
            # Final per-rank MoE output (in-kernel reduce target).
            ("monolithic_output", torch.bfloat16, (max_tok, hidden)),
            # In-kernel control barrier (kernel-internal state words
            # plus host-zeroed scratch). Stays in NVLink because the
            # kernel reads peer ranks' progress. Size must cover the
            # per-expert route tables: 10 kernel-state words + 3 * N.
            ("monolithic_control", torch.int64, (self._control_words(),)),
        ]

    def _alloc_nvlink_workspace(self) -> None:
        """Allocate (or fetch from cache) the NVLink rank-strided workspace.

        For ``ep_size > 1`` allocation runs ``MnnvlMemory(mapping, nbytes)``
        which executes a collective IPC handle exchange across EP ranks.
        This must run in the model-build lockstep window; calling it
        from ``run_moe`` would deadlock under PP / layer-skip and would
        also fail under CUDA graph capture (host-side IPC operation).
        ``create_weights`` is the right call site because ConfigurableMoE
        invokes it on every EP rank in lockstep after EPLB sync.

        For ``ep_size == 1`` (single-GPU unit-test path) MnnvlMemory is
        unavailable because it depends on an MPI communicator that
        pytest does not initialize. We fall back to a plain CUDA
        allocation with a leading ``ep_size=1`` axis; the rank-strided
        contract (``view[0]`` == this rank's slot) is trivially preserved
        because there is only one rank.
        """
        if self._nvlink is not None:
            return
        layout = self._build_nvlink_layout()
        key = (
            id(self._ep_pg),
            self.num_experts,
            self.num_slots,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.ep_size,
            self.activation_type,
        )
        cached = _MEGA_MOE_CUTE_DSL_WORKSPACE_CACHE.get(key)
        if cached is not None:
            self._nvlink = cached
            self._workspace_key = key
            return

        # Compute the layout offsets. Each region is sized to one rank's
        # slab (the ``dim=0`` peer stride). ``MnnvlMemory`` will replicate
        # the per-rank slab across ``ep_size`` peers and return a 2D
        # ``(ep_size, bytes_per_rank)`` uint8 tensor we slice up.
        offsets: List[int] = []
        cursor = 0
        for _, dtype, shape in layout:
            offsets.append(cursor)
            nbytes_per_rank = 1
            for d in shape:
                nbytes_per_rank *= d
            nbytes_per_rank *= torch.empty((), dtype=dtype).element_size()
            cursor += _aligned(nbytes_per_rank)
        total_bytes_per_rank = cursor

        if self.ep_size > 1:
            MnnvlMemory.initialize()
            mnnvl_mem = MnnvlMemory(self.mapping, total_bytes_per_rank)
            flat = mnnvl_mem.as_torch_strided_tensor(torch.uint8)
        else:
            # Single-rank fallback: no MPI, no peer exchange. Plain CUDA
            # allocation with the same ``(ep_size, nbytes)`` shape so
            # the slicing logic below is identical to the multi-rank
            # path.
            mnnvl_mem = None
            flat = torch.empty(
                (1, total_bytes_per_rank), dtype=torch.uint8, device=torch.device("cuda")
            )
        # ``flat`` is ``(ep_size, total_bytes_per_rank)`` uint8. Slice
        # each region as a flat-uint8 view then reinterpret to the
        # requested dtype and shape. ``view`` requires the slice to be
        # contiguous, which holds because the layout is in declared
        # order.
        regions: Dict[str, torch.Tensor] = {}
        for (name, dtype, shape), off in zip(layout, offsets):
            nbytes_per_rank = 1
            for d in shape:
                nbytes_per_rank *= d
            nbytes_per_rank *= torch.empty((), dtype=dtype).element_size()
            region_u8 = flat[:, off : off + nbytes_per_rank]
            regions[name] = region_u8.view(dtype).view(self.ep_size, *shape)

        # Zero the control barrier once at allocation. The kernel reads
        # the first 6 state words at launch time; uninitialized memory
        # there would non-deterministically skip producer/consumer
        # arrival counts.
        regions["monolithic_control"].zero_()

        cached = {
            "mnnvl_mem": mnnvl_mem,
            "flat": flat,
            "nbytes_per_rank": total_bytes_per_rank,
            **regions,
        }
        _MEGA_MOE_CUTE_DSL_WORKSPACE_CACHE[key] = cached
        self._nvlink = cached
        self._workspace_key = key

        log_fn = logger.info if self.layer_idx == 0 else logger.debug
        backing = "MnnvlMemory" if mnnvl_mem is not None else "local CUDA"
        log_fn(
            f"[MegaMoECuteDSL] layer={self.layer_idx} allocated NVLink "
            f"workspace ({backing}): "
            f"{total_bytes_per_rank / 2**30:.3f} GiB per rank "
            f"(ep_size={self.ep_size}, max_tokens={self.max_num_tokens}, "
            f"top_k={self.top_k}, hidden={self.hidden_size})."
        )

    # ------------------------------------------------------------------
    # Weight lifecycle (delegated to the quant method)
    # ------------------------------------------------------------------
    def _get_quant_method(self) -> NVFP4MegaMoECuteDSLMethod:
        if (
            self.quant_config is None
            or not self.quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
            or not self.quant_config.layer_quant_mode.has_nvfp4()
        ):
            raise NotImplementedError(
                f"MegaMoECuteDSL requires an NVFP4 quant config; got {self.quant_config!r}."
            )
        return NVFP4MegaMoECuteDSLMethod()

    def create_weights(self) -> None:
        if self._weights_created:
            return
        # Allocate the NVLink workspace here (lazily) rather than from
        # ``__init__`` because ConfigurableMoE only syncs the EPLB-derived
        # attributes (``num_slots``, ``expert_size_per_partition``,
        # ``slot_start``, ...) onto the backend AFTER backend ``__init__``
        # returns, just before calling ``backend.create_weights()``. The
        # workspace sizing in ``__init__`` would otherwise use the
        # placeholder ``num_slots = num_experts`` and mis-size the slabs
        # for EPLB layers. ConfigurableMoE drives this on every EP rank
        # in lockstep, preserving the MnnvlMemory rendezvous safety
        # invariant.
        self._alloc_nvlink_workspace()
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        self._weights_created = True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None:
        if self.quant_method is None:
            self.create_weights()
        assert self._weights_created
        assert len(weights) == 1, (
            "MegaMoECuteDSL expects a single weight dict (one MoE layer per "
            f"call); got {len(weights)}."
        )
        weights = weights[0]
        kargs: Dict = {}
        # ``allow_partial_loading`` is only honored by quant methods whose
        # ``load_weights`` signature declares it; detect at call time to
        # match how ``CutlassFusedMoE.load_weights`` delegates.
        if "allow_partial_loading" in inspect.getfullargspec(self.quant_method.load_weights).args:
            kargs["allow_partial_loading"] = allow_partial_loading
        self.quant_method.load_weights(self, weights, self.weight_loading_mode, **kargs)

    def post_load_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.quant_method.post_load_weights(self)

    # ------------------------------------------------------------------
    # MoE-contract methods
    # ------------------------------------------------------------------
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        *,
        post_quant_comm: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 -> NVFP4 packed FP4x2 + per-token uint8 block scales.

        ``FusedCommMoEScheduler`` invariant 1 rejects ``Fp4QuantizedTensor``
        at the scheduler level, but we accept the kwarg for parity with
        other backends.

        Returns:
            ``(x_packed, x_sf)`` where ``x_packed`` is
            ``uint8 (num_tokens, hidden // 2)`` (each byte holds two
            packed FP4 values) and ``x_sf`` is
            ``uint8 (num_tokens, hidden // scaling_vector_size)`` (one
            byte per scaling group). Both are unswizzled per-token; the
            fused kernel re-swizzles internally.
        """
        del post_quant_comm, kwargs  # MegaMoE has no host-side comm timing.
        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "MegaMoECuteDSL.quantize_input expects BF16 input; "
                "Fp4QuantizedTensor pre-quantized payloads are rejected "
                "by FusedCommMoEScheduler invariant 1."
            )
        x_row = x.shape[0]
        # ``fp4_quantize(x, global_sf, vec_size, is_sf_swizzled=False,
        # is_pad_to_128=False)`` matches the per-token (non-swizzled)
        # layout consumed by the monolithic kernel via TMA.
        x_q, x_sf = torch.ops.trtllm.fp4_quantize(
            x, self.fc31_input_scale, self._SCALING_VECTOR_SIZE, False, False
        )
        # ``fp4_quantize`` returns x_q as 2D uint8 ``(rows, hidden//2)``
        # directly; the SF is returned as a flat 1D tensor that we view
        # as ``(rows, scale_k)`` so downstream copy_ into the NVLink slab
        # matches the (max_tok, scale_k) slice trivially.
        if x_sf is not None:
            x_sf = x_sf.view(x_row, -1)
        return x_q, x_sf

    def _empty_quantized_input(
        self,
        num_tokens: int,
        *,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Placeholder for ``quantize_input`` output when ``num_tokens == 0``.

        ``FusedCommMoEScheduler`` calls this for zero-token chunks so the
        per-chunk path still hands ``run_moe`` shape-correct tensors
        without invoking the NVFP4 quantizer on an empty input.

        The contract must match ``quantize_input`` byte-for-byte:
        ``x_packed`` is ``uint8 (num_tokens, hidden // 2)`` and ``x_sf``
        is ``uint8 (num_tokens, hidden // scaling_vector_size)``.
        """
        if device is None:
            device = torch.device("cuda")
        if num_tokens != 0:
            raise NotImplementedError(
                "MegaMoECuteDSL._empty_quantized_input only supports the "
                "zero-token placeholder shape used by FusedCommMoEScheduler; "
                f"got num_tokens={num_tokens}. Use quantize_input(x) for "
                "non-empty activations."
            )
        hidden = self.hidden_size
        x_packed = torch.empty((0, hidden // 2), dtype=torch.uint8, device=device)
        x_sf = torch.empty(
            (0, hidden // self._SCALING_VECTOR_SIZE), dtype=torch.uint8, device=device
        )
        return x_packed, x_sf

    # ------------------------------------------------------------------
    # run_moe
    # ------------------------------------------------------------------
    def _normalize_all_rank_num_tokens(
        self,
        all_rank_num_tokens: Optional[Sequence[int]],
        *,
        local_num_tokens: int,
    ) -> List[int]:
        """Materialize a length-``ep_size`` list of per-rank token counts.

        ``FusedCommMoEScheduler`` always passes the per-chunk counts when
        chunk metadata exists (which is the live path under ADP /
        multi-chunk). For the single-rank standalone path (``had_meta=False``,
        used by unit tests), the scheduler passes ``None`` and we fall
        back to ``[local_num_tokens]``.
        """
        if all_rank_num_tokens is None:
            assert self.ep_size == 1, (
                f"MegaMoECuteDSL needs all_rank_num_tokens for EP > 1; "
                f"got ep_size={self.ep_size} but scheduler passed None."
            )
            return [local_num_tokens]
        counts = [int(v) for v in all_rank_num_tokens]
        if len(counts) != self.ep_size:
            raise ValueError(
                f"MegaMoECuteDSL: all_rank_num_tokens length "
                f"({len(counts)}) != ep_size ({self.ep_size})"
            )
        local_rank = self.mapping.moe_ep_rank
        if counts[local_rank] != local_num_tokens:
            raise ValueError(
                f"MegaMoECuteDSL: local rank token count mismatch "
                f"(all_rank_num_tokens[{local_rank}]={counts[local_rank]} "
                f"vs x.shape[0]={local_num_tokens})"
            )
        if max(counts) > self.max_num_tokens:
            raise ValueError(
                f"MegaMoECuteDSL: per-rank token count "
                f"({max(counts)}) exceeds workspace capacity "
                f"({self.max_num_tokens}). Raise moe_max_num_tokens."
            )
        return counts

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        *,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[Sequence[int]] = None,
        **unused_kwargs,
    ) -> torch.Tensor:
        """Run the fused monolithic dispatch + FC1 + SwiGLU + FC2 + reduce.

        Contract:
        - ``x`` / ``x_sf`` come from :meth:`quantize_input` or
          :meth:`_empty_quantized_input`; both are local-rank, contiguous,
          and shaped ``(num_tokens, hidden//2 or scale_k)`` uint8.
        - ``token_selected_experts`` carries slot ids (range
          ``[0, num_slots)``), cast to int64 inside this method to match
          the kernel staging contract.
        - ``token_final_scales`` carries the routing weights as float32.
        - ``all_rank_num_tokens`` is forwarded by ``FusedCommMoEScheduler``
          (invariant 11); it is required for ``ep_size > 1`` because the
          kernel reads ``direct_topk_token_counts`` to know where each
          peer's tokens end inside the NVLink slab.

        Returns:
            ``(num_tokens, hidden)`` bfloat16 tensor with the per-token
            MoE output for the local rank (a view into the rank-local
            slot of the NVLink monolithic output, cast / cloned to the
            requested ``output_dtype`` to detach from the workspace).
        """
        assert not unused_kwargs, (
            f"MegaMoECuteDSL.run_moe got unexpected kwargs: {sorted(unused_kwargs)}"
        )
        if self._nvlink is None:
            raise RuntimeError(
                "MegaMoECuteDSL run_moe called before create_weights() "
                "allocated the NVLink workspace."
            )
        if x_sf is None:
            raise ValueError("MegaMoECuteDSL requires x_sf from quantize_input.")
        if output_dtype is None:
            output_dtype = self.dtype or torch.bfloat16

        device = x.device
        local_rank = self.mapping.moe_ep_rank
        local_num_tokens = int(x.shape[0])
        counts = self._normalize_all_rank_num_tokens(
            all_rank_num_tokens, local_num_tokens=local_num_tokens
        )

        # NVLink rank-strided views. Index ``dim=0`` selects the peer
        # rank's slot; the kernel reads / writes across peers via this
        # slab. ``direct_topk_*`` are uint8 / int64 / float32 staged
        # inputs, ``combine_buffer`` is the FC2 -> reduce hand-off,
        # ``monolithic_output`` is the in-kernel reduce target.
        nv = self._nvlink
        direct_topk_input = nv["direct_topk_input"]
        direct_topk_input_scale = nv["direct_topk_input_scale"]
        direct_topk_idx = nv["direct_topk_idx"]
        direct_topk_scales = nv["direct_topk_scales"]
        combine_buffer = nv["combine_buffer"]
        monolithic_output = nv["monolithic_output"]
        monolithic_control = nv["monolithic_control"]

        # Per-rank int32 token counts on device. We mirror the
        # MegaMoEDeepGemm pattern (see ``mega_moe_deepgemm.py:_symm_buffer``):
        # a single instance-level ``(ep_size,)`` device tensor backed by a
        # pinned-host source, updated via ``copy_`` each call so the
        # allocations are stable across CUDA Graph captures and replays.
        # The earlier ``torch.tensor(counts, device=...)`` form did a fresh
        # CUDA malloc per call which the graph captor cannot record (it
        # raises ``cudaErrorStreamCaptureInvalidated`` / ``Offset increment
        # outside graph capture``). Replays read whatever value was last
        # written into the pinned-host buffer, which is the standard
        # dynamic-input CUDA Graph idiom.
        if self._token_counts_cpu is None:
            # ``device='cpu'`` must be explicit: at run_moe time the caller
            # is inside a ``with torch.device(f'cuda:{rank}')`` context,
            # and ``pin_memory=...`` without an explicit device tries to
            # pin a CUDA allocation, which raises "Only dense CPU tensors
            # can be pinned". ``prefer_pinned()`` is the project-wide
            # gate that disables pinning under Confidential Compute (see
            # tensorrt_llm/_utils.py), enforced by the pinned-memory
            # policy pre-commit hook.
            self._token_counts_cpu = torch.empty(
                (self.ep_size,),
                dtype=torch.int32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self._token_counts_gpu = torch.empty((self.ep_size,), dtype=torch.int32, device=device)
        cpu_buf = self._token_counts_cpu
        for i, c in enumerate(counts):
            cpu_buf[i] = c
        self._token_counts_gpu.copy_(cpu_buf, non_blocking=True)
        token_counts_tensor = self._token_counts_gpu

        # Local-staging tensors handed to the kernel. The kernel copies
        # them into ``direct_topk_*`` rank-r slot internally (because we
        # set ``monolithic_direct_topk_stage_inputs=True``); we only need
        # to ensure dtype / contiguity matches the kernel's strict
        # validation in ``cute_dsl_custom_ops`` (line ~4801).
        local_input = x.contiguous().view(torch.uint8)
        local_input_scale = x_sf.contiguous().view(torch.uint8)
        local_topk_idx = token_selected_experts.contiguous().to(torch.int64)
        local_topk_scales = token_final_scales.contiguous().to(torch.float32)

        # Reset host-managed control words. Kernel state words [0, 6)
        # are managed by the kernel itself across the producer/consumer
        # hand-off; everything from word 6 onward (grid-sync counters +
        # per-expert route tables sized ``_control_words()``) must be
        # zero at launch. Slicing ``[6:]`` covers the full dynamic
        # remainder regardless of ``expert_size_per_partition``.
        monolithic_control[local_rank, 6:].zero_()

        # Monotonic epoch tag for the in-kernel barrier. Each successive
        # ``run_moe`` call needs a unique value so old writes from the
        # previous launch are not mistaken for current-epoch data.
        self._monolithic_epoch += 1
        epoch = self._monolithic_epoch

        # Per-call scratch tensors. Sized to ``num_max_pool_tokens`` which
        # is the tile-aligned upper bound on permuted tokens that pass
        # through FC1's expert-major layout.
        num_pool_tokens = self._num_max_pool_tokens
        interm_per_partition = self.intermediate_size_per_partition
        scale_k = self.hidden_size // self._SCALING_VECTOR_SIZE
        tile_count = num_pool_tokens // self._TILE_SIZE

        monolithic_input_tensor = torch.empty(
            (num_pool_tokens, self.hidden_size // 2),
            dtype=torch.float4_e2m1fn_x2,
            device=device,
        )
        monolithic_input_scale = torch.empty(
            (num_pool_tokens, scale_k),
            dtype=torch.uint8,
            device=device,
        )
        tile_idx_to_group_idx = torch.empty((tile_count,), dtype=torch.int32, device=device)
        tile_idx_to_mn_limit = torch.empty((tile_count,), dtype=torch.int32, device=device)
        token_id_mapping = torch.empty((num_pool_tokens,), dtype=torch.int32, device=device)
        output_mapping = torch.empty((num_pool_tokens,), dtype=torch.int32, device=device)
        num_non_exiting_tiles = torch.zeros((1,), dtype=torch.int32, device=device)
        pool_token_final_scales = torch.empty(
            (num_pool_tokens, 1), dtype=torch.float32, device=device
        )
        monolithic_pool_tensor = torch.empty(
            (num_pool_tokens, interm_per_partition // 2),
            dtype=torch.float4_e2m1fn_x2,
            device=device,
        )
        monolithic_pool_sf_tensor = torch.empty(
            (num_pool_tokens * interm_per_partition // self._SCALING_VECTOR_SIZE,),
            dtype=torch.uint8,
            device=device,
        )
        monolithic_l2_arrival_mask = torch.empty((tile_count,), dtype=torch.int64, device=device)

        # Weights are owned by NVFP4MegaMoECuteDSLMethod (registered on
        # ``self`` in create_weights). Pass them through the dtype the
        # kernel expects: weights are NVFP4-packed (FP4x2 view) and
        # weight scales are uint8 view of the swizzled tile-aligned SF
        # tensor.
        weight_l1 = self.w3_w1_weight.view(torch.float4_e2m1fn_x2)
        weight_scale_l1 = self.quant_scales.fc1_weight_block.view(torch.uint8)
        weight_l2 = self.w2_weight.view(torch.float4_e2m1fn_x2)
        weight_scale_l2 = self.quant_scales.fc2_weight_block.view(torch.uint8)

        # The fused op mutates ``direct_topk_*`` (staging), ``combine_buffer``
        # (FC2 hand-off), ``monolithic_output`` (final reduce), and a
        # handful of scratch tensors; see the ``mutates_args`` tuple on
        # the custom-op registration in cute_dsl_custom_ops.py.
        torch.ops.trtllm.cute_dsl_nvfp4_mega_moe_blackwell_monolithic_direct_topk_reduce(
            direct_topk_input,
            direct_topk_input_scale,
            direct_topk_idx,
            direct_topk_scales,
            token_counts_tensor,
            weight_l1,
            weight_scale_l1,
            self.quant_scales.fc1_global,
            weight_l2,
            weight_scale_l2,
            self.quant_scales.fc2_global,
            self.fc2_input_scale,
            combine_buffer,
            monolithic_output,
            monolithic_control,
            num_pool_tokens,
            self.num_slots,
            self.expert_size_per_partition,
            self.slot_start,
            self._TILE_SIZE,
            self._SCALING_VECTOR_SIZE,
            self.ep_size,
            self.top_k,
            self.max_num_tokens,
            combine_buffer.stride(0),
            local_rank,
            local_num_tokens,
            epoch,
            True,  # monolithic_direct_topk_stage_inputs
            True,  # monolithic_direct_topk_materialize
            local_input,
            local_input_scale,
            local_topk_idx,
            local_topk_scales,
            monolithic_input_tensor,
            monolithic_input_scale,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            output_mapping,
            num_non_exiting_tiles,
            pool_token_final_scales,
            monolithic_pool_tensor,
            monolithic_pool_sf_tensor,
            monolithic_l2_arrival_mask,
        )

        # Detach the rank-local result slice from the NVLink workspace so
        # the caller can hold onto it past the next kernel launch (the
        # next launch would overwrite the workspace slot in place).
        result = monolithic_output[local_rank, :local_num_tokens].clone()
        if output_dtype is not torch.bfloat16:
            result = result.to(output_dtype)
        return result
