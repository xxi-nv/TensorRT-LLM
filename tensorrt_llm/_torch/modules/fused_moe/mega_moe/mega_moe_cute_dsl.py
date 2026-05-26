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
"""MegaMoE CuteDSL NVFP4 backend.

ConfigurableMoE-compatible MoE backend wrapping the ported
``Sm100MegaMoEKernel`` (fused dispatch + FC1 + activation + FC2 +
combine) from
``tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4``. The kernel is
invoked through the standard CuteDSL TunableRunner / torch op pattern;
the runner + op live in
``tensorrt_llm/_torch/custom_ops/cute_dsl_megamoe_custom_op.py``. This
file only owns:

  * capability gating (``can_implement``)
  * lifecycle hooks (``__init__`` / ``create_weights`` /
    ``load_weights`` / ``post_load_weights`` /
    ``validate_configurable_moe``)
  * EP process group resolution
  * BF16 -> NVFP4 activation quantization (``quantize_input``)
  * ``run_moe`` boundary: stage activation + topk into the kernel ABI,
    build the ``MegaMoECuteDslWeightView`` from the quant method, call
    ``torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell``, sum the
    per-topk axis (form A), return ``(T, hidden)`` output.

``run_moe`` is a single unified path for both topologies. Only the
SOURCE of the kernel's input/output buffers branches on ``ep_size``:

  * ``ep_size == 1``: local CUDA tensors (cudaMalloc). No
    ``torch.distributed`` dependency, no rendezvous, no cuMem VMM
    overhead. ``peer_offsets = [0]`` collapses the kernel's
    ``peer_rank_ptr_mapper.map(local_addr, 0, off) == local_addr +
    off`` to a self-mapped pointer (NVSHMEM degenerate convention).
  * ``ep_size > 1``: regions carved out of the build-time-rendezvous'd
    :class:`~tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op.MegaMoeSymmMemProvider`
    symmetric buffer; ``peer_offsets[r] = peer_base[r] - local_base``
    enables in-kernel cross-GPU NVLink load/store via
    ``peer_rank_ptr_mapper.map``.

``_acquire_buffers`` is the only branch point; staging, kernel launch,
and the host-side top-k reduction are identical across topologies.

The v1 hard gates documented in ``MEGAMOE_CUTEDSL_DESIGN.md`` remain:

  * Multi-rank execution requires the cuMem symmetric-memory provider
    to have completed its rendezvous at ``create_weights`` time
    (``self._symm_provider`` non-None); ``run_moe`` raises
    ``MegaMoeCuteDslUnavailable`` otherwise with an actionable message
    pointing at Ray / DeviceMesh / mpirun.
  * Checkpoints with per-expert / per-layer NVFP4 scales != 1.0 are
    rejected by ``NVFP4MegaMoECuteDslMethod.post_load_weights`` until
    the ported kernel ABI is extended to thread ``fc31_alpha`` /
    ``fc2_alpha`` / ``fc2_input_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..interface import MoE, MoESchedulerKind, MoEWeightLoadingMode
from ..quantization import NVFP4MegaMoECuteDslMethod
from ..routing import BaseMoeRoutingMethod

__all__ = [
    "MegaMoECuteDsl",
    "MegaMoeCuteDslUnavailable",
    "MegaMoECuteDslWeightView",
    "is_megamoe_cute_dsl_runtime_available",
]


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


class MegaMoeCuteDslUnavailable(RuntimeError):
    """Raised when the active environment cannot import the symbols required by
    the ported ``Sm100MegaMoEKernel`` (cu13 Cutlass DSL + cute_nvgpu MMA
    atoms / cutlass._mlir APIs used by sym_buffer)."""


_RUNTIME_PROBE_CACHE: Optional[Union[bool, str]] = None


def is_megamoe_cute_dsl_runtime_available() -> Tuple[bool, Optional[str]]:
    """Return whether the CUDA 13 Cutlass DSL runtime exposes all symbols the
    ported MegaMoE CuteDSL kernel needs.

    Stricter than ``IS_CUTLASS_DSL_AVAILABLE``, which only confirms that
    ``cutlass`` / ``cutlass.cute`` import cleanly. The MegaMoE kernel
    ABI also requires ``cutlass.torch.from_dlpack``, ``cutlass._mlir``
    APIs used by ``sym_buffer.py``, the ``cute_nvgpu`` MMA atoms used
    by ``kernel_fc12.py``, and the async-copy helpers used by
    ``dispatch_kernel.py``. PR
    https://github.com/NVIDIA/TensorRT-LLM/pull/14354 pins
    ``nvidia-cutlass-dsl[cu13]==4.5.0`` which is the first release that
    ships all of them; older wheels return ``(False, reason)``.

    Returns ``(True, None)`` on success or ``(False, reason)`` with an
    actionable message. The result is cached for the process lifetime.
    """
    global _RUNTIME_PROBE_CACHE
    if _RUNTIME_PROBE_CACHE is True:
        return True, None
    if isinstance(_RUNTIME_PROBE_CACHE, str):
        return False, _RUNTIME_PROBE_CACHE

    if not IS_CUTLASS_DSL_AVAILABLE:
        reason = (
            "Cutlass DSL is not importable on this environment; install "
            "nvidia-cutlass-dsl[cu13] to enable MegaMoECuteDsl."
        )
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        import cutlass.pipeline  # noqa: F401
        import cutlass.torch  # noqa: F401
        from cutlass._mlir import ir  # noqa: F401
        from cutlass.base_dsl.native_struct import native_struct  # noqa: F401
        from cutlass.cutlass_dsl import (  # noqa: F401
            Int32,
            Int64,
            Uint8,
            dsl_user_op,
            extract_mlir_values,
            new_from_mlir_values,
        )
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires CUDA 13 Cutlass DSL symbols; got "
            f"ImportError={e!r}. Install nvidia-cutlass-dsl[cu13]>=4.5.0 "
            f"(see PR #14354)."
        )
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    try:
        from cutlass.cute.nvgpu import cpasync, tcgen05  # noqa: F401
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires cutlass.cute.nvgpu.tcgen05 + cpasync; "
            f"missing {e!r}. Install a Blackwell-capable cutlass-dsl wheel."
        )
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    try:
        # mega_moe_cute_dsl.py lives at
        # tensorrt_llm/_torch/modules/fused_moe/mega_moe/mega_moe_cute_dsl.py;
        # four dots take us back to tensorrt_llm._torch where
        # cute_dsl_kernels.mega_moe_nvfp4 is registered.
        from ....cute_dsl_kernels.mega_moe_nvfp4 import (  # noqa: F401
            Nvfp4BlockSize,
            SfPaddingBlock,
            to_blocked,
        )
    except ImportError as e:
        reason = (
            f"Ported MegaMoE NVFP4 kernel package failed to import: "
            f"{e!r}. Verify tensorrt_llm/_torch/cute_dsl_kernels/"
            f"mega_moe_nvfp4 is in the install tree."
        )
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    _RUNTIME_PROBE_CACHE = True
    return True, None


# ---------------------------------------------------------------------------
# Weight view passed to ``run_moe``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MegaMoECuteDslWeightView:
    """Bundles the MegaMoE-format weight tensors built by
    ``NVFP4MegaMoECuteDslMethod.process_weights_after_loading``.

    The kernel reads these as local-only (NOT through symmetric heap);
    placement is unconstrained CUDA memory. Shapes match the
    ``Sm100MegaMoEKernel.__call__`` ABI documented in MEGAMOE_CUTEDSL_DESIGN.md.

    ``fc31_alpha`` / ``fc2_alpha`` / ``fc2_input_scale`` are bundled here
    even though the v1 kernel hard-codes alpha=1 / norm_const=1 (see
    "NVFP4 scale and alpha ABI"): the view is forward-compatible with the
    kernel ABI extension and tests assert non-1 values round-trip through
    the view unchanged so the gate fails loudly the moment a real
    checkpoint reaches the backend before the ABI lands.
    """

    # NVFP4 packed bytes; stride(-2) == 1 view exposes the kernel-required
    # ``(slots, hidden, expand_intermediate)`` logical layout. See the quant
    # method's _build_mega_format_weights for the byte-layout convention.
    fc1_weight: torch.Tensor  # uint8 storage (slots, hidden//2, expand_I)
    # FP8 atom-swizzled per-slot blocked scale, flattened to 1-D per slot.
    fc1_weight_sf: torch.Tensor  # uint8 storage (slots, fc1_sf_flat_size)
    fc2_weight: torch.Tensor  # uint8 storage (slots, intermediate//2, hidden)
    fc2_weight_sf: torch.Tensor  # uint8 storage (slots, fc2_sf_flat_size)
    # NVFP4 alpha / norm_const tensors. v1 kernel still hard-codes
    # alpha=1 / norm_const=1; non-1 values are rejected by the quant
    # method's _check_v1_alpha_gate. The fields stay in the view so the
    # kernel ABI extension lands without touching the backend boundary.
    fc31_alpha: torch.Tensor  # (slots,) fp32; FC1 per-expert global scale
    fc2_alpha: torch.Tensor  # (slots,) fp32; FC2 per-expert global scale
    fc2_input_scale: torch.Tensor  # () fp32; FC1-output quant norm_const


@dataclass(frozen=True)
class _MegaMoeBuffers:
    """Unified kernel-ABI view over MegaMoE CuteDSL's user-domain buffers.

    Single-rank and multi-rank execution differ ONLY in where these
    tensors physically live:

      * ``ep_size == 1``: local CUDA memory; ``peer_offsets == [0]``.
      * ``ep_size > 1``: peer-mapped symmetric heap regions from
        ``MegaMoeSymmMemProvider``; ``peer_offsets[r] = peer_base[r] -
        local_base``.

    ``topk_idx_local`` stays in plain CUDA memory in BOTH paths because
    the kernel reads it through ``input_topk_idx_buffer[token, slot]``
    only on the local rank -- peers never call
    ``peer_rank_ptr_mapper.map`` on it.

    All tensors are sized to ``max_num_tokens`` along the leading
    dimension so the kernel's compile-time constexpr matches the
    buffer-time ``max_tokens_per_rank``.
    """

    activation: torch.Tensor  # (max_T, hidden // 2) uint8 (NVFP4 packed)
    activation_sf: torch.Tensor  # (max_T, sf_bytes_per_row) uint8 (FP8 SF)
    topk_weights: torch.Tensor  # (max_T, top_k) float32
    combine_output: torch.Tensor  # (max_T, top_k, hidden) bf16
    shared_workspace: torch.Tensor  # (shared_ws_bytes,) uint8
    peer_offsets: List[int]  # length == world_size; [0] for single-rank
    topk_idx_local: torch.Tensor  # (max_T, top_k) int64, always-local


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class MegaMoECuteDsl(MoE):
    """MoE backend wrapping the ported MegaMoE CuteDSL NVFP4 fused kernel.

    Capability gate (``can_implement``): SM100 family + NVFP4 +
    bfloat16 activation + CUDA 13 Cutlass DSL runtime present.
    Multi-rank execution additionally requires an NVSHMEM-backed
    symmetric-memory provider; that provider design is tracked as a
    hard gate in MEGAMOE_CUTEDSL_DESIGN.md ("Symmetric Memory Design")
    and is NOT yet wired. ``run_moe`` therefore raises on multi-rank
    topologies; single-rank execution flows through the kernel via the
    NVSHMEM degenerate convention (``peer_offsets = (0,) * world_size``).
    """

    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})

    # Kernel owns dispatch + GEMM1 + SwiGLU + GEMM2 + combine via the
    # CuteDSL three-stage dispatch primitives + NVLink barrier; the
    # scheduler must skip host-side comm and lockstep every chunk.
    scheduler_kind = MoESchedulerKind.FUSED_COMM

    # ------------------------------------------------------------------
    # Capability gating
    # ------------------------------------------------------------------
    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Static capability query: SM/dtype/quant/shape only.

        Does NOT probe checkpoint tensor values. The v1 "all alphas ==
        1" fallback (until the kernel ABI extension lands) is enforced
        in ``NVFP4MegaMoECuteDslMethod.post_load_weights``;
        ``can_implement`` does not see checkpoint tensors.

        Multi-rank execution gate (NVSHMEM provider) is NOT in this
        query either, by analogy to ``MegaMoEDeepGemm.can_implement``;
        ``run_moe`` is where the provider absence becomes a hard error
        for ``ep_size > 1`` topologies.
        """
        sm = get_sm_version()
        if not is_sm_100f(sm):
            return False, (f"MegaMoECuteDsl requires SM100 family (SM100 or SM103); got SM{sm}.")
        if dtype_activation not in cls._SUPPORTED_ACTIVATION_DTYPES:
            return False, (
                f"MegaMoECuteDsl supports activations in "
                f"{cls._SUPPORTED_ACTIVATION_DTYPES}, got {dtype_activation}."
            )
        if swiglu_gptoss_style:
            return False, "MegaMoECuteDsl does not support swiglu_gptoss_style."
        if quant_algo != QuantAlgo.NVFP4:
            return False, (f"MegaMoECuteDsl supports NVFP4 only, got quant_algo={quant_algo}.")
        if hidden_size is not None:
            # ProblemDesc.__post_init__ at mega_runner.py:312-339 requires
            # hidden % (2 * Nvfp4BlockSize) == 0 -> hidden % 32 == 0.
            # 32-aligned hidden sizes that are NOT 64-aligned (1568,
            # 1632, 2080, ...) are allowed because the backend explicitly
            # pads the SF row width to ``round_up(ceil(hidden/16), 4)``
            # via ``megamoe_activation_sf_bytes_per_row`` at every site
            # that allocates or quantizes SF tensors (symm provider,
            # single-rank local staging, and ``quantize_input``). See
            # MEGAMOE_CUTEDSL_DESIGN.md hard-gate "Activation SF row
            # width must be round_up(...)".
            if hidden_size <= 0 or hidden_size % 32 != 0:
                return False, (
                    f"MegaMoECuteDsl requires positive hidden_size divisible "
                    f"by 32 (NVFP4 SF leg alignment); got {hidden_size}."
                )
        if intermediate_size is not None:
            # TRT-LLM's intermediate_size is the down-projection width. The
            # external kernel expects expand_intermediate = 2 * intermediate
            # to be divisible by 2 * Fc1GateUpInterleave (= 32) which
            # reduces to intermediate % 16 == 0.
            if intermediate_size <= 0 or intermediate_size % 16 != 0:
                return False, (
                    f"MegaMoECuteDsl requires positive intermediate_size "
                    f"divisible by 16 (Fc1GateUpInterleave); got "
                    f"{intermediate_size}."
                )
        ok, reason = is_megamoe_cute_dsl_runtime_available()
        if not ok:
            return False, reason
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
        # ``aux_stream_dict`` is accepted for ``create_moe_backend`` signature
        # uniformity but ignored: FUSED_COMM kernels must not use the chunk
        # overlap stream because launch order must be lockstep across EP.
        del aux_stream_dict
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=None,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            activation_type=activation_type,
            init_load_balancer=init_load_balancer,
        )

        assert self.tp_size == 1, (
            f"MegaMoECuteDsl is EP-only in v1 (moe_tp_size=1); got tp_size={self.tp_size}."
        )
        assert self.cluster_size == 1, (
            f"MegaMoECuteDsl assumes cluster_size=1; got cluster_size={self.cluster_size}."
        )
        if self.num_slots % max(self.ep_size, 1) != 0:
            raise ValueError(
                f"MegaMoECuteDsl requires num_slots ({self.num_slots}) "
                f"divisible by ep_size ({self.ep_size})."
            )

        if self.use_dp and self.parallel_size > 1:
            assert self.ep_size == self.parallel_size, (
                f"MegaMoECuteDsl with enable_attention_dp=True requires "
                f"ep_size == parallel_size (got ep_size={self.ep_size}, "
                f"parallel_size={self.parallel_size}). ADP > EP would "
                f"require an outer allgather + reducescatter wrapper."
            )

        if apply_router_weight_on_input:
            raise ValueError(
                "MegaMoECuteDsl does not support apply_router_weight_on_input; "
                "the fused kernel applies routing weights on the MoE output."
            )
        if activation_type != ActivationType.Swiglu:
            raise ValueError(
                f"MegaMoECuteDsl only supports ActivationType.Swiglu (got {activation_type})."
            )
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Buffer sizing. MoE layers execute serially per forward; one pool
        # sized to the worst-case per-rank tokens covers every layer. The
        # kernel compile takes this as the static ``max_tokens_per_rank``.
        self.max_num_tokens = int(
            getattr(model_config, "moe_max_num_tokens", 0)
            or getattr(model_config, "max_num_tokens", 0)
            or 4096
        )

        # Resolve EP ProcessGroup at construction. Resolving at forward
        # time would be collective on a non-synchronous call stack and
        # deadlock under PP / layer-skip. Construction is globally
        # synchronous across ranks during model build.
        try:
            self._ep_pg = self._resolve_ep_pg()
        except RuntimeError as e:
            # Single-rank tests do not always initialize torch.distributed.
            # The kernel's single-rank degenerate path does not need a PG.
            logger.debug(
                f"[MegaMoECuteDsl] EP PG not resolvable ({e!r}); falling back "
                f"to single-rank degenerate mode at run_moe time."
            )
            self._ep_pg = None

        # Weight tensors are owned by the quant method. ``_symm_provider``
        # is the symmetric-memory provider for multi-rank EP execution;
        # allocated build-time in ``create_weights`` (collective
        # rendezvous), shared across MoE layers via the module-scope
        # cache in ``cute_dsl_megamoe_custom_op.py``. ``None`` for the
        # single-rank degenerate path.
        self._symm_provider = None
        self._weights_loaded = False
        self._weights_created = False
        self._post_load_done = False
        self.quant_method = None
        # Per-instance staging tensor cache + last-staged-T tracker, used
        # by ``_ensure_local_staging`` and ``_stage_inputs`` to pad
        # topk_idx to ``-1`` and refresh only the rows that changed
        # between launches. See ``_run_moe`` for the always-pad-to-max_T
        # contract.
        self._local_staging_cache: dict = {}
        self._last_staged_T: Optional[int] = None
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def _supports_load_balancer(self) -> bool:
        # MegaMoECuteDsl supports both static and dynamic EPLB:
        # ``NVFP4MegaMoECuteDslMethod.process_weights_after_loading``
        # builds CPU shared-staging tensors for the four mega-format
        # derived parameters (``mega_fc1_weight`` /
        # ``mega_fc1_weight_sf`` / ``mega_fc2_weight`` /
        # ``mega_fc2_weight_sf``) via ``_build_mega_shared_staging``
        # and registers them with the load balancer through
        # ``_register_mega_shared_staging`` ->
        # ``register_all_parameter_slot_and_to_fix_weight_fns``.
        # Slot migration replaces all four mega-format derivatives
        # atomically with the underlying NVFP4 raw weights + scales
        # already migrated by the base/grandparent registrations
        # (``w3_w1_weight`` / ``w2_weight`` / ``w*_weight_scale`` /
        # ``fc*_alpha``).
        return True

    def validate_configurable_moe(self, moe) -> None:
        """Mirrors :meth:`MegaMoEDeepGemm.validate_configurable_moe`.

        See MEGAMOE_CUTEDSL_DESIGN.md "validate_configurable_moe" for
        the list of invariants enforced here.

        ``ConfigurableMoE.__init__`` calls this at the very end (after
        ``self.comm`` / ``self.moe_max_num_tokens`` and every EPLB /
        num_slots / ep_size attribute are populated -- see
        ``configurable_moe.py`` ``validate_backend`` docstring), so
        every attribute touched below may be read directly.
        """
        if moe.comm is not None:
            raise ValueError(
                f"MegaMoECuteDsl requires moe.comm is None (FUSED_COMM "
                f"backends must not layer host-side communication on top "
                f"of the fused kernel); got moe.comm={type(moe.comm).__name__}."
            )
        if moe.mapping.moe_tp_size != 1:
            raise ValueError(
                f"MegaMoECuteDsl v1 is EP-only (moe_tp_size=1); got {moe.mapping.moe_tp_size}."
            )
        # NOTE: ``mapping.tp_size`` is the *wrapper-level* TP size used by
        # attention, not by the MoE layer. In DEP / TEP modes the wrapper
        # sets ``tp_size = world_size`` while ``moe_tp_size = 1``; the
        # MegaMoECuteDsl kernel only cares about the MoE axes
        # (``moe_ep_size`` / ``moe_tp_size``) — see
        # ``_create_mapping_for_parallel_mode`` in test_moe_module.py.
        if moe.num_slots % moe.mapping.moe_ep_size != 0:
            raise ValueError(
                f"MegaMoECuteDsl requires num_slots ({moe.num_slots}) "
                f"divisible by moe_ep_size ({moe.mapping.moe_ep_size})."
            )
        if moe.use_dp and moe.parallel_size > 1 and moe.mapping.moe_ep_size != moe.parallel_size:
            raise ValueError(
                f"MegaMoECuteDsl with enable_attention_dp requires "
                f"moe_ep_size == parallel_size (got "
                f"moe_ep_size={moe.mapping.moe_ep_size}, "
                f"parallel_size={moe.parallel_size})."
            )
        top_k = moe.routing_method.experts_per_token
        if top_k > 13:
            raise ValueError(
                f"MegaMoECuteDsl v1 supports experts_per_token <= 13 "
                f"(matches external coverage); got {top_k}."
            )
        if moe.moe_max_num_tokens <= 0:
            raise ValueError(
                f"MegaMoECuteDsl requires moe_max_num_tokens > 0; got {moe.moe_max_num_tokens}."
            )
        # Dynamic EPLB IS supported: NVFP4MegaMoECuteDslMethod
        # registers the four mega-format derived parameters
        # (``mega_fc1_weight``/``mega_fc1_weight_sf``/
        # ``mega_fc2_weight``/``mega_fc2_weight_sf``) with the load
        # balancer alongside the parent NVFP4 raw weights + scales.
        # Both sets are migrated atomically per slot, and the source
        # rank built ``mega = transform(raw)`` once at load time, so
        # the migrated raw and mega bytes stay byte-consistent.

    # ------------------------------------------------------------------
    # EP process-group resolution (no collective at forward time)
    # ------------------------------------------------------------------
    def _resolve_ep_pg(self):
        """Return the torch.distributed ProcessGroup for the EP sub-world.

        Mirrors :meth:`MegaMoEDeepGemm._resolve_ep_pg` so the two MegaMoE
        backends share the same fallback chain.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "MegaMoECuteDsl requires torch.distributed to be initialized "
                "before module construction (mpirun or Ray) for multi-rank "
                "execution."
            )
        try:
            pg = self.mapping.moe_ep_group_pg
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoECuteDsl] layer={self.layer_idx} using "
                f"mapping.moe_ep_group_pg (DeviceMesh path)."
            )
            return pg
        except (NotImplementedError, AttributeError):
            pass
        world_size = dist.get_world_size()
        if self.ep_size == world_size:
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoECuteDsl] layer={self.layer_idx} using dist.group.WORLD "
                f"(EP == world_size == {world_size})."
            )
            return dist.group.WORLD
        raise RuntimeError(
            f"MegaMoECuteDsl: cannot resolve EP ProcessGroup. The current "
            f"mapping does not expose ``moe_ep_group_pg`` and EP "
            f"({self.ep_size}) is a strict subset of world ({world_size})."
        )

    # ------------------------------------------------------------------
    # Weight lifecycle
    # ------------------------------------------------------------------
    def _get_quant_method(self):
        if self.quant_config is None or not self.quant_config.layer_quant_mode.has_nvfp4():
            raise NotImplementedError("MegaMoECuteDsl supports NVFP4 quantization only.")
        return NVFP4MegaMoECuteDslMethod()

    def create_weights(self):
        """Build-time weight + symmetric-buffer allocation.

        Order follows MEGAMOE_CUTEDSL_DESIGN.md "Weight lifecycle APIs":
          1. Allocate symmetric-memory provider for multi-rank EP
             (collective rendezvous; MUST run at build time -- not from
             ``run_moe`` -- because forward time may be inside CUDA
             graph capture or non-lockstep PP/layer-skip).
          2. Resolve quantization method.
          3. Delegate parameter registration to the quant method.
          4. Flip ``_weights_created``.

        The symm provider is shared across MoE layers with the same
        (group, layout) via the module-scope cache in
        ``cute_dsl_megamoe_custom_op.py``; only the first layer that
        reaches this point pays the rendezvous cost, and every EP rank
        hits this code in lockstep because ``ConfigurableMoE`` calls
        ``create_weights`` on every rank after backend construction.
        """
        if self._weights_created:
            return
        # Step 1: build-time symmetric memory allocation (multi-rank only).
        # Single-rank degenerate uses local CUDA tensors and skips here.
        self._symm_provider = None
        if self.ep_size > 1:
            self._symm_provider = self._alloc_symm_provider()
        # Step 2-3: quant method registers all NVFP4 + MegaMoE-format params.
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        # Step 4.
        self._weights_created = True

    def _alloc_symm_provider(self):
        """Build-time symmetric provider allocation. See ``create_weights``.

        Returns a :class:`MegaMoeSymmMemProvider` from the module-scope
        cache. Raises :class:`MegaMoeCuteDslUnavailable` with an
        actionable message when no ProcessGroup is available -- that
        would block the rendezvous and is a hard error for multi-rank.
        """
        from ....custom_ops.cute_dsl_megamoe_custom_op import (
            get_megamoe_symm_provider,
            query_megamoe_shared_workspace_bytes,
        )

        if self._ep_pg is None:
            raise MegaMoeCuteDslUnavailable(
                "MegaMoECuteDsl multi-rank requires a torch.distributed EP "
                "ProcessGroup. Use Ray / DeviceMesh (mapping.moe_ep_group_pg) "
                "or initialize torch.distributed before model build."
            )
        top_k = self.routing_method.experts_per_token
        shared_workspace_bytes = query_megamoe_shared_workspace_bytes(
            world_size=self.ep_size,
            local_rank=self.ep_rank,
            num_topk=top_k,
            num_experts_per_rank=int(self.expert_size_per_partition),
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=int(self.intermediate_size_per_partition),
            expand_intermediate_size_per_partition=int(self.expand_intermediate_size_per_partition),
            max_tokens_per_rank=int(self.max_num_tokens),
        )
        return get_megamoe_symm_provider(
            process_group=self._ep_pg,
            world_size=self.ep_size,
            rank=self.ep_rank,
            hidden_size=self.hidden_size,
            max_tokens_per_rank=int(self.max_num_tokens),
            num_topk=top_k,
            output_dtype=self.dtype or torch.bfloat16,
            shared_workspace_bytes=shared_workspace_bytes,
        )

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None:
        if self.quant_method is None:
            self.create_weights()
        # Match CutlassFusedMoE.load_weights: callers pass ``[weights_dict]``.
        # ``FusedMoEMethodBase.load_expert_weights_to_dst`` treats the inner
        # value as a Dict (``weights[f"{expert_id}.w1.weight"]``), so unwrap
        # the single-element list before forwarding. Forward
        # ``weight_loading_mode`` explicitly because the base signature is
        # ``(module, weights, weight_loading_mode, allow_partial_loading=False)``;
        # passing ``allow_partial_loading`` (a bool) as the 3rd positional arg
        # would be interpreted as the mode and trip ``NotImplementedError``.
        assert len(weights) == 1, (
            "MegaMoECuteDsl.load_weights expects a single-element list, "
            f"got {len(weights)} entries."
        )
        weights = weights[0]

        self.quant_method.load_weights(
            self, weights, self.weight_loading_mode, allow_partial_loading=allow_partial_loading
        )

    def post_load_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.quant_method.post_load_weights(self)

    def process_weights_after_loading(self) -> None:
        """Idempotent alias retained for ``MoE`` interface compatibility.

        Real weight transforms run in ``NVFP4MegaMoECuteDslMethod.process_weights_after_loading``
        through ``self.post_load_weights()`` (the parent ``MoE`` interface
        only requires the post-load hook; this method just guarantees the
        idempotent behavior documented in MEGAMOE_CUTEDSL_DESIGN.md
        "Weight lifecycle APIs" rule 3).
        """
        if getattr(self, "_post_load_done", False):
            return
        self.post_load_weights()
        self._post_load_done = True

    def pre_reload_weights(self) -> None:
        """Reset cached state before a hot weight reload.

        ``_post_load_done`` is cleared so the next ``process_weights_after_loading``
        re-runs the MegaMoE-format weight transforms over the new
        checkpoint bytes. The symmetric-memory provider is forward-time
        scratch that does not need to be re-rendezvoused on weight
        reload; we keep it as-is to avoid an unnecessary collective.
        """
        self._post_load_done = False
        if self.quant_method is not None and hasattr(self.quant_method, "pre_reload_weights"):
            self.quant_method.pre_reload_weights(self)

    def _build_weight_view(self) -> MegaMoECuteDslWeightView:
        """Bundle the MegaMoE-format weight tensors registered by the
        quant method. ``run_moe`` calls this once per chunk so the
        kernel sees the latest dynamic-EPLB migration outcome (once
        that path lands; currently the slots are static).
        """
        return MegaMoECuteDslWeightView(
            fc1_weight=self.mega_fc1_weight,
            fc1_weight_sf=self.mega_fc1_weight_sf,
            fc2_weight=self.mega_fc2_weight,
            fc2_weight_sf=self.mega_fc2_weight_sf,
            fc31_alpha=self.fc31_alpha,
            fc2_alpha=self.fc2_alpha,
            fc2_input_scale=self.fc2_input_scale,
        )

    # ------------------------------------------------------------------
    # MoE-contract methods
    # ------------------------------------------------------------------
    def quantize_input(
        self,
        x: Union[torch.Tensor, "Fp4QuantizedTensor"],
        *,
        post_quant_comm: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 -> NVFP4 packed activation + plain K-major FP8 SF.

        Reuses ``torch.ops.trtllm.fp4_quantize`` with
        ``is_sf_swizzled=False`` so the SF tensor lands in the plain
        K-major layout expected by the MegaMoE kernel.
        ``self.fc31_input_scale`` is the per-tensor FP32 input scale
        registered by the quantization method's ``create_weights``; the
        value defaults to 1.0 until the checkpoint loader sets it.

        Empty input (``x.shape[0] == 0``) short-circuits to empty NVFP4
        + empty SF without launching a quantization kernel, so the
        ``FusedCommMoEScheduler`` can call ``quantize_input`` uniformly
        for zero-token chunks.
        """
        del post_quant_comm  # MegaMoE owns dispatch / combine in-kernel.
        del kwargs
        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "MegaMoECuteDsl.quantize_input expects BF16 activation; "
                "pre-quantized Fp4QuantizedTensor is not yet supported."
            )
        # Import lazily so the backend file stays importable on hosts
        # without the cutlass-dsl runtime; the module-level
        # ``IS_MEGAMOE_OP_AVAILABLE`` gate has already filtered those
        # out by the time run_moe/quantize_input is reachable on a
        # capable GPU.
        from ....custom_ops.cute_dsl_megamoe_custom_op import megamoe_activation_sf_bytes_per_row

        hidden = x.shape[1]
        sf_cols = megamoe_activation_sf_bytes_per_row(hidden)
        x_bf16 = x.to(torch.bfloat16).contiguous()
        if x_bf16.shape[0] == 0:
            empty_x = torch.empty((0, hidden // 2), dtype=torch.uint8, device=x_bf16.device)
            empty_sf = torch.empty((0, sf_cols), dtype=torch.uint8, device=x_bf16.device)
            return empty_x, empty_sf
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
            x_bf16,
            self.fc31_input_scale,
            16,  # scaling_vector_size == Nvfp4BlockSize
            False,  # sf_use_ue8m0
            False,  # is_sf_swizzled - MegaMoE expects plain K-major
        )
        # ``fp4_quantize(is_sf_swizzled=False)`` returns the LINEAR layout
        # which is ``(rows, ceil(hidden/16))`` FP8 bytes -- NO column pad.
        # The MegaMoE kernel TMA load expects each row to be at least
        # ``round_up(ceil(hidden/16), 4)`` bytes (``sf_uint32_per_token``
        # in megamoe_kernel.py). For hidden sizes that are 32-aligned but
        # not 64-aligned (1568, 1632, 2080, ...) the raw output is 2
        # bytes short per row; allocate a padded tensor and copy.
        raw_cols = (hidden + 15) // 16
        x_sf_raw = x_sf.view(x_bf16.shape[0], raw_cols)
        if sf_cols == raw_cols:
            return x_fp4, x_sf_raw
        padded_sf = torch.zeros((x_bf16.shape[0], sf_cols), dtype=torch.uint8, device=x_bf16.device)
        padded_sf[:, :raw_cols] = x_sf_raw
        return x_fp4, padded_sf

    # NOTE: The symmetric-memory provider is built at ``create_weights``
    # time via ``_alloc_symm_provider`` and cached on ``self._symm_provider``.
    # ``_acquire_buffers`` reads ``self._symm_provider`` directly --
    # doing the rendezvous at forward time would violate the build-time
    # collective rule documented in MEGAMOE_CUTEDSL_DESIGN.md
    # "Allocation timing".

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        *,
        output_dtype: Optional[torch.dtype] = None,
        **unused_kwargs,
    ) -> torch.Tensor:
        """Run the fused MegaMoE CuteDSL kernel on pre-quantized inputs.

        Steps (matches MEGAMOE_CUTEDSL_DESIGN.md "run_moe"):
          1. Build :class:`MegaMoECuteDslWeightView` from the quant
             method's MegaMoE-format derived tensors.
          2. Stage activation + per-token topk metadata. ``topk_idx``
             is converted to ``int64`` here (scheduler keeps ``int32``).
             For ``ep_size > 1`` the user-domain tensors must live in
             the symmetric heap; the backend stages them into the
             provider's pre-allocated regions, then the kernel reads
             from peer ranks via ``peer_rank_ptr_mapper.map(...)``.
          3. Allocate / reuse ``combine_output`` shape
             ``(T, top_k, hidden)`` BF16. Form A: kernel writes one
             cell per ``(src_rank, src_token, src_topk)``; the host
             sums the top-k axis after the kernel returns.
          4. Invoke ``torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell``
             (the runner + compile cache live in
             ``custom_ops/cute_dsl_megamoe_custom_op.py``).
          5. Sum the top-k axis -> return ``(T, hidden_size)``.
        """
        del unused_kwargs
        if output_dtype is None:
            output_dtype = self.dtype or torch.bfloat16
        if x_sf is None:
            raise ValueError("MegaMoECuteDsl requires x_sf from quantize_input")

        # Multi-rank symmetric-memory provider rendezvous is a build-time
        # collective; if it never ran (ep_size flipped to > 1 after
        # construction, or rendezvous failed) we have to surface
        # MegaMoeCuteDslUnavailable now so callers can switch to Ray /
        # DeviceMesh / mpirun. Check this BEFORE the weights guard so the
        # provider failure path remains distinguishable from the
        # "no weights loaded" path in tests and on real runs.
        if self.ep_size > 1 and getattr(self, "_symm_provider", None) is None:
            raise MegaMoeCuteDslUnavailable(
                "MegaMoECuteDsl multi-rank run_moe requires the cuMem "
                "symmetric-memory provider, but no provider was allocated "
                "for this backend instance. The provider rendezvous runs at "
                "create_weights() time and needs a live torch.distributed "
                "EP ProcessGroup; spawn the workload via Ray / DeviceMesh "
                "or mpirun so the rendezvous can complete."
            )

        if not self._weights_created or self.quant_method is None:
            raise RuntimeError(
                "MegaMoECuteDsl.run_moe called before create_weights / "
                "load_weights / post_load_weights finished. The MegaMoE-"
                "format weight tensors are missing."
            )

        weight_view = self._build_weight_view()
        num_tokens = int(x.shape[0])
        hidden = self.hidden_size
        top_k = int(token_selected_experts.shape[-1])
        device = x.device

        # The kernel reads ``topk_idx`` as Int64 (see TokenCommArgs in
        # megamoe_kernel.py); the scheduler keeps Int32 for the EPLB
        # stats kernel. Cast at this boundary. ``topk_idx`` is local-only
        # (peers never read it) so it stays in CUDA memory.
        topk_idx_i64 = token_selected_experts.to(torch.int64).contiguous()
        topk_weights_f32 = token_final_scales.to(torch.float32).contiguous()

        return self._run_moe(
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx_i64,
            topk_weights=topk_weights_f32,
            weight_view=weight_view,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
        )

    def _ensure_local_staging(self, *, top_k: int, hidden: int, device, output_dtype):
        """Allocate (and cache) the per-instance local staging tensors.

        Returns a dict whose ``"topk_idx"`` entry is allocated in BOTH
        topologies (the kernel reads it as a local-only buffer; peers
        never call ``peer_rank_ptr_mapper.map`` on it). The remaining
        entries -- ``activation`` / ``activation_sf`` / ``topk_weights``
        / ``combine_output`` / ``shared_workspace`` -- are allocated
        ONLY when ``ep_size == 1``; multi-rank pulls those from the
        symmetric-memory regions instead.

        All staging tensors are sized to ``max_num_tokens`` along dim 0
        so that the kernel's constexpr ``num_tokens`` constant equals
        the symmetric buffer-time ``max_tokens_per_rank``. Mismatching
        these two values would make ``_dispatch_prep`` round 3 (see
        cute_dsl_kernels/mega_moe_nvfp4/dispatch_kernel.py:355,
        ``MAX_SLOT_C = num_tokens * num_topk``) write per-(expert,rank)
        advertise cards at the wrong stride relative to the buffer
        allocation ``max_tokens_per_rank * num_topk`` in
        megamoe_kernel.py:787 -- silent multi-rank metadata corruption.
        """
        from ....custom_ops.cute_dsl_megamoe_custom_op import megamoe_activation_sf_bytes_per_row

        max_T = int(self.max_num_tokens)
        cache_key = (max_T, top_k, hidden, str(device), output_dtype)
        cached = getattr(self, "_local_staging_cache", None)
        if cached is None:
            cached = {}
            self._local_staging_cache = cached
        if cache_key in cached:
            return cached[cache_key]

        # ``topk_idx`` default value is ``-1`` so any padded tail rows
        # are skipped by dispatch_prep round-1 / round-3 loops
        # (dispatch_kernel.py:341, ``if expert_id >= Int32(0):``).
        staging = {
            "topk_idx": torch.full((max_T, top_k), -1, dtype=torch.int64, device=device),
        }
        if self.ep_size == 1:
            sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden)
            # ``topk_weights`` default is 0.0 so any stale combine
            # reduction rows contribute nothing. Multi-rank uses the
            # symm provider's topk_weights region instead and does not
            # need this local copy.
            staging["topk_weights"] = torch.zeros(
                (max_T, top_k), dtype=torch.float32, device=device
            )
            staging["activation"] = torch.empty(
                (max_T, hidden // 2), dtype=torch.uint8, device=device
            )
            staging["activation_sf"] = torch.empty(
                (max_T, sf_bytes_per_row), dtype=torch.uint8, device=device
            )
            staging["combine_output"] = torch.empty(
                (max_T, top_k, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            from ....custom_ops.cute_dsl_megamoe_custom_op import (
                query_megamoe_shared_workspace_bytes,
            )

            shared_bytes = query_megamoe_shared_workspace_bytes(
                world_size=1,
                local_rank=0,
                num_topk=top_k,
                num_experts_per_rank=int(self.expert_size_per_partition),
                hidden_size=hidden,
                intermediate_size_per_partition=int(self.intermediate_size_per_partition),
                expand_intermediate_size_per_partition=int(
                    self.expand_intermediate_size_per_partition
                ),
                max_tokens_per_rank=max_T,
            )
            staging["shared_workspace"] = torch.empty(
                shared_bytes, dtype=torch.uint8, device=device
            )
        cached[cache_key] = staging
        return staging

    def _acquire_buffers(self, *, top_k: int, hidden: int, device, output_dtype) -> _MegaMoeBuffers:
        """Resolve the kernel's input/output buffers for the current
        ``ep_size``.

        This is the ONLY structural branch between single-rank and
        multi-rank execution: where do activation / activation_sf /
        topk_weights / combine_output / shared_workspace live?

          * ``ep_size == 1``: local CUDA memory (``cudaMalloc`` via
            ``_ensure_local_staging``). No ``torch.distributed``
            dependency, no rendezvous, no cuMem VMM overhead.
            ``peer_offsets == [0]`` makes the kernel's
            ``peer_rank_ptr_mapper.map(local_addr, 0, off) ==
            local_addr + off`` (self-mapped, NVSHMEM degenerate
            convention).
          * ``ep_size > 1``: symmetric-memory regions from
            ``self._symm_provider`` (allocated at build time via the
            collective ``rendezvous``). ``peer_offsets[r] =
            peer_base[r] - local_base`` lets the kernel do in-kernel
            cross-GPU NVLink load/store.

        ``topk_idx_local`` stays in CUDA memory in BOTH paths because
        the kernel reads it through ``input_topk_idx_buffer[token,
        slot]`` only on the local rank, never via
        ``peer_rank_ptr_mapper.map``.
        """
        staging = self._ensure_local_staging(
            top_k=top_k, hidden=hidden, device=device, output_dtype=output_dtype
        )
        if self.ep_size == 1:
            return _MegaMoeBuffers(
                activation=staging["activation"],
                activation_sf=staging["activation_sf"],
                topk_weights=staging["topk_weights"],
                combine_output=staging["combine_output"],
                shared_workspace=staging["shared_workspace"],
                peer_offsets=[0],
                topk_idx_local=staging["topk_idx"],
            )
        # Multi-rank: the provider must have been rendezvous'd at
        # build time (``create_weights``) -- doing it at forward time
        # would violate the build-time collective rule and deadlock
        # under PP / layer-skip.
        if self._symm_provider is None:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl multi-rank (ep_size={self.ep_size}) "
                f"requires a symmetric-memory provider built at "
                f"create_weights time. self._symm_provider is None -- "
                f"check that the EP ProcessGroup was resolvable when the "
                f"backend was constructed (mapping.moe_ep_group_pg or a "
                f"named dist.new_group), or that "
                f"model_config.skip_create_weights_in_init was not set "
                f"without a follow-up create_weights() call."
            )
        if self._symm_provider.num_topk != top_k:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl symm provider was built for top_k="
                f"{self._symm_provider.num_topk} but run_moe called with "
                f"top_k={top_k}; recreate the backend."
            )
        regions = self._symm_provider.get_regions()
        return _MegaMoeBuffers(
            activation=regions.activation,
            activation_sf=regions.activation_sf,
            topk_weights=regions.topk_weights,
            combine_output=regions.combine_output,
            shared_workspace=regions.shared_workspace,
            peer_offsets=regions.peer_offsets,
            topk_idx_local=staging["topk_idx"],
        )

    def _stage_inputs(
        self,
        *,
        bufs: _MegaMoeBuffers,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> None:
        """Copy live rows of the user-domain inputs into the kernel's
        pre-allocated buffers and refresh the padded tail.

        Same code for single-rank (writes land in local CUDA) and
        multi-rank (writes land in symmetric heap regions visible to
        peers). The buffer source is selected upstream by
        :meth:`_acquire_buffers`.

        Tail policy:

          * ``topk_idx_local``: always ``-1`` outside live rows.
            Allocated as ``-1`` once in ``_ensure_local_staging``;
            only the rows we previously wrote (``[num_tokens,
            last_T)``) need resetting. New tail rows
            ``[last_T, max_T)`` already hold ``-1`` from prior calls.
          * ``topk_weights``: always ``0.0`` outside live rows. The
            combine kernel writes one cell per
            ``(token, k in [0, top_k))`` regardless of the
            ``topk_idx == -1`` mask, so a stale non-zero weight in
            the tail could corrupt the combine reduction (especially
            on peer ranks via NVLink). One cheap zero kernel covers
            it.
        """
        max_T = bufs.topk_idx_local.shape[0]
        last_T = getattr(self, "_last_staged_T", None)
        if last_T is not None and last_T > num_tokens:
            bufs.topk_idx_local[num_tokens:last_T].fill_(-1)
        if num_tokens > 0:
            bufs.topk_idx_local[:num_tokens].copy_(topk_idx, non_blocking=True)
            bufs.activation[:num_tokens].copy_(x.view(torch.uint8), non_blocking=True)
            bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
            bufs.topk_weights[:num_tokens, :top_k].copy_(topk_weights, non_blocking=True)
        if num_tokens < max_T:
            bufs.topk_weights[num_tokens:max_T, :top_k].zero_()
        self._last_staged_T = num_tokens

    def _launch_megamoe_kernel(
        self,
        *,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        weight_view: MegaMoECuteDslWeightView,
        combine_output: torch.Tensor,
        shared_workspace: torch.Tensor,
        world_size: int,
        local_rank: int,
        top_k: int,
        hidden: int,
        peer_offsets: List[int],
        num_tokens: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Launch the common MegaMoE CuteDSL kernel and reduce form-A output.

        Single-rank and multi-rank both reach this point with identical
        kernel inputs; the only difference upstream is where the staged
        tensors physically live (decided by :meth:`_acquire_buffers`).
        The kernel ABI and the host-side top-k reduction are otherwise
        identical across topologies.
        """

        # The CuteDSL kernel reads ``a_dtype`` / ``b_dtype`` / ``sf_dtype``
        # from the cute tensor ``element_type`` (kernel_fc12.py:946-950).
        # ``_to_cute`` -> ``cutlass_torch.from_dlpack`` preserves the torch
        # dtype, so the raw uint8 buffers we built must be re-viewed as the
        # NVFP4 packed / FP8 block-scale torch dtypes here:
        #   * activation / fc{1,2}_weight  -> torch.float4_e2m1fn_x2
        #     (cute element_type = Float4E2M1FN, what MmaMXF4NVF4Op needs
        #     for the NVFP4 path; raw uint8 trips
        #     "unsupported a_dtype/b_dtype: Int8 / Float4E2M1FN").
        #   * {activation_sf, fc{1,2}_weight_sf}  -> torch.float8_e4m3fn
        #     (cute element_type = Float8E4M3FN, what
        #     ``make_blockscaled_trivial_tiled_mma`` requires for NVFP4
        #     scales; raw uint8 trips "expects the 'sf_dtype' Op parameter
        #     to be one of Float8E8M0FNU" since cute falls back to the
        #     MXFP4 path when the sf_dtype isn't FP8).
        # Doing both views here (before the custom op call) keeps
        # autotuner's ``_create_tensor_like`` aligned with the runner's
        # ``_to_cute``: every code path sees the same dtype.
        def _as_nvfp4(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == torch.float4_e2m1fn_x2 else t.view(torch.float4_e2m1fn_x2)

        def _as_fp8_sf(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)

        torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
            activation=_as_nvfp4(activation),
            activation_sf=_as_fp8_sf(activation_sf),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            fc1_weight=_as_nvfp4(weight_view.fc1_weight),
            fc1_weight_sf=_as_fp8_sf(weight_view.fc1_weight_sf),
            fc2_weight=_as_nvfp4(weight_view.fc2_weight),
            fc2_weight_sf=_as_fp8_sf(weight_view.fc2_weight_sf),
            combine_output=combine_output,
            shared_workspace=shared_workspace,
            world_size=world_size,
            local_rank=local_rank,
            num_topk=top_k,
            num_experts_per_rank=int(self.expert_size_per_partition),
            hidden_size=hidden,
            intermediate_size_per_partition=int(self.intermediate_size_per_partition),
            expand_intermediate_size_per_partition=int(self.expand_intermediate_size_per_partition),
            max_tokens_per_rank=int(self.max_num_tokens),
            peer_offsets=peer_offsets,
        )
        if num_tokens == 0:
            return torch.empty((0, hidden), dtype=output_dtype, device=combine_output.device)
        out = combine_output[:num_tokens].sum(dim=1)
        if out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def _run_moe(
        self,
        *,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        weight_view: MegaMoECuteDslWeightView,
        num_tokens: int,
        top_k: int,
        hidden: int,
        device,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Unified MegaMoE CuteDSL forward path.

        Steps:

          1. :meth:`_acquire_buffers` decides where the kernel's
             input/output buffers live -- local CUDA for
             ``ep_size == 1`` or symmetric heap for ``ep_size > 1``.
             This is the only structural branch between the two
             topologies.
          2. :meth:`_stage_inputs` copies the live rows of
             ``activation`` / ``activation_sf`` / ``topk_idx`` /
             ``topk_weights`` into those buffers and refreshes the
             ``[num_tokens, max_T)`` padded tail.
          3. :meth:`_launch_megamoe_kernel` invokes the fused kernel
             via ``torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell``
             and reduces the top-k axis to ``(T, hidden)``.

        Always launches the kernel with ``T = max_num_tokens`` (the
        kernel compiles to that constexpr). Real tokens occupy the
        first ``num_tokens`` rows; the tail is masked via
        ``topk_idx == -1`` (dispatch_kernel.py:341 skip) and zero
        ``topk_weights`` (combine-side stale-data guard).

        ``FusedCommMoEScheduler`` invariant 7 requires every EP rank
        to launch every chunk for the NVLink barrier even when its
        local ``num_tokens`` is zero -- the ``-1``-padded
        ``topk_idx_local`` makes those launches a no-op on the MMA
        path while still crossing the dispatch barrier. Single-rank
        has no peer waiting on that barrier, so it may short-circuit
        when ``num_tokens == 0``.
        """
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"MegaMoECuteDsl run_moe got {num_tokens} tokens but the "
                f"staging buffer is sized for {self.max_num_tokens}. Raise "
                f"model_config.moe_max_num_tokens so peers do not read "
                f"invalid rows."
            )
        # Single-rank perf optimization: no peer waits on the NVLink
        # barrier, so the kernel launch is pure waste when the local
        # chunk has no tokens. Multi-rank MUST launch regardless (see
        # FusedCommMoEScheduler invariant 7 above).
        if num_tokens == 0 and self.ep_size == 1:
            return torch.empty((0, hidden), dtype=output_dtype, device=device)

        bufs = self._acquire_buffers(
            top_k=top_k, hidden=hidden, device=device, output_dtype=output_dtype
        )
        self._stage_inputs(
            bufs=bufs,
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
            top_k=top_k,
        )
        return self._launch_megamoe_kernel(
            activation=bufs.activation,
            activation_sf=bufs.activation_sf,
            topk_idx=bufs.topk_idx_local,
            topk_weights=bufs.topk_weights[:, :top_k],
            weight_view=weight_view,
            combine_output=bufs.combine_output,
            shared_workspace=bufs.shared_workspace,
            world_size=self.ep_size,
            local_rank=self.ep_rank,
            top_k=top_k,
            hidden=hidden,
            peer_offsets=bufs.peer_offsets,
            num_tokens=num_tokens,
            output_dtype=output_dtype,
        )
