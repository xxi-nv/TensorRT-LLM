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
ConfigurableMoE: Composition-based Configurable MoE Module

This module provides a universal MoE execution flow using composition pattern:
- MoE Backend: Pluggable computation backend (Cutlass, TRTLLMGen, etc.)
- Communication Strategy: Pluggable communication (AllGather, AllToAll, etc.)
- EPLB: Optional load balancing (can be toggled on/off)

Design Principles:
1. Use composition instead of inheritance for flexibility
2. Backend declares its capabilities (separated vs fused routing)
3. ConfigurableMoE adapts flow based on backend capabilities
4. Unified EPLB integration for backends that support it
"""

import copy
from contextlib import contextmanager
from typing import Dict, List, Optional, Type, Union

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.impl_base import MoEImplBase
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    canonical_quant,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoE, MoESchedulerKind, _reject
from tensorrt_llm._torch.moe.fused_moe.routing import BaseMoeRoutingMethod
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager
from tensorrt_llm._torch.utils import AuxStreamType, EventType, Fp4QuantizedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from .activation import install_activation_params
from .communication import AllGatherReduceScatter, Communication, CommunicationFactory
from .moe_scheduler import MoEScheduler, create_moe_scheduler


def _canonical_quant_key(quant_config: Optional[QuantConfig]) -> str:
    """The format string a resolved implementation identity is keyed on.

    ``canonical_quant`` folds a calibration recipe onto the format the kernel
    actually runs (NVFP4_AWQ to nvfp4) and maps the model-level markers
    (MIXED_PRECISION, NO_QUANT) to ``None``, which the identities spell
    ``"none"``. Comparing at exactly this granularity is what keeps a
    re-resolution a no-op unless the format itself moved.
    """
    algo = None if quant_config is None else quant_config.quant_algo
    return canonical_quant(algo) or "none"


# Attributes that ConfigurableMoE owns (computed in MoE.__init__ from real
# layer_idx + load balancer) and must be mirrored onto the backend after
# the backend was constructed with layer_idx=None / init_load_balancer=False.
# Adding a new EPLB-derived attribute? Append it here so the sync stays
# in one place and __init__ does not silently drift.
_BACKEND_SYNC_ATTRS = (
    "layer_idx",
    "layer_idx_str",
    "num_slots",
    "layer_load_balancer",
    "repeat_count",
    "repeat_idx",
    "initial_local_expert_ids",
    "initial_global_assignments",
    "slot_start",
    "slot_end",
    "expert_size_per_partition",
)


class ConfigurableMoE(MoE):
    # ConfigurableMoE is a thin wrapper that dispatches to a concrete backend
    # (CuteDslFusedMoE / CutlassFusedMoE / ...). ``MoE.__init__`` ->
    # ``_init_load_balancer`` runs before the backend exists, so the wrapper
    # cannot answer for it there and passes its own gate. The real check runs in
    # ``_reject_non_divisible_ep_backend()`` once the backend class is known.
    # Do not treat this ``True`` as "the wrapper supports it": nothing else
    # checks, because the backend is built with ``init_load_balancer=False``.
    _supports_non_divisible_ep: bool = True
    """
    Configurable MoE layer using composition pattern with automatic configuration

    This class orchestrates the MoE execution flow by composing:
    - moe_backend: Existing FusedMoE implementation used as a pluggable backend.
                   Currently supported backends (see
                   ``moe_resolution.IMPL_PRIORITY``):
                   CutlassFusedMoE, TRTLLMGenFusedMoE, DeepGemmFusedMoE,
                   CuteDslFusedMoE, DenseGEMMFusedMoE, MegaMoEDeepGemm.
                   Note: Current FusedMoE implementations are used as backends (transitional).
                         Future will have dedicated MoEBackend interface.
    - Communication: Handles distributed communication (auto-selected)
    - EPLB (optional): Handles expert parallel load balancing (auto-detected)

    Args:
        routing_method: Routing method for token-to-expert assignment
        num_experts: Total number of experts
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        dtype: Data type for weight
        reduce_results: Whether to reduce results
        model_config: Model configuration
        aux_stream_dict: Auxiliary CUDA streams for overlap
        weight_loading_mode: Weight loading mode
        layer_idx: Layer index
        **kwargs: Additional arguments
            - tune_max_num_tokens: Max tokens for profiling (passed to backend)
            - Other backend-specific arguments

    Key Attributes:
        - backend: MoE computation backend (auto-created attribute)
        - comm: Communication strategy (auto-created attribute, can be None)
        - layer_load_balancer: EPLB instance (auto-detected, optional)

    Auto-Detection:
        - EPLB: Enabled if get_moe_load_balancer() is not None
        - Backend: Resolved from ``model_config.moe_backend`` by
                   ``moe_resolution.resolve_moe_impl``, which degrades to
                   CutlassFusedMoE when the requested backend cannot serve the
                   layer and records the reason in the returned report.
                   ``create_moe`` passes the resolved class in as ``moe_cls``;
                   constructing this wrapper directly resolves on demand.
        - Communication: Auto-selected based on hardware (NVLINK > DeepEP > AllGather);
                         skipped entirely for FUSED_COMM backends (e.g. MegaMoEDeepGemm).
    """

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """Always ineligible: ConfigurableMoE delegates, it does not compute.

        Answering ``False`` rather than raising is what lets a registry walk
        include this class harmlessly. Query the backend it would delegate to
        (``CutlassFusedMoE``, ``TRTLLMGenFusedMoE``, ...) instead.
        """
        del p, d  # a wrapper's answer cannot depend on the question
        return _reject(
            MoERejectReason.NOT_AN_IMPL,
            "ConfigurableMoE is a wrapper class. "
            "Query the specific backend (CutlassFusedMoE, TRTLLMGenFusedMoE, etc.) directly.",
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
        weight_loading_mode=None,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        override_quant_config: Optional["QuantConfig"] = None,
        moe_cls: Optional[Type] = None,
        communication_method: Optional[str] = None,
        allow_backend_degradation: bool = True,
        **kwargs,
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
            layer_idx=layer_idx,  # ConfigurableMoE needs correct layer_idx for EPLB initialization
            **kwargs,
        )
        # A starting value for this layer's quant_config, not an authority over
        # it: ``__post_init__`` may still give the layer a layerwise entry or an
        # exclusion, and ``create_weights`` resolves against whatever is final
        # by then. Callers that look ``quant_config_dict`` up themselves
        # therefore keep working, and exclusions reach the MoE too -- which
        # they did not while this value outranked ``self.quant_config``.
        if override_quant_config is not None:
            self.quant_config = override_quant_config

        # Store model_config and aux_stream_dict for later use (e.g., backend setter)
        self.model_config = model_config
        self.aux_stream_dict = aux_stream_dict
        self.communication_method = communication_method

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # ========== Backend binding state (backend itself comes later) ==========
        # The backend is resolved and constructed in ``create_weights``, not
        # here: its class *is* one quantization format, and this layer's format
        # is not final until ``__post_init__`` has applied both halves of
        # layerwise quantization. See ``create_weights``.
        #
        # ``self.backend`` is deliberately left *absent* rather than set to
        # None. The delegating properties below reach it through
        # ``self.backend``, so absence makes them raise AttributeError, which
        # is what the ``hasattr`` / ``getattr`` guards at their call sites are
        # written for; ``None`` would instead trip their inner asserts and
        # raise AssertionError straight through those guards.
        self._pinned_moe_cls = moe_cls
        self._allow_backend_degradation = allow_backend_degradation
        self._backend_bound = False
        self._bound_quant_key: Optional[str] = None
        # Read by the communication factory, so it needs an answer before the
        # backend that owns it exists. Restated from the backend's class in
        # ``_create_backend``.
        self.use_flashinfer = False

        # ========== Optional DWDP integration ==========
        # Eligibility is a question about the backend, so both it and the comm
        # strategy are decided in ``_install_backend_dependents``. Declared
        # here because the
        # wrapper answers about itself in the unbound window: ``destroy()``,
        # ``calculate_num_chunks``, and ``enable_alltoall`` all read these.
        self.dwdp_manager = get_global_dwdp_manager()
        self.enable_dwdp = False
        self.comm: Optional[Communication] = None

        # ========== Chunking Configuration ==========
        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size

        # Auxiliary stream for chunking overlap
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

        # Validate configuration
        self.validate_config()

        # Mark as _weights_removed to skip ConfigurableMoE's post_load_weights in model_loader
        # The backend's post_load_weights will be called directly by model_loader
        # This avoids duplicate post_load_weights calls (once for ConfigurableMoE, once for backend)
        # TODO: in the future, all the weights related work should be done only in backend.
        self._weights_removed = True

        # Bound here rather than from ``__post_init__`` for the construction
        # paths that have no ``__post_init__`` to run: direct construction in
        # tests and microbenchmarks, which then read ``moe.backend`` or call
        # ``load_weights()`` straight away. ``AutoModelForCausalLM`` forces
        # ``skip_create_weights_in_init=True`` for every
        # ``DecoderModelForCausalLM``, so no production layer takes this path.
        #
        # Last in ``__init__`` because binding builds everything derived from
        # the backend's class, and the scheduler among those reads the chunking
        # state assigned above.
        #
        # The ``quant_config_dict`` half of the gate defers to __post_init__
        # when the checkpoint states formats per layer. Exclusions alone do not
        # defer -- they reset ``_weights_created`` on matching modules during
        # __post_init__, and ``create_weights`` rebinds rather than silently
        # keeping the identity picked for the pre-exclusion format.
        has_post_init_quant_config = model_config.quant_config_dict is not None
        if not model_config.skip_create_weights_in_init and not has_post_init_quant_config:
            self.create_weights()

    @staticmethod
    @contextmanager
    def _temporarily_skip_weight_creation(model_config: ModelConfig):
        """Force ``model_config.skip_create_weights_in_init = True`` for the duration.

        The backend is constructed with ``layer_idx=None`` and an unset load
        balancer, so weight allocation must be deferred until ConfigurableMoE
        has synced the real EPLB-derived attributes onto the backend (see
        ``_BACKEND_SYNC_ATTRS``). The flag is also flipped through the
        ``_frozen`` Pydantic guard, hence the bracketing dance. Using a
        contextmanager guarantees the original state is restored even if
        backend construction raises.
        """
        previous = model_config.skip_create_weights_in_init
        model_config._frozen = False
        model_config.skip_create_weights_in_init = True
        model_config._frozen = True
        try:
            yield
        finally:
            model_config._frozen = False
            model_config.skip_create_weights_in_init = previous
            model_config._frozen = True

    def _backend_model_config(self, quant_config: Optional[QuantConfig]) -> ModelConfig:
        """``self.model_config`` with ``quant_config`` replaced, shallow.

        Shallow on purpose. ``extra_attrs`` is a model-level shared registry:
        the MoE layer weakrefs, the aux streams, and the fault-tolerance
        ``EPGroupHealth`` all live there, and the last of those holds a
        ``threading.Lock``, which no deep copy can carry. Every field other
        than ``quant_config`` therefore keeps its object identity, and a copy
        per layer stays cheap on a 61-layer model.

        No ``_frozen`` bracketing: ``ModelConfig.__setattr__`` exempts
        ``quant_config`` from the freeze, precisely so a layer can be given its
        own format.
        """
        if quant_config is self.model_config.quant_config:
            return self.model_config
        backend_model_config = copy.copy(self.model_config)
        backend_model_config.quant_config = quant_config
        return backend_model_config

    def _resolve_backend_cls(self, quant_config: Optional[QuantConfig]) -> Type:
        """The implementation class serving ``quant_config`` for this layer.

        ``_pinned_moe_cls`` short-circuits this for the two callers that name a
        class: direct construction, and ``create_moe``'s one family whose
        candidates mix a complete layer with a backend, where wrapped-vs-bare
        could only be answered by resolving. A pin is taken at its word for the
        same reason resolution takes a pinned identity at its word -- a caller
        that named a class is asking for that one.

        Refuses a class that is not an execution unit, which resolution can
        return: those families hold a complete MoE layer, and one cannot be
        installed as a backend. Refused rather than degraded, because the shape
        of the layer was already decided in ``create_moe`` and the routing
        precision the gate was built with went with it.
        """
        from tensorrt_llm._torch.moe.fused_moe.create_moe import (
            infer_swiglu_gptoss_style,
            resolve_moe_cls,
        )

        if self._pinned_moe_cls is not None:
            return self._pinned_moe_cls
        moe_cls = resolve_moe_cls(
            self.model_config,
            override_quant_config=quant_config,
            dtype=self.dtype,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            swiglu_gptoss_style=infer_swiglu_gptoss_style(
                bias=self.bias,
                activation_type=self.activation_type,
            ),
            bias=self.bias,
            activation=self.activation,
            routing=self.routing_method,
            layer_idx=self.layer_idx,
            # Without this the fallback resolution silently allowed
            # degradation, so a caller that asked for one specific backend
            # could be handed Cutlass's numbers under that backend's name.
            allow_degradation=self._allow_backend_degradation,
        )
        if not issubclass(moe_cls, MoEImplBase):
            raise ValueError(
                f"Layer {self.layer_idx} ends up at "
                f"quant={_canonical_quant_key(quant_config)}, which resolves "
                f"to {moe_cls.__name__} -- a complete MoE layer rather than an "
                f"execution unit installable as ConfigurableMoE.backend. "
                f"Choose a moe_backend that serves every format this model's "
                f"layers use, or drop the layerwise override on this one."
            )
        return moe_cls

    def _create_backend(self, quant_config: Optional[QuantConfig], moe_cls: Type) -> None:
        """Construct and install ``moe_cls`` as the backend; mirror the EPLB attrs.

        Weight allocation and everything derived from the backend's class are
        the caller's next two steps, in that order -- see ``create_weights``.

        Why this dance:
        - ``init_load_balancer=False``: the backend would otherwise
          re-register itself with the load balancer; ConfigurableMoE owns it.
        - ``layer_idx=None``: the wrapper passes the real ``layer_idx`` to
          ``MoE.__init__`` to drive load-balancer setup. The backend
          receives ``None`` so its own EPLB hooks no-op until we sync the
          real values via ``_BACKEND_SYNC_ATTRS`` below.
        - ``skip_create_weights_in_init=True`` (via contextmanager): weights
          depend on ``layer_load_balancer`` / ``initial_local_expert_ids``
          / etc., which only become known after the sync, so allocation waits
          for the caller's ``create_weights()``.
        """
        from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend

        backend_model_config = self._backend_model_config(quant_config)
        with self._temporarily_skip_weight_creation(backend_model_config):
            backend = create_moe_backend(
                moe_cls=moe_cls,
                routing_method=self.routing_method,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                dtype=self.dtype,
                reduce_results=self.reduce_results,
                model_config=backend_model_config,
                aux_stream_dict=self.aux_stream_dict,
                weight_loading_mode=self.weight_loading_mode,
                bias=self.bias,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                layer_idx=None,
                init_load_balancer=False,
                activation=self.activation,
            )

        # Backend acceptance is validated by ``_install_backend_dependents``
        # instead of here so the validation hook can inspect ``self.comm``,
        # which that method assigns. Backends like ``MegaMoECuteDsl`` rely on
        # that to enforce ``moe.comm is None`` without ``getattr`` guards.
        self.backend = backend
        self._reject_non_divisible_ep_backend()
        self.use_flashinfer = getattr(self.backend, "use_flashinfer", False)

        # Mirror wrapper-owned EPLB / layer-id state onto the backend so any
        # backend code path that reads e.g. ``self.layer_load_balancer`` or
        # ``self.num_slots`` sees the real values resolved by MoE.__init__.
        for attr in _BACKEND_SYNC_ATTRS:
            setattr(self.backend, attr, getattr(self, attr))
        # ``expert_size_per_partition`` may have just changed (EPLB slots),
        # and it sizes every per-expert activation constant.
        install_activation_params(self.backend)

        self._adopt_backend_routing_scales_dtype()

        # The format this identity was picked for, so a later ``create_weights``
        # can tell whether the layer has since been moved to a different one.
        self._bound_quant_key = _canonical_quant_key(quant_config)

    def _adopt_backend_routing_scales_dtype(self) -> None:
        """Have routing emit the scale precision the bound backend reads.

        Changes no numbers. ``MoEScheduler`` narrows ``token_final_scales`` to
        ``input_requirement.routing_scales_dtype`` either way, and the routing
        op takes its output dtype as a kernel template parameter while
        computing in float regardless -- so this only moves the conversion into
        that kernel. What it saves is the separate elementwise launch the cast
        would otherwise be, once per layer per forward, on every backend that
        asks for something narrower than float32 (TRTLLM-Gen asks for
        bfloat16).

        Bind time is the earliest this is answerable: the requirement belongs
        to the leaf, and which leaf serves this layer is not settled until the
        format is. The pre-split form of this lived in the model, where
        ``Qwen3Gate`` compared a resolved class against ``TRTLLMGenFusedMoE``;
        that is now the abstract family base, so the comparison could only
        return False once the family had leaves.

        Float32 is the only value safe to overwrite, being routing's own
        full-precision output. A routing method the model configured narrower
        has already dropped mantissa bits that no assignment here can restore,
        so it is left alone and the scheduler's assertion stays the authority
        on whether it conflicts with the backend. That is also what keeps a
        routing method shared across layers honest: if two of them bind to
        backends that disagree, the second finds a non-float32 dtype, declines,
        and the disagreement surfaces at that assertion instead of being
        decided by whichever layer bound last.
        """
        required = self.backend.input_requirement.routing_scales_dtype
        if required is None or required == torch.float32:
            return
        if getattr(self.routing_method, "output_dtype", None) != torch.float32:
            return
        self.routing_method.output_dtype = required

    def _register_dwdp(self) -> None:
        """Claim this layer for distributed weight sharing, if eligible.

        Separate from ``_install_backend_dependents`` because it is keyed on
        the layer, not on the backend: ``add_layer`` appends to the manager's
        list, so a rebind that re-runs the rest must not re-run this.

        Runs before the communication factory, which skips alltoall strategies
        for a DWDP layer (the VA path swaps ``param.data`` and the backend
        reads its own weight attrs, so no comm strategy is needed).

        Safe this late in the lifecycle: ``DwdpManager.setup()`` runs from
        ``py_executor_creator``, after weights are loaded.
        """
        if self.dwdp_manager is not None and self._should_enable_dwdp():
            self.enable_dwdp = True
            self.dwdp_manager.add_layer(layer_idx=self.layer_idx)

    def _install_backend_dependents(self) -> None:
        """Build everything downstream of the backend's class.

        Kept in the order the backends were written against: communication
        strategy, then validation, then the scheduler -- last so it may read
        any wrapper state (comm, aux_stream, event_dict, moe_max_num_tokens,
        dwdp_*) without ordering surprises.
        """
        self.comm = self._create_comm_strategy_auto()
        self.validate_backend(self.backend)
        # Selection is based on ``backend.scheduler_kind``, a class attribute.
        self.scheduler: MoEScheduler = create_moe_scheduler(self)

    def _reject_non_divisible_ep_backend(self) -> None:
        """Enforce the non-divisible-EP contract on the resolved backend.

        This is the wrapper half of the check documented in ``MoE.__init__``.
        ``_init_load_balancer`` cannot do it: the wrapper runs it before the
        backend exists, and the backend never runs it at all because it is
        constructed with ``init_load_balancer=False``. So this is the only place
        that can consult the class actually executing the ceil/floor partition.

        Skipped when EPLB is active -- there every rank holds ``num_slots //
        ep_size`` slots and ``_init_load_balancer`` already required that to
        divide evenly, so local slot counts are uniform whatever
        ``num_experts % ep_size`` is.
        """
        if self.backend is None or self.layer_load_balancer is not None:
            return
        if self.num_experts % self.ep_size == 0:
            return
        if type(self.backend)._supports_non_divisible_ep:
            return
        raise ValueError(
            f"{type(self.backend).__name__} does not support non-divisible EP: "
            f"num_experts ({self.num_experts}) must be divisible by ep_size "
            f"({self.ep_size}). Enable EPLB with num_slots divisible by "
            f"ep_size, pick a backend that opts in, or override "
            f"`_supports_non_divisible_ep = True` on that backend after "
            f"verifying its kernel/comm path handles ceil/floor partitioning."
        )

    def _supports_load_balancer(self) -> bool:
        """Check if this MoE implementation supports load balancer.

        ``MoE.__init__`` can query this before ``ConfigurableMoE`` has
        created ``self.backend``. In that initialization window, fall back to
        the wrapper-level DP/parallelism condition; ``validate_backend`` runs
        after backend construction and enforces the backend-specific answer.
        """
        # During initialization, backend might not be created yet.
        # Backend-specific support is checked later by validate_backend.
        if not hasattr(self, "backend") or self.backend is None:
            return self.use_dp and self.parallel_size > 1
        return self.backend._supports_load_balancer()

    @property
    def num_fused_shared_expert(self) -> int:
        """Expose the backend's fused-shared-expert count so model code (e.g.
        DeepseekV3 post_load_weights) sees it through this wrapper.

        Zero while unbound, and zero for a backend without fusion. Callers that
        need the answer *before* binding -- shared-expert TP sizing runs inside
        the model's ``__init__`` -- must ask ``will_fuse_shared_expert``
        instead, which does not need the instance.
        """
        if not self._backend_bound:
            return 0
        return getattr(self.backend, "num_fused_shared_expert", 0)

    def will_fuse_shared_expert(self) -> bool:
        """Whether the backend this layer binds folds the shared experts in.

        Answerable before ``create_weights`` has bound anything, which model
        code building the layer needs: DeepseekV3 sizes its shared-expert TP
        from this, and the ``GatedMLP`` that sizing feeds is constructed in the
        same ``__init__``.

        Predicted from ``self.quant_config`` as it stands now, which is the
        model-level format (plus any explicit override) -- exactly the input the
        backend used to be resolved from at construction. So the prediction and
        the eventual binding agree unless per-layer quantization later moves
        this layer to a format whose class does not fuse.

        A disagreement is only reachable at all with fusion turned on, which is
        opt-in per ``TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION``; with it unset every
        class answers zero and the prediction cannot be wrong.
        """
        moe_cls = self._resolve_backend_cls(self.quant_config)
        count = getattr(moe_cls, "fused_shared_expert_count", None)
        if count is None:
            return False
        return count(self.model_config) > 0

    def fuse_shared_expert(self, shared_experts):
        """Delegate shared-expert fusion to the backend.

        Unguarded because every caller reaches this through
        ``num_fused_shared_expert > 0`` above, which only a backend that
        implements fusion can report -- so a backend without the method is not
        asked for it.
        """
        return self.backend.fuse_shared_expert(shared_experts)

    def validate_config(self):
        """
        Validate configuration parameters

        Validates:
        - apply_router_weight_on_input: Only supports top-1 routing
        """
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, (
                "apply_router_weight_on_input only supports top-1 routing"
            )

    def _should_enable_dwdp(self) -> bool:
        # DWDP is currently supported only by CuteDSL backends, and only with
        # NVFP4 quantization.
        if not self.backend.capabilities.supports_dwdp:
            return False

        # DWDP rebinds the backend parameters, which would strand the localized
        # weight shards. Not enabling it is the correct outcome, not an error.
        if self.backend.uses_locality_domain:
            return False

        quant_config = getattr(self.backend, "quant_config", None)
        if quant_config is None:
            quant_config = getattr(self.model_config, "quant_config", None)
        if quant_config is None:
            return False

        quant_mode = getattr(quant_config, "layer_quant_mode", None)
        return bool(
            quant_mode is not None and hasattr(quant_mode, "has_nvfp4") and quant_mode.has_nvfp4()
        )

    def _get_quant_config_dict(self, model_config: ModelConfig) -> Optional[Dict]:
        """
        Extract quantization configuration from model_config

        """
        if model_config.quant_config is None:
            return None

        quant_mode = model_config.quant_config.layer_quant_mode
        return {
            "has_fp8_qdq": quant_mode.has_fp8_qdq()
            if hasattr(quant_mode, "has_fp8_qdq")
            else False,
            "has_nvfp4": quant_mode.has_nvfp4() if hasattr(quant_mode, "has_nvfp4") else False,
            "has_w4afp8": quant_mode.is_int4_weight_only_per_group()
            if hasattr(quant_mode, "is_int4_weight_only_per_group")
            else False,
            "has_fp8_block_scales": quant_mode.has_fp8_block_scales()
            if hasattr(quant_mode, "has_fp8_block_scales")
            else False,
        }

    @staticmethod
    def _dp_padded_num_rows(all_rank_num_tokens: List[int]) -> int:
        """Padded total rows after DP dispatch: num_dp_ranks * max_tokens_per_rank."""
        return len(all_rank_num_tokens) * max(all_rank_num_tokens)

    def calculate_num_chunks(self, all_rank_num_tokens: List[int]) -> int:
        """
        Calculate how many chunks are needed based on total tokens after dispatch.

        When using DP communication, the dispatch (AllGather/AllToAll) collects
        tokens from all DP ranks, so total tokens = num_dp_ranks * max_tokens_per_rank.
        """
        if self.use_dp and self.comm is not None:
            num_rows = self._dp_padded_num_rows(all_rank_num_tokens)
        elif self.enable_dwdp:
            # DWDP prefetches expert weights instead of dispatching tokens, so a
            # rank only ever processes its own tokens, never more. Keyed off
            # ``enable_dwdp`` so no non-DWDP path changes the branch it takes.
            num_rows = max(all_rank_num_tokens)
        else:
            # non-DP: no cross-rank dispatch. The scheduler fills all_rank_num_tokens
            # from [x.shape[0]] before calling here, so it must be a single-element list.
            assert len(all_rank_num_tokens) == 1, (
                f"non-DP path expects a single-element list, got {len(all_rank_num_tokens)}"
            )
            num_rows = all_rank_num_tokens[0]
        return (num_rows + self.moe_max_num_tokens - 1) // self.moe_max_num_tokens

    def split_chunk(self, split_token_num: int, split_num_chunks: int) -> List[int]:
        """
        Split token count into multiple chunks as evenly as possible

        """
        val_div = split_token_num // split_num_chunks
        val_mod = split_token_num % split_num_chunks
        split_chunk_size_list = [val_div + 1] * val_mod + [val_div] * (split_num_chunks - val_mod)
        return split_chunk_size_list

    def determine_communication_method(
        self, all_rank_num_tokens: List[int], num_chunks: int
    ) -> None:
        """
        Determine and setup communication method with automatic fallback

        This method:
        1. Returns early if comm is None or already AllGather (nothing to validate)
        2. Validates if current AllToAll strategy can be used for given workload
        3. Falls back to AllGather if current strategy cannot be used (logs info message)

        After calling this method, use enable_alltoall to check which method is active.

        Args:
            all_rank_num_tokens: Token counts per rank
            num_chunks: Number of chunks

        Side effects:
            - May switch self.comm to AllGather if current strategy cannot be used

        Note: This method does NOT create strategy if None (creation happens lazily elsewhere).
              It only validates and potentially falls back existing AllToAll strategies.

        """

        # Early return if nothing to validate:
        # - None: Atten is TP or single rank, no communication needed
        # - AllGather: Already using fallback strategy, no validation needed
        if self.comm is None or isinstance(self.comm, AllGatherReduceScatter):
            return

        # Check if current strategy can be used
        feasible_workload = self.comm.is_workload_feasible(all_rank_num_tokens, num_chunks)

        if not feasible_workload:
            all_rank_max_num_tokens = max(all_rank_num_tokens)
            logger.info(
                f"Communication strategy {self.comm.__class__.__name__} "
                f"cannot be used (num_chunks={num_chunks}, max_num_tokens={all_rank_max_num_tokens}). "
                f"Falling back to AllGatherReduceScatter."
            )

            self.comm.destroy()
            self.comm = AllGatherReduceScatter(mapping=self.mapping)

    def destroy(self):
        """Release communication resources.

        Must be called on ALL ranks before the module is discarded.
        DeepEP Buffer.__del__ calls intranode::barrier (a collective op);
        without an explicit, synchronous release, non-deterministic GC
        timing across ranks causes some to enter the barrier while others
        proceed, resulting in an indefinite hang.

        Prefer using ConfigurableMoE as a context manager (``with``) so
        that destroy() is called automatically on scope exit.
        """
        if self.comm is not None:
            self.comm.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.destroy()

    def _create_comm_strategy_auto(self) -> Optional[Communication]:
        """
        Auto-create the best communication strategy based on hardware and configuration

        Uses factory to select optimal strategy. Backends whose fused kernel
        owns cross-rank exchange (``scheduler_kind=FUSED_COMM``) skip
        host-side comm entirely; layering Communication.dispatch / combine
        on top of the fused exchange would double-count traffic and break
        the in-kernel NVLink barrier semantics.

        DWDP VA path: returns None — there is no expert parallelism from the
        backend's perspective (fixup_moe_backends sets ep_size=1 / slot_start=0
        so the kernel sees full weights via param.data pointer swap).
        """
        if self.backend.scheduler_kind == MoESchedulerKind.FUSED_COMM:
            return None
        if self.enable_dwdp:
            return None
        return CommunicationFactory.create_strategy(
            model_config=self.model_config,
            num_experts=self.num_experts,
            num_slots=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            expert_size_per_partition=self.expert_size_per_partition,
            payload_in_workspace=False,  # ConfigurableMoE does not use workspace output for now
            # Currently the TRTLLMGEN reduce sum internally.
            # Keep updated with more supported backends.
            alltoall_result_do_sum=True,
            use_flashinfer=self.use_flashinfer,
            hidden_size=self.hidden_size,
            communication_method=self.communication_method,
        )

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        lora_params: Optional[Dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward entry point.

        Acts as a thin wrapper that:

        1. Validates / fills ``output_dtype``.
        2. Delegates the per-pass execution to ``self.scheduler`` (chosen
           once at init time from ``backend.scheduler_kind``).
        3. Records DWDP compute/prefetch (per layer, not per chunk).
        4. Advances the EPLB ``repeat_idx``.

        DP-padding handling and chunking live in the scheduler.
        """
        if not self._backend_bound:
            raise RuntimeError(
                f"MoE layer {self.layer_idx} has no backend yet: the backend's "
                f"class is this layer's quantization format, so it is bound in "
                f"create_weights(), once __post_init__ has settled that format. "
                f"Call create_weights() (or let __post_init__ run) before "
                f"forward."
            )
        input_ids = kwargs.get("input_ids")

        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
        else:
            output_dtype = x.dtype

        # DWDP: wait for prefetch to complete and swap backend weight param.data
        # to the composite (full-experts) tensor. Per-layer, not per-chunk —
        # owned at the wrapper so the scheduler does not run it twice (e.g.
        # external-comm may enter via single- or multi-chunk paths).
        if self.enable_dwdp:
            self.dwdp_manager.wait_and_bind(self.backend, self.layer_idx)

        outputs = self.scheduler.forward(
            x,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            input_ids=input_ids,
            lora_params=lora_params,
        )

        # DWDP: record compute and trigger next prefetch (per-layer, not per-chunk).
        # Owned at the wrapper because schedulers must not run it twice (external-comm
        # might enter via single- or multi-chunk paths).
        if self.enable_dwdp:
            self.dwdp_manager.record_compute_and_prefetch_next(self.layer_idx)

        # EPLB repeat counter: advance once per forward, regardless of chunk count.
        # Schedulers are forbidden from rotating ``repeat_idx`` themselves.
        self.repeat_idx = (self.repeat_idx + 1) % self.repeat_count

        return outputs

    # ========== Backend Validation ==========

    def validate_backend(self, backend: MoE | MoEImplBase | None) -> None:
        """Validate MoE backend compatibility with this ConfigurableMoE.

        Generic checks (always run):
          1. ``backend`` is not None.
          2. If EPLB is enabled, the backend must support routing
             separation (``backend._supports_load_balancer()``).

        Backend-specific checks are delegated to
        ``backend.validate_configurable_moe(self)``; backends with extra
        constraints (e.g. fused-comm backends rejecting dynamic
        EPLB) override that hook.

        Call site contract: invoked from ``__init__`` *after* every
        wrapper-owned attribute is assigned (EPLB / num_slots /
        ep_size via ``MoE.__init__`` -> ``_init_load_balancer``,
        ``self.comm`` from ``_create_comm_strategy_auto``, and
        ``self.moe_max_num_tokens`` from ``model_config``). Backend
        hooks can therefore inspect them directly without ``getattr``
        guards or sentinel defaults.
        """
        if backend is None:
            raise ValueError("Backend cannot be None")

        if self._using_load_balancer() and not backend._supports_load_balancer():
            raise ValueError(
                f"EPLB is enabled but backend {backend.__class__.__name__} "
                f"does not support load balancer. "
                f"Either disable EPLB or use a backend that supports load balancer."
            )

        backend.validate_configurable_moe(self)

    def create_weights(self):
        """Bind the backend to this layer's final format, then allocate.

        The wrapper's half of the quantization lifecycle, and the reason
        binding lives here rather than in ``__init__``: this runs from
        ``__post_init__`` *after* ``apply_layerwise_quant_config`` and
        ``apply_quant_config_exclude_modules``, so ``self.quant_config`` is
        final. A backend whose class *is* one quantization format therefore
        gets picked for the format the layer actually runs.

        ``Linear``, ``Attention``, and ``MLA`` already use ``create_weights``
        as the post-quantization hook for the same reason -- ``TrtllmAttention``
        rebuilds its ``FmhaManager`` from there.

        Three outcomes, and the last is why the guard is not a plain
        ``if self._backend_bound: return``:

        - unbound: resolve, construct, allocate, then build everything derived
          from the backend's class.
        - bound to this same format: idempotent. Reached on every run, because
          the backend is a registered submodule and ``__post_init__`` walks
          ``named_modules()``, so it gets its own ``create_weights`` call right
          after this one.
        - bound to a *different* format: rebound, see ``_rebind_backend_to``.
          Reachable only from the eager binding in ``__init__``, which an
          exclusion can then move; returning early there would leave the layer
          executing one format's kernels against another's config, silently.
        """
        quant_config = self.quant_config

        if not self._backend_bound:
            self._create_backend(quant_config, self._resolve_backend_cls(quant_config))
            self.backend.quant_config = quant_config
            result = self.backend.create_weights()
            # Both after allocation, because that is the order the backends
            # were written against: weights exist before the comm strategy and
            # the scheduler.
            self._register_dwdp()
            self._install_backend_dependents()
            self._backend_bound = True
            return result

        swapped = False
        if _canonical_quant_key(quant_config) != self._bound_quant_key:
            swapped = self._rebind_backend_to(quant_config)

        self.backend.quant_config = quant_config
        # Re-install because ``expert_size_per_partition`` sets the length of
        # every per-expert activation constant, and it is wrapper-owned: the
        # install inside the backend's own construction ran before the EPLB
        # attrs were mirrored onto it.
        #
        # Guarded because a re-install re-materializes from ``activation``,
        # undoing the in-place division NVFP4TRTLLMGenFusedMoEBaseMethod
        # applies to beta and clamp.
        if not self.backend._weights_created:
            install_activation_params(self.backend)
        result = self.backend.create_weights()
        if swapped:
            # After allocation, matching the first-bind order above: the comm
            # strategy and the scheduler are written against a backend whose
            # weights exist.
            self._install_backend_dependents()
        return result

    def _rebind_backend_to(self, quant_config: Optional[QuantConfig]) -> bool:
        """Re-resolve for ``quant_config``; swap the backend if it moved.

        Restores what one class serving every format used to do for free: it
        re-read ``quant_config`` at allocation time to pick its quant method.
        Now that the format is the implementation's identity, the same
        correction has to replace the implementation.

        Returns whether the backend was replaced, so the caller can rebuild
        everything derived from its class *after* allocating weights rather
        than before -- the order the backends were written against.

        DWDP registration is not repeated: ``add_layer`` appends to the
        manager's list, so re-running it would claim this layer twice. The
        eligibility *conclusion* is about the backend, though, and there is no
        way to withdraw a claim, so a swap that would change it is refused
        rather than left stale.

        When the same class serves both formats there is nothing to replace,
        and the caller's re-run of ``quant_method.create_weights`` is what
        adopts the new format. That re-run registers only what the new method
        needs, so a scale tensor the previous method registered stays in
        ``_parameters`` -- carried over from the single-class era rather than
        introduced here, and out of reach of a wrapper that does not know which
        names a method claimed.
        """
        moe_cls = self._resolve_backend_cls(quant_config)
        if moe_cls is type(self.backend):
            # Record it so the comparison above does not re-resolve on every
            # subsequent call.
            self._bound_quant_key = _canonical_quant_key(quant_config)
            return False

        # Explicitly, before the last reference to it is dropped: DeepEP's
        # ``Buffer.__del__`` enters a collective barrier, so letting GC decide
        # when the old strategy is released lets ranks reach that barrier at
        # different times and hang. ``_install_backend_dependents`` builds the
        # replacement once the new backend has its weights.
        if self.comm is not None:
            self.comm.destroy()
            self.comm = None

        previously_claimed_for_dwdp = self.enable_dwdp
        self._create_backend(quant_config, moe_cls)
        if previously_claimed_for_dwdp and not self._should_enable_dwdp():
            raise ValueError(
                f"Layer {self.layer_idx} was claimed for distributed weight "
                f"sharing while bound to a backend that supports it, and "
                f"per-layer quantization has since moved it to "
                f"{moe_cls.__name__}, which does not. The claim cannot be "
                f"withdrawn, so this layer would swap parameter data on a "
                f"backend that does not expect it. Drop the layerwise "
                f"override or the exclusion on this layer, or disable DWDP."
            )
        return True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        """
        Load weights - delegated to backend

        """
        assert hasattr(self.backend, "load_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement load_weights()"
        )
        result = self.backend.load_weights(weights, allow_partial_loading)
        if weights:
            self._weights_transformed = False
        return result

    def transform_weights(self) -> None:
        """
        Transform weights - delegated to backend

        """
        if getattr(self, "_weights_transformed", False):
            return
        assert hasattr(self.backend, "transform_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement transform_weights()"
        )
        self.backend.transform_weights()
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        """
        Cache derived state - delegated to backend

        """
        assert hasattr(self.backend, "cache_derived_state"), (
            f"Backend {self.backend.__class__.__name__} must implement cache_derived_state()"
        )
        self.backend.cache_derived_state()

    def process_weights_after_loading(self):
        """
        Process weights after loading - delegated to backend

        """
        assert hasattr(self.backend, "process_weights_after_loading"), (
            f"Backend {self.backend.__class__.__name__} must implement process_weights_after_loading()"
        )
        return self.backend.process_weights_after_loading()

    def pre_reload_weights(self):
        """
        Pre reload weights - delegated to backend
        """
        assert hasattr(self.backend, "pre_reload_weights"), (
            f"Backend {self.backend.__class__.__name__} must implement pre_reload_weights()"
        )
        return self.backend.pre_reload_weights()

    # ========== Communication and Quantization Properties ==========

    @property
    def enable_alltoall(self):
        """
        Check if alltoall is enabled

        This delegates to the communication strategy to determine if alltoall is available.

        """
        if self.comm is None:
            return False
        # Simplified check - AllGather strategy means no alltoall
        return not isinstance(self.comm, AllGatherReduceScatter)

    @property
    def _weights_created(self):
        """Whether this layer's weights exist, which the backend owns.

        False while unbound, which is a real answer rather than a guard: with
        no backend there are no weights. ``apply_quant_config_exclude_modules``
        reads this through ``hasattr`` and then writes it, so an unbound layer
        has to answer both without raising.
        """
        if not self._backend_bound:
            return False
        return self.backend._weights_created

    @_weights_created.setter
    def _weights_created(self, value: bool) -> None:
        """Update backend weight state during post-init quantization changes.

        A no-op while unbound: nothing has been allocated to invalidate, and
        the exclusion that is writing this also rewrote ``self.quant_config``,
        which is what ``create_weights`` resolves the backend from.
        """
        if not self._backend_bound:
            return
        self.backend._weights_created = value

    # ========== Explicit Backend Attribute Proxies ==========
    # These properties delegate to backend for commonly accessed attributes
    # TODO: Unify the property access to backend in ConfigurableMoE.
    # At the same time, we need to keep the existing test cases working.

    @property
    def quant_method(self):
        """Delegate quant_method to backend"""
        return getattr(self.backend, "quant_method", None)

    @property
    def w3_w1_weight(self):
        """Delegate w3_w1_weight to backend"""
        return getattr(self.backend, "w3_w1_weight", None)

    @property
    def w2_weight(self):
        """Delegate w2_weight to backend"""
        return getattr(self.backend, "w2_weight", None)

    @property
    def has_nvfp4(self):
        """Delegate has_nvfp4 to backend"""
        return getattr(self.backend, "has_nvfp4", False)

    def forward_fake(
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
        Fake forward for shape inference during torch.compile

        Delegates to backend's forward_fake if available, otherwise calls parent's forward_fake

        Args:
            x: Input tensor
            router_logits: Router logits for expert selection
            do_finalize: Whether to finalize MoE output
            output_dtype: Output data type
            all_rank_num_tokens: Token counts per rank
            use_dp_padding: Whether to use data parallel padding
            **kwargs: Additional arguments

        Returns:
            Empty tensor(s) with correct shape for torch.compile
        """
        if hasattr(self.backend, "forward_fake"):
            # Backend has forward_fake, delegate to it
            return self.backend.forward_fake(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )
        else:
            # Backend doesn't have forward_fake, use parent's implementation
            return super().forward_fake(
                x,
                router_logits,
                do_finalize=do_finalize,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                **kwargs,
            )
