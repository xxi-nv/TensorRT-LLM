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

from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from packaging.version import Version

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, Fp4QuantizedTensor
from .activation import MoEActivationSupport
from .cutlass import CutlassFusedMoEBase, TrtllmCutlassNvfp4Impl
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    require_comm_plan,
)
from .impl_environment import MoEDep
from .interface import _reject

# Shared MoE output buffer pool, keyed by (max_num_tokens, hidden_size, dtype,
# device). ``B12xMoEWrapper.__init__`` allocates a private
# ``(max_num_tokens, hidden_size)`` output tensor per instance; with one
# wrapper per MoE layer that is ``num_layers * max_num_tokens * hidden_size``
# bytes of GPU memory holding identical-shape buffers that are written
# sequentially. We fold them into a single shared buffer because MoE layers
# run sequentially on the same CUDA stream, and the wrapper consumes its
# previous output before the next layer is dispatched.
_SHARED_MOE_OUTPUT_BUF: dict = {}

# ActivationType -> b12x activation string. b12x currently exposes "relu2"
# (Nemotron-style x * relu(x)) and "silu" (SwiGLU-style x * silu(gate)).
_ACTIVATION_MAP = {
    ActivationType.Relu2: "relu2",
    ActivationType.Swiglu: "silu",
}

# FlashInfer's SM12x W4A16 fused MoE (0.6.17 and 0.6.18) selects a TC-decode
# "ultra-wide" FC2 tile of tile_n=512 / tile_k=32 for small m whenever that
# collapses FC2 to FC1's wave count (m=3 and m=4 for Qwen3.6-35B-A3B on the
# 188-SM RTX PRO 6000). tile_k=32 sits below the tile_k>=64 floor of its
# generic ``_candidate_tile_fits`` check, which the auto-selection skips but
# the ``force_tile_config`` re-pin across the custom-op boundary does not, so
# the kernel it just selected is rejected with "force_tile_config fc2 tile
# (tile_k=32, tile_n=512) does not fit ..." and CUDA-graph warmup aborts
# executor init (nvbug 6721561). Accepting the ultra tile in the re-validation
# under the same rule that produced it is the upstream fix; this shim mirrors
# it so the pinned wheel behaves like the fixed one. Drop it once the
# flashinfer pin carries ``_ultra_wide_fc2_tile_fits``.
_FLASHINFER_W4A16_ULTRA_WIDE_FC2_TILE = (512, 32, 256)  # (tile_n, tile_k, cta_threads)
_FLASHINFER_W4A16_ULTRA_WIDE_FC2_SINCE = Version("0.6.17")


def _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() -> bool:
    """Make flashinfer's W4A16 ``force_tile_config`` re-validation accept the
    ultra-wide FC2 tile its own auto-selection produces.

    Idempotent. Returns True when the shim is active, False when flashinfer is
    absent, predates the ultra-wide override, or already ships the fix.
    """
    try:
        import flashinfer
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel as kernel
    except ImportError:
        return False
    if Version(flashinfer.__version__) < _FLASHINFER_W4A16_ULTRA_WIDE_FC2_SINCE:
        return False
    if hasattr(kernel, "_ultra_wide_fc2_tile_fits"):
        return False
    upstream_fits = getattr(kernel, "_candidate_tile_fits", None)
    smem_footprint = getattr(kernel, "_shared_memory_footprint", None)
    scale_group_size = getattr(kernel, "_scale_group_size", None)
    if upstream_fits is None or smem_footprint is None or scale_group_size is None:
        return False
    if getattr(upstream_fits, "_trtllm_ultra_wide_fc2_shim", False):
        return True

    def candidate_tile_fits(
        *,
        problem_n: int,
        problem_k: int,
        cta_m_blocks: int,
        tile_n: int,
        tile_k: int,
        cta_threads: int,
        max_shared_mem: int,
        scale_format: str = "e4m3_k16",
        weight_layout: str = "packed",
        allow_logical_tail: bool = False,
    ) -> bool:
        if (int(tile_n), int(tile_k), int(cta_threads)) != _FLASHINFER_W4A16_ULTRA_WIDE_FC2_TILE:
            return upstream_fits(
                problem_n=problem_n,
                problem_k=problem_k,
                cta_m_blocks=cta_m_blocks,
                tile_n=tile_n,
                tile_k=tile_k,
                cta_threads=cta_threads,
                max_shared_mem=max_shared_mem,
                scale_format=scale_format,
                weight_layout=weight_layout,
                allow_logical_tail=allow_logical_tail,
            )
        # Same rule as the auto-selection override: exact N/K tiling, one scale
        # group per k-tile, and shared-memory fit.
        if int(problem_n) % int(tile_n) != 0 or int(problem_k) % int(tile_k) != 0:
            return False
        if int(tile_k) % int(scale_group_size(scale_format)) != 0:
            return False
        smem_bytes = smem_footprint(
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            scale_format=scale_format,
            weight_layout=weight_layout,
        )
        return int(smem_bytes) <= int(max_shared_mem)

    candidate_tile_fits._trtllm_ultra_wide_fc2_shim = True
    candidate_tile_fits._trtllm_upstream_fits = upstream_fits
    kernel._candidate_tile_fits = candidate_tile_fits
    logger.info_once(
        f"flashinfer {flashinfer.__version__}: accepting the SM12x W4A16 ultra-wide FC2 "
        "tile (tile_n=512, tile_k=32) in force_tile_config re-validation (nvbug 6721561).",
        key="flashinfer_w4a16_ultra_wide_fc2_tile_shim",
    )
    return True


class CuteDslB12xFusedMoEBase:
    """What the two B12x SM120/121 leaves share.

    A mixin rather than a base with its own MoE ancestry, because the two
    leaves differ in exactly that ancestry: the NVFP4 leaf inherits the CUTLASS
    NVFP4 leaf so it can hand prefill chunks to the grouped GEMM, while the
    W4A16 leaf inherits only ``CutlassFusedMoEBase`` for construction and the
    weight lifecycle and never issues a CUTLASS op at all. Everything above
    that split -- the shared gates, the wrapper construction, the decode call
    -- lives here.

    Kept as the family name (``CuteDslB12xFusedMoE`` is an alias) so that
    ``isinstance`` / ``issubclass`` against it still matches both leaves.

    ``CuteDslFusedMoE.run_moe_nvfp4*`` is never reached from either leaf, so
    the ``AuxStreamType.MoeOutputMemset`` / ``EventType`` entries it needs are
    not set up and ``event_dict`` can be None. Restate them, and
    ``limit_when_absent``, before routing any CuteDSL path through these
    classes.
    """

    # No code path here reads ``w3_w1_bias`` / ``w2_bias`` or fuses LoRA, and
    # ``supports_eplb`` stays False -- which is why the gates below have to
    # decline ``d.eplb_enabled`` explicitly, since the inherited
    # ``_supports_load_balancer()`` answers True.
    # ``supports_apply_router_weight_on_input`` is False where the CUTLASS
    # parent says True: only an NVFP4 prefill chunk reaches the inherited
    # ``run_moe``, while the decode path hands ``token_final_scales`` straight
    # to the flashinfer b12x wrapper, which has no declared behaviour for the
    # ``None`` the scheduler's fold leaves there.
    capabilities = MoEStaticCapability(
        supports_moe_lora=False,
        supports_dwdp=True,
        supports_expert_bias=False,
        supports_apply_router_weight_on_input=False,
    )

    # Same value the CUTLASS parent declares, pinned so a change there cannot
    # silently retarget these backends.
    input_requirement = MoEInputRequirement(routing_scales_dtype=torch.float32)

    # The kinds ``_ACTIVATION_MAP`` above gates on. The b12x decode kernel takes
    # no activation constants, so none are declared.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2})
    )

    # Read by ``ConfigurableMoE._reject_non_divisible_ep_backend()``; moot in
    # practice because the gates below reject ``ep_size != 1`` outright.
    _supports_non_divisible_ep: bool = True

    # SM versions on which the FlashInfer b12x MoE kernels are available.
    # SM120 = desktop Blackwell (RTX 5090 / GB202); SM121 = GB10 / DGX Spark.
    _SUPPORTED_SM_VERSIONS = frozenset({120, 121})

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        return self.has_nvfp4

    @classmethod
    def _check_b12x_common(cls, p: MoEProblem, d: MoEDeployment) -> Optional[MoEEligibility]:
        """The gates both leaves apply. Quantization format is checked by each.

        Returns a rejection, or ``None`` when every shared condition holds.
        """
        sm_version = d.env.sm
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            sm_list = "/".join(f"SM{v}" for v in sorted(cls._SUPPORTED_SM_VERSIONS))
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"{cls.__name__} requires {sm_list}, got SM{sm_version}",
            )
        if p.dtype_act not in {torch.float16, torch.bfloat16}:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"{cls.__name__} requires float16 or bfloat16 activation dtype (got {p.dtype_act})",
            )
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support swiglu_gptoss_style",
            )
        if p.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support activation {p.activation}; "
                f"supported: {supported}",
            )
        # The decode kernel ships in the FlashInfer wheel.
        if not d.env.has_dep(MoEDep.FLASHINFER):
            return _reject(
                MoERejectReason.DEP_MISSING,
                f"{cls.__name__} requires the flashinfer package",
            )
        # The only backend the construction-time allow-list turned down that
        # never said so during selection, so an EPLB run resolved to b12x and
        # then died in the factory. Declining here degrades with the usual
        # warning instead, matching VanillaMoE / TritonFusedMoE / Marlin.
        if d.eplb_enabled:
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED,
                f"{cls.__name__} does not support the MoE load balancer",
            )
        # No expert-parallel dispatch/combine kernel: EP must stay at 1.
        if d.ep_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"{cls.__name__} requires ep_size == 1 (got {d.ep_size})",
            )
        # Attention-DP is a separate axis from EP: with moe_tp == tp the layer
        # can have ep_size == 1 and still sit behind a DP allgather /
        # reducescatter that the b12x wrapper has never been exercised under.
        # ``use_dp and parallel_size > 1`` is exactly ``mapping.dp_size > 1``
        # (``Mapping.dp_size`` is ``tp_size`` when attention-DP is on).
        if d.use_dp and d.parallel_size > 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"{cls.__name__} does not support attention-DP (parallel_size={d.parallel_size})",
            )
        return None

    def __init__(self, *args, **kwargs):
        # ``ModelConfig`` is consumed by the inherited ``__init__`` for cache
        # / mapping setup but isn't kept on ``self``. The b12x wrapper needs the
        # ``use_cuda_graph`` flag at construction time, so capture it here
        # before delegating.
        model_config = kwargs.get("model_config", None)
        self._b12x_use_cuda_graph = bool(getattr(model_config, "use_cuda_graph", False))

        super().__init__(*args, **kwargs)

        # No alltoall guard here: alltoall is picked by the wrapper's
        # communication strategy, and the gates already reject the topologies
        # that could pick it (ep_size != 1, attention-DP with parallel_size > 1).
        self._b12x_weights: Optional[dict] = None
        self.b12x_wrapper = None

    def _b12x_nvfp4_quant_method(self):
        """The b12x-aware NVFP4 quant method, or ``None`` if the layer is not
        NVFP4-quantized.

        Both leaves carry NVFP4 weights, so both want this: weight prep (SF
        un-normalization, ``convert_sf_to_mma_layout``, ``B12xMoEWrapper``
        instantiation) lives next to the rest of the NVFP4 quant-method family
        rather than in the backend. What differs is what a non-NVFP4 layer
        means, so each leaf decides that for itself -- there is no
        ``_get_quant_method`` above this mixin to fall back to.
        """
        if (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
            and self.quant_config.layer_quant_mode.has_nvfp4()
        ):
            from .quantization import NVFP4CuteDslB12xFusedMoEMethod

            return NVFP4CuteDslB12xFusedMoEMethod()
        return None

    # ``post_load_weights`` is inherited from the Cutlass base and dispatches to
    # ``self.quant_method.transform_weights(self)`` -- here that is
    # ``NVFP4CuteDslB12xFusedMoEMethod``, which performs the SF
    # un-normalization, the ``convert_sf_to_mma_layout`` reshape, the
    # ``B12xMoEWrapper`` instantiation and the cross-layer shared output buffer
    # dance. The wrapper and the bundled weight dict are attached to the module
    # as ``self.b12x_wrapper`` / ``self._b12x_weights``, which the decode path
    # below consumes.

    @staticmethod
    def _reject_fp4_input(x: Union[torch.Tensor, Fp4QuantizedTensor], who: str) -> None:
        """b12x quantizes activations itself and cannot take pre-quantized FP4."""
        if isinstance(x, Fp4QuantizedTensor):
            raise ValueError(
                f"{who} does not accept Fp4QuantizedTensor input on the b12x "
                "path; b12x performs its own input quantization."
            )

    @nvtx_range("[b12x] decode")
    def _run_b12x_decode(self, ctx: MoERunContext) -> torch.Tensor:
        """Run the FlashInfer b12x kernel. The path both leaves reach."""
        plan = require_comm_plan(self, ctx)
        moe_output = plan.moe_output
        if self.b12x_wrapper is None or self._b12x_weights is None:
            raise RuntimeError(
                f"{type(self).__name__}.run_moe called before "
                "process_weights_after_loading completed."
            )
        if ctx.x_sf is not None:
            raise ValueError(
                f"{type(self).__name__} expects unquantized input (x_sf=None) "
                "on the b12x path; got a precomputed scale factor."
            )

        # Annotate the kwargs spread + wrapper entry separately so we can
        # attribute the per-layer Python dispatch cost vs. the kernel cost.
        with nvtx_range("[b12x] wrapper.run"):
            out = self.b12x_wrapper.run(
                x=ctx.x,
                token_selected_experts=ctx.token_selected_experts,
                token_final_scales=ctx.token_final_scales,
                **self._b12x_weights,
            )

        # B12xMoEWrapper allocates its own output buffer for CUDA-graph
        # compatibility. If the caller provided ``moe_output`` (e.g. an alltoall
        # workspace tensor), copy into it; the gates reject the topologies that
        # could pick alltoall, so this is a defensive path for future
        # workspace-driven uses.
        if moe_output is not None:
            with nvtx_range("[b12x] out_copy"):
                moe_output.copy_(out)
            return moe_output
        return out


#: Family name. Both leaves are subclasses, so ``isinstance`` / ``issubclass``
#: against it matches either -- which is what every gate outside this module
#: wants. It is not itself constructible: it carries no MoE ancestry.
CuteDslB12xFusedMoE = CuteDslB12xFusedMoEBase


class CuteDslB12xNvfp4FusedMoE(CuteDslB12xFusedMoEBase, TrtllmCutlassNvfp4Impl):
    """NVFP4 on SM120 / SM121: CUTLASS for prefill, b12x for decode.

    Inherits the CUTLASS NVFP4 leaf because the prefill chunk it routes away is
    NVFP4 grouped GEMM, so ``super().run_moe`` / ``super().quantize_input``
    reach exactly the right implementation.

    Both weight layouts are staged once at load time on the same module -- the
    inherited ``post_load_weights`` builds the standard CUTLASS NVFP4 layout
    first, then ``NVFP4CuteDslB12xFusedMoEMethod`` adds the b12x views and
    wrapper on top -- so the per-call switch below picks a compute path and
    never swaps a layout.
    """

    # Prefill chunks (``x.shape[0] >= threshold``) route via CUTLASS NVFP4
    # GroupGEMM; decode (``x.shape[0] < threshold``) uses b12x. 64 cleanly
    # separates conc=1 prefill (m=2048 with ``max_num_tokens=2048``) from
    # decode (m=1) and stays robust against future chunked-prefill splits
    # that might shrink prefill chunk size.
    _PREFILL_VIA_CUTLASS_THRESHOLD = 64

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        if p.quant_algo != QuantAlgo.NVFP4:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"{cls.__name__} only supports NVFP4 quantization (got quant_algo={p.quant_algo})",
            )
        rejection = cls._check_b12x_common(p, d)
        return rejection if rejection is not None else MoEEligibility.ok()

    def _get_quant_method(self) -> object:
        # An unquantized layer keeps the CUTLASS leaf's own method, which is
        # what the prefill path would need anyway.
        return self._b12x_nvfp4_quant_method() or super()._get_quant_method()

    def _route_to_cutlass(self, x) -> bool:
        """Return ``True`` iff this call should take the inherited CUTLASS path.

        ``Fp4QuantizedTensor`` inputs always stay on the b12x path (which
        rejects them) so the existing error message is preserved.
        """
        return isinstance(x, torch.Tensor) and x.shape[0] >= self._PREFILL_VIA_CUTLASS_THRESHOLD

    @nvtx_range("[b12x] quantize_input")
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prefill chunks take the inherited NVFP4 quantization so the
        downstream ``run_moe`` can call CUTLASS NVFP4 GroupGEMM. Decode chunks
        pass through unchanged because b12x quantizes activations internally.
        """
        if self._route_to_cutlass(x):
            return super().quantize_input(x, post_quant_comm=post_quant_comm, **kwargs)
        self._reject_fp4_input(x, type(self).__name__)
        return x, None

    @nvtx_range("[b12x] run_moe")
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        if self._route_to_cutlass(ctx.x):
            # The inherited ``run_moe`` forwards ``output_dtype`` straight into
            # the C++ ``trtllm::fused_moe`` op, which requires a concrete
            # high-precision ``ScalarType`` (uint8 / FP4-packed activations are
            # rejected at the kernel epilogue with "Invalid output type Byte").
            # ``ConfigurableMoE.forward`` always fills ``output_dtype``, so this
            # only narrows the type for anything driving ``run_moe`` without it.
            _HIGH_PRECISION = {torch.float16, torch.bfloat16, torch.float32}
            cutlass_output_dtype = ctx.output_dtype
            if cutlass_output_dtype is None:
                cutlass_output_dtype = (
                    ctx.x.dtype
                    if isinstance(ctx.x, torch.Tensor) and ctx.x.dtype in _HIGH_PRECISION
                    else torch.bfloat16
                )
            return super().run_moe(
                replace(ctx, output_dtype=cutlass_output_dtype),
                workspace=workspace,
            )
        del workspace  # The b12x wrapper allocates its own intermediates.
        return self._run_b12x_decode(ctx)


class CuteDslB12xW4a16Nvfp4FusedMoE(CuteDslB12xFusedMoEBase, CutlassFusedMoEBase):
    """W4A16_NVFP4 on SM120 / SM121: b12x for every call.

    Deliberately does NOT inherit a CUTLASS leaf. This format never reaches a
    CUTLASS kernel -- there is no token-count threshold and no prefill
    fallback -- so inheriting one would advertise an execution path that
    cannot be taken. What it does take from ``CutlassFusedMoEBase`` is
    construction, the weight lifecycle and ``post_load_weights``, which is how
    ``NVFP4CuteDslB12xFusedMoEMethod`` gets to build the b12x wrapper.

    The FlashInfer W4A16 tile-selector shim is applied unconditionally here,
    rather than behind a ``quant_algo`` test, because reaching this class is
    already the condition it used to test for.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only the W4A16 decode path reaches flashinfer's W4A16 tile selector.
        _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation()

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        if p.quant_algo != QuantAlgo.W4A16_NVFP4:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"{cls.__name__} only supports W4A16_NVFP4 quantization "
                f"(got quant_algo={p.quant_algo})",
            )
        rejection = cls._check_b12x_common(p, d)
        return rejection if rejection is not None else MoEEligibility.ok()

    def _get_quant_method(self) -> object:
        quant_method = self._b12x_nvfp4_quant_method()
        if quant_method is None:
            # Unreachable via resolution: ``can_implement`` admits only
            # W4A16_NVFP4, whose quant mode reports NVFP4 weights. Stated
            # because this class has no CUTLASS leaf to defer to.
            raise ValueError(
                f"{type(self).__name__} requires an NVFP4-quantized layer, got "
                f"quant_config={self.quant_config}"
            )
        return quant_method

    @nvtx_range("[b12x] quantize_input")
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Always a pass-through: b12x consumes bf16 / fp16 and produces its
        own scale factors."""
        del post_quant_comm, kwargs
        self._reject_fp4_input(x, type(self).__name__)
        return x, None

    @nvtx_range("[b12x] run_moe")
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        del workspace  # The b12x wrapper allocates its own intermediates.
        return self._run_b12x_decode(ctx)
