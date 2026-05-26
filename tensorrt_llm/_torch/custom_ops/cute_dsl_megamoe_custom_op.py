# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuteDSL MegaMoE NVFP4 custom op + TunableRunner.

Wraps the ported ``Sm100MegaMoEKernel`` (see
``tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/``) into the
standard TRT-LLM CuteDSL op pattern used by
``cute_dsl_custom_ops.py``:

* :class:`Sm100MegaMoENvfp4Runner` is the :class:`TunableRunner` that
  owns the kernel compile cache, the candidate-tactic enumeration, and
  the per-launch ``cute.compile`` + invocation. Tactic representation
  is a tuple of JSON-friendly primitives so ``eval(repr(tactic))``
  round-trips (required by the autotuner cache).
* ``trtllm::cute_dsl_megamoe_nvfp4_blackwell`` is the registered torch
  custom op that the ``MegaMoECuteDsl`` backend calls from
  ``run_moe``. It runs ``AutoTuner.choose_one`` once per call to pick
  the best tactic and forwards to the runner.

The backend never instantiates :class:`Sm100MegaMoENvfp4Runner`
directly; this mirrors how ``CuteDslFusedMoE`` only consumes
``torch.ops.trtllm.cute_dsl_nvfp4_*`` and never reaches into
``cute_dsl_custom_ops.py`` for its inner runners. Keeping the boundary
here lets us evolve the tactic enumeration / compile cache without
touching the MoE backend.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..._utils import get_sm_version
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ..utils import get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2

__all__ = [
    "DEFAULT_MEGAMOE_TACTIC",
    "IS_MEGAMOE_OP_AVAILABLE",
    "enumerate_megamoe_candidate_tactics",
    "megamoe_activation_sf_bytes_per_row",
    "resolve_megamoe_group_hint",
    "validate_megamoe_tactic",
]

# Set to ``True`` if every symbol the op registration needs imports
# cleanly. ``False`` keeps the op unregistered so callers fall back via
# the factory instead of crashing the whole ``custom_ops`` package
# import. ``IS_CUTLASS_DSL_AVAILABLE`` only probes ``cutlass.cute``;
# half-installed or older cutlass-dsl wheels can still expose
# ``cutlass.cute`` but miss ``cutlass.torch`` / ``cutlass._mlir`` /
# ``cute_nvgpu`` / the symm-memory adapter symbols this op needs.
IS_MEGAMOE_OP_AVAILABLE: bool = False


# ---------------------------------------------------------------------------
# Tactic representation
# ---------------------------------------------------------------------------
#
# A tactic is a tuple of JSON-friendly primitives (lists / ints / bools /
# strings) so it round-trips through ``json.dumps``/``json.loads`` *and*
# ``eval(repr(tactic))`` — both are required by ``TunableRunner`` cache
# serialization. Order matches the kernel constructor kwargs.
#
#   (mma_tiler_mnk,        # list[int] of length 3
#    cluster_shape_mnk,    # list[int] of length 3
#    use_2cta_instrs,      # bool
#    resolved_group_hint,  # int (always resolved before cache lookup)
#    load_balance_mode,    # str: "static" | "atomic_counter"
#    use_bf16_redg)        # bool: form A (False) vs form B (True)
#
# Tuple wrapping makes the tactic hashable, which AutoTuner needs for the
# tactics cache. Lists nested inside the tuple are reconstructed from
# JSON intact.


DEFAULT_MEGAMOE_TACTIC: Tuple[List[int], List[int], bool, int, str, bool] = (
    [128, 128, 256],
    [1, 1, 1],
    False,
    1,  # placeholder; the launcher always resolves group_hint first
    "static",
    False,
)


# Candidate tactic geometries derived from the upstream functional test
# matrix ``moe_nvfp4_swapab/run_mega_tests.sh`` (M01..M20). Each entry is
# ``(mma_tiler_mnk, cluster_shape_mnk, use_2cta_instrs)``. Other tactic
# fields (load_balance_mode, use_bf16_redg) are held to their upstream
# defaults in v1; expanding either axis needs a new kernel-side smoke
# test (see MEGAMOE_CUTEDSL_DESIGN.md "for the first autotuner sweep").
_RUN_MEGA_TESTS_CANDIDATE_GEOMETRIES: Tuple[
    Tuple[Tuple[int, int, int], Tuple[int, int, int], bool], ...
] = (
    ((128, 128, 256), (1, 1, 1), False),
    ((256, 256, 256), (2, 1, 1), True),
    ((256, 256, 256), (4, 1, 1), True),
)

# Load-balance modes swept by the v1 autotuner. Both modes are kernel-
# supported (see ImplDesc.__post_init__ in fc1_fc2_fuse_sched.py); the
# external functional matrix exercises ``atomic_counter`` in M17-M19.
# ``clc`` is intentionally excluded -- it routes through a separate
# scheduler class not wired through the fused FC12 kernel here.
_LOAD_BALANCE_MODE_CANDIDATES: Tuple[str, ...] = ("static", "atomic_counter")

# Kernel-construction knobs locked to upstream defaults in v1.
_LOCKED_KERNEL_KWARGS = {
    "force_static_sched": True,
    "clc_bundle_size": None,
    "num_sched_stages": None,
}


def _is_pow2_in_range(val: int, lo: int, hi: int) -> bool:
    return lo <= val <= hi and (val & (val - 1)) == 0


def megamoe_activation_sf_bytes_per_row(hidden_size: int) -> int:
    """Return the per-row byte width the MegaMoE kernel expects for the
    activation SF tensor.

    The kernel reads ``ceil(hidden / (4 * sf_vec_size=64)) * 4`` FP8
    bytes per token (see ``sf_uint32_per_token`` in megamoe_kernel.py
    -- one uint32 packs 4 FP8 scales, each covering 16 NVFP4 elements,
    so each uint32 covers 64 elements). For hidden sizes that are
    multiples of 32 but not 64 (e.g. 1568, 1632, 2080) the naive
    ``hidden // 16`` byte-row width is short by 2 bytes; the kernel's
    TMA load would then either read uninitialized bytes or trip the
    last-row stride check. Always use this helper when allocating /
    sizing activation SF tensors at the backend boundary.
    """
    if hidden_size <= 0 or hidden_size % 32 != 0:
        raise ValueError(f"hidden_size must be a positive multiple of 32, got {hidden_size}")
    # ceil(hidden / 16) rounded up to multiples of 4 FP8 columns. Matches
    # `round_up(ceil(hidden_size / scaling_vector_size), 4)` documented
    # in MEGAMOE_CUTEDSL_DESIGN.md ("can_implement" hidden_size
    # alignment rule).
    sf_cols = (hidden_size + 15) // 16
    return ((sf_cols + 3) // 4) * 4


def validate_megamoe_tactic(tactic: Tuple) -> None:
    """Validate a tactic tuple against the kernel-side constraints
    documented in MEGAMOE_CUTEDSL_DESIGN.md "MegaMoECuteDsl tactic
    representation". Raises ``ValueError`` with a clear message on
    failure; the caller (``get_valid_tactics`` / ``forward``) catches
    and filters.
    """
    from ..cute_dsl_kernels.mega_moe_nvfp4 import (
        Nvfp4BlockSize,
        SupportedMmaTileM,
        SupportedMmaTileN,
    )

    if (not isinstance(tactic, tuple)) or len(tactic) != 6:
        raise ValueError(
            f"MegaMoE tactic must be a 6-tuple, got {type(tactic).__name__}={tactic!r}"
        )
    (mma_tiler, cluster_shape, use_2cta, resolved_group_hint, load_balance_mode, use_bf16_redg) = (
        tactic
    )

    if (not isinstance(mma_tiler, (list, tuple))) or len(mma_tiler) != 3:
        raise ValueError(f"mma_tiler_mnk must be a 3-tuple/list, got {mma_tiler!r}")
    if mma_tiler[0] not in SupportedMmaTileM:
        raise ValueError(
            f"mma_tiler_mnk[0]={mma_tiler[0]} not in SupportedMmaTileM={SupportedMmaTileM}"
        )
    if mma_tiler[1] not in SupportedMmaTileN:
        raise ValueError(
            f"mma_tiler_mnk[1]={mma_tiler[1]} not in SupportedMmaTileN={SupportedMmaTileN}"
        )
    if mma_tiler[2] % (Nvfp4BlockSize * 4) != 0:
        raise ValueError(
            f"mma_tiler_mnk[2]={mma_tiler[2]} must be a multiple of "
            f"{Nvfp4BlockSize * 4} (= sf_vec_size * 4); see kernel_fc12 "
            f"_validate_mma_*."
        )

    if (not isinstance(cluster_shape, (list, tuple))) or len(cluster_shape) != 3:
        raise ValueError(f"cluster_shape_mnk must be a 3-tuple/list, got {cluster_shape!r}")
    if cluster_shape[2] != 1:
        raise ValueError(f"cluster_shape_mnk[2] (L axis) must be 1, got {cluster_shape[2]}")
    if cluster_shape[1] != 1:
        raise ValueError(
            f"cluster_shape_mnk[1] (N axis) must be 1 in v1 (kernel raises "
            f"NotImplementedError for cluster_n>1); got {cluster_shape[1]}."
        )
    for axis, val in zip(("M", "N"), (cluster_shape[0], cluster_shape[1])):
        if not _is_pow2_in_range(val, 1, 4):
            raise ValueError(
                f"cluster_shape_mnk {axis}-axis must be a power of two in [1, 4], got {val}."
            )
    if cluster_shape[0] * cluster_shape[1] > 16:
        raise ValueError(
            f"cluster_shape_mnk[M]*cluster_shape_mnk[N] must be <= 16, got "
            f"{cluster_shape[0] * cluster_shape[1]}."
        )

    if not isinstance(use_2cta, bool):
        raise ValueError(f"use_2cta_instrs must be bool, got {use_2cta!r}.")
    expected_2cta = mma_tiler[0] == 256
    if use_2cta != expected_2cta:
        raise ValueError(
            f"use_2cta_instrs must be {expected_2cta} for mma_tiler_mnk[0]={mma_tiler[0]}, got {use_2cta}."
        )

    if cluster_shape[0] % (2 if use_2cta else 1) != 0:
        raise ValueError(
            f"cluster_shape_mnk[0] ({cluster_shape[0]}) must be divisible by "
            f"{(2 if use_2cta else 1)} when use_2cta_instrs={use_2cta}."
        )

    if (not isinstance(resolved_group_hint, int)) or resolved_group_hint <= 0:
        raise ValueError(
            f"resolved_group_hint must be a positive int (resolved before "
            f"cache lookup), got {resolved_group_hint!r}."
        )

    if load_balance_mode not in {"static", "atomic_counter"}:
        raise ValueError(
            f"load_balance_mode must be 'static' or 'atomic_counter', got {load_balance_mode!r}."
        )

    if not isinstance(use_bf16_redg, bool):
        raise ValueError(f"use_bf16_redg must be bool, got {use_bf16_redg!r}.")


def resolve_megamoe_group_hint(cluster_shape_mnk: Tuple[int, int, int]) -> int:
    """Resolve ``group_hint=None`` to ``HardwareInfo().get_max_active_clusters``.

    The kernel uses ``group_hint`` as a construction-time constant
    (``Sm100MegaMoEKernel.__init__``); caching under ``None`` would
    produce a wrong cache key. Falls back to 1 on hosts without
    CUDA / Cutlass DSL so the tactic remains JSON-serializable.
    """
    cluster_size = cluster_shape_mnk[0] * cluster_shape_mnk[1] * cluster_shape_mnk[2]
    if cluster_size <= 0:
        cluster_size = 1
    try:
        from cutlass.utils import HardwareInfo

        return max(1, int(HardwareInfo().get_max_active_clusters(cluster_size)))
    except Exception:  # pragma: no cover - host without CUDA / Cutlass DSL
        return 1


def enumerate_megamoe_candidate_tactics() -> List[Tuple]:
    """Return the v1 candidate tactic list, fully resolved.

    Each candidate has its ``resolved_group_hint`` stamped to the value
    returned by ``HardwareInfo.get_max_active_clusters`` for that
    cluster shape. Static load balance + form A only in v1.
    """
    candidates: List[Tuple] = []
    for mma_tiler, cluster_shape, use_2cta in _RUN_MEGA_TESTS_CANDIDATE_GEOMETRIES:
        for load_balance_mode in _LOAD_BALANCE_MODE_CANDIDATES:
            tactic = (
                list(mma_tiler),
                list(cluster_shape),
                use_2cta,
                resolve_megamoe_group_hint(cluster_shape),
                load_balance_mode,
                False,
            )
            try:
                validate_megamoe_tactic(tactic)
            except ValueError as e:
                logger.debug(f"[MegaMoE] dropping candidate tactic {tactic!r}: {e}")
                continue
            candidates.append(tactic)
    return candidates


if IS_CUTLASS_DSL_AVAILABLE:
    # Stricter than ``IS_CUTLASS_DSL_AVAILABLE``: the op needs
    # ``cutlass.torch.from_dlpack``, ``cutlass._mlir`` adapters for the
    # ``SymBufferHost`` struct, ``cute_nvgpu.tcgen05`` for the MMA
    # atoms, and the ported MegaMoE NVFP4 kernel package. A half-
    # installed or older cutlass-dsl wheel exposes ``cutlass.cute`` but
    # is missing one of the above; we catch every such failure so the
    # rest of the ``custom_ops`` package still imports and only this
    # op is unregistered.
    try:
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as torch_symm_mem
        from cutlass.cute.nvgpu import cpasync, tcgen05  # noqa: F401

        try:
            from cuda.bindings import driver as cuda
        except ImportError:
            from cuda import cuda

        # ``Nvfp4BlockSize`` is probe-only at module load to fail fast
        # when the kernel package is partially installed; it is consumed
        # by the lazy import inside ``validate_megamoe_tactic``.
        from ..cute_dsl_kernels.mega_moe_nvfp4 import (
            Nvfp4BlockSize,  # noqa: F401
            SfPaddingBlock,
        )
        from ..cute_dsl_kernels.mega_moe_nvfp4.sym_buffer import SymBufferHost

        IS_MEGAMOE_OP_AVAILABLE = True
    except ImportError as _megamoe_import_err:  # pragma: no cover - env-specific
        logger.info(
            "MegaMoE CuteDSL op skipped: %s. Backend ``MegaMoECuteDsl`` "
            "stays uninstalled; ``torch.ops.trtllm."
            "cute_dsl_megamoe_nvfp4_blackwell`` is not registered. The "
            "factory falls back to CutlassFusedMoE.",
            _megamoe_import_err,
        )
        IS_MEGAMOE_OP_AVAILABLE = False


if IS_MEGAMOE_OP_AVAILABLE:
    # ----- Local workspace cache --------------------------------------------
    #
    # ``local_workspace`` is per-rank CUDA-only and sized by
    # ``kernel.get_workspace_sizes()``. It is shape-stable across forward
    # calls (size only depends on kernel construction kwargs), so we cache
    # it per static shape so multiple MoE layers / chunks amortize the
    # allocation. ``shared_workspace`` lives in the symmetric heap for
    # multi-rank and is supplied by the caller (the MegaMoECuteDsl backend's
    # MegaMoeSymmMemProvider carves it out of the rendezvous'd buffer).
    _MEGAMOE_LOCAL_WORKSPACE_CACHE: dict = {}

    def _get_or_alloc_local_workspace(
        kernel, cache_key: Tuple, device: torch.device
    ) -> torch.Tensor:
        cached = _MEGAMOE_LOCAL_WORKSPACE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        local_bytes, _ = kernel.get_workspace_sizes()
        # Local-workspace regions include Int32 atomic counters
        # (l1_arrival_count, fc1_done_counter, grid_sync_counter)
        # that the kernel's spin_wait expects to start at 0.
        # torch.empty leaves them as uninitialised bytes -- a single
        # negative Int32 in any counter slot makes spin_wait
        # (v >= positive_threshold) impossible to satisfy and the
        # kernel hangs at 100% SM / 0% memory-bandwidth.
        local_workspace = torch.zeros(local_bytes, dtype=torch.uint8, device=device)
        _MEGAMOE_LOCAL_WORKSPACE_CACHE[cache_key] = local_workspace
        return local_workspace

    # ----- Symmetric-memory provider (NVSHMEM-equivalent) -------------------
    #
    # PyTorch's ``torch.distributed._symmetric_memory`` is an NVSHMEM-equivalent
    # symmetric-heap provider built on cuMem APIs. It exposes per-rank buffer
    # pointers (``handle.buffer_ptrs``) which we use to populate the
    # ``SymBufferHost(base_addr, offsets, rank_idx, num_max_ranks)`` payload
    # the MegaMoE kernel expects.
    #
    # We allocate ONE large symmetric buffer per (group, layout_key) and
    # carve out fixed-size regions for activation / activation_sf /
    # topk_weights / combine_output / shared_workspace. All ranks have
    # identical region offsets inside the buffer, so peer_offsets =
    # [handle.buffer_ptrs[r] - local_base for r in range(world_size)]
    # correctly maps any region's local pointer to its peer counterpart.
    #
    # Cache lives at module scope so multiple MoE layers with the same
    # (group, layout) share the same allocation (mirrors
    # ``_MEGA_MOE_SYMM_BUFFER_CACHE`` in ``mega_moe_deepgemm.py``).
    _MEGAMOE_SYMM_PROVIDER_CACHE: dict = {}

    def _round_up_to(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    @dataclasses.dataclass
    class MegaMoeSymmRegions:
        """User-domain symmetric tensors carved out of a single rendezvous'd
        buffer. All views share the same underlying allocation, so they
        share ``peer_offsets``.

        ``base_buf`` is kept alive for the lifetime of the provider; views
        only stay valid while the provider is in the cache.
        """

        base_buf: torch.Tensor
        activation: torch.Tensor  # (max_T, hidden // 2) uint8
        activation_sf: torch.Tensor  # (max_T, hidden // 16) uint8
        topk_weights: torch.Tensor  # (max_T, num_topk) float32
        combine_output: torch.Tensor  # (max_T, num_topk, hidden) bf16
        shared_workspace: torch.Tensor  # (shared_ws_bytes,) uint8
        peer_offsets: List[int]  # symmetric peer-pointer deltas
        rank: int
        world_size: int

    class MegaMoeSymmMemProvider:
        """Allocate + carve symmetric-memory regions for MegaMoE multi-rank
        execution.

        The provider is bound to a ProcessGroup (the EP sub-world the
        kernel exchanges over) and a layout key (hidden, num_topk,
        max_tokens_per_rank, output_dtype, shared_workspace_bytes). It
        survives across MoE layers with identical layouts so the
        expensive ``torch_symm_mem.rendezvous`` collective only runs
        once per build.

        The MegaMoECuteDsl backend reaches the provider through
        :func:`get_megamoe_symm_provider`; the provider is created
        lazily on the first ``run_moe`` call that needs multi-rank,
        which still runs synchronously across all EP ranks because
        ``rendezvous`` itself is a collective.
        """

        # Alignment in bytes between region boundaries. 128 B keeps each
        # region aligned to the TMA load requirement; matches the
        # alignment used for blocked_scale / FP8 SF inside the kernel.
        _REGION_ALIGN = 128

        def __init__(
            self,
            *,
            process_group,
            world_size: int,
            rank: int,
            hidden_size: int,
            max_tokens_per_rank: int,
            num_topk: int,
            output_dtype: torch.dtype,
            shared_workspace_bytes: int,
        ) -> None:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("MegaMoeSymmMemProvider requires torch.distributed initialized.")
            if process_group is None:
                raise ValueError(
                    "MegaMoeSymmMemProvider requires a non-None process_group "
                    "(MegaMoECuteDsl resolves this from mapping.moe_ep_group_pg)."
                )
            if not hasattr(process_group, "group_name"):
                raise RuntimeError(
                    "MegaMoeSymmMemProvider requires a torch.distributed "
                    "ProcessGroup with a stable group_name. Use Ray / DeviceMesh "
                    "or pass a group created with dist.new_group(...)."
                )

            self.process_group = process_group
            self.group_name = str(process_group.group_name)
            self.world_size = int(world_size)
            self.rank = int(rank)
            self.hidden_size = int(hidden_size)
            self.max_tokens_per_rank = int(max_tokens_per_rank)
            self.num_topk = int(num_topk)
            self.output_dtype = output_dtype

            # Region byte sizes (worst case across launches; staging
            # writes only the live ``T`` rows). NVFP4 packs 2 elems / byte
            # along K so activation rows are hidden // 2 bytes. The SF
            # row width matches the kernel's TMA load expectation
            # ``round_up(ceil(hidden / 16), 4)`` -- naive ``hidden // 16``
            # under-allocates by 2 bytes when ``hidden % 64 != 0`` (e.g.
            # 1568, 1632, 2080).
            act_bytes_per_row = hidden_size // 2
            sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden_size)
            topkw_bytes_per_row = num_topk * 4  # float32
            combine_bytes_per_row = num_topk * hidden_size * output_dtype.itemsize

            act_region = _round_up_to(max_tokens_per_rank * act_bytes_per_row, self._REGION_ALIGN)
            sf_region = _round_up_to(max_tokens_per_rank * sf_bytes_per_row, self._REGION_ALIGN)
            topkw_region = _round_up_to(
                max_tokens_per_rank * topkw_bytes_per_row, self._REGION_ALIGN
            )
            combine_region = _round_up_to(
                max_tokens_per_rank * combine_bytes_per_row, self._REGION_ALIGN
            )
            shared_region = _round_up_to(shared_workspace_bytes, self._REGION_ALIGN)

            self._region_offsets: dict = {}
            self._region_sizes: dict = {}
            cursor = 0
            for name, region in (
                ("activation", act_region),
                ("activation_sf", sf_region),
                ("topk_weights", topkw_region),
                ("combine_output", combine_region),
                ("shared_workspace", shared_region),
            ):
                self._region_offsets[name] = cursor
                self._region_sizes[name] = region
                cursor += region
            total_bytes = cursor

            # Enable symm mem on the group exactly once (idempotent).
            torch_symm_mem.enable_symm_mem_for_group(self.group_name)

            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._buf = torch_symm_mem.empty(total_bytes, device=device, dtype=torch.uint8)
            # Zero the entire symmetric buffer once at construction.
            # ``torch_symm_mem.empty`` returns uninitialised CUmem; the
            # MegaMoE kernel's dispatch path TMA-pulls peer activation
            # / SF / topk_weights rows by ``peer_rank_ptr_mapper.map``,
            # and the kernel design assumes the trailing per-row tail
            # (the K_sf elements paired with NVFP4 data that fall in
            # the TMA OOB-fill-0 region) is zero. Upstream mirrors this
            # via ``_sym_zeros_byte_view`` -- staging tensors that
            # always start zero so per-call only the live ``T`` rows
            # are written and the padded tail stays 0. Without this,
            # peers read uninitialised cuMem bytes and the run is
            # silently non-deterministic.
            self._buf.zero_()
            # ``rendezvous`` is a collective; every rank in the group must
            # call it in lockstep. The MegaMoECuteDsl backend resolves the
            # EP PG at construction time and lazily creates the provider
            # only when run_moe needs it -- which is consistent across
            # ranks because the wrapper calls run_moe at every layer.
            self._handle = torch_symm_mem.rendezvous(self._buf, self.group_name)
            local_base = int(self._buf.data_ptr())
            self.peer_offsets: List[int] = []
            for r in range(self.world_size):
                peer_ptr = int(self._handle.buffer_ptrs[r])
                self.peer_offsets.append(peer_ptr - local_base)

            logger.info(
                "[MegaMoeSymmMemProvider] group=%s rank=%d/%d total_bytes=%d "
                "(activation=%d sf=%d topk_weights=%d combine=%d shared=%d)",
                self.group_name,
                self.rank,
                self.world_size,
                total_bytes,
                act_region,
                sf_region,
                topkw_region,
                combine_region,
                shared_region,
            )

        def _region_view(
            self, name: str, shape: Tuple[int, ...], dtype: torch.dtype
        ) -> torch.Tensor:
            offset = self._region_offsets[name]
            numel = 1
            for d in shape:
                numel *= d
            byte_len = numel * dtype.itemsize
            byte_view = self._buf[offset : offset + byte_len]
            return byte_view.view(dtype).view(shape)

        def get_regions(self) -> MegaMoeSymmRegions:
            hidden = self.hidden_size
            max_t = self.max_tokens_per_rank
            top_k = self.num_topk
            return MegaMoeSymmRegions(
                base_buf=self._buf,
                activation=self._region_view("activation", (max_t, hidden // 2), torch.uint8),
                activation_sf=self._region_view(
                    "activation_sf", (max_t, hidden // 16), torch.uint8
                ),
                topk_weights=self._region_view("topk_weights", (max_t, top_k), torch.float32),
                combine_output=self._region_view(
                    "combine_output", (max_t, top_k, hidden), self.output_dtype
                ),
                shared_workspace=self._region_view(
                    "shared_workspace", (self._region_sizes["shared_workspace"],), torch.uint8
                ),
                peer_offsets=list(self.peer_offsets),
                rank=self.rank,
                world_size=self.world_size,
            )

    def get_megamoe_symm_provider(
        *,
        process_group,
        world_size: int,
        rank: int,
        hidden_size: int,
        max_tokens_per_rank: int,
        num_topk: int,
        output_dtype: torch.dtype,
        shared_workspace_bytes: int,
    ) -> MegaMoeSymmMemProvider:
        """Return a cached provider for (group, layout). The cache is
        keyed on the group ``group_name`` plus every layout knob that
        affects allocation size so two MoE layers with the same shape
        share the same symmetric buffer.

        First call from each rank performs the (collective)
        ``torch_symm_mem.rendezvous``; subsequent calls return the
        cached provider with no further collectives.
        """
        if not hasattr(process_group, "group_name"):
            raise RuntimeError(
                "process_group must expose .group_name (ProcessGroup created "
                "via Ray DeviceMesh or dist.new_group). Use ConfigurableMoE "
                "with mapping.moe_ep_group_pg."
            )
        cache_key = (
            str(process_group.group_name),
            int(hidden_size),
            int(max_tokens_per_rank),
            int(num_topk),
            str(output_dtype),
            int(shared_workspace_bytes),
        )
        cached = _MEGAMOE_SYMM_PROVIDER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        provider = MegaMoeSymmMemProvider(
            process_group=process_group,
            world_size=world_size,
            rank=rank,
            hidden_size=hidden_size,
            max_tokens_per_rank=max_tokens_per_rank,
            num_topk=num_topk,
            output_dtype=output_dtype,
            shared_workspace_bytes=shared_workspace_bytes,
        )
        _MEGAMOE_SYMM_PROVIDER_CACHE[cache_key] = provider
        return provider

    def query_megamoe_shared_workspace_bytes(
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        num_experts_per_rank: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        expand_intermediate_size_per_partition: int,
        max_tokens_per_rank: int,
        tactic: Optional[Tuple] = None,
    ) -> int:
        """Probe ``Sm100MegaMoEKernel.get_workspace_sizes()`` for the
        shared workspace byte count. The shared workspace size is
        invariant across all candidate tactics (its regions depend only
        on world_size / num_experts_per_rank / num_topk /
        max_tokens_per_rank -- see _build_shared_region_specs in
        megamoe_kernel.py), so we use the default tactic for the probe.
        """
        from ..cute_dsl_kernels.mega_moe_nvfp4 import import_kernel

        if tactic is None:
            cluster = tuple(DEFAULT_MEGAMOE_TACTIC[1])
            tactic = (
                list(DEFAULT_MEGAMOE_TACTIC[0]),
                list(cluster),
                DEFAULT_MEGAMOE_TACTIC[2],
                resolve_megamoe_group_hint(cluster),
                DEFAULT_MEGAMOE_TACTIC[4],
                DEFAULT_MEGAMOE_TACTIC[5],
            )
        kernel_cls = import_kernel()
        probe = kernel_cls(
            mma_tiler_mnk=tuple(tactic[0]),
            cluster_shape_mnk=tuple(tactic[1]),
            use_2cta_instrs=bool(tactic[2]),
            group_hint=int(tactic[3]),
            token_padding_block=64,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=str(tactic[4]),
            static_expert_shape=(
                num_experts_per_rank,
                expand_intermediate_size_per_partition,
                hidden_size,
            ),
            world_size=int(world_size),
            local_rank=int(local_rank),
            num_topk=int(num_topk),
            max_tokens_per_rank=int(max_tokens_per_rank),
            hidden=int(hidden_size),
            fc2_in_kernel_topk_reduce=bool(tactic[5]),
            **_LOCKED_KERNEL_KWARGS,
        )
        _, shared_bytes = probe.get_workspace_sizes()
        return int(shared_bytes)

    def _to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> "cute.Tensor":
        cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        leading_dim = cutlass_torch.get_leading_dim(tensor)
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    class Sm100MegaMoENvfp4Runner(TunableRunner):
        """TunableRunner for the ported MegaMoE CuteDSL NVFP4 kernel.

        Owns a process-global ``kernel_cache`` keyed on the full
        ``(static_shape + tactic)`` tuple so multiple MoE layers with
        identical shapes amortize the (expensive) ``cute.compile``.
        ``get_valid_tactics`` enumerates the upstream-tested geometries
        and validates each against the kernel-side constraints.
        """

        # Module-scope compile cache shared by every runner instance.
        kernel_cache: dict = {}

        def __init__(
            self,
            *,
            world_size: int,
            local_rank: int,
            num_topk: int,
            num_experts_per_rank: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            expand_intermediate_size_per_partition: int,
            max_tokens_per_rank: int,
            output_dtype: torch.dtype,
        ) -> None:
            super().__init__()
            if (sm_version := get_sm_version()) not in (100, 103):
                raise ValueError(
                    f"Sm100MegaMoENvfp4Runner requires SM 100 (B200) or SM 103 "
                    f"(B300); got SM {sm_version}."
                )
            if num_experts_per_rank <= 0:
                raise ValueError(
                    f"num_experts_per_rank must be positive, got {num_experts_per_rank}"
                )
            if max_tokens_per_rank <= 0:
                raise ValueError(f"max_tokens_per_rank must be positive, got {max_tokens_per_rank}")
            if output_dtype != torch.bfloat16:
                raise ValueError(
                    f"Sm100MegaMoENvfp4Runner only supports bfloat16 output; got {output_dtype}"
                )
            self.world_size = int(world_size)
            self.local_rank = int(local_rank)
            self.num_topk = int(num_topk)
            self.num_experts_per_rank = int(num_experts_per_rank)
            self.hidden_size = int(hidden_size)
            self.intermediate_size_per_partition = int(intermediate_size_per_partition)
            self.expand_intermediate_size_per_partition = int(
                expand_intermediate_size_per_partition
            )
            self.max_tokens_per_rank = int(max_tokens_per_rank)
            self.output_dtype = output_dtype

        def unique_id(self):
            return (
                self.world_size,
                self.local_rank,
                self.num_topk,
                self.num_experts_per_rank,
                self.hidden_size,
                self.intermediate_size_per_partition,
                self.expand_intermediate_size_per_partition,
                self.max_tokens_per_rank,
                str(self.output_dtype),
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple]:
            del inputs, profile, kwargs
            return enumerate_megamoe_candidate_tactics()

        def _autotuner_inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            """Sanitize autotuner-generated fake inputs for the MegaMoE kernel.

            The MegaMoE dispatch path uses ``topk_idx`` to index a per-CTA
            SMEM histogram of size ``num_experts_per_rank * world_size`` and
            to compute ``dst_rank = expert_id // num_experts_per_rank`` /
            ``local_expert = expert_id % num_experts_per_rank`` (see
            ``dispatch_kernel.py`` lines 262-265 and 340-364). Random
            ``int64`` values produced by ``AutoTuner._create_tensor_like``
            (range ``[-5, 4]``) work only for the smallest fixtures and
            blow up on larger ``num_experts`` because:
              * Out-of-range ``expert_id >= num_experts_per_rank*world_size``
                makes ``smem_count_ptr + expert_id`` overflow the SMEM
                histogram, triggering ``illegal memory access``.
              * ``dst_rank`` decoded from a garbage expert_id can index
                the peer-rank pointer table out of bounds.

            Overwriting ``topk_idx`` with a deterministic round-robin
            assignment in ``[0, total_experts)`` keeps the dispatch
            arithmetic in-bounds for every tactic the autotuner profiles,
            without affecting the chosen tactic (autotuning measures
            runtime, not numerics).

            Also zero ``topk_weights`` / SF buffers so the FP32/FP8 paths
            avoid NaN propagation that some Tensor Core paths reject.
            ``combine_output`` is the output buffer (kernel writes); leave
            it alone so the kernel allocator sees the runtime layout.
            """
            inputs = list(inputs)
            total_experts = self.num_experts_per_rank * self.world_size
            if total_experts <= 0:
                return inputs

            topk_idx = inputs[2]
            if isinstance(topk_idx, torch.Tensor) and topk_idx.dim() == 2:
                T, K = topk_idx.shape
                valid = (
                    torch.arange(
                        T * K,
                        dtype=topk_idx.dtype,
                        device=topk_idx.device,
                    )
                    % total_experts
                ).view(T, K)
                topk_idx.copy_(valid)

            # Zero the FP8 / FP32 scaling buffers so the FC1/FC2 epilogues
            # don't trip on NaN-encoded byte patterns picked up from random
            # ``uint8`` -> FP8 reinterpretation.
            for sf_idx in (1, 5, 7):  # activation_sf, fc1_weight_sf, fc2_weight_sf
                tensor = inputs[sf_idx]
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
            topk_weights = inputs[3]
            if isinstance(topk_weights, torch.Tensor):
                topk_weights.zero_()

            return inputs

        def get_tuning_config(self) -> TuningConfig:
            """Tuning config: only the activation token-axis is dynamic.

            Constraints chain activation_sf / topk_idx / topk_weights /
            combine_output to the activation token count, so the
            autotuner does not double-enumerate tile sizes for
            independent token axes.
            """

            # Constraints reuse the runner's own shape-derivation rules
            # (the activation token count drives every other tensor's
            # leading axis). We pass shape-derivation lambdas that pull
            # the runtime ``num_tokens`` from input[0].
            def _num_tokens(shapes: List[torch.Size]) -> int:
                return shapes[0][0]

            return TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        0, 0, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
                    ),
                ),
                constraint_specs=(
                    ConstraintSpec(1, 0, _num_tokens),  # activation_sf
                    ConstraintSpec(2, 0, _num_tokens),  # topk_idx
                    ConstraintSpec(3, 0, _num_tokens),  # topk_weights
                    ConstraintSpec(8, 0, _num_tokens),  # combine_output
                ),
                inputs_pre_hook=self._autotuner_inputs_pre_hook,
                use_cold_l2_cache=True,
                # MegaMoE's symm-memory dispatch barrier and per-CTA SMEM
                # histogram updates issue host-visible state (peer pointer
                # mapper, ``expert_recv_count_sum`` atomic counters, etc.)
                # that CUDA Graph capture cannot reproduce: a captured
                # graph replays a frozen kernel sequence but cannot
                # re-evaluate the runtime peer-pointer table or the
                # dispatch-counter view recomputed each chunk. Profiling
                # via ``cuda.graph(capture); graph.replay()`` therefore
                # spins inside the captured dispatch barrier when the
                # autotuner's L2-cache circular buffers swap behind a
                # tactic that captured the original buffer pointers,
                # causing the autotuner to appear to hang at 100 %% GPU
                # util without progress. Plain ``for _ in range(repeat):
                # kernel()`` profiling is correct for MegaMoE and only
                # marginally slower (the host-side prep is the same
                # constant cost per launch).
                use_cuda_graph=False,
                # FUSED_COMM hard requirement: every EP rank must launch
                # the same compiled kernel of the same tactic for every
                # chunk so the in-kernel NVLink dispatch barrier and
                # peer pointer mapping line up. Without
                # ``DistributedTuningStrategy.PARALLEL`` the AutoTuner
                # benchmarks each tactic locally on each rank and may
                # pick different winners on different ranks -- the
                # dispatch barrier then deadlocks (or worse, peers read
                # the wrong symm-memory region for a tactic-dependent
                # workspace layout).
                #
                # See the same flag on every multi-rank CuteDSL op in
                # ``cute_dsl_custom_ops.py`` (NVFP4 GEMM, swiglu GEMM,
                # FP4-out GEMM, dense GEMM swiglu). MegaMoE is strictly
                # more sensitive because every chunk has a barrier;
                # NVFP4 dense ops only need lockstep at allreduce
                # boundaries.
                distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
            )

        def _build_kernel(self, tactic: Tuple):
            (
                mma_tiler,
                cluster_shape,
                use_2cta,
                resolved_group_hint,
                load_balance_mode,
                use_bf16_redg,
            ) = tactic
            from ..cute_dsl_kernels.mega_moe_nvfp4 import import_kernel

            kernel_cls = import_kernel()
            return kernel_cls(
                mma_tiler_mnk=tuple(mma_tiler),
                cluster_shape_mnk=tuple(cluster_shape),
                use_2cta_instrs=bool(use_2cta),
                group_hint=int(resolved_group_hint),
                token_padding_block=64,
                sf_padding_block=SfPaddingBlock,
                load_balance_mode=str(load_balance_mode),
                static_expert_shape=(
                    self.num_experts_per_rank,
                    self.expand_intermediate_size_per_partition,
                    self.hidden_size,
                ),
                world_size=self.world_size,
                local_rank=self.local_rank,
                num_topk=self.num_topk,
                max_tokens_per_rank=self.max_tokens_per_rank,
                hidden=self.hidden_size,
                fc2_in_kernel_topk_reduce=bool(use_bf16_redg),
                **_LOCKED_KERNEL_KWARGS,
            )

        def _compile_or_get(self, tactic: Tuple, kernel, runtime_kwargs):
            cache_key = (
                self.unique_id(),
                tuple(tactic[0]),
                tuple(tactic[1]),
                bool(tactic[2]),
                int(tactic[3]),
                str(tactic[4]),
                bool(tactic[5]),
            )
            compiled = self.__class__.kernel_cache.get(cache_key)
            if compiled is None:
                compile_kwargs = dict(runtime_kwargs)
                hardware_info = cutlass.utils.HardwareInfo()
                cluster_size = tactic[1][0] * tactic[1][1] * tactic[1][2]
                compile_kwargs["max_active_clusters"] = hardware_info.get_max_active_clusters(
                    max(cluster_size, 1)
                )
                import sys as _sys

                t_compile_start = time.perf_counter()
                print(
                    f"[MegaMoECuteDsl] cute.compile START tactic="
                    f"(mma_tiler={tactic[0]}, cluster={tactic[1]}, "
                    f"use_2cta={tactic[2]}, group_hint={tactic[3]}, "
                    f"load_balance={tactic[4]!r}, use_bf16_redg={tactic[5]})",
                    file=_sys.stderr,
                    flush=True,
                )
                compiled = cute.compile(kernel, **compile_kwargs)
                t_compile_ms = (time.perf_counter() - t_compile_start) * 1000
                print(
                    f"[MegaMoECuteDsl] cute.compile DONE in {t_compile_ms:.0f} ms "
                    f"(cache_keys_now={len(self.__class__.kernel_cache) + 1})",
                    file=_sys.stderr,
                    flush=True,
                )
                self.__class__.kernel_cache[cache_key] = compiled
            return compiled

        def forward(
            self,
            inputs: List[torch.Tensor],
            *,
            tactic: Any = -1,
            do_preparation: bool = False,
            peer_offsets: Optional[List[int]] = None,
            shared_workspace: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> None:
            del kwargs
            # AutoTuner calls forward with ``do_preparation=True`` once
            # per profile to let the runner pre-allocate warm-up buffers
            # **before** the timed tactic loop. The MegaMoE kernel does
            # not need any preallocation here (its symm-memory provider
            # workspace is owned by the caller and the per-tactic CuTe
            # JIT cache lives on the runner class). Running the full
            # kernel here would launch outside the tactic loop's
            # try/except, so any crash on autotuner-generated fake data
            # (e.g. an unsupported tactic+shape combo) propagates as an
            # async CUDA error to the *next* CUDA call and looks like a
            # ``torch.rand`` failure inside ``_create_tensor_like``.
            # Early-return mirrors the ``AllReduceRunner`` pattern in
            # ``torch_custom_ops.py`` (lines 1924-1945).
            if do_preparation:
                return None
            t_forward_start = time.perf_counter()
            # Resolve fallback tactic.
            if tactic == -1 or tactic is None:
                tactic_t = (
                    list(DEFAULT_MEGAMOE_TACTIC[0]),
                    list(DEFAULT_MEGAMOE_TACTIC[1]),
                    DEFAULT_MEGAMOE_TACTIC[2],
                    resolve_megamoe_group_hint(tuple(DEFAULT_MEGAMOE_TACTIC[1])),
                    DEFAULT_MEGAMOE_TACTIC[4],
                    DEFAULT_MEGAMOE_TACTIC[5],
                )
            elif isinstance(tactic, list):
                tactic_t = tuple(tactic)
            else:
                tactic_t = tactic
            validate_megamoe_tactic(tactic_t)

            (
                activation,
                activation_sf,
                topk_idx,
                topk_weights,
                fc1_weight,
                fc1_weight_sf,
                fc2_weight,
                fc2_weight_sf,
                combine_output,
            ) = inputs[:9]
            assert peer_offsets is not None, (
                "Sm100MegaMoENvfp4Runner.forward requires peer_offsets kwarg "
                "(length = world_size); single-rank degenerate mode passes "
                "(0,) * world_size."
            )
            assert len(peer_offsets) == self.world_size, (
                f"peer_offsets length {len(peer_offsets)} != world_size {self.world_size}"
            )

            kernel = self._build_kernel(tactic_t)

            # ``local_workspace`` is per-rank private; cached across calls.
            local_workspace = _get_or_alloc_local_workspace(
                kernel,
                cache_key=(
                    self.unique_id(),
                    tuple(tactic_t[0]),
                    tuple(tactic_t[1]),
                    bool(tactic_t[2]),
                    int(tactic_t[3]),
                    str(tactic_t[4]),
                    bool(tactic_t[5]),
                ),
                device=activation.device,
            )
            # ``shared_workspace`` is peer-mapped (symmetric heap) for
            # multi-rank or local CUDA for the single-rank degenerate
            # path. The MegaMoECuteDsl backend supplies it; for the rare
            # call-site that omits it (legacy / tests) we fall back to a
            # local CUDA tensor sized by the kernel.
            if shared_workspace is None:
                if self.world_size > 1:
                    raise RuntimeError(
                        f"Sm100MegaMoENvfp4Runner: multi-rank "
                        f"(world_size={self.world_size}) requires the caller "
                        f"to supply a symmetric-memory shared_workspace via "
                        f"MegaMoeSymmMemProvider; got None."
                    )
                _, shared_bytes = kernel.get_workspace_sizes()
                shared_workspace = torch.empty(
                    shared_bytes, dtype=torch.uint8, device=activation.device
                )
            # The kernel relies on ``shared_workspace`` counters / signals
            # starting at zero. Backend zeros once per chunk boundary;
            # the runner enforces it unconditionally so a stale cached
            # buffer can never feed garbage into the dispatch primitives
            # (cheap memset vs hour-debug failure mode).
            shared_workspace.zero_()
            # Same rationale as the local_workspace init in
            # _get_or_alloc_local_workspace: the cached buffer is
            # reused across forward calls, so every call must restart
            # the atomic counters from 0 -- otherwise a tactic's
            # second forward sees a saturated fc1_done_counter and
            # the fc2 spin can complete before fc1 has actually
            # produced this round's outputs (silent correctness bug).
            local_workspace.zero_()

            activation_cute = _to_cute(activation)
            activation_sf_cute = _to_cute(activation_sf)
            topk_idx_cute = _to_cute(topk_idx)
            topk_weights_cute = _to_cute(topk_weights)
            fc1_weight_cute = _to_cute(fc1_weight)
            fc1_weight_sf_cute = _to_cute(fc1_weight_sf)
            fc2_weight_cute = _to_cute(fc2_weight)
            fc2_weight_sf_cute = _to_cute(fc2_weight_sf)
            combine_output_cute = _to_cute(combine_output)
            local_workspace_cute = _to_cute(local_workspace)
            shared_workspace_cute = _to_cute(shared_workspace)

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            # SymBufferHost contract: ``base_addr`` is a local pointer
            # inside the symmetric heap (we use the activation's
            # data_ptr); ``offsets[r] = peer_base - local_base`` for each
            # rank r. Since activation / activation_sf / topk_weights /
            # combine_output / shared_workspace are all carved out of the
            # SAME symmetric buffer (by MegaMoeSymmMemProvider), they
            # share the same delta -- the kernel's
            # ``peer_rank_ptr_mapper.map(local_ptr, peer, off)`` correctly
            # maps any region's local pointer to its peer counterpart.
            sym_buf = SymBufferHost(
                base_addr=int(activation.data_ptr()),
                offsets=tuple(int(off) for off in peer_offsets),
                rank_idx=int(self.local_rank),
                num_max_ranks=int(self.world_size),
            )

            runtime_kwargs = dict(
                activation=activation_cute,
                activation_sf=activation_sf_cute,
                topk_idx=topk_idx_cute,
                topk_weights=topk_weights_cute,
                fc1_weight=fc1_weight_cute,
                fc1_weight_sf=fc1_weight_sf_cute,
                fc2_weight=fc2_weight_cute,
                fc2_weight_sf=fc2_weight_sf_cute,
                combine_output=combine_output_cute,
                local_workspace=local_workspace_cute,
                shared_workspace=shared_workspace_cute,
                peer_rank_ptr_mapper_host=sym_buf,
                stream=stream,
            )
            compiled = self._compile_or_get(tactic_t, kernel, runtime_kwargs)
            t_launch_start = time.perf_counter()
            compiled(**runtime_kwargs)
            # Force a host-side sync so the kernel-runtime is excluded
            # from the next tactic's compile-time logging slot; this is
            # the cheap diagnostic complement to the compile/total log
            # above. ``torch.cuda.synchronize`` is safe because no
            # CUDA Graph capture is in flight when ``forward`` is the
            # outermost runner entry point (the autotuner profile loop
            # owns capture, not us).
            torch.cuda.synchronize()
            t_launch_ms = (time.perf_counter() - t_launch_start) * 1000
            t_forward_ms = (time.perf_counter() - t_forward_start) * 1000
            import sys as _sys

            print(
                f"[MegaMoECuteDsl] forward DONE tactic="
                f"(mma_tiler={tactic_t[0]}, cluster={tactic_t[1]}, "
                f"load_balance={tactic_t[4]!r}) "
                f"launch+sync={t_launch_ms:.0f}ms total={t_forward_ms:.0f}ms",
                file=_sys.stderr,
                flush=True,
            )
            return combine_output

    # ----- torch op ---------------------------------------------------------

    @torch.library.custom_op(
        "trtllm::cute_dsl_megamoe_nvfp4_blackwell",
        mutates_args=("combine_output", "shared_workspace"),
        device_types="cuda",
    )
    def cute_dsl_megamoe_nvfp4_blackwell(
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
        combine_output: torch.Tensor,
        shared_workspace: torch.Tensor,
        world_size: int,
        local_rank: int,
        num_topk: int,
        num_experts_per_rank: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        expand_intermediate_size_per_partition: int,
        max_tokens_per_rank: int,
        peer_offsets: List[int],
    ) -> None:
        """Run the fused MegaMoE CuteDSL NVFP4 kernel.

        Inputs are pre-staged by the caller (the ``MegaMoECuteDsl``
        backend in ``mega_moe_cute_dsl.py``). The op runs AutoTuner once
        per call to pick the best tactic for the current shape and
        invokes the runner.

        ``shared_workspace`` MUST be a symmetric-heap tensor for
        ``world_size > 1`` (use :class:`MegaMoeSymmMemProvider`); a
        local CUDA tensor is acceptable for the single-rank degenerate
        path. ``combine_output`` is mutated in place; the op does not
        return it because torch custom_op forbids the return value from
        aliasing any mutated input. Form A semantics: ``combine_output``
        keeps its ``(T, num_topk, hidden)`` layout, and the caller is
        responsible for the host-side ``.sum(dim=1)``.
        """
        sm_version = get_sm_version()
        if sm_version not in (100, 103):
            raise RuntimeError(
                f"cute_dsl_megamoe_nvfp4_blackwell requires SM 100 (B200) or "
                f"SM 103 (B300); got SM {sm_version}."
            )

        runner = Sm100MegaMoENvfp4Runner(
            world_size=world_size,
            local_rank=local_rank,
            num_topk=num_topk,
            num_experts_per_rank=num_experts_per_rank,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            expand_intermediate_size_per_partition=expand_intermediate_size_per_partition,
            max_tokens_per_rank=max_tokens_per_rank,
            output_dtype=combine_output.dtype,
        )
        inputs = [
            activation,
            activation_sf,
            topk_idx,
            topk_weights,
            fc1_weight,
            fc1_weight_sf,
            fc2_weight,
            fc2_weight_sf,
            combine_output,
        ]
        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            "trtllm::cute_dsl_megamoe_nvfp4_blackwell",
            [runner],
            runner.get_tuning_config(),
            inputs,
            peer_offsets=peer_offsets,
            shared_workspace=shared_workspace,
        )
        runner(
            inputs,
            tactic=best_tactic,
            peer_offsets=peer_offsets,
            shared_workspace=shared_workspace,
        )

    @torch.library.register_fake("trtllm::cute_dsl_megamoe_nvfp4_blackwell")
    def _(
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
        combine_output: torch.Tensor,
        shared_workspace: torch.Tensor,
        world_size: int,
        local_rank: int,
        num_topk: int,
        num_experts_per_rank: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        expand_intermediate_size_per_partition: int,
        max_tokens_per_rank: int,
        peer_offsets: List[int],
    ) -> None:
        return None
