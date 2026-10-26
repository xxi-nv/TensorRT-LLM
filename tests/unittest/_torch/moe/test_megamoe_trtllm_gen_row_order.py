# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE CuteDSL (Blackwell) reading NVFP4 expert weights in TRTLLM-Gen's row order.

``weight_row_order="trtllm_gen"`` lets the Blackwell MegaMoE epilogue consume FC1 / FC2 weights
and block scales whose rows follow TRTLLM-Gen's layout: the gated-activation row interleave, then
the 32-row shuffle selected by ``epilogue_tile_m=128``. Each epilogue warp owns a 32-row TMEM
window, so the epilogue adapts in registers instead of repacking the weights:

* FC1: one xor-1 lane exchange turns the window's up / gate row groups into a (gate, up) pair per
  lane, and a register rename restores natural channel order before quantization.
* FC2: a register rename after the MOVM transpose, or a different SMEM column on the BF16 UBLK
  path, maps each TMEM row to the hidden channel it carries.

The CPU tests replay those register mappings on the real TRTLLM-Gen permutation from
``fp4_utils``. The GPU tests run the kernel on both layouts of the same weights and require
bit-identical output; a third launch (native kernel, TRTLLM-Gen weights) must differ, which
shows the comparison can fail.
"""

import functools
import importlib
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.quantization.utils.fp4_utils import (
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices,
)

# TRTLLM-Gen's NVFP4 MoE epilogue tile; a multiple of 128 selects the 32-row shuffle block.
EPILOGUE_TILE_M = 128
# TMEM lanes owned by one MegaMoE epilogue warp: 16 FC1 channels (gate and up) or 32 FC2 rows.
WINDOW_ROWS = 32
FC1_WINDOW_CHANNELS = WINDOW_ROWS // 2

# Register mappings the epilogue applies under "trtllm_gen" (closed forms of the spec). The
# simulation below proves them against fp4_utils; test_kernel_uses_the_simulated_mappings checks
# that the kernel module uses the same ones.
FC1_CHANNEL_OF_SLOT = tuple(2 * (slot % 8) + slot // 8 for slot in range(FC1_WINDOW_CHANNELS))
FC1_SLOT_OF_CHANNEL = tuple(8 * (channel % 2) + channel // 2 for channel in range(16))
FC2_HIDDEN_OF_ROW = tuple(4 * (row % 8) + row // 8 for row in range(WINDOW_ROWS))
FC2_ROW_OF_HIDDEN = tuple(8 * (hidden % 4) + hidden // 4 for hidden in range(WINDOW_ROWS))

Label = Tuple[str, int]  # ("gate" | "up", channel)


# ---------------------------------------------------------------------------
# Row orders
# ---------------------------------------------------------------------------


def trtllm_gen_fc1_rows(intermediate: int) -> torch.Tensor:
    """Row of the natural ``[w3 (up); w1 (gate)]`` weight held by each TRTLLM-Gen FC1 row.

    Chains the helpers the same way as ``trtllmgen_maybe_get_cached_w3_w1_permute_indices``.
    """
    w3_w1 = torch.empty(2 * intermediate, 1)
    gated = get_reorder_rows_for_gated_act_gemm_row_indices(w3_w1)
    shuffled = get_shuffle_matrix_a_row_indices(w3_w1, epilogue_tile_m=EPILOGUE_TILE_M)
    return gated[shuffled]


def trtllm_gen_fc2_rows(hidden: int) -> torch.Tensor:
    """Natural hidden row held by each TRTLLM-Gen FC2 row."""
    return get_shuffle_matrix_a_row_indices(torch.empty(hidden, 1), epilogue_tile_m=EPILOGUE_TILE_M)


def w3_w1_row_label(row: int, intermediate: int) -> Label:
    return ("up", row) if row < intermediate else ("gate", row - intermediate)


def megamoe_fc1_row(label: Label) -> int:
    """Row of MegaMoE's native FC1 weight that holds ``label``: [gate16, up16] per window."""
    kind, channel = label
    group, offset = divmod(channel, FC1_WINDOW_CHANNELS)
    return WINDOW_ROWS * group + (0 if kind == "gate" else FC1_WINDOW_CHANNELS) + offset


def megamoe_to_trtllm_gen_fc1_rows(intermediate: int) -> torch.Tensor:
    """``trtllm_gen_fc1 = megamoe_fc1[..., rows, :]`` (rows of weights and block scales)."""
    labels = [w3_w1_row_label(r, intermediate) for r in trtllm_gen_fc1_rows(intermediate).tolist()]
    return torch.tensor([megamoe_fc1_row(label) for label in labels], dtype=torch.long)


# ---------------------------------------------------------------------------
# Epilogue register simulation
# ---------------------------------------------------------------------------


def ld16x64b_tmem_row(lane: int) -> int:
    """TMEM row, within a 16-lane half, that ``tcgen05.ld.16x64b`` hands to ``lane``.

    CUTLASS ``Copy_Traits<SM100_TMEM_LOAD_16dp64b16x>``: thread t = t0 + 2 * t1 + 4 * t2 reads
    datapath 8 * t0 + t2 and 32-bit column t1 + 2 * reg. This is the input distribution
    documented on ``TmemTranspose16x32``.
    """
    return 8 * (lane % 2) + lane // 4


def ld16x64b_token(lane: int, reg: int) -> int:
    """Token column that register ``reg`` of ``lane`` holds after ``tcgen05.ld.16x64b``."""
    return 2 * reg + (lane // 2) % 2


def simulate_fc1_slots(window: List[Label], exchange: bool) -> Dict[Tuple[int, int], Tuple]:
    """Replay the FC1 epilogue for one warp window, up to the transpose output.

    ``window[r]`` labels TMEM row r of the 32-row window. The top load covers rows 0-15 and the
    bottom load rows 16-31. With ``exchange`` the lane pair swap of
    ``SwapABFc1Epilogue._exchange_trtllm_gen_gate_up`` is applied. The 16x32 transposes then give
    each lane one token with the input's TMEM row L as register (slot) L, and the activation is
    elementwise on matched pairs, so the result is ``{(token, slot): (gate_label, up_label)}``.
    """

    def top(lane: int, reg: int) -> Label:
        return window[ld16x64b_tmem_row(lane)]

    def bottom(lane: int, reg: int) -> Label:
        return window[FC1_WINDOW_CHANNELS + ld16x64b_tmem_row(lane)]

    slots = {}
    for lane in range(32):
        for reg in range(16):
            token = ld16x64b_token(lane, reg)
            if exchange:
                partner = lane ^ 1
                # The shuffle only pairs registers of the same token column.
                assert ld16x64b_token(partner, reg) == token
                # send = even ? bottom : top; gate = even ? recv : bottom; up = even ? top : recv
                received = bottom(partner, reg) if partner % 2 == 0 else top(partner, reg)
                is_even = lane % 2 == 0
                gate = received if is_even else bottom(lane, reg)
                up = top(lane, reg) if is_even else received
            else:
                gate, up = top(lane, reg), bottom(lane, reg)
            key = (token, ld16x64b_tmem_row(lane))
            assert key not in slots, "two lanes claim the same (token, slot)"
            slots[key] = (gate, up)
    assert len(slots) == 32 * FC1_WINDOW_CHANNELS
    return slots


def fc1_window(labels: List[Label], window_idx: int) -> List[Label]:
    """Window-relative labels; every row must belong to the warp's own 16 channels."""
    base = window_idx * FC1_WINDOW_CHANNELS
    window = []
    for kind, channel in labels[window_idx * WINDOW_ROWS : (window_idx + 1) * WINDOW_ROWS]:
        assert base <= channel < base + FC1_WINDOW_CHANNELS, (
            f"row of channel {channel} left FC1 window {window_idx}"
        )
        window.append((kind, channel - base))
    return window


@pytest.mark.cpu_only
@pytest.mark.parametrize("intermediate", [32, 512, 1056])
def test_trtllm_gen_fc1_window_layout_matches_the_spec(intermediate):
    labels = [w3_w1_row_label(r, intermediate) for r in trtllm_gen_fc1_rows(intermediate).tolist()]
    for window_idx in range(2 * intermediate // WINDOW_ROWS):
        window = fc1_window(labels, window_idx)
        # Rows 0-7 / 8-15 / 16-23 / 24-31 are up(2b) / gate(2b) / up(2b+1) / gate(2b+1).
        assert window == [
            ("up" if (row // 8) % 2 == 0 else "gate", 2 * (row % 8) + row // 16)
            for row in range(WINDOW_ROWS)
        ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("intermediate", [32, 512, 1056])
def test_fc1_exchange_and_rename_restore_natural_gate_up_pairs(intermediate):
    labels = [w3_w1_row_label(r, intermediate) for r in trtllm_gen_fc1_rows(intermediate).tolist()]
    for window_idx in range(2 * intermediate // WINDOW_ROWS):
        slots = simulate_fc1_slots(fc1_window(labels, window_idx), exchange=True)
        for (token, slot), (gate, up) in slots.items():
            assert gate == ("gate", FC1_CHANNEL_OF_SLOT[slot]), (window_idx, token, slot)
            assert up == ("up", FC1_CHANNEL_OF_SLOT[slot]), (window_idx, token, slot)
        # _trtllm_gen_slots_to_channels: channel c reads slot FC1_SLOT_OF_CHANNEL[c]. The 16
        # values per token are then in natural order, so the NVFP4 scale block (16 channels, or
        # one half of a paired 32-channel MXFP8 block) and the packed FP4 bytes are natural too.
        for token in range(32):
            for channel in range(FC1_WINDOW_CHANNELS):
                gate, up = slots[(token, FC1_SLOT_OF_CHANNEL[channel])]
                assert gate == ("gate", channel) and up == ("up", channel), (token, channel)


@pytest.mark.cpu_only
def test_fc1_simulation_controls():
    # The native layout needs neither step: slot == channel.
    native = [("gate", c) for c in range(16)] + [("up", c) for c in range(16)]
    for (token, slot), pair in simulate_fc1_slots(native, exchange=False).items():
        assert pair == (("gate", slot), ("up", slot))
    # Without the exchange TRTLLM-Gen rows never form a gate / up pair: the simulation can fail.
    labels = [w3_w1_row_label(r, 512) for r in trtllm_gen_fc1_rows(512).tolist()]
    pairs = simulate_fc1_slots(fc1_window(labels, 0), exchange=False).values()
    assert all(gate[0] != "gate" or up[0] != "up" for gate, up in pairs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("hidden", [128, 1024, 2880, 7168])
def test_fc2_rename_and_ublk_columns_restore_natural_hidden(hidden):
    rows = trtllm_gen_fc2_rows(hidden).tolist()
    for window_idx in range(hidden // WINDOW_ROWS):
        base = window_idx * WINDOW_ROWS
        window = [row - base for row in rows[base : base + WINDOW_ROWS]]
        assert window == list(FC2_HIDDEN_OF_ROW), f"FC2 window {window_idx}"
        # MOVM paths (STG / TMA / quantized combine / REDG before its re-transpose):
        # TmemTranspose32x64B16Movm hands lane t, element e the token t + 32 * (e // 32) and TMEM
        # row e % 32; fc2_tmem_rows_to_hidden_order moves element 32 q + FC2_ROW_OF_HIDDEN[h] to
        # 32 q + h. Every element must land on its hidden channel, which also keeps the 16 / 32
        # wide combine scale blocks natural and contiguous.
        for element in range(64):
            token_group, hidden_in_window = divmod(element, WINDOW_ROWS)
            source = token_group * WINDOW_ROWS + FC2_ROW_OF_HIDDEN[hidden_in_window]
            assert window[source % WINDOW_ROWS] == hidden_in_window, (window_idx, element)
        # BF16 UBLK path: lane t loaded TMEM row t (tcgen05.ld.32x32b) and stores it to SMEM
        # column warp * 32 + FC2_HIDDEN_OF_ROW[t].
        for lane in range(WINDOW_ROWS):
            assert FC2_HIDDEN_OF_ROW[lane] == window[lane], (window_idx, lane)


@pytest.mark.cpu_only
def test_megamoe_fc1_row_map_matches_the_production_packer():
    # The GPU tests derive TRTLLM-Gen weights from MegaMoE-packed ones via megamoe_fc1_row.
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    intermediate, hidden = 64, 64
    rows = 2 * intermediate
    # Byte k of every row stores the row id, so the packed output reveals each row's source.
    row_ids = torch.arange(rows, dtype=torch.uint8).view(1, rows, 1)
    mega_fc1, _, _, _ = method._build_mega_format_buffers(
        raw_w3_w1=row_ids.expand(1, rows, hidden // 2).contiguous(),
        raw_w3_w1_sf=row_ids.expand(1, rows, hidden // 16).contiguous(),
        raw_w2=torch.zeros(1, hidden, intermediate // 2, dtype=torch.uint8),
        raw_w2_sf=torch.zeros(1, hidden, intermediate // 16, dtype=torch.uint8),
        num_slots=1,
        intermediate=intermediate,
        hidden=hidden,
        expand_intermediate=rows,
    )
    source_of_mega_row = mega_fc1[0, :, 0].tolist()
    for w3_w1_row in range(rows):
        mega_row = megamoe_fc1_row(w3_w1_row_label(w3_w1_row, intermediate))
        assert source_of_mega_row[mega_row] == w3_w1_row


@pytest.mark.cpu_only
def test_kernel_uses_the_simulated_mappings():
    pytest.importorskip("cutlass")
    epilogue = importlib.import_module(
        "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega."
        "block_scaled_swap_ab_fc12_epilogue"
    )
    assert epilogue.WEIGHT_ROW_ORDERS == ("megamoe", "trtllm_gen")
    assert epilogue._TRTLLM_GEN_FC1_SLOT_OF_CHANNEL == FC1_SLOT_OF_CHANNEL
    assert tuple(map(epilogue._trtllm_gen_fc1_channel_of_slot, range(16))) == FC1_CHANNEL_OF_SLOT
    assert epilogue._TRTLLM_GEN_FC2_ROW_OF_HIDDEN == FC2_ROW_OF_HIDDEN
    assert tuple(map(epilogue._trtllm_gen_fc2_hidden_of_row, range(32))) == FC2_HIDDEN_OF_ROW


@pytest.mark.cpu_only
def test_weight_row_order_descriptor_validation():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import ProblemDesc

    epilogue = importlib.import_module(
        "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega."
        "block_scaled_swap_ab_fc12_epilogue"
    )
    resolve = epilogue.SwapABGatedActEpilogue._resolve_weight_row_order
    shape = {"hidden_size": 1024, "intermediate_gateup_size": 1024}
    assert resolve(ProblemDesc(shape)) == "megamoe"
    for order in ("megamoe", "trtllm_gen"):
        assert resolve(ProblemDesc({**shape, "weight_row_order": order})) == order
    with pytest.raises(ValueError, match="weight_row_order must be one of"):
        resolve(ProblemDesc({**shape, "weight_row_order": "trtllm-gen"}))
    with pytest.raises(ValueError, match="hidden_size to be a multiple of 32"):
        resolve(ProblemDesc({**shape, "hidden_size": 1040, "weight_row_order": "trtllm_gen"}))

    # The default runner identity is unchanged; the TRTLLM-Gen one is distinct.
    key_args = dict(
        sm_version=100,
        world_size=1,
        num_topk=4,
        num_experts_per_rank=8,
        hidden_size=1024,
        intermediate_size_per_partition=512,
        expand_intermediate_size_per_partition=1024,
        max_tokens_per_rank=512,
        output_dtype=torch.bfloat16,
        apply_topk_in_fc1=True,
        gate_up_clamp=None,
        situ_beta=None,
        situ_linear_beta=None,
        in_kernel_fc2_reduce=False,
        combine_format="bf16",
    )
    native_key = megamoe_op._megamoe_problem_key(**key_args)
    assert len(native_key) == 17
    assert megamoe_op._megamoe_problem_key(**key_args, weight_row_order="megamoe") == native_key
    trtllm_gen_key = megamoe_op._megamoe_problem_key(**key_args, weight_row_order="trtllm_gen")
    assert trtllm_gen_key[:17] == native_key and trtllm_gen_key != native_key

    # The Rubin kernels do not implement the TRTLLM-Gen order and must not silently ignore it.
    with pytest.raises(NotImplementedError, match="only implemented by the Blackwell"):
        megamoe_op._construct_megamoe_kernel(
            sm_version=107,
            common={"static_expert_shape": (8, 1024, 1024), "weight_row_order": "trtllm_gen"},
            combine_format="bf16",
        )


# ---------------------------------------------------------------------------
# GPU: the kernel on both layouts of the same weights
# ---------------------------------------------------------------------------

blackwell_only = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="weight_row_order='trtllm_gen' is implemented by the Blackwell (SM100/SM103) kernel",
)

NUM_EXPERTS = 8  # ep = 1, so these are the local experts
HIDDEN = 1024
INTERMEDIATE = 512
MAX_TOKENS = 512  # launch bucket; live tokens are the leading rows
NVFP4_BLOCK = 16
FP4_GLOBAL = 448.0 * 6.0  # e4m3 max * e2m1 max

TACTIC_TMA_BF16 = (
    [256, 256, 256],
    [2, 1, 1],
    None,
    ("grouped", 512),
    "grid_stride",
    "reuse_dispatch_warps",
    True,
    2,
    4,
    (1, 1),
)
TACTIC_UBLK_N256 = (
    [256, 256, 256],
    [2, 1, 1],
    None,
    ("grouped", 512),
    "grid_stride",
    "epi_warps",
    True,
    1,
    4,
    (1, 1),
)
TACTIC_STG_LOCAL = (
    [256, 256, 256],
    [2, 1, 1],
    None,
    ("grouped", 512),
    "grid_stride",
    "standalone_warps",
    False,
    None,
    1,
    (2, 4),
)
TACTIC_STG_PEER = (
    [256, 256, 256],
    [2, 1, 1],
    None,
    ("grouped", 512),
    "grid_stride",
    "epi_warps",
    False,
    None,
    1,
    (1, 1),
)


def _megamoe_op():
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    if not megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        pytest.skip(str(megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON))
    return megamoe_op


def _quantize_nvfp4(weight: torch.Tensor, global_scale: torch.Tensor):
    """``[rows, k]`` bf16 -> packed e2m1 ``[rows, k / 2]`` and linear e4m3 scales ``[rows, k / 16]``."""
    packed, scales = torch.ops.trtllm.fp4_quantize(weight, global_scale, NVFP4_BLOCK, False, False)
    rows, k = weight.shape
    return packed.view(torch.uint8), scales.view(torch.uint8).view(rows, k // NVFP4_BLOCK)


def _to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """``[rows, cols]`` (rows % 128 == 0, cols % 4 == 0) -> the kernel's flat 128x4 block layout.

    Same view / permute as ``NVFP4MegaMoECuteDslMethod._build_mega_sf``.
    """
    rows, cols = scales.shape
    return scales.view(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def _from_blocked(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    # The (0, 3, 2, 1, 4) permutation is its own inverse.
    return flat.view(rows // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, cols)


def _permute_scale_rows(flat_per_slot: torch.Tensor, rows: torch.Tensor, k: int) -> torch.Tensor:
    """Unswizzle -> permute the leading ``len(rows)`` rows -> swizzle, for every expert slot."""
    padded_rows = -(-len(rows) // 128) * 128
    padded_cols = -(-(k // NVFP4_BLOCK) // 4) * 4
    out = torch.empty_like(flat_per_slot)
    for slot in range(flat_per_slot.shape[0]):
        used = padded_rows * padded_cols
        linear = _from_blocked(flat_per_slot[slot, :used], padded_rows, padded_cols).clone()
        linear[: len(rows)] = linear[rows.to(linear.device)]
        out[slot, :used] = _to_blocked(linear)
        out[slot, used:] = flat_per_slot[slot, used:]
    return out


@functools.lru_cache(maxsize=None)
def _expert_weights():
    """One expert bank in both layouts plus the bf16 weights used to pick scales."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    generator = torch.Generator(device="cuda").manual_seed(2026)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    w1 = (randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN) * HIDDEN**-0.5).bfloat16()  # gate
    w3 = (randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN) * HIDDEN**-0.5).bfloat16()  # up
    w2 = (randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * INTERMEDIATE**-0.5).bfloat16()
    w3_w1 = torch.cat([w3, w1], dim=1)  # TRT-LLM's [w3; w1] FC1 checkpoint order

    fc1_global = FP4_GLOBAL / w3_w1.float().abs().amax(dim=(1, 2))
    fc2_global = FP4_GLOBAL / w2.float().abs().amax(dim=(1, 2))
    raw = [_quantize_nvfp4(w3_w1[e], fc1_global[e]) for e in range(NUM_EXPERTS)]
    raw2 = [_quantize_nvfp4(w2[e], fc2_global[e]) for e in range(NUM_EXPERTS)]

    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    mega_fc1, mega_fc1_sf, mega_fc2, mega_fc2_sf = method._build_mega_format_buffers(
        raw_w3_w1=torch.stack([w for w, _ in raw]),
        raw_w3_w1_sf=torch.stack([s for _, s in raw]),
        raw_w2=torch.stack([w for w, _ in raw2]),
        raw_w2_sf=torch.stack([s for _, s in raw2]),
        num_slots=NUM_EXPERTS,
        intermediate=INTERMEDIATE,
        hidden=HIDDEN,
        expand_intermediate=2 * INTERMEDIATE,
    )
    # The local 128x4 helpers must reproduce the production swizzle before they are trusted
    # to re-swizzle the permuted scales.
    fc1_sf_linear = _from_blocked(mega_fc1_sf[0], 2 * INTERMEDIATE, HIDDEN // NVFP4_BLOCK)
    assert torch.equal(_to_blocked(fc1_sf_linear), mega_fc1_sf[0])
    fc1_mega_rows = torch.tensor(
        [megamoe_fc1_row(w3_w1_row_label(r, INTERMEDIATE)) for r in range(2 * INTERMEDIATE)],
        device="cuda",
    )
    assert torch.equal(fc1_sf_linear[fc1_mega_rows], raw[0][1])

    fc1_rows = megamoe_to_trtllm_gen_fc1_rows(INTERMEDIATE).cuda()
    fc2_rows = trtllm_gen_fc2_rows(HIDDEN).cuda()
    layouts = {
        "megamoe": (mega_fc1, mega_fc1_sf, mega_fc2, mega_fc2_sf),
        "trtllm_gen": (
            mega_fc1[:, fc1_rows].contiguous(),
            _permute_scale_rows(mega_fc1_sf, fc1_rows, HIDDEN),
            mega_fc2[:, fc2_rows].contiguous(),
            _permute_scale_rows(mega_fc2_sf, fc2_rows, INTERMEDIATE),
        ),
    }
    return layouts, (w1, w3, w2), fc1_global, fc2_global


def _launch_inputs(num_tokens: int, top_k: int, seed: int = 7):
    """Activation, routing and per-expert scalars for ``num_tokens`` live rows."""
    _, (w1, w3, _), fc1_global, fc2_global = _expert_weights()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator, device="cuda").bfloat16() * 2.0
    x_global = FP4_GLOBAL / x.float().abs().amax()
    x_fp4, x_sf = _quantize_nvfp4(x, x_global)

    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        megamoe_activation_sf_bytes_per_row,
    )

    activation = torch.zeros(MAX_TOKENS, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    activation_sf = torch.zeros(
        MAX_TOKENS, megamoe_activation_sf_bytes_per_row(HIDDEN), dtype=torch.uint8, device="cuda"
    )
    activation[:num_tokens] = x_fp4
    activation_sf[:num_tokens, : x_sf.shape[1]] = x_sf

    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=generator, device="cuda")
    topk_logits, topk_ids = logits.topk(top_k, dim=-1)
    topk_idx = torch.full((MAX_TOKENS, top_k), -1, dtype=torch.int64, device="cuda")
    topk_weights = torch.zeros(MAX_TOKENS, top_k, dtype=torch.float32, device="cuda")
    topk_idx[:num_tokens] = topk_ids
    topk_weights[:num_tokens] = topk_logits.softmax(dim=-1)

    # FC1 output global scale from an unquantized SwiGLU estimate of the intermediate, with 4x
    # headroom so that no activation variant can push a block scale past the e4m3 range.
    gate = torch.einsum("th,eih->eti", x.float(), w1.float())
    up = torch.einsum("th,eih->eti", x.float(), w3.float())
    intermediate_global = FP4_GLOBAL / (4.0 * (torch.nn.functional.silu(gate) * up).abs().amax())
    fc1_alpha = (1.0 / (x_global * fc1_global)).float().contiguous()
    fc2_alpha = (1.0 / (intermediate_global * fc2_global)).float().contiguous()
    fc1_norm_const = intermediate_global.expand(NUM_EXPERTS).float().contiguous()
    return activation, activation_sf, topk_idx, topk_weights, fc1_alpha, fc2_alpha, fc1_norm_const


def _run_megamoe(
    megamoe_op,
    *,
    layout: str,
    weight_row_order: str,
    inputs,
    num_tokens: int,
    top_k: int,
    tactic,
    combine_format: str = "bf16",
    in_kernel_fc2_reduce: bool = False,
    activation_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    layouts, *_ = _expert_weights()
    fc1, fc1_sf, fc2, fc2_sf = layouts[layout]
    activation, activation_sf, topk_idx, topk_weights, fc1_alpha, fc2_alpha, norm_const = inputs
    problem = dict(
        world_size=1,
        num_topk=top_k,
        num_experts_per_rank=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        expand_intermediate_size_per_partition=2 * INTERMEDIATE,
        max_tokens_per_rank=MAX_TOKENS,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        combine_format=combine_format,
        **(activation_kwargs or {}),
    )
    shared_bytes, _ = megamoe_op.query_megamoe_kernel_properties(
        **problem, tactic=None if tactic == -1 else tactic
    )
    runner = megamoe_op.Sm100MegaMoENvfp4Runner(
        local_rank=0, output_dtype=torch.bfloat16, weight_row_order=weight_row_order, **problem
    )
    combine_output = torch.zeros(MAX_TOKENS, 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    fp4, fp8 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
    runner(
        [
            activation.view(fp4),
            activation_sf.view(fp8),
            topk_idx,
            topk_weights,
            fc1.view(fp4),
            fc1_sf.view(fp8),
            fc2.view(fp4),
            fc2_sf.view(fp8),
            fc1_alpha,
            fc2_alpha,
            norm_const,
            combine_output,
        ],
        tactic=tactic,
        peer_offsets=[0],
        shared_workspace=torch.zeros(shared_bytes, dtype=torch.uint8, device="cuda"),
    )
    torch.cuda.synchronize()
    return combine_output[:num_tokens, 0].clone()


def _assert_row_orders_agree(megamoe_op, *, num_tokens: int, top_k: int, **run_kwargs):
    inputs = _launch_inputs(num_tokens, top_k)
    run = functools.partial(
        _run_megamoe, megamoe_op, inputs=inputs, num_tokens=num_tokens, top_k=top_k, **run_kwargs
    )
    reference = run(layout="megamoe", weight_row_order="megamoe")
    assert torch.isfinite(reference).all() and reference.abs().sum() > 0
    reference_again = run(layout="megamoe", weight_row_order="megamoe")
    actual = run(layout="trtllm_gen", weight_row_order="trtllm_gen")
    if torch.equal(reference, reference_again):
        # Row order only moves accumulator rows and registers, so the math is order-free.
        max_diff = (actual.float() - reference.float()).abs().max().item()
        assert torch.equal(actual, reference), f"not bit-identical, max |diff| = {max_diff}"
    else:
        # A non-deterministic baseline cannot demand bitwise equality.
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    # Control: the native epilogue on TRTLLM-Gen weights must not match.
    mismatched = run(layout="trtllm_gen", weight_row_order="megamoe")
    assert not torch.allclose(mismatched, reference, rtol=5e-2, atol=5e-2)


ACTIVATIONS = {
    "swiglu": {},
    "swiglu_clamp": {"gate_up_clamp": 7.0},
    "swiglu_bias_clamp": {"swiglu_alpha": 1.702, "swiglu_beta": 1.0, "gate_up_clamp": 7.0},
    "situ": {"situ_beta": 4.0, "situ_linear_beta": 25.0},
    "swiglu_topk_at_fc2": {"apply_topk_in_fc1": False},
}


@blackwell_only
@pytest.mark.parametrize("num_tokens", [1, 64, 300])
@pytest.mark.parametrize("activation", list(ACTIVATIONS))
def test_trtllm_gen_rows_match_megamoe_rows(num_tokens, activation):
    """Default tactic (decode bucket: BF16 UBLK token-back) for several token counts."""
    _assert_row_orders_agree(
        _megamoe_op(),
        num_tokens=num_tokens,
        top_k=4,
        tactic=-1,
        activation_kwargs=ACTIVATIONS[activation],
    )


FC2_PATHS = {
    # name: (tactic, combine_format, in_kernel_fc2_reduce, top_k)
    "ublk_bf16_n256": (TACTIC_UBLK_N256, "bf16", False, 4),
    "tma_bf16": (TACTIC_TMA_BF16, "bf16", False, 4),
    "stg_local_bf16": (TACTIC_STG_LOCAL, "bf16", False, 4),
    "stg_peer_bf16": (TACTIC_STG_PEER, "bf16", False, 4),
    # Two contributions per token keep the in-kernel BF16 atomic reduction order-independent.
    "redg_bf16": (TACTIC_STG_PEER, "bf16", True, 2),
    "stg_local_fp8": (TACTIC_STG_LOCAL, "32e4m3xe8m0", False, 4),
    "ublk_fp8": (TACTIC_UBLK_N256, "32e4m3xe8m0", False, 4),
    "tma_fp4": (TACTIC_TMA_BF16, "16e2m1xbf16", False, 4),
    "stg_peer_fp4": (TACTIC_STG_PEER, "16e2m1xbf16", False, 4),
}


@blackwell_only
@pytest.mark.parametrize("path", list(FC2_PATHS))
def test_trtllm_gen_rows_on_every_fc2_store_path(path):
    """Explicit tactics covering each FC2 store path, with N=256 tiles (overlapped accumulators)."""
    tactic, combine_format, in_kernel_fc2_reduce, top_k = FC2_PATHS[path]
    _assert_row_orders_agree(
        _megamoe_op(),
        num_tokens=300,
        top_k=top_k,
        tactic=tactic,
        combine_format=combine_format,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
    )
