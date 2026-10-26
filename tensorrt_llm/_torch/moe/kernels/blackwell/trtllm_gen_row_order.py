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
"""Where an output channel sits when NVFP4 expert weights are in TRTLLM-Gen row order.

TRTLLM-Gen's NVFP4 MoE kernels read expert weights whose rows are permuted
inside every 32-row block (``shuffle_matrix_a`` with a 128-row epilogue tile),
and the gated FC1 additionally interleaves up and gate rows one by one before
that (``reorder_rows_for_gated_act_gemm``). A grouped GEMM that takes these
weights as its B operand gets the same permutation on every aligned 32-column
group of its accumulator, so its epilogue can restore the natural channel
order by renaming registers, provided one thread holds a whole group.

The maps are plain integer functions so kernels can evaluate them at trace
time and tests can check them against the host-side permutation helpers.
"""

TRTLLM_GEN_ROW_GROUP = 32
"""int: Rows (accumulator columns) covered by one TRTLLM-Gen shuffle block."""

TRTLLM_GEN_GATED_CHANNELS_PER_GROUP = TRTLLM_GEN_ROW_GROUP // 2
"""int: Gated-FC1 output channels produced by one 32-row block."""


def trtllm_gen_row_source(channel: int) -> int:
    """Column of a 32-column group that holds output ``channel`` of a non-gated weight.

    Applies to FC2 and to a non-gated FC1: row ``r`` of a block holds channel
    ``4 * (r % 8) + r // 8``, so channel ``c`` sits in row ``8 * (c % 4) + c // 4``.

    Args:
        channel: Output channel inside the group, in ``[0, 32)``.

    Returns:
        The accumulator column inside the same group, in ``[0, 32)``.
    """
    if not 0 <= channel < TRTLLM_GEN_ROW_GROUP:
        raise ValueError(f"channel must be in [0, {TRTLLM_GEN_ROW_GROUP}), got {channel}")
    return 8 * (channel % 4) + channel // 4


def trtllm_gen_gated_sources(channel: int) -> tuple[int, int]:
    """Columns of a 32-column group holding the up and gate rows of gated-FC1 ``channel``.

    A block holds 16 output channels: rows 0-7 are up of the even channels,
    rows 8-15 gate of the even channels, rows 16-23 up of the odd channels and
    rows 24-31 gate of the odd channels.

    Args:
        channel: Output channel inside the group, in ``[0, 16)``.

    Returns:
        ``(up_column, gate_column)`` inside the same group, each in ``[0, 32)``.
    """
    if not 0 <= channel < TRTLLM_GEN_GATED_CHANNELS_PER_GROUP:
        raise ValueError(
            f"channel must be in [0, {TRTLLM_GEN_GATED_CHANNELS_PER_GROUP}), got {channel}"
        )
    pair, parity = divmod(channel, 2)
    up_column = 16 * parity + pair
    return up_column, up_column + 8
