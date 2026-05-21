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

"""CPU-only tests for bench_moe.py routing-control planning helpers.

The tests intentionally do not touch GPU / MPI / TRT-LLM kernels. They only
exercise the pure-Python plan construction, materialisation, and observation
helpers that live in ``bench_moe.py``.

Run with::

    pytest tests/microbenchmarks/test_bench_moe_routing_control.py
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

# Mirror the path setup in bench_moe.py so this test can import the benchmark
# module without requiring an installed TRT-LLM wheel.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TESTS_UNITTEST_DIR = _REPO_ROOT / "tests" / "unittest"
if str(_TESTS_UNITTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_UNITTEST_DIR))

# Importing bench_moe triggers heavy ``tensorrt_llm`` imports. Skip when those
# are not available (e.g. lean container without the in-tree TRT-LLM build).
bench_moe = pytest.importorskip("bench_moe")  # noqa: E402

RoutingControlSpec = bench_moe.RoutingControlSpec
RoutingPlan = bench_moe.RoutingPlan


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------


def test_parse_comm_pattern_balanced():
    name, kwargs = bench_moe._parse_comm_pattern("balanced_alltoall")
    assert name == "balanced_alltoall"
    assert kwargs == {}


def test_parse_comm_pattern_receiver_hotspot():
    name, kwargs = bench_moe._parse_comm_pattern("receiver_hotspot,hotness=0.75,rank=2")
    assert name == "receiver_hotspot"
    assert kwargs == {"hotness": 0.75, "rank": 2}


def test_parse_comm_pattern_pair_hotspot():
    name, kwargs = bench_moe._parse_comm_pattern("pair_hotspot,hotness=0.5,src=0,dst=1")
    assert name == "pair_hotspot"
    assert kwargs == {"hotness": 0.5, "src": 0, "dst": 1}


def test_parse_comm_pattern_rejects_file_input():
    with pytest.raises(ValueError):
        bench_moe._parse_comm_pattern("file:/tmp/matrix.json")


def test_parse_comm_pattern_invalid():
    with pytest.raises(ValueError):
        bench_moe._parse_comm_pattern("unknown_pattern")


def test_parse_expert_pattern_active_experts():
    name, kwargs = bench_moe._parse_expert_pattern("hotspot,active_experts=2")
    assert name == "hotspot"
    assert kwargs == {"active_experts": 2}


def test_parse_expert_pattern_requires_args():
    with pytest.raises(ValueError):
        bench_moe._parse_expert_pattern("hotspot")


# ---------------------------------------------------------------------------
# Dispatch matrix builders
# ---------------------------------------------------------------------------


def test_balanced_alltoall_row_sums_match_top_k():
    per_rank = [4, 4, 4, 4]
    top_k = 4
    ep_size = 4
    matrix = bench_moe._build_dispatch_matrix("balanced_alltoall", per_rank, top_k, ep_size)
    for src in range(ep_size):
        assert sum(matrix[src]) == per_rank[src] * top_k
    # Uniform within rounding.
    for row in matrix:
        assert max(row) - min(row) <= 1


def test_receiver_hotspot_adds_traffic_to_target_column():
    per_rank = [8, 8, 8, 8]
    top_k = 4
    ep_size = 4
    matrix = bench_moe._build_dispatch_matrix(
        "receiver_hotspot,hotness=0.75,rank=1", per_rank, top_k, ep_size
    )
    col_sums = [sum(matrix[s][d] for s in range(ep_size)) for d in range(ep_size)]
    # rank 1 should dominate.
    assert col_sums[1] == max(col_sums)


def test_pair_hotspot_concentrates_on_pair():
    per_rank = [16, 16, 16, 16]
    top_k = 4
    ep_size = 4
    matrix = bench_moe._build_dispatch_matrix(
        "pair_hotspot,hotness=0.5,src=0,dst=1", per_rank, top_k, ep_size
    )
    # Row 0 should concentrate on dst=1; row 2 should stay balanced.
    assert matrix[0][1] == max(matrix[0])
    row2 = matrix[2]
    assert max(row2) - min(row2) <= 1


def test_local_only_diagonal_dominates():
    per_rank = [16, 16, 16, 16]
    top_k = 4
    ep_size = 4
    matrix = bench_moe._build_dispatch_matrix("local_only", per_rank, top_k, ep_size)
    for src in range(ep_size):
        assert matrix[src][src] == per_rank[src] * top_k
        for dst in range(ep_size):
            if dst != src:
                assert matrix[src][dst] == 0


def test_ring_pattern_targets_next_rank():
    per_rank = [16, 16, 16, 16]
    top_k = 4
    ep_size = 4
    matrix = bench_moe._build_dispatch_matrix("ring", per_rank, top_k, ep_size)
    for src in range(ep_size):
        nxt = (src + 1) % ep_size
        assert matrix[src][nxt] == per_rank[src] * top_k
        for dst in range(ep_size):
            if dst != nxt:
                assert matrix[src][dst] == 0


# ---------------------------------------------------------------------------
# Expert histogram builders
# ---------------------------------------------------------------------------


def test_expert_pattern_balanced_distributes_uniformly():
    per_rank = [16, 16, 16, 16]
    top_k = 4
    ep_size = 4
    experts_per_rank = 4
    dispatch = bench_moe._build_dispatch_matrix("balanced_alltoall", per_rank, top_k, ep_size)
    hist = bench_moe._build_expert_histogram("balanced", dispatch, experts_per_rank, ep_size)
    for row in hist:
        assert max(row) - min(row) <= 1


def test_expert_pattern_hotspot_active_experts_limits_active_count():
    per_rank = [16, 16, 16, 16]
    top_k = 4
    ep_size = 4
    experts_per_rank = 8
    dispatch = bench_moe._build_dispatch_matrix("balanced_alltoall", per_rank, top_k, ep_size)
    hist = bench_moe._build_expert_histogram(
        "hotspot,active_experts=2", dispatch, experts_per_rank, ep_size
    )
    for row in hist:
        active = sum(1 for v in row if v > 0)
        assert active <= 2


# ---------------------------------------------------------------------------
# Full plan construction + materialisation
# ---------------------------------------------------------------------------


def _build_simple_plan(spec_kwargs=None, num_tokens=64, world_size=4, top_k=4, num_experts=16):
    spec = RoutingControlSpec(**(spec_kwargs or {}))
    return bench_moe._build_routing_plan(
        spec,
        num_tokens=num_tokens,
        world_size=world_size,
        top_k=top_k,
        num_experts=num_experts,
        moe_ep_size=world_size,
    ), spec


def test_routing_plan_invariants_balanced():
    plan, _ = _build_simple_plan()
    # Row sums.
    for src, row in enumerate(plan.dispatch_matrix):
        assert sum(row) == plan.per_rank_num_tokens[src] * 4
    # Histogram column sums match dispatch column sums.
    ep_size = len(plan.dispatch_matrix)
    for dst in range(ep_size):
        col_sum = sum(plan.dispatch_matrix[s][dst] for s in range(ep_size))
        assert sum(plan.expert_histogram[dst]) == col_sum


def test_materialize_balanced_per_rank_shape_and_unique():
    plan, _ = _build_simple_plan()
    for src in range(4):
        ids, scales = bench_moe._materialize_selected_experts_for_rank(
            plan,
            src_rank=src,
            top_k=4,
            experts_per_rank=4,
            moe_ep_size=4,
            device=torch.device("cpu"),
            scale_dtype=torch.float32,
        )
        assert ids.shape == (plan.per_rank_num_tokens[src], 4)
        assert ids.dtype == torch.int32
        # Each row distinct expert ids.
        for row in ids.tolist():
            assert len(set(row)) == 4, f"row {row} has duplicates"
        # Scales are uniform 1/top_k.
        assert torch.allclose(scales, torch.full_like(scales, 0.25, dtype=scales.dtype), atol=1e-6)


def test_materialize_observes_match_plan_after_aggregation():
    plan, _ = _build_simple_plan()
    per_rank_ids = []
    for src in range(4):
        ids, _ = bench_moe._materialize_selected_experts_for_rank(
            plan,
            src_rank=src,
            top_k=4,
            experts_per_rank=4,
            moe_ep_size=4,
            device=torch.device("cpu"),
            scale_dtype=torch.float32,
        )
        per_rank_ids.append(ids)
    obs_slot, obs_token, obs_hist = bench_moe._observe_routing_metrics(
        plan, per_rank_ids, experts_per_rank=4, moe_ep_size=4
    )
    # Aggregated slot matrix must equal the planned matrix.
    for src in range(4):
        for dst in range(4):
            assert obs_slot[src][dst] == plan.dispatch_matrix[src][dst], (
                f"mismatch at ({src},{dst}): observed={obs_slot[src][dst]} planned={plan.dispatch_matrix[src][dst]}"
            )
    # Token traffic <= slot traffic per cell.
    for src in range(4):
        for dst in range(4):
            assert obs_token[src][dst] <= obs_slot[src][dst]


def test_explicit_per_rank_num_tokens_drives_max_per_rank():
    spec = RoutingControlSpec(per_rank_num_tokens=(128, 128, 512, 128))
    plan = bench_moe._build_routing_plan(
        spec,
        num_tokens=128 + 128 + 512 + 128,
        world_size=4,
        top_k=4,
        num_experts=16,
        moe_ep_size=4,
    )
    assert list(plan.per_rank_num_tokens) == [128, 128, 512, 128]
    # The matrix row sums reflect the explicit per-rank counts.
    assert sum(plan.dispatch_matrix[2]) == 512 * 4


def test_routing_pattern_file_drives_dispatch_and_expert_histogram(tmp_path):
    path = tmp_path / "routing_pattern.json"
    matrix = [
        [64, 0, 0, 0],
        [0, 64, 0, 0],
        [0, 0, 64, 0],
        [0, 0, 0, 64],
    ]
    histogram = [
        [64, 0, 0, 0],
        [0, 64, 0, 0],
        [0, 0, 64, 0],
        [0, 0, 0, 64],
    ]
    with open(path, "w") as f:
        json.dump(
            {
                "ep_size": 4,
                "experts_per_rank": 4,
                "slot_dispatch_matrix": matrix,
                "expert_histogram": histogram,
            },
            f,
        )

    spec = RoutingControlSpec(routing_pattern_file=str(path))
    plan = bench_moe._build_routing_plan(
        spec,
        num_tokens=64,
        world_size=4,
        top_k=4,
        num_experts=16,
        moe_ep_size=4,
    )

    assert plan.dispatch_matrix == tuple(tuple(row) for row in matrix)
    assert plan.expert_histogram == tuple(tuple(row) for row in histogram)


def test_per_rank_num_tokens_length_mismatch_raises():
    spec = RoutingControlSpec(per_rank_num_tokens=(1, 2, 3))
    with pytest.raises(ValueError):
        bench_moe._build_routing_plan(
            spec,
            num_tokens=6,
            world_size=4,
            top_k=2,
            num_experts=8,
            moe_ep_size=4,
        )


def test_invalid_num_experts_divisibility_raises():
    spec = RoutingControlSpec()
    with pytest.raises(ValueError):
        bench_moe._build_routing_plan(
            spec,
            num_tokens=16,
            world_size=3,
            top_k=2,
            num_experts=10,  # 10 % 3 != 0
            moe_ep_size=3,
        )


def test_top_k_exceeds_experts_per_rank_for_single_target():
    # Single-target case (local_only) would need top_k distinct local experts.
    # With experts_per_rank=1 and top_k=2 the materialiser should fail to
    # produce unique-per-token ids; we surface this via downstream errors.
    spec = RoutingControlSpec(comm_pattern="local_only")
    plan = bench_moe._build_routing_plan(
        spec,
        num_tokens=8,
        world_size=4,
        top_k=2,
        num_experts=4,  # experts_per_rank = 1
        moe_ep_size=4,
    )
    # Per-rank size will be 2 here; ensure materialisation reports duplicate
    # row entries because top_k > experts_per_rank.
    ids, _ = bench_moe._materialize_selected_experts_for_rank(
        plan,
        src_rank=0,
        top_k=2,
        experts_per_rank=1,
        moe_ep_size=4,
        device=torch.device("cpu"),
        scale_dtype=torch.float32,
    )
    # With only 1 local expert per rank and local_only routing, every slot
    # in a row maps to the same expert id, which is intentionally illegal.
    rows = ids.tolist()
    duplicated = any(len(set(row)) < len(row) for row in rows)
    assert duplicated, "expected duplicates when top_k > experts_per_rank in local_only"


# ---------------------------------------------------------------------------
# File-based input
# ---------------------------------------------------------------------------


def test_dispatch_matrix_file_round_trip(tmp_path):
    path = tmp_path / "matrix.json"
    matrix = [
        [64, 64, 64, 64],
        [64, 64, 64, 64],
        [64, 64, 64, 64],
        [64, 64, 64, 64],
    ]
    with open(path, "w") as f:
        json.dump({"ep_size": 4, "slot_dispatch_matrix": matrix}, f)
    loaded = bench_moe._load_dispatch_matrix_file(str(path), ep_size=4)
    assert loaded == matrix


def test_dispatch_matrix_file_does_not_require_schema(tmp_path):
    path = tmp_path / "no_schema.json"
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    with open(path, "w") as f:
        json.dump({"ep_size": 4, "slot_dispatch_matrix": matrix}, f)

    assert bench_moe._load_dispatch_matrix_file(str(path), ep_size=4) == matrix


def test_dispatch_matrix_file_ep_mismatch(tmp_path):
    path = tmp_path / "wrong_ep.json"
    with open(path, "w") as f:
        json.dump(
            {"ep_size": 2, "slot_dispatch_matrix": [[1, 0], [0, 1]]},
            f,
        )
    with pytest.raises(ValueError):
        bench_moe._load_dispatch_matrix_file(str(path), ep_size=4)


# ---------------------------------------------------------------------------
# is_active flag and routing_control summary output
# ---------------------------------------------------------------------------


def test_is_active_default_false():
    assert RoutingControlSpec().is_active is False


def test_is_active_when_pattern_changed():
    assert RoutingControlSpec(comm_pattern="local_only").is_active is True


def test_is_active_when_routing_pattern_file_set():
    assert RoutingControlSpec(routing_pattern_file="/tmp/routing_pattern.json").is_active is True


def test_is_active_when_per_rank_tokens_set():
    assert RoutingControlSpec(per_rank_num_tokens=(1, 2)).is_active is True


def test_resolve_workloads_uses_balanced_total_num_tokens():
    args = Namespace(
        balanced_total_num_tokens=[4, 64, 32],
        num_tokens=None,
        routing_mode="native",
        projection_policy="project",
        comm_pattern="balanced_alltoall",
        expert_pattern="balanced",
        routing_pattern_file=None,
        per_rank_num_tokens=None,
        routing_dump_matrix=False,
        routing_seed=0,
    )
    workloads = bench_moe._resolve_workloads_from_args(args)

    assert [w.num_tokens for w in workloads] == [4, 64, 32]
    assert all(w.routing_control.per_rank_num_tokens is None for w in workloads)


def test_resolve_workloads_derives_total_from_per_rank_tokens():
    args = Namespace(
        balanced_total_num_tokens=None,
        num_tokens=None,
        routing_mode="native",
        projection_policy="project",
        comm_pattern="balanced_alltoall",
        expert_pattern="balanced",
        routing_pattern_file=None,
        per_rank_num_tokens=[1, 1, 1, 1],
        routing_dump_matrix=False,
        routing_seed=0,
    )
    workloads = bench_moe._resolve_workloads_from_args(args)

    assert len(workloads) == 1
    assert workloads[0].num_tokens == 4
    assert workloads[0].routing_control.per_rank_num_tokens == (1, 1, 1, 1)


def test_resolve_workloads_rejects_total_and_per_rank_token_knobs_together():
    args = Namespace(
        balanced_total_num_tokens=[4],
        num_tokens=None,
        routing_mode="native",
        projection_policy="project",
        comm_pattern="balanced_alltoall",
        expert_pattern="balanced",
        routing_pattern_file=None,
        per_rank_num_tokens=[1, 1, 1, 1],
        routing_dump_matrix=False,
        routing_seed=0,
    )
    with pytest.raises(ValueError):
        bench_moe._resolve_workloads_from_args(args)


def test_resolve_workloads_requires_explicit_expert_pattern_for_hotspot():
    args = Namespace(
        balanced_total_num_tokens=[64],
        num_tokens=[64],
        routing_mode="native",
        projection_policy="project",
        comm_pattern="balanced_alltoall",
        expert_pattern="hotspot,hotness=0.5",
        routing_pattern_file=None,
        per_rank_num_tokens=None,
        routing_dump_matrix=False,
        routing_seed=0,
    )
    workloads = bench_moe._resolve_workloads_from_args(args)
    assert len(workloads) == 1
    assert workloads[0].routing_control.expert_pattern == "hotspot,hotness=0.5"


def test_resolve_workloads_rejects_legacy_total_and_per_rank_token_knobs_together():
    args = Namespace(
        balanced_total_num_tokens=[64, 128],
        num_tokens=[64, 128],
        routing_mode="native",
        projection_policy="project",
        comm_pattern="balanced_alltoall",
        expert_pattern="balanced",
        routing_pattern_file=None,
        per_rank_num_tokens=[16, 16, 16, 16],
        routing_dump_matrix=False,
        routing_seed=0,
    )
    with pytest.raises(ValueError):
        bench_moe._resolve_workloads_from_args(args)


def test_parse_analysis_none_disables_kernel_collection():
    assert bench_moe._parse_analysis(["none"]) == ()
    assert bench_moe._parse_analysis("none") == ()
    assert bench_moe._parse_analysis(["kernels"]) == ("kernels",)


def test_parse_search_axes_accepts_space_and_comma_separated_values():
    assert bench_moe._parse_search_axes(["backend", "comm", "eplb"]) == ("backend", "comm", "eplb")
    assert bench_moe._parse_search_axes("backend,comm,eplb") == ("backend", "comm", "eplb")
    assert bench_moe._parse_search_axes(["backend,comm", "eplb"]) == ("backend", "comm", "eplb")


def test_parse_search_axes_rejects_conflicting_modes():
    with pytest.raises(ValueError):
        bench_moe._parse_search_axes(["none", "backend"])
    with pytest.raises(ValueError):
        bench_moe._parse_search_axes(["full", "comm"])


def test_resolve_search_combines_cli_axes_without_full_extras():
    args = Namespace(
        search=["backend", "comm"],
        backend="ALL",
        parallel_mode="DEP",
        comm_method="AUTO",
        eplb_mode="off",
        _config_search_axes={},
    )
    search = bench_moe._resolve_search_from_args(
        args, bench_moe.ConfigSpec(backend="TRTLLM", parallel_mode="DEP")
    )

    assert search.mode == "backend,comm"
    assert search.backends == tuple(bench_moe._ALL_BACKENDS)
    assert search.comm_methods == bench_moe._FORCED_COMM_ENV_VALUES + ("AUTO",)
    assert search.parallel_modes == ()
    assert search.eplb_modes == ()
    assert search.cuda_graph_options == ()


def test_config_search_lists_limit_expanded_axes(tmp_path):
    config_path = tmp_path / "bench_moe_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "search": {
                    "backend": ["TRTLLM", "CUTLASS"],
                    "parallel_mode": ["DEP"],
                    "comm_method": ["AUTO", "DEEPEP"],
                    "eplb_mode": ["off"],
                }
            },
            f,
        )

    args = Namespace(
        config_file=str(config_path),
        _cli_provided=set(),
        search="none",
        backend="TRTLLM",
        parallel_mode="DEP",
        comm_method="AUTO",
        eplb_mode="off",
    )
    args = bench_moe._maybe_load_config_file(args)
    search = bench_moe._resolve_search_from_args(
        args, bench_moe.ConfigSpec(backend="TRTLLM", parallel_mode="DEP")
    )

    assert search.mode == "backend,parallel,comm,eplb"
    assert search.backends == ("TRTLLM", "CUTLASS")
    assert search.parallel_modes == ("DEP",)
    assert search.comm_methods == ("AUTO", "DEEPEP")
    assert search.eplb_modes == ("off",)
    assert search.cuda_graph_options == ()


def test_config_search_list_respects_cli_override(tmp_path):
    config_path = tmp_path / "bench_moe_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "search": {
                    "backend": ["TRTLLM", "CUTLASS"],
                    "comm_method": ["AUTO", "DEEPEP"],
                }
            },
            f,
        )

    args = Namespace(
        config_file=str(config_path),
        _cli_provided={"backend"},
        search="none",
        backend="DENSEGEMM",
        parallel_mode="DEP",
        comm_method="AUTO",
        eplb_mode="off",
    )
    args = bench_moe._maybe_load_config_file(args)
    search = bench_moe._resolve_search_from_args(
        args, bench_moe.ConfigSpec(backend="DENSEGEMM", parallel_mode="DEP")
    )

    assert search.mode == "comm"
    assert search.backends == ()
    assert search.comm_methods == ("AUTO", "DEEPEP")


def test_observe_summary_matches_exact_plan():
    plan, _ = _build_simple_plan()
    requested = [list(row) for row in plan.dispatch_matrix]
    observed = [list(row) for row in plan.dispatch_matrix]
    max_abs, max_rel = bench_moe._observe_summary(requested, observed)
    assert max_abs == 0
    assert max_rel == 0.0
