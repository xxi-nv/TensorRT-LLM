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
"""Type-axis tests for swapping execution-unit backends onto ``MoEImplBase``.

The swap is the change that makes ``ConfigurableMoE.backend`` a different
type from a complete MoE layer. Failure mode of the old factory: an
execution-unit class constructed as a standalone layer has no ``forward``
and dies on the first call. These tests require that path to fail at
construction instead.
"""

from __future__ import annotations

from typing import Type, get_args

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.modules.fused_moe.create_moe import (
    EXECUTION_UNIT_IMPL_CLASSES,
    MoEImplClass,
    create_moe_backend,
)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_vanilla import VanillaMoE
from tensorrt_llm._torch.modules.fused_moe.impl_base import STANDALONE_MOE_IMPL_ERROR, MoEImplBase
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.weight_owner import is_moe_weight_owner
from tensorrt_llm.mapping import Mapping

_EXECUTION_UNIT_IMPLS = tuple(sorted(EXECUTION_UNIT_IMPL_CLASSES, key=lambda cls: cls.__name__))


def _model_config(*, skip_create_weights: bool = True) -> ModelConfig:
    return ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        skip_create_weights_in_init=skip_create_weights,
    )


# ---------------------------------------------------------------------------
# Type axis: the nine inherit MoEImplBase; self-contained layers stay on MoE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("moe_cls", _EXECUTION_UNIT_IMPLS, ids=lambda cls: cls.__name__)
def test_execution_unit_inherits_impl_base_not_moe(moe_cls: type) -> None:
    assert issubclass(moe_cls, MoEImplBase)
    assert not issubclass(moe_cls, MoE)


def test_self_contained_layers_stay_on_moe() -> None:
    assert issubclass(TritonFusedMoE, MoE)
    assert not issubclass(TritonFusedMoE, MoEImplBase)
    assert issubclass(ConfigurableMoE, MoE)
    assert not issubclass(ConfigurableMoE, MoEImplBase)


def test_vanilla_moe_is_a_third_shape_the_class_union_must_cover() -> None:
    """The reference path inherits neither base, so two-way unions exclude it.

    It is a ``ModuleList`` of per-expert ``GatedMLP`` / ``MLP`` submodules and
    owns no packed expert tensors, which is also why the weight-owner gate does
    not name it. Annotating a resolver result as ``MoE | MoEImplBase`` would
    silently leave out the one backend used as the accuracy reference.
    """
    assert not issubclass(VanillaMoE, MoE)
    assert not issubclass(VanillaMoE, MoEImplBase)
    assert issubclass(VanillaMoE, torch.nn.ModuleList)
    assert Type[VanillaMoE] in get_args(MoEImplClass)


def test_execution_unit_group_has_exactly_nine_members() -> None:
    assert {cls.__name__ for cls in EXECUTION_UNIT_IMPL_CLASSES} == {
        "CuteDslB12xFusedMoE",
        "CuteDslFusedMoE",
        "CutlassFusedMoE",
        "DeepGemmFusedMoE",
        "DenseGEMMFusedMoE",
        "MarlinFusedMoE",
        "MegaMoECuteDsl",
        "MegaMoEDeepGemm",
        "TRTLLMGenFusedMoE",
    }


# ---------------------------------------------------------------------------
# Standalone construction must fail at the constructor, not at first forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("moe_cls", _EXECUTION_UNIT_IMPLS, ids=lambda cls: cls.__name__)
def test_execution_unit_standalone_construction_fails(moe_cls: type) -> None:
    with pytest.raises(TypeError, match="execution unit"):
        create_moe_backend(
            moe_cls=moe_cls,
            routing_method=RenormalizeMoeRoutingMethod(top_k=1),
            num_experts=8,
            hidden_size=64,
            intermediate_size=128,
            dtype=torch.bfloat16,
            model_config=_model_config(),
        )


def test_standalone_error_names_the_class() -> None:
    assert "CutlassFusedMoE" in STANDALONE_MOE_IMPL_ERROR.format(name="CutlassFusedMoE")


# ---------------------------------------------------------------------------
# Wrapper path: one ``.backend`` child, still a weight owner
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ConfigurableMoE construction touches CUDA streams"
)
def test_wrapper_keeps_a_single_backend_child() -> None:
    moe = ConfigurableMoE(
        moe_cls=CutlassFusedMoE,
        routing_method=RenormalizeMoeRoutingMethod(top_k=1),
        num_experts=8,
        hidden_size=64,
        intermediate_size=128,
        dtype=torch.bfloat16,
        model_config=_model_config(),
    )

    assert moe.backend is not None
    assert type(moe.backend) is CutlassFusedMoE
    assert isinstance(moe.backend, MoEImplBase)
    assert not isinstance(moe.backend, MoE)
    assert [name for name, _ in moe.named_children() if name == "backend"] == ["backend"]
    assert is_moe_weight_owner(moe.backend)


@pytest.mark.parametrize("moe_cls", _EXECUTION_UNIT_IMPLS, ids=lambda cls: cls.__name__)
def test_backend_factory_still_constructs_an_impl_when_not_a_layer(moe_cls: type) -> None:
    """Every unit, not just Cutlass: the success path is where a member lost in
    the base swap actually bites. Covering one class here is what let a
    constructor reading a now-missing ``MoE`` property stay green.
    """
    backend = create_moe_backend(
        moe_cls=moe_cls,
        routing_method=RenormalizeMoeRoutingMethod(top_k=1),
        num_experts=8,
        hidden_size=64,
        intermediate_size=128,
        dtype=torch.bfloat16,
        model_config=_model_config(),
        init_load_balancer=False,
    )

    assert isinstance(backend, moe_cls)
    assert isinstance(backend, MoEImplBase)
    assert not isinstance(backend, MoE)
    assert not hasattr(backend, "forward_impl")
    assert is_moe_weight_owner(backend)


# ---------------------------------------------------------------------------
# Contract completeness: the general defense, not one member at a time
# ---------------------------------------------------------------------------

# Everything the wrapper, the scheduler, or the custom op's ``register_fake``
# calls on ``moe.backend``. An execution unit that cannot answer one of these
# does not fail at construction -- it fails at the first call that asks, which
# may be a rarely-exercised communication or tracing path.
BACKEND_CALLED_MEMBERS = (
    "capabilities",
    "create_weights",
    "forward_fake",
    "get_workspaces",
    "input_requirement",
    "load_weights",
    "post_load_weights",
    "pre_reload_weights",
    "process_weights_after_loading",
    "quantize_input",
    "run_moe",
    "scheduler_kind",
    "supports_moe_output_in_alltoall_workspace",
    "validate_configurable_moe",
    "_supports_load_balancer",
)

# Complete-layer members an execution unit must NOT acquire. ``enable_alltoall``
# is here because the answer depends on the communication strategy, which the
# wrapper owns: a backend that could read it would be reading a default that is
# never the real answer.
LAYER_ONLY_MEMBERS = (
    "forward_impl",
    "_register_layer",
    "enable_alltoall",
    "reducescatter_or_allreduce",
)


@pytest.mark.parametrize("moe_cls", _EXECUTION_UNIT_IMPLS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("member", BACKEND_CALLED_MEMBERS)
def test_execution_unit_answers_everything_called_on_a_backend(moe_cls: type, member: str) -> None:
    assert hasattr(moe_cls, member), (
        f"{moe_cls.__name__} lost {member!r}; the scheduler/wrapper calls it on "
        f"moe.backend, so this is an AttributeError at run time"
    )


@pytest.mark.parametrize("moe_cls", _EXECUTION_UNIT_IMPLS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("member", LAYER_ONLY_MEMBERS)
def test_execution_unit_does_not_acquire_layer_members(moe_cls: type, member: str) -> None:
    assert not hasattr(moe_cls, member), (
        f"{moe_cls.__name__} has layer-only {member!r}; an execution unit that "
        f"looks like a layer can be mistaken for one"
    )


@pytest.mark.parametrize("moe_cls", _EXECUTION_UNIT_IMPLS, ids=lambda cls: cls.__name__)
def test_execution_unit_leaves_forward_unimplemented(moe_cls: type) -> None:
    """``hasattr(cls, "forward")`` cannot express this: every ``nn.Module``
    carries ``_forward_unimplemented``. What matters is that nobody overrode it.
    """
    assert moe_cls.forward is torch.nn.Module.forward, (
        f"{moe_cls.__name__} overrides forward, so it can be called as a layer"
    )


# ---------------------------------------------------------------------------
# Model-side consumers: a model may extend the wrapper, never an execution unit
# ---------------------------------------------------------------------------


def test_model_side_moe_subclasses_extend_the_wrapper() -> None:
    """Subclassing an execution unit is the failure this swap introduces.

    The subclass inherits a constructor that rejects the standalone path and a
    ``forward`` that was never defined, so it dies at model build -- on a path
    (``enable_min_latency``) that no GPU CI stage covers.
    """
    from tensorrt_llm._torch.models.modeling_llama_min_latency import Llama4MinLatencyFusedMoE

    assert issubclass(Llama4MinLatencyFusedMoE, ConfigurableMoE)
    assert not issubclass(Llama4MinLatencyFusedMoE, MoEImplBase)
    assert Llama4MinLatencyFusedMoE.forward is not torch.nn.Module.forward


def test_shared_defaults_have_a_single_definition() -> None:
    """A default both bases need lives on the mixin, not once per base.

    Hand-copying it into each base is what silently dropped members during the
    swap: the copy in ``MoEImplBase`` omitted two that ``MoE`` had.
    """
    for member in (
        "capabilities",
        "input_requirement",
        "supports_moe_output_in_alltoall_workspace",
        "validate_configurable_moe",
        "forward_fake",
        "_supports_non_divisible_ep",
    ):
        assert member not in vars(MoE), f"{member!r} is defined on MoE as well as the shared mixin"
        assert member not in vars(MoEImplBase), (
            f"{member!r} is defined on MoEImplBase as well as the shared mixin"
        )
        assert hasattr(MoE, member) and hasattr(MoEImplBase, member)
