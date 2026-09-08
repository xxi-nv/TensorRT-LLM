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

from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.moe.fused_moe.activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivationSupport,
)
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import _BACKEND_SYNC_ATTRS, ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEInputRequirement
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Every test here drives ``create_weights`` with construction stubbed out, so
# nothing allocates. The marker is also what makes the file reachable: the CPU
# stage lists this directory but collects only files that carry it.
pytestmark = pytest.mark.cpu_only


def _wrapper() -> ConfigurableMoE:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.num_experts = 8
    # Divides ``num_experts`` so ``_reject_non_divisible_ep_backend`` returns at
    # the divisibility check. It must not fall past it: the branch below reads
    # ``type(self.backend)._supports_non_divisible_ep``, and the backend here is
    # a ``Mock`` *instance*, so that lookup lands on the ``Mock`` class and
    # raises. Assigned because ``__new__`` skipped the ``MoE.__init__`` that
    # normally derives it from the mapping.
    wrapper.ep_size = 1
    wrapper.hidden_size = 16
    wrapper.intermediate_size = 32
    wrapper.dtype = torch.bfloat16
    wrapper.reduce_results = False
    wrapper.aux_stream_dict = None
    wrapper.weight_loading_mode = None
    wrapper.apply_router_weight_on_input = False
    # Both are read while the backend is being built: the carrier goes to
    # ``resolve_moe_cls`` and to ``install_activation_params``, the kind to
    # ``infer_swiglu_gptoss_style``. ``__new__`` skipped the ``__init__`` that
    # normally assigns them, so a fixture that sets neither fails before
    # reaching the quant-config behaviour these tests are about.
    wrapper.activation = DEFAULT_MOE_ACTIVATION
    wrapper.activation_type = ActivationType(DEFAULT_MOE_ACTIVATION.kind)
    wrapper.routing_method = Mock()
    wrapper.bias = False
    # The binding state ``__init__`` sets up and ``create_weights`` consumes.
    # ``__new__`` skipped that, so a fixture without these would take the
    # already-bound branch against a backend that does not exist.
    #
    # ``_bind_backend`` pins the class, once the stub it pins to exists.
    wrapper._pinned_moe_cls = None
    wrapper._allow_backend_degradation = True
    wrapper._backend_bound = False
    wrapper._bound_quant_key = None
    # Read by a rebind that replaces the backend, which releases the old comm
    # strategy and refuses to move a layer already claimed for DWDP.
    wrapper.comm = None
    wrapper.dwdp_manager = None
    wrapper.enable_dwdp = False
    for attr in _BACKEND_SYNC_ATTRS:
        setattr(wrapper, attr, None)
    return wrapper


def _backend_mock(routing_scales_dtype: torch.dtype | None = None) -> Mock:
    """A ``Mock`` backend that survives ``install_activation_params``.

    That call cannot be patched out here: ``ConfigurableMoE`` makes it on the
    backend after the EPLB sync *and* inside ``create_weights``, which is the
    code path under test. A bare ``Mock`` slips past
    ``resolve_activation_support`` -- every attribute of a Mock is callable, so
    the override branch is taken -- and then fails inside
    ``materialize_activation_params``, which uses the returned Mock as a real
    declaration. Declaring a real one keeps the failure surface at the quant
    config. Still GPU-free: ``DEFAULT_MOE_ACTIVATION`` carries no constants, so
    every register short-circuits to None without allocating.
    """
    backend = Mock()
    backend.activation = DEFAULT_MOE_ACTIVATION
    backend.resolve_activation_support = Mock(
        return_value=MoEActivationSupport(
            kinds=frozenset({ActivationType.Swiglu}),
            alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
            limit=ActivationParamShape.PER_EXPERT_TENSOR,
        )
    )
    # Declared rather than left to auto-Mock so binding reads a real dtype (or
    # a real ``None``) when it offers routing the backend's scale precision.
    backend.input_requirement = MoEInputRequirement(routing_scales_dtype=routing_scales_dtype)
    return backend


def _bind_backend(
    wrapper: ConfigurableMoE,
    model_config: ModelConfig,
    routing_scales_dtype: torch.dtype | None = None,
) -> Mock:
    """Run the real binding path with construction and its dependents stubbed.

    ``create_weights`` is the entry point under test, so it is driven rather
    than bypassed; only the two things it delegates that need a GPU -- building
    a real implementation, and the comm strategy / scheduler built from its
    class -- are replaced.
    """
    backend = _backend_mock(routing_scales_dtype)
    wrapper.model_config = model_config
    # A rebind compares the resolved class against ``type(self.backend)``, so
    # the pin has to be that exact class for a format change to stay a
    # same-class rebind. It cannot be spelled ``Mock``: ``NonCallableMock``
    # gives every instance its own dynamically built subclass, all named
    # "Mock", so ``type(Mock()) is not Mock`` while both repr identically.
    wrapper._pinned_moe_cls = type(backend)
    with (
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.create_moe_backend",
            return_value=backend,
        ),
        patch.object(ConfigurableMoE, "_register_dwdp"),
        patch.object(ConfigurableMoE, "_install_backend_dependents"),
    ):
        wrapper.create_weights()
    return backend


def test_layerwise_quant_config_is_applied_before_weight_creation() -> None:
    global_config = QuantConfig()
    layer_config = QuantConfig()
    model_config = ModelConfig(
        quant_config=global_config,
        quant_config_dict={"model.layers.0.mlp.experts": layer_config},
    )
    wrapper = _wrapper()
    wrapper.quant_config = global_config

    # What ``__post_init__`` does between construction and weight creation.
    wrapper.quant_config = layer_config
    backend = _bind_backend(wrapper, model_config)

    assert backend.quant_config is layer_config
    backend.create_weights.assert_called_once_with()


def test_exclusions_only_recreate_matching_moe_weights() -> None:
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        exclude_modules=["*kv_b_proj*", "*k_b_proj*", "*eh_proj"],
    )
    model_config = ModelConfig(quant_config=quant_config)
    wrapper = _wrapper()
    wrapper.quant_config = quant_config

    backend = _bind_backend(wrapper, model_config)

    assert backend.quant_config is quant_config
    backend.create_weights.assert_called_once_with()
    backend.create_weights.reset_mock()
    backend._weights_created = True

    root = torch.nn.Module()
    root.model_config = ModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8,
            exclude_modules=["experts"],
        )
    )
    root.experts = wrapper

    DecoderModelForCausalLM.apply_quant_config_exclude_modules(root)

    assert not backend._weights_created
    # The pin keeps the class fixed, so the exclusion is a same-class rebind:
    # the backend object stays and adopts the stripped config.
    wrapper.create_weights()
    assert wrapper.quant_config.quant_algo is None
    assert backend.quant_config.quant_algo is None
    backend.create_weights.assert_called_once_with()


def test_exclusion_outranks_an_explicit_override() -> None:
    """An override is a starting value, not an authority over the final format.

    It used to outrank ``self.quant_config`` at every ``create_weights``, which
    meant exclusions never reached a layer whose caller had looked the layerwise
    entry up itself -- and every caller that passes an override does exactly
    that. Pinned here because the priority is the reason exclusions work at all
    for those models, and nothing else in the suite would notice it flipping
    back.
    """
    override = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, exclude_modules=["experts"])
    )
    wrapper = _wrapper()
    wrapper.quant_config = override

    root = torch.nn.Module()
    root.model_config = model_config
    root.experts = wrapper
    DecoderModelForCausalLM.apply_quant_config_exclude_modules(root)

    backend = _bind_backend(wrapper, model_config)

    assert wrapper.quant_config is not override
    assert backend.quant_config.quant_algo is None


def test_replacing_the_backend_releases_the_old_comm_and_defers_dependents() -> None:
    """A swap has to keep the first bind's order and not leak the old strategy.

    DeepEP's buffer enters a collective barrier from ``__del__``, so dropping
    the reference and letting GC pick the moment lets ranks reach that barrier
    at different times. The comm strategy and the scheduler are also written
    against a backend whose weights exist, which the first bind guarantees by
    building them last.
    """
    first, second = _backend_mock(), _backend_mock()
    old_comm = Mock()
    events = []
    second.create_weights.side_effect = lambda: events.append("weights")

    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))
    wrapper = _wrapper()
    wrapper.model_config = model_config
    wrapper.quant_config = model_config.quant_config

    with (
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.create_moe_backend",
            side_effect=[first, second],
        ),
        # Each ``Mock`` instance gets its own dynamically built subclass, so
        # these two are distinct classes and the second resolution reads as a
        # move rather than a same-class rebind.
        patch.object(
            ConfigurableMoE,
            "_resolve_backend_cls",
            side_effect=[type(first), type(second)],
        ),
        patch.object(ConfigurableMoE, "_register_dwdp") as register_dwdp,
        patch.object(
            ConfigurableMoE,
            "_install_backend_dependents",
            side_effect=lambda: events.append("dependents"),
        ),
    ):
        wrapper.create_weights()
        # What ``_install_backend_dependents`` would have built, had it run for
        # real: the strategy this layer holds when the format moves.
        wrapper.comm = old_comm
        events.clear()
        wrapper.quant_config = QuantConfig(quant_algo=None)
        wrapper.create_weights()

    assert wrapper.backend is second
    old_comm.destroy.assert_called_once_with()
    assert events == ["weights", "dependents"]
    # Keyed on the layer, not on the backend: the manager has no way to drop a
    # claim, so a second claim would count this layer twice.
    register_dwdp.assert_called_once_with()


def test_a_dwdp_claimed_layer_refuses_a_backend_that_cannot_honor_it() -> None:
    first, second = _backend_mock(), _backend_mock()
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))
    wrapper = _wrapper()
    wrapper.model_config = model_config
    wrapper.quant_config = model_config.quant_config

    with (
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.create_moe_backend",
            side_effect=[first, second],
        ),
        patch.object(
            ConfigurableMoE,
            "_resolve_backend_cls",
            side_effect=[type(first), type(second)],
        ),
        patch.object(ConfigurableMoE, "_register_dwdp"),
        patch.object(ConfigurableMoE, "_install_backend_dependents"),
        patch.object(ConfigurableMoE, "_should_enable_dwdp", return_value=False),
    ):
        wrapper.create_weights()
        # What ``_register_dwdp`` would have left behind, had it run for real
        # against a backend that supports DWDP.
        wrapper.enable_dwdp = True
        wrapper.quant_config = QuantConfig(quant_algo=None)
        with pytest.raises(ValueError, match="claimed for distributed weight sharing"):
            wrapper.create_weights()


def test_routing_adopts_the_bound_backends_scale_precision() -> None:
    """Binding is what lets routing emit the narrower dtype in the first place.

    ``MoEScheduler`` converts to it either way, so leaving routing at float32
    only buys an extra elementwise launch per layer per forward. Which dtype to
    ask for belongs to the leaf, and the leaf is not known until the format is,
    so this cannot move back into the gate.
    """
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))
    wrapper = _wrapper()
    wrapper.quant_config = model_config.quant_config
    wrapper.routing_method.output_dtype = torch.float32

    _bind_backend(wrapper, model_config, routing_scales_dtype=torch.bfloat16)

    assert wrapper.routing_method.output_dtype == torch.bfloat16


def test_routing_already_narrowed_by_the_model_is_left_alone() -> None:
    """Widening cannot recover what a narrower routing method already dropped.

    So the offer only applies to float32, routing's own full precision. What is
    left is a real disagreement between the model and the backend, and the
    scheduler asserts on it -- which is also what keeps a routing method shared
    across layers from being decided by whichever one bound last.
    """
    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))
    wrapper = _wrapper()
    wrapper.quant_config = model_config.quant_config
    wrapper.routing_method.output_dtype = torch.bfloat16

    _bind_backend(wrapper, model_config, routing_scales_dtype=torch.float32)

    assert wrapper.routing_method.output_dtype == torch.bfloat16
