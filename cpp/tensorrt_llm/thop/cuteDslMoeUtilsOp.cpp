/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <limits>

namespace btg = batchedGemm::trtllm::gen;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
// Sort
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType;

std::vector<torch::Tensor> moe_topk_sort_impl(torch::optional<torch::Tensor> const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, torch::optional<torch::Tensor> const& token_selected_experts,
    torch::optional<torch::Tensor> const& token_final_scales, int64_t const num_experts, int64_t const top_k,
    std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const local_expert_offset,
    int64_t const local_num_experts, std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim,
    RoutingMethodType const routing_method_type)
{
    int64_t const num_tokens
        = token_selected_experts.has_value() ? token_selected_experts->size(0) : routing_logits->size(0);
    int64_t const max_num_padded_tokens
        = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxPermutedPaddedCount(
            num_tokens, top_k, local_num_experts, tile_tokens_dim);
    int64_t const max_num_ctas = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxNumCtasInBatchDim(
        num_tokens, top_k, local_num_experts, tile_tokens_dim);
    int64_t const size_of_expert_count_histogram = std::max(num_experts * 2, int64_t(256 * 2));
    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias->scalar_type() : torch::kBFloat16;

    auto routing_logits_ptr = routing_logits.has_value() ? routing_logits->data_ptr() : nullptr;
    auto routing_bias_ptr = routing_bias.has_value() ? routing_bias->data_ptr() : nullptr;
    auto token_selected_experts_ptr
        = token_selected_experts.has_value() ? token_selected_experts->data_ptr<int32_t>() : nullptr;
    auto token_final_scales_ptr = token_final_scales.has_value() ? token_final_scales->data_ptr() : nullptr;

    torch::optional<torch::Tensor> new_token_final_scales;
    if (token_final_scales_ptr == nullptr)
    {
        new_token_final_scales
            = torch::empty({num_tokens, top_k}, torch::dtype(routing_bias_dtype).device(torch::kCUDA));
        token_final_scales_ptr = new_token_final_scales->data_ptr();
    }

    auto expert_indexes = torch::empty({num_tokens, top_k}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto expert_count_histogram
        = torch::empty({size_of_expert_count_histogram}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto total_num_padded_tokens = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto expanded_idx_to_permuted_idx
        = torch::empty({num_tokens, top_k}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto permuted_idx_to_expanded_idx
        = torch::empty({max_num_padded_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_expert = torch::empty({num_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto tile_idx_to_expert_idx = torch::empty({max_num_ctas}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto tile_idx_to_mn_limit = torch::empty({max_num_ctas}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto num_non_exiting_tiles = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::Runner routing_runner(tile_tokens_dim);
    auto const& stream = at::cuda::getCurrentCUDAStream(
        routing_logits.has_value() ? routing_logits->get_device() : token_selected_experts->get_device());
    auto const dtypeRoutingLogits = routing_logits.has_value()
        ? (routing_logits->scalar_type() == at::ScalarType::Float ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16)
        : btg::Dtype::Bfloat16;
    routing_runner.run(routing_logits_ptr, routing_bias_ptr, num_tokens, num_experts, top_k, n_group.value_or(0),
        topk_group.value_or(0), local_expert_offset, local_num_experts, routed_scaling_factor.value_or(1.0),
        expert_indexes.data_ptr<int>(), expert_count_histogram.data_ptr<int>(), total_num_padded_tokens.data_ptr<int>(),
        expanded_idx_to_permuted_idx.data_ptr<int>(), permuted_idx_to_expanded_idx.data_ptr<int>(),
        nullptr /*permuted_idx_to_token_idx.data_ptr<int>()*/, token_final_scales_ptr, token_selected_experts_ptr,
        num_tokens_per_expert.data_ptr<int>(), tile_idx_to_expert_idx.data_ptr<int>(),
        tile_idx_to_mn_limit.data_ptr<int>(), num_non_exiting_tiles.data_ptr<int>(),
        batchedGemm::trtllm::gen::Dtype::Void /* dtypeElt */, false /* use_routing_scales_on_input */,
        false /* use_deep_seek_fp8 */, routing_method_type, stream, dtypeRoutingLogits);

    std::vector<torch::Tensor> results{tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles};
    if (new_token_final_scales.has_value())
    {
        results.push_back(new_token_final_scales.value());
    }
    return results;
}

std::vector<torch::Tensor> moe_topk_sort(torch::Tensor const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, int64_t const num_experts, int64_t const top_k,
    std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const local_expert_offset,
    int64_t const local_num_experts, std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim,
    int64_t const routing_method_type)
{
    TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be 2D.");
    TORCH_CHECK(routing_logits.size(1) == num_experts, "routing_logits.size(1) must be num_experts.");
    if (routing_bias.has_value())
    {
        TORCH_CHECK(routing_bias->dim() == 1, "routing_bias must be 1D.");
        TORCH_CHECK(routing_bias->size(0) == num_experts, "routing_bias.size(0) must be num_experts.");
    }
    return moe_topk_sort_impl(routing_logits, routing_bias, std::nullopt, std::nullopt, num_experts, top_k, n_group,
        topk_group, local_expert_offset, local_num_experts, routed_scaling_factor, tile_tokens_dim,
        static_cast<RoutingMethodType>(routing_method_type));
}

std::vector<torch::Tensor> moe_sort(torch::Tensor const& token_selected_experts,
    torch::Tensor const& token_final_scales, int64_t const num_experts, int64_t const top_k,
    int64_t const local_expert_offset, int64_t const local_num_experts, int64_t const tile_tokens_dim)
{
    TORCH_CHECK(token_selected_experts.dim() == 2, "token_selected_experts must be 2D.");
    int64_t const num_tokens = token_selected_experts.size(0);
    TORCH_CHECK(token_selected_experts.size(1) == top_k, "token_selected_experts.size(1) must be top_k.");
    TORCH_CHECK(token_final_scales.dim() == 2, "token_final_scales must be 2D.");
    TORCH_CHECK(token_final_scales.size(0) == num_tokens, "token_final_scales.size(0) must be num_tokens.");
    TORCH_CHECK(token_final_scales.size(1) == top_k, "token_final_scales.size(1) must be top_k.");
    return moe_topk_sort_impl(std::nullopt, std::nullopt, token_selected_experts, token_final_scales, num_experts,
        top_k, 1, 1, local_expert_offset, local_num_experts, std::nullopt, tile_tokens_dim,
        RoutingMethodType::DeepSeekV3);
}

// Permute

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> moe_permute(torch::Tensor const& input,
    torch::optional<torch::Tensor> const& input_sf, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& permuted_idx_to_expanded_idx, torch::Tensor const& num_non_exiting_tiles,
    int64_t const tile_tokens_dim, int64_t const top_k)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    int64_t const num_tokens = input.size(0);
    int64_t const hidden_size = input.scalar_type() == torch::kFloat4_e2m1fn_x2 ? input.size(1) * 2 : input.size(1);

    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const num_tiles = tile_idx_to_mn_limit.size(0);
    TORCH_CHECK(permuted_idx_to_expanded_idx.dim() == 1, "permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(
        permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32, "permuted_idx_to_expanded_idx must be int32.");
    int64_t const max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0);
    TORCH_CHECK(max_num_permuted_tokens == tile_tokens_dim * num_tiles,
        "max_num_permuted_tokens must be equal to tile_tokens_dim * num_tiles.");
    TORCH_CHECK(max_num_permuted_tokens >= num_tokens * top_k,
        "max_num_permuted_tokens must be greater than or equal to num_tokens * top_k.");

    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    auto permuted_output = torch::empty(
        {max_num_permuted_tokens, input.size(1)}, torch::dtype(input.scalar_type()).device(torch::kCUDA));

    void* input_sf_ptr = nullptr;
    void* permuted_sf_ptr = nullptr;
    torch::optional<torch::Tensor> permuted_sf;
    if (input.scalar_type() == torch::kFloat4_e2m1fn_x2)
    {
        TORCH_CHECK(input_sf.has_value(), "input_sf is required for NVFP4.");
        input_sf_ptr = input_sf->data_ptr();
        int64_t constexpr kSFVecSize = 16;
        permuted_sf = torch::empty({max_num_permuted_tokens * hidden_size / kSFVecSize},
            torch::dtype(input_sf->scalar_type()).device(torch::kCUDA));
        permuted_sf_ptr = permuted_sf->data_ptr();
    }

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define DISPATCH_MOE_PERMUTE(InputType, SFType)                                                                        \
    tensorrt_llm::kernels::cute_dsl::moePermute<InputType, SFType>(static_cast<InputType*>(input.data_ptr()),          \
        static_cast<InputType*>(permuted_output.data_ptr()), static_cast<SFType*>(input_sf_ptr),                       \
        static_cast<SFType*>(permuted_sf_ptr), tile_idx_to_mn_limit.data_ptr<int32_t>(),                               \
        permuted_idx_to_expanded_idx.data_ptr<int32_t>(), num_non_exiting_tiles.data_ptr<int32_t>(),                   \
        max_num_permuted_tokens, hidden_size, top_k, tile_tokens_dim, stream)

    if (input.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_PERMUTE(half, uint8_t);
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_PERMUTE(__nv_bfloat16, uint8_t);
    }
    else if (input.scalar_type() == torch::kFloat8_e4m3fn)
    {
        DISPATCH_MOE_PERMUTE(__nv_fp8_e4m3, uint8_t);
    }
    else if (input.scalar_type() == torch::kFloat4_e2m1fn_x2)
    {
        DISPATCH_MOE_PERMUTE(__nv_fp4_e2m1, uint8_t);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }

#undef DISPATCH_MOE_PERMUTE

    return {permuted_output, permuted_sf};
}

// Unpermute

void moe_unpermute_inplace(torch::Tensor const& permuted_input, torch::Tensor const& output,
    torch::Tensor const& expanded_idx_to_permuted_idx, torch::Tensor const& topk_scales)
{
    TORCH_CHECK(permuted_input.dim() == 2, "permuted_input must be 2D.");
    int64_t const max_num_permuted_tokens = permuted_input.size(0);
    int64_t const hidden_size = permuted_input.size(1);
    TORCH_CHECK(output.dim() == 2, "output must be 2D.");
    int64_t const num_tokens = output.size(0);
    TORCH_CHECK(output.size(1) == hidden_size, "output.size(1) must be hidden_size.");

    TORCH_CHECK(expanded_idx_to_permuted_idx.dim() == 2, "expanded_idx_to_permuted_idx must be 2D.");
    TORCH_CHECK(
        expanded_idx_to_permuted_idx.size(0) == num_tokens, "expanded_idx_to_permuted_idx.size(0) must be num_tokens.");
    int64_t const top_k = expanded_idx_to_permuted_idx.size(1);
    TORCH_CHECK(topk_scales.dim() == 2, "topk_scales must be 2D.");
    TORCH_CHECK(topk_scales.size(0) == num_tokens, "topk_scales.size(0) must be num_tokens.");
    TORCH_CHECK(topk_scales.size(1) == top_k, "topk_scales.size(1) must be top_k.");
    TORCH_CHECK(max_num_permuted_tokens >= num_tokens * top_k,
        "max_num_permuted_tokens must be greater than or equal to num_tokens * top_k.");

    auto const& stream = at::cuda::getCurrentCUDAStream(permuted_input.get_device());

#define DISPATCH_MOE_UNPERMUTE(InputType, TopKScaleType)                                                               \
    tensorrt_llm::kernels::cute_dsl::moeUnpermute<InputType>(static_cast<InputType*>(permuted_input.data_ptr()),       \
        static_cast<InputType*>(output.data_ptr()), expanded_idx_to_permuted_idx.data_ptr<int32_t>(),                  \
        static_cast<TopKScaleType*>(topk_scales.data_ptr()), num_tokens, hidden_size, top_k, stream)

    if (permuted_input.scalar_type() == torch::kHalf && topk_scales.scalar_type() == torch::kFloat)
    {
        DISPATCH_MOE_UNPERMUTE(half, float);
    }
    else if (permuted_input.scalar_type() == torch::kHalf && topk_scales.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_UNPERMUTE(half, half);
    }
    else if (permuted_input.scalar_type() == torch::kBFloat16 && topk_scales.scalar_type() == torch::kFloat)
    {
        DISPATCH_MOE_UNPERMUTE(__nv_bfloat16, float);
    }
    else if (permuted_input.scalar_type() == torch::kBFloat16 && topk_scales.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_UNPERMUTE(__nv_bfloat16, __nv_bfloat16);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", permuted_input.scalar_type(),
            " and/or topk_scales dtype: ", topk_scales.scalar_type());
    }

#undef DISPATCH_MOE_UNPERMUTE
}

torch::Tensor moe_unpermute(torch::Tensor const& permuted_input, torch::Tensor const& expanded_idx_to_permuted_idx,
    torch::Tensor const& topk_scales)
{
    TORCH_CHECK(permuted_input.dim() == 2, "permuted_input must be 2D.");
    int64_t const hidden_size = permuted_input.size(1);
    TORCH_CHECK(expanded_idx_to_permuted_idx.dim() == 2, "expanded_idx_to_permuted_idx must be 2D.");
    int64_t const num_tokens = expanded_idx_to_permuted_idx.size(0);

    auto output
        = torch::empty({num_tokens, hidden_size}, torch::dtype(permuted_input.scalar_type()).device(torch::kCUDA));
    moe_unpermute_inplace(permuted_input, output, expanded_idx_to_permuted_idx, topk_scales);
    return output;
}

void moe_output_memset_inplace(torch::Tensor const& input, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& expanded_idx_to_permuted_idx, torch::Tensor const& permuted_idx_to_expanded_idx,
    torch::Tensor const& num_non_exiting_tiles, int64_t const tile_tokens_dim, int64_t const top_k,
    int64_t const ep_size, bool const enable_alltoall = false)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    int64_t const num_tokens = input.size(0);
    int64_t const hidden_size = input.size(1);
    TORCH_CHECK(expanded_idx_to_permuted_idx.dim() == 2, "expanded_idx_to_permuted_idx must be 2D.");
    TORCH_CHECK(
        expanded_idx_to_permuted_idx.scalar_type() == torch::kInt32, "expanded_idx_to_permuted_idx must be int32.");
    TORCH_CHECK(
        expanded_idx_to_permuted_idx.size(0) == num_tokens, "expanded_idx_to_permuted_idx.size(0) must be num_tokens.");
    TORCH_CHECK(expanded_idx_to_permuted_idx.size(1) == top_k, "expanded_idx_to_permuted_idx.size(1) must be top_k.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const num_tiles = tile_idx_to_mn_limit.size(0);
    TORCH_CHECK(permuted_idx_to_expanded_idx.dim() == 1, "permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(
        permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32, "permuted_idx_to_expanded_idx must be int32.");
    int64_t const max_num_permuted_tokens = permuted_idx_to_expanded_idx.size(0);
    TORCH_CHECK(max_num_permuted_tokens == tile_tokens_dim * num_tiles,
        "max_num_permuted_tokens must be equal to tile_tokens_dim * num_tiles.");
    TORCH_CHECK(max_num_permuted_tokens >= num_tokens * top_k,
        "max_num_permuted_tokens must be greater than or equal to num_tokens * top_k.");

    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define DISPATCH_MOE_OUTPUT_MEMSET(InputType)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!enable_alltoall || ep_size <= top_k)                                                                      \
        {                                                                                                              \
            cudaMemsetAsync(input.data_ptr(), 0x0, sizeof(InputType) * num_tokens * hidden_size, stream);              \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            tensorrt_llm::kernels::cute_dsl::moeOutputMemset<InputType>(static_cast<InputType*>(input.data_ptr()),     \
                tile_idx_to_mn_limit.data_ptr<int32_t>(), expanded_idx_to_permuted_idx.data_ptr<int32_t>(),            \
                permuted_idx_to_expanded_idx.data_ptr<int32_t>(), num_non_exiting_tiles.data_ptr<int32_t>(),           \
                max_num_permuted_tokens, hidden_size, top_k, tile_tokens_dim, stream);                                 \
        }                                                                                                              \
    } while (0)

    if (input.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_OUTPUT_MEMSET(half);
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_OUTPUT_MEMSET(__nv_bfloat16);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }

#undef DISPATCH_MOE_OUTPUT_MEMSET
}

void mega_moe_m5_materialize_from_moe_sort(torch::Tensor const& input, torch::Tensor const& input_sf,
    torch::Tensor const& topk_scales, torch::Tensor const& token_offsets, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& permuted_idx_to_expanded_idx, torch::Tensor const& num_non_exiting_tiles,
    torch::Tensor const& l1_acts_pool, torch::Tensor const& l1_acts_sf_pool, torch::Tensor const& l1_topk_weights_pool,
    torch::Tensor const& token_src_metadata, torch::Tensor const& l1_arrival_count, int64_t const tile_tokens_dim)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor.");
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8.");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");
    int64_t const total_tokens = input.size(0);
    int64_t const hidden_packed_size = input.size(1);

    TORCH_CHECK(input_sf.is_cuda(), "input_sf must be a CUDA tensor.");
    TORCH_CHECK(input_sf.dim() == 2, "input_sf must be 2D.");
    TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8, "input_sf must be uint8.");
    TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous.");
    TORCH_CHECK(input_sf.size(0) == total_tokens, "input_sf.size(0) must match input.size(0).");
    int64_t const sf_hidden_size = input_sf.size(1);

    TORCH_CHECK(topk_scales.is_cuda(), "topk_scales must be a CUDA tensor.");
    TORCH_CHECK(topk_scales.dim() == 2, "topk_scales must be 2D.");
    TORCH_CHECK(topk_scales.scalar_type() == torch::kFloat32, "topk_scales must be float32.");
    TORCH_CHECK(topk_scales.is_contiguous(), "topk_scales must be contiguous.");
    TORCH_CHECK(topk_scales.size(0) == total_tokens, "topk_scales.size(0) must match input.size(0).");
    int64_t const top_k = topk_scales.size(1);

    TORCH_CHECK(token_offsets.is_cuda(), "token_offsets must be a CUDA tensor.");
    TORCH_CHECK(token_offsets.dim() == 1, "token_offsets must be 1D.");
    TORCH_CHECK(token_offsets.scalar_type() == torch::kInt32, "token_offsets must be int32.");
    TORCH_CHECK(token_offsets.is_contiguous(), "token_offsets must be contiguous.");
    TORCH_CHECK(token_offsets.numel() >= 2, "token_offsets must contain ep_size + 1 entries.");
    int64_t const ep_size = token_offsets.numel() - 1;

    TORCH_CHECK(tile_idx_to_mn_limit.is_cuda(), "tile_idx_to_mn_limit must be a CUDA tensor.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    TORCH_CHECK(permuted_idx_to_expanded_idx.is_cuda(), "permuted_idx_to_expanded_idx must be a CUDA tensor.");
    TORCH_CHECK(permuted_idx_to_expanded_idx.dim() == 1, "permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(
        permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32, "permuted_idx_to_expanded_idx must be int32.");
    TORCH_CHECK(num_non_exiting_tiles.is_cuda(), "num_non_exiting_tiles must be a CUDA tensor.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    TORCH_CHECK(l1_acts_pool.is_cuda(), "l1_acts_pool must be a CUDA tensor.");
    TORCH_CHECK(l1_acts_pool.dim() == 2, "l1_acts_pool must be 2D.");
    TORCH_CHECK(l1_acts_pool.scalar_type() == torch::kUInt8, "l1_acts_pool must be uint8.");
    TORCH_CHECK(l1_acts_pool.is_contiguous(), "l1_acts_pool must be contiguous.");
    TORCH_CHECK(
        l1_acts_pool.size(1) == hidden_packed_size, "l1_acts_pool.size(1) must match input hidden packed size.");
    TORCH_CHECK(l1_acts_sf_pool.is_cuda(), "l1_acts_sf_pool must be a CUDA tensor.");
    TORCH_CHECK(l1_acts_sf_pool.dim() == 2, "l1_acts_sf_pool must be 2D.");
    TORCH_CHECK(l1_acts_sf_pool.scalar_type() == torch::kUInt8, "l1_acts_sf_pool must be uint8.");
    TORCH_CHECK(l1_acts_sf_pool.is_contiguous(), "l1_acts_sf_pool must be contiguous.");
    TORCH_CHECK(l1_acts_sf_pool.size(1) == sf_hidden_size, "l1_acts_sf_pool.size(1) must match input_sf hidden size.");

    TORCH_CHECK(l1_topk_weights_pool.is_cuda(), "l1_topk_weights_pool must be a CUDA tensor.");
    TORCH_CHECK(l1_topk_weights_pool.dim() == 1, "l1_topk_weights_pool must be 1D.");
    TORCH_CHECK(l1_topk_weights_pool.scalar_type() == torch::kFloat32, "l1_topk_weights_pool must be float32.");
    TORCH_CHECK(token_src_metadata.is_cuda(), "token_src_metadata must be a CUDA tensor.");
    TORCH_CHECK(token_src_metadata.dim() == 2, "token_src_metadata must be 2D.");
    TORCH_CHECK(token_src_metadata.size(1) == 3, "token_src_metadata.size(1) must be 3.");
    TORCH_CHECK(token_src_metadata.scalar_type() == torch::kInt32, "token_src_metadata must be int32.");
    TORCH_CHECK(l1_arrival_count.is_cuda(), "l1_arrival_count must be a CUDA tensor.");
    TORCH_CHECK(l1_arrival_count.dim() == 1, "l1_arrival_count must be 1D.");
    TORCH_CHECK(l1_arrival_count.scalar_type() == torch::kInt32, "l1_arrival_count must be int32.");

    int64_t num_available_pool_slots = std::min(permuted_idx_to_expanded_idx.numel(), l1_acts_pool.size(0));
    num_available_pool_slots = std::min(num_available_pool_slots, tile_idx_to_mn_limit.numel() * tile_tokens_dim);
    TORCH_CHECK(l1_topk_weights_pool.numel() >= num_available_pool_slots,
        "l1_topk_weights_pool must cover available pool slots.");
    TORCH_CHECK(
        token_src_metadata.size(0) >= num_available_pool_slots, "token_src_metadata must cover available pool slots.");

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM5MaterializeFromMoeSort(input.data_ptr<uint8_t>(),
        input_sf.data_ptr<uint8_t>(), topk_scales.data_ptr<float>(), token_offsets.data_ptr<int32_t>(),
        tile_idx_to_mn_limit.data_ptr<int32_t>(), permuted_idx_to_expanded_idx.data_ptr<int32_t>(),
        num_non_exiting_tiles.data_ptr<int32_t>(), l1_acts_pool.data_ptr<uint8_t>(),
        l1_acts_sf_pool.data_ptr<uint8_t>(), l1_topk_weights_pool.data_ptr<float>(),
        token_src_metadata.data_ptr<int32_t>(), l1_arrival_count.data_ptr<int32_t>(),
        static_cast<int32_t>(total_tokens), static_cast<int32_t>(ep_size), static_cast<int32_t>(hidden_packed_size),
        static_cast<int32_t>(sf_hidden_size), static_cast<int32_t>(top_k), static_cast<int32_t>(tile_tokens_dim),
        static_cast<int32_t>(num_available_pool_slots), static_cast<int32_t>(l1_acts_sf_pool.size(0)),
        static_cast<int32_t>(l1_arrival_count.numel()), stream);
}

void mega_moe_m5_materialize_direct_from_moe_sort(torch::Tensor const& input, torch::Tensor const& input_sf,
    torch::Tensor const& topk_scales, torch::Tensor const& token_offsets, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& permuted_idx_to_expanded_idx, torch::Tensor const& num_non_exiting_tiles,
    torch::Tensor const& l1_acts_pool, torch::Tensor const& l1_acts_sf_pool, torch::Tensor const& l1_topk_weights_pool,
    torch::Tensor const& token_src_metadata, torch::Tensor const& l1_arrival_count,
    torch::Tensor const& active_pool_slots, torch::Tensor const& active_combine_rows,
    torch::Tensor const& active_route_count, torch::Tensor const& output_permuted_idx_to_expanded_idx,
    torch::Tensor const& output_token_final_scales, int64_t const tile_tokens_dim,
    int64_t const max_num_tokens_per_rank, int64_t const combine_layout_rows)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor.");
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8.");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");
    int64_t const total_tokens = input.size(0);
    int64_t const hidden_packed_size = input.size(1);
    int64_t const input_device = input.get_device();

    TORCH_CHECK(input_sf.is_cuda(), "input_sf must be a CUDA tensor.");
    TORCH_CHECK(input_sf.get_device() == input_device, "input_sf must be on the same CUDA device as input.");
    TORCH_CHECK(input_sf.dim() == 2, "input_sf must be 2D.");
    TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8, "input_sf must be uint8.");
    TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous.");
    TORCH_CHECK(input_sf.size(0) == total_tokens, "input_sf.size(0) must match input.size(0).");
    int64_t const sf_hidden_size = input_sf.size(1);

    TORCH_CHECK(topk_scales.is_cuda(), "topk_scales must be a CUDA tensor.");
    TORCH_CHECK(topk_scales.get_device() == input_device, "topk_scales must be on the same CUDA device as input.");
    TORCH_CHECK(topk_scales.dim() == 2, "topk_scales must be 2D.");
    TORCH_CHECK(topk_scales.scalar_type() == torch::kFloat32, "topk_scales must be float32.");
    TORCH_CHECK(topk_scales.is_contiguous(), "topk_scales must be contiguous.");
    TORCH_CHECK(topk_scales.size(0) == total_tokens, "topk_scales.size(0) must match input.size(0).");
    int64_t const top_k = topk_scales.size(1);

    TORCH_CHECK(token_offsets.is_cuda(), "token_offsets must be a CUDA tensor.");
    TORCH_CHECK(token_offsets.get_device() == input_device, "token_offsets must be on the same CUDA device as input.");
    TORCH_CHECK(token_offsets.dim() == 1, "token_offsets must be 1D.");
    TORCH_CHECK(token_offsets.scalar_type() == torch::kInt32, "token_offsets must be int32.");
    TORCH_CHECK(token_offsets.is_contiguous(), "token_offsets must be contiguous.");
    TORCH_CHECK(token_offsets.numel() >= 2, "token_offsets must contain ep_size + 1 entries.");
    int64_t const ep_size = token_offsets.numel() - 1;

    TORCH_CHECK(tile_idx_to_mn_limit.is_cuda(), "tile_idx_to_mn_limit must be a CUDA tensor.");
    TORCH_CHECK(tile_idx_to_mn_limit.get_device() == input_device,
        "tile_idx_to_mn_limit must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    TORCH_CHECK(permuted_idx_to_expanded_idx.is_cuda(), "permuted_idx_to_expanded_idx must be a CUDA tensor.");
    TORCH_CHECK(permuted_idx_to_expanded_idx.get_device() == input_device,
        "permuted_idx_to_expanded_idx must be on the same CUDA device as input.");
    TORCH_CHECK(permuted_idx_to_expanded_idx.dim() == 1, "permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(
        permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32, "permuted_idx_to_expanded_idx must be int32.");
    TORCH_CHECK(num_non_exiting_tiles.is_cuda(), "num_non_exiting_tiles must be a CUDA tensor.");
    TORCH_CHECK(num_non_exiting_tiles.get_device() == input_device,
        "num_non_exiting_tiles must be on the same CUDA device as input.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    TORCH_CHECK(l1_acts_pool.is_cuda(), "l1_acts_pool must be a CUDA tensor.");
    TORCH_CHECK(l1_acts_pool.get_device() == input_device, "l1_acts_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_acts_pool.dim() == 2, "l1_acts_pool must be 2D.");
    TORCH_CHECK(l1_acts_pool.scalar_type() == torch::kUInt8, "l1_acts_pool must be uint8.");
    TORCH_CHECK(l1_acts_pool.is_contiguous(), "l1_acts_pool must be contiguous.");
    TORCH_CHECK(
        l1_acts_pool.size(1) == hidden_packed_size, "l1_acts_pool.size(1) must match input hidden packed size.");
    TORCH_CHECK(l1_acts_sf_pool.is_cuda(), "l1_acts_sf_pool must be a CUDA tensor.");
    TORCH_CHECK(
        l1_acts_sf_pool.get_device() == input_device, "l1_acts_sf_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_acts_sf_pool.dim() == 2, "l1_acts_sf_pool must be 2D.");
    TORCH_CHECK(l1_acts_sf_pool.scalar_type() == torch::kUInt8, "l1_acts_sf_pool must be uint8.");
    TORCH_CHECK(l1_acts_sf_pool.is_contiguous(), "l1_acts_sf_pool must be contiguous.");
    TORCH_CHECK(l1_acts_sf_pool.size(1) == sf_hidden_size, "l1_acts_sf_pool.size(1) must match input_sf hidden size.");

    TORCH_CHECK(l1_topk_weights_pool.is_cuda(), "l1_topk_weights_pool must be a CUDA tensor.");
    TORCH_CHECK(l1_topk_weights_pool.get_device() == input_device,
        "l1_topk_weights_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_topk_weights_pool.dim() == 1, "l1_topk_weights_pool must be 1D.");
    TORCH_CHECK(l1_topk_weights_pool.scalar_type() == torch::kFloat32, "l1_topk_weights_pool must be float32.");
    TORCH_CHECK(token_src_metadata.is_cuda(), "token_src_metadata must be a CUDA tensor.");
    TORCH_CHECK(token_src_metadata.get_device() == input_device,
        "token_src_metadata must be on the same CUDA device as input.");
    TORCH_CHECK(token_src_metadata.dim() == 2, "token_src_metadata must be 2D.");
    TORCH_CHECK(token_src_metadata.size(1) == 3, "token_src_metadata.size(1) must be 3.");
    TORCH_CHECK(token_src_metadata.scalar_type() == torch::kInt32, "token_src_metadata must be int32.");
    TORCH_CHECK(l1_arrival_count.is_cuda(), "l1_arrival_count must be a CUDA tensor.");
    TORCH_CHECK(
        l1_arrival_count.get_device() == input_device, "l1_arrival_count must be on the same CUDA device as input.");
    TORCH_CHECK(l1_arrival_count.dim() == 1, "l1_arrival_count must be 1D.");
    TORCH_CHECK(l1_arrival_count.scalar_type() == torch::kInt32, "l1_arrival_count must be int32.");

    int64_t num_available_pool_slots = std::min(permuted_idx_to_expanded_idx.numel(), l1_acts_pool.size(0));
    num_available_pool_slots = std::min(num_available_pool_slots, tile_idx_to_mn_limit.numel() * tile_tokens_dim);
    TORCH_CHECK(l1_topk_weights_pool.numel() >= num_available_pool_slots,
        "l1_topk_weights_pool must cover available pool slots.");
    TORCH_CHECK(
        token_src_metadata.size(0) >= num_available_pool_slots, "token_src_metadata must cover available pool slots.");

    TORCH_CHECK(active_pool_slots.is_cuda(), "active_pool_slots must be a CUDA tensor.");
    TORCH_CHECK(
        active_pool_slots.get_device() == input_device, "active_pool_slots must be on the same CUDA device as input.");
    TORCH_CHECK(active_pool_slots.dim() == 1, "active_pool_slots must be 1D.");
    TORCH_CHECK(active_pool_slots.scalar_type() == torch::kInt64, "active_pool_slots must be int64.");
    TORCH_CHECK(active_combine_rows.is_cuda(), "active_combine_rows must be a CUDA tensor.");
    TORCH_CHECK(active_combine_rows.get_device() == input_device,
        "active_combine_rows must be on the same CUDA device as input.");
    TORCH_CHECK(active_combine_rows.dim() == 1, "active_combine_rows must be 1D.");
    TORCH_CHECK(active_combine_rows.scalar_type() == torch::kInt64, "active_combine_rows must be int64.");
    TORCH_CHECK(active_route_count.is_cuda(), "active_route_count must be a CUDA tensor.");
    TORCH_CHECK(active_route_count.get_device() == input_device,
        "active_route_count must be on the same CUDA device as input.");
    TORCH_CHECK(active_route_count.dim() == 1, "active_route_count must be 1D.");
    TORCH_CHECK(active_route_count.numel() == 1, "active_route_count must have 1 element.");
    TORCH_CHECK(active_route_count.scalar_type() == torch::kInt32, "active_route_count must be int32.");
    TORCH_CHECK(
        output_permuted_idx_to_expanded_idx.is_cuda(), "output_permuted_idx_to_expanded_idx must be a CUDA tensor.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.get_device() == input_device,
        "output_permuted_idx_to_expanded_idx must be on the same CUDA device as input.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.dim() == 1, "output_permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32,
        "output_permuted_idx_to_expanded_idx must be int32.");
    TORCH_CHECK(output_token_final_scales.is_cuda(), "output_token_final_scales must be a CUDA tensor.");
    TORCH_CHECK(output_token_final_scales.get_device() == input_device,
        "output_token_final_scales must be on the same CUDA device as input.");
    TORCH_CHECK(output_token_final_scales.dim() == 2, "output_token_final_scales must be 2D.");
    TORCH_CHECK(output_token_final_scales.size(1) == 1, "output_token_final_scales.size(1) must be 1.");
    TORCH_CHECK(
        output_token_final_scales.scalar_type() == torch::kFloat32, "output_token_final_scales must be float32.");

    int64_t const output_mapping_rows = output_permuted_idx_to_expanded_idx.numel();
    int64_t const output_scale_rows = output_token_final_scales.size(0);
    TORCH_CHECK(output_mapping_rows <= num_available_pool_slots,
        "output_permuted_idx_to_expanded_idx must not exceed available pool slots.");
    TORCH_CHECK(active_pool_slots.numel() >= output_mapping_rows, "active_pool_slots must cover output mapping rows.");
    TORCH_CHECK(
        active_combine_rows.numel() >= output_mapping_rows, "active_combine_rows must cover output mapping rows.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "max_num_tokens_per_rank must be positive.");
    TORCH_CHECK(combine_layout_rows > 0, "combine_layout_rows must be positive.");
    TORCH_CHECK(output_scale_rows == 0 || output_scale_rows >= combine_layout_rows,
        "output_token_final_scales must either be empty or cover combine-layout rows.");

    float* output_token_final_scales_ptr
        = output_scale_rows == 0 ? nullptr : output_token_final_scales.data_ptr<float>();

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM5MaterializeDirectFromMoeSort(input.data_ptr<uint8_t>(),
        input_sf.data_ptr<uint8_t>(), topk_scales.data_ptr<float>(), token_offsets.data_ptr<int32_t>(),
        tile_idx_to_mn_limit.data_ptr<int32_t>(), permuted_idx_to_expanded_idx.data_ptr<int32_t>(),
        num_non_exiting_tiles.data_ptr<int32_t>(), l1_acts_pool.data_ptr<uint8_t>(),
        l1_acts_sf_pool.data_ptr<uint8_t>(), l1_topk_weights_pool.data_ptr<float>(),
        token_src_metadata.data_ptr<int32_t>(), l1_arrival_count.data_ptr<int32_t>(),
        active_pool_slots.data_ptr<int64_t>(), active_combine_rows.data_ptr<int64_t>(),
        active_route_count.data_ptr<int32_t>(), output_permuted_idx_to_expanded_idx.data_ptr<int32_t>(),
        output_token_final_scales_ptr, static_cast<int32_t>(total_tokens), static_cast<int32_t>(ep_size),
        static_cast<int32_t>(hidden_packed_size), static_cast<int32_t>(sf_hidden_size), static_cast<int32_t>(top_k),
        static_cast<int32_t>(tile_tokens_dim), static_cast<int32_t>(num_available_pool_slots),
        static_cast<int32_t>(l1_acts_sf_pool.size(0)), static_cast<int32_t>(l1_arrival_count.numel()),
        static_cast<int32_t>(max_num_tokens_per_rank), static_cast<int32_t>(combine_layout_rows),
        static_cast<int32_t>(output_mapping_rows), static_cast<int32_t>(output_scale_rows), stream);
}

void mega_moe_m5_materialize_direct_from_ranked_topk(torch::Tensor const& input, torch::Tensor const& input_sf,
    torch::Tensor const& topk_idx, torch::Tensor const& topk_scales, torch::Tensor const& token_counts,
    torch::Tensor const& expert_route_offsets, torch::Tensor const& l1_acts_pool, torch::Tensor const& l1_acts_sf_pool,
    torch::Tensor const& l1_topk_weights_pool, torch::Tensor const& token_src_metadata,
    torch::Tensor const& l1_arrival_count, torch::Tensor const& active_pool_slots,
    torch::Tensor const& active_combine_rows, torch::Tensor const& active_route_count,
    torch::Tensor const& output_permuted_idx_to_expanded_idx, torch::Tensor const& output_token_final_scales,
    torch::Tensor const& tile_idx_to_expert_idx, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& num_non_exiting_tiles, int64_t const local_rank, int64_t const tile_tokens_dim,
    int64_t const combine_layout_rows)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor.");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [ep, max_tokens, hidden_packed].");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8.");
    int64_t const ep_size = input.size(0);
    int64_t const max_num_tokens_per_rank = input.size(1);
    int64_t const hidden_packed_size = input.size(2);
    int64_t const input_device = input.get_device();
    TORCH_CHECK(ep_size > 0, "input ep dimension must be positive.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "input max-token dimension must be positive.");

    TORCH_CHECK(input_sf.is_cuda() && input_sf.get_device() == input_device,
        "input_sf must be on the same CUDA device as input.");
    TORCH_CHECK(input_sf.dim() == 3, "input_sf must be 3D [ep, max_tokens, sf_hidden].");
    TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8, "input_sf must be uint8.");
    TORCH_CHECK(input_sf.size(0) == ep_size && input_sf.size(1) == max_num_tokens_per_rank,
        "input_sf rank/token dimensions must match input.");
    int64_t const sf_hidden_size = input_sf.size(2);

    TORCH_CHECK(topk_idx.is_cuda() && topk_idx.get_device() == input_device,
        "topk_idx must be on the same CUDA device as input.");
    TORCH_CHECK(topk_idx.dim() == 3, "topk_idx must be 3D [ep, max_tokens, top_k].");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64, "topk_idx must be int64.");
    TORCH_CHECK(topk_idx.size(0) == ep_size && topk_idx.size(1) == max_num_tokens_per_rank,
        "topk_idx rank/token dimensions must match input.");
    int64_t const top_k = topk_idx.size(2);

    TORCH_CHECK(topk_scales.is_cuda() && topk_scales.get_device() == input_device,
        "topk_scales must be on the same CUDA device as input.");
    TORCH_CHECK(topk_scales.dim() == 3, "topk_scales must be 3D [ep, max_tokens, top_k].");
    TORCH_CHECK(topk_scales.scalar_type() == torch::kFloat32, "topk_scales must be float32.");
    TORCH_CHECK(topk_scales.sizes() == topk_idx.sizes(), "topk_scales shape must match topk_idx.");

    TORCH_CHECK(token_counts.is_cuda() && token_counts.get_device() == input_device,
        "token_counts must be on the same CUDA device as input.");
    TORCH_CHECK(token_counts.dim() == 1, "token_counts must be 1D.");
    TORCH_CHECK(token_counts.scalar_type() == torch::kInt32, "token_counts must be int32.");
    TORCH_CHECK(token_counts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(token_counts.numel() == ep_size, "token_counts must have one entry per EP rank.");
    TORCH_CHECK(local_rank >= 0 && local_rank < ep_size, "local_rank must fit ep_size.");

    TORCH_CHECK(expert_route_offsets.is_cuda() && expert_route_offsets.get_device() == input_device,
        "expert_route_offsets must be on the same CUDA device as input.");
    TORCH_CHECK(expert_route_offsets.dim() == 1, "expert_route_offsets must be 1D.");
    TORCH_CHECK(expert_route_offsets.scalar_type() == torch::kInt32, "expert_route_offsets must be int32.");
    TORCH_CHECK(expert_route_offsets.is_contiguous(), "expert_route_offsets must be contiguous.");
    int64_t const num_experts_per_rank = expert_route_offsets.numel();
    TORCH_CHECK(num_experts_per_rank > 0, "expert_route_offsets must not be empty.");

    TORCH_CHECK(l1_acts_pool.is_cuda() && l1_acts_pool.get_device() == input_device,
        "l1_acts_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_acts_pool.dim() == 2, "l1_acts_pool must be 2D.");
    TORCH_CHECK(l1_acts_pool.scalar_type() == torch::kUInt8, "l1_acts_pool must be uint8.");
    TORCH_CHECK(l1_acts_pool.is_contiguous(), "l1_acts_pool must be contiguous.");
    TORCH_CHECK(
        l1_acts_pool.size(1) == hidden_packed_size, "l1_acts_pool.size(1) must match input hidden packed size.");
    int64_t const num_pool_slots = l1_acts_pool.size(0);

    TORCH_CHECK(l1_acts_sf_pool.is_cuda() && l1_acts_sf_pool.get_device() == input_device,
        "l1_acts_sf_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_acts_sf_pool.dim() == 2, "l1_acts_sf_pool must be 2D.");
    TORCH_CHECK(l1_acts_sf_pool.scalar_type() == torch::kUInt8, "l1_acts_sf_pool must be uint8.");
    TORCH_CHECK(l1_acts_sf_pool.is_contiguous(), "l1_acts_sf_pool must be contiguous.");
    TORCH_CHECK(l1_acts_sf_pool.size(1) == sf_hidden_size, "l1_acts_sf_pool.size(1) must match input_sf hidden size.");

    TORCH_CHECK(l1_topk_weights_pool.is_cuda() && l1_topk_weights_pool.get_device() == input_device,
        "l1_topk_weights_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_topk_weights_pool.dim() == 1, "l1_topk_weights_pool must be 1D.");
    TORCH_CHECK(l1_topk_weights_pool.scalar_type() == torch::kFloat32, "l1_topk_weights_pool must be float32.");
    TORCH_CHECK(l1_topk_weights_pool.numel() >= num_pool_slots, "l1_topk_weights_pool must cover pool slots.");

    TORCH_CHECK(token_src_metadata.is_cuda() && token_src_metadata.get_device() == input_device,
        "token_src_metadata must be on the same CUDA device as input.");
    TORCH_CHECK(token_src_metadata.dim() == 2 && token_src_metadata.size(1) == 3,
        "token_src_metadata must be 2D with size(1) == 3.");
    TORCH_CHECK(token_src_metadata.scalar_type() == torch::kInt32, "token_src_metadata must be int32.");
    TORCH_CHECK(token_src_metadata.numel() >= num_pool_slots * 3, "token_src_metadata must cover pool slots.");

    TORCH_CHECK(l1_arrival_count.is_cuda() && l1_arrival_count.get_device() == input_device,
        "l1_arrival_count must be on the same CUDA device as input.");
    TORCH_CHECK(l1_arrival_count.dim() == 1, "l1_arrival_count must be 1D.");
    TORCH_CHECK(l1_arrival_count.scalar_type() == torch::kInt32, "l1_arrival_count must be int32.");

    TORCH_CHECK(active_pool_slots.is_cuda() && active_pool_slots.get_device() == input_device,
        "active_pool_slots must be on the same CUDA device as input.");
    TORCH_CHECK(active_pool_slots.dim() == 1, "active_pool_slots must be 1D.");
    TORCH_CHECK(active_pool_slots.scalar_type() == torch::kInt64, "active_pool_slots must be int64.");
    TORCH_CHECK(active_combine_rows.is_cuda() && active_combine_rows.get_device() == input_device,
        "active_combine_rows must be on the same CUDA device as input.");
    TORCH_CHECK(active_combine_rows.dim() == 1, "active_combine_rows must be 1D.");
    TORCH_CHECK(active_combine_rows.scalar_type() == torch::kInt64, "active_combine_rows must be int64.");
    TORCH_CHECK(active_pool_slots.numel() >= num_pool_slots, "active_pool_slots must cover pool slots.");
    TORCH_CHECK(active_combine_rows.numel() >= num_pool_slots, "active_combine_rows must cover pool slots.");

    TORCH_CHECK(active_route_count.is_cuda() && active_route_count.get_device() == input_device,
        "active_route_count must be on the same CUDA device as input.");
    TORCH_CHECK(active_route_count.dim() == 1 && active_route_count.numel() == 1,
        "active_route_count must be 1D with 1 element.");
    TORCH_CHECK(active_route_count.scalar_type() == torch::kInt32, "active_route_count must be int32.");

    TORCH_CHECK(output_permuted_idx_to_expanded_idx.is_cuda()
            && output_permuted_idx_to_expanded_idx.get_device() == input_device,
        "output_permuted_idx_to_expanded_idx must be on the same CUDA device as input.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.dim() == 1, "output_permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32,
        "output_permuted_idx_to_expanded_idx must be int32.");
    int64_t const output_mapping_rows = output_permuted_idx_to_expanded_idx.numel();

    TORCH_CHECK(output_token_final_scales.is_cuda() && output_token_final_scales.get_device() == input_device,
        "output_token_final_scales must be on the same CUDA device as input.");
    TORCH_CHECK(output_token_final_scales.dim() == 2 && output_token_final_scales.size(1) == 1,
        "output_token_final_scales must be 2D with size(1) == 1.");
    TORCH_CHECK(
        output_token_final_scales.scalar_type() == torch::kFloat32, "output_token_final_scales must be float32.");
    int64_t const output_scale_rows = output_token_final_scales.size(0);

    TORCH_CHECK(tile_idx_to_expert_idx.is_cuda() && tile_idx_to_expert_idx.get_device() == input_device,
        "tile_idx_to_expert_idx must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_expert_idx.dim() == 1, "tile_idx_to_expert_idx must be 1D.");
    TORCH_CHECK(tile_idx_to_expert_idx.scalar_type() == torch::kInt32, "tile_idx_to_expert_idx must be int32.");
    TORCH_CHECK(tile_idx_to_mn_limit.is_cuda() && tile_idx_to_mn_limit.get_device() == input_device,
        "tile_idx_to_mn_limit must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const route_layout_capacity = std::min(tile_idx_to_expert_idx.numel(), tile_idx_to_mn_limit.numel());

    TORCH_CHECK(num_non_exiting_tiles.is_cuda() && num_non_exiting_tiles.get_device() == input_device,
        "num_non_exiting_tiles must be on the same CUDA device as input.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    float* output_token_final_scales_ptr
        = output_scale_rows == 0 ? nullptr : output_token_final_scales.data_ptr<float>();

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM5MaterializeDirectFromRankedTopK(input.data_ptr<uint8_t>(),
        input.stride(0), input.stride(1), input_sf.data_ptr<uint8_t>(), input_sf.stride(0), input_sf.stride(1),
        topk_idx.data_ptr<int64_t>(), topk_idx.stride(0), topk_idx.stride(1), topk_scales.data_ptr<float>(),
        topk_scales.stride(0), topk_scales.stride(1), token_counts.data_ptr<int32_t>(),
        expert_route_offsets.data_ptr<int32_t>(), l1_acts_pool.data_ptr<uint8_t>(), l1_acts_sf_pool.data_ptr<uint8_t>(),
        l1_topk_weights_pool.data_ptr<float>(), token_src_metadata.data_ptr<int32_t>(),
        l1_arrival_count.data_ptr<int32_t>(), active_pool_slots.data_ptr<int64_t>(),
        active_combine_rows.data_ptr<int64_t>(), active_route_count.data_ptr<int32_t>(),
        output_permuted_idx_to_expanded_idx.data_ptr<int32_t>(), output_token_final_scales_ptr,
        tile_idx_to_expert_idx.data_ptr<int32_t>(), tile_idx_to_mn_limit.data_ptr<int32_t>(),
        num_non_exiting_tiles.data_ptr<int32_t>(), static_cast<int32_t>(ep_size), static_cast<int32_t>(local_rank),
        static_cast<int32_t>(num_experts_per_rank), static_cast<int32_t>(hidden_packed_size),
        static_cast<int32_t>(sf_hidden_size), static_cast<int32_t>(top_k), static_cast<int32_t>(tile_tokens_dim),
        static_cast<int32_t>(num_pool_slots), static_cast<int32_t>(l1_acts_sf_pool.size(0)),
        static_cast<int32_t>(l1_arrival_count.numel()), static_cast<int32_t>(max_num_tokens_per_rank),
        static_cast<int32_t>(combine_layout_rows), static_cast<int32_t>(output_mapping_rows),
        static_cast<int32_t>(output_scale_rows), static_cast<int32_t>(route_layout_capacity), stream);
}

void mega_moe_stage_dispatch_inputs(torch::Tensor const& input, torch::Tensor const& input_sf,
    torch::Tensor const& topk_idx, torch::Tensor const& topk_scales, torch::Tensor const& input_buffer,
    torch::Tensor const& input_sf_buffer, torch::Tensor const& topk_idx_buffer, torch::Tensor const& topk_scales_buffer)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor.");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");
    int64_t const input_device = input.get_device();

    TORCH_CHECK(input_sf.is_cuda() && input_sf.get_device() == input_device,
        "input_sf must be on the same CUDA device as input.");
    TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous.");
    TORCH_CHECK(topk_idx.is_cuda() && topk_idx.get_device() == input_device,
        "topk_idx must be on the same CUDA device as input.");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64, "topk_idx must be int64.");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous.");
    TORCH_CHECK(topk_scales.is_cuda() && topk_scales.get_device() == input_device,
        "topk_scales must be on the same CUDA device as input.");
    TORCH_CHECK(topk_scales.scalar_type() == torch::kFloat32, "topk_scales must be float32.");
    TORCH_CHECK(topk_scales.is_contiguous(), "topk_scales must be contiguous.");

    auto check_buffer = [input_device](torch::Tensor const& buffer, char const* name, int64_t required_bytes)
    {
        TORCH_CHECK(buffer.is_cuda() && buffer.get_device() == input_device, name, " must be on the same CUDA device.");
        TORCH_CHECK(buffer.scalar_type() == torch::kUInt8, name, " must be uint8.");
        TORCH_CHECK(buffer.is_contiguous(), name, " must be contiguous.");
        TORCH_CHECK(buffer.numel() >= required_bytes, name, " is too small for staged source bytes.");
    };

    int64_t const input_bytes = input.numel() * input.element_size();
    int64_t const input_sf_bytes = input_sf.numel() * input_sf.element_size();
    int64_t const topk_idx_bytes = topk_idx.numel() * topk_idx.element_size();
    int64_t const topk_scales_bytes = topk_scales.numel() * topk_scales.element_size();
    check_buffer(input_buffer, "input_buffer", input_bytes);
    check_buffer(input_sf_buffer, "input_sf_buffer", input_sf_bytes);
    check_buffer(topk_idx_buffer, "topk_idx_buffer", topk_idx_bytes);
    check_buffer(topk_scales_buffer, "topk_scales_buffer", topk_scales_bytes);

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeStageDispatchInputs(static_cast<uint8_t const*>(input.data_ptr()),
        input_bytes, static_cast<uint8_t const*>(input_sf.data_ptr()), input_sf_bytes,
        static_cast<uint8_t const*>(topk_idx.data_ptr()), topk_idx_bytes,
        static_cast<uint8_t const*>(topk_scales.data_ptr()), topk_scales_bytes, input_buffer.data_ptr<uint8_t>(),
        input_sf_buffer.data_ptr<uint8_t>(), topk_idx_buffer.data_ptr<uint8_t>(),
        topk_scales_buffer.data_ptr<uint8_t>(), stream);
}

void mega_moe_m5_init_direct_input_route_metadata(torch::Tensor const& expert_route_offsets,
    torch::Tensor const& tile_idx_to_expert_idx, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& num_non_exiting_tiles)
{
    TORCH_CHECK(expert_route_offsets.is_cuda(), "expert_route_offsets must be a CUDA tensor.");
    TORCH_CHECK(expert_route_offsets.dim() == 1, "expert_route_offsets must be 1D.");
    TORCH_CHECK(expert_route_offsets.scalar_type() == torch::kInt32, "expert_route_offsets must be int32.");
    TORCH_CHECK(expert_route_offsets.is_contiguous(), "expert_route_offsets must be contiguous.");
    int64_t const input_device = expert_route_offsets.get_device();
    int64_t const num_experts_per_rank = expert_route_offsets.numel();
    TORCH_CHECK(num_experts_per_rank > 0, "expert_route_offsets must not be empty.");

    TORCH_CHECK(tile_idx_to_expert_idx.is_cuda() && tile_idx_to_expert_idx.get_device() == input_device,
        "tile_idx_to_expert_idx must be on the same CUDA device as expert_route_offsets.");
    TORCH_CHECK(tile_idx_to_expert_idx.dim() == 1, "tile_idx_to_expert_idx must be 1D.");
    TORCH_CHECK(tile_idx_to_expert_idx.scalar_type() == torch::kInt32, "tile_idx_to_expert_idx must be int32.");
    TORCH_CHECK(tile_idx_to_expert_idx.is_contiguous(), "tile_idx_to_expert_idx must be contiguous.");
    TORCH_CHECK(tile_idx_to_mn_limit.is_cuda() && tile_idx_to_mn_limit.get_device() == input_device,
        "tile_idx_to_mn_limit must be on the same CUDA device as expert_route_offsets.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    TORCH_CHECK(tile_idx_to_mn_limit.is_contiguous(), "tile_idx_to_mn_limit must be contiguous.");
    int64_t const route_layout_capacity = std::min(tile_idx_to_expert_idx.numel(), tile_idx_to_mn_limit.numel());
    TORCH_CHECK(route_layout_capacity > 0, "route layout tensors must not be empty.");

    TORCH_CHECK(num_non_exiting_tiles.is_cuda() && num_non_exiting_tiles.get_device() == input_device,
        "num_non_exiting_tiles must be on the same CUDA device as expert_route_offsets.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");
    TORCH_CHECK(num_non_exiting_tiles.is_contiguous(), "num_non_exiting_tiles must be contiguous.");

    auto const& stream = at::cuda::getCurrentCUDAStream(expert_route_offsets.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM5InitDirectInputRouteMetadata(expert_route_offsets.data_ptr<int32_t>(),
        tile_idx_to_expert_idx.data_ptr<int32_t>(), tile_idx_to_mn_limit.data_ptr<int32_t>(),
        num_non_exiting_tiles.data_ptr<int32_t>(), static_cast<int32_t>(num_experts_per_rank),
        static_cast<int32_t>(route_layout_capacity), stream);
}

void mega_moe_m5_build_direct_input_route_from_ranked_topk(torch::Tensor const& input, torch::Tensor const& input_sf,
    torch::Tensor const& topk_idx, torch::Tensor const& topk_scales, torch::Tensor const& token_counts,
    torch::Tensor const& direct_input, torch::Tensor const& direct_input_sf, torch::Tensor const& expert_route_offsets,
    torch::Tensor const& expert_route_base_offsets, torch::Tensor const& token_id_mapping,
    torch::Tensor const& output_permuted_idx_to_expanded_idx, torch::Tensor const& output_token_final_scales,
    torch::Tensor const& tile_idx_to_expert_idx, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& num_non_exiting_tiles, int64_t const local_rank, int64_t const tile_tokens_dim,
    int64_t const combine_layout_rows, bool const direct_atomic_output, bool const direct_token_major_output)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor.");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [ep, max_tokens, hidden_packed].");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8.");
    int64_t const ep_size = input.size(0);
    int64_t const max_num_tokens_per_rank = input.size(1);
    int64_t const hidden_packed_size = input.size(2);
    int64_t const input_device = input.get_device();
    TORCH_CHECK(ep_size > 0, "input ep dimension must be positive.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "input max-token dimension must be positive.");
    TORCH_CHECK(hidden_packed_size > 0, "input hidden dimension must be positive.");
    TORCH_CHECK(local_rank >= 0 && local_rank < ep_size, "local_rank must fit ep_size.");
    TORCH_CHECK(tile_tokens_dim > 0, "tile_tokens_dim must be positive.");
    TORCH_CHECK(combine_layout_rows > 0, "combine_layout_rows must be positive.");

    TORCH_CHECK(input_sf.is_cuda() && input_sf.get_device() == input_device,
        "input_sf must be on the same CUDA device as input.");
    TORCH_CHECK(input_sf.dim() == 3, "input_sf must be 3D [ep, max_tokens, sf_hidden].");
    TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8, "input_sf must be uint8.");
    TORCH_CHECK(input_sf.size(0) == ep_size && input_sf.size(1) == max_num_tokens_per_rank,
        "input_sf leading dimensions must match input.");
    int64_t const sf_hidden_size = input_sf.size(2);
    TORCH_CHECK(sf_hidden_size > 0, "input_sf hidden dimension must be positive.");

    TORCH_CHECK(topk_idx.is_cuda() && topk_idx.get_device() == input_device,
        "topk_idx must be on the same CUDA device as input.");
    TORCH_CHECK(topk_idx.dim() == 3, "topk_idx must be 3D [ep, max_tokens, top_k].");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64, "topk_idx must be int64.");
    TORCH_CHECK(topk_idx.size(0) == ep_size && topk_idx.size(1) == max_num_tokens_per_rank,
        "topk_idx leading dimensions must match input.");
    int64_t const top_k = topk_idx.size(2);
    TORCH_CHECK(top_k > 0, "topk_idx top-k dimension must be positive.");

    TORCH_CHECK(topk_scales.is_cuda() && topk_scales.get_device() == input_device,
        "topk_scales must be on the same CUDA device as input.");
    TORCH_CHECK(topk_scales.dim() == 3, "topk_scales must be 3D [ep, max_tokens, top_k].");
    TORCH_CHECK(topk_scales.scalar_type() == torch::kFloat32, "topk_scales must be float32.");
    TORCH_CHECK(topk_scales.sizes() == topk_idx.sizes(), "topk_scales shape must match topk_idx.");

    TORCH_CHECK(token_counts.is_cuda() && token_counts.get_device() == input_device,
        "token_counts must be on the same CUDA device as input.");
    TORCH_CHECK(token_counts.dim() == 1, "token_counts must be 1D.");
    TORCH_CHECK(token_counts.scalar_type() == torch::kInt32, "token_counts must be int32.");
    TORCH_CHECK(token_counts.is_contiguous(), "token_counts must be contiguous.");
    TORCH_CHECK(token_counts.numel() == ep_size, "token_counts must have one entry per EP rank.");

    int64_t const flat_input_rows = ep_size * max_num_tokens_per_rank;
    TORCH_CHECK(direct_input.is_cuda() && direct_input.get_device() == input_device,
        "direct_input must be on the same CUDA device as input.");
    TORCH_CHECK(direct_input.dim() == 2, "direct_input must be 2D.");
    TORCH_CHECK(direct_input.scalar_type() == torch::kUInt8, "direct_input must be uint8.");
    TORCH_CHECK(direct_input.is_contiguous(), "direct_input must be contiguous.");
    TORCH_CHECK(direct_input.size(0) >= flat_input_rows, "direct_input must cover all flattened input rows.");
    TORCH_CHECK(direct_input.size(1) == hidden_packed_size, "direct_input hidden size must match input.");

    TORCH_CHECK(direct_input_sf.is_cuda() && direct_input_sf.get_device() == input_device,
        "direct_input_sf must be on the same CUDA device as input.");
    TORCH_CHECK(direct_input_sf.dim() == 2, "direct_input_sf must be 2D.");
    TORCH_CHECK(direct_input_sf.scalar_type() == torch::kUInt8, "direct_input_sf must be uint8.");
    TORCH_CHECK(direct_input_sf.is_contiguous(), "direct_input_sf must be contiguous.");
    TORCH_CHECK(direct_input_sf.size(0) >= flat_input_rows, "direct_input_sf must cover all flattened input rows.");
    TORCH_CHECK(direct_input_sf.size(1) == sf_hidden_size, "direct_input_sf hidden size must match input_sf.");

    TORCH_CHECK(expert_route_offsets.is_cuda() && expert_route_offsets.get_device() == input_device,
        "expert_route_offsets must be on the same CUDA device as input.");
    TORCH_CHECK(expert_route_offsets.dim() == 1, "expert_route_offsets must be 1D.");
    TORCH_CHECK(expert_route_offsets.scalar_type() == torch::kInt32, "expert_route_offsets must be int32.");
    TORCH_CHECK(expert_route_offsets.is_contiguous(), "expert_route_offsets must be contiguous.");
    int64_t const num_experts_per_rank = expert_route_offsets.numel();
    TORCH_CHECK(num_experts_per_rank > 0, "expert_route_offsets must not be empty.");

    TORCH_CHECK(expert_route_base_offsets.is_cuda() && expert_route_base_offsets.get_device() == input_device,
        "expert_route_base_offsets must be on the same CUDA device as input.");
    TORCH_CHECK(expert_route_base_offsets.dim() == 1, "expert_route_base_offsets must be 1D.");
    TORCH_CHECK(expert_route_base_offsets.scalar_type() == torch::kInt32, "expert_route_base_offsets must be int32.");
    TORCH_CHECK(expert_route_base_offsets.is_contiguous(), "expert_route_base_offsets must be contiguous.");
    TORCH_CHECK(expert_route_base_offsets.numel() == num_experts_per_rank,
        "expert_route_base_offsets must match expert_route_offsets shape.");

    TORCH_CHECK(token_id_mapping.is_cuda() && token_id_mapping.get_device() == input_device,
        "token_id_mapping must be on the same CUDA device as input.");
    TORCH_CHECK(token_id_mapping.dim() == 1, "token_id_mapping must be 1D.");
    TORCH_CHECK(token_id_mapping.scalar_type() == torch::kInt32, "token_id_mapping must be int32.");
    int64_t const num_pool_slots = token_id_mapping.numel();
    TORCH_CHECK(num_pool_slots > 0, "token_id_mapping must not be empty.");

    TORCH_CHECK(output_permuted_idx_to_expanded_idx.is_cuda()
            && output_permuted_idx_to_expanded_idx.get_device() == input_device,
        "output_permuted_idx_to_expanded_idx must be on the same CUDA device as input.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.dim() == 1, "output_permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32,
        "output_permuted_idx_to_expanded_idx must be int32.");
    int64_t const output_mapping_rows = output_permuted_idx_to_expanded_idx.numel();
    TORCH_CHECK(
        output_mapping_rows >= num_pool_slots, "output_permuted_idx_to_expanded_idx must cover token_id_mapping rows.");

    TORCH_CHECK(output_token_final_scales.is_cuda() && output_token_final_scales.get_device() == input_device,
        "output_token_final_scales must be on the same CUDA device as input.");
    TORCH_CHECK(output_token_final_scales.dim() == 2, "output_token_final_scales must be 2D.");
    TORCH_CHECK(
        output_token_final_scales.scalar_type() == torch::kFloat32, "output_token_final_scales must be float32.");
    int64_t const output_scale_rows
        = direct_atomic_output ? output_token_final_scales.numel() : output_token_final_scales.size(0);
    if (direct_atomic_output)
    {
        TORCH_CHECK(
            output_token_final_scales.size(1) == top_k, "atomic output_token_final_scales.size(1) must match top_k.");
        TORCH_CHECK(output_scale_rows >= flat_input_rows * top_k,
            "atomic output_token_final_scales must cover flattened input rows and top_k.");
    }
    else
    {
        TORCH_CHECK(output_token_final_scales.size(1) == 1, "output_token_final_scales.size(1) must be 1.");
        TORCH_CHECK(output_scale_rows == 0 || output_scale_rows >= combine_layout_rows,
            "output_token_final_scales must either be empty or cover combine-layout rows.");
    }

    TORCH_CHECK(tile_idx_to_expert_idx.is_cuda() && tile_idx_to_expert_idx.get_device() == input_device,
        "tile_idx_to_expert_idx must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_expert_idx.dim() == 1, "tile_idx_to_expert_idx must be 1D.");
    TORCH_CHECK(tile_idx_to_expert_idx.scalar_type() == torch::kInt32, "tile_idx_to_expert_idx must be int32.");
    TORCH_CHECK(tile_idx_to_mn_limit.is_cuda() && tile_idx_to_mn_limit.get_device() == input_device,
        "tile_idx_to_mn_limit must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const route_layout_capacity = std::min(tile_idx_to_expert_idx.numel(), tile_idx_to_mn_limit.numel());

    TORCH_CHECK(num_non_exiting_tiles.is_cuda() && num_non_exiting_tiles.get_device() == input_device,
        "num_non_exiting_tiles must be on the same CUDA device as input.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    float* output_token_final_scales_ptr
        = output_scale_rows == 0 ? nullptr : output_token_final_scales.data_ptr<float>();

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM5BuildDirectInputRouteFromRankedTopK(input.data_ptr<uint8_t>(),
        input.stride(0), input.stride(1), input_sf.data_ptr<uint8_t>(), input_sf.stride(0), input_sf.stride(1),
        direct_input.data_ptr<uint8_t>(), direct_input_sf.data_ptr<uint8_t>(), topk_idx.data_ptr<int64_t>(),
        topk_idx.stride(0), topk_idx.stride(1), topk_scales.data_ptr<float>(), topk_scales.stride(0),
        topk_scales.stride(1), token_counts.data_ptr<int32_t>(), expert_route_offsets.data_ptr<int32_t>(),
        expert_route_base_offsets.data_ptr<int32_t>(), token_id_mapping.data_ptr<int32_t>(),
        output_permuted_idx_to_expanded_idx.data_ptr<int32_t>(), output_token_final_scales_ptr,
        tile_idx_to_expert_idx.data_ptr<int32_t>(), tile_idx_to_mn_limit.data_ptr<int32_t>(),
        num_non_exiting_tiles.data_ptr<int32_t>(), static_cast<int32_t>(ep_size), static_cast<int32_t>(local_rank),
        static_cast<int32_t>(num_experts_per_rank), static_cast<int32_t>(hidden_packed_size),
        static_cast<int32_t>(sf_hidden_size), static_cast<int32_t>(top_k), static_cast<int32_t>(tile_tokens_dim),
        static_cast<int32_t>(num_pool_slots), static_cast<int32_t>(max_num_tokens_per_rank),
        static_cast<int32_t>(combine_layout_rows), static_cast<int32_t>(output_mapping_rows),
        static_cast<int32_t>(output_scale_rows), static_cast<int32_t>(route_layout_capacity), direct_atomic_output,
        direct_token_major_output, stream);
}

void mega_moe_m5_materialize_direct_from_topk(torch::Tensor const& input, torch::Tensor const& input_sf,
    torch::Tensor const& topk_idx, torch::Tensor const& topk_scales, torch::Tensor const& token_offsets,
    torch::Tensor const& expert_recv_count_sum, torch::Tensor const& expert_route_offsets,
    torch::Tensor const& l1_acts_pool, torch::Tensor const& l1_acts_sf_pool, torch::Tensor const& l1_topk_weights_pool,
    torch::Tensor const& token_src_metadata, torch::Tensor const& l1_arrival_count,
    torch::Tensor const& active_pool_slots, torch::Tensor const& active_combine_rows,
    torch::Tensor const& active_route_count, torch::Tensor const& output_permuted_idx_to_expanded_idx,
    torch::Tensor const& output_token_final_scales, torch::Tensor const& tile_idx_to_expert_idx,
    torch::Tensor const& tile_idx_to_mn_limit, torch::Tensor const& num_non_exiting_tiles, int64_t const local_rank,
    int64_t const tile_tokens_dim, int64_t const max_num_tokens_per_rank, int64_t const combine_layout_rows)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor.");
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(input.scalar_type() == torch::kUInt8, "input must be uint8.");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");
    int64_t const total_tokens = input.size(0);
    int64_t const hidden_packed_size = input.size(1);
    int64_t const input_device = input.get_device();

    TORCH_CHECK(input_sf.is_cuda(), "input_sf must be a CUDA tensor.");
    TORCH_CHECK(input_sf.get_device() == input_device, "input_sf must be on the same CUDA device as input.");
    TORCH_CHECK(input_sf.dim() == 2, "input_sf must be 2D.");
    TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8, "input_sf must be uint8.");
    TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous.");
    TORCH_CHECK(input_sf.size(0) == total_tokens, "input_sf.size(0) must match input.size(0).");
    int64_t const sf_hidden_size = input_sf.size(1);

    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor.");
    TORCH_CHECK(topk_idx.get_device() == input_device, "topk_idx must be on the same CUDA device as input.");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be 2D.");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64, "topk_idx must be int64.");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous.");
    TORCH_CHECK(topk_idx.size(0) == total_tokens, "topk_idx.size(0) must match input.size(0).");
    int64_t const top_k = topk_idx.size(1);

    TORCH_CHECK(topk_scales.is_cuda(), "topk_scales must be a CUDA tensor.");
    TORCH_CHECK(topk_scales.get_device() == input_device, "topk_scales must be on the same CUDA device as input.");
    TORCH_CHECK(topk_scales.dim() == 2, "topk_scales must be 2D.");
    TORCH_CHECK(topk_scales.scalar_type() == torch::kFloat32, "topk_scales must be float32.");
    TORCH_CHECK(topk_scales.is_contiguous(), "topk_scales must be contiguous.");
    TORCH_CHECK(topk_scales.sizes() == topk_idx.sizes(), "topk_scales shape must match topk_idx.");

    TORCH_CHECK(token_offsets.is_cuda(), "token_offsets must be a CUDA tensor.");
    TORCH_CHECK(token_offsets.get_device() == input_device, "token_offsets must be on the same CUDA device as input.");
    TORCH_CHECK(token_offsets.dim() == 1, "token_offsets must be 1D.");
    TORCH_CHECK(token_offsets.scalar_type() == torch::kInt32, "token_offsets must be int32.");
    TORCH_CHECK(token_offsets.is_contiguous(), "token_offsets must be contiguous.");
    TORCH_CHECK(token_offsets.numel() >= 2, "token_offsets must contain ep_size + 1 entries.");
    int64_t const ep_size = token_offsets.numel() - 1;
    TORCH_CHECK(local_rank >= 0 && local_rank < ep_size, "local_rank must fit ep_size.");

    TORCH_CHECK(expert_recv_count_sum.is_cuda(), "expert_recv_count_sum must be a CUDA tensor.");
    TORCH_CHECK(expert_recv_count_sum.get_device() == input_device,
        "expert_recv_count_sum must be on the same CUDA device as input.");
    TORCH_CHECK(expert_recv_count_sum.dim() == 1, "expert_recv_count_sum must be 1D.");
    TORCH_CHECK(expert_recv_count_sum.scalar_type() == torch::kInt64, "expert_recv_count_sum must be int64.");
    TORCH_CHECK(expert_recv_count_sum.is_contiguous(), "expert_recv_count_sum must be contiguous.");
    int64_t const num_experts_per_rank = expert_recv_count_sum.numel();
    TORCH_CHECK(num_experts_per_rank > 0, "expert_recv_count_sum must not be empty.");

    TORCH_CHECK(expert_route_offsets.is_cuda(), "expert_route_offsets must be a CUDA tensor.");
    TORCH_CHECK(expert_route_offsets.get_device() == input_device,
        "expert_route_offsets must be on the same CUDA device as input.");
    TORCH_CHECK(expert_route_offsets.dim() == 1, "expert_route_offsets must be 1D.");
    TORCH_CHECK(expert_route_offsets.scalar_type() == torch::kInt32, "expert_route_offsets must be int32.");
    TORCH_CHECK(expert_route_offsets.is_contiguous(), "expert_route_offsets must be contiguous.");
    TORCH_CHECK(
        expert_route_offsets.numel() >= num_experts_per_rank, "expert_route_offsets must cover expert_recv_count_sum.");

    TORCH_CHECK(l1_acts_pool.is_cuda() && l1_acts_pool.get_device() == input_device,
        "l1_acts_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_acts_pool.dim() == 2, "l1_acts_pool must be 2D.");
    TORCH_CHECK(l1_acts_pool.scalar_type() == torch::kUInt8, "l1_acts_pool must be uint8.");
    TORCH_CHECK(l1_acts_pool.is_contiguous(), "l1_acts_pool must be contiguous.");
    TORCH_CHECK(
        l1_acts_pool.size(1) == hidden_packed_size, "l1_acts_pool.size(1) must match input hidden packed size.");
    int64_t const num_pool_slots = l1_acts_pool.size(0);

    TORCH_CHECK(l1_acts_sf_pool.is_cuda() && l1_acts_sf_pool.get_device() == input_device,
        "l1_acts_sf_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_acts_sf_pool.dim() == 2, "l1_acts_sf_pool must be 2D.");
    TORCH_CHECK(l1_acts_sf_pool.scalar_type() == torch::kUInt8, "l1_acts_sf_pool must be uint8.");
    TORCH_CHECK(l1_acts_sf_pool.is_contiguous(), "l1_acts_sf_pool must be contiguous.");
    TORCH_CHECK(l1_acts_sf_pool.size(1) == sf_hidden_size, "l1_acts_sf_pool.size(1) must match input_sf hidden size.");

    TORCH_CHECK(l1_topk_weights_pool.is_cuda() && l1_topk_weights_pool.get_device() == input_device,
        "l1_topk_weights_pool must be on the same CUDA device as input.");
    TORCH_CHECK(l1_topk_weights_pool.dim() == 1, "l1_topk_weights_pool must be 1D.");
    TORCH_CHECK(l1_topk_weights_pool.scalar_type() == torch::kFloat32, "l1_topk_weights_pool must be float32.");
    TORCH_CHECK(l1_topk_weights_pool.numel() >= num_pool_slots, "l1_topk_weights_pool must cover pool slots.");

    TORCH_CHECK(token_src_metadata.is_cuda() && token_src_metadata.get_device() == input_device,
        "token_src_metadata must be on the same CUDA device as input.");
    TORCH_CHECK(token_src_metadata.dim() == 2, "token_src_metadata must be 2D.");
    TORCH_CHECK(token_src_metadata.size(1) == 3, "token_src_metadata.size(1) must be 3.");
    TORCH_CHECK(token_src_metadata.scalar_type() == torch::kInt32, "token_src_metadata must be int32.");
    TORCH_CHECK(token_src_metadata.numel() >= num_pool_slots * 3, "token_src_metadata must cover pool slots.");

    TORCH_CHECK(l1_arrival_count.is_cuda() && l1_arrival_count.get_device() == input_device,
        "l1_arrival_count must be on the same CUDA device as input.");
    TORCH_CHECK(l1_arrival_count.dim() == 1, "l1_arrival_count must be 1D.");
    TORCH_CHECK(l1_arrival_count.scalar_type() == torch::kInt32, "l1_arrival_count must be int32.");

    TORCH_CHECK(active_pool_slots.is_cuda() && active_pool_slots.get_device() == input_device,
        "active_pool_slots must be on the same CUDA device as input.");
    TORCH_CHECK(active_pool_slots.dim() == 1, "active_pool_slots must be 1D.");
    TORCH_CHECK(active_pool_slots.scalar_type() == torch::kInt64, "active_pool_slots must be int64.");
    TORCH_CHECK(active_combine_rows.is_cuda() && active_combine_rows.get_device() == input_device,
        "active_combine_rows must be on the same CUDA device as input.");
    TORCH_CHECK(active_combine_rows.dim() == 1, "active_combine_rows must be 1D.");
    TORCH_CHECK(active_combine_rows.scalar_type() == torch::kInt64, "active_combine_rows must be int64.");
    TORCH_CHECK(active_pool_slots.numel() >= num_pool_slots, "active_pool_slots must cover pool slots.");
    TORCH_CHECK(active_combine_rows.numel() >= num_pool_slots, "active_combine_rows must cover pool slots.");

    TORCH_CHECK(active_route_count.is_cuda() && active_route_count.get_device() == input_device,
        "active_route_count must be on the same CUDA device as input.");
    TORCH_CHECK(active_route_count.dim() == 1, "active_route_count must be 1D.");
    TORCH_CHECK(active_route_count.numel() == 1, "active_route_count must have 1 element.");
    TORCH_CHECK(active_route_count.scalar_type() == torch::kInt32, "active_route_count must be int32.");

    TORCH_CHECK(output_permuted_idx_to_expanded_idx.is_cuda()
            && output_permuted_idx_to_expanded_idx.get_device() == input_device,
        "output_permuted_idx_to_expanded_idx must be on the same CUDA device as input.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.dim() == 1, "output_permuted_idx_to_expanded_idx must be 1D.");
    TORCH_CHECK(output_permuted_idx_to_expanded_idx.scalar_type() == torch::kInt32,
        "output_permuted_idx_to_expanded_idx must be int32.");
    int64_t const output_mapping_rows = output_permuted_idx_to_expanded_idx.numel();

    TORCH_CHECK(output_token_final_scales.is_cuda() && output_token_final_scales.get_device() == input_device,
        "output_token_final_scales must be on the same CUDA device as input.");
    TORCH_CHECK(output_token_final_scales.dim() == 2, "output_token_final_scales must be 2D.");
    TORCH_CHECK(output_token_final_scales.size(1) == 1, "output_token_final_scales.size(1) must be 1.");
    TORCH_CHECK(
        output_token_final_scales.scalar_type() == torch::kFloat32, "output_token_final_scales must be float32.");
    int64_t const output_scale_rows = output_token_final_scales.size(0);

    TORCH_CHECK(tile_idx_to_expert_idx.is_cuda() && tile_idx_to_expert_idx.get_device() == input_device,
        "tile_idx_to_expert_idx must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_expert_idx.dim() == 1, "tile_idx_to_expert_idx must be 1D.");
    TORCH_CHECK(tile_idx_to_expert_idx.scalar_type() == torch::kInt32, "tile_idx_to_expert_idx must be int32.");
    TORCH_CHECK(tile_idx_to_mn_limit.is_cuda() && tile_idx_to_mn_limit.get_device() == input_device,
        "tile_idx_to_mn_limit must be on the same CUDA device as input.");
    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const route_layout_capacity = std::min(tile_idx_to_expert_idx.numel(), tile_idx_to_mn_limit.numel());

    TORCH_CHECK(num_non_exiting_tiles.is_cuda() && num_non_exiting_tiles.get_device() == input_device,
        "num_non_exiting_tiles must be on the same CUDA device as input.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    float* output_token_final_scales_ptr
        = output_scale_rows == 0 ? nullptr : output_token_final_scales.data_ptr<float>();

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM5MaterializeDirectFromTopK(input.data_ptr<uint8_t>(),
        input_sf.data_ptr<uint8_t>(), topk_idx.data_ptr<int64_t>(), topk_scales.data_ptr<float>(),
        token_offsets.data_ptr<int32_t>(), expert_recv_count_sum.data_ptr<int64_t>(),
        expert_route_offsets.data_ptr<int32_t>(), l1_acts_pool.data_ptr<uint8_t>(), l1_acts_sf_pool.data_ptr<uint8_t>(),
        l1_topk_weights_pool.data_ptr<float>(), token_src_metadata.data_ptr<int32_t>(),
        l1_arrival_count.data_ptr<int32_t>(), active_pool_slots.data_ptr<int64_t>(),
        active_combine_rows.data_ptr<int64_t>(), active_route_count.data_ptr<int32_t>(),
        output_permuted_idx_to_expanded_idx.data_ptr<int32_t>(), output_token_final_scales_ptr,
        tile_idx_to_expert_idx.data_ptr<int32_t>(), tile_idx_to_mn_limit.data_ptr<int32_t>(),
        num_non_exiting_tiles.data_ptr<int32_t>(), static_cast<int32_t>(total_tokens), static_cast<int32_t>(ep_size),
        static_cast<int32_t>(local_rank), static_cast<int32_t>(num_experts_per_rank),
        static_cast<int32_t>(hidden_packed_size), static_cast<int32_t>(sf_hidden_size), static_cast<int32_t>(top_k),
        static_cast<int32_t>(tile_tokens_dim), static_cast<int32_t>(num_pool_slots),
        static_cast<int32_t>(l1_acts_sf_pool.size(0)), static_cast<int32_t>(l1_arrival_count.numel()),
        static_cast<int32_t>(max_num_tokens_per_rank), static_cast<int32_t>(combine_layout_rows),
        static_cast<int32_t>(output_mapping_rows), static_cast<int32_t>(output_scale_rows),
        static_cast<int32_t>(route_layout_capacity), stream);
}

void mega_moe_m6_reduce_combine_buffer_out(
    torch::Tensor const& combine_buffer, torch::Tensor const& output, int64_t const local_num_tokens)
{
    TORCH_CHECK(combine_buffer.is_cuda(), "combine_buffer must be a CUDA tensor.");
    TORCH_CHECK(combine_buffer.dim() == 3, "combine_buffer must be 3D [top_k, max_tokens_per_rank, hidden_size].");
    TORCH_CHECK(combine_buffer.scalar_type() == torch::kBFloat16, "combine_buffer must be bfloat16.");
    TORCH_CHECK(combine_buffer.is_contiguous(), "combine_buffer must be contiguous.");
    TORCH_CHECK(local_num_tokens >= 0, "local_num_tokens must be non-negative.");

    int64_t const top_k = combine_buffer.size(0);
    int64_t const max_num_tokens_per_rank = combine_buffer.size(1);
    int64_t const hidden_size = combine_buffer.size(2);
    TORCH_CHECK(top_k > 0, "combine_buffer top_k dimension must be positive.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "combine_buffer max-token dimension must be positive.");
    TORCH_CHECK(hidden_size > 0, "combine_buffer hidden dimension must be positive.");
    TORCH_CHECK(local_num_tokens <= max_num_tokens_per_rank,
        "local_num_tokens must not exceed combine_buffer max-token dimension.");
    TORCH_CHECK(output.is_cuda() && output.get_device() == combine_buffer.get_device(),
        "output must be on the same CUDA device as combine_buffer.");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [max_tokens_per_rank, hidden_size].");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32.");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous.");
    TORCH_CHECK(output.size(0) >= local_num_tokens, "output must cover local_num_tokens rows.");
    TORCH_CHECK(output.size(1) == hidden_size, "output hidden dimension must match combine_buffer.");
    TORCH_CHECK(top_k <= std::numeric_limits<int32_t>::max(), "top_k exceeds int32 range.");
    TORCH_CHECK(
        max_num_tokens_per_rank <= std::numeric_limits<int32_t>::max(), "max_num_tokens_per_rank exceeds int32 range.");
    TORCH_CHECK(hidden_size <= std::numeric_limits<int32_t>::max(), "hidden_size exceeds int32 range.");
    TORCH_CHECK(local_num_tokens <= std::numeric_limits<int32_t>::max(), "local_num_tokens exceeds int32 range.");

    if (local_num_tokens == 0)
    {
        return;
    }

    auto const& stream = at::cuda::getCurrentCUDAStream(combine_buffer.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM6ReduceCombineBufferOut(
        reinterpret_cast<__nv_bfloat16 const*>(combine_buffer.data_ptr()), output.data_ptr<float>(),
        static_cast<int32_t>(top_k), static_cast<int32_t>(local_num_tokens),
        static_cast<int32_t>(max_num_tokens_per_rank), static_cast<int32_t>(hidden_size), stream);
}

void mega_moe_m6_reduce_combine_buffer_bf16_out(
    torch::Tensor const& combine_buffer, torch::Tensor const& output, int64_t const local_num_tokens)
{
    TORCH_CHECK(combine_buffer.is_cuda(), "combine_buffer must be a CUDA tensor.");
    TORCH_CHECK(combine_buffer.dim() == 3, "combine_buffer must be 3D [top_k, max_tokens_per_rank, hidden_size].");
    TORCH_CHECK(combine_buffer.scalar_type() == torch::kBFloat16, "combine_buffer must be bfloat16.");
    TORCH_CHECK(combine_buffer.is_contiguous(), "combine_buffer must be contiguous.");
    TORCH_CHECK(local_num_tokens >= 0, "local_num_tokens must be non-negative.");

    int64_t const top_k = combine_buffer.size(0);
    int64_t const max_num_tokens_per_rank = combine_buffer.size(1);
    int64_t const hidden_size = combine_buffer.size(2);
    TORCH_CHECK(top_k > 0, "combine_buffer top_k dimension must be positive.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "combine_buffer max-token dimension must be positive.");
    TORCH_CHECK(hidden_size > 0, "combine_buffer hidden dimension must be positive.");
    TORCH_CHECK(local_num_tokens <= max_num_tokens_per_rank,
        "local_num_tokens must not exceed combine_buffer max-token dimension.");
    TORCH_CHECK(output.is_cuda() && output.get_device() == combine_buffer.get_device(),
        "output must be on the same CUDA device as combine_buffer.");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [max_tokens_per_rank, hidden_size].");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16.");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous.");
    TORCH_CHECK(output.size(0) >= local_num_tokens, "output must cover local_num_tokens rows.");
    TORCH_CHECK(output.size(1) == hidden_size, "output hidden dimension must match combine_buffer.");
    TORCH_CHECK(top_k <= std::numeric_limits<int32_t>::max(), "top_k exceeds int32 range.");
    TORCH_CHECK(
        max_num_tokens_per_rank <= std::numeric_limits<int32_t>::max(), "max_num_tokens_per_rank exceeds int32 range.");
    TORCH_CHECK(hidden_size <= std::numeric_limits<int32_t>::max(), "hidden_size exceeds int32 range.");
    TORCH_CHECK(local_num_tokens <= std::numeric_limits<int32_t>::max(), "local_num_tokens exceeds int32 range.");

    if (local_num_tokens == 0)
    {
        return;
    }

    auto const& stream = at::cuda::getCurrentCUDAStream(combine_buffer.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM6ReduceCombineBufferBf16Out(
        reinterpret_cast<__nv_bfloat16 const*>(combine_buffer.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<int32_t>(top_k),
        static_cast<int32_t>(local_num_tokens), static_cast<int32_t>(max_num_tokens_per_rank),
        static_cast<int32_t>(hidden_size), stream);
}

void mega_moe_m6_reduce_token_major_combine_buffer_bf16_out(
    torch::Tensor const& combine_buffer, torch::Tensor const& output, int64_t const local_num_tokens)
{
    TORCH_CHECK(combine_buffer.is_cuda(), "combine_buffer must be a CUDA tensor.");
    TORCH_CHECK(combine_buffer.dim() == 3, "combine_buffer must be 3D [max_tokens_per_rank, top_k, hidden_size].");
    TORCH_CHECK(combine_buffer.scalar_type() == torch::kBFloat16, "combine_buffer must be bfloat16.");
    TORCH_CHECK(combine_buffer.is_contiguous(), "combine_buffer must be contiguous.");
    TORCH_CHECK(local_num_tokens >= 0, "local_num_tokens must be non-negative.");

    int64_t const max_num_tokens_per_rank = combine_buffer.size(0);
    int64_t const top_k = combine_buffer.size(1);
    int64_t const hidden_size = combine_buffer.size(2);
    TORCH_CHECK(top_k > 0, "combine_buffer top_k dimension must be positive.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "combine_buffer max-token dimension must be positive.");
    TORCH_CHECK(hidden_size > 0, "combine_buffer hidden dimension must be positive.");
    TORCH_CHECK(local_num_tokens <= max_num_tokens_per_rank,
        "local_num_tokens must not exceed combine_buffer max-token dimension.");
    TORCH_CHECK(output.is_cuda() && output.get_device() == combine_buffer.get_device(),
        "output must be on the same CUDA device as combine_buffer.");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [max_tokens_per_rank, hidden_size].");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16.");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous.");
    TORCH_CHECK(output.size(0) >= local_num_tokens, "output must cover local_num_tokens rows.");
    TORCH_CHECK(output.size(1) == hidden_size, "output hidden dimension must match combine_buffer.");
    TORCH_CHECK(top_k <= std::numeric_limits<int32_t>::max(), "top_k exceeds int32 range.");
    TORCH_CHECK(
        max_num_tokens_per_rank <= std::numeric_limits<int32_t>::max(), "max_num_tokens_per_rank exceeds int32 range.");
    TORCH_CHECK(hidden_size <= std::numeric_limits<int32_t>::max(), "hidden_size exceeds int32 range.");
    TORCH_CHECK(local_num_tokens <= std::numeric_limits<int32_t>::max(), "local_num_tokens exceeds int32 range.");

    if (local_num_tokens == 0)
    {
        return;
    }

    auto const& stream = at::cuda::getCurrentCUDAStream(combine_buffer.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM6ReduceTokenMajorCombineBufferBf16Out(
        reinterpret_cast<__nv_bfloat16 const*>(combine_buffer.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<int32_t>(top_k),
        static_cast<int32_t>(local_num_tokens), static_cast<int32_t>(max_num_tokens_per_rank),
        static_cast<int32_t>(hidden_size), stream);
}

torch::Tensor mega_moe_m6_reduce_combine_buffer(torch::Tensor const& combine_buffer, int64_t const local_num_tokens)
{
    TORCH_CHECK(combine_buffer.is_cuda(), "combine_buffer must be a CUDA tensor.");
    TORCH_CHECK(combine_buffer.dim() == 3, "combine_buffer must be 3D [top_k, max_tokens_per_rank, hidden_size].");
    TORCH_CHECK(combine_buffer.scalar_type() == torch::kBFloat16, "combine_buffer must be bfloat16.");
    TORCH_CHECK(combine_buffer.is_contiguous(), "combine_buffer must be contiguous.");
    TORCH_CHECK(local_num_tokens >= 0, "local_num_tokens must be non-negative.");

    int64_t const top_k = combine_buffer.size(0);
    int64_t const max_num_tokens_per_rank = combine_buffer.size(1);
    int64_t const hidden_size = combine_buffer.size(2);
    TORCH_CHECK(top_k > 0, "combine_buffer top_k dimension must be positive.");
    TORCH_CHECK(max_num_tokens_per_rank > 0, "combine_buffer max-token dimension must be positive.");
    TORCH_CHECK(hidden_size > 0, "combine_buffer hidden dimension must be positive.");
    TORCH_CHECK(local_num_tokens <= max_num_tokens_per_rank,
        "local_num_tokens must not exceed combine_buffer max-token dimension.");
    TORCH_CHECK(top_k <= std::numeric_limits<int32_t>::max(), "top_k exceeds int32 range.");
    TORCH_CHECK(
        max_num_tokens_per_rank <= std::numeric_limits<int32_t>::max(), "max_num_tokens_per_rank exceeds int32 range.");
    TORCH_CHECK(hidden_size <= std::numeric_limits<int32_t>::max(), "hidden_size exceeds int32 range.");
    TORCH_CHECK(local_num_tokens <= std::numeric_limits<int32_t>::max(), "local_num_tokens exceeds int32 range.");

    auto output = torch::empty({local_num_tokens, hidden_size}, combine_buffer.options().dtype(torch::kFloat32));
    if (local_num_tokens == 0)
    {
        return output;
    }

    auto const& stream = at::cuda::getCurrentCUDAStream(combine_buffer.get_device());
    tensorrt_llm::kernels::cute_dsl::megaMoeM6ReduceCombineBuffer(
        reinterpret_cast<__nv_bfloat16 const*>(combine_buffer.data_ptr()), output.data_ptr<float>(),
        static_cast<int32_t>(top_k), static_cast<int32_t>(local_num_tokens),
        static_cast<int32_t>(max_num_tokens_per_rank), static_cast<int32_t>(hidden_size), stream);
    return output;
}

// Activation

torch::Tensor moe_swiglu(torch::Tensor const& input, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& num_non_exiting_tiles, int64_t const tile_tokens_dim)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(input.size(1) % 2 == 0, "input.size(1) must be even.");
    int64_t const max_num_permuted_tokens = input.size(0);
    int64_t const interm_size = input.size(1) / 2;

    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const num_tiles = tile_idx_to_mn_limit.size(0);
    TORCH_CHECK(max_num_permuted_tokens == tile_tokens_dim * num_tiles,
        "max_num_permuted_tokens must be equal to tile_tokens_dim * num_tiles.");

    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    auto output
        = torch::empty({max_num_permuted_tokens, interm_size}, torch::dtype(input.scalar_type()).device(torch::kCUDA));
    tensorrt_llm::kernels::cutlass_kernels::ActivationParams activation_params{
        tensorrt_llm::kernels::cutlass_kernels::ActivationType::Swiglu};

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define DISPATCH_MOE_ACTIVATION(InputType, OutputType, SFType)                                                         \
    tensorrt_llm::kernels::cute_dsl::moeActivation<InputType, OutputType, SFType>(                                     \
        static_cast<InputType*>(input.data_ptr()), static_cast<OutputType*>(output.data_ptr()), nullptr, nullptr,      \
        tile_idx_to_mn_limit.data_ptr<int32_t>(), num_non_exiting_tiles.data_ptr<int32_t>(), activation_params,        \
        max_num_permuted_tokens, interm_size, tile_tokens_dim, stream)

    if (input.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_ACTIVATION(half, half, uint8_t);
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_ACTIVATION(__nv_bfloat16, __nv_bfloat16, uint8_t);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }

#undef DISPATCH_MOE_ACTIVATION

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> moe_swiglu_nvfp4_quantize(torch::Tensor const& input,
    torch::Tensor const& global_sf, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& num_non_exiting_tiles, int64_t const tile_tokens_dim)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    TORCH_CHECK(input.size(1) % 2 == 0, "input.size(1) must be even.");
    int64_t const max_num_permuted_tokens = input.size(0);
    int64_t const interm_size = input.size(1) / 2;

    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const num_tiles = tile_idx_to_mn_limit.size(0);
    TORCH_CHECK(max_num_permuted_tokens == tile_tokens_dim * num_tiles,
        "max_num_permuted_tokens must be equal to tile_tokens_dim * num_tiles.");

    TORCH_CHECK(global_sf.numel() == 1, "global_sf must have 1 element.");
    TORCH_CHECK(global_sf.scalar_type() == torch::kFloat32, "global_sf must be float32.");
    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    auto output = torch::empty(
        {max_num_permuted_tokens, interm_size / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(torch::kCUDA));
    int64_t constexpr kSFVecSize = 16;
    auto output_sf = torch::empty(
        {max_num_permuted_tokens * interm_size / kSFVecSize}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    tensorrt_llm::kernels::cutlass_kernels::ActivationParams activation_params{
        tensorrt_llm::kernels::cutlass_kernels::ActivationType::Swiglu};

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define DISPATCH_MOE_ACTIVATION(InputType, OutputType, SFType)                                                         \
    tensorrt_llm::kernels::cute_dsl::moeActivation<InputType, OutputType, SFType>(                                     \
        static_cast<InputType*>(input.data_ptr()), static_cast<OutputType*>(output.data_ptr()),                        \
        global_sf.data_ptr<float>(), static_cast<SFType*>(output_sf.data_ptr()),                                       \
        tile_idx_to_mn_limit.data_ptr<int32_t>(), num_non_exiting_tiles.data_ptr<int32_t>(), activation_params,        \
        max_num_permuted_tokens, interm_size, tile_tokens_dim, stream)

    if (input.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_ACTIVATION(half, __nv_fp4_e2m1, uint8_t);
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_ACTIVATION(__nv_bfloat16, __nv_fp4_e2m1, uint8_t);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }

#undef DISPATCH_MOE_ACTIVATION

    return {output, output_sf};
}

torch::Tensor moe_gelu(torch::Tensor const& input, torch::Tensor const& tile_idx_to_mn_limit,
    torch::Tensor const& num_non_exiting_tiles, int64_t const tile_tokens_dim)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D.");
    int64_t const max_num_permuted_tokens = input.size(0);
    int64_t const interm_size = input.size(1);

    TORCH_CHECK(tile_idx_to_mn_limit.dim() == 1, "tile_idx_to_mn_limit must be 1D.");
    TORCH_CHECK(tile_idx_to_mn_limit.scalar_type() == torch::kInt32, "tile_idx_to_mn_limit must be int32.");
    int64_t const num_tiles = tile_idx_to_mn_limit.size(0);
    TORCH_CHECK(max_num_permuted_tokens == tile_tokens_dim * num_tiles,
        "max_num_permuted_tokens must be equal to tile_tokens_dim * num_tiles.");

    TORCH_CHECK(num_non_exiting_tiles.numel() == 1, "num_non_exiting_tiles must have 1 element.");
    TORCH_CHECK(num_non_exiting_tiles.scalar_type() == torch::kInt32, "num_non_exiting_tiles must be int32.");

    auto output
        = torch::empty({max_num_permuted_tokens, interm_size}, torch::dtype(input.scalar_type()).device(torch::kCUDA));
    tensorrt_llm::kernels::cutlass_kernels::ActivationParams activation_params{
        tensorrt_llm::kernels::cutlass_kernels::ActivationType::Gelu};

    auto const& stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define DISPATCH_MOE_ACTIVATION(InputType, OutputType, SFType)                                                         \
    tensorrt_llm::kernels::cute_dsl::moeActivation<InputType, OutputType, SFType>(                                     \
        static_cast<InputType*>(input.data_ptr()), static_cast<OutputType*>(output.data_ptr()), nullptr, nullptr,      \
        tile_idx_to_mn_limit.data_ptr<int32_t>(), num_non_exiting_tiles.data_ptr<int32_t>(), activation_params,        \
        max_num_permuted_tokens, interm_size, tile_tokens_dim, stream)

    if (input.scalar_type() == torch::kHalf)
    {
        DISPATCH_MOE_ACTIVATION(half, half, uint8_t);
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        DISPATCH_MOE_ACTIVATION(__nv_bfloat16, __nv_bfloat16, uint8_t);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }

#undef DISPATCH_MOE_ACTIVATION

    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_topk_sort(Tensor routing_logits, Tensor? routing_bias, int num_experts, int top_k, int? n_group, "
        "int? topk_group, int local_expert_offset, int local_num_experts, float? routed_scaling_factor, int "
        "tile_tokens_dim, int routing_method_type) -> Tensor[]");
    m.def(
        "moe_sort(Tensor token_selected_experts, Tensor token_final_scales, int num_experts, int top_k, "
        "int local_expert_offset, int local_num_experts, int tile_tokens_dim) -> Tensor[]");
    m.def(
        "moe_permute(Tensor input, Tensor? input_sf, Tensor tile_idx_to_mn_limit, Tensor permuted_idx_to_expanded_idx, "
        "Tensor num_non_exiting_tiles, int tile_tokens_dim, int top_k) -> (Tensor, Tensor?)");
    m.def(
        "moe_unpermute_inplace(Tensor permuted_input, Tensor(a!) output, Tensor expanded_idx_to_permuted_idx, Tensor "
        "topk_scales) -> ()");
    m.def("moe_unpermute(Tensor permuted_input, Tensor expanded_idx_to_permuted_idx, Tensor topk_scales) -> Tensor");
    m.def(
        "moe_output_memset_inplace(Tensor(a!) input, Tensor tile_idx_to_mn_limit, Tensor expanded_idx_to_permuted_idx, "
        "Tensor permuted_idx_to_expanded_idx, Tensor num_non_exiting_tiles, int tile_tokens_dim, int top_k, int "
        "ep_size, bool enable_alltoall = False) -> ()");
    m.def(
        "mega_moe_m5_materialize_from_moe_sort(Tensor input, Tensor input_sf, Tensor topk_scales, Tensor "
        "token_offsets, "
        "Tensor tile_idx_to_mn_limit, Tensor permuted_idx_to_expanded_idx, Tensor num_non_exiting_tiles, "
        "Tensor(a!) l1_acts_pool, Tensor(b!) l1_acts_sf_pool, Tensor(c!) l1_topk_weights_pool, "
        "Tensor(d!) token_src_metadata, Tensor(e!) l1_arrival_count, int tile_tokens_dim) -> ()");
    m.def(
        "mega_moe_m5_materialize_direct_from_moe_sort(Tensor input, Tensor input_sf, Tensor topk_scales, Tensor "
        "token_offsets, Tensor tile_idx_to_mn_limit, Tensor permuted_idx_to_expanded_idx, "
        "Tensor num_non_exiting_tiles, Tensor(a!) l1_acts_pool, Tensor(b!) l1_acts_sf_pool, "
        "Tensor(c!) l1_topk_weights_pool, Tensor(d!) token_src_metadata, Tensor(e!) l1_arrival_count, "
        "Tensor(f!) active_pool_slots, Tensor(g!) active_combine_rows, Tensor(h!) active_route_count, "
        "Tensor(i!) output_permuted_idx_to_expanded_idx, Tensor(j!) output_token_final_scales, "
        "int tile_tokens_dim, int max_num_tokens_per_rank, int combine_layout_rows) -> ()");
    m.def(
        "mega_moe_m5_materialize_direct_from_ranked_topk(Tensor input, Tensor input_sf, Tensor topk_idx, Tensor "
        "topk_scales, Tensor token_counts, Tensor(a!) expert_route_offsets, Tensor(b!) l1_acts_pool, "
        "Tensor(c!) l1_acts_sf_pool, Tensor(d!) l1_topk_weights_pool, Tensor(e!) token_src_metadata, "
        "Tensor(f!) l1_arrival_count, Tensor(g!) active_pool_slots, Tensor(h!) active_combine_rows, "
        "Tensor(i!) active_route_count, Tensor(j!) output_permuted_idx_to_expanded_idx, "
        "Tensor(k!) output_token_final_scales, Tensor(l!) tile_idx_to_expert_idx, Tensor(m!) tile_idx_to_mn_limit, "
        "Tensor(n!) num_non_exiting_tiles, int local_rank, int tile_tokens_dim, int combine_layout_rows) -> ()");
    m.def(
        "mega_moe_stage_dispatch_inputs(Tensor input, Tensor input_sf, Tensor topk_idx, Tensor topk_scales, "
        "Tensor(a!) input_buffer, Tensor(b!) input_sf_buffer, Tensor(c!) topk_idx_buffer, "
        "Tensor(d!) topk_scales_buffer) -> ()");
    m.def(
        "mega_moe_m5_init_direct_input_route_metadata(Tensor(a!) expert_route_offsets, "
        "Tensor(b!) tile_idx_to_expert_idx, Tensor(c!) tile_idx_to_mn_limit, "
        "Tensor(d!) num_non_exiting_tiles) -> ()");
    m.def(
        "mega_moe_m5_build_direct_input_route_from_ranked_topk(Tensor input, Tensor input_sf, Tensor topk_idx, "
        "Tensor topk_scales, Tensor token_counts, Tensor(a!) direct_input, Tensor(b!) direct_input_sf, "
        "Tensor(c!) expert_route_offsets, Tensor(d!) expert_route_base_offsets, Tensor(e!) token_id_mapping, "
        "Tensor(f!) output_permuted_idx_to_expanded_idx, Tensor(g!) output_token_final_scales, "
        "Tensor(h!) tile_idx_to_expert_idx, Tensor(i!) tile_idx_to_mn_limit, Tensor(j!) num_non_exiting_tiles, "
        "int local_rank, int tile_tokens_dim, int combine_layout_rows, bool direct_atomic_output = False, "
        "bool direct_token_major_output = False) -> ()");
    m.def(
        "mega_moe_m5_materialize_direct_from_topk(Tensor input, Tensor input_sf, Tensor topk_idx, Tensor "
        "topk_scales, Tensor token_offsets, Tensor expert_recv_count_sum, Tensor(a!) expert_route_offsets, "
        "Tensor(b!) l1_acts_pool, Tensor(c!) l1_acts_sf_pool, Tensor(d!) l1_topk_weights_pool, "
        "Tensor(e!) token_src_metadata, Tensor(f!) l1_arrival_count, Tensor(g!) active_pool_slots, "
        "Tensor(h!) active_combine_rows, Tensor(i!) active_route_count, "
        "Tensor(j!) output_permuted_idx_to_expanded_idx, Tensor(k!) output_token_final_scales, "
        "Tensor(l!) tile_idx_to_expert_idx, Tensor(m!) tile_idx_to_mn_limit, Tensor(n!) num_non_exiting_tiles, "
        "int local_rank, int tile_tokens_dim, int max_num_tokens_per_rank, int combine_layout_rows) -> ()");
    m.def(
        "mega_moe_m6_reduce_combine_buffer_out(Tensor combine_buffer, Tensor(a!) output, int local_num_tokens) -> ()");
    m.def(
        "mega_moe_m6_reduce_combine_buffer_bf16_out(Tensor combine_buffer, Tensor(a!) output, int local_num_tokens) -> "
        "()");
    m.def(
        "mega_moe_m6_reduce_token_major_combine_buffer_bf16_out(Tensor combine_buffer, Tensor(a!) output, "
        "int local_num_tokens) -> ()");
    m.def("mega_moe_m6_reduce_combine_buffer(Tensor combine_buffer, int local_num_tokens) -> Tensor");
    m.def(
        "moe_swiglu(Tensor input, Tensor tile_idx_to_mn_limit, Tensor num_non_exiting_tiles, "
        "int tile_tokens_dim) -> Tensor");
    m.def(
        "moe_swiglu_nvfp4_quantize(Tensor input, Tensor global_sf, Tensor tile_idx_to_mn_limit, Tensor "
        "num_non_exiting_tiles, int tile_tokens_dim) -> (Tensor, Tensor)");
    m.def(
        "moe_gelu(Tensor input, Tensor tile_idx_to_mn_limit, Tensor num_non_exiting_tiles, "
        "int tile_tokens_dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_topk_sort", &tensorrt_llm::torch_ext::moe_topk_sort);
    m.impl("moe_sort", &tensorrt_llm::torch_ext::moe_sort);
    m.impl("moe_permute", &tensorrt_llm::torch_ext::moe_permute);
    m.impl("moe_unpermute_inplace", &tensorrt_llm::torch_ext::moe_unpermute_inplace);
    m.impl("moe_unpermute", &tensorrt_llm::torch_ext::moe_unpermute);
    m.impl("moe_output_memset_inplace", &tensorrt_llm::torch_ext::moe_output_memset_inplace);
    m.impl("mega_moe_m5_materialize_from_moe_sort", &tensorrt_llm::torch_ext::mega_moe_m5_materialize_from_moe_sort);
    m.impl("mega_moe_m5_materialize_direct_from_moe_sort",
        &tensorrt_llm::torch_ext::mega_moe_m5_materialize_direct_from_moe_sort);
    m.impl("mega_moe_m5_materialize_direct_from_ranked_topk",
        &tensorrt_llm::torch_ext::mega_moe_m5_materialize_direct_from_ranked_topk);
    m.impl("mega_moe_stage_dispatch_inputs", &tensorrt_llm::torch_ext::mega_moe_stage_dispatch_inputs);
    m.impl("mega_moe_m5_init_direct_input_route_metadata",
        &tensorrt_llm::torch_ext::mega_moe_m5_init_direct_input_route_metadata);
    m.impl("mega_moe_m5_build_direct_input_route_from_ranked_topk",
        &tensorrt_llm::torch_ext::mega_moe_m5_build_direct_input_route_from_ranked_topk);
    m.impl(
        "mega_moe_m5_materialize_direct_from_topk", &tensorrt_llm::torch_ext::mega_moe_m5_materialize_direct_from_topk);
    m.impl("mega_moe_m6_reduce_combine_buffer_out", &tensorrt_llm::torch_ext::mega_moe_m6_reduce_combine_buffer_out);
    m.impl("mega_moe_m6_reduce_combine_buffer_bf16_out",
        &tensorrt_llm::torch_ext::mega_moe_m6_reduce_combine_buffer_bf16_out);
    m.impl("mega_moe_m6_reduce_token_major_combine_buffer_bf16_out",
        &tensorrt_llm::torch_ext::mega_moe_m6_reduce_token_major_combine_buffer_bf16_out);
    m.impl("mega_moe_m6_reduce_combine_buffer", &tensorrt_llm::torch_ext::mega_moe_m6_reduce_combine_buffer);
    m.impl("moe_swiglu", &tensorrt_llm::torch_ext::moe_swiglu);
    m.impl("moe_swiglu_nvfp4_quantize", &tensorrt_llm::torch_ext::moe_swiglu_nvfp4_quantize);
    m.impl("moe_gelu", &tensorrt_llm::torch_ext::moe_gelu);
}
