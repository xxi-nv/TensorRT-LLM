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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization.h"

#include <algorithm>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cute/numeric/numeric_types.hpp>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cute_dsl
{
namespace
{
using ElemCopyType = uint4;
using SFCopyType = uint32_t;
using ActivationType = tensorrt_llm::kernels::cutlass_kernels::ActivationType;

template <typename T>
auto constexpr bitsPerElem()
{
#ifdef ENABLE_FP4
    return std::is_same_v<T, __nv_fp4_e2m1> ? 4 : cute::sizeof_bits_v<T>;
#else
    return cute::sizeof_bits_v<T>;
#endif
}

template <typename T>
auto constexpr elemPerCopy()
{
    return bitsPerElem<ElemCopyType>() / bitsPerElem<T>();
}

template <typename T>
auto constexpr sfElemPerCopy()
{
    return bitsPerElem<SFCopyType>() / bitsPerElem<T>();
}
} // namespace

template <typename InputType, typename SFType, int32_t kSFVecSize, int32_t kThreadsPerBlock>
__global__ void moePermuteKernel(InputType const* input, InputType* permuted_output, SFType const* input_sf,
    SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size)
{
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    int32_t constexpr kSFElemPerCopy = sfElemPerCopy<SFType>();
    // Need int64_t to prevent overflow when computing pointer offsets.
    int64_t const kCopyPerToken = hidden_size / kElemPerCopy;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const tile_idx = permuted_idx / tile_size;
        if (permuted_idx >= tile_idx_to_mn_limit[tile_idx])
        {
            continue;
        }
        int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
        int32_t const token_idx = expanded_idx / top_k;

        auto const* src_ptr = reinterpret_cast<ElemCopyType const*>(input) + token_idx * kCopyPerToken;
        auto* dst_ptr = reinterpret_cast<ElemCopyType*>(permuted_output) + permuted_idx * kCopyPerToken;
        for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
        {
            dst_ptr[i] = src_ptr[i];
        }

#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<InputType, __nv_fp4_e2m1>)
        {
            int32_t const sf_hidden_size = hidden_size / kSFVecSize;
            int64_t const kSFCopyPerToken = sf_hidden_size / kSFElemPerCopy;
            auto const* sf_src_ptr = reinterpret_cast<SFCopyType const*>(input_sf);
            auto* sf_dst_ptr = reinterpret_cast<SFCopyType*>(permuted_sf);
            for (int32_t i = threadIdx.x; i < kSFCopyPerToken; i += kThreadsPerBlock)
            {
                // input_sf is not swizzled, while permuted_sf is swizzled.
                int64_t const src_offset = token_idx * kSFCopyPerToken + i;
                int64_t const dst_offset = get_sf_out_offset_128x4(/* batchIdx= */ std::nullopt, permuted_idx,
                                               i * kSFElemPerCopy, /* numRows= */ std::nullopt, sf_hidden_size)
                    / kSFElemPerCopy;

                sf_dst_ptr[dst_offset] = sf_src_ptr[src_offset];
            }
        }
#endif
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_output, SFType const* input_sf, SFType* permuted_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens, int32_t const hidden_size,
    int32_t const top_k, int32_t const tile_size, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kSFVecSize = 16;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.", kElemPerCopy);

#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<InputType, __nv_fp4_e2m1>)
    {
        int32_t constexpr kSFMAlignment = 128;
        int32_t constexpr kSFKAlignment = 4;
        int32_t constexpr kSFElemPerCopy = sfElemPerCopy<SFType>();
        static_assert(kSFElemPerCopy == kSFKAlignment);
        TLLM_CHECK_WITH_INFO(max_num_permuted_tokens % kSFMAlignment == 0,
            "max_num_permuted_tokens must be divisible by %d.", kSFMAlignment);
        TLLM_CHECK_WITH_INFO(hidden_size % (kSFVecSize * kSFKAlignment) == 0, "hidden_size must be divisible by %d.",
            kSFVecSize * kSFKAlignment);
        TLLM_CHECK_WITH_INFO(input_sf != nullptr, "input_sf is required for NVFP4.");
        TLLM_CHECK_WITH_INFO(permuted_sf != nullptr, "permuted_sf is required for NVFP4.");
    }
#endif

    auto kernel = &moePermuteKernel<InputType, SFType, kSFVecSize, kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
    int32_t const threads = kThreadsPerBlock;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, permuted_output, input_sf, permuted_sf, tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx, num_non_exiting_tiles, hidden_size, top_k, tile_size);
}

#define INSTANTIATE_MOE_PERMUTE(InputType, SFType)                                                                     \
    template void moePermute<InputType, SFType>(InputType const* input, InputType* permuted_output,                    \
        SFType const* input_sf, SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit,                              \
        int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,                             \
        int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k,                         \
        int32_t const tile_size, cudaStream_t stream)

INSTANTIATE_MOE_PERMUTE(half, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_PERMUTE(__nv_bfloat16, uint8_t);
#endif
#ifdef ENABLE_FP8
INSTANTIATE_MOE_PERMUTE(__nv_fp8_e4m3, uint8_t);
#endif
#ifdef ENABLE_FP4
INSTANTIATE_MOE_PERMUTE(__nv_fp4_e2m1, uint8_t);
#endif
#undef INSTANTIATE_MOE_PERMUTE

template <typename InputType, typename TopKScaleType, int32_t kThreadsPerBlock>
__global__ void moeUnpermuteKernel(InputType const* permuted_input, InputType* output,
    int32_t const* expanded_idx_to_permuted_idx, TopKScaleType const* topk_scales, int32_t const hidden_size,
    int32_t const top_k)
{
    using AccumType = float;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    // Need int64_t to prevent overflow when computing pointer offsets.
    int64_t const kCopyPerToken = hidden_size / kElemPerCopy;
    InputType rmem[kElemPerCopy];
    AccumType rmemAccum[kElemPerCopy];

    int32_t const token_idx = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    auto* dst_ptr = reinterpret_cast<ElemCopyType*>(output) + token_idx * kCopyPerToken;
    for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
    {
#pragma unroll
        for (int32_t j = 0; j < kElemPerCopy; j++)
        {
            rmemAccum[j] = 0;
        }
        for (int32_t k = 0; k < top_k; k++)
        {
            int32_t const permuted_idx = expanded_idx_to_permuted_idx[token_idx * top_k + k];
            if (permuted_idx < 0)
            {
                continue;
            }
            auto const* src_ptr = reinterpret_cast<ElemCopyType const*>(permuted_input) + permuted_idx * kCopyPerToken;
            *reinterpret_cast<ElemCopyType*>(rmem) = src_ptr[i];
            TopKScaleType const scale = topk_scales[token_idx * top_k + k];

#pragma unroll
            for (int32_t j = 0; j < kElemPerCopy; j++)
            {
                rmemAccum[j] += static_cast<AccumType>(rmem[j]) * static_cast<AccumType>(scale);
            }
        }
#pragma unroll
        for (int32_t j = 0; j < kElemPerCopy; j++)
        {
            rmem[j] = static_cast<InputType>(rmemAccum[j]);
        }
        dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename TopKScaleType>
void moeUnpermute(InputType const* permuted_input, InputType* output, int32_t const* expanded_idx_to_permuted_idx,
    TopKScaleType const* topk_scales, int32_t const num_tokens, int32_t const hidden_size, int32_t const top_k,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.", kElemPerCopy);

    int32_t const blocks = num_tokens;
    int32_t const threads = kThreadsPerBlock;

    auto kernel = &moeUnpermuteKernel<InputType, TopKScaleType, kThreadsPerBlock>;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(
        &config, kernel, permuted_input, output, expanded_idx_to_permuted_idx, topk_scales, hidden_size, top_k);
}

#define INSTANTIATE_MOE_UNPERMUTE(InputType, TopKScaleType)                                                            \
    template void moeUnpermute<InputType>(InputType const* permuted_input, InputType* output,                          \
        int32_t const* expanded_idx_to_permuted_idx, TopKScaleType const* topk_scales, int32_t const num_tokens,       \
        int32_t const hidden_size, int32_t const top_k, cudaStream_t stream)

INSTANTIATE_MOE_UNPERMUTE(half, float);
INSTANTIATE_MOE_UNPERMUTE(half, half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_UNPERMUTE(__nv_bfloat16, float);
INSTANTIATE_MOE_UNPERMUTE(__nv_bfloat16, __nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_UNPERMUTE

template <typename InputType, int32_t kThreadsPerBlock>
__global__ void moeOutputMemsetKernel(InputType* input, int32_t const* tile_idx_to_mn_limit,
    int32_t const* expanded_idx_to_permuted_idx, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size)
{
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    int64_t const kCopyPerToken = hidden_size / kElemPerCopy;

    InputType rmem[kElemPerCopy];
#pragma unroll
    for (int32_t j = 0; j < kElemPerCopy; j++)
    {
        rmem[j] = 0;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const tile_idx = permuted_idx / tile_size;
        if (permuted_idx >= tile_idx_to_mn_limit[tile_idx])
        {
            continue;
        }
        int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
        int32_t const token_idx = expanded_idx / top_k;
        int32_t const topk_idx = expanded_idx % top_k;

        bool is_first_in_topk = true;
        for (int32_t k = 0; k < topk_idx; k++)
        {
            if (expanded_idx_to_permuted_idx[token_idx * top_k + k] >= 0)
            {
                is_first_in_topk = false;
                break;
            }
        }
        if (!is_first_in_topk)
        {
            continue;
        }

        auto* dst_ptr = reinterpret_cast<ElemCopyType*>(input) + token_idx * kCopyPerToken;
        for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
        {
            dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType>
void moeOutputMemset(InputType* input, int32_t const* tile_idx_to_mn_limit, int32_t const* expanded_idx_to_permuted_idx,
    int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
    int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.", kElemPerCopy);

    auto kernel = &moeOutputMemsetKernel<InputType, kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
    int32_t const threads = kThreadsPerBlock;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx, num_non_exiting_tiles, hidden_size, top_k, tile_size);
}

#define INSTANTIATE_MOE_OUTPUT_MEMSET(InputType)                                                                       \
    template void moeOutputMemset<InputType>(InputType * input, int32_t const* tile_idx_to_mn_limit,                   \
        int32_t const* expanded_idx_to_permuted_idx, int32_t const* permuted_idx_to_expanded_idx,                      \
        int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens, int32_t const hidden_size,        \
        int32_t const top_k, int32_t const tile_size, cudaStream_t stream)

INSTANTIATE_MOE_OUTPUT_MEMSET(half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_OUTPUT_MEMSET(__nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_OUTPUT_MEMSET

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5MaterializeFromMoeSortKernel(uint8_t const* input, uint8_t const* inputSf,
    float const* topKScales, int32_t const* tokenOffsets, int32_t const* tileIdxToMnLimit,
    int32_t const* permutedIdxToExpandedIdx, int32_t const* numNonExitingTiles, uint8_t* l1ActsPool,
    uint8_t* l1ActsSfPool, float* l1TopKWeightsPool, int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount,
    int64_t* activePoolSlots, int64_t* activeCombineRows, int32_t* activeRouteCount,
    int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales, int32_t const totalTokens,
    int32_t const epSize, int32_t const hiddenPackedSize, int32_t const sfHiddenSize, int32_t const topK,
    int32_t const tileSize, int32_t const numAvailablePoolSlots, int32_t const numPaddedSfPoolTokens,
    int32_t const l1ArrivalCountEntries, int32_t const maxNumTokensPerRank, int32_t const combineLayoutRows,
    int32_t const outputMappingRows, int32_t const outputScaleRows)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int32_t const maxActivePoolSlots = numNonExitingTiles[0] * tileSize;
    int32_t const numActivePoolSlots
        = maxActivePoolSlots < numAvailablePoolSlots ? maxActivePoolSlots : numAvailablePoolSlots;
    int32_t const sfBlockTokens = ((tileSize + 127) / 128) * 128;
    for (int32_t poolSlot = blockIdx.x; poolSlot < numActivePoolSlots; poolSlot += gridDim.x)
    {
        int32_t const tileIdx = poolSlot / tileSize;
        if (poolSlot >= tileIdxToMnLimit[tileIdx])
        {
            continue;
        }

        int32_t const expandedIdx = permutedIdxToExpandedIdx[poolSlot];
        if (expandedIdx < 0 || expandedIdx >= totalTokens * topK)
        {
            continue;
        }

        int32_t const tokenIdx = expandedIdx / topK;
        int32_t const topKIdx = expandedIdx - tokenIdx * topK;
        int32_t sourceRank = 0;
        while (sourceRank + 1 < epSize && tokenIdx >= tokenOffsets[sourceRank + 1])
        {
            ++sourceRank;
        }
        int32_t const sourceTokenIdx = tokenIdx - tokenOffsets[sourceRank];
        if (sourceTokenIdx < 0 || sourceTokenIdx >= tokenOffsets[sourceRank + 1] - tokenOffsets[sourceRank])
        {
            continue;
        }

        for (int32_t i = threadIdx.x; i < hiddenPackedSize; i += kThreadsPerBlock)
        {
            l1ActsPool[static_cast<int64_t>(poolSlot) * hiddenPackedSize + i]
                = input[static_cast<int64_t>(tokenIdx) * hiddenPackedSize + i];
        }

        int32_t const sfSlot = tileIdx * sfBlockTokens + poolSlot % tileSize;
        if (sfSlot < numPaddedSfPoolTokens)
        {
            for (int32_t i = threadIdx.x; i < sfHiddenSize; i += kThreadsPerBlock)
            {
                l1ActsSfPool[static_cast<int64_t>(sfSlot) * sfHiddenSize + i]
                    = inputSf[static_cast<int64_t>(tokenIdx) * sfHiddenSize + i];
            }
        }

        if (threadIdx.x == 0)
        {
            float const topKScale = topKScales[static_cast<int64_t>(tokenIdx) * topK + topKIdx];
            l1TopKWeightsPool[poolSlot] = topKScale;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3] = sourceRank;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3 + 1] = sourceTokenIdx;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3 + 2] = topKIdx;
            if (tileIdx < l1ArrivalCountEntries)
            {
                atomicAdd(l1ArrivalCount + tileIdx, 1);
            }
            if (activeRouteCount != nullptr && activePoolSlots != nullptr && activeCombineRows != nullptr
                && maxNumTokensPerRank > 0 && combineLayoutRows > 0)
            {
                int32_t const combineRow = (sourceRank * topK + topKIdx) * maxNumTokensPerRank + sourceTokenIdx;
                if (combineRow >= 0 && combineRow < combineLayoutRows)
                {
                    int32_t const activeIdx = atomicAdd(activeRouteCount, 1);
                    activePoolSlots[activeIdx] = static_cast<int64_t>(poolSlot);
                    activeCombineRows[activeIdx] = static_cast<int64_t>(combineRow);
                    if (outputPermutedIdxToExpandedIdx != nullptr && poolSlot < outputMappingRows)
                    {
                        outputPermutedIdxToExpandedIdx[poolSlot] = combineRow;
                    }
                    if (outputTokenFinalScales != nullptr && combineRow < outputScaleRows)
                    {
                        outputTokenFinalScales[combineRow] = topKScale;
                    }
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void megaMoeM5MaterializeDirectFromMoeSort(uint8_t const* input, uint8_t const* inputSf, float const* topKScales,
    int32_t const* tokenOffsets, int32_t const* tileIdxToMnLimit, int32_t const* permutedIdxToExpandedIdx,
    int32_t const* numNonExitingTiles, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int64_t* activePoolSlots, int64_t* activeCombineRows,
    int32_t* activeRouteCount, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t const totalTokens, int32_t const epSize, int32_t const hiddenPackedSize, int32_t const sfHiddenSize,
    int32_t const topK, int32_t const tileSize, int32_t const numAvailablePoolSlots,
    int32_t const numPaddedSfPoolTokens, int32_t const l1ArrivalCountEntries, int32_t const maxNumTokensPerRank,
    int32_t const combineLayoutRows, int32_t const outputMappingRows, int32_t const outputScaleRows,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (numAvailablePoolSlots <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM5MaterializeFromMoeSortKernel<kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, numAvailablePoolSlots);

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, inputSf, topKScales, tokenOffsets, tileIdxToMnLimit,
        permutedIdxToExpandedIdx, numNonExitingTiles, l1ActsPool, l1ActsSfPool, l1TopKWeightsPool, tokenSrcMetadata,
        l1ArrivalCount, activePoolSlots, activeCombineRows, activeRouteCount, outputPermutedIdxToExpandedIdx,
        outputTokenFinalScales, totalTokens, epSize, hiddenPackedSize, sfHiddenSize, topK, tileSize,
        numAvailablePoolSlots, numPaddedSfPoolTokens, l1ArrivalCountEntries, maxNumTokensPerRank, combineLayoutRows,
        outputMappingRows, outputScaleRows);
}

void megaMoeM5MaterializeFromMoeSort(uint8_t const* input, uint8_t const* inputSf, float const* topKScales,
    int32_t const* tokenOffsets, int32_t const* tileIdxToMnLimit, int32_t const* permutedIdxToExpandedIdx,
    int32_t const* numNonExitingTiles, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int32_t const totalTokens, int32_t const epSize,
    int32_t const hiddenPackedSize, int32_t const sfHiddenSize, int32_t const topK, int32_t const tileSize,
    int32_t const numAvailablePoolSlots, int32_t const numPaddedSfPoolTokens, int32_t const l1ArrivalCountEntries,
    cudaStream_t stream)
{
    megaMoeM5MaterializeDirectFromMoeSort(input, inputSf, topKScales, tokenOffsets, tileIdxToMnLimit,
        permutedIdxToExpandedIdx, numNonExitingTiles, l1ActsPool, l1ActsSfPool, l1TopKWeightsPool, tokenSrcMetadata,
        l1ArrivalCount, nullptr, nullptr, nullptr, nullptr, nullptr, totalTokens, epSize, hiddenPackedSize,
        sfHiddenSize, topK, tileSize, numAvailablePoolSlots, numPaddedSfPoolTokens, l1ArrivalCountEntries, 0, 0, 0, 0,
        stream);
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5MaterializeDirectFromTopKKernel(uint8_t const* input, uint8_t const* inputSf,
    int64_t const* topKIdx, float const* topKScales, int32_t const* tokenOffsets, int64_t const* expertRecvCountSum,
    int32_t* expertRouteOffsets, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int64_t* activePoolSlots, int64_t* activeCombineRows,
    int32_t* activeRouteCount, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const totalTokens,
    int32_t const epSize, int32_t const localRank, int32_t const numExpertsPerRank, int32_t const hiddenPackedSize,
    int32_t const sfHiddenSize, int32_t const topK, int32_t const tileSize, int32_t const numPoolSlots,
    int32_t const numPaddedSfPoolTokens, int32_t const l1ArrivalCountEntries, int32_t const maxNumTokensPerRank,
    int32_t const combineLayoutRows, int32_t const outputMappingRows, int32_t const outputScaleRows,
    int32_t const routeLayoutCapacity)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int32_t const totalRoutes = totalTokens * topK;
    int32_t const sfBlockTokens = ((tileSize + 127) / 128) * 128;
    int32_t const firstLocalExpert = localRank * numExpertsPerRank;
    __shared__ int32_t routeOrdinal;

    for (int32_t routeIdx = blockIdx.x; routeIdx < totalRoutes; routeIdx += gridDim.x)
    {
        int32_t const tokenIdx = routeIdx / topK;
        int32_t const topKOrdinal = routeIdx - tokenIdx * topK;
        int64_t const selectedExpert64 = topKIdx[routeIdx];
        if (selectedExpert64 < firstLocalExpert || selectedExpert64 >= firstLocalExpert + numExpertsPerRank)
        {
            continue;
        }

        int32_t const localExpertIdx = static_cast<int32_t>(selectedExpert64) - firstLocalExpert;
        int64_t const numRoutes64 = expertRecvCountSum[localExpertIdx];
        int32_t const numRoutes = numRoutes64 > 0 ? static_cast<int32_t>(numRoutes64) : 0;
        int32_t const paddedRoutes = ((numRoutes + tileSize - 1) / tileSize) * tileSize;
        int32_t poolStart = 0;
        for (int32_t expertIdx = 0; expertIdx < localExpertIdx; ++expertIdx)
        {
            int64_t const previousRoutes64 = expertRecvCountSum[expertIdx];
            int32_t const previousRoutes = previousRoutes64 > 0 ? static_cast<int32_t>(previousRoutes64) : 0;
            poolStart += ((previousRoutes + tileSize - 1) / tileSize) * tileSize;
        }
        int32_t const poolEnd = poolStart + paddedRoutes;
        if (numRoutes <= 0 || poolEnd > numPoolSlots)
        {
            continue;
        }

        if (threadIdx.x == 0)
        {
            routeOrdinal = atomicAdd(expertRouteOffsets + localExpertIdx, 1);
        }
        __syncthreads();
        int32_t const ordinal = routeOrdinal;
        if (ordinal >= numRoutes)
        {
            continue;
        }

        int32_t const poolSlot = poolStart + ordinal;
        int32_t sourceRank = 0;
        while (sourceRank + 1 < epSize && tokenIdx >= tokenOffsets[sourceRank + 1])
        {
            ++sourceRank;
        }
        int32_t const sourceTokenIdx = tokenIdx - tokenOffsets[sourceRank];
        if (sourceTokenIdx < 0 || sourceTokenIdx >= tokenOffsets[sourceRank + 1] - tokenOffsets[sourceRank])
        {
            continue;
        }

        for (int32_t i = threadIdx.x; i < hiddenPackedSize; i += kThreadsPerBlock)
        {
            l1ActsPool[static_cast<int64_t>(poolSlot) * hiddenPackedSize + i]
                = input[static_cast<int64_t>(tokenIdx) * hiddenPackedSize + i];
        }

        int32_t const sfSlot = (poolSlot / tileSize) * sfBlockTokens + poolSlot % tileSize;
        if (sfSlot < numPaddedSfPoolTokens)
        {
            for (int32_t i = threadIdx.x; i < sfHiddenSize; i += kThreadsPerBlock)
            {
                l1ActsSfPool[static_cast<int64_t>(sfSlot) * sfHiddenSize + i]
                    = inputSf[static_cast<int64_t>(tokenIdx) * sfHiddenSize + i];
            }
        }

        if (threadIdx.x == 0)
        {
            int32_t const tileStart = poolStart / tileSize;
            int32_t const tileCount = paddedRoutes / tileSize;
            int32_t const tileOffset = ordinal / tileSize;
            int32_t const tileIdx = tileStart + tileOffset;
            int32_t const remainingRoutesInTile = numRoutes - tileOffset * tileSize;
            int32_t const tileRoutes = remainingRoutesInTile < tileSize ? remainingRoutesInTile : tileSize;
            if (tileIdx < routeLayoutCapacity)
            {
                tileIdxToExpertIdx[tileIdx] = localExpertIdx;
                tileIdxToMnLimit[tileIdx] = poolStart + tileOffset * tileSize + tileRoutes;
            }
            if (tileIdx < l1ArrivalCountEntries)
            {
                l1ArrivalCount[tileIdx] = tileRoutes;
            }
            if (tileCount > 0)
            {
                atomicMax(numNonExitingTiles, tileStart + tileCount);
            }

            float const topKScale = topKScales[routeIdx];
            l1TopKWeightsPool[poolSlot] = topKScale;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3] = sourceRank;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3 + 1] = sourceTokenIdx;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3 + 2] = topKOrdinal;

            int32_t const combineRow = (sourceRank * topK + topKOrdinal) * maxNumTokensPerRank + sourceTokenIdx;
            if (combineRow >= 0 && combineRow < combineLayoutRows)
            {
                int32_t const activeIdx = atomicAdd(activeRouteCount, 1);
                if (activeIdx < numPoolSlots)
                {
                    activePoolSlots[activeIdx] = static_cast<int64_t>(poolSlot);
                    activeCombineRows[activeIdx] = static_cast<int64_t>(combineRow);
                }
                if (outputPermutedIdxToExpandedIdx != nullptr && poolSlot < outputMappingRows)
                {
                    outputPermutedIdxToExpandedIdx[poolSlot] = combineRow;
                }
                if (outputTokenFinalScales != nullptr && combineRow < outputScaleRows)
                {
                    outputTokenFinalScales[combineRow] = topKScale;
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5MaterializeDirectFromRankedTopKKernel(uint8_t const* input, int64_t inputRankStride,
    int64_t inputTokenStride, uint8_t const* inputSf, int64_t inputSfRankStride, int64_t inputSfTokenStride,
    int64_t const* topKIdx, int64_t topKIdxRankStride, int64_t topKIdxTokenStride, float const* topKScales,
    int64_t topKScalesRankStride, int64_t topKScalesTokenStride, int32_t const* tokenCounts,
    int32_t* expertRouteOffsets, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int64_t* activePoolSlots, int64_t* activeCombineRows,
    int32_t* activeRouteCount, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const epSize,
    int32_t const localRank, int32_t const numExpertsPerRank, int32_t const hiddenPackedSize,
    int32_t const sfHiddenSize, int32_t const topK, int32_t const tileSize, int32_t const numPoolSlots,
    int32_t const numPaddedSfPoolTokens, int32_t const l1ArrivalCountEntries, int32_t const maxNumTokensPerRank,
    int32_t const combineLayoutRows, int32_t const outputMappingRows, int32_t const outputScaleRows,
    int32_t const routeLayoutCapacity)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int32_t const totalRoutes = epSize * maxNumTokensPerRank * topK;
    int32_t const routesPerRank = maxNumTokensPerRank * topK;
    int32_t const sfBlockTokens = ((tileSize + 127) / 128) * 128;
    int32_t const firstLocalExpert = localRank * numExpertsPerRank;
    int32_t const expertStride = ((epSize * maxNumTokensPerRank + tileSize - 1) / tileSize) * tileSize;
    __shared__ int32_t routeOrdinal;

    for (int32_t routeIdx = blockIdx.x; routeIdx < totalRoutes; routeIdx += gridDim.x)
    {
        int32_t const sourceRank = routeIdx / routesPerRank;
        int32_t const rankRouteIdx = routeIdx - sourceRank * routesPerRank;
        int32_t const sourceTokenIdx = rankRouteIdx / topK;
        int32_t const topKOrdinal = rankRouteIdx - sourceTokenIdx * topK;
        int32_t const tokenCount = tokenCounts[sourceRank];
        if (sourceTokenIdx < 0 || sourceTokenIdx >= tokenCount || sourceTokenIdx >= maxNumTokensPerRank)
        {
            continue;
        }

        int64_t const selectedExpert64 = topKIdx[static_cast<int64_t>(sourceRank) * topKIdxRankStride
            + static_cast<int64_t>(sourceTokenIdx) * topKIdxTokenStride + topKOrdinal];
        if (selectedExpert64 < firstLocalExpert || selectedExpert64 >= firstLocalExpert + numExpertsPerRank)
        {
            continue;
        }

        int32_t const localExpertIdx = static_cast<int32_t>(selectedExpert64) - firstLocalExpert;
        int32_t const poolStart = localExpertIdx * expertStride;
        if (poolStart < 0 || poolStart >= numPoolSlots)
        {
            continue;
        }

        if (threadIdx.x == 0)
        {
            routeOrdinal = atomicAdd(expertRouteOffsets + localExpertIdx, 1);
        }
        __syncthreads();
        int32_t const ordinal = routeOrdinal;
        if (ordinal < 0 || ordinal >= expertStride || poolStart + ordinal >= numPoolSlots)
        {
            continue;
        }

        int32_t const poolSlot = poolStart + ordinal;
        for (int32_t i = threadIdx.x; i < hiddenPackedSize; i += kThreadsPerBlock)
        {
            l1ActsPool[static_cast<int64_t>(poolSlot) * hiddenPackedSize + i]
                = input[static_cast<int64_t>(sourceRank) * inputRankStride
                    + static_cast<int64_t>(sourceTokenIdx) * inputTokenStride + i];
        }

        int32_t const sfSlot = (poolSlot / tileSize) * sfBlockTokens + poolSlot % tileSize;
        if (sfSlot < numPaddedSfPoolTokens)
        {
            for (int32_t i = threadIdx.x; i < sfHiddenSize; i += kThreadsPerBlock)
            {
                l1ActsSfPool[static_cast<int64_t>(sfSlot) * sfHiddenSize + i]
                    = inputSf[static_cast<int64_t>(sourceRank) * inputSfRankStride
                        + static_cast<int64_t>(sourceTokenIdx) * inputSfTokenStride + i];
            }
        }

        if (threadIdx.x == 0)
        {
            int32_t const tileStart = poolStart / tileSize;
            int32_t const tileOffset = ordinal / tileSize;
            int32_t const tileIdx = tileStart + tileOffset;
            int32_t const tileMnLimit = poolSlot + 1;
            if (tileIdx < routeLayoutCapacity)
            {
                tileIdxToExpertIdx[tileIdx] = localExpertIdx;
                atomicMax(tileIdxToMnLimit + tileIdx, tileMnLimit);
            }
            if (tileIdx < l1ArrivalCountEntries)
            {
                atomicMax(l1ArrivalCount + tileIdx, ordinal - tileOffset * tileSize + 1);
            }
            atomicMax(numNonExitingTiles, tileIdx + 1);

            float const topKScale = topKScales[static_cast<int64_t>(sourceRank) * topKScalesRankStride
                + static_cast<int64_t>(sourceTokenIdx) * topKScalesTokenStride + topKOrdinal];
            l1TopKWeightsPool[poolSlot] = topKScale;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3] = sourceRank;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3 + 1] = sourceTokenIdx;
            tokenSrcMetadata[static_cast<int64_t>(poolSlot) * 3 + 2] = topKOrdinal;

            int32_t const combineRow = (sourceRank * topK + topKOrdinal) * maxNumTokensPerRank + sourceTokenIdx;
            if (combineRow >= 0 && combineRow < combineLayoutRows)
            {
                int32_t const activeIdx = atomicAdd(activeRouteCount, 1);
                if (activeIdx < numPoolSlots)
                {
                    activePoolSlots[activeIdx] = static_cast<int64_t>(poolSlot);
                    activeCombineRows[activeIdx] = static_cast<int64_t>(combineRow);
                }
                if (outputPermutedIdxToExpandedIdx != nullptr && poolSlot < outputMappingRows)
                {
                    outputPermutedIdxToExpandedIdx[poolSlot] = combineRow;
                }
                if (outputTokenFinalScales != nullptr && combineRow < outputScaleRows)
                {
                    outputTokenFinalScales[combineRow] = topKScale;
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void megaMoeM5MaterializeDirectFromRankedTopK(uint8_t const* input, int64_t inputRankStride, int64_t inputTokenStride,
    uint8_t const* inputSf, int64_t inputSfRankStride, int64_t inputSfTokenStride, int64_t const* topKIdx,
    int64_t topKIdxRankStride, int64_t topKIdxTokenStride, float const* topKScales, int64_t topKScalesRankStride,
    int64_t topKScalesTokenStride, int32_t const* tokenCounts, int32_t* expertRouteOffsets, uint8_t* l1ActsPool,
    uint8_t* l1ActsSfPool, float* l1TopKWeightsPool, int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount,
    int64_t* activePoolSlots, int64_t* activeCombineRows, int32_t* activeRouteCount,
    int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const epSize, int32_t const localRank,
    int32_t const numExpertsPerRank, int32_t const hiddenPackedSize, int32_t const sfHiddenSize, int32_t const topK,
    int32_t const tileSize, int32_t const numPoolSlots, int32_t const numPaddedSfPoolTokens,
    int32_t const l1ArrivalCountEntries, int32_t const maxNumTokensPerRank, int32_t const combineLayoutRows,
    int32_t const outputMappingRows, int32_t const outputScaleRows, int32_t const routeLayoutCapacity,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t const expertStride = ((epSize * maxNumTokensPerRank + tileSize - 1) / tileSize) * tileSize;
    if (numPoolSlots <= 0 || numExpertsPerRank <= 0 || expertStride * numExpertsPerRank > numPoolSlots)
    {
        return;
    }

    auto kernel = &megaMoeM5MaterializeDirectFromRankedTopKKernel<kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const totalRoutes = epSize * maxNumTokensPerRank * topK;
    int32_t const routeBlocks = (totalRoutes > 0) ? totalRoutes : 1;
    int32_t const platformBlocks = smCount * maxBlocksPerSM;
    int32_t const blocks = std::max(1, std::min(platformBlocks, routeBlocks));

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, inputRankStride, inputTokenStride, inputSf, inputSfRankStride,
        inputSfTokenStride, topKIdx, topKIdxRankStride, topKIdxTokenStride, topKScales, topKScalesRankStride,
        topKScalesTokenStride, tokenCounts, expertRouteOffsets, l1ActsPool, l1ActsSfPool, l1TopKWeightsPool,
        tokenSrcMetadata, l1ArrivalCount, activePoolSlots, activeCombineRows, activeRouteCount,
        outputPermutedIdxToExpandedIdx, outputTokenFinalScales, tileIdxToExpertIdx, tileIdxToMnLimit,
        numNonExitingTiles, epSize, localRank, numExpertsPerRank, hiddenPackedSize, sfHiddenSize, topK, tileSize,
        numPoolSlots, numPaddedSfPoolTokens, l1ArrivalCountEntries, maxNumTokensPerRank, combineLayoutRows,
        outputMappingRows, outputScaleRows, routeLayoutCapacity);
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5BuildDirectInputRouteCopyAndCountKernel(uint8_t const* input, int64_t inputRankStride,
    int64_t inputTokenStride, uint8_t const* inputSf, int64_t inputSfRankStride, int64_t inputSfTokenStride,
    uint8_t* directInput, uint8_t* directInputSf, int64_t const* topKIdx, int64_t topKIdxRankStride,
    int64_t topKIdxTokenStride, int32_t const* tokenCounts, int32_t* expertRouteOffsets, int32_t const epSize,
    int32_t const localRank, int32_t const numExpertsPerRank, int32_t const hiddenPackedSize,
    int32_t const sfHiddenSize, int32_t const topK, int32_t const maxNumTokensPerRank, bool const useVectorInputCopy,
    bool const useVectorInputSfCopy)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int32_t const totalRoutes = epSize * maxNumTokensPerRank * topK;
    int32_t const routesPerRank = maxNumTokensPerRank * topK;
    int32_t const firstLocalExpert = localRank * numExpertsPerRank;
    int64_t const gridStride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t const firstLinearIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (int32_t sourceRank = 0; sourceRank < epSize; ++sourceRank)
    {
        int32_t rankTokenCount = tokenCounts[sourceRank];
        if (rankTokenCount < 0)
        {
            rankTokenCount = 0;
        }
        if (rankTokenCount > maxNumTokensPerRank)
        {
            rankTokenCount = maxNumTokensPerRank;
        }

        int64_t const directInputRankOffset = static_cast<int64_t>(sourceRank) * maxNumTokensPerRank * hiddenPackedSize;
        if (useVectorInputCopy)
        {
            int32_t constexpr kBytesPerVector = sizeof(ElemCopyType);
            int64_t const hiddenPackedVectors = hiddenPackedSize / kBytesPerVector;
            int64_t const directInputRankVectorOffset = directInputRankOffset / kBytesPerVector;
            int64_t const inputRankVectorOffset = static_cast<int64_t>(sourceRank) * inputRankStride / kBytesPerVector;
            int64_t const inputTokenVectorStride = inputTokenStride / kBytesPerVector;
            int64_t const inputVectors = static_cast<int64_t>(rankTokenCount) * hiddenPackedVectors;
            auto const* inputVector = reinterpret_cast<ElemCopyType const*>(input);
            auto* directInputVector = reinterpret_cast<ElemCopyType*>(directInput);
            for (int64_t linearIdx = firstLinearIdx; linearIdx < inputVectors; linearIdx += gridStride)
            {
                int32_t const sourceTokenIdx = static_cast<int32_t>(linearIdx / hiddenPackedVectors);
                int32_t const hiddenIdx
                    = static_cast<int32_t>(linearIdx - static_cast<int64_t>(sourceTokenIdx) * hiddenPackedVectors);
                directInputVector[directInputRankVectorOffset + linearIdx] = inputVector[inputRankVectorOffset
                    + static_cast<int64_t>(sourceTokenIdx) * inputTokenVectorStride + hiddenIdx];
            }
        }
        else
        {
            int64_t const inputElements = static_cast<int64_t>(rankTokenCount) * hiddenPackedSize;
            for (int64_t linearIdx = firstLinearIdx; linearIdx < inputElements; linearIdx += gridStride)
            {
                int32_t const sourceTokenIdx = static_cast<int32_t>(linearIdx / hiddenPackedSize);
                int32_t const hiddenIdx
                    = static_cast<int32_t>(linearIdx - static_cast<int64_t>(sourceTokenIdx) * hiddenPackedSize);
                directInput[directInputRankOffset + linearIdx]
                    = input[static_cast<int64_t>(sourceRank) * inputRankStride
                        + static_cast<int64_t>(sourceTokenIdx) * inputTokenStride + hiddenIdx];
            }
        }

        int64_t const directInputSfRankOffset = static_cast<int64_t>(sourceRank) * maxNumTokensPerRank * sfHiddenSize;
        if (useVectorInputSfCopy)
        {
            int32_t constexpr kBytesPerVector = sizeof(ElemCopyType);
            int64_t const sfHiddenVectors = sfHiddenSize / kBytesPerVector;
            int64_t const directInputSfRankVectorOffset = directInputSfRankOffset / kBytesPerVector;
            int64_t const inputSfRankVectorOffset
                = static_cast<int64_t>(sourceRank) * inputSfRankStride / kBytesPerVector;
            int64_t const inputSfTokenVectorStride = inputSfTokenStride / kBytesPerVector;
            int64_t const inputSfVectors = static_cast<int64_t>(rankTokenCount) * sfHiddenVectors;
            auto const* inputSfVector = reinterpret_cast<ElemCopyType const*>(inputSf);
            auto* directInputSfVector = reinterpret_cast<ElemCopyType*>(directInputSf);
            for (int64_t linearIdx = firstLinearIdx; linearIdx < inputSfVectors; linearIdx += gridStride)
            {
                int32_t const sourceTokenIdx = static_cast<int32_t>(linearIdx / sfHiddenVectors);
                int32_t const hiddenIdx
                    = static_cast<int32_t>(linearIdx - static_cast<int64_t>(sourceTokenIdx) * sfHiddenVectors);
                directInputSfVector[directInputSfRankVectorOffset + linearIdx] = inputSfVector[inputSfRankVectorOffset
                    + static_cast<int64_t>(sourceTokenIdx) * inputSfTokenVectorStride + hiddenIdx];
            }
        }
        else
        {
            int64_t const inputSfElements = static_cast<int64_t>(rankTokenCount) * sfHiddenSize;
            for (int64_t linearIdx = firstLinearIdx; linearIdx < inputSfElements; linearIdx += gridStride)
            {
                int32_t const sourceTokenIdx = static_cast<int32_t>(linearIdx / sfHiddenSize);
                int32_t const hiddenIdx
                    = static_cast<int32_t>(linearIdx - static_cast<int64_t>(sourceTokenIdx) * sfHiddenSize);
                directInputSf[directInputSfRankOffset + linearIdx]
                    = inputSf[static_cast<int64_t>(sourceRank) * inputSfRankStride
                        + static_cast<int64_t>(sourceTokenIdx) * inputSfTokenStride + hiddenIdx];
            }
        }
    }

    for (int32_t routeIdx = blockIdx.x * blockDim.x + threadIdx.x; routeIdx < totalRoutes;
         routeIdx += static_cast<int32_t>(gridDim.x) * blockDim.x)
    {
        int32_t const sourceRank = routeIdx / routesPerRank;
        int32_t const rankRouteIdx = routeIdx - sourceRank * routesPerRank;
        int32_t const sourceTokenIdx = rankRouteIdx / topK;
        int32_t const topKOrdinal = rankRouteIdx - sourceTokenIdx * topK;
        int32_t const tokenCount = tokenCounts[sourceRank];
        if (sourceTokenIdx < 0 || sourceTokenIdx >= tokenCount || sourceTokenIdx >= maxNumTokensPerRank)
        {
            continue;
        }

        int64_t const selectedExpert64 = topKIdx[static_cast<int64_t>(sourceRank) * topKIdxRankStride
            + static_cast<int64_t>(sourceTokenIdx) * topKIdxTokenStride + topKOrdinal];
        if (selectedExpert64 < firstLocalExpert || selectedExpert64 >= firstLocalExpert + numExpertsPerRank)
        {
            continue;
        }

        int32_t const localExpertIdx = static_cast<int32_t>(selectedExpert64) - firstLocalExpert;
        atomicAdd(expertRouteOffsets + localExpertIdx, 1);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5PrefixDirectInputRouteOffsetsKernel(int32_t* expertRouteOffsets,
    int32_t* expertRouteBaseOffsets, int32_t* numNonExitingTiles, int32_t const numExpertsPerRank,
    int32_t const tileSize, int32_t const numPoolSlots)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int32_t runningOffset = 0;
        for (int32_t expertIdx = 0; expertIdx < numExpertsPerRank; ++expertIdx)
        {
            int32_t count = expertRouteOffsets[expertIdx];
            if (count < 0)
            {
                count = 0;
            }
            int32_t const alignedCount = ((count + tileSize - 1) / tileSize) * tileSize;
            expertRouteBaseOffsets[expertIdx] = runningOffset;
            expertRouteOffsets[expertIdx] = 0;
            runningOffset += alignedCount;
        }
        if (runningOffset > numPoolSlots)
        {
            numNonExitingTiles[0] = (numPoolSlots + tileSize - 1) / tileSize + 1;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5FillPackedDirectInputRouteKernel(int64_t const* topKIdx, int64_t topKIdxRankStride,
    int64_t topKIdxTokenStride, float const* topKScales, int64_t topKScalesRankStride, int64_t topKScalesTokenStride,
    int32_t const* tokenCounts, int32_t* expertRouteOffsets, int32_t const* expertRouteBaseOffsets,
    int32_t* tokenIdMapping, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const epSize,
    int32_t const localRank, int32_t const numExpertsPerRank, int32_t const topK, int32_t const tileSize,
    int32_t const numPoolSlots, int32_t const maxNumTokensPerRank, int32_t const combineLayoutRows,
    int32_t const outputMappingRows, int32_t const outputScaleRows, int32_t const routeLayoutCapacity,
    bool const directAtomicOutput, bool const directTokenMajorOutput)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int32_t const totalRoutes = epSize * maxNumTokensPerRank * topK;
    int32_t const routesPerRank = maxNumTokensPerRank * topK;
    int32_t const firstLocalExpert = localRank * numExpertsPerRank;

    for (int32_t routeIdx = blockIdx.x * blockDim.x + threadIdx.x; routeIdx < totalRoutes;
         routeIdx += static_cast<int32_t>(gridDim.x) * blockDim.x)
    {
        int32_t const sourceRank = routeIdx / routesPerRank;
        int32_t const rankRouteIdx = routeIdx - sourceRank * routesPerRank;
        int32_t const sourceTokenIdx = rankRouteIdx / topK;
        int32_t const topKOrdinal = rankRouteIdx - sourceTokenIdx * topK;
        int32_t const tokenCount = tokenCounts[sourceRank];
        if (sourceTokenIdx < 0 || sourceTokenIdx >= tokenCount || sourceTokenIdx >= maxNumTokensPerRank)
        {
            continue;
        }

        int64_t const selectedExpert64 = topKIdx[static_cast<int64_t>(sourceRank) * topKIdxRankStride
            + static_cast<int64_t>(sourceTokenIdx) * topKIdxTokenStride + topKOrdinal];
        if (selectedExpert64 < firstLocalExpert || selectedExpert64 >= firstLocalExpert + numExpertsPerRank)
        {
            continue;
        }

        int32_t const localExpertIdx = static_cast<int32_t>(selectedExpert64) - firstLocalExpert;
        int32_t const expertBase = expertRouteBaseOffsets[localExpertIdx];
        int32_t const ordinal = atomicAdd(expertRouteOffsets + localExpertIdx, 1);
        int32_t const poolSlot = expertBase + ordinal;
        if (poolSlot < 0 || poolSlot >= numPoolSlots)
        {
            continue;
        }

        int32_t const tileIdx = poolSlot / tileSize;
        int32_t const tileMnLimit = poolSlot + 1;
        if (tileIdx < routeLayoutCapacity)
        {
            tileIdxToExpertIdx[tileIdx] = localExpertIdx;
            atomicMax(tileIdxToMnLimit + tileIdx, tileMnLimit);
        }
        atomicMax(numNonExitingTiles, tileIdx + 1);

        int32_t const tokenRow = sourceRank * maxNumTokensPerRank + sourceTokenIdx;
        bool const tokenMajorOutput = directAtomicOutput || directTokenMajorOutput;
        int32_t const combineRow = tokenMajorOutput
            ? tokenRow * topK + topKOrdinal
            : (sourceRank * topK + topKOrdinal) * maxNumTokensPerRank + sourceTokenIdx;
        int32_t const combineRowLimit = tokenMajorOutput ? outputScaleRows : combineLayoutRows;
        if (poolSlot < outputMappingRows && combineRow >= 0 && combineRow < combineRowLimit)
        {
            outputPermutedIdxToExpandedIdx[poolSlot] = combineRow;
        }
        tokenIdMapping[poolSlot] = directAtomicOutput ? combineRow : tokenRow;
        if (outputTokenFinalScales != nullptr && combineRow >= 0 && combineRow < combineRowLimit
            && combineRow < outputScaleRows)
        {
            outputTokenFinalScales[combineRow] = topKScales[static_cast<int64_t>(sourceRank) * topKScalesRankStride
                + static_cast<int64_t>(sourceTokenIdx) * topKScalesTokenStride + topKOrdinal];
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeStageDispatchInputsKernel(uint8_t const* input, int64_t inputBytes, uint8_t const* inputSf,
    int64_t inputSfBytes, uint8_t const* topKIdx, int64_t topKIdxBytes, uint8_t const* topKScales,
    int64_t topKScalesBytes, uint8_t* inputBuffer, uint8_t* inputSfBuffer, uint8_t* topKIdxBuffer,
    uint8_t* topKScalesBuffer, bool const useVectorCopy)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int64_t workItems = inputBytes > inputSfBytes ? inputBytes : inputSfBytes;
    workItems = workItems > topKIdxBytes ? workItems : topKIdxBytes;
    workItems = workItems > topKScalesBytes ? workItems : topKScalesBytes;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    if (useVectorCopy)
    {
        int32_t constexpr kBytesPerVector = sizeof(ElemCopyType);
        int64_t const inputVectors = inputBytes / kBytesPerVector;
        int64_t const inputSfVectors = inputSfBytes / kBytesPerVector;
        int64_t const topKIdxVectors = topKIdxBytes / kBytesPerVector;
        int64_t const topKScalesVectors = topKScalesBytes / kBytesPerVector;
        int64_t vectorWorkItems = inputVectors > inputSfVectors ? inputVectors : inputSfVectors;
        vectorWorkItems = vectorWorkItems > topKIdxVectors ? vectorWorkItems : topKIdxVectors;
        vectorWorkItems = vectorWorkItems > topKScalesVectors ? vectorWorkItems : topKScalesVectors;
        auto const* inputVector = reinterpret_cast<ElemCopyType const*>(input);
        auto const* inputSfVector = reinterpret_cast<ElemCopyType const*>(inputSf);
        auto const* topKIdxVector = reinterpret_cast<ElemCopyType const*>(topKIdx);
        auto const* topKScalesVector = reinterpret_cast<ElemCopyType const*>(topKScales);
        auto* inputBufferVector = reinterpret_cast<ElemCopyType*>(inputBuffer);
        auto* inputSfBufferVector = reinterpret_cast<ElemCopyType*>(inputSfBuffer);
        auto* topKIdxBufferVector = reinterpret_cast<ElemCopyType*>(topKIdxBuffer);
        auto* topKScalesBufferVector = reinterpret_cast<ElemCopyType*>(topKScalesBuffer);
        for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < vectorWorkItems;
             idx += stride)
        {
            if (idx < inputVectors)
            {
                inputBufferVector[idx] = inputVector[idx];
            }
            if (idx < inputSfVectors)
            {
                inputSfBufferVector[idx] = inputSfVector[idx];
            }
            if (idx < topKIdxVectors)
            {
                topKIdxBufferVector[idx] = topKIdxVector[idx];
            }
            if (idx < topKScalesVectors)
            {
                topKScalesBufferVector[idx] = topKScalesVector[idx];
            }
        }
    }
    else
    {
        for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < workItems; idx += stride)
        {
            if (idx < inputBytes)
            {
                inputBuffer[idx] = input[idx];
            }
            if (idx < inputSfBytes)
            {
                inputSfBuffer[idx] = inputSf[idx];
            }
            if (idx < topKIdxBytes)
            {
                topKIdxBuffer[idx] = topKIdx[idx];
            }
            if (idx < topKScalesBytes)
            {
                topKScalesBuffer[idx] = topKScales[idx];
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void megaMoeStageDispatchInputs(uint8_t const* input, int64_t const inputBytes, uint8_t const* inputSf,
    int64_t const inputSfBytes, uint8_t const* topKIdx, int64_t const topKIdxBytes, uint8_t const* topKScales,
    int64_t const topKScalesBytes, uint8_t* inputBuffer, uint8_t* inputSfBuffer, uint8_t* topKIdxBuffer,
    uint8_t* topKScalesBuffer, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int64_t workItems = inputBytes > inputSfBytes ? inputBytes : inputSfBytes;
    workItems = workItems > topKIdxBytes ? workItems : topKIdxBytes;
    workItems = workItems > topKScalesBytes ? workItems : topKScalesBytes;
    if (workItems <= 0)
    {
        return;
    }

    auto kernel = &megaMoeStageDispatchInputsKernel<kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const workBlocks = static_cast<int32_t>((workItems + kThreadsPerBlock - 1) / kThreadsPerBlock);
    int32_t const platformBlocks = smCount * maxBlocksPerSM;
    int32_t const blocks = std::max(1, std::min(platformBlocks, workBlocks));

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    int32_t constexpr kBytesPerVector = sizeof(ElemCopyType);
    auto const isAlignedForVectorCopy
        = [](void const* ptr) { return reinterpret_cast<std::uintptr_t>(ptr) % sizeof(ElemCopyType) == 0; };
    bool const useVectorCopy = inputBytes % kBytesPerVector == 0 && inputSfBytes % kBytesPerVector == 0
        && topKIdxBytes % kBytesPerVector == 0 && topKScalesBytes % kBytesPerVector == 0
        && isAlignedForVectorCopy(input) && isAlignedForVectorCopy(inputSf) && isAlignedForVectorCopy(topKIdx)
        && isAlignedForVectorCopy(topKScales) && isAlignedForVectorCopy(inputBuffer)
        && isAlignedForVectorCopy(inputSfBuffer) && isAlignedForVectorCopy(topKIdxBuffer)
        && isAlignedForVectorCopy(topKScalesBuffer);
    cudaLaunchKernelEx(&config, kernel, input, inputBytes, inputSf, inputSfBytes, topKIdx, topKIdxBytes, topKScales,
        topKScalesBytes, inputBuffer, inputSfBuffer, topKIdxBuffer, topKScalesBuffer, useVectorCopy);
}

template <int32_t kThreadsPerBlock>
__global__ void megaMoeM5InitDirectInputRouteMetadataKernel(int32_t* expertRouteOffsets, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t numExpertsPerRank, int32_t routeLayoutCapacity)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int32_t const workItems = numExpertsPerRank > routeLayoutCapacity ? numExpertsPerRank : routeLayoutCapacity;
    for (int32_t idx = static_cast<int32_t>(blockIdx.x) * kThreadsPerBlock + static_cast<int32_t>(threadIdx.x);
         idx < workItems; idx += static_cast<int32_t>(gridDim.x) * kThreadsPerBlock)
    {
        if (idx < numExpertsPerRank)
        {
            expertRouteOffsets[idx] = 0;
        }
        if (idx < routeLayoutCapacity)
        {
            tileIdxToExpertIdx[idx] = -1;
            tileIdxToMnLimit[idx] = 0;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        numNonExitingTiles[0] = 0;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void megaMoeM5InitDirectInputRouteMetadata(int32_t* expertRouteOffsets, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const numExpertsPerRank,
    int32_t const routeLayoutCapacity, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t const workItems = std::max(numExpertsPerRank, routeLayoutCapacity);
    if (workItems <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM5InitDirectInputRouteMetadataKernel<kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const workBlocks = (workItems + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int32_t const platformBlocks = smCount * maxBlocksPerSM;
    int32_t const blocks = std::max(1, std::min(platformBlocks, workBlocks));

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, expertRouteOffsets, tileIdxToExpertIdx, tileIdxToMnLimit, numNonExitingTiles,
        numExpertsPerRank, routeLayoutCapacity);
}

void megaMoeM5BuildDirectInputRouteFromRankedTopK(uint8_t const* input, int64_t inputRankStride,
    int64_t inputTokenStride, uint8_t const* inputSf, int64_t inputSfRankStride, int64_t inputSfTokenStride,
    uint8_t* directInput, uint8_t* directInputSf, int64_t const* topKIdx, int64_t topKIdxRankStride,
    int64_t topKIdxTokenStride, float const* topKScales, int64_t topKScalesRankStride, int64_t topKScalesTokenStride,
    int32_t const* tokenCounts, int32_t* expertRouteOffsets, int32_t* expertRouteBaseOffsets, int32_t* tokenIdMapping,
    int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const epSize, int32_t const localRank,
    int32_t const numExpertsPerRank, int32_t const hiddenPackedSize, int32_t const sfHiddenSize, int32_t const topK,
    int32_t const tileSize, int32_t const numPoolSlots, int32_t const maxNumTokensPerRank,
    int32_t const combineLayoutRows, int32_t const outputMappingRows, int32_t const outputScaleRows,
    int32_t const routeLayoutCapacity, bool const directAtomicOutput, bool const directTokenMajorOutput,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (numPoolSlots <= 0 || numExpertsPerRank <= 0 || hiddenPackedSize <= 0 || sfHiddenSize <= 0 || tileSize <= 0)
    {
        return;
    }

    auto copyCountKernel = &megaMoeM5BuildDirectInputRouteCopyAndCountKernel<kThreadsPerBlock>;
    auto prefixKernel = &megaMoeM5PrefixDirectInputRouteOffsetsKernel<kThreadsPerBlock>;
    auto fillKernel = &megaMoeM5FillPackedDirectInputRouteKernel<kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(copyCountKernel, kThreadsPerBlock, 0);
    int32_t const totalRoutes = epSize * maxNumTokensPerRank * topK;
    int64_t const flatInputRows = static_cast<int64_t>(epSize) * maxNumTokensPerRank;
    int64_t const flatInputElements = flatInputRows * hiddenPackedSize;
    int64_t const flatInputSfElements = flatInputRows * sfHiddenSize;
    int64_t const maxWorkItems
        = std::max(static_cast<int64_t>(totalRoutes), std::max(flatInputElements, flatInputSfElements));
    int32_t const workBlocks
        = static_cast<int32_t>(std::max<int64_t>(1, (maxWorkItems + kThreadsPerBlock - 1) / kThreadsPerBlock));
    int32_t const routeBlocks = static_cast<int32_t>(
        std::max<int64_t>(1, (static_cast<int64_t>(totalRoutes) + kThreadsPerBlock - 1) / kThreadsPerBlock));
    int32_t const platformBlocks = smCount * maxBlocksPerSM;
    int32_t const copyCountBlocks = std::max(1, std::min(platformBlocks, workBlocks));
    int32_t const fillBlocks = std::max(1, std::min(platformBlocks, routeBlocks));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();

    cudaLaunchConfig_t copyConfig;
    copyConfig.gridDim = copyCountBlocks;
    copyConfig.blockDim = kThreadsPerBlock;
    copyConfig.dynamicSmemBytes = 0;
    copyConfig.stream = stream;
    copyConfig.numAttrs = 1;
    copyConfig.attrs = attrs;
    int32_t constexpr kBytesPerVector = sizeof(ElemCopyType);
    auto const isAlignedForVectorCopy
        = [](void const* ptr) { return reinterpret_cast<std::uintptr_t>(ptr) % sizeof(ElemCopyType) == 0; };
    bool const useVectorInputCopy = hiddenPackedSize % kBytesPerVector == 0 && inputRankStride % kBytesPerVector == 0
        && inputTokenStride % kBytesPerVector == 0 && isAlignedForVectorCopy(input)
        && isAlignedForVectorCopy(directInput);
    bool const useVectorInputSfCopy = sfHiddenSize % kBytesPerVector == 0 && inputSfRankStride % kBytesPerVector == 0
        && inputSfTokenStride % kBytesPerVector == 0 && isAlignedForVectorCopy(inputSf)
        && isAlignedForVectorCopy(directInputSf);
    cudaLaunchKernelEx(&copyConfig, copyCountKernel, input, inputRankStride, inputTokenStride, inputSf,
        inputSfRankStride, inputSfTokenStride, directInput, directInputSf, topKIdx, topKIdxRankStride,
        topKIdxTokenStride, tokenCounts, expertRouteOffsets, epSize, localRank, numExpertsPerRank, hiddenPackedSize,
        sfHiddenSize, topK, maxNumTokensPerRank, useVectorInputCopy, useVectorInputSfCopy);

    cudaLaunchConfig_t prefixConfig;
    prefixConfig.gridDim = 1;
    prefixConfig.blockDim = kThreadsPerBlock;
    prefixConfig.dynamicSmemBytes = 0;
    prefixConfig.stream = stream;
    prefixConfig.numAttrs = 1;
    prefixConfig.attrs = attrs;
    cudaLaunchKernelEx(&prefixConfig, prefixKernel, expertRouteOffsets, expertRouteBaseOffsets, numNonExitingTiles,
        numExpertsPerRank, tileSize, numPoolSlots);

    cudaLaunchConfig_t fillConfig;
    fillConfig.gridDim = fillBlocks;
    fillConfig.blockDim = kThreadsPerBlock;
    fillConfig.dynamicSmemBytes = 0;
    fillConfig.stream = stream;
    fillConfig.numAttrs = 1;
    fillConfig.attrs = attrs;
    cudaLaunchKernelEx(&fillConfig, fillKernel, topKIdx, topKIdxRankStride, topKIdxTokenStride, topKScales,
        topKScalesRankStride, topKScalesTokenStride, tokenCounts, expertRouteOffsets, expertRouteBaseOffsets,
        tokenIdMapping, outputPermutedIdxToExpandedIdx, outputTokenFinalScales, tileIdxToExpertIdx, tileIdxToMnLimit,
        numNonExitingTiles, epSize, localRank, numExpertsPerRank, topK, tileSize, numPoolSlots, maxNumTokensPerRank,
        combineLayoutRows, outputMappingRows, outputScaleRows, routeLayoutCapacity, directAtomicOutput,
        directTokenMajorOutput);
}

void megaMoeM5MaterializeDirectFromTopK(uint8_t const* input, uint8_t const* inputSf, int64_t const* topKIdx,
    float const* topKScales, int32_t const* tokenOffsets, int64_t const* expertRecvCountSum,
    int32_t* expertRouteOffsets, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int64_t* activePoolSlots, int64_t* activeCombineRows,
    int32_t* activeRouteCount, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t const totalTokens,
    int32_t const epSize, int32_t const localRank, int32_t const numExpertsPerRank, int32_t const hiddenPackedSize,
    int32_t const sfHiddenSize, int32_t const topK, int32_t const tileSize, int32_t const numPoolSlots,
    int32_t const numPaddedSfPoolTokens, int32_t const l1ArrivalCountEntries, int32_t const maxNumTokensPerRank,
    int32_t const combineLayoutRows, int32_t const outputMappingRows, int32_t const outputScaleRows,
    int32_t const routeLayoutCapacity, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (numPoolSlots <= 0 || numExpertsPerRank <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM5MaterializeDirectFromTopKKernel<kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const totalRoutes = totalTokens * topK;
    int32_t const routeBlocks = (totalRoutes > 0) ? totalRoutes : 1;
    int32_t const platformBlocks = smCount * maxBlocksPerSM;
    int32_t const blocks = std::max(1, std::min(platformBlocks, routeBlocks));

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, inputSf, topKIdx, topKScales, tokenOffsets, expertRecvCountSum,
        expertRouteOffsets, l1ActsPool, l1ActsSfPool, l1TopKWeightsPool, tokenSrcMetadata, l1ArrivalCount,
        activePoolSlots, activeCombineRows, activeRouteCount, outputPermutedIdxToExpandedIdx, outputTokenFinalScales,
        tileIdxToExpertIdx, tileIdxToMnLimit, numNonExitingTiles, totalTokens, epSize, localRank, numExpertsPerRank,
        hiddenPackedSize, sfHiddenSize, topK, tileSize, numPoolSlots, numPaddedSfPoolTokens, l1ArrivalCountEntries,
        maxNumTokensPerRank, combineLayoutRows, outputMappingRows, outputScaleRows, routeLayoutCapacity);
}

template <int32_t kThreadsPerBlock, typename OutputType, bool kTokenMajor = false>
__global__ void megaMoeM6ReduceCombineBufferKernel(__nv_bfloat16 const* combineBuffer, OutputType* output,
    int32_t const topK, int32_t const localTokens, int32_t const maxNumTokensPerRank, int32_t const hiddenSize)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int64_t const totalElements = static_cast<int64_t>(localTokens) * hiddenSize;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linearIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; linearIdx < totalElements;
         linearIdx += stride)
    {
        int32_t const tokenIdx = static_cast<int32_t>(linearIdx / hiddenSize);
        int32_t const hiddenIdx = static_cast<int32_t>(linearIdx - static_cast<int64_t>(tokenIdx) * hiddenSize);
        float accum = 0.0F;
        for (int32_t topKIdx = 0; topKIdx < topK; ++topKIdx)
        {
            int64_t const offset = kTokenMajor
                ? (static_cast<int64_t>(tokenIdx) * topK + topKIdx) * hiddenSize + hiddenIdx
                : (static_cast<int64_t>(topKIdx) * maxNumTokensPerRank + tokenIdx) * hiddenSize + hiddenIdx;
            accum += __bfloat162float(combineBuffer[offset]);
        }
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>)
        {
            output[linearIdx] = __float2bfloat16(accum);
        }
        else
        {
            output[linearIdx] = accum;
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void megaMoeM6ReduceCombineBufferOut(__nv_bfloat16 const* combineBuffer, float* output, int32_t const topK,
    int32_t const localTokens, int32_t const maxNumTokensPerRank, int32_t const hiddenSize, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (topK <= 0 || localTokens <= 0 || maxNumTokensPerRank <= 0 || hiddenSize <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM6ReduceCombineBufferKernel<kThreadsPerBlock, float>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int64_t const totalElements = static_cast<int64_t>(localTokens) * hiddenSize;
    int32_t const elementBlocks = static_cast<int32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, elementBlocks);
    if (blocks <= 0)
    {
        return;
    }

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, combineBuffer, output, topK, localTokens, maxNumTokensPerRank, hiddenSize);
}

void megaMoeM6ReduceCombineBufferBf16Out(__nv_bfloat16 const* combineBuffer, __nv_bfloat16* output, int32_t const topK,
    int32_t const localTokens, int32_t const maxNumTokensPerRank, int32_t const hiddenSize, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (topK <= 0 || localTokens <= 0 || maxNumTokensPerRank <= 0 || hiddenSize <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM6ReduceCombineBufferKernel<kThreadsPerBlock, __nv_bfloat16>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int64_t const totalElements = static_cast<int64_t>(localTokens) * hiddenSize;
    int32_t const elementBlocks = static_cast<int32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, elementBlocks);
    if (blocks <= 0)
    {
        return;
    }

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, combineBuffer, output, topK, localTokens, maxNumTokensPerRank, hiddenSize);
}

void megaMoeM6ReduceTokenMajorCombineBufferBf16Out(__nv_bfloat16 const* combineBuffer, __nv_bfloat16* output,
    int32_t const topK, int32_t const localTokens, int32_t const maxNumTokensPerRank, int32_t const hiddenSize,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (topK <= 0 || localTokens <= 0 || maxNumTokensPerRank <= 0 || hiddenSize <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM6ReduceCombineBufferKernel<kThreadsPerBlock, __nv_bfloat16, true>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int64_t const totalElements = static_cast<int64_t>(localTokens) * hiddenSize;
    int32_t const elementBlocks = static_cast<int32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, elementBlocks);
    if (blocks <= 0)
    {
        return;
    }

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, combineBuffer, output, topK, localTokens, maxNumTokensPerRank, hiddenSize);
}

void megaMoeM6ReduceCombineBuffer(__nv_bfloat16 const* combineBuffer, float* output, int32_t const topK,
    int32_t const localTokens, int32_t const maxNumTokensPerRank, int32_t const hiddenSize, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    if (topK <= 0 || localTokens <= 0 || maxNumTokensPerRank <= 0 || hiddenSize <= 0)
    {
        return;
    }

    auto kernel = &megaMoeM6ReduceCombineBufferKernel<kThreadsPerBlock, float>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int64_t const totalElements = static_cast<int64_t>(localTokens) * hiddenSize;
    int32_t const elementBlocks = static_cast<int32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, elementBlocks);
    if (blocks <= 0)
    {
        return;
    }

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, combineBuffer, output, topK, localTokens, maxNumTokensPerRank, hiddenSize);
}

template <typename InputType, typename OutputType, typename SFType, int32_t kSFVecSize, typename ActFn,
    int32_t kThreadsPerBlock>
__global__ void moeActivationKernel(InputType const* input, OutputType* output, float const* global_sf,
    SFType* output_sf, int32_t const* tile_idx_to_mn_limit, int32_t const* num_non_exiting_tiles,
    int32_t const interm_size, int32_t const tile_size)
{
    using ComputeType = float;
#ifdef ENABLE_FP4
    using ElemOutputCopyType = std::conditional_t<std::is_same_v<OutputType, __nv_fp4_e2m1>, uint32_t, ElemCopyType>;
#else
    using ElemOutputCopyType = ElemCopyType;
#endif
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    // Need int64_t to prevent overflow when computing pointer offsets.
    int64_t const kCopyPerToken = interm_size / kElemPerCopy;
    InputType rmem[kElemPerCopy];
    InputType rmemGate[kElemPerCopy];
    ActFn act{};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    float global_sf_val = global_sf == nullptr ? 1.0f : global_sf[0];

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const tile_idx = permuted_idx / tile_size;
        if (permuted_idx >= tile_idx_to_mn_limit[tile_idx])
        {
            continue;
        }
        auto const* src_ptr
            = reinterpret_cast<ElemCopyType const*>(input) + permuted_idx * kCopyPerToken * (ActFn::IS_GLU ? 2 : 1);
        auto* dst_ptr = reinterpret_cast<ElemOutputCopyType*>(output) + permuted_idx * kCopyPerToken;
        for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
        {
            *reinterpret_cast<ElemCopyType*>(rmem) = src_ptr[i];
            if constexpr (ActFn::IS_GLU)
            {
                *reinterpret_cast<ElemCopyType*>(rmemGate) = src_ptr[i + kCopyPerToken];
#pragma unroll
                for (int32_t j = 0; j < kElemPerCopy; j++)
                {
                    rmem[j] = static_cast<InputType>(
                        act(static_cast<ComputeType>(rmemGate[j]), static_cast<ComputeType>(rmem[j])));
                }
            }
            else
            {
#pragma unroll
                for (int32_t j = 0; j < kElemPerCopy; j++)
                {
                    rmem[j] = static_cast<InputType>(act(static_cast<ComputeType>(rmem[j])));
                }
            }

#ifdef ENABLE_FP4
            if constexpr (std::is_same_v<OutputType, __nv_fp4_e2m1>)
            {
                auto* sf_dst_ptr = cvt_quant_get_sf_out_offset<SFType, kSFVecSize / kElemPerCopy>(
                    /* batchIdx= */ std::nullopt, permuted_idx, i, /*numRows=*/std::nullopt, interm_size / kSFVecSize,
                    output_sf, QuantizationSFLayout::SWIZZLED);
                dst_ptr[i] = cvt_warp_fp16_to_fp4<InputType, kSFVecSize, false>(
                    *reinterpret_cast<PackedVec<InputType>*>(rmem), global_sf_val, sf_dst_ptr);
            }
            else
#endif
            {
                dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename OutputType, typename SFType>
void moeActivation(InputType const* input, OutputType* output, float const* global_sf, SFType* output_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* num_non_exiting_tiles,
    cutlass_kernels::ActivationParams activation_params, int32_t const max_num_permuted_tokens,
    int32_t const interm_size, int32_t const tile_size, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kSFVecSize = 16;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(interm_size % kElemPerCopy == 0, "interm_size must be divisible by %d.", kElemPerCopy);

#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<InputType, __nv_fp4_e2m1>)
    {
        int32_t constexpr kSFMAlignment = 128;
        int32_t constexpr kSFKAlignment = 4;
        TLLM_CHECK_WITH_INFO(max_num_permuted_tokens % kSFMAlignment == 0,
            "max_num_permuted_tokens must be divisible by %d.", kSFMAlignment);
        TLLM_CHECK_WITH_INFO(interm_size % (kSFVecSize * kSFKAlignment) == 0, "interm_size must be divisible by %d.",
            kSFVecSize * kSFKAlignment);
        TLLM_CHECK_WITH_INFO(global_sf != nullptr, "global_sf is required for NVFP4.");
        TLLM_CHECK_WITH_INFO(output_sf != nullptr, "output_sf is required for NVFP4.");
    }
#endif

    auto get_act_kernel = [](ActivationType activation_type) -> void (*)(InputType const* input, OutputType* output,
                                                                 float const* global_sf, SFType* output_sf,
                                                                 int32_t const* tile_idx_to_mn_limit,
                                                                 int32_t const* num_non_exiting_tiles,
                                                                 int32_t const interm_size, int32_t const tile_size)
    {
        switch (activation_type)
        {
        case ActivationType::Identity:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::Identity>, kThreadsPerBlock>;
        case ActivationType::Gelu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::GELU>, kThreadsPerBlock>;
        case ActivationType::Geglu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::GLUAdaptor<cutlass::epilogue::thread::GELU>, kThreadsPerBlock>;
        case ActivationType::Relu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::ReLu>, kThreadsPerBlock>;
        case ActivationType::Silu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::SiLu>, kThreadsPerBlock>;
        case ActivationType::Swiglu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::GLUAdaptor<cutlass::epilogue::thread::SiLu>, kThreadsPerBlock>;
        case ActivationType::SwigluBias:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize, cutlass_kernels::SwigluBiasAdaptor,
                kThreadsPerBlock>;
        case ActivationType::Relu2:
            // Unsupported activation type
            break;
        }
        TLLM_CHECK_WITH_INFO(false, "Unsupported activation type: %d", int(activation_type));
        return nullptr;
    };
    auto kernel = get_act_kernel(activation_params.activation_type);

    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
    int32_t const threads = kThreadsPerBlock;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, output, global_sf, output_sf, tile_idx_to_mn_limit,
        num_non_exiting_tiles, interm_size, tile_size);
}

#define INSTANTIATE_MOE_ACTIVATION(InputType, OutputType, SFType)                                                      \
    template void moeActivation<InputType, OutputType, SFType>(InputType const* input, OutputType* output,             \
        float const* global_sf, SFType* output_sf, int32_t const* tile_idx_to_mn_limit,                                \
        int32_t const* num_non_exiting_tiles, cutlass_kernels::ActivationParams activation_params,                     \
        int32_t const max_num_permuted_tokens, int32_t const interm_size, int32_t const tile_size,                     \
        cudaStream_t stream)

INSTANTIATE_MOE_ACTIVATION(half, half, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_ACTIVATION(__nv_bfloat16, __nv_bfloat16, uint8_t);
#endif
#ifdef ENABLE_FP4
INSTANTIATE_MOE_ACTIVATION(half, __nv_fp4_e2m1, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_ACTIVATION(__nv_bfloat16, __nv_fp4_e2m1, uint8_t);
#endif
#endif
#undef INSTANTIATE_MOE_ACTIVATION

} // namespace kernels::cute_dsl

TRTLLM_NAMESPACE_END
