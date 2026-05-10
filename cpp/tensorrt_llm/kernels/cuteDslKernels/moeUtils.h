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

#pragma once
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cute_dsl
{
template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_output, SFType const* input_sf, SFType* permuted_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens, int32_t const hidden_size,
    int32_t const top_k, int32_t const tile_size, cudaStream_t stream);

template <typename InputType, typename TopKScaleType>
void moeUnpermute(InputType const* permuted_input, InputType* output, int32_t const* expanded_idx_to_permuted_idx,
    TopKScaleType const* topk_scales, int32_t const num_tokens, int32_t const hidden_size, int32_t const top_k,
    cudaStream_t stream);

template <typename InputType>
void moeOutputMemset(InputType* input, int32_t const* tile_idx_to_mn_limit, int32_t const* expanded_idx_to_permuted_idx,
    int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
    int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size,
    cudaStream_t stream);

void megaMoeM5MaterializeFromMoeSort(uint8_t const* input, uint8_t const* inputSf, float const* topKScales,
    int32_t const* tokenOffsets, int32_t const* tileIdxToMnLimit, int32_t const* permutedIdxToExpandedIdx,
    int32_t const* numNonExitingTiles, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int32_t totalTokens, int32_t epSize, int32_t hiddenPackedSize,
    int32_t sfHiddenSize, int32_t topK, int32_t tileSize, int32_t numAvailablePoolSlots, int32_t numPaddedSfPoolTokens,
    int32_t l1ArrivalCountEntries, cudaStream_t stream);

void megaMoeM5MaterializeDirectFromMoeSort(uint8_t const* input, uint8_t const* inputSf, float const* topKScales,
    int32_t const* tokenOffsets, int32_t const* tileIdxToMnLimit, int32_t const* permutedIdxToExpandedIdx,
    int32_t const* numNonExitingTiles, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int64_t* activePoolSlots, int64_t* activeCombineRows,
    int32_t* activeRouteCount, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t totalTokens, int32_t epSize, int32_t hiddenPackedSize, int32_t sfHiddenSize, int32_t topK, int32_t tileSize,
    int32_t numAvailablePoolSlots, int32_t numPaddedSfPoolTokens, int32_t l1ArrivalCountEntries,
    int32_t maxNumTokensPerRank, int32_t combineLayoutRows, int32_t outputMappingRows, int32_t outputScaleRows,
    cudaStream_t stream);

void megaMoeM5MaterializeDirectFromTopK(uint8_t const* input, uint8_t const* inputSf, int64_t const* topKIdx,
    float const* topKScales, int32_t const* tokenOffsets, int64_t const* expertRecvCountSum,
    int32_t* expertRouteOffsets, uint8_t* l1ActsPool, uint8_t* l1ActsSfPool, float* l1TopKWeightsPool,
    int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount, int64_t* activePoolSlots, int64_t* activeCombineRows,
    int32_t* activeRouteCount, int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t totalTokens,
    int32_t epSize, int32_t localRank, int32_t numExpertsPerRank, int32_t hiddenPackedSize, int32_t sfHiddenSize,
    int32_t topK, int32_t tileSize, int32_t numPoolSlots, int32_t numPaddedSfPoolTokens, int32_t l1ArrivalCountEntries,
    int32_t maxNumTokensPerRank, int32_t combineLayoutRows, int32_t outputMappingRows, int32_t outputScaleRows,
    int32_t routeLayoutCapacity, cudaStream_t stream);

void megaMoeM5MaterializeDirectFromRankedTopK(uint8_t const* input, int64_t inputRankStride, int64_t inputTokenStride,
    uint8_t const* inputSf, int64_t inputSfRankStride, int64_t inputSfTokenStride, int64_t const* topKIdx,
    int64_t topKIdxRankStride, int64_t topKIdxTokenStride, float const* topKScales, int64_t topKScalesRankStride,
    int64_t topKScalesTokenStride, int32_t const* tokenCounts, int32_t* expertRouteOffsets, uint8_t* l1ActsPool,
    uint8_t* l1ActsSfPool, float* l1TopKWeightsPool, int32_t* tokenSrcMetadata, int32_t* l1ArrivalCount,
    int64_t* activePoolSlots, int64_t* activeCombineRows, int32_t* activeRouteCount,
    int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t epSize, int32_t localRank,
    int32_t numExpertsPerRank, int32_t hiddenPackedSize, int32_t sfHiddenSize, int32_t topK, int32_t tileSize,
    int32_t numPoolSlots, int32_t numPaddedSfPoolTokens, int32_t l1ArrivalCountEntries, int32_t maxNumTokensPerRank,
    int32_t combineLayoutRows, int32_t outputMappingRows, int32_t outputScaleRows, int32_t routeLayoutCapacity,
    cudaStream_t stream);

void megaMoeStageDispatchInputs(uint8_t const* input, int64_t inputBytes, uint8_t const* inputSf, int64_t inputSfBytes,
    uint8_t const* topKIdx, int64_t topKIdxBytes, uint8_t const* topKScales, int64_t topKScalesBytes,
    uint8_t* inputBuffer, uint8_t* inputSfBuffer, uint8_t* topKIdxBuffer, uint8_t* topKScalesBuffer,
    cudaStream_t stream);

void megaMoeM5InitDirectInputRouteMetadata(int32_t* expertRouteOffsets, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t numExpertsPerRank, int32_t routeLayoutCapacity,
    cudaStream_t stream);

void megaMoeM5BuildDirectInputRouteFromRankedTopK(uint8_t const* input, int64_t inputRankStride,
    int64_t inputTokenStride, uint8_t const* inputSf, int64_t inputSfRankStride, int64_t inputSfTokenStride,
    uint8_t* directInput, uint8_t* directInputSf, int64_t const* topKIdx, int64_t topKIdxRankStride,
    int64_t topKIdxTokenStride, float const* topKScales, int64_t topKScalesRankStride, int64_t topKScalesTokenStride,
    int32_t const* tokenCounts, int32_t* expertRouteOffsets, int32_t* expertRouteBaseOffsets, int32_t* tokenIdMapping,
    int32_t* outputPermutedIdxToExpandedIdx, float* outputTokenFinalScales, int32_t* tileIdxToExpertIdx,
    int32_t* tileIdxToMnLimit, int32_t* numNonExitingTiles, int32_t epSize, int32_t localRank,
    int32_t numExpertsPerRank, int32_t hiddenPackedSize, int32_t sfHiddenSize, int32_t topK, int32_t tileSize,
    int32_t numPoolSlots, int32_t maxNumTokensPerRank, int32_t combineLayoutRows, int32_t outputMappingRows,
    int32_t outputScaleRows, int32_t routeLayoutCapacity, bool directAtomicOutput, bool directTokenMajorOutput,
    cudaStream_t stream);

void megaMoeM6ReduceCombineBuffer(__nv_bfloat16 const* combineBuffer, float* output, int32_t topK, int32_t localTokens,
    int32_t maxNumTokensPerRank, int32_t hiddenSize, cudaStream_t stream);

void megaMoeM6ReduceCombineBufferOut(__nv_bfloat16 const* combineBuffer, float* output, int32_t topK,
    int32_t localTokens, int32_t maxNumTokensPerRank, int32_t hiddenSize, cudaStream_t stream);

void megaMoeM6ReduceCombineBufferBf16Out(__nv_bfloat16 const* combineBuffer, __nv_bfloat16* output, int32_t topK,
    int32_t localTokens, int32_t maxNumTokensPerRank, int32_t hiddenSize, cudaStream_t stream);

void megaMoeM6ReduceTokenMajorCombineBufferBf16Out(__nv_bfloat16 const* combineBuffer, __nv_bfloat16* output,
    int32_t topK, int32_t localTokens, int32_t maxNumTokensPerRank, int32_t hiddenSize, cudaStream_t stream);

template <typename InputType, typename OutputType, typename SFType>
void moeActivation(InputType const* input, OutputType* output, float const* global_sf, SFType* output_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* num_non_exiting_tiles,
    cutlass_kernels::ActivationParams activation_params, int32_t const max_num_permuted_tokens,
    int32_t const interm_size, int32_t const tile_size, cudaStream_t stream);

} // namespace kernels::cute_dsl

TRTLLM_NAMESPACE_END
