/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expert_routed_mxfp4.hpp"

using namespace pto;
using namespace grouped_mxfp4;

#ifdef __DAV_CUBE__
namespace {
struct StorePair {
    __gm__ bfloat16_t* out;
    int64_t rows;
    int64_t width;
    int64_t row;
    int64_t col;

    template <typename Acc>
    AICORE inline void operator()(Acc& acc) const
    {
        using Global = GlobalTensor<bfloat16_t, Shape<1, 1, 1, DYNAMIC, N_TILE>,
                                    pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
        Global target(out + row * width + col,
                      Global::Shape(1, 1, 1, acc.GetValidRow(), N_TILE),
                      Global::Stride(rows * width, rows * width, rows * width, width, 1));
        TSTORE<STPhase::Final>(target, acc);
    }
};
}  // namespace
#endif

#ifdef __DAV_VEC__
AICORE inline void copyRouteWeights(__gm__ float* input, __gm__ float* output, int64_t count)
{
    using W = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, DYNAMIC>;
    using G = GlobalTensor<float, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
    W tile(count);
    TASSIGN(tile, 0);
    G source(input, G::Shape(1, 1, 1, 1, count));
    G target(output, G::Shape(1, 1, 1, 1, count));
    TLOAD(tile, source);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    TSTORE(target, tile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    auto* counts = tensorData<int32_t>(args, 2);
#ifdef EXPERT_SLOT_MAP
    constexpr int scalarBase = 11;
#else
    constexpr int scalarBase = 10;
#endif
    const int64_t experts = args[scalarBase];
    const int64_t stride = args[scalarBase + 1];
    const int64_t width = args[scalarBase + 2];
    const int64_t hidden = args[scalarBase + 3];
    const int64_t activationRows = args[scalarBase + 4];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
    const int64_t hTiles = hidden / N_TILE;
    const int64_t rowLimit = selectRowTileLimit(counts, experts, 2 * hTiles, blocks);
#ifdef __DAV_VEC__
    auto* w1 = tensorData<uint8_t>(args, 3);
    auto* w3 = tensorData<uint8_t>(args, 5);
#endif
#ifdef __DAV_CUBE__
    auto* x = tensorData<float8_e4m3_t>(args, 0);
    auto* xScale = tensorData<float8_e8m0_t>(args, 1);
    auto* s1 = tensorData<float8_e8m0_t>(args, 4);
    auto* s3 = tensorData<float8_e8m0_t>(args, 6);
    auto* out = tensorData<bfloat16_t>(args, 7);
#endif
    beginGemm();
    int64_t workBase = 0;
    int64_t previousRowTile = 0;
    int64_t activationBase = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
        const int64_t count = acquireCount(counts, expert);
#ifdef EXPERT_SLOT_MAP
        // Each slot owns a complete, independently packed MX_B_NN scale tensor.
        const int64_t slot = count ? acquireCount(tensorData<int32_t>(args, 10), expert) : 0;
        const int64_t scaleExperts = 1;
        const int64_t scaleExpert = 0;
        const int64_t scaleOffset = slot * (width / 32) * hidden;
#else
        const int64_t slot = expert;
        const int64_t scaleExperts = experts;
        const int64_t scaleExpert = expert;
        const int64_t scaleOffset = 0;
#endif
        const int64_t countTile = selectRowTile(count);
        const int64_t rowTile = countTile < rowLimit ? countTile : rowLimit;
        if (count > 0) {
            // Drain both lanes before changing the shared L1 partition.
            if (previousRowTile != 0 && rowTile != previousRowTile) {
                endGemm();
                beginGemm();
            }
            previousRowTile = rowTile;
        }
        const int64_t workCount = ((count + rowTile - 1) / rowTile) * 2 * hTiles;
        const int64_t first = (block + blocks - workBase % blocks) % blocks;
        for (int64_t work = first; work < workCount; work += blocks) {
            const int64_t row = work / (2 * hTiles) * rowTile;
            const int64_t col = work % hTiles * N_TILE;
            const bool up = (work / hTiles) % 2 != 0;
            const int64_t valid = count - row < rowTile ? count - row : rowTile;
#ifdef __DAV_VEC__
            if (work % (2 * hTiles) == 0 && get_subblockid() == 0) {
                copyRouteWeights(tensorData<float>(args, 8) + expert * stride + row,
                                 tensorData<float>(args, 9) + activationBase + row, valid);
            }
            dispatchWeight(rowTile, up ? w3 : w1, width, hidden, slot, col);
#endif
#ifdef __DAV_CUBE__
            const Chunk chunk{scaleExpert, expert * stride + row, valid};
            dispatchProjection(rowTile, x, xScale, (up ? s3 : s1) + scaleOffset, width, hidden, scaleExperts,
                                      experts * stride, chunk, col,
                                      StorePair{out, activationRows, 2 * hidden,
                                                activationBase + row,
                                                col + (up ? hidden : 0)});
#endif
        }
        workBase += workCount;
        activationBase += count;
    }
    endGemm();
}
