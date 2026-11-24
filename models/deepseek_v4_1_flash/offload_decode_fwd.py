# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Decode backbone for the host-backed expert slot bank."""

import argparse
import pypto.language as pl

import pypto.language.distributed as pld
import config as C

if C.EP_SIZE % C.LM_HEAD_TP_SIZE:
    raise ValueError("EP must be divisible by LM head TP")

_shape_parser = argparse.ArgumentParser(add_help=False)
_default_experts = C.FLASH.n_routed_experts // C.EP_SIZE
_shape_parser.add_argument("--experts-per-rank", type=int, default=_default_experts)
_shape_parser.add_argument("--weight-bank-size", type=int, default=C.FLASH.num_hidden_layers)
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.experts_per_rank <= 0 or C.EP_SIZE * _shape_options.experts_per_rank < C.FLASH.num_experts_per_tok:
    _shape_parser.error("--experts-per-rank must give at least TopK routed experts")
C.N_LOCAL_EXPERTS = _shape_options.experts_per_rank
WEIGHT_BANK_SIZE = _shape_options.weight_bank_size
if WEIGHT_BANK_SIZE not in (1, C.FLASH.num_hidden_layers):
    _shape_parser.error("--weight-bank-size must be 1 or the full layer count")


from config import COMPRESSED_POOL_UNIT_ROWS, C2A_STORAGE_ROWS, C1A_STORAGE_ROWS
from config import (
    FLASH as M, B_DYN, Q_START_DYN, ROUTE_T_DYN, T_DYN, BLOCK_SIZE, DECODE_MAX_TOKENS,
    DECODE_INPUT_MAX_TOKENS, INDEX_CACHE_GROUP, STATE_CAPACITY, MOE_TOKENS, DSPARK_DRAFT_LAYERS,
    DSPARK_LAST_TARGET_LAYER, DSPARK_TARGET_LAYER_IDS, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)
from compressor import STATE_WIDTH
from decode_c1a_full import decode_c1a_full
from decode_c1a_reindex import decode_c1a_reindex
from decode_c1a_reuse import decode_c1a_reuse
from decode_c2a_full import decode_c2a_full
from decode_c2a_reuse import decode_c2a_reuse
from decode_swa import decode_swa
from input_pack import prepare_compressed_lens
from rope_tables import ROPE_ROWS_DYN, materialize_rope_rows
from engram import ENGRAM_K, KV_OUT, engram_from_rows
from expert_context import SCALE_BANK_LAYOUT, EXPERT_SLOTS, RankDescriptors, RankState
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS
from expert_routed import DISPATCH_PACKET_WIDTH
from offload_decode_moe import (
    BALANCED_ROUTING, initialize_balanced_routes, moe, SIGNAL_PAD, EP_SIZE, N_LOCAL_EXPERTS, RECV_MAX, ROUTE_WIDTH,
)


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
D = M.hidden_size
DSPARK_MAIN_IN = DSPARK_DRAFT_LAYERS * D
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
CANDIDATE_BLOCK_SIZE = M.candidate_block_size
CANDIDATE_TOPK_BLOCKS = M.candidate_topk_blocks
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
MX_GROUP = M.weight_block_size[0]


N_LAYERS = M.num_hidden_layers
C2A_SOURCE_COUNT = 3
KV_SOURCE_COUNT = 4
COMPRESSED_SOURCE_UNITS = 5
INDEX_SOURCE_COUNT = 8

BLOCK_SIZE = BLOCK_SIZE
DECODE_MAX_TOKENS = DECODE_MAX_TOKENS
STATE_CAPACITY = STATE_CAPACITY

FIXTURE_BATCH = 12
FIXTURE_SEQUENCE = 6
FIXTURE_TOKENS = FIXTURE_BATCH * FIXTURE_SEQUENCE


FWD_ORI_BLOCKS_DYN = pl.dynamic("FWD_ORI_BLOCKS_DYN")
FWD_CMP_BLOCKS_DYN = pl.dynamic("FWD_CMP_BLOCKS_DYN")
FWD_INDEX_BLOCKS_DYN = pl.dynamic("FWD_INDEX_BLOCKS_DYN")
FWD_STATE_BLOCKS_DYN = pl.dynamic("FWD_STATE_BLOCKS_DYN")
FWD_C2A_TABLE_DYN = pl.dynamic("FWD_C2A_TABLE_DYN")
FWD_C1A_TABLE_DYN = pl.dynamic("FWD_C1A_TABLE_DYN")


ROUTED_BANK_ROWS = EXPERT_SLOTS
ROUTED_LAYER_ROWS = EXPERT_SLOTS
ROUTED_LAYER_STRIDE = 0


@pl.jit.incore
def capture_dspark_target_hidden(
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    target_hidden: pl.InOut[pl.Tensor[[T_DYN, DSPARK_MAIN_IN], pl.BF16]],
    active_rows: pl.Scalar[pl.INT32],
    target_column: pl.Scalar[pl.INDEX],
):
    """Register the residual as InOut so the next layer waits for this snapshot."""
    token = pl.tile.get_block_idx()
    if token < active_rows:
        hc_row = x_hc[token : token + 1, 0:HC_MULT, 0:D]
        hc_mean = pl.mul(pl.col_sum(pl.reshape(hc_row, [HC_MULT, D])), 1.0 / HC_MULT)
        target_hidden[token : token + 1, target_column : target_column + D] = pl.cast(hc_mean, pl.BF16, mode="rint")
    return x_hc, target_hidden


@pl.jit.inline(auto_scope=False)
def decode_encoder(
    engram_lookup: pl.Tensor[[2, T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    hc_attn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    token_rope_cos: pl.Out[pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32]],
    token_rope_sin: pl.Out[pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32]],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2: pl.Tensor[[ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3: pl.Tensor[[ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    shared_w1: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], pl.INT8],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2_scale: pl.Tensor[[ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    expert_state: RankState = 0,
    expert_descriptors: RankDescriptors = 0,
) -> tuple[
    pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
    pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
]:
    attention_count = pl.read(attention_num_tokens, [rank])
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    window_cache_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    c2a_compressed_slots.bind_dynamic(0, T_DYN)
    c2a_index_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(1, FWD_C2A_TABLE_DYN)
    state_cache_pool.bind_dynamic(0, FWD_STATE_BLOCKS_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    position_ids.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(1, ROPE_ROWS_DYN)
    compressed_token_rope_cos = pl.slice(token_rope_cos, [tokens, ROPE_DIM // 2], [0, 0])
    compressed_token_rope_sin = pl.slice(token_rope_sin, [tokens, ROPE_DIM // 2], [0, 0])
    base_rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    base_rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    c2a_rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    c2a_rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    materialize_rope_rows(
        freqs_cos[0], freqs_sin[0], position_ids, attention_count, base_rope_cos, base_rope_sin, False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids, attention_count,
        compressed_token_rope_cos, compressed_token_rope_sin, False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids, attention_count, c2a_rope_cos, c2a_rope_sin, True,
        worker_limit=32,
    )
    c2a_compressed_lens = pl.create_tensor([tokens], dtype=pl.INT32)
    prepare_compressed_lens(position_ids, c2a_compressed_lens, 1)
    window_blocks_per_layer = pl.tensor.dim(window_cache_pool, 0) // N_LAYERS
    compressed_blocks_per_source = pl.tensor.dim(compressed_cache_pool, 0) // COMPRESSED_SOURCE_UNITS
    index_blocks_per_source = pl.tensor.dim(index_cache_pool, 0) // COMPRESSED_SOURCE_UNITS
    state_blocks_per_source = pl.tensor.dim(state_cache_pool, 0) // C2A_SOURCE_COUNT
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    topk_indices = pl.create_tensor([tokens, INDEX_TOPK], dtype=pl.INT32)
    x_mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
    if BALANCED_ROUTING:
        initialize_balanced_routes(force_topk, local_moe_tokens, rank, N_LOCAL_EXPERTS, TOPK)
    recv_x_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.FP8E4M3FN)
    recv_x_last_use = pl.array.create(1, pl.TASK_ID)
    for layer in pl.range(0, 20):
        if layer == 1 or layer == 14:
            with pl.scope():
                engram_slot = layer // 14
                layer_engram_scale = pl.slice(engram_wkv_scale, [ENGRAM_K // 32, KV_OUT], [engram_slot * (ENGRAM_K // 32), 0])
                engram_from_rows(
                    engram_lookup[engram_slot], engram_wkv[engram_slot], layer_engram_scale,
                    engram_weight[engram_slot], x_hc, x_hc,
                    mask_zero_rows=False,
                )
        with pl.scope():
            weight_layer = layer % WEIGHT_BANK_SIZE
            layer_window_cache = pl.slice(window_cache_pool, [window_blocks_per_layer, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], [layer * window_blocks_per_layer, 0, 0, 0])
            layer_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [weight_layer * MIX_HC, 0])
            layer_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [weight_layer * 3])
            layer_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [weight_layer * MIX_HC])
            layer_norm_weight: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [weight_layer * D])
            layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[weight_layer]
            layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [weight_layer * (D // MX_GROUP), 0])
            layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(q_norm_weight, [Q_LORA], [weight_layer * Q_LORA])
            layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[weight_layer]
            layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [weight_layer * (Q_LORA // MX_GROUP), 0])
            layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[weight_layer]
            layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [weight_layer * (D // MX_GROUP), 0])
            layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(kv_norm_weight, [HEAD_DIM], [weight_layer * HEAD_DIM])
            layer_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [weight_layer * LOCAL_H])
            layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [weight_layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = pl.slice(wo_a_scale, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [weight_layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[weight_layer]
            layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [weight_layer * (LOCAL_O_WIDTH // MX_GROUP), 0])
            if layer < 2:
                decode_swa(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    base_rope_cos, base_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            elif (layer - 2) % 6 == 0:
                full_source_id = (layer - 2) // 6
                full_compressed_cache = pl.slice(compressed_cache_pool, [compressed_blocks_per_source, C2A_STORAGE_ROWS, 1, COMPRESSED_CACHE_BYTES], [full_source_id * compressed_blocks_per_source, 0, 0, 0])
                full_index_cache = pl.slice(index_cache_pool, [index_blocks_per_source, C2A_STORAGE_ROWS, 1, INDEX_DIM // 2], [full_source_id * index_blocks_per_source, 0, 0, 0])
                full_index_cache_scale = pl.slice(index_cache_scale_pool, [index_blocks_per_source, C2A_STORAGE_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], [full_source_id * index_blocks_per_source, 0, 0, 0])
                full_state_cache = pl.slice(state_cache_pool, [state_blocks_per_source, STATE_CAPACITY, STATE_WIDTH], [full_source_id * state_blocks_per_source, 0, 0])
                full_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [(layer - 2) // 6 * (Q_LORA // MX_GROUP), 0])
                full_c2a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wkv[full_source_id]
                full_c2a_compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wgate[full_source_id]
                full_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [full_source_id * HEAD_DIM])
                full_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[full_source_id]
                full_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [full_source_id * INDEX_DIM])
                full_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = index_wq_b[full_source_id]
                full_index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[full_source_id]
                decode_c2a_full(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache,
                    full_compressed_cache,
                    token_to_req_indices, c2a_compressed_lens,
                    full_index_cache, full_index_cache_scale,
                    c2a_index_block_table, position_ids,
                    c2a_rope_cos, c2a_rope_sin,
                    full_c2a_compressor_wkv, full_c2a_compressor_wgate,
                    query_start_loc, state_block_table,
                    full_state_cache, full_compressor_norm,
                    c2a_compressed_slots,
                    full_index_wk, full_index_norm,
                    full_index_wq_b, full_index_wq_b_scale, full_index_weights_proj,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            else:
                reuse_source_id = (layer - 2) // 6
                reuse_compressed_cache = pl.slice(compressed_cache_pool, [compressed_blocks_per_source, C2A_STORAGE_ROWS, 1, COMPRESSED_CACHE_BYTES], [reuse_source_id * compressed_blocks_per_source, 0, 0, 0])
                decode_c2a_reuse(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache,
                    reuse_compressed_cache,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            ffn_routed_base = weight_layer * ROUTED_LAYER_STRIDE
            ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [weight_layer * MIX_HC, 0])
            ffn_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [weight_layer * 3])
            ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [weight_layer * MIX_HC])
            layer_ffn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_weight, [D], [weight_layer * D])
            ffn_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16] = pl.slice(gate_weight, [N_EXPERTS, D], [weight_layer * N_EXPERTS, 0])
            ffn_correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(correction_bias, [N_EXPERTS], [weight_layer * N_EXPERTS])
            ffn_routed_w1: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w1, [ROUTED_LAYER_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w1_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w1_scale, [ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], [weight_layer * (ROUTED_LAYER_STRIDE * (D // MX_GROUP)), 0])
            ffn_routed_w2: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w2, [ROUTED_LAYER_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w2_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w2_scale, [ROUTED_LAYER_ROWS * (MOE_INTER // MX_GROUP), D], [weight_layer * (ROUTED_LAYER_STRIDE * (MOE_INTER // MX_GROUP)), 0])
            ffn_routed_w3: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w3, [ROUTED_LAYER_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w3_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w3_scale, [ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], [weight_layer * (ROUTED_LAYER_STRIDE * (D // MX_GROUP)), 0])
            ffn_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[weight_layer]
            ffn_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [weight_layer * (D // MX_GROUP), 0])
            ffn_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[weight_layer]
            ffn_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [weight_layer * (MOE_INTER // MX_GROUP), 0])
            ffn_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[weight_layer]
            ffn_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [weight_layer * (D // MX_GROUP), 0])
            moe(
                attention_hidden, attention_pre_mix,
                ffn_hc_fn, ffn_hc_scale, ffn_hc_base,
                layer_ffn_norm,
                ffn_gate_weight, ffn_correction_bias,
                ffn_routed_w1, ffn_routed_w1_scale,
                ffn_routed_w2, ffn_routed_w2_scale,
                ffn_routed_w3, ffn_routed_w3_scale,
                ffn_shared_w1, ffn_shared_w1_scale,
                ffn_shared_w2, ffn_shared_w2_scale,
                ffn_shared_w3, ffn_shared_w3_scale,
                pre_mix, x_mixed, x_hc,
                recv_meta, recv_x, recv_routes,
                data_arrived,
                routed_output, combine_arrived,
                local_moe_tokens, rank, layer + 1,
                force_topk, recv_x_local, recv_x_last_use,
                expert_state=expert_state, expert_descriptors=expert_descriptors,
                expert_layer=layer, local_experts=N_LOCAL_EXPERTS, topk=TOPK, dispatch_worker_limit=0,
            )
    return x_hc, pre_mix, token_rope_cos, token_rope_sin


@pl.jit.inline(auto_scope=False)
def decode_decoder(
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[T_DYN, DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    token_rope_cos: pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
    token_rope_sin: pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2: pl.Tensor[[ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3: pl.Tensor[[ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    shared_w1: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], pl.INT8],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2_scale: pl.Tensor[[ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    expert_state: RankState = 0,
    expert_descriptors: RankDescriptors = 0,
):
    attention_count = pl.read(attention_num_tokens, [rank])
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    dspark_target_hidden.bind_dynamic(0, T_DYN)
    window_cache_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    c1a_compressed_slots.bind_dynamic(0, T_DYN)
    c1a_index_block_table.bind_dynamic(0, B_DYN)
    c1a_index_block_table.bind_dynamic(1, FWD_C1A_TABLE_DYN)
    state_cache_pool.bind_dynamic(0, FWD_STATE_BLOCKS_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    position_ids.bind_dynamic(0, T_DYN)
    compressed_token_rope_cos = pl.slice(token_rope_cos, [tokens, ROPE_DIM // 2], [0, 0])
    compressed_token_rope_sin = pl.slice(token_rope_sin, [tokens, ROPE_DIM // 2], [0, 0])
    c1a_compressed_lens = pl.create_tensor([tokens], dtype=pl.INT32)
    prepare_compressed_lens(position_ids, c1a_compressed_lens, 0)
    window_blocks_per_layer = pl.tensor.dim(window_cache_pool, 0) // N_LAYERS
    compressed_blocks_per_source = pl.tensor.dim(compressed_cache_pool, 0) // COMPRESSED_SOURCE_UNITS
    index_blocks_per_source = pl.tensor.dim(index_cache_pool, 0) // COMPRESSED_SOURCE_UNITS
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    topk_indices = pl.create_tensor([tokens, INDEX_TOPK], dtype=pl.INT32)
    pool_blocks_raw = pl.tensor.dim(c1a_index_block_table, 1) * (BLOCK_SIZE // CANDIDATE_BLOCK_SIZE)
    pool_block_capacity = pl.min(CANDIDATE_TOPK_BLOCKS, pl.max(32, (pool_blocks_raw + 31) // 32 * 32))
    pool_block_ids = pl.create_tensor([tokens, pool_block_capacity], dtype=pl.INT32)
    pool_block_counts = pl.create_tensor([tokens], dtype=pl.INT32)
    x_mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
    if BALANCED_ROUTING:
        initialize_balanced_routes(force_topk, local_moe_tokens, rank, N_LOCAL_EXPERTS, TOPK)
    recv_x_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.FP8E4M3FN)
    recv_x_last_use = pl.array.create(1, pl.TASK_ID)
    for layer in pl.range(20, 40):
        with pl.scope():
            weight_layer = layer % WEIGHT_BANK_SIZE
            layer_window_cache = pl.slice(window_cache_pool, [window_blocks_per_layer, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], [layer * window_blocks_per_layer, 0, 0, 0])
            layer_source_id = 3
            layer_compressed_cache_units = pl.slice(compressed_cache_pool, [2 * compressed_blocks_per_source, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], [layer_source_id * compressed_blocks_per_source, 0, 0, 0])
            layer_compressed_cache = pl.reshape(layer_compressed_cache_units, [compressed_blocks_per_source, C1A_STORAGE_ROWS, 1, COMPRESSED_CACHE_BYTES])
            layer_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [weight_layer * MIX_HC, 0])
            layer_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [weight_layer * 3])
            layer_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [weight_layer * MIX_HC])
            layer_norm_weight: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [weight_layer * D])
            layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[weight_layer]
            layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [weight_layer * (D // MX_GROUP), 0])
            layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(q_norm_weight, [Q_LORA], [weight_layer * Q_LORA])
            layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[weight_layer]
            layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [weight_layer * (Q_LORA // MX_GROUP), 0])
            layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[weight_layer]
            layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [weight_layer * (D // MX_GROUP), 0])
            layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(kv_norm_weight, [HEAD_DIM], [weight_layer * HEAD_DIM])
            layer_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [weight_layer * LOCAL_H])
            layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [weight_layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = pl.slice(wo_a_scale, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [weight_layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[weight_layer]
            layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [weight_layer * (LOCAL_O_WIDTH // MX_GROUP), 0])
            if layer == 20:
                full_index_source_id = 3
                full_index_cache_units = pl.slice(index_cache_pool, [2 * index_blocks_per_source, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], [full_index_source_id * index_blocks_per_source, 0, 0, 0])
                full_index_cache = pl.reshape(full_index_cache_units, [index_blocks_per_source, C1A_STORAGE_ROWS, 1, INDEX_DIM // 2])
                full_index_cache_scale_units = pl.slice(index_cache_scale_pool, [2 * index_blocks_per_source, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], [full_index_source_id * index_blocks_per_source, 0, 0, 0])
                full_index_cache_scale = pl.reshape(full_index_cache_scale_units, [index_blocks_per_source, C1A_STORAGE_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP])
                full_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [3 * (Q_LORA // MX_GROUP), 0])
                full_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [layer_source_id * HEAD_DIM])
                full_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[full_index_source_id]
                full_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [full_index_source_id * INDEX_DIM])
                full_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = index_wq_b[full_index_source_id]
                full_index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[full_index_source_id]
                decode_c1a_full(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache,
                    layer_compressed_cache,
                    token_to_req_indices, c1a_compressed_lens,
                    full_index_cache, full_index_cache_scale,
                    c1a_index_block_table,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    c1a_compressor_wkv, full_compressor_norm,
                    c1a_compressed_slots,
                    full_index_wk, full_index_norm,
                    full_index_wq_b, full_index_wq_b_scale, full_index_weights_proj,
                    topk_indices, pool_block_ids, pool_block_counts,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            elif (layer - 20) % 4 == 0:
                reindex_index_source_id = (layer - 24) // 4 + 4
                reindex_index_cache_units = pl.slice(index_cache_pool, [2 * index_blocks_per_source, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], [layer_source_id * index_blocks_per_source, 0, 0, 0])
                reindex_index_cache = pl.reshape(reindex_index_cache_units, [index_blocks_per_source, C1A_STORAGE_ROWS, 1, INDEX_DIM // 2])
                reindex_index_cache_scale_units = pl.slice(index_cache_scale_pool, [2 * index_blocks_per_source, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], [layer_source_id * index_blocks_per_source, 0, 0, 0])
                reindex_index_cache_scale = pl.reshape(reindex_index_cache_scale_units, [index_blocks_per_source, C1A_STORAGE_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP])
                reindex_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [((layer - 24) // 4 + 4) * (Q_LORA // MX_GROUP), 0])
                reindex_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = index_wq_b[reindex_index_source_id]
                reindex_index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[reindex_index_source_id]
                decode_c1a_reindex(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache,
                    layer_compressed_cache,
                    token_to_req_indices, c1a_compressed_lens,
                    reindex_index_cache, reindex_index_cache_scale,
                    c1a_index_block_table,
                    pool_block_ids, pool_block_counts,
                    reindex_index_wq_b, reindex_index_wq_b_scale, reindex_index_weights_proj,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            else:
                decode_c1a_reuse(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache,
                    layer_compressed_cache,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            ffn_routed_base = weight_layer * ROUTED_LAYER_STRIDE
            ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [weight_layer * MIX_HC, 0])
            ffn_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [weight_layer * 3])
            ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [weight_layer * MIX_HC])
            layer_ffn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_weight, [D], [weight_layer * D])
            ffn_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16] = pl.slice(gate_weight, [N_EXPERTS, D], [weight_layer * N_EXPERTS, 0])
            ffn_correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(correction_bias, [N_EXPERTS], [weight_layer * N_EXPERTS])
            ffn_routed_w1: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w1, [ROUTED_LAYER_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w1_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w1_scale, [ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], [weight_layer * (ROUTED_LAYER_STRIDE * (D // MX_GROUP)), 0])
            ffn_routed_w2: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w2, [ROUTED_LAYER_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w2_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w2_scale, [ROUTED_LAYER_ROWS * (MOE_INTER // MX_GROUP), D], [weight_layer * (ROUTED_LAYER_STRIDE * (MOE_INTER // MX_GROUP)), 0])
            ffn_routed_w3: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w3, [ROUTED_LAYER_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w3_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w3_scale, [ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], [weight_layer * (ROUTED_LAYER_STRIDE * (D // MX_GROUP)), 0])
            ffn_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[weight_layer]
            ffn_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [weight_layer * (D // MX_GROUP), 0])
            ffn_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[weight_layer]
            ffn_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [weight_layer * (MOE_INTER // MX_GROUP), 0])
            ffn_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[weight_layer]
            ffn_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [weight_layer * (D // MX_GROUP), 0])
            moe(
                attention_hidden, attention_pre_mix,
                ffn_hc_fn, ffn_hc_scale, ffn_hc_base,
                layer_ffn_norm,
                ffn_gate_weight, ffn_correction_bias,
                ffn_routed_w1, ffn_routed_w1_scale,
                ffn_routed_w2, ffn_routed_w2_scale,
                ffn_routed_w3, ffn_routed_w3_scale,
                ffn_shared_w1, ffn_shared_w1_scale,
                ffn_shared_w2, ffn_shared_w2_scale,
                ffn_shared_w3, ffn_shared_w3_scale,
                pre_mix, x_mixed, x_hc,
                recv_meta, recv_x, recv_routes,
                data_arrived,
                routed_output, combine_arrived,
                local_moe_tokens, rank, layer + 1,
                force_topk, recv_x_local, recv_x_last_use,
                expert_state=expert_state, expert_descriptors=expert_descriptors,
                expert_layer=layer, local_experts=N_LOCAL_EXPERTS, topk=TOPK, dispatch_worker_limit=0,
            )
        # This residual enters the next layer's attention.
        target_layer = layer + 1
        if target_layer >= DSPARK_TARGET_LAYER_IDS[0] and target_layer <= DSPARK_LAST_TARGET_LAYER:
            target_column = (target_layer - DSPARK_TARGET_LAYER_IDS[0]) * D
            (x_hc, dspark_target_hidden), _capture_tid = pl.spmd_submit(
                capture_dspark_target_hidden, x_hc, dspark_target_hidden, local_moe_tokens,
                target_column, core_num=MOE_TOKENS,
            )
    return x_hc


@pl.jit.inline(auto_scope=False)
def decode_fwd(
    engram_lookup: pl.Tensor[[2, T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[T_DYN, DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2: pl.Tensor[[ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3: pl.Tensor[[ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    shared_w1: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], pl.INT8],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2_scale: pl.Tensor[[ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    expert_state: RankState = 0,
    expert_descriptors: RankDescriptors = 0,
):
    """Run the decode backbone and retain final HC state in the input buffers."""
    compressed_token_rope_cos = pl.create_tensor([DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    compressed_token_rope_sin = pl.create_tensor([DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    encoder_hidden, encoder_pre_mix, token_cos, token_sin = decode_encoder(
        engram_lookup, engram_wkv, engram_wkv_scale, engram_weight,
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, q_norm_weight, wq_b,
        wkv, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b,
        freqs_cos, freqs_sin,
        compressed_token_rope_cos, compressed_token_rope_sin,
        window_slots, window_indices,
        window_cache_pool,
        compressed_cache_pool,
        token_to_req_indices, position_ids,
        index_cache_pool, index_cache_scale_pool,
        c2a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv,
        query_start_loc, state_block_table, state_cache_pool,
        compressor_norm_weight,
        c2a_compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_weights_proj,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w2, routed_w3,
        shared_w1, shared_w2, shared_w3,
        recv_meta, recv_x, recv_routes,
        data_arrived, routed_output, combine_arrived,
        moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale,
        expert_state=expert_state, expert_descriptors=expert_descriptors,
    )
    result = decode_decoder(
        encoder_hidden, encoder_pre_mix, dspark_target_hidden,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, q_norm_weight, wq_b,
        wkv, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b,
        token_cos, token_sin,
        window_slots, window_indices,
        window_cache_pool,
        compressed_cache_pool,
        token_to_req_indices, position_ids,
        index_cache_pool, index_cache_scale_pool,
        c1a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv,
        query_start_loc, state_block_table, state_cache_pool,
        compressor_norm_weight,
        c1a_compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_weights_proj,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w2, routed_w3,
        shared_w1, shared_w2, shared_w3,
        recv_meta, recv_x, recv_routes,
        data_arrived, routed_output, combine_arrived,
        moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale,
        expert_state=expert_state, expert_descriptors=expert_descriptors,
    )
    return result, encoder_pre_mix
