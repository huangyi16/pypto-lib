# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Shared prefill geometry and attention for the resident and offload graphs."""
import argparse

import pypto.language as pl
import pypto.language.distributed as pld

import config as C

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--checkpoint")
_shape_parser.add_argument("--ep", type=int, choices=C.SUPPORTED_EP_SIZES if C.EXPERT_SLOTS else [2, 4, 8])
_shape_parser.add_argument("--capacity", type=int, default=128)
_shape_parser.add_argument("--moe-tokens", type=int, default=1024)
_shape_parser.add_argument("--dp", type=int)
_shape_options = _shape_parser.parse_known_args()[0]
C.EP_SIZE = _shape_options.ep or (4 if _shape_options.checkpoint else 2)
_MAX_CHUNK_TOKENS = 4096
CAPACITY = _shape_options.capacity
CP_SIZE = C.CP_SIZE
LOCAL_CAPACITY = 2 * ((CAPACITY + 2 * CP_SIZE - 1) // (2 * CP_SIZE))
C.DP_SIZE = C.EP_SIZE // CP_SIZE
DP_SIZE = C.DP_SIZE
if C.EP_SIZE % CP_SIZE or _shape_options.dp not in (None, DP_SIZE):
    _shape_parser.error("prefill requires EP = CP * DP")
if C.EP_SIZE % C.LM_HEAD_TP_SIZE:
    _shape_parser.error("EP must be divisible by LM head TP")
if not 16 <= CAPACITY <= _MAX_CHUNK_TOKENS or CAPACITY % 16:
    _shape_parser.error("--capacity must be between 16 and 4096 and a multiple of 16")
# Keep MoE scratch bounded independently of the current-row capacity.
C.set_moe_tokens(min((LOCAL_CAPACITY + 15) // 16 * 16, _shape_options.moe_tokens))

from config import CACHE_ROWS_DYN
from config import (
    FLASH as M, INDEX_CACHE_GROUP, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, ORI_BLOCKS_DYN, B_DYN,
    INDEX_POOL_BLOCKS_DYN, PREFILL_MAX_TOKENS, Q_START_DYN, STATE_BLOCKS_DYN, STATE_CAPACITY, DSPARK_DRAFT_LAYERS, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)


BIAS_DYN = pl.dynamic("PREFILL_FORWARD_BIAS_DYN")
from compressor import STATE_WIDTH
from prefill_c1a_full import prefill_c1a_full, prefill_c1a_full_cp
from prefill_swa import prefill_swa, prefill_swa_cp
from prefill_c2a_full import prefill_c2a_full, prefill_c2a_full_cp
from prefill_c1a_reindex import prefill_c1a_reindex, prefill_c1a_reindex_cp
from prefill_c1a_reuse import prefill_c1a_reuse, prefill_c1a_reuse_cp

RANKS = C.EP_SIZE
D = M.hidden_size
DSPARK_MAIN_IN = DSPARK_DRAFT_LAYERS * D
HC_MULT = M.hc_mult
VOCAB = M.vocab_size
PAGES = pl.dynamic("PREFILL_HISTORY_PAGES")
WINDOW_PAGES = pl.dynamic("PREFILL_WINDOW_PAGES")
C2A_TABLE_DYN = pl.dynamic("PREFILL_C2A_TABLE")
C1A_TABLE_DYN = pl.dynamic("PREFILL_C1A_TABLE")
N_LAYERS = M.num_hidden_layers
RING_HEAP_BYTES = 4 * 1024**3
FINAL_SLOT = 2 * CAPACITY
C2A_SOURCE_COUNT = sum(M.compress_ratios[source] == 2 for source in M.kv_source_layer_ids)
KV_SOURCE_COUNT = len(M.kv_source_layer_ids)
INDEX_SOURCE_COUNT = len(M.index_source_layer_ids)
MX_GROUP = M.weight_block_size[0]
RECV_MAX = C.MOE_RECV_MAX
HC_DIM = M.hc_dim
HEAD_DIM = M.head_dim
INDEX_DIM = M.index_head_dim
INDEX_H = M.index_n_heads
INDEX_TOPK = M.index_topk
INDEX_POOL_BLOCKS = M.candidate_topk_blocks
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * M.o_lora_rank
MIX_HC = M.mix_hc
MOE_INTER = M.moe_intermediate_size
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
Q_LORA = M.q_lora_rank
TOPK = M.num_experts_per_tok


ROPE_DIM = M.qk_rope_head_dim


@pl.jit.inline(auto_scope=False)
def restore_window_tail(
    cache: pl.Tensor,
    source_slots: pl.Tensor,
    destination_slots: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
):
    """Copy final prefill rows from transient scratch into private window rings."""
    pages = pl.tensor.dim(cache, 1)
    rows = pages * 128
    flat = pl.reshape(cache, [40 * rows, WINDOW_CACHE_BYTES])
    tokens = pl.tensor.dim(destination_slots, 0)
    with pl.spmd(32, name_hint="prefill_window_tail", deps=[ready]) as restored:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, 40 * tokens, 32):
            layer = work // tokens
            token = work % tokens
            destination = pl.read(destination_slots, [token])
            if destination >= 0:
                source = pl.read(source_slots, [token])
                payload = pl.load(flat, [layer * rows + source, 0], [1, WINDOW_CACHE_BYTES])
                pl.store(payload, [layer * rows + destination, 0], flat)
    return restored


@pl.jit.inline(auto_scope=False)
def prefill_attention(
    hc_input: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[CAPACITY], pl.INT64],
    q_window_indices: pl.Tensor[[LOCAL_CAPACITY, 128], pl.INT32],
    q_rope_cos: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_request_ids: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    q_compressed_lens: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    kv_restore_indices: pl.Tensor[[CAPACITY], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, CACHE_ROWS_DYN, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    token_to_req_indices: pl.Tensor[[CAPACITY], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[CAPACITY], pl.INT32],
    compressed_rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    pool_block_ids: pl.Tensor[[LOCAL_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    topk_indices: pl.Tensor[[LOCAL_CAPACITY, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_hidden: pl.Tensor[[CAPACITY, D], pl.BF16],
    hc_output: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    local_tokens: pl.Scalar[pl.INT32],
    cp_group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    mode: pl.Scalar[pl.INT32],
    c1a_phase: pl.constexpr,
):
    if not c1a_phase:
        if mode == 0:
            prefill_swa(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, q_window_indices,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
        elif mode == 1:
            prefill_c2a_full(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, compressor_wkv,
                compressor_wgate, compressor_norm_weight, index_wk, index_norm_weight, index_wq_b,
                index_wq_b_scale, index_weights_proj, q_window_indices, q_request_ids, q_compressed_lens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_rope_cos, compressed_rope_sin, c2a_compressed_slots, token_to_req_indices, position_ids,
                query_start_loc, state_block_table, state_cache, compressed_cache, index_cache,
                index_cache_scale, c2a_index_block_table, kv_hidden, hc_output,
                hc_next_pre_mix, topk_indices,
            )
        elif mode == 2:
            prefill_c1a_reuse(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, q_window_indices,
                topk_indices, rope_cos, rope_sin, window_slots, num_tokens,
                window_cache, compressed_cache, kv_hidden, hc_output,
                hc_next_pre_mix,
            )
    else:
        if mode == 3:
            prefill_c1a_full(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, compressor_wkv,
                compressor_norm_weight, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
                index_weights_proj, q_window_indices, q_request_ids, q_compressed_lens, rope_cos,
                rope_sin, window_slots, num_tokens, window_cache, compressed_rope_cos,
                compressed_rope_sin, c2a_compressed_slots, compressed_cache, index_cache, index_cache_scale,
                c1a_index_block_table, kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices, pool_block_ids, pool_block_counts,
                publish_compressed=False,
            )
        elif mode == 4:
            prefill_c1a_reindex(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, index_wq_b,
                index_wq_b_scale, index_weights_proj, q_window_indices, q_request_ids, q_compressed_lens,
                pool_block_ids, pool_block_counts, rope_cos, rope_sin, window_slots,
                num_tokens, window_cache, compressed_cache, index_cache, index_cache_scale,
                c1a_index_block_table, kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices,
            )
        elif mode == 5:
            prefill_c1a_reuse(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, q_window_indices,
                topk_indices, rope_cos, rope_sin, window_slots, num_tokens,
                window_cache, compressed_cache, kv_hidden, hc_output,
                hc_next_pre_mix,
            )
    return hc_output, hc_next_pre_mix


@pl.jit.inline(auto_scope=False)
def prefill_attention_cp(
    hc_input: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[CAPACITY], pl.INT64],
    q_window_indices: pl.Tensor[[LOCAL_CAPACITY, 128], pl.INT32],
    q_rope_cos: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_request_ids: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    q_compressed_lens: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    kv_restore_indices: pl.Tensor[[CAPACITY], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, CACHE_ROWS_DYN, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    token_to_req_indices: pl.Tensor[[CAPACITY], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[CAPACITY], pl.INT32],
    compressed_rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    pool_block_ids: pl.Tensor[[LOCAL_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    topk_indices: pl.Tensor[[LOCAL_CAPACITY, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_hidden: pl.Tensor[[CAPACITY, D], pl.BF16],
    hc_output: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    local_tokens: pl.Scalar[pl.INT32],
    cp_group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    mode: pl.Scalar[pl.INT32],
    c1a_phase: pl.constexpr,
):
    if not c1a_phase:
        if mode == 0:
            prefill_swa_cp(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, q_rope_cos,
                q_rope_sin, q_window_indices, local_tokens, rope_cos, rope_sin,
                window_slots, num_tokens, window_cache, output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch, kv_hidden,
                hc_output, hc_next_pre_mix,
            )
        elif mode == 1:
            prefill_c2a_full_cp(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, compressor_wkv,
                compressor_wgate, compressor_norm_weight, index_wk, index_norm_weight, index_wq_b,
                index_wq_b_scale, index_weights_proj, q_rope_cos, q_rope_sin, q_window_indices,
                q_request_ids, q_compressed_lens, local_tokens, rope_cos, rope_sin,
                window_slots, num_tokens, window_cache, compressed_rope_cos, compressed_rope_sin,
                c2a_compressed_slots, token_to_req_indices, position_ids, query_start_loc, state_block_table,
                state_cache, compressed_cache, index_cache, index_cache_scale, c2a_index_block_table,
                output_window, output_arrived, kv_restore_indices, cp_group_base, cp_rank,
                attention_epoch, kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices,
            )
        elif mode == 2:
            prefill_c1a_reuse_cp(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, q_rope_cos,
                q_rope_sin, q_window_indices, topk_indices, local_tokens, rope_cos,
                rope_sin, window_slots, num_tokens, window_cache, compressed_cache,
                output_window, output_arrived, kv_restore_indices, cp_group_base, cp_rank,
                attention_epoch, kv_hidden, hc_output, hc_next_pre_mix,
            )
    else:
        if mode == 3:
            prefill_c1a_full_cp(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, compressor_wkv,
                compressor_norm_weight, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
                index_weights_proj, q_rope_cos, q_rope_sin, q_window_indices, q_request_ids,
                q_compressed_lens, local_tokens, rope_cos, rope_sin, window_slots,
                num_tokens, window_cache, compressed_rope_cos, compressed_rope_sin, c2a_compressed_slots,
                compressed_cache, index_cache, index_cache_scale, c1a_index_block_table, output_window,
                output_arrived, kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix, topk_indices,
                pool_block_ids, pool_block_counts,
                publish_compressed=False,
            )
        elif mode == 4:
            prefill_c1a_reindex_cp(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, index_wq_b,
                index_wq_b_scale, index_weights_proj, q_rope_cos, q_rope_sin, q_window_indices,
                q_request_ids, q_compressed_lens, pool_block_ids, pool_block_counts, local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache, index_cache, index_cache_scale, c1a_index_block_table, output_window,
                output_arrived, kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix, topk_indices,
            )
        elif mode == 5:
            prefill_c1a_reuse_cp(
                hc_input, hc_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale, q_rope_cos,
                q_rope_sin, q_window_indices, topk_indices, local_tokens, rope_cos,
                rope_sin, window_slots, num_tokens, window_cache, compressed_cache,
                output_window, output_arrived, kv_restore_indices, cp_group_base, cp_rank,
                attention_epoch, kv_hidden, hc_output, hc_next_pre_mix,
            )
    return hc_output, hc_next_pre_mix


prefill_attention_entry = prefill_attention_cp if CP_SIZE > 1 else prefill_attention
