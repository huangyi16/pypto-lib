# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""CED prefill with routed experts supplied from a host-backed slot bank."""
import pypto.language as pl
import pypto.language.distributed as pld
import config as C
from prefill_graph_common import (
    BIAS_DYN, C1A_TABLE_DYN, C2A_SOURCE_COUNT, C2A_TABLE_DYN, CAPACITY, CP_SIZE, D, DSPARK_MAIN_IN,
    FINAL_SLOT, HC_DIM, HC_MULT, HEAD_DIM, INDEX_DIM, INDEX_H, INDEX_POOL_BLOCKS, INDEX_SOURCE_COUNT,
    INDEX_TOPK, KV_SOURCE_COUNT, LOCAL_CAPACITY, LOCAL_H, LOCAL_O_GROUPS, LOCAL_O_WIDTH, MIX_HC,
    MOE_INTER, MX_GROUP, N_LAYERS, O_GROUP_IN, O_LORA, PAGES, Q_LORA, RANKS, RECV_MAX, TOPK, VOCAB,
    WINDOW_PAGES, prefill_attention_entry, restore_window_tail,
)
from config import (
    LM_HEAD_TP_SIZE, B_DYN,
    PREFILL_MAX_TOKENS, Q_START_DYN, STATE_BLOCKS_DYN, MOE_TOKENS,
    DSPARK_LAST_TARGET_LAYER, DSPARK_TARGET_LAYER_IDS, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)

from prefill_cp_allgather import prefill_cp_gather_hc
from hc import hc_head
from input_pack import gather_rows, pack_x_hc
from rope_tables import ROPE_ROWS_DYN, materialize_rope_rows
from engram import ENGRAM_K, KV_OUT, engram_from_rows
from lm_head import MAX_LOGIT_ROWS, clear_lm_head_signals, lm_head
from gate import gather_routing_bias
from sample import sample

from compressor import STATE_WIDTH
from prefill_c1a_full import prefill_publish_decoder
from o_proj import retire_prefill_signals
from rmsnorm import rms_norm
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS

from expert_context import (
    BANK_DIRECTIONS, EXPERT_SLOTS, RankState, RankStates, RankDescriptors, RankDescriptorTables,
    SCALE_BANK_LAYOUT,
)
from offload_prefill_moe import AUX_WIDTH, ROUTE_WIDTH, clear_moe_signals, prefill_moe_slabs

LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
N_EXPERTS = RANKS * LOCAL_EXPERTS
ROUTED_BANK_ROWS = EXPERT_SLOTS
ROUTED_LAYER_ROWS = EXPERT_SLOTS
ROUTED_LAYER_STRIDE = 0


@pl.jit(auto_scope=False, **BANK_DIRECTIONS)
def l2_prefill_fwd(
    embedding: pl.Tensor[[CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    counts: pl.Tensor[[2], pl.INT32],
    replay_rows: pl.Tensor[[CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[2, CAPACITY], pl.INT64],
    window_restore_slots: pl.Tensor[[CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[40, WINDOW_PAGES, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16, pl.NZ],
    logit_rows: pl.Tensor[[16], pl.INT32],
    sampling_temperatures: pl.Tensor[[16], pl.FP32],
    sampling_top_ks: pl.Tensor[[16], pl.INT32],
    sampling_top_ps: pl.Tensor[[16], pl.FP32],
    sampling_seeds: pl.Tensor[[16], pl.INT32],
    sampling_positions: pl.Tensor[[16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[CAPACITY, DSPARK_MAIN_IN], pl.BF16]],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    recv_meta: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D // 32], pl.UINT8],
    recv_weights: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    data_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    routed_output: pld.DistributedTensor[[MOE_TOKENS * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hc_window: pld.DistributedTensor[[LOCAL_CAPACITY, HC_DIM + 16], pl.FP32],
    hc_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[LM_HEAD_TP_SIZE * 16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.LM_HEAD_TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.BF16],
    head_logits_done: pld.DistributedTensor[[C.LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    expert_state: pl.InOut[RankState] = 0,
    expert_descriptors: RankDescriptors = 0,
):
    """Run both CED stages and retire communication signals after the decoder."""
    correction_bias.bind_dynamic(0, BIAS_DYN)
    compressed_cache.bind_dynamic(1, PAGES)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(1, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(0, B_DYN)
    c1a_index_block_table.bind_dynamic(1, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(1, ROPE_ROWS_DYN)
    rope_cos = pl.create_tensor([3, CAPACITY, 32], dtype=pl.FP32)
    rope_sin = pl.create_tensor([3, CAPACITY, 32], dtype=pl.FP32)
    compressed_cos = pl.create_tensor([CAPACITY, 32], dtype=pl.FP32)
    compressed_sin = pl.create_tensor([CAPACITY, 32], dtype=pl.FP32)
    encoder_count = pl.read(counts, [0])
    decoder_count = pl.read(counts, [1])
    materialize_rope_rows(
        freqs_cos[0], freqs_sin[0], position_ids[0], encoder_count, rope_cos[0], rope_sin[0], False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[0], encoder_count, rope_cos[1], rope_sin[1], False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[1], decoder_count, rope_cos[2], rope_sin[2], False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[0], encoder_count, compressed_cos, compressed_sin, True,
        worker_limit=32,
    )
    q_rows = pl.create_tensor([2, LOCAL_CAPACITY], dtype=pl.INT32)
    restore_indices = pl.create_tensor([2, CAPACITY], dtype=pl.INT32)
    q_counts = pl.create_tensor([2], dtype=pl.INT32)
    q_rope_cos = pl.create_tensor([3, LOCAL_CAPACITY, 32], dtype=pl.FP32)
    q_rope_sin = pl.create_tensor([3, LOCAL_CAPACITY, 32], dtype=pl.FP32)
    q_window_indices = pl.create_tensor([2, LOCAL_CAPACITY, 128], dtype=pl.INT32)
    q_request_ids = pl.create_tensor([2, LOCAL_CAPACITY], dtype=pl.INT32)
    q_compressed_lens = pl.create_tensor([2, LOCAL_CAPACITY], dtype=pl.INT32)
    for stage_index in pl.range(2):
        count = pl.read(counts, [stage_index])
        chunk = (count + 2 * CP_SIZE - 1) // (2 * CP_SIZE)
        cp_rank = rank % CP_SIZE
        head_start = cp_rank * chunk
        tail_start = (2 * CP_SIZE - 1 - cp_rank) * chunk
        head_count = pl.max(0, pl.min(chunk, count - head_start))
        tail_count = pl.max(0, pl.min(chunk, count - tail_start))
        with pl.spmd(1, name_hint="prefill_cp_rows") as rows_ready:
            writer = pl.tile.get_block_idx()
            pl.write(q_counts, [stage_index], pl.cast(head_count + tail_count, pl.INT32))
            for row in pl.range(writer, LOCAL_CAPACITY):
                source = pl.cast(-1, pl.INT32)
                if row < head_count:
                    source = pl.cast(head_start + row, pl.INT32)
                elif row < head_count + tail_count:
                    source = pl.cast(tail_start + row - head_count, pl.INT32)
                pl.write(q_rows, [stage_index, row], source)
                request = pl.cast(0, pl.INT32)
                length = pl.cast(0, pl.INT32)
                if source >= 0:
                    request = pl.read(token_to_req_indices, [stage_index, source])
                    position = pl.read(position_ids, [stage_index, source])
                    length = pl.cast(position + 1, pl.INT32)
                    if stage_index == 0:
                        length = pl.cast(length // 2, pl.INT32)
                pl.write(q_request_ids, [stage_index, row], request)
                pl.write(q_compressed_lens, [stage_index, row], length)
            for row in pl.range(CAPACITY):
                restored = pl.cast(-1, pl.INT32)
                if row < count:
                    block = row // chunk
                    owner = block
                    local_offset = row % chunk
                    if block >= CP_SIZE:
                        owner = 2 * CP_SIZE - 1 - block
                        local_offset = chunk + local_offset
                    restored = pl.cast(owner * LOCAL_CAPACITY + local_offset, pl.INT32)
                pl.write(restore_indices, [stage_index, row], restored)
        with pl.spmd(32, name_hint="prefill_cp_windows", deps=[rows_ready]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, LOCAL_CAPACITY, 32):
                source = pl.read(q_rows, [stage_index, row])
                if source >= 0:
                    q_window_indices[stage_index, row:row + 1, :] = window_indices[stage_index, source:source + 1, :]
                else:
                    q_window_indices[stage_index, row:row + 1, :] = pl.full([1, 128], dtype=pl.INT32, value=-1)
        rope_slot = stage_index + 1
        gather_rows(rope_cos[rope_slot], q_rows[stage_index], q_rope_cos[rope_slot], rows_ready, column_tile=32)
        gather_rows(rope_sin[rope_slot], q_rows[stage_index], q_rope_sin[rope_slot], rows_ready, column_tile=32)
        if stage_index == 0:
            gather_rows(rope_cos[0], q_rows[0], q_rope_cos[0], rows_ready, column_tile=32)
            gather_rows(rope_sin[0], q_rows[0], q_rope_sin[0], rows_ready, column_tile=32)
    hidden_pair = pl.create_tensor([4 * CAPACITY, HC_MULT, D], dtype=pl.FP32)
    pre_pair = pl.create_tensor([4 * CAPACITY, HC_MULT], dtype=pl.FP32)
    initial_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
    initial_mix = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
    gather_ready = pl.system.task_dummy(deps=[])
    pack_x_hc(embedding_ids, embedding, initial_hidden, hidden_tile=D)
    gather_rows(initial_pre, identity_rows, initial_mix, gather_ready, column_tile=512)
    topk_indices = pl.create_tensor([LOCAL_CAPACITY, INDEX_TOPK], dtype=pl.INT32)
    history_positions = pl.tensor.dim(compressed_cache, 1) * 128
    pool_blocks = pl.min(INDEX_POOL_BLOCKS, (history_positions // 8 + 31) // 32 * 32)
    pool_block_ids = pl.create_tensor([LOCAL_CAPACITY, pool_blocks], dtype=pl.INT32)
    pool_block_counts = pl.create_tensor([LOCAL_CAPACITY], dtype=pl.INT32)
    # Reuse bounded MoE slabs across all layers and rounds.
    attn_hidden = pl.create_tensor([LOCAL_CAPACITY, HC_MULT, D], dtype=pl.FP32)
    attn_pre = pl.create_tensor([LOCAL_CAPACITY, HC_MULT], dtype=pl.FP32)
    ffn_mixed = pl.create_tensor([LOCAL_CAPACITY, D], dtype=pl.BF16)
    slab_hidden = pl.create_tensor([2 * MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([2 * MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    slab_rows = pl.create_tensor([MOE_TOKENS], dtype=pl.INT32)
    local_hidden = pl.create_tensor([LOCAL_CAPACITY, HC_MULT, D], dtype=pl.FP32)
    local_pre = pl.create_tensor([LOCAL_CAPACITY, HC_MULT], dtype=pl.FP32)
    local_next_hidden = pl.create_tensor([LOCAL_CAPACITY, HC_MULT, D], dtype=pl.FP32)
    local_next_pre = pl.create_tensor([LOCAL_CAPACITY, HC_MULT], dtype=pl.FP32)
    moe_ready = pl.system.task_dummy(deps=[])
    history_pages = pl.tensor.dim(compressed_cache, 1)
    c1a_cache_units = pl.slice(compressed_cache, [2, history_pages, 64, 1, COMPRESSED_CACHE_BYTES], [3, 0, 0, 0, 0])
    c1a_cache = pl.reshape(c1a_cache_units, [history_pages, 128, 1, COMPRESSED_CACHE_BYTES])
    c1a_index_units = index_cache[3:5]
    c1a_index = pl.reshape(c1a_index_units, [pl.tensor.dim(index_cache, 1), 128, 1, 64])
    c1a_scale_units = index_scales[3:5]
    c1a_scale = pl.reshape(c1a_scale_units, [pl.tensor.dim(index_scales, 1), 128, 1, 4])
    for layer in pl.range(N_LAYERS):
        stage = layer // 20
        active = pl.read(counts, [stage])
        current_slot = (2 * stage + layer % 2) * CAPACITY
        next_slot = (2 * stage + (layer + 1) % 2) * CAPACITY
        hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [current_slot, 0, 0])
        delayed_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [current_slot, 0])
        next_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [next_slot, 0, 0])
        next_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [next_slot, 0])
        kv_source = pl.min(pl.max((layer - 2) // 6, 0), 2)
        if layer >= 20:
            kv_source = 3
        index_source = kv_source
        if layer >= 24:
            index_source = (layer - 24) // 4 + 4
        c2a_source = pl.min(kv_source, 2)
        layer_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
        layer_attn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer * 3])
        layer_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer * MIX_HC])
        layer_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
        layer_ffn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer * 3])
        layer_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer * MIX_HC])
        layer_attn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [layer * D])
        layer_ffn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_weight, [D], [layer * D])
        layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(q_norm_weight, [Q_LORA], [layer * Q_LORA])
        layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(kv_norm_weight, [HEAD_DIM], [layer * HEAD_DIM])
        layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[layer]
        layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [layer * (D // MX_GROUP), 0])
        layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[layer]
        layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [layer * (Q_LORA // MX_GROUP), 0])
        layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[layer]
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [layer * (D // MX_GROUP), 0])
        layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[layer]
        layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [layer * (LOCAL_O_WIDTH // MX_GROUP), 0])
        layer_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[layer]
        layer_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
        layer_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[layer]
        layer_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [layer * (MOE_INTER // MX_GROUP), 0])
        layer_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[layer]
        layer_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
        layer_attn_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [layer * LOCAL_H])
        layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
        layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = pl.slice(wo_a_scale, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
        layer_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wkv[c2a_source]
        layer_compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wgate[c2a_source]
        layer_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [kv_source * HEAD_DIM])
        layer_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[index_source]
        layer_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [index_source * INDEX_DIM])
        layer_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = index_wq_b[index_source]
        layer_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [index_source * (Q_LORA // MX_GROUP), 0])
        layer_index_weights: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[index_source]
        layer_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16] = pl.slice(gate_weight, [N_EXPERTS, D], [layer * N_EXPERTS, 0])
        bias_rows = pl.tensor.dim(correction_bias, 0) // (N_LAYERS * N_EXPERTS)
        bias_length = bias_rows * N_EXPERTS
        source_bias = pl.slice(correction_bias, [bias_length], [layer * bias_length])
        stage_gate_bias = pl.create_tensor([bias_length], dtype=pl.FP32)
        if stage == 0:
            gather_routing_bias(source_bias, identity_rows, stage_gate_bias, moe_ready, n_experts=N_EXPERTS)
        else:
            gather_routing_bias(source_bias, replay_rows, stage_gate_bias, moe_ready, n_experts=N_EXPERTS)
        local_bias_rows = pl.min(bias_rows, LOCAL_CAPACITY)
        layer_gate_bias = pl.create_tensor([local_bias_rows * N_EXPERTS], dtype=pl.FP32)
        gather_routing_bias(stage_gate_bias, q_rows[stage], layer_gate_bias, moe_ready, n_experts=N_EXPERTS)
        layer_routed_w1: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w1, [ROUTED_LAYER_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [layer * ROUTED_LAYER_STRIDE, 0, 0])
        layer_routed_w1_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w1_scale, [ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], [layer * (ROUTED_LAYER_STRIDE * (D // MX_GROUP)), 0])
        layer_routed_w2: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w2, [ROUTED_LAYER_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [layer * ROUTED_LAYER_STRIDE, 0, 0])
        layer_routed_w2_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w2_scale, [ROUTED_LAYER_ROWS * (MOE_INTER // MX_GROUP), D], [layer * (ROUTED_LAYER_STRIDE * (MOE_INTER // MX_GROUP)), 0])
        layer_routed_w3: pl.Tensor[[ROUTED_LAYER_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w3, [ROUTED_LAYER_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [layer * ROUTED_LAYER_STRIDE, 0, 0])
        layer_routed_w3_scale: pl.Tensor[[ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, SCALE_BANK_LAYOUT] = pl.slice(routed_w3_scale, [ROUTED_LAYER_ROWS * (D // MX_GROUP), MOE_INTER], [layer * (ROUTED_LAYER_STRIDE * (D // MX_GROUP)), 0])
        if layer == 20:
            with pl.scope():
                # Publish C1A source KV for ALL encoder rows before selecting replay.
                encoder_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
                encoder_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
                encoder_collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                encoder_normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                hc_head(encoder_hidden, encoder_pre, encoder_collapsed, worker_limit=64)
                norm_tokens = pl.tensor.dim(encoder_collapsed, 0)
                rms_norm(encoder_collapsed, layer_attn_norm, encoder_normalized, norm_tokens, width=D, zero_tail=False)
                publisher_ready = prefill_publish_decoder(
                    encoder_normalized, c1a_compressor_wkv, layer_compressor_norm, layer_index_wk, layer_index_norm,
                    rope_cos[1], rope_sin[1], c1a_compressed_slots,
                    c1a_cache, c1a_index, c1a_scale,
                    pl.read(counts, [0]),
                )
                gathered_hidden = pl.reshape(hidden, [CAPACITY, HC_DIM])
                source_hidden = pl.reshape(encoder_hidden, [CAPACITY, HC_DIM])
                gather_rows(source_hidden, replay_rows, gathered_hidden, publisher_ready, column_tile=D)
                gather_rows(encoder_pre, replay_rows, delayed_pre, publisher_ready, column_tile=512)
        if layer == 1 or layer == 14:
            with pl.scope():
                engram_slot = layer // 14
                layer_engram_scale = pl.slice(engram_wkv_scale, [ENGRAM_K // 32, KV_OUT], [engram_slot * (ENGRAM_K // 32), 0])
                engram_from_rows(
                    engram_lookup[engram_slot], engram_wkv[engram_slot], layer_engram_scale,
                    engram_weight[engram_slot], hidden, hidden,
                    mask_zero_rows=True,
                )
        with pl.scope():
            normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            global_hc = pl.reshape(hidden, [CAPACITY, HC_DIM])
            local_hc = pl.reshape(local_hidden, [LOCAL_CAPACITY, HC_DIM])
            gather_rows(global_hc, q_rows[stage], local_hc, moe_ready, column_tile=D)
            gather_rows(delayed_pre, q_rows[stage], local_pre, moe_ready, column_tile=16)
            local_active = pl.read(q_counts, [stage])
            mode = 0
            cache_slot = 0
            rope_slot = 0
            if layer >= 20:
                cache_slot = 3
                rope_slot = 2
                mode = 5
                if layer == 20:
                    mode = 3
                elif layer % 4 == 0:
                    mode = 4
            elif layer >= 2:
                cache_slot = (layer - 2) // 6
                rope_slot = 1
                mode = 2
                if (layer - 2) % 6 == 0:
                    mode = 1
            state_slot = pl.min(cache_slot, 2)
            if layer >= 20:
                prefill_attention_entry(
                    local_hidden, local_pre, layer_attn_fn, layer_attn_scale, layer_attn_base, layer_attn_norm,
                    layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale, layer_kv_norm, layer_attn_sink,
                    layer_wo_a, layer_wo_a_scale, layer_wo_b, layer_wo_b_scale,
                    rope_cos[rope_slot], rope_sin[rope_slot], window_slots[stage], q_window_indices[stage],
                    q_rope_cos[rope_slot], q_rope_sin[rope_slot], q_request_ids[stage], q_compressed_lens[stage],
                    restore_indices[stage], window_cache[layer], c1a_cache, token_to_req_indices[stage],
                    c1a_index, c1a_scale, c2a_index_block_table, c1a_index_block_table,
                    position_ids[stage], compressed_cos, compressed_sin, layer_compressor_wkv, layer_compressor_wgate,
                    query_start_loc[stage], state_block_table, state_cache[state_slot], layer_compressor_norm,
                    c2a_compressed_slots, layer_index_wk, layer_index_norm, layer_index_wq_b, layer_index_wq_b_scale,
                    layer_index_weights, pool_block_ids, pool_block_counts, topk_indices, attention_window, attention_arrived,
                    normalized, attn_hidden, attn_pre, local_active,
                    rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, active, layer + 1, mode,
                    c1a_phase=True,
                )
            else:
                prefill_attention_entry(
                    local_hidden, local_pre, layer_attn_fn, layer_attn_scale, layer_attn_base, layer_attn_norm,
                    layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale, layer_kv_norm, layer_attn_sink,
                    layer_wo_a, layer_wo_a_scale, layer_wo_b, layer_wo_b_scale,
                    rope_cos[rope_slot], rope_sin[rope_slot], window_slots[stage], q_window_indices[stage],
                    q_rope_cos[rope_slot], q_rope_sin[rope_slot], q_request_ids[stage], q_compressed_lens[stage],
                    restore_indices[stage], window_cache[layer], compressed_cache[cache_slot], token_to_req_indices[stage],
                    index_cache[cache_slot], index_scales[cache_slot], c2a_index_block_table, c1a_index_block_table,
                    position_ids[stage], compressed_cos, compressed_sin, layer_compressor_wkv, layer_compressor_wgate,
                    query_start_loc[stage], state_block_table, state_cache[state_slot], layer_compressor_norm,
                    c2a_compressed_slots, layer_index_wk, layer_index_norm, layer_index_wq_b, layer_index_wq_b_scale,
                    layer_index_weights, pool_block_ids, pool_block_counts, topk_indices, attention_window, attention_arrived,
                    normalized, attn_hidden, attn_pre, local_active,
                    rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, active, layer + 1, mode,
                    c1a_phase=False,
                )
        moe_ready = prefill_moe_slabs(
            attn_hidden, attn_pre, layer_ffn_fn, layer_ffn_scale, layer_ffn_base, layer_ffn_norm, layer_gate_weight, layer_gate_bias,
            layer_routed_w1, layer_routed_w1_scale, layer_routed_w2, layer_routed_w2_scale, layer_routed_w3, layer_routed_w3_scale,
            layer_shared_w1, layer_shared_w1_scale, layer_shared_w2, layer_shared_w2_scale,
            layer_shared_w3, layer_shared_w3_scale, local_next_pre, ffn_mixed, local_next_hidden,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
            routed_output, combine_arrived, slab_hidden, slab_pre, slab_mixed, slab_rows,
            moe_ready, pl.read(q_counts, [stage]), rank, layer + 1, expert_state=expert_state, expert_descriptors=expert_descriptors,
            expert_layer=layer, capacity=LOCAL_CAPACITY,
        )
        if CP_SIZE > 1:
            moe_ready = prefill_cp_gather_hc(
                local_next_hidden, local_next_pre, restore_indices[stage], next_hidden, next_pre,
                hc_window, hc_arrived, rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, layer + 1, moe_ready,
                group_size=CP_SIZE,
            )
        else:
            with pl.spmd(32, name_hint="prefill_local_hc", deps=[moe_ready]) as copied:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, CAPACITY, 32):
                    next_hidden[row:row + 1, :, :] = local_next_hidden[row:row + 1, :, :]
            with pl.spmd(1, name_hint="prefill_local_pre", deps=[moe_ready]) as pre_copied:
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, CAPACITY):
                    for lane in pl.range(HC_MULT):
                        value = pl.read(local_next_pre, [row, lane])
                        pl.write(next_pre, [row, lane], value)
            moe_ready = pl.system.task_dummy(deps=[copied, pre_copied])
        # This residual enters the next layer's attention.
        target_layer = layer + 1
        if target_layer >= DSPARK_TARGET_LAYER_IDS[0] and target_layer <= DSPARK_LAST_TARGET_LAYER:
            target_column = (target_layer - DSPARK_TARGET_LAYER_IDS[0]) * D
            for core in pl.spmd(48, name_hint="prefill_dspark_target_hidden", deps=[moe_ready]):
                for token in pl.range(core, CAPACITY, 48):
                    if token < active:
                        hc_row = next_hidden[token : token + 1, 0:HC_MULT, 0:D]
                        hc_mean = pl.mul(pl.col_sum(pl.reshape(hc_row, [HC_MULT, D])), 1.0 / HC_MULT)
                        target_bf16 = pl.cast(hc_mean, pl.BF16, mode="rint")
                    else:
                        target_bf16 = pl.full([1, D], dtype=pl.BF16, value=0.0)
                    dspark_target_hidden[token : token + 1, target_column : target_column + D] = target_bf16
    window_ready = restore_window_tail(window_cache, window_slots[0], window_restore_slots, moe_ready)
    final_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [FINAL_SLOT, 0, 0])
    final_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [FINAL_SLOT, 0])
    source_hidden = pl.reshape(final_hidden, [CAPACITY, HC_DIM])
    output_hidden = pl.reshape(x_hc_out, [CAPACITY, HC_DIM])
    gather_rows(source_hidden, identity_rows, output_hidden, gather_ready, column_tile=D)
    gather_rows(final_pre, identity_rows, pre_mix_out, gather_ready, column_tile=512)
    clear_moe_signals(x_hc_out, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    if CP_SIZE > 1:
        retire_prefill_signals(x_hc_out, attention_arrived, rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, group_size=CP_SIZE)
        retire_prefill_signals(x_hc_out, hc_arrived, rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, group_size=CP_SIZE)
    owner_logit_rows = pl.create_tensor([16], dtype=pl.INT32)
    with pl.spmd(1, name_hint="prefill_logit_owners"):
        writer = pl.tile.get_block_idx()
        for row in pl.range(writer, 16):
            source = pl.read(logit_rows, [row])
            owner_source = pl.cast(-1, pl.INT32)
            if source >= 0 and source < CAPACITY:
                owner_source = source
            pl.write(owner_logit_rows, [row], owner_source)
    head_rows = pl.tensor.dim(final_hidden, 0)
    head_collapsed = pl.create_tensor([head_rows, D], dtype=pl.BF16)
    hc_head(final_hidden, final_pre, head_collapsed, worker_limit=64)
    head_normalized = pl.create_tensor([head_rows, D], dtype=pl.BF16)
    rms_norm(head_collapsed, final_norm_weight, head_normalized, head_rows, width=D, zero_tail=False)
    lm_head(
        head_normalized, lm_head_weight, owner_logit_rows, logits,
        head_hidden, head_hidden_done, head_logits, head_logits_done,
        rank, MAX_LOGIT_ROWS,
    )
    clear_lm_head_signals(logits, head_hidden_done, head_logits_done)
    sample(
        logits, sampling_temperatures, sampling_top_ks, sampling_top_ps,
        sampling_seeds, sampling_positions, sampled_ids,
    )
    return logits, sampled_ids


@pl.jit.host(**BANK_DIRECTIONS)
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[RANKS, 2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[RANKS, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[RANKS, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[RANKS, 2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[RANKS, N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS, N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS, N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS, N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS, N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS, N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS, N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS, N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[RANKS, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS, N_LAYERS * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[RANKS, BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[RANKS, ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[RANKS, ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[RANKS, ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[RANKS, ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[RANKS, ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[RANKS, ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[RANKS, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS, N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS, N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT64],
    window_restore_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[RANKS, 2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[RANKS, 2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[RANKS, B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[RANKS, B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS, B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, WINDOW_PAGES, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_temperatures: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_top_ks: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_top_ps: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_seeds: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_positions: pl.Tensor[[RANKS, 16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[RANKS, CAPACITY, DSPARK_MAIN_IN], pl.BF16]],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
    expert_state: pl.InOut[RankStates] = 0,
    expert_descriptors: RankDescriptorTables = 0,
):
    """Append one chunk with EP = DP * CP and return global state/logits.

    CAPACITY bounds current rows to at most 4096; the dynamic page extent
    independently sizes retained history. For full causal continuation replay
    all current rows:
    counts[:, 0] == counts[:, 1], replay_rows and identity_rows are current-row
    identities, padded with -1. Positions, lengths and physical slots are
    absolute within the request. Both stages' window indices include committed
    history; the compressed families have independent local page tables.

    embedding holds caller-gathered BF16 rows; embedding_ids indexes that
    rank-local slab, not the model vocabulary. initial_pre is the current
    rows' delayed mix (identity for new tokens). position_ids, window_slots,
    window_indices, token_to_req_indices and query_start_loc have stage order
    encoder, decoder. Each stage has packed query_start_loc and absolute
    position_ids; token_to_req_indices indexes request-table rows.
    C2A/C1A index_block_table arguments have independent widths and source-local
    physical page IDs. KV and index keys share their family's mapping. C2A pages
    contain 64 stored rows covering 128 original tokens; C1A and SWA pages
    contain 128 stored rows covering 128 original tokens.
    Visible lengths are derived once as floor((position+1)/ratio), including
    completed current groups: C2A follows encoder rows and C1A decoder replay rows.
    freqs_cos/sin are resident full FP32 half-width tables with profile order
    base, compressed. Their row capacity is independent of current CAPACITY
    and covers all active absolute positions. The graph gathers encoder base,
    encoder compressed, decoder compressed and C2A group-first rows once.
    C1A publication reuses encoder compressed rows before replay selection.
    c2a_compressed_slots and c1a_compressed_slots both follow encoder rows;
    C1A publishes all encoder rows before decoder replay selection. Slots are
    source-local physical rows; -1 disables publication. C2A publishes only at
    completed pair positions. state_block_table maps requests to source-local
    state blocks shared across three independent C2A source slices; -1 is inactive.
    logit_rows indexes the decoder replay slab; -1 disables that output row.

    Preserve all seven InOut cache tensors across chunks and decode. Their
    source/page dimensions can be flattened into decode pool views.
    Residual/pre-mix outputs describe these rows after layer 39. The DSpark
    output packs HC-mean hidden states from layers 37, 38 and 39 per row.
    New tokens
    start from their own embedding and identity pre-mix; historical HC state
    is represented by the committed KV/index/compressor caches.

    Zero-count ranks participate in EP communication without publishing cache
    rows. Wait for runtime completion before advancing lengths, launching a
    dependent call or reusing buffers. Use persistent=True with
    reset_persistent_windows=False; signals are retired at forward completion.
    After request completion, release pages and reset the request slot in all C2A sources before
    assigning them to another request; failed calls require cache restoration
    or reset before retry. Engram consumes host lookup rows, with zero rows
    suppressing its update. correction_bias accepts the text layout
    [RANKS, N_LAYERS * N_EXPERTS] or flattened per-token biases
    [RANKS, N_LAYERS * CAPACITY * N_EXPERTS] in encoder input-row order.
    Decoder replay and slab ownership gather the corresponding token bias.
    """
    correction_bias.bind_dynamic(1, BIAS_DYN)
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
    compressed_cache.bind_dynamic(2, PAGES)
    state_cache.bind_dynamic(2, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(2, Q_START_DYN)
    recv_meta_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
    recv_weights_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    hc_window_buffer = pld.alloc_window_buffer([LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
    hc_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.BF16)
    head_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buffer, [LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buffer, [LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buffer, [LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buffer, [RANKS, 128], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        attention_arrived = pld.window(attention_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        hc_window = pld.window(hc_window_buffer, [LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
        hc_arrived = pld.window(hc_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.BF16)
        head_logits_done = pld.window(head_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_fwd(
            embedding[rank],
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            embedding_ids[rank],
            initial_pre[rank],
            identity_rows[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale[rank],
            c2a_compressor_wkv[rank], c2a_compressor_wgate[rank], c1a_compressor_wkv[rank],
            compressor_norm_weight[rank],
            index_wk[rank], index_norm_weight[rank],
            index_wq_b[rank], index_wq_b_scale[rank], index_weights_proj[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            counts[rank],
            replay_rows[rank],
            position_ids[rank],
            freqs_cos[rank], freqs_sin[rank],
            window_slots[rank],
            window_restore_slots[rank],
            window_indices[rank],
            c2a_compressed_slots[rank],
            c1a_compressed_slots[rank],
            token_to_req_indices[rank],
            query_start_loc[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank],
            state_block_table[rank],
            window_cache[rank],
            compressed_cache[rank],
            index_cache[rank],
            index_scales[rank],
            state_cache[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            logit_rows[rank],
            sampling_temperatures[rank], sampling_top_ks[rank], sampling_top_ps[rank],
            sampling_seeds[rank], sampling_positions[rank],
            x_hc_out[rank], pre_mix_out[rank], dspark_target_hidden[rank],
            logits[rank],
            sampled_ids[rank],
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            attention_window,
            attention_arrived,
            hc_window, hc_arrived,
            head_hidden, head_hidden_done,
            head_logits,
            head_logits_done,
            rank, expert_state=expert_state[rank],
            expert_descriptors=expert_descriptors[rank], device=rank,
        )
