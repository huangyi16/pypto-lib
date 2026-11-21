# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
"""Direct-to-expert prefill transport on PyPTO windows."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP_SIZE, FLASH, N_LOCAL_EXPERTS, PREFILL_MOE_TOKENS as MOE_TOKENS
from expert_routed import _expert_routed_output, routed_worker_counts

D = FLASH.hidden_size
INTER = FLASH.moe_intermediate_size
TOPK = FLASH.num_experts_per_tok
EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
HIST_TILE = (N_LOCAL_EXPERTS + 7) // 8 * 8
PUSH_WORKERS = min(EXPERTS, 56)
SCALES = D // 32
ALIGN = 32
ROUTE_WORDS = 8
ROWS_PER_ROUND = EP_SIZE * MOE_TOKENS * TOPK
RECV_MAX = EP_SIZE * MOE_TOKENS
COMPACT_ROWS = ROWS_PER_ROUND + N_LOCAL_EXPERTS * (ALIGN - 1)
REPLICA_COMPACT_ROWS = COMPACT_ROWS + ALIGN - 1
WEIGHT_COLS = 128
WEIGHT_ROWS = D * INTER // (2 * WEIGHT_COLS)
S13_ROWS = D * INTER // (32 * WEIGHT_COLS)
REPLICA_ROWS = 3 * (WEIGHT_ROWS + S13_ROWS)
CUBE_ROWS = 64
COPY_TILE_ROWS = 16
SCALE_TMP = 2 * ((SCALES + 63) // 32) * 32
ROUTED_CUBE_WORKER_TILE, ROUTED_VECTOR_WORKER_TILE = routed_worker_counts(MOE_TOKENS)


@pl.jit.inline(auto_scope=False)
def push_replica_projection(
    weight: pl.Tensor, raw_scale: pl.Tensor,
    replica_bank: pld.DistributedTensor,
    home_slot: pl.Scalar[pl.INDEX], peer: pl.Scalar[pl.INT32], worker: pl.Scalar[pl.INDEX],
    projection: pl.constexpr, k: pl.constexpr, n: pl.constexpr,
):
    source_begin = home_slot * WEIGHT_ROWS
    payload_begin = projection * WEIGHT_ROWS
    for block in pl.range(worker, WEIGHT_ROWS // COPY_TILE_ROWS, 32):
        row = block * COPY_TILE_ROWS
        pld.tensor.put(
            dst=replica_bank, peer=peer, src=weight,
            dst_offsets=[payload_begin + row, 0], src_offsets=[source_begin + row, 0],
            shape=[COPY_TILE_ROWS, WEIGHT_COLS],
        )
    scale_begin = 3 * WEIGHT_ROWS + projection * S13_ROWS
    panel_rows = k // (2 * WEIGHT_COLS)
    for panel in pl.range(worker, n // 16, 32):
        source = (panel * N_LOCAL_EXPERTS + home_slot) * panel_rows
        target = scale_begin + panel * panel_rows
        for row in pl.range(panel_rows):
            scale_tile = pl.load(raw_scale, [source + row, 0], [1, WEIGHT_COLS])
            scale_bytes = pl.reinterpret_view(scale_tile, pl.UINT8)
            pld.tile.remote_store(scale_bytes, replica_bank, peer=peer, offsets=[target + row, 0])


@pl.jit.inline(auto_scope=False)
def copy_replica(
    move: pl.Tensor, bank: pld.DistributedTensor,
    w1: pl.Tensor, s1: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor,
    data_arrived: pld.DistributedTensor,
    plan_ready: pl.Scalar[pl.TASK_ID], rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
):
    payload_w1 = pl.reshape(w1, [N_LOCAL_EXPERTS * WEIGHT_ROWS, WEIGHT_COLS])
    payload_w3 = pl.reshape(w3, [N_LOCAL_EXPERTS * WEIGHT_ROWS, WEIGHT_COLS])
    payload_w2 = pl.reshape(w2, [N_LOCAL_EXPERTS * WEIGHT_ROWS, WEIGHT_COLS])
    with pl.spmd(32, name_hint="moonep_replica_copy", deps=[plan_ready]) as copied:
        expert = pl.read(move, [0])
        destination = pl.read(move, [1])
        needs_copy = pl.read(move, [3])
        if expert // N_LOCAL_EXPERTS == rank and needs_copy != 0:
            worker = pl.tile.get_block_idx()
            home_slot = pl.cast(expert % N_LOCAL_EXPERTS, pl.INDEX)
            push_replica_projection(payload_w1, s1, bank, home_slot, destination, worker, 0, D, INTER)
            push_replica_projection(payload_w3, s3, bank, home_slot, destination, worker, 1, D, INTER)
            push_replica_projection(payload_w2, s2, bank, home_slot, destination, worker, 2, INTER, D)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_replica_notify", deps=[copied]) as signaled:
        expert_to_send = pl.read(move, [0])
        receiver = pl.read(move, [1])
        if expert_to_send // N_LOCAL_EXPERTS == rank and pl.read(move, [3]) != 0:
            pld.system.notify(target=data_arrived, peer=receiver, offsets=[rank, 1, 0], value=epoch, op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_replica_wait", deps=[signaled]) as ready:
        if pl.read(move, [1]) == rank and pl.read(move, [3]) != 0:
            home = pl.read(move, [0]) // N_LOCAL_EXPERTS
            pld.system.defer_wait(signal=data_arrived, offsets=[home, 1, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    return ready


@pl.jit.inline
def place_replica(
    histogram: pld.DistributedTensor, cached: pl.Tensor, move: pl.Tensor, copy_cost_tiles: pl.constexpr,
):
    totals = pl.array.create(EXPERTS, pl.INT32)
    loads = pl.array.create(EP_SIZE, pl.INT32)
    for owner in pl.range(EP_SIZE):
        load = pl.cast(0, pl.INT32)
        for local_e in pl.range(N_LOCAL_EXPERTS):
            candidate = owner * N_LOCAL_EXPERTS + local_e
            rows = pl.cast(0, pl.INT32)
            for source in pl.range(EP_SIZE):
                rows = rows + pl.read(histogram, [source, candidate])
            totals[candidate] = rows
            tiles = pl.cast((rows + CUBE_ROWS - 1) // CUBE_ROWS, pl.INT32)
            load = load + tiles
        loads[owner] = load
    home = pl.cast(0, pl.INDEX)
    for owner in pl.range(1, EP_SIZE):
        if loads[owner] > loads[home]:
            home = owner
    baseline = loads[home]
    best = baseline
    expert = pl.cast(-1, pl.INT32)
    destination = pl.cast(-1, pl.INT32)
    shifted = pl.cast(0, pl.INT32)
    copy_required = pl.cast(0, pl.INT32)
    previous_expert = pl.read(cached, [0])
    previous_rank = pl.read(cached, [1])
    for local_e in pl.range(N_LOCAL_EXPERTS):
        candidate = home * N_LOCAL_EXPERTS + local_e
        rows = totals[candidate]
        if rows > 0:
            tiles = pl.cast((rows + CUBE_ROWS - 1) // CUBE_ROWS, pl.INT32)
            for peer in pl.range(EP_SIZE):
                if peer != home:
                    penalty = pl.cast(copy_cost_tiles, pl.INT32)
                    if candidate == previous_expert and peer == previous_rank:
                        penalty = pl.cast(0, pl.INT32)
                    other_finish = pl.cast(0, pl.INT32)
                    for other in pl.range(EP_SIZE):
                        if other != home and other != peer:
                            other_finish = pl.max(other_finish, loads[other])
                    for moving_tiles in pl.range(1, tiles + 1):
                        moved = pl.cast(pl.min(rows, moving_tiles * CUBE_ROWS), pl.INT32)
                        left = pl.cast((rows - moved + CUBE_ROWS - 1) // CUBE_ROWS, pl.INT32)
                        current_finish = pl.max(baseline - tiles + left, loads[peer] + moving_tiles + penalty)
                        finish = pl.cast(pl.max(current_finish, other_finish), pl.INT32)
                        if finish < best:
                            best = finish
                            expert = pl.cast(candidate, pl.INT32)
                            destination = pl.cast(peer, pl.INT32)
                            shifted = moved
                            copy_required = pl.cast(1, pl.INT32)
                            if candidate == previous_expert and peer == previous_rank:
                                copy_required = pl.cast(0, pl.INT32)
    pl.write(move, [0], expert)
    pl.write(move, [1], destination)
    pl.write(move, [2], shifted)
    pl.write(move, [3], copy_required)
    if shifted > 0:
        pl.write(cached, [0], expert)
        pl.write(cached, [1], destination)


@pl.jit.inline(auto_scope=False)
def dispatch_direct(
    indices: pl.Tensor, x: pl.Tensor, scale_rows: pl.Tensor, weights: pl.Tensor,
    histogram: pld.DistributedTensor, recv_x: pld.DistributedTensor,
    recv_scale: pld.DistributedTensor, recv_weights: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor, arrived: pld.DistributedTensor, data_arrived: pld.DistributedTensor,
    counts: pl.Tensor, starts: pl.Tensor, packed_scale: pl.Tensor, packed_weights: pl.Tensor,
    replica_bank: pld.DistributedTensor, replica_cached: pl.Tensor, route_trace: pl.Tensor,
    trace_start: pl.Scalar[pl.INDEX],
    w1: pl.Tensor, s1: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor,
    scale_ready: pl.Scalar[pl.TASK_ID], active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], layout: pl.constexpr, copy_cost_tiles: pl.constexpr, trace_mode: pl.constexpr,
):
    route_positions = pl.create_tensor([EXPERTS, MOE_TOKENS], dtype=pl.INT32, manual_dep=True)
    send_counts = pl.create_tensor([EXPERTS], dtype=pl.INT32, manual_dep=True)
    send_starts = pl.create_tensor([EXPERTS], dtype=pl.INT32, manual_dep=True)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_histogram") as published:
        cursor = pl.array.create(EXPERTS, pl.INT32)
        for expert in pl.range(EXPERTS):
            cursor[expert] = 0
        route_ids = pl.load(indices, [0, 0], [MOE_TOKENS, ROUTE_WORDS], valid_shape=[MOE_TOKENS, TOPK])
        for token in pl.range(active):
            for k in pl.range(TOPK):
                routed_expert = pl.tile.read(route_ids, [token, k])
                entry = cursor[routed_expert]
                route = pl.cast(token * TOPK + k, pl.INT32)
                pl.write(route_positions, [routed_expert, entry], route)
                cursor[routed_expert] = entry + 1
        for expert in pl.range(EXPERTS):
            pl.write(send_counts, [expert], cursor[expert])
        for group in pl.range(EP_SIZE):
            group_begin = group * N_LOCAL_EXPERTS
            histogram_tile = pl.tile.full([1, HIST_TILE], dtype=pl.INT32, value=0)
            for slot in pl.range(N_LOCAL_EXPERTS):
                pl.tile.write(histogram_tile, [0, slot], cursor[group_begin + slot])
            histogram_rows = pl.tile.set_validshape(histogram_tile, 1, N_LOCAL_EXPERTS)
            for peer in pl.range(EP_SIZE):
                pld.tile.remote_store(histogram_rows, target=histogram, peer=peer, offsets=[rank, group_begin])
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.notify(target=arrived, peer=peer, offsets=[rank, 0], value=epoch, op=pld.NotifyOp.Set)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_histogram_wait", deps=[published]) as hist_ready:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.defer_wait(signal=arrived, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    move = pl.create_tensor([4], dtype=pl.INT32)
    candidate = pl.create_tensor([4], dtype=pl.INT32)
    replica_rows = pl.create_tensor([3], dtype=pl.INT32)
    preloaded = hist_ready
    if layout >= 4:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_read_placement", deps=[published]) as known:
            for field in pl.range(4):
                saved = pl.read(route_trace, [trace_start, 2 * TOPK + field])
                pl.write(move, [field], pl.cast(saved, pl.INT32))
        preloaded = copy_replica(move, replica_bank, w1, s1, w3, s3, w2, s2, data_arrived, known, rank, epoch)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_placement", deps=[hist_ready, preloaded]) as placement:
        if layout == 3:
            place_replica(histogram, replica_cached, move, copy_cost_tiles)
            if trace_mode == 1:
                for field in pl.range(4):
                    value = pl.read(move, [field])
                    pl.write(route_trace, [trace_start, 2 * TOPK + field], pl.cast(value, pl.FP32))
        elif layout == 5:
            place_replica(histogram, replica_cached, candidate, copy_cost_tiles)
            matches = pl.cast(1, pl.INT32)
            for field in pl.range(4):
                if pl.read(candidate, [field]) != pl.read(move, [field]):
                    matches = pl.cast(0, pl.INT32)
            if matches == 0:
                pl.write(route_trace, [trace_start, 2 * TOPK + 3], -1.0)
        elif layout < 3:
            pl.write(move, [0], pl.cast(-1, pl.INT32))
            pl.write(move, [1], pl.cast(-1, pl.INT32))
            pl.write(move, [2], pl.cast(0, pl.INT32))
            pl.write(move, [3], pl.cast(0, pl.INT32))

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_row_plan", deps=[placement]) as plan_ready:
        moved_expert = pl.read(move, [0])
        moved_rank = pl.read(move, [1])
        moved_count = pl.read(move, [2])
        keep = pl.cast(0, pl.INT32)
        if moved_count > 0:
            full = pl.cast(0, pl.INT32)
            for source in pl.range(EP_SIZE):
                full = full + pl.read(histogram, [source, moved_expert])
            keep = full - moved_count
        moved_base = pl.cast(0, pl.INT32)

        for destination in pl.range(EP_SIZE):
            segment = pl.cast(0, pl.INT32)
            for slot in pl.range(N_LOCAL_EXPERTS):
                expert = destination * N_LOCAL_EXPERTS + slot
                if layout == 1:
                    segment = pl.cast(slot * RECV_MAX, pl.INT32)
                total = pl.cast(0, pl.INT32)
                for source in pl.range(EP_SIZE):
                    if source == rank:
                        pl.write(send_starts, [expert], segment + total)
                    total = total + pl.read(histogram, [source, expert])
                if expert == moved_expert:
                    total = keep
                if destination == rank:
                    pl.write(counts, [slot, 0], total)
                    pl.write(starts, [slot], segment)
                segment = pl.cast(segment + (total + ALIGN - 1) // ALIGN * ALIGN, pl.INT32)
            if destination == moved_rank:
                moved_base = segment
        if layout >= 3:
            remaining = pl.cast(0, pl.INT32)
            if rank == moved_rank:
                remaining = moved_count
            pl.write(starts, [N_LOCAL_EXPERTS], moved_base)
            pl.write(counts, [N_LOCAL_EXPERTS, 0], remaining)
        own_home = pl.cast(0, pl.INT32)
        send_move_start = moved_base
        if moved_count > 0:
            before = pl.cast(0, pl.INT32)
            for previous in pl.range(rank):
                before = before + pl.read(histogram, [previous, moved_expert])
            own_count = pl.read(histogram, [rank, moved_expert])
            own_home = pl.cast(pl.min(own_count, pl.max(keep - before, 0)), pl.INT32)
            send_move_start = pl.cast(moved_base + pl.max(before - keep, 0), pl.INT32)
        pl.write(replica_rows, [0], own_home)
        pl.write(replica_rows, [1], send_move_start)
        pl.write(replica_rows, [2], moved_rank)

    replica_ready = plan_ready
    if layout == 3:
        replica_ready = copy_replica(move, replica_bank, w1, s1, w3, s3, w2, s2, data_arrived, plan_ready, rank, epoch)

    with pl.spmd(PUSH_WORKERS, name_hint="moonep_push", deps=[plan_ready, scale_ready, published]) as pushed:
        worker = pl.tile.get_block_idx()
        for expert in pl.range(worker, EXPERTS, PUSH_WORKERS):
            count = pl.cast(pl.read(send_counts, [expert]), pl.INDEX)
            if count > 0:
                destination = expert // N_LOCAL_EXPERTS
                base = pl.cast(pl.read(send_starts, [expert]), pl.INDEX)
                positions = pl.load(route_positions, [expert, 0], [1, MOE_TOKENS])
                route_weights = pl.load(weights, [0, 0], [MOE_TOKENS, ROUTE_WORDS], valid_shape=[MOE_TOKENS, TOPK])
                weight_row = pl.tile.full([1, ROUTE_WORDS], dtype=pl.FP32, value=0)
                route_row = pl.tile.full([1, ROUTE_WORDS], dtype=pl.INT32, value=0)
                pl.tile.write(route_row, [0, 1], rank)
                selected = pl.read(move, [0])
                home_count = pl.cast(pl.read(replica_rows, [0]), pl.INDEX)
                replica_base = pl.cast(pl.read(replica_rows, [1]), pl.INDEX)
                replica_rank = pl.cast(pl.read(replica_rows, [2]), pl.INDEX)
                for offset in pl.range(count):
                    route = pl.tile.read(positions, [0, offset])
                    token = pl.cast(route // TOPK, pl.INDEX)
                    k = pl.cast(route % TOPK, pl.INDEX)
                    row = base + offset
                    peer = destination
                    if expert == selected and offset >= home_count:
                        row = replica_base + offset - home_count
                        peer = replica_rank
                    payload = pl.load(x, [token, 0], [1, D])
                    pld.tile.remote_store(pl.reinterpret_view(payload, pl.INT8), recv_x, peer=peer, offsets=[row, 0])
                    pld.tensor.put(
                        dst=recv_scale, peer=peer, src=scale_rows,
                        dst_offsets=[row, 0], src_offsets=[token, 0], shape=[1, SCALES],
                    )
                    pl.tile.write(weight_row, [0, 0], pl.tile.read(route_weights, [token, k]))
                    pl.tile.write(route_row, [0, 0], route)
                    pld.tile.remote_store(weight_row, recv_weights, peer=peer, offsets=[row, 0])
                    pld.tile.remote_store(route_row, recv_routes, peer=peer, offsets=[row, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_publish", deps=[pushed]) as sent:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.notify(
                    target=data_arrived, peer=peer, offsets=[rank, 0, 0], value=epoch, op=pld.NotifyOp.Set,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_wait", deps=[sent]) as received:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.defer_wait(
                    signal=data_arrived, offsets=[peer, 0, 0], expected=epoch, cmp=pld.WaitCmp.Ge,
                )

    execution_slots = N_LOCAL_EXPERTS
    if layout >= 3:
        execution_slots = N_LOCAL_EXPERTS + 1
    with pl.spmd(32, name_hint="moonep_pack_scale", deps=[received, plan_ready]) as packed:
        worker = pl.tile.get_block_idx()
        for slot in pl.range(worker, execution_slots, 32):
            count = pl.cast(pl.read(counts, [slot, 0]), pl.INDEX)
            base = pl.cast(pl.read(starts, [slot]), pl.INDEX)
            for offset in pl.range(0, count, ALIGN):
                first = base + offset
                valid = pl.min(ALIGN, count - offset)
                raw = pl.load(recv_scale, [first, 0], [ALIGN, SCALES], valid_shape=[valid, SCALES])
                scale_tile = pl.tile.full([ALIGN, SCALES], dtype=pl.UINT8, value=127)
                scale_tile = pl.tile.assemble(scale_tile, raw, [0, 0])
                tmp = pl.create_tile([1, SCALE_TMP], dtype=pl.UINT8)
                mx = pl.tmov_x2zz(scale_tile, tmp, group_axis=1, dst_rows=ALIGN, dst_cols=SCALES)
                mx = pl.reinterpret_view(mx, pl.FP8E8M0)
                mx_flat = pl.reshape(mx, [1, ALIGN * SCALES])
                packed_scale = pl.store(mx_flat, [0, first * SCALES], packed_scale)
                for index in pl.range(valid):
                    row = first + index
                    route_weight = pl.read(recv_weights, [row, 0])
                    pl.write(packed_weights, [row, 0], route_weight)
    return pl.system.task_dummy(deps=[packed, replica_ready])


@pl.jit.inline(auto_scope=False)
def expert_direct_scatter(
    x: pld.DistributedTensor, scales: pl.Tensor, weights: pl.Tensor, counts: pl.Tensor, starts: pl.Tensor,
    w1: pl.Tensor, w1_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    w3: pl.Tensor, w3_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    w2: pl.Tensor, w2_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    routes: pld.DistributedTensor, returned: pld.DistributedTensor, replica_bank: pld.DistributedTensor,
    input_ready: pl.Scalar[pl.TASK_ID], execution_slots: pl.Scalar[pl.INDEX],
):
    output, computed = _expert_routed_output(
        x, scales, weights, counts, w1, w1_scale, w3, w3_scale, w2, w2_scale, input_ready,
        starts, replica_bank, execution_slots, N_LOCAL_EXPERTS, 0, D, INTER, ROWS_PER_ROUND,
        ROUTED_CUBE_WORKER_TILE, ROUTED_VECTOR_WORKER_TILE,
    )
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="moonep_scatter", deps=[computed]) as scattered:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moonep_scatter", no_dep_args=[returned]):
            worker = pl.tile.get_block_idx()
            compact_base = 0
            for slot in pl.range(execution_slots):
                count = pl.cast(pl.read(counts, [slot, 0]), pl.INDEX)
                first_row = pl.cast(pl.read(starts, [slot]), pl.INDEX)
                first = (worker + ROUTED_VECTOR_WORKER_TILE - compact_base % ROUTED_VECTOR_WORKER_TILE)
                for row in pl.range(first % ROUTED_VECTOR_WORKER_TILE, count, ROUTED_VECTOR_WORKER_TILE):
                    source = pl.read(routes, [first_row + row, 1])
                    route = pl.read(routes, [first_row + row, 0])
                    value = pl.load(output, [compact_base + row, 0], [1, D])
                    pld.tile.remote_store(value, returned, peer=source, offsets=[route, 0])
                compact_base = compact_base + count
    return scattered
