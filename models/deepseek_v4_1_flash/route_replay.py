# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Read and summarize V4.1 prefill gate traces."""

import argparse
import json

import torch


def read_route_trace(path, counts, shape, experts, topk=6, *, placement_tokens=None):
    capture = torch.load(path, map_location="cpu", weights_only=True)
    trace = capture["routes"]
    if trace.shape != shape or trace.dtype != torch.float32:
        raise ValueError("route trace shape or dtype differs from the compiled replay shape")
    if not torch.equal(counts, capture["counts"]):
        raise ValueError("route trace rank counts differ from the replay workload")
    layers_per_stage = shape[1] // counts.shape[1]
    for rank, stage_counts in enumerate(counts.tolist()):
        for stage, active in enumerate(stage_counts):
            first_layer = stage * layers_per_stage
            entries = trace[rank, first_layer:first_layer + layers_per_stage, :active]
            ids = entries[..., :topk]
            weights = entries[..., topk:2 * topk]
            if not torch.isfinite(weights).all() or not ((ids == ids.trunc()) & (ids >= 0) & (ids < experts)).all():
                raise ValueError("route trace contains an invalid expert ID or weight")
            ordered = ids.sort(dim=-1).values
            if (ordered[..., 1:] == ordered[..., :-1]).any():
                raise ValueError("route trace repeats an expert within a token")
    if placement_tokens is not None:
        if capture.get("layout") != "online":
            raise ValueError("placement replay requires a capture from the online layout")
        validate_placement_trace(trace, counts, placement_tokens, experts, topk)
    return trace


def validate_placement_trace(trace, counts, tokens, experts, topk):
    world, layers = trace.shape[:2]
    per_rank = experts // world
    for layer in range(layers):
        stage = layer * counts.shape[1] // layers
        previous = None
        active_by_rank = counts[:, stage].tolist()
        for start in range(0, max(active_by_rank), tokens):
            moves = trace[:, layer, start, 2 * topk:2 * topk + 4]
            if not (moves == moves[0]).all() or not (moves == moves.trunc()).all():
                raise ValueError("placement trace must contain the same integral decision on each rank")
            expert, destination, moved, copy = (int(value) for value in moves[0])
            if moved == 0 and (expert, destination, copy) == (-1, -1, 0):
                continue
            if not (0 <= expert < experts and 0 <= destination < world and destination != expert // per_rank
                    and moved > 0 and copy in (0, 1)):
                raise ValueError("placement trace contains an invalid expert, destination or copy flag")
            routed = sum(
                (trace[rank, layer, start:min(active, start + tokens), :topk] == expert).sum().item()
                for rank, active in enumerate(active_by_rank)
            )
            if moved > routed or copy != int(previous != (expert, destination)):
                raise ValueError("placement trace moves nonexistent routes or reuses a different replica")
            previous = (expert, destination)


def validate_planner_result(trace, counts, tokens, topk=6):
    for layer in range(trace.shape[1]):
        stage = layer * counts.shape[1] // trace.shape[1]
        rows = int(counts[:, stage].max())
        if (trace[:, layer, :rows:tokens, 2 * topk + 3] < 0).any():
            raise ValueError(f"online planner disagreed with the captured placement at layer {layer}")


def summarize_routes(trace, counts, tokens, experts, *, placements=False):
    """Yield one hardware-independent home-load sample per layer and valid round."""
    world, layers = trace.shape[:2]
    per_rank = experts // world
    for layer in range(layers):
        stage = layer * counts.shape[1] // layers
        active_rows = counts[:, stage].tolist()
        for start in range(0, max(active_rows), tokens):
            histogram = torch.zeros(experts, dtype=torch.int64)
            sources = []
            for rank, active in enumerate(active_rows):
                active_in_round = max(0, min(tokens, active - start))
                sources.append(active_in_round)
                indices = trace[rank, layer, start:start + active_in_round, :6].reshape(-1)
                histogram += torch.bincount(indices.to(torch.int64), minlength=experts)
            owners = histogram.reshape(world, per_rank)
            sample = {
                "layer": layer, "round": start // tokens, "source_tokens": sources,
                "home_rows": owners.sum(dim=1).tolist(),
                "home_active_experts": (owners > 0).sum(dim=1).tolist(),
                "home_tiles_64": ((owners + 63) // 64).sum(dim=1).tolist(),
            }
            if placements:
                expert, destination, moved, copy = (int(value) for value in trace[0, layer, start, 12:16])
                sample["replica_expert"] = expert
                sample["replica_rank"] = destination
                sample["replica_rows"] = moved
                sample["replica_copies"] = copy
            yield sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--tokens", type=int, default=80)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    trace, counts = capture["routes"], capture["counts"]
    placements = capture.get("layout") == "online"
    for sample in summarize_routes(trace, counts, args.tokens, 384, placements=placements):
        print(json.dumps(sample))
