# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Serving and pypto-lib route trace checks."""

import pytest
import torch

from models.deepseek_v4_1_flash.route_replay import read_route_trace, summarize_routes, validate_planner_result


def test_online_route_trace_preserves_empty_sources_sections_and_replica_hits(tmp_path):
    counts = torch.tensor([[0, 0], [3, 1]], dtype=torch.int32)
    routes = torch.zeros(2, 2, 4, 16, dtype=torch.float32)
    routes[1, :, :, :6] = torch.tensor([0, 1, 2, 3, 4, 5])
    routes[1, :, :, 6:12] = 0.5
    routes[:, :, 0, 12:] = torch.tensor([0, 1, 1, 1])
    routes[:, 0, 2, 12:] = torch.tensor([0, 1, 1, 0])
    path = tmp_path / "prefill.pt"

    torch.save({"routes": routes, "counts": counts, "layout": "online"}, path)
    replay = read_route_trace(path, counts, routes.shape, 8, placement_tokens=2)
    torch.testing.assert_close(replay, routes)
    validate_planner_result(replay, counts, 2)
    load = list(summarize_routes(replay, counts, 2, 8))
    assert [sample["source_tokens"] for sample in load] == [[0, 2], [0, 1], [0, 1]]
    assert [sample["home_rows"] for sample in load] == [[8, 4], [4, 2], [4, 2]]

    routes[1, 0, 2, 15] = -1
    with pytest.raises(ValueError, match="disagreed"):
        validate_planner_result(routes, counts, 2)

    routes[:, 0, 2, 12:] = torch.tensor([1, 1, 1, 0])
    torch.save({"routes": routes, "counts": counts, "layout": "online"}, path)
    with pytest.raises(ValueError, match="reuses a different replica"):
        read_route_trace(path, counts, routes.shape, 8, placement_tokens=2)


def test_route_trace_rejects_duplicate_experts_in_active_rows(tmp_path):
    counts = torch.tensor([[1, 0], [0, 0]], dtype=torch.int32)
    routes = torch.zeros(2, 2, 1, 16, dtype=torch.float32)
    path = tmp_path / "prefill.pt"
    torch.save({"routes": routes, "counts": counts}, path)
    with pytest.raises(ValueError, match="repeats an expert"):
        read_route_trace(path, counts, routes.shape, 8)
