# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep the generation wrapper paired with the reusable backbone ABI."""
import ast
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[2] / "models/deepseek_v4_1_flash"


@pytest.mark.parametrize("level", ["l2", "l3"])
def test_generation_keeps_backbone_and_head_arguments(level):
    backbone = parameters("decode_fwd.py", level + "_decode_fwd")
    head = parameters("prefill_fwd.py", level + "_generation_head")
    generation = parameters("decode_generate.py", level + "_decode_generate")
    descriptors = ("expert_state", "expert_descriptors")
    shared = {"x_hc", "pre_mix", "rank"}
    backbone_inputs = list(backbone)
    if level == "l2":
        index = backbone_inputs.index("moe_workspace")
        backbone_inputs[index:index + 1] = [
            "recv_meta", "recv_x", "recv_routes", "data_arrived", "routed_output", "combine_arrived",
        ]
    assert list(generation) == (
        backbone_inputs
        + [name for name in head if name not in shared]
        + list(descriptors)
    )
    for name, annotation in backbone.items():
        if name.startswith("routed_w"):
            assert "ROUTED_BANK_ROWS" in generation[name], name
            if not name.endswith("_scale"):
                assert "MX_PACKED_LANE_COLS" in generation[name], name
        elif name != "moe_workspace":
            assert generation[name] == annotation, name
    for name, annotation in head.items():
        if name not in shared:
            assert generation[name] == annotation.replace("RANKS", "EP_SIZE"), name


def parameters(filename, name):
    module = ast.parse((MODEL / filename).read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return {argument.arg: ast.unparse(argument.annotation) for argument in function.args.args}
