# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Tensor layouts for the host-backed expert slot bank."""

import pypto.language as pl
from models.common.expert_cache import expert_cache_shape
from config import EP_SIZE, EXPERT_SLOTS, FLASH, N_LOCAL_EXPERTS


ROUTED_FIELDS = ("routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale")
BANK_DIRECTIONS = {"inout_params": ROUTED_FIELDS}
# Each mapped slot contains one contiguous, independently packed scale matrix.
SCALE_BANK_LAYOUT = pl.ND
CACHE_ROWS, CACHE_COLS = expert_cache_shape(EXPERT_SLOTS, FLASH.num_hidden_layers, N_LOCAL_EXPERTS)
RankState = pl.Tensor[[CACHE_ROWS, CACHE_COLS], pl.INT32]
RankDescriptors = pl.Tensor[[FLASH.num_hidden_layers, 6, 4], pl.INT64]
SlotMap = pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32]
RankStates = pl.Tensor[[EP_SIZE, CACHE_ROWS, CACHE_COLS], pl.INT32]
RankDescriptorTables = pl.Tensor[[EP_SIZE, FLASH.num_hidden_layers, 6, 4], pl.INT64]
