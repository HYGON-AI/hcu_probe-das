# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

FSDP_MODULE_MAP = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",  # 这些模块映射到 qkv_proj
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj"  # 这两个模块映射到 gate_up_proj
}

MEGATRON_MODULE_MAP = {
    "word_embeddings": "embed_tokens",
    "self_attention": "self_attn",
    "linear_qkv": "qkv_proj",
    "q_layernorm": "q_norm",
    "k_layernorm": "k_norm",
    "core_attention": "attn",
    "linear_proj": "o_proj",
    "linear_fc1": "gate_up_proj",
    "linear_fc2": "down_proj"
}
