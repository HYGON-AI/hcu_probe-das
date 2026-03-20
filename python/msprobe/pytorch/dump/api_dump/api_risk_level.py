# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
import os
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.const import Const


def _load_core_apis_from_yaml():
    try:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
        ops = load_yaml(yaml_path)
        
        core_apis = set()
        
        dynamic_categories = ['mindspeed', 'torch_npu', 'distributed', 'npu_distributed']
        
        for category in dynamic_categories:
            api_list = ops.get(category, [])
            for api in api_list:
                if Const.SEP in api:
                    api = api.rsplit(Const.SEP, 1)[-1]
                core_apis.add(api)
        
        return core_apis
    except Exception:
        return set()


CORE_API_SET = {
    '_fused_adamw_', '_fused_adam_',
    'matmul', '__matmul__', 'bmm', 'addmm', 'addmm_', 'addbmm', 'addbmm_',
    'baddbmm', 'baddbmm_', 'mm', 'mv', 'addmv', 'addmv_', 'addr', 'addr_',
    'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 
    'conv_transpose3d', 'conv_tbc', 'convolution', '_convolution', "chain_matmul"
    'linear', 'bilinear', 'linalg.matmul', 'linalg.multi_dot', 'linalg.vecdot',
    'batch_norm', 'instance_norm', 'layer_norm', 'group_norm',
    'local_response_norm', '_batch_norm_impl_index', 'batch_norm_elemt',
    'batch_norm_stats', 'batch_norm_update_stats', 'batch_norm_gather_stats',
    'batch_norm_gather_stats_with_counts', 'batch_norm_backward_elemt',
    'batch_norm_backward_reduce',
    'embedding', 'embedding_bag', 'multi_head_attention_forward',
    'scaled_dot_product_attention'
}


CORE_API_SET.update(_load_core_apis_from_yaml())


LOW_RISK_API_SET = {
    'squeeze', 'unsqueeze', 'reshape', 'reshape_as',
    'view', 'view_as', 'flatten', 'unflatten', 'expand', 'expand_as', 
    'repeat', 'repeat_interleave', 'tile', 'transpose', 
    'permute', 'moveaxis', 'movedim', 'swapaxes', 'swapdims',
    'chunk', 'unsafe_chunk', 'split', 'split_with_sizes',
    'unsafe_split', 'unsafe_split_with_sizes', 'tensor_split', 'hsplit', 
    'vsplit', 'dsplit', 'unbind', 'stack', 'cat', 'concat', 'concatenate',
    'hstack', 'vstack', 'dstack', 'column_stack', 'row_stack', 'align_as', 
    'align_to', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'ravel', 'clamp', 
    'unfold', 'type_as',
    'to', 'float', 'double', 'int', 'long', 'half',
    'bfloat16', 'byte', 'char', 'bool', 'cfloat',
    'numpy', 'empty', 'empty_like', 'zeros', 'zeros_like', 'ones', 'ones_like',
    'full', 'full_like', 'arange', 'linspace', 'logspace', 'eye', 'identity',
    'diag', 'diagflat', 'diag_embed', 'triu', 'tril', 'triu_indices',
    'tril_indices','clone', '__bool__', 'stride', 
}


def get_api_risk_level(api_name):
    if api_name in CORE_API_SET:
        return Const.RISK_LEVEL_CORE
    if api_name in LOW_RISK_API_SET:
        return Const.RISK_LEVEL_LOW
    return Const.RISK_LEVEL_FOCUS
