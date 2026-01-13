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

from msprobe.pytorch.bench_functions.apply_adam_w import npu_apply_adam_w
from msprobe.pytorch.bench_functions.confusion_transpose import npu_confusion_transpose, \
    npu_confusion_transpose_backward
from msprobe.pytorch.bench_functions.fast_gelu import npu_fast_gelu, npu_fast_gelu_backward
from msprobe.pytorch.bench_functions.layer_norm_eval import npu_layer_norm_eval
from msprobe.pytorch.bench_functions.linear import npu_linear, npu_linear_backward
from msprobe.pytorch.bench_functions.matmul_backward import matmul_backward
from msprobe.pytorch.bench_functions.npu_fusion_attention import npu_fusion_attention, npu_fusion_attention_grad, \
    gpu_fusion_attention
from msprobe.pytorch.bench_functions.rms_norm import npu_rms_norm, npu_rms_norm_backward
from msprobe.pytorch.bench_functions.rotary_mul import npu_rotary_mul, npu_rotary_mul_backward
from msprobe.pytorch.bench_functions.scaled_mask_softmax import npu_scaled_masked_softmax, \
    npu_scaled_masked_softmax_backward
from msprobe.pytorch.bench_functions.swiglu import npu_swiglu, npu_swiglu_backward
from msprobe.pytorch.bench_functions.apply_adam import npu_apply_adam
from msprobe.pytorch.bench_functions.group_norm_silu import npu_group_norm_silu
from msprobe.pytorch.bench_functions.mish import npu_mish
from msprobe.pytorch.bench_functions.moe_gating_top_k_softmax import npu_moe_gating_top_k_softmax
from msprobe.pytorch.bench_functions.sort_v2 import npu_sort_v2
from msprobe.pytorch.common.utils import logger


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target_func_list):
        for target in target_func_list:
            self.register(target)
        return

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def register(self, target):

        def add_register_item(key, value):
            if key in self._dict:
                logger.warning(f"{value.__name__} has been registered before, so we will override it.")
            self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            raise Exception(f"The func {target} is not callable.")


# register for npu custom bench functions
npu_custom_functions = Register()
npu_custom_functions([
    npu_apply_adam_w, npu_confusion_transpose, npu_fast_gelu, npu_layer_norm_eval, npu_linear, npu_fusion_attention,
    npu_rms_norm, npu_rotary_mul, npu_scaled_masked_softmax, npu_swiglu, gpu_fusion_attention, npu_apply_adam,
    npu_group_norm_silu, npu_mish, npu_moe_gating_top_k_softmax, npu_sort_v2
])

# register for npu custom backward bench functions
npu_custom_grad_functions = Register()
npu_custom_grad_functions([
    npu_confusion_transpose_backward, npu_fast_gelu_backward, npu_linear_backward, matmul_backward,
    npu_fusion_attention_grad, npu_rms_norm_backward, npu_rotary_mul_backward, npu_scaled_masked_softmax_backward,
    npu_swiglu_backward
])
