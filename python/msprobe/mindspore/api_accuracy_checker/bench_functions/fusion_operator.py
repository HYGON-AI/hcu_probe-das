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

from msprobe.mindspore.api_accuracy_checker.bench_functions.flash_attention_score import FlashAttentionScore


class FusionOperator:
    """
    所有融合算子的父类，定义了通用的接口和属性。
    """

    # 初始化操作符字典
    def __init__(self):
        self.flash_attention_score = None  # 用于存放 FlashAttentionScore 操作符
        self._register_operators()

    def __getattr__(self, name):
        """ 动态获取算子类 """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'FusionOperator' object has no attribute '{name}'")

    def _register_operators(self):
        """ 注册操作符到父类，以便通过 fusion.xxx 调用 """
        self.flash_attention_score = FlashAttentionScore()


fusion = FusionOperator()
