# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
