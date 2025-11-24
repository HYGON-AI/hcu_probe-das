# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os

CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


class DumpArgsAdapter:
    def __init__(self,
                 model_path,
                 input_data="",
                 cann_path=CANN_PATH,
                 output_path="./output",
                 input_shape="",
                 rank="0",
                 dym_shape_range="",
                 onnx_fusion_switch=True,
                 custom_op="",
                 dump=True,
                 single_op=""
                 ):
        self.golden_path = model_path
        self.input_data = input_data
        self.cann_path = cann_path
        self.output_path = output_path
        self.input_shape = input_shape
        self.rank = rank
        self.dym_shape_range = dym_shape_range
        self.onnx_fusion_switch = onnx_fusion_switch
        self.custom_op = custom_op
        self.dump = dump
        self.single_op = single_op
