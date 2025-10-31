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

import os
from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import load_yaml


class AtenIrMapping():
    def __init__(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path, "mapping.yaml")
        self.aten_mapping = load_yaml(yaml_path)
    
    def match(self, op1, op2):
        if "Aten" in op1 and "Aten" not in op2:
            return self.match_op(op1, op2)
        else:
            return self.match_op(op2, op1)

    def match_op(self, aten_op, torch_op):
        try:
            aten_op_raw_name_overload = '_'.join(aten_op.split("_")[1:-3])
            aten_op_raw_name = aten_op_raw_name_overload.split('.')[0]
            torch_op_raw_name = '_'.join(torch_op.split("_")[1:-3]).lower()
        except IndexError as e:
            err_msg = f"Dump op name format error: {aten_op}, {torch_op}. Your dump data may be corrupted."
            raise CompareException.INVALID_DATA_ERROR(err_msg) from e
        matching_op = self.aten_mapping.get(aten_op_raw_name)
        if matching_op is None:
            return False
        if matching_op.lower() == torch_op_raw_name:
            return True
        return False


graph_mapping = AtenIrMapping()
