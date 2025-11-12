# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
import importlib
import inspect

from msprobe.core.common.file_utils import load_yaml, check_link
from msprobe.core.common.log import logger


def get_ops():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
    ops = load_yaml(yaml_path)
    wrap_functional = ops.get('functional')
    wrap_tensor = ops.get('tensor')
    wrap_torch = ops.get('torch')
    wrap_npu_ops = ops.get('torch_npu')
    return set(wrap_functional) | set(wrap_tensor) | set(wrap_torch) | set(wrap_npu_ops)


def dynamic_import_op(package, white_list):
    package_name = package.__name__
    ops = {}
    ops_dir, _ = os.path.split(package.__file__)
    check_link(ops_dir)
    for file_name in os.listdir(ops_dir):
        if file_name in white_list:
            sub_module_name = file_name[:-3]
            module_name = f"{package_name}.{sub_module_name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                logger.warning(f"import {module_name} failed!")
                continue

            func_members = inspect.getmembers(module, inspect.isfunction)
            for func_member in func_members:
                func_name, func = func_member[0], func_member[1]
                ops[f"{sub_module_name}.{func_name}"] = func
    return ops
