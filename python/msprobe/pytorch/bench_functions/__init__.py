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

"""
gpu and cpu not implement benchmark function, supplementary benchmarking function implementation
"""

import os
from pkgutil import iter_modules
from importlib import import_module

package_path = os.path.dirname(os.path.realpath(__file__))
for _, module_name, _ in iter_modules([package_path]):
    module = import_module(f"{__name__}.{module_name}")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr) and "npu_custom" not in attr_name:
            globals()[attr_name] = attr
