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

from dataclasses import dataclass

from typing import Dict, List

from msprobe.core.common.const import Const


@dataclass
class APIInfo:
    api_name: str
    torch_api_name: str
    input_args: List[Dict]
    input_kwargs: Dict
    output_data: List[Dict]

    def __init__(self, api_name, input_args=None, input_kwargs=None, output_data=None):
        self.api_name = api_name
        self.input_args = input_args
        self.input_kwargs = input_kwargs
        self.output_data = output_data
        self.torch_api_name = self.extract_torch_api(self.api_name)

    @staticmethod
    def extract_torch_api(api_name) -> str:
        """
        Process tensor api name to extract first two fields in lowercase.
        """
        # Empty string checking
        if not api_name.strip():
            return ""

        parts = api_name.split(Const.SEP)

        # Handle different cases based on number of parts
        if len(parts) == 0:
            return ""
        elif len(parts) == 1:
            return parts[0].lower()
        else:
            return Const.SEP.join(parts[:2]).lower()
