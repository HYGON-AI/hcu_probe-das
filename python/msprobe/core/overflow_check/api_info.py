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
