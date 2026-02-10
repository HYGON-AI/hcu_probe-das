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
from abc import ABC, abstractmethod


class BaseParser(ABC):
    """
    解析器基类，定义了解析器的通用接口
    
    所有具体的解析器都应该继承此类并实现抽象方法
    """
    
    @abstractmethod
    def parse(self, dump_path, output_path, parse_type):
        pass
    
    @staticmethod
    def get_output_file_path(input_file_path, output_dir, parse_type, extension=None):
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        if extension is None:
            extension = parse_type
        output_file = os.path.join(output_dir, f"{base_name}.{extension}")
        return output_file
