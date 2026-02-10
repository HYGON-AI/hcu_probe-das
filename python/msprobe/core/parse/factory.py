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

from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.parse.tensor_bin_parser import TensorBinFileParser
from msprobe.core.parse.msaccucmp_parser import MsaccucmpParser


def get_first_file_in_directory(directory):
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                return item_path
    except (OSError, PermissionError):
        return None
    return None


class ParserFactory:
    def get_parser(self, dump_path):
        """
        根据输入路径选择合适的解析器  
        """
        if os.path.isfile(dump_path):
            check_file_or_directory_path(dump_path, isdir=False)
            if dump_path.lower().endswith('.bin'):
                logger.info("Using ATB dump parser to parse .bin file")
                return TensorBinFileParser()
            else:
                logger.info("Using adump parser to parse file")
                return MsaccucmpParser()
        elif os.path.isdir(dump_path):
            check_file_or_directory_path(dump_path, isdir=True)
            first_file = get_first_file_in_directory(dump_path)
            if first_file is None:
                error_msg = f"The directory '{dump_path}' is empty or cannot be accessed. Please check the directory path and permissions."
                logger.error(error_msg)
                raise FileCheckException(FileCheckException.INVALID_FILE_ERROR, error_msg)
            elif first_file.lower().endswith('.bin'):
                logger.info("Using ATB dump parser to parse .bin files in directory")
                return TensorBinFileParser()
            else:
                logger.info("Using adump parser to parse files in directory")
                return MsaccucmpParser()
        else:
            error_msg = f"The path '{dump_path}' does not exist or is not a valid file or directory. Please check the path."
            logger.error(error_msg)
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR, error_msg)

