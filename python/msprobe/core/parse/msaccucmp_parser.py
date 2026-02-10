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
import sys
import subprocess

from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.parse.base import BaseParser


class MsaccucmpParser(BaseParser):
    """
    使用msaccucmp.py解析数据的解析器
    """
    
    def parse(self, dump_path, output_path, parse_type):
        try:
            msaccucmp_script_path = self._get_msaccucmp_script_path()
            python_cmd = sys.executable
            cmd_args = [
                python_cmd,
                msaccucmp_script_path,
                "convert",
                "-d", dump_path,
                "-t", parse_type,
                "-out", output_path
            ]
            
            logger.info(f"Calling msaccucmp convert with command: {' '.join(cmd_args)}")
            process = subprocess.Popen(
                cmd_args,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.raw(line.rstrip())
            
            process.stdout.close()
            process.wait()
            
            return_code = process.returncode
            
            if return_code != 0:
                error_msg = f"msaccucmp convert execution failed with return code {return_code}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.info("msaccucmp convert execution completed successfully")
                
        except FileNotFoundError as e:
            error_msg = f"Failed to find msaccucmp.py: {str(e)}"
            logger.error(error_msg)
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR, error_msg) from e
        except Exception as e:
            error_msg = f"Failed to execute msaccucmp convert: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _get_msaccucmp_script_path(self):
        current_file = os.path.abspath(__file__)
        msprobe_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        script_path = os.path.join(msprobe_dir, "msaccucmp", "msaccucmp.py")
        check_file_or_directory_path(script_path, isdir=False)
        if os.path.exists(script_path):
            return os.path.abspath(script_path)
        
        error_msg = "msaccucmp.py file not found in msprobe package"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

