# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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
import re
import sys
import subprocess

from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException

CANN_PATH = os.environ.get("ASCEND_TOOLKIT_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
MSACCUCMP_DIR_PATH = "toolkit/tools/operator_cmp/compare"
MSACCUCMP_SCRIPT = "msaccucmp.py"


def compare_offline_data_mode(args):
    cmd_args = []
    
    # 必需参数
    if args.target_path:
        cmd_args.extend(['-m', args.target_path])
    
    if args.golden_path:
        cmd_args.extend(['-g', args.golden_path])
    
    # 可选参数
    if args.fusion_rule_file:
        cmd_args.extend(['-f', args.fusion_rule_file])
    
    if args.quant_fusion_rule_file:
        cmd_args.extend(['-q', args.quant_fusion_rule_file])
    
    if args.close_fusion_rule_file:
        cmd_args.extend(['-cf', args.close_fusion_rule_file])
    
    if args.output_path:
        cmd_args.extend(['-out', args.output_path])
    
    call_msaccucmp(cmd_args)


def _check_msaccucmp_file(msaccucmp_script_path):
    msaccucmp_command_file_path = os.path.join(msaccucmp_script_path, MSACCUCMP_SCRIPT)
    if os.path.exists(msaccucmp_command_file_path):
        return msaccucmp_command_file_path
    else:
        logger.warning(
            'The path {} is not exist.Please check the file'.format(msaccucmp_command_file_path))
    logger.error(
        'Does not exist in {} directory msaccucmp.py and msaccucmp.pyc file'.format(msaccucmp_script_path))
    raise CompareException(CompareException.INVALID_PATH_ERROR)


def call_msaccucmp(cmd_args):
    """
    调用 msaccucmp.py 工具，透传从 args 中提取的参数。
    
    Args:
        args: argparse.Namespace 对象，包含所有解析的参数
        msaccucmp_script_path: msaccucmp.py 的路径，如果为 None，则自动查找
        
    Returns:
        subprocess.CompletedProcess: 命令执行结果
    """
    msaccucmp_script_path = _check_msaccucmp_file(os.path.join(CANN_PATH, MSACCUCMP_DIR_PATH))
    
    if not os.path.exists(msaccucmp_script_path):
        logger.error(f"msaccucmp script not found at: {msaccucmp_script_path}")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    
    # 构建完整命令
    python_cmd = sys.executable  # 使用当前 Python 解释器
    full_cmd = [python_cmd, msaccucmp_script_path, "compare"] + cmd_args
    
    logger.info(f"Calling msaccucmp with command: {' '.join(full_cmd)}")
  
    # 执行命令
    try:
        process = subprocess.Popen(
            full_cmd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if not re.match(r'^\d{4}-\d{2}-\d{2}', line):
                continue
            logger.raw(line)
        
        process.stdout.close()
        process.wait()

        return_code = process.returncode
        
        if return_code != 0 and return_code != 2:
            logger.error(f"msaccucmp execution failed with return code {return_code}")
        else:
            logger.info("msaccucmp execution completed successfully")
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to execute msaccucmp: {str(e)}")
        raise CompareException(CompareException.UNKNOWN_ERROR) from e
