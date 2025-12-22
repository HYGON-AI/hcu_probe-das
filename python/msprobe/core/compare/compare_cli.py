# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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


from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.atb_data_compare import compare_atb_mode
from msprobe.core.compare.auto_compare import compare_auto_mode
from msprobe.core.compare.offline_data_compare import compare_offline_data_mode
from msprobe.core.compare.torchair_acc_cmp import compare_torchair_mode
from msprobe.infer.offline.compare.msquickcmp.main import compare_offline_model_mode


MODE_DISPATCHER = {
    'auto': compare_auto_mode,
    'offline_data': compare_offline_data_mode,
    'torchair': compare_torchair_mode,
    'offline_model': compare_offline_model_mode,
    'atb': compare_atb_mode
}

VALID_ARGS_MAP = {
    'auto': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o', 
        '--fuzzy_match', '-fm', '--cell_mapping', '-cm', '--api_mapping', '-am', '--data_mapping',
        '-dm', '--layer_mapping', '-lm', '--diff_analyze', '-da', '--rank', '--step',
        '--is_print_compare_log', '-tensor_log'
    ],
    'offline_data': [
        "-m", "-tp", "-gp", "-fr", "-qfr", "-cfr", "-o", "--mode", "--target_path", "--golden_path", 
        "--fusion_rule_file", "--quant_fusion_rule_file", "--close_fusion_rule_file", "--output_path"
    ],
    'torchair': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o'
    ],
    'offline_model': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o',
        '--rank', '--input_data', '--input_shape', '--output_size', '--dym_shape_range',
        '-ofs', '--onnx_fusion_switch'
    ],
    'atb': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o'
    ],
}


def check_valid_args(sys_argv, mode):
    valid_args = VALID_ARGS_MAP.get(mode)
    for arg in sys_argv:
        if arg.startswith("-") and arg not in valid_args:
            logger.error(f"Invalid argument '{arg}' for mode '{mode}'")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)


def compare_cli(args, sys_argv):
    """
    Dispatch comparison based on mode parameter
    """
    mode = getattr(args, 'mode', 'auto')
    # Get the appropriate function based on mode
    compare_func = MODE_DISPATCHER.get(mode)
    if compare_func is None:
        logger.error(f"Invalid mode '{mode}'. Available modes: {list(MODE_DISPATCHER.keys())}")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
    check_valid_args(sys_argv, mode)
    # Execute the comparison function
    return compare_func(args)
