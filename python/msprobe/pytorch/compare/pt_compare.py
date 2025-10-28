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

from python.msprobe.core.common.const import Const
from python.msprobe.core.compare.acc_compare import Comparator, ModeConfig, MappingConfig, setup_comparison
from python.msprobe.pytorch.compare.utils import read_pt_data
from python.msprobe.core.compare.utils import check_input_param_path_and_framework


def read_real_data(npu_dir, npu_data_name, bench_dir, bench_data_name, _) -> tuple:
    n_value = read_pt_data(npu_dir, npu_data_name)
    b_value = read_pt_data(bench_dir, bench_data_name)
    return n_value, b_value


def compare(input_param, output_path, **kwargs):
    check_input_param_path_and_framework(input_param, target_framework=Const.PT_FRAMEWORK)

    config = setup_comparison(input_param, output_path, **kwargs)

    config_dict = {
        'stack_mode': config.stack_mode,
        'auto_analyze': config.auto_analyze,
        'fuzzy_match': config.fuzzy_match,
        'highlight': config.highlight,
        'dump_mode': config.dump_mode,
        'first_diff_analyze': config.first_diff_analyze,
        'compared_file_type': config.compared_file_type
    }
    mode_config = ModeConfig(**config_dict)
    mapping_config = MappingConfig(data_mapping=config.data_mapping)
    pt_comparator = Comparator(read_real_data, mode_config, mapping_config)
    pt_comparator.compare_core(input_param, output_path, suffix=config.suffix)
