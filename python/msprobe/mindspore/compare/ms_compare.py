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
from python.msprobe.core.compare.layer_mapping import generate_data_mapping_by_layer_mapping
from python.msprobe.mindspore.compare.utils import read_npy_data, check_cross_framework
from python.msprobe.core.compare.utils import check_input_param_path_and_framework


def read_real_data(npu_dir, npu_data_name, bench_dir, bench_data_name, cross_frame) -> tuple:
    n_value = read_npy_data(npu_dir, npu_data_name)
    if cross_frame:
        from python.msprobe.pytorch.compare.utils import read_pt_data
        b_value = read_pt_data(bench_dir, bench_data_name)
    else:
        b_value = read_npy_data(bench_dir, bench_data_name)
    return n_value, b_value


def ms_compare(input_param, output_path, **kwargs):
    check_input_param_path_and_framework(input_param, target_framework=Const.MS_FRAMEWORK)

    config = setup_comparison(input_param, output_path, **kwargs)

    if config.layer_mapping:
        config.data_mapping = generate_data_mapping_by_layer_mapping(input_param, config.layer_mapping, output_path)

    is_cross_framework = check_cross_framework(input_param.get('bench_path'))

    config_dict = {
        'stack_mode': config.stack_mode,
        'auto_analyze': config.auto_analyze,
        'fuzzy_match': config.fuzzy_match,
        'highlight': config.highlight,
        'dump_mode': config.dump_mode,
        'compared_file_type': config.compared_file_type
    }
    mode_config = ModeConfig(**config_dict)
    mapping_config = MappingConfig(config.cell_mapping, config.api_mapping, config.data_mapping)
    ms_comparator = Comparator(read_real_data, mode_config, mapping_config, is_cross_framework)
    ms_comparator.compare_core(input_param, output_path, suffix=config.suffix)
