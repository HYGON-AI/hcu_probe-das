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

from msprobe.core.compare.acc_compare import Comparator, ModeConfig, MappingConfig, setup_comparison
from msprobe.pytorch.compare.utils import read_pt_data


def read_real_data(npu_dir, npu_data_name, bench_dir, bench_data_name, _) -> tuple:
    n_value = read_pt_data(npu_dir, npu_data_name)
    b_value = read_pt_data(bench_dir, bench_data_name)
    return n_value, b_value


def pt_compare(input_param, output_path, **kwargs):
    config = setup_comparison(input_param, output_path, **kwargs)

    config_dict = {
        'stack_mode': config.stack_mode,
        'fuzzy_match': config.fuzzy_match,
        'dump_mode': config.dump_mode,
        'first_diff_analyze': config.first_diff_analyze,
        'compared_file_type': config.compared_file_type
    }
    mode_config = ModeConfig(**config_dict)
    mapping_config = MappingConfig(data_mapping=config.data_mapping)
    pt_comparator = Comparator(read_real_data, mode_config, mapping_config)
    pt_comparator.compare_core(input_param, output_path, suffix=config.suffix)
