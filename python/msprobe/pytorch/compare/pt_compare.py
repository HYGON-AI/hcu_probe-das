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
    mapping_config = MappingConfig(cell_mapping=config.cell_mapping, data_mapping=config.data_mapping)
    pt_comparator = Comparator(read_real_data, mode_config, mapping_config)
    pt_comparator.compare_core(input_param, output_path, suffix=config.suffix)
