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

from msprobe.core.common.const import CompareConst
from msprobe.core.common.file_utils import logger
from msprobe.core.common.utils import CompareException


def replace_compare_index_dict(compare_index_dict, compare_index_list, rank_num):
    """
    比对指标值为N/A、unsupported、Nan，将比对指标值替换成NPU max 和 Bench max(几个统计量相同)

    示例：
    Distributed.all_reduce.0.forward.output.group的比对指标值是N/A
    替换后：
    比对指标值为:
        NPU: tp-0-1-2-3
        Bench: tp-0-1-2-3
    """

    if CompareConst.NPU_MAX not in compare_index_dict or CompareConst.BENCH_MAX not in compare_index_dict:
        compare_index_dict.pop(CompareConst.NPU_MAX, None)
        compare_index_dict.pop(CompareConst.BENCH_MAX, None)
        return compare_index_dict

    # 遍历比对指标列表，排除最后两个指标NPU max， Bench max
    for compare_index in compare_index_list[:-2]:
        op_name_index_dict = compare_index_dict[compare_index]
        # 遍历op_item名称和对应的比对指标值
        for op_name, index_value in op_name_index_dict.items():
            npu_max = compare_index_dict[CompareConst.NPU_MAX][op_name][rank_num]
            bench_max = compare_index_dict[CompareConst.BENCH_MAX][op_name][rank_num]
            # 如果当前比对指标值是N/A、unsupported、Nan，并且NPU和Bench的最大值是类型相同，进行替换
            if index_value[rank_num] in [CompareConst.N_A, CompareConst.UNSUPPORTED, CompareConst.NAN]:
                compare_index_dict[compare_index][op_name][rank_num] = f'NPU:{str(npu_max)}  Bench:{str(bench_max)}'

    # 删除NPU_MAX和BENCH_MAX
    compare_index_dict.pop(CompareConst.NPU_MAX, None)
    compare_index_dict.pop(CompareConst.BENCH_MAX, None)
    return compare_index_dict


def check_config(config):
    """
    config.yaml 内容检查
    Args: config:
    Returns: config
    """
    if not config:
        logger.error('config.yaml is empty, please check.')
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)

    api_list = config.get('api')
    if not api_list:
        logger.error('The APIs required to merge data were not found.')
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)
    if not isinstance(api_list, list):
        logger.error("The config format of 'api' is incorrect, please check.")
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)

    compare_index_list = config.get('compare_index', [])
    if compare_index_list is None:
        compare_index_list = []
        config['compare_index'] = compare_index_list
    if not isinstance(compare_index_list, list):
        logger.error("The config format of 'compare_index' is incorrect, please check.")
        raise CompareException(CompareException.MERGE_COMPARE_RESULT_ERROR)

    return config
