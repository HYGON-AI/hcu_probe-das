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

import re

from msprobe.core.compare.acc_compare import ModeConfig
from msprobe.core.compare.multiprocessing_compute import CompareRealData
from msprobe.core.compare.utils import read_op, merge_tensor, get_accuracy, make_result_table
from msprobe.core.common.utils import set_dump_path, get_dump_mode
from msprobe.visualization.utils import GraphConst
from msprobe.core.common.const import Const


# 用于将节点名字解析成对应的NodeOp的规则
op_patterns = [
    # NodeOp.module
    r'^(Module.|Cell.|optimizer|clip_grad)',
    # NodeOp.function_api
    r'^(Tensor.|Torch.|Functional.|NPU.|VF.|Distributed.|Aten.|Mint.|Primitive.|Jit.|MintFunctional.|MindSpeed.)'
]


def get_compare_mode(dump_path_param):
    """
    获得比较模式，包括summary、MD5和真实数据三种模式
    Args:
        dump_path_param: 调用acc_compare接口所依赖的参数
    Returns: 0 summary mode, 1 md5 mode, 2 true data mode
    """
    set_dump_path(dump_path_param)
    dump_mode = get_dump_mode(dump_path_param)
    compare_mode = GraphConst.DUMP_MODE_TO_GRAPHCOMPARE_MODE_MAPPING.get(dump_mode)
    return compare_mode


def run_real_data(dump_path_param, csv_path, framework, is_cross_frame=False):
    """
    多进程运行生成真实数据
    Args:
        dump_path_param: 调用acc_compare接口所依赖的参数
        csv_path: 生成文件路径
        framework: 框架类型, pytorch或mindspore
        is_cross_frame: 是否进行跨框架比对，仅支持mindspore比pytorch, 其中pytorch为标杆
    """
    config_dict = {
        'stack_mode': False,
        'auto_analyze': True,
        'fuzzy_match': False,
        'dump_mode': Const.ALL
    }
    mode_config = ModeConfig(**config_dict)

    if framework == Const.PT_FRAMEWORK:
        from msprobe.pytorch.compare.pt_compare import read_real_data
        return CompareRealData(read_real_data, mode_config, is_cross_frame).do_multi_process(dump_path_param, csv_path)
    else:
        from msprobe.mindspore.compare.ms_compare import read_real_data
        return CompareRealData(read_real_data, mode_config, is_cross_frame).do_multi_process(dump_path_param, csv_path)


def get_input_output(node_data, node_id):
    """
    将dump的原始数据进行拆解，分解为output和input两个数据
    Args:
        node_data: 属于单个节点的dump数据
        node_id: 节点名字
    """
    input_data = {}
    output_data = {}
    op_parsed_list = read_op(node_data, node_id)
    for item in op_parsed_list:
        full_op_name = item.get('full_op_name', '')
        if not full_op_name:
            continue
        if GraphConst.OUTPUT in full_op_name and GraphConst.INPUT not in full_op_name:
            output_data[full_op_name] = item
        else:
            name = item.get('data_name')
            # 节点参数名称尽量使用落盘数据的名称
            if isinstance(name, str) and name != '-1':
                input_data[name.rsplit(Const.SEP, 1)[0]] = item
            else:
                input_data[full_op_name] = item
    return input_data, output_data


def compare_data(data_dict_list1, data_dict_list2):
    """
    比较get_input_output中输出的结果是否结构一致，比较一致返回True
    """
    if len(data_dict_list1) != len(data_dict_list2):
        return False
    # 用于比较两个节点是否相等的关键字段
    tag_keys = ['type', 'shape']
    for key1, key2 in zip(data_dict_list1, data_dict_list2):
        dict1 = data_dict_list1[key1]
        dict2 = data_dict_list2[key2]
        for tag_key in tag_keys:
            tag_value1 = dict1.get(tag_key, None)
            tag_value2 = dict2.get(tag_key, None)
            if tag_value1 != tag_value2:
                return False
    return True


def compare_data_fuzzy(data_dict_list1, data_dict_list2):
    """
    模糊匹配，仅校验参数shape是否一致
    """
    for x, y in zip(data_dict_list1.values(), data_dict_list2.values()):
        x_shape = x.get(Const.SHAPE)
        y_shape = y.get(Const.SHAPE)
        if x_shape != y_shape:
            return False
    return True


def format_node_data(data_dict, node_id=None, compare_mode=None):
    """
    删除节点数据中不需要展示的字段
    """
    del_list = ['state', 'full_op_name']
    if GraphConst.MD5_COMPARE != compare_mode:
        del_list.append(Const.MD5)
    if node_id and GraphConst.BATCH_P2P in node_id:
        del_list.extend(['op', 'peer', 'tag', 'group_id'])
    for _, value in data_dict.items():
        if not isinstance(value, dict):
            continue
        for item in del_list:
            if item in value:
                del value[item]
        _format_data(value)
    return data_dict


def compare_node(node_n, node_b, compare_mode):
    """
    调用acc_compare.py中的get_accuracy获得精度对比指标
    真实数据对比模式无法获得精度对比指标，需要调用多进程比对接口
    Returns: 包含参数信息和对比指标（真实数据对比模式除外）的list
    """
    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    merge_n = _parse_node(node_n, dump_mode)
    merge_b = _parse_node(node_b, dump_mode)
    result = []
    get_accuracy(result, merge_n, merge_b, dump_mode)
    return result


def _parse_node(node, dump_mode):
    """
    转换节点，使其能够作为acc_compare.py中的get_accuracy的入参
    """
    op_parsed_list = []
    op_parsed_list.extend(node.input_data.values())
    op_parsed_list.extend(node.output_data.values())
    result = merge_tensor(op_parsed_list, dump_mode)
    if not result:
        result['op_name'] = []
    return result


def _format_decimal_string(s):
    """
    使用正则表达式匹配包含数字、小数点和可选的百分号的字符串
    """
    pattern = re.compile(r'^\d{1,20}\.\d{1,20}%?$')
    matches = pattern.findall(s)
    for match in matches:
        is_percent = match.endswith('%')
        number_str = match.rstrip('%')
        decimal_part = number_str.split('.')[1]
        # 如果小数位数大于6，进行处理
        if len(decimal_part) > GraphConst.ROUND_TH:
            number_float = float(number_str)
            formatted_number = f"{number_float:.{GraphConst.ROUND_TH}f}"
            # 如果原来是百分数，加回百分号
            if is_percent:
                formatted_number += '%'
            # 替换原字符串中的数值部分
            s = s.replace(match, formatted_number)
    return s


def _format_data(data_dict):
    """
    格式化数据，小数保留6位，处理一些异常值
    """
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)$'
    all_null = False

    keys_to_keep = ['type', 'group_ranks', 'group_id', 'data_name']
    if data_dict.get('type') == 'torch.ProcessGroup':
        keys_to_remove = [key for key in data_dict if key not in keys_to_keep]
        for key in keys_to_remove:
            del data_dict[key]

    for key, value in data_dict.items():
        if isinstance(value, str):
            # 将单引号删掉，None换成null避免前端解析错误
            value = value.replace("'", "").replace(GraphConst.NONE, GraphConst.NULL)
            value = _format_decimal_string(value)
        elif value is None or value == ' ':
            value = GraphConst.NULL
        # 科学计数法1.123123123123e-11，格式化为1.123123e-11
        elif isinstance(value, float) and len(str(value)) < GraphConst.STR_MAX_LEN and re.match(pattern, str(value)):
            value = "{:.6e}".format(value)
        elif isinstance(value, float):
            value = round(value, GraphConst.ROUND_TH)
        # Inf会走入这里，确保转成Inf。另外给其他不符合预期的类型做兜底方案
        if key != GraphConst.ERROR_KEY:
            # 除了error_key不转str，其他都转str, 避免前端解析错误
            value = str(value)
        # max为null, 意味着这个参数值为null
        if key == Const.MAX and value == GraphConst.NULL:
            all_null = True
        data_dict[key] = value
    # 字典里的value全null，只保留一个null
    if all_null:
        data_dict.clear()
        data_dict[GraphConst.VALUE] = GraphConst.NULL


def get_csv_df(stack_mode, csv_data, compare_mode):
    """
    调用acc接口写入csv
    """

    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    return make_result_table(csv_data, dump_mode, stack_mode)
