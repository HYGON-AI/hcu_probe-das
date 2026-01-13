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
import re
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import load_json, remove_path, create_directory, save_excel, save_json
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, add_time_with_xlsx, check_op_str_pattern_valid, \
    set_dump_path, get_dump_mode, check_compare_param, load_stack_json, get_file_type, add_time_with_json
from msprobe.core.compare.check import check_dump_json_str, check_stack_json_str, cross_dtype_mapping, \
    check_configuration_param
from msprobe.core.compare.utils import print_compare_ends_info, read_op, set_stack_json_path, reorder_index
from msprobe.core.compare.config import ModeConfig, MappingConfig, MappingDict
from msprobe.core.compare.multiprocessing_compute import CompareRealData
from msprobe.core.compare.diff_analyze.first_diff_analyze import FirstDiffAnalyze
from msprobe.core.compare.indicator_analysis.calculator import calculate_excel_result_df


@dataclass
class ComparisonConfig:
    dump_mode: str
    stack_mode: bool
    fuzzy_match: bool
    data_mapping: dict
    suffix: str
    cell_mapping: dict
    api_mapping: dict
    layer_mapping: dict
    compared_file_type: str
    first_diff_analyze: bool
    is_print_compare_log: bool


class Comparator:
    def __init__(self, file_reader, mode_config: ModeConfig, mapping_config: MappingConfig, is_cross_framework=False):
        self.file_reader = file_reader
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.cross_frame = is_cross_framework
        self.mapping_dict = MappingDict(mapping_config)

    def process_output_file(self, output_path, suffix, compared_file_type):
        file_name_prefix_mapping = {
            Const.DUMP_JSON_FILE: "compare_result",
            Const.DEBUG_JSON_FILE: "debug_compare_result"
        }
        file_name_prefix = file_name_prefix_mapping.get(compared_file_type, "compare_result")
        if self.mode_config.first_diff_analyze:
            file_name = add_time_with_json("compare_result" + suffix)
        else:
            file_name = add_time_with_xlsx(file_name_prefix + suffix)
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        if os.path.exists(file_path):
            logger.warning(f"{file_path} will be deleted.")
            remove_path(file_path)
        return file_path

    def compare_core(self, input_param, output_path, **kwargs):
        """
        Compares data from multiple JSON files and generates a comparison report.

        Args:
            input_param (dict): A dictionary containing paths to JSON files ("npu_path", "bench_path",
                                "stack_path").
            output_path (str): The path where the output Excel report will be saved.
            **kwargs: Additional keyword arguments including:
            - stack_mode (bool, optional): Enables stack mode comparison. Defaults to False.
            - auto_analyze (bool, optional): If True, triggers automatic analysis after comparison. Defaults to True.
            - suffix (str, optional): Suffix to append to the output file name. Defaults to ''.
            - fuzzy_match (bool, optional): Enables fuzzy matching during comparison. Defaults to False.
            - dump_mode (str): ALL, SUMMARY, MD5.

        Returns:
        """
        logger.info("Please check whether the input data belongs to you. If not, there may be security risks.")

        # get kwargs or set default value
        suffix = kwargs.get('suffix', '')
        rank = suffix[1:]

        # process output file
        file_path = self.process_output_file(output_path, suffix, self.mode_config.compared_file_type)

        # initialize the compare result table and compare general data(name, dtype, shape, statistics/md5, etc.)
        npu_json = input_param.get("npu_path")
        bench_json = input_param.get("bench_path")
        stack_json = input_param.get("stack_path")
        parse_data = ParseData(self.mode_config, rank)  # load and parse json data
        npu_df, bench_df = parse_data.parse([npu_json, bench_json, stack_json])
        result_df = self.compare_statistics(npu_df, bench_df)
        if not result_df.values.tolist():
            logger.warning("Can`t match any op. No compare result file generated.")
            return

        if self.mode_config.first_diff_analyze:
            # add P2POp additional info from npu_df and bench_df to result_df
            result_df['NPU P2POp op'] = npu_df['op']
            result_df['Bench P2POp op'] = bench_df['op']
            result_df['NPU P2POp peer'] = npu_df['peer']
            result_df['Bench P2POp peer'] = bench_df['peer']

            first_diff_analyze = FirstDiffAnalyze(self.mode_config, rank)
            check_result = first_diff_analyze.check(result_df)
            save_json(file_path, check_result, indent=4)
            logger.info(f"Saving json file to disk: {file_path}")
            return

        # compare real data
        if self.mode_config.dump_mode == Const.ALL:
            compare_real_data = CompareRealData(self.file_reader, self.mode_config, self.cross_frame)
            result_df = compare_real_data.do_multi_process(input_param, result_df)

        # save result excel file
        logger.info(f'Saving result excel file in progress. The file path is: {file_path}.')
        calculate_excel_result_df(result_df, self.mode_config.dump_mode)
        result_df.drop(columns=['state', 'api_origin_name'], inplace=True)  # 删除中间数据，两列不落盘
        save_excel(file_path, result_df)

        print_compare_ends_info()

    def compare_statistics(self, npu_df, bench_df):
        npu_df[[Const.DTYPE, Const.SHAPE]] = npu_df[[Const.DTYPE, Const.SHAPE]].astype(str)
        bench_df[[Const.DTYPE, Const.SHAPE]] = bench_df[[Const.DTYPE, Const.SHAPE]].astype(str)

        # create new columns for compare op_name and shape
        # process npu_df's COMPARE_KEY whether same or different framework
        process_df = ProcessDf(self.mode_config, self.mapping_config, self.mapping_dict)
        npu_df, bench_df = process_df.process_compare_key_and_shape(npu_df, bench_df)

        # match npu and bench, match_result contains both npu_info and bench_info
        match = Match(self.mode_config, self.mapping_config, self.cross_frame)
        match_result = match.match_api_infos(npu_df, bench_df)
        # 筛选出npu_name存在的行并填充筛选出行中的缺失值为N/A
        match_result = match_result[match_result['op_name_x'].notna()].fillna(CompareConst.N_A)
        bench_columns = [i + '_y' for i in bench_df.columns]
        match_result.loc[~match.gen_dtype_condition(match_result), bench_columns] = CompareConst.N_A

        # organize compare result table by renaming columns
        if self.mode_config.dump_mode == Const.ALL and self.mode_config.first_diff_analyze:
            self.mode_config.dump_mode = Const.SUMMARY
        create_table = CreateTable(self.mode_config)
        result_df, header = create_table.make_result_df(match_result)

        # calculate statistics diff
        calc_stats_diff = CalcStatsDiff(self.mode_config)
        return calc_stats_diff.calc_accuracy(result_df, header)


class ParseData:
    def __init__(self, mode_config: ModeConfig, rank):
        self.mode_config = mode_config
        self.rank = rank

    def parse(self, file_list):
        npu_json_path, bench_json_path, stack_json_path = file_list
        npu_json_data = load_json(npu_json_path)
        bench_json_data = load_json(bench_json_path)
        stack_json_data = load_stack_json(stack_json_path) if self.mode_config.stack_mode else None

        # parse json data and generate df
        npu_df = self.gen_data_df(npu_json_data, stack_json_data, 'NPU')
        bench_df = self.gen_data_df(bench_json_data, stack_json_data, 'Bench')

        return npu_df, bench_df

    def gen_data_df(self, data_json, stack_json_data, device: str):
        result = {
            CompareConst.OP_NAME: [],
            Const.DTYPE: [],
            Const.SHAPE: [],
            Const.SUMMARY: [],
            Const.STACK_INFO: [],
            Const.STATE: [],
            Const.API_ORIGIN_NAME: [],
            Const.REQ_GRAD: []
        }
        if self.mode_config.dump_mode == Const.ALL:
            result[Const.DATA_NAME] = []
        elif self.mode_config.dump_mode == Const.MD5:
            result[Const.MD5] = []

        apis_data = data_json.get('data', None)
        if not apis_data:
            logger.warning('No APIs found in dump.json.')
            return pd.DataFrame(result)

        api_nums = len(apis_data)
        default_bar_desc = f'{device} API/Module Read Progress'
        bar_desc_add_rank = f'[{self.rank}]' + default_bar_desc if self.rank else default_bar_desc
        progress_bar = tqdm(total=api_nums, desc=bar_desc_add_rank, unit="api/module", ncols=100)

        # 从json中循环解析API数据，遍历所有API
        for data_name in apis_data:
            check_op_str_pattern_valid(data_name)
            op_parsed_list = self.gen_merge_list(data_json, data_name, stack_json_data)
            if not op_parsed_list:
                continue
            reordered_index_list = reorder_index(op_parsed_list)
            for i, index in enumerate(reordered_index_list):
                op_item = op_parsed_list[index]

                # common key
                result[CompareConst.OP_NAME].append(op_item.get('full_op_name'))
                result[Const.DTYPE].append(op_item.get(Const.DTYPE))
                result[Const.SHAPE].append(op_item.get(Const.SHAPE))
                result[Const.STATE].append(op_item.get(Const.STATE))
                result[Const.REQ_GRAD].append(op_item.get(Const.REQ_GRAD))
                result[Const.API_ORIGIN_NAME].append(data_name)
                summary_data = [
                    str(op_item.get(key)) if op_item.get(key) is None else op_item.get(key)
                    for key in Const.SUMMARY_METRICS_LIST
                ]
                result[Const.SUMMARY].append(summary_data)

                # dump_mode differ key
                if self.mode_config.dump_mode == Const.MD5:
                    result[Const.MD5].append(op_parsed_list[index].get(Const.MD5))
                if self.mode_config.dump_mode == Const.ALL:
                    result[Const.DATA_NAME].append(op_item.get(Const.DATA_NAME))

                # mode_config stack_mode addition key
                if i == 0 and self.mode_config.stack_mode:
                    result[Const.STACK_INFO].append(op_parsed_list[-1].get('full_info'))
                else:
                    result[Const.STACK_INFO].append(None)

                # mode_config first_diff_analyze addition key
                if self.mode_config.first_diff_analyze:
                    result.setdefault('op', []).append(op_item.get('op', str(None)))
                    result.setdefault('peer', []).append(op_item.get('peer', str(None)))

            progress_bar.update(1)
        progress_bar.close()
        return pd.DataFrame(result)

    def gen_merge_list(self, json_data, op_name, stack_json_data):
        op_data = json_data['data'][op_name]
        if self.mode_config.compared_file_type == Const.DUMP_JSON_FILE:
            check_dump_json_str(op_data, op_name)
        op_parsed_list = read_op(op_data, op_name)

        if self.mode_config.stack_mode:
            stack_info = stack_json_data.get(op_name)
            if stack_info is not None:
                check_stack_json_str(stack_info, op_name)
        else:
            stack_info = None
        # always add stack_info whether stack_mode is True
        op_parsed_list.append({
            'full_op_name': op_name,
            'full_info': stack_info
        })
        return op_parsed_list


class ProcessDf:
    def __init__(self, mode_config: ModeConfig, mapping_config: MappingConfig, mapping_dict: MappingDict):
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.mapping_dict = mapping_dict

    @staticmethod
    def get_api_name(api_list):
        try:
            api_name = api_list[0] + Const.SEP + api_list[1]
        except IndexError as error:
            logger.error('Failed to retrieve API name, please check if the dump data is reasonable')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        return api_name

    def process_compare_key_and_shape(self, npu_df, bench_df):
        npu_df = self.assign_npu_df_compare_key(npu_df, bench_df)
        npu_df[CompareConst.CMP_SHAPE] = npu_df[Const.SHAPE]
        bench_df[CompareConst.CMP_KEY] = bench_df[CompareConst.OP_NAME]
        bench_df[CompareConst.CMP_SHAPE] = bench_df[Const.SHAPE]
        return npu_df, bench_df

    def assign_npu_df_compare_key(self, npu_df, bench_df):
        """
        处理 npu_df 的 COMPARE_KEY 赋值逻辑

        :param npu_df: DataFrame，NPU 对比数据
        :param bench_df: DataFrame，Bench 对比数据
        :return: compare_key(name)处理后的 npu_df
        """
        # 处理api_mapping映射
        if self.mapping_config.api_mapping:
            # 如果用户不传api_mapping.yaml，先使用内置api_mapping.yaml替换npu_op_name
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_internal_api_mapping)
            # 如果用户传入api_mapping.yaml，再使用传入api_mapping.yaml进一步替换npu_op_name
            if isinstance(self.mapping_config.api_mapping, str):
                self.modify_compare_data_with_user_mapping(npu_df, bench_df)
        # 处理cell_mapping映射
        elif self.mapping_config.cell_mapping:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_cell_mapping)
        # 处理data_mapping映射
        elif self.mapping_config.data_mapping:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_data_mapping)
        else:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME]
        return npu_df

    def process_internal_api_mapping(self, npu_op_name):
        # get api name & class name from op_name
        ms_api_name = self.get_api_name(npu_op_name.split(Const.SEP))
        class_name = ms_api_name.split(Const.SEP)[0]
        if class_name == "Mint":
            return npu_op_name.replace("Mint", "Torch")
        elif class_name == "MintFunctional":
            return npu_op_name.replace("MintFunctional", "Functional")
        elif self.mapping_dict.ms_to_pt_mapping.get(ms_api_name):
            return npu_op_name.replace(ms_api_name, self.mapping_dict.ms_to_pt_mapping.get(ms_api_name))
        else:
            return npu_op_name

    def modify_compare_data_with_user_mapping(self, npu_df, bench_df):
        def remove_prefix(string, prefix):
            if string.startswith(prefix):
                return string[len(prefix):]
            return string

        def gen_input_compare_key(pattern, term):
            is_unmatched = True
            for i, prefix in enumerate(mapping_dict.get(f'ms_{term}')):
                if remove_prefix(op_name, api_origin_name + pattern) == str(prefix):
                    npu_df.loc[index, CompareConst.CMP_KEY] = (
                        op_name.replace(pattern + str(prefix), pattern + str(mapping_dict.get(f'pt_{term}')[i])))
                    is_unmatched = False
            return is_unmatched

        ms_api_indices_dict = self.get_api_indices_dict(npu_df)
        pt_api_indices_dict = self.get_api_indices_dict(bench_df)

        for mapping_dict in self.mapping_dict.api_mapping_dict:
            all_length_equal = True
            for k1, k2 in CompareConst.API_MAPPING_KEYS_TO_COMPARE:
                if len(mapping_dict.get(k1, [])) != len(mapping_dict.get(k2, [])):
                    all_length_equal = False
            if not all_length_equal:
                logger.warning('The user-defined mapping table is incorrect,\
                                make sure that the number of parameters is equal')
                continue

            ms_api, pt_api = mapping_dict.get('ms_api'), mapping_dict.get('pt_api')
            if ms_api not in ms_api_indices_dict or pt_api not in pt_api_indices_dict:
                continue
            for index in ms_api_indices_dict.get(ms_api):
                op_name = npu_df.loc[index, CompareConst.OP_NAME].replace(ms_api, pt_api, 1)
                state = npu_df.loc[index, Const.STATE]
                api_origin_name = npu_df.loc[index, Const.API_ORIGIN_NAME].replace(ms_api, pt_api, 1)
                if state == Const.INPUT:
                    is_abandoned = gen_input_compare_key(CompareConst.INPUT_PATTERN, 'args')
                elif state == Const.KWARGS:
                    is_abandoned = gen_input_compare_key(CompareConst.KWARGS_PATTERN, 'args')
                elif state == Const.OUTPUT:
                    is_abandoned = gen_input_compare_key(CompareConst.OUTPUT_PATTERN, 'output')
                elif state == Const.PARAMS:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_PATTERN, 'parameters')
                elif state == Const.PARAMS_GRAD:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_GRAD_PATTERN, 'parameters_grad')
                else:
                    logger.error(f'Excepted op_name: {op_name}')
                    raise CompareException(CompareException.INVALID_DATA_ERROR)
                if is_abandoned:
                    npu_df.loc[index, CompareConst.CMP_KEY] = op_name + 'abandoned'

    def get_api_indices_dict(self, op_name_df):
        """
        生成多个api对应的各自的所有的input、output等的index的键值对字典
        示例：
        {'Functional.conv2d': [0, 1, 2, 3],
        'Functional.batch_norm': [4, 5, 6, 7, 8]
        }
        """
        api_indices_dict = defaultdict(list)
        for op_index, name in enumerate(op_name_df[CompareConst.OP_NAME]):
            api_name = self.get_api_name(name.split(Const.SEP))
            api_indices_dict[api_name].append(op_index)
        return api_indices_dict

    def process_cell_mapping(self, npu_op_name):
        if not npu_op_name:
            return CompareConst.N_A
        param_grad_flag = Const.PARAMS_GRAD in npu_op_name.split(Const.SEP)
        if not param_grad_flag and not re.search(Const.REGEX_FORWARD_BACKWARD, npu_op_name):
            return CompareConst.N_A
        npu_op_name = npu_op_name.replace("Cell", "Module", 1)
        if self.mapping_dict.cell_mapping_dict:
            # get cell name & class name from op_name
            # Cell.fc1.Dense.forward.0.input.0
            # npu_op_name.split(Const.SEP, 1)[-1] 表示Module或Cell字段后面的部分
            cell_name = re.split(r'\.(?:forward|backward|parameters_grad)\.', npu_op_name.split(Const.SEP, 1)[-1])[0]

            # 1. 精确整段匹配（保持原行为）
            if cell_name in self.mapping_dict.cell_mapping_dict:
                new_cell_name = self.mapping_dict.cell_mapping_dict[cell_name]
            # 2. 兜底：简单字符串替换（不做任何限制）
            else:
                new_cell_name = cell_name
                for target_name, golden_name in self.mapping_dict.cell_mapping_dict.items():
                    if target_name in new_cell_name:
                        new_cell_name = new_cell_name.replace(target_name, golden_name, 1)
            # 3. 应用替换
            if new_cell_name != cell_name:
                npu_op_name = npu_op_name.replace(cell_name, new_cell_name, 1)

        return npu_op_name

    def process_data_mapping(self, npu_op_name):
        return self.mapping_dict.data_mapping_dict.get(npu_op_name, npu_op_name)


class Match:
    def __init__(self, mode_config: ModeConfig, mapping_config: MappingConfig, cross_frame):
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.cross_frame = cross_frame

    @staticmethod
    def put_unmatched_in_table(match_result, npu_op_item):
        npu_columns = npu_op_item.index.tolist()[:-2]
        bench_columns = [name + '_y' for name in npu_columns]
        na_series = pd.Series([CompareConst.N_A] * len(bench_columns), index=bench_columns)
        new_result_item = pd.concat([npu_op_item, na_series]).to_frame().T
        new_result_item.columns = CompareConst.MATCH_RESULT_COLUMNS
        match_result = pd.concat([match_result, new_result_item])
        return match_result

    @staticmethod
    def put_matched_in_table(match_result, npu_op_item, bench_op_item):
        head_len = len(CompareConst.MATCH_RESULT_COLUMNS)
        new_result_item = pd.concat([npu_op_item, bench_op_item]).head(head_len).to_frame().T
        new_result_item.columns = CompareConst.MATCH_RESULT_COLUMNS
        match_result = pd.concat([match_result, new_result_item])
        return match_result

    @staticmethod
    def rename_api(op_name):
        """
        原api： {api_type}.{api_name}.{API调用次数}.{前向反向}.{input/output}.{参数序号}
        rename后： {api_type}.{api_name}.{前向反向}.{input/output}.{参数序号}
        """
        if Const.FORWARD not in op_name and Const.BACKWARD not in op_name:
            return op_name
        process = Const.FORWARD if Const.FORWARD in op_name else Const.BACKWARD
        name_split = op_name.split(process)
        try:
            torch_func_index, in_out = name_split[0], name_split[1]
        except IndexError as error:
            logger.error(f'{op_name} can not be split with {process}, please check!')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        torch_func_split = torch_func_index.rsplit(Const.SEP, 2)
        torch_func = str(torch_func_split[0]) + Const.SEP + process + str(in_out)
        return torch_func

    def check_op_item(self, npu_op_item, bench_op_item):
        name_match = self.rename_api(npu_op_item[CompareConst.CMP_KEY]) == self.rename_api(
            bench_op_item[CompareConst.CMP_KEY])
        if name_match:
            return True
        else:
            npu_op_name = npu_op_item[CompareConst.OP_NAME]
            bench_op_name = bench_op_item[CompareConst.OP_NAME]
            check_op_str_pattern_valid(npu_op_name)
            check_op_str_pattern_valid(bench_op_name)
            logger.warning(f"{npu_op_name} and {bench_op_name} can not fuzzy match")
            return False

    def match_api_infos(self, npu_df, bench_df):
        """
        正常匹配和模糊匹配
        """
        if self.mapping_config.data_mapping:
            match_result = pd.merge(npu_df, bench_df, on=[CompareConst.CMP_KEY], how='left')

            # reorder match_result by op_name of npu
            op_name_order = npu_df[CompareConst.OP_NAME].tolist()
            match_result[CompareConst.OP_NAME_X] = pd.Categorical(match_result[CompareConst.OP_NAME_X],
                                                                  categories=op_name_order, ordered=True)
            match_result = match_result.sort_values(CompareConst.OP_NAME_X).reset_index(drop=True)
            match_result[CompareConst.OP_NAME_X] = match_result[CompareConst.OP_NAME_X].astype('object')
        elif not self.mode_config.fuzzy_match:
            match_result = pd.merge(npu_df, bench_df, on=[CompareConst.CMP_KEY],
                                    how='outer')
        else:
            match_result = self.process_fuzzy_match(npu_df, bench_df)
        return match_result

    def process_fuzzy_match(self, npu_df, bench_df):
        """
        模糊匹配通过循环方式匹配api
        """
        npu_ops_queue = []
        bench_ops_queue = []
        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)

        max_len = max(len(npu_df), len(bench_df))
        min_len = min(len(npu_df), len(bench_df))
        for i in range(max_len):
            if i < min_len:
                npu_ops_queue.append(npu_df.iloc[i])
                bench_ops_queue.append(bench_df.iloc[i])
            else:
                try:
                    npu_ops_queue.append(npu_df.iloc[i])
                except IndexError:
                    pass
                try:
                    bench_ops_queue.append(bench_df.iloc[i])
                except IndexError:
                    pass

            # 如果append之后queue状态不一致，则判断结束
            if bool(npu_ops_queue) ^ bool(bench_ops_queue):
                break

            npu_match_point, bench_match_point = self.match_op(npu_ops_queue, bench_ops_queue)

            # 如果没有匹配到，数据放到队列中，跳过。直到后面匹配到，把匹配之前的api放到不匹配中
            if npu_match_point == -1 and bench_match_point == -1:
                continue

            npu_op_item = npu_ops_queue[npu_match_point]
            bench_op_item = bench_ops_queue[bench_match_point]
            unmatched_data = npu_ops_queue[0: npu_match_point]
            for op_item in unmatched_data:
                match_result = self.put_unmatched_in_table(match_result, op_item)
            match_result = self.put_matched_in_table(match_result, npu_op_item, bench_op_item)
            del npu_ops_queue[0: npu_match_point + 1]
            del bench_ops_queue[0: bench_match_point + 1]

        if npu_ops_queue:
            for op_item in npu_ops_queue:
                match_result = self.put_unmatched_in_table(match_result, op_item)

        match_result.reset_index(drop=True, inplace=True)
        return match_result

    def match_op(self, npu_queue, bench_queue):
        for b_index, b_op in enumerate(bench_queue[0: -1]):
            if self.check_op_item(npu_queue[-1], b_op):
                return len(npu_queue) - 1, b_index
        if self.check_op_item(npu_queue[-1], bench_queue[-1]):
            return len(npu_queue) - 1, len(bench_queue) - 1
        for n_index, n_op in enumerate(npu_queue[0: -1]):
            if self.check_op_item(n_op, bench_queue[-1]):
                return n_index, len(bench_queue) - 1
        return -1, -1

    def gen_dtype_condition(self, match_result):
        """
        dtype匹配条件为npu、bench的dtype一致或属于规定的映射关系
        """
        # 如果使用了data_mapping，不校验dtype，返回全True的DataFrame
        if self.mapping_config.data_mapping:
            return pd.Series(True, index=match_result.index)

        npu_dtype = match_result['dtype_x']
        bench_dtype = match_result['dtype_y']
        npu_dtype = self.process_cross_frame_dtype(npu_dtype)
        bench_dtype = self.process_cross_frame_dtype(bench_dtype)

        equal_condition = npu_dtype == bench_dtype
        match_condition = (
                (npu_dtype.isin(CompareConst.DTYPE_MATCH_GROUPS[0]) & bench_dtype.isin(
                    CompareConst.DTYPE_MATCH_GROUPS[0])) |
                (npu_dtype.isin(CompareConst.DTYPE_MATCH_GROUPS[1]) & bench_dtype.isin(
                    CompareConst.DTYPE_MATCH_GROUPS[1]))
        )
        return equal_condition | match_condition

    def process_cross_frame_dtype(self, dtype):
        if self.cross_frame:
            dtype = dtype.map(cross_dtype_mapping).fillna(dtype)
        return dtype


class CreateTable:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    @staticmethod
    def process_data_name(result):
        result['data_name_x'] = result.apply(lambda row: [row['data_name_x'], row['data_name_y']], axis=1)
        return result

    @staticmethod
    def set_summary(summary):
        if summary == CompareConst.N_A:
            return [CompareConst.N_A] * 4  # 4为统计值个数
        summary_list = []
        for i in summary:
            if str(i).lower() == 'nan':
                summary_list.append(CompareConst.NAN)
            else:
                summary_list.append(i)
        return summary_list

    def make_result_df(self, result):
        # get header
        header = CompareConst.HEAD_OF_COMPARE_MODE[self.mode_config.dump_mode][:]
        if self.mode_config.stack_mode:
            header.append(CompareConst.STACK)
        if self.mode_config.dump_mode == Const.ALL:
            header.append(CompareConst.DATA_NAME)
            result = self.process_data_name(result)

        # rename match_result columns
        result.rename(columns={'op_name_x': CompareConst.NPU_NAME,
                               'op_name_y': CompareConst.BENCH_NAME,
                               'dtype_x': CompareConst.NPU_DTYPE,
                               'dtype_y': CompareConst.BENCH_DTYPE,
                               'shape_x': CompareConst.NPU_SHAPE,
                               'shape_y': CompareConst.BENCH_SHAPE,
                               'md5_x': CompareConst.NPU_MD5,
                               'md5_y': CompareConst.BENCH_MD5,
                               'data_name_x': CompareConst.DATA_NAME,
                               'stack_info_x': CompareConst.STACK,
                               'state_x': Const.STATE,
                               'api_origin_name_x': Const.API_ORIGIN_NAME,
                               'requires_grad_x': CompareConst.NPU_REQ_GRAD,
                               'requires_grad_y': CompareConst.BENCH_REQ_GRAD
                               },
                      inplace=True)

        # process summary data
        npu_summary = [CompareConst.NPU_MAX, CompareConst.NPU_MIN, CompareConst.NPU_MEAN, CompareConst.NPU_NORM]
        bench_summary = [CompareConst.BENCH_MAX, CompareConst.BENCH_MIN, CompareConst.BENCH_MEAN,
                         CompareConst.BENCH_NORM]
        # process requires_grad
        result[CompareConst.REQ_GRAD_CONSIST] = result[CompareConst.NPU_REQ_GRAD] == result[CompareConst.BENCH_REQ_GRAD]

        if result.empty:
            result[npu_summary] = pd.DataFrame(columns=npu_summary)
            result[bench_summary] = pd.DataFrame(columns=bench_summary)
        else:
            result[npu_summary] = result['summary_x'].apply(self.set_summary).tolist()
            result[bench_summary] = result['summary_y'].apply(self.set_summary).tolist()

        header.extend([Const.STATE, Const.API_ORIGIN_NAME])
        result_df = pd.DataFrame(columns=header)
        for h in header:
            if h in result.columns:
                result_df[h] = result[h]
        return result_df, header


class CalcStatsDiff:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    @staticmethod
    def type_check(val):
        """
        检查是否为数值或字符串形式的nan, 如果是返回True
        """
        check_series = pd.Series(False, index=val.index)
        val_str = val.astype(str)
        check_series[pd.to_numeric(val_str, errors='coerce').notna() | val_str.str.lower().eq('nan')] = True
        return check_series

    @staticmethod
    def get_number(val):
        return pd.to_numeric(val.astype(str), errors='coerce')

    def calc_summary_diff(self, result_df, cond_no_bench, stats_index: str):
        npu_val = result_df['NPU ' + stats_index]
        bench_val = result_df['Bench ' + stats_index]
        diff_name = stats_index.capitalize() + ' diff'
        rel_err_name = ('norm' if stats_index == 'l2norm' else stats_index).capitalize() + 'RelativeErr'

        # npu、bench中统计量均为数字或nan
        cond_num_nan = self.type_check(npu_val) & self.type_check(bench_val)

        # 如果统计量不是数字或nan，就赋值统计量差异为N/A
        result_df.loc[~cond_num_nan, [diff_name, rel_err_name]] = CompareConst.N_A
        cond_valid_stat = ~cond_no_bench & cond_num_nan  # 有效统计条件：bench_name不是N/A，并且NPU和bench的统计量都是数字或nan
        result_df.loc[cond_valid_stat, diff_name] = self.get_number(npu_val) - self.get_number(bench_val)

        cond_diff_nan = result_df[diff_name].isna()  # 统计量差异是nan
        cond_nan_diff = cond_valid_stat & cond_diff_nan
        result_df.loc[cond_nan_diff, [diff_name, rel_err_name]] = CompareConst.NAN

        cond_not_nan_diff = cond_valid_stat & ~cond_diff_nan
        condition_pt_zero = self.get_number(bench_val) == 0
        result_df.loc[cond_not_nan_diff & condition_pt_zero, rel_err_name] = CompareConst.N_A

        # 相对误差转成百分比字符串
        cond_ref_err = cond_not_nan_diff & ~condition_pt_zero
        result_df.loc[cond_ref_err, rel_err_name] = (
                result_df.loc[cond_ref_err, diff_name] / bench_val[cond_ref_err].astype(float) * 100)
        result_df.loc[cond_ref_err, rel_err_name] = (result_df.loc[cond_ref_err, rel_err_name].abs().astype(str) + '%')

        magnitude = self.get_number(result_df[diff_name]).abs() / (pd.Series(
            np.maximum(self.get_number(npu_val), self.get_number(bench_val))).abs() + CompareConst.EPSILON)
        return magnitude > CompareConst.MAGNITUDE

    def calc_accuracy(self, result_df, header):
        # bench name N/A represents no bench data, err_msg adds "No bench data matched."
        condition_no_bench = result_df[CompareConst.BENCH_NAME] == CompareConst.N_A
        result_df[condition_no_bench] = result_df[condition_no_bench].fillna(CompareConst.N_A)
        result_df.loc[condition_no_bench, CompareConst.ERROR_MESSAGE] = CompareConst.NO_BENCH
        condition_req_grad_consist = result_df[CompareConst.NPU_REQ_GRAD] == result_df[CompareConst.BENCH_REQ_GRAD]

        if self.mode_config.dump_mode == Const.MD5:
            condition_md5_equal = result_df[CompareConst.NPU_MD5] == result_df[CompareConst.BENCH_MD5]
            result_df.loc[condition_md5_equal, CompareConst.RESULT] = CompareConst.PASS
            result_df.loc[~condition_md5_equal & ~condition_no_bench, CompareConst.RESULT] = CompareConst.DIFF
        elif self.mode_config.first_diff_analyze or self.mode_config.dump_mode == Const.SUMMARY:
            warning_list = [
                self.calc_summary_diff(result_df, condition_no_bench, stats_index)
                for stats_index in ['max', 'min', 'mean', 'l2norm']
            ]
            warning_flag = pd.DataFrame(warning_list).any()
            result_df.loc[~condition_no_bench, [CompareConst.RESULT, CompareConst.ERROR_MESSAGE]] = ''
            result_df.loc[warning_flag, CompareConst.RESULT] = CompareConst.WARNING
            result_df.loc[warning_flag, CompareConst.ERROR_MESSAGE] = 'Need double check api accuracy. '
            result_df.loc[~condition_req_grad_consist, CompareConst.ERROR_MESSAGE] += 'Requires_grad inconsistent. '
        else:
            fill_cols = [CompareConst.COSINE, CompareConst.EUC_DIST,
                         CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
                         CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO,
                         CompareConst.ERROR_MESSAGE]
            result_df.loc[~condition_no_bench, fill_cols] = ''  # 默认填充'', df默认省缺值为nan，不便后续处理，容易出现意外情况
            result_df.loc[~condition_no_bench, CompareConst.ACCURACY] = CompareConst.ACCURACY_CHECK_YES
            result_df.loc[~condition_req_grad_consist, CompareConst.ERROR_MESSAGE] = 'Requires_grad inconsistent. '

        return result_df[header]


def setup_comparison(input_param, output_path, **kwargs) -> ComparisonConfig:
    """公共的前置处理逻辑，返回封装后的 ComparisonConfig 对象"""
    try:
        config = ComparisonConfig(
            dump_mode='',
            stack_mode=False,
            fuzzy_match=kwargs.get('fuzzy_match', False),
            data_mapping=kwargs.get('data_mapping', {}),
            suffix=kwargs.get('suffix', ''),
            cell_mapping=kwargs.get('cell_mapping', {}),
            api_mapping=kwargs.get('api_mapping', {}),
            layer_mapping=kwargs.get('layer_mapping', {}),
            first_diff_analyze=kwargs.get('first_diff_analyze', False),
            compared_file_type='',
            is_print_compare_log=kwargs.get('is_print_compare_log', False)
        )

        set_dump_path(input_param)
        config.dump_mode = get_dump_mode(input_param)
        config.compared_file_type = get_file_type(input_param.get("npu_path", None))

        # set stack_mode and set "stack_path" in input_param
        config.stack_mode, input_param = set_stack_json_path(input_param)

        check_configuration_param(config)
        create_directory(output_path)
        check_compare_param(input_param, output_path, config.dump_mode, config.stack_mode)

        return config

    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
