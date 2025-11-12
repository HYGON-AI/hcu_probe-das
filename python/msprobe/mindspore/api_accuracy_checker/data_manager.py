# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import os
import csv

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.file_utils import FileOpen, create_directory, write_csv, read_csv
from msprobe.core.common.utils import add_time_as_suffix, MsprobeBaseException
from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.const import MsCompareConst


class ResultCsvEntry:
    def __init__(self) -> None:
        self.forward_pass_status = None
        self.backward_pass_status = None
        self.forward_err_msg = ""
        self.backward_err_msg = ""
        self.overall_err_msg = None


def write_csv_header(csv_path, header_func):
    """如果是第一次写入，则写入 CSV 表头"""
    header = header_func()  # 获取表头
    logger.debug(f"Writing CSV header: {header}")
    write_csv([header], csv_path, mode="a+")


def get_result_csv_header():
    """获取结果 CSV 文件的表头"""
    return [
        MsCompareConst.DETAIL_CSV_API_NAME,
        MsCompareConst.RESULT_CSV_FORWARD_TEST_SUCCESS,
        MsCompareConst.RESULT_CSV_BACKWARD_TEST_SUCCESS,
        MsCompareConst.DETAIL_CSV_MESSAGE,
    ]


def get_detail_csv_header():
    """获取详细 CSV 文件的表头"""
    detail_csv_header_basic_info = [
        MsCompareConst.DETAIL_CSV_API_NAME,
        MsCompareConst.DETAIL_CSV_BENCH_DTYPE,
        MsCompareConst.DETAIL_CSV_TESTED_DTYPE,
        MsCompareConst.DETAIL_CSV_SHAPE,
    ]
    detail_csv_header_compare_result = list(compare_algorithms.keys())
    detail_csv_header_status = [
        MsCompareConst.DETAIL_CSV_PASS_STATUS,
        MsCompareConst.DETAIL_CSV_MESSAGE,
    ]
    return detail_csv_header_basic_info + detail_csv_header_compare_result + detail_csv_header_status


def check_csv_header(headers, required_constants, csv_path):
    """校验 CSV 文件表头是否包含所有必需的常量"""
    missing_constants = [const for const in required_constants if not any(const in header for header in headers)]

    if missing_constants:
        raise MsprobeBaseException(
            MsprobeBaseException.MISSING_HEADER_ERROR,
            f"{csv_path} 缺少以下必需的表头字段: {missing_constants}"
        )
    return True


class DataManager:
    def __init__(self, csv_dir, result_csv_path):
        self.results = {}
        self.results_exception_skip = {}
        self.is_first_write = True  # 标记用于添加表头
        self.csv_dir = csv_dir
        self.api_names_set = set()  # 存储已经出现的 API 名称的集合
        # 如果传入了 result_csv_path，则启用断点续检
        if result_csv_path:
            self.resume_from_last_csv(result_csv_path)
            self.initialize_api_names_set(result_csv_path)
        else:
            # 默认情况下，设置输出路径为空，等待首次写入时初始化
            self.result_out_path = os.path.join(self.csv_dir, add_time_as_suffix(MsCompareConst.RESULT_CSV_FILE_NAME))
            self.detail_out_path = os.path.join(
                self.csv_dir,
                os.path.basename(self.result_out_path).replace("result", "details")
            )

            if self.detail_out_path and os.path.exists(self.detail_out_path):
                check_file_or_directory_path(self.detail_out_path)

            if self.result_out_path and os.path.exists(self.result_out_path):
                check_file_or_directory_path(self.result_out_path)

    def initialize_api_names_set(self, result_csv_path):
        """读取现有的 CSV 文件并存储已经出现的 API 名称到集合中"""
        # 使用新的 read_csv 函数读取数据
        csv_data = read_csv(result_csv_path, as_pd=False)

        # 读取标题行
        headers = csv_data[0] if csv_data else []  # 如果文件为空，则 headers 会为空

        # 使用提取的表头校验函数
        if check_csv_header(headers, get_result_csv_header(), result_csv_path):

            # 获取 "API Name" 列的索引
            api_name_index = None
            for i, header in enumerate(headers):
                if MsCompareConst.DETAIL_CSV_API_NAME in header:  # CSV 文件的标题行包含了字节顺序标记,所以使用通过包含方式来查找
                    api_name_index = i
                    break

            if api_name_index is None:
                logger.warning(f"{result_csv_path} No column contains 'API Name'.")
                return

            # 读取每一行的 API 名称
            for row in csv_data[1:]:  # 跳过标题行，从第二行开始
                if row and len(row) > api_name_index:
                    api_name = row[api_name_index]
                    if api_name:
                        self.api_names_set.add(api_name)

            logger.debug(f"Initialized API names set from existing CSV: {self.api_names_set}")

    def is_unique_api(self, api_name):
        """检查 API 名称是否唯一，如果已经存在则返回 False，否则加入集合并返回 True"""
        if api_name in self.api_names_set:
            return False
        self.api_names_set.add(api_name)
        return True

    def resume_from_last_csv(self, result_csv_path):
        """从上次运行的 result_csv_path 恢复断点"""
        # 获取上次的目录路径
        last_dir = os.path.dirname(result_csv_path)

        # 设置当前目录和输出路径，确保在首次写入时使用
        self.csv_dir = last_dir
        self.detail_out_path = os.path.join(last_dir, os.path.basename(result_csv_path).replace("result", "details"))
        if self.detail_out_path and os.path.exists(self.detail_out_path):
            check_file_or_directory_path(self.detail_out_path)
        self.result_out_path = result_csv_path
        self.is_first_write = False

    def save_results(self, api_name_str):
        if self.is_first_write:
            # 直接写入表头
            logger.info("Writing CSV headers for the first time.")
            write_csv_header(self.detail_out_path, get_detail_csv_header)
            write_csv_header(self.result_out_path, get_result_csv_header)
            self.is_first_write = False  # 写入后标记为 False，避免重复写入表头

        """写入详细输出和结果摘要并清理结果"""
        logger.debug("Starting to write detailed output to CSV.")
        self.to_detail_csv(self.detail_out_path)
        logger.debug(f"Detailed output for {api_name_str} written to {self.detail_out_path}.")

        logger.debug("Starting to write result summary to CSV.")
        self.to_result_csv(self.result_out_path)
        logger.debug(f"Result summary for {api_name_str} written to {self.result_out_path}.")

        # 清理记录，准备下一次调用
        self.clear_results()

    def record(self, output_list):
        if output_list is None:
            return
        for output in output_list:
            api_real_name, forward_or_backward, basic_info, compare_result_dict = output
            key = (api_real_name, forward_or_backward)
            if key not in self.results:
                self.results[key] = []
            self.results[key].append((basic_info, compare_result_dict))
            logger.debug(f"Updated self.results for key {key}: {self.results[key]}")
        logger.debug(f"Complete self.results after recording: {self.results}")

    def record_exception_skip(self, api_name, forward_or_backward, err_msg):
        '''
            record exception_skip information into self.record_exception_skip.
            self.record_exception_skip: dict{str: dict{"forward": str/None, "backward": str/None}}
            string in key is api_name, string in value is err_msg
        '''
        if api_name not in self.results_exception_skip:
            self.results_exception_skip[api_name] = {Const.FORWARD: None, Const.BACKWARD: None}
        self.results_exception_skip[api_name][forward_or_backward] = err_msg

    def clear_results(self):
        """清空 self.results 数据"""
        logger.debug("Clearing self.results data.")
        self.results.clear()
        self.results_exception_skip.clear()

    def to_detail_csv(self, csv_path):
        logger.debug("Preparing detail CSV headers and rows.")
        detail_csv = []

        detail_csv_header_compare_result = list(compare_algorithms.keys())

        for _, results in self.results.items():
            for res in results:
                basic_info, compare_result_dict = res
                csv_row_basic_info = [
                    basic_info.api_name,
                    basic_info.bench_dtype,
                    basic_info.tested_dtype,
                    basic_info.shape
                ]
                csv_row_compare_result = [
                    compare_result_dict.get(algorithm_name).compare_value
                    for algorithm_name in detail_csv_header_compare_result
                ]
                csv_row_status = [basic_info.status, basic_info.err_msg]
                csv_row = csv_row_basic_info + csv_row_compare_result + csv_row_status
                detail_csv.append(csv_row)
                logger.debug(f"Detail CSV row added: {csv_row}")

        logger.debug(f"Writing detail CSV to {csv_path}.")
        write_csv(detail_csv, csv_path, mode="a+")
        logger.debug(f"Detail CSV written successfully to {csv_path}.")

    def to_result_csv(self, csv_path):
        '''
            depend on both self.results and self.results_exception_skip
        '''
        logger.debug("Preparing result CSV data.")
        result_csv = []

        result_csv_dict = {}
        for key, results in self.results.items():
            api_real_name, forward_or_backward = key
            pass_status = CompareConst.PASS
            overall_err_msg = ""

            for res in results:
                basic_info, _ = res
                if basic_info.status != CompareConst.PASS:
                    pass_status = CompareConst.ERROR
                overall_err_msg += basic_info.err_msg

            overall_err_msg = "" if pass_status == CompareConst.PASS else overall_err_msg

            if api_real_name not in result_csv_dict:
                result_csv_dict[api_real_name] = ResultCsvEntry()
            if forward_or_backward == Const.FORWARD:
                result_csv_dict[api_real_name].forward_pass_status = pass_status
                result_csv_dict[api_real_name].forward_err_msg = overall_err_msg
            else:
                result_csv_dict[api_real_name].backward_pass_status = pass_status
                result_csv_dict[api_real_name].backward_err_msg = overall_err_msg

        for api_name, entry in result_csv_dict.items():
            overall_err_msg = "" if (entry.forward_pass_status == CompareConst.PASS and
                                     entry.backward_pass_status == CompareConst.PASS) else \
                entry.forward_err_msg + entry.backward_err_msg
            row = [
                api_name,
                entry.forward_pass_status,
                entry.backward_pass_status,
                overall_err_msg
            ]
            # change row if this api has exception_skip information
            if api_name in self.results_exception_skip:
                if self.results_exception_skip[api_name][Const.FORWARD] is not None:
                    row[1] = CompareConst.SKIP
                    row[-1] += self.results_exception_skip[api_name][Const.FORWARD]
                if self.results_exception_skip[api_name][Const.BACKWARD] is not None:
                    row[2] = CompareConst.SKIP
                    row[-1] += self.results_exception_skip[api_name][Const.BACKWARD]
                del self.results_exception_skip[api_name]
            result_csv.append(row)
            logger.debug(f"Result CSV row added: {row}")
        for api_name in self.results_exception_skip:
            current_exception_skip = self.results_exception_skip[api_name]
            forward_status = None
            backward_status = None
            err_msg = ""
            if current_exception_skip[Const.FORWARD] is not None:
                forward_status = CompareConst.SKIP
                err_msg += current_exception_skip[Const.FORWARD]
            if current_exception_skip[Const.BACKWARD] is not None:
                backward_status = CompareConst.SKIP
                err_msg += current_exception_skip[Const.BACKWARD]
            row = [api_name, forward_status, backward_status, err_msg]
            result_csv.append(row)

        write_csv(result_csv, csv_path, mode="a+")
        logger.debug(f"Result CSV written successfully to {csv_path}.")

        # 设置标记为 False，防止后续重复添加表头
        self.is_first_write = False
