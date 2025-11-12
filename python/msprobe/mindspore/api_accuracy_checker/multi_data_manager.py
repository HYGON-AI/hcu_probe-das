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


import multiprocessing
import os

from msprobe.mindspore.api_accuracy_checker.data_manager import (DataManager, ResultCsvEntry, write_csv_header,
                                                                 get_result_csv_header, get_detail_csv_header,
                                                                 check_csv_header)
from msprobe.mindspore.common.log import logger


class MultiDataManager(DataManager):
    def __init__(self, csv_dir, result_csv_path, shared_is_first_write):
        super().__init__(csv_dir, result_csv_path)

        # 使用共享的 is_first_write 变量来控制表头写入
        self.shared_is_first_write = shared_is_first_write
        # 创建锁对象，确保线程安全
        self.lock = multiprocessing.Lock()

    def save_results(self, api_name_str):
        """保存结果，线程安全操作"""

        with self.lock:  # 确保保存操作不会被多个进程同时进行
            if self.is_first_write and self.shared_is_first_write.value:
                self.shared_is_first_write.value = False
                self.is_first_write = False  # 写入后标记为 False，避免重复写入表头
                # 直接写入表头
                logger.info("Writing CSV headers for the first time.")
                write_csv_header(self.detail_out_path, get_detail_csv_header)
                write_csv_header(self.result_out_path, get_result_csv_header)

            """写入详细输出和结果摘要并清理结果"""
            self.to_detail_csv(self.detail_out_path)
            logger.debug(f"Detailed output for {api_name_str} written to {self.detail_out_path}.")

            self.to_result_csv(self.result_out_path)
            logger.debug(f"Result summary for {api_name_str} written to {self.result_out_path}.")

            # 清理记录，准备下一次调用
            self.clear_results()

    def clear_results(self):
        """清空 self.results 数据，线程安全操作"""
        logger.debug("Clearing results data.")
        self.results.clear()
