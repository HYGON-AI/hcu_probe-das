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

import concurrent
import copy
import csv
import os
import threading
import traceback
from datetime import datetime, timezone, timedelta

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.file_utils import change_mode, FileOpen, save_json, check_path_before_create
from msprobe.core.common.log import logger

lock = threading.Lock()


class DataWriter:

    def __init__(self) -> None:
        self.dump_file_path = None
        self.stack_file_path = None
        self.construct_file_path = None
        self.free_benchmark_file_path = None
        self.dump_tensor_data_dir = None
        self.debug_file_path = None
        self.dump_error_info_path = None
        self.flush_size = 1000
        self.md5_flush_size = 5000
        self.larger_flush_size = 20000
        self.cache_data = {}
        self.cache_stack = {}
        self.cache_construct = {}
        self.cache_debug = {}
        self.stat_stack_list = []
        self._error_log_initialized = False
        self._cache_logged_error_types = set()
        self.crc32_stack_list = []
        self.data_updated = False

    @staticmethod
    def write_data_to_csv(result: list, result_header: tuple, file_path: str):
        if not result:
            return
        is_exists = os.path.exists(file_path)
        append = "a+" if is_exists else "w+"
        with FileOpen(file_path, append) as csv_file:
            spawn_writer = csv.writer(csv_file)
            if not is_exists:
                spawn_writer.writerow(result_header)
            spawn_writer.writerows([result, ])
        is_new_file = not is_exists
        if is_new_file:
            change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)

    @recursion_depth_decorator("JsonWriter: DataWriter._replace_crc32_placeholders")
    def _replace_crc32_placeholders(self, data, crc32_results):
        """
        遍历 JSON 结构，将所有 md5_index 占位符替换成真实的 CRC32
        """
        if isinstance(data, dict):
            for k, v in list(data.items()):
                if k == Const.MD5_INDEX and isinstance(v, int):
                    idx = v
                    # 防越界
                    crc = crc32_results[idx] if idx < len(crc32_results) else None
                    # 删除占位符，改成真实字段
                    del data[k]
                    data[Const.MD5] = crc
                else:
                    self._replace_crc32_placeholders(v, crc32_results)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._replace_crc32_placeholders(item, crc32_results)

    @recursion_depth_decorator("JsonWriter: DataWriter._replace_stat_placeholders")
    def _replace_stat_placeholders(self, data, stat_result):
        if isinstance(data, dict):
            keys = list(data.keys())  # 获取当前所有键
            for key in keys:  # 递归所有变量
                value = data[key]
                if key == Const.TENSOR_STAT_INDEX and isinstance(value, int):
                    if value >= 0:
                        idx = value
                    else:
                        return
                    stat_values = stat_result[idx] if idx < len(stat_result) else [None] * 4

                    new_entries = {
                        Const.TYPE: data["type"],
                        Const.DTYPE: data["dtype"],
                        Const.SHAPE: data["shape"],
                        Const.MAX: stat_values[0],
                        Const.MIN: stat_values[1],
                        Const.MEAN: stat_values[2],
                        Const.NORM: stat_values[3],
                    }
                    del data[key]

                    # 重构字典顺序
                    updated_dict = {}
                    # 通过插入排序后字段保证字段写入json的有序
                    updated_dict.update(new_entries)
                    # 遍历原字典其他字段（排除已删除的tensor_stat_index）
                    for k in data:
                        if k not in new_entries:
                            updated_dict[k] = data[k]
                    data.clear()
                    data.update(updated_dict)
                else:
                    self._replace_stat_placeholders(value, stat_result)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._replace_stat_placeholders(item, stat_result)

    def reset_cache(self):
        self.cache_data = {}
        self.cache_stack = {}
        self.cache_construct = {}
        self.cache_debug = {}
        self._cache_logged_error_types = set()

    def append_crc32_to_buffer(self, future: concurrent.futures.Future) -> int:
        """
        把一个计算 CRC32 的 Future 放入队列，返回占位符索引
        """
        idx = len(self.crc32_stack_list)
        self.crc32_stack_list.append(future)
        return idx

    def flush_crc32_stack(self):
        """
        等待所有 CRC32 计算完成，返回结果列表
        """
        if not self.crc32_stack_list:
            return []
        results = [f.result() for f in self.crc32_stack_list]
        self.crc32_stack_list = []
        return results

    def initialize_json_file(self, **kwargs):
        if kwargs["level"] == Const.LEVEL_DEBUG and not self.cache_debug:
            # debug level case only create debug.json
            debug_dict = copy.deepcopy(kwargs)
            debug_dict.update({"dump_data_dir": self.dump_tensor_data_dir, Const.DATA: {}})
            self.cache_debug = debug_dict
            save_json(self.debug_file_path, self.cache_debug, indent=1)
            return
        if not self.cache_data:
            kwargs.update({"dump_data_dir": self.dump_tensor_data_dir, Const.DATA: {}})
            self.cache_data = kwargs
            save_json(self.dump_file_path, self.cache_data, indent=1)
        if not self.cache_stack:
            save_json(self.stack_file_path, self.cache_stack, indent=1)
        if not self.cache_construct:
            save_json(self.construct_file_path, self.cache_construct, indent=1)

    def update_dump_paths(self, dump_path_aggregation):
        self.dump_file_path = dump_path_aggregation.dump_file_path
        self.stack_file_path = dump_path_aggregation.stack_file_path
        self.construct_file_path = dump_path_aggregation.construct_file_path
        self.dump_tensor_data_dir = dump_path_aggregation.dump_tensor_data_dir
        self.free_benchmark_file_path = dump_path_aggregation.free_benchmark_file_path
        self.debug_file_path = dump_path_aggregation.debug_file_path
        self.dump_error_info_path = dump_path_aggregation.dump_error_info_path

    def flush_data_periodically(self):
        dump_data = self.cache_data.get(Const.DATA)

        if not dump_data or not isinstance(dump_data, dict):
            return

        length = len(dump_data)

        # 1) 先取到 config（如果没有，就拿 None）
        cfg = getattr(self, "config", None)
        # 2) 再取 summary_mode（如果 cfg 是 None 或者没 summary_mode，就拿 None）
        summary_mode = getattr(cfg, "summary_mode", None)

        if summary_mode == Const.MD5:
            threshold = self.md5_flush_size
        else:
            threshold = self.flush_size if length < self.larger_flush_size else self.larger_flush_size

        if length % threshold == 0:
            self.write_json()

    def write_error_log(self, message: str, error_type: str):
        """
        写错误日志：
          - 第一次调用时以 'w' 模式清空文件，之后都用 'a' 模式追加
          - 添加时间戳
          - 在 message 后写入当前的调用栈（方便追踪日志来源）
        """
        # 如果同类型错误已经记录过，跳过
        if error_type in self._cache_logged_error_types:
            return
        # 否则添加到已记录集合，并继续写日志
        self._cache_logged_error_types.add(error_type)

        try:
            mode = "w" if not self._error_log_initialized else "a"
            self._error_log_initialized = True

            check_path_before_create(self.dump_error_info_path)

            with FileOpen(self.dump_error_info_path, mode) as f:
                cst_timezone = timezone(timedelta(hours=8), name="CST")
                timestamp = datetime.now(cst_timezone).strftime("%Y-%m-%d %H:%M:%S %z")
                f.write(f"[{timestamp}] {message}\n")
                f.write("Call stack (most recent call last):\n")

                f.write("".join(traceback.format_stack()[:-1]))  # 去掉自己这一层
                f.write("\n")
        except Exception as e:
            # 如果连写日志都失败了，就打印到 stderr
            logger.warning(f"[FallbackError] Failed to write error log: {e}")

    def update_data(self, new_data):
        with lock:
            if not isinstance(new_data, dict) or len(new_data.keys()) != 1:
                logger.warning(f"The data info({new_data}) should be a dict with only one outer key.")
                return
            dump_data = self.cache_data.get(Const.DATA)
            if not isinstance(dump_data, dict):
                logger.warning(f"The dump data({dump_data}) should be a dict.")
                return

            self.data_updated = True
            key = next(iter(new_data.keys()))
            if key in dump_data:
                dump_data.get(key).update(new_data.get(key))
            else:
                dump_data.update(new_data)

    def update_stack(self, name, stack_data):
        with lock:
            self.data_updated = True
            api_list = self.cache_stack.get(stack_data)
            if api_list is None:
                self.cache_stack.update({stack_data: [name]})
            else:
                api_list.append(name)

    def update_construct(self, new_data):
        with lock:
            self.data_updated = True
            self.cache_construct.update(new_data)

    def update_debug(self, new_data):
        with lock:
            self.data_updated = True
            self.cache_debug['data'].update(new_data)

    def write_data_json(self, file_path):
        logger.info(f"dump.json is at {os.path.dirname(os.path.dirname(file_path))}. ")
        save_json(file_path, self.cache_data, indent=1)

    def write_stack_info_json(self, file_path):
        num, new_cache_stack = 0, {}
        for key, value in self.cache_stack.items():
            new_cache_stack[num] = [value, key]
            num += 1
        save_json(file_path, new_cache_stack, indent=1)

    def write_construct_info_json(self, file_path):
        save_json(file_path, self.cache_construct, indent=1)

    def write_debug_info_json(self, file_path):
        save_json(file_path, self.cache_debug, indent=1)

    def append_stat_to_buffer(self, stat_vector):
        """
        直接使用 Python list 存储 stat_vector,
        将 stat_vector 存入 self.stat_stack_list 的方式
        """
        self.stat_stack_list.append(stat_vector)
        return len(self.stat_stack_list) - 1

    def get_buffer_values_max(self, index):
        if 0 <= index < len(self.stat_stack_list) and len(self.stat_stack_list[index]) >= 1:
            return self.stat_stack_list[index][0]
        else:
            logger.warning(f"stat_stack_list[{index}] The internal data is incomplete,"
                           f" and the maximum value cannot be obtained.")
            return None

    def get_buffer_values_min(self, index):
        if 0 <= index < len(self.stat_stack_list) and len(self.stat_stack_list[index]) >= 1:
            return self.stat_stack_list[index][1]
        else:
            logger.warning(f"stat_stack_list[{index}] Internal data is incomplete"
                           f" and minimum values cannot be obtained.")
            return None

    def flush_stat_stack(self):
        """
        在 flush 阶段，将所有存储的统计值从设备搬到 CPU，
        这里返回一个列表，每个元素是 [Max, Min, Mean, Norm] 的数值列表
        """
        if not self.stat_stack_list:
            return []
        result = [
            [
                x.item() if hasattr(x, "item") else x
                for x in stat_values
            ]
            for stat_values in self.stat_stack_list
        ]
        self.stat_stack_list = []
        return result

    def write_json(self):
        with lock:
            # 在写 JSON 前，统一获取统计值
            stat_result = self.flush_stat_stack()
            # 遍历 cache_data，将占位符替换为最终统计值
            if stat_result:
                self.data_updated = True
                self._replace_stat_placeholders(self.cache_data, stat_result)
                if self.cache_debug:
                    self._replace_stat_placeholders(self.cache_debug, stat_result)

            crc32_result = self.flush_crc32_stack()
            if crc32_result:
                self.data_updated = True
                self._replace_crc32_placeholders(self.cache_data, crc32_result)
                if self.cache_debug:
                    self._replace_crc32_placeholders(self.cache_debug, crc32_result)

            if not self.data_updated:
                return

            if self.cache_data:
                self.write_data_json(self.dump_file_path)
            if self.cache_stack:
                self.write_stack_info_json(self.stack_file_path)
            if self.cache_construct:
                self.write_construct_info_json(self.construct_file_path)
            if self.cache_debug:
                self.write_debug_info_json(self.debug_file_path)
            self.data_updated = False
