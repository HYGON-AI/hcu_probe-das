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
import shutil

import pandas as pd

from msprobe.core.common.file_utils import save_excel, split_zip_file_path, \
    create_directory, extract_zip
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.config_check.checkers.base_checker import PackInput
from msprobe.core.config_check.utils.utils import config_checking_print
from msprobe.core.common.const import Const


class ConfigChecker:
    checkers = {}
    pre_forward_fun_list = []
    result_filename = "result.xlsx"
    result_header = ["filename", "pass_check"]
    step = 0

    def __init__(self, model=None, shell_path=None, output_zip_path="./config_check_pack.zip", fmk="pytorch"):
        FmkAdp.set_fmk(fmk)
        self.pack_input = PackInput(output_zip_path, model, shell_path)
        file_path, file_name = split_zip_file_path(self.pack_input.output_zip_path)
        if not os.path.exists(file_path):
            create_directory(file_path)
        self.pack()

    @staticmethod
    def compare(bench_zip_path, cmp_zip_path, output_path, fmk=Const.PT_FRAMEWORK):
        create_directory(output_path)
        bench_dir = os.path.join(output_path, "bench")
        cmp_dir = os.path.join(output_path, "cmp")
        extract_zip(bench_zip_path, bench_dir)
        config_checking_print(f"extract zip file {bench_zip_path} to {bench_dir}")
        extract_zip(cmp_zip_path, cmp_dir)
        config_checking_print(f"extract zip file {cmp_zip_path} to {cmp_dir}")

        result = []
        summary_result = []
        for checker in ConfigChecker.checkers.values():
            checker_name, pass_check, df = checker.compare_ex(bench_dir, cmp_dir, output_path, fmk)
            if checker_name:
                summary_result.append([checker_name, pass_check])
            if df is not None:
                result.append((df, checker_name))
        summary_result_df = pd.DataFrame(summary_result, columns=ConfigChecker.result_header)
        result.insert(0, (summary_result_df, "summary"))
        save_excel(os.path.join(output_path, ConfigChecker.result_filename), result)
        config_checking_print(f"config checking result save to {os.path.realpath(output_path)}")

    @staticmethod
    def apply_patches(fmk=Const.PT_FRAMEWORK):
        for checker in ConfigChecker.checkers.values():
            checker.apply_patches(fmk)

    def pack(self):
        config_checking_print(f"pack result zip path {os.path.realpath(self.pack_input.output_zip_path)}")

        def hook(model, args, kwargs):
            for collect_func in self.pre_forward_fun_list:
                collect_func(model, args, kwargs, ConfigChecker.step)
            ConfigChecker.step += 1

        if self.pack_input.model:
            FmkAdp.register_forward_pre_hook(self.pack_input.model, hook, with_kwargs=True)
        for checker in ConfigChecker.checkers.values():
            if checker.input_needed and not getattr(self.pack_input, checker.input_needed):
                continue
            if FmkAdp.is_initialized() and FmkAdp.get_rank() != 0 and not checker.multi_rank:
                continue
            checker.pack(self.pack_input)


def register_checker_item(key, cls=None):
    if cls is None:
        # 无参数时，返回装饰器函数
        return lambda cls: register_checker_item(key, cls)
    ConfigChecker.checkers[key] = cls
    return cls


def register_pre_forward_fun_list(func):
    ConfigChecker.pre_forward_fun_list.append(func)
