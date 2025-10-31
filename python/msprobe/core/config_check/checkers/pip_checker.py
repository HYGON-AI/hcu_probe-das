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
import pandas as pd
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata

from msprobe.core.common.file_utils import load_yaml, create_file_in_zip
from msprobe.core.config_check.checkers.base_checker import BaseChecker
from msprobe.core.config_check.config_checker import register_checker_item
from msprobe.core.config_check.utils.utils import config_checking_print, process_pass_check
from msprobe.core.common.file_utils import FileOpen, save_excel
from msprobe.core.common.const import Const

dirpath = os.path.dirname(__file__)
depend_path = os.path.join(dirpath, "../resource/dependency.yaml")


def load_pip_txt(file_path):
    output_dir = {}
    with FileOpen(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            info_list = line.strip().split("=")
            output_dir[info_list[0]] = "" if len(info_list) != 2 else info_list[1]
    return output_dir


def collect_pip_data():
    result = ""
    packages = metadata.distributions()
    for pkg in packages:
        if pkg.metadata:
            result += f"{pkg.metadata.get('Name')}={pkg.version}\n"
    return result


def compare_pip_data(bench_pip_path, cmp_pip_path, fmk):
    necessary_dependency = load_yaml(depend_path)["dependency"]
    necessary_dependency.append(fmk)
    bench_data = load_pip_txt(bench_pip_path)
    cmp_data = load_pip_txt(cmp_pip_path)
    data = []
    for package in necessary_dependency:
        bench_version = bench_data.get(package)
        cmp_version = cmp_data.get(package)

        if bench_version != cmp_version:
            data.append([package, bench_version if bench_version else 'None',
                         cmp_version if cmp_version else 'None',
                         Const.CONFIG_CHECK_ERROR])

    df = pd.DataFrame(data, columns=PipPackageChecker.result_header)
    return df


@register_checker_item("pip")
class PipPackageChecker(BaseChecker):

    target_name_in_zip = "pip"
    result_header = ['package', 'bench version', 'cmp version', 'level']

    @staticmethod
    def pack(pack_input):
        output_zip_path = pack_input.output_zip_path
        pip_data = collect_pip_data()
        create_file_in_zip(output_zip_path, PipPackageChecker.target_name_in_zip, pip_data)
        config_checking_print(f"add pip info to zip")

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path, fmk):
        bench_pip_path = os.path.join(bench_dir, PipPackageChecker.target_name_in_zip)
        cmp_pip_path = os.path.join(cmp_dir, PipPackageChecker.target_name_in_zip)
        df = compare_pip_data(bench_pip_path, cmp_pip_path, fmk)
        pass_check = process_pass_check(df['level'].values)
        return PipPackageChecker.target_name_in_zip, pass_check, df
