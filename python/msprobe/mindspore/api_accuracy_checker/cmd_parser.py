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

import argparse
import os

from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory
from msprobe.core.common.utils import Const, MsprobeBaseException


class UniqueDeviceAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        unique_values = set(values)
        if len(values) != len(unique_values):
            parser.error("device id must be unique")
        for device_id in values:
            if not 0 <= device_id <= 4095:
                parser.error(f"the argument 'device_id' must be in range [0, 4095], but got {device_id}")
        setattr(namespace, self.dest, values)


def add_api_accuracy_checker_argument(parser):
    parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", type=str, required=True,
                        help="<Required> The api param tool result file: generate from api param tool, "
                             "a json file.")
    parser.add_argument("-o", "--out_path", dest="out_path", default="./", type=str, required=False,
                        help="<optional> The ut task result out path.")
    parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path", default="", type=str, required=False,
                        help="<optional> the exit csv for continue")
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)


def multi_add_api_accuracy_checker_argument(parser):
    parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", type=str, required=True,
                        help="<Required> The api param tool result file: generate from api param tool, "
                             "a json file.")
    parser.add_argument("-o", "--out_path", dest="out_path", default="./", type=str, required=False,
                        help="<optional> The ut task result out path.")
    parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path", default="", type=str, required=False,
                        help="<optional> the exit csv for continue")
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    #以下属于多线程参数
    parser.add_argument("-d", "--device", dest="device_id", nargs='+', type=int,
                        help="<optional> set device id to run ut, must be unique and in range 0-7",
                        default=[0], required=False, action=UniqueDeviceAction)


def check_args(args):
    args.api_info_file = os.path.abspath(args.api_info_file)
    check_file_or_directory_path(args.api_info_file)

    if args.out_path == "":
        args.out_path = "./"
    args.out_path = os.path.abspath(args.out_path)
    create_directory(args.out_path)

    if args.result_csv_path:
        args.result_csv_path = os.path.abspath(args.result_csv_path)
        check_file_or_directory_path(args.result_csv_path)
