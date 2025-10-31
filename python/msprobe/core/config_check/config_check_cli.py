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

from msprobe.core.config_check.config_checker import ConfigChecker
from msprobe.core.config_check.ckpt_compare.ckpt_comparator import compare_checkpoints
from msprobe.core.common.log import logger


def pack(shell_path, output_path, framework):
    ConfigChecker(shell_path=shell_path, output_zip_path=output_path, fmk=framework)


def compare(bench_zip_path, cmp_zip_path, output_path, framework):
    ConfigChecker.compare(bench_zip_path, cmp_zip_path, output_path, framework)


def _config_checking_parser(parser):
    parser.add_argument('-d', '--dump', nargs='*', help='Collect the train config into a zip file')
    parser.add_argument('-c', '--compare', nargs=2, help='Compare two zip files or checkpoints')
    parser.add_argument('-o', '--output', help='output path, default is ./config_check_result')


def _run_config_checking_command(args):
    if args.dump is not None:
        output_dirpath = args.output if args.output else "./config_check_pack.zip"
        pack(args.dump, output_dirpath, args.framework)
    elif args.compare:
        if args.compare[0].endswith('zip'):
            logger.info('The input paths is zip files, comparing packed config.')
            output_dirpath = args.output if args.output else "./config_check_result"
            compare(args.compare[0], args.compare[1], output_dirpath, args.framework)
        else:
            logger.info('Comparing model checkpoint.')
            output_dirpath = args.output if args.output else "./ckpt_similarity.json"
            compare_checkpoints(args.compare[0], args.compare[1], output_dirpath)

    else:
        logger.error("The param is not correct, you need to give '-d' for dump or '-c' for compare.")
        raise Exception("The param is not correct, you need to give '-d' for dump or '-c' for compare.")
