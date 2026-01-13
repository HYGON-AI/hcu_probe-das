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

from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker

from msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker import MultiApiAccuracyChecker

from msprobe.mindspore.api_accuracy_checker.cmd_parser import check_args


def api_checker_main(args):
    check_args(args)
    api_accuracy_checker = ApiAccuracyChecker(args)
    api_accuracy_checker.parse(args.api_info_file)
    api_accuracy_checker.run_and_compare()


def mul_api_checker_main(args):
    check_args(args)
    api_accuracy_checker = MultiApiAccuracyChecker(args)
    api_accuracy_checker.parse(args.api_info_file)
    api_accuracy_checker.run_and_compare()
