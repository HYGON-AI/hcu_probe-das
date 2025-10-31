# Copyright (c) 2022-2024, Huawei Technologies Co., Ltd.
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

from msprobe.core.advisor.advisor_result import AdvisorResult
from msprobe.core.advisor.advisor_const import AdvisorConst
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.const import Const, CompareConst, FileCheckConst


class Advisor:
    """
    Class for generate advisor
    """

    def __init__(self, input_data, out_path="", suffix=""):
        self.input_data = input_data
        self.out_path = os.path.realpath(out_path)
        self.file_type = None
        self.suffix = suffix

    @staticmethod
    def deterministic_advisor(message, node_name):
        for api_name in AdvisorConst.NEED_DETERMINISTIC_API:
            if api_name in node_name:
                return AdvisorConst.DETERMINISTIC_SUGGEST
        return message

    @staticmethod
    def batch_norm_advisor(message, node_name):
        if AdvisorConst.FUNC_BATCH_NORM in node_name and AdvisorConst.FORWARD_INPUT_1 in node_name:
            message = AdvisorConst.BATCH_NORM_SUGGEST
        return message

    def analyze_unmatched(self, analyze_data):
        if self.file_type == Const.ALL:
            accuracy_unmatched = analyze_data[
                analyze_data[CompareConst.ACCURACY] == CompareConst.ACCURACY_CHECK_UNMATCH]
        else:
            accuracy_unmatched = analyze_data[(analyze_data[CompareConst.NPU_SHAPE] == CompareConst.NAN) |
                                              (analyze_data[CompareConst.BENCH_SHAPE] == CompareConst.NAN)]
        num_unmatch = len(accuracy_unmatched)
        if num_unmatch != 0:
            for i in range(len(accuracy_unmatched)):
                item = accuracy_unmatched.iloc[i]
                logger.warning("The tensor name matches but the shape or dtype does not match: {}"
                            .format(item[CompareConst.NPU_NAME]))

    def gen_advisor_result(self, pd_data):
        try:
            first_failing_data = pd_data.iloc[0]
        except IndexError as e:
            err_msg = "index out of bounds error occurs, pd_data is empty, please check!"
            logger.error(err_msg)
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e
        node_name = first_failing_data[CompareConst.NPU_NAME]
        index = first_failing_data['index']
        message = self.gen_advisor_message(node_name)
        logger.warning("Find %s accuracy not reached, the line is %s" % (node_name, index))
        result = AdvisorResult(node_name, index, message)
        return result

    def gen_advisor_message(self, node_name):
        if AdvisorConst.FORWARD in node_name:
            if AdvisorConst.INPUT in node_name:
                message = AdvisorConst.FORWARD_INPUT_SUGGEST
            else:
                message = AdvisorConst.FORWARD_OUTPUT_SUGGEST
                message = self.deterministic_advisor(message, node_name)
        else:
            if AdvisorConst.INPUT in node_name:
                message = AdvisorConst.BACKWARD_INPUT_SUGGEST
            else:
                message = AdvisorConst.BACKWARD_OUTPUT_SUGGEST
                message = self.deterministic_advisor(message, node_name)
        message = self.batch_norm_advisor(message, node_name)
        return message

    def analysis(self):
        self._check_path_valid()
        analyze_data = self._parse_input_data()
        logger.info("Start analyzing the comparison result: %s" % self.file_type)
        self.analyze_unmatched(analyze_data)
        if self.file_type == Const.ALL:
            failing_data = analyze_data[analyze_data[CompareConst.ACCURACY] == CompareConst.ACCURACY_CHECK_NO]
        elif self.file_type == Const.MD5:
            failing_data = analyze_data[analyze_data[CompareConst.RESULT] == CompareConst.DIFF]
        elif self.file_type == Const.SUMMARY:
            failing_data = analyze_data[analyze_data[CompareConst.RESULT] == CompareConst.WARNING]
        if failing_data.empty:
            logger.info("All data from api input/output accuracy reached")
            result = AdvisorResult(AdvisorConst.NO_ERROR_API, AdvisorConst.NO_ERROR_API, AdvisorConst.NO_ERR_SUGGEST)
        else:
            result = self.gen_advisor_result(failing_data)
        message_list = result.print_advisor_log()
        result.gen_summary_file(self.out_path, message_list, suffix=self.suffix)

    def _parse_input_data(self):
        data_columns = self.input_data.columns.values
        if {CompareConst.ACCURACY, CompareConst.NPU_NAME}.issubset(data_columns):
            self.file_type = Const.ALL
        elif {CompareConst.RESULT, CompareConst.NPU_MD5}.issubset(data_columns):
            self.file_type = Const.MD5
        elif {CompareConst.MAX_DIFF, CompareConst.RESULT}.issubset(data_columns):
            self.file_type = Const.SUMMARY
        else:
            logger.error('Compare result does not meet the required conditions.')
            raise CompareException(CompareException.INVALID_DATA_ERROR)
        df = self.input_data.reset_index()
        return df

    def _check_path_valid(self):
        out_path_checker = FileChecker(self.out_path, FileCheckConst.DIR, FileCheckConst.WRITE_ABLE)
        out_path_checker.common_check()
