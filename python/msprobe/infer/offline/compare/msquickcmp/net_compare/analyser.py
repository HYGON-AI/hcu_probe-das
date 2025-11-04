# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import math
from collections import namedtuple

from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.util import load_file_to_read_common_check
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE

INVALID_ROW_VALUES = {
    "OpType": ["TransData"],
    "GroundTruth": ["*"],
    "DataType": ["NaN"],
}

MONITOR_THRESHOLD = {
    "CosineSimilarity": 0.99,
    "RelativeEuclideanDistance": 0.05,
    "KullbackLeiblerDivergence": 0.005,
    "RootMeanSquareError": 1.0,
    "MeanRelativeError": 1.0,
}

REVERSE_MONITORS = ["CosineSimilarity"]
PRINT_COLUMNS = ["Index", "OpType", "NPUDump", "GroundTruth"]

_STRATEGY_NAMES = ["FIRST_INVALID_OVERALL", "FIRST_INVALID_EACH", "ALL_INVALID"]
STRATEGIES = namedtuple("STRATEGIES", _STRATEGY_NAMES)(*_STRATEGY_NAMES)


def type_to_str(value_type):
    return " or ".join([ii.__name__ for ii in value_type]) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value", additional_check_func=None, additional_msg=None):
    if not isinstance(value, value_type):
        raise TypeError(
            "{} needs to be {}, but got {}.".format(param_name, type_to_str(value_type), type(value).__name__)
        )


def check_element_type(value, element_type, param_name="value"):
    if len(value) == 0:
        raise ValueError("{} is empty".format(param_name))
    if not all([isinstance(ii, element_type) for ii in value]):
        raise ValueError("Elements in {} need to be all {}.".format(param_name, type_to_str(element_type)))


class Analyser:
    def __init__(self, csv_path):
        """
        Analyser for csv output compare result.
        Args:
          csv_path: str value for csv file path, or the folder name containing a single csv result file.

        Examples:
        >>> from compare import analyser
        >>> aa = analyser.Analyser({csv_path})
        >>> _ = aa(strategy=analyser.STRATEGIES.FIRST_INVALID_OVERALL)
        """
        check_type(csv_path, str, param_name="csv_path")
        if os.path.isdir(csv_path):
            csv_path = self._get_single_csv_in_folder(csv_path)
        if not csv_path.endswith(".csv"):
            raise ValueError(f"csv_path={csv_path} not endswith csv")
        if not os.path.exists(csv_path):
            raise IOError(f"csv_path={csv_path} not exists")
        utils.logger.info(f"Analyser init parameter csv_path={csv_path}")

        self.csv_path = csv_path
        self.monitor_threshold = {}
        for monitor, threshold in MONITOR_THRESHOLD.items():
            self.monitor_threshold[monitor] = (1 - threshold) if monitor in REVERSE_MONITORS else threshold

        self._strategy_func_dict = {
            STRATEGIES.FIRST_INVALID_OVERALL: self._first_invalid_overall,
            STRATEGIES.FIRST_INVALID_EACH: self._first_invalid_each,
            STRATEGIES.ALL_INVALID: self._all_invalid,
        }

    def __call__(self, strategy=STRATEGIES.FIRST_INVALID_OVERALL, max_column_len=30):
        """
        Run analyser and print result info.
        Args:
          strategy: one of analyser.STRATEGIES.
              - STRATEGIES.FIRST_INVALID_OVERALL means printing the first operator,
                that any monitor value is not within threshold.
              - STRATEGIES.FIRST_INVALID_OVERALL means printing the first operators
                whose value is not within threshold for each monitor.
              - STRATEGIES.ALL_INVALID means printing all the operators
                whose value is not within threshold for each monitor.
          max_column_len: int value for each column max print length.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy Should be one of {list(STRATEGIES)}")
        try:
            self.csv_path = load_file_to_read_common_check(self.csv_path)
            with ms_open(self.csv_path, "r", max_size=TENSOR_MAX_SIZE) as csv_file:
                csv_rows = [row for row in csv.DictReader(csv_file) if self._is_valid_row(row)]
        except IOError as csv_file_except:
            utils.logger.error('Failed to open"' + self.csv_path + '", ' + str(csv_file_except))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR) from csv_file_except
        utils.logger.info(f"Analyser call parameter strategy={strategy}, max_column_len={max_column_len}")

        invalid_rows, invalid_monitors = self._strategy_func_dict.get(strategy, self._first_invalid_overall)(csv_rows)

        self._show_result(invalid_rows, invalid_monitors, max_column_len=max_column_len)
        return invalid_rows, invalid_monitors

    @staticmethod
    def _get_single_csv_in_folder(csv_path):
        for file_name in os.listdir(csv_path):
            if file_name.endswith('.csv'):
                return os.path.join(csv_path, file_name)
        raise IOError(f"None csv file exists in folder {csv_path}")

    @staticmethod
    def _show_result(invalid_rows, invalid_monitors, max_column_len=30):
        if len(invalid_rows) == 0:
            utils.logger.info("None operator with accuracy issue reported")
            return

        utils.logger.info("Operators may lead to inaccuracy:")
        results = {}
        for row, monitors in zip(invalid_rows, invalid_monitors):
            for monitor in monitors:
                results.setdefault("Monitor", []).append(monitor)
                results.setdefault("Value", []).append("{:.6g}".format(float(row[monitor])))

            for column in PRINT_COLUMNS:
                results.setdefault(column, []).extend([row[column]] * len(monitors))
        print_in_markdown_table(results, max_column_len=max_column_len)

    @staticmethod
    def _get_monitors_exceeding_threshold(row, monitor_threshold):
        invalid_monitors = []
        for monitor, threshold in monitor_threshold.items():
            row_value = float(row.get(monitor, "NaN"))
            if math.isnan(row_value) or math.isinf(row_value):
                continue

            if monitor in REVERSE_MONITORS:
                row_value = 1 - row_value
            if row_value > threshold:
                invalid_monitors.append(monitor)
        return invalid_monitors

    @staticmethod
    def _is_valid_row(row):
        for item_key, invalid_values in INVALID_ROW_VALUES.items():
            if row.get(item_key) in invalid_values:
                return False
        return True

    def _first_invalid_overall(self, csv_rows):
        for row in csv_rows:
            cur_invalid_monitors = self._get_monitors_exceeding_threshold(row, self.monitor_threshold)
            if len(cur_invalid_monitors) > 0:
                return [row], [cur_invalid_monitors]
        return [], []

    def _first_invalid_each(self, csv_rows):
        monitor_threshold = self.monitor_threshold.copy()  # use a copy, as will pop item later
        invalid_rows, invalid_monitors = [], []
        for row in csv_rows:
            cur_invalid_monitors = self._get_monitors_exceeding_threshold(row, monitor_threshold)
            if len(cur_invalid_monitors) > 0:
                invalid_rows.append(row)
                invalid_monitors.append(cur_invalid_monitors)
                for monitor in cur_invalid_monitors:
                    monitor_threshold.pop(monitor)

        return invalid_rows, invalid_monitors

    def _all_invalid(self, csv_rows):
        invalid_rows = []
        invalid_monitors = []
        for row in csv_rows:
            cur_invalid_monitors = self._get_monitors_exceeding_threshold(row, self.monitor_threshold)
            if len(cur_invalid_monitors) > 0:
                invalid_rows.append(row)
                invalid_monitors.append(cur_invalid_monitors)
        return invalid_rows, invalid_monitors


def print_in_markdown_table(input_dict, max_column_len=30):
    """
    Print a dict in markdown table format.
    Dict is in format liek `{"column_1": ["value1", "value2"], "column_2": ["value_3", "value_4"]}`.

    Args:
      input_dict: dict value with keys being all str and values being all list or tuple
          Each element in values needs to be a str.
      max_column_len: int value for each column max print length.

    Exaples:
    >>> from compare import analyser
    >>> aa = {"aa": ["11", "22", "334455"], "bb": ["dd", "eeff"]}
    >>> analyser.print_in_markdown_table(aa)
    # |     aa |   bb |
    # |-------:|-----:|
    # |     11 |   dd |
    # |     22 | eeff |
    # | 334455 |      |

    >>> # Similar with pandas function `to_markdown`
    >>> import pandas as pd
    >>> tt = pd.DataFrame(aa)
    >>> print(tt.to_markdown(index=False))
    """
    check_type(max_column_len, int, param_name="max_column_len")
    if max_column_len <= 0:
        raise ValueError(f"max_column_len needs to be > 0, but got {max_column_len}")

    check_type(input_dict, dict, param_name="input_dict")
    if len(input_dict) == 0:
        raise ValueError("input_dict is empty")
    check_element_type(input_dict.keys(), element_type=str, param_name="keys of input_dict")
    check_element_type(input_dict.values(), element_type=(list, tuple), param_name="values of input_dict")
    for key, value in input_dict.items():
        check_element_type(value, element_type=str, param_name=f"values of input_dict['{key}']")

    max_lens = {key: max([len(ii) for ii in value]) for key, value in input_dict.items()}
    max_lens = {key: min(max(value + 1, len(key) + 1), max_column_len) for key, value in max_lens.items()}

    # Table header
    print_str = "\n"
    print_str += "|" + " |".join([" " * (max_len - len(key)) + key for key, max_len in max_lens.items()]) + " |\n"
    # Sep line
    print_str += "|" + ":|".join(["-" * max_len for max_len in max_lens.values()]) + ":|\n"

    # Body
    num_rows = max([len(values) for values in input_dict.values()])
    for row_id in range(num_rows):
        body = []
        for key, max_len in max_lens.items():
            cur_values = input_dict[key]
            cur_value = cur_values[row_id] if len(cur_values) > row_id else ""
            if len(cur_value) >= max_len:
                cur_value = " " + cur_value[: max_len - 4] + "..."
            body.append(" " * (max_len - len(cur_value)) + cur_value)
        print_str += "|" + " |".join(body) + " |\n"

    utils.logger.info(print_str)
