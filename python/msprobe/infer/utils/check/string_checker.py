# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import re
import os
from typing import Union

from msprobe.infer.utils.check.checker import Checker, CheckResult, rule


WHITE_LIST_PATTERN = r"^(?!.*\.\.)[a-zA-Z0-9_./-]+$"
BLACK_LIST_PATTERN = r"[^_A-Za-z0-9/.-]"
IDS_PATTERN = r'^(\d+(?:_\d+)*)(,\d+(?:_\d+)*)*$'
INVALID_CHAR = "[\n\f\r\b\t\v\u000D\u000A\u000C\u000B\u0009\u0008\u007F&%$*^#@;]"


class StringChecker(Checker):

    @rule()
    def is_str(self) -> Union["StringChecker", CheckResult]:
        err_msg = f"{self.instance} is not a string"
        return isinstance(self.instance, str), err_msg

    @rule()
    def is_file_name_too_long(self) -> Union["StringChecker", CheckResult]:
        err_msg = "File name too long"
        ret = len(self.instance) > 4095 or any(map(lambda s: len(s) > 255, self.instance.split(os.path.sep)))
        return not ret, err_msg

    @rule()
    def is_str_safe(self) -> Union["StringChecker", CheckResult]:
        err_msg = "String parameter contains invalid characters"
        return re.search(WHITE_LIST_PATTERN, self.instance), err_msg
    
    @rule()
    def is_str_valid_bool(self) -> Union["StringChecker", CheckResult]:
        err_msg = "Boolean value expected 'yes', 'y', 'Y', 'YES', 'true', 't', 'TRUE', 'True', '1' for true"
        return self.instance.lower() in ('yes', 'y', 'Y', 'YES', 'true', 't', 'TRUE', 'True', '1'), err_msg
    
    @rule()
    def is_str_valid_path(self) -> Union["StringChecker", CheckResult]:
        err_msg = "Input path contains invalid characters"
        return not re.search(BLACK_LIST_PATTERN, self.instance), err_msg
    
    @rule()
    def is_str_valid_ids(self):
        err_msg = f"dym range string \"{self.instance}\" is not a legal string"
        return re.match(IDS_PATTERN, self.instance), err_msg

    @rule()
    def str_has_no_invalid_char(self):
        err_msg = "Input string contains invalid chars" 
        return not re.search(INVALID_CHAR, self.instance), err_msg
