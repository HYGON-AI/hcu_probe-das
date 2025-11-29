# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import stat
import unittest
import argparse
import tempfile

from unittest.mock import patch, Mock

import pytest

from msprobe.infer.utils.check import PathChecker
from msprobe.infer.utils.security_check import (
    is_belong_to_user_or_group,
    is_endswith_extensions,
    get_valid_path,
    check_write_directory,
    find_existing_path,
    is_enough_disk_space_left,
    ms_makedirs
)
from msprobe.infer.utils.file_open_check import FileStat


MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MIN_DUMP_DISK_SPACE = 2147483648  # 2G, 2 * 1024 * 1024 * 1024
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


class TestMakedirs(unittest.TestCase):

    def setUp(self) -> None:
        self.dp = tempfile.TemporaryDirectory()
        self.dp_invalid = tempfile.TemporaryDirectory()
        os.chmod(self.dp_invalid.name, mode=0o777)

    def test_makedirs_valid(self) -> None:
        target_dir = os.path.join(self.dp.name, "d1")
        ms_makedirs(target_dir)
        assert os.path.exists(target_dir)

        target_dir = os.path.join(self.dp.name, "d2/d3/d4")
        ms_makedirs(target_dir)
        assert os.path.exists(target_dir)

    @patch('msprobe.core.common.log.logger')
    def test_makedirs_invalid(self, mock_logger) -> None:
        target_dir = os.path.join(self.dp_invalid.name, "d1")
        if not PathChecker().is_safe_parent_dir().check(os.path.join(self.dp_invalid.name, "d1")):
            ms_makedirs(target_dir)
            mock_logger.assert_called_once_with(f"Output parent directory path {target_dir} is not safe.")

    def tearDown(self) -> None:
        self.dp.cleanup()


def test_is_belong_to_user_or_group_given_path_when_valid_then_pass():
    mock_stat = Mock(st_uid=1000, st_gid=1001)
    with patch('os.getuid', return_value=1000), \
         patch('os.getgroups', return_value=[1001]):
        result = is_belong_to_user_or_group(mock_stat)
        assert result is True


def test_is_belong_to_user_or_group_given_path_when_invalid_then_fail():
    mock_stat = Mock(st_uid=2000, st_gid=2001)
    with patch('os.getuid', return_value=1000), \
         patch('os.getgroups', return_value=[1001]):
        result = is_belong_to_user_or_group(mock_stat)
        assert result is False


def test_is_endswith_extensions_given_list_of_extensions_when_matches_then_true():
    path = "file.txt"
    extensions = ["txt", "md"]
    result = is_endswith_extensions(path, extensions)
    assert result is True


def test_is_endswith_extensions_given_single_extension_when_not_match_then_false():
    path = "file.txt"
    extensions = "md"
    result = is_endswith_extensions(path, extensions)
    assert result is False


# Additional tests would follow the same pattern for each function in security_check.py

def test_get_valid_path_given_empty_path_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        get_valid_path("")


def test_get_valid_path_given_path_with_special_char_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        get_valid_path("/path/with*special?char")


def test_get_valid_path_given_soft_link_when_called_then_raises_value_error():
    with patch('os.path.islink', return_value=True), \
         patch('os.path.abspath', return_value='/path/to/symlink'):
        with pytest.raises(ValueError, match="cannot be soft link"):
            get_valid_path("/path/to/symlink")


def test_get_valid_path_given_long_filename_when_called_then_raises_value_error():
    long_filename = "a" * 257
    with pytest.raises(ValueError):
        get_valid_path(f"/path/{long_filename}")


def test_get_valid_path_given_long_path_when_called_then_raises_value_error():
    long_path = "/".join(["a" * 1000] * 5)
    with pytest.raises(ValueError):
        get_valid_path(long_path)


# The following tests are simplified examples. More tests should be added to reach the required coverage.

def test_check_write_directory_given_nonexistent_directory_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        check_write_directory("/nonexistent/directory")


def test_find_existing_path_given_nonexistent_path_when_depth_exceeded_then_raises_recursion_error():
    with pytest.raises(RecursionError):
        find_existing_path("/nonexistent/path", depth=0)


def test_is_enough_disk_space_left_given_insufficient_space_when_called_then_returns_false():
    dump_path = "/path/to/check"
    with patch('shutil.disk_usage', return_value=Mock(free=MIN_DUMP_DISK_SPACE - 1)):
        result = is_enough_disk_space_left(dump_path)
        assert result is False
