# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import sys
import argparse
import re
import shutil

from msprobe.infer.utils.file_open_check import FileStat
from msprobe.infer.utils.constants import PATH_WHITE_LIST_REGEX
from msprobe.core.common.log import logger
from msprobe.infer.utils.util import safe_int


STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MIN_DUMP_DISK_SPACE = 2147483648  # 2G, 2 * 1024 * 1024 * 1024
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


def is_belong_to_user_or_group(file_stat):
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def is_endswith_extensions(path, extensions):
    result = False
    if isinstance(extensions, (list, tuple)):
        for extension in extensions:
            if path.endswith(extension):
                result = True
                break
    elif isinstance(extensions, str):
        result = path.endswith(extensions)
    return result


def get_valid_path(path, extensions=None):
    if not path or len(path) == 0:
        raise ValueError("The value of the path cannot be empty.")

    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
        raise ValueError("The value of the path cannot be soft link: {}.".format(path))

    real_path = os.path.realpath(path)

    file_name = os.path.split(real_path)[1]
    if len(file_name) > 255:
        raise ValueError("The length of filename should be less than 256.")
    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if extensions and not is_endswith_extensions(path, extensions):  # Check whether the file name endswith extension
        raise ValueError("The filename {} doesn't endswith \"{}\".".format(path, extensions))

    return real_path


def check_write_directory(dir_name, check_user_stat=True):
    real_dir_name = get_valid_path(dir_name)
    if not os.path.isdir(real_dir_name):
        raise ValueError("The file writen directory {} doesn't exist.".format(dir_name))

    file_stat = os.stat(real_dir_name)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file writen directory {} doesn't belong to the current user or group.".format(dir_name))
    if not os.access(real_dir_name, os.W_OK):
        raise ValueError("Current user doesn't have writen permission to file writen directory {}.".format(dir_name))


def get_valid_write_path(path, extensions=None, check_user_stat=True, is_dir=False):
    real_path = get_valid_path(path, extensions)
    real_path_dir = real_path if is_dir else os.path.dirname(real_path)
    check_write_directory(real_path_dir, check_user_stat=check_user_stat)

    if not is_dir and os.path.exists(real_path):
        if os.path.isdir(real_path):
            raise ValueError("The file {} exist and is a directory.".format(path))
        if check_user_stat and os.stat(real_path).st_uid != os.getuid():  # Has to be exactly belonging to current user
            raise ValueError("The file {} doesn't belong to the current user.".format(path))
        if check_user_stat and os.stat(real_path).st_mode & WRITE_FILE_NOT_PERMITTED_STAT > 0:
            raise ValueError("The file {} permission for others is writable, or is group writable.".format(path))
        if not os.access(real_path, os.W_OK):
            raise ValueError("The file {} exist and not writable.".format(path))
    return real_path


def find_existing_path(path, depth):
    if os.path.exists(path):
        return path
    if depth <= 0:
        raise RecursionError("Output path was not valied")
    parent_path = os.path.dirname(path)
    # 递归查找父目录
    if parent_path and parent_path != path:
        return find_existing_path(parent_path, depth - 1)
    else:
        raise ValueError("Output path was not valied.")


def is_enough_disk_space_left(dump_path, required_space=MIN_DUMP_DISK_SPACE, max_path_depth=200):
    dump_path = os.path.abspath(dump_path)
    existing_path = None
    try:
        existing_path = find_existing_path(dump_path, max_path_depth)
    except ValueError:
        logger.warning("Please check your output path parameter, it seems that it does not exist.")
    except RecursionError:
        logger.warning("The depth of the 'output' path is too large, maximum depth is 200.")
    
    if existing_path:
        empty_disk_space = shutil.disk_usage(existing_path).free
    else:
        logger.warning("Please make sure that the disk has enough space to dump data.")
        root_path = os.path.abspath(os.sep)
        empty_disk_space = shutil.disk_usage(root_path).free

    return empty_disk_space >= required_space


def _check_parent_dir_safe(dir_path):
    from msprobe.infer.utils.check import PathChecker

    def get_root(dir_path, max_depth=200):
        if max_depth <= 0:
            raise OSError(f"Output parent directory path {dir_path} is not safe.")
        # 递归获取需要创建的最高级目录
        if dir_path.parent.exists():
            return dir_path
        return get_root(dir_path.parent, max_depth - 1)

    from pathlib import Path

    dir_path = Path(dir_path)
    root_path = get_root(dir_path)

    if not PathChecker().is_safe_parent_dir().check(str(root_path)):
        logger.warning(f"Output parent directory path {root_path} is not safe.")


def ms_makedirs(dir_path, **kwargs):
    _check_parent_dir_safe(dir_path)
    os.makedirs(dir_path, **kwargs)
