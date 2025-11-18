# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

import csv
import fcntl
import json
import multiprocessing
import os
import re
import shutil
import stat
import sys
import zipfile

import numpy as np
import pandas as pd

from msprobe.infer.utils.constants import FileCheckConst
from msprobe.core.common.log import logger
from msprobe.core.common.decorator import recursion_depth_decorator

proc_lock = multiprocessing.Lock()


class FileCheckException(Exception):
    INVALID_FILE_ERROR = 0
    FILE_PERMISSION_ERROR = 1
    SOFT_LINK_ERROR = 2
    ILLEGAL_PATH_ERROR = 3
    ILLEGAL_PARAM_ERROR = 4
    FILE_TOO_LARGE_ERROR = 5

    err_strs = {
        SOFT_LINK_ERROR: "[msit] 检测到软链接： ",
        FILE_PERMISSION_ERROR: "[msit] 文件权限错误： ",
        INVALID_FILE_ERROR: "[msit] 无效文件： ",
        ILLEGAL_PATH_ERROR: "[msit] 非法文件路径： ",
        ILLEGAL_PARAM_ERROR: "[msit] 非法打开方式： ",
        FILE_TOO_LARGE_ERROR: "[msit] 文件过大： "
    }

    def __init__(self, code, error_info=''):
        super().__init__()
        self.code = code
        self.error_info = self.err_strs.get(code) + error_info

    def __str__(self):
        return self.error_info


class FileChecker:
    """
    The class for check file.

    Attributes:
        file_path: The file or dictionary path to be verified
        path_type: file or dictionary type
        ability(str): one of ["r", "w", "x", "rw", "rx", "wx", "rwx"], r: read, w: write, x: execute
        file_type(str): The correct file type for file
    """

    def __init__(
        self,
        file_path,
        path_type,
        ability=None,
        file_type=None,
        max_size=None
    ):
        self.file_path = file_path
        self.path_type = self._check_path_type(path_type)
        self.ability = self._check_ability_type(ability)
        self.file_type = file_type
        self.max_size = max_size

    @staticmethod
    def _check_path_type(path_type):
        if path_type not in [FileCheckConst.DIR, FileCheckConst.FILE]:
            logger.error(f'The path_type must be {FileCheckConst.DIR} or {FileCheckConst.FILE}.')
            raise FileCheckException(FileCheckException.ILLEGAL_PARAM_ERROR)
        return path_type

    @staticmethod
    def _check_ability_type(ability):
        if ability and ability not in FileCheckConst.PERM_OPTIONS:
            logger.error(f'The ability must be one of {FileCheckConst.PERM_OPTIONS}.')
            raise FileCheckException(FileCheckException.ILLEGAL_PARAM_ERROR)
        return ability

    def common_check(self):
        """
        功能：基本文件权限校验，包括文件存在性、软连接、文件长度、文件类型、文件读写权限、文件属组、文件路径特殊字符、文件后缀名等
        注意：文件后缀的合法性，非通用操作，可使用其他独立接口实现
        """
        check_path_exists(self.file_path)
        check_link(self.file_path)
        self.file_path = os.path.realpath(self.file_path)
        check_path_length(self.file_path)
        check_path_type(self.file_path, self.path_type)
        self.check_path_ability()
        check_path_owner_consistent(self.file_path)
        check_path_pattern_valid(self.file_path)
        check_common_file_size(self.file_path, self.max_size)
        check_file_suffix(self.file_path, self.file_type)
        check_path_no_others_write(self.file_path)
        if self.path_type == FileCheckConst.FILE:
            check_dirpath_permission(self.file_path)
        return self.file_path

    def check_path_ability(self):
        if not self.ability:
            return

        if FileCheckConst.READ_ABLE in self.ability:
            check_path_readability(self.file_path)
        if FileCheckConst.WRITE_ABLE in self.ability:
            check_path_writability(self.file_path)
        if FileCheckConst.EXECUTE_ABLE in self.ability:
            check_path_executable(self.file_path)


class FileOpen:
    """
    The class for open file by a safe way.

    Attributes:
        file_path: The file or dictionary path to be opened.
        mode(str): The file open mode
    """
    SUPPORT_READ_MODE = ["r", "rb"]
    SUPPORT_WRITE_MODE = ["w", "wb", "a", "ab"]
    SUPPORT_READ_WRITE_MODE = ["r+", "rb+", "w+", "wb+", "a+", "ab+"]

    def __init__(self, file_path, mode, encoding='utf-8'):
        self.file_path = file_path
        self.mode = mode
        self.encoding = encoding
        self._handle = None

    def __enter__(self):
        self.check_file_path()
        binary_mode = "b"
        if binary_mode not in self.mode:
            self._handle = open(self.file_path, self.mode, encoding=self.encoding)
        else:
            self._handle = open(self.file_path, self.mode)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle.close()

    def check_file_path(self):
        support_mode = self.SUPPORT_READ_MODE + self.SUPPORT_WRITE_MODE + self.SUPPORT_READ_WRITE_MODE
        if self.mode not in support_mode:
            logger.error(f"File open not support {self.mode} mode")
            raise FileCheckException(FileCheckException.ILLEGAL_PARAM_ERROR)
        check_link(self.file_path)
        self.file_path = os.path.realpath(self.file_path)
        check_path_length(self.file_path)
        self.check_ability_and_owner()
        check_path_pattern_valid(self.file_path)
        if os.path.exists(self.file_path):
            check_common_file_size(self.file_path)
            check_path_no_others_write(self.file_path)
            check_dirpath_permission(self.file_path)

    def check_ability_and_owner(self):
        if self.mode in self.SUPPORT_READ_MODE:
            check_path_exists(self.file_path)
            check_path_readability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_WRITE_MODE and os.path.exists(self.file_path):
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_READ_WRITE_MODE and os.path.exists(self.file_path):
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)


def check_link(path):
    abs_path = os.path.abspath(path)
    if os.path.islink(abs_path):
        logger.error(f"The file path {path} is a soft link.")
        raise FileCheckException(FileCheckException.SOFT_LINK_ERROR)


def check_path_length(path, name_length=None):
    file_max_name_length = name_length if name_length else FileCheckConst.FILE_NAME_LENGTH
    if len(path) > FileCheckConst.DIRECTORY_LENGTH or \
            len(os.path.basename(path)) > file_max_name_length:
        logger.error('The file path length exceeds limit.')
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_path_exists(path):
    if not os.path.exists(path):
        logger.error(f"The file path {path} does not exist.")
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_path_not_exists(path):
    if os.path.exists(path):
        logger.error(f"The file path {path} already exist.")
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_path_readability(path):
    if not os.access(path, os.R_OK):
        logger.error(f"The file path {path} is not readable.")
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_writability(path):
    if not os.access(path, os.W_OK):
        logger.error(f"The file path {path} is not writable.")
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_executable(path):
    if not os.access(path, os.X_OK):
        logger.error(f"The file path {path} is not executable.")
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_other_user_writable(path):
    st = os.stat(path)
    if st.st_mode & 0o002:
        logger.error(f"The file path {path} may be insecure because other users have write permissions.")
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_owner_consistent(path):
    file_owner = os.stat(path).st_uid
    if file_owner != os.getuid() and os.getuid() != 0:
        logger.error(f"The file path {path} may be insecure because is does not belong to you.")
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_pattern_valid(path):
    if not re.match(FileCheckConst.FILE_VALID_PATTERN, path):
        logger.error(f"The file path {path} contains special characters.")
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_file_size(file_path, max_size):
    try:
        file_size = os.path.getsize(file_path)
    except OSError as os_error:
        logger.error(f'Failed to open "{file_path}". {str(os_error)}')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR) from os_error
    if file_size >= max_size:
        logger.error(f'The size ({file_size}) of {file_path} exceeds ({max_size}) bytes, tools not support.')
        raise FileCheckException(FileCheckException.FILE_TOO_LARGE_ERROR)


def check_common_file_size(file_path, max_file_size=None):
    if not os.path.isfile(file_path):
        return

    if max_file_size is None:
        for suffix, max_size in FileCheckConst.FILE_SIZE_DICT.items():
            if file_path.endswith(suffix):
                check_file_size(file_path, max_size)
                return
        check_file_size(file_path, FileCheckConst.COMMOM_FILE_SIZE)
    else:
        check_file_size(file_path, max_file_size)


def check_file_suffix(file_path, file_suffix):
    if file_suffix:
        if not file_path.endswith(file_suffix):
            logger.error(f"The {file_path} should be a {file_suffix} file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)


def check_path_type(file_path, file_type):
    if file_type == FileCheckConst.FILE:
        if not os.path.isfile(file_path):
            logger.error(f"The {file_path} should be a file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)
    if file_type == FileCheckConst.DIR:
        if not os.path.isdir(file_path):
            logger.error(f"The {file_path} should be a dictionary!")
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)


def check_group_writable(file_path):
    path_stat = os.stat(file_path)
    is_writable = bool(path_stat.st_mode & stat.S_IWGRP)
    return is_writable


def check_others_writable(file_path):
    path_stat = os.stat(file_path)
    is_writable = bool(path_stat.st_mode & stat.S_IWOTH)
    return is_writable


def make_dir(dir_path):
    check_path_before_create(dir_path)
    dir_path = os.path.realpath(dir_path)
    if os.path.isdir(dir_path):
        return
    try:
        os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR,
                                 f"Failed to create {dir_path}. "
                                 f"Please check the path permission or disk space. {str(ex)}") from ex
    file_check = FileChecker(dir_path, FileCheckConst.DIR)
    file_check.common_check()


@recursion_depth_decorator('components.utils.file_utils.create_directory', max_depth=16)
def create_directory(dir_path):
    """
    Function Description:
        creating a safe directory with specified permissions
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    check_link(dir_path)
    check_path_before_create(dir_path)
    dir_path = os.path.realpath(dir_path)
    parent_dir = os.path.dirname(dir_path)
    if not os.path.isdir(parent_dir):
        create_directory(parent_dir)
    make_dir(dir_path)


def check_path_before_create(path):
    check_link(path)
    if path_len_exceeds_limit(path):
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR, 'The file path length exceeds limit.')

    if not re.match(FileCheckConst.FILE_PATTERN, os.path.realpath(path)):
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR,
                                 f"The file path {path} contains special characters.")


def check_dirpath_permission(path):
    path = os.path.realpath(path)
    dirpath = os.path.dirname(path)
    if check_others_writable(dirpath):
        logger.warning(f"The directory is writable by others: {dirpath}.")
    try:
        check_path_owner_consistent(dirpath)
    except FileCheckException:
        logger.warning(f"The directory {dirpath} is not yours.")


def check_file_or_directory_path(path, isdir=False, is_strict=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
        is_strict: whether to perform stricter validation (e.g., verify group cannot write to path)
    Exception Description:
        when invalid data throw exception
    """
    if isdir:
        path_checker = FileChecker(path, FileCheckConst.DIR, FileCheckConst.WRITE_ABLE)
    else:
        path_checker = FileChecker(path, FileCheckConst.FILE, FileCheckConst.READ_ABLE)
    path_checker.common_check()

    if is_strict:
        if check_group_writable(path):
            raise FileCheckException(
                FileCheckException.FILE_PERMISSION_ERROR,
                f"The directory/file must not allow write access to group. Directory/File path: {path}"
            )


def check_path_no_others_write(file_path):
    if check_group_writable(file_path):
        logger.warning(f"The directory/file path is writable by group: {file_path}.")

    if check_others_writable(file_path):
        raise FileCheckException(
            FileCheckException.FILE_PERMISSION_ERROR,
            f"The directory/file must not allow write access to others. Directory/File path: {file_path}"
        )


def check_path_no_group_others_write(file_path):
    if check_group_writable(file_path) or check_others_writable(file_path):
        raise FileCheckException(
            FileCheckException.FILE_PERMISSION_ERROR,
            f"The directory/file must not allow write access to group or others. Directory/File path: {file_path}"
        )


def change_mode(path, mode):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, mode)
    except PermissionError as ex:
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR,
                                 f"Failed to change {path} authority. {str(ex)}") from ex


@recursion_depth_decorator('components.utils.file_utils.recursive_chmod')
def recursive_chmod(path):
    """
    递归地修改目录及其子目录和文件的权限，文件修改为640，路径修改为750

    :param path: 要修改权限的目录路径
    """
    for _, dirs, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(path, file_name)
            change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)
        for dir_name in dirs:
            dir_path = os.path.join(path, dir_name)
            change_mode(dir_path, FileCheckConst.DATA_DIR_AUTHORITY)
            recursive_chmod(dir_path)


def path_len_exceeds_limit(file_path):
    return len(os.path.realpath(file_path)) > FileCheckConst.DIRECTORY_LENGTH or \
        len(os.path.basename(file_path)) > FileCheckConst.FILE_NAME_LENGTH


def check_file_type(path):
    """
    Function Description:
        determine if it is a file or a directory
    Parameter:
        path: path
    Exception Description:
        when neither a file nor a directory throw exception
    """
    if os.path.isdir(path):
        return FileCheckConst.DIR
    elif os.path.isfile(path):
        return FileCheckConst.FILE
    else:
        logger.error('path does not exist, please check!')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)


def root_privilege_warning():
    if os.getuid() == 0:
        logger.warning(
            "msit is being run as root. "
            "To avoid security risks, it is recommended to switch to a regular user to run it."
        )


def load_npy(filepath):
    check_file_or_directory_path(filepath)
    try:
        npy = np.load(filepath, allow_pickle=False)
    except Exception as e:
        logger.error(f"The numpy file failed to load. Please check the path: {filepath}.")
        raise RuntimeError(f"Load numpy file {filepath} failed.") from e
    return npy


def load_json(json_path):
    try:
        with FileOpen(json_path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'load json file "{os.path.basename(json_path)}" failed.')
        raise RuntimeError(f"Load json file {json_path} failed.") from e
    return data


def save_json(json_path, data, indent=None, mode="w"):
    check_path_before_create(json_path)
    json_path = os.path.realpath(json_path)
    try:
        with FileOpen(json_path, mode) as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=indent)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'Save json file "{os.path.basename(json_path)}" failed.')
        raise RuntimeError(f"Save json file {json_path} failed.") from e
    change_mode(json_path, FileCheckConst.DATA_FILE_AUTHORITY)


def move_directory(src_path, dst_path):
    check_file_or_directory_path(src_path, isdir=True)
    check_path_before_create(dst_path)
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        logger.error(f"move directory {src_path} to {dst_path} failed")
        raise RuntimeError(f"move directory {src_path} to {dst_path} failed") from e
    change_mode(dst_path, FileCheckConst.DATA_DIR_AUTHORITY)


def move_file(src_path, dst_path):
    check_file_or_directory_path(src_path)
    check_path_before_create(dst_path)
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        logger.error(f"move file {src_path} to {dst_path} failed")
        raise RuntimeError(f"move file {src_path} to {dst_path} failed") from e
    change_mode(dst_path, FileCheckConst.DATA_FILE_AUTHORITY)


def save_npy(data, filepath):
    check_path_before_create(filepath)
    filepath = os.path.realpath(filepath)
    try:
        np.save(filepath, data)
    except Exception as e:
        logger.error(f"The numpy file failed to save. Please check the path: {filepath}.")
        raise RuntimeError(f"Save numpy file {filepath} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def save_npy_to_txt(data, dst_file='', align=0):
    if os.path.exists(dst_file):
        logger.info(f"Dst file {dst_file} exists, will not save new one.")
        return
    shape = data.shape
    data = data.flatten()
    if align == 0:
        align = 1 if len(shape) == 0 else shape[-1]
    elif data.size % align != 0:
        pad_array = np.zeros((align - data.size % align,))
        data = np.append(data, pad_array)
    check_path_before_create(dst_file)
    dst_file = os.path.realpath(dst_file)
    try:
        np.savetxt(dst_file, data.reshape((-1, align)), delimiter=' ', fmt='%g')
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)} when savetxt to{dst_file}")
    change_mode(dst_file, FileCheckConst.DATA_FILE_AUTHORITY)


def save_onnx(model, filepath):
    check_path_before_create(filepath)
    filepath = os.path.realpath(filepath)
    try:
        import onnx
        onnx.save(model, filepath)
    except Exception as e:
        logger.error(f"The onnx model failed to save. Please check the path: {filepath}.")
        raise RuntimeError(f"Save onnx model {filepath} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def write_csv(data, filepath, mode="a+", malicious_check=False):
    def csv_value_is_valid(value: str) -> bool:
        if not isinstance(value, str):
            return True
        try:
            # -1.00 or +1.00 should be considered as digit numbers
            float(value)
        except ValueError:
            # otherwise, they will be considered as formular injections
            return not bool(re.compile(FileCheckConst.CSV_BLACK_LIST).search(value))
        return True

    if malicious_check:
        for row in data:
            for cell in row:
                if not csv_value_is_valid(cell):
                    raise RuntimeError(f"Malicious value [{cell}] is not allowed "
                                       f"to be written into the csv: {filepath}.")

    check_path_before_create(filepath)
    file_path = os.path.realpath(filepath)
    try:
        with FileOpen(filepath, mode, encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    except Exception as e:
        logger.error(f'Save csv file "{os.path.basename(file_path)}" failed')
        raise RuntimeError(f"Save csv file {file_path} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def read_csv(filepath, as_pd=True, header='infer'):
    check_file_or_directory_path(filepath)
    try:
        if as_pd:
            csv_data = pd.read_csv(filepath, header=header)
        else:
            with FileOpen(filepath, 'r', encoding='utf-8-sig') as f:
                csv_reader = csv.reader(f, delimiter=',')
                csv_data = list(csv_reader)
    except Exception as e:
        logger.error(f"The csv file failed to load. Please check the path: {filepath}.")
        raise RuntimeError(f"Read csv file {filepath} failed.") from e
    return csv_data


def write_df_to_csv(data, filepath, mode="w", header=True, malicious_check=False):
    def csv_value_is_valid(value: str) -> bool:
        if not isinstance(value, str):
            return True
        try:
            # -1.00 or +1.00 should be considered as digit numbers
            float(value)
        except ValueError:
            # otherwise, they will be considered as formular injections
            return not bool(re.compile(FileCheckConst.CSV_BLACK_LIST).search(value))
        return True

    if not isinstance(data, pd.DataFrame):
        raise ValueError("The data type of data is not supported. Only support pd.DataFrame.")

    if malicious_check:
        for i in range(len(data)):
            for j in range(len(data.columns)):
                cell = data.iloc[i, j]
                if not csv_value_is_valid(cell):
                    raise RuntimeError(f"Malicious value [{cell}] is not allowed "
                                       f"to be written into the csv: {filepath}.")

    check_path_before_create(filepath)
    file_path = os.path.realpath(filepath)
    try:
        data.to_csv(filepath, mode=mode, header=header, index=False)
    except Exception as e:
        logger.error(f'Save csv file "{os.path.basename(file_path)}" failed')
        raise RuntimeError(f"Save csv file {file_path} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def remove_path(path):
    if not os.path.exists(path):
        return
    if os.path.islink(path):
        logger.error(f"Failed to delete {path}, it is a symbolic link.")
        raise RuntimeError("Delete file or directory failed.")
    try:
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    except PermissionError as err:
        logger.error(f"Failed to delete {path}. Please check the permission.")
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR) from err
    except Exception as e:
        logger.error(f"Failed to delete {path}. Please check.")
        raise RuntimeError("Delete file or directory failed.") from e


def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        logger.error('Failed to load json.')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        logger.error('Json file content is not a dictionary!')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)
    return json_obj


def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()


# 对os.walk设置遍历深度
def os_walk_for_files(path, depth):
    res = []
    for root, _, files in os.walk(path, topdown=True):
        check_file_or_directory_path(root, isdir=True)
        if root.count(os.sep) - path.count(os.sep) >= depth:
            _[:] = []
        else:
            for file in files:
                res.append({"file": file, "root": root})
    return res


def check_zip_file(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        total_size = 0
        if len(zip_file.infolist()) > FileCheckConst.MAX_FILE_IN_ZIP_SIZE:
            raise ValueError(f"Too many files in {os.path.basename(zip_file_path)}")
        for file_info in zip_file.infolist():
            if file_info.file_size > FileCheckConst.MAX_FILE_SIZE:
                raise ValueError(f"File {file_info.filename} is too large to extract")

            total_size += file_info.file_size
            if total_size > FileCheckConst.MAX_ZIP_SIZE:
                raise ValueError(f"Total extracted size exceeds the limit of {FileCheckConst.MAX_ZIP_SIZE} bytes")


def create_file_with_list(result_list, filepath):
    check_path_before_create(filepath)
    filepath = os.path.realpath(filepath)
    try:
        with FileOpen(filepath, 'w', encoding='utf-8') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            for item in result_list:
                file.write(item + '\n')
            fcntl.flock(file, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'Save list to file "{os.path.basename(filepath)}" failed.')
        raise RuntimeError(f"Save list to file {os.path.basename(filepath)} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def create_file_with_content(data, filepath):
    check_path_before_create(filepath)
    filepath = os.path.realpath(filepath)
    try:
        with FileOpen(filepath, 'w', encoding='utf-8') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(data)
            fcntl.flock(file, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'Save content to file "{os.path.basename(filepath)}" failed.')
        raise RuntimeError(f"Save content to file {os.path.basename(filepath)} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def check_file_whether_exist_or_not(filepath):
    if os.path.exists(filepath):
        check_file_or_directory_path(filepath)
    else:
        check_path_before_create(filepath)


def add_file_to_zip(zip_file_path, file_path, arc_path=None):
    """
    Add a file to a ZIP archive, if zip does not exist, create one.

    :param zip_file_path: Path to the ZIP archive
    :param file_path: Path to the file to add
    :param arc_path: Optional path inside the ZIP archive where the file should be added
    """
    check_file_or_directory_path(file_path)
    check_file_suffix(zip_file_path, FileCheckConst.ZIP_SUFFIX)
    check_file_whether_exist_or_not(zip_file_path)
    check_file_size(file_path, FileCheckConst.MAX_FILE_IN_ZIP_SIZE)
    zip_size = os.path.getsize(zip_file_path) if os.path.exists(zip_file_path) else 0
    if zip_size + os.path.getsize(file_path) > FileCheckConst.MAX_ZIP_SIZE:
        raise RuntimeError(f"ZIP file size exceeds the limit of {FileCheckConst.MAX_ZIP_SIZE} bytes")
    try:
        proc_lock.acquire()
        with zipfile.ZipFile(zip_file_path, 'a') as zip_file:
            zip_file.write(file_path, arc_path)
    except Exception as e:
        logger.error(f'add file to zip "{os.path.basename(zip_file_path)}" failed.')
        raise RuntimeError(f"add file to zip {os.path.basename(zip_file_path)} failed.") from e
    finally:
        proc_lock.release()
    change_mode(zip_file_path, FileCheckConst.DATA_FILE_AUTHORITY)


def create_file_in_zip(zip_file_path, file_name, content):
    """
    Create a file with content inside a ZIP archive.

    :param zip_file_path: Path to the ZIP archive
    :param file_name: Name of the file to create
    :param content: Content to write to the file
    """
    check_file_suffix(zip_file_path, FileCheckConst.ZIP_SUFFIX)
    check_file_whether_exist_or_not(zip_file_path)
    zip_size = os.path.getsize(zip_file_path) if os.path.exists(zip_file_path) else 0
    if zip_size + sys.getsizeof(content) > FileCheckConst.MAX_ZIP_SIZE:
        raise RuntimeError(f"ZIP file size exceeds the limit of {FileCheckConst.MAX_ZIP_SIZE} bytes")
    try:
        with open(zip_file_path, 'a+') as f:  # 必须用 'a+' 模式才能 flock
            # 2. 获取排他锁（阻塞直到成功）
            fcntl.flock(f, fcntl.LOCK_EX)  # LOCK_EX: 独占锁
            with zipfile.ZipFile(zip_file_path, 'a') as zip_file:
                zip_info = zipfile.ZipInfo(file_name)
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_file.writestr(zip_info, content)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'Save content to file "{os.path.basename(zip_file_path)}" failed.')
        raise RuntimeError(f"Save content to file {os.path.basename(zip_file_path)} failed.") from e
    change_mode(zip_file_path, FileCheckConst.DATA_FILE_AUTHORITY)


def extract_zip(zip_file_path, extract_dir):
    """
    Extract the contents of a ZIP archive to a specified directory.

    :param zip_file_path: Path to the ZIP archive
    :param extract_dir: Directory to extract the contents to
    """
    check_file_suffix(zip_file_path, FileCheckConst.ZIP_SUFFIX)
    check_file_or_directory_path(zip_file_path)
    create_directory(extract_dir)
    try:
        proc_lock.acquire()
        check_zip_file(zip_file_path)
    except Exception as e:
        logger.error(f'Save content to file "{os.path.basename(zip_file_path)}" failed.')
        raise RuntimeError(f"Save content to file {os.path.basename(zip_file_path)} failed.") from e
    finally:
        proc_lock.release()
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(extract_dir)
    except Exception as e:
        raise RuntimeError(f"extract zip file {os.path.basename(zip_file_path)} failed") from e
    recursive_chmod(extract_dir)


def split_zip_file_path(zip_file_path):
    check_file_suffix(zip_file_path, FileCheckConst.ZIP_SUFFIX)
    zip_file_path = os.path.realpath(zip_file_path)
    return os.path.dirname(zip_file_path), os.path.basename(zip_file_path)


def check_path_type_is_file(path):
    if not os.path.isfile(path):
        logger.error(f'The path {path} is not file.')
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_path_type_is_dir(path):
    if not os.path.isdir(path):
        logger.error(f'The path {path} is not dir.')
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_input_file_path(path, file_max_size=FileCheckConst.MAX_COMMON_FILE_SIZE, check_executable=False):
    path = os.path.realpath(path)
    check_path_exists(path)
    check_path_type_is_file(path)
    check_link(path)
    check_path_length(path)
    check_path_pattern_valid(path)
    check_path_readability(path)
    check_file_size(path, file_max_size)
    check_path_owner_consistent(path)
    if check_executable:
        check_path_executable(path)


def check_input_dir_path(path):
    path = os.path.realpath(path)
    check_path_exists(path)
    check_path_type_is_dir(path)
    check_link(path)
    check_path_length(path)
    check_path_pattern_valid(path)
    check_path_readability(path)
    check_path_owner_consistent(path)


def check_output_file_path(path):
    path = os.path.realpath(path)
    check_link(path)
    check_path_length(path)
    check_path_pattern_valid(path)
    if os.path.exists(path):
        check_path_owner_consistent(path)


def check_output_dir_path(path):
    path = os.path.realpath(path)
    check_link(path)
    check_path_length(path)
    check_path_pattern_valid(path)
    if os.path.exists(path):
        check_path_writability(path)
        check_path_owner_consistent(path)
