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

import os
from enum import Enum, auto, unique
from typing import Union

from msprobe.infer.utils.check.checker import Checker, CheckResult, rule, EnumInstance


@unique
class FileType(Enum):
    DIRECTORY = auto()
    CHARACTER = auto()
    BLOCK = auto()
    FILE = auto()
    FIFO = auto()
    SYMLINK = auto()
    SOCKET = auto()


class FileStatus(object):
    def __init__(self, file_name: str) -> None:
        file_status = os.lstat(file_name)
        self._file_name = file_name
        self.status_mode = file_status.st_mode
        self._file_uid = file_status.st_uid
        self._file_gid = file_status.st_gid
        self._file_size = file_status.st_size
        self._file_extension = os.path.splitext(file_name)[1]

    @property
    def file_name(self) -> int:
        return self._file_name

    @property
    def size(self) -> int:
        return self._file_size

    @property
    def perm_bits(self) -> int:
        return os.st.S_IMODE(self.status_mode)

    @property
    def uid(self) -> int:
        return self._file_uid

    @property
    def gid(self) -> int:
        return self._file_gid

    @property
    def extension(self) -> str:
        return self._file_extension

    @property
    def ftype(self):
        file_type_map = {
            os.st.S_IFDIR: FileType.DIRECTORY,
            os.st.S_IFCHR: FileType.CHARACTER,
            os.st.S_IFBLK: FileType.BLOCK,
            os.st.S_IFREG: FileType.FILE,
            os.st.S_IFIFO: FileType.FIFO,
            os.st.S_IFLNK: FileType.SYMLINK,
            os.st.S_IFSOCK: FileType.SOCKET,
        }

        file_mode = os.st.S_IFMT(self.status_mode)
        return file_type_map.get(file_mode, None)


class PathChecker(Checker):
    def __init__(self, instance=EnumInstance.NO_INSTANCE, converter=None):
        super().__init__(instance, converter)
        self.f_status = None
        self.f_state = False
        self.converter = converter or self.path_converter
        self.status_err_msg = None

    def path_converter(self, ori_path):
        ori_path = os.path.realpath(ori_path)
        try:
            self.f_status = FileStatus(ori_path)
        except OSError as e:
            self.status_err_msg = e.strerror + ': ' + e.filename
        except TypeError:
            self.status_err_msg = f'TypeError: {ori_path}'
        except Exception as e:
            self.status_err_msg = str(e)
        else:
            self.f_state = True

        return ori_path, True, self.status_err_msg

    @rule()
    def exists(self) -> Union["PathChecker", CheckResult]:
        err_msg = f"No such file or directory: {self.instance}. {self.status_err_msg}"
        return self.f_state, err_msg

    @rule()
    def is_file(self) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return self.f_status.ftype is FileType.FILE, f"Not a file: {self.instance}"

    @rule()
    def is_dir(self) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return self.f_status.ftype is FileType.DIRECTORY, f"Not a directory: {self.instance}"

    @rule()
    def is_softlink(self) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return self.f_status.ftype is FileType.SYMLINK, f"Not a soft link: {self.instance}"

    @rule()
    def forbidden_softlink(self, flag=True) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        if flag:
            return self.f_status.ftype is not FileType.SYMLINK, f"Soft link: {self.instance}"
        else:
            return True, "Soft link check passed."

    @rule()
    def is_uid_matched(self, *uids: int) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return (
                self.f_status.uid in uids,
                f"User ID not matched: {self.instance}[{self.f_status.uid} ∉ {str(uids)}]. ",
            )

    @rule()
    def is_owner(self, *uids: int) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return (
                os.getuid() == 0 or self.f_status.uid == os.getuid(),
                f"User ID not matched: {self.instance}[{self.f_status.uid} ∉ {str(uids)}]. ",
            )

    @rule()
    def is_gid_matched(self, *gids: int) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return (
                self.f_status.gid in gids,
                f"Group ID not matched: {self.instance}[{self.f_status.gid} ∉ {str(gids)}]. ",
            )

    @rule()
    def is_readable(self) -> Union["PathChecker", CheckResult]:
        return os.access(self.instance, os.R_OK), self.instance + " is not readable"

    @rule()
    def is_writeable(self) -> Union["PathChecker", CheckResult]:
        return os.access(self.instance, os.W_OK), self.instance + " is not writable"

    @rule()
    def is_executable(self) -> Union["PathChecker", CheckResult]:
        return os.access(self.instance, os.X_OK), self.instance + " is not executable"

    @rule()
    def is_not_readable_to_others(self) -> Union["PathChecker", CheckResult]:
        ins = self.instance + " is readable to others"
        return CheckResult(not bool(self.f_status.status_mode & os.st.S_IROTH), ins)

    @rule()
    def is_not_writable_to_group(self) -> Union["PathChecker", CheckResult]:
        ins = self.instance + " is writable to groups"
        return CheckResult(not bool(self.f_status.status_mode & os.st.S_IWGRP), ins)

    @rule()
    def is_not_writable_to_others(self) -> Union["PathChecker", CheckResult]:
        ins = self.instance + " is writable to others"
        return CheckResult(not bool(self.f_status.status_mode & os.st.S_IWOTH), ins)

    @rule()
    def is_not_executable_to_others(self) -> Union["PathChecker", CheckResult]:
        ins = self.instance + " is executable to others"
        return CheckResult(not bool(self.f_status.status_mode & os.st.S_IXOTH), ins)

    @rule()
    def max_perm(self, perm_bits: int) -> Union["PathChecker", CheckResult]:
        if 0o777 < perm_bits or perm_bits < 0:
            msg = "Permission bits should be in range from 0 to 0o777"
            raise ValueError(f"{msg}")

        part_mapping = ["others", "groups", "users"]
        perm_mapping = ["executing", "writing", "reading"]

        for count in range(9):
            mask = 1 << count

            if (self.f_status.perm_bits & mask) and not (perm_bits & mask):
                err_msg = (
                    f"{part_mapping[count // 3]} "
                    f"should not have {perm_mapping[count % 3]} "
                    f"permissions: {self.instance}"
                )

                return CheckResult(False, err_msg)

        return CheckResult(True)

    @rule("file size larger than expected")
    def max_size(self, expected_size: int) -> Union["PathChecker", CheckResult]:
        err_msg = f"File size larger than expected: {self.instance}"
        return CheckResult(self.f_status.size < expected_size, err_msg)

    @rule("Wrong file suffix")
    def check_extensions(self, extensions) -> Union["PathChecker", CheckResult]:
        return self.f_status.extension == extensions or self.f_status.extension == '.' + extensions

    @rule()
    def is_safe_parent_dir(self) -> Union["PathChecker", CheckResult]:
        path = os.path.realpath(self.instance)
        dirpath = os.path.dirname(path)
        if os.getuid() == 0:
            return True

        dir_checker = PathChecker().any(
            PathChecker().anti(PathChecker().exists()),
            PathChecker().is_dir().is_owner(os.getuid()).is_not_writable_to_others().is_not_writable_to_group(),
        )
        return dir_checker.check(dirpath)

