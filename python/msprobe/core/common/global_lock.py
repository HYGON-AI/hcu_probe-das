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

import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import random
import time
import atexit
import os

from msprobe.core.common.log import logger


def is_main_process():
    return multiprocessing.current_process().name == 'MainProcess'


class GlobalLock:
    def __init__(self):
        self.name = self.get_lock_name()
        try:
            self._shm = SharedMemory(create=False, name=self.name)
            time.sleep(random.randint(0, 500) / 10000) # 等待随机时长以避免同时获得锁
        except FileNotFoundError:
            try:
                self._shm = SharedMemory(create=True, name=self.name, size=1)
                self._shm.buf[0] = 0
                logger.debug(f'{self.name} is created.')
            except FileExistsError:
                self.__init__()

    @classmethod
    def get_lock_name(cls):
        if is_main_process():
            return f'global_lock_{os.getpid()}'
        return f'global_lock_{os.getppid()}'

    @classmethod
    def is_lock_exist(cls):
        try:
            SharedMemory(create=False, name=cls.get_lock_name()).close()
            return True
        except FileNotFoundError:
            return False

    def cleanup(self):
        self._shm.close()
        if is_main_process():
            try:
                self._shm.unlink()
                logger.debug(f'{self.name} is unlinked.')
            except FileNotFoundError:
                logger.warning(f'{self.name} has already been unlinked.')

    def acquire(self, timeout=180):
        """
        acquire global lock, default timeout is 3 minutes.

        :param float timeout: timeout(seconds), default value is 180.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self._shm.buf[0] == 0:
                self._shm.buf[0] = 1
                return
            time.sleep(random.randint(10, 500) / 10000)  # 自旋，等待1-50ms
        self._shm.buf[0] = 1

    def release(self):
        self._shm.buf[0] = 0


global_lock = GlobalLock()
atexit.register(global_lock.cleanup)
