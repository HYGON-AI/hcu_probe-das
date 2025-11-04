# coding=utf-8
# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import logging


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL
}

SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]

LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


def set_log_level(level="info"):
    if level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def set_logger(msit_logger):
    msit_logger.propagate = False
    msit_logger.setLevel(logging.INFO)
    if not msit_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        msit_logger.addHandler(stream_handler)


def get_filter_handle(handle, self):
    def filter_handle(self, record):
        for char in SPECIAL_CHAR:
            record.msg = record.msg.replace(char, '_')
        return handle(record)

    return filter_handle.__get__(self, type(self))


logger = logging.getLogger("msit_logger")
set_logger(logger)
if hasattr(logger, 'handle'):
    logger.handle = get_filter_handle(logger.handle, logger)
else:
    raise RuntimeError('The Python version is not suitable')


def msg_filter(msg):
    if not isinstance(msg, str):
        raise RuntimeError('msg type is not string, please check.')
    for char in SPECIAL_CHAR:
        msg = msg.replace(char, '_')
    return msg
