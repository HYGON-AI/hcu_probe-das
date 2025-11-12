/*
 * Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace MindStudioDebugger {

enum class DebuggerErrno {
    OK = 0,
    ERROR,
    NONE,

    /* 文件操作类 */
    ERROR_FILE_NOT_EXISTS = 100,
    ERROR_FILE_ALREADY_EXISTS,
    ERROR_FAILED_TO_OPEN_FILE,
    ERROR_FAILED_TO_WRITE_FILE,
    ERROR_DIR_NOT_EXISTS,
    ERROR_PERMISSION_DENINED,
    ERROR_NOT_ALLOW_SOFTLINK,
    ERROR_ILLEGAL_FILE_TYPE,
    ERROR_PATH_TOO_LOOG,
    ERROR_PATH_TOO_DEEP,
    ERROR_PATH_CONTAINS_INVALID_CHAR,
    ERROR_FILE_TOO_LARGE,
    ERROR_UNKNOWN_FILE_SUFFIX,
    ERROR_CANNOT_PARSE_PATH,

    /* 数据解析类 */
    ERROR_INVALID_OPERATION = 200,
    ERROR_INVALID_FORMAT,
    ERROR_INVALID_VALUE,
    ERROR_UNKNOWN_FIELD,
    ERROR_UNKNOWN_VALUE,
    ERROR_UNKNOWN_TRANS,
    ERROR_FIELD_NOT_EXISTS,
    ERROR_VALUE_OVERFLOW,

    /* 系统调用类 */
    ERROR_NO_MEMORY = 300,
    ERROR_BUFFER_OVERFLOW,
    ERROR_SYSCALL_FAILED,
    ERROR_OPERATION_FAILED,

    /* 环境依赖类 */
    ERROR_DEPENDENCY_NOT_FIND = 400,
    ERROR_CONFIGURATION_CONFLICTS,
    ERROR_EXTERNAL_API_ERROR,
};

}