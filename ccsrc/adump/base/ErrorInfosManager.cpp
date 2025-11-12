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

#include <utility>
#include <map>
#include <string>
#include <cstdlib>
#include <mutex>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include "utils/FileUtils.h"
#include "ErrorInfosManager.h"

namespace MindStudioDebugger {

static std::mutex g_errInfoMtx;
DebuggerErrLevel ErrorInfosManager::topLevel = DebuggerErrLevel::LEVEL_NONE;
DebuggerErrLevel ErrorInfosManager::threshold = DebuggerErrLevel::LEVEL_INFO;

static std::map<DebuggerErrLevel, const char*> ErrorLevelString = {
    {DebuggerErrLevel::LEVEL_CRITICAL, "CRITICAL"},
    {DebuggerErrLevel::LEVEL_ERROR, "ERROR"},
    {DebuggerErrLevel::LEVEL_WARNING, "WARNING"},
    {DebuggerErrLevel::LEVEL_INFO, "INFO"},
    {DebuggerErrLevel::LEVEL_DEBUG, "DEBUG"},
    {DebuggerErrLevel::LEVEL_NONE, "NONE"},
};

static std::map<DebuggerErrno, const char*> ErrnoString = {
    {DebuggerErrno::OK, "OK"},
    {DebuggerErrno::ERROR, "ERROR"},

    {DebuggerErrno::ERROR_FILE_NOT_EXISTS, "FILE_NOT_EXISTS"},
    {DebuggerErrno::ERROR_FILE_ALREADY_EXISTS, "FILE_ALREADY_EXISTS"},
    {DebuggerErrno::ERROR_FAILED_TO_OPEN_FILE, "FAILED_TO_OPEN_FILE"},
    {DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "FAILED_TO_WRITE_FILE"},
    {DebuggerErrno::ERROR_DIR_NOT_EXISTS, "DIR_NOT_EXISTS"},
    {DebuggerErrno::ERROR_PERMISSION_DENINED, "PERMISSION_DENINED"},
    {DebuggerErrno::ERROR_NOT_ALLOW_SOFTLINK, "NOT_ALLOW_SOFTLINK"},
    {DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE, "ILLEGAL_FILE_TYPE"},
    {DebuggerErrno::ERROR_PATH_TOO_LOOG, "PATH_TOO_LOOG"},
    {DebuggerErrno::ERROR_PATH_TOO_DEEP, "PATH_TOO_DEEP"},
    {DebuggerErrno::ERROR_PATH_CONTAINS_INVALID_CHAR, "PATH_CONTAINS_INVALID_CHAR"},
    {DebuggerErrno::ERROR_FILE_TOO_LARGE, "FILE_TOO_LARGE"},
    {DebuggerErrno::ERROR_UNKNOWN_FILE_SUFFIX, "UNKNOWN_FILE_SUFFIX"},
    {DebuggerErrno::ERROR_CANNOT_PARSE_PATH, "CANNOT_PARSE_PATH"},

    {DebuggerErrno::ERROR_INVALID_OPERATION, "INVALID_OPERATION"},
    {DebuggerErrno::ERROR_INVALID_FORMAT, "INVALID_FORMAT"},
    {DebuggerErrno::ERROR_INVALID_VALUE, "INVALID_VALUE"},
    {DebuggerErrno::ERROR_UNKNOWN_FIELD, "UNKNOWN_FIELD"},
    {DebuggerErrno::ERROR_UNKNOWN_VALUE, "UNKNOWN_VALUE"},
    {DebuggerErrno::ERROR_UNKNOWN_TRANS, "UNKNOWN_TRANS"},
    {DebuggerErrno::ERROR_FIELD_NOT_EXISTS, "FIELD_NOT_EXISTS"},
    {DebuggerErrno::ERROR_VALUE_OVERFLOW, "VALUE_OVERFLOW"},

    {DebuggerErrno::ERROR_NO_MEMORY, "NO_MEMORY"},
    {DebuggerErrno::ERROR_BUFFER_OVERFLOW, "BUFFER_OVERFLOW"},
    {DebuggerErrno::ERROR_SYSCALL_FAILED, "SYSCALL_FAILED"},
    {DebuggerErrno::ERROR_OPERATION_FAILED, "OPERATION_FAILED"},

    {DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "DEPENDENCY_NOT_FIND"},
    {DebuggerErrno::ERROR_EXTERNAL_API_ERROR, "EXTERNAL_API_ERROR"},
};


const std::unordered_map<std::string, std::string>& GetInvalidChar(void)
{
    static const std::unordered_map<std::string, std::string> INVALID_CHAR = {
        {"\n", "\\n"}, {"\f", "\\f"}, {"\r", "\\r"}, {"\b", "\\b"},
        {"\t", "\\t"}, {"\v", "\\v"}, {"\u007F", "\\u007F"}
    };
    return INVALID_CHAR;
}

// convert unsafe string to safe string
std::string ToSafeString(const std::string &str)
{
    std::string safeStr(str);
    const std::unordered_map<std::string, std::string> invalidChar = GetInvalidChar();
    size_t i = 0;
    while (i < safeStr.length()) {
        std::string chr(1, safeStr[i]);
        if (invalidChar.find(chr) != invalidChar.end()) {
            const std::string &validStr = invalidChar.at(chr);
            safeStr.replace(i, 1, validStr);
            i += validStr.length();
            continue;
        }
        i++;
    }
    return safeStr;
}


void ErrorInfosManager::LogErrorInfo(DebuggerErrLevel level, DebuggerErrno errId, const std::string& info)
{
    if (level < threshold) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_errInfoMtx);
    std::ostream& output = std::cout;
    output << "[" << ErrorLevelString[level] << "]";
    if (errId != DebuggerErrno::NONE) {
        output << "[" << ErrnoString[errId] << "]";
    }
    output << ToSafeString(info) << std::endl;

    if (level > topLevel) {
        topLevel = level;
    }

    return;
}

DebuggerErrLevel ErrorInfosManager::GetTopErrLevelInDuration()
{
    std::lock_guard<std::mutex> lk(g_errInfoMtx);
    DebuggerErrLevel ret = topLevel;
    topLevel = DebuggerErrLevel::LEVEL_NONE;
    return ret;
}

__attribute__((constructor)) void InitDebuggerThreshold()
{
    const char* msprobeLogLevelEnv = getenv("MSPROBE_LOG_LEVEL");
    if (msprobeLogLevelEnv == nullptr) {
        return;
    }

    int msprobeLogLevel = 1;
    try {
        msprobeLogLevel = std::stoi(msprobeLogLevelEnv);
    } catch (const std::exception& e) {
        return;
    }

    if (msprobeLogLevel >= static_cast<int>(DebuggerErrLevel::LEVEL_DEBUG) &&
        msprobeLogLevel <= static_cast<int>(DebuggerErrLevel::LEVEL_CRITICAL)) {
        ErrorInfosManager::SetLogThreshold(static_cast<DebuggerErrLevel>(msprobeLogLevel));
    }
}

}