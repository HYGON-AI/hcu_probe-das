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

#include <string>
#include <vector>
#include "include/ErrorCode.h"

namespace MindStudioDebugger {

enum class DebuggerErrLevel {
    LEVEL_NONE = -1,          /* 无 */
    LEVEL_DEBUG = 0,          /* 仅作为调试信息，不影响功能 */
    LEVEL_INFO,               /* 用户需要感知的信息，一般不影响功能 */
    LEVEL_WARNING,            /* 告警，可能会影响部分功能，但基础功能还能继续运行 */
    LEVEL_ERROR,              /* 功能发生错误，本模块无法继续正常执行 */
    LEVEL_CRITICAL,           /* 系统级严重错误，需要立即强制停止程序执行，无法屏蔽 */
};

class ErrorInfosManager {
public:
    static void LogErrorInfo(DebuggerErrLevel level, DebuggerErrno errId, const std::string& info);
    static DebuggerErrLevel GetTopErrLevelInDuration();
    static void SetLogThreshold(DebuggerErrLevel t) { threshold = t; }
private:
    static DebuggerErrLevel topLevel;
    static DebuggerErrLevel threshold;
};

inline void CleanErrorInfoCache()
{
    ErrorInfosManager::GetTopErrLevelInDuration();
}

#ifdef __DEBUG__

#define SOURCE_CODE_INFO \
    ("[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " @ " + std::string(__FUNCTION__) + "]:")
#define LOG_CRITICAL(errid, msg) \
    ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_CRITICAL, errid, SOURCE_CODE_INFO + (msg))
#define LOG_ERROR(errid, msg) \
    ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_ERROR, errid, SOURCE_CODE_INFO + (msg))
#define LOG_WARNING(errid, msg) \
    ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_WARNING, errid, SOURCE_CODE_INFO + (msg))
#define LOG_INFO(msg) \
    ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_INFO, DebuggerErrno::NONE, SOURCE_CODE_INFO + (msg))
#define LOG_DEBUG(msg) \
    ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_DEBUG, DebuggerErrno::NONE, SOURCE_CODE_INFO + (msg))
#define DEBUG_FUNC_TRACE() \
    ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_DEBUG, DebuggerErrno::NONE, \
                                    "TRACE: enter " + std::string(__FUNCTION__))

#else

#define LOG_CRITICAL(errid, msg) ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_CRITICAL, errid, msg)
#define LOG_ERROR(errid, msg) ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_ERROR, errid, msg)
#define LOG_WARNING(errid, msg) ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_WARNING, errid, msg)
#define LOG_INFO(msg) ErrorInfosManager::LogErrorInfo(DebuggerErrLevel::LEVEL_INFO, DebuggerErrno::NONE, msg)
#define LOG_DEBUG(msg)
#define DEBUG_FUNC_TRACE()

#endif

}