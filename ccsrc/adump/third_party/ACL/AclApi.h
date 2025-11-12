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

#include <cstdint>

#include "include/ErrorCode.h"

extern "C" {
using aclError = int;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_ERROR_NONE = 0;
constexpr int ACL_ERROR_REPEAT_INITIALIZE = 100002;

#define ACL_DUMP_MAX_FILE_PATH_LENGTH 4096
typedef struct AclDumpChunk  {
    char       fileName[ACL_DUMP_MAX_FILE_PATH_LENGTH];    // 待落盘的Dump数据文件名，ACL_DUMP_MAX_FILE_PATH_LENGTH表示文件名最大长度，当前为4096
    uint32_t   bufLen;                                     // dataBuf数据长度，单位Byte
    uint32_t   isLastChunk;                                // 标识Dump数据是否为最后一个分片，0表示不是最后一个分片，1表示最后一个分片
    int64_t    offset;                                     // Dump数据文件内容的偏移，其中-1表示文件追加内容
    int32_t    flag;                                       // 预留Dump数据标识，当前数据无标识
    uint8_t    dataBuf[0];                                 // Dump数据的内存地址
} AclDumpChunk;
}

namespace MindStudioDebugger {
namespace AscendCLApi {

DebuggerErrno LoadAclApi();

using AclDumpCallbackFuncType = int32_t (*)(const AclDumpChunk*, int32_t);
aclError AclApiAclInit(const char* cfg);
aclError AclApiAclmdlInitDump();
aclError AclApiAclmdlSetDump(const char* cfg);
aclError AclApiAclmdlFinalizeDump();
aclError AclApiAcldumpRegCallback(AclDumpCallbackFuncType messageCallback, int32_t flag);

aclError AclApiAclrtSynchronizeDevice();

#define CALL_ACL_API(func, ...) MindStudioDebugger::AscendCLApi::AclApi##func(__VA_ARGS__)

}
}
