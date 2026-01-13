/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
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
