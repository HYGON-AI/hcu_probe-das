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

#include <dlfcn.h>
#include <map>
#include <stdexcept>

#include "base/ErrorInfosManager.h"
#include "AclApi.h"

namespace MindStudioDebugger {
namespace AscendCLApi {

using namespace MindStudioDebugger;

constexpr const char* LIB_ASCEND_CL_NAME = "libascendcl.so";
constexpr const char* LIB_MS_ASCEND_NAME = "libmindspore_ascend.so.2";
constexpr const char* LIB_ASCEND_DUMP_NAME = "libascend_dump.so";

using AclInitFuncType = aclError (*)(const char *);
using AclmdlInitDumpFuncType = aclError (*)();
using AclmdlSetDumpFuncType = aclError (*)(const char *);
using AclmdlFinalizeDumpFuncType = aclError (*)();
using AcldumpRegCallbackFuncType = aclError (*)(AclDumpCallbackFuncType, int32_t);
using AclrtSynchronizeDeviceFuncType = aclError (*)();

static AclInitFuncType g_aclInitFunc = nullptr;
static AclmdlInitDumpFuncType g_aclmdlInitDumpFunc = nullptr;
static AclmdlSetDumpFuncType g_aclmdlSetDumpFunc = nullptr;
static AclmdlFinalizeDumpFuncType g_aclmdlFinalizeDumpFunc = nullptr;
static AcldumpRegCallbackFuncType g_acldumpRegCallbackFunc = nullptr;
static AcldumpRegCallbackFuncType g_acldumpRegCallbackFuncInSo = nullptr;
static AclrtSynchronizeDeviceFuncType g_aclrtSynchronizeDeviceFunc = nullptr;

static const std::map<const char*, void**> functionMap = {
    {"aclInit", reinterpret_cast<void**>(&g_aclInitFunc)},
    {"aclmdlInitDump", reinterpret_cast<void**>(&g_aclmdlInitDumpFunc)},
    {"aclmdlSetDump", reinterpret_cast<void**>(&g_aclmdlSetDumpFunc)},
    {"aclmdlFinalizeDump", reinterpret_cast<void**>(&g_aclmdlFinalizeDumpFunc)},
    {"aclrtSynchronizeDevice", reinterpret_cast<void**>(&g_aclrtSynchronizeDeviceFunc)},
};

DebuggerErrno LoadAclApi()
{
    static void* hLibAscendcl = nullptr;

    if (hLibAscendcl != nullptr) {
        LOG_INFO("No need to load acl api again.");
        return DebuggerErrno::OK;
    }

    hLibAscendcl = dlopen(LIB_ASCEND_CL_NAME, RTLD_LAZY | RTLD_NOLOAD);
    if (hLibAscendcl == nullptr) {
        LOG_ERROR(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND,
                  "Failed to search libascendcl.so." + std::string(dlerror()));
        return DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND;
    }

    for (auto& iter : functionMap) {
        if (*(iter.second) != nullptr) { continue; }
        *(iter.second) = dlsym(hLibAscendcl, iter.first);
        if (*(iter.second) == nullptr) {
            LOG_ERROR(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "Failed to load function " +
                      std::string(iter.first) + " from libascendcl.so." + std::string(dlerror()));
            dlclose(hLibAscendcl);
            hLibAscendcl = nullptr;
            return DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND;
        }
        LOG_DEBUG("Load function " + std::string(iter.first) + " from libascendcl.so.");
    }

    /* 规避adump的bug，mindspore场景优先使用libmindspore_ascend.so中的符号 */
    void* handler = dlopen(LIB_MS_ASCEND_NAME, RTLD_LAZY | RTLD_NOLOAD);
    std::string libName = LIB_MS_ASCEND_NAME;
    if (handler == nullptr) {
        handler = hLibAscendcl;
        libName = LIB_ASCEND_CL_NAME;
    }

    g_acldumpRegCallbackFunc = reinterpret_cast<AcldumpRegCallbackFuncType>(dlsym(handler, "acldumpRegCallback"));
    if (g_acldumpRegCallbackFunc == nullptr) {
        LOG_WARNING(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "Failed to load function acldumpRegCallback from " +
                    libName + ".");
    }
    LOG_DEBUG("Load function acldumpRegCallback from " + libName + ".");

    if (handler != hLibAscendcl) { dlclose(handler); }

    void* dumpHandler = dlopen(LIB_ASCEND_DUMP_NAME, RTLD_LAZY | RTLD_NOLOAD);
    if (dumpHandler == nullptr) {
        LOG_WARNING(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "Failed to load libascend_dump.so.");
    } else {
        g_acldumpRegCallbackFuncInSo = reinterpret_cast<AcldumpRegCallbackFuncType>(dlsym(dumpHandler, "acldumpRegCallback"));
        if (g_acldumpRegCallbackFuncInSo == nullptr) {
            LOG_WARNING(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND,
                        "Failed to load function acldumpRegCallback from libascend_dump.so.");
        }
        LOG_DEBUG("Load function acldumpRegCallback from libascend_dump.so.");
        dlclose(dumpHandler);
    }

    return DebuggerErrno::OK;
}

aclError AclApiAclInit(const char* cfg)
{
    if (g_aclInitFunc == nullptr) {
        throw std::runtime_error("API aclInit does not have a definition.");
    }
    return g_aclInitFunc(cfg);
}

aclError AclApiAclmdlInitDump()
{
    if (g_aclmdlInitDumpFunc == nullptr) {
        throw std::runtime_error("API aclmdlInitDump does not have a definition.");
    }
    return g_aclmdlInitDumpFunc();
}

aclError AclApiAclmdlSetDump(const char* cfg)
{
    if (g_aclmdlSetDumpFunc == nullptr) {
        throw std::runtime_error("API aclmdlSetDump does not have a definition.");
    }
    return g_aclmdlSetDumpFunc(cfg);
}

aclError AclApiAclmdlFinalizeDump()
{
    if (g_aclmdlFinalizeDumpFunc == nullptr) {
        throw std::runtime_error("API aclmdlFinalizeDump does not have a definition.");
    }
    return g_aclmdlFinalizeDumpFunc();
}

aclError AclApiAcldumpRegCallback(AclDumpCallbackFuncType messageCallback, int32_t flag)
{
    if (g_acldumpRegCallbackFunc == nullptr && g_acldumpRegCallbackFuncInSo == nullptr) {
        throw std::runtime_error("API acldumpRegCallback does not have a definition.");
    }
    aclError staticAclRet = -1;
    aclError dynamicAclRet = -1;
    if (g_acldumpRegCallbackFunc != nullptr) {
        staticAclRet = g_acldumpRegCallbackFunc(messageCallback, flag);
    }
    if (g_acldumpRegCallbackFuncInSo != nullptr) {
        dynamicAclRet = g_acldumpRegCallbackFuncInSo(messageCallback, flag);
    }
    if (staticAclRet != ACL_SUCCESS && dynamicAclRet != ACL_SUCCESS) {
        return dynamicAclRet;
    }
    return ACL_SUCCESS;
}

aclError AclApiAclrtSynchronizeDevice()
{
    if (g_aclrtSynchronizeDeviceFunc == nullptr) {
        throw std::runtime_error("API aclrtSynchronizeDevice does not have a definition.");
    }
    return g_aclrtSynchronizeDeviceFunc();
}

}
}
