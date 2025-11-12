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

#include <cstring>

#include "base/ErrorInfosManager.h"
#include "base/DebuggerConfig.h"
#include "third_party/ACL/AclApi.h"
#include "core/mindspore/MSAclDumper.h"
#include "PrecisionDebugger.h"

namespace MindStudioDebugger {

void PrecisionDbgTaskBase::Register()
{
    PrecisionDebugger::GetInstance().RegisterDebuggerTask(this);
}

void PrecisionDebugger::RegisterDebuggerTask(PrecisionDbgTaskBase* task)
{
    DEBUG_FUNC_TRACE();
    std::vector<PrecisionDbgTaskBase *>::iterator iter;
    const DebuggerConfig& cfg = DebuggerConfig::GetInstance();

    if (cfg.IsCfgLoaded() && !task->Condition(cfg)) {
        return;
    }

    for (iter = subDebuggers.begin(); iter != subDebuggers.end(); ++iter) {
        if (*iter == task) {
            return;
        }
    }

    for (iter = subDebuggers.begin(); iter != subDebuggers.end(); ++iter) {
        if ((*iter)->Priority() > task->Priority()) {
            break;
        }
    }

    subDebuggers.insert(iter, task);

    if (cfg.IsCfgLoaded()) {
        /* 如果配置还没加载，先加入到缓存中，等加载时再根据条件过滤一遍 */
        task->Initialize(cfg);
        LOG_DEBUG("PrecisionDebugger: " + task->Name() + " registered.");
    }
    return;
}

void PrecisionDebugger::UnRegisterDebuggerTask(PrecisionDbgTaskBase* task)
{
    DEBUG_FUNC_TRACE();
    for (auto iter = subDebuggers.begin(); iter != subDebuggers.end(); iter++) {
        if (*iter == task) {
            LOG_DEBUG("PrecisionDebugger: " + task->Name() + " unregistered.");
            subDebuggers.erase(iter);
            return;
        }
    }

    return;
}

int32_t PrecisionDebugger::Initialize(const std::string& framework, const std::string& cfgFile)
{
    DEBUG_FUNC_TRACE();

    int32_t ret = DebuggerConfig::GetInstance().LoadConfig(framework, cfgFile);
    if (ret != 0) {
        return ret;
    }

    if (AscendCLApi::LoadAclApi() != DebuggerErrno::OK) {
        return -1;
    }

    const DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    for (auto iter = subDebuggers.begin(); iter != subDebuggers.end();) {
        if (!(*iter)->Condition(cfg)) {
            iter = subDebuggers.erase(iter);
        } else {
            (*iter)->Initialize(cfg);
            LOG_DEBUG("PrecisionDebugger: " + (*iter)->Name() + " registered.");
            iter++;
        }
    }

    initialized = true;
    return 0;
}

void PrecisionDebugger::Start()
{
    DEBUG_FUNC_TRACE();
    if (!initialized) {
        return;
    }

    enable = true;

    for (auto task : subDebuggers) {
        task->OnStart();
    }
}

void PrecisionDebugger::Stop()
{
    DEBUG_FUNC_TRACE();
    if (!initialized) {
        return;
    }

    enable = false;
    CALL_ACL_API(AclrtSynchronizeDevice);

    for (auto task : subDebuggers) {
        task->OnStop();
    }
}

void PrecisionDebugger::Step()
{
    MSAclDumper::GetInstance().Step();
}

}