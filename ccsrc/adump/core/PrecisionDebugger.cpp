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