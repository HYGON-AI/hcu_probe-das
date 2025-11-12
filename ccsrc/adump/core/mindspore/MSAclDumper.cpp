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

#include <cstdlib>

#include "base/ErrorInfosManager.h"
#include "base/DebuggerConfig.h"
#include "base/Environment.h"
#include "core/AclDumper.h"
#include "MSAclDumper.h"

namespace MindStudioDebugger {

void MSAclDumper::OnStepBegin(uint32_t device, ExtArgs& args)
{
    DEBUG_FUNC_TRACE();
    if (!PrecisionDebugger::GetInstance().IsEnable()) {
        return;
    }
    const bool* isKbk = GetExtArgs<bool*>(args, MindStudioExtensionArgs::IS_KBK);
    if (isKbk != nullptr && *isKbk) {
        /* acldump只用于非kbk场景 */
        return;
    }

    int32_t rank = Environment::GetRankID();
    if (rank < 0) {
        rank = static_cast<int32_t>(device);
    }

    AclDumper::GetInstance().SetDump(rank, msprobeStep, args);
    return;
}

void MSAclDumper::OnStepEnd(ExtArgs& args)
{
    DEBUG_FUNC_TRACE();
    AclDumper::GetInstance().FinalizeDump(args);
}

void MSAclDumper::Step()
{
    msprobeStep++;
}

__attribute__((constructor)) void RegisterMSAclDumper()
{
    MSAclDumper::GetInstance().Register();
}

}