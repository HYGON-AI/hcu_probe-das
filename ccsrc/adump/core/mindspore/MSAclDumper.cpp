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