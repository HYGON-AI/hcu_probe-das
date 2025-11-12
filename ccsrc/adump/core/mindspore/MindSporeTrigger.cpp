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

#include "include/Macro.h"
#include "base/ErrorInfosManager.h"
#include "MSAclDumper.h"
#include "MindSporeTrigger.h"

namespace MindStudioDebugger {

bool MindSporeTrigger::stepBeginFlag = false;

void MindSporeTrigger::TriggerOnStepBegin(uint32_t device, uint32_t /* curStep */, ExtArgs& args)
{
    DEBUG_FUNC_TRACE();
    CleanErrorInfoCache();
    
    MSAclDumper::GetInstance().OnStepBegin(device, args);

    stepBeginFlag = true;

    CleanErrorInfoCache();
    return;
}

void MindSporeTrigger::TriggerOnStepEnd(ExtArgs& args)
{
    DEBUG_FUNC_TRACE();
    CleanErrorInfoCache();

    if (!stepBeginFlag) {
        return;
    }
    MSAclDumper::GetInstance().OnStepEnd(args);
    stepBeginFlag = false;

    CleanErrorInfoCache();
    return;
}

}