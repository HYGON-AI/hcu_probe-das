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

#include "include/ExtArgs.h"

namespace MindStudioDebugger {

class MindSporeTrigger {
public:
    static void TriggerOnStepBegin(uint32_t device, uint32_t curStep, ExtArgs& args);
    static void TriggerOnStepEnd(ExtArgs& args);
    static void LaunchPreDbg() {}
    static void LaunchPostDbg() {}

private:
    MindSporeTrigger() = default;
    ~MindSporeTrigger() = default;

    static bool stepBeginFlag;
};

}