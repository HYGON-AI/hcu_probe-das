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