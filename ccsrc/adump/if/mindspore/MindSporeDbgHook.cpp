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

#define _GLIBCXX_USE_CXX11_ABI 0

#include <string>
#include <vector>

#include "include/Macro.h"
#include "include/ExtArgs.h"
#include "core/mindspore/MindSporeTrigger.h"

EXPORT_SYMBOL void MS_DbgOnStepBegin(uint32_t device, int32_t curStep,
                                     std::map<uint32_t, void*> exts)
{
    MindStudioDebugger::ExtArgs args;
    const char** strBuf = nullptr;
    for (auto& ext : exts) {
        if (ext.first >= static_cast<uint32_t>(MindStudioDebugger::MindStudioExtensionArgs::ARG_MAX)) {
            continue;
        }
        /* mindspore使用了_GLIBCXX_USE_CXX11_ABI=0，为了解决CXX版本兼容问题，此处将string转char*使用 */
        if (ext.first == static_cast<uint32_t>(MindStudioDebugger::MindStudioExtensionArgs::ALL_KERNEL_NAMES)) {
            if (ext.second == nullptr) {
                continue;
            }
            std::vector<std::string>* ss = reinterpret_cast<std::vector<std::string>*>(ext.second);
            strBuf = new const char* [(*ss).size() + 1];
            strBuf[(*ss).size()] = nullptr;
            size_t i = 0;
            for (std::string& s : *ss) {
                strBuf[i] = s.c_str();
                i++;
            }
            args[static_cast<MindStudioDebugger::MindStudioExtensionArgs>(ext.first)] = reinterpret_cast<void*>(strBuf);
            continue;
        }
        args[static_cast<MindStudioDebugger::MindStudioExtensionArgs>(ext.first)] = ext.second;
    }

    MindStudioDebugger::MindSporeTrigger::TriggerOnStepBegin(device, static_cast<uint32_t>(curStep), args);
    if (strBuf != nullptr) {
        delete[] strBuf;
    }

    return;
}

EXPORT_SYMBOL void MS_DbgOnStepEnd(std::map<uint32_t, void*>& exts)
{
    MindStudioDebugger::ExtArgs args;
    for (auto& ext : exts) {
        if (ext.first >= static_cast<uint32_t>(MindStudioDebugger::MindStudioExtensionArgs::ARG_MAX)) {
            continue;
        }
        args[static_cast<MindStudioDebugger::MindStudioExtensionArgs>(ext.first)] = ext.second;
    }
    return MindStudioDebugger::MindSporeTrigger::TriggerOnStepEnd(args);
}