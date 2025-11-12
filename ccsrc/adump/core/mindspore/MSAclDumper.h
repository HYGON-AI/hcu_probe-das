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
#include "core/PrecisionDebugger.h"

namespace MindStudioDebugger {

class MSAclDumper : public PrecisionDbgTaskBase {
public:
    static MSAclDumper& GetInstance()
    {
        static MSAclDumper dumperInstance;
        return dumperInstance;
    }

    std::string Name() const override {return "MindSpore AclDumper";}
    bool Condition(const DebuggerConfig& cfg) const override
    {
        return cfg.GetFramework() == DebuggerFramework::FRAMEWORK_MINDSPORE &&
               cfg.GetDebugLevel() == DebuggerLevel::L2;
    }

    void OnStepBegin(uint32_t device, ExtArgs& args);
    void OnStepEnd(ExtArgs& args);
    void Step();

private:
    MSAclDumper() = default;
    ~MSAclDumper() = default;
    explicit MSAclDumper(const MSAclDumper &obj) = delete;
    MSAclDumper& operator=(const MSAclDumper &obj) = delete;
    explicit MSAclDumper(MSAclDumper &&obj) = delete;
    MSAclDumper& operator=(MSAclDumper &&obj) = delete;
    uint32_t msprobeStep{0};
};

}