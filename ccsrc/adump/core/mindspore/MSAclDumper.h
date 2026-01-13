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