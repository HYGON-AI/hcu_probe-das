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

#include <cstdint>
#include <vector>

#include "base/DebuggerConfig.h"

namespace MindStudioDebugger {

class PrecisionDbgTaskBase {
public:
    virtual bool Condition(const DebuggerConfig& cfg) const = 0;
    virtual std::string Name() const = 0;
    virtual uint32_t Priority() const {return 100;}

    virtual void Initialize(const DebuggerConfig& cfg) {};
    virtual void OnStart() {};
    virtual void OnStop() {};
    virtual void OnStep(uint32_t curStep) {};

    void Register();

protected:
    PrecisionDbgTaskBase() = default;
    ~PrecisionDbgTaskBase() = default;
};

class PrecisionDebugger {
public:
    static PrecisionDebugger& GetInstance()
    {
        static PrecisionDebugger debuggerInstance;
        return debuggerInstance;
    }

    int32_t Initialize(const std::string& framework, const std::string& cfgFile);
    bool HasInitialized() const {return initialized;}

    void Start();
    void Stop();
    void Step();
    void Step(uint32_t step);

    bool IsEnable() const {return enable;}
    uint32_t GetCurStep() const {return curStep;}

    void RegisterDebuggerTask(PrecisionDbgTaskBase* task);
    void UnRegisterDebuggerTask(PrecisionDbgTaskBase* task);

private:
    PrecisionDebugger() = default;
    ~PrecisionDebugger() = default;
    explicit PrecisionDebugger(const PrecisionDebugger &obj) = delete;
    PrecisionDebugger& operator=(const PrecisionDebugger &obj) = delete;
    explicit PrecisionDebugger(PrecisionDebugger &&obj) = delete;
    PrecisionDebugger& operator=(PrecisionDebugger &&obj) = delete;

    bool initialized{false};
    bool enable{false};
    uint32_t curStep{0};
    std::vector<PrecisionDbgTaskBase*> subDebuggers;
};

}