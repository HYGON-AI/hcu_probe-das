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