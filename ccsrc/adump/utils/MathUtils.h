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
#include <string>

namespace MindStudioDebugger {
namespace  MathUtils {

template <typename T>
T Gcd(T a, T b)
{
    if (a == 0 || b == 0) {
        return 0;
    }
    T c = b;
    while (a % b != 0) {
        c = a % b;
        a = b;
        b = c;
    }
    return c;
}

template <typename T>
T Lcm(T a, T b)
{
    if (a == 0 || b == 0) {
        return 0;
    }
    T ret = (a * b) / (Gcd(a, b));
    return ret;
}

template <typename T>
T DivCeil(T v, T divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (v + divisor - 1) / divisor;
}

template <typename T>
T AlignCeil(T v, T block)
{
    return DivCeil(v, block) * block;
}

float Random();
float Random(float floor, float ceil);
int32_t RandomInt(int32_t floor, int32_t ceil);
std::string RandomString(uint32_t len, char min = ' ', char max = '~');

std::string CalculateMD5(const uint8_t* data, size_t length);

}
}