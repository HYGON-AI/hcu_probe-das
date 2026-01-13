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