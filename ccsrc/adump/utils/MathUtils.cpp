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

#include <cstdint>
#include <random>
#include "openssl/md5.h"

namespace MindStudioDebugger {
namespace  MathUtils {

float Random()
{
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

float Random(float floor, float ceil)
{
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(floor, ceil);
    return distribution(generator);
}

int32_t RandomInt(int32_t floor, int32_t ceil)
{
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int32_t> distribution(floor, ceil - 1);

    return distribution(generator);
}

std::string RandomString(uint32_t len, char min, char max)
{
    std::mt19937 generator(std::random_device{}());
    std::string output(len, '\0');
    if (min > max) {
        return output;
    }

    std::uniform_int_distribution<char> distribution(min, max);
    for (uint32_t i = 0; i < len; i++) {
        output[i] = distribution(generator);
    }

    return output;
}

std::string CalculateMD5(const uint8_t* data, size_t length)
{
    MD5_CTX md5ctx;
    /*
     * 不用于数据加密，不用于文件完整性校验，不用于密码存储，不用于数据唯一性检查
     * 只用于Tensor的统计信息呈现，不涉及数据安全
    */
    MD5_Init(&md5ctx);
    MD5_Update(&md5ctx, data, length);

    unsigned char digest[MD5_DIGEST_LENGTH];
    /*
     * 不用于数据加密，不用于文件完整性校验，不用于密码存储，不用于数据唯一性检查
     * 只用于Tensor的统计信息呈现，不涉及数据安全
    */
    MD5_Final(digest, &md5ctx);

    static const char HEX_CHAR[] = "0123456789abcdef";
    constexpr const uint8_t hexbase = 16;
    constexpr const size_t byteToStrWidth = 2;
    char md5string[MD5_DIGEST_LENGTH * byteToStrWidth + 1];
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        md5string[i * byteToStrWidth] = HEX_CHAR[digest[i] / hexbase];
        md5string[i * byteToStrWidth + 1] = HEX_CHAR[digest[i] % hexbase];
    }
    md5string[sizeof(md5string) - 1] = '\0';

    return std::string(md5string);
}

}
}