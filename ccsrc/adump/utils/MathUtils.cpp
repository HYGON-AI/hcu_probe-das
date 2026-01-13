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