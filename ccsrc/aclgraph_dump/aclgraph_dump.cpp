/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
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
 * ------------------------------------------------------------------------- */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <pthread.h>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "torch_npu/csrc/npu/Stream.h"

using Callback = std::function<void(void)>;

struct ThreadArgs {
    ThreadArgs(aclrtContext context, bool exitFlag)
        : context(context), exitFlag(exitFlag) {}

    aclrtContext context;
    bool exitFlag;
};

constexpr int processReportTimeout = 1800;
static ThreadArgs* threadArgs = nullptr;
static pthread_t threadId = -1;
static std::unordered_set<aclrtStream> subscribed_stream;

namespace {
static std::atomic<uint64_t> serial_num{0};
static constexpr const char* kAclRuntimeInitError =
    "ACL runtime not initialized (no current context). Ensure NPU backend is initialized before calling acl_save.";

static void check_acl(aclError err, const char* msg) {
    if (err != ACL_ERROR_NONE) {
        std::ostringstream oss;
        oss << msg << " (aclError=" << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

static std::string build_final_path(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    size_t dot_pos = filename.find_last_of('.');
    std::string base = (dot_pos == std::string::npos) ? filename : filename.substr(0, dot_pos);
    const uint64_t seq = serial_num.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream oss_name;
    oss_name << base << "_" << seq << ".pt";
    if (last_slash == std::string::npos) {
        return oss_name.str();
    }
    return path.substr(0, last_slash + 1) + oss_name.str();
}

static void ensure_acl_runtime_initialized() {
    aclrtContext ctx = nullptr;
    aclError err = aclrtGetCurrentContext(&ctx);
    if (err != ACL_ERROR_NONE || ctx == nullptr) {
        throw std::runtime_error(kAclRuntimeInitError);
    }
}

static void write_pt_or_throw(const at::Tensor& tensor, const std::string& path) {
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        std::ostringstream oss;
        oss << "Failed to open tensor save file: " << path;
        throw std::runtime_error(oss.str());
    }

    auto ivalue = torch::jit::IValue(tensor);
    auto data = torch::pickle_save(ivalue);
    ofs.write(data.data(), data.size());
    if (!ofs.good()) {
        std::ostringstream oss;
        oss << "Failed to save tensor to: " << path;
        throw std::runtime_error(oss.str());
    }

    ofs.close();
    if (!ofs) {
        std::ostringstream oss;
        oss << "Failed to close file after write: " << path;
        throw std::runtime_error(oss.str());
    }
}

void AclrtLaunchCallback(void* user_data) {
    Callback* callback_func = reinterpret_cast<Callback*>(user_data);
    (*callback_func)();
    delete callback_func;
}

static void acl_save_callback(aclrtStream /*stream*/, const at::Tensor& x_dev_c, const std::string& path) {
    at::Tensor xc = x_dev_c.is_contiguous() ? x_dev_c : x_dev_c.contiguous();
    auto out = at::empty_like(
        xc,
        xc.options().device(at::kCPU),
        at::MemoryFormat::Contiguous);
    const size_t nbytes =
        static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
    if (nbytes == 0) {
        write_pt_or_throw(out, path);
        return;
    }
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    aclmdlRICaptureThreadExchangeMode(&mode);
    auto memcpy_status = aclrtMemcpy(
        out.data_ptr(),
        nbytes,
        xc.data_ptr(),
        nbytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    aclmdlRICaptureThreadExchangeMode(&mode);
    if (memcpy_status != ACL_ERROR_NONE) {
        std::cout << "Memcpy failed with error: " << static_cast<int>(memcpy_status) << std::endl;
        return;
    }
    try {
        write_pt_or_throw(out, path);
    } catch (...) {
        std::cout << "Failed to write pt file." << std::endl;
    }
}

static aclError launch_blocking_callback(aclrtStream stream, aclrtCallback callback, void* user_data) {
    return aclrtLaunchCallback(callback, user_data, ACL_CALLBACK_BLOCK, stream);
}

static at::Tensor copy_to_cpu(const at::Tensor& x) {
    auto out = at::empty_like(
        x,
        x.options().device(at::kCPU),
        at::MemoryFormat::Contiguous);

    const size_t nbytes =
        static_cast<size_t>(x.numel()) * static_cast<size_t>(x.element_size());
    if (nbytes == 0) {
        return out;
    }

    const auto dev_type = x.device().type();

    if (dev_type == at::DeviceType::CPU) {
        at::Tensor xc = x.contiguous();
        std::memcpy(out.data_ptr(), xc.const_data_ptr(), nbytes);
        return out;
    }
    return x.to(at::kCPU, /*non_blocking=*/false).contiguous();
}

void* process_callback(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    auto ret = aclrtSetCurrentContext(args->context);
    (void)ret;
    while (!args->exitFlag) {
        (void)aclrtProcessReport(processReportTimeout);
    }
    delete args;
    args = nullptr;
    return nullptr;
}

static at::Tensor acl_save_impl(const at::Tensor& x, const std::string& path) {
    const auto dev_type = x.device().type();
    const std::string final_path = build_final_path(path);
    if (dev_type != at::DeviceType::PrivateUse1) {
        at::Tensor out = copy_to_cpu(x);
        write_pt_or_throw(out, final_path);
        return out;
    }

    ensure_acl_runtime_initialized();
    auto stream = c10_npu::getCurrentNPUStream().stream();
    if (subscribed_stream.find(stream) == subscribed_stream.end()) {
        aclrtContext context;
        check_acl(aclrtGetCurrentContext(&context), "aclrtGetCurrentContext failed");

        if ((threadArgs == nullptr) || (threadId == -1)) {
            threadArgs = new ThreadArgs(context, false);
            pthread_create(&threadId, nullptr, process_callback, threadArgs);
        }
        aclrtSubscribeReport(threadId, stream);
        subscribed_stream.insert(stream);
    }

    auto callback_func = [stream, x, final_path]() {
        acl_save_callback(stream, x, final_path);
    };
    auto callback_func_ptr = new Callback(callback_func);
    auto cb_status = launch_blocking_callback(stream, AclrtLaunchCallback, callback_func_ptr);
    if (cb_status != ACL_ERROR_NONE) {
        delete callback_func_ptr;
        check_acl(cb_status, "aclrtLaunchCallback failed");
    }
    return x;
}

static at::Tensor acl_save_meta(const at::Tensor& x, const std::string& /*path*/) {
    return at::empty_like(x, x.options().device(at::kMeta));
}

} // namespace

TORCH_LIBRARY(my_ns, m) {
    m.def("acl_save(Tensor x, str path) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ns, Meta, m) {
    m.impl("acl_save", acl_save_meta);
}

TORCH_LIBRARY_IMPL(my_ns, CPU, m) {
    m.impl("acl_save", acl_save_impl);
}

TORCH_LIBRARY_IMPL(my_ns, PrivateUse1, m) {
    m.impl("acl_save", acl_save_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "aclgraph_dump_ext: NPU->CPU memcpy via aclrtMemcpy (+ optional pt dump), Meta+CPU+NPU";
}
