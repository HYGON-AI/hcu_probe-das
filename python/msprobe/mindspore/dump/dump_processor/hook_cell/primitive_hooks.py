# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import threading

from mindspore import ops
from mindspore.common.tensor import Tensor
from msprobe.core.common.utils import Const, DumpException, ThreadSafe
from msprobe.core.dump.data_dump.data_processor.base import (
    ModuleBackwardInputs,
    ModuleBackwardOutputs,
    ModuleForwardInputsOutputs
)
from msprobe.core.dump.hook_manager import BaseHookManager
from msprobe.mindspore.common.log import logger


class PrimitiveHookService:
    def __init__(self, service_instance):
        self.primitive_counters = {}
        self.service_instance = service_instance

    def wrap_primitive(self, origin_func, primitive_name):
        """
        包装原始的 primitive 函数，添加输入和输出的 hook 以捕获前向和反向数据。

        Args:
            origin_func (callable): 原始 的 primitive 函数。
            primitive_name (str): 原始的 primitive 名称。

        Returns:
            callable: 包装后的 primitive 函数。
        """

        def create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type):
            """
            创建反向 hook 函数，用于捕获梯度。

            Args:
                captured_grads (list): 用于保存捕获的梯度。
                num_tensors (int): 张量数量。
                updated_primitive_name (str): 更新后的 primitive 名称。
                hook_type (str): hook 类型 (输入/输出)。

            Returns:
                callable: 反向 hook 函数。
            """

            @ThreadSafe.synchronized
            def backward_hook(grad):
                tid = threading.get_ident()
                BaseHookManager.inner_switch[tid] = True

                captured_grads.extend(grad)
                backward_primitive_name = f"{updated_primitive_name}{Const.SEP}{Const.BACKWARD}"
                try:
                    if hook_type == Const.INPUT:
                        self.service_instance.data_collector.update_api_or_module_name(backward_primitive_name)
                        new_module_input_output = ModuleBackwardOutputs(grad_output=tuple(captured_grads))
                        self.service_instance.data_collector.backward_output_data_collect(
                            backward_primitive_name, self, os.getpid(), new_module_input_output
                        )
                    elif hook_type == Const.OUTPUT:
                        self.service_instance.data_collector.update_api_or_module_name(backward_primitive_name)
                        new_module_input_output = ModuleBackwardInputs(grad_input=tuple(captured_grads))
                        self.service_instance.data_collector.backward_input_data_collect(
                            backward_primitive_name, self, os.getpid(), new_module_input_output
                        )

                except Exception as exception:
                    logger.error(f"This is a primitive op {hook_type}_backward dump error: {exception}, "
                                 f"updated_primitive_name: {updated_primitive_name}")
                    raise DumpException(DumpException.BACKWARD_DATA_COLLECTION_ERROR) from exception
                BaseHookManager.inner_switch[tid] = False

            return backward_hook

        def hook_primitive_inputs(args, captured_grads_input, updated_primitive_name):
            """
            针对前向输入添加 hook。

            Args:
                args (tuple): primitive 输入参数。
                captured_grads_input (list): 捕获的输入梯度。
                updated_primitive_name (str): 更新后的 primitive 名称。

            Returns:
                list: 添加了 hook 的输入。
            """
            hooked_inputs = []
            num_tensors = sum(isinstance(arg, Tensor) for arg in args)
            input_backward_hook = create_backward_hook(captured_grads_input, num_tensors, updated_primitive_name,
                                                       Const.INPUT)
            for arg in args:
                if isinstance(arg, Tensor):
                    arg_hooked = ops.HookBackward(input_backward_hook)(arg)
                    hooked_inputs.append(arg_hooked)
                else:
                    hooked_inputs.append(arg)
            return tuple(hooked_inputs)

        def hook_primitive_outputs(out, captured_grads_output, updated_primitive_name):
            """
            针对前向输出添加 hook。

            Args:
                out (Tensor/tuple): primitive 输出。
                captured_grads_output (list): 捕获的输出梯度。
                updated_primitive_name (str): 更新后的 primitive 名称。

            Returns:
                Tensor/tuple: 添加了 hook 的输出。
            """
            if isinstance(out, tuple):
                num_output_tensors = sum(isinstance(tensor, Tensor) for tensor in out)
            else:
                num_output_tensors = 1
            output_backward_hook = create_backward_hook(captured_grads_output, num_output_tensors,
                                                        updated_primitive_name, Const.OUTPUT)

            if isinstance(out, Tensor):
                return ops.HookBackward(output_backward_hook)(out)
            elif isinstance(out, tuple):
                hooked_outputs = []
                for tensor in out:
                    if isinstance(tensor, Tensor):
                        hooked_outputs.append(ops.HookBackward(output_backward_hook)(tensor))
                    else:
                        hooked_outputs.append(tensor)
                return tuple(hooked_outputs)
            return out

        @ThreadSafe.synchronized
        def pre_forward_hook(primitive_name, primitive_instance, args, kwargs):
            tid = threading.get_ident()
            BaseHookManager.inner_switch[tid] = True

            self.service_instance.data_collector.update_api_or_module_name(primitive_name)
            module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
            try:
                self.service_instance.data_collector.forward_input_data_collect(
                    primitive_name,
                    primitive_instance,
                    os.getpid(),
                    module_input_output
                )
            except Exception as exception:
                logger.error(f"This is a primitive op dump error during forward input data collection: {exception}, "
                             f"primitive_name: {primitive_name}")
                raise DumpException(DumpException.FORWARD_DATA_COLLECTION_ERROR) from exception
            BaseHookManager.inner_switch[tid] = False

        @ThreadSafe.synchronized
        def post_forward_hook(primitive_name, primitive_instance, args, kwargs, output):
            tid = threading.get_ident()
            BaseHookManager.inner_switch[tid] = True

            self.service_instance.data_collector.update_api_or_module_name(primitive_name)
            module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
            try:
                self.service_instance.data_collector.forward_output_data_collect(
                    primitive_name,
                    primitive_instance,
                    os.getpid(),
                    module_input_output
                )
            except Exception as exception:
                logger.error(f"This is a primitive op dump error during forward output data collection: {exception}, "
                             f"primitive_name: {primitive_name}")
                raise DumpException(DumpException.FORWARD_DATA_COLLECTION_ERROR) from exception
            BaseHookManager.inner_switch[tid] = False

        def wrapped_primitive_call(instance_self, *args, **kwargs):
            """
            包装后的 primitive 调用函数，添加输入和输出的 hook。

            Args:
                instance_self (object): primitive 的实例。
                *args: primitive 输入参数。
                **kwargs: primitive 关键字参数。

            Returns:
                Tensor/tuple: primitive 的返回值。
            """
            self.update_primitive_counters(primitive_name)
            current_count = self.primitive_counters.get(primitive_name, 0)
            updated_primitive_name = f"{Const.PRIMITIVE_PREFIX}{Const.SEP}{primitive_name}{Const.SEP}{current_count}"

            tid = threading.get_ident()
            if not self.service_instance.primitive_switch or BaseHookManager.inner_switch[tid]:
                return origin_func(*args, **kwargs)

            captured_grads_input, captured_grads_output = [], []

            try:
                hooked_inputs = hook_primitive_inputs(args, captured_grads_input, updated_primitive_name)
            except Exception as exception:
                logger.error(f"This is a primitive op dump error during input hooking: {exception}, "
                             f"primitive_name: {primitive_name}")
                raise DumpException(DumpException.INPUT_HOOK_ERROR) from exception

            forward_primitive_name = f"{updated_primitive_name}{Const.SEP}{Const.FORWARD}"
            pre_forward_hook(forward_primitive_name, instance_self, hooked_inputs, kwargs)
            try:
                out = origin_func(*hooked_inputs, **kwargs)
            except Exception as exception:
                logger.error(f"This is a primitive op dump error during function call: {exception}, "
                             f"primitive_name: {primitive_name}")
                raise DumpException(DumpException.FUNCTION_CALL_ERROR) from exception
            post_forward_hook(forward_primitive_name, instance_self, hooked_inputs, kwargs, out)

            try:
                out = hook_primitive_outputs(out, captured_grads_output, updated_primitive_name)
            except Exception as exception:
                logger.error(f"This is a primitive op dump error during output hooking: {exception}, "
                             f"primitive_name: {primitive_name}")
                raise DumpException(DumpException.OUTPUT_HOOK_ERROR) from exception

            return out

        return wrapped_primitive_call

    @ThreadSafe.synchronized
    def update_primitive_counters(self, primitive_name):
        if primitive_name not in self.primitive_counters:
            self.primitive_counters[primitive_name] = 0
        else:
            self.primitive_counters[primitive_name] += 1
