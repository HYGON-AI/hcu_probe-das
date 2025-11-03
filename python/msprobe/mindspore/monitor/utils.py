# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from mindspore import dtype as mstype, Tensor

from msprobe.mindspore.monitor.features import FUNC_MAP, cal_entropy, cal_stable_rank


def get_single_metrics(op_list, tag, tensor, eps=1e-8, output=None):
    if output is None:
        output = {}
    if tag not in output:
        output[tag] = {}
    for op in op_list:
        func = FUNC_MAP.get(op)
        if op == "zeros":
            statistic = func(tensor, eps)
        else:
            statistic = func(tensor)
        if hasattr(statistic, "dtype") and statistic.dtype == mstype.bfloat16:
            statistic = float(statistic)
            statistic = Tensor(statistic)
        if isinstance(statistic, Tensor):
            output[tag][op] = statistic.astype(mstype.float32)
        else:
            output[tag][op] = statistic


def get_metrics(op_list, tag2tensor, eps, output=None):
    if output is None:
        output = {}
    for tag, tensor in tag2tensor.items():
        if tag not in output:
            output[tag] = {}
        get_single_metrics(op_list, tag, tensor, eps, output)
    return output


def get_summary_writer_tag_name(module_or_param_name: str, tag: str, rank):
    if rank is None:
        return f"{module_or_param_name}/{tag}"
    else:
        return f"{module_or_param_name}/rank{rank}/{tag}"


def step_accumulates_one(context, micro_batch_number):
    """
    :param context: ModuleHookContext
    :param micro_batch_number: mbs of training model.
    :return:
    """
    context.micro_step += 1
    if context.micro_step == micro_batch_number:
        context.micro_step = 0
        context.step += 1


def is_skip_step(step, start_step, step_interval, has_collect_times=0, collect_times=1e8):
    """
    If current step less than start_step or not reach step_interval, skip current step.
    :param step: current training step, int
    :param start_step: int
    :param step_interval: int
    :return: whether skip or not, bool
    """
    return step < start_step or (step - start_step) % step_interval != 0 or has_collect_times >= collect_times


def get_entropy_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if tag not in out_dict:
            out_dict[tag] = {}
        entropy, softmax = cal_entropy(tensor)
        out_dict[tag]["entropy"] = entropy
        out_dict[tag]["softmax"] = softmax
    return out_dict


def get_sr_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if "sr" not in tag:
            continue
        if tag not in out_dict:
            out_dict[tag] = {}
        sr, eig = cal_stable_rank(tensor)
        out_dict[tag]["sr"] = sr
        out_dict[tag]["eig"] = eig
    return out_dict
