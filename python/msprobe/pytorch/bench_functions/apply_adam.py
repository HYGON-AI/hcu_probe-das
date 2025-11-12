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

from collections import namedtuple
import torch


VarParams = namedtuple('VarParams', ['var', 'lr_t', 'm_t', 'beta1_broad', 'grad', 'epsilon', 'v_t'])


def _output_m_compute(m, beta1_broad, grad):
    """
    _output_m_compute
    do compute m_t = m + (beta1 - 1) * (m - grad)
    """
    input_dtype = m.dtype

    sneg_one = torch.ones((1), dtype=input_dtype) * -1
    sneg_one = sneg_one.to(beta1_broad.device)

    # `formula; beta1 -1`
    vsub_beta1_1 = torch.add(beta1_broad, sneg_one)

    # `formula; m - grad`
    vsub_m_grad = torch.sub(m, grad)

    # `formula; (beta1 - 1) * (m - grad)`
    vmul_m = torch.mul(vsub_beta1_1, vsub_m_grad)

    # `formula; m_t = m + (beta1 - 1) * (m - grad)`
    m_t = torch.add(m, vmul_m)

    return m_t


def _output_v_compute(v, beta2, grad):
    """
    _output_v_compute
    do compute v_t = v + (1 - beta2)*(grad*grad -v)
    """
    input_dtype = v.dtype

    sneg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; broadcast beta2 to vector`
    beta2_tensor = torch.tensor(beta2, dtype=input_dtype)
    beta2_broad = beta2_tensor.expand_as(v)

    # `formula; beta2 - 1`
    vsub_beta2_1 = torch.add(beta2_broad, sneg_one)
    vsub_beta2_1 = vsub_beta2_1.to(v.device)

    # `formula; grad * grad`
    vmul_grad_grad = torch.mul(grad, grad)

    # `formula; (v - grad*grad)`
    vsub_v_grad = torch.sub(v, vmul_grad_grad)

    # `formula; (beta2 -1) * (v - grad * grad)`
    vmul_grad = torch.mul(vsub_beta2_1, vsub_v_grad)

    # `formula; v_t = v + (beta2 - 1) * (v - grad * grad)`
    v_t = torch.add(v, vmul_grad)

    return v_t


def _inner_lr_compute(lr, beta2_power, beta1_power, compute_shape_tensor):
    """
    _inner_lr_compute
    `formula; lr_t = learning_rate * (sqrt(1-beta2_power)) / (1 - beta1_power)`
    """

    input_dtype = compute_shape_tensor.dtype

    s_one = torch.ones((1), dtype=input_dtype)

    s_neg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; (1 - beta2_power)`
    v_neg_beta2_power = torch.mul(beta2_power, s_neg_one)
    v_add_beta2_power = torch.add(v_neg_beta2_power, s_one)

    # `formula; sqrt(1 - beta2_power)`
    v_sqrt_beta2_power = torch.sqrt(v_add_beta2_power)

    # `formula; (1 - beta1_power)`
    v_neg_beta1_power = torch.mul(beta1_power, s_neg_one)
    v_add_beta1_power = torch.add(v_neg_beta1_power, s_one)

    # `formula; learning_rate * (sqrt(1-beta2_power)`
    res = torch.mul(lr, v_sqrt_beta2_power)

    # `formula; learning_rate*(sqrt(1-beta2_power))/(1-beta1_power)`
    res = torch.div(res, v_add_beta1_power)
    return res.expand_as(compute_shape_tensor)


def _inner_eps_add_sqrt_vt_compute(epsilon, v_t):
    """
    (epsilon + sqrt(v_t) )
    """
    # `formula; sqrt(v_t)`
    sqrt_vt = torch.sqrt(v_t)

    # `formula; broadcast epsilon  to vector`
    input_dtype = v_t.dtype
    epsilon_tensor = torch.tensor(epsilon, dtype=input_dtype)
    epsilon_broad = epsilon_tensor.expand_as(v_t)
    epsilon_broad = epsilon_broad.to(sqrt_vt.device)

    # `formula; epsilon + sqrt(v_t)`
    v_add_sqrt_v = torch.add(sqrt_vt, epsilon_broad)

    return v_add_sqrt_v


def _output_var_t_compute_use_nesterov(varparams):
    """
    _output_var_t_compute_use_nesterov
    `formula; var_t = var - lr_t * (m_t * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_t))`
    `formula; var_t = var - lr_t * (m_t * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_t))`
    """
    var = varparams.var
    lr_t = varparams.lr_t
    m_t = varparams.m_t
    beta1_broad = varparams.beta1_broad
    grad = varparams.grad
    epsilon = varparams.epsilon
    v_t = varparams.v_t

    input_dtype = var.dtype

    s_one = torch.ones((1), dtype=input_dtype)

    s_neg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; m_t * beta1`
    v_muls_mt_beta1 = torch.mul(m_t, beta1_broad)

    # `formula; 1 -beta1`
    v_neg_beta1 = torch.mul(beta1_broad, s_neg_one)
    vsub_1_beta1 = torch.add(v_neg_beta1, s_one)

    # `formula; (1-beta1)* grad`
    v_mul_grad = torch.mul(vsub_1_beta1, grad)

    # `formula; (m_t*beta1 + (1 - beta1)*grad)`
    v_div_left = torch.add(v_muls_mt_beta1, v_mul_grad)

    # `formula; lr_t * (m_t*beta1 + (1 - beta1) * grad)`
    # broadcast lr_t to vector

    lrt_broad = lr_t.expand_as(var)
    v_mul_left = torch.mul(lrt_broad, v_div_left)

    # `formula; (epsilon + sqrt(v_t))`
    v_add_sqrt_v = _inner_eps_add_sqrt_vt_compute(epsilon, v_t)

    # `formula; lr_t * (m_t*beta1 + (1-beta1)*grad / (epsilon + sqrt(v_t))`
    v_div_res = torch.div(v_mul_left, v_add_sqrt_v)

    # `formula; var - lr_t * (m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))`
    v_t = torch.sub(var, v_div_res)

    return v_t


def _output_var_t_compute(var, lr_t, m_t, epsilon, v_t):
    """
    _output_var_t_compute
    `var_t = var - lr_t * m_t / (epsilon + sqrt(v_t))`
    """
    # `formula; lr_t * m_t`
    lr_t = lr_t.to(m_t.device)
    v_mul_left = torch.mul(lr_t, m_t)

    # `formula; (epsilon + sqrt(v_t))`
    v_add_sqrt_v = _inner_eps_add_sqrt_vt_compute(epsilon, v_t)

    # `formula; lr_t * m_t /(epsilon + sqrt(v_t))`
    v_div_res = torch.div(v_mul_left, v_add_sqrt_v)

    # `formula; var - lr_t * m_t / (epsilon + sqrt(v_t))`
    v_t = torch.sub(var, v_div_res)

    return v_t


def npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out):
    var, m, v = out
    input_dtype = m.dtype
    beta1_tensor = torch.tensor(beta1, dtype=input_dtype).to(m.device)
    beta1_broad = beta1_tensor.expand_as(m)
    m_t = _output_m_compute(m, beta1_broad, grad)
    v_t = _output_v_compute(v, beta2, grad)
    lr_t = _inner_lr_compute(lr, beta2_power, beta1_power, grad)
    if use_nesterov:
        var_params = VarParams(var, lr_t, m_t, beta1_broad, grad, epsilon, v_t)
        var_t = _output_var_t_compute_use_nesterov(var_params)
    else:
        var_t = _output_var_t_compute(var, lr_t, m_t, epsilon, v_t)
    return var_t, m_t, v_t
