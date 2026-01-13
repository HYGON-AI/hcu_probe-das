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

import unittest
from unittest.mock import Mock, patch
import numpy as np
import mindspore
from mindspore import Tensor, ops

from msprobe.mindspore.monitor.distributed.wrap_distributed import (
    catch_data,
    CatchDataParams,
    DistributedOPTemplate,
    ApiRegistry,
    get_distributed_ops,
    is_target_line,
    op_aggregate,
    update_data
)
from msprobe.core.common.const import MonitorConst

class TestWrapDistributed(unittest.TestCase):
    def setUp(self):
        self.mock_ops = ['min', 'max', 'norm']
        self.mock_rank = '0'

    def hook(self):
        def forward_pre_hook(nope, inputs):
            return inputs

        def forward_hook(nope, inputs, output):
            return 2

        return [forward_pre_hook], [forward_hook]

    def test_catch_data(self):
        # 准备测试数据
        cc_context = Mock()
        cc_context.data = {}
        cc_name = "all_reduce"
        args = [Tensor(np.array([1.0, 2.0, 3.0]))]
        
        # 测试输入数据捕获
        catch_data(CatchDataParams(cc_context, cc_name, self.mock_ops, args, MonitorConst.PREFIX_PRE))
        self.assertTrue('all_reduce/pre_0_0' in cc_context.data)
        
        # 测试输出数据捕获
        catch_data(CatchDataParams(cc_context, cc_name, self.mock_ops, args, MonitorConst.PREFIX_POST))
        self.assertTrue('all_reduce/post_0_0' in cc_context.data)

    def test_distributed_op_template(self):
        # 测试分布式算子模板
        pre_hooks, post_hooks = self.hook()
        op = DistributedOPTemplate("all_reduce", pre_hooks, post_hooks)
        
        self.assertEqual(op.op_name_, "all_reduce")
        self.assertEqual(len(op.cc_hooks), 2)

    def test_api_registry(self):
        # 测试API注册器
        registry = ApiRegistry()
        
        # 测试API存储
        ori_api_group = Mock()
        api_list = ["all_reduce", "all_gather"]
        api_ori_attr = {}
        
        ApiRegistry.store_ori_attr(ori_api_group, api_list, api_ori_attr)
        self.assertEqual(len(api_ori_attr), 2)

    def test_op_aggregate(self):
        # 测试算子聚合
        tensor_list = [Tensor(1.0), Tensor(2.0), Tensor(3.0)]
        
        # 测试min操作
        result = op_aggregate('min', tensor_list)
        self.assertEqual(result.asnumpy(), 1.0)
        
        # 测试max操作
        result = op_aggregate('max', tensor_list)
        self.assertEqual(result.asnumpy(), 3.0)
        
        # 测试mean操作
        result = op_aggregate('mean', tensor_list)
        self.assertEqual(result.asnumpy(), 2.0)

    def test_update_data(self):
        # 测试数据更新
        old_data = {}
        new_data = {
            'tag1': {
                'min': Tensor(1.0),
                'max': Tensor(2.0)
            }
        }
        
        result = update_data(old_data, new_data)
        self.assertTrue('tag1' in result)
        self.assertTrue('min' in result['tag1'])
        self.assertTrue('max' in result['tag1'])

    def test_is_target_line(self):
        # 测试目标行检查
        # 空代码行列表应该返回True
        self.assertTrue(is_target_line([]))
        
        # 测试匹配模式
        codeline = ['test_pattern']
        with patch('msprobe.mindspore.monitor.distributed.wrap_distributed.get_callstack') as mock_callstack:
            mock_callstack.return_value = ['test_pattern_line']
            self.assertTrue(is_target_line(codeline))

    def test_get_distributed_ops(self):
        # 测试获取分布式算子列表
        ops = get_distributed_ops()
        self.assertIsInstance(ops, set)
