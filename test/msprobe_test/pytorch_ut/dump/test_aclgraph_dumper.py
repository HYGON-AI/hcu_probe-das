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
import unittest
from unittest.mock import MagicMock, patch

import torch

try:
    from msprobe.lib import aclgraph_dump_ext as _aclgraph_dump_ext
    import msprobe.pytorch.aclgraph_dumper as aclgraph_dumper_module
    from msprobe.pytorch.aclgraph_dumper import AclGraphDumper
    IMPORT_OK = True
except Exception:
    _aclgraph_dump_ext = None
    aclgraph_dumper_module = None
    AclGraphDumper = None
    IMPORT_OK = False


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


@unittest.skipUnless(IMPORT_OK, "aclgraph_dump_ext or aclgraph_dumper import failed, skip tests")
class TestAclGraphDumper(unittest.TestCase):
    def test_aclgraph_dump_ext_import_ok(self):
        self.assertTrue(hasattr(_aclgraph_dump_ext, "get_acl_stat_dict"))

    def test_init_validate_dump_path(self):
        with patch.object(aclgraph_dumper_module, "create_directory") as mock_create_dir, \
                patch.object(aclgraph_dumper_module, "check_and_get_real_path") as mock_real_path:
            mock_real_path.return_value = "/tmp/aclgraph_dump"
            dumper = AclGraphDumper(dump_path="./dump")

        self.assertEqual(dumper.dump_path, "/tmp/aclgraph_dump")
        mock_real_path.assert_called_once()
        mock_create_dir.assert_called_once_with("/tmp/aclgraph_dump")

    def test_start_and_step_return_none(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_resolve_rank_id", return_value=0), \
                patch.object(AclGraphDumper, "_patch"), \
                patch.object(AclGraphDumper, "_synchronize"), \
                patch.object(AclGraphDumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(aclgraph_dumper_module, "get_acl_stat_dict", return_value={}), \
                patch.object(aclgraph_dumper_module, "save_json"):
            dumper = AclGraphDumper("./dump")
            start_ret = dumper.start(MagicMock())
            step_ret = dumper.step()

        self.assertIsNone(start_ret)
        self.assertIsNone(step_ret)

    def test_step_writes_dump_json(self):
        stats = {
            "toy.forward.input.0": {
                "dtype": "Float",
                "shape": [2, 8],
                "max": 1.0,
                "min": -1.0,
                "mean": 0.0,
                "norm": 2.0,
            },
            "toy.forward.output.0": {
                "dtype": "Float",
                "shape": [2, 4],
                "max": 2.0,
                "min": -2.0,
                "mean": 0.1,
                "norm": 3.0,
            },
        }

        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(AclGraphDumper, "_resolve_rank_id", return_value=0), \
                patch.object(AclGraphDumper, "_synchronize"), \
                patch.object(AclGraphDumper, "_step_rank_dir", return_value="./dump/step0/rank0"), \
                patch.object(aclgraph_dumper_module, "get_acl_stat_dict", return_value=stats), \
                patch.object(aclgraph_dumper_module, "save_json") as mock_save_json:
            dumper = AclGraphDumper("./dump")
            dumper._running = True
            dumper.step()

        save_path, dump_json = mock_save_json.call_args[0][0], mock_save_json.call_args[0][1]
        self.assertEqual(save_path, os.path.join("./dump/step0/rank0", "dump.json"))
        self.assertEqual(dump_json["task"], aclgraph_dumper_module.Const.STATISTICS)
        self.assertEqual(dump_json["framework"], aclgraph_dumper_module.Const.PT_FRAMEWORK)
        self.assertIn("toy.forward", dump_json["data"])
        self.assertEqual(mock_save_json.call_args.kwargs["indent"], 2)

    def test_collect_acl_stat_called_after_start(self):
        with patch.object(AclGraphDumper, "_validate_dump_path", return_value="./dump"), \
                patch.object(aclgraph_dumper_module, "acl_stat") as mock_acl_stat:
            model = ToyModel()
            dumper = AclGraphDumper("./dump")
            dumper.start(model)
            _ = model(torch.randn(2, 8))

        self.assertGreater(mock_acl_stat.call_count, 0)
