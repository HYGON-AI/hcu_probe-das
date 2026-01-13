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
import pytest
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from msprobe.infer.offline.compare.msquickcmp.npu.om_parser import OmParser
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.offline.compare.msquickcmp.npu.om_parser import (
    GRAPH_OBJECT,
    OP_OBJECT,
    NAME_OBJECT,
    TYPE_OBJECT,
    INPUT_DESC_OBJECT,
    INPUT_OBJECT,
    ATTR_OBJECT,
    SHAPE_OBJECT,
    SHAPE_RANGE_OBJECT,
    DIM_OBJECT,
    DATA_OBJECT,
    NET_OUTPUT_OBJECT,
    ATC_CMDLINE_OBJECT,
    INPUT_SHAPE,
    INPUT_SHAPE_RANGE,
    LIST_LIST_INT_OBJECT,
    LIST_LIST_I_OBJECT,
    LIST_I_OBJECT,
    LIST_OBJECT,
    KEY_OBJECT,
    VALUE_OBJECT,
    SUBGRAPH_NAME,
    S_OBJECT,
    DTYPE_OBJECT,
    DTYPE_MAP,
    OUT_NODES_NAME,
    AIPP_CONFIG_PATH,
    LAYOUT_OBJECT,
    SPECIAL_OPS_TYPE
)


@pytest.fixture(scope="module", autouse=True)
def om_parser() -> OmParser:
    ut_dir = os.path.dirname(os.path.realpath(__file__))
    om_parser = OmParser(os.path.join(ut_dir, "..", "resource", "msquickcmp", "om", "model.json"))
    return om_parser


def test_dynamic_scenario(om_parser):
    is_dynamic_scenario, _ = om_parser.get_dynamic_scenario_info()
    assert is_dynamic_scenario is False


def test_get_shape_size(om_parser):
    shape_size_array = om_parser.get_shape_size()
    assert shape_size_array == [1280000, 320000]


def test_net_output_count(om_parser):
    count = om_parser.get_net_output_count()
    assert count == 3

    atc_cmd = om_parser.get_atc_cmdline()
    assert "model" in atc_cmd

    net_output = om_parser.get_expect_net_output_name()
    assert len(net_output) == 3
    assert net_output.get(0) == "Cast_1219:0:output0"
    assert net_output.get(1) == 'PartitionedCall_Gather_1221_gatherv2_3:0:output2'
    assert net_output.get(2) == 'Reshape_1213:0:output1'


def test_get_atc_cmdline(om_parser):
    atc_cmd = om_parser.get_atc_cmdline()
    assert "model" in atc_cmd


def test_get_expect_net_output_name(om_parser):
    net_output = om_parser.get_expect_net_output_name()
    assert len(net_output) == 3
    assert net_output.get(0) == "Cast_1219:0:output0"
    assert net_output.get(1) == 'PartitionedCall_Gather_1221_gatherv2_3:0:output2'
    assert net_output.get(2) == 'Reshape_1213:0:output1'


class TestOmParserMethods(unittest.TestCase):

    # ------------------------------------------------------------
    # 构造完全隔离 OmParser.__init__ 的伪 parser
    # ------------------------------------------------------------
    def _make_parser(self, shape_range=False):
        parser = SimpleNamespace()
        parser.contain_negative_1 = False
        parser.shape_range = shape_range

        parser._get_shape_list = OmParser._get_shape_list.__get__(parser)
        parser._get_range_shape_size_list = OmParser._get_range_shape_size_list.__get__(parser)
        parser._process_inputs_to_list = OmParser._process_inputs_to_list.__get__(parser)

        return parser

    # ------------------------------------------------------------
    # 1. _get_shape_list
    # ------------------------------------------------------------
    def test_get_shape_list_contains_negative_1(self):
        parser = self._make_parser()
        list_list_int_obj = {
            LIST_LIST_I_OBJECT: [{LIST_I_OBJECT: [1, -1]}]
        }

        shape_list = []
        parser._get_shape_list(list_list_int_obj, shape_list)

        self.assertTrue(parser.contain_negative_1)
        self.assertEqual(shape_list, [])

    # ------------------------------------------------------------
    # 2. 正常 shape_range
    # ------------------------------------------------------------
    def test_get_range_shape_size_list_normal(self):
        parser = self._make_parser(shape_range=True)

        input_object = {
            ATTR_OBJECT: [{
                KEY_OBJECT: SHAPE_RANGE_OBJECT,
                VALUE_OBJECT: {
                    LIST_LIST_INT_OBJECT: {
                        LIST_LIST_I_OBJECT: [
                            {LIST_I_OBJECT: [1, 2]},
                            {LIST_I_OBJECT: [3, 3]},
                        ]
                    }
                }
            }]
        }

        res = parser._get_range_shape_size_list(input_object)
        self.assertEqual(res, [3, 6])

    # ------------------------------------------------------------
    # 3. 遇到 -1
    # ------------------------------------------------------------
    def test_get_range_shape_size_list_negative_1(self):
        parser = self._make_parser(shape_range=True)

        input_object = {
            ATTR_OBJECT: [{
                KEY_OBJECT: SHAPE_RANGE_OBJECT,
                VALUE_OBJECT: {
                    LIST_LIST_INT_OBJECT: {
                        LIST_LIST_I_OBJECT: [
                            {LIST_I_OBJECT: [1, -1]},
                        ]
                    }
                }
            }]
        }

        res = parser._get_range_shape_size_list(input_object)
        self.assertEqual(res, [])
        self.assertTrue(parser.contain_negative_1)

    # ------------------------------------------------------------
    # 4. _process_inputs_to_list 缺 shape
    # ------------------------------------------------------------
    def test_process_inputs_to_list_no_shape(self):
        parser = self._make_parser()
        with self.assertRaises(AccuracyCompareException):
            parser._process_inputs_to_list([{DTYPE_OBJECT: "float32"}])

    # ------------------------------------------------------------
    # 5. shape_range=True 禁止 dim
    # ------------------------------------------------------------
    def test_process_inputs_to_list_shape_range_error(self):
        parser = self._make_parser(shape_range=True)
        input_desc = [
            {DTYPE_OBJECT: "float32", SHAPE_OBJECT: {DIM_OBJECT: [1, 2, 3, 4]}}
        ]
        with self.assertRaises(AccuracyCompareException):
            parser._process_inputs_to_list(input_desc)

    # ------------------------------------------------------------
    # 6. NHWC → NCHW
    # ------------------------------------------------------------
    def test_process_inputs_to_list_reorder(self):
        parser = self._make_parser()

        input_desc = [{
            DTYPE_OBJECT: "DT_FLOAT",
            SHAPE_OBJECT: {DIM_OBJECT: [1, 2, 3, 4]},
            LAYOUT_OBJECT: "NHWC"
        }]

        res = parser._process_inputs_to_list(input_desc)
        self.assertEqual(res, [[1, 4, 2, 3]])

    # ------------------------------------------------------------
    # 7. 测试静态方法 _parse_net_output_node_attr
    # ------------------------------------------------------------
    def test_parse_net_output_node_attr(self):
        operator = {
            INPUT_DESC_OBJECT: [
                {DTYPE_OBJECT: "DT_FLOAT", SHAPE_OBJECT: {DIM_OBJECT: [2, 3]}},
                {DTYPE_OBJECT: "DT_INT32", SHAPE_OBJECT: {DIM_OBJECT: [5]}},
            ]
        }

        result = OmParser._parse_net_output_node_attr(operator)

        self.assertEqual(result[0][0], np.float32)
        self.assertEqual(result[0][1], [2, 3])
        self.assertEqual(result[1][0], np.int32)
        self.assertEqual(result[1][1], [5])

    # ------------------------------------------------------------
    # 8. get_net_output_data_info — 用 Fake OmParser 替代实例
    # ------------------------------------------------------------
    def test_get_net_output_data_info_single(self):

        fake = SimpleNamespace()
        fake._parse_net_output_node_attr = OmParser._parse_net_output_node_attr
        fake.get_dynamic_scenario_info = MagicMock(return_value=(None, None))

        fake._gen_operator_list = MagicMock(return_value=[
            {TYPE_OBJECT: NET_OUTPUT_OBJECT,
             INPUT_DESC_OBJECT: [
                 {DTYPE_OBJECT: "DT_FLOAT", SHAPE_OBJECT: {DIM_OBJECT: [1, 3]}}
             ]}
        ])

        result = OmParser.get_net_output_data_info(fake, dump_data_path=None)

        self.assertEqual(result[0][0], np.float32)
        self.assertEqual(result[0][1], [1, 3])

    # ------------------------------------------------------------
    # 9. AIPP config 解析
    # ------------------------------------------------------------
    def test_get_aipp_config_content(self):

        fake = SimpleNamespace()
        fake.json_object = {
            GRAPH_OBJECT: [{
                OP_OBJECT: [
                    {ATTR_OBJECT: [{KEY_OBJECT: AIPP_CONFIG_PATH, VALUE_OBJECT: {S_OBJECT: "/path/a1"}}]},
                    {ATTR_OBJECT: [{KEY_OBJECT: "other"}]},
                    {ATTR_OBJECT: [{KEY_OBJECT: AIPP_CONFIG_PATH, VALUE_OBJECT: {S_OBJECT: "/path/a2"}}]},
                ]
            }]
        }

        result = OmParser.get_aipp_config_content(fake)
        self.assertEqual(result, ["/path/a1", "/path/a2"])


class TestOmParserShapeReplace(unittest.TestCase):
    def test_replace_input_shape_with_input_shape_range_multi_inputs(self):
        """
        Test replacing --input_shape with --input_shape_range for multiple dynamic inputs.
        This UT avoids OmParser full initialization by patching attributes manually.
        """

        # Create object without calling __init__
        parser = OmParser.__new__(OmParser)

        # Mock properties used in the method
        parser.shape_range = False  # Ensure log branch goes inside (shape_range=False -> will set True)

        # dynamic input case
        atc_cmd = "--input_shape=input1:1,3,-1,224;input2:1,64,64,-1"

        # call the private method
        result = parser._replace_input_shape_with_input_shape_range_in_atc_cmd_if_dynamic(atc_cmd)

        # expected transformed version
        expected = "--input_shape_range input1:[1,3,-1,224];input2:[1,64,64,-1]"

        self.assertEqual(result, expected)

        # Ensure shape_range updated to True
        self.assertTrue(parser.shape_range)
