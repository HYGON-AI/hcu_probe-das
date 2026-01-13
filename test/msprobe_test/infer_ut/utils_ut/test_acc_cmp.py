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

import sys
from unittest.mock import patch, MagicMock, mock_open

import pytest
import numpy as np
import torch

from msprobe.infer.utils.acc_cmp import set_msaccucmp_path_from_cann, parse_torchair_dump_data, \
    default_tensor_converter, IS_MSACCUCMP_PATH_SET, GLOBAL_TENSOR_CONVERTER


@pytest.fixture(scope="module", autouse=True)
def reset_globals():
    # Import the original globals from the acc_cmp module
    from msprobe.infer.utils.acc_cmp import IS_MSACCUCMP_PATH_SET as orig_is_set, GLOBAL_TENSOR_CONVERTER as orig_converter

    yield

    # Reset the globals back to their original state after all tests
    global IS_MSACCUCMP_PATH_SET, GLOBAL_TENSOR_CONVERTER
    IS_MSACCUCMP_PATH_SET = orig_is_set
    GLOBAL_TENSOR_CONVERTER = orig_converter


def test_set_msaccucmp_path_from_cann_given_ascend_toolkit_home_when_valid_then_path_set():
    with patch.dict('os.environ', {'TOOLCHAIN_HOME': '/mock/path'}), \
         patch('msprobe.infer.utils.acc_cmp.os.path.exists', return_value=True), \
         patch('msprobe.infer.utils.acc_cmp.sys.path', new=[]), \
         patch('msprobe.infer.utils.acc_cmp.GLOBAL_TENSOR_CONVERTER', return_value=True):
        set_msaccucmp_path_from_cann()
        assert '/mock/path/tools/operator_cmp/compare' in sys.path


def test_set_msaccucmp_path_from_cann_given_no_env_when_invalid_then_oserror():
    with patch.dict('os.environ', {}, clear=True), \
         pytest.raises(OSError) as exc_info:
        set_msaccucmp_path_from_cann()
    assert "CANN toolkit in not installed or not set" in str(exc_info.value)


def test_set_msaccucmp_path_from_cann_given_nonexistent_path_when_invalid_then_oserror():
    with patch.dict('os.environ', {'TOOLCHAIN_HOME': '/mock/path'}), \
         patch('os.path.exists') as mock_exists, \
         pytest.raises(OSError) as exc_info:
        mock_exists.return_value = False
        set_msaccucmp_path_from_cann()
    assert "/mock/path/tools/operator_cmp/compare not exists" in str(exc_info.value)


def test_parse_torchair_dump_data_given_npz_file_when_valid_then_return_inputs_outputs():
    mock_loaded = {
        'inputs': [np.array([1])],
        'outputs': [np.array([2])]
    }
    with patch('msprobe.infer.utils.acc_cmp.ms_open', mock_open(read_data=None)) as mocked_open, \
         patch('msprobe.infer.utils.acc_cmp.np.load') as mock_np_load:
        mock_np_load.return_value = mock_loaded
        inputs, outputs = parse_torchair_dump_data('dummy.npz')
        assert isinstance(inputs, list) and len(inputs) == 1
        assert isinstance(outputs, list) and len(outputs) == 1


def test_parse_torchair_dump_data_given_bin_file_when_valid_then_call_parser():
    # mock cmp_utils module
    mock_cmp_utils = MagicMock()
    mock_constant = MagicMock()
    mock_const_manager = MagicMock()
    mock_manager = MagicMock()    
    mock_cmp_utils.constant = mock_constant
    mock_constant.const_manager = mock_const_manager
    mock_const_manager.ConstManager = mock_manager
    # mock dump_parse module and it's submodule
    mock_dump_parse =  MagicMock()
    mock_dump_utils = MagicMock()
    mock_parse_dump_file = MagicMock()
    mock_dump_parse.dump_utils = mock_dump_utils
    mock_dump_utils.parse_dump_file = mock_parse_dump_file

    with patch('msprobe.infer.utils.acc_cmp.IS_MSACCUCMP_PATH_SET', True), \
         patch('msprobe.infer.utils.acc_cmp.GLOBAL_TENSOR_CONVERTER') as mock_converter, \
         patch.dict('sys.modules', {
             'dump_parse': mock_dump_parse,
             'dump_parse.dump_utils': mock_dump_utils,
             'cmp_utils': mock_cmp_utils,
             'cmp_utils.constant': mock_constant,
             'cmp_utils.constant.const_manager': mock_const_manager
         }):
        mock_manager.OLD_DUMP_TYPE = "old_dump_type"
        mock_parse_dump_file.return_value.input_data = [np.array([3])]
        mock_parse_dump_file.return_value.output_data = [np.array([4])]
        def mock_converter_side_effect(x):
            return x * 2
        mock_converter.side_effect = mock_converter_side_effect
        inputs, outputs = parse_torchair_dump_data('dummy.bin')
        mock_parse_dump_file.assert_called_once()
        assert (inputs[0] == np.array([6])).all()
        assert (outputs[0] == np.array([8])).all()


def test_parse_torchair_dump_data_given_non_npz_and_uninitialized_when_set_path_called():
    mock_cmp_utils = MagicMock()
    mock_constant = MagicMock()
    mock_const_manager = MagicMock()
    mock_manager = MagicMock()    
    mock_cmp_utils.constant = mock_constant
    mock_constant.const_manager = mock_const_manager
    mock_const_manager.ConstManager = mock_manager
    mock_dump_parse =  MagicMock()
    mock_dump_utils = MagicMock()
    mock_parse_dump_file = MagicMock()
    mock_dump_parse.dump_utils = mock_dump_utils
    mock_dump_utils.parse_dump_file = mock_parse_dump_file
    with patch('msprobe.infer.utils.acc_cmp.IS_MSACCUCMP_PATH_SET', False), \
         patch('msprobe.infer.utils.acc_cmp.set_msaccucmp_path_from_cann') as mock_set_path, \
         patch.dict('sys.modules', {
             'dump_parse': mock_dump_parse,
             'dump_parse.dump_utils': mock_dump_utils,
             'cmp_utils': mock_cmp_utils,
             'cmp_utils.constant': mock_constant,
             'cmp_utils.constant.const_manager': mock_const_manager
         }):
        parse_torchair_dump_data('dummy.bin')
        mock_set_path.assert_called_once()


def test_default_tensor_converter_given_tensor_when_any_then_returns_reshaped_tensor():
    test_tensor = torch.tensor([1, 2, 3])
    reshaped_tensor = default_tensor_converter(test_tensor)
    assert reshaped_tensor.shape == test_tensor.shape


def test_default_tensor_converter_given_list_when_any_then_raise_error():
    with pytest.raises(TypeError) as exc_info:
        reshaped_tensor = default_tensor_converter([1,2,3])
    assert "Expected a torch.Tensor, but got list" == str(exc_info.value)