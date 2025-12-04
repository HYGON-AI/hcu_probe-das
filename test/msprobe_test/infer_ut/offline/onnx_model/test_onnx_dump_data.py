# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
import shutil
from unittest import mock

import pytest
import torch
from torch.onnx.utils import export
import onnx
import numpy as np

from msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException


FAKE_ONNX_MODEL_PATH = "fake_msquickcmp_test_onnx_model.onnx"
OUT_PATH = FAKE_ONNX_MODEL_PATH.replace(".onnx", "")
INPUT_SHAPE = (1, 3, 32, 32)


class Args:
    def __init__(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)


@pytest.fixture(scope="module", autouse=True)
def width_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 1, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 32),
        torch.nn.Linear(32, 10),
    )
    export(model, torch.ones(INPUT_SHAPE), FAKE_ONNX_MODEL_PATH)
    yield FAKE_ONNX_MODEL_PATH

    if os.path.exists(FAKE_ONNX_MODEL_PATH):
        os.remove(FAKE_ONNX_MODEL_PATH)
    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)


@pytest.fixture(scope="module")
def reshape_model():
    temp_onnx_name = "reshape" + FAKE_ONNX_MODEL_PATH
    original_shape = [2, 3, 4]
    new_shape = [1, 1, 24]
    node = onnx.helper.make_node(
        'Reshape', inputs=['data', 'shape'], outputs=['reshaped'], name="Reshape_1"
    )
    node2 = onnx.helper.make_node(
        'Reshape', inputs=['data', 'shape'], outputs=['reshaped2']
    )

    values = np.array([1, 1, 24]).astype(np.int64)
    node_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["shape"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT64,
            dims=values.shape,
            vals=values.flatten().astype(int),
        ),
    )

    graph = onnx.helper.make_graph(
        [node, node2, node_const],
        'test_graph',
        [
            onnx.helper.make_tensor_value_info(
                'data', onnx.TensorProto.FLOAT, original_shape
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                'reshaped', onnx.TensorProto.FLOAT, new_shape
            ),
            onnx.helper.make_tensor_value_info(
                'reshaped2', onnx.TensorProto.FLOAT, new_shape
            ),
        ],
    )
    model = onnx.helper.make_model(graph, producer_name='ONNX respace', opset_imports=[onnx.helper.make_opsetid('ai.onnx', 18)])
    onnx.save(model, temp_onnx_name)
    model = onnx.load(temp_onnx_name)
    model.ir_version = 9  # 修改IR版本适配流水onnx和onnxruntime版本
    onnx.save(model, temp_onnx_name)

    yield temp_onnx_name

    if os.path.exists(temp_onnx_name):
        os.remove(temp_onnx_name)


@pytest.fixture(scope="function")
def fake_arguments():
    return Args(
        golden_path=FAKE_ONNX_MODEL_PATH,
        output_path=OUT_PATH,
        input_data="",
        input_shape="",
        dym_shape_range="",
        onnx_fusion_switch=False,
        dump=True,
        cann_path="",
    )


def test_init_given_valid_when_any_then_pass(fake_arguments):
    aa = OnnxDumpData(fake_arguments)

    assert aa.origin_model is not None
    assert aa.origin_model is aa.model_with_inputs


def test_init_given_model_path_when_not_exists_then_error(fake_arguments):
    fake_arguments.golden_path = "not_exists_msquickcmp_test_onnx_model.onnx"
    with pytest.raises(AccuracyCompareException):
        OnnxDumpData(fake_arguments)


def test_init_given_model_path_when_not_onnx_then_error(fake_arguments):
    fake_arguments.golden_path = fake_arguments.golden_path.replace(".onnx", ".om")
    with pytest.raises(AccuracyCompareException):
        OnnxDumpData(fake_arguments)


def test_generate_inputs_data_given_random_when_valid_then_pass(fake_arguments):
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()

    assert aa.inputs_map is not None
    assert len(aa.inputs_map) == 1
    assert list(aa.inputs_map.values())[0].shape == INPUT_SHAPE


def test_generate_inputs_data_given_input_path_when_valid_then_pass(fake_arguments):
    input_path = os.path.join(OUT_PATH, "input_0.bin")
    input_data = np.random.uniform(size=INPUT_SHAPE).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_data = input_path
    aa = OnnxDumpData(fake_arguments)
    with mock.patch('msprobe.infer.utils.check.checker.Checker.check', return_value=True):
        aa.generate_inputs_data()
        
    assert aa.inputs_map is not None
    assert len(aa.inputs_map) == 1
    assert np.allclose(list(aa.inputs_map.values())[0], input_data, atol=1e-7)


def test_generate_inputs_data_given_input_shape_when_valid_then_pass(fake_arguments):
    fake_arguments.input_shape = "input.1:1,3,32,32"
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()

    assert aa.inputs_map is not None
    assert len(aa.inputs_map) == 1


def test_generate_inputs_data_given_input_path_when_not_equal_then_error(fake_arguments):
    input_path = os.path.join(OUT_PATH, "input_0.bin")
    fake_arguments.input_data = ",".join([input_path, input_path])
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_path_when_not_exists_then_error(fake_arguments):
    input_path = os.path.join(OUT_PATH, "not_exists_input_0.bin")
    fake_arguments.input_data = input_path
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_path_when_shape_not_match_then_error(fake_arguments):
    input_path = os.path.join(OUT_PATH, "input_0.bin")
    input_data = np.random.uniform(size=(1,)).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_data = input_path
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        with mock.patch('msprobe.infer.utils.check.checker.Checker.check', return_value=True):
            aa.generate_inputs_data()


def test_generate_inputs_data_given_input_shape_when_shape_not_match_then_error(fake_arguments):
    fake_arguments.input_shape = "input.1:1,3,32"
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_shape_when_invalid_name_then_error(fake_arguments):
    fake_arguments.input_shape = "fake_input.1:1,3,32,32"
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_use_aipp_when_npu_dump_data_path_none_then_error(fake_arguments):
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data(use_aipp=True)


def test_generate_dump_data_given_valid_when_any_then_pass(fake_arguments):
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()
    onnx_dump_data_dir = aa.generate_dump_data()

    assert onnx_dump_data_dir.endswith("onnx") or onnx_dump_data_dir.endswith("onnx/")
    assert os.path.exists(onnx_dump_data_dir)
    assert len(os.listdir(onnx_dump_data_dir)) > 0


def test_load_onnx_given_no_named_node_model_when_any_then_pass(reshape_model):
    x, _ = OnnxDumpData._load_onnx(None, reshape_model)

    assert len(set((y.name for y in x.graph.node))) == len(x.graph.node)


def test_get_input_map(fake_arguments, reshape_model):
    fake_arguments.golden_path = reshape_model
    origin_model = OnnxDumpData(fake_arguments)
    input_data, inputs_map = np.random.random((2, 3, 4)).astype("float32"), {}
    inputs_map["data"] = input_data
    origin_model.inputs_map = inputs_map
    session = origin_model._load_session(reshape_model)
    dump_bins = origin_model._run_model(session, inputs_map)
    augment_inputs_map = origin_model.get_input_map(origin_model.inputs_map, dump_bins)
    
    # augment data success!
    assert list(augment_inputs_map.keys()) == ['data', 'reshaped', 'reshaped2']
