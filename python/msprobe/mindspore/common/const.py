# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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

import numpy as np
import mindspore as ms
from mindspore import dtype as mstype

from msprobe.core.common.const import Const as CoreConst


class Const:
    CELL = "cell"
    API = "api"
    KERNEL = "kernel"
    CELL_AND_API = 'cell_and_api'
    TOOL_LEVEL_DICT = {
        CoreConst.LEVEL_L0: CELL,
        CoreConst.LEVEL_L1: API,
        CoreConst.LEVEL_L2: KERNEL,
        CoreConst.LEVEL_MIX: CELL_AND_API
    }

    PYNATIVE_MODE = CoreConst.PYNATIVE_MODE
    GRAPH_MODE = "graph"
    GRAPH_GE_MODE = "graph_ge"
    GRAPH_KBYK_MODE = "graph_kbyk"
    PYNATIVE_GRAPH_MODE = CoreConst.PYNATIVE_GRAPH_MODE

    JIT_LEVEL = "jit_level"
    JIT_LEVEL_O0 = "O0"
    JIT_LEVEL_O1 = "O1"
    JIT_LEVEL_O2 = "O2"
    ASCEND_910A = "ascend910"

    OPS_PREFIX = "mindspore.ops."
    TENSOR_PREFIX = "mindspore.Tensor."
    MINT_PREFIX = "mindspore.mint."
    MINT_NN_FUNC_PREFIX = "mindspore.mint.nn.functional."

    TENSOR_DATA_PREFIX = "Tensor."
    STUB_TENSOR_DATA_PREFIX = "Tensor."
    OPS_DATA_PREFIX = "Functional."
    MINT_DATA_PREFIX = "Mint."
    MINT_NN_FUNC_DATA_PREFIX = "MintFunctional."
    DISTRIBUTED_DATA_PREFIX = "Distributed."
    TORCH_DATA_PREFIX = "Torch."
    TORCH_NPU_DATA_PREFIX = "NPU."

    SUPPORTED_API_LIST_FILE = "support_wrap_ops.yaml"
    SUPPORTED_TENSOR_LIST_KEY = "tensor"
    SUPPORTED_OPS_LIST_KEY = "ops"
    SUPPORTED_MINT_LIST_KEY = "mint.ops"
    SUPPORTED__MINT_NN_FUNC_LIST_KEY = "mint.nn.functional"
    SUPPORTED_COMM_LIST_KEY = "communication.comm_func"

    DROPOUT_API_NAME_PREFIX = "dropout"

    GRAPH_DATA_MODE_LIST = [CoreConst.ALL, CoreConst.INPUT, CoreConst.OUTPUT]
    GRAPH_CELL_DUMP_DATA_MODE_LIST = [CoreConst.ALL, CoreConst.FORWARD, CoreConst.BACKWARD]

    HOOK_MS_PREFIX_DICT = {
        OPS_DATA_PREFIX: OPS_PREFIX,
        TENSOR_DATA_PREFIX: TENSOR_PREFIX,
        MINT_DATA_PREFIX: MINT_PREFIX,
        MINT_NN_FUNC_DATA_PREFIX: MINT_NN_FUNC_PREFIX
    }

    NonDifferentiableType = (
        mstype.bool_, mstype.int8, mstype.byte, mstype.uint8, mstype.ubyte,
        mstype.int16, mstype.short, mstype.uint16, mstype.ushort,
        mstype.int32, mstype.intc, mstype.uint32, mstype.uintc,
        mstype.int64, mstype.intp, mstype.uint64, mstype.uintp
    )


class MsCompareConst:
    # api_info field
    MINT = "Mint"
    MINT_FUNCTIONAL = "MintFunctional"
    TENSOR_API = "Tensor"
    FUNCTIONAL_API = "Functional"
    FUSION_API = "FUSION"

    API_NAME_STR_LENGTH = 4
    MAX_RECURSION_DEPTH = 20

    # Mindtorch api_info field
    MINDTORCH_TENSOR = "Tensor"
    MINDTORCH = "Torch"
    MINDTORCH_FUNC = "Functional"
    MINDTORCH_NPU = "NPU"
    MINDTORCH_DIST = "Distributed"

    MT_VALID_API_TYPES = [
        MINDTORCH, MINDTORCH_FUNC, MINDTORCH_TENSOR
    ]
    SUPPORTED_FUSION_LIST = ["flash_attention_score"]

    TASK_FIELD = "task"
    STATISTICS_TASK = "statistics"
    FRAMEWORK = "framework"
    TENSOR_TASK = "tensor"
    DUMP_DATA_DIR_FIELD = "dump_data_dir"
    DATA_FIELD = "data"

    # supported api yaml
    SUPPORTED_API_LIST_FILE = "checker_support_api.yaml"
    SUPPORTED_TENSOR_LIST_KEY = "tensor"

    # detail_csv
    DETAIL_CSV_API_NAME = "API Name"
    DETAIL_CSV_BENCH_DTYPE = "Bench Dtype"
    DETAIL_CSV_TESTED_DTYPE = "Tested Dtype"
    DETAIL_CSV_SHAPE = "Shape"
    DETAIL_CSV_PASS_STATUS = "Status"
    DETAIL_CSV_MESSAGE = "Message"
    DETAIL_CSV_FILE_NAME = "accuracy_checking_details"

    # result_csv
    RESULT_CSV_FORWARD_TEST_SUCCESS = "Forward Test Success"
    RESULT_CSV_BACKWARD_TEST_SUCCESS = "Backward Test Success"
    RESULT_CSV_FILE_NAME = "accuracy_checking_result"

    EPSILON = 1e-8

    class ProcessStatus:
        SUCCESS = "success"
        API_NOT_FOUND = "api_not_found"
        EXCEPTION_SKIP = "exception_skip"


class FreeBenchmarkConst:
    ADD_NOISE = "add_noise"
    BIT_NOISE = "bit_noise"
    NO_CHANGE = "no_change"
    EXCHANGE_VALUE = "change_value"
    IMPROVE_PRECISION = "improve_precision"
    CHECK = "check"
    FIX = "fix"
    DEFAULT_DEVICE = "npu"
    DEFAULT_STAGE = CoreConst.FORWARD
    DEFAULT_DUMP_LEVEL = "L1"
    DEFAULT_PERT_TYPE = IMPROVE_PRECISION
    DEFAULT_HANDLER_TYPE = CHECK
    DEVICE_LIST = [DEFAULT_DEVICE]
    STAGE_LIST = [CoreConst.FORWARD, CoreConst.BACKWARD]
    DUMP_LEVEL_LIST = [DEFAULT_DUMP_LEVEL]
    PERT_TYPE_LIST = [IMPROVE_PRECISION, ADD_NOISE, BIT_NOISE, NO_CHANGE, EXCHANGE_VALUE]
    HANDLER_TYPE_LIST = [CHECK, FIX]
    NO_CHANGE_ERROR_THRESHOLD = 1.0
    SYMBOL_FLIPPING_RATIO = 8.0

    SUPPORTED_CHECK_API_FILE = "support_wrap_ops.yaml"
    CHECK_RESULT_FILE = "free_benchmark.csv"

    API_PREFIX_DICT = {
        "ops": Const.OPS_PREFIX,
        "Tensor": Const.TENSOR_PREFIX,
        "mint": Const.MINT_PREFIX,
        "mint.nn.functional": Const.MINT_NN_FUNC_PREFIX
    }

    PERT_VALUE_DICT = {
        ms.bfloat16: 1e-4,
        ms.float16: 1e-6,
        ms.float32: 1e-8,
        ms.float64: 1e-16
    }

    ERROR_THRESHOLD = {
        ms.bfloat16: 1.004,
        ms.float16: 1.002,
        ms.float32: 1.0002
    }

    PERT_BIT_DICT = {
        ms.float16: np.int16,
        ms.float32: np.int32,
        ms.float64: np.int64
    }

    MS_NUMPY_DTYPE_DICT = {
        ms.int16: np.int16,
        ms.int32: np.int32,
        ms.int64: np.int64,
        ms.float16: np.float16,
        ms.float32: np.float32,
        ms.float64: np.float64
    }
