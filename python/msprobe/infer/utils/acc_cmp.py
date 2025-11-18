# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
import sys

import numpy as np
import torch

from msprobe.core.common.log import logger
from msprobe.infer.utils.check.string_checker import StringChecker
from msprobe.infer.utils.file_open_check import ms_open, MAX_SIZE_LIMITE_NORMAL_FILE


IS_MSACCUCMP_PATH_SET = False
GLOBAL_TENSOR_CONVERTER = None


def default_tensor_converter(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, but got {type(tensor).__name__}")
    return tensor.data.reshape(tensor.shape)


def set_msaccucmp_path_from_cann():
    global IS_MSACCUCMP_PATH_SET
    global GLOBAL_TENSOR_CONVERTER

    # env TOOLCHAIN_HOME works for both development and product packages.
    cann_path = os.environ.get("TOOLCHAIN_HOME", os.environ.get("ASCEND_TOOLKIT_HOME", ""))
    if not cann_path:
        raise OSError("CANN toolkit in not installed or not set, try installing the latest CANN toolkit.")
    if StringChecker().is_str_safe().check(cann_path, True):
        cann_path = cann_path.split(":")[0]  # Could be multiple split by :, should use the first one

    msaccucmp_path = os.path.join(cann_path, "tools", "operator_cmp", "compare")
    if not os.path.exists(msaccucmp_path):
        raise OSError(f"{msaccucmp_path} not exists, try installing the latest CANN toolkit.")

    if msaccucmp_path not in sys.path:
        sys.path.append(msaccucmp_path)
    IS_MSACCUCMP_PATH_SET = True
    logger.info(f"Set msaccucmp_path={msaccucmp_path}")

    if GLOBAL_TENSOR_CONVERTER is None:
        from conversion import tensor_conversion

        if hasattr(tensor_conversion, "ConvertSingleTensorFormat"):
            GLOBAL_TENSOR_CONVERTER = tensor_conversion.ConvertSingleTensorFormat()
        else:
            GLOBAL_TENSOR_CONVERTER = default_tensor_converter
            logger.warning("ConvertSingleTensorFormat not found in msaccucmp, connot convert tensor format."
                           " Try installing the latest CANN toolkit."
                           )


def parse_torchair_dump_data(dump_file):
    if dump_file.endswith(".npz"):  # Custom converted data info
        with ms_open(dump_file, "rb", max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as f:
            loaded = np.load(f)
        return loaded.get("inputs", []), loaded.get("outputs", [])

    if not IS_MSACCUCMP_PATH_SET:
        set_msaccucmp_path_from_cann()
    from dump_parse.dump_utils import parse_dump_file  # Parser tool from CANN msaccucmp
    from cmp_utils.constant.const_manager import ConstManager

    bin_dump_data = parse_dump_file(dump_file, dump_version=ConstManager.OLD_DUMP_TYPE)
    inputs = [GLOBAL_TENSOR_CONVERTER(input_data) for input_data in bin_dump_data.input_data]
    outputs = [GLOBAL_TENSOR_CONVERTER(output_data) for output_data in bin_dump_data.output_data]
    return inputs, outputs