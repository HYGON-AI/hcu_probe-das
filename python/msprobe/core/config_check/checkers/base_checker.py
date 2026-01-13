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

from msprobe.core.common.file_utils import check_path_pattern_valid
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.const import FileCheckConst


class PackInput:

    def __init__(self, output_zip_path, model, shell_path):
        self.output_zip_path = output_zip_path
        self.shell_path = shell_path
        self.model = model[0] if isinstance(model, list) and len(model) > 0 else model
        self.check_input_params()

    def check_input_params(self):
        if self.model and not FmkAdp.is_nn_module(self.model):
            raise Exception(f"model is not torch.nn.Module/mindspore.nn.Cell or module list.")
        if not isinstance(self.output_zip_path, str) or not self.output_zip_path.endswith(FileCheckConst.ZIP_SUFFIX):
            raise Exception(f"output zip path must be a string and ends with '.zip'")
        check_path_pattern_valid(self.output_zip_path)


class BaseChecker:
    input_needed = None
    target_name_in_zip = None
    multi_rank = False

    @staticmethod
    def pack(pack_input):
        pass

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path):
        pass

    @staticmethod
    def apply_patches(fmk):
        pass

    @classmethod
    def compare_ex(cls, bench_dir, cmp_dir, output_path):
        bench_filepath = os.path.join(bench_dir, cls.target_name_in_zip)
        cmp_filepath = os.path.join(cmp_dir, cls.target_name_in_zip)
        if not os.path.exists(bench_filepath) or not os.path.exists(cmp_filepath):
            return None, None, None
        return cls.compare(bench_dir, cmp_dir, output_path)
