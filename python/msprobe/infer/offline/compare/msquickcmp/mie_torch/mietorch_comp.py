# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import csv 

import torch 

from msprobe.infer.offline.common import logger
from msprobe.infer.offline.compare.utils.ge_dump_reader import GEDumpFileReader
from msprobe.infer.offline.compare.utils.torch_dump_reader import TorchDumpFileReader
from msprobe.infer.utils.cmp_algorithm import CMP_ALG_MAP, CUSTOM_ALG_MAP
from msprobe.infer.utils.file_open_check import ms_open, sanitize_csv_value


class MIETorchCompare:
    def __init__(self, cpu_path: str, npu_path: str, json_path: str, output_path: str = "."):
        self.cpu_path = cpu_path
        self.npu_path = npu_path
        self.json_path = json_path
        self.output_path = output_path

        self.npu_reader = GEDumpFileReader(npu_path, json_path)
        self.cpu_reader = self.get_torch_dump_reader(cpu_path, json_path)
        self.output_path = output_path
    
    @staticmethod
    def check_tensor(golden_data_fp32, my_data_fp32):
        tensor_pass = True
        fail_reasons = []

        if len(golden_data_fp32) != len(my_data_fp32):
            fail_reasons.append("data shape doesn't match.")
            tensor_pass = False 
        if not torch.all(torch.isfinite(golden_data_fp32)):
            fail_reasons.append("cpu_data includes NAN or inf.")
            tensor_pass = False 
        if not torch.all(torch.isfinite(my_data_fp32)):
            fail_reasons.append("npu_data includes NAN or inf.")
            tensor_pass = False
        
        return tensor_pass, " ".join(fail_reasons)

    def get_torch_dump_reader(self, cpu_path, json_path):
        torch_mode = self.npu_reader.torch_mode
        if torch_mode == "TorchScript":
            return TorchDumpFileReader(cpu_path, json_path, "TorchScript")
        else:
            return TorchDumpFileReader(cpu_path, json_path, "TorchExport")

    def compare(self):
        tensors = {}
        cpu_keys = self.cpu_reader.get_keys()
        npu_keys = self.npu_reader.get_keys()
        for cpu_key in cpu_keys:
            if cpu_key in npu_keys:
                cpu_tensor = self.cpu_reader.get_tensor(cpu_key)
                npu_tensor = self.npu_reader.get_tensor(cpu_key)
                if cpu_tensor is None or npu_tensor.shape == (0,):
                    logger.warning("Could not find matched tensor which key is: %s", cpu_key)
                    continue
                if cpu_tensor.shape == npu_tensor.shape:
                    tensors[cpu_key] = (cpu_tensor, npu_tensor)
        all_rows_data = []
        logger.info(f"{len(tensors)} tensors were matched in total.")

        for key, (cpu_tensor, npu_tensor) in tensors.items():
            row_data = {"Key": self.cpu_reader.key_to_folder[key]}
            npu_tensor = torch.from_numpy(npu_tensor)
            cpu_tensor = cpu_tensor.reshape(-1).float()
            npu_tensor = npu_tensor.reshape(-1).float()

            tensor_pass, message = self.check_tensor(cpu_tensor, npu_tensor)

            if not tensor_pass:
                logger.debug("check_tensor failed: %s", message)
                row_data["cmp_fail_reason"] = message 
            else:
                fail_messages = []
                for name, cmp_func in list(CMP_ALG_MAP.items()) + list(CUSTOM_ALG_MAP.items()):
                    result, message = cmp_func(cpu_tensor, npu_tensor)
                    row_data[name] = result 
                    if len(message) > 0:
                        fail_messages.append(message)
                row_data["cmp_fail_reason"] = " ".join(fail_messages)

            all_rows_data.append(row_data)
        
        self.save_compare_result_to_csv(all_rows_data)

    
    def save_compare_result_to_csv(self, all_rows_data: list):
        if not all_rows_data:
            logger.info("There is no comparison data to save.")
            return
        
        sorted_rows = sorted(
            all_rows_data,
            key=lambda x: self.cpu_reader.key_to_id.get(x["Key"], float('inf'))
            )
        
        for header in sorted_rows[0].keys():
            sanitize_csv_value(header)
        for row in sorted_rows:
            for _, row_value in row.items():
                sanitize_csv_value(row_value)
        
        csv_file_path = os.path.join(self.output_path, 'comparison_results.csv')
        
        with ms_open(csv_file_path, mode="w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted_rows[0].keys())
            writer.writeheader()
            writer.writerows(sorted_rows)
        
        logger.info("compare resut has been saved on %r ." % csv_file_path)
        