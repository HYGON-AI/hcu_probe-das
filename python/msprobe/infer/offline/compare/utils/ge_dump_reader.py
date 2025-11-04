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
import re
import json
import torch

from msprobe.infer.offline.compare.utils.base_dump_reader import DumpFileReader
from msprobe.infer.utils.acc_cmp import parse_torchair_dump_data
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.constants import JSON_FILE_MAX_SIZE

IS_MSACCUCMP_PATH_SET = False
GLOBAL_TENSOR_CONVERTER = None
MAX_DEPTH = 2


def find_npu_dump_files(root_dir, matching_files, depth=0):
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        
        if os.path.isfile(full_path):
            matching_files.append(full_path)
        # 目录给到Graph，最大往下匹配两个子目录，且最后一层文件名是 '0'
        elif depth < MAX_DEPTH:
            if os.path.isdir(full_path):
                sub_dir = os.path.join(full_path, '0')
                if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
                    find_npu_dump_files(sub_dir, matching_files, depth + 1)
                else:
                    find_npu_dump_files(full_path, matching_files, depth + 1)


class GEDumpFileReader(DumpFileReader):
    def __init__(self, npu_path: str, json_path: str, path: str):
        super().__init__(path)
        self.path = npu_path
        self.json_path = json_path
        self.torch_mode = self.get_torch_mode(json_path)
        self.process_json_files()
        self.key_to_folder = self._map_keys_to_folders()
        self.npu_files = None

    @staticmethod
    def get_torch_mode(json_path):
        with ms_open(os.path.join(json_path, 'mindie_torch_op_mapping.json'), max_size=JSON_FILE_MAX_SIZE) as f:
            torch_op_map = json.load(f)
        if not torch_op_map:
            raise ValueError("Please check your mindie_torch_op_mapping.json file, it's empty.")
        item_keys = torch_op_map[0].keys()
        if "jit_node" in item_keys:
            return "TorchScript"
        elif "fx_node" in item_keys:
            return "TorchExport"
        else:
            return "Unsupport"

    def get_torch_rt_mapping(self, torch_op_map):
        if self.torch_mode == "TorchScript":
            torch_rt_map = {item["rt_layer"]: item["jit_node"] for item in torch_op_map}
        else:
            torch_rt_map = {item["rt_layer"]: item["fx_node"] for item in torch_op_map}
        
        return torch_rt_map

    def process_json_files(self):
        with ms_open(os.path.join(self.json_path, 'mindie_torch_op_mapping.json'), max_size=JSON_FILE_MAX_SIZE) as f:
            torch_op_map = json.load(f)

        if self.torch_mode not in ["TorchScript", "TorchExport"]:
            raise NotImplementedError("Current only support 'TorchScript' and 'TorchExport' model.")
        
        rt_torch_map = self.get_torch_rt_mapping(torch_op_map)

        with ms_open(os.path.join(self.json_path, 'mindie_rt_op_mapping.json'), max_size=JSON_FILE_MAX_SIZE) as f:
            op_map = json.load(f)

        op_map = sorted(op_map, key=lambda x: x["id"])

        cur_fuseop = ""
        id_ = 1
        new_op_map = {}

        for item in op_map:
            ge_op = item.get("ge_op")
            rt_layer = item.get("rt_layer")
            jit_node = rt_torch_map.get(rt_layer, None)
            fusion_op = item.get("fusion_op", ge_op)

            if cur_fuseop != fusion_op:
                new_op_map[fusion_op] = {
                    "id": id_,
                    "jit_node": jit_node,
                    "fuse_path": [{"ge_op": ge_op, "jit_node": jit_node}]
                }
                id_ += 1
                fuse_path = [{"ge_op": ge_op, "jit_node": jit_node}]

                if cur_fuseop in new_op_map:
                    new_op_map[cur_fuseop]["fuse_path"] = fuse_path
            else:
                new_op_map[fusion_op]["jit_node"] = jit_node
                fuse_path.append({"ge_op": ge_op, "jit_node": jit_node})

            cur_fuseop = fusion_op

        if cur_fuseop in new_op_map:
            new_op_map[cur_fuseop]["fuse_path"] = fuse_path

        with ms_open(os.path.join(self.json_path, 'op_map_updated.json'), mode="w") as f:
            json.dump(new_op_map, f, indent=4)

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.npu_files is None:
            self.npu_files = []
            find_npu_dump_files(self.path, self.npu_files)
        pattern = re.compile(rf'{re.escape(key)}\.\d')
        matching_files = [file for file in self.npu_files if pattern.search(file)]
        try:
            tensor_file_path = matching_files[0]
        except IndexError:
            npu_tensor = torch.empty(0)
        else:
            bin_dump_data = parse_torchair_dump_data(tensor_file_path)
            npu_tensor = bin_dump_data[1][0]
        finally:
            pass

        return npu_tensor
    
    def get_keys(self) -> set:
        return set(self.key_to_folder.keys())

    def _map_keys_to_folders(self) -> dict:
        key_to_folder = {}
        json_path = os.path.join(self.json_path, 'op_map_updated.json')

        with ms_open(json_path, max_size=JSON_FILE_MAX_SIZE) as f:
            data = json.load(f)
            for fusion_op, details in data.items():
                jit_node = details.get('jit_node', '')
                if jit_node:
                    key_to_folder[fusion_op] = jit_node

        return key_to_folder
  