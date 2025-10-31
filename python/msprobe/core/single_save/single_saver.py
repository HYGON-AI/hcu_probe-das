# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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
from collections import defaultdict

from msprobe.core.common.file_utils import create_directory, save_json
from msprobe.core.common.const import Const
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger


support_nested_data_type = (list, tuple, dict)


class SingleSave:
    _instance = None

    def __new__(cls, dump_path, fmk=Const.PT_FRAMEWORK):
        if cls._instance is None:
            cls._instance = super(SingleSave, cls).__new__(cls)
            FmkAdp.set_fmk(fmk)
            create_directory(dump_path)

            cls._instance.dump_path = dump_path
            cls._instance.rank = FmkAdp.get_rank_id()
            cls._instance.step_count = 0
            cls._instance.tag_count = defaultdict(int)
        return cls._instance

    @staticmethod
    def _analyze_tensor_data(data, data_name=None, save_dir=None):
        '''
        data: Tensor
        return:
            result_data: with keys {"max", "min", "mean", "norm", "shape"}
        '''
        result_data = {}
        result_data["max"] = FmkAdp.tensor_max(data)
        result_data["min"] = FmkAdp.tensor_min(data)
        result_data["mean"] = FmkAdp.tensor_mean(data)
        result_data["norm"] = FmkAdp.tensor_norm(data)
        result_data["shape"] = list(data.shape)
        if save_dir is not None and data_name is not None:
            real_save_path = os.path.join(save_dir, data_name + ".npy")
            FmkAdp.save_tensor(data, real_save_path)
        return result_data

    @classmethod
    def save_config(cls, data):
        dump_file = os.path.join(cls._instance.dump_path, 'configurations.json')
        save_json(dump_file, data, indent=4)

    @classmethod
    def save_ex(cls, data, micro_batch=None):
        '''
        data: dict{str: Union[Tensor, tuple, list]}

        return: void
        '''

        instance = cls._instance

        if not isinstance(data, dict):
            logger.warning("SingleSave data type not valid, "
                             "should be dict. "
                             "Skip current save process.")
            return
        for key, value in data.items():
            if not isinstance(key, str):
                logger.warning("key should be string when save data")
                continue
            if not isinstance(value, support_nested_data_type) and not FmkAdp.is_tensor(value):
                logger.warning(f"value should be {support_nested_data_type} or Tensor when save data")
                continue
            real_dump_dir = os.path.join(
                instance.dump_path, 
                "data", 
                key, 
                f"step{instance.step_count}", 
                f"rank{instance.rank}")
            if micro_batch is not None:
                real_dump_dir = os.path.join(real_dump_dir, f"micro_step{micro_batch}")
            create_directory(real_dump_dir)

            if FmkAdp.is_tensor(value):
                result = cls._analyze_tensor_data(value, key, real_dump_dir)
            elif isinstance(value, (tuple, list)):
                result = cls._analyze_list_tuple_data(value, key, real_dump_dir)
            elif isinstance(value, dict):
                result = cls._analyze_dict_data(value, key, real_dump_dir)

            result_json = {"data": result}
            json_path = os.path.join(real_dump_dir, key + ".json")
            save_json(json_path, result_json, indent=4)


    @classmethod
    def step(cls):
        instance = cls._instance
        instance.tag_count = defaultdict(int)
        instance.step_count += 1

    @classmethod
    def save(cls, data):
        instance = cls._instance
        if not isinstance(data, dict):
            logger.warning("SingleSave data type not valid, "
                             "should be dict. "
                             "Skip current save process.")
            return
        for key, value in data.items():
            cls.save_ex({key: value}, micro_batch=instance.tag_count[key])
            instance.tag_count[key] += 1

    @classmethod
    def _analyze_list_tuple_data(cls, data, data_name=None, save_dir=None):
        lst = []
        for index, element in enumerate(data):
            if not FmkAdp.is_tensor(element):
                raise TypeError(f"SingleSave: Unsupported type: {type(element)}")
            element_name = data_name + "." + str(index)
            lst.append(cls._analyze_tensor_data(element, element_name, save_dir))
        return lst

    @classmethod
    def _analyze_dict_data(cls, data, data_name=None, save_dir=None):
        result_data = {}
        for key, value in data.items():
            if not FmkAdp.is_tensor(value):
                raise TypeError(f"SingleSave: Unsupported type: {type(value)}")
            key_name = data_name + "." + str(key)
            result_data[key] = cls._analyze_tensor_data(value, key_name, save_dir)
        return result_data
