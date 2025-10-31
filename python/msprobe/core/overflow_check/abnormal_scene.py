# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

from typing import List, Dict, Union, Any

import numpy as np

from msprobe.core.overflow_check.api_info import APIInfo
from msprobe.core.overflow_check.level import OverflowLevel
from msprobe.core.overflow_check.utils import has_nan_inf
from msprobe.core.common.decorator import recursion_depth_decorator


class AnomalyScene:
    """异常场景的基类"""

    def __init__(self, api_info: APIInfo):
        self.api_name = api_info.api_name
        self.api_data = api_info

    @property
    def rank(self) -> OverflowLevel:
        """获取异常等级"""
        raise NotImplementedError

    @staticmethod
    @recursion_depth_decorator("AbnormalScene: AnomalyScene._has_anomaly")
    def _has_anomaly(data: Union[Dict, Any]) -> bool:
        """检查张量是否包含异常值"""
        if isinstance(data, dict):
            return has_nan_inf(data)
        elif isinstance(data, list):
            return any(AnomalyScene._has_anomaly(x) for x in data)
        return False

    def get_details(self) -> Dict:
        """获取异常详情"""
        return {
            'api_name': self.api_name,
            'rank': self.rank.value,
            'scene_type': self.__class__.__name__,
            'input_args_anomaly_indices': self._get_anomaly_indices_from_list(self.api_data.input_args),
            'input_kwargs_anomaly_keys': self._get_anomaly_keys_from_dict(self.api_data.input_kwargs),
            'output_anomaly_indices': self._get_anomaly_indices_from_list(self.api_data.output_data)
        }

    def matches(self) -> bool:
        """
            待子类实现对应匹配逻辑
        Returns:

        """
        raise NotImplementedError

    def _get_anomaly_indices_from_list(self, data_list: List[Dict]) -> List[int]:
        return [i for i, data in enumerate(data_list) if self._has_anomaly(data)]

    def _get_anomaly_keys_from_dict(self, data_dict: Dict) -> List[str]:
        return [key for key, data in data_dict.items() if self._has_anomaly(data)]


class InputOutputAnomalyScene(AnomalyScene):
    """输入输出异常检测的基类"""
    def has_input_anomaly(self) -> bool:
        """检查输入是否有异常（包括args和kwargs）"""
        # args
        args_anomaly = any(self._has_anomaly(x) for x in self.api_data.input_args)
        # kwargs
        kwargs_anomaly = any(self._has_anomaly(x) for x in self.api_data.input_kwargs.values())
        return args_anomaly or kwargs_anomaly

    def has_output_anomaly(self) -> bool:
        """检查输出是否有异常"""
        return any(self._has_anomaly(x) for x in self.api_data.output_data)

    def matches(self) -> bool:
        """判断是否匹配该场景"""
        raise NotImplementedError


class InputAnomalyOutputNormalScene(InputOutputAnomalyScene):
    """输入异常，输出正常场景"""

    @property
    def rank(self) -> OverflowLevel:
        return OverflowLevel.MEDIUM

    def matches(self) -> bool:
        return self.has_input_anomaly() and not self.has_output_anomaly()


class InputAnomalyOutputAnomalyScene(InputOutputAnomalyScene):
    """输入异常，输出异常场景"""

    @property
    def rank(self) -> OverflowLevel:
        return OverflowLevel.HIGH

    def matches(self) -> bool:
        return self.has_input_anomaly() and self.has_output_anomaly()


class InputNormalOutputAnomalyScene(InputOutputAnomalyScene):
    """输入正常，输出异常场景"""

    @property
    def rank(self) -> OverflowLevel:
        return OverflowLevel.CRITICAL

    def matches(self) -> bool:
        return not self.has_input_anomaly() and self.has_output_anomaly()


class NumericalMutationScene(AnomalyScene):
    """
        检查数值突变，统计输入args、kwargs中norm值，同时统计输出的norm最大值，计算差异，大于 threshold 则认为是异常情况
    """
    def __init__(self, api_info: APIInfo, threshold: float = 100.0):
        super().__init__(api_info)
        self.threshold = threshold

    @property
    def rank(self) -> OverflowLevel:
        return OverflowLevel.HIGH

    @staticmethod
    def _get_tensor_norms(data_list: List[Dict]) -> List[float]:
        norms = []
        for data in data_list:
            if isinstance(data, dict) and data.get('type') == 'torch.Tensor':
                norm = data.get('Norm')
                if norm is not None and not np.isnan(norm):
                    norms.append(norm)
        return norms

    @staticmethod
    def _get_kwargs_norms(data_dict: Dict) -> List[float]:
        """
            获取kwargs中张量的范数列表
        Args:
            data_dict:
        Returns:
        """
        norms = []
        for data in data_dict.values():
            if isinstance(data, dict) and data.get('type') == 'torch.Tensor':
                norm = data.get('Norm')
                if norm is not None and not np.isnan(norm):
                    norms.append(norm)
        return norms

    def matches(self) -> bool:
        """
            继承父类函数，实现数值突变检查
        Returns:
        """
        # 收集所有输入的范数
        input_norms = (self._get_tensor_norms(self.api_data.input_args) +
                       self._get_kwargs_norms(self.api_data.input_kwargs))
        # 收集所有输出的范数
        output_norms = self._get_tensor_norms(self.api_data.output_data)

        if not input_norms or not output_norms:
            return False

        max_input = max(input_norms)
        max_output = max(output_norms)

        if max_input == 0:
            return max_output > self.threshold
        return max_output / max_input > self.threshold

    def get_details(self) -> Dict:
        details = super().get_details()
        details.update({
            'threshold': self.threshold,
            'scale_change_detected': self.matches()
        })
        return details
