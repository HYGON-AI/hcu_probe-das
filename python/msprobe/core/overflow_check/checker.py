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

from typing import Dict, List, Optional, Any

from msprobe.core.common.const import Const

from msprobe.core.overflow_check.abnormal_scene import InputAnomalyOutputNormalScene, InputAnomalyOutputAnomalyScene, \
    InputNormalOutputAnomalyScene, NumericalMutationScene, AnomalyScene
from msprobe.core.overflow_check.api_info import APIInfo
from msprobe.core.overflow_check.filter import IgnoreFilter
from msprobe.core.overflow_check.level import OverflowLevel


class StatisticsFields:
    """统计字段常量类"""
    CRITICAL_APIS = 'critical_apis'
    HIGH_PRIORITY_APIS = 'high_priority_apis'
    MEDIUM_PRIORITY_APIS = 'medium_priority_apis'
    ANOMALY_DETAILS = 'anomaly_details'

    # 所有字段
    ALL_FIELDS = [CRITICAL_APIS, HIGH_PRIORITY_APIS, MEDIUM_PRIORITY_APIS, ANOMALY_DETAILS]


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, dump_data: Dict):
        """
            初始化检测器，并保存dump_data
        Args:
            dump_data: 数据格式如下
                {
                    "api/module": {statistics}
                }
        """
        self.dump_data = dump_data
        self.ignore_filter = IgnoreFilter()
        self.scene_types = [
            InputNormalOutputAnomalyScene,      # 输入正常，输出异常
            InputAnomalyOutputAnomalyScene,     # 输入异常，输出异常
            InputAnomalyOutputNormalScene,      # 输入异常，输出正常
            NumericalMutationScene              # 输出较输入值突变
        ]
        self.anomaly_scenes: Dict[str, AnomalyScene] = dict()

    @staticmethod
    def _create_api_info(api_name: str, data: Dict) -> APIInfo:
        """从原始数据创建APIInfo实例"""
        return APIInfo(
            api_name=api_name,
            input_args=data.get(Const.INPUT_ARGS, data.get(Const.INPUT, [])),
            input_kwargs=data.get(Const.INPUT_KWARGS, {}),
            output_data=data.get(Const.OUTPUT, [])
        )

    def get_statistics(self) -> Dict[str, List]:
        """获取统计信息

        使用StatisticsFields类统一管理字段名称，避免硬编码

        Returns:
            Dict[str, List]: 包含各优先级API列表和异常详情的字典
        """
        stats = {field: [] for field in StatisticsFields.ALL_FIELDS}

        # 定义rank到结果key的映射关系
        rank_to_key = {
            OverflowLevel.CRITICAL: StatisticsFields.CRITICAL_APIS,
            OverflowLevel.HIGH: StatisticsFields.HIGH_PRIORITY_APIS,
            OverflowLevel.MEDIUM: StatisticsFields.MEDIUM_PRIORITY_APIS
        }

        for scene in self.anomaly_scenes.values():
            stats[StatisticsFields.ANOMALY_DETAILS].append(scene.get_details())
            # 根据rank分类API
            key = rank_to_key.get(scene.rank, None)
            if not key:
                stats[key].append(scene.api_name)

        return stats

    def analyze(self):
        """
            按照异常场景对调用数据进行分析
        Returns:
            返回类本身，若不进行过滤，则仅调用analyze即可
        """
        # 遍历data item
        for api_name, data in self.dump_data.items():
            api_info = self._create_api_info(api_name, data)

            # 每种都进行检测，可能涉及多种命中，原则如下：
            #   - 就高原则
            #   - 优先原则，数据异常放最后检测
            for scene_type in self.scene_types:
                scene = scene_type(api_info)
                if hasattr(scene, 'matches') and scene.matches():
                    self.anomaly_scenes[api_name] = scene
                    break  # 直接跳过，就高原则
        return self

    def filter(self):
        """
            对误检数据进行过滤
        Returns:
            检查checker自身，方便链式调用
        """
        result = dict()
        for api_name, scene in self.anomaly_scenes.items():
            if self.ignore_filter.apply_filter(scene.api_data):
                continue
            result[api_name] = scene
        self.anomaly_scenes = result
        return self

    def overflow_result(self) -> Dict[str, AnomalyScene]:
        return self.anomaly_scenes

    def has_overflow(self, api_name: str) -> bool:
        return api_name in self.anomaly_scenes.keys()

    def get_overflow_level(self, api_name: str) -> Optional[Any]:
        scene = self.anomaly_scenes.get(api_name, None)
        return scene.rank if scene else None
