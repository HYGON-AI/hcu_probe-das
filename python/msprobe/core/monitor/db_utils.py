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

from collections import OrderedDict
from collections.abc import Iterable
from typing import Dict, List, Optional, Set, Tuple

from msprobe.core.common.const import Const, Data2DBConst
from msprobe.core.common.db_manager import DBManager, TrendSql


def update_ordered_dict(main_dict: OrderedDict, new_list: List) -> OrderedDict:
    """Update ordered dictionary with new items"""
    for item in new_list:
        if item not in main_dict:
            main_dict[item] = None
    return main_dict


def get_ordered_list(data: Iterable, order: Iterable) -> List[str]:
    if not isinstance(data, Iterable) or not isinstance(order, Iterable):
        return []
    return [value for value in order if value in data]

def get_tags_from_target(target_name):
    """
    full target name to tags
    
    :param target_name: e.g. layers.22.self_attention.linear_qkv.layer_norm_weight
    :return tags: set of tags
    """
    tags = set()
    parts = target_name.split(Const.SEP)[::-1]
    max_idx = len(parts) - 1
    for i, part in enumerate(parts):
        if not part.isdigit():
            tags.add((part, Data2DBConst.TAG_MODULE))
            continue
        if i == max_idx or parts[i+1].isdigit():  # 确保数字前面有非数字部分, 组成layer标签
            continue
        layer_tag = (f"{parts[i+1]}.{parts[i]}", Data2DBConst.TAG_LAYER)
        tags.add(layer_tag)
    return tags
    

class MonitorDB:
    """Main class for monitoring database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_manager = DBManager(db_path)

    def init_schema(self) -> None:
        """Initialize database schema"""
        self.db_manager.execute_multi_sql(TrendSql.get_table_definition())

    def insert_dimensions(
        self,
        targets: Dict[str,OrderedDict],
    ) -> None:
        """Insert dimension data into database"""
        # Insert targets
        metrics = get_ordered_list(set(targets.keys()), Data2DBConst.METRICS_TRENDVIS_SUPPORTED)
        for metric_name in metrics:
            self.db_manager.insert_data(
                "monitoring_targets",
                [(name, vpp_stage, micro_step)
                for (name, vpp_stage, micro_step) in targets[metric_name]],
                key_list=["target_name", "vpp_stage", "micro_step"]
            )

        # Insert metrics
        self.db_manager.insert_data(
            "monitoring_metrics",
            [(metric,) for metric in metrics],
            key_list=["metric_name"]
        )

    def insert_rows(self, rows):
        table_name = "trend_data"
        inserted = self.db_manager.insert_data(table_name, rows)
        inserted = 0 if inserted is None else inserted
        return inserted

    def init_global_stats_data(self, config: Dict):
        """初始化全局统计表数据"""
        # 准备插入数据
        self.db_manager.execute_sql(
            TrendSql.create_global_stats_table(config)
        )
        
        column_values = []
        for _, column_value in config.items():
            if isinstance(column_value, (list, set)):
                column_values.append(",".join(column_value))
            elif isinstance(column_value, (int, float)):
                column_values.append(column_value)
        self.db_manager.insert_data("global_stats", [column_values,])

    def get_metric_mapping(self) -> Dict[str, Tuple[int, List[str]]]:
        """获取metric名称到ID的映射及对应的统计信息"""
        metric_results = self.db_manager.execute_sql(
            TrendSql.get_metric_mapping_sql()
        )        
        global_stats_result = self.db_manager.execute_sql(
            TrendSql.get_global_stats_sql()
        )
        if not global_stats_result:
            return {}
        
        global_stats_config = global_stats_result[0]
        # 构建映射字典
        metric_mapping = {}
        for row in metric_results:
            metric_name = row["metric_name"]
            metric_id = row["metric_id"]
            
            # 从global_stats中获取该metric对应的统计信息
            stats_value = global_stats_config.get(metric_name)
            ordered_stats = []
            if stats_value:
                stats_list = stats_value.split(",") if isinstance(stats_value, str) else []
                ordered_stats = get_ordered_list(stats_list, Data2DBConst.OP_TRENDVIS_SUPPORTED)                
            metric_mapping[metric_name] = (metric_id, ordered_stats)
        return metric_mapping

    def get_target_mapping(self) -> Dict[Tuple[str, int, int], int]:
        """Get target mapping dictionary"""
        results = self.db_manager.select_data(
            table_name="monitoring_targets",
            columns=["target_id", "target_name", "vpp_stage", "micro_step"]
        )
        if not results:
            return {}
        return {
            (row["target_name"], row["vpp_stage"], row["micro_step"]): row["target_id"]
            for row in results
        }

    def create_trend_data(self, stats):
        self.db_manager.execute_sql(TrendSql.create_trend_table(stats))

    def _get_metric_id(self, metric_name: str) -> Optional[int]:
        """Get metric ID by name"""
        result = self.db_manager.select_data(
            table_name="monitoring_metrics",
            columns=["metric_id"],
            where={"metric_name": metric_name}
        )
        return result[0]["metric_id"] if result else None

    def extract_tags_from_processed_targets(self, targets:dict , metric_id_dict: dict, target_dict: dict):
        """从已处理的targets中提取标签并建立映射关系"""

        for metric_name in targets:
            tag_info_to_id = {}
            mapping_data = []
            new_tags_data = set()
            target_to_tags = {}
            metric_id = metric_id_dict.get(metric_name, [None,])[0]
            for target in targets[metric_name]:
                target_id = target_dict[target]
                name, _, _ = target
                tags = get_tags_from_target(name)
                new_tags_data = new_tags_data.union(tags)
                target_to_tags[target_id] = tags

            # 插入新标签
            if not new_tags_data:
                continue
            self.db_manager.insert_data(
                "monitoring_tags",
                [(tag_name, category, metric_id)
                    for tag_name, category in new_tags_data],
                key_list=["tag_name", "category", "metric_id"]
            )

            # 重新获取tag_id用于映射
            tag_records = self.db_manager.select_data(
                "monitoring_tags",
                columns=["tag_id", "tag_name", "category", "metric_id"]
            )
            for row in tag_records:
                tag_info_to_id[(row["tag_name"], row["category"],
                                row["metric_id"])] = row["tag_id"]

            for target_id, tags_list in target_to_tags.items():
                for tag_name, category in tags_list:
                    tag_id = tag_info_to_id.get(
                        (tag_name, category, metric_id))
                    if tag_id:
                        mapping_data.append((target_id, tag_id))

            # 插入映射关系
            if mapping_data:
                self.db_manager.insert_data(
                    "tag_target_mapping",
                    mapping_data,
                    key_list=["target_id", "tag_id"]
                )