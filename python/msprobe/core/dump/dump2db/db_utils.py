# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

from collections import defaultdict

from msprobe.core.common.const import Data2DBConst
from msprobe.core.common.db_manager import DBManager
from msprobe.core.common.log import logger


def parse_full_key(full_key):
    """
    Module.blocks.3.layer.1.conv2.Conv2d.input.2
    """
    tags = set()
    parts = full_key.split('.')
    if len(parts) < 3:
        logger.debug(
            f"parse_full_key: key '{full_key}' has less than 3 parts,"
            f"returning empty set of tags")
        return set()
    # 关联index标签 full_key中最后一部分，如 'input.2'
    tags.add((".".join(parts[-2:]), Data2DBConst.TAG_INDEX))

    # 关联默认标签
    filtered_parts = []
    for part in parts[:-2]:
        if part not in Data2DBConst.DEFAULT_TAGS:
            filtered_parts.append(part)
        else:
            tags.add((part, Data2DBConst.TAG_DEFAULT))

    if not filtered_parts:
        return set()

    processed_indices = [-1]
    for i, part in enumerate(filtered_parts):
        if part.isdigit():
            if i == 0 or i - 1 in processed_indices:  # 确保数字前面有非数字部分, 组成layer标签
                processed_indices.append(i)
                continue
            layer_tag = (
                f"{filtered_parts[i-1]}.{filtered_parts[i]}.", Data2DBConst.TAG_LAYER)
            tags.add(layer_tag)

            before_layer = ".".join(
                filtered_parts[processed_indices[-1] + 1: i - 1])
            if before_layer:
                module_tag = (before_layer, Data2DBConst.TAG_MODULE)
                tags.add(module_tag)
            processed_indices.append(i)

    if processed_indices[-1] != len(filtered_parts) - 1 and \
            ".".join(filtered_parts[processed_indices[-1] + 1:]):
        module_tag = (
            ".".join(filtered_parts[processed_indices[-1] + 1:]), Data2DBConst.TAG_MODULE)
        tags.add(module_tag)
    if len(processed_indices) == 1:
        api_tag = (".".join(filtered_parts), Data2DBConst.TAG_FUNCTION)
        tags.add(api_tag)
    return tags


class DumpSql:
    """dump场景数据库表参数类"""

    @staticmethod
    def create_monitoring_targets_table():
        """监测目标表"""
        return """
        CREATE TABLE IF NOT EXISTS monitoring_targets (
            target_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name TEXT NOT NULL,
            vpp_stage INTEGER NOT NULL,
            micro_step INTEGER NOT NULL DEFAULT 0,
            UNIQUE(target_name, vpp_stage, micro_step) 
        )"""

    @staticmethod
    def create_monitoring_metrics_table():
        """监测指标表"""
        return """
        CREATE TABLE IF NOT EXISTS monitoring_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT UNIQUE NOT NULL
        )"""

    @staticmethod
    def create_global_stats_table(columns_config):
        """根据字典配置创建全局统计表"""
        # 根据字典的键值类型动态创建列
        column_definitions = []
        for column_name, column_value in columns_config.items():
            if isinstance(column_value, (list, set)):
                column_definitions.append(f"{column_name} TEXT DEFAULT NULL")
            elif isinstance(column_value, (int, float)):
                column_definitions.append(f"{column_name} INTEGER DEFAULT 0")
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS global_stats (
            {', '.join(column_definitions)}
        )"""
        return create_sql

    @staticmethod
    def create_tags_table():
        # 新增：标签表
        return """
        CREATE TABLE IF NOT EXISTS monitoring_tags (
            tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_name TEXT NOT NULL,
            category TEXT NOT NULL,
            metric_id INTEGER NOT NULL,
            UNIQUE(tag_name, category, metric_id),
            FOREIGN KEY (metric_id) REFERENCES monitoring_metrics(metric_id)
        )"""

    @staticmethod
    def create_tag_mapping_table():
        return """
        CREATE TABLE IF NOT EXISTS tag_target_mapping (
            tag_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            PRIMARY KEY (tag_id, target_id),
            FOREIGN KEY (tag_id) REFERENCES monitoring_tags(tag_id),
            FOREIGN KEY (target_id) REFERENCES monitoring_target(target_id)
        )"""

    @staticmethod
    def create_trend_table(stats):
        stat_columns = [f"{stat} REAL DEFAULT NULL" for stat in stats]
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS trend_data (
            rank INTEGER NOT NULL,
            step INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            metric_id INTEGER NOT NULL,
            {', '.join(stat_columns)},
            PRIMARY KEY (rank, step, target_id, metric_id),
            FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id),
            FOREIGN KEY (metric_id) REFERENCES monitoring_metrics(metric_id)
        ) WITHOUT ROWID"""
        return create_sql
    
    @classmethod
    def get_table_definition(cls, table_name=""):
        """
        获取表定义SQL
        :param table_name: 表名
        :return: 建表SQL语句
        :raises ValueError: 当表名不存在时
        """
        table_creators = {
            "monitoring_targets": cls.create_monitoring_targets_table,
            "monitoring_metrics": cls.create_monitoring_metrics_table,
            "monitoring_tags": cls.create_tags_table,
            "tag_target_mapping": cls.create_tag_mapping_table,
        }
        if not table_name:
            return [table_creators.get(table, lambda x: "")() for table in table_creators]
        if table_name not in table_creators:
            raise ValueError(f"Unsupported table name: {table_name}")
        return table_creators[table_name]()


class DumpDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_manager = DBManager(db_path)

        # 缓存
        self._metric_id_cache = {}  # metric_name -> metric_id
        self._stats_cache = {}      # metric_name -> stats list
        self._processed_targets = defaultdict(dict)  # 记录所有metric_id下的targets
        self._new_targets = defaultdict(list)  # 记录新增targets
        self._init_schema()
    
    def init_global_stats_data(self, config):
        """初始化全局统计表数据"""
        # 准备插入数据
        self.db_manager.execute_sql(
            DumpSql.create_global_stats_table(config)
        )
        
        column_values = []
        for _, column_value in config.items():
            if isinstance(column_value, (list, set)):
                column_values.append(",".join(column_value))
            elif isinstance(column_value, (int, float)):
                column_values.append(column_value)
        self.db_manager.insert_data("global_stats", [column_values,])

    def get_metric_id(self, metric_name: str):
        """Get metric ID by name"""
        if metric_name in self._metric_id_cache:
            return self._metric_id_cache[metric_name]
        result = self.db_manager.select_data(
            table_name="monitoring_metrics",
            columns=["metric_id"],
            where={"metric_name": metric_name}
        )
        metric_id = result[0]["metric_id"] if result else None
        self._metric_id_cache[metric_name] = metric_id
        return metric_id

    def cache_targets(self, target, metric_id):
        if target not in self._processed_targets[metric_id]:
            self._processed_targets[metric_id][target] = {'id': 0}
            self._new_targets[metric_id].append(target)
        return self._processed_targets[metric_id][target]

    def batch_insert_targets(self):
        """批量插入targets并通过rowid范围获取ID"""

        for metric_id in sorted(self._new_targets.keys()):
            targets = self._new_targets[metric_id]
            if not targets:
                continue
            self.db_manager.insert_data("monitoring_targets", list(targets), key_list=[
                                        "target_name", "vpp_stage", "micro_step"])
            results = self.db_manager.select_data(
                table_name="monitoring_targets",
                columns=["target_id", "target_name", "vpp_stage", "micro_step"]
            )
            # 更新缓存
            for row in results:
                cache_key = (row["target_name"],
                             row["vpp_stage"], row["micro_step"])
                if cache_key in self._processed_targets[metric_id]:
                    self._processed_targets[metric_id][cache_key]['id'] = row["target_id"]
        self._new_targets = defaultdict(list)

    def batch_insert_data(self, batch_data):
        """批量插入数据"""
        if not batch_data:
            return
        # 刷新target id
        for row in batch_data:
            if len(row) < 3 or not isinstance(row[2], dict):
                return
            row[2] = row[2]["id"]

        self.db_manager.insert_data("trend_data", batch_data)

    def extract_tags_from_processed_targets(self):
        """从已处理的targets中提取标签并建立映射关系"""

        for metric_id in self._processed_targets:
            tag_info_to_id = {}
            mapping_data = []
            target_to_tags = {}
            new_tags_data = set()
            for target, info in self._processed_targets[metric_id].items():
                target_name, _, _ = target
                target_id = info.get("id")

                if target_id is None:
                    continue
                # 从target_name中提取标签
                tags = parse_full_key(target_name)
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

    def _init_schema(self) -> None:
        """Initialize database schema"""
        self.db_manager.execute_multi_sql(DumpSql.get_table_definition())
        self.db_manager.execute_sql(DumpSql.create_trend_table(Data2DBConst.ORDERED_STAT))
        for metric_name in Data2DBConst.METRICS:
            self.db_manager.insert_data("monitoring_metrics", [
                                        (metric_name,)], ["metric_name"])
