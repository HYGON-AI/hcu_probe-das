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
from msprobe.core.common.db_manager import DBManager, check_identifier_safety
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
        """监控目标表"""
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
        """监控指标表"""
        return """
        CREATE TABLE IF NOT EXISTS monitoring_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT UNIQUE NOT NULL
        )"""

    @staticmethod
    def create_metric_stats_table():
        """指标统计表"""
        return """
        CREATE TABLE IF NOT EXISTS metric_stats (
            metric_id INTEGER NOT NULL,
            stat_name TEXT NOT NULL,
            PRIMARY KEY (metric_id, stat_name),
            FOREIGN KEY (metric_id) REFERENCES monitoring_metrics(metric_id)
        ) WITHOUT ROWID"""

    @staticmethod
    def create_global_stat_table():
        return """
        CREATE TABLE IF NOT EXISTS global_stats (
            stat_name TEXT PRIMARY KEY,
            stat_value INTEGER NOT NULL
        ) WITHOUT ROWID"""

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
            "metric_stats": cls.create_metric_stats_table,
            "global_stats": cls.create_global_stat_table,
            "monitoring_tags": cls.create_tags_table,
            "tag_target_mapping": cls.create_tag_mapping_table
        }
        if not table_name:
            return [table_creators.get(table, lambda x: "")() for table in table_creators]
        if table_name not in table_creators:
            raise ValueError(f"Unsupported table name: {table_name}")
        return table_creators[table_name]()

    @classmethod
    def get_metric_table_definition(cls, table_name, stats, start_step, end_step):
        check_identifier_safety(table_name)

        stat_columns = [f"{stat} REAL DEFAULT NULL" for stat in stats]
        step_column = f"""step INTEGER NOT NULL CHECK(step BETWEEN {start_step} 
                AND {end_step}),"""
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                rank INTEGER NOT NULL,
                {step_column}
                target_id INTEGER NOT NULL,
                {', '.join(stat_columns)},
                PRIMARY KEY (rank, step, target_id),
                FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id)
            ) WITHOUT ROWID
            """
        return create_sql


class DumpDB:
    def __init__(self, db_path, step_partition):
        self.db_path = db_path
        self.step_partition = step_partition
        self.db_manager = DBManager(db_path)

        # 缓存
        self._metric_id_cache = {}  # metric_name -> metric_id
        self._stats_cache = {}      # metric_name -> stats list
        self._processed_targets = defaultdict(dict)  # 记录所有metric_id下的targets
        self._new_targets = defaultdict(list)  # 记录新增targets
        self._init_schema()

    @staticmethod
    def get_metric_table_name(metric_id, step_start, step_end):
        return f"metric_{metric_id}_step_{step_start}_{step_end}"

    def update_global_stats(self, max_rank=None, min_step=None, max_step=None) -> None:
        """Update global statistics"""
        updates = [
            ("max_rank", max_rank),
            ("min_step", min_step),
            ("max_step", max_step)
        ]
        for stat_name, value in updates:
            if not value:
                continue
            self.db_manager.update_data(
                table_name="global_stats",
                updates={"stat_value": value},
                where={"stat_name": stat_name}
            )

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

    def create_all_metric_tables(self, min_step, max_step):
        # 预先创建所有需要的分区表
        for metric_name in Data2DBConst.METRICS:
            metric_id = self.get_metric_id(metric_name)
            stats = Data2DBConst.ORDERED_STAT

            # 为每个step分区创建表
            for step in range(min_step, max_step + 1, self.step_partition):
                partition_start = (
                    step // self.step_partition) * self.step_partition
                table_name = DumpDB.get_metric_table_name(
                    metric_id,
                    partition_start,
                    partition_start + self.step_partition - 1
                )
                create_sql = DumpSql.get_metric_table_definition(
                    table_name, stats, partition_start, partition_start + self.step_partition - 1)
                self.db_manager.execute_sql(create_sql)

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
        for table_name, rows in batch_data.items():
            if not rows:
                continue
            if not self.db_manager.table_exists(table_name):
                raise RuntimeError(
                    f"{table_name} not existed in {self.db_path}")
            # 刷新target id
            for row in rows:
                if len(row) < 3 or not isinstance(row[2], dict):
                    continue
                row[2] = row[2]["id"]

            self.db_manager.insert_data(table_name, rows)
        return

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

    def _init_metrics_stats(self) -> None:
        for metric_name in Data2DBConst.METRICS:
            self.db_manager.insert_data("monitoring_metrics", [
                                        (metric_name,)], ["metric_name"])
            metric_id = self.get_metric_id(metric_name)
            for stat in Data2DBConst.ORDERED_STAT:
                self.db_manager.insert_data("metric_stats", [(metric_id, stat)], [
                                            "metric_id", "stat_name"])

    def _init_schema(self) -> None:
        """Initialize database schema"""
        self.db_manager.execute_multi_sql(DumpSql.get_table_definition())

        # Insert initial global stats
        global_stats = [
            ('max_rank', 0),
            ('min_step', 0),
            ('max_step', 0),
            ('step_partition_size', self.step_partition)
        ]
        self.db_manager.insert_data("global_stats", global_stats)
        self._init_metrics_stats()
