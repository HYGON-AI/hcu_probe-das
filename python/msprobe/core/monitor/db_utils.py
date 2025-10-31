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
from collections import OrderedDict
from collections.abc import Iterable
from typing import Dict, List, Optional, Set, Tuple

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.db_manager import DBManager


def update_ordered_dict(main_dict: OrderedDict, new_list: List) -> OrderedDict:
    """Update ordered dictionary with new items"""
    for item in new_list:
        if item not in main_dict:
            main_dict[item] = None
    return main_dict


def get_ordered_stats(stats: Iterable) -> List[str]:
    """Get statistics in predefined order"""
    if not isinstance(stats, Iterable):
        return []
    return [stat for stat in MonitorConst.OP_MONVIS_SUPPORTED if stat in stats]


class MonitorSql:
    """数据库表参数类"""

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
    def get_metric_mapping_sql():
        return """
        SELECT m.metric_id, m.metric_name, GROUP_CONCAT(ms.stat_name) as stats 
        FROM monitoring_metrics m 
        LEFT JOIN metric_stats ms ON m.metric_id = ms.metric_id
        GROUP BY m.metric_id
        """

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
        }
        if not table_name:
            return [table_creators.get(table, lambda x: "")() for table in table_creators]
        if table_name not in table_creators:
            raise ValueError(f"Unsupported table name: {table_name}")
        return table_creators[table_name]()

    @classmethod
    def get_metric_table_definition(cls, table_name, stats, patition=None):
        stat_columns = [f"{stat} REAL DEFAULT NULL" for stat in stats]
        if patition and len(patition) == 2:
            partition_start_step, partition_end_step = patition
            step_column = f"""step INTEGER NOT NULL CHECK(step BETWEEN {partition_start_step} 
                    AND {partition_end_step}),"""
        else:
            step_column = "step INTEGER NOT NULL"
        create_sql = f"""
            CREATE TABLE {table_name} (
                rank INTEGER NOT NULL,
                {step_column}
                target_id INTEGER NOT NULL,
                {', '.join(stat_columns)},
                PRIMARY KEY (rank, step, target_id),
                FOREIGN KEY (target_id) REFERENCES monitoring_targets(target_id)
            ) WITHOUT ROWID
            """
        return create_sql


class MonitorDB:
    """Main class for monitoring database operations"""

    def __init__(self, db_path: str, step_partition_size: int = 500):
        self.db_path = db_path
        self.db_manager = DBManager(db_path)
        self.step_partition_size = step_partition_size

    def get_metric_table_name(self, metric_id: int, step: int) -> str:
        """Generate metric table name"""
        step_start = (
            step // self.step_partition_size) * self.step_partition_size
        step_end = step_start + self.step_partition_size - 1
        return f"metric_{metric_id}_step_{step_start}_{step_end}", step_start, step_end

    def init_schema(self) -> None:
        """Initialize database schema"""
        self.db_manager.execute_multi_sql(MonitorSql.get_table_definition())

        # Insert initial global stats
        global_stats = [
            ('max_rank', 0),
            ('min_step', 0),
            ('max_step', 0),
            ('step_partition_size', self.step_partition_size)
        ]
        self.db_manager.insert_data("global_stats", global_stats)

    def insert_dimensions(
        self,
        targets: OrderedDict,
        metrics: Set[str],
        metric_stats: Dict[str, Set[str]],
        min_step: Optional[int] = None,
        max_step: int = None,
    ) -> None:
        """Insert dimension data into database"""
        # Insert targets
        self.db_manager.insert_data(
            "monitoring_targets",
            [(name, vpp_stage, micro_step)
             for (name, vpp_stage, micro_step) in targets],
            key_list=["target_name", "vpp_stage", "micro_step"]
        )

        # Insert metrics
        self.db_manager.insert_data(
            "monitoring_metrics",
            [(metric,) for metric in metrics],
            key_list=["metric_name"]
        )

        # Insert metric-stat relationships
        for metric, stats in metric_stats.items():
            metric_id = self._get_metric_id(metric)
            ordered_stats = get_ordered_stats(stats)

            self.db_manager.insert_data(
                "metric_stats",
                [(metric_id, stat) for stat in ordered_stats],
                key_list=["metric_id", "stat_name"]
            )

            # Create metric tables for each partition
            if min_step is not None and max_step is not None:
                first_partition = min_step // self.step_partition_size
                last_partition = max_step // self.step_partition_size

                for partition in range(first_partition, last_partition + 1):
                    step_start = partition * self.step_partition_size
                    self.create_metric_table(
                        metric_id, step_start, ordered_stats)

    def insert_rows(self, table_name, rows):
        if not self.db_manager.table_exists(table_name):
            raise RuntimeError(f"{table_name} not existed in {self.db_path}")
        inserted = self.db_manager.insert_data(table_name, rows)
        inserted = 0 if inserted is None else inserted
        return inserted

    def create_metric_table(self, metric_id: int, step: int, stats: List[str]) -> str:
        """Create metric table for a specific partition"""
        table_name, partition_start_step, partition_end_step = self.get_metric_table_name(
            metric_id,
            step
        )
        if self.db_manager.table_exists(table_name):
            return table_name

        create_sql = MonitorSql.get_metric_table_definition(
            table_name, stats, patition=(
                partition_start_step, partition_end_step)
        )
        self.db_manager.execute_sql(create_sql)
        return table_name

    def update_global_stats(self, max_rank: int = None, min_step: Optional[int] = None, max_step: int = None) -> None:
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

    def get_metric_mapping(self) -> Dict[str, Tuple[int, List[str]]]:
        """Get metric name to ID mapping with statistics"""
        results = self.db_manager.execute_sql(
            MonitorSql.get_metric_mapping_sql()
        )

        return {
            row["metric_name"]: (
                row["metric_id"],
                get_ordered_stats(row["stats"].split(",")
                                  ) if row["stats"] else []
            ) for row in results
        }

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

    def _get_metric_id(self, metric_name: str) -> Optional[int]:
        """Get metric ID by name"""
        result = self.db_manager.select_data(
            table_name="monitoring_metrics",
            columns=["metric_id"],
            where={"metric_name": metric_name}
        )
        return result[0]["metric_id"] if result else None
