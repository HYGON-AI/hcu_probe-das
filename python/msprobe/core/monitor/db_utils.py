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
    def get_metric_mapping_sql():
        """从monitoring_metrics表获取所有metric名称和ID"""
        return """
        SELECT metric_id, metric_name
        FROM monitoring_metrics
        """

    @staticmethod
    def get_global_stats_sql():
        """从global_stats表获取最新的统计配置"""
        return """
        SELECT * FROM global_stats 
        ORDER BY ROWID DESC 
        LIMIT 1
        """

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
        }
        if not table_name:
            return [table_creators.get(table, lambda x: "")() for table in table_creators]
        if table_name not in table_creators:
            raise ValueError(f"Unsupported table name: {table_name}")
        return table_creators[table_name]()


class MonitorDB:
    """Main class for monitoring database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_manager = DBManager(db_path)

    def init_schema(self) -> None:
        """Initialize database schema"""
        self.db_manager.execute_multi_sql(MonitorSql.get_table_definition())

    def insert_dimensions(
        self,
        targets: OrderedDict,
        metrics: Set[str]
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

    def insert_rows(self, rows):
        table_name = "trend_data"
        inserted = self.db_manager.insert_data(table_name, rows)
        inserted = 0 if inserted is None else inserted
        return inserted

    def init_global_stats_data(self, config: Dict):
        """初始化全局统计表数据"""
        # 准备插入数据
        self.db_manager.execute_sql(
            MonitorSql.create_global_stats_table(config)
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
            MonitorSql.get_metric_mapping_sql()
        )        
        global_stats_result = self.db_manager.execute_sql(
            MonitorSql.get_global_stats_sql()
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
                ordered_stats = get_ordered_stats(stats_list)                
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
        self.db_manager.execute_sql(MonitorSql.create_trend_table(stats))

    def _get_metric_id(self, metric_name: str) -> Optional[int]:
        """Get metric ID by name"""
        result = self.db_manager.select_data(
            table_name="monitoring_metrics",
            columns=["metric_id"],
            where={"metric_name": metric_name}
        )
        return result[0]["metric_id"] if result else None
