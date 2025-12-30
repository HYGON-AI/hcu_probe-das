import unittest
import os
import re
import tempfile
from collections import OrderedDict
from unittest.mock import patch

from msprobe.core.common.const import MonitorConst
from msprobe.core.monitor.db_utils import MonitorDB, MonitorSql, update_ordered_dict, get_ordered_stats


def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text)


class TestDBUtils(unittest.TestCase):
    def test_update_ordered_dict(self):
        """测试update_ordered_dict函数"""
        main_dict = OrderedDict([('a', 1), ('b', 2)])
        new_list = ['b', 'c', 'd']

        result = update_ordered_dict(main_dict, new_list)

        self.assertEqual(list(result.keys()), ['a', 'b', 'c', 'd'])
        self.assertEqual(result['a'], 1)
        self.assertIsNone(result['c'])

    def test_get_ordered_stats(self):
        """测试get_ordered_stats函数"""
        test_stats = ['stat2', 'stat1', 'stat3']
        supported_stats = ['stat1', 'stat2', 'stat3', 'stat4']

        with patch.object(MonitorConst, 'OP_MONVIS_SUPPORTED', supported_stats):
            result = get_ordered_stats(test_stats)

        self.assertEqual(result, ['stat1', 'stat2', 'stat3'])

    def test_get_ordered_stats_with_non_iterable(self):
        """测试get_ordered_stats处理非可迭代对象"""
        result = get_ordered_stats(123)
        self.assertEqual(result, [])


class TestMonitorSql(unittest.TestCase):
    def test_get_table_definition_all_tables(self):
        """测试获取所有表定义"""
        result = MonitorSql.get_table_definition()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all("CREATE TABLE" in sql for sql in result))

    def test_get_table_definition_single_table(self):
        """测试获取单个表定义"""
        for table in ["monitoring_targets", "monitoring_metrics"]:
            result = MonitorSql.get_table_definition(table)
            result = normalize_spaces(result)
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", result)

    def test_get_table_definition_invalid_table(self):
        """测试获取不存在的表定义"""
        with self.assertRaises(ValueError):
            MonitorSql.get_table_definition("invalid_table")

    def test_get_metric_mapping_sql(self):
        """测试获取指标映射SQL"""
        result = MonitorSql.get_metric_mapping_sql()
        result = normalize_spaces(result)
        self.assertIn("SELECT metric_id, metric_name", result)


class TestMonitorDB(unittest.TestCase):
    def setUp(self):
        # 创建临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.monitor_db = MonitorDB(self.db_path)

        # 初始化数据库schema
        self.monitor_db.init_schema()

    def tearDown(self):
        # 关闭并删除临时数据库文件
        if hasattr(self, 'temp_db'):
            self.temp_db.close()
            os.unlink(self.db_path)

    def test_init_schema(self):
        """测试初始化数据库schema"""
        # 验证表是否创建成功
        for table in ["monitoring_targets", "monitoring_metrics"]:
            self.assertTrue(self.monitor_db.db_manager.table_exists(table))

    def test_insert_dimensions(self):
        """测试插入维度数据"""
        targets = OrderedDict()
        targets[("layer1", 0, 0)] = None
        targets[("layer2", 0, 1)] = None

        metrics = {"metric1", "metric2"}
        metric_stats = {
            "metric1": {"norm", "max"},
            "metric2": {"min", "max"}
        }

        self.monitor_db.insert_dimensions(
            targets=targets,
            metrics=metrics
        )

        # 验证目标插入
        target_results = self.monitor_db.db_manager.select_data(
            "monitoring_targets", columns=["target_id"])
        self.assertEqual(len(target_results), 2)

        # 验证指标插入
        metric_results = self.monitor_db.db_manager.select_data(
            "monitoring_metrics", columns=["metric_id"])
        self.assertEqual(len(metric_results), 2)

    def test_get_metric_mapping(self):
        """测试获取指标映射"""
        # 先插入测试数据
        self.monitor_db.db_manager.insert_data(
            "monitoring_metrics",
            [("metric1",), ("metric2",)],
            ["metric_name"]
        )

        # 获取metric_id
        metric1_id = self.monitor_db._get_metric_id("metric1")
        metric2_id = self.monitor_db._get_metric_id("metric2")

        # 插入统计关系
        self.monitor_db.init_global_stats_data({
            "metric1": {"norm"},
            "metric2": {"max", "min"},
            "max_step": 3,
            "min_step": 0,
            "max_rank": 3,
        }
        )

        # 测试获取映射
        mapping = self.monitor_db.get_metric_mapping()

        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping["metric1"][0], metric1_id)
        self.assertEqual(mapping["metric1"][1], ["norm"])
        self.assertEqual(sorted(mapping["metric2"][1]), ["max", "min"])

    def test_get_target_mapping(self):
        """测试获取目标映射"""
        # 先插入测试数据
        self.monitor_db.db_manager.insert_data(
            "monitoring_targets",
            [("target1", 0, 0), ("target2", 0, 1)],
            ["target_name", "vpp_stage", "micro_step"]
        )

        # 测试获取映射
        mapping = self.monitor_db.get_target_mapping()

        self.assertEqual(len(mapping), 2)
        self.assertIn(("target1", 0, 0), mapping)
        self.assertIn(("target2", 0, 1), mapping)

    def test_insert_rows(self):
        """测试插入行数据"""
        # 先创建测试表
        self.monitor_db.db_manager.execute_sql(
            "CREATE TABLE trend_data (id INTEGER PRIMARY KEY, name TEXT)"
        )

        # 测试插入
        inserted = self.monitor_db.insert_rows(
            [(1, "item1"), (2, "item2")]
        )

        self.assertEqual(inserted, 2)

        # 验证数据
        results = self.monitor_db.db_manager.select_data(
            "trend_data", columns=["id", "name"])
        self.assertEqual(len(results), 2)
