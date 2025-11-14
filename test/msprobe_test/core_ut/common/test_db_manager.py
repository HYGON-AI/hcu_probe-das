import unittest
import sqlite3
import os
import tempfile
from typing import Dict, List
from unittest.mock import patch, MagicMock

from msprobe.pytorch.common.log import logger
from msprobe.core.common.db_manager import DBManager


class TestDBManager(unittest.TestCase):
    def setUp(self):
        # 创建临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.db_manager = DBManager(self.db_path)

        # 创建测试表
        self.test_table = "test_table"
        self.create_test_table()

    def tearDown(self):
        # 关闭并删除临时数据库文件
        if hasattr(self, 'temp_db'):
            self.temp_db.close()
            os.unlink(self.db_path)

    def create_test_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.test_table} (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def test_get_connection_success(self):
        """测试成功获取数据库连接"""
        conn, curs = self.db_manager._get_connection()
        self.assertIsInstance(conn, sqlite3.Connection)
        self.assertIsInstance(curs, sqlite3.Cursor)
        self.db_manager._release_connection(conn, curs)

    @patch.object(logger, 'error')
    def test_get_connection_success_failed(self, mock_logger):
        """测试错误日志记录"""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")):
            with self.assertRaises(sqlite3.Error):
                self.db_manager._get_connection()
            mock_logger.assert_called_with(
                "Database connection failed: Test error")

    def test_insert_data_basic(self):
        """测试基本数据插入"""
        test_data = [
            (1, "item1", 100),
            (2, "item2", 200)
        ]
        columns = ["id", "name", "value"]

        inserted = self.db_manager.insert_data(
            table_name=self.test_table,
            data=test_data,
            key_list=columns
        )
        self.assertEqual(inserted, 2)

        # 验证数据是否实际插入
        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "item1")

    def test_insert_data_without_keys(self):
        """测试无列名的数据插入"""
        test_data = [
            (3, "item3", 300, 333),
            (4, "item4", 400, 333)
        ]

        inserted = self.db_manager.insert_data(
            table_name=self.test_table,
            data=test_data
        )
        self.assertEqual(inserted, 2)

    def test_insert_data_empty(self):
        """测试空数据插入"""
        inserted = self.db_manager.insert_data(
            table_name=self.test_table,
            data=[]
        )
        self.assertEqual(inserted, 0)

    def test_insert_data_mismatch_keys(self):
        """测试列名与数据不匹配的情况"""
        test_data = [(5, "item5")]
        with self.assertRaises(ValueError):
            self.db_manager.insert_data(
                table_name=self.test_table,
                data=test_data,
                key_list=["id", "name", "value"]  # 多了一个列
            )

    def test_select_data_basic(self):
        """测试基本数据查询"""
        # 先插入测试数据
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(10, "test10", 1000)],
            key_list=["id", "name", "value"]
        )

        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["name", "value"],
            where={"id": 10}
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "test10")
        self.assertEqual(results[0]["value"], 1000)

    def test_select_data_no_where(self):
        """测试无条件查询"""
        # 插入多条数据
        test_data = [
            (20, "item20", 2000),
            (21, "item21", 2100)
        ]
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=test_data,
            key_list=["id", "name", "value"]
        )

        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"]
        )
        self.assertGreaterEqual(len(results), 2)

    def test_update_data_basic(self):
        """测试基本数据更新"""
        # 先插入测试数据
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(30, "old_name", 3000)],
            key_list=["id", "name", "value"]
        )

        updated = self.db_manager.update_data(
            table_name=self.test_table,
            updates={"name": "new_name", "value": 3500},
            where={"id": 30}
        )
        self.assertEqual(updated, 1)

        # 验证更新结果
        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"],
            where={"id": 30}
        )
        self.assertEqual(results[0]["name"], "new_name")
        self.assertEqual(results[0]["value"], 3500)

    def test_execute_sql_select(self):
        """测试执行SELECT SQL语句"""
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(50, "sql_item", 5000)],
            key_list=["id", "name", "value"]
        )

        results = self.db_manager.execute_sql(
            sql=f"SELECT name, value FROM {self.test_table} WHERE id = ?",
            params=(50,)
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "sql_item")

    def test_execute_sql_non_select(self):
        """测试执行非SELECT SQL语句"""
        # 先插入数据
        self.db_manager.insert_data(
            table_name=self.test_table,
            data=[(60, "to_delete", 6000)],
            key_list=["id", "name", "value"]
        )

        # 执行DELETE语句
        self.db_manager.execute_sql(
            sql=f"DELETE FROM {self.test_table} WHERE id = 60"
        )

        # 验证数据已被删除
        results = self.db_manager.select_data(
            table_name=self.test_table,
            columns=["id", "name", "value"],
            where={"id": 60}
        )
        self.assertEqual(len(results), 0)

    def test_table_exists_true(self):
        """测试表存在检查(存在的情况)"""
        exists = self.db_manager.table_exists(self.test_table)
        self.assertTrue(exists)

    def test_table_exists_false(self):
        """测试表存在检查(不存在的情况)"""
        exists = self.db_manager.table_exists("non_existent_table")
        self.assertFalse(exists)

    def test_execute_multi_sql(self):
        """测试批量执行多个SQL语句"""
        sql_commands = [
            f"INSERT INTO {self.test_table} (id, name, value) VALUES (70, 'multi1', 7000)",
            f"INSERT INTO {self.test_table} (id, name, value) VALUES (71, 'multi2', 7100)",
            f"SELECT * FROM {self.test_table} WHERE id IN (70, 71)"
        ]

        results = self.db_manager.execute_multi_sql(sql_commands)

        # 应该只有最后一个SELECT语句有结果
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)

    @patch.object(logger, 'error')
    def test_db_operation_decorator(self, mock_logger):
        """测试数据库操作装饰器"""
        # 模拟一个会失败的操作
        with patch.object(self.db_manager, '_get_connection',
                          side_effect=sqlite3.Error("Test error")):
            result = self.db_manager.select_data(table_name=self.test_table)
            self.assertIsNone(result)  # 装饰器会捕获异常并返回None
            mock_logger.assert_called_with(
                "Database operation failed: Test error")
