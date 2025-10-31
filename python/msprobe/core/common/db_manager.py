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
import re
import sqlite3
from typing import List, Tuple, Dict, Any
from functools import wraps

from msprobe.pytorch.common.log import logger
from msprobe.core.common.file_utils import check_path_before_create, change_mode
from msprobe.core.common.const import FileCheckConst

SAFE_SQL_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


def check_identifier_safety(name):
    """验证标识符是否安全（防止SQL注入）"""
    if not isinstance(name, str) or SAFE_SQL_PATTERN.match(name) is None:
        raise ValueError(f"Invalid SQL identifier: {name}, potential SQL injection risk!")


def _db_operation(func):
    """数据库操作装饰器，自动管理连接"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        conn, curs = None, None
        try:
            conn, curs = self._get_connection()
            result = func(self, conn, curs, *args, **kwargs)
            return result  # 显式返回正常结果
            
        except sqlite3.Error as err:
            logger.error(f"Database operation failed: {err}")
            if conn:
                conn.rollback()
            return None  # 显式返回错误情况下的None
            
        finally:
            self._release_connection(conn, curs)
    return wrapper


class DBManager:
    """
    数据库管理类，封装常用数据库操作
    """

    DEFAULT_FETCH_SIZE = 10000
    DEFAULT_INSERT_SIZE = 10000
    MAX_ROW_COUNT = 100000000

    def __init__(self, db_path: str):
        """
        初始化DBManager
        :param db_path: 数据库文件路径
        :param table_config: 表配置对象
        """
        self.db_path = db_path

    @staticmethod
    def _get_where_sql(where_list):
        if not where_list:
            return "", tuple()

        where_clauses = []
        where_values = []
        if where_list:
            for col, val in where_list.items():
                check_identifier_safety(col)
                where_clauses.append(f"{col} = ?")
                where_values.append(val)
            if where_clauses:
                where_sql = " WHERE " + " AND ".join(where_clauses)
        return where_sql, tuple(where_values)

    @_db_operation
    def insert_data(self, conn: sqlite3.Connection, curs: sqlite3.Cursor,
                    table_name: str, data: List[Tuple], key_list: List[str] = None) -> int:
        """
        批量插入数据
        :param table_name: 表名
        :param data: 要插入的数据列表
        :param batch_size: 每批插入的大小
        :return: 插入的行数
        """
        check_identifier_safety(table_name)

        if not data:
            return 0
        columns = len(data[0])
        if key_list:
            if not isinstance(key_list, list):
                raise TypeError(
                    f"key_list must be a list, got {type(key_list)}"
                )
            if columns != len(key_list):
                raise ValueError(
                    f"When inserting into table {table_name}, the length of key list ({key_list})"
                    f"does not match the data({columns}).")
            for key in key_list:
                check_identifier_safety(key)

        batch_size = self.DEFAULT_INSERT_SIZE
        placeholders = ", ".join(["?"] * columns)
        if key_list:
            keys = ", ".join(key_list)
            sql = f"INSERT OR IGNORE INTO {table_name} ({keys}) VALUES ({placeholders})"
        else:
            sql = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"

        inserted_rows = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            curs.executemany(sql, batch)
            inserted_rows += curs.rowcount

        conn.commit()
        return inserted_rows

    @_db_operation
    def select_data(self, conn: sqlite3.Connection, curs: sqlite3.Cursor,
                    table_name: str,
                    columns: List[str] = None,
                    where: dict = None) -> List[Dict]:
        """
        查询数据
        :param table_name: 表名
        :param columns: 要查询的列
        :param where: WHERE条件
        :return: 查询结果列表(字典形式)
        """
        check_identifier_safety(table_name)

        if not columns:
            raise ValueError("columns parameter cannot be empty, specify columns to select (e.g. ['id', 'name'])")
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise TypeError("columns must be a list of strings (e.g. ['id', 'name'])")
        
        for col in columns:
            check_identifier_safety(col)
        
        cols = ", ".join(columns)
        sql = f"SELECT {cols} FROM {table_name}"

        where_sql, where_parems = self._get_where_sql(where)
        curs.execute(sql + where_sql, where_parems)

        return [dict(row) for row in curs.fetchall()]

    @_db_operation
    def update_data(self, conn: sqlite3.Connection, curs: sqlite3.Cursor,
                    table_name: str, updates: Dict[str, Any],
                    where: dict = None) -> int:
        """
        更新数据
        :param table_name: 表名
        :param updates: 要更新的字段和值
        :param where: WHERE条件
        :param where_params: WHERE条件参数
        :return: 影响的行数
        """
        check_identifier_safety(table_name)
        if not updates:
            raise ValueError("columns parameter cannot be empty, specify it to update (e.g. {'name': 'xxx'}")
        if not isinstance(updates, dict):
            raise TypeError(f"updates must be a dictionary, got: {type(updates)}")
        for key in updates.keys():
            check_identifier_safety(key)

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        sql = f"UPDATE {table_name} SET {set_clause}"

        params = tuple(updates.values())

        where_sql, where_parems = self._get_where_sql(where)

        curs.execute(sql + where_sql, params + where_parems)
        conn.commit()
        return curs.rowcount

    @_db_operation
    def execute_sql(self, conn: sqlite3.Connection, curs: sqlite3.Cursor,
                    sql: str, params: Tuple = None) -> List[Dict]:
        """
        执行自定义SQL查询
        :param sql: SQL语句
        :param params: 参数
        :return: 查询结果
        """
        curs.execute(sql, params or ())
        if sql.strip().upper().startswith("SELECT"):
            return [dict(row) for row in curs.fetchall()]
        conn.commit()
        return []

    def table_exists(self, table_name: str) -> bool:
        """
        :param table_name: 表名
        :return: 查询结果
        """
        result = self.select_data(
            table_name="sqlite_master",
            columns=["name"],
            where={"type": "table", "name": table_name}
        )
        return len(result) > 0

    @_db_operation
    def execute_multi_sql(self, conn: sqlite3.Connection, curs: sqlite3.Cursor,
                          sql_commands: List[str]) -> List[List[Dict]]:
        """
        批量执行多个SQL语句
        :param sql_commands: [sql1, sql2, ...]
        :return: 每个SELECT语句的结果列表
        """
        results = []
        for sql in sql_commands:
            curs.execute(sql)
            if sql.strip().upper().startswith("SELECT"):
                results.append([dict(row) for row in curs.fetchall()])
        conn.commit()
        return results

    def _get_connection(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """获取数据库连接和游标"""
        check_path_before_create(self.db_path)
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使用Row工厂获取字典形式的结果
            curs = conn.cursor()
            return conn, curs
        except sqlite3.Error as err:
            logger.error(f"Database connection failed: {err}")
            raise

    def _release_connection(self, conn: sqlite3.Connection, curs: sqlite3.Cursor) -> None:
        """释放数据库连接"""
        try:
            if curs is not None:
                curs.close()
            if conn is not None:
                conn.close()
        except sqlite3.Error as err:
            logger.error(f"Failed to release database connection: {err}")
        change_mode(self.db_path, FileCheckConst.DATA_FILE_AUTHORITY)
