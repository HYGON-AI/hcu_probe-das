# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import os
import sqlite3
import json
import re
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import change_mode, check_path_before_create, FileChecker
from msprobe.core.common.const import FileCheckConst
from msprobe.visualization.utils import GraphConst
from msprobe.visualization.builder.msprobe_adapter import format_node_data

TEXT_PRIMARY_KEY = 'TEXT PRIMARY KEY'
TEXT_NOT_NULL = 'TEXT NOT NULL'
INTEGER_NOT_NULL = 'INTEGER NOT NULL'
TEXT = 'TEXT'
INTEGER = 'INTEGER'

node_columns = {
    'id': TEXT_PRIMARY_KEY,
    'graph_id': TEXT_NOT_NULL,
    'node_order': INTEGER_NOT_NULL,
    'node_name': TEXT_NOT_NULL,
    'node_type': TEXT_NOT_NULL,
    'up_node': TEXT,
    'sub_nodes': TEXT,
    'precision_index': INTEGER,
    'overflow_level': TEXT,
    'micro_step_id': INTEGER_NOT_NULL,
    'matched_node_link': TEXT,
    'stack_id': TEXT,
    'parallel_merge_info': TEXT,
    'matched_distributed': TEXT,
    'modified': INTEGER_NOT_NULL,
    'input_data': TEXT,
    'output_data': TEXT,
    'data_source': TEXT,
    'dump_data_dir': TEXT,
    'step': INTEGER_NOT_NULL,
    'rank': INTEGER_NOT_NULL
}

config_columns = {
    'id': TEXT_PRIMARY_KEY,
    'graph_type': TEXT_NOT_NULL,
    'task': TEXT,
    'tool_tip': TEXT,
    'micro_steps': INTEGER,
    'overflow_check': INTEGER,
    'node_colors': TEXT_NOT_NULL,
    'rank_list': TEXT_NOT_NULL,
    'step_list': TEXT_NOT_NULL
}

stack_columns = {
    'id': TEXT_PRIMARY_KEY,
    'stack_info': TEXT
}

indexes = {
    "index1": ["step", "rank", "data_source", "up_node", "node_order"],
    "index2": ["step", "rank", "data_source", "node_name"],
    "index3": ["step", "rank", "data_source", "node_order"],
    "index4": ["step", "rank", "node_order"],
    "index5": ["step", "rank", "micro_step_id", "node_order"],
    "index6": ["step", "rank", "modified", "matched_node_link"]
}

SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


def is_safe_identifier(name):
    """验证标识符是否安全（防止SQL注入）"""
    return isinstance(name, str) and SAFE_NAME_PATTERN.match(name) is not None


def create_table_sql_from_dict(table_name, columns_dict):
    """
    根据提供的表名和列定义字典生成CREATE TABLE SQL语句。
    """
    if not is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name} - potential SQL injection risk!")

    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"

    column_definitions = []
    for column_name, column_type in columns_dict.items():
        if not is_safe_identifier(column_name):
            raise ValueError(f"Invalid column name: {column_name} - potential SQL injection risk!")

        column_definitions.append(f"    {column_name} {column_type}")

    sql += ",\n".join(column_definitions)
    sql += "\n);"

    return sql


def create_insert_sql_from_dict(table_name, columns_dict, ignore_insert=False):
    """
    根据提供的表名和数据字典生成INSERT INTO SQL语句。
    """
    if not is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name} - potential SQL injection risk!")

    columns = list(columns_dict.keys())

    for column_name in columns:
        if not is_safe_identifier(column_name):
            raise ValueError(f"Invalid column name: {column_name} - potential SQL injection risk!")

    placeholders = ["?"] * len(columns)

    columns_string = ", ".join(columns)
    placeholders_string = ", ".join(placeholders)

    sql_prefix = "INSERT OR IGNORE INTO" if ignore_insert else "INSERT INTO"
    sql = f"{sql_prefix} {table_name} ({columns_string}) VALUES ({placeholders_string})"
    return sql


def to_db(db_path, create_table_sql, insert_sql, data, db_insert_size=1000):
    if not os.path.exists(db_path):
        check_path_before_create(db_path)
    else:
        FileChecker(db_path, FileCheckConst.FILE, FileCheckConst.READ_WRITE_ABLE,
                    FileCheckConst.DB_SUFFIX).common_check()
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.error(f"Unable to create database connection: {e}")
        raise RuntimeError("Unable to create database connection") from e

    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        if len(data) == 1:
            cursor.execute(insert_sql, data[0])
            conn.commit()
        else:
            for i in range(0, len(data), db_insert_size):
                batch = data[i:i + db_insert_size]
                cursor.executemany(insert_sql, batch)
                conn.commit()
    except sqlite3.Error as e:
        logger.error(f"An sqlite3 error occurred: {e}")
        raise RuntimeError("An sqlite3 error occurred") from e
    finally:
        conn.close()


def add_table_index(db_path):
    FileChecker(db_path, FileCheckConst.FILE, FileCheckConst.READ_WRITE_ABLE, FileCheckConst.DB_SUFFIX).common_check()
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.error(f"Unable to create database connection: {e}")
        raise RuntimeError("Unable to create database connection") from e

    try:
        cursor = conn.cursor()
        for index_name, columns in indexes.items():
            if not is_safe_identifier(index_name):
                raise ValueError(f"Invalid index name: {index_name} - potential SQL injection risk!")

            for column in columns:
                if not is_safe_identifier(column):
                    raise ValueError(f"Invalid column name in index: {column} - potential SQL injection risk!")

            columns_str = ', '.join(columns)
            index_sql = f'''
                CREATE INDEX IF NOT EXISTS {index_name} ON tb_nodes ({columns_str});
            '''
            cursor.execute(index_sql)
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to add table index: {e}")
        raise RuntimeError("Failed to add table index") from e
    finally:
        conn.close()


def post_process_db(db_path):
    add_table_index(db_path)
    change_mode(db_path, FileCheckConst.DATA_FILE_AUTHORITY)


def node_to_db(graph, db_name):
    create_table_sql = create_table_sql_from_dict('tb_nodes', node_columns)
    insert_sql = create_insert_sql_from_dict('tb_nodes', node_columns)
    data = []
    stack_dict = {}
    for i, node in enumerate(graph.get_sorted_nodes()):
        stack_info_text = json.dumps(node.stack_info)
        if stack_info_text not in stack_dict:
            stack_dict[stack_info_text] = get_stack_unique_id(graph, stack_dict)
        data.append((get_node_unique_id(graph, node), get_graph_unique_id(graph), i, node.id, node.op.value,
                     node.upnode.id if node.upnode else '',
                     json.dumps([node.id for node in node.subnodes]) if node.subnodes else '',
                     node.data.get(GraphConst.JSON_INDEX_KEY), node.data.get(GraphConst.OVERFLOW_LEVEL),
                     node.micro_step_id if node.micro_step_id is not None else 0, json.dumps(node.matched_node_link),
                     stack_dict.get(stack_info_text),
                     json.dumps(node.parallel_merge_info) if node.parallel_merge_info else '',
                     json.dumps(node.matched_distributed), 0,
                     json.dumps(format_node_data(node.input_data, node.id, graph.compare_mode)),
                     json.dumps(format_node_data(node.output_data, node.id, graph.compare_mode)),
                     graph.data_source, graph.data_path, graph.step, graph.rank))
    to_db(db_name, create_table_sql, insert_sql, data)
    stack_to_db(stack_dict, db_name)


def config_to_db(config, db_name):
    create_table_sql = create_table_sql_from_dict('tb_config', config_columns)
    insert_sql = create_insert_sql_from_dict('tb_config', config_columns, ignore_insert=True)
    data = [("1", "compare" if config.graph_b else "build", config.task, config.tool_tip, config.micro_steps,
             config.overflow_check, json.dumps(config.node_colors), json.dumps(config.rank_list),
             json.dumps(config.step_list))]
    to_db(db_name, create_table_sql, insert_sql, data)


def stack_to_db(stack_dict, db_name):
    create_table_sql = create_table_sql_from_dict('tb_stack', stack_columns)
    insert_sql = create_insert_sql_from_dict('tb_stack', stack_columns)
    data = []
    for stack_info_text, unique_id in stack_dict.items():
        data.append((unique_id, stack_info_text))
    to_db(db_name, create_table_sql, insert_sql, data)


def get_graph_unique_id(graph):
    return f'{graph.data_source}_{graph.step}_{graph.rank}'


def get_node_unique_id(graph, node):
    return f'{get_graph_unique_id(graph)}_{node.id}'


def get_stack_unique_id(graph, stack_dict):
    return f'{get_graph_unique_id(graph)}_{len(stack_dict)}'
