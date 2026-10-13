"""dump / monitor 数据导入 SQLite,便于后续 SQL 查询和趋势分析。"""

from typing import Optional

from hcu_probe.core.compare.data2db import DBImporter


def run(
    *,
    db: str,
    data: str,
    format: str = "auto",
    mapping: Optional[str] = None,
    micro_step: bool = True,
    process_num: int = 1,
) -> str:
    """把 data 目录下的 dump/monitor 数据导入 SQLite。"""
    importer = DBImporter(
        db_path=db,
        data_path=data,
        format=format,
        mapping_path=mapping,
        micro_step=micro_step,
        process_num=process_num,
    )
    importer.import_data()
    return db
