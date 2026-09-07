"""
data_sampler.py

零 LLM 数据采样模块 —— 从 SQLite 数据库中抽取真实数据，以解决纯 DDL 无法回答的模式歧义问题。

只做一件事：对每列执行 SELECT DISTINCT ... LIMIT 5，让 LLM 知道列里实际存了什么值。
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional


class DataSampler:
    """数据采样器 —— 对每列采样 Top-5 不同值，消除列值的歧义。

    用法::

        sampler = DataSampler()
        result = sampler.sample(relevant_schema, db_path)
    """

    def sample(
        self,
        relevant_schema: Dict[str, Any],
        db_path: str,
        question: str = "",  # noqa: ARG002
    ) -> Dict[str, Any]:
        """对 relevant_schema 中每个候选表的每列采样最多 5 个不同值。

        参数
        ----
        relevant_schema:
            SchemaLinker 输出的字典，期望结构::

                {
                    "candidate_tables": [
                        {"name": "table1", "relevant_columns": [{"name": "col_a"}, ...]},
                        ...
                    ]
                }

        db_path:
            SQLite 数据库文件路径。

        返回
        ----
        dict:
            {"table_samples": {table_name: {"columns": {col_name: {"sample_values": [...]}}}}}
        """
        tables_input = relevant_schema.get("candidate_tables", [])
        if not tables_input:
            return {"table_samples": {}}

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA query_only = 1;")
        except sqlite3.Error:
            return {"table_samples": {}}

        table_samples: Dict[str, Dict[str, Any]] = {}

        for table_def in tables_input:
            table_name = table_def.get("name", "")
            columns = table_def.get("relevant_columns", [])
            if not table_name or not columns:
                continue

            col_info: Dict[str, Dict[str, Any]] = {}
            for col in columns:
                col_name = col.get("name") if isinstance(col, dict) else col
                if not col_name:
                    continue
                values = self._sample_values(conn, table_name, col_name)
                col_info[col_name] = {"sample_values": values}

            if col_info:
                table_samples[table_name] = {"columns": col_info}

        conn.close()
        return {"table_samples": table_samples}

    @staticmethod
    def _sample_values(
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
    ) -> List[str]:
        """返回某列的最多 5 个不同值。"""
        try:
            cursor = conn.execute(
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" LIMIT 5'
            )
            return [
                str(row[0]) for row in cursor.fetchall()
                if row[0] is not None
            ]
        except sqlite3.Error:
            return []
