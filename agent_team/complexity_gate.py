# nl2sql/agent_team/complexity_gate.py
"""复杂度门控 —— Zero-LLM，仅凭表数和列数判定走简单路径还是复杂路径。

复杂路径启用 Planner（LLM 全量 schema 选表 + 执行计划）。
简单路径保持 SchemaLinker → Builder 直通。
"""


class ComplexityGate:
    """根据数据库 schema 大小判定查询复杂度。

    阈值:
      - table_threshold: 表数 > 此值 → 复杂
      - column_threshold: 表数 ≥ 3 且总列数 > 此值 → 复杂
    """

    def __init__(self, table_threshold: int = 5, column_threshold: int = 30):
        self.table_threshold = table_threshold
        self.column_threshold = column_threshold

    def check(self, db_schema: dict) -> dict:
        """判定给定的数据库 schema 是否属于复杂场景。

        参数:
            db_schema: BIRD dev_tables.json 中单个 db_id 对应的条目，
                       包含 table_names_original 和 column_names_original。

        返回:
            {"complex": bool, "table_count": int, "total_columns": int}
        """
        table_count = len(db_schema.get("table_names_original", []))
        total_columns = len([
            c for c in db_schema.get("column_names_original", [])
            if c[0] != -1  # 排除 column_names_original 中的 "*" 占位行
        ])

        complex_ = (
            table_count > self.table_threshold
            or (table_count >= 3 and total_columns > self.column_threshold)
        )

        return {
            "complex": complex_,
            "table_count": table_count,
            "total_columns": total_columns,
        }
