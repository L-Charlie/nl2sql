# nl2sql/agent_team/judge_scoring_card.py
"""NL2SQL Judge 评分卡 —— prompt 构建。

使用 LLM-as-a-Verifier 模式：逐项核查 → 证据锚定 → 汇总打分。
不再直接让 LLM 给抽象分数，而是生成具体 YES/NO 核查清单，每项带证据。
"""

import json
import os

from agent_team.state_prompts import (
    EXECUTION_STATE_PROMPT,
    GATE_CODE_PROMPT,
    INTENT_STATE_PROMPT,
)


class JudgeScoringCard:
    """构建 LLM-as-a-Verifier 的核查 prompt。"""

    def __init__(self):
        self._system_prompt = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            prompt_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(prompt_dir, "prompts", "judge_prompt_generic.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self._system_prompt = f.read()
            else:
                self._system_prompt = self._default_system_prompt()
            self._system_prompt += (
                EXECUTION_STATE_PROMPT + GATE_CODE_PROMPT + INTENT_STATE_PROMPT
            )
        return self._system_prompt

    def build_prompt(
        self,
        question: str,
        sql: str,
        exec_result: dict,
        relevant_schema: dict = None,
        full_schema: list[dict] = None,
        evidence: str = "",
        plan: dict = None,
        sampling_data: dict = None,
    ) -> str:
        """构造发送给 Judge LLM 的核查 prompt。

        参数:
            question: 用户自然语言问题
            sql: 待核查的 SQL
            exec_result: 执行结果 {"ok", "error", "row_count", "sample_rows"}
            relevant_schema: 关联的表结构（SchemaLinker 选中的）
            full_schema: 全量数据库 schema（用于检测漏表）
            evidence: 外部知识（可选）
            plan: Planner 执行计划（可选）
            sampling_data: 数据采样结果，用于验证 SQL 中的字符串值是否正确

        返回:
            格式化的核查 prompt 字符串
        """
        parts = []

        # 全量 Schema DDL —— Judge 用此检测是否遗漏了必要的表
        if full_schema:
            full_ddl = self._render_full_schema_ddl(full_schema)
            parts.append(f"## 全量数据库 Schema（所有表）\n```sql\n{full_ddl}\n```")

        # 已选 Schema DDL —— SchemaLinker 选中的表
        if relevant_schema:
            ddl = self._render_schema_ddl(relevant_schema)
            parts.append(f"## 已选中的表（SchemaLinker 输出）\n```sql\n{ddl}\n```")

        # 问题
        parts.append(f"## 用户问题\n{question}")

        # 外部知识
        if evidence:
            parts.append(f"## 外部知识\n{evidence}")

        # Planner 的结构化计划是 SQL 意图回译的重要证据。
        if plan:
            plan_str = json.dumps(plan, ensure_ascii=False, indent=2)
            parts.append(f"## 执行计划\n```json\n{plan_str}\n```")

        # 采样数据：Judge 用此验证 SQL 中的字符串值是否与实际数据一致
        if sampling_data:
            sample_text = self._format_sampling_data(sampling_data)
            if sample_text:
                parts.append(sample_text)

        # SQL
        parts.append(f"## 待验证 SQL\n```sql\n{sql}\n```")

        # 执行结果
        parts.append("## 执行结果")
        # 保留实际状态，不能在展示时丢失 probe 截断信息或为旧调用补造默认值。
        state_fields = (
            "stage", "attempted", "ok", "mode", "gate", "columns", "truncated",
            "row_count", "row_count_exact", "execution_ms", "result_profile",
            "error_code", "read_only_enforced",
        )
        execution_state = {key: exec_result[key] for key in state_fields if key in exec_result}
        parts.append(
            "### 执行状态字段\n```json\n"
            + json.dumps(execution_state, ensure_ascii=False, indent=2)
            + "\n```"
        )
        if exec_result.get("ok"):
            parts.append(f"- 状态: 执行成功")
            count_label = "精确行数" if exec_result.get("row_count_exact") is True else "报告行数（总行数未确认）"
            parts.append(f"- {count_label}: {exec_result.get('row_count', 'N/A')}")
            sample_rows = exec_result.get("sample_rows", [])
            if sample_rows:
                sample_str = "\n".join(
                    str(r)[:200] for r in sample_rows[:5]
                )
                parts.append(f"- 样本数据（前5行）:\n```\n{sample_str}\n```")
        else:
            if exec_result.get("attempted") is False or exec_result.get("stage") == "gate":
                parts.append("- 状态: 未执行")
            elif exec_result.get("ok") is False:
                parts.append("- 状态: 执行器报告失败")
            else:
                parts.append("- 状态: 未知（未提供成功或失败标记）")
            parts.append(f"- 错误信息: {exec_result.get('error', 'Unknown')[:500]}")

        return "\n\n".join(parts)

    @staticmethod
    def _format_sampling_data(sampling_data: dict) -> str:
        """格式化采样数据为 Judge 可用的文本（复用 Builder 的格式化逻辑）。"""
        from agent_team.builder import SQLBuilder
        return SQLBuilder._format_sampling_data(sampling_data)

    @staticmethod
    def _render_schema_ddl(relevant_schema: dict) -> str:
        """将 relevant_schema 转为精简 DDL，供 Judge 引用表名和列名。"""
        ddl_parts = []
        for table in relevant_schema.get("candidate_tables", []):
            table_name = table.get("name", "unknown")
            columns = table.get("relevant_columns", [])
            col_strs = []
            for col in columns:
                col_name = col.get("name", "?")
                col_type = col.get("type", "TEXT").upper()
                col_strs.append(f"    {col_name} {col_type}")
            ddl_parts.append(
                f"CREATE TABLE {table_name} (\n" + ",\n".join(col_strs) + "\n);"
            )
        return "\n".join(ddl_parts)

    @staticmethod
    def _render_full_schema_ddl(full_schema: list[dict]) -> str:
        """将全量 schema 转为 CREATE TABLE DDL，供 Judge 检测漏表。"""
        ddl_parts = []
        for table in full_schema:
            table_name = table.get("table_name", table.get("name", "unknown"))
            columns = table.get("columns", [])
            col_strs = []
            pk_cols = []
            for col in columns:
                col_name = col.get("col", col.get("name", "?"))
                col_type = col.get("type", "TEXT").upper()
                desc = col.get("description", "")
                if "PK" in desc:
                    pk_cols.append(col_name)
                fk = ""
                if "FK->" in desc:
                    fk_target = desc.split("FK->")[-1].strip().split()[0]
                    fk = f" REFERENCES {fk_target}"
                col_strs.append(f"    {col_name} {col_type}{fk}")
            if pk_cols:
                col_strs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
            ddl_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_strs) + "\n);")
        return "\n".join(ddl_parts)

    @staticmethod
    def _default_system_prompt() -> str:
        return """你是 SQL 质量验证专家。使用"逐项核查"方法验证 NL2SQL 生成的 SQL：

1. 针对四个维度（语法/语义/逻辑/结果质量）逐项生成 YES/NO 核查问题
2. 每项给出具体证据（SQL 片段、Schema 列名、执行结果数据）
3. 从核查结果汇总打分

以问题意图为最终标准。评估客观、具体、基于证据。"""
