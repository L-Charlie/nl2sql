# nl2sql/agent_team/planner.py
"""规划器智能体 —— 复杂查询场景下，LLM 读全量 schema 做表选 + 执行计划。

被 Orchestrator 在需要 LLM 选表时启用。小库(≤10表)全量 DDL 直喂 plan()，
大库(>10表)先做 embedding 粗筛，再由 plan_with_hints() 精筛。

固定使用 generic prompt，不做 game 领域兼容。
"""

import json
import os
import re
from typing import Optional
from openai import OpenAI


class Planner:
    """LLM 智能体：读全量 schema DDL → 选表+列 → 输出结构化执行计划。

    仅在复杂查询路径中使用。简单路径走 SchemaLinker。
    """

    def __init__(self, model_client: Optional[OpenAI] = None, model: str = "deepseek-v4-flash"):
        self.model_client = model_client or OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = model

        prompt_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(prompt_dir, "prompts", "planner_prompt_generic.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

    def plan(self, question: str, full_schema: list[dict],
             evidence: str = "") -> dict:
        """复杂路径：读全量 schema，做表选 + 列选 + JOIN 规划。

        参数:
            question: 用户自然语言问题
            full_schema: BIRD dev_tables.json 中对应 db_id 的完整 schema 列表，
                         每项含 table_name, table_description, columns
            evidence: BIRD 外部知识（可选）

        返回:
            {
                "selected_tables": ["t1", "t2"],
                "selected_columns": {"t1": ["c1", "c2"], "t2": ["c3"]},
                "anchor_table": "主表名",
                "join_plan": [...],
                "aggregation_strategy": {...},
                "validation_checks": [...],
                "confidence_risks": [...]
            }
        """
        ddl = self._render_full_schema_ddl(full_schema)
        parts = [
            f"/* Full database schema: */\n{ddl}",
            f"/* Question: {question} */",
        ]
        if evidence:
            parts.append(f"/* Additional knowledge */\n{evidence}")
        parts.append(
            "/* Step 1: Select relevant tables and columns from the full schema.\n"
            "   Step 2: Plan JOINs, aggregations, and validation checks.\n"
            "   Return ONLY valid JSON, no markdown, no SQL, no explanation. */"
        )
        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        response_text = self._call_llm(messages)
        return self._parse_plan(response_text)

    @staticmethod
    def _render_full_schema_ddl(full_schema: list[dict]) -> str:
        """将 BIRD 格式的完整 schema 列表转换为 CREATE TABLE DDL。

        与 _render_schema_ddl 不同：此方法渲染所有表和所有列，
        不做裁剪——Planner 需要看到完整 schema 才能自己选表。

        参数:
            full_schema: list of {table_name, table_description, columns: [{col, type, description}]}

        返回:
            完整的 CREATE TABLE DDL 字符串，含 FK 关系注释
        """
        ddl_parts = []
        for table in full_schema:
            table_name = table.get("table_name", table.get("name", "unknown"))
            table_desc = table.get("table_description", table.get("description", ""))
            columns = table.get("columns", [])
            col_defs = []
            pk_cols = []

            for col in columns:
                col_name = col.get("col", col.get("name", "?"))
                col_type = col.get("type", "TEXT").upper()

                if col_type in ("NUMBER", "NUMERIC", "DECIMAL"):
                    col_type = "NUMERIC"
                elif col_type in ("INT", "INTEGER", "BIGINT"):
                    col_type = "INTEGER"
                elif col_type in ("VARCHAR", "STRING", "CHAR"):
                    col_type = "TEXT"
                elif col_type in ("BOOL", "BOOLEAN"):
                    col_type = "INTEGER"

                desc = col.get("description", "")
                if "PK" in desc:
                    pk_cols.append(col_name)

                fk_match = ""
                if "FK->" in desc:
                    fk_target = desc.split("FK->")[-1].strip().split()[0]
                    fk_match = f" REFERENCES {fk_target}"

                col_defs.append(f"    {col_name} {col_type}{fk_match}")

            if pk_cols:
                col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

            header = f"CREATE TABLE {table_name} ("
            if table_desc:
                header = f"-- {table_desc}\n{header}"
            ddl_parts.append(header + "\n" + ",\n".join(col_defs) + "\n);")

        return "\n".join(ddl_parts)

    def plan_with_hints(self, question: str, full_schema: list[dict],
                        embedding_hints: dict, evidence: str = "") -> dict:
        """大规模表场景：Embedding 先粗筛，LLM 再精筛。

        embedding_hints 来自 SchemaLinker.link() 的输出，包含：
          - candidate_tables: embedding 召回的表及其分数
          - join_paths: 候选表间的 JOIN 路径
          - time_fields: 时间字段信息
          - dangerous_fields: 危险字段警告

        LLM 以 embedding 建议为参考，结合完整 DDL 做最终的表和列选择。
        """
        ddl = self._render_full_schema_ddl(full_schema)
        hints_text = self._render_embedding_hints(embedding_hints)
        parts = [
            f"/* Full database schema: */\n{ddl}",
            hints_text,
            f"/* Question: {question} */",
        ]
        if evidence:
            parts.append(f"/* Additional knowledge */\n{evidence}")
        parts.append(
            "/* Step 1: Review the embedding suggestions above. You may adopt, "
            "adjust, or override them.\n"
            "   Step 2: Select relevant tables and columns from the full schema.\n"
            "   Step 3: Plan JOINs, aggregations, and validation checks.\n"
            "   Return ONLY valid JSON, no markdown, no SQL, no explanation. */"
        )
        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        response_text = self._call_llm(messages)
        return self._parse_plan(response_text)

    @staticmethod
    def _render_embedding_hints(hints: dict) -> str:
        """将 SchemaLinker embedding 召回结果格式化为 LLM 选表参考文本。

        生成结构清晰的提示块，包含候选表、JOIN 路径、时间字段和警告，
        LLM 可以采纳、调整或推翻这些建议。
        """
        lines = ["/* ===== Embedding-Based Table Suggestions ====="]
        lines.append("   These are vector-similarity suggestions. Use them as a")
        lines.append("   starting point — verify against the full DDL above.")
        lines.append("   You may override any suggestion if the DDL tells you")
        lines.append("   otherwise. */")

        candidate_tables = hints.get("candidate_tables", [])
        if candidate_tables:
            lines.append("\n-- Suggested candidate tables (by embedding score):")
            for t in candidate_tables:
                name = t.get("name", "?")
                score = t.get("score", 0)
                desc = t.get("description", "")[:100]
                layer = t.get("layer", "")
                suffix = t.get("suffix_type", "")
                meta = f" layer={layer}" if layer else ""
                meta += f" suffix={suffix}" if suffix else ""
                lines.append(f"--   [{score:.3f}] {name}{meta}  -- {desc}")

        join_paths = hints.get("join_paths", [])
        if join_paths:
            lines.append("\n-- Suggested JOIN paths (shared column names):")
            for jp in join_paths:
                conf = jp.get("confidence", "?")
                on_cols = ", ".join(jp.get("on", []))
                lines.append(
                    f"--   {jp.get('from', '?')} ↔ {jp.get('to', '?')}"
                    f"  ON ({on_cols})  confidence={conf}"
                )

        time_fields = hints.get("time_fields", {})
        primary = time_fields.get("primary", {})
        if primary.get("field"):
            lines.append(
                f"\n-- Suggested primary time field: "
                f"{primary.get('table', '?')}.{primary['field']}"
                f" (type: {primary.get('type', '?')})"
            )

        dangerous = hints.get("dangerous_fields", [])
        if dangerous:
            lines.append("\n-- WARNING: Dangerous/reserved fields (verify context):")
            for df in dangerous:
                lines.append(
                    f"--   {df.get('table', '?')}.{df.get('field', '?')}"
                    f" -- {df.get('warning', '')}"
                )

        lines.append("\n/* ===== End Embedding Suggestions ===== */")
        return "\n".join(lines)

    def _call_llm(self, messages, max_retries=5):
        """调用 LLM 生成响应，带重试机制。

        使用 temperature=0.3 以在确定性和创造性之间取得平衡。
        最多重试 max_retries 次。

        参数:
            messages: OpenAI 格式的消息列表
            max_retries: 最大重试次数

        返回:
            LLM 响应的文本内容，或错误占位 JSON
        """
        for attempt in range(max_retries):
            try:
                response = self.model_client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3,
                    n=1,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
        return '{"error": "empty response"}'

    def _parse_plan(self, text: str) -> dict:
        """从 LLM 的响应文本中解析出 JSON 格式的执行计划。

        LLM 可能以各种格式返回 JSON：
        1. ```json { ... } ```（标准 markdown 代码块）
        2. 纯 JSON 文本（没有包裹）
        3. JSON 前后有额外文字

        这个方法使用多种策略尝试提取 JSON：
        策略 A：查找 ```json ... ``` 代码块
        策略 B：直接尝试解析全文为 JSON
        策略 C：找到第一个 { 和最后一个 } 截取 JSON

        参数:
            text: LLM 返回的原始响应文本

        返回:
            解析后的执行计划字典。如果全部解析失败，
            返回包含 error 和 raw_response 字段的错误字典。
        """
        # 策略 A：先尝试匹配 ```json { ... } ``` 格式的代码块
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # 策略 B：直接尝试解析文本为 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略 C：找到文本中第一个 { 和最后一个 } 之间的内容
        # 这样即使 JSON 前后有额外文字也能提取出来
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        # 所有策略都失败了，返回错误信息
        return {"error": "plan_parse_failed", "raw_response": text}
