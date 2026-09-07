# nl2sql/agent_team/builder.py
"""SQL 构建器智能体 —— 将执行计划 + 数据库 Schema 转换为可执行的 SQL 语句。

这个模块是整个 NL2SQL 系统的"施工队"：
1. 接收 SchemaLinker 筛选出的相关表结构（DDL）
2. 接收可选的执行计划（当前为占位计划 _DIRECT_PLAN）
3. 接收可选的采样数据（DataSampler 从数据库中抽取的样本值）
4. 调用 LLM 生成最终的 SQL 语句
5. 支持基于修复提示的重新构建

核心流程：
  用户 Prompt（含 DDL + 问题 + 计划 + 采样数据）
  → LLM 调用
  → 原始响应文本
  → SQL 提取（支持 ```sql 包裹和纯文本两种格式）
  → 反片段检测（确保输出是完整 SQL 而非片段）
  → 返回完整的 SQL 字符串
"""

import json
import os
import re
from typing import Optional
from openai import OpenAI

from agent_team.contracts import SQLArtifact


class SQLBuilder:
    """LLM 智能体，将执行计划 + Schema 转换为可执行的 SQL。

    本类的作用是"翻译官"：把结构化的执行意图（执行计划）和数据库结构（Schema）
    翻译成 LLM 能理解的 Prompt，然后从 LLM 的回复中提取出正确的 SQL 语句。

    Domain（领域）参数决定加载哪套 prompt 模板文件：
      - "generic": prompts/builder_prompt_generic.txt
                  标准 SQLite 兼容的 SQL 生成
      - "enterprise": prompts/builder_prompt_enterprise.txt
                     支持 CTE、窗口函数、递归查询等企业级 SQL 特性
    """

    # 领域名称 → prompt 模板文件名 的映射表
    # prompt 文件存放在与 builder.py 同级的 prompts/ 目录下
    DOMAIN_PROMPT_FILES = {
        "generic": "builder_prompt_generic.txt",
        "enterprise": "builder_prompt_enterprise.txt",
    }

    def __init__(self, model_client: Optional[OpenAI] = None, model: str = "deepseek-v4-flash",
                 domain: str = "generic", dialect: str = "sqlite"):
        """初始化 SQL 构建器。

        参数:
            model_client: OpenAI 客户端实例。如果为 None，则从环境变量自动创建。
            model: 使用的 LLM 模型名称（如 "deepseek-v4-flash", "gpt-3.5-turbo" 等）
            domain: 领域名称，决定了使用哪套 prompt 模板
        """
        self.model_client = model_client or OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = model
        self.domain = domain
        self.dialect = dialect

        # 加载领域对应的 system prompt 文件
        # prompt 文件中包含了该领域的 SQL 生成规则、格式要求和领域特殊说明
        prompt_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = self.DOMAIN_PROMPT_FILES.get(domain, self.DOMAIN_PROMPT_FILES["generic"])
        prompt_path = os.path.join(prompt_dir, "prompts", prompt_file)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

    def build(self, plan=None, relevant_schema=None, knowledge="",
              question="", sampling_data=None) -> str:
        """根据执行计划（或仅 Schema + 问题）生成 SQL 语句。

        这是主要的 SQL 生成方法。当 plan 为 None 时，LLM 需要直接从
        Schema DDL 和用户问题推导出完整的 SQL（因为 Planner 环节已被移除）。

        参数:
            plan: 执行计划字典（当前使用 _DIRECT_PLAN 占位符）
            relevant_schema: SchemaLinker 输出的关联表结构
            knowledge: 额外的领域知识或业务规则
            question: 用户的自然语言问题
            sampling_data: DataSampler 的采样结果（可选），包含示例值和数据分布信息

        返回:
            生成的 SQL 字符串（已从 LLM 响应中提取和清理）
        """
        return self.build_artifact(
            plan, relevant_schema, knowledge, question, sampling_data
        ).sql

    def build_artifact(self, plan=None, relevant_schema=None, knowledge="",
                       question="", sampling_data=None) -> SQLArtifact:
        """Generate a SQLArtifact, falling back to SQL-only model responses."""
        user_content = self._build_user_prompt(plan, relevant_schema, knowledge, question, sampling_data)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        response_text = self._call_llm(messages)
        return self.parse_artifact_response(response_text)

    def _build_user_prompt(self, plan, relevant_schema, knowledge="", question="", sampling_data=None):
        """构造发送给 LLM 的 User Prompt。

        将数据库 Schema、执行计划、采样数据、领域知识等组装成
        结构化的文本提示，让 LLM 能够理解上下文并生成正确的 SQL。

        组装顺序：
        1. DDL（数据库表结构定义）
        2. 数据采样说明（如果有）
        3. 执行计划（如果有，以 JSON 格式给出）
        4. 领域知识/业务规则
        5. 用户问题和输出指令

        参数:
            plan: 执行计划字典
            relevant_schema: SchemaLinker 的输出
            knowledge: 领域知识文本
            question: 用户问题
            sampling_data: DataSampler 的输出

        返回:
            组装好的 User Prompt 字符串
        """
        ddl = self._render_schema_ddl(relevant_schema)
        parts = [f"/* Given the following database schema: */\n{ddl}"]

        # 执行计划（如果有）
        if plan:
            plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
            parts.append(f"/* Execution plan hints: */\n```json\n{plan_json}\n```")

        # Evidence/知识（可能会有误导）
        if knowledge:
            parts.append(f"/* Reference hints (may be inaccurate) */\n{knowledge}")

        # 采样数据放在最后 —— 这是数据库的实际值，优先级最高
        if sampling_data:
            sample_text = self._format_sampling_data(sampling_data)
            if sample_text:
                parts.append(sample_text)

        if question:
            parts.append(
                f"-- Question: {question}\n"
                f"/* CRITICAL: WHERE clauses MUST use exact sampled values. "
                f"Write one {self.dialect} read-only query and return the required JSON object. */"
            )
        else:
            parts.append(
                f"/* Write one {self.dialect} read-only query and return the required JSON object. */"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_sampling_data(sampling_data: dict) -> str:
        """将 DataSampler 的采样结果格式化为 LLM 能理解的数据提示。

        输出实际采样值——让 LLM 知道列里存的是什么数据（如 'EUR'/'CZK'/'USD'），
        从而在 WHERE 条件中写出正确的值。
        """
        table_samples = sampling_data.get("table_samples", {})
        if not table_samples:
            return ""

        lines = ["/* 从真实数据库采集的样本数据。WHERE 条件中的字符串值必须使用此处的实际值。 */"]
        for table_name, info in table_samples.items():
            cols = info.get("columns", {})
            if not cols:
                continue
            lines.append(f"\n[{table_name}]")
            for col_name, col_info in cols.items():
                sample_vals = col_info.get("sample_values", [])
                if sample_vals:
                    lines.append(f"  {col_name}: {', '.join(sample_vals)}")

        return "\n".join(lines)

    @staticmethod
    def _render_schema_ddl(relevant_schema: dict) -> str:
        """将 relevant_schema 字典转换为 CREATE TABLE DDL 语句。

        这是 SchemaLinker 输出的"结构化数据"到"LLM 能理解的文本"的转换器。
        将每个候选表的列名、类型、主键、外键等转换为标准的 SQL DDL 格式，
        让 LLM 能够准确理解数据库结构。

        类型映射规则：
        - NUMBER/NUMERIC/DECIMAL → NUMERIC
        - INT/INTEGER/BIGINT → INTEGER
        - VARCHAR/STRING/CHAR → TEXT
        - BOOL/BOOLEAN → INTEGER（SQLite 没有布尔类型）

        参数:
            relevant_schema: SchemaLinker 的输出字典，包含 candidate_tables 列表

        返回:
            格式化的 DDL 字符串，例如：
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            );
        """
        ddl_parts = []
        fk_pairs: list[tuple[str, str, str, str]] = []  # 暂存外键信息：(表, 列, 引用表, 引用列)

        for table in relevant_schema.get("candidate_tables", []):
            table_name = table.get("name", "unknown")
            columns = table.get("relevant_columns", [])
            col_defs = []
            pk_cols = []

            for col in columns:
                col_name = col.get("name", "?")
                col_type = col.get("type", "TEXT").upper()

                # ---- 类型映射：将各种数据库类型映射为 SQLite 兼容类型 ----
                if col_type in ("NUMBER", "NUMERIC", "DECIMAL"):
                    col_type = "NUMERIC"
                elif col_type in ("INT", "INTEGER", "BIGINT"):
                    col_type = "INTEGER"
                elif col_type in ("VARCHAR", "STRING", "CHAR"):
                    col_type = "TEXT"
                elif col_type in ("BOOL", "BOOLEAN"):
                    col_type = "INTEGER"
                # 其他类型（DATE, TIMESTAMP, FLOAT, DOUBLE 等）保持原样

                desc = col.get("description", "")

                # 检查描述中是否包含 PK（主键）标记
                if "PK" in desc:
                    pk_cols.append(col_name)

                # 检查描述中是否包含 FK（外键）标记
                # FK 标记格式：FK->引用表名.引用列名
                fk_match = ""
                if "FK->" in desc:
                    fk_target = desc.split("FK->")[-1].strip().split()[0]
                    fk_match = f" REFERENCES {fk_target}"

                col_defs.append(f"    {col_name} {col_type}{fk_match}")

            # 如果有主键列，追加 PRIMARY KEY 约束
            if pk_cols:
                col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

            ddl_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);")

        return "\n".join(ddl_parts)

    def _call_llm(self, messages, max_retries=5):
        """调用 LLM 生成响应，带重试和反片段检测机制。

        这个方法是 Builder 与 LLM 交互的核心。它有以下特性：
        1. **自动重试**：最多重试 max_retries 次
        2. **动态温度调整**：空响应时降低 temperature 到 0.0
        3. **反片段检测**：如果 LLM 输出不是完整 SQL（缺少 SELECT/FROM），
           追加强制 Prompt 要求 LLM 输出完整 SQL

        为什么需要反片段检测？
        LLM 有时会输出 SQL 片段（如只说 WHERE 子句部分），而不是完整的 SQL。
        片段对执行和验证都没有意义，所以需要检测并强制重试。

        参数:
            messages: OpenAI 格式的消息列表 [{"role": ..., "content": ...}]
            max_retries: 最大重试次数

        返回:
            LLM 响应的文本内容
        """
        temp = 0.1  # 初始温度：0.1，在确定性和创造性之间取得平衡
        for attempt in range(max_retries):
            try:
                response = self.model_client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temp,
                    n=1,
                )
                content = response.choices[0].message.content
                if not content:
                    # 空响应：降低温度再试
                    temp = 0.0
                    continue
                # 反片段检测：提取 SQL 后检查是否为完整的 SQL 语句
                extracted = self.parse_artifact_response(content).sql
                if extracted and not self._looks_like_sql(extracted) and attempt < max_retries - 1:
                    # 输出不是完整 SQL：追加纠正指令后重试
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": ("That is NOT a complete SQL query. It is missing SELECT and FROM. "
                                    "Output a FULL SQL query starting with SELECT or WITH that includes "
                                    "FROM, JOINs, WHERE, GROUP BY as needed, inside the required "
                                    "SQLArtifact JSON contract.")
                    })
                    temp = 0.0  # 降低温度提高确定性
                    continue
                # 截断检测：SQL 以 ON/WHERE/AND 等不完整关键词结尾
                if extracted and self._looks_truncated(extracted) and attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "The SQL appears to have been cut off. Return the complete SQLArtifact "
                            "JSON contract with a non-truncated SQL query."
                        )
                    })
                    temp = 0.0
                    continue
                return content
            except Exception as e:
                if attempt == max_retries - 1:
                    # 最后一次重试也失败了，抛出异常给上层处理
                    raise
        return "SELECT 'error: empty response' AS _error"

    def _extract_sql(self, text: str) -> str:
        """从 LLM 响应文本中提取纯 SQL。

        LLM 的回复可能有多种格式：
        1. ```sql ... ``` 代码块包裹（最标准）
        2. ``` ... ``` 无语言标签的代码块
        3. 纯文本（没有任何包裹）
        4. 被引号包裹的文本

        这个方法处理所有这些情况，返回纯净的 SQL 字符串。

        参数:
            text: LLM 返回的原始文本

        返回:
            提取出的纯 SQL 字符串
        """
        artifact = self._parse_json_artifact(text)
        if artifact is not None:
            return artifact.sql

        # 1. 尝试匹配 ```sql ... ``` 或 ``` ... ``` 代码块
        sql_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        # 2. 分步输出: 找以 SELECT/WITH 开头的行，收集连续 SQL 行
        lines = text.strip().split('\n')
        sql_lines = []
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.upper().startswith(('SELECT ', 'WITH ', 'INSERT ', 'UPDATE ', 'DELETE ', 'CREATE ')):
                sql_lines.insert(0, stripped)
            elif sql_lines and stripped:
                sql_lines.insert(0, stripped)
            elif sql_lines and not stripped:
                break
        if sql_lines:
            return '\n'.join(sql_lines).strip()

        # 3. 纯文本查找 SELECT/WITH 开头
        for keyword in ('SELECT', 'WITH', 'select', 'with'):
            idx = text.find(keyword)
            if idx >= 0:
                return text[idx:].strip()

        # 4. 去掉外层引号
        t = text.strip()
        if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
            t = t[1:-1].strip()
        return t

    @classmethod
    def parse_artifact_response(
        cls,
        text: str,
        previous_artifact: SQLArtifact | None = None,
    ) -> SQLArtifact:
        """Parse the structured contract, with a non-blocking SQL-only fallback."""
        artifact = cls._parse_json_artifact(text)
        if artifact is not None:
            artifact.raw_response = text
            if previous_artifact and artifact.intent_version <= previous_artifact.intent_version:
                changed = artifact.draft_intent != previous_artifact.draft_intent
                artifact.intent_version = previous_artifact.intent_version + (1 if changed else 0)
                if changed and not artifact.intent_revision:
                    artifact.intent_revision = {"reason": "Refiner revised the draft intent"}
            return artifact

        extractor = object.__new__(cls)
        sql = cls._extract_sql(extractor, text)
        warnings = [{
            "code": "generation.contract_missing",
            "message": "Model returned SQL without a valid DraftIntent contract",
        }]
        return SQLArtifact(
            sql=sql,
            draft_intent=previous_artifact.draft_intent if previous_artifact else None,
            intent_version=previous_artifact.intent_version if previous_artifact else 1,
            generation_warnings=warnings,
            raw_response=text,
        )

    @staticmethod
    def _parse_json_artifact(text: str) -> SQLArtifact | None:
        candidates = [text.strip()]
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            artifact = SQLArtifact.from_value(value)
            if artifact is not None and artifact.sql:
                return artifact
        return None

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        """判断文本看起来是否像一条完整的 SQL 语句。

        用于反片段检测：排除 JSON、说明文字、SQL 片段等非完整 SQL。

        判断标准：
        - 不能以 { 开头（JSON 对象）
        - 不能以 ```json 开头
        - 必须以 SQL 关键词开头（SELECT/WITH/INSERT/UPDATE/DELETE/CREATE/ALTER/DROP）

        参数:
            text: 要检查的文本

        返回:
            True 如果文本看起来像完整的 SQL 语句
        """
        t = text.strip().lower()
        if not t:
            return False
        if t.startswith("{") or t.startswith("```json"):
            # JSON 格式不是 SQL
            return False
        # SQL 语句应该以这些关键词之一开头
        sql_keywords = ["select ", "with ", "insert ", "update ", "delete ", "create ", "alter ", "drop "]
        return any(t.startswith(kw) for kw in sql_keywords)

    @staticmethod
    def _looks_truncated(sql: str) -> bool:
        """Check if SQL was truncated by API (incomplete response from deepseek etc).

        Why: APIs may return incomplete SQL due to max_tokens or network issues,
        causing syntax errors that repair loops cannot fix.
        """
        stripped = sql.strip().rstrip(";").strip()
        if not stripped:
            return True
        truncated_endings = [
            " on", " where", " and", " or", " join", " set",
            " from", " group", " order", " having", " limit",
            " inner", " left", " right", " full", " cross",
            " when", " then", " else", " union", " except",
            " as", " by", " in", " like", " between",
            "(", ".", ",",  # 函数调用截断 (如 "SUM(") / 列名截断 (如 "budget.")
        ]
        lower = stripped.lower()
        for ending in truncated_endings:
            if lower.endswith(ending):
                return True
        return False
