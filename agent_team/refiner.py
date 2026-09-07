# nl2sql/agent_team/refiner.py
"""修复器智能体 —— 根据验证失败信息或执行错误来修复 SQL 语句。

Refiner 是整个 NL2SQL 系统的"修理工"：
当 Builder 生成的 SQL 有问题时（无论是被静态验证引擎发现，
还是实际执行时出错），Refiner 负责分析问题并给出修复方案。

修复策略分为三个层级（L1/L2/L3），从"表面修补"到"推倒重来"：

┌─────────────────────────────────────────────────────┐
│ 修复层级树                                            │
│                                                      │
│ L1（表面修复）— 直接修改 SQL 文本                      │
│   ├─ 修正列名/表名拼写错误                             │
│   ├─ 调整 JOIN 条件                                   │
│   ├─ 修改 WHERE 条件/聚合逻辑                          │
│   └─ 无需重建，无需重新关联 Schema                      │
│                                                      │
│ L2（重新关联）— SchemaLinker 遗漏了必要表或列           │
│   └─ 带着诊断信息重新做 Schema Linking                 │
│                                                      │
│ L3（重新规划）— 整体策略有问题，需要改变查询计划        │
│   └─ 当前降级为带扩展提示的重建                        │
│                                                      │
│ TERMINATE — 判定为无法修复，终止循环                   │
│ PASS — 没有需要修复的内容                              │
└─────────────────────────────────────────────────────┘

本模块提供两种修复模式：
1. repair_from_exec_error() — 基于实际 SQL 执行错误的修复
2. repair_from_judge_feedback() — 基于 Judge 评分细目的修复
"""

import json
import os
import re
from typing import Callable, Optional
from openai import OpenAI

from agent_team.contracts import SQLArtifact
from agent_team.state_prompts import (
    EXECUTION_STATE_PROMPT,
    GATE_CODE_PROMPT,
    INTENT_STATE_PROMPT,
    REPAIR_STATE_PROMPT,
)


class Refiner:
    """LLM 智能体，根据验证失败信息执行修复策略。

    Domain（领域）参数决定加载哪套 prompt 模板：
      - "generic": prompts/refiner_prompt_generic.txt  （标准 SQLite）

    工作流程：
    1. 收到失败的 SQL 和失败原因（验证报告或执行错误）
    2. 使用 RefinerDecisionEngine 判断修复级别
    3. 根据修复级别执行对应策略：
       - L1：直接修改 SQL 文本
       - L2：建议重新关联 Schema
       - L3：建议重新规划（降级为扩展提示重建）
       - TERMINATE：放弃修复
       - PASS：无需修复
    """

    # 领域名称 → prompt 模板文件名 的映射表
    DOMAIN_PROMPT_FILES = {
        "generic": "refiner_prompt_generic.txt",
    }

    def __init__(
        self,
        model_client=None,
        model="deepseek-v4-flash",
        domain: str = "generic",
        schema_retriever: Optional[Callable] = None,
    ):
        """初始化修复器。

        参数:
            model_client: OpenAI 客户端实例，为 None 时从环境变量创建
            model: LLM 模型名称
            domain: 领域名称（generic）
            schema_retriever: Judge 请求补表时调用的受限 Schema 检索回调
        """
        self.model_client = model_client or OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = model
        self.domain = domain
        self.schema_retriever = schema_retriever

        # 加载领域对应的 system prompt 文件
        prompt_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = self.DOMAIN_PROMPT_FILES.get(domain, self.DOMAIN_PROMPT_FILES["generic"])
        prompt_path = os.path.join(prompt_dir, "prompts", prompt_file)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

    def repair(
        self,
        artifact: SQLArtifact | str,
        failure_stage: str,
        feedback: dict,
        relevant_schema: dict,
        question: str = "",
        sampling_data: dict = None,
        knowledge: str = "",
        allow_schema_retrieval: bool = True,
        max_new_tables: int = 2,
    ) -> dict:
        """Route repair with facts appropriate to the failure stage."""
        current = artifact if isinstance(artifact, SQLArtifact) else SQLArtifact.from_sql(artifact)
        if failure_stage == "gate":
            return self._repair_from_gate_feedback(
                current, feedback, relevant_schema, question, sampling_data, knowledge
            )
        if failure_stage == "database":
            error = feedback.get("error", "")
            if not error:
                issues = feedback.get("structured_feedback", {}).get("issues", [])
                error = issues[0].get("message", "Database execution failed") if issues else "Database execution failed"
            return self.repair_from_exec_error(
                current.sql, error, relevant_schema, question,
                sampling_data=sampling_data, knowledge=knowledge, artifact=current,
            )
        if failure_stage == "semantic":
            return self.repair_from_judge_feedback(
                current.sql, feedback, relevant_schema, question,
                sampling_data=sampling_data, knowledge=knowledge,
                allow_schema_retrieval=allow_schema_retrieval,
                max_new_tables=max_new_tables, artifact=current,
            )
        return {
            "action": "give_up",
            "diagnosis": f"Unsupported repair stage: {failure_stage}",
            "repair_route": "unsupported",
        }

    def _repair_from_gate_feedback(
        self,
        artifact: SQLArtifact,
        judge_result: dict,
        relevant_schema: dict,
        question: str,
        sampling_data: dict,
        knowledge: str,
    ) -> dict:
        """Repair pre-execution failures without claiming the SQL was executed."""
        from agent_team.builder import SQLBuilder

        ddl = SQLBuilder._render_schema_ddl(relevant_schema)
        feedback = judge_result.get("structured_feedback", judge_result)
        parts = [
            "The SQL was NOT executed because a deterministic pre-execution Gate blocked it.",
            f"## Database Schema\n{ddl}",
            f"## Question\n{question}",
            f"## Blocked SQL\n```sql\n{artifact.sql}\n```",
            "## Gate Feedback\n" + json.dumps(feedback, ensure_ascii=False, indent=2),
        ]
        if sampling_data:
            parts.append(SQLBuilder._format_sampling_data(sampling_data))
        if knowledge:
            parts.append(f"## External Knowledge\n{knowledge}")
        parts.append(
            "Repair only the deterministic safety, syntax, placeholder, or authorized-schema issue. "
            "Return the SQLArtifact JSON contract with draft_intent and one read-only SQL query."
        )
        messages = [
            {"role": "system", "content": (
                "You repair SQL blocked before database execution. Never use tables or columns "
                "outside the provided schema. Return JSON containing draft_intent and sql."
                + EXECUTION_STATE_PROMPT + GATE_CODE_PROMPT + REPAIR_STATE_PROMPT
            )},
            {"role": "user", "content": "\n\n".join(part for part in parts if part)},
        ]
        response = self._call_llm(messages)
        fixed_artifact = SQLBuilder.parse_artifact_response(response, artifact)
        base = {
            "repair_route": "gate_sql_repair",
            "schema_retrieval": {"requested": False, "triggered": False, "added_tables": []},
            "relevant_schema": relevant_schema,
        }
        if fixed_artifact.sql and self._looks_like_sql(fixed_artifact.sql) and fixed_artifact.sql != artifact.sql:
            return {
                **base,
                "action": "fixed",
                "fixed_sql": fixed_artifact.sql,
                "fixed_artifact": fixed_artifact.to_dict(),
                "diagnosis": "Repaired deterministic Gate failure",
            }
        return {
            **base,
            "action": "give_up",
            "diagnosis": "LLM could not produce a different SQL for the Gate failure",
        }

    def repair_from_exec_error(
        self,
        sql: str,
        exec_error: str,
        relevant_schema: dict,
        question: str = "",
        sampling_data: dict = None,
        knowledge: str = "",
        artifact: SQLArtifact = None,
    ) -> dict:
        """根据 SQLite 的实际执行错误信息来修复 SQL。

        这是"新模式"的修复入口，与旧版 repair() 的区别：
        - repair() 接收的是静态验证报告（规则检查）
        - repair_from_exec_error() 接收的是真实的 SQL 执行错误（运行时错误）

        参数:
            sql: 执行失败的 SQL 语句
            exec_error: SQLite 返回的错误信息
            relevant_schema: 数据库表结构
            question: 用户原始问题（用于上下文）
            sampling_data: 数据采样结果（含列实际值）
            knowledge: 外部知识/evidence

        返回:
            {"action": "fixed", "fixed_sql": str, "diagnosis": str}
            或 {"action": "give_up", "diagnosis": str}
        """
        from agent_team.builder import SQLBuilder
        ddl = SQLBuilder._render_schema_ddl(relevant_schema)

        # 构造修复 Prompt
        parts = [
            f"The following SQL query failed to execute against a SQLite database.\n",
            f"## Database Schema\n{ddl}\n",
        ]

        if sampling_data:
            sample_text = SQLBuilder._format_sampling_data(sampling_data)
            if sample_text:
                parts.append(f"{sample_text}\n")

        if knowledge:
            parts.append(f"## External Knowledge\n{knowledge}\n")

        parts.extend([
            f"## Question\n{question}\n",
            f"## Failed SQL\n```sql\n{sql}\n```\n",
            f"## SQLite Error\n{exec_error}\n",
            f"## Task\n"
            f"Fix the SQL query to resolve the SQLite execution error. "
            f"Return the SQLArtifact JSON contract with draft_intent and corrected sql.",
        ])

        user_prompt = "\n".join(parts)

        # 使用本路径的 SQLArtifact 输出约束，状态字典直接进入实际 system 消息。
        messages = [
            {"role": "system", "content": (
                "You are a SQL expert. Fix database execution errors. Return only a JSON object "
                "containing draft_intent, intent_version, intent_revision, and sql."
                + EXECUTION_STATE_PROMPT + GATE_CODE_PROMPT + REPAIR_STATE_PROMPT
            )},
            {"role": "user", "content": user_prompt},
        ]

        response = self._call_llm(messages)
        current_artifact = artifact or SQLArtifact.from_sql(sql)
        fixed_artifact = SQLBuilder.parse_artifact_response(response, current_artifact)
        fixed_sql = fixed_artifact.sql

        # 检查修复是否有效：
        # 1. 提取出了 SQL（不为空）
        # 2. SQL 看起来是合法的（不是 JSON 或说明文字）
        # 3. 修复后的 SQL 确实和原来不同（避免无意义的修复）
        if fixed_sql and self._looks_like_sql(fixed_sql) and fixed_sql != sql:
            return {
                "action": "fixed",
                "fixed_sql": fixed_sql,
                "fixed_artifact": fixed_artifact.to_dict(),
                "diagnosis": "Repaired based on SQLite execution error feedback",
                "repair_route": "database_sql_repair",
                "schema_retrieval": {"requested": False, "triggered": False, "added_tables": []},
                "relevant_schema": relevant_schema,
            }
        else:
            return {
                "action": "give_up",
                "diagnosis": "LLM could not produce a corrected SQL from the execution error",
                "repair_route": "database_sql_repair",
                "schema_retrieval": {"requested": False, "triggered": False, "added_tables": []},
                "relevant_schema": relevant_schema,
            }

    # ════════════════════════════════════════════════════════════════
    # 基于 Judge 评分反馈的修复方法
    # ════════════════════════════════════════════════════════════════

    def repair_from_judge_feedback(
        self,
        sql: str,
        judge_result: dict,
        relevant_schema: dict,
        question: str = "",
        sampling_data: dict = None,
        knowledge: str = "",
        full_schema: list[dict] = None,
        allow_schema_retrieval: bool = True,
        max_new_tables: int = 2,
        artifact: SQLArtifact = None,
    ) -> dict:
        """基于 NL2SQL Judge 的评分细目修复 SQL。

        与 repair_from_exec_error() 不同：
        - exec_error 只知道"SQL 不能执行"或"执行了但不知道对不对"
        - Judge feedback 知道"SQL 能执行但语义/逻辑/结果可能有问题"

        参数:
            sql: 待修复的 SQL
            judge_result: NL2SQLJudge.evaluate() 的返回字典
            relevant_schema: SchemaLinker 选中的表结构
            question: 用户原始问题
            sampling_data: 数据采样结果（含列实际值）
            knowledge: 外部知识/evidence
            full_schema: 保留的兼容参数；Refiner 不直接暴露全量 DDL
            allow_schema_retrieval: 本轮是否仍有二次检索预算
            max_new_tables: 本轮最多补充的表数

        返回:
            {"action": "fixed", "fixed_sql": str, "diagnosis": str}
            或 {"action": "give_up", "diagnosis": str}
        """
        from agent_team.builder import SQLBuilder
        structured_feedback = judge_result.get("structured_feedback", {})
        schema_request = structured_feedback.get("schema_search", {})
        effective_schema = relevant_schema
        schema_retrieval = {
            "requested": bool(schema_request.get("required")),
            "triggered": False,
            "added_tables": [],
        }
        repair_route = "sql_repair"

        # Judge 只负责指出缺口；是否补 Schema 由 Refiner 在自己的修复阶段决定。
        if schema_request.get("required"):
            repair_route = "l2_reschema"
            if allow_schema_retrieval and self.schema_retriever is not None:
                try:
                    before_names = {
                        table.get("name")
                        for table in relevant_schema.get("candidate_tables", [])
                    }
                    effective_schema = self.schema_retriever(
                        question=question,
                        current_schema=relevant_schema,
                        structured_feedback=structured_feedback,
                        max_new_tables=max_new_tables,
                    )
                    after_names = {
                        table.get("name")
                        for table in effective_schema.get("candidate_tables", [])
                    }
                    schema_retrieval.update({
                        "triggered": True,
                        "added_tables": sorted(
                            name for name in after_names - before_names if name
                        ),
                        "query_terms": schema_request.get("query_terms", [])[:12],
                        "max_new_tables": max_new_tables,
                    })
                except Exception as exc:
                    schema_retrieval["error"] = str(exc)[:300]
            elif not allow_schema_retrieval:
                schema_retrieval["skipped_reason"] = "reschema_budget_exhausted"
            else:
                schema_retrieval["skipped_reason"] = "schema_retriever_unavailable"

        # Refiner 只看到初始候选 + 受限补充结果，不直接接触全量 Schema。
        ddl = SQLBuilder._render_schema_ddl(effective_schema)

        # 把 Judge 的评分细目拼接为修复提示
        hints = self._format_judge_hints(judge_result)
        if not hints:
            return {
                "action": "give_up",
                "diagnosis": "Judge 未提供具体修复建议",
                "repair_route": repair_route,
                "schema_retrieval": schema_retrieval,
                "relevant_schema": effective_schema,
            }

        parts = [
            f"The following SQL query executes but has quality issues.\n",
            f"## Database Schema\n{ddl}\n",
        ]

        # 采样数据：让 Refiner 看到列的真实值，避免猜错枚举值
        if sampling_data:
            sample_text = SQLBuilder._format_sampling_data(sampling_data)
            if sample_text:
                parts.append(f"{sample_text}\n")

        # 外部知识（BIRD evidence）
        if knowledge:
            parts.append(f"## External Knowledge\n{knowledge}\n")

        parts.extend([
            f"## Question\n{question}\n",
            f"## Current SQL\n```sql\n{sql}\n```\n",
            f"## Quality Issues Found by Judge\n{hints}\n",
        ])

        if structured_feedback:
            parts.append(
                "## Structured Repair Contract\n"
                + json.dumps(structured_feedback, ensure_ascii=False, indent=2)
                + "\n"
            )
        parts.append(
            "## Schema Retrieval State\n"
            + json.dumps(schema_retrieval, ensure_ascii=False, indent=2)
        )

        intent_comparison = judge_result.get("intent_comparison", {})
        if intent_comparison:
            parts.append(
                "## Structured Intent Comparison\n"
                + json.dumps(intent_comparison, ensure_ascii=False, indent=2)
                + "\n"
            )

        # 注入语义往返信息：让 Refiner 知道 SQL↔问题之间的语义偏差
        reverse_q = judge_result.get("reverse_question", "")
        sem_sim = judge_result.get("semantic_similarity", 0)
        sem_score = judge_result.get("semantic_score", 0)
        if reverse_q:
            parts.append(
                f"## Semantic Round-Trip Analysis\n"
                f"The Judge reverse-engineered this question from your SQL:\n"
                f"  \"{reverse_q}\"\n"
                f"Semantic similarity to the original question: {sem_score}/50 "
                f"(cosine sim={sem_sim:.3f}).\n"
                f"Low similarity means the SQL may be answering a different question "
                f"than intended. Focus on aligning the SQL with the ORIGINAL question, "
                f"not the reverse-engineered one.\n"
            )

        parts.extend([
            f"## Task\n"
            f"Fix the SQL query to address all listed issues. "
            f"Keep the correct parts unchanged. "
            f"Return the SQLArtifact JSON contract with a challengeable draft_intent and corrected sql. "
            f"If the original DraftIntent was wrong, revise it, increment intent_version, and record intent_revision.reason.",
        ])

        user_prompt = "\n".join(parts)

        messages = [
            {"role": "system", "content": (
                "You are a SQL expert. Fix SQL queries based on quality review feedback. "
                "Pay special attention to the Semantic Round-Trip Analysis: "
                "if the reverse-engineered question differs from the original, "
                "your SQL is answering the wrong question. Re-read the original "
                "question carefully and rewrite the SQL to match it. "
                "Return only a JSON object containing draft_intent, intent_version, "
                "intent_revision, and sql, with no explanations."
                + EXECUTION_STATE_PROMPT + GATE_CODE_PROMPT
                + INTENT_STATE_PROMPT + REPAIR_STATE_PROMPT
            )},
            {"role": "user", "content": user_prompt},
        ]

        response = self._call_llm(messages)
        current_artifact = artifact or SQLArtifact.from_sql(sql)
        fixed_artifact = SQLBuilder.parse_artifact_response(response, current_artifact)
        fixed_sql = fixed_artifact.sql

        if fixed_sql and self._looks_like_sql(fixed_sql) and fixed_sql != sql:
            return {
                "action": "fixed",
                "fixed_sql": fixed_sql,
                "fixed_artifact": fixed_artifact.to_dict(),
                "diagnosis": f"Repaired based on Judge feedback: {judge_result.get('overall_assessment', '')[:200]}",
                "repair_route": repair_route,
                "schema_retrieval": schema_retrieval,
                "relevant_schema": effective_schema,
            }
        else:
            return {
                "action": "give_up",
                "diagnosis": "LLM could not produce a corrected SQL from Judge feedback",
                "repair_route": repair_route,
                "schema_retrieval": schema_retrieval,
                "relevant_schema": effective_schema,
            }

    @staticmethod
    def _render_full_schema_ddl(full_schema: list[dict]) -> str:
        """将全量 Schema 渲染为 DDL，供 Refiner 在 Judge 建议加表时使用。"""
        from agent_team.builder import SQLBuilder
        ddl_parts = []
        for table in full_schema:
            tn = table.get("table_name", "?")
            cols = table.get("columns", [])
            col_strs = [f"    {c['col']} {c.get('type', 'TEXT').upper()}" for c in cols]
            ddl_parts.append(f"CREATE TABLE {tn} (\n" + ",\n".join(col_strs) + "\n);")
        return "\n".join(ddl_parts)

    @staticmethod
    def _format_judge_hints(judge_result: dict) -> str:
        """将 Judge 评分细目格式化为 LLM 可用的修复提示文本。"""
        lines = []

        feedback = judge_result.get("structured_feedback", {})
        for issue in feedback.get("issues", []):
            lines.append(
                f"- [STRUCTURED:{issue.get('code', 'unknown')}] "
                f"{issue.get('message', '')}"
            )
            if issue.get("suggestion"):
                lines.append(f"  -> SUGGESTION: {issue['suggestion']}")
        schema_search = feedback.get("schema_search", {})
        if schema_search.get("required"):
            lines.append(
                "- [RESCHEMA] Restricted schema retrieval was requested. "
                f"Terms={schema_search.get('query_terms', [])}; "
                f"tables={schema_search.get('suggested_tables', [])}"
            )

        # 语义往返偏差提示（最重要的修复信号）
        reverse_q = judge_result.get("reverse_question", "")
        sem_score = judge_result.get("semantic_score", 0)
        if reverse_q:
            lines.append(f"- [SEMANTIC GAP] Judge reverse-engineered question: \"{reverse_q}\"")
            lines.append(f"  Semantic similarity score: {sem_score}/50. "
                          f"Your SQL may be answering a DIFFERENT question. "
                          f"Align SQL with the ORIGINAL question, not the reverse-engineered one.")

        for p in judge_result.get("repair_priority", []):
            lines.append(f"- [PRIORITY] {p}")

        for dim_name in ["semantics", "logic", "result_quality", "syntax"]:
            dim = judge_result.get("dimensions", {}).get(dim_name, {})
            label = dim_name
            for issue in dim.get("issues", []):
                lines.append(f"- [{label}] ISSUE: {issue}")
            for sug in dim.get("suggestions", []):
                lines.append(f"  -> SUGGESTION: {sug}")

        return "\n".join(lines) if lines else ""

    # ════════════════════════════════════════════════════════════════
    # 内部辅助方法
    # ════════════════════════════════════════════════════════════════

    def _call_llm(self, messages, max_retries=3):
        """调用 LLM 生成修复建议，带重试机制。

        使用较低的温度（0.1），因为修复工作更倾向于确定性而非创造性。

        参数:
            messages: OpenAI 格式的消息列表
            max_retries: 最大重试次数

        返回:
            LLM 响应文本，或空字符串（全部重试失败时）
        """
        for attempt in range(max_retries):
            try:
                response = self.model_client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.1,
                    n=1,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
        return ""

    def _extract_sql(self, text: str) -> str:
        """从 LLM 响应文本中提取 SQL 语句。

        与 Builder 中的提取逻辑相同，处理多种 LLM 输出格式：
        - ```sql ... ``` 标准代码块
        - ``` ... ``` 无标签代码块
        - 纯文本
        - 引号包裹的文本

        参数:
            text: LLM 返回的原始文本

        返回:
            提取出的纯净 SQL 字符串
        """
        # 匹配 ```sql ... ``` 或 ``` ... ``` 代码块
        sql_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        # 如果没有代码块，尝试去掉外层引号
        t = text.strip()
        if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
            t = t[1:-1].strip()
        return t

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        """判断一段文本是否看起来像合法的 SQL 语句。

        用于过滤掉 LLM 返回的非 SQL 内容（如 JSON、解释文字等）。

        判断规则：
        - 非空
        - 不以 { 或 ```json 开头（排除 JSON）
        - 以 SQL 关键词开头：SELECT/WITH/INSERT/UPDATE/DELETE/CREATE/ALTER/DROP

        参数:
            text: 要判断的文本

        返回:
            True 如果文本看起来像 SQL 语句
        """
        t = text.strip().lower()
        if not t:
            return False
        if t.startswith("{") or t.startswith("```json"):
            return False
        sql_keywords = ["select ", "with ", "insert ", "update ", "delete ", "create ", "alter ", "drop "]
        return any(t.startswith(kw) for kw in sql_keywords)
