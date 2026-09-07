# nl2sql/agent_team/orchestrator.py
"""统一的 NL2SQL 管道编排器：SchemaLinker → DataSampler → Builder → Execute → Judge → Refine 循环。

本文件是整个 NL2SQL（自然语言转 SQL）多智能体系统的"总指挥"。
它串联起各个子模块，形成一个完整的处理流水线：

  用户提问 → 关联 Schema → (可选)数据采样 → 生成 SQL → 执行 SQL → Judge 评分 → (Refine 修复)*

已移除的组件：
  - Planner: LLM 规划环节已被移除，对 BIRD 场景引入不必要的复杂性和错误
  - 静态 ValidationEngine: 会误报合法 SQL，已被基于实际执行的 Judge 评分机制取代
  - game 领域静态验证路径: 游戏数仓专用逻辑已整体移除
"""

import json
import os
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent_team.contracts import RetryState, SQLArtifact
from agent_team.executor import (
    CallbackExecutorAdapter,
    DatabaseExecutor,
    SafeExecutor,
    SQLiteReadOnlyExecutor,
)
from agent_team.schema_linker import SchemaLinker
from agent_team.builder import SQLBuilder
from agent_team.refiner import Refiner
from agent_team.planner import Planner
from agent_team.nl2sql_judge import NL2SQLJudge
from agent_team.shared_memory import SharedMemory
from agent_team.memory_manager import AgentMemoryManager, MemoryEntry


# 执行回调函数的类型签名（Callable 类型别名）
# 输入：一条 SQL 字符串
# 输出：一个字典，包含执行结果信息
ExecuteCallback = Callable[[str], dict]
# 回调函数返回的字典格式：
# {
#   "ok": bool,       # SQL 是否执行成功
#   "error": str,     # 若失败，错误信息
#   "rows": list,     # 查询结果的行数据
#   "row_count": int  # 结果行数
# }


def _estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数量。

    对于英文文本，大约 4 个字符对应 1 个 token。
    这个估算用于日志记录和成本追踪，不需要精确。

    参数:
        text: 要估算的文本

    返回:
        估算的 token 数量（至少为 1）
    """
    return max(1, len(text) // 4)


@dataclass
class QueryResult:
    """保存一次 NL2SQL 管道运行的完整结果。

    这个数据类（dataclass）记录了从用户提问到最终 SQL 生成的
    全过程信息，包括中间产物（plan、relevant_schema）和
    最终结果（sql、success、error）。

    属性说明:
        question: 用户的原始自然语言问题
        sql: 最终生成的 SQL 语句
        plan: 执行计划（当前为占位符 _DIRECT_PLAN）
        relevant_schema: SchemaLinker 筛选出的相关表结构
        validation_report: 验证报告（已废弃，保留兼容性）
        iterations: 实际执行的迭代次数（含重试）
        success: 是否最终生成可执行的正确 SQL
        error: 错误信息（如果有）
        conversation_log: 每次迭代的详细日志，用于调试和追踪
        exec_ok: 基于执行反馈的 SQL 执行是否成功
        exec_error: 执行 SQL 时的错误信息
        sampling_data: DataSampler 采样得到的数据，用于辅助 LLM 生成更准确的 SQL
    """
    question: str
    sql: Optional[str] = None
    plan: Optional[dict] = None
    relevant_schema: Optional[dict] = None
    validation_report: Optional[dict] = None
    iterations: int = 0
    success: bool = False
    error: Optional[str] = None
    conversation_log: list = field(default_factory=list)
    exec_ok: bool = False
    exec_error: Optional[str] = None
    sampling_data: Optional[dict] = None
    status: str = "failed"
    sql_generated: bool = False
    gate_passed: Optional[bool] = None
    execution_attempted: bool = False
    judge_passed: Optional[bool] = None
    draft_intent: Optional[dict] = None
    retry_state: dict = field(default_factory=dict)


class Orchestrator:
    """统一的 NL2SQL 管道编排器：Schema 关联 → DataSampler → Builder → Execute → (Refine 循环)。

    本类是整个 NL2SQL 系统的核心入口，负责：
    1. 接收用户的自然语言问题
    2. Schema 关联（LLM 始终参与选表）：
       小库(≤10表): Planner(全量DDL) 直接选表
       大库(>10表): SchemaLinker(embedding粗筛) → Planner(精筛)
    3. 可选地使用 DataSampler 从数据库中采样数据
    4. 使用 SQLBuilder 生成 SQL
    5. 若提供了执行回调，执行 SQL 并将错误反馈给 Builder 进行修复
    6. 在没有执行回调时，回退到旧版静态验证循环

    Domain（领域）参数控制加载哪套 prompt：
      - "generic": 通用 SQLite / 标准 NL2SQL 场景（默认）
      - "enterprise": CTE、窗口函数、递归查询等企业级 SQL 特性

    执行反馈模式: SchemaLinker → Builder → Execute → Judge → (Refine 循环)
    """

    MAX_GATE_REPAIRS = 2
    MAX_EXECUTION_REPAIRS = 2
    MAX_SEMANTIC_REPAIRS = 5
    MAX_TOTAL_SQL_ATTEMPTS = 10
    # Judge 失败后最多做一次受限增量检索；每次最多补 2 张表。
    MAX_RESCHEMA_ITERATIONS = 1

    _DIRECT_PLAN = {"approach": "direct_sql", "anchor_table": ""}

    def __init__(
        self,
        schema: list[dict],
        model: str = "deepseek-v4-flash",
        model_client: Optional[object] = None,
        domain: str = "generic",
        dialect: str = "sqlite",
    ):
        """初始化 NL2SQL 管道的编排器。

        参数:
            schema: 数据库表结构列表，供 SchemaLinker 使用
            model: 使用的 LLM 模型名称
            model_client: LLM 客户端（例如 OpenAI 实例），None 时会自动创建
            domain: 使用领域（generic/enterprise），决定使用哪套 prompt
        """
        self.schema = schema
        self.model = model
        self.model_client = model_client
        self.domain = domain
        self.dialect = dialect

        # Planner + Judge（惰性初始化）
        self._planner = None
        self._judge = None

        self.schema_linker = SchemaLinker(schema)
        self.builder = SQLBuilder(
            model_client=model_client, model=model, domain=domain, dialect=dialect
        )
        self.refiner = Refiner(
            model_client=model_client,
            model=model,
            domain=domain,
            schema_retriever=self.schema_linker.relink_from_feedback,
        )

        # DataSampler 惰性初始化（只在需要时才创建，节省资源）
        self._data_sampler = None
        # SQL 经验 RAG 惰性初始化（首次使用时加载或创建）
        self._rag = None

        # 记忆管理器：记录成功和失败的构建模式，用于后续学习改进
        memories_dir = os.path.join(os.path.dirname(__file__), "memories")
        self.memory_manager = AgentMemoryManager(memories_dir)

        # 线程锁：保护并发场景下的共享资源写入
        self._rag_lock = threading.Lock()
        self._rag_init_lock = threading.Lock()

    @property
    def data_sampler(self):
        """惰性加载的 DataSampler 实例。"""
        if self._data_sampler is None:
            from agent_team.data_sampler import DataSampler
            self._data_sampler = DataSampler()
        return self._data_sampler

    @property
    def planner(self) -> Planner:
        if self._planner is None:
            self._planner = Planner(
                model_client=self.model_client, model=self.model,
            )
        return self._planner

    @property
    def judge(self) -> NL2SQLJudge:
        if self._judge is None:
            self._judge = NL2SQLJudge(
                model_client=self.model_client, model=self.model,
            )
        return self._judge

    @property
    def rag(self):
        """惰性加载的 SQL 经验 RAG 存储（线程安全）。

        首次访问时尝试从 data/sql_experiences.json 加载已有经验库，
        若文件不存在则创建空库。然后合并 spider pre-built 库作为冷启动数据。
        后续每次管道运行成功/失败都会写入。
        """
        if self._rag is None:
            with self._rag_init_lock:
                if self._rag is None:
                    from agent_team.sql_rag import SQLExperienceStore
                    self._rag = SQLExperienceStore(
                        model_client=self.model_client, model=self.model,
                    )
                    data_dir = os.path.join(os.path.dirname(__file__), "data")
                    rag_path = os.path.join(data_dir, "sql_experiences.json")
                    if os.path.exists(rag_path):
                        try:
                            self._rag.load(rag_path)
                        except Exception:
                            pass

        return self._rag

    @staticmethod
    def _log(module: str, stage: str, **kwargs) -> dict:
        """创建一条标准化的结构化日志条目。

        每条日志都包含时间戳和模块名，方便追踪每个步骤的耗时和来源。
        后续的 kwargs 可以是任意的键值对（如 ok、error、sql、tables 等），
        会被合并到日志字典中。

        参数:
            module: 模块名称（如 "schema_linker", "builder", "refiner"）
            stage: 阶段标识（如 "link", "build_iter_0", "execute_iter_1"）

        返回:
            一个字典，包含 timestamp、module、stage 以及所有 kwargs 的键值对
        """
        entry = {
            "timestamp": time.time(),
            "module": module,
            "stage": stage,
        }
        entry.update(kwargs)
        return entry

    def _log_elapsed(self, entry: dict, start_time: float) -> dict:
        """在日志条目中补充从 start_time 到现在的耗时（毫秒）。

        用于在步骤完成后回填 duration_ms，方便分析性能瓶颈。

        参数:
            entry: 已创建的日志条目字典
            start_time: 该步骤开始时 time.time() 的返回值

        返回:
            增加了 duration_ms 字段的日志条目
        """
        entry["duration_ms"] = round((time.time() - start_time) * 1000)
        return entry

    def _rag_add_experience(self, question: str, result: QueryResult,
                            relevant_schema: dict):
        """将本次运行结果写入 SQL 经验 RAG（线程安全）。

        无论成功还是失败都写入——失败的 error_type 可以帮助后续
        相似问题避免踩同样的坑。

        参数:
            question: 用户原始问题
            result: 管道的完整运行结果
            relevant_schema: SchemaLinker 的分析结果
        """
        try:
            from agent_team.sql_rag import SQLExperience, classify_error
            tables_used = [
                t["name"] for t in relevant_schema.get("candidate_tables", [])
            ]
            exp = SQLExperience(
                question=question,
                sql=result.sql or "",
                db_id="",
                tables_used=tables_used,
                success=result.success,
                error_type=classify_error(result.error or "") if not result.success else "",
                iteration_count=result.iterations,
            )
            with self._rag_lock:
                self.rag.add(exp)
                rag_path = os.path.join(
                    os.path.dirname(__file__), "data", "sql_experiences.json"
                )
                os.makedirs(os.path.dirname(rag_path), exist_ok=True)
                self.rag.save(rag_path)
        except Exception:
            pass  # RAG 写入失败不影响主流程

    def _save_trace(self, result: QueryResult):
        """将本次运行的完整 trace 自动落盘到 traces/ 目录。

        文件名格式: traces/{timestamp}_{question前30字}.json
        这份文件可以直接用于事后分析和调试。

        参数:
            result: 管道的完整运行结果
        """
        try:
            traces_dir = os.path.join(os.path.dirname(__file__), "traces")
            os.makedirs(traces_dir, exist_ok=True)
            safe_question = "".join(
                c if c.isalnum() or c in " _-" else "_"
                for c in result.question[:30]
            ).strip().replace(" ", "_")
            uid = uuid.uuid4().hex[:8]
            filename = f"{int(time.time())}_{uid}_{safe_question}.json"
            filepath = os.path.join(traces_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "question": result.question,
                    "pred_sql": result.sql,
                    "success": result.success,
                    "error": result.error,
                    "iterations": result.iterations,
                    "conversation_log": result.conversation_log,
                    "relevant_schema": result.relevant_schema,
                    "plan": result.plan,
                    "validation_report": result.validation_report,
                    "exec_ok": result.exec_ok,
                    "exec_error": result.exec_error,
                    "status": result.status,
                    "sql_generated": result.sql_generated,
                    "gate_passed": result.gate_passed,
                    "execution_attempted": result.execution_attempted,
                    "judge_passed": result.judge_passed,
                    "draft_intent": result.draft_intent,
                    "retry_state": result.retry_state,
                }, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass  # trace 落盘失败不影响主流程

    # ── 主入口方法 ──────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        knowledge: str = "",
        memory: Optional[SharedMemory] = None,
        execute_sql: Optional[ExecuteCallback] = None,
        db_path: str = "",
        database_executor: Optional[DatabaseExecutor] = None,
    ) -> QueryResult:
        """执行完整的 NL2SQL 管道。

        流程：
        1. SchemaLinker (BGE embedding) 统一选表
        2. Execute → Judge → Refine 循环

        参数:
            question: 用户的自然语言查询
            knowledge: 可选的额外领域知识（BIRD evidence）
            memory: 可选 SharedMemory
            execute_sql: SQL 执行回调
            db_path: SQLite 数据库路径，用于数据采样

        返回:
            QueryResult 对象
        """
        result = QueryResult(question=question)
        mem = memory or SharedMemory()
        mem.session.set("question", question)

        # 步骤 1：Schema 关联 — SchemaLinker (BGE embedding) 统一选表
        t0 = time.time()
        try:
            relevant_schema = self.schema_linker.link(question, evidence=knowledge)
            if not relevant_schema.get("candidate_tables"):
                result.error = "SchemaLinker: no relevant tables found"
                self._save_trace(result)
                return result

            result.relevant_schema = relevant_schema
            tables_info = [
                {"name": t["name"], "score": t.get("score", 0)}
                for t in relevant_schema.get("candidate_tables", [])
            ]
            result.conversation_log.append(self._log_elapsed(self._log(
                "schema_linker", "link", ok=True,
                candidate_table_count=len(tables_info),
                tables=tables_info[:10],
            ), t0))
        except Exception as e:
            result.error = f"Schema Linking error: {e}"
            self._save_trace(result)
            return result

        if not relevant_schema.get("candidate_tables"):
            result.error = "No relevant tables found for this question"
            self._save_trace(result)
            return result

        # 步骤 2：数据采样（零 LLM 调用——纯 SQL 查询）
        # 仅当提供了 db_path 时才执行此步骤
        # 采样的数据（如列值的分布、示例值等）可以帮助 LLM 更准确地生成 SQL
        sampling_data = None
        if db_path:
            t0_samp = time.time()
            try:
                sampling_data = self.data_sampler.sample(relevant_schema, db_path, question)
                result.sampling_data = sampling_data
                sampled_tables = list(sampling_data.get("table_samples", {}).keys())
                result.conversation_log.append(self._log_elapsed(self._log(
                    "data_sampler", "sample", ok=True,
                    sampled_table_count=len(sampled_tables),
                    sampled_tables=sampled_tables,
                ), t0_samp))
            except Exception as e:
                # 数据采样失败不是致命错误，Builder 仍然可以在没有采样数据的情况下工作
                result.conversation_log.append(self._log_elapsed(self._log(
                    "data_sampler", "sample", ok=False,
                    error=str(e)[:300],
                ), t0_samp))

        # 步骤 3：执行反馈 + Judge 模式
        executor = database_executor
        if executor is None and execute_sql is not None:
            executor = CallbackExecutorAdapter(execute_sql)
        elif executor is None and db_path:
            executor = SQLiteReadOnlyExecutor(db_path)
        safe_executor = (
            SafeExecutor(executor, dialect=self.dialect) if executor is not None else None
        )

        result = self._run_with_execution_feedback(
            result, relevant_schema, sampling_data, knowledge,
            safe_executor, question,
        )
        self._save_trace(result)
        return result

    # ── 辅助方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _format_judge_feedback(judge_result: dict) -> str:
        """将 Judge 评分细目拼接为 Builder 可用的 repair_hints 字符串。"""
        lines = []
        for p in judge_result.get("repair_priority", []):
            lines.append(f"- [PRIORITY] {p}")
        for dim_name in ["semantics", "logic", "result_quality", "syntax"]:
            dim = judge_result.get("dimensions", {}).get(dim_name, {})
            for issue in dim.get("issues", []):
                lines.append(f"- [{dim_name}] {issue}")
            for sug in dim.get("suggestions", []):
                lines.append(f"  Suggestion: {sug}")
        return "\n".join(lines) if lines else ""

    # ── 执行反馈循环（执行 + Judge 评分驱动修复） ─────────────────────

    def _run_with_execution_feedback(
        self,
        result: QueryResult,
        relevant_schema: dict,
        sampling_data: Optional[dict],
        knowledge: str,
        safe_executor: Optional[SafeExecutor],
        question: str,
    ) -> QueryResult:
        """Run Builder -> SafeExecutor -> Judge -> Refiner as an explicit state machine."""
        if safe_executor is None:
            t0 = time.time()
            artifact = self.builder.build_artifact(
                self._DIRECT_PLAN, relevant_schema, knowledge,
                question=question, sampling_data=sampling_data,
            )
            result.sql = artifact.sql
            result.sql_generated = bool(artifact.sql)
            result.draft_intent = (
                artifact.draft_intent.__dict__ if artifact.draft_intent else None
            )
            result.status = "generated_unverified"
            result.success = False
            result.gate_passed = None
            result.execution_attempted = False
            result.judge_passed = None
            result.conversation_log.append(self._log_elapsed(self._log(
                "builder", "build_direct", sql=artifact.sql,
                sql_length=len(artifact.sql or ""), model=self.model,
                domain=self.domain, status=result.status,
                generation_warnings=artifact.generation_warnings,
            ), t0))
            return result

        base_knowledge = knowledge or ""
        effective_plan = self._DIRECT_PLAN

        # RAG 检索
        experiences = self.rag.retrieve(question, top_k=3)
        if experiences:
            exp_text = "## 历史相似经验（RAG 检索）\n\n"
            exp_text += "以下经验仅供参考。请根据当前具体问题调整 SQL。\n\n"
            for i, exp in enumerate(experiences, 1):
                status = "成功" if exp.success else f"失败({exp.error_type})"
                exp_text += (
                    f"### 经验 {i} [{status}]\n"
                    f"问题: {exp.question[:200]}\n"
                    f"使用的表: {', '.join(exp.tables_used) if exp.tables_used else '未知'}\n"
                    f"SQL 结构: JOIN {exp._count_joins()} 个表"
                )
                if exp.sql:
                    from agent_team.sql_rag import SQLExperienceStore as _SES
                    sql_structure = _SES._sql_structure_key(exp.sql)
                    if sql_structure.get("has_set_op"):
                        exp_text += f", 使用 {exp._detect_set_op()}"
                    if sql_structure.get("has_group_by"):
                        exp_text += ", 含 GROUP BY"
                    if sql_structure.get("has_subquery"):
                        exp_text += ", 含子查询"
                exp_text += "\n\n"
            base_knowledge = exp_text + base_knowledge
            result.conversation_log.append(self._log(
                "rag", "retrieve",
                retrieved_count=len(experiences),
                experiences_summary=[
                    {"question": e.question[:80], "success": e.success}
                    for e in experiences
                ],
            ))

        retry_state = RetryState()
        seen_sqls: set[str] = set()
        previous_fingerprint = ""
        repeated_fingerprint = 0

        # ── iter=0: Builder 生成初始 SQL ──
        t0 = time.time()
        try:
            artifact = self.builder.build_artifact(
                effective_plan, relevant_schema, base_knowledge,
                question=question, sampling_data=sampling_data,
            )
        except Exception as e:
            result.error = f"SQL Build error: {e}"
            return result

        result.sql = artifact.sql
        result.sql_generated = bool(artifact.sql)
        result.draft_intent = artifact.draft_intent.__dict__ if artifact.draft_intent else None
        result.conversation_log.append(self._log_elapsed(self._log(
            "builder", "build_iter_0",
            action="initial_build", sql=artifact.sql, sql_length=len(artifact.sql or ""),
            model=self.model, domain=self.domain,
            intent_version=artifact.intent_version,
            draft_intent=result.draft_intent,
            generation_warnings=artifact.generation_warnings,
            estimated_input_tokens=_estimate_tokens(
                f"{self.builder.system_prompt}\n{question}\n"
                f"{json.dumps(relevant_schema, ensure_ascii=False)}"
            ),
            estimated_output_tokens=_estimate_tokens(artifact.raw_response or artifact.sql),
        ), t0))

        while retry_state.total_sql_attempts < self.MAX_TOTAL_SQL_ATTEMPTS:
            retry_state.total_sql_attempts += 1
            result.iterations = retry_state.total_sql_attempts
            result.retry_state = retry_state.to_dict()

            sql_key = self._normalize_sql(artifact.sql)
            if sql_key in seen_sqls:
                result.error = "Refiner 生成重复 SQL，终止"
                result.status = "stopped_duplicate_sql"
                break
            seen_sqls.add(sql_key)

            # Every artifact gets a fresh Gate result before any database call.
            t0_exec = time.time()
            attempt = safe_executor.attempt(
                artifact, relevant_schema, self.schema, mode="probe"
            )
            exec_result = attempt.to_dict()
            result.exec_ok = attempt.ok
            result.exec_error = attempt.error
            result.gate_passed = attempt.gate.passed
            result.execution_attempted = attempt.attempted

            result.conversation_log.append(self._log_elapsed(self._log(
                "safe_executor", f"attempt_{result.iterations}",
                attempt_stage=attempt.stage, attempted=attempt.attempted, ok=attempt.ok,
                error=(result.exec_error or "")[:500],
                error_code=attempt.error_code,
                gate=attempt.gate.to_dict(),
                mode=attempt.mode, row_count=attempt.row_count,
                row_count_exact=attempt.row_count_exact,
                truncated=attempt.truncated,
                read_only_enforced=attempt.read_only_enforced,
            ), t0_exec))

            t0_judge = time.time()
            try:
                judge_result = self.judge.evaluate(
                    question=question, sql=artifact.sql, exec_result=exec_result,
                    relevant_schema=relevant_schema,
                    full_schema=self.schema,
                    evidence=knowledge,
                    plan=effective_plan if effective_plan.get("anchor_table") else None,
                    sampling_data=sampling_data,
                    gate_result=attempt.gate,
                    draft_intent=artifact.draft_intent,
                    dialect=self.dialect,
                )
                failure_stage = judge_result.get("judge_mode", "semantic")
                result.conversation_log.append(self._log_elapsed(self._log(
                    "judge", f"judge_{result.iterations}",
                    mode=failure_stage,
                    overall_confidence=judge_result.get("overall_confidence", 0),
                    pass_=judge_result.get("pass", False),
                    dimensions={
                        k: v.get("score", 0)
                        for k, v in judge_result.get("dimensions", {}).items()
                    },
                    critical_flaws=judge_result.get("critical_flaws", []),
                    repair_priority=judge_result.get("repair_priority", [])[:5],
                    reverse_question=judge_result.get("reverse_question", ""),
                    semantic_similarity=judge_result.get("semantic_similarity", 0),
                    semantic_score=judge_result.get("semantic_score", 0),
                    deterministic_blockers=len(
                        judge_result.get("deterministic_checks", {}).get("blocking_issues", [])
                    ),
                    intent_match=judge_result.get("intent_comparison", {}).get("match"),
                    structured_feedback=judge_result.get("structured_feedback", {}),
                ), t0_judge))
            except Exception as e:
                result.conversation_log.append(self._log(
                    "judge", f"judge_{result.iterations}",
                    ok=False, error=str(e)[:200],
                ))
                result.error = f"Judge error: {e}"
                result.status = "judge_error"
                self._rag_add_experience(question, result, relevant_schema)
                return result

            if judge_result.get("pass"):
                result.success = True
                result.status = "verified"
                result.judge_passed = True
                result.validation_report = judge_result
                result.retry_state = retry_state.to_dict()
                self._rag_add_experience(question, result, relevant_schema)
                return result

            result.judge_passed = False
            result.validation_report = judge_result
            fingerprint = self._failure_fingerprint(failure_stage, judge_result)
            if fingerprint == previous_fingerprint:
                repeated_fingerprint += 1
            else:
                previous_fingerprint = fingerprint
                repeated_fingerprint = 1
            if repeated_fingerprint >= 2:
                result.error = f"同类 {failure_stage} 失败连续出现，终止"
                result.status = "stopped_repeated_failure"
                break

            budget_error = self._consume_repair_budget(retry_state, failure_stage)
            result.retry_state = retry_state.to_dict()
            if budget_error:
                result.error = budget_error
                result.status = "repair_budget_exhausted"
                break
            if not judge_result.get("structured_feedback", {}).get("retryable", True):
                result.error = "当前失败不可通过 SQL 重试修复"
                result.status = "non_retryable_failure"
                break

            t0_repair = time.time()
            try:
                repair_feedback = dict(judge_result)
                repair_feedback["error"] = attempt.error
                judge_repair = self.refiner.repair(
                    artifact=artifact,
                    failure_stage=failure_stage,
                    feedback=repair_feedback,
                    relevant_schema=relevant_schema,
                    question=question,
                    sampling_data=sampling_data, knowledge=knowledge,
                    allow_schema_retrieval=(
                        failure_stage == "semantic"
                        and retry_state.reschema_attempts < self.MAX_RESCHEMA_ITERATIONS
                    ),
                    max_new_tables=2,
                )
                repair_action = judge_repair.get("action", "?")
            except Exception:
                repair_action = "error"
                judge_repair = {"action": "error", "diagnosis": "Refiner exception"}

            schema_retrieval = judge_repair.get("schema_retrieval", {})
            if schema_retrieval.get("triggered"):
                retry_state.reschema_attempts += 1
                relevant_schema = judge_repair.get("relevant_schema", relevant_schema)
                result.relevant_schema = relevant_schema

            result.conversation_log.append(self._log_elapsed(self._log(
                "refiner", f"refine_{result.iterations}",
                failure_stage=failure_stage,
                action=repair_action,
                repair_route=judge_repair.get("repair_route", "unknown"),
                diagnosis=judge_repair.get("diagnosis", "")[:300],
                schema_retrieval=schema_retrieval,
                reschema_attempt=retry_state.reschema_attempts,
            ), t0_repair))

            if judge_repair.get("action") == "fixed":
                fixed_value = judge_repair.get("fixed_artifact")
                artifact = SQLArtifact.from_value(fixed_value) or SQLArtifact.from_sql(
                    judge_repair.get("fixed_sql", "")
                )
                result.sql = artifact.sql
                result.sql_generated = bool(artifact.sql)
                result.draft_intent = artifact.draft_intent.__dict__ if artifact.draft_intent else None
                result.retry_state = retry_state.to_dict()
                continue

            result.error = f"Refiner 无法修复 (action={repair_action})"
            result.status = "refiner_give_up"
            break

        if retry_state.total_sql_attempts >= self.MAX_TOTAL_SQL_ATTEMPTS and not result.error:
            result.error = f"超过最大 SQL 尝试次数 {self.MAX_TOTAL_SQL_ATTEMPTS}"
            result.status = "total_attempt_budget_exhausted"
        result.retry_state = retry_state.to_dict()
        self._rag_add_experience(question, result, relevant_schema)
        return result

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return " ".join((sql or "").lower().split())

    @staticmethod
    def _failure_fingerprint(stage: str, judge_result: dict) -> str:
        issues = judge_result.get("structured_feedback", {}).get("issues", [])
        normalized = []
        for issue in issues:
            normalized.append({
                "code": issue.get("code", "unknown"),
                "evidence": issue.get("evidence", ""),
            })
        return json.dumps(
            {"stage": stage, "issues": normalized},
            ensure_ascii=True, sort_keys=True, default=str,
        )

    def _consume_repair_budget(self, state: RetryState, stage: str) -> str:
        limits = {
            "gate": ("gate_repairs", self.MAX_GATE_REPAIRS),
            "database": ("execution_repairs", self.MAX_EXECUTION_REPAIRS),
            "semantic": ("semantic_repairs", self.MAX_SEMANTIC_REPAIRS),
        }
        if stage not in limits:
            return f"未知失败阶段: {stage}"
        field_name, limit = limits[stage]
        used = getattr(state, field_name)
        if used >= limit:
            return f"{stage} 修复预算已耗尽 ({used}/{limit})"
        setattr(state, field_name, used + 1)
        return ""
