# nl2sql/agent_team/shared_memory.py
#
# 共享内存系统 —— 多 Agent 之间的共享记忆层。
#
# 为什么需要共享内存？
# ===================
# 在多 Agent 系统中，不同 Agent 各自有独立的职责，但它们需要协作解决同一个问题。
# 共享内存就是它们交换信息的"黑板"（blackboard）：
#   - Planner 把规划结果写到共享内存
#   - Builder 从中读取规划来构建 SQL
#   - Refiner 把修复结果写回去
#   - SchemaLinker 把模式分析结果放进去
# 所有 Agent 通过共享内存来沟通，而不是直接调用彼此的方法。
# 这种"松耦合"（loose coupling）设计让系统更灵活、更容易扩展。
#
# 三层内存架构
# ============
# 本系统将共享内存分为三个层次，每个层次存储不同类型的记忆：
#
# 1. SessionMemory（会话内存）—— 存储当前问题的临时状态
#    - 分析过程中的临时变量
#    - 做出的假设（assumptions）
#    - 发现的歧义点（ambiguities）及其解决状态
#    生命周期：仅当前问题（一次自然语言查询）
#
# 2. SchemaMemory（模式内存）—— 存储数据库模式相关的信息
#    - 锚表（anchor_table）：最核心的表
#    - 候选表（candidate_tables）：可能与问题相关的表
#    - JOIN 路径（join_paths）：表之间的连接关系
#    - 时间字段（time_field）：用于时间过滤的字段
#    - 危险字段（dangerous_fields）：需要小心处理的字段
#    生命周期：跨问题，但随对话上下文变化
#
# 3. QueryMemory（查询内存）—— 存储 SQL 生成和验证相关的信息
#    - SQL 变体（sql_variants）：生成的多种 SQL 方案
#    - 验证结果（validation_results）：每种方案检查的结果
#    - 失败尝试（failed_attempts）：执行失败的记录
#    - 置信度（confidence）：当前方案的可靠程度评估
#    生命周期：当前问题的 SQL 生成过程
#
# 为什么用三层？
# =============
# 分层是为了"关注点分离"（Separation of Concerns）：
#   - 会话相关 vs 模式相关 vs 查询相关的信息互不干扰
#   - 每层可以独立重置和序列化
#   - 不同 Agent 主要关注不同的层（如 SchemaLinker 主要读写 SchemaMemory）
from typing import Any, Optional


class SessionMemory:
    """会话内存 —— 存储当前用户问题的临时状态。

    初学者理解：
    -----------
    SessionMemory 就像你在白板上的草稿 —— 写下当前问题的思考过程：
    - 你做了一些假设（"用户可能指的是销售表"）
    - 你发现了一些不明确的地方（"销售额是按年还是按月？"）
    - 你记下了临时变量方便后续使用
    这些问题处理完之后，白板就可以擦掉了。

    生命周期
    ========
    仅存在于当前自然语言问题的处理过程中。
    下一个问题进来时，会通过 SharedMemory.reset() 清空。
    """

    def __init__(self):
        # _store 字典：通用的键值存储，用于存放任意临时数据
        # 为什么用私有属性（_store）而不是公有属性？
        # 因为通过 set/get 方法存取可以更好地控制接口，
        # 未来可以方便地加入日志、验证等逻辑。
        self._store: dict[str, Any] = {}
        # 分析过程中的假设列表
        # 例如："假设 'sales' 指的是 'sales_2023' 表"
        self.assumptions: list[str] = []
        # 发现的歧义点列表
        # 每个元素格式：{"description": "歧义描述", "resolved": False}
        self.ambiguities: list[dict] = []

    def set(self, key: str, value: Any) -> None:
        """存储一个临时变量到会话内存中。"""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """从会话内存中读取一个临时变量。"""
        return self._store.get(key, default)

    def add_assumption(self, text: str) -> None:
        """记录一条假设。

        什么是"假设"？
        -------------
        在分析自然语言问题时，Agent 经常需要做一些合理推测。
        例如用户说"上个月的销售额"，Agent 会假设"上个月"指的是
        当前日期的前一个月。记录这些假设有助于：
        1. 后续 Agent 了解推理过程
        2. 如果发现假设错误，可以回溯修正
        """
        self.assumptions.append(text)

    def add_ambiguity(self, description: str) -> int:
        """记录一个歧义点。

        什么是"歧义"？
        -------------
        自然语言天然存在歧义。"销售额最高的产品"可能指：
        - 总销售额最高的产品
        - 平均单价最高的产品
        - 销量最多的产品
        记录歧义点意味着 Agent 意识到了这种不确定性。

        参数
        ----
        description: 歧义描述。

        返回
        ----
        歧义点的索引（可用于后续标记为已解决）。
        """
        idx = len(self.ambiguities)
        self.ambiguities.append({"description": description, "resolved": False})
        return idx

    def resolve_ambiguity(self, idx: int) -> None:
        """将指定索引的歧义点标记为"已解决"。

        当 Agent 通过后续分析（或追问用户）消除了歧义后，
        调用此方法更新状态。
        """
        if 0 <= idx < len(self.ambiguities):
            self.ambiguities[idx]["resolved"] = True

    def to_dict(self) -> dict:
        """将会话内存序列化为字典，用于快照保存。"""
        return {
            "store": self._store,
            "assumptions": self.assumptions,
            "ambiguities": self.ambiguities,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionMemory":
        """从字典恢复会话内存（反序列化）。"""
        sm = cls()
        sm._store = d.get("store", {})
        sm.assumptions = d.get("assumptions", [])
        sm.ambiguities = d.get("ambiguities", [])
        return sm


class SchemaMemory:
    """模式内存 —— 存储数据库模式（schema）级别的信息。

    初学者理解：
    -----------
    SchemaMemory 就像一张数据库关系图 —— 它记录了：
    - 哪些表是与问题最相关的（锚表和候选表）
    - 这些表之间如何关联（JOIN 路径）
    - 哪些关联被排除了（被拒绝的 JOIN）
    - 哪个字段用于时间过滤（如 created_at）
    - 哪些字段有潜在风险（如 NULL 很多、类型特殊等）

    这些信息主要由 SchemaLinker（模式链接器）填充，
    由 Planner（规划器）和 Builder（构建器）使用。
    """

    def __init__(self):
        # 锚表（Anchor Table）：问题中最重要的核心表
        # 例如查询"每个客户的订单总额"，锚表可能是 "orders"
        self.anchor_table: Optional[str] = None

        # 候选表列表：可能与问题相关的所有表
        # 每个元素格式：{"name": "table_name", ...}
        self.candidate_tables: list[dict] = []

        # JOIN 路径列表：表之间的关联关系
        # 每个元素格式：
        #   {"from": "table_a", "to": "table_b", "on": [...], "confidence": "medium"}
        self.join_paths: list[dict] = []

        # 被拒绝的 JOIN 路径列表：经过分析认为不可靠的关联
        # 记录这些可以避免重复尝试已知无效的 JOIN
        self.rejected_joins: list[dict] = []

        # 时间字段信息：用于时间范围过滤（如"上个月"、"今年"）
        # confirmed=False 表示尚未确认
        self.time_field: dict = {
            "field": None,
            "table": None,
            "rationale": None,
            "confirmed": False,
        }

        # 危险字段列表：有特殊需要注意的字段
        # 例如：高 NULL 率字段、类型可疑字段等
        self.dangerous_fields: list[dict] = []

    def set_anchor_table(self, name: str) -> None:
        """设置锚表。

        锚表是整个查询中最核心的表，通常 FROM 子句的第一个表。
        正确选择锚表对后续的 JOIN 路径规划至关重要。
        """
        self.anchor_table = name

    def add_candidate_table(self, name: str, **kwargs) -> None:
        """添加一个候选表。

        候选表是可能参与查询的表，但还不确定是否一定会用到。
        kwargs 可以传递额外信息（如相关性评分、涉及的列等）。
        """
        entry = {"name": name, **kwargs}
        self.candidate_tables.append(entry)

    def add_join_path(
        self,
        from_table: str,
        to_table: str,
        on: list,
        confidence: str = "medium",
    ) -> None:
        """添加一条 JOIN 路径。

        参数
        ----
        from_table: 源表名。
        to_table: 目标表名。
        on: JOIN 条件列表。
        confidence: 置信度 —— "high"、"medium"、"low"。

        例如：add_join_path("orders", "customers", ["orders.customer_id = customers.id"])
        表示 orders 表和 customers 表可以通过 customer_id 关联。
        """
        self.join_paths.append({
            "from": from_table,
            "to": to_table,
            "on": on,
            "confidence": confidence,
        })

    def add_rejected_join_path(
        self, from_table: str, to_table: str, reason: str
    ) -> None:
        """添加一条被拒绝的 JOIN 路径及拒绝原因。

        为什么需要记录被拒绝的路径？
        因为 Agent 可能会多次考虑同一个 JOIN 组合。
        记录拒绝原因可以避免重复思考，节省 Token（上下文窗口）。
        """
        self.rejected_joins.append({
            "from": from_table,
            "to": to_table,
            "reason": reason,
        })

    def is_join_rejected(self, from_table: str, to_table: str) -> bool:
        """检查两个表之间的 JOIN 是否已被拒绝。

        在尝试新的 JOIN 之前调用此方法，避免重复已知无效的路径。
        """
        for rj in self.rejected_joins:
            if rj["from"] == from_table and rj["to"] == to_table:
                return True
        return False

    def set_time_field(
        self,
        field: str,
        table: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> None:
        """设置时间字段。

        时间字段用于处理涉及时间范围的查询（如"上个月"、"今年初至今"）。
        例如：set_time_field("created_at", "orders", "订单创建时间")
        表示 "orders" 表的 "created_at" 字段记录了订单创建时间，
        可用于按时间筛选。

        设置后 confirmed=True，表示已确认。
        """
        self.time_field = {
            "field": field,
            "table": table,
            "rationale": rationale,
            "confirmed": True,
        }

    def add_dangerous_field(self, field: str, warning: str) -> None:
        """添加一个危险字段及其警告信息。

        什么是危险字段？
        - NULL 比例极高的字段（使用时要 COALESCE）
        - 类型模糊的字段（如 TEXT 存数字）
        - 命名容易误解的字段
        - 有特殊业务含义的字段

        记录危险字段可以让 SQL Builder 在生成查询时避开常见的坑。
        """
        self.dangerous_fields.append({"field": field, "warning": warning})

    def to_dict(self) -> dict:
        """将模式内存序列化为字典。"""
        return {
            "anchor_table": self.anchor_table,
            "candidate_tables": self.candidate_tables,
            "join_paths": self.join_paths,
            "rejected_joins": self.rejected_joins,
            "time_field": self.time_field,
            "dangerous_fields": self.dangerous_fields,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaMemory":
        """从字典恢复模式内存（反序列化）。"""
        sm = cls()
        sm.anchor_table = d.get("anchor_table")
        sm.candidate_tables = d.get("candidate_tables", [])
        sm.join_paths = d.get("join_paths", [])
        sm.rejected_joins = d.get("rejected_joins", [])
        sm.time_field = d.get(
            "time_field",
            {"field": None, "table": None, "rationale": None, "confirmed": False},
        )
        sm.dangerous_fields = d.get("dangerous_fields", [])
        return sm


class QueryMemory:
    """查询内存 —— 存储 SQL 生成、验证和修复过程中的状态。

    初学者理解：
    -----------
    QueryMemory 就像开发者的调试笔记 —— 它记录了：
    - 尝试过的 SQL 写法（variants）
    - 每种写法通过了哪些检查（validation）
    - 哪些写法执行失败了，原因是什么（failed_attempts）
    - 当前对你最终方案的信心有多高（confidence）

    这些信息帮助 Refiner（优化器）决定下一步如何修复问题。
    """

    def __init__(self):
        # SQL 变体列表：尝试过的各种 SQL 写法
        # 每个元素格式：{"sql": "SELECT ...", "status": "draft"}
        self.sql_variants: list[dict] = []

        # 验证结果列表：对 SQL 进行各种检查的记录
        # 每个元素格式：
        #   {"sql_hash": "...", "check_type": "syntax", "passed": True, "detail": "..."}
        self.validation_results: list[dict] = []

        # 失败尝试列表：执行失败的 SQL 及其错误信息
        # 每个元素格式：
        #   {"sql": "SELECT ...", "error": "near ... syntax error", "failure_type": "syntax"}
        self.failed_attempts: list[dict] = []

        # 内部置信度（私有属性，通过属性方法访问）
        self._confidence: Optional[float] = None

    def add_sql_variant(self, sql: str, status: str = "draft") -> None:
        """添加一个 SQL 变体。

        SQL 变体是同一问题的不同 SQL 写法。
        例如，一个查询可以用 JOIN 或子查询实现，它们就是两个变体。

        status 状态：
        - "draft"：草稿，尚未验证
        - "validated"：已验证通过
        - "failed"：验证失败
        """
        self.sql_variants.append({"sql": sql, "status": status})

    def add_validation_result(
        self, sql_hash: str, check_type: str, passed: bool, detail: str
    ) -> None:
        """添加一条验证结果。

        验证类型（check_type）示例：
        - "syntax"：SQL 语法检查
        - "semantics"：语义检查（表/列是否存在）
        - "join_path"：JOIN 路径是否有效
        - "type_match"：类型匹配检查
        """
        self.validation_results.append({
            "sql_hash": sql_hash,
            "check_type": check_type,
            "passed": passed,
            "detail": detail,
        })

    def add_failed_attempt(
        self, sql: str, error: str, failure_type: str
    ) -> None:
        """记录一次失败的 SQL 执行尝试。

        失败类型（failure_type）示例：
        - "syntax"：语法错误
        - "no_such_table"：表不存在
        - "no_such_column"：列不存在
        - "join_no_overlap"：JOIN 没有匹配的行
        - "type_mismatch"：类型不匹配

        记录失败原因可以帮助 Refiner 更精确地修复问题。
        """
        self.failed_attempts.append({
            "sql": sql,
            "error": error,
            "failure_type": failure_type,
        })

    def has_tried_join(self, table_a: str, table_b: str) -> bool:
        """检查是否已经尝试过两个表之间的 JOIN 并失败了。

        这个检查可以防止 Refiner 反复尝试同一个失败的 JOIN 策略。
        它通过搜索 failed_attempts 来实现：
        1. 过滤出 failure_type == "join_no_overlap" 的记录
        2. 检查失败的 SQL 中是否同时包含两个表名

        注意：这是基于文本的粗略检查，不是精确的 AST 分析。
        """
        for fa in self.failed_attempts:
            if (
                fa["failure_type"] == "join_no_overlap"
                and table_a in fa["sql"]
                and table_b in fa["sql"]
            ):
                return True
        return False

    @property
    def latest_confidence(self) -> Optional[float]:
        """获取最新的置信度分数。

        置信度（confidence）表示对当前 SQL 方案正确性的信心程度。
        取值范围 0.0 ~ 1.0，1.0 表示完全确定。
        """
        return self._confidence

    def update_confidence(self, score: float) -> None:
        """更新置信度分数。

        置信度通常由验证过程自动计算：
        - 通过的所有检查越多，置信度越高
        - 失败的尝试越少，置信度越高
        """
        self._confidence = score

    @property
    def has_errors(self) -> bool:
        """检查是否有任何失败尝试记录。

        如果有失败的尝试，说明当前的 SQL 方案还存在问题，
        需要 Refiner 继续修复。
        """
        return len(self.failed_attempts) > 0

    def to_dict(self) -> dict:
        """将查询内存序列化为字典。"""
        return {
            "sql_variants": self.sql_variants,
            "validation_results": self.validation_results,
            "failed_attempts": self.failed_attempts,
            "confidence": self._confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueryMemory":
        """从字典恢复查询内存（反序列化）。"""
        qm = cls()
        qm.sql_variants = d.get("sql_variants", [])
        qm.validation_results = d.get("validation_results", [])
        qm.failed_attempts = d.get("failed_attempts", [])
        qm._confidence = d.get("confidence")
        return qm


class SharedMemory:
    """共享内存顶层容器 —— 聚合所有三种内存类型。

    架构概览
    ========
    SharedMemory 是"门面模式"（Facade Pattern）的典型应用。
    它提供统一的入口，背后隐藏了三层内存的复杂性。

    ```
    SharedMemory
    ├── session: SessionMemory   ← 当前问题的临时状态
    ├── schema: SchemaMemory     ← 数据库模式相关信息
    └── query: QueryMemory       ← SQL 生成与验证信息
    ```

    各 Agent 如何与共享内存交互？
    =============================
    1. SchemaLinker（模式链接器）
       - 写入：SchemaMemory（填写候选表、JOIN 路径）
       - 写入：SessionMemory（记录分析过程中的假设和歧义）

    2. Planner（规划器）
       - 读取：SchemaMemory（了解候选表和 JOIN 路径）
       - 写入：SessionMemory（记录规划决策）

    3. Builder（构建器）
       - 读取：SchemaMemory（读取锚表、时间字段等）
       - 写入：QueryMemory（添加 SQL 变体）

    4. Refiner（优化器）
       - 读取：QueryMemory（了解失败的尝试和验证结果）
       - 写入：QueryMemory（添加修复后的变体和验证结果）

    5. KnowledgeKeeper（知识维护器）
       - 读取：SessionMemory（了解当前问题上下文）
       - 写入：SessionMemory（注入相关知识）

    快照机制（Snapshot）
    ====================
    支持将整个共享内存序列化为字典（snapshot），
    以及从字典恢复（from_snapshot）。
    这在以下场景中很有用：
      - 将内存状态保存到日志中供调试
      - 在不同处理阶段之间传递状态快照
      - 实现"时光回溯"（回退到之前的内存状态）
    """

    def __init__(self):
        """初始化共享内存，包含三层子内存。

        初始状态全部为空：
        - session: 空的 SessionMemory
        - schema: 空的 SchemaMemory
        - query: 空的 QueryMemory
        """
        self.session = SessionMemory()
        self.schema = SchemaMemory()
        self.query = QueryMemory()

    def reset(self) -> None:
        """重置所有内存（为新的问题做准备）。

        在每个新的自然语言问题开始处理时调用此方法。
        这会清空三层内存中的所有数据。
        """
        self.session = SessionMemory()
        self.schema = SchemaMemory()
        self.query = QueryMemory()

    def snapshot(self) -> dict:
        """对整个共享内存拍快照（序列化）。

        返回
        ----
        包含三层内存数据的字典。
        可以通过 from_snapshot 恢复。
        """
        return {
            "session": self.session.to_dict(),
            "schema": self.schema.to_dict(),
            "query": self.query.to_dict(),
        }

    @classmethod
    def from_snapshot(cls, d: dict) -> "SharedMemory":
        """从快照恢复共享内存（反序列化）。

        参数
        ----
        d: 由 snapshot() 方法生成的字典。

        用法示例
        --------
        >>> mem = SharedMemory()
        >>> # ... 做一些操作 ...
        >>> snap = mem.snapshot()
        >>> # ... 后来想恢复 ...
        >>> mem = SharedMemory.from_snapshot(snap)
        """
        mem = cls()
        mem.session = SessionMemory.from_dict(d.get("session", {}))
        mem.schema = SchemaMemory.from_dict(d.get("schema", {}))
        mem.query = QueryMemory.from_dict(d.get("query", {}))
        return mem
