# ============================================================================
# knowledge_keeper_generic.py — 通用领域的知识检索层
# ============================================================================
#
# 设计特点：
#   1. 业务规则从 Schema 的 FK/PK 注解中自动提取，而非硬编码
#   2. 编码模式来自通用 SQL 最佳实践（JOIN、INTERSECT、EXCEPT、GROUP BY 等）
#   3. 时间字段通过启发式（heuristic）匹配列名中的 date/time 关键词
#
# 适用场景：
#   BIRD、Spider、以及任意 SQLite 数据库的 NL2SQL 场景。
#
# 设计模式：
#   Facade（外观模式），对外提供简单的 retrieve_xxx 和 inject_xxx 接口，
#   屏蔽内部知识检索的复杂度。
# ============================================================================

# === 标准库与项目内部依赖 ===
from agent_team.embedding_index import EmbeddingIndex
from agent_team.knowledge_graph_builder import KnowledgeGraphBuilder


class KnowledgeKeeper:
    # ========================================================================
    # 类文档（Class Docstring）
    # ========================================================================
    """通用领域的知识检索门面（Facade）—— 适用于任意 SQLite 数据库。

    本类**不是** LLM Agent，而是一个**检索层**（Retrieval Layer），
    它将 Schema 中的元数据（表名、列名、外键、主键）组织成知识索引，
    供 Planner、Builder、Refiner 在各自的阶段使用。

    设计要点：
      - 没有硬编码的游戏业务规则（DAU/留存/新进/回流/付费）
      - 没有游戏数仓特有的编码模式（cbitmap/platid/device_type）
      - 编码模式基于通用 SQL 最佳实践（INTERSECT、EXCEPT、COUNT DISTINCT 等）
      - 业务规则通过分析 FK（外键）/PK（主键）注解自动生成
      - 时间字段通过列名关键词（date、time、year、month）启发式检测
    """

    # ========================================================================
    # CODING_PATTERNS（通用编码模式字典）
    # ========================================================================
    # 这些编码模式适用于**任何** SQL 数据库，不限于游戏数仓。
    # 每个模式包含一个 SQL 写法的最佳实践说明。
    #
    # 通用版只有 6 个模式，全部是跨领域的 SQL 通用知识
    CODING_PATTERNS = {
        # join_fk：外键连接的最佳实践
        "join_fk": "使用 Schema 中标注的 FK 关系进行 JOIN。优先使用显式外键，次选同名列推断。",
        # count_distinct：唯一计数的正确写法
        "count_distinct": "统计唯一实体数时使用 COUNT(DISTINCT column)。去重前确认列是否可能为 NULL。",
        # intersect_pattern：多条件同时满足
        "intersect_pattern": "需要同时满足多个条件时：使用 INTERSECT 或 GROUP BY + HAVING COUNT(DISTINCT CASE WHEN) = N。",
        # not_exists：排除型查询
        "not_exists": "排除型查询使用 NOT EXISTS (SELECT 1 FROM ...) 或 LEFT JOIN + IS NULL。",
        # group_by：GROUP BY 的注意事项
        "group_by": "GROUP BY 必须包含 SELECT 中所有非聚合列，否则 SQLite 结果不确定。",
        # alias：别名规范（与 game 版一致）
        "alias": "SQL 别名使用简短英文标识符。禁止中文别名。",
    }

    def __init__(self, schema: list[dict]):
        """初始化通用版 KnowledgeKeeper。

        在 Phase 0（准备阶段）预先构建知识索引。
        区别是不加载案例库（generic 场景通常没有历史案例）。

        参数:
            schema: Schema JSON 解析出的列表，每个元素是一张表的定义
        """
        self.schema = schema
        # 建立"表名 -> 表定义"的快速字典索引
        self._table_index = {t["table_name"]: t for t in schema}

        # ---------------------------------------------------------------
        # 构建嵌入索引（EmbeddingIndex）
        # ---------------------------------------------------------------
        # 不传入 business_terms 参数，
        # 因为通用场景没有预定义的领域术语词典
        self._embedding_index = EmbeddingIndex()
        self._embedding_index.build_from_schema(schema)

        # ---------------------------------------------------------------
        # 构建知识图谱（KnowledgeGraph）
        # ---------------------------------------------------------------
        # 知识图谱的构建逻辑：
        #   - 创建表节点（解析表名获取 layer/gamecode/suffix）
        #   - 创建列节点（每列一个节点，连接到对应的表节点）
        #   - 发现外键边（同名列出现在两张以上表时猜测为 FK）
        #   - 标注语义信息（多值字段、危险字段等）
        #   - 提取业务概念（从 spider_agent.txt 规则中）
        builder = KnowledgeGraphBuilder(schema)
        self._knowledge_graph = builder.build()

    # ========================================================================
    # 属性（Properties）
    # ========================================================================

    @property
    def embedding_index(self):
        """返回嵌入索引，用于语义相似度搜索。"""
        return self._embedding_index

    @property
    def knowledge_graph(self):
        """返回知识图谱，用于关系推理和表连接发现。"""
        return self._knowledge_graph

    # ===================================================================
    # 检索方法（Retrieval Methods）
    # ===================================================================

    def retrieve_for_planner(self, question: str, relevant_schema: dict) -> dict:
        """为 Planner 阶段检索规划知识。

        通用版的 retrieve_for_planner 有两个主要特点：
          1. 业务规则不是从硬编码的字典中匹配，而是从 Schema 的
             FK/PK 注解中动态提取
          2. 不返回相似案例（generic 场景通常没有案例库）和危险字段警告

        参数:
            question: 用户的自然语言问题
            relevant_schema: Schema Analyzer 分析出的相关表结构

        返回:
            一个字典，包含:
              - business_rules: 从 Schema 注解中提取的 FK/PK 规则
              - time_field_guidance: 时间字段使用建议
              - similar_cases: 空列表（通用场景暂无案例库）
              - dangerous_fields_warning: 空字符串（通用场景暂不标记危险字段）
        """
        # 从 Schema 的 FK/PK 注释中提取业务规则
        business_rules = self._extract_schema_rules(relevant_schema)
        # 构建时间字段使用建议
        time_guidance = self._build_time_guidance(relevant_schema)

        return {
            "business_rules": business_rules,
            "time_field_guidance": time_guidance,
            # 以下两个字段在通用版中返回空值
            # 如果以后需要，可以扩展支持
            "similar_cases": [],
            "dangerous_fields_warning": "",
        }

    def retrieve_for_builder(self, plan: dict, relevant_schema: dict, question: str = "") -> dict:
        """为 Builder 阶段检索编码知识。

        这是根据用户问题的自然语言特征来推断需要的编码模式的方法。

        比如：
          - 问题中包含 "both" / "同时" —— 需要 INTERSECT 模式
          - 问题中包含 "most" / "最大" —— 需要 Top-N 模式
          - 问题中包含 "each" / "每" —— 需要聚合模式

        参数:
            plan: Planner 生成的执行计划
            relevant_schema: Schema Analyzer 分析出的相关表结构
            question: 用户的原始问题文本（用于关键词检测）

        返回:
            一个字典，包含:
              - coding_patterns: 相关编码模式的文本
              - ratio_format_rule: 通用场景通常为空字符串
        """
        patterns = []

        # 将问题转为小写，用于不区分大小写的关键词匹配
        question_lower = (question or "").lower()

        # ---------------------------------------------------------------
        # INTERSECT / 交集模式检测
        # ---------------------------------------------------------------
        # 当问题需要"同时满足多个条件"时（如"既买了A又买了B的用户"），
        # 需要提醒 Builder 使用 INTERSECT 或 GROUP BY + HAVING 的组合方式
        if any(kw in question_lower for kw in ["both", "also", "who have", "who has", "既", "又", "同时"]):
            patterns.append(
                "INTERSECT模式：必须用顶层 INTERSECT 连接两个独立查询，"
                "不要用 EXISTS 或 GROUP BY + HAVING 替代。"
            )

        # ---------------------------------------------------------------
        # EXCEPT / 排除模式检测
        # ---------------------------------------------------------------
        # 当问题需要"排除某些条件"时（如"买了A但没买B的用户"），
        # 提醒 Builder 使用 EXCEPT 或 NOT EXISTS
        if any(kw in question_lower for kw in ["but no", "without", "not", "但非", "排除"]):
            patterns.append("排除模式：用 EXCEPT 或 NOT EXISTS (SELECT 1 FROM ...)。")

        # ---------------------------------------------------------------
        # Top-N / 排序模式检测
        # ---------------------------------------------------------------
        # 当问题涉及"最大/最小/最高/最低"等程度词时，
        # 提醒 Builder 使用 ORDER BY + LIMIT 而非嵌套子查询
        if any(kw in question_lower for kw in [
            "most", "least", "largest", "smallest", "greatest",
            "top", "highest", "lowest",
            "最大", "最小", "最高", "最低", "最多", "最少",
        ]):
            patterns.append("Top-N模式：用 ORDER BY + LIMIT 1/N，不要嵌套子查询+MAX。")

        # ---------------------------------------------------------------
        # 聚合模式检测
        # ---------------------------------------------------------------
        # 当问题涉及"每个"、"各"、"统计"、"平均"等分组词汇时，
        # 提醒 Builder 注意 GROUP BY 和 COUNT(DISTINCT) 的正确使用
        if any(kw in question_lower for kw in [
            "each", "per", "every", "count", "total", "average",
            "每", "各", "统计", "多少",
        ]):
            patterns.append(
                "聚合模式：确认 GROUP BY 包含所有非聚合列。"
                "检查是否需要 COUNT(DISTINCT)。"
            )

        # ---------------------------------------------------------------
        # 始终注入的基本提醒
        # ---------------------------------------------------------------
        # 无论什么场景，这两条提醒都是有益的
        patterns.append("简单优先：先确认所需列是否都在单张表中，不要无必要地 JOIN。")
        patterns.append("列名精确匹配DDL：大小写和拼写必须与 DDL 完全一致。")

        return {
            "coding_patterns": "\n\n".join(patterns),
            "ratio_format_rule": "",  # 通用场景没有特定的比例格式规范
            # 从知识图谱中提取的 Schema 结构信息
            "join_path_info": self._get_join_path_info(relevant_schema),
            "dangerous_fields_warning": self._get_dangerous_warnings(relevant_schema),
        }

    def _get_join_path_info(self, relevant_schema: dict) -> str:
        """从知识图谱中提取候选表之间的 JOIN 路径信息。

        当 SchemaLinker 已经发现候选表之间的潜在 FK 关系时，
        将这些 JOIN 路径放入 Builder 的 prompt，让 LLM 能
        更准确地写出多表连接。

        参数:
            relevant_schema: SchemaLinker 的分析结果（含 join_paths 字段）

        返回:
            格式化的 JOIN 路径提示文本，若无则返回空字符串
        """
        join_paths = relevant_schema.get("join_paths", [])
        if not join_paths:
            return ""

        lines = ["已知 JOIN 路径（候选表之间）:"]
        for jp in join_paths:
            from_t = jp.get("from", "?")
            to_t = jp.get("to", "?")
            on_cols = jp.get("on", [])
            confidence = jp.get("confidence", "unknown")
            lines.append(
                f"  {from_t} ↔ {to_t} ON {', '.join(on_cols)} "
                f"(置信度: {confidence})"
            )
        return "\n".join(lines)

    def _get_dangerous_warnings(self, relevant_schema: dict) -> str:
        """从知识图谱中提取候选表中的危险字段警告。

        危险字段是那些语义依赖于上下文（如 platid）的预留字段，
        Builder 不应该在不知道上下文的情况下随意使用它们。

        参数:
            relevant_schema: SchemaLinker 的分析结果

        返回:
            格式化的危险字段警告文本，若无则返回空字符串
        """
        warnings = []
        candidate_table_names = {
            t.get("name", "") for t in relevant_schema.get("candidate_tables", [])
        }

        for node, attrs in self._knowledge_graph.nodes(data=True):
            if attrs.get("is_dangerous"):
                # 节点 ID 格式: "table.column"
                parts = node.split(".")
                if len(parts) >= 2:
                    table_name = parts[0]
                    col_name = parts[1]
                    if table_name in candidate_table_names:
                        warnings.append(
                            f"⚠ {table_name}.{col_name}: "
                            f"{attrs.get('warning', '预留字段，需理解上下文才能正确使用')}"
                        )

        return "\n".join(warnings) if warnings else ""

    def retrieve_for_refiner(self, sql: str, validation_report: dict) -> dict:
        """为 Refiner 阶段检索修复知识。

        通用版的修复策略根据 Validator 的 L1/L2/L3
        错误报告来匹配修复策略。区别是匹配的关键词更通用（如检查
        "table"、"column"、"syntax"等英文关键词）。

        参数:
            sql: 待修复的 SQL 语句
            validation_report: Validator 产生的验证报告

        返回:
            一个字典，包含:
              - fix_strategies: 修复策略文本
        """
        strategies = []

        # 收集所有层级的错误描述
        all_failures = []
        for layer_key in ["l1", "l2", "l3"]:
            layer = validation_report.get(layer_key, {})
            for f in layer.get("failures", []):
                all_failures.append(f.get("description", ""))

        # 针对每种错误类型匹配修复策略
        for desc in all_failures:
            # 中文别名错误
            if "别名" in desc or "中文" in desc:
                strategies.append("L1修复：将中文别名替换为英文标识符。")
            # 表名错误（支持中英文关键词）
            if "表" in desc or "table" in desc.lower():
                strategies.append("L2修复：检查表名拼写，确认该表是否存在。检查 FK 关系是否有遗漏。")
            # 列名错误
            if "列" in desc or "column" in desc.lower():
                strategies.append("L2修复：检查列名拼写，确认该列存在于候选表中。")
            # 语法错误
            if "syntax" in desc.lower() or "语法" in desc:
                strategies.append("L1修复：检查 SQL 语法——括号匹配、关键字拼写、引号配对。")

        # 没有匹配到任何策略时，给出通用修复建议
        if not strategies:
            strategies.append("通用修复：检查 SQL 是否符合 SQLite 语法和 Schema 定义。")

        return {"fix_strategies": "\n".join(strategies)}

    # ===================================================================
    # 内部辅助方法（Internal Helpers）
    # ===================================================================

    def _extract_schema_rules(self, relevant_schema: dict) -> str:
        """从 Schema 的 FK/PK 注解中提取业务规则。

        这是通用版获取"业务规则"的方式。通用版没有
        硬编码的业务术语字典，而是从表字段的描述（description）中
        自动发现 PK（主键）和 FK（外键）标注。

        工作流程：
          1. 遍历候选表的所有列
          2. 检查列描述中是否包含 "PK" 或 "FK" 字样
          3. 汇总主键列和外键列信息
          4. 如果有 Schema Analyzer 提供的 JOIN 路径，也一并列出

        参数:
            relevant_schema: Schema Analyzer 的分析结果

        返回:
            格式化后的 Schema 规则文本
        """
        rules = []

        # 遍历候选表，提取 FK/PK 信息
        for table in relevant_schema.get("candidate_tables", []):
            table_name = table.get("name", "")
            pk_cols = []
            fk_cols = []
            for col in table.get("relevant_columns", []):
                desc = col.get("description", "")
                # 检查列描述中是否包含 "PK"（主键）标记
                if "PK" in desc:
                    pk_cols.append(col["name"])
                # 检查列描述中是否包含 "FK"（外键）标记，附上完整描述
                if "FK" in desc:
                    fk_cols.append(f"{col['name']}({desc})")

            # 如果该表有主键或外键，加入规则列表
            if pk_cols:
                rules.append(f"表 {table_name} 主键: {', '.join(pk_cols)}")
            if fk_cols:
                rules.append(f"表 {table_name} 外键: {', '.join(fk_cols)}")

        # 如果有 Schema Analyzer 预计算的 JOIN 路径，也一并展示
        join_paths = relevant_schema.get("join_paths", [])
        if join_paths:
            rules.append("\n可用 JOIN 路径:")
            for jp in join_paths:
                rules.append(
                    f"  {jp['from']} ↔ {jp['to']} ON {', '.join(jp.get('on', []))} "
                    f"(confidence: {jp.get('confidence', 'unknown')})"
                )

        return "\n".join(rules) if rules else "无额外 Schema 规则。"

    def _build_time_guidance(self, relevant_schema: dict) -> str:
        """根据 Schema 构建时间字段使用建议。

        通用版没有 DWS/DWD 分层概念，因此时间字段的
        检测采用启发式方法——检查列名中是否包含 date / time / year / month
        等关键词。

        如果 Schema Analyzer 已经明确标注了主时间字段，直接使用该标注。
        否则做自动检测；如果什么都没找到，给出一个中性的提示信息。

        参数:
            relevant_schema: Schema Analyzer 的分析结果

        返回:
            时间字段使用建议文本
        """
        tf = relevant_schema.get("time_fields", {})
        primary = tf.get("primary", {})

        # 优先使用 Schema Analyzer 标注的主时间字段
        if primary.get("field"):
            return (
                f"主时间字段：{primary.get('table', 'unknown')}.{primary['field']} "
                f"(类型: {primary.get('type', 'unknown')})"
            )

        # ---------------------------------------------------------------
        # 启发式检测：从列名中识别可能的时间字段
        # ---------------------------------------------------------------
        # 遍历所有候选表的所有列，检查列名中是否包含常见的时间关键词
        date_cols = []
        for table in relevant_schema.get("candidate_tables", []):
            for col in table.get("relevant_columns", []):
                col_name = col.get("name", "").lower()
                col_type = col.get("type", "").upper()
                # 关键词匹配：date、time、year、month 都是时间字段的常见前缀
                if any(kw in col_name for kw in ["date", "time", "year", "month"]):
                    date_cols.append(f"{table['name']}.{col['name']} ({col_type})")

        if date_cols:
            return "检测到可能的时间字段: " + ", ".join(date_cols)

        # 兜底返回
        return "未检测到明确的时间字段。如果问题不涉及时间过滤，可忽略。"

    # ===================================================================
    # 自动注入方法（Auto-Injection）
    # ===================================================================
    #
    # 以下两个方法会被 Orchestrator 自动调用，将知识注入到
    # Planner 和 Builder 的 Prompt 中。
    # 接口签名保持一致，便于 Orchestrator 统一调用。

    def inject_into_planner_prompt(self, question: str, relevant_schema: dict) -> str:
        """构建注入到 Planner Prompt 的知识片段。

        参数:
            question: 用户的自然语言问题
            relevant_schema: Schema Analyzer 的分析结果

        返回:
            一段 Markdown 格式的知识文本
        """
        k = self.retrieve_for_planner(question, relevant_schema)

        injection = "## 相关知识（自动注入）\n\n"
        if k["business_rules"]:
            injection += f"### Schema 规则\n{k['business_rules']}\n\n"
        if k["time_field_guidance"]:
            injection += f"### 时间字段建议\n{k['time_field_guidance']}\n\n"

        return injection

    def inject_into_builder_prompt(self, plan: dict, relevant_schema: dict, question: str = "") -> str:
        """构建注入到 Builder Prompt 的知识片段。

        注意：通用版的 inject_into_builder_prompt 多了一个 question 参数，
        这是为了支持"根据问题文本检测编码模式"的特性。

        参数:
            plan: Planner 生成的执行计划
            relevant_schema: Schema Analyzer 的分析结果
            question: 用户的原始问题文本

        返回:
            一段 Markdown 格式的编码规范文本
        """
        k = self.retrieve_for_builder(plan, relevant_schema, question=question)

        injection = "## 编码规范与 Schema 知识（自动注入）\n\n"

        # 1. SQL 编码模式（基于问题关键词推断）
        if k["coding_patterns"]:
            injection += f"### SQL 编码模式建议\n{k['coding_patterns']}\n\n"

        # 2. JOIN 路径信息（从知识图谱提取的 FK 关系）
        if k.get("join_path_info"):
            injection += f"### 表连接路径\n{k['join_path_info']}\n\n"

        # 3. 危险字段警告（从知识图谱提取的 is_dangerous 标记）
        if k.get("dangerous_fields_warning"):
            injection += f"### 危险字段提醒\n{k['dangerous_fields_warning']}\n\n"

        return injection
