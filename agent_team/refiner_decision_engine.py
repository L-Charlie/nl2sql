# ============================================================================
# refiner_decision_engine.py —— 零 LLM 的修复决策引擎
# ============================================================================
#
# 【什么是"零 LLM 决策引擎"？】
# 通常，要判断一条 SQL 出了什么错、该怎么修，你需要调一次大语言模型（LLM）
# 来分析错误信息。但这里的设计思路完全不同：
#
#   ❌ 传统方式：报错 → 调 LLM 分析 → LLM 返回修复方案（耗时、昂贵、不稳定）
#   ✅ 本引擎：   报错 → 规则引擎匹配 → 确定修复级别（快速、低成本、确定性强）
#
# 这个引擎完全基于规则（rule-based），不调用任何 API。它通过正则表达式匹配
# 错误描述中的关键词，来判断错误的类型和严重程度，然后决定该由哪个层级的
# 修复流程来处理。
#
# 【引擎的工作流程】
#   Refiner（精炼器）验证 SQL 后，如果发现错误，会将错误描述列表交给本引擎。
#   引擎会：
#   1. 对每个错误描述进行分类 → 确定是哪种 FailureCategory
#   2. 根据所有错误中的"最高严重级别" → 确定初始 RepairLevel
#   3. 检查该级别已经尝试了多少次 → 如果超限则升级（escalation）
#   4. 返回 (RepairLevel, 策略描述, 是否升级)
#
# 【修复级别说明】
#   级别     含义         做什么
#   ──────────────────────────────────────────────
#   L1       表面修复     直接在 SQL 文本上做替换（改别名、改引号、加排序）
#   L2       语义修复     重新检索 Schema 重新生成 SQL（保留查询计划框架）
#   L3       计划重生成   整个查询计划推倒重来，重新规划查询思路
#   TERMINATE 终止       放弃修复，返回当前最佳结果
#
# 【升级规则（Escalation）】
#   L1 尝试 ≥ 3 次仍失败 → 升级到 L2
#   L2 尝试 ≥ 2 次仍失败 → 升级到 L3
#   L3 尝试 ≥ 2 次仍失败 → 终止（TERMINATE），返回尽力而为的结果
# ============================================================================

from enum import Enum, auto       # 枚举：用于定义分类和级别的常量
from dataclasses import dataclass # 数据类：用于结构化存储修复决策信息


# ============================================================================
# 枚举：FailureCategory —— 失败分类
# ============================================================================
# 这个枚举定义了 SQL 验证失败的所有可能类型。
# 每一类对应一组特定的错误模式和修复策略。
# ============================================================================
class FailureCategory(Enum):
    """
    SQL 验证失败的类型分类。

    根据错误性质分为四个类别，从表面格式问题到深层业务逻辑问题：
    - SYNTAX_FORMAT （语法格式问题） → 最轻量，对应 L1 修复
    - SCHEMA_ERROR  （Schema 错误）   → 中等，对应 L2 修复
    - DATA_COVERAGE （数据覆盖问题）  → 中等，对应 L2 修复
    - BUSINESS_LOGIC（业务逻辑问题）  → 最严重，对应 L3 修复
    """
    # L1 级别错误：SQL 语法格式上的表面问题
    # 例如：别名的使用方式不对、日期格式写错了、引号不匹配、缺少 ORDER BY 等
    SYNTAX_FORMAT = "syntax_format"

    # L2 级别错误：数据库 Schema 相关的问题
    # 例如：使用了错误的表名或列名、缺少 JOIN 的键
    SCHEMA_ERROR = "schema_error"

    # L2 级别错误：数据覆盖范围的问题
    # 例如：查询结果为零行、时间字段的数据覆盖不完整、统计值应为正数却为 0
    DATA_COVERAGE = "data_coverage"

    # L3 级别错误：业务逻辑层面的问题
    # 例如：比例字段计算超越了合理范围、错误的业务主体锚定、指标定义与需求不符
    BUSINESS_LOGIC = "business_logic"


# ============================================================================
# 枚举：RepairLevel —— 修复级别
# ============================================================================
# 这个枚举定义了四个修复层级，从最轻的 SQL 文本修补到最重的计划重生成。
# 修复级别越高，代价越大（需要更多 LLM 调用和时间），但修复能力也更强。
# ============================================================================
class RepairLevel(Enum):
    """
    修复操作的实施级别。

    级别数值越大，修复的"力度"越大（但也越耗时）：
    L1 = 1  → 表面修复：直接在 SQL 字符串上做文本替换
    L2 = 2  → 语义修复：重新检索 Schema 并重建 SQL（保留原有的查询计划）
    L3 = 3  → 计划重生成：重新规划整个查询（最彻底的修复方式）
    TERMINATE = 0 → 终止修复：一切尝试都失败了，返回当前最好的结果
    """
    L1 = 1       # 表面修复：直接在 SQL 文本上做修正（不改动查询逻辑）
    L2 = 2       # 重新检索 Schema + 重新构建 SQL（保留原始查询计划框架）
    L3 = 3       # 重新规划整个查询（推倒重来，从 Plan 阶段开始）
    TERMINATE = 0  # 终止修复：放弃继续尝试，返回尽力而为的结果


# ============================================================================
# 类：RefinerDecisionEngine —— 零 LLM 修复决策引擎
# ============================================================================
# 这是整个 NL2SQL 系统中"成本最低但最关键"的决策点。
# 它不做任何 API 调用，全都靠正则表达式 + 状态机逻辑来判断。
#
# 设计哲学：
# - 用规则代替模型：把 spider_agent.txt 中的错误分类规则编码为关键词列表
# - 用升级机制兜底：低级别的修复如果反复失败，自动升级到更彻底的修复
# - 用尝试次数控制无限循环：每个级别都有最大尝试次数，防止死循环
# ============================================================================
class RefinerDecisionEngine:
    """零 LLM 的修复决策引擎 —— 将验证失败映射到修复级别的规则引擎。

    本引擎实现了 spider_agent.txt 中定义的错误分类和升级规则：
    - ADAPTIVE_VALIDATION_FRAMEWORK（自适应验证框架，spider_agent.txt 第 708-1185 行）
      — 定义了错误如何分类、每个类别对应什么修复策略
    - CONFIDENCE_VALIDATION（置信度验证，第 1187-1287 行）
      — 定义了如何评估修复后的 SQL 是否"可信"
    """

    # ------------------------------------------------------------------
    # 关键词 → FailureCategory 映射（L1 / L2 / L3 模式匹配列表）
    # ------------------------------------------------------------------
    # 这些列表是从 spider_agent.txt 中提取的关键词。
    # 引擎会用 re.search() 将错误描述文本与这些关键词逐一匹配。
    #
    # 【匹配策略】:
    #   1. 先从 L3_PATTERNS 开始匹配（最严重优先）
    #   2. 再匹配 L2_PATTERNS
    #   3. 最后匹配 L1_PATTERNS
    #   4. 如果都不匹配，默认归类为 SCHEMA_ERROR（保守策略）
    #
    # 为什么要"最严重优先"？
    #   因为 L3 的关键词（如"业务指标"）如果出现在错误描述中，
    #   即使也匹配了 L1 的模式，本质上问题更严重。
    #   优先匹配 L3 可以避免"把小病当感冒治"的情况。
    # ------------------------------------------------------------------

    # L1 级别模式：语法格式问题（表面修复可以处理）
    # 匹配关键词包括：
    # - 别名/alias 类：别名问题、中文字段名
    # - 日期函数类：DATE_FORMAT 函数使用、CAST 类型转换
    # - 排序/限制类：缺少 ORDER BY、缺少 LIMIT、缺少排序
    # - 多值/模糊查询类：多值字段使用等值判断、LIKE 语法
    # - 引号/括号类：引号不匹配、括号使用错误
    L1_PATTERNS = [
        "别名", "中文", "alias",
        "日期函数", "DATE_FORMAT", "CAST",
        "缺少 ORDER BY", "缺少 LIMIT", "缺少排序",
        "多值字段.*等值", "LIKE",
        "引号", "quote", "括号",
    ]

    # L2 级别模式：Schema 或数据覆盖问题（需要重新检索 Schema）
    # 匹配关键词包括：
    # - 表不存在/列不存在：表名不在 Schema 中、列名缺失
    # - 数据为空：结果为零行、zero rows
    # - 统计异常：统计字段值为 0、本应大于 0 的值
    # - 执行错误：SQL 执行报错
    # - 时间字段：时间字段覆盖范围有问题
    L2_PATTERNS = [
        "表 '", "不在.*schema", "不存在",
        "列 '", "column.*missing",
        "结果为零行", "zero rows", "零行",
        "统计字段.*值为 0", "应大于 0",
        "执行失败", "execution error",
        "time field", "时间字段",
    ]

    # L3 级别模式：业务逻辑问题（需要重新规划整个查询）
    # 匹配关键词包括：
    # - 比例计算：比例字段超过合理范围、比率超出限制
    # - 值差异：期望值与实际值存在显著分歧
    # - 主体锚定：业务主体的锚定逻辑错误（如"公司"和"子公司"混淆）
    # - 业务指标：业务指标定义与用户需求不符
    # - 时间边界：紧凑时间格式的边界处理异常
    # - 分子/分母：比例计算中分子分母逻辑错误
    # - 列不一致：业务相关的列之间不一致
    L3_PATTERNS = [
        "比例字段.*超过", "ratio.*exceed",
        "值差异", "diverge",
        "主体锚定", "anchor",
        "业务指标", "metric definition",
        "紧凑时间格式.*边界", "boundary",
        "分子分母", "ratio logic",
        "列不一致", "column mismatch",
    ]

    # ------------------------------------------------------------------
    # 最大尝试次数常量
    # ------------------------------------------------------------------
    # 这些常量防止系统在同一种修复级别上陷入无限循环。
    # 每次 Refiner 调用引擎后，如果决定执行修复，修复完成后 SQL 会
    # 被再次验证，验证失败就会再进引擎。这个"尝试次数"就是用来记录
    # 这种循环次数的。
    #
    # 数值含义：
    #   MAX_L1_ATTEMPTS = 3  —— L1 表面修复最多做 3 次
    #                           如果 3 次 L1 都不能解决 → 升级到 L2
    #   MAX_L2_ATTEMPTS = 2  —— L2 语义修复最多做 2 次
    #                           如果 2 次 L2 都不能解决 → 升级到 L3
    #   MAX_L3_ATTEMPTS = 2  —— L3 计划重生成最多做 2 次
    #                           如果 2 次 L3 都不能解决 → 终止修复
    # ------------------------------------------------------------------
    MAX_L1_ATTEMPTS = 3   # L1 表面修复的最大尝试次数
    MAX_L2_ATTEMPTS = 2   # L2 语义修复的最大尝试次数
    MAX_L3_ATTEMPTS = 2   # L3 计划重生成的最大尝试次数

    # ==================================================================
    # 方法：classify_failure —— 对单个验证失败进行分类
    # ==================================================================
    def classify_failure(self, description: str) -> FailureCategory:
        """对单个验证失败描述进行分类，返回对应的 FailureCategory。

        分类策略（从最严重到最轻微的顺序匹配，即 L3 → L2 → L1）：
        1. 先检查是否匹配 L3_PATTERNS → 如果是 → BUSINESS_LOGIC
        2. 再检查是否匹配 L2_PATTERNS → 如果是 → DATA_COVERAGE 或 SCHEMA_ERROR
        3. 最后检查是否匹配 L1_PATTERNS → 如果是 → SYNTAX_FORMAT
        4. 都不匹配 → 默认返回 SCHEMA_ERROR（保守策略：宁高勿低）

        参数:
            description: str —— 验证失败的文本描述（来自 Refiner 的验证结果）

        返回:
            FailureCategory —— 失败分类枚举值
        """
        import re  # 导入正则表达式模块用于关键词匹配

        # 第一步：检查 L3 级别（业务逻辑问题）
        # 优先检查最严重的级别。因为如果错误涉及"业务指标"或"比例逻辑"，
        # 即使它也包含语法问题，本质上是业务逻辑层面的错误。
        for pattern in self.L3_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                return FailureCategory.BUSINESS_LOGIC

        # 第二步：检查 L2 级别（Schema 或数据覆盖问题）
        # L2 内部需要进一步区分 SCHEMA_ERROR 和 DATA_COVERAGE：
        #   - 如果描述包含"零行"、"zero"、"值为 0" → 数据覆盖问题
        #   - 否则 → Schema 错误（表/列不存在等）
        for pattern in self.L2_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                # 根据描述中是否包含"数据为空"相关关键词来进一步细分
                if "零行" in description or "zero" in description.lower() or "值为 0" in description:
                    return FailureCategory.DATA_COVERAGE
                else:
                    return FailureCategory.SCHEMA_ERROR

        # 第三步：检查 L1 级别（语法格式问题）
        for pattern in self.L1_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                return FailureCategory.SYNTAX_FORMAT

        # 第四步：默认分类（没有匹配任何已知模式）
        # 返回 SCHEMA_ERROR 是一个保守的（conservative）选择：
        # 宁可将未知错误归类为较严重的 L2，也不要归为 L1 而"治标不治本"。
        return FailureCategory.SCHEMA_ERROR

    # ==================================================================
    # 方法：determine_repair_level —— 确定修复级别
    # ==================================================================
    def determine_repair_level(
        self,
        failure_descriptions: list[str],  # 验证失败描述的列表（可能有多条错误）
        l1_attempts: int = 0,             # L1 已经尝试的次数
        l2_attempts: int = 0,             # L2 已经尝试的次数
        l3_attempts: int = 0,             # L3 已经尝试的次数
    ) -> tuple[RepairLevel, str, bool]:
        """综合所有验证失败和尝试次数，确定最终修复级别。

        返回三元组:
            (RepairLevel,       策略描述文本,    是否触发升级)
            (修复级别枚举值,  human-readable 字符串,  True/False)

        决策逻辑：
        1. 对每条失败描述进行分类（classify_failure）
        2. 取所有分类中的"最高严重级别"作为初始修复级别
           （例如：一个是 SYNTAX_FORMAT，一个是 BUSINESS_LOGIC → 取 L3）
        3. 检查该级别是否已经尝试超限
           - L1 尝试 ≥ 3 次 → 升级到 L2
           - L2 尝试 ≥ 2 次 → 升级到 L3
           - L3 尝试 ≥ 2 次 → 终止（TERMINATE）

        参数:
            failure_descriptions: list[str] —— Refiner 验证后发现的错误描述列表
            l1_attempts: int —— 本轮对话中 L1 修复已被尝试的次数
            l2_attempts: int —— 本轮对话中 L2 修复已被尝试的次数
            l3_attempts: int —— 本轮对话中 L3 修复已被尝试的次数
        """
        # 如果没有失败描述，说明 SQL 是正确的，不需要修复
        if not failure_descriptions:
            return None, "", False

        # ----- 第一步：对每条失败描述进行分类 -----
        # 使用 classify_failure() 逐条分析错误，得到分类列表
        categories = [self.classify_failure(desc) for desc in failure_descriptions]

        # ----- 第二步：确定最高严重级别作为初始修复级别 -----
        # 多个错误同时存在时，以最严重的那个为准。
        # 例如：同时有"缺少 ORDER BY"（L1）和"业务指标"（L3）→ 按 L3 处理
        if FailureCategory.BUSINESS_LOGIC in categories:
            # 只要有 BUSINESS_LOGIC 错误 → 整个修复级别提升到 L3
            natural_level = RepairLevel.L3
        elif FailureCategory.SCHEMA_ERROR in categories or FailureCategory.DATA_COVERAGE in categories:
            # 有 SCHEMA_ERROR 或 DATA_COVERAGE → 至少是 L2
            natural_level = RepairLevel.L2
        else:
            # 只有 SYNTAX_FORMAT 错误（或没有已知类型的错误）→ L1
            natural_level = RepairLevel.L1

        # ----- 第三步：应用升级规则（escalation rules）-----
        # 即使错误本身只是 L1 级别，但如果 L1 已经尝试了很多次都没修复好，
        # 说明问题可能比表面上更复杂，需要升级到更深入的修复方式。
        escalate = False

        # 升级规则 1：L1 尝试太多次 → 自动升级到 L2
        # 场景：Refiner 已经做了 3 次 L1 表面修复（改引号、加 ORDER BY 等），
        #       但 SQL 还是验证不通过。此时"表面修复"已经不够，需要重新理解 Schema。
        if natural_level == RepairLevel.L1 and l1_attempts >= self.MAX_L1_ATTEMPTS:
            natural_level = RepairLevel.L2
            escalate = True

        # 升级规则 2：L2 尝试太多次 → 自动升级到 L3
        # 场景：Builder 重新检索 Schema、重新构建了 2 次 SQL，但验证仍然失败。
        #       这意味着"查询计划"本身就有问题，需要重新规划。
        if natural_level == RepairLevel.L2 and l2_attempts >= self.MAX_L2_ATTEMPTS:
            natural_level = RepairLevel.L3
            escalate = True

        # 升级规则 3：L3 尝试太多次 → 终止修复
        # 场景：Planner 已经推倒重来了 2 次查询计划，但 SQL 仍然不行。
        #       此时说明当前问题超出了自动修复的能力范围，放弃是最经济的选择。
        if natural_level == RepairLevel.L3 and l3_attempts >= self.MAX_L3_ATTEMPTS:
            return RepairLevel.TERMINATE, "L3 重规划已达上限，终止修复", True

        # 构建修复策略的文本描述（用于日志记录和上下文提示）
        strategy = self._build_strategy(natural_level, failure_descriptions, escalate)
        return natural_level, strategy, escalate

    # ==================================================================
    # 方法：_build_strategy —— 构建修复策略描述文本
    # ==================================================================
    def _build_strategy(self, level: RepairLevel, failures: list[str], escalated: bool) -> str:
        """根据修复级别和失败信息，构建 human-readable 的策略描述。

        这个描述会被传给后续的修复流程（L1 传给 Builder 做 SQL 编辑，
        L2 传给 Builder 做 Schema 检索，L3 传给 Planner 重新规划），
        作为 LLM prompt 的一部分，告诉 LLM 当前要做什么级别的修复。

        参数:
            level: RepairLevel —— 确定的修复级别
            failures: list[str] —— 原始的错误描述列表（最多取前 3 条）
            escalated: bool —— 是否触发了升级

        返回:
            str —— 格式化的策略描述
        """
        # 如果是升级而来的，在描述前加上"【升级】"标记
        # 这让后续的 Builder/Planner 知道当前的修复级别是"被逼升级"的，
        # 而非"自然匹配"的，有助于 LLM 调整修复策略
        prefix = "【升级】" if escalated else ""

        # 取前 3 条错误描述（防止错误列表太长，超出 LLM 上下文窗口）
        # 用 "; " 连接成一句话
        failure_text = "; ".join(failures[:3])

        # 根据修复级别返回不同的策略描述模板
        if level == RepairLevel.L1:
            # L1：只在 SQL 字符串层面做文本替换
            # 例如：把 "date_format" 改成 "DATE_FORMAT"，去掉多余引号等
            return f"{prefix}L1 表面修复：直接在 SQL 文本上修正。问题: {failure_text}"
        elif level == RepairLevel.L2:
            # L2：保留 Planner 制定的查询计划，但重新检索 Schema 并重建 SQL
            # 例如：之前用了错误的表名，Builder 重新查 Schema 找到正确的表重新构建
            return f"{prefix}L2 语义修复：重新检索 Schema 并重新生成 SQL（保留 Plan 框架）。问题: {failure_text}"
        elif level == RepairLevel.L3:
            # L3：推倒重来，让 Planner 重新规划查询，Builder 再根据新计划构建 SQL
            # 代价最大，但能解决最深层次的业务逻辑问题
            return f"{prefix}L3 计划重生成：重新规划整个查询。问题: {failure_text}"
        # 理论上不会走到这里（TERMINATE 在上一级方法里已经提前返回了），
        # 但为了代码健壮性，保留兜底逻辑
        return f"{prefix}终止：无法修复。问题: {failure_text}"
