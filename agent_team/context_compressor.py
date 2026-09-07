# nl2sql/agent_team/context_compressor.py
#
# 上下文压缩模块 —— Token 估算与对话历史压缩。
#
# 什么是 Token？
# ===============
# 在 LLM（大语言模型）的世界里，Token 是模型处理文本的最小单位。
# 一个 Token 可以是一个词、一个词的一部分、或一个标点符号。
# 例如，"Hello world" 大约是 2 个 Token，"你好世界" 大约是 4 个 Token。
#
# 为什么需要压缩？
# ===============
# 每个 LLM 都有上下文窗口（context window）限制，即一次能处理的 Token 数。
# GPT-4 的上下文窗口通常是 8K、32K 或 128K Token。
# 在多轮对话中，历史信息会不断累积，很容易超过窗口限制。
# 超出的部分会被截断（truncate），导致"失忆"。
#
# 本模块通过"选择性压缩"来解决这个问题：
#   1. 估算当前文本的 Token 数
#   2. 如果超过阈值，进行结构化压缩（保留关键信息，丢弃冗余细节）
#
# 不同 Agent 的阈值（来自架构文档）：
#   - planner（规划器）：         40,000 Token —— 需要保留规划全貌
#   - refiner（优化器）：         25,000 Token —— 需要保留修复历史
#   - knowledge_keeper（知识维护器）：15,000 Token —— 知识条目较简短
#   - builder（构建器）：         不做压缩 —— 状态无关，每次调用独立

try:
    import tiktoken
except ImportError:
    tiktoken = None


class ContextCompressor:
    """Token 估算与对话历史压缩。

    核心能力：
    1. Token 估算 —— 准确计算一段文本的 Token 数量
    2. 阈值判断 —— 判断是否超过了给定 Agent 的上下文窗口限制
    3. 历史压缩 —— 将冗长的历史记录压缩为结构化摘要

    关于 tiktoken：
    ---------------
    tiktoken 是 OpenAI 开源的快速 Token 计数库。
    如果环境中没有安装 tiktoken，本模块会自动回退（fallback）到
    基于 char/4 的粗略估算。这意味着即使缺少依赖，系统也不会崩溃。
    """

    # 每个 Agent 类型的 Token 阈值（超过就需要压缩）
    # 这些值来自架构设计文档，权衡了"保留足够信息"和"不超窗口"。
    # planner 的阈值最高（40000），因为它需要看到完整的历史规划来制定下一步。
    # knowledge_keeper 的阈值最低（15000），因为知识条目通常简短且可独立理解。
    THRESHOLDS = {
        "planner": 40_000,
        "refiner": 25_000,
        "knowledge_keeper": 15_000,
    }

    def __init__(self):
        """初始化压缩器，尝试加载 tiktoken 编码器。

        如果 tiktoken 不可用（未安装或加载失败），
        self.encoder 会被设为 None，后续会使用后备估算方法。

        cl100k_base 是什么？
        这是 GPT-4、GPT-3.5-turbo 等模型使用的 Token 编码器名称。
        它知道如何将文本切分为模型能理解的 Token 序列。
        """
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 Token 数量。

        参数
        ----
        text: 要估算的文本字符串。

        返回
        ----
        Token 数量（整数）。

        后备策略（fallback）：
        --------------------
        当 tiktoken 不可用时，使用 len(text) // 4 估算。
        为什么是除以 4？
        这是一个经验法则（rule of thumb）：
          - 英文平均每个 Token 约 4 个字符
          - 中文平均每个 Token 约 1.5-2 个字符
        用 char/4 作为上限估算，虽然不精确，但能确保不会低估 Token 数，
        从而避免因实际 Token 超出窗口而被截断。
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        # 后备方案：字符数除以 4（粗略估算，偏向保守）
        return len(text) // 4

    def should_compress(self, text: str, agent_type: str) -> bool:
        """判断是否需要压缩。

        逻辑
        ----
        估算当前文本的 Token 数，如果超过对应 Agent 的阈值，
        则返回 True（需要压缩）。

        参数
        ----
        text: 要检查的文本。
        agent_type: Agent 类型名称（如 "planner"、"refiner"）。

        返回
        ----
        True 表示需要压缩，False 表示不需要。
        """
        threshold = self.THRESHOLDS.get(agent_type, 30_000)
        return self.estimate_tokens(text) > threshold

    def compress_plan_history(self, plans: list[dict]) -> str:
        """压缩规划器（Planner）的多轮规划历史。

        压缩策略
        --------
        - 如果规划轮数 <= 2：不压缩，完整输出。
        - 如果规划轮数 > 2：
            - 保留最近 2 轮的完整细节
            - 将更早的轮次合并为一段摘要（聚合锚表候选和已知失败原因）

        为什么这么设计？
        ----------------
        最近的规划最有可能影响当前决策，需要保留完整信息。
        早期的规划更多是"探索过程"，只需要知道关键结论（尝试了哪些锚表、
        哪些方案失败了）就可以了。

        参数
        ----
        plans: 规划历史列表，每个元素是一个包含策略信息的字典。

        返回
        ----
        压缩后的规划摘要字符串。
        """
        if len(plans) <= 2:
            # 轮次少，没必要压缩，直接格式化输出
            return self._format_plans(plans)

        # 分离最近轮次和早期轮次
        recent = plans[-2:]  # 保留最近 2 轮
        older = plans[:-2]   # 早期轮次需要压缩

        # 构建压缩后的摘要
        summary = "## 历史规划摘要（自动压缩）\n\n"
        summary += f"前 {len(older)} 轮规划的聚合：\n"

        # 从早期轮次中提取关键信息
        anchors = set()
        failures = []
        for p in older:
            if p.get("anchor_table"):
                anchors.add(p["anchor_table"])
            if p.get("error"):
                failures.append(p["error"])

        # 聚合锚表候选（去重）
        if anchors:
            summary += f"  锚表候选: {', '.join(anchors)}\n"
        # 聚合已知失败原因（只保留最近 3 个不同的错误）
        if failures:
            summary += f"  已知失败: {'; '.join(set(failures[-3:]))}\n"

        summary += "\n## 最近规划\n\n"
        # 保留最近轮次的完整信息
        summary += self._format_plans(recent)
        return summary

    def compress_refiner_attempts(self, attempts: list[dict]) -> str:
        """压缩优化器（Refiner）的修复尝试历史。

        压缩策略
        --------
        - 如果尝试次数 <= 2：不压缩，完整输出。
        - 如果尝试次数 > 2：
            - 按修复级别（L1/L2/L3）统计次数
            - 保留最后一次尝试的详细信息

        什么是 L1/L2/L3 修复级别？
        ============================
        L1（表面修复）：语法错误、列名拼写错误等——最低成本修复
        L2（语义修复）：逻辑错误、JOIN 条件错误等——中等成本修复
        L3（计划重生成）：整体策略错误，需要从头重新规划——最高成本修复

        参数
        ----
        attempts: 修复尝试历史列表。

        返回
        ----
        压缩后的修复摘要字符串。
        """
        if len(attempts) <= 2:
            # 尝试次数少，直接完整输出
            return self._format_attempts(attempts)

        # 按修复级别统计次数
        l1_count = sum(1 for a in attempts if a.get("level") == "L1")
        l2_count = sum(1 for a in attempts if a.get("level") == "L2")
        l3_count = sum(1 for a in attempts if a.get("level") == "L3")

        # 构建统计摘要
        summary = "## 修复历史（自动压缩）\n\n"
        if l1_count:
            summary += f"- L1 表面修复: {l1_count} 次\n"
        if l2_count:
            summary += f"- L2 语义修复: {l2_count} 次\n"
        if l3_count:
            summary += f"- L3 计划重生成: {l3_count} 次\n"

        # 保留最后一次尝试的详细信息
        # 因为最后一次尝试最能反映当前状态
        last = attempts[-1]
        summary += f"\n最近修复 ({last.get('level', '?')}): {last.get('action', '?')}\n"
        return summary

    # ------------------------------------------------------------------
    # 私有格式化方法
    # ------------------------------------------------------------------

    def _format_plans(self, plans: list[dict]) -> str:
        """将规划列表格式化为可读文本（不压缩，保留全部信息）。"""
        parts = []
        for i, p in enumerate(plans, 1):
            parts.append(
                f"  Round {i}: anchor={p.get('anchor_table', '?')}, "
                f"time_field={p.get('time_strategy', {}).get('field', '?')}"
            )
        return "\n".join(parts)

    def _format_attempts(self, attempts: list[dict]) -> str:
        """将修复尝试列表格式化为可读文本（不压缩，保留全部信息）。"""
        parts = []
        for a in attempts:
            # 截断诊断信息前 100 个字符，防止单条过长
            parts.append(
                f"  {a.get('level', '?')}: {a.get('action', '?')}"
                f" - {a.get('diagnosis', '?')[:100]}"
            )
        return "\n".join(parts)
