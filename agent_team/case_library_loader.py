# nl2sql/agent_team/case_library_loader.py
#
# 案例库加载模块 —— 管理从历史执行中积累的 SQL 案例。
#
# 什么是"案例"（Case）？
# ======================
# 案例就是"曾经执行过的一条 SQL 及其上下文"。当系统为一个自然语言问题生成 SQL
# 并执行后，这一过程就被记录为一个案例。未来遇到类似问题时，可以参考历史案例，
# 避免犯同样的错误。
#
# 为什么案例很重要？
# ===================
# LLM 生成 SQL 时有两大痛点：
#   1. 同样的错误可能反复出现（比如总是忘记 GROUP BY 的列在 SELECT 中）
#   2. 成功的模式没有被系统性地复用
# 案例库通过"记住历史"来解决这两个问题 —— 成功的案例可以指导未来，
# 失败的案例可以警示未来。
#
# 三级体系（Tier System）
# =======================
# 每个案例有一个 tier（等级）字段，表示其可靠程度：
#   Tier A（金牌）—— golden_match=True，表示这是经过验证的"标准答案"
#   Tier B（执行成功）—— SQL 执行通过，但未经验证是最优解
#   Tier C（执行失败）—— SQL 执行出错，记录了失败原因供参考
#
# 变体（Variant）机制
# ===================
# 同一个自然语言问题可能对应多条不同的 SQL（称为"变体"），
# 比如使用 JOIN 的不同写法、不同的聚合方式等。这些变体都会被记录，
# 但只有符合等级标准的才会进入对应 tier。
#
# 失败模式（Failure Pattern）跟踪
# ===============================
# 系统会跟踪常见的失败模式（如"JOIN 条件缺失"、"使用了不存在的列"等），
# 并将这些模式与具体的 sql_id 关联。未来遇到类似 SQL 时，可以提前预警。
import json
import os
import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Case:
    """案例类 —— 表示案例库中的一个单条记录。

    什么是 dataclass？
    Python 的 dataclass 是一个便捷的工具，自动为类生成 __init__、__repr__
    等方法，减少样板代码。每个字段的类型注解让代码更清晰。

    字段说明：
    --------
    case_id: 案例的唯一标识符，由 sql_id + sql_hash 拼接而成。
    sql_id: SQL 的 ID，用于关联同一问题的多个变体。
    question: 用户的原始自然语言问题。
    sql: 生成的 SQL 语句。
    tables: 该 SQL 涉及的数据表列表。
    tier: 案例等级 —— "A"（金牌）、"B"（执行成功）、"C"（执行失败）。
    consensus_ratio: 共识比例。如果同一个问题生成了多个 SQL，
        其中某个 SQL 被多次选中，consensus_ratio 就是选中次数/总尝试次数。
        取值范围 0.0 ~ 1.0，越接近 1.0 说明共识度越高。
    golden_match: 是否为经过人工或自动验证的"标准答案"（仅 Tier A）。
    execution_status: 执行状态，如 "ok"、"fail"、"unknown"。
    execution_row_count: 执行后返回的行数。
    knowledge: 与该案例相关的领域知识（如业务规则、特殊约定等）。
    complexity: 问题复杂度标签（如"简单"、"中等"、"复杂"）。
    sql_hash: SQL 的 MD5 哈希（取前 12 位），用于快速去重和比较。
    """
    case_id: str
    sql_id: str
    question: str
    sql: str
    tables: list[str]
    tier: str  # "A" = 金牌, "B" = 执行成功, "C" = 执行失败
    consensus_ratio: float = 0.0
    golden_match: bool = False
    execution_status: str = "unknown"
    execution_row_count: Optional[int] = None
    knowledge: str = ""
    complexity: str = "未知"
    sql_hash: str = ""

    def to_dict(self) -> dict:
        """将 Case 对象序列化为字典，方便 JSON 存储。"""
        return {
            "case_id": self.case_id,
            "sql_id": self.sql_id,
            "question": self.question,
            "sql": self.sql,
            "tables": self.tables,
            "tier": self.tier,
            "consensus_ratio": self.consensus_ratio,
            "golden_match": self.golden_match,
            "execution_status": self.execution_status,
            "execution_row_count": self.execution_row_count,
            "knowledge": self.knowledge,
            "complexity": self.complexity,
            "sql_hash": self.sql_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        """从字典创建 Case 对象（反序列化）。

        参数
        ----
        d: 字典，通常来自 JSON 文件的解析结果。

        返回
        ----
        Case 实例。

        注意
        ----
        使用 d.get(key, default) 而不是 d[key] 来提供默认值，
        这样即使 JSON 文件中缺少某些字段，也不会抛出 KeyError。
        """
        return cls(
            case_id=d["case_id"],
            sql_id=d["sql_id"],
            question=d.get("question", ""),
            sql=d.get("sql", ""),
            tables=d.get("tables", []),
            tier=d.get("tier", "B"),
            consensus_ratio=d.get("consensus_ratio", 0.0),
            golden_match=d.get("golden_match", False),
            execution_status=d.get("execution_status", "unknown"),
            execution_row_count=d.get("execution_row_count"),
            knowledge=d.get("knowledge", ""),
            complexity=d.get("complexity", "未知"),
            sql_hash=d.get("sql_hash", ""),
        )


class CaseLibrary:
    """可查询的案例库集合，带失败模式跟踪功能。

    这个类类似于一个"内存数据库"，提供了按 ID、tier 等级、sql_id 等
    维度查询案例的方法。

    初学者理解：
    -----------
    CaseLibrary 就像你的个人笔记本：
      - 你记录下每次解决问题的经验（add 方法）
      - 你可以按类别翻阅（get_by_tier）
      - 你可以查询某个特定问题的所有尝试（get_by_sql_id）
      - 你还可以记录哪些方法行不通（failure_patterns）
    """

    def __init__(self):
        # cases 字典：case_id -> Case 对象，提供 O(1) 的查找速度
        self.cases: dict[str, Case] = {}
        # failure_patterns 字典：sql_id -> 失败模式描述列表
        # 例如：{"sql_001": ["JOIN 条件缺失", "使用了不存在的列别名"]}
        self.failure_patterns: dict[str, list[str]] = {}

    def add(self, case: Case) -> None:
        """向案例库中添加一个案例。

        如果 case_id 已存在，会覆盖旧记录（因为字典的 key 是唯一的）。
        """
        self.cases[case.case_id] = case

    def get(self, case_id: str) -> Optional[Case]:
        """通过 case_id 获取案例。如果不存在，返回 None。"""
        return self.cases.get(case_id)

    def get_by_tier(self, tier: str) -> list[Case]:
        """获取指定等级的所有案例。

        参数
        ----
        tier: "A"、"B" 或 "C"。

        返回
        ----
        匹配该等级的案例列表。
        """
        return [c for c in self.cases.values() if c.tier == tier]

    def get_by_sql_id(self, sql_id: str) -> list[Case]:
        """获取指定 sql_id 的所有案例（即同一个问题的所有变体）。

        当系统为一个问题生成了多个 SQL 变体时，它们共享相同的 sql_id
        但有不同的 case_id。这个方法就是用来查找所有这些变体的。
        """
        return [c for c in self.cases.values() if c.sql_id == sql_id]

    def add_failure_pattern(self, sql_id: str, pattern: str) -> None:
        """为一个 sql_id 添加失败模式描述。

        failure_patterns 的作用：
        - 帮助识别反复出现的同类错误
        - 在生成 SQL 时可以参考历史失败模式，主动避开已知陷阱
        - 如果某个 sql_id 已经有相同的 pattern，不会重复添加（防重复）
        """
        if sql_id not in self.failure_patterns:
            self.failure_patterns[sql_id] = []
        if pattern not in self.failure_patterns[sql_id]:
            self.failure_patterns[sql_id].append(pattern)

    def get_failure_patterns(self, sql_id: str) -> list[str]:
        """获取指定 sql_id 的失败模式列表。"""
        return self.failure_patterns.get(sql_id, [])

    # ------------------------------------------------------------------
    # 便捷属性（properties）
    # ------------------------------------------------------------------
    # Python 的 @property 装饰器让方法可以像属性一样访问。
    # 例如 library.tier_a_count 而不是 library.tier_a_count()。

    @property
    def tier_a_count(self) -> int:
        """金牌案例（Tier A）的数量。"""
        return len(self.get_by_tier("A"))

    @property
    def tier_b_count(self) -> int:
        """执行成功案例（Tier B）的数量。"""
        return len(self.get_by_tier("B"))

    @property
    def tier_c_count(self) -> int:
        """执行失败案例（Tier C）的数量。"""
        return len(self.get_by_tier("C"))

    @property
    def total_count(self) -> int:
        """案例库中的总案例数。"""
        return len(self.cases)

    def summary(self) -> dict:
        """生成案例库的统计摘要。

        返回
        ----
        包含总数和各等级数量的字典。
        这在调试和监控中非常有用。
        """
        return {
            "total_cases": self.total_count,
            "tier_a_count": self.tier_a_count,
            "tier_b_count": self.tier_b_count,
            "tier_c_count": self.tier_c_count,
            "sql_ids_with_failure_patterns": len(self.failure_patterns),
        }

    def to_dict(self) -> dict:
        """将整个案例库序列化为字典，包括所有案例和失败模式。"""
        return {
            "cases": [c.to_dict() for c in self.cases.values()],
            "failure_patterns": self.failure_patterns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseLibrary":
        """从字典恢复案例库（反序列化）。

        参数
        ----
        d: 字典，通常来自从 JSON 读取的数据。
        """
        lib = cls()
        for case_data in d.get("cases", []):
            case = Case.from_dict(case_data)
            lib.add(case)
        lib.failure_patterns = d.get("failure_patterns", {})
        return lib


class CaseLibraryLoader:
    """案例库加载器 —— 从磁盘上的 tier 文件中加载案例数据。

    什么是 tier 文件？
    -----------------
    tier 文件是存储在磁盘上的 JSON 文件，按等级分类：
      - tier_a_golden_match.json —— 金牌案例（A 级）
      - tier_b_executed_ok.json —— 执行成功案例（B 级）
      - tier_c_execution_fail.json —— 执行失败案例（C 级）
      - tier_c_failure_patterns.json —— 失败模式数据（可选）

    加载流程：
    1. 扫描目录中的 tier 文件
    2. 逐个解析 JSON
    3. 对每个条目，处理其 variants（变体）
    4. 根据 tier 等级过滤和分类
    5. 统一合并到 CaseLibrary 中
    """

    @staticmethod
    def _hash_sql(sql: str) -> str:
        """计算 SQL 的 MD5 哈希值（取前 12 位）用于唯一标识。

        为什么用 MD5 前 12 位？
        MD5 完整输出是 32 位十六进制字符串，取前 12 位已经足够
        在绝大多数场景下避免碰撞（collision），同时更简洁。
        """
        return hashlib.md5(sql.encode()).hexdigest()[:12]

    def load_from_directory(self, cases_dir: str) -> CaseLibrary:
        """从指定目录加载所有案例文件到统一的 CaseLibrary。

        文件命名规则（预期）：
        - tier_a_golden_match.json  -> Tier A（金牌案例）
        - tier_b_executed_ok.json   -> Tier B（执行成功案例）
        - tier_c_execution_fail.json -> Tier C（执行失败案例）
        - tier_c_failure_patterns.json -> 失败模式数据（可选）

        参数
        ----
        cases_dir: 案例文件所在的目录路径。

        返回
        ----
        合并了所有 tier 案例的 CaseLibrary 实例。
        """
        library = CaseLibrary()

        # 文件名到 tier 等级的映射表
        file_map = {
            "tier_a_golden_match.json": "A",
            "tier_b_executed_ok.json": "B",
            "tier_c_execution_fail.json": "C",
        }

        # 逐个加载 tier 文件
        for filename, tier in file_map.items():
            path = os.path.join(cases_dir, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tier_lib = self._load_tier(data, tier)
                # 将当前 tier 的案例合并到主 library 中
                for case in tier_lib.cases.values():
                    library.add(case)

        # 额外加载失败模式数据（如果存在的话）
        patterns_path = os.path.join(cases_dir, "tier_c_failure_patterns.json")
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            self._load_failure_patterns(library, patterns_data)

        return library

    def _load_tier(self, data: list[dict], tier: str) -> CaseLibrary:
        """从某个 tier 文件的数据列表中加载案例。

        参数
        ----
        data: JSON 解析后的列表，每个元素是一个案例条目。
        tier: 当前处理的等级（"A"、"B" 或 "C"）。

        处理逻辑：
        ---------
        每个条目可能包含一个或多个 SQL 变体（all_variants 字段）。
          - 如果没有变体：直接从条目本身创建一个 Case。
          - 如果有变体：为每个符合等级标准的变体创建一个 Case。

        等级过滤规则：
        - Tier A：只保留 golden_match=True 的变体
        - Tier B：所有执行通过的变体
        - Tier C：只保留执行失败的变体（golden_match=False）

        consensus_ratio（共识比例）的计算：
        当有变体时，每个变体的 consensus_ratio = 该变体被选中次数 / 总尝试次数。
        例如：一个问题被尝试了 5 次，某个变体被选中了 3 次，则 consensus_ratio = 0.6。
        """
        library = CaseLibrary()

        for entry in data:
            # 提取条目的公共字段
            sql_id = entry["sql_id"]
            question = entry.get("question", "")
            tables = entry.get("table_list", [])
            knowledge = entry.get("knowledge", "")
            # 兼容中英文的复杂度字段名
            complexity = entry.get("复杂度", entry.get("complexity", "未知"))
            consensus_ratio = entry.get("consensus_ratio", 0.0)

            # 检查是否包含变体列表
            variants = entry.get("all_variants", [])
            if not variants:
                # ---- 情况 1：没有变体 ----
                # 直接从条目自身创建单个 Case
                sql = entry.get("sql", "")
                sql_hash = self._hash_sql(sql)
                case = Case(
                    case_id=f"case_{sql_id}_{sql_hash}",
                    sql_id=sql_id,
                    question=question,
                    sql=sql,
                    tables=tables,
                    tier=tier,
                    consensus_ratio=consensus_ratio,
                    golden_match=entry.get("golden_match", False),
                    execution_status=entry.get("execution_status", "unknown"),
                    execution_row_count=entry.get("execution_row_count"),
                    knowledge=knowledge,
                    complexity=complexity,
                    sql_hash=sql_hash,
                )
                library.add(case)
            else:
                # ---- 情况 2：有变体 ----
                # 为每个符合 Tier 标准的变体创建一个 Case
                for variant in variants:
                    sql_hash = variant.get(
                        "sql_hash",
                        self._hash_sql(variant.get("sql", ""))
                    )

                    # Tier A 过滤：只保留 golden_match=True 的变体
                    if tier == "A" and not variant.get("golden_match", False):
                        continue

                    # Tier C 过滤：只保留 golden_match=False 的变体
                    if tier == "C":
                        if variant.get("golden_match", False):
                            continue

                    sql = variant.get("sql", entry.get("sql", ""))
                    case = Case(
                        case_id=f"case_{sql_id}_{sql_hash}",
                        sql_id=sql_id,
                        question=question,
                        sql=sql,
                        tables=tables,
                        tier=tier,
                        # consensus_ratio: variant 的 consensus_count / 总运行次数
                        # max(..., 1) 防止除零错误
                        consensus_ratio=(
                            variant.get("consensus_count", 1)
                            / max(entry.get("consensus_total_runs", 1), 1)
                        ),
                        golden_match=variant.get("golden_match", False),
                        execution_status=variant.get("execution_status", "unknown"),
                        execution_row_count=variant.get("execution_row_count"),
                        knowledge=knowledge,
                        complexity=complexity,
                        sql_hash=sql_hash,
                    )
                    library.add(case)

        return library

    def _load_failure_patterns(
        self, library: CaseLibrary, patterns_data: list[dict]
    ) -> None:
        """将失败模式数据加载到案例库中。

        参数
        ----
        library: 目标 CaseLibrary 实例。
        patterns_data: 失败模式列表，每个元素是包含以下字段的字典：
            - failure_pattern: 失败模式描述文本
            - affected_sql_ids: 受此模式影响的 sql_id 列表

        一个失败模式可能影响多个 sql_id（反之亦然）。
        例如：模式"缺少 GROUP BY"可能影响 sql_001、sql_002、sql_003。
        """
        for entry in patterns_data:
            pattern_desc = entry.get("failure_pattern", "")
            for sql_id in entry.get("affected_sql_ids", []):
                library.add_failure_pattern(sql_id, pattern_desc)
