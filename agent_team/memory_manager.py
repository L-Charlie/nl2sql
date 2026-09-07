# nl2sql/agent_team/memory_manager.py
#
# 每个 Agent 的私有记忆管理器 —— 带热/冷分层的记忆系统。
#
# 什么是 Agent 记忆？
# ===================
# 在多人多 Agent 系统中，每个 Agent 都有自己的"工作记忆"。
# 比如规划器（Planner）会记住"上次处理类似问题时选择了哪个锚表"，
# 构建器（Builder）会记住"上次用的 JOIN 模式哪里出错了"。
#
# 为什么每个 Agent 需要自己的记忆？
# =================================
# 不同 Agent 有不同的职责，需要记住的信息也不同：
#   - Planner：记住哪些规划策略有效，哪些锚表路径走得通
#   - Builder：记住 SQL 构建模式、踩过的坑
#   - Refiner：记住修复模式，哪些修改能解决特定的错误
#   - SchemaLinker：记住模式链接的经验
#   - KnowledgeKeeper：记住领域知识条目
#
# 信任等级系统（Trust Level）
# ==========================
# 每条记忆都有一个信任等级，基于成功使用次数：
#   - trusted（信任）：成功 >= 10 次 —— 几乎不会出错，可以跳过验证直接使用
#   - verified（已验证）：成功 >= 3 次 —— 经过多次验证，可以放心使用
#   - tentative（初步）：成功 < 3 次 —— 还不太确定，需要重新验证
#   - cold（冷归档）：超过 90 天未使用 —— 已移至冷存储
#
# 为什么需要这个等级？
# 一条记忆被成功使用 10 次，远比只成功 1 次更可信。
# 这类似于人类的经验：做成功 10 次的事情，你也会更信任这个方法。
#
# 热/冷分层（Hot/Cold Tiering）
# =============================
# 为了避免内存占用过大，记忆分为两层：
#   - 热层（Hot）：保存在内存中，访问速度快，只保留最近活跃的记忆
#   - 冷层（Cold）：保存在磁盘上，需要时加载，存放长期未使用的记忆
# 这种设计借鉴了 CPU 缓存的分层思想。
#
# 记忆衰减（Decay）
# =================
# 如果一条记忆长时间未被使用，它会从热层移到冷层。
# 类似于人的记忆 —— 长期不用的信息会逐渐淡忘。
# 默认衰减阈值是 90 天。
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryEntry:
    """单条 Agent 记忆记录。

    理解 dataclass：
    Python 的 @dataclass 自动为类生成 __init__ 方法，
    我们只需要声明字段和类型即可。

    字段说明：
    --------
    signature: 记忆的签名（唯一标识符），用于去重和查询。
        格式通常是"关键词_摘要"，如 "join_sales_customers"。
    data: 实际的记忆内容（字典），可以存储任意结构化数据。
    entry_type: 记忆类型 —— "success"（成功）、"failure"（失败）、
        "coding_pattern"（编码模式）、"mapping"（映射关系）。
    success_count: 该记忆被成功使用的次数。
    failure_count: 该记忆被标记为失败的次数。
    last_used_days_ago: 距上次使用过去了多少天（0.0 表示刚刚使用过）。
    created_at: 创建时间的时间戳（UNIX 时间）。
    """

    signature: str
    data: dict = field(default_factory=dict)
    entry_type: str = "success"  # 可选值: "success", "failure", "coding_pattern", "mapping"
    success_count: int = 1
    failure_count: int = 0
    last_used_days_ago: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def trust_level(self) -> str:
        """根据成功使用次数计算信任等级。

        计算规则
        --------
        - success_count >= 10: "trusted"（信任级，无需再验证）
        - success_count >= 3:  "verified"（已验证级，正常使用）
        - 其他:                "tentative"（初步级，需要再次验证）

        为什么是 10 和 3 这两个阈值？
        - 3 次成功 = "事不过三"，足以确认模式基本正确
        - 10 次成功 = "十拿九稳"，可以认为是高度可靠的经验
        """
        if self.success_count >= 10:
            return "trusted"
        elif self.success_count >= 3:
            return "verified"
        return "tentative"

    def increment_success(self):
        """增加成功计数，重置使用时间。

        当一条记忆被成功使用时调用：
        1. 成功计数 +1
        2. 将 last_used_days_ago 重置为 0.0（表示"刚刚使用过"）
        """
        self.success_count += 1
        self.last_used_days_ago = 0.0

    def decrement_success(self):
        """减少成功计数（但不会低于 0）。

        当发现之前的成功记忆其实是错误时调用。
        使用 max(0, ...) 确保计数不会变成负数。
        """
        self.success_count = max(0, self.success_count - 1)

    def to_dict(self) -> dict:
        """序列化为字典，用于 JSON 存储。"""
        return {
            "signature": self.signature,
            "data": self.data,
            "entry_type": self.entry_type,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_used_days_ago": self.last_used_days_ago,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        """从字典反序列化为 MemoryEntry。

        使用 d.get(key, default) 而不是 d[key]，
        确保即使 JSON 中缺少字段也不会抛异常。
        """
        return cls(
            signature=d.get("signature", ""),
            data=d.get("data", {}),
            entry_type=d.get("entry_type", "success"),
            success_count=d.get("success_count", 1),
            failure_count=d.get("failure_count", 0),
            last_used_days_ago=d.get("last_used_days_ago", 0.0),
            created_at=d.get("created_at", time.time()),
        )


class AgentMemoryManager:
    """管理单个 Agent 的记忆文件，支持热/冷分层与衰减。

    架构说明
    ========
    ```
    memories_dir/
    ├── planner_memory.json
    ├── builder_memory.json
    ├── refiner_memory.json
    ├── schema_linker_memory.json
    └── knowledge_keeper_memory.json
    ```

    热/冷分层
    ==========
    热层（Hot Cache）：
      - 保存在内存的 self._hot 字典中
      - 只包含 last_used_days_ago < 90 的记忆
      - 查询（query）只搜索热层，响应速度快

    冷层（Cold Storage）：
      - 只保存在磁盘 JSON 文件中
      - 包含 last_used_days_ago >= 90 的记忆
      - 需要时通过 _load_cold() 加载
      - 不参与日常查询，节省内存

    为什么这么做？
    ==============
    如果所有记忆都放在内存中：
      1. 长期运行的 Agent 会积累大量记忆，内存占用越来越高
      2. 大部分冷记忆很少被访问，放在内存中是浪费
    分层设计是"权衡"（trade-off）的典型例子：
      用少许的磁盘 IO 换取了大量内存的节省。

    信任等级：
      trusted:  success_count >= 10  （跳过重新验证）
      verified: success_count >= 3   （正常使用）
      tentative: success_count < 3   （需要重新验证）
      cold:     last_used > 90 天    （已归档）
    """

    # 系统支持的 Agent 类型列表
    AGENTS = ["planner", "builder", "refiner", "schema_linker", "knowledge_keeper"]

    def __init__(self, memories_dir: str):
        """初始化记忆管理器。

        参数
        ----
        memories_dir: 记忆文件存储目录的路径。

        初始化流程：
        1. 确保记忆目录存在（不存在则创建）
        2. 为每个 Agent 初始化空的热缓存（self._hot）
        3. 如果某个 Agent 的记忆文件不存在，创建空文件
        4. 从文件加载热层记忆
        """
        self.memories_dir = memories_dir
        # 自动创建目录（exist_ok=True 表示目录已存在时不报错）
        os.makedirs(memories_dir, exist_ok=True)

        # 热缓存：agent_name -> [MemoryEntry, ...]
        self._hot: dict[str, list[MemoryEntry]] = {
            agent: [] for agent in self.AGENTS
        }

        # 确保每个 Agent 都有对应的记忆文件
        for agent in self.AGENTS:
            path = self._path_for(agent)
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump([], f)

        # 从磁盘加载热层记忆
        self._load_all()

    def _path_for(self, agent: str) -> str:
        """获取指定 Agent 的记忆文件路径。

        例如：agent="planner" -> "memories_dir/planner_memory.json"
        """
        return os.path.join(self.memories_dir, f"{agent}_memory.json")

    def _load_all(self):
        """从磁盘加载所有 Agent 的记忆条目到热缓存。

        加载逻辑：
        - 读取每个 Agent 的 JSON 文件
        - 将每条记录反序列化为 MemoryEntry 对象
        - 只加载 last_used_days_ago < 90 的条目到热层
        - 超过 90 天的条目留在磁盘中（冷层）

        异常处理：
        - 如果文件不存在或 JSON 格式错误，该 Agent 的热缓存保持为空列表
        - 程序不会因此崩溃
        """
        for agent in self.AGENTS:
            path = self._path_for(agent)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entries = [MemoryEntry.from_dict(d) for d in data]
                # 热层过滤：只保留近期使用过的（last_used_days_ago < 90）
                self._hot[agent] = [
                    e for e in entries if e.last_used_days_ago < 90
                ]
            except (FileNotFoundError, json.JSONDecodeError):
                # 文件不存在或损坏时，使用空列表
                self._hot[agent] = []

    def _save(self, agent: str):
        """将指定 Agent 的全部记忆（热层 + 冷层）写回磁盘。

        为什么每次修改都要写磁盘？
        虽然频繁写盘会影响性能，但这样可以确保：
        1. 程序意外退出时不会丢失数据
        2. 多会话之间记忆是持久化的
        如果未来性能成为瓶颈，可以改为定时批量写入。
        """
        path = self._path_for(agent)
        # 合并热层和冷层的记忆
        all_entries = self._hot[agent] + self._load_cold(agent)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(
                [e.to_dict() for e in all_entries],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _load_cold(self, agent: str) -> list[MemoryEntry]:
        """从磁盘加载指定 Agent 的冷层记忆。

        冷层记忆是 last_used_days_ago >= 90 的条目。
        这些条目平时不加载到内存，只在需要保存时才读取。
        """
        path = self._path_for(agent)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [
                MemoryEntry.from_dict(d)
                for d in data
                if d.get("last_used_days_ago", 0) >= 90
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def query(
        self, agent: str, signature_fragment: str, top_k: int = 5
    ) -> list[MemoryEntry]:
        """在 Agent 的热缓存中按签名子串搜索记忆。

        什么是签名子串匹配？
        就是搜索 signature 中包含指定关键词的记录。
        例如，搜索 "join" 会匹配所有 signature 中含有 "join" 的记忆。

        为什么只搜索热层（hot cache）？
        因为热层是近期使用的记忆，最有可能与当前问题相关。
        冷层记忆是"遗忘"的旧信息，一般不需要搜索。

        参数
        ----
        agent: Agent 名称。
        signature_fragment: 要搜索的关键词（不区分大小写）。
        top_k: 最多返回的结果数，默认为 5。

        返回
        ----
        匹配的记忆条目列表，按成功次数降序排列。
        （成功次数多的排在前面，代表更可靠的经验。）
        """
        if agent not in self._hot:
            return []
        # 子串匹配（不区分大小写）
        matches = [
            e for e in self._hot[agent]
            if signature_fragment.lower() in e.signature.lower()
        ]
        # 按 success_count 降序排列 —— 最可靠的记忆在前
        matches.sort(key=lambda e: e.success_count, reverse=True)
        return matches[:top_k]

    def get_trusted(self, agent: str) -> list[MemoryEntry]:
        """获取指定 Agent 的所有信任级记忆（success_count >= 10）。

        信任级记忆意味着已经成功使用了至少 10 次，
        在使用时可以直接采用，不需要再验证其有效性。
        """
        return [
            e for e in self._hot.get(agent, [])
            if e.trust_level == "trusted"
        ]

    def record(self, agent: str, entry: MemoryEntry):
        """记录新的记忆条目，或合并到已有条目中。

        工作原理
        ========
        1. 检查热缓存中是否已有相同 signature 的条目
        2. 如果有（去重合并）：
           - 累加成功/失败次数
           - 重置 last_used_days_ago 为 0.0
           - 合并 data 字典
        3. 如果没有（新增）：
           - 直接追加到热缓存中
        4. 写回磁盘持久化

        基于签名的去重（Signature-based Deduplication）
        ==============================================
        同一段"经验"不应该被重复存储。
        通过 signature 字段来识别重复：如果新条目的 signature
        已经存在，就合并到已有条目中，而不是新建一条。
        这样可以避免记忆膨胀，并保持每条记忆的统计数据准确。
        """
        if agent not in self._hot:
            return

        # 检查是否已有相同的 signature
        for existing in self._hot[agent]:
            if existing.signature == entry.signature:
                # 合并模式：累加统计数据，更新内容
                existing.success_count += entry.success_count
                existing.failure_count += entry.failure_count
                existing.last_used_days_ago = 0.0
                existing.data.update(entry.data)
                self._save(agent)
                return

        # 新增模式：没有重复，直接追加
        self._hot[agent].append(entry)
        self._save(agent)

    def record_failure(self, agent: str, entry: MemoryEntry):
        """记录一条失败记忆。

        与 record 的区别：
        - 自动将 entry_type 设为 "failure"
        - 设置 failure_count = 1
        - 然后调用 record 方法进行去重合并或新增

        这样失败经验也会被记住，下次遇到类似情况时可以提前预警。
        """
        entry.entry_type = "failure"
        entry.failure_count = 1
        self.record(agent, entry)

    def apply_decay(self, agent: str, days_threshold: int = 90):
        """执行记忆衰减 —— 将超过指定天数未使用的记忆移到冷层。

        什么是衰减（Decay）？
        =====================
        类似于人类记忆的"遗忘"过程 —— 长时间不用的信息逐渐被归档。
        衰减不是删除，而是从热缓存移到冷存储。

        参数
        ----
        agent: Agent 名称。
        days_threshold: 天数阈值（默认 90 天）。

        逻辑
        ----
        1. 遍历该 Agent 热缓存中的所有条目
        2. 移除 last_used_days_ago >= days_threshold 的条目
        3. 如果有条目被移除，保存到磁盘（这些条目会通过 _save 的
           冷层加载逻辑继续保存在文件中，但不再驻留内存）
        """
        if agent not in self._hot:
            return
        before = len(self._hot[agent])
        self._hot[agent] = [
            e for e in self._hot[agent]
            if e.last_used_days_ago < days_threshold
        ]
        after = len(self._hot[agent])
        if before != after:
            # 有条目被移出热层，需要更新磁盘文件
            self._save(agent)
