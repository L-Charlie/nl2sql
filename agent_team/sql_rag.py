# nl2sql/agent_team/sql_rag.py
"""SQL 经验记忆 RAG —— 把每次 NL2SQL 运行结果变成可检索的经验。

核心理念：不往 prompt 硬塞大量 few-shot，而是维护一个可检索的经验库。
每次 Builder 生成 SQL 前，先去经验库里查"有没有类似问题我是怎么处理的"，
把最相关的 1-3 条作为参考注入 prompt。

检索流程：
  当前问题 → 向量召回(动态阈值>0.7) → Cross-Encoder精排 + 结构分融合(0.8+0.2) → top-3

去重策略（写时去重）：
  同 db + 同表集 + 问题相似度 > 0.85 → 判定为同一题，在 add() 时合并/替换，
  不等到检索时再去重。
"""

from __future__ import annotations

import json
import os
import hashlib
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SQLExperience:
    """一条 SQL 生成经验——每次运行成功后写入 RAG 的一条记录。

    属性说明:
        question: 用户的自然语言问题原文
        question_embedding: 问题文本的语义向量（sentence-transformers 编码）
        sql: 最终生成的 SQL 语句
        db_id: 数据库标识（如 "pets_1"），用于去重时判定同一道题
        tables_used: 这条 SQL 涉及的表名列表，用于去重判定和结果展示
        success: 最终是否执行成功
        error_type: 若失败，错误分类（syntax / schema / logic / time_field / other）
        iteration_count: 修了几次才成功（1 = 一次过）
        timestamp: 创建时间戳
        failure_notes: 若这条是成功记录，但曾有同类失败，这里记录失败类型
    """
    question: str
    question_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    sql: str = ""
    db_id: str = ""
    tables_used: list[str] = field(default_factory=list)
    success: bool = True
    error_type: str = ""
    iteration_count: int = 1
    timestamp: float = field(default_factory=time.time)
    failure_notes: list[str] = field(default_factory=list)

    def _count_joins(self) -> int:
        """返回 SQL 中的 JOIN 数量（用于经验注入时的结构摘要）。"""
        return (self.sql or "").lower().count(" join ")

    def _detect_set_op(self) -> str:
        """检测 SQL 中使用的集合操作类型（用于经验注入时的结构摘要）。"""
        s = (self.sql or "").lower()
        if "intersect" in s:
            return "INTERSECT"
        if "except" in s:
            return "EXCEPT"
        if "union" in s:
            return "UNION"
        return ""

    def to_dict(self) -> dict:
        """序列化为 JSON 可存储的字典（不含向量，向量单独存 .npy）。"""
        return {
            "question": self.question,
            "sql": self.sql,
            "db_id": self.db_id,
            "tables_used": self.tables_used,
            "success": self.success,
            "error_type": self.error_type,
            "iteration_count": self.iteration_count,
            "timestamp": self.timestamp,
            "failure_notes": self.failure_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SQLExperience":
        """从 JSON 反序列化（向量需从 .npy 单独加载）。"""
        return cls(
            question=d.get("question", ""),
            sql=d.get("sql", ""),
            db_id=d.get("db_id", ""),
            tables_used=d.get("tables_used", []),
            success=d.get("success", True),
            error_type=d.get("error_type", ""),
            iteration_count=d.get("iteration_count", 1),
            timestamp=d.get("timestamp", time.time()),
            failure_notes=d.get("failure_notes", []),
        )


# ============================================================================
# 全局编码器单例 —— 避免每次检索重新加载模型
# ============================================================================

_encoder = None
_encoder_model_name = "BAAI/bge-m3"

# bge-m3 要求检索时 query 带指令前缀，document 不带
_BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _get_encoder():
    """获取全局 sentence-transformers 编码器单例。

    首次调用时加载模型（耗时 ~3-5秒），后续调用直接返回。
    如果 sentence-transformers 未安装，返回 None（回退到字符二元组编码）。
    """
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer(_encoder_model_name)
        except ImportError:
            _encoder = None
    return _encoder


# ============================================================================
# 全局 Cross-Encoder Reranker 单例 —— 替代 LLM 精排
# ============================================================================

_reranker = None


def _get_reranker():
    """获取全局 Cross-Encoder reranker 单例。

    首次调用时加载 BAAI/bge-reranker-v2-m3（耗时 ~2-3秒）。
    不可用时返回 None，retrieve() 回退到纯向量分数排序。
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        except Exception:
            _reranker = None
    return _reranker


# ============================================================================
# 结构意图提取 —— 用于检索时的 0.2 权重结构分融合
# ============================================================================

_STRUCTURE_KW = {
    "needs_window": [
        "排名", "第.*名", "前.*名", "rank", "row_number",
        "连续.*天", "连续.*月", "consecutive", "lag", "lead",
        "上一条", "下一条", "上一行", "下一行",
    ],
    "needs_group_by": [
        "每个", "每组", "各", "每类", "各种", "不同.*的",
        "per", "each", "every",
    ],
    "needs_set_op": [
        "交集", "intersect", "差集", "except", "并集", "union",
    ],
    "needs_subquery": [
        "子查询", "subquery", "嵌套查询",
    ],
}


def _question_structure_intent(question: str) -> dict[str, bool]:
    """从问题文本提取 SQL 结构意图。纯关键词匹配，允许误报（0.2 权重容忍噪声）。"""
    q = question.lower()
    return {
        key: any(re.search(kw, q) for kw in kws)
        for key, kws in _STRUCTURE_KW.items()
    }


def _structure_overlap_score(intent: dict[str, bool], sql_key: dict) -> float:
    """计算问题结构意图与候选 SQL 结构特征的重叠度。

    返回值 [0, 1]，1 = 完全匹配，0 = 所有意图都不匹配。
    如果问题没有任何结构意图，返回 0.5（中性分）。
    """
    mapping = [
        (intent.get("needs_window", False), sql_key.get("has_window", False)),
        (intent.get("needs_group_by", False), sql_key.get("has_group_by", False)),
        (intent.get("needs_set_op", False), sql_key.get("has_set_op", False)),
        (intent.get("needs_subquery", False), sql_key.get("has_subquery", False)),
    ]
    active = sum(1 for i, _ in mapping if i)
    if active == 0:
        return 0.5
    matches = sum(1 for i, has_f in mapping if i and has_f)
    return matches / active


# ============================================================================
# 简单的 BM25 实现 —— 用于 SchemaLinker 的关键词相关性计算
# ============================================================================

class SimpleBM25:
    """BM25 实现，用于对少量候选做关键词排序。

    支持 fit() 语料库后进行完整 BM25 计算（含 IDF + avgdl 归一化）。
    未 fit() 时回退到简化公式，保证向后兼容。

    注意：此类别被 schema_linker.py 直接导入，不可删除。
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._avgdl: float = 1.0
        self._fitted: bool = False

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：英文按空格，中文按单字。"""
        text = text.lower()
        tokens = []
        for word in re.findall(r'[一-鿿]|[a-zA-Z0-9_]+', text):
            tokens.append(word)
        return tokens

    def fit(self, docs: list[str]):
        """从文档集合学习 IDF 和平均文档长度。

        调用此方法后，score() 将使用完整 BM25 公式。
        不调用则回退到简化 TF 公式（向后兼容）。
        """
        if not docs:
            return

        tokenized = [self._tokenize(d) for d in docs]
        N = len(tokenized)
        self._avgdl = sum(len(t) for t in tokenized) / N

        df: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {
            t: float(np.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0))
            for t, df_t in df.items()
        }
        self._fitted = True

    def score(self, query: str, doc: str) -> float:
        """计算 query 和 doc 之间的 BM25 相似度分数。

        如果调过 fit()，使用完整 BM25：IDF × TF / (TF + k1 × length_norm)。
        否则回退到简化 TF 公式（无 IDF，无长度归一化）。
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(doc)

        if not query_tokens or not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        doc_tf: dict[str, int] = {}
        for t in doc_tokens:
            doc_tf[t] = doc_tf.get(t, 0) + 1

        score = 0.0
        for qt in query_tokens:
            if qt not in doc_tf:
                continue
            tf = doc_tf[qt]
            numerator = tf * (self.k1 + 1)
            # 完整公式时用 avgdl 做长度归一化，回退时不做
            avgdl = self._avgdl if self._fitted else float(doc_len)
            length_norm = 1.0 - self.b + self.b * doc_len / max(avgdl, 1.0)
            denominator = tf + self.k1 * length_norm
            term_score = numerator / max(denominator, 0.001)
            if self._fitted:
                term_score *= self._idf.get(qt, 0.0)
            score += term_score

        return score


# ============================================================================
# SQL 经验 RAG 存储 —— 核心类
# ============================================================================

class SQLExperienceStore:
    """SQL 经验记忆的 RAG 存储。

    写入时自动去重，检索时两阶段：向量召回 → Cross-Encoder精排+结构融合。

    使用方式:
        store = SQLExperienceStore()
        store.add(experience)                    # 写入（自动去重）
        results = store.retrieve(question)       # 检索 top-3

    持久化:
        store.save("data/sql_experiences.json")  # 元数据
        store.load("data/sql_experiences.json")  # 加载
    """

    def __init__(self, model_client=None, model: str = "deepseek-v4-flash"):
        self._experiences: list[SQLExperience] = []
        self._question_embeddings: Optional[np.ndarray] = None  # (N, dim)
        self._model_client = model_client
        self._model = model
        self._sim_threshold = 0.95  # 问题相似度>0.95 + 同表 + SQL结构相近 → 同一题
        # 回退编码器的共享词表（保证建库和查询时维度一致）
        self._fallback_vocab: Optional[dict[str, int]] = None

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._experiences)

    @property
    def success_count(self) -> int:
        return sum(1 for e in self._experiences if e.success)

    # ── 编码 ──────────────────────────────────────────────────────────────

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        """编码存储的文档（不带指令前缀）。

        bge-m3 要求 document 不做指令前缀注入，与 query 不对称编码。
        """
        encoder = _get_encoder()
        if encoder is not None:
            return np.array(encoder.encode(texts, normalize_embeddings=True))
        return self._fallback_encode(texts)

    def _encode_query(self, text: str) -> np.ndarray:
        """编码检索查询（带 bge-m3 指令前缀）。

        bge-m3 通过 prompt 参数注入指令前缀以提升检索精度。
        """
        encoder = _get_encoder()
        if encoder is not None:
            return np.array(encoder.encode(
                [text], normalize_embeddings=True,
                prompt=_BGE_INSTRUCTION,
            ))
        return self._fallback_encode([text])

    def _fallback_encode(self, texts: list[str]) -> np.ndarray:
        """字符二元组回退编码。

        建库时：从 texts 构建词表 → 存入 self._fallback_vocab
        查询时：用 self._fallback_vocab 编码，OOV bigram 直接忽略

        这样保证无论什么文本，输出的向量维度始终一致。
        """
        # 如果有共享词表（查询阶段），直接用
        if self._fallback_vocab is not None:
            vectors = np.zeros((len(texts), len(self._fallback_vocab)))
            for idx, text in enumerate(texts):
                chars = text.lower()
                for i in range(len(chars) - 1):
                    bigram = chars[i:i+2]
                    if bigram in self._fallback_vocab:
                        vectors[idx, self._fallback_vocab[bigram]] += 1
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vectors / norms

        # 没有共享词表（建库阶段）：从 texts 构建
        vocab = {}
        for text in texts:
            chars = text.lower()
            for i in range(len(chars) - 1):
                bigram = chars[i:i+2]
                if bigram not in vocab:
                    vocab[bigram] = len(vocab)

        if not vocab:
            return np.zeros((len(texts), 1))

        self._fallback_vocab = vocab  # 保存为共享词表

        vectors = np.zeros((len(texts), len(vocab)))
        for idx, text in enumerate(texts):
            chars = text.lower()
            for i in range(len(chars) - 1):
                bigram = chars[i:i+2]
                if bigram in vocab:
                    vectors[idx, vocab[bigram]] += 1

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    # ── 写入（带去重） ────────────────────────────────────────────────────

    def add(self, experience: SQLExperience) -> bool:
        """写入一条新经验。若已存在同一道题，按策略合并/替换。

        同一道题判定: db_id 相同 + 表集完全相同 + 问题 embedding 余弦相似度 > 0.85。

        合并策略:
            - 已有成功 + 新增成功 → 保留 iterations 更少的
            - 已有成功 + 新增失败 → 不替换，给旧的加 failure_notes
            - 已有失败 + 新增成功 → 替换（最有价值的更新）
            - 已有失败 + 新增失败且同 error_type → 跳过
            - 已有失败 + 新增失败且不同 error_type → 两条都保留

        返回:
            True 表示新增或替换了一条记录，False 表示被去重跳过。
        """
        # 确保有问题向量
        if experience.question_embedding is None and experience.question:
            vecs = self._encode_documents([experience.question])
            experience.question_embedding = vecs[0]

        # 1. 查找是否已有同一道题
        existing_idx = self._find_similar(
            experience.question_embedding,
            experience.db_id,
            frozenset(experience.tables_used),
            new_sql=experience.sql,
        )

        if existing_idx is None:
            # 新题，直接插入
            self._insert(experience)
            return True

        existing = self._experiences[existing_idx]

        # 2. 合并策略
        if existing.success and experience.success:
            # 两次都成功 → 保留 iterations 更少的（一步到位优于修很多次）
            if experience.iteration_count < existing.iteration_count:
                self._experiences[existing_idx] = experience
                self._update_embeddings()
                return True
            return False

        elif existing.success and not experience.success:
            # 旧的已成功，新的失败 → 不替换，只记录失败类型供参考
            if experience.error_type and experience.error_type not in existing.failure_notes:
                existing.failure_notes.append(experience.error_type)
            return False

        elif not existing.success and experience.success:
            # 旧的失败、新的成功 → 替换！最有价值的更新
            self._experiences[existing_idx] = experience
            self._update_embeddings()
            return True

        else:
            # 两次都失败 → 不同 error_type 都保留
            if experience.error_type and experience.error_type != existing.error_type:
                self._insert(experience)
                return True
            return False

    def _insert(self, experience: SQLExperience):
        """插入一条新经验并更新向量矩阵。"""
        self._experiences.append(experience)
        if self._question_embeddings is not None and experience.question_embedding is not None:
            self._question_embeddings = np.vstack([
                self._question_embeddings,
                experience.question_embedding.reshape(1, -1),
            ])
        else:
            self._update_embeddings()

    def _update_embeddings(self):
        """重建问题向量矩阵（在替换/删除条目后调用）。"""
        q_embs = []
        for e in self._experiences:
            if e.question_embedding is not None:
                q_embs.append(e.question_embedding)
        self._question_embeddings = np.array(q_embs) if q_embs else None

    @staticmethod
    def _sql_structure_key(sql: str) -> dict:
        """从 SQL 文本中提取结构特征，用于判断两条 SQL 的复杂度是否在同一级别。

        返回的特征字典:
            - join_count: JOIN 数量（近似）
            - has_set_op: 有 INTERSECT/EXCEPT/UNION
            - has_group_by: 有 GROUP BY
            - has_subquery: 有子查询
            - has_window: 有窗口函数
        """
        s = (sql or "").lower()
        return {
            "join_count": s.count(" join "),
            "has_set_op": any(kw in s for kw in ("intersect", "except", "union")),
            "has_group_by": "group by" in s,
            "has_subquery": "(select" in s or "in (select" in s,
            "has_window": any(kw in s for kw in ("over(", "partition by", "row_number", "rank(")),
        }

    @staticmethod
    def _sql_structure_too_different(sql_a: str, sql_b: str, max_join_diff: int = 2) -> bool:
        """判断两条 SQL 的结构差异是否过大，不应视为同一题。

        SQL 结构差异过大意味着题目难度/类型不同：
        - 简单的单表查询 vs 复杂的多表 INTERSECT
        - 即使问题文本相似，SQL 完全不同也不应该去重合并

        参数:
            sql_a, sql_b: 两条待比较的 SQL
            max_join_diff: JOIN 数量差超过此值视为结构不同

        返回:
            True = 结构差异太大，不应合并
        """
        ka = SQLExperienceStore._sql_structure_key(sql_a)
        kb = SQLExperienceStore._sql_structure_key(sql_b)

        # JOIN 数差超过阈值
        if abs(ka["join_count"] - kb["join_count"]) > max_join_diff:
            return True
        # 一个用了集合操作（INTERSECT/EXCEPT），一个没用 → 难度不同
        if ka["has_set_op"] != kb["has_set_op"]:
            return True
        # 一个用了窗口函数，一个没用 → 题型不同
        if ka["has_window"] != kb["has_window"]:
            return True
        # 一个用了子查询，一个没用 → 复杂度不同
        if ka["has_subquery"] != kb["has_subquery"]:
            return True
        # 一个用了 GROUP BY，一个没用 → 聚合类型不同
        if ka["has_group_by"] != kb["has_group_by"]:
            return True

        return False

    def _find_similar(self, query_embedding, db_id: str, tables: frozenset,
                      new_sql: str = "") -> Optional[int]:
        """在已有经验中查找「同一道题」的索引。

        判定条件（全部满足才算同一题）:
            1. db_id 相同
            2. 表集合完全相同
            3. 问题 embedding 余弦相似度 > 0.95（高阈值）
            4. SQL 结构差异不大（JOIN 数、集合操作、子查询等）

        第 4 条保护了"同库同类但不同难度"的情况：
        "How many students have pets?"（0 JOIN，简单）
        vs
        "Find students who have both cats and dogs"（4 JOIN + INTERSECT，困难）
        → SQL 结构不同，不会被错误合并。
        """
        if self._question_embeddings is None or query_embedding is None:
            return None

        for idx, exp in enumerate(self._experiences):
            if exp.db_id != db_id:
                continue
            if frozenset(exp.tables_used) != tables:
                continue
            if exp.question_embedding is None:
                continue
            sim = float(np.dot(query_embedding, exp.question_embedding))
            if sim <= self._sim_threshold:
                continue
            # 相似度高，但还要检查 SQL 结构
            if new_sql and self._sql_structure_too_different(new_sql, exp.sql):
                continue  # 结构差异大，保留为独立经验
            return idx

        return None

    # ── 检索 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_reranker_query(question: str) -> str:
        """构建 Cross-Encoder 的 query 侧文本。"""
        return f"Query: {question}"

    @staticmethod
    def _format_reranker_doc(exp: SQLExperience) -> str:
        """构建 Cross-Encoder 的 doc 侧富文本：问题 + 表名 + SQL 片段。

        Cross-Encoder 能看到表名和 SQL 结构后，可以有效区分
        "同库不同表"和"同库同表不同问法"的经验，提升精排精度。
        """
        parts = [f"Question: {exp.question}"]
        if exp.tables_used:
            parts.append(f"Tables: {', '.join(exp.tables_used)}")
        if exp.sql:
            parts.append(f"SQL: {exp.sql[:300]}")
        if not exp.success:
            parts.append(f"Status: FAILED ({exp.error_type})")
        return "\n".join(parts)

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[SQLExperience]:
        """检索最相关的历史经验。

        两阶段检索:
            1. 问题向量召回 → 动态阈值(>0.7) 或 top-10 兜底 → top-30
            2. Cross-Encoder精排 + 结构分融合(0.8CE + 0.2结构) → top-k

        参数:
            question: 当前用户问题
            top_k: 最终返回的结果数

        返回:
            最相关的 top_k 条 SQLExperience
        """
        if not self._experiences:
            return []

        # 阶段 0: 编码当前问题（bge-m3 query 模式，带指令前缀）
        q_embedding = self._encode_query(question)[0]

        # 阶段 1: 问题语义向量召回
        q_scores = np.zeros(len(self._experiences))
        if self._question_embeddings is not None:
            q_scores = np.dot(self._question_embeddings, q_embedding)

        # 动态阈值——保留 top_score * 0.7 以上的候选
        # bge-m3 不对称编码下自匹配约 0.88，阈值 ≈ 0.62；
        # 纯二元组编码下自匹配 1.0，阈值 = 0.7，等价旧行为。
        top_score = float(q_scores.max()) if len(q_scores) > 0 else 0.0
        dynamic_threshold = max(top_score * 0.7, 0.35)
        mask = q_scores > dynamic_threshold
        top_indices = np.where(mask)[0]
        top_indices = top_indices[np.argsort(q_scores[top_indices])[::-1]]
        if len(top_indices) == 0:
            top_indices = np.argsort(q_scores)[::-1][:20]
        elif len(top_indices) < 20:
            # 候选不足时从 top-N 补充，保证 cross-encoder 有足够素材
            all_sorted = np.argsort(q_scores)[::-1]
            existing = set(top_indices)
            for idx in all_sorted:
                if len(top_indices) >= 30:
                    break
                if idx not in existing:
                    top_indices = np.append(top_indices, idx)
        else:
            # 超过 30 条则截断，控制精排阶段的计算量
            top_indices = top_indices[:30]

        # 阶段 2: Cross-Encoder 精排 + 结构分融合 → top-k
        candidates = [self._experiences[idx] for idx in top_indices]

        reranker = _get_reranker()
        if reranker and len(candidates) > top_k:
            # 2a. Cross-Encoder 批量打分（富文本 doc：问题+表名+SQL）
            query_doc = self._format_reranker_query(question)
            pairs = [[query_doc, self._format_reranker_doc(exp)] for exp in candidates]
            raw_scores = np.array(reranker.predict(pairs, batch_size=len(pairs)))

            # 2b. Softmax 归一化到 (0,1)，使量纲与结构分一致
            exp_scores = np.exp(raw_scores - raw_scores.max())  # 减 max 防溢出
            rerank_scores = exp_scores / exp_scores.sum()

            # 2c. 结构意图分
            intent = _question_structure_intent(question)
            struct_scores = np.array([
                _structure_overlap_score(intent, SQLExperienceStore._sql_structure_key(exp.sql))
                for exp in candidates
            ])

            # 2d. 加权融合: 0.8 reranker + 0.2 structure
            fused = 0.8 * rerank_scores + 0.2 * struct_scores
            ranked_idx = np.argsort(fused)[::-1][:top_k]
            return [candidates[i] for i in ranked_idx]
        else:
            # 回退：纯向量分数排序（CrossEncoder 不可用 或 候选极少）
            return candidates[:top_k]

    # ── 持久化 ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """保存所有经验到磁盘（JSON 元数据 + 回退词表）。

        向量不在磁盘持久化——加载后由 _rebuild_embeddings() 重新编码。
        回退词表 (_fallback_vocab) 需要持久化以保证查询时维度一致。
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "version": 3,
            "count": len(self._experiences),
            "experiences": [e.to_dict() for e in self._experiences],
            "fallback_vocab": self._fallback_vocab,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """从磁盘加载经验库。向量在加载后自动重建。

        加载回退词表（如果保存过），这样查询时编码维度与建库时一致。
        旧的 MiniLM 384d 向量会被 bge-m3 1024d 重建覆盖。
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._experiences = [
            SQLExperience.from_dict(d) for d in data.get("experiences", [])
        ]
        # 恢复回退词表（如果之前保存过）
        raw_vocab = data.get("fallback_vocab")
        if raw_vocab is not None:
            self._fallback_vocab = raw_vocab

        # 重建向量（会用 _fallback_vocab 或真实 encoder）
        self._rebuild_embeddings()

    def load_append(self, path: str):
        """追加加载另一个经验库文件（不覆盖已有经验）。

        用于冷启动：先 load() 主库，再 load_append() pre-built 库。
        去重由 _insert() 处理（skip_dedup=False 时走 add() 去重逻辑），
        这里为性能直接插入不走去重。
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_experiences = [
            SQLExperience.from_dict(d) for d in data.get("experiences", [])
        ]
        if not new_experiences:
            return

        # 直接插入，去重太昂贵（715条 × N条 = O(N²)）
        self._experiences.extend(new_experiences)

        # 增量更新向量矩阵
        new_questions = [e.question for e in new_experiences]
        new_embs = self._encode_documents(new_questions)
        for i, e in enumerate(new_experiences):
            if i < len(new_embs):
                e.question_embedding = new_embs[i]

        if self._question_embeddings is not None and len(new_embs) > 0:
            self._question_embeddings = np.vstack([
                self._question_embeddings, new_embs,
            ])

    def _rebuild_embeddings(self):
        """重新编码所有经验（从文件加载但没有向量时调用）。"""
        questions = [e.question for e in self._experiences]
        if questions:
            self._question_embeddings = self._encode_documents(questions)

        if self._question_embeddings is not None:
            for i, e in enumerate(self._experiences):
                if i < len(self._question_embeddings):
                    e.question_embedding = self._question_embeddings[i]

    # ── 统计 ──────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """返回经验库的统计摘要。"""
        return {
            "total": self.count,
            "success": self.success_count,
            "failure": self.count - self.success_count,
            "avg_iterations": (
                sum(e.iteration_count for e in self._experiences) / max(self.count, 1)
            ),
            "unique_db_ids": len(set(e.db_id for e in self._experiences)),
        }


# ============================================================================
# 冷启动：从 case_library 导入
# ============================================================================

def load_from_case_library(case_library_path: str, db_id: str = "",
                           skip_dedup: bool = True) -> SQLExperienceStore:
    """从 case_library_game.json 导入全部案例到 RAG，作为冷启动数据。

    冷启动阶段跳过写时去重——因为同一数据库的不同问题可能用到相同的表，
    去重会过于激进地把不同的问题合并掉。
    运行时通过 orchestrator 的 add() 会自动在写时做去重。

    参数:
        case_library_path: case_library JSON 文件路径
        db_id: 数据库标识（如 "game"），会写入每条经验的 db_id 字段
        skip_dedup: True=直接插入跳过去重（冷启动），False=按正常去重逻辑

    返回:
        已填充的 SQLExperienceStore
    """
    from agent_team.case_library_loader import CaseLibrary

    with open(case_library_path, 'r', encoding='utf-8') as f:
        lib = CaseLibrary.from_dict(json.load(f))

    store = SQLExperienceStore()
    # 批量编码以提高效率
    cases_list = [c for c in lib.cases.values() if c.question and c.sql]
    questions = [c.question for c in cases_list]
    sqls = [c.sql for c in cases_list]

    if not questions:
        return store

    # 统一走 _encode_documents，保证 bge-m3 document 模式编码
    q_embs = store._encode_documents(questions)

    for i, case in enumerate(cases_list):
        exp = SQLExperience(
            question=case.question,
            question_embedding=q_embs[i] if i < len(q_embs) else None,
            sql=case.sql,
            db_id=db_id,
            tables_used=case.tables,
            success=(case.tier == "A"),
            error_type="" if case.tier in ("A", "B") else "execution_fail",
            iteration_count=1,
        )
        if skip_dedup:
            store._insert(exp)
        else:
            store.add(exp)

    return store


# ============================================================================
# 错误分类辅助函数 —— 把 Orchestrator 的 error 字符串转为 error_type
# ============================================================================

def classify_error(error_msg: str) -> str:
    """根据错误信息字符串分类失败类型。

    用于 RAG 写入时的 error_type 字段和去重逻辑。
    """
    if not error_msg:
        return ""
    msg = error_msg.lower()
    if any(kw in msg for kw in ["syntax", "near", "语法", "括号", "quote", "引号"]):
        return "syntax"
    if any(kw in msg for kw in ["no such table", "no such column", "表", "列", "column", "table"]):
        return "schema"
    if any(kw in msg for kw in ["零行", "zero rows", "0 rows", "null", "空"]):
        return "data_coverage"
    if any(kw in msg for kw in ["ratio", "比例", "diverge", "差异", "超过"]):
        return "logic"
    return "other"
