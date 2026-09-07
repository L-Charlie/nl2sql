# nl2sql/agent_team/embedding_index.py
"""
EmbeddingIndex —— 语义匹配的向量索引引擎。

什么是 Embedding（向量嵌入）？
  Embedding 是将文本（如一句话、一个词）转换为一串固定长度的数字（向量）的技术。
  例如："活跃用户数" 可能被转换为 [0.12, -0.45, 0.78, ...] 这样一串数字。
  语义相近的文本，它们的向量在空间中也是相近的（可以通过点积或余弦相似度衡量）。

为什么用 Embedding 而不是关键词匹配？
  传统的关键词匹配只能找到字面相同的词。
  而 Embedding 可以理解语义相似性，例如：
    - 用户说"日活跃用户"，Embedding 能找到描述中包含 "DAU" 的表
    - 用户说"充值金额"，Embedding 能找到列名为 "pay_amount" 的字段
  即使字面不完全匹配，语义相近也能找到。

两种工作模式：
  1. 正常模式（推荐）：使用 sentence-transformers 库加载预训练的多语言模型，
     生成高质量的语义向量。
  2. 回退模式（Fallback）：当 sentence-transformers 未安装时，使用字符二元组
     （character bigram）编码作为替代方案，确保系统在任何环境下都能正常运行。

数据持久化：
  支持 save() 和 load() 方法，可以将构建好的索引保存到磁盘，
  下次启动时直接加载，避免重复计算 Embedding。
"""

import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextEntry:
    """
    待编码和索引的文本条目。

    每个 TextEntry 代表索引中的一个数据点，它包含：
      - id:         唯一标识符，格式为 "table:表名" 或 "col:表名.列名"
      - text:       用于生成 Embedding 的文本（表名 + 描述或列名 + 类型 + 描述）
      - entry_type: 条目类型（"table"=表、"column"=列、"business_concept"=业务概念）
      - metadata:   附加信息字典，不参与 Embedding 计算但随搜索结果返回

    举个例子：
      对于一张名为 "dws_user_dau" 的表，可能会创建这样一个 TextEntry：
        TextEntry(
          id="table:dws_user_dau",
          text="dws_user_dau 日活跃用户汇总表",
          entry_type="table",
          metadata={"table_name": "dws_user_dau", "description": "日活跃用户汇总表"}
        )
    """
    id: str
    text: str
    entry_type: str  # "table"（表）, "column"（列）, "business_concept"（业务概念）
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    单个搜索结果，包含命中的条目和相似度分数。

      - entry: 命中的 TextEntry 对象
      - score: 相似度分数，范围 [-1, 1]（使用 sentence-transformers 时）
              分数越高表示与查询越相关

    用法示例：
      for result in index.search("日活跃用户"):
          print(f"命中: {result.entry.id}, 分数: {result.score:.4f}")
    """
    entry: TextEntry
    score: float


class EmbeddingIndex:
    """
    语义匹配的向量索引。

    核心功能：
      1. 将文本条目编码为向量并存储。
      2. 接收查询文本，返回最相似的 N 个条目。
      3. 支持持久化保存和加载。

    工作模式：
      - 正常模式：使用 sentence-transformers（需要安装 sentence-transformers 库）
      - 回退模式：使用字符二元组（character bigram）编码
        （无需额外安装，但精度较低）

    使用示例：
      # 从 Schema 构建索引
      index = EmbeddingIndex()
      index.build_from_schema(schema)

      # 搜索
      results = index.search("DAU 趋势", top_k=5)

      # 保存/加载
      index.save("index.json")
      index.load("index.json")
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化 EmbeddingIndex。

        参数：
          model_name: sentence-transformers 的模型名称。
                      默认为 "paraphrase-multilingual-MiniLM-L12-v2"，
                      这是一个多语言（含中文）的轻量级模型，适合语义匹配任务。
                      仅在 sentence-transformers 可用时生效。
        """
        self.entries: list[TextEntry] = []          # 所有已注册的文本条目
        self.embeddings: Optional[np.ndarray] = None # 所有条目的向量矩阵，shape=(条目数, 向量维度)
        self.model_name = model_name                  # 模型名称
        self._model = None                            # sentence-transformers 模型实例（懒加载）
        self._fallback_vocab: Optional[dict[str, int]] = None  # 回退模式的词汇表

    @property
    def model(self):
        """
        获取 sentence-transformers 模型（懒加载模式）。

        为什么是懒加载？
          sentence-transformers 库导入和模型加载都比较耗时（可能数秒），
          所以只在第一次需要时才加载。如果后续不再使用，可以避免不必要的开销。

        回退机制：
          如果 sentence-transformers 没有安装（ImportError），
          将 _model 设为 None，后续会使用回退编码方案。
          这样即使没有安装依赖，整个系统仍能正常运行。
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                # sentence-transformers 未安装，使用回退方案
                self._model = None
        return self._model

    def build(self, entries: list[TextEntry]) -> None:
        """
        从 TextEntry 列表构建索引。

        这是核心构建方法：
          1. 保存所有条目到 self.entries。
          2. 提取所有条目的文本，批量编码为向量。
          3. 如果 sentence-transformers 可用，生成高质量的语义向量。
          4. 如果不可用，使用回退的二元组编码。

        参数：
          entries: TextEntry 对象的列表，每个对象代表一个待索引的条目。
        """
        self.entries = entries
        texts = [e.text for e in entries]

        if self.model is not None:
            # 正常模式：使用 sentence-transformers 编码
            # normalize_embeddings=True 会对向量做归一化，
            # 这样点积（dot product）就等价于余弦相似度（cosine similarity）
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        else:
            # 回退模式：使用字符二元组编码
            # 这种方式不需要任何外部依赖，纯 Python + NumPy 实现
            self.embeddings, self._fallback_vocab = self._fallback_encode(texts)

    def _fallback_encode(self, texts: list[str], vocab: Optional[dict[str, int]] = None) -> tuple[np.ndarray, dict[str, int]]:
        """
        回退编码方案：基于字符二元组（character bigram）的向量化。

        什么是字符二元组？
          将一个字符串中所有相邻的两个字符作为一组，称为一个"二元组"。
          例如 "活跃" 的二元组是 ["活跃", "跃"]（注意最后一个字符的边界处理）。
          更准确地说，对于字符串 "hello"：
            - 二元组列表：["he", "el", "ll", "lo"]

        为什么二元组能用于文本匹配？
          语义相近的词通常包含相同的字符组合。
          例如"登录"和"登陆"虽然不同，但二元组特征高度重叠。
          二元组编码将文本转换为其二元组特征的"词袋"向量，
          通过比较这些向量的相似度来衡量文本的相似度。

        实现步骤：
          1. 如果未提供词汇表，扫描所有文本构建二元组词汇表。
             词汇表记录了所有出现过的二元组及其分配的唯一索引。
          2. 对每个文本，创建一个全零向量（长度 = 词汇表大小）。
          3. 统计文本中每个二元组出现的次数，填入向量对应位置。
          4. 对向量做 L2 归一化（除以向量的模长），使不同长度的文本可比较。

        参数：
          texts: 需要编码的文本列表
          vocab: 可选的已有词汇表。如果提供，则使用它而不是重新构建。
                 这在编码查询时很有用——保证查询的词汇表和索引的词汇表一致。

        返回：
          (vectors, vocab) 元组：
            - vectors: shape=(文本数, 词汇表大小) 的归一化向量矩阵
            - vocab:   二元组 -> 索引 的映射字典
        """
        # 如果没有提供词汇表，就从头构建
        if vocab is None:
            vocab = {}
            for text in texts:
                chars = text.lower()  # 统一转小写，实现大小写不敏感
                for i in range(len(chars) - 1):
                    bigram = chars[i:i+2]  # 提取相邻两个字符
                    if bigram not in vocab:
                        vocab[bigram] = len(vocab)  # 给新二元组分配一个索引

        # 如果词汇表为空（所有文本都是空字符串），返回零向量
        if not vocab:
            return np.zeros((len(texts), 1)), vocab

        # 为每个文本构建二元组频次向量
        vectors = np.zeros((len(texts), len(vocab)))
        for idx, text in enumerate(texts):
            chars = text.lower()
            for i in range(len(chars) - 1):
                bigram = chars[i:i+2]
                if bigram in vocab:
                    vectors[idx, vocab[bigram]] += 1  # 计数加 1

        # L2 归一化：每个向量除以其模长
        # 这样不同长度的文本之间可以公平比较
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 避免除零错误
        return vectors / norms, vocab

    def _encode_query(self, query: str) -> np.ndarray:
        """
        将查询文本编码为向量，使其能与已索引的条目进行相似度比较。

        实现方式取决于当前使用的模式：
          - 正常模式：使用 sentence-transformers 模型编码
          - 回退模式：使用字符二元组编码，并使用与索引相同的词汇表

        参数：
          query: 用户的自然语言查询文本

        返回：
          shape=(1, 向量维度) 的二维 NumPy 数组（保持与 .T 点积兼容）
        """
        if self.model is not None:
            # 正常模式：用 sentence-transformers 编码
            vec = self.model.encode([query], normalize_embeddings=True)
        else:
            # 回退模式：确保词汇表存在
            # 如果词汇表丢失但有条目，从条目中重建词汇表
            vocab = self._fallback_vocab
            if vocab is None and self.entries:
                # 重建词汇表：从所有条目的文本中提取二元组
                vocab = {}
                for entry in self.entries:
                    chars = entry.text.lower()
                    for i in range(len(chars) - 1):
                        bigram = chars[i:i+2]
                        if bigram not in vocab:
                            vocab[bigram] = len(vocab)
                self._fallback_vocab = vocab
            vec, _ = self._fallback_encode([query], vocab=vocab)
        return vec.reshape(1, -1)

    def search(self, query: str, top_k: int = 10, filter_type: Optional[str] = None) -> list[SearchResult]:
        """
        搜索与查询文本最相似的条目。

        搜索流程：
          1. 将查询文本编码为向量（与索引使用相同的编码方式）。
          2. 计算查询向量与所有条目向量的点积（dot product）。
             由于向量都是归一化的，点积等价于余弦相似度。
             结果是一个分数数组，每个条目一个值，范围 [-1, 1]。
          3. 按分数降序排列（分数越高越相似）。
          4. 可选地按类型过滤（如只返回 "table" 类型）。
          5. 返回分数最高的前 top_k 个结果。

        参数：
          query:       用户的自然语言查询文本
          top_k:       返回的最多结果数量
          filter_type: 可选的类型过滤器。
                       "table" = 只返回表条目
                       "column" = 只返回列条目
                       None = 不限制类型

        返回：
          SearchResult 列表，每个包含命中的 TextEntry 和相似度分数，
          按分数从高到低排列。
        """
        if not self.entries or self.embeddings is None:
            return []

        # 将查询转为向量
        query_vec = self._encode_query(query)

        # 计算查询向量与所有条目向量的点积
        # embeddings 的 shape 是 (条目数, 向量维度)
        # query_vec.T 的 shape 是 (向量维度, 1)
        # 结果 scores 的 shape 是 (条目数,)
        scores = np.dot(self.embeddings, query_vec.T).flatten()

        # 按分数从高到低排序，argsort 返回排序后的索引
        sorted_indices = np.argsort(scores)[::-1]

        # 构建结果列表，同时应用类型过滤
        results = []
        for idx in sorted_indices:
            entry = self.entries[idx]
            if filter_type and entry.entry_type != filter_type:
                continue  # 跳过不符合类型过滤条件的条目
            score = float(scores[idx])
            results.append(SearchResult(entry=entry, score=score))
            if len(results) >= top_k:
                break

        return results

    def build_from_schema(self, schema: list[dict], business_terms: Optional[dict] = None) -> None:
        """
        从 schema.json 格式的数据构建索引。

        这个方法会为 schema 中的每张表、每个列创建对应的 TextEntry，
        并可以选择性地添加业务术语的定义条目。

        参数：
          schema: 数据库 Schema 的列表，每个元素是一个表描述字典。
                  每张表应包含：
                    - table_name:        表名
                    - table_description: 表描述（可选）
                    - columns:           列列表，每个列包含：
                      - col:         列名
                      - description:  列描述（可选）
                      - type:         数据类型（可选）
          business_terms: 可选，业务术语词典 { 概念名: 定义 }。
                          这些术语会作为独立的 "business_concept" 条目加入索引，
                          帮助系统理解常见的业务词汇。

        构建的条目类型：
          - "table"（表条目）：   文本 = "表名 表描述"，用于表级召回
          - "column"（列条目）：  文本 = "列名 类型 描述"，用于列级召回
          - "business_concept"（业务概念）：文本 = "概念名 定义"，用于理解业务术语
        """
        entries: list[TextEntry] = []

        for table in schema:
            table_name = table["table_name"]
            table_desc = table.get("table_description", "")

            # =================================================================
            # 创建表条目
            # 将表名和描述合并作为索引文本，这样用户问"日活跃用户"时，
            # 能匹配到描述中包含"日活跃用户"的表。
            # =================================================================
            entries.append(TextEntry(
                id=f"table:{table_name}",
                text=f"{table_name} {table_desc}",
                entry_type="table",
                metadata={"table_name": table_name, "description": table_desc},
            ))

            # =================================================================
            # 创建列条目
            # 每列一条，文本包含列名 + 数据类型 + 列描述。
            # 这样用户问"金额"时，即使列名叫 pay_amt 但描述写的是"充值金额"，
            # 也能被匹配到。
            # =================================================================
            for col in table.get("columns", []):
                col_name = col["col"]
                col_desc = col.get("description", "")
                col_type = col.get("type", "")

                entries.append(TextEntry(
                    id=f"col:{table_name}.{col_name}",
                    text=f"{col_name} {col_type} {col_desc}",
                    entry_type="column",
                    metadata={
                        "table_name": table_name,
                        "col": col_name,
                        "type": col_type,
                        "description": col_desc,
                    },
                ))

        # =====================================================================
        # 添加业务概念条目（可选）
        # 例如：{ "DAU": "日活跃用户数（Daily Active Users）" }
        # 这些条目帮助系统理解常见的业务缩写和术语。
        # =====================================================================
        if business_terms:
            for concept_name, definition in business_terms.items():
                entries.append(TextEntry(
                    id=f"concept:{concept_name}",
                    text=f"{concept_name} {definition}",
                    entry_type="business_concept",
                    metadata={"concept": concept_name, "definition": definition},
                ))

        self.build(entries)

    def save(self, path: str) -> None:
        """
        将索引保存到磁盘文件。

        保存内容包括：
          - entries:    所有 TextEntry 对象
          - embeddings: 向量矩阵（转为 Python 列表存储）
          - model_name: 使用的模型名称
          - fallback_vocab: 回退模式的词汇表（如果使用了回退模式）

        这样可以下次启动时直接加载，避免重新编码。
        """
        data = {
            "entries": [
                {
                    "id": e.id,
                    "text": e.text,
                    "entry_type": e.entry_type,
                    "metadata": e.metadata,
                }
                for e in self.entries
            ],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "model_name": self.model_name,
            "fallback_vocab": self._fallback_vocab,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        从磁盘文件加载索引。

        恢复内容包括：
          - entries:    所有 TextEntry 对象
          - embeddings: 向量矩阵（从列表恢复为 NumPy 数组）
          - model_name: 模型名称
          - fallback_vocab: 回退模式的词汇表

        注意：
          保存和加载使用的 model_name 必须一致，否则编码方式不匹配，
          会导致搜索结果不准确。
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.model_name = data.get("model_name", self.model_name)
        self.entries = [
            TextEntry(
                id=e["id"],
                text=e["text"],
                entry_type=e["entry_type"],
                metadata=e.get("metadata", {}),
            )
            for e in data["entries"]
        ]

        if data.get("embeddings"):
            self.embeddings = np.array(data["embeddings"])

        # 恢复回退模式的词汇表
        raw_vocab = data.get("fallback_vocab")
        if raw_vocab is not None:
            # JSON 的键总是字符串，对于二元组词汇表来说键就是二元组本身，所以直接使用
            self._fallback_vocab = raw_vocab

    @classmethod
    def from_files(cls, schema_path: str, knowledge_path: Optional[str] = None) -> "EmbeddingIndex":
        """
        工厂方法：从文件路径构建索引。

        便捷的构建方式，一步完成：
          1. 读取 schema.json 文件
          2. 可选地读取业务知识文件，提取业务术语
          3. 构建索引

        参数：
          schema_path:   schema.json 文件的路径
          knowledge_path: 可选的业务知识文件路径。
                         支持两种格式：
                           - common_knowledge.md
                           - spider_agent.txt
                         从这些文件中提取业务术语定义。

        使用示例：
          index = EmbeddingIndex.from_files("schema.json", "common_knowledge.md")
        """
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        business_terms = None
        if knowledge_path:
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 从常见的知识文件格式中提取业务术语定义
                business_terms = cls._extract_business_terms(content)
            except FileNotFoundError:
                pass

        idx = cls()
        idx.build_from_schema(schema, business_terms=business_terms)
        return idx

    @staticmethod
    def _extract_business_terms(text: str) -> dict[str, str]:
        """
        从知识文本中提取业务术语的定义。

        支持的格式：
          - DAO：日活跃用户
          - 留存: 次日留存率
          或
          - DAU: 日活跃用户数

        查找模式：
          遍历 known_terms 列表，在文本中搜索类似 "- 术语: 定义" 的行，
          提取术语的定义文本。

        参数：
          text: 包含术语定义的文本内容

        返回：
          术语名 -> 定义 的字典
        """
        terms = {}
        known_terms = ["DAU", "留存", "新进", "回流", "活跃", "流水"]

        for term in known_terms:
            # 匹配类似 "- DAU：日活跃用户数" 或 "DAU: 日活跃用户数" 的行
            pattern = rf"-?\s*{term}[：:]\s*(.+?)(?=\n|$)"
            match = re.search(pattern, text)
            if match:
                terms[term] = match.group(1).strip()

        return terms
