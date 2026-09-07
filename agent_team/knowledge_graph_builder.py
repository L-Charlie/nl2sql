# ============================================================================
# knowledge_graph_builder.py — 知识图谱构建器
# ============================================================================
#
# 什么是知识图谱（Knowledge Graph）？
#   知识图谱是一种用"节点（Node）"和"边（Edge）"来表示实体及其关系的
#   数据结构。想象一个社交网络：人是"节点"，朋友关系是"边"。
#   在我们的 NL2SQL 场景中：
#     - 表和列是"节点"
#     - "属于某张表"、"外键关联"等是"边"
#
# 为什么知识图谱对 NL2SQL 有帮助？
#   知识图谱可以让 LLM 做"关系推理"（Relational Reasoning）：
#   1. 找到某列后，可以沿"属于表"的边找到它所属的表
#   2. 知道两张表之间有"潜在外键"边后，LLM 知道可以做 JOIN
#   3. 语义标注（"多值字段"、"危险字段"）让 LLM 知道特殊的处理方式
#   4. 业务概念节点（"DAU"、"留存"）与相关表/列建立关联
#
# 本构建器的完整构建流水线（由 build() 方法驱动）：
#   Step 1: _add_table_nodes()      — 为每张表创建一个节点
#   Step 2: _add_column_nodes()     — 为每列创建一个节点，连到所属表
#   Step 3: _add_fk_edges()         — 按约束证据和保守规则构建分级关系
#   Step 4: _add_semantic_annotations() — 标注多值字段、危险字段
#   Step 5: _extract_business_concepts() — 提取业务概念节点
#
# 技术选型：
#   使用 NetworkX 的 MultiDiGraph（有向多重图）：
#   - Multi = 两个节点之间可以有多条边（比如两张表之间可能有多个外键）
#   - Di = 有向边（边有方向，从"表 -> 列"表示"包含"，从"列 -> 列"表示"外键"）
# ============================================================================

import json          # 用于图数据的序列化和反序列化
import re            # 正则表达式，用于表名解析和模式匹配
import networkx as nx  # 业界标准的 Python 图论库
from typing import Optional  # 类型注解：表示返回值可能为 None


class KnowledgeGraphBuilder:
    """知识图谱构建器——从 Schema JSON 和 spider_agent.txt 规则构建语义知识图谱。

    知识点（给新手）：
      - 本类不依赖任何 LLM，纯粹通过规则和结构分析来构建图谱
      - 构建过程是完全确定性的（Deterministic）：同样的输入永远产生同样的输出
      - 构建结果是一个 NetworkX MultiDiGraph 对象

    用法示例：
      builder = KnowledgeGraphBuilder(schema_json)
      graph = builder.build()  # 返回 nx.MultiDiGraph

      # 构建完成后，可以通过 graph.nodes(data=True) 遍历所有节点
      # 通过 graph.edges(data=True) 遍历所有边
    """

    # ========================================================================
    # DATE_COLUMN_PATTERNS（日期列检测模式）
    # ========================================================================
    # 这是一个正则表达式列表，用于判断某列是否为"日期/时间"列。
    # 检测时会将列名 + 列描述 + 列类型拼接成一个字符串，
    # 然后用这些模式依次匹配。
    #
    # 为什么需要这个？
    #   NL2SQL 中时间字段是最常用的过滤条件。如果系统能够自动识别
    #   哪些列是时间列，就可以在生成 SQL 时自动加上正确的时间范围条件。
    #
    # 包含的模式：
    #   中文关键词：日期、时间
    #   英文关键词：date、time
    #   游戏数仓标准字段名：dtstatdate、dteventtime、tdbank_imp_date、
    #                       dregdate、iregdate
    DATE_COLUMN_PATTERNS = [
        r"日期", r"时间", r"date", r"time",
        r"dtstatdate", r"dteventtime",
        r"tdbank_imp_date", r"dregdate", r"iregdate",
    ]

    # ========================================================================
    # CONCEPT_PATTERNS（业务概念提取模式）
    # ========================================================================
    # 这些正则用于从 spider_agent.txt 规则文件中提取业务概念定义。
    # spider_agent.txt 是一个文本文件，通常包含游戏业务的规则说明，
    # 比如 "DAU：日活跃用户数 = COUNT(DISTINCT vplayerid)" 之类的定义。
    #
    # 正则结构解释（以第一条为例）：
    #   r"-\\s*(DAU|日活跃用户数)[：:]\\s*(.+?)(?=\\n|$)"
    #   匹配以 "- " 开头，后跟 "DAU" 或 "日活跃用户数"，
    #   然后是冒号，再捕获后面的定义文本（直到换行或字符串结束）
    CONCEPT_PATTERNS = [
        (r"-\s*(DAU|日活跃用户数)[：:]\s*(.+?)(?=\n|$)", "DAU"),
        (r"-\s*(留存)[：:]\s*(.+?)(?=\n|$)", "留存"),
        (r"-\s*(新进)[：:]\s*(.+?)(?=\n|$)", "新进"),
        (r"-\s*(回流)[：:]\s*(.+?)(?=\n|$)", "回流"),
    ]

    EDGE_CONFIDENCE = {
        "DECLARED_FK": 1.00,
        "SCHEMA_FK": 0.95,
        "INFERRED_FK": 0.80,
        "WEAK_SIMILARITY": 0.10,
    }
    GENERIC_SHARED_COLUMNS = {
        "id", "name", "status", "type", "date", "text", "description",
        "created_at", "updated_at", "create_time", "update_time",
    }

    def __init__(self, schema: list[dict], spider_rules_text: str = ""):
        """初始化知识图谱构建器。

        参数:
            schema: Schema JSON 解析出的列表，每个元素包含
                    table_name、columns（列表）、table_description 等字段
            spider_rules_text: spider_agent.txt 文件的文本内容（可选），
                               用于提取业务概念和语义规则。
                               如果留空，则跳过概念提取和语义标注步骤。
        """
        self.schema = schema
        self.spider_rules_text = spider_rules_text
        # 初始化一个空的有向多重图（MultiDiGraph）
        # MultiDiGraph 允许：
        #   1. 两个节点之间有**多条**边（Multi）
        #   2. 边有方向，从源节点指向目标节点（Di）
        self.graph = nx.MultiDiGraph()

    # ====================================================================
    # Step 0: 构建入口（build）
    # ====================================================================
    # 这是外部调用的唯一入口方法，按顺序执行 5 个构建步骤。

    def build(self) -> nx.MultiDiGraph:
        """构建完整的知识图谱并返回。

        按顺序执行 5 个构建步骤：
          1. 创建所有表节点，附带层的元数据（DWS/DWD/DIM 等）
          2. 创建所有列节点，并连接到对应的表
          3. 按数据库约束、Schema 标记和保守命名规则添加分级关系
          4. 添加语义标注（多值字段、危险字段）
          5. 提取业务概念节点（DAU/留存/新进/回流）

        返回:
            构建完成的 NetworkX MultiDiGraph 对象
        """
        # 每次调用 build() 都重新创建一个空图，保证幂等性
        self.graph = nx.MultiDiGraph()
        self._add_table_nodes()               # Step 1
        self._add_column_nodes()              # Step 2
        self._add_fk_edges()                  # Step 3
        self._add_semantic_annotations()      # Step 4
        self._extract_business_concepts()     # Step 5
        return self.graph

    # ====================================================================
    # Step 1: 创建表节点（_add_table_nodes）
    # ====================================================================

    def _add_table_nodes(self) -> None:
        """Step 1: 遍历 Schema 中的每张表，创建对应的图节点。

        每个表节点包含以下属性：
          - node_type: 固定为 "table"（用于区分表节点和列节点）
          - layer: 数据分层（DWS / DWD / DIM / ODS / ADS 或无）
          - gamecode: 游戏编码（从表名中提取）
          - suffix_type: 表名后缀（_di / _df / _hi / _nf 或无）
          - description: 表的中文描述
          - column_count: 表中的列数

        为什么解析表名？
          游戏数仓的表名遵循约定结构：{层}_{游戏编码}_{描述}_{后缀}
          比如：dws_jordass_mode_roundrecord_di
          解析出：layer=dws, gamecode=jordass, suffix=di
          这些元数据在后续生成 SQL 时非常重要（比如 _di 和 _df 的日期处理不同）。
        """
        for table in self.schema:
            name = table["table_name"]
            # 解析表名，获取层、游戏编码、后缀类型
            layer, gamecode, suffix_type = self._parse_table_name(name)
            # 将表作为"节点"添加到图中
            # add_node 的第一个参数是节点的唯一 ID（用表名）
            # 后面的 key=value 是节点的属性，可以通过 data=True 查询到
            self.graph.add_node(
                name,
                node_type="table",              # 标记为"表"节点
                layer=layer,                    # 数据分层
                gamecode=gamecode,              # 游戏编码
                suffix_type=suffix_type,        # 表后缀类型
                description=table.get("table_description", ""),
                column_count=len(table.get("columns", [])),
            )

    def _parse_table_name(self, name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """解析游戏数仓的表名，提取层、游戏编码和后缀。

        游戏数仓表名格式：{layer}_{gamecode}_{description}_{suffix}
        示例解析：
          输入: "dws_jordass_mode_roundrecord_di"
          输出: layer="DWS", gamecode="jordass", suffix="di"

        解析规则：
          1. layer（数据分层）：取第一个下划线前的部分，如果在
             （dwd、dws、dim、ods、ads）中则识别为层名，否则为 None
          2. suffix（表后缀）：取最后一部分，如果在（di、df、hi、nf）中
             则识别为后缀，否则为 None
          2. gamecode（游戏编码）：取第一部分（如果识别为层）或
             第一部分（如果未识别到层），通常是游戏项目的英文代号

        参数:
            name: 原始表名字符串

        返回:
            (layer, gamecode, suffix) 三元组，可能为 None
        """
        # 按下划线分割表名
        # 例如 "dws_jordass_mode_roundrecord_di" -> ["dws", "jordass", "mode", "roundrecord", "di"]
        parts = name.split("_")

        # 判断第一部分是否为数据分层标识
        # 如果是 DWD/DWS/DIM/ODS/ADS，转为大写作为层名
        layer = parts[0].upper() if parts and parts[0].lower() in ("dwd", "dws", "dim", "ods", "ads") else None

        # 判断最后一部分是否为后缀
        # 如果是 di/df/hi/nf 之一，作为后缀类型
        suffix = parts[-1] if parts and parts[-1].lower() in ("di", "df", "hi", "nf") else None

        # 游戏编码：通常在第二部分（紧跟在层名之后）
        # 如果层名存在，取 parts[1]；否则取 parts[0] 作为默认
        gamecode = parts[1] if len(parts) > 1 else None

        return layer, gamecode, suffix

    # ====================================================================
    # Step 2: 创建列节点（_add_column_nodes）
    # ====================================================================

    def _add_column_nodes(self) -> None:
        """Step 2: 为每张表的每列创建一个列节点，并连接到对应的表节点。

        列节点的 ID 格式："{table_name}.{col_name}"
        例如：dws_jordass_login_di.vplayerid

        每个列节点包含以下属性：
          - node_type: 固定为 "column"
          - data_type: 列的数据类型（VARCHAR、INT、BIGINT 等）
          - description: 列的描述/注释
          - is_date_field: 布尔值，标记是否为时间/日期列

        边的关系：
          从"表节点"指向"列节点"，关系标签为 "HAS_COLUMN"
          例如：dws_jordass_login_di ->[HAS_COLUMN]-> dws_jordass_login_di.vplayerid

        为什么列名要加上表名前缀？
          不同表的列名可能相同（比如都有 vplayerid）。
          如果不加前缀，节点的 ID 会冲突。
          使用 "表名.列名" 作为唯一 ID 可以避免命名冲突。
        """
        for table in self.schema:
            table_name = table["table_name"]
            for col in table.get("columns", []):
                col_name = col["col"]
                # 列节点的全局唯一 ID = "表名.列名"
                node_id = f"{table_name}.{col_name}"
                # 添加列节点，附带数据类型的属性
                self.graph.add_node(
                    node_id,
                    node_type="column",                    # 标记为"列"节点
                    data_type=col.get("type", "unknown"),   # 数据类型
                    description=col.get("description", ""), # 列注释/描述
                    # 自动判断是否为日期列（通过列名+描述+类型匹配 DATE_COLUMN_PATTERNS）
                    is_date_field=self._is_date_column(
                        col_name,
                        col.get("description", ""),
                        col.get("type", ""),
                    ),
                )
                # 创建"表 -> 列"的边，表示该列属于这张表
                # 有向边的方向：表 -> 列
                self.graph.add_edge(table_name, node_id, relationship="HAS_COLUMN")

    def _is_date_column(self, col_name: str, description: str, col_type: str) -> bool:
        """判断一个列是否属于时间/日期类型。

        判断方法：
          将"列名 + 描述 + 类型"拼接后，用 DATE_COLUMN_PATTERNS 中的
          正则表达式依次匹配。只要有一个模式匹配，就判定为日期列。

        为什么需要这个方法？
          在 SQL 生成中，时间字段经常需要特殊的函数处理（如 DATE_FORMAT、
          DATE_ADD 等）。提前标记日期列，可以让后续的编码模式更精准。

        参数:
            col_name: 列名
            description: 列的中文描述/注释
            col_type: 列的数据类型

        返回:
            如果匹配到任何日期模式，返回 True；否则返回 False
        """
        # 将三个信息拼接成一个大字符串，便于正则匹配
        combined = f"{col_name} {description} {col_type}".lower()
        for pattern in self.DATE_COLUMN_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return True
        return False

    # ====================================================================
    # Step 3: 发现外键关系（_add_fk_edges）
    # ====================================================================

    def _add_fk_edges(self) -> None:
        """Build typed, scored relationships from explicit and inferred evidence."""
        table_lookup = {table["table_name"].casefold(): table for table in self.schema}
        column_lookup = {
            table["table_name"].casefold(): {
                column["col"].casefold(): column for column in table.get("columns", [])
            }
            for table in self.schema
        }
        relationships: dict[tuple[str, str, str, str], dict] = {}

        # Database adapters can preserve real catalog constraints structurally.
        for source_table in self.schema:
            source_name = source_table["table_name"]
            for foreign_key in source_table.get("foreign_keys", []):
                self._record_relationship(
                    relationships,
                    source_name,
                    str(foreign_key.get("from", "")),
                    str(foreign_key.get("to_table", "")),
                    str(foreign_key.get("to_column", "")),
                    "DECLARED_FK",
                    ["database_foreign_key"],
                    table_lookup,
                    column_lookup,
                )

        # Manually supplied schemas can use description="FK->users.id".
        for source_table in self.schema:
            source_name = source_table["table_name"]
            for column in source_table.get("columns", []):
                for target in re.findall(r"FK->([^\s,;]+)", column.get("description", "")):
                    if "." not in target:
                        continue
                    target_table, target_column = target.rsplit(".", 1)
                    self._record_relationship(
                        relationships,
                        source_name,
                        column["col"],
                        target_table,
                        target_column,
                        "SCHEMA_FK",
                        ["schema_fk_marker"],
                        table_lookup,
                        column_lookup,
                    )

        # Conservative convention inference: user_id -> users.id is accepted only
        # when the target is a key and both column types are compatible.
        for source_table in self.schema:
            source_name = source_table["table_name"]
            for source_column in source_table.get("columns", []):
                source_column_name = source_column["col"]
                source_folded = source_column_name.casefold()
                if not source_folded.endswith("_id"):
                    continue
                entity_name = source_folded[:-3]
                for target_table in self.schema:
                    target_name = target_table["table_name"]
                    if target_name == source_name:
                        continue
                    normalized_table = self._singularize(target_name.casefold())
                    if self._normalize_identifier(entity_name) != self._normalize_identifier(normalized_table):
                        continue
                    for target_column in target_table.get("columns", []):
                        target_folded = target_column["col"].casefold()
                        if target_folded not in {"id", source_folded}:
                            continue
                        if not self._is_key_column(target_column):
                            continue
                        if not self._types_compatible(source_column, target_column):
                            continue
                        self._record_relationship(
                            relationships,
                            source_name,
                            source_column_name,
                            target_name,
                            target_column["col"],
                            "INFERRED_FK",
                            ["identifier_naming", "target_key", "type_compatible"],
                            table_lookup,
                            column_lookup,
                        )

        # Same-name columns are retained only as weak diagnostics. Generic fields
        # such as id/name/status are discarded entirely and can never expand tables.
        col_index: dict[str, list[tuple[str, dict]]] = {}
        for table in self.schema:
            table_name = table["table_name"]
            for col in table.get("columns", []):
                folded_name = col["col"].casefold()
                if folded_name in self.GENERIC_SHARED_COLUMNS or folded_name.startswith("other_"):
                    continue
                col_index.setdefault(folded_name, []).append((table_name, col))

        for entries in col_index.values():
            if len(entries) >= 2:
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        left_table, left_column = entries[i]
                        right_table, right_column = entries[j]
                        if not self._types_compatible(left_column, right_column):
                            continue
                        self._record_relationship(
                            relationships,
                            left_table,
                            left_column["col"],
                            right_table,
                            right_column["col"],
                            "WEAK_SIMILARITY",
                            ["same_column_name", "type_compatible"],
                            table_lookup,
                            column_lookup,
                        )

        for relationship in relationships.values():
            source_node = f"{relationship['from_table']}.{relationship['from_column']}"
            target_node = f"{relationship['to_table']}.{relationship['to_column']}"
            self.graph.add_edge(
                source_node,
                target_node,
                relationship="FK_RELATIONSHIP",
                **relationship,
            )

    def _record_relationship(
        self,
        relationships: dict,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
        edge_type: str,
        evidence: list[str],
        table_lookup: dict,
        column_lookup: dict,
    ) -> None:
        source_table_obj = table_lookup.get(source_table.casefold())
        target_table_obj = table_lookup.get(target_table.casefold())
        if source_table_obj is None or target_table_obj is None:
            return
        source_column_obj = column_lookup[source_table.casefold()].get(source_column.casefold())
        target_column_obj = column_lookup[target_table.casefold()].get(target_column.casefold())
        if source_column_obj is None or target_column_obj is None:
            return

        canonical_source_table = source_table_obj["table_name"]
        canonical_target_table = target_table_obj["table_name"]
        canonical_source_column = source_column_obj["col"]
        canonical_target_column = target_column_obj["col"]
        key = (
            canonical_source_table,
            canonical_source_column,
            canonical_target_table,
            canonical_target_column,
        )
        confidence = self.EDGE_CONFIDENCE[edge_type]
        current = relationships.get(key)
        if current and current["confidence"] >= confidence:
            return
        relationships[key] = {
            "from_table": canonical_source_table,
            "from_column": canonical_source_column,
            "to_table": canonical_target_table,
            "to_column": canonical_target_column,
            "edge_type": edge_type,
            "confidence": confidence,
            "evidence": evidence,
        }

    @staticmethod
    def _is_key_column(column: dict) -> bool:
        description = str(column.get("description", "")).upper()
        return "PK" in description or "UNIQUE" in description

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    @staticmethod
    def _singularize(value: str) -> str:
        if value.endswith("ies") and len(value) > 3:
            return value[:-3] + "y"
        if value.endswith("s") and not value.endswith("ss"):
            return value[:-1]
        return value

    @staticmethod
    def _types_compatible(left: dict, right: dict) -> bool:
        def family(column: dict) -> str:
            value = str(column.get("type", "")).upper()
            if any(token in value for token in ("INT", "NUM", "DEC", "REAL", "FLOAT", "DOUBLE")):
                return "numeric"
            if any(token in value for token in ("CHAR", "TEXT", "CLOB", "STRING")):
                return "text"
            if any(token in value for token in ("DATE", "TIME")):
                return "temporal"
            return value or "unknown"

        left_family = family(left)
        right_family = family(right)
        return "unknown" in {left_family, right_family} or left_family == right_family

    # ====================================================================
    # Step 4: 添加语义标注（_add_semantic_annotations）
    # ====================================================================

    def _add_semantic_annotations(self) -> None:
        """Step 4: 从 spider_agent.txt 中解析语义规则，标注到图节点上。

        这一步做什么？
          spider_agent.txt 是游戏项目的规则文件，里面包含了很多
          数据使用的约定和规范。本方法从中提取两类信息：
            1. 多值字段（Multi-value fields）：用分隔符拼接了多个值的字段
            2. 危险字段（Dangerous fields）：语义不明确的预留字段

        多值字段的标注方法：
          在 spider_agent.txt 中搜索"多值字段"、"分隔符"、"multi value"
          等关键词，找到对应的字段名，然后在图节点上标记 is_multi_value=True。

        危险字段的标注方法：
          硬编码了一个列表（itemp1、itemp2、vtemp1、vtemp2），这些是游戏
          数仓中常见的预留字段，语义依赖于上下文（比如 platid 的值），
          新手很容易误用。
        """
        # 如果没有提供 spider_agent.txt 内容，跳过语义标注
        if not self.spider_rules_text:
            return

        # ---------------------------------------------------------------
        # 标注多值字段（Multi-Value Fields）
        # ---------------------------------------------------------------
        # 在 spider_agent.txt 中搜索类似"多值字段 buttontypes"这样的行
        mv_pattern = re.compile(
            r'(?:多值字段|分隔符|multi.value).*?[`"\\]]?(\w+)[`"\\]]?',
            re.IGNORECASE,
        )
        for match in mv_pattern.finditer(self.spider_rules_text):
            field_name = match.group(1)  # 提取字段名
            # 遍历图中所有节点，找到同名的列节点
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get("node_type") == "column" and node.endswith(f".{field_name}"):
                    # 标记该列为"多值字段"
                    self.graph.nodes[node]["is_multi_value"] = True

        # ---------------------------------------------------------------
        # 标注危险/预留字段（Dangerous / Reserved Fields）
        # ---------------------------------------------------------------
        # 这些字段在不同表中含义不同，使用前需要确认上下文
        # itemp1, itemp2 是整数型预留字段
        # vtemp1, vtemp2 是字符串型预留字段
        dangerous_fields = ["itemp1", "itemp2", "vtemp1", "vtemp2"]
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == "column":
                col_name = node.split(".")[-1]
                if col_name in dangerous_fields:
                    # 标记为危险字段，并附加警告信息
                    self.graph.nodes[node]["is_dangerous"] = True
                    self.graph.nodes[node]["warning"] = (
                        "预留字段，语义依赖于 platid 等上下文"
                    )

    # ====================================================================
    # Step 5: 提取业务概念（_extract_business_concepts）
    # ====================================================================

    def _extract_business_concepts(self) -> None:
        """Step 5: 从 spider_agent.txt 中提取业务概念，作为独立的图节点。

        为什么需要业务概念节点？
          知识图谱中的表和列是"数据结构层面"的实体。
          但 NL2SQL 还需要"业务层面"的知识——比如"DAU"这个概念
          涉及哪些表和列？有了业务概念节点，后续可以建立
          "概念 -> 相关表 -> 相关列"的多跳关联。

        提取方式：
          如果提供了 spider_agent.txt，本方法会创建 4 个预定义的
          业务概念节点（DAU、留存、新进、回流），每个节点包含概念名和定义。
          未来可以扩展为从 spider_agent.txt 中动态解析更多概念。

        节点 ID 格式：
          "concept:{concept_name}"，例如 "concept:DAU"
          使用 "concept:" 前缀可以避免与表名/列名冲突。
        """
        # 如果没有提供 spider_agent.txt 内容，跳过概念提取
        if not self.spider_rules_text:
            return

        # 预定义的 4 个核心游戏业务概念及其定义
        concepts = {
            "DAU": "日活跃用户数",
            "留存": "以次留为例，当天活跃第二天依然活跃的用户定义为次留用户",
            "新进": "新注册用户",
            "回流": "历史活跃但在前N天未活跃，当天重新活跃的用户",
        }

        # 为每个概念创建一个"业务概念"类型的节点
        for concept_name, definition in concepts.items():
            node_id = f"concept:{concept_name}"  # 全局唯一 ID
            self.graph.add_node(
                node_id,
                node_type="business_concept",  # 节点类型：业务概念
                name=concept_name,              # 概念名称
                definition=definition,          # 概念定义
            )

    # ====================================================================
    # 查询方法（Query Methods）
    # ====================================================================
    # 以下方法用于在构建完成的图谱上进行查询，供 KnowledgeKeeper
    # 和其他组件使用。

    def get_tables_by_keyword(self, keyword: str) -> list[str]:
        """根据关键词搜索相关表。

        遍历图中所有"表"节点，检查其名称或描述中是否包含关键词。
        这是一个简单的字符串包含匹配，不涉及语义理解。

        参数:
            keyword: 搜索关键词（不区分大小写）

        返回:
            匹配到的表名列表
        """
        results = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == "table":
                desc = attrs.get("description", "")
                if keyword.lower() in node.lower() or keyword.lower() in desc.lower():
                    results.append(node)
        return results

    def get_join_paths(self, table_a: str, table_b: str) -> list[dict]:
        """Return scored FK relationships connecting two tables."""
        paths = []
        requested = {table_a.casefold(), table_b.casefold()}
        for from_node, to_node, attrs in self.graph.edges(data=True):
            if attrs.get("relationship") != "FK_RELATIONSHIP":
                continue
            connected = {
                str(attrs.get("from_table", "")).casefold(),
                str(attrs.get("to_table", "")).casefold(),
            }
            if connected == requested:
                paths.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    **{
                        key: attrs[key]
                        for key in (
                            "from_table", "from_column", "to_table", "to_column",
                            "edge_type", "confidence", "evidence",
                        )
                    },
                })
        return sorted(paths, key=lambda item: item["confidence"], reverse=True)

    def get_column_info(self, table_name: str, column_name: str) -> Optional[dict]:
        """获取某个列的完整属性信息。

        通过节点 ID 格式 "{table_name}.{column_name}" 直接查找节点，
        返回该节点的所有属性字典。

        参数:
            table_name: 表名
            column_name: 列名

        返回:
            如果找到，返回一个字典包含该列的所有属性（data_type、description 等）；
            如果找不到，返回 None
        """
        node_id = f"{table_name}.{column_name}"
        if node_id in self.graph.nodes:
            # dict() 将 NetworkX 的属性视图转为普通字典，方便使用
            return dict(self.graph.nodes[node_id])
        return None

    # ====================================================================
    # 导入/导出方法（Export / Import）
    # ====================================================================
    # NetworkX 的图可以通过 JSON 持久化到磁盘，避免每次启动都要重新构建。
    # 对于大型 Schema，重构知识图谱可能需要几百毫秒，导出后可以直接
    # 从 JSON 加载，大幅减少 Phase 0 的时间。

    def export_json(self, output_path: str) -> None:
        """将知识图谱导出为 JSON 文件，用于持久化存储。

        序列化格式：
          {
            "nodes": [
              {"id": "table_name", "node_type": "table", "layer": "DWS", ...},
              {"id": "table_name.col_name", "node_type": "column", ...},
              ...
            ],
            "edges": [
              {"from": "table_name", "to": "table_name.col_name",
               "relationship": "HAS_COLUMN", ...},
              ...
            ]
          }

        参数:
            output_path: 输出 JSON 文件的路径
        """
        data = {
            "nodes": [
                # 将每个节点的 ID 提取到 "id" 字段中，其余属性平铺
                {"id": node, **attrs}
                for node, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                # 将每条边的源和目标提取到 "from"/"to" 字段中
                {"from": u, "to": v, **attrs}
                for u, v, attrs in self.graph.edges(data=True)
            ],
        }
        # 写入 JSON，设置 ensure_ascii=False 以保留中文
        # indent=2 让输出格式可读
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> nx.MultiDiGraph:
        """从 JSON 文件加载知识图谱（类方法，无需先创建实例）。

        这是 export_json() 的逆操作：
          1. 读取 JSON 文件
          2. 创建一个新的 MultiDiGraph
          3. 遍历 JSON 中的 nodes 列表，用 node_data 作为属性添加节点
          4. 遍历 JSON 中的 edges 列表，用 edge_data 作为属性添加边

        参数:
            path: JSON 文件的路径

        返回:
            重建的 NetworkX MultiDiGraph 对象
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = nx.MultiDiGraph()
        # 重建所有节点
        for node_data in data["nodes"]:
            node_id = node_data.pop("id")  # 取出 ID，剩下的都是属性
            graph.add_node(node_id, **node_data)

        # 重建所有边
        for edge_data in data["edges"]:
            u = edge_data.pop("from")  # 源节点
            v = edge_data.pop("to")    # 目标节点
            graph.add_edge(u, v, **edge_data)

        return graph

    @classmethod
    def from_files(cls, schema_path: str, rules_path: str = "") -> "KnowledgeGraphBuilder":
        """工厂方法：从文件路径创建构建器实例。

        读取 schema.json 和可选的规则文件，创建 KnowledgeGraphBuilder 实例。

        用法：
          builder = KnowledgeGraphBuilder.from_files("schema.json", "rules.txt")
          graph = builder.build()

        参数:
            schema_path: Schema JSON 文件的路径
            rules_path: 规则文件的路径（可选）

        返回:
            一个已加载好数据的 KnowledgeGraphBuilder 实例
        """
        # 读取 Schema JSON
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        # 尝试读取规则文件，如果未提供或文件不存在则使用空字符串
        rules = ""
        if rules_path:
            try:
                with open(rules_path, 'r', encoding='utf-8') as f:
                    rules = f.read()
            except FileNotFoundError:
                rules = ""

        return cls(schema, rules)
