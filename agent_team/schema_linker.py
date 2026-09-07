# nl2sql/agent_team/schema_linker.py
"""
SchemaLinker —— 零大模型（Zero-LLM）的 Schema 召回引擎。

核心目标：
  给定用户的一条自然语言问题（NL Question）和完整的数据库 Schema，
  SchemaLinker 输出一个结构化的 relevant_schema，包含：
  候选表、候选列、表间 JOIN 路径、时间字段选择、以及警告信息。

整个流程完全基于 BM25 关键词检索 + 规则，不调用任何大语言模型（LLM），
从而保证低延迟、可解释、确定性的输出。

工作流程：
  第 1 步：表级召回（_recall_tables）
    用 BM25 对表名+表描述+列信息做关键词匹配。
  第 2 步：外键扩展（_expand_by_fk）
    按外键边置信度与种子表 BM25 相关度计分，有界地补充关联表。
  第 3 步：列级召回（_recall_columns）
    对每张候选表，用 BM25 找到与问题最相关的列。
  第 4 步：时间字段解析 + JOIN 路径发现
    调用 TimeFieldResolver 确定每张表的最佳时间字段；
    仅暴露达到 JOIN 阈值的分级关系，同名弱候选不作为可执行 JOIN。

SchemaLinker 依赖：
  - SimpleBM25           用于关键词匹配
  - KnowledgeGraphBuilder 用于外键扩展和 JOIN 发现
  - TimeFieldResolver    用于时间字段的选取规则
"""

import re
import json
from typing import Optional

from agent_team.sql_rag import SimpleBM25
from agent_team.knowledge_graph_builder import KnowledgeGraphBuilder
from agent_team.time_field_resolver import TimeFieldResolver


class SchemaLinker:
    """SchemaLinker —— 零大模型的 Schema 检索引擎。

    职责：
      接收自然语言问题 + 完整 Schema，输出结构化的 relevant_schema，
      作为后续 SQL 生成的 Schema 上下文。

    使用方式：
      linker = SchemaLinker(schema)
      result = linker.link("上个月 DAU 是多少？")
    """

    AUTO_EXPAND_THRESHOLD = 0.75
    JOIN_PATH_THRESHOLD = 0.60
    HOP_DECAY = 0.85
    MAX_EXPANSIONS_PER_SEED = 2
    MAX_CANDIDATE_TABLES = 8

    def __init__(self, schema: list[dict], knowledge_graph: Optional[object] = None):
        self.schema = schema
        self.time_resolver = TimeFieldResolver()
        self._table_index: dict[str, dict] = {t["table_name"]: t for t in schema}
        self._knowledge_graph = knowledge_graph

        # 预构建 BM25 文档
        self._table_docs: list[tuple[str, str]] = self._build_table_docs()
        self._column_docs: dict[str, list[tuple[str, str]]] = self._build_column_docs()

        # 表级 BM25：对全部表文档 fit，IDF 全表共享
        self._bm25_tables = SimpleBM25()
        self._bm25_tables.fit([doc for _, doc in self._table_docs])

        # 列级 BM25：对全部列文档 fit，IDF 跨表共享，降低 id/name/date 等通用列名权重
        self._bm25_columns = SimpleBM25()
        all_col_docs = []
        for col_list in self._column_docs.values():
            all_col_docs.extend(doc for _, doc in col_list)
        self._bm25_columns.fit(all_col_docs)

    def _build_table_docs(self) -> list[tuple[str, str]]:
        """为每张表构建 BM25 文档字符串。"""
        docs = []
        for table_name, table in self._table_index.items():
            parts = [table_name, table.get("table_description", "")]
            col_parts = []
            for col in table.get("columns", []):
                col_parts.append(col.get("col", ""))
                desc = col.get("description", "")
                if desc:
                    col_parts.append(desc)
            parts.append(" ".join(col_parts))
            docs.append((table_name, " ".join(parts)))
        return docs

    def _build_column_docs(self) -> dict[str, list[tuple[str, str]]]:
        """为每张表的每列构建 BM25 文档字符串。"""
        docs = {}
        for table_name, table in self._table_index.items():
            col_docs = []
            for col in table.get("columns", []):
                col_name = col.get("col", "")
                desc = col.get("description", "")
                col_docs.append((col_name, f"{col_name} {desc}"))
            docs[table_name] = col_docs
        return docs

    @property
    def knowledge_graph(self):
        if self._knowledge_graph is None:
            builder = KnowledgeGraphBuilder(self.schema)
            self._knowledge_graph = builder.build()
        return self._knowledge_graph

    def link(self, question: str, top_tables: int = 5, top_columns_per_table: int = 8,
             evidence: str = "") -> dict:
        """主入口：将自然语言问题链接到 Schema 实体。

        参数:
            question: 用户自然语言问题
            top_tables: 最多召回表数
            top_columns_per_table: 每张表最多召回列数
            evidence: BIRD 外部知识/evidence，用于增强 BM25 表名匹配
        """
        if not question.strip():
            return self._empty_result(question)

        # 第 1 步：表级 BM25 召回（evidence 参与加分）
        candidate_tables = self._recall_tables(question, top_k=top_tables, evidence=evidence)

        # 第 2 步：按 FK 可信度和起点表相关度做一跳扩展
        candidate_tables, expansion_trace = self._expand_by_fk(candidate_tables)

        # 第 3 步：列级 BM25 召回
        for table in candidate_tables:
            table["relevant_columns"] = self._recall_columns(
                question, table["name"], top_k=top_columns_per_table
            )

        # 第 4 步：时间字段 + JOIN 路径
        time_fields = self._resolve_time_fields(candidate_tables, question)
        join_paths = self._discover_join_paths(candidate_tables)
        dangerous_fields = self._get_dangerous_fields()
        multi_value_fields = self._get_multi_value_fields(candidate_tables)

        return {
            "question": question,
            "candidate_tables": candidate_tables,
            "join_paths": join_paths,
            "time_fields": time_fields,
            "multi_value_fields": multi_value_fields,
            "dangerous_fields": dangerous_fields,
            "ambiguity_flags": [],
            "fk_expansion": expansion_trace,
        }

    def relink_from_feedback(
        self,
        question: str,
        current_schema: dict,
        structured_feedback: dict,
        max_new_tables: int = 2,
        top_columns_per_table: int = 12,
    ) -> dict:
        """根据 Judge 反馈做一次受限的增量 Schema 检索。

        只补充 Judge 明示或 BM25 命中的少量表，并刷新相关列；不会执行 FK
        全图扩展，避免修复阶段把上下文重新膨胀为全量 Schema。
        """
        search = (structured_feedback or {}).get("schema_search", {})
        if not search.get("required"):
            return current_schema

        requested_limit = search.get("max_new_tables", max_new_tables)
        try:
            requested_limit = int(requested_limit)
        except (TypeError, ValueError):
            requested_limit = max_new_tables
        limit = max(0, min(max_new_tables, requested_limit))

        query_terms = [str(term).strip() for term in search.get("query_terms", []) if str(term).strip()]
        suggested = [str(name).strip() for name in search.get("suggested_tables", []) if str(name).strip()]
        retrieval_query = " ".join([question] + query_terms)

        current_tables = [dict(table) for table in current_schema.get("candidate_tables", [])]
        existing_names = {table.get("name", "").lower() for table in current_tables}
        table_name_lookup = {name.lower(): name for name in self._table_index}

        ranked_names = []
        for name in suggested:
            canonical = table_name_lookup.get(name.lower())
            if canonical and canonical.lower() not in existing_names and canonical not in ranked_names:
                ranked_names.append(canonical)

        # Judge 已给出合法表名时只补这些表；只有未给出可解析表名时才用
        # query_terms 做 BM25 兜底，避免把“最多两表”误解成“必须补满两表”。
        if not ranked_names:
            for candidate in self._recall_tables(retrieval_query, top_k=max(5, limit + 2)):
                name = candidate["name"]
                if name.lower() not in existing_names and name not in ranked_names:
                    ranked_names.append(name)

        added_names = ranked_names[:limit]

        # 刷新当前表的列召回，使 Judge 提到的新指标/过滤字段能进入 Refiner 上下文。
        for table in current_tables:
            name = table.get("name")
            if not name:
                continue
            refreshed = self._recall_columns(
                retrieval_query, name, top_k=top_columns_per_table
            )
            existing_columns = {
                column.get("name"): column
                for column in table.get("relevant_columns", [])
                if column.get("name")
            }
            for column in refreshed:
                existing_columns.setdefault(column.get("name"), column)
            table["relevant_columns"] = list(existing_columns.values())[:top_columns_per_table]

        for name in added_names:
            table_info = self._table_index[name]
            current_tables.append({
                "name": name,
                "layer": self.knowledge_graph.nodes.get(name, {}).get("layer", ""),
                "suffix_type": self.knowledge_graph.nodes.get(name, {}).get("suffix_type", ""),
                "description": table_info.get("table_description", ""),
                "score": 0.0,
                "relevant_columns": self._recall_columns(
                    retrieval_query, name, top_k=top_columns_per_table
                ),
                "retrieval_reason": "judge_feedback",
            })

        refined = dict(current_schema)
        refined["candidate_tables"] = current_tables
        refined["join_paths"] = self._discover_join_paths(current_tables)
        refined["time_fields"] = self._resolve_time_fields(current_tables, retrieval_query)
        refined["multi_value_fields"] = self._get_multi_value_fields(current_tables)
        refined["reschema"] = {
            "triggered": True,
            "query_terms": query_terms[:12],
            "suggested_tables": suggested[:6],
            "added_tables": added_names,
            "max_new_tables": limit,
        }
        return refined

    def _recall_tables(self, question: str, top_k: int = 5,
                        evidence: str = "") -> list[dict]:
        """BM25 关键词匹配召回最相关的表。

        evidence 中出现表名时，该表 BM25 分 +2.0（强信号），
        确保 BIRD evidence 中明确指出的表不会被漏掉。
        """
        # 从 evidence 中提取显式表名（完整单词匹配）
        evidence_table_names = set()
        if evidence:
            evidence_lower = evidence.lower()
            for table_name in self._table_index:
                if table_name.lower() in evidence_lower:
                    evidence_table_names.add(table_name)

        scores = []
        for table_name, doc in self._table_docs:
            score = self._bm25_tables.score(question, doc)
            # evidence 中出现的表：+2.0 分（BM25 典型值在 0~8 之间）
            if table_name in evidence_table_names:
                score += 2.0
            scores.append((table_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        max_positive_score = max((score for _, score in scores), default=0.0)

        candidate_tables = []
        for table_name, score in scores[:top_k]:
            table = self._table_index[table_name]
            layer = ""
            suffix_type = ""
            if table_name in self.knowledge_graph.nodes:
                node = self.knowledge_graph.nodes[table_name]
                layer = node.get("layer", "")
                suffix_type = node.get("suffix_type", "")

            normalized_relevance = (
                max(0.0, score) / max_positive_score if max_positive_score > 0 else 0.0
            )
            candidate_tables.append({
                "name": table_name,
                "layer": layer,
                "suffix_type": suffix_type,
                "description": table.get("table_description", ""),
                "score": round(normalized_relevance, 4),
                "bm25_score": round(score, 4),
                "normalized_relevance": round(normalized_relevance, 4),
                "final_score": round(normalized_relevance, 4),
                "retrieval_reason": "bm25",
            })

        return candidate_tables

    def _expand_by_fk(self, candidate_tables: list[dict]) -> tuple[list[dict], dict]:
        """Expand one hop using separate relevance and relationship scores."""
        before_names = [table["name"] for table in candidate_tables]
        existing_names = set(before_names)
        proposals: dict[str, dict] = {}
        rejected_low_confidence = 0

        for seed in candidate_tables:
            seed_relevance = float(seed.get("normalized_relevance", 0.0) or 0.0)
            if seed_relevance <= 0:
                continue
            seed_proposals = []
            for target_name in self._table_index:
                if target_name in existing_names:
                    continue
                relationships = self._get_join_between(seed["name"], target_name)
                if not relationships:
                    continue
                eligible = [
                    relationship for relationship in relationships
                    if float(relationship.get("confidence", 0.0)) >= self.AUTO_EXPAND_THRESHOLD
                ]
                rejected_low_confidence += len(relationships) - len(eligible)
                if not eligible:
                    continue
                best_edge = max(eligible, key=lambda item: item["confidence"])
                expansion_score = (
                    seed_relevance * float(best_edge["confidence"]) * self.HOP_DECAY
                )
                seed_proposals.append({
                    "name": target_name,
                    "expanded_from": seed["name"],
                    "expansion_score": expansion_score,
                    "edge": best_edge,
                })

            seed_proposals.sort(
                key=lambda item: (item["expansion_score"], item["name"]),
                reverse=True,
            )
            for proposal in seed_proposals[:self.MAX_EXPANSIONS_PER_SEED]:
                current = proposals.get(proposal["name"])
                if current is None or proposal["expansion_score"] > current["expansion_score"]:
                    proposals[proposal["name"]] = proposal

        available_slots = max(0, self.MAX_CANDIDATE_TABLES - len(candidate_tables))
        selected_proposals = sorted(
            proposals.values(),
            key=lambda item: (item["expansion_score"], item["name"]),
            reverse=True,
        )[:available_slots]

        new_tables = []
        expansion_details = []
        for proposal in selected_proposals:
            name = proposal["name"]
            table = self._table_index[name]
            node_attrs = self.knowledge_graph.nodes.get(name, {})
            best_edge = proposal["edge"]
            final_score = round(proposal["expansion_score"], 4)
            new_tables.append({
                "name": name,
                "layer": node_attrs.get("layer", ""),
                "suffix_type": node_attrs.get("suffix_type", ""),
                "description": table.get("table_description", ""),
                "score": final_score,
                "bm25_score": 0.0,
                "normalized_relevance": 0.0,
                "expansion_score": final_score,
                "final_score": final_score,
                "retrieval_reason": "fk_expansion",
                "expanded_from": proposal["expanded_from"],
                "edge_type": best_edge["edge_type"],
                "edge_confidence": best_edge["confidence"],
            })
            expansion_details.append({
                "table": name,
                "expanded_from": proposal["expanded_from"],
                "expansion_score": final_score,
                "edge_type": best_edge["edge_type"],
                "edge_confidence": best_edge["confidence"],
                "from_column": best_edge["from_column"],
                "to_column": best_edge["to_column"],
                "evidence": best_edge["evidence"],
            })

        combined = candidate_tables + new_tables
        combined.sort(key=lambda table: table.get("final_score", 0.0), reverse=True)
        return combined, {
            "before_tables": before_names,
            "added_tables": [item["table"] for item in expansion_details],
            "details": expansion_details,
            "auto_expand_threshold": self.AUTO_EXPAND_THRESHOLD,
            "hop_decay": self.HOP_DECAY,
            "max_expansions_per_seed": self.MAX_EXPANSIONS_PER_SEED,
            "max_candidate_tables": self.MAX_CANDIDATE_TABLES,
            "rejected_low_confidence_edges": rejected_low_confidence,
        }

    def _get_join_between(self, table_a: str, table_b: str) -> list[dict]:
        """Return typed graph relationships between two tables, strongest first."""
        requested = {table_a.casefold(), table_b.casefold()}
        relationships = []
        for _, _, attrs in self.knowledge_graph.edges(data=True):
            if attrs.get("relationship") != "FK_RELATIONSHIP":
                continue
            connected = {
                str(attrs.get("from_table", "")).casefold(),
                str(attrs.get("to_table", "")).casefold(),
            }
            if connected == requested:
                relationships.append({
                    key: attrs[key]
                    for key in (
                        "from_table", "from_column", "to_table", "to_column",
                        "edge_type", "confidence", "evidence",
                    )
                })
        return sorted(
            relationships,
            key=lambda item: item["confidence"],
            reverse=True,
        )

    def _recall_columns(self, question: str, table_name: str, top_k: int = 8) -> list[dict]:
        """BM25 关键词匹配召回指定表中最相关的列。"""
        table = self._table_index.get(table_name)
        if not table:
            return []

        col_docs = self._column_docs.get(table_name, [])
        scores = []
        for col_name, doc in col_docs:
            score = self._bm25_columns.score(question, doc)
            scores.append((col_name, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        relevant_cols = []
        seen = set()
        for col_name, score in scores[:top_k]:
            seen.add(col_name)
            relevant_cols.append(self._get_column_info(table_name, col_name))

        # 兜底：BM25 召回不够 top_k，补充未选中的列
        if len(relevant_cols) < top_k:
            for col in table.get("columns", []):
                if col["col"] not in seen:
                    relevant_cols.append({
                        "name": col["col"],
                        "type": col.get("type", "unknown"),
                        "description": col.get("description", ""),
                        "is_date": False,
                    })
                    if len(relevant_cols) >= top_k:
                        break

        return relevant_cols

    def _get_column_info(self, table_name: str, col_name: str) -> dict:
        """获取列的完整信息。"""
        table = self._table_index.get(table_name, {})
        for col in table.get("columns", []):
            if col["col"] == col_name:
                node_id = f"{table_name}.{col_name}"
                kg_node = self.knowledge_graph.nodes.get(node_id, {})

                return {
                    "name": col_name,
                    "type": col.get("type", "unknown"),
                    "description": col.get("description", ""),
                    "is_date": kg_node.get("is_date_field", False),
                    "is_dangerous": kg_node.get("is_dangerous", False),
                    "is_multi_value": kg_node.get("is_multi_value", False),
                }
        return {"name": col_name, "type": "unknown", "description": "", "is_date": False}

    def _resolve_time_fields(self, candidate_tables: list[dict], question: str) -> dict:
        """解析候选表的时间字段。"""
        if not candidate_tables:
            return {"primary": {"table": None, "field": None, "type": "none"}, "notes": []}

        per_table = {}
        for ct in candidate_tables:
            table_name = ct["name"]
            table = self._table_index.get(table_name)
            if table:
                tr_table = {
                    "name": table_name,
                    "layer": ct.get("layer", ""),
                    "columns": [
                        {"name": c["col"], "type": c.get("type", ""), "description": c.get("description", "")}
                        for c in table.get("columns", [])
                    ]
                }
                resolved = self.time_resolver.resolve(tr_table, question)
                if resolved["primary"]["field"]:
                    per_table[table_name] = resolved

        primary = {"table": None, "field": None, "type": "none"}
        notes = []

        for table_name, resolved in per_table.items():
            p = resolved["primary"]
            if p["field"] == "dtstatdate" and primary["field"] is None:
                primary = p
                notes = resolved.get("notes", [])
            elif p["field"] == "dteventtime" and primary["field"] is None:
                primary = p
                notes = resolved.get("notes", [])

        if primary["field"] is None and per_table:
            first_table = list(per_table.values())[0]
            primary = first_table["primary"]
            notes = first_table.get("notes", [])

        return {
            "primary": primary,
            "all_per_table": {t: r["primary"] for t, r in per_table.items()},
            "notes": notes,
        }

    def _discover_join_paths(self, candidate_tables: list[dict]) -> list[dict]:
        """Expose only relationships strong enough to be actionable JOIN paths."""
        join_paths = []
        table_names = [t["name"] for t in candidate_tables]

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                relationships = self._get_join_between(table_names[i], table_names[j])
                for relationship in relationships:
                    edge_confidence = float(relationship.get("confidence", 0.0))
                    if edge_confidence < self.JOIN_PATH_THRESHOLD:
                        continue
                    join_paths.append({
                        "from": relationship["from_table"],
                        "to": relationship["to_table"],
                        "on": [
                            f"{relationship['from_column']}={relationship['to_column']}"
                        ],
                        "from_column": relationship["from_column"],
                        "to_column": relationship["to_column"],
                        "edge_type": relationship["edge_type"],
                        "edge_confidence": edge_confidence,
                        "confidence": (
                            "high" if edge_confidence >= self.AUTO_EXPAND_THRESHOLD else "medium"
                        ),
                        "evidence": relationship["evidence"],
                    })

        return join_paths

    def _get_dangerous_fields(self) -> list[dict]:
        """收集知识图谱中标记为危险字段的信息。"""
        dangerous = []
        for node, attrs in self.knowledge_graph.nodes(data=True):
            if attrs.get("is_dangerous"):
                col_name = node.split(".")[-1]
                table_name = node.split(".")[0]
                dangerous.append({
                    "field": col_name,
                    "table": table_name,
                    "warning": attrs.get("warning", "预留字段，使用前需确认上下文含义"),
                })
        return dangerous

    def _get_multi_value_fields(self, candidate_tables: list[dict]) -> list[dict]:
        """识别候选表中的多值字段。"""
        mv_fields = []
        for ct in candidate_tables:
            table_name = ct["name"]
            for col in ct.get("relevant_columns", []):
                if col.get("is_multi_value"):
                    mv_fields.append({
                        "table": table_name,
                        "field": col["name"],
                    })
        return mv_fields

    def _empty_result(self, question: str) -> dict:
        return {
            "question": question,
            "candidate_tables": [],
            "join_paths": [],
            "time_fields": {"primary": {"table": None, "field": None, "type": "none"}, "notes": []},
            "multi_value_fields": [],
            "dangerous_fields": [],
            "ambiguity_flags": [],
            "fk_expansion": {
                "before_tables": [],
                "added_tables": [],
                "details": [],
            },
        }

    @classmethod
    def from_files(cls, schema_path: str) -> "SchemaLinker":
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return cls(schema)
