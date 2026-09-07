# nl2sql/agent_team/nl2sql_judge.py
"""NL2SQL Judge —— LLM-as-a-Verifier，逐项核查 + 证据锚定。

不依赖 agent-scorer 的 provider 层，直接复用 Orchestrator 的 model_client。
"""

import json
import re
from collections import defaultdict

import numpy as np
import sqlglot
from sqlglot import exp as sqlglot_exp

from agent_team.contracts import DraftIntent, GateResult
from agent_team.judge_scoring_card import JudgeScoringCard
from agent_team.pre_execution_gate import PreExecutionGate


class NL2SQLJudge:
    """LLM-as-a-Verifier 核查器。

    使用四步验证法：
    1. 生成 YES/NO 核查清单（24项，满分50）
    2. 逐项给出证据 (SQL 片段 / Schema 列名 / 执行结果)
    3. LLM 从 SQL 反向生成英文问题，计算与原始问题的余弦相似度（满分50）
    4. 总分 = 核查分(0-50) + 语义往返保真度分(0-50)，满分为100

    通过阈值: overall_confidence >= 80 且无 critical_flaws
    """

    PASS_CONFIDENCE = 80
    PASS_SEMANTICS = 32
    BYPASS_SEMANTICS = 40
    HIGH_CONFIDENCE_SEMANTIC = 0.90  # 语义往返极高相似度 → 直接 PASS，不看白盒分

    def __init__(self, model_client=None, model: str = "deepseek-v4-flash"):
        self.model_client = model_client
        self.model = model
        self.scoring_card = JudgeScoringCard()

    def _verify_sqlglot_legacy(
        self,
        sql: str,
        relevant_schema: dict,
        full_schema: list[dict] = None,
        exec_result: dict = None,
    ) -> dict:
        """确定性检查语法、访问范围、执行状态、结果 schema 和聚合粒度。

        返回:
            {
                "parse_ok": bool,
                "parse_error": str,
                "tables_in_sql": [str],
                "missing_tables": [str],
                "missing_columns": [str],
                "all_tables_ok": bool,
                "all_columns_ok": bool,
                "checks": [...],
                "blocking_issues": [...],
                "result_schema": {...},
                "aggregation_grain": {...},
            }
        """
        result = {
            "parse_ok": False,
            "parse_error": "",
            "tables_in_sql": [],
            "missing_tables": [],
            "missing_columns": [],
            "all_tables_ok": False,
            "all_columns_ok": False,
            "checks": [],
            "blocking_issues": [],
            "result_schema": {
                "expected_columns": [],
                "actual_columns": [],
                "matches": None,
            },
            "aggregation_grain": {
                "has_aggregation": False,
                "group_by": [],
                "aggregate_functions": [],
                "valid": True,
            },
        }

        def add_check(code: str, passed: bool, message: str, evidence=None, blocking=False):
            check = {
                "code": code,
                "passed": passed,
                "message": message,
                "evidence": evidence if evidence is not None else "",
                "blocking": bool(blocking and not passed),
            }
            result["checks"].append(check)
            if check["blocking"]:
                result["blocking_issues"].append(check)

        # 1. SQL 解析，并限制为单条只读查询
        try:
            statements = [stmt for stmt in sqlglot.parse(sql, dialect="sqlite") if stmt is not None]
            if len(statements) != 1:
                result["parse_error"] = f"expected exactly one statement, got {len(statements)}"
                add_check("syntax.single_statement", False, result["parse_error"], blocking=True)
                return result
            ast = statements[0]
            if ast is None or not isinstance(ast, sqlglot_exp.Query):
                result["parse_error"] = "sqlglot returned None"
                add_check(
                    "permission.read_only",
                    False,
                    "Only a single SELECT/WITH query is allowed",
                    type(ast).__name__ if ast is not None else "",
                    blocking=True,
                )
                return result
            result["parse_ok"] = True
            add_check("syntax.parse", True, "SQL parsed successfully")
            add_check("permission.read_only", True, "SQL is a read-only query")
        except Exception as e:
            result["parse_error"] = str(e)[:200]
            add_check("syntax.parse", False, "SQL parse failed", result["parse_error"], blocking=True)
            return result

        # 2. 从 AST 提取引用的表名（含 CTE）
        tables_in_sql = set()
        cte_names = set()
        table_aliases = {}
        for node in ast.find_all(sqlglot_exp.Table):
            name = node.name.lower() if node.name else ""
            if name:
                tables_in_sql.add(name)
                table_aliases[(node.alias_or_name or name).lower()] = name
        for cte in ast.find_all(sqlglot_exp.CTE):
            if cte.alias:
                cte_names.add(cte.alias.lower())
        # CTE 不算外部表引用
        tables_in_sql -= cte_names
        result["tables_in_sql"] = sorted(tables_in_sql)

        # 3. 构建已授权 Schema 索引。full_schema 是数据库权限边界，
        # relevant_schema 是本轮检索范围；两者分别记录，避免把漏召回误报成越权。
        selected_tables: dict[str, set[str]] = {}
        for t in relevant_schema.get("candidate_tables", []) if relevant_schema else []:
            tname = t["name"].lower()
            selected_cols = set()
            for c in t.get("relevant_columns", []):
                selected_cols.add(c["name"].lower())
            selected_tables[tname] = selected_cols

        allowed_tables: dict[str, set[str]] = {}
        for table in full_schema or []:
            tname = table.get("table_name", table.get("name", "")).lower()
            if not tname:
                continue
            allowed_tables[tname] = {
                col.get("col", col.get("name", "")).lower()
                for col in table.get("columns", [])
                if col.get("col", col.get("name", ""))
            }
        if not allowed_tables:
            allowed_tables = selected_tables

        # 4. 验证表引用
        missing_tables = [t for t in tables_in_sql if t not in allowed_tables]
        outside_retrieval = [
            t for t in tables_in_sql
            if t in allowed_tables and t not in selected_tables
        ]
        result["missing_tables"] = missing_tables
        result["all_tables_ok"] = len(missing_tables) == 0 and len(tables_in_sql) > 0
        add_check(
            "permission.tables",
            not missing_tables,
            "All referenced tables are allowed" if not missing_tables else "SQL references unknown or unauthorized tables",
            {"referenced": sorted(tables_in_sql), "unauthorized": missing_tables},
            blocking=True,
        )
        add_check(
            "schema.retrieval_scope",
            not outside_retrieval,
            "SQL stays within the retrieved schema" if not outside_retrieval else "SQL uses tables outside the retrieved schema",
            {"outside_retrieval": outside_retrieval},
        )

        # 5. 从 AST 提取列引用，验证是否存在
        missing_columns = []
        ambiguous_columns = []
        column_owners = defaultdict(set)
        for table_name, columns in allowed_tables.items():
            for column_name in columns:
                column_owners[column_name].add(table_name)
        select_aliases = {
            expression.alias.lower()
            for expression in ast.expressions
            if getattr(expression, "alias", "")
        }
        derived_columns = set()
        for cte in ast.find_all(sqlglot_exp.CTE):
            cte_select = cte.this.find(sqlglot_exp.Select)
            if cte_select:
                derived_columns.update(
                    expression.alias_or_name.lower()
                    for expression in cte_select.expressions
                    if expression.alias_or_name
                )

        for node in ast.find_all(sqlglot_exp.Column):
            col_name = node.name.lower() if node.name else ""
            table_name = node.table.lower() if node.table else ""
            if not col_name or col_name == "*":
                continue
            if table_name:
                actual_table = table_aliases.get(table_name, table_name)
                qualified = f"{actual_table}.{col_name}"
                if actual_table in allowed_tables and col_name not in allowed_tables[actual_table]:
                    missing_columns.append(qualified)
            else:
                containing_select = node.find_ancestor(sqlglot_exp.Select)
                scope_uses_cte = False
                if containing_select is not None:
                    scope_uses_cte = any(
                        table.name.lower() in cte_names
                        for table in containing_select.find_all(sqlglot_exp.Table)
                        if table.find_ancestor(sqlglot_exp.Select) is containing_select
                    )
                if col_name in select_aliases or (scope_uses_cte and col_name in derived_columns):
                    continue
                owners = column_owners.get(col_name, set()) & tables_in_sql
                if not owners:
                    missing_columns.append(col_name)
                elif len(owners) > 1 and len(tables_in_sql) > 1:
                    ambiguous_columns.append(col_name)

        result["missing_columns"] = sorted(set(missing_columns))[:20]
        result["all_columns_ok"] = not result["missing_columns"]
        add_check(
            "permission.columns",
            not result["missing_columns"],
            "All referenced columns are allowed" if not result["missing_columns"] else "SQL references unknown or unauthorized columns",
            {
                "unauthorized": result["missing_columns"],
                "ambiguous_unqualified": sorted(set(ambiguous_columns)),
            },
            blocking=True,
        )

        # 6. 执行状态
        exec_ok = bool((exec_result or {}).get("ok", False))
        add_check(
            "execution.status",
            exec_ok,
            "SQL executed successfully" if exec_ok else "SQL execution failed",
            (exec_result or {}).get("error", "")[:300],
            blocking=True,
        )

        # 7. 结果 schema：比较 SELECT 输出名与执行器返回列名。
        expected_columns = []
        has_star = False
        for expression in ast.expressions:
            target = expression.this if isinstance(expression, sqlglot_exp.Alias) else expression
            if (
                isinstance(target, sqlglot_exp.Star)
                or isinstance(target, sqlglot_exp.Column) and target.is_star
            ):
                has_star = True
            output_name = expression.alias_or_name
            if output_name and output_name != "*":
                expected_columns.append(output_name)

        actual_columns = list((exec_result or {}).get("columns", []) or [])
        if not actual_columns:
            sample_rows = (exec_result or {}).get("sample_rows", []) or (exec_result or {}).get("rows", [])
            if sample_rows and isinstance(sample_rows[0], dict):
                actual_columns = list(sample_rows[0].keys())

        schema_matches = None
        if expected_columns and actual_columns and not has_star:
            schema_matches = [c.lower() for c in expected_columns] == [c.lower() for c in actual_columns]
            add_check(
                "result.schema",
                schema_matches,
                "Execution result schema matches SELECT output" if schema_matches else "Execution result schema differs from SELECT output",
                {"expected": expected_columns, "actual": actual_columns},
                blocking=True,
            )
        else:
            add_check(
                "result.schema",
                True,
                "Result schema check skipped because output metadata is incomplete or SELECT uses *",
                {"expected": expected_columns, "actual": actual_columns},
            )
        result["result_schema"] = {
            "expected_columns": expected_columns,
            "actual_columns": actual_columns,
            "matches": schema_matches,
        }

        # 8. 聚合口径：提取指标函数和 GROUP BY 粒度，并检查混合聚合是否合法。
        aggregate_functions = []
        invalid_non_aggregates = []
        grain_scopes = []
        root_group_sql = []
        for scope_index, select in enumerate(ast.find_all(sqlglot_exp.Select), 1):
            aggregate_nodes = [
                node for node in select.find_all(sqlglot_exp.AggFunc)
                if node.find_ancestor(sqlglot_exp.Select) is select
            ]
            group = select.args.get("group")
            group_expressions = list(group.expressions) if group else []
            group_sql = [expr.sql(dialect="sqlite") for expr in group_expressions]
            if select is ast:
                root_group_sql = group_sql
            group_normalized = {
                expr.sql(dialect="sqlite").lower() for expr in group_expressions
            }
            selected_targets = [
                expression.this if isinstance(expression, sqlglot_exp.Alias) else expression
                for expression in select.expressions
            ]
            alias_targets = {
                expression.alias.lower(): expression.this.sql(dialect="sqlite").lower()
                for expression in select.expressions
                if isinstance(expression, sqlglot_exp.Alias) and expression.alias
            }
            for group_expression in group_expressions:
                if isinstance(group_expression, sqlglot_exp.Literal) and group_expression.is_int:
                    position = int(group_expression.this)
                    if 1 <= position <= len(selected_targets):
                        group_normalized.add(
                            selected_targets[position - 1].sql(dialect="sqlite").lower()
                        )
                elif isinstance(group_expression, sqlglot_exp.Column) and not group_expression.table:
                    alias_target = alias_targets.get(group_expression.name.lower())
                    if alias_target:
                        group_normalized.add(alias_target)

            scope_invalid = []
            if aggregate_nodes:
                for target in selected_targets:
                    direct_aggregates = [
                        node for node in target.find_all(sqlglot_exp.AggFunc)
                        if node.find_ancestor(sqlglot_exp.Select) is select
                    ]
                    if direct_aggregates or isinstance(target, sqlglot_exp.Literal):
                        continue
                    if target.sql(dialect="sqlite").lower() not in group_normalized:
                        scope_invalid.append(target.sql(dialect="sqlite"))
            aggregate_functions.extend(node.sql_name() for node in aggregate_nodes)
            invalid_non_aggregates.extend(scope_invalid)
            if aggregate_nodes or group_expressions:
                grain_scopes.append({
                    "scope": scope_index,
                    "group_by": group_sql,
                    "aggregate_functions": [node.sql_name() for node in aggregate_nodes],
                    "invalid_non_aggregates": scope_invalid,
                })
        grain_valid = not invalid_non_aggregates
        add_check(
            "aggregation.grain",
            grain_valid,
            "Aggregation grain is structurally valid" if grain_valid else "Non-aggregated outputs are missing from GROUP BY",
            {
                "group_by": root_group_sql,
                "aggregate_functions": aggregate_functions,
                "invalid_non_aggregates": invalid_non_aggregates,
                "scopes": grain_scopes,
            },
            blocking=True,
        )
        result["aggregation_grain"] = {
            "has_aggregation": bool(aggregate_functions),
            "group_by": root_group_sql,
            "aggregate_functions": aggregate_functions,
            "invalid_non_aggregates": invalid_non_aggregates,
            "scopes": grain_scopes,
            "valid": grain_valid,
        }

        return result

    def _verify_sqlglot(
        self,
        sql: str,
        relevant_schema: dict,
        full_schema: list[dict] = None,
        exec_result: dict = None,
        dialect: str = "sqlite",
    ) -> dict:
        """Compatibility view composed from Gate and post-execution metadata."""
        gate = PreExecutionGate(dialect=dialect).check(
            sql, relevant_schema, full_schema, dialect=dialect
        )
        blocker_codes = {issue.code for issue in gate.blockers}
        missing_tables = []
        missing_columns = []
        for issue in gate.blockers:
            details = issue.details if isinstance(issue.details, dict) else {}
            if issue.code == "schema.table_not_authorized":
                missing_tables.extend(details.get("tables", []))
            elif issue.code == "schema.column_not_authorized":
                missing_columns.extend(details.get("columns", []))

        checks = []
        blocking_issues = []
        for issue in [*gate.blockers, *gate.warnings]:
            is_blocker = issue in gate.blockers
            check = {
                "code": issue.code,
                "passed": False,
                "message": issue.message,
                "evidence": issue.details if issue.details is not None else "",
                "blocking": is_blocker,
            }
            checks.append(check)
            if check["blocking"]:
                blocking_issues.append(check)

        exec_result = exec_result or {}
        exec_ok = bool(exec_result.get("ok", False))
        execution_check = {
            "code": "execution.status",
            "passed": exec_ok,
            "message": "SQL executed successfully" if exec_ok else "SQL execution failed",
            "evidence": str(exec_result.get("error", ""))[:300],
            "blocking": not exec_ok,
        }
        checks.append(execution_check)
        if not exec_ok:
            blocking_issues.append(execution_check)

        expected_columns = list(gate.sql_signature.get("projected_columns", []) or [])
        actual_columns = list(exec_result.get("columns", []) or [])
        if not actual_columns:
            sample_rows = exec_result.get("sample_rows", []) or exec_result.get("rows", []) or []
            if sample_rows and isinstance(sample_rows[0], dict):
                actual_columns = list(sample_rows[0].keys())

        has_star = any(column == "*" for column in expected_columns)
        schema_matches = None
        if expected_columns and actual_columns and not has_star:
            schema_matches = [str(c).lower() for c in expected_columns] == [
                str(c).lower() for c in actual_columns
            ]
            schema_check = {
                "code": "result.schema",
                "passed": schema_matches,
                "message": (
                    "Execution result schema matches SELECT output"
                    if schema_matches else "Execution result schema differs from SELECT output"
                ),
                "evidence": {"expected": expected_columns, "actual": actual_columns},
                "blocking": not schema_matches,
            }
            checks.append(schema_check)
            if not schema_matches:
                blocking_issues.append(schema_check)

        aggregation = dict(gate.sql_signature.get("aggregation", {}) or {})
        aggregation["valid"] = not bool(aggregation.get("invalid_non_aggregates"))
        return {
            "parse_ok": not bool(blocker_codes & {
                "syntax.empty_sql", "syntax.parse_error", "structure.multiple_statements",
                "safety.non_read_only", "config.unsupported_dialect",
            }),
            "parse_error": next((
                issue.message for issue in gate.blockers
                if issue.code.startswith("syntax.") or issue.code.startswith("structure.")
            ), ""),
            "tables_in_sql": list(gate.sql_signature.get("tables", []) or []),
            "missing_tables": sorted(set(missing_tables)),
            "missing_columns": sorted(set(missing_columns)),
            "all_tables_ok": not missing_tables,
            "all_columns_ok": not missing_columns,
            "checks": checks,
            "blocking_issues": blocking_issues,
            "result_schema": {
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "matches": schema_matches,
            },
            "aggregation_grain": aggregation,
            "gate": gate.to_dict(),
        }

    def evaluate(
        self,
        question: str,
        sql: str,
        exec_result: dict,
        relevant_schema: dict = None,
        full_schema: list[dict] = None,
        evidence: str = "",
        plan: dict = None,
        sampling_data: dict = None,
        gate_result: GateResult | dict = None,
        draft_intent: DraftIntent | dict = None,
        dialect: str = "sqlite",
    ) -> dict:
        """对一条 SQL 进行逐项核查 + 评分。

        参数:
            question: 用户自然语言问题
            sql: 待核查的 SQL
            exec_result: 执行结果 {"ok", "error", "row_count", "sample_rows"}
            relevant_schema: SchemaLinker 选中的表结构
            full_schema: 全量数据库 schema（用于检测漏表）
            evidence: 外部知识（可选）
            plan: Planner 执行计划（可选）
            sampling_data: 数据采样结果，用于验证 SQL 中的字符串值

        返回:
            {
                "overall_confidence": int, "pass": bool,
                "checks": {...}, "dimensions": {...},
                "critical_flaws": [...], "repair_priority": [...],
                "overall_assessment": str, "raw_response": str,
            }
        """
        normalized_gate = self._coerce_gate_result(gate_result, dialect)
        if normalized_gate is not None and not normalized_gate.passed:
            return self._deterministic_result(
                mode="gate", gate=normalized_gate, exec_result=exec_result
            )
        if not exec_result.get("ok", False):
            return self._deterministic_result(
                mode="database", gate=normalized_gate, exec_result=exec_result
            )

        user_prompt = self.scoring_card.build_prompt(
            question=question,
            sql=sql,
            exec_result=exec_result,
            relevant_schema=relevant_schema,
            full_schema=full_schema,
            evidence=evidence,
            plan=plan,
            sampling_data=sampling_data,
        )
        if draft_intent is not None:
            if isinstance(draft_intent, DraftIntent):
                draft_value = draft_intent.__dict__
            else:
                draft_value = draft_intent
            user_prompt += (
                "\n\n## Builder DraftIntent (challengeable, not authoritative)\n"
                + json.dumps(draft_value, ensure_ascii=False, indent=2)
                + "\nCompare it independently with both the original question and SQL intent."
            )

        messages = [
            {"role": "system", "content": self.scoring_card.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = self._call_llm(messages)
        result = self._parse_response(response_text)

        # 白盒维度或结构化意图缺失 → 重试最多 3 次。
        for _ in range(3):
            dims = result.get("dimensions", {})
            if (
                not self._dimensions_empty(dims)
                and self._intent_comparison_present(result.get("intent_comparison"))
            ):
                break
            response_text = self._call_llm(messages)
            result = self._parse_response(response_text)

        # ── sqlglot 确定性校验：覆盖 LLM 的 syntax + schema 维度 ──
        glot = self._verify_sqlglot(
            sql,
            relevant_schema,
            full_schema=full_schema,
            exec_result=exec_result,
            dialect=dialect,
        )
        dims = result.get("dimensions", {})

        if glot["parse_ok"]:
            # sqlglot 确认语法正确 → 强制 syntax = 7，清除 LLM 误判的语法问题
            dims["syntax"] = {
                "score": 7, "max": 7,
                "issues": [], "suggestions": [],
            }
        elif not glot["parse_ok"] and dims.get("syntax", {}).get("score", 0) > 0:
            # LLM 认为语法正确但 sqlglot 解析失败 → 直接降分
            dims["syntax"] = {
                "score": 0, "max": 7,
                "issues": [f"sqlglot parse error: {glot['parse_error'][:100]}"],
                "suggestions": ["Fix SQL syntax"],
            }

        if glot["all_tables_ok"] and glot["all_columns_ok"]:
            # sqlglot 确认所有表/列引用有效 → 清除 LLM 误判的 schema issues
            sem = dims.get("semantics", {})
            semantic_issues = sem.get("issues", [])
            # 过滤掉 "column not found" / "table doesn't exist" 类的误判
            filtered_issues = [
                i for i in semantic_issues
                if not any(kw in str(i).lower() for kw in
                          ("doesn't exist", "does not exist", "no such",
                           "not present", "not found", "invalid column",
                           "invalid table", "nonexistent", "missing column"))
            ]
            sem["issues"] = filtered_issues
            sem["suggestions"] = [s for s in sem.get("suggestions", [])
                                  if "add" not in s.lower() or "column" not in s.lower()]
            # 如果 LLM 给的语义分很低但 sqlglot 确认结构正确，至少给底分
            if sem.get("score", 0) < 8:
                sem["score"] = 8
            dims["semantics"] = sem

        if glot["missing_tables"]:
            # sqlglot 发现 LLM 没检测到的缺表 → 补充到 critical_flaws
            mt_msg = f"sqlglot: 表引用不存在: {', '.join(glot['missing_tables'])}"
            if mt_msg not in result.get("critical_flaws", []):
                result.setdefault("critical_flaws", []).append(mt_msg)

        if glot["missing_columns"]:
            mc_msg = f"sqlglot: 列引用不存在: {', '.join(glot['missing_columns'][:5])}"
            if mc_msg not in result.get("critical_flaws", []):
                result.setdefault("critical_flaws", []).append(mc_msg)

        # 过滤 critical_flaws 中与 sqlglot 结论矛盾的条目
        if glot["parse_ok"]:
            result["critical_flaws"] = [
                f for f in result.get("critical_flaws", [])
                if not any(kw in str(f).lower() for kw in
                          ("syntax error", "parse error", "invalid sql",
                           "sql syntax", "malformed"))
            ]
        result["dimensions"] = dims
        result["deterministic_checks"] = glot
        result["judge_mode"] = "semantic"
        if draft_intent is not None:
            result["draft_intent"] = draft_value

        # 将 LLM 的结构化意图比对规范化，并生成 Refiner 可直接消费的反馈契约。
        comparison = self._normalize_intent_comparison(result.get("intent_comparison"))
        result["intent_comparison"] = comparison
        structured_feedback = self._build_structured_feedback(glot, comparison, result)
        result["structured_feedback"] = structured_feedback
        for issue in structured_feedback["issues"]:
            if issue["severity"] == "critical" and issue["message"] not in result["critical_flaws"]:
                result["critical_flaws"].append(issue["message"])
            if issue.get("suggestion") and issue["suggestion"] not in result["repair_priority"]:
                result["repair_priority"].append(issue["suggestion"])

        # ── 语义往返保真度：反向生成问题 + 向量相似度 ──
        reverse_q = self._reverse_question(sql, relevant_schema)
        similarity = self._compute_similarity(question, reverse_q)
        similarity_score = round(similarity * 50)

        # 拼装总分：核查清单分 (0-50) + 语义相似度分 (0-50)
        checklist_score = result.get("overall_confidence", 0)
        combined_score = checklist_score + similarity_score

        result["overall_confidence"] = combined_score
        result["reverse_question"] = reverse_q
        result["semantic_similarity"] = round(similarity, 4)
        result["semantic_score"] = similarity_score

        # 语义往返极高置信度 → 直接 PASS，不看白盒维度分
        # （SQL→反向问题→与原问题几乎一致，说明 SQL 语义正确）
        no_critical = (
            not result.get("critical_flaws")
            and not glot.get("blocking_issues")
            and comparison.get("match") is not False
        )
        if similarity >= self.HIGH_CONFIDENCE_SEMANTIC and no_critical:
            result["pass"] = True
            return result

        # 语义往返相似度不足 → 总分封顶，强制不通过
        if similarity_score < self.PASS_SEMANTICS and combined_score >= self.PASS_CONFIDENCE:
            combined_score = self.PASS_CONFIDENCE - 1
            result["overall_confidence"] = combined_score

        # 通过判定：总分达标 或 语义往返高分 bypass
        if combined_score >= self.PASS_CONFIDENCE and no_critical:
            result["pass"] = True
        elif similarity_score >= self.BYPASS_SEMANTICS and no_critical:
            result["pass"] = True
        else:
            result["pass"] = False

        return result

    @staticmethod
    def _coerce_gate_result(value, dialect: str) -> GateResult | None:
        if value is None or isinstance(value, GateResult):
            return value
        if not isinstance(value, dict):
            return None
        from agent_team.contracts import GateIssue
        return GateResult(
            passed=bool(value.get("passed", False)),
            dialect=str(value.get("dialect", dialect)),
            blockers=[GateIssue(**item) for item in value.get("blockers", [])],
            warnings=[GateIssue(**item) for item in value.get("warnings", [])],
            sql_signature=dict(value.get("sql_signature", {}) or {}),
        )

    @staticmethod
    def _deterministic_result(mode: str, gate: GateResult | None, exec_result: dict) -> dict:
        issues = []
        if mode == "gate" and gate is not None:
            for blocker in gate.blockers:
                issues.append({
                    "code": blocker.code,
                    "category": blocker.code.split(".", 1)[0],
                    "severity": "critical",
                    "message": blocker.message,
                    "evidence": blocker.details if blocker.details is not None else "",
                    "suggestion": "Repair the deterministic Gate failure before execution",
                })
        else:
            issues.append({
                "code": exec_result.get("error_code") or "database.execution_error",
                "category": "database",
                "severity": "critical",
                "message": str(exec_result.get("error", "Database execution failed"))[:300],
                "evidence": str(exec_result.get("error", ""))[:300],
                "suggestion": "Repair the SQL using the database execution error",
            })

        retryable = not any(
            issue["code"] == "config.unsupported_dialect" for issue in issues
        )
        feedback = {
            "version": "1.0",
            "failure_stage": mode,
            "retryable": retryable,
            "categories": sorted({issue["category"] for issue in issues}),
            "issues": issues,
            "schema_search": {
                "required": False,
                "query_terms": [],
                "suggested_tables": [],
                "max_new_tables": 0,
            },
            "regeneration": {
                "mode": "sql_repair",
                "preserve_correct_parts": True,
                "original_question_is_authoritative": True,
            },
        }
        return {
            "judge_mode": mode,
            "overall_confidence": 0,
            "pass": False,
            "checks": [],
            "dimensions": {
                "syntax": {"score": 0, "max": 7, "issues": [], "suggestions": []},
                "semantics": {"score": 0, "max": 18, "issues": [], "suggestions": []},
                "logic": {"score": 0, "max": 12, "issues": [], "suggestions": []},
                "result_quality": {"score": 0, "max": 13, "issues": [], "suggestions": []},
            },
            "critical_flaws": [issue["message"] for issue in issues],
            "repair_priority": [issue["suggestion"] for issue in issues],
            "overall_assessment": f"Deterministic {mode} failure",
            "structured_feedback": feedback,
            "intent_comparison": {
                "available": False,
                "question_intent": {},
                "sql_intent": {},
                "match": None,
                "mismatches": [],
            },
            "deterministic_checks": gate.to_dict() if gate is not None else {},
            "reverse_question": "",
            "semantic_similarity": 0,
            "semantic_score": 0,
        }

    @staticmethod
    def _normalize_intent_comparison(comparison) -> dict:
        """补全指标/维度/过滤/时间范围的结构化语义契约。"""
        available = NL2SQLJudge._intent_comparison_present(comparison)
        comparison = comparison if isinstance(comparison, dict) else {}

        def normalize_intent(value):
            value = value if isinstance(value, dict) else {}
            return {
                "metrics": list(value.get("metrics", []) or []),
                "dimensions": list(value.get("dimensions", []) or []),
                "filters": list(value.get("filters", []) or []),
                "time_range": value.get("time_range") or {},
                "source_tables": list(value.get("source_tables", []) or []),
            }

        mismatches = []
        for mismatch in comparison.get("mismatches", []) or []:
            if not isinstance(mismatch, dict):
                continue
            mismatches.append({
                "component": mismatch.get("component", "unknown"),
                "expected": mismatch.get("expected"),
                "actual": mismatch.get("actual"),
                "severity": mismatch.get("severity", "major"),
                "feedback": mismatch.get("feedback", ""),
                "schema_search_terms": list(mismatch.get("schema_search_terms", []) or []),
                "suggested_tables": list(mismatch.get("suggested_tables", []) or []),
            })

        match = comparison.get("match")
        if match is None and mismatches:
            match = False
        return {
            "available": available,
            "question_intent": normalize_intent(comparison.get("question_intent")),
            "sql_intent": normalize_intent(comparison.get("sql_intent")),
            "draft_intent": normalize_intent(comparison.get("draft_intent")),
            "draft_intent_match": comparison.get("draft_intent_match"),
            "diagnosis": comparison.get("diagnosis", ""),
            "match": match,
            "mismatches": mismatches,
        }

    @staticmethod
    def _build_structured_feedback(glot: dict, comparison: dict, result: dict) -> dict:
        """把确定性失败和语义差异合并为版本化、可执行的修复反馈。"""
        issues = []
        categories = set()
        search_terms = set()
        suggested_tables = set()

        if not comparison.get("available"):
            categories.add("semantic")
            issues.append({
                "code": "semantic.intent_contract_missing",
                "category": "semantic",
                "severity": "critical",
                "message": "Judge did not return the required structured intent comparison",
                "evidence": "",
                "suggestion": "Re-evaluate metrics, dimensions, filters, and time range before repairing SQL",
            })

        if comparison.get("diagnosis") == "contract_issue" and not any(
            mismatch.get("component") == "draft_intent"
            for mismatch in comparison.get("mismatches", [])
        ):
            categories.add("semantic")
            issues.append({
                "code": "semantic.contract_issue",
                "category": "semantic",
                "severity": "critical",
                "message": "DraftIntent and SQL agree with each other but not with the original question",
                "evidence": {
                    "draft_intent": comparison.get("draft_intent", {}),
                    "sql_intent": comparison.get("sql_intent", {}),
                    "question_intent": comparison.get("question_intent", {}),
                },
                "suggestion": "Revise both DraftIntent and SQL using the original question as authority",
            })

        for check in glot.get("blocking_issues", []):
            code = check.get("code", "deterministic.unknown")
            category = code.split(".", 1)[0]
            categories.add(category)
            issues.append({
                "code": code,
                "category": category,
                "severity": "critical",
                "message": check.get("message", code),
                "evidence": check.get("evidence", ""),
                "suggestion": {
                    "syntax": "Fix the SQL syntax before semantic repair",
                    "permission": "Use only tables and columns present in the allowed schema",
                    "execution": "Repair the SQL using the execution error",
                    "result": "Align SELECT aliases and output columns with the requested result",
                    "aggregation": "Align non-aggregated outputs with GROUP BY",
                }.get(category, "Repair this deterministic validation failure"),
            })

        for index, mismatch in enumerate(comparison.get("mismatches", []), 1):
            component = mismatch["component"]
            categories.add("semantic")
            search_terms.update(str(term) for term in mismatch.get("schema_search_terms", []) if term)
            suggested_tables.update(str(table) for table in mismatch.get("suggested_tables", []) if table)
            issues.append({
                "code": f"semantic.{component}.{index}",
                "category": "semantic",
                "severity": mismatch.get("severity", "major"),
                "message": mismatch.get("feedback") or f"{component} intent does not match",
                "evidence": {
                    "expected": mismatch.get("expected"),
                    "actual": mismatch.get("actual"),
                },
                "suggestion": mismatch.get("feedback") or f"Align SQL {component} with the original question",
            })

        schema_required = bool(search_terms or suggested_tables)
        if glot.get("missing_tables") or glot.get("missing_columns"):
            schema_required = True
            search_terms.update(glot.get("missing_tables", []))
            search_terms.update(glot.get("missing_columns", []))

        retryable = bool(issues) and not any(
            issue["code"] == "permission.read_only" for issue in issues
        )
        return {
            "version": "1.0",
            "retryable": retryable,
            "categories": sorted(categories),
            "issues": issues,
            "schema_search": {
                "required": schema_required,
                "query_terms": sorted(search_terms)[:12],
                "suggested_tables": sorted(suggested_tables)[:6],
                "max_new_tables": 2,
            },
            "regeneration": {
                "mode": "restricted_reschema" if schema_required else "sql_repair",
                "preserve_correct_parts": True,
                "original_question_is_authoritative": True,
            },
        }

    def _call_llm(self, messages, max_retries=3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.model_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    n=1,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception:
                if attempt == max_retries - 1:
                    raise
        return "{}"

    def _parse_response(self, text: str) -> dict:
        """从 LLM 响应中解析评分 JSON（多层兜底）。"""
        original = text

        # 策略 A: ```json ... ``` 代码块（贪婪匹配以处理嵌套JSON）
        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # 策略 B: 直接解析
        try:
            return self._validate_result(json.loads(text))
        except json.JSONDecodeError:
            pass

        # 策略 C: 截取第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return self._validate_result(json.loads(text[start:end + 1]))
            except json.JSONDecodeError:
                pass

        # 解析失败 → 返回零分
        return {
            "overall_confidence": 0,
            "pass": False,
            "dimensions": {
                "syntax": {"score": 0, "max": 7, "issues": [], "suggestions": []},
                "semantics": {"score": 0, "max": 18, "issues": [], "suggestions": []},
                "logic": {"score": 0, "max": 12, "issues": [], "suggestions": []},
                "result_quality": {"score": 0, "max": 13, "issues": [], "suggestions": []},
            },
            "critical_flaws": ["Judge 响应解析失败"],
            "repair_priority": [],
            "overall_assessment": "",
            "raw_response": original[:500],
        }

    def _validate_result(self, raw: dict) -> dict:
        """补全缺失字段，确保返回结构完整。兼容新旧两种 Judge 输出格式。"""

        # ── 新格式适配：简化的 score/critical/fix → 旧格式 ──
        if "score" in raw and "overall_confidence" not in raw:
            raw["overall_confidence"] = raw["score"]
        if "critical" in raw and "critical_flaws" not in raw:
            raw["critical_flaws"] = raw["critical"]
        if "fix" in raw and "repair_priority" not in raw:
            raw["repair_priority"] = raw["fix"]

        # ── checks 格式适配：list → dict by category ──
        checks = raw.get("checks", {})
        if isinstance(checks, list):
            # 新格式: [{"id": "A1", "v": "YES", ...}, ...]
            # → 旧格式: {"syntax": [...], "semantics": [...], ...}
            new_checks = {}
            for c in checks:
                cid = c.get("id", "")
                if cid.startswith("A"):
                    new_checks.setdefault("syntax", []).append({
                        "id": cid, "question": c.get("e", ""),
                        "verdict": c.get("v", "YES"),
                        "evidence": c.get("e", ""),
                        "fix": c.get("f"),
                    })
                elif cid.startswith("B"):
                    new_checks.setdefault("semantics", []).append({
                        "id": cid, "question": c.get("e", ""),
                        "verdict": c.get("v", "YES"),
                        "evidence": c.get("e", ""),
                        "fix": c.get("f"),
                    })
                elif cid.startswith("C"):
                    new_checks.setdefault("logic", []).append({
                        "id": cid, "question": c.get("e", ""),
                        "verdict": c.get("v", "YES"),
                        "evidence": c.get("e", ""),
                        "fix": c.get("f"),
                    })
                elif cid.startswith("D"):
                    new_checks.setdefault("logic", []).append({
                        "id": cid, "question": c.get("e", ""),
                        "verdict": c.get("v", "YES"),
                        "evidence": c.get("e", ""),
                        "fix": c.get("f"),
                    })
                elif cid.startswith("E"):
                    new_checks.setdefault("syntax", []).append({
                        "id": cid, "question": c.get("e", ""),
                        "verdict": c.get("v", "YES"),
                        "evidence": c.get("e", ""),
                        "fix": c.get("f"),
                    })
                elif cid.startswith("F"):
                    new_checks.setdefault("result_quality", []).append({
                        "id": cid, "question": c.get("e", ""),
                        "verdict": c.get("v", "YES"),
                        "evidence": c.get("e", ""),
                        "fix": c.get("f"),
                    })
            raw["checks"] = new_checks

        dims = raw.get("dimensions", {})
        for key, default_max in [("syntax", 7), ("semantics", 18),
                                  ("logic", 12), ("result_quality", 13)]:
            if key not in dims:
                dims[key] = {"score": 0, "max": default_max,
                             "issues": [], "suggestions": []}
            else:
                dims[key].setdefault("score", 0)
                dims[key].setdefault("max", default_max)
                dims[key].setdefault("issues", [])
                dims[key].setdefault("suggestions", [])

        # 保留逐项核查记录
        raw.setdefault("checks", {})

        raw.setdefault("overall_confidence", 0)
        raw.setdefault("pass", False)
        raw.setdefault("critical_flaws", [])
        raw.setdefault("repair_priority", [])
        raw.setdefault("overall_assessment", "")
        raw["dimensions"] = dims
        raw["raw_response"] = raw.get("raw_response", "")
        return raw

    # ════════════════════════════════════════════════════════════════
    # 语义往返保真度（Semantic Round-Trip Fidelity）
    # ════════════════════════════════════════════════════════════════

    def _reverse_question(self, sql: str, relevant_schema: dict = None) -> str:
        """从 SQL 反向生成英文自然语言问题。

        调用 LLM，给定 SQL 和 Schema DDL，让模型写出"这条 SQL 在回答什么问题"。
        用于后续与原始问题进行向量相似度对比。

        参数:
            sql: 待反向解析的 SQL
            relevant_schema: 关联的表结构（用于渲染 DDL）

        返回:
            反向生成的英文问题字符串；失败时返回空字符串
        """
        from agent_team.judge_scoring_card import JudgeScoringCard

        ddl = ""
        if relevant_schema:
            ddl = JudgeScoringCard._render_schema_ddl(relevant_schema)

        system_prompt = (
            "You are a SQL expert. Given a SQL query and its database schema, "
            "write the English natural language question that this SQL query answers. "
            "Be precise and include all filtering conditions, aggregation logic, "
            "and output requirements evident in the SQL. "
            "Output ONLY the question, no explanations."
        )

        user_prompt = (
            f"## Database Schema\n```sql\n{ddl}\n```\n\n"
            f"## SQL Query\n```sql\n{sql}\n```\n\n"
            f"## Task\n"
            f"What English question does this SQL query answer? "
            f"Write the question in one sentence."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._call_llm(messages)
            # 清理：去掉可能的引号包裹和首尾空白
            q = response.strip().strip('"').strip("'").strip()
            return q
        except Exception:
            return ""

    @staticmethod
    def _compute_similarity(q1: str, q2: str) -> float:
        """计算两个英文问题的语义余弦相似度。

        使用 sentence-transformers 编码后计算余弦相似度。
        相似度范围 [0, 1]，越高表示语义越接近。

        参数:
            q1: 原始英文问题
            q2: 从 SQL 反向生成的问题

        返回:
            余弦相似度 float，范围 [0, 1]；编码失败时返回 0.0
        """
        if not q1 or not q2:
            return 0.0

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            emb1 = model.encode([q1], normalize_embeddings=True)[0]
            emb2 = model.encode([q2], normalize_embeddings=True)[0]
            # normalize_embeddings=True 后 dot product 即为 cosine similarity
            sim = float(np.dot(emb1, emb2))
            return max(0.0, min(1.0, sim))
        except ImportError:
            # sentence-transformers 未安装 → 回退到字符 bigram 相似度
            return NL2SQLJudge._char_bigram_similarity(q1, q2)
        except Exception:
            return 0.0

    @staticmethod
    def _char_bigram_similarity(q1: str, q2: str) -> float:
        """字符 bigram 的 Jaccard 相似度（sentence-transformers 不可用时的回退方案）。"""
        def bigrams(s):
            s = s.lower()
            return {s[i:i+2] for i in range(len(s) - 1)}

        b1 = bigrams(q1)
        b2 = bigrams(q2)
        if not b1 or not b2:
            return 0.0
        return len(b1 & b2) / len(b1 | b2)

    @staticmethod
    def _dimensions_empty(dims: dict) -> bool:
        """检查白盒维度分数是否全为空（LLM 未输出维度诊断）。"""
        if not dims:
            return True
        scores = [dims.get(k, {}).get("score", 0) for k in ("syntax", "semantics", "logic", "result_quality")]
        return all(s == 0 for s in scores)

    @staticmethod
    def _intent_comparison_present(comparison) -> bool:
        """判断 LLM 是否返回了完整的结构化意图比对骨架。"""
        if not isinstance(comparison, dict):
            return False
        return (
            isinstance(comparison.get("question_intent"), dict)
            and isinstance(comparison.get("sql_intent"), dict)
            and isinstance(comparison.get("mismatches"), list)
            and isinstance(comparison.get("match"), bool)
        )
