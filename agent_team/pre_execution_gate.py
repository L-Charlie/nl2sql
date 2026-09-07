"""Deterministic pre-execution checks for a single read-only SQL query."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import sqlglot
from sqlglot import exp

from agent_team.contracts import GateIssue, GateResult


class PreExecutionGate:
    """Fail closed on safety and high-confidence schema violations only."""

    SUPPORTED_DIALECTS = {"sqlite"}
    _PLACEHOLDER_RE = re.compile(
        r"\{\{[^{}]+\}\}|\$\{[^{}]+\}|<%[^%]+%>|\[\[[^\]]+\]\]"
    )

    def __init__(self, dialect: str = "sqlite"):
        self.dialect = (dialect or "").lower()

    def check(
        self,
        sql: str,
        relevant_schema: dict | None,
        full_schema: list[dict] | None = None,
        dialect: str | None = None,
    ) -> GateResult:
        dialect = (dialect or self.dialect or "").lower()
        blockers: list[GateIssue] = []
        warnings: list[GateIssue] = []
        signature = self._empty_signature()

        if dialect not in self.SUPPORTED_DIALECTS:
            blockers.append(GateIssue(
                "config.unsupported_dialect",
                f"Unsupported SQL dialect: {dialect or '<empty>'}",
                {"supported": sorted(self.SUPPORTED_DIALECTS)},
            ))
            return GateResult(False, dialect, blockers, warnings, signature)

        sql = (sql or "").strip()
        if not sql:
            blockers.append(GateIssue("syntax.empty_sql", "SQL is empty"))
            return GateResult(False, dialect, blockers, warnings, signature)

        placeholders = self._PLACEHOLDER_RE.findall(sql)
        if placeholders:
            blockers.append(GateIssue(
                "structure.unresolved_placeholder",
                "SQL contains unresolved template placeholders",
                {"placeholders": placeholders[:10]},
            ))

        try:
            statements = [item for item in sqlglot.parse(sql, dialect=dialect) if item is not None]
        except Exception as exc:
            blockers.append(GateIssue(
                "syntax.parse_error", "SQL could not be parsed reliably", str(exc)[:300]
            ))
            return GateResult(False, dialect, blockers, warnings, signature)

        if len(statements) != 1:
            blockers.append(GateIssue(
                "structure.multiple_statements",
                "Exactly one SQL statement is allowed",
                {"statement_count": len(statements)},
            ))
            return GateResult(False, dialect, blockers, warnings, signature)

        ast = statements[0]
        signature["statement_type"] = type(ast).__name__.lower()
        if not isinstance(ast, exp.Query) or self._contains_write_operation(ast):
            blockers.append(GateIssue(
                "safety.non_read_only",
                "Only a single read-only SELECT/WITH query is allowed",
                {"statement_type": type(ast).__name__},
            ))
            return GateResult(False, dialect, blockers, warnings, signature)

        selected_tables = self._schema_index(relevant_schema, full=False)
        allowed_tables = self._schema_index(full_schema, full=True)
        if not allowed_tables:
            allowed_tables = selected_tables

        tables, aliases, cte_names = self._table_references(ast)
        signature["tables"] = sorted(tables)
        missing_tables = sorted(table for table in tables if table not in allowed_tables)
        if missing_tables:
            blockers.append(GateIssue(
                "schema.table_not_authorized",
                "SQL references tables absent from the authorized schema",
                {"tables": missing_tables},
            ))

        outside_retrieval = sorted(
            table for table in tables if table in allowed_tables and table not in selected_tables
        )
        if outside_retrieval:
            warnings.append(GateIssue(
                "schema.outside_retrieval",
                "SQL uses authorized tables outside the retrieved schema",
                {"tables": outside_retrieval},
            ))

        missing_columns, ambiguous_columns = self._missing_columns(
            ast, tables, aliases, cte_names, allowed_tables
        )
        if missing_columns:
            blockers.append(GateIssue(
                "schema.column_not_authorized",
                "SQL references columns absent from the authorized schema",
                {"columns": missing_columns[:20]},
            ))
        if ambiguous_columns:
            warnings.append(GateIssue(
                "schema.ambiguous_unqualified_column",
                "Unqualified columns have multiple possible table owners",
                {"columns": ambiguous_columns[:20]},
            ))

        signature.update(self._build_signature(ast, dialect))
        invalid_outputs = signature.get("aggregation", {}).get("invalid_non_aggregates", [])
        if invalid_outputs:
            warnings.append(GateIssue(
                "aggregation.grain_observation",
                "Non-aggregated outputs are not present in GROUP BY",
                {"expressions": invalid_outputs},
            ))

        return GateResult(not blockers, dialect, blockers, warnings, signature)

    @staticmethod
    def _empty_signature() -> dict[str, Any]:
        return {
            "statement_type": "",
            "tables": [],
            "projected_columns": [],
            "aggregation": {
                "has_aggregation": False,
                "aggregate_functions": [],
                "group_by": [],
                "invalid_non_aggregates": [],
            },
            "joins": [],
            "filters": [],
            "order_by": [],
            "limit": None,
        }

    @staticmethod
    def _contains_write_operation(ast: exp.Expression) -> bool:
        write_names = ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Merge", "Command")
        write_types = tuple(
            cls for name in write_names if (cls := getattr(exp, name, None)) is not None
        )
        return bool(write_types and any(ast.find_all(*write_types)))

    @staticmethod
    def _schema_index(schema: Any, full: bool) -> dict[str, set[str]]:
        tables: dict[str, set[str]] = {}
        if full:
            source = schema or []
            for table in source:
                name = str(table.get("table_name", table.get("name", ""))).lower()
                if not name:
                    continue
                columns = table.get("columns", table.get("relevant_columns", []))
                tables[name] = {
                    str(column.get("col", column.get("name", ""))).lower()
                    for column in columns
                    if column.get("col", column.get("name", ""))
                }
            return tables

        source = (schema or {}).get("candidate_tables", [])
        for table in source:
            name = str(table.get("name", table.get("table_name", ""))).lower()
            if not name:
                continue
            tables[name] = {
                str(column.get("name", column.get("col", ""))).lower()
                for column in table.get("relevant_columns", table.get("columns", []))
                if column.get("name", column.get("col", ""))
            }
        return tables

    @staticmethod
    def _table_references(ast: exp.Expression) -> tuple[set[str], dict[str, str], set[str]]:
        cte_names = {
            cte.alias.lower() for cte in ast.find_all(exp.CTE) if cte.alias
        }
        tables: set[str] = set()
        aliases: dict[str, str] = {}
        for node in ast.find_all(exp.Table):
            name = node.name.lower() if node.name else ""
            if not name:
                continue
            aliases[(node.alias_or_name or name).lower()] = name
            if name not in cte_names:
                tables.add(name)
        return tables, aliases, cte_names

    @staticmethod
    def _missing_columns(
        ast: exp.Expression,
        tables: set[str],
        aliases: dict[str, str],
        cte_names: set[str],
        allowed_tables: dict[str, set[str]],
    ) -> tuple[list[str], list[str]]:
        owners: dict[str, set[str]] = defaultdict(set)
        for table, columns in allowed_tables.items():
            for column in columns:
                owners[column].add(table)

        select_aliases = {
            expression.alias.lower()
            for select in ast.find_all(exp.Select)
            for expression in select.expressions
            if getattr(expression, "alias", "")
        }
        derived_columns: dict[str, set[str]] = {}
        for cte in ast.find_all(exp.CTE):
            cte_select = cte.this.find(exp.Select)
            if cte_select and cte.alias:
                derived_columns[cte.alias.lower()] = {
                    expression.alias_or_name.lower()
                    for expression in cte_select.expressions
                    if expression.alias_or_name
                }

        missing: set[str] = set()
        ambiguous: set[str] = set()
        all_derived = set().union(*derived_columns.values()) if derived_columns else set()
        for node in ast.find_all(exp.Column):
            column = node.name.lower() if node.name else ""
            qualifier = node.table.lower() if node.table else ""
            if not column or column == "*":
                continue
            if qualifier:
                table = aliases.get(qualifier, qualifier)
                if table in cte_names:
                    if column not in derived_columns.get(table, set()):
                        missing.add(f"{table}.{column}")
                elif table in allowed_tables and column not in allowed_tables[table]:
                    missing.add(f"{table}.{column}")
                continue
            if column in select_aliases or column in all_derived:
                continue
            possible = owners.get(column, set()) & tables
            if not possible:
                missing.add(column)
            elif len(possible) > 1 and len(tables) > 1:
                ambiguous.add(column)
        return sorted(missing), sorted(ambiguous)

    @staticmethod
    def _build_signature(ast: exp.Expression, dialect: str) -> dict[str, Any]:
        root_select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        projected = []
        if root_select:
            projected = [
                expression.alias_or_name or expression.sql(dialect=dialect)
                for expression in root_select.expressions
            ]

        aggregates = [node.sql_name() for node in ast.find_all(exp.AggFunc)]
        group_by = []
        invalid_non_aggregates = []
        if root_select:
            group = root_select.args.get("group")
            group_expressions = list(group.expressions) if group else []
            group_by = [item.sql(dialect=dialect) for item in group_expressions]
            if aggregates:
                normalized_group = {item.lower() for item in group_by}
                for expression in root_select.expressions:
                    target = expression.this if isinstance(expression, exp.Alias) else expression
                    if isinstance(target, exp.Literal) or any(target.find_all(exp.AggFunc)):
                        continue
                    if target.sql(dialect=dialect).lower() not in normalized_group:
                        invalid_non_aggregates.append(target.sql(dialect=dialect))

        joins = [join.sql(dialect=dialect) for join in ast.find_all(exp.Join)]
        where = ast.args.get("where") if hasattr(ast, "args") else None
        order = ast.args.get("order") if hasattr(ast, "args") else None
        limit = ast.args.get("limit") if hasattr(ast, "args") else None
        return {
            "projected_columns": projected,
            "aggregation": {
                "has_aggregation": bool(aggregates),
                "aggregate_functions": aggregates,
                "group_by": group_by,
                "invalid_non_aggregates": invalid_non_aggregates,
            },
            "joins": joins,
            "filters": [where.this.sql(dialect=dialect)] if where is not None else [],
            "order_by": [item.sql(dialect=dialect) for item in order.expressions] if order else [],
            "limit": limit.expression.this if limit and limit.expression else None,
        }
