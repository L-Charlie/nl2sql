"""Safe SQL execution: deterministic Gate followed by a read-only executor."""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from agent_team.contracts import ExecutionAttempt, GateResult, SQLArtifact
from agent_team.pre_execution_gate import PreExecutionGate


class DatabaseExecutor(ABC):
    read_only_enforced = False

    @abstractmethod
    def execute(self, sql: str, mode: str = "probe") -> dict[str, Any]:
        raise NotImplementedError


class SQLiteReadOnlyExecutor(DatabaseExecutor):
    """SQLite executor protected by URI read-only mode and query_only."""

    read_only_enforced = True

    def __init__(self, db_path: str, probe_rows: int = 50, timeout: float = 10.0):
        self.db_path = str(db_path)
        self.probe_rows = max(1, int(probe_rows))
        self.timeout = timeout

    def execute(self, sql: str, mode: str = "probe") -> dict[str, Any]:
        if mode not in {"probe", "full"}:
            return {
                "ok": False,
                "error": f"Execution mode is not supported: {mode}",
                "error_code": "execution.mode_not_supported",
                "read_only_enforced": True,
            }

        started = time.perf_counter()
        connection = None
        try:
            absolute_path = Path(self.db_path).expanduser().resolve().as_posix()
            uri = f"file:{quote(absolute_path, safe='/:')}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=self.timeout)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            cursor = connection.execute((sql or "").strip())
            columns = [item[0] for item in (cursor.description or [])]

            if mode == "probe":
                raw_rows = cursor.fetchmany(self.probe_rows + 1)
                truncated = len(raw_rows) > self.probe_rows
                raw_rows = raw_rows[:self.probe_rows]
            else:
                raw_rows = cursor.fetchall()
                truncated = False

            rows = [dict(row) for row in raw_rows]
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return {
                "ok": True,
                "error": "",
                "error_code": "",
                "columns": columns,
                "sample_rows": rows if mode == "probe" else rows[:self.probe_rows],
                "rows": rows if mode == "full" else [],
                "truncated": truncated,
                "row_count": len(rows),
                "row_count_exact": not truncated,
                "execution_ms": elapsed,
                "result_profile": self._profile(columns, rows),
                "read_only_enforced": True,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "error_code": "database.execution_error",
                "columns": [],
                "sample_rows": [],
                "rows": [],
                "truncated": False,
                "row_count": 0,
                "row_count_exact": False,
                "execution_ms": round((time.perf_counter() - started) * 1000, 3),
                "result_profile": None,
                "read_only_enforced": True,
            }
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _profile(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "sample_size": len(rows),
            "column_count": len(columns),
            "columns": {
                column: {
                    "sample_null_count": sum(row.get(column) is None for row in rows),
                    "sample_value_types": sorted({
                        type(row.get(column)).__name__
                        for row in rows
                        if row.get(column) is not None
                    }),
                }
                for column in columns
            },
        }


class CallbackExecutorAdapter(DatabaseExecutor):
    """Compatibility adapter. The callback itself cannot prove read-only safety."""

    read_only_enforced = False

    def __init__(self, callback: Callable[[str], dict], probe_rows: int = 50):
        self.callback = callback
        self.probe_rows = max(1, int(probe_rows))

    def execute(self, sql: str, mode: str = "probe") -> dict[str, Any]:
        if mode not in {"probe", "full"}:
            return {
                "ok": False,
                "error": f"Execution mode is not supported: {mode}",
                "error_code": "execution.mode_not_supported",
                "read_only_enforced": False,
            }
        started = time.perf_counter()
        try:
            result = dict(self.callback(sql) or {})
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}

        rows = list(result.get("rows", []) or result.get("sample_rows", []) or [])
        columns = list(result.get("columns", []) or [])
        if not columns and rows and isinstance(rows[0], dict):
            columns = list(rows[0].keys())
        truncated = mode == "probe" and len(rows) > self.probe_rows
        sample_rows = rows[:self.probe_rows]
        raw_count = result.get("row_count")
        result.update({
            "ok": bool(result.get("ok", False)),
            "error": str(result.get("error", "") or ""),
            "error_code": str(result.get("error_code", "") or ""),
            "columns": columns,
            "sample_rows": sample_rows,
            "rows": rows if mode == "full" else [],
            "truncated": bool(result.get("truncated", truncated)),
            "row_count": raw_count if raw_count is not None else len(rows),
            "row_count_exact": bool(result.get("row_count_exact", mode == "full")),
            "execution_ms": result.get(
                "execution_ms", round((time.perf_counter() - started) * 1000, 3)
            ),
            "result_profile": result.get("result_profile"),
            "read_only_enforced": False,
        })
        return result


class SafeExecutor:
    """The only public execution path for generated SQL."""

    def __init__(
        self,
        database_executor: DatabaseExecutor,
        gate: PreExecutionGate | None = None,
        dialect: str = "sqlite",
    ):
        self.database_executor = database_executor
        self.dialect = dialect
        self.gate = gate or PreExecutionGate(dialect=dialect)

    def attempt(
        self,
        artifact_or_sql: SQLArtifact | str,
        relevant_schema: dict | None,
        full_schema: list[dict] | None,
        mode: str = "probe",
    ) -> ExecutionAttempt:
        sql = artifact_or_sql.sql if isinstance(artifact_or_sql, SQLArtifact) else str(artifact_or_sql or "")
        gate_result = self.gate.check(
            sql, relevant_schema=relevant_schema, full_schema=full_schema, dialect=self.dialect
        )
        if not gate_result.passed:
            return ExecutionAttempt(
                stage="gate", attempted=False, ok=False, mode=mode, gate=gate_result,
                error=gate_result.blockers[0].message if gate_result.blockers else "Gate blocked SQL",
                error_code=gate_result.blockers[0].code if gate_result.blockers else "gate.blocked",
                read_only_enforced=self.database_executor.read_only_enforced,
            )

        raw = self.database_executor.execute(sql, mode=mode)
        return ExecutionAttempt(
            stage="database",
            attempted=True,
            ok=bool(raw.get("ok", False)),
            mode=mode,
            gate=gate_result,
            columns=list(raw.get("columns", []) or []),
            sample_rows=list(raw.get("sample_rows", []) or []),
            rows=list(raw.get("rows", []) or []),
            truncated=bool(raw.get("truncated", False)),
            row_count=raw.get("row_count"),
            row_count_exact=bool(raw.get("row_count_exact", False)),
            execution_ms=raw.get("execution_ms"),
            result_profile=raw.get("result_profile"),
            error=str(raw.get("error", "") or ""),
            error_code=str(raw.get("error_code", "") or ""),
            read_only_enforced=bool(raw.get("read_only_enforced", False)),
        )
