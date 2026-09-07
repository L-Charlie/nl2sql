"""Database-backed schema discovery for the NL2SQL pipeline.

Discoverers translate database metadata into the schema structure accepted by
``Orchestrator``.  They do not execute user queries and should use read-only
connections whenever the database driver supports them.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote


class SchemaDiscoveryError(RuntimeError):
    """Raised when database metadata cannot be converted into a schema."""


class SchemaDiscoverer(ABC):
    """Interface for converting live database metadata into agent schema data."""

    @abstractmethod
    def discover(self, table_names: Sequence[str] | None = None) -> list[dict]:
        """Discover all tables, or only the explicitly requested table names."""


class SQLiteSchemaDiscoverer(SchemaDiscoverer):
    """Discover table, column, primary-key, and foreign-key metadata from SQLite."""

    def __init__(self, db_path: str | Path, timeout: float = 10.0):
        self.db_path = Path(db_path).expanduser().resolve()
        self.timeout = timeout

    def discover(self, table_names: Sequence[str] | None = None) -> list[dict]:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")

        connection = self._connect_read_only()
        try:
            available_tables = self._load_table_names(connection)
            selected_tables = self._select_tables(available_tables, table_names)
            return [self._describe_table(connection, name) for name in selected_tables]
        except sqlite3.DatabaseError as exc:
            raise SchemaDiscoveryError(
                f"Unable to discover SQLite schema from {self.db_path}: {exc}"
            ) from exc
        finally:
            connection.close()

    def _connect_read_only(self) -> sqlite3.Connection:
        database_uri = f"file:{quote(self.db_path.as_posix(), safe='/:')}?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    @staticmethod
    def _load_table_names(connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [str(row["name"]) for row in rows]

    @staticmethod
    def _select_tables(
        available_tables: Sequence[str],
        requested_tables: Sequence[str] | None,
    ) -> list[str]:
        if requested_tables is None:
            return list(available_tables)

        canonical_names = {name.casefold(): name for name in available_tables}
        selected = []
        missing = []
        seen = set()
        for requested in requested_tables:
            requested_name = str(requested).strip()
            canonical = canonical_names.get(requested_name.casefold())
            if canonical is None:
                missing.append(requested_name)
                continue
            if canonical not in seen:
                selected.append(canonical)
                seen.add(canonical)

        if missing:
            raise SchemaDiscoveryError(
                "Requested SQLite tables were not found: " + ", ".join(missing)
            )
        return selected

    def _describe_table(self, connection: sqlite3.Connection, table_name: str) -> dict:
        sql_name = self._as_sql_string(table_name)
        column_rows = connection.execute(f"PRAGMA table_xinfo({sql_name})").fetchall()
        foreign_key_rows = connection.execute(
            f"PRAGMA foreign_key_list({sql_name})"
        ).fetchall()

        foreign_keys: dict[str, list[str]] = {}
        foreign_key_metadata = []
        for row in foreign_key_rows:
            source_column = str(row["from"])
            target_table = str(row["table"])
            target_column = str(row["to"])
            target = f"{target_table}.{target_column}"
            foreign_keys.setdefault(source_column, []).append(target)
            foreign_key_metadata.append({
                "from": source_column,
                "to_table": target_table,
                "to_column": target_column,
            })

        columns = []
        for row in column_rows:
            # hidden=1 represents virtual-table implementation columns. Generated
            # columns (hidden=2/3) remain queryable and should stay in the schema.
            if int(row["hidden"]) == 1:
                continue
            column_name = str(row["name"])
            markers = []
            if int(row["pk"]) > 0:
                markers.append("PK")
            markers.extend(
                f"FK->{target}" for target in foreign_keys.get(column_name, [])
            )
            columns.append({
                "col": column_name,
                "type": str(row["type"] or "UNKNOWN").upper(),
                "description": " ".join(markers),
            })

        return {
            "table_name": table_name,
            "table_description": "",
            "columns": columns,
            "foreign_keys": foreign_key_metadata,
        }

    @staticmethod
    def _as_sql_string(value: str) -> str:
        """Return a safely quoted SQLite string literal for PRAGMA arguments."""
        return "'" + value.replace("'", "''") + "'"


def discover_sqlite_schema(
    db_path: str | Path,
    table_names: Sequence[str] | None = None,
) -> list[dict]:
    """Convenience API for one-shot SQLite schema discovery."""
    return SQLiteSchemaDiscoverer(db_path).discover(table_names=table_names)
