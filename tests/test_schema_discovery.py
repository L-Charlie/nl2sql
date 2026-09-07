import sqlite3

import pytest

from agent_team.schema_discovery import (
    SchemaDiscoveryError,
    SQLiteSchemaDiscoverer,
    discover_sqlite_schema,
)


def create_database(tmp_path):
    path = tmp_path / "schema discovery.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "CREATE TABLE users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE orders ("
        "order_id INTEGER PRIMARY KEY, "
        "user_id INTEGER NOT NULL, "
        "amount NUMERIC, "
        "FOREIGN KEY (user_id) REFERENCES users(id))"
    )
    connection.commit()
    connection.close()
    return path


def test_discovers_tables_columns_primary_keys_and_foreign_keys(tmp_path):
    path = create_database(tmp_path)

    schema = SQLiteSchemaDiscoverer(path).discover()

    assert [table["table_name"] for table in schema] == ["orders", "users"]
    assert all(table["table_name"] != "sqlite_sequence" for table in schema)

    orders = schema[0]
    columns = {column["col"]: column for column in orders["columns"]}
    assert columns["order_id"] == {
        "col": "order_id",
        "type": "INTEGER",
        "description": "PK",
    }
    assert columns["user_id"]["description"] == "FK->users.id"
    assert columns["amount"]["type"] == "NUMERIC"
    assert orders["foreign_keys"] == [{
        "from": "user_id",
        "to_table": "users",
        "to_column": "id",
    }]


def test_requested_tables_are_a_case_insensitive_authorization_boundary(tmp_path):
    path = create_database(tmp_path)

    schema = discover_sqlite_schema(path, table_names=["USERS", "users"])

    assert [table["table_name"] for table in schema] == ["users"]


def test_empty_table_selection_discovers_nothing(tmp_path):
    path = create_database(tmp_path)

    schema = discover_sqlite_schema(path, table_names=[])

    assert schema == []


def test_unknown_requested_table_is_reported(tmp_path):
    path = create_database(tmp_path)

    with pytest.raises(SchemaDiscoveryError, match="missing_table"):
        SQLiteSchemaDiscoverer(path).discover(["users", "missing_table"])


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError):
        SQLiteSchemaDiscoverer(path).discover()

    assert not path.exists()
