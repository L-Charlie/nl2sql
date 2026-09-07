import sqlite3

from agent_team.executor import (
    CallbackExecutorAdapter,
    SQLiteReadOnlyExecutor,
    SafeExecutor,
)


FULL_SCHEMA = [{
    "table_name": "items",
    "columns": [
        {"col": "id", "type": "INTEGER"},
        {"col": "name", "type": "TEXT"},
    ],
}]
RELEVANT_SCHEMA = {
    "candidate_tables": [{
        "name": "items",
        "relevant_columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"},
        ],
    }]
}


def create_database(tmp_path):
    path = tmp_path / "safe executor.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    connection.executemany(
        "INSERT INTO items VALUES (?, ?)",
        [(1, "one"), (2, "two"), (3, None)],
    )
    connection.commit()
    connection.close()
    return path


def test_gate_failure_never_calls_database_callback():
    calls = []
    safe = SafeExecutor(CallbackExecutorAdapter(lambda sql: calls.append(sql) or {"ok": True}))

    attempt = safe.attempt("UPDATE items SET name = 'x'", RELEVANT_SCHEMA, FULL_SCHEMA)

    assert attempt.stage == "gate"
    assert attempt.attempted is False
    assert calls == []


def test_sqlite_probe_is_bounded_and_profiled(tmp_path):
    path = create_database(tmp_path)
    safe = SafeExecutor(SQLiteReadOnlyExecutor(str(path), probe_rows=2))

    attempt = safe.attempt(
        "SELECT id, name FROM items ORDER BY id", RELEVANT_SCHEMA, FULL_SCHEMA
    )

    assert attempt.ok is True
    assert attempt.attempted is True
    assert attempt.truncated is True
    assert attempt.row_count == 2
    assert attempt.row_count_exact is False
    assert len(attempt.sample_rows) == 2
    assert attempt.rows == []
    assert attempt.result_profile["column_count"] == 2
    assert attempt.read_only_enforced is True


def test_sqlite_full_mode_returns_all_rows(tmp_path):
    path = create_database(tmp_path)
    safe = SafeExecutor(SQLiteReadOnlyExecutor(str(path), probe_rows=2))

    attempt = safe.attempt(
        "SELECT id, name FROM items ORDER BY id", RELEVANT_SCHEMA, FULL_SCHEMA, mode="full"
    )

    assert attempt.ok is True
    assert len(attempt.rows) == 3
    assert attempt.row_count == 3
    assert attempt.row_count_exact is True


def test_database_executor_rejects_write_even_without_gate(tmp_path):
    path = create_database(tmp_path)
    executor = SQLiteReadOnlyExecutor(str(path))

    result = executor.execute("DELETE FROM items", mode="full")

    assert result["ok"] is False
    verify = sqlite3.connect(path).execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert verify == 3


def test_unsupported_mode_is_explicit(tmp_path):
    path = create_database(tmp_path)
    safe = SafeExecutor(SQLiteReadOnlyExecutor(str(path)))

    attempt = safe.attempt(
        "SELECT * FROM items", RELEVANT_SCHEMA, FULL_SCHEMA, mode="paged"
    )

    assert attempt.ok is False
    assert attempt.error_code == "execution.mode_not_supported"


def test_callback_probe_does_not_claim_exact_row_count():
    safe = SafeExecutor(CallbackExecutorAdapter(lambda sql: {
        "ok": True,
        "rows": [{"id": 1}],
        "row_count": 1,
    }))

    attempt = safe.attempt("SELECT id FROM items", RELEVANT_SCHEMA, FULL_SCHEMA)

    assert attempt.ok is True
    assert attempt.row_count_exact is False
    assert attempt.read_only_enforced is False
