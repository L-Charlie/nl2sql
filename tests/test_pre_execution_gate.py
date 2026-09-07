from agent_team.pre_execution_gate import PreExecutionGate


FULL_SCHEMA = [
    {
        "table_name": "orders",
        "columns": [
            {"col": "user_id", "type": "INTEGER"},
            {"col": "amount", "type": "REAL"},
        ],
    },
    {
        "table_name": "users",
        "columns": [
            {"col": "id", "type": "INTEGER"},
            {"col": "region", "type": "TEXT"},
        ],
    },
]

RELEVANT_SCHEMA = {
    "candidate_tables": [{
        "name": "orders",
        "relevant_columns": [
            {"name": "user_id", "type": "INTEGER"},
            {"name": "amount", "type": "REAL"},
        ],
    }]
}


def issue_codes(result, kind="blockers"):
    return {item.code for item in getattr(result, kind)}


def test_gate_allows_single_read_only_query():
    result = PreExecutionGate().check(
        "SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
    )

    assert result.passed is True
    assert result.blockers == []
    assert result.sql_signature["aggregation"]["aggregate_functions"] == ["SUM"]


def test_gate_blocks_writes_and_multiple_statements():
    gate = PreExecutionGate()

    write = gate.check("DELETE FROM orders", RELEVANT_SCHEMA, FULL_SCHEMA)
    multiple = gate.check(
        "SELECT * FROM orders; DELETE FROM orders", RELEVANT_SCHEMA, FULL_SCHEMA
    )

    assert issue_codes(write) == {"safety.non_read_only"}
    assert issue_codes(multiple) == {"structure.multiple_statements"}


def test_gate_uses_full_schema_as_authorization_boundary():
    allowed = PreExecutionGate().check(
        "SELECT region FROM users", RELEVANT_SCHEMA, FULL_SCHEMA
    )
    missing = PreExecutionGate().check(
        "SELECT secret FROM users", RELEVANT_SCHEMA, FULL_SCHEMA
    )

    assert allowed.passed is True
    assert "schema.outside_retrieval" in issue_codes(allowed, "warnings")
    assert "schema.column_not_authorized" in issue_codes(missing)


def test_aggregation_observation_never_blocks_gate():
    result = PreExecutionGate().check(
        "SELECT user_id, amount, SUM(amount) FROM orders GROUP BY user_id",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
    )

    assert result.passed is True
    assert "aggregation.grain_observation" in issue_codes(result, "warnings")


def test_gate_blocks_placeholders_and_unsupported_dialect():
    placeholder = PreExecutionGate().check(
        "SELECT * FROM orders WHERE user_id = {{ user_id }}",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
    )
    unsupported = PreExecutionGate().check(
        "SELECT * FROM orders", RELEVANT_SCHEMA, FULL_SCHEMA, dialect="mysql"
    )

    assert "structure.unresolved_placeholder" in issue_codes(placeholder)
    assert issue_codes(unsupported) == {"config.unsupported_dialect"}
