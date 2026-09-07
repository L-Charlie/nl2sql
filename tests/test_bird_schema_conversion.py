from benchmark import _convert_bird_schema


def test_bird_foreign_keys_keep_target_table_and_column():
    entry = {
        "table_names_original": ["users", "orders"],
        "column_names_original": [
            [-1, "*"],
            [0, "id"],
            [1, "order_id"],
            [1, "user_id"],
        ],
        "column_types": ["text", "number", "number", "number"],
        "primary_keys": [1, 2],
        "foreign_keys": [[3, 1]],
    }

    schema = _convert_bird_schema(entry)

    orders = next(table for table in schema if table["table_name"] == "orders")
    user_id = next(column for column in orders["columns"] if column["col"] == "user_id")
    assert user_id["description"] == "FK->users.id"
    assert orders["foreign_keys"] == [{
        "from": "user_id",
        "to_table": "users",
        "to_column": "id",
    }]
