from agent_team.knowledge_graph_builder import KnowledgeGraphBuilder
from agent_team.schema_linker import SchemaLinker


def column(name, col_type="INTEGER", description=""):
    return {"col": name, "type": col_type, "description": description}


def test_declared_fk_expands_with_auditable_score():
    schema = [
        {
            "table_name": "orders",
            "table_description": "订单金额",
            "columns": [column("order_id", description="PK"), column("user_id")],
            "foreign_keys": [
                {"from": "user_id", "to_table": "users", "to_column": "id"}
            ],
        },
        {
            "table_name": "users",
            "table_description": "用户",
            "columns": [column("id", description="PK"), column("display_name", "TEXT")],
        },
    ]

    result = SchemaLinker(schema).link("订单金额", top_tables=1)

    assert result["fk_expansion"]["before_tables"] == ["orders"]
    assert result["fk_expansion"]["added_tables"] == ["users"]
    users = next(table for table in result["candidate_tables"] if table["name"] == "users")
    assert users["retrieval_reason"] == "fk_expansion"
    assert users["edge_type"] == "DECLARED_FK"
    assert users["edge_confidence"] == 1.0
    assert users["expansion_score"] == 0.85
    assert result["join_paths"] == [{
        "from": "orders",
        "to": "users",
        "on": ["user_id=id"],
        "from_column": "user_id",
        "to_column": "id",
        "edge_type": "DECLARED_FK",
        "edge_confidence": 1.0,
        "confidence": "high",
        "evidence": ["database_foreign_key"],
    }]


def test_schema_marker_is_stronger_than_convention_inference():
    schema = [
        {
            "table_name": "orders",
            "table_description": "订单金额",
            "columns": [
                column("order_id", description="PK"),
                column("user_id", description="FK->users.id"),
            ],
        },
        {
            "table_name": "users",
            "columns": [column("id", description="PK")],
        },
    ]

    graph = KnowledgeGraphBuilder(schema).build()
    relationships = [
        attrs for _, _, attrs in graph.edges(data=True)
        if attrs.get("relationship") == "FK_RELATIONSHIP"
    ]

    assert len(relationships) == 1
    assert relationships[0]["edge_type"] == "SCHEMA_FK"
    assert relationships[0]["confidence"] == 0.95


def test_naming_inference_requires_target_key_and_type_compatibility():
    schema = [
        {
            "table_name": "orders",
            "table_description": "订单金额",
            "columns": [column("user_id")],
        },
        {
            "table_name": "users",
            "columns": [column("id", description="PK")],
        },
        {
            "table_name": "products",
            "columns": [column("id", "TEXT", description="PK")],
        },
    ]

    result = SchemaLinker(schema).link("订单金额", top_tables=1)

    assert result["fk_expansion"]["added_tables"] == ["users"]
    assert result["fk_expansion"]["details"][0]["edge_type"] == "INFERRED_FK"
    assert result["fk_expansion"]["details"][0]["edge_confidence"] == 0.8


def test_same_name_columns_are_weak_and_never_auto_expand():
    schema = [
        {
            "table_name": "orders",
            "table_description": "订单金额",
            "columns": [column("tenant_code", "TEXT"), column("id", description="PK")],
        },
        {
            "table_name": "audit_logs",
            "columns": [column("tenant_code", "TEXT"), column("id", description="PK")],
        },
    ]

    result = SchemaLinker(schema).link("订单金额", top_tables=1)
    graph = SchemaLinker(schema).knowledge_graph
    weak_edges = [
        attrs for _, _, attrs in graph.edges(data=True)
        if attrs.get("edge_type") == "WEAK_SIMILARITY"
    ]

    assert result["fk_expansion"]["added_tables"] == []
    assert result["join_paths"] == []
    assert len(weak_edges) == 1
    assert weak_edges[0]["confidence"] == 0.1


def test_expansion_respects_per_seed_and_total_candidate_limits():
    targets = []
    foreign_keys = []
    source_columns = [column("order_id", description="PK")]
    for index in range(4):
        table_name = f"dimension_{index}"
        source_column = f"dimension_{index}_id"
        source_columns.append(column(source_column))
        foreign_keys.append({
            "from": source_column,
            "to_table": table_name,
            "to_column": "id",
        })
        targets.append({
            "table_name": table_name,
            "columns": [column("id", description="PK")],
        })
    schema = [{
        "table_name": "orders",
        "table_description": "订单金额",
        "columns": source_columns,
        "foreign_keys": foreign_keys,
    }] + targets

    result = SchemaLinker(schema).link("订单金额", top_tables=1)

    assert len(result["fk_expansion"]["added_tables"]) == 2
    assert len(result["candidate_tables"]) == 3
