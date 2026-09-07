import json

from agent_team.nl2sql_judge import NL2SQLJudge
from agent_team.refiner import Refiner
from agent_team.schema_linker import SchemaLinker


FULL_SCHEMA = [
    {
        "table_name": "orders",
        "table_description": "customer orders",
        "columns": [
            {"col": "user_id", "type": "INTEGER", "description": ""},
            {"col": "amount", "type": "REAL", "description": ""},
            {"col": "created_at", "type": "TEXT", "description": ""},
        ],
    },
    {
        "table_name": "users",
        "table_description": "registered users and regions",
        "columns": [
            {"col": "id", "type": "INTEGER", "description": "PK"},
            {"col": "name", "type": "TEXT", "description": ""},
            {"col": "region", "type": "TEXT", "description": ""},
        ],
    },
    {
        "table_name": "regions",
        "table_description": "region lookup",
        "columns": [
            {"col": "region", "type": "TEXT", "description": "PK"},
            {"col": "country", "type": "TEXT", "description": ""},
        ],
    },
    {
        "table_name": "audit_log",
        "table_description": "system audit events",
        "columns": [{"col": "event", "type": "TEXT", "description": ""}],
    },
]

RELEVANT_SCHEMA = {
    "question": "total order amount by region",
    "candidate_tables": [
        {
            "name": "orders",
            "relevant_columns": [
                {"name": "user_id", "type": "INTEGER"},
                {"name": "amount", "type": "REAL"},
                {"name": "created_at", "type": "TEXT"},
            ],
        }
    ],
}


def test_deterministic_checks_cover_schema_result_and_grain():
    judge = NL2SQLJudge()
    valid = judge._verify_sqlglot(
        "SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
        {
            "ok": True,
            "columns": ["user_id", "total"],
            "sample_rows": [{"user_id": 1, "total": 10.0}],
        },
    )
    assert valid["blocking_issues"] == []
    assert valid["result_schema"]["matches"] is True
    assert valid["aggregation_grain"]["group_by"] == ["user_id"]
    assert valid["aggregation_grain"]["aggregate_functions"] == ["SUM"]

    invalid = judge._verify_sqlglot(
        "SELECT user_id, amount, SUM(amount) AS total FROM orders GROUP BY user_id",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
        {
            "ok": True,
            "columns": ["user_id", "amount", "total"],
            "sample_rows": [{"user_id": 1, "amount": 10.0, "total": 10.0}],
        },
    )
    assert "aggregation.grain_observation" in {
        check["code"] for check in invalid["checks"]
    }
    assert "aggregation.grain_observation" not in {
        issue["code"] for issue in invalid["blocking_issues"]
    }

    ordinal_group = judge._verify_sqlglot(
        "SELECT user_id, SUM(amount) AS total FROM orders GROUP BY 1",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
        {"ok": True, "columns": ["user_id", "total"]},
    )
    assert "aggregation.grain_observation" not in {
        issue["code"] for issue in ordinal_group["blocking_issues"]
    }

    cte = judge._verify_sqlglot(
        "WITH totals AS (SELECT user_id, SUM(amount) AS total "
        "FROM orders GROUP BY user_id) SELECT user_id, total FROM totals",
        RELEVANT_SCHEMA,
        FULL_SCHEMA,
        {"ok": True, "columns": ["user_id", "total"]},
    )
    assert not {"schema.column_not_authorized", "aggregation.grain_observation"} & {
        issue["code"] for issue in cte["blocking_issues"]
    }


def test_gate_and_database_failures_skip_judge_llm(monkeypatch):
    judge = NL2SQLJudge()
    calls = []
    monkeypatch.setattr(judge, "_call_llm", lambda messages: calls.append(messages))

    gate_result = {
        "passed": False,
        "dialect": "sqlite",
        "blockers": [{
            "code": "safety.non_read_only",
            "message": "Only read-only SQL is allowed",
            "details": {},
        }],
        "warnings": [],
        "sql_signature": {},
    }
    gate_failure = judge.evaluate(
        question="Delete all orders",
        sql="DELETE FROM orders",
        exec_result={"ok": False, "error": "blocked"},
        gate_result=gate_result,
    )
    database_failure = judge.evaluate(
        question="List orders",
        sql="SELECT * FROM orders",
        exec_result={"ok": False, "error": "database is locked"},
        gate_result={
            "passed": True,
            "dialect": "sqlite",
            "blockers": [],
            "warnings": [],
            "sql_signature": {},
        },
    )

    assert calls == []
    assert gate_failure["judge_mode"] == "gate"
    assert database_failure["judge_mode"] == "database"
    assert not any(
        issue["code"] == "semantic.intent_contract_missing"
        for issue in gate_failure["structured_feedback"]["issues"]
    )


def test_semantic_mismatch_blocks_high_similarity_pass(monkeypatch):
    judge_payload = {
        "checks": [],
        "dimensions": {
            "syntax": {"score": 7, "max": 7, "issues": [], "suggestions": []},
            "semantics": {"score": 18, "max": 18, "issues": [], "suggestions": []},
            "logic": {"score": 12, "max": 12, "issues": [], "suggestions": []},
            "result_quality": {"score": 13, "max": 13, "issues": [], "suggestions": []},
        },
        "overall_confidence": 50,
        "critical_flaws": [],
        "repair_priority": [],
        "intent_comparison": {
            "question_intent": {
                "metrics": ["sum amount"],
                "dimensions": ["region"],
                "filters": [],
                "time_range": {},
                "source_tables": ["orders", "users"],
            },
            "sql_intent": {
                "metrics": ["sum amount"],
                "dimensions": ["user_id"],
                "filters": [],
                "time_range": {},
                "source_tables": ["orders"],
            },
            "match": False,
            "mismatches": [
                {
                    "component": "dimensions",
                    "expected": "region",
                    "actual": "user_id",
                    "severity": "major",
                    "feedback": "Group by user region",
                    "schema_search_terms": ["user region"],
                    "suggested_tables": ["users"],
                }
            ],
        },
    }
    responses = iter([json.dumps(judge_payload), "total order amount by region"])
    judge = NL2SQLJudge()
    monkeypatch.setattr(judge, "_call_llm", lambda messages: next(responses))
    monkeypatch.setattr(judge, "_compute_similarity", lambda q1, q2: 0.99)

    result = judge.evaluate(
        question="What is the total order amount by region?",
        sql="SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id",
        exec_result={
            "ok": True,
            "columns": ["user_id", "total"],
            "sample_rows": [{"user_id": 1, "total": 10.0}],
        },
        relevant_schema=RELEVANT_SCHEMA,
        full_schema=FULL_SCHEMA,
    )

    assert result["semantic_similarity"] == 0.99
    assert result["pass"] is False
    assert result["structured_feedback"]["schema_search"]["required"] is True
    assert result["structured_feedback"]["schema_search"]["suggested_tables"] == ["users"]


def test_restricted_reschema_adds_at_most_two_tables():
    linker = SchemaLinker(FULL_SCHEMA)
    feedback = {
        "schema_search": {
            "required": True,
            "query_terms": ["region", "country", "registered user"],
            "suggested_tables": ["users", "regions", "audit_log"],
            "max_new_tables": 99,
        }
    }

    refined = linker.relink_from_feedback(
        "total order amount by region",
        RELEVANT_SCHEMA,
        feedback,
        max_new_tables=2,
    )

    assert refined["reschema"]["triggered"] is True
    assert refined["reschema"]["added_tables"] == ["users", "regions"]
    assert refined["reschema"]["max_new_tables"] == 2
    assert len(refined["candidate_tables"]) == 3


def test_missing_intent_contract_is_a_critical_feedback_issue():
    judge = NL2SQLJudge()
    comparison = judge._normalize_intent_comparison(None)
    feedback = judge._build_structured_feedback(
        {"blocking_issues": [], "missing_tables": [], "missing_columns": []},
        comparison,
        {},
    )

    assert comparison["available"] is False
    assert feedback["issues"][0]["code"] == "semantic.intent_contract_missing"
    assert feedback["issues"][0]["severity"] == "critical"


def test_wrong_draft_intent_can_be_overruled_by_original_question():
    judge = NL2SQLJudge()
    comparison = judge._normalize_intent_comparison({
        "question_intent": {"dimensions": ["region"]},
        "sql_intent": {"dimensions": ["user_id"]},
        "draft_intent": {"dimensions": ["user_id"]},
        "draft_intent_match": True,
        "diagnosis": "contract_issue",
        "match": False,
        "mismatches": [],
    })
    feedback = judge._build_structured_feedback(
        {"blocking_issues": [], "missing_tables": [], "missing_columns": []},
        comparison,
        {},
    )

    assert comparison["draft_intent_match"] is True
    assert comparison["match"] is False
    assert "semantic.contract_issue" in {
        issue["code"] for issue in feedback["issues"]
    }


def test_refiner_owns_restricted_schema_retrieval(monkeypatch):
    linker = SchemaLinker(FULL_SCHEMA)
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return linker.relink_from_feedback(**kwargs)

    refiner = Refiner(model_client=object(), schema_retriever=retrieve)
    captured = {}

    def fake_llm(messages):
        captured["prompt"] = messages[-1]["content"]
        return (
            "SELECT u.region, SUM(o.amount) AS total "
            "FROM orders o JOIN users u ON o.user_id = u.id "
            "GROUP BY u.region"
        )

    monkeypatch.setattr(refiner, "_call_llm", fake_llm)
    judge_result = {
        "structured_feedback": {
            "issues": [{
                "code": "semantic.dimensions.1",
                "message": "Group by user region",
                "suggestion": "Add the users table and group by users.region",
            }],
            "schema_search": {
                "required": True,
                "query_terms": ["user region"],
                "suggested_tables": ["users"],
                "max_new_tables": 2,
            },
        },
        "intent_comparison": {
            "available": True,
            "match": False,
            "question_intent": {"dimensions": ["region"]},
            "sql_intent": {"dimensions": ["user_id"]},
            "mismatches": [],
        },
        "dimensions": {},
        "critical_flaws": [],
        "repair_priority": [],
    }

    result = refiner.repair_from_judge_feedback(
        sql="SELECT user_id, SUM(amount) FROM orders GROUP BY user_id",
        judge_result=judge_result,
        relevant_schema=RELEVANT_SCHEMA,
        question="What is the total order amount by region?",
        allow_schema_retrieval=True,
        max_new_tables=2,
    )

    assert len(calls) == 1
    assert result["action"] == "fixed"
    assert result["repair_route"] == "l2_reschema"
    assert result["schema_retrieval"]["added_tables"] == ["users"]
    assert "CREATE TABLE users" in captured["prompt"]
    assert "CREATE TABLE audit_log" not in captured["prompt"]


def test_gate_repair_does_not_trigger_schema_retrieval(monkeypatch):
    retrieval_calls = []
    refiner = Refiner(
        model_client=object(),
        schema_retriever=lambda **kwargs: retrieval_calls.append(kwargs),
    )
    captured = {}

    def fake_llm(messages):
        captured["prompt"] = messages[-1]["content"]
        return "SELECT user_id FROM orders"

    monkeypatch.setattr(refiner, "_call_llm", fake_llm)
    result = refiner.repair(
        artifact="SELECT {{ user_id }} FROM orders",
        failure_stage="gate",
        feedback={
            "structured_feedback": {
                "issues": [{
                    "code": "structure.unresolved_placeholder",
                    "message": "placeholder",
                }],
                "schema_search": {"required": True, "suggested_tables": ["users"]},
            }
        },
        relevant_schema=RELEVANT_SCHEMA,
        question="List order users",
    )

    assert result["action"] == "fixed"
    assert retrieval_calls == []
    assert "NOT executed" in captured["prompt"]
    assert result["schema_retrieval"]["triggered"] is False
