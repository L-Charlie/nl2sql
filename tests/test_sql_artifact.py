import json

from agent_team.builder import SQLBuilder
from agent_team.contracts import DraftIntent, SQLArtifact


def test_parse_structured_sql_artifact():
    response = json.dumps({
        "draft_intent": {
            "metrics": ["sum amount"],
            "dimensions": ["region"],
            "filters": [],
            "time_range": {},
            "output_grain": "one row per region",
            "required_concepts": ["orders", "users"],
        },
        "intent_version": 1,
        "intent_revision": None,
        "sql": "SELECT region, SUM(amount) FROM orders GROUP BY region",
    })

    artifact = SQLBuilder.parse_artifact_response(response)

    assert artifact.sql.startswith("SELECT region")
    assert artifact.draft_intent.dimensions == ["region"]
    assert artifact.generation_warnings == []


def test_sql_only_response_is_nonblocking_fallback():
    artifact = SQLBuilder.parse_artifact_response("SELECT id FROM users")

    assert artifact.sql == "SELECT id FROM users"
    assert artifact.draft_intent is None
    assert artifact.generation_warnings[0]["code"] == "generation.contract_missing"


def test_revised_intent_increments_version_and_records_reason():
    previous = SQLArtifact(
        sql="SELECT user_id FROM orders",
        draft_intent=DraftIntent(dimensions=["user"]),
        intent_version=1,
    )
    response = json.dumps({
        "draft_intent": {
            "metrics": [],
            "dimensions": ["region"],
            "filters": [],
            "time_range": {},
            "output_grain": "one row per region",
            "required_concepts": ["users"],
        },
        "intent_version": 1,
        "sql": "SELECT region FROM users",
    })

    artifact = SQLBuilder.parse_artifact_response(response, previous)

    assert artifact.intent_version == 2
    assert artifact.intent_revision["reason"]


def test_sql_fallback_preserves_previous_intent():
    previous = SQLArtifact(
        sql="SELECT user_id FROM orders",
        draft_intent=DraftIntent(dimensions=["user"]),
        intent_version=3,
    )

    artifact = SQLBuilder.parse_artifact_response("SELECT region FROM users", previous)

    assert artifact.intent_version == 3
    assert artifact.draft_intent == previous.draft_intent
