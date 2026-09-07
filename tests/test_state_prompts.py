import json
from types import SimpleNamespace

import pytest

from agent_team.executor import SafeExecutor, SQLiteReadOnlyExecutor
from agent_team.judge_scoring_card import JudgeScoringCard
from agent_team.nl2sql_judge import NL2SQLJudge
from agent_team.refiner import Refiner


class RecordingClient:
    def __init__(self):
        self.messages = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"sql": "SELECT 2", "draft_intent": {}})
        ))])


def execution_state(prompt):
    return json.loads(prompt.split("### 执行状态字段\n```json\n", 1)[1].split("\n```", 1)[0])


@pytest.mark.parametrize("stage", ["gate", "database", "semantic"])
def test_refiner_real_model_messages_include_state_contracts(stage):
    client = RecordingClient()
    refiner = Refiner(model_client=client)
    result = refiner.repair(
        artifact="SELECT 1",
        failure_stage=stage,
        relevant_schema={"candidate_tables": []},
        question="Return two",
        feedback={
            "error": "test execution error",
            "structured_feedback": {
                "failure_stage": stage,
                "retryable": True,
                "issues": [{"code": "semantic.filters.1", "message": "Return two"}],
                "schema_search": {"required": True, "suggested_tables": ["missing"]},
            },
            "intent_comparison": {"available": True, "match": False},
        },
        allow_schema_retrieval=False,
    )
    assert result["action"] == "fixed"
    assert len(client.messages) == 1
    system, user = client.messages[0]
    assert system["role"] == "system"
    for token in ("stage=gate", "attempted=false", "row_count_exact=true",
                  "schema.outside_retrieval", "retryable=true", "action=fixed"):
        assert token in system["content"]
    assert "SQLArtifact" in user["content"]
    if stage == "semantic":
        assert "match=null" in system["content"]
        assert "contract_issue" in system["content"]
        assert '"skipped_reason": "reschema_budget_exhausted"' in user["content"]
        assert '"triggered": false' in user["content"]
    elif stage == "gate":
        assert "NOT executed" in user["content"]


@pytest.mark.parametrize("mode,exact,count", [("probe", False, 1), ("full", True, 2)])
def test_judge_receives_actual_executor_state(tmp_path, monkeypatch, mode, exact, count):
    import sqlite3

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path):
        pass
    executor = SafeExecutor(SQLiteReadOnlyExecutor(str(db_path), probe_rows=1))
    sql = "SELECT 1 AS n UNION ALL SELECT 2 AS n"
    attempt = executor.attempt(sql, {}, [], mode=mode)
    captured = []

    class CapturedCall(Exception):
        pass

    def capture(messages):
        captured.extend(messages)
        raise CapturedCall

    judge = NL2SQLJudge(model_client=object())
    monkeypatch.setattr(judge, "_call_llm", capture)
    with pytest.raises(CapturedCall):
        judge.evaluate("Return one and two", sql, attempt.to_dict(), {}, gate_result=attempt.gate)

    assert captured[0]["role"] == "system"
    for token in ("stage=database", "row_count_exact=true", "truncated=true", "match=null"):
        assert token in captured[0]["content"]
    state = execution_state(captured[1]["content"])
    assert state["stage"] == "database"
    assert state["attempted"] is True
    assert state["gate"]["passed"] is True
    assert state["ok"] is True
    assert state["mode"] == mode
    assert state["row_count_exact"] is exact
    assert state["truncated"] is not exact
    assert state["row_count"] == count
    assert state["read_only_enforced"] is True


@pytest.mark.parametrize("result,label", [
    ({"stage": "gate", "attempted": False, "ok": False}, "未执行"),
    ({"stage": "database", "attempted": True, "ok": False}, "执行器报告失败"),
    ({}, "未知（未提供成功或失败标记）"),
])
def test_judge_display_preserves_unexecuted_failed_and_unknown(result, label):
    prompt = JudgeScoringCard().build_prompt("test", "SELECT 1", result)
    assert f"- 状态: {label}" in prompt
    assert execution_state(prompt) == result


def test_legacy_execution_result_does_not_invent_exactness():
    prompt = JudgeScoringCard().build_prompt("test", "SELECT 1", {"ok": True, "row_count": 50})
    assert "报告行数（总行数未确认）: 50" in prompt
    assert "row_count_exact" not in execution_state(prompt)


def test_fallback_judge_system_prompt_keeps_state_contract(monkeypatch):
    monkeypatch.setattr("agent_team.judge_scoring_card.os.path.exists", lambda path: False)
    card = JudgeScoringCard()
    first = card.system_prompt
    assert "stage=gate" in first
    assert "contract_issue" in first
    assert card.system_prompt == first
    assert first.count("## 执行状态契约") == 1
