from agent_team.contracts import (
    ExecutionAttempt,
    GateIssue,
    GateResult,
    SQLArtifact,
)
from agent_team.orchestrator import Orchestrator, QueryResult


RELEVANT_SCHEMA = {
    "candidate_tables": [{
        "name": "orders",
        "relevant_columns": [{"name": "id", "type": "INTEGER"}],
    }]
}
FULL_SCHEMA = [{
    "table_name": "orders",
    "columns": [{"col": "id", "type": "INTEGER"}],
}]


class BuilderStub:
    system_prompt = "test"

    def __init__(self, artifact):
        self.artifact = artifact

    def build_artifact(self, *args, **kwargs):
        return self.artifact


class RagStub:
    def retrieve(self, question, top_k=3):
        return []


class SafeExecutorStub:
    def __init__(self, attempts):
        self.attempts = list(attempts)
        self.sqls = []

    def attempt(self, artifact, relevant_schema, full_schema, mode="probe"):
        self.sqls.append(artifact.sql)
        return self.attempts.pop(0)


class JudgeStub:
    PASS_CONFIDENCE = 80

    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        gate = kwargs["gate_result"]
        if not gate.passed:
            return failure_result("gate", "syntax.parse_error")
        return {
            "judge_mode": "semantic",
            "pass": True,
            "overall_confidence": 90,
            "dimensions": {},
            "critical_flaws": [],
            "repair_priority": [],
            "structured_feedback": {"issues": [], "retryable": False},
        }


class RefinerStub:
    def __init__(self, fixed):
        self.fixed = fixed
        self.calls = []

    def repair(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "action": "fixed",
            "fixed_sql": self.fixed.sql,
            "fixed_artifact": self.fixed.to_dict(),
            "repair_route": "gate_sql_repair",
            "schema_retrieval": {"triggered": False},
        }


def failure_result(stage, code):
    return {
        "judge_mode": stage,
        "pass": False,
        "overall_confidence": 0,
        "dimensions": {},
        "critical_flaws": [code],
        "repair_priority": ["repair"],
        "structured_feedback": {
            "retryable": True,
            "issues": [{"code": code, "evidence": "same"}],
        },
    }


def make_orchestrator(builder, judge=None, refiner=None):
    orchestrator = object.__new__(Orchestrator)
    orchestrator.schema = FULL_SCHEMA
    orchestrator.model = "test-model"
    orchestrator.domain = "generic"
    orchestrator.dialect = "sqlite"
    orchestrator.builder = builder
    orchestrator.refiner = refiner
    orchestrator._judge = judge
    orchestrator._rag = RagStub()
    orchestrator._rag_add_experience = lambda *args, **kwargs: None
    return orchestrator


def test_no_executor_is_generated_but_unverified():
    artifact = SQLArtifact.from_sql("SELECT id FROM orders")
    orchestrator = make_orchestrator(BuilderStub(artifact))

    result = orchestrator._run_with_execution_feedback(
        QueryResult(question="list orders"), RELEVANT_SCHEMA, None, "", None, "list orders"
    )

    assert result.status == "generated_unverified"
    assert result.success is False
    assert result.sql_generated is True
    assert result.gate_passed is None
    assert result.execution_attempted is False
    assert result.judge_passed is None


def test_gate_repair_does_not_consume_semantic_budget_and_rechecks_sql():
    initial = SQLArtifact.from_sql("SELECT {{ id }} FROM orders")
    fixed = SQLArtifact.from_sql("SELECT id FROM orders")
    blocked_gate = GateResult(
        False,
        "sqlite",
        blockers=[GateIssue("structure.unresolved_placeholder", "placeholder")],
    )
    passed_gate = GateResult(True, "sqlite")
    attempts = [
        ExecutionAttempt("gate", False, False, "probe", blocked_gate, error="placeholder"),
        ExecutionAttempt(
            "database", True, True, "probe", passed_gate,
            columns=["id"], sample_rows=[{"id": 1}], row_count=1, row_count_exact=True,
        ),
    ]
    safe = SafeExecutorStub(attempts)
    judge = JudgeStub()
    refiner = RefinerStub(fixed)
    orchestrator = make_orchestrator(BuilderStub(initial), judge, refiner)

    result = orchestrator._run_with_execution_feedback(
        QueryResult(question="list orders"), RELEVANT_SCHEMA, None, "", safe, "list orders"
    )

    assert result.success is True
    assert result.status == "verified"
    assert safe.sqls == [initial.sql, fixed.sql]
    assert refiner.calls[0]["failure_stage"] == "gate"
    assert result.retry_state["gate_repairs"] == 1
    assert result.retry_state["semantic_repairs"] == 0
    assert result.retry_state["total_sql_attempts"] == 2
