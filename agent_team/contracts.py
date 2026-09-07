"""Shared contracts for generated SQL, Gate checks, and execution attempts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DraftIntent:
    metrics: list[Any] = field(default_factory=list)
    dimensions: list[Any] = field(default_factory=list)
    filters: list[Any] = field(default_factory=list)
    time_range: dict[str, Any] = field(default_factory=dict)
    output_grain: str = ""
    required_concepts: list[Any] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: Any) -> Optional["DraftIntent"]:
        if not isinstance(value, dict):
            return None
        return cls(
            metrics=list(value.get("metrics", []) or []),
            dimensions=list(value.get("dimensions", []) or []),
            filters=list(value.get("filters", []) or []),
            time_range=value.get("time_range") if isinstance(value.get("time_range"), dict) else {},
            output_grain=str(value.get("output_grain", "") or ""),
            required_concepts=list(value.get("required_concepts", []) or []),
        )


@dataclass
class SQLArtifact:
    sql: str
    draft_intent: Optional[DraftIntent] = None
    intent_version: int = 1
    intent_revision: Optional[dict[str, Any]] = None
    generation_warnings: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @classmethod
    def from_sql(cls, sql: str) -> "SQLArtifact":
        return cls(sql=sql or "")

    @classmethod
    def from_value(cls, value: Any) -> Optional["SQLArtifact"]:
        if not isinstance(value, dict) or not isinstance(value.get("sql"), str):
            return None
        warnings = value.get("generation_warnings", [])
        return cls(
            sql=value["sql"].strip(),
            draft_intent=DraftIntent.from_value(value.get("draft_intent")),
            intent_version=max(1, int(value.get("intent_version", 1) or 1)),
            intent_revision=(
                value.get("intent_revision")
                if isinstance(value.get("intent_revision"), dict)
                else None
            ),
            generation_warnings=list(warnings) if isinstance(warnings, list) else [],
            raw_response=str(value.get("raw_response", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateIssue:
    code: str
    message: str
    details: Any = None


@dataclass
class GateResult:
    passed: bool
    dialect: str
    blockers: list[GateIssue] = field(default_factory=list)
    warnings: list[GateIssue] = field(default_factory=list)
    sql_signature: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionAttempt:
    stage: str
    attempted: bool
    ok: bool
    mode: str
    gate: GateResult
    columns: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    row_count: Optional[int] = None
    row_count_exact: bool = False
    execution_ms: Optional[float] = None
    result_profile: Optional[dict[str, Any]] = None
    error: str = ""
    error_code: str = ""
    read_only_enforced: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gate"] = self.gate.to_dict()
        return data


@dataclass
class RetryState:
    total_sql_attempts: int = 0
    gate_repairs: int = 0
    execution_repairs: int = 0
    semantic_repairs: int = 0
    reschema_attempts: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
