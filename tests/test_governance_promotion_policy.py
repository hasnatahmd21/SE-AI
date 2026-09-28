from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryStore
from sebrain.c29 import (
    CandidateChange, ChangeKind, ChangeRisk, GovernanceEngine,
    StageOutcome, PromotionStatus,
)


def _change(scope="c29.cap", risk=ChangeRisk.LOW):
    return CandidateChange(
        scope=scope, kind=ChangeKind.HEURISTIC, risk=risk,
        description="test", old_value="1", new_value="2",
        rationale="test", artifact_ref="sha256:test", isolated=True,
    )


class T:
    passed, failed, errors = 2, 0, 0


class S:
    findings = []


class V:
    results = [type("R", (), {"result": type("E", (), {"value": "supported"})()})()]


def _ev():
    return {
        "test_run": T(), "regressions": [],
        "baseline_passed": 2, "current_passed": 2,
        "metric_before": 1.0, "metric_after": 1.0,
        "security_report": S(), "verification_bundle": V(),
    }


def test_promotion_cap_is_enforced_before_second_promotion():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = GovernanceEngine(memory=memory, max_changes_per_scope=1)
        first = engine.submit(_change(), evidence=_ev(), project_id="p")
        second = engine.submit(_change(), evidence=_ev(), project_id="p")
        assert first.final_status is PromotionStatus.PROMOTED
        assert second.final_status is PromotionStatus.HELD
        assert second.promotion_version is None
        assert engine._active_version("c29.cap", project_id="p") == 1


def test_critical_warning_cannot_be_overridden_by_manual_approval():
    ch = _change(risk=ChangeRisk.CRITICAL)
    stages = [
        type("S", (), {"outcome": StageOutcome.WARN})(),
    ]
    outcome, _, approved = __import__("sebrain.c29", fromlist=["ApprovalPolicy"]).ApprovalPolicy().decide(
        ch, stages, manual_approval="reviewer"
    )
    assert outcome is StageOutcome.BLOCKED
    assert approved == ""
