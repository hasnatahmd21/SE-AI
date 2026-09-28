from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryStore
from sebrain.c29 import (
    CandidateChange,
    ChangeKind,
    ChangeRisk,
    GovernanceEngine,
    RollbackReason,
)


def _change(scope="c29.test"):
    return CandidateChange(
        scope=scope,
        kind=ChangeKind.HEURISTIC,
        risk=ChangeRisk.LOW,
        description="test",
        old_value="1",
        new_value="2",
        rationale="test",
        artifact_ref="sha256:test",
        isolated=True,
    )


def _evidence():
    class T:
        passed = 2
        failed = 0
        errors = 0

    class S:
        findings = []

    class V:
        results = [type("R", (), {"result": type("E", (), {"value": "supported"})()})()]

    return {
        "test_run": T(),
        "regressions": [],
        "baseline_passed": 2,
        "current_passed": 2,
        "metric_before": 1.0,
        "metric_after": 1.0,
        "security_report": S(),
        "verification_bundle": V(),
    }


def test_rollback_restores_previous_active_version():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = GovernanceEngine(memory=memory)

        first = engine.submit(_change(), evidence=_evidence())
        second = engine.submit(_change(), evidence=_evidence())

        assert first.final_status.value == "promoted"
        assert second.final_status.value == "promoted"
        assert engine._active_version("c29.test") == 2

        rb = engine.rollback(
            scope="c29.test",
            reason=RollbackReason.MANUAL,
        )
        assert rb.from_version == 2
        assert rb.to_version == 1
        assert engine._active_version("c29.test") == 1


def test_rollback_rejects_nonexistent_target():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = GovernanceEngine(memory=memory)

        engine.submit(_change(), evidence=_evidence())
        try:
            engine.rollback(
                scope="c29.test",
                reason=RollbackReason.MANUAL,
                to_version=99,
            )
        except Exception as exc:
            assert "rollback target v99" in str(exc)
        else:
            raise AssertionError("nonexistent rollback target was accepted")
