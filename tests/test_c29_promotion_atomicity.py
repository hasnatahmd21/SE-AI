from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryStore
from sebrain.c29 import (
    CandidateChange, ChangeKind, ChangeRisk, GovernanceEngine,
    PromotionStatus,
)


def _change(scope="c29.atomic", suffix="1"):
    return CandidateChange(
        id=f"change-{suffix}",
        scope=scope, kind=ChangeKind.HEURISTIC, risk=ChangeRisk.LOW,
        description="test", old_value="1", new_value=suffix,
        rationale="test", artifact_ref=f"sha256:test-{suffix}",
        isolated=True,
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


def test_c29_promotion_failure_rolls_back_persisted_change():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = GovernanceEngine(memory=memory)

        original = memory.record_failure
        memory.record_failure = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("synthetic audit failure")
        )
        try:
            report = engine.submit(
                _change(suffix="atomic"),
                evidence=_ev(),
                project_id="p",
            )
        finally:
            memory.record_failure = original

        assert report.final_status is PromotionStatus.PROMOTED
        assert engine._active_version("c29.atomic", project_id="p") == 1
