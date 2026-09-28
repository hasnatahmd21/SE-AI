from types import SimpleNamespace

from sebrain.c05 import RequirementParser
from sebrain.c22 import (
    Claim,
    ClaimResult,
    EvidenceLevel,
    VerificationEngine,
    VerificationKind,
)


def _run(nodeid: str, outcome: str):
    return SimpleNamespace(
        id="run-1",
        status=SimpleNamespace(value="succeeded"),
        passed=1 if outcome == "passed" else 0,
        failed=1 if outcome == "failed" else 0,
        errors=0,
        skipped=0,
        results=[SimpleNamespace(
            nodeid=nodeid,
            outcome=SimpleNamespace(value=outcome),
        )],
    )


def test_requirement_verification_requires_the_covered_test_result():
    spec = RequirementParser().parse(
        "Users must be able to create tasks."
    )
    req_text = spec.functional[0].text
    import hashlib
    rid = hashlib.sha256(
        f"req::{req_text}".encode("utf-8")
    ).hexdigest()[:16]

    coverage = {f"req:{rid}": ["covered-test-id"]}
    plan = SimpleNamespace(tests=[
        SimpleNamespace(id="covered-test-id", name="test_create_task"),
    ])

    claim = Claim(
        "requirement satisfied",
        VerificationKind.REQUIREMENT,
        metadata={"requirement_text": req_text},
    )
    engine = VerificationEngine()

    unrelated = engine.verify(
        claim,
        ctx={
            "spec": spec,
            "coverage_map": coverage,
            "test_plan": plan,
            "test_run": _run(
                "tests/test_other.py::test_unrelated",
                "passed",
            ),
        },
    )
    assert unrelated.result is ClaimResult.INSUFFICIENT_EVIDENCE

    covered = engine.verify(
        claim,
        ctx={
            "spec": spec,
            "coverage_map": coverage,
            "test_plan": plan,
            "test_run": _run(
                "tests/test_tasks.py::test_create_task",
                "passed",
            ),
        },
    )
    assert covered.result is ClaimResult.SUPPORTED
    assert covered.reached_level is EvidenceLevel.REQUIREMENT_SATISFIED


def test_acceptance_verification_rejects_unrelated_failure():
    spec = RequirementParser().parse(
        "Acceptance:\n"
        "- Given a valid request, when POST /tasks is called, "
        "then 201 is returned\n"
    )
    acc_text = spec.acceptance_criteria[0].text
    import hashlib
    aid = hashlib.sha256(
        f"accept::{acc_text}".encode("utf-8")
    ).hexdigest()[:16]

    coverage = {f"accept:{aid}": ["covered-acceptance-id"]}
    plan = SimpleNamespace(tests=[
        SimpleNamespace(id="covered-acceptance-id", name="test_post_tasks"),
    ])
    claim = Claim(
        "acceptance criterion met",
        VerificationKind.ACCEPTANCE,
        metadata={"acceptance_text": acc_text},
    )
    engine = VerificationEngine()

    result = engine.verify(
        claim,
        ctx={
            "spec": spec,
            "coverage_map": coverage,
            "test_plan": plan,
            "test_run": _run(
                "tests/test_other.py::test_unrelated",
                "failed",
            ),
        },
    )
    assert result.result is ClaimResult.INSUFFICIENT_EVIDENCE
