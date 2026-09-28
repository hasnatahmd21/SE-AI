"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C22 — VERIFICATION ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04.

Purpose:
    Turn EVIDENCE into VERIFIED CLAIMS — without ever confusing:

        GENERATED  ≠  EXECUTED  ≠  TESTED  ≠  VERIFIED  ≠  REQUIREMENT_SATISFIED

    Every VerificationResult carries:
        claim          — what is being asserted
        level          — the epistemic level reached (never higher than evidence)
        method         — name + description of how it was verified
        result         — SUPPORTED / REFUTED / INCONCLUSIVE / INSUFFICIENT_EVIDENCE
        evidence       — list of EvidenceRef (kind, ref, digest, observed_at)
        confidence     — VERIFIED / HIGH / MEDIUM / LOW / UNKNOWN
        timestamp      — when the verification ran
        artifact_ref   — what artifact was verified
        artifact_revision — hash / id / version of that artifact

Capabilities:
    - Verifier methods for 6 areas from the spec:
        * ExistenceVerifier            (GENERATED)
        * ExecutionVerifier            (EXECUTED)
        * TestSuiteVerifier            (TESTED)
        * RequirementVerifier          (REQUIREMENT_SATISFIED)
        * ArchitectureVerifier         (VERIFIED)
        * SecurityVerifier             (VERIFIED)
        * AcceptanceVerifier           (REQUIREMENT_SATISFIED)
        * EvidenceChainVerifier        (VERIFIED)
    - Claim ladder: refuses to promote a claim beyond what its evidence allows.
    - Bundle: aggregate many claims → ladder summary
      ("3 generated, 3 executed, 3 tested, 2 verified, 2 satisfied")
    - Persistence: C04 memory + C02 ontology (VERIFICATION + EVIDENCE).

Invariants honored:
    - NO external LLM. Deterministic evidence evaluation.
    - NEVER claims VERIFIED without an EvidenceRef.
    - NEVER claims SATISFIED without a passing test or explicit acceptance evidence.
    - EvidenceRefs carry a digest (tamper-evident) + observed_at.
    - Missing evidence → INSUFFICIENT_EVIDENCE (not REFUTED).
    - REFUTED only when evidence actively contradicts (test failed, etc.).
    - INCONCLUSIVE when evidence exists but is ambiguous/stale.
    - Staleness is measured (max_age_seconds policy per verifier).
    - Same inputs + same timestamps → same results (deterministic).

Explicit limitations:
    - Requirement satisfaction requires the CALLER to pass a "coverage map"
      (typically produced by C17) + a test run (C18). Without them, we
      honestly return INSUFFICIENT_EVIDENCE, never fabricate PASS.
    - Architecture / security verification is duck-typed against C10 / C21
      outputs; if the shape differs, we refuse rather than guess.
    - No formal proof, no model checking. This is evidence aggregation
      over structured observations produced by earlier phases.

Contents:
  1.  Enums: EvidenceLevel, EvidenceKind, VerificationKind, ClaimResult
  2.  Dataclasses: EvidenceRef, Claim, VerificationResult, ClaimLadder,
                   VerificationBundle
  3.  Evidence helpers (digest, freshness)
  4.  Verifiers (8)
  5.  VerificationEngine facade
  6.  VerificationRepository (persist)
  7.  Self-tests (~30)
  8.  Demo

Run as script:
    python -m sebrain.c22            # demo
    python -m sebrain.c22 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from sebrain.c01 import (
    Confidence, Config, SEBrainApp, SQLiteStorage, ValidationError,
    execution_scope, get_logger,
)
from sebrain.c02 import (
    EntityKind, Ontology, Provenance, ProvenanceType, RelationKind,
)
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _digest(s: Any) -> str:
    if not isinstance(s, (str, bytes)):
        try:
            s = json.dumps(s, sort_keys=True, default=str)
        except Exception:
            s = str(s)
    if isinstance(s, str):
        s = s.encode("utf-8")
    return "sha256:" + hashlib.sha256(s).hexdigest()[:32]


def _parse_iso(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class EvidenceLevel(str, Enum):
    """The epistemic ladder. A claim NEVER outranks its evidence level."""
    GENERATED = "generated"                     # artifact exists
    EXECUTED = "executed"                       # artifact ran (exit code known)
    TESTED = "tested"                           # tests exercised it
    VERIFIED = "verified"                       # evidence checked by method
    REQUIREMENT_SATISFIED = "requirement_satisfied"  # acceptance criteria met


_LEVEL_RANK: dict[EvidenceLevel, int] = {
    EvidenceLevel.GENERATED: 1,
    EvidenceLevel.EXECUTED: 2,
    EvidenceLevel.TESTED: 3,
    EvidenceLevel.VERIFIED: 4,
    EvidenceLevel.REQUIREMENT_SATISFIED: 5,
}


class EvidenceKind(str, Enum):
    """Category of evidence. Every EvidenceRef carries one."""
    FILE_HASH = "file_hash"
    SANDBOX_RUN = "sandbox_run"
    TEST_RUN = "test_run"
    TEST_NODEID = "test_nodeid"
    ARCHITECTURE_GRAPH = "architecture_graph"
    SECURITY_SCAN = "security_scan"
    REQUIREMENT_COVERAGE = "requirement_coverage"
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    MANUAL_OBSERVATION = "manual_observation"


class VerificationKind(str, Enum):
    """Which broad area from the spec is being verified."""
    EXISTENCE = "existence"
    EXECUTION = "execution"
    TEST_SUITE = "test_suite"
    REQUIREMENT = "requirement"
    ACCEPTANCE = "acceptance"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    EVIDENCE_CHAIN = "evidence_chain"
    INTEGRATION = "integration"


class ClaimResult(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class EvidenceRef:
    """One piece of evidence with tamper-evident digest + timestamp."""
    kind: EvidenceKind
    ref: str                        # id / path / nodeid / url
    digest: str = ""
    observed_at: str = field(default_factory=now_iso)
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ref:
            raise ValidationError("EvidenceRef.ref is required")
        if not self.digest:
            self.digest = _digest({
                "kind": self.kind.value,
                "ref": self.ref,
                "payload": self.payload,
            })

    def age_seconds(self, at: str | None = None) -> float:
        when = _parse_iso(at or now_iso())
        obs = _parse_iso(self.observed_at)
        return max(0.0, (when - obs).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "ref": self.ref,
            "digest": self.digest, "observed_at": self.observed_at,
            "description": self.description,
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class Claim:
    """A specific assertion to be verified."""
    text: str
    kind: VerificationKind
    # Default is the top of the ladder (not a mid-rung like VERIFIED) so
    # that, unless the caller deliberately sets a lower ceiling, a claim
    # reports whatever level its verifier actually achieved rather than
    # being silently capped down to a one-size-fits-all default that
    # doesn't fit every VerificationKind (e.g. REQUIREMENT/ACCEPTANCE
    # verifiers are specifically designed to reach REQUIREMENT_SATISFIED).
    target_level: EvidenceLevel = EvidenceLevel.REQUIREMENT_SATISFIED
    artifact_ref: str = ""
    artifact_revision: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "kind": self.kind.value,
            "target_level": self.target_level.value,
            "artifact_ref": self.artifact_ref,
            "artifact_revision": self.artifact_revision,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class VerificationResult:
    """Complete structured verification record.

    Enforced fields: claim, level, method, result, confidence,
                     timestamp, artifact_ref, artifact_revision.
    """
    id: str = field(default_factory=_new_id)
    claim: Claim = field(default_factory=lambda: Claim("", VerificationKind.EXISTENCE))
    reached_level: EvidenceLevel = EvidenceLevel.GENERATED
    method_name: str = ""
    method_description: str = ""
    result: ClaimResult = ClaimResult.INSUFFICIENT_EVIDENCE
    confidence: Confidence = Confidence.UNKNOWN
    evidence: list[EvidenceRef] = field(default_factory=list)
    rationale: str = ""
    timestamp: str = field(default_factory=now_iso)
    artifact_ref: str = ""
    artifact_revision: str = ""
    stale_evidence: bool = False
    provenance: Provenance = field(default_factory=Provenance)

    # ---- invariants ----
    def __post_init__(self) -> None:
        # Never claim a higher level than evidence allows.
        if _LEVEL_RANK[self.reached_level] >= _LEVEL_RANK[EvidenceLevel.VERIFIED]:
            if not self.evidence:
                raise ValidationError(
                    "VERIFIED/SATISFIED without evidence is forbidden"
                )
        # SUPPORTED must have at least one evidence ref.
        if (self.result is ClaimResult.SUPPORTED
                and not self.evidence):
            raise ValidationError(
                "SUPPORTED result requires at least one EvidenceRef"
            )
        # artifact fields always come from claim (single source of truth)
        if not self.artifact_ref:
            self.artifact_ref = self.claim.artifact_ref
        if not self.artifact_revision:
            self.artifact_revision = self.claim.artifact_revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim.to_dict(),
            "reached_level": self.reached_level.value,
            "method_name": self.method_name,
            "method_description": self.method_description,
            "result": self.result.value,
            "confidence": self.confidence.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "artifact_ref": self.artifact_ref,
            "artifact_revision": self.artifact_revision,
            "stale_evidence": self.stale_evidence,
            "provenance": self.provenance.to_dict(),
        }

    def summary(self) -> str:
        return (
            f"[{self.result.value:22s}] lvl={self.reached_level.value:22s} "
            f"conf={self.confidence.value:10s} "
            f"method={self.method_name}  "
            f"evidence={len(self.evidence)}  "
            f"claim={_short(self.claim.text, 70)}"
        )


@dataclass(slots=True)
class ClaimLadder:
    """Aggregate of the epistemic ladder across a set of results."""
    generated: int = 0
    executed: int = 0
    tested: int = 0
    verified: int = 0
    satisfied: int = 0

    def add(self, level: EvidenceLevel) -> None:
        if level is EvidenceLevel.GENERATED:
            self.generated += 1
        elif level is EvidenceLevel.EXECUTED:
            self.executed += 1
        elif level is EvidenceLevel.TESTED:
            self.tested += 1
        elif level is EvidenceLevel.VERIFIED:
            self.verified += 1
        elif level is EvidenceLevel.REQUIREMENT_SATISFIED:
            self.satisfied += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "executed": self.executed,
            "tested": self.tested,
            "verified": self.verified,
            "satisfied": self.satisfied,
        }

    def __str__(self) -> str:
        return (
            f"generated={self.generated}  executed={self.executed}  "
            f"tested={self.tested}  verified={self.verified}  "
            f"satisfied={self.satisfied}"
        )


@dataclass(slots=True)
class VerificationBundle:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    results: list[VerificationResult] = field(default_factory=list)
    ladder: ClaimLadder = field(default_factory=ClaimLadder)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def add(self, r: VerificationResult) -> None:
        self.results.append(r)
        self.ladder.add(r.reached_level)

    def refuted(self) -> list[VerificationResult]:
        return [r for r in self.results if r.result is ClaimResult.REFUTED]

    def insufficient(self) -> list[VerificationResult]:
        return [r for r in self.results
                if r.result is ClaimResult.INSUFFICIENT_EVIDENCE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "results": [r.to_dict() for r in self.results],
            "ladder": self.ladder.to_dict(),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Verification Bundle ===\n"
            f"results={len(self.results)}  ladder: {self.ladder}\n"
            f"refuted={len(self.refuted())}  "
            f"insufficient={len(self.insufficient())}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. EVIDENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════
def evidence_from_file(path: str | Any) -> EvidenceRef | None:
    """Read a file, hash it, return an EvidenceRef. Best-effort."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return EvidenceRef(
            kind=EvidenceKind.FILE_HASH,
            ref=str(p),
            digest="sha256:" + h.hexdigest()[:32],
            payload={"size_bytes": p.stat().st_size},
            description=f"file hash of {p.name}",
        )
    except Exception:
        return None


def evidence_is_fresh(
    ref: EvidenceRef, *, max_age_seconds: float,
    at: str | None = None,
) -> bool:
    return ref.age_seconds(at) <= max_age_seconds


# ════════════════════════════════════════════════════════════════════════════
# 4. VERIFIERS
# ════════════════════════════════════════════════════════════════════════════
class _Verifier:
    """Base class. Subclasses override `verify(claim, ctx)`."""
    name: str = "verifier"
    description: str = ""
    kind: VerificationKind = VerificationKind.EXISTENCE
    min_level: EvidenceLevel = EvidenceLevel.GENERATED
    max_age_seconds: float = 3600.0    # evidence freshness policy

    def can_handle(self, claim: Claim) -> bool:
        return claim.kind is self.kind

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        raise NotImplementedError

    # ---- helpers ----
    def _insufficient(
        self, claim: Claim, *, reason: str,
        evidence: Iterable[EvidenceRef] = (),
        level: EvidenceLevel = EvidenceLevel.GENERATED,
    ) -> VerificationResult:
        return VerificationResult(
            claim=claim, reached_level=level,
            method_name=self.name, method_description=self.description,
            result=ClaimResult.INSUFFICIENT_EVIDENCE,
            confidence=Confidence.UNKNOWN,
            evidence=list(evidence),
            rationale=reason,
            provenance=Provenance(
                source="verification_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.UNKNOWN,
            ),
        )

    def _refuted(
        self, claim: Claim, *, reason: str,
        evidence: Iterable[EvidenceRef],
        level: EvidenceLevel,
        confidence: Confidence = Confidence.HIGH,
    ) -> VerificationResult:
        return VerificationResult(
            claim=claim, reached_level=level,
            method_name=self.name, method_description=self.description,
            result=ClaimResult.REFUTED, confidence=confidence,
            evidence=list(evidence), rationale=reason,
            provenance=Provenance(
                source="verification_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=confidence,
            ),
        )

    def _supported(
        self, claim: Claim, *, reason: str,
        evidence: Iterable[EvidenceRef],
        level: EvidenceLevel,
        confidence: Confidence = Confidence.VERIFIED,
        stale: bool = False,
    ) -> VerificationResult:
        return VerificationResult(
            claim=claim, reached_level=level,
            method_name=self.name, method_description=self.description,
            result=ClaimResult.SUPPORTED, confidence=confidence,
            evidence=list(evidence), rationale=reason,
            stale_evidence=stale,
            provenance=Provenance(
                source="verification_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=confidence,
            ),
        )

    def _check_staleness(
        self, refs: Sequence[EvidenceRef],
    ) -> tuple[bool, list[str]]:
        stale: list[str] = []
        for r in refs:
            if not evidence_is_fresh(r, max_age_seconds=self.max_age_seconds):
                stale.append(r.ref)
        return (len(stale) > 0, stale)


# ---- 1. Existence ----
class ExistenceVerifier(_Verifier):
    name = "existence"
    kind = VerificationKind.EXISTENCE
    description = "artifact is present on disk (level: GENERATED)"
    min_level = EvidenceLevel.GENERATED

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        path = claim.metadata.get("path") or claim.artifact_ref
        if not path:
            return self._insufficient(
                claim, reason="no path in claim.metadata['path'] or artifact_ref",
            )
        ref = evidence_from_file(path)
        if ref is None:
            return self._refuted(
                claim, reason=f"artifact not found at {path}",
                evidence=[], level=EvidenceLevel.GENERATED,
                confidence=Confidence.HIGH,
            )
        return self._supported(
            claim, reason=f"artifact exists at {path}",
            evidence=[ref], level=EvidenceLevel.GENERATED,
            confidence=Confidence.VERIFIED,
        )


# ---- 2. Execution ----
class ExecutionVerifier(_Verifier):
    name = "execution"
    kind = VerificationKind.EXECUTION
    description = "artifact ran successfully (level: EXECUTED)"
    min_level = EvidenceLevel.EXECUTED
    max_age_seconds = 1800.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        """Expects ctx['sandbox_result'] (duck-typed C16 SandboxResult)."""
        sr = ctx.get("sandbox_result")
        if sr is None:
            return self._insufficient(
                claim, reason="no sandbox_result in context",
                level=EvidenceLevel.GENERATED,
            )
        exit_code = int(getattr(sr, "exit_code", -999))
        stdout = str(getattr(sr, "stdout", "") or "")
        stderr = str(getattr(sr, "stderr", "") or "")
        status = getattr(getattr(sr, "status", None), "value", "")
        rid = str(getattr(sr, "id", ""))
        ref = EvidenceRef(
            kind=EvidenceKind.SANDBOX_RUN,
            ref=rid or f"sandbox:{claim.artifact_ref}",
            payload={
                "exit_code": exit_code,
                "status": status,
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
            },
            description=f"sandbox run (exit={exit_code})",
        )
        if exit_code == 0 and status in ("succeeded", ""):
            return self._supported(
                claim, reason=f"process exited with code 0 (status={status})",
                evidence=[ref], level=EvidenceLevel.EXECUTED,
                confidence=Confidence.HIGH,
            )
        return self._refuted(
            claim,
            reason=f"process exited with code {exit_code} (status={status})",
            evidence=[ref], level=EvidenceLevel.EXECUTED,
            confidence=Confidence.HIGH,
        )


# ---- 3. Test suite ----
class TestSuiteVerifier(_Verifier):
    name = "test_suite"
    kind = VerificationKind.TEST_SUITE
    description = "test suite passed (level: TESTED)"
    min_level = EvidenceLevel.TESTED
    max_age_seconds = 3600.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        """Expects ctx['test_run'] (duck-typed C18 TestRunResult)."""
        run = ctx.get("test_run")
        if run is None:
            return self._insufficient(
                claim, reason="no test_run in context",
                level=EvidenceLevel.EXECUTED,
            )
        rid = str(getattr(run, "id", ""))
        status = getattr(getattr(run, "status", None), "value", "unknown")
        passed = int(getattr(run, "passed", 0) or 0)
        failed = int(getattr(run, "failed", 0) or 0)
        errors = int(getattr(run, "errors", 0) or 0)
        skipped = int(getattr(run, "skipped", 0) or 0)
        ref = EvidenceRef(
            kind=EvidenceKind.TEST_RUN,
            ref=rid or f"test_run:{claim.artifact_ref}",
            payload={
                "status": status, "passed": passed, "failed": failed,
                "errors": errors, "skipped": skipped,
            },
            description=f"test run ({status})",
        )
        if status in ("succeeded", "no_tests") and failed == 0 and errors == 0:
            return self._supported(
                claim,
                reason=(
                    f"test suite {status}: passed={passed} "
                    f"failed={failed} errors={errors} skipped={skipped}"
                ),
                evidence=[ref], level=EvidenceLevel.TESTED,
                confidence=Confidence.HIGH,
            )
        return self._refuted(
            claim,
            reason=(
                f"test suite {status}: failed={failed} errors={errors}"
            ),
            evidence=[ref], level=EvidenceLevel.TESTED,
            confidence=Confidence.HIGH,
        )


# ---- 4. Requirement satisfaction ----
class RequirementVerifier(_Verifier):
    name = "requirement"
    kind = VerificationKind.REQUIREMENT
    description = "functional requirement is satisfied by a passing test (level: REQUIREMENT_SATISFIED)"
    min_level = EvidenceLevel.REQUIREMENT_SATISFIED
    max_age_seconds = 3600.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        """
        Context needs:
            spec          — C05 RequirementSpec
            coverage_map  — dict target → [test_ids] (from C17 TestPlan.coverage)
            test_run      — C18 TestRunResult
        """
        spec = ctx.get("spec")
        coverage_map = ctx.get("coverage_map") or {}
        test_run = ctx.get("test_run")
        req_text = claim.metadata.get("requirement_text") or claim.text

        if spec is None or test_run is None:
            return self._insufficient(
                claim,
                reason="spec and test_run required for requirement verification",
                level=EvidenceLevel.TESTED,
            )

        # Find the functional requirement id
        from hashlib import sha256 as _s
        matched_id: str | None = None
        for item in getattr(spec, "functional", []) or []:
            text = getattr(item, "text", "")
            # Same "stable id" formula as C17 uses.
            h = _s(f"req::{text}".encode("utf-8")).hexdigest()[:16]
            if text == req_text:
                matched_id = h
                break
        if matched_id is None:
            return self._insufficient(
                claim,
                reason=(
                    "requirement text not found in spec.functional; "
                    "cannot map to a test"
                ),
                level=EvidenceLevel.TESTED,
            )

        # Convert coverage_map (target → test_ids) into test nodeids.
        target = f"req:{matched_id}"
        test_ids = list(coverage_map.get(target, []))
        if not test_ids:
            return self._refuted(
                claim,
                reason=f"no test covers requirement id {matched_id}",
                evidence=[
                    EvidenceRef(
                        kind=EvidenceKind.REQUIREMENT_COVERAGE,
                        ref=f"req:{matched_id}",
                        payload={"covered_by": []},
                        description="coverage map has no entry for requirement",
                    )
                ],
                level=EvidenceLevel.TESTED,
                confidence=Confidence.HIGH,
            )

        # C17 test_ids are stable identifiers; C18 nodeids use pytest names.
        # A coverage claim is only valid when the covered test id can be mapped
        # to an actual C18 result. Never treat an unrelated passing test as
        # evidence for this requirement.
        test_plan = ctx.get("test_plan")
        test_name_by_id = {
            str(getattr(t, "id", "")): str(getattr(t, "name", ""))
            for t in (getattr(test_plan, "tests", []) or [])
            if getattr(t, "id", None) and getattr(t, "name", None)
        }
        if not test_plan:
            return self._insufficient(
                claim,
                reason=(
                    "test_plan required to map C17 coverage test ids "
                    "to C18 pytest nodeids"
                ),
                level=EvidenceLevel.TESTED,
            )
        test_run_results = list(getattr(test_run, "results", []) or [])
        matching_evidence: list[EvidenceRef] = []
        matched_results: list[Any] = []
        for r in test_run_results:
            nid = str(getattr(r, "nodeid", ""))
            if any(
                tid in test_name_by_id
                and (
                    nid.endswith("::" + test_name_by_id[tid])
                    or nid.startswith(test_name_by_id[tid] + "[")
                    or ("::" + test_name_by_id[tid] + "[") in nid
                )
                for tid in test_ids
            ):
                matched_results.append(r)
                oc = getattr(getattr(r, "outcome", None), "value", "")
                matching_evidence.append(EvidenceRef(
                    kind=EvidenceKind.TEST_NODEID,
                    ref=nid, payload={"outcome": oc},
                    description="covered test node",
                ))
        if not matched_results:
            return self._insufficient(
                claim,
                reason="coverage test ids did not map to any C18 test result",
                evidence=[
                    EvidenceRef(
                        kind=EvidenceKind.REQUIREMENT_COVERAGE,
                        ref=target,
                        payload={"covered_by": test_ids},
                    )
                ],
                level=EvidenceLevel.TESTED,
            )
        any_failed = any(
            getattr(getattr(r, "outcome", None), "value", "")
            in ("failed", "error")
            for r in matched_results
        )
        any_passed = any(
            getattr(getattr(r, "outcome", None), "value", "") == "passed"
            for r in matched_results
        )

        if any_failed:
            return self._refuted(
                claim,
                reason=(
                    f"requirement has failing/erroring tests "
                    f"(target={target})"
                ),
                evidence=matching_evidence or [
                    EvidenceRef(
                        kind=EvidenceKind.REQUIREMENT_COVERAGE,
                        ref=target, payload={"covered_by": test_ids},
                    )
                ],
                level=EvidenceLevel.TESTED,
                confidence=Confidence.HIGH,
            )
        if any_passed:
            # Level reached is SATISFIED because we have:
            # (a) a coverage mapping, and (b) passing tests
            return self._supported(
                claim,
                reason=(
                    f"requirement covered by {len(test_ids)} test id(s); "
                    f"suite has {sum(1 for r in test_run_results if getattr(getattr(r,'outcome',None),'value','')=='passed')} passing"
                ),
                evidence=matching_evidence[:3] + [
                    EvidenceRef(
                        kind=EvidenceKind.REQUIREMENT_COVERAGE,
                        ref=target, payload={"covered_by": test_ids},
                        description="coverage map entry",
                    )
                ],
                level=EvidenceLevel.REQUIREMENT_SATISFIED,
                confidence=Confidence.MEDIUM,
            )
        return self._insufficient(
            claim,
            reason="coverage exists but no passing test in the run",
            level=EvidenceLevel.TESTED,
        )


# ---- 5. Acceptance criteria ----
class AcceptanceVerifier(_Verifier):
    name = "acceptance"
    kind = VerificationKind.ACCEPTANCE
    description = "acceptance criterion is asserted by tests (level: REQUIREMENT_SATISFIED)"
    min_level = EvidenceLevel.REQUIREMENT_SATISFIED
    max_age_seconds = 3600.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        spec = ctx.get("spec")
        coverage_map = ctx.get("coverage_map") or {}
        test_run = ctx.get("test_run")
        if spec is None or test_run is None:
            return self._insufficient(
                claim, reason="spec and test_run required",
                level=EvidenceLevel.TESTED,
            )
        # Find acceptance criterion id
        from hashlib import sha256 as _s
        target_text = claim.metadata.get("acceptance_text") or claim.text
        matched = None
        for it in getattr(spec, "acceptance_criteria", []) or []:
            t = getattr(it, "text", "")
            if t == target_text:
                h = _s(f"accept::{t}".encode("utf-8")).hexdigest()[:16]
                matched = h
                break
        if matched is None:
            return self._insufficient(
                claim, reason="acceptance text not found in spec",
                level=EvidenceLevel.TESTED,
            )
        target = f"accept:{matched}"
        test_ids = list(coverage_map.get(target, []))
        if not test_ids:
            return self._refuted(
                claim, reason="no test covers this acceptance criterion",
                evidence=[EvidenceRef(
                    kind=EvidenceKind.ACCEPTANCE_CRITERION,
                    ref=target, payload={"covered_by": []},
                )],
                level=EvidenceLevel.TESTED,
                confidence=Confidence.HIGH,
            )
        # Map C17 coverage ids to actual C18 pytest results. An unrelated
        # passing test must never satisfy an acceptance criterion.
        test_plan = ctx.get("test_plan")
        test_name_by_id = {
            str(getattr(t, "id", "")): str(getattr(t, "name", ""))
            for t in (getattr(test_plan, "tests", []) or [])
            if getattr(t, "id", None) and getattr(t, "name", None)
        }
        if not test_plan:
            return self._insufficient(
                claim,
                reason=(
                    "test_plan required to map C17 coverage test ids "
                    "to C18 pytest nodeids"
                ),
                level=EvidenceLevel.TESTED,
            )
        results = list(getattr(test_run, "results", []) or [])
        matched = [
            r for r in results
            if any(
                tid in test_name_by_id
                and (
                    str(getattr(r, "nodeid", "")).endswith(
                        "::" + test_name_by_id[tid]
                    )
                    or ("::" + test_name_by_id[tid] + "[") in str(
                        getattr(r, "nodeid", "")
                    )
                )
                for tid in test_ids
            )
        ]
        if not matched:
            return self._insufficient(
                claim,
                reason="coverage test ids did not map to any C18 test result",
                evidence=[EvidenceRef(
                    kind=EvidenceKind.ACCEPTANCE_CRITERION,
                    ref=target, payload={"covered_by": test_ids},
                )],
                level=EvidenceLevel.TESTED,
            )
        any_failed = any(
            getattr(getattr(r, "outcome", None), "value", "") in ("failed", "error")
            for r in matched
        )
        any_passed = any(
            getattr(getattr(r, "outcome", None), "value", "") == "passed"
            for r in matched
        )
        if any_failed:
            return self._refuted(
                claim, reason="some tests failed during acceptance run",
                evidence=[EvidenceRef(
                    kind=EvidenceKind.ACCEPTANCE_CRITERION,
                    ref=target, payload={"covered_by": test_ids},
                )],
                level=EvidenceLevel.TESTED,
                confidence=Confidence.HIGH,
            )
        return self._supported(
            claim,
            reason=f"acceptance covered by {len(test_ids)} test(s); no failures",
            evidence=[EvidenceRef(
                kind=EvidenceKind.ACCEPTANCE_CRITERION,
                ref=target, payload={"covered_by": test_ids},
            )],
            level=EvidenceLevel.REQUIREMENT_SATISFIED,
            confidence=Confidence.MEDIUM,
        )


# ---- 6. Architecture ----
class ArchitectureVerifier(_Verifier):
    name = "architecture"
    kind = VerificationKind.ARCHITECTURE
    description = "architecture has components/interfaces/boundaries (level: VERIFIED)"
    min_level = EvidenceLevel.VERIFIED
    max_age_seconds = 7200.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        arch = ctx.get("architecture")
        if arch is None:
            return self._insufficient(
                claim, reason="no architecture in context",
                level=EvidenceLevel.GENERATED,
            )
        decision = getattr(arch, "decision", None)
        if decision is None:
            return self._refuted(
                claim, reason="architecture has no decision",
                evidence=[], level=EvidenceLevel.GENERATED,
                confidence=Confidence.HIGH,
            )
        selected = getattr(decision, "selected", None)
        if selected is None:
            return self._refuted(
                claim, reason="architecture decision has no selection",
                evidence=[], level=EvidenceLevel.GENERATED,
            )
        comps = list(getattr(selected, "components", []) or [])
        ifaces = list(getattr(selected, "interfaces", []) or [])
        f_bounds = list(getattr(selected, "failure_boundaries", []) or [])
        s_bounds = list(getattr(selected, "security_boundaries", []) or [])
        aid = str(getattr(arch, "id", ""))
        # digest = hash of the shape
        shape = {
            "components": [getattr(c, "id", "") for c in comps],
            "interfaces": [getattr(i, "id", "") for i in ifaces],
            "failure_boundaries": [getattr(b, "id", "") for b in f_bounds],
            "security_boundaries": [getattr(b, "id", "") for b in s_bounds],
        }
        ref = EvidenceRef(
            kind=EvidenceKind.ARCHITECTURE_GRAPH,
            ref=aid or "architecture",
            payload={k: len(v) for k, v in shape.items()},
            description="architecture structure summary",
        )
        problems: list[str] = []
        if not comps: problems.append("no components")
        if not ifaces: problems.append("no interfaces")
        if not f_bounds: problems.append("no failure boundaries")
        if not s_bounds: problems.append("no security boundaries")
        if problems:
            return self._refuted(
                claim, reason="architecture incomplete: " + "; ".join(problems),
                evidence=[ref], level=EvidenceLevel.VERIFIED,
                confidence=Confidence.HIGH,
            )
        return self._supported(
            claim,
            reason=(
                f"components={len(comps)} interfaces={len(ifaces)} "
                f"failure_boundaries={len(f_bounds)} "
                f"security_boundaries={len(s_bounds)}"
            ),
            evidence=[ref], level=EvidenceLevel.VERIFIED,
            confidence=Confidence.HIGH,
        )


# ---- 7. Security ----
class SecurityVerifier(_Verifier):
    name = "security"
    kind = VerificationKind.SECURITY
    description = "no CRITICAL security findings (level: VERIFIED)"
    min_level = EvidenceLevel.VERIFIED
    max_age_seconds = 3600.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        report = ctx.get("critique_report")  # C21 CritiqueReport
        if report is None:
            return self._insufficient(
                claim, reason="no critique_report in context",
                level=EvidenceLevel.GENERATED,
            )
        all_findings = list(getattr(report, "findings", lambda: [])())
        sec = [
            f for f in all_findings
            if getattr(getattr(f, "category", None), "value", "") == "security"
        ]
        crit = [f for f in sec
                if getattr(getattr(f, "severity", None), "value", "") == "critical"]
        high = [f for f in sec
                if getattr(getattr(f, "severity", None), "value", "") == "high"]
        rid = str(getattr(report, "id", ""))
        ref = EvidenceRef(
            kind=EvidenceKind.SECURITY_SCAN,
            ref=rid or "security-scan",
            payload={
                "critical": len(crit), "high": len(high),
                "total_security_findings": len(sec),
            },
            description="security critique summary",
        )
        if crit:
            return self._refuted(
                claim,
                reason=f"{len(crit)} CRITICAL security finding(s)",
                evidence=[ref], level=EvidenceLevel.VERIFIED,
                confidence=Confidence.HIGH,
            )
        if high:
            # not refuted, but not fully supported → INCONCLUSIVE
            return VerificationResult(
                claim=claim, reached_level=EvidenceLevel.VERIFIED,
                method_name=self.name, method_description=self.description,
                result=ClaimResult.INCONCLUSIVE,
                confidence=Confidence.LOW,
                evidence=[ref],
                rationale=f"{len(high)} HIGH security finding(s) but no CRITICAL",
                provenance=Provenance(
                    source="verification_engine",
                    source_type=ProvenanceType.SYSTEM,
                    confidence=Confidence.LOW,
                ),
            )
        return self._supported(
            claim,
            reason=f"no CRITICAL/HIGH security findings (of {len(sec)} security checks)",
            evidence=[ref], level=EvidenceLevel.VERIFIED,
            confidence=Confidence.HIGH,
        )


# ---- 8. Evidence chain (self-check) ----
class EvidenceChainVerifier(_Verifier):
    name = "evidence_chain"
    kind = VerificationKind.EVIDENCE_CHAIN
    description = "all referenced evidence refs are present and fresh (level: VERIFIED)"
    min_level = EvidenceLevel.VERIFIED
    max_age_seconds = 3600.0

    def verify(self, claim: Claim, *, ctx: dict[str, Any]) -> VerificationResult:
        refs = ctx.get("evidence_refs") or []
        refs = [r for r in refs if isinstance(r, EvidenceRef)]
        if not refs:
            return self._insufficient(
                claim, reason="no evidence_refs supplied",
                level=EvidenceLevel.GENERATED,
            )
        # Recompute digests and compare
        bad: list[str] = []
        for r in refs:
            expected = _digest({
                "kind": r.kind.value, "ref": r.ref, "payload": r.payload,
            })
            if r.digest != expected:
                bad.append(r.ref)
        stale, stale_refs = self._check_staleness(refs)
        if bad:
            return self._refuted(
                claim,
                reason=f"tamper detected: digest mismatch on {bad}",
                evidence=refs, level=EvidenceLevel.VERIFIED,
                confidence=Confidence.HIGH,
            )
        if stale:
            return VerificationResult(
                claim=claim, reached_level=EvidenceLevel.VERIFIED,
                method_name=self.name, method_description=self.description,
                result=ClaimResult.INCONCLUSIVE,
                confidence=Confidence.LOW,
                evidence=refs,
                rationale=f"stale evidence: {len(stale_refs)} ref(s) older than {self.max_age_seconds}s",
                stale_evidence=True,
                provenance=Provenance(
                    source="verification_engine",
                    source_type=ProvenanceType.SYSTEM,
                    confidence=Confidence.LOW,
                ),
            )
        return self._supported(
            claim,
            reason=f"{len(refs)} evidence ref(s) verified (digest ok, fresh)",
            evidence=refs, level=EvidenceLevel.VERIFIED,
            confidence=Confidence.HIGH,
        )


# ════════════════════════════════════════════════════════════════════════════
# 5. VERIFICATION ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
class VerificationEngine:
    """Runs claims through verifiers. Enforces the epistemic ladder."""

    def __init__(
        self,
        *,
        max_age_seconds_default: float = 3600.0,
    ) -> None:
        self.max_age_seconds_default = max_age_seconds_default
        self.verifiers: list[_Verifier] = [
            ExistenceVerifier(),
            ExecutionVerifier(),
            TestSuiteVerifier(),
            RequirementVerifier(),
            AcceptanceVerifier(),
            ArchitectureVerifier(),
            SecurityVerifier(),
            EvidenceChainVerifier(),
        ]

    # ---- single claim ----
    def verify(
        self, claim: Claim, *, ctx: dict[str, Any] | None = None,
    ) -> VerificationResult:
        ctx = ctx or {}
        # Find first verifier that handles this kind
        for v in self.verifiers:
            if not v.can_handle(claim):
                continue
            try:
                result = v.verify(claim, ctx=ctx)
            except Exception as exc:
                log.warning("c22.verifier_error",
                            verifier=v.name, error=str(exc))
                return VerificationResult(
                    claim=claim, reached_level=EvidenceLevel.GENERATED,
                    method_name=v.name,
                    method_description=f"{v.description} (raised)",
                    result=ClaimResult.INSUFFICIENT_EVIDENCE,
                    confidence=Confidence.UNKNOWN,
                    evidence=[],
                    rationale=f"verifier raised: {type(exc).__name__}: {exc}",
                    provenance=Provenance(
                        source="verification_engine",
                        source_type=ProvenanceType.SYSTEM,
                        confidence=Confidence.UNKNOWN,
                    ),
                )
            # Enforce: no promotion above what evidence supports.
            if _LEVEL_RANK[result.reached_level] > _LEVEL_RANK[claim.target_level]:
                result.reached_level = claim.target_level
            return result
        # No verifier matched
        return VerificationResult(
            claim=claim, reached_level=EvidenceLevel.GENERATED,
            method_name="<none>",
            method_description="no verifier handles this claim kind",
            result=ClaimResult.INSUFFICIENT_EVIDENCE,
            confidence=Confidence.UNKNOWN,
            evidence=[],
            rationale=f"no verifier for kind '{claim.kind.value}'",
            provenance=Provenance(
                source="verification_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.UNKNOWN,
            ),
        )

    # ---- bundle ----
    def verify_bundle(
        self, claims: Sequence[Claim], *,
        project_id: str = "",
        contexts: Sequence[dict[str, Any]] | None = None,
    ) -> VerificationBundle:
        bundle = VerificationBundle(
            project_id=project_id,
            provenance=Provenance(
                source="verification_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        ctx_list = list(contexts or [])
        for i, claim in enumerate(claims):
            ctx = ctx_list[i] if i < len(ctx_list) else {}
            r = self.verify(claim, ctx=ctx)
            bundle.add(r)
        refuted = len(bundle.refuted())
        insufficient = len(bundle.insufficient())
        bundle.rationale = (
            f"claims={len(claims)}  ladder: {bundle.ladder}  "
            f"refuted={refuted}  insufficient={insufficient}"
        )
        return bundle


# ════════════════════════════════════════════════════════════════════════════
# 6. VERIFICATION REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class VerificationRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, bundle: VerificationBundle, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"verification_bundle:{bundle.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, bundle.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["verification", "c22"],
            provenance=bundle.provenance,
        )
        # Record failure memory for each REFUTED claim
        for r in bundle.refuted():
            self.memory.record_failure(
                f"verification_refuted:{bundle.id}:{r.id}",
                what=f"claim refuted: {_short(r.claim.text, 120)}",
                root_cause=r.rationale,
                fix=None,
                scope_id=project_id,
                provenance=r.provenance,
                confidence=Confidence.HIGH,
            )
        if self.ontology is None:
            return key

        root = self.ontology.add(
            EntityKind.VERIFICATION,
            _short(
                f"VerificationBundle {bundle.id[:8]} "
                f"({bundle.ladder})", 120,
            ),
            attributes={
                "bundle_id": bundle.id,
                "project_id": project_id,
                "ladder": bundle.ladder.to_dict(),
                "refuted": len(bundle.refuted()),
                "insufficient": len(bundle.insufficient()),
            },
            tags=["verification-bundle"],
            provenance=bundle.provenance,
        )
        for r in bundle.results:
            ve = self.ontology.add(
                EntityKind.VERIFICATION,
                _short(f"{r.method_name}: {r.claim.text}", 120),
                attributes={
                    "claim": r.claim.text,
                    "kind": r.claim.kind.value,
                    "level": r.reached_level.value,
                    "result": r.result.value,
                    "confidence": r.confidence.value,
                    "artifact_ref": r.artifact_ref,
                    "artifact_revision": r.artifact_revision,
                },
                tags=["verification-result", r.result.value],
                provenance=r.provenance,
            )
            try:
                self.ontology.link(RelationKind.CONTAINS, root.id, ve.id)
            except ValidationError:
                pass
            for e in r.evidence:
                ee = self.ontology.add(
                    EntityKind.EVIDENCE,
                    _short(f"{e.kind.value}: {e.ref}", 120),
                    attributes={
                        "kind": e.kind.value,
                        "ref": e.ref,
                        "digest": e.digest,
                        "observed_at": e.observed_at,
                    },
                    tags=["evidence", e.kind.value],
                    provenance=r.provenance,
                )
                try:
                    self.ontology.link(RelationKind.SUPPORTS, ee.id, ve.id)
                except ValidationError:
                    pass
        return root.id

    def load(self, bundle_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"verification_bundle:{bundle_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 7. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _run_self_tests() -> int:
    import importlib.util
    has_pytest = importlib.util.find_spec("pytest") is not None

    failures: list[str] = []
    passed = 0
    skipped = 0

    def check(name: str, fn: Callable[[], None], *, requires_pytest: bool = False) -> None:
        nonlocal passed, skipped
        if requires_pytest and not has_pytest:
            skipped += 1
            print(f"  – {name} (skipped: pytest not installed)")
            return
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C22 self-tests…")
    engine = VerificationEngine()

    # ---- invariants ----
    def t_no_verified_without_evidence() -> None:
        try:
            VerificationResult(
                claim=Claim("x", VerificationKind.EXISTENCE),
                reached_level=EvidenceLevel.VERIFIED,
                method_name="m",
                result=ClaimResult.SUPPORTED,
                evidence=[],
            )
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    def t_no_supported_without_evidence() -> None:
        try:
            VerificationResult(
                claim=Claim("x", VerificationKind.EXISTENCE),
                reached_level=EvidenceLevel.GENERATED,
                method_name="m",
                result=ClaimResult.SUPPORTED,
                evidence=[],
            )
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    check("invariant: VERIFIED without evidence rejected",
          t_no_verified_without_evidence)
    check("invariant: SUPPORTED without evidence rejected",
          t_no_supported_without_evidence)

    # ---- existence ----
    def t_existence_present() -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.py"
            p.write_text("X = 1\n")
            claim = Claim(
                "artifact exists", VerificationKind.EXISTENCE,
                artifact_ref=str(p),
            )
            r = engine.verify(claim)
            assert r.result is ClaimResult.SUPPORTED
            assert r.reached_level is EvidenceLevel.GENERATED
            assert r.confidence is Confidence.VERIFIED
            assert r.evidence and r.evidence[0].digest.startswith("sha256:")

    def t_existence_missing() -> None:
        claim = Claim(
            "artifact exists", VerificationKind.EXISTENCE,
            artifact_ref="/nonexistent/path/xyz.py",
        )
        r = engine.verify(claim)
        assert r.result is ClaimResult.REFUTED
        assert r.reached_level is EvidenceLevel.GENERATED

    check("existence: present → SUPPORTED/GENERATED", t_existence_present)
    check("existence: missing → REFUTED", t_existence_missing)

    # ---- execution ----
    def t_execution_ok() -> None:
        class SR:
            id = "sandbox-1"
            exit_code = 0
            stdout = "ok"
            stderr = ""
            status = type("S", (), {"value": "succeeded"})()
        claim = Claim("process runs", VerificationKind.EXECUTION)
        r = engine.verify(claim, ctx={"sandbox_result": SR()})
        assert r.result is ClaimResult.SUPPORTED
        assert r.reached_level is EvidenceLevel.EXECUTED

    def t_execution_failed() -> None:
        class SR:
            id = "sandbox-1"
            exit_code = 1
            stdout = ""
            stderr = "boom"
            status = type("S", (), {"value": "failed"})()
        claim = Claim("process runs", VerificationKind.EXECUTION)
        r = engine.verify(claim, ctx={"sandbox_result": SR()})
        assert r.result is ClaimResult.REFUTED

    def t_execution_no_evidence() -> None:
        claim = Claim("process runs", VerificationKind.EXECUTION)
        r = engine.verify(claim)
        assert r.result is ClaimResult.INSUFFICIENT_EVIDENCE

    check("execution: exit 0 → SUPPORTED/EXECUTED", t_execution_ok)
    check("execution: nonzero → REFUTED", t_execution_failed)
    check("execution: no evidence → INSUFFICIENT", t_execution_no_evidence)

    # ---- test suite ----
    def t_test_suite_pass() -> None:
        class Run:
            id = "run-1"
            status = type("S", (), {"value": "succeeded"})()
            passed = 5; failed = 0; errors = 0; skipped = 1
            results = []
        claim = Claim("suite passes", VerificationKind.TEST_SUITE)
        r = engine.verify(claim, ctx={"test_run": Run()})
        assert r.result is ClaimResult.SUPPORTED
        assert r.reached_level is EvidenceLevel.TESTED

    def t_test_suite_fail() -> None:
        class Run:
            id = "run-2"
            status = type("S", (), {"value": "tests_failed"})()
            passed = 3; failed = 2; errors = 1; skipped = 0
            results = []
        claim = Claim("suite passes", VerificationKind.TEST_SUITE)
        r = engine.verify(claim, ctx={"test_run": Run()})
        assert r.result is ClaimResult.REFUTED

    check("test suite: all pass → SUPPORTED/TESTED", t_test_suite_pass)
    check("test suite: failures → REFUTED", t_test_suite_fail)

    # ---- requirement ----
    def _mk_spec(*texts: str) -> Any:
        from sebrain.c05 import RequirementParser
        text = "Functional requirements:\n" + "".join(
            f"- {t}\n" for t in texts
        )
        return RequirementParser().parse(text)

    def t_requirement_satisfied() -> None:
        spec = _mk_spec("Users must be able to create tasks")
        # Build coverage_map with matching id (same formula as C17/C22)
        from hashlib import sha256 as _s
        req_text = spec.functional[0].text
        rid = _s(f"req::{req_text}".encode("utf-8")).hexdigest()[:16]
        coverage = {f"req:{rid}": ["test-id-1"]}

        class R:
            def __init__(self, nid, oc):
                self.nodeid = nid
                self.outcome = type("O", (), {"value": oc})()
        class Run:
            id = "run"
            status = type("S", (), {"value": "succeeded"})()
            passed = 1; failed = 0; errors = 0; skipped = 0
            results = [R("tests/test_x.py::test_create", "passed")]

        claim = Claim(
            "requirement satisfied", VerificationKind.REQUIREMENT,
            metadata={"requirement_text": req_text},
        )
        r = engine.verify(claim, ctx={
            "spec": spec, "coverage_map": coverage, "test_run": Run(),
        })
        assert r.result is ClaimResult.SUPPORTED
        assert r.reached_level is EvidenceLevel.REQUIREMENT_SATISFIED

    def t_requirement_uncovered() -> None:
        spec = _mk_spec("Users must be able to archive widgets")
        req_text = spec.functional[0].text

        class Run:
            id = "run"
            status = type("S", (), {"value": "succeeded"})()
            passed = 0; failed = 0; errors = 0; skipped = 0
            results = []

        claim = Claim(
            "requirement satisfied", VerificationKind.REQUIREMENT,
            metadata={"requirement_text": req_text},
        )
        r = engine.verify(claim, ctx={
            "spec": spec, "coverage_map": {}, "test_run": Run(),
        })
        assert r.result is ClaimResult.REFUTED
        assert r.reached_level is EvidenceLevel.TESTED

    def t_requirement_no_evidence() -> None:
        spec = _mk_spec("Users must do X")
        req_text = spec.functional[0].text
        claim = Claim(
            "requirement satisfied", VerificationKind.REQUIREMENT,
            metadata={"requirement_text": req_text},
        )
        r = engine.verify(claim, ctx={"spec": spec})
        assert r.result is ClaimResult.INSUFFICIENT_EVIDENCE

    check("requirement: covered + passing → SATISFIED",
          t_requirement_satisfied)
    check("requirement: uncovered → REFUTED",
          t_requirement_uncovered)
    check("requirement: no test_run → INSUFFICIENT",
          t_requirement_no_evidence)

    # ---- acceptance ----
    def t_acceptance_satisfied() -> None:
        from sebrain.c05 import RequirementParser
        text = (
            "Acceptance:\n"
            "- Given a valid request, when POST /tasks is called, "
            "then 201 is returned\n"
        )
        spec = RequirementParser().parse(text)
        from hashlib import sha256 as _s
        acc_text = spec.acceptance_criteria[0].text
        aid = _s(f"accept::{acc_text}".encode("utf-8")).hexdigest()[:16]
        coverage = {f"accept:{aid}": ["t1"]}

        class R:
            nodeid = "tests/test_api.py::test_post"
            outcome = type("O", (), {"value": "passed"})()
        class Run:
            results = [R()]

        claim = Claim(
            "acceptance criterion met", VerificationKind.ACCEPTANCE,
            metadata={"acceptance_text": acc_text},
        )
        r = engine.verify(claim, ctx={
            "spec": spec, "coverage_map": coverage, "test_run": Run(),
        })
        assert r.result is ClaimResult.SUPPORTED
        assert r.reached_level is EvidenceLevel.REQUIREMENT_SATISFIED

    check("acceptance: covered + passing → SATISFIED",
          t_acceptance_satisfied)

    # ---- architecture ----
    def t_architecture_ok() -> None:
        class C:
            def __init__(self, cid):
                self.id = cid; self.name = cid; self.depends_on = []
        class I:
            id = "i1"; from_component = "a"; to_component = "b"
        class Sel:
            kind = type("K", (), {"value": "layered"})()
            components = [C("a"), C("b")]
            interfaces = [I()]
            failure_boundaries = [C("fb")]
            security_boundaries = [C("sb")]
        class Dec:
            selected = Sel()
        class Arch:
            id = "arch-1"
            decision = Dec()
        claim = Claim("architecture complete", VerificationKind.ARCHITECTURE)
        r = engine.verify(claim, ctx={"architecture": Arch()})
        assert r.result is ClaimResult.SUPPORTED
        assert r.reached_level is EvidenceLevel.VERIFIED
        assert r.confidence is Confidence.HIGH

    def t_architecture_missing_boundaries() -> None:
        class Sel:
            kind = type("K", (), {"value": "layered"})()
            components = []
            interfaces = []
            failure_boundaries = []
            security_boundaries = []
        class Dec:
            selected = Sel()
        class Arch:
            id = "arch-2"
            decision = Dec()
        claim = Claim("architecture complete", VerificationKind.ARCHITECTURE)
        r = engine.verify(claim, ctx={"architecture": Arch()})
        assert r.result is ClaimResult.REFUTED

    check("architecture: complete → VERIFIED", t_architecture_ok)
    check("architecture: missing parts → REFUTED",
          t_architecture_missing_boundaries)

    # ---- security ----
    def t_security_clean() -> None:
        class F:
            class _Cat:
                value = "security"
            category = _Cat
            class _Sev:
                value = "low"
            severity = _Sev
        class R:
            id = "crit-1"
            def findings(self): return [F()]
        claim = Claim("no critical security issues", VerificationKind.SECURITY)
        r = engine.verify(claim, ctx={"critique_report": R()})
        assert r.result is ClaimResult.SUPPORTED
        assert r.reached_level is EvidenceLevel.VERIFIED

    def t_security_critical() -> None:
        class F:
            class _Cat:
                value = "security"
            category = _Cat
            class _Sev:
                value = "critical"
            severity = _Sev
        class R:
            id = "crit-2"
            def findings(self): return [F()]
        claim = Claim("no critical security issues", VerificationKind.SECURITY)
        r = engine.verify(claim, ctx={"critique_report": R()})
        assert r.result is ClaimResult.REFUTED

    def t_security_high_not_critical() -> None:
        class F:
            class _Cat:
                value = "security"
            category = _Cat
            class _Sev:
                value = "high"
            severity = _Sev
        class R:
            id = "crit-3"
            def findings(self): return [F()]
        claim = Claim("no critical security issues", VerificationKind.SECURITY)
        r = engine.verify(claim, ctx={"critique_report": R()})
        assert r.result is ClaimResult.INCONCLUSIVE

    check("security: clean → SUPPORTED", t_security_clean)
    check("security: critical finding → REFUTED", t_security_critical)
    check("security: only high (no critical) → INCONCLUSIVE",
          t_security_high_not_critical)

    # ---- evidence chain ----
    def t_evidence_chain_valid() -> None:
        ref = EvidenceRef(
            kind=EvidenceKind.MANUAL_OBSERVATION,
            ref="obs-1", payload={"x": 1},
        )
        claim = Claim("evidence is intact", VerificationKind.EVIDENCE_CHAIN)
        r = engine.verify(claim, ctx={"evidence_refs": [ref]})
        assert r.result is ClaimResult.SUPPORTED

    def t_evidence_chain_tampered() -> None:
        ref = EvidenceRef(
            kind=EvidenceKind.MANUAL_OBSERVATION,
            ref="obs-2", payload={"x": 1},
        )
        # Tamper with payload but keep digest
        ref.payload = {"x": 999}
        claim = Claim("evidence is intact", VerificationKind.EVIDENCE_CHAIN)
        r = engine.verify(claim, ctx={"evidence_refs": [ref]})
        assert r.result is ClaimResult.REFUTED
        assert "tamper" in r.rationale.lower()

    def t_evidence_chain_stale() -> None:
        old_ts = (datetime.now(timezone.utc)
                  - timedelta(hours=5)).isoformat(timespec="microseconds")
        ref = EvidenceRef(
            kind=EvidenceKind.MANUAL_OBSERVATION,
            ref="obs-3", payload={"x": 1}, observed_at=old_ts,
        )
        claim = Claim("evidence is intact", VerificationKind.EVIDENCE_CHAIN)
        r = engine.verify(claim, ctx={"evidence_refs": [ref]})
        assert r.result is ClaimResult.INCONCLUSIVE
        assert r.stale_evidence is True

    check("evidence-chain: intact → SUPPORTED", t_evidence_chain_valid)
    check("evidence-chain: tampered → REFUTED", t_evidence_chain_tampered)
    check("evidence-chain: stale → INCONCLUSIVE",
          t_evidence_chain_stale)

    # ---- epistemic ladder enforcement ----
    def t_ladder_no_leap() -> None:
        """A claim with target_level=GENERATED cannot be reported above it,
        even if the verifier tried to push it."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.py"
            p.write_text("X = 1\n")
            claim = Claim(
                "artifact exists", VerificationKind.EXISTENCE,
                target_level=EvidenceLevel.GENERATED,
                artifact_ref=str(p),
            )
            r = engine.verify(claim)
            assert r.reached_level is EvidenceLevel.GENERATED

    def t_ladder_target_lower_than_evidence() -> None:
        """If verifier can reach TESTED, but claim only asks for EXECUTED,
        we cap at EXECUTED."""
        class Run:
            id = "run"
            status = type("S", (), {"value": "succeeded"})()
            passed = 1; failed = 0; errors = 0; skipped = 0
            results = []
        claim = Claim(
            "suite ok", VerificationKind.TEST_SUITE,
            target_level=EvidenceLevel.EXECUTED,
        )
        r = engine.verify(claim, ctx={"test_run": Run()})
        assert r.reached_level is EvidenceLevel.EXECUTED

    check("ladder: cannot leap above target", t_ladder_no_leap)
    check("ladder: capped at target_level",
          t_ladder_target_lower_than_evidence)

    # ---- bundle ----
    def t_bundle_ladder_counts() -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.py"
            f.write_text("X = 1\n")
            claims = [
                Claim("file exists", VerificationKind.EXISTENCE,
                      artifact_ref=str(f)),
                Claim("file exists (missing)", VerificationKind.EXISTENCE,
                      artifact_ref="/nope/x.py"),
            ]
            bundle = engine.verify_bundle(claims, project_id="p")
            # 1 SUPPORTED (GENERATED) + 1 REFUTED (GENERATED)
            assert bundle.ladder.generated == 2
            assert bundle.ladder.satisfied == 0
            assert len(bundle.refuted()) == 1

    check("bundle: ladder aggregate counts",
          t_bundle_ladder_counts)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.py"
            f.write_text("X = 1\n")
            claim = Claim("exists", VerificationKind.EXISTENCE,
                          artifact_ref=str(f))
            r = engine.verify(claim)
            d = r.to_dict()
            assert d["id"] == r.id
            assert "claim" in d and "method_name" in d
            assert d["reached_level"] == "generated"
            assert "timestamp" in d
            assert "artifact_ref" in d
            s = r.summary()
            assert "SUPPORTED" in s or "supported" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                with tempfile.TemporaryDirectory() as tdt:
                    f = Path(tdt) / "x.py"
                    f.write_text("X = 1\n")
                    claims = [
                        Claim("file exists", VerificationKind.EXISTENCE,
                              artifact_ref=str(f)),
                        Claim("nonexistent", VerificationKind.EXISTENCE,
                              artifact_ref="/nope/x.py"),
                    ]
                    bundle = engine.verify_bundle(claims, project_id="proj-x")
                    repo = VerificationRepository(memory=mem, ontology=ont)
                    ent = repo.save(bundle, project_id="proj-x")
                    assert ent
                    loaded = repo.load(bundle.id, project_id="proj-x")
                    assert loaded is not None
                    assert loaded["ladder"]["generated"] == 2
                    # Ontology: VERIFICATION entities (bundle root + results)
                    assert ont.count(kind=EntityKind.VERIFICATION) >= 3
                    # EVIDENCE entities created for supported result
                    assert ont.count(kind=EntityKind.EVIDENCE) >= 1
                    # Refuted → failure memory
                    fails = mem.find(
                        kind=MemoryKind.FAILURE,
                        scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                    )
                    assert len(fails) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (VERIFICATION + EVIDENCE) + failures",
          t_persist)

    # ---- E2E with C05 + C18 ----
    def t_e2e_spec_test_satisfaction() -> None:
        """Real pipeline: C05 spec + C18 run + manual coverage → verify."""
        from sebrain.c05 import RequirementParser
        from sebrain.c18 import TestExecutionEngine
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text('"""pkg."""\n')
            (root / "pkg" / "x.py").write_text(
                "def add(a, b):\n    return a + b\n"
            )
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("")
            (root / "tests" / "test_x.py").write_text(
                "from pkg.x import add\n"
                "\n"
                "def test_create_item() -> None:\n"
                "    assert add(1, 2) == 3\n"
            )
            run = TestExecutionEngine().run_full(
                root=root, project_id="demo",
            )
            assert run.passed == 1

            text = (
                "Functional requirements:\n"
                "- Users must be able to create items\n"
            )
            spec = RequirementParser().parse(text)
            from hashlib import sha256 as _s
            req_text = spec.functional[0].text
            rid = _s(f"req::{req_text}".encode("utf-8")).hexdigest()[:16]

            claim = Claim(
                "requirement: create items",
                VerificationKind.REQUIREMENT,
                metadata={"requirement_text": req_text},
            )
            r = engine.verify(claim, ctx={
                "spec": spec,
                "coverage_map": {f"req:{rid}": ["test-create"]},
                "test_run": run,
            })
            assert r.result is ClaimResult.SUPPORTED
            assert r.reached_level is EvidenceLevel.REQUIREMENT_SATISFIED

    check("e2e: C05+C18 → REQUIREMENT_SATISFIED",
          t_e2e_spec_test_satisfaction, requires_pytest=True)

    # ---- E2E with C10 + C21 ----
    def t_e2e_architecture_and_security() -> None:
        from sebrain.c05 import RequirementParser
        from sebrain.c09 import TechnologySelector
        from sebrain.c10 import ArchitectureReasoner
        from sebrain.c21 import (
            Artifact, ArtifactKind, CriticEngine,
        )
        from sebrain.c06 import IntentContextEngine

        text = (
            "Build a small production-quality REST API for tasks.\n"
            "Non-functional:\n- All traffic must use HTTPS.\n"
        )
        spec = RequirementParser().parse(text)
        ic = IntentContextEngine().analyze(text, project_id="demo")
        tech = TechnologySelector().select(spec, ic, project_id="demo")
        arch = ArchitectureReasoner().reason(spec, ic, tech, project_id="demo")

        bundle_claims = [
            Claim("architecture complete", VerificationKind.ARCHITECTURE),
        ]
        bundle = engine.verify_bundle(
            bundle_claims, project_id="demo",
            contexts=[{"architecture": arch}],
        )
        assert bundle.results[0].result is ClaimResult.SUPPORTED
        assert bundle.results[0].reached_level is EvidenceLevel.VERIFIED

        # Security: run C21 critic on clean code
        critic = CriticEngine()
        art = Artifact(
            kind=ArtifactKind.CODE,
            raw_text='"""Doc."""\n\n'
                     "def add(x: int, y: int) -> int:\n"
                     '    """Add."""\n'
                     "    return x + y\n",
        )
        crit = critic.analyze(art, project_id="demo")
        sec_claims = [
            Claim("no critical security issues", VerificationKind.SECURITY),
        ]
        sec_bundle = engine.verify_bundle(
            sec_claims, project_id="demo",
            contexts=[{"critique_report": crit}],
        )
        assert sec_bundle.results[0].result is ClaimResult.SUPPORTED

    check("e2e: C10+C21 → architecture/security VERIFIED",
          t_e2e_architecture_and_security)

    # ---- missing verifier ----
    def t_missing_verifier() -> None:
        # VerificationKind.INTEGRATION has no verifier class
        claim = Claim("integration ok", VerificationKind.INTEGRATION)
        r = engine.verify(claim)
        assert r.result is ClaimResult.INSUFFICIENT_EVIDENCE
        assert "<none>" in r.method_name

    check("no verifier for kind → INSUFFICIENT_EVIDENCE (no silent pass)",
          t_missing_verifier)

    print()
    skip_note = f", {skipped} skipped (pytest not installed)" if skipped else ""
    print(f"Self-tests: {passed} passed, {len(failures)} failed{skip_note}")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 8. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C22 — Verification Engine")
    print("=" * 78)
    print("Epistemic ladder: Generated ≠ Executed ≠ Tested ≠ Verified ≠ Satisfied")
    print()

    engine = VerificationEngine()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_repo"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "__init__.py").write_text('"""pkg."""\n')
        (root / "pkg" / "x.py").write_text(
            "def add(a, b):\n    return a + b\n"
        )
        (root / "tests").mkdir()
        (root / "tests" / "__init__.py").write_text("")
        (root / "tests" / "test_x.py").write_text(
            "from pkg.x import add\n"
            "\n"
            "def test_create_item() -> None:\n"
            "    assert add(1, 2) == 3\n"
        )

        print("[1] Existence check (GENERATED):")
        r = engine.verify(Claim(
            "pkg/x.py exists", VerificationKind.EXISTENCE,
            artifact_ref=str(root / "pkg" / "x.py"),
        ))
        print(f"    {r.summary()}")

        print("\n[2] Test suite (TESTED):")
        from sebrain.c18 import TestExecutionEngine
        run = TestExecutionEngine().run_full(root=root, project_id="demo")
        r = engine.verify(
            Claim("suite passes", VerificationKind.TEST_SUITE),
            ctx={"test_run": run},
        )
        print(f"    {r.summary()}")

        print("\n[3] Requirement satisfaction (SATISFIED):")
        from sebrain.c05 import RequirementParser
        from hashlib import sha256 as _s
        text = (
            "Functional requirements:\n"
            "- Users must be able to create items\n"
        )
        spec = RequirementParser().parse(text)
        req_text = spec.functional[0].text
        rid = _s(f"req::{req_text}".encode("utf-8")).hexdigest()[:16]
        r = engine.verify(
            Claim(
                "requirement satisfied", VerificationKind.REQUIREMENT,
                metadata={"requirement_text": req_text},
            ),
            ctx={
                "spec": spec,
                "coverage_map": {f"req:{rid}": ["test-create-item"]},
                "test_run": run,
            },
        )
        print(f"    {r.summary()}")

        print("\n[4] Evidence chain tamper check (VERIFIED / REFUTED):")
        ref = EvidenceRef(
            kind=EvidenceKind.MANUAL_OBSERVATION,
            ref="obs-1", payload={"x": 1},
        )
        r = engine.verify(
            Claim("evidence intact", VerificationKind.EVIDENCE_CHAIN),
            ctx={"evidence_refs": [ref]},
        )
        print(f"    valid: {r.summary()}")
        # tamper
        ref.payload = {"x": 999}
        r = engine.verify(
            Claim("evidence intact", VerificationKind.EVIDENCE_CHAIN),
            ctx={"evidence_refs": [ref]},
        )
        print(f"    tampered: {r.summary()}")

        print("\n[5] Bundle aggregation (ladder):")
        bundle = engine.verify_bundle([
            Claim("pkg/x.py exists", VerificationKind.EXISTENCE,
                  artifact_ref=str(root / "pkg" / "x.py")),
            Claim("suite passes", VerificationKind.TEST_SUITE),
            Claim(
                "requirement satisfied", VerificationKind.REQUIREMENT,
                metadata={"requirement_text": req_text},
            ),
        ], project_id="demo", contexts=[
            {},
            {"test_run": run},
            {"spec": spec,
             "coverage_map": {f"req:{rid}": ["t1"]},
             "test_run": run},
        ])
        print(f"    {bundle.summary()}")
        print(f"    ladder: {bundle.ladder}")

        print("\n[6] Persistence:")
        with tempfile.TemporaryDirectory() as sdt:
            cfg = Config(data_dir=Path(sdt) / "sebrain", log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                with execution_scope(project_id="demo"):
                    mem = MemoryStore(app.storage)
                    ont = Ontology(app.storage)
                    repo = VerificationRepository(memory=mem, ontology=ont)
                    ent = repo.save(bundle, project_id="demo")
                    print(f"    ontology entity: {ent[:12]}…")
                    print(f"    VERIFICATION count: "
                          f"{ont.count(kind=EntityKind.VERIFICATION)}")
                    print(f"    EVIDENCE count: "
                          f"{ont.count(kind=EntityKind.EVIDENCE)}")
            finally:
                app.stop()

    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
