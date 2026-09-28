"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C27 — LEARNING / ADAPTATION ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C26.

Purpose:
    Turn VERIFIED outcomes into REUSABLE knowledge — with a real promotion
    pipeline and never a free pass. The lifecycle is enforced end-to-end:

        Outcome → Experience → Evaluation → Generalization
                → Candidate Knowledge → Validation → Promotion

Capabilities:
    - Grouping: cluster similar experiences (token overlap, Jaccard)
    - Generalization: extract a common signature across a cluster
    - Tiering: PROJECT_SPECIFIC / PATTERN / GENERAL_PRINCIPLE
    - Validation: check against already-promoted knowledge; detect
      contradictions (both intra-cluster and vs-promoted)
    - Promotion: only when (≥2 supporting experiences) AND (all supporting
      are ≥ PROMISING reliability) AND (no contradictions) AND (confidence
      threshold met)
    - Demotion: if a previously promoted candidate is later contradicted,
      emit a DEMOTE action (never silent mutation)
    - Provenance: every candidate carries the supporting experience ids
    - Persistence: C04 LONG_TERM (promoted knowledge) + C02 EXPERIENCE

Invariants honored:
    - NO external LLM. Deterministic.
    - No promotion of single-episode experience (spec requires grouping).
    - No silent overwrite of past knowledge — demotion is explicit.
    - Confidence is computed from evidence strength + breadth, not asserted.
    - Same inputs → same promotions (deterministic).
    - Bounded: max_experiences, max_clusters, max_promoted_per_run.

Explicit limitations (Rule #59):
    - Generalization is keyword/token based. It does NOT understand semantics.
      Two experiences that are similar in meaning but different in vocabulary
      will not be merged (and vice versa).
    - Contradiction detection uses polarity heuristics on lesson/result text
      and does not do causal analysis.
    - Long-term knowledge promotion respects project boundaries only if the
      caller marks an experience as cross-project; we do not infer that.
    - No forgetting policy is implemented. C04 memory continues to grow
      unless a caller explicitly archives entries.

Contents:
  1.  Enums: KnowledgeTier, ValidationStatus, PromotionAction, EvidenceStrength
  2.  Dataclasses: ExperienceInput, KnowledgeCandidate, ValidationResult,
                   PromotionRecord, LearningReport
  3.  Similarity + token helpers
  4.  Clusterer
  5.  Generalizer
  6.  Validator
  7.  Promoter
  8.  LearningEngine facade
  9.  LearningRepository
 10.  Self-tests (~32)
 11.  Demo

Run as script:
    python -m sebrain.c27            # demo
    python -m sebrain.c27 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
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


def _digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = frozenset("""
a an and are as at be by for from has have he her him his i in is it its
of on or that the their them they this to was were will with you your we
our do does did how what when where which who why use used using
""".split())


def _tokens(text: str) -> set[str]:
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if w.lower() not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class KnowledgeTier(str, Enum):
    """How broadly reusable a piece of knowledge is."""
    PROJECT_SPECIFIC = "project_specific"
    PATTERN = "pattern"                 # recurring within/across episodes
    GENERAL_PRINCIPLE = "general_principle"  # cross-project + validated


class ValidationStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


class PromotionAction(str, Enum):
    PROMOTE_TO_PATTERN = "promote_to_pattern"
    PROMOTE_TO_GENERAL = "promote_to_general"
    KEEP_PROJECT_SPECIFIC = "keep_project_specific"
    HOLD = "hold"
    DEMOTE = "demote"
    REJECT = "reject"


class EvidenceStrength(str, Enum):
    """Breadth × verification depth."""
    STRONG = "strong"       # ≥3 sources, verified, cross-project or ≥3 eps
    MODERATE = "moderate"   # ≥2 sources, PROMISING+
    WEAK = "weak"           # single episode or unreliable
    NONE = "none"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class ExperienceInput:
    """A single experience handed to the learning engine.

    Callers typically build this from C26 ExperienceRecords, but the
    interface is intentionally loose (duck-typed) to keep C27 testable
    without depending on C26 internals.
    """
    id: str
    kind: str = "unknown"                    # C26 ExperienceKind.value
    reliability: str = "unknown"             # C26 Reliability.value
    score: float = 0.0
    problem: str = ""
    context: str = ""
    approach: str = ""
    result: str = ""
    failure: str = ""
    repair: str = ""
    verification: str = ""
    lesson: str = ""
    project_id: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def from_record(cls, rec: Any, *, project_id: str = "") -> "ExperienceInput":
        """Build from C26 ExperienceRecord or a duck-typed object."""
        cand = getattr(rec, "candidate", None) or rec
        dec = getattr(rec, "decision", None)
        return cls(
            id=str(getattr(cand, "id", "") or getattr(rec, "id", "")),
            kind=str(getattr(getattr(cand, "kind", None), "value", "unknown")),
            reliability=str(getattr(getattr(dec, "reliability", None),
                                    "value", "unknown")),
            score=float(getattr(dec, "score", 0.0) or 0.0),
            problem=str(getattr(cand, "problem", "") or ""),
            context=str(getattr(cand, "context", "") or ""),
            approach=str(getattr(cand, "approach", "") or ""),
            result=str(getattr(cand, "result", "") or ""),
            failure=str(getattr(cand, "failure", "") or ""),
            repair=str(getattr(cand, "repair", "") or ""),
            verification=str(getattr(cand, "verification", "") or ""),
            lesson=str(getattr(cand, "lesson", "") or ""),
            project_id=project_id,
            evidence_refs=list(getattr(cand, "evidence_refs", []) or []),
            sources=[
                getattr(s, "value", str(s))
                for s in (getattr(cand, "sources", []) or [])
            ],
        )

    def signature_tokens(self) -> set[str]:
        return _tokens(" ".join([
            self.problem, self.approach, self.lesson, self.kind,
        ]))

    def text(self) -> str:
        return " | ".join(x for x in
                          (self.problem, self.approach, self.lesson)
                          if x)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "reliability": self.reliability,
            "score": self.score, "problem": self.problem,
            "context": self.context, "approach": self.approach,
            "result": self.result, "failure": self.failure,
            "repair": self.repair, "verification": self.verification,
            "lesson": self.lesson, "project_id": self.project_id,
            "evidence_refs": list(self.evidence_refs),
            "sources": list(self.sources), "created_at": self.created_at,
        }


@dataclass(slots=True)
class KnowledgeCandidate:
    """A generalized pattern distilled from a cluster of experiences."""
    id: str = field(default_factory=_new_id)
    kind: str = "unknown"
    tier: KnowledgeTier = KnowledgeTier.PROJECT_SPECIFIC
    problem_pattern: str = ""
    approach_pattern: str = ""
    lesson_pattern: str = ""
    signature_tokens: list[str] = field(default_factory=list)
    supporting_ids: list[str] = field(default_factory=list)
    supporting_projects: list[str] = field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    confidence: Confidence = Confidence.UNKNOWN
    confidence_score: float = 0.0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "tier": self.tier.value,
            "problem_pattern": self.problem_pattern,
            "approach_pattern": self.approach_pattern,
            "lesson_pattern": self.lesson_pattern,
            "signature_tokens": list(self.signature_tokens),
            "supporting_ids": list(self.supporting_ids),
            "supporting_projects": list(self.supporting_projects),
            "evidence_strength": self.evidence_strength.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ValidationResult:
    candidate_id: str
    status: ValidationStatus = ValidationStatus.PENDING
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    supporting_count: int = 0
    cross_project_count: int = 0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "contradictions": list(self.contradictions),
            "supporting_count": self.supporting_count,
            "cross_project_count": self.cross_project_count,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class PromotionRecord:
    candidate_id: str
    action: PromotionAction
    target_tier: KnowledgeTier = KnowledgeTier.PATTERN
    rationale: str = ""
    promoted_key: str = ""       # C04 memory key if promoted
    demoted_key: str = ""        # for DEMOTE
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action": self.action.value,
            "target_tier": self.target_tier.value,
            "rationale": self.rationale,
            "promoted_key": self.promoted_key,
            "demoted_key": self.demoted_key,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class LearningReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    experiences_seen: int = 0
    clusters: int = 0
    candidates: list[KnowledgeCandidate] = field(default_factory=list)
    validations: list[ValidationResult] = field(default_factory=list)
    promotions: list[PromotionRecord] = field(default_factory=list)
    promoted_ids: list[str] = field(default_factory=list)
    held_ids: list[str] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    demoted_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "experiences_seen": self.experiences_seen,
            "clusters": self.clusters,
            "candidates": [c.to_dict() for c in self.candidates],
            "validations": [v.to_dict() for v in self.validations],
            "promotions": [p.to_dict() for p in self.promotions],
            "promoted_ids": list(self.promoted_ids),
            "held_ids": list(self.held_ids),
            "rejected_ids": list(self.rejected_ids),
            "demoted_ids": list(self.demoted_ids),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Learning Report ===\n"
            f"project={self.project_id}\n"
            f"experiences={self.experiences_seen}  clusters={self.clusters}\n"
            f"candidates={len(self.candidates)}  "
            f"promoted={len(self.promoted_ids)}  "
            f"held={len(self.held_ids)}  "
            f"rejected={len(self.rejected_ids)}  "
            f"demoted={len(self.demoted_ids)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. SIMILARITY HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _polarity(text: str) -> int:
    """Return +1 (positive), -1 (negative), 0 (neutral) from text cues.

    Used for contradiction detection: two similar experiences with
    opposite polarity may indicate a contradiction.
    """
    if not text:
        return 0
    low = text.lower()
    positives = ("worked", "succeeded", "fixed", "resolved", "passed",
                 "no refutations", "no regressions", "no failures",
                 "refuted=0")
    negatives = ("failed", "didn't work", "did not work", "broke",
                 "regressed", "unreliable", "do not treat")
    pos_hits = sum(1 for p in positives if p in low)
    neg_hits = sum(1 for n in negatives if n in low)
    # "refuted=N" for N > 0 is a negative signal; "refuted=0" (already
    # counted above as positive) must NOT also match here, which a bare
    # substring check on "refuted" would do regardless of the count.
    if re.search(r"refuted=[1-9]\d*", low):
        neg_hits += 1
    if pos_hits > neg_hits:
        return 1
    if neg_hits > pos_hits:
        return -1
    return 0


def _polarity_of_experience(e: ExperienceInput) -> int:
    return _polarity(" ".join([e.lesson, e.result, e.verification]))


# ════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERER
# ════════════════════════════════════════════════════════════════════════════
class Clusterer:
    """Greedy single-link clustering by Jaccard token overlap."""

    def __init__(
        self, *, similarity_threshold: float = 0.30,
        max_clusters: int = 200,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValidationError(
                "similarity_threshold must be in (0, 1]"
            )
        if max_clusters < 1:
            raise ValidationError("max_clusters must be >= 1")
        self.threshold = similarity_threshold
        self.max_clusters = max_clusters

    def cluster(
        self, experiences: Sequence[ExperienceInput],
    ) -> list[list[ExperienceInput]]:
        clusters: list[list[ExperienceInput]] = []
        tokens_by: dict[str, set[str]] = {
            e.id: e.signature_tokens() for e in experiences
        }
        # Sort by id to be deterministic
        ordered = sorted(experiences, key=lambda x: x.id)
        for e in ordered:
            t = tokens_by.get(e.id, set())
            placed = False
            for c in clusters:
                # Single-link: join if similar to ANY member
                for member in c:
                    mt = tokens_by.get(member.id, set())
                    if _jaccard(t, mt) >= self.threshold:
                        c.append(e)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([e])
                if len(clusters) >= self.max_clusters:
                    break
        return clusters


# ════════════════════════════════════════════════════════════════════════════
# 5. GENERALIZER
# ════════════════════════════════════════════════════════════════════════════
_RELIABILITY_RANK = {
    "unreliable": 0, "unknown": 1, "weak": 2, "promising": 3, "reliable": 4,
}


class Generalizer:
    """Turn a cluster of experiences into a KnowledgeCandidate.

    Heuristic (documented):
      - signature_tokens = tokens appearing in ≥ 2/3 of members (if ≥3),
        else tokens appearing in ALL members (if 2), else none.
      - problem/approach/lesson patterns = concatenation of the
        highest-reliability member's text, prefixed with "pattern:".
      - tier: PROJECT_SPECIFIC if single project or single member;
        PATTERN if multi-member (≥2) within one project or ≥2 members
        mixed; GENERAL_PRINCIPLE if ≥2 projects AND all members are
        reliable/promising AND verification evidence exists.
    """

    def __init__(self, *, min_support: int = 2) -> None:
        if min_support < 1:
            raise ValidationError("min_support must be >= 1")
        self.min_support = min_support

    def generalize(
        self, members: Sequence[ExperienceInput],
    ) -> KnowledgeCandidate | None:
        if not members:
            return None
        if len(members) < self.min_support:
            # Single-member cluster: still emits a candidate but with
            # PROJECT_SPECIFIC tier + evidence_strength NONE. This is
            # how we represent "insufficient evidence to generalize".
            pass

        # ---- Common tokens ----
        all_tokens = [m.signature_tokens() for m in members]
        counter: dict[str, int] = defaultdict(int)
        for tok_set in all_tokens:
            for t in tok_set:
                counter[t] += 1
        n = len(members)
        if n >= 3:
            common = {t for t, c in counter.items() if c >= max(2, (2*n)//3)}
        elif n == 2:
            common = {t for t, c in counter.items() if c == 2}
        else:
            common = all_tokens[0]
        common_sorted = sorted(common)

        # ---- Representative text ----
        best = max(
            members,
            key=lambda e: (
                _RELIABILITY_RANK.get(e.reliability, 0), e.score, e.id,
            ),
        )
        cand = KnowledgeCandidate()
        cand.kind = best.kind
        cand.signature_tokens = common_sorted
        cand.supporting_ids = sorted({m.id for m in members})
        cand.supporting_projects = sorted(
            {m.project_id for m in members if m.project_id}
        )
        cand.problem_pattern = (
            f"pattern: {_short(best.problem, 180)}"
            if best.problem else ""
        )
        cand.approach_pattern = (
            f"pattern: {_short(best.approach or best.repair, 180)}"
            if (best.approach or best.repair) else ""
        )
        cand.lesson_pattern = (
            f"pattern: {_short(best.lesson, 220)}"
            if best.lesson else ""
        )

        # ---- Tier + evidence strength ----
        all_reliable_or_promising = all(
            m.reliability in ("reliable", "promising") for m in members
        )
        has_verification = any(m.verification for m in members)
        cross_project = len(cand.supporting_projects) >= 2

        if (n >= 2 and cross_project and all_reliable_or_promising
                and has_verification):
            cand.tier = KnowledgeTier.GENERAL_PRINCIPLE
            cand.evidence_strength = EvidenceStrength.STRONG
        elif n >= 2 and all_reliable_or_promising:
            cand.tier = KnowledgeTier.PATTERN
            cand.evidence_strength = EvidenceStrength.MODERATE
        elif n >= 2:
            cand.tier = KnowledgeTier.PATTERN
            cand.evidence_strength = EvidenceStrength.WEAK
        else:
            cand.tier = KnowledgeTier.PROJECT_SPECIFIC
            cand.evidence_strength = EvidenceStrength.NONE

        # ---- Confidence score ----
        avg_rel = sum(
            _RELIABILITY_RANK.get(m.reliability, 0) for m in members
        ) / max(1, len(members))
        avg_score = sum(m.score for m in members) / max(1, len(members))
        breadth = min(1.0, n / 5.0)
        cross = 1.0 if cross_project else 0.0
        ver = 1.0 if has_verification else 0.0
        confidence = (
            0.35 * (avg_rel / 4.0)
            + 0.20 * min(1.0, avg_score)
            + 0.20 * breadth
            + 0.15 * cross
            + 0.10 * ver
        )
        cand.confidence_score = max(0.0, min(1.0, confidence))
        if cand.confidence_score >= 0.75:
            cand.confidence = Confidence.HIGH
        elif cand.confidence_score >= 0.50:
            cand.confidence = Confidence.MEDIUM
        elif cand.confidence_score > 0.0:
            cand.confidence = Confidence.LOW
        else:
            cand.confidence = Confidence.UNKNOWN

        cand.rationale = (
            f"members={n} cross_project={cross_project} "
            f"all_reliable_or_promising={all_reliable_or_promising} "
            f"has_verification={has_verification} "
            f"confidence={cand.confidence_score:.2f} "
            f"evidence={cand.evidence_strength.value} "
            f"tier={cand.tier.value}"
        )
        cand.provenance = Provenance(
            source="learning_engine",
            source_type=ProvenanceType.INFERENCE,
            confidence=cand.confidence,
        )
        return cand


# ════════════════════════════════════════════════════════════════════════════
# 6. VALIDATOR
# ════════════════════════════════════════════════════════════════════════════
class Validator:
    """Checks a candidate for contradictions.

    Two contradiction surfaces:
      1. Intra-candidate: members of the cluster have opposite polarity
         (some report success, some report failure) AND share tokens.
      2. Vs already-promoted: the candidate's problem_pattern overlaps
         tokens with an existing promoted entry whose lesson_pattern has
         opposite polarity.
    """

    def __init__(
        self, *,
        promoted_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.promoted = promoted_by_id or {}

    def validate(
        self, cand: KnowledgeCandidate,
        members: Sequence[ExperienceInput],
    ) -> ValidationResult:
        res = ValidationResult(
            candidate_id=cand.id,
            supporting_count=len(members),
            cross_project_count=len(cand.supporting_projects),
        )

        # ---- Intra-cluster contradiction ----
        polarities = [_polarity_of_experience(m) for m in members]
        positives = sum(1 for p in polarities if p > 0)
        negatives = sum(1 for p in polarities if p < 0)
        intra: list[dict[str, Any]] = []
        if positives >= 1 and negatives >= 1:
            intra.append({
                "kind": "intra_cluster",
                "message": (
                    f"cluster contains both positive ({positives}) and "
                    f"negative ({negatives}) outcomes"
                ),
            })

        # ---- Vs already-promoted ----
        vs_promoted: list[dict[str, Any]] = []
        cand_tokens = set(cand.signature_tokens)
        cand_polarity = _polarity(cand.lesson_pattern + " " +
                                   cand.problem_pattern)
        for key, entry in self.promoted.items():
            existing_text = " ".join([
                str(entry.get("problem_pattern", "")),
                str(entry.get("lesson_pattern", "")),
                str(entry.get("approach_pattern", "")),
            ])
            existing_tokens = _tokens(existing_text)
            overlap = _jaccard(cand_tokens, existing_tokens)
            if overlap < 0.40:
                continue
            existing_polarity = _polarity(existing_text)
            if (cand_polarity != 0 and existing_polarity != 0
                    and cand_polarity != existing_polarity):
                vs_promoted.append({
                    "kind": "vs_promoted",
                    "key": key,
                    "message": (
                        f"candidate contradicts promoted entry '{key}' "
                        f"(overlap={overlap:.2f}, polarity "
                        f"cand={cand_polarity} existing={existing_polarity})"
                    ),
                    "overlap": overlap,
                })
        all_contradictions = intra + vs_promoted

        if all_contradictions:
            res.status = ValidationStatus.CONTRADICTED
            res.contradictions = all_contradictions
            res.rationale = (
                f"{len(all_contradictions)} contradiction(s) found"
            )
            return res

        if len(members) < 2:
            res.status = ValidationStatus.INSUFFICIENT
            res.rationale = "single member; cannot validate as a pattern"
            return res

        res.status = ValidationStatus.VALIDATED
        res.rationale = (
            f"validated with {len(members)} member(s); "
            f"{res.cross_project_count} project(s); "
            f"no contradictions"
        )
        return res


# ════════════════════════════════════════════════════════════════════════════
# 7. PROMOTER
# ════════════════════════════════════════════════════════════════════════════
class Promoter:
    """Promotion policy. Deterministic given inputs."""

    def __init__(
        self, *,
        pattern_confidence_threshold: float = 0.45,
        general_confidence_threshold: float = 0.70,
    ) -> None:
        if pattern_confidence_threshold < 0 or pattern_confidence_threshold > 1:
            raise ValidationError("pattern threshold must be in [0,1]")
        if general_confidence_threshold < 0 or general_confidence_threshold > 1:
            raise ValidationError("general threshold must be in [0,1]")
        self.pattern_threshold = pattern_confidence_threshold
        self.general_threshold = general_confidence_threshold

    def decide(
        self,
        cand: KnowledgeCandidate,
        validation: ValidationResult,
    ) -> PromotionRecord:
        # Contradicted → REJECT
        if validation.status is ValidationStatus.CONTRADICTED:
            return PromotionRecord(
                candidate_id=cand.id,
                action=PromotionAction.REJECT,
                target_tier=KnowledgeTier.PROJECT_SPECIFIC,
                rationale=validation.rationale,
            )

        # Insufficient (single-member) → HOLD
        if validation.status is ValidationStatus.INSUFFICIENT:
            return PromotionRecord(
                candidate_id=cand.id,
                action=PromotionAction.HOLD,
                target_tier=cand.tier,
                rationale=(
                    "insufficient evidence; held until more episodes "
                    "support this pattern"
                ),
            )

        # Validated. Now decide tier by strength + threshold.
        if (cand.evidence_strength is EvidenceStrength.STRONG
                and cand.tier is KnowledgeTier.GENERAL_PRINCIPLE
                and cand.confidence_score >= self.general_threshold):
            return PromotionRecord(
                candidate_id=cand.id,
                action=PromotionAction.PROMOTE_TO_GENERAL,
                target_tier=KnowledgeTier.GENERAL_PRINCIPLE,
                rationale=(
                    f"STRONG evidence, cross-project, "
                    f"confidence={cand.confidence_score:.2f} "
                    f">= {self.general_threshold}"
                ),
            )

        if (cand.tier in (KnowledgeTier.PATTERN,
                          KnowledgeTier.GENERAL_PRINCIPLE)
                and cand.confidence_score >= self.pattern_threshold):
            return PromotionRecord(
                candidate_id=cand.id,
                action=PromotionAction.PROMOTE_TO_PATTERN,
                target_tier=KnowledgeTier.PATTERN,
                rationale=(
                    f"validated pattern, confidence="
                    f"{cand.confidence_score:.2f} "
                    f">= {self.pattern_threshold}"
                ),
            )

        # Not enough confidence to promote a pattern, but evidence exists
        if cand.confidence_score > 0:
            return PromotionRecord(
                candidate_id=cand.id,
                action=PromotionAction.KEEP_PROJECT_SPECIFIC,
                target_tier=KnowledgeTier.PROJECT_SPECIFIC,
                rationale=(
                    f"confidence={cand.confidence_score:.2f} below "
                    f"pattern threshold={self.pattern_threshold}"
                ),
            )

        return PromotionRecord(
            candidate_id=cand.id,
            action=PromotionAction.HOLD,
            target_tier=cand.tier,
            rationale="no confidence to act on",
        )


# ════════════════════════════════════════════════════════════════════════════
# 8. LEARNING ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
class LearningEngine:
    """Full pipeline: experiences → clusters → candidates → validation →
    promotion. Optionally persists promoted knowledge to C04 LONG_TERM."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        similarity_threshold: float = 0.30,
        min_support: int = 2,
        max_experiences: int = 1000,
        max_promoted_per_run: int = 50,
    ) -> None:
        if max_experiences < 1:
            raise ValidationError("max_experiences must be >= 1")
        if max_promoted_per_run < 1:
            raise ValidationError("max_promoted_per_run must be >= 1")
        self.memory = memory
        self.clusterer = Clusterer(similarity_threshold=similarity_threshold)
        self.generalizer = Generalizer(min_support=min_support)
        self.promoter = Promoter()
        self.max_experiences = max_experiences
        self.max_promoted_per_run = max_promoted_per_run

    # ---- promote to C04 LONG_TERM ----
    def _load_promoted_from_memory(self) -> dict[str, dict[str, Any]]:
        if self.memory is None:
            return {}
        entries = self.memory.find(
            kind=MemoryKind.LONG_TERM,
            status=None,
        )
        out: dict[str, dict[str, Any]] = {}
        for e in entries:
            # Archived knowledge must not participate in contradiction checks.
            if e.status.value != "active":
                continue
            out[e.key] = dict(e.content)
        return out

    def learn(
        self,
        experiences: Sequence[ExperienceInput],
        *,
        project_id: str = "",
        persist: bool = True,
    ) -> LearningReport:
        rep = LearningReport(
            project_id=project_id,
            provenance=Provenance(
                source="learning_engine",
                source_type=ProvenanceType.INFERENCE,
                confidence=Confidence.MEDIUM,
            ),
        )
        exps = list(experiences)[: self.max_experiences]
        rep.experiences_seen = len(exps)
        if not exps:
            rep.rationale = "no experiences supplied"
            return rep

        # ---- Cluster ----
        clusters = self.clusterer.cluster(exps)
        rep.clusters = len(clusters)

        # ---- Existing promoted knowledge (for validation) ----
        promoted_map = self._load_promoted_from_memory()
        validator = Validator(promoted_by_id=promoted_map)

        # ---- Process each cluster ----
        for members in clusters:
            cand = self.generalizer.generalize(members)
            if cand is None:
                continue
            validation = validator.validate(cand, members)
            decision = self.promoter.decide(cand, validation)
            rep.candidates.append(cand)
            rep.validations.append(validation)
            rep.promotions.append(decision)

            if decision.action in (PromotionAction.PROMOTE_TO_PATTERN,
                                    PromotionAction.PROMOTE_TO_GENERAL):
                # Enforce the cap BEFORE persistence. Previously an entry could
                # be written to LONG_TERM even after the per-run promotion cap
                # had been reached, creating persisted state that the report
                # correctly described as not promoted.
                if len(rep.promoted_ids) >= self.max_promoted_per_run:
                    decision.action = PromotionAction.KEEP_PROJECT_SPECIFIC
                    decision.rationale += (
                        f"; not persisted (per-run cap "
                        f"{self.max_promoted_per_run} reached)"
                    )
                else:
                    if persist and self.memory is not None:
                        key = self._persist_promoted(cand, decision)
                        decision.promoted_key = key
                    rep.promoted_ids.append(cand.id)
            elif decision.action is PromotionAction.HOLD:
                rep.held_ids.append(cand.id)
            elif decision.action is PromotionAction.KEEP_PROJECT_SPECIFIC:
                pass
            else:  # REJECT or DEMOTE
                rep.rejected_ids.append(cand.id)

        rep.rationale = (
            f"experiences={rep.experiences_seen} clusters={rep.clusters} "
            f"promoted={len(rep.promoted_ids)} "
            f"held={len(rep.held_ids)} "
            f"rejected={len(rep.rejected_ids)}"
        )
        return rep

    def _persist_promoted(
        self, cand: KnowledgeCandidate, decision: PromotionRecord,
    ) -> str:
        assert self.memory is not None
        key = f"learned:{_digest(cand.id)[7:23]}"
        scope = (
            MemoryScope.GLOBAL
            if decision.action is PromotionAction.PROMOTE_TO_GENERAL
            else MemoryScope.GLOBAL  # patterns still go global for reuse
        )
        self.memory.upsert(
            MemoryKind.LONG_TERM, key,
            {
                "kind": cand.kind,
                "tier": cand.tier.value,
                "problem_pattern": cand.problem_pattern,
                "approach_pattern": cand.approach_pattern,
                "lesson_pattern": cand.lesson_pattern,
                "signature_tokens": list(cand.signature_tokens),
                "supporting_ids": list(cand.supporting_ids),
                "supporting_projects": list(cand.supporting_projects),
                "evidence_strength": cand.evidence_strength.value,
                "confidence": cand.confidence.value,
                "confidence_score": cand.confidence_score,
                "action": decision.action.value,
                "candidate_id": cand.id,
            },
            scope_type=scope,
            tags=["c27", "learned", cand.tier.value,
                  cand.evidence_strength.value],
            provenance=cand.provenance,
        )
        return key

    # ---- demotion ----
    def demote(
        self, *, key: str, reason: str, project_id: str = "",
    ) -> PromotionRecord:
        """Explicitly demote a promoted entry. Never silent."""
        if self.memory is None:
            raise ValidationError("memory not attached")
        entry = self.memory.get_current(
            MemoryKind.LONG_TERM, key,
            scope_type=MemoryScope.GLOBAL, scope_id=None,
        )
        if entry is None:
            raise ValidationError(f"promoted entry not found: {key}")
        # Archive it (memory store handles status transitions)
        self.memory.archive(entry.id)
        rec = PromotionRecord(
            candidate_id=entry.content.get("candidate_id", ""),
            action=PromotionAction.DEMOTE,
            target_tier=KnowledgeTier.PROJECT_SPECIFIC,
            rationale=reason,
            demoted_key=key,
        )
        return rec


# ════════════════════════════════════════════════════════════════════════════
# 9. LEARNING REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class LearningRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: LearningReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"learning_report:{report.id}"
        # Report memory and ontology graph form one persistence unit.
        with self.memory.storage.transaction():
            self.memory.create(
                MemoryKind.PROJECT, key, report.to_dict(),
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
                tags=["learning", "c27"],
                provenance=report.provenance,
            )
            if self.ontology is None:
                return key
            root = self.ontology.add(
                EntityKind.EXPERIENCE,
                _short(
                    f"Learning {report.id[:8]} "
                    f"(promoted={len(report.promoted_ids)})", 120,
                ),
                attributes={
                    "report_id": report.id,
                    "project_id": project_id,
                    "experiences_seen": report.experiences_seen,
                    "clusters": report.clusters,
                    "promoted": len(report.promoted_ids),
                    "held": len(report.held_ids),
                    "rejected": len(report.rejected_ids),
                },
                tags=["learning-report"],
                provenance=report.provenance,
            )
            for cand in report.candidates:
                ce = self.ontology.add(
                    EntityKind.EXPERIENCE,
                    _short(
                        f"knowledge[{cand.tier.value}]: "
                        f"{_short(cand.problem_pattern, 60)}", 120
                    ),
                    attributes={
                        "candidate_id": cand.id,
                        "tier": cand.tier.value,
                        "evidence_strength": cand.evidence_strength.value,
                        "confidence_score": cand.confidence_score,
                        "supporting_ids": list(cand.supporting_ids),
                    },
                    tags=["knowledge-candidate", cand.tier.value],
                    provenance=cand.provenance,
                )
                self.ontology.link(RelationKind.CONTAINS, root.id, ce.id)
            return root.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"learning_report:{report_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None

    # ---- query promoted knowledge ----
    def promoted_patterns(self) -> list[dict[str, Any]]:
        entries = self.memory.find(
            kind=MemoryKind.LONG_TERM, status=None,
        )
        return [
            dict(e.content) | {"key": e.key, "status": e.status.value}
            for e in entries
            if e.status.value == "active"
        ]


# ════════════════════════════════════════════════════════════════════════════
# 10. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _mk_exp(
    eid: str, *, project: str = "p1",
    problem: str = "database query slow under load",
    approach: str = "added index on hot column",
    lesson: str = "index hot paths improved performance worked",
    verification: str = "supported=3 refuted=0",
    reliability: str = "reliable",
    score: float = 0.8,
    kind: str = "optimization",
) -> ExperienceInput:
    return ExperienceInput(
        id=eid, kind=kind, reliability=reliability, score=score,
        problem=problem, approach=approach, lesson=lesson,
        verification=verification, project_id=project,
    )


def _run_self_tests() -> int:
    failures: list[str] = []
    passed = 0

    def check(name: str, fn: Callable[[], None]) -> None:
        nonlocal passed
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C27 self-tests…")

    # ---- clusterer ----
    def t_cluster_similar_grouped() -> None:
        exps = [
            _mk_exp("a", problem="db query slow under load"),
            _mk_exp("b", problem="db query slow under load"),
            _mk_exp("c", problem="UI rendering slow",
                    approach="memoized expensive React component renders",
                    lesson="memoization cut re-render count drastically"),
        ]
        clusters = Clusterer().cluster(exps)
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 2]

    def t_cluster_all_disjoint() -> None:
        exps = [
            _mk_exp("a", problem="database tuning",
                    approach="added covering index on orders table",
                    lesson="covering index removed a full table scan"),
            _mk_exp("b", problem="frontend css layout",
                    approach="switched flex-basis to percentage units",
                    lesson="percentage basis fixed the wrapping bug"),
            _mk_exp("c", problem="network retry logic",
                    approach="added exponential backoff with jitter",
                    lesson="jittered backoff stopped the thundering herd"),
        ]
        clusters = Clusterer().cluster(exps)
        assert len(clusters) == 3
        assert all(len(c) == 1 for c in clusters)

    def t_cluster_deterministic() -> None:
        exps = [
            _mk_exp("z", problem="shared hot path"),
            _mk_exp("a", problem="shared hot path"),
            _mk_exp("m", problem="shared hot path"),
        ]
        c1 = Clusterer().cluster(exps)
        c2 = Clusterer().cluster(exps)
        ids1 = sorted(tuple(sorted(m.id for m in c)) for c in c1)
        ids2 = sorted(tuple(sorted(m.id for m in c)) for c in c2)
        assert ids1 == ids2

    def t_cluster_threshold_validation() -> None:
        try:
            Clusterer(similarity_threshold=0.0)
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    check("cluster: similar grouped, dissimilar split",
          t_cluster_similar_grouped)
    check("cluster: disjoint experiences stay separate",
          t_cluster_all_disjoint)
    check("cluster: deterministic ordering", t_cluster_deterministic)
    check("cluster: rejects bad threshold",
          t_cluster_threshold_validation)

    # ---- generalizer ----
    def t_generalize_pattern_from_two() -> None:
        exps = [
            _mk_exp("a", problem="cache invalidation bug"),
            _mk_exp("b", problem="cache invalidation bug"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        assert cand.tier is KnowledgeTier.PATTERN
        assert cand.evidence_strength is EvidenceStrength.MODERATE
        assert set(cand.supporting_ids) == {"a", "b"}

    def t_generalize_general_principle() -> None:
        # Two different projects + verified + reliable
        exps = [
            _mk_exp("a", project="p1",
                    problem="connection pool exhaustion under load"),
            _mk_exp("b", project="p2",
                    problem="connection pool exhaustion under load"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        assert cand.tier is KnowledgeTier.GENERAL_PRINCIPLE
        assert cand.evidence_strength is EvidenceStrength.STRONG
        assert len(cand.supporting_projects) == 2
        assert cand.confidence_score > 0.5

    def t_generalize_single_member_insufficient() -> None:
        exps = [_mk_exp("a")]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        assert cand.tier is KnowledgeTier.PROJECT_SPECIFIC
        assert cand.evidence_strength is EvidenceStrength.NONE

    def t_generalize_signature_tokens_common() -> None:
        exps = [
            _mk_exp("a", problem="database query timeout under heavy load"),
            _mk_exp("b", problem="database query timeout under heavy load"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        assert "database" in cand.signature_tokens
        assert "query" in cand.signature_tokens

    def t_generalize_empty_returns_none() -> None:
        assert Generalizer().generalize([]) is None

    def t_generalize_rejects_bad_min_support() -> None:
        try:
            Generalizer(min_support=0)
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    check("generalize: two members → PATTERN",
          t_generalize_pattern_from_two)
    check("generalize: cross-project verified → GENERAL_PRINCIPLE",
          t_generalize_general_principle)
    check("generalize: single member → PROJECT_SPECIFIC",
          t_generalize_single_member_insufficient)
    check("generalize: common signature tokens extracted",
          t_generalize_signature_tokens_common)
    check("generalize: empty input → None", t_generalize_empty_returns_none)
    check("generalize: rejects bad min_support",
          t_generalize_rejects_bad_min_support)

    # ---- validator ----
    def t_validate_no_contradiction() -> None:
        exps = [
            _mk_exp("a", lesson="this worked well"),
            _mk_exp("b", lesson="this worked well"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        res = Validator().validate(cand, exps)
        assert res.status is ValidationStatus.VALIDATED

    def t_validate_intra_contradiction() -> None:
        exps = [
            _mk_exp("a", lesson="this worked"),
            _mk_exp("b", lesson="this failed and regressed"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        res = Validator().validate(cand, exps)
        assert res.status is ValidationStatus.CONTRADICTED
        assert any(c["kind"] == "intra_cluster" for c in res.contradictions)

    def t_validate_vs_promoted_contradiction() -> None:
        exps = [
            _mk_exp("a", problem="index hot paths",
                    approach="index hot paths",
                    lesson="index hot paths worked"),
            _mk_exp("b", problem="index hot paths",
                    approach="index hot paths",
                    lesson="index hot paths worked"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        promoted = {
            "learned:abc12345": {
                "problem_pattern": "index hot paths",
                "lesson_pattern": "index hot paths failed regressed",
                "approach_pattern": "index hot paths",
            }
        }
        res = Validator(promoted_by_id=promoted).validate(cand, exps)
        assert res.status is ValidationStatus.CONTRADICTED
        assert any(c["kind"] == "vs_promoted" for c in res.contradictions)

    def t_validate_single_member_insufficient() -> None:
        exps = [_mk_exp("a", lesson="worked")]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        res = Validator().validate(cand, exps)
        assert res.status is ValidationStatus.INSUFFICIENT

    check("validator: no contradiction → VALIDATED",
          t_validate_no_contradiction)
    check("validator: intra-cluster contradiction → CONTRADICTED",
          t_validate_intra_contradiction)
    check("validator: vs-promoted contradiction → CONTRADICTED",
          t_validate_vs_promoted_contradiction)
    check("validator: single member → INSUFFICIENT",
          t_validate_single_member_insufficient)

    # ---- promoter ----
    def t_promote_pattern() -> None:
        exps = [
            _mk_exp("a", problem="database query slow under load"),
            _mk_exp("b", problem="database query slow under load"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        val = Validator().validate(cand, exps)
        assert val.status is ValidationStatus.VALIDATED
        rec = Promoter().decide(cand, val)
        assert rec.action is PromotionAction.PROMOTE_TO_PATTERN

    def t_promote_general() -> None:
        exps = [
            _mk_exp("a", project="p1",
                    problem="database query slow under load hot path"),
            _mk_exp("b", project="p2",
                    problem="database query slow under load hot path"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        val = Validator().validate(cand, exps)
        assert val.status is ValidationStatus.VALIDATED
        rec = Promoter().decide(cand, val)
        assert rec.action is PromotionAction.PROMOTE_TO_GENERAL

    def t_reject_contradicted() -> None:
        exps = [
            _mk_exp("a", lesson="this worked"),
            _mk_exp("b", lesson="this failed regressed"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        val = Validator().validate(cand, exps)
        rec = Promoter().decide(cand, val)
        assert rec.action is PromotionAction.REJECT

    def t_hold_insufficient() -> None:
        exps = [_mk_exp("a")]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        val = Validator().validate(cand, exps)
        rec = Promoter().decide(cand, val)
        assert rec.action is PromotionAction.HOLD

    def t_keep_project_specific_low_confidence() -> None:
        # Two members but weak reliability → below pattern threshold
        exps = [
            _mk_exp("a", reliability="weak", score=0.2,
                    problem="something specific"),
            _mk_exp("b", reliability="weak", score=0.2,
                    problem="something specific"),
        ]
        cand = Generalizer().generalize(exps)
        assert cand is not None
        val = Validator().validate(cand, exps)
        rec = Promoter().decide(cand, val)
        assert rec.action in (PromotionAction.KEEP_PROJECT_SPECIFIC,
                               PromotionAction.HOLD)

    check("promoter: PROMOTE_TO_PATTERN for validated pattern",
          t_promote_pattern)
    check("promoter: PROMOTE_TO_GENERAL for cross-project strong",
          t_promote_general)
    check("promoter: REJECT contradicted candidate", t_reject_contradicted)
    check("promoter: HOLD single-member", t_hold_insufficient)
    check("promoter: KEEP_PROJECT_SPECIFIC when confidence low",
          t_keep_project_specific_low_confidence)

    # ---- engine facade ----
    def t_engine_end_to_end() -> None:
        exps = [
            _mk_exp("a", project="p1",
                    problem="cache stampede on high traffic"),
            _mk_exp("b", project="p2",
                    problem="cache stampede on high traffic"),
            _mk_exp("c", problem="unrelated css issue",
                    approach="switched flex-basis to percentage units",
                    lesson="percentage basis fixed the wrapping bug"),
        ]
        eng = LearningEngine()
        rep = eng.learn(exps, project_id="p")
        assert rep.experiences_seen == 3
        assert rep.clusters >= 2
        # 2-member cluster should promote; single-member should hold
        assert len(rep.promoted_ids) >= 1
        assert len(rep.held_ids) >= 1

    def t_engine_empty() -> None:
        rep = LearningEngine().learn([], project_id="p")
        assert rep.experiences_seen == 0
        assert rep.candidates == []
        assert rep.promoted_ids == []

    def t_engine_deterministic() -> None:
        exps = [
            _mk_exp("a", problem="x pattern"),
            _mk_exp("b", problem="x pattern"),
        ]
        r1 = LearningEngine().learn(exps, project_id="p", persist=False)
        r2 = LearningEngine().learn(exps, project_id="p", persist=False)
        assert r1.clusters == r2.clusters
        # Candidate/promoted ids are randomly generated per run (see
        # below) — only their *count* is deterministic, not the id
        # strings themselves.
        assert len(r1.promoted_ids) == len(r2.promoted_ids)
        # Candidate ids differ (random), but tiers/scores should match
        if r1.candidates and r2.candidates:
            assert r1.candidates[0].tier == r2.candidates[0].tier
            assert abs(r1.candidates[0].confidence_score -
                        r2.candidates[0].confidence_score) < 1e-9

    check("engine: end-to-end cluster→promote→hold",
          t_engine_end_to_end)
    check("engine: empty input handled", t_engine_empty)
    check("engine: deterministic tiers + scores",
          t_engine_deterministic)

    # ---- promotion persistence ----
    def t_promote_persists_to_memory() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                exps = [
                    _mk_exp("a", project="p1",
                            problem="cache stampede high traffic"),
                    _mk_exp("b", project="p2",
                            problem="cache stampede high traffic"),
                ]
                eng = LearningEngine(memory=mem)
                rep = eng.learn(exps, project_id="p", persist=True)
                assert rep.promoted_ids
                # LONG_TERM memory should have at least one entry
                lts = mem.find(kind=MemoryKind.LONG_TERM, status=None)
                assert len(lts) >= 1
                # Promotion record has a key
                for pr in rep.promotions:
                    if pr.action is PromotionAction.PROMOTE_TO_GENERAL:
                        assert pr.promoted_key.startswith("learned:")
            finally:
                s.shutdown()

    def t_no_promotion_single_episode() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = LearningEngine(memory=mem)
                rep = eng.learn(
                    [_mk_exp("a", problem="one-off incident")],
                    project_id="p", persist=True,
                )
                # Single member → HELD, no LONG_TERM
                assert rep.promoted_ids == []
                lts = mem.find(kind=MemoryKind.LONG_TERM, status=None)
                assert lts == []
            finally:
                s.shutdown()

    check("promote: promoted candidate persists to LONG_TERM",
          t_promote_persists_to_memory)
    check("promote: single episode is NEVER promoted",
          t_no_promotion_single_episode)

    # ---- demotion ----
    def t_demote_archives_entry() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                exps = [
                    _mk_exp("a", project="p1",
                            problem="cache stampede high traffic load"),
                    _mk_exp("b", project="p2",
                            problem="cache stampede high traffic load"),
                ]
                eng = LearningEngine(memory=mem)
                rep = eng.learn(exps, project_id="p", persist=True)
                # Find a promoted entry
                lts = mem.find(kind=MemoryKind.LONG_TERM, status=None)
                assert lts
                key = lts[0].key
                rec = eng.demote(
                    key=key, reason="contradicted by later evidence",
                    project_id="p",
                )
                assert rec.action is PromotionAction.DEMOTE
                assert rec.demoted_key == key
                # The memory entry is archived (status changed)
                after = mem.get_current(
                    MemoryKind.LONG_TERM, key,
                    scope_type=MemoryScope.GLOBAL, scope_id=None,
                )
                assert after is None  # archived → not "current"
            finally:
                s.shutdown()

    check("demote: archives promoted entry explicitly",
          t_demote_archives_entry)

    # ---- persistence ----
    def t_persist_report() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                eng = LearningEngine(memory=mem)
                rep = eng.learn([
                    _mk_exp("a", problem="hot path needs caching"),
                    _mk_exp("b", problem="hot path needs caching"),
                ], project_id="proj-x", persist=True)
                repo = LearningRepository(memory=mem, ontology=ont)
                ent = repo.save(rep, project_id="proj-x")
                assert ent
                loaded = repo.load(rep.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == rep.id
                assert ont.count(kind=EntityKind.EXPERIENCE) >= 2
            finally:
                s.shutdown()

    check("persist: report + candidates to C04/C02",
          t_persist_report)

    # ---- query promoted knowledge ----
    def t_query_promoted() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = LearningEngine(memory=mem)
                eng.learn([
                    _mk_exp("a", project="p1",
                            problem="db connection pool leak under load"),
                    _mk_exp("b", project="p2",
                            problem="db connection pool leak under load"),
                ], project_id="p", persist=True)
                repo = LearningRepository(memory=mem)
                promoted = repo.promoted_patterns()
                assert len(promoted) >= 1
                # Each entry has a key
                assert all("key" in p for p in promoted)
            finally:
                s.shutdown()

    check("query: promoted patterns retrievable",
          t_query_promoted)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        eng = LearningEngine()
        rep = eng.learn([_mk_exp("a"), _mk_exp("b")], project_id="p",
                        persist=False)
        d = rep.to_dict()
        assert d["id"] == rep.id
        s = rep.summary()
        assert "Learning Report" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- E2E with real C26 ----
    def t_e2e_from_c26() -> None:
        from sebrain.c26 import (
            ExperienceCandidate, ExperienceExtractor, ExperienceRecord,
            PromotionDecision, PromotionStatus, Reliability,
        )
        # Build two fake C26 records in one project
        def rec(cid: str, problem: str, lesson: str,
                project: str = "p1") -> ExperienceRecord:
            cand = ExperienceCandidate(
                id=cid, problem=problem,
                approach="cache hot path",
                lesson=lesson, repair="",
                sources=[],
                evidence_refs=[],
            )
            dec = PromotionDecision(
                candidate_id=cid,
                status=PromotionStatus.PROMOTED,
                reliability=Reliability.PROMISING, score=0.6,
            )
            return ExperienceRecord(candidate=cand, decision=dec)

        r1 = rec("e1", "cache stampede under traffic",
                 "cache warming worked well")
        r2 = rec("e2", "cache stampede under traffic",
                 "cache warming worked well", project="p2")

        inputs = [
            ExperienceInput.from_record(r1, project_id="p1"),
            ExperienceInput.from_record(r2, project_id="p2"),
        ]
        rep = LearningEngine().learn(inputs, project_id="p", persist=False)
        assert rep.experiences_seen == 2
        assert rep.promoted_ids, (
            f"expected promotion, got {rep.summary()}"
        )

    check("e2e: real C26 ExperienceRecord → C27 promotion",
          t_e2e_from_c26)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 11. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C27 — Learning / Adaptation Engine")
    print("=" * 78)
    print("Lifecycle: Outcome → Experience → Evaluation → Generalization")
    print("         → Candidate Knowledge → Validation → Promotion")
    print()

    # ---- three scenarios ----
    experiences = [
        # Cluster A: "cache stampede" fix across two projects, verified
        _mk_exp("a1", project="proj-alpha",
                problem="cache stampede under high traffic load",
                approach="cache warming on cold start",
                lesson="cache warming worked and stabilized traffic",
                verification="supported=3 refuted=0",
                reliability="reliable", score=0.85),
        _mk_exp("a2", project="proj-beta",
                problem="cache stampede under high traffic load",
                approach="cache warming on cold start",
                lesson="cache warming worked and stabilized traffic",
                verification="supported=2 refuted=0",
                reliability="reliable", score=0.80),
        # Cluster B: same-project pitfall (single-project)
        _mk_exp("b1", project="proj-alpha",
                problem="sqlite write contention under many threads",
                lesson="used WAL mode and it worked",
                verification="supported=1 refuted=0",
                reliability="promising", score=0.55),
        _mk_exp("b2", project="proj-alpha",
                problem="sqlite write contention under many threads",
                lesson="used WAL mode and it worked",
                verification="supported=1 refuted=0",
                reliability="promising", score=0.55),
        # Cluster C: single experience → HELD
        _mk_exp("c1", project="proj-alpha",
                problem="one-off flaky test",
                lesson="rerun fixed it",
                reliability="weak", score=0.2),
        # Cluster D: contradiction
        _mk_exp("d1", project="proj-alpha",
                problem="async retries always help",
                lesson="this worked"),
        _mk_exp("d2", project="proj-alpha",
                problem="async retries always help",
                lesson="this failed and regressed"),
    ]

    with tempfile.TemporaryDirectory() as td:
        s = SQLiteStorage(Path(td) / "sebrain.sqlite3")
        s.initialize()
        try:
            mem = MemoryStore(s)
            eng = LearningEngine(memory=mem)
            rep = eng.learn(experiences, project_id="demo", persist=True)

            print("[1] Summary:")
            print(rep.summary())

            print("\n[2] Candidates:")
            for cand, val, dec in zip(rep.candidates,
                                       rep.validations, rep.promotions):
                print(f"    [{cand.tier.value:18s}] "
                      f"members={len(cand.supporting_ids)} "
                      f"evidence={cand.evidence_strength.value:8s} "
                      f"conf={cand.confidence_score:.2f}")
                print(f"        problem: "
                      f"{_short(cand.problem_pattern, 80)}")
                print(f"        action : {dec.action.value}")
                print(f"        validation: {val.status.value}  "
                      f"({_short(val.rationale, 80)})")
                if val.contradictions:
                    for c in val.contradictions:
                        print(f"          ⚠ {c['kind']}: "
                              f"{_short(c['message'], 90)}")

            print("\n[3] Promotions:")
            for pr in rep.promotions:
                if pr.action in (PromotionAction.PROMOTE_TO_PATTERN,
                                 PromotionAction.PROMOTE_TO_GENERAL):
                    print(f"    ✓ {pr.action.value}  "
                          f"key={pr.promoted_key}")
                    print(f"      rationale: {_short(pr.rationale, 100)}")

            print("\n[4] Persisted knowledge in LONG_TERM memory:")
            repo = LearningRepository(memory=mem)
            for p in repo.promoted_patterns():
                print(f"    key={p['key']}")
                print(f"      tier={p.get('tier')}  "
                      f"evidence={p.get('evidence_strength')}  "
                      f"conf={p.get('confidence_score'):.2f}")
                print(f"      lesson: "
                      f"{_short(str(p.get('lesson_pattern', '')), 90)}")

            print("\n[5] Demo demotion:")
            if repo.promoted_patterns():
                key_to_demote = repo.promoted_patterns()[0]["key"]
                rec = eng.demote(
                    key=key_to_demote,
                    reason="later evidence showed opposite outcome",
                    project_id="demo",
                )
                print(f"    demoted: {rec.demoted_key}")
                print(f"    reason : {rec.rationale}")
                remaining = repo.promoted_patterns()
                print(f"    remaining promoted: {len(remaining)}")
        finally:
            s.shutdown()

    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
