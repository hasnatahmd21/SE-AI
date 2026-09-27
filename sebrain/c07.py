"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C07 — REASONING ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04.

Purpose:
    Deterministic, explainable reasoning over facts, rules, constraints,
    evidence, alternatives, and dependencies.

Capabilities:
    - Constraint reasoning      (hard / soft, operators, scoring)
    - Evidence evaluation       (support/refute aggregation → verdict)
    - Comparison                (multi-dimension, weighted)
    - Inference                 (forward-chaining rule engine + variable unification)
    - Contradiction detection   (functional-predicate + negation-aware)
    - Alternative evaluation    (rank options on dimensions)
    - Uncertainty handling      (confidence propagation through rules)
    - Dependency reasoning      (transitive closure, cycle detection, topo sort)
    - Trade-off analysis        (Pareto front + weighted winner + rationale)
    - Explanation               (every conclusion carries a reasoning trace)

Invariants honored:
  - No LLM. Pure deterministic algorithms.
  - No string-template fake reasoning. Every conclusion references:
      * which rule fired (rule name + premises) OR
      * which evaluation produced it (scores, weights, thresholds)
  - Confidence propagates via min() for rules, weighted sums for evidence.
  - Cycles are detected, never infinite-loop.
  - Max iterations hard-capped on every loop.
  - Unknown ≠ inference ≠ conclusion — never collapsed.

Contents:
  1.  Enums: Verdict, InferenceRule, EvidenceStrength, TradeoffDimension
  2.  Dataclasses: Fact, Pattern, Rule, InferenceStep, Conclusion,
                   EvidenceItem, Constraint, ConstraintResult,
                   ComparisonResult, DimensionResult, ContradictionReport,
                   DependencyAnalysis, TradeoffResult, ParetoEntry,
                   ReasoningTrace
  3.  FactBase (add/get/find, dedupe, provenance)
  4.  RuleEngine (forward chaining + unification)
  5.  ConstraintReasoner
  6.  EvidenceEvaluator
  7.  Comparator
  8.  ContradictionDetector
  9.  DependencyReasoner
  10. TradeoffAnalyzer
  11. ReasoningEngine (facade + persistence)
  12. __main__ demo + self-tests

Run as script:
    python -m sebrain.c07            # demo
    python -m sebrain.c07 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import heapq
import json
import re
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from sebrain.c01 import (
    Confidence,
    Config,
    SEBrainApp,
    SQLiteStorage,
    ValidationError,
    execution_scope,
    get_logger,
)
from sebrain.c02 import (
    EntityKind,
    Ontology,
    Provenance,
    ProvenanceType,
)
from sebrain.c04 import (
    MemoryKind,
    MemoryScope,
    MemoryStore,
)


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


_CONF_RANK: dict[Confidence, int] = {
    Confidence.UNKNOWN: 0,
    Confidence.ASSUMPTION: 1,
    Confidence.LOW: 2,
    Confidence.MEDIUM: 3,
    Confidence.HIGH: 4,
    Confidence.VERIFIED: 5,
}

_RANK_CONF: dict[int, Confidence] = {v: k for k, v in _CONF_RANK.items()}


def _min_conf(*cs: Confidence) -> Confidence:
    if not cs:
        return Confidence.UNKNOWN
    return _RANK_CONF[min(_CONF_RANK[c] for c in cs)]


def _is_var(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("?")


def _is_wildcard(s: Any) -> bool:
    return s is None or s == "_"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().rstrip(".!?").lower())


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class Verdict(str, Enum):
    ENTAILED = "entailed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONTRADICTED = "contradicted"
    PARTIAL = "partial"
    UNDETERMINED = "undetermined"


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class TradeoffDimension(str, Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    COST = "cost"
    COMPLEXITY = "complexity"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    TIME_TO_MARKET = "time_to_market"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Fact:
    """(subject, predicate, object) assertion with confidence + provenance."""
    subject: str
    predicate: str
    obj: str
    id: str = field(default_factory=_new_id)
    confidence: Confidence = Confidence.MEDIUM
    provenance: Provenance = field(default_factory=Provenance)
    derived_from: list[str] = field(default_factory=list)   # fact ids
    rule_name: str | None = None
    created_at: str = field(default_factory=now_iso)

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "subject": self.subject, "predicate": self.predicate,
            "obj": self.obj, "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict(),
            "derived_from": list(self.derived_from),
            "rule_name": self.rule_name, "created_at": self.created_at,
        }


@dataclass(slots=True)
class Pattern:
    """Antecedent pattern. Fields may be literals, `?vars`, or wildcards."""
    subject: str | None
    predicate: str | None
    obj: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "obj": self.obj}


@dataclass(slots=True)
class Rule:
    name: str
    antecedents: list[Pattern]
    consequent: Pattern
    confidence: Confidence = Confidence.HIGH
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "antecedents": [p.to_dict() for p in self.antecedents],
            "consequent": self.consequent.to_dict(),
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class InferenceStep:
    rule_name: str
    premises: list[Fact]
    conclusion: Fact
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "premises": [f.to_dict() for f in self.premises],
            "conclusion": self.conclusion.to_dict(),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class EvidenceItem:
    text: str
    supports: bool                   # True = supports, False = refutes
    strength: float = 0.5            # 0..1
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "supports": self.supports,
            "strength": self.strength,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(slots=True)
class Conclusion:
    claim: str
    verdict: Verdict
    confidence: Confidence
    supporting: list[EvidenceItem] = field(default_factory=list)
    refuting: list[EvidenceItem] = field(default_factory=list)
    net_score: float = 0.0
    threshold: float = 0.0
    rationale: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim, "verdict": self.verdict.value,
            "confidence": self.confidence.value,
            "supporting": [e.to_dict() for e in self.supporting],
            "refuting": [e.to_dict() for e in self.refuting],
            "net_score": self.net_score, "threshold": self.threshold,
            "rationale": self.rationale, "trace": list(self.trace),
        }


@dataclass(slots=True)
class Constraint:
    """Data-driven constraint over a design dict."""
    name: str
    target: str                          # dot-path in design dict
    op: str                              # eq|ne|lt|le|gt|ge|in|not_in|contains|regex
    value: Any
    hard: bool = True
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "target": self.target, "op": self.op,
            "value": self.value, "hard": self.hard, "weight": self.weight,
        }


@dataclass(slots=True)
class ConstraintResult:
    constraint: Constraint
    satisfied: bool
    reason: str
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint.to_dict(),
            "satisfied": self.satisfied,
            "reason": self.reason,
            "actual": self.actual,
        }


@dataclass(slots=True)
class ConstraintReport:
    hard_satisfied: int
    hard_violated: int
    hard_unsatisfied: int
    soft_score: float
    soft_max: float
    results: list[ConstraintResult]
    valid: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_satisfied": self.hard_satisfied,
            "hard_violated": self.hard_violated,
            "hard_unsatisfied": self.hard_unsatisfied,
            "soft_score": self.soft_score,
            "soft_max": self.soft_max,
            "results": [r.to_dict() for r in self.results],
            "valid": self.valid,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class DimensionResult:
    dimension: str
    scores: dict[str, float]           # option -> score
    winner: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "scores": dict(self.scores),
            "winner": self.winner,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ComparisonResult:
    options: list[str]
    dimensions: list[DimensionResult]
    weights: dict[str, float]
    weighted_scores: dict[str, float]
    winner: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "options": list(self.options),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "weights": dict(self.weights),
            "weighted_scores": dict(self.weighted_scores),
            "winner": self.winner,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ContradictionReport:
    a: Fact
    b: Fact
    kind: str                    # "functional_value" | "explicit_negation" | "polarity"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a.to_dict(), "b": self.b.to_dict(),
            "kind": self.kind, "reason": self.reason,
        }


@dataclass(slots=True)
class DependencyAnalysis:
    root: str
    reachable: list[str]
    cycles: list[list[str]]
    topological_order: list[str] | None
    max_depth: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "reachable": list(self.reachable),
            "cycles": [list(c) for c in self.cycles],
            "topological_order": (list(self.topological_order)
                                  if self.topological_order is not None else None),
            "max_depth": self.max_depth,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ParetoEntry:
    option: str
    dominated: bool
    dominated_by: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option, "dominated": self.dominated,
            "dominated_by": list(self.dominated_by),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class TradeoffResult:
    options: list[str]
    dimensions: list[str]
    weights: dict[str, float]
    per_option: dict[str, dict[str, float]]
    weighted_scores: dict[str, float]
    winner: str
    pareto_front: list[str]
    pareto: list[ParetoEntry]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "options": list(self.options),
            "dimensions": list(self.dimensions),
            "weights": dict(self.weights),
            "per_option": {k: dict(v) for k, v in self.per_option.items()},
            "weighted_scores": dict(self.weighted_scores),
            "winner": self.winner,
            "pareto_front": list(self.pareto_front),
            "pareto": [p.to_dict() for p in self.pareto],
            "rationale": self.rationale,
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. FACT BASE
# ════════════════════════════════════════════════════════════════════════════
class FactBase:
    """In-memory fact store with dedupe on (subject, predicate, object)."""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str, str], Fact] = {}
        self._by_id: dict[str, Fact] = {}

    def add(self, fact: Fact) -> Fact:
        """Add fact. If identical key exists, keep highest-confidence one,
        but merge derivation chains."""
        key = fact.key()
        existing = self._facts.get(key)
        if existing is not None:
            if _CONF_RANK[fact.confidence] > _CONF_RANK[existing.confidence]:
                # Promote confidence, merge derivations
                existing.confidence = fact.confidence
            for pid in fact.derived_from:
                if pid not in existing.derived_from:
                    existing.derived_from.append(pid)
            if fact.rule_name and existing.rule_name != fact.rule_name:
                existing.rule_name = fact.rule_name
            return existing
        self._facts[key] = fact
        self._by_id[fact.id] = fact
        return fact

    def has(self, subject: str, predicate: str, obj: str) -> bool:
        return (subject, predicate, obj) in self._facts

    def get(self, fact_id: str) -> Fact | None:
        return self._by_id.get(fact_id)

    def all(self) -> list[Fact]:
        return list(self._facts.values())

    def find(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        min_confidence: Confidence | None = None,
    ) -> list[Fact]:
        min_rank = _CONF_RANK[min_confidence] if min_confidence else 0
        out: list[Fact] = []
        for f in self._facts.values():
            if subject is not None and f.subject != subject:
                continue
            if predicate is not None and f.predicate != predicate:
                continue
            if obj is not None and f.obj != obj:
                continue
            if _CONF_RANK[f.confidence] < min_rank:
                continue
            out.append(f)
        return out

    def __len__(self) -> int:
        return len(self._facts)

    def count(self) -> int:
        return len(self._facts)


# ════════════════════════════════════════════════════════════════════════════
# 4. RULE ENGINE — forward chaining with variable unification
# ════════════════════════════════════════════════════════════════════════════
def _unify_pattern(pattern: Pattern, fact: Fact, bindings: dict[str, str]) -> dict[str, str] | None:
    """Try to match `pattern` against `fact` given current bindings.

    Returns updated bindings on success, or None on failure.
    """
    new = dict(bindings)
    for p_val, f_val in ((pattern.subject, fact.subject),
                         (pattern.predicate, fact.predicate),
                         (pattern.obj, fact.obj)):
        if _is_wildcard(p_val):
            continue
        if _is_var(p_val):
            existing = new.get(p_val)
            if existing is None:
                new[p_val] = f_val
            elif existing != f_val:
                return None
        else:
            if p_val != f_val:
                return None
    return new


def _instantiate_pattern(pattern: Pattern, bindings: dict[str, str]) -> tuple[str, str, str] | None:
    """Instantiate a fully-bound pattern into a fact key, or None if unbound."""
    out: list[str] = []
    for val in (pattern.subject, pattern.predicate, pattern.obj):
        if _is_wildcard(val):
            return None
        if _is_var(val):
            if val not in bindings:
                return None
            out.append(bindings[val])
        else:
            out.append(val)  # type: ignore[arg-type]
    return (out[0], out[1], out[2])


class RuleEngine:
    """Forward-chaining inference engine.

    Deterministic: rules evaluated in insertion order, facts deduped by key.
    Cycle-safe: capped iterations. Every derived fact carries its derivation.
    """

    def __init__(self, *, max_iterations: int = 100) -> None:
        if max_iterations < 1:
            raise ValidationError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def infer(
        self,
        facts: FactBase,
        rules: Sequence[Rule],
    ) -> tuple[FactBase, list[InferenceStep], int]:
        """Run forward chaining. Mutates a COPY of the fact base.

        Returns (result_fact_base, inference_steps, iterations).
        """
        working = FactBase()
        for f in facts.all():
            working.add(Fact(
                subject=f.subject, predicate=f.predicate, obj=f.obj,
                id=f.id, confidence=f.confidence, provenance=f.provenance,
                derived_from=list(f.derived_from), rule_name=f.rule_name,
                created_at=f.created_at,
            ))

        steps: list[InferenceStep] = []
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            added = 0
            for rule in rules:
                matches = self._match_rule(rule, working)
                for binding, premises in matches:
                    instantiated = _instantiate_pattern(rule.consequent, binding)
                    if instantiated is None:
                        continue
                    subj, pred, obj = instantiated
                    if working.has(subj, pred, obj):
                        continue
                    # Propagate confidence: min(antecedents) combined with rule
                    ant_conf = _min_conf(*(p.confidence for p in premises))
                    new_conf = _min_conf(ant_conf, rule.confidence)
                    # Provenance: derived from rule
                    prov = Provenance(
                        source=f"rule:{rule.name}",
                        source_type=ProvenanceType.INFERENCE,
                        reference=",".join(p.id for p in premises),
                        confidence=new_conf,
                        notes=rule.rationale or "forward chaining",
                    )
                    conclusion = Fact(
                        subject=subj, predicate=pred, obj=obj,
                        confidence=new_conf, provenance=prov,
                        derived_from=[p.id for p in premises],
                        rule_name=rule.name,
                    )
                    working.add(conclusion)
                    steps.append(InferenceStep(
                        rule_name=rule.name,
                        premises=premises,
                        conclusion=conclusion,
                        rationale=(
                            f"Rule '{rule.name}' fired on {len(premises)} premises "
                            f"→ {subj} {pred} {obj}; "
                            f"confidence=min({ant_conf.value},{rule.confidence.value})"
                            f"={new_conf.value}"
                        ),
                    ))
                    added += 1
            if added == 0:
                break
        return working, steps, iteration

    def _match_rule(
        self, rule: Rule, facts: FactBase
    ) -> list[tuple[dict[str, str], list[Fact]]]:
        """Return all (bindings, premises) that satisfy every antecedent."""
        results: list[tuple[dict[str, str], list[Fact]]] = []
        if not rule.antecedents:
            return results
        current_states: list[tuple[dict[str, str], list[Fact]]] = [({}, [])]
        for pattern in rule.antecedents:
            next_states: list[tuple[dict[str, str], list[Fact]]] = []
            for bindings, premises in current_states:
                for f in facts.all():
                    # premises must be distinct facts
                    if any(f.id == p.id for p in premises):
                        continue
                    new_b = _unify_pattern(pattern, f, bindings)
                    if new_b is None:
                        continue
                    next_states.append((new_b, premises + [f]))
            current_states = next_states
            if not current_states:
                break
        for b, prem in current_states:
            results.append((b, prem))
        return results


# ════════════════════════════════════════════════════════════════════════════
# 5. CONSTRAINT REASONER
# ════════════════════════════════════════════════════════════════════════════
def _get_path(design: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Return (found, value) for a dot-path."""
    parts = path.split(".") if path else []
    cur: Any = design
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return (False, None)
    return (True, cur)


def _numeric(v: Any) -> float | None:
    """Best-effort numeric coercion for ordering comparisons. Returns None
    (not comparable) rather than raising, so callers can fall back to raw
    comparison for genuinely non-numeric values."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _apply_op(op: str, actual: Any, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op in ("lt", "le", "gt", "ge"):
        # Prefer numeric comparison when both sides look like numbers —
        # raw Python comparison on numeric *strings* is lexicographic
        # ("8" >= "11" is True as strings, since "8" > "1"), which silently
        # gives the wrong answer for version/threshold-style constraints.
        na, nb = _numeric(actual), _numeric(expected)
        if na is not None and nb is not None:
            actual, expected = na, nb
        if op == "lt":
            return actual < expected
        if op == "le":
            return actual <= expected
        if op == "gt":
            return actual > expected
        return actual >= expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    if op == "contains":
        return expected in actual
    if op == "regex":
        return re.search(str(expected), str(actual)) is not None
    raise ValidationError(f"unsupported op: {op}")


class ConstraintReasoner:
    """Data-driven constraint evaluation with hard/soft distinction."""

    def evaluate(
        self, design: dict[str, Any], constraints: Sequence[Constraint]
    ) -> ConstraintReport:
        results: list[ConstraintResult] = []
        h_ok = h_viol = h_unsat = 0
        soft_score = 0.0
        soft_max = 0.0

        for c in constraints:
            found, actual = _get_path(design, c.target)
            if not found:
                r = ConstraintResult(
                    constraint=c, satisfied=False,
                    reason=f"target '{c.target}' missing in design",
                    actual=None,
                )
                results.append(r)
                if c.hard:
                    h_unsat += 1
                else:
                    soft_max += c.weight
                continue
            try:
                ok = _apply_op(c.op, actual, c.value)
            except Exception as exc:
                r = ConstraintResult(
                    constraint=c, satisfied=False,
                    reason=f"operator error: {exc}", actual=actual,
                )
                results.append(r)
                if c.hard:
                    h_unsat += 1
                else:
                    soft_max += c.weight
                continue
            r = ConstraintResult(
                constraint=c, satisfied=ok,
                reason=f"{c.target} {c.op} {c.value!r} → {actual!r}",
                actual=actual,
            )
            results.append(r)
            if c.hard:
                if ok:
                    h_ok += 1
                else:
                    h_viol += 1
            else:
                soft_max += c.weight
                if ok:
                    soft_score += c.weight

        valid = (h_viol == 0 and h_unsat == 0)
        rationale = (
            f"hard: {h_ok} satisfied, {h_viol} violated, {h_unsat} unsatisfied; "
            f"soft: {soft_score:.2f}/{soft_max:.2f}; valid={valid}"
        )
        return ConstraintReport(
            hard_satisfied=h_ok, hard_violated=h_viol, hard_unsatisfied=h_unsat,
            soft_score=soft_score, soft_max=soft_max,
            results=results, valid=valid, rationale=rationale,
        )


# ════════════════════════════════════════════════════════════════════════════
# 6. EVIDENCE EVALUATOR
# ════════════════════════════════════════════════════════════════════════════
class EvidenceEvaluator:
    """Aggregate supporting/refuting evidence into a verdict.

    score = sum(+strength for supports) - sum(+strength for refutes)
    Threshold defaults: ±0.3 (configurable).
    """

    def __init__(self, *, threshold: float = 0.3) -> None:
        if threshold <= 0:
            raise ValidationError("threshold must be > 0")
        self.threshold = threshold

    def evaluate(
        self, claim: str, evidence: Sequence[EvidenceItem]
    ) -> Conclusion:
        supporting = [e for e in evidence if e.supports]
        refuting = [e for e in evidence if not e.supports]
        pos = sum(e.strength for e in supporting)
        neg = sum(e.strength for e in refuting)
        net = pos - neg

        if not evidence:
            verdict = Verdict.UNDETERMINED
            conf = Confidence.UNKNOWN
            reason = "no evidence provided"
        elif net >= self.threshold:
            verdict = Verdict.SUPPORTED
            # confidence by strength of net
            if net >= 2 * self.threshold:
                conf = Confidence.HIGH
            elif net >= self.threshold:
                conf = Confidence.MEDIUM
            else:
                conf = Confidence.LOW
            reason = f"net={net:.3f} >= threshold={self.threshold}"
        elif net <= -self.threshold:
            verdict = Verdict.REFUTED
            if net <= -2 * self.threshold:
                conf = Confidence.HIGH
            elif net <= -self.threshold:
                conf = Confidence.MEDIUM
            else:
                conf = Confidence.LOW
            reason = f"net={net:.3f} <= -threshold={-self.threshold}"
        elif abs(net) < 1e-9 and evidence and (pos >= self.threshold or neg >= self.threshold):
            # Net-zero but with *substantial* evidence on (at least) one
            # side that's being exactly canceled out by the other — that's
            # genuine conflicting evidence (PARTIAL), not an absence of
            # evidence. Net-zero from two sides that are each individually
            # below the threshold (e.g. 0.1 vs 0.1) is just weak evidence
            # overall, which falls through to UNDETERMINED below.
            verdict = Verdict.PARTIAL
            conf = Confidence.LOW
            reason = f"net={net:.3f} exactly balanced (pos={pos:.3f}, neg={neg:.3f})"
        else:
            verdict = Verdict.UNDETERMINED
            conf = Confidence.LOW
            reason = f"net={net:.3f} within ±threshold"

        trace = [
            {
                "step": "aggregate",
                "pos_score": pos, "neg_score": neg, "net": net,
                "threshold": self.threshold,
                "supporting_count": len(supporting),
                "refuting_count": len(refuting),
            }
        ]
        return Conclusion(
            claim=claim, verdict=verdict, confidence=conf,
            supporting=supporting, refuting=refuting,
            net_score=net, threshold=self.threshold,
            rationale=reason, trace=trace,
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. COMPARATOR
# ════════════════════════════════════════════════════════════════════════════
class Comparator:
    """Multi-dimensional option comparison with weights.

    scores: dict[option, dict[dimension, float]]
    weights: dict[dimension, float] — positive = higher is better
    """

    def compare(
        self,
        options: Sequence[str],
        scores: dict[str, dict[str, float]],
        weights: dict[str, float] | None = None,
    ) -> ComparisonResult:
        if not options:
            raise ValidationError("options must be non-empty")
        if len(set(options)) != len(options):
            raise ValidationError("duplicate option names")
        for o in options:
            if o not in scores:
                raise ValidationError(f"missing scores for option: {o}")

        # collect dimensions (union)
        dims: list[str] = []
        for o in options:
            for d in scores[o]:
                if d not in dims:
                    dims.append(d)
        if not dims:
            raise ValidationError("no dimensions found in scores")

        w = dict(weights or {})
        for d in dims:
            w.setdefault(d, 1.0)

        # per-dimension winner
        dim_results: list[DimensionResult] = []
        for d in dims:
            per = {o: float(scores[o].get(d, 0.0)) for o in options}
            best = max(per.values())
            winners = [o for o in options if abs(per[o] - best) < 1e-12]
            winner = sorted(winners)[0]
            reason = (
                f"best score {best:.3f} on '{d}' by {winner}; "
                f"scores={{{', '.join(f'{o}={per[o]:.3f}' for o in options)}}}"
            )
            dim_results.append(DimensionResult(
                dimension=d, scores=per, winner=winner, rationale=reason,
            ))

        # weighted sum
        weighted: dict[str, float] = {o: 0.0 for o in options}
        for o in options:
            for d in dims:
                weighted[o] += w[d] * float(scores[o].get(d, 0.0))

        best_w = max(weighted.values())
        winners = [o for o in options if abs(weighted[o] - best_w) < 1e-12]
        winner = sorted(winners)[0]
        if len(winners) > 1:
            rationale = (
                f"tie on weighted score {best_w:.3f} between {winners}; "
                f"picked lexicographically: {winner}"
            )
        else:
            rationale = (
                f"weighted score {best_w:.3f} for '{winner}'; "
                f"weights={w}; totals={weighted}"
            )
        return ComparisonResult(
            options=list(options), dimensions=dim_results,
            weights=w, weighted_scores=weighted,
            winner=winner, rationale=rationale,
        )


# ════════════════════════════════════════════════════════════════════════════
# 8. CONTRADICTION DETECTOR
# ════════════════════════════════════════════════════════════════════════════
_FUNCTIONAL_PREDICATES = frozenset({
    "is_a", "has_version", "has_value", "equals", "located_in",
    "belongs_to", "has_status", "has_type",
})

_NEGATION_PREFIX_RE = re.compile(
    r"^\s*(?:not[\s_-]+|no[\s_-]+|never[\s_-]+|non[\s_-]+)", re.I,
)


class ContradictionDetector:
    """Detect contradictions between facts.

    Strategies:
      1. Functional-predicate conflict: same (subject, predicate) with
         different objects where predicate is functional.
      2. Negation: object is `not X` vs `X` for same (subject, predicate).
      3. Polarity: `has_status true` vs `has_status false` for same subject.
    """

    def detect(self, facts: Sequence[Fact]) -> list[ContradictionReport]:
        reports: list[ContradictionReport] = []
        by_sp: dict[tuple[str, str], list[Fact]] = {}
        for f in facts:
            by_sp.setdefault((f.subject, f.predicate), []).append(f)

        for (s, p), group in by_sp.items():
            for i in range(len(group)):
                a = group[i]
                for j in range(i + 1, len(group)):
                    b = group[j]
                    if a.obj == b.obj:
                        continue
                    # 1) negation — checked first: when the two values are
                    # explicit negations of each other ("ready" vs "not
                    # ready"), that's a more specific and informative
                    # signal than the generic "functional predicate can't
                    # have two values" message, so it should win even for
                    # a functional predicate like has_status.
                    a_neg = _NEGATION_PREFIX_RE.sub("", a.obj).strip()
                    b_neg = _NEGATION_PREFIX_RE.sub("", b.obj).strip()
                    a_is_neg = (a_neg != a.obj)
                    b_is_neg = (b_neg != b.obj)
                    if a_is_neg != b_is_neg:
                        pos = a_neg if a_is_neg else b_neg
                        neg_side = b_neg if a_is_neg else a_neg
                        if pos == neg_side:
                            reports.append(ContradictionReport(
                                a=a, b=b, kind="explicit_negation",
                                reason=(
                                    f"'{s}' {p} '{pos}' vs its explicit negation"
                                ),
                            ))
                            continue
                    # 2) functional predicate
                    if p in _FUNCTIONAL_PREDICATES:
                        reports.append(ContradictionReport(
                            a=a, b=b, kind="functional_value",
                            reason=(
                                f"predicate '{p}' is functional; "
                                f"'{s}' cannot have both '{a.obj}' and '{b.obj}'"
                            ),
                        ))
                        continue
                    # 3) boolean polarity
                    if a.obj in ("true", "false") and b.obj in ("true", "false"):
                        reports.append(ContradictionReport(
                            a=a, b=b, kind="polarity",
                            reason=(
                                f"'{s}' {p} both '{a.obj}' and '{b.obj}'"
                            ),
                        ))
                        continue
        return reports


# ════════════════════════════════════════════════════════════════════════════
# 9. DEPENDENCY REASONER
# ════════════════════════════════════════════════════════════════════════════
class DependencyReasoner:
    """Reason over (X, depends_on, Y) facts.

    - Transitive closure (BFS)
    - Cycle detection (Tarjan-ish DFS)
    - Topological sort (Kahn) if acyclic
    """

    def __init__(self, *, max_nodes: int = 10_000) -> None:
        if max_nodes < 1:
            raise ValidationError("max_nodes must be >= 1")
        self.max_nodes = max_nodes

    def analyze(
        self, facts: Sequence[Fact], root: str, *,
        predicate: str = "depends_on",
    ) -> DependencyAnalysis:
        # Build adjacency list
        adj: dict[str, list[str]] = {}
        for f in facts:
            if f.predicate != predicate:
                continue
            adj.setdefault(f.subject, []).append(f.obj)

        # Transitive closure from root
        reachable: list[str] = []
        depth_of: dict[str, int] = {root: 0}
        queue: list[str] = [root]
        visited: set[str] = {root}
        while queue:
            cur = queue.pop(0)
            for nxt in adj.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    depth_of[nxt] = depth_of[cur] + 1
                    reachable.append(nxt)
                    if len(visited) > self.max_nodes:
                        raise ValidationError(
                            f"dependency graph exceeds max_nodes={self.max_nodes}"
                        )
                    queue.append(nxt)
        max_depth = max(depth_of.values()) if depth_of else 0

        # Cycle detection on nodes reachable from root
        cycles = self._find_cycles(adj, nodes=visited)

        # Topological sort if acyclic
        topo: list[str] | None = None
        if not cycles:
            topo = self._kahn(adj, nodes=visited)

        rationale = (
            f"root={root} reachable={len(reachable)} "
            f"max_depth={max_depth} cycles={len(cycles)}"
        )
        return DependencyAnalysis(
            root=root, reachable=reachable, cycles=cycles,
            topological_order=topo, max_depth=max_depth, rationale=rationale,
        )

    def _find_cycles(
        self, adj: dict[str, list[str]], nodes: set[str]
    ) -> list[list[str]]:
        cycles: list[list[str]] = []
        seen_canonical: set[tuple[str, ...]] = set()
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in nodes}
        stack: list[str] = []

        def dfs(u: str) -> None:
            color[u] = GRAY
            stack.append(u)
            for v in adj.get(u, []):
                if v not in color:
                    continue
                if color[v] == GRAY:
                    # extract cycle from stack
                    try:
                        idx = stack.index(v)
                    except ValueError:
                        idx = 0
                    cyc = stack[idx:] + [v]
                    canon = tuple(sorted(set(cyc)))
                    if canon not in seen_canonical:
                        seen_canonical.add(canon)
                        cycles.append(cyc)
                elif color[v] == WHITE:
                    dfs(v)
            stack.pop()
            color[u] = BLACK

        for n in sorted(nodes):
            if color[n] == WHITE:
                dfs(n)
        return cycles

    def _kahn(
        self, adj: dict[str, list[str]], nodes: set[str]
    ) -> list[str] | None:
        # adj[u] = [v, ...] means "u depends_on v" (v is a prerequisite of
        # u). For a leaf-first order (prerequisites before dependents), we
        # need Kahn's algorithm on the *reversed* graph: radj[v] = [u, ...]
        # ("v is required by u"), so that a node with no prerequisites
        # (indegree 0 in the reversed graph) is emitted first. Running
        # Kahn directly on `adj` as given would instead put the root
        # (which depends on everything) first and the leaves last.
        radj: dict[str, list[str]] = {n: [] for n in nodes}
        indeg: dict[str, int] = {n: 0 for n in nodes}
        for u in nodes:
            for v in adj.get(u, []):
                if v in indeg:
                    radj[v].append(u)
                    indeg[u] += 1
        heap: list[str] = [n for n, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        out: list[str] = []
        while heap:
            u = heapq.heappop(heap)
            out.append(u)
            for w in radj.get(u, []):
                indeg[w] -= 1
                if indeg[w] == 0:
                    heapq.heappush(heap, w)
        if len(out) != len(nodes):
            return None   # cycle present (shouldn't happen when caller guards)
        return out


# ════════════════════════════════════════════════════════════════════════════
# 10. TRADEOFF ANALYZER
# ════════════════════════════════════════════════════════════════════════════
class TradeoffAnalyzer:
    """Pareto front + weighted winner + rationale."""

    def analyze(
        self,
        options: Sequence[str],
        scores: dict[str, dict[str, float]],
        weights: dict[str, float] | None = None,
        minimize: Iterable[str] | None = None,
    ) -> TradeoffResult:
        if not options:
            raise ValidationError("options must be non-empty")
        min_dims = set(minimize or ())
        dims: list[str] = []
        for o in options:
            if o not in scores:
                raise ValidationError(f"missing scores for {o}")
            for d in scores[o]:
                if d not in dims:
                    dims.append(d)
        if not dims:
            raise ValidationError("no dimensions in scores")
        w = dict(weights or {})
        for d in dims:
            w.setdefault(d, 1.0)

        # Pareto front: option is dominated if another option is at least
        # as good on ALL dims and strictly better on at least one, where
        # "better" means >= (bigger-is-better) except for dims listed in
        # `minimize` (e.g. "cost", "latency"), where "better" means <=.
        pareto_entries: list[ParetoEntry] = []
        for o in options:
            dominators: list[str] = []
            for p in options:
                if p == o:
                    continue
                def _better_or_eq(d: str) -> bool:
                    a, b = scores[p].get(d, 0.0), scores[o].get(d, 0.0)
                    return (a <= b + 1e-12) if d in min_dims else (a >= b - 1e-12)
                def _strictly_better(d: str) -> bool:
                    a, b = scores[p].get(d, 0.0), scores[o].get(d, 0.0)
                    return (a < b - 1e-12) if d in min_dims else (a > b + 1e-12)
                ge_all = all(_better_or_eq(d) for d in dims)
                gt_any = any(_strictly_better(d) for d in dims)
                if ge_all and gt_any:
                    dominators.append(p)
            dominated = len(dominators) > 0
            reason = (
                f"dominated by {dominators}" if dominated
                else "Pareto-optimal (no option dominates on all dimensions)"
            )
            pareto_entries.append(ParetoEntry(
                option=o, dominated=dominated,
                dominated_by=dominators, rationale=reason,
            ))
        pareto_front = sorted(p.option for p in pareto_entries if not p.dominated)

        # Weighted winner. For a `minimize` dimension, a smaller raw value
        # should contribute *more* to the weighted score, so its sign is
        # flipped before weighting (the reported per_option/weighted_scores
        # still show the caller's original numbers, only the winner-ranking
        # sum is sign-adjusted).
        weighted = {o: 0.0 for o in options}
        for o in options:
            for d in dims:
                raw = float(scores[o].get(d, 0.0))
                signed = -raw if d in min_dims else raw
                weighted[o] += w[d] * signed
        best = max(weighted.values())
        winners = sorted(o for o in options if abs(weighted[o] - best) < 1e-12)
        winner = winners[0]

        rationale_parts = [
            f"weighted_scores={{{', '.join(f'{o}={weighted[o]:.3f}' for o in options)}}}",
            f"winner='{winner}' (score={best:.3f})",
            f"pareto_front={pareto_front}",
        ]
        if len(winners) > 1:
            rationale_parts.append(f"tie between {winners} → lexicographic pick")
        return TradeoffResult(
            options=list(options), dimensions=dims, weights=w,
            per_option={o: dict(scores[o]) for o in options},
            weighted_scores=weighted, winner=winner,
            pareto_front=pareto_front, pareto=pareto_entries,
            rationale="; ".join(rationale_parts),
        )


# ════════════════════════════════════════════════════════════════════════════
# 11. REASONING ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
class ReasoningEngine:
    """Single entry point composing all reasoning capabilities."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        ontology: Ontology | None = None,
        evidence_threshold: float = 0.3,
        max_inference_iterations: int = 100,
    ) -> None:
        self.memory = memory
        self.ontology = ontology
        self.rules = RuleEngine(max_iterations=max_inference_iterations)
        self.constraints = ConstraintReasoner()
        self.evidence = EvidenceEvaluator(threshold=evidence_threshold)
        self.comparator = Comparator()
        self.contradictions = ContradictionDetector()
        self.dependencies = DependencyReasoner()
        self.tradeoffs = TradeoffAnalyzer()

    # ---- fact helpers ----
    @staticmethod
    def make_fact(
        subject: str, predicate: str, obj: str, *,
        confidence: Confidence = Confidence.MEDIUM,
        source: str = "user",
        source_type: ProvenanceType = ProvenanceType.USER,
    ) -> Fact:
        return Fact(
            subject=subject, predicate=predicate, obj=obj,
            confidence=confidence,
            provenance=Provenance(
                source=source, source_type=source_type,
                confidence=confidence,
            ),
        )

    # ---- high-level API ----
    def infer(
        self, facts: Sequence[Fact] | FactBase, rules: Sequence[Rule],
    ) -> dict[str, Any]:
        fb = facts if isinstance(facts, FactBase) else self._to_fact_base(facts)
        result_fb, steps, iterations = self.rules.infer(fb, rules)
        return {
            "fact_base": result_fb,
            "new_fact_count": result_fb.count() - fb.count(),
            "total_fact_count": result_fb.count(),
            "iterations": iterations,
            "steps": steps,
        }

    def check_constraints(
        self, design: dict[str, Any], constraints: Sequence[Constraint],
    ) -> ConstraintReport:
        return self.constraints.evaluate(design, constraints)

    def evaluate_claim(
        self, claim: str, evidence: Sequence[EvidenceItem],
    ) -> Conclusion:
        return self.evidence.evaluate(claim, evidence)

    def compare(
        self, options: Sequence[str],
        scores: dict[str, dict[str, float]],
        weights: dict[str, float] | None = None,
    ) -> ComparisonResult:
        return self.comparator.compare(options, scores, weights)

    def find_contradictions(
        self, facts: Sequence[Fact] | FactBase,
    ) -> list[ContradictionReport]:
        fb = facts if isinstance(facts, FactBase) else self._to_fact_base(facts)
        return self.contradictions.detect(fb.all())

    def analyze_dependencies(
        self, facts: Sequence[Fact] | FactBase, root: str, *,
        predicate: str = "depends_on",
    ) -> DependencyAnalysis:
        fb = facts if isinstance(facts, FactBase) else self._to_fact_base(facts)
        return self.dependencies.analyze(fb.all(), root, predicate=predicate)

    def analyze_tradeoffs(
        self, options: Sequence[str],
        scores: dict[str, dict[str, float]],
        weights: dict[str, float] | None = None,
        minimize: Iterable[str] | None = None,
    ) -> TradeoffResult:
        return self.tradeoffs.analyze(options, scores, weights, minimize)

    # ---- persistence ----
    def persist_conclusion(self, conclusion: Conclusion, *, project_id: str) -> str:
        if self.memory is None:
            raise ValidationError("memory not attached")
        key = f"conclusion:{_norm_text(conclusion.claim)[:80]}"
        self.memory.upsert(
            MemoryKind.PROJECT, key, conclusion.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["reasoning", "conclusion", conclusion.verdict.value],
            provenance=Provenance(
                source="reasoning_engine",
                source_type=ProvenanceType.INFERENCE,
                confidence=conclusion.confidence,
            ),
        )
        if self.ontology is not None:
            ent = self.ontology.add(
                EntityKind.DECISION,
                f"Conclusion: {conclusion.claim[:80]}",
                attributes={
                    "verdict": conclusion.verdict.value,
                    "confidence": conclusion.confidence.value,
                    "net_score": conclusion.net_score,
                },
                tags=["conclusion", conclusion.verdict.value],
                provenance=Provenance(
                    source="reasoning_engine",
                    source_type=ProvenanceType.INFERENCE,
                    confidence=conclusion.confidence,
                ),
            )
            return ent.id
        return key

    # ---- internal ----
    @staticmethod
    def _to_fact_base(facts: Sequence[Fact]) -> FactBase:
        fb = FactBase()
        for f in facts:
            fb.add(f)
        return fb


# ════════════════════════════════════════════════════════════════════════════
# 12. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
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
            failures.append(name)
            print(f"  ✗ {name}")
            traceback.print_exc()

    print("Running C07 self-tests…")
    eng = ReasoningEngine()

    # ---- fact base ----
    def t_fact_add_dedupe() -> None:
        fb = FactBase()
        f1 = Fact("a", "p", "b")
        f2 = Fact("a", "p", "b", confidence=Confidence.HIGH)
        fb.add(f1); fb.add(f2)
        assert fb.count() == 1
        got = fb.all()[0]
        # promoted confidence
        assert got.confidence is Confidence.HIGH

    def t_fact_find() -> None:
        fb = FactBase()
        fb.add(Fact("a", "p", "b"))
        fb.add(Fact("a", "q", "c"))
        fb.add(Fact("x", "p", "b"))
        assert len(fb.find(subject="a")) == 2
        assert len(fb.find(predicate="p")) == 2
        assert len(fb.find(subject="a", predicate="p")) == 1

    check("fact base: add + dedupe + confidence promotion", t_fact_add_dedupe)
    check("fact base: find by fields", t_fact_find)

    # ---- rule engine ----
    def t_infer_single_step() -> None:
        facts = [
            eng.make_fact("postgres", "is_a", "database"),
        ]
        rules = [Rule(
            name="db_needs_driver",
            antecedents=[Pattern("?x", "is_a", "database")],
            consequent=Pattern("?x", "requires", "driver"),
            confidence=Confidence.HIGH,
            rationale="databases need a driver",
        )]
        result = eng.infer(facts, rules)
        fb = result["fact_base"]
        assert fb.has("postgres", "requires", "driver")
        assert result["new_fact_count"] == 1
        assert len(result["steps"]) == 1
        step = result["steps"][0]
        assert step.rule_name == "db_needs_driver"
        assert "driver" in step.rationale

    def t_infer_multi_step() -> None:
        facts = [
            eng.make_fact("task-api", "uses", "postgres"),
            eng.make_fact("postgres", "is_a", "database"),
        ]
        rules = [
            Rule("db_needs_driver",
                 [Pattern("?x", "is_a", "database")],
                 Pattern("?x", "requires", "driver"),
                 Confidence.HIGH),
            Rule("uses_requires",
                 [Pattern("?a", "uses", "?x"),
                  Pattern("?x", "requires", "?y")],
                 Pattern("?a", "requires", "?y"),
                 Confidence.HIGH),
        ]
        result = eng.infer(facts, rules)
        fb = result["fact_base"]
        assert fb.has("postgres", "requires", "driver")
        assert fb.has("task-api", "requires", "driver")
        assert result["new_fact_count"] == 2

    def t_infer_confidence_propagation() -> None:
        facts = [eng.make_fact("a", "p", "b", confidence=Confidence.LOW)]
        rules = [Rule(
            name="prop", antecedents=[Pattern("a", "p", "b")],
            consequent=Pattern("a", "q", "c"),
            confidence=Confidence.HIGH,
        )]
        result = eng.infer(facts, rules)
        derived = result["fact_base"].find(subject="a", predicate="q")
        assert len(derived) == 1
        # min(LOW, HIGH) = LOW
        assert derived[0].confidence is Confidence.LOW

    def t_infer_dedup() -> None:
        facts = [eng.make_fact("a", "p", "b")]
        rules = [
            Rule("r1", [Pattern("a", "p", "b")], Pattern("a", "q", "c"), Confidence.HIGH),
            Rule("r2", [Pattern("a", "p", "b")], Pattern("a", "q", "c"), Confidence.HIGH),
        ]
        result = eng.infer(facts, rules)
        # Only one derived fact despite two rules producing the same conclusion
        assert result["fact_base"].count() == 2
        assert result["new_fact_count"] == 1

    def t_infer_cycle_safe() -> None:
        facts = [eng.make_fact("a", "p", "b")]
        rules = [
            Rule("r1", [Pattern("a", "p", "b")], Pattern("b", "p", "a"), Confidence.HIGH),
            Rule("r2", [Pattern("b", "p", "a")], Pattern("a", "p", "b"), Confidence.HIGH),
        ]
        result = eng.infer(facts, rules)
        assert result["fact_base"].count() == 2
        assert result["iterations"] <= eng.rules.max_iterations

    def t_infer_variable_consistency() -> None:
        # Rule: if ?x is_a db and ?x uses ?y then ?y is_a dep
        facts = [
            eng.make_fact("postgres", "is_a", "database"),
            eng.make_fact("postgres", "uses", "tls"),
            eng.make_fact("mysql", "is_a", "database"),
            eng.make_fact("mysql", "uses", "innodb"),
        ]
        rules = [Rule(
            "dep_of_db",
            [Pattern("?x", "is_a", "database"),
             Pattern("?x", "uses", "?y")],
            Pattern("?y", "is_a", "dep"),
            Confidence.HIGH,
        )]
        result = eng.infer(facts, rules)
        fb = result["fact_base"]
        assert fb.has("tls", "is_a", "dep")
        assert fb.has("innodb", "is_a", "dep")

    check("infer: single step", t_infer_single_step)
    check("infer: multi-step chaining", t_infer_multi_step)
    check("infer: confidence = min(antecedents, rule)", t_infer_confidence_propagation)
    check("infer: dedupes conclusions from multiple rules", t_infer_dedup)
    check("infer: cycle-safe", t_infer_cycle_safe)
    check("infer: variable binding consistent across antecedents", t_infer_variable_consistency)

    # ---- constraints ----
    def t_constraint_hard_ok() -> None:
        design = {"language": "python", "version": "3.11", "memory_mb": 512}
        cons = [
            Constraint("lang", "language", "eq", "python", hard=True),
            Constraint("ver", "version", "ge", "3.10", hard=True),
            Constraint("mem", "memory_mb", "le", 1024, hard=True),
        ]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is True
        assert rep.hard_satisfied == 3
        assert rep.hard_violated == 0

    def t_constraint_hard_violation() -> None:
        design = {"language": "java", "version": "8"}
        cons = [
            Constraint("lang", "language", "eq", "python", hard=True),
            Constraint("ver", "version", "ge", "11", hard=True),
        ]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is False
        assert rep.hard_violated == 2

    def t_constraint_missing_target() -> None:
        design = {"language": "python"}
        cons = [Constraint("ver", "version", "ge", "3.10", hard=True)]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is False
        assert rep.hard_unsatisfied == 1

    def t_constraint_soft_scoring() -> None:
        design = {"latency_ms": 100, "memory_mb": 4096}
        cons = [
            Constraint("fast", "latency_ms", "le", 200, hard=False, weight=2.0),
            Constraint("light", "memory_mb", "le", 1024, hard=False, weight=1.0),
        ]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is True   # no hard constraints
        assert abs(rep.soft_score - 2.0) < 1e-9
        assert abs(rep.soft_max - 3.0) < 1e-9

    def t_constraint_in_and_regex() -> None:
        design = {"region": "us-east-1", "name": "task-api-v2"}
        cons = [
            Constraint("region", "region", "in", ["us-east-1", "eu-west-1"], hard=True),
            Constraint("name_pat", "name", "regex", r"^task-api-v\d+$", hard=True),
        ]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is True

    def t_constraint_nested_path() -> None:
        design = {"db": {"engine": "postgres", "version": "15"}}
        cons = [Constraint("engine", "db.engine", "eq", "postgres", hard=True)]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is True

    check("constraints: all hard satisfied", t_constraint_hard_ok)
    check("constraints: hard violation → invalid", t_constraint_hard_violation)
    check("constraints: missing target → unsatisfied", t_constraint_missing_target)
    check("constraints: soft scoring", t_constraint_soft_scoring)
    check("constraints: 'in' + 'regex' operators", t_constraint_in_and_regex)
    check("constraints: nested dot-path", t_constraint_nested_path)

    # ---- evidence ----
    def t_evidence_supported() -> None:
        ev = [
            EvidenceItem("benchmark shows 10k rps", supports=True, strength=0.8),
            EvidenceItem("docs confirm design", supports=True, strength=0.6),
        ]
        c = eng.evaluate_claim("FastAPI scales", ev)
        assert c.verdict is Verdict.SUPPORTED
        assert c.net_score > 0

    def t_evidence_refuted() -> None:
        ev = [
            EvidenceItem("benchmark shows 100 rps", supports=False, strength=0.9),
        ]
        c = eng.evaluate_claim("FastAPI scales", ev)
        assert c.verdict is Verdict.REFUTED
        assert c.net_score < 0

    def t_evidence_undetermined() -> None:
        ev = [
            EvidenceItem("one person says yes", supports=True, strength=0.1),
            EvidenceItem("another says no", supports=False, strength=0.1),
        ]
        c = eng.evaluate_claim("X works", ev)
        assert c.verdict is Verdict.UNDETERMINED

    def t_evidence_empty() -> None:
        c = eng.evaluate_claim("X works", [])
        assert c.verdict is Verdict.UNDETERMINED
        assert c.confidence is Confidence.UNKNOWN

    def t_evidence_partial_balanced() -> None:
        ev = [
            EvidenceItem("a", supports=True, strength=0.5),
            EvidenceItem("b", supports=False, strength=0.5),
        ]
        c = eng.evaluate_claim("X", ev)
        assert c.verdict is Verdict.PARTIAL

    def t_evidence_trace() -> None:
        ev = [EvidenceItem("a", supports=True, strength=0.8)]
        c = eng.evaluate_claim("X", ev)
        assert len(c.trace) >= 1
        assert c.trace[0]["step"] == "aggregate"
        assert c.trace[0]["pos_score"] == 0.8

    check("evidence: SUPPORTED verdict", t_evidence_supported)
    check("evidence: REFUTED verdict", t_evidence_refuted)
    check("evidence: UNDETERMINED verdict", t_evidence_undetermined)
    check("evidence: empty → UNDETERMINED + UNKNOWN", t_evidence_empty)
    check("evidence: balanced → PARTIAL", t_evidence_partial_balanced)
    check("evidence: trace populated", t_evidence_trace)

    # ---- comparison ----
    def t_compare_clear_winner() -> None:
        options = ["fastapi", "flask"]
        scores = {
            "fastapi": {"perf": 9.0, "maint": 8.0, "eco": 8.0},
            "flask": {"perf": 6.0, "maint": 8.0, "eco": 9.0},
        }
        weights = {"perf": 2.0, "maint": 1.0, "eco": 1.0}
        r = eng.compare(options, scores, weights)
        assert r.winner == "fastapi"
        assert abs(r.weighted_scores["fastapi"] - 34.0) < 1e-9
        assert abs(r.weighted_scores["flask"] - 29.0) < 1e-9

    def t_compare_tie() -> None:
        options = ["a", "b"]
        scores = {"a": {"x": 5.0}, "b": {"x": 5.0}}
        r = eng.compare(options, scores)
        assert r.winner == "a"   # lexicographic
        assert "tie" in r.rationale.lower()

    def t_compare_per_dimension() -> None:
        options = ["a", "b"]
        scores = {"a": {"x": 10.0, "y": 1.0}, "b": {"x": 1.0, "y": 10.0}}
        r = eng.compare(options, scores)
        dim_map = {d.dimension: d.winner for d in r.dimensions}
        assert dim_map["x"] == "a"
        assert dim_map["y"] == "b"

    check("compare: clear winner + weighted totals", t_compare_clear_winner)
    check("compare: tie → lexicographic + rationale", t_compare_tie)
    check("compare: per-dimension winners", t_compare_per_dimension)

    # ---- contradictions ----
    def t_contradiction_functional() -> None:
        facts = [
            eng.make_fact("postgres", "is_a", "database"),
            eng.make_fact("postgres", "is_a", "message_queue"),
        ]
        rep = eng.find_contradictions(facts)
        assert any(r.kind == "functional_value" for r in rep)

    def t_contradiction_negation() -> None:
        facts = [
            eng.make_fact("service", "has_status", "ready"),
            eng.make_fact("service", "has_status", "not ready"),
        ]
        rep = eng.find_contradictions(facts)
        assert any(r.kind == "explicit_negation" for r in rep)

    def t_contradiction_polarity() -> None:
        facts = [
            eng.make_fact("feature", "enabled", "true"),
            eng.make_fact("feature", "enabled", "false"),
        ]
        rep = eng.find_contradictions(facts)
        assert any(r.kind == "polarity" for r in rep)

    def t_contradiction_none_when_clean() -> None:
        facts = [
            eng.make_fact("a", "uses", "b"),
            eng.make_fact("a", "uses", "c"),
        ]
        rep = eng.find_contradictions(facts)
        assert rep == []

    check("contradiction: functional predicate", t_contradiction_functional)
    check("contradiction: explicit negation", t_contradiction_negation)
    check("contradiction: boolean polarity", t_contradiction_polarity)
    check("contradiction: none on non-functional multi-value", t_contradiction_none_when_clean)

    # ---- dependencies ----
    def t_dep_transitive() -> None:
        facts = [
            eng.make_fact("a", "depends_on", "b"),
            eng.make_fact("b", "depends_on", "c"),
            eng.make_fact("c", "depends_on", "d"),
        ]
        r = eng.analyze_dependencies(facts, "a")
        assert set(r.reachable) == {"b", "c", "d"}
        assert r.max_depth == 3
        assert r.cycles == []

    def t_dep_cycle_detected() -> None:
        facts = [
            eng.make_fact("a", "depends_on", "b"),
            eng.make_fact("b", "depends_on", "c"),
            eng.make_fact("c", "depends_on", "a"),
        ]
        r = eng.analyze_dependencies(facts, "a")
        assert len(r.cycles) >= 1

    def t_dep_topological() -> None:
        facts = [
            eng.make_fact("a", "depends_on", "b"),
            eng.make_fact("b", "depends_on", "c"),
        ]
        r = eng.analyze_dependencies(facts, "a")
        assert r.topological_order is not None
        # topological order: dependencies (leaves) first, then dependents
        order = r.topological_order
        assert order.index("c") < order.index("b") < order.index("a")

    def t_dep_no_cycle_topo() -> None:
        facts = [
            eng.make_fact("a", "depends_on", "b"),
            eng.make_fact("b", "depends_on", "c"),
            eng.make_fact("c", "depends_on", "a"),
        ]
        r = eng.analyze_dependencies(facts, "a")
        assert r.topological_order is None  # cycles present

    check("dep: transitive closure", t_dep_transitive)
    check("dep: cycle detected", t_dep_cycle_detected)
    check("dep: topological order (leaf-first)", t_dep_topological)
    check("dep: no topo order when cyclic", t_dep_no_cycle_topo)

    # ---- tradeoffs ----
    def t_tradeoff_weighted() -> None:
        options = ["sqlite", "postgres"]
        scores = {
            "sqlite": {"perf": 7.0, "ops": 10.0, "scale": 3.0},
            "postgres": {"perf": 8.0, "ops": 5.0, "scale": 10.0},
        }
        weights = {"perf": 1.0, "ops": 1.0, "scale": 2.0}
        r = eng.analyze_tradeoffs(options, scores, weights)
        # sqlite: 7+10+6 = 23; postgres: 8+5+20 = 33
        assert r.winner == "postgres"
        assert r.weighted_scores["postgres"] == 33.0

    def t_tradeoff_pareto() -> None:
        options = ["a", "b", "c"]
        scores = {
            "a": {"perf": 5.0, "cost": 5.0},
            "b": {"perf": 8.0, "cost": 6.0},    # dominates a on both dims
            "c": {"perf": 3.0, "cost": 10.0},   # best on "cost", so not dominated
        }
        r = eng.analyze_tradeoffs(options, scores)
        # a is dominated by b; c is Pareto-optimal (best on the cost dim)
        a_entry = next(p for p in r.pareto if p.option == "a")
        assert a_entry.dominated is True
        assert "b" in a_entry.dominated_by
        assert "c" in r.pareto_front
        assert "b" in r.pareto_front

    def t_tradeoff_no_weights() -> None:
        options = ["x", "y"]
        scores = {"x": {"d": 1.0}, "y": {"d": 2.0}}
        r = eng.analyze_tradeoffs(options, scores)
        assert r.winner == "y"
        assert r.weights["d"] == 1.0

    check("tradeoff: weighted winner", t_tradeoff_weighted)
    check("tradeoff: Pareto front computed", t_tradeoff_pareto)
    check("tradeoff: default weight = 1.0", t_tradeoff_no_weights)

    # ---- explanations ----
    def t_explanations_nonempty() -> None:
        # Every conclusion/analysis carries a rationale
        facts = [eng.make_fact("a", "p", "b")]
        rules = [Rule("r", [Pattern("a", "p", "b")],
                      Pattern("a", "q", "c"), Confidence.HIGH)]
        result = eng.infer(facts, rules)
        assert result["steps"][0].rationale
        # evidence
        c = eng.evaluate_claim("X", [EvidenceItem("e", True, 0.9)])
        assert c.rationale
        # comparison
        cmp = eng.compare(["a", "b"], {"a": {"d": 1}, "b": {"d": 2}})
        assert cmp.rationale
        # dep
        d = eng.analyze_dependencies(facts, "a")
        assert d.rationale
        # tradeoff
        tr = eng.analyze_tradeoffs(["a", "b"], {"a": {"d": 1}, "b": {"d": 2}})
        assert tr.rationale

    check("explanations: every result has rationale", t_explanations_nonempty)

    # ---- persistence ----
    def t_persist_conclusion() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                eng2 = ReasoningEngine(memory=mem, ontology=ont)
                c = eng2.evaluate_claim("FastAPI is fast", [
                    EvidenceItem("benchmark 10k rps", True, 0.9),
                ])
                ent_id = eng2.persist_conclusion(c, project_id="proj-x")
                assert ent_id
                # memory has entry
                entry = mem.get_current(
                    MemoryKind.PROJECT,
                    f"conclusion:{_norm_text(c.claim)[:80]}",
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert entry is not None
                assert entry.content["verdict"] == c.verdict.value
            finally:
                s.shutdown()

    check("persist: conclusion → memory + ontology", t_persist_conclusion)

    # ---- e2e reasoning scenario ----
    def t_e2e_full() -> None:
        # Facts about a project
        facts = [
            eng.make_fact("task-api", "uses", "postgres",
                          confidence=Confidence.HIGH),
            eng.make_fact("postgres", "is_a", "database",
                          confidence=Confidence.VERIFIED),
            eng.make_fact("postgres", "requires", "driver",
                          confidence=Confidence.HIGH),
        ]
        rules = [
            Rule("uses_requires",
                 [Pattern("?a", "uses", "?x"),
                  Pattern("?x", "requires", "?y")],
                 Pattern("?a", "requires", "?y"),
                 Confidence.HIGH,
                 rationale="using X implies inheriting X's requirements"),
        ]
        infer_result = eng.infer(facts, rules)
        fb = infer_result["fact_base"]
        assert fb.has("task-api", "requires", "driver")

        # Dependencies
        deps = eng.analyze_dependencies(fb.all(), "task-api",
                                        predicate="requires")
        assert "driver" in deps.reachable

        # Constraints on a candidate design
        design = {"db": "postgres", "driver": "psycopg2", "version": "15"}
        cons = [
            Constraint("db", "db", "eq", "postgres", hard=True),
            Constraint("driver_set", "driver", "ne", "", hard=True),
            Constraint("ver", "version", "ge", "13", hard=True),
        ]
        rep = eng.check_constraints(design, cons)
        assert rep.valid is True

        # Evidence for a design decision
        ev = [
            EvidenceItem("benchmark 10k rps", supports=True, strength=0.8),
            EvidenceItem("ops team lacks skill", supports=False, strength=0.4),
        ]
        c = eng.evaluate_claim("postgres is production-ready", ev)
        assert c.verdict is Verdict.SUPPORTED

        # Comparison of two options
        cmp = eng.compare(
            ["postgres", "sqlite"],
            {
                "postgres": {"perf": 8.0, "scale": 9.0, "ops": 5.0},
                "sqlite": {"perf": 7.0, "scale": 3.0, "ops": 10.0},
            },
            {"perf": 1.0, "scale": 2.0, "ops": 1.0},
        )
        # postgres: 8 + 18 + 5 = 31; sqlite: 7 + 6 + 10 = 23
        assert cmp.winner == "postgres"

        # Tradeoff pareto
        tr = eng.analyze_tradeoffs(
            ["postgres", "sqlite"],
            {
                "postgres": {"perf": 8.0, "scale": 9.0, "ops": 5.0},
                "sqlite": {"perf": 7.0, "scale": 3.0, "ops": 10.0},
            },
        )
        # Neither strictly dominates: pareto_front should have both
        assert set(tr.pareto_front) == {"postgres", "sqlite"}

        # Contradiction detection on conflicting facts
        contra_facts = [
            eng.make_fact("postgres", "is_a", "database"),
            eng.make_fact("postgres", "is_a", "cache"),
        ]
        contras = eng.find_contradictions(contra_facts)
        assert len(contras) >= 1

    check("e2e: reasoning across inference / deps / constraints / evidence / compare / tradeoff / contradiction", t_e2e_full)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C07 — Reasoning Engine")
    print("=" * 78)

    eng = ReasoningEngine()

    print("\n[1] Forward chaining:")
    facts = [
        eng.make_fact("task-api", "uses", "postgres", confidence=Confidence.HIGH),
        eng.make_fact("postgres", "is_a", "database", confidence=Confidence.VERIFIED),
        eng.make_fact("postgres", "requires", "driver", confidence=Confidence.HIGH),
    ]
    rules = [Rule(
        "uses_requires",
        [Pattern("?a", "uses", "?x"), Pattern("?x", "requires", "?y")],
        Pattern("?a", "requires", "?y"),
        Confidence.HIGH,
        rationale="using X implies inheriting X's requirements",
    )]
    r = eng.infer(facts, rules)
    print(f"    new facts: {r['new_fact_count']}, iterations: {r['iterations']}")
    for s in r["steps"]:
        print(f"    · {s.rationale}")

    print("\n[2] Constraint check:")
    design = {"db": "postgres", "driver": "psycopg2", "version": "15"}
    cons = [
        Constraint("db", "db", "eq", "postgres", hard=True),
        Constraint("driver_set", "driver", "ne", "", hard=True),
        Constraint("ver", "version", "ge", "13", hard=True),
        Constraint("fast", "version", "eq", "15", hard=False, weight=2.0),
    ]
    rep = eng.check_constraints(design, cons)
    print(f"    valid={rep.valid}  {rep.rationale}")
    for rr in rep.results:
        mark = "✓" if rr.satisfied else "✗"
        print(f"    {mark} {rr.constraint.name}: {rr.reason}")

    print("\n[3] Evidence evaluation:")
    ev = [
        EvidenceItem("benchmark 10k rps", supports=True, strength=0.8),
        EvidenceItem("ops team lacks skill", supports=False, strength=0.4),
        EvidenceItem("docs confirm prod usage", supports=True, strength=0.5),
    ]
    c = eng.evaluate_claim("postgres is production-ready", ev)
    print(f"    verdict    = {c.verdict.value}")
    print(f"    confidence = {c.confidence.value}")
    print(f"    net_score  = {c.net_score:.3f} (threshold={c.threshold})")

    print("\n[4] Multi-dim comparison:")
    cmp = eng.compare(
        ["postgres", "sqlite"],
        {
            "postgres": {"perf": 8.0, "scale": 9.0, "ops": 5.0},
            "sqlite": {"perf": 7.0, "scale": 3.0, "ops": 10.0},
        },
        {"perf": 1.0, "scale": 2.0, "ops": 1.0},
    )
    print(f"    winner = {cmp.winner}")
    for d in cmp.dimensions:
        print(f"    · {d.dimension}: {d.scores} → {d.winner}")
    print(f"    totals = {cmp.weighted_scores}")

    print("\n[5] Contradictions:")
    contra_facts = [
        eng.make_fact("postgres", "is_a", "database"),
        eng.make_fact("postgres", "is_a", "cache"),
        eng.make_fact("service", "has_status", "ready"),
        eng.make_fact("service", "has_status", "not ready"),
    ]
    contras = eng.find_contradictions(contra_facts)
    for rep_ in contras:
        print(f"    [{rep_.kind}] {rep_.reason}")

    print("\n[6] Dependency analysis:")
    dep_facts = [
        eng.make_fact("app", "depends_on", "api"),
        eng.make_fact("api", "depends_on", "db"),
        eng.make_fact("db", "depends_on", "storage"),
    ]
    d = eng.analyze_dependencies(dep_facts, "app")
    print(f"    reachable  = {d.reachable}")
    print(f"    max_depth  = {d.max_depth}")
    print(f"    topo order = {d.topological_order}")

    print("\n[7] Tradeoff analysis (Pareto + weighted):")
    tr = eng.analyze_tradeoffs(
        ["postgres", "sqlite"],
        {
            "postgres": {"perf": 8.0, "scale": 9.0, "ops": 5.0},
            "sqlite": {"perf": 7.0, "scale": 3.0, "ops": 10.0},
        },
        {"perf": 1.0, "scale": 2.0, "ops": 1.0},
    )
    print(f"    winner       = {tr.winner}")
    print(f"    pareto front = {tr.pareto_front}")
    for p in tr.pareto:
        mark = "dominated" if p.dominated else "pareto"
        print(f"    · {p.option}: {mark}  ({p.rationale})")

    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
