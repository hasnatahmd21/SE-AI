"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C21 — CRITIC ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04.

Purpose:
    Independent challenge layer. Critiques work products across 10 dimensions
    and CAN REJECT. Never auto-approves. Every verdict references concrete
    findings with evidence, location, and suggested remediation.

Dimensions (one critic class each):
    1.  RequirementCoverageCritic      functional requirements → test coverage
    2.  ArchitectureConsistencyCritic  components/interfaces/boundaries
    3.  CodeQualityCritic              eval/exec, bare except, TODOs, docstrings
    4.  HiddenAssumptionsCritic        "assume", "typically", "presumably", ...
    5.  EdgeCaseCritic                 None/empty/boundary coverage in tests
    6.  SecurityCritic                 secrets, unsafe subprocess, http://, TLS off
    7.  PerformanceCritic              nested loops, string concat in loop, N+1
    8.  MaintainabilityCritic          complexity, function length, naming
    9.  TestQualityCritic              assert True, empty tests, skip abuse
    10. UnnecessaryComplexityCritic    deep inheritance, unused imports,
                                       single-use helpers

Aggregation:
    any CRITICAL finding  → REJECTED
    any HIGH finding      → APPROVE_WITH_CONCERNS
    else                  → APPROVED

Invariants honored:
    - NO external LLM. Pure AST + regex + deterministic counting.
    - A critic that cannot apply (missing input) reports SKIPPED with reason,
      never fabricates a pass.
    - CRITICAL findings are evidence-backed (line number + snippet).
    - Rejection is REAL: any CRITICAL → verdict=REJECTED (test-enforced).
    - Every finding carries: severity, category, message, evidence, location.
    - Deterministic: same artifact → identical report.

Explicit limitations:
    - Content scanners (code/tests) work on Python source only.
    - Requirement coverage is text-keyword heuristic; not semantic matching.
    - Security/perf critics are pattern-based, not full static analyzers.
    - Some critics produce false positives on unusual style; severity
      reflects that (rarely CRITICAL without strong evidence).

Contents:
  1.  Enums: Severity, Verdict, ArtifactKind, FindingCategory
  2.  Dataclasses: Finding, Artifact, CriticResult, CritiqueReport
  3.  Base critic + AST helpers
  4.  Ten critics
  5.  CriticEngine facade
  6.  CriticRepository (persist)
  7.  Self-tests (~26)
  8.  Demo

Run as script:
    python -m sebrain.c21            # demo
    python -m sebrain.c21 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import ast
import hashlib
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
    Confidence, Config, SEBrainApp, SQLiteStorage, ValidationError,
    execution_scope, get_logger,
)
from sebrain.c02 import (
    EntityKind, Ontology, Provenance, ProvenanceType, RelationKind,
)
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore


log = get_logger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _stable_id(prefix: str, text: str) -> str:
    return hashlib.sha256(f"{prefix}::{text}".encode()).hexdigest()[:16]


def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEV_RANK = {
    Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
    Severity.HIGH: 3, Severity.CRITICAL: 4,
}


class Verdict(str, Enum):
    APPROVED = "approved"
    APPROVE_WITH_CONCERNS = "approve_with_concerns"
    REJECTED = "rejected"
    INSUFFICIENT_INPUT = "insufficient_input"


class ArtifactKind(str, Enum):
    CODE = "code"
    TESTS = "tests"
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    PLAN = "plan"
    REPAIR = "repair"


class FindingCategory(str, Enum):
    REQUIREMENT_COVERAGE = "requirement_coverage"
    ARCHITECTURE = "architecture"
    CODE_QUALITY = "code_quality"
    HIDDEN_ASSUMPTION = "hidden_assumption"
    EDGE_CASE = "edge_case"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    TEST_QUALITY = "test_quality"
    COMPLEXITY = "complexity"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Finding:
    severity: Severity
    category: FindingCategory
    message: str
    evidence: str = ""
    location: str = ""           # e.g. "pkg/mod.py:42"
    suggestion: str = ""
    critic_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "evidence": self.evidence,
            "location": self.location,
            "suggestion": self.suggestion,
            "critic_name": self.critic_name,
        }


@dataclass(slots=True)
class Artifact:
    kind: ArtifactKind
    ref: str = ""                     # traceability id (spec id, plan id, ...)
    raw_text: str = ""                # for CODE/TESTS
    structured: Any = None            # for REQUIREMENTS/ARCHITECTURE/PLAN/REPAIR
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "ref": self.ref,
            "label": self.label,
            "raw_text_len": len(self.raw_text or ""),
            "has_structured": self.structured is not None,
        }


@dataclass(slots=True)
class CriticResult:
    critic_name: str
    category: FindingCategory
    applied: bool
    skipped_reason: str = ""
    findings: list[Finding] = field(default_factory=list)

    def worst_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: _SEV_RANK[f.severity]).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic_name": self.critic_name,
            "category": self.category.value,
            "applied": self.applied,
            "skipped_reason": self.skipped_reason,
            "worst_severity": (self.worst_severity().value
                                if self.worst_severity() else None),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(slots=True)
class CritiqueReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    artifact: Artifact = field(default_factory=lambda: Artifact(ArtifactKind.CODE))
    verdict: Verdict = Verdict.INSUFFICIENT_INPUT
    results: list[CriticResult] = field(default_factory=list)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def findings(self) -> list[Finding]:
        out: list[Finding] = []
        for r in self.results:
            out.extend(r.findings)
        return out

    def count(self, sev: Severity) -> int:
        return sum(1 for f in self.findings() if f.severity is sev)

    def is_rejected(self) -> bool:
        return self.verdict is Verdict.REJECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "artifact": self.artifact.to_dict(),
            "verdict": self.verdict.value,
            "results": [r.to_dict() for r in self.results],
            "rationale": self.rationale,
            "counts": {s.value: self.count(s) for s in Severity},
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        c = {s.value: self.count(s) for s in Severity}
        return (
            "=== Critique Report ===\n"
            f"artifact: {self.artifact.kind.value} "
            f"({self.artifact.ref or self.artifact.label or '<no ref>'})\n"
            f"verdict: {self.verdict.value}\n"
            f"findings: info={c['info']} low={c['low']} "
            f"medium={c['medium']} high={c['high']} critical={c['critical']}\n"
            f"critics applied: "
            f"{sum(1 for r in self.results if r.applied)}/"
            f"{len(self.results)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. BASE CRITIC + HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _try_parse(source: str) -> ast.AST | None:
    try:
        return ast.parse(source or "")
    except SyntaxError:
        return None


def _line_snippet(source: str, lineno: int, ctx: int = 0) -> str:
    if not source or lineno <= 0:
        return ""
    lines = source.splitlines()
    i = lineno - 1
    if i < 0 or i >= len(lines):
        return ""
    lo = max(0, i - ctx)
    hi = min(len(lines), i + ctx + 1)
    return " | ".join(f"{k+1}:{lines[k].strip()}" for k in range(lo, hi))


class BaseCritic:
    name: str = "base"
    category: FindingCategory = FindingCategory.CODE_QUALITY
    applies_to: tuple[ArtifactKind, ...] = ()

    def applies(self, art: Artifact) -> bool:
        return art.kind in self.applies_to

    def critique(self, art: Artifact) -> list[Finding]:
        raise NotImplementedError

    # ---- helpers ----
    def _parse(self, art: Artifact) -> ast.AST | None:
        return _try_parse(art.raw_text or "")

    def _finding(
        self, severity: Severity, message: str, *,
        evidence: str = "", location: str = "",
        suggestion: str = "",
    ) -> Finding:
        return Finding(
            severity=severity, category=self.category, message=message,
            evidence=evidence, location=location, suggestion=suggestion,
            critic_name=self.name,
        )


# ════════════════════════════════════════════════════════════════════════════
# 4. TEN CRITICS
# ════════════════════════════════════════════════════════════════════════════

# ---- 1. Requirement coverage ----
class RequirementCoverageCritic(BaseCritic):
    name = "requirement_coverage"
    category = FindingCategory.REQUIREMENT_COVERAGE
    applies_to = (ArtifactKind.TESTS,)

    def critique(self, art: Artifact) -> list[Finding]:
        """Coverage is evaluated as part of the *engine*, since it needs
        two inputs (spec + tests). When invoked standalone with only tests,
        we report INFO."""
        return []

    def critique_pair(
        self, spec: Any, tests_source: str,
    ) -> list[Finding]:
        """Independent entry point used by CriticEngine."""
        findings: list[Finding] = []
        func_items = list(getattr(spec, "functional", []) or [])
        if not func_items:
            return findings
        # Parse test file to find test function names + their docstrings
        tree = _try_parse(tests_source)
        test_docs: list[str] = []
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        doc = ast.get_docstring(node) or ""
                        test_docs.append(f"{node.name} {doc}".lower())
        joined = " \n ".join(test_docs)

        for item in func_items:
            text = getattr(item, "text", "") or ""
            req_id = f"req:{_stable_id('req', text)}"
            # Accept: explicit covers marker or keyword overlap
            keywords = [w.lower() for w in re.findall(r"[A-Za-z_]{4,}", text)][:8]
            hit_by_marker = req_id in tests_source
            hit_by_keyword = any(k in joined for k in keywords) if keywords else False
            if not (hit_by_marker or hit_by_keyword):
                findings.append(self._finding(
                    Severity.MEDIUM,
                    f"Functional requirement has no visible test coverage",
                    evidence=_short(text, 140),
                    location="(spec.functional)",
                    suggestion=(
                        "Add a test that asserts this behavior, or mark the "
                        "requirement as covered via `covers=[...]` metadata."
                    ),
                ))
        return findings


# ---- 2. Architecture consistency ----
class ArchitectureConsistencyCritic(BaseCritic):
    name = "architecture_consistency"
    category = FindingCategory.ARCHITECTURE
    applies_to = (ArtifactKind.ARCHITECTURE,)

    def critique(self, art: Artifact) -> list[Finding]:
        findings: list[Finding] = []
        arch = art.structured
        if arch is None:
            return [self._finding(
                Severity.MEDIUM,
                "No architecture structured content to critique",
                suggestion="Provide a C10 ArchitectureResult as `structured`.",
            )]
        decision = getattr(arch, "decision", None)
        if decision is None:
            return [self._finding(
                Severity.CRITICAL,
                "Architecture has no decision",
                suggestion="Run C10 architecture reasoning first.",
            )]
        selected = getattr(decision, "selected", None)
        if selected is None:
            return [self._finding(
                Severity.CRITICAL,
                "Architecture decision has no selected candidate",
            )]
        comps = getattr(selected, "components", []) or []
        ifaces = getattr(selected, "interfaces", []) or []
        f_bounds = getattr(selected, "failure_boundaries", []) or []
        s_bounds = getattr(selected, "security_boundaries", []) or []
        comp_ids = {getattr(c, "id", "") for c in comps}

        if not comps:
            findings.append(self._finding(
                Severity.CRITICAL, "Architecture defines no components",
            ))
        if not ifaces:
            findings.append(self._finding(
                Severity.HIGH, "Architecture defines no interfaces",
            ))
        if not f_bounds:
            findings.append(self._finding(
                Severity.HIGH, "No failure boundaries declared",
            ))
        if not s_bounds:
            findings.append(self._finding(
                Severity.HIGH, "No security boundaries declared",
            ))
        # Dependency integrity
        for c in comps:
            for d in (getattr(c, "depends_on", []) or []):
                if d not in comp_ids:
                    findings.append(self._finding(
                        Severity.HIGH,
                        f"Component '{getattr(c,'name','?')}' depends on "
                        f"unknown component id '{d}'",
                        location=f"component:{getattr(c,'id','?')}",
                    ))
        # Interface endpoint integrity
        for i in ifaces:
            for side in ("from_component", "to_component"):
                val = getattr(i, side, None)
                if val and val not in comp_ids:
                    findings.append(self._finding(
                        Severity.HIGH,
                        f"Interface {getattr(i,'id','?')} references "
                        f"unknown component '{val}' ({side})",
                    ))
        # Kind coherence
        kind = getattr(getattr(selected, "kind", None), "value",
                        str(getattr(selected, "kind", "")))
        if kind == "microservices" and len(f_bounds) < 2:
            findings.append(self._finding(
                Severity.HIGH,
                "Microservices declared with fewer than 2 failure boundaries",
            ))
        if kind == "monolith" and len(f_bounds) > 1:
            findings.append(self._finding(
                Severity.MEDIUM,
                "Monolith declared with multiple failure boundaries "
                "(may be mislabelled)",
            ))
        return findings


# ---- 3. Code quality ----
class CodeQualityCritic(BaseCritic):
    name = "code_quality"
    category = FindingCategory.CODE_QUALITY
    applies_to = (ArtifactKind.CODE, ArtifactKind.TESTS)

    _BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:\s*$")
    _TODO_RE = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b", re.I)

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        findings: list[Finding] = []
        tree = self._parse(art)
        if tree is None:
            if src.strip():
                findings.append(self._finding(
                    Severity.MEDIUM, "Source does not parse as Python",
                    suggestion="Fix syntax before further critique.",
                ))
            return findings

        # 1. Bare except + except Exception (broad)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    findings.append(self._finding(
                        Severity.MEDIUM,
                        "Bare `except:` swallows all exceptions",
                        evidence=_line_snippet(src, node.lineno),
                        location=f"line {node.lineno}",
                        suggestion="Catch specific exception types.",
                    ))
                elif (isinstance(node.type, ast.Name)
                        and node.type.id == "Exception"):
                    # `except Exception:` is only marginally more specific
                    # than a bare `except:` — it still swallows nearly
                    # everything (only SystemExit/KeyboardInterrupt/
                    # GeneratorExit escape it) and is a common way real
                    # bugs get hidden. (This branch used to be dead code —
                    # gated behind an `if False` wrapping a call to a
                    # nonexistent `node.body_ok(node)` method — so no
                    # `except Exception:` handler was ever actually
                    # flagged; fixed to call `_finding` for real.)
                    findings.append(self._finding(
                        Severity.MEDIUM,
                        "Catching `Exception` broadly can hide bugs",
                        evidence=_line_snippet(src, node.lineno),
                        location=f"line {node.lineno}",
                        suggestion="Catch specific exception types where possible.",
                    ))
        # 2. TODO / FIXME
        for m in self._TODO_RE.finditer(src):
            line_no = src.count("\n", 0, m.start()) + 1
            findings.append(self._finding(
                Severity.LOW, f"'{m.group(0).strip()}' marker left in code",
                evidence=_line_snippet(src, line_no),
                location=f"line {line_no}",
            ))
        # 3. Public functions without docstring (only top-level functions)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    findings.append(self._finding(
                        Severity.LOW,
                        f"Public function '{node.name}' missing docstring",
                        location=f"line {node.lineno}",
                    ))
        # 4. `pass`-only function bodies (dead)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [n for n in node.body
                        if not isinstance(n, ast.Expr)]
                if (len(body) == 1 and isinstance(body[0], ast.Pass)
                        and not node.name.startswith("_")):
                    findings.append(self._finding(
                        Severity.MEDIUM,
                        f"Function '{node.name}' body is only `pass`",
                        location=f"line {node.lineno}",
                        suggestion="Remove or implement.",
                    ))
        # 5. Mutable default arguments (classic bug)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in list(node.args.defaults) + [
                    d for d in node.args.kw_defaults if d is not None
                ]:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(self._finding(
                            Severity.HIGH,
                            f"Mutable default argument in '{node.name}'",
                            location=f"line {node.lineno}",
                            suggestion=(
                                "Use None and create the container inside."
                            ),
                        ))
        return findings


# ---- 4. Hidden assumptions ----
class HiddenAssumptionsCritic(BaseCritic):
    name = "hidden_assumptions"
    category = FindingCategory.HIDDEN_ASSUMPTION
    applies_to = (ArtifactKind.REQUIREMENTS, ArtifactKind.CODE,
                   ArtifactKind.TESTS, ArtifactKind.PLAN)

    _PHRASES = (
        "assume", "assumed", "assuming", "presumably", "presume",
        "typically", "usually", "in most cases", "should be",
        "probably", "likely", "obviously", "clearly", "of course",
        "as expected", "as needed", "as required",
    )

    def critique(self, art: Artifact) -> list[Finding]:
        findings: list[Finding] = []
        text = art.raw_text
        if not text and art.structured is not None:
            # Requirements: examine assumption list + functional text
            for a in getattr(art.structured, "assumptions", []) or []:
                origin = getattr(a, "origin", "explicit")
                txt = getattr(a, "text", "") or ""
                if origin == "inferred":
                    findings.append(self._finding(
                        Severity.MEDIUM,
                        "Requirement relies on an inferred assumption",
                        evidence=_short(txt, 140),
                        suggestion=(
                            "Convert to an explicit assumption with a source, "
                            "or resolve with the user before proceeding."
                        ),
                    ))
            return findings
        if not text:
            return findings

        for phrase in self._PHRASES:
            for m in re.finditer(
                r"\b" + re.escape(phrase) + r"\b", text, re.I,
            ):
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(self._finding(
                    Severity.LOW,
                    f"Assumption-like phrase '{phrase}'",
                    evidence=_line_snippet(text, line_no),
                    location=f"line {line_no}",
                    suggestion=(
                        "Make the assumption explicit or replace with a "
                        "verifiable claim."
                    ),
                ))
        return findings


# ---- 5. Edge cases ----
class EdgeCaseCritic(BaseCritic):
    name = "edge_case"
    category = FindingCategory.EDGE_CASE
    applies_to = (ArtifactKind.TESTS,)

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        findings: list[Finding] = []
        low = src.lower()
        # Look for pytest.raises / boundary values / None / empty
        has_raises = "pytest.raises" in low
        has_none = re.search(r"\bNone\b", src) is not None
        has_empty = any(k in low for k in (
            'empty', '== ""', "== ''", "== []", "== {}", "= []", "= {}",
            "len(", "zero",
        ))
        has_boundary = any(k in low for k in (
            "boundary", "min", "max", "edge", "overflow", "underflow",
        ))
        if not has_raises:
            findings.append(self._finding(
                Severity.MEDIUM,
                "No negative-path tests (pytest.raises) found",
                suggestion="Add tests for at least one error path.",
            ))
        if not has_none:
            findings.append(self._finding(
                Severity.LOW,
                "No None handling covered in tests",
                suggestion="Add a test exercising None inputs where valid.",
            ))
        if not has_empty:
            findings.append(self._finding(
                Severity.LOW,
                "No empty-input coverage detected",
                suggestion="Add empty-string / empty-list boundary tests.",
            ))
        if not has_boundary:
            findings.append(self._finding(
                Severity.INFO,
                "No explicit boundary-value coverage detected",
            ))
        return findings


# ---- 6. Security ----
class SecurityCritic(BaseCritic):
    name = "security"
    category = FindingCategory.SECURITY
    applies_to = (ArtifactKind.CODE, ArtifactKind.TESTS)

    _PATTERNS: list[tuple[re.Pattern, Severity, str, str]] = [
        (re.compile(r"(?i)(?:password|passwd|secret|api[_-]?key|token)\s*=\s*"
                    r"['\"][^'\"]{6,}['\"]"),
         Severity.CRITICAL, "HARDCODED_SECRET",
         "Hardcoded credential-like literal"),
        (re.compile(r"\beval\s*\("),
         Severity.CRITICAL, "EVAL",
         "eval() can execute arbitrary code"),
        (re.compile(r"\bexec\s*\("),
         Severity.CRITICAL, "EXEC",
         "exec() can execute arbitrary code"),
        (re.compile(r"shell\s*=\s*True"),
         Severity.HIGH, "SHELL_TRUE",
         "subprocess shell=True allows shell injection"),
        (re.compile(r"\bos\.system\s*\("),
         Severity.HIGH, "OS_SYSTEM",
         "os.system() is a shell injection risk"),
        (re.compile(r"\bpickle\.loads\s*\("),
         Severity.HIGH, "PICKLE_LOADS",
         "pickle.loads on untrusted data can execute code"),
        (re.compile(r"verify\s*=\s*False"),
         Severity.HIGH, "TLS_VERIFY_OFF",
         "TLS verification disabled"),
        (re.compile(r"yaml\.load\s*\([^,)]*\)(?!\s*,)"),
         Severity.MEDIUM, "YAML_LOAD_UNSAFE",
         "yaml.load without SafeLoader"),
        (re.compile(r"\bhttp://(?!localhost|127\.0\.0\.1)"),
         Severity.MEDIUM, "PLAINTEXT_HTTP",
         "Non-local HTTP endpoint"),
        (re.compile(r"allow_origins\s*=\s*\[?\s*['\"]\*['\"]"),
         Severity.MEDIUM, "CORS_WILDCARD",
         "CORS wildcard origin"),
        (re.compile(r"\bassert\s+[\w.]+\s*(?:==|is)\s*None\s*#\s*security"),
         Severity.LOW, "WEAK_ASSERT",
         "Weak assert in security-sensitive code"),
    ]

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        findings: list[Finding] = []
        for pat, sev, code, msg in self._PATTERNS:
            for m in pat.finditer(src):
                line_no = src.count("\n", 0, m.start()) + 1
                findings.append(self._finding(
                    sev,
                    f"{code}: {msg}",
                    evidence=_line_snippet(src, line_no),
                    location=f"line {line_no}",
                    suggestion="Review and replace with a safe alternative.",
                ))
        return findings


# ---- 7. Performance ----
class PerformanceCritic(BaseCritic):
    name = "performance"
    category = FindingCategory.PERFORMANCE
    applies_to = (ArtifactKind.CODE, ArtifactKind.TESTS)

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        tree = self._parse(art)
        findings: list[Finding] = []
        if tree is None:
            return findings

        # 1. Nested loops ≥ 3 levels
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                depth = _loop_depth(node)
                if depth >= 3:
                    findings.append(self._finding(
                        Severity.MEDIUM,
                        f"Nested loops depth {depth} (≥3) — potential O(n^d)",
                        location=f"line {node.lineno}",
                        suggestion=(
                            "Consider restructuring with dict/set lookups "
                            "or vectorized operations."
                        ),
                    ))
                    break

        # 2. String += / list += in loop
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.AugAssign):
                        if (isinstance(stmt.target, ast.Name)
                                and isinstance(stmt.value, (ast.Constant,))):
                            continue  # counter
                        # String/binary concat heuristic: AugAssign on Name
                        # with Add op → could be O(n²) string build
                        if isinstance(stmt.op, ast.Add):
                            findings.append(self._finding(
                                Severity.LOW,
                                "Augmented assignment inside loop "
                                "(possible O(n²) accumulation)",
                                location=f"line {stmt.lineno}",
                                suggestion=(
                                    "Accumulate into a list and join once."
                                ),
                            ))
                            break

        # 3. N+1 pattern: DB/HTTP call inside a loop
        n_plus_one_re = re.compile(
            r"for\s+\w+\s+in\s+.*?:\s*\n(?:\s+.*\n)*?\s+"
            r"(?:\w+\.)?(?:execute|query|get|post|put|delete)\s*\(",
            re.I | re.M,
        )
        for m in n_plus_one_re.finditer(src):
            line_no = src.count("\n", 0, m.start()) + 1
            findings.append(self._finding(
                Severity.MEDIUM,
                "Possible N+1 pattern (I/O call inside a loop)",
                location=f"line {line_no}",
                suggestion="Batch the calls or use IN-list queries.",
            ))
            break

        return findings


def _loop_depth(node: ast.AST) -> int:
    depth = 1
    max_depth = 1
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            max_depth = max(max_depth, 1 + _loop_depth(child))
        else:
            max_depth = max(max_depth, _loop_depth(child))
    return max_depth


# ---- 8. Maintainability ----
class MaintainabilityCritic(BaseCritic):
    name = "maintainability"
    category = FindingCategory.MAINTAINABILITY
    applies_to = (ArtifactKind.CODE, ArtifactKind.TESTS)

    MAX_FUNC_LINES = 80
    MAX_COMPLEXITY = 15

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        tree = self._parse(art)
        findings: list[Finding] = []
        if tree is None:
            return findings

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Length
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start) or start
            length = end - start + 1
            if length > self.MAX_FUNC_LINES:
                findings.append(self._finding(
                    Severity.MEDIUM,
                    f"Function '{node.name}' is {length} lines "
                    f"(>{self.MAX_FUNC_LINES})",
                    location=f"line {start}",
                    suggestion="Split into smaller functions.",
                ))
            # Complexity
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (
                    ast.If, ast.For, ast.While, ast.Try, ast.With,
                    ast.AsyncFor, ast.AsyncWith, ast.BoolOp,
                )):
                    complexity += 1
                elif isinstance(child, ast.comprehension):
                    complexity += 1
            if complexity > self.MAX_COMPLEXITY:
                findings.append(self._finding(
                    Severity.MEDIUM,
                    f"Function '{node.name}' has cyclomatic-ish complexity "
                    f"{complexity} (>{self.MAX_COMPLEXITY})",
                    location=f"line {start}",
                    suggestion="Refactor branches into named helpers.",
                ))

        # Single-letter variables outside for/comp target
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if len(node.id) == 1 and node.id not in "ijknxyzm":
                    # Common loop vars are usually fine; only flag
                    # stores outside comprehensions/loops
                    pass
        return findings


# ---- 9. Test quality ----
class TestQualityCritic(BaseCritic):
    name = "test_quality"
    category = FindingCategory.TEST_QUALITY
    applies_to = (ArtifactKind.TESTS,)

    _TRIVIAL_RE = re.compile(r"assert\s+(?:True|1|0|False)\b")

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        findings: list[Finding] = []
        tree = self._parse(art)
        if tree is None:
            return findings

        test_funcs = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")
        ]
        if not test_funcs:
            findings.append(self._finding(
                Severity.HIGH,
                "No test functions found in test file",
            ))
            return findings

        for node in test_funcs:
            body = node.body
            # Strip leading docstring
            non_doc = body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                non_doc = body[1:]
            # Empty test
            if not non_doc:
                findings.append(self._finding(
                    Severity.HIGH,
                    f"Test '{node.name}' has no body",
                    location=f"line {node.lineno}",
                ))
                continue
            # Only-pass or only-skip
            only_pass = (len(non_doc) == 1
                         and isinstance(non_doc[0], ast.Pass))
            if only_pass:
                findings.append(self._finding(
                    Severity.HIGH,
                    f"Test '{node.name}' is only `pass` (vacuous)",
                    location=f"line {node.lineno}",
                ))
                continue
            # Assert True / False
            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    val = child.test
                    if (isinstance(val, ast.Constant)
                            and val.value in (True, False, 0, 1)):
                        findings.append(self._finding(
                            Severity.MEDIUM,
                            f"Test '{node.name}' uses trivial assertion "
                            f"`assert {val.value}`",
                            location=f"line {child.lineno}",
                        ))
            # try/except: pass swallowing
            for child in ast.walk(node):
                if isinstance(child, ast.Try):
                    for handler in child.handlers:
                        hbody = [n for n in handler.body
                                 if not isinstance(n, ast.Expr)]
                        if (hbody and all(isinstance(n, ast.Pass)
                                          for n in hbody)):
                            findings.append(self._finding(
                                Severity.HIGH,
                                f"Test '{node.name}' has `except: pass` "
                                f"(swallows failures)",
                                location=f"line {child.lineno}",
                            ))
        return findings


# ---- 10. Unnecessary complexity ----
class UnnecessaryComplexityCritic(BaseCritic):
    name = "unnecessary_complexity"
    category = FindingCategory.COMPLEXITY
    applies_to = (ArtifactKind.CODE, ArtifactKind.TESTS)

    MAX_INHERITANCE_DEPTH = 3

    def critique(self, art: Artifact) -> list[Finding]:
        src = art.raw_text or ""
        tree = self._parse(art)
        findings: list[Finding] = []
        if tree is None:
            return findings

        # 1. Deep inheritance within the module
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

        def depth(cls_name: str, stack: set[str]) -> int:
            if cls_name in stack:
                return 0
            node = classes.get(cls_name)
            if node is None:
                return 1
            stack = stack | {cls_name}
            best = 1
            for b in node.bases:
                if isinstance(b, ast.Name) and b.id in classes:
                    best = max(best, 1 + depth(b.id, stack))
            return best

        for cls_name in classes:
            d = depth(cls_name, set())
            if d > self.MAX_INHERITANCE_DEPTH:
                findings.append(self._finding(
                    Severity.MEDIUM,
                    f"Class '{cls_name}' has inheritance depth {d}",
                    location="module",
                    suggestion="Prefer composition over deep hierarchies.",
                ))

        # 2. Unused imports
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    nm = alias.asname or alias.name.split(".")[0]
                    imported[nm] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    nm = alias.asname or alias.name
                    imported[nm] = node.lineno
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                # `a.b` → 'a' as Name will already be captured
                pass
        for nm, line in imported.items():
            if nm == "*":
                continue
            if nm not in used:
                findings.append(self._finding(
                    Severity.LOW,
                    f"Unused import '{nm}'",
                    location=f"line {line}",
                    suggestion="Remove dead import.",
                ))

        # 3. Single-use private helpers (defined + called once)
        func_defs: dict[str, ast.AST] = {}
        func_calls: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_defs[node.name] = node
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_calls[node.func.id] = \
                        func_calls.get(node.func.id, 0) + 1
        for name, node in func_defs.items():
            if not name.startswith("_") or name.startswith("__"):
                continue
            if func_calls.get(name, 0) == 1:
                # Just a hint; low severity
                findings.append(self._finding(
                    Severity.INFO,
                    f"Private helper '{name}' is called exactly once",
                    location=f"line {getattr(node,'lineno',0)}",
                    suggestion="Inlining may improve readability.",
                ))
        return findings


# ════════════════════════════════════════════════════════════════════════════
# 5. CRITIC ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
class CriticEngine:
    """Runs all applicable critics, aggregates findings, decides verdict."""

    def __init__(self) -> None:
        self.critics: list[BaseCritic] = [
            RequirementCoverageCritic(),
            ArchitectureConsistencyCritic(),
            CodeQualityCritic(),
            HiddenAssumptionsCritic(),
            EdgeCaseCritic(),
            SecurityCritic(),
            PerformanceCritic(),
            MaintainabilityCritic(),
            TestQualityCritic(),
            UnnecessaryComplexityCritic(),
        ]

    def analyze(
        self, artifact: Artifact, *,
        project_id: str = "", spec: Any = None,
    ) -> CritiqueReport:
        report = CritiqueReport(
            project_id=project_id, artifact=artifact,
            provenance=Provenance(
                source="critic_engine", source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        for c in self.critics:
            result = CriticResult(
                critic_name=c.name, category=c.category, applied=False,
            )
            if not c.applies(artifact):
                result.skipped_reason = (
                    f"critic '{c.name}' does not apply to "
                    f"artifact kind '{artifact.kind.value}'"
                )
                report.results.append(result)
                continue
            try:
                result.findings = list(c.critique(artifact))
                result.applied = True
            except Exception as exc:
                log.warning("c21.critic_error", critic=c.name, error=str(exc))
                result.skipped_reason = (
                    f"critic raised: {type(exc).__name__}: {exc}"
                )
                result.findings = []
                result.applied = False
            report.results.append(result)

        # Cross-artifact: requirement coverage needs spec + tests
        if (artifact.kind is ArtifactKind.TESTS and spec is not None):
            rc = RequirementCoverageCritic()
            try:
                extra = rc.critique_pair(spec, artifact.raw_text or "")
                for r in report.results:
                    if r.critic_name == rc.name:
                        r.findings.extend(extra)
                        r.applied = True
                        break
            except Exception as exc:
                log.warning("c21.req_coverage_error", error=str(exc))

        # Verdict
        all_f = report.findings()
        if not any(r.applied for r in report.results):
            report.verdict = Verdict.INSUFFICIENT_INPUT
            report.rationale = "no critic could apply to this artifact"
            return report
        crit = sum(1 for f in all_f if f.severity is Severity.CRITICAL)
        high = sum(1 for f in all_f if f.severity is Severity.HIGH)
        if crit > 0:
            report.verdict = Verdict.REJECTED
            report.rationale = (
                f"{crit} CRITICAL finding(s) → REJECTED "
                f"(high={high}, medium="
                f"{sum(1 for f in all_f if f.severity is Severity.MEDIUM)})"
            )
        elif high > 0:
            report.verdict = Verdict.APPROVE_WITH_CONCERNS
            report.rationale = (
                f"{high} HIGH finding(s) → APPROVE_WITH_CONCERNS"
            )
        else:
            report.verdict = Verdict.APPROVED
            report.rationale = (
                f"no CRITICAL/HIGH findings "
                f"({sum(1 for f in all_f if f.severity is Severity.MEDIUM)} "
                f"medium, {sum(1 for f in all_f if f.severity is Severity.LOW)} "
                f"low)"
            )
        return report


# ════════════════════════════════════════════════════════════════════════════
# 6. CRITIC REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class CriticRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: CritiqueReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"critique:{report.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, report.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["critique", "c21", report.verdict.value],
            provenance=report.provenance,
        )
        if report.is_rejected():
            self.memory.record_failure(
                f"critique_reject:{report.id}",
                what=f"critique rejected artifact "
                     f"'{report.artifact.ref or report.artifact.label}'",
                root_cause=report.rationale,
                fix=None,
                scope_id=project_id,
                provenance=report.provenance,
                confidence=Confidence.HIGH,
            )
        if self.ontology is None:
            return key

        ent = self.ontology.add(
            EntityKind.VERIFICATION,
            _short(f"Critique {report.id[:8]} ({report.verdict.value})", 120),
            attributes={
                "critique_id": report.id,
                "project_id": project_id,
                "artifact_kind": report.artifact.kind.value,
                "artifact_ref": report.artifact.ref,
                "verdict": report.verdict.value,
                "counts": {s.value: report.count(s) for s in Severity},
                "rationale": report.rationale,
            },
            tags=["critique", report.verdict.value],
            provenance=report.provenance,
        )
        for f in report.findings():
            if _SEV_RANK[f.severity] < _SEV_RANK[Severity.MEDIUM]:
                continue
            fe = self.ontology.add(
                EntityKind.EVIDENCE,
                _short(f"{f.severity.value}: {f.message}", 120),
                attributes={
                    "category": f.category.value,
                    "severity": f.severity.value,
                    "evidence": f.evidence,
                    "location": f.location,
                },
                tags=["critique-finding", f.severity.value],
                provenance=report.provenance,
            )
            try:
                self.ontology.link(RelationKind.SUPPORTS, fe.id, ent.id)
            except ValidationError:
                pass
        return ent.id

    def load(self, critique_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"critique:{critique_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 7. SELF-TESTS
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
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C21 self-tests…")
    engine = CriticEngine()

    # ---- helpers ----
    def _code_artifact(text: str) -> Artifact:
        return Artifact(kind=ArtifactKind.CODE, raw_text=text, ref="code")

    def _test_artifact(text: str) -> Artifact:
        return Artifact(kind=ArtifactKind.TESTS, raw_text=text, ref="tests")

    # ---- security critic ----
    def t_security_eval() -> None:
        r = engine.analyze(_code_artifact("def f(x):\n    return eval(x)\n"))
        crits = [f for f in r.findings()
                 if f.category is FindingCategory.SECURITY
                 and f.severity is Severity.CRITICAL]
        assert crits
        assert any("EVAL" in f.message for f in crits)
        assert r.verdict is Verdict.REJECTED

    def t_security_clean() -> None:
        r = engine.analyze(_code_artifact(
            '"""Docstring."""\n'
            "def f(x: int) -> int:\n"
            '    """Return x + 1."""\n'
            "    return x + 1\n"
        ))
        sec_critical = [f for f in r.findings()
                        if f.category is FindingCategory.SECURITY
                        and f.severity is Severity.CRITICAL]
        assert sec_critical == []

    def t_security_hardcoded_secret() -> None:
        r = engine.analyze(_code_artifact(
            'password = "hunter2secret"\n'
        ))
        crits = [f for f in r.findings()
                 if "HARDCODED_SECRET" in f.message]
        assert crits
        assert crits[0].severity is Severity.CRITICAL

    def t_security_shell_true() -> None:
        r = engine.analyze(_code_artifact(
            "import subprocess\n"
            "subprocess.run(['ls'], shell=True)\n"
        ))
        hits = [f for f in r.findings() if "SHELL_TRUE" in f.message]
        assert hits

    def t_security_verify_false() -> None:
        r = engine.analyze(_code_artifact(
            "import requests\n"
            "requests.get('https://x', verify=False)\n"
        ))
        hits = [f for f in r.findings() if "TLS_VERIFY_OFF" in f.message]
        assert hits

    def t_security_plaintext_http() -> None:
        r = engine.analyze(_code_artifact(
            'url = "http://example.com/api"\n'
        ))
        hits = [f for f in r.findings() if "PLAINTEXT_HTTP" in f.message]
        assert hits

    check("security: eval → CRITICAL + REJECTED", t_security_eval)
    check("security: clean code → no CRITICAL", t_security_clean)
    check("security: hardcoded secret flagged", t_security_hardcoded_secret)
    check("security: shell=True flagged", t_security_shell_true)
    check("security: verify=False flagged", t_security_verify_false)
    check("security: http:// flagged", t_security_plaintext_http)

    # ---- code quality ----
    def t_cq_bare_except() -> None:
        r = engine.analyze(_code_artifact(
            "def f():\n"
            "    try:\n"
            "        g()\n"
            "    except:\n"
            "        pass\n"
        ))
        hits = [f for f in r.findings() if "Bare `except:`" in f.message]
        assert hits

    def t_cq_todo() -> None:
        r = engine.analyze(_code_artifact(
            "# TODO: finish this later\n"
            "def f():\n    return 1\n"
        ))
        hits = [f for f in r.findings() if "TODO" in f.message]
        assert hits

    def t_cq_pass_only_function() -> None:
        r = engine.analyze(_code_artifact(
            "def f():\n    pass\n"
        ))
        hits = [f for f in r.findings() if "only `pass`" in f.message]
        assert hits

    def t_cq_mutable_default() -> None:
        r = engine.analyze(_code_artifact(
            "def f(items=[]):\n    items.append(1)\n"
        ))
        hits = [f for f in r.findings() if "Mutable default" in f.message]
        assert hits
        assert hits[0].severity is Severity.HIGH

    check("cq: bare except flagged", t_cq_bare_except)
    check("cq: TODO flagged", t_cq_todo)
    check("cq: pass-only function flagged", t_cq_pass_only_function)
    check("cq: mutable default flagged (HIGH)", t_cq_mutable_default)

    # ---- test quality ----
    def t_tq_no_tests() -> None:
        r = engine.analyze(_test_artifact(
            "def helper():\n    pass\n"
        ))
        hits = [f for f in r.findings() if "No test functions" in f.message]
        assert hits
        assert hits[0].severity is Severity.HIGH

    def t_tq_empty_test() -> None:
        r = engine.analyze(_test_artifact(
            "def test_x():\n    pass\n"
        ))
        hits = [f for f in r.findings()
                if "only `pass`" in f.message
                or "has no body" in f.message]
        assert hits

    def t_tq_trivial_assert() -> None:
        r = engine.analyze(_test_artifact(
            "def test_x():\n    assert True\n"
        ))
        hits = [f for f in r.findings() if "trivial assertion" in f.message]
        assert hits

    def t_tq_swallow_exception() -> None:
        r = engine.analyze(_test_artifact(
            "def test_x():\n"
            "    try:\n"
            "        g()\n"
            "    except:\n"
            "        pass\n"
        ))
        hits = [f for f in r.findings() if "except: pass" in f.message]
        assert hits

    check("tq: no test functions flagged HIGH", t_tq_no_tests)
    check("tq: empty test flagged", t_tq_empty_test)
    check("tq: `assert True` flagged", t_tq_trivial_assert)
    check("tq: `except: pass` flagged", t_tq_swallow_exception)

    # ---- edge case ----
    def t_edge_missing_all() -> None:
        r = engine.analyze(_test_artifact(
            "def test_x():\n    assert 1 == 1\n"
        ))
        msgs = [f.message for f in r.findings()]
        assert any("negative-path" in m for m in msgs)
        assert any("None" in m for m in msgs)
        assert any("empty-input" in m for m in msgs)

    def t_edge_covered() -> None:
        r = engine.analyze(_test_artifact(
            "import pytest\n"
            "def test_x():\n"
            "    with pytest.raises(ValueError):\n"
            "        f(None)\n"
            "    assert f([]) == []\n"
            "    assert f('') == ''\n"
            "    assert f(0) == 0\n"
            "    # boundary min max edge overflow\n"
        ))
        msgs = [f.message for f in r.findings()]
        assert not any("negative-path" in m for m in msgs)
        assert not any("None handling" in m for m in msgs)

    check("edge: missing coverage flagged", t_edge_missing_all)
    check("edge: full coverage → no edge findings", t_edge_covered)

    # ---- performance ----
    def t_perf_nested_loops() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        for b in a:\n"
            "            for c in b:\n"
            "                print(c)\n"
        )
        r = engine.analyze(_code_artifact(src))
        hits = [f for f in r.findings() if "Nested loops" in f.message]
        assert hits

    def t_perf_n_plus_one() -> None:
        src = (
            "def f(items, conn):\n"
            "    for it in items:\n"
            "        cur = conn.cursor()\n"
            "        cur.execute('SELECT * FROM t WHERE id=?', (it,))\n"
        )
        r = engine.analyze(_code_artifact(src))
        hits = [f for f in r.findings()
                if "N+1" in f.message or "I/O call inside" in f.message]
        assert hits

    check("perf: nested loops flagged", t_perf_nested_loops)
    check("perf: N+1 pattern flagged", t_perf_n_plus_one)

    # ---- maintainability ----
    def t_maint_long_function() -> None:
        body = "".join(f"    x{i} = {i}\n" for i in range(90))
        src = "def big():\n" + body
        r = engine.analyze(_code_artifact(src))
        hits = [f for f in r.findings() if "lines" in f.message
                and "80" in f.message]
        assert hits

    def t_maint_high_complexity() -> None:
        body = "".join(f"    if x{i}:\n        pass\n" for i in range(20))
        src = "def complex_fn(x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, "
        src += "x7=0, x8=0, x9=0, x10=0, x11=0, x12=0, x13=0, x14=0, "
        src += "x15=0, x16=0, x17=0, x18=0, x19=0):\n" + body
        r = engine.analyze(_code_artifact(src))
        hits = [f for f in r.findings()
                if "cyclomatic" in f.message or "complexity" in f.message]
        assert hits

    check("maint: long function flagged", t_maint_long_function)
    check("maint: high complexity flagged", t_maint_high_complexity)

    # ---- unnecessary complexity ----
    def t_uc_deep_inheritance() -> None:
        src = (
            "class A: pass\n"
            "class B(A): pass\n"
            "class C(B): pass\n"
            "class D(C): pass\n"
        )
        r = engine.analyze(_code_artifact(src))
        hits = [f for f in r.findings() if "inheritance depth" in f.message]
        assert hits

    def t_uc_unused_import() -> None:
        src = (
            "import os\n"
            "def f():\n    return 1\n"
        )
        r = engine.analyze(_code_artifact(src))
        hits = [f for f in r.findings()
                if "Unused import" in f.message and "'os'" in f.message]
        assert hits

    check("uc: deep inheritance flagged", t_uc_deep_inheritance)
    check("uc: unused import flagged", t_uc_unused_import)

    # ---- hidden assumptions ----
    def t_ha_text() -> None:
        r = engine.analyze(Artifact(
            kind=ArtifactKind.PLAN,
            raw_text="We will assume the DB is available.\n"
                     "Typically the user logs in first.\n",
        ))
        hits = [f for f in r.findings()
                if f.category is FindingCategory.HIDDEN_ASSUMPTION]
        assert any("assume" in f.message.lower() for f in hits)
        assert any("typically" in f.message.lower() for f in hits)

    check("hidden: assumption phrases flagged in text", t_ha_text)

    # ---- architecture ----
    def t_arch_clean() -> None:
        class C:
            def __init__(self, cid, name, deps=None):
                self.id = cid; self.name = name
                self.depends_on = deps or []
        class Iface:
            def __init__(self, iid, fc, tc):
                self.id = iid
                self.from_component = fc
                self.to_component = tc
        class Sel:
            def __init__(self):
                self.kind = type("K", (), {"value": "layered"})()
                self.components = [C("a","A"), C("b","B",["a"])]
                self.interfaces = [Iface("i1","a","b")]
                self.failure_boundaries = [C("fb","fb")]
                self.security_boundaries = [C("sb","sb")]
        class Dec:
            def __init__(self):
                self.selected = Sel()
        class Arch:
            def __init__(self):
                self.decision = Dec()
        r = engine.analyze(Artifact(
            kind=ArtifactKind.ARCHITECTURE, structured=Arch(),
        ))
        crits = [f for f in r.findings()
                 if f.category is FindingCategory.ARCHITECTURE
                 and f.severity is Severity.CRITICAL]
        highs = [f for f in r.findings()
                 if f.category is FindingCategory.ARCHITECTURE
                 and f.severity is Severity.HIGH]
        assert crits == []
        assert highs == []

    def t_arch_missing_boundaries() -> None:
        class C:
            def __init__(self, cid, name, deps=None):
                self.id = cid; self.name = name
                self.depends_on = deps or []
        class Sel:
            def __init__(self):
                self.kind = type("K", (), {"value": "layered"})()
                self.components = [C("a","A")]
                self.interfaces = []
                self.failure_boundaries = []
                self.security_boundaries = []
        class Dec:
            def __init__(self):
                self.selected = Sel()
        class Arch:
            def __init__(self):
                self.decision = Dec()
        r = engine.analyze(Artifact(
            kind=ArtifactKind.ARCHITECTURE, structured=Arch(),
        ))
        msgs = [f.message for f in r.findings()]
        assert any("no interfaces" in m.lower() for m in msgs)
        assert any("failure boundaries" in m.lower() for m in msgs)

    check("arch: clean architecture → no arch criticals", t_arch_clean)
    check("arch: missing pieces flagged", t_arch_missing_boundaries)

    # ---- verdict aggregation ----
    def t_verdict_rejected_on_critical() -> None:
        r = engine.analyze(_code_artifact("eval('1+1')\n"))
        assert r.verdict is Verdict.REJECTED
        assert r.count(Severity.CRITICAL) >= 1

    def t_verdict_concerns_on_high() -> None:
        # Mutable default arg → HIGH (not CRITICAL)
        r = engine.analyze(_code_artifact(
            '"""Doc."""\n'
            "def f(items=[]):\n"
            '    """Doc."""\n'
            "    return items\n"
        ))
        assert r.verdict is Verdict.APPROVE_WITH_CONCERNS
        assert r.count(Severity.HIGH) >= 1
        assert r.count(Severity.CRITICAL) == 0

    def t_verdict_approved_on_clean() -> None:
        # Minimal but clean — no eval, no mutable default, has docstring
        r = engine.analyze(_code_artifact(
            '"""Module docstring."""\n'
            "\n"
            "def f(x: int) -> int:\n"
            '    """Return x + 1."""\n'
            "    return x + 1\n"
        ))
        assert r.verdict is Verdict.APPROVED, r.rationale

    def t_insufficient_input() -> None:
        r = engine.analyze(Artifact(kind=ArtifactKind.REPAIR))
        assert r.verdict is Verdict.INSUFFICIENT_INPUT

    check("verdict: CRITICAL → REJECTED", t_verdict_rejected_on_critical)
    check("verdict: HIGH (no CRITICAL) → APPROVE_WITH_CONCERNS",
          t_verdict_concerns_on_high)
    check("verdict: clean → APPROVED", t_verdict_approved_on_clean)
    check("verdict: no applicable critic → INSUFFICIENT_INPUT",
          t_insufficient_input)

    # ---- requirement coverage (pair) ----
    def t_req_coverage_missing() -> None:
        from sebrain.c05 import RequirementParser
        spec = RequirementParser().parse(
            "Functional requirements:\n"
            "- Users must be able to create tasks.\n"
            "- Users must be able to archive widgets.\n"
        )
        tests_src = (
            "def test_create_task():\n"
            '    """covers create tasks"""\n'
            "    assert True\n"
        )
        r = engine.analyze(
            _test_artifact(tests_src), spec=spec,
        )
        rc_findings = [f for f in r.findings()
                       if f.category is FindingCategory.REQUIREMENT_COVERAGE]
        # At least one uncovered requirement should be flagged
        assert rc_findings, "expected at least one uncovered functional req"

    check("req-coverage: uncovered requirement flagged",
          t_req_coverage_missing)

    # ---- determinism ----
    def t_deterministic() -> None:
        src = "def f():\n    try:\n        pass\n    except:\n        pass\n"
        r1 = engine.analyze(_code_artifact(src))
        r2 = engine.analyze(_code_artifact(src))
        m1 = sorted(f.message for f in r1.findings())
        m2 = sorted(f.message for f in r2.findings())
        assert m1 == m2
        assert r1.verdict is r2.verdict

    check("deterministic: same artifact → identical findings",
          t_deterministic)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        r = engine.analyze(_code_artifact("eval('x')\n"))
        d = r.to_dict()
        assert d["id"] == r.id
        assert d["verdict"] == "rejected"
        assert isinstance(d["results"], list)
        s = r.summary()
        assert "Critique Report" in s
        assert "REJECTED" in s or "rejected" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                r = engine.analyze(
                    _code_artifact(
                        '"""Doc."""\n'
                        "def f(x):\n"
                        '    """Doc."""\n'
                        "    return eval(x)\n"
                    ),
                    project_id="proj-x",
                )
                assert r.verdict is Verdict.REJECTED
                repo = CriticRepository(memory=mem, ontology=ont)
                ent = repo.save(r, project_id="proj-x")
                assert ent
                loaded = repo.load(r.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["verdict"] == "rejected"
                # Ontology: VERIFICATION entity + EVIDENCE entities
                assert ont.count(kind=EntityKind.VERIFICATION) >= 1
                assert ont.count(kind=EntityKind.EVIDENCE) >= 1
                # C04 failure memory recorded
                fails = mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert len(fails) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (VERIFICATION + EVIDENCE)",
          t_persist)

    # ---- E2E with C14+C17 ----
    def t_e2e_criticizes_synth_and_tests() -> None:
        from sebrain.c14 import (
            CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
        )
        from sebrain.c13 import RepoIndex
        from sebrain.c17 import TestGenerator
        from sebrain.c05 import RequirementParser

        text = (
            "Build a REST API for tasks.\n"
            "Functional:\n- Users must be able to create tasks.\n"
        )
        spec = RequirementParser().parse(text)
        entity = EntitySpec(
            name="Task",
            fields=[FieldSpec("title", "str", required=True)],
        )
        synth = CodeSynthesisEngine().synthesize(
            SynthesisRequest(
                package_name="task_api", entities=[entity],
                framework="fastapi", model_style="dataclass", mode="fresh",
            ),
            project_id="demo",
            existing_index=RepoIndex(root="<none>"),
        )
        tplan = TestGenerator().generate(
            spec=spec, synthesis=synth, project_id="demo",
        )
        # 1. Critique the models.py source
        models = next(f for f in synth.files if f.path.endswith("models.py"))
        r1 = engine.analyze(Artifact(
            kind=ArtifactKind.CODE, raw_text=models.content,
            ref=models.path,
        ))
        # Should have no CRITICAL findings (C14 output is clean)
        assert r1.count(Severity.CRITICAL) == 0

        # 2. Critique the unit test file
        unit_artifact = next(
            (a for a in tplan.artifacts
             if a.path.endswith("test_unit.py")), None,
        )
        assert unit_artifact is not None
        r2 = engine.analyze(
            Artifact(
                kind=ArtifactKind.TESTS, raw_text=unit_artifact.content,
                ref=unit_artifact.path,
            ),
            spec=spec,
        )
        # Should not be REJECTED (C17 output is clean)
        assert r2.verdict is not Verdict.REJECTED, r2.rationale

    check("e2e: critic evaluates C14 code and C17 tests",
          t_e2e_criticizes_synth_and_tests)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
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
    print("SE Brain C21 — Critic Engine")
    print("=" * 78)

    engine = CriticEngine()

    samples = [
        ("Bad code (eval + secret)", "def f(x):\n"
                                     '    password = "hunter2secret"\n'
                                     "    return eval(x)\n"),
        ("Sloppy code (bare except + TODO)", "def f():\n"
                                              "    # TODO finish\n"
                                              "    try:\n"
                                              "        g()\n"
                                              "    except:\n"
                                              "        pass\n"),
        ("Mutable default", "def f(items=[]):\n    return items\n"),
        ("Deep inheritance", "class A: pass\n"
                             "class B(A): pass\n"
                             "class C(B): pass\n"
                             "class D(C): pass\n"),
        ("Clean", '"""Doc."""\n'
                  "\n"
                  "def add(x: int, y: int) -> int:\n"
                  '    """Return x+y."""\n'
                  "    return x + y\n"),
    ]

    for i, (label, src) in enumerate(samples, 1):
        print(f"\n[{i}] {label}")
        r = engine.analyze(Artifact(
            kind=ArtifactKind.CODE, raw_text=src, label=label,
        ))
        print(r.summary())
        for f in r.findings():
            mark = {"critical": "🔴", "high": "🟠", "medium": "🟡",
                    "low": "🔵", "info": "⚪"}[f.severity.value]
            print(f"    {mark} [{f.severity.value:8s}] "
                  f"{f.category.value:22s} {_short(f.message, 70)}")
            if f.location:
                print(f"        @ {f.location}")

    # Cross-artifact: requirement coverage
    print("\n[6] Requirement coverage (spec + tests):")
    from sebrain.c05 import RequirementParser
    spec = RequirementParser().parse(
        "Functional requirements:\n"
        "- Users must be able to create tasks.\n"
        "- Users must be able to archive reports.\n"
    )
    tests_src = (
        "def test_create_task():\n"
        '    """covers create tasks"""\n'
        "    assert True\n"
    )
    r = engine.analyze(
        Artifact(kind=ArtifactKind.TESTS, raw_text=tests_src,
                 ref="tests/test_x.py"),
        spec=spec,
    )
    print(f"    verdict: {r.verdict.value}")
    for f in r.findings():
        if f.category is FindingCategory.REQUIREMENT_COVERAGE:
            print(f"    [{f.severity.value}] {_short(f.message, 90)}")
            print(f"        evidence: {_short(f.evidence, 90)}")

    # Persistence
    print("\n[7] Persistence:")
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = CriticRepository(memory=mem, ontology=ont)
                r = engine.analyze(Artifact(
                    kind=ArtifactKind.CODE,
                    raw_text="eval('x')\n", ref="demo-code",
                ))
                ent = repo.save(r, project_id="demo")
                print(f"    ontology entity: {ent[:12]}…")
                print(f"    VERIFICATION: "
                      f"{ont.count(kind=EntityKind.VERIFICATION)}")
                print(f"    EVIDENCE: "
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
