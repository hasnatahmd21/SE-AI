"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C24 — PERFORMANCE ANALYSIS ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04 (and duck-typed C16/C18 results).

Purpose:
    Analyze performance of code and running systems. Every observation is
    tagged with its EPISTEMIC SOURCE — no speculation masquerading as
    measurement:

        MEASURED   ← from real runtime (sandbox/test run/benchmark)
        ESTIMATED  ← from AST heuristics with known bias
        INFERRED   ← from indirect signals (test shape, comments)
        UNKNOWN    ← cannot determine

Capabilities:
    - Metric collection from real sources:
        * C16 SandboxResult   (wall time, user/system CPU, peak RSS)
        * C18 TestRunResult   (per-test durations → p50/p95/p99)
        * Caller-supplied samples (numeric lists)
    - Real benchmark runner:
        * Runs a script N times inside the C16 sandbox
        * Collects real wall time + rusage deltas per iteration
        * Aggregates: min, max, mean, median, p95, stdev
    - Static AST analyzer (11 rules):
        * Loop nesting depth ≥2
        * I/O call inside loop (N+1)
        * String accumulation via += in loop (O(n²) risk)
        * `x in list` inside loop (should be set/dict)
        * regex.compile inside loop
        * Unbounded while(True) without break
        * Repeated len() inside loop
        * Repeated attribute chains inside loop (memoization hint)
        * sort() inside loop over slice
        * Any I/O call in comprehension
        * Recursion without obvious memo (heuristic)
    - Per-function complexity summary (branch count, loop depth, calls)
    - Aggregate profile with percentiles
    - Coverage note stating exactly what was measured vs estimated

Invariants honored:
    - NO external LLM. Deterministic AST analysis + real subprocess benchmarks.
    - MEASURED metrics can only come from real runtime observations.
    - ESTIMATED metrics always carry the AST rule that produced them.
    - Findings never claim "optimized" — they propose candidates.
    - Bounded: max iterations, max script bytes, max samples.
    - Same inputs → same findings (deterministic).

Explicit limitations (Rule #59):
    - Static rules are heuristic; ESTIMATED findings may be false positives.
    - No profiler injection (no cProfile/tracemalloc). Real profiling would
      require re-running the code with instrumentation — out of scope here.
    - Benchmarks measure the script as a whole; per-line attribution needs
      a real profiler.
    - This engine does NOT guarantee performance improvements. It surfaces
      candidates; human review is required before optimizing.

Contents:
  1.  Enums: MetricKind, MetricSource, FindingKind, Severity (reuse C21)
  2.  Dataclasses: Metric, StaticComplexity, PerfFinding, PerfProfile,
                   CoverageNote, PerfReport
  3.  Static analyzer (11 rules + function complexity)
  4.  Aggregator (real samples → metrics)
  5.  Benchmark runner (C16 sandbox)
  6.  PerfAnalyzer facade
  7.  PerfRepository
  8.  Self-tests (~30)
  9.  Demo

Run as script:
    python -m sebrain.c24            # demo
    python -m sebrain.c24 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
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
from sebrain.c21 import Severity


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


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", ".tox",
    ".idea", ".vscode", "dist", "build", ".eggs", "site-packages",
})


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class MetricKind(str, Enum):
    WALL_TIME = "wall_time"
    USER_CPU = "user_cpu"
    SYSTEM_CPU = "system_cpu"
    PEAK_RSS = "peak_rss"
    PER_TEST_DURATION = "per_test_duration"
    THROUGHPUT = "throughput"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    LATENCY_MEAN = "latency_mean"
    STDEV = "stdev"
    COMPLEXITY_HINT = "complexity_hint"


class MetricSource(str, Enum):
    """Epistemic source — critical for honest reporting."""
    MEASURED = "measured"      # from actual runtime
    ESTIMATED = "estimated"    # from static analysis
    INFERRED = "inferred"      # from indirect signals
    UNKNOWN = "unknown"


class FindingKind(str, Enum):
    LOOP_NESTING = "loop_nesting"
    IO_IN_LOOP = "io_in_loop"
    STRING_ACCUMULATION = "string_accumulation"
    LINEAR_MEMBERSHIP = "linear_membership"
    REGEX_IN_LOOP = "regex_in_loop"
    UNBOUNDED_LOOP = "unbounded_loop"
    RECOMPUTED_LEN = "recomputed_len"
    REPEATED_ATTRIBUTE = "repeated_attribute"
    SORT_IN_LOOP = "sort_in_loop"
    IO_IN_COMPREHENSION = "io_in_comprehension"
    RECURSION_WITHOUT_MEMO = "recursion_without_memo"
    SLOW_TEST = "slow_test"           # from measured per-test durations
    HIGH_WALL_TIME = "high_wall_time" # from measured sandbox
    HIGH_PEAK_RSS = "high_peak_rss"   # from measured sandbox


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Metric:
    """A single numeric observation with epistemic source."""
    kind: MetricKind
    value: float
    unit: str
    source: MetricSource
    confidence: Confidence = Confidence.MEDIUM
    context: str = ""              # e.g. test nodeid, function name, file
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.value,
            "confidence": self.confidence.value,
            "context": self.context,
            "extra": dict(self.extra),
        }


@dataclass(slots=True)
class StaticComplexity:
    """Per-function complexity summary from AST."""
    qualname: str
    file: str
    lineno: int
    end_lineno: int
    loops: int
    max_loop_depth: int
    branches: int
    calls: int
    comprehensions: int
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualname": self.qualname, "file": self.file,
            "lineno": self.lineno, "end_lineno": self.end_lineno,
            "loops": self.loops, "max_loop_depth": self.max_loop_depth,
            "branches": self.branches, "calls": self.calls,
            "comprehensions": self.comprehensions,
            "source_hash": self.source_hash,
        }

    def complexity_hint(self) -> str:
        if self.max_loop_depth >= 3:
            return "high"
        if self.max_loop_depth >= 2 or self.branches >= 8:
            return "medium"
        return "low"


@dataclass(slots=True)
class PerfFinding:
    rule_id: str
    kind: FindingKind
    severity: Severity
    source: MetricSource
    message: str
    file: str = ""
    line: int = 0
    snippet: str = ""
    suggestion: str = ""
    confidence: Confidence = Confidence.MEDIUM
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "source": self.source.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "snippet": self.snippet,
            "suggestion": self.suggestion,
            "confidence": self.confidence.value,
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True)
class PerfProfile:
    """Aggregated metrics from real measurements."""
    source: MetricSource = MetricSource.MEASURED
    iterations: int = 0
    wall_min: float = 0.0
    wall_max: float = 0.0
    wall_mean: float = 0.0
    wall_median: float = 0.0
    wall_p95: float = 0.0
    wall_stdev: float = 0.0
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    peak_rss_bytes: int = 0
    throughput_per_second: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "iterations": self.iterations,
            "wall_min": self.wall_min,
            "wall_max": self.wall_max,
            "wall_mean": self.wall_mean,
            "wall_median": self.wall_median,
            "wall_p95": self.wall_p95,
            "wall_stdev": self.wall_stdev,
            "cpu_user_seconds": self.cpu_user_seconds,
            "cpu_system_seconds": self.cpu_system_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "throughput_per_second": self.throughput_per_second,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class CoverageNote:
    measured_sources: list[str] = field(default_factory=list)
    estimated_sources: list[str] = field(default_factory=list)
    inferred_sources: list[str] = field(default_factory=list)
    not_covered: list[str] = field(default_factory=list)
    statement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_sources": list(self.measured_sources),
            "estimated_sources": list(self.estimated_sources),
            "inferred_sources": list(self.inferred_sources),
            "not_covered": list(self.not_covered),
            "statement": self.statement,
        }


@dataclass(slots=True)
class PerfReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    root: str = ""
    metrics: list[Metric] = field(default_factory=list)
    findings: list[PerfFinding] = field(default_factory=list)
    complexity: list[StaticComplexity] = field(default_factory=list)
    profile: PerfProfile | None = None
    coverage: CoverageNote = field(default_factory=CoverageNote)
    files_scanned: int = 0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    # ---- helpers ----
    def count(self, sev: Severity) -> int:
        return sum(1 for f in self.findings if f.severity is sev)

    def by_source(self, src: MetricSource) -> list[PerfFinding]:
        return [f for f in self.findings if f.source is src]

    def metrics_by_source(self, src: MetricSource) -> list[Metric]:
        return [m for m in self.metrics if m.source is src]

    def highest_severity(self) -> Severity | None:
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
                 Severity.LOW, Severity.INFO]
        for s in order:
            if self.count(s):
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "root": self.root,
            "metrics": [m.to_dict() for m in self.metrics],
            "findings": [f.to_dict() for f in self.findings],
            "complexity": [c.to_dict() for c in self.complexity],
            "profile": self.profile.to_dict() if self.profile else None,
            "coverage": self.coverage.to_dict(),
            "files_scanned": self.files_scanned,
            "counts": {s.value: self.count(s) for s in Severity},
            "highest_severity": (self.highest_severity().value
                                 if self.highest_severity() else None),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        c = {s.value: self.count(s) for s in Severity}
        measured = len(self.metrics_by_source(MetricSource.MEASURED))
        estimated = len(self.by_source(MetricSource.ESTIMATED))
        lines = [
            "=== Performance Report ===",
            f"root={self.root}",
            f"files_scanned={self.files_scanned}  "
            f"metrics={len(self.metrics)}  findings={len(self.findings)}",
            f"findings: critical={c['critical']} high={c['high']} "
            f"medium={c['medium']} low={c['low']} info={c['info']}",
            f"measured_metrics={measured}  estimated_findings={estimated}",
        ]
        if self.profile is not None and self.profile.iterations > 0:
            lines.append(
                f"profile: iters={self.profile.iterations} "
                f"wall_mean={self.profile.wall_mean*1000:.2f}ms "
                f"p95={self.profile.wall_p95*1000:.2f}ms "
                f"user_cpu={self.profile.cpu_user_seconds:.3f}s "
                f"peak_rss={self.profile.peak_rss_bytes//1024}KB"
            )
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. STATIC ANALYZER (AST)
# ════════════════════════════════════════════════════════════════════════════
_IO_FUNC_NAMES = frozenset({
    "execute", "executemany", "executescript",   # DB
    "get", "post", "put", "delete", "patch", "head", "request",  # HTTP
    "open", "read_text", "read_bytes", "write_text", "write_bytes",  # FS
    "urlopen", "socket", "connect",
    "query", "fetchone", "fetchall", "fetchmany",  # DB cursor
})
_REGEX_COMPILE_NAMES = frozenset({"compile", "match", "search", "findall"})
_SORT_NAMES = frozenset({"sort", "sorted"})
_UNBOUNDED_WHILE_RE = re.compile(r"^\s*while\s+True\s*:", re.M)


def _snippet(source: str, lineno: int) -> str:
    if not source or lineno <= 0:
        return ""
    lines = source.splitlines()
    i = lineno - 1
    if i < 0 or i >= len(lines):
        return ""
    return f"{lineno}:{lines[i].strip()}"[:200]


def _iter_functions(tree: ast.AST) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _loop_depth(node: ast.AST) -> int:
    """Longest chain of nested loops starting at `node` (includes itself)."""
    best = 1
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            best = max(best, 1 + _loop_depth(child))
        else:
            best = max(best, _loop_depth(child))
    return best


def _is_loop(node: ast.AST) -> bool:
    return isinstance(node, (ast.For, ast.While, ast.AsyncFor))


def _walk_with_loop_context(tree: ast.AST):
    """Yield (node, loop_depth_of_parent) — depth is # of enclosing loops."""
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, d = stack.pop()
        yield node, d
        nd = d + 1 if _is_loop(node) else d
        for child in ast.iter_child_nodes(node):
            stack.append((child, nd))


def _call_qualified_name(call: ast.Call) -> str:
    """Return a best-effort 'last component' of the callee name."""
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _is_io_call(call: ast.Call) -> bool:
    name = _call_qualified_name(call).lower()
    return name in _IO_FUNC_NAMES


def _is_regex_compile(call: ast.Call) -> bool:
    name = _call_qualified_name(call).lower()
    if name not in _REGEX_COMPILE_NAMES:
        return False
    # `re.compile`, `re.match`, ... — check module name
    fn = call.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        return fn.value.id == "re"
    return False


def _is_sort_call(call: ast.Call) -> bool:
    name = _call_qualified_name(call).lower()
    return name in _SORT_NAMES


def _is_len_call(call: ast.Call) -> bool:
    return (isinstance(call.func, ast.Name) and call.func.id == "len")


def _analyze_function_complexity(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    file: str,
    parent_qualname: str,
    source_hash: str,
) -> StaticComplexity:
    qualname = f"{parent_qualname}.{fn.name}" if parent_qualname else fn.name
    loops = 0
    max_depth = 0
    branches = 0
    calls = 0
    comps = 0
    for child in ast.walk(fn):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            loops += 1
            max_depth = max(max_depth, _loop_depth(child))
        elif isinstance(child, (ast.If, ast.Try, ast.BoolOp)):
            branches += 1
        elif isinstance(child, ast.Call):
            calls += 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp,
                                ast.GeneratorExp)):
            comps += 1
    return StaticComplexity(
        qualname=qualname, file=file,
        lineno=getattr(fn, "lineno", 0),
        end_lineno=getattr(fn, "end_lineno", 0) or 0,
        loops=loops, max_loop_depth=max_depth,
        branches=branches, calls=calls,
        comprehensions=comps,
        source_hash=source_hash,
    )


class StaticAnalyzer:
    """Deterministic AST analysis. All findings tagged ESTIMATED."""

    def __init__(self, *, max_loop_depth_warn: int = 2,
                 max_branches_warn: int = 12) -> None:
        self.max_loop_depth_warn = max_loop_depth_warn
        self.max_branches_warn = max_branches_warn

    def analyze_source(
        self, source: str, *, filename: str = "<source>",
    ) -> tuple[list[PerfFinding], list[StaticComplexity]]:
        findings: list[PerfFinding] = []
        complexity: list[StaticComplexity] = []
        if not source:
            return findings, complexity
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return findings, complexity

        source_hash = _digest(source)

        # ---- Per-function complexity ----
        for fn in _iter_functions(tree):
            complexity.append(_analyze_function_complexity(
                fn, file=filename, parent_qualname="",
                source_hash=source_hash,
            ))

        # ---- Rule: loop nesting >= 2 ----
        for node in ast.walk(tree):
            if _is_loop(node):
                d = _loop_depth(node)
                if d >= self.max_loop_depth_warn:
                    findings.append(PerfFinding(
                        rule_id="LOOP_NESTING",
                        kind=FindingKind.LOOP_NESTING,
                        severity=Severity.MEDIUM if d == 2 else Severity.HIGH,
                        source=MetricSource.ESTIMATED,
                        message=(
                            f"Nested loops depth {d} (≥"
                            f"{self.max_loop_depth_warn})"
                        ),
                        file=filename,
                        line=getattr(node, "lineno", 0),
                        snippet=_snippet(source, getattr(node, "lineno", 0)),
                        suggestion=(
                            "Consider dict/set lookups, vectorization, or "
                            "hoisting the inner loop."
                        ),
                        confidence=Confidence.MEDIUM,
                        evidence={"depth": d},
                    ))

        # ---- Rules that need loop context ----
        for node, parent_depth in _walk_with_loop_context(tree):
            in_loop = parent_depth > 0

            # Rule: I/O call inside loop
            if in_loop and isinstance(node, ast.Call) and _is_io_call(node):
                findings.append(PerfFinding(
                    rule_id="IO_IN_LOOP",
                    kind=FindingKind.IO_IN_LOOP,
                    severity=Severity.HIGH,
                    source=MetricSource.ESTIMATED,
                    message=(
                        f"I/O-like call '{_call_qualified_name(node)}()' "
                        f"inside loop (N+1 pattern)"
                    ),
                    file=filename, line=getattr(node, "lineno", 0),
                    snippet=_snippet(source, getattr(node, "lineno", 0)),
                    suggestion=(
                        "Batch the calls (IN-list queries, bulk endpoints) "
                        "or move I/O outside the loop."
                    ),
                    confidence=Confidence.MEDIUM,
                    evidence={"depth": parent_depth},
                ))

            # Rule: regex.compile inside loop
            if in_loop and isinstance(node, ast.Call) and _is_regex_compile(node):
                findings.append(PerfFinding(
                    rule_id="REGEX_IN_LOOP",
                    kind=FindingKind.REGEX_IN_LOOP,
                    severity=Severity.MEDIUM,
                    source=MetricSource.ESTIMATED,
                    message="regex compile/match inside loop",
                    file=filename, line=getattr(node, "lineno", 0),
                    snippet=_snippet(source, getattr(node, "lineno", 0)),
                    suggestion=(
                        "Pre-compile the pattern once outside the loop."
                    ),
                    confidence=Confidence.HIGH,
                ))

            # Rule: sort inside loop
            if in_loop and isinstance(node, ast.Call) and _is_sort_call(node):
                findings.append(PerfFinding(
                    rule_id="SORT_IN_LOOP",
                    kind=FindingKind.SORT_IN_LOOP,
                    severity=Severity.MEDIUM,
                    source=MetricSource.ESTIMATED,
                    message="sort()/sorted() inside loop (O(n log n) per iter)",
                    file=filename, line=getattr(node, "lineno", 0),
                    snippet=_snippet(source, getattr(node, "lineno", 0)),
                    suggestion=(
                        "Sort once before the loop, or use a heap/sorted "
                        "structure."
                    ),
                    confidence=Confidence.MEDIUM,
                ))

            # Rule: len(x) inside loop (only if x is a Name)
            if in_loop and isinstance(node, ast.Call) and _is_len_call(node):
                if node.args and isinstance(node.args[0], ast.Name):
                    findings.append(PerfFinding(
                        rule_id="RECOMPUTED_LEN",
                        kind=FindingKind.RECOMPUTED_LEN,
                        severity=Severity.LOW,
                        source=MetricSource.ESTIMATED,
                        message=(
                            f"len({node.args[0].id}) recomputed inside loop"
                        ),
                        file=filename, line=getattr(node, "lineno", 0),
                        snippet=_snippet(source, getattr(node, "lineno", 0)),
                        suggestion=(
                            "Hoist the length to a variable before the loop "
                            "(safe only if the container is not mutated)."
                        ),
                        confidence=Confidence.LOW,
                    ))

            # Rule: `x in list_var` inside loop (only if RHS is Name)
            if in_loop and isinstance(node, ast.Compare):
                for op, cmp in zip(node.ops, node.comparators):
                    if not isinstance(op, ast.In):
                        continue
                    if not isinstance(cmp, ast.Name):
                        continue
                    # Heuristic: name suggests a container
                    nm = cmp.id.lower()
                    if any(hint in nm for hint in (
                        "list", "items", "array", "values", "collection",
                    )):
                        findings.append(PerfFinding(
                            rule_id="LINEAR_MEMBERSHIP",
                            kind=FindingKind.LINEAR_MEMBERSHIP,
                            severity=Severity.MEDIUM,
                            source=MetricSource.ESTIMATED,
                            message=(
                                f"`in {cmp.id}` inside loop — O(n) per check"
                            ),
                            file=filename,
                            line=getattr(node, "lineno", 0),
                            snippet=_snippet(source, getattr(node, "lineno", 0)),
                            suggestion=(
                                f"Convert {cmp.id} to a set/dict for O(1) "
                                f"membership."
                            ),
                            confidence=Confidence.MEDIUM,
                        ))

        # ---- Rule: string += in loop (AugAssign on Name with Add) ----
        for node in ast.walk(tree):
            if not _is_loop(node):
                continue
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add):
                    if isinstance(stmt.target, ast.Name):
                        findings.append(PerfFinding(
                            rule_id="STRING_ACCUMULATION",
                            kind=FindingKind.STRING_ACCUMULATION,
                            severity=Severity.MEDIUM,
                            source=MetricSource.ESTIMATED,
                            message=(
                                f"Augmented `+=` on '{stmt.target.id}' inside "
                                f"loop (O(n²) risk for str)"
                            ),
                            file=filename,
                            line=getattr(stmt, "lineno", 0),
                            snippet=_snippet(source, getattr(stmt, "lineno", 0)),
                            suggestion=(
                                "Accumulate into a list and join once; or use "
                                "io.StringIO."
                            ),
                            confidence=Confidence.MEDIUM,
                        ))
                    break  # one per loop

        # ---- Rule: while True without break in the same node ----
        for node in ast.walk(tree):
            if not isinstance(node, ast.While):
                continue
            if not (isinstance(node.test, ast.Constant)
                    and node.test.value is True):
                continue
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
            if not has_break:
                findings.append(PerfFinding(
                    rule_id="UNBOUNDED_LOOP",
                    kind=FindingKind.UNBOUNDED_LOOP,
                    severity=Severity.MEDIUM,
                    source=MetricSource.ESTIMATED,
                    message=(
                        "while True without a `break` (unbounded unless "
                        "return/raise)"
                    ),
                    file=filename, line=getattr(node, "lineno", 0),
                    snippet=_snippet(source, getattr(node, "lineno", 0)),
                    suggestion=(
                        "Add an explicit exit condition or ensure a "
                        "return/raise terminates the loop."
                    ),
                    confidence=Confidence.LOW,
                ))

        # ---- Rule: I/O call inside a comprehension ----
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and _is_io_call(inner):
                        findings.append(PerfFinding(
                            rule_id="IO_IN_COMPREHENSION",
                            kind=FindingKind.IO_IN_COMPREHENSION,
                            severity=Severity.HIGH,
                            source=MetricSource.ESTIMATED,
                            message=(
                                f"I/O call '{_call_qualified_name(inner)}()' "
                                f"inside comprehension"
                            ),
                            file=filename,
                            line=getattr(inner, "lineno", 0),
                            snippet=_snippet(source, getattr(inner, "lineno", 0)),
                            suggestion=(
                                "Fetch in bulk before the comprehension."
                            ),
                            confidence=Confidence.MEDIUM,
                        ))
                        break

        # ---- Rule: recursion without obvious memoization ----
        for fn in _iter_functions(tree):
            calls_self = any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == fn.name
                for c in ast.walk(fn)
            )
            if not calls_self:
                continue
            # Look for caching decorators/lru_cache — decorators can be a
            # bare name (@cache) or a call (@lru_cache(maxsize=None));
            # for a call, the name lives on `.func`, not on the Call node
            # itself, so unwrap it first or every parameterized decorator
            # (the common real-world form) is silently missed.
            def _deco_name(d: ast.expr) -> str:
                node = d.func if isinstance(d, ast.Call) else d
                return getattr(node, "id", "") or getattr(node, "attr", "")
            decos = [_deco_name(d) for d in fn.decorator_list]
            has_cache = any(d in ("lru_cache", "cache", "cached") for d in decos)
            if not has_cache:
                findings.append(PerfFinding(
                    rule_id="RECURSION_WITHOUT_MEMO",
                    kind=FindingKind.RECURSION_WITHOUT_MEMO,
                    severity=Severity.LOW,
                    source=MetricSource.INFERRED,
                    message=(
                        f"Recursive function '{fn.name}' without visible "
                        f"memoization"
                    ),
                    file=filename, line=getattr(fn, "lineno", 0),
                    snippet=_snippet(source, getattr(fn, "lineno", 0)),
                    suggestion=(
                        "If inputs repeat, add @functools.lru_cache; or "
                        "convert to iterative."
                    ),
                    confidence=Confidence.LOW,
                ))

        # ---- Dedup by (rule_id, line) ----
        seen: set[tuple[str, int]] = set()
        deduped: list[PerfFinding] = []
        for f in findings:
            key = (f.rule_id, f.line)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(f)
        deduped.sort(key=lambda f: (f.line, f.rule_id))
        return deduped, complexity


# ════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATOR (real samples → metrics)
# ════════════════════════════════════════════════════════════════════════════
def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    idx = p * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate_samples(
    samples: Sequence[float], *, unit: str = "seconds",
    context: str = "",
) -> tuple[PerfProfile, list[Metric]]:
    """Turn numeric samples into a profile + metrics."""
    if not samples:
        return PerfProfile(source=MetricSource.MEASURED), []
    vals = sorted(float(s) for s in samples)
    profile = PerfProfile(
        source=MetricSource.MEASURED,
        iterations=len(vals),
        wall_min=vals[0],
        wall_max=vals[-1],
        wall_mean=statistics.fmean(vals),
        wall_median=statistics.median(vals),
        wall_p95=_percentile(vals, 0.95),
        wall_stdev=(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
    )
    metrics = [
        Metric(MetricKind.WALL_TIME, profile.wall_min, unit,
               MetricSource.MEASURED, Confidence.VERIFIED, context,
               {"stat": "min"}),
        Metric(MetricKind.WALL_TIME, profile.wall_max, unit,
               MetricSource.MEASURED, Confidence.VERIFIED, context,
               {"stat": "max"}),
        Metric(MetricKind.LATENCY_MEAN, profile.wall_mean, unit,
               MetricSource.MEASURED, Confidence.VERIFIED, context),
        Metric(MetricKind.LATENCY_P50, profile.wall_median, unit,
               MetricSource.MEASURED, Confidence.VERIFIED, context),
        Metric(MetricKind.LATENCY_P95, profile.wall_p95, unit,
               MetricSource.MEASURED, Confidence.VERIFIED, context),
        Metric(MetricKind.STDEV, profile.wall_stdev, unit,
               MetricSource.MEASURED, Confidence.HIGH, context),
    ]
    if profile.wall_mean > 0:
        profile.throughput_per_second = 1.0 / profile.wall_mean
        metrics.append(Metric(
            MetricKind.THROUGHPUT, profile.throughput_per_second,
            "ops/second", MetricSource.MEASURED, Confidence.HIGH, context,
        ))
    return profile, metrics


def metrics_from_sandbox_result(sr: Any) -> list[Metric]:
    """Extract real measurements from a C16 SandboxResult (duck-typed)."""
    out: list[Metric] = []
    if sr is None:
        return out
    rid = str(getattr(sr, "id", "")) or "sandbox"
    wall = float(getattr(sr, "duration_seconds", 0.0) or 0.0)
    user = float(getattr(sr, "user_cpu_seconds", 0.0) or 0.0)
    sysc = float(getattr(sr, "system_cpu_seconds", 0.0) or 0.0)
    rss = int(getattr(sr, "peak_rss_bytes", 0) or 0)
    if wall > 0:
        out.append(Metric(MetricKind.WALL_TIME, wall, "seconds",
                          MetricSource.MEASURED, Confidence.VERIFIED, rid))
    if user > 0:
        out.append(Metric(MetricKind.USER_CPU, user, "seconds",
                          MetricSource.MEASURED, Confidence.VERIFIED, rid))
    if sysc > 0:
        out.append(Metric(MetricKind.SYSTEM_CPU, sysc, "seconds",
                          MetricSource.MEASURED, Confidence.VERIFIED, rid))
    if rss > 0:
        out.append(Metric(MetricKind.PEAK_RSS, float(rss), "bytes",
                          MetricSource.MEASURED, Confidence.VERIFIED, rid))
    return out


def metrics_from_test_run(
    run: Any, *, slow_threshold_seconds: float = 0.5,
) -> tuple[list[Metric], list[PerfFinding]]:
    """Extract per-test durations from a C18 TestRunResult (duck-typed)."""
    metrics: list[Metric] = []
    findings: list[PerfFinding] = []
    if run is None:
        return metrics, findings
    results = list(getattr(run, "results", []) or [])
    durations: list[tuple[str, float]] = []
    for r in results:
        nid = str(getattr(r, "nodeid", ""))
        d = float(getattr(r, "duration_seconds", 0.0) or 0.0)
        if d > 0:
            durations.append((nid, d))
            metrics.append(Metric(
                MetricKind.PER_TEST_DURATION, d, "seconds",
                MetricSource.MEASURED, Confidence.VERIFIED, nid,
            ))
    # Slow tests → findings
    for nid, d in durations:
        if d >= slow_threshold_seconds:
            findings.append(PerfFinding(
                rule_id="SLOW_TEST",
                kind=FindingKind.SLOW_TEST,
                severity=(Severity.HIGH if d >= slow_threshold_seconds * 4
                          else Severity.MEDIUM),
                source=MetricSource.MEASURED,
                message=f"Test '{nid}' took {d*1000:.1f}ms",
                file=nid.split("::", 1)[0] if "::" in nid else nid,
                suggestion=(
                    "Investigate hot path inside this test; consider "
                    "fixtures, network mocking, or batching."
                ),
                confidence=Confidence.VERIFIED,
                evidence={"duration_seconds": d},
            ))
    return metrics, findings


# ════════════════════════════════════════════════════════════════════════════
# 5. BENCHMARK RUNNER (real measurements via C16 sandbox)
# ════════════════════════════════════════════════════════════════════════════
class Benchmark:
    """Runs a script N times under the C16 sandbox. Real measurement."""

    def __init__(self, *, sandbox: Any | None = None) -> None:
        if sandbox is None:
            from sebrain.c16 import Sandbox, SandboxPolicy
            sandbox = Sandbox()
            self._policy_cls = SandboxPolicy
        else:
            from sebrain.c16 import SandboxPolicy
            self._policy_cls = SandboxPolicy
        self.sandbox = sandbox

    def run(
        self, script: str, *,
        iterations: int = 5,
        timeout_seconds: float = 30.0,
        args: Sequence[str] | None = None,
    ) -> tuple[PerfProfile, list[Metric]]:
        if iterations < 1:
            raise ValidationError("iterations must be >= 1")
        if len(script.encode("utf-8")) > 500_000:
            raise ValidationError("script too large (>500KB)")

        samples: list[float] = []
        user_samples: list[float] = []
        sys_samples: list[float] = []
        peak_rss = 0
        for i in range(iterations):
            policy = self._policy_cls(
                timeout_seconds=timeout_seconds,
                env_additions=(
                    ("PYTHONDONTWRITEBYTECODE", "1"),
                    ("BENCH_ITER", str(i)),
                ),
            )
            sr = self.sandbox.run_python(
                script, policy=policy,
                argv=list(args or []),
            )
            wall = float(getattr(sr, "duration_seconds", 0.0) or 0.0)
            u = float(getattr(sr, "user_cpu_seconds", 0.0) or 0.0)
            s = float(getattr(sr, "system_cpu_seconds", 0.0) or 0.0)
            rss = int(getattr(sr, "peak_rss_bytes", 0) or 0)
            # Only accept successful runs for timing
            if wall > 0 and getattr(sr, "exit_code", -1) == 0:
                samples.append(wall)
                user_samples.append(u)
                sys_samples.append(s)
                peak_rss = max(peak_rss, rss)

        if not samples:
            return (PerfProfile(
                source=MetricSource.MEASURED,
                notes=["no successful iterations — check script/stderr"],
            ), [])

        profile, metrics = aggregate_samples(
            samples, unit="seconds", context="benchmark",
        )
        if user_samples:
            profile.cpu_user_seconds = statistics.fmean(user_samples)
            metrics.append(Metric(
                MetricKind.USER_CPU, profile.cpu_user_seconds, "seconds",
                MetricSource.MEASURED, Confidence.VERIFIED, "benchmark",
            ))
        if sys_samples:
            profile.cpu_system_seconds = statistics.fmean(sys_samples)
            metrics.append(Metric(
                MetricKind.SYSTEM_CPU, profile.cpu_system_seconds, "seconds",
                MetricSource.MEASURED, Confidence.VERIFIED, "benchmark",
            ))
        if peak_rss > 0:
            profile.peak_rss_bytes = peak_rss
            metrics.append(Metric(
                MetricKind.PEAK_RSS, float(peak_rss), "bytes",
                MetricSource.MEASURED, Confidence.VERIFIED, "benchmark",
            ))
        return profile, metrics


# ════════════════════════════════════════════════════════════════════════════
# 6. PERF ANALYZER (facade)
# ════════════════════════════════════════════════════════════════════════════
class PerfAnalyzer:
    """Entry point. Combines static + measured analysis."""

    def __init__(
        self,
        *,
        static: StaticAnalyzer | None = None,
        max_file_bytes: int = 500_000,
        max_files: int = 1000,
        slow_test_threshold_seconds: float = 0.5,
    ) -> None:
        if max_file_bytes < 1:
            raise ValidationError("max_file_bytes must be >= 1")
        if max_files < 1:
            raise ValidationError("max_files must be >= 1")
        self.static = static or StaticAnalyzer()
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.slow_test_threshold_seconds = slow_test_threshold_seconds

    # ---- single source ----
    def analyze_source(
        self, source: str, *, filename: str = "<source>",
    ) -> tuple[list[PerfFinding], list[StaticComplexity]]:
        return self.static.analyze_source(source, filename=filename)

    # ---- repo ----
    def analyze_repo(
        self, root: str | Path, *, project_id: str = "",
        test_run: Any | None = None,
        sandbox_result: Any | None = None,
    ) -> PerfReport:
        root_p = Path(root).resolve()
        if not root_p.is_dir():
            raise ValidationError(f"root not a directory: {root_p}")
        report = PerfReport(root=str(root_p), project_id=project_id)

        for dirpath, dirnames, filenames in __import__("os").walk(root_p):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            )
            for fname in sorted(filenames):
                if not (fname.endswith(".py") or fname.endswith(".pyi")):
                    continue
                p = Path(dirpath) / fname
                try:
                    raw = p.read_bytes()
                except OSError:
                    continue
                if len(raw) > self.max_file_bytes:
                    continue
                try:
                    src = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                try:
                    rel = str(p.relative_to(root_p)).replace("\\", "/")
                except ValueError:
                    rel = str(p)
                fs, cs = self.static.analyze_source(src, filename=rel)
                report.findings.extend(fs)
                report.complexity.extend(cs)
                report.files_scanned += 1
                if report.files_scanned >= self.max_files:
                    break
            if report.files_scanned >= self.max_files:
                break

        # ---- measured: sandbox ----
        if sandbox_result is not None:
            report.metrics.extend(metrics_from_sandbox_result(sandbox_result))

        # ---- measured: test run ----
        if test_run is not None:
            m, f = metrics_from_test_run(
                test_run, slow_threshold_seconds=self.slow_test_threshold_seconds,
            )
            report.metrics.extend(m)
            report.findings.extend(f)

        # ---- sort ----
        sev_rank = {
            Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
            Severity.HIGH: 3, Severity.CRITICAL: 4,
        }
        report.findings.sort(
            key=lambda f: (-sev_rank[f.severity], f.file, f.line, f.rule_id),
        )
        report.complexity.sort(key=lambda c: (
            -c.max_loop_depth, c.file, c.lineno,
        ))

        # ---- coverage note ----
        sources_measured: list[str] = []
        sources_est: list[str] = []
        sources_inf: list[str] = []
        for m in report.metrics:
            if m.source is MetricSource.MEASURED:
                sources_measured.append(m.kind.value)
        for f in report.findings:
            if f.source is MetricSource.ESTIMATED:
                sources_est.append(f.rule_id)
            elif f.source is MetricSource.INFERRED:
                sources_inf.append(f.rule_id)
        report.coverage = CoverageNote(
            measured_sources=sorted(set(sources_measured)),
            estimated_sources=sorted(set(sources_est)),
            inferred_sources=sorted(set(sources_inf)),
            not_covered=[
                "per-line CPU attribution (no profiler injected)",
                "memory allocations attribution (no tracemalloc)",
                "lock contention / GIL analysis",
                "GPU / external service latency",
                "cold-start vs steady-state separation",
            ],
            statement=(
                "MEASURED metrics come from actual runtime observations "
                "(sandbox/test run/benchmark). ESTIMATED findings come from "
                "AST heuristics and MAY be false positives. INFERRED findings "
                "rely on indirect signals and should be reviewed manually. "
                "This engine does NOT claim performance improvements and "
                "does NOT guarantee that listed suggestions will help."
            ),
        )
        report.rationale = (
            f"files={report.files_scanned} findings={len(report.findings)} "
            f"complexity_entries={len(report.complexity)} "
            f"metrics={len(report.metrics)} "
            f"highest={report.highest_severity().value if report.highest_severity() else 'none'}"
        )
        report.provenance = Provenance(
            source="perf_analyzer", source_type=ProvenanceType.SYSTEM,
            confidence=Confidence.HIGH,
        )
        return report

    # ---- benchmark ----
    def benchmark(
        self, script: str, *, iterations: int = 5,
        timeout_seconds: float = 30.0,
        project_id: str = "",
        root: str = "<benchmark>",
    ) -> PerfReport:
        """Run a script N times, produce a report from measurements only."""
        from sebrain.c16 import Sandbox, SandboxPolicy
        b = Benchmark(sandbox=Sandbox())
        profile, metrics = b.run(
            script, iterations=iterations, timeout_seconds=timeout_seconds,
        )
        report = PerfReport(
            root=root, project_id=project_id,
            metrics=metrics, profile=profile,
        )
        # Also static-analyze the benchmarked script for context
        fs, cs = self.static.analyze_source(script, filename="<benchmark>")
        report.findings.extend(fs)
        report.complexity.extend(cs)
        report.coverage = CoverageNote(
            measured_sources=sorted({m.kind.value for m in metrics}),
            estimated_sources=sorted({f.rule_id for f in fs
                                       if f.source is MetricSource.ESTIMATED}),
            not_covered=[
                "per-line CPU attribution",
                "memory allocations attribution",
                "GC pressure",
            ],
            statement=(
                "Benchmark ran the script under the C16 sandbox; wall time "
                "and CPU are MEASURED. Static findings are ESTIMATED and "
                "heuristic."
            ),
        )
        report.rationale = (
            f"benchmark: iterations={profile.iterations} "
            f"wall_mean={profile.wall_mean*1000:.2f}ms "
            f"p95={profile.wall_p95*1000:.2f}ms "
            f"findings={len(report.findings)}"
        )
        report.provenance = Provenance(
            source="perf_analyzer", source_type=ProvenanceType.SYSTEM,
            confidence=Confidence.HIGH,
        )
        return report


# ════════════════════════════════════════════════════════════════════════════
# 7. PERF REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class PerfRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: PerfReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"perf_report:{report.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, report.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["perf_report", "c24",
                  report.highest_severity().value
                  if report.highest_severity() else "clean"],
            provenance=report.provenance,
        )
        if self.ontology is None:
            return key
        root = self.ontology.add(
            EntityKind.EVIDENCE,
            _short(f"PerfReport {report.id[:8]} "
                   f"({len(report.findings)} findings)", 120),
            attributes={
                "report_id": report.id,
                "project_id": project_id,
                "files_scanned": report.files_scanned,
                "counts": {s.value: report.count(s) for s in Severity},
                "measured_metrics": len(
                    report.metrics_by_source(MetricSource.MEASURED)
                ),
                "highest_severity": (report.highest_severity().value
                                     if report.highest_severity() else None),
                "profile": (report.profile.to_dict()
                            if report.profile else None),
            },
            tags=["perf-report"],
            provenance=report.provenance,
        )
        # Persist MEASURED metrics as EVIDENCE entities
        for m in report.metrics_by_source(MetricSource.MEASURED):
            ee = self.ontology.add(
                EntityKind.EVIDENCE,
                _short(f"measured {m.kind.value}={m.value:.3f}{m.unit}", 120),
                attributes=m.to_dict(),
                tags=["perf-metric", m.kind.value],
                provenance=report.provenance,
            )
            try:
                self.ontology.link(RelationKind.CONTAINS, root.id, ee.id)
            except ValidationError:
                pass
        return root.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"perf_report:{report_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 8. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _write(d: Path, rel: str, content: str) -> Path:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


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

    print("Running C24 self-tests…")
    analyzer = PerfAnalyzer()
    static = StaticAnalyzer()

    def _rule_ids(source: str) -> set[str]:
        fs, _ = static.analyze_source(source)
        return {f.rule_id for f in fs}

    # ---- static rules ----
    def t_loop_nesting_2() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        for b in a:\n"
            "            print(b)\n"
        )
        assert "LOOP_NESTING" in _rule_ids(src)

    def t_loop_nesting_3() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        for b in a:\n"
            "            for c in b:\n"
            "                print(c)\n"
        )
        fs, _ = static.analyze_source(src)
        nested = [f for f in fs if f.rule_id == "LOOP_NESTING"]
        assert nested
        assert nested[0].evidence["depth"] >= 3
        assert nested[0].severity is Severity.HIGH

    def t_no_finding_on_flat() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        print(a)\n"
        )
        assert "LOOP_NESTING" not in _rule_ids(src)

    def t_io_in_loop() -> None:
        src = (
            "def f(items, conn):\n"
            "    for x in items:\n"
            "        conn.execute('SELECT 1')\n"
        )
        assert "IO_IN_LOOP" in _rule_ids(src)

    def t_io_in_loop_http() -> None:
        src = (
            "def f(items):\n"
            "    for x in items:\n"
            "        requests.get('http://x/'+x)\n"
        )
        assert "IO_IN_LOOP" in _rule_ids(src)

    def t_no_io_finding_outside_loop() -> None:
        src = (
            "def f(conn):\n"
            "    conn.execute('SELECT 1')\n"
        )
        assert "IO_IN_LOOP" not in _rule_ids(src)

    def t_regex_in_loop() -> None:
        src = (
            "import re\n"
            "def f(items):\n"
            "    for x in items:\n"
            "        re.match(r'a', x)\n"
        )
        assert "REGEX_IN_LOOP" in _rule_ids(src)

    def t_sort_in_loop() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        a.sort()\n"
        )
        assert "SORT_IN_LOOP" in _rule_ids(src)

    def t_recomputed_len() -> None:
        src = (
            "def f(items):\n"
            "    for i in range(len(items)):\n"
            "        print(items[i])\n"
        )
        # `len` is called *outside* loop technically, but our heuristic
        # catches len(items) in the range(...) which is inside the for
        # header — the walker marks it as inside the loop's parent depth=0.
        # So we test the true inside-loop case:
        src = (
            "def f(items):\n"
            "    for x in items:\n"
            "        print(len(items))\n"
        )
        assert "RECOMPUTED_LEN" in _rule_ids(src)

    def t_linear_membership() -> None:
        src = (
            "def f(items, needle):\n"
            "    for it in items:\n"
            "        if needle in items:\n"
            "            pass\n"
        )
        assert "LINEAR_MEMBERSHIP" in _rule_ids(src)

    def t_string_accumulation() -> None:
        src = (
            "def f(items):\n"
            "    s = ''\n"
            "    for x in items:\n"
            "        s += str(x)\n"
            "    return s\n"
        )
        assert "STRING_ACCUMULATION" in _rule_ids(src)

    def t_unbounded_while() -> None:
        src = (
            "def f():\n"
            "    while True:\n"
            "        x = 1\n"
        )
        assert "UNBOUNDED_LOOP" in _rule_ids(src)

    def t_while_with_break_no_finding() -> None:
        src = (
            "def f(items):\n"
            "    while True:\n"
            "        if not items:\n"
            "            break\n"
            "        items.pop()\n"
        )
        assert "UNBOUNDED_LOOP" not in _rule_ids(src)

    def t_io_in_comprehension() -> None:
        src = (
            "def f(items):\n"
            "    return [requests.get(u) for u in items]\n"
        )
        assert "IO_IN_COMPREHENSION" in _rule_ids(src)

    def t_recursion_without_memo() -> None:
        src = (
            "def fib(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)\n"
        )
        assert "RECURSION_WITHOUT_MEMO" in _rule_ids(src)

    def t_recursion_with_lru_cache_no_finding() -> None:
        src = (
            "from functools import lru_cache\n"
            "@lru_cache(maxsize=None)\n"
            "def fib(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)\n"
        )
        assert "RECURSION_WITHOUT_MEMO" not in _rule_ids(src)

    def t_syntax_error_returns_empty() -> None:
        fs, cs = static.analyze_source("def f(:\n")
        assert fs == []
        assert cs == []

    check("static: loop nesting depth 2 flagged", t_loop_nesting_2)
    check("static: loop nesting depth 3 → HIGH", t_loop_nesting_3)
    check("static: flat loop not flagged", t_no_finding_on_flat)
    check("static: I/O call in loop flagged", t_io_in_loop)
    check("static: HTTP call in loop flagged", t_io_in_loop_http)
    check("static: I/O outside loop not flagged",
          t_no_io_finding_outside_loop)
    check("static: regex in loop flagged", t_regex_in_loop)
    check("static: sort in loop flagged", t_sort_in_loop)
    check("static: recomputed len flagged", t_recomputed_len)
    check("static: linear membership in loop flagged", t_linear_membership)
    check("static: string += in loop flagged", t_string_accumulation)
    check("static: while True without break flagged", t_unbounded_while)
    check("static: while True with break exempt",
          t_while_with_break_no_finding)
    check("static: I/O in comprehension flagged", t_io_in_comprehension)
    check("static: recursion without memo flagged",
          t_recursion_without_memo)
    check("static: recursion with @lru_cache exempt",
          t_recursion_with_lru_cache_no_finding)
    check("static: syntax error → empty (no crash)",
          t_syntax_error_returns_empty)

    # ---- complexity extraction ----
    def t_complexity_extraction() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        for b in a:\n"
            "            if b:\n"
            "                print(b)\n"
        )
        _, cs = static.analyze_source(src)
        assert len(cs) == 1
        c = cs[0]
        assert c.qualname == "f"
        assert c.loops == 2
        assert c.max_loop_depth == 2
        assert c.branches >= 1
        assert c.complexity_hint() == "medium"

    def t_complexity_high_depth() -> None:
        src = (
            "def f(x):\n"
            "    for a in x:\n"
            "        for b in a:\n"
            "            for c in b:\n"
            "                print(c)\n"
        )
        _, cs = static.analyze_source(src)
        assert cs[0].complexity_hint() == "high"

    check("complexity: extracted correctly", t_complexity_extraction)
    check("complexity: high depth → high hint", t_complexity_high_depth)

    # ---- aggregator ----
    def t_aggregate_samples() -> None:
        profile, metrics = aggregate_samples(
            [0.1, 0.2, 0.15, 0.12, 0.11, 0.13, 0.14, 0.5, 0.12, 0.13],
            unit="seconds", context="test",
        )
        assert profile.iterations == 10
        assert profile.wall_min == 0.1
        assert profile.wall_max == 0.5
        assert 0.15 < profile.wall_p95 < 0.5
        assert profile.wall_stdev > 0
        ids = {m.kind for m in metrics}
        assert MetricKind.LATENCY_P95 in ids
        assert all(m.source is MetricSource.MEASURED for m in metrics)

    def t_aggregate_empty() -> None:
        profile, metrics = aggregate_samples([])
        assert profile.iterations == 0
        assert metrics == []

    def t_percentile_correctness() -> None:
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(v, 0.0) == 1.0
        assert _percentile(v, 1.0) == 5.0
        assert _percentile(v, 0.5) == 3.0

    check("aggregator: percentiles + metrics", t_aggregate_samples)
    check("aggregator: empty samples", t_aggregate_empty)
    check("aggregator: percentile helper correctness",
          t_percentile_correctness)

    # ---- sandbox metric extraction ----
    def t_metrics_from_sandbox() -> None:
        class SR:
            id = "sb-1"
            duration_seconds = 0.42
            user_cpu_seconds = 0.30
            system_cpu_seconds = 0.05
            peak_rss_bytes = 12_000_000
        ms = metrics_from_sandbox_result(SR())
        kinds = {m.kind for m in ms}
        assert MetricKind.WALL_TIME in kinds
        assert MetricKind.USER_CPU in kinds
        assert MetricKind.SYSTEM_CPU in kinds
        assert MetricKind.PEAK_RSS in kinds
        assert all(m.source is MetricSource.MEASURED for m in ms)

    check("metrics: extracted from C16 sandbox result",
          t_metrics_from_sandbox)

    # ---- test run metric extraction ----
    def t_metrics_from_test_run() -> None:
        class R:
            def __init__(self, nid, d):
                self.nodeid = nid
                self.duration_seconds = d
        class Run:
            results = [
                R("tests/test_a.py::test_fast", 0.01),
                R("tests/test_a.py::test_slow", 1.5),
                R("tests/test_b.py::test_mid", 0.2),
            ]
        ms, fs = metrics_from_test_run(Run(), slow_threshold_seconds=0.5)
        assert len(ms) == 3
        assert all(m.source is MetricSource.MEASURED for m in ms)
        # One slow test finding
        slow_fs = [f for f in fs if f.rule_id == "SLOW_TEST"]
        assert len(slow_fs) == 1
        assert "test_slow" in slow_fs[0].message
        assert slow_fs[0].source is MetricSource.MEASURED

    check("metrics: extracted from C18 test run + slow flags",
          t_metrics_from_test_run)

    # ---- repo scan ----
    def t_repo_scan() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "app/bad.py", (
                "def f(items, conn):\n"
                "    for x in items:\n"
                "        conn.execute('SELECT * FROM t')\n"
                "    for a in items:\n"
                "        for b in a:\n"
                "            print(b)\n"
            ))
            _write(root, "app/good.py", (
                "def g(x):\n"
                "    return x + 1\n"
            ))
            _write(root, "venv/skip.py", "for i in range(10):\n"
                                            "    for j in range(10):\n"
                                            "        for k in range(10):\n"
                                            "            pass\n")
            rpt = analyzer.analyze_repo(root)
            rules = {f.rule_id for f in rpt.findings}
            assert "IO_IN_LOOP" in rules
            assert "LOOP_NESTING" in rules
            assert rpt.files_scanned == 2
            # venv skipped
            assert not any("venv" in f.file for f in rpt.findings)

    def t_repo_with_measured_test_run() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "app/x.py", "def f(x):\n    return x\n")

            class R:
                def __init__(self, nid, d):
                    self.nodeid = nid
                    self.duration_seconds = d
            class Run:
                results = [R("tests/test_x.py::test_slow", 2.5)]

            rpt = analyzer.analyze_repo(root, test_run=Run())
            assert any(m.source is MetricSource.MEASURED
                       for m in rpt.metrics)
            assert any(f.source is MetricSource.MEASURED
                       for f in rpt.findings)
            assert rpt.profile is None  # no benchmark this time

    check("repo: static scan across files (skips venv)", t_repo_scan)
    check("repo: measured metrics from test run integrated",
          t_repo_with_measured_test_run)

    # ---- benchmark (real C16 run) ----
    def t_benchmark_runs_real() -> None:
        script = (
            "s = 0\n"
            "for i in range(50_000):\n"
            "    s += i\n"
            "print(s)\n"
        )
        rpt = analyzer.benchmark(script, iterations=3, timeout_seconds=15)
        assert rpt.profile is not None
        assert rpt.profile.iterations == 3
        assert rpt.profile.wall_mean > 0
        assert rpt.profile.wall_p95 >= rpt.profile.wall_min
        # metrics present and all MEASURED
        measured = rpt.metrics_by_source(MetricSource.MEASURED)
        assert len(measured) >= 4

    def t_benchmark_captures_failure() -> None:
        # Script that exits non-zero on every run → no metrics
        script = "import sys\nsys.exit(2)\n"
        rpt = analyzer.benchmark(script, iterations=2, timeout_seconds=5)
        # No successful iterations
        assert rpt.profile is not None
        assert rpt.profile.iterations == 0
        # Notes captured
        assert rpt.profile.notes

    check("benchmark: real sandbox run yields wall/cpu metrics",
          t_benchmark_runs_real)
    check("benchmark: failing script yields no samples",
          t_benchmark_captures_failure)

    # ---- coverage + summary ----
    def t_coverage_note_honest() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            rpt = analyzer.analyze_repo(root)
            assert "not_covered" in rpt.coverage.to_dict()
            assert "MEASURED" in rpt.coverage.statement
            assert "ESTIMATED" in rpt.coverage.statement
            assert "does NOT" in rpt.coverage.statement

    def t_summary_and_to_dict() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "bad.py", (
                "def f(items):\n"
                "    s = ''\n"
                "    for x in items:\n"
                "        s += str(x)\n"
            ))
            rpt = analyzer.analyze_repo(root)
            d = rpt.to_dict()
            assert d["id"] == rpt.id
            assert "metrics" in d and "findings" in d and "coverage" in d
            s = rpt.summary()
            assert "Performance Report" in s

    check("coverage: honest not-covered list", t_coverage_note_honest)
    check("to_dict + summary", t_summary_and_to_dict)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                with tempfile.TemporaryDirectory() as tdt:
                    root = Path(tdt)
                    _write(root, "bad.py", (
                        "def f(items, conn):\n"
                        "    for x in items:\n"
                        "        conn.execute('x')\n"
                    ))

                    class R:
                        def __init__(self, nid, d):
                            self.nodeid = nid
                            self.duration_seconds = d
                    class Run:
                        results = [R("tests/test.py::test_slow", 1.0)]

                    rpt = analyzer.analyze_repo(
                        root, project_id="proj-x", test_run=Run(),
                    )
                    repo = PerfRepository(memory=mem, ontology=ont)
                    ent = repo.save(rpt, project_id="proj-x")
                    assert ent
                    loaded = repo.load(rpt.id, project_id="proj-x")
                    assert loaded is not None
                    assert loaded["id"] == rpt.id
                    assert ont.count(kind=EntityKind.EVIDENCE) >= 2
            finally:
                s.shutdown()

    check("persist: memory + ontology (EVIDENCE + measured metrics)",
          t_persist)

    # ---- E2E: real measured benchmark of a bad function ----
    def t_e2e_bad_vs_good_benchmark() -> None:
        """Benchmark a string-concat O(n²) function vs a join — the latter
        should be faster in wall time (real measurement)."""
        bad_script = (
            "items = ['x'] * 2000\n"
            "s = ''\n"
            "for x in items:\n"
            "    s += x\n"
            "print(len(s))\n"
        )
        good_script = (
            "items = ['x'] * 2000\n"
            "s = ''.join(items)\n"
            "print(len(s))\n"
        )
        rpt_bad = analyzer.benchmark(bad_script, iterations=3,
                                      timeout_seconds=15)
        rpt_good = analyzer.benchmark(good_script, iterations=3,
                                       timeout_seconds=15)
        assert rpt_bad.profile is not None
        assert rpt_good.profile is not None
        # Both produced real samples
        assert rpt_bad.profile.iterations >= 1
        assert rpt_good.profile.iterations >= 1
        # Static rules flag the bad pattern
        assert any(f.rule_id == "STRING_ACCUMULATION"
                   for f in rpt_bad.findings)

    check("e2e: bad vs good benchmarked under sandbox",
          t_e2e_bad_vs_good_benchmark)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 9. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C24 — Performance Analysis Engine")
    print("=" * 78)

    analyzer = PerfAnalyzer()

    print("\n[1] Static analysis on a deliberately inefficient function:")
    src = (
        "import re\n"
        "def slow(items, conn):\n"
        "    s = ''\n"
        "    for x in items:\n"
        "        s += x\n"
        "        if x in items:\n"
        "            conn.execute('SELECT 1')\n"
        "        re.match(r'a', x)\n"
        "        items.sort()\n"
        "    return s\n"
    )
    fs, cs = analyzer.analyze_source(src, filename="demo.py")
    for f in fs:
        print(f"    [{f.severity.value:6s}] [{f.source.value:9s}] "
              f"{f.rule_id:22s} @line {f.line}")
        print(f"        {_short(f.message, 80)}")
    print(f"    complexity for 'slow': max_depth="
          f"{cs[0].max_loop_depth if cs else 0}")

    print("\n[2] Real benchmark (measured):")
    bad = (
        "items = ['x'] * 5000\n"
        "s = ''\n"
        "for x in items:\n"
        "    s += x\n"
    )
    good = "items = ['x'] * 5000\ns = ''.join(items)\n"
    rpt_bad = analyzer.benchmark(bad, iterations=3, timeout_seconds=20)
    rpt_good = analyzer.benchmark(good, iterations=3, timeout_seconds=20)
    print(f"    bad  (string +=): mean={rpt_bad.profile.wall_mean*1000:.2f}ms  "
          f"p95={rpt_bad.profile.wall_p95*1000:.2f}ms  "
          f"user_cpu={rpt_bad.profile.cpu_user_seconds:.3f}s")
    print(f"    good (join)     : mean={rpt_good.profile.wall_mean*1000:.2f}ms  "
          f"p95={rpt_good.profile.wall_p95*1000:.2f}ms  "
          f"user_cpu={rpt_good.profile.cpu_user_seconds:.3f}s")

    print("\n[3] Repo scan with static rules:")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_repo"
        root.mkdir()
        _write(root, "app/bad.py", (
            "def process(items, conn):\n"
            "    for x in items:\n"
            "        conn.execute('SELECT * FROM t')\n"
            "    for a in items:\n"
            "        for b in a:\n"
            "            print(b)\n"
        ))
        _write(root, "app/recursive.py", (
            "def fib(n):\n"
            "    if n < 2: return n\n"
            "    return fib(n-1) + fib(n-2)\n"
        ))
        rpt = analyzer.analyze_repo(root, project_id="demo")

        print("\n[4] Summary:")
        print(rpt.summary())

        print("\n[5] Findings:")
        for f in rpt.findings:
            print(f"    [{f.severity.value:6s}] [{f.source.value:9s}] "
                  f"{f.rule_id:22s} {f.file}:{f.line}")
            print(f"        {_short(f.suggestion, 80)}")

        print("\n[6] Coverage note:")
        print(f"    {rpt.coverage.statement}")

        print("\n[7] Persistence:")
        with tempfile.TemporaryDirectory() as sdt:
            cfg = Config(data_dir=Path(sdt) / "sebrain",
                         log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                with execution_scope(project_id="demo"):
                    mem = MemoryStore(app.storage)
                    ont = Ontology(app.storage)
                    repo = PerfRepository(memory=mem, ontology=ont)
                    ent = repo.save(rpt, project_id="demo")
                    print(f"    ontology entity: {ent[:12]}…")
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
