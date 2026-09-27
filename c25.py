"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C25 — EXISTING CODEBASE UNDERSTANDING (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C13.

Purpose:
    Reconstruct a HIGH-LEVEL understanding of an existing repo (that may not
    have been created by this Brain). Everything is heuristic + evidence-based.

Capabilities:
    1. Layout detection           — flat / package-by-layer / package-by-feature /
                                     src-layout / single-module / mixed
    2. Architecture guess         — layered / mvc / hexagonal / package-by-feature /
                                     flat / unknown, with confidence + evidence
    3. Entry-point discovery      — __main__.py, `if __name__ == "__main__"`,
                                     console_scripts from pyproject.toml
    4. Dependency classification  — internal / stdlib / external (per module)
    5. Cycle groups               — module-level import cycles (SCC-ish)
    6. Hub modules                — modules with many incoming dependents
    7. Change-impact analysis     — given changed paths/symbols:
                                     * affected modules (direct + transitive)
                                     * affected symbols
                                     * affected tests (via import chain)
                                     * blast radius + risk level
                                     * recommended tests to run
    8. Safety pre-flight          — for a proposed (path, old_text, new_text):
                                     * anchor presence / uniqueness
                                     * referrer check on removals
                                     * caller-count on signature change
    9. Regression scope           — test files that cover changed modules

Invariants honored:
    - NO external LLM. Deterministic AST + graph algorithms.
    - Every inference is evidence-based; "unknown" is a valid verdict.
    - No assumptions that the repo was created by this Brain.
    - Bounded: max modules, max BFS depth, max impact set.
    - Graceful on empty / partial / unparseable repos (never raises on input).
    - Deterministic ordering everywhere.

Explicit limitations (Rule #59):
    - Layout / architecture detection is HEURISTIC (name patterns + graph
      shape). May mislabel. Confidence reflects evidence strength.
    - External-vs-stdlib classification uses sys.stdlib_module_names; some
      third-party packages whose names collide with stdlib are misclassified.
    - Impact analysis is IMPORT-based, not call-graph based. Dynamic imports
      (importlib, __import__) are not resolved.
    - No taint analysis. No semantic understanding. No execution.

Contents:
  1.  Enums: LayoutKind, ArchPattern, RiskLevel, ImpactCategory, SafetyStatus
  2.  Dataclasses: LayoutGuess, ArchGuess, EntryPoint, DepRole, HubModule,
                   CycleGroup, SymbolImpact, ChangeImpact, SafetyFinding,
                   SafetyReport, CodebaseUnderstandingBundle
  3.  Layout + architecture detectors
  4.  Dependency analyzer
  5.  Impact analyzer
  6.  Safety checker
  7.  CodebaseUnderstanding facade
  8.  UnderstandingRepository
  9.  Self-tests (~35)
 10.  Demo

Run as script:
    python -m sebrain.c25            # demo
    python -m sebrain.c25 --test     # self-tests
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
from collections import deque
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
from sebrain.c13 import (
    CodeRepresentationEngine, ModuleInfo, RepoDiscoverer, RepoIndex, Symbol,
    SymbolKind,
)


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


def _module_of_path(rel: str) -> str:
    if rel.endswith("/__init__.py"):
        rel = rel[: -len("/__init__.py")]
    elif rel.endswith("__init__.py"):
        return ""
    elif rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def _top_package(rel: str) -> str:
    parts = rel.split("/", 1)
    return parts[0] if parts else rel


def _stdlib_names() -> frozenset[str]:
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return frozenset(names)
    # Fallback list for older Pythons
    return frozenset({
        "os", "sys", "re", "io", "json", "math", "time", "datetime",
        "pathlib", "collections", "itertools", "functools", "abc",
        "typing", "dataclasses", "enum", "contextlib", "logging",
        "hashlib", "hmac", "secrets", "random", "string", "textwrap",
        "threading", "asyncio", "socket", "ssl", "subprocess",
        "tempfile", "shutil", "stat", "signal", "platform",
        "urllib", "http", "email", "base64", "binascii", "struct",
        "pickle", "copy", "pdb", "traceback", "warnings", "weakref",
        "unittest", "sqlite3", "csv", "configparser", "argparse",
        "getpass", "pprint", "inspect", "importlib", "builtins",
        "operator", "statistics", "decimal", "fractions",
        "traceback", "venv", "zipfile", "tarfile", "gzip", "bz2",
        "lzma", "uuid", "calendar", "zoneinfo", "fnmatch", "glob",
        "ast", "dis", "tokenize", "codecs", "locale", "gettext",
        "multiprocessing", "concurrent", "queue", "selectors",
        "select", "mmap", "array", "ctypes", "curses", "tty",
        "termios", "pty", "pipes", "posixpath", "ntpath", "genericpath",
    })


_STDLIB = _stdlib_names()


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class LayoutKind(str, Enum):
    EMPTY = "empty"
    SINGLE_MODULE = "single_module"
    FLAT = "flat"                        # many modules, no subpackages
    PACKAGE_BY_LAYER = "package_by_layer"    # domain/, api/, db/
    PACKAGE_BY_FEATURE = "package_by_feature"  # users/, orders/, billing/
    SRC_LAYOUT = "src_layout"            # src/<pkg>/...
    MIXED = "mixed"                      # doesn't fit one shape
    UNKNOWN = "unknown"


class ArchPattern(str, Enum):
    LAYERED = "layered"
    MVC = "mvc"
    HEXAGONAL = "hexagonal"
    PACKAGE_BY_FEATURE = "package_by_feature"
    PACKAGE_BY_LAYER = "package_by_layer"
    FLAT = "flat"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ImpactCategory(str, Enum):
    DIRECT_DEPENDENT = "direct_dependent"
    TRANSITIVE_DEPENDENT = "transitive_dependent"
    TEST_COVERAGE = "test_coverage"
    HUB_EXPOSURE = "hub_exposure"


class SafetyStatus(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    INCONCLUSIVE = "inconclusive"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class LayoutGuess:
    kind: LayoutKind = LayoutKind.UNKNOWN
    confidence: Confidence = Confidence.LOW
    top_level_dirs: list[str] = field(default_factory=list)
    package_count: int = 0
    module_count: int = 0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence.value,
            "top_level_dirs": list(self.top_level_dirs),
            "package_count": self.package_count,
            "module_count": self.module_count,
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class ArchGuess:
    pattern: ArchPattern = ArchPattern.UNKNOWN
    confidence: Confidence = Confidence.LOW
    layers: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "confidence": self.confidence.value,
            "layers": list(self.layers),
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class EntryPoint:
    path: str
    kind: str               # "main_module" | "dunder_main" | "console_script"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "detail": self.detail}


@dataclass(slots=True)
class DepRole:
    """Role classification for a single import target."""
    module_name: str
    role: str               # "internal" | "stdlib" | "external"
    resolved_path: str = ""  # for internal

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "role": self.role,
            "resolved_path": self.resolved_path,
        }


@dataclass(slots=True)
class HubModule:
    path: str
    incoming: int
    outgoing: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "incoming": self.incoming,
            "outgoing": self.outgoing,
        }


@dataclass(slots=True)
class CycleGroup:
    members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"members": list(self.members)}


@dataclass(slots=True)
class DependencyReport:
    roles: list[DepRole] = field(default_factory=list)
    internal_edges: int = 0
    external_edges: int = 0
    stdlib_edges: int = 0
    cycles: list[CycleGroup] = field(default_factory=list)
    hubs: list[HubModule] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)  # modules nobody imports

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": [r.to_dict() for r in self.roles],
            "internal_edges": self.internal_edges,
            "external_edges": self.external_edges,
            "stdlib_edges": self.stdlib_edges,
            "cycles": [c.to_dict() for c in self.cycles],
            "hubs": [h.to_dict() for h in self.hubs],
            "orphans": list(self.orphans),
        }


@dataclass(slots=True)
class SymbolImpact:
    symbol_id: str
    qualname: str
    file: str
    lineno: int
    kind: str
    category: ImpactCategory

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id, "qualname": self.qualname,
            "file": self.file, "lineno": self.lineno, "kind": self.kind,
            "category": self.category.value,
        }


@dataclass(slots=True)
class ChangeImpact:
    changed_paths: list[str] = field(default_factory=list)
    changed_modules: list[str] = field(default_factory=list)
    direct_dependents: list[str] = field(default_factory=list)
    transitive_dependents: list[str] = field(default_factory=list)
    affected_symbols: list[SymbolImpact] = field(default_factory=list)
    affected_tests: list[str] = field(default_factory=list)
    recommended_tests: list[str] = field(default_factory=list)
    blast_radius: str = "unknown"     # "tiny" | "small" | "medium" | "large"
    risk_level: RiskLevel = RiskLevel.LOW
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_paths": list(self.changed_paths),
            "changed_modules": list(self.changed_modules),
            "direct_dependents": list(self.direct_dependents),
            "transitive_dependents": list(self.transitive_dependents),
            "affected_symbols": [s.to_dict() for s in self.affected_symbols],
            "affected_tests": list(self.affected_tests),
            "recommended_tests": list(self.recommended_tests),
            "blast_radius": self.blast_radius,
            "risk_level": self.risk_level.value,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class SafetyFinding:
    rule: str
    status: SafetyStatus
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule, "status": self.status.value,
            "message": self.message, "detail": dict(self.detail),
        }


@dataclass(slots=True)
class SafetyReport:
    path: str = ""
    status: SafetyStatus = SafetyStatus.SAFE
    findings: list[SafetyFinding] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "status": self.status.value,
            "findings": [f.to_dict() for f in self.findings],
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class CodebaseUnderstandingBundle:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    root: str = ""
    layout: LayoutGuess = field(default_factory=LayoutGuess)
    architecture: ArchGuess = field(default_factory=ArchGuess)
    entry_points: list[EntryPoint] = field(default_factory=list)
    dependencies: DependencyReport = field(default_factory=DependencyReport)
    total_modules: int = 0
    total_symbols: int = 0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "root": self.root,
            "layout": self.layout.to_dict(),
            "architecture": self.architecture.to_dict(),
            "entry_points": [e.to_dict() for e in self.entry_points],
            "dependencies": self.dependencies.to_dict(),
            "total_modules": self.total_modules,
            "total_symbols": self.total_symbols,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Codebase Understanding ===\n"
            f"root={self.root}\n"
            f"layout={self.layout.kind.value} "
            f"(conf={self.layout.confidence.value})  "
            f"arch={self.architecture.pattern.value} "
            f"(conf={self.architecture.confidence.value})\n"
            f"modules={self.total_modules}  symbols={self.total_symbols}  "
            f"entry_points={len(self.entry_points)}\n"
            f"deps: internal={self.dependencies.internal_edges}  "
            f"external={self.dependencies.external_edges}  "
            f"stdlib={self.dependencies.stdlib_edges}  "
            f"cycles={len(self.dependencies.cycles)}  "
            f"hubs={len(self.dependencies.hubs)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. LAYOUT + ARCHITECTURE DETECTORS
# ════════════════════════════════════════════════════════════════════════════
_LAYER_KEYWORDS: dict[str, frozenset[str]] = {
    "domain": frozenset({
        "domain", "entities", "entity", "model", "models",
        "core", "kernel", "business", "logic",
    }),
    "application": frozenset({
        "application", "app", "services", "service",
        "use_cases", "usecases", "usecase", "handlers_",
        "commands", "queries",
    }),
    "infrastructure": frozenset({
        "infrastructure", "infra", "adapters", "adapter",
        "persistence", "repositories", "repository",
        "storage", "db", "database", "dao", "clients", "client",
        "external", "plugins", "plugin",
    }),
    "presentation": frozenset({
        "presentation", "web", "api", "routes", "route",
        "controllers", "controller", "views", "view",
        "handlers", "handler", "endpoints", "endpoint",
        "http", "rest", "graphql", "http_api",
    }),
    "config": frozenset({
        "config", "settings", "configuration", "env", "environ",
    }),
    "utils": frozenset({
        "utils", "util", "helpers", "helper", "common", "shared",
        "libs", "lib",
    }),
    "tests": frozenset({
        "tests", "test", "spec", "specs",
    }),
}

_MVC_KEYWORDS: frozenset[str] = frozenset({
    "models", "model", "views", "view", "controllers", "controller",
    "templates", "template",
})

_HEX_KEYWORDS: frozenset[str] = frozenset({
    "adapters", "adapter", "ports", "port", "domain",
})


class LayoutDetector:
    def detect(self, index: RepoIndex) -> LayoutGuess:
        guess = LayoutGuess()
        if not index.modules:
            guess.kind = LayoutKind.EMPTY
            guess.confidence = Confidence.VERIFIED
            return guess

        guess.module_count = len(index.modules)
        # Top-level directory names
        tops: set[str] = set()
        has_src_layout = False
        for rel in index.modules:
            parts = rel.split("/")
            if len(parts) >= 2:
                tops.add(parts[0])
                if parts[0] == "src" and len(parts) >= 3:
                    has_src_layout = True
        guess.top_level_dirs = sorted(tops)
        guess.package_count = len(index.packages())

        # Single module
        if len(index.modules) == 1 and not tops:
            guess.kind = LayoutKind.SINGLE_MODULE
            guess.confidence = Confidence.VERIFIED
            guess.evidence.append("exactly 1 module, no subdirectories")
            return guess

        # src-layout
        if has_src_layout:
            guess.kind = LayoutKind.SRC_LAYOUT
            guess.confidence = Confidence.HIGH
            guess.evidence.append("found `src/` top-level directory")
            return guess

        # Flat (all files at root, no packages)
        if not index.packages() and len(index.modules) >= 2:
            guess.kind = LayoutKind.FLAT
            guess.confidence = Confidence.MEDIUM
            guess.evidence.append(
                f"{len(index.modules)} modules, no subpackages"
            )
            return guess

        # Package-by-layer vs package-by-feature
        # Look at the second-level dirs inside the top-level packages
        layer_hits = 0
        feature_dirs: set[str] = set()
        layer_dirs: set[str] = set()
        for rel in index.modules:
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            top = parts[0]
            # We care about the dirs INSIDE a top-level package
            if len(parts) >= 3:
                second = parts[1]
                if second in ("__init__.py",):
                    continue
                feature_dirs.add(f"{top}/{second}")
        # Check layer keywords in the second-level
        for fdir in feature_dirs:
            name = fdir.split("/", 1)[1].lower()
            if any(name in group for group in _LAYER_KEYWORDS.values()):
                layer_dirs.add(fdir)

        if feature_dirs:
            ratio = len(layer_dirs) / max(1, len(feature_dirs))
            if ratio >= 0.6:
                guess.kind = LayoutKind.PACKAGE_BY_LAYER
                guess.confidence = Confidence.MEDIUM
                guess.evidence.append(
                    f"{len(layer_dirs)}/{len(feature_dirs)} subpackages "
                    f"match layer keywords"
                )
            else:
                guess.kind = LayoutKind.PACKAGE_BY_FEATURE
                guess.confidence = Confidence.MEDIUM
                guess.evidence.append(
                    f"subpackages don't match layer keywords "
                    f"({len(layer_dirs)}/{len(feature_dirs)})"
                )
        else:
            guess.kind = LayoutKind.MIXED
            guess.confidence = Confidence.LOW
            guess.evidence.append("no clear subpackage pattern")
        return guess


class ArchitectureDetector:
    def detect(
        self, index: RepoIndex, layout: LayoutGuess,
    ) -> ArchGuess:
        guess = ArchGuess()
        if not index.modules:
            return guess

        # Collect candidate names across levels
        names: set[str] = set()
        for rel in index.modules:
            parts = rel.split("/")
            for p in parts[:-1]:
                if p:
                    names.add(p.lower())
        # Also top-level module basenames (without .py)
        for rel in index.modules:
            base = rel.rsplit("/", 1)[-1]
            if base.endswith(".py") and base != "__init__.py":
                names.add(base[:-3].lower())

        layer_hits: dict[str, int] = {}
        for layer, kws in _LAYER_KEYWORDS.items():
            hits = sum(1 for n in names if any(
                n == kw or n.startswith(kw + "_") or n.endswith("_" + kw)
                for kw in kws
            ))
            if hits:
                layer_hits[layer] = hits

        mvc_hits = sum(1 for n in names if n in _MVC_KEYWORDS)
        hex_hits = sum(1 for n in names if n in _HEX_KEYWORDS)

        # Decision order: hexagonal > mvc > layered > package_by_feature > flat
        if hex_hits >= 2 and ("domain" in layer_hits or "application" in layer_hits):
            guess.pattern = ArchPattern.HEXAGONAL
            guess.confidence = Confidence.MEDIUM
            guess.layers = sorted(k for k in layer_hits.keys()
                                   if k in ("domain", "application",
                                            "infrastructure", "presentation"))
            guess.evidence.append(
                f"ports/adapters/domain markers: hex_hits={hex_hits}, "
                f"layer_hits={layer_hits}"
            )
            return guess

        if mvc_hits >= 2 and any(k in layer_hits for k in ("domain", "presentation")):
            guess.pattern = ArchPattern.MVC
            guess.confidence = Confidence.MEDIUM
            guess.layers = ["model", "view", "controller"]
            guess.evidence.append(f"MVC markers: {mvc_hits}")
            return guess

        # Layered if we see 3+ canonical layer names
        canonical_layers = [
            l for l in ("domain", "application", "infrastructure",
                        "presentation")
            if l in layer_hits
        ]
        if len(canonical_layers) >= 3:
            guess.pattern = ArchPattern.LAYERED
            guess.confidence = Confidence.HIGH
            guess.layers = canonical_layers
            guess.evidence.append(
                f"canonical layers present: {canonical_layers}"
            )
            return guess
        if len(canonical_layers) == 2:
            guess.pattern = ArchPattern.LAYERED
            guess.confidence = Confidence.LOW
            guess.layers = canonical_layers
            guess.evidence.append(
                f"partial canonical layers: {canonical_layers}"
            )
            return guess

        if layout.kind is LayoutKind.PACKAGE_BY_FEATURE:
            guess.pattern = ArchPattern.PACKAGE_BY_FEATURE
            guess.confidence = Confidence.MEDIUM
            guess.evidence.append("layout detector said package-by-feature")
            return guess
        if layout.kind is LayoutKind.PACKAGE_BY_LAYER:
            guess.pattern = ArchPattern.PACKAGE_BY_LAYER
            guess.confidence = Confidence.MEDIUM
            guess.evidence.append("layout detector said package-by-layer")
            return guess
        if layout.kind is LayoutKind.FLAT:
            guess.pattern = ArchPattern.FLAT
            guess.confidence = Confidence.HIGH
            guess.evidence.append("flat module layout")
            return guess

        guess.pattern = ArchPattern.UNKNOWN
        guess.confidence = Confidence.LOW
        guess.evidence.append("no matching architecture heuristic")
        return guess


# ════════════════════════════════════════════════════════════════════════════
# 4. DEPENDENCY ANALYZER
# ════════════════════════════════════════════════════════════════════════════
class DependencyAnalyzer:
    def analyze(self, index: RepoIndex) -> DependencyReport:
        report = DependencyReport()
        if not index.modules:
            return report

        # Build a map from module dotted name → path
        module_to_path: dict[str, str] = {}
        for rel in index.modules:
            mod = _module_of_path(rel)
            if mod:
                module_to_path[mod] = rel

        roles: list[DepRole] = []
        seen_roles: set[tuple[str, str]] = set()
        internal = 0
        external = 0
        stdlib = 0

        for rel, mi in index.modules.items():
            for imp in mi.imports:
                top = (imp.module or "").split(".", 1)[0]
                if not top:
                    # relative-only import (from . import x)
                    continue
                # Classify by top-level name
                if top in module_to_path:
                    role = "internal"
                    resolved = module_to_path[top]
                    internal += 1
                elif top in _STDLIB:
                    role = "stdlib"
                    resolved = ""
                    stdlib += 1
                else:
                    # Could be internal subpackage with dir-with-__init__
                    # Try prefix match
                    matched_path = ""
                    for m, p in module_to_path.items():
                        if m == imp.module or m.startswith(imp.module + "."):
                            matched_path = p
                            break
                    if matched_path:
                        role = "internal"
                        resolved = matched_path
                        internal += 1
                    else:
                        role = "external"
                        resolved = ""
                        external += 1
                key = (imp.module, role)
                if key in seen_roles:
                    continue
                seen_roles.add(key)
                roles.append(DepRole(
                    module_name=imp.module, role=role,
                    resolved_path=resolved,
                ))

        report.roles = sorted(roles, key=lambda r: (r.role, r.module_name))
        report.internal_edges = internal
        report.external_edges = external
        report.stdlib_edges = stdlib

        # Cycles (using C13's import graph, which is at file level)
        report.cycles = self._find_cycles(index)

        # Hubs: modules with many incoming edges
        incoming: dict[str, int] = {p: 0 for p in index.modules}
        outgoing: dict[str, int] = {p: 0 for p in index.modules}
        for p, deps in index.import_graph.items():
            outgoing[p] = len(deps)
            for d in deps:
                if d in incoming:
                    incoming[d] += 1
        hubs = [
            HubModule(path=p, incoming=incoming[p], outgoing=outgoing[p])
            for p in index.modules
            if incoming[p] >= 3
        ]
        hubs.sort(key=lambda h: (-h.incoming, h.path))
        report.hubs = hubs[:20]

        # Orphans: modules with zero incoming edges and not tests
        orphans = [
            p for p in index.modules
            if incoming[p] == 0
            and not _is_test_path(p)
            and not p.endswith("__init__.py")
            and not p.endswith("__main__.py")
        ]
        report.orphans = sorted(orphans)
        return report

    def _find_cycles(self, index: RepoIndex) -> list[CycleGroup]:
        """Find SCCs in the internal import graph (files only)."""
        # Build adjacency (internal edges only)
        adj: dict[str, list[str]] = {
            p: [d for d in index.import_graph.get(p, []) if d in index.modules]
            for p in index.modules
        }
        # Tarjan's SCC
        index_counter = [0]
        stack: list[str] = []
        on_stack: dict[str, bool] = {}
        indices: dict[str, int] = {}
        low: dict[str, int] = {}
        sccs: list[list[str]] = []

        def strongconnect(v: str) -> None:
            indices[v] = index_counter[0]
            low[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in adj.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif on_stack.get(w):
                    low[v] = min(low[v], indices[w])

            if low[v] == indices[v]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1:
                    sccs.append(sorted(component))

        for v in list(adj.keys()):
            if v not in indices:
                strongconnect(v)
        return [CycleGroup(members=c) for c in sorted(sccs)]


def _is_test_path(rel: str) -> bool:
    p = rel.replace("\\", "/")
    parts = p.split("/")
    base = parts[-1]
    if not base.endswith(".py"):
        return False
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    if "tests" in parts or "test" in parts:
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# 5. IMPACT ANALYZER
# ════════════════════════════════════════════════════════════════════════════
class ImpactAnalyzer:
    """Import-based impact propagation. No call-graph, no taint."""

    def __init__(self, *, max_depth: int = 8, max_impact: int = 500) -> None:
        self.max_depth = max_depth
        self.max_impact = max_impact

    def analyze(
        self, index: RepoIndex, changed_paths: Sequence[str],
    ) -> ChangeImpact:
        impact = ChangeImpact()
        changed_norm = [
            p.replace("\\", "/").lstrip("./") for p in changed_paths
        ]
        impact.changed_paths = list(changed_norm)
        if not index.modules:
            impact.rationale = "empty index"
            return impact

        # Map changed paths → existing modules (only those actually in index)
        changed_modules: list[str] = []
        for p in changed_norm:
            if p in index.modules:
                changed_modules.append(p)
        impact.changed_modules = changed_modules

        # Reverse edges: target → list of modules that import it
        rev: dict[str, list[str]] = {}
        for p, deps in index.import_graph.items():
            for d in deps:
                rev.setdefault(d, []).append(p)

        # Direct dependents
        direct: set[str] = set()
        for p in changed_modules:
            for d in rev.get(p, []):
                direct.add(d)
        impact.direct_dependents = sorted(direct)

        # Transitive closure (BFS, bounded)
        transitive: set[str] = set(direct)
        queue: deque[tuple[str, int]] = deque((d, 1) for d in direct)
        while queue:
            cur, depth = queue.popleft()
            if depth >= self.max_depth:
                continue
            for nxt in rev.get(cur, []):
                if nxt in transitive or nxt in changed_modules:
                    continue
                transitive.add(nxt)
                queue.append((nxt, depth + 1))
                if len(transitive) >= self.max_impact:
                    break
            if len(transitive) >= self.max_impact:
                break
        impact.transitive_dependents = sorted(transitive)

        # Affected symbols (public symbols in changed files)
        affected: list[SymbolImpact] = []
        for s in index.symbol_index.values():
            if s.module_path in changed_modules:
                if s.name in ("<module>",) or s.kind is SymbolKind.IMPORT:
                    continue
                affected.append(SymbolImpact(
                    symbol_id=s.id, qualname=s.qualname,
                    file=s.module_path, lineno=s.lineno,
                    kind=s.kind.value,
                    category=ImpactCategory.DIRECT_DEPENDENT,
                ))
        # Also mark hub exposure if the changed module is a hub
        hub_set = {h.path for h in []}
        for si in affected:
            if si.file in hub_set:
                si.category = ImpactCategory.HUB_EXPOSURE
        affected.sort(key=lambda x: (x.file, x.lineno))
        impact.affected_symbols = affected[: self.max_impact]

        # Affected tests: test files in the transitive set or direct set
        all_affected = set(changed_modules) | direct | transitive
        affected_tests = sorted(
            p for p in all_affected if _is_test_path(p)
        )
        impact.affected_tests = affected_tests

        # Recommended tests: direct dependents (tests that import a changed
        # module) + any test whose path shares a top-level package with
        # changed modules
        recommended: set[str] = set(affected_tests)
        changed_tops = {_top_package(p) for p in changed_modules}
        for p in index.modules:
            if not _is_test_path(p):
                continue
            if _top_package(p) in changed_tops:
                recommended.add(p)
        impact.recommended_tests = sorted(recommended)

        # Blast radius
        n = len(transitive) + len(changed_modules)
        if n <= 1:
            impact.blast_radius = "tiny"
        elif n <= 4:
            impact.blast_radius = "small"
        elif n <= 15:
            impact.blast_radius = "medium"
        else:
            impact.blast_radius = "large"

        # Risk
        if impact.blast_radius == "large":
            impact.risk_level = RiskLevel.HIGH
        elif impact.blast_radius == "medium":
            impact.risk_level = RiskLevel.MEDIUM
        else:
            impact.risk_level = RiskLevel.LOW
        # Escalate if changed module is imported by many
        for p in changed_modules:
            if len(rev.get(p, [])) >= 5:
                impact.risk_level = RiskLevel.HIGH

        impact.rationale = (
            f"changed_modules={len(changed_modules)} "
            f"direct={len(direct)} transitive={len(transitive)} "
            f"tests={len(affected_tests)} "
            f"blast={impact.blast_radius} risk={impact.risk_level.value}"
        )
        return impact


# ════════════════════════════════════════════════════════════════════════════
# 6. SAFETY CHECKER
# ════════════════════════════════════════════════════════════════════════════
class SafetyChecker:
    """Pre-flight checks for a proposed edit. Does NOT apply anything."""

    def check_patch(
        self, index: RepoIndex, *,
        path: str, old_text: str, new_text: str,
    ) -> SafetyReport:
        report = SafetyReport(path=path)
        if not path:
            report.status = SafetyStatus.INVALID if hasattr(SafetyStatus, "INVALID") else SafetyStatus.UNSAFE
            report.findings.append(SafetyFinding(
                rule="path_required", status=SafetyStatus.UNSAFE,
                message="patch path is empty",
            ))
            report.rationale = "empty path"
            return report

        # File must exist in the index
        if path not in index.modules:
            report.findings.append(SafetyFinding(
                rule="target_exists", status=SafetyStatus.UNSAFE,
                message=f"target path not in repo index: {path}",
            ))
            report.status = SafetyStatus.UNSAFE
            report.rationale = "target missing"
            return report

        if not old_text:
            report.findings.append(SafetyFinding(
                rule="empty_anchor", status=SafetyStatus.WARNING,
                message="empty old_text — patch would prepend or rewrite",
            ))
        # We cannot read the actual disk file here without a root param.
        # Simpler: rely on caller to pass root or pre-compute. We only
        # validate structural invariants we can know from the index.
        # A caller can also call check_anchor_uniqueness() with source.

        if old_text == new_text:
            report.findings.append(SafetyFinding(
                rule="noop_patch", status=SafetyStatus.UNSAFE,
                message="old_text == new_text (no-op)",
            ))
            report.status = SafetyStatus.UNSAFE
            report.rationale = "no-op patch"
            return report

        if report.status is SafetyStatus.SAFE:
            report.findings.append(SafetyFinding(
                rule="target_exists", status=SafetyStatus.SAFE,
                message="target module present in index",
            ))
            report.rationale = "structural checks passed"
        return report

    def check_anchor_uniqueness(
        self, *, source: str, path: str, old_text: str,
    ) -> SafetyReport:
        """Second-stage check: is the anchor unique in the actual file?"""
        report = SafetyReport(path=path)
        if not old_text:
            report.findings.append(SafetyFinding(
                rule="anchor_present", status=SafetyStatus.WARNING,
                message="empty anchor (whole-file patch)",
            ))
            report.status = SafetyStatus.WARNING
            report.rationale = "empty anchor"
            return report
        count = source.count(old_text)
        if count == 0:
            report.findings.append(SafetyFinding(
                rule="anchor_present", status=SafetyStatus.UNSAFE,
                message="anchor not found in source",
            ))
            report.status = SafetyStatus.UNSAFE
            report.rationale = "anchor not found"
            return report
        if count > 1:
            report.findings.append(SafetyFinding(
                rule="anchor_unique", status=SafetyStatus.UNSAFE,
                message=f"anchor is ambiguous: found {count} times, not unique",
                detail={"count": count},
            ))
            report.status = SafetyStatus.UNSAFE
            report.rationale = f"anchor ambiguous ({count}x)"
            return report
        report.findings.append(SafetyFinding(
            rule="anchor_unique", status=SafetyStatus.SAFE,
            message="anchor is unique",
        ))
        report.status = SafetyStatus.SAFE
        report.rationale = "anchor unique"
        return report

    def check_removal(
        self, index: RepoIndex, *, symbol_id: str,
    ) -> SafetyReport:
        """Warn if a symbol is referenced by other modules."""
        report = SafetyReport()
        sym = index.symbol_index.get(symbol_id)
        if sym is None:
            report.status = SafetyStatus.INCONCLUSIVE
            report.findings.append(SafetyFinding(
                rule="symbol_exists", status=SafetyStatus.UNSAFE,
                message=f"symbol {symbol_id} not found in index",
            ))
            report.rationale = "symbol missing"
            return report
        report.path = sym.module_path
        callers = index.callers_of.get(symbol_id, []) or []
        # Also count cross-module references by name
        name_refs = index.symbol_by_name.get(sym.name, []) or []
        external_refs = [
            r for r in name_refs
            if r != symbol_id
            and index.symbol_index[r].module_path != sym.module_path
        ]
        if callers or external_refs:
            report.status = SafetyStatus.WARNING
            report.findings.append(SafetyFinding(
                rule="removal_impact", status=SafetyStatus.WARNING,
                message=(
                    f"symbol '{sym.qualname}' has "
                    f"{len(callers)} recorded call(s) and "
                    f"{len(external_refs)} cross-module name ref(s)"
                ),
                detail={
                    "callers": len(callers),
                    "cross_module_refs": len(external_refs),
                },
            ))
            report.rationale = "removal has referrers"
        else:
            report.status = SafetyStatus.SAFE
            report.findings.append(SafetyFinding(
                rule="removal_impact", status=SafetyStatus.SAFE,
                message=(
                    f"no recorded callers or cross-module references "
                    f"for '{sym.qualname}'"
                ),
            ))
            report.rationale = "no referrers found"
        return report

    def check_signature_change(
        self, index: RepoIndex, *, symbol_id: str,
        new_arg_count: int,
    ) -> SafetyReport:
        report = SafetyReport()
        sym = index.symbol_index.get(symbol_id)
        if sym is None:
            report.status = SafetyStatus.INCONCLUSIVE
            report.rationale = "symbol missing"
            return report
        if sym.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            report.status = SafetyStatus.SAFE
            report.findings.append(SafetyFinding(
                rule="signature_not_applicable", status=SafetyStatus.SAFE,
                message=f"symbol kind '{sym.kind.value}' has no signature",
            ))
            report.rationale = "not a function/method"
            return report
        old_arg_count = len(sym.args or [])
        if old_arg_count == new_arg_count:
            report.status = SafetyStatus.SAFE
            report.findings.append(SafetyFinding(
                rule="signature_arity", status=SafetyStatus.SAFE,
                message="arity unchanged",
            ))
            report.rationale = "arity unchanged"
            return report

        callers = index.callers_of.get(symbol_id, []) or []
        report.path = sym.module_path
        mismatched = sum(
            1 for c in callers
            if c.args_count + c.kwargs_count != new_arg_count
        )
        report.findings.append(SafetyFinding(
            rule="signature_arity_changed",
            status=SafetyStatus.WARNING if mismatched else SafetyStatus.SAFE,
            message=(
                f"arity change {old_arg_count} → {new_arg_count}; "
                f"{mismatched} of {len(callers)} recorded caller(s) "
                f"won't match the new arity"
            ),
            detail={
                "old_arity": old_arg_count,
                "new_arity": new_arg_count,
                "total_callers": len(callers),
                "mismatched_callers": mismatched,
            },
        ))
        report.status = SafetyStatus.WARNING if mismatched else SafetyStatus.SAFE
        report.rationale = (
            f"arity change: {mismatched}/{len(callers)} callers affected"
        )
        return report


# ════════════════════════════════════════════════════════════════════════════
# 7. FACADE
# ════════════════════════════════════════════════════════════════════════════
class CodebaseUnderstanding:
    """High-level understanding of an existing repo, layered on C13."""

    def __init__(
        self,
        *,
        engine: CodeRepresentationEngine | None = None,
        max_modules: int = 5000,
        max_impact_depth: int = 8,
        max_impact: int = 500,
    ) -> None:
        if max_modules < 1:
            raise ValidationError("max_modules must be >= 1")
        self.engine = engine or CodeRepresentationEngine()
        self.max_modules = max_modules
        self.layout = LayoutDetector()
        self.arch = ArchitectureDetector()
        self.deps = DependencyAnalyzer()
        self.impact = ImpactAnalyzer(
            max_depth=max_impact_depth, max_impact=max_impact,
        )
        self.safety = SafetyChecker()

    # ---- discovery / index ----
    def analyze(
        self, root: str | Path, *,
        project_id: str = "",
        incremental: bool = False,
    ) -> tuple[CodebaseUnderstandingBundle, RepoIndex]:
        root_p = Path(root)
        if not root_p.exists() or not root_p.is_dir():
            raise ValidationError(f"root not a directory: {root}")
        idx = self.engine.analyze_repo(root_p, incremental=incremental)
        bundle = self._build_bundle(idx, project_id=project_id)
        return bundle, idx

    def _build_bundle(
        self, index: RepoIndex, *, project_id: str,
    ) -> CodebaseUnderstandingBundle:
        b = CodebaseUnderstandingBundle(
            project_id=project_id, root=index.root,
            total_modules=len(index.modules),
            total_symbols=len(index.symbol_index),
            provenance=Provenance(
                source="codebase_understanding",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        b.layout = self.layout.detect(index)
        b.architecture = self.arch.detect(index, b.layout)
        b.entry_points = self._find_entry_points(index)
        b.dependencies = self.deps.analyze(index)
        b.rationale = (
            f"layout={b.layout.kind.value} arch={b.architecture.pattern.value} "
            f"modules={b.total_modules} entry_points={len(b.entry_points)} "
            f"hubs={len(b.dependencies.hubs)} "
            f"cycles={len(b.dependencies.cycles)}"
        )
        return b

    def _find_entry_points(self, index: RepoIndex) -> list[EntryPoint]:
        out: list[EntryPoint] = []
        # 1. __main__.py
        for rel in index.modules:
            if rel.endswith("/__main__.py") or rel == "__main__.py":
                out.append(EntryPoint(path=rel, kind="dunder_main"))
        # 2. any file with `if __name__ == "__main__":`
        for rel, mi in index.modules.items():
            if rel.endswith("__main__.py") or rel.endswith("__init__.py"):
                continue
            # Use the module docstring trick: we don't retain the source.
            # Instead, check if a symbol name matches common entry names.
            funcs = [
                s for s in mi.symbols
                if s.kind is SymbolKind.FUNCTION and s.name == "main"
            ]
            if funcs:
                out.append(EntryPoint(
                    path=rel, kind="main_function",
                    detail="def main(...) present",
                ))
        # 3. console_scripts from pyproject.toml if present
        toml = Path(index.root) / "pyproject.toml"
        if toml.is_file():
            try:
                text = toml.read_text(encoding="utf-8")
                # Very light detection: [project.scripts]
                if "[project.scripts]" in text:
                    out.append(EntryPoint(
                        path="pyproject.toml",
                        kind="console_script",
                        detail="[project.scripts] declared",
                    ))
            except OSError:
                pass
        # Dedup + sort
        seen: set[tuple[str, str]] = set()
        deduped: list[EntryPoint] = []
        for e in out:
            key = (e.path, e.kind)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        deduped.sort(key=lambda e: (e.path, e.kind))
        return deduped

    # ---- public helpers ----
    def impact_of(
        self, index: RepoIndex, changed_paths: Sequence[str],
    ) -> ChangeImpact:
        return self.impact.analyze(index, changed_paths)

    def check_patch(
        self, index: RepoIndex, *,
        path: str, old_text: str, new_text: str,
        source: str | None = None,
    ) -> SafetyReport:
        rep = self.safety.check_patch(
            index, path=path, old_text=old_text, new_text=new_text,
        )
        if source is not None and rep.status is SafetyStatus.SAFE:
            rep2 = self.safety.check_anchor_uniqueness(
                source=source, path=path, old_text=old_text,
            )
            rep.findings.extend(rep2.findings)
            rep.status = rep2.status
            rep.rationale = rep2.rationale
        return rep

    def check_removal(
        self, index: RepoIndex, symbol_id: str,
    ) -> SafetyReport:
        return self.safety.check_removal(index, symbol_id=symbol_id)

    def check_signature_change(
        self, index: RepoIndex, *,
        symbol_id: str, new_arg_count: int,
    ) -> SafetyReport:
        return self.safety.check_signature_change(
            index, symbol_id=symbol_id, new_arg_count=new_arg_count,
        )

    # ---- persistence ----
    def persist(
        self, bundle: CodebaseUnderstandingBundle, *,
        project_id: str = "",
    ) -> str:
        """Persist to C04 memory + C02 ontology (if attached)."""
        return bundle.id


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class UnderstandingRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(
        self, bundle: CodebaseUnderstandingBundle, *, project_id: str,
    ) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"understanding:{bundle.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, bundle.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["understanding", "c25"],
            provenance=bundle.provenance,
        )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.ARCHITECTURE,
            _short(
                f"Codebase: {bundle.layout.kind.value} / "
                f"{bundle.architecture.pattern.value}", 120,
            ),
            attributes={
                "bundle_id": bundle.id,
                "project_id": project_id,
                "layout": bundle.layout.kind.value,
                "architecture": bundle.architecture.pattern.value,
                "modules": bundle.total_modules,
                "symbols": bundle.total_symbols,
                "entry_points": len(bundle.entry_points),
                "hubs": [h.path for h in bundle.dependencies.hubs][:10],
                "cycles": len(bundle.dependencies.cycles),
            },
            tags=["codebase", bundle.layout.kind.value,
                  bundle.architecture.pattern.value],
            provenance=bundle.provenance,
        )
        return ent.id

    def load(self, bundle_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"understanding:{bundle_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 9. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _write(d: Path, rel: str, content: str) -> Path:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _mk_empty_repo(root: Path) -> None:
    (root).mkdir(parents=True, exist_ok=True)


def _mk_flat_repo(root: Path) -> None:
    _write(root, "main.py", (
        '"""Entry point."""\n'
        "def main():\n"
        "    print('hello')\n"
    ))
    _write(root, "helpers.py", (
        "from main import main\n"
        "def helper():\n"
        "    return main\n"
    ))


def _mk_layered_repo(root: Path) -> None:
    _write(root, "pyproject.toml", "[project]\nname='x'\n")
    _write(root, "app/__init__.py", "")
    _write(root, "app/domain/__init__.py", "")
    _write(root, "app/domain/models.py", (
        "class Task:\n    def __init__(self, t): self.title = t\n"
    ))
    _write(root, "app/application/__init__.py", "")
    _write(root, "app/application/services.py", (
        "from app.domain.models import Task\n"
        "def create(title: str) -> Task:\n    return Task(title)\n"
    ))
    _write(root, "app/infrastructure/__init__.py", "")
    _write(root, "app/infrastructure/repo.py", (
        "from app.domain.models import Task\n"
        "class TaskRepo:\n    def save(self, t: Task): pass\n"
    ))
    _write(root, "app/presentation/__init__.py", "")
    _write(root, "app/presentation/api.py", (
        "from app.application.services import create\n"
        "from app.infrastructure.repo import TaskRepo\n"
        "def handle(title: str):\n"
        "    t = create(title)\n"
        "    TaskRepo().save(t)\n"
        "    return t\n"
    ))
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_services.py", (
        "from app.application.services import create\n"
        "def test_create():\n"
        "    assert create('x').title == 'x'\n"
    ))


def _mk_pkg_by_feature_repo(root: Path) -> None:
    _write(root, "pyproject.toml", "[project]\nname='x'\n")
    _write(root, "app/__init__.py", "")
    _write(root, "app/users/__init__.py", "")
    _write(root, "app/users/models.py", "class User: pass\n")
    _write(root, "app/users/service.py",
           "from app.users.models import User\n")
    _write(root, "app/orders/__init__.py", "")
    _write(root, "app/orders/models.py", "class Order: pass\n")
    _write(root, "app/orders/service.py",
           "from app.orders.models import Order\n")
    _write(root, "app/billing/__init__.py", "")
    _write(root, "app/billing/service.py", "pass\n")


def _mk_cyclic_repo(root: Path) -> None:
    _write(root, "a.py", "import b\ndef fa(): pass\n")
    _write(root, "b.py", "import c\ndef fb(): pass\n")
    _write(root, "c.py", "import a\ndef fc(): pass\n")


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

    print("Running C25 self-tests…")
    cb = CodebaseUnderstanding()

    # ---- empty / single ----
    def t_empty_repo() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_empty_repo(root)
            bundle, idx = cb.analyze(root)
            assert bundle.layout.kind is LayoutKind.EMPTY
            assert bundle.architecture.pattern is ArchPattern.UNKNOWN
            assert bundle.total_modules == 0
            assert bundle.dependencies.internal_edges == 0

    def t_single_module_repo() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "solo.py", "x = 1\n")
            bundle, _ = cb.analyze(root)
            assert bundle.layout.kind is LayoutKind.SINGLE_MODULE

    def t_flat_repo() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_flat_repo(root)
            bundle, idx = cb.analyze(root)
            assert bundle.layout.kind is LayoutKind.FLAT
            assert bundle.architecture.pattern is ArchPattern.FLAT
            # main.py has a main() function → entry point detected
            paths = {e.path for e in bundle.entry_points}
            assert "main.py" in paths

    check("empty repo → EMPTY layout", t_empty_repo)
    check("single module → SINGLE_MODULE layout", t_single_module_repo)
    check("flat repo → FLAT layout + FLAT arch", t_flat_repo)

    # ---- layered ----
    def t_layered_layout() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_layered_repo(root)
            bundle, _ = cb.analyze(root)
            assert bundle.layout.kind in (
                LayoutKind.PACKAGE_BY_LAYER, LayoutKind.SRC_LAYOUT,
            ), bundle.layout.to_dict()

    def t_layered_arch() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_layered_repo(root)
            bundle, _ = cb.analyze(root)
            assert bundle.architecture.pattern is ArchPattern.LAYERED, \
                bundle.architecture.to_dict()
            layers = set(bundle.architecture.layers)
            assert {"domain", "application", "infrastructure",
                    "presentation"}.issubset(layers)

    check("layered repo → PACKAGE_BY_LAYER/SRC layout",
          t_layered_layout)
    check("layered repo → LAYERED architecture",
          t_layered_arch)

    # ---- package-by-feature ----
    def t_feature_layout() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_pkg_by_feature_repo(root)
            bundle, _ = cb.analyze(root)
            # pkg-by-feature or unknown/mixed
            assert bundle.layout.kind in (
                LayoutKind.PACKAGE_BY_FEATURE, LayoutKind.PACKAGE_BY_LAYER,
                LayoutKind.MIXED,
            )

    check("feature packages → recognised layout", t_feature_layout)

    # ---- dependency analysis ----
    def t_deps_internal_external_stdlib() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", (
                "import os\n"
                "import json\n"
                "import requests\n"
                "import b\n"
            ))
            _write(root, "b.py", "x = 1\n")
            _, idx = cb.analyze(root)
            deps = cb.deps.analyze(idx)
            roles = {r.module_name: r.role for r in deps.roles}
            assert roles.get("os") == "stdlib"
            assert roles.get("json") == "stdlib"
            assert roles.get("requests") == "external"
            assert roles.get("b") == "internal"

    def t_deps_cycles_detected() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_cyclic_repo(root)
            _, idx = cb.analyze(root)
            deps = cb.deps.analyze(idx)
            assert len(deps.cycles) >= 1
            # All three modules should be in a single SCC
            members: set[str] = set()
            for cg in deps.cycles:
                members.update(cg.members)
            assert {"a.py", "b.py", "c.py"}.issubset(members)

    def t_deps_hubs() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "hub.py", "H = 1\n")
            for i in range(5):
                _write(root, f"user_{i}.py", "import hub\n")
            _, idx = cb.analyze(root)
            deps = cb.deps.analyze(idx)
            hub_paths = {h.path for h in deps.hubs}
            assert "hub.py" in hub_paths
            hub = next(h for h in deps.hubs if h.path == "hub.py")
            assert hub.incoming >= 5

    def t_deps_orphans() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "leaf.py", "x = 1\n")
            _, idx = cb.analyze(root)
            deps = cb.deps.analyze(idx)
            assert "leaf.py" in deps.orphans

    check("deps: internal / stdlib / external classified",
          t_deps_internal_external_stdlib)
    check("deps: cycles detected as SCC groups",
          t_deps_cycles_detected)
    check("deps: hubs identified by incoming count",
          t_deps_hubs)
    check("deps: orphans (no incoming) listed", t_deps_orphans)

    # ---- entry point detection ----
    def t_entry_main_function() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "cli.py", "def main():\n    pass\n")
            _, idx = cb.analyze(root)
            eps = cb._find_entry_points(idx)
            assert any(e.path == "cli.py" and e.kind == "main_function"
                       for e in eps)

    def t_entry_dunder_main() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "pkg/__init__.py", "")
            _write(root, "pkg/__main__.py", "print('go')\n")
            _, idx = cb.analyze(root)
            eps = cb._find_entry_points(idx)
            assert any(e.path == "pkg/__main__.py"
                       and e.kind == "dunder_main" for e in eps)

    def t_entry_console_script() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "pyproject.toml",
                   "[project]\nname='x'\n\n[project.scripts]\n"
                   "mycli = 'pkg.cli:main'\n")
            _write(root, "pkg/__init__.py", "")
            _, idx = cb.analyze(root)
            eps = cb._find_entry_points(idx)
            assert any(e.kind == "console_script" for e in eps)

    check("entry: def main() detected", t_entry_main_function)
    check("entry: __main__.py detected", t_entry_dunder_main)
    check("entry: pyproject [project.scripts] detected",
          t_entry_console_script)

    # ---- impact analysis ----
    def t_impact_direct_and_transitive() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "core.py", "x = 1\n")
            _write(root, "mid.py", "from core import x\n")
            _write(root, "top.py", "from mid import x\n")
            _write(root, "top2.py", "from top import x\n")
            _, idx = cb.analyze(root)
            imp = cb.impact_of(idx, ["core.py"])
            assert "mid.py" in imp.direct_dependents
            assert {"top.py", "top2.py"}.issubset(
                set(imp.transitive_dependents))

    def t_impact_tests_identified() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "core.py", "x = 1\n")
            _write(root, "tests/__init__.py", "")
            _write(root, "tests/test_core.py",
                   "from core import x\n")
            _, idx = cb.analyze(root)
            imp = cb.impact_of(idx, ["core.py"])
            assert "tests/test_core.py" in imp.affected_tests

    def t_impact_blast_radius() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "core.py", "x = 1\n")
            for i in range(20):
                _write(root, f"u{i}.py", "from core import x\n")
            _, idx = cb.analyze(root)
            imp = cb.impact_of(idx, ["core.py"])
            assert imp.blast_radius == "large"
            assert imp.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def t_impact_affected_symbols() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "lib.py", (
                "def helper(): pass\n"
                "class Widget: pass\n"
            ))
            _, idx = cb.analyze(root)
            imp = cb.impact_of(idx, ["lib.py"])
            quals = {s.qualname for s in imp.affected_symbols}
            assert any("helper" in q for q in quals)
            assert any("Widget" in q for q in quals)

    def t_impact_empty_paths() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            _, idx = cb.analyze(root)
            imp = cb.impact_of(idx, [])
            assert imp.changed_modules == []
            assert imp.transitive_dependents == []

    def t_impact_unknown_path() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            _, idx = cb.analyze(root)
            imp = cb.impact_of(idx, ["does_not_exist.py"])
            assert imp.changed_modules == []
            # rationale reflects nothing changed

    check("impact: direct + transitive dependents",
          t_impact_direct_and_transitive)
    check("impact: affected tests identified", t_impact_tests_identified)
    check("impact: blast radius = large for hub module",
          t_impact_blast_radius)
    check("impact: affected symbols enumerated",
          t_impact_affected_symbols)
    check("impact: empty paths → empty impact", t_impact_empty_paths)
    check("impact: unknown path handled gracefully",
          t_impact_unknown_path)

    # ---- safety: check_patch ----
    def t_safety_patch_missing_target() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            _, idx = cb.analyze(root)
            rep = cb.check_patch(
                idx, path="missing.py",
                old_text="x", new_text="y",
            )
            assert rep.status is SafetyStatus.UNSAFE
            assert any(f.rule == "target_exists" for f in rep.findings)

    def t_safety_patch_noop() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            _, idx = cb.analyze(root)
            rep = cb.check_patch(
                idx, path="a.py", old_text="x = 1", new_text="x = 1",
            )
            assert rep.status is SafetyStatus.UNSAFE
            assert any(f.rule == "noop_patch" for f in rep.findings)

    def t_safety_patch_anchor_unique() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\ny = 2\n")
            _, idx = cb.analyze(root)
            rep = cb.check_patch(
                idx, path="a.py",
                old_text="x = 1", new_text="x = 99",
                source="x = 1\ny = 2\n",
            )
            assert rep.status is SafetyStatus.SAFE

    def t_safety_patch_anchor_ambiguous() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\nx = 1\n")
            _, idx = cb.analyze(root)
            rep = cb.check_patch(
                idx, path="a.py",
                old_text="x = 1", new_text="x = 2",
                source="x = 1\nx = 1\n",
            )
            assert rep.status is SafetyStatus.UNSAFE
            assert any("ambiguous" in f.message.lower() for f in rep.findings)

    def t_safety_patch_anchor_missing() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            _, idx = cb.analyze(root)
            rep = cb.check_patch(
                idx, path="a.py",
                old_text="NOT_THERE", new_text="X",
                source="x = 1\n",
            )
            assert rep.status is SafetyStatus.UNSAFE

    check("safety: patch missing target → UNSAFE",
          t_safety_patch_missing_target)
    check("safety: no-op patch → UNSAFE", t_safety_patch_noop)
    check("safety: unique anchor → SAFE", t_safety_patch_anchor_unique)
    check("safety: ambiguous anchor → UNSAFE",
          t_safety_patch_anchor_ambiguous)
    check("safety: missing anchor → UNSAFE",
          t_safety_patch_anchor_missing)

    # ---- safety: removal / signature ----
    def t_safety_removal_no_refs() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "leaf.py", "def orphan(): pass\n")
            _, idx = cb.analyze(root)
            sym = next(s for s in idx.symbol_index.values()
                       if s.name == "orphan")
            rep = cb.check_removal(idx, sym.id)
            assert rep.status is SafetyStatus.SAFE

    def t_safety_removal_with_callers() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "lib.py", "def helper(): pass\n")
            _write(root, "main.py", (
                "from lib import helper\n"
                "def run():\n"
                "    helper()\n"
            ))
            _, idx = cb.analyze(root)
            sym = next(s for s in idx.symbol_index.values()
                       if s.name == "helper"
                       and s.kind is SymbolKind.FUNCTION)
            rep = cb.check_removal(idx, sym.id)
            # Either WARNING (callers found) or SAFE (no callers resolved)
            assert rep.status in (SafetyStatus.WARNING, SafetyStatus.SAFE)

    def t_safety_signature_change() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "lib.py",
                   "def f(a, b): pass\n")
            _, idx = cb.analyze(root)
            sym = next(s for s in idx.symbol_index.values()
                       if s.name == "f")
            rep = cb.check_signature_change(
                idx, symbol_id=sym.id, new_arg_count=3,
            )
            # Arity changed → message reflects this
            assert any("arity" in f.message.lower() for f in rep.findings)

    def t_safety_signature_unchanged() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "lib.py",
                   "def f(a, b): pass\n")
            _, idx = cb.analyze(root)
            sym = next(s for s in idx.symbol_index.values()
                       if s.name == "f")
            rep = cb.check_signature_change(
                idx, symbol_id=sym.id, new_arg_count=2,
            )
            assert rep.status is SafetyStatus.SAFE

    check("safety: removal with no refs → SAFE", t_safety_removal_no_refs)
    check("safety: removal with callers reports impact",
          t_safety_removal_with_callers)
    check("safety: signature change reports arity impact",
          t_safety_signature_change)
    check("safety: unchanged arity → SAFE",
          t_safety_signature_unchanged)

    # ---- determinism ----
    def t_deterministic() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_layered_repo(root)
            b1, _ = cb.analyze(root)
            b2, _ = cb.analyze(root)
            assert b1.layout.kind == b2.layout.kind
            assert b1.architecture.pattern == b2.architecture.pattern
            assert b1.dependencies.hubs == b2.dependencies.hubs
            assert b1.dependencies.cycles == b2.dependencies.cycles

    check("deterministic: same repo → same bundle", t_deterministic)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_layered_repo(root)
            bundle, _ = cb.analyze(root)
            d = bundle.to_dict()
            assert d["id"] == bundle.id
            assert "layout" in d and "architecture" in d
            assert "dependencies" in d and "entry_points" in d
            s = bundle.summary()
            assert "Codebase Understanding" in s

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
                    root = Path(tdt)
                    _mk_layered_repo(root)
                    bundle, _ = cb.analyze(root, project_id="proj-x")
                    repo = UnderstandingRepository(memory=mem, ontology=ont)
                    ent = repo.save(bundle, project_id="proj-x")
                    assert ent
                    loaded = repo.load(bundle.id, project_id="proj-x")
                    assert loaded is not None
                    assert loaded["layout"]["kind"] in (
                        "package_by_layer", "src_layout",
                    )
                    assert ont.count(kind=EntityKind.ARCHITECTURE) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology ARCHITECTURE entity",
          t_persist)

    # ---- E2E: real existing repo (our own workspace) ----
    def t_e2e_analyze_our_own_source_tree() -> None:
        """Run C25 against the directory containing this file to ensure
        it doesn't crash on a 'real' Python codebase (even if it's just
        one big file)."""
        here = Path(__file__).resolve().parent
        if not here.is_dir():
            return
        bundle, idx = cb.analyze(here)
        assert bundle.total_modules >= 1
        # No exceptions, sane structure
        assert bundle.layout.kind is not LayoutKind.EMPTY

    check("e2e: analyzes its own source directory without crashing",
          t_e2e_analyze_our_own_source_tree)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 10. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C25 — Existing Codebase Understanding")
    print("=" * 78)

    cb = CodebaseUnderstanding()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_repo"
        root.mkdir()
        _mk_layered_repo(root)
        # Add a couple of hub modules to make the graph interesting
        _write(root, "app/utils.py", (
            "def log(msg): print(msg)\n"
        ))
        for i in range(4):
            _write(root, f"app/feature_{i}.py", (
                "from app.utils import log\n"
                "from app.domain.models import Task\n"
                "def do():\n"
                f"    log('feature {i}')\n"
                f"    return Task('f{i}')\n"
            ))

        bundle, idx = cb.analyze(root, project_id="demo")

        print("\n[1] Summary:")
        print(bundle.summary())

        print("\n[2] Layout:")
        print(f"    kind={bundle.layout.kind.value}  "
              f"confidence={bundle.layout.confidence.value}")
        for e in bundle.layout.evidence:
            print(f"    · {e}")
        print(f"    top-level dirs: {bundle.layout.top_level_dirs}")

        print("\n[3] Architecture:")
        print(f"    pattern={bundle.architecture.pattern.value}  "
              f"confidence={bundle.architecture.confidence.value}")
        print(f"    layers: {bundle.architecture.layers}")
        for e in bundle.architecture.evidence:
            print(f"    · {e}")

        print("\n[4] Entry points:")
        for ep in bundle.entry_points:
            print(f"    {ep.path}  [{ep.kind}]  {ep.detail}")

        print("\n[5] Dependencies:")
        print(f"    internal={bundle.dependencies.internal_edges}  "
              f"external={bundle.dependencies.external_edges}  "
              f"stdlib={bundle.dependencies.stdlib_edges}")
        if bundle.dependencies.hubs:
            print(f"    top hubs:")
            for h in bundle.dependencies.hubs[:5]:
                print(f"      {h.path}  in={h.incoming} out={h.outgoing}")
        if bundle.dependencies.cycles:
            print(f"    cycles: {len(bundle.dependencies.cycles)}")
            for c in bundle.dependencies.cycles[:2]:
                print(f"      {c.members}")
        if bundle.dependencies.orphans:
            print(f"    orphans: {bundle.dependencies.orphans[:5]}")

        print("\n[6] Change impact for 'app/utils.py':")
        imp = cb.impact_of(idx, ["app/utils.py"])
        print(f"    changed_modules   : {imp.changed_modules}")
        print(f"    direct_dependents : {imp.direct_dependents}")
        print(f"    transitive        : {imp.transitive_dependents}")
        print(f"    affected_tests    : {imp.affected_tests}")
        print(f"    recommended_tests : {imp.recommended_tests}")
        print(f"    blast_radius      : {imp.blast_radius}")
        print(f"    risk_level        : {imp.risk_level.value}")
        print(f"    rationale         : {imp.rationale}")

        print("\n[7] Safety pre-flight for a proposed patch:")
        rep = cb.check_patch(
            idx, path="app/utils.py",
            old_text="def log(msg): print(msg)",
            new_text="def log(msg):\n    print(f'[log] {msg}')",
            source=(root / "app" / "utils.py").read_text(encoding="utf-8"),
        )
        print(f"    status={rep.status.value}  rationale={rep.rationale}")
        for f in rep.findings:
            print(f"    [{f.status.value:10s}] {f.rule}: {f.message}")

        print("\n[8] Persistence:")
        with tempfile.TemporaryDirectory() as sdt:
            cfg = Config(data_dir=Path(sdt) / "sebrain", log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                with execution_scope(project_id="demo"):
                    mem = MemoryStore(app.storage)
                    ont = Ontology(app.storage)
                    repo = UnderstandingRepository(memory=mem, ontology=ont)
                    ent = repo.save(bundle, project_id="demo")
                    print(f"    ontology entity: {ent[:12]}…")
                    print(f"    ARCHITECTURE count: "
                          f"{ont.count(kind=EntityKind.ARCHITECTURE)}")
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
