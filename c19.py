"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C19 — DEBUGGING ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C13, C16, C18.

Purpose:
    Turn a failure observation into a STRUCTURED diagnosis:
        Failure Observation
            → Evidence Collection
            → Classification (11 categories)
            → Localization (deepest user frame + symbol + snippet)
            → Root Cause Candidates (ranked)
            → Root Cause Evaluation

Capabilities:
    - Inputs: C18 TestResult | C16 SandboxResult | raw traceback/text
    - Signature extraction (exception_type, message, exit_code, signal)
    - Category classification
        SYNTAX · TYPE · LOGIC · RUNTIME · DEPENDENCY · CONFIG ·
        ENVIRONMENT · INTEGRATION · CONCURRENCY · RESOURCE · SECURITY · UNKNOWN
    - Frame localization:
        * Parse Python traceback frames
        * Skip stdlib / site-packages → deepest USER frame
        * Read source snippet (±2 lines) from disk (bounded)
        * Look up enclosing symbol via C13 RepoIndex (optional)
    - Ranked root cause candidates with suggested probes
    - Severity mapping per category
    - Status: DIAGNOSED | PARTIAL | UNDIAGNOSED
    - Persistence to C04 memory + C02 ontology (BUG, FAILURE, ROOT_CAUSE)

Invariants honored:
    - NO external LLM. Deterministic rules only.
    - NEVER auto-regenerates code (that's C20's job)
    - Every candidate is explicit about its confidence and evidence
    - Every diagnosis carries structured evidence (kind + text + weight)
    - Snippet reads are bounded (max bytes) and never raise
    - Sandbox signals (SIGXCPU/SIGXFSZ/SIGKILL) map to specific categories
    - Same input → same report (deterministic ranking)

Explicit limitations:
    - Does not execute the code to reproduce. It analyses the observation.
    - Message→root-cause heuristics are pattern-based, not causal proofs.
      The 'confidence' field reflects that (rarely VERIFIED).
    - Only Python tracebacks. Other languages → C30.
    - Localization relies on frames actually appearing in the traceback.
      Silent failures (e.g. no traceback, only exit code) yield PARTIAL.

Contents:
  1.  Enums: FailureCategory, Severity, DebugStatus, InputKind
  2.  Dataclasses: Evidence, LocalizedFrame, FailureSignature,
                   RootCauseCandidate, DebugReport
  3.  Traceback parser
  4.  Exception → category classifier (+ refinements)
  5.  Localizer (frame selection, snippet read, C13 symbol lookup)
  6.  Candidate generator (per category)
  7.  Debugger facade (analyze_test_result / analyze_sandbox_result /
                       analyze_text)
  8.  DebugRepository (persist BUG / FAILURE / ROOT_CAUSE)
  9.  Self-tests (~30)
 10.  Demo

Run as script:
    python -m sebrain.c19            # demo
    python -m sebrain.c19 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import re
import signal as _signal
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

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
    RelationKind,
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


def _short(s: str, n: int = 100) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class FailureCategory(str, Enum):
    SYNTAX = "syntax"
    TYPE = "type"
    LOGIC = "logic"
    RUNTIME = "runtime"
    DEPENDENCY = "dependency"
    CONFIG = "config"
    ENVIRONMENT = "environment"
    INTEGRATION = "integration"
    CONCURRENCY = "concurrency"
    RESOURCE = "resource"
    SECURITY = "security"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DebugStatus(str, Enum):
    DIAGNOSED = "diagnosed"        # category + localized + candidates
    PARTIAL = "partial"            # some fields missing
    UNDIAGNOSED = "undiagnosed"    # cannot classify


class InputKind(str, Enum):
    TEST_RESULT = "test_result"      # from C18 TestResult
    SANDBOX_RESULT = "sandbox_result"  # from C16 SandboxResult
    RAW = "raw"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Evidence:
    kind: str
    text: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "text": self.text,
            "weight": self.weight, "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LocalizedFrame:
    file: str
    lineno: int
    function: str
    snippet: str = ""
    is_user_code: bool = True
    symbol_id: str | None = None
    symbol_qualname: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file, "lineno": self.lineno,
            "function": self.function, "snippet": self.snippet,
            "is_user_code": self.is_user_code,
            "symbol_id": self.symbol_id,
            "symbol_qualname": self.symbol_qualname,
        }


@dataclass(slots=True)
class FailureSignature:
    exception_type: str = ""
    exception_message: str = ""
    exit_code: int = 0
    signal_number: int = -1
    frame_count: int = 0
    raw_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "exit_code": self.exit_code,
            "signal_number": self.signal_number,
            "frame_count": self.frame_count,
            "raw_excerpt": self.raw_excerpt,
        }


@dataclass(slots=True)
class RootCauseCandidate:
    category: FailureCategory
    description: str
    confidence: Confidence = Confidence.MEDIUM
    score: float = 0.0
    rationale: str = ""
    suggested_probe: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "confidence": self.confidence.value,
            "score": float(self.score),
            "rationale": self.rationale,
            "suggested_probe": self.suggested_probe,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(slots=True)
class DebugReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    input_kind: InputKind = InputKind.RAW
    input_ref: str = ""
    category: FailureCategory = FailureCategory.UNKNOWN
    severity: Severity = Severity.LOW
    status: DebugStatus = DebugStatus.UNDIAGNOSED
    category_rationale: str = ""
    signature: FailureSignature = field(default_factory=FailureSignature)
    frames: list[LocalizedFrame] = field(default_factory=list)
    localized: LocalizedFrame | None = None
    evidence: list[Evidence] = field(default_factory=list)
    candidates: list[RootCauseCandidate] = field(default_factory=list)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def best_candidate(self) -> RootCauseCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "input_kind": self.input_kind.value,
            "input_ref": self.input_ref,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "category_rationale": self.category_rationale,
            "signature": self.signature.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "localized": self.localized.to_dict() if self.localized else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "candidates": [c.to_dict() for c in self.candidates],
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        loc = (
            f"{self.localized.file}:{self.localized.lineno}"
            if self.localized else "<none>"
        )
        best = self.best_candidate()
        best_line = (
            f"[{best.category.value}] {_short(best.description, 80)} "
            f"({best.confidence.value})"
            if best else "<none>"
        )
        return (
            "=== Debug Report ===\n"
            f"status={self.status.value}  category={self.category.value}  "
            f"severity={self.severity.value}\n"
            f"signature={self.signature.exception_type}: "
            f"{_short(self.signature.exception_message, 80)}\n"
            f"localized={loc}\n"
            f"best_candidate={best_line}\n"
            f"candidates={len(self.candidates)}  "
            f"evidence={len(self.evidence)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. TRACEBACK PARSER
# ════════════════════════════════════════════════════════════════════════════
_TB_HEADER_RE = re.compile(r"^\s*Traceback \(most recent call last\):")
_TB_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)(?:,\s*in\s+(?P<func>\S+))?\s*$'
)
# A "TypeError: msg" or "asyncio.TimeoutError: msg" line.
_TB_EXC_RE = re.compile(
    r"^(?P<type>[A-Za-z_][\w\.]*)(?::\s*(?P<msg>.*))?\s*$"
)
# Pytest's short failure form used by C18's structured reporter, e.g.
# ``tests/test_calc.py:4: AssertionError`` or with a message appended.
_PYTEST_FAILURE_RE = re.compile(
    r"^(?P<file>.+):(?P<line>\d+):\s*(?P<type>[A-Za-z_][\w\.]*)"
    r"(?:\s*:\s*(?P<msg>.*))?$"
)


def _strip_pytest_prefix(line: str) -> str:
    """Pytest prepends failure lines with 'E   ' (and indentation)."""
    s = line.rstrip()
    if s.lstrip().startswith("E"):
        s = s.lstrip()[1:].lstrip()
    return s


def parse_traceback(text: str) -> tuple[FailureSignature, list[LocalizedFrame]]:
    """Best-effort traceback parser.

    Returns (signature, frames). Never raises. If no traceback header is
    present, returns a signature with just the last non-empty line as the
    exception message and an empty frame list.
    """
    if not text or not isinstance(text, str):
        return FailureSignature(), []

    lines = text.splitlines()
    sig = FailureSignature(raw_excerpt=text[-500:])

    # Find the last traceback block (pytest may emit several)
    header_idx = -1
    for i, ln in enumerate(lines):
        if _TB_HEADER_RE.match(ln):
            header_idx = i

    frames: list[LocalizedFrame] = []
    exc_line = ""

    if header_idx >= 0:
        # Scan from header to end, extracting frames + exception line
        i = header_idx + 1
        while i < len(lines):
            ln = lines[i]
            m = _TB_FRAME_RE.match(ln)
            if m:
                frames.append(LocalizedFrame(
                    file=m.group("file"),
                    lineno=int(m.group("line")),
                    function=(m.group("func") or "<module>"),
                ))
                # Skip the source-line body if present (next line, indented
                # and not another File/last exception)
                i += 1
                continue
            stripped = _strip_pytest_prefix(ln)
            m2 = _TB_EXC_RE.match(stripped.strip())
            if m2 and m2.group("type"):
                # Ignore obvious non-exception lines
                t = m2.group("type")
                if not t.startswith("File") and not t.startswith("Traceback"):
                    exc_line = stripped.strip()
            i += 1
    else:
        # No header — look for a final exception-looking line
        for ln in reversed(lines):
            stripped = _strip_pytest_prefix(ln).strip()
            if not stripped:
                continue
            pm = _PYTEST_FAILURE_RE.match(stripped)
            if pm:
                sig.exception_type = pm.group("type") or ""
                sig.exception_message = pm.group("msg") or ""
                frames.append(LocalizedFrame(
                    file=pm.group("file"),
                    lineno=int(pm.group("line")),
                    function="<pytest>_failure",
                ))
                break
            m = _TB_EXC_RE.match(stripped)
            if m and m.group("type") and ":" in stripped:
                exc_line = stripped
                break

    # Split exception line
    if exc_line:
        parts = exc_line.split(":", 1)
        sig.exception_type = parts[0].strip()
        sig.exception_message = parts[1].strip() if len(parts) > 1 else ""
    sig.frame_count = len(frames)
    return sig, frames


# ════════════════════════════════════════════════════════════════════════════
# 4. CATEGORY CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════
def _sig_attr(name: str) -> int:
    """Return signal number if platform has it, else -1."""
    return getattr(_signal, name, -1)


_EXCEPTION_TABLE: dict[str, FailureCategory] = {
    # Syntax
    "SyntaxError": FailureCategory.SYNTAX,
    "IndentationError": FailureCategory.SYNTAX,
    "TabError": FailureCategory.SYNTAX,

    # Type
    "TypeError": FailureCategory.TYPE,
    "AttributeError": FailureCategory.TYPE,

    # Logic
    "ValueError": FailureCategory.LOGIC,
    "AssertionError": FailureCategory.LOGIC,
    "ZeroDivisionError": FailureCategory.LOGIC,
    "NotImplementedError": FailureCategory.LOGIC,

    # Generic runtime
    "RuntimeError": FailureCategory.RUNTIME,

    # Resource
    "MemoryError": FailureCategory.RESOURCE,
    "RecursionError": FailureCategory.RESOURCE,

    # Dependency
    "ImportError": FailureCategory.DEPENDENCY,
    "ModuleNotFoundError": FailureCategory.DEPENDENCY,

    # Config (refinable)
    "KeyError": FailureCategory.CONFIG,
    "FileNotFoundError": FailureCategory.CONFIG,
    "IsADirectoryError": FailureCategory.CONFIG,
    "NotADirectoryError": FailureCategory.CONFIG,

    # Environment
    "PermissionError": FailureCategory.ENVIRONMENT,
    "EnvironmentError": FailureCategory.ENVIRONMENT,
    "OSError": FailureCategory.ENVIRONMENT,

    # Integration
    "ConnectionError": FailureCategory.INTEGRATION,
    "ConnectionRefusedError": FailureCategory.INTEGRATION,
    "ConnectionResetError": FailureCategory.INTEGRATION,
    "ConnectionAbortedError": FailureCategory.INTEGRATION,
    "BrokenPipeError": FailureCategory.INTEGRATION,
    "TimeoutError": FailureCategory.INTEGRATION,
    "HTTPError": FailureCategory.INTEGRATION,

    # Concurrency
    "asyncio.TimeoutError": FailureCategory.CONCURRENCY,
    "CancelledError": FailureCategory.CONCURRENCY,
}


def _refine_key_error(
    msg: str, frames: Sequence[LocalizedFrame],
) -> FailureCategory:
    # If any frame lives in a config/settings/env file, keep CONFIG.
    for f in frames:
        low = f.file.lower()
        if any(k in low for k in ("config", "settings", ".env", "conftest")):
            return FailureCategory.CONFIG
    # Otherwise, a dict lookup miss is more often a logic bug.
    return FailureCategory.LOGIC


def _refine_file_not_found(msg: str) -> FailureCategory:
    low = msg.lower()
    # Extract the quoted path (if present)
    m = re.search(r"['\"]([^'\"]+)['\"]", msg)
    path = m.group(1) if m else ""
    plow = path.lower()
    # System paths → environment
    if plow.startswith(("/etc/", "/usr/", "/var/", "/library/", "/system/",
                        "/opt/", "c:\\windows")):
        return FailureCategory.ENVIRONMENT
    # Temp dirs → environment
    if plow.startswith(("/tmp/", "/var/tmp/")) or "/temp/" in plow:
        return FailureCategory.ENVIRONMENT
    # Config-like filenames → config
    base = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    if base in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
                ".env", "config.yaml", "config.yml", "config.json",
                "settings.toml"):
        return FailureCategory.CONFIG
    return FailureCategory.CONFIG


def _refine_runtime(msg: str) -> FailureCategory:
    low = msg.lower()
    if "coroutine" in low or "event loop" in low or "loop is" in low:
        return FailureCategory.CONCURRENCY
    if "dictionary changed size" in low or "thread" in low:
        return FailureCategory.CONCURRENCY
    return FailureCategory.RUNTIME


def _refine_oserror(msg: str) -> FailureCategory:
    low = msg.lower()
    if "no space" in low:
        return FailureCategory.RESOURCE
    if "too many open files" in low:
        return FailureCategory.RESOURCE
    if "permission" in low or "access is denied" in low:
        return FailureCategory.ENVIRONMENT
    if "connection" in low or "network" in low:
        return FailureCategory.INTEGRATION
    return FailureCategory.ENVIRONMENT


def classify(
    sig: FailureSignature, frames: Sequence[LocalizedFrame],
    *,
    signal_number: int = -1, exit_code: int = 0,
) -> tuple[FailureCategory, str, Severity]:
    """Classify a failure signature into a category.

    Returns (category, rationale, severity).
    Order of precedence:
      1. Signal (from sandbox)
      2. Exception type table (with refinements)
      3. Fallback heuristics on message
      4. UNKNOWN
    """
    # --- 1. Signals ---
    if signal_number > 0:
        if signal_number == _sig_attr("SIGXCPU"):
            return (FailureCategory.RESOURCE,
                    "killed by SIGXCPU (CPU limit exceeded)",
                    Severity.CRITICAL)
        if signal_number == _sig_attr("SIGXFSZ"):
            return (FailureCategory.RESOURCE,
                    "killed by SIGXFSZ (file size limit exceeded)",
                    Severity.CRITICAL)
        if signal_number == _sig_attr("SIGKILL"):
            return (FailureCategory.RESOURCE,
                    "killed by SIGKILL (memory/CPU/OOM)",
                    Severity.CRITICAL)
        if signal_number == _sig_attr("SIGSEGV"):
            return (FailureCategory.RUNTIME,
                    "segmentation fault (SIGSEGV)", Severity.HIGH)
        if signal_number == _sig_attr("SIGINT"):
            return (FailureCategory.ENVIRONMENT,
                    "interrupted by SIGINT", Severity.LOW)
        return (FailureCategory.RUNTIME,
                f"killed by signal {signal_number}", Severity.MEDIUM)

    # --- 2. Exception table ---
    exc = sig.exception_type or ""
    msg = sig.exception_message or ""

    if exc in _EXCEPTION_TABLE:
        cat = _EXCEPTION_TABLE[exc]
        # Refinements
        if exc == "KeyError":
            cat = _refine_key_error(msg, frames)
        elif exc == "FileNotFoundError":
            cat = _refine_file_not_found(msg)
        elif exc == "RuntimeError":
            cat = _refine_runtime(msg)
        elif exc in ("OSError", "EnvironmentError"):
            cat = _refine_oserror(msg)

        sev = _severity_for(cat)
        return cat, f"exception type '{exc}' → {cat.value}", sev

    # --- 3. Substring fallbacks (unknown subclasses) ---
    low_exc = exc.lower()
    if "timeout" in low_exc:
        return (FailureCategory.INTEGRATION,
                f"exception '{exc}' looks like a timeout", Severity.HIGH)
    if "connection" in low_exc:
        return (FailureCategory.INTEGRATION,
                f"exception '{exc}' looks like a connection error",
                Severity.HIGH)
    if "import" in low_exc or "module" in low_exc:
        return (FailureCategory.DEPENDENCY,
                f"exception '{exc}' looks like an import failure",
                Severity.HIGH)
    if "permission" in low_exc or "access" in low_exc:
        return (FailureCategory.ENVIRONMENT,
                f"exception '{exc}' looks like a permission issue",
                Severity.HIGH)

    # --- 4. Unknown ---
    if not exc and not msg:
        return (FailureCategory.UNKNOWN, "no exception information",
                Severity.LOW)
    return (FailureCategory.RUNTIME,
            f"unmapped exception '{exc or '<empty>'}'", Severity.MEDIUM)


def _severity_for(cat: FailureCategory) -> Severity:
    if cat in (FailureCategory.RESOURCE, FailureCategory.SECURITY):
        return Severity.CRITICAL
    if cat in (FailureCategory.SYNTAX, FailureCategory.DEPENDENCY,
               FailureCategory.CONFIG, FailureCategory.ENVIRONMENT,
               FailureCategory.INTEGRATION):
        return Severity.HIGH
    if cat in (FailureCategory.TYPE, FailureCategory.RUNTIME,
               FailureCategory.CONCURRENCY, FailureCategory.LOGIC):
        return Severity.MEDIUM
    return Severity.LOW


# ════════════════════════════════════════════════════════════════════════════
# 5. LOCALIZER
# ════════════════════════════════════════════════════════════════════════════
_MAX_SOURCE_FILE_BYTES = 500_000
_SNIPPET_CONTEXT = 2


def _is_stdlib_or_site(file_path: str) -> bool:
    p = file_path.replace("\\", "/")
    return (
        "/lib/python" in p
        or "site-packages" in p
        or p.startswith("<frozen ")
        or p.startswith("<string>")
        or p.startswith("<ipython-input")
        or p.startswith("<stdin>")
    )


def _read_snippet(abs_path: str, lineno: int,
                   context: int = _SNIPPET_CONTEXT) -> str:
    try:
        p = Path(abs_path)
        if not p.is_file():
            return ""
        if p.stat().st_size > _MAX_SOURCE_FILE_BYTES:
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    i = lineno - 1
    if i < 0 or i >= len(lines):
        return ""
    lo = max(0, i - context)
    hi = min(len(lines), i + context + 1)
    out: list[str] = []
    for k in range(lo, hi):
        marker = ">>" if k == i else "  "
        out.append(f"{marker} {k+1:>4d} | {lines[k]}")
    return "\n".join(out)


def _find_enclosing_symbol(
    repo_index: Any, module_path: str, lineno: int,
) -> tuple[str, str] | None:
    """Return (symbol_id, qualname) of the innermost symbol enclosing
    (module_path, lineno), or None.
    """
    if repo_index is None:
        return None
    sym_index = getattr(repo_index, "symbol_index", None)
    if not sym_index:
        return None
    best = None
    best_span = None
    for s in sym_index.values():
        if getattr(s, "module_path", None) != module_path:
            continue
        lo = getattr(s, "lineno", 0)
        hi = getattr(s, "end_lineno", lo) or lo
        if lo <= lineno <= hi:
            span = hi - lo
            if best_span is None or span < best_span:
                best = s
                best_span = span
    if best is None:
        return None
    return (getattr(best, "id", ""), getattr(best, "qualname", ""))


def localize(
    frames: Sequence[LocalizedFrame], *,
    root: str | Path | None = None,
    repo_index: Any = None,
) -> list[LocalizedFrame]:
    """Populate `is_user_code`, snippet, and (if index provided) symbol info.

    Returns a new list — input frames are treated as immutable.
    """
    root_path = Path(root).resolve() if root else None
    out: list[LocalizedFrame] = []
    for f in frames:
        is_user = not _is_stdlib_or_site(f.file)
        # C18 may provide repo-relative paths (pytest's normal form). Resolve
        # those against the supplied repository root rather than the process
        # cwd; otherwise an otherwise-valid frame cannot be localized when
        # the caller runs from another directory.
        original_file = Path(f.file)
        file_for_snippet = original_file
        if root_path is not None and not file_for_snippet.is_absolute():
            file_for_snippet = root_path / file_for_snippet
        try:
            file_for_snippet = file_for_snippet.resolve()
        except OSError:
            pass
        snippet = _read_snippet(str(file_for_snippet), f.lineno)
        # When C18/pytest gives us a repo-relative frame, expose the resolved
        # repository path to downstream consumers. This makes localization
        # deterministic regardless of the caller's working directory and also
        # aligns the e2e contract with absolute traceback paths.
        display_file = (
            str(file_for_snippet)
            if root_path is not None and not original_file.is_absolute()
            else f.file
        )
        sym_id = ""
        sym_qn = ""
        if repo_index is not None and root_path is not None:
            try:
                rel = str(
                    file_for_snippet.relative_to(root_path)
                ).replace("\\", "/")
            except (OSError, ValueError):
                rel = ""
            if rel:
                found = _find_enclosing_symbol(repo_index, rel, f.lineno)
                if found:
                    sym_id, sym_qn = found
        out.append(LocalizedFrame(
            file=display_file, lineno=f.lineno, function=f.function,
            snippet=snippet, is_user_code=is_user,
            symbol_id=sym_id, symbol_qualname=sym_qn,
        ))
    return out


def pick_primary(frames: Sequence[LocalizedFrame]) -> LocalizedFrame | None:
    """Return the deepest user frame, or deepest frame overall."""
    if not frames:
        return None
    user = [f for f in frames if f.is_user_code]
    if user:
        return user[-1]
    return frames[-1]


# ════════════════════════════════════════════════════════════════════════════
# 6. CANDIDATE GENERATOR
# ════════════════════════════════════════════════════════════════════════════
_CATEGORY_BASE_WEIGHT: dict[FailureCategory, float] = {
    FailureCategory.SYNTAX: 0.90,
    FailureCategory.DEPENDENCY: 0.90,
    FailureCategory.RESOURCE: 0.85,
    FailureCategory.SECURITY: 0.70,
    FailureCategory.INTEGRATION: 0.70,
    FailureCategory.TYPE: 0.70,
    FailureCategory.CONFIG: 0.60,
    FailureCategory.ENVIRONMENT: 0.60,
    FailureCategory.CONCURRENCY: 0.50,
    FailureCategory.LOGIC: 0.50,
    FailureCategory.RUNTIME: 0.40,
    FailureCategory.UNKNOWN: 0.20,
}


def _score_to_confidence(score: float) -> Confidence:
    if score >= 0.85:
        return Confidence.HIGH
    if score >= 0.65:
        return Confidence.MEDIUM
    if score >= 0.40:
        return Confidence.LOW
    return Confidence.UNKNOWN


def _extract_missing_module(msg: str) -> str:
    """From a ModuleNotFoundError message like
    "No module named 'fastapi'" → "fastapi"
    """
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
    return m.group(1) if m else ""


def _extract_key(msg: str) -> str:
    # KeyError: 'foo'  (repr-style in message)
    m = re.search(r"^'(.+)'$", msg.strip())
    if m:
        return m.group(1)
    m = re.search(r"^\"(.+)\"$", msg.strip())
    if m:
        return m.group(1)
    return msg.strip()


def _extract_path(msg: str) -> str:
    m = re.search(r"['\"]([^'\"]+)['\"]", msg)
    return m.group(1) if m else ""


def generate_candidates(
    sig: FailureSignature, frames: Sequence[LocalizedFrame],
    localized: LocalizedFrame | None, category: FailureCategory,
) -> list[RootCauseCandidate]:
    base = _CATEGORY_BASE_WEIGHT.get(category, 0.3)
    user_frames = [f for f in frames if f.is_user_code]
    out: list[RootCauseCandidate] = []

    if localized is None:
        # Nothing to localize against; emit a single generic candidate.
        out.append(RootCauseCandidate(
            category=category,
            description=f"{category.value} failure; no localized frame",
            confidence=_score_to_confidence(base),
            score=base,
            rationale=f"category '{category.value}' inferred from signature "
                      f"but no user frame available",
            suggested_probe="Re-run with full traceback capture (--tb=long).",
            evidence_ids=["signature"],
        ))
        return out

    loc = f"{localized.file}:{localized.lineno} ({localized.function})"

    # ---- 1. Primary: category-specific ----
    if category is FailureCategory.DEPENDENCY:
        missing = _extract_missing_module(sig.exception_message)
        desc = (
            f"Missing dependency '{missing}' imported at {loc}"
            if missing else f"Import failure at {loc}"
        )
        out.append(RootCauseCandidate(
            category=category, description=desc,
            confidence=_score_to_confidence(base),
            score=base + 0.05,
            rationale="ModuleNotFoundError → missing runtime dependency",
            suggested_probe=(
                f"Install '{missing}' (pip install {missing}) or vendor it"
                if missing else
                "Verify the module is installed and on PYTHONPATH"
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.SYNTAX:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Syntax error near {loc}",
            confidence=_score_to_confidence(base),
            score=base + 0.05,
            rationale="Python failed to compile the module",
            suggested_probe=(
                "Inspect the marked line for indentation, quotes, or "
                "unmatched brackets."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.TYPE:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Type mismatch at {loc}: {sig.exception_message}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Operation applied to incompatible types at this line",
            suggested_probe=(
                "Print type(x) for the operands at this line, or add an "
                "explicit isinstance / type assertion."
            ),
            evidence_ids=["signature", "localized"],
        ))
        # Secondary: caller mismatch
        if len(user_frames) >= 2:
            caller = user_frames[-2]
            out.append(RootCauseCandidate(
                category=category,
                description=(
                    f"Caller {caller.file}:{caller.lineno} "
                    f"({caller.function}) may have passed the wrong type"
                ),
                confidence=Confidence.LOW,
                score=base - 0.15,
                rationale=(
                    "The failing operation is inside a callee; the wrong "
                    "type often originates at the call site."
                ),
                suggested_probe=(
                    "Inspect the arguments at the caller's line before the "
                    "call."
                ),
                evidence_ids=["localized", "caller_frame"],
            ))
    elif category is FailureCategory.CONFIG:
        key = _extract_key(sig.exception_message) if sig.exception_type == "KeyError" else ""
        path = _extract_path(sig.exception_message) if sig.exception_type in (
            "FileNotFoundError", "IsADirectoryError", "NotADirectoryError"
        ) else ""
        desc_bits: list[str] = []
        if key:
            desc_bits.append(f"missing key '{key}'")
        if path:
            desc_bits.append(f"missing path '{path}'")
        desc = (
            f"Configuration issue at {loc}"
            + (f" ({', '.join(desc_bits)})" if desc_bits else "")
        )
        out.append(RootCauseCandidate(
            category=category, description=desc,
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Missing key/path most likely expected from config",
            suggested_probe=(
                f"Add '{key or path}' to the relevant configuration."
                if (key or path) else
                "Verify the config file / environment variable is set."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.ENVIRONMENT:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Environment issue at {loc}: {sig.exception_message}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Permission/OS-level problem at this call site",
            suggested_probe=(
                "Verify filesystem permissions / user identity / env vars "
                "available to the process."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.INTEGRATION:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Integration failure at {loc}: {sig.exception_message}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Network / peer / timeout problem at this call site",
            suggested_probe=(
                "Check reachability, credentials, timeouts, and retry "
                "policy for the external dependency."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.CONCURRENCY:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Concurrency issue at {loc}: {sig.exception_message}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Async/event-loop/threading misuse detected",
            suggested_probe=(
                "Check await points, loop ownership, and shared mutable "
                "state around this call."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.RESOURCE:
        out.append(RootCauseCandidate(
            category=category,
            description=(
                f"Resource limit hit near {loc} "
                f"({sig.exception_type or 'signal'})"
            ),
            confidence=_score_to_confidence(base),
            score=base,
            rationale="CPU/memory/file-descriptor limit reached",
            suggested_probe=(
                "Profile the hot path; reduce allocations; or raise the "
                "sandbox limit if the workload is legitimate."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.LOGIC:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Logic defect at {loc}: {sig.exception_message}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Value/assertion failed at this line",
            suggested_probe=(
                "Inspect the guard / assertion / branch around this line; "
                "check boundary conditions."
            ),
            evidence_ids=["signature", "localized"],
        ))
    elif category is FailureCategory.SECURITY:
        out.append(RootCauseCandidate(
            category=category,
            description=f"Security-relevant failure at {loc}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Failure class overlaps with security-sensitive ops",
            suggested_probe=(
                "Review input validation and trust boundaries around this "
                "call."
            ),
            evidence_ids=["signature", "localized"],
        ))
    else:  # RUNTIME / UNKNOWN
        out.append(RootCauseCandidate(
            category=category,
            description=f"Runtime failure at {loc}: {sig.exception_message}",
            confidence=_score_to_confidence(base),
            score=base,
            rationale="Generic runtime error at the localized frame",
            suggested_probe=(
                "Add logging immediately above this line to inspect state."
            ),
            evidence_ids=["signature", "localized"],
        ))

    # ---- 2. Boost if the localized frame has a symbol id (index hit) ----
    if localized.symbol_id:
        for c in out:
            c.score = min(1.0, c.score + 0.05)
            c.evidence_ids.append("symbol_index_hit")
            c.confidence = _score_to_confidence(c.score)

    # ---- 3. Sort by score desc, then description asc (deterministic) ----
    out.sort(key=lambda c: (-c.score, c.description))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 7. DEBUGGER (facade)
# ════════════════════════════════════════════════════════════════════════════
class Debugger:
    """Deterministic failure diagnosis.

    Usage:
        dbg = Debugger(root=Path("/path/to/repo"), repo_index=idx)
        report = dbg.analyze_test_result(test_result)
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        repo_index: Any = None,
    ) -> None:
        self.root = Path(root).resolve() if root else None
        self.repo_index = repo_index

    # ---- inputs ----
    def analyze_text(
        self, text: str, *,
        exit_code: int = 0, signal_number: int = -1,
        project_id: str = "",
    ) -> DebugReport:
        sig, frames = parse_traceback(text or "")
        sig.exit_code = exit_code
        sig.signal_number = signal_number
        if not sig.raw_excerpt:
            sig.raw_excerpt = (text or "")[-500:]
        return self._assemble(
            sig, frames, project_id=project_id,
            input_kind=InputKind.RAW, input_ref="",
        )

    def analyze_test_result(
        self, test_result: Any, *, project_id: str = "",
    ) -> DebugReport:
        """Accept a C18 TestResult (duck-typed)."""
        message = str(getattr(test_result, "failure_message", "") or "")
        tb = str(getattr(test_result, "failure_traceback", "") or "")
        text = (tb + ("\n" + message if message and message not in tb else "")).strip()
        sig, frames = parse_traceback(text)
        # C18 TestResult has no exit code; use 1 if failed/error
        outcome = getattr(getattr(test_result, "outcome", None), "value", "")
        sig.exit_code = 1 if outcome in ("failed", "error") else 0
        ref = str(getattr(test_result, "nodeid", "") or "")
        return self._assemble(
            sig, frames, project_id=project_id,
            input_kind=InputKind.TEST_RESULT, input_ref=ref,
        )

    def analyze_sandbox_result(
        self, sandbox_result: Any, *, project_id: str = "",
    ) -> DebugReport:
        """Accept a C16 SandboxResult (duck-typed)."""
        stdout = str(getattr(sandbox_result, "stdout", "") or "")
        stderr = str(getattr(sandbox_result, "stderr", "") or "")
        # Prefer stderr for traceback, fall back to stdout
        text = stderr if "Traceback" in stderr else (stderr + "\n" + stdout)
        sig, frames = parse_traceback(text)
        sig.exit_code = int(getattr(sandbox_result, "exit_code", 0) or 0)
        sig.signal_number = int(getattr(sandbox_result, "signal_number", -1) or -1)
        if not sig.raw_excerpt:
            sig.raw_excerpt = (stdout + stderr)[-500:]
        ref = str(getattr(sandbox_result, "id", "") or "")
        return self._assemble(
            sig, frames, project_id=project_id,
            input_kind=InputKind.SANDBOX_RESULT, input_ref=ref,
        )

    # ---- core ----
    def _assemble(
        self, sig: FailureSignature, raw_frames: Sequence[LocalizedFrame],
        *, project_id: str, input_kind: InputKind, input_ref: str,
    ) -> DebugReport:
        # 1. Localize frames
        frames = localize(raw_frames, root=self.root, repo_index=self.repo_index)
        localized = pick_primary(frames)

        # 2. Classify
        cat, cat_reason, severity = classify(
            sig, frames,
            signal_number=sig.signal_number, exit_code=sig.exit_code,
        )

        # 3. Build evidence
        evidence: list[Evidence] = [
            Evidence("signature", f"{sig.exception_type}: "
                                  f"{_short(sig.exception_message, 200)}",
                     weight=1.0, metadata=sig.to_dict()),
        ]
        if localized is not None:
            evidence.append(Evidence(
                "localized_frame",
                f"{localized.file}:{localized.lineno} "
                f"in {localized.function}",
                weight=1.0,
                metadata=localized.to_dict(),
            ))
            if localized.snippet:
                evidence.append(Evidence(
                    "source_snippet", localized.snippet, weight=0.5,
                ))
        if sig.signal_number > 0:
            evidence.append(Evidence(
                "signal", f"signal={sig.signal_number}", weight=1.0,
            ))
        if sig.exit_code:
            evidence.append(Evidence(
                "exit_code", f"exit_code={sig.exit_code}", weight=0.5,
            ))

        # 4. Candidates
        candidates = generate_candidates(sig, frames, localized, cat)

        # 5. Status
        if (not sig.exception_type and not sig.exception_message
                and not frames):
            status = DebugStatus.UNDIAGNOSED
        elif localized is None or cat is FailureCategory.UNKNOWN:
            status = DebugStatus.PARTIAL
        else:
            status = DebugStatus.DIAGNOSED

        # 6. Rationale
        rationale_bits = [cat_reason]
        if localized is not None:
            rationale_bits.append(
                f"localized to {localized.file}:{localized.lineno}"
            )
        if candidates:
            rationale_bits.append(
                f"top candidate score={candidates[0].score:.2f}"
            )

        return DebugReport(
            project_id=project_id,
            input_kind=input_kind, input_ref=input_ref,
            category=cat, severity=severity, status=status,
            category_rationale=cat_reason,
            signature=sig, frames=frames, localized=localized,
            evidence=evidence, candidates=candidates,
            rationale="; ".join(rationale_bits),
            provenance=Provenance(
                source="debugging_engine",
                source_type=ProvenanceType.INFERENCE,
                confidence=(
                    Confidence.HIGH if status is DebugStatus.DIAGNOSED
                    else Confidence.LOW
                ),
            ),
        )


# ════════════════════════════════════════════════════════════════════════════
# 8. DEBUG REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class DebugRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: DebugReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"debug_report:{report.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, report.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["debug_report", "c19", report.category.value],
            provenance=report.provenance,
        )
        # Record failure memory for C33
        self.memory.record_failure(
            f"debug_failure:{report.id}",
            what=f"{report.signature.exception_type}: "
                 f"{_short(report.signature.exception_message, 120)}",
            root_cause=(
                report.candidates[0].description
                if report.candidates else report.category_rationale
            ),
            fix=None,
            scope_id=project_id,
            provenance=report.provenance,
            confidence=Confidence.HIGH,
        )
        if self.ontology is None:
            return key

        # BUG entity
        bug = self.ontology.add(
            EntityKind.BUG,
            _short(
                f"{report.signature.exception_type or 'failure'} at "
                f"{report.localized.file if report.localized else '?'}", 120,
            ),
            attributes={
                "debug_report_id": report.id,
                "project_id": project_id,
                "category": report.category.value,
                "severity": report.severity.value,
                "status": report.status.value,
                "exception_type": report.signature.exception_type,
                "exception_message": report.signature.exception_message,
            },
            tags=["bug", report.category.value, report.severity.value],
            provenance=report.provenance,
        )
        # FAILURE entity
        failure = self.ontology.add(
            EntityKind.FAILURE,
            _short(report.signature.exception_type or "failure", 120),
            attributes={
                "input_kind": report.input_kind.value,
                "input_ref": report.input_ref,
                "exit_code": report.signature.exit_code,
                "signal_number": report.signature.signal_number,
            },
            tags=["failure", report.input_kind.value],
            provenance=report.provenance,
        )
        try:
            self.ontology.link(RelationKind.PRODUCES, failure.id, bug.id)
        except ValidationError:
            pass
        # ROOT_CAUSE entities for each candidate
        for c in report.candidates:
            rc = self.ontology.add(
                EntityKind.ROOT_CAUSE,
                _short(c.description, 200),
                attributes={
                    "category": c.category.value,
                    "confidence": c.confidence.value,
                    "score": c.score,
                    "suggested_probe": c.suggested_probe,
                    "rationale": c.rationale,
                },
                tags=["root_cause", c.category.value],
                provenance=report.provenance,
            )
            try:
                self.ontology.link(RelationKind.FIXES, rc.id, bug.id)
            except ValidationError:
                pass
        return bug.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"debug_report:{report_id}",
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

    print("Running C19 self-tests…")
    dbg = Debugger()

    # ---- traceback parser ----
    def t_parse_simple() -> None:
        tb = (
            'Traceback (most recent call last):\n'
            '  File "/app/main.py", line 10, in main\n'
            '    x = add(1, "a")\n'
            '  File "/app/math.py", line 3, in add\n'
            '    return a + b\n'
            'TypeError: unsupported operand type(s) for +: \'int\' and \'str\'\n'
        )
        sig, frames = parse_traceback(tb)
        assert sig.exception_type == "TypeError"
        assert "unsupported operand" in sig.exception_message
        assert len(frames) == 2
        assert frames[0].file == "/app/main.py"
        assert frames[0].lineno == 10
        assert frames[0].function == "main"
        assert frames[1].function == "add"

    def t_parse_pytest_prefix() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/x/test.py", line 5, in test_x\n'
            "    assert 1 == 2\n"
            "E   AssertionError: assert 1 == 2\n"
        )
        sig, frames = parse_traceback(tb)
        assert sig.exception_type == "AssertionError"
        assert "assert 1 == 2" in sig.exception_message

    def t_parse_no_traceback() -> None:
        sig, frames = parse_traceback("Something exploded")
        # No header, no exception-looking line with ':'
        assert frames == []
        # signature empty
        assert sig.exception_type == "" or sig.exception_type == "Something exploded"

    def t_parse_empty() -> None:
        sig, frames = parse_traceback("")
        assert frames == []
        assert sig.exception_type == ""

    def t_parse_last_block_wins() -> None:
        # pytest may output several tracebacks; we want the last one
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/a.py", line 1, in old\n'
            "ValueError: first\n"
            "some noise\n"
            "Traceback (most recent call last):\n"
            '  File "/b.py", line 2, in new\n'
            "RuntimeError: second\n"
        )
        sig, frames = parse_traceback(tb)
        assert sig.exception_type == "RuntimeError"
        assert "second" in sig.exception_message
        assert frames[0].file == "/b.py"

    def t_parse_frame_without_function() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/x.py", line 3\n'
            '    bad(\n'
            "SyntaxError: invalid syntax\n"
        )
        sig, frames = parse_traceback(tb)
        assert len(frames) == 1
        assert frames[0].function == "<module>"

    check("parse: simple traceback", t_parse_simple)
    check("parse: pytest 'E' prefix stripped", t_parse_pytest_prefix)
    check("parse: no traceback → empty", t_parse_no_traceback)
    check("parse: empty input", t_parse_empty)
    check("parse: last block wins (pytest multi-block)", t_parse_last_block_wins)
    check("parse: frame without 'in <func>'", t_parse_frame_without_function)

    # ---- classification ----
    def _sig(t: str, m: str = "") -> FailureSignature:
        return FailureSignature(exception_type=t, exception_message=m)

    def t_classify_syntax() -> None:
        cat, _, sev = classify(_sig("SyntaxError", "invalid syntax"), [])
        assert cat is FailureCategory.SYNTAX
        assert sev is Severity.HIGH

    def t_classify_type() -> None:
        cat, _, _ = classify(_sig("TypeError", "unsupported operand"), [])
        assert cat is FailureCategory.TYPE

    def t_classify_logic() -> None:
        cat, _, _ = classify(_sig("AssertionError", "assert 1 == 2"), [])
        assert cat is FailureCategory.LOGIC

    def t_classify_dependency() -> None:
        cat, _, _ = classify(
            _sig("ModuleNotFoundError", "No module named 'fastapi'"), [],
        )
        assert cat is FailureCategory.DEPENDENCY

    def t_classify_keyerror_default() -> None:
        # No config-ish frame → LOGIC (via refinement)
        cat, _, _ = classify(_sig("KeyError", "'foo'"), [])
        assert cat is FailureCategory.LOGIC

    def t_classify_keyerror_in_config() -> None:
        frames = [LocalizedFrame(file="/app/settings.py", lineno=5,
                                  function="load", is_user_code=True)]
        cat, _, _ = classify(_sig("KeyError", "'DATABASE_URL'"), frames)
        assert cat is FailureCategory.CONFIG

    def t_classify_fnf_config() -> None:
        cat, _, _ = classify(
            _sig("FileNotFoundError", "[Errno 2] No such file or directory: "
                                       "'/app/pyproject.toml'"), [],
        )
        assert cat is FailureCategory.CONFIG

    def t_classify_fnf_env() -> None:
        cat, _, _ = classify(
            _sig("FileNotFoundError", "[Errno 2] No such file or directory: "
                                       "'/etc/secret.key'"), [],
        )
        assert cat is FailureCategory.ENVIRONMENT

    def t_classify_permission() -> None:
        cat, _, _ = classify(_sig("PermissionError", "denied"), [])
        assert cat is FailureCategory.ENVIRONMENT

    def t_classify_connection() -> None:
        cat, _, _ = classify(_sig("ConnectionRefusedError", "refused"), [])
        assert cat is FailureCategory.INTEGRATION

    def t_classify_memory() -> None:
        cat, _, sev = classify(_sig("MemoryError", ""), [])
        assert cat is FailureCategory.RESOURCE
        assert sev is Severity.CRITICAL

    def t_classify_recursion() -> None:
        cat, _, _ = classify(_sig("RecursionError", "maximum recursion"), [])
        assert cat is FailureCategory.RESOURCE

    def t_classify_runtime_event_loop() -> None:
        cat, _, _ = classify(
            _sig("RuntimeError", "no running event loop"), [],
        )
        assert cat is FailureCategory.CONCURRENCY

    def t_classify_runtime_generic() -> None:
        cat, _, _ = classify(_sig("RuntimeError", "unexpected state"), [])
        assert cat is FailureCategory.RUNTIME

    def t_classify_unknown() -> None:
        cat, _, sev = classify(_sig("", ""), [])
        assert cat is FailureCategory.UNKNOWN
        assert sev is Severity.LOW

    def t_classify_signal_sigxcpu() -> None:
        sn = getattr(_signal, "SIGXCPU", -1)
        if sn < 0:
            return
        cat, reason, sev = classify(_sig("", ""), [], signal_number=sn)
        assert cat is FailureCategory.RESOURCE
        assert sev is Severity.CRITICAL
        assert "SIGXCPU" in reason

    def t_classify_signal_sigkill() -> None:
        sn = getattr(_signal, "SIGKILL", -1)
        if sn < 0:
            return
        cat, _, _ = classify(_sig("", ""), [], signal_number=sn)
        assert cat is FailureCategory.RESOURCE

    def t_classify_signal_sigsegv() -> None:
        sn = getattr(_signal, "SIGSEGV", -1)
        if sn < 0:
            return
        cat, _, _ = classify(_sig("", ""), [], signal_number=sn)
        assert cat is FailureCategory.RUNTIME

    check("classify: SyntaxError → SYNTAX/HIGH", t_classify_syntax)
    check("classify: TypeError → TYPE", t_classify_type)
    check("classify: AssertionError → LOGIC", t_classify_logic)
    check("classify: ModuleNotFoundError → DEPENDENCY", t_classify_dependency)
    check("classify: KeyError default → LOGIC", t_classify_keyerror_default)
    check("classify: KeyError in config frame → CONFIG",
          t_classify_keyerror_in_config)
    check("classify: FileNotFoundError on config → CONFIG",
          t_classify_fnf_config)
    check("classify: FileNotFoundError under /etc → ENVIRONMENT",
          t_classify_fnf_env)
    check("classify: PermissionError → ENVIRONMENT", t_classify_permission)
    check("classify: ConnectionRefusedError → INTEGRATION",
          t_classify_connection)
    check("classify: MemoryError → RESOURCE/CRITICAL", t_classify_memory)
    check("classify: RecursionError → RESOURCE", t_classify_recursion)
    check("classify: RuntimeError 'event loop' → CONCURRENCY",
          t_classify_runtime_event_loop)
    check("classify: RuntimeError generic → RUNTIME",
          t_classify_runtime_generic)
    check("classify: empty → UNKNOWN", t_classify_unknown)
    check("classify: SIGXCPU → RESOURCE", t_classify_signal_sigxcpu)
    check("classify: SIGKILL → RESOURCE", t_classify_signal_sigkill)
    check("classify: SIGSEGV → RUNTIME", t_classify_signal_sigsegv)

    # ---- localizer ----
    def t_localize_snippet_read() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "app/math.py", (
                "def add(a, b):\n"
                "    return a + b\n"           # line 2
                "\n"
                "def sub(a, b):\n"
                "    return a - b\n"
            ))
            frames = [LocalizedFrame(file=str(root / "app" / "math.py"),
                                      lineno=2, function="add")]
            out = localize(frames, root=root)
            assert out[0].is_user_code is True
            assert "return a + b" in out[0].snippet
            assert ">>" in out[0].snippet

    def t_localize_skips_stdlib() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "app/x.py", "def f(): pass\n")
            frames = [
                LocalizedFrame(
                    file="/usr/lib/python3.11/json/__init__.py",
                    lineno=10, function="loads",
                ),
                LocalizedFrame(
                    file=str(root / "app" / "x.py"),
                    lineno=1, function="f",
                ),
            ]
            out = localize(frames, root=root)
            assert out[0].is_user_code is False
            assert out[1].is_user_code is True

    def t_pick_primary_prefers_user() -> None:
        frames = [
            LocalizedFrame(file="/usr/lib/python/x.py", lineno=1,
                            function="f", is_user_code=False),
            LocalizedFrame(file="/app/y.py", lineno=2,
                            function="g", is_user_code=True),
            LocalizedFrame(file="/app/z.py", lineno=3,
                            function="h", is_user_code=True),
        ]
        prim = pick_primary(frames)
        assert prim is not None and prim.file == "/app/z.py"

    def t_localize_with_index_symbol_lookup() -> None:
        # Build a tiny index using C13
        from sebrain.c13 import CodeRepresentationEngine, RepoIndex
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "app/math.py", (
                "def add(a, b):\n"
                "    return a + b\n"
            ))
            eng = CodeRepresentationEngine()
            idx = eng.analyze_repo(root)
            frames = [LocalizedFrame(
                file=str(root / "app" / "math.py"),
                lineno=2, function="add",
            )]
            out = localize(frames, root=root, repo_index=idx)
            assert out[0].symbol_id  # non-empty
            assert "add" in out[0].symbol_qualname

    check("localize: snippet read + marker", t_localize_snippet_read)
    check("localize: marks stdlib frames", t_localize_skips_stdlib)
    check("localize: pick_primary prefers deepest user frame",
          t_pick_primary_prefers_user)
    check("localize: uses C13 index for symbol lookup",
          t_localize_with_index_symbol_lookup)

    # ---- candidates ----
    def t_candidate_type_primary_and_caller() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/app/a.py", line 5, in caller\n'
            "    r = add(1, 'x')\n"
            '  File "/app/b.py", line 2, in add\n'
            "    return a + b\n"
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n"
        )
        report = dbg.analyze_text(tb)
        assert report.category is FailureCategory.TYPE
        # Primary candidate should mention the callee frame
        assert report.candidates
        prim = report.candidates[0]
        assert "/app/b.py" in prim.description
        # Caller candidate present too
        assert any("/app/a.py" in c.description for c in report.candidates[1:])

    def t_candidate_dependency_missing_module() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/app/x.py", line 1, in <module>\n'
            "    import fastapi\n"
            "ModuleNotFoundError: No module named 'fastapi'\n"
        )
        report = dbg.analyze_text(tb)
        assert report.category is FailureCategory.DEPENDENCY
        prim = report.candidates[0]
        assert "fastapi" in prim.description
        assert "fastapi" in prim.suggested_probe

    def t_candidate_keyerror_names_key() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/app/config.py", line 3, in load\n'
            "    return cfg['DATABASE_URL']\n"
            "KeyError: 'DATABASE_URL'\n"
        )
        report = dbg.analyze_text(tb)
        assert report.category is FailureCategory.CONFIG
        prim = report.candidates[0]
        assert "DATABASE_URL" in prim.description

    def t_candidate_no_localized() -> None:
        report = dbg.analyze_text("just some message no traceback")
        assert report.localized is None
        assert report.candidates
        assert "no localized frame" in report.candidates[0].description

    def t_candidate_sorted_by_score_desc() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/a.py", line 1, in f\n'
            "    x = undefined_name\n"
            "NameError: name 'undefined_name' is not defined\n"
        )
        report = dbg.analyze_text(tb)
        scores = [c.score for c in report.candidates]
        assert scores == sorted(scores, reverse=True)

    check("candidate: TYPE gives callee + caller", t_candidate_type_primary_and_caller)
    check("candidate: DEPENDENCY names missing module",
          t_candidate_dependency_missing_module)
    check("candidate: CONFIG names the missing key",
          t_candidate_keyerror_names_key)
    check("candidate: no localized frame → fallback candidate",
          t_candidate_no_localized)
    check("candidate: sorted by score desc", t_candidate_sorted_by_score_desc)

    # ---- status / severity ----
    def t_status_diagnosed() -> None:
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/a.py", line 1, in f\n'
            "    1/0\n"
            "ZeroDivisionError: division by zero\n"
        )
        report = dbg.analyze_text(tb)
        assert report.status is DebugStatus.DIAGNOSED
        assert report.category is FailureCategory.LOGIC

    def t_status_undiagnosed_on_empty() -> None:
        report = dbg.analyze_text("")
        assert report.status is DebugStatus.UNDIAGNOSED
        assert report.category is FailureCategory.UNKNOWN

    def t_severity_mapping() -> None:
        tb_res = dbg.analyze_text(
            "Traceback (most recent call last):\n"
            '  File "/a.py", line 1, in f\n'
            "    raise MemoryError()\n"
            "MemoryError\n"
        )
        assert tb_res.severity is Severity.CRITICAL

    check("status: DIAGNOSED on full traceback", t_status_diagnosed)
    check("status: UNDIAGNOSED on empty", t_status_undiagnosed_on_empty)
    check("severity: MemoryError → CRITICAL", t_severity_mapping)

    # ---- adapters ----
    def t_adapter_test_result() -> None:
        # Fake C18 TestResult
        class TR:
            nodeid = "tests/test_x.py::test_y"
            outcome = type("O", (), {"value": "failed"})()
            failure_message = "AssertionError: assert 1 == 2"
            failure_traceback = (
                "Traceback (most recent call last):\n"
                '  File "/app/test_x.py", line 5, in test_y\n'
                "    assert 1 == 2\n"
                "AssertionError: assert 1 == 2\n"
            )
        report = dbg.analyze_test_result(TR())
        assert report.input_kind is InputKind.TEST_RESULT
        assert report.input_ref == "tests/test_x.py::test_y"
        assert report.category is FailureCategory.LOGIC

    def t_adapter_sandbox_result() -> None:
        class SR:
            id = "abc"
            exit_code = -9
            signal_number = getattr(_signal, "SIGKILL", 9)
            stdout = ""
            stderr = ""
        report = dbg.analyze_sandbox_result(SR())
        assert report.input_kind is InputKind.SANDBOX_RESULT
        assert report.input_ref == "abc"
        assert report.category is FailureCategory.RESOURCE

    def t_adapter_c18_pytest_short_failure() -> None:
        class TR:
            nodeid = "tests/test_calc.py::test_add"
            outcome = type("O", (), {"value": "failed"})()
            failure_message = "tests/test_calc.py:4: AssertionError"
            failure_traceback = (
                "def test_add() -> None:\n"
                ">       assert add(1, 2) == 3\n"
                "E       assert -1 == 3\n"
            )
        report = dbg.analyze_test_result(TR())
        assert report.signature.exception_type == "AssertionError"
        assert report.category is FailureCategory.LOGIC
        assert report.localized is not None
        assert report.localized.file == "tests/test_calc.py"
        assert report.localized.lineno == 4

    check("adapter: C18 pytest short failure", t_adapter_c18_pytest_short_failure)
    check("adapter: C18 TestResult", t_adapter_test_result)
    check("adapter: C16 SandboxResult (signal)", t_adapter_sandbox_result)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        report = dbg.analyze_text(
            "Traceback (most recent call last):\n"
            '  File "/a.py", line 1, in f\n'
            "    x\n"
            "NameError: name 'x' is not defined\n"
        )
        d = report.to_dict()
        assert d["id"] == report.id
        assert d["category"] == "runtime"  # NameError isn't in table → fallback
        assert isinstance(d["candidates"], list)
        s = report.summary()
        assert "Debug Report" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                report = dbg.analyze_text(
                    "Traceback (most recent call last):\n"
                    '  File "/a.py", line 1, in f\n'
                    "    import fastapi\n"
                    "ModuleNotFoundError: No module named 'fastapi'\n"
                )
                repo = DebugRepository(memory=mem, ontology=ont)
                ent = repo.save(report, project_id="proj-x")
                assert ent
                loaded = repo.load(report.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["category"] == "dependency"
                assert ont.count(kind=EntityKind.BUG) >= 1
                assert ont.count(kind=EntityKind.FAILURE) >= 1
                assert ont.count(kind=EntityKind.ROOT_CAUSE) >= 1
            finally:
                s.shutdown()

    def t_persist_records_failure_memory() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                report = dbg.analyze_text(
                    "Traceback (most recent call last):\n"
                    '  File "/a.py", line 1, in f\n'
                    "    1/0\n"
                    "ZeroDivisionError: division by zero\n"
                )
                repo = DebugRepository(memory=mem, ontology=ont)
                repo.save(report, project_id="proj-y")
                fails = mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-y",
                )
                assert len(fails) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (BUG/FAILURE/ROOT_CAUSE)",
          t_persist)
    check("persist: failure memory recorded for C33",
          t_persist_records_failure_memory)

    # ---- e2e with C18 ----
    def t_e2e_from_c18_failure() -> None:
        from sebrain.c18 import (
            TestExecutionEngine, RunStatus,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "pkg/__init__.py", '"""pkg."""\n')
            _write(root, "pkg/math.py", (
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n"
            ))
            _write(root, "tests/__init__.py", "")
            _write(root, "tests/test_math.py", (
                "from pkg.math import add\n"
                "def test_add_broken() -> None:\n"
                "    assert add(1, 'x') == 3   # will raise TypeError\n"
            ))
            engine = TestExecutionEngine()
            res = engine.run_full(root=root, project_id="demo")
            assert res.status is RunStatus.TESTS_FAILED
            failed = next(r for r in res.results if r.outcome.value == "failed")

            dbg2 = Debugger(root=root)
            report = dbg2.analyze_test_result(failed, project_id="demo")
            assert report.category in (FailureCategory.TYPE,
                                        FailureCategory.RUNTIME), report.category
            assert report.localized is not None
            # Localized frame should be in the user test file or pkg
            assert ("/pkg/math.py" in report.localized.file
                    or "test_math.py" in report.localized.file)

    check("e2e: C18 failure → C19 diagnosis", t_e2e_from_c18_failure, requires_pytest=True)

    print()
    skip_note = f", {skipped} skipped (pytest not installed)" if skipped else ""
    print(f"Self-tests: {passed} passed, {len(failures)} failed{skip_note}")
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
    print("SE Brain C19 — Debugging Engine")
    print("=" * 78)

    samples = [
        (
            "Type mismatch (callee + caller frames)",
            'Traceback (most recent call last):\n'
            '  File "/app/main.py", line 42, in handle_request\n'
            '    total = add(1, "two")\n'
            '  File "/app/math.py", line 3, in add\n'
            '    return a + b\n'
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n",
        ),
        (
            "Missing dependency",
            'Traceback (most recent call last):\n'
            '  File "/app/server.py", line 1, in <module>\n'
            "    import fastapi\n"
            "ModuleNotFoundError: No module named 'fastapi'\n",
        ),
        (
            "Config key miss",
            'Traceback (most recent call last):\n'
            '  File "/app/settings.py", line 12, in load_settings\n'
            "    return cfg['DATABASE_URL']\n"
            "KeyError: 'DATABASE_URL'\n",
        ),
        (
            "Resource exhaustion (signal)",
            "",  # empty text; signal only
        ),
    ]
    signals = [-1, -1, -1, getattr(_signal, "SIGKILL", 9)]

    dbg = Debugger()

    for i, ((label, text), signum) in enumerate(zip(samples, signals), 1):
        print(f"\n[{i}] {label}")
        report = dbg.analyze_text(
            text, signal_number=signum, project_id="demo",
        )
        print(report.summary())
        print("    Evidence:")
        for e in report.evidence[:3]:
            print(f"      · [{e.kind}] {_short(e.text, 80)}")
        print("    Candidates:")
        for c in report.candidates[:3]:
            print(f"      [{c.category.value}] score={c.score:.2f} "
                  f"conf={c.confidence.value}")
            print(f"        {_short(c.description, 90)}")
            if c.suggested_probe:
                print(f"        probe: {_short(c.suggested_probe, 90)}")

    # Persistence demo
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = DebugRepository(memory=mem, ontology=ont)
                r = dbg.analyze_text(
                    'Traceback (most recent call last):\n'
                    '  File "/a.py", line 1, in f\n'
                    "    1/0\n"
                    "ZeroDivisionError: division by zero\n"
                )
                ent = repo.save(r, project_id="demo")
                print(f"\n[P] Persisted → ontology entity: {ent[:12]}…")
                print(f"    BUG: {ont.count(kind=EntityKind.BUG)}  "
                      f"FAILURE: {ont.count(kind=EntityKind.FAILURE)}  "
                      f"ROOT_CAUSE: {ont.count(kind=EntityKind.ROOT_CAUSE)}")
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
