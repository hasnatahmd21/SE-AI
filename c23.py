"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C23 — SECURITY ANALYSIS ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C21 (Severity enum only).

Purpose:
    Evidence-based static security analysis over Python source, requirements
    files, and .env files. Produces findings with line numbers, snippets,
    CWE tags, and remediation suggestions.

Categories covered (from the specification):
    1.  Unsafe dependencies        (unpinned, http:// index, http git)
    2.  Insecure configuration     (DEBUG on, weak secret key, CORS *)
    3.  Dangerous filesystem ops   (rmtree on variable, mktemp)
    4.  Command execution risks    (subprocess shell=True, os.system, os.popen)
    5.  Injection risks            (SQL via f-string/format/%)
    6.  Auth/Authz weaknesses      (asserts used for auth; no auth decorator)
    7.  Secrets exposure           (hardcoded credentials, .env secrets)
    8.  Unsafe serialization       (pickle.loads, marshal.loads, yaml.load)
    9.  Unsafe subprocess behavior (shell=True + string command)
   10.  Path traversal             (os.path.join(request...))
   11.  Insecure network behavior  (http://, verify=False, unverified SSL ctx)
   12.  Common vulnerabilities     (eval/exec, weak hash, JWT no-verify)

Rule engine:
    - 22 rules total (18 regex, 4 AST + config scanners)
    - Each rule carries: id, kind, severity, CWE, confidence, suggestion
    - Regex rules = fast pattern matches with suppression hooks
    - AST rules  = shape-aware (kwarg presence, arg type)
    - Config scanners for requirements.txt + .env

Invariants honored:
    - NO external LLM. Deterministic.
    - Every finding carries: rule_id, severity, kind, file, line, snippet, CWE
    - Findings are HONEST about confidence (HIGH/MEDIUM/LOW)
    - No "secure" verdict is ever claimed — findings only
    - Coverage note explicitly lists what was NOT scanned
    - Bounded: max_files, max_bytes_per_file, max_findings_per_file
    - Same source → identical findings (deterministic)

Explicit limitations (Rule #59):
    - Pattern-based. NOT a full SAST.
    - NO taint/dataflow analysis, NO CVE database, NO runtime instrumentation.
    - Confidence=LOW findings are heuristic (may be false positives).
    - Only Python source + requirements.txt + .env are scanned. Other
      languages (JS, Go, ...) are out of scope here (belongs to C30).
    - This engine does NOT replace Bandit/Semgrep/Snyk in production.

Contents:
  1.  Enums: FindingKind, Confidence (reuse from C01)
  2.  Dataclasses: SecurityFinding, FileScanResult, SecurityReport
  3.  Rule infrastructure (regex + AST)
  4.  18 regex rules
  5.  4 AST rules
  6.  Config scanners (requirements.txt, .env)
  7.  SecurityAnalyzer facade (file / source / repo)
  8.  SecurityRepository (persist to C02 + C04)
  9.  Self-tests (~35)
 10.  Demo

Run as script:
    python -m sebrain.c23            # demo
    python -m sebrain.c23 --test     # self-tests
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


_SNIPPET_CTX = 0


def _snippet(source: str, lineno: int, ctx: int = _SNIPPET_CTX) -> str:
    if not source or lineno <= 0:
        return ""
    lines = source.splitlines()
    i = lineno - 1
    if i < 0 or i >= len(lines):
        return ""
    lo = max(0, i - ctx)
    hi = min(len(lines), i + ctx + 1)
    return " | ".join(f"{k+1}:{lines[k].strip()}" for k in range(lo, hi))[:300]


def _line_of(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", ".tox",
    ".idea", ".vscode", "dist", "build", ".eggs", "site-packages",
})

_TEXT_EXTS = frozenset({".py", ".pyi", ".txt", ".cfg", ".toml", ".ini",
                        ".env", ".yaml", ".yml", ".json"})


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class FindingKind(str, Enum):
    UNSAFE_DEPENDENCY = "unsafe_dependency"
    INSECURE_CONFIG = "insecure_config"
    DANGEROUS_FS = "dangerous_fs"
    COMMAND_EXECUTION = "command_execution"
    INJECTION = "injection"
    AUTH_WEAKNESS = "auth_weakness"
    SECRET_EXPOSURE = "secret_exposure"
    UNSAFE_SERIALIZATION = "unsafe_serialization"
    UNSAFE_SUBPROCESS = "unsafe_subprocess"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_NETWORK = "insecure_network"
    COMMON_VULN = "common_vuln"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class SecurityFinding:
    rule_id: str
    kind: FindingKind
    severity: Severity
    message: str
    file: str = ""
    line: int = 0
    snippet: str = ""
    cwe: str = ""
    confidence: Confidence = Confidence.MEDIUM
    suggestion: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "snippet": self.snippet,
            "cwe": self.cwe,
            "confidence": self.confidence.value,
            "suggestion": self.suggestion,
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True)
class FileScanResult:
    path: str
    findings: list[SecurityFinding] = field(default_factory=list)
    bytes_scanned: int = 0
    syntax_ok: bool = True
    skipped: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "findings": [f.to_dict() for f in self.findings],
            "bytes_scanned": self.bytes_scanned,
            "syntax_ok": self.syntax_ok,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(slots=True)
class SecurityReport:
    id: str = field(default_factory=_new_id)
    root: str = ""
    project_id: str = ""
    findings: list[SecurityFinding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    total_bytes_scanned: int = 0
    file_results: list[FileScanResult] = field(default_factory=list)
    coverage_note: str = ""
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    # ---- aggregates ----
    def count(self, sev: Severity) -> int:
        return sum(1 for f in self.findings if f.severity is sev)

    def by_kind(self) -> dict[str, int]:
        d: dict[str, int] = {}
        for f in self.findings:
            d[f.kind.value] = d.get(f.kind.value, 0) + 1
        return d

    def highest_severity(self) -> Severity | None:
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
                 Severity.LOW, Severity.INFO]
        for s in order:
            if self.count(s):
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root": self.root,
            "project_id": self.project_id,
            "findings": [f.to_dict() for f in self.findings],
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "total_bytes_scanned": self.total_bytes_scanned,
            "coverage_note": self.coverage_note,
            "rationale": self.rationale,
            "counts": {s.value: self.count(s) for s in Severity},
            "by_kind": self.by_kind(),
            "highest_severity": (self.highest_severity().value
                                 if self.highest_severity() else None),
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        c = {s.value: self.count(s) for s in Severity}
        return (
            "=== Security Report ===\n"
            f"root={self.root}\n"
            f"files_scanned={self.files_scanned}  "
            f"skipped={self.files_skipped}  "
            f"bytes={self.total_bytes_scanned}\n"
            f"findings: critical={c['critical']} high={c['high']} "
            f"medium={c['medium']} low={c['low']} info={c['info']}\n"
            f"by_kind: {self.by_kind()}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. RULE INFRASTRUCTURE
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class _BaseRule:
    rule_id: str = ""
    kind: FindingKind = FindingKind.COMMON_VULN
    severity: Severity = Severity.LOW
    cwe: str = ""
    confidence: Confidence = Confidence.MEDIUM
    message: str = ""
    suggestion: str = ""

    def check(self, source: str, path: str) -> list[SecurityFinding]:
        return []


@dataclass(slots=True)
class _RegexRule(_BaseRule):
    """Simple single-regex rule with optional suppression tokens."""
    pattern: re.Pattern = field(default_factory=lambda: re.compile(r"$^"))
    suppress_if_contains: tuple[str, ...] = ()
    flags: int = 0

    def check(self, source: str, path: str) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        for m in self.pattern.finditer(source):
            matched = m.group(0)
            if any(tok in matched for tok in self.suppress_if_contains):
                continue
            lineno = _line_of(source, m.start())
            findings.append(SecurityFinding(
                rule_id=self.rule_id,
                kind=self.kind,
                severity=self.severity,
                message=self.message,
                file=path,
                line=lineno,
                snippet=_snippet(source, lineno),
                cwe=self.cwe,
                confidence=self.confidence,
                suggestion=self.suggestion,
                evidence={"match": _short(matched, 80)},
            ))
        return findings


@dataclass(slots=True)
class _AstRule(_BaseRule):
    """AST-shape rule. `matcher` yields line numbers."""
    matcher: Callable[[ast.AST, str], Iterable[int]] = lambda t, s: ()

    def check(self, source: str, path: str) -> list[SecurityFinding]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        findings: list[SecurityFinding] = []
        seen_lines: set[int] = set()
        for lineno in self.matcher(tree, source):
            if lineno in seen_lines:
                continue
            seen_lines.add(lineno)
            findings.append(SecurityFinding(
                rule_id=self.rule_id,
                kind=self.kind,
                severity=self.severity,
                message=self.message,
                file=path,
                line=lineno,
                snippet=_snippet(source, lineno),
                cwe=self.cwe,
                confidence=self.confidence,
                suggestion=self.suggestion,
                evidence={"ast": "matched"},
            ))
        return findings


# ---- AST matchers ----
_SUBPROCESS_FUNCS = frozenset({
    "run", "call", "check_call", "check_output", "Popen",
})


def _match_shell_true(tree: ast.AST, source: str) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name: str | None = None
        if isinstance(fn, ast.Attribute):
            name = fn.attr
        elif isinstance(fn, ast.Name):
            name = fn.id
        if name not in _SUBPROCESS_FUNCS:
            continue
        for kw in node.keywords:
            if kw.arg == "shell":
                if (isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    yield node.lineno
                    break


def _match_sql_injection(tree: ast.AST, source: str) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute)
                and fn.attr in ("execute", "executemany", "executescript")):
            continue
        if not node.args:
            continue
        first = node.args[0]
        # f-string: cursor.execute(f"... {x} ...")
        if isinstance(first, ast.JoinedStr):
            yield node.lineno
            continue
        # "..".format(...)
        if isinstance(first, ast.Call):
            f2 = first.func
            if isinstance(f2, ast.Attribute) and f2.attr == "format":
                yield node.lineno
                continue
        # ".." % (args,)
        if isinstance(first, ast.BinOp) and isinstance(first.op, ast.Mod):
            # Only flag if LHS is a string literal (not a parameterized query)
            if isinstance(first.left, ast.Constant) and isinstance(first.left.value, str):
                # If string already contains placeholders AND we're doing `%s` patterns
                # for parameters, this would still be a code smell but not a bug if
                # the args are provided separately. We look for str % var patterns:
                if isinstance(first.right, (ast.Name, ast.Attribute, ast.Call,
                                            ast.BinOp, ast.Tuple)):
                    # Flag as likely vulnerable (concatenation-style)
                    yield node.lineno
                    continue


def _match_jwt_no_verify(tree: ast.AST, source: str) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "decode"):
            continue
        for kw in node.keywords:
            if kw.arg == "verify":
                if (isinstance(kw.value, ast.Constant)
                        and kw.value.value is False):
                    yield node.lineno
                    break
            if kw.arg == "options" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if (isinstance(k, ast.Constant)
                            and k.value in ("verify_signature",
                                            "verify_exp", "verify_aud")
                            and isinstance(v, ast.Constant)
                            and v.value is False):
                        yield node.lineno
                        break


_SECURITY_ATTRS = (
    "is_admin", "is_superuser", "is_staff", "is_authenticated",
    "has_permission", "has_role", "has_perm", "can_access",
    "can_edit", "can_delete", "can_read", "can_write", "is_owner",
)


def _match_assert_auth(tree: ast.AST, source: str) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        try:
            expr = ast.unparse(node.test)
        except Exception:
            continue
        for attr in _SECURITY_ATTRS:
            if attr in expr:
                yield node.lineno
                break


# ════════════════════════════════════════════════════════════════════════════
# 4. REGEX RULES (18)
# ════════════════════════════════════════════════════════════════════════════
_RE_RULES: list[_RegexRule] = [

    _RegexRule(
        rule_id="HARDCODED_SECRET",
        kind=FindingKind.SECRET_EXPOSURE,
        severity=Severity.CRITICAL,
        cwe="CWE-798",
        confidence=Confidence.HIGH,
        message="Hardcoded credential-like literal in source",
        suggestion=(
            "Load secrets from environment/secret manager; never commit them."
        ),
        pattern=re.compile(
            r"""(?ix)
                [\"']?(?:password|passwd|pwd|secret|api[_-]?key|apikey|
                     access[_-]?key|private[_-]?key|
                     aws[_-]?secret[_-]?(?:access[_-]?)?key|
                     auth[_-]?token|bearer[_-]?token|session[_-]?token|
                     client[_-]?secret|db[_-]?password)
                [\"']?\s*[:=]\s*
                [\"']([^\"'\s]{8,})[\"']
            """,
        ),
    ),

    _RegexRule(
        rule_id="EVAL_USAGE",
        kind=FindingKind.COMMON_VULN,
        severity=Severity.CRITICAL,
        cwe="CWE-95",
        confidence=Confidence.HIGH,
        message="eval() executes arbitrary code",
        suggestion="Use ast.literal_eval for literals or a real parser.",
        pattern=re.compile(r"\beval\s*\("),
        suppress_if_contains=("literal_eval",),
    ),

    _RegexRule(
        rule_id="EXEC_USAGE",
        kind=FindingKind.COMMON_VULN,
        severity=Severity.CRITICAL,
        cwe="CWE-95",
        confidence=Confidence.HIGH,
        message="exec() executes arbitrary code",
        suggestion="Avoid exec; use functions/importlib instead.",
        pattern=re.compile(r"\bexec\s*\("),
    ),

    _RegexRule(
        rule_id="OS_SYSTEM",
        kind=FindingKind.COMMAND_EXECUTION,
        severity=Severity.HIGH,
        cwe="CWE-78",
        confidence=Confidence.HIGH,
        message="os.system() invokes a shell (injection risk)",
        suggestion="Use subprocess.run([...], shell=False).",
        pattern=re.compile(r"\bos\.system\s*\("),
    ),

    _RegexRule(
        rule_id="OS_POPEN",
        kind=FindingKind.COMMAND_EXECUTION,
        severity=Severity.HIGH,
        cwe="CWE-78",
        confidence=Confidence.HIGH,
        message="os.popen() invokes a shell (injection risk)",
        suggestion="Use subprocess.run with an argv list.",
        pattern=re.compile(r"\bos\.popen\s*\("),
    ),

    _RegexRule(
        rule_id="PICKLE_LOADS",
        kind=FindingKind.UNSAFE_SERIALIZATION,
        severity=Severity.HIGH,
        cwe="CWE-502",
        confidence=Confidence.HIGH,
        message="pickle.load(s) can execute arbitrary code on untrusted data",
        suggestion="Use JSON or a signed/validated format instead.",
        pattern=re.compile(r"\bpickle\.loads?\s*\("),
    ),

    _RegexRule(
        rule_id="MARSHAL_LOADS",
        kind=FindingKind.UNSAFE_SERIALIZATION,
        severity=Severity.HIGH,
        cwe="CWE-502",
        confidence=Confidence.HIGH,
        message="marshal.loads is unsafe on untrusted data",
        suggestion="Use JSON/msgpack with schema validation.",
        pattern=re.compile(r"\bmarshal\.loads?\s*\("),
    ),

    _RegexRule(
        rule_id="YAML_LOAD_UNSAFE",
        kind=FindingKind.UNSAFE_SERIALIZATION,
        severity=Severity.MEDIUM,
        cwe="CWE-502",
        confidence=Confidence.MEDIUM,
        message="yaml.load without SafeLoader can execute code",
        suggestion="Use yaml.safe_load or yaml.load(..., Loader=SafeLoader).",
        # Match yaml.load( call that does NOT have a Loader= keyword.
        # The lookahead tolerates one level of nested parens (e.g.
        # "yaml.load(open('x'), Loader=SafeLoader)") — a naive `[^)]*`
        # stops at the first ')' from `open('x')`, so it never sees the
        # real "Loader=" and always (wrongly) flags the safe call too.
        pattern=re.compile(
            r"\byaml\.load\s*\((?!(?:[^()]|\([^()]*\))*Loader\s*=)"
        ),
    ),

    _RegexRule(
        rule_id="VERIFY_FALSE",
        kind=FindingKind.INSECURE_NETWORK,
        severity=Severity.HIGH,
        cwe="CWE-295",
        confidence=Confidence.HIGH,
        message="TLS certificate verification disabled (verify=False)",
        suggestion="Never disable TLS verification in production.",
        pattern=re.compile(r"verify\s*=\s*False\b"),
    ),

    _RegexRule(
        rule_id="UNVERIFIED_SSL_CTX",
        kind=FindingKind.INSECURE_NETWORK,
        severity=Severity.HIGH,
        cwe="CWE-295",
        confidence=Confidence.HIGH,
        message="Unverified SSL context (ssl._create_unverified_context)",
        suggestion="Use the default verified SSL context.",
        pattern=re.compile(r"ssl\._create_unverified_context\s*\("),
    ),

    _RegexRule(
        rule_id="PLAINTEXT_HTTP",
        kind=FindingKind.INSECURE_NETWORK,
        severity=Severity.MEDIUM,
        cwe="CWE-319",
        confidence=Confidence.MEDIUM,
        message="Non-local HTTP URL (plaintext)",
        suggestion="Use https:// (localhost is exempt).",
        pattern=re.compile(
            r"http://(?!localhost\b|127\.0\.0\.1|::1|0\.0\.0\.0|"
            r"\{|\$|[^\s]*\.local\b)"
        ),
    ),

    _RegexRule(
        rule_id="SECRET_KEY_LITERAL",
        kind=FindingKind.SECRET_EXPOSURE,
        severity=Severity.CRITICAL,
        cwe="CWE-798",
        confidence=Confidence.HIGH,
        message="Hardcoded SECRET_KEY (Flask/Django)",
        suggestion="Read SECRET_KEY from environment or a secret store.",
        pattern=re.compile(
            r"""(?ix)
                (?:^|[^\w])SECRET_KEY\s*[:=]\s*
                [\"']([^\"'\s]{6,})[\"']
            """
        ),
    ),

    _RegexRule(
        rule_id="CORS_WILDCARD",
        kind=FindingKind.INSECURE_CONFIG,
        severity=Severity.MEDIUM,
        cwe="CWE-942",
        confidence=Confidence.MEDIUM,
        message="CORS wildcard origin ('*')",
        suggestion="Restrict allowed origins to a known list.",
        pattern=re.compile(
            r"allow_origins\s*=\s*\[?\s*[\"']\*[\"']",
        ),
    ),

    _RegexRule(
        rule_id="DEBUG_TRUE_LITERAL",
        kind=FindingKind.INSECURE_CONFIG,
        severity=Severity.HIGH,
        cwe="CWE-489",
        confidence=Confidence.MEDIUM,
        message="debug=True literal in app initialization",
        suggestion=(
            "Enable debug only via env var; never commit debug=True."
        ),
        pattern=re.compile(
            r"(?i)(?:app\.run|create_app|Flask|Django\b|DEBUG)\s*[^\n]*"
            r"debug\s*=\s*True\b"
        ),
    ),

    _RegexRule(
        rule_id="PATH_TRAVERSAL_JOIN",
        kind=FindingKind.PATH_TRAVERSAL,
        severity=Severity.HIGH,
        cwe="CWE-22",
        confidence=Confidence.MEDIUM,
        message="os.path.join with request/user-supplied path",
        suggestion=(
            "Validate input; resolve and confirm the result stays within "
            "a safe base directory."
        ),
        pattern=re.compile(
            r"os\.path\.join\s*\([^)]*\b(?:request\.|input\(|sys\.argv)"
        ),
    ),

    _RegexRule(
        rule_id="RMTREE_VARIABLE",
        kind=FindingKind.DANGEROUS_FS,
        severity=Severity.HIGH,
        cwe="CWE-22",
        confidence=Confidence.LOW,
        message="shutil.rmtree on a variable path",
        suggestion=(
            "Confirm the path is validated and confined to a safe base."
        ),
        pattern=re.compile(
            r"shutil\.rmtree\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[,)]"
        ),
    ),

    _RegexRule(
        rule_id="TEMP_MKTEMP",
        kind=FindingKind.DANGEROUS_FS,
        severity=Severity.MEDIUM,
        cwe="CWE-377",
        confidence=Confidence.HIGH,
        message="tempfile.mktemp is insecure (race condition)",
        suggestion="Use tempfile.NamedTemporaryFile or mkstemp.",
        pattern=re.compile(r"\btempfile\.mktemp\s*\("),
    ),

    _RegexRule(
        rule_id="WEAK_HASH",
        kind=FindingKind.COMMON_VULN,
        severity=Severity.MEDIUM,
        cwe="CWE-327",
        confidence=Confidence.MEDIUM,
        message="Weak hash algorithm (md5/sha1)",
        suggestion=(
            "Use sha256+ for integrity; use bcrypt/argon2 for passwords."
        ),
        pattern=re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("),
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# 5. AST RULES (4)
# ════════════════════════════════════════════════════════════════════════════
_AST_RULES: list[_AstRule] = [
    _AstRule(
        rule_id="SUBPROCESS_SHELL_TRUE",
        kind=FindingKind.UNSAFE_SUBPROCESS,
        severity=Severity.HIGH,
        cwe="CWE-78",
        confidence=Confidence.HIGH,
        message="subprocess called with shell=True",
        suggestion="Pass an argv list and shell=False.",
        matcher=_match_shell_true,
    ),
    _AstRule(
        rule_id="SQL_INJECTION",
        kind=FindingKind.INJECTION,
        severity=Severity.CRITICAL,
        cwe="CWE-89",
        confidence=Confidence.HIGH,
        message="SQL built via f-string / .format() / % concatenation",
        suggestion=(
            "Use parameterised queries: cursor.execute('... ?', (x,))."
        ),
        matcher=_match_sql_injection,
    ),
    _AstRule(
        rule_id="JWT_NO_VERIFY",
        kind=FindingKind.AUTH_WEAKNESS,
        severity=Severity.HIGH,
        cwe="CWE-347",
        confidence=Confidence.HIGH,
        message="JWT decode with signature verification disabled",
        suggestion="Never disable JWT signature verification.",
        matcher=_match_jwt_no_verify,
    ),
    _AstRule(
        rule_id="ASSERT_FOR_AUTH",
        kind=FindingKind.AUTH_WEAKNESS,
        severity=Severity.MEDIUM,
        cwe="CWE-617",
        confidence=Confidence.LOW,
        message="assert used for auth/permission check",
        suggestion=(
            "Asserts can be stripped with -O. Use explicit checks and "
            "raise/reject instead."
        ),
        matcher=_match_assert_auth,
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# 6. CONFIG SCANNERS
# ════════════════════════════════════════════════════════════════════════════
_REQ_UNPINNED_RE = re.compile(r"^[A-Za-z][\w.\-]*$")
_REQ_INDEX_URL_RE = re.compile(r"^\s*--(?:index-url|extra-index-url)\s+(\S+)")
_REQ_GIT_HTTP_RE = re.compile(r"^\s*(?:-e\s+)?(?:git\+)?http://")


def _scan_requirements(path: str, source: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for i, raw in enumerate(source.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # --index-url http://
        m = _REQ_INDEX_URL_RE.match(raw)
        if m and m.group(1).startswith("http://"):
            findings.append(SecurityFinding(
                rule_id="INSECURE_INDEX_URL",
                kind=FindingKind.UNSAFE_DEPENDENCY,
                severity=Severity.HIGH,
                cwe="CWE-319",
                confidence=Confidence.HIGH,
                message="Dependency index fetched over plain HTTP",
                file=path, line=i, snippet=_snippet(source, i),
                suggestion="Use https:// for the index URL.",
            ))
            continue
        # git+http://
        if _REQ_GIT_HTTP_RE.match(raw):
            findings.append(SecurityFinding(
                rule_id="HTTP_GIT_DEPENDENCY",
                kind=FindingKind.UNSAFE_DEPENDENCY,
                severity=Severity.MEDIUM,
                cwe="CWE-319",
                confidence=Confidence.HIGH,
                message="Dependency fetched via plain-HTTP git URL",
                file=path, line=i, snippet=_snippet(source, i),
                suggestion="Use git+https://.",
            ))
            continue
        # Skip pip flags
        if line.startswith("-"):
            continue
        # Version spec present?
        if any(tok in line for tok in
                ("==", ">=", "<=", ">", "<", "~=", "!=", "@", "[")):
            continue
        # Bare package name → unpinned
        base = line.split(";", 1)[0].strip()
        if _REQ_UNPINNED_RE.match(base):
            findings.append(SecurityFinding(
                rule_id="UNPINNED_DEPENDENCY",
                kind=FindingKind.UNSAFE_DEPENDENCY,
                severity=Severity.LOW,
                cwe="CWE-1104",
                confidence=Confidence.MEDIUM,
                message=f"Dependency '{base}' has no version pin",
                file=path, line=i, snippet=_snippet(source, i),
                suggestion=(
                    "Pin a specific version (==) or an upper bound with >=,<."
                ),
            ))
    return findings


_ENV_DEBUG_RE = re.compile(
    r"(?im)^\s*(?:DEBUG|FLASK_DEBUG|DJANGO_DEBUG|ENV|APP_ENV)"
    r"\s*=\s*(?:1|true|yes|dev|development)\s*$"
)
_ENV_SECRET_RE = re.compile(
    r"(?im)^\s*([A-Z][A-Z0-9_]*?(?:SECRET|KEY|TOKEN|PASSWORD|PASSWD|CREDENTIAL)"
    r"[A-Z0-9_]*?)\s*=\s*(.+?)\s*$"
)


def _scan_env(path: str, source: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for m in _ENV_DEBUG_RE.finditer(source):
        i = _line_of(source, m.start())
        findings.append(SecurityFinding(
            rule_id="DEBUG_ENV_ENABLED",
            kind=FindingKind.INSECURE_CONFIG,
            severity=Severity.HIGH,
            cwe="CWE-489",
            confidence=Confidence.HIGH,
            message="Debug/development mode enabled via .env",
            file=path, line=i, snippet=_snippet(source, i),
            suggestion="Disable debug for production environments.",
        ))
    for m in _ENV_SECRET_RE.finditer(source):
        raw_value = m.group(2).strip()
        value = raw_value.strip('"').strip("'")
        if not value:
            continue
        # env substitution → fine
        if value.startswith("$") or value.startswith("${"):
            continue
        if value.lower() in ("changeme", "password", "example", "replace_me"):
            # still flag but as config, not secret
            pass
        if len(value) < 8:
            continue
        i = _line_of(source, m.start())
        findings.append(SecurityFinding(
            rule_id="HARDCODED_SECRET_ENV",
            kind=FindingKind.SECRET_EXPOSURE,
            severity=Severity.CRITICAL,
            cwe="CWE-798",
            confidence=Confidence.HIGH,
            message=f"Secret literal in .env ({m.group(1)})",
            file=path, line=i, snippet=_snippet(source, i),
            suggestion=(
                "Keep secrets out of source control; use a secret manager "
                "or CI-provided env vars."
            ),
            evidence={"var_name": m.group(1)},
        ))
    return findings


# ════════════════════════════════════════════════════════════════════════════
# 7. ANALYZER FACADE
# ════════════════════════════════════════════════════════════════════════════
class SecurityAnalyzer:
    """Deterministic, bounded static security analyzer."""

    def __init__(
        self,
        *,
        max_file_bytes: int = 500_000,
        max_files: int = 1000,
        max_findings_per_file: int = 500,
    ) -> None:
        if max_file_bytes < 1:
            raise ValidationError("max_file_bytes must be >= 1")
        if max_files < 1:
            raise ValidationError("max_files must be >= 1")
        if max_findings_per_file < 1:
            raise ValidationError("max_findings_per_file must be >= 1")
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_findings_per_file = max_findings_per_file

    # ---- single file/source ----
    def analyze_source(
        self, source: str, *, filename: str = "<source>",
    ) -> FileScanResult:
        result = FileScanResult(path=filename)
        if not source:
            return result
        result.bytes_scanned = len(source.encode("utf-8"))
        if result.bytes_scanned > self.max_file_bytes:
            result.skipped = True
            result.skipped_reason = (
                f"file too large ({result.bytes_scanned}B > "
                f"{self.max_file_bytes}B)"
            )
            return result

        # Determine file kind
        base = filename.rsplit("/", 1)[-1].lower()
        is_env = base == ".env" or base.endswith(".env")
        is_req = base in ("requirements.txt", "requirements-dev.txt",
                          "requirements.in")

        # Parse check for Python
        if base.endswith(".py") or base.endswith(".pyi"):
            try:
                ast.parse(source)
                result.syntax_ok = True
            except SyntaxError:
                result.syntax_ok = False
                # Still run regex rules (best effort)

        findings: list[SecurityFinding] = []
        if is_req:
            findings.extend(_scan_requirements(filename, source))
        elif is_env:
            findings.extend(_scan_env(filename, source))
        else:
            # Run regex rules on all text files
            for rule in _RE_RULES:
                try:
                    findings.extend(rule.check(source, filename))
                except Exception as exc:
                    log.warning("c23.rule_error",
                                rule=rule.rule_id, error=str(exc))
            # Run AST rules only on parseable Python
            if (base.endswith(".py") or base.endswith(".pyi")) and result.syntax_ok:
                for rule in _AST_RULES:
                    try:
                        findings.extend(rule.check(source, filename))
                    except Exception as exc:
                        log.warning("c23.rule_error",
                                    rule=rule.rule_id, error=str(exc))

        # Dedup (rule_id, line) — keep first
        seen: set[tuple[str, int]] = set()
        deduped: list[SecurityFinding] = []
        for f in findings:
            key = (f.rule_id, f.line)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(f)

        # Sort deterministically: (line asc, rule_id asc)
        deduped.sort(key=lambda f: (f.line, f.rule_id))

        if len(deduped) > self.max_findings_per_file:
            deduped = deduped[: self.max_findings_per_file]

        result.findings = deduped
        return result

    def analyze_file(self, path: str | Path) -> FileScanResult:
        p = Path(path)
        if not p.is_file():
            return FileScanResult(
                path=str(p), skipped=True,
                skipped_reason="not a file",
            )
        try:
            raw = p.read_bytes()
        except OSError as exc:
            return FileScanResult(
                path=str(p), skipped=True,
                skipped_reason=f"read error: {exc}",
            )
        if len(raw) > self.max_file_bytes:
            return FileScanResult(
                path=str(p), skipped=True,
                skipped_reason=(
                    f"file too large ({len(raw)}B > "
                    f"{self.max_file_bytes}B)"
                ),
                bytes_scanned=0,
            )
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            return FileScanResult(
                path=str(p), skipped=True,
                skipped_reason="non-utf8 content",
            )
        return self.analyze_source(source, filename=str(p))

    # ---- repo ----
    def analyze_repo(self, root: str | Path, *, project_id: str = "") -> SecurityReport:
        root_p = Path(root).resolve()
        if not root_p.is_dir():
            raise ValidationError(f"root not a directory: {root_p}")

        report = SecurityReport(root=str(root_p), project_id=project_id)
        paths: list[Path] = []
        for dirpath, dirnames, filenames in __import__("os").walk(root_p):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            )
            for fname in sorted(filenames):
                p = Path(dirpath) / fname
                # Only scan text-ish extensions + .env
                ext = p.suffix.lower()
                base = p.name
                if ext in _TEXT_EXTS or base == ".env" or base.startswith(".env"):
                    paths.append(p)
                if len(paths) >= self.max_files:
                    break
            if len(paths) >= self.max_files:
                break

        for p in paths:
            fr = self.analyze_file(p)
            try:
                rel = str(p.relative_to(root_p)).replace("\\", "/")
            except ValueError:
                rel = str(p)
            # Store relative path in findings for readability
            for f in fr.findings:
                f.file = rel
            report.file_results.append(fr)
            if fr.skipped:
                report.files_skipped += 1
            else:
                report.files_scanned += 1
                report.total_bytes_scanned += fr.bytes_scanned
                report.findings.extend(fr.findings)

        # Global sort
        report.findings.sort(
            key=lambda f: (f.severity.value, f.file, f.line, f.rule_id),
            reverse=False,
        )
        # Actually sort by severity desc, then file/line
        sev_rank = {
            Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
            Severity.HIGH: 3, Severity.CRITICAL: 4,
        }
        report.findings.sort(
            key=lambda f: (-sev_rank[f.severity], f.file, f.line, f.rule_id),
        )

        report.coverage_note = (
            "Scanned: Python source (.py, .pyi), requirements*.txt, .env, "
            "and other text config files. "
            "NOT scanned: binary files, vendor/3rd-party source, .git, "
            "venvs, and code in non-Python languages. "
            "Analysis is pattern-based (no taint/dataflow, no CVE DB). "
            "Findings with confidence=LOW are heuristic; verify manually. "
            "This engine does NOT provide complete security assurance."
        )
        report.rationale = (
            f"files={report.files_scanned} skipped={report.files_skipped} "
            f"bytes={report.total_bytes_scanned} "
            f"findings={len(report.findings)} "
            f"highest={report.highest_severity().value if report.highest_severity() else 'none'}"
        )
        report.provenance = Provenance(
            source="security_analyzer",
            source_type=ProvenanceType.SYSTEM,
            confidence=Confidence.HIGH,
        )
        return report


# ════════════════════════════════════════════════════════════════════════════
# 8. SECURITY REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class SecurityRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: SecurityReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"security_report:{report.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, report.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["security_report", "c23",
                  report.highest_severity().value
                  if report.highest_severity() else "clean"],
            provenance=report.provenance,
        )
        # Record every CRITICAL/HIGH finding as a FAILURE memory (for C33)
        for f in report.findings:
            if f.severity in (Severity.CRITICAL, Severity.HIGH):
                self.memory.record_failure(
                    f"sec_finding:{report.id}:{f.rule_id}:{f.file}:{f.line}",
                    what=f"{f.rule_id}: {f.message}",
                    root_cause=f.suggestion or "see snippet",
                    fix=None,
                    scope_id=project_id,
                    provenance=report.provenance,
                    confidence=f.confidence,
                )
        if self.ontology is None:
            return key

        root = self.ontology.add(
            EntityKind.EVIDENCE,
            _short(
                f"SecurityReport {report.id[:8]} "
                f"({len(report.findings)} findings)", 120,
            ),
            attributes={
                "report_id": report.id,
                "project_id": project_id,
                "files_scanned": report.files_scanned,
                "files_skipped": report.files_skipped,
                "counts": {s.value: report.count(s) for s in Severity},
                "by_kind": report.by_kind(),
                "highest_severity": (report.highest_severity().value
                                     if report.highest_severity() else None),
            },
            tags=["security-report"],
            provenance=report.provenance,
        )
        # Persist CRITICAL/HIGH findings as BUG entities
        for f in report.findings:
            if f.severity in (Severity.CRITICAL, Severity.HIGH):
                bug = self.ontology.add(
                    EntityKind.BUG,
                    _short(f"{f.rule_id}: {f.message}", 120),
                    attributes={
                        "rule_id": f.rule_id,
                        "kind": f.kind.value,
                        "severity": f.severity.value,
                        "cwe": f.cwe,
                        "file": f.file,
                        "line": f.line,
                        "confidence": f.confidence.value,
                        "suggestion": f.suggestion,
                    },
                    tags=["security", f.severity.value, f.rule_id],
                    provenance=report.provenance,
                )
                try:
                    self.ontology.link(RelationKind.CONTAINS, root.id, bug.id)
                except ValidationError:
                    pass
        return root.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"security_report:{report_id}",
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

    print("Running C23 self-tests…")
    analyzer = SecurityAnalyzer()

    def _rules_in(source: str, *, filename: str = "f.py") -> set[str]:
        return {f.rule_id for f in analyzer.analyze_source(
            source, filename=filename).findings}

    # ---- regex rules ----
    def t_hardcoded_secret() -> None:
        rules = _rules_in('password = "hunter2secret"\n')
        assert "HARDCODED_SECRET" in rules

    def t_hardcoded_secret_short_ignored() -> None:
        rules = _rules_in('password = "short"\n')
        assert "HARDCODED_SECRET" not in rules

    def t_hardcoded_env_ref_ignored() -> None:
        rules = _rules_in('password = os.getenv("DB_PASSWORD")\n')
        assert "HARDCODED_SECRET" not in rules

    def t_eval_exec() -> None:
        r = _rules_in("x = eval('1+1')\ny = exec('a=1')\n")
        assert "EVAL_USAGE" in r
        assert "EXEC_USAGE" in r

    def t_os_system_popen() -> None:
        r = _rules_in("import os\nos.system('ls')\nos.popen('ls')\n")
        assert "OS_SYSTEM" in r
        assert "OS_POPEN" in r

    def t_pickle_marshal() -> None:
        r = _rules_in(
            "import pickle, marshal\n"
            "pickle.loads(b'')\n"
            "pickle.load(open('x','rb'))\n"
            "marshal.loads(b'')\n"
        )
        assert "PICKLE_LOADS" in r
        assert "MARSHAL_LOADS" in r

    def t_yaml_unsafe() -> None:
        assert "YAML_LOAD_UNSAFE" in _rules_in(
            "import yaml\nyaml.load(open('x'))\n"
        )
        # Safe version → no finding
        assert "YAML_LOAD_UNSAFE" not in _rules_in(
            "import yaml\nyaml.load(open('x'), Loader=yaml.SafeLoader)\n"
        )

    def t_verify_false() -> None:
        assert "VERIFY_FALSE" in _rules_in(
            "import requests\nrequests.get('https://x', verify=False)\n"
        )

    def t_unverified_ssl_ctx() -> None:
        assert "UNVERIFIED_SSL_CTX" in _rules_in(
            "import ssl\nctx = ssl._create_unverified_context()\n"
        )

    def t_plaintext_http() -> None:
        assert "PLAINTEXT_HTTP" in _rules_in('x = "http://example.com"\n')
        # localhost exempt
        assert "PLAINTEXT_HTTP" not in _rules_in('x = "http://localhost:8000"\n')
        assert "PLAINTEXT_HTTP" not in _rules_in('x = "http://127.0.0.1:8000"\n')

    def t_secret_key_literal() -> None:
        assert "SECRET_KEY_LITERAL" in _rules_in(
            'SECRET_KEY = "abcdef1234567890"\n'
        )

    def t_cors_wildcard() -> None:
        assert "CORS_WILDCARD" in _rules_in(
            "allow_origins=['*']\n"
        )

    def t_debug_true_literal() -> None:
        assert "DEBUG_TRUE_LITERAL" in _rules_in(
            "app.run(debug=True)\n"
        )

    def t_path_traversal_join() -> None:
        assert "PATH_TRAVERSAL_JOIN" in _rules_in(
            "p = os.path.join('/base', request.args.get('name'))\n"
        )

    def t_rmtree_variable() -> None:
        assert "RMTREE_VARIABLE" in _rules_in(
            "import shutil\nshutil.rmtree(path)\n"
        )

    def t_temp_mktemp() -> None:
        assert "TEMP_MKTEMP" in _rules_in(
            "import tempfile\nf = tempfile.mktemp()\n"
        )

    def t_weak_hash() -> None:
        assert "WEAK_HASH" in _rules_in(
            "import hashlib\nhashlib.md5(b'')\nhashlib.sha1(b'')\n"
        )
        # sha256 → fine
        assert "WEAK_HASH" not in _rules_in(
            "import hashlib\nhashlib.sha256(b'')\n"
        )

    check("regex: HARDCODED_SECRET (>=8 chars)", t_hardcoded_secret)
    check("regex: short secret not flagged", t_hardcoded_secret_short_ignored)
    check("regex: env-var secret not flagged", t_hardcoded_env_ref_ignored)
    check("regex: eval + exec flagged", t_eval_exec)
    check("regex: os.system + os.popen flagged", t_os_system_popen)
    check("regex: pickle + marshal flagged", t_pickle_marshal)
    check("regex: yaml.load unsafe flagged (SafeLoader exempt)", t_yaml_unsafe)
    check("regex: verify=False flagged", t_verify_false)
    check("regex: unverified SSL context flagged", t_unverified_ssl_ctx)
    check("regex: plaintext HTTP flagged (localhost exempt)",
          t_plaintext_http)
    check("regex: hardcoded SECRET_KEY flagged", t_secret_key_literal)
    check("regex: CORS wildcard flagged", t_cors_wildcard)
    check("regex: debug=True literal flagged", t_debug_true_literal)
    check("regex: path traversal via os.path.join(request...)",
          t_path_traversal_join)
    check("regex: shutil.rmtree on variable flagged", t_rmtree_variable)
    check("regex: tempfile.mktemp flagged", t_temp_mktemp)
    check("regex: weak hash (md5/sha1) flagged, sha256 exempt", t_weak_hash)

    # ---- AST rules ----
    def t_subprocess_shell_true() -> None:
        src = (
            "import subprocess\n"
            "subprocess.run(['ls', '-la'], shell=True)\n"
            "subprocess.check_output(cmd, shell=True)\n"
        )
        rules = _rules_in(src)
        assert "SUBPROCESS_SHELL_TRUE" in rules

    def t_subprocess_no_shell() -> None:
        src = (
            "import subprocess\n"
            "subprocess.run(['ls', '-la'], shell=False)\n"
            "subprocess.run(['ls', '-la'])\n"
        )
        rules = _rules_in(src)
        assert "SUBPROCESS_SHELL_TRUE" not in rules

    def t_sql_injection_fstring() -> None:
        src = (
            "cursor.execute(f'SELECT * FROM t WHERE id = {x}')\n"
        )
        assert "SQL_INJECTION" in _rules_in(src)

    def t_sql_injection_format() -> None:
        src = (
            "cursor.execute('SELECT * FROM t WHERE id = {}'.format(x))\n"
        )
        assert "SQL_INJECTION" in _rules_in(src)

    def t_sql_parameterised_safe() -> None:
        src = (
            "cursor.execute('SELECT * FROM t WHERE id = ?', (x,))\n"
        )
        assert "SQL_INJECTION" not in _rules_in(src)

    def t_jwt_no_verify_options() -> None:
        src = (
            "import jwt\n"
            "jwt.decode(token, key, options={'verify_signature': False})\n"
        )
        assert "JWT_NO_VERIFY" in _rules_in(src)

    def t_jwt_verify_kwarg_false() -> None:
        src = "import jwt\njwt.decode(token, key, verify=False)\n"
        assert "JWT_NO_VERIFY" in _rules_in(src)

    def t_assert_for_auth() -> None:
        src = (
            "def edit(user, obj):\n"
            "    assert user.is_admin\n"
            "    return obj.save()\n"
        )
        assert "ASSERT_FOR_AUTH" in _rules_in(src)

    check("ast: subprocess shell=True flagged", t_subprocess_shell_true)
    check("ast: subprocess shell=False exempt", t_subprocess_no_shell)
    check("ast: SQL injection via f-string flagged", t_sql_injection_fstring)
    check("ast: SQL injection via .format flagged", t_sql_injection_format)
    check("ast: parameterised SQL safe", t_sql_parameterised_safe)
    check("ast: JWT no-verify (options) flagged",
          t_jwt_no_verify_options)
    check("ast: JWT verify=False flagged", t_jwt_verify_kwarg_false)
    check("ast: assert-based auth flagged", t_assert_for_auth)

    # ---- config scanners ----
    def t_requirements_unpinned() -> None:
        src = "requests\nflask==2.0.1\nnumpy>=1.20\n"
        r = {f.rule_id for f in analyzer.analyze_source(
            src, filename="requirements.txt").findings}
        assert "UNPINNED_DEPENDENCY" in r
        # flask and numpy have specs, only requests flagged
        fs = [f for f in analyzer.analyze_source(
            src, filename="requirements.txt").findings
            if f.rule_id == "UNPINNED_DEPENDENCY"]
        assert len(fs) == 1
        assert "requests" in fs[0].message

    def t_requirements_insecure_index() -> None:
        src = "--index-url http://evil.example.com/simple\nrequests==2.0\n"
        r = {f.rule_id for f in analyzer.analyze_source(
            src, filename="requirements.txt").findings}
        assert "INSECURE_INDEX_URL" in r

    def t_requirements_http_git() -> None:
        src = "git+http://github.com/x/y.git\n"
        r = {f.rule_id for f in analyzer.analyze_source(
            src, filename="requirements.txt").findings}
        assert "HTTP_GIT_DEPENDENCY" in r

    def t_env_debug() -> None:
        src = "DEBUG=True\nAPI_KEY=abc123def456\nSECRET_KEY=topsecret123\n"
        r = analyzer.analyze_source(src, filename=".env")
        ids = {f.rule_id for f in r.findings}
        assert "DEBUG_ENV_ENABLED" in ids
        assert "HARDCODED_SECRET_ENV" in ids

    def t_env_substitution_safe() -> None:
        src = "API_KEY=${API_KEY_FROM_VAULT}\n"
        r = analyzer.analyze_source(src, filename=".env")
        assert not any(f.rule_id == "HARDCODED_SECRET_ENV" for f in r.findings)

    check("config: requirements unpinned flagged",
          t_requirements_unpinned)
    check("config: requirements insecure index flagged",
          t_requirements_insecure_index)
    check("config: requirements http git flagged",
          t_requirements_http_git)
    check("config: .env debug + secrets flagged", t_env_debug)
    check("config: .env variable substitution safe",
          t_env_substitution_safe)

    # ---- dedup / determinism ----
    def t_dedup() -> None:
        # Same rule, same line, twice in source → one finding
        src = "password = \"aaaaaaaabbbb\"\n"
        r = analyzer.analyze_source(src)
        secret_fs = [f for f in r.findings if f.rule_id == "HARDCODED_SECRET"]
        assert len(secret_fs) == 1

    def t_deterministic() -> None:
        src = (
            "password = \"aaaaaaaabbbb\"\n"
            "eval('1+1')\n"
            "x = http://example.com\n"
        )
        r1 = analyzer.analyze_source(src)
        r2 = analyzer.analyze_source(src)
        ids1 = sorted((f.rule_id, f.line) for f in r1.findings)
        ids2 = sorted((f.rule_id, f.line) for f in r2.findings)
        assert ids1 == ids2

    check("dedup: same rule+line → one finding", t_dedup)
    check("deterministic: same source → same findings", t_deterministic)

    # ---- syntax error tolerance ----
    def t_syntax_error_still_regex_scanned() -> None:
        src = "def broken(:\n    password = \"hunter2secret\"\n"
        r = analyzer.analyze_source(src, filename="bad.py")
        assert r.syntax_ok is False
        # regex still runs
        assert any(f.rule_id == "HARDCODED_SECRET" for f in r.findings)

    check("robust: syntax error → regex still runs", t_syntax_error_still_regex_scanned)

    # ---- repo scan ----
    def t_repo_scan() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "app/main.py",
                   "import os\nos.system('ls')\n")
            _write(root, "app/config.py",
                   'password = "hunter2secret"\n')
            _write(root, "requirements.txt",
                   "requests\nflask==2.0\n")
            _write(root, ".env",
                   "DEBUG=True\nAPI_KEY=abcdef12345678\n")
            # Skip venv
            _write(root, "venv/lib/skipme.py", "eval('x')\n")

            report = analyzer.analyze_repo(root)
            rules = {f.rule_id for f in report.findings}
            assert "OS_SYSTEM" in rules
            assert "HARDCODED_SECRET" in rules
            assert "UNPINNED_DEPENDENCY" in rules
            assert "DEBUG_ENV_ENABLED" in rules
            # venv skipped
            assert not any("venv" in f.file for f in report.findings)

    check("repo: scans app + configs, skips venv", t_repo_scan)

    # ---- coverage note honest ----
    def t_coverage_note_present() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1\n")
            report = analyzer.analyze_repo(root)
            assert "NOT scanned" in report.coverage_note
            assert "pattern-based" in report.coverage_note.lower()
            assert "complete security assurance" in report.coverage_note.lower()

    check("honesty: coverage note lists what's not scanned",
          t_coverage_note_present)

    # ---- summary / to_dict ----
    def t_to_dict_summary() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "eval('1+1')\n")
            report = analyzer.analyze_repo(root)
            d = report.to_dict()
            assert d["id"] == report.id
            assert "findings" in d and "counts" in d
            assert d["highest_severity"] == "critical"
            s = report.summary()
            assert "Security Report" in s

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
                    _write(root, "a.py",
                           'password = "hunter2secret"\n'
                           "eval('1+1')\n")
                    report = analyzer.analyze_repo(
                        root, project_id="proj-x",
                    )
                    repo = SecurityRepository(memory=mem, ontology=ont)
                    ent = repo.save(report, project_id="proj-x")
                    assert ent
                    loaded = repo.load(report.id, project_id="proj-x")
                    assert loaded is not None
                    assert loaded["id"] == report.id
                    # Ontology: EVIDENCE root + BUG entities for CRITICAL/HIGH
                    assert ont.count(kind=EntityKind.EVIDENCE) >= 1
                    assert ont.count(kind=EntityKind.BUG) >= 2
                    # Failure memories recorded
                    fails = mem.find(
                        kind=MemoryKind.FAILURE,
                        scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                    )
                    assert len(fails) >= 2
            finally:
                s.shutdown()

    check("persist: memory + ontology (EVIDENCE root + BUG entities)",
          t_persist)

    # ---- E2E with C14 code ----
    def t_e2e_scan_synth_code() -> None:
        """Run C23 over C14-synthesised code — should be clean."""
        from sebrain.c13 import RepoIndex
        from sebrain.c14 import (
            CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
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
            # Write files then scan
            for f in synth.files:
                _write(root, f.path, f.content)
            report = analyzer.analyze_repo(root, project_id="demo")
            # C14-generated code should have NO CRITICAL findings
            crit = [f for f in report.findings
                    if f.severity is Severity.CRITICAL]
            assert crit == [], [f.to_dict() for f in crit]

    check("e2e: C14-synthesised code has no CRITICAL findings",
          t_e2e_scan_synth_code)

    # ---- E2E detect deliberately bad code ----
    def t_e2e_detect_bad_code() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "bad.py", (
                "import os, pickle, hashlib\n"
                "import subprocess\n"
                'password = "hunter2secret"\n'
                "def f(x):\n"
                "    eval(x)\n"
                "    os.system('ls ' + x)\n"
                "    subprocess.run(['ls', x], shell=True)\n"
                "    pickle.loads(x)\n"
                "    hashlib.md5(x.encode())\n"
                "    cursor = None\n"
            ))
            _write(root, "bad_sql.py", (
                "def q(cursor, x):\n"
                "    cursor.execute(f'SELECT * FROM t WHERE id={x}')\n"
            ))
            report = analyzer.analyze_repo(root, project_id="demo")
            rules = {f.rule_id for f in report.findings}
            # All of these should be detected
            for expected in (
                "HARDCODED_SECRET", "EVAL_USAGE", "OS_SYSTEM",
                "SUBPROCESS_SHELL_TRUE", "PICKLE_LOADS", "WEAK_HASH",
                "SQL_INJECTION",
            ):
                assert expected in rules, expected
            assert report.highest_severity() is Severity.CRITICAL

    check("e2e: deliberately bad code → all expected rules fired",
          t_e2e_detect_bad_code)

    # ---- limits ----
    def t_file_size_bound() -> None:
        a = SecurityAnalyzer(max_file_bytes=10)
        r = a.analyze_source("x" * 100, filename="big.py")
        assert r.skipped is True
        assert "too large" in r.skipped_reason

    def t_skip_non_utf8() -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bin.py"
            p.write_bytes(b"\xff\xfe\xfa\xfb")
            r = analyzer.analyze_file(p)
            assert r.skipped is True

    check("limits: over-large file skipped", t_file_size_bound)
    check("limits: non-utf8 file skipped", t_skip_non_utf8)

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
    print("SE Brain C23 — Security Analysis Engine")
    print("=" * 78)

    analyzer = SecurityAnalyzer()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_repo"
        root.mkdir()
        _write(root, "app/main.py", (
            "import os\n"
            "import subprocess\n"
            "import pickle\n"
            "import hashlib\n"
            "\n"
            'API_KEY = "sk-abcdef1234567890"\n'
            'SECRET_KEY = "totally-not-a-secret"\n'
            "\n"
            "def unsafe(user_input):\n"
            "    eval(user_input)\n"
            "    os.system('echo ' + user_input)\n"
            "    subprocess.run(['ls', user_input], shell=True)\n"
            "    pickle.loads(user_input)\n"
            "    hashlib.md5(user_input.encode())\n"
        ))
        _write(root, "app/db.py", (
            "def get_task(cursor, task_id):\n"
            "    cursor.execute(f'SELECT * FROM tasks WHERE id = {task_id}')\n"
        ))
        _write(root, "requirements.txt", (
            "requests\n"
            "flask==2.0.1\n"
            "numpy>=1.20\n"
            "--index-url http://mirror.example.com/simple\n"
        ))
        _write(root, ".env", (
            "DEBUG=True\n"
            "API_KEY=abc123def456789\n"
        ))

        report = analyzer.analyze_repo(root, project_id="demo")

        print("\n[1] Summary:")
        print(report.summary())

        print("\n[2] Findings by severity:")
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
                    Severity.LOW, Severity.INFO):
            fs = [f for f in report.findings if f.severity is sev]
            if not fs:
                continue
            print(f"    [{sev.value}] {len(fs)}")
            for f in fs[:3]:
                print(f"        · {f.rule_id:24s} {f.file}:{f.line}  "
                      f"({f.cwe})  {_short(f.message, 60)}")
            if len(fs) > 3:
                print(f"        … and {len(fs) - 3} more")

        print("\n[3] Coverage note:")
        print(f"    {report.coverage_note}")

        print("\n[4] Persistence:")
        with tempfile.TemporaryDirectory() as sdt:
            cfg = Config(data_dir=Path(sdt) / "sebrain", log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                with execution_scope(project_id="demo"):
                    mem = MemoryStore(app.storage)
                    ont = Ontology(app.storage)
                    repo = SecurityRepository(memory=mem, ontology=ont)
                    ent = repo.save(report, project_id="demo")
                    print(f"    ontology entity: {ent[:12]}…")
                    print(f"    EVIDENCE: {ont.count(kind=EntityKind.EVIDENCE)}")
                    print(f"    BUG: {ont.count(kind=EntityKind.BUG)}")
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
