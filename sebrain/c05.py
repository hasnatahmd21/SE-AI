"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C05 — REQUIREMENT UNDERSTANDING ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01 (`sebrain/c01.py`), C02 (`sebrain/c02.py`), C04 (`sebrain/c04.py`).

Purpose:
    Convert natural-language requirement text into a STRUCTURED specification.

Extracts:
    objective · scope_in · scope_out · functional requirements ·
    non-functional requirements (categorised: performance, security, ...) ·
    constraints · inputs · outputs · users · dependencies ·
    acceptance criteria · risks · assumptions

Also produces:
    ambiguities (vague terms, undefined acronyms, placeholders, unresolved pronouns)
    missing-information report (per aspect)

Invariants honored:
  - Pure deterministic rule-based parser — NO external LLM.
  - Missing requirements are NOT invented. They are reported under `missing`.
  - Unverifiable statements are marked as `assumptions` with explicit origin.
  - Every extracted item carries Provenance (C02) + Confidence (C01).
  - Section headers + sentence patterns are used together; duplicates removed.
  - Persistence via C04 memory (PROJECT scope) + C02 ontology (REQUIREMENT entity).

Contents:
  1.  Enums: RequirementKind / AmbiguityKind / MissingKind
  2.  Dataclasses: ExtractedItem / Ambiguity / MissingInfo / Assumption /
                   RequirementSpec
  3.  Tokenizer + sentence splitter + section extractor
  4.  Pattern tables (NFR categories, vague terms, markers, ...)
  5.  RequirementParser (deterministic rules)
  6.  RequirementRepository (save/load via C04 memory + C02 ontology)
  7.  __main__ demo + self-tests

Run as script:
    python -m sebrain.c05            # demo
    python -m sebrain.c05 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import re
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(".!?").lower())


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class RequirementKind(str, Enum):
    OBJECTIVE = "objective"
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"
    INPUT = "input"
    OUTPUT = "output"
    DEPENDENCY = "dependency"
    ACCEPTANCE = "acceptance"
    RISK = "risk"
    ASSUMPTION = "assumption"


class AmbiguityKind(str, Enum):
    VAGUE_TERM = "vague_term"
    UNDEFINED_ACRONYM = "undefined_acronym"
    UNRESOLVED_PRONOUN = "unresolved_pronoun"
    PLACEHOLDER = "placeholder"


class MissingKind(str, Enum):
    OBJECTIVE = "objective"
    USERS = "users"
    SCOPE = "scope"
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINTS = "constraints"
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    ACCEPTANCE = "acceptance"
    DETAIL = "detail"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class ExtractedItem:
    text: str
    kind: RequirementKind
    source: str = "sentence"        # "sentence" | "section"
    tags: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind.value,
            "source": self.source,
            "tags": list(self.tags),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(slots=True)
class Ambiguity:
    text: str
    reason: str
    kind: AmbiguityKind
    term: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "reason": self.reason,
            "kind": self.kind.value, "term": self.term,
        }


@dataclass(slots=True)
class MissingInfo:
    kind: MissingKind
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "why": self.why}


@dataclass(slots=True)
class Assumption:
    text: str
    origin: str = "explicit"        # "explicit" | "inferred"
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "origin": self.origin,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(slots=True)
class RequirementSpec:
    id: str = field(default_factory=_new_id)
    raw_text: str = ""
    objective: ExtractedItem | None = None
    scope_in: list[ExtractedItem] = field(default_factory=list)
    scope_out: list[ExtractedItem] = field(default_factory=list)
    functional: list[ExtractedItem] = field(default_factory=list)
    non_functional: list[ExtractedItem] = field(default_factory=list)
    constraints: list[ExtractedItem] = field(default_factory=list)
    inputs: list[ExtractedItem] = field(default_factory=list)
    outputs: list[ExtractedItem] = field(default_factory=list)
    dependencies: list[ExtractedItem] = field(default_factory=list)
    acceptance_criteria: list[ExtractedItem] = field(default_factory=list)
    risks: list[ExtractedItem] = field(default_factory=list)
    users: list[str] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)
    missing: list[MissingInfo] = field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "objective": self.objective.to_dict() if self.objective else None,
            "scope_in": [x.to_dict() for x in self.scope_in],
            "scope_out": [x.to_dict() for x in self.scope_out],
            "functional": [x.to_dict() for x in self.functional],
            "non_functional": [x.to_dict() for x in self.non_functional],
            "constraints": [x.to_dict() for x in self.constraints],
            "inputs": [x.to_dict() for x in self.inputs],
            "outputs": [x.to_dict() for x in self.outputs],
            "dependencies": [x.to_dict() for x in self.dependencies],
            "acceptance_criteria": [x.to_dict() for x in self.acceptance_criteria],
            "risks": [x.to_dict() for x in self.risks],
            "users": list(self.users),
            "assumptions": [a.to_dict() for a in self.assumptions],
            "ambiguities": [a.to_dict() for a in self.ambiguities],
            "missing": [m.to_dict() for m in self.missing],
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def counts(self) -> dict[str, int]:
        return {
            "scope_in": len(self.scope_in),
            "scope_out": len(self.scope_out),
            "functional": len(self.functional),
            "non_functional": len(self.non_functional),
            "constraints": len(self.constraints),
            "inputs": len(self.inputs),
            "outputs": len(self.outputs),
            "dependencies": len(self.dependencies),
            "acceptance_criteria": len(self.acceptance_criteria),
            "risks": len(self.risks),
            "users": len(self.users),
            "assumptions": len(self.assumptions),
            "ambiguities": len(self.ambiguities),
            "missing": len(self.missing),
        }

    def summary(self) -> str:
        lines = ["=== Requirement Specification ==="]
        if self.objective:
            lines.append(f"Objective: {self.objective.text}")
        else:
            lines.append("Objective: (not identified)")
        if self.users:
            lines.append(f"Users: {', '.join(self.users)}")
        lines.append(f"Confidence: {self.confidence.value}")
        lines.append("Counts: " + ", ".join(
            f"{k}={v}" for k, v in self.counts().items()
        ))
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. TOKENIZER / SENTENCE SPLITTER / SECTION EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+(.+?)\s*$")

# Section headers (order matters — specific first).
_SCOPE_IN_HEADER = re.compile(
    r"^\s*(?:in[\s_-]?scope|included|includes)\s*:?\s*$", re.IGNORECASE
)
_SCOPE_OUT_HEADER = re.compile(
    r"^\s*(?:out[\s_-]?of[\s_-]?scope|excluded|excludes|not\s+included)\s*:?\s*$",
    re.IGNORECASE,
)
_ACCEPTANCE_HEADER = re.compile(
    r"^\s*(?:acceptance(?:\s+criteria)?|definition\s+of\s+done|dod)\s*:?\s*$",
    re.IGNORECASE,
)
_RISK_HEADER = re.compile(
    r"^\s*(?:risks?|concerns?|issues?|known\s+risks?)\s*:?\s*$", re.IGNORECASE
)
_ASSUMPTION_HEADER = re.compile(r"^\s*assumptions?\s*:?\s*$", re.IGNORECASE)
_CONSTRAINT_HEADER = re.compile(r"^\s*constraints?\s*:?\s*$", re.IGNORECASE)
_INPUT_HEADER = re.compile(
    r"^\s*(?:inputs?|accepts|receives)\s*:?\s*$", re.IGNORECASE
)
_OUTPUT_HEADER = re.compile(
    r"^\s*(?:outputs?|returns?|produces?|responses?)\s*:?\s*$", re.IGNORECASE
)
_NFR_HEADER = re.compile(
    r"^\s*(?:non[\s_-]?functional(?:\s+requirements?)?|quality(?:\s+attributes?)?|nfr)\s*:?\s*$",
    re.IGNORECASE,
)
_FR_HEADER = re.compile(
    r"^\s*(?:functional(?:\s+requirements?)?|requirements?|features?)\s*:?\s*$",
    re.IGNORECASE,
)
_OBJECTIVE_HEADER = re.compile(
    r"^\s*(?:goal|objective|purpose|aim)\s*:?\s*$", re.IGNORECASE
)

_HEADER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_SCOPE_IN_HEADER, "scope_in"),
    (_SCOPE_OUT_HEADER, "scope_out"),
    (_ACCEPTANCE_HEADER, "acceptance"),
    (_RISK_HEADER, "risks"),
    (_ASSUMPTION_HEADER, "assumptions"),
    (_CONSTRAINT_HEADER, "constraints"),
    (_INPUT_HEADER, "inputs"),
    (_OUTPUT_HEADER, "outputs"),
    (_NFR_HEADER, "non_functional"),
    (_FR_HEADER, "functional"),
    (_OBJECTIVE_HEADER, "objective"),
]


def _classify_header(line: str) -> str | None:
    for pattern, name in _HEADER_PATTERNS:
        if pattern.match(line):
            return name
    return None


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, per line, ignoring bullets but keeping content."""
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _BULLET_RE.match(raw_line)
        body = m.group(1) if m else line
        parts = re.split(r"(?<=[.!?])\s+", body)
        for p in parts:
            p = p.strip()
            if len(p) >= 3:
                out.append(p)
    return out


def _extract_sections(text: str) -> dict[str, list[str]]:
    """Return {section_name: [content, ...]} based on headers + bullets."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            current = None
            continue
        header = _classify_header(line)
        if header is not None:
            current = header
            sections.setdefault(current, [])
            continue
        m = _BULLET_RE.match(raw_line)
        if m and current:
            sections[current].append(m.group(1).strip())
            continue
        if current:
            sections[current].append(line)
    return sections


# ════════════════════════════════════════════════════════════════════════════
# 4. PATTERN TABLES
# ════════════════════════════════════════════════════════════════════════════
NFR_PATTERNS: dict[str, list[re.Pattern]] = {
    "performance": [
        re.compile(r"\blatency\b", re.I),
        re.compile(r"\bthroughput\b", re.I),
        re.compile(r"\bresponse[\s_-]time\b", re.I),
        re.compile(r"\brequests?\s+per\s+second\b", re.I),
        re.compile(r"\bwithin\s+\d+\s*(?:ms|milliseconds?|seconds?|minutes?)\b", re.I),
        re.compile(r"\bunder\s+\d+\s*(?:ms|milliseconds?|seconds?|minutes?)\b", re.I),
        re.compile(r"\bfaster\s+than\b", re.I),
        re.compile(r"\bperformance\b", re.I),
    ],
    "scalability": [
        re.compile(r"\bscal(?:e|able|ability)\b", re.I),
        re.compile(r"\bconcurrent(?:ly)?\b", re.I),
        re.compile(r"\bhorizontally\b", re.I),
        re.compile(r"\bvertically\b", re.I),
        re.compile(r"\b\d+\s*(?:k\s*)?(?:users?|requests?|rps)\b", re.I),
    ],
    "security": [
        re.compile(r"\bsecur(?:e|ity)\b", re.I),
        re.compile(r"\bauthenticat(?:e|ion|ed)\b", re.I),
        re.compile(r"\bauthoriz(?:e|ation|ed)\b", re.I),
        re.compile(r"\bencrypt(?:ion|ed)?\b", re.I),
        re.compile(r"\bTLS\b"),
        re.compile(r"\bHTTPS\b", re.I),
        re.compile(r"\bSSL\b"),
        re.compile(r"\btoken\b", re.I),
        re.compile(r"\bpassword\b", re.I),
        re.compile(r"\bcredential\b", re.I),
        re.compile(r"\bOWASP\b"),
        re.compile(r"\bCSRF\b"),
        re.compile(r"\bXSS\b"),
        re.compile(r"\bSQL\s*injection\b", re.I),
    ],
    "reliability": [
        re.compile(r"\breliab(?:le|ility)\b", re.I),
        re.compile(r"\bavailab(?:le|ility)\b", re.I),
        re.compile(r"\buptime\b", re.I),
        re.compile(r"\bfailover\b", re.I),
        re.compile(r"\brecover(?:y|able)?\b", re.I),
        re.compile(r"\bbackup\b", re.I),
        re.compile(r"\bconsisten(?:t|cy)\b", re.I),
        re.compile(r"\bidempotent\b", re.I),
        re.compile(r"\bdurab(?:le|ility)\b", re.I),
        re.compile(r"\bproduction[\s_-]quality\b", re.I),
    ],
    "usability": [
        re.compile(r"\busab(?:le|ility)\b", re.I),
        re.compile(r"\buser[\s_-]friendly\b", re.I),
        re.compile(r"\bintuitive\b", re.I),
        re.compile(r"\baccessib(?:le|ility)\b", re.I),
        re.compile(r"\ba11y\b", re.I),
        re.compile(r"\bmobile[\s_-]friendly\b", re.I),
    ],
    "maintainability": [
        re.compile(r"\bmaintainab(?:le|ility)\b", re.I),
        re.compile(r"\btestab(?:le|ility)\b", re.I),
        re.compile(r"\bmodular\b", re.I),
        re.compile(r"\bextensib(?:le|ility)\b", re.I),
        re.compile(r"\breadab(?:le|ility)\b", re.I),
        re.compile(r"\btyped\b", re.I),
        re.compile(r"\bdocument(?:ed|ation)\b", re.I),
    ],
    "portability": [
        re.compile(r"\bportab(?:le|ility)\b", re.I),
        re.compile(r"\bcross[\s_-]platform\b", re.I),
        re.compile(r"\bdocker\b", re.I),
        re.compile(r"\bcontainer(?:ized|isation|ization)\b", re.I),
        re.compile(r"\bcloud[\s_-]agnostic\b", re.I),
    ],
    "compliance": [
        re.compile(r"\bGDPR\b", re.I),
        re.compile(r"\bHIPAA\b", re.I),
        re.compile(r"\bPCI(?:[\s_-]DSS)?\b", re.I),
        re.compile(r"\bSOX\b", re.I),
        re.compile(r"\bISO\s*27001\b", re.I),
        re.compile(r"\bcompliance\b", re.I),
        re.compile(r"\bregulatory\b", re.I),
        re.compile(r"\baudit\b", re.I),
    ],
}

VAGUE_TERMS: frozenset[str] = frozenset({
    "fast", "quick", "quickly", "slow", "prompt",
    "many", "few", "some", "several", "various", "etc",
    "appropriate", "sufficient", "reasonable", "adequate",
    "good", "bad", "nice", "great", "poor",
    "large", "small", "big", "tiny", "huge",
    "soon", "later", "recently",
    "robust", "flexible",
})

KNOWN_ACRONYMS: frozenset[str] = frozenset({
    "api", "rest", "http", "https", "url", "uri", "json", "xml",
    "yaml", "html", "css", "sql", "tcp", "udp", "ip", "os", "ui", "ux",
    "cli", "sdk", "id", "jwt", "oauth", "tls", "ssl", "crud", "db",
})

PLACEHOLDERS = ("TBD", "TODO", "FIXME", "XXX", "???")

CONSTRAINT_MARKERS_STRONG = [
    "must not", "cannot", "can't", "should not", "shall not",
]
CONSTRAINT_MARKERS_SOFT = [
    "no more than", "no less than", "at most", "at least",
    "up to ", "within ", "budget", "deadline",
    "limited to", "restricted to",
]
CONSTRAINT_MARKERS = CONSTRAINT_MARKERS_STRONG + CONSTRAINT_MARKERS_SOFT

FUNCTIONAL_MARKERS = [
    "must ", "shall ", "should ", "will ",
    "needs to ", "need to ", "has to ", "have to ",
    "is able to ", "are able to ",
    "can ", "can't ", "could ",
]

DEPENDENCY_MARKERS = [
    "depends on", "depend on", "depends upon",
    "requires ", "require ",
    "prerequisite", "built on top of", "built upon",
    "relies on", "rely on",
]

RISK_MARKERS = [
    " risk", "risky", "concern", "threat", "danger",
    "vulnerability", "may fail", "might fail",
    "could break", "uncertain",
]

INPUT_MARKERS = [
    " input", "accepts", "accept ", "receives", "receive ",
    "takes ", "take ", "parses", "parse ",
    "reads ", "read ", "loads ", "load ",
]

OUTPUT_MARKERS = [
    " output", "returns", "return ", "produces", "produce ",
    "emits", "emit ", "generates", "generate ",
    "responds", "respond with", "serves ",
    "prints ", "writes ",
]

OBJECTIVE_VERBS = (
    "build", "create", "develop", "implement", "make", "deliver",
    "provide", "design", "construct",
)
OBJECTIVE_NOUNS = (
    "system", "application", "app", "service", "api", "platform",
    "tool", "library", "framework", "website", "site", "product",
    "server", "client", "backend", "frontend", "database",
    "rest api", "web app", "web application", "mobile app",
    "microservice", "cli",
)

_GWT_RE = re.compile(r"\bgiven\b.*?\bwhen\b.*?\bthen\b", re.I | re.S)

USER_ROLES: list[str] = [
    # longer / more specific first — matters for dedup
    "authenticated user", "end user", "superuser", "power user",
    "administrator", "admin",
    "developer", "operator", "customer", "client",
    "guest", "member", "owner", "viewer", "editor",
    "manager", "staff", "employee", "visitor",
    "user",
]

_PRONOUN_STARTERS = {"it", "they", "them", "those", "these"}


# ---- helpers using pattern tables ----
def _nfr_category(text: str) -> str | None:
    for category, patterns in NFR_PATTERNS.items():
        for p in patterns:
            if p.search(text):
                return category
    return None


def _starts_with_objective_verb(low: str) -> bool:
    """Strong signal: the sentence *opens* with a goal verb ('Build a...',
    'Create a...') or an explicit 'want/need to build/create/...'. This is
    checked ahead of the (much noisier) NFR/dependency/risk keyword
    patterns, since a sentence that literally opens with "Build a small
    production-quality REST API..." is the objective even though it also
    happens to contain an NFR-triggering phrase like "production-quality".
    """
    s = low.strip()
    for v in OBJECTIVE_VERBS:
        if s.startswith(v + " "):
            return True
    if re.search(
        r"\b(?:want|need)\s+to\s+(?:build|create|develop|implement|make|deliver|provide|design)\b",
        low,
    ):
        return True
    return False


def _looks_like_objective(low: str) -> bool:
    if _starts_with_objective_verb(low):
        return True
    has_verb = any(f" {v} " in low for v in OBJECTIVE_VERBS)
    has_noun = any(n in low for n in OBJECTIVE_NOUNS)
    return has_verb and has_noun


def _extract_users(text: str) -> set[str]:
    low = " " + text.lower() + " "
    matched: set[str] = set()
    for role in USER_ROLES:
        # allow a simple trailing plural ("users", "admins", ...) — real
        # requirement text routinely says "Users must..."/"Admins can..."
        if re.search(r"\b" + re.escape(role) + r"s?\b", low):
            if any(role != r and role in r for r in matched):
                continue
            matched.add(role)
    # drop plain "user" if a longer *user role matched
    if any(r != "user" and "user" in r for r in matched):
        matched.discard("user")
    return matched


def _classify_sentence(text: str) -> tuple[RequirementKind, list[str]] | None:
    low = " " + text.lower() + " "

    # 1. Given/When/Then
    if _GWT_RE.search(text):
        return RequirementKind.ACCEPTANCE, ["gwt"]

    # 2. Acceptance cues
    if any(m in low for m in ("acceptance", "definition of done")):
        return RequirementKind.ACCEPTANCE, []
    if ("should return" in low or "should produce" in low) and "should" in low:
        return RequirementKind.ACCEPTANCE, []

    # 3. Objective — strong signal only (sentence *opens* with a goal verb).
    # Checked ahead of constraints/NFR/etc. so an opening sentence like
    # "Build a small production-quality REST API..." is captured as the
    # objective rather than misfiled under NFR just because it also
    # contains a phrase like "production-quality".
    if _starts_with_objective_verb(low):
        # A leading goal verb is an objective only when the sentence does not
        # also contain an explicit functional/action requirement.
        if not any(m in low for m in FUNCTIONAL_MARKERS) and not re.search(
            r"\b(build|create|develop|implement|run|execute|test|tests|support|provide|allow|enable)\b",
            low,
        ):
            return RequirementKind.OBJECTIVE, []

    # 4. Constraints (strong) — explicit prohibition markers ("must not",
    # "cannot", "shall not"...) are an unambiguous signal and should win
    # over a looser NFR keyword/pattern match (e.g. "cannot exceed 100
    # requests per minute" would otherwise be swallowed by the
    # scalability pattern for "<number> requests"). The *soft*/quantitative
    # constraint markers ("within ", "at most", "budget", ...) are checked
    # later, after NFR — they legitimately overlap with NFR performance
    # phrasing (e.g. "must respond within 200ms" is an NFR, not a
    # constraint), so they shouldn't preempt it.
    if any(m in low for m in CONSTRAINT_MARKERS_STRONG):
        return RequirementKind.CONSTRAINT, []

    # 5. NFR
    cat = _nfr_category(text)
    if cat:
        return RequirementKind.NON_FUNCTIONAL, [cat]

    # 6. Inputs / outputs
    if any(m in low for m in INPUT_MARKERS):
        return RequirementKind.INPUT, []
    if any(m in low for m in OUTPUT_MARKERS):
        return RequirementKind.OUTPUT, []

    # 7. Dependencies
    if any(m in low for m in DEPENDENCY_MARKERS):
        return RequirementKind.DEPENDENCY, []

    # 8. Risks
    if any(m in low for m in RISK_MARKERS):
        return RequirementKind.RISK, []

    # 9. Constraints (soft/quantitative)
    if any(m in low for m in CONSTRAINT_MARKERS_SOFT):
        return RequirementKind.CONSTRAINT, []

    # 10. Objective (weak signal: verb+noun anywhere in the sentence)
    if _looks_like_objective(low):
        return RequirementKind.OBJECTIVE, []

    # 11. Functional
    # Mixed action requirements may also look like objectives. Recognize
    # explicit functional/test actions so they are not lost as objectives.
    if any(m in low for m in FUNCTIONAL_MARKERS):
        return RequirementKind.FUNCTIONAL, []
    if re.search(r"\b(build|create|develop|implement|run|execute|test|tests|support|provide|allow|enable)\b", low):
        return RequirementKind.FUNCTIONAL, []

    return None


def _kind_for_section(name: str) -> RequirementKind:
    return {
        "functional": RequirementKind.FUNCTIONAL,
        "non_functional": RequirementKind.NON_FUNCTIONAL,
        "constraints": RequirementKind.CONSTRAINT,
        "inputs": RequirementKind.INPUT,
        "outputs": RequirementKind.OUTPUT,
        "acceptance": RequirementKind.ACCEPTANCE,
        "risks": RequirementKind.RISK,
        "objective": RequirementKind.OBJECTIVE,
    }.get(name, RequirementKind.CONSTRAINT)


# ════════════════════════════════════════════════════════════════════════════
# 5. PARSER
# ════════════════════════════════════════════════════════════════════════════
def _dedupe(items: list[ExtractedItem]) -> list[ExtractedItem]:
    seen: set[str] = set()
    out: list[ExtractedItem] = []
    for it in items:
        k = _norm(it.text)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _find_ambiguities(text: str) -> list[Ambiguity]:
    out: list[Ambiguity] = []

    # Vague terms
    sentences = _split_sentences(text)
    low_text = text.lower()
    for term in sorted(VAGUE_TERMS):
        pat = re.compile(r"\b" + re.escape(term) + r"\b")
        if not pat.search(low_text):
            continue
        source_sentence = next(
            (s for s in sentences if pat.search(s.lower())), term
        )
        out.append(Ambiguity(
            text=source_sentence,
            kind=AmbiguityKind.VAGUE_TERM,
            term=term,
            reason=f"vague term '{term}' used without measurable threshold",
        ))

    # Undefined acronyms
    seen_acr: set[str] = set()
    for m in re.finditer(r"\b[A-Z]{2,6}\b", text):
        acr = m.group(0)
        if acr.lower() in KNOWN_ACRONYMS:
            continue
        if acr in seen_acr:
            continue
        seen_acr.add(acr)
        after = text[m.end():m.end() + 40].lstrip()
        if after.startswith("("):
            continue
        out.append(Ambiguity(
            text=acr,
            kind=AmbiguityKind.UNDEFINED_ACRONYM,
            term=acr,
            reason=f"acronym '{acr}' used without definition",
        ))

    # Placeholders
    for ph in PLACEHOLDERS:
        if re.search(r"\b" + re.escape(ph) + r"\b", text):
            out.append(Ambiguity(
                text=ph,
                kind=AmbiguityKind.PLACEHOLDER,
                term=ph,
                reason=f"placeholder '{ph}' present in requirement text",
            ))

    # Unresolved pronouns at sentence start
    for s in sentences:
        first_word = re.split(r"\W+", s, maxsplit=1)[0].lower()
        if first_word in _PRONOUN_STARTERS:
            out.append(Ambiguity(
                text=s,
                kind=AmbiguityKind.UNRESOLVED_PRONOUN,
                term=first_word,
                reason=f"sentence starts with pronoun '{first_word}' — referent unclear",
            ))
    return out


def _find_missing(
    spec: RequirementSpec, *, sections: dict[str, list[str]]
) -> list[MissingInfo]:
    out: list[MissingInfo] = []
    if spec.objective is None:
        out.append(MissingInfo(MissingKind.OBJECTIVE,
                               "no clear objective/goal sentence found"))
    if not spec.users:
        out.append(MissingInfo(MissingKind.USERS,
                               "no user roles/personas mentioned"))
    if not spec.scope_in and not spec.scope_out:
        out.append(MissingInfo(MissingKind.SCOPE,
                               "no scope (in/out) identified"))
    if not spec.functional and not spec.non_functional:
        out.append(MissingInfo(MissingKind.FUNCTIONAL,
                               "no functional requirements identified"))
    if not spec.non_functional:
        out.append(MissingInfo(MissingKind.NON_FUNCTIONAL,
                               "no non-functional requirements (performance/security/etc.)"))
    if not spec.constraints:
        out.append(MissingInfo(MissingKind.CONSTRAINTS,
                               "no explicit constraints identified"))
    if not spec.inputs:
        out.append(MissingInfo(MissingKind.INPUTS, "no inputs identified"))
    if not spec.outputs:
        out.append(MissingInfo(MissingKind.OUTPUTS, "no outputs identified"))
    if not spec.acceptance_criteria:
        out.append(MissingInfo(MissingKind.ACCEPTANCE,
                               "no acceptance criteria identified"))
    if len(spec.raw_text.strip()) < 30:
        out.append(MissingInfo(MissingKind.DETAIL,
                               "requirement text is very short (<30 chars)"))
    return out


def _compute_confidence(spec: RequirementSpec) -> Confidence:
    if not spec.raw_text.strip():
        return Confidence.UNKNOWN
    score = 0
    if spec.objective: score += 2
    if spec.functional: score += 1
    if spec.non_functional: score += 1
    if spec.constraints: score += 1
    if spec.inputs or spec.outputs: score += 1
    if spec.acceptance_criteria: score += 1
    if spec.users: score += 1
    if spec.scope_in or spec.scope_out: score += 1
    score -= len(spec.ambiguities)
    score -= len(spec.missing)
    if score >= 5: return Confidence.HIGH
    if score >= 2: return Confidence.MEDIUM
    if score >= 0: return Confidence.LOW
    return Confidence.UNKNOWN


class RequirementParser:
    """Deterministic NL → structured spec parser. No LLM. Fully testable."""

    def parse(
        self,
        text: str,
        *,
        provenance: Provenance | None = None,
    ) -> RequirementSpec:
        text = text or ""
        prov = provenance or Provenance(
            source="user",
            source_type=ProvenanceType.USER,
            confidence=Confidence.HIGH,
        )
        spec = RequirementSpec(raw_text=text, provenance=prov)

        # ---- Section-based extraction ----
        sections = _extract_sections(text)
        for name, items in sections.items():
            if name == "assumptions":
                continue
            kind = _kind_for_section(name)
            for content in items:
                if name == "non_functional":
                    # Prefer the specific NFR category (performance,
                    # security, ...) over the generic section-name tag —
                    # downstream engines (e.g. the Planner) key off tags
                    # like "security"/"performance" to add matching tasks,
                    # and this item wins over its sentence-loop duplicate
                    # in _dedupe() (first occurrence kept), so tagging it
                    # generically here would silently lose that signal.
                    tags = [_nfr_category(content) or name]
                else:
                    tags = []
                item = ExtractedItem(
                    text=content, kind=kind, source="section",
                    tags=tags,
                    provenance=prov,
                )
                self._route(spec, item, section_name=name)
                spec.users = sorted(set(spec.users) | _extract_users(content))

        # ---- Sentence-based extraction ----
        for sent in _split_sentences(text):
            classified = _classify_sentence(sent)
            if classified is None:
                continue
            kind, tags = classified
            item = ExtractedItem(
                text=sent, kind=kind, source="sentence", tags=tags,
                provenance=prov,
            )
            self._route(spec, item)
            spec.users = sorted(set(spec.users) | _extract_users(sent))

        # ---- Objective fallback: section header ----
        if spec.objective is None and sections.get("objective"):
            spec.objective = ExtractedItem(
                text=sections["objective"][0],
                kind=RequirementKind.OBJECTIVE,
                source="section", provenance=prov,
            )

        # ---- Dedupe ----
        spec.scope_in = _dedupe(spec.scope_in)
        spec.scope_out = _dedupe(spec.scope_out)
        spec.functional = _dedupe(spec.functional)
        spec.non_functional = _dedupe(spec.non_functional)
        spec.constraints = _dedupe(spec.constraints)
        spec.inputs = _dedupe(spec.inputs)
        spec.outputs = _dedupe(spec.outputs)
        spec.dependencies = _dedupe(spec.dependencies)
        spec.acceptance_criteria = _dedupe(spec.acceptance_criteria)
        spec.risks = _dedupe(spec.risks)
        spec.users = sorted(set(spec.users))

        # ---- Explicit assumptions ----
        spec.assumptions = [
            Assumption(text=t, origin="explicit", provenance=prov)
            for t in sections.get("assumptions", [])
        ]
        # Inferred assumptions (never invented requirements — only meta notes)
        if not spec.acceptance_criteria:
            spec.assumptions.append(Assumption(
                text="Acceptance criteria were not specified.",
                origin="inferred", provenance=prov,
            ))
        if not spec.non_functional:
            spec.assumptions.append(Assumption(
                text="No non-functional requirements were stated.",
                origin="inferred", provenance=prov,
            ))

        # ---- Ambiguities ----
        spec.ambiguities = _find_ambiguities(text)

        # ---- Missing-info report ----
        spec.missing = _find_missing(spec, sections=sections)

        # ---- Overall confidence ----
        spec.confidence = _compute_confidence(spec)
        return spec

    # ---- routing ----
    def _route(
        self,
        spec: RequirementSpec,
        item: ExtractedItem,
        *,
        section_name: str | None = None,
    ) -> None:
        if section_name == "scope_in":
            spec.scope_in.append(item); return
        if section_name == "scope_out":
            spec.scope_out.append(item); return
        if section_name == "objective":
            if spec.objective is None:
                spec.objective = item
            return
        kind = item.kind
        if kind is RequirementKind.OBJECTIVE:
            if spec.objective is None:
                spec.objective = item
        elif kind is RequirementKind.FUNCTIONAL:
            spec.functional.append(item)
        elif kind is RequirementKind.NON_FUNCTIONAL:
            spec.non_functional.append(item)
        elif kind is RequirementKind.CONSTRAINT:
            spec.constraints.append(item)
        elif kind is RequirementKind.INPUT:
            spec.inputs.append(item)
        elif kind is RequirementKind.OUTPUT:
            spec.outputs.append(item)
        elif kind is RequirementKind.DEPENDENCY:
            spec.dependencies.append(item)
        elif kind is RequirementKind.ACCEPTANCE:
            spec.acceptance_criteria.append(item)
        elif kind is RequirementKind.RISK:
            spec.risks.append(item)


# ════════════════════════════════════════════════════════════════════════════
# 6. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class RequirementRepository:
    """Save/load RequirementSpec via C04 memory + optional C02 ontology."""

    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, spec: RequirementSpec, *, project_id: str) -> str:
        """Persist spec. Returns ontology entity id if ontology provided,
        else spec.id."""
        key = f"requirement_spec:{spec.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, spec.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["requirement", "spec"],
            provenance=spec.provenance,
        )
        if self.ontology is None:
            return spec.id
        name = spec.objective.text if spec.objective else "requirement"
        entity = self.ontology.add(
            EntityKind.REQUIREMENT, (name or "requirement")[:120],
            attributes={
                "spec_id": spec.id,
                "project_id": project_id,
                "confidence": spec.confidence.value,
            },
            tags=["spec"],
            provenance=spec.provenance,
        )
        return entity.id

    def load(self, spec_id: str, *, project_id: str) -> dict[str, Any] | None:
        entry = self.memory.get_current(
            MemoryKind.PROJECT, f"requirement_spec:{spec_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        if entry is None:
            return None
        return dict(entry.content)


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
            failures.append(name)
            print(f"  ✗ {name}")
            traceback.print_exc()

    print("Running C05 self-tests…")
    parser = RequirementParser()

    # ---- empty / trivial ----
    def t_empty() -> None:
        spec = parser.parse("")
        assert spec.objective is None
        assert spec.confidence is Confidence.UNKNOWN
        missing_kinds = {m.kind for m in spec.missing}
        assert MissingKind.OBJECTIVE in missing_kinds
        assert MissingKind.DETAIL in missing_kinds

    def t_only_objective() -> None:
        spec = parser.parse("Build a REST API for managing tasks.")
        assert spec.objective is not None
        assert "REST API" in spec.objective.text or "rest api" in spec.objective.text.lower()

    check("empty input → UNKNOWN, missing objective", t_empty)
    check("objective-only sentence", t_only_objective)

    # ---- section-based extraction ----
    def t_scope_sections() -> None:
        text = (
            "In scope:\n"
            "- CRUD operations on tasks\n"
            "- REST API\n\n"
            "Out of scope:\n"
            "- Mobile app\n"
        )
        spec = parser.parse(text)
        assert len(spec.scope_in) == 2
        assert len(spec.scope_out) == 1
        assert any("CRUD" in x.text for x in spec.scope_in)

    def t_assumptions_section() -> None:
        text = "Assumptions:\n- Python 3.11 available\n- User has DB access\n"
        spec = parser.parse(text)
        assert len(spec.assumptions) >= 2
        assert any("Python" in a.text for a in spec.assumptions)

    def t_fr_section() -> None:
        text = "Functional requirements:\n- Users can create tasks\n- Users can list tasks\n"
        spec = parser.parse(text)
        assert len(spec.functional) == 2

    def t_nfr_section() -> None:
        text = "Non-functional:\n- API must respond within 200ms\n- Data must be encrypted\n"
        spec = parser.parse(text)
        assert len(spec.non_functional) >= 2

    def t_acceptance_section() -> None:
        text = "Acceptance:\n- Given a valid request, when POST /tasks is called, then 201 is returned\n"
        spec = parser.parse(text)
        assert len(spec.acceptance_criteria) >= 1

    check("scope_in / scope_out sections", t_scope_sections)
    check("assumptions section", t_assumptions_section)
    check("functional requirements section", t_fr_section)
    check("non-functional section", t_nfr_section)
    check("acceptance section", t_acceptance_section)

    # ---- sentence-based NFR detection ----
    def t_nfr_performance() -> None:
        spec = parser.parse("The API must respond within 200ms.")
        tags = {t for it in spec.non_functional for t in it.tags}
        assert "performance" in tags, tags

    def t_nfr_security() -> None:
        spec = parser.parse("All traffic must use TLS 1.3.")
        tags = {t for it in spec.non_functional for t in it.tags}
        assert "security" in tags, tags

    def t_nfr_reliability() -> None:
        spec = parser.parse("The service must achieve 99.9% uptime.")
        tags = {t for it in spec.non_functional for t in it.tags}
        assert "reliability" in tags, tags

    def t_nfr_compliance() -> None:
        spec = parser.parse("System must comply with GDPR.")
        tags = {t for it in spec.non_functional for t in it.tags}
        assert "compliance" in tags, tags

    check("NFR: performance tag", t_nfr_performance)
    check("NFR: security tag", t_nfr_security)
    check("NFR: reliability tag", t_nfr_reliability)
    check("NFR: compliance tag", t_nfr_compliance)

    # ---- constraint detection ----
    def t_constraint_must_not() -> None:
        spec = parser.parse("The system must not allow anonymous writes.")
        assert any("must not" in c.text.lower() for c in spec.constraints)

    def t_constraint_cannot() -> None:
        spec = parser.parse("Users cannot exceed 100 requests per minute.")
        # "cannot" → constraint
        assert any("cannot" in c.text.lower() for c in spec.constraints)

    check("constraint: 'must not'", t_constraint_must_not)
    check("constraint: 'cannot'", t_constraint_cannot)

    # ---- inputs / outputs ----
    def t_inputs_outputs() -> None:
        text = (
            "The API accepts JSON payloads.\n"
            "The endpoint returns a JSON object with the created task.\n"
        )
        spec = parser.parse(text)
        assert len(spec.inputs) >= 1
        assert len(spec.outputs) >= 1

    check("inputs + outputs", t_inputs_outputs)

    # ---- dependencies ----
    def t_dependencies() -> None:
        spec = parser.parse("The service depends on PostgreSQL 15.")
        assert len(spec.dependencies) >= 1

    check("dependency detection", t_dependencies)

    # ---- risks ----
    def t_risks() -> None:
        spec = parser.parse("Risk: third-party API may fail during peak hours.")
        assert len(spec.risks) >= 1

    check("risk detection", t_risks)

    # ---- users ----
    def t_users() -> None:
        text = (
            "The user can create tasks.\n"
            "An admin can delete tasks.\n"
            "A guest can view public tasks.\n"
        )
        spec = parser.parse(text)
        assert "user" in spec.users
        assert "admin" in spec.users
        assert "guest" in spec.users

    def t_users_dedup_longer_role() -> None:
        text = "The authenticated user can log in."
        spec = parser.parse(text)
        assert "authenticated user" in spec.users
        assert "user" not in spec.users  # suppressed by longer match

    check("users extraction", t_users)
    check("users: longer role suppresses plain 'user'", t_users_dedup_longer_role)

    # ---- acceptance GWT ----
    def t_gwt() -> None:
        text = "Given a valid token, when GET /tasks is called, then a list is returned."
        spec = parser.parse(text)
        assert len(spec.acceptance_criteria) >= 1
        tags = {t for it in spec.acceptance_criteria for t in it.tags}
        assert "gwt" in tags

    check("Given/When/Then acceptance", t_gwt)

    # ---- ambiguities ----
    def t_vague_term() -> None:
        spec = parser.parse("The system must be fast and reliable.")
        vague = [a for a in spec.ambiguities if a.kind is AmbiguityKind.VAGUE_TERM]
        terms = {a.term for a in vague}
        assert "fast" in terms
        assert "reliable" not in terms  # "reliable" is not in VAGUE_TERMS

    def t_undefined_acronym() -> None:
        spec = parser.parse("Use the ABCD protocol for communication.")
        acr = [a for a in spec.ambiguities if a.kind is AmbiguityKind.UNDEFINED_ACRONYM]
        assert any(a.term == "ABCD" for a in acr)
        # API is known — should NOT be flagged
        spec2 = parser.parse("Expose a REST API.")
        assert not any(a.term == "API" for a in spec2.ambiguities)

    def t_placeholder() -> None:
        spec = parser.parse("The retention period is TBD.")
        assert any(a.kind is AmbiguityKind.PLACEHOLDER for a in spec.ambiguities)

    def t_pronoun() -> None:
        spec = parser.parse("It should process the request.")
        pron = [a for a in spec.ambiguities if a.kind is AmbiguityKind.UNRESOLVED_PRONOUN]
        assert len(pron) >= 1

    check("ambiguity: vague term", t_vague_term)
    check("ambiguity: undefined acronym (known excluded)", t_undefined_acronym)
    check("ambiguity: placeholder (TBD)", t_placeholder)
    check("ambiguity: unresolved pronoun", t_pronoun)

    # ---- missing info ----
    def t_missing_info() -> None:
        spec = parser.parse("Build a small service.")
        kinds = {m.kind for m in spec.missing}
        assert MissingKind.USERS in kinds
        assert MissingKind.ACCEPTANCE in kinds
        assert MissingKind.INPUTS in kinds
        assert MissingKind.OUTPUTS in kinds
        assert MissingKind.CONSTRAINTS in kinds

    check("missing-info report", t_missing_info)

    # ---- dedupe across section + sentence ----
    def t_dedupe() -> None:
        text = (
            "Functional requirements:\n"
            "- Users must be able to create tasks.\n"
        )
        spec = parser.parse(text)
        # The bullet goes to functional once; the sentence-scan should not
        # produce a duplicate because of normalized dedupe.
        texts = [_norm(x.text) for x in spec.functional]
        assert len(texts) == len(set(texts))

    check("no duplicates across section + sentence", t_dedupe)

    # ---- confidence scoring ----
    def t_confidence_high() -> None:
        text = (
            "Build a REST API for managing tasks.\n"
            "Users can create tasks.\n"
            "The API must respond within 200ms.\n"
            "The API accepts JSON input.\n"
            "The API returns JSON output.\n"
            "Given a valid request, when POST /tasks is called, then 201 is returned.\n"
            "In scope:\n- Task CRUD\n"
        )
        spec = parser.parse(text)
        assert spec.confidence in (Confidence.HIGH, Confidence.MEDIUM)

    def t_confidence_low() -> None:
        spec = parser.parse("Make something.")
        assert spec.confidence in (Confidence.LOW, Confidence.UNKNOWN)

    check("confidence HIGH/MEDIUM for rich input", t_confidence_high)
    check("confidence LOW/UNKNOWN for sparse input", t_confidence_low)

    # ---- to_dict roundtrip ----
    def t_to_dict() -> None:
        spec = parser.parse(
            "Build a REST API.\n"
            "Users can create tasks.\n"
            "The API must respond within 200ms.\n"
        )
        d = spec.to_dict()
        assert d["id"] == spec.id
        assert d["objective"] is not None
        assert isinstance(d["functional"], list)
        assert isinstance(d["non_functional"], list)
        assert isinstance(d["missing"], list)
        assert "confidence" in d
        assert "provenance" in d

    check("to_dict produces JSON-serializable dict", t_to_dict)

    # ---- repository save/load ----
    def t_repository() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                repo = RequirementRepository(memory=mem, ontology=ont)
                spec = parser.parse(
                    "Build a REST API for tasks.\n"
                    "Users can create tasks.\n"
                )
                ent_id = repo.save(spec, project_id="proj-1")
                assert ent_id
                loaded = repo.load(spec.id, project_id="proj-1")
                assert loaded is not None
                assert loaded["id"] == spec.id
                assert loaded["objective"] is not None
                # Ontology entity exists
                ent = ont.get(ent_id)
                assert ent is not None
                assert ent.kind is EntityKind.REQUIREMENT
            finally:
                s.shutdown()

    check("repository: save/load via memory + ontology", t_repository)

    # ---- e2e (prompt's canonical example) ----
    def t_e2e_canonical() -> None:
        text = (
            "Build a small production-quality REST API for managing tasks.\n\n"
            "Functional requirements:\n"
            "- Users must be able to create, read, update, and delete tasks.\n"
            "- Admins must be able to list all tasks.\n\n"
            "Non-functional:\n"
            "- The API must respond within 200ms for read operations.\n"
            "- All traffic must use HTTPS.\n\n"
            "Constraints:\n"
            "- Must not require external authentication services.\n\n"
            "Inputs:\n"
            "- JSON request bodies.\n\n"
            "Outputs:\n"
            "- JSON responses.\n\n"
            "Acceptance:\n"
            "- Given a valid request, when POST /tasks is called, then a 201 response is returned.\n\n"
            "In scope:\n- Task CRUD operations\n- REST API\n\n"
            "Out of scope:\n- Mobile client\n\n"
            "Assumptions:\n- Python 3.11 or newer is available\n\n"
            "Risks:\n- Concurrency on SQLite may cause lock contention\n"
        )
        spec = parser.parse(text)
        # Objective captured
        assert spec.objective is not None
        assert "rest api" in spec.objective.text.lower()
        # Users
        assert "user" in spec.users or "admin" in spec.users
        # Functional & NFR present
        assert len(spec.functional) >= 1
        assert len(spec.non_functional) >= 1
        # Constraints captured
        assert len(spec.constraints) >= 1
        # Inputs & outputs
        assert len(spec.inputs) >= 1
        assert len(spec.outputs) >= 1
        # Acceptance (GWT)
        assert len(spec.acceptance_criteria) >= 1
        # Scope
        assert len(spec.scope_in) >= 1
        assert len(spec.scope_out) >= 1
        # Assumptions
        assert any("Python" in a.text for a in spec.assumptions)
        # Risks
        assert len(spec.risks) >= 1
        # Overall confidence should be at least MEDIUM
        assert spec.confidence in (Confidence.HIGH, Confidence.MEDIUM), spec.confidence
        # Integrity: nothing missing about the core stuff
        missing_kinds = {m.kind for m in spec.missing}
        assert MissingKind.OBJECTIVE not in missing_kinds
        assert MissingKind.USERS not in missing_kinds
        assert MissingKind.ACCEPTANCE not in missing_kinds

    check("e2e: canonical 'Build a REST API for tasks' input", t_e2e_canonical)

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
    print("SE Brain C05 — Requirement Understanding Engine")
    print("=" * 78)

    text = (
        "Build a small production-quality REST API for managing tasks.\n\n"
        "Functional requirements:\n"
        "- Users must be able to create, read, update, and delete tasks.\n"
        "- Admins must be able to list all tasks.\n\n"
        "Non-functional:\n"
        "- The API must respond within 200ms for read operations.\n"
        "- All traffic must use HTTPS.\n\n"
        "Constraints:\n"
        "- Must not require external authentication services.\n\n"
        "Inputs:\n"
        "- JSON request bodies.\n\n"
        "Outputs:\n"
        "- JSON responses.\n\n"
        "Acceptance:\n"
        "- Given a valid request, when POST /tasks is called, then a 201 response is returned.\n\n"
        "In scope:\n- Task CRUD operations\n- REST API\n\n"
        "Out of scope:\n- Mobile client\n\n"
        "Assumptions:\n- Python 3.11 or newer is available\n\n"
        "Risks:\n- Concurrency on SQLite may cause lock contention\n"
    )

    parser = RequirementParser()
    spec = parser.parse(text)

    print("\n[1] Summary:")
    print(spec.summary())

    print("\n[2] Objective:")
    print(f"    {spec.objective.text if spec.objective else '(none)'}")

    print("\n[3] Users:")
    for u in spec.users:
        print(f"    - {u}")

    print("\n[4] Functional requirements:")
    for it in spec.functional:
        print(f"    - {it.text}")

    print("\n[5] Non-functional requirements:")
    for it in spec.non_functional:
        print(f"    - [{','.join(it.tags)}] {it.text}")

    print("\n[6] Constraints:")
    for it in spec.constraints:
        print(f"    - {it.text}")

    print("\n[7] Inputs / Outputs:")
    for it in spec.inputs:
        print(f"    IN : {it.text}")
    for it in spec.outputs:
        print(f"    OUT: {it.text}")

    print("\n[8] Acceptance criteria:")
    for it in spec.acceptance_criteria:
        print(f"    - {it.text}")

    print("\n[9] Scope:")
    for it in spec.scope_in:
        print(f"    IN : {it.text}")
    for it in spec.scope_out:
        print(f"    OUT: {it.text}")

    print("\n[10] Assumptions:")
    for a in spec.assumptions:
        print(f"    [{a.origin}] {a.text}")

    print("\n[11] Risks:")
    for it in spec.risks:
        print(f"    - {it.text}")

    print("\n[12] Ambiguities:")
    for a in spec.ambiguities:
        print(f"    [{a.kind.value}] {a.term}: {a.reason}")

    print("\n[13] Missing information:")
    for m in spec.missing:
        print(f"    [{m.kind.value}] {m.why}")

    print("\n[14] Overall confidence:", spec.confidence.value)

    # Persistence demo
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = RequirementRepository(memory=mem, ontology=ont)
                ent_id = repo.save(spec, project_id="demo-proj")
                print(f"\n[15] Persisted — ontology entity: {ent_id[:12]}…")
                loaded = repo.load(spec.id, project_id="demo-proj")
                print(f"    reloaded keys: {sorted(loaded.keys())[:5]}…")
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
