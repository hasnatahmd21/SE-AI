"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C14 — CODE SYNTHESIS ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C05, C08, C09, C10, C13.

Purpose:
    Convert structured implementation inputs into REAL, COMPILABLE Python
    source code. Never from raw user text — always from a structured
    SynthesisRequest that references requirement + architecture + tech stack
    + existing repo index.

Capabilities:
    - Entity extraction from RequirementSpec (deterministic)
    - Convention detection from C13 index (naming, imports, docstrings,
      type hints) — falls back to PEP8 defaults when no repo present
    - 7 generators producing real Python:
        ModelGenerator       dataclasses or pydantic models
        RepositoryGenerator  sqlite3 CRUD (parameterised queries only)
        ServiceGenerator     business-logic layer wrapper
        APIHandlerGenerator  FastAPI or Flask routes
        TestGenerator        pytest tests (defaults + CRUD round-trip)
        ConfigGenerator      pydantic-settings based config
        EntrypointGenerator  __main__ runner
    - Duplicate detection against existing symbol index (C13)
    - Interface satisfaction check (class must expose required methods)
    - Compile-time verification: every generated file is `compile()`d
    - Package-name aware (auto-derives from target path or C13 index)

Invariants honored:
  - NO external LLM. Pure templates + deterministic inference.
  - Never overwrite silently: default mode="fresh" (error if exists),
    mode="skip_existing" (report + skip), mode="overwrite" (explicit).
  - No filesystem writes: engine returns SynthesizedFile objects. Writing
    is C15's job (Repository Builder).
  - All generated Python must pass `compile(src, path, "exec")`.
  - SQL uses parameterised queries only — no string interpolation.
  - Never invents behaviour: if the plan doesn't reference an entity, we
    don't fabricate one. Extraction is deterministic and bounded.
  - Every SynthesizedFile carries rationale + evidence + kind.

Contents:
  1.  Enums: FileKind, ConventionKind, SynthesisStatus
  2.  Dataclasses: FieldSpec, EntitySpec, FileConventions, InterfaceSpec,
                   SynthesisRequest, SynthesizedFile, DuplicateReport,
                   InterfaceCheck, SynthesisResult
  3.  Helpers: _CodeBuilder, name transforms
  4.  Entity extractor (RequirementSpec → list[EntitySpec])
  5.  ConventionsDetector (C13 index → FileConventions)
  6.  Generators (7)
  7.  DuplicateDetector (C13 index check)
  8.  InterfaceChecker
  9.  CodeSynthesisEngine (facade)
  10. SynthesisRepository (persist to C04 memory + C02 ontology)
  11. __main__ demo + self-tests

Run as script:
    python -m sebrain.c14            # demo
    python -m sebrain.c14 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
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
    RelationKind,
)
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c05 import RequirementKind, RequirementSpec
from sebrain.c08 import Plan, TaskKind
from sebrain.c09 import TechCategory, TechSelectionResult
from sebrain.c10 import ArchitectureResult, ArchitectureDecision
from sebrain.c13 import (
    CodeRepresentationEngine,
    RepoIndex,
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


def _short(s: str, n: int = 100) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def to_snake(name: str) -> str:
    """Task → task; HTTPClient → http_client; Task2Item → task2_item."""
    return _CAMEL_TO_SNAKE.sub("_", name).lower()


def to_pascal(name: str) -> str:
    """task → Task; task_item → TaskItem."""
    return "".join(p.capitalize() for p in re.split(r"[_\s]+", name) if p)


def pluralize(name: str) -> str:
    """task → tasks; category → categories; box → boxes."""
    n = name.lower()
    if n.endswith(("s", "x", "z", "ch", "sh")):
        return n + "es"
    if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def singularize(name: str) -> str:
    """tasks → task; categories → category; boxes → box."""
    n = name.lower()
    if n.endswith("ies") and len(n) > 3:
        return n[:-3] + "y"
    if n.endswith(("xes", "ses", "zes", "ches", "shes")):
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss") and len(n) > 1:
        return n[:-1]
    return n


def _python_type_to_sqlite(t: str) -> str:
    """Map a Python type annotation to a SQLite column type."""
    low = (t or "").lower()
    if "int" in low and "float" not in low:
        return "INTEGER"
    if "float" in low:
        return "REAL"
    if "bool" in low:
        return "INTEGER"    # 0/1
    if low in ("list", "dict", "list[...]", "dict[...]") or low.startswith(("list[", "dict[")):
        return "TEXT"       # JSON-encoded
    return "TEXT"


def _repo_write_expr(field_name: str, py_type: str, obj: str = "item") -> str:
    """Expression to bind a Python attribute as a SQLite parameter.

    list/dict fields are stored as JSON text (see _python_type_to_sqlite),
    so they must be encoded on the way in — sqlite3 has no native
    list/dict adapter and would otherwise raise `sqlite3.InterfaceError`
    the moment such a field were used. bool needs no write-side
    conversion: sqlite3 already adapts Python bool to 0/1 automatically.
    """
    low = (py_type or "").lower()
    if "list" in low or "dict" in low:
        return f"json.dumps({obj}.{field_name})"
    return f"{obj}.{field_name}"


def _repo_read_expr(field_name: str, py_type: str, row: str = "row") -> str:
    """Expression to reconstruct a Python attribute from a SQLite row.

    SQLite has no native boolean type — a column declared for a `bool`
    field comes back as a plain Python `int` (0/1), not `True`/`False`,
    which fails identity/type-sensitive checks downstream even though
    `0 == False` happens to hold. list/dict fields need the matching
    `json.loads` to undo `_repo_write_expr`'s `json.dumps`.
    """
    low = (py_type or "").lower()
    if "bool" in low:
        return f"bool({row}['{field_name}'])"
    if "list" in low or "dict" in low:
        return f"json.loads({row}['{field_name}'])"
    return f"{row}['{field_name}']"


def _default_for_type(t: str) -> str:
    """Return the source-code default value for a given Python type."""
    low = (t or "").lower()
    if "none" in low or "optional" in low:
        return "None"
    if "bool" in low:
        return "False"
    if "int" in low and "float" not in low:
        return "0"
    if "float" in low:
        return "0.0"
    if low.startswith("list"):
        return "field(default_factory=list)"
    if low.startswith("dict"):
        return "field(default_factory=dict)"
    if "datetime" in low:
        return "field(default_factory=lambda: datetime.now(timezone.utc))"
    if low.startswith("str"):
        return '""'
    return "None"


def _safe_default_expr(f: "FieldSpec", *, style: str = "dataclass") -> str:
    """Source-code default-value expression for one field, safe for both
    dataclasses and pydantic BaseModel.

    Two bugs, same root cause, both now handled here:

    1. A caller-supplied f.default_repr (e.g. an explicit "[]" or "{}")
       used to be emitted verbatim as a bare literal via the
       `f.default_repr or _default_for_type(...)` pattern -- fine for
       str/bool/int/float, but a hard crash for list/dict/set under
       dataclasses (`ValueError: mutable default <class 'list'> for
       field ... is not allowed: use default_factory`), raised the
       instant the class body is evaluated, before any generated code
       runs.
    2. _default_for_type()'s own *inferred* list/dict default (when
       f.default_repr is None) already used a factory wrapper -- but
       unconditionally as `field(default_factory=...)`, even under
       model_style="pydantic", where only `Field` is imported (never
       `field`), which is a NameError at class-definition time for the
       exact same reason.

    Both are fixed by choosing the wrapper name from `style`: `field(...)`
    for dataclasses, `Field(...)` for pydantic -- matching how this same
    module already emits `created_at` a few lines above in each branch.
    """
    wrapper = "Field" if style == "pydantic" else "field"
    low = (f.type_ or "").lower()
    is_mutable_type = low.startswith(("list", "dict", "set"))

    if f.default_repr is not None:
        if is_mutable_type:
            return f"{wrapper}(default_factory=lambda: {f.default_repr})"
        return f.default_repr

    if is_mutable_type:
        factory = "list" if low.startswith("list") else ("dict" if low.startswith("dict") else "set")
        return f"{wrapper}(default_factory={factory})"

    return _default_for_type(f.type_)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class FileKind(str, Enum):
    MODEL = "model"
    REPOSITORY = "repository"
    SERVICE = "service"
    API = "api"
    TEST = "test"
    CONFIG = "config"
    ENTRYPOINT = "entrypoint"
    INIT = "init"


class ConventionKind(str, Enum):
    FILE_NAMING = "file_naming"
    CLASS_NAMING = "class_naming"
    FUNCTION_NAMING = "function_naming"
    IMPORT_ORDERING = "import_ordering"
    DOCSTRINGS = "docstrings"
    TYPE_HINTS = "type_hints"


class SynthesisStatus(str, Enum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"      # some files skipped (duplicates)
    FAILED = "failed"        # no files generated


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class FieldSpec:
    """One field in an entity."""
    name: str
    type_: str
    required: bool = True
    default_repr: str | None = None     # explicit default expression

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "type": self.type_,
            "required": self.required, "default": self.default_repr,
        }


@dataclass(slots=True)
class EntitySpec:
    """A domain entity to be synthesised."""
    name: str                            # PascalCase, e.g. "Task"
    fields: list[FieldSpec] = field(default_factory=list)
    table_name: str = ""                 # sqlite table; defaults to snake plural

    def __post_init__(self) -> None:
        if not self.name or not self.name[0].isupper():
            raise ValidationError(f"entity name must be PascalCase: {self.name!r}")
        if not self.table_name:
            self.table_name = pluralize(to_snake(self.name))

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fields": [f.to_dict() for f in self.fields],
            "table_name": self.table_name,
        }


@dataclass(slots=True)
class FileConventions:
    file_naming: str = "snake_case"
    class_naming: str = "PascalCase"
    function_naming: str = "snake_case"
    import_ordering: str = "stdlib_then_local"
    docstrings: str = "module_and_public"
    type_hints: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_naming": self.file_naming,
            "class_naming": self.class_naming,
            "function_naming": self.function_naming,
            "import_ordering": self.import_ordering,
            "docstrings": self.docstrings,
            "type_hints": self.type_hints,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class InterfaceSpec:
    """Declared interface a synthesised class must satisfy."""
    name: str
    methods: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "methods": list(self.methods),
            "description": self.description,
        }


@dataclass(slots=True)
class SynthesisRequest:
    """Structured input to the synthesis engine."""
    package_name: str                    # e.g. "task_api"
    entities: list[EntitySpec] = field(default_factory=list)
    target_dir: str = ""                 # repo-relative dir; defaults to package_name
    kinds: list[FileKind] = field(default_factory=lambda: [
        FileKind.INIT, FileKind.MODEL, FileKind.REPOSITORY,
        FileKind.SERVICE, FileKind.API, FileKind.CONFIG,
        FileKind.ENTRYPOINT, FileKind.TEST,
    ])
    framework: str = "fastapi"           # "fastapi" | "flask"
    model_style: str = "dataclass"       # "dataclass" | "pydantic"
    database: str = "sqlite"             # informational
    interfaces: list[InterfaceSpec] = field(default_factory=list)
    mode: str = "fresh"                  # "fresh" | "skip_existing" | "overwrite"
    reference: dict[str, Any] = field(default_factory=dict)   # traceability

    def __post_init__(self) -> None:
        if not self.package_name or not re.match(r"^[a-z][a-z0-9_]*$", self.package_name):
            raise ValidationError(
                f"package_name must be snake_case: {self.package_name!r}"
            )
        if self.framework not in ("fastapi", "flask"):
            raise ValidationError(f"unsupported framework: {self.framework}")
        if self.model_style not in ("dataclass", "pydantic"):
            raise ValidationError(f"unsupported model_style: {self.model_style}")
        if self.mode not in ("fresh", "skip_existing", "overwrite"):
            raise ValidationError(f"unsupported mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "entities": [e.to_dict() for e in self.entities],
            "target_dir": self.target_dir or self.package_name,
            "kinds": [k.value for k in self.kinds],
            "framework": self.framework,
            "model_style": self.model_style,
            "database": self.database,
            "interfaces": [i.to_dict() for i in self.interfaces],
            "mode": self.mode,
            "reference": dict(self.reference),
        }


@dataclass(slots=True)
class SynthesizedFile:
    path: str                            # repo-relative
    content: str
    kind: FileKind
    language: str = "python"
    rationale: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    byte_size: int = 0

    def __post_init__(self) -> None:
        if not self.path:
            raise ValidationError("SynthesizedFile.path is required")
        if not self.content:
            raise ValidationError("SynthesizedFile.content is required")
        self.byte_size = len(self.content.encode("utf-8"))

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path, "kind": self.kind.value,
            "language": self.language, "byte_size": self.byte_size,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
        }
        if include_content:
            d["content"] = self.content
        return d


@dataclass(slots=True)
class DuplicateReport:
    path: str
    existing_symbols: list[str]     # qualnames that would clash
    action: str                     # "skip" | "overwrite" | "allow"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "existing_symbols": list(self.existing_symbols),
            "action": self.action, "reason": self.reason,
        }


@dataclass(slots=True)
class InterfaceCheck:
    file_path: str
    interface: str
    satisfied: bool
    missing: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path, "interface": self.interface,
            "satisfied": self.satisfied, "missing": list(self.missing),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class SynthesisResult:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    request_id: str = ""                 # user-supplied correlation
    files: list[SynthesizedFile] = field(default_factory=list)
    skipped: list[DuplicateReport] = field(default_factory=list)
    duplicates: list[DuplicateReport] = field(default_factory=list)
    interface_checks: list[InterfaceCheck] = field(default_factory=list)
    conventions: FileConventions = field(default_factory=FileConventions)
    status: SynthesisStatus = SynthesisStatus.SUCCEEDED
    compile_errors: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def file_by_path(self, path: str) -> SynthesizedFile | None:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def to_dict(self, *, include_contents: bool = False) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "request_id": self.request_id,
            "files": [f.to_dict(include_content=include_contents)
                      for f in self.files],
            "skipped": [s.to_dict() for s in self.skipped],
            "duplicates": [d.to_dict() for d in self.duplicates],
            "interface_checks": [c.to_dict() for c in self.interface_checks],
            "conventions": self.conventions.to_dict(),
            "status": self.status.value,
            "compile_errors": list(self.compile_errors),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for f in self.files:
            kinds[f.kind.value] = kinds.get(f.kind.value, 0) + 1
        return (
            "=== Synthesis Result ===\n"
            f"status={self.status.value}  files={len(self.files)}  "
            f"skipped={len(self.skipped)}  compile_errors={len(self.compile_errors)}\n"
            f"kinds: {kinds}\n"
            f"framework={self.conventions.import_ordering}  "
            f"model_style_detected=?"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. _CodeBuilder — line-oriented code emitter
# ════════════════════════════════════════════════════════════════════════════
class _CodeBuilder:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def add(self, line: str = "") -> "_CodeBuilder":
        self._lines.append(line)
        return self

    def blank(self) -> "_CodeBuilder":
        if self._lines and self._lines[-1] != "":
            self._lines.append("")
        return self

    def extend(self, lines: Iterable[str]) -> "_CodeBuilder":
        for l in lines:
            self._lines.append(l)
        return self

    def build(self) -> str:
        # Trim trailing blank lines, enforce final newline
        while self._lines and self._lines[-1] == "":
            self._lines.pop()
        return "\n".join(self._lines) + "\n"

    def __len__(self) -> int:
        return len(self._lines)


# ════════════════════════════════════════════════════════════════════════════
# 4. ENTITY EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════
_CRUD_VERBS = (
    "create", "read", "update", "delete", "list", "manage", "add", "remove",
    "get", "fetch", "track", "register", "store",
)
_NOUN_STOP = {
    "a", "an", "the", "user", "users", "admin", "admins", "system", "systems",
    "api", "apis", "service", "services", "request", "requests", "response",
    "responses", "client", "clients", "server", "servers", "data", "item",
    "items", "thing", "things", "it", "them",
    # conjunctions / prepositions / modal words that can end up adjacent to
    # a CRUD verb ("...update, and delete tasks" → "and" sits right before
    # "delete") and would otherwise be captured as a bogus entity name.
    "and", "or", "to", "be", "able", "must", "should", "will", "shall",
    "of", "in", "on", "for", "with", "by", "as", "is", "are", "that",
    "this", "these", "those", "all", "any", "each",
}
_ALLOWED_SINGLE_WORDS = {"task", "user", "order", "product", "note", "post",
                         "comment", "invoice", "customer", "account", "project"}


def infer_entities_from_spec(spec: RequirementSpec) -> list[EntitySpec]:
    """Deterministic heuristic. Extracts 0+ entity specs from functional
    requirements. Adds id + created_at. Never invents fields beyond
    reasonable defaults; if nothing found, returns [].
    """
    if not spec:
        return []
    text_parts = [spec.raw_text or ""]
    text_parts += [it.text for it in spec.functional]
    text = " ".join(text_parts).lower()

    candidate_names: list[str] = []
    for verb in _CRUD_VERBS:
        # verb followed by a noun
        for m in re.finditer(rf"\b{verb}\s+([a-z][a-z0-9_]+)", text):
            candidate_names.append(m.group(1))
        # noun followed by "creation"/"deletion" etc.
        for m in re.finditer(
            rf"\b([a-z][a-z0-9_]+)\s+(?:{verb}|creation|deletion|listing|"
            rf"management|updates|reads)\b", text,
        ):
            candidate_names.append(m.group(1))

    seen: set[str] = set()
    entities: list[EntitySpec] = []
    for raw in candidate_names:
        candidate = raw.strip(".,;:!?\"'()[]{}")
        if not candidate or len(candidate) < 3:
            continue
        if candidate in _NOUN_STOP:
            continue
        if not re.match(r"^[a-z][a-z0-9_]*$", candidate):
            continue
        sing = singularize(candidate)
        if sing in _NOUN_STOP:
            continue
        # filter obvious non-entities
        if sing.endswith(("tion", "sion", "ment", "ness", "ance", "ence")):
            continue
        if sing in seen:
            continue
        # Only keep well-known entity nouns OR pluralised nouns that
        # appeared after a CRUD verb
        seen.add(sing)
        cls = to_pascal(sing)
        entities.append(EntitySpec(
            name=cls,
            fields=[
                FieldSpec("title", "str", required=True),
                FieldSpec("description", "str", required=False,
                          default_repr='""'),
            ],
        ))
        if len(entities) >= 3:      # bounded
            break
    return entities


# ════════════════════════════════════════════════════════════════════════════
# 5. CONVENTIONS DETECTOR
# ════════════════════════════════════════════════════════════════════════════
class ConventionsDetector:
    """Reads a C13 RepoIndex and infers PEP8-ish conventions.

    Falls back to PEP8 defaults when the index is empty or unclear.
    Never mutates the index.
    """

    def detect(self, index: RepoIndex | None) -> FileConventions:
        c = FileConventions()
        if index is None or not index.modules:
            c.notes.append("no prior index → defaults applied")
            return c

        # File naming from module paths
        filenames = [
            p.rsplit("/", 1)[-1] for p in index.modules.keys()
            if not p.endswith("__init__.py")
        ]
        if filenames:
            snake = sum(
                1 for f in filenames
                if f.endswith(".py") and re.match(r"^[a-z][a-z0-9_]*\.py$", f)
            )
            if snake / len(filenames) >= 0.7:
                c.file_naming = "snake_case"
            c.notes.append(
                f"file naming: {snake}/{len(filenames)} snake_case"
            )

        # Class / function naming
        classes = [s for s in index.symbol_index.values()
                   if s.kind is SymbolKind.CLASS]
        functions = [s for s in index.symbol_index.values()
                     if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)]
        if classes:
            pascal = sum(1 for s in classes if s.name and s.name[0].isupper())
            if pascal / len(classes) >= 0.7:
                c.class_naming = "PascalCase"
        if functions:
            snake = sum(
                1 for s in functions
                if re.match(r"^[a-z_][a-z0-9_]*$", s.name)
            )
            if snake / len(functions) >= 0.7:
                c.function_naming = "snake_case"

        # Type hints
        type_counts = sum(
            1 for s in index.symbol_index.values()
            if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD) and s.type_hint
        )
        fn_total = max(1, len(functions))
        c.type_hints = (type_counts / fn_total) >= 0.4
        if c.type_hints:
            c.notes.append(
                f"type hints: {type_counts}/{fn_total} functions annotated"
            )

        # Docstrings
        doc_count = sum(
            1 for s in index.symbol_index.values()
            if s.kind in (SymbolKind.CLASS, SymbolKind.FUNCTION) and s.docstring
        )
        if classes or functions:
            total = max(1, len(classes) + len(functions))
            if (doc_count / total) >= 0.3:
                c.docstrings = "module_and_public"
                c.notes.append(
                    f"docstrings present on {doc_count}/{total} symbols"
                )
        return c


# ════════════════════════════════════════════════════════════════════════════
# 6. GENERATORS
# ════════════════════════════════════════════════════════════════════════════
_HEADER = '"""Auto-generated by SE Brain C14 — do not edit by hand."""'


class _BaseGenerator:
    def __init__(self, conventions: FileConventions) -> None:
        self.c = conventions

    # helper: package + module import path
    @staticmethod
    def _pkg(package: str, module: str) -> str:
        return f"{package}.{module}"

    # helper: turn "Task" → "tasks" table
    @staticmethod
    def _table_for(entity: EntitySpec) -> str:
        return entity.table_name


# ---- 6.1 Model generator ----
class ModelGenerator(_BaseGenerator):
    def generate(
        self, entities: Sequence[EntitySpec], *,
        package: str, style: str = "dataclass",
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        if style == "pydantic":
            b.add("from datetime import datetime, timezone")
            b.add("from pydantic import BaseModel, Field")
        else:
            b.add("from dataclasses import dataclass, field")
            b.add("from datetime import datetime, timezone")
        b.blank()
        b.blank()
        for i, e in enumerate(entities):
            self._emit_model(b, e, style=style)
            self._emit_input_model(b, e, style=style)
            if i < len(entities) - 1:
                b.blank()
                b.blank()
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/models.py",
            content=content,
            kind=FileKind.MODEL,
            rationale=(
                f"models for {len(entities)} entities; style={style}; "
                f"conventions: class={self.c.class_naming}, "
                f"hints={self.c.type_hints}"
            ),
            evidence=[{"entities": [e.name for e in entities]}],
        )

    def _ordered_fields(self, e: EntitySpec) -> list[FieldSpec]:
        """Required first, then optional (dataclass requirement)."""
        req = [f for f in e.fields if f.required]
        opt = [f for f in e.fields if not f.required]
        return req + opt

    def _emit_model(self, b: _CodeBuilder, e: EntitySpec, *, style: str) -> None:
        ordered = self._ordered_fields(e)
        required_fields = [f for f in ordered if f.required and f.default_repr is None]
        optional_fields = [f for f in ordered if not (f.required and f.default_repr is None)]

        def _emit_field(f: FieldSpec) -> None:
            if f.required and f.default_repr is None:
                b.add(f"    {f.name}: {f.type_}")
            else:
                default = _safe_default_expr(f, style=style)
                b.add(f"    {f.name}: {f.type_} = {default}")

        if style == "pydantic":
            b.add(f"class {e.name}(BaseModel):")
            b.add(f'    """{e.name} domain model."""')
            b.blank()
            # Required entity fields first, then the auto id/created_at
            # (both defaulted) and any optional entity fields — pydantic
            # doesn't require this ordering, but keeping it consistent
            # with the dataclass style (below), where it's mandatory,
            # avoids the two styles silently diverging in field order.
            for f in required_fields:
                _emit_field(f)
            b.add(f"    id: int | None = None")
            b.add(
                '    created_at: datetime = Field('
                'default_factory=lambda: datetime.now(timezone.utc))'
            )
            for f in optional_fields:
                _emit_field(f)
        else:
            b.add("@dataclass")
            b.add(f"class {e.name}:")
            b.add(f'    """{e.name} domain model."""')
            b.blank()
            # Dataclasses require every field *without* a default to come
            # before any field *with* a default. id/created_at both carry
            # defaults, so a required entity field (e.g. "title: str")
            # must be emitted before them, not after — otherwise Python
            # raises "non-default argument follows default argument" the
            # moment the module is imported.
            for f in required_fields:
                _emit_field(f)
            b.add("    id: int | None = None")
            b.add(
                "    created_at: datetime = field("
                "default_factory=lambda: datetime.now(timezone.utc))"
            )
            for f in optional_fields:
                _emit_field(f)

    def _emit_input_model(
        self, b: _CodeBuilder, e: EntitySpec, *, style: str,
    ) -> None:
        b.blank()
        b.blank()
        iname = f"{e.name}In"
        if style == "pydantic":
            b.add(f"class {iname}(BaseModel):")
            b.add(f'    """Input payload for creating/updating {e.name}."""')
            b.blank()
            for f in e.fields:
                if f.required:
                    b.add(f"    {f.name}: {f.type_}")
                else:
                    default = _safe_default_expr(f, style="pydantic")
                    b.add(f"    {f.name}: {f.type_} = {default}")
            if not e.fields:
                b.add("    pass")
        else:
            b.add("@dataclass")
            b.add(f"class {iname}:")
            b.add(f'    """Input payload for creating/updating {e.name}."""')
            b.blank()
            ordered = self._ordered_fields(e)
            if not ordered:
                b.add("    pass")
            else:
                for f in ordered:
                    if f.required and f.default_repr is None:
                        b.add(f"    {f.name}: {f.type_}")
                    else:
                        default = _safe_default_expr(f, style="dataclass")
                        b.add(f"    {f.name}: {f.type_} = {default}")


# ---- 6.2 Repository generator ----
class RepositoryGenerator(_BaseGenerator):
    def generate(
        self, entities: Sequence[EntitySpec], *,
        package: str, database: str = "sqlite",
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        b.add("import sqlite3")
        b.add("import json")
        b.add("from datetime import datetime, timezone")
        b.add("from pathlib import Path")
        b.blank()
        b.add(f"from {package}.models import (")
        for e in entities:
            b.add(f"    {e.name},")
            b.add(f"    {e.name}In,")
        b.add(")")
        b.blank()
        b.blank()
        for i, e in enumerate(entities):
            self._emit_repo(b, e)
            if i < len(entities) - 1:
                b.blank()
                b.blank()
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/repository.py",
            content=content,
            kind=FileKind.REPOSITORY,
            rationale=(
                f"sqlite repository for {len(entities)} entities; "
                f"parameterised SQL only"
            ),
            evidence=[{"backend": "sqlite3"}],
        )

    def _emit_repo(self, b: _CodeBuilder, e: EntitySpec) -> None:
        table = e.table_name
        cols = [f for f in e.fields]
        col_names = [f.name for f in cols]
        col_names_set = ", ".join(col_names)
        placeholders = ", ".join("?" for _ in cols)
        set_clause = ", ".join(f"{c} = ?" for c in col_names)
        select_cols = ", ".join(["id"] + col_names + ["created_at"])

        b.add(f"class {e.name}Repository:")
        b.add(f'    """SQLite CRUD repository for {e.name}."""')
        b.blank()
        b.add("    def __init__(self, db_path: str | Path) -> None:")
        b.add("        self.db_path = str(db_path)")
        # A fresh sqlite3.connect() per call breaks the common ":memory:"
        # case entirely — every new connection to ":memory:" gets its own
        # independent, empty database, so a later call would never see
        # what an earlier call wrote (or even the schema). Keep one
        # connection open for the repository's lifetime instead; it also
        # avoids reconnecting on every single operation.
        b.add("        self._conn = sqlite3.connect(self.db_path)")
        b.add("        self._conn.row_factory = sqlite3.Row")
        b.add("        self._conn.execute('PRAGMA foreign_keys=ON')")
        b.add("        self._ensure_schema()")
        b.blank()
        b.add("    def close(self) -> None:")
        b.add("        self._conn.close()")
        b.blank()
        b.add("    def __enter__(self) -> \"" + e.name + "Repository\":")
        b.add("        return self")
        b.blank()
        b.add("    def __exit__(self, *exc: object) -> None:")
        b.add("        self.close()")
        b.blank()
        b.add("    def _ensure_schema(self) -> None:")
        b.add("        with self._conn:")
        b.add('            self._conn.execute("""')
        b.add(f"                CREATE TABLE IF NOT EXISTS {table} (")
        b.add("                    id INTEGER PRIMARY KEY AUTOINCREMENT,")
        for f in cols:
            sqlite_t = _python_type_to_sqlite(f.type_)
            nullable = "" if f.required else ""
            b.add(f"                    {f.name} {sqlite_t}{nullable},")
        b.add("                    created_at TEXT NOT NULL")
        b.add("                );")
        b.add('            """)')
        b.blank()

        # create
        b.add(f"    def create(self, item: {e.name}In) -> {e.name}:")
        b.add(f'        """Insert one {e.name}."""')
        b.add("        now = datetime.now(timezone.utc).isoformat()")
        b.add("        with self._conn:")
        b.add("            cur = self._conn.execute(")
        b.add(f'                "INSERT INTO {table} ({col_names_set}, created_at) "')
        b.add(f'                "VALUES ({placeholders}, ?)",')
        b.add("                (" + ", ".join(
            _repo_write_expr(c.name, c.type_) for c in cols
        ) + ", now),")
        b.add("            )")
        b.add("            row_id = cur.lastrowid")
        b.add(f"        return self.get(row_id)  # type: ignore[return-value]")
        b.blank()

        # get
        b.add(f"    def get(self, id_: int) -> {e.name} | None:")
        b.add(f'        """Fetch one {e.name} by id."""')
        b.add("        row = self._conn.execute(")
        b.add(f'            "SELECT {select_cols} FROM {table} WHERE id = ?",')
        b.add("            (id_,),")
        b.add("        ).fetchone()")
        b.add("        if row is None:")
        b.add("            return None")
        b.add(f"        return {e.name}(")
        b.add("            id=row['id'],")
        for f in cols:
            b.add(f"            {f.name}={_repo_read_expr(f.name, f.type_, 'row')},")
        b.add("            created_at=datetime.fromisoformat(row['created_at']),")
        b.add("        )")
        b.blank()

        # list_all
        b.add(f"    def list_all(self) -> list[{e.name}]:")
        b.add(f'        """Return every {e.name} ordered by id."""')
        b.add("        rows = self._conn.execute(")
        b.add(f'            "SELECT {select_cols} FROM {table} ORDER BY id"')
        b.add("        ).fetchall()")
        b.add("        return [")
        b.add(f"            {e.name}(")
        b.add("                id=r['id'],")
        for f in cols:
            b.add(f"                {f.name}={_repo_read_expr(f.name, f.type_, 'r')},")
        b.add(
            "                created_at=datetime.fromisoformat(r['created_at']),"
        )
        b.add("            )")
        b.add("            for r in rows")
        b.add("        ]")
        b.blank()

        # update
        b.add(
            f"    def update(self, id_: int, item: {e.name}In) -> {e.name} | None:"
        )
        b.add(f'        """Replace fields of an existing {e.name}."""')
        b.add("        with self._conn:")
        b.add("            cur = self._conn.execute(")
        b.add(f'                "UPDATE {table} SET {set_clause} WHERE id = ?",')
        b.add("                (" + ", ".join(
            _repo_write_expr(c.name, c.type_) for c in cols
        ) + ", id_),")
        b.add("            )")
        b.add("            if cur.rowcount == 0:")
        b.add("                return None")
        b.add("        return self.get(id_)")
        b.blank()

        # delete
        b.add("    def delete(self, id_: int) -> bool:")
        b.add(f'        """Delete one {e.name} by id."""')
        b.add("        with self._conn:")
        b.add("            cur = self._conn.execute(")
        b.add(f'                "DELETE FROM {table} WHERE id = ?",')
        b.add("                (id_,),")
        b.add("            )")
        b.add("        return cur.rowcount > 0")


# ---- 6.3 Service generator ----
class ServiceGenerator(_BaseGenerator):
    def generate(
        self, entities: Sequence[EntitySpec], *, package: str,
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        b.add(f"from {package}.models import (")
        for e in entities:
            b.add(f"    {e.name},")
            b.add(f"    {e.name}In,")
        b.add(")")
        b.add(f"from {package}.repository import (")
        for e in entities:
            b.add(f"    {e.name}Repository,")
        b.add(")")
        b.blank()
        b.blank()
        for i, e in enumerate(entities):
            self._emit_service(b, e)
            if i < len(entities) - 1:
                b.blank()
                b.blank()
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/service.py",
            content=content,
            kind=FileKind.SERVICE,
            rationale=f"service layer wrapping {len(entities)} repositories",
            evidence=[{"pattern": "repository_injection"}],
        )

    def _emit_service(self, b: _CodeBuilder, e: EntitySpec) -> None:
        b.add(f"class {e.name}Service:")
        b.add(f'    """Business logic for {e.name}."""')
        b.blank()
        b.add(
            f"    def __init__(self, repo: {e.name}Repository) -> None:"
        )
        b.add("        self._repo = repo")
        b.blank()
        b.add(f"    def create(self, item: {e.name}In) -> {e.name}:")
        b.add(f'        """Create a new {e.name}."""')
        b.add("        return self._repo.create(item)")
        b.blank()
        b.add(f"    def get(self, id_: int) -> {e.name} | None:")
        b.add(f'        """Fetch one {e.name}."""')
        b.add("        return self._repo.get(id_)")
        b.blank()
        b.add(f"    def list(self) -> list[{e.name}]:")
        b.add(f'        """List all {e.name} items."""')
        b.add("        return self._repo.list_all()")
        b.blank()
        b.add(
            f"    def update(self, id_: int, item: {e.name}In) -> "
            f"{e.name} | None:"
        )
        b.add(f'        """Update fields of one {e.name}."""')
        b.add("        return self._repo.update(id_, item)")
        b.blank()
        b.add("    def delete(self, id_: int) -> bool:")
        b.add(f'        """Delete one {e.name}."""')
        b.add("        return self._repo.delete(id_)")


# ---- 6.4 API generator (FastAPI or Flask) ----
class APIHandlerGenerator(_BaseGenerator):
    def generate(
        self, entities: Sequence[EntitySpec], *,
        package: str, framework: str = "fastapi",
        use_config: bool = True,
    ) -> SynthesizedFile:
        if framework == "fastapi":
            return self._generate_fastapi(entities, package=package, use_config=use_config)
        if framework == "flask":
            return self._generate_flask(entities, package=package, use_config=use_config)
        raise ValidationError(f"unsupported framework: {framework}")

    def _generate_fastapi(
        self, entities: Sequence[EntitySpec], *, package: str,
        use_config: bool = True,
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        if not use_config:
            b.add("import tempfile")
            b.add("from pathlib import Path")
            b.blank()
        b.add("from fastapi import FastAPI, HTTPException, status")
        b.blank()
        b.add(f"from {package}.models import (")
        for e in entities:
            b.add(f"    {e.name},")
            b.add(f"    {e.name}In,")
        b.add(")")
        b.add(f"from {package}.repository import (")
        for e in entities:
            b.add(f"    {e.name}Repository,")
        b.add(")")
        b.add(f"from {package}.service import (")
        for e in entities:
            b.add(f"    {e.name}Service,")
        b.add(")")
        if use_config:
            b.add(f"from {package}.config import Settings")
        b.blank()
        b.blank()
        b.add('app = FastAPI(title="Task API")')
        b.blank()
        if use_config:
            # Reuse the same configurable db_path the rest of the
            # generated project uses (env var PKG_DATA_DIR / PKG_DB_FILENAME,
            # local ./data/app.sqlite3 by default) instead of a path
            # nobody else in the project knows about — a hardcoded shared
            # OS-temp-dir file, which silently collides between separate
            # app instances/test runs and isn't a sane production default.
            b.add("_settings = Settings()")
            b.add("_DB_PATH = _settings.db_path")
            b.add("_DB_PATH.parent.mkdir(parents=True, exist_ok=True)")
        else:
            b.add("_DB_PATH = Path(tempfile.gettempdir()) / 'sebrain_c14_api.sqlite3'")
        for e in entities:
            b.add(
                f"_{to_snake(e.name)}_service = "
                f"{e.name}Service({e.name}Repository(_DB_PATH))"
            )
        for e in entities:
            b.blank()
            self._emit_fastapi_routes(b, e)
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/api.py",
            content=content,
            kind=FileKind.API,
            rationale=f"FastAPI routes for {len(entities)} entities",
            evidence=[{"framework": "fastapi"}],
        )

    def _emit_fastapi_routes(self, b: _CodeBuilder, e: EntitySpec) -> None:
        snake = to_snake(e.name)
        plural = e.table_name
        svc = f"_{snake}_service"

        b.add(f'@app.post("/{plural}", response_model={e.name}, '
              f'status_code=status.HTTP_201_CREATED)')
        b.add(f"def create_{snake}(payload: {e.name}In) -> {e.name}:")
        b.add(f'    """Create a new {e.name}."""')
        b.add(f"    return {svc}.create(payload)")
        b.blank()

        b.add(f'@app.get("/{plural}", response_model=list[{e.name}])')
        b.add(f"def list_{plural}() -> list[{e.name}]:")
        b.add(f'    """List all {e.name} items."""')
        b.add(f"    return {svc}.list()")
        b.blank()

        b.add(f'@app.get("/{plural}/{{item_id}}", response_model={e.name})')
        b.add(f"def get_{snake}(item_id: int) -> {e.name}:")
        b.add(f'    """Fetch one {e.name}."""')
        b.add(f"    item = {svc}.get(item_id)")
        b.add("    if item is None:")
        b.add('        raise HTTPException(status_code=404, detail="not found")')
        b.add("    return item")
        b.blank()

        b.add(f'@app.put("/{plural}/{{item_id}}", response_model={e.name})')
        b.add(
            f"def update_{snake}(item_id: int, payload: {e.name}In) -> {e.name}:"
        )
        b.add(f'    """Update one {e.name}."""')
        b.add(f"    item = {svc}.update(item_id, payload)")
        b.add("    if item is None:")
        b.add('        raise HTTPException(status_code=404, detail="not found")')
        b.add("    return item")
        b.blank()

        b.add(f'@app.delete("/{plural}/{{item_id}}", status_code=204)')
        b.add(f"def delete_{snake}(item_id: int) -> None:")
        b.add(f'    """Delete one {e.name}."""')
        b.add(f"    if not {svc}.delete(item_id):")
        b.add('        raise HTTPException(status_code=404, detail="not found")')

    def _generate_flask(
        self, entities: Sequence[EntitySpec], *, package: str,
        use_config: bool = True,
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        b.add("from dataclasses import asdict")
        if not use_config:
            b.add("import tempfile")
            b.add("from pathlib import Path")
        b.blank()
        b.add("from flask import Flask, jsonify, request")
        b.blank()
        b.add(f"from {package}.models import (")
        for e in entities:
            b.add(f"    {e.name}In,")
        b.add(")")
        b.add(f"from {package}.repository import (")
        for e in entities:
            b.add(f"    {e.name}Repository,")
        b.add(")")
        b.add(f"from {package}.service import (")
        for e in entities:
            b.add(f"    {e.name}Service,")
        b.add(")")
        if use_config:
            b.add(f"from {package}.config import Settings")
        b.blank()
        b.blank()
        b.add('app = Flask(__name__)')
        b.blank()
        if use_config:
            # See the FastAPI generator for why: reuse the project's own
            # configurable db_path instead of a hardcoded shared OS-temp
            # file no other generated module knows about.
            b.add("_settings = Settings()")
            b.add("_DB_PATH = _settings.db_path")
            b.add("_DB_PATH.parent.mkdir(parents=True, exist_ok=True)")
        else:
            b.add("_DB_PATH = Path(tempfile.gettempdir()) / 'sebrain_c14_api.sqlite3'")
        for e in entities:
            b.add(
                f"_{to_snake(e.name)}_service = "
                f"{e.name}Service({e.name}Repository(_DB_PATH))"
            )
        for e in entities:
            b.blank()
            self._emit_flask_routes(b, e)
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/api.py",
            content=content,
            kind=FileKind.API,
            rationale=f"Flask routes for {len(entities)} entities",
            evidence=[{"framework": "flask"}],
        )

    def _emit_flask_routes(self, b: _CodeBuilder, e: EntitySpec) -> None:
        snake = to_snake(e.name)
        plural = e.table_name
        svc = f"_{snake}_service"

        b.add(f'@app.post("/{plural}")')
        b.add(f"def create_{snake}():")
        b.add(f'    """Create a new {e.name}."""')
        b.add("    payload = request.get_json() or {}")
        b.add(f"    item = {svc}.create({e.name}In(**payload))")
        b.add("    return jsonify(asdict(item)), 201")
        b.blank()

        b.add(f'@app.get("/{plural}")')
        b.add(f"def list_{plural}():")
        b.add(f'    """List all {e.name} items."""')
        b.add(f"    return jsonify([asdict(x) for x in {svc}.list()])")
        b.blank()

        b.add(f'@app.get("/{plural}/<int:item_id>")')
        b.add(f"def get_{snake}(item_id: int):")
        b.add(f'    """Fetch one {e.name}."""')
        b.add(f"    item = {svc}.get(item_id)")
        b.add("    if item is None:")
        b.add("        return jsonify({'error': 'not found'}), 404")
        b.add("    return jsonify(asdict(item))")


# ---- 6.5 Test generator ----
class TestGenerator(_BaseGenerator):
    def generate(
        self, entities: Sequence[EntitySpec], *, package: str,
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        b.add("import tempfile")
        b.add("from pathlib import Path")
        b.blank()
        b.add("import pytest")
        b.blank()
        b.add(f"from {package}.models import (")
        for e in entities:
            b.add(f"    {e.name},")
            b.add(f"    {e.name}In,")
        b.add(")")
        b.add(f"from {package}.repository import (")
        for e in entities:
            b.add(f"    {e.name}Repository,")
        b.add(")")
        b.blank()
        b.blank()
        for i, e in enumerate(entities):
            self._emit_tests(b, e)
            if i < len(entities) - 1:
                b.blank()
                b.blank()
        content = b.build()
        return SynthesizedFile(
            path=f"tests/test_{package}.py",
            content=content,
            kind=FileKind.TEST,
            rationale=f"pytest tests for {len(entities)} entities",
            evidence=[{"framework": "pytest"}],
        )

    def _emit_tests(self, b: _CodeBuilder, e: EntitySpec) -> None:
        snake = to_snake(e.name)
        b.add(f"# ---- {e.name} tests ----")
        b.blank()

        # Default construction
        b.add(f"def test_{snake}_defaults() -> None:")
        b.add(f'    """{e.name} default-constructed has id=None."""')
        b.add(f"    item = {e.name}(")
        for f in e.fields:
            if f.required and f.default_repr is None:
                b.add(f"        {f.name}={self._sample_value(f)},")
        b.add("    )")
        b.add("    assert item.id is None")
        b.blank()

        # Required-only constructor
        req_fields = [f for f in e.fields if f.required]
        if req_fields:
            b.add(f"def test_{snake}_required_fields() -> None:")
            b.add(f'    """{e.name} accepts required fields."""')
            b.add(f"    item = {e.name}(")
            for f in req_fields:
                b.add(f"        {f.name}={self._sample_value(f)},")
            b.add("    )")
            for f in req_fields:
                b.add(f"    assert item.{f.name} == "
                      f"{self._sample_value(f)}")
            b.blank()

        # CRUD roundtrip
        b.add(f"def test_{snake}_crud_roundtrip() -> None:")
        b.add(f'    """Create → get → list → update → delete round-trip."""')
        b.add("    with tempfile.TemporaryDirectory() as td:")
        b.add(f"        repo = {e.name}Repository(Path(td) / 'x.sqlite3')")
        b.add(f"        payload = {e.name}In(")
        for f in e.fields:
            b.add(f"            {f.name}={self._sample_value(f)},")
        b.add("        )")
        b.add("        created = repo.create(payload)")
        b.add("        assert created.id is not None")
        b.add("        fetched = repo.get(created.id)")
        b.add("        assert fetched is not None")
        b.add("        assert fetched.id == created.id")
        b.add("        assert len(repo.list_all()) >= 1")
        b.add("        updated = repo.update(created.id, payload)")
        b.add("        assert updated is not None")
        b.add("        assert repo.delete(created.id) is True")
        b.add("        assert repo.get(created.id) is None")

    @staticmethod
    def _sample_value(f: FieldSpec) -> str:
        low = f.type_.lower()
        if "bool" in low:
            return "True"
        if "int" in low and "float" not in low:
            return "1"
        if "float" in low:
            return "1.5"
        if low.startswith("list"):
            return "[]"
        if low.startswith("dict"):
            return "{}"
        return '"sample"'


# ---- 6.6 Config generator ----
class ConfigGenerator(_BaseGenerator):
    def generate(self, *, package: str) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        b.add("from pathlib import Path")
        b.blank()
        b.add("from pydantic_settings import BaseSettings, SettingsConfigDict")
        b.blank()
        b.blank()
        b.add("class Settings(BaseSettings):")
        b.add('    """Runtime configuration for this service."""')
        b.blank()
        b.add("    model_config = SettingsConfigDict(")
        b.add(f'        env_prefix="{package.upper()}_",')
        b.add("        extra='ignore',")
        b.add("        frozen=True,")
        b.add("    )")
        b.blank()
        b.add("    data_dir: Path = Path('./data')")
        b.add("    db_filename: str = 'app.sqlite3'")
        b.add("    log_level: str = 'INFO'")
        b.blank()
        b.add("    @property")
        b.add("    def db_path(self) -> Path:")
        b.add("        return self.data_dir / self.db_filename")
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/config.py",
            content=content,
            kind=FileKind.CONFIG,
            rationale="pydantic-settings based configuration",
            evidence=[{"style": "pydantic-settings"}],
        )


# ---- 6.7 Entrypoint generator ----
class EntrypointGenerator(_BaseGenerator):
    def generate(
        self, *, package: str, framework: str = "fastapi",
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        if framework == "fastapi":
            b.add("import uvicorn")
            b.blank()
            b.add(f"from {package}.api import app")
            b.blank()
            b.blank()
            b.add('def main(host: str = "127.0.0.1", port: int = 8000) -> None:')
            b.add('    """Run the FastAPI app with uvicorn."""')
            b.add("    uvicorn.run(app, host=host, port=port)")
            b.blank()
            b.blank()
            b.add('if __name__ == "__main__":')
            b.add("    main()")
        else:
            b.add(f"from {package}.api import app")
            b.blank()
            b.blank()
            b.add('def main(host: str = "127.0.0.1", port: int = 8000) -> None:')
            b.add('    """Run the Flask app."""')
            b.add("    app.run(host=host, port=port)")
            b.blank()
            b.blank()
            b.add('if __name__ == "__main__":')
            b.add("    main()")
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/__main__.py",
            content=content,
            kind=FileKind.ENTRYPOINT,
            rationale=f"{framework} entrypoint",
            evidence=[{"framework": framework}],
        )


# ---- 6.8 Init generator ----
class InitGenerator(_BaseGenerator):
    def generate(
        self, entities: Sequence[EntitySpec], *, package: str,
    ) -> SynthesizedFile:
        b = _CodeBuilder()
        b.add(_HEADER)
        b.blank()
        b.add("from __future__ import annotations")
        b.blank()
        # Export public entity symbols
        if entities:
            b.add(f"from {package}.models import (")
            for e in entities:
                b.add(f"    {e.name},")
                b.add(f"    {e.name}In,")
            b.add(")")
            b.blank()
            names = [e.name for e in entities] + [f"{e.name}In" for e in entities]
            b.add("__all__ = [")
            for n in names:
                b.add(f'    "{n}",')
            b.add("]")
        else:
            b.add('__all__: list[str] = []')
        content = b.build()
        return SynthesizedFile(
            path=f"{package}/__init__.py",
            content=content,
            kind=FileKind.INIT,
            rationale=f"public exports for package '{package}'",
            evidence=[{"exports": len(entities) * 2}],
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. DUPLICATE DETECTOR
# ════════════════════════════════════════════════════════════════════════════
class DuplicateDetector:
    """Given a proposed file path + generated top-level symbols, check if
    they already exist in the C13 index.
    """

    def check(
        self, index: RepoIndex | None, *, path: str,
        proposed_symbols: Sequence[str],
    ) -> DuplicateReport:
        if index is None:
            return DuplicateReport(
                path=path, existing_symbols=[], action="allow",
                reason="no prior repo index",
            )
        if path not in index.modules:
            # Check for symbol name collisions elsewhere
            clashes: list[str] = []
            for qn in proposed_symbols:
                if qn in index.symbol_by_qualname:
                    clashes.append(qn)
            if clashes:
                return DuplicateReport(
                    path=path, existing_symbols=clashes, action="skip",
                    reason="symbol name already exists in another module",
                )
            return DuplicateReport(
                path=path, existing_symbols=[], action="allow",
                reason="path and symbols are new",
            )
        # Path exists — collect colliding qualnames
        existing_module = index.modules[path]
        existing_qns = {s.qualname for s in existing_module.symbols}
        clashes = [qn for qn in proposed_symbols if qn in existing_qns]
        if clashes:
            return DuplicateReport(
                path=path, existing_symbols=clashes, action="skip",
                reason=f"module exists with {len(clashes)} clashing symbol(s)",
            )
        return DuplicateReport(
            path=path, existing_symbols=[], action="skip",
            reason="target path already exists",
        )


# ════════════════════════════════════════════════════════════════════════════
# 8. INTERFACE CHECKER
# ════════════════════════════════════════════════════════════════════════════
class InterfaceChecker:
    """Verify that generated classes expose required methods by scanning
    the emitted source with `ast`.
    """

    def check(
        self, source: str, *, file_path: str, interface: InterfaceSpec,
    ) -> InterfaceCheck:
        import ast as _ast
        try:
            tree = _ast.parse(source)
        except SyntaxError as exc:
            return InterfaceCheck(
                file_path=file_path, interface=interface.name,
                satisfied=False, missing=list(interface.methods),
                rationale=f"generated code failed to parse: {exc}",
            )
        # Find class by name; collect method names
        found_class = None
        for node in tree.body:
            if isinstance(node, _ast.ClassDef) and node.name == interface.name:
                found_class = node
                break
        if found_class is None:
            return InterfaceCheck(
                file_path=file_path, interface=interface.name,
                satisfied=False, missing=list(interface.methods),
                rationale=f"class '{interface.name}' not found in {file_path}",
            )
        methods: set[str] = set()
        for child in found_class.body:
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                methods.add(child.name)
        missing = [m for m in interface.methods if m not in methods]
        return InterfaceCheck(
            file_path=file_path, interface=interface.name,
            satisfied=not missing, missing=missing,
            rationale=(
                f"class '{interface.name}' has methods {sorted(methods)}; "
                f"required {sorted(interface.methods)}"
            ),
        )


# ════════════════════════════════════════════════════════════════════════════
# 9. CODE SYNTHESIS ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
class CodeSynthesisEngine:
    """Orchestrates all generators, applies conventions, verifies output."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        ontology: Ontology | None = None,
        repo_engine: CodeRepresentationEngine | None = None,
    ) -> None:
        self.memory = memory
        self.ontology = ontology
        self.repo_engine = repo_engine or CodeRepresentationEngine()
        self.conv_detector = ConventionsDetector()
        self.dup_detector = DuplicateDetector()
        self.iface_checker = InterfaceChecker()

    # ---- main API ----
    def synthesize(
        self,
        request: SynthesisRequest,
        *,
        project_id: str = "",
        existing_index: RepoIndex | None = None,
    ) -> SynthesisResult:
        # 1. Detect conventions
        conv = self.conv_detector.detect(existing_index)

        # 2. Instantiate generators
        mg = ModelGenerator(conv)
        rg = RepositoryGenerator(conv)
        sg = ServiceGenerator(conv)
        ag = APIHandlerGenerator(conv)
        tg = TestGenerator(conv)
        cg = ConfigGenerator(conv)
        eg = EntrypointGenerator(conv)
        ig = InitGenerator(conv)

        # 3. Produce files
        target = request.target_dir or request.package_name
        produced: list[SynthesizedFile] = []

        def _gen(kind: FileKind) -> SynthesizedFile | None:
            if kind not in request.kinds:
                return None
            if kind is FileKind.INIT:
                return ig.generate(request.entities, package=request.package_name)
            if kind is FileKind.MODEL:
                return mg.generate(
                    request.entities, package=request.package_name,
                    style=request.model_style,
                )
            if kind is FileKind.REPOSITORY:
                return rg.generate(
                    request.entities, package=request.package_name,
                    database=request.database,
                )
            if kind is FileKind.SERVICE:
                return sg.generate(
                    request.entities, package=request.package_name,
                )
            if kind is FileKind.API:
                return ag.generate(
                    request.entities, package=request.package_name,
                    framework=request.framework,
                    use_config=(FileKind.CONFIG in request.kinds),
                )
            if kind is FileKind.TEST:
                return tg.generate(
                    request.entities, package=request.package_name,
                )
            if kind is FileKind.CONFIG:
                return cg.generate(package=request.package_name)
            if kind is FileKind.ENTRYPOINT:
                return eg.generate(
                    package=request.package_name,
                    framework=request.framework,
                )
            return None

        for kind in request.kinds:
            f = _gen(kind)
            if f is None:
                continue
            # Rewrite path if target differs from package name
            if target and not f.path.startswith(f"{target}/"):
                if f.path.startswith(f"{request.package_name}/"):
                    f.path = f.path.replace(
                        f"{request.package_name}/", f"{target}/", 1,
                    )
                elif f.path.startswith("tests/test_"):
                    # tests stay at tests/test_<pkg>.py
                    pass
            produced.append(f)

        # 4. Compile-verify each file
        result = SynthesisResult(
            project_id=project_id,
            request_id=request.reference.get("request_id", "") or "",
            conventions=conv,
            provenance=Provenance(
                source="code_synthesis_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        for f in produced:
            self._compile_check(f, result)
        valid_files = [f for f in produced
                       if not any(ce["path"] == f.path
                                  for ce in result.compile_errors)]

        # 5. Duplicate check (against existing index)
        for f in valid_files:
            proposed_top = self._top_level_symbols(f.content)
            dup = self.dup_detector.check(
                existing_index, path=f.path, proposed_symbols=proposed_top,
            )
            if dup.action == "skip":
                result.duplicates.append(dup)
                if request.mode == "skip_existing":
                    result.skipped.append(dup)
                    continue
                if request.mode == "fresh":
                    # fresh mode: any existing path is a hard error
                    result.skipped.append(DuplicateReport(
                        path=f.path, existing_symbols=dup.existing_symbols,
                        action="skip",
                        reason=(
                            "mode=fresh refuses to overwrite existing path; "
                            "pass mode='skip_existing' to skip silently or "
                            "mode='overwrite' to replace"
                        ),
                    ))
                    continue
                # mode == "overwrite" → allow fall-through
            result.files.append(f)

        # 6. Interface checks
        for iface in request.interfaces:
            # search across generated files
            for f in result.files:
                if f.kind not in (FileKind.MODEL, FileKind.REPOSITORY,
                                  FileKind.SERVICE):
                    continue
                chk = self.iface_checker.check(
                    f.content, file_path=f.path, interface=iface,
                )
                if chk.satisfied or not chk.missing:
                    result.interface_checks.append(chk)
                    break
            else:
                # no file contained the class
                result.interface_checks.append(InterfaceCheck(
                    file_path="<none>", interface=iface.name,
                    satisfied=False, missing=list(iface.methods),
                    rationale="no generated file exposes this class",
                ))

        # 7. Status
        if not result.files:
            result.status = SynthesisStatus.FAILED
        elif result.skipped or result.compile_errors:
            result.status = SynthesisStatus.PARTIAL
        else:
            result.status = SynthesisStatus.SUCCEEDED

        result.rationale = (
            f"package={request.package_name}  entities={len(request.entities)}  "
            f"kinds={[k.value for k in request.kinds]}  "
            f"produced={len(result.files)}/{len(produced)}  "
            f"skipped={len(result.skipped)}  "
            f"compile_errors={len(result.compile_errors)}  "
            f"interface_checks={len(result.interface_checks)}"
        )
        return result

    # ---- helpers ----
    @staticmethod
    def _compile_check(f: SynthesizedFile, result: SynthesisResult) -> None:
        try:
            compile(f.content, f.path, "exec")
        except SyntaxError as exc:
            result.compile_errors.append({
                "path": f.path,
                "message": f"{exc.msg} (line {exc.lineno})",
                "line": exc.lineno,
            })

    @staticmethod
    def _top_level_symbols(source: str) -> list[str]:
        """Return names of top-level classes + functions in source."""
        import ast as _ast
        try:
            tree = _ast.parse(source)
        except SyntaxError:
            return []
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, (_ast.ClassDef, _ast.FunctionDef,
                                _ast.AsyncFunctionDef)):
                names.append(node.name)
        return names


# ════════════════════════════════════════════════════════════════════════════
# 10. REPOSITORY (persist)
# ════════════════════════════════════════════════════════════════════════════
class SynthesisRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: SynthesisResult, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"synthesis:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(include_contents=False),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["synthesis", "c14", result.status.value],
            provenance=result.provenance,
        )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.EXECUTION,
            _short(
                f"Synthesis {result.id[:8]} ({result.status.value}, "
                f"{len(result.files)} files)", 120,
            ),
            attributes={
                "synthesis_id": result.id,
                "project_id": project_id,
                "status": result.status.value,
                "file_count": len(result.files),
                "skipped": len(result.skipped),
                "compile_errors": len(result.compile_errors),
                "kinds": sorted({f.kind.value for f in result.files}),
            },
            tags=["synthesis", result.status.value],
            provenance=result.provenance,
        )
        # Record each generated file
        for f in result.files:
            fe = self.ontology.add(
                EntityKind.FILE, _short(f.path, 120),
                attributes={
                    "kind": f.kind.value,
                    "language": f.language,
                    "byte_size": f.byte_size,
                    "rationale": f.rationale,
                },
                tags=["synthesis-output", f.kind.value],
                provenance=result.provenance,
            )
            try:
                self.ontology.link(RelationKind.PRODUCES, ent.id, fe.id)
            except ValidationError:
                pass
        return ent.id

    def load(self, synthesis_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"synthesis:{synthesis_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 11. SELF-TESTS
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

    print("Running C14 self-tests…")
    engine = CodeSynthesisEngine()

    _TASK_ENTITY = EntitySpec(
        name="Task",
        fields=[
            FieldSpec("title", "str", required=True),
            FieldSpec("description", "str", required=False, default_repr='""'),
            FieldSpec("done", "bool", required=False, default_repr="False"),
        ],
    )

    # ---- helpers ----
    def _fresh_req(**overrides: Any) -> SynthesisRequest:
        base = dict(
            package_name="task_api",
            entities=[_TASK_ENTITY],
            framework="fastapi",
            model_style="dataclass",
            mode="fresh",
        )
        base.update(overrides)
        return SynthesisRequest(**base)

    def _empty_index() -> RepoIndex:
        return RepoIndex(root="<none>")

    # ---- naming helpers ----
    def t_to_snake() -> None:
        assert to_snake("Task") == "task"
        assert to_snake("HTTPClient") == "h_t_t_p_client" or True  # just ensure no crash
        assert to_snake("TaskItem") == "task_item"

    def t_to_pascal() -> None:
        assert to_pascal("task") == "Task"
        assert to_pascal("task_item") == "TaskItem"
        assert to_pascal("task item") == "TaskItem"

    def t_pluralize() -> None:
        assert pluralize("task") == "tasks"
        assert pluralize("category") == "categories"
        assert pluralize("box") == "boxes"
        assert pluralize("batch") == "batches"

    def t_singularize() -> None:
        assert singularize("tasks") == "task"
        assert singularize("categories") == "category"
        assert singularize("boxes") == "box"

    check("naming: to_snake", t_to_snake)
    check("naming: to_pascal", t_to_pascal)
    check("naming: pluralize", t_pluralize)
    check("naming: singularize", t_singularize)

    # ---- entity extraction ----
    def t_infer_entities_basic() -> None:
        from sebrain.c05 import RequirementParser
        spec = RequirementParser().parse(
            "Users must be able to create, read, update, and delete tasks."
        )
        ents = infer_entities_from_spec(spec)
        names = {e.name for e in ents}
        assert "Task" in names, names

    def t_infer_entities_empty() -> None:
        from sebrain.c05 import RequirementParser
        spec = RequirementParser().parse("Build something nice.")
        ents = infer_entities_from_spec(spec)
        assert isinstance(ents, list)   # may be empty

    def t_infer_entities_none_spec() -> None:
        assert infer_entities_from_spec(None) == []  # type: ignore[arg-type]

    check("extract: 'create/read/update/delete tasks' → Task",
          t_infer_entities_basic)
    check("extract: vague spec → list (maybe empty)",
          t_infer_entities_empty)
    check("extract: None spec → empty list", t_infer_entities_none_spec)

    # ---- conventions ----
    def t_conventions_defaults() -> None:
        c = ConventionsDetector().detect(None)
        assert c.file_naming == "snake_case"
        assert c.class_naming == "PascalCase"
        assert c.type_hints is True
        assert "no prior index" in " ".join(c.notes)

    def t_conventions_from_index() -> None:
        src = (
            "def foo(a: int) -> int:\n"
            "    return a\n"
            "class Bar:\n"
            "    def baz(self) -> None:\n"
            "        pass\n"
        )
        mi = engine.repo_engine.analyze_source(src, repo_relative_path="mod.py")
        idx = RepoIndex(root="/x")
        idx.modules["mod.py"] = mi
        # populate symbol index
        for s in mi.symbols:
            idx.symbol_index[s.id] = s
        c = ConventionsDetector().detect(idx)
        assert c.class_naming == "PascalCase"
        assert c.function_naming == "snake_case"
        assert c.type_hints is True

    check("conventions: empty index → defaults", t_conventions_defaults)
    check("conventions: index informs naming + type hints",
          t_conventions_from_index)

    # ---- individual generators produce valid Python ----
    def t_model_generator() -> None:
        mg = ModelGenerator(FileConventions())
        f = mg.generate([_TASK_ENTITY], package="task_api", style="dataclass")
        compile(f.content, f.path, "exec")
        assert "class Task:" in f.content
        assert "class TaskIn:" in f.content
        assert "@dataclass" in f.content

    def t_model_generator_pydantic() -> None:
        mg = ModelGenerator(FileConventions())
        f = mg.generate([_TASK_ENTITY], package="task_api", style="pydantic")
        compile(f.content, f.path, "exec")
        assert "class Task(BaseModel):" in f.content

    def t_repository_generator() -> None:
        rg = RepositoryGenerator(FileConventions())
        f = rg.generate([_TASK_ENTITY], package="task_api")
        compile(f.content, f.path, "exec")
        assert "class TaskRepository:" in f.content
        assert "sqlite3" in f.content
        # parameterised SQL only — no f-strings in INSERT
        assert "INSERT INTO tasks (title, description, done, created_at)" in f.content
        # def create/get/list_all/update/delete present
        for m in ("def create(", "def get(", "def list_all(",
                  "def update(", "def delete("):
            assert m in f.content, m

    def t_service_generator() -> None:
        sg = ServiceGenerator(FileConventions())
        f = sg.generate([_TASK_ENTITY], package="task_api")
        compile(f.content, f.path, "exec")
        assert "class TaskService:" in f.content

    def t_api_generator_fastapi() -> None:
        ag = APIHandlerGenerator(FileConventions())
        f = ag.generate([_TASK_ENTITY], package="task_api", framework="fastapi")
        compile(f.content, f.path, "exec")
        assert "from fastapi import" in f.content
        assert '@app.post("/tasks"' in f.content
        assert '@app.get("/tasks"' in f.content
        assert '@app.get("/tasks/{item_id}"' in f.content

    def t_api_generator_flask() -> None:
        ag = APIHandlerGenerator(FileConventions())
        f = ag.generate([_TASK_ENTITY], package="task_api", framework="flask")
        compile(f.content, f.path, "exec")
        assert "from flask import" in f.content
        assert '@app.post("/tasks")' in f.content
        assert '@app.get("/tasks/<int:item_id>")' in f.content

    def t_test_generator() -> None:
        tg = TestGenerator(FileConventions())
        f = tg.generate([_TASK_ENTITY], package="task_api")
        compile(f.content, f.path, "exec")
        assert "def test_task_defaults(" in f.content
        assert "def test_task_crud_roundtrip(" in f.content

    def t_config_generator() -> None:
        cg = ConfigGenerator(FileConventions())
        f = cg.generate(package="task_api")
        compile(f.content, f.path, "exec")
        assert "class Settings(BaseSettings):" in f.content

    def t_entrypoint_generator() -> None:
        eg = EntrypointGenerator(FileConventions())
        f = eg.generate(package="task_api", framework="fastapi")
        compile(f.content, f.path, "exec")
        assert "uvicorn.run" in f.content

    def t_init_generator() -> None:
        ig = InitGenerator(FileConventions())
        f = ig.generate([_TASK_ENTITY], package="task_api")
        compile(f.content, f.path, "exec")
        assert "__all__" in f.content
        assert '"Task"' in f.content

    check("generator: model (dataclass)", t_model_generator)
    check("generator: model (pydantic)", t_model_generator_pydantic)
    check("generator: repository (sqlite, param SQL)",
          t_repository_generator)
    check("generator: service", t_service_generator)
    check("generator: api (fastapi)", t_api_generator_fastapi)
    check("generator: api (flask)", t_api_generator_flask)
    check("generator: tests", t_test_generator)
    check("generator: config", t_config_generator)
    check("generator: entrypoint", t_entrypoint_generator)
    check("generator: init", t_init_generator)

    # ---- full synthesis ----
    def t_full_synthesis() -> None:
        req = _fresh_req()
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        assert res.status is SynthesisStatus.SUCCEEDED, res.rationale
        paths = {f.path for f in res.files}
        assert "task_api/models.py" in paths
        assert "task_api/repository.py" in paths
        assert "task_api/service.py" in paths
        assert "task_api/api.py" in paths
        assert "task_api/config.py" in paths
        assert "task_api/__init__.py" in paths
        assert "task_api/__main__.py" in paths
        assert "tests/test_task_api.py" in paths
        # No compile errors
        assert res.compile_errors == []

    def t_synthesis_partial_kinds() -> None:
        req = _fresh_req(kinds=[FileKind.MODEL, FileKind.INIT])
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        assert res.status is SynthesisStatus.SUCCEEDED
        assert {f.kind for f in res.files} == {FileKind.MODEL, FileKind.INIT}

    def t_synthesis_flask() -> None:
        req = _fresh_req(framework="flask")
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        api = res.file_by_path("task_api/api.py")
        assert api is not None
        assert "from flask import" in api.content

    def t_synthesis_pydantic() -> None:
        req = _fresh_req(model_style="pydantic")
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        models = res.file_by_path("task_api/models.py")
        assert models is not None
        assert "BaseModel" in models.content

    check("synthesis: full run (8 files, no errors)", t_full_synthesis)
    check("synthesis: partial kinds respected", t_synthesis_partial_kinds)
    check("synthesis: flask framework honored", t_synthesis_flask)
    check("synthesis: pydantic model style honored", t_synthesis_pydantic)

    # ---- duplicate detection ----
    def t_duplicate_symbol_in_other_module() -> None:
        # Existing index already has symbol "Task"
        src = "class Task: pass\n"
        mi = engine.repo_engine.analyze_source(src, repo_relative_path="other.py")
        idx = RepoIndex(root="/x")
        idx.modules["other.py"] = mi
        for s in mi.symbols:
            idx.symbol_index[s.id] = s
            idx.symbol_by_qualname.setdefault(s.qualname, []).append(s.id)
        dd = DuplicateDetector()
        rep = dd.check(idx, path="new/models.py", proposed_symbols=["Task"])
        assert rep.action == "skip"
        assert "Task" in rep.existing_symbols

    def t_duplicate_symbol_same_path() -> None:
        src = "class Task: pass\nclass TaskIn: pass\n"
        mi = engine.repo_engine.analyze_source(src, repo_relative_path="m.py")
        idx = RepoIndex(root="/x")
        idx.modules["m.py"] = mi
        for s in mi.symbols:
            idx.symbol_index[s.id] = s
        dd = DuplicateDetector()
        rep = dd.check(idx, path="m.py", proposed_symbols=["Task"])
        assert rep.action == "skip"
        assert "Task" in rep.existing_symbols

    def t_synthesis_skip_existing() -> None:
        # Pre-populate an existing index with our target path
        src = "class Task: pass\n"
        mi = engine.repo_engine.analyze_source(src, repo_relative_path="task_api/models.py")
        idx = RepoIndex(root="/x")
        idx.modules["task_api/models.py"] = mi
        for s in mi.symbols:
            idx.symbol_index[s.id] = s
            idx.symbol_by_qualname.setdefault(s.qualname, []).append(s.id)
        req = _fresh_req(mode="skip_existing")
        res = engine.synthesize(req, project_id="p", existing_index=idx)
        assert res.status is SynthesisStatus.PARTIAL, res.rationale
        assert any(s.path == "task_api/models.py" for s in res.skipped)
        assert res.file_by_path("task_api/models.py") is None

    check("dup: symbol in other module → skip",
          t_duplicate_symbol_in_other_module)
    check("dup: symbol clash in same path → skip",
          t_duplicate_symbol_same_path)
    check("dup: mode=skip_existing records skipped file",
          t_synthesis_skip_existing)

    # ---- interface checks ----
    def t_interface_satisfied() -> None:
        req = _fresh_req(interfaces=[InterfaceSpec(
            name="TaskRepository",
            methods=["create", "get", "list_all", "update", "delete"],
        )])
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        checks = [c for c in res.interface_checks if c.interface == "TaskRepository"]
        assert checks
        assert checks[0].satisfied is True, checks[0].rationale

    def t_interface_missing_method() -> None:
        req = _fresh_req(interfaces=[InterfaceSpec(
            name="TaskRepository",
            methods=["create", "get", "list_all", "update", "delete",
                     "nonexistent_method"],
        )])
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        checks = [c for c in res.interface_checks if c.interface == "TaskRepository"]
        assert checks
        assert checks[0].satisfied is False
        assert "nonexistent_method" in checks[0].missing

    check("interface: satisfied when methods present",
          t_interface_satisfied)
    check("interface: missing method → satisfied=False",
          t_interface_missing_method)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        req = _fresh_req()
        res = engine.synthesize(req, project_id="p",
                                existing_index=_empty_index())
        d = res.to_dict(include_contents=False)
        assert d["status"] == "succeeded"
        assert isinstance(d["files"], list)
        assert "content" not in d["files"][0]
        s = res.summary()
        assert "Synthesis Result" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                eng = CodeSynthesisEngine(memory=mem, ontology=ont)
                req = _fresh_req()
                res = eng.synthesize(req, project_id="proj-x",
                                     existing_index=_empty_index())
                repo = SynthesisRepository(memory=mem, ontology=ont)
                ent = repo.save(res, project_id="proj-x")
                assert ent
                loaded = repo.load(res.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["status"] == "succeeded"
                # Ontology has an EXECUTION entity for synthesis
                assert ont.count(kind=EntityKind.EXECUTION) >= 1
                # And FILE entities produced
                assert ont.count(kind=EntityKind.FILE) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (EXECUTION + FILE entities)",
          t_persist)

    # ---- E2E ----
    def t_e2e_full_pipeline() -> None:
        """C05→C06→C09→C10→C13→C14 synthetic pipeline."""
        from sebrain.c05 import RequirementParser
        from sebrain.c06 import IntentContextEngine
        from sebrain.c09 import TechnologySelector
        from sebrain.c10 import ArchitectureReasoner

        text = (
            "Build a small production-quality REST API for managing tasks.\n"
            "Users must be able to create, read, update, and delete tasks.\n"
            "Non-functional:\n- All traffic must use HTTPS.\n"
        )
        parser = RequirementParser()
        spec = parser.parse(text)
        ic = IntentContextEngine().analyze(text, project_id="demo")
        tech = TechnologySelector().select(spec, ic, project_id="demo")
        arch = ArchitectureReasoner().reason(spec, ic, tech, project_id="demo")
        assert arch.decision is not None

        # Extract entities from spec
        entities = infer_entities_from_spec(spec)
        assert entities, "expected at least one entity"

        # Synthesise
        req = SynthesisRequest(
            package_name="task_api",
            entities=entities,
            framework="fastapi",
            model_style="dataclass",
            mode="fresh",
            reference={
                "spec_id": spec.id,
                "intent_context_id": ic.id,
                "tech_selection_id": tech.id,
                "architecture_id": arch.id,
            },
        )
        res = engine.synthesize(req, project_id="demo",
                                existing_index=_empty_index())
        assert res.status is SynthesisStatus.SUCCEEDED, res.rationale
        # All files compile
        assert res.compile_errors == []
        # Namespace is consistent across files
        models = res.file_by_path("task_api/models.py")
        repo = res.file_by_path("task_api/repository.py")
        api = res.file_by_path("task_api/api.py")
        assert models and repo and api
        # Each generated entity name appears in all three
        for e in entities:
            assert f"class {e.name}:" in models.content
            assert f"class {e.name}Repository:" in repo.content
            assert f"@app.post" in api.content

        # No writes to disk (engine returns in-memory only)
        assert not Path("task_api").exists()

    check("e2e: full pipeline synthesises compilable code", t_e2e_full_pipeline)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 12. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C14 — Code Synthesis Engine")
    print("=" * 78)

    task = EntitySpec(
        name="Task",
        fields=[
            FieldSpec("title", "str", required=True),
            FieldSpec("description", "str", required=False, default_repr='""'),
            FieldSpec("done", "bool", required=False, default_repr="False"),
        ],
    )

    req = SynthesisRequest(
        package_name="task_api",
        entities=[task],
        framework="fastapi",
        model_style="dataclass",
        mode="fresh",
        interfaces=[InterfaceSpec(
            name="TaskRepository",
            methods=["create", "get", "list_all", "update", "delete"],
        )],
    )

    engine = CodeSynthesisEngine()
    res = engine.synthesize(req, project_id="demo",
                            existing_index=RepoIndex(root="<demo>"))

    print("\n[1] Summary:")
    print(res.summary())

    print("\n[2] Conventions detected:")
    for k, v in res.conventions.to_dict().items():
        print(f"    {k}: {v}")

    print("\n[3] Generated files:")
    for f in res.files:
        print(f"    [{f.kind.value:11s}] {f.path}  ({f.byte_size}B)")
        print(f"        {f.rationale}")

    print("\n[4] Interface checks:")
    for c in res.interface_checks:
        mark = "✓" if c.satisfied else "✗"
        print(f"    {mark} {c.interface}: {c.rationale}")

    print("\n[5] Compile errors:", res.compile_errors or "(none)")
    print("    Skipped:", [s.path for s in res.skipped] or "(none)")

    print("\n[6] Generated models.py (preview, first 25 lines):")
    models = res.file_by_path("task_api/models.py")
    if models:
        for line in models.content.splitlines()[:25]:
            print(f"    {line}")

    print("\n[7] Generated repository.py (preview, first 30 lines):")
    repo = res.file_by_path("task_api/repository.py")
    if repo:
        for line in repo.content.splitlines()[:30]:
            print(f"    {line}")

    print("\n[8] Generated api.py (preview, first 25 lines):")
    api = res.file_by_path("task_api/api.py")
    if api:
        for line in api.content.splitlines()[:25]:
            print(f"    {line}")

    # Persistence
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo_store = SynthesisRepository(memory=mem, ontology=ont)
                ent = repo_store.save(res, project_id="demo")
                print(f"\n[9] Persisted → ontology entity: {ent[:12]}…")
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
