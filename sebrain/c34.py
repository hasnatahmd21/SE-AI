"""C34 — Knowledge Fabric Loader.

Loads the repository's intended Knowledge Fabric datasets into the Brain's shared
SQLite storage. The current repository export uses grouped, extensionless
files containing one or more JSON documents; canonical .json/.jsonl files
are also supported.

The loader is deterministic, idempotent, provenance-aware, and keeps an audit
trail for duplicate/conflicting record identities.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_DATASET_ID_RE = re.compile(r"^D(\d{1,2})$", re.IGNORECASE)
_DATASET_RANGE_RE = re.compile(r"^D\s*(\d{1,2})\s*-\s*D\s*(\d{1,2})$", re.IGNORECASE)
_RECORD_DATASET_RE = re.compile(r"^(D\d{1,2})(?:-|$)", re.IGNORECASE)
# D26 is intentionally not part of the repository dataset plan.  The grouped
# D26-D30 export contains D27-D30 records, so coverage must not treat D26 as
# missing merely because the range filename begins at D26.
_EXPECTED_DATASETS = tuple(
    f"D{i:02d}" for i in range(1, 59) if i != 26
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(slots=True)
class FabricRecord:
    record_id: str
    dataset_id: str
    topic: str = ""
    concept: str = ""
    knowledge_type: str = ""
    question: str = ""
    answer: str = ""
    explanation: str = ""
    language: str = ""
    framework: str = ""
    version: str = ""
    tags: list[str] = field(default_factory=list)
    source_file: str = ""
    source_line: int | None = None
    content_hash: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "topic": self.topic,
            "concept": self.concept,
            "knowledge_type": self.knowledge_type,
            "question": self.question,
            "answer": self.answer,
            "explanation": self.explanation,
            "language": self.language,
            "framework": self.framework,
            "version": self.version,
            "tags": list(self.tags),
            "source_file": self.source_file,
            "source_line": self.source_line,
            "content_hash": self.content_hash,
        }


class KnowledgeFabricLoader:
    """Persistent Knowledge Fabric loader backed by the Brain's Storage interface."""

    def __init__(self, storage: Any, datasets_dir: str | Path):
        self.storage = storage
        self.datasets_dir = Path(datasets_dir)
        # Direct loader users may provide a fresh SQLiteStorage. Initialize it
        # before schema creation so the loader has a valid persistence layer.
        initialize = getattr(self.storage, "initialize", None)
        if callable(initialize):
            initialize()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.storage.execute("""
            CREATE TABLE IF NOT EXISTS fabric_records (
                record_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                topic TEXT, concept TEXT, knowledge_type TEXT,
                question TEXT, answer TEXT, explanation TEXT,
                language TEXT, framework TEXT, version TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                raw_json TEXT NOT NULL,
                source_file TEXT,
                source_line INTEGER,
                content_hash TEXT,
                search_text TEXT NOT NULL DEFAULT '',
                loaded_at TEXT NOT NULL
            );
        """)
        existing = {
            r["name"] for r in self.storage.query("PRAGMA table_info(fabric_records)")
        }
        required = {
            "source_file": "TEXT",
            "source_line": "INTEGER",
            "content_hash": "TEXT",
            "search_text": "TEXT NOT NULL DEFAULT ''",
        }
        for column, sql_type in required.items():
            if column not in existing:
                self.storage.execute(
                    f"ALTER TABLE fabric_records ADD COLUMN {column} {sql_type}"
                )

        for name, column in (
            ("idx_fabric_dataset", "dataset_id"),
            ("idx_fabric_language", "language"),
            ("idx_fabric_concept", "concept"),
            ("idx_fabric_hash", "content_hash"),
        ):
            self.storage.execute(
                f"CREATE INDEX IF NOT EXISTS {name} ON fabric_records({column});"
            )

        self.storage.execute("""
            CREATE TABLE IF NOT EXISTS fabric_progress (
                dataset_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                total_lines INTEGER NOT NULL DEFAULT 0,
                valid_records INTEGER NOT NULL DEFAULT 0,
                loaded INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
        """)

        progress_existing = {
            r["name"] for r in self.storage.query("PRAGMA table_info(fabric_progress)")
        }
        progress_required = {
            "source_file": "TEXT DEFAULT ''",
            "total_lines": "INTEGER DEFAULT 0",
            "valid_records": "INTEGER DEFAULT 0",
            "loaded": "INTEGER DEFAULT 0",
            "errors": "INTEGER DEFAULT 0",
            "completed": "INTEGER DEFAULT 0",
            "updated_at": "TEXT DEFAULT ''",
        }
        for column, sql_type in progress_required.items():
            if column not in progress_existing:
                self.storage.execute(
                    f"ALTER TABLE fabric_progress ADD COLUMN {column} {sql_type}"
                )

        self.storage.execute("""
            CREATE TABLE IF NOT EXISTS fabric_datasets (
                dataset_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL DEFAULT '',
                dataset_version TEXT NOT NULL DEFAULT '',
                schema_version TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT '',
                authority TEXT NOT NULL DEFAULT '',
                scope TEXT NOT NULL DEFAULT '',
                scope_boundary TEXT NOT NULL DEFAULT '',
                purpose TEXT NOT NULL DEFAULT '',
                manifest_json TEXT NOT NULL DEFAULT '{}',
                source_file TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
        """)

        self.storage.execute("""
            CREATE TABLE IF NOT EXISTS fabric_sources (
                source_file TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                source_format TEXT NOT NULL,
                documents_parsed INTEGER NOT NULL DEFAULT 0,
                records_seen INTEGER NOT NULL DEFAULT 0,
                inserted INTEGER NOT NULL DEFAULT 0,
                duplicate_records INTEGER NOT NULL DEFAULT 0,
                conflicts INTEGER NOT NULL DEFAULT 0,
                warnings INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
        """)
        source_existing = {
            r["name"] for r in self.storage.query("PRAGMA table_info(fabric_sources)")
        }
        source_required = {
            "documents_parsed": "INTEGER NOT NULL DEFAULT 0",
            "conflicts": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, sql_type in source_required.items():
            if column not in source_existing:
                self.storage.execute(
                    f"ALTER TABLE fabric_sources ADD COLUMN {column} {sql_type}"
                )

        self.storage.execute("""
            CREATE TABLE IF NOT EXISTS fabric_record_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                dataset_id TEXT,
                source_file TEXT NOT NULL,
                source_line INTEGER,
                action TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        self.storage.execute(
            "CREATE INDEX IF NOT EXISTS idx_fabric_record_audit_record "
            "ON fabric_record_audit(record_id);"
        )
        self.storage.execute(
            "CREATE INDEX IF NOT EXISTS idx_fabric_record_audit_source "
            "ON fabric_record_audit(source_file);"
        )

    def _discover_files(self) -> list[Path]:
        if not self.datasets_dir.exists():
            return []
        files: list[Path] = []
        for path in self.datasets_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() in {"readme.md", "readme.txt"}:
                continue
            if path.suffix.lower() in {".json", ".jsonl"}:
                files.append(path)
                continue
            if _DATASET_RANGE_RE.fullmatch(path.name):
                files.append(path)
        return sorted(files, key=lambda p: p.as_posix().lower())

    def load_all_datasets(self) -> dict[str, Any]:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        files = self._discover_files()
        report: dict[str, Any] = {
            "files": 0,
            "records": 0,
            "errors": [],
            "warnings": [],
            "datasets": [],
            "source_formats": {},
            "expected_dataset_ids": list(_EXPECTED_DATASETS),
            "missing_dataset_ids": list(_EXPECTED_DATASETS),
        }
        seen_datasets: set[str] = set()

        for path in files:
            result = self.load_dataset_file(path)
            report["files"] += 1
            report["records"] += result["inserted"]
            report["errors"].extend(result["errors"])
            report["warnings"].extend(result["warnings"])
            report["datasets"].append(result)
            report["source_formats"][result["source_format"]] = (
                report["source_formats"].get(result["source_format"], 0) + 1
            )
            seen_datasets.update(result["datasets_found"])

        missing = sorted(set(_EXPECTED_DATASETS) - seen_datasets)
        report["found_dataset_ids"] = sorted(seen_datasets)
        report["missing_dataset_ids"] = missing
        if missing:
            report["warnings"].append({
                "warning": "expected datasets are missing from the uploaded fabric",
                "missing_dataset_ids": missing,
            })
        return report

    def load_dataset_file(self, path: str | Path) -> dict[str, Any]:
        """Load one source file atomically into the Knowledge Fabric."""
        with self.storage.transaction():
            return self._load_dataset_file(path)

    def _load_dataset_file(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            path.resolve().relative_to(self.datasets_dir.resolve())
        except ValueError as exc:
            raise ValueError("dataset file must be inside datasets_dir") from exc

        text = path.read_text(encoding="utf-8")
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (
            source_format,
            documents,
            parse_warnings,
            parse_errors,
        ) = self._parse_source(text, path)

        errors = list(parse_errors)
        warnings = list(parse_warnings)
        seen_ids: set[str] = set()
        inserted = 0
        duplicates = 0
        conflicts = 0
        valid = 0
        per_dataset: dict[str, dict[str, int]] = {}
        dataset_contexts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

        for document in documents:
            context = document.get("context", {})
            manifest = document.get("manifest", {})
            for record_entry in document.get("records", []):
                raw = record_entry["raw"]
                source_line = record_entry.get("source_line")
                source_line_text = record_entry.get("source_line_text", "")
                dataset_id = self._resolve_dataset_id(
                    raw, context, manifest, path.name
                )
                if not dataset_id:
                    errors.append({
                        "file": path.name,
                        "line": source_line,
                        "record_id": str(raw.get("record_id") or raw.get("id") or ""),
                        "dataset_id": raw.get("dataset_id"),
                        "dataset": raw.get("dataset"),
                        "context_dataset_id": context.get("dataset_id"),
                        "manifest_dataset_id": manifest.get("dataset_id"),
                        "error": "unable to resolve dataset_id for record",
                    })
                    continue

                # The repository plan is authoritative: D26 is intentionally
                # absent, and unknown dataset IDs must never silently enter
                # the shared fabric and distort coverage statistics.
                if dataset_id not in _EXPECTED_DATASETS:
                    errors.append({
                        "file": path.name,
                        "line": source_line,
                        "record_id": str(raw.get("record_id") or raw.get("id") or ""),
                        "dataset_id": dataset_id,
                        "error": "dataset_id is outside the planned Knowledge Fabric scope",
                    })
                    continue

                record_id = str(raw.get("record_id") or raw.get("id") or "").strip()
                if not record_id:
                    record_id = self._hash_id(
                        dataset_id,
                        int(source_line or (valid + 1)),
                        source_line_text or _canonical_json(raw),
                    )

                if record_id in seen_ids:
                    duplicates += 1
                    with self.storage.transaction():
                        self._audit_occurrence(
                            record_id,
                            dataset_id,
                            path.name,
                            source_line,
                            "duplicate_in_source",
                            self._content_hash(raw),
                            raw,
                        )
                    continue
                seen_ids.add(record_id)
                valid += 1

                # Record insertion, occurrence audit, and catalog metadata
                # form one atomic unit. A failure in any part must not leave
                # an apparently loaded record without its audit trail.
                with self.storage.transaction():
                    status = self._insert_record(
                        dataset_id=dataset_id,
                        record_id=record_id,
                        raw=raw,
                        source_file=path.name,
                        source_line=source_line,
                        source_line_text=source_line_text or _canonical_json(raw),
                    )
                    self._audit_occurrence(
                        record_id,
                        dataset_id,
                        path.name,
                        source_line,
                        status,
                        self._content_hash(raw),
                        raw,
                    )
                    self._upsert_dataset_catalog(
                        dataset_id,
                        raw,
                        context,
                        manifest,
                        path.name,
                    )

                if status == "inserted":
                    inserted += 1
                elif status == "duplicate_same":
                    duplicates += 1
                elif status == "duplicate_conflict":
                    conflicts += 1
                    warnings.append({
                        "file": path.name,
                        "line": source_line,
                        "warning": f"record_id exists with different content; canonical record retained: {record_id}",
                    })

                counts = per_dataset.setdefault(
                    dataset_id,
                    {"seen": 0, "inserted": 0, "duplicates": 0, "conflicts": 0, "errors": 0},
                )
                counts["seen"] += 1
                if status == "inserted":
                    counts["inserted"] += 1
                elif status == "duplicate_same":
                    counts["duplicates"] += 1
                elif status == "duplicate_conflict":
                    counts["conflicts"] += 1

        error_count = len(errors)
        self.storage.execute(
            """
            INSERT INTO fabric_sources
              (source_file, source_hash, source_format, documents_parsed,
               records_seen, inserted, duplicate_records, conflicts,
               warnings, errors, completed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_file) DO UPDATE SET
              source_hash=excluded.source_hash,
              source_format=excluded.source_format,
              documents_parsed=excluded.documents_parsed,
              records_seen=excluded.records_seen,
              inserted=excluded.inserted,
              duplicate_records=excluded.duplicate_records,
              conflicts=excluded.conflicts,
              warnings=excluded.warnings,
              errors=excluded.errors,
              completed=excluded.completed,
              updated_at=excluded.updated_at;
            """,
            (
                path.name,
                source_hash,
                source_format,
                len(documents),
                valid,
                inserted,
                duplicates,
                conflicts,
                len(warnings),
                error_count,
                int(error_count == 0),
                now_iso(),
            ),
        )

        for dataset_id, counts in per_dataset.items():
            total = self._count_dataset(dataset_id)
            self.storage.execute(
                """
                INSERT INTO fabric_progress
                  (dataset_id, source_file, total_lines, valid_records, loaded,
                   errors, completed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                  source_file=excluded.source_file,
                  total_lines=excluded.total_lines,
                  valid_records=excluded.valid_records,
                  loaded=excluded.loaded,
                  errors=excluded.errors,
                  completed=excluded.completed,
                  updated_at=excluded.updated_at;
                """,
                (
                    dataset_id,
                    path.name,
                    counts["seen"],
                    counts["seen"] - counts["errors"],
                    total,
                    counts["errors"],
                    int(counts["errors"] == 0 and error_count == 0),
                    now_iso(),
                ),
            )

        return {
            "file": path.name,
            "source_format": source_format,
            "documents_parsed": len(documents),
            "datasets_found": sorted(per_dataset),
            "lines": valid,
            "valid": valid,
            "inserted": inserted,
            "duplicates": duplicates,
            "conflicts": conflicts,
            "loaded_total": sum(
                self._count_dataset(dataset_id) for dataset_id in per_dataset
            ),
            "errors": errors,
            "warnings": warnings,
            "completed": error_count == 0,
        }

    @staticmethod
    def _repair_unescaped_json_quotes(line: str) -> str | None:
        """Repair only provable quote defects inside JSON string values."""
        if not line.lstrip().startswith("{") or '"record_id"' not in line:
            return None

        chars: list[str] = []
        stack: list[str] = []
        in_string = False
        escaped = False
        changed = False

        def next_non_space(index: int) -> int:
            while index < len(line) and line[index].isspace():
                index += 1
            return index

        def quoted_token_end(index: int) -> int | None:
            escaped_token = False
            i = index + 1
            while i < len(line):
                char = line[i]
                if escaped_token:
                    escaped_token = False
                elif char == "\\":
                    escaped_token = True
                elif char == '"':
                    return i
                i += 1
            return None

        def comma_is_structural(index: int) -> bool:
            next_index = next_non_space(index + 1)
            if next_index >= len(line):
                return False
            next_char = line[next_index]
            container = stack[-1] if stack else "object"

            if next_char == '"':
                end_index = quoted_token_end(next_index)
                if end_index is None:
                    return False
                after = next_non_space(end_index + 1)
                if container == "object":
                    return after < len(line) and line[after] == ":"
                return True

            if container == "object":
                # An object boundary after a value must begin the next key.
                return False

            return (
                next_char in {"{", "[", "-"}
                or next_char.isdigit()
                or next_char in {"t", "f", "n"}
            )

        for index, char in enumerate(line):
            if not in_string:
                chars.append(char)
                if char == '"':
                    in_string = True
                    escaped = False
                elif char in "{[":
                    stack.append("object" if char == "{" else "array")
                elif char in "}]":
                    expected = "object" if char == "}" else "array"
                    if stack and stack[-1] == expected:
                        stack.pop()
                continue

            if escaped:
                chars.append(char)
                escaped = False
                continue
            if char == "\\":
                chars.append(char)
                escaped = True
                continue
            if char != '"':
                chars.append(char)
                continue

            next_index = next_non_space(index + 1)
            next_char = line[next_index] if next_index < len(line) else ""
            container = stack[-1] if stack else "object"
            closes = (
                next_char in {":", ""}
                or (next_char == "}" and container == "object")
                or (next_char == "]" and container == "array")
                or (next_char == "," and comma_is_structural(next_index))
            )
            if closes:
                chars.append('"')
                in_string = False
            else:
                chars.append('\"')
                changed = True

        if not changed or in_string:
            return None

        repaired = "".join(chars)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or not (
            payload.get("record_id") or payload.get("id")
        ):
            return None
        return repaired

    @staticmethod
    def _repair_common_json_defects(line: str) -> str | None:
        """Apply only narrowly scoped, deterministic export repairs."""
        if not line.lstrip().startswith("{") or '"record_id"' not in line:
            return None

        candidates: list[str] = []
        # Conservative recovery for a malformed final string field whose
        # value contains raw double quotes (common in exported code examples).
        field_marker = '":"'
        field_start = line.rfind(field_marker)
        value_end = line.rfind('"}')
        if field_start >= 0 and value_end > field_start + len(field_marker):
            value_start = field_start + len(field_marker)
            value = line[value_start:value_end]
            escaped_value = value.replace('"', '\\"')
            candidate = line[:value_start] + escaped_value + line[value_end:]
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and (payload.get("record_id") or payload.get("id")):
                candidates.append(candidate)


        if '"language_agnostic","' in line:
            candidates.append(
                line.replace('"language_agnostic","', '"language_agnostic":true,"')
            )

        if ',"]' in line:
            candidates.append(line.replace(',"]', ']'))

        if ",}" in line:
            candidates.append(line.replace(",}", "}"))

        if ",]" in line:
            candidates.append(line.replace(",]", "]"))

        completed = KnowledgeFabricLoader._balanced_delimiter_completion(line)
        if completed is not None:
            candidates.append(completed)

        quote_repaired = KnowledgeFabricLoader._repair_unescaped_json_quotes(line)
        if quote_repaired is not None:
            candidates.append(quote_repaired)

            if ',"]' in quote_repaired:
                candidates.append(quote_repaired.replace(',"]', ']'))
            if ",}" in quote_repaired:
                candidates.append(quote_repaired.replace(",}", "}"))
            completed = KnowledgeFabricLoader._balanced_delimiter_completion(
                quote_repaired
            )
            if completed is not None:
                candidates.append(completed)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and (
                payload.get("record_id") or payload.get("id")
            ):
                return candidate

        return None

    @staticmethod
    def _balanced_delimiter_completion(line: str) -> str | None:
        """Append only objectively missing closing JSON delimiters."""
        in_string = False
        escaped = False
        stack: list[str] = []

        for char in line:
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not stack or stack[-1] != expected:
                    return None
                stack.pop()

        if in_string or not stack:
            return None

        completed = line
        while stack:
            opening = stack.pop()
            completed += "}" if opening == "{" else "]"
        return completed

    @staticmethod
    def _parse_source(
        text: str, path: Path
    ) -> tuple[
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Parse grouped Knowledge Fabric exports.

        Dataset files are intentionally human-readable exports: explanatory
        prose/markdown may surround JSON records. The parser therefore scans
        the entire byte/text stream for complete JSON values instead of
        assuming every JSON value begins at a physical line boundary.
        """

        if not text.strip():
            return "empty", [], [], []

        decoder = json.JSONDecoder()
        documents: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        i = 0
        json_values = 0
        skipped_regions = 0

        while i < len(text):
            # Find the next possible JSON value. This deliberately skips
            # prose, headings, code fences, and other human-readable text.
            next_obj = text.find("{", i)
            next_arr = text.find("[", i)
            candidates = [p for p in (next_obj, next_arr) if p >= 0]
            if not candidates:
                if text[i:].strip():
                    skipped_regions += 1
                break

            start = min(candidates)
            if text[i:start].strip():
                skipped_regions += 1

            try:
                payload, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError as exc:
                # Before discarding a malformed line, apply the loader's
                # deliberately narrow quote-repair rule to record objects.
                # This preserves the ability to recover generated code/prose
                # containing raw double quotes without performing broad or
                # unsafe structural repair.
                line_end = text.find("\n", start)
                if line_end < 0:
                    line_end = len(text)
                candidate_line = text[start:line_end]
                repaired = KnowledgeFabricLoader._repair_common_json_defects(
                    candidate_line
                )
                if repaired is not None:
                    try:
                        payload = json.loads(repaired)
                    except json.JSONDecodeError:
                        payload = None
                    if payload is not None:
                        json_values += 1
                        warnings.append({
                            "file": path.name,
                            "line": text.count("\n", 0, start) + 1,
                            "warning": "repaired narrowly scoped JSON string quoting in records",
                        })
                        line_no = text.count("\n", 0, start) + 1
                        document = KnowledgeFabricLoader._document_from_payload(
                            payload, line_no
                        )
                        if document is not None:
                            documents.append(document)
                        i = line_end
                        continue

                # Malformed JSON that is clearly not a record is treated as
                # export metadata/prose. A malformed candidate containing a
                # record identity must remain an error because dropping it
                # would hide a dataset record failure.
                line_no = text.count("\\n", 0, start) + 1
                if '"record_id"' in candidate_line or '"id"' in candidate_line:
                    errors.append({
                        "file": path.name,
                        "line": line_no,
                        "error": "invalid JSON value",
                        "detail": str(getattr(exc, "msg", "") or "JSON decode failed"),
                    })
                else:
                    warnings.append({
                        "file": path.name,
                        "line": line_no,
                        "warning": "ignored malformed non-record JSON metadata region",
                    })
                # Do not scan nested braces inside the malformed value as if
                # they were independent top-level records; that can fabricate
                # records from fields such as test_input or code examples.
                i = line_end
                continue

            json_values += 1
            line_no = text.count("\n", 0, start) + 1
            document = KnowledgeFabricLoader._document_from_payload(
                payload, line_no
            )
            if document is not None:
                documents.append(document)
            i = end

        if skipped_regions:
            warnings.append({
                "file": path.name,
                "warning": "ignored non-JSON explanatory/header regions",
                "region_count": skipped_regions,
            })

        if json_values == 0:
            errors.append({
                "file": path.name,
                "line": 1,
                "error": "no JSON dataset records/documents were found",
            })
        elif not documents:
            errors.append({
                "file": path.name,
                "line": 1,
                "error": "JSON values were found but none contained dataset records",
            })

        return "json_document_stream", documents, warnings, errors

    @staticmethod
    def _document_from_payload(
        payload: Any, source_line: int
    ) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            manifest = payload.get("dataset_manifest")
            if not isinstance(manifest, dict):
                manifest = {}

            records: list[dict[str, Any]] = []
            if isinstance(payload.get("records"), list):
                for item in payload["records"]:
                    if isinstance(item, dict):
                        records.append({
                            "raw": item,
                            "source_line": source_line,
                            "source_line_text": _canonical_json(item),
                        })
            elif payload.get("record_id") or payload.get("id"):
                records.append({
                    "raw": payload,
                    "source_line": source_line,
                    "source_line_text": _canonical_json(payload),
                })
            else:
                return None

            context = dict(payload)
            return {
                "context": context,
                "manifest": manifest,
                "records": records,
            }

        if isinstance(payload, list):
            records = [
                {
                    "raw": item,
                    "source_line": source_line,
                    "source_line_text": _canonical_json(item),
                }
                for item in payload
                if isinstance(item, dict)
                and KnowledgeFabricLoader._looks_like_record(item)
            ]
            if not records:
                return None
            return {"context": {}, "manifest": {}, "records": records}

        return None

    @staticmethod
    def _looks_like_record(payload: dict[str, Any]) -> bool:
        """Return True only for dictionaries that resemble dataset records."""
        if payload.get("record_id") or payload.get("id"):
            return True
        if payload.get("dataset_id") or payload.get("dataset"):
            return True
        recordish = sum(
            bool(payload.get(key))
            for key in (
                "topic",
                "concept",
                "knowledge_type",
                "question",
                "answer",
                "objective",
            )
        )
        return recordish >= 3

    @staticmethod
    def _normalise_dataset_id(value: Any) -> str | None:
        if value is None:
            return None
        match = _DATASET_ID_RE.fullmatch(str(value).strip())
        if not match:
            return None
        return f"D{int(match.group(1)):02d}"

    def _resolve_dataset_id(
        self,
        raw: dict[str, Any],
        context: dict[str, Any],
        manifest: dict[str, Any],
        source_name: str,
    ) -> str | None:
        for value in (
            raw.get("dataset_id"),
            raw.get("dataset"),
            context.get("dataset_id"),
            context.get("dataset"),
            manifest.get("dataset_id"),
        ):
            dataset_id = self._normalise_dataset_id(value)
            if dataset_id:
                return dataset_id

        record_id = str(raw.get("record_id") or raw.get("id") or "").strip()
        match = _RECORD_DATASET_RE.match(record_id)
        if match:
            dataset_id = self._normalise_dataset_id(match.group(1))
            if dataset_id:
                return dataset_id

        stem = Path(source_name).stem
        direct = self._normalise_dataset_id(stem)
        if direct:
            return direct

        range_match = _DATASET_RANGE_RE.fullmatch(source_name)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            # A range name cannot identify one dataset when it contains more
            # than one dataset, so only use this fallback for a single range.
            if start == end:
                return f"D{start:02d}"
        return None

    @staticmethod
    def _catalog_value(
        raw: dict[str, Any],
        context: dict[str, Any],
        manifest: dict[str, Any],
        key: str,
    ) -> str:
        for source in (raw, context, manifest):
            value = source.get(key)
            if value is not None:
                if isinstance(value, (dict, list)):
                    return _canonical_json(value)
                return str(value)
        return ""

    def _upsert_dataset_catalog(
        self,
        dataset_id: str,
        raw: dict[str, Any],
        context: dict[str, Any],
        manifest: dict[str, Any],
        source_file: str,
    ) -> None:
        manifest_json = manifest or {}
        self.storage.execute(
            """
            INSERT INTO fabric_datasets
              (dataset_id, dataset_name, dataset_version, schema_version, status,
               language, authority, scope, scope_boundary, purpose, manifest_json,
               source_file, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
              dataset_name=CASE WHEN excluded.dataset_name != '' THEN excluded.dataset_name ELSE fabric_datasets.dataset_name END,
              dataset_version=CASE WHEN excluded.dataset_version != '' THEN excluded.dataset_version ELSE fabric_datasets.dataset_version END,
              schema_version=CASE WHEN excluded.schema_version != '' THEN excluded.schema_version ELSE fabric_datasets.schema_version END,
              status=CASE WHEN excluded.status != '' THEN excluded.status ELSE fabric_datasets.status END,
              language=CASE WHEN excluded.language != '' THEN excluded.language ELSE fabric_datasets.language END,
              authority=CASE WHEN excluded.authority != '' THEN excluded.authority ELSE fabric_datasets.authority END,
              scope=CASE WHEN excluded.scope != '' THEN excluded.scope ELSE fabric_datasets.scope END,
              scope_boundary=CASE WHEN excluded.scope_boundary != '' THEN excluded.scope_boundary ELSE fabric_datasets.scope_boundary END,
              purpose=CASE WHEN excluded.purpose != '' THEN excluded.purpose ELSE fabric_datasets.purpose END,
              manifest_json=CASE WHEN excluded.manifest_json != '{}' THEN excluded.manifest_json ELSE fabric_datasets.manifest_json END,
              source_file=excluded.source_file,
              updated_at=excluded.updated_at;
            """,
            (
                dataset_id,
                self._catalog_value(raw, context, manifest, "dataset_name"),
                self._catalog_value(raw, context, manifest, "dataset_version"),
                self._catalog_value(raw, context, manifest, "schema_version"),
                self._catalog_value(raw, context, manifest, "status"),
                self._catalog_value(raw, context, manifest, "language"),
                self._catalog_value(raw, context, manifest, "authority"),
                self._catalog_value(raw, context, manifest, "scope"),
                self._catalog_value(raw, context, manifest, "scope_boundary"),
                self._catalog_value(raw, context, manifest, "purpose"),
                _canonical_json(manifest_json),
                source_file,
                now_iso(),
            ),
        )

    @staticmethod
    def _content_hash(raw: dict[str, Any]) -> str:
        return hashlib.sha256(_canonical_json(raw).encode("utf-8")).hexdigest()

    def _insert_record(
        self,
        *,
        dataset_id: str,
        record_id: str,
        raw: dict[str, Any],
        source_file: str,
        source_line: int | None,
        source_line_text: str,
    ) -> str:
        content_hash = self._content_hash(raw)
        existing = self.storage.query_one(
            "SELECT content_hash FROM fabric_records WHERE record_id=?",
            (record_id,),
        )
        if existing:
            return (
                "duplicate_same"
                if existing["content_hash"] == content_hash
                else "duplicate_conflict"
            )

        tags = raw.get("tags", [])
        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(tag) for tag in tags]

        search_parts: list[str] = []
        for key, value in raw.items():
            if isinstance(value, (str, int, float, bool)):
                search_parts.append(f"{key}={value}")
            elif isinstance(value, list):
                search_parts.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
        search_text = " ".join(search_parts)

        self.storage.execute(
            """
            INSERT INTO fabric_records
              (record_id, dataset_id, topic, concept, knowledge_type, question,
               answer, explanation, language, framework, version, tags_json,
               raw_json, source_file, source_line, content_hash, search_text,
               loaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record_id,
                dataset_id,
                str(raw.get("topic", "")),
                str(raw.get("concept", "")),
                str(raw.get("knowledge_type", raw.get("type", ""))),
                str(raw.get("question", "")),
                str(raw.get("answer", "")),
                str(raw.get("explanation", raw.get("principle", ""))),
                str(raw.get("language", "")),
                str(raw.get("framework", "")),
                str(raw.get("version", raw.get("version_range", ""))),
                json.dumps(tags, ensure_ascii=False, default=str),
                _canonical_json(raw),
                source_file,
                source_line,
                content_hash,
                search_text,
                now_iso(),
            ),
        )
        return "inserted"

    def _audit_occurrence(
        self,
        record_id: str,
        dataset_id: str,
        source_file: str,
        source_line: int | None,
        action: str,
        content_hash: str,
        raw: dict[str, Any],
    ) -> None:
        self.storage.execute(
            """
            INSERT INTO fabric_record_audit
              (record_id, dataset_id, source_file, source_line, action,
               content_hash, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record_id,
                dataset_id,
                source_file,
                source_line,
                action,
                content_hash,
                _canonical_json(raw),
                now_iso(),
            ),
        )

    def get_batch(
        self,
        dataset_id: str | None = None,
        *,
        batch_size: int = 200,
        offset: int = 0,
        language: str | None = None,
        concept: str | None = None,
    ) -> list[FabricRecord]:
        if batch_size <= 0 or offset < 0:
            raise ValueError("batch_size must be > 0 and offset must be >= 0")
        sql = "SELECT * FROM fabric_records WHERE 1=1"
        params: list[Any] = []
        if dataset_id:
            sql += " AND dataset_id=?"
            params.append(self._normalise_dataset_id(dataset_id) or dataset_id)
        if language:
            sql += " AND language=?"
            params.append(language)
        if concept:
            sql += " AND concept=?"
            params.append(concept)
        sql += " ORDER BY record_id LIMIT ? OFFSET ?"
        params.extend([batch_size, offset])
        return [self._row_to_record(r) for r in self.storage.query(sql, params)]

    def iterate_batches(
        self, *, batch_size: int = 200, **filters: Any
    ) -> Iterator[list[FabricRecord]]:
        offset = 0
        while True:
            batch = self.get_batch(batch_size=batch_size, offset=offset, **filters)
            if not batch:
                break
            yield batch
            offset += len(batch)
            if len(batch) < batch_size:
                break

    def search(
        self,
        query: str,
        *,
        limit: int | None = 10,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> list[FabricRecord]:
        if not query or (limit is not None and limit <= 0):
            return []
        terms = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_+#.-]+", query)
            if len(token) >= 2
        ]
        if not terms:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        fields = (
            "question", "answer", "explanation", "concept", "topic",
            "tags_json", "framework", "language", "search_text",
        )
        for term in terms:
            escaped = (
                term.replace("!", "!!")
                .replace("%", "!%")
                .replace("_", "!_")
            )
            like = f"%{escaped}%"
            clauses.append("(" + " OR ".join(
                f"LOWER({field}) LIKE ? ESCAPE '!'"
                for field in fields
            ) + ")")
            params.extend([like] * len(fields))

        sql = "SELECT * FROM fabric_records WHERE (" + " OR ".join(clauses) + ")"
        if dataset_id:
            sql += " AND dataset_id=?"
            params.append(self._normalise_dataset_id(dataset_id) or dataset_id)
        if language:
            sql += " AND language=?"
            params.append(language)
        if concept:
            sql += " AND concept=?"
            params.append(concept)
        sql += " ORDER BY record_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_record(r) for r in self.storage.query(sql, params)]

    def dataset_catalog(self) -> list[dict[str, Any]]:
        return self.storage.query(
            "SELECT * FROM fabric_datasets ORDER BY dataset_id"
        )

    def coverage_audit(self) -> dict[str, Any]:
        present_rows = self.storage.query(
            "SELECT dataset_id, COUNT(*) AS c FROM fabric_records "
            "GROUP BY dataset_id ORDER BY dataset_id"
        )
        present = {row["dataset_id"]: int(row["c"]) for row in present_rows}
        missing = sorted(set(_EXPECTED_DATASETS) - set(present))
        return {
            "expected": list(_EXPECTED_DATASETS),
            "present": sorted(present),
            "missing": missing,
            "dataset_count": len(present),
            "complete": not missing,
            "record_counts": present,
        }

    def stats(self) -> dict[str, Any]:
        total = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM fabric_records"
        )
        datasets = self.storage.query(
            "SELECT dataset_id, COUNT(*) AS c FROM fabric_records "
            "GROUP BY dataset_id ORDER BY dataset_id"
        )
        langs = self.storage.query(
            "SELECT language, COUNT(*) AS c FROM fabric_records "
            "WHERE language != '' GROUP BY language ORDER BY c DESC"
        )
        progress = self.storage.query(
            "SELECT dataset_id, total_lines, valid_records, loaded, errors, completed "
            "FROM fabric_progress ORDER BY dataset_id"
        )
        sources = self.storage.query(
            "SELECT source_file, source_format, documents_parsed, records_seen, "
            "inserted, duplicate_records, conflicts, warnings, errors, completed "
            "FROM fabric_sources ORDER BY source_file"
        )
        return {
            "total_records": int(total["c"]) if total else 0,
            "datasets": {r["dataset_id"]: int(r["c"]) for r in datasets},
            "languages": {r["language"]: int(r["c"]) for r in langs},
            "dataset_count": len(datasets),
            "dataset_catalog": self.dataset_catalog(),
            "coverage": self.coverage_audit(),
            "progress": progress,
            "sources": sources,
        }

    def _count_dataset(self, dataset_id: str) -> int:
        row = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM fabric_records WHERE dataset_id=?",
            (dataset_id,),
        )
        return int(row["c"]) if row else 0

    @staticmethod
    def _hash_id(dataset_id: str, line_num: int, content: str) -> str:
        digest = hashlib.sha256(
            f"{dataset_id}:{line_num}:{content}".encode("utf-8")
        ).hexdigest()[:16]
        return f"{dataset_id}-{digest}"

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> FabricRecord:
        try:
            tags = json.loads(row.get("tags_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            tags = []
        try:
            raw = json.loads(row.get("raw_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            raw = {}
        return FabricRecord(
            record_id=row["record_id"],
            dataset_id=row["dataset_id"],
            topic=row.get("topic") or "",
            concept=row.get("concept") or "",
            knowledge_type=row.get("knowledge_type") or "",
            question=row.get("question") or "",
            answer=row.get("answer") or "",
            explanation=row.get("explanation") or "",
            language=row.get("language") or "",
            framework=row.get("framework") or "",
            version=row.get("version") or "",
            tags=tags if isinstance(tags, list) else [],
            source_file=row.get("source_file") or "",
            source_line=row.get("source_line"),
            content_hash=row.get("content_hash") or "",
            raw=raw if isinstance(raw, dict) else {},
        )
