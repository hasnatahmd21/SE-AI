"""C34 — Knowledge Fabric Loader.

Loads the repository's D01-D58 Knowledge Fabric into the Brain's shared
SQLite storage. Sources may be canonical JSONL files or the grouped,
extensionless dataset files used by the current repository export.

The loader is deterministic, idempotent, provenance-aware and tolerant of
non-data preamble lines that appear in grouped exports. Record-level raw data
is preserved so later engines can use fields beyond the normalized retrieval
columns.
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
_DATASET_RANGE_RE = re.compile(r"^D(\d{1,2})\s*-\s*D(\d{1,2})$", re.IGNORECASE)
_RECORD_DATASET_RE = re.compile(r"^(D\d{1,2})(?:-|$)", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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
        }


class KnowledgeFabricLoader:
    """Persistent D01-D58 loader backed by the Brain's Storage interface."""

    def __init__(self, storage: Any, datasets_dir: str | Path):
        self.storage = storage
        self.datasets_dir = Path(datasets_dir)
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
                records_seen INTEGER NOT NULL DEFAULT 0,
                inserted INTEGER NOT NULL DEFAULT 0,
                duplicate_records INTEGER NOT NULL DEFAULT 0,
                warnings INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
        """)

    def _discover_files(self) -> list[Path]:
        """Discover canonical JSON/JSONL and the current grouped D-range exports."""
        if not self.datasets_dir.exists():
            return []
        found: list[Path] = []
        for path in self.datasets_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() in {"readme.md", "readme.txt"}:
                continue
            suffix = path.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                found.append(path)
                continue
            if _DATASET_RANGE_RE.fullmatch(path.name):
                found.append(path)
        return sorted(found, key=lambda p: p.as_posix().lower())

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
        }
        for path in files:
            result = self.load_dataset_file(path)
            report["files"] += 1
            report["records"] += result["inserted"]
            report["errors"].extend(result["errors"])
            report["warnings"].extend(result["warnings"])
            report["datasets"].append(result)
            fmt = result["source_format"]
            report["source_formats"][fmt] = report["source_formats"].get(fmt, 0) + 1
        return report

    def load_dataset_file(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            path.resolve().relative_to(self.datasets_dir.resolve())
        except ValueError as exc:
            raise ValueError("dataset file must be inside datasets_dir") from exc

        text = path.read_text(encoding="utf-8")
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_format, manifest, entries, parse_warnings, parse_errors = self._parse_source(
            text, path
        )

        issues: list[dict[str, Any]] = list(parse_errors)
        warnings: list[dict[str, Any]] = list(parse_warnings)
        seen_ids: set[str] = set()
        inserted = 0
        duplicates = 0
        valid = 0
        per_dataset: dict[str, dict[str, int]] = {}

        for entry in entries:
            raw = entry["raw"]
            source_line = entry.get("source_line")
            source_line_text = entry.get("source_line_text", "")
            dataset_id = self._resolve_dataset_id(raw, manifest, path.name)
            if not dataset_id:
                issues.append({
                    "file": path.name,
                    "line": source_line,
                    "error": "unable to resolve dataset_id for record",
                })
                continue

            record_id = str(raw.get("record_id") or raw.get("id") or "").strip()
            if not record_id:
                record_id = self._hash_id(
                    dataset_id,
                    int(source_line or (valid + 1)),
                    source_line_text or json.dumps(raw, ensure_ascii=False, sort_keys=True),
                )
            if record_id in seen_ids:
                duplicates += 1
                issues.append({
                    "file": path.name,
                    "line": source_line,
                    "error": f"duplicate record_id in source: {record_id}",
                })
                continue
            seen_ids.add(record_id)
            valid += 1

            status = self._insert_record(
                dataset_id,
                record_id,
                raw,
                path.name,
                source_line,
                source_line_text or json.dumps(raw, ensure_ascii=False, sort_keys=True),
            )
            if status == "inserted":
                inserted += 1
            elif status == "duplicate_same":
                duplicates += 1
            else:
                issues.append({
                    "file": path.name,
                    "line": source_line,
                    "error": f"record_id already exists with different content: {record_id}",
                })

            counts = per_dataset.setdefault(
                dataset_id, {"seen": 0, "inserted": 0, "duplicates": 0, "errors": 0}
            )
            counts["seen"] += 1
            if status == "inserted":
                counts["inserted"] += 1
            elif status == "duplicate_same":
                counts["duplicates"] += 1
            elif status == "duplicate_conflict":
                counts["errors"] += 1

            self._upsert_dataset_catalog(
                dataset_id,
                raw,
                manifest,
                path.name,
            )

        error_count = len(issues)
        self.storage.execute(
            """
            INSERT INTO fabric_sources
              (source_file, source_hash, source_format, records_seen, inserted,
               duplicate_records, warnings, errors, completed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_file) DO UPDATE SET
              source_hash=excluded.source_hash,
              source_format=excluded.source_format,
              records_seen=excluded.records_seen,
              inserted=excluded.inserted,
              duplicate_records=excluded.duplicate_records,
              warnings=excluded.warnings,
              errors=excluded.errors,
              completed=excluded.completed,
              updated_at=excluded.updated_at;
            """,
            (
                path.name,
                source_hash,
                source_format,
                len(entries),
                inserted,
                duplicates,
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
                    int(counts["errors"] == 0),
                    now_iso(),
                ),
            )

        return {
            "file": path.name,
            "source_format": source_format,
            "datasets_found": sorted(per_dataset),
            "lines": len(entries),
            "valid": valid,
            "inserted": inserted,
            "duplicates": duplicates,
            "loaded_total": sum(self._count_dataset(d) for d in per_dataset),
            "errors": issues,
            "warnings": warnings,
            "completed": error_count == 0,
        }

    def _parse_source(
        self, text: str, path: Path
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        stripped = text.lstrip()
        if not stripped:
            return "empty", {}, [], [], []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return (*self._parse_jsonl_like(text, path),)

        if isinstance(payload, dict):
            manifest = payload.get("dataset_manifest")
            if not isinstance(manifest, dict):
                manifest = {}
            if isinstance(payload.get("records"), list):
                entries = [
                    {
                        "raw": raw,
                        "source_line": index + 1,
                        "source_line_text": json.dumps(raw, ensure_ascii=False, sort_keys=True),
                    }
                    for index, raw in enumerate(payload["records"])
                    if isinstance(raw, dict)
                ]
                invalid = [
                    index + 1
                    for index, raw in enumerate(payload["records"])
                    if not isinstance(raw, dict)
                ]
                errors = [
                    {"file": path.name, "line": line, "error": "record must be a JSON object"}
                    for line in invalid
                ]
                return "json_manifest", manifest, entries, [], errors
            if payload.get("record_id") or payload.get("id"):
                return (
                    "json_record",
                    manifest,
                    [{
                        "raw": payload,
                        "source_line": 1,
                        "source_line_text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    }],
                    [],
                    [],
                )
            return (
                "json_metadata",
                manifest,
                [],
                [],
                [{"file": path.name, "line": 1, "error": "JSON object has no records"}],
            )

        if isinstance(payload, list):
            entries = [
                {
                    "raw": raw,
                    "source_line": index + 1,
                    "source_line_text": json.dumps(raw, ensure_ascii=False, sort_keys=True),
                }
                for index, raw in enumerate(payload)
                if isinstance(raw, dict)
            ]
            errors = [
                {"file": path.name, "line": index + 1, "error": "record must be a JSON object"}
                for index, raw in enumerate(payload)
                if not isinstance(raw, dict)
            ]
            return "json_array", {}, entries, [], errors

        return (
            "json_invalid",
            {},
            [],
            [],
            [{"file": path.name, "line": 1, "error": "top-level JSON value is unsupported"}],
        )

    @staticmethod
    def _parse_jsonl_like(
        text: str, path: Path
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        entries: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for line_num, raw_line in enumerate(text.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                # Current grouped exports contain human-readable batch headers.
                warnings.append({
                    "file": path.name,
                    "line": line_num,
                    "warning": "ignored non-JSON preamble/header line",
                })
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({
                    "file": path.name,
                    "line": line_num,
                    "error": f"invalid JSON: {exc.msg}",
                })
                continue
            if not isinstance(raw, dict):
                errors.append({
                    "file": path.name,
                    "line": line_num,
                    "error": "record must be a JSON object",
                })
                continue
            entries.append({
                "raw": raw,
                "source_line": line_num,
                "source_line_text": line,
            })
        return "jsonl_or_grouped", {}, entries, warnings, errors

    @staticmethod
    def _normalise_dataset_id(value: Any) -> str | None:
        if value is None:
            return None
        match = _DATASET_ID_RE.fullmatch(str(value).strip())
        if not match:
            return None
        return f"D{int(match.group(1)):02d}"

    def _resolve_dataset_id(
        self, raw: dict[str, Any], manifest: dict[str, Any], source_name: str
    ) -> str | None:
        for value in (
            raw.get("dataset_id"),
            raw.get("dataset"),
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

        range_match = _DATASET_RANGE_RE.fullmatch(source_name)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start == end:
                return f"D{start:02d}"
        return None

    @staticmethod
    def _catalog_value(raw: dict[str, Any], manifest: dict[str, Any], key: str) -> str:
        for source in (raw, manifest):
            value = source.get(key)
            if value is not None:
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False, sort_keys=True)
                return str(value)
        return ""

    def _upsert_dataset_catalog(
        self, dataset_id: str, raw: dict[str, Any], manifest: dict[str, Any], source_file: str
    ) -> None:
        manifest_json = manifest if manifest else {}
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
                self._catalog_value(raw, manifest, "dataset_name"),
                self._catalog_value(raw, manifest, "dataset_version"),
                self._catalog_value(raw, manifest, "schema_version"),
                self._catalog_value(raw, manifest, "status"),
                self._catalog_value(raw, manifest, "language"),
                self._catalog_value(raw, manifest, "authority"),
                self._catalog_value(raw, manifest, "scope"),
                self._catalog_value(raw, manifest, "scope_boundary"),
                self._catalog_value(raw, manifest, "purpose"),
                json.dumps(manifest_json, ensure_ascii=False, sort_keys=True),
                source_file,
                now_iso(),
            ),
        )

    def _insert_record(
        self,
        dataset_id: str,
        record_id: str,
        raw: dict[str, Any],
        source_file: str,
        source_line: int | None,
        source_line_text: str,
    ) -> str:
        content_hash = hashlib.sha256(source_line_text.encode("utf-8")).hexdigest()
        existing = self.storage.query_one(
            "SELECT content_hash FROM fabric_records WHERE record_id=?",
            (record_id,),
        )
        if existing:
            return "duplicate_same" if existing["content_hash"] == content_hash else "duplicate_conflict"

        tags = raw.get("tags", [])
        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(x) for x in tags]

        self.storage.execute(
            """
            INSERT INTO fabric_records
              (record_id, dataset_id, topic, concept, knowledge_type, question,
               answer, explanation, language, framework, version, tags_json,
               raw_json, source_file, source_line, content_hash, loaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                json.dumps(tags, ensure_ascii=False),
                json.dumps(raw, ensure_ascii=False, default=str),
                source_file,
                source_line,
                content_hash,
                now_iso(),
            ),
        )
        return "inserted"

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
            params.append(dataset_id)
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
        limit: int = 10,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> list[FabricRecord]:
        if not query or limit <= 0:
            return []
        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_+#.-]+", query) if len(t) >= 2]
        if not terms:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        for term in terms:
            escaped = (
                term.replace("\\", "\\\\")
                .replace("%", "\%")
                .replace("_", "\_")
            )
            like = f"%{escaped}%"
            clauses.append(
                "(LOWER(question) LIKE ? ESCAPE '\\' OR "
                "LOWER(answer) LIKE ? ESCAPE '\\' OR "
                "LOWER(explanation) LIKE ? ESCAPE '\\' OR "
                "LOWER(concept) LIKE ? ESCAPE '\\' OR "
                "LOWER(topic) LIKE ? ESCAPE '\\' OR "
                "LOWER(tags_json) LIKE ? ESCAPE '\\' OR "
                "LOWER(framework) LIKE ? ESCAPE '\\' OR "
                "LOWER(language) LIKE ? ESCAPE '\\')"
            )
            params.extend([like] * 8)

        sql = "SELECT * FROM fabric_records WHERE (" + " OR ".join(clauses) + ")"
        if dataset_id:
            sql += " AND dataset_id=?"
            params.append(dataset_id)
        if language:
            sql += " AND language=?"
            params.append(language)
        if concept:
            sql += " AND concept=?"
            params.append(concept)
        sql += " ORDER BY record_id LIMIT ?"
        params.append(limit)
        return [self._row_to_record(r) for r in self.storage.query(sql, params)]

    def dataset_catalog(self) -> list[dict[str, Any]]:
        return self.storage.query(
            "SELECT * FROM fabric_datasets ORDER BY dataset_id"
        )

    def stats(self) -> dict[str, Any]:
        total = self.storage.query_one("SELECT COUNT(*) AS c FROM fabric_records")
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
            "SELECT source_file, source_format, records_seen, inserted, "
            "duplicate_records, warnings, errors, completed "
            "FROM fabric_sources ORDER BY source_file"
        )
        catalog = self.dataset_catalog()
        return {
            "total_records": int(total["c"]) if total else 0,
            "datasets": {r["dataset_id"]: int(r["c"]) for r in datasets},
            "languages": {r["language"]: int(r["c"]) for r in langs},
            "dataset_count": len(catalog),
            "dataset_catalog": catalog,
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
            raw=raw if isinstance(raw, dict) else {},
        )
