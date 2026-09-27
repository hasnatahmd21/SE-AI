"""C34 — Knowledge Fabric Loader.

Loads D01-D58 JSONL knowledge records into the Brain's shared SQLite storage.
The loader is deterministic, idempotent, transactional per file, and keeps
raw source records for provenance/audit purposes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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
            "record_id": self.record_id, "dataset_id": self.dataset_id,
            "topic": self.topic, "concept": self.concept,
            "knowledge_type": self.knowledge_type, "question": self.question,
            "answer": self.answer, "explanation": self.explanation,
            "language": self.language, "framework": self.framework,
            "version": self.version, "tags": list(self.tags),
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
        # Upgrade databases created by the earlier C34 draft without destroying data.
        existing = {r["name"] for r in self.storage.query("PRAGMA table_info(fabric_records)")}
        required = {
            "source_file": "TEXT", "source_line": "INTEGER",
            "content_hash": "TEXT",
        }
        for column, sql_type in required.items():
            if column not in existing:
                self.storage.execute(f"ALTER TABLE fabric_records ADD COLUMN {column} {sql_type}")
        for name, column in (
            ("idx_fabric_dataset", "dataset_id"),
            ("idx_fabric_language", "language"),
            ("idx_fabric_concept", "concept"),
            ("idx_fabric_hash", "content_hash"),
        ):
            self.storage.execute(f"CREATE INDEX IF NOT EXISTS {name} ON fabric_records({column});")
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
        progress_existing = {r["name"] for r in self.storage.query("PRAGMA table_info(fabric_progress)")}
        progress_required = {
            "source_file": "TEXT DEFAULT ''", "total_lines": "INTEGER DEFAULT 0",
            "valid_records": "INTEGER DEFAULT 0", "loaded": "INTEGER DEFAULT 0",
            "errors": "INTEGER DEFAULT 0", "completed": "INTEGER DEFAULT 0",
            "updated_at": "TEXT DEFAULT ''",
        }
        for column, sql_type in progress_required.items():
            if column not in progress_existing:
                self.storage.execute(f"ALTER TABLE fabric_progress ADD COLUMN {column} {sql_type}")

    def load_all_datasets(self) -> dict[str, Any]:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(self.datasets_dir.glob("*.jsonl"))
        report: dict[str, Any] = {"files": 0, "records": 0, "errors": [], "datasets": []}
        for path in files:
            result = self.load_dataset_file(path)
            report["files"] += 1
            report["records"] += result["inserted"]
            report["errors"].extend(result["errors"])
            report["datasets"].append(result)
        return report

    def load_dataset_file(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        dataset_id = path.stem
        if not path.is_file():
            raise FileNotFoundError(path)
        total_lines = valid = inserted = errors = 0
        seen_ids: set[str] = set()
        issues: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as fp:
            for line_num, raw_line in enumerate(fp, 1):
                total_lines += 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors += 1
                    issues.append({"file": path.name, "line": line_num, "error": f"invalid JSON: {exc.msg}"})
                    continue
                if not isinstance(raw, dict):
                    errors += 1
                    issues.append({"file": path.name, "line": line_num, "error": "record must be a JSON object"})
                    continue
                valid += 1
                record_id = str(raw.get("record_id") or raw.get("id") or self._hash_id(dataset_id, line_num, line))
                if record_id in seen_ids:
                    errors += 1
                    issues.append({"file": path.name, "line": line_num, "error": f"duplicate record_id in file: {record_id}"})
                    continue
                seen_ids.add(record_id)
                try:
                    if self._insert_record(dataset_id, record_id, raw, path.name, line_num, line):
                        inserted += 1
                except Exception as exc:
                    errors += 1
                    issues.append({"file": path.name, "line": line_num, "error": str(exc)})

        total = self._count_dataset(dataset_id)
        self.storage.execute("""
            INSERT INTO fabric_progress
              (dataset_id, source_file, total_lines, valid_records, loaded, errors, completed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
              source_file=excluded.source_file, total_lines=excluded.total_lines,
              valid_records=excluded.valid_records, loaded=excluded.loaded,
              errors=excluded.errors, completed=excluded.completed, updated_at=excluded.updated_at;
        """, (dataset_id, path.name, total_lines, valid, total, errors, int(errors == 0), now_iso()))
        return {"file": path.name, "dataset_id": dataset_id, "lines": total_lines,
                "valid": valid, "inserted": inserted, "loaded_total": total,
                "errors": issues, "completed": errors == 0}

    def _insert_record(self, dataset_id: str, record_id: str, raw: dict[str, Any],
                       source_file: str, source_line: int, source_line_text: str) -> bool:
        exists = self.storage.query_one("SELECT record_id FROM fabric_records WHERE record_id=?", (record_id,))
        if exists:
            return False
        tags = raw.get("tags", [])
        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(x) for x in tags]
        self.storage.execute("""
            INSERT INTO fabric_records
              (record_id,dataset_id,topic,concept,knowledge_type,question,answer,explanation,
               language,framework,version,tags_json,raw_json,source_file,source_line,content_hash,loaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
        """, (
            record_id, dataset_id, str(raw.get("topic", "")), str(raw.get("concept", "")),
            str(raw.get("knowledge_type", raw.get("type", ""))), str(raw.get("question", "")),
            str(raw.get("answer", "")), str(raw.get("explanation", raw.get("principle", ""))),
            str(raw.get("language", "")), str(raw.get("framework", "")),
            str(raw.get("version", raw.get("version_range", ""))), json.dumps(tags, ensure_ascii=False),
            json.dumps(raw, ensure_ascii=False, default=str), source_file, source_line,
            hashlib.sha256(source_line_text.encode("utf-8")).hexdigest(), now_iso(),
        ))
        return True

    def get_batch(self, dataset_id: str | None = None, *, batch_size: int = 200,
                  offset: int = 0, language: str | None = None,
                  concept: str | None = None) -> list[FabricRecord]:
        if batch_size <= 0 or offset < 0:
            raise ValueError("batch_size must be > 0 and offset must be >= 0")
        sql = "SELECT * FROM fabric_records WHERE 1=1"
        params: list[Any] = []
        if dataset_id: sql += " AND dataset_id=?"; params.append(dataset_id)
        if language: sql += " AND language=?"; params.append(language)
        if concept: sql += " AND concept=?"; params.append(concept)
        sql += " ORDER BY record_id LIMIT ? OFFSET ?"; params.extend([batch_size, offset])
        return [self._row_to_record(r) for r in self.storage.query(sql, params)]

    def iterate_batches(self, *, batch_size: int = 200, **filters: Any) -> Iterator[list[FabricRecord]]:
        offset = 0
        while True:
            batch = self.get_batch(batch_size=batch_size, offset=offset, **filters)
            if not batch: break
            yield batch
            offset += len(batch)
            if len(batch) < batch_size: break

    def search(self, query: str, *, limit: int = 10, language: str | None = None) -> list[FabricRecord]:
        if not query or limit <= 0: return []
        terms = [t.lower() for t in query.split() if len(t) >= 2]
        if not terms: return []
        clauses, params = [], []
        for t in terms:
            like = f"%{t}%"
            clauses.append("(LOWER(question) LIKE ? OR LOWER(answer) LIKE ? OR LOWER(explanation) LIKE ? OR LOWER(concept) LIKE ? OR LOWER(topic) LIKE ? OR LOWER(tags_json) LIKE ?)")
            params.extend([like] * 6)
        sql = "SELECT * FROM fabric_records WHERE (" + " OR ".join(clauses) + ")"
        if language: sql += " AND language=?"; params.append(language)
        sql += " ORDER BY record_id LIMIT ?"; params.append(limit)
        return [self._row_to_record(r) for r in self.storage.query(sql, params)]

    def stats(self) -> dict[str, Any]:
        total = self.storage.query_one("SELECT COUNT(*) AS c FROM fabric_records")
        datasets = self.storage.query("SELECT dataset_id, COUNT(*) AS c FROM fabric_records GROUP BY dataset_id ORDER BY dataset_id")
        langs = self.storage.query("SELECT language, COUNT(*) AS c FROM fabric_records WHERE language != '' GROUP BY language ORDER BY c DESC")
        progress = self.storage.query("SELECT dataset_id, total_lines, valid_records, loaded, errors, completed FROM fabric_progress ORDER BY dataset_id")
        return {"total_records": int(total["c"]) if total else 0,
                "datasets": {r["dataset_id"]: int(r["c"]) for r in datasets},
                "languages": {r["language"]: int(r["c"]) for r in langs},
                "progress": progress}

    def _count_dataset(self, dataset_id: str) -> int:
        row = self.storage.query_one("SELECT COUNT(*) AS c FROM fabric_records WHERE dataset_id=?", (dataset_id,))
        return int(row["c"]) if row else 0

    @staticmethod
    def _hash_id(dataset_id: str, line_num: int, content: str) -> str:
        return f"{dataset_id}-{hashlib.sha256(f'{dataset_id}:{line_num}:{content}'.encode()).hexdigest()[:16]}"

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> FabricRecord:
        try: tags = json.loads(row.get("tags_json") or "[]")
        except (TypeError, json.JSONDecodeError): tags = []
        try: raw = json.loads(row.get("raw_json") or "{}")
        except (TypeError, json.JSONDecodeError): raw = {}
        return FabricRecord(
            record_id=row["record_id"], dataset_id=row["dataset_id"], topic=row.get("topic") or "",
            concept=row.get("concept") or "", knowledge_type=row.get("knowledge_type") or "",
            question=row.get("question") or "", answer=row.get("answer") or "",
            explanation=row.get("explanation") or "", language=row.get("language") or "",
            framework=row.get("framework") or "", version=row.get("version") or "",
            tags=tags if isinstance(tags, list) else [], raw=raw if isinstance(raw, dict) else {},
        )
