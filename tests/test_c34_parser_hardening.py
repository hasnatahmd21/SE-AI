import json
import pytest
from pathlib import Path

from sebrain.c01 import SQLiteStorage
from sebrain.c34 import KnowledgeFabricLoader


def test_malformed_json_is_reported_not_silently_dropped(tmp_path: Path):
    p = tmp_path / "D01.jsonl"
    p.write_text(
        '{"record_id":"D01-R001","dataset_id":"D01","topic":"ok"}\n'
        '{"record_id":"D01-R002","dataset_id":"D01","topic":"broken"\n',
        encoding="utf-8",
    )
    loader = KnowledgeFabricLoader(SQLiteStorage(":memory:"), tmp_path)
    result = loader.load_dataset_file(p)
    assert result["errors"]
    assert any(e["line"] == 2 for e in result["errors"])


def test_valid_json_document_still_loads(tmp_path: Path):
    p = tmp_path / "D01.jsonl"
    p.write_text(
        json.dumps({"record_id":"D01-R001","dataset_id":"D01","topic":"ok"}) + "\n",
        encoding="utf-8",
    )
    loader = KnowledgeFabricLoader(SQLiteStorage(":memory:"), tmp_path)
    result = loader.load_dataset_file(p)
    assert result["errors"] == []
    assert result["inserted"] == 1
