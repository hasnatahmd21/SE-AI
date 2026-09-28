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


def test_malformed_code_expression_quotes_are_repaired():
    line = '{"record_id":"D16-B57-R02","dataset_id":"D16","invalid_example":"Concatenate a client-supplied sort field directly into `ORDER BY " + sort_field + "`."}'
    repaired = KnowledgeFabricLoader._repair_common_json_defects(line)
    assert repaired is not None
    payload = json.loads(repaired)
    assert payload["record_id"] == "D16-B57-R02"
    assert "sort_field" in payload["invalid_example"]


def test_literal_control_characters_inside_record_strings_are_repaired():
    line = '{"record_id":"D26-R001","dataset_id":"D27","example":"first' + "\t" + 'second"}'
    repaired = KnowledgeFabricLoader._repair_common_json_defects(line)
    assert repaired is not None
    payload = json.loads(repaired)
    assert payload["example"] == "first\tsecond"

def test_json_quote_repair_handles_database_security_sql_examples():
    line = (
        '{"record_id":"D16-B20-SEC-0002","dataset_id":"D16",'
        '"invalid_example":"SELECT * FROM users WHERE username = \'" + "USER_INPUT" + "\'"}'
    )
    repaired = KnowledgeFabricLoader._repair_common_json_defects(line)
    assert repaired is not None
    payload = json.loads(repaired)
    assert payload["record_id"] == "D16-B20-SEC-0002"
    assert payload["invalid_example"] == (
        "SELECT * FROM users WHERE username = '" + "USER_INPUT" + "'"
    )
