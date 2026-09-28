from pathlib import Path

from sebrain.c01 import SQLiteStorage
from sebrain.c34 import KnowledgeFabricLoader


def test_progress_not_completed_when_source_has_parse_error(tmp_path: Path):
    p = tmp_path / "D01.jsonl"
    p.write_text(
        '{"record_id":"D01-R001","dataset_id":"D01","topic":"valid"}\n'
        '{"record_id":"D01-R002","dataset_id":"D01","topic":"broken"\n',
        encoding="utf-8",
    )
    loader = KnowledgeFabricLoader(SQLiteStorage(":memory:"), tmp_path)
    result = loader.load_dataset_file(p)
    assert result["inserted"] == 1
    assert result["errors"]
    progress = loader.storage.query_one(
        "SELECT completed, loaded FROM fabric_progress WHERE dataset_id='D01'"
    )
    assert progress["loaded"] == 1
    assert progress["completed"] == 0
