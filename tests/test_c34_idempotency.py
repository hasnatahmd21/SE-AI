from pathlib import Path

from sebrain.c01 import SQLiteStorage
from sebrain.c34 import KnowledgeFabricLoader


def test_reload_is_idempotent_and_conflict_is_visible(tmp_path: Path):
    p = tmp_path / "D01.jsonl"
    p.write_text(
        '{"record_id":"D01-R001","dataset_id":"D01","topic":"original"}\n',
        encoding="utf-8",
    )
    loader = KnowledgeFabricLoader(SQLiteStorage(":memory:"), tmp_path)
    first = loader.load_dataset_file(p)
    second = loader.load_dataset_file(p)
    assert first["inserted"] == 1
    assert second["inserted"] == 0
    assert second["duplicates"] == 1
    assert loader.stats()["total_records"] == 1

    p.write_text(
        '{"record_id":"D01-R001","dataset_id":"D01","topic":"changed"}\n',
        encoding="utf-8",
    )
    third = loader.load_dataset_file(p)
    assert third["inserted"] == 0
    assert third["conflicts"] == 1
    assert any("canonical record retained" in w["warning"] for w in third["warnings"])
    assert loader.stats()["total_records"] == 1
    assert loader.storage.query_one(
        "SELECT topic FROM fabric_records WHERE record_id='D01-R001'"
    )["topic"] == "original"
    actions = loader.storage.query(
        "SELECT action FROM fabric_record_audit WHERE record_id='D01-R001' ORDER BY audit_id"
    )
    assert [row["action"] for row in actions] == [
        "inserted", "duplicate_same", "duplicate_conflict"
    ]
