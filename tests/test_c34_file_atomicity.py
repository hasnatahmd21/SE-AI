from pathlib import Path
import pytest

from sebrain.c01 import SQLiteStorage, StorageError
from sebrain.c34 import KnowledgeFabricLoader


class FailingSourceStorage(SQLiteStorage):
    def __init__(self, path):
        super().__init__(path)
        self.fail_progress = False

    def execute(self, sql, params=()):
        if self.fail_progress and "fabric_progress" in sql:
            raise StorageError("synthetic progress failure")
        return super().execute(sql, params)


def test_whole_file_load_rolls_back_on_metadata_failure(tmp_path: Path):
    p = tmp_path / "D01.jsonl"
    p.write_text(
        '{"record_id":"D01-R001","dataset_id":"D01","topic":"atomic"}\n',
        encoding="utf-8",
    )
    storage = FailingSourceStorage(tmp_path / "fabric.db")
    loader = KnowledgeFabricLoader(storage, tmp_path)
    storage.fail_progress = True

    with pytest.raises(StorageError, match="synthetic progress failure"):
        loader.load_dataset_file(p)

    assert storage.query_one(
        "SELECT * FROM fabric_records WHERE record_id='D01-R001'"
    ) is None
    assert storage.query_one(
        "SELECT * FROM fabric_record_audit WHERE record_id='D01-R001'"
    ) is None
    assert storage.query_one(
        "SELECT * FROM fabric_sources WHERE source_file='D01.jsonl'"
    ) is None
    assert storage.query_one(
        "SELECT * FROM fabric_progress WHERE dataset_id='D01'"
    ) is None
