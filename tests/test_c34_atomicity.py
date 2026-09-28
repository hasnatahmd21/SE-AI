from pathlib import Path
import pytest

from sebrain.c01 import SQLiteStorage, StorageError
from sebrain.c34 import KnowledgeFabricLoader


def _record():
    return '{"record_id":"D01-R001","dataset_id":"D01","topic":"atomic"}\n'


class FailingStorage(SQLiteStorage):
    def __init__(self, path):
        super().__init__(path)
        self.fail_audit = False

    def execute(self, sql, params=()):
        if self.fail_audit and "fabric_record_audit" in sql:
            raise StorageError("synthetic audit failure")
        return super().execute(sql, params)


def test_record_audit_catalog_are_atomic(tmp_path: Path):
    p = tmp_path / "D01.jsonl"
    p.write_text(_record(), encoding="utf-8")
    storage = FailingStorage(tmp_path / "fabric.db")
    loader = KnowledgeFabricLoader(storage, tmp_path)
    storage.fail_audit = True
    with pytest.raises(StorageError, match="synthetic audit failure"):
        loader.load_dataset_file(p)
    assert storage.query_one("SELECT * FROM fabric_records WHERE record_id='D01-R001'") is None
    assert storage.query_one("SELECT * FROM fabric_record_audit WHERE record_id='D01-R001'") is None
    assert storage.query_one("SELECT * FROM fabric_datasets WHERE dataset_id='D01'") is None
