from pathlib import Path
from sebrain.c01 import SQLiteStorage
from sebrain.c34 import KnowledgeFabricLoader


def test_unplanned_dataset_is_rejected(tmp_path: Path):
    p = tmp_path / "D26-D30"
    p.write_text(
        '{"record_id":"D26-R001","dataset_id":"D26","topic":"out of plan"}\n'
        '{"record_id":"D27-R001","dataset_id":"D27","topic":"planned"}\n',
        encoding="utf-8",
    )
    loader = KnowledgeFabricLoader(SQLiteStorage(":memory:"), tmp_path)
    result = loader.load_dataset_file(p)
    assert result["inserted"] == 1
    assert any(
        e.get("dataset_id") == "D26"
        and "outside the planned" in e.get("error", "")
        for e in result["errors"]
    )
    assert loader.coverage_audit()["record_counts"] == {"D27": 1}
