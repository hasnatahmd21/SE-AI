from pathlib import Path

from sebrain import Config, SEBrain


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


def test_grouped_fabric_exports_are_discovered(tmp_path: Path):
    with SEBrain(Config(data_dir=tmp_path / "brain")) as brain:
        report = brain.connect_knowledge_fabric(DATASETS)

    assert report["files"] == 13
    expected = {f"D{i:02d}" for i in range(1, 59) if i != 26}
    found = set(report["found_dataset_ids"])
    missing = set(report["missing_dataset_ids"])
    assert {"D01", "D10", "D18", "D25", "D27", "D40", "D50", "D58"} <= found
    assert "D26" not in found
    assert "D26" not in missing
    assert found | missing == expected
    assert not found & missing
    assert len(found) == 57
    assert report["records"] > 10000
    assert all(
        isinstance(item.get("file"), str) and isinstance(item.get("error"), str)
        for item in report["errors"]
    )


def test_real_fabric_query_path_uses_c34_c36_c35(tmp_path: Path):
    with SEBrain(Config(data_dir=tmp_path / "brain")) as brain:
        report = brain.connect_knowledge_fabric(DATASETS)
        response = brain.ask("database migration regression", top_k=3)

    assert report["records"] > 10000
    assert response.retrieval_method == "lexical"
    assert response.knowledge
    assert response.evidence
    assert response.evidence[0]["record_id"] == response.knowledge[0].record_id
    assert response.evidence[0]["dataset_id"] == response.knowledge[0].dataset_id
    assert response.evidence[0]["source_file"]
    assert response.evidence[0]["content_hash"]
