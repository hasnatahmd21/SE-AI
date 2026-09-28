from pathlib import Path

from sebrain import Config, SEBrain


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


def test_grouped_fabric_exports_are_discovered():
    with SEBrain(Config(data_dir=ROOT / ".pytest_sebrain_layout")) as brain:
        report = brain.connect_knowledge_fabric(DATASETS)

    assert report["files"] == 13
    assert {"D01", "D10", "D18", "D25", "D40", "D50", "D58"} <= set(
        report["found_dataset_ids"]
    )
    assert report["records"] > 10000
    assert report["errors"] == []
    assert "D26" in report["missing_dataset_ids"]


def test_real_fabric_query_path_uses_c34_c36_c35():
    with SEBrain(Config(data_dir=ROOT / ".pytest_sebrain_query")) as brain:
        report = brain.connect_knowledge_fabric(DATASETS)
        response = brain.ask("database migration regression", top_k=3)

    assert report["records"] > 10000
    assert response.retrieval_method == "lexical"
    assert response.knowledge
    assert response.evidence
    assert response.evidence[0]["record_id"] == response.knowledge[0].record_id
    assert response.evidence[0]["dataset_id"] == response.knowledge[0].dataset_id
