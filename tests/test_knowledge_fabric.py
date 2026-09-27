from pathlib import Path
import json

from sebrain import Config, SEBrain


def test_knowledge_fabric_load_and_retrieve(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "D01.jsonl").write_text(
        json.dumps({
            "record_id": "D01-001",
            "topic": "FastAPI",
            "concept": "endpoint",
            "knowledge_type": "pattern",
            "question": "How do I create a FastAPI endpoint?",
            "answer": "Define a route with FastAPI.",
            "explanation": "Routes expose application operations.",
            "language": "Python",
            "framework": "FastAPI",
            "tags": ["api", "web"],
        }) + "\n", encoding="utf-8")
    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        report = brain.connect_knowledge_fabric(datasets)
        assert report["files"] == 1
        assert report["records"] == 1
        response = brain.ask("FastAPI endpoint", top_k=1)
        assert response.knowledge
        assert response.knowledge[0].record_id == "D01-001"


def test_loader_is_idempotent(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "D01.jsonl").write_text(json.dumps({"record_id": "r1", "answer": "x"}) + "\n", encoding="utf-8")
    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        first = brain.connect_knowledge_fabric(datasets)
        second = brain.connect_knowledge_fabric(datasets)
        assert first["records"] == 1
        assert second["records"] == 0
        assert brain.fabric_stats()["total_records"] == 1
