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


def test_rag_reranking_sees_candidates_beyond_fixed_window(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    records = []
    for index in range(1, 201):
        records.append({
            "record_id": f"D01-{index:03d}",
            "topic": "FastAPI",
            "concept": "endpoint",
            "question": "How do I build a FastAPI endpoint?",
            "answer": "Use a FastAPI route for the endpoint.",
        })
    records.append({
        "record_id": "D01-999",
        "topic": "FastAPI",
        "concept": "unique-target",
        "question": "How do I build the unique-target FastAPI endpoint?",
        "answer": "Use the unique-target route.",
    })
    (datasets / "D01.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        report = brain.connect_knowledge_fabric(datasets)
        assert report["records"] == 201
        response = brain.ask("FastAPI unique-target endpoint", top_k=1)
        assert response.knowledge
        assert response.knowledge[0].record_id == "D01-999"
