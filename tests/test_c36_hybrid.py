from pathlib import Path
import json

from sebrain import Config, KnowledgeFabricLoader, RAGPipeline, SEBrain


def _fake_encoder(texts):
    vectors = []
    for text in texts:
        value = text.lower()
        if "use-after-free" in value or "dangling pointer" in value:
            vectors.append([1.0, 0.0, 0.0])
        elif "rust" in value and "ownership" in value:
            vectors.append([0.0, 1.0, 0.0])
        elif "typescript" in value:
            vectors.append([0.0, 0.0, 1.0])
        else:
            vectors.append([0.05, 0.05, 0.05])
    return vectors


def test_semantic_retrieval_can_rescue_no_keyword_overlap(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    records = [
        {
            "record_id": "D05-001",
            "dataset_id": "D05",
            "topic": "Memory safety",
            "concept": "use-after-free",
            "question": "What happens when memory is accessed after deallocation?",
            "answer": "A use-after-free accesses an object after its lifetime ended.",
        },
        {
            "record_id": "D27-001",
            "dataset_id": "D27",
            "topic": "Operations",
            "concept": "structured execution evidence",
            "question": "How should commands be evidenced?",
            "answer": "Record structured execution evidence.",
        },
    ]
    (datasets / "D05.jsonl").write_text(
        json.dumps(records[0]) + "\n", encoding="utf-8"
    )
    (datasets / "D27.jsonl").write_text(
        json.dumps(records[1]) + "\n", encoding="utf-8"
    )

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        brain.connect_knowledge_fabric(datasets)
        brain.bridge.rag = RAGPipeline(
            brain.fabric,
            semantic_encoder=_fake_encoder,
            semantic_top_k=10,
        )
        response = brain.ask("What is use-after-free?", top_k=1)
        assert response.knowledge
        assert response.knowledge[0].record_id == "D05-001"
        assert response.retrieval_method == "hybrid"
        assert response.evidence[0]["retrieval_sources"]


def test_semantic_rag_preserves_lexical_fallback(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    record = {
        "record_id": "D29-001",
        "dataset_id": "D29",
        "topic": "Rust",
        "concept": "ownership",
        "question": "How does Rust ownership work?",
        "answer": "Every value has an owner.",
    }
    (datasets / "D29.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        brain.connect_knowledge_fabric(datasets)
        response = brain.ask("Rust ownership", top_k=1)
        assert response.knowledge[0].record_id == "D29-001"
        assert response.retrieval_method == "lexical"
