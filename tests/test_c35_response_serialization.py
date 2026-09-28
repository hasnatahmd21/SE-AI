from pathlib import Path

from sebrain import Config, SEBrain

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


def test_brain_response_dict_preserves_answer_and_evidence(tmp_path: Path):
    with SEBrain(Config(data_dir=tmp_path / "brain")) as brain:
        brain.connect_knowledge_fabric(DATASETS)
        response = brain.ask("database migration regression", top_k=3)
        payload = response.to_dict()

    assert payload["answer"] == response.answer
    assert payload["answer"]
    assert payload["knowledge_count"] == len(response.knowledge)
    assert payload["evidence"]
    assert payload["evidence"][0]["record_id"] == response.knowledge[0].record_id
    assert payload["retrieval_method"] == "lexical"
