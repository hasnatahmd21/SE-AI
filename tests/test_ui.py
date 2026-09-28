from pathlib import Path
import json

from sebrain.ui import build_app


def test_ui_builds_against_real_brain_facade(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    payload = {
        "dataset_manifest": {
            "dataset_id": "D58",
            "dataset_name": "UI Integration Fixture",
            "status": "LOCKED",
        },
        "records": [
            {
                "record_id": "D58-UI-001",
                "dataset_id": "D58",
                "topic": "FastAPI",
                "concept": "endpoint",
                "knowledge_type": "pattern",
                "question": "How should an endpoint be defined?",
                "answer": "Define the route and handler using the framework contract.",
            }
        ],
    }
    (datasets / "D56 - D58").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    demo, brain = build_app(
        datasets_dir=datasets,
        data_dir=tmp_path / ".ui-brain",
    )
    try:
        assert hasattr(demo, "launch")
        assert brain.fabric is not None
        response = brain.ask("FastAPI endpoint", top_k=1)
        assert response.knowledge
        assert response.knowledge[0].record_id == "D58-UI-001"
        assert response.evidence
    finally:
        brain.stop()
