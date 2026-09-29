from pathlib import Path
import json

from sebrain import Config, SEBrain


def test_canonical_engineering_pipeline_wires_requirement_intent_plan_and_rag(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "D01.jsonl").write_text(
        json.dumps({
            "record_id": "D01-PIPE-001",
            "dataset_id": "D01",
            "topic": "testing",
            "concept": "regression testing",
            "question": "How should regression tests be used?",
            "answer": "Run targeted regression tests after a repair.",
        }) + "\n",
        encoding="utf-8",
    )
    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        brain.connect_knowledge_fabric(datasets)
        analysis = brain.analyze_engineering_task(
            "Build a tested API and run regression tests.",
            project_id="pipeline-test",
            top_k=2,
        )
        assert analysis.spec.functional
        assert analysis.intent.intent is not None
        assert analysis.plan.tasks
        assert analysis.knowledge is not None
        assert analysis.knowledge.records
        assert analysis.knowledge.records[0].record_id == "D01-PIPE-001"
