from pathlib import Path

import pytest

from sebrain import (
    CallableModelGateway,
    Config,
    ModelRequest,
    ModelResponse,
    ModelUnavailableError,
    SEBrain,
    TrainingDatasetExporter,
)


def test_model_gateway_is_explicit_and_provider_agnostic(tmp_path: Path):
    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        with pytest.raises(ModelUnavailableError):
            brain.generate("hello")

        seen = {}

        def infer(request: ModelRequest):
            seen["prompt"] = request.prompt
            return ModelResponse("real adapter response", "test-model")

        brain.set_model_gateway(CallableModelGateway(infer, model_id="test-model"))
        response = brain.generate("hello", context=({"record_id": "D01-1"},))
        assert response.text == "real adapter response"
        assert response.model_id == "test-model"
        assert seen["prompt"] == "hello"


def test_training_export_is_deterministic_and_provenance_preserving():
    from sebrain.c34 import FabricRecord

    records = [
        FabricRecord(
            record_id="D01-2",
            dataset_id="D01",
            question="How should a migration be tested?",
            answer="Run the migration and regression tests.",
            source_file="D1 - D5",
            content_hash="hash-2",
        ),
        FabricRecord(
            record_id="D01-1",
            dataset_id="D01",
            question="How should input be validated?",
            answer="Validate at the boundary.",
            source_file="D1 - D5",
            content_hash="hash-1",
        ),
        FabricRecord(record_id="D01-skip", dataset_id="D01"),
    ]
    exporter = TrainingDatasetExporter()
    first = exporter.convert(records)
    second = exporter.convert(reversed(records))
    assert [x.to_dict() for x in first] == [x.to_dict() for x in second]
    assert len(first) == 2
    assert {x.record_id for x in first} == {"D01-1", "D01-2"}
    assert all(x.source_file == "D1 - D5" for x in first)
    assert all(x.content_hash.startswith("hash-") for x in first)


def test_training_export_preserves_validation_state_without_fabricating_readiness():
    from sebrain.c34 import FabricRecord

    records = [
        FabricRecord(
            record_id="D46-B21-R001",
            dataset_id="D46",
            question="How is this regression case validated?",
            answer="Validate it against the target database behavior.",
            source_file="D46 - D50",
            content_hash="h1",
            raw={
                "execution_status": "NOT_EXECUTED",
                "validation_status": "REQUIRES_TARGET_DATABASE_VALIDATION",
                "evidence_level": "DATABASE_BEHAVIOR_PATTERN",
                "provenance": {"source_type": "structured_engineering_synthesis"},
                "relationships": [{"relation": "related_to", "target": "D46-B19-R001"}],
            },
        ),
        FabricRecord(
            record_id="D18-B01-R001",
            dataset_id="D18",
            question="What is the verified invariant?",
            answer="The invariant is preserved.",
            source_file="D18 - D20",
            content_hash="h2",
            raw={
                "execution_status": "EXECUTED",
                "validation_status": "VERIFIED",
                "evidence_level": "TESTED",
                "provenance": {"source_type": "test"},
                "relationships": ["supports"],
            },
        ),
    ]

    examples = TrainingDatasetExporter().convert(records)
    by_id = {x.record_id: x for x in examples}
    assert by_id["D46-B21-R001"].training_eligible is False
    assert by_id["D46-B21-R001"].execution_status == "NOT_EXECUTED"
    assert by_id["D46-B21-R001"].validation_status == "REQUIRES_TARGET_DATABASE_VALIDATION"
    assert by_id["D46-B21-R001"].relationships[0]["target"] == "D46-B19-R001"
    assert by_id["D18-B01-R001"].training_eligible is True
