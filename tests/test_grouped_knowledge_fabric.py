from pathlib import Path
import json

from sebrain import Config, SEBrain
from sebrain.c34 import KnowledgeFabricLoader


def test_grouped_knowledge_fabric_source_is_supported(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    grouped = {
        "dataset_manifest": {
            "dataset_id": "D01",
            "dataset_name": "Language Fundamentals",
            "dataset_version": "1.1.0",
            "schema_version": "KF-1.1",
            "status": "LOCKED",
            "language": "English",
            "authority": "TEST",
            "scope": "Foundational language knowledge",
            "purpose": "Integration test",
        },
        "records": [
            {
                "record_id": "D01-TEST-001",
                "topic": "Python",
                "concept": "function",
                "knowledge_type": "pattern",
                "question": "How do I define a Python function?",
                "answer": "Use def followed by the function name and parameters.",
                "explanation": "A function groups reusable behavior.",
                "language": "Python",
                "tags": ["function", "syntax"],
            }
        ],
    }
    (datasets / "D1 - D5").write_text(
        json.dumps(grouped, ensure_ascii=False), encoding="utf-8"
    )

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        report = brain.connect_knowledge_fabric(datasets)
        assert report["files"] == 1
        assert report["errors"] == []
        assert report["found_dataset_ids"] == ["D01"]
        assert report["missing_dataset_ids"] == [f"D{i:02d}" for i in range(2, 59)]

        catalog = brain.knowledge_catalog()
        assert catalog[0]["dataset_id"] == "D01"
        assert catalog[0]["status"] == "LOCKED"

        response = brain.ask("How do I define a Python function?", top_k=1)
        assert response.knowledge
        assert response.knowledge[0].record_id == "D01-TEST-001"
        assert response.evidence
        assert response.retrieval_method == "lexical"


def test_grouped_knowledge_fabric_load_is_idempotent(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    payload = {
        "dataset_manifest": {"dataset_id": "D58", "dataset_name": "Test"},
        "records": [{"record_id": "D58-TEST-001", "answer": "deterministic answer"}],
    }
    (datasets / "D56 - D58").write_text(json.dumps(payload), encoding="utf-8")

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        first = brain.connect_knowledge_fabric(datasets)
        second = brain.connect_knowledge_fabric(datasets)
        assert first["records"] == 1
        assert second["records"] == 0
        assert brain.fabric_stats()["total_records"] == 1
        assert brain.fabric_stats()["coverage"]["present"] == ["D58"]


def test_jsonl_records_are_not_skipped(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    records = [
        {"record_id": f"D58-TEST-{index:03d}", "dataset_id": "D58", "answer": f"answer {index}"}
        for index in range(1, 4)
    ]
    (datasets / "D56 - D58").write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        report = brain.connect_knowledge_fabric(datasets)
        assert report["errors"] == []
        assert report["records"] == 3
        assert brain.fabric_stats()["coverage"]["record_counts"] == {"D58": 3}


def test_malformed_record_quotes_are_repaired_and_metadata_fragments_do_not_fail(
    tmp_path: Path,
):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    malformed_record = (
        '{"record_id":"D58-TEST-QUOTE-001","dataset_id":"D58",'
        '"topic":"JSON repair","invalid_example":"SELECT * FROM users WHERE name = \'" + "USER_INPUT" + "\'"}'
    )
    source = """Header written by the dataset export process.
D58 validation ledger
{
  "dataset_id": "D58",
  "status": "VALIDATING",
  "note": "This metadata block is not a record and may contain non-JSON text.
}
""" + malformed_record + "\n"
    (datasets / "D56 - D58").write_text(source, encoding="utf-8")

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        report = brain.connect_knowledge_fabric(datasets)
        assert report["errors"] == []
        assert report["records"] == 1
        assert any(
            item.get("warning") == "repaired narrowly scoped JSON string quoting in records"
            for item in report["warnings"]
        )

        response = brain.ask("JSON repair USER_INPUT", top_k=1)
        assert response.knowledge
        assert response.knowledge[0].record_id == "D58-TEST-QUOTE-001"
        assert response.knowledge[0].raw["invalid_example"] == (
            "SELECT * FROM users WHERE name = '\" + \"USER_INPUT\" + \"'"
        )


def test_grouped_export_allows_prose_between_json_documents(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    grouped = """D26 B41-B42 generated knowledge export.

{"record_id":"D26-TEST-001","dataset_id":"D26","topic":"parser","answer":"first"}

Additional explanatory text between records.

{"record_id":"D26-TEST-002","dataset_id":"D26","topic":"parser","answer":"second"}

D26 final-range note.
"""
    (datasets / "D26 - D30").write_text(grouped, encoding="utf-8")

    with SEBrain(Config(data_dir=tmp_path / ".brain")) as brain:
        report = brain.connect_knowledge_fabric(datasets)
        assert report["errors"] == []
        assert report["found_dataset_ids"] == ["D26"]
        assert brain.fabric_stats()["total_records"] == 2
        response = brain.ask("parser first", top_k=1)
        assert response.knowledge



def test_json_quote_repair_preserves_embedded_array_and_function_quotes():
    lines = [
        (
            '{"record_id":"D34-QUOTE-001","dataset_id":"D34",'
            '"corrected_code":"validate_non_null(orders, "order_id")\\n'
            'validate_allowed_values(orders, "status", ["created", "paid"])"'
            '}'
        ),
        (
            '{"record_id":"D35-QUOTE-001","dataset_id":"D35",'
            '"corrected_code":"raise ValueError("partition_count must be positive")"'
            '}'
        ),
    ]
    expected = [
        'validate_non_null(orders, "order_id")\\n'
        'validate_allowed_values(orders, "status", ["created", "paid"])',
        'raise ValueError("partition_count must be positive")',
    ]

    for line, expected_code in zip(lines, expected):
        repaired = KnowledgeFabricLoader._repair_common_json_defects(line)
        assert repaired is not None
        payload = json.loads(repaired)
        assert payload["corrected_code"] == expected_code


def test_json_quote_repair_helper_recovers_embedded_code_quotes():
    line = (
        '{"record_id":"D58-HELPER-001","dataset_id":"D58",'
        '"invalid_example":"SELECT * FROM users WHERE name = \'" + "USER_INPUT" + "\'"}'
    )
    repaired = KnowledgeFabricLoader._repair_common_json_defects(line)
    assert repaired is not None
    payload = json.loads(repaired)
    assert payload["record_id"] == "D58-HELPER-001"
    # The repair layer fixes JSON quoting only; it must preserve the
    # record's original semantic content for provenance and round-tripping.
    assert payload["invalid_example"] == (
        "SELECT * FROM users WHERE name = '\" + \"USER_INPUT\" + \"'"
    )
