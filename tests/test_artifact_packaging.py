import json
import tarfile

from scripts.package_training_artifact import main


def test_package_training_artifact(tmp_path, monkeypatch):
    run = tmp_path / "run-1"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "training_config.json").write_text("{}", encoding="utf-8")
    (run / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (run / "metrics").mkdir()
    (run / "metrics" / "training_metrics.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["package_training_artifact.py", str(run)],
    )
    assert main() == 0

    archive = tmp_path / "run-1-artifact.tar.gz"
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "run-1/adapter/adapter_model.safetensors" in names
    manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]
