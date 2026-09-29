"""Package a completed SE Brain training run as a reproducible artifact.

The script never uploads credentials or model weights to Git. It creates a
portable archive plus a SHA-256 manifest that can be stored as a Kaggle output,
Kaggle Dataset, or another model-artifact store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


INCLUDE_FILES = (
    "training_config.json",
    "dataset_manifest.json",
    "model_manifest.json",
    "tokenizer_manifest.json",
    "lora_config.json",
    "metrics/training_metrics.json",
    "metrics/evaluation.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_adapter(run_dir: Path, explicit: Path | None) -> Path:
    if explicit:
        candidate = explicit if explicit.is_absolute() else run_dir / explicit
        if candidate.is_dir():
            return candidate
        raise SystemExit(f"adapter directory not found: {candidate}")

    candidates = [
        run_dir / "adapter",
        run_dir / "final_adapter",
        run_dir / "artifacts" / "adapter",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "final adapter directory not found; pass --adapter-dir with the exact "
        "directory created by the training run"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    adapter = find_adapter(run_dir, args.adapter_dir)
    files: list[Path] = []
    for rel in INCLUDE_FILES:
        p = run_dir / rel
        if p.is_file():
            files.append(p)
    for p in sorted(adapter.rglob("*")):
        if p.is_file():
            files.append(p)

    if not files:
        raise SystemExit("no artifact files found")

    unique: dict[str, Path] = {}
    for p in files:
        unique[str(p.relative_to(run_dir))] = p

    manifest = {
        "run_dir_name": run_dir.name,
        "adapter_relative_path": str(adapter.relative_to(run_dir)),
        "files": [
            {"path": rel, "sha256": sha256_file(p), "size": p.stat().st_size}
            for rel, p in sorted(unique.items())
        ],
    }

    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    output = args.output or (run_dir.parent / f"{run_dir.name}-artifact.tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output, "w:gz") as archive:
        for rel in sorted(unique):
            archive.add(unique[rel], arcname=f"{run_dir.name}/{rel}")
        archive.add(manifest_path, arcname=f"{run_dir.name}/artifact_manifest.json")

    print(json.dumps({
        "artifact": str(output),
        "sha256": sha256_file(output),
        "manifest": str(manifest_path),
        "file_count": len(unique),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
