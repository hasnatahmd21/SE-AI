"""Static and data-level training readiness gate for SE Brain.

This command does not start model training. It verifies that the repository
contains the required training machinery, that the selected configuration is
valid, and optionally that the prepared JSONL contains eligible records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sebrain.training_config import TrainingConfig
from sebrain.training_engine import TrainingDataError, load_eligible_examples


REQUIRED_PATHS = (
    "sebrain/training_config.py",
    "sebrain/training_engine.py",
    "sebrain/training_registry.py",
    "scripts/prepare_training_data.py",
    "scripts/train.py",
    "scripts/kaggle_train.sh",
    "configs/training.example.json",
    "requirements-training.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.example.json"))
    parser.add_argument("--dataset", type=Path, default=Path("training/knowledge_fabric.jsonl"))
    parser.add_argument("--require-training-data", action="store_true")
    args = parser.parse_args()

    missing = [p for p in REQUIRED_PATHS if not Path(p).is_file()]
    if missing:
        print(json.dumps({"ready": False, "stage": "repository", "missing": missing}, indent=2))
        return 1

    try:
        cfg = TrainingConfig.from_file(args.config)
    except Exception as exc:
        print(json.dumps({"ready": False, "stage": "config", "error": str(exc)}, indent=2))
        return 1

    placeholder = "REPLACE_WITH_A_COMPATIBLE_HUGGINGFACE_CAUSAL_LM"
    if cfg.base_model.strip() == placeholder:
        print(json.dumps({
            "ready": False,
            "stage": "config",
            "error": "A real base_model must be selected before full training.",
            "config": str(args.config),
        }, indent=2))
        return 1

    result = {
        "ready": True,
        "config": str(args.config),
        "base_model": cfg.base_model,
        "dataset": str(args.dataset),
        "training_data_checked": False,
    }

    if args.dataset.exists():
        try:
            _, manifest = load_eligible_examples(args.dataset)
        except TrainingDataError as exc:
            print(json.dumps({"ready": False, **result, "stage": "training-data", "error": str(exc)}, indent=2))
            return 1
        result["training_data_checked"] = True
        result["dataset_manifest"] = manifest
    elif args.require_training_data:
        print(json.dumps({"ready": False, **result, "stage": "training-data", "error": f"missing: {args.dataset}"}, indent=2))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
