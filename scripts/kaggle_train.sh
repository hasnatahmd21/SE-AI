#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${SEBRAIN_TRAIN_CONFIG:-configs/training.example.json}"

python -m pip install -r requirements-training.txt
python scripts/validate_knowledge_fabric.py
python scripts/prepare_training_data.py --output training/knowledge_fabric.jsonl

python - "$CONFIG_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
model = str(cfg.get("base_model", "")).strip()
if not model or model == "REPLACE_WITH_A_COMPATIBLE_HUGGINGFACE_CAUSAL_LM":
    raise SystemExit(
        "Training configuration still uses the placeholder base_model. "
        "Set a real compatible causal language model before starting Kaggle training."
    )
PY

python scripts/train.py --config "$CONFIG_PATH" --dry-run

echo "Preflight passed. Starting the configured LoRA/PEFT training run..."
python scripts/train.py --config "$CONFIG_PATH"
