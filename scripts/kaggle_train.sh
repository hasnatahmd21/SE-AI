#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${SEBRAIN_TRAIN_CONFIG:-configs/training.example.json}"

python -m pip install -r requirements-training.txt
python scripts/validate_knowledge_fabric.py
python scripts/prepare_training_data.py --output training/knowledge_fabric.jsonl
python scripts/train.py --config "$CONFIG_PATH" --dry-run

echo "Preflight passed. Starting the configured LoRA/PEFT training run..."
python scripts/train.py --config "$CONFIG_PATH"
