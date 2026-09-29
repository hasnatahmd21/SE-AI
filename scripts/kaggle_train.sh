#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements-training.txt
python scripts/validate_knowledge_fabric.py
python scripts/prepare_training_data.py --output training/knowledge_fabric.jsonl
python scripts/train.py --config configs/training.example.json --dry-run
echo "Kaggle preflight complete. Set base_model in the config, then execute:"
echo "python scripts/train.py --config configs/training.example.json"
