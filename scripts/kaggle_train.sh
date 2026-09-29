#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${SEBRAIN_TRAIN_CONFIG:-configs/training.example.json}"

python -m pip install -r requirements-training.txt
python scripts/validate_knowledge_fabric.py
python scripts/brain_smoke.py
python scripts/prepare_training_data.py --output training/knowledge_fabric.jsonl

python scripts/training_readiness.py \
  --config "$CONFIG_PATH" \
  --dataset training/knowledge_fabric.jsonl \
  --require-training-data

python scripts/model_preflight.py --config "$CONFIG_PATH"
python scripts/train.py --config "$CONFIG_PATH" --dry-run

RESULT_PATH="${SEBRAIN_TRAIN_RESULT:-/kaggle/working/sebrain_training_result.json}"
python scripts/train.py --config "$CONFIG_PATH" > "$RESULT_PATH"

RUN_ID="$(python - "$RESULT_PATH" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if result.get('status') != 'COMPLETED':
    raise SystemExit(f"Training did not complete successfully: {result.get('status')!r}")
run_id = result.get('run_id')
if not run_id: raise SystemExit('Training result has no run_id')
print(run_id)
PY
)"

python scripts/package_training_artifact.py "runs/$RUN_ID"
python scripts/inspect_training_run.py "$RUN_ID" --verify-artifacts

if [[ -n "${HF_REPO_ID:-}" ]]; then
  python scripts/upload_training_artifact_hf.py --adapter "runs/$RUN_ID/adapter"
fi

echo "Training and artifact packaging completed: runs/$RUN_ID"