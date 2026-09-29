# SE Brain — Autonomous Software Engineering Brain

SE Brain is a modular software-engineering foundation with C01–C33 core engines and a Knowledge Fabric layer (C34–C36). The canonical Python package is **sebrain/**.

## Architecture

- **C01–C33:** core software-engineering engines.
- **C34:** persistent intended Knowledge Fabric loader with provenance, validation, duplicate/conflict auditing and idempotent SQLite loading.
- **C35:** Brain/Dataset Bridge combining requirement understanding, intent/risk analysis and evidence-linked retrieval.
- **C36:** deterministic local lexical retrieval; no external LLM or paid API is required for the Knowledge Fabric retrieval path.

### Knowledge path

```text
intended grouped Knowledge Fabric source files (D26 intentionally unplanned)
          │
          ▼
        C34
  parse → validate → normalize
  provenance → catalog → SQLite
          │
          ▼
        C36
  deterministic retrieval
  scoring + provenance trace
          │
          ▼
        C35
  requirements + intent/risk
  evidence-linked Brain response
          │
          ▼
       SEBrain.ask()
```

The Brain facade exposes this path through `connect_knowledge_fabric()`,
`ask()`, `analyze_engineering_task()`, `fabric_stats()`, and
`knowledge_catalog()`. The same canonical facade is available through the
FastAPI boundary in `sebrain/api.py` and the local Gradio UI.

## Layout

```text
sebrain/       # canonical package: C01–C36 + UI
datasets/      # grouped Knowledge Fabric sources (D26 intentionally unplanned)
tests/         # unit and integration tests
scripts/       # repository validation commands
run_ui.py
requirements.txt
pyproject.toml
.github/
  workflows/ci.yml
```

The dataset sources are grouped files such as `D1 - D5`, `D21 - D25`,
and `D56 - D58`. C34 discovers these extensionless grouped files as well as
standard JSON/JSONL sources.

## Install

Python 3.11+:

```bash
python -m pip install -e .
```

## Connect the Knowledge Fabric

```python
from pathlib import Path
from sebrain import SEBrain, Config

with SEBrain(Config(data_dir=Path("./.sebrain"))) as brain:
    report = brain.connect_knowledge_fabric("./datasets")
    print(report)
    response = brain.ask("How do I build a FastAPI endpoint?")
    print(response.to_english())
```

## Validate the real repository datasets

```bash
python scripts/validate_knowledge_fabric.py
```

This validates the actual repository Knowledge Fabric export through the same C34 → C36
→ C35 path used by the Brain. It requires complete dataset-ID coverage and zero
loader errors, while intentionally avoiding an artificial fixed record-count
requirement.

## API

Run the real API boundary with:

```bash
uvicorn sebrain.api:app --host 0.0.0.0 --port 8000
```

The API validates the Knowledge Fabric during application startup and exposes
`/health`, `/knowledge`, `/analyze`, and `/ask`. It does not fabricate
model-backed generation; a real `ModelGateway` must be attached by the host
application for that capability.

## UI

```bash
python -m sebrain.ui --datasets ./datasets --host 127.0.0.1 --port 7860
```

Add `--share` for a temporary Gradio share link.

## Tests

```bash
python -m compileall -q sebrain
python -m pytest -q
```

GitHub Actions runs the package tests on Python 3.11 and 3.12 and separately
validates the real repository Knowledge Fabric export on Python 3.12.


## Model Training — LoRA/PEFT

The repository includes a separate heavy training layer so the normal SE Brain installation remains lightweight.

### Training architecture

~~~text
Knowledge Fabric
      ↓
eligible training records
      ↓
deterministic JSONL
      ↓
TrainingConfig
      ↓
base model + tokenizer
      ↓
real PEFT/LoRA adapter
      ↓
Transformers Trainer
      ↓
checkpoints + evaluation
      ↓
Training Run Registry
      ↓
versioned adapter + reproducibility manifest
~~~

Install training-only dependencies:

~~~bash
python -m pip install -r requirements-training.txt
~~~

Prepare the canonical training artifact:

~~~bash
python scripts/validate_knowledge_fabric.py
python scripts/prepare_training_data.py --output training/knowledge_fabric.jsonl
~~~

Copy configs/training.example.json, set a real Hugging Face-compatible causal language model, and review every training parameter before starting.

Preflight without downloading a model:

~~~bash
python scripts/train.py --config configs/training.example.json --dry-run
~~~

Start real LoRA/PEFT training:

~~~bash
python scripts/train.py --config configs/training.example.json
~~~

Resume a specific interrupted run:

~~~bash
python scripts/train.py --config configs/training.example.json --resume --run-id <run_id>
~~~

Inspect a run:

~~~bash
python scripts/inspect_training_run.py <run_id>
~~~

Each run stores configuration, dataset/model/tokenizer manifests, LoRA configuration, checkpoints, logs, metrics, evaluation results and the final adapter under runs/<run_id>/.

The training layer refuses to train on records that are not explicitly training-eligible. It does not convert illustrative, unexecuted, planned, or target-validation-required records into empirical training evidence.

### Kaggle

After cloning the repository into a Kaggle environment:

~~~bash
bash scripts/kaggle_train.sh
~~~

This performs dependency installation, Knowledge Fabric validation, training-data preparation, a dry-run preflight, and then starts the configured LoRA/PEFT training run. Set SEBRAIN_TRAIN_CONFIG if using a different config. A real model must be configured before starting expensive training. Kaggle still requires execution of the notebook/script; repository upload alone cannot execute training.

## Training readiness hardening

Use `scripts/training_readiness.py` before an expensive run. The repository also provides `configs/training.smoke.json` for a small GPU smoke test, `scripts/validate_training_artifact.py` for loading the final adapter with its base model, and `scripts/package_training_artifact.py` for a portable archive with SHA-256 manifest. The Kaggle launcher performs these gates automatically and can optionally back up the adapter to a Hugging Face model repository when `HF_REPO_ID` and `HF_TOKEN` are provided as environment secrets. Generated training data, runs, and archives are excluded from Git.
