# SE Brain — Autonomous Software Engineering Brain

SE Brain is a modular software-engineering foundation with C01–C33 core engines and a Knowledge Fabric layer (C34–C36). The canonical Python package is **sebrain/**.

## Architecture

- **C01–C33:** core software-engineering engines.
- **C34:** persistent D01–D58 JSONL Knowledge Fabric loader with provenance, validation and idempotent SQLite loading.
- **C35:** Brain/Dataset Bridge combining requirement understanding, intent/risk analysis and retrieval.
- **C36:** deterministic local lexical retrieval; no external LLM or paid API is required for the Knowledge Fabric path.

## Layout

```
sebrain/       # canonical package: C01–C36 + UI
  __init__.py
  c01.py ... c36.py
  c34_loader.py
  c36_rag.py
  ui.py
datasets/      # D01–D58 JSONL datasets are placed here
run_ui.py
requirements.txt
pyproject.toml
```

## Install

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Connect the Knowledge Fabric

Place D01.jsonl through D58.jsonl under `./datasets`.

```python
from pathlib import Path
from sebrain import SEBrain, Config

with SEBrain(Config(data_dir=Path("./.sebrain"))) as brain:
    report = brain.connect_knowledge_fabric("./datasets")
    print(report)
    response = brain.ask("How do I build a FastAPI endpoint?")
    print(response.to_english())
```

## UI

```bash
python -m sebrain.ui --datasets ./datasets --host 127.0.0.1 --port 7860
```

Add `--share` for a temporary Gradio share link.

## Engine tests

Each engine is designed to be independently runnable, for example:

```bash
python -m sebrain.c01 --test
python -m sebrain.c18 --test
```

The historical audit report records the prior C01–C33 audit. A fresh environment validation with the real dependencies remains the authoritative deployment check.
