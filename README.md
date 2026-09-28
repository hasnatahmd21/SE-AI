# SE Brain — Autonomous Software Engineering Brain

SE Brain is a modular software-engineering foundation with C01–C33 core engines and a Knowledge Fabric integration layer (C34–C36). The canonical Python package is **sebrain/**.

## Architecture

- **C01–C33:** core software-engineering engines and shared Brain foundation.
- **C34:** persistent D01–D58 Knowledge Fabric loader with dataset catalog, provenance, duplicate/conflict audit, source tracking, validation reporting, and idempotent SQLite loading.
- **C35:** Brain/Dataset Bridge that connects requirement understanding and intent/risk analysis to retrieval and produces a structured English response with an evidence trace.
- **C36:** deterministic local lexical retrieval over the full indexed Knowledge Fabric. No external LLM, embedding API, vector database, or paid API is required for this path.

Active Knowledge Fabric path:

```text
User Query
   ↓
SEBrain
   ↓
C35 Brain/Dataset Bridge
   ├── C05 Requirement Understanding
   ├── C06 Intent & Context / Risk
   └── C36 Deterministic Retrieval
            ↓
          C34
            ↓
     SQLite Knowledge Fabric
```

C34 is the persistent data layer; C36 is the retrieval/ranking layer; C35 is the query/response integration layer; `SEBrain` is the public facade.

## Repository layout

```text
sebrain/
  __init__.py
  c01.py ... c36.py
  c34_loader.py      # compatibility shim
  c36_rag.py         # compatibility shim
  ui.py

datasets/
  D1 - D5
  D6 - D10
  ...
  D56 - D58
  README.md

tests/
  test_package_smoke.py
  test_knowledge_fabric.py
  test_real_dataset_fabric.py

run_ui.py
requirements.txt
pyproject.toml
```

The repository's current dataset export uses 13 grouped, extensionless files. C34 discovers those files recursively and also supports canonical `D01.jsonl` / `D01.json` style files.

## Install

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Connect the Knowledge Fabric

Place the repository datasets under `./datasets`, then:

```python
from pathlib import Path
from sebrain import Config, SEBrain

with SEBrain(Config(data_dir=Path("./.sebrain"))) as brain:
    report = brain.connect_knowledge_fabric("./datasets")
    print("Files:", report["files"])
    print("Records inserted:", report["records"])
    print("Found:", report["found_dataset_ids"])
    print("Missing:", report["missing_dataset_ids"])

    response = brain.ask(
        "How do I design a database migration safely?",
        top_k=5,
    )
    print(response.to_english())
```

The same connection exposes:

```python
stats = brain.fabric_stats()
catalog = brain.knowledge_catalog()
print(stats["coverage"])
```

## UI

```bash
python -m sebrain.ui --datasets ./datasets --host 127.0.0.1 --port 7860
```

Add `--share` for a temporary Gradio share link.

## Tests

GitHub Actions runs compilation and pytest on Python 3.11 and 3.12.

The tests exercise:

- all C01–C36 module imports;
- C34 loading and idempotence;
- grouped D01–D58 repository export discovery;
- C34 → C36 retrieval → C35 evidence trace → `SEBrain.ask()`.

The CI result for the current commit is the authoritative environment-level validation.

## Dataset completeness

C34 does not invent missing datasets. It explicitly reports expected D01–D58 IDs that are absent from uploaded content.

After connecting, inspect:

```python
brain.fabric_stats()["coverage"]
```

before treating the Knowledge Fabric as complete.
