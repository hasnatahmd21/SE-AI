# SE Brain — Audited + Knowledge Fabric Integrated

SE Brain is an autonomous software-engineering foundation with C01–C33 engines plus the Knowledge Fabric integration layer:

- **C34 — Knowledge Fabric Loader:** loads D01–D58 JSONL into the Brain's shared SQLite storage, preserves raw records/provenance, reports malformed records, supports idempotent reloads, and upgrades the earlier C34 schema safely.
- **C36 — Retrieval Pipeline:** deterministic local lexical retrieval with field-aware scoring. No LLM or paid API is required.
- **C35 — Brain/Dataset Bridge:** combines C05 requirement understanding + C06 intent/risk analysis + C36 retrieval and returns structured English output.
- **UI:** `python -m sebrain.ui --datasets ./datasets`

## Connect datasets programmatically

```python
from pathlib import Path
from sebrain import SEBrain, Config

with SEBrain(Config(data_dir=Path('./.sebrain'))) as brain:
    report = brain.connect_knowledge_fabric('./datasets')
    print(report)
    print(brain.ask('How do I build a FastAPI endpoint?').to_english())
```

The loader expects JSONL files such as `D01.jsonl` … `D58.jsonl`. Each line must be a JSON object. Common fields such as `record_id`, `topic`, `concept`, `knowledge_type`, `question`, `answer`, `explanation`, `language`, `framework`, `version`, and `tags` are preserved; the complete original record is retained in `raw_json`.

## UI

Install dependencies from `requirements.txt`, place datasets in `./datasets`, then:

```bash
python -m sebrain.ui --datasets ./datasets --host 127.0.0.1 --port 7860
```

For a temporary Gradio share link, add `--share`.

## Important dependency note

Core C01 requires `pydantic`, `pydantic-settings`, and `structlog`. The UI additionally requires `gradio`. No external LLM/API is required by C34–C36.
