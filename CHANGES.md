# Changes — Knowledge Fabric Integration

## Added
- C34 persistent Knowledge Fabric Loader for D01–D58 JSONL datasets.
- C36 deterministic local RAG/retrieval pipeline.
- C35 Brain/Dataset Bridge combining requirement parsing, intent/risk analysis, and retrieval.
- `SEBrain.connect_knowledge_fabric()`, `SEBrain.ask()`, and `SEBrain.fabric_stats()` facade methods.
- `sebrain.ui` dark Gradio interface using the same C35 bridge as the programmatic API.
- Backward-compatible `sebrain.c34_loader` and `sebrain.c36_rag` import shims.

## Hardening
- Idempotent record loading.
- Duplicate-record detection within a JSONL file.
- Explicit malformed JSON/object validation and error reporting.
- Safe schema upgrades for databases created by the earlier C34 draft.
- Source file/line and content-hash provenance metadata.
- Batch parameter validation and deterministic ordering.
- Field-aware retrieval scoring and deterministic tie-breaking.
