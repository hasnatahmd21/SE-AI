# Changes — Knowledge Fabric Integration

## Added
- C34 persistent Knowledge Fabric loader for D01–D58.
- C35 Brain/Dataset Bridge combining requirement understanding, intent/risk analysis, retrieval, and evidence tracing.
- C36 deterministic local retrieval and field-aware ranking.
- SEBrain.connect_knowledge_fabric(), SEBrain.ask(), SEBrain.fabric_stats(), and SEBrain.knowledge_catalog().
- sebrain.fabric_audit operational audit/query entry point.
- Real Knowledge Fabric integration tests against the repository's uploaded dataset exports.
- UI routed through the canonical SEBrain facade instead of duplicating C34/C35 wiring.
- Backward-compatible sebrain.c34_loader and sebrain.c36_rag import shims.

## Hardening
- Recursive discovery of grouped extensionless dataset exports and canonical JSON/JSONL files.
- JSON document-stream parsing that preserves JSON document boundaries in grouped exports.
- Support for grouped filenames with optional whitespace such as D 26 - D30.
- Direct dataset-ID fallback for canonical D01.jsonl / D01.json files.
- Idempotent record loading based on canonical content hashes.
- Duplicate and conflicting record identity auditing without silent overwrite.
- Persistent source, dataset-catalog, record-occurrence, and progress metadata.
- Raw-record preservation plus flattened searchable text for code-aware lexical retrieval.
- Dataset coverage audit for all expected D01–D58 IDs; missing datasets are reported explicitly.
- SQL wildcard-safe deterministic retrieval matching.
- Evidence trace from C36-ranked records through C35 and the public SEBrain.ask() facade.
- GitHub Actions CI for Python 3.11 and 3.12 with compile and pytest validation.

## Validation notes
- The real repository integration test exercised the large grouped dataset corpus on CI; an early failure exposed a filename-pattern gap (D 26 - D30) and was fixed.
- Dataset completeness remains data-driven: the loader does not fabricate D26 or any other missing dataset ID.