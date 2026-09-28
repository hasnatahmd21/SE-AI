# Knowledge Fabric Datasets

The repository export stores the D01–D58 Knowledge Fabric in grouped source
files:

- `D1 - D5`
- `D6 - D10`
- `D11 - D15`
- `D16 - D17`
- `D18 - D20`
- `D21 - D25`
- `D 26 - D30`
- `D31 - D35`
- `D36 - D40`
- `D41 - D45`
- `D46 - D50`
- `D51 - D55`
- `D56 - D58`

Each grouped source contains JSON dataset documents with manifests and records.
C34 discovers these files recursively and normalizes them into the persistent
SQLite Knowledge Fabric. Canonical `.json` and `.jsonl` sources are also
supported.

## Quality rule

The locked quality-first rule remains authoritative: correctness, scope
coverage, evidence quality, implementation usefulness, deduplication,
validation, and completeness take priority over record count.

## Validation

Run the real repository validation with:

```bash
python scripts/validate_knowledge_fabric.py
```

The validator requires all expected D01–D58 dataset IDs to be discoverable and
requires zero loader errors. It does not impose a fixed record-count target.
