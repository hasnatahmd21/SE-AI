# C34 — Knowledge Fabric Structure

C34 is the **single authoritative Knowledge Fabric loader**.

## Canonical runtime layout

```text
datasets/
├── D1 - D5
├── D6 - D10
├── D11 - D15
├── D16 - D17
├── D18 - D20
├── D21 - D25
├── D 26 - D30   # export container only; D26 is NOT part of the plan
├── D31 - D35
├── D36 - D40
├── D41 - D45
├── D46 - D50
├── D51 - D55
└── D56 - D58

sebrain/
├── c34.py          # AUTHORITATIVE implementation
├── c34_loader.py   # compatibility import shim only
├── c36.py          # retrieval layer; imports C34 from c34.py
└── c35.py          # Brain bridge; imports C34 from c34.py
```

## Dataset contract

The planned fabric contains **57 datasets**:

- D01–D25
- D27–D58
- D26 is intentionally excluded.

Grouped source filenames are transport/storage organization only. C34 resolves the dataset ID from record data/context/manifest and never treats the D26-D30 container name as proof that D26 exists.

## Persistence contract

C34 uses the Brain's shared SQLite storage abstraction. It owns these Knowledge Fabric tables:

- `fabric_records`
- `fabric_progress`
- `fabric_datasets`
- `fabric_sources`
- `fabric_record_audit`

SQLite database files are runtime artifacts and are intentionally ignored by Git. The repository source of truth is the dataset export plus C34 code, not a checked-in SQLite database.

## Import contract

Use:

```python
from sebrain.c34 import KnowledgeFabricLoader, FabricRecord
```

The older `sebrain.c34_loader` path remains supported for compatibility, but `c34_loader.py` must not contain a second implementation.

## Data flow

```text
grouped dataset sources
        ↓
      C34
parse → validate → normalize → provenance/audit → SQLite
        ↓
      C36
deterministic retrieval/ranking
        ↓
      C35
Brain response + evidence
        ↓
    SEBrain.ask()
```

C34 is deliberately kept separate from model training and model inference. Training artifacts do not belong inside the C34 loader.