# Kaggle Readiness — SE Brain

> **CURRENT STATUS: BLOCKED — NOT READY FOR KAGGLE**
>
> The repository CI currently verifies C01–C36 compilation/tests successfully,
> but the real D01–D58 fabric validation reports **D26 missing**. The grouped
> source datasets/D 26 - D30 currently contains D27–D30 records and no D26
> records. No synthetic D26 records are created to satisfy coverage.
>
> This is a real data-coverage blocker. Kaggle transition must wait until the
> authoritative D26 dataset export is restored/added and the complete
> C34 → C36 → C35 → SEBrain validation passes with zero loader errors and
> complete D01–D58 coverage.

## Purpose

This repository is prepared for the next-stage Kaggle environment without changing the canonical Brain architecture.

## Repository components

- `sebrain/`: canonical C01–C36 package.
- `datasets/`: grouped D01–D58 Knowledge Fabric source exports.
- `tests/`: unit, integration, and Knowledge Fabric regression tests.
- `scripts/validate_knowledge_fabric.py`: real D01–D58 validation through C34 → C36 → C35 → SEBrain.
- `.github/workflows/ci.yml`: Python 3.11/3.12 CI plus real-fabric validation.
- No generated `.sebrain_validation/` runtime database is required in Git.

## Kaggle setup

From a Kaggle notebook, run:

```bash
python -m pip install -e .
python -m compileall -q sebrain
python -m pytest -q
python scripts/validate_knowledge_fabric.py
```

The Knowledge Fabric is local and deterministic. The C34 loader creates its SQLite state under the configured runtime data directory; the source datasets remain authoritative.

## First smoke test

```python
from pathlib import Path
from sebrain import Config, SEBrain

with SEBrain(Config(data_dir=Path("/kaggle/working/.sebrain"))) as brain:
    report = brain.connect_knowledge_fabric(Path("/kaggle/input/se-ai/datasets"))
    print(report)
    response = brain.ask("How should I design a safe database migration?", top_k=5)
    print(response.to_english())
```

Adjust the dataset path to the actual Kaggle dataset mount point.

## Validation standard

Kaggle validation must establish:

1. C01–C36 imports and compilation succeed.
2. The complete test suite passes.
3. D01–D58 coverage is complete.
4. The real grouped dataset exports load without errors.
5. Retrieval returns evidence linked to source records.
6. No fabricated records are introduced to satisfy a fixed count.

Record count is not a substitute for quality, coverage, correctness, or evidence.
