# Kaggle Readiness — SE Brain

> **CURRENT STATUS: IN PROGRESS — D26 INTENTIONALLY UNPLANNED**
>
> The repository contains the intended Knowledge Fabric datasets. D26 is not
> part of the dataset plan; the grouped `D 26 - D30` source contains D27–D30
> records. The system therefore must not fabricate D26 records or report D26 as
> a missing expected dataset.
>
> Kaggle transition remains gated on the broader repository completion and
> hardening checks, including zero loader errors, real retrieval, Brain wiring,
> UI/API validation, integration tests, and end-to-end verification.

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
3. Coverage is complete for all 57 planned dataset IDs (D01–D58, intentionally excluding D26).
4. The real grouped dataset exports load without errors.
5. Retrieval returns evidence linked to source records.
6. No fabricated records are introduced to satisfy a fixed count.

Record count is not a substitute for quality, coverage, correctness, or evidence.

## Training readiness hardening

The repository now includes a training-readiness gate, a two-step Qwen2.5-0.5B smoke configuration, model/LoRA compatibility preflight, optional 4-bit QLoRA support, final LoRA adapter load validation, reproducible artifact packaging with SHA-256 manifest, and an optional Hugging Face model-artifact backup. The full Kaggle flow refuses the placeholder base model, prepares and revalidates eligible training data, runs a dry-run preflight, trains, packages the completed adapter, verifies registered artifact hashes, and optionally uploads the adapter when `HF_REPO_ID` and `HF_TOKEN` are supplied as environment secrets.

The smoke configuration is only an infrastructure test; it is not the final SE Brain base model or final training run.
