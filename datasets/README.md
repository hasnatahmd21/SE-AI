# Knowledge Fabric Datasets

This directory contains the D01–D58 Knowledge Fabric used by SE Brain.

## Supported repository layout

The current repository export stores datasets as grouped, extensionless files such as:

- `D1 - D5`
- `D6 - D10`
- `D11 - D15`
- ...
- `D56 - D58`

Each grouped file may contain multiple JSON documents, JSON record arrays, individual records, and human-readable batch/header lines. C34 parses the JSON document stream without requiring the files to be renamed.

Canonical files are also supported:

- `D01.jsonl` through `D58.jsonl`
- `D01.json` through `D58.json`

## Data rules

Record identity is based on `record_id`. Records are retained in the canonical fabric store only once. Same-content duplicate occurrences are audited, and conflicting duplicate identities are preserved in the C34 audit table rather than silently overwritten.

The raw record JSON is preserved for provenance and for downstream engines that need fields beyond the normalized retrieval columns. C34 also stores dataset catalog metadata, source hashes, source locations, content hashes, and a searchable flattened field.

The locked quality-first rule remains authoritative: correctness, scope coverage, evidence quality, implementation usefulness, deduplication, validation, and completeness take priority over record count.

C34 expects D01–D58 and reports missing dataset IDs explicitly instead of pretending the fabric is complete.
