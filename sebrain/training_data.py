"""Deterministic Knowledge Fabric -> generic training-data exporter."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .c34 import FabricRecord


@dataclass(frozen=True, slots=True)
class TrainingExample:
    record_id: str
    dataset_id: str
    instruction: str
    output: str
    source_file: str
    content_hash: str
    split: str
    execution_status: str
    validation_status: str
    evidence_level: str
    provenance: dict
    relationships: list
    training_eligible: bool

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "instruction": self.instruction,
            "output": self.output,
            "split": self.split,
            "execution_status": self.execution_status,
            "validation_status": self.validation_status,
            "evidence_level": self.evidence_level,
            "provenance": {
                **dict(self.provenance),
                "source_file": self.source_file,
                "content_hash": self.content_hash,
            },
            "relationships": list(self.relationships),
            "training_eligible": self.training_eligible,
        }


class TrainingDatasetExporter:
    """Convert validated FabricRecords into deterministic generic JSONL.

    This is preparation, not model training. Records are never invented and
    provenance remains attached to every emitted example.
    """

    def __init__(self, *, train_ratio: float = 0.8, validation_ratio: float = 0.1):
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 <= validation_ratio < 1:
            raise ValueError("validation_ratio must be >= 0")
        if train_ratio + validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be < 1")
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio

    def convert(self, records: Iterable[FabricRecord]) -> list[TrainingExample]:
        out: list[TrainingExample] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            instruction = (record.question or record.concept or record.topic).strip()
            output = (record.answer or record.explanation).strip()
            if not instruction or not output:
                continue
            identity = (record.record_id, record.content_hash)
            if identity in seen:
                continue
            seen.add(identity)
            digest = hashlib.sha256(
                f"{record.record_id}\0{record.content_hash}".encode("utf-8")
            ).hexdigest()
            bucket = int(digest[:8], 16) / 0x100000000
            if bucket < self.train_ratio:
                split = "train"
            elif bucket < self.train_ratio + self.validation_ratio:
                split = "validation"
            else:
                split = "test"
            raw = record.raw if isinstance(record.raw, dict) else {}
            execution_status = str(raw.get("execution_status", "")).strip()
            validation_status = str(raw.get("validation_status", "")).strip()
            evidence_level = str(raw.get("evidence_level", "")).strip()
            provenance = raw.get("provenance", {})
            if not isinstance(provenance, dict):
                provenance = {"source": str(provenance)}
            relationships = raw.get("relationships", raw.get("related_records", []))
            if not isinstance(relationships, list):
                relationships = [relationships]
            # Preparation must never imply that unexecuted or merely
            # illustrative material has been empirically validated. Keep the
            # record for audit/review, but make eligibility explicit.
            training_eligible = (
                bool(validation_status)
                and validation_status.upper() not in {
                    "ILLUSTRATIVE",
                    "REQUIRES_TARGET_VALIDATION",
                    "UNVALIDATED",
                    "UNKNOWN",
                }
                and execution_status.upper() not in {
                    "NOT_EXECUTED",
                    "UNEXECUTED",
                    "UNKNOWN",
                }
            )
            out.append(TrainingExample(
                record_id=record.record_id,
                dataset_id=record.dataset_id,
                instruction=instruction,
                output=output,
                source_file=record.source_file,
                content_hash=record.content_hash,
                split=split,
                execution_status=execution_status,
                validation_status=validation_status,
                evidence_level=evidence_level,
                provenance=provenance,
                relationships=relationships,
                training_eligible=training_eligible,
            ))
        return sorted(out, key=lambda x: x.record_id)

    def write_jsonl(self, records: Iterable[FabricRecord], path: str | Path) -> dict[str, int]:
        examples = self.convert(records)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return {
            "total": len(examples),
            "train": sum(x.split == "train" for x in examples),
            "validation": sum(x.split == "validation" for x in examples),
            "test": sum(x.split == "test" for x in examples),
            "training_eligible": sum(x.training_eligible for x in examples),
            "training_ineligible": sum(not x.training_eligible for x in examples),
        }
