"""Verify the repository C34 structure and Knowledge Fabric contract.

This is a structural/integration check. It uses a temporary SQLite database by
default, so running it never modifies the repository's persistent .sebrain DB.
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path

EXPECTED_DATASETS = tuple(
    f"D{i:02d}" for i in range(1, 59) if i != 26
)
EXPECTED_TABLES = {
    "fabric_records",
    "fabric_progress",
    "fabric_datasets",
    "fabric_sources",
    "fabric_record_audit",
}

def fail(message: str) -> None:
    raise SystemExit(f"C34 STRUCTURE CHECK FAILED: {message}")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    c34_path = root / "sebrain" / "c34.py"
    shim_path = root / "sebrain" / "c34_loader.py"
    c36_path = root / "sebrain" / "c36.py"
    c35_path = root / "sebrain" / "c35.py"

    for path in (c34_path, shim_path, c36_path, c35_path):
        if not path.is_file():
            fail(f"missing required file: {path.relative_to(root)}")

    c34_text = c34_path.read_text(encoding="utf-8")
    shim_text = shim_path.read_text(encoding="utf-8")
    c36_text = c36_path.read_text(encoding="utf-8")
    c35_text = c35_path.read_text(encoding="utf-8")

    if "class KnowledgeFabricLoader" not in c34_text:
        fail("authoritative KnowledgeFabricLoader is missing from sebrain/c34.py")
    if "class KnowledgeFabricLoader" in shim_text:
        fail("c34_loader.py contains a duplicate loader implementation")
    if "from .c34 import" not in shim_text:
        fail("c34_loader.py is not a compatibility import shim")
    if "from .c34 import" not in c36_text:
        fail("C36 is not importing the canonical C34 implementation")
    if "from .c34 import" not in c35_text:
        fail("C35 is not importing the canonical C34 implementation")

    if len(EXPECTED_DATASETS) != 57 or "D26" in EXPECTED_DATASETS:
        fail("dataset contract is not D01-D58 excluding D26")

    c01 = importlib.import_module("sebrain.c01")
    c34 = importlib.import_module("sebrain.c34")
    shim = importlib.import_module("sebrain.c34_loader")

    if shim.KnowledgeFabricLoader is not c34.KnowledgeFabricLoader:
        fail("c34_loader compatibility import is not identical to c34.KnowledgeFabricLoader")

    with tempfile.TemporaryDirectory(prefix="sebrain_c34_check_") as tmp:
        storage = c01.SQLiteStorage(Path(tmp) / "c34_check.sqlite3")
        storage.initialize()
        try:
            loader = c34.KnowledgeFabricLoader(storage, args.datasets)
            rows = storage.query("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fabric_%' ORDER BY name")
            tables = {row["name"] for row in rows}
            missing_tables = EXPECTED_TABLES - tables
            if missing_tables:
                fail(f"missing C34 tables: {sorted(missing_tables)}")

            if args.datasets.exists():
                report = loader.load_all_datasets()
                if report["errors"]:
                    fail(f"loader reported {len(report['errors'])} errors")
                missing = report["missing_dataset_ids"]
                if missing:
                    fail(f"missing planned datasets: {missing}")
                stats = loader.stats()
                if stats["coverage"]["dataset_count"] != 57:
                    fail("expected 57 datasets, got " f"{stats['coverage']['dataset_count']}")
        finally:
            storage.shutdown()

    print("C34 STRUCTURE CHECK: PASS")
    print("  authoritative: sebrain/c34.py")
    print("  compatibility: sebrain/c34_loader.py")
    print("  C35/C36 imports: canonical")
    print("  planned datasets: 57 (D26 excluded)")
    print("  schema tables: 5")
    if args.datasets.exists():
        print("  real dataset export: validated")
    else:
        print("  real dataset export: not present; structural check only")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())