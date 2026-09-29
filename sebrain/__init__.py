"""
================================================================================
SE BRAIN — Autonomous Software Engineering Brain
Top-level package
================================================================================

This package is the repaired, de-duplicated, properly-modularized version of
the original single-file `SE_brain_.py` dump. It contains 36 engines
(c01 .. c36, with C34–C36 forming the Knowledge Fabric layer) each in its own
file/namespace:

    c01  Core Foundation              c18  Test Execution Engine
    c02  Software Ontology Engine     c19  Debugging Engine
    c03  Knowledge Retrieval Engine   c20  Code Repair Engine
    c04  Persistent Memory            c21  Critic Engine
    c05  Requirement Understanding    c22  Verification Engine
    c06  Intent & Context Engine      c23  Security Analysis Engine
    c07  Reasoning Engine             c24  Performance Analysis Engine
    c08  Planning Engine              c25  Existing Codebase Understanding
    c09  Technology Selection Engine  c26  Experience Extraction
    c10  Architecture Reasoning       c27  Learning / Adaptation Engine
    c11  Agent Orchestrator           c28  Meta-Reasoning
    c12  Specialist Agent Framework   c29  Self-Improvement Governance
    c13  Code Representation Engine   c30  Cross-Language Reasoning
    c14  Code Synthesis Engine        c31  Cross-Project Knowledge Engine
    c15  Repository / Project Builder c32  Evaluation Laboratory
    c16  Execution Sandbox            c33  Failure Analysis Laboratory
    c17  Test Generation Engine
    c34  Knowledge Fabric Loader
    c35  Brain / Dataset Bridge
    c36  Local RAG Retrieval Pipeline

Every engine is independently usable:

    from sebrain.c08 import Planner
    from sebrain import c08          # same thing

or runnable/self-testable on its own:

    python -m sebrain.c08            # demo
    python -m sebrain.c08 --test     # self-tests

`c01` uses pydantic, pydantic-settings, and structlog; `c18` invokes
pytest for project test execution. See pyproject.toml. The
remaining engine implementations use the Python standard library.

This file additionally provides `SEBrain`, a small orchestrator that starts
the shared storage/lifecycle (C01), wires up the shared Ontology (C02) and
MemoryStore (C04) that most other engines are designed to read/write
through, and exposes `ingest_dataset()` so external knowledge can be
attached to the brain before you hand it a task. Wiring the 36 engines'
*business logic* together beyond that is intentionally left to the caller:
each engine has its own "Depends on C.." list in its module docstring, and
composing them (e.g. Planning -> Code Synthesis -> Test Generation -> Test
Execution -> Debugging -> Code Repair) is a task-specific pipeline, not a
single fixed wiring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from .c01 import (
    Config,
    SEBrainApp,
    VERSION,
    __version__,
    load_config,
    NotInitializedError,
)
from .c02 import (
    Ontology,
    Entity,
    EntityKind,
    RelationKind,
    Provenance,
    ProvenanceType,
)
from .c04 import (
    MemoryStore,
    MemoryEntry,
    MemoryKind,
    MemoryScope,
)
from .c34 import KnowledgeFabricLoader, FabricRecord
from .c36 import RAGPipeline, RAGContext
from .c35 import BrainDatasetBridge, BrainResponse
from .model_gateway import (
    CallableModelGateway,
    ModelGateway,
    ModelRequest,
    ModelResponse,
    ModelUnavailableError,
    UnavailableModelGateway,
)
from .training_data import TrainingDatasetExporter, TrainingExample

__all__ = [
    "SEBrain",
    "Config",
    "SEBrainApp",
    "Ontology",
    "EntityKind",
    "RelationKind",
    "Provenance",
    "ProvenanceType",
    "MemoryStore",
    "MemoryKind",
    "MemoryScope",
    "VERSION",
    "__version__",
    "KnowledgeFabricLoader",
    "FabricRecord",
    "RAGPipeline",
    "RAGContext",
    "BrainDatasetBridge",
    "BrainResponse",
    "ModelGateway",
    "ModelRequest",
    "ModelResponse",
    "ModelUnavailableError",
    "UnavailableModelGateway",
    "CallableModelGateway",
    "TrainingDatasetExporter",
    "TrainingExample",
]


class SEBrain:
    """Top-level facade wiring the SE Brain's foundational layer together.

    Usage:
        with SEBrain() as brain:
            brain.ingest_dataset(my_records, dataset_name="past_incidents")
            ...  # hand a task to whichever engine(s) you need, e.g.
            from sebrain.c08 import Planner
            plan = Planner(memory=brain.memory, ontology=brain.ontology).plan(...)

    `brain.memory` (MemoryStore) and `brain.ontology` (Ontology) are the two
    shared substrates nearly every other engine accepts as constructor
    arguments — pass them straight through.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.app: SEBrainApp = SEBrainApp(config=config) if config is not None else SEBrainApp()
        self.memory: MemoryStore | None = None
        self.ontology: Ontology | None = None
        self.fabric: KnowledgeFabricLoader | None = None
        self.bridge: BrainDatasetBridge | None = None
        self.model_gateway: ModelGateway | None = None

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> "SEBrain":
        self.app.start()
        self.memory = MemoryStore(self.app.storage)
        self.ontology = Ontology(self.app.storage)
        return self

    def stop(self) -> None:
        self.app.stop()
        self.memory = None
        self.ontology = None
        self.fabric = None
        self.bridge = None
        self.model_gateway = None

    def __enter__(self) -> "SEBrain":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def health(self) -> Any:
        return self.app.health()

    # ---- Model boundary ------------------------------------------------
    def set_model_gateway(self, gateway: ModelGateway) -> None:
        """Attach a real inference adapter without coupling the Brain to a provider."""
        if not isinstance(gateway, ModelGateway):
            raise TypeError("gateway must implement ModelGateway")
        self.model_gateway = gateway

    def generate(self, prompt: str, *, system: str = "", context: Iterable[dict] = ()) -> ModelResponse:
        """Generate through the configured model boundary; never fabricate output."""
        if self.model_gateway is None:
            raise ModelUnavailableError(
                "No model gateway is configured. Attach a trained/local model adapter first."
            )
        request = ModelRequest(
            prompt=prompt,
            system=system,
            context=tuple(dict(item) for item in context),
        )
        return self.model_gateway.generate(request)

    # ---- Knowledge Fabric integration ---------------------------------
    def connect_knowledge_fabric(self, datasets_dir: str | Path = "./datasets") -> dict[str, Any]:
        if self.app.storage is None:
            raise NotInitializedError("SEBrain is not started")
        self.fabric = KnowledgeFabricLoader(self.app.storage, datasets_dir)
        report = self.fabric.load_all_datasets()
        self.bridge = BrainDatasetBridge(brain=self.app, loader=self.fabric)
        return report

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> BrainResponse:
        """Ask the connected Brain using the C34 -> C36 -> C35 pipeline."""
        if self.bridge is None:
            raise NotInitializedError(
                "Knowledge Fabric is not connected — call connect_knowledge_fabric() first"
            )
        return self.bridge.answer(
            query,
            top_k=top_k,
            language=language,
            dataset_id=dataset_id,
            concept=concept,
        )

    def fabric_stats(self) -> dict[str, Any]:
        if self.fabric is None:
            raise NotInitializedError("Knowledge Fabric is not connected")
        return self.fabric.stats()

    def knowledge_catalog(self) -> list[dict[str, Any]]:
        """Return the persistent catalog populated by C34 (57 planned datasets: D01-D58, excluding D26)."""
        if self.fabric is None:
            raise NotInitializedError("Knowledge Fabric is not connected")
        return self.fabric.dataset_catalog()

    # ---- dataset ingestion --------------------------------------------
    def ingest_dataset(
        self,
        records: Iterable[dict],
        *,
        dataset_name: str,
        entity_kind: "EntityKind | str | None" = None,
        key_field: str | None = None,
        name_field: str = "name",
    ) -> dict:
        """Attach an external dataset so the brain can draw on it as knowledge.

        Every record (a plain dict) becomes a durable, globally-scoped
        MemoryStore entry (`MemoryKind.LONG_TERM`), tagged with
        ``dataset_name`` so it can be retrieved later via::

            brain.memory.find(kind=MemoryKind.LONG_TERM, tags=["dataset_name"])

        If ``entity_kind`` is given (e.g. ``EntityKind.BUG`` for a dataset of
        known bugs, ``EntityKind.REQUIREMENT`` for a requirements corpus),
        every record is *also* added to the Ontology as a structured entity
        of that kind, so engines that reason over the ontology graph
        (Planning, Debugging, Existing-Codebase-Understanding, ...) can use
        it directly rather than only via free-text memory lookup.

        Args:
            records: iterable of plain dicts (e.g. rows from a CSV/JSON
                dataset you loaded yourself — this method does not read
                files, it only ingests already-parsed records).
            dataset_name: short identifier for this dataset; used as a tag
                and as part of the memory key when ``key_field`` is unset.
            entity_kind: optional EntityKind to also create ontology
                entities for each record.
            key_field: field in each record to use as the memory key
                (must be unique per record). If omitted, records are keyed
                ``f"{dataset_name}:{index}"``.
            name_field: field to use as the ontology entity's display name
                when ``entity_kind`` is set (defaults to ``"name"``, falls
                back to the memory key if the field is missing).

        Returns:
            ``{"memory_entries": <int>, "ontology_entities": <int>}``
        """
        if self.memory is None or self.ontology is None:
            raise NotInitializedError(
                "SEBrain is not started — call .start() or use 'with SEBrain() as brain:'"
            )

        mem_count = 0
        ent_count = 0
        provenance = Provenance(
            source=dataset_name,
            source_type=ProvenanceType.IMPORT,
            notes="ingested via SEBrain.ingest_dataset",
        )

        for i, record in enumerate(records):
            key = str(record[key_field]) if key_field else f"{dataset_name}:{i}"
            self.memory.upsert(
                MemoryKind.LONG_TERM,
                key,
                content=dict(record),
                scope_type=MemoryScope.GLOBAL,
                tags=["dataset", dataset_name],
                provenance=provenance,
            )
            mem_count += 1

            if entity_kind is not None:
                name = str(record.get(name_field, key))
                self.ontology.add(
                    entity_kind,
                    name,
                    attributes=dict(record),
                    tags=["dataset", dataset_name],
                    provenance=provenance,
                )
                ent_count += 1

        return {"memory_entries": mem_count, "ontology_entities": ent_count}
