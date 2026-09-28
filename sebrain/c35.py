"""C35 — Brain / Knowledge Fabric Bridge.

C35 is the query-facing integration layer:
query -> requirement understanding -> intent/risk -> C36 retrieval -> 
structured, evidence-linked Brain response.

No external LLM is required for the Knowledge Fabric path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .c01 import Config, SEBrainApp
from .c05 import RequirementParser
from .c06 import IntentContextEngine
from .c34 import FabricRecord, KnowledgeFabricLoader
from .c36 import RAGContext, RAGPipeline


@dataclass(slots=True)
class BrainResponse:
    query: str = ""
    intent: str = ""
    objective: str = ""
    functional: list[str] = field(default_factory=list)
    non_functional: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    risk_level: str = ""
    knowledge: list[FabricRecord] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = ""
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    retrieval_method: str = "lexical"

    def to_english(self) -> str:
        return self.answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "objective": self.objective,
            "functional": self.functional,
            "non_functional": self.non_functional,
            "priorities": self.priorities,
            "risk": self.risk_level,
            "knowledge_count": len(self.knowledge),
            "answer": self.answer,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "sources": self.sources,
            "retrieval_method": self.retrieval_method,
        }


class BrainDatasetBridge:
    """Connect the Brain facade to C34 persistence and C36 retrieval."""

    def __init__(
        self,
        *,
        brain: SEBrainApp,
        loader: KnowledgeFabricLoader,
        rag: RAGPipeline | None = None,
    ):
        self.brain = brain
        self.loader = loader
        self.rag = rag or RAGPipeline(loader)
        self.parser = RequirementParser()
        self.intent_engine = IntentContextEngine()

    def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> BrainResponse:
        query = (query or "").strip()
        if not query:
            return BrainResponse(answer="No query provided.")

        spec = self.parser.parse(query)
        intent_ctx = self.intent_engine.analyze(
            query,
            project_id="bridge",
            spec=spec,
        )

        rag_ctx = self.rag.retrieve(
            query,
            top_k=top_k,
            language=language,
            dataset_id=dataset_id,
            concept=concept,
        )
        return self._build_response(query, spec, intent_ctx, rag_ctx)

    def _build_response(
        self,
        query: str,
        spec: Any,
        intent_ctx: Any,
        rag_ctx: RAGContext,
    ) -> BrainResponse:
        resp = BrainResponse(
            query=query,
            retrieval_method=rag_ctx.retrieval_method,
        )
        if spec:
            resp.objective = spec.objective.text if spec.objective else ""
            resp.functional = [requirement.text for requirement in spec.functional]
            resp.non_functional = [
                requirement.text for requirement in spec.non_functional
            ]
            resp.confidence = spec.confidence.value
        if intent_ctx:
            if intent_ctx.intent:
                resp.intent = intent_ctx.intent.kind.value
            if intent_ctx.risk:
                resp.risk_level = intent_ctx.risk.level.value
            resp.priorities = [
                f"[{priority.level.value}] {priority.text}"
                for priority in intent_ctx.priorities[:5]
            ]

        resp.knowledge = rag_ctx.records
        resp.evidence = [item.to_dict() for item in rag_ctx.items]
        resp.sources = sorted({record.dataset_id for record in resp.knowledge})
        resp.answer = self._format_english(resp, rag_ctx)
        return resp

    def _format_english(
        self, resp: BrainResponse, rag_ctx: RAGContext
    ) -> str:
        lines = [
            "=" * 70,
            "  SE BRAIN — ANALYSIS",
            "=" * 70,
            "",
            "QUERY:",
            f"  {resp.query}",
            "",
            "-" * 70,
            "UNDERSTANDING",
            "-" * 70,
            f"  Intent          : {resp.intent or 'unknown'}",
            f"  Objective       : {resp.objective or '(not detected)'}",
            f"  Overall Conf.   : {resp.confidence or 'unknown'}",
            f"  Risk Level      : {resp.risk_level or 'unknown'}",
            "",
        ]
        if resp.functional:
            lines += [
                "-" * 70,
                f"FUNCTIONAL REQUIREMENTS ({len(resp.functional)})",
                "-" * 70,
            ]
            lines += [
                f"  {index:02d}. {value}"
                for index, value in enumerate(resp.functional, 1)
            ] + [""]
        if resp.non_functional:
            lines += [
                "-" * 70,
                f"NON-FUNCTIONAL REQUIREMENTS ({len(resp.non_functional)})",
                "-" * 70,
            ]
            lines += [
                f"  {index:02d}. {value}"
                for index, value in enumerate(resp.non_functional, 1)
            ] + [""]

        if resp.priorities:
            lines += [
                "-" * 70,
                "PRIORITY ITEMS",
                "-" * 70,
                *[f"  • {value}" for value in resp.priorities],
                "",
            ]

        lines += [
            "-" * 70,
            f"KNOWLEDGE FROM DATASETS ({len(resp.knowledge)} records)",
            "-" * 70,
        ]
        if not resp.knowledge:
            lines += ["  No relevant knowledge found in datasets.", ""]
        else:
            for index, record in enumerate(resp.knowledge, 1):
                lines += [
                    "",
                    f"  [{index}] {record.dataset_id} › {record.concept or record.topic}",
                ]
                evidence = resp.evidence[index - 1] if index - 1 < len(resp.evidence) else {}
                if evidence:
                    lines.append(
                        f"      Retrieval  : score={evidence.get('score', 0):.2f}; "
                        f"matched={', '.join(evidence.get('matched_terms', [])) or 'none'}"
                    )
                if record.language:
                    lines.append(f"      Language   : {record.language}")
                if record.framework:
                    lines.append(f"      Framework  : {record.framework}")
                if record.knowledge_type:
                    lines.append(f"      Type       : {record.knowledge_type}")
                if record.question:
                    lines.append(f"      Q: {record.question}")
                if record.answer:
                    lines.append("      A:")
                    lines += [
                        f"        {value}"
                        for value in self._wrap_text(record.answer, 60)[:15]
                    ]
                if record.explanation:
                    lines.append("      Explanation:")
                    lines += [
                        f"        {value}"
                        for value in self._wrap_text(record.explanation, 60)[:8]
                    ]
                if record.tags:
                    lines.append(f"      Tags       : {', '.join(record.tags)}")

        lines += ["", "-" * 70, "SUMMARY", "-" * 70]
        if resp.knowledge:
            lines += [
                f"  ✓ Found {len(resp.knowledge)} relevant records in "
                f"{len(resp.sources)} dataset(s).",
                f"  ✓ Data sources: {', '.join(resp.sources)}",
                f"  ✓ Retrieval: {resp.retrieval_method}",
                f"  ✓ Confidence: {resp.confidence or 'unknown'}",
                f"  ✓ Evidence trace entries: {len(resp.evidence)}",
            ]
        else:
            lines.append("  ✗ No matching knowledge found.")
        lines += ["", "=" * 70]
        return "\n".join(lines)

    @staticmethod
    def _wrap_text(text: str, width: int = 60) -> list[str]:
        lines: list[str] = []
        for paragraph in (text or "").splitlines() or [""]:
            words = paragraph.split()
            current = ""
            for word in words:
                if len(word) > width:
                    if current:
                        lines.append(current)
                        current = ""
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                    current = word
                    continue
                candidate = f"{current} {word}".strip()
                if len(candidate) <= width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            elif not words:
                lines.append("")
        return lines

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> str:
        return self.answer(
            query,
            top_k=top_k,
            language=language,
            dataset_id=dataset_id,
            concept=concept,
        ).to_english()

    def search_only(
        self,
        query: str,
        *,
        top_k: int = 5,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> list[FabricRecord]:
        return self.rag.retrieve(
            query,
            top_k=top_k,
            language=language,
            dataset_id=dataset_id,
            concept=concept,
        ).records


def create_bridge(
    config: Config | None = None,
    datasets_dir: str = "./datasets",
) -> tuple[
    SEBrainApp,
    KnowledgeFabricLoader,
    BrainDatasetBridge,
    dict[str, Any],
]:
    brain = SEBrainApp(config=config)
    brain.start()
    loader = KnowledgeFabricLoader(brain.storage, datasets_dir)
    report = loader.load_all_datasets()
    return brain, loader, BrainDatasetBridge(brain=brain, loader=loader), report
