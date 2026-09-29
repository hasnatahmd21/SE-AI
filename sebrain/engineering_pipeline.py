"""High-level engineering workflow wiring the canonical SE Brain engines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .c05 import RequirementParser, RequirementSpec
from .c06 import IntentContext, IntentContextEngine
from .c08 import Plan, Planner
from .c11 import AgentRegistry, Orchestrator, OrchestrationResult
from .c36 import RAGContext, RAGPipeline


@dataclass(slots=True)
class EngineeringAnalysis:
    spec: RequirementSpec
    intent: IntentContext
    plan: Plan
    knowledge: RAGContext | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "intent": self.intent.to_dict(),
            "plan": self.plan.to_dict(),
            "knowledge": self.knowledge.to_dict() if self.knowledge else None,
        }


class EngineeringPipeline:
    """Connect requirement understanding through planning and execution.

    Execution remains explicit: callers provide the governed C11 agent registry.
    This prevents the facade from inventing unsafe worker capabilities while
    still giving the Brain one canonical end-to-end control path.
    """

    def __init__(
        self,
        *,
        memory=None,
        ontology=None,
        rag: RAGPipeline | None = None,
        parser: RequirementParser | None = None,
    ) -> None:
        self.parser = parser or RequirementParser()
        self.intent_engine = IntentContextEngine(
            memory=memory, ontology=ontology, parser=self.parser
        )
        self.planner = Planner()
        self.rag = rag

    def analyze(
        self,
        text: str,
        *,
        project_id: str = "",
        task_id: str | None = None,
        top_k: int = 5,
    ) -> EngineeringAnalysis:
        spec = self.parser.parse(text)
        intent = self.intent_engine.analyze(
            text, project_id=project_id, task_id=task_id, spec=spec
        )
        plan = self.planner.plan(spec, intent, project_id=project_id)
        knowledge = (
            self.rag.retrieve(text, top_k=top_k)
            if self.rag is not None and text.strip()
            else None
        )
        return EngineeringAnalysis(
            spec=spec, intent=intent, plan=plan, knowledge=knowledge
        )

    def execute(
        self,
        analysis: EngineeringAnalysis,
        registry: AgentRegistry,
        *,
        project_id: str = "",
        orchestrator: Orchestrator | None = None,
    ) -> OrchestrationResult:
        runner = orchestrator or Orchestrator(registry)
        return runner.run(analysis.plan, project_id=project_id or analysis.plan.project_id)
