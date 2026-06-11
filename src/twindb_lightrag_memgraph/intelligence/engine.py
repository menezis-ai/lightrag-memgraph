"""
twin_rag_intelligence/engine.py
================================
Single entry point for the package.
Orchestrates the full ReAct + AFFINE pipeline.

Pattern: Facade over react/ and features/ modules.
The Agentic Platform (L4) only needs to instantiate TwinRAGEngine
and call aquery(). Everything else is encapsulated.
"""

import asyncio
import logging
import os
from typing import Optional

from lightrag import LightRAG, QueryParam

from .._constants import MEMGRAPH_WORKSPACE_ENV
from .config import TwinRAGConfig
from .features.cognitive_reranker import CognitiveReranker
from .features.feedback import FeedbackStore
from .features.intent_classifier import IntentClassifier
from .features.query_expander import QueryExpander
from .models.schemas import IntentType, QueryResult, QueryTrace
from .ontology.config import OntologyConfig, load_ontology_config
from .react.act import SearchEngine
from .react.observe import SynthesisEngine
from .react.reason import ReasoningEngine

logger = logging.getLogger("twin_rag_intelligence")


class TwinRAGEngine:
    """
    RAG Intelligence engine for TwinDB.

    Encapsulates the full pipeline:
    F05 (Intent) -> REASON (Coref + F03 Expansion) -> ACT (Search + F04 Rerank) -> OBSERVE (Synthesis)

    Usage:
        engine = TwinRAGEngine(config)
        result = await engine.aquery("Pourquoi ORA-04030 ?", history=[...], workspace="cib")
    """

    def __init__(self, config: Optional[TwinRAGConfig] = None) -> None:
        self.config = config or TwinRAGConfig()

        # Sub-modules
        self.intent_classifier = IntentClassifier(self.config)
        self.reasoning = ReasoningEngine(self.config)
        self.search = SearchEngine(self.config)
        self.synthesis = SynthesisEngine(self.config)
        self.reranker = CognitiveReranker(self.config)
        self.expander = QueryExpander(self.config)
        self.feedback = FeedbackStore(self.config)

        # Ontology config (opt-in via ontology.json)
        self.ontology_config: Optional[OntologyConfig] = load_ontology_config()

        # LightRAG instances (lazy, per workspace)
        self._rag_instances: dict[str, LightRAG] = {}

    def _get_rag(self, workspace: str) -> LightRAG:
        """Return (or create) a LightRAG instance for a given workspace."""
        if workspace not in self._rag_instances:
            from .. import register

            register()

            os.environ[MEMGRAPH_WORKSPACE_ENV] = workspace

            self._rag_instances[workspace] = LightRAG(
                working_dir=f"/tmp/lightrag_{workspace}",
                kv_storage="MemgraphKVStorage",
                vector_storage="MemgraphVectorDBStorage",
                doc_status_storage="MemgraphDocStatusStorage",
                graph_storage="MemgraphStorage",
            )
        return self._rag_instances[workspace]

    async def aquery(
        self,
        question: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        workspace: str = "commons",
        workspaces_publics: Optional[list[str]] = None,
        user_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Main entry point -- Full ReAct + AFFINE pipeline.

        Args:
            question: User question in natural language.
            conversation_history: List of messages [{role, content}], max N recent.
            workspace: Agent's private workspace (e.g., "cib", "bp2i").
            workspaces_publics: List of public workspaces to query (e.g., ["commons"]).
            user_id: User identifier for feedback.

        Returns:
            QueryResult containing answer, citations, trace, intent.
        """
        trace = QueryTrace(question=question, workspace=workspace, user_id=user_id)
        trace.start()
        history = (conversation_history or [])[-self.config.conversation_memory_depth :]
        workspaces_publics = workspaces_publics or ["commons"]

        # ---- STEP 0: INTENT CLASSIFICATION (F05) ----
        if self.config.enable_oos_detection:
            intent_result = await self.intent_classifier.classify(question)
            trace.intent = intent_result
            logger.info(
                "Intent: %s (conf: %.2f) -- %s",
                intent_result.intent.value,
                intent_result.confidence,
                intent_result.reason,
            )

            if (
                intent_result.intent == IntentType.OUT_OF_SCOPE
                and intent_result.confidence >= self.config.oos_confidence_threshold
            ):
                trace.stop(early_exit="OOS")
                return QueryResult(
                    answer=self._scripted_response(intent_result.intent),
                    citations=[],
                    trace=trace,
                    intent=intent_result,
                )

            if intent_result.intent == IntentType.GREETING:
                trace.stop(early_exit="GREETING")
                return QueryResult(
                    answer=self._scripted_response(IntentType.GREETING),
                    citations=[],
                    trace=trace,
                    intent=intent_result,
                )

            if intent_result.intent == IntentType.MALICIOUS:
                logger.warning("Malicious attempt detected: %s", question)
                trace.stop(early_exit="MALICIOUS")
                return QueryResult(
                    answer=self._scripted_response(IntentType.MALICIOUS),
                    citations=[],
                    trace=trace,
                    intent=intent_result,
                )

        # ---- STEP 1: REASON (Coreference + Query Expansion F03) ----
        reasoning_result = await self.reasoning.analyze(question, history)
        search_query = reasoning_result.search_query
        trace.thought = reasoning_result.thought
        trace.resolved_query = search_query

        # F03: Query Expansion with IT/Ops thesaurus (v2: graph-based if ontology enabled)
        if self.config.enable_query_expansion:
            expanded = await self._expand_query(
                search_query, workspace, reasoning_result.domain_hint
            )
            trace.expansion_terms = expanded.added_terms
            search_query = expanded.expanded_query
            logger.info("Query expansion: +%d terms -> %s", len(expanded.added_terms), search_query)

        # ---- STEP 2: ACT (Hybrid Search multi-workspace + Reranking F04) ----
        all_workspaces = [workspace] + workspaces_publics
        search_tasks = []
        for ws in all_workspaces:
            rag = self._get_rag(ws)
            search_tasks.append(self.search.hybrid_search(rag, search_query, self.config))
        workspace_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Fusion and deduplication
        all_chunks = self.search.fuse_and_dedup(workspace_results, all_workspaces)
        trace.raw_chunks_count = len(all_chunks)

        # F04: Cognitive Reranking v2
        if self.config.enable_cognitive_reranking and all_chunks:
            all_chunks = await self.reranker.rerank(question, all_chunks)
            trace.reranked_chunks_count = len(all_chunks)
        else:
            all_chunks = all_chunks[: self.config.final_limit]

        # ---- STEP 3: OBSERVE (Synthesis + Citations) ----
        synthesis_result = await self.synthesis.synthesize(
            question=question,
            chunks=all_chunks,
            conversation_history=history,
        )

        trace.stop()
        trace.tokens_used = synthesis_result.tokens_used

        return QueryResult(
            answer=synthesis_result.answer,
            citations=synthesis_result.citations,
            trace=trace,
            intent=trace.intent,
        )

    def _scripted_response(self, intent: IntentType) -> str:
        """Scripted responses for non-VALID intents."""
        responses = {
            IntentType.GREETING: (
                "Bonjour ! Je suis l'assistant Twin, specialise dans les operations IT "
                "du Groupe BNP Paribas. Comment puis-je vous aider ?"
            ),
            IntentType.OUT_OF_SCOPE: (
                "Cette question semble en dehors de mon perimetre d'expertise "
                "(operations IT, infrastructure, incidents). "
                "Puis-je vous aider avec une question technique ?"
            ),
            IntentType.MALICIOUS: (
                "Je ne peux pas repondre a cette demande. "
                "Mon role est d'assister avec des questions techniques legitimes."
            ),
            IntentType.ESCALATION: (
                "Je comprends l'urgence. Pour un incident critique, contactez "
                "directement l'equipe de garde via le bridge de crise."
            ),
        }
        return responses.get(intent, responses[IntentType.OUT_OF_SCOPE])

    async def _expand_query(
        self,
        query: str,
        workspace: str,
        domain_hint: Optional[str],
    ):
        """Run query expansion (v2 graph-based or v1 thesaurus)."""
        if self.ontology_config and self.ontology_config.enabled:
            return await self.expander.expand_v2(
                query, workspace=workspace, domain_hint=domain_hint
            )
        return self.expander.expand(query, domain_hint=domain_hint)
