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
import hashlib
import hmac
import logging
import secrets
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Optional

from lightrag import LightRAG

from .._constants import validate_identifier
from .config import TwinRAGConfig
from .fallbacks import query_fallback_scope
from .features.cognitive_reranker import CognitiveReranker
from .features.feedback import FeedbackStore
from .features.workspace_router import FolderRouter
from .features.intent_classifier import IntentClassifier
from .features.query_expander import QueryExpander
from .models.schemas import AnswerStatus, IntentType, QueryResult, QueryTrace
from .ontology.config import OntologyConfig, load_ontology_config
from .react.act import SearchEngine
from .react.observe import SynthesisEngine
from .react.reason import ReasoningEngine

logger = logging.getLogger("twin_rag_intelligence")
_LOG_FINGERPRINT_KEY = secrets.token_bytes(32)


def _safe_text_metadata(value: str | None, prefix: str) -> dict[str, int | str]:
    """Return process-local, non-reversible metadata for sensitive text."""
    text = value or ""
    digest = hmac.new(
        _LOG_FINGERPRINT_KEY,
        text.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return {
        f"{prefix}_fingerprint": digest,
        f"{prefix}_length": len(text),
    }


class TwinRAGEngine:
    """
    RAG intelligence engine for Twin KMS.

    Encapsulates the full pipeline:
    F05 (Intent) -> REASON (Coref + F03 Expansion) -> ACT (Search + F04 Rerank) -> OBSERVE (Synthesis)

    Usage:
        engine = TwinRAGEngine(config)
        result = await engine.aquery(
            "Pourquoi ORA-04030 ?",
            conversation_history=[...],
            workspace="demo",
            authorized_folders={"demo", "commons"},
        )
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
        self.feedback = (
            FeedbackStore(self.config) if self.config.enable_feedback else None
        )
        self.folder_router = self._build_folder_router()

        # Ontology config (opt-in via ontology.json)
        self.ontology_config: Optional[OntologyConfig] = load_ontology_config()

        # LightRAG instances (lazy, per workspace)
        self._rag_instances: dict[str, LightRAG] = {}

    def _get_rag(self, workspace: str) -> LightRAG:
        """Return (or create) a LightRAG instance for a given workspace."""
        workspace = validate_identifier(str(workspace), "workspace")
        if workspace not in self._rag_instances:
            from .. import register

            register()

            from lightrag.kg.memgraph_impl import MemgraphStorage

            if not getattr(
                MemgraphStorage.__init__,
                "_twindb_explicit_workspace_patch",
                False,
            ):
                raise RuntimeError(
                    "The installed LightRAG Memgraph constructor is not "
                    "compatible with explicit workspace isolation"
                )

            self._rag_instances[workspace] = LightRAG(
                working_dir=f"/tmp/lightrag_{workspace}",
                workspace=workspace,
                kv_storage="MemgraphKVStorage",
                vector_storage="MemgraphVectorDBStorage",
                doc_status_storage="MemgraphDocStatusStorage",
                graph_storage="MemgraphStorage",
            )
        return self._rag_instances[workspace]

    def _build_folder_router(self) -> FolderRouter | None:
        """Build the folder router when the feature flag is enabled."""
        if not self.config.effective_enable_folder_routing:
            return None

        rules_path = self.config.effective_folder_routing_rules_path
        if rules_path:
            path = Path(rules_path)
        else:
            path = Path(__file__).parent / "routing" / "routing_rules.json"

        return FolderRouter.from_json(
            path,
            default_folder=self.config.effective_default_folder,
        )

    def _resolve_folders(self, folder, workspace, folders_publics, workspaces_publics):
        """Resolve (active_folder, public_folders, explicit_override) from the
        folder/workspace args, honouring the deprecated ``workspace*`` aliases."""
        active_folder = folder or workspace or self.config.effective_default_folder
        public_folders = (
            folders_publics if folders_publics is not None else workspaces_publics
        )
        if public_folders is None:
            public_folders = [self.config.effective_default_folder]
        explicit_folder_override = (
            folder is not None
            or workspace is not None
            or folders_publics is not None
            or workspaces_publics is not None
        )
        return active_folder, public_folders, explicit_folder_override

    async def _classify_and_short_circuit(self, question, trace):
        """STEP 0 (F05): classify and stop non-retrieval intents early."""
        if not self.config.enable_oos_detection:
            return None
        intent_result = await self.intent_classifier.classify(question)
        trace.intent = intent_result
        reason_meta = _safe_text_metadata(intent_result.reason, "reason")
        logger.info(
            "Intent classified: intent=%s confidence=%.2f reason_fingerprint=%s "
            "reason_length=%d",
            intent_result.intent.value,
            intent_result.confidence,
            reason_meta["reason_fingerprint"],
            reason_meta["reason_length"],
            extra={
                "intent": intent_result.intent.value,
                "confidence": intent_result.confidence,
                **reason_meta,
            },
        )

        if (
            intent_result.intent == IntentType.OUT_OF_SCOPE
            and intent_result.confidence >= self.config.oos_confidence_threshold
        ):
            trace.stop(early_exit="OOS")
            return QueryResult(
                answer=self._scripted_response(intent_result.intent),
                citations=[],
                answer_status=AnswerStatus.NO_RETRIEVAL,
                trace=trace,
                intent=intent_result,
            )

        if intent_result.intent == IntentType.GREETING:
            trace.stop(early_exit="GREETING")
            return QueryResult(
                answer=self._scripted_response(IntentType.GREETING),
                citations=[],
                answer_status=AnswerStatus.NO_RETRIEVAL,
                trace=trace,
                intent=intent_result,
            )

        if intent_result.intent == IntentType.MALICIOUS:
            question_meta = _safe_text_metadata(question, "question")
            logger.warning(
                "Malicious attempt detected: question_fingerprint=%s question_length=%d",
                question_meta["question_fingerprint"],
                question_meta["question_length"],
                extra=question_meta,
            )
            trace.stop(early_exit="MALICIOUS")
            return QueryResult(
                answer=self._scripted_response(IntentType.MALICIOUS),
                citations=[],
                answer_status=AnswerStatus.NO_RETRIEVAL,
                trace=trace,
                intent=intent_result,
            )

        if (
            intent_result.intent == IntentType.ESCALATION
            and intent_result.confidence >= self.config.escalation_confidence_threshold
        ):
            trace.stop(early_exit="ESCALATION")
            return QueryResult(
                answer=self._scripted_response(IntentType.ESCALATION),
                citations=[],
                answer_status=AnswerStatus.NO_RETRIEVAL,
                trace=trace,
                intent=intent_result,
            )
        return None

    async def aquery(
        self,
        question: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        workspace: Optional[str] = None,
        workspaces_publics: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        folder: Optional[str] = None,
        folders_publics: Optional[list[str]] = None,
        authorized_folders: Optional[set[str]] = None,
        retrieval_provider: Callable[[str, str], Awaitable[list]] | None = None,
        enable_rerank: bool | None = None,
        on_synthesis_token: Callable[[str], Awaitable[None]] | None = None,
        on_stage: Callable[[str], Awaitable[None]] | None = None,
        response_type: str | None = None,
    ) -> QueryResult:
        """Run one request under an isolated, bounded degradation trace."""
        with query_fallback_scope() as fallbacks:
            result = await self._aquery_impl(
                question=question,
                conversation_history=conversation_history,
                workspace=workspace,
                workspaces_publics=workspaces_publics,
                user_id=user_id,
                folder=folder,
                folders_publics=folders_publics,
                authorized_folders=authorized_folders,
                retrieval_provider=retrieval_provider,
                enable_rerank=enable_rerank,
                on_synthesis_token=on_synthesis_token,
                on_stage=on_stage,
                response_type=response_type,
            )
            if result.trace is not None:
                result.trace.fallbacks = list(fallbacks)
            return result

    async def _aquery_impl(
        self,
        question: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        workspace: Optional[str] = None,
        workspaces_publics: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        folder: Optional[str] = None,
        folders_publics: Optional[list[str]] = None,
        authorized_folders: Optional[set[str]] = None,
        retrieval_provider: Callable[[str, str], Awaitable[list]] | None = None,
        enable_rerank: bool | None = None,
        on_synthesis_token: Callable[[str], Awaitable[None]] | None = None,
        on_stage: Callable[[str], Awaitable[None]] | None = None,
        response_type: str | None = None,
    ) -> QueryResult:
        """
        Main entry point -- Full ReAct + AFFINE pipeline.

        Args:
            question: User question in natural language.
            conversation_history: List of messages [{role, content}], max N recent.
            folder: Agent's private folder (e.g., "demo", "demo_secondary").
            folders_publics: List of public folders to query (e.g., ["commons"]).
            workspace: Deprecated alias for folder.
            workspaces_publics: Deprecated alias for folders_publics.
            user_id: User identifier for feedback.
            authorized_folders: Authoritative folder ids the caller may query.

        Returns:
            QueryResult containing answer, citations, trace, intent.
        """
        active_folder, public_folders, explicit_folder_override = self._resolve_folders(
            folder, workspace, folders_publics, workspaces_publics
        )
        if authorized_folders is None:
            raise PermissionError("An authoritative folder scope is required")

        authorized = {
            validate_identifier(str(item), "folder") for item in authorized_folders
        }
        active_folder = validate_identifier(str(active_folder), "folder")
        public_folders = [
            validate_identifier(str(item), "folder") for item in public_folders
        ]
        denied = {active_folder, *public_folders} - authorized
        if denied:
            raise PermissionError(f"Unauthorized folders: {sorted(denied)}")

        trace = QueryTrace(question=question, workspace=active_folder, user_id=user_id)
        trace.start()
        history = (conversation_history or [])[-self.config.conversation_memory_depth :]

        # ---- STEP 0: INTENT CLASSIFICATION (F05) ----
        early_exit = await self._classify_and_short_circuit(question, trace)
        if early_exit is not None:
            if on_stage is not None:
                await on_stage("generation")
            return early_exit

        # ---- STEP 1: REASON (Coreference + Query Expansion F03) ----
        reasoning_result = await self.reasoning.analyze(question, history)
        search_query = reasoning_result.search_query
        trace.thought = reasoning_result.thought
        trace.resolved_query = search_query

        # F03: Query Expansion with IT/Ops thesaurus (v2: graph-based if ontology enabled)
        if self.config.enable_query_expansion:
            expanded = await self._expand_query(
                search_query, active_folder, reasoning_result.domain_hint
            )
            trace.expansion_terms = expanded.added_terms
            search_query = expanded.expanded_query
            expanded_meta = _safe_text_metadata(search_query, "expanded_query")
            logger.info(
                "Query expansion applied: added_terms_count=%d "
                "expanded_query_fingerprint=%s expanded_query_length=%d",
                len(expanded.added_terms),
                expanded_meta["expanded_query_fingerprint"],
                expanded_meta["expanded_query_length"],
                extra={
                    "added_terms_count": len(expanded.added_terms),
                    **expanded_meta,
                },
            )

        # ---- STEP 2: ACT (Hybrid Search multi-workspace + Reranking F04) ----
        all_workspaces = await self._resolve_search_folders(
            query=search_query,
            active_folder=active_folder,
            public_folders=public_folders,
            explicit_folder_override=explicit_folder_override,
        )
        all_workspaces = [
            validate_identifier(str(item), "folder") for item in all_workspaces
        ]
        denied = set(all_workspaces) - authorized
        if denied:
            raise PermissionError(f"Unauthorized folders: {sorted(denied)}")
        search_tasks = []
        for ws in all_workspaces:
            if retrieval_provider is not None:
                search_tasks.append(retrieval_provider(ws, search_query))
            else:
                rag = self._get_rag(ws)
                search_tasks.append(
                    self.search.hybrid_search(rag, search_query, self.config)
                )
        workspace_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Fusion and deduplication
        all_chunks = self.search.fuse_and_dedup(workspace_results, all_workspaces)
        trace.raw_chunks_count = len(all_chunks)

        # F04: Cognitive Reranking v2
        rerank_enabled = (
            self.config.enable_cognitive_reranking
            if enable_rerank is None
            else enable_rerank
        )
        if rerank_enabled and all_chunks:
            all_chunks = await self.reranker.rerank(question, all_chunks)
            trace.reranked_chunks_count = len(all_chunks)
        else:
            all_chunks = all_chunks[: self.config.final_limit]

        # ---- STEP 3: OBSERVE (Synthesis + Citations) ----
        if on_stage is not None:
            await on_stage("generation")
        synthesis_result = await self.synthesis.synthesize(
            question=question,
            chunks=all_chunks,
            conversation_history=history,
            on_token=on_synthesis_token,
            response_type=response_type,
        )

        trace.stop()
        trace.tokens_used = synthesis_result.tokens_used

        return QueryResult(
            answer=synthesis_result.answer,
            citations=synthesis_result.citations,
            answer_status=synthesis_result.answer_status,
            trace=trace,
            intent=trace.intent,
        )

    async def _resolve_search_folders(
        self,
        *,
        query: str,
        active_folder: str,
        public_folders: list[str],
        explicit_folder_override: bool,
    ) -> list[str]:
        """Resolve the folder list, then return LightRAG workspace names."""
        if not self.folder_router:
            return list(dict.fromkeys([active_folder] + public_folders))

        if explicit_folder_override:
            routing = await self.folder_router.route(
                query,
                provided_folders=[active_folder],
                provided_public_folders=public_folders,
            )
        else:
            routing = await self.folder_router.route(query)

        folders = routing.folders + routing.public_folders
        if not folders:
            folders = [active_folder] + public_folders
        return list(dict.fromkeys(folders))

    def _scripted_response(self, intent: IntentType) -> str:
        """Scripted responses for non-VALID intents."""
        responses = {
            IntentType.GREETING: (
                "Bonjour ! Je suis l'assistant Twin, specialise dans les operations IT "
                "de votre organisation. Comment puis-je vous aider ?"
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
        if (
            self.config.enable_ontology
            and self.ontology_config
            and self.ontology_config.enabled
        ):
            return await self.expander.expand_v2(
                query, workspace=workspace, domain_hint=domain_hint
            )
        return self.expander.expand(query, domain_hint=domain_hint)
