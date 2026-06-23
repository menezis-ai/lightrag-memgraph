"""
intelligence/ontology/pipeline.py
==================================
Orchestrates the 4-step ontology pipeline:
EXTRACT -> CLUSTER -> ENRICH -> VALIDATE

Each step is a separate module. This class wires them together.

When require_review is True (default), results are returned as a dry-run.
Call approve(result) to persist after human validation.
"""

import logging

from ..config import TwinRAGConfig
from .config import OntologyConfig
from .steps.cluster import ClusterResult, cluster
from .steps.enrich import EnrichmentResult, enrich
from .steps.extract import ExtractedEntity, ExtractionResult, extract
from .steps.validate import ValidationResult, validate
from .storage import OntologyStorage

logger = logging.getLogger("twin_rag_intelligence.ontology.pipeline")


class OntologyPipeline:
    """4-step ontology pipeline: Extract -> Cluster -> Enrich -> Validate."""

    def __init__(
        self, config: TwinRAGConfig, onto_config: OntologyConfig
    ) -> None:
        self.config = config
        self.onto_config = onto_config

    async def run(
        self,
        documents: list[str],
        workspace: str,
        chunks: list[str] | None = None,
    ) -> ValidationResult:
        """Run the full pipeline on a list of documents.

        Args:
            documents: List of document text contents.
            workspace: Target workspace.
            chunks: Optional list of text chunks for local-pass extraction.
                Only used when dual_pass is enabled.

        Returns:
            ValidationResult with accepted/rejected nodes and edges.
            If require_review is True, this is a dry-run preview.
            Call approve(result, workspace) to persist.
        """
        ws_config = self.onto_config.workspaces.get(workspace)
        if ws_config is None:
            logger.warning(
                "No ontology config for workspace '%s', skipping", workspace
            )
            return ValidationResult()

        logger.info(
            "[Pipeline:%s] Starting ontology pipeline (mode=%s, %d docs)",
            workspace,
            ws_config.mode,
            len(documents),
        )

        # Step 1: EXTRACT
        if self.onto_config.dual_pass:
            global_result = await self._extract_global(documents, workspace)
            local_items = chunks if chunks else documents
            local_result = await self._extract_local(local_items, workspace)
            extracted = self._merge_extractions(global_result, local_result)
        else:
            extracted = await self._extract(documents, workspace)

        # Step 2: CLUSTER
        clustered = await self._cluster(extracted)

        # Step 3: ENRICH
        enriched = await self._enrich(clustered)

        # Step 4: VALIDATE
        validated = self._validate(enriched)

        # Auto-persist only if review is disabled
        if not self.onto_config.require_review:
            await self.approve(validated, workspace)

        return validated

    async def approve(
        self, result: ValidationResult, workspace: str
    ) -> None:
        """Persist a validated result to Memgraph after human review.

        Args:
            result: ValidationResult from run() or from manual review.
            workspace: Target workspace.
        """
        storage = OntologyStorage(workspace)
        await storage.initialize()
        await storage.upsert_nodes(result.nodes)
        await storage.upsert_edges(result.edges)
        result.is_dry_run = False
        logger.info(
            "[Pipeline:%s] Approved and persisted %d nodes, %d edges",
            workspace,
            len(result.nodes),
            len(result.edges),
        )

    async def _extract(
        self, documents: list[str], workspace: str
    ) -> ExtractionResult:
        """Run EXTRACT step across all documents and merge results."""
        ws_config = self.onto_config.workspaces[workspace]
        merged = ExtractionResult()

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            result = await extract(
                document=doc,
                doc_id=doc_id,
                config=self.config,
                ws_config=ws_config,
                dsep_enabled=self.onto_config.dsep_enabled,
            )
            merged.entities.extend(result.entities)
            merged.relations.extend(result.relations)
            if not merged.source_doc:
                merged.source_doc = result.source_doc

        logger.info(
            "[Pipeline] Extracted %d entities, %d relations from %d docs",
            len(merged.entities),
            len(merged.relations),
            len(documents),
        )
        return merged

    async def _extract_global(
        self, documents: list[str], workspace: str
    ) -> ExtractionResult:
        """Run global EXTRACT pass -- high-level structure from full documents."""
        ws_config = self.onto_config.workspaces[workspace]
        merged = ExtractionResult()

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}_global"
            result = await extract(
                document=doc,
                doc_id=doc_id,
                config=self.config,
                ws_config=ws_config,
                dsep_enabled=self.onto_config.dsep_enabled,
                pass_type="global",
                global_max_tokens=self.onto_config.global_max_tokens,
            )
            merged.entities.extend(result.entities)
            merged.relations.extend(result.relations)
            if not merged.source_doc:
                merged.source_doc = result.source_doc

        logger.info(
            "[Pipeline] Global pass: %d entities, %d relations from %d docs",
            len(merged.entities),
            len(merged.relations),
            len(documents),
        )
        return merged

    async def _extract_local(
        self, items: list[str], workspace: str
    ) -> ExtractionResult:
        """Run local EXTRACT pass -- precise entities from chunks or documents."""
        ws_config = self.onto_config.workspaces[workspace]
        merged = ExtractionResult()

        for i, item in enumerate(items):
            doc_id = f"chunk_{i}_local"
            result = await extract(
                document=item,
                doc_id=doc_id,
                config=self.config,
                ws_config=ws_config,
                dsep_enabled=self.onto_config.dsep_enabled,
                pass_type="local",
            )
            merged.entities.extend(result.entities)
            merged.relations.extend(result.relations)
            if not merged.source_doc:
                merged.source_doc = result.source_doc

        logger.info(
            "[Pipeline] Local pass: %d entities, %d relations from %d items",
            len(merged.entities),
            len(merged.relations),
            len(items),
        )
        return merged

    @staticmethod
    def _merge_extractions(
        global_result: ExtractionResult, local_result: ExtractionResult
    ) -> ExtractionResult:
        """Merge global and local extraction results with entity deduplication.

        Entities are deduplicated by name (case-insensitive), keeping the one
        with highest confidence. Relations are kept from both passes without
        deduplication (different granularity may yield different relations).
        """
        # Deduplicate entities by exact name match (case-insensitive),
        # keep highest confidence.
        # Phase 2 follow-up: add fuzzy match (Levenshtein ratio > 0.85) to detect
        # near-matches like PGA_AGGREGATE_LIMIT vs PGA_AGGREGATE_TARGET and
        # emit RELATED_TO relations instead of merging them.
        entity_map: dict[str, ExtractedEntity] = {}
        for entity in global_result.entities + local_result.entities:
            key = entity.name.lower()
            if key not in entity_map or entity.confidence > entity_map[key].confidence:
                entity_map[key] = entity

        # Relations: merge without dedup
        all_relations = global_result.relations + local_result.relations

        return ExtractionResult(
            entities=list(entity_map.values()),
            relations=all_relations,
            source_doc=global_result.source_doc or local_result.source_doc,
        )

    async def _cluster(self, extraction: ExtractionResult) -> ClusterResult:
        """Run CLUSTER step."""
        result = await cluster(extraction, self.config)
        logger.info(
            "[Pipeline] Clustered into %d domains", len(result.domains)
        )
        return result

    async def _enrich(self, cluster_result: ClusterResult) -> EnrichmentResult:
        """Run ENRICH step."""
        result = await enrich(cluster_result, self.config)
        logger.info(
            "[Pipeline] Enriched with %d new relations",
            len(result.new_relations),
        )
        return result

    def _validate(self, enrichment: EnrichmentResult) -> ValidationResult:
        """Run VALIDATE step."""
        return validate(
            enrichment,
            confidence_threshold=self.onto_config.confidence_threshold,
            require_review=self.onto_config.require_review,
        )
