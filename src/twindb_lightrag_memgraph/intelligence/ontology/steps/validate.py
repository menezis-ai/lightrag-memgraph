"""
intelligence/ontology/steps/validate.py
========================================
VALIDATE step -- Quality gate for ontology data.

Filters entities/relations below confidence_threshold,
detects duplicates, and returns a dry-run preview when
require_review is True.
"""

import logging
from dataclasses import dataclass, field

from ..steps.enrich import EnrichmentResult
from ..storage import OntologyEdge, OntologyNode

logger = logging.getLogger("twin_rag_intelligence.ontology.validate")


@dataclass
class ValidationResult:
    nodes: list[OntologyNode] = field(default_factory=list)
    edges: list[OntologyEdge] = field(default_factory=list)
    rejected_nodes: list[dict] = field(default_factory=list)
    rejected_edges: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_dry_run: bool = True


def validate(
    enrichment: EnrichmentResult,
    confidence_threshold: float = 0.7,
    require_review: bool = True,
) -> ValidationResult:
    """Validate and filter ontology data.

    Args:
        enrichment: Result from the ENRICH step.
        confidence_threshold: Minimum confidence to accept.
        require_review: If True, mark as dry-run (no MERGE).

    Returns:
        ValidationResult with accepted/rejected nodes and edges.
    """
    result = ValidationResult(is_dry_run=require_review)
    extraction = enrichment.clusters.extraction

    # --- Validate and deduplicate nodes ---
    seen_names: dict[str, str] = {}  # lowercase name -> original name

    for entity in extraction.entities:
        name_lower = entity.name.lower().strip()

        if entity.confidence < confidence_threshold:
            result.rejected_nodes.append(
                {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "confidence": entity.confidence,
                    "reason": "below_threshold",
                }
            )
            continue

        if name_lower in seen_names:
            result.warnings.append(
                f"Duplicate entity: '{entity.name}' "
                f"(already seen as '{seen_names[name_lower]}')"
            )
            continue

        seen_names[name_lower] = entity.name
        result.nodes.append(
            OntologyNode(
                name=entity.name,
                node_type=entity.entity_type,
                properties=entity.properties,
                confidence=entity.confidence,
                source_doc=extraction.source_doc,
            )
        )

    # --- Add Domain nodes from clusters ---
    for domain in enrichment.clusters.domains:
        domain_lower = domain.domain_name.lower().strip()
        if domain_lower not in seen_names:
            seen_names[domain_lower] = domain.domain_name
            result.nodes.append(
                OntologyNode(
                    name=domain.domain_name,
                    node_type="Domain",
                    properties={"description": domain.description},
                    confidence=1.0,
                    source_doc=extraction.source_doc,
                )
            )

        # Add PART_OF edges for domain members
        for member in domain.member_terms:
            result.edges.append(
                OntologyEdge(
                    source_name=member,
                    source_type="Term",
                    target_name=domain.domain_name,
                    target_type="Domain",
                    relation_type="PART_OF",
                    confidence=0.9,
                    source_doc=extraction.source_doc,
                )
            )

    # --- Validate edges from extraction ---
    all_edges = list(extraction.relations) + enrichment.new_relations

    for rel in all_edges:
        if rel.confidence < confidence_threshold:
            result.rejected_edges.append(
                {
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.relation_type,
                    "confidence": rel.confidence,
                    "reason": "below_threshold",
                }
            )
            continue

        result.edges.append(
            OntologyEdge(
                source_name=rel.source,
                source_type=rel.source_type,
                target_name=rel.target,
                target_type=rel.target_type,
                relation_type=rel.relation_type,
                confidence=rel.confidence,
                source_doc=extraction.source_doc,
            )
        )

    logger.info(
        "[Validate] Accepted: %d nodes, %d edges. Rejected: %d nodes, %d edges. "
        "Warnings: %d. Dry-run: %s",
        len(result.nodes),
        len(result.edges),
        len(result.rejected_nodes),
        len(result.rejected_edges),
        len(result.warnings),
        result.is_dry_run,
    )

    return result
