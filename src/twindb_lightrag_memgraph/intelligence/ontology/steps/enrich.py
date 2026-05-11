"""
intelligence/ontology/steps/enrich.py
======================================
ENRICH step -- LLM adds missing relationships and validates consistency.

Cross-references existing ontology in Memgraph (if any) and adds
SYNONYM, CO_OCCURS, DEPENDS_ON relations with confidence scores.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from ...config import TwinRAGConfig
from ..steps.cluster import ClusterResult
from ..steps.extract import ExtractedRelation

logger = logging.getLogger("twin_rag_intelligence.ontology.enrich")

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "ontology_enrich.txt"
)


@dataclass
class EnrichmentResult:
    clusters: ClusterResult = field(default_factory=ClusterResult)
    new_relations: list[ExtractedRelation] = field(default_factory=list)


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are an ontology enrichment agent for IT operations.

Given entities and their current relationships, identify MISSING relationships.
Focus on: SYNONYM, CO_OCCURS, DEPENDS_ON, CAUSED_BY, MITIGATED_BY.

ENTITIES:
{entities_json}

EXISTING RELATIONS:
{relations_json}

DOMAINS:
{domains_json}

TASK: Identify missing relationships between these entities.
Assign a confidence score (0.0-1.0) to each new relationship.

RESPONSE (JSON):
{{
  "new_relations": [
    {{"source": "...", "source_type": "...", "target": "...", "target_type": "...", \
"relation_type": "...", "confidence": 0.8}}
  ]
}}
"""


async def enrich(
    cluster_result: ClusterResult,
    config: TwinRAGConfig,
) -> EnrichmentResult:
    """Enrich ontology with missing relationships.

    Args:
        cluster_result: Result from the CLUSTER step.
        config: TwinRAG configuration.

    Returns:
        EnrichmentResult with new relations.
    """
    extraction = cluster_result.extraction
    if not extraction.entities:
        return EnrichmentResult(clusters=cluster_result)

    entities_json = json.dumps(
        [
            {"name": e.name, "type": e.entity_type, "definition": e.definition}
            for e in extraction.entities
        ],
        indent=2,
    )

    relations_json = json.dumps(
        [
            {
                "source": r.source,
                "target": r.target,
                "type": r.relation_type,
            }
            for r in extraction.relations
        ],
        indent=2,
    )

    domains_json = json.dumps(
        [
            {
                "domain": d.domain_name,
                "description": d.description,
                "members": d.member_terms,
            }
            for d in cluster_result.domains
        ],
        indent=2,
    )

    prompt_template = _load_prompt()
    prompt = prompt_template.format(
        entities_json=entities_json,
        relations_json=relations_json,
        domains_json=domains_json,
    )

    try:
        client = AsyncOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_api_base,
        )

        response = await client.chat.completions.create(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1500,
        )

        content = response.choices[0].message.content
        data = json.loads(content) if content else {}

        new_relations = [
            ExtractedRelation(
                source=r["source"],
                source_type=r.get("source_type", "Term"),
                target=r["target"],
                target_type=r.get("target_type", "Term"),
                relation_type=r.get("relation_type", "RELATED_TO"),
                confidence=r.get("confidence", 0.8),
            )
            for r in data.get("new_relations", [])
        ]

        return EnrichmentResult(
            clusters=cluster_result,
            new_relations=new_relations,
        )

    except Exception as e:
        logger.error("Enrichment error: %s", e)
        return EnrichmentResult(clusters=cluster_result)
