"""
intelligence/ontology/steps/cluster.py
=======================================
CLUSTER step -- Groups extracted entities into emergent domains.

Domains emerge from the data, not from a static list.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from ...config import TwinRAGConfig
from ..steps.extract import ExtractionResult

logger = logging.getLogger("twin_rag_intelligence.ontology.cluster")

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "ontology_cluster.txt"
)


@dataclass
class DomainCluster:
    domain_name: str
    description: str = ""
    member_terms: list[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    domains: list[DomainCluster] = field(default_factory=list)
    extraction: ExtractionResult = field(default_factory=ExtractionResult)


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are a domain clustering agent for an IT operations ontology.

Given a list of extracted entities, identify coherent domains.
Do NOT use predefined categories. Domains must emerge from the data.

ENTITIES:
{entities_json}

TASK: Group these entities into coherent domains.

RESPONSE (JSON):
{{
  "domains": [
    {{
      "domain_name": "...",
      "description": "Short description of this domain",
      "member_terms": ["entity1", "entity2", ...]
    }}
  ]
}}
"""


async def cluster(
    extraction: ExtractionResult,
    config: TwinRAGConfig,
) -> ClusterResult:
    """Group extracted entities into emergent domains.

    Args:
        extraction: Result from the EXTRACT step.
        config: TwinRAG configuration.

    Returns:
        ClusterResult with domain clusters.
    """
    if not extraction.entities:
        return ClusterResult(extraction=extraction)

    entities_json = json.dumps(
        [
            {"name": e.name, "type": e.entity_type, "definition": e.definition}
            for e in extraction.entities
        ],
        indent=2,
    )

    prompt_template = _load_prompt()
    prompt = prompt_template.format(entities_json=entities_json)

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

        domains = [
            DomainCluster(
                domain_name=d["domain_name"],
                description=d.get("description", ""),
                member_terms=d.get("member_terms", []),
            )
            for d in data.get("domains", [])
        ]

        return ClusterResult(domains=domains, extraction=extraction)

    except Exception as e:
        logger.error("Clustering error: %s", e)
        return ClusterResult(extraction=extraction)
