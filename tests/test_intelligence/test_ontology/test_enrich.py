"""Tests for the ontology ENRICH step.

The enrich LLM call ingests entities/relations derived from untrusted
documents, so its JSON output is untrusted too. ``relation_type`` is
later interpolated into a Cypher MERGE that has no ``$param``
parameterization for relationship types — so an un-validated value
here is a direct Cypher-injection vector. These tests pin the
allow-list defense.
"""

import json
from unittest.mock import patch

import pytest

from twindb_lightrag_memgraph.intelligence.ontology.steps.cluster import (
    ClusterResult,
)
from twindb_lightrag_memgraph.intelligence.ontology.steps.enrich import (
    enrich,
)
from twindb_lightrag_memgraph.intelligence.ontology.steps.extract import (
    ExtractedEntity,
    ExtractionResult,
)


def _cluster_result_with_entities() -> ClusterResult:
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(name="alpha", entity_type="Term"),
            ExtractedEntity(name="beta", entity_type="Term"),
        ],
        relations=[],
        source_doc="doc_0",
    )
    return ClusterResult(extraction=extraction, domains=[])


def _enrich_payload(relations: list[dict]) -> str:
    return json.dumps({"new_relations": relations})


@pytest.fixture
def cluster_result():
    return _cluster_result_with_entities()


async def test_malicious_relation_type_is_downgraded(
    config, cluster_result, mock_openai_client
):
    payload = _enrich_payload(
        [
            {
                "source": "alpha",
                "source_type": "Term",
                "target": "beta",
                "target_type": "Term",
                "relation_type": "REL`]->(x) WITH x MATCH (n) DETACH DELETE n //",
                "confidence": 0.9,
            }
        ]
    )
    client = mock_openai_client(payload)

    with patch(
        "twindb_lightrag_memgraph.intelligence.ontology.steps.enrich."
        "AsyncOpenAI",
        return_value=client,
    ):
        result = await enrich(cluster_result, config)

    assert len(result.new_relations) == 1
    rel = result.new_relations[0]
    assert rel.relation_type == "RELATED_TO"
    assert "`" not in rel.relation_type


async def test_unknown_relation_type_is_downgraded(
    config, cluster_result, mock_openai_client
):
    payload = _enrich_payload(
        [
            {
                "source": "alpha",
                "source_type": "Term",
                "target": "beta",
                "target_type": "Term",
                "relation_type": "TOTALLY_MADE_UP",
                "confidence": 0.7,
            }
        ]
    )
    client = mock_openai_client(payload)

    with patch(
        "twindb_lightrag_memgraph.intelligence.ontology.steps.enrich."
        "AsyncOpenAI",
        return_value=client,
    ):
        result = await enrich(cluster_result, config)

    assert len(result.new_relations) == 1
    assert result.new_relations[0].relation_type == "RELATED_TO"


async def test_valid_relation_type_preserved(
    config, cluster_result, mock_openai_client
):
    payload = _enrich_payload(
        [
            {
                "source": "alpha",
                "source_type": "Term",
                "target": "beta",
                "target_type": "Term",
                "relation_type": "CO_OCCURS",
                "confidence": 0.85,
            }
        ]
    )
    client = mock_openai_client(payload)

    with patch(
        "twindb_lightrag_memgraph.intelligence.ontology.steps.enrich."
        "AsyncOpenAI",
        return_value=client,
    ):
        result = await enrich(cluster_result, config)

    assert len(result.new_relations) == 1
    assert result.new_relations[0].relation_type == "CO_OCCURS"


async def test_unknown_node_type_normalised(
    config, cluster_result, mock_openai_client
):
    payload = _enrich_payload(
        [
            {
                "source": "alpha",
                "source_type": "BogusType",
                "target": "beta",
                "target_type": "Term",
                "relation_type": "RELATED_TO",
                "confidence": 0.8,
            }
        ]
    )
    client = mock_openai_client(payload)

    with patch(
        "twindb_lightrag_memgraph.intelligence.ontology.steps.enrich."
        "AsyncOpenAI",
        return_value=client,
    ):
        result = await enrich(cluster_result, config)

    assert len(result.new_relations) == 1
    assert result.new_relations[0].source_type == "Term"


async def test_relation_missing_source_is_dropped(
    config, cluster_result, mock_openai_client
):
    payload = _enrich_payload(
        [
            {
                "target": "beta",
                "target_type": "Term",
                "relation_type": "RELATED_TO",
                "confidence": 0.8,
            }
        ]
    )
    client = mock_openai_client(payload)

    with patch(
        "twindb_lightrag_memgraph.intelligence.ontology.steps.enrich."
        "AsyncOpenAI",
        return_value=client,
    ):
        result = await enrich(cluster_result, config)

    assert result.new_relations == []
