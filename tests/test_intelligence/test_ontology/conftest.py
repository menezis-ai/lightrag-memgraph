"""
Common test fixtures for ontology tests.
All tests run WITHOUT external infrastructure (no Memgraph, no LLM).
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twindb_lightrag_memgraph.intelligence.config import TwinRAGConfig
from twindb_lightrag_memgraph.intelligence.ontology.config import (
    OntologyConfig,
    WorkspaceOntologyConfig,
)


@pytest.fixture
def config():
    return TwinRAGConfig(
        llm_api_key="test-key",
        llm_api_base="http://mock:8080",
    )


@pytest.fixture
def onto_config_dedicated():
    return OntologyConfig(
        enabled=True,
        confidence_threshold=0.7,
        require_review=True,
        dsep_enabled=True,
        workspaces={
            "oracle_ws": WorkspaceOntologyConfig(
                mode="dedicated",
                subject="Oracle Database",
                context="Oracle DBA knowledge base",
            ),
        },
    )


@pytest.fixture
def onto_config_emergence():
    return OntologyConfig(
        enabled=True,
        confidence_threshold=0.7,
        require_review=True,
        dsep_enabled=True,
        workspaces={
            "commons": WorkspaceOntologyConfig(
                mode="emergence",
                subject="IT Operations",
                context="General IT operations documentation",
            ),
        },
    )


@pytest.fixture
def onto_config_deep_extraction():
    return OntologyConfig(
        enabled=True,
        confidence_threshold=0.7,
        require_review=True,
        dsep_enabled=True,
        workspaces={
            "deep_ws": WorkspaceOntologyConfig(
                mode="deep_extraction",
                subject="Symbolic analysis",
                context="Deep exploration",
            ),
        },
    )


@pytest.fixture
def mock_llm_response():
    """Factory to simulate LLM responses."""

    def _make(content: str, total_tokens: int = 100):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.usage = MagicMock(total_tokens=total_tokens)
        return mock_response

    return _make


@pytest.fixture
def mock_openai_client(mock_llm_response):
    """Factory for a mock AsyncOpenAI client with a preset response."""

    def _make(content: str, total_tokens: int = 100):
        client = AsyncMock()
        response = mock_llm_response(content, total_tokens)
        client.chat.completions.create = AsyncMock(return_value=response)
        return client

    return _make


@pytest.fixture
def mock_extract_response():
    return json.dumps(
        {
            "entities": [
                {
                    "name": "ORA-04030",
                    "type": "Term",
                    "definition": "Oracle PGA memory exhaustion error",
                    "confidence": 0.95,
                },
                {
                    "name": "PGA",
                    "type": "Term",
                    "definition": "Program Global Area",
                    "confidence": 0.90,
                },
                {
                    "name": "SGA",
                    "type": "Term",
                    "definition": "System Global Area",
                    "confidence": 0.92,
                },
                {
                    "name": "Oracle DBA",
                    "type": "Role",
                    "definition": "Database administrator",
                    "confidence": 0.85,
                },
                {
                    "name": "AWR Report",
                    "type": "Tool",
                    "definition": "Automatic Workload Repository",
                    "confidence": 0.88,
                },
            ],
            "relations": [
                {
                    "source": "ORA-04030",
                    "source_type": "Term",
                    "target": "PGA",
                    "target_type": "Term",
                    "relation_type": "RELATED_TO",
                    "confidence": 0.90,
                },
                {
                    "source": "Oracle DBA",
                    "source_type": "Role",
                    "target": "AWR Report",
                    "target_type": "Tool",
                    "relation_type": "USES",
                    "confidence": 0.85,
                },
            ],
        }
    )


@pytest.fixture
def mock_cluster_response():
    return json.dumps(
        {
            "domains": [
                {
                    "domain_name": "Oracle Memory Management",
                    "description": "Memory allocation and management in Oracle Database",
                    "member_terms": ["ORA-04030", "PGA", "SGA"],
                },
                {
                    "domain_name": "Database Administration",
                    "description": "DBA tools and processes",
                    "member_terms": ["Oracle DBA", "AWR Report"],
                },
            ]
        }
    )


@pytest.fixture
def mock_enrich_response():
    return json.dumps(
        {
            "new_relations": [
                {
                    "source": "PGA",
                    "source_type": "Term",
                    "target": "SGA",
                    "target_type": "Term",
                    "relation_type": "CO_OCCURS",
                    "confidence": 0.88,
                },
                {
                    "source": "ORA-04030",
                    "source_type": "Term",
                    "target": "AWR Report",
                    "target_type": "Tool",
                    "relation_type": "DIAGNOSED_WITH",
                    "confidence": 0.82,
                },
            ]
        }
    )


@pytest.fixture
def onto_config_dual_pass():
    return OntologyConfig(
        enabled=True,
        confidence_threshold=0.7,
        require_review=True,
        dsep_enabled=True,
        dual_pass=True,
        global_max_tokens=20000,
        workspaces={
            "ws": WorkspaceOntologyConfig(
                mode="emergence",
                subject="IT Operations",
                context="General IT operations documentation",
            ),
        },
    )


@pytest.fixture
def mock_extract_global_response():
    return json.dumps(
        {
            "entities": [
                {
                    "name": "Infrastructure Management",
                    "type": "Domain",
                    "definition": "IT infrastructure lifecycle management",
                    "confidence": 0.92,
                },
                {
                    "name": "Incident Response",
                    "type": "Domain",
                    "definition": "Process for handling IT incidents",
                    "confidence": 0.88,
                },
                {
                    "name": "Change Advisory Board",
                    "type": "Process",
                    "definition": "Review board for infrastructure changes",
                    "confidence": 0.85,
                },
            ],
            "relations": [
                {
                    "source": "Incident Response",
                    "source_type": "Domain",
                    "target": "Change Advisory Board",
                    "target_type": "Process",
                    "relation_type": "FEEDS_INTO",
                    "confidence": 0.82,
                },
            ],
        }
    )


@pytest.fixture
def mock_extract_local_response():
    return json.dumps(
        {
            "entities": [
                {
                    "name": "ORA-04030",
                    "type": "Term",
                    "definition": "Oracle PGA memory exhaustion error",
                    "confidence": 0.95,
                },
                {
                    "name": "PGA",
                    "type": "Term",
                    "definition": "Program Global Area",
                    "confidence": 0.90,
                },
                {
                    "name": "SGA",
                    "type": "Term",
                    "definition": "System Global Area",
                    "confidence": 0.92,
                },
                {
                    "name": "AWR Report",
                    "type": "Tool",
                    "definition": "Automatic Workload Repository report",
                    "confidence": 0.88,
                },
            ],
            "relations": [
                {
                    "source": "ORA-04030",
                    "source_type": "Term",
                    "target": "PGA",
                    "target_type": "Term",
                    "relation_type": "CAUSED_BY",
                    "confidence": 0.90,
                },
            ],
        }
    )


@pytest.fixture
def sample_documents():
    return [
        (
            "ORA-04030 is a critical Oracle Database error that occurs when "
            "the PGA (Program Global Area) memory allocation exceeds the limit. "
            "The DBA should use AWR Reports to diagnose the issue and check "
            "PGA_AGGREGATE_LIMIT and SGA_TARGET parameters."
        ),
        (
            "The Oracle DBA team is responsible for monitoring database performance. "
            "Key tools include AWR Reports, ASH Reports, and Enterprise Manager. "
            "SLA P1 incidents require a 1-hour response time."
        ),
    ]
