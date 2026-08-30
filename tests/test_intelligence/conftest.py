"""
Common test fixtures: mock LLM, mock LightRAG, mock Memgraph.
All unit tests run WITHOUT external infrastructure.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twindb_lightrag_memgraph.intelligence.config import TwinRAGConfig
from twindb_lightrag_memgraph.intelligence.features.workspace_router import (
    TopologyContext,
    WorkspaceRouter,
)
from twindb_lightrag_memgraph.intelligence.react.act import ChunkResult


@pytest.fixture
def config():
    return TwinRAGConfig(
        llm_api_key="test-key",
        llm_api_base="http://mock:8080",
        enable_oos_detection=True,
        enable_query_expansion=True,
        enable_cognitive_reranking=True,
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
def sample_chunks():
    """Sample ChunkResult list for testing."""
    return [
        ChunkResult(
            chunk_id="chunk_0",
            text="ORA-04030 occurs when the PGA memory allocation exceeds the limit. "
            "Check the PGA_AGGREGATE_LIMIT parameter and consider increasing it.",
            score=0.95,
            source_workspace="demo",
            document_id="doc_oracle_001",
            document_path="/docs/oracle/memory.pdf",
        ),
        ChunkResult(
            chunk_id="chunk_1",
            text="To diagnose heap memory issues, use V$PROCESS_MEMORY and V$SESSTAT views. "
            "Monitor the PGA used memory vs allocated.",
            score=0.88,
            source_workspace="commons",
            document_id="doc_oracle_002",
            document_path="/docs/oracle/diagnostic.pdf",
        ),
        ChunkResult(
            chunk_id="chunk_2",
            text="The VLAN configuration guide covers segmentation for production environments. "
            "Use 802.1Q trunk between switches.",
            score=0.45,
            source_workspace="commons",
            document_id="doc_network_001",
            document_path="/docs/network/vlan_guide.pdf",
        ),
        ChunkResult(
            chunk_id="chunk_3",
            text="Kubernetes pod autoscaling requires HPA configuration with appropriate "
            "CPU and memory thresholds. Use kubectl top to monitor.",
            score=0.30,
            source_workspace="commons",
            document_id="doc_k8s_001",
            document_path="/docs/cloud/k8s_scaling.pdf",
        ),
    ]


@pytest.fixture
def intent_in_scope_json():
    return json.dumps(
        {
            "intent": "IN_SCOPE",
            "confidence": 0.95,
            "reason": "Technical Oracle question",
        }
    )


@pytest.fixture
def intent_oos_json():
    return json.dumps(
        {"intent": "OUT_OF_SCOPE", "confidence": 0.97, "reason": "Weather question"}
    )


@pytest.fixture
def intent_greeting_json():
    return json.dumps(
        {"intent": "GREETING", "confidence": 0.99, "reason": "User greeting"}
    )


@pytest.fixture
def intent_malicious_json():
    return json.dumps(
        {"intent": "MALICIOUS", "confidence": 0.98, "reason": "Jailbreak attempt"}
    )


@pytest.fixture
def reasoning_result_json():
    return json.dumps(
        {
            "thought": "User asks about ORA-04030 memory error",
            "search_query": "ORA-04030 heap memory PGA SGA out of process memory",
            "domain_hint": "oracle",
            "coreference_resolved": False,
        }
    )


@pytest.fixture
def routing_rules_json(tmp_path):
    """Test routing_rules.json file."""
    rules = {
        "default_workspace": "commons",
        "rules": [
            {
                "keywords": ["Oracle", "ORA-", "RMAN"],
                "target_workspace": "commons_oracle",
                "workspace_type": "public",
                "confidence": 1.0,
            },
            {
                "keywords": ["RedHat", "RHEL", "Linux"],
                "target_workspace": "commons_linux",
                "workspace_type": "public",
                "confidence": 0.9,
            },
            {
                "keywords": ["demo-app", "Demo"],
                "target_workspace": "demo",
                "workspace_type": "private",
                "confidence": 0.95,
            },
        ],
    }
    path = tmp_path / "routing_rules.json"
    path.write_text(json.dumps(rules))
    return path


@pytest.fixture
def router(routing_rules_json):
    """WorkspaceRouter loaded from test rules."""
    return WorkspaceRouter.from_json(routing_rules_json)


@pytest.fixture
def topology_context_demo():
    """TopologyContext for Demo workspace."""
    return TopologyContext(
        servers=["srv-demo-01"],
        workspaces=["demo"],
        workspaces_publics=["commons", "commons_oracle"],
        topology_path="(App:RH)-[:RUNS_ON]->(srv-demo-01)",
    )


@pytest.fixture
def reasoning_coref_json():
    return json.dumps(
        {
            "thought": "User refers to 'this error' which is ORA-04030 from history",
            "search_query": "ORA-04030 crash cause diagnostic heap memory",
            "domain_hint": "oracle",
            "coreference_resolved": True,
        }
    )


@pytest.fixture
def reranking_scores_json():
    return json.dumps(
        {
            "scores": [
                {"passage": 0, "score": 9},
                {"passage": 1, "score": 8},
                {"passage": 2, "score": 3},
                {"passage": 3, "score": 1},
            ]
        }
    )
