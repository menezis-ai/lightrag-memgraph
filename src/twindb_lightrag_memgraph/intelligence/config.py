"""
twin_rag_intelligence/config.py
================================
Centralized configuration via environment variables.
All features are toggleable via feature flags.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class TwinRAGConfig(BaseSettings):
    """RAG Intelligence engine configuration."""

    model_config = {"env_prefix": "TWIN_RAG_"}

    # --- LLM Configuration ---
    llm_provider: str = "openai_compatible"
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_api_version: str = "2025-01-01"

    # Single model, reasoning effort varies per cognitive task
    llm_model: str = "gpt-oss-120b"

    # Reasoning effort per phase (low / medium / high)
    llm_effort_intent: str = "low"           # F05: simple classification
    llm_effort_reason: str = "medium"        # REASON: coreference + domain detection
    llm_effort_reranker: str = "low"         # F04: passage scoring
    llm_effort_synthesis: str = "high"       # OBSERVE: complex synthesis + citations

    # --- Feature Flags (AFFINE) ---
    enable_oos_detection: bool = True
    enable_query_expansion: bool = True
    enable_cognitive_reranking: bool = True
    enable_feedback: bool = True

    # --- Feature Thresholds ---
    oos_confidence_threshold: float = 0.85
    domain_confidence_threshold: float = 0.70
    reranking_score_threshold: float = 7.0

    # --- Search Parameters ---
    vector_limit: int = 10
    fulltext_limit: int = 10
    final_limit: int = 8
    conversation_memory_depth: int = 10

    # --- Query Expansion ---
    max_synonyms_per_term: int = 3
    max_total_synonyms: int = 5

    # --- Memgraph / LightRAG ---
    memgraph_uri: str = "bolt://localhost:7687"
    memgraph_workspace: str = "commons"
    lightrag_mode: str = "hybrid"

    # --- Ontology ---
    enable_ontology: bool = False  # Overridden by ontology.json if present

    # --- Observability ---
    enable_tracing: bool = True
    log_level: str = "INFO"
