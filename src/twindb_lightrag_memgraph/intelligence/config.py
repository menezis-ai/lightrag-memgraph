"""
twin_rag_intelligence/config.py
================================
Centralized configuration via environment variables.
All features are toggleable via feature flags.
"""

from typing import Optional

from pydantic_settings import BaseSettings

from .._constants import DEFAULT_MEMGRAPH_URI


class TwinRAGConfig(BaseSettings):
    """RAG Intelligence engine configuration."""

    model_config = {"env_prefix": "TWIN_RAG_"}

    # --- Chat LLM (dev/staging — user-facing queries) ---
    llm_provider: str = "openai_compatible"
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_api_version: str = "2025-01-01"

    # Single model, reasoning effort varies per cognitive task
    llm_model: str = "gpt-oss-120b"

    # Reasoning effort per phase (low / medium / high)
    llm_effort_intent: str = "low"  # F05: simple classification
    llm_effort_reason: str = "medium"  # REASON: coreference + domain detection
    llm_effort_reranker: str = "low"  # F04: passage scoring
    llm_effort_synthesis: str = "high"  # OBSERVE: complex synthesis + citations

    # --- Indexing LLM (prod — document processing pipeline) ---
    # Separate credentials to isolate GPU resources: dev chat doesn't
    # compete with prod document ingestion.
    # Falls back to chat LLM config when unset.
    indexing_api_key: Optional[str] = None
    indexing_api_base: Optional[str] = None
    indexing_model: Optional[str] = None
    indexing_embedding_model: str = "text-embedding-3-small"
    indexing_embedding_dim: int = 1536

    # --- Feature Flags (AFFINE) ---
    enable_oos_detection: bool = True
    enable_query_expansion: bool = True
    enable_cognitive_reranking: bool = True
    enable_feedback: bool = True
    enable_folder_routing: bool = True
    enable_workspace_routing: Optional[bool] = None  # Deprecated: use enable_folder_routing.

    # --- F06 Folder Router ---
    folder_routing_rules_path: str = (
        ""  # Path to routing_rules.json. Empty = use embedded default.
    )
    default_folder: str = "commons"
    routing_rules_path: str = ""  # Deprecated: use folder_routing_rules_path.
    default_workspace: Optional[str] = None  # Deprecated: use default_folder.

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
    memgraph_uri: str = DEFAULT_MEMGRAPH_URI
    memgraph_workspace: str = "commons"
    lightrag_mode: str = "hybrid"

    # --- Ontology ---
    enable_ontology: bool = False  # Overridden by ontology.json if present

    # --- Observability ---
    enable_tracing: bool = True
    log_level: str = "INFO"

    @property
    def effective_enable_folder_routing(self) -> bool:
        """Return the canonical folder-routing flag with legacy env compatibility."""
        if self.enable_workspace_routing is not None:
            return self.enable_workspace_routing
        return self.enable_folder_routing

    @property
    def effective_folder_routing_rules_path(self) -> str:
        """Return the configured folder routing rules path."""
        return self.folder_routing_rules_path or self.routing_rules_path

    @property
    def effective_default_folder(self) -> str:
        """Return the default folder, accepting the legacy workspace alias."""
        return self.default_workspace or self.default_folder
