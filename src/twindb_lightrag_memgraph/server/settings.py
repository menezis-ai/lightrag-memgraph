"""Server configuration for the LightRAG FastAPI wrapper.

All settings are read from the environment with prefix LIGHTRAG_.
For API keys, if the LIGHTRAG_* key is not set, code may fall back
to OPENAI_API_KEY where applicable.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LightRAGServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIGHTRAG_",
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # -- Server --
    working_dir: str = Field(
        default="./kg_working_dir_data",
        description="Working directory for LightRAG storage (local files)",
    )
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=9621, description="Bind port")

    # -- LLM (OpenAI-compatible) --
    llm_model: str = Field(default="deepseek-chat", description="LLM model name")
    llm_binding_host: str = Field(
        default="https://api.deepseek.com",
        description="LLM API base URL",
    )
    llm_binding_api_key: str | None = Field(
        default=None,
        description="LLM API key (fallback: OPENAI_API_KEY)",
    )

    # -- Embedding --
    embedding_binding: str = Field(
        default="openai",
        description="Embedding binding: 'openai' or 'ollama'",
    )
    embedding_model: str = Field(default="bge-m3", description="Embedding model name")
    embedding_binding_host: str = Field(
        default="http://localhost:11434",
        description="Embedding API host (Ollama or OpenAI-compatible)",
    )
    embedding_binding_api_key: str | None = Field(
        default=None,
        description="Embedding API key (fallback: OPENAI_API_KEY)",
    )
    embedding_dim: int = Field(default=1024, description="Embedding dimension")
    max_embed_tokens: int = Field(
        default=1200, description="Max tokens per embedding input"
    )

    # -- Reranking --
    reranking_model: str = Field(
        default="bge-reranker-base", description="Rerank model name"
    )
    reranking_base_url: str = Field(
        default="http://localhost:8000",
        description="Rerank API base URL",
    )
    reranking_api_key: str | None = Field(
        default=None,
        description="Rerank API key (fallback: OPENAI_API_KEY)",
    )

    # -- Tokenizer --
    tokenizer_path: str | None = Field(
        default=None,
        description="Path to local tokenizer dir (e.g. HuggingFace model)",
    )

    # -- Storage backends (via register() L1 patch) --
    kv_storage: str = Field(
        default="MemgraphKVStorage", description="KV storage class name"
    )
    vector_storage: str = Field(
        default="MemgraphVectorDBStorage", description="Vector storage class name"
    )
    graph_storage: str = Field(
        default="MemgraphStorage", description="Graph storage class name"
    )
    doc_status_storage: str = Field(
        default="MemgraphDocStatusStorage",
        description="Doc status storage class name",
    )

    # -- Workspace --
    workspace: str = Field(
        default="",
        description="Workspace name for data isolation (empty = default workspace)",
    )

    # -- Chunking --
    chunk_token_size: int = Field(
        default=1200,
        description="Maximum tokens per chunk when splitting documents",
    )
    chunk_overlap_token_size: int = Field(
        default=100,
        description="Token overlap between consecutive chunks",
    )

    # -- Entity extraction --
    max_gleaning: int = Field(
        default=2,
        description="Max entity extraction attempts for ambiguous content",
    )

    # -- Observability --
    enable_langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing for LLM/embedding/rerank spans",
    )

    # -- WebUI phase-1 surface --
    enable_webui_routes: bool = Field(
        default=True,
        description=(
            "Mount the WebUI phase-1 router (/documents, /workspaces, /tags, "
            "/activity, /graph/*, etc.) backed by in-memory seed data. Set "
            "False to expose only the LightRAG core endpoints."
        ),
    )
    webui_tag_backend: str = Field(
        default="memory",
        description=(
            "Persistence backend for WebUI tag governance. 'memory' (default) "
            "keeps tags + categories in the seed-loaded in-process store. "
            "'memgraph' persists them as :WebuiTag_{workspace} / "
            ":WebuiTagCategory_{workspace} nodes and bootstraps from the seed "
            "on first init when the workspace KV is empty."
        ),
    )
    webui_activity_backend: str = Field(
        default="memory",
        description=(
            "Persistence backend for the WebUI /activity audit feed. "
            "'memory' (default) seeds an in-memory list. 'memgraph' "
            "persists events as :WebuiActivity_{workspace} nodes."
        ),
    )
    webui_notifications_backend: str = Field(
        default="memory",
        description=(
            "Persistence backend for the WebUI /notifications surface. "
            "'memory' (default) seeds an in-memory list. 'memgraph' "
            "persists notifications as :WebuiNotification_{workspace} nodes."
        ),
    )

    # -- Auth --
    api_key: str | None = Field(
        default=None,
        description="Static API key for agent auth (Bearer token). None = no auth.",
    )
    jwt_secret: str | None = Field(
        default=None,
        description="JWT secret for token-based auth. None = JWT disabled.",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_hours: int = Field(
        default=4, description="JWT token expiration in hours"
    )
    jwt_username: str = Field(
        default="admin", description="Username for /login endpoint"
    )
    jwt_password: str = Field(
        default="changeme", description="Password for /login endpoint"
    )


def get_settings() -> LightRAGServerSettings:
    """Load LightRAG server settings from environment."""
    return LightRAGServerSettings()
