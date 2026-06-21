"""Server configuration for the LightRAG FastAPI wrapper.

All settings are read from the environment with prefix LIGHTRAG_.
For API keys, if the LIGHTRAG_* key is not set, code may fall back
to OPENAI_API_KEY where applicable.
"""

import json

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

SettingsConfigDict = ConfigDict


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
    max_request_body_bytes: int = Field(
        default=16 * 1024 * 1024,
        ge=0,
        description=(
            "Maximum Content-Length accepted for ordinary HTTP request bodies. "
            "0 disables the guard."
        ),
    )
    max_upload_body_bytes: int = Field(
        default=100 * 1024 * 1024,
        ge=0,
        description=(
            "Maximum Content-Length accepted for native document upload bodies. "
            "0 disables the guard."
        ),
    )

    # -- WebUI phase-1 surface --
    enable_webui_routes: bool = Field(
        default=True,
        description=(
            "Mount the WebUI phase-1 router (/documents, /folders, /tags, "
            "/activity, /graph/*, etc.) backed by runtime stores. Set "
            "False to expose only the LightRAG core endpoints."
        ),
    )
    webui_tag_backend: str = Field(
        default="memgraph",
        description=(
            "Persistence backend for WebUI tag governance. 'memgraph' "
            "(default) persists them as :WebuiTag_{workspace} / "
            ":WebuiTagCategory_{workspace} nodes. Production app wiring boots "
            "fresh folders without demo tags; only governance categories are "
            "bootstrapped unless an explicit seed/bootstrap path is used. "
            "'memory' keeps tags + categories in the seed-loaded in-process "
            "store and is demo/dev only."
        ),
    )
    webui_activity_backend: str = Field(
        default="memgraph",
        description=(
            "Persistence backend for the WebUI /activity audit feed. "
            "'memgraph' (default) persists events as "
            ":WebuiActivity_{workspace} nodes and starts empty. 'memory' "
            "seeds an in-memory list and is demo/dev only."
        ),
    )
    webui_notifications_backend: str = Field(
        default="memgraph",
        description=(
            "Persistence backend for the WebUI /notifications surface. "
            "'memgraph' (default) persists notifications as "
            ":WebuiNotification_{workspace} nodes and starts empty. 'memory' "
            "seeds an in-memory list and is demo/dev only."
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

    # -- CORS --
    cors_allowed_origins: list[str] | str = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:9621",
            "http://127.0.0.1:9621",
        ],
        description=(
            "Allowed browser origins for credentialed CORS. Use a JSON list or "
            "comma-separated string in LIGHTRAG_CORS_ALLOWED_ORIGINS."
        ),
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description=(
            "Whether CORS responses may include credentials. Must be false "
            "when LIGHTRAG_CORS_ALLOWED_ORIGINS contains '*'."
        ),
    )

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_allowed_origins(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return json.loads(stripped)
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return value


def get_settings() -> LightRAGServerSettings:
    """Load LightRAG server settings from environment."""
    return LightRAGServerSettings()
