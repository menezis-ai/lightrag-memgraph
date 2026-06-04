"""Tests for server settings (Pydantic BaseSettings)."""

import os

import pytest

from twindb_lightrag_memgraph.server.settings import (
    LightRAGServerSettings,
    get_settings,
)


class TestLightRAGServerSettings:
    def test_defaults(self):
        s = LightRAGServerSettings()
        assert s.host == "0.0.0.0"
        assert s.port == 9621
        assert s.kv_storage == "MemgraphKVStorage"
        assert s.vector_storage == "MemgraphVectorDBStorage"
        assert s.graph_storage == "MemgraphStorage"
        assert s.doc_status_storage == "MemgraphDocStatusStorage"
        assert s.enable_langsmith_tracing is False
        assert s.api_key is None
        assert s.jwt_secret is None
        assert s.jwt_expiration_hours == 4
        assert s.cors_allow_credentials is True
        assert "http://127.0.0.1:4173" in s.cors_allowed_origins
        assert s.embedding_dim == 1024
        assert s.chunk_token_size == 1200
        assert s.chunk_overlap_token_size == 100

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_PORT", "8888")
        monkeypatch.setenv("LIGHTRAG_WORKSPACE", "test_ws")
        monkeypatch.setenv("LIGHTRAG_ENABLE_LANGSMITH_TRACING", "true")
        s = LightRAGServerSettings()
        assert s.port == 8888
        assert s.workspace == "test_ws"
        assert s.enable_langsmith_tracing is True

    def test_auth_settings(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_API_KEY", "sk-test-123")
        monkeypatch.setenv("LIGHTRAG_JWT_SECRET", "super-secret")
        monkeypatch.setenv("LIGHTRAG_JWT_EXPIRATION_HOURS", "8")
        s = LightRAGServerSettings()
        assert s.api_key == "sk-test-123"
        assert s.jwt_secret == "super-secret"
        assert s.jwt_expiration_hours == 8

    def test_cors_allowed_origins_comma_separated(self, monkeypatch):
        monkeypatch.setenv(
            "LIGHTRAG_CORS_ALLOWED_ORIGINS",
            "https://spa.example, https://admin.example",
        )
        s = LightRAGServerSettings()
        assert s.cors_allowed_origins == [
            "https://spa.example",
            "https://admin.example",
        ]

    def test_get_settings_returns_instance(self):
        s = get_settings()
        assert isinstance(s, LightRAGServerSettings)
