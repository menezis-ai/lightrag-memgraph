"""Pydantic wire models for Twin query routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .._lightrag_compat import ANSWER_STATUS_GROUNDED, AnswerStatus


class TwinQueryBody(BaseModel):
    query: str
    actor: str | None = Field(default=None, max_length=200)
    mode: str = Field(default="mix")
    response_type: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)
    chunk_top_k: int | None = Field(default=None, ge=1, le=200)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    only_need_context: bool = Field(default=False)
    only_need_prompt: bool = Field(default=False)
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    history_turns: int | None = Field(default=None, ge=0, le=20)
    user_prompt: str | None = Field(default=None, max_length=4000)
    enable_rerank: bool | None = Field(default=None)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tag_filter: dict[str, list[str]] | None = Field(default=None)
    doc_filter: dict[str, list[str]] | None = Field(default=None)
    fallback_to_mix: bool = Field(default=True)

    @field_validator("tag_filter", "doc_filter")
    @classmethod
    def _validate_advanced_filter(
        cls, value: dict[str, list[str]] | None
    ) -> dict[str, list[str]] | None:
        if value is None:
            return None
        allowed_keys = {"all", "any"}
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            raise ValueError("advanced filter keys must be a subset of {'all', 'any'}")
        return value


class TwinRetrievalSource(BaseModel):
    n: int
    type: str = "file"
    name: str
    meta: str | None = None
    score: float = 0.0
    doc_id: str | None = None
    chunk_id: str | None = None


class TwinQueryResponse(BaseModel):
    response: str
    sources: list[TwinRetrievalSource] = Field(default_factory=list)
    # TR-RET-02: ``"insufficient_information"`` when LightRAG signalled
    # no usable retrieval context (canonical ``[no-context]`` marker in
    # the fail response). The React port uses this to suppress the
    # Sources panel honestly rather than parsing the LLM prose.
    # Typed as ``AnswerStatus`` so the generated OpenAPI schema
    # advertises the enum to clients/tooling instead of an open str.
    answer_status: AnswerStatus = Field(default=ANSWER_STATUS_GROUNDED)


class TwinQueryDataResponse(BaseModel):
    status: str = "success"
    message: str = "Query executed successfully"
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "TwinQueryBody",
    "TwinQueryDataResponse",
    "TwinQueryResponse",
    "TwinRetrievalSource",
]
