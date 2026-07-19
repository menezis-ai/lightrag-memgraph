"""Pydantic wire models for Twin query routes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .._lightrag_compat import ANSWER_STATUS_GROUNDED, AnswerStatus


class ConversationMessage(BaseModel):
    """A caller-supplied conversational turn safe for LLM history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class TwinQueryBody(BaseModel):
    query: str
    actor: str | None = Field(default=None, max_length=200)
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    response_type: (
        Literal["Multiple Paragraphs", "Single Paragraph", "Bullet Points"] | None
    ) = None
    top_k: int = Field(default=20, ge=1, le=200)
    chunk_top_k: int | None = Field(default=None, ge=1, le=200)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    only_need_context: bool = Field(default=False)
    only_need_prompt: bool = Field(default=False)
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list, max_length=40
    )
    history_turns: int | None = Field(default=None, ge=0, le=20)
    user_prompt: str | None = Field(default=None, max_length=4000)
    enable_rerank: bool | None = Field(default=None)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tag_filter: dict[str, list[str]] | None = Field(default=None)
    doc_filter: dict[str, list[str]] | None = Field(default=None)
    fallback_to_mix: bool = Field(default=True)

    @field_validator("only_need_prompt")
    @classmethod
    def _reject_prompt_disclosure(cls, value: bool) -> bool:
        if value:
            raise ValueError("only_need_prompt is disabled on the external API")
        return value

    @field_validator("user_prompt")
    @classmethod
    def _reject_raw_prompt_override(cls, value: str | None) -> None:
        if value and value.strip():
            raise ValueError("raw user_prompt overrides are disabled")
        return None

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
    # ``None`` means the aquery_llm retrieval envelope did not expose a real
    # numeric metric for any chunk behind this reference.  Do not substitute a
    # rank-derived value: callers must be able to distinguish measured
    # similarity from display order.
    score: float | None = None
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
    "ConversationMessage",
    "TwinQueryBody",
    "TwinQueryDataResponse",
    "TwinQueryResponse",
    "TwinRetrievalSource",
]
