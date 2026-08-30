"""Pydantic wire models for Twin query routes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .._lightrag_compat import ANSWER_STATUS_GROUNDED, AnswerStatus
from .source_filters import TagFilter

# Hard cap on OR-groups per tag_filter: keeps the storage-side Cypher predicate
# bounded while covering the real operator use-case (a handful of tag families).
_TAG_FILTER_MAX_GROUPS = 5


class TagFilterGroup(BaseModel):
    """One conjunctive group of a grouped ``tag_filter``.

    Inside a group the flat semantics hold: every ``all`` tag must be present
    AND at least one ``any`` tag must be present (when the list is non-empty).
    """

    model_config = ConfigDict(extra="forbid")

    all: list[str] = Field(
        default_factory=list,
        description="Every listed tag must be present on the document (AND).",
    )
    any: list[str] = Field(
        default_factory=list,
        description="At least one listed tag must be present on the document.",
    )

    @model_validator(mode="after")
    def _require_one_tag(self) -> "TagFilterGroup":
        # A term-less group would be vacuously true and silently disable the
        # whole filter through the OR — refuse it at the wire boundary.
        if not any(term.strip() for term in [*self.all, *self.any]):
            raise ValueError("each tag_filter group needs at least one non-blank tag")
        return self


class TagFilterFlat(BaseModel):
    """Flat ``tag_filter`` form: ``all`` and ``any`` combine with AND."""

    model_config = ConfigDict(extra="forbid")

    all: list[str] = Field(
        default_factory=list,
        description="Every listed tag must be present on the document (AND).",
    )
    any: list[str] = Field(
        default_factory=list,
        description="At least one listed tag must be present on the document.",
    )


class TagFilterGrouped(BaseModel):
    """Grouped ``tag_filter`` form: a document matches when at least one
    group matches (OR between groups)."""

    model_config = ConfigDict(extra="forbid")

    groups: list[TagFilterGroup] = Field(
        min_length=1,
        max_length=_TAG_FILTER_MAX_GROUPS,
        description=(
            "OR-ed conjunctive groups; a document matches when at least "
            "one group matches."
        ),
    )


class ConversationMessage(BaseModel):
    """A caller-supplied conversational turn safe for LLM history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class TwinQueryBody(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What is the approval process for a new supplier?",
                    "mode": "mix",
                    "top_k": 20,
                    "min_score": 0.3,
                    "tag_filter": {"any": ["procurement"]},
                }
            ]
        }
    )

    query: str = Field(
        description="The question to answer from the knowledge base.",
        examples=["What is the approval process for a new supplier?"],
    )
    actor: str | None = Field(
        default=None,
        max_length=200,
        description="Optional caller name recorded in the audit trail.",
    )
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="mix",
        description=(
            "Retrieval strategy: `local` (entity-centred), `global` "
            "(relation-centred), `hybrid` (both), `naive` (plain vector "
            "search), `mix` (graph + vector, the recommended default)."
        ),
    )
    response_type: (
        Literal["Multiple Paragraphs", "Single Paragraph", "Bullet Points"] | None
    ) = Field(
        default=None,
        description="Requested answer format (default: multiple paragraphs).",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="How many entities/relations to retrieve for grounding.",
    )
    chunk_top_k: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="How many text chunks to keep after retrieval.",
    )
    max_entity_tokens: int | None = Field(
        default=None, ge=1, description="Token budget for the entity context."
    )
    max_relation_tokens: int | None = Field(
        default=None, ge=1, description="Token budget for the relation context."
    )
    max_total_tokens: int | None = Field(
        default=None, ge=1, description="Overall token budget for the prompt context."
    )
    only_need_context: bool = Field(
        default=False,
        description=("Return the retrieved context instead of generating an answer."),
    )
    only_need_prompt: bool = Field(
        default=False,
        description="Disabled on this API (prompt disclosure); must stay false.",
    )
    hl_keywords: list[str] = Field(
        default_factory=list,
        description="High-level keywords to steer retrieval (themes, concepts).",
    )
    ll_keywords: list[str] = Field(
        default_factory=list,
        description="Low-level keywords to steer retrieval (specific terms).",
    )
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=40,
        description="Previous turns, for follow-up questions.",
    )
    history_turns: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="How many history turns the model should consider.",
    )
    user_prompt: str | None = Field(
        default=None,
        max_length=4000,
        description="Disabled on this API (raw prompt override); must stay empty.",
    )
    enable_rerank: bool | None = Field(
        default=None,
        description="Force reranking on/off (default: deployment setting).",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Drop sources whose relevance score is below this threshold.",
    )
    tag_filter: TagFilterFlat | TagFilterGrouped | None = Field(
        default=None,
        description=(
            "Restrict retrieval to content whose document tags match "
            '(case-insensitive). Flat form: `{"all": [...]}` (every listed '
            'tag present) and/or `{"any": [...]}` (at least one present) — '
            "when both keys are provided, BOTH conditions must hold (they "
            'combine with AND, never OR). Grouped form: `{"groups": [{...}, '
            "{...}]}` where each group uses the flat semantics and a "
            "document matches when AT LEAST ONE group matches (OR between "
            "groups, 5 groups maximum). The two forms cannot be mixed in "
            "one request."
        ),
        examples=[
            {"any": ["procurement", "finance"]},
            {
                "groups": [
                    {"all": ["procurement", "emea"]},
                    {"any": ["global-policy"]},
                ]
            },
        ],
    )
    doc_filter: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Restrict retrieval to specific documents, same `any`/`all` "
            "flat shape as `tag_filter` (both conditions must hold when "
            "both keys are provided), matching document names or ids. The "
            "grouped form is not supported here."
        ),
        examples=[{"any": ["supplier-policy.pdf"]}],
    )
    fallback_to_mix: bool = Field(
        default=True,
        description=(
            "When the chosen mode yields nothing, retry with `mix` "
            "instead of returning an empty answer."
        ),
    )

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
        return

    @staticmethod
    def _check_flat_filter(value: dict[str, Any]) -> None:
        """Shared flat-form rules: keys ⊆ {all, any}, values = lists of str."""
        allowed_keys = {"all", "any"}
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            raise ValueError("advanced filter keys must be a subset of {'all', 'any'}")
        for terms in value.values():
            if not isinstance(terms, list) or not all(
                isinstance(term, str) for term in terms
            ):
                raise ValueError("advanced filter values must be lists of strings")

    @field_validator("doc_filter")
    @classmethod
    def _validate_doc_filter(
        cls, value: dict[str, list[str]] | None
    ) -> dict[str, list[str]] | None:
        if value is None:
            return None
        cls._check_flat_filter(value)
        return value

    @property
    def tag_filter_payload(self) -> TagFilter | None:
        """Wire dict of the validated ``tag_filter`` for the downstream chain.

        ``tag_filter`` itself stays the validated pydantic model — the union
        exists for validation and the OpenAPI schema, and keeping it intact
        keeps ``model_dump()`` warning-free. Everything downstream
        (``_tag_filter_groups`` normaliser, post-filters, ``QueryParam``
        attach, activity echo) speaks the plain payload dict, produced here
        explicitly.
        """
        if self.tag_filter is None:
            return None
        return self.tag_filter.model_dump(exclude_unset=True)


class TwinSourceAnchor(BaseModel):
    """Intra-chunk paragraph anchor for a citation (offsets only, no text).

    Heuristic and explicitly non-authoritative: it is published only when
    the lexical-overlap confidence clears the server-side floor, and
    consumers must render correctly without it. ``start``/``end`` are
    half-open offsets into the cited chunk's content counted in **Unicode
    code points** (Python string indices) — verifiable server-side as
    ``content[start:end]``. JavaScript strings count UTF-16 units, so JS
    consumers must slice on ``Array.from(content)`` (code points), never
    ``String.prototype.slice``, or any astral character (emoji, some CJK)
    before the paragraph shifts the range.
    """

    start: int = Field(
        ge=0,
        description=(
            "Start offset (inclusive) of the anchored paragraph within "
            "the chunk content, in Unicode code points."
        ),
    )
    end: int = Field(
        gt=0,
        description=(
            "End offset (exclusive) of the anchored paragraph within "
            "the chunk content, in Unicode code points (JavaScript "
            "consumers: slice on Array.from(content), not the UTF-16 "
            "String.prototype.slice)."
        ),
    )
    paragraph_idx: int = Field(
        ge=0, description="0-based index of the anchored paragraph in the chunk."
    )
    paragraph_count: int = Field(
        ge=1, description="Total number of paragraphs detected in the chunk."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Anchor confidence (0-1). Lowered when the citation's evidence "
            "is spread over several paragraphs."
        ),
    )
    method: str = Field(
        description="Anchoring method identifier.",
        examples=["lexical_overlap"],
    )


class TwinRetrievalSource(BaseModel):
    n: int = Field(description="Citation number, as referenced in the answer.")
    type: str = Field(default="file", description="Source type.")
    name: str = Field(description="Source document name.")
    meta: str | None = Field(
        default=None, description="Extra display information about the source."
    )
    # ``None`` means the aquery_llm retrieval envelope did not expose a real
    # numeric metric for any chunk behind this reference.  Do not substitute a
    # rank-derived value: callers must be able to distinguish measured
    # similarity from display order.
    score: float | None = Field(
        default=None,
        description=(
            "Measured relevance score (0-1), or null when the retrieval "
            "did not expose one."
        ),
    )
    retrieval_origin: Literal["vector", "graph"] | None = Field(
        default=None,
        description=(
            "Grounding provenance. Graph sources have no chunk-vector "
            "similarity and must not be displayed as missing data."
        ),
    )
    doc_id: str | None = Field(
        default=None, description="Id of the source document, when resolved."
    )
    chunk_id: str | None = Field(
        default=None,
        description=(
            "Id of the grounding chunk — usable with "
            "`GET /chunks/{chunk_id}/context`."
        ),
    )
    source_links: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Document-level provenance links inherited by this chunk. "
            "The server never fetches these URLs."
        ),
    )
    anchor: TwinSourceAnchor | None = Field(
        default=None,
        description=(
            "Optional paragraph-level anchor inside the cited chunk. "
            "A heuristic hint (absent when confidence is low), never a "
            "guarantee — render the plain chunk when null."
        ),
    )


class TwinQueryResponse(BaseModel):
    response: str = Field(description="The generated answer, as plain text.")
    model: str | None = Field(
        default=None,
        description="Configured LLM model used to synthesize the answer.",
        exclude_if=lambda value: value is None,
    )
    sources: list[TwinRetrievalSource] = Field(
        default_factory=list,
        description="The chunks the answer was grounded on.",
    )
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
    "TwinSourceAnchor",
]
