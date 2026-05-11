"""
intelligence/ontology/steps/extract.py
=======================================
EXTRACT step -- LLM call to extract entities and relations from documents.

Prompt includes DSEP constraint block. Mode determines extraction bias:
- dedicated: biased toward the configured subject
- emergence: neutral extraction
- deep_extraction: full DSEP operators with explicit reasoning
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from ...config import TwinRAGConfig
from ..config import WorkspaceOntologyConfig
from ..dsep import build_dsep_block, get_mode_defaults, get_pass_defaults

logger = logging.getLogger("twin_rag_intelligence.ontology.extract")

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "ontology_extract.txt"


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    definition: str = ""
    confidence: float = 0.8
    properties: dict = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    source: str
    source_type: str
    target: str
    target_type: str
    relation_type: str
    confidence: float = 0.8


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    source_doc: str = ""


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_PROMPT


_DEFAULT_PROMPT = """\
You are an ontology extraction agent.

{mode_instruction}

{dsep_block}

DOCUMENT:
{document}

TASK: Extract all entities and relationships from this document.

RESPONSE (JSON):
{{
  "entities": [
    {{"name": "...", "type": "Term|Role|Team|Tool|Process|Asset", "definition": "...", "confidence": 0.9}}
  ],
  "relations": [
    {{"source": "...", "source_type": "...", "target": "...", "target_type": "...", \
"relation_type": "SYNONYM|RELATED_TO|CAUSED_BY|...", "confidence": 0.8}}
  ]
}}
"""


_MODE_INSTRUCTIONS = {
    "dedicated": (
        "You are a domain expert in {subject}. "
        "Focus extraction on concepts, tools, and processes related to {subject}. "
        "Context: {context}"
    ),
    "emergence": (
        "Extract all entities and relationships from this document neutrally. "
        "Do NOT apply predefined categories. Let domains emerge from the content. "
        "Context: {context}"
    ),
    "deep_extraction": (
        "You are performing deep symbolic analysis. "
        "Apply ALL DSEP operators below with explicit reasoning. "
        "Context: {context}"
    ),
}

_PASS_MODE_INSTRUCTIONS = {
    "global": (
        "Extract high-level domains, processes, organizational structure. "
        "Focus on Domain, Process, Team, and Methodology entities. "
        "Context: {context}"
    ),
    "local": (
        "Extract precise technical entities, error codes, causal relationships. "
        "Focus on Term, Tool, Asset entities and CAUSED_BY, DIAGNOSED_WITH relations. "
        "Context: {context}"
    ),
}


def _truncate_clean(text: str, max_chars: int) -> str:
    """Truncate text at the last paragraph or sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try paragraph boundary first
    para_idx = truncated.rfind("\n\n")
    if para_idx > max_chars // 2:
        return truncated[:para_idx]
    # Fall back to sentence boundary
    sent_idx = truncated.rfind(". ")
    if sent_idx > max_chars // 2:
        return truncated[: sent_idx + 1]
    return truncated


def _resolve_operators(
    ws_config: WorkspaceOntologyConfig,
    pass_type: str | None,
    mode: str,
) -> list[str]:
    """Resolve DSEP operators based on pass type and workspace config."""
    if pass_type == "global":
        return ws_config.dsep_operators_global or get_pass_defaults("global")
    if pass_type == "local":
        return ws_config.dsep_operators_local or get_pass_defaults("local")
    return ws_config.dsep_operators or get_mode_defaults(mode)


async def extract(
    document: str,
    doc_id: str,
    config: TwinRAGConfig,
    ws_config: WorkspaceOntologyConfig,
    dsep_enabled: bool = True,
    pass_type: str | None = None,
    global_max_tokens: int = 20000,
) -> ExtractionResult:
    """Extract entities and relations from a document via LLM.

    Args:
        document: Document text content.
        doc_id: Document identifier.
        config: TwinRAG configuration.
        ws_config: Workspace ontology configuration.
        dsep_enabled: Whether to include DSEP constraints.
        pass_type: None for single-pass (default behavior),
            "global" for high-level structure, "local" for precise entities.
        global_max_tokens: Max tokens for global pass (~3 chars/token
            for technical IT content with acronyms and codes).

    Returns:
        ExtractionResult with extracted entities and relations.
    """
    mode = ws_config.mode

    # Build DSEP block
    dsep_block = ""
    if dsep_enabled:
        operators = _resolve_operators(ws_config, pass_type, mode)
        dsep_block = build_dsep_block(operators, mode)

    # Build mode instruction
    if pass_type and pass_type in _PASS_MODE_INSTRUCTIONS:
        mode_instruction = _PASS_MODE_INSTRUCTIONS[pass_type].format(
            context=ws_config.context,
        )
    else:
        mode_template = _MODE_INSTRUCTIONS.get(
            mode, _MODE_INSTRUCTIONS["emergence"]
        )
        mode_instruction = mode_template.format(
            subject=ws_config.subject,
            context=ws_config.context,
        )

    # Truncate document at a clean boundary (paragraph or sentence)
    if pass_type == "global":
        max_chars = global_max_tokens * 3  # ~3 chars/token for technical IT text
        doc_text = _truncate_clean(document, max_chars)
    else:
        doc_text = _truncate_clean(document, 24000)

    # Build full prompt
    prompt_template = _load_prompt()
    prompt = prompt_template.format(
        mode_instruction=mode_instruction,
        dsep_block=dsep_block,
        document=doc_text,
    )

    try:
        client = AsyncOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_api_base,
        )

        response = await client.chat.completions.create(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=2000,
        )

        content = response.choices[0].message.content
        data = json.loads(content) if content else {}

        entities = [
            ExtractedEntity(
                name=e["name"],
                entity_type=e.get("type", "Term"),
                definition=e.get("definition", ""),
                confidence=e.get("confidence", 0.8),
                properties={
                    k: v
                    for k, v in e.items()
                    if k not in ("name", "type", "definition", "confidence")
                },
            )
            for e in data.get("entities", [])
        ]

        relations = [
            ExtractedRelation(
                source=r["source"],
                source_type=r.get("source_type", "Term"),
                target=r["target"],
                target_type=r.get("target_type", "Term"),
                relation_type=r.get("relation_type", "RELATED_TO"),
                confidence=r.get("confidence", 0.8),
            )
            for r in data.get("relations", [])
        ]

        return ExtractionResult(
            entities=entities,
            relations=relations,
            source_doc=doc_id,
        )

    except Exception as e:
        logger.error("Extraction error for doc %s: %s", doc_id, e)
        return ExtractionResult(source_doc=doc_id)
