"""
twin_rag_intelligence/features/cognitive_reranker.py
=====================================================
F04: Cognitive Reranking v2.

Impact measured (CNCAC production):
  +45% answer completeness

Strategy "Expand -> Filter -> Select":
  1. EXPAND  : Retrieve 10+10 chunks (vector + fulltext)
  2. RERANK  : LLM scores each chunk on relevance (0-10)
  3. FILTER  : Keep only chunks >= 7.0/10
  4. SELECT  : Return top-8 chunks
"""

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

from ..config import TwinRAGConfig
from ..react.act import ChunkResult

logger = logging.getLogger("twin_rag_intelligence.reranker")

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reranking_system.txt"


def _load_rerank_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_RERANK_PROMPT


_DEFAULT_RERANK_PROMPT = """\
Pour chaque passage, evalue sa pertinence pour repondre a la question sur une echelle de 0 a 10.

ECHELLE :
- 10 : Contient directement la reponse technique complete
- 7-9 : Tres pertinent, information technique utile
- 4-6 : Contexte partiel, indirectement lie
- 1-3 : Faiblement lie au sujet
- 0 : Non pertinent

QUESTION : "{question}"

PASSAGES :
{passages}

REPONSE (JSON uniquement, pas de commentaire) :
{{"scores": [{{"passage": 0, "score": 8}}, {{"passage": 1, "score": 2}}, ...]}}
"""


class CognitiveReranker:
    """
    Cognitive reranking via LLM.

    Each chunk is individually evaluated by the LLM on its actual relevance
    for answering the question (vs simple cosine similarity).
    """

    def __init__(self, config: TwinRAGConfig) -> None:
        self.config = config
        self._rerank_prompt = _load_rerank_prompt()

    async def rerank(
        self,
        question: str,
        chunks: list[ChunkResult],
    ) -> list[ChunkResult]:
        """
        Rerank chunks by LLM relevance scoring.

        Args:
            question: Original question.
            chunks: Chunks to score.

        Returns:
            Filtered chunks (score >= threshold) sorted, top final_limit.
        """
        if not chunks:
            return []

        passages_text = "\n".join(
            f"--- Passage {i} ---\n{chunk.text[:800]}" for i, chunk in enumerate(chunks)
        )

        prompt = self._rerank_prompt.format(
            question=question,
            passages=passages_text,
        )

        try:
            client = AsyncOpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_api_base,
            )

            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500,
                extra_body={"reasoning_effort": self.config.llm_effort_reranker},
            )

            content = response.choices[0].message.content
            data = json.loads(content) if content else {}
            scores = data.get("scores", [])

            for score_entry in scores:
                idx = score_entry.get("passage", -1)
                score = score_entry.get("score", 0)
                if 0 <= idx < len(chunks):
                    chunks[idx].rerank_score = float(score)

            # Filter: keep only >= threshold
            filtered = [
                c for c in chunks if (c.rerank_score or 0) >= self.config.reranking_score_threshold
            ]

            # If filtering is too aggressive, fallback to top-K by raw score
            if len(filtered) < 2:
                logger.warning(
                    "Reranking too strict (%d chunks). Fallback top-%d",
                    len(filtered),
                    self.config.final_limit,
                )
                chunks.sort(key=lambda c: c.rerank_score or c.score, reverse=True)
                return chunks[: self.config.final_limit]

            # Select: top final_limit
            filtered.sort(key=lambda c: c.rerank_score or 0, reverse=True)
            result = filtered[: self.config.final_limit]

            logger.info(
                "Reranking: %d -> %d (>=%s) -> %d selected",
                len(chunks),
                len(filtered),
                self.config.reranking_score_threshold,
                len(result),
            )
            return result

        except Exception as e:
            logger.error("Reranking error: %s", e)
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[: self.config.final_limit]
