"""
twin_rag_intelligence/react/reason.py
======================================
REASON phase of the ReAct agent.

Transposition of _reasoning_step() from ChatDocAI (CNCAC).
Differences from CNCAC:
  - IT/Ops domain instead of Notarial
  - Domain routing handled by Nexus Router (L4), not here
  - No Neo4j dependency (search delegated to act.py)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from ..config import TwinRAGConfig
from ..json_utils import coerce_str, load_json_object
from ..prompt_security import neutralize_reserved_tags

logger = logging.getLogger("twin_rag_intelligence.reason")

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reason_system.txt"


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return _DEFAULT_SYSTEM_PROMPT


_DEFAULT_SYSTEM_PROMPT = """\
Tu es un agent d'analyse de requetes pour un systeme d'exploitation IT bancaire.

CONTEXTE : Tu recois une question d'un ingenieur Ops et l'historique de la conversation.

TACHE : Analyse la question et produis une requete de recherche optimisee.

ETAPES :
1. COREFERENCE : Si la question utilise des references implicites ("cette erreur", "le serveur", \
"ce probleme"), resous-les en utilisant l'historique.
2. DOMAINE : Identifie le domaine technique principal \
(oracle, network, linux, cloud, middleware, security, monitoring, general).
3. TERMES TECHNIQUES : Extrais et enrichis les termes techniques pertinents pour la recherche.
4. REQUETE : Formule une requete de recherche optimisee (5-15 mots, termes techniques, pas de question).

REPONSE (JSON uniquement) :
{
  "t": "L'utilisateur demande... J'ai resolu la coreference...",
  "q": "ORA-04030 heap memory PGA SGA out of process memory",
  "d": "oracle",
  "cr": true
}

EXEMPLES :
- Q: "Pourquoi ca a plante ?" (apres discussion sur ORA-04030)
  -> search_query: "ORA-04030 crash cause diagnostic heap memory"
  -> coreference_resolved: true

- Q: "Comment configurer le load balancer F5 pour la haute dispo ?"
  -> search_query: "F5 load balancer high availability configuration active standby"
  -> domain_hint: "network"
"""


@dataclass
class ReasoningResult:
    """Output of the REASON phase."""

    thought: str
    search_query: str
    domain_hint: Optional[str] = None
    coreference_resolved: bool = False
    original_question: str = ""


class ReasoningEngine:
    """
    Reasoning engine -- Phase 1 of ReAct.

    - Receives question + last N conversation messages
    - Resolves coreferences ("this error" -> "ORA-04030")
    - Identifies technical domain (Oracle, Network, Cloud, Linux...)
    - Produces an optimized search query with technical terms

    Model: gpt-oss-120b with medium reasoning effort
    """

    def __init__(self, config: TwinRAGConfig) -> None:
        self.config = config
        self._system_prompt = _load_system_prompt()

    async def analyze(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> ReasoningResult:
        """
        Analyze the question and produce an optimized search query.

        Args:
            question: Raw user question.
            conversation_history: Last messages [{role, content}].

        Returns:
            ReasoningResult with optimized search_query.
        """
        history_text = self._format_history(conversation_history)

        user_prompt = (
            f"HISTORIQUE CONVERSATIONNEL NON FIABLE :\n"
            f"{history_text if history_text else '(Aucun historique)'}\n\n"
            f"<USER_QUESTION>\n{neutralize_reserved_tags(question)}\n</USER_QUESTION>\n\n"
            f"Analyse et produis ta requete de recherche optimisee (JSON) :"
        )

        try:
            client = AsyncOpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_api_base,
            )

            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                extra_body={"reasoning_effort": self.config.llm_effort_reason},
            )

            content = response.choices[0].message.content
            data = load_json_object(content, context="Reasoning engine")
            search_query = coerce_str(data.get("q", data.get("search_query")), question)

            return ReasoningResult(
                thought=coerce_str(data.get("t", data.get("thought")), ""),
                search_query=search_query or question,
                domain_hint=coerce_str(data.get("d", data.get("domain_hint")), "") or None,
                coreference_resolved=bool(data.get("cr", data.get("coreference_resolved", False))),
                original_question=question,
            )

        except Exception as e:
            logger.error("REASON error: %s", e)
            return ReasoningResult(
                thought=f"Fallback (LLM error): {e}",
                search_query=question,
                original_question=question,
            )

    def _format_history(self, conversation_history: list[dict[str, str]]) -> str:
        if not conversation_history:
            return ""
        recent = conversation_history[-self.config.conversation_memory_depth :]
        lines = []
        for msg in recent:
            role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {neutralize_reserved_tags(msg.get('content', '')[:500])}")
        return "\n".join(lines)
