"""Intra-chunk paragraph anchoring for Twin citations (phases A and B1).

Design record: ``PARAGRAPH-CITATION-PLAN.md`` §5/§5.1. Twin cites at the
chunk; this module locates the paragraph *inside* the already-stored chunk
content that most plausibly supports a ``[n]`` citation, without any extra
LLM call and without retaining anything new (offsets only, never paragraph
text — returning paragraph bodies was rejected because the WebUI persists
whole threads to localStorage).

Three cooperating pieces, all pure (no I/O, no LLM — mutmut-eligible):

- :func:`split_paragraphs` — deterministic blank-line split with offsets
  into the ORIGINAL string, so ``content[start:end]`` is verifiable by any
  consumer. The string is never normalized in place (CRLF handled by the
  separator pattern, not by rewriting).
- :class:`CitationEvidenceCollector` — bounded lexical collector for the
  answer sentences carrying ``[n]`` markers. Never materialises the answer:
  it keeps only normalized token sets, citation ids and overflow flags,
  bounded by the three §5.1 caps (:data:`MAX_TOKEN_CHARS`,
  :data:`MAX_TOKENS_PER_SENTENCE`, :data:`MAX_EVIDENCE_TOKENS_TOTAL`).
  Both the non-stream and the streaming query paths feed the same class so
  the published anchor is identical on both (§5.1 parity requirement).
- :func:`compute_anchor` — lexical-overlap scorer. One anchor per citation
  (the best paragraph against the union of that citation's sentences), with
  ``confidence`` penalized when the top-2 paragraphs are close (dispersed
  evidence is exactly what the operator should be told about). Below
  :data:`MIN_ANCHOR_CONFIDENCE`, or on incomplete (overflowed) evidence, no
  anchor is produced — fail-soft, never a truncated result presented as
  reliable.

The anchor is an explicitly NON-authoritative hint (§9): consumers must
render correctly without it, and enrichment failures upstream must leave
the fail-closed source projection contract untouched.

Known residual, accepted for phase A: the answer's trailing
``### References`` block is part of the scored text (the server never
strips it — that is the client's job on the joined stream), so reference
list lines dilute the evidence union slightly. The confidence threshold is
calibrated with that noise present. Phase B1 adds the STRUCTURAL election
(:func:`compute_best_structural_anchor`, over ingestion-persisted
``twin_block_boundaries``) which replaces the paragraph-splitting heuristic
for seam-ingested chunks — but the References-block dilution sits on the
EVIDENCE-collection side, shared by both methods, so it remains an accepted
residual for structural anchors too.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# §5.1 caps — bound collector memory independently of answer length and
# citation rate. Exceeding any of them marks the affected citation
# incomplete, which suppresses its anchor (fail-soft) instead of scoring a
# truncation presented as reliable.
MAX_TOKEN_CHARS = 64
MAX_TOKENS_PER_SENTENCE = 256
MAX_EVIDENCE_TOKENS_TOTAL = 4096

# Minimum post-penalty confidence to publish an anchor. Calibrated on the
# unit corpus in tests/test_server/test_paragraph_anchor.py — a cited
# sentence paraphrasing one paragraph lands well above; evidence spread
# evenly across paragraphs lands below.
MIN_ANCHOR_CONFIDENCE = 0.2

ANCHOR_METHOD_LEXICAL = "lexical_overlap"
# Phase B1: elected over ingestion-persisted twin_block_boundaries instead
# of the runtime paragraph split — real document structure, same contract.
ANCHOR_METHOD_STRUCTURAL = "structural_block"

# Unprocessed tail kept between feed() calls: at most one in-flight token
# (or citation marker) plus slack. Beyond this the run cannot be a valid
# token, so it is dropped and the current sentence flagged as overflowed.
_TAIL_CAP = MAX_TOKEN_CHARS + 16

_PARAGRAPH_BREAK = re.compile(r"(?:\r?\n)(?:[ \t]*\r?\n)+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_CITE_RE = re.compile(r"\[(\d{1,4})\]")
_SENTENCE_BOUNDARY = re.compile(r"[.!?\n]")
# A "safe cut" ends at any char that can never continue a token or a
# citation marker; everything after the last such char may still grow.
_DELIMITER_RE = re.compile(r"[^\w\[\]]|\]")
_TRIM_CHARS = " \t\r\n"


@dataclass(frozen=True)
class Paragraph:
    """Half-open ``[start, end)`` character span in the original content."""

    start: int
    end: int


@dataclass(frozen=True)
class CitationEvidence:
    """Bounded lexical evidence for one ``[n]`` citation.

    ``incomplete`` is True when any §5.1 cap was hit while collecting: the
    scorer must then omit the anchor rather than score a truncation.
    """

    tokens: frozenset[str]
    incomplete: bool = False


def split_paragraphs(content: str) -> list[Paragraph]:
    """Split ``content`` on blank lines into non-empty paragraph spans.

    Offsets index the ORIGINAL string (no normalization pass), each span is
    trimmed of surrounding whitespace, and empty paragraphs are dropped —
    so every returned span satisfies ``0 <= start < end <= len(content)``
    and ``content[start:end].strip() == content[start:end]``.
    """
    if not content:
        return []
    paragraphs: list[Paragraph] = []
    pos = 0
    for separator in _PARAGRAPH_BREAK.finditer(content):
        _append_trimmed(paragraphs, content, pos, separator.start())
        pos = separator.end()
    _append_trimmed(paragraphs, content, pos, len(content))
    return paragraphs


def _append_trimmed(out: list[Paragraph], content: str, start: int, end: int) -> None:
    while start < end and content[start] in _TRIM_CHARS:
        start += 1
    while end > start and content[end - 1] in _TRIM_CHARS:
        end -= 1
    if end > start:
        out.append(Paragraph(start, end))


def _iter_normalized_tokens(text: str) -> Iterable[str]:
    """Lazy form of the shared token pipeline — see :func:`_normalized_tokens`."""
    return map(str.casefold, _TOKEN_RE.findall(text))


def _normalized_tokens(text: str) -> list[str]:
    """Shared token pipeline: extract word runs first, casefold after.

    Order matters — casefolding can change codepoint counts (e.g. ``İ``),
    so both the evidence side and the paragraph side must extract tokens
    from the raw text and only then casefold each token.
    """
    return list(_iter_normalized_tokens(text))


class CitationEvidenceCollector:
    """Incremental, bounded collector of per-citation lexical evidence.

    Feed it the SAME safe text the operator reads (the marker-stripper
    output on the streaming path, ``clean_answer`` on the non-stream path)
    and call :meth:`finalize` once. Feeding fragment-by-fragment or as one
    string yields the same observable result: normal text is cut only at
    delimiter characters, and a token run too long to be valid is flagged
    as overflow on both shapes (which suppresses the anchor either way).
    """

    def __init__(self) -> None:
        self._tail = ""
        self._sentence_tokens: set[str] = set()
        self._sentence_cites: set[int] = set()
        self._sentence_overflow = False
        self._evidence: dict[int, set[str]] = {}
        self._incomplete: set[int] = set()
        self._total_tokens = 0
        self._finalized = False

    def feed(self, text: str) -> None:
        if self._finalized or not text:
            return
        combined = self._tail + text
        cut = self._last_delimiter_end(combined)
        processable, tail = combined[:cut], combined[cut:]
        if len(tail) > _TAIL_CAP:
            # Unbroken run longer than any valid token: drop it and taint
            # the current sentence so its citations stay anchor-less.
            self._sentence_overflow = True
            tail = ""
        self._tail = tail
        if processable:
            self._process(processable)

    def finalize(self) -> dict[int, CitationEvidence]:
        """Flush pending state and return the evidence per citation id."""
        if not self._finalized:
            if self._tail:
                self._process(self._tail)
                self._tail = ""
            self._flush_sentence()
            self._finalized = True
        return {
            n: CitationEvidence(frozenset(tokens), n in self._incomplete)
            for n, tokens in self._evidence.items()
        }

    @staticmethod
    def _last_delimiter_end(text: str) -> int:
        last = None
        for match in _DELIMITER_RE.finditer(text):
            last = match
        return last.end() if last is not None else 0

    def _process(self, text: str) -> None:
        pos = 0
        for boundary in _SENTENCE_BOUNDARY.finditer(text):
            self._ingest_segment(text[pos : boundary.start()])
            self._flush_sentence()
            pos = boundary.end()
        self._ingest_segment(text[pos:])

    def _ingest_segment(self, segment: str) -> None:
        if not segment:
            return
        cursor = 0
        plain_parts: list[str] = []
        for marker in _CITE_RE.finditer(segment):
            self._sentence_cites.add(int(marker.group(1)))
            plain_parts.append(segment[cursor : marker.start()])
            cursor = marker.end()
        plain_parts.append(segment[cursor:])
        for token in _normalized_tokens(" ".join(plain_parts)):
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        if len(token) > MAX_TOKEN_CHARS:
            self._sentence_overflow = True
            return
        if token in self._sentence_tokens:
            return
        if len(self._sentence_tokens) >= MAX_TOKENS_PER_SENTENCE:
            self._sentence_overflow = True
            return
        self._sentence_tokens.add(token)

    def _flush_sentence(self) -> None:
        tokens = self._sentence_tokens
        cites = self._sentence_cites
        overflow = self._sentence_overflow
        self._sentence_tokens = set()
        self._sentence_cites = set()
        self._sentence_overflow = False
        if not cites:
            return
        for n in cites:
            evidence = self._evidence.setdefault(n, set())
            if overflow:
                self._incomplete.add(n)
            for token in tokens:
                if token in evidence:
                    continue
                if self._total_tokens >= MAX_EVIDENCE_TOKENS_TOTAL:
                    self._incomplete.add(n)
                    break
                evidence.add(token)
                self._total_tokens += 1


def collect_citation_evidence(answer: str) -> dict[int, CitationEvidence]:
    """Non-stream convenience: collect evidence from a materialised answer."""
    collector = CitationEvidenceCollector()
    collector.feed(answer)
    return collector.finalize()


def compute_best_anchor(
    candidates: list[tuple[str, str]],
    evidence: CitationEvidence,
) -> tuple[str, dict[str, Any]] | None:
    """Elect ONE anchor across every chunk behind a reference, or None.

    LightRAG mints one ``reference_id`` per *file*, so several chunks of the
    same document legitimately share ``[n]`` (PR #418 review, finding 1).
    Scoring only the first chunk can publish a plausible-but-wrong anchor:
    the election therefore runs over the paragraphs of ALL candidate chunks,
    and the §5.1 top-2 penalty applies globally — a runner-up paragraph in
    ANOTHER chunk lowers the confidence exactly like one in the same chunk
    (duplicated content across chunks honestly suppresses the anchor).

    ``candidates`` is ``[(chunk_id, content), …]``. Returns the winning
    ``(chunk_id, anchor)`` pair so the caller can publish both atomically —
    an anchor must never be attached to a chunk it was not scored in.

    Offsets in the returned anchor are Unicode CODE POINTS in the winning
    chunk's content (Python string indices) — JavaScript consumers must
    slice on ``Array.from(content)``, not UTF-16 ``String.prototype.slice``.
    """
    if evidence.incomplete or not evidence.tokens:
        return None
    segmented = [
        (
            chunk_id,
            content,
            [(p.start, p.end) for p in split_paragraphs(content)],
        )
        for chunk_id, content in candidates
        if content
    ]
    return _elect_best_segment(segmented, evidence, ANCHOR_METHOD_LEXICAL)


def compute_best_structural_anchor(
    candidates: list[tuple[str, str, list[dict[str, Any]]]],
    evidence: CitationEvidence,
) -> tuple[str, dict[str, Any]] | None:
    """Elect over ingestion-persisted block boundaries (phase B1).

    Same election, penalty and floor as :func:`compute_best_anchor`, but the
    segments are the ``twin_block_boundaries`` the 1.5.x preconverted-parse
    seam persisted at ingestion — real document structure instead of a
    runtime paragraph heuristic. ``candidates`` is
    ``[(chunk_id, content, boundaries), …]``. A chunk whose boundary list is
    malformed (non-int offsets, out of range, inverted) contributes NO
    segments — plan §6 point 3: a malformed sidecar yields no structural
    anchor, never a fabricated one. Callers fall back to the lexical
    election when this returns None.
    """
    if evidence.incomplete or not evidence.tokens:
        return None
    segmented: list[tuple[str, str, list[tuple[int, int]]]] = []
    for chunk_id, content, boundaries in candidates:
        if not content or not isinstance(boundaries, list):
            continue
        segments: list[tuple[int, int]] = []
        for boundary in boundaries:
            if not isinstance(boundary, dict):
                segments = []
                break
            start = boundary.get("start")
            end = boundary.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not (0 <= start < end <= len(content))
            ):
                segments = []
                break
            segments.append((start, end))
        if segments:
            segmented.append((chunk_id, content, segments))
    if not segmented:
        return None
    return _elect_best_segment(segmented, evidence, ANCHOR_METHOD_STRUCTURAL)


def _elect_best_segment(
    segmented: list[tuple[str, str, list[tuple[int, int]]]],
    evidence: CitationEvidence,
    method: str,
) -> tuple[str, dict[str, Any]] | None:
    """Shared election core: containment scoring + §5.1 top-2 penalty.

    One implementation for both methods so the lexical and structural
    variants can never drift on scoring, tie-breaking or the confidence
    floor.
    """
    best: tuple[int, str, tuple[int, int], int] | None = None
    best_score = 0.0
    second_score = 0.0
    for chunk_id, content, segments in segmented:
        for segment_idx, (start, end) in enumerate(segments):
            score = _containment(evidence.tokens, content[start:end])
            # Deterministic winner: strict > keeps the first segment in
            # envelope order when scores tie (§5.1 contract).
            if best is None or score > best_score:
                if best is not None:
                    second_score = max(second_score, best_score)
                best_score = score
                best = (segment_idx, chunk_id, (start, end), len(segments))
            else:
                second_score = max(second_score, score)
    if best is None or best_score <= 0.0:
        return None
    segment_idx, chunk_id, span, segment_count = best
    confidence = best_score - 0.5 * second_score
    if confidence < MIN_ANCHOR_CONFIDENCE:
        return None
    return chunk_id, {
        "start": span[0],
        "end": span[1],
        "paragraph_idx": segment_idx,
        "paragraph_count": segment_count,
        "confidence": round(confidence, 4),
        "method": method,
    }


def compute_anchor(content: str, evidence: CitationEvidence) -> dict[str, Any] | None:
    """Single-chunk convenience over :func:`compute_best_anchor`.

    Score = containment of the citation's evidence tokens in each
    paragraph's token set. One anchor only (§5.1 contract decision):
    ``confidence = best - 0.5 * second`` so evidence dispersed over two
    close paragraphs is published with a measurably lower confidence — or
    not at all, below :data:`MIN_ANCHOR_CONFIDENCE`.
    """
    if not content:
        return None
    result = compute_best_anchor([("", content)], evidence)
    return result[1] if result is not None else None


def _containment(evidence_tokens: frozenset[str], paragraph_text: str) -> float:
    if not evidence_tokens:
        return 0.0
    # Keep the allocation bounded by the evidence set instead of building a
    # set containing every token in a potentially long source paragraph.
    matches = evidence_tokens.intersection(_iter_normalized_tokens(paragraph_text))
    return len(matches) / len(evidence_tokens)


__all__ = [
    "ANCHOR_METHOD_LEXICAL",
    "ANCHOR_METHOD_STRUCTURAL",
    "MAX_EVIDENCE_TOKENS_TOTAL",
    "MAX_TOKEN_CHARS",
    "MAX_TOKENS_PER_SENTENCE",
    "MIN_ANCHOR_CONFIDENCE",
    "CitationEvidence",
    "CitationEvidenceCollector",
    "Paragraph",
    "collect_citation_evidence",
    "compute_anchor",
    "compute_best_anchor",
    "compute_best_structural_anchor",
    "split_paragraphs",
]
