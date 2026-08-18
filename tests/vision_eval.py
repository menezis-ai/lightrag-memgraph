"""Scored evaluation harness for the Vision tier (Knowledge-Bot methodology).

The reference Knowledge-Bot project did not stop at "the pipeline returned
something". It ran a **golden set** with declared ground truth, scored every
answer on a rubric, repeated each item several times (its ``Test 1 / Test 2 /
Test 3`` blocks), marked pass/fail against a threshold, and aggregated accuracy
per topic so the number could be tracked across model and prompt versions.

We had copied its *pipeline* (RapidOCR pre-filter → vision classification →
drop ``invalid,logo,signature``) but not its *evaluation*. The live gate
asserted a handful of substrings on a corpus of two, one sample each — so a
model answering correctly four times out of five flipped CI red at random, and
nothing measured that.

This module supplies the missing half:

* a rubric turning one outcome into a 0..4 score with an explicit per-check
  breakdown (a failure says *which* property broke, not just "assert False");
* repetition, so non-determinism is measured rather than sampled once;
* aggregation into global accuracy, per-topic accuracy and per-case variance;
* a JSON report artifact, so the number is comparable across runs the way the
  reference tracked its accuracy table over time.

The scoring functions are pure and are covered by the *free* offline suite
against a stubbed endpoint — the live gate only adds the real network.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field

from tests.vision_corpus import VisionCase, compact

#: A case scores 0..4; >= this is a pass. Mirrors the reference's "score >= 3
#: out of 5 counts as correct" convention.
PASS_THRESHOLD = 3
MAX_SCORE = 4

OCR_SECTION_HEADING = "## Extracted text (OCR)"


def repeats() -> int:
    """Samples per case. Each one costs a model call on ingest cases."""
    raw = os.environ.get("TWIN_VISION_EVAL_REPEATS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 2
    return max(1, min(value, 5))


def min_accuracy() -> float:
    """Global pass rate the gate requires."""
    raw = os.environ.get("TWIN_VISION_EVAL_MIN_ACCURACY", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.75
    return value if 0.0 < value <= 1.0 else 0.75


@dataclass
class CaseScore:
    """One scored observation of one case."""

    key: str
    topic: str
    repeat: int
    score: int
    checks: dict[str, bool] = field(default_factory=dict)
    classification: str | None = None
    reason: str = ""
    model_calls: int = 0

    @property
    def passed(self) -> bool:
        return self.score >= PASS_THRESHOLD

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]


def split_model_content(markdown: str | None) -> str:
    """The model's own prose, excluding the OCR block we appended ourselves.

    Asserting an anchor against the whole markdown would be circular: our own
    RapidOCR transcript is concatenated into it, so a model that returned
    nothing useful would still "contain" the expected strings.
    """
    if not markdown:
        return ""
    return markdown.partition(OCR_SECTION_HEADING)[0]


def score_case(
    case: VisionCase, outcome, *, repeat: int, model_calls: int
) -> CaseScore:
    """Apply the rubric to one outcome."""
    if case.expect_ingest:
        return _score_ingest(case, outcome, repeat=repeat, model_calls=model_calls)
    return _score_refusal(case, outcome, repeat=repeat, model_calls=model_calls)


def _score_ingest(case: VisionCase, outcome, *, repeat: int, model_calls: int):
    markdown = getattr(outcome, "markdown", None)
    classification = getattr(outcome, "classification", None) or ""
    content = split_model_content(markdown)
    compact_content = compact(content)

    checks = {
        "ingested": bool(markdown) and getattr(outcome, "reason", "") == "ok",
        "classification_expected": (
            classification.strip().lower() in case.expected_classes
            if case.expected_classes
            else bool(classification.strip())
        ),
        "semantic_anchors_present": all(
            compact(anchor) in compact_content for anchor in case.semantic_anchors
        ),
        "no_hallucination": not any(
            compact(bad) and compact(bad) in compact_content for bad in case.forbidden
        ),
    }
    # A refusal cannot earn the downstream points: without markdown the other
    # checks are vacuously true and would inflate the score to 3 (a pass).
    if not checks["ingested"]:
        checks["semantic_anchors_present"] = False
        checks["no_hallucination"] = False
        checks["classification_expected"] = False

    return CaseScore(
        key=case.key,
        topic=case.topic,
        repeat=repeat,
        score=sum(1 for ok in checks.values() if ok),
        checks=checks,
        classification=classification or None,
        reason=getattr(outcome, "reason", ""),
        model_calls=model_calls,
    )


def _score_refusal(case: VisionCase, outcome, *, repeat: int, model_calls: int):
    markdown = getattr(outcome, "markdown", None)
    reason = getattr(outcome, "reason", "") or ""
    refused = markdown is None
    checks = {
        "refused": refused,
        # Refusing is worth two points: it is the property that matters.
        "refused_confirmed": refused,
        "expected_reason": bool(case.expect_reason) and case.expect_reason in reason,
        # Noise must never reach the paid endpoint. This is the pre-filter's
        # whole economic purpose, so it is worth a rubric point of its own.
        "no_model_call_spent": model_calls == 0 if case.free_refusal else True,
    }
    return CaseScore(
        key=case.key,
        topic=case.topic,
        repeat=repeat,
        score=sum(1 for ok in checks.values() if ok),
        checks=checks,
        classification=getattr(outcome, "classification", None),
        reason=reason,
        model_calls=model_calls,
    )


def aggregate(scores: list[CaseScore]) -> dict:
    """Accuracy, per-topic and per-case breakdown, plus the failure list."""
    if not scores:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "mean_score": 0.0,
            "by_case": {},
            "by_topic": {},
            "failures": [],
            "model_calls": 0,
        }

    by_case: dict[str, dict] = {}
    for score in scores:
        entry = by_case.setdefault(
            score.key, {"topic": score.topic, "scores": [], "failed_checks": []}
        )
        entry["scores"].append(score.score)
        entry["failed_checks"].extend(score.failed_checks)

    for key, entry in by_case.items():
        values = entry["scores"]
        entry["mean_score"] = round(statistics.fmean(values), 2)
        entry["min_score"] = min(values)
        # Non-determinism, the thing a single sample per case cannot see.
        entry["spread"] = max(values) - min(values)
        entry["pass_rate"] = round(
            sum(1 for v in values if v >= PASS_THRESHOLD) / len(values), 3
        )
        entry["failed_checks"] = sorted(set(entry["failed_checks"]))

    by_topic: dict[str, dict] = {}
    for score in scores:
        entry = by_topic.setdefault(score.topic, {"samples": 0, "passed": 0})
        entry["samples"] += 1
        entry["passed"] += int(score.passed)
    for entry in by_topic.values():
        entry["accuracy"] = round(entry["passed"] / entry["samples"], 3)

    return {
        "samples": len(scores),
        "accuracy": round(sum(1 for s in scores if s.passed) / len(scores), 3),
        "mean_score": round(statistics.fmean(s.score for s in scores), 2),
        "max_score": MAX_SCORE,
        "pass_threshold": PASS_THRESHOLD,
        "by_case": by_case,
        "by_topic": by_topic,
        "model_calls": sum(s.model_calls for s in scores),
        "failures": [
            {
                "case": s.key,
                "repeat": s.repeat,
                "score": s.score,
                "failed_checks": s.failed_checks,
                "classification": s.classification,
                "reason": s.reason[:300],
            }
            for s in scores
            if not s.passed
        ],
    }


def format_summary(report: dict) -> str:
    """Human-readable block for the CI log."""
    lines = [
        f"accuracy {report['accuracy']:.0%} over {report['samples']} samples "
        f"(mean score {report['mean_score']}/{report['max_score']}, "
        f"threshold {report['pass_threshold']}, "
        f"{report['model_calls']} model call(s))",
        f"{'case':26} {'mean':>5} {'min':>4} {'spread':>7} {'pass':>6}  failed checks",
    ]
    for key, entry in sorted(report["by_case"].items()):
        lines.append(
            f"{key:26} {entry['mean_score']:5.2f} {entry['min_score']:4d} "
            f"{entry['spread']:7d} {entry['pass_rate']:6.0%}  "
            f"{','.join(entry['failed_checks']) or '-'}"
        )
    for topic, entry in sorted(report["by_topic"].items()):
        lines.append(f"topic {topic:20} accuracy {entry['accuracy']:.0%}")
    return "\n".join(lines)
