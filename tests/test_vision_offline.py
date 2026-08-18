"""Free OCR/Vision acceptance gate — real pixels, real RapidOCR, no paid call.

This suite is the middle tier the project was missing. Before it there were
only two extremes:

* ``tests/test_vision.py`` — fast, but every image is ``b"\\x89PNG" + b"\\x00"
  * 64`` and ``_ocr_text_sync`` is monkeypatched in **every** test, so no
  decoder and no OCR model ever runs;
* ``tests/test_vision_live.py`` — real everything, but **paid**, and gated
  behind ``needs: [unit-tests, integration-tests]``, so a broken ONNX model,
  a Pillow decoder regression or a colour-mode crash surfaces only at the very
  end of the pipeline, at a cost.

Here the *only* stubbed component is the HTTP transport to the vision endpoint.
Real bytes are rendered, really decoded, really OCR'd, really base64-encoded
into the request the endpoint receives, and the response really flows back
through parsing, validation, drop policy and markdown composition. That is the
composed path — the wiring is where these tiers break, and it was never
executed for free.

Ground truth and calibrated recall floors live in ``tests/vision_corpus.py``.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import os
import re
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

REQUIRED_MODULES = ("rapidocr_onnxruntime", "PIL", "numpy", "pypdfium2")
_MISSING = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]

# The dedicated CI job sets this. Without it, a runner that simply lacks the
# [vision] extra would skip every test below and report a green gate — the
# exact "verification that verified nothing" failure mode this suite exists to
# prevent. Under strict mode a missing dependency is a hard collection error.
_STRICT = os.environ.get("TWIN_VISION_OFFLINE_STRICT") == "1"
if _MISSING and _STRICT:
    raise RuntimeError(
        "TWIN_VISION_OFFLINE_STRICT=1 but the offline OCR/Vision gate cannot "
        f"run: missing {', '.join(_MISSING)}. Install '.[vision,procedure,test]'."
    )

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"offline vision gate needs {', '.join(_MISSING)} (extra: [vision])",
)

if not _MISSING:  # keep collection clean on a bare [test] install
    from tests.vision_corpus import (
        CORPUS,
        MALFORMED_BUILDERS,
        anchor_recall,
        build_visual_pdf,
        compact,
        materialise,
    )
    from twindb_lightrag_memgraph import _pdf_vision, _vision

    TEXT_CASES = [c for c in CORPUS if c.ocr_anchors]
    REFUSAL_CASES = [c for c in CORPUS if c.free_refusal]
    INGEST_CASES = [c for c in CORPUS if c.expect_ingest]


# ---------------------------------------------------------------------------
# Stub vision endpoint — the ONLY mocked component
# ---------------------------------------------------------------------------


class StubVisionEndpoint:
    """Records every request and replays a scripted reply.

    Stands exactly where the network does: everything above it
    (``vision_chat_sync`` → base64 data URI → JSON parse → validation → drop
    policy → markdown) is the production code path.
    """

    def __init__(self, responder=None):
        self.calls: list[dict] = []
        self._responder = responder or (
            lambda endpoint, request: json.dumps(
                {
                    "image_classification": "document",
                    "content": "A business document with a totals table.",
                }
            )
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kwargs):
        request = {"model": model, "messages": messages, **kwargs}
        self.calls.append(request)
        reply = self._responder(self, request)
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    # -- request introspection ------------------------------------------

    @staticmethod
    def image_data_url(request: dict) -> str:
        for message in request["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    return part["image_url"]["url"]
        raise AssertionError("request carried no image_url part")

    @classmethod
    def image_bytes(cls, request: dict) -> tuple[str, bytes]:
        url = cls.image_data_url(request)
        match = re.fullmatch(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
        assert match, f"malformed data URI: {url[:60]!r}"
        return match.group(1), base64.b64decode(match.group(2))


@pytest.fixture
def endpoint(monkeypatch):
    """Configure the vision tier and route every call to the stub."""
    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("TWIN_VISION_MODEL", "stub-vision-model")
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "20")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", "invalid,logo,signature")
    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "60")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    _vision.reset_caches()
    _pdf_vision.reset_caches()

    stub = StubVisionEndpoint()
    monkeypatch.setattr(_vision, "_get_client", lambda: stub)
    yield stub
    _vision.reset_caches()
    _pdf_vision.reset_caches()


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> Path:
    """Render the corpus once; OCR is the slow part, rendering need not be."""
    return tmp_path_factory.mktemp("vision-corpus")


# ---------------------------------------------------------------------------
# 1. Real OCR quality, scored against calibrated floors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", TEXT_CASES if not _MISSING else [], ids=lambda c: c.key
)
def test_real_ocr_meets_calibrated_recall_floor(case, corpus_dir, record_property):
    """Real RapidOCR must transcribe the declared anchors of each document."""
    path = materialise(case, corpus_dir)
    text = _vision._ocr_text_sync(path)

    assert text is not None, f"{case.key}: OCR returned None (engine or decoder failed)"
    recall, missing = anchor_recall(text, case.ocr_anchors)
    record_property(f"ocr_recall_{case.key}", recall)
    assert recall >= case.min_ocr_recall, (
        f"{case.key}: OCR recall {recall:.2f} below the calibrated floor "
        f"{case.min_ocr_recall:.2f}; missing anchors {missing}; "
        f"transcript={text[:300]!r}"
    )


def test_cmyk_jpeg_is_not_silently_dropped_by_the_prefilter(corpus_dir):
    """Regression guard: RapidOCR's file loader yields nothing for CMYK JPEG.

    That empty result used to be indistinguishable from "the image has no
    text", so the pre-filter refused a fully legible purchase order and it
    never reached the vision model. ``_ocr_text_sync`` now re-decodes through
    Pillow before concluding. Measured 0.00 recall before the fix, 1.00 after.
    """
    case = next(c for c in CORPUS if c.key == "cmyk_purchase_order")
    path = materialise(case, corpus_dir)

    from PIL import Image

    with Image.open(path) as image:
        assert image.mode == "CMYK", "fixture must stay a genuine CMYK JPEG"

    text = _vision._ocr_text_sync(path)
    assert text is not None
    assert len(text.strip()) >= _vision.min_ocr_chars(), (
        "CMYK document fell below the pre-filter threshold — the re-decode "
        f"retry regressed; transcript={text!r}"
    )
    assert "BC20260442" in compact(text)


def test_corpus_ocr_recall_aggregate(corpus_dir, tmp_path):
    """Corpus-wide accuracy, reported as a number rather than asserted blind.

    Per-case floors catch one document collapsing; this catches a broad
    degradation (an onnxruntime bump, a model-file swap) that keeps every
    individual case just above its own floor.
    """
    scores = {}
    for case in TEXT_CASES:
        path = materialise(case, corpus_dir)
        recall, missing = anchor_recall(_vision._ocr_text_sync(path), case.ocr_anchors)
        scores[case.key] = {
            "recall": round(recall, 3),
            "floor": case.min_ocr_recall,
            "missing": missing,
            "topic": case.topic,
        }

    mean = statistics.fmean(entry["recall"] for entry in scores.values())
    report = {"mean_recall": round(mean, 3), "cases": scores}
    (tmp_path / "ocr-recall-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert mean >= 0.80, f"corpus OCR recall collapsed to {mean:.2f}: {report}"


# ---------------------------------------------------------------------------
# 2. The pre-filter's economic contract, on real pixels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", REFUSAL_CASES if not _MISSING else [], ids=lambda c: c.key
)
async def test_noise_is_refused_without_spending_a_model_call(
    case, corpus_dir, endpoint
):
    """A logo, a signature, a blank scan: refused by OCR alone, zero calls.

    This is the pre-filter's entire economic justification, and until now it
    was only ever asserted against a monkeypatched OCR returning a hardcoded
    short string — never against real pixels through the real engine.
    """
    path = materialise(case, corpus_dir)
    outcome = await _vision.aprocess_image(path)

    assert outcome.markdown is None, f"{case.key} should not be ingested"
    assert case.expect_reason in outcome.reason, outcome.reason
    assert endpoint.calls == [], (
        f"{case.key} reached the vision endpoint — the pre-filter no longer "
        f"protects the paid path ({len(endpoint.calls)} call(s))"
    )


# ---------------------------------------------------------------------------
# 3. Composed pipeline on real bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", INGEST_CASES if not _MISSING else [], ids=lambda c: c.key
)
async def test_document_cases_compose_markdown_end_to_end(case, corpus_dir, endpoint):
    """Real image → OCR → endpoint → parse → policy → markdown, unmocked between."""
    path = materialise(case, corpus_dir)
    outcome = await _vision.aprocess_image(path)

    assert outcome.markdown is not None, f"{case.key} refused: {outcome.reason}"
    assert outcome.reason == "ok"
    assert len(endpoint.calls) == 1
    assert outcome.markdown.startswith(f"# {case.filename}")
    assert "_Image type: document_" in outcome.markdown
    assert "A business document with a totals table." in outcome.markdown

    if case.ocr_anchors:
        semantic, separator, appended = outcome.markdown.partition(
            "## Extracted text (OCR)"
        )
        assert separator, "OCR section missing from composed markdown"
        recall, _ = anchor_recall(appended, case.ocr_anchors)
        assert recall >= case.min_ocr_recall


@pytest.mark.parametrize(
    ("case_key", "expected_mime"),
    [
        ("invoice_table", "image/png"),
        ("cmyk_purchase_order", "image/jpeg"),
        ("noisy_fax", "image/jpeg"),
    ],
)
async def test_endpoint_receives_the_real_image_bytes(
    case_key, expected_mime, corpus_dir, endpoint
):
    """The data URI must carry the actual file, with the right MIME type.

    Two proven halves wired by an encoder nobody ran is not a proven pipeline:
    this decodes what the endpoint actually received and re-opens it.
    """
    from PIL import Image

    from tests.vision_corpus import CORPUS_BY_KEY

    case = CORPUS_BY_KEY[case_key]
    path = materialise(case, corpus_dir)
    await _vision.aprocess_image(path)

    mime, payload = StubVisionEndpoint.image_bytes(endpoint.calls[0])
    assert mime == expected_mime
    assert payload == path.read_bytes(), "transmitted bytes differ from the file"
    with Image.open(io.BytesIO(payload)) as received:
        assert received.size == Image.open(path).size

    assert endpoint.calls[0]["temperature"] == 0
    assert endpoint.calls[0]["response_format"] == {"type": "json_object"}


async def test_drop_class_is_enforced_on_a_real_document(corpus_dir, monkeypatch):
    """A model calling a legible page a logo must still be refused."""
    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("TWIN_VISION_MODEL", "stub-vision-model")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", "invalid,logo,signature")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    _vision.reset_caches()

    stub = StubVisionEndpoint(
        lambda endpoint, request: json.dumps(
            {"image_classification": "Logo", "content": "A corporate mark."}
        )
    )
    monkeypatch.setattr(_vision, "_get_client", lambda: stub)

    case = next(c for c in CORPUS if c.key == "invoice_table")
    outcome = await _vision.aprocess_image(materialise(case, corpus_dir))

    assert outcome.markdown is None
    assert "image-dropped" in outcome.reason
    assert outcome.classification == "Logo"
    _vision.reset_caches()


# ---------------------------------------------------------------------------
# 4. Anomaly injection — malformed bytes and a hostile endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "filename", "build"),
    MALFORMED_BUILDERS if not _MISSING else [],
    ids=[entry[0] for entry in (MALFORMED_BUILDERS if not _MISSING else [])],
)
async def test_malformed_inputs_degrade_without_raising(
    key, filename, build, tmp_path, endpoint
):
    """Truncated, empty, bogus and bomb inputs must never crash the seam.

    ``_vision`` contracts that it never raises into ingestion. These are real
    malformed bytes, not a short buffer with a PNG magic prefix.

    FOLLOW-UP (not asserted here so the test does not enshrine it): an
    undecodable file currently still reaches the endpoint, because OCR failure
    returns ``None`` and correctly bypasses the pre-filter — which means a
    0-byte upload spends one paid call before failing.
    """
    path = build(tmp_path / filename)
    outcome = await _vision.aprocess_image(path)

    assert isinstance(outcome, _vision.VisionOutcome)
    if outcome.markdown is None:
        assert outcome.reason and outcome.reason != "ok"


async def test_decompression_bomb_is_rejected_before_pixels_are_materialised(
    tmp_path, endpoint, caplog
):
    """The OCR re-decode must consult the header, not allocate 100 MP of RGB."""
    from tests.vision_corpus import build_decompression_bomb

    path = build_decompression_bomb(tmp_path / "bomb.png")
    with caplog.at_level("WARNING"):
        text = _vision._ocr_text_sync(path)

    assert text is not None
    assert "skipping OCR re-decode" in caplog.text


@pytest.mark.parametrize(
    ("label", "reply", "expected_reason"),
    [
        ("empty_reply", "", "vision-llm-error"),
        (
            "truncated_json",
            '{"image_classification": "table", "content": "abc',
            "vision-llm-error",
        ),
        ("prose_not_json", "I am unable to process this image.", "vision-llm-error"),
        ("empty_object", "{}", "vision-llm-error"),
        (
            "wrong_types",
            '{"image_classification": 7, "content": ["a"]}',
            "vision-llm-error",
        ),
        (
            "null_content",
            '{"image_classification": "table", "content": null}',
            "vision-llm-error",
        ),
        (
            "empty_content",
            '{"image_classification": "table", "content": "   "}',
            "image-dropped",
        ),
    ],
)
async def test_hostile_endpoint_replies_are_contained(
    label, reply, expected_reason, corpus_dir, monkeypatch
):
    """A degraded model response degrades the document, never the process."""
    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("TWIN_VISION_MODEL", "stub-vision-model")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    _vision.reset_caches()
    monkeypatch.setattr(
        _vision, "_get_client", lambda: StubVisionEndpoint(lambda e, r: reply)
    )

    case = next(c for c in CORPUS if c.key == "invoice_table")
    outcome = await _vision.aprocess_image(materialise(case, corpus_dir))

    assert outcome.markdown is None, f"{label} should not have produced markdown"
    assert expected_reason in outcome.reason, outcome.reason
    _vision.reset_caches()


async def test_endpoint_exception_and_hang_are_both_bounded(corpus_dir, monkeypatch):
    """Transport failure and a hung endpoint both end in a reasoned refusal."""
    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("TWIN_VISION_MODEL", "stub-vision-model")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    case = next(c for c in CORPUS if c.key == "invoice_table")
    path = materialise(case, corpus_dir)

    _vision.reset_caches()
    monkeypatch.setattr(
        _vision,
        "_get_client",
        lambda: StubVisionEndpoint(lambda e, r: ConnectionError("endpoint down")),
    )
    failed = await _vision.aprocess_image(path)
    assert failed.markdown is None
    assert "vision-llm-error" in failed.reason
    assert "endpoint down" in failed.reason

    _vision.reset_caches()
    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "0.2")

    def hang(endpoint, request):
        import time

        time.sleep(5)
        return "{}"

    monkeypatch.setattr(_vision, "_get_client", lambda: StubVisionEndpoint(hang))
    hung = await asyncio.wait_for(_vision.aprocess_image(path), timeout=10)
    assert hung.markdown is None
    assert "vision-timeout" in hung.reason
    _vision.reset_caches()


# ---------------------------------------------------------------------------
# 5. Generic PDF visual tier — real PDFium render, real OCR
# ---------------------------------------------------------------------------


async def test_pdf_visual_tier_renders_ocrs_and_appends_with_provenance(
    tmp_path, endpoint, monkeypatch
):
    """A PDF embedding a real raster image: render → OCR → describe → append."""
    monkeypatch.setenv("TWIN_PDF_VISION", "on")
    _pdf_vision.reset_caches()
    assert _pdf_vision.is_enabled() is True

    pdf = build_visual_pdf(tmp_path / "annexe-comptable.pdf")
    base = "# annexe-comptable.pdf\n\nAnnexe comptable - piece justificative."

    outcome = await _pdf_vision.aprocess_pdf(pdf, base_markdown=base)

    assert outcome.markdown is not None, outcome.reason
    assert outcome.candidates >= 1, f"no visual discovered: {outcome.reason}"
    assert outcome.accepted >= 1, f"visual not accepted: {outcome.reason}"
    assert endpoint.calls, "the PDF visual never reached the vision endpoint"
    assert "A business document with a totals table." in outcome.markdown
    assert base.strip() in outcome.markdown

    mime, payload = StubVisionEndpoint.image_bytes(endpoint.calls[0])
    assert mime == "image/png"
    assert payload.startswith(b"\x89PNG\r\n\x1a\n"), "PDFium did not render a PNG"

    ocr_of_render = _vision.ocr_png_bytes_sync(payload, label="pdf visual")
    assert ocr_of_render is not None
    assert "FACTUREDATACENTER" in compact(ocr_of_render), (
        "the rendered region does not contain the embedded invoice — the PDF "
        f"raster path regressed; transcript={ocr_of_render[:200]!r}"
    )


# ---------------------------------------------------------------------------
# 6. The scoring rubric itself
# ---------------------------------------------------------------------------
#
# The paid gate decides pass/fail from tests/vision_eval.py. A rubric nobody
# exercised is an unverified verifier — so it is covered here, for free, and
# the live job only adds the network.


def test_rubric_awards_full_marks_to_a_correct_ingest():
    from tests.vision_eval import MAX_SCORE, score_case
    from tests.vision_corpus import CORPUS_BY_KEY

    case = CORPUS_BY_KEY["invoice_table"]
    outcome = _vision.VisionOutcome(
        markdown=(
            "# invoice-table.png\n\n_Image type: table_\n\n"
            "Facture totalisant 12 480 EUR dont 9 600 EUR d'hebergement.\n\n"
            "## Extracted text (OCR)\n\nFACTURE DATACENTER"
        ),
        reason="ok",
        classification="table",
    )
    score = score_case(case, outcome, repeat=1, model_calls=1)

    assert score.score == MAX_SCORE
    assert score.passed
    assert score.failed_checks == []


def test_rubric_refuses_to_reward_a_refused_ingest_case():
    """A refusal must score 0, not collect points from vacuous checks.

    ``all(... for _ in ())`` is True and ``not any(...)`` is True, so a
    markdown-less outcome would otherwise bank two free points and land on the
    pass threshold. This is the rubric's own failure mode.
    """
    from tests.vision_eval import score_case
    from tests.vision_corpus import CORPUS_BY_KEY

    case = CORPUS_BY_KEY["architecture_diagram"]
    outcome = _vision.VisionOutcome(markdown=None, reason="vision-llm-error: boom")
    score = score_case(case, outcome, repeat=1, model_calls=1)

    assert score.score == 0
    assert not score.passed


def test_rubric_penalises_an_unexpected_classification_by_one_point():
    from tests.vision_eval import score_case
    from tests.vision_corpus import CORPUS_BY_KEY

    case = CORPUS_BY_KEY["invoice_table"]
    markdown = (
        "# invoice-table.png\n\n_Image type: photo_\n\n"
        "Facture de 12 480 EUR, ligne hebergement 9 600 EUR.\n\n"
        "## Extracted text (OCR)\n\nFACTURE"
    )
    score = score_case(
        case,
        _vision.VisionOutcome(markdown=markdown, reason="ok", classification="photo"),
        repeat=1,
        model_calls=1,
    )

    assert score.score == 3
    assert score.failed_checks == ["classification_expected"]


def test_rubric_does_not_credit_ocr_text_as_model_comprehension():
    """Semantic anchors are checked BEFORE the appended OCR block.

    Our own transcript is concatenated into the markdown; scoring against the
    whole document would let a model that understood nothing still pass.
    """
    from tests.vision_eval import score_case, split_model_content
    from tests.vision_corpus import CORPUS_BY_KEY

    case = CORPUS_BY_KEY["invoice_table"]
    markdown = (
        "# invoice-table.png\n\n_Image type: table_\n\nUne image.\n\n"
        "## Extracted text (OCR)\n\nTOTAL HT 12 480 EUR 9 600 EUR"
    )
    assert "12 480" not in split_model_content(markdown)

    score = score_case(
        case,
        _vision.VisionOutcome(markdown=markdown, reason="ok", classification="table"),
        repeat=1,
        model_calls=1,
    )
    assert "semantic_anchors_present" in score.failed_checks


def test_rubric_scores_a_free_refusal_and_penalises_a_spent_call():
    from tests.vision_eval import MAX_SCORE, score_case
    from tests.vision_corpus import CORPUS_BY_KEY

    case = CORPUS_BY_KEY["brand_logo"]
    outcome = _vision.VisionOutcome(
        markdown=None, reason="vision-prefilter: image rejected before vision analysis"
    )

    free = score_case(case, outcome, repeat=1, model_calls=0)
    assert free.score == MAX_SCORE and free.passed

    spent = score_case(case, outcome, repeat=1, model_calls=1)
    assert spent.failed_checks == ["no_model_call_spent"]
    assert spent.score == MAX_SCORE - 1


def test_aggregate_reports_accuracy_variance_and_topics():
    from tests.vision_eval import CaseScore, aggregate, format_summary

    scores = [
        CaseScore(key="a", topic="finance", repeat=1, score=4, checks={}),
        CaseScore(key="a", topic="finance", repeat=2, score=2, checks={"x": False}),
        CaseScore(key="b", topic="noise", repeat=1, score=4, checks={}),
        CaseScore(key="b", topic="noise", repeat=2, score=4, checks={}),
    ]
    report = aggregate(scores)

    assert report["samples"] == 4
    assert report["accuracy"] == 0.75
    assert report["by_case"]["a"]["spread"] == 2, "non-determinism must be visible"
    assert report["by_case"]["a"]["pass_rate"] == 0.5
    assert report["by_case"]["b"]["spread"] == 0
    assert report["by_topic"]["finance"]["accuracy"] == 0.5
    assert report["by_topic"]["noise"]["accuracy"] == 1.0
    assert len(report["failures"]) == 1
    assert "accuracy 75%" in format_summary(report)


async def test_live_gate_loop_scores_the_paid_corpus_against_a_stub(
    corpus_dir, endpoint
):
    """Dry-run of the paid gate's own loop, with the network replaced.

    Proves the live driver's structure — corpus iteration, per-case call
    accounting, rubric, aggregation, threshold — before any money is spent.
    The only thing the paid job adds on top is the real endpoint.
    """
    from tests.vision_corpus import PAID_CASES
    from tests.vision_eval import aggregate, min_accuracy, score_case

    scores = []
    for case in PAID_CASES:
        path = materialise(case, corpus_dir)
        before = len(endpoint.calls)
        outcome = await _vision.aprocess_image(path)
        scores.append(
            score_case(
                case,
                outcome,
                repeat=1,
                model_calls=len(endpoint.calls) - before,
            )
        )

    report = aggregate(scores)
    assert report["samples"] == len(PAID_CASES)
    # The three noise cases must contribute zero calls; only ingest cases pay.
    ingest_cases = sum(1 for case in PAID_CASES if case.expect_ingest)
    assert report["model_calls"] == ingest_cases
    assert report["by_topic"]["noise"]["accuracy"] == 1.0
    assert report["accuracy"] >= min_accuracy()
