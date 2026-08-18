"""Live OCR/Vision acceptance gate — real RapidOCR, real PDFium, real model.

Runs only under ``RUN_VISION_LIVE=1`` so ordinary developer and
compatibility-matrix runs never spend external model calls.

What changed, and why
---------------------
This gate used to assert a handful of substrings over a corpus of two, one
sample per call. A model that answers correctly four times out of five flipped
CI red at random, and nothing measured that rate. It is now a **scored
evaluation** on the Knowledge-Bot methodology (see ``tests/vision_eval.py``):
a golden corpus with declared ground truth, a 0..4 rubric per case, N
repetitions to expose non-determinism, aggregation into global and per-topic
accuracy, and a JSON report so the number is comparable across runs.

The free suite ``tests/test_vision_offline.py`` already proves the pipeline
wiring and the rubric itself against a stubbed endpoint. What only this gate
can prove is the part that is genuinely external: that the real RapidOCR model
files load, that PDFium really rasterises, and that the live vision endpoint
still honours its contract.

Cost envelope
-------------
``TWIN_VISION_EVAL_REPEATS`` (default 2) × the corpus' paid ingest cases (3)
= 6 image calls, plus 3 procedure calls (blind, informed, comparator) = **9
model calls per run**. The three noise cases (logo, signature, blank scan) are
in the evaluation but must cost zero calls — the pre-filter is supposed to stop
them, and "no_model_call_spent" is a scored rubric check, not an afterthought.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.vision_corpus import PAID_CASES, materialise
from tests.vision_eval import (
    aggregate,
    format_summary,
    min_accuracy,
    repeats,
    score_case,
)
from tests.vision_live_fixture import PROCEDURE_TASK_IDS, write_visual_procedure_pdf
from twindb_lightrag_memgraph import _procedure, _procedure_store, _vision

pytestmark = pytest.mark.live_vision

EXPECTED_VISION_CONFIG = {
    "TWIN_VISION_BASE_URL": "https://openrouter.ai/api/v1",
    "TWIN_VISION_MODEL": "google/gemma-4-31b-it",
    "TWIN_VISION_TIMEOUT": "120",
}


def _compact(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


@pytest.fixture(autouse=True)
def _live_openrouter_config(monkeypatch, tmp_path):
    api_key = os.environ.get("TWIN_VISION_API_KEY", "").strip()
    if not api_key:
        pytest.fail(
            "RUN_VISION_LIVE=1 requires TWIN_VISION_API_KEY; "
            "the CI job maps it from the OPENROUTER_API_KEY repository secret"
        )
    for name, expected in EXPECTED_VISION_CONFIG.items():
        actual = os.environ.get(name, "").strip()
        if actual != expected:
            pytest.fail(
                f"{name} must be {expected!r} for the live gate, got {actual!r}"
            )

    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "20")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", "invalid,logo,signature")
    monkeypatch.setenv("TWIN_PROCEDURE", "on")
    monkeypatch.setenv(
        "TWIN_PROCEDURE_STORE_FILE", str(tmp_path / "procedure-bundles.json")
    )
    _vision.reset_caches()
    _procedure.reset_caches()
    yield
    _vision.reset_caches()
    _procedure.reset_caches()


@pytest.fixture
def live_artifacts(tmp_path) -> Path:
    configured = os.environ.get("TWIN_VISION_LIVE_ARTIFACTS_DIR", "").strip()
    path = Path(configured) if configured else tmp_path / "live-vision-artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


class _CallCounter:
    """Counts real endpoint calls so 'the pre-filter saved a call' is provable."""

    def __init__(self, monkeypatch):
        self.count = 0
        original = _vision.vision_chat_sync

        def counted(messages, **kwargs):
            self.count += 1
            # Forward kwargs: swallowing max_tokens here would silently
            # disable the procedure passes' completion cap under test.
            return original(messages, **kwargs)

        monkeypatch.setattr(_vision, "vision_chat_sync", counted)


async def test_corpus_evaluation_meets_accuracy_threshold(live_artifacts, monkeypatch):
    """Score the golden corpus against the real model and gate on accuracy.

    A single flaky sample no longer fails the build; a genuine degradation of
    the model, the prompt or the pre-filter does.
    """
    counter = _CallCounter(monkeypatch)
    sample_count = repeats()
    scores = []

    for repeat in range(1, sample_count + 1):
        for case in PAID_CASES:
            path = materialise(case, live_artifacts / f"run-{repeat}")
            before = counter.count
            outcome = await _vision.aprocess_image(path)
            score = score_case(
                case,
                outcome,
                repeat=repeat,
                model_calls=counter.count - before,
            )
            scores.append(score)
            if outcome.markdown:
                (live_artifacts / f"{case.key}-run{repeat}.md").write_text(
                    outcome.markdown, encoding="utf-8"
                )

    report = aggregate(scores)
    report["repeats"] = sample_count
    report["model"] = os.environ.get("TWIN_VISION_MODEL", "")
    (live_artifacts / "vision-eval-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + format_summary(report))

    threshold = min_accuracy()
    assert report["accuracy"] >= threshold, (
        f"vision evaluation accuracy {report['accuracy']:.0%} below the "
        f"{threshold:.0%} threshold\n{format_summary(report)}"
    )
    # Accuracy alone can hide one case that never works while the others carry
    # the average. A case failing every single repeat is a hard failure.
    always_failing = [
        key for key, entry in report["by_case"].items() if entry["pass_rate"] == 0.0
    ]
    assert (
        not always_failing
    ), f"cases failed on every repeat: {always_failing}\n{format_summary(report)}"


async def test_noise_never_reaches_the_paid_endpoint(live_artifacts, monkeypatch):
    """Real logo/signature/blank pixels must die at the free OCR pre-filter."""
    counter = _CallCounter(monkeypatch)
    free_cases = [case for case in PAID_CASES if case.free_refusal]
    assert free_cases, "corpus lost its noise cases"

    for case in free_cases:
        outcome = await _vision.aprocess_image(
            materialise(case, live_artifacts / "noise")
        )
        assert outcome.markdown is None, f"{case.key} was ingested: {outcome.reason}"
        assert case.expect_reason in outcome.reason, outcome.reason

    assert counter.count == 0, (
        f"{counter.count} paid call(s) spent on noise images — the OCR "
        "pre-filter no longer protects the vision budget"
    )


async def test_representative_procedure_runs_real_dual_vision_pipeline(
    live_artifacts,
):
    """Procedure profile: real render, blind + informed passes, comparator."""
    pdf = write_visual_procedure_pdf(
        live_artifacts / "representative-itg-procedure.pdf"
    )

    assert _vision.is_enabled() is True
    assert _procedure.is_enabled() is True
    outcome = await _procedure.aprocess_procedure(pdf, "live-vision-ci")

    assert outcome is not None
    assert outcome.state == "pending", outcome.reason
    assert outcome.bundle_id is not None
    bundle = _procedure_store.get_bundle(outcome.bundle_id)
    assert bundle is not None
    assert bundle["state"] == "pending"
    assert bundle["schematics_total"] == 1
    assert len(bundle["schematics"]) == 1

    schematic = bundle["schematics"][0]
    assert schematic["png_base64"]
    assert schematic["error"] is None
    assert schematic["blind"] is not None
    assert schematic["informed"] is not None
    divergence = schematic["divergence"]
    assert divergence is not None

    tasks_by_pass = {
        pass_name: {
            task["id"].strip().upper(): task for task in schematic[pass_name]["tasks"]
        }
        for pass_name in ("blind", "informed")
    }

    # --- Deterministic structure: the render, the schema and the box set.
    # These do not depend on how the model phrases anything, so they stay hard.
    for pass_name, tasks in tasks_by_pass.items():
        assert PROCEDURE_TASK_IDS <= tasks.keys(), (
            f"{pass_name} pass missed task boxes: "
            f"{sorted(PROCEDURE_TASK_IDS - tasks.keys())}"
        )

    # --- Field attribution: scored, not asserted one by one.
    #
    # The schematic text is legible — RapidOCR transcribes "Condition: If major
    # incident", "Linked procedure:" and "CONF" from the rendered page at the
    # production scale. What varies between runs is whether the model files that
    # text under the RIGHT task field. A single binary assert on one sample
    # therefore fails on model non-determinism rather than on a regression: that
    # is exactly what happened on 2026-07-25, where id/title/responsible/actors
    # were all correct and only `conditions` came back empty.
    #
    # Same discipline as the image corpus: score the probes, require a majority,
    # and print the full picture on failure. A total miss still fails, because
    # each fact must be recovered by at least one of the two passes.
    probes = []
    for pass_name, tasks in tasks_by_pass.items():
        qualification = tasks["T2.1"]
        probes.append(
            (
                f"{pass_name}.conditions~major-incident",
                "MAJORINCIDENT" in _compact(qualification["conditions"]),
            )
        )
        probes.append(
            (f"{pass_name}.links~CONF", "CONF" in _compact(qualification["links"]))
        )
    informed_qualification = tasks_by_pass["informed"]["T2.1"]
    probes.append(
        (
            "informed.responsible~incident-manager",
            "INCIDENTMANAGER" in _compact(informed_qualification["responsible"]),
        )
    )
    probes.append(
        (
            "informed.actors~L1-support",
            "L1SUPPORT" in _compact(informed_qualification["actors"]),
        )
    )

    recovered = {name for name, ok in probes if ok}
    detail = "\n".join(f"  {'OK ' if ok else 'MISS'} {name}" for name, ok in probes)

    # Each fact must survive in at least one pass — a fact no pass recovered is
    # a real extraction failure, not jitter.
    for fact, names in (
        (
            "the T2.1 condition",
            ("blind.conditions~major-incident", "informed.conditions~major-incident"),
        ),
        ("the T2.1 cross-procedure link", ("blind.links~CONF", "informed.links~CONF")),
    ):
        assert recovered & set(names), (
            f"neither vision pass recovered {fact}\n{detail}\n" f"tasks={tasks_by_pass}"
        )

    assert len(recovered) >= 5, (
        f"only {len(recovered)}/{len(probes)} attribution probes recovered "
        f"(threshold 5)\n{detail}"
    )

    # The comparator's coherence verdict tracks whether the two passes agreed,
    # so it is reported rather than asserted: a pass that missed one field
    # legitimately diverges without the pipeline being broken.
    print(
        f"\nprocedure attribution {len(recovered)}/{len(probes)}\n{detail}\n"
        f"comparator coherent={divergence['coherent']} "
        f"divergences={divergence['divergences']}"
    )

    descriptions = " ".join(
        schematic[pass_name]["description"] for pass_name in ("blind", "informed")
    ).lower()
    assert "incident" in descriptions

    report = {key: value for key, value in schematic.items() if key != "png_base64"}
    (live_artifacts / "representative-itg-procedure-result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
