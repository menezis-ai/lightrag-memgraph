from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import retrieval_tuning_probe as probe
from scripts import retrieval_tuning_summarize as summarize


def _row(
    *,
    mode: str,
    repeat: int,
    metrics: dict[str, float] | None,
    ranking: list[str] | None = None,
    qrels: dict[str, float] | None = None,
    qid: str = "q1",
) -> dict:
    return {
        "run_id": "run-1",
        "repeat_index": repeat,
        "question": {
            "id": qid,
            "axis": "TEST",
            "query": "query",
            "qrels": qrels,
        },
        "params": {
            "mode": mode,
            "top_k": 20,
            "chunk_top_k": 10,
            "enable_rerank": True,
        },
        "ok": True,
        "ranked_chunk_ids": ranking or [],
        "retrieval_metrics": metrics,
        "exact_redundancy": {"exact_duplicate_count": 0},
    }


def test_rank_metrics_use_gold_labels_and_penalize_duplicate_rank_slots():
    qrels = {"a": 3.0, "b": 2.0, "c": 1.0}

    metrics = probe._retrieval_metrics(["a", "a", "b"], qrels, cutoffs=(2, 5))

    assert metrics["mrr"] == 1.0
    assert metrics["recall@2"] == pytest.approx(1 / 3)
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["ndcg@5"] == pytest.approx(
        probe._dcg_at_k(["a", "a", "b"], qrels, 5)
        / probe._dcg_at_k(["a", "b", "c"], qrels, 5)
    )
    assert (
        metrics["ndcg@5"]
        < probe._retrieval_metrics(["a", "b"], qrels, cutoffs=(5,))["ndcg@5"]
    )


def test_rank_metrics_penalize_relevant_chunks_at_lower_ranks():
    qrels = {"a": 2.0, "b": 1.0}

    ideal = probe._retrieval_metrics(["a", "b"], qrels)
    delayed = probe._retrieval_metrics(["x", "b", "a"], qrels)

    assert delayed["recall@5"] == ideal["recall@5"] == 1.0
    assert delayed["mrr"] == 0.5
    assert delayed["ndcg@5"] < ideal["ndcg@5"]


def test_record_separates_retrieval_metrics_from_qualitative_rows():
    body = {
        "status": "success",
        "data": {"chunks": [{"chunk_id": "a"}, {"id": "a"}, {"id": "b"}]},
    }
    common = {
        "run_id": "run",
        "repeat_index": 1,
        "params": {"mode": "naive"},
        "payload": {"query": "q"},
        "started_at": "now",
        "duration_ms": 1,
        "http_status": 200,
        "response_body": body,
        "error": None,
    }

    gold = probe._record(
        question={"id": "q", "query": "q", "qrels": {"a": 2.0, "b": 1.0}},
        **common,
    )
    qualitative = probe._record(
        question={"id": "q", "query": "q", "qrels": None},
        **common,
    )

    assert gold["evaluation_mode"] == "gold_qrels"
    assert gold["ranked_chunk_ids"] == ["a", "b"]
    assert gold["retrieval_metrics"]["ndcg@5"] == pytest.approx(
        probe._dcg_at_k(["a", "a", "b"], {"a": 2.0, "b": 1.0}, 5)
        / probe._dcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 5)
    )
    assert gold["retrieval_metrics"]["ndcg@5"] < 1.0
    assert gold["exact_redundancy"]["exact_duplicate_count"] == 1
    assert qualitative["evaluation_mode"] == "qualitative"
    assert qualitative["retrieval_metrics"] is None


def test_backend_failure_is_not_scored_as_zero_recall():
    row = probe._record(
        run_id="run",
        repeat_index=1,
        question={"id": "q", "query": "q", "qrels": {"a": 1.0}},
        params={"mode": "naive"},
        payload={"query": "q"},
        started_at="now",
        duration_ms=1,
        http_status=200,
        response_body={"status": "failure", "data": {"chunks": []}},
        error=None,
    )

    assert row["ok"] is False
    assert row["retrieval_metrics"] is None


def test_question_loader_accepts_optional_qrels_and_rejects_invalid_gold(
    tmp_path: Path,
):
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        json.dumps({"id": "gold", "query": "q", "qrels": {"chunk": 2}})
        + "\n"
        + json.dumps({"id": "qual", "query": "q"})
        + "\n",
        encoding="utf-8",
    )

    questions = probe._load_questions(valid)

    assert questions[0]["qrels"] == {"chunk": 2.0}
    assert questions[1]["qrels"] is None

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps({"id": "bad", "query": "q", "qrels": {"chunk": 0}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="at least one relevant chunk"):
        probe._load_questions(invalid)


def test_repeat_dimension_expands_each_configuration():
    rows = probe._iter_matrix(
        [{"id": "q", "query": "q"}],
        modes=["mix", "naive"],
        top_k_values=[20],
        chunk_top_k_values=[10],
        rerank_values=[True],
        repeat=3,
    )

    assert len(rows) == 6
    assert [repeat for _, _, repeat in rows[:3]] == [1, 2, 3]


def test_stability_reports_set_and_order_changes():
    identical = summarize._stability_summary([["a", "b"], ["a", "b"]], 10)
    reordered = summarize._stability_summary([["a", "b"], ["b", "a"]], 10)

    assert identical["jaccard_mean"] == 1.0
    assert identical["rank_weighted_jaccard_mean"] == 1.0
    assert reordered["jaccard_mean"] == 1.0
    assert 0.0 < reordered["rank_weighted_jaccard_mean"] < 1.0

    single = summarize._stability_summary([["a"]], 10)
    assert single["pair_count"] == 0
    assert single["jaccard_mean"] is None


def test_graph_vs_naive_deltas_are_paired_by_repeat_and_configuration():
    qrels = {"a": 1.0}
    rows = [
        _row(
            mode="naive",
            repeat=1,
            metrics={"ndcg@10": 0.4, "recall@10": 0.5, "mrr": 0.5},
            qrels=qrels,
        ),
        _row(
            mode="mix",
            repeat=1,
            metrics={"ndcg@10": 0.7, "recall@10": 0.75, "mrr": 1.0},
            qrels=qrels,
        ),
        _row(
            mode="naive",
            repeat=2,
            metrics={"ndcg@10": 0.5, "recall@10": 0.5, "mrr": 0.5},
            qrels=qrels,
        ),
        _row(
            mode="mix",
            repeat=2,
            metrics={"ndcg@10": 0.6, "recall@10": 0.75, "mrr": 1.0},
            qrels=qrels,
        ),
    ]

    result = summarize._paired_deltas(rows)

    assert len(result) == 1
    ndcg_mean, ndcg_std, count = result[0]["metrics"]["ndcg@10"]
    assert ndcg_mean == pytest.approx(0.2)
    assert ndcg_std == pytest.approx(0.1)
    assert count == 2
    assert result[0]["metrics"]["mrr"] == pytest.approx((0.5, 0.0, 2))


def test_qualitative_summary_refuses_scientific_ranking(tmp_path: Path):
    row = _row(mode="mix", repeat=1, metrics=None, ranking=["a"], qrels=None)

    markdown = summarize._summarize(tmp_path / "results.jsonl", [row])

    assert "No question contains qrels" in markdown
    assert "Not a Scientific Ranking" in markdown
    assert "Best Candidates" not in markdown
    assert "answer_status`, and source counts are not relevance labels" in markdown


def test_gold_summary_reports_metrics_stability_and_paired_naive_delta(tmp_path: Path):
    qrels = {"a": 2.0, "b": 1.0}
    rows = [
        _row(
            mode="naive",
            repeat=1,
            metrics={"ndcg@10": 0.5, "recall@10": 0.5, "mrr": 0.5},
            ranking=["x", "a"],
            qrels=qrels,
        ),
        _row(
            mode="naive",
            repeat=2,
            metrics={"ndcg@10": 0.5, "recall@10": 0.5, "mrr": 0.5},
            ranking=["x", "a"],
            qrels=qrels,
        ),
        _row(
            mode="mix",
            repeat=1,
            metrics={"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0},
            ranking=["a", "b"],
            qrels=qrels,
        ),
        _row(
            mode="mix",
            repeat=2,
            metrics={"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0},
            ranking=["a", "b"],
            qrels=qrels,
        ),
    ]

    markdown = summarize._summarize(tmp_path / "gold.jsonl", rows)

    assert "Gold-Qrel Retrieval Results" in markdown
    assert "1.000 ± 0.000 (n=2)" in markdown
    assert "Paired Graph-vs-Naive Deltas" in markdown
    assert "0.500 ± 0.000 (n=2)" in markdown
    assert "set Jaccard@10" in markdown
    assert "rank-weighted Jaccard@10" in markdown
