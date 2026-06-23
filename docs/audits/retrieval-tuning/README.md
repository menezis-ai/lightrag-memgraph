# Retrieval Tuning Probe — TR-RETQ

Date: 2026-06-14  
Scope: recette Alberto TR-RETQ-01/02/03 after the C3 migration to `LightRAG.aquery_llm()`.

## Goal

This folder makes retrieval tuning reproducible before changing defaults. It does not tune the runtime by itself.

The probe calls the truthful Twin query contract:

- `POST /twin/api/query`
- no `tag_filter` on `/query` (C1: LightRAG 1.4.x ignores it)
- sources projected from `aquery_llm().data.references` (C3: no second vector retrieval)
- `answer_status` captured from the backend contract

## Run

Start a real Twin KMS server, then run:

```bash
python3 scripts/retrieval_tuning_probe.py \
  --base-url http://localhost:9621 \
  --questions docs/audits/retrieval-tuning/questions-tr-retq.jsonl \
  --out docs/audits/retrieval-tuning/results-local.jsonl
```

Useful narrower smoke:

```bash
python3 scripts/retrieval_tuning_probe.py \
  --dry-run \
  --modes mix,hybrid \
  --top-k 20 \
  --chunk-top-k 10 \
  --rerank true
```

If the deployment requires auth or folder binding:

```bash
TWIN_API_TOKEN=... python3 scripts/retrieval_tuning_probe.py \
  --base-url http://localhost:9621 \
  --folder default
```

## Default Matrix

The script defaults to a bounded matrix:

- modes: `mix,hybrid,global,local,naive`
- `top_k`: `20,40`
- `chunk_top_k`: `10,20`
- rerank: `true,false`

That is 40 calls per question. Use `--max-calls` or narrower CLI values when running against a slow or costly LLM backend.

## Reading Results

Each JSONL output row contains:

- question metadata (`id`, `axis`, `expected_signal`)
- query parameters (`mode`, `top_k`, `chunk_top_k`, `enable_rerank`)
- HTTP status and duration
- `answer_status`
- response text
- full source list returned by Twin
- derived metrics:
  - `source_count`
  - `unique_source_count`
  - `duplicate_source_names`
  - `chunk_count`
  - `unique_chunk_count`

Generate a Markdown summary from any probe output:

```bash
python3 scripts/retrieval_tuning_summarize.py \
  docs/audits/retrieval-tuning/results-local.jsonl \
  --out docs/audits/retrieval-tuning/summary-local.md
```

The summary flags HTTP errors, insufficient answers on non-control questions,
grounded answers on the negative control, duplicate source concentration, and
grounded answers without sources.

Interpretation by recipe axis:

- TR-RETQ-01, redundancy: compare `source_count` vs `unique_source_count`, then inspect whether repeated chunks from the same document add evidence or just duplicate prose.
- TR-RETQ-02, source relevance: inspect source names and response attribution. Scores are currently a zero baseline on the C3 path unless LightRAG exposes richer reference metrics later.
- TR-RETQ-03, conceptual recall: compare `answer_status` and source coverage across modes. `global`, `hybrid`, and `mix` should be watched closely because the failure is thematic/multi-hop rather than purely lexical.
- CONTROL: the negative question should return `insufficient_information` and `sources: []`. A grounded answer here is a regression candidate.

## Decision Rule

Do not change production defaults from one anecdotal run. A tuning proposal should include:

1. The JSONL result file.
2. A short table of winners per question axis.
3. A failure list for settings that produced irrelevant sources or false grounded answers.
4. A clear recommendation, for example `mode=mix`, `top_k=40`, `chunk_top_k=20`, `enable_rerank=true`, or no change.

If the matrix shows that no setting solves TR-RETQ-03, the next technical track is not a frontend change; it is retrieval strategy: graph multi-hop, query expansion, or explicit use of document titles/summaries in the indexed context.
