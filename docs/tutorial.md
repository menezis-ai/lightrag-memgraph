# LightRAG with Memgraph: ingest, query and delete

Run a complete storage lifecycle against a real Memgraph server. The default
example needs no model API key. It uses fixed extraction output, deterministic
SHA-256 vectors (384 dimensions) and a character tokenizer. These fixtures
exercise storage and retrieval; they do not demonstrate semantic embedding or
answer quality. No tokenizer/model download is needed in fixture mode.

## Install and start Memgraph

Prerequisites: Git, Python 3.10+ and Docker Compose. This example is qualified
with LightRAG 1.5.7, Neo4j driver 6.4.1 and Memgraph MAGE 3.14.0.

```bash
git clone https://github.com/menezis-ai/lightrag-memgraph.git
cd lightrag-memgraph
python -m venv .venv
source .venv/bin/activate
pip install -e . "lightrag-hku==1.5.7"
docker compose up -d memgraph
export MEMGRAPH_URI=bolt://localhost:7687
python examples/lifecycle.py
```

On Windows, activate the virtual environment with `.venv\Scripts\Activate.ps1`
and set `$env:MEMGRAPH_URI = "bolt://localhost:7687"` in PowerShell.

The script waits up to 30 attempts for Bolt. If connection fails, inspect
`docker compose logs memgraph` and verify the URI and credentials. If port 7687
is occupied, use an available local endpoint and set `MEMGRAPH_URI` accordingly.

The repository's Compose database has no persistent volume. Use a local,
disposable instance for this tutorial. The script allocates a random workspace;
it does not accept an existing application workspace as input.

## What the example verifies

The [standalone script](../examples/lifecycle.py) registers the four storage
backends, initializes the shared LightRAG state and pipeline, then:

1. Inserts `paris.txt` with `ainsert()` and checks its processed status and
   persisted KV, vector, graph and status records.
2. Calls `aquery()` in local context-only mode and checks that retrieved context
   contains Paris. It does not present a canned model answer as retrieval proof.
3. Calls `adelete_by_doc_id()` and checks that document data, chunk records,
   vectors, status and graph nodes are absent.
4. Drops only the demo's stores and indexes, closes the drivers and releases the
   temporary working directory, including on failure. The ownership registry
   keeps provenance records for emptied stores.

The successful run ends with:

```text
QUERY: retrieved context containing Paris
DELETE: document, chunks, vectors, status and graph verified empty
```

To stop this tutorial's database, run `docker compose down` from this checkout.

## Use a real LLM and embedding endpoint

The optional `--provider openai` mode uses the OpenAI-compatible
[chat-completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
HTTP contracts. A local server must implement both endpoints, return text
completions and produce embeddings as float arrays. Configure the models
available at your endpoint explicitly; no model name is selected for you.

```bash
pip install httpx
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-local-or-remote-key"
export DEMO_LLM_MODEL="your-chat-model"
export DEMO_EMBEDDING_MODEL="your-embedding-model"
export DEMO_EMBEDDING_DIM="384"
python examples/lifecycle.py --provider openai
```

Set the dimension to the actual output dimension of your embedding model. The
example does not request dimension reduction. Local deployments may need a
non-empty dummy API key if their endpoint ignores authentication. Omitting
`OPENAI_BASE_URL` selects `https://api.openai.com/v1`; this mode sends the sample
document/prompts to that endpoint and may incur inference charges. The default
fixture run never selects it. Keys are read from the environment and not logged.

The LLM must follow LightRAG's extraction/keyword prompts. Context-only querying
still requires model calls for extraction and keyword generation. Live mode uses
LightRAG's default tokenizer, which may download its vocabulary on first use.
This is a small HTTP example without retry/backoff or a provider-specific token
budget; configure those in your application's model integration as needed.

Only fixture inference has been executed in the documented qualification; no
remote model quality, availability or billing outcome is claimed. Keep using the
same embedding model/dimension within a persistent namespace. Changing models
requires a reviewed rebuild or a new namespace.

## Before deploying

Read the [compatibility guide](memgraph-compatibility.md) and
[integrity notes](data-integrity.md). Version 0.7 changes physical identities;
existing databases require the explicit [migration procedure](migration-0.7.0.md).
The [isolation qualification](storage-identity-qualification.md) covers issue #21.
Workspace scoping does not enforce authorization: retain separately controlled
databases/access credentials for independent tenants.
