"""Qualify 201 chunks with the real 1.4.9 scheduler and Memgraph persistence."""

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from lightrag.utils import EmbeddingFunc, priority_limit_async_func_call

from lightrag_memgraph import _pool
from lightrag_memgraph.kv_impl import MemgraphKVStorage
from lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


async def main(label):
    started = asyncio.Event()
    batch_sizes = []

    async def provider(texts):
        batch_sizes.append(len(texts))
        started.set()
        await asyncio.sleep(0.31 * len(texts))
        return np.ones((len(texts), 4))

    queued = priority_limit_async_func_call(8, llm_timeout=30)(provider)
    workspace = "review_" + uuid4().hex
    vectors = MemgraphVectorDBStorage(
        "chunks", {"embedding_batch_num": 10, "embedding_func_max_async": 8},
        EmbeddingFunc(embedding_dim=4, func=queued), workspace=workspace,
    )
    kv = MemgraphKVStorage("probe", {}, None, workspace=workspace)
    result = {"label": label, "lightrag": importlib.metadata.version("lightrag-hku"),
              "driver": importlib.metadata.version("neo4j"), "chunks": 201,
              "llm_timeout": 30, "execution_timeout": 60,
              "provider_seconds_per_text": 0.31, "write_concurrency": 1}
    task = None
    try:
        await vectors.initialize()
        await kv.initialize()
        before = time.perf_counter()
        task = asyncio.create_task(vectors.upsert({str(i): {"content": str(i)} for i in range(201)}))
        await asyncio.wait_for(started.wait(), 5)
        probe_before = time.perf_counter()
        try:
            await asyncio.wait_for(kv.upsert({"probe": {"content": "independent write"}}), 1)
            result["kv_blocked_during_embedding"] = False
        except asyncio.TimeoutError:
            result["kv_blocked_during_embedding"] = True
        result["kv_probe_seconds"] = time.perf_counter() - probe_before
        try:
            await asyncio.wait_for(task, 85)
            result["vector_upsert"] = "success"
        except Exception as exc:
            result["vector_upsert"] = "error"
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
        result["vector_elapsed_seconds"] = time.perf_counter() - before
        result["batch_sizes"] = batch_sizes
        await asyncio.wait_for(kv.upsert({"after": {"content": "slot released"}}), 2)
        async with _pool.get_read_session() as session:
            rows = await session.run(f"MATCH (n:`{vectors._label()}`) RETURN count(n) AS count")
            result["persisted_vectors"] = (await rows.single())["count"]
            await rows.consume()
        if label == "baseline":
            assert result["kv_blocked_during_embedding"]
            assert result["vector_upsert"] == "error" and result["error_type"] == "TimeoutError"
            assert result["persisted_vectors"] == 0 and batch_sizes == [201]
        else:
            assert not result["kv_blocked_during_embedding"]
            assert result["vector_upsert"] == "success" and result["persisted_vectors"] == 201
            assert batch_sizes == [10] * 20 + [1]
        Path(f"/evidence/large-upsert-{label}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await queued.shutdown()
        for store in (vectors, kv):
            await store.drop()
            async with _pool.get_session() as session:
                rows = await (await session.run("SHOW INDEX INFO")).data()
                for row in rows:
                    if row.get("label") == store._label():
                        prop = row.get("property")
                        prop = prop[0] if isinstance(prop, list) and len(prop) == 1 else prop
                        if isinstance(prop, str):
                            await (await session.run(f"DROP INDEX ON :`{store._label()}`(`{prop}`)")).consume()
        await _pool.close_driver()


parser = argparse.ArgumentParser()
parser.add_argument("label", choices=("baseline", "fixed"))
asyncio.run(main(parser.parse_args().label))
