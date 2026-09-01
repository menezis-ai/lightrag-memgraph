"""Runtime dependency boundary for the standalone intelligence extra."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_chat_completion_does_not_require_prometheus_client():
    """L3 correlation must work with only the intelligence runtime extra."""
    project_root = Path(__file__).parents[2]
    script = textwrap.dedent("""
        import asyncio
        import builtins
        from types import SimpleNamespace

        real_import = builtins.__import__

        def import_without_prometheus(name, *args, **kwargs):
            if name == "prometheus_client" or name.startswith("prometheus_client."):
                raise ModuleNotFoundError(
                    "prometheus_client is absent from the intelligence extra",
                    name=name,
                )
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_prometheus

        from twindb_lightrag_memgraph.intelligence.config import (
            LLMProfileKind,
            TwinRAGConfig,
        )
        from twindb_lightrag_memgraph.intelligence.llm import create_chat_completion

        class FakeCompletions:
            async def create(self, **request):
                assert request["model"] == "chat-model"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        class FakeClient:
            chat = SimpleNamespace(completions=FakeCompletions())

        async def main():
            response = await create_chat_completion(
                TwinRAGConfig(
                    llm_api_key="test-key",
                    llm_api_base="https://llm.invalid/v1",
                    llm_model="chat-model",
                ),
                LLMProfileKind.CHAT,
                client_factory=lambda **kwargs: FakeClient(),
                messages=[{"role": "user", "content": "hello"}],
            )
            assert response.choices[0].message.content == "ok"

        asyncio.run(main())
        """)
    env = os.environ.copy()
    source_root = str(project_root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, env.get("PYTHONPATH")) if item
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
