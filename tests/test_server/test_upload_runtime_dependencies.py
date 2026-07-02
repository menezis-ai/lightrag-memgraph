"""Runtime dependencies required by LightRAG's native upload route."""

from __future__ import annotations

import importlib.util


def test_lightrag_upload_extractors_are_installed():
    """The server extra must include LightRAG's lazy upload parsers.

    LightRAG imports these inside ``document_routes.py`` only when an upload is
    processed. Missing packages make ``POST /documents/upload`` return 200 and
    then mark every document of that type as failed during background
    extraction, so this needs to fail at install/CI time instead.
    """

    missing = [
        module
        for module in ("pypdf", "docx", "pptx", "openpyxl")
        if importlib.util.find_spec(module) is None
    ]
    assert missing == []
