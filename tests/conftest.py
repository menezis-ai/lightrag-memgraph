"""
Shared fixtures for twindb-lightrag-memgraph tests.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a running Memgraph instance"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless MEMGRAPH_URI is set."""
    if os.environ.get("MEMGRAPH_URI"):
        return

    skip = pytest.mark.skip(reason="MEMGRAPH_URI not set, skipping integration test")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
