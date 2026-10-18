"""
Shared fixtures for twindb-lightrag-memgraph tests.
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-scale",
        action="store_true",
        help="Run isolated 10K/100K scale qualification",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a running Memgraph instance"
    )
    config.addinivalue_line(
        "markers", "scale: opt-in large datasets and latency budgets"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless MEMGRAPH_URI is set."""
    if not config.getoption("--run-scale"):
        scale_skip = pytest.mark.skip(
            reason="Pass --run-scale to enable large dataset tests"
        )
        for item in items:
            if "scale" in item.keywords:
                item.add_marker(scale_skip)
    if os.environ.get("MEMGRAPH_URI"):
        return

    skip = pytest.mark.skip(reason="MEMGRAPH_URI not set, skipping integration test")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
