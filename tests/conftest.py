"""
Shared fixtures for twindb-lightrag-memgraph tests.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a running Memgraph instance"
    )
    config.addinivalue_line(
        "markers",
        "no_default_auth: opt out of the default LIGHTRAG_API_KEY fixture "
        "(used by tests that exercise the boot-fail-without-auth path)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless MEMGRAPH_URI is set."""
    if os.environ.get("MEMGRAPH_URI"):
        return

    skip = pytest.mark.skip(reason="MEMGRAPH_URI not set, skipping integration test")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _default_auth_backend(request, monkeypatch):
    """Set ``TWIN_ALLOW_OPEN_ACCESS=1`` by default.

    Audit 2026-06-10 H1: ``create_app()`` and ``_mount_twin_subapp()``
    refuse to start without any auth backend configured. The vast
    majority of unit tests instantiate ``LightRAGServerSettings`` with
    ``api_key=None, jwt_secret=None`` so they can hit routes without a
    Bearer header — that's intentional, the route shape is what they
    assert. To keep those tests working, this fixture flips the
    process into the explicit "open access" posture (boot passes,
    ``require_auth`` returns ``anonymous-open-access``).

    Opt out with ``@pytest.mark.no_default_auth`` when the test
    explicitly exercises the boot-fail-without-auth path or a route
    that demands a real identity.
    """
    if request.node.get_closest_marker("no_default_auth"):
        return
    if (
        not os.environ.get("LIGHTRAG_API_KEY")
        and not os.environ.get("LIGHTRAG_JWT_SECRET")
        and not os.environ.get("TOKEN_SECRET")
        and not os.environ.get("TWIN_IDP_JWKS_URL")
    ):
        monkeypatch.setenv("TWIN_ALLOW_OPEN_ACCESS", "1")
