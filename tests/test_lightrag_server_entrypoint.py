import sys
import types

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import lightrag_server


def test_explicit_entrypoint_registers_overlay_before_lightrag_main(monkeypatch):
    calls = []

    def fake_register(**kwargs):
        calls.append(("register", kwargs))

    fake_server = types.SimpleNamespace(
        main=lambda: calls.append(("lightrag_main", {}))
    )

    monkeypatch.setattr(twindb_lightrag_memgraph, "register", fake_register)
    monkeypatch.setitem(sys.modules, "lightrag.api.lightrag_server", fake_server)

    lightrag_server.main()

    assert calls == [
        (
            "register",
            {
                "replace_ui": True,
                "mount_server": True,
                "shim_native_routes": True,
            },
        ),
        ("lightrag_main", {}),
    ]
