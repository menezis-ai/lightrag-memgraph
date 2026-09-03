"""Skip a module whose inputs live outside the delivered tree.

``tests/`` is not only run here: ``EXPORT_PROCEDURE.md`` ships it to BNP as the
validation handoff, and that tree deliberately carries **no** ``scripts/``, no
``docs/`` and no ``.forgejo/``. A test module that reads one of those at import
time therefore raises during collection over there — and pytest answers a
collection error with ``Interrupted``, which means **no test runs at all**.
Measured on the reconstructed 1.2.0 export tree: 3407 tests collected, 7 errors,
zero executed.

So the guard has to run *before* the import that would fail, which is why it
skips at module level rather than decorating the tests::

    from tests._repo_only import require_repo_path

    require_repo_path("scripts")
    from scripts import coverage_floor  # noqa: E402

The skip is the honest outcome: these modules test repository tooling, not the
product. They have nothing to say about a delivered bundle, and saying it as a
skip keeps the rest of the suite runnable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def require_repo_path(relative: str, *, module_level: bool = True) -> Path:
    """Return ``REPO_ROOT / relative``, or skip if it is absent.

    ``module_level=False`` skips only the calling test. Use it when the missing
    input gates one assertion rather than the whole file — skipping 16 tests
    because one of them reads a JSON schema would throw away real coverage of
    the delivered product.
    """
    path = REPO_ROOT / relative
    if not path.exists():
        pytest.skip(
            f"{relative} is not part of this tree — it is excluded from the BNP "
            "export (EXPORT_PROCEDURE.md §Exclude).",
            allow_module_level=module_level,
        )
    return path
